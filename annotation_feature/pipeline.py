from pathlib import Path
import asyncio
import copy
import json
import os
import re
import sys
import base64
from typing import Any, Dict, List

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from prompts.rgb_prompts import RGB_PROMPTS
from annotation_feature.demo_result import DEMO_RESULT
from annotation_feature.video_preprocessor import preprocess_videos

video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".mpeg", ".mpg"}

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"


def load_environment() -> None:
    """
    Load local environment variables from the project .env file when available.
    """
    if load_dotenv is not None:
        load_dotenv(dotenv_path=ENV_FILE, override=True)


def create_gemini_client():
    """
    Build a Gemini client after confirming the SDK and API key are available.
    """
    load_environment()

    if genai is None:
        raise ImportError(
            "The Google GenAI SDK is not installed. Install dependencies from requirements.txt first."
        )

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            f"Missing GEMINI_API_KEY. Set it in your environment or add it to {ENV_FILE}."
        )

    return genai.Client()


def get_pair_key(file: Path) -> str:
    """
    Build a shared key for matching day/night RGB videos from the same scene.
    """
    stem = file.stem.lower()
    for token in ("_night", "_day", "night", "day"):
        stem = stem.replace(token, "")
    stem = stem.replace("__", "_").strip("_")
    return str(file.parent / stem)


def encode_frames_to_base64(frame_paths: list) -> list:
    """
    Encode image frames to base64 for API transmission.
    
    Args:
        frame_paths: List of Path objects to image files
        
    Returns:
        List of base64 encoded image strings
    """
    encoded_frames = []
    for frame_path in frame_paths:
        if not frame_path.exists():
            continue
        with open(frame_path, "rb") as f:
            encoded = base64.standard_b64encode(f.read()).decode("utf-8")
            encoded_frames.append(encoded)
    return encoded_frames


def build_image_parts(encoded_frames: list[str]) -> list:
    return [
        types.Part.from_bytes(data=base64.b64decode(encoded), mime_type="image/png")
        for encoded in encoded_frames
    ]


def build_mega_prompt(annotation_types: list[str], night_frames: list[Path], day_frames: list[Path]) -> str:
    prompt_parts = [
        "You are a video QA assistant. You will receive NIGHT frames and DAY frames as images.",
        "For each annotation type, follow these steps exactly:",
        "1. Generate a caption from NIGHT frames using the caption prompt.",
        "2. Generate a question from the caption using the question prompt.",
        "3. Generate an answer from DAY frames and the question using the answering prompt.",
        "Return ONLY valid JSON with the following structure:",
        "{",
    ]

    for index, annotation_type in enumerate(annotation_types):
        line = f'  "{annotation_type}": {{"caption": "...", "question": "...", "answer": "..."}}'
        if index < len(annotation_types) - 1:
            line += ","
        prompt_parts.append(line)

    prompt_parts.extend([
        "}",
        "Do not include any markdown, explanation, or additional text. Output must be parseable JSON only.",
        f"NIGHT frames ({len(night_frames)} images): {', '.join([path.name for path in night_frames])}",
        f"DAY frames ({len(day_frames)} images): {', '.join([path.name for path in day_frames])}",
        "",
        "Use the following prompts for each annotation type:",
    ])

    for annotation_type in annotation_types:
        prompt_parts.extend([
            f"### {annotation_type}",
            "CAPTION PROMPT:",
            RGB_PROMPTS[annotation_type]["caption_prompt"],
            "",
            "QUESTION PROMPT:",
            RGB_PROMPTS[annotation_type]["question_prompt"],
            "",
            "ANSWERING PROMPT:",
            RGB_PROMPTS[annotation_type]["answering_prompt"],
            "",
        ])

    prompt_parts.append(
        "Produce exactly one JSON object with all annotation types and no additional commentary."
    )
    return "\n".join(prompt_parts)


def parse_json_response(text: str) -> dict:
    if not text:
        raise ValueError("Empty response text")

    cleaned_text = text.strip()
    cleaned_text = re.sub(r"^```(?:json)?\\s*", "", cleaned_text, flags=re.I)
    cleaned_text = re.sub(r"\\s*```$", "", cleaned_text, flags=re.I)

    match = re.search(r"\{.*\}", cleaned_text, flags=re.S)
    if not match:
        raise ValueError("No JSON object found in response")

    json_text = match.group(0)
    return json.loads(json_text)


def normalize_annotation_results(raw_results: Any) -> dict:
    normalized: dict = {}
    for annotation_type in RGB_PROMPTS.keys():
        fallback = DEMO_RESULT.get(annotation_type, {})
        item = raw_results.get(annotation_type) if isinstance(raw_results, dict) else None

        if not isinstance(item, dict):
            normalized[annotation_type] = copy.deepcopy(fallback)
            continue

        caption = item.get("caption")
        question = item.get("question")
        answer = item.get("answer")

        if not all(isinstance(value, str) for value in (caption, question, answer)):
            normalized[annotation_type] = copy.deepcopy(fallback)
            continue

        normalized[annotation_type] = {
            "caption": caption,
            "question": question,
            "answer": answer,
        }

    return normalized


async def call_gemini_with_retry(client, contents: list, max_retries: int = 3) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model="gemini-3-flash-preview",
                contents=contents,
            )
            return response.text
        except Exception as e:
            if attempt == max_retries:
                raise
            await asyncio.sleep(2)


async def process_single_pair_batch(
    client,
    pair_key: str,
    night_frames: list[Path],
    day_frames: list[Path],
    skip_api: bool = False,
) -> dict:
    if skip_api:
        return copy.deepcopy(DEMO_RESULT)

    if not night_frames or not day_frames:
        print(f"    WARNING: Missing night or day frames for pair {pair_key}; falling back to demo results")
        return copy.deepcopy(DEMO_RESULT)

    selected_night = night_frames[:6]
    selected_day = day_frames[:6]

    night_encoded = encode_frames_to_base64(selected_night)
    day_encoded = encode_frames_to_base64(selected_day)

    if not night_encoded or not day_encoded:
        print(f"    WARNING: Could not encode frames for pair {pair_key}; falling back to demo results")
        return copy.deepcopy(DEMO_RESULT)

    image_parts = build_image_parts(night_encoded) + build_image_parts(day_encoded)
    prompt = build_mega_prompt(list(RGB_PROMPTS.keys()), selected_night, selected_day)
    contents = image_parts + [prompt]

    try:
        response_text = await call_gemini_with_retry(client, contents, max_retries=3)
        parsed = parse_json_response(response_text)
        return normalize_annotation_results(parsed)
    except Exception as e:
        print(f"    ERROR: Gemini batch call failed for {pair_key}: {e}")
        print(f"    Falling back to DEMO_RESULT for pair {pair_key}")
        return copy.deepcopy(DEMO_RESULT)


async def run_parallel_pipeline(
    client,
    paired_frames: Dict[str, Dict[str, list]],
    max_concurrent: int = 3,
    delay_between_pairs: int = 4,
    skip_api: bool = False,
) -> Dict[str, dict]:
    semaphore = asyncio.Semaphore(max_concurrent)
    results: Dict[str, dict] = {}

    async def worker(pair_key: str, frames: Dict[str, list]) -> tuple[str, dict]:
        async with semaphore:
            print(f"\nProcessing batch pair: {pair_key}")
            return pair_key, await process_single_pair_batch(
                client,
                pair_key,
                frames.get("night", []) or [],
                frames.get("day", []) or [],
                skip_api=skip_api,
            )

    tasks = []
    for pair_key, frames in paired_frames.items():
        tasks.append(asyncio.create_task(worker(pair_key, frames)))
        await asyncio.sleep(delay_between_pairs)

    for completed_task in asyncio.as_completed(tasks):
        pair_key, annotation_results = await completed_task
        results[pair_key] = annotation_results

    return results


def get_caption_from_gemini(client, frame_paths: list, caption_prompt: str) -> str:
    """
    Call Gemini API to generate a caption for video frames.
    
    Args:
        client: Gemini client instance
        frame_paths: List of Path objects to frame images
        caption_prompt: The prompt to send to the API
        
    Returns:
        The caption text from the API response
    """
    if not frame_paths:
        raise ValueError("No frames provided for captioning")
    
    # Build image content blocks (using first 10 frames to avoid context limits)
    frames_to_use = frame_paths[:10]
    encoded_frames = encode_frames_to_base64(frames_to_use)
    
    image_parts = [
        types.Part.from_bytes(data=base64.b64decode(encoded), mime_type="image/png")
        for encoded in encoded_frames
    ]
    
    contents = image_parts + [caption_prompt]
    
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=contents,
    )
    
    return response.text


def get_question_from_gemini(client, caption: str, question_prompt: str) -> str:
    """
    Call Gemini API to generate a question from a caption.
    
    Args:
        client: Gemini client instance
        caption: The caption text to generate a question from
        question_prompt: The prompt to send to the API
        
    Returns:
        The question text from the API response
    """
    contents = [f"{caption}\n\n{question_prompt}"]
    
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=contents,
    )
    
    return response.text


def get_answer_from_gemini(client, frame_paths: list, question: str, answering_prompt: str) -> str:
    """
    Call Gemini API to generate an answer for a question based on video frames.
    
    Args:
        client: Gemini client instance
        frame_paths: List of Path objects to frame images
        question: The question to answer
        answering_prompt: The prompt to send to the API
        
    Returns:
        The answer text from the API response
    """
    if not frame_paths:
        raise ValueError("No frames provided for answering")
    
    # Build image content blocks (using first 10 frames to avoid context limits)
    frames_to_use = frame_paths[:10]
    encoded_frames = encode_frames_to_base64(frames_to_use)
    
    image_parts = [
        types.Part.from_bytes(data=base64.b64decode(encoded), mime_type="image/png")
        for encoded in encoded_frames
    ]
    
    contents = image_parts + [f"Question: {question}\n\n{answering_prompt}"]
    
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=contents,
    )
    
    return response.text


def run(
    test_mode: bool = False,
    test_pair_index: int = 0,
    skip_api: bool = False,
    dataset_folder: Path | str = "dataset",
):
    """
    Run the annotation pipeline.

    Args:
        test_mode: If True, only process one video pair for testing
        test_pair_index: Which video pair to process in test mode (0 = first)
        skip_api: If True, skip Gemini API calls and use DEMO_RESULT instead
        dataset_folder: Dataset directory containing the source videos
    """
    if test_mode:
        print("=" * 50)
        print("TEST MODE: Processing only one video pair")
        print("=" * 50)
        if skip_api:
            print("Gemini API calls disabled - using DEMO_RESULT data\n")

    client = None
    if not skip_api:
        client = create_gemini_client()

    dataset_folder = Path(dataset_folder)
    results = {}

    if not dataset_folder.exists():
        print("ERROR: Dataset folder not found!")
        print(f"Expected to find videos in: {dataset_folder}")
        return results

    print(f"Dataset directory listing for {dataset_folder}:")
    print(os.listdir(dataset_folder))

    # Preprocess all videos and extract frames
    print("Preprocessing videos...")
    paired_frames = preprocess_videos(dataset_folder, fps=1)
    print(f"Found {len(paired_frames)} video pairs\n")

    if len(paired_frames) == 0:
        print("ERROR: No video pairs found in dataset folder!")
        print(f"Expected to find videos in: {dataset_folder}")
        return results

    # In test mode, only process one pair
    if test_mode:
        pairs_to_process = list(paired_frames.items())[test_pair_index:test_pair_index + 1]
        print(f"Processing pair {test_pair_index} of {len(paired_frames)}:")
    else:
        pairs_to_process = list(paired_frames.items())

    available_pairs = {
        pair_key: frames
        for pair_key, frames in pairs_to_process
        if frames.get("night") or frames.get("day")
    }

    if not available_pairs:
        print("ERROR: No usable video frames found for selected pairs.")
        return results

    print(
        f"Processing {len(available_pairs)} batch pairs with up to 3 concurrent tasks and 4-second spacing..."
    )

    batch_results = asyncio.run(
        run_parallel_pipeline(
            client,
            available_pairs,
            max_concurrent=3,
            delay_between_pairs=4,
            skip_api=skip_api,
        )
    )

    for pair_key, frames in pairs_to_process:
        night_frames = frames.get("night") or []
        day_frames = frames.get("day") or []

        if not night_frames and not day_frames:
            print(f"Skipping {pair_key} - no frames found")
            continue

        file_results = batch_results.get(pair_key)
        if file_results is None:
            print(f"WARNING: No batch output for pair {pair_key}. Falling back to DEMO_RESULT.")
            file_results = copy.deepcopy(DEMO_RESULT)

        night_file = None
        day_file = None
        for file in dataset_folder.rglob("*"):
            if not file.is_file() or file.suffix.lower() not in video_extensions:
                continue
            name = file.name.lower()
            if "rgb" not in name:
                continue
            if get_pair_key(file) == pair_key:
                if "night" in name:
                    night_file = file
                elif "day" in name:
                    day_file = file

        results[pair_key] = {
            "night_file": str(night_file) if night_file else None,
            "day_file": str(day_file) if day_file else None,
            "annotations": file_results,
        }
        print(f"✓ Done: {pair_key}")

    # Save results to JSON file at the project root
    output_file = Path("qa_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n" + "=" * 50)
    print(f"Results saved to: {output_file}")
    if test_mode:
        print("TEST MODE COMPLETE")
    print("=" * 50)
    return results


if __name__ == "__main__":
    run()
