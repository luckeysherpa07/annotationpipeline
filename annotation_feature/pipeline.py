from pathlib import Path
import os
import json
import base64
from prompts.rgb_prompts import RGB_PROMPTS
from annotation_feature.demo_result import DEMO_RESULT
from annotation_feature.video_preprocessor import preprocess_videos

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"


def load_environment() -> None:
    """
    Load local environment variables from the project .env file when available.
    """
    if load_dotenv is not None:
        load_dotenv(dotenv_path=ENV_FILE, override=True)


def create_openai_client():
    """
    Build an OpenAI client after confirming the SDK and API key are available.
    """
    load_environment()

    if OpenAI is None:
        raise ImportError(
            "The OpenAI SDK is not installed. Install dependencies from requirements.txt first."
        )

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            f"Missing OPENAI_API_KEY. Set it in your environment or add it to {ENV_FILE}."
        )

    return OpenAI()


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


def get_caption_from_openai(client, frame_paths: list, caption_prompt: str) -> str:
    """
    Call OpenAI API to generate a caption for video frames.
    
    Args:
        client: OpenAI client instance
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
    
    image_content = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": encoded_frame,
            },
        }
        for encoded_frame in encoded_frames
    ]
    
    # Add the text prompt
    image_content.append({
        "type": "text",
        "text": caption_prompt,
    })
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": image_content,
            }
        ],
        max_tokens=1000,
    )
    
    return response.choices[0].message.content


def get_question_from_openai(client, caption: str, question_prompt: str) -> str:
    """
    Call OpenAI API to generate a question from a caption.
    
    Args:
        client: OpenAI client instance
        caption: The caption text to generate a question from
        question_prompt: The prompt to send to the API
        
    Returns:
        The question text from the API response
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": f"{caption}\n\n{question_prompt}",
            }
        ],
        max_tokens=500,
    )
    
    return response.choices[0].message.content


def get_answer_from_openai(client, frame_paths: list, question: str, answering_prompt: str) -> str:
    """
    Call OpenAI API to generate an answer for a question based on video frames.
    
    Args:
        client: OpenAI client instance
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
    
    image_content = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": encoded_frame,
            },
        }
        for encoded_frame in encoded_frames
    ]
    
    # Add the question and answering prompt
    image_content.append({
        "type": "text",
        "text": f"Question: {question}\n\n{answering_prompt}",
    })
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": image_content,
            }
        ],
        max_tokens=1000,
    )
    
    return response.choices[0].message.content


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
        skip_api: If True, skip OpenAI API calls and use DEMO_RESULT instead
        dataset_folder: Dataset directory containing the source videos
    """
    if test_mode:
        print("=" * 50)
        print("TEST MODE: Processing only one video pair")
        print("=" * 50)
        if skip_api:
            print("API calls disabled - using DEMO_RESULT data\n")
    
    client = None
    if not skip_api:
        client = create_openai_client()

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
        pairs_to_process = list(paired_frames.items())[test_pair_index:test_pair_index+1]
        print(f"Processing pair {test_pair_index} of {len(paired_frames)}:")
    else:
        pairs_to_process = list(paired_frames.items())

    # Process selected video pairs
    for pair_key, frames in pairs_to_process:
        night_frames = frames["night"]
        day_frames = frames["day"]

        if not night_frames and not day_frames:
            print(f"Skipping {pair_key} - no frames found")
            continue

        try:
            print(f"\nProcessing pair: {pair_key}")
            print(f"  Night frames: {len(night_frames) if night_frames else 0}")
            print(f"  Day frames: {len(day_frames) if day_frames else 0}")

            file_results = {}

            # Process each annotation type from RGB_PROMPTS
            for annotation_type in RGB_PROMPTS.keys():
                print(f"  Processing annotation type: {annotation_type}")
                
                caption_prompt = RGB_PROMPTS[annotation_type]["caption_prompt"]
                question_prompt = RGB_PROMPTS[annotation_type]["question_prompt"]
                answering_prompt = RGB_PROMPTS[annotation_type]["answering_prompt"]

                caption = None
                question = None
                answer = None

                # Step 1: Get caption from night frames
                if night_frames:
                    if skip_api:
                        caption = DEMO_RESULT[annotation_type]["caption"]
                        print(f"    Caption (DEMO): {caption[:50]}...")
                    else:
                        caption = get_caption_from_openai(client, night_frames, caption_prompt)
                        print(f"    Caption (API): {caption[:50]}...")
                
                # Step 2: Get question from caption
                if caption:
                    if skip_api:
                        question = DEMO_RESULT[annotation_type]["question"]
                        print(f"    Question (DEMO): {question[:50]}...")
                    else:
                        question = get_question_from_openai(client, caption, question_prompt)
                        print(f"    Question (API): {question[:50]}...")
                
                # Step 3: Get answer from day frames and question
                if day_frames and question:
                    if skip_api:
                        answer = DEMO_RESULT[annotation_type]["answer"]
                        print(f"    Answer (DEMO): {answer[:50]}...")
                    else:
                        answer = get_answer_from_openai(client, day_frames, question, answering_prompt)
                        print(f"    Answer (API): {answer[:50]}...")

                file_results[annotation_type] = {
                    "caption": caption,
                    "question": question,
                    "answer": answer
                }

            # Get the original video file paths for reference (if they exist)
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

        except Exception as e:
            results[pair_key] = f"ERROR: {e}"
            print(f"✗ Failed: {pair_key} -> {e}")

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
