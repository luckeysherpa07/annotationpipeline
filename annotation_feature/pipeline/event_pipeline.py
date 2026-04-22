import asyncio
import copy
import json
import re
from typing import Any, Dict, List
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from prompts.event_prompts import EVENT_PROMPTS
from annotation_feature.demo_result import DEMO_RESULT

try:
    from google.genai import types
except ImportError:
    types = None


def build_event_mega_prompt(annotation_types: list[str], day_frames: list[Path], night_frames: list[Path]) -> str:
    """Build prompt for event-based QA generation (caption, question, and answer)"""
    prompt_parts = [
        "You are a video QA assistant specialized in event-based visual understanding.",
        "You will receive DAY frames and NIGHT frames as images.",
        "For each annotation type, follow these steps exactly:",
        "1. Generate a caption based on the event stream analysis.",
        "2. Generate a question from the caption using the question prompt.",
        "3. Generate an answer using the answering prompt.",
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
        f"DAY frames ({len(day_frames)} images): {', '.join([path.name for path in day_frames])}",
        f"NIGHT frames ({len(night_frames)} images): {', '.join([path.name for path in night_frames])}",
        "",
        "Use the following prompts for each annotation type:",
    ])

    for annotation_type in annotation_types:
        prompt_parts.extend([
            f"### {annotation_type}",
            "CAPTION PROMPT:",
            EVENT_PROMPTS[annotation_type]["caption_prompt"],
            "",
            "QUESTION PROMPT:",
            EVENT_PROMPTS[annotation_type]["question_prompt"],
            "",
            "ANSWERING PROMPT:",
            EVENT_PROMPTS[annotation_type]["answering_prompt"],
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


def normalize_event_results(raw_results: Any) -> dict:
    """Normalize event results to ensure all annotation types have caption, question, and answer"""
    normalized: dict = {}
    for annotation_type in EVENT_PROMPTS.keys():
        # Create fallback with empty values for event data
        fallback = {"caption": "", "question": "", "answer": ""}
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


async def process_event_pair_batch(
    client,
    pair_key: str,
    day_frames: list[Path],
    night_frames: list[Path],
    skip_api: bool = False,
) -> dict:
    """Process a single event pair and return captions, questions, and answers"""
    if skip_api:
        # Return demo results with all three fields
        demo_results = {}
        for annotation_type in EVENT_PROMPTS.keys():
            demo_results[annotation_type] = {
                "caption": "Demo caption",
                "question": "Demo question?",
                "answer": "Demo answer"
            }
        return demo_results

    if not day_frames or not night_frames:
        print(f"    WARNING: Missing day or night frames for pair {pair_key}; falling back to empty results")
        return {anno_type: {"caption": "", "question": "", "answer": ""} for anno_type in EVENT_PROMPTS.keys()}

    selected_day = day_frames[:6]
    selected_night = night_frames[:6]

    from .utils import encode_frames_to_base64, build_image_parts
    day_encoded = encode_frames_to_base64(selected_day)
    night_encoded = encode_frames_to_base64(selected_night)

    if not day_encoded or not night_encoded:
        print(f"    WARNING: Could not encode frames for pair {pair_key}; falling back to empty results")
        return {anno_type: {"caption": "", "question": "", "answer": ""} for anno_type in EVENT_PROMPTS.keys()}

    image_parts = build_image_parts(day_encoded) + build_image_parts(night_encoded)
    prompt = build_event_mega_prompt(list(EVENT_PROMPTS.keys()), selected_day, selected_night)
    contents = image_parts + [prompt]

    try:
        response_text = await call_gemini_with_retry(client, contents, max_retries=3)
        parsed = parse_json_response(response_text)
        return normalize_event_results(parsed)
    except Exception as e:
        print(f"    ERROR: Gemini batch call failed for {pair_key}: {e}")
        print(f"    Falling back to empty results for pair {pair_key}")
        return {anno_type: {"caption": "", "question": "", "answer": ""} for anno_type in EVENT_PROMPTS.keys()}


async def run_event_parallel_pipeline(
    client,
    paired_frames: Dict[str, Dict[str, list]],
    max_concurrent: int = 3,
    delay_between_pairs: int = 4,
    skip_api: bool = False,
) -> Dict[str, dict]:
    """Run event pipeline in parallel"""
    semaphore = asyncio.Semaphore(max_concurrent)
    results: Dict[str, dict] = {}

    async def worker(pair_key: str, frames: Dict[str, list]) -> tuple[str, dict]:
        async with semaphore:
            print(f"\nProcessing event pair: {pair_key}")
            return pair_key, await process_event_pair_batch(
                client,
                pair_key,
                frames.get("day", []) or [],
                frames.get("night", []) or [],
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
