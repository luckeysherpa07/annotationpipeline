"""RGB modality pipeline for QA annotation.

This module handles RGB video annotation using the Gemini API.
It processes night and day frames to generate captions, questions, and answers.
"""
import asyncio
import copy
import json
import re
from typing import Any, Dict, List
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from prompts.rgb_prompts import RGB_PROMPTS
from annotation_feature.demo_result import DEMO_RESULT

try:
    from google.genai import types
except ImportError:
    types = None


def build_rgb_mega_prompt(annotation_types: list[str], night_frames: list[Path], day_frames: list[Path]) -> str:
    """Build mega prompt for RGB QA generation.
    
    Args:
        annotation_types: List of annotation types to process
        night_frames: List of night frame paths
        day_frames: List of day frame paths
        
    Returns:
        Formatted prompt string for the Gemini API
    """
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
    """Parse JSON response from Gemini API.
    
    Args:
        text: Response text from the API
        
    Returns:
        Parsed JSON as a dictionary
        
    Raises:
        ValueError: If JSON cannot be parsed
    """
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
    """Normalize RGB annotation results to ensure consistency.
    
    Args:
        raw_results: Raw results from the API
        
    Returns:
        Normalized results dictionary with all annotation types
    """
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
    """Call Gemini API with retry logic.
    
    Args:
        client: Gemini client instance
        contents: Content to send to the API
        max_retries: Maximum number of retries
        
    Returns:
        API response text
        
    Raises:
        Exception: If all retries fail
    """
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
    """Process a single RGB video pair.
    
    Args:
        client: Gemini client instance
        pair_key: Identifier for the video pair
        night_frames: List of night frame paths
        day_frames: List of day frame paths
        skip_api: If True, skip API calls and use demo results
        
    Returns:
        Annotation results dictionary
    """
    if skip_api:
        return copy.deepcopy(DEMO_RESULT)

    if not night_frames or not day_frames:
        print(f"    WARNING: Missing night or day frames for pair {pair_key}; falling back to demo results")
        return copy.deepcopy(DEMO_RESULT)

    selected_night = night_frames[:6]
    selected_day = day_frames[:6]

    from ...utils import encode_frames_to_base64, build_image_parts
    night_encoded = encode_frames_to_base64(selected_night)
    day_encoded = encode_frames_to_base64(selected_day)

    if not night_encoded or not day_encoded:
        print(f"    WARNING: Could not encode frames for pair {pair_key}; falling back to demo results")
        return copy.deepcopy(DEMO_RESULT)

    image_parts = build_image_parts(night_encoded) + build_image_parts(day_encoded)
    prompt = build_rgb_mega_prompt(list(RGB_PROMPTS.keys()), selected_night, selected_day)
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
    """Run RGB annotation pipeline in parallel.
    
    Args:
        client: Gemini client instance
        paired_frames: Dictionary of video pairs and their frames
        max_concurrent: Maximum concurrent API calls
        delay_between_pairs: Delay between pair processing in seconds
        skip_api: If True, skip API calls and use demo results
        
    Returns:
        Dictionary of annotation results keyed by pair_key
    """
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
