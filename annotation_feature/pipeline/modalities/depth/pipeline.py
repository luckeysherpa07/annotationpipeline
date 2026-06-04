"""Depth modality pipeline for QA annotation.

This module handles depth-based video annotation using the Gemini API.
It processes day and night depth frames to generate depth-based captions, questions, and answers.
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

from prompts.depth_prompts import DEPTH_PROMPTS
from annotation_feature.demo_result import DEMO_RESULT

DEPTH_MODEL_NAME = "gemini-3.1-flash-lite"
DEPTH_SKIPPED_MISSING_SIDE_STATUS = "skipped_missing_side"

try:
    from google.genai import types
except ImportError:
    types = None


def build_depth_mega_prompt(annotation_types: list[str], day_frames: list[Path], night_frames: list[Path]) -> str:
    """Build mega prompt for depth-based QA generation.
    
    Args:
        annotation_types: List of annotation types to process
        day_frames: List of day depth frame paths
        night_frames: List of night depth frame paths
        
    Returns:
        Formatted prompt string for the Gemini API
    """
    prompt_parts = [
        "You are a video QA assistant specialized in depth-based visual understanding.",
        "You will receive DAY frames and NIGHT frames as depth images.",
        "For each annotation type, follow these steps exactly:",
        "1. Generate a caption based on the depth stream analysis.",
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
        "Every requested annotation type must contain non-empty caption, question, and answer strings.",
        "Generate exactly one clean question and one clean answer per requested annotation type.",
        "Do not copy, quote, paraphrase, or include any prompt instructions in the caption, question, or answer.",
        "Do not generate numbered question lists or numbered answer lists.",
        "If a requested depth-based capability is absent, unclear, or not visible in the frames, still produce a negative QA that explicitly states the absence.",
        "Do not return empty strings, null values, placeholder text, or omit any requested annotation type.",
        f"DAY frames ({len(day_frames)} images): {', '.join([path.name for path in day_frames])}",
        f"NIGHT frames ({len(night_frames)} images): {', '.join([path.name for path in night_frames])}",
        "",
        "Use the following prompts for each annotation type:",
    ])

    for annotation_type in annotation_types:
        prompt_parts.extend([
            f"### {annotation_type}",
            "CAPTION PROMPT:",
            DEPTH_PROMPTS[annotation_type]["caption_prompt"],
            "",
            "QUESTION PROMPT:",
            DEPTH_PROMPTS[annotation_type]["question_prompt"],
            "",
            "ANSWERING PROMPT:",
            DEPTH_PROMPTS[annotation_type]["answering_prompt"],
            "",
        ])

    prompt_parts.append(
        "Produce exactly one JSON object with all requested annotation types and no additional commentary."
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


def normalize_depth_results(raw_results: Any, annotation_types: list[str] | None = None) -> dict:
    """Normalize depth annotation results to ensure consistency.
    
    Args:
        raw_results: Raw results from the API
        
    Returns:
        Normalized results dictionary with all annotation types
    """
    normalized: dict = {}
    expected_types = annotation_types or list(DEPTH_PROMPTS.keys())
    for annotation_type in expected_types:
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
                model=DEPTH_MODEL_NAME,
                contents=contents,
            )
            return response.text
        except Exception as e:
            if attempt == max_retries:
                raise
            await asyncio.sleep(2)


async def process_depth_pair_batch(
    client,
    pair_key: str,
    day_frames: list[Path],
    night_frames: list[Path],
    skip_api: bool = False,
    empty_on_failure: bool = True,
    mark_missing_side: bool = False,
) -> dict | None:
    """Process a single depth video pair.
    
    Args:
        client: Gemini client instance
        pair_key: Identifier for the video pair
        day_frames: List of day depth frame paths
        night_frames: List of night depth frame paths
        skip_api: If True, skip API calls and use demo results
        
    Returns:
        Annotation results dictionary
    """
    return await process_depth_pair_sections(
        client,
        pair_key,
        day_frames,
        night_frames,
        list(DEPTH_PROMPTS.keys()),
        skip_api=skip_api,
        empty_on_failure=empty_on_failure,
        mark_missing_side=mark_missing_side,
    )


async def process_depth_pair_sections(
    client,
    pair_key: str,
    day_frames: list[Path],
    night_frames: list[Path],
    annotation_types: list[str],
    skip_api: bool = False,
    empty_on_failure: bool = True,
    mark_missing_side: bool = False,
) -> dict | None:
    """Process selected depth annotation sections for a single video pair."""
    if skip_api:
        return {
            annotation_type: {
                "caption": "Demo caption",
                "question": "Demo question?",
                "answer": "Demo answer",
            }
            for annotation_type in annotation_types
        }

    if not day_frames or not night_frames:
        if mark_missing_side:
            print(f"    WARNING: Missing day or night frames for pair {pair_key}; marking as skipped")
            return {
                "status": DEPTH_SKIPPED_MISSING_SIDE_STATUS,
                "reason": "missing_day_or_night_frames",
                "day_depth_count": len(day_frames),
                "night_depth_count": len(night_frames),
            }
        print(f"    WARNING: Missing day or night frames for pair {pair_key}; falling back to empty results")
        return {anno_type: {"caption": "", "question": "", "answer": ""} for anno_type in annotation_types}

    if not annotation_types:
        return {}

    selected_day = day_frames
    selected_night = night_frames

    from ...utils import encode_frames_to_base64, build_image_parts
    day_encoded = encode_frames_to_base64(selected_day)
    night_encoded = encode_frames_to_base64(selected_night)

    if not day_encoded or not night_encoded:
        if not empty_on_failure:
            print(f"    WARNING: Could not encode frames for pair {pair_key}; keeping it pending for resume")
            return None
        print(f"    WARNING: Could not encode frames for pair {pair_key}; falling back to empty results")
        return {anno_type: {"caption": "", "question": "", "answer": ""} for anno_type in annotation_types}

    image_parts = build_image_parts(day_encoded) + build_image_parts(night_encoded)
    prompt = build_depth_mega_prompt(annotation_types, selected_day, selected_night)
    contents = image_parts + [prompt]

    try:
        response_text = await call_gemini_with_retry(client, contents, max_retries=3)
        parsed = parse_json_response(response_text)
        return normalize_depth_results(parsed, annotation_types)
    except Exception as e:
        print(f"    ERROR: Gemini batch call failed for {pair_key}: {e}")
        if not empty_on_failure:
            print(f"    Skipping checkpoint for pair {pair_key}; it will remain pending for resume.")
            return None
        print(f"    Falling back to empty results for pair {pair_key}")
        return {anno_type: {"caption": "", "question": "", "answer": ""} for anno_type in annotation_types}


async def run_depth_parallel_pipeline(
    client,
    paired_frames: Dict[str, Dict[str, list]],
    max_concurrent: int = 3,
    delay_between_pairs: int = 4,
    skip_api: bool = False,
    empty_on_failure: bool = True,
    mark_missing_side: bool = False,
    on_pair_complete=None,
) -> Dict[str, dict]:
    """Run depth annotation pipeline in parallel.
    
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

    async def run_one(pair_key: str, frames: Dict[str, list]) -> dict | None:
        print(f"\nProcessing depth pair: {pair_key}")
        return await process_depth_pair_batch(
            client,
            pair_key,
            frames.get("day", []) or [],
            frames.get("night", []) or [],
            skip_api=skip_api,
            empty_on_failure=empty_on_failure,
            mark_missing_side=mark_missing_side,
        )

    if max_concurrent <= 1:
        items = list(paired_frames.items())
        for index, (pair_key, frames) in enumerate(items):
            annotation_results = await run_one(pair_key, frames)
            if annotation_results is not None:
                results[pair_key] = annotation_results
                if on_pair_complete is not None:
                    on_pair_complete(pair_key, annotation_results)
            if index < len(items) - 1 and delay_between_pairs > 0:
                await asyncio.sleep(delay_between_pairs)
        return results

    async def worker(pair_key: str, frames: Dict[str, list]) -> tuple[str, dict | None]:
        async with semaphore:
            return pair_key, await run_one(pair_key, frames)

    tasks = []
    for pair_key, frames in paired_frames.items():
        tasks.append(asyncio.create_task(worker(pair_key, frames)))
        await asyncio.sleep(delay_between_pairs)

    for completed_task in asyncio.as_completed(tasks):
        pair_key, annotation_results = await completed_task
        if annotation_results is None:
            continue
        results[pair_key] = annotation_results
        if on_pair_complete is not None:
            on_pair_complete(pair_key, annotation_results)

    return results


async def run_depth_missing_sections_pipeline(
    client,
    repair_jobs: Dict[str, Dict[str, Any]],
    max_concurrent: int = 1,
    delay_between_pairs: int = 70,
    skip_api: bool = False,
    on_pair_complete=None,
) -> Dict[str, dict]:
    """Run depth repair for selected missing annotation sections."""
    results: Dict[str, dict] = {}
    items = list(repair_jobs.items())

    async def run_one(pair_key: str, job: Dict[str, Any]) -> dict | None:
        print(
            f"\nRepairing depth pair: {pair_key} "
            f"({len(job.get('missing_sections', []))} missing section(s))"
        )
        return await process_depth_pair_sections(
            client,
            pair_key,
            job.get("day", []) or [],
            job.get("night", []) or [],
            job.get("missing_sections", []) or [],
            skip_api=skip_api,
            empty_on_failure=False,
            mark_missing_side=True,
        )

    if max_concurrent <= 1:
        for index, (pair_key, job) in enumerate(items):
            annotation_results = await run_one(pair_key, job)
            if annotation_results is not None:
                results[pair_key] = annotation_results
                if on_pair_complete is not None:
                    on_pair_complete(pair_key, annotation_results)
            if index < len(items) - 1 and delay_between_pairs > 0:
                await asyncio.sleep(delay_between_pairs)
        return results

    semaphore = asyncio.Semaphore(max_concurrent)

    async def worker(pair_key: str, job: Dict[str, Any]) -> tuple[str, dict | None]:
        async with semaphore:
            return pair_key, await run_one(pair_key, job)

    tasks = []
    for pair_key, job in items:
        tasks.append(asyncio.create_task(worker(pair_key, job)))
        await asyncio.sleep(delay_between_pairs)

    for completed_task in asyncio.as_completed(tasks):
        pair_key, annotation_results = await completed_task
        if annotation_results is None:
            continue
        results[pair_key] = annotation_results
        if on_pair_complete is not None:
            on_pair_complete(pair_key, annotation_results)

    return results
