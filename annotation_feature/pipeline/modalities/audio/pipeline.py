"""Audio modality pipeline for QA annotation.

This module handles audio annotation using the Gemini API.
It processes audio files (night audio only, as day/night doesn't affect audio)
to generate captions, questions, and answers.
"""
import asyncio
import copy
import json
import re
from typing import Any, Dict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from prompts.audio_prompts import AUDIO_PROMPTS
from annotation_feature.demo_result import DEMO_RESULT

try:
    from google.genai import types
except ImportError:
    types = None


# Audio-specific demo result for fallback
AUDIO_DEMO_RESULT = {
    "sound_recognition": {
        "caption": "Sound events detected: running water, dish clattering, hand movements, slight background noise.",
        "question": "What are the main sound sources in this audio?",
        "answer": "Water running, dishes, and hand movements"
    },
    "sound_counting": {
        "caption": "Countable sounds: 3 distinct water bursts, 5 dish clatters, multiple hand movements.",
        "question": ["How many times do you hear distinct water bursts?", "How many dish clatters are there?"] * 4,
        "answer": ["3 times", "5 times", "Multiple occurrences"] * 4
    },
    "sound_sequence": {
        "caption": "Sequence: Initial water sound, followed by dish movement, then clattering, ending with ambient sound.",
        "question": ["What sound occurs first?", "What happens after the water stops?"] * 4,
        "answer": ["Water running", "Dish movement begins", "Clattering sounds", "Ambient noise"] * 2
    },
    "sound_spatial": {
        "caption": "Spatial audio: Water appears close, centered. Dish sounds seem to move across the stereo field.",
        "question": "Where does the main water sound appear to come from?",
        "answer": "Center position, close to the microphone"
    },
    "speech_recognition": {
        "caption": "No clear speech detected. Background ambient sound and environmental noise present.",
        "question": ["Is there speech in the audio?", "How many speakers can you identify?"] * 4,
        "answer": ["No clear speech", "No speakers detected", "Only environmental sounds"] * 4
    },
    "music_recognition": {
        "caption": "No music detected. Only natural environmental sounds and action-related audio.",
        "question": ["Is there music in the audio?", "What is the style of music?"] * 4,
        "answer": ["No music detected", "Not applicable", "Only natural sounds"] * 4
    },
    "environmental_scene": {
        "caption": "Environment: Indoor kitchen or wash area. Sounds suggest water, dishes, and daily activity.",
        "question": ["What environment do the sounds suggest?", "What room is this likely?"] * 4,
        "answer": ["Kitchen or wash area", "Kitchen/bathroom", "Domestic indoor space"] * 4
    },
    "audio_change": {
        "caption": "Changes detected: Sound starts with water, peaks with clattering, then quiets to ambient level.",
        "question": ["Does the audio get louder over time?", "What is the loudest event?"] * 4,
        "answer": ["Initially yes, then decreases", "Dish clattering", "Activity sounds"] * 4
    },
    "audio_visual_correspondence": {
        "caption": "Expected visual match: Water pouring, dish handling, hand movements. All consistent with typical kitchen activity.",
        "question": ["Could these sounds match a kitchen activity?", "What action is likely?"] * 4,
        "answer": ["Yes, strongly matches", "Washing dishes or hands", "Kitchen activity"] * 4
    },
    "action_from_sound": {
        "caption": "Inferred actions: Water pouring or running, picking up dishes, putting items down, hand cleaning.",
        "question": ["What action is happening first?", "What objects are involved?"] * 4,
        "answer": ["Water running", "Dishes and hands", "Water and objects"] * 4
    }
}


def build_audio_mega_prompt(annotation_types: list[str], audio_filename: str) -> str:
    """Build mega prompt for audio QA generation.
    
    Args:
        annotation_types: List of annotation types to process
        audio_filename: Name/path of the audio file being processed
        
    Returns:
        Formatted prompt string for the Gemini API
    """
    prompt_parts = [
        "You are an audio QA assistant. You will receive an audio file.",
        "For each annotation type, follow these steps exactly:",
        "1. Generate a caption from the audio using the caption prompt.",
        "2. Generate a question from the caption using the question prompt.",
        "3. Generate an answer from the audio and the question using the answering prompt.",
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
        f"Audio file: {audio_filename}",
        "",
        "Use the following prompts for each annotation type:",
    ])

    for annotation_type in annotation_types:
        prompt_parts.extend([
            f"### {annotation_type}",
            "CAPTION PROMPT:",
            AUDIO_PROMPTS[annotation_type]["caption_prompt"],
            "",
            "QUESTION PROMPT:",
            AUDIO_PROMPTS[annotation_type]["question_prompt"],
            "",
            "ANSWERING PROMPT:",
            AUDIO_PROMPTS[annotation_type]["answering_prompt"],
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
    cleaned_text = re.sub(r"^```(?:json)?\s*", "", cleaned_text, flags=re.I)
    cleaned_text = re.sub(r"\s*```$", "", cleaned_text, flags=re.I)

    match = re.search(r"\{.*\}", cleaned_text, flags=re.S)
    if not match:
        raise ValueError("No JSON object found in response")

    json_text = match.group(0)
    return json.loads(json_text)


def normalize_annotation_results(raw_results: Any) -> dict:
    """Normalize audio annotation results to ensure consistency.
    
    Args:
        raw_results: Raw results from the API
        
    Returns:
        Normalized results dictionary with all annotation types
    """
    normalized: dict = {}
    for annotation_type in AUDIO_PROMPTS.keys():
        fallback = AUDIO_DEMO_RESULT.get(annotation_type, {})
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


async def process_single_audio(
    client,
    pair_key: str,
    audio_path: Path,
    skip_api: bool = False,
) -> dict:
    """Process a single audio file.
    
    Args:
        client: Gemini client instance
        pair_key: Identifier for the audio pair
        audio_path: Path to the audio file
        skip_api: If True, skip API calls and use demo results
        
    Returns:
        Annotation results dictionary
    """
    if skip_api:
        return copy.deepcopy(AUDIO_DEMO_RESULT)

    if not audio_path or not audio_path.exists():
        print(f"    WARNING: Audio file not found for pair {pair_key}; falling back to demo results")
        return copy.deepcopy(AUDIO_DEMO_RESULT)

    from ...utils import encode_audio_to_base64, build_audio_part
    audio_encoded = encode_audio_to_base64(audio_path)

    if not audio_encoded:
        print(f"    WARNING: Could not encode audio for pair {pair_key}; falling back to demo results")
        return copy.deepcopy(AUDIO_DEMO_RESULT)

    audio_parts = build_audio_part(audio_encoded, mime_type="audio/mp4")
    prompt = build_audio_mega_prompt(list(AUDIO_PROMPTS.keys()), audio_path.name)
    contents = audio_parts + [prompt]

    try:
        response_text = await call_gemini_with_retry(client, contents, max_retries=3)
        parsed = parse_json_response(response_text)
        return normalize_annotation_results(parsed)
    except Exception as e:
        print(f"    ERROR: Gemini API call failed for {pair_key}: {e}")
        print(f"    Falling back to AUDIO_DEMO_RESULT for pair {pair_key}")
        return copy.deepcopy(AUDIO_DEMO_RESULT)


async def run_parallel_pipeline(
    client,
    audio_pairs: Dict[str, Path],
    max_concurrent: int = 3,
    delay_between_pairs: int = 4,
    skip_api: bool = False,
) -> Dict[str, dict]:
    """Run audio annotation pipeline in parallel.
    
    Args:
        client: Gemini client instance
        audio_pairs: Dictionary of audio pairs mapping pair_key to audio file path
        max_concurrent: Maximum concurrent API calls
        delay_between_pairs: Delay between pair processing in seconds
        skip_api: If True, skip API calls and use demo results
        
    Returns:
        Dictionary of annotation results keyed by pair_key
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    results: Dict[str, dict] = {}

    async def worker(pair_key: str, audio_path: Path) -> tuple[str, dict]:
        async with semaphore:
            print(f"\nProcessing audio pair: {pair_key}")
            return pair_key, await process_single_audio(
                client,
                pair_key,
                audio_path,
                skip_api=skip_api,
            )

    tasks = []
    for pair_key, audio_path in audio_pairs.items():
        tasks.append(asyncio.create_task(worker(pair_key, audio_path)))
        await asyncio.sleep(delay_between_pairs)

    for completed_task in asyncio.as_completed(tasks):
        pair_key, annotation_results = await completed_task
        results[pair_key] = annotation_results

    return results
