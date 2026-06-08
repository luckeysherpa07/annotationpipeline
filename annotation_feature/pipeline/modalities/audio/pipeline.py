"""Audio-visual cascade pipeline for QA annotation.

This module processes each audio/video-with-audio pair in three steps:
1. Generate human interaction annotations from source day RGB frames.
2. Generate a timestamped audio-visual caption from HIA and with-audio media.
3. Generate sound-centric QA pairs from the timestamped caption.
"""
import asyncio
import copy
import json
import mimetypes
import re
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from prompts.audio_prompts import (
    AUDIO_VISUAL_CAPTION_GENERATION_PROMPT,
    HUMAN_INTERACTION_ANNOTATION_PROMPT,
    QNA_GENERATION_PROMPT,
)

try:
    from google.genai import types
except ImportError:
    types = None


MODEL_NAME = "gemini-3.1-flash-lite"


def _is_quota_exhausted_error(exc: Exception) -> bool:
    error_text = str(exc)
    return (
        "RESOURCE_EXHAUSTED" in error_text
        or "quota" in error_text.lower()
        or "GenerateRequestsPerDayPerProjectPerModel" in error_text
        or "GenerateContentInputTokensPerModelPerMinute" in error_text
        or "GenerateRequestsPerMinutePerProjectPerModel" in error_text
    )

DEMO_HIA_CAPTION = (
    "The egocentric video shows the camera wearer interacting with nearby "
    "objects using their hands. The main interactions involve reaching, "
    "grasping, moving, placing, and manipulating items on a work surface."
)

DEMO_TIMESTAMPED_CAPTION = (
    "[00:00 - 00:04] The user begins manipulating objects near the camera; "
    "soft handling sounds align with visible hand movement.\n"
    "[00:04 - 00:09] A short sequence of sharper contact sounds occurs as an "
    "object is placed or adjusted on the surface.\n"
    "[00:09 - 00:14] The interaction continues with quieter rustling and "
    "movement sounds, consistent with ongoing hand-object activity."
)

DEMO_QA_PAIRS = [
    {
        "timestamp": "00:00 - 00:04",
        "context": "[00:00] Soft handling sounds occur while the user manipulates nearby objects.",
        "question_type": "Sound Source Identification",
        "question": "What action most likely produced the soft handling sounds at the start?",
        "answer": "The user's hand-object manipulation produced the sounds.",
    },
    {
        "timestamp": "00:04 - 00:09",
        "context": "[00:04] Sharper contact sounds occur as an object is placed or adjusted.",
        "question_type": "Sound Characteristics",
        "question": "What was the character of the contact sounds in the middle segment?",
        "answer": "They were sharper contact sounds from placing or adjusting an object.",
    },
]


def build_hia_prompt() -> str:
    prompt = HUMAN_INTERACTION_ANNOTATION_PROMPT["caption_prompt"]
    return "\n".join(
        [
            "You are analyzing an egocentric RGB video.",
            prompt,
            "Return only the final HIA caption as plain text.",
        ]
    )


def build_audio_visual_prompt(hia_caption: str) -> str:
    return "\n".join(
        [
            AUDIO_VISUAL_CAPTION_GENERATION_PROMPT,
            "",
            "Human Interaction Annotations (HIA):",
            hia_caption,
            "",
            "Return only the timestamped audio-visual caption.",
        ]
    )


def build_qna_prompt(timestamped_caption: str) -> str:
    return "\n".join(
        [
            QNA_GENERATION_PROMPT,
            "",
            "Detailed audiovisual caption:",
            timestamped_caption,
            "",
            "Return only a valid JSON list. Do not include markdown.",
        ]
    )


def build_caption_and_qna_prompt(hia_caption: str) -> str:
    return "\n".join(
        [
            AUDIO_VISUAL_CAPTION_GENERATION_PROMPT,
            "",
            "Human Interaction Annotations (HIA):",
            hia_caption,
            "",
            "After generating the timestamped audio-visual caption, generate high-quality QA pairs grounded in that caption.",
            "Return only a single valid JSON object with this exact structure:",
            '{"caption": "<timestamped audio-visual caption>", "qa_pairs": [ ... ]}',
            "Each qa_pairs item must follow the same JSON schema required by the QA generation step.",
            "Do not include markdown fences or any extra text.",
        ]
    )


def _strip_markdown_fence(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.I)
    return cleaned.strip()


def parse_hia_response(text: str) -> str:
    cleaned = _strip_markdown_fence(text)
    if not cleaned:
        raise ValueError("Empty HIA response")
    return cleaned


def parse_caption_response(text: str) -> str:
    cleaned = _strip_markdown_fence(text)
    if not cleaned:
        raise ValueError("Empty audio-visual caption response")
    return cleaned


def parse_qna_response(text: str) -> list[dict]:
    cleaned = _strip_markdown_fence(text)
    match = re.search(r"\[.*\]", cleaned, flags=re.S)
    if not match:
        raise ValueError("No JSON list found in Q&A response")

    parsed = json.loads(match.group(0))
    if not isinstance(parsed, list):
        raise ValueError("Q&A response must be a JSON list")

    normalized: list[dict] = []
    for item in parsed:
        if isinstance(item, dict):
            normalized.append(item)
    if not normalized:
        raise ValueError("Q&A response did not contain any JSON objects")
    return normalized


def parse_caption_and_qna_response(text: str) -> tuple[str, list[dict]]:
    cleaned = _strip_markdown_fence(text)
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if not match:
        raise ValueError("No JSON object found in combined audio response")

    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Combined audio response must be a JSON object")

    caption = str(parsed.get("caption", "")).strip()
    if not caption:
        raise ValueError("Combined audio response caption is empty")

    qa_pairs = parsed.get("qa_pairs", [])
    if not isinstance(qa_pairs, list):
        raise ValueError("Combined audio response qa_pairs must be a JSON list")

    normalized: list[dict] = []
    for item in qa_pairs:
        if isinstance(item, dict):
            normalized.append(item)

    if not normalized:
        raise ValueError("Combined audio response did not contain any QA objects")

    return caption, normalized


def _slug_question_type(question_type: Any) -> str:
    text = str(question_type or "").strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return slug or "qa_pair"


def _derive_audio_quality_flags(raw_results: dict) -> list[str]:
    metadata = raw_results.get("_audio_quality", {})
    if not isinstance(metadata, dict):
        return []

    flags: list[str] = []
    if metadata.get("missing_hia_source"):
        flags.append("missing_hia_source")
    if metadata.get("demo_hia_fallback"):
        flags.append("demo_hia_fallback")
    return flags


def _fallback_audio_category_caption(
    context: str,
    timestamp: Any,
    chronological_caption: str,
) -> str:
    context_text = str(context or "").strip()
    if context_text:
        return context_text

    timestamp_text = str(timestamp or "").strip()
    caption_text = str(chronological_caption or "").strip()
    if timestamp_text and caption_text:
        return f"[{timestamp_text}] {caption_text}"
    return caption_text


def enrich_audio_annotations(annotation_results: dict, day_rgb_file: str | None = None) -> dict:
    if not isinstance(annotation_results, dict):
        return {}

    enriched = copy.deepcopy(annotation_results)
    quality_flags = list(enriched.get("quality_flags", [])) if isinstance(enriched.get("quality_flags"), list) else []

    if not str(day_rgb_file or "").strip():
        if "missing_hia_source" not in quality_flags:
            quality_flags.append("missing_hia_source")

    hia_caption = str(((enriched.get("audio_hia") or {}).get("caption")) or "").strip()
    if hia_caption == DEMO_HIA_CAPTION.strip():
        if "demo_hia_fallback" not in quality_flags:
            quality_flags.append("demo_hia_fallback")

    chronological_caption = str(((enriched.get("audio_chronological_caption") or {}).get("caption")) or "").strip()
    categories = enriched.get("categories", {})
    if isinstance(categories, dict):
        for section_name, item in categories.items():
            if not isinstance(item, dict):
                continue
            item_flags = list(item.get("quality_flags", [])) if isinstance(item.get("quality_flags"), list) else []
            has_qa = bool(str(item.get("question", "")).strip() and str(item.get("answer", "")).strip())
            empty_caption = not str(item.get("caption", "")).strip()
            if has_qa and empty_caption:
                if "empty_qa_caption" not in item_flags:
                    item_flags.append("empty_qa_caption")
                if "has_empty_qa_caption" not in quality_flags:
                    quality_flags.append("has_empty_qa_caption")
                item["caption"] = _fallback_audio_category_caption(
                    item.get("caption", ""),
                    item.get("timestamp"),
                    chronological_caption,
                )
            if item_flags:
                item["quality_flags"] = item_flags
            elif "quality_flags" in item:
                item.pop("quality_flags", None)
            categories[section_name] = item

    enriched["categories"] = categories if isinstance(categories, dict) else {}
    if quality_flags:
        enriched["quality_flags"] = quality_flags
    else:
        enriched.pop("quality_flags", None)
    return enriched


def format_audio_annotations(raw_results: Any) -> dict:
    """Convert raw audio cascade output into section-level annotations."""
    if not isinstance(raw_results, dict):
        raw_results = {}

    formatted: dict = {}
    categories: dict = {}
    quality_flags = _derive_audio_quality_flags(raw_results)

    hia_caption = raw_results.get("hia", "")
    if isinstance(hia_caption, str) and hia_caption.strip():
        formatted["audio_hia"] = {"caption": hia_caption}

    chronological_caption = raw_results.get("caption", "")
    if isinstance(chronological_caption, str) and chronological_caption.strip():
        formatted["audio_chronological_caption"] = {"caption": chronological_caption}

    section_counts: dict[str, int] = {}
    qa_pairs = raw_results.get("qa_pairs", [])
    if isinstance(qa_pairs, list):
        for qa_pair in qa_pairs:
            if not isinstance(qa_pair, dict):
                continue

            section_base = f"audio_{_slug_question_type(qa_pair.get('question_type'))}"
            section_counts[section_base] = section_counts.get(section_base, 0) + 1
            section_name = section_base
            if section_counts[section_base] > 1:
                section_name = f"{section_base}_{section_counts[section_base]}"

            context = qa_pair.get("context", "")
            question = qa_pair.get("question", "")
            answer = qa_pair.get("answer", "")
            timestamp = qa_pair.get("timestamp")
            item_quality_flags: list[str] = []
            if str(question or "").strip() and str(answer or "").strip() and not str(context or "").strip():
                item_quality_flags.append("empty_qa_caption")
                if "has_empty_qa_caption" not in quality_flags:
                    quality_flags.append("has_empty_qa_caption")

            categories[section_name] = {
                "timestamp": timestamp,
                "caption": _fallback_audio_category_caption(context, timestamp, chronological_caption),
                "question": question if isinstance(question, str) else "",
                "answer": answer if isinstance(answer, str) else "",
            }
            if item_quality_flags:
                categories[section_name]["quality_flags"] = item_quality_flags

    formatted["categories"] = categories
    if quality_flags:
        formatted["quality_flags"] = quality_flags
    return enrich_audio_annotations(formatted)


def _nest_audio_categories(annotation_results: dict) -> dict:
    """Move flat audio QA sections under annotations.categories."""
    formatted: dict = {}

    for section_name in ("audio_hia", "audio_chronological_caption"):
        section_value = annotation_results.get(section_name)
        if isinstance(section_value, dict):
            formatted[section_name] = section_value

    categories: dict = {}
    existing_categories = annotation_results.get("categories")
    if isinstance(existing_categories, dict):
        categories.update(existing_categories)

    for section_name, section_value in annotation_results.items():
        if section_name in {"audio_hia", "audio_chronological_caption", "categories"}:
            continue
        categories[section_name] = section_value

    formatted["categories"] = categories
    return formatted


async def call_gemini_with_retry(client, contents: list, max_retries: int = 3) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=MODEL_NAME,
                contents=contents,
            )
            return response.text
        except Exception as exc:
            if _is_quota_exhausted_error(exc):
                raise
            if attempt == max_retries:
                raise
            await asyncio.sleep(2)
    raise RuntimeError("Gemini call failed")


def _guess_mime_type(path: Path, fallback: str = "video/mp4") -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type or fallback


async def _wait_for_active_file(client, uploaded_file, timeout_seconds: int = 180):
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    current_file = uploaded_file

    while True:
        state = getattr(current_file, "state", None)
        state_value = getattr(state, "value", state)

        if state_value == "ACTIVE":
            return current_file
        if state_value == "FAILED":
            raise RuntimeError(f"Uploaded file failed processing: {getattr(current_file, 'name', '')}")
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for uploaded file to become ACTIVE: {getattr(current_file, 'name', '')}"
            )

        await asyncio.sleep(2)
        current_file = await asyncio.to_thread(
            client.files.get,
            name=current_file.name,
        )


async def _upload_file_part(client, path: Path, mime_type: str):
    if types is None:
        raise ImportError("google-genai types are not available")

    uploaded_file = await asyncio.to_thread(
        client.files.upload,
        file=str(path),
        config={"mime_type": mime_type},
    )
    uploaded_file = await _wait_for_active_file(client, uploaded_file)
    return types.Part(uploaded_file)


async def generate_hia_caption(client, rgb_frames: list[Path] | None, skip_api: bool = False) -> str:
    if skip_api:
        return DEMO_HIA_CAPTION

    if not rgb_frames:
        print("    WARNING: Source day RGB frames missing; using demo HIA caption")
        return DEMO_HIA_CAPTION

    try:
        from ...utils import build_image_parts, encode_frames_to_base64

        encoded_frames = encode_frames_to_base64(rgb_frames)
        if not encoded_frames:
            print("    WARNING: Could not encode source day RGB frames; using demo HIA caption")
            return DEMO_HIA_CAPTION

        image_parts = build_image_parts(encoded_frames)
        response_text = await call_gemini_with_retry(client, image_parts + [build_hia_prompt()])
        return parse_hia_response(response_text)
    except Exception as e:
        if _is_quota_exhausted_error(e):
            print(f"    WARNING: HIA generation hit quota for current model: {e}")
            raise
        print(f"    WARNING: HIA generation failed from source day RGB frames: {e}")
        return DEMO_HIA_CAPTION


async def generate_audiovisual_caption(
    client,
    hia_caption: str,
    audio_path: Path | None,
    skip_api: bool = False,
) -> str:
    if skip_api:
        return DEMO_TIMESTAMPED_CAPTION

    if not audio_path or not audio_path.exists():
        print("    WARNING: With-audio media missing; using demo timestamped caption")
        return DEMO_TIMESTAMPED_CAPTION

    try:
        media_part = await _upload_file_part(client, audio_path, _guess_mime_type(audio_path))
        prompt = build_audio_visual_prompt(hia_caption)
        response_text = await call_gemini_with_retry(client, [media_part, prompt])
        return parse_caption_response(response_text)
    except Exception as e:
        print(f"    WARNING: Audio-visual caption generation failed for {audio_path}: {e}")
        return DEMO_TIMESTAMPED_CAPTION


async def generate_caption_and_qa_pairs(
    client,
    hia_caption: str,
    audio_path: Path | None,
    skip_api: bool = False,
) -> tuple[str, list[dict]] | None:
    if skip_api:
        return DEMO_TIMESTAMPED_CAPTION, copy.deepcopy(DEMO_QA_PAIRS)

    if not audio_path or not audio_path.exists():
        print("    WARNING: With-audio media missing; leaving pair pending")
        return None

    try:
        media_part = await _upload_file_part(client, audio_path, _guess_mime_type(audio_path))
        prompt = build_caption_and_qna_prompt(hia_caption)
        response_text = await call_gemini_with_retry(client, [media_part, prompt])
        return parse_caption_and_qna_response(response_text)
    except Exception as e:
        print(f"    WARNING: Combined audio caption/QA generation failed for {audio_path}: {e}")
        return None


async def generate_qa_pairs(
    client,
    timestamped_caption: str,
    skip_api: bool = False,
) -> list[dict]:
    if skip_api:
        return copy.deepcopy(DEMO_QA_PAIRS)

    try:
        response_text = await call_gemini_with_retry(
            client,
            [build_qna_prompt(timestamped_caption)],
        )
        return parse_qna_response(response_text)
    except Exception as e:
        print(f"    WARNING: Q&A generation failed: {e}")
        return copy.deepcopy(DEMO_QA_PAIRS)


async def process_single_audio_pair(
    client,
    pair_key: str,
    rgb_videos: Dict[str, Any],
    audio_path: Path | None,
    skip_api: bool = False,
    ) -> dict | None:
    try:
        day_rgb_frames = rgb_videos.get("day_frames", []) if isinstance(rgb_videos, dict) else []
        missing_hia_source = not day_rgb_frames
        hia_caption = await generate_hia_caption(client, day_rgb_frames, skip_api=skip_api)
        combined_result = await generate_caption_and_qa_pairs(
            client,
            hia_caption,
            audio_path,
            skip_api=skip_api,
        )
        if combined_result is None:
            return None
        timestamped_caption, qa_pairs = combined_result
        return {
            "hia": hia_caption,
            "caption": timestamped_caption,
            "qa_pairs": qa_pairs,
            "_audio_quality": {
                "missing_hia_source": missing_hia_source,
                "demo_hia_fallback": hia_caption == DEMO_HIA_CAPTION,
            },
        }
    except Exception as e:
        print(f"    ERROR: Audio cascade failed for {pair_key}: {e}")
        return None


async def process_single_audio(
    client,
    pair_key: str,
    audio_path: Path,
    skip_api: bool = False,
) -> dict | None:
    """Backward-compatible wrapper for callers that only provide audio media."""
    return await process_single_audio_pair(
        client,
        pair_key,
        {"day": None, "night": None},
        audio_path,
        skip_api=skip_api,
    )


def normalize_annotation_results(raw_results: Any) -> dict:
    """Backward-compatible normalizer for legacy imports."""
    if isinstance(raw_results, dict):
        if {"hia", "caption", "qa_pairs"} <= set(raw_results):
            return format_audio_annotations(raw_results)
        if any(str(key).startswith("audio_") for key in raw_results) or "categories" in raw_results:
            return enrich_audio_annotations(_nest_audio_categories(raw_results))
    return format_audio_annotations({
        "hia": DEMO_HIA_CAPTION,
        "caption": DEMO_TIMESTAMPED_CAPTION,
        "qa_pairs": copy.deepcopy(DEMO_QA_PAIRS),
    })


async def run_parallel_pipeline(
    client,
    audio_pairs: Dict[str, Path],
    rgb_frames_dict: Dict[str, Dict[str, Any]] | None = None,
    max_concurrent: int = 3,
    delay_between_pairs: int = 4,
    skip_api: bool = False,
    on_pair_complete=None,
) -> Dict[str, dict]:
    """Run the audio-visual cascade in parallel."""
    semaphore = asyncio.Semaphore(max_concurrent)
    results: Dict[str, dict] = {}
    rgb_video_dict = rgb_frames_dict or {}

    async def worker(pair_key: str, audio_path: Path) -> tuple[str, dict | None]:
        async with semaphore:
            print(f"\nProcessing audio-visual pair: {pair_key}")
            return pair_key, await process_single_audio_pair(
                client,
                pair_key,
                rgb_video_dict.get(pair_key, {}),
                audio_path,
                skip_api=skip_api,
            )

    tasks = []
    for pair_key, audio_path in audio_pairs.items():
        tasks.append(asyncio.create_task(worker(pair_key, audio_path)))
        await asyncio.sleep(delay_between_pairs)

    for completed_task in asyncio.as_completed(tasks):
        pair_key, annotation_results = await completed_task
        if annotation_results is None:
            continue
        results[pair_key] = annotation_results
        if on_pair_complete:
            on_pair_complete(pair_key, annotation_results)

    return results
