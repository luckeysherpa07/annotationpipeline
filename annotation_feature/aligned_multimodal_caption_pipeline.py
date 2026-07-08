"""Generate cross-modal disambiguation captions from aligned frame caches."""

from __future__ import annotations

import argparse
import asyncio
import base64
import copy
import json
import hashlib
import re
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

try:
    from google.genai import types as genai_types
except ImportError:
    genai_types = None

from annotation_feature.pipeline.client import create_gemini_client
from annotation_feature.pipeline.utils import build_image_parts

from annotation_feature.aligned_multimodal_sampling import (
    MultimodalSamplingJob,
    load_frame_dirs,
    pair_frame_dirs,
    select_paired_frames,
    frame_index,
    frames_by_index,
    ADAPTIVE_CHANGE_WEIGHT,
    ADAPTIVE_COVERAGE_WEIGHT
)

from annotation_feature.aligned_caption_schema import (
    CaptionParseError,
    CaptionValidationError,
    MIN_DETAILED_CAPTION_WORDS,
)
from annotation_feature.aligned_caption_prompt import (
    _build_caption_prompt,
    _template_caption,
)
from annotation_feature.aligned_caption_validation import _validate_caption_schema

SELECTION_ALGORITHM_VERSION = "uniform_adaptive_v1"

def build_selection_config_fingerprint(
    num_uniform_frames: int,
    num_adaptive_frames: int,
) -> str:
    payload = {
        "algorithm": "uniform_adaptive",
        "algorithm_version": SELECTION_ALGORITHM_VERSION,
        "num_uniform_frames": num_uniform_frames,
        "num_adaptive_frames": num_adaptive_frames,
        "adaptive_change_weight": ADAPTIVE_CHANGE_WEIGHT,
        "adaptive_coverage_weight": ADAPTIVE_COVERAGE_WEIGHT,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]



DEFAULT_INPUT_PATH = Path("outputs/aligned_multimodal_visual_evidence_units_filtered.json")
DEFAULT_OUTPUT_PATH = Path("outputs/aligned_cross_modal_disambiguation_captions_gemini.json")
DEFAULT_COMPOSITE_ROOT = Path("outputs/composite_frames")
DEFAULT_DATASET_ROOT = Path("aligned_dataset")
DEFAULT_MODEL_NAME = "gemini-3.1-flash-lite"
DEFAULT_MAX_TRANSPORT_RETRIES = 6
MAX_TRANSPORT_RETRY_WAIT_SECONDS = 60

@dataclass(frozen=True)
class CaptionTask:
    caption_id: str
    segment_id: str
    split_dir: str
    segment_name: str
    side: str
    modality1: str
    modality2: str
    frame_dir1: Path
    frame_dir2: Path
    frames1: tuple[Path, ...]
    frames2: tuple[Path, ...]
    composite_frames: tuple[Path, ...]
    sampling_strategy: str
    uniform_anchor_indexes: tuple[int, ...]
    adaptive_frame_indexes: tuple[int, ...]
    selected_frame_indexes: tuple[int, ...]
    candidate_frame_indexes: tuple[int, ...]
    selection_config_fingerprint: str





def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object in {path}")
    return data


def _save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")


_FONT_CACHE: dict[int, ImageFont.ImageFont] = {}


def _load_font(size: int = 28) -> ImageFont.ImageFont:
    if size not in _FONT_CACHE:
        try:
            _FONT_CACHE[size] = ImageFont.truetype("arial.ttf", size=size)
        except OSError:
            _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def _resize_to_height(image: Image.Image, height: int) -> Image.Image:
    if image.height == height:
        return image
    width = max(1, round(image.width * height / image.height))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _draw_label(image: Image.Image, label: str, xy: tuple[int, int]) -> None:
    draw = ImageDraw.Draw(image)
    font = _load_font()
    x, y = xy
    padding = 10
    bbox = draw.textbbox((x, y), label, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    draw.rectangle(
        (x, y, x + width + padding * 2, y + height + padding * 2),
        fill=(0, 0, 0),
    )
    draw.text((x + padding, y + padding), label, fill=(255, 255, 255), font=font)


def _compose_frame(
    frame1: Path,
    frame2: Path,
    modality1: str,
    modality2: str,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(frame1) as left_raw, Image.open(frame2) as right_raw:
        left = left_raw.convert("RGB")
        right = right_raw.convert("RGB")
        target_height = min(left.height, right.height)
        left = _resize_to_height(left, target_height)
        right = _resize_to_height(right, target_height)
        canvas = Image.new("RGB", (left.width + right.width, target_height), (0, 0, 0))
        canvas.paste(left, (0, 0))
        canvas.paste(right, (left.width, 0))
        _draw_label(canvas, f"LEFT: {modality1.upper()}", (0, 0))
        _draw_label(canvas, f"RIGHT: {modality2.upper()}", (left.width, 0))
        canvas.save(output_path)
    return output_path


def _ensure_composite_frame(
    frame1: Path,
    frame2: Path,
    modality1: str,
    modality2: str,
    output_path: Path,
) -> Path:
    if not output_path.exists():
        _compose_frame(
            frame1,
            frame2,
            modality1,
            modality2,
            output_path,
        )
    return output_path


def _ensure_composite_frames(task: CaptionTask) -> None:
    for frame1, frame2, composite_frame in zip(
        task.frames1,
        task.frames2,
        task.composite_frames,
    ):
        _ensure_composite_frame(
            frame1,
            frame2,
            task.modality1,
            task.modality2,
            composite_frame,
        )


def _caption_id(segment_id: str, side: str, modality1: str, modality2: str) -> str:
    return "__".join(
        [
            _safe_name(segment_id).lower(),
            _safe_name(side).lower(),
            modality1,
            modality2,
        ]
    )


def _parse_sides(values: str | None) -> set[str] | None:
    if not values:
        return None
    return {val.strip().lower() for val in values.split(",") if val.strip()}


def _parse_pairs(values: str | None) -> set[tuple[str, str]] | None:
    if not values:
        return None
    pairs: set[tuple[str, str]] = set()
    for chunk in values.split(","):
        token = chunk.strip().lower()
        if not token:
            continue
        if "+" in token:
            first, second = token.split("+", 1)
        elif "->" in token:
            first, second = token.split("->", 1)
        else:
            raise ValueError(f"Invalid pair/direction token: {chunk}")
        pairs.add((first.strip(), second.strip()))
    return pairs


def _directions_for_pair(pair: list[str] | tuple[str, str], allowed: set[tuple[str, str]] | None) -> list[tuple[str, str]]:
    first, second = str(pair[0]).lower(), str(pair[1]).lower()
    # A single caption call can represent cross-modal relations in either direction or both directions.
    # This does not imply that both directions must provide gain.
    # Asymmetric contribution, confirmation-only relations, unidirectional disambiguation, and mutual complementarity are all valid.
    # By default we only generate one canonical task per pair (the ordering from the input data) to avoid calling Gemini twice
    # on the same pair. Use --directions to explicitly override the ordering.
    if allowed is None:
        return [(first, second)]
    return [(d_first, d_second) for d_first, d_second in [(first, second), (second, first)] if (d_first, d_second) in allowed]


def build_caption_tasks(
    input_path: Path,
    dataset_root: Path,
    composite_root: Path,
    sampling_strategy: str,
    num_uniform_frames: int,
    num_adaptive_frames: int,
    existing_items: list[dict[str, Any]],
    existing_skipped: list[dict[str, Any]] | None = None,
    allowed_pairs: set[tuple[str, str]] | None = None,
    allowed_directions: set[tuple[str, str]] | None = None,
    allowed_sides: set[str] | None = None,
    limit: int | None = None,
    limit_scenes: int | None = None,
    limit_scene_folders: int | None = None,
    write_composites: bool = True,
) -> tuple[list[CaptionTask], list[dict[str, Any]], int]:
    data = _load_json(input_path)
    segments = data.get("segments")
    if not isinstance(segments, dict):
        raise ValueError(f"Expected {input_path} to contain a segments object")

    skipped: list[dict[str, Any]] = []
    all_jobs: list[MultimodalSamplingJob] = []
    
    for segment_id, segment in sorted(segments.items()):
        if not isinstance(segment, dict):
            continue
            
        split_dir = str(segment.get("split_dir") or "")
        segment_name = str(segment.get("segment_name") or "")
        if not split_dir or not segment_name:
            skipped.append({"segment_id": segment_id, "reason": "missing split_dir or segment_name"})
            continue
            
        pairs = segment.get("modality_pairs") or []
        for pair in pairs:
            if not isinstance(pair, list | tuple) or len(pair) != 2:
                continue
            
            pair_tuple = (str(pair[0]).lower(), str(pair[1]).lower())
            if allowed_pairs is not None and pair_tuple not in allowed_pairs and pair_tuple[::-1] not in allowed_pairs:
                continue
            pair_directions = _directions_for_pair(pair, allowed_directions)
                
            for modality1, modality2 in pair_directions:
                dirs1 = load_frame_dirs(dataset_root, split_dir, segment_name, modality1)
                dirs2 = load_frame_dirs(dataset_root, split_dir, segment_name, modality2)
                if not dirs1 or not dirs2:
                    skipped.append({
                        "segment_id": segment_id, "split_dir": split_dir, "segment_name": segment_name,
                        "modality1": modality1, "modality2": modality2, "reason": "missing frame cache directory",
                    })
                    continue
                for side, dir1, dir2 in pair_frame_dirs(dirs1, dirs2):
                    if allowed_sides is not None and side.lower() not in allowed_sides:
                        continue
                        
                    by_index1 = frames_by_index(dir1, modality1)
                    by_index2 = frames_by_index(dir2, modality2)
                    shared_indexes = tuple(sorted(set(by_index1) & set(by_index2)))
                    
                    if not shared_indexes:
                        skipped.append({
                            "segment_id": segment_id, "side": side, "modality1": modality1, "modality2": modality2,
                            "frame_dir1": dir1.as_posix(), "frame_dir2": dir2.as_posix(), "reason": "no shared frame indexes",
                        })
                        continue
                    
                    if len(shared_indexes) != 30:
                        skipped.append({
                            "segment_id": segment_id, "side": side, "modality1": modality1, "modality2": modality2,
                            "frame_dir1": dir1.as_posix(), "frame_dir2": dir2.as_posix(), "reason": f"expected 30 frames, got {len(shared_indexes)}",
                        })
                        continue

                    all_jobs.append(MultimodalSamplingJob(
                        segment_id=segment_id, split_dir=split_dir, segment_name=segment_name,
                        side=side, modality1=modality1, modality2=modality2, dir1=dir1, dir2=dir2,
                        shared_indexes=shared_indexes, by_index1=by_index1, by_index2=by_index2,
                    ))

    selection_config_fingerprint = build_selection_config_fingerprint(num_uniform_frames, num_adaptive_frames)

    completed_keys = {
        (item.get("caption_id"), item.get("selection_config_fingerprint"))
        for item in existing_items
    }

    selected_jobs = []
    selected_scenes = set()
    selected_scene_folders = set()
    
    for job in all_jobs:
        if limit_scenes is not None and len(selected_scenes) >= limit_scenes and job.segment_id not in selected_scenes:
            continue
        if limit_scene_folders is not None and len(selected_scene_folders) >= limit_scene_folders and job.split_dir not in selected_scene_folders:
            continue
            
        selected_jobs.append(job)
        selected_scenes.add(job.segment_id)
        selected_scene_folders.add(job.split_dir)

    if limit is not None:
        selected_jobs = selected_jobs[:limit]

    total_selected_jobs = len(selected_jobs)
    selected_job_keys = {
        (
            str(job.segment_id),
            str(job.side).lower(),
            str(job.modality1).lower(),
            str(job.modality2).lower(),
        )
        for job in selected_jobs
    }
    selected_scope_segments = {str(job.segment_id) for job in selected_jobs}
    selected_scope_folders = {str(job.split_dir) for job in selected_jobs}

    def _skip_matches_selected_scope(item: dict[str, Any]) -> bool:
        if limit is None and limit_scenes is None and limit_scene_folders is None:
            return True

        side = item.get("side")
        modality1 = item.get("modality1")
        modality2 = item.get("modality2")
        if side is not None and modality1 is not None and modality2 is not None:
            return (
                str(item.get("segment_id")),
                str(side).lower(),
                str(modality1).lower(),
                str(modality2).lower(),
            ) in selected_job_keys

        if str(item.get("segment_id")) in selected_scope_segments:
            return True
        if str(item.get("split_dir")) in selected_scope_folders:
            return True
        return False

    all_skipped = skipped
    if existing_skipped:
        all_skipped = existing_skipped + skipped

    run_scoped_skipped = [item for item in all_skipped if _skip_matches_selected_scope(item)]

    pending_jobs = []
    for job in selected_jobs:
        caption_id = _caption_id(str(job.segment_id), job.side, job.modality1, job.modality2)
        if (caption_id, selection_config_fingerprint) not in completed_keys:
            pending_jobs.append(job)



    tasks: list[CaptionTask] = []
    
    for job in pending_jobs:
        frames1, frames2, strategy, anchors, adaptive, selected = select_paired_frames(
            job.dir1,
            job.dir2,
            job.modality1,
            job.modality2,
            sampling_strategy,
            num_uniform_frames,
            num_adaptive_frames,
        )

        caption_id = _caption_id(str(job.segment_id), job.side, job.modality1, job.modality2)
        output_dir = (
            composite_root
            / _safe_name(job.split_dir)
            / _safe_name(job.segment_name)
            / _safe_name(job.side)
            / f"{job.modality1}__{job.modality2}"
        )
        composite_frames: list[Path] = []
        for index, (f1, f2) in enumerate(zip(frames1, frames2), start=1):
            frame_number = frame_index(f1)
            suffix = f"{frame_number:06d}" if frame_number is not None else f"{index:03d}"
            output_path = output_dir / f"frame_{suffix}.png"
            if write_composites:
                _compose_frame(
                    f1,
                    f2,
                    job.modality1,
                    job.modality2,
                    output_path,
                )
            composite_frames.append(output_path)
            
        tasks.append(
            CaptionTask(
                caption_id=caption_id,
                segment_id=str(job.segment_id),
                split_dir=job.split_dir,
                segment_name=job.segment_name,
                side=job.side,
                modality1=job.modality1,
                modality2=job.modality2,
                frame_dir1=job.dir1,
                frame_dir2=job.dir2,
                frames1=frames1,
                frames2=frames2,
                composite_frames=tuple(composite_frames),
                sampling_strategy=strategy,
                uniform_anchor_indexes=tuple(anchors),
                adaptive_frame_indexes=tuple(adaptive),
                selected_frame_indexes=tuple(selected),
                candidate_frame_indexes=tuple(job.shared_indexes),
                selection_config_fingerprint=selection_config_fingerprint,
            )
        )
        
    return tasks, run_scoped_skipped, total_selected_jobs


def _encode_images(paths: tuple[Path, ...]) -> list[str]:
    encoded: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        with open(path, "rb") as handle:
            encoded.append(base64.standard_b64encode(handle.read()).decode("utf-8"))
    return encoded


def _parse_json_response(text: str) -> dict[str, Any]:
    if not text:
        raise CaptionParseError("Empty Gemini response")
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.I)
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if not match:
        raise CaptionParseError("No JSON object found in Gemini response")
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise CaptionParseError(f"Failed to decode JSON: {e}")
    if not isinstance(parsed, dict):
        raise CaptionParseError("Gemini response must be a JSON object")
    return parsed


def _build_validation_retry_hint(exc: Exception, category: str) -> str:
    message = str(exc).lower()
    hints: list[str] = []
    
    if "information_gain" in message:
        hints.append(
            "For information_gain, if emitted, it must be fully populated. "
            "All *_can_determine, *_cannot_determine, and fusion_additionally_reveals fields "
            "must be JSON lists, even when empty. Required shape: "
            '{"entity_id": "...", "video1_evidence_refs": [], "video2_evidence_refs": [], '
            '"video1_can_determine": [], "video1_cannot_determine": [], "video2_can_determine": [], '
            '"video2_cannot_determine": [], "fusion_additionally_reveals": [], '
            '"gain_type": "...", "gain_rating": "..."}'
        )
        
    if "qa_relevant_details" in message:
        hints.append(
            "For qa_relevant_details, if emitted, it must be fully populated. "
            "Required shape: "
            '{"detail_id": "qa_detail_...", "reasoning_pattern": "<allowed enum>", '
            '"supporting_refs": ["..."], "why_question_worthy": "..."}'
        )
        
    if "ambiguity_events" in message:
        hints.append(
            "For ambiguity_events, if emitted, it must contain all required fields: "
            "ambiguity_id, target_entity, direction, ambiguous_video, resolving_video, "
            "low_confidence_observation, why_ambiguous_video_cannot_resolve, candidate_hypotheses, "
            "resolving_discriminative_evidence, eliminated_hypotheses, fusion_conclusion, "
            "missing_attribute_type, ambiguous_evidence_refs, resolving_evidence_refs."
        )

    if category == "blocklist_failure":
        hints.append(
            "Rewrite the reported field using physical-world wording only. "
            "Do not mention sensor names, modality names, camera/frame/image-processing terms, "
            "or image-quality descriptions. Keep detailed captions above the minimum word count."
        )
    if "too short" in message:
        hints.append(
            "Expand the reported detailed_caption into a complete source-local paragraph of at least "
            f"{MIN_DETAILED_CAPTION_WORDS} words, while keeping forbidden sensor-quality wording out."
        )
    if "generic sensor-theory" in message:
        hints.append(
            "Rewrite the reported field using segment-specific evidence. "
            "Describe what is directly observable or difficult to determine in the supplied frames, "
            "rather than stating general sensor theory."
        )
    if category in {"invalid_reference", "missing_attribute_recovery"}:
        hints.append(
            "Re-check all IDs and references after the edit. Every referenced atom, entity, event, and ambiguity item "
            "must exist and must keep the required prefix."
        )
    if not hints:
        return ""
    return " Targeted repair guidance: " + " ".join(hints)


def _is_transport_error(exc: Exception) -> bool:
    text = str(exc).lower()
    transport_markers = (
        "timed out",
        "timeout",
        "connection reset",
        "connection aborted",
        "connection error",
        "temporarily unavailable",
        "503",
        "504",
    )
    return any(marker in text for marker in transport_markers)


def _transport_error_category(exc: Exception) -> str:
    text = str(exc).lower()
    if "timeout" in text or "timed out" in text or "504" in text:
        return "transport_timeout"
    return "transport_other"


def _transport_retry_wait_seconds(transport_attempt: int) -> int:
    return min(MAX_TRANSPORT_RETRY_WAIT_SECONDS, 5 * (2 ** max(0, transport_attempt - 1)))


async def _call_gemini_caption(
    client,
    task: CaptionTask,
    model_name: str,
    max_retries: int,
    max_transport_retries: int,
    api_stats: list[int] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    _ensure_composite_frames(task)
    encoded = _encode_images(task.composite_frames)
    if not encoded:
        raise ValueError("No composite frames found for Gemini call")
    base_contents = build_image_parts(encoded) + [_build_caption_prompt(task)]
    contents = base_contents
    raw_text = None
    validation_attempt = 1
    while validation_attempt <= max_retries:
        try:
            response = None
            for transport_attempt in range(1, max_transport_retries + 1):
                if api_stats is not None:
                    api_stats[0] += 1
                try:
                    response = await asyncio.to_thread(
                        client.models.generate_content,
                        model=model_name,
                        contents=contents,
                    )
                    break
                except Exception as exc:
                    exc_str = str(exc).lower()
                    if "429" in exc_str or "quota" in exc_str:
                        raise
                    if not _is_transport_error(exc):
                        raise

                    category = _transport_error_category(exc)
                    log_msg = {
                        "caption_id": task.caption_id,
                        "attempt": transport_attempt,
                        "validation_attempt": validation_attempt,
                        "transport_attempt": transport_attempt,
                        "max_transport_retries": max_transport_retries,
                        "stage": "generation",
                        "category": category,
                        "message": str(exc),
                    }
                    if transport_attempt < max_transport_retries:
                        log_msg["retry_after_seconds"] = _transport_retry_wait_seconds(transport_attempt)
                    print(f"    Failure: {json.dumps(log_msg)}")

                    if transport_attempt == max_transport_retries:
                        raise
                    await asyncio.sleep(_transport_retry_wait_seconds(transport_attempt))

            if response is None:
                raise RuntimeError("Gemini caption call failed before receiving a response")
            raw_text = response.text
            valid_frame_keys = {path.stem for path in task.composite_frames}
            caption, warnings = _validate_caption_schema(_parse_json_response(raw_text), valid_frame_keys, task.modality1, task.modality2)
            return caption, warnings
        except Exception as exc:
            exc_str = str(exc).lower()
            # Quota / rate-limit errors are permanent for the current key — don't waste retries
            if "429" in exc_str or "quota" in exc_str:
                raise

            # Transport failures are fully handled by the inner retry loop.
            if _is_transport_error(exc):
                raise

            # If it's not a semantic error, fail fast.
            if not isinstance(exc, (CaptionParseError, CaptionValidationError)):
                raise

            if isinstance(exc, CaptionParseError):
                category = "parse_error"
            else:
                if "blocklist" in exc_str or "forbidden" in exc_str:
                    category = "blocklist_failure"
                elif "missing_key_attributes" in exc_str:
                    category = "missing_attribute_recovery"
                elif "qa_relevant_details" in exc_str:
                    category = "qa_mapping_failure"
                elif "reference" in exc_str or "duplicate" in exc_str or "unknown" in exc_str:
                    category = "invalid_reference"
                else:
                    category = "schema_validation_error"
            
            log_msg = {
                "caption_id": task.caption_id,
                "validation_attempt": validation_attempt,
                "max_validation_retries": max_retries,
                "stage": "validation" if isinstance(exc, CaptionValidationError) else ("parse" if isinstance(exc, CaptionParseError) else "generation"),
                "category": category,
                "message": str(exc)
            }
            print(f"    Failure: {json.dumps(log_msg)}")

            if validation_attempt == max_retries:
                if raw_text:
                    exc.last_invalid_response = raw_text
                raise

            # Error-guided retry: tell the model exactly what went wrong
            previous_context = ""
            if raw_text:
                previous_context = f"\n\nHere is your previous invalid response:\n```json\n{raw_text}\n```\n\n"
            error_feedback = (
                f"{previous_context}"
                f"Your previous response failed validation. "
                f"The first detected validation error was: [{exc}]. "
                f"{_build_validation_retry_hint(exc, category)} "
                f"Correct this issue and re-check the entire JSON for all related "
                f"consistency constraints, references, entity IDs, event types, "
                f"evidence links, and cross-field dependencies. "
                f"Return the complete corrected JSON."
            )
            contents = base_contents + [error_feedback]
            validation_attempt += 1
    raise RuntimeError("Gemini caption call failed")


def _task_metadata(task: CaptionTask) -> dict[str, Any]:
    return {
        "caption_id": task.caption_id,
        "segment_id": task.segment_id,
        "split_dir": task.split_dir,
        "segment_name": task.segment_name,
        "side": task.side,
        "modality1": task.modality1,
        "modality2": task.modality2,
        "frame_dir1": task.frame_dir1.as_posix(),
        "frame_dir2": task.frame_dir2.as_posix(),
        "frames1": [path.as_posix() for path in task.frames1],
        "frames2": [path.as_posix() for path in task.frames2],
        "composite_frames": [f.name for f in task.composite_frames],
        "sampling_strategy": task.sampling_strategy,
        "uniform_anchor_indexes": list(task.uniform_anchor_indexes),
        "adaptive_frame_indexes": list(task.adaptive_frame_indexes),
        "selected_frame_indexes": list(task.selected_frame_indexes),
        "candidate_frame_indexes": list(task.candidate_frame_indexes),
        "num_selected_frames": len(task.selected_frame_indexes),
        "selection_config_fingerprint": task.selection_config_fingerprint,
    }

def _task_to_item(task: CaptionTask, status: str, caption: dict[str, Any] | None = None, validation_warnings: list[str] | None = None, reason: str | None = None, attempts: int | None = None, first_attempt_success: bool | None = None, final_error_category: str | None = None, last_invalid_response: str | None = None) -> dict[str, Any]:
    item = _task_metadata(task)
    item.update({
        "status": status,
        "reason": reason,
        "attempts": attempts,
        "first_attempt_success": first_attempt_success,
        "final_error_category": final_error_category,
        "caption": caption,
        "validation_warnings": validation_warnings or [],
    })
    if last_invalid_response is not None:
        item["last_invalid_response"] = last_invalid_response
    return item


def _batch_state_path(output_path: Path | str) -> Path:
    out = Path(output_path)
    return out.with_name(f".{out.stem}_batch_state.json")

def _build_batch_request(task: CaptionTask, model_name: str, req_id: str) -> "genai_types.InlinedRequest":
    if genai_types is None:
        raise ImportError("google.genai is not installed")
    _ensure_composite_frames(task)
    encoded = _encode_images(task.composite_frames)
    if not encoded:
        raise ValueError("No composite frames found for Gemini batch call")
    prompt_part = genai_types.Part.from_text(text=_build_caption_prompt(task))
    base_contents = build_image_parts(encoded) + [prompt_part]
    return genai_types.InlinedRequest(
        model=model_name,
        contents=[genai_types.Content(role="user", parts=base_contents)],
        metadata={"id": req_id}
    )

def _submit_batch(client, tasks: list[CaptionTask], model_name: str, output_path: Path | str, api_key_source: str, planned_total: int) -> None:
    if not tasks:
        print("No pending tasks to submit in batch mode.")
        return
    print(f"Building batch request for {len(tasks)} tasks...")
    requests = [_build_batch_request(t, model_name, str(i)) for i, t in enumerate(tasks)]
    
    MAX_ROTATIONS = 5
    job = None
    for attempt in range(MAX_ROTATIONS + 1):
        print(f"Submitting batch job to Gemini API (Attempt {attempt+1}/{MAX_ROTATIONS+1})...")
        try:
            job = client.batches.create(model=model_name, src=requests)
            break
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str or "resource exhausted" in err_str or "resource_exhausted" in err_str or "resourceexhausted" in err_str:
                if attempt == MAX_ROTATIONS:
                    print(f"FATAL: Exceeded maximum API key rotations during batch submission: {e}")
                    raise
                print(f"WARNING: Quota exhausted or rate limit hit submitting batch. Rotating API key...")
                try:
                    from annotation_feature.pipeline.client import rotate_gemini_client
                    client = rotate_gemini_client()
                except Exception as rot_e:
                    print(f"FATAL: All API keys exhausted or failed to rotate: {rot_e}")
                    raise
            else:
                raise

    print(f"Batch job submitted successfully! Job Name: {job.name}")
    
    state_path = _batch_state_path(output_path)
    state = {
        "job_name": job.name,
        "model_name": model_name,
        "api_key_source": api_key_source,
        "output_path": str(output_path),
        "planned_total": planned_total,
        "submitted_at": datetime.datetime.now().isoformat(),
        "pending_tasks": [_task_metadata(t) for t in tasks]
    }
    _save_json(state, state_path)
    print(f"Saved batch state to {state_path}")
    print("You can exit the script now. Check status later using --fetch-batch.")

def _fetch_batch(client, output_path: Path | str, batch_state_path: Path | str | None = None) -> None:
    out_path = Path(output_path)
    state_path = Path(batch_state_path) if batch_state_path else _batch_state_path(out_path)
    if not state_path.exists():
        print(f"Batch state file not found: {state_path}. Cannot fetch.")
        return
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
        
    saved_api_key_source = state.get("api_key_source", "list")
    if client is None:
        client = create_gemini_client(api_key_source=saved_api_key_source)
            
    if "fetched_at" in state:
        print(f"Batch state file {state_path} was already fetched at {state['fetched_at']}.")
        return

    job_name = state["job_name"]
    print(f"Checking status for batch job: {job_name}")
    job = client.batches.get(name=job_name)
    
    if job.state.name in ("JOB_STATE_PENDING", "JOB_STATE_RUNNING", "JOB_STATE_CANCELLING", "JOB_STATE_PARTIALLY_SUCCEEDED"):
        print(f"Job is still running. Current state: {job.state.name}")
        return
    elif job.state.name in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED"):
        print(f"Job finished with state: {job.state.name}")
        print(f"Error details: {job.error}")
        return
        
    print(f"Job succeeded! Fetching results...")
    
    existing_items, existing_skipped = _load_resume(out_path)
    items = existing_items
    skipped = existing_skipped
    existing_keys = {
        (str(item.get("caption_id")), item.get("selection_config_fingerprint"))
        for item in items 
        if item.get("caption_id")
    }
    
    pending_tasks_list = state.get("pending_tasks", [])
    pending_tasks_map = {str(i): t for i, t in enumerate(pending_tasks_list)}
    
    seen_in_this_fetch = set()
    
    for resp_entry in job.dest.inlined_responses:
        req_id = resp_entry.metadata.get("id") if resp_entry.metadata else None
        if req_id is None:
            continue
        
        task_dict = pending_tasks_map.get(req_id)
        if not task_dict:
            continue
            
        caption_id = task_dict.get("caption_id")
        fingerprint = task_dict.get("selection_config_fingerprint")
        selection_key = (str(caption_id), str(fingerprint))
        
        if not caption_id or selection_key in existing_keys:
            continue
            
        if selection_key in seen_in_this_fetch:
            continue
        seen_in_this_fetch.add(selection_key)
        base_item = task_dict.copy()
        
        if resp_entry.error:
            base_item["reason"] = f"Batch API Error: {resp_entry.error}"
            skipped[:] = [item for item in skipped if item.get("caption_id") != caption_id]
            skipped.append(base_item)
            continue
            
        try:
            raw_text = "".join(part.text for part in resp_entry.response.candidates[0].content.parts)
            valid_frame_keys = {Path(p).stem for p in task_dict.get("composite_frames", [])}
            caption, warnings = _validate_caption_schema(_parse_json_response(raw_text), valid_frame_keys, task_dict.get("modality1", ""), task_dict.get("modality2", ""))
            base_item["status"] = "generated_batch"
            base_item["caption"] = caption
            base_item["validation_warnings"] = warnings
            skipped[:] = [item for item in skipped if item.get("caption_id") != caption_id]
            items[:] = [item for item in items if item.get("caption_id") != caption_id]
            items.append(base_item)
        except Exception as exc:
            base_item["reason"] = f"Validation Error: {exc}"
            skipped[:] = [item for item in skipped if item.get("caption_id") != caption_id]
            skipped.append(base_item)
            
    _save_json(
        _build_output_payload(
            input_path=Path("batch_mode"),
            dataset_root=Path("batch_mode"),
            composite_root=Path("batch_mode"),
            output_path=out_path,
            model_name=state.get("model_name", "unknown"),
            generation_mode="batch",
            items=items,
            skipped=skipped,
            planned_total=state.get("planned_total", len(items) + len(skipped)),
            gemini_calls=len(job.dest.inlined_responses),
        ),
        out_path,
    )
    
    state["fetched_at"] = datetime.datetime.now().isoformat()
    _save_json(state, state_path)
    
    print(f"Wrote batch fetched results to {out_path}")


def _build_output_payload(
    input_path: Path,
    dataset_root: Path,
    composite_root: Path,
    output_path: Path,
    model_name: str,
    generation_mode: str,
    items: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    planned_total: int,
    gemini_calls: int,
) -> dict[str, Any]:
    return {
        "metadata": {
            "task": "cross_modal_disambiguation_caption",
            "input": input_path.as_posix(),
            "output": output_path.as_posix(),
            "dataset_root": dataset_root.as_posix(),
            "composite_root": composite_root.as_posix(),
            "generation_mode": generation_mode,
            "model_name": model_name,
            "planned_items": planned_total,
            "completed_items": len(items),
            "skipped_items": len(skipped),
            "gemini_calls": gemini_calls,
        },
        "items": items,
        "skipped": skipped,
    }


def _load_resume(output_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not output_path.exists():
        return [], []
    try:
        data = _load_json(output_path)
    except Exception as exc:
        print(f"WARNING: Could not load existing caption output for resume: {exc}")
        return [], []
    items = data.get("items") if isinstance(data.get("items"), list) else []
    skipped = data.get("skipped") if isinstance(data.get("skipped"), list) else []
    
    valid_items = []
    for item in items:
        cap = item.get("caption")
        if not isinstance(cap, dict):
            continue

        try:
            composite_frames = item.get("composite_frames") or []
            valid_frame_keys = {
                Path(frame_name).stem
                for frame_name in composite_frames
                if isinstance(frame_name, str) and frame_name.strip()
            }
            modality1 = str(item.get("modality1") or "")
            modality2 = str(item.get("modality2") or "")

            if not valid_frame_keys or not modality1 or not modality2:
                continue

            _validate_caption_schema(
                copy.deepcopy(cap),
                valid_frame_keys,
                modality1,
                modality2,
            )
        except (CaptionValidationError, CaptionParseError, ValueError, TypeError):
            continue

        valid_items.append(item)

    return valid_items, skipped


async def run_caption_pipeline_async(
    input_path: Path,
    output_path: Path,
    dataset_root: Path,
    composite_root: Path,
    model_name: str,
    generation_mode: str,
    api_key_source: str,
    num_uniform_frames: int,
    num_adaptive_frames: int,
    pairs: str | None,
    directions: str | None,
    sides: str | None,
    limit: int | None,
    limit_scenes: int | None,
    limit_scene_folders: int | None,
    max_retries: int,
    max_transport_retries: int,
    delay_between_calls: int,
    checkpoint_every: int,
    resume: bool,
) -> Path:
    allowed_pairs = _parse_pairs(pairs)
    allowed_directions = _parse_pairs(directions)
    allowed_sides = _parse_sides(sides)
    
    client = create_gemini_client(api_key_source=api_key_source) if generation_mode in ("gemini", "batch") else None
    
    existing_items, existing_skipped = _load_resume(output_path) if resume else ([], [])
    items = existing_items
    
    tasks, scoped_skipped, total_selected_jobs = build_caption_tasks(
        input_path=input_path,
        dataset_root=dataset_root,
        composite_root=composite_root,
        sampling_strategy="uniform_adaptive",
        num_uniform_frames=num_uniform_frames,
        num_adaptive_frames=num_adaptive_frames,
        existing_items=existing_items,
        existing_skipped=existing_skipped,
        allowed_pairs=allowed_pairs,
        allowed_directions=allowed_directions,
        limit=limit,
        limit_scenes=limit_scenes,
        limit_scene_folders=limit_scene_folders,
        allowed_sides=allowed_sides,
        write_composites=False,
    )
    
    seen_skipped = set()
    deduped_skipped = []
    for s in scoped_skipped:
        key = (s.get("segment_id"), s.get("side"), s.get("modality1"), s.get("modality2"), s.get("reason"), s.get("caption_id"))
        if key not in seen_skipped:
            seen_skipped.add(key)
            deduped_skipped.append(s)
    skipped = deduped_skipped
    pending_tasks = tasks # Pending tasks are already filtered inside build_caption_tasks

    # client already created if needed
    api_stats = [0]
    checkpoint_counter = 0

    print(
        f"Generating cross-modal captions: {len(tasks)} planned item(s), "
        f"{len(pending_tasks)} pending, mode={generation_mode}, model={model_name}."
    )
    
    if generation_mode == "batch":
        assert client is not None
        _submit_batch(client, pending_tasks, model_name, output_path, api_key_source, total_selected_jobs)
        return output_path

    def save_checkpoint() -> None:
        _save_json(
            _build_output_payload(
                input_path=input_path,
                dataset_root=dataset_root,
                composite_root=composite_root,
                output_path=output_path,
                model_name=model_name,
                generation_mode=generation_mode,
                items=items,
                skipped=skipped,
                planned_total=total_selected_jobs,
                gemini_calls=api_stats[0],
            ),
            output_path,
        )

    task_index = 0
    MAX_KEY_ROTATIONS = 5
    rotation_attempts = 0

    while task_index < len(pending_tasks):
        task = pending_tasks[task_index]
        print(
            f"  Caption item [{task_index + 1}/{len(pending_tasks)}] "
            f"{task.caption_id}"
        )
        initial_api_stats = api_stats[0]
        try:
            if generation_mode == "gemini":
                assert client is not None
                caption, warnings = await _call_gemini_caption(
                    client,
                    task,
                    model_name,
                    max_retries=max_retries,
                    max_transport_retries=max_transport_retries,
                    api_stats=api_stats,
                )
                attempts_used = api_stats[0] - initial_api_stats
                status = "generated"
                
                skipped[:] = [item for item in skipped if item.get("caption_id") != task.caption_id]
                items[:] = [item for item in items if item.get("caption_id") != task.caption_id]
                items.append(_task_to_item(
                    task, 
                    status=status, 
                    caption=caption,
                    validation_warnings=warnings,
                    attempts=attempts_used,
                    first_attempt_success=(attempts_used == 1),
                    final_error_category=None
                ))
            else:
                _ensure_composite_frames(task)
                valid_frame_keys = {path.stem for path in task.composite_frames}
                caption, warnings = _validate_caption_schema(_template_caption(task), valid_frame_keys, task.modality1, task.modality2)
                status = "template"
                skipped[:] = [item for item in skipped if item.get("caption_id") != task.caption_id]
                items[:] = [item for item in items if item.get("caption_id") != task.caption_id]
                items.append(_task_to_item(task, status=status, caption=caption, validation_warnings=warnings))
            rotation_attempts = 0
        except Exception as exc:
            exc_str = str(exc).lower()
            if "429" in exc_str or "quota" in exc_str:
                rotation_attempts += 1
                if rotation_attempts > MAX_KEY_ROTATIONS:
                    raise RuntimeError("Exceeded maximum API-key rotation attempts")
                print(f"WARNING: Quota exhausted or rate limit hit. Attempting to rotate API key...")
                try:
                    client = create_gemini_client(api_key_source=api_key_source)
                    # Retry the current task seamlessly
                    continue
                except Exception as rotate_exc:
                    print(f"FATAL: All API keys exhausted or failed to rotate: {rotate_exc}")
                    break
            
            # Attempt to determine final error category
            final_error_category = "transport_other"
            exc_str_lower = str(exc).lower()
            if isinstance(exc, CaptionParseError):
                final_error_category = "parse_error"
            elif isinstance(exc, CaptionValidationError):
                if "blocklist" in exc_str_lower or "forbidden" in exc_str_lower:
                    final_error_category = "blocklist_failure"
                elif "missing_key_attributes" in exc_str_lower:
                    final_error_category = "missing_attribute_recovery"
                elif "qa_relevant_details" in exc_str_lower:
                    final_error_category = "qa_mapping_failure"
                elif "reference" in exc_str_lower or "duplicate" in exc_str_lower or "unknown" in exc_str_lower:
                    final_error_category = "invalid_reference"
                else:
                    final_error_category = "schema_validation_error"
            elif _is_transport_error(exc):
                final_error_category = _transport_error_category(exc)
            elif "429" in exc_str_lower or "quota" in exc_str_lower:
                final_error_category = "quota_exhausted"
                
            skipped[:] = [item for item in skipped if item.get("caption_id") != task.caption_id]
            
            skipped.append(
                _task_to_item(
                    task,
                    status="failed",
                    reason=str(exc),
                    attempts=api_stats[0] - initial_api_stats,
                    first_attempt_success=False,
                    final_error_category=final_error_category,
                    last_invalid_response=getattr(exc, "last_invalid_response", None)
                )
            )
            print(f"WARNING: Caption generation failed for {task.caption_id}: {exc}")

        checkpoint_counter += 1
        if checkpoint_every > 0 and checkpoint_counter >= checkpoint_every:
            checkpoint_counter = 0
            save_checkpoint()
        if generation_mode == "gemini" and delay_between_calls > 0 and task_index < len(pending_tasks) - 1:
            await asyncio.sleep(delay_between_calls)
            
        task_index += 1

    save_checkpoint()
    print(f"Wrote cross-modal caption output to {output_path}")
    return output_path


def run_caption_pipeline(
    input_path: Path | str = DEFAULT_INPUT_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    dataset_root: Path | str = DEFAULT_DATASET_ROOT,
    composite_root: Path | str = DEFAULT_COMPOSITE_ROOT,
    model_name: str = DEFAULT_MODEL_NAME,
    generation_mode: str = "template",
    api_key_source: str = "list",
    num_uniform_frames: int = 8,
    num_adaptive_frames: int = 2,
    pairs: str | None = None,
    directions: str | None = None,
    sides: str | None = None,
    limit: int | None = None,
    limit_scenes: int | None = None,
    limit_scene_folders: int | None = None,
    max_retries: int = 3,
    max_transport_retries: int = DEFAULT_MAX_TRANSPORT_RETRIES,
    delay_between_calls: int = 5,
    checkpoint_every: int = 1,
    resume: bool = True,
) -> Path:
    return asyncio.run(
        run_caption_pipeline_async(
            input_path=Path(input_path),
            output_path=Path(output_path),
            dataset_root=Path(dataset_root),
            composite_root=Path(composite_root),
            model_name=model_name,
            generation_mode=generation_mode,
            api_key_source=api_key_source,
            num_uniform_frames=num_uniform_frames,
            num_adaptive_frames=num_adaptive_frames,
            pairs=pairs,
            directions=directions,
            sides=sides,
            limit=limit,
            limit_scenes=limit_scenes,
            limit_scene_folders=limit_scene_folders,
            max_retries=max_retries,
            max_transport_retries=max_transport_retries,
            delay_between_calls=delay_between_calls,
            checkpoint_every=checkpoint_every,
            resume=resume,
        )
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--composite-root", default=str(DEFAULT_COMPOSITE_ROOT))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--generation-mode",
        choices=("template", "gemini", "batch"),
        default="template",
        help="Use template to build composite frames without calling Gemini. Use batch for async batch API.",
    )
    parser.add_argument("--fetch-batch", action="store_true", help="Fetch results for a pending batch job instead of creating tasks.")
    parser.add_argument("--batch-state", default=None, help="Path to batch state file (optional).")
    parser.add_argument("--api-key-source", choices=("env", "list"), default="list", help="Source for Gemini API keys.")
    parser.add_argument("--num-uniform-frames", type=int, default=8)
    parser.add_argument("--num-adaptive-frames", type=int, default=2)
    parser.add_argument(
        "--pairs",
        default=None,
        help="Comma-separated modality pairs such as rgb+depth,rgb+event. Defaults to input modality_pairs.",
    )
    parser.add_argument(
        "--sides",
        default=None,
        help="Comma-separated sides to process, such as day,night,aligned. Defaults to all available.",
    )
    parser.add_argument(
        "--directions",
        default=None,
        help="Comma-separated modality1->modality2 directions such as rgb->depth,event->ir. Defaults to the canonical ordering from input modality_pairs.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--limit-scenes",
        "--limit-segments",
        dest="limit_scenes",
        type=int,
        default=None,
        help="Limit the number of unique segment_id scenes. Unlike --limit, this keeps all matching items in each selected scene.",
    )
    parser.add_argument(
        "--limit-scene-folders",
        "--limit-split-dirs",
        dest="limit_scene_folders",
        type=int,
        default=None,
        help="Limit the number of top-level aligned_dataset scene folders/split_dir values, such as brew_tea_split.",
    )
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum parse/schema validation retries per item.")
    parser.add_argument(
        "--max-transport-retries",
        type=int,
        default=DEFAULT_MAX_TRANSPORT_RETRIES,
        help="Maximum transient transport retries per validation attempt for 503/504/timeouts.",
    )
    parser.add_argument("--delay-between-calls", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    
    return parser


def main() -> None:
    import sys
    import asyncio
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    args = _build_arg_parser().parse_args()
    
    if args.fetch_batch:
        _fetch_batch(None, args.output, args.batch_state)
        return

    run_caption_pipeline(
        input_path=args.input,
        output_path=args.output,
        dataset_root=args.dataset_root,
        composite_root=args.composite_root,
        model_name=args.model_name,
        generation_mode=args.generation_mode,
        api_key_source=args.api_key_source,
        num_uniform_frames=max(1, args.num_uniform_frames),
        num_adaptive_frames=max(0, args.num_adaptive_frames),
        pairs=args.pairs,
        directions=args.directions,
        sides=args.sides,
        limit=args.limit,
        limit_scenes=args.limit_scenes,
        limit_scene_folders=args.limit_scene_folders,
        max_retries=max(1, args.max_retries),
        max_transport_retries=max(1, args.max_transport_retries),
        delay_between_calls=max(0, args.delay_between_calls),
        checkpoint_every=max(0, args.checkpoint_every),
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
