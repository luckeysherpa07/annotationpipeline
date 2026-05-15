"""Run modality QA pipelines on semantic task segments."""

from __future__ import annotations

import asyncio
import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from annotation_feature.pipeline.client import create_gemini_client
from annotation_feature.pipeline.modalities.audio.pipeline import (
    DEMO_HIA_CAPTION,
    DEMO_QA_PAIRS,
    DEMO_TIMESTAMPED_CAPTION,
    _guess_mime_type,
    _upload_file_part,
    call_gemini_with_retry as call_audio_gemini_with_retry,
    format_audio_annotations,
    generate_hia_caption,
    generate_qa_pairs,
    parse_caption_response,
)
from annotation_feature.pipeline.modalities.depth import normalize_depth_results, run_depth_parallel_pipeline
from annotation_feature.pipeline.modalities.depth.pipeline import DEPTH_PROMPTS
from annotation_feature.pipeline.modalities.event import normalize_event_results, run_event_parallel_pipeline
from annotation_feature.pipeline.modalities.event.pipeline import EVENT_PROMPTS
from annotation_feature.pipeline.modalities.ir import normalize_ir_results, run_ir_parallel_pipeline
from annotation_feature.pipeline.modalities.ir.pipeline import IR_PROMPTS
from annotation_feature.pipeline.modalities.rgb import (
    normalize_annotation_results as normalize_rgb_results,
    run_parallel_pipeline as run_rgb_parallel_pipeline,
)
from annotation_feature.pipeline.modalities.rgb.pipeline import (
    DEMO_RESULT as RGB_DEMO_RESULT,
    RGB_PROMPTS,
)
from annotation_feature.pipeline.utils import build_image_parts, encode_frames_to_base64


MODALITY_CACHE_DIRS = {
    "rgb": ".frames_cache",
    "event": ".frames_cache_event",
    "depth": ".frames_cache_depth",
    "ir": ".frames_cache_ir",
}

SEGMENTED_OUTPUT_FILES = {
    "rgb": "segmented_rgb_qa_results.json",
    "event": "segmented_event_qa_results.json",
    "depth": "segmented_depth_qa_results.json",
    "ir": "segmented_ir_qa_results.json",
    "audio": "segmented_audio_qa_results.json",
}
SEGMENTED_MODALITIES = tuple(SEGMENTED_OUTPUT_FILES.keys())
IMAGE_MODALITIES = ("rgb", "event", "depth", "ir")
DEFAULT_SEGMENTED_API_DELAY_SECONDS = 12
DEFAULT_SEGMENTED_MAX_CONCURRENT = 1
SEGMENTED_MODEL_NAME = "gemini-3-flash-preview"


IMAGE_MODALITY_CONFIGS = {
    "rgb": {
        "prompts": RGB_PROMPTS,
        "normalizer": normalize_rgb_results,
    },
    "event": {
        "prompts": EVENT_PROMPTS,
        "normalizer": normalize_event_results,
    },
    "depth": {
        "prompts": DEPTH_PROMPTS,
        "normalizer": normalize_depth_results,
    },
    "ir": {
        "prompts": IR_PROMPTS,
        "normalizer": normalize_ir_results,
    },
}


@dataclass
class SegmentJob:
    key: str
    manifest_path: Path
    source_prefix: str
    split_dir: Path
    side: str
    segment: dict[str, Any]
    source_files: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object in {path}")
    return data


def _load_segment_jobs(
    dataset_folder: Path,
    task_segments_folder: Path,
    test_mode: bool = False,
) -> list[SegmentJob]:
    jobs: list[SegmentJob] = []
    manifest_paths = sorted(task_segments_folder.rglob("*_task_segments.json")) if task_segments_folder.exists() else []

    for manifest_path in manifest_paths:
        manifest = _load_json(manifest_path)
        source_prefix = str(manifest.get("source_prefix") or manifest_path.name.replace("_task_segments.json", ""))
        split_dir_name = str(manifest.get("split_dir") or manifest_path.parent.name)
        source_split_dir = dataset_folder / split_dir_name
        side = str(manifest.get("side") or "unknown")
        source_files = manifest.get("source_files", {})
        if not isinstance(source_files, dict):
            source_files = {}

        segments = manifest.get("segments", [])
        if not isinstance(segments, list):
            print(f"WARNING: Skipping {manifest_path}: segments must be a list")
            continue

        for segment in segments:
            if not isinstance(segment, dict):
                continue
            segment_id = str(segment.get("segment_id") or "")
            if not segment_id:
                print(f"WARNING: Segment without segment_id in {manifest_path}; skipping")
                continue

            key = str(source_split_dir / segment_id)
            jobs.append(
                SegmentJob(
                    key=key,
                    manifest_path=manifest_path,
                    source_prefix=source_prefix,
                    split_dir=source_split_dir,
                    side=side,
                    segment=segment,
                    source_files=source_files,
                )
            )

    if test_mode:
        return jobs[:1]
    return jobs


def _get_video_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 30.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps if fps > 0 else 30.0


def _frame_index_from_path(frame_path: Path) -> int | None:
    match = re.search(r"frame_(\d+)", frame_path.name)
    if not match:
        return None
    return int(match.group(1))


def _segment_time_bounds(segment: dict[str, Any]) -> tuple[float, float]:
    return float(segment.get("start_seconds", 0.0)), float(segment.get("end_seconds", 0.0))


def _resolve_source_video(job: SegmentJob, modality: str) -> Path | None:
    videos = job.source_files.get("videos", {})
    if not isinstance(videos, dict):
        return None
    raw_path = videos.get(modality)
    if not raw_path:
        return None
    path = Path(raw_path)
    return path if path.exists() else None


def _infer_marigold_pair_and_side(job: SegmentJob) -> tuple[str, str]:
    prefix = job.source_prefix.lower()
    side = job.side.lower()
    if side not in {"day", "night"}:
        if re.search(r"_day(?:_|$)", prefix):
            side = "day"
        elif re.search(r"_night\d*(?:_|$)", prefix):
            side = "night"
    if side not in {"day", "night"}:
        side = "night" if "night" in prefix else "day"

    pair_key = re.sub(r"_(day|night\d*)$", "", prefix)
    pair_key = re.sub(r"_(rgb|depth|event|ir)$", "", pair_key)
    pair_key = re.sub(r"__+", "_", pair_key).strip("_")
    return pair_key, side


def _load_marigold_depth_frames(dataset_folder: Path, job: SegmentJob) -> list[Path]:
    pair_key, side = _infer_marigold_pair_and_side(job)
    cache_dir = dataset_folder / ".frames_cache_marigold" / pair_key / side
    cached_frames = sorted(cache_dir.glob("frame_*_depth.png"))
    if not cached_frames:
        print(
            f"WARNING: No Marigold depth frames found for {job.key}: {cache_dir}. "
            "Run option 19 first to generate dataset/.frames_cache_marigold."
        )
    return cached_frames


def _load_segment_frames(dataset_folder: Path, job: SegmentJob, modality: str) -> list[Path]:
    video_path = _resolve_source_video(job, modality)
    if video_path is None:
        print(f"WARNING: Missing {modality} source video for {job.key}")
        return []

    if modality == "depth":
        cached_frames = _load_marigold_depth_frames(dataset_folder, job)
    else:
        cache_subdir = MODALITY_CACHE_DIRS[modality]
        cache_dir = dataset_folder / cache_subdir / video_path.stem
        cached_frames = sorted(cache_dir.glob("frame_*.png"))
    if not cached_frames:
        if modality != "depth":
            print(f"WARNING: No cached {modality} frames found for {job.key}: {cache_dir}")
        return []

    fps = _get_video_fps(video_path)
    start_seconds, end_seconds = _segment_time_bounds(job.segment)
    selected_frames: list[Path] = []
    for frame_path in cached_frames:
        frame_index = _frame_index_from_path(frame_path)
        if frame_index is None:
            continue
        frame_time = frame_index / fps
        if start_seconds <= frame_time < end_seconds:
            selected_frames.append(frame_path)

    if not selected_frames:
        print(
            f"WARNING: No {modality} frames inside {job.segment.get('segment_id')} "
            f"({start_seconds:.3f}-{end_seconds:.3f}s)"
        )
    return selected_frames


def _segment_payload(job: SegmentJob) -> dict[str, Any]:
    return {
        "segment_id": job.segment.get("segment_id"),
        "task_label": job.segment.get("task_label"),
        "start_seconds": job.segment.get("start_seconds"),
        "end_seconds": job.segment.get("end_seconds"),
        "start_timestamp": job.segment.get("start_timestamp"),
        "end_timestamp": job.segment.get("end_timestamp"),
    }


def _wrap_modality_results(
    jobs: list[SegmentJob],
    modality_results: dict[str, dict],
    modality: str,
) -> dict[str, dict[str, Any]]:
    wrapped: dict[str, dict[str, Any]] = {}
    for job in jobs:
        video_path = _resolve_source_video(job, modality)
        wrapped[job.key] = {
            "source_prefix": job.source_prefix,
            "side": job.side,
            "segment": _segment_payload(job),
            "source_file": str(video_path) if video_path else None,
            "annotations": modality_results.get(job.key, {}),
        }
    return wrapped


def _build_segmented_frame_pairs(
    dataset_folder: Path,
    jobs: list[SegmentJob],
    modality: str,
) -> tuple[dict[str, dict[str, list[Path]]], list[SegmentJob]]:
    paired_frames: dict[str, dict[str, list[Path]]] = {}
    runnable_jobs: list[SegmentJob] = []

    for job in jobs:
        frames = _load_segment_frames(dataset_folder, job, modality)
        if not frames:
            continue

        # Existing modality processors expect both day and night frame lists.
        # Segment manifests are per recording side, so use the same segment frames
        # for both roles while preserving the original side in output metadata.
        paired_frames[job.key] = {"day": frames, "night": frames}
        runnable_jobs.append(job)

    return paired_frames, runnable_jobs


def _save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def _load_existing_results(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = _load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"WARNING: Could not load existing segmented results from {path}: {exc}")
        return {}
    return data


def _has_text(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return any(_has_text(child) for child in value.values())
    if isinstance(value, list):
        return any(_has_text(child) for child in value)
    return value is not None


def _looks_like_demo_annotations(modality: str, annotations: Any) -> bool:
    if not isinstance(annotations, dict):
        return False
    if modality == "rgb" and annotations == RGB_DEMO_RESULT:
        return True

    text_values: list[str] = []

    def collect_text(value: Any) -> None:
        if isinstance(value, str):
            text_values.append(value.strip())
        elif isinstance(value, dict):
            for child in value.values():
                collect_text(child)
        elif isinstance(value, list):
            for child in value:
                collect_text(child)

    collect_text(annotations)
    non_empty_values = [value for value in text_values if value]
    if not non_empty_values:
        return False
    return all(value in {"Demo caption", "Demo question?", "Demo answer"} for value in non_empty_values)


def _has_complete_annotations(
    existing_item: Any,
    modality: str,
    skip_api: bool,
) -> bool:
    if not isinstance(existing_item, dict):
        return False
    annotations = existing_item.get("annotations")
    if not _has_text(annotations):
        return False
    if not skip_api and _looks_like_demo_annotations(modality, annotations):
        return False
    return True


def _merge_results(
    existing_results: dict[str, Any],
    new_results: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(existing_results)
    merged.update(new_results)
    return merged


def _normalize_modalities(modalities: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if modalities is None:
        return SEGMENTED_MODALITIES

    normalized: list[str] = []
    for modality in modalities:
        value = str(modality).strip().lower()
        if not value:
            continue
        if value == "all":
            return SEGMENTED_MODALITIES
        if value not in SEGMENTED_OUTPUT_FILES:
            valid = ", ".join(SEGMENTED_MODALITIES)
            raise ValueError(f"Unsupported segmented modality: {modality}. Expected one of: {valid}")
        if value not in normalized:
            normalized.append(value)

    if not normalized:
        valid = ", ".join(SEGMENTED_MODALITIES)
        raise ValueError(f"No segmented modalities selected. Expected one of: {valid}")
    return tuple(normalized)


def _is_quota_or_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(token in text for token in ("quota", "rate limit", "rate_limit", "429", "resource_exhausted"))


async def _call_segmented_gemini_with_retry(client, contents: list, max_retries: int = 3) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=SEGMENTED_MODEL_NAME,
                contents=contents,
            )
            return response.text
        except Exception as exc:
            if attempt == max_retries:
                raise
            wait_seconds = 30 * attempt if _is_quota_or_rate_limit_error(exc) else 2 * attempt
            print(
                f"    Segmented Gemini call failed on attempt {attempt}/{max_retries}; "
                f"retrying in {wait_seconds}s: {exc}"
            )
            await asyncio.sleep(wait_seconds)
    raise RuntimeError("Segmented Gemini call failed")


def _parse_json_object_response(text: str) -> dict[str, Any]:
    if not text:
        raise ValueError("Empty Gemini response")
    cleaned_text = text.strip()
    cleaned_text = re.sub(r"^```(?:json)?\s*", "", cleaned_text, flags=re.I)
    cleaned_text = re.sub(r"\s*```$", "", cleaned_text, flags=re.I)
    match = re.search(r"\{.*\}", cleaned_text, flags=re.S)
    if not match:
        raise ValueError("No JSON object found in Gemini response")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Gemini response must be a JSON object")
    return parsed


def _group_jobs_by_manifest(jobs: list[SegmentJob]) -> list[list[SegmentJob]]:
    grouped: dict[Path, list[SegmentJob]] = {}
    for job in jobs:
        grouped.setdefault(job.manifest_path, []).append(job)
    return [grouped[path] for path in sorted(grouped)]


def _source_cache_frames(dataset_folder: Path, job: SegmentJob, modality: str) -> list[Path]:
    video_path = _resolve_source_video(job, modality)
    if video_path is None:
        return []
    if modality == "depth":
        return _load_marigold_depth_frames(dataset_folder, job)
    cache_dir = dataset_folder / MODALITY_CACHE_DIRS[modality] / video_path.stem
    return sorted(cache_dir.glob("frame_*.png"))


def _build_segments_for_prompt(jobs: list[SegmentJob]) -> list[dict[str, Any]]:
    return [
        {
            "segment_id": str(job.segment.get("segment_id") or job.key),
            "task_label": job.segment.get("task_label"),
            "start_seconds": job.segment.get("start_seconds"),
            "end_seconds": job.segment.get("end_seconds"),
            "start_timestamp": job.segment.get("start_timestamp"),
            "end_timestamp": job.segment.get("end_timestamp"),
            "notes": job.segment.get("notes"),
        }
        for job in jobs
    ]


def _build_batched_image_prompt(modality: str, jobs: list[SegmentJob], frames: list[Path]) -> str:
    config = IMAGE_MODALITY_CONFIGS[modality]
    prompts = config["prompts"]
    prompt_parts = [
        f"You are a segmented {modality.upper()} video QA assistant.",
        "You will receive frames from one source recording and a list of semantic task segments.",
        "For each segment, analyze only frames whose frame index/time belongs to that segment.",
        "Generate the same annotation sections used by the existing per-segment pipeline.",
        "Return ONLY valid JSON with this exact structure:",
        "{",
        '  "segment_id": {',
        '    "annotation_type": {"caption": "...", "question": "...", "answer": "..."}',
        "  }",
        "}",
        "Use the segment_id values exactly as provided.",
        "Do not include markdown, explanation, or extra keys.",
        "",
        f"Source prefix: {jobs[0].source_prefix}",
        f"Side: {jobs[0].side}",
        "Segments:",
        json.dumps(_build_segments_for_prompt(jobs), indent=2),
        "",
        f"Frames ({len(frames)} images): {', '.join(path.name for path in frames)}",
        "",
        "Annotation prompts:",
    ]

    for annotation_type in prompts:
        prompt_parts.extend(
            [
                f"### {annotation_type}",
                "CAPTION PROMPT:",
                prompts[annotation_type]["caption_prompt"],
                "",
                "QUESTION PROMPT:",
                prompts[annotation_type]["question_prompt"],
                "",
                "ANSWERING PROMPT:",
                prompts[annotation_type]["answering_prompt"],
                "",
            ]
        )

    prompt_parts.append("Produce one JSON object keyed by segment_id.")
    return "\n".join(prompt_parts)


def _normalize_batched_image_response(modality: str, jobs: list[SegmentJob], parsed: dict[str, Any]) -> dict[str, dict]:
    normalizer = IMAGE_MODALITY_CONFIGS[modality]["normalizer"]
    normalized_by_job_key: dict[str, dict] = {}
    for job in jobs:
        segment_id = str(job.segment.get("segment_id") or job.key)
        raw_segment = parsed.get(segment_id)
        if not isinstance(raw_segment, dict):
            raw_segment = parsed.get(job.key, {})
        normalized_by_job_key[job.key] = normalizer(raw_segment if isinstance(raw_segment, dict) else {})
    return normalized_by_job_key


async def _run_batched_image_modality(
    modality: str,
    client,
    dataset_folder: Path,
    jobs: list[SegmentJob],
    skip_api: bool,
    delay_between_batches: int,
) -> dict[str, dict[str, Any]]:
    grouped_jobs = _group_jobs_by_manifest(jobs)
    print(
        f"Running batched segmented {modality.upper()} QA for "
        f"{len(jobs)} segment(s) in {len(grouped_jobs)} source batch(es)..."
    )
    modality_results: dict[str, dict] = {}
    runnable_jobs: list[SegmentJob] = []

    for index, group in enumerate(grouped_jobs, start=1):
        if skip_api:
            if modality == "rgb":
                group_results = {job.key: copy.deepcopy(RGB_DEMO_RESULT) for job in group}
            else:
                normalizer = IMAGE_MODALITY_CONFIGS[modality]["normalizer"]
                group_results = {job.key: normalizer({}) for job in group}
            modality_results.update(group_results)
            runnable_jobs.extend(group)
            continue

        frames = _source_cache_frames(dataset_folder, group[0], modality)
        if not frames:
            print(f"WARNING: No cached source {modality} frames found for {group[0].source_prefix}")
            continue

        encoded_frames = encode_frames_to_base64(frames)
        if not encoded_frames:
            print(f"WARNING: Could not encode source {modality} frames for {group[0].source_prefix}")
            continue

        print(
            f"Running batched {modality.upper()} source batch "
            f"[{index}/{len(grouped_jobs)}]: {group[0].manifest_path}"
        )
        contents = build_image_parts(encoded_frames) + [_build_batched_image_prompt(modality, group, frames)]
        try:
            response_text = await _call_segmented_gemini_with_retry(client, contents)
            parsed = _parse_json_object_response(response_text)
            modality_results.update(_normalize_batched_image_response(modality, group, parsed))
            runnable_jobs.extend(group)
        except Exception as exc:
            print(f"WARNING: Batched segmented {modality.upper()} QA failed for {group[0].manifest_path}: {exc}")

        if not skip_api and index < len(grouped_jobs) and delay_between_batches > 0:
            await asyncio.sleep(delay_between_batches)

    wrapped = _wrap_modality_results(jobs, {}, modality)
    wrapped.update(_wrap_modality_results(runnable_jobs, modality_results, modality))
    return wrapped


async def _run_image_modality(
    modality: str,
    client,
    dataset_folder: Path,
    jobs: list[SegmentJob],
    skip_api: bool,
    delay_between_segments: int,
    max_concurrent: int,
) -> dict[str, dict[str, Any]]:
    paired_frames, runnable_jobs = _build_segmented_frame_pairs(dataset_folder, jobs, modality)
    if not paired_frames:
        print(f"WARNING: No runnable {modality} segments found.")
        return _wrap_modality_results(jobs, {}, modality)

    print(f"Running segmented {modality.upper()} QA for {len(paired_frames)} segment(s)...")
    if modality == "rgb":
        results = await run_rgb_parallel_pipeline(
            client,
            paired_frames,
            max_concurrent=max_concurrent,
            delay_between_pairs=delay_between_segments,
            skip_api=skip_api,
        )
    elif modality == "event":
        results = await run_event_parallel_pipeline(
            client,
            paired_frames,
            max_concurrent=max_concurrent,
            delay_between_pairs=delay_between_segments,
            skip_api=skip_api,
        )
    elif modality == "depth":
        results = await run_depth_parallel_pipeline(
            client,
            paired_frames,
            max_concurrent=max_concurrent,
            delay_between_pairs=delay_between_segments,
            skip_api=skip_api,
        )
    elif modality == "ir":
        results = await run_ir_parallel_pipeline(
            client,
            paired_frames,
            max_concurrent=max_concurrent,
            delay_between_pairs=delay_between_segments,
            skip_api=skip_api,
        )
    else:
        raise ValueError(f"Unsupported image modality: {modality}")

    wrapped = _wrap_modality_results(jobs, {}, modality)
    wrapped.update(_wrap_modality_results(runnable_jobs, results, modality))
    return wrapped


def _segment_audio_prompt(hia_caption: str, job: SegmentJob) -> str:
    start_seconds, end_seconds = _segment_time_bounds(job.segment)
    task_label = job.segment.get("task_label", "")
    return "\n".join(
        [
            "You are analyzing one semantic task segment from a longer egocentric audio/video recording.",
            f"Only describe events between {start_seconds:.3f} and {end_seconds:.3f} seconds.",
            f"Task label: {task_label}",
            "Ignore sounds and visual events outside this time window.",
            "",
            "Human Interaction Annotations (HIA):",
            hia_caption,
            "",
            "Return only a timestamped audio-visual caption for this segment.",
        ]
    )


async def _generate_segment_audio_annotations(client, job: SegmentJob, rgb_frames: list[Path], skip_api: bool) -> dict:
    if skip_api:
        return format_audio_annotations(
            {
                "hia": DEMO_HIA_CAPTION,
                "caption": DEMO_TIMESTAMPED_CAPTION,
                "qa_pairs": DEMO_QA_PAIRS,
            }
        )

    media_path = job.source_files.get("with_audio") or job.source_files.get("audio")
    media_file = Path(media_path) if media_path else None
    if media_file is None or not media_file.exists():
        print(f"WARNING: Missing audio media for {job.key}")
        return {"categories": {}}

    try:
        hia_caption = await generate_hia_caption(client, rgb_frames, skip_api=False)
        media_part = await _upload_file_part(client, media_file, _guess_mime_type(media_file))
        response_text = await call_audio_gemini_with_retry(
            client,
            [media_part, _segment_audio_prompt(hia_caption, job)],
        )
        timestamped_caption = parse_caption_response(response_text)
        qa_pairs = await generate_qa_pairs(client, timestamped_caption, skip_api=False)
        return format_audio_annotations(
            {
                "hia": hia_caption,
                "caption": timestamped_caption,
                "qa_pairs": qa_pairs,
            }
        )
    except Exception as exc:
        print(f"WARNING: Segmented audio QA failed for {job.key}: {exc}")
        return {"categories": {}}


def _build_batched_audio_prompt(jobs: list[SegmentJob]) -> str:
    return "\n".join(
        [
            "You are analyzing semantic task segments from one egocentric audio/video recording.",
            "Use the provided media to produce audio-visual QA annotations for each segment.",
            "Analyze only the time window for each segment and keep segment outputs independent.",
            "For each segment, include:",
            "- audio_hia: a concise human-object interaction caption for that segment",
            "- audio_chronological_caption: a timestamped audio-visual caption for that segment",
            "- categories: sound-centric QA sections keyed by category names",
            "",
            "Return ONLY valid JSON with this exact structure:",
            "{",
            '  "segment_id": {',
            '    "audio_hia": {"caption": "..."},',
            '    "audio_chronological_caption": {"caption": "..."},',
            '    "categories": {',
            '      "audio_sound_source_identification": {',
            '        "timestamp": "00:00 - 00:05",',
            '        "caption": "...",',
            '        "question": "...",',
            '        "answer": "..."',
            "      }",
            "    }",
            "  }",
            "}",
            "Use the segment_id values exactly as provided. Do not include markdown or explanations.",
            "",
            f"Source prefix: {jobs[0].source_prefix}",
            f"Side: {jobs[0].side}",
            "Segments:",
            json.dumps(_build_segments_for_prompt(jobs), indent=2),
        ]
    )


def _normalize_batched_audio_annotation(raw_annotation: Any) -> dict[str, Any]:
    if not isinstance(raw_annotation, dict):
        return {"categories": {}}

    if "qa_pairs" in raw_annotation or "caption" in raw_annotation or "hia" in raw_annotation:
        return format_audio_annotations(raw_annotation)

    formatted: dict[str, Any] = {}
    hia = raw_annotation.get("audio_hia")
    if isinstance(hia, dict):
        formatted["audio_hia"] = {"caption": str(hia.get("caption", "") or "")}
    elif isinstance(hia, str) and hia.strip():
        formatted["audio_hia"] = {"caption": hia}

    chronological = raw_annotation.get("audio_chronological_caption")
    if isinstance(chronological, dict):
        formatted["audio_chronological_caption"] = {
            "caption": str(chronological.get("caption", "") or "")
        }
    elif isinstance(chronological, str) and chronological.strip():
        formatted["audio_chronological_caption"] = {"caption": chronological}

    categories = raw_annotation.get("categories", {})
    formatted_categories: dict[str, dict[str, Any]] = {}
    if isinstance(categories, dict):
        for category_name, category_data in categories.items():
            if not isinstance(category_data, dict):
                continue
            formatted_categories[str(category_name)] = {
                "timestamp": category_data.get("timestamp"),
                "caption": str(category_data.get("caption", "") or ""),
                "question": str(category_data.get("question", "") or ""),
                "answer": str(category_data.get("answer", "") or ""),
            }
    formatted["categories"] = formatted_categories
    return formatted


async def _run_batched_audio_modality(
    client,
    jobs: list[SegmentJob],
    skip_api: bool,
    delay_between_batches: int,
) -> dict[str, dict[str, Any]]:
    grouped_jobs = _group_jobs_by_manifest(jobs)
    print(
        f"Running batched segmented AUDIO QA for "
        f"{len(jobs)} segment(s) in {len(grouped_jobs)} source batch(es)..."
    )

    wrapped: dict[str, dict[str, Any]] = {}
    for index, group in enumerate(grouped_jobs, start=1):
        if skip_api:
            parsed = {
                str(job.segment.get("segment_id") or job.key): format_audio_annotations(
                    {
                        "hia": DEMO_HIA_CAPTION,
                        "caption": DEMO_TIMESTAMPED_CAPTION,
                        "qa_pairs": DEMO_QA_PAIRS,
                    }
                )
                for job in group
            }
        else:
            media_path = group[0].source_files.get("with_audio") or group[0].source_files.get("audio")
            media_file = Path(media_path) if media_path else None
            if media_file is None or not media_file.exists():
                print(f"WARNING: Missing audio media for {group[0].manifest_path}")
                parsed = {}
            else:
                print(f"Running batched AUDIO source batch [{index}/{len(grouped_jobs)}]: {group[0].manifest_path}")
                try:
                    media_part = await _upload_file_part(client, media_file, _guess_mime_type(media_file))
                    response_text = await _call_segmented_gemini_with_retry(
                        client,
                        [media_part, _build_batched_audio_prompt(group)],
                    )
                    parsed = _parse_json_object_response(response_text)
                except Exception as exc:
                    print(f"WARNING: Batched segmented AUDIO QA failed for {group[0].manifest_path}: {exc}")
                    parsed = {}

        for job in group:
            segment_id = str(job.segment.get("segment_id") or job.key)
            annotations = _normalize_batched_audio_annotation(parsed.get(segment_id, {}))
            wrapped[job.key] = {
                "source_prefix": job.source_prefix,
                "side": job.side,
                "segment": _segment_payload(job),
                "source_file": job.source_files.get("with_audio") or job.source_files.get("audio"),
                "annotations": annotations,
            }

        if not skip_api and index < len(grouped_jobs) and delay_between_batches > 0:
            await asyncio.sleep(delay_between_batches)

    return wrapped


async def _run_audio_modality(
    client,
    dataset_folder: Path,
    jobs: list[SegmentJob],
    skip_api: bool,
    delay_between_segments: int,
) -> dict[str, dict[str, Any]]:
    wrapped: dict[str, dict[str, Any]] = {}
    for index, job in enumerate(jobs, start=1):
        print(f"Running segmented AUDIO QA [{index}/{len(jobs)}]: {job.key}")
        rgb_frames = _load_segment_frames(dataset_folder, job, "rgb")
        annotations = await _generate_segment_audio_annotations(client, job, rgb_frames, skip_api=skip_api)
        wrapped[job.key] = {
            "source_prefix": job.source_prefix,
            "side": job.side,
            "segment": _segment_payload(job),
            "source_file": job.source_files.get("with_audio") or job.source_files.get("audio"),
            "annotations": annotations,
        }
        if not skip_api and index < len(jobs) and delay_between_segments > 0:
            await asyncio.sleep(delay_between_segments)
    return wrapped


async def _run_segmented_pipeline_async(
    dataset_folder: Path,
    task_segments_folder: Path,
    output_folder: Path,
    test_mode: bool,
    skip_api: bool,
    delay_between_segments: int,
    modalities: tuple[str, ...],
    resume: bool,
    max_concurrent: int,
    batch_segments: bool,
    batch_audio: bool,
) -> dict[str, Path]:
    jobs = _load_segment_jobs(dataset_folder, task_segments_folder, test_mode=test_mode)
    if not jobs:
        print(f"ERROR: No task segment manifests found in {task_segments_folder}. Run task slicing first.")
        return {}

    print(f"Found {len(jobs)} task segment(s).")
    client = None if skip_api else create_gemini_client()

    output_folder.mkdir(parents=True, exist_ok=True)
    written_paths: dict[str, Path] = {}

    print(f"Selected segmented QA modalities: {', '.join(modalities)}")

    for modality in IMAGE_MODALITIES:
        if modality not in modalities:
            continue
        output_path = output_folder / SEGMENTED_OUTPUT_FILES[modality]
        existing_results = _load_existing_results(output_path) if resume else {}
        jobs_to_run = [
            job
            for job in jobs
            if not _has_complete_annotations(existing_results.get(job.key), modality, skip_api=skip_api)
        ]
        skipped_count = len(jobs) - len(jobs_to_run)
        if skipped_count:
            print(f"Resuming segmented {modality.upper()}: skipping {skipped_count} completed segment(s).")
        if not jobs_to_run:
            print(f"All segmented {modality.upper()} results already complete.")
            written_paths[modality] = output_path
            continue
        if batch_segments:
            results = await _run_batched_image_modality(
                modality=modality,
                client=client,
                dataset_folder=dataset_folder,
                jobs=jobs_to_run,
                skip_api=skip_api,
                delay_between_batches=delay_between_segments,
            )
        else:
            results = await _run_image_modality(
                modality=modality,
                client=client,
                dataset_folder=dataset_folder,
                jobs=jobs_to_run,
                skip_api=skip_api,
                delay_between_segments=delay_between_segments,
                max_concurrent=max_concurrent,
            )
        results = _merge_results(existing_results, results)
        _save_json(results, output_path)
        written_paths[modality] = output_path
        print(f"Wrote {len(results)} segmented {modality} sample(s) to {output_path}")

    if "audio" in modalities:
        audio_output_path = output_folder / SEGMENTED_OUTPUT_FILES["audio"]
        existing_audio_results = _load_existing_results(audio_output_path) if resume else {}
        audio_jobs = [
            job
            for job in jobs
            if not _has_complete_annotations(existing_audio_results.get(job.key), "audio", skip_api=skip_api)
        ]
        skipped_count = len(jobs) - len(audio_jobs)
        if skipped_count:
            print(f"Resuming segmented AUDIO: skipping {skipped_count} completed segment(s).")
        if not audio_jobs:
            print("All segmented AUDIO results already complete.")
            written_paths["audio"] = audio_output_path
            return written_paths
        if batch_audio:
            audio_results = await _run_batched_audio_modality(
                client=client,
                jobs=audio_jobs,
                skip_api=skip_api,
                delay_between_batches=delay_between_segments,
            )
        else:
            audio_results = await _run_audio_modality(
                client=client,
                dataset_folder=dataset_folder,
                jobs=audio_jobs,
                skip_api=skip_api,
                delay_between_segments=delay_between_segments,
            )
        audio_results = _merge_results(existing_audio_results, audio_results)
        _save_json(audio_results, audio_output_path)
        written_paths["audio"] = audio_output_path
        print(f"Wrote {len(audio_results)} segmented audio sample(s) to {audio_output_path}")

    return written_paths


def run_segmented_pipeline(
    dataset_folder: Path | str = "dataset",
    output_folder: Path | str = "segmented_outputs",
    task_segments_folder: Path | str | None = None,
    test_mode: bool = False,
    skip_api: bool = False,
    delay_between_segments: int | None = None,
    modalities: list[str] | tuple[str, ...] | None = None,
    resume: bool = True,
    max_concurrent: int = DEFAULT_SEGMENTED_MAX_CONCURRENT,
    batch_segments: bool = True,
    batch_audio: bool = True,
) -> dict[str, Path]:
    """Run selected QA modalities over task segments."""
    dataset_folder = Path(dataset_folder)
    output_folder = Path(output_folder)
    task_segments_folder = Path(task_segments_folder) if task_segments_folder is not None else output_folder / "dataset"
    if not dataset_folder.exists():
        print(f"ERROR: Dataset folder not found: {dataset_folder}")
        return {}

    selected_modalities = _normalize_modalities(modalities)

    resolved_delay = 0 if skip_api else DEFAULT_SEGMENTED_API_DELAY_SECONDS
    if delay_between_segments is not None:
        resolved_delay = delay_between_segments

    return asyncio.run(
        _run_segmented_pipeline_async(
            dataset_folder=dataset_folder,
            task_segments_folder=task_segments_folder,
            output_folder=output_folder,
            test_mode=test_mode,
            skip_api=skip_api,
            delay_between_segments=resolved_delay,
            modalities=selected_modalities,
            resume=resume,
            max_concurrent=max_concurrent,
            batch_segments=batch_segments,
            batch_audio=batch_audio,
        )
    )


def estimate_segmented_work(
    dataset_folder: Path | str = "dataset",
    output_folder: Path | str = "segmented_outputs",
    task_segments_folder: Path | str | None = None,
    modalities: list[str] | tuple[str, ...] | None = None,
    skip_api: bool = False,
    resume: bool = True,
    batch_segments: bool = True,
    batch_audio: bool = True,
) -> dict[str, Any]:
    """Estimate segmented QA work and Gemini calls for a menu preview."""
    dataset_folder = Path(dataset_folder)
    output_folder = Path(output_folder)
    task_segments_folder = Path(task_segments_folder) if task_segments_folder is not None else output_folder / "dataset"
    selected_modalities = _normalize_modalities(modalities)
    jobs = (
        _load_segment_jobs(dataset_folder, task_segments_folder, test_mode=False)
        if dataset_folder.exists()
        else []
    )

    modality_counts: dict[str, dict[str, int]] = {}
    estimated_calls = 0
    for modality in selected_modalities:
        output_path = output_folder / SEGMENTED_OUTPUT_FILES[modality]
        existing_results = _load_existing_results(output_path) if resume else {}
        pending = [
            job
            for job in jobs
            if not _has_complete_annotations(existing_results.get(job.key), modality, skip_api=skip_api)
        ]
        pending_count = len(pending)
        source_batches = len(_group_jobs_by_manifest(pending))
        if modality == "audio":
            modality_calls = source_batches if batch_audio else pending_count * 3
        else:
            modality_calls = source_batches if batch_segments else pending_count
        estimated_calls += modality_calls
        modality_counts[modality] = {
            "total_segments": len(jobs),
            "pending_segments": pending_count,
            "skipped_segments": len(jobs) - pending_count,
            "source_batches": source_batches,
            "estimated_calls": modality_calls,
        }

    return {
        "total_segments": len(jobs),
        "modalities": selected_modalities,
        "modality_counts": modality_counts,
        "estimated_gemini_calls": estimated_calls,
        "batch_segments": batch_segments,
        "batch_audio": batch_audio,
    }
