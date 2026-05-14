"""Run modality QA pipelines on semantic task segments."""

from __future__ import annotations

import asyncio
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
from annotation_feature.pipeline.modalities.depth import run_depth_parallel_pipeline
from annotation_feature.pipeline.modalities.event import run_event_parallel_pipeline
from annotation_feature.pipeline.modalities.ir import run_ir_parallel_pipeline
from annotation_feature.pipeline.modalities.rgb import run_parallel_pipeline as run_rgb_parallel_pipeline


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


@dataclass
class SegmentJob:
    key: str
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


def _load_segment_jobs(dataset_folder: Path, test_mode: bool = False) -> list[SegmentJob]:
    jobs: list[SegmentJob] = []
    manifest_paths = sorted(dataset_folder.rglob("*_task_segments.json"))

    for manifest_path in manifest_paths:
        manifest = _load_json(manifest_path)
        source_prefix = str(manifest.get("source_prefix") or manifest_path.name.replace("_task_segments.json", ""))
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

            key = str(manifest_path.parent / segment_id)
            jobs.append(
                SegmentJob(
                    key=key,
                    source_prefix=source_prefix,
                    split_dir=manifest_path.parent,
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


def _load_segment_frames(dataset_folder: Path, job: SegmentJob, modality: str) -> list[Path]:
    video_path = _resolve_source_video(job, modality)
    if video_path is None:
        print(f"WARNING: Missing {modality} source video for {job.key}")
        return []

    cache_subdir = MODALITY_CACHE_DIRS[modality]
    cache_dir = dataset_folder / cache_subdir / video_path.stem
    cached_frames = sorted(cache_dir.glob("frame_*.png"))
    if not cached_frames:
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


async def _run_image_modality(
    modality: str,
    client,
    dataset_folder: Path,
    jobs: list[SegmentJob],
    skip_api: bool,
    delay_between_segments: int,
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
            delay_between_pairs=delay_between_segments,
            skip_api=skip_api,
        )
    elif modality == "event":
        results = await run_event_parallel_pipeline(
            client,
            paired_frames,
            delay_between_pairs=delay_between_segments,
            skip_api=skip_api,
        )
    elif modality == "depth":
        results = await run_depth_parallel_pipeline(
            client,
            paired_frames,
            delay_between_pairs=delay_between_segments,
            skip_api=skip_api,
        )
    elif modality == "ir":
        results = await run_ir_parallel_pipeline(
            client,
            paired_frames,
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


async def _run_audio_modality(
    client,
    dataset_folder: Path,
    jobs: list[SegmentJob],
    skip_api: bool,
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
    return wrapped


async def _run_segmented_pipeline_async(
    dataset_folder: Path,
    output_folder: Path,
    test_mode: bool,
    skip_api: bool,
    delay_between_segments: int,
    modalities: tuple[str, ...],
) -> dict[str, Path]:
    jobs = _load_segment_jobs(dataset_folder, test_mode=test_mode)
    if not jobs:
        print("ERROR: No task segment manifests found. Run task slicing first.")
        return {}

    print(f"Found {len(jobs)} task segment(s).")
    client = None if skip_api else create_gemini_client()

    output_folder.mkdir(parents=True, exist_ok=True)
    written_paths: dict[str, Path] = {}

    print(f"Selected segmented QA modalities: {', '.join(modalities)}")

    for modality in IMAGE_MODALITIES:
        if modality not in modalities:
            continue
        results = await _run_image_modality(
            modality=modality,
            client=client,
            dataset_folder=dataset_folder,
            jobs=jobs,
            skip_api=skip_api,
            delay_between_segments=delay_between_segments,
        )
        output_path = output_folder / SEGMENTED_OUTPUT_FILES[modality]
        _save_json(results, output_path)
        written_paths[modality] = output_path
        print(f"Wrote {len(results)} segmented {modality} sample(s) to {output_path}")

    if "audio" in modalities:
        audio_results = await _run_audio_modality(
            client=client,
            dataset_folder=dataset_folder,
            jobs=jobs,
            skip_api=skip_api,
        )
        audio_output_path = output_folder / SEGMENTED_OUTPUT_FILES["audio"]
        _save_json(audio_results, audio_output_path)
        written_paths["audio"] = audio_output_path
        print(f"Wrote {len(audio_results)} segmented audio sample(s) to {audio_output_path}")

    return written_paths


def run_segmented_pipeline(
    dataset_folder: Path | str = "dataset",
    output_folder: Path | str = "segmented_outputs",
    test_mode: bool = False,
    skip_api: bool = False,
    delay_between_segments: int | None = None,
    modalities: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Path]:
    """Run selected QA modalities over task segments."""
    dataset_folder = Path(dataset_folder)
    output_folder = Path(output_folder)
    if not dataset_folder.exists():
        print(f"ERROR: Dataset folder not found: {dataset_folder}")
        return {}

    selected_modalities = _normalize_modalities(modalities)

    resolved_delay = 0 if skip_api else 4
    if delay_between_segments is not None:
        resolved_delay = delay_between_segments

    return asyncio.run(
        _run_segmented_pipeline_async(
            dataset_folder=dataset_folder,
            output_folder=output_folder,
            test_mode=test_mode,
            skip_api=skip_api,
            delay_between_segments=resolved_delay,
            modalities=selected_modalities,
        )
    )
