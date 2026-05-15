"""Automatic semantic task slicing for dataset media.

This module creates editable metadata manifests that divide each day/night
recording into semantic task segments. It does not create physical clips.
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

from annotation_feature.pipeline.client import create_gemini_client
from annotation_feature.pipeline.utils import build_image_parts, encode_frames_to_base64

try:
    from google.genai import types
except ImportError:
    types = None


MODEL_NAME = "gemini-3-flash-preview"
MIN_SEGMENT_SECONDS = 15.0
VIDEO_MODALITIES = ("rgb", "event", "depth", "ir")
AUDIO_EXTENSIONS = {".m4a", ".mp3", ".wav", ".aac", ".flac"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".mpeg", ".mpg"}


@dataclass
class MediaSample:
    split_dir: Path
    prefix: str
    side: str
    videos: dict[str, Path] = field(default_factory=dict)
    audio: Path | None = None
    with_audio: Path | None = None


def _strip_markdown_fence(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.I)
    return cleaned.strip()


def _parse_json_response(text: str) -> dict[str, Any]:
    cleaned = _strip_markdown_fence(text)
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if not match:
        raise ValueError("No JSON object found in task slicing response")

    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Task slicing response must be a JSON object")
    return parsed


def _seconds_to_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    whole_seconds = int(seconds)
    millis = int(round((seconds - whole_seconds) * 1000))
    if millis == 1000:
        whole_seconds += 1
        millis = 0
    minutes, sec = divmod(whole_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}.{millis:03d}"


def _get_video_duration_seconds(path: Path) -> float | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    if fps <= 0 or frame_count <= 0:
        return None
    return float(frame_count / fps)


def _sample_video_frames(
    video_path: Path,
    output_dir: Path,
    max_frames: int = 5,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        cap.release()
        return []

    if max_frames <= 1:
        indices = [0]
    else:
        indices = sorted(
            {
                min(frame_count - 1, round((frame_count - 1) * index / (max_frames - 1)))
                for index in range(max_frames)
            }
        )

    frames: list[Path] = []
    for frame_index in indices:
        frame_path = output_dir / f"{video_path.stem}_sample_{frame_index:06d}.png"
        if frame_path.exists():
            frames.append(frame_path)
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue
        cv2.imwrite(str(frame_path), frame)
        frames.append(frame_path)

    cap.release()
    return frames


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
        current_file = await asyncio.to_thread(client.files.get, name=current_file.name)


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


def discover_media_samples(dataset_folder: Path) -> list[MediaSample]:
    samples: dict[tuple[Path, str], MediaSample] = {}

    for file in dataset_folder.rglob("*"):
        if not file.is_file():
            continue

        stem = file.stem.lower()
        suffix = file.suffix.lower()
        if suffix in VIDEO_EXTENSIONS:
            for modality in VIDEO_MODALITIES:
                marker = f"_{modality}"
                if not stem.endswith(marker):
                    continue

                prefix = file.stem[: -len(marker)]
                side = "night" if "_night" in prefix.lower() else "day" if "_day" in prefix.lower() else "unknown"
                key = (file.parent, prefix)
                sample = samples.setdefault(key, MediaSample(file.parent, prefix, side))
                sample.videos[modality] = file
                break

            if stem.endswith("_rgb_with_audio"):
                prefix = file.stem[: -len("_rgb_with_audio")]
                side = "night" if "_night" in prefix.lower() else "day" if "_day" in prefix.lower() else "unknown"
                key = (file.parent, prefix)
                sample = samples.setdefault(key, MediaSample(file.parent, prefix, side))
                sample.with_audio = file

        if suffix in AUDIO_EXTENSIONS:
            side = "night" if "_night" in stem else "day" if "_day" in stem else "unknown"
            key = (file.parent, file.stem)
            sample = samples.setdefault(key, MediaSample(file.parent, file.stem, side))
            sample.audio = file

    return sorted(samples.values(), key=lambda item: (str(item.split_dir), item.prefix))


def _build_prompt(sample: MediaSample, duration_seconds: float | None, sampled_frames: dict[str, list[Path]]) -> str:
    duration_text = f"{duration_seconds:.3f} seconds" if duration_seconds else "unknown"
    modality_lines = []
    for modality in VIDEO_MODALITIES:
        path = sample.videos.get(modality)
        frame_names = [frame.name for frame in sampled_frames.get(modality, [])]
        if path:
            modality_lines.append(f"- {modality}: {path.name}; sampled frames: {', '.join(frame_names) or 'none'}")
    if sample.audio:
        modality_lines.append(f"- audio: {sample.audio.name}")
    if sample.with_audio:
        modality_lines.append(f"- with_audio_video: {sample.with_audio.name}")

    return "\n".join(
        [
            "You are segmenting an egocentric multimodal recording into semantic task steps.",
            "Use all provided modality evidence: RGB, event, depth, IR, and audio when available.",
            "Create editable task-boundary suggestions, not fine-grained object motions.",
            "A segment should be a coherent semantic task such as peeling a carrot, slicing a carrot, washing hands, or drying hands.",
            "Do not invent tiny segments for brief pauses, reaches, or object adjustments unless the task truly changes.",
            f"Every segment must be at least {MIN_SEGMENT_SECONDS:.0f} seconds long unless the full recording is shorter.",
            "Merge brief actions like applying soap, turning on water, picking up an object, or short transitions into the surrounding semantic task.",
            "Return ONLY valid JSON with this exact structure:",
            "{",
            '  "segments": [',
            "    {",
            '      "task_label": "short semantic label",',
            '      "start_seconds": 0.0,',
            '      "end_seconds": 12.5,',
            '      "confidence": 0.0,',
            '      "evidence_modalities": ["rgb", "audio"],',
            '      "notes": "brief evidence for this segment"',
            "    }",
            "  ]",
            "}",
            "Rules:",
            "- Cover the meaningful task portion of the recording with ordered, non-overlapping segments.",
            f"- Each final segment must be at least {MIN_SEGMENT_SECONDS:.0f} seconds long.",
            "- Use seconds as numbers, not timestamp strings.",
            "- Keep task_label concise and lowercase.",
            "- confidence must be between 0 and 1.",
            f"Source prefix: {sample.prefix}",
            f"Approximate duration: {duration_text}",
            "Available evidence:",
            *modality_lines,
        ]
    )


async def _call_gemini_with_retry(client, contents: list, max_retries: int = 3) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=MODEL_NAME,
                contents=contents,
            )
            return response.text
        except Exception as exc:
            if attempt == max_retries:
                raise
            text = str(exc).lower()
            is_quota_error = any(
                token in text
                for token in ("quota", "rate limit", "rate_limit", "429", "resource_exhausted")
            )
            wait_seconds = 30 * attempt if is_quota_error else 2 * attempt
            print(
                f"  Gemini task-slicing call failed on attempt {attempt}/{max_retries}; "
                f"retrying in {wait_seconds}s: {exc}"
            )
            await asyncio.sleep(wait_seconds)
    raise RuntimeError("Gemini call failed")


def _segment_duration(segment: dict[str, Any]) -> float:
    return float(segment["end_seconds"]) - float(segment["start_seconds"])


def _combine_labels(first_label: str, second_label: str) -> str:
    generic_labels = {"", "semantic task"}
    first = (first_label or "").strip().lower()
    second = (second_label or "").strip().lower()

    if first in generic_labels:
        return second or first
    if second in generic_labels or second == first:
        return first
    return f"{first} / {second}"


def _combine_evidence(first: Any, second: Any) -> list[str]:
    combined: list[str] = []
    for source in (first, second):
        if not isinstance(source, list):
            continue
        for item in source:
            value = str(item)
            if value not in combined:
                combined.append(value)
    return combined


def _combine_notes(first_note: str, second_note: str) -> str:
    notes = []
    for note in (first_note, second_note):
        text = (note or "").strip()
        if text and text not in notes:
            notes.append(text)
    return " ".join(notes)


def _merge_two_segments(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    first_duration = max(0.0, _segment_duration(first))
    second_duration = max(0.0, _segment_duration(second))
    total_duration = first_duration + second_duration
    if total_duration > 0:
        confidence = (
            float(first.get("confidence", 0.0)) * first_duration
            + float(second.get("confidence", 0.0)) * second_duration
        ) / total_duration
    else:
        confidence = min(float(first.get("confidence", 0.0)), float(second.get("confidence", 0.0)))

    start_seconds = float(first["start_seconds"])
    end_seconds = float(second["end_seconds"])
    return {
        "segment_id": first["segment_id"],
        "task_label": _combine_labels(str(first.get("task_label", "")), str(second.get("task_label", ""))),
        "start_seconds": round(start_seconds, 3),
        "end_seconds": round(end_seconds, 3),
        "start_timestamp": _seconds_to_timestamp(start_seconds),
        "end_timestamp": _seconds_to_timestamp(end_seconds),
        "confidence": round(max(0.0, min(1.0, confidence)), 3),
        "evidence_modalities": _combine_evidence(
            first.get("evidence_modalities"),
            second.get("evidence_modalities"),
        ),
        "notes": _combine_notes(str(first.get("notes", "")), str(second.get("notes", ""))),
    }


def _renumber_segments(segments: list[dict[str, Any]], sample: MediaSample) -> list[dict[str, Any]]:
    renumbered: list[dict[str, Any]] = []
    for index, segment in enumerate(segments, start=1):
        start_seconds = float(segment["start_seconds"])
        end_seconds = float(segment["end_seconds"])
        renumbered.append(
            {
                **segment,
                "segment_id": f"{sample.prefix}_seg_{index:03d}",
                "start_seconds": round(start_seconds, 3),
                "end_seconds": round(end_seconds, 3),
                "start_timestamp": _seconds_to_timestamp(start_seconds),
                "end_timestamp": _seconds_to_timestamp(end_seconds),
            }
        )
    return renumbered


def _enforce_minimum_segment_seconds(
    segments: list[dict[str, Any]],
    sample: MediaSample,
    minimum_seconds: float = MIN_SEGMENT_SECONDS,
) -> list[dict[str, Any]]:
    if len(segments) <= 1:
        return _renumber_segments(segments, sample)

    merged = [dict(segment) for segment in segments]
    while len(merged) > 1:
        short_index = next(
            (
                index
                for index, segment in enumerate(merged)
                if _segment_duration(segment) < minimum_seconds
            ),
            None,
        )
        if short_index is None:
            break

        if short_index == 0:
            merged[1] = _merge_two_segments(merged[0], merged[1])
            del merged[0]
        else:
            merged[short_index - 1] = _merge_two_segments(
                merged[short_index - 1],
                merged[short_index],
            )
            del merged[short_index]

    return _renumber_segments(merged, sample)


def _normalize_segments(raw_segments: Any, duration_seconds: float | None, sample: MediaSample) -> list[dict[str, Any]]:
    if not isinstance(raw_segments, list):
        raw_segments = []

    segments: list[dict[str, Any]] = []
    previous_end = 0.0
    duration_limit = duration_seconds if duration_seconds and duration_seconds > 0 else None

    for index, raw in enumerate(raw_segments, start=1):
        if not isinstance(raw, dict):
            continue

        try:
            start_seconds = float(raw.get("start_seconds", previous_end))
            end_seconds = float(raw.get("end_seconds", start_seconds))
        except (TypeError, ValueError):
            continue

        start_seconds = max(previous_end, start_seconds)
        if duration_limit is not None:
            start_seconds = min(start_seconds, duration_limit)
            end_seconds = min(end_seconds, duration_limit)
        if end_seconds <= start_seconds:
            continue

        label = str(raw.get("task_label") or f"task segment {index}").strip().lower()
        try:
            confidence = float(raw.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        evidence_modalities = raw.get("evidence_modalities")
        if not isinstance(evidence_modalities, list):
            evidence_modalities = list(sample.videos.keys())
            if sample.audio or sample.with_audio:
                evidence_modalities.append("audio")

        segment_index = len(segments) + 1
        segments.append(
            {
                "segment_id": f"{sample.prefix}_seg_{segment_index:03d}",
                "task_label": label,
                "start_seconds": round(start_seconds, 3),
                "end_seconds": round(end_seconds, 3),
                "start_timestamp": _seconds_to_timestamp(start_seconds),
                "end_timestamp": _seconds_to_timestamp(end_seconds),
                "confidence": confidence,
                "evidence_modalities": [str(item) for item in evidence_modalities],
                "notes": str(raw.get("notes") or "").strip(),
            }
        )
        previous_end = end_seconds

    if segments:
        return _enforce_minimum_segment_seconds(segments, sample)

    fallback_end = duration_limit if duration_limit is not None else 0.0
    return _enforce_minimum_segment_seconds([
        {
            "segment_id": f"{sample.prefix}_seg_001",
            "task_label": "semantic task",
            "start_seconds": 0.0,
            "end_seconds": round(fallback_end, 3),
            "start_timestamp": _seconds_to_timestamp(0.0),
            "end_timestamp": _seconds_to_timestamp(fallback_end),
            "confidence": 0.0,
            "evidence_modalities": list(sample.videos.keys()) + (["audio"] if sample.audio or sample.with_audio else []),
            "notes": "Fallback segment created because automatic task slicing did not return usable boundaries.",
        }
    ], sample)


async def _slice_sample(
    client,
    sample: MediaSample,
    dataset_folder: Path,
    skip_api: bool = False,
    max_frames_per_modality: int = 5,
) -> dict[str, Any]:
    reference_video = sample.videos.get("rgb") or next(iter(sample.videos.values()), None)
    duration_seconds = _get_video_duration_seconds(reference_video) if reference_video else None

    sample_cache_dir = dataset_folder / ".task_slice_cache" / sample.prefix
    sampled_frames: dict[str, list[Path]] = {}
    for modality, path in sample.videos.items():
        sampled_frames[modality] = _sample_video_frames(
            path,
            sample_cache_dir / modality,
            max_frames=max_frames_per_modality,
        )

    raw_segments: list[dict[str, Any]] = []
    model_error: str | None = None
    if not skip_api and client is not None:
        try:
            image_paths = [
                frame
                for modality in VIDEO_MODALITIES
                for frame in sampled_frames.get(modality, [])
            ]
            image_parts = build_image_parts(encode_frames_to_base64(image_paths))
            media_parts = []
            media_path = sample.with_audio or sample.audio
            if media_path:
                media_parts.append(await _upload_file_part(client, media_path, _guess_mime_type(media_path)))

            prompt = _build_prompt(sample, duration_seconds, sampled_frames)
            parsed = _parse_json_response(await _call_gemini_with_retry(client, image_parts + media_parts + [prompt]))
            raw_segments = parsed.get("segments", [])
        except Exception as exc:
            model_error = str(exc)
            print(f"  WARNING: Task slicing model call failed for {sample.prefix}: {exc}")

    segments = _normalize_segments(raw_segments, duration_seconds, sample)
    return {
        "source_prefix": sample.prefix,
        "split_dir": sample.split_dir.name,
        "side": sample.side,
        "status": "suggested" if model_error is None and raw_segments else "fallback",
        "segment_source": "all_modalities",
        "minimum_segment_seconds": MIN_SEGMENT_SECONDS,
        "duration_seconds": round(duration_seconds, 3) if duration_seconds is not None else None,
        "source_files": {
            "videos": {modality: str(path) for modality, path in sorted(sample.videos.items())},
            "audio": str(sample.audio) if sample.audio else None,
            "with_audio": str(sample.with_audio) if sample.with_audio else None,
        },
        "model": None if skip_api else MODEL_NAME,
        "model_error": model_error,
        "segments": segments,
    }


async def _run_task_slicing_async(
    samples: list[MediaSample],
    dataset_folder: Path,
    skip_api: bool,
) -> list[Path]:
    client = None
    if not skip_api:
        try:
            client = create_gemini_client()
        except Exception as exc:
            print(f"WARNING: Could not initialize Gemini client for task slicing: {exc}")
            print("Falling back to one editable whole-recording segment per sample.")
            skip_api = True

    output_paths: list[Path] = []
    for index, sample in enumerate(samples, start=1):
        print(f"\n[{index}/{len(samples)}] Slicing task segments for {sample.prefix}")
        manifest = await _slice_sample(client, sample, dataset_folder, skip_api=skip_api)
        output_path = sample.split_dir / f"{sample.prefix}_task_segments.json"
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
        output_paths.append(output_path)
        print(f"  Wrote {len(manifest['segments'])} segment(s) to {output_path}")

    return output_paths


def run_task_slicing(
    dataset_folder: Path | str = "dataset",
    test_mode: bool = False,
    test_sample_index: int = 0,
    skip_api: bool = False,
) -> list[Path]:
    """Generate editable semantic task segment manifests for dataset samples."""
    dataset_folder = Path(dataset_folder)
    if not dataset_folder.exists():
        print(f"ERROR: Dataset folder not found: {dataset_folder}")
        return []

    samples = discover_media_samples(dataset_folder)
    if test_mode:
        samples = samples[test_sample_index:test_sample_index + 1]

    if not samples:
        print("ERROR: No media samples found for task slicing.")
        return []

    print(f"Found {len(samples)} media sample(s) for task slicing.")
    print("Output: metadata-only *_task_segments.json manifests beside the source media.")
    return asyncio.run(_run_task_slicing_async(samples, dataset_folder, skip_api=skip_api))
