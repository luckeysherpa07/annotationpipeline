"""Semantic and motion alignment for day/night RGB activity pairs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
FEATURE_FPS = 5.0
COARSE_STRIDE = 5
PREVIEW_FPS = 10.0
LOW_CONFIDENCE_THRESHOLD = 0.35
_CLIP_RUNTIME: tuple[Any, Any, Any] | None = None


def _video_metadata(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open RGB video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if fps <= 0 or frame_count <= 0:
        raise ValueError(f"RGB video has invalid metadata: {path}")
    return {
        "path": str(path),
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": frame_count / fps,
        "width": width,
        "height": height,
    }


def _sample_video_frames(path: Path, sample_fps: float) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    meta = _video_metadata(path)
    source_fps = float(meta["fps"])
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    times: list[float] = []
    motion: list[float] = []
    previous_gray: np.ndarray | None = None
    next_sample_time = 0.0
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_time = frame_index / source_fps
        frame_index += 1
        if frame_time + (0.5 / source_fps) < next_sample_time:
            continue
        resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        if previous_gray is None:
            motion.append(0.0)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                previous_gray,
                gray,
                None,
                pyr_scale=0.5,
                levels=2,
                winsize=15,
                iterations=2,
                poly_n=5,
                poly_sigma=1.1,
                flags=0,
            )
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion.append(float(np.mean(magnitude)))
        previous_gray = gray
        frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        times.append(frame_time)
        next_sample_time += 1.0 / sample_fps

    cap.release()
    if len(frames) < 3:
        raise ValueError(f"RGB video is too short to align: {path}")
    return frames, np.asarray(times, dtype=np.float32), np.asarray(motion, dtype=np.float32)


def _normalize_rows(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-8)


def _robust_normalize(values: np.ndarray) -> np.ndarray:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1e-6, 1.4826 * mad)
    return np.clip((values - median) / scale, -4.0, 4.0).astype(np.float32)


def _get_clip_runtime() -> tuple[Any, Any, Any]:
    global _CLIP_RUNTIME
    if _CLIP_RUNTIME is not None:
        return _CLIP_RUNTIME
    try:
        import torch
        from transformers import CLIPImageProcessor, CLIPVisionModel
    except ImportError as exc:
        raise RuntimeError(
            "Day/night alignment requires torch and transformers. Run it with the project virtual environment."
        ) from exc

    try:
        processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_NAME, local_files_only=True)
        model = CLIPVisionModel.from_pretrained(CLIP_MODEL_NAME, local_files_only=True)
    except OSError as exc:
        raise RuntimeError(
            f"Cached model {CLIP_MODEL_NAME!r} was not found; download it before running option 79."
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    _CLIP_RUNTIME = processor, model, device
    return _CLIP_RUNTIME


def _extract_clip_embeddings(frames: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Day/night alignment requires torch and transformers. Run it with the project virtual environment."
        ) from exc

    processor, model, device = _get_clip_runtime()
    embeddings: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(frames), batch_size):
            batch = processor(images=frames[start : start + batch_size], return_tensors="pt")
            pixel_values = batch["pixel_values"].to(device)
            pooled = model(pixel_values=pixel_values).pooler_output
            pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
            embeddings.append(pooled.cpu().numpy())
    return np.concatenate(embeddings, axis=0).astype(np.float32)


def _feature_cache_key(path: Path, sample_fps: float) -> dict[str, Any]:
    stat = path.stat()
    return {
        "source": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sample_fps": sample_fps,
        "model": CLIP_MODEL_NAME,
    }


def _load_or_extract_features(path: Path, cache_path: Path, sample_fps: float) -> dict[str, np.ndarray]:
    expected_key = _feature_cache_key(path, sample_fps)
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                cache_key = json.loads(str(cached["cache_key"].item()))
                if cache_key == expected_key:
                    return {
                        "semantic": cached["semantic"].astype(np.float32),
                        "motion": cached["motion"].astype(np.float32),
                        "times": cached["times"].astype(np.float32),
                    }
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass

    frames, times, motion = _sample_video_frames(path, sample_fps)
    semantic = _extract_clip_embeddings(frames)
    result = {
        "semantic": _normalize_rows(semantic).astype(np.float32),
        "motion": _robust_normalize(motion),
        "times": times,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, cache_key=json.dumps(expected_key), **result)
    return result


def _active_interval(motion: np.ndarray) -> tuple[int, int]:
    """Return a conservative activity interval while retaining at least 70% of samples."""
    if motion.size < 5:
        return 0, max(0, motion.size - 1)
    window = min(11, motion.size if motion.size % 2 else motion.size - 1)
    kernel = np.ones(window, dtype=np.float32) / window
    smooth = np.convolve(np.abs(motion), kernel, mode="same")
    threshold = float(np.median(smooth) + 0.5 * np.median(np.abs(smooth - np.median(smooth))))
    active = np.flatnonzero(smooth >= threshold)
    if active.size == 0:
        return 0, motion.size - 1
    padding = COARSE_STRIDE * 2
    start = max(0, int(active[0]) - padding)
    end = min(motion.size - 1, int(active[-1]) + padding)
    minimum = int(np.ceil(0.70 * motion.size))
    if end - start + 1 < minimum:
        missing = minimum - (end - start + 1)
        start = max(0, start - (missing + 1) // 2)
        end = min(motion.size - 1, end + missing // 2)
        if end - start + 1 < minimum:
            start = max(0, end - minimum + 1)
            end = min(motion.size - 1, start + minimum - 1)
    return start, end


def _pairwise_cost(
    reference_semantic: np.ndarray,
    target_semantic: np.ndarray,
    reference_motion: np.ndarray,
    target_motion: np.ndarray,
) -> np.ndarray:
    semantic_cost = 1.0 - np.clip(reference_semantic @ target_semantic.T, -1.0, 1.0)
    motion_cost = np.abs(reference_motion[:, None] - target_motion[None, :]) / 8.0
    return (0.8 * semantic_cost + 0.2 * np.clip(motion_cost, 0.0, 1.0)).astype(np.float32)


def _constrained_dtw(cost: np.ndarray, bounds: list[tuple[int, int]] | None = None) -> list[tuple[int, int]]:
    """Global DTW using (1,1), (1,2), and (2,1) steps, limiting local speed to 0.5-2x."""
    reference_count, target_count = cost.shape
    if reference_count < 2 or target_count < 2:
        raise ValueError("DTW requires at least two samples per side.")
    ratio = target_count / reference_count
    if not 0.5 <= ratio <= 2.0:
        raise ValueError(f"Active day/night duration ratio {ratio:.3f} is outside 0.5-2.0.")

    accumulated = np.full((reference_count, target_count), np.inf, dtype=np.float32)
    backtrack = np.zeros((reference_count, target_count), dtype=np.uint8)
    accumulated[0, 0] = cost[0, 0]
    steps = ((1, 1, 1), (1, 2, 2), (2, 1, 3))
    for i in range(reference_count):
        low, high = bounds[i] if bounds is not None else (0, target_count - 1)
        low = max(0, low)
        high = min(target_count - 1, high)
        for j in range(low, high + 1):
            if i == 0 and j == 0:
                continue
            best = np.inf
            direction = 0
            for di, dj, code in steps:
                previous_i, previous_j = i - di, j - dj
                if previous_i < 0 or previous_j < 0:
                    continue
                candidate = accumulated[previous_i, previous_j]
                if candidate < best:
                    best, direction = candidate, code
            if np.isfinite(best):
                accumulated[i, j] = best + cost[i, j]
                backtrack[i, j] = direction

    if not np.isfinite(accumulated[-1, -1]):
        raise ValueError("No constrained DTW path found for the day/night RGB pair.")
    path: list[tuple[int, int]] = []
    i, j = reference_count - 1, target_count - 1
    while True:
        path.append((i, j))
        if i == 0 and j == 0:
            break
        direction = int(backtrack[i, j])
        if direction == 1:
            i, j = i - 1, j - 1
        elif direction == 2:
            i, j = i - 1, j - 2
        elif direction == 3:
            i, j = i - 2, j - 1
        else:
            raise ValueError("DTW backtracking encountered an unreachable cell.")
    path.reverse()
    return path


def _path_by_reference(path: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[int, list[int]] = {}
    for reference_index, target_index in path:
        grouped.setdefault(reference_index, []).append(target_index)
    reference = np.asarray(sorted(grouped), dtype=np.float32)
    target = np.asarray([np.median(grouped[int(index)]) for index in reference], dtype=np.float32)
    return reference, np.maximum.accumulate(target)


def _fine_alignment(
    night: dict[str, np.ndarray],
    day: dict[str, np.ndarray],
    band_seconds: float = 5.0,
) -> tuple[list[tuple[int, int]], np.ndarray, tuple[int, int], tuple[int, int]]:
    night_start, night_end = _active_interval(night["motion"])
    day_start, day_end = _active_interval(day["motion"])
    night_coarse_indices = np.arange(night_start, night_end + 1, COARSE_STRIDE)
    day_coarse_indices = np.arange(day_start, day_end + 1, COARSE_STRIDE)
    if night_coarse_indices[-1] != night_end:
        night_coarse_indices = np.append(night_coarse_indices, night_end)
    if day_coarse_indices[-1] != day_end:
        day_coarse_indices = np.append(day_coarse_indices, day_end)

    coarse_cost = _pairwise_cost(
        night["semantic"][night_coarse_indices],
        day["semantic"][day_coarse_indices],
        night["motion"][night_coarse_indices],
        day["motion"][day_coarse_indices],
    )
    coarse_path = _constrained_dtw(coarse_cost)
    coarse_ref, coarse_target = _path_by_reference(coarse_path)
    coarse_night_global = night_coarse_indices[coarse_ref.astype(np.int32)]
    coarse_day_global = day_coarse_indices[np.rint(coarse_target).astype(np.int32)]

    night_indices = np.arange(night_start, night_end + 1)
    day_indices = np.arange(day_start, day_end + 1)
    fine_cost = _pairwise_cost(
        night["semantic"][night_indices],
        day["semantic"][day_indices],
        night["motion"][night_indices],
        day["motion"][day_indices],
    )
    predicted_day = np.interp(night_indices, coarse_night_global, coarse_day_global) - day_start
    radius = max(1, int(round(band_seconds * FEATURE_FPS)))
    bounds = [
        (int(np.floor(prediction - radius)), int(np.ceil(prediction + radius)))
        for prediction in predicted_day
    ]
    bounds[0] = (0, max(bounds[0][1], 0))
    bounds[-1] = (min(bounds[-1][0], len(day_indices) - 1), len(day_indices) - 1)
    fine_path_local = _constrained_dtw(fine_cost, bounds=bounds)
    fine_path = [(night_start + i, day_start + j) for i, j in fine_path_local]
    return fine_path, fine_cost, (night_start, night_end), (day_start, day_end)


def _path_knots_and_confidence(
    path: list[tuple[int, int]],
    cost: np.ndarray,
    night_interval: tuple[int, int],
    day_interval: tuple[int, int],
    night_times: np.ndarray,
    day_times: np.ndarray,
) -> list[dict[str, Any]]:
    night_start, _ = night_interval
    day_start, _ = day_interval
    ref_indices, target_indices = _path_by_reference(path)
    ref_indices = ref_indices.astype(np.int32)
    target_indices_int = np.rint(target_indices).astype(np.int32)
    local_costs = np.asarray(
        [cost[ref - night_start, target - day_start] for ref, target in zip(ref_indices, target_indices_int)],
        dtype=np.float32,
    )
    low, high = np.percentile(local_costs, [10, 90])
    scale = max(1e-6, float(high - low))
    cost_confidence = 1.0 - np.clip((local_costs - low) / scale, 0.0, 1.0)
    knots: list[dict[str, Any]] = []
    exclusion = max(1, int(round(FEATURE_FPS)))
    for position, (night_index, day_index) in enumerate(zip(ref_indices, target_indices_int)):
        row = cost[night_index - night_start]
        local_day = day_index - day_start
        mask = np.ones(row.size, dtype=bool)
        mask[max(0, local_day - exclusion) : min(row.size, local_day + exclusion + 1)] = False
        alternative = float(np.min(row[mask])) if np.any(mask) else float(row[local_day])
        selected = float(row[local_day])
        margin_confidence = float(np.clip((alternative - selected) / max(abs(alternative), 1e-6), 0.0, 1.0))
        confidence = float(np.clip(0.7 * cost_confidence[position] + 0.3 * margin_confidence, 0.0, 1.0))
        knots.append(
            {
                "night_sample_index": int(night_index),
                "night_time_seconds": round(float(night_times[night_index]), 6),
                "day_sample_index": int(day_index),
                "day_time_seconds": round(float(day_times[day_index]), 6),
                "cost": round(selected, 6),
                "confidence": round(confidence, 6),
                "review": confidence < LOW_CONFIDENCE_THRESHOLD,
            }
        )
    return knots


def _interpolate_mapping(
    source_meta: dict[str, Any],
    target_meta: dict[str, Any],
    source_times: np.ndarray,
    target_times: np.ndarray,
    confidence: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    first, last = float(source_times[0]), float(source_times[-1])
    for source_frame in range(int(source_meta["frame_count"])):
        source_time = source_frame / float(source_meta["fps"])
        if source_time < first or source_time > last:
            rows.append(
                {
                    "source_frame": source_frame,
                    "source_time_seconds": round(source_time, 6),
                    "target_frame": "",
                    "target_time_seconds": "",
                    "confidence": 0.0,
                    "status": "unmatched",
                }
            )
            continue
        target_time = float(np.interp(source_time, source_times, target_times))
        target_frame = int(round(target_time * float(target_meta["fps"])))
        target_frame = min(max(0, target_frame), int(target_meta["frame_count"]) - 1)
        score = float(np.interp(source_time, source_times, confidence))
        rows.append(
            {
                "source_frame": source_frame,
                "source_time_seconds": round(source_time, 6),
                "target_frame": target_frame,
                "target_time_seconds": round(target_frame / float(target_meta["fps"]), 6),
                "confidence": round(score, 6),
                "status": "review" if score < LOW_CONFIDENCE_THRESHOLD else "matched",
            }
        )
    return rows


def _inverse_knots(knots: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for knot in knots:
        grouped.setdefault(int(knot["day_sample_index"]), []).append(knot)
    day_indices = sorted(grouped)
    day_times = np.asarray(
        [np.median([item["day_time_seconds"] for item in grouped[index]]) for index in day_indices],
        dtype=np.float32,
    )
    night_times = np.asarray(
        [np.median([item["night_time_seconds"] for item in grouped[index]]) for index in day_indices],
        dtype=np.float32,
    )
    confidence = np.asarray(
        [np.median([item["confidence"] for item in grouped[index]]) for index in day_indices],
        dtype=np.float32,
    )
    return day_times, np.maximum.accumulate(night_times), confidence


def _write_mapping_csv(path: Path, rows: list[dict[str, Any]], source_label: str, target_label: str) -> None:
    fieldnames = [
        f"{source_label}_frame",
        f"{source_label}_time_seconds",
        f"{target_label}_frame",
        f"{target_label}_time_seconds",
        "confidence",
        "status",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    f"{source_label}_frame": row["source_frame"],
                    f"{source_label}_time_seconds": row["source_time_seconds"],
                    f"{target_label}_frame": row["target_frame"],
                    f"{target_label}_time_seconds": row["target_time_seconds"],
                    "confidence": row["confidence"],
                    "status": row["status"],
                }
            )


def _review_intervals(knots: list[dict[str, Any]]) -> list[dict[str, float]]:
    flagged = [item for item in knots if item["review"]]
    if not flagged:
        return []
    intervals: list[dict[str, float]] = []
    start = previous = flagged[0]
    for item in flagged[1:]:
        if float(item["night_time_seconds"]) - float(previous["night_time_seconds"]) > 1.0:
            intervals.append(
                {
                    "night_start_seconds": start["night_time_seconds"],
                    "night_end_seconds": previous["night_time_seconds"],
                }
            )
            start = item
        previous = item
    intervals.append(
        {
            "night_start_seconds": start["night_time_seconds"],
            "night_end_seconds": previous["night_time_seconds"],
        }
    )
    return intervals


def _unmatched_ranges(start: float, end: float, duration: float) -> list[dict[str, float]]:
    ranges: list[dict[str, float]] = []
    if start > 1e-6:
        ranges.append({"start_seconds": 0.0, "end_seconds": round(start, 6)})
    if duration - end > 1e-6:
        ranges.append({"start_seconds": round(end, 6), "end_seconds": round(duration, 6)})
    return ranges


def _read_frame(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    return frame if ok else None


def _preview_panel(frame: np.ndarray, size: tuple[int, int], label: str, detail: str, color: tuple[int, int, int]) -> np.ndarray:
    panel = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    cv2.rectangle(panel, (0, 0), (size[0], 56), (0, 0, 0), thickness=-1)
    cv2.putText(panel, label, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    cv2.putText(panel, detail, (12, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    return panel


def _write_preview(
    path: Path,
    night_path: Path,
    day_path: Path,
    night_meta: dict[str, Any],
    day_meta: dict[str, Any],
    mapping_rows: list[dict[str, Any]],
) -> int:
    width, height = 640, 360
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), PREVIEW_FPS, (width * 2, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create alignment preview: {path}")
    night_cap = cv2.VideoCapture(str(night_path))
    day_cap = cv2.VideoCapture(str(day_path))
    output_count = int(np.ceil(float(night_meta["duration_seconds"]) * PREVIEW_FPS))
    written = 0
    try:
        for output_index in range(output_count):
            night_time = output_index / PREVIEW_FPS
            night_frame_index = min(
                int(round(night_time * float(night_meta["fps"]))), int(night_meta["frame_count"]) - 1
            )
            night_frame = _read_frame(night_cap, night_frame_index)
            if night_frame is None:
                break
            row = mapping_rows[night_frame_index]
            confidence = float(row["confidence"])
            status = str(row["status"])
            color = (60, 220, 60) if status == "matched" else (0, 180, 255) if status == "review" else (80, 80, 255)
            night_panel = _preview_panel(
                night_frame,
                (width, height),
                "NIGHT (reference)",
                f"frame {night_frame_index} | {night_time:.3f}s | {status} {confidence:.2f}",
                color,
            )
            if row["target_frame"] == "":
                day_frame = np.zeros_like(night_frame)
                day_detail = "unmatched"
            else:
                day_frame_index = int(row["target_frame"])
                day_frame = _read_frame(day_cap, day_frame_index)
                if day_frame is None:
                    day_frame = np.zeros_like(night_frame)
                day_detail = f"frame {day_frame_index} | {float(row['target_time_seconds']):.3f}s"
            day_panel = _preview_panel(day_frame, (width, height), "DAY (warped)", day_detail, color)
            writer.write(np.hstack((night_panel, day_panel)))
            written += 1
    finally:
        night_cap.release()
        day_cap.release()
        writer.release()
    return written


def run_day_night_rgb_pair_alignment(
    sample_name: str,
    dataset_folder: Path | str = "dataset",
    output_folder: Path | str | None = None,
    split_folder_name: str | None = None,
    write_preview: bool = True,
) -> dict[str, Any]:
    """Align one named night RGB recording to its day recording."""
    if not sample_name or any(character in sample_name for character in ("/", "\\")):
        raise ValueError("sample_name must be a non-empty filename-safe dataset sample name.")
    dataset_folder = Path(dataset_folder)
    split_folder_name = split_folder_name or f"{sample_name}_split"
    output_folder = Path(output_folder or f"day_night_alignment/{split_folder_name}")
    split_folder = dataset_folder / split_folder_name
    day_path = split_folder / f"{sample_name}_day_rgb.mp4"
    night_path = split_folder / f"{sample_name}_night_rgb.mp4"
    for path in (day_path, night_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required {sample_name} RGB video does not exist: {path}")

    output_folder.mkdir(parents=True, exist_ok=True)
    cache_folder = output_folder / ".feature_cache"
    day_meta = _video_metadata(day_path)
    night_meta = _video_metadata(night_path)
    day = _load_or_extract_features(
        day_path, cache_folder / f"{sample_name}_day_rgb_clip_motion.npz", FEATURE_FPS
    )
    night = _load_or_extract_features(
        night_path, cache_folder / f"{sample_name}_night_rgb_clip_motion.npz", FEATURE_FPS
    )

    path, fine_cost, night_interval, day_interval = _fine_alignment(night, day)
    knots = _path_knots_and_confidence(
        path,
        fine_cost,
        night_interval,
        day_interval,
        night["times"],
        day["times"],
    )
    night_times = np.asarray([item["night_time_seconds"] for item in knots], dtype=np.float32)
    day_times = np.maximum.accumulate(
        np.asarray([item["day_time_seconds"] for item in knots], dtype=np.float32)
    )
    confidence = np.asarray([item["confidence"] for item in knots], dtype=np.float32)
    night_to_day = _interpolate_mapping(night_meta, day_meta, night_times, day_times, confidence)
    inverse_day_times, inverse_night_times, inverse_confidence = _inverse_knots(knots)
    day_to_night = _interpolate_mapping(
        day_meta,
        night_meta,
        inverse_day_times,
        inverse_night_times,
        inverse_confidence,
    )

    night_csv = output_folder / f"{sample_name}_night_to_day_frames.csv"
    day_csv = output_folder / f"{sample_name}_day_to_night_frames.csv"
    _write_mapping_csv(night_csv, night_to_day, "night", "day")
    _write_mapping_csv(day_csv, day_to_night, "day", "night")
    preview_path = output_folder / f"{sample_name}_day_night_rgb_alignment_preview.mp4"
    preview_frames = 0
    if write_preview:
        preview_frames = _write_preview(preview_path, night_path, day_path, night_meta, day_meta, night_to_day)

    summary = {
        "sample": sample_name,
        "split_folder_name": split_folder_name,
        "reference_side": "night",
        "method": "coarse_to_fine_clip_motion_constrained_dtw",
        "settings": {
            "clip_model": CLIP_MODEL_NAME,
            "feature_fps": FEATURE_FPS,
            "coarse_fps": FEATURE_FPS / COARSE_STRIDE,
            "semantic_cost_weight": 0.8,
            "motion_cost_weight": 0.2,
            "fine_band_seconds": 5.0,
            "local_playback_ratio": [0.5, 2.0],
            "minimum_active_coverage": 0.70,
            "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
        },
        "sources": {"day": day_meta, "night": night_meta},
        "matched_ranges": {
            "night": {
                "start_seconds": knots[0]["night_time_seconds"],
                "end_seconds": knots[-1]["night_time_seconds"],
            },
            "day": {
                "start_seconds": knots[0]["day_time_seconds"],
                "end_seconds": knots[-1]["day_time_seconds"],
            },
        },
        "coverage": {
            "night": round((night_interval[1] - night_interval[0] + 1) / len(night["times"]), 6),
            "day": round((day_interval[1] - day_interval[0] + 1) / len(day["times"]), 6),
        },
        "unmatched_ranges": {
            "night": _unmatched_ranges(
                float(knots[0]["night_time_seconds"]),
                float(knots[-1]["night_time_seconds"]),
                float(night_meta["duration_seconds"]),
            ),
            "day": _unmatched_ranges(
                float(knots[0]["day_time_seconds"]),
                float(knots[-1]["day_time_seconds"]),
                float(day_meta["duration_seconds"]),
            ),
        },
        "review_intervals": _review_intervals(knots),
        "knots": knots,
        "outputs": {
            "alignment_json": str(output_folder / f"{sample_name}_day_night_rgb_alignment.json"),
            "night_to_day_csv": str(night_csv),
            "day_to_night_csv": str(day_csv),
            "preview": str(preview_path) if write_preview else None,
            "preview_frames": preview_frames,
        },
    }
    json_path = output_folder / f"{sample_name}_day_night_rgb_alignment.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def run_wash_cup_day_night_rgb_alignment(
    dataset_folder: Path | str = "dataset",
    output_folder: Path | str = "day_night_alignment/wash_cup_split",
    write_preview: bool = True,
) -> dict[str, Any]:
    """Align the wash-cup night RGB recording to its day recording."""
    return run_day_night_rgb_pair_alignment(
        sample_name="wash_cup",
        dataset_folder=dataset_folder,
        output_folder=output_folder,
        write_preview=write_preview,
    )


def run_cut_carrot_day_night_rgb_alignment(
    dataset_folder: Path | str = "dataset",
    output_folder: Path | str = "day_night_alignment/cut_carrot_split",
    write_preview: bool = True,
) -> dict[str, Any]:
    """Align the cut-carrot night RGB recording to its day recording."""
    return run_day_night_rgb_pair_alignment(
        sample_name="cut_carrot",
        dataset_folder=dataset_folder,
        output_folder=output_folder,
        write_preview=write_preview,
    )


def run_check_mailbox_day_night_rgb_alignment(
    dataset_folder: Path | str = "dataset",
    output_folder: Path | str = "day_night_alignment/check_mailbox_split",
    write_preview: bool = True,
) -> dict[str, Any]:
    """Align the check-mailbox night RGB recording to its day recording."""
    return run_day_night_rgb_pair_alignment(
        sample_name="check_mailbox",
        dataset_folder=dataset_folder,
        output_folder=output_folder,
        write_preview=write_preview,
    )


def _discover_day_night_rgb_pairs(dataset_folder: Path) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    for day_path in sorted(dataset_folder.glob("*_split/*_day_rgb.mp4")):
        sample_name = day_path.stem[: -len("_day_rgb")]
        night_path = day_path.with_name(f"{sample_name}_night_rgb.mp4")
        if night_path.is_file():
            pairs.append(
                {
                    "sample": sample_name,
                    "split_folder_name": day_path.parent.name,
                    "day_file": str(day_path),
                    "night_file": str(night_path),
                }
            )
    return pairs


def run_all_day_night_rgb_pair_alignments(
    dataset_folder: Path | str = "dataset",
    output_folder: Path | str = "day_night_alignment",
    write_preview: bool = True,
) -> dict[str, Any]:
    """Discover and align every split with a complete day/night RGB pair."""
    dataset_folder = Path(dataset_folder)
    output_folder = Path(output_folder)
    pairs = _discover_day_night_rgb_pairs(dataset_folder)
    aligned: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    for pair in pairs:
        sample_name = pair["sample"]
        split_folder_name = pair["split_folder_name"]
        try:
            result = run_day_night_rgb_pair_alignment(
                sample_name=sample_name,
                dataset_folder=dataset_folder,
                output_folder=output_folder / split_folder_name,
                split_folder_name=split_folder_name,
                write_preview=write_preview,
            )
            aligned.append(
                {
                    "sample": sample_name,
                    "split_folder_name": split_folder_name,
                    "coverage": result["coverage"],
                    "review_interval_count": len(result["review_intervals"]),
                    "outputs": result["outputs"],
                }
            )
        except Exception as exc:
            failed.append(
                {
                    "sample": sample_name,
                    "split_folder_name": split_folder_name,
                    "reason": str(exc),
                }
            )

    output_folder.mkdir(parents=True, exist_ok=True)
    summary_path = output_folder / "day_night_rgb_alignment_summary.json"
    summary = {
        "dataset_folder": str(dataset_folder),
        "output_folder": str(output_folder),
        "write_preview": write_preview,
        "discovered_count": len(pairs),
        "aligned_count": len(aligned),
        "failed_count": len(failed),
        "aligned": aligned,
        "failed": failed,
        "summary_file": str(summary_path),
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary
