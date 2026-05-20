"""Estimate temporal offsets between modalities using RGB as reference."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

import cv2
import numpy as np


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".mpeg", ".mpg"}
REFERENCE_MODALITY = "rgb"
TARGET_MODALITIES = ("event", "ir", "depth")
ALL_MODALITIES = (REFERENCE_MODALITY, *TARGET_MODALITIES)
DEFAULT_OUTPUT_PATH = Path("temporal_alignment_results.json")
DEFAULT_DAY_OUTPUT_PATH = Path("temporal_alignment_day_results.json")
DEFAULT_NIGHT_OUTPUT_PATH = Path("temporal_alignment_night_results.json")
DEFAULT_PLOT_OUTPUT_FOLDER = Path("temporal_alignment_plots")
DEFAULT_EXPORT_OUTPUT_FOLDER = Path("temporal_alignment_exports")
EXPORT_PANEL_WIDTH = 320
EXPORT_PANEL_HEIGHT = 180
EXPORT_PREVIEW_FPS = 10
EXPORT_LABEL_FONT_SIZE = 18

MODALITY_COLORS = {
    "rgb": (46, 105, 230),
    "event": (30, 150, 70),
    "ir": (190, 70, 170),
    "depth": (35, 160, 210),
}
AXIS_COLOR = (170, 170, 170)
TEXT_COLOR = (45, 45, 45)
GRID_COLOR = (230, 230, 230)
TOP_LAG_CANDIDATE_COUNT = 8
LOW_CONFIDENCE_LARGE_EVENT_OFFSET_SECONDS = 5.0
LOW_CONFIDENCE_LARGE_VISUAL_OFFSET_SECONDS = 3.0
IR_DEPTH_EXPECTED_OFFSET_SECONDS = 0.0

MODALITY_PRIORS: dict[str, dict[str, float]] = {
    "ir": {
        "max_lag_seconds": 3.0,
        "expected_offset_seconds": IR_DEPTH_EXPECTED_OFFSET_SECONDS,
        "prior_scale_seconds": 1.0,
        "prior_weight": 0.22,
    },
    "depth": {
        "max_lag_seconds": 3.0,
        "expected_offset_seconds": IR_DEPTH_EXPECTED_OFFSET_SECONDS,
        "prior_scale_seconds": 1.0,
        "prior_weight": 0.20,
    },
    "event": {
        "max_lag_seconds": 30.0,
        "expected_offset_seconds": 0.0,
        "prior_scale_seconds": 8.0,
        "prior_weight": 0.08,
    },
}


def _discover_modality_sets(dataset_folder: Path, side: str) -> list[dict[str, Any]]:
    samples: dict[str, dict[str, Any]] = {}
    side = side.lower()

    for file in dataset_folder.rglob("*"):
        if not file.is_file() or file.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        stem = file.stem.lower()
        for modality in ALL_MODALITIES:
            suffix = f"_{side}_{modality}"
            if not stem.endswith(suffix):
                continue

            pair_key = str(file.parent / file.stem[: -len(suffix)])
            sample = samples.setdefault(pair_key, {"pair_key": pair_key, "side": side, "videos": {}})
            sample["videos"][modality] = file
            break

    return [samples[pair_key] for pair_key in sorted(samples)]


def _video_metadata(video_path: Path) -> dict[str, float | int | bool]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"opened": False, "fps": 0.0, "frame_count": 0, "duration_seconds": 0.0}

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    duration_seconds = float(frame_count / fps) if fps > 0 else 0.0
    return {
        "opened": True,
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": duration_seconds,
    }


def _motion_energy_trace(video_path: Path, resize_width: int = 160) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    previous_gray: np.ndarray | None = None
    energies: list[float] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if resize_width > 0 and gray.shape[1] > resize_width:
            scale = resize_width / float(gray.shape[1])
            resize_height = max(1, int(round(gray.shape[0] * scale)))
            gray = cv2.resize(gray, (resize_width, resize_height), interpolation=cv2.INTER_AREA)

        gray = gray.astype(np.float32)
        if previous_gray is not None:
            energies.append(float(np.mean(np.abs(gray - previous_gray))))
        previous_gray = gray

    cap.release()
    return np.asarray(energies, dtype=np.float32)


def _normalize_for_plot(trace: np.ndarray) -> np.ndarray:
    if trace.size == 0:
        return trace
    low = float(np.percentile(trace, 1))
    high = float(np.percentile(trace, 99))
    if high <= low:
        high = float(np.max(trace))
        low = float(np.min(trace))
    if high <= low:
        return np.zeros_like(trace, dtype=np.float32)
    normalized = (trace.astype(np.float32) - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0)


def _times_for_trace(trace: np.ndarray, fps: float) -> np.ndarray:
    return np.arange(trace.size, dtype=np.float32) / max(fps, 1e-6)


def _trace_polyline(
    times: np.ndarray,
    values: np.ndarray,
    bounds: tuple[int, int, int, int],
    min_time: float,
    max_time: float,
) -> np.ndarray | None:
    if times.size == 0 or values.size == 0 or max_time <= min_time:
        return None

    left, top, right, bottom = bounds
    plot_width = max(1, right - left)
    plot_height = max(1, bottom - top)
    x_values = left + np.clip((times - min_time) / (max_time - min_time), 0.0, 1.0) * plot_width
    y_values = bottom - np.clip(values, 0.0, 1.0) * plot_height
    points = np.column_stack([x_values, y_values]).round().astype(np.int32)
    return points.reshape((-1, 1, 2))


def _draw_panel(
    canvas: np.ndarray,
    title: str,
    bounds: tuple[int, int, int, int],
    min_time: float,
    max_time: float,
    traces: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int], str]],
) -> None:
    left, top, right, bottom = bounds
    cv2.rectangle(canvas, (left, top), (right, bottom), AXIS_COLOR, 1)
    cv2.putText(canvas, title, (left, top - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1, cv2.LINE_AA)

    for tick in np.linspace(min_time, max_time, 5):
        fraction = (tick - min_time) / (max_time - min_time) if max_time > min_time else 0.0
        x = int(round(left + fraction * (right - left)))
        cv2.line(canvas, (x, top), (x, bottom), GRID_COLOR, 1)
        cv2.putText(
            canvas,
            f"{tick:.0f}s",
            (max(left, x - 18), bottom + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )

    legend_x = right - 210
    legend_y = top + 18
    for index, (_, _, color, label) in enumerate(traces):
        y = legend_y + index * 22
        cv2.line(canvas, (legend_x, y - 5), (legend_x + 28, y - 5), color, 2)
        cv2.putText(canvas, label, (legend_x + 36, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1, cv2.LINE_AA)

    for times, values, color, _ in traces:
        polyline = _trace_polyline(times, values, bounds, min_time, max_time)
        if polyline is not None and len(polyline) > 1:
            cv2.polylines(canvas, [polyline], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)


def _safe_plot_stem(pair_key: str) -> str:
    return Path(pair_key).name.replace(" ", "_")


def _time_range(trace_items: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int], str]]) -> tuple[float, float]:
    min_time = 0.0
    max_time = 1.0
    has_times = False
    for times, _, _, _ in trace_items:
        if times.size == 0:
            continue
        has_times = True
        min_time = min(min_time, float(np.min(times)))
        max_time = max(max_time, float(np.max(times)))
    if not has_times or max_time <= min_time:
        return 0.0, 1.0
    return min_time, max_time


def _format_alignment_summary(alignments: dict[str, dict[str, Any]]) -> str:
    parts = []
    for modality in TARGET_MODALITIES:
        item = alignments.get(modality, {})
        offset = item.get("offset_seconds")
        confidence = item.get("confidence_label", "low")
        if offset is None:
            parts.append(f"{modality}=unknown/{confidence}")
        else:
            parts.append(f"{modality}={offset:.3f}s/{confidence}")
    return "  ".join(parts)


def _write_activity_signal_plot(
    output_path: Path,
    title: str,
    raw_traces: dict[str, tuple[np.ndarray, float]],
    aligned_offsets: dict[str, float | None],
    alignments: dict[str, dict[str, Any]],
    modalities: tuple[str, ...],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 1500, 780
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    left, right = 90, width - 50
    raw_bounds = (left, 145, right, 335)
    aligned_bounds = (left, 485, right, 675)

    raw_items = []
    aligned_items = []
    for modality in modalities:
        trace, fps = raw_traces[modality]
        values = _normalize_for_plot(trace)
        raw_times = _times_for_trace(trace, fps)
        offset = aligned_offsets.get(modality)
        aligned_times = raw_times - offset if offset is not None else raw_times
        color = MODALITY_COLORS[modality]
        raw_items.append((raw_times, values, color, modality))
        aligned_label = modality if modality == REFERENCE_MODALITY else f"{modality} shifted"
        aligned_items.append((aligned_times, values, color, aligned_label))

    raw_min_time, raw_max_time = _time_range(raw_items)
    aligned_min_time, aligned_max_time = _time_range(aligned_items)

    cv2.putText(canvas, title, (left, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.86, TEXT_COLOR, 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        _format_alignment_summary(alignments),
        (left, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )

    _draw_panel(canvas, "Raw motion-energy traces", raw_bounds, raw_min_time, raw_max_time, raw_items)
    _draw_panel(canvas, "Aligned traces on RGB timeline", aligned_bounds, aligned_min_time, aligned_max_time, aligned_items)

    cv2.imwrite(str(output_path), canvas)


def _is_flat_trace(trace: np.ndarray) -> bool:
    return trace.size < 3 or float(np.std(trace)) < 1e-6


def _robust_standardize_trace(trace: np.ndarray) -> np.ndarray:
    if trace.size == 0:
        return trace.astype(np.float32)
    trace = trace.astype(np.float32)
    low = float(np.percentile(trace, 5))
    high = float(np.percentile(trace, 95))
    clipped = np.clip(trace, low, high)
    median = float(np.median(clipped))
    mad = float(np.median(np.abs(clipped - median)))
    if mad > 1e-6:
        return ((clipped - median) / (1.4826 * mad)).astype(np.float32)

    std = float(np.std(clipped))
    if std > 1e-6:
        return ((clipped - float(np.mean(clipped))) / std).astype(np.float32)
    return np.zeros_like(clipped, dtype=np.float32)


def _smooth_trace(trace: np.ndarray, fps: float, window_seconds: float = 0.35) -> np.ndarray:
    if trace.size < 3 or fps <= 0:
        return trace.astype(np.float32)
    window = max(3, int(round(window_seconds * fps)))
    if window % 2 == 0:
        window += 1
    window = min(window, trace.size if trace.size % 2 == 1 else trace.size - 1)
    if window < 3:
        return trace.astype(np.float32)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(trace.astype(np.float32), kernel, mode="same").astype(np.float32)


def _prepare_alignment_trace(trace: np.ndarray, fps: float) -> np.ndarray:
    return _smooth_trace(_robust_standardize_trace(trace), fps=fps)


def _overlap_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    if first.size < 3 or second.size < 3:
        return None

    first_std = float(np.std(first))
    second_std = float(np.std(second))
    if first_std < 1e-6 or second_std < 1e-6:
        return None

    first_z = (first - float(np.mean(first))) / first_std
    second_z = (second - float(np.mean(second))) / second_std
    return float(np.mean(first_z * second_z))


def _lag_overlaps(reference_trace: np.ndarray, target_trace: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag >= 0:
        overlap = min(reference_trace.size, target_trace.size - lag)
        if overlap <= 0:
            return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)
        return reference_trace[:overlap], target_trace[lag : lag + overlap]

    overlap = min(reference_trace.size + lag, target_trace.size)
    if overlap <= 0:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)
    return reference_trace[-lag : -lag + overlap], target_trace[:overlap]


def _confidence_from_correlation(correlation: float | None, peak_margin: float = 0.0) -> str:
    if correlation is None:
        return "low"
    if correlation >= 0.80 or (correlation >= 0.55 and peak_margin >= 0.02):
        return "high"
    if correlation >= 0.30:
        return "medium"
    return "low"


def _estimate_offset(
    reference_trace: np.ndarray,
    target_trace: np.ndarray,
    fps: float,
    max_lag_seconds: float,
    target_modality: str,
) -> dict[str, Any]:
    if _is_flat_trace(reference_trace) or _is_flat_trace(target_trace):
        return {
            "offset_seconds": None,
            "offset_frames": None,
            "peak_correlation": None,
            "confidence_label": "low",
            "candidate_offsets": [],
            "selected_by": "unavailable",
            "warnings": ["Motion energy trace is too short or too flat for reliable alignment."],
        }

    prior = MODALITY_PRIORS.get(target_modality, {})
    effective_max_lag_seconds = min(max_lag_seconds, float(prior.get("max_lag_seconds", max_lag_seconds)))
    expected_offset = float(prior.get("expected_offset_seconds", 0.0))
    prior_scale = max(1e-6, float(prior.get("prior_scale_seconds", 5.0)))
    prior_weight = float(prior.get("prior_weight", 0.0))
    prepared_reference = _prepare_alignment_trace(reference_trace, fps=fps)
    prepared_target = _prepare_alignment_trace(target_trace, fps=fps)
    max_lag_frames = min(
        int(round(effective_max_lag_seconds * fps)),
        max(0, prepared_reference.size - 3),
        max(0, prepared_target.size - 3),
    )

    candidates: list[dict[str, Any]] = []
    max_overlap = max(1, min(prepared_reference.size, prepared_target.size))
    for lag in range(-max_lag_frames, max_lag_frames + 1):
        reference_overlap, target_overlap = _lag_overlaps(prepared_reference, prepared_target, lag)
        correlation = _overlap_correlation(reference_overlap, target_overlap)
        if correlation is None:
            continue

        offset_seconds = float(lag / fps)
        overlap_ratio = float(reference_overlap.size / max_overlap)
        distance_penalty = min(1.0, abs(offset_seconds - expected_offset) / prior_scale)
        score = float(correlation + 0.12 * overlap_ratio - prior_weight * distance_penalty)
        candidates.append(
            {
                "offset_seconds": offset_seconds,
                "offset_frames": int(lag),
                "correlation": float(correlation),
                "overlap_ratio": overlap_ratio,
                "score": score,
            }
        )

    if not candidates:
        return {
            "offset_seconds": None,
            "offset_frames": None,
            "peak_correlation": None,
            "confidence_label": "low",
            "candidate_offsets": [],
            "selected_by": "unavailable",
            "warnings": ["No valid cross-correlation overlap was available."],
        }

    candidates_by_correlation = sorted(candidates, key=lambda item: item["correlation"], reverse=True)
    candidates_by_score = sorted(candidates, key=lambda item: item["score"], reverse=True)
    selected = candidates_by_score[0]
    best_by_correlation = candidates_by_correlation[0]
    second_correlation = candidates_by_correlation[1]["correlation"] if len(candidates_by_correlation) > 1 else None
    peak_margin = (
        float(selected["correlation"] - second_correlation)
        if second_correlation is not None and selected["offset_frames"] == best_by_correlation["offset_frames"]
        else 0.0
    )
    selected_by = (
        "best_correlation"
        if selected["offset_frames"] == best_by_correlation["offset_frames"]
        else "modality_prior"
    )
    confidence = _confidence_from_correlation(float(selected["correlation"]), peak_margin=peak_margin)
    warnings: list[str] = []
    if (
        selected_by == "modality_prior"
        and abs(float(selected["offset_seconds"]) - float(best_by_correlation["offset_seconds"])) > 0.5
    ):
        warnings.append(
            "Selected offset uses modality prior because the raw highest-correlation lag was less plausible."
        )

    return {
        "offset_seconds": float(selected["offset_seconds"]),
        "offset_frames": int(selected["offset_frames"]),
        "peak_correlation": float(selected["correlation"]),
        "confidence_label": confidence,
        "candidate_offsets": [
            {
                "offset_seconds": round(float(item["offset_seconds"]), 6),
                "offset_frames": int(item["offset_frames"]),
                "correlation": round(float(item["correlation"]), 6),
                "score": round(float(item["score"]), 6),
                "overlap_ratio": round(float(item["overlap_ratio"]), 6),
            }
            for item in candidates_by_score[:TOP_LAG_CANDIDATE_COUNT]
        ],
        "selected_by": selected_by,
        "raw_best_offset_seconds": round(float(best_by_correlation["offset_seconds"]), 6),
        "raw_best_correlation": float(best_by_correlation["correlation"]),
        "peak_margin": float(peak_margin),
        "lag_fps": float(fps),
        "warnings": warnings,
    }


def _overlap_windows(
    reference_duration: float,
    target_duration: float,
    offset_seconds: float | None,
) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    if offset_seconds is None:
        return None, None

    reference_start = max(0.0, -offset_seconds)
    target_start = max(0.0, offset_seconds)
    overlap_duration = max(0.0, min(reference_duration - reference_start, target_duration - target_start))
    return (
        {"start": round(reference_start, 6), "end": round(reference_start + overlap_duration, 6)},
        {"start": round(target_start, 6), "end": round(target_start + overlap_duration, 6)},
    )


def _set_alignment_offset(
    alignment: dict[str, Any],
    offset_seconds: float | None,
    selected_by: str,
    reference_duration: float,
) -> None:
    alignment["offset_seconds"] = float(offset_seconds) if offset_seconds is not None else None
    lag_fps = float(alignment.get("lag_fps") or 30.0)
    alignment["offset_frames"] = int(round(offset_seconds * lag_fps)) if offset_seconds is not None else None
    alignment["selected_by"] = selected_by
    reference_overlap, target_overlap = _overlap_windows(
        reference_duration,
        float(alignment.get("duration_seconds") or 0.0),
        alignment["offset_seconds"],
    )
    alignment["overlap_reference_seconds"] = reference_overlap
    alignment["overlap_target_seconds"] = target_overlap


def _nearest_candidate(
    candidates: list[dict[str, Any]],
    target_offset: float,
    max_distance_seconds: float,
) -> dict[str, Any] | None:
    nearby = [
        candidate
        for candidate in candidates
        if abs(float(candidate.get("offset_seconds", 0.0)) - target_offset) <= max_distance_seconds
    ]
    if not nearby:
        return None
    return max(nearby, key=lambda item: float(item.get("score", item.get("correlation", -1.0))))


def _apply_consensus_corrections(result: dict[str, Any]) -> None:
    alignments = result.get("alignments", {})
    if not isinstance(alignments, dict):
        return

    reference_duration = float(result.get("reference_duration_seconds") or 0.0)
    ir_alignment = alignments.get("ir", {})
    ir_offset = ir_alignment.get("offset_seconds") if isinstance(ir_alignment, dict) else None
    ir_confidence = ir_alignment.get("confidence_label") if isinstance(ir_alignment, dict) else None
    if isinstance(ir_offset, (int, float)) and ir_confidence in {"high", "medium"} and abs(float(ir_offset)) <= 1.0:
        anchor_offset = float(ir_offset)
    else:
        anchor_offset = 0.0

    depth_alignment = alignments.get("depth", {})
    if isinstance(depth_alignment, dict):
        depth_offset = depth_alignment.get("offset_seconds")
        depth_peak = depth_alignment.get("peak_correlation")
        depth_confidence = depth_alignment.get("confidence_label")
        if (
            isinstance(depth_offset, (int, float))
            and abs(float(depth_offset) - anchor_offset) > LOW_CONFIDENCE_LARGE_VISUAL_OFFSET_SECONDS
            and (depth_confidence == "low" or not isinstance(depth_peak, (int, float)) or float(depth_peak) < 0.55)
        ):
            rejected_offset = float(depth_offset)
            candidate = _nearest_candidate(
                list(depth_alignment.get("candidate_offsets") or []),
                anchor_offset,
                LOW_CONFIDENCE_LARGE_VISUAL_OFFSET_SECONDS,
            )
            fallback_offset = float(candidate["offset_seconds"]) if candidate else anchor_offset
            if candidate:
                depth_alignment["peak_correlation"] = float(candidate.get("correlation", depth_peak or 0.0))
            _set_alignment_offset(depth_alignment, fallback_offset, "consensus_fallback", reference_duration)
            depth_alignment["confidence_label"] = "medium" if candidate else "low"
            depth_alignment.setdefault("warnings", []).append(
                f"Large low-confidence depth offset {rejected_offset:.3f}s rejected; "
                f"using consensus fallback {fallback_offset:.3f}s."
            )
            depth_alignment["rejected_offset_seconds"] = rejected_offset

    event_alignment = alignments.get("event", {})
    if isinstance(event_alignment, dict):
        event_offset = event_alignment.get("offset_seconds")
        event_peak = event_alignment.get("peak_correlation")
        event_confidence = event_alignment.get("confidence_label")
        if (
            isinstance(event_offset, (int, float))
            and abs(float(event_offset) - anchor_offset) > LOW_CONFIDENCE_LARGE_EVENT_OFFSET_SECONDS
            and (event_confidence == "low" or not isinstance(event_peak, (int, float)) or float(event_peak) < 0.30)
        ):
            rejected_offset = float(event_offset)
            candidate = _nearest_candidate(
                list(event_alignment.get("candidate_offsets") or []),
                anchor_offset,
                LOW_CONFIDENCE_LARGE_EVENT_OFFSET_SECONDS,
            )
            fallback_offset = float(candidate["offset_seconds"]) if candidate else anchor_offset
            if candidate:
                event_alignment["peak_correlation"] = float(candidate.get("correlation", event_peak or 0.0))
            _set_alignment_offset(event_alignment, fallback_offset, "consensus_fallback", reference_duration)
            event_alignment["confidence_label"] = "medium" if candidate else "low"
            event_alignment.setdefault("warnings", []).append(
                f"Large low-confidence event offset {rejected_offset:.3f}s rejected; "
                f"using consensus fallback {fallback_offset:.3f}s."
            )
            event_alignment["rejected_offset_seconds"] = rejected_offset


def _missing_alignment(modality: str, message: str) -> dict[str, Any]:
    return {
        "modality": modality,
        "file": None,
        "duration_seconds": None,
        "offset_seconds": None,
        "offset_frames": None,
        "peak_correlation": None,
        "confidence_label": "low",
        "overlap_reference_seconds": None,
        "overlap_target_seconds": None,
        "activity_plot_file": None,
        "candidate_offsets": [],
        "selected_by": "missing",
        "warnings": [message],
    }


def _estimate_modality_alignment(
    reference_file: Path,
    reference_meta: dict[str, Any],
    reference_trace: np.ndarray,
    target_modality: str,
    target_file: Path,
    target_trace: np.ndarray,
    max_lag_seconds: float,
    plot_output_path: Path | None = None,
) -> dict[str, Any]:
    warnings: list[str] = []
    target_meta = _video_metadata(target_file)
    result: dict[str, Any] = {
        "modality": target_modality,
        "file": str(target_file),
        "duration_seconds": float(target_meta["duration_seconds"]),
        "offset_seconds": None,
        "offset_frames": None,
        "peak_correlation": None,
        "confidence_label": "low",
        "overlap_reference_seconds": None,
        "overlap_target_seconds": None,
        "activity_plot_file": str(plot_output_path) if plot_output_path else None,
        "candidate_offsets": [],
        "selected_by": "unavailable",
        "warnings": warnings,
    }

    if not target_meta["opened"]:
        warnings.append(f"Could not open {target_modality} video: {target_file}")
        return result

    reference_fps = float(reference_meta["fps"])
    target_fps = float(target_meta["fps"])
    if reference_fps <= 0 or target_fps <= 0:
        warnings.append("Could not determine FPS for one or both videos.")
        return result
    if abs(reference_fps - target_fps) > 0.01:
        warnings.append(
            f"FPS differs between videos: RGB={reference_fps:.3f}, "
            f"{target_modality}={target_fps:.3f}. Offset is estimated using the lower FPS as the lag unit."
        )

    lag_fps = min(reference_fps, target_fps)
    estimate = _estimate_offset(reference_trace, target_trace, lag_fps, max_lag_seconds, target_modality)
    warnings.extend(estimate.pop("warnings", []))
    result.update(estimate)

    reference_overlap, target_overlap = _overlap_windows(
        float(reference_meta["duration_seconds"]),
        float(target_meta["duration_seconds"]),
        result["offset_seconds"],
    )
    result["overlap_reference_seconds"] = reference_overlap
    result["overlap_target_seconds"] = target_overlap

    if plot_output_path is not None:
        raw_traces = {
            REFERENCE_MODALITY: (reference_trace, reference_fps),
            target_modality: (target_trace, target_fps),
        }
        alignments = {target_modality: result}
        _write_activity_signal_plot(
            plot_output_path,
            f"Activity signal: {reference_file.parent.name}/{reference_file.stem} vs {target_modality}",
            raw_traces,
            {REFERENCE_MODALITY: 0.0, target_modality: result["offset_seconds"]},
            alignments,
            (REFERENCE_MODALITY, target_modality),
        )

    return result


def run_temporal_alignment(
    dataset_folder: Path | str = "dataset",
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    plot_output_folder: Path | str | None = DEFAULT_PLOT_OUTPUT_FOLDER,
    side: str = "day",
    max_lag_seconds: float = 30.0,
    resize_width: int = 160,
) -> list[dict[str, Any]]:
    dataset_folder = Path(dataset_folder)
    output_path = Path(output_path)
    plot_output_folder = Path(plot_output_folder) if plot_output_folder is not None else None
    side = side.lower()
    if side not in {"day", "night"}:
        raise ValueError(f"Unsupported temporal alignment side: {side}")
    results: list[dict[str, Any]] = []

    for sample in _discover_modality_sets(dataset_folder, side=side):
        pair_key = str(sample["pair_key"])
        videos: dict[str, Path] = sample["videos"]
        reference_file = videos.get(REFERENCE_MODALITY)
        warnings: list[str] = []
        print(f"Aligning {side} modalities against RGB: {pair_key}")

        if reference_file is None:
            warnings.append(f"Missing {side} RGB reference video.")
            results.append(
                {
                    "pair_key": pair_key,
                    "side": side,
                    "reference_modality": REFERENCE_MODALITY,
                    "reference_file": None,
                    "reference_duration_seconds": None,
                    "combined_activity_plot_file": None,
                    "warnings": warnings,
                    "alignments": {
                        modality: _missing_alignment(modality, f"Missing {side} RGB reference video.")
                        for modality in TARGET_MODALITIES
                    },
                }
            )
            continue

        reference_meta = _video_metadata(reference_file)
        result: dict[str, Any] = {
            "pair_key": pair_key,
            "side": side,
            "reference_modality": REFERENCE_MODALITY,
            "reference_file": str(reference_file),
            "reference_duration_seconds": float(reference_meta["duration_seconds"]),
            "combined_activity_plot_file": None,
            "warnings": warnings,
            "alignments": {},
        }

        if not reference_meta["opened"]:
            warnings.append(f"Could not open RGB reference video: {reference_file}")
            for modality in TARGET_MODALITIES:
                result["alignments"][modality] = _missing_alignment(modality, "RGB reference video could not be opened.")
            results.append(result)
            continue

        reference_fps = float(reference_meta["fps"])
        if reference_fps <= 0:
            warnings.append("Could not determine FPS for RGB reference video.")
            for modality in TARGET_MODALITIES:
                result["alignments"][modality] = _missing_alignment(modality, "RGB reference FPS is unavailable.")
            results.append(result)
            continue

        reference_trace = _motion_energy_trace(reference_file, resize_width=resize_width)
        trace_cache: dict[str, tuple[np.ndarray, float]] = {REFERENCE_MODALITY: (reference_trace, reference_fps)}
        alignment_offsets: dict[str, float | None] = {REFERENCE_MODALITY: 0.0}
        pair_stem = _safe_plot_stem(pair_key)

        for modality in TARGET_MODALITIES:
            target_file = videos.get(modality)
            if target_file is None:
                result["alignments"][modality] = _missing_alignment(modality, f"Missing {side} {modality} video.")
                alignment_offsets[modality] = None
                continue

            target_meta = _video_metadata(target_file)
            target_fps = float(target_meta["fps"]) if target_meta["opened"] else 0.0
            target_trace = _motion_energy_trace(target_file, resize_width=resize_width) if target_meta["opened"] else np.asarray([], dtype=np.float32)
            if target_meta["opened"]:
                trace_cache[modality] = (target_trace, target_fps)

            plot_path = None
            if plot_output_folder is not None:
                plot_path = plot_output_folder / f"{pair_stem}_{side}_rgb_{modality}_activity_signal.png"

            alignment = _estimate_modality_alignment(
                reference_file=reference_file,
                reference_meta=reference_meta,
                reference_trace=reference_trace,
                target_modality=modality,
                target_file=target_file,
                target_trace=target_trace,
                max_lag_seconds=max_lag_seconds,
                plot_output_path=plot_path,
            )
            result["alignments"][modality] = alignment
            alignment_offsets[modality] = alignment.get("offset_seconds")

        _apply_consensus_corrections(result)
        for modality in TARGET_MODALITIES:
            alignment_offsets[modality] = result["alignments"].get(modality, {}).get("offset_seconds")

        if plot_output_folder is not None:
            combined_plot_path = plot_output_folder / f"{pair_stem}_{side}_activity_signal_all.png"
            available_modalities = tuple(modality for modality in ALL_MODALITIES if modality in trace_cache)
            _write_activity_signal_plot(
                combined_plot_path,
                f"Activity signal: {Path(pair_key).name} ({side})",
                trace_cache,
                alignment_offsets,
                result["alignments"],
                available_modalities,
            )
            result["combined_activity_plot_file"] = str(combined_plot_path)

        results.append(result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    return results


def run_day_night_temporal_alignment(
    dataset_folder: Path | str = "dataset",
    day_output_path: Path | str = DEFAULT_DAY_OUTPUT_PATH,
    night_output_path: Path | str = DEFAULT_NIGHT_OUTPUT_PATH,
    plot_output_folder: Path | str | None = DEFAULT_PLOT_OUTPUT_FOLDER,
    max_lag_seconds: float = 30.0,
    resize_width: int = 160,
) -> dict[str, list[dict[str, Any]]]:
    return {
        "day": run_temporal_alignment(
            dataset_folder=dataset_folder,
            output_path=day_output_path,
            plot_output_folder=plot_output_folder,
            side="day",
            max_lag_seconds=max_lag_seconds,
            resize_width=resize_width,
        ),
        "night": run_temporal_alignment(
            dataset_folder=dataset_folder,
            output_path=night_output_path,
            plot_output_folder=plot_output_folder,
            side="night",
            max_lag_seconds=max_lag_seconds,
            resize_width=resize_width,
        ),
    }


def _load_alignment_results(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a top-level JSON list in {path}")
    return [item for item in data if isinstance(item, dict)]


def _ffmpeg_escape_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")


def _encoder_args(prefer_gpu: bool) -> list[str]:
    if prefer_gpu:
        return ["-c:v", "h264_nvenc", "-preset", "p1", "-cq", "35"]
    return ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "35"]


def _run_ffmpeg_with_optional_gpu(command_prefix: list[str], output_path: Path, prefer_gpu: bool) -> tuple[str | None, str]:
    tmp_output_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    tmp_output_path.unlink(missing_ok=True)
    encoder_used = "h264_nvenc" if prefer_gpu else "libx264"
    command = [*command_prefix, *_encoder_args(prefer_gpu), "-movflags", "+faststart", str(tmp_output_path)]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode == 0:
        tmp_output_path.replace(output_path)
        return encoder_used, ""

    if prefer_gpu:
        tmp_output_path.unlink(missing_ok=True)
        encoder_used = "libx264"
        fallback_command = [*command_prefix, *_encoder_args(False), "-movflags", "+faststart", str(tmp_output_path)]
        fallback = subprocess.run(fallback_command, check=False, capture_output=True, text=True)
        if fallback.returncode == 0:
            tmp_output_path.replace(output_path)
            return encoder_used, completed.stderr.strip()
        tmp_output_path.unlink(missing_ok=True)
        return None, fallback.stderr.strip() or completed.stderr.strip()

    tmp_output_path.unlink(missing_ok=True)
    return None, completed.stderr.strip()


def _sample_name_from_alignment(sample: dict[str, Any]) -> str:
    pair_key = str(sample.get("pair_key") or "sample")
    return Path(pair_key).name or "sample"


def _read_target_alignment(sample: dict[str, Any], modality: str) -> dict[str, Any]:
    alignment = sample.get("alignments", {}).get(modality, {})
    if not isinstance(alignment, dict):
        raise ValueError(f"Missing {modality.upper()} alignment.")

    file_path = Path(str(alignment.get("file", "")))
    offset = alignment.get("offset_seconds")
    duration = alignment.get("duration_seconds")
    if offset is None or duration is None:
        raise ValueError(f"{modality.upper()} alignment must include offset_seconds and duration_seconds.")
    if not file_path.exists():
        raise FileNotFoundError(f"{modality.upper()} file does not exist: {file_path}")

    return {
        "file": file_path,
        "offset": float(offset),
        "duration": float(duration),
    }


def _export_rgb_event_depth_ir_grid_for_sample(
    sample: dict[str, Any],
    output_folder: Path,
    prefer_gpu: bool,
) -> dict[str, Any]:
    sample_name = _sample_name_from_alignment(sample)
    side = str(sample.get("side") or "unknown")
    output_path = output_folder / f"{sample_name}_{side}_rgb_event_depth_ir_aligned.mp4"

    reference_file = Path(str(sample.get("reference_file", "")))
    reference_duration = float(sample.get("reference_duration_seconds") or 0.0)
    if not reference_file.exists():
        raise FileNotFoundError(f"RGB reference file does not exist: {reference_file}")
    if reference_duration <= 0:
        raise ValueError("RGB reference_duration_seconds must be positive.")

    targets = {
        "event": _read_target_alignment(sample, "event"),
        "depth": _read_target_alignment(sample, "depth"),
        "ir": _read_target_alignment(sample, "ir"),
    }
    reference_start = max(0.0, *(-target["offset"] for target in targets.values()))
    reference_end = min(
        reference_duration,
        *(target["duration"] - target["offset"] for target in targets.values()),
    )
    duration = max(0.0, reference_end - reference_start)
    if duration <= 0:
        raise ValueError("No positive RGB/EVENT/DEPTH/IR overlap window is available.")

    rgb_seek = reference_start
    event_seek = max(0.0, reference_start + targets["event"]["offset"])
    depth_seek = max(0.0, reference_start + targets["depth"]["offset"])
    ir_seek = max(0.0, reference_start + targets["ir"]["offset"])
    rgb_label = _ffmpeg_escape_text("RGB")
    event_label = _ffmpeg_escape_text(f"EVENT offset {targets['event']['offset']:.3f}s")
    depth_label = _ffmpeg_escape_text(f"DEPTH offset {targets['depth']['offset']:.3f}s")
    ir_label = _ffmpeg_escape_text(f"IR offset {targets['ir']['offset']:.3f}s")
    filter_complex = (
        f"[0:v]fps={EXPORT_PREVIEW_FPS},"
        f"scale={EXPORT_PANEL_WIDTH}:{EXPORT_PANEL_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={EXPORT_PANEL_WIDTH}:{EXPORT_PANEL_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
        f"drawtext=text='{rgb_label}':x=12:y=12:fontsize={EXPORT_LABEL_FONT_SIZE}:fontcolor=white:"
        f"box=1:boxcolor=black@0.55:boxborderw=8,setsar=1,setpts=PTS-STARTPTS[rgb];"
        f"[1:v]fps={EXPORT_PREVIEW_FPS},"
        f"scale={EXPORT_PANEL_WIDTH}:{EXPORT_PANEL_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={EXPORT_PANEL_WIDTH}:{EXPORT_PANEL_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
        f"drawtext=text='{event_label}':x=12:y=12:fontsize={EXPORT_LABEL_FONT_SIZE}:fontcolor=white:"
        f"box=1:boxcolor=black@0.55:boxborderw=8,setsar=1,setpts=PTS-STARTPTS[event];"
        f"[2:v]fps={EXPORT_PREVIEW_FPS},"
        f"scale={EXPORT_PANEL_WIDTH}:{EXPORT_PANEL_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={EXPORT_PANEL_WIDTH}:{EXPORT_PANEL_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
        f"drawtext=text='{depth_label}':x=12:y=12:fontsize={EXPORT_LABEL_FONT_SIZE}:fontcolor=white:"
        f"box=1:boxcolor=black@0.55:boxborderw=8,setsar=1,setpts=PTS-STARTPTS[depth];"
        f"[3:v]fps={EXPORT_PREVIEW_FPS},"
        f"scale={EXPORT_PANEL_WIDTH}:{EXPORT_PANEL_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={EXPORT_PANEL_WIDTH}:{EXPORT_PANEL_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
        f"drawtext=text='{ir_label}':x=12:y=12:fontsize={EXPORT_LABEL_FONT_SIZE}:fontcolor=white:"
        f"box=1:boxcolor=black@0.55:boxborderw=8,setsar=1,setpts=PTS-STARTPTS[ir];"
        "[rgb][event]hstack=inputs=2[top];"
        "[depth][ir]hstack=inputs=2[bottom];"
        "[top][bottom]vstack=inputs=2,format=yuv420p[outv]"
    )
    command_prefix = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{rgb_seek:.6f}",
        "-t",
        f"{duration:.6f}",
        "-i",
        str(reference_file),
        "-ss",
        f"{event_seek:.6f}",
        "-t",
        f"{duration:.6f}",
        "-i",
        str(targets["event"]["file"]),
        "-ss",
        f"{depth_seek:.6f}",
        "-t",
        f"{duration:.6f}",
        "-i",
        str(targets["depth"]["file"]),
        "-ss",
        f"{ir_seek:.6f}",
        "-t",
        f"{duration:.6f}",
        "-i",
        str(targets["ir"]["file"]),
        "-filter_complex",
        filter_complex,
        "-map",
        "[outv]",
        "-an",
    ]
    encoder_used, error = _run_ffmpeg_with_optional_gpu(command_prefix, output_path, prefer_gpu=prefer_gpu)
    if encoder_used is None:
        raise RuntimeError(f"Failed to export aligned RGB/EVENT/DEPTH/IR video: {error}")

    return {
        "sample": sample_name,
        "side": side,
        "output_file": str(output_path),
        "reference_file": str(reference_file),
        "event_file": str(targets["event"]["file"]),
        "depth_file": str(targets["depth"]["file"]),
        "ir_file": str(targets["ir"]["file"]),
        "event_offset_seconds": targets["event"]["offset"],
        "depth_offset_seconds": targets["depth"]["offset"],
        "ir_offset_seconds": targets["ir"]["offset"],
        "rgb_seek_seconds": round(rgb_seek, 6),
        "event_seek_seconds": round(event_seek, 6),
        "depth_seek_seconds": round(depth_seek, 6),
        "ir_seek_seconds": round(ir_seek, 6),
        "duration_seconds": round(duration, 6),
        "encoder": encoder_used,
        "gpu_fallback_warning": error if encoder_used == "libx264" and error else None,
    }


def export_day_night_rgb_event_depth_ir_alignment_grids(
    day_alignment_input_path: Path | str = DEFAULT_DAY_OUTPUT_PATH,
    night_alignment_input_path: Path | str = DEFAULT_NIGHT_OUTPUT_PATH,
    output_folder: Path | str = DEFAULT_EXPORT_OUTPUT_FOLDER,
    prefer_gpu: bool = True,
) -> dict[str, Any]:
    """Export all day/night RGB/EVENT/DEPTH/IR alignment previews from JSON offsets."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    exported: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for side, input_path in (("day", Path(day_alignment_input_path)), ("night", Path(night_alignment_input_path))):
        for sample in _load_alignment_results(input_path):
            try:
                exported.append(
                    _export_rgb_event_depth_ir_grid_for_sample(
                        sample=sample,
                        output_folder=output_folder,
                        prefer_gpu=prefer_gpu,
                    )
                )
            except Exception as exc:
                skipped.append(
                    {
                        "sample": _sample_name_from_alignment(sample),
                        "side": str(sample.get("side") or side),
                        "reason": str(exc),
                    }
                )

    summary = {
        "exported_count": len(exported),
        "skipped_count": len(skipped),
        "exported": exported,
        "skipped": skipped,
    }
    summary_path = output_folder / "rgb_event_depth_ir_export_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    summary["summary_file"] = str(summary_path)
    return summary


def export_check_mailbox_day_rgb_ir_event_alignment(
    alignment_input_path: Path | str = DEFAULT_DAY_OUTPUT_PATH,
    output_folder: Path | str = DEFAULT_EXPORT_OUTPUT_FOLDER,
    prefer_gpu: bool = True,
) -> dict[str, Any]:
    """Export a day RGB/IR/EVENT video for check_mailbox using stored offsets."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / "check_mailbox_day_rgb_ir_event_aligned.mp4"

    sample = None
    for item in _load_alignment_results(Path(alignment_input_path)):
        if item.get("pair_key") == "dataset/check_mailbox_split/check_mailbox" and item.get("side") == "day":
            sample = item
            break
    if sample is None:
        raise ValueError("Could not find day check_mailbox alignment in temporal alignment JSON.")

    reference_file = Path(str(sample.get("reference_file", "")))
    reference_duration = float(sample.get("reference_duration_seconds") or 0.0)
    ir_alignment = sample.get("alignments", {}).get("ir", {})
    if not isinstance(ir_alignment, dict):
        raise ValueError("Missing IR alignment for day check_mailbox.")
    event_alignment = sample.get("alignments", {}).get("event", {})
    if not isinstance(event_alignment, dict):
        raise ValueError("Missing EVENT alignment for day check_mailbox.")

    ir_file = Path(str(ir_alignment.get("file", "")))
    ir_offset = ir_alignment.get("offset_seconds")
    ir_duration = ir_alignment.get("duration_seconds")
    if ir_offset is None or ir_duration is None:
        raise ValueError("IR alignment must include offset_seconds and duration_seconds.")
    ir_offset = float(ir_offset)
    ir_duration = float(ir_duration)
    event_file = Path(str(event_alignment.get("file", "")))
    event_offset = event_alignment.get("offset_seconds")
    event_duration = event_alignment.get("duration_seconds")
    if event_offset is None or event_duration is None:
        raise ValueError("EVENT alignment must include offset_seconds and duration_seconds.")
    event_offset = float(event_offset)
    event_duration = float(event_duration)

    if not reference_file.exists():
        raise FileNotFoundError(f"RGB reference file does not exist: {reference_file}")
    if not ir_file.exists():
        raise FileNotFoundError(f"IR file does not exist: {ir_file}")
    if not event_file.exists():
        raise FileNotFoundError(f"EVENT file does not exist: {event_file}")

    reference_start = max(0.0, -ir_offset, -event_offset)
    reference_end = min(reference_duration, ir_duration - ir_offset, event_duration - event_offset)
    duration = max(0.0, reference_end - reference_start)
    if duration <= 0:
        raise ValueError("No positive RGB/IR/EVENT overlap window is available.")

    rgb_seek = reference_start
    ir_seek = max(0.0, reference_start + ir_offset)
    event_seek = max(0.0, reference_start + event_offset)
    rgb_label = _ffmpeg_escape_text("RGB")
    ir_label = _ffmpeg_escape_text(f"IR offset {ir_offset:.3f}s")
    event_label = _ffmpeg_escape_text(f"EVENT offset {event_offset:.3f}s")
    filter_complex = (
        f"[0:v]fps={EXPORT_PREVIEW_FPS},"
        f"scale={EXPORT_PANEL_WIDTH}:{EXPORT_PANEL_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={EXPORT_PANEL_WIDTH}:{EXPORT_PANEL_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
        f"drawtext=text='{rgb_label}':x=12:y=12:fontsize={EXPORT_LABEL_FONT_SIZE}:fontcolor=white:"
        f"box=1:boxcolor=black@0.55:boxborderw=8,setsar=1,setpts=PTS-STARTPTS[rgb];"
        f"[1:v]fps={EXPORT_PREVIEW_FPS},"
        f"scale={EXPORT_PANEL_WIDTH}:{EXPORT_PANEL_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={EXPORT_PANEL_WIDTH}:{EXPORT_PANEL_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
        f"drawtext=text='{ir_label}':x=12:y=12:fontsize={EXPORT_LABEL_FONT_SIZE}:fontcolor=white:"
        f"box=1:boxcolor=black@0.55:boxborderw=8,setsar=1,setpts=PTS-STARTPTS[ir];"
        f"[2:v]fps={EXPORT_PREVIEW_FPS},"
        f"scale={EXPORT_PANEL_WIDTH}:{EXPORT_PANEL_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={EXPORT_PANEL_WIDTH}:{EXPORT_PANEL_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
        f"drawtext=text='{event_label}':x=12:y=12:fontsize={EXPORT_LABEL_FONT_SIZE}:fontcolor=white:"
        f"box=1:boxcolor=black@0.55:boxborderw=8,setsar=1,setpts=PTS-STARTPTS[event];"
        "[rgb][ir][event]hstack=inputs=3,format=yuv420p[outv]"
    )
    command_prefix = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{rgb_seek:.6f}",
        "-t",
        f"{duration:.6f}",
        "-i",
        str(reference_file),
        "-ss",
        f"{ir_seek:.6f}",
        "-t",
        f"{duration:.6f}",
        "-i",
        str(ir_file),
        "-ss",
        f"{event_seek:.6f}",
        "-t",
        f"{duration:.6f}",
        "-i",
        str(event_file),
        "-filter_complex",
        filter_complex,
        "-map",
        "[outv]",
        "-an",
    ]
    encoder_used, error = _run_ffmpeg_with_optional_gpu(command_prefix, output_path, prefer_gpu=prefer_gpu)
    if encoder_used is None:
        raise RuntimeError(f"Failed to export aligned RGB/IR/EVENT video: {error}")

    return {
        "output_file": str(output_path),
        "reference_file": str(reference_file),
        "ir_file": str(ir_file),
        "event_file": str(event_file),
        "ir_offset_seconds": ir_offset,
        "event_offset_seconds": event_offset,
        "rgb_seek_seconds": round(rgb_seek, 6),
        "ir_seek_seconds": round(ir_seek, 6),
        "event_seek_seconds": round(event_seek, 6),
        "duration_seconds": round(duration, 6),
        "encoder": encoder_used,
        "gpu_fallback_warning": error if encoder_used == "libx264" and error else None,
    }


def export_check_mailbox_day_rgb_ir_alignment(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return export_check_mailbox_day_rgb_ir_event_alignment(*args, **kwargs)


def export_cut_carrot_day_rgb_ir_event_depth_alignment(
    alignment_input_path: Path | str = DEFAULT_DAY_OUTPUT_PATH,
    output_folder: Path | str = DEFAULT_EXPORT_OUTPUT_FOLDER,
    prefer_gpu: bool = True,
) -> dict[str, Any]:
    """Export a day RGB/IR/EVENT/DEPTH video for cut_carrot using stored offsets."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / "cut_carrot_day_rgb_ir_event_depth_aligned.mp4"

    sample = None
    for item in _load_alignment_results(Path(alignment_input_path)):
        if item.get("pair_key") == "dataset/cut_carrot_split/cut_carrot" and item.get("side") == "day":
            sample = item
            break
    if sample is None:
        raise ValueError("Could not find day cut_carrot alignment in temporal alignment JSON.")

    reference_file = Path(str(sample.get("reference_file", "")))
    reference_duration = float(sample.get("reference_duration_seconds") or 0.0)
    ir_alignment = sample.get("alignments", {}).get("ir", {})
    if not isinstance(ir_alignment, dict):
        raise ValueError("Missing IR alignment for day cut_carrot.")
    event_alignment = sample.get("alignments", {}).get("event", {})
    if not isinstance(event_alignment, dict):
        raise ValueError("Missing EVENT alignment for day cut_carrot.")
    depth_alignment = sample.get("alignments", {}).get("depth", {})
    if not isinstance(depth_alignment, dict):
        raise ValueError("Missing DEPTH alignment for day cut_carrot.")

    ir_file = Path(str(ir_alignment.get("file", "")))
    ir_offset = ir_alignment.get("offset_seconds")
    ir_duration = ir_alignment.get("duration_seconds")
    if ir_offset is None or ir_duration is None:
        raise ValueError("IR alignment must include offset_seconds and duration_seconds.")
    ir_offset = float(ir_offset)
    ir_duration = float(ir_duration)
    event_file = Path(str(event_alignment.get("file", "")))
    event_offset = event_alignment.get("offset_seconds")
    event_duration = event_alignment.get("duration_seconds")
    if event_offset is None or event_duration is None:
        raise ValueError("EVENT alignment must include offset_seconds and duration_seconds.")
    event_offset = float(event_offset)
    event_duration = float(event_duration)
    depth_file = Path(str(depth_alignment.get("file", "")))
    depth_offset = depth_alignment.get("offset_seconds")
    depth_duration = depth_alignment.get("duration_seconds")
    if depth_offset is None or depth_duration is None:
        raise ValueError("DEPTH alignment must include offset_seconds and duration_seconds.")
    depth_offset = float(depth_offset)
    depth_duration = float(depth_duration)

    if not reference_file.exists():
        raise FileNotFoundError(f"RGB reference file does not exist: {reference_file}")
    if not ir_file.exists():
        raise FileNotFoundError(f"IR file does not exist: {ir_file}")
    if not event_file.exists():
        raise FileNotFoundError(f"EVENT file does not exist: {event_file}")
    if not depth_file.exists():
        raise FileNotFoundError(f"DEPTH file does not exist: {depth_file}")

    reference_start = max(0.0, -ir_offset, -event_offset, -depth_offset)
    reference_end = min(
        reference_duration,
        ir_duration - ir_offset,
        event_duration - event_offset,
        depth_duration - depth_offset,
    )
    duration = max(0.0, reference_end - reference_start)
    if duration <= 0:
        raise ValueError("No positive RGB/IR/EVENT/DEPTH overlap window is available.")

    rgb_seek = reference_start
    ir_seek = max(0.0, reference_start + ir_offset)
    event_seek = max(0.0, reference_start + event_offset)
    depth_seek = max(0.0, reference_start + depth_offset)
    rgb_label = _ffmpeg_escape_text("RGB")
    ir_label = _ffmpeg_escape_text(f"IR offset {ir_offset:.3f}s")
    event_label = _ffmpeg_escape_text(f"EVENT offset {event_offset:.3f}s")
    depth_label = _ffmpeg_escape_text(f"DEPTH offset {depth_offset:.3f}s")
    filter_complex = (
        f"[0:v]fps={EXPORT_PREVIEW_FPS},"
        f"scale={EXPORT_PANEL_WIDTH}:{EXPORT_PANEL_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={EXPORT_PANEL_WIDTH}:{EXPORT_PANEL_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
        f"drawtext=text='{rgb_label}':x=12:y=12:fontsize={EXPORT_LABEL_FONT_SIZE}:fontcolor=white:"
        f"box=1:boxcolor=black@0.55:boxborderw=8,setsar=1,setpts=PTS-STARTPTS[rgb];"
        f"[1:v]fps={EXPORT_PREVIEW_FPS},"
        f"scale={EXPORT_PANEL_WIDTH}:{EXPORT_PANEL_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={EXPORT_PANEL_WIDTH}:{EXPORT_PANEL_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
        f"drawtext=text='{ir_label}':x=12:y=12:fontsize={EXPORT_LABEL_FONT_SIZE}:fontcolor=white:"
        f"box=1:boxcolor=black@0.55:boxborderw=8,setsar=1,setpts=PTS-STARTPTS[ir];"
        f"[2:v]fps={EXPORT_PREVIEW_FPS},"
        f"scale={EXPORT_PANEL_WIDTH}:{EXPORT_PANEL_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={EXPORT_PANEL_WIDTH}:{EXPORT_PANEL_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
        f"drawtext=text='{event_label}':x=12:y=12:fontsize={EXPORT_LABEL_FONT_SIZE}:fontcolor=white:"
        f"box=1:boxcolor=black@0.55:boxborderw=8,setsar=1,setpts=PTS-STARTPTS[event];"
        f"[3:v]fps={EXPORT_PREVIEW_FPS},"
        f"scale={EXPORT_PANEL_WIDTH}:{EXPORT_PANEL_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={EXPORT_PANEL_WIDTH}:{EXPORT_PANEL_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
        f"drawtext=text='{depth_label}':x=12:y=12:fontsize={EXPORT_LABEL_FONT_SIZE}:fontcolor=white:"
        f"box=1:boxcolor=black@0.55:boxborderw=8,setsar=1,setpts=PTS-STARTPTS[depth];"
        "[rgb][ir]hstack=inputs=2[top];"
        "[event][depth]hstack=inputs=2[bottom];"
        "[top][bottom]vstack=inputs=2,format=yuv420p[outv]"
    )
    command_prefix = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{rgb_seek:.6f}",
        "-t",
        f"{duration:.6f}",
        "-i",
        str(reference_file),
        "-ss",
        f"{ir_seek:.6f}",
        "-t",
        f"{duration:.6f}",
        "-i",
        str(ir_file),
        "-ss",
        f"{event_seek:.6f}",
        "-t",
        f"{duration:.6f}",
        "-i",
        str(event_file),
        "-ss",
        f"{depth_seek:.6f}",
        "-t",
        f"{duration:.6f}",
        "-i",
        str(depth_file),
        "-filter_complex",
        filter_complex,
        "-map",
        "[outv]",
        "-an",
    ]
    encoder_used, error = _run_ffmpeg_with_optional_gpu(command_prefix, output_path, prefer_gpu=prefer_gpu)
    if encoder_used is None:
        raise RuntimeError(f"Failed to export aligned RGB/IR/EVENT/DEPTH video: {error}")

    return {
        "output_file": str(output_path),
        "reference_file": str(reference_file),
        "ir_file": str(ir_file),
        "event_file": str(event_file),
        "depth_file": str(depth_file),
        "ir_offset_seconds": ir_offset,
        "event_offset_seconds": event_offset,
        "depth_offset_seconds": depth_offset,
        "rgb_seek_seconds": round(rgb_seek, 6),
        "ir_seek_seconds": round(ir_seek, 6),
        "event_seek_seconds": round(event_seek, 6),
        "depth_seek_seconds": round(depth_seek, 6),
        "duration_seconds": round(duration, 6),
        "encoder": encoder_used,
        "gpu_fallback_warning": error if encoder_used == "libx264" and error else None,
    }


def export_cut_carrot_day_rgb_ir_event_alignment(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return export_cut_carrot_day_rgb_ir_event_depth_alignment(*args, **kwargs)


def export_cut_carrot_day_rgb_ir_alignment(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return export_cut_carrot_day_rgb_ir_event_depth_alignment(*args, **kwargs)
