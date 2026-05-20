"""Estimate temporal offsets between modalities using RGB as reference."""

from __future__ import annotations

import json
from pathlib import Path
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

MODALITY_COLORS = {
    "rgb": (46, 105, 230),
    "event": (30, 150, 70),
    "ir": (190, 70, 170),
    "depth": (35, 160, 210),
}
AXIS_COLOR = (170, 170, 170)
TEXT_COLOR = (45, 45, 45)
GRID_COLOR = (230, 230, 230)


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


def _estimate_offset(
    reference_trace: np.ndarray,
    target_trace: np.ndarray,
    fps: float,
    max_lag_seconds: float,
) -> dict[str, Any]:
    if _is_flat_trace(reference_trace) or _is_flat_trace(target_trace):
        return {
            "offset_seconds": None,
            "offset_frames": None,
            "peak_correlation": None,
            "confidence_label": "low",
            "warnings": ["Motion energy trace is too short or too flat for reliable alignment."],
        }

    max_lag_frames = min(
        int(round(max_lag_seconds * fps)),
        max(0, reference_trace.size - 3),
        max(0, target_trace.size - 3),
    )

    best_lag: int | None = None
    best_correlation: float | None = None
    for lag in range(-max_lag_frames, max_lag_frames + 1):
        if lag >= 0:
            overlap = min(reference_trace.size, target_trace.size - lag)
            reference_overlap = reference_trace[:overlap]
            target_overlap = target_trace[lag : lag + overlap]
        else:
            overlap = min(reference_trace.size + lag, target_trace.size)
            reference_overlap = reference_trace[-lag : -lag + overlap]
            target_overlap = target_trace[:overlap]

        correlation = _overlap_correlation(reference_overlap, target_overlap)
        if correlation is None:
            continue
        if best_correlation is None or correlation > best_correlation:
            best_correlation = correlation
            best_lag = lag

    if best_lag is None or best_correlation is None:
        return {
            "offset_seconds": None,
            "offset_frames": None,
            "peak_correlation": None,
            "confidence_label": "low",
            "warnings": ["No valid cross-correlation overlap was available."],
        }

    if best_correlation >= 0.55:
        confidence = "high"
    elif best_correlation >= 0.30:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "offset_seconds": float(best_lag / fps),
        "offset_frames": int(best_lag),
        "peak_correlation": float(best_correlation),
        "confidence_label": confidence,
        "warnings": [],
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
    estimate = _estimate_offset(reference_trace, target_trace, lag_fps, max_lag_seconds)
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
