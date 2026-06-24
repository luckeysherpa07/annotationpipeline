"""Estimate temporal offsets between modalities using RGB as reference."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
from typing import Any

import cv2
import numpy as np

from .day_night_pair_alignment import run_wash_cup_day_night_rgb_alignment


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".mpeg", ".mpg"}
REFERENCE_MODALITY = "rgb"
TARGET_MODALITIES = ("event", "ir", "depth")
ALL_MODALITIES = (REFERENCE_MODALITY, *TARGET_MODALITIES)
DEFAULT_ALIGNMENT_JSON_FOLDER = Path("temporal_alignment_json")
DEFAULT_OUTPUT_PATH = DEFAULT_ALIGNMENT_JSON_FOLDER / "temporal_alignment_results.json"
DEFAULT_DAY_OUTPUT_PATH = DEFAULT_ALIGNMENT_JSON_FOLDER / "temporal_alignment_day_results.json"
DEFAULT_NIGHT_OUTPUT_PATH = DEFAULT_ALIGNMENT_JSON_FOLDER / "temporal_alignment_night_results.json"
DEFAULT_OPTICAL_FLOW_CHECK_MAILBOX_OUTPUT_PATH = (
    DEFAULT_ALIGNMENT_JSON_FOLDER / "temporal_alignment_optical_flow_check_mailbox_day_event.json"
)
DEFAULT_DTW_CHECK_MAILBOX_OUTPUT_PATH = DEFAULT_ALIGNMENT_JSON_FOLDER / "temporal_alignment_dtw_check_mailbox_day_event.json"
DEFAULT_FEATURE_CHECK_MAILBOX_OUTPUT_PATH = (
    DEFAULT_ALIGNMENT_JSON_FOLDER / "temporal_alignment_feature_check_mailbox_day_event.json"
)
DEFAULT_RGB_AUDIO_CHECK_MAILBOX_OUTPUT_PATH = (
    DEFAULT_ALIGNMENT_JSON_FOLDER / "temporal_alignment_cross_correlation_check_mailbox_day_audio.json"
)
DEFAULT_PLOT_OUTPUT_FOLDER = Path("temporal_alignment_plots")
DEFAULT_EXPORT_OUTPUT_FOLDER = Path("temporal_alignment_exports")
DEFAULT_ALIGNED_DATASET_FOLDER = Path("aligned_dataset")
EXPORT_PANEL_WIDTH = 320
EXPORT_PANEL_HEIGHT = 180
EXPORT_PREVIEW_FPS = 10
DTW_EXPORT_PREVIEW_FPS = 30
DTW_EXPORT_SMOOTHING_SECONDS = 1.5
AUDIO_ALIGNMENT_SAMPLE_RATE = 16000
AUDIO_ALIGNMENT_MAX_LAG_SECONDS = 30.0
SOURCE_RGB_AUDIO_ALIGNMENT_MAX_LAG_SECONDS = 2.0
FEATURE_ALIGNMENT_SAMPLE_STRIDE_SECONDS = 4.0
FEATURE_ALIGNMENT_OFFSET_STEP_SECONDS = 0.5
FEATURE_ALIGNMENT_MAX_OFFSET_SECONDS = 20.0
FEATURE_ALIGNMENT_MIN_OFFSET_SECONDS = -20.0
FEATURE_ALIGNMENT_RESIZE_WIDTH = 320
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


def _ffprobe_json(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        return {"streams": [], "format": {}, "error": completed.stderr.strip()}
    try:
        data = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"streams": [], "format": {}, "error": "Could not parse ffprobe JSON output."}
    return data if isinstance(data, dict) else {"streams": [], "format": {}}


def _parse_fps(value: str | None) -> float:
    if not value:
        return 0.0
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        try:
            denominator_float = float(denominator)
            return float(numerator) / denominator_float if denominator_float else 0.0
        except ValueError:
            return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _source_video_encoding_info(path: Path) -> dict[str, Any]:
    data = _ffprobe_json(path)
    video_stream = next((item for item in data.get("streams", []) if item.get("codec_type") == "video"), {})
    format_info = data.get("format", {}) if isinstance(data.get("format"), dict) else {}
    bit_rate = video_stream.get("bit_rate") or format_info.get("bit_rate")
    try:
        bit_rate_int = int(float(bit_rate)) if bit_rate is not None else None
    except (TypeError, ValueError):
        bit_rate_int = None
    return {
        "codec_name": video_stream.get("codec_name"),
        "width": int(video_stream.get("width") or 0),
        "height": int(video_stream.get("height") or 0),
        "fps": _parse_fps(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate")),
        "bit_rate": bit_rate_int,
        "duration_seconds": float(format_info.get("duration") or 0.0),
    }


def _source_audio_encoding_info(path: Path) -> dict[str, Any]:
    data = _ffprobe_json(path)
    audio_stream = next((item for item in data.get("streams", []) if item.get("codec_type") == "audio"), {})
    format_info = data.get("format", {}) if isinstance(data.get("format"), dict) else {}
    bit_rate = audio_stream.get("bit_rate") or format_info.get("bit_rate")
    try:
        bit_rate_int = int(float(bit_rate)) if bit_rate is not None else None
    except (TypeError, ValueError):
        bit_rate_int = None
    return {
        "codec_name": audio_stream.get("codec_name"),
        "sample_rate": int(audio_stream.get("sample_rate") or 0),
        "channels": int(audio_stream.get("channels") or 0),
        "bit_rate": bit_rate_int,
        "duration_seconds": float(format_info.get("duration") or 0.0),
    }


def _audio_energy_trace(
    audio_path: Path,
    sample_rate: int = AUDIO_ALIGNMENT_SAMPLE_RATE,
    fps: float = DTW_EXPORT_PREVIEW_FPS,
) -> tuple[np.ndarray, float]:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
    if fps <= 0:
        raise ValueError("Audio activity trace requires a positive target FPS.")

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        str(int(sample_rate)),
        "-f",
        "f32le",
        "pipe:1",
    ]
    completed = subprocess.run(command, check=False, capture_output=True)
    if completed.returncode != 0:
        error = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Failed to decode audio activity trace: {error}")

    samples = np.frombuffer(completed.stdout, dtype=np.float32).astype(np.float32, copy=True)
    if samples.size < 3:
        raise ValueError(f"Audio file is too short or empty: {audio_path}")
    samples = np.nan_to_num(samples, copy=False)
    duration_seconds = float(samples.size / max(1, int(sample_rate)))

    hop_size = max(1, int(round(sample_rate / fps)))
    frame_count = int(np.ceil(samples.size / hop_size))
    padded_size = frame_count * hop_size
    if padded_size > samples.size:
        samples = np.pad(samples, (0, padded_size - samples.size), mode="constant")
    windows = samples.reshape(frame_count, hop_size)
    rms = np.sqrt(np.mean(windows * windows, axis=1)).astype(np.float32)
    return rms, duration_seconds


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


def _resize_gray_frame(frame: np.ndarray, resize_width: int) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if resize_width > 0 and gray.shape[1] > resize_width:
        scale = resize_width / float(gray.shape[1])
        resize_height = max(1, int(round(gray.shape[0] * scale)))
        gray = cv2.resize(gray, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
    return gray


def _optical_flow_magnitude_trace(video_path: Path, resize_width: int = 160) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    previous_gray: np.ndarray | None = None
    magnitudes: list[float] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = _resize_gray_frame(frame, resize_width).astype(np.uint8)
        if previous_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                previous_gray,
                gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            magnitudes.append(float(np.mean(magnitude)))
        previous_gray = gray

    cap.release()
    return np.asarray(magnitudes, dtype=np.float32)


def _dynamic_time_warping_path(
    reference_trace: np.ndarray,
    target_trace: np.ndarray,
    reference_fps: float,
    target_fps: float,
    window_seconds: float,
) -> tuple[list[tuple[int, int]], float]:
    if reference_trace.size < 3 or target_trace.size < 3:
        raise ValueError("DTW requires at least three samples in both traces.")
    if reference_fps <= 0 or target_fps <= 0:
        raise ValueError("DTW requires positive FPS values for both traces.")

    reference = _prepare_alignment_trace(reference_trace, reference_fps)
    target = _prepare_alignment_trace(target_trace, target_fps)
    ref_count = int(reference.size)
    target_count = int(target.size)
    window_frames = max(1, int(round(window_seconds * max(reference_fps, target_fps))))

    costs = np.full((ref_count + 1, target_count + 1), np.inf, dtype=np.float32)
    backtrack = np.zeros((ref_count, target_count), dtype=np.uint8)
    costs[0, 0] = 0.0

    for ref_index in range(ref_count):
        expected_target = int(round(ref_index * (target_count - 1) / max(1, ref_count - 1)))
        target_start = max(0, expected_target - window_frames)
        target_end = min(target_count, expected_target + window_frames + 1)
        for target_index in range(target_start, target_end):
            previous_costs = (
                costs[ref_index, target_index],
                costs[ref_index, target_index + 1],
                costs[ref_index + 1, target_index],
            )
            direction = int(np.argmin(previous_costs))
            best_previous = previous_costs[direction]
            if not np.isfinite(best_previous):
                continue
            distance = float(reference[ref_index] - target[target_index])
            costs[ref_index + 1, target_index + 1] = best_previous + distance * distance
            backtrack[ref_index, target_index] = direction

    if not np.isfinite(costs[ref_count, target_count]):
        raise ValueError(
            f"No DTW path found. Increase window_seconds above {window_seconds:.3f}s for these durations."
        )

    path: list[tuple[int, int]] = []
    ref_index = ref_count - 1
    target_index = target_count - 1
    while ref_index >= 0 and target_index >= 0:
        path.append((ref_index, target_index))
        direction = int(backtrack[ref_index, target_index])
        if direction == 0:
            ref_index -= 1
            target_index -= 1
        elif direction == 1:
            ref_index -= 1
        else:
            target_index -= 1
    path.reverse()
    normalized_cost = float(costs[ref_count, target_count] / max(1, len(path)))
    return path, normalized_cost


def _offset_curve_from_dtw_path(
    path: list[tuple[int, int]],
    reference_fps: float,
    target_fps: float,
) -> list[dict[str, float]]:
    by_reference: dict[int, list[int]] = {}
    for reference_index, target_index in path:
        by_reference.setdefault(reference_index, []).append(target_index)

    offset_curve: list[dict[str, float]] = []
    for reference_index in sorted(by_reference):
        target_index = float(np.median(np.asarray(by_reference[reference_index], dtype=np.float32)))
        reference_time = float(reference_index / reference_fps)
        target_time = float(target_index / target_fps)
        offset_curve.append(
            {
                "reference_time_seconds": round(reference_time, 6),
                "event_time_seconds": round(target_time, 6),
                "offset_seconds": round(target_time - reference_time, 6),
            }
        )
    return offset_curve


def _sample_offset_curve(
    offset_curve: list[dict[str, float]],
    max_points: int = 900,
) -> list[dict[str, float]]:
    if len(offset_curve) <= max_points:
        return offset_curve
    indices = np.linspace(0, len(offset_curve) - 1, max_points).round().astype(np.int32)
    return [offset_curve[int(index)] for index in np.unique(indices)]


def _event_time_from_offset_curve(offset_curve: list[dict[str, float]], reference_time: float) -> float:
    if not offset_curve:
        raise ValueError("DTW offset curve is empty.")
    reference_times = np.asarray([item["reference_time_seconds"] for item in offset_curve], dtype=np.float32)
    event_times = np.asarray([item["event_time_seconds"] for item in offset_curve], dtype=np.float32)
    return float(np.interp(reference_time, reference_times, event_times))


def _smooth_dtw_offset_curve_for_export(
    offset_curve: list[dict[str, float]],
    smoothing_seconds: float = DTW_EXPORT_SMOOTHING_SECONDS,
) -> list[dict[str, float]]:
    if len(offset_curve) < 3:
        return offset_curve

    reference_times = np.asarray([item["reference_time_seconds"] for item in offset_curve], dtype=np.float32)
    offsets = np.asarray([item["offset_seconds"] for item in offset_curve], dtype=np.float32)
    median_step = float(np.median(np.diff(reference_times))) if reference_times.size > 1 else 0.0
    if median_step <= 0:
        return offset_curve

    window = max(3, int(round(smoothing_seconds / median_step)))
    if window % 2 == 0:
        window += 1
    window = min(window, offsets.size if offsets.size % 2 == 1 else offsets.size - 1)
    if window < 3:
        smoothed_offsets = offsets
    else:
        pad = window // 2
        padded = np.pad(offsets, (pad, pad), mode="edge")
        kernel = np.ones(window, dtype=np.float32) / float(window)
        smoothed_offsets = np.convolve(padded, kernel, mode="valid").astype(np.float32)

    event_times = reference_times + smoothed_offsets
    event_times = np.maximum.accumulate(event_times)
    smoothed_offsets = event_times - reference_times
    return [
        {
            "reference_time_seconds": round(float(reference_time), 6),
            "event_time_seconds": round(float(event_time), 6),
            "offset_seconds": round(float(offset), 6),
        }
        for reference_time, event_time, offset in zip(reference_times, event_times, smoothed_offsets)
    ]


def _preprocess_feature_frame(frame: np.ndarray, modality: str, resize_width: int = FEATURE_ALIGNMENT_RESIZE_WIDTH) -> np.ndarray:
    gray = _resize_gray_frame(frame, resize_width)
    if modality == "event":
        low = float(np.percentile(gray, 2))
        high = float(np.percentile(gray, 98))
        if high > low:
            gray = np.clip((gray.astype(np.float32) - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
        else:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 35, 110)
        return cv2.dilate(edges, np.ones((2, 2), dtype=np.uint8), iterations=1)

    gray = cv2.equalizeHist(gray.astype(np.uint8))
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.Canny(gray, 45, 135)


def _read_video_frame_by_index(video_path: Path, frame_index: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _read_video_frame_by_time(video_path: Path, time_seconds: float, fps: float) -> np.ndarray | None:
    return _read_video_frame_by_index(video_path, int(round(max(0.0, time_seconds) * fps)))


def _detect_orb_features(
    frame: np.ndarray | None,
    modality: str,
    orb: cv2.ORB,
    resize_width: int = FEATURE_ALIGNMENT_RESIZE_WIDTH,
) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    if frame is None:
        return [], None
    prepared = _preprocess_feature_frame(frame, modality=modality, resize_width=resize_width)
    keypoints, descriptors = orb.detectAndCompute(prepared, None)
    return list(keypoints or []), descriptors


def _score_feature_match(
    reference_features: tuple[list[cv2.KeyPoint], np.ndarray | None],
    event_features: tuple[list[cv2.KeyPoint], np.ndarray | None],
    matcher: cv2.BFMatcher,
) -> dict[str, Any]:
    reference_keypoints, reference_descriptors = reference_features
    event_keypoints, event_descriptors = event_features
    if reference_descriptors is None or event_descriptors is None:
        return {"score": 0.0, "match_count": 0, "inlier_count": 0, "mean_distance": None}
    if len(reference_descriptors) < 2 or len(event_descriptors) < 2:
        return {"score": 0.0, "match_count": 0, "inlier_count": 0, "mean_distance": None}

    raw_matches = matcher.knnMatch(reference_descriptors, event_descriptors, k=2)
    good_matches = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < 0.78 * second.distance:
            good_matches.append(first)

    match_count = len(good_matches)
    if match_count == 0:
        return {"score": 0.0, "match_count": 0, "inlier_count": 0, "mean_distance": None}

    mean_distance = float(np.mean([match.distance for match in good_matches]))
    inlier_count = 0
    if match_count >= 8:
        reference_points = np.float32([reference_keypoints[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
        event_points = np.float32([event_keypoints[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(reference_points, event_points, cv2.RANSAC, 5.0)
        if mask is not None:
            inlier_count = int(mask.ravel().sum())

    score = float(inlier_count * 2.0 + match_count * 0.35 - mean_distance * 0.015)
    return {
        "score": max(0.0, score),
        "match_count": int(match_count),
        "inlier_count": int(inlier_count),
        "mean_distance": round(mean_distance, 6),
    }


def _confidence_from_feature_window(match_count: int, inlier_count: int, score: float) -> str:
    if inlier_count >= 12 and score >= 20:
        return "high"
    if inlier_count >= 6 or match_count >= 18 or score >= 8:
        return "medium"
    return "low"


def _load_dtw_offset_curve(path: Path = DEFAULT_DTW_CHECK_MAILBOX_OUTPUT_PATH) -> list[dict[str, float]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        alignment = data.get("alignment", {}) if isinstance(data, dict) else {}
        curve = alignment.get("smoothed_offset_curve") or alignment.get("offset_curve") or []
        return [item for item in curve if isinstance(item, dict)]
    except Exception:
        return []


def _interpolated_offset_curve_from_windows(
    windows: list[dict[str, Any]],
    reference_duration: float,
    target_duration: float,
    fps: float,
) -> list[dict[str, float]]:
    anchors = [
        window
        for window in windows
        if isinstance(window.get("selected_offset_seconds"), (int, float)) and window.get("confidence_label") != "low"
    ]
    if len(anchors) < 2:
        anchors = [window for window in windows if isinstance(window.get("selected_offset_seconds"), (int, float))]
    if not anchors:
        return []

    anchor_times = np.asarray([float(window["reference_time_seconds"]) for window in anchors], dtype=np.float32)
    anchor_offsets = np.asarray([float(window["selected_offset_seconds"]) for window in anchors], dtype=np.float32)
    order = np.argsort(anchor_times)
    anchor_times = anchor_times[order]
    anchor_offsets = anchor_offsets[order]
    times = np.arange(0.0, max(0.0, reference_duration), 1.0 / max(fps, 1e-6), dtype=np.float32)
    offsets = np.interp(times, anchor_times, anchor_offsets).astype(np.float32)

    curve = [
        {
            "reference_time_seconds": round(float(reference_time), 6),
            "event_time_seconds": round(float(np.clip(reference_time + offset, 0.0, target_duration)), 6),
            "offset_seconds": round(float(np.clip(reference_time + offset, 0.0, target_duration) - reference_time), 6),
        }
        for reference_time, offset in zip(times, offsets)
    ]
    return _smooth_dtw_offset_curve_for_export(curve, smoothing_seconds=DTW_EXPORT_SMOOTHING_SECONDS)


def _draw_label(
    frame: np.ndarray,
    lines: list[str],
    x: int = 12,
    y: int = 12,
) -> None:
    if not lines:
        return
    line_height = 22
    width = max(1, max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0][0] for line in lines))
    height = line_height * len(lines) + 12
    cv2.rectangle(frame, (x - 4, y - 4), (x + width + 12, y + height), (0, 0, 0), -1)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 4, y - 4), (x + width + 12, y + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + 18 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _read_video_frame_at(cap: cv2.VideoCapture, time_seconds: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_seconds) * 1000.0)
    ok, frame = cap.read()
    return frame if ok else None


def _read_video_frame_forward(
    cap: cv2.VideoCapture,
    frame_index: int,
    state: dict[str, Any],
) -> np.ndarray | None:
    frame_index = max(0, int(frame_index))
    current_index = int(state.get("next_index", 0))
    frame_cache = state.setdefault("frame_cache", {})
    if frame_index in frame_cache:
        return frame_cache[frame_index].copy()

    if frame_index < current_index:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        current_index = frame_index

    frame: np.ndarray | None = None
    while current_index <= frame_index:
        ok, candidate = cap.read()
        if not ok:
            break
        frame = candidate
        current_index += 1

    state["next_index"] = current_index
    if frame is not None:
        frame_cache[frame_index] = frame.copy()
        for cached_index in sorted(frame_cache)[:-8]:
            frame_cache.pop(cached_index, None)
    return frame


def _read_video_frame_pair_forward(
    cap: cv2.VideoCapture,
    first_frame_index: int,
    state: dict[str, Any],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    first_frame = _read_video_frame_forward(cap, first_frame_index, state)
    second_frame = _read_video_frame_forward(cap, first_frame_index + 1, state)
    return first_frame, second_frame


def _blend_frames(first_frame: np.ndarray | None, second_frame: np.ndarray | None, weight: float) -> np.ndarray | None:
    if first_frame is None:
        return second_frame
    if second_frame is None:
        return first_frame
    weight = float(np.clip(weight, 0.0, 1.0))
    if weight <= 1e-6:
        return first_frame
    if weight >= 1.0 - 1e-6:
        return second_frame
    return cv2.addWeighted(first_frame, 1.0 - weight, second_frame, weight, 0.0)


def _prepare_preview_panel(frame: np.ndarray | None) -> np.ndarray:
    panel = np.zeros((EXPORT_PANEL_HEIGHT, EXPORT_PANEL_WIDTH, 3), dtype=np.uint8)
    if frame is None:
        cv2.putText(panel, "missing frame", (76, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        return panel

    height, width = frame.shape[:2]
    scale = min(EXPORT_PANEL_WIDTH / max(1, width), EXPORT_PANEL_HEIGHT / max(1, height))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    x = (EXPORT_PANEL_WIDTH - resized_width) // 2
    y = (EXPORT_PANEL_HEIGHT - resized_height) // 2
    panel[y : y + resized_height, x : x + resized_width] = resized
    return panel


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


def _write_rgb_audio_activity_signal_plot(
    output_path: Path,
    title: str,
    reference_trace: np.ndarray,
    reference_fps: float,
    audio_trace: np.ndarray,
    audio_fps: float,
    alignment: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 1500, 780
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    left, right = 90, width - 50
    raw_bounds = (left, 145, right, 335)
    aligned_bounds = (left, 485, right, 675)

    rgb_color = MODALITY_COLORS["rgb"]
    audio_color = (210, 115, 35)
    offset_seconds = alignment.get("offset_seconds")
    offset_seconds = float(offset_seconds) if isinstance(offset_seconds, (int, float)) else None
    correlation = alignment.get("peak_correlation")
    overlap_ratio = alignment.get("overlap_ratio")
    confidence = str(alignment.get("confidence_label") or "unknown")

    reference_times = _times_for_trace(reference_trace, reference_fps)
    audio_times = _times_for_trace(audio_trace, audio_fps)
    reference_values = _normalize_for_plot(reference_trace)
    audio_values = _normalize_for_plot(audio_trace)
    aligned_audio_times = audio_times - offset_seconds if offset_seconds is not None else audio_times

    raw_items = [
        (reference_times, reference_values, rgb_color, "rgb optical flow"),
        (audio_times, audio_values, audio_color, "audio RMS"),
    ]
    aligned_items = [
        (reference_times, reference_values, rgb_color, "rgb optical flow"),
        (aligned_audio_times, audio_values, audio_color, "audio RMS shifted"),
    ]
    raw_min_time, raw_max_time = _time_range(raw_items)
    aligned_min_time, aligned_max_time = _time_range(aligned_items)

    summary = (
        f"offset {offset_seconds:+.3f}s  " if offset_seconds is not None else "offset unknown  "
    )
    if isinstance(correlation, (int, float)):
        summary += f"correlation {float(correlation):.3f}  "
    if isinstance(overlap_ratio, (int, float)):
        summary += f"overlap {float(overlap_ratio):.3f}  "
    summary += f"confidence {confidence}"

    cv2.putText(canvas, title, (left, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.86, TEXT_COLOR, 2, cv2.LINE_AA)
    cv2.putText(canvas, summary, (left, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.52, TEXT_COLOR, 1, cv2.LINE_AA)
    _draw_panel(canvas, "Raw RGB/audio activity traces", raw_bounds, raw_min_time, raw_max_time, raw_items)
    _draw_panel(
        canvas,
        "Fixed-offset audio trace on RGB timeline",
        aligned_bounds,
        aligned_min_time,
        aligned_max_time,
        aligned_items,
    )

    cv2.imwrite(str(output_path), canvas)


def _write_dtw_activity_signal_plot(
    output_path: Path,
    title: str,
    reference_trace: np.ndarray,
    reference_fps: float,
    event_trace: np.ndarray,
    event_fps: float,
    offset_curve: list[dict[str, float]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 1500, 820
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    left, right = 90, width - 50
    raw_bounds = (left, 145, right, 335)
    aligned_bounds = (left, 505, right, 695)

    reference_times = _times_for_trace(reference_trace, reference_fps)
    event_times = _times_for_trace(event_trace, event_fps)
    reference_values = _normalize_for_plot(reference_trace)
    event_values = _normalize_for_plot(event_trace)

    curve_reference_times = np.asarray([item["reference_time_seconds"] for item in offset_curve], dtype=np.float32)
    curve_event_times = np.asarray([item["event_time_seconds"] for item in offset_curve], dtype=np.float32)
    event_by_reference = np.interp(curve_event_times, event_times, event_values).astype(np.float32)
    reference_by_curve = np.interp(curve_reference_times, reference_times, reference_values).astype(np.float32)

    offsets = np.asarray([item["offset_seconds"] for item in offset_curve], dtype=np.float32)
    if offsets.size:
        summary = (
            f"DTW median offset {float(np.median(offsets)):.3f}s  "
            f"start {float(offsets[0]):.3f}s  end {float(offsets[-1]):.3f}s  "
            f"drift {float(offsets[-1] - offsets[0]):.3f}s"
        )
    else:
        summary = "DTW offset curve unavailable"

    cv2.putText(canvas, title, (left, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.86, TEXT_COLOR, 2, cv2.LINE_AA)
    cv2.putText(canvas, summary, (left, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.52, TEXT_COLOR, 1, cv2.LINE_AA)

    raw_items = [
        (reference_times, reference_values, MODALITY_COLORS["rgb"], "rgb"),
        (event_times, event_values, MODALITY_COLORS["event"], "event"),
    ]
    aligned_items = [
        (curve_reference_times, reference_by_curve, MODALITY_COLORS["rgb"], "rgb"),
        (curve_reference_times, event_by_reference, MODALITY_COLORS["event"], "event warped by DTW"),
    ]
    raw_min_time, raw_max_time = _time_range(raw_items)
    aligned_min_time, aligned_max_time = _time_range(aligned_items)
    _draw_panel(canvas, "Raw optical-flow traces", raw_bounds, raw_min_time, raw_max_time, raw_items)
    _draw_panel(canvas, "DTW-warped EVENT trace on RGB timeline", aligned_bounds, aligned_min_time, aligned_max_time, aligned_items)

    cv2.imwrite(str(output_path), canvas)


def _write_feature_offset_plot(
    output_path: Path,
    title: str,
    windows: list[dict[str, Any]],
    offset_curve: list[dict[str, float]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1500, 640
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    left, right = 90, width - 50
    top, bottom = 110, height - 90

    all_times = [float(item["reference_time_seconds"]) for item in offset_curve]
    all_offsets = [float(item["offset_seconds"]) for item in offset_curve]
    for window in windows:
        if isinstance(window.get("selected_offset_seconds"), (int, float)):
            all_times.append(float(window["reference_time_seconds"]))
            all_offsets.append(float(window["selected_offset_seconds"]))
    if not all_times or not all_offsets:
        cv2.putText(canvas, "No feature offsets available", (left, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2)
        cv2.imwrite(str(output_path), canvas)
        return

    min_time = min(all_times)
    max_time = max(all_times)
    min_offset = min(all_offsets)
    max_offset = max(all_offsets)
    if max_time <= min_time:
        max_time = min_time + 1.0
    if max_offset <= min_offset:
        max_offset = min_offset + 1.0
    padding = max(0.5, (max_offset - min_offset) * 0.08)
    min_offset -= padding
    max_offset += padding

    cv2.putText(canvas, title, (left, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.86, TEXT_COLOR, 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (left, top), (right, bottom), AXIS_COLOR, 1)
    for tick in np.linspace(min_time, max_time, 6):
        x = int(round(left + (tick - min_time) / (max_time - min_time) * (right - left)))
        cv2.line(canvas, (x, top), (x, bottom), GRID_COLOR, 1)
        cv2.putText(canvas, f"{tick:.0f}s", (x - 18, bottom + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1)
    for tick in np.linspace(min_offset, max_offset, 5):
        y = int(round(bottom - (tick - min_offset) / (max_offset - min_offset) * (bottom - top)))
        cv2.line(canvas, (left, y), (right, y), GRID_COLOR, 1)
        cv2.putText(canvas, f"{tick:.1f}s", (20, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1)

    if offset_curve:
        curve_points = []
        for item in offset_curve:
            x = int(round(left + (float(item["reference_time_seconds"]) - min_time) / (max_time - min_time) * (right - left)))
            y = int(round(bottom - (float(item["offset_seconds"]) - min_offset) / (max_offset - min_offset) * (bottom - top)))
            curve_points.append((x, y))
        if len(curve_points) > 1:
            cv2.polylines(canvas, [np.asarray(curve_points, dtype=np.int32).reshape((-1, 1, 2))], False, MODALITY_COLORS["event"], 2, cv2.LINE_AA)

    confidence_colors = {"high": (30, 130, 30), "medium": (30, 140, 210), "low": (150, 150, 150)}
    for window in windows:
        if not isinstance(window.get("selected_offset_seconds"), (int, float)):
            continue
        x = int(round(left + (float(window["reference_time_seconds"]) - min_time) / (max_time - min_time) * (right - left)))
        y = int(round(bottom - (float(window["selected_offset_seconds"]) - min_offset) / (max_offset - min_offset) * (bottom - top)))
        color = confidence_colors.get(str(window.get("confidence_label")), (120, 120, 120))
        cv2.circle(canvas, (x, y), 4, color, -1, cv2.LINE_AA)

    cv2.putText(canvas, "line=smoothed offset curve, dots=local feature estimates", (left, height - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
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


def _estimate_raw_cross_correlation_offset(
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
            "candidate_offsets": [],
            "selected_by": "unavailable",
            "warnings": ["Activity trace is too short or too flat for reliable alignment."],
        }

    prepared_reference = _prepare_alignment_trace(reference_trace, fps=fps)
    prepared_target = _prepare_alignment_trace(target_trace, fps=fps)
    max_lag_frames = min(
        int(round(max_lag_seconds * fps)),
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

        candidates.append(
            {
                "offset_seconds": float(lag / fps),
                "offset_frames": int(lag),
                "correlation": float(correlation),
                "overlap_ratio": float(reference_overlap.size / max_overlap),
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
    selected = candidates_by_correlation[0]
    second_correlation = candidates_by_correlation[1]["correlation"] if len(candidates_by_correlation) > 1 else None
    peak_margin = float(selected["correlation"] - second_correlation) if second_correlation is not None else 0.0

    return {
        "offset_seconds": float(selected["offset_seconds"]),
        "offset_frames": int(selected["offset_frames"]),
        "peak_correlation": float(selected["correlation"]),
        "confidence_label": _confidence_from_correlation(float(selected["correlation"]), peak_margin=peak_margin),
        "candidate_offsets": [
            {
                "offset_seconds": round(float(item["offset_seconds"]), 6),
                "offset_frames": int(item["offset_frames"]),
                "correlation": round(float(item["correlation"]), 6),
                "overlap_ratio": round(float(item["overlap_ratio"]), 6),
            }
            for item in candidates_by_correlation[:TOP_LAG_CANDIDATE_COUNT]
        ],
        "selected_by": "best_raw_correlation",
        "peak_margin": float(peak_margin),
        "lag_fps": float(fps),
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


def run_check_mailbox_day_rgb_event_optical_flow_alignment(
    dataset_folder: Path | str = "dataset",
    output_path: Path | str = DEFAULT_OPTICAL_FLOW_CHECK_MAILBOX_OUTPUT_PATH,
    plot_output_folder: Path | str | None = DEFAULT_PLOT_OUTPUT_FOLDER,
    max_lag_seconds: float = 30.0,
    resize_width: int = 160,
) -> dict[str, Any]:
    dataset_folder = Path(dataset_folder)
    output_path = Path(output_path)
    plot_output_folder = Path(plot_output_folder) if plot_output_folder is not None else None
    reference_file = dataset_folder / "check_mailbox_split" / "check_mailbox_day_rgb.mp4"
    event_file = dataset_folder / "check_mailbox_split" / "check_mailbox_day_event.mp4"

    reference_meta = _video_metadata(reference_file)
    event_meta = _video_metadata(event_file)
    warnings: list[str] = []
    result: dict[str, Any] = {
        "pair_key": "dataset/check_mailbox_split/check_mailbox",
        "side": "day",
        "method": "optical_flow",
        "reference_modality": REFERENCE_MODALITY,
        "reference_file": str(reference_file),
        "reference_duration_seconds": float(reference_meta["duration_seconds"]),
        "target_modality": "event",
        "target_file": str(event_file),
        "target_duration_seconds": float(event_meta["duration_seconds"]),
        "plot_file": None,
        "alignment": None,
        "comparison": {},
        "warnings": warnings,
    }

    if not reference_meta["opened"]:
        warnings.append(f"Could not open RGB reference video: {reference_file}")
    if not event_meta["opened"]:
        warnings.append(f"Could not open EVENT video: {event_file}")
    reference_fps = float(reference_meta["fps"])
    event_fps = float(event_meta["fps"])
    if reference_fps <= 0 or event_fps <= 0:
        warnings.append("Could not determine FPS for one or both videos.")
    if warnings:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        return result

    reference_trace = _optical_flow_magnitude_trace(reference_file, resize_width=resize_width)
    event_trace = _optical_flow_magnitude_trace(event_file, resize_width=resize_width)
    lag_fps = min(reference_fps, event_fps)
    estimate = _estimate_offset(
        reference_trace,
        event_trace,
        lag_fps,
        max_lag_seconds=max_lag_seconds,
        target_modality="event",
    )
    warnings.extend(estimate.pop("warnings", []))
    reference_overlap, target_overlap = _overlap_windows(
        float(reference_meta["duration_seconds"]),
        float(event_meta["duration_seconds"]),
        estimate["offset_seconds"],
    )
    alignment = {
        "modality": "event",
        "file": str(event_file),
        "duration_seconds": float(event_meta["duration_seconds"]),
        **estimate,
        "overlap_reference_seconds": reference_overlap,
        "overlap_target_seconds": target_overlap,
        "activity_plot_file": None,
        "warnings": warnings,
    }

    if plot_output_folder is not None:
        plot_path = plot_output_folder / "check_mailbox_day_rgb_event_optical_flow_activity_signal.png"
        _write_activity_signal_plot(
            plot_path,
            "Optical-flow activity signal: check_mailbox day RGB vs EVENT",
            {
                REFERENCE_MODALITY: (reference_trace, reference_fps),
                "event": (event_trace, event_fps),
            },
            {REFERENCE_MODALITY: 0.0, "event": alignment["offset_seconds"]},
            {"event": alignment},
            (REFERENCE_MODALITY, "event"),
        )
        alignment["activity_plot_file"] = str(plot_path)
        result["plot_file"] = str(plot_path)

    result["alignment"] = alignment
    try:
        day_results = _load_alignment_results(DEFAULT_DAY_OUTPUT_PATH)
        check_mailbox = next(
            item for item in day_results if item.get("pair_key") == "dataset/check_mailbox_split/check_mailbox"
        )
        current_event = check_mailbox.get("alignments", {}).get("event", {})
        result["comparison"] = {
            "motion_energy_selected_offset_seconds": current_event.get("offset_seconds"),
            "motion_energy_raw_best_offset_seconds": current_event.get("raw_best_offset_seconds"),
            "motion_energy_peak_correlation": current_event.get("peak_correlation"),
            "optical_flow_selected_offset_seconds": alignment.get("offset_seconds"),
            "optical_flow_peak_correlation": alignment.get("peak_correlation"),
        }
    except Exception as exc:
        result["comparison"] = {"warning": f"Could not load current motion-energy comparison: {exc}"}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return result


def run_check_mailbox_day_rgb_event_dtw_alignment(
    dataset_folder: Path | str = "dataset",
    output_path: Path | str = DEFAULT_DTW_CHECK_MAILBOX_OUTPUT_PATH,
    plot_output_folder: Path | str | None = DEFAULT_PLOT_OUTPUT_FOLDER,
    window_seconds: float = 10.0,
    resize_width: int = 160,
) -> dict[str, Any]:
    return _run_rgb_event_dtw_alignment(
        sample_name="check_mailbox",
        split_folder_name="check_mailbox_split",
        side="day",
        dataset_folder=dataset_folder,
        output_path=output_path,
        plot_output_folder=plot_output_folder,
        window_seconds=window_seconds,
        resize_width=resize_width,
    )


def _dtw_alignment_output_path(output_folder: Path, sample_name: str, side: str) -> Path:
    safe_sample_name = sample_name.replace(" ", "_")
    return output_folder / f"temporal_alignment_dtw_{safe_sample_name}_{side}_event.json"


def _audio_alignment_output_path(output_folder: Path, sample_name: str, side: str) -> Path:
    safe_sample_name = sample_name.replace(" ", "_")
    return output_folder / f"temporal_alignment_cross_correlation_{safe_sample_name}_{side}_audio.json"


def _discover_rgb_event_dtw_pairs(dataset_folder: Path) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for side in ("day", "night"):
        for reference_file in sorted(dataset_folder.rglob(f"*_{side}_rgb.mp4")):
            sample_name = reference_file.name[: -len(f"_{side}_rgb.mp4")]
            event_file = reference_file.with_name(f"{sample_name}_{side}_event.mp4")
            pairs.append(
                {
                    "sample": sample_name,
                    "side": side,
                    "split_folder_name": reference_file.parent.name,
                    "pair_key": str(reference_file.parent / sample_name),
                    "reference_file": reference_file,
                    "event_file": event_file,
                    "complete": event_file.exists(),
                }
            )
    return pairs


def _discover_rgb_event_audio_triplets(dataset_folder: Path) -> list[dict[str, Any]]:
    triplets: list[dict[str, Any]] = []
    for side in ("day", "night"):
        for reference_file in sorted(dataset_folder.rglob(f"*_{side}_rgb.mp4")):
            if reference_file.name.endswith(f"_{side}_rgb_with_audio.mp4"):
                continue
            sample_name = reference_file.name[: -len(f"_{side}_rgb.mp4")]
            event_file = reference_file.with_name(f"{sample_name}_{side}_event.mp4")
            ir_file = reference_file.with_name(f"{sample_name}_{side}_ir.mp4")
            depth_file = reference_file.with_name(f"{sample_name}_{side}_depth.mp4")
            audio_file = reference_file.with_name(f"{sample_name}_{side}.m4a")
            missing = []
            if not event_file.exists():
                missing.append(str(event_file))
            if not ir_file.exists():
                missing.append(str(ir_file))
            if not depth_file.exists():
                missing.append(str(depth_file))
            if not audio_file.exists():
                missing.append(str(audio_file))
            triplets.append(
                {
                    "sample": sample_name,
                    "side": side,
                    "split_folder_name": reference_file.parent.name,
                    "pair_key": str(reference_file.parent / sample_name),
                    "reference_file": reference_file,
                    "event_file": event_file,
                    "ir_file": ir_file,
                    "depth_file": depth_file,
                    "audio_file": audio_file,
                    "complete": not missing,
                    "missing": missing,
                }
            )
    return triplets


def _discover_aligned_dataset_sets(dataset_folder: Path) -> list[dict[str, Any]]:
    sets: list[dict[str, Any]] = []
    for reference_file in sorted(dataset_folder.rglob("*_rgb.mp4")):
        if reference_file.name.endswith("_rgb_with_audio.mp4"):
            continue
        base_stem = reference_file.name[: -len("_rgb.mp4")]
        if "_" in base_stem:
            sample_name, side = base_stem.rsplit("_", 1)
        else:
            sample_name = base_stem
            side = "default"
        event_file = reference_file.with_name(f"{base_stem}_event.mp4")
        ir_file = reference_file.with_name(f"{base_stem}_ir.mp4")
        depth_file = reference_file.with_name(f"{base_stem}_depth.mp4")
        audio_file = reference_file.with_name(f"{base_stem}.m4a")
        missing = []
        for path in (event_file, ir_file, depth_file, audio_file):
            if not path.exists():
                missing.append(str(path))
        sets.append(
            {
                "base_stem": base_stem,
                "sample": sample_name,
                "side": side,
                "split_folder_name": reference_file.parent.name,
                "reference_file": reference_file,
                "event_file": event_file,
                "ir_file": ir_file,
                "depth_file": depth_file,
                "audio_file": audio_file,
                "complete": not missing,
                "missing": missing,
            }
        )
    return sets


def _parse_aligned_rgb_audio_segment_file(reference_file: Path) -> dict[str, Any] | None:
    match = re.match(r"^(?P<sample>.+)_(?P<side>day|night\d*)_rgb$", reference_file.stem.lower())
    if not match:
        return None
    return {
        "sample_name": match.group("sample"),
        "side": match.group("side"),
        "split_folder_name": reference_file.parent.parent.name,
        "segment_name": reference_file.parent.name,
    }


def discover_aligned_rgb_audio_segment_pairs(
    dataset_folder: Path | str = "aligned_dataset",
) -> list[dict[str, Any]]:
    """Discover aligned RGB/.m4a segment pairs that can be muxed into with-audio videos."""
    dataset_folder = Path(dataset_folder)
    discovered: list[dict[str, Any]] = []

    for reference_file in sorted(dataset_folder.rglob("*_rgb.mp4")):
        if not reference_file.is_file() or reference_file.stem.lower().endswith("_rgb_with_audio"):
            continue

        parsed = _parse_aligned_rgb_audio_segment_file(reference_file)
        if not parsed:
            continue

        audio_file = reference_file.with_name(f"{parsed['sample_name']}_{parsed['side']}.m4a")
        output_file = reference_file.with_name(f"{parsed['sample_name']}_{parsed['side']}_rgb_with_audio.mp4")
        if not audio_file.exists():
            continue

        discovered.append(
            {
                **parsed,
                "reference_file": str(reference_file),
                "audio_file": str(audio_file),
                "output_file": str(output_file),
            }
        )

    return discovered


def _parse_source_rgb_audio_file(reference_file: Path) -> dict[str, Any] | None:
    match = re.match(r"^(?P<sample>.+)_(?P<side>day|night\d*)_rgb$", reference_file.stem.lower())
    if not match:
        return None
    return {
        "sample_name": match.group("sample"),
        "side": match.group("side"),
        "split_folder_name": reference_file.parent.name,
    }


def discover_source_rgb_audio_pairs(
    dataset_folder: Path | str = "dataset",
) -> list[dict[str, Any]]:
    """Discover source RGB/.m4a pairs that can be segmented into with-audio videos."""
    dataset_folder = Path(dataset_folder)
    discovered: list[dict[str, Any]] = []

    for reference_file in sorted(dataset_folder.glob("*_split/*_rgb.mp4")):
        if not reference_file.is_file() or reference_file.stem.lower().endswith("_rgb_with_audio"):
            continue

        parsed = _parse_source_rgb_audio_file(reference_file)
        if not parsed:
            continue

        audio_file = reference_file.with_name(f"{parsed['sample_name']}_{parsed['side']}.m4a")
        if not audio_file.exists():
            continue

        discovered.append(
            {
                **parsed,
                "reference_file": str(reference_file),
                "audio_file": str(audio_file),
            }
        )

    return discovered


def _run_rgb_event_dtw_alignment(
    sample_name: str,
    split_folder_name: str,
    side: str,
    dataset_folder: Path | str,
    output_path: Path | str,
    plot_output_folder: Path | str | None,
    window_seconds: float,
    resize_width: int,
    reference_file: Path | str | None = None,
    event_file: Path | str | None = None,
) -> dict[str, Any]:
    dataset_folder = Path(dataset_folder)
    output_path = Path(output_path)
    plot_output_folder = Path(plot_output_folder) if plot_output_folder is not None else None
    side = side.lower()
    if reference_file is None:
        reference_file = dataset_folder / split_folder_name / f"{sample_name}_{side}_rgb.mp4"
    else:
        reference_file = Path(reference_file)
    if event_file is None:
        event_file = dataset_folder / split_folder_name / f"{sample_name}_{side}_event.mp4"
    else:
        event_file = Path(event_file)

    reference_meta = _video_metadata(reference_file)
    event_meta = _video_metadata(event_file)
    warnings: list[str] = []
    result: dict[str, Any] = {
        "sample": sample_name,
        "pair_key": str(reference_file.parent / sample_name),
        "side": side,
        "method": "dynamic_time_warping_optical_flow",
        "reference_modality": REFERENCE_MODALITY,
        "reference_file": str(reference_file),
        "reference_duration_seconds": float(reference_meta["duration_seconds"]),
        "target_modality": "event",
        "target_file": str(event_file),
        "target_duration_seconds": float(event_meta["duration_seconds"]),
        "plot_file": None,
        "alignment": None,
        "export": None,
        "warnings": warnings,
    }

    if not reference_meta["opened"]:
        warnings.append(f"Could not open RGB reference video: {reference_file}")
    if not event_meta["opened"]:
        warnings.append(f"Could not open EVENT video: {event_file}")
    reference_fps = float(reference_meta["fps"])
    event_fps = float(event_meta["fps"])
    if reference_fps <= 0 or event_fps <= 0:
        warnings.append("Could not determine FPS for one or both videos.")
    if warnings:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        return result

    reference_trace = _optical_flow_magnitude_trace(reference_file, resize_width=resize_width)
    event_trace = _optical_flow_magnitude_trace(event_file, resize_width=resize_width)
    path, normalized_cost = _dynamic_time_warping_path(
        reference_trace,
        event_trace,
        reference_fps=reference_fps,
        target_fps=event_fps,
        window_seconds=window_seconds,
    )
    full_offset_curve = _offset_curve_from_dtw_path(path, reference_fps=reference_fps, target_fps=event_fps)
    offset_curve = _sample_offset_curve(full_offset_curve)
    offsets = np.asarray([item["offset_seconds"] for item in full_offset_curve], dtype=np.float32)
    start_offset = float(offsets[0]) if offsets.size else None
    end_offset = float(offsets[-1]) if offsets.size else None
    median_offset = float(np.median(offsets)) if offsets.size else None
    drift = float(end_offset - start_offset) if start_offset is not None and end_offset is not None else None

    reference_overlap, target_overlap = _overlap_windows(
        float(reference_meta["duration_seconds"]),
        float(event_meta["duration_seconds"]),
        median_offset,
    )
    alignment = {
        "modality": "event",
        "file": str(event_file),
        "duration_seconds": float(event_meta["duration_seconds"]),
        "offset_seconds": median_offset,
        "offset_frames": int(round(median_offset * min(reference_fps, event_fps))) if median_offset is not None else None,
        "start_offset_seconds": start_offset,
        "end_offset_seconds": end_offset,
        "offset_drift_seconds": drift,
        "dtw_normalized_cost": normalized_cost,
        "dtw_path_length": len(path),
        "dtw_window_seconds": float(window_seconds),
        "offset_curve": offset_curve,
        "offset_curve_full_count": len(full_offset_curve),
        "confidence_label": "diagnostic",
        "selected_by": "dynamic_time_warping",
        "peak_correlation": None,
        "candidate_offsets": [],
        "overlap_reference_seconds": reference_overlap,
        "overlap_target_seconds": target_overlap,
        "activity_plot_file": None,
        "warnings": warnings,
    }

    if plot_output_folder is not None:
        plot_path = plot_output_folder / f"{sample_name}_{side}_rgb_event_dtw_activity_signal.png"
        _write_dtw_activity_signal_plot(
            plot_path,
            f"DTW optical-flow activity signal: {sample_name} {side} RGB vs EVENT",
            reference_trace=reference_trace,
            reference_fps=reference_fps,
            event_trace=event_trace,
            event_fps=event_fps,
            offset_curve=full_offset_curve,
        )
        alignment["activity_plot_file"] = str(plot_path)
        result["plot_file"] = str(plot_path)

    result["alignment"] = alignment
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return result


def run_check_mailbox_day_rgb_event_feature_alignment(
    dataset_folder: Path | str = "dataset",
    output_path: Path | str = DEFAULT_FEATURE_CHECK_MAILBOX_OUTPUT_PATH,
    plot_output_folder: Path | str | None = DEFAULT_PLOT_OUTPUT_FOLDER,
    resize_width: int = FEATURE_ALIGNMENT_RESIZE_WIDTH,
    sample_stride_seconds: float = FEATURE_ALIGNMENT_SAMPLE_STRIDE_SECONDS,
    offset_step_seconds: float = FEATURE_ALIGNMENT_OFFSET_STEP_SECONDS,
    min_offset_seconds: float = FEATURE_ALIGNMENT_MIN_OFFSET_SECONDS,
    max_offset_seconds: float = FEATURE_ALIGNMENT_MAX_OFFSET_SECONDS,
) -> dict[str, Any]:
    dataset_folder = Path(dataset_folder)
    output_path = Path(output_path)
    plot_output_folder = Path(plot_output_folder) if plot_output_folder is not None else None
    reference_file = dataset_folder / "check_mailbox_split" / "check_mailbox_day_rgb.mp4"
    event_file = dataset_folder / "check_mailbox_split" / "check_mailbox_day_event.mp4"

    reference_meta = _video_metadata(reference_file)
    event_meta = _video_metadata(event_file)
    warnings: list[str] = []
    result: dict[str, Any] = {
        "pair_key": "dataset/check_mailbox_split/check_mailbox",
        "side": "day",
        "method": "feature_based_local_offsets",
        "reference_modality": REFERENCE_MODALITY,
        "reference_file": str(reference_file),
        "reference_duration_seconds": float(reference_meta["duration_seconds"]),
        "target_modality": "event",
        "target_file": str(event_file),
        "target_duration_seconds": float(event_meta["duration_seconds"]),
        "plot_file": None,
        "alignment": None,
        "comparison": {},
        "export": None,
        "warnings": warnings,
    }

    if not reference_meta["opened"]:
        warnings.append(f"Could not open RGB reference video: {reference_file}")
    if not event_meta["opened"]:
        warnings.append(f"Could not open EVENT video: {event_file}")
    reference_fps = float(reference_meta["fps"])
    event_fps = float(event_meta["fps"])
    if reference_fps <= 0 or event_fps <= 0:
        warnings.append("Could not determine FPS for one or both videos.")
    if warnings:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        return result

    dtw_curve = _load_dtw_offset_curve()
    if dtw_curve:
        result["comparison"]["dtw_curve_available"] = True
    else:
        result["comparison"]["dtw_curve_available"] = False
        warnings.append("DTW curve unavailable; feature search uses the full configured offset range.")

    orb = cv2.ORB_create(nfeatures=1200, fastThreshold=8)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    reference_feature_cache: dict[int, tuple[list[cv2.KeyPoint], np.ndarray | None]] = {}
    event_feature_cache: dict[int, tuple[list[cv2.KeyPoint], np.ndarray | None]] = {}

    def reference_features_at(time_seconds: float) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        frame_index = int(round(np.clip(time_seconds, 0.0, float(reference_meta["duration_seconds"])) * reference_fps))
        if frame_index not in reference_feature_cache:
            frame = _read_video_frame_by_index(reference_file, frame_index)
            reference_feature_cache[frame_index] = _detect_orb_features(frame, REFERENCE_MODALITY, orb, resize_width=resize_width)
        return reference_feature_cache[frame_index]

    def event_features_at(time_seconds: float) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        frame_index = int(round(np.clip(time_seconds, 0.0, float(event_meta["duration_seconds"])) * event_fps))
        if frame_index not in event_feature_cache:
            frame = _read_video_frame_by_index(event_file, frame_index)
            event_feature_cache[frame_index] = _detect_orb_features(frame, "event", orb, resize_width=resize_width)
        return event_feature_cache[frame_index]

    window_times = np.arange(
        0.0,
        max(0.0, float(reference_meta["duration_seconds"])),
        max(0.5, sample_stride_seconds),
        dtype=np.float32,
    )
    windows: list[dict[str, Any]] = []
    for reference_time in window_times:
        reference_time = float(reference_time)
        reference_features = reference_features_at(reference_time)
        if dtw_curve:
            prior_event_time = _event_time_from_offset_curve(dtw_curve, reference_time)
            prior_offset = float(prior_event_time - reference_time)
            candidate_min = max(min_offset_seconds, prior_offset - 4.0)
            candidate_max = min(max_offset_seconds, prior_offset + 4.0)
        else:
            prior_offset = None
            candidate_min = min_offset_seconds
            candidate_max = max_offset_seconds

        candidates: list[dict[str, Any]] = []
        for offset in np.arange(candidate_min, candidate_max + offset_step_seconds * 0.5, offset_step_seconds):
            event_time = reference_time + float(offset)
            if event_time < 0.0 or event_time > float(event_meta["duration_seconds"]):
                continue
            score = _score_feature_match(reference_features, event_features_at(event_time), matcher)
            score.update(
                {
                    "offset_seconds": round(float(offset), 6),
                    "event_time_seconds": round(float(event_time), 6),
                }
            )
            candidates.append(score)

        candidates.sort(
            key=lambda item: (
                float(item.get("score", 0.0)),
                int(item.get("inlier_count", 0)),
                int(item.get("match_count", 0)),
            ),
            reverse=True,
        )
        best = candidates[0] if candidates else {}
        match_count = int(best.get("match_count", 0) or 0)
        inlier_count = int(best.get("inlier_count", 0) or 0)
        score = float(best.get("score", 0.0) or 0.0)
        confidence = _confidence_from_feature_window(match_count, inlier_count, score)
        selected_offset = best.get("offset_seconds")
        if selected_offset is None and prior_offset is not None:
            selected_offset = round(prior_offset, 6)
        windows.append(
            {
                "reference_time_seconds": round(reference_time, 6),
                "selected_offset_seconds": selected_offset,
                "selected_event_time_seconds": best.get("event_time_seconds"),
                "score": round(score, 6),
                "match_count": match_count,
                "inlier_count": inlier_count,
                "mean_distance": best.get("mean_distance"),
                "confidence_label": confidence,
                "prior_offset_seconds": round(prior_offset, 6) if prior_offset is not None else None,
                "candidate_offsets": candidates[:5],
            }
        )

    smoothed_curve = _interpolated_offset_curve_from_windows(
        windows,
        reference_duration=float(reference_meta["duration_seconds"]),
        target_duration=float(event_meta["duration_seconds"]),
        fps=reference_fps,
    )
    offset_curve = _sample_offset_curve(smoothed_curve)
    offsets = np.asarray([item["offset_seconds"] for item in smoothed_curve], dtype=np.float32)
    median_offset = float(np.median(offsets)) if offsets.size else None
    start_offset = float(offsets[0]) if offsets.size else None
    end_offset = float(offsets[-1]) if offsets.size else None
    drift = float(end_offset - start_offset) if start_offset is not None and end_offset is not None else None
    reference_overlap, target_overlap = _overlap_windows(
        float(reference_meta["duration_seconds"]),
        float(event_meta["duration_seconds"]),
        median_offset,
    )

    alignment = {
        "modality": "event",
        "file": str(event_file),
        "duration_seconds": float(event_meta["duration_seconds"]),
        "offset_seconds": median_offset,
        "offset_frames": int(round(median_offset * min(reference_fps, event_fps))) if median_offset is not None else None,
        "start_offset_seconds": start_offset,
        "end_offset_seconds": end_offset,
        "offset_drift_seconds": drift,
        "offset_curve": offset_curve,
        "smoothed_offset_curve": offset_curve,
        "offset_curve_full_count": len(smoothed_curve),
        "local_windows": windows,
        "feature_settings": {
            "resize_width": resize_width,
            "sample_stride_seconds": sample_stride_seconds,
            "offset_step_seconds": offset_step_seconds,
            "min_offset_seconds": min_offset_seconds,
            "max_offset_seconds": max_offset_seconds,
            "uses_dtw_prior_when_available": True,
        },
        "confidence_label": "diagnostic",
        "selected_by": "feature_based_local_offsets",
        "overlap_reference_seconds": reference_overlap,
        "overlap_target_seconds": target_overlap,
        "activity_plot_file": None,
        "warnings": warnings,
    }

    if dtw_curve:
        try:
            dtw_offsets = np.asarray([float(item["offset_seconds"]) for item in dtw_curve], dtype=np.float32)
            result["comparison"].update(
                {
                    "dtw_median_offset_seconds": float(np.median(dtw_offsets)),
                    "dtw_start_offset_seconds": float(dtw_offsets[0]),
                    "dtw_end_offset_seconds": float(dtw_offsets[-1]),
                    "feature_median_offset_seconds": median_offset,
                    "feature_start_offset_seconds": start_offset,
                    "feature_end_offset_seconds": end_offset,
                }
            )
        except Exception as exc:
            result["comparison"]["dtw_comparison_warning"] = str(exc)

    if plot_output_folder is not None:
        plot_path = plot_output_folder / "check_mailbox_day_rgb_event_feature_offsets.png"
        _write_feature_offset_plot(
            plot_path,
            "Feature-based local offsets: check_mailbox day RGB vs EVENT",
            windows=windows,
            offset_curve=smoothed_curve,
        )
        alignment["activity_plot_file"] = str(plot_path)
        result["plot_file"] = str(plot_path)

    result["alignment"] = alignment
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return result


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


def _load_optical_flow_alignment(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a top-level JSON object in {path}")
    return data


def _export_rgb_event_optical_flow_for_result(
    result: dict[str, Any],
    output_folder: Path,
    prefer_gpu: bool,
) -> dict[str, Any]:
    output_path = output_folder / "check_mailbox_day_rgb_event_optical_flow_aligned.mp4"

    reference_file = Path(str(result.get("reference_file", "")))
    reference_duration = float(result.get("reference_duration_seconds") or 0.0)
    alignment = result.get("alignment")
    if not isinstance(alignment, dict):
        raise ValueError("Optical-flow result must include an alignment object.")

    event_file = Path(str(alignment.get("file") or result.get("target_file") or ""))
    event_offset = alignment.get("offset_seconds")
    event_duration = alignment.get("duration_seconds") or result.get("target_duration_seconds")
    if event_offset is None or event_duration is None:
        raise ValueError("Optical-flow alignment must include offset_seconds and duration_seconds.")
    event_offset = float(event_offset)
    event_duration = float(event_duration)

    if not reference_file.exists():
        raise FileNotFoundError(f"RGB reference file does not exist: {reference_file}")
    if not event_file.exists():
        raise FileNotFoundError(f"EVENT file does not exist: {event_file}")
    if reference_duration <= 0:
        raise ValueError("reference_duration_seconds must be positive.")

    reference_start = max(0.0, -event_offset)
    reference_end = min(reference_duration, event_duration - event_offset)
    duration = max(0.0, reference_end - reference_start)
    if duration <= 0:
        raise ValueError("No positive RGB/EVENT optical-flow overlap window is available.")

    rgb_seek = reference_start
    event_seek = max(0.0, reference_start + event_offset)
    rgb_label = _ffmpeg_escape_text("RGB")
    event_label = _ffmpeg_escape_text(f"EVENT optical-flow offset {event_offset:.3f}s")
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
        "[rgb][event]hstack=inputs=2,format=yuv420p[outv]"
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
        str(event_file),
        "-filter_complex",
        filter_complex,
        "-map",
        "[outv]",
        "-an",
    ]
    encoder_used, error = _run_ffmpeg_with_optional_gpu(command_prefix, output_path, prefer_gpu=prefer_gpu)
    if encoder_used is None:
        raise RuntimeError(f"Failed to export aligned RGB/EVENT optical-flow video: {error}")

    return {
        "sample": "check_mailbox",
        "side": "day",
        "output_file": str(output_path),
        "reference_file": str(reference_file),
        "event_file": str(event_file),
        "event_offset_seconds": event_offset,
        "rgb_seek_seconds": round(rgb_seek, 6),
        "event_seek_seconds": round(event_seek, 6),
        "duration_seconds": round(duration, 6),
        "encoder": encoder_used,
        "gpu_fallback_warning": error if encoder_used == "libx264" and error else None,
    }


def export_check_mailbox_day_rgb_event_optical_flow_alignment(
    alignment_input_path: Path | str = DEFAULT_OPTICAL_FLOW_CHECK_MAILBOX_OUTPUT_PATH,
    output_folder: Path | str = DEFAULT_EXPORT_OUTPUT_FOLDER,
    prefer_gpu: bool = True,
) -> dict[str, Any]:
    """Export the check_mailbox day RGB/EVENT optical-flow alignment preview."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    exported: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    try:
        result = _load_optical_flow_alignment(Path(alignment_input_path))
        exported.append(
            _export_rgb_event_optical_flow_for_result(
                result=result,
                output_folder=output_folder,
                prefer_gpu=prefer_gpu,
            )
        )
    except Exception as exc:
        skipped.append(
            {
                "sample": "check_mailbox",
                "side": "day",
                "reason": str(exc),
            }
        )

    summary = {
        "exported_count": len(exported),
        "skipped_count": len(skipped),
        "exported": exported,
        "skipped": skipped,
    }
    summary_path = output_folder / "rgb_event_optical_flow_export_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    summary["summary_file"] = str(summary_path)
    return summary


def _export_rgb_event_dtw_for_result(
    result: dict[str, Any],
    output_folder: Path,
) -> dict[str, Any]:
    sample_name = str(result.get("sample") or Path(str(result.get("pair_key") or "check_mailbox")).name)
    side = str(result.get("side") or "day")
    output_path = output_folder / f"{sample_name}_{side}_rgb_event_dtw_sliced_aligned.mp4"

    reference_file = Path(str(result.get("reference_file", "")))
    reference_duration = float(result.get("reference_duration_seconds") or 0.0)
    alignment = result.get("alignment")
    if not isinstance(alignment, dict):
        raise ValueError("DTW result must include an alignment object.")

    event_file = Path(str(alignment.get("file") or result.get("target_file") or ""))
    event_duration = float(alignment.get("duration_seconds") or result.get("target_duration_seconds") or 0.0)
    offset_curve = alignment.get("offset_curve")
    if not isinstance(offset_curve, list) or not offset_curve:
        raise ValueError("DTW alignment must include a non-empty offset_curve.")
    render_offset_curve = _smooth_dtw_offset_curve_for_export(offset_curve)

    if not reference_file.exists():
        raise FileNotFoundError(f"RGB reference file does not exist: {reference_file}")
    if not event_file.exists():
        raise FileNotFoundError(f"EVENT file does not exist: {event_file}")
    if reference_duration <= 0 or event_duration <= 0:
        raise ValueError("DTW export requires positive RGB and EVENT durations.")

    reference_start = max(0.0, float(offset_curve[0]["reference_time_seconds"]))
    reference_end = min(reference_duration, float(offset_curve[-1]["reference_time_seconds"]))
    duration = max(0.0, reference_end - reference_start)
    if duration <= 0:
        raise ValueError("No positive RGB/EVENT DTW overlap window is available.")

    rgb_cap = cv2.VideoCapture(str(reference_file))
    event_cap = cv2.VideoCapture(str(event_file))
    if not rgb_cap.isOpened():
        raise ValueError(f"Could not open RGB reference video: {reference_file}")
    if not event_cap.isOpened():
        rgb_cap.release()
        raise ValueError(f"Could not open EVENT video: {event_file}")
    reference_fps = float(rgb_cap.get(cv2.CAP_PROP_FPS) or 0.0)
    event_fps = float(event_cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if reference_fps <= 0 or event_fps <= 0:
        rgb_cap.release()
        event_cap.release()
        raise ValueError("Could not determine FPS for one or both videos.")

    output_folder.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    tmp_output_path.unlink(missing_ok=True)
    writer = cv2.VideoWriter(
        str(tmp_output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(DTW_EXPORT_PREVIEW_FPS),
        (EXPORT_PANEL_WIDTH * 2, EXPORT_PANEL_HEIGHT),
    )
    if not writer.isOpened():
        rgb_cap.release()
        event_cap.release()
        tmp_output_path.unlink(missing_ok=True)
        raise RuntimeError("Could not create OpenCV VideoWriter for DTW preview.")

    frame_count = max(1, int(np.floor(duration * DTW_EXPORT_PREVIEW_FPS)))
    written_count = 0
    rgb_state: dict[str, Any] = {}
    event_state: dict[str, Any] = {}
    try:
        for frame_index in range(frame_count):
            rgb_time = reference_start + frame_index / float(DTW_EXPORT_PREVIEW_FPS)
            event_time = min(event_duration, max(0.0, _event_time_from_offset_curve(render_offset_curve, rgb_time)))
            offset = event_time - rgb_time
            rgb_frame_index = int(round(rgb_time * reference_fps))
            event_position = event_time * event_fps
            event_frame_index = int(np.floor(event_position))
            event_blend_weight = float(event_position - event_frame_index)

            rgb_panel = _prepare_preview_panel(_read_video_frame_forward(rgb_cap, rgb_frame_index, rgb_state))
            event_frame_a, event_frame_b = _read_video_frame_pair_forward(event_cap, event_frame_index, event_state)
            event_panel = _prepare_preview_panel(_blend_frames(event_frame_a, event_frame_b, event_blend_weight))
            _draw_label(rgb_panel, ["RGB", f"t={rgb_time:.2f}s"])
            _draw_label(event_panel, ["EVENT DTW", f"t={event_time:.2f}s", f"offset={offset:+.3f}s"])
            writer.write(np.hstack([rgb_panel, event_panel]))
            written_count += 1
    finally:
        writer.release()
        rgb_cap.release()
        event_cap.release()

    if written_count == 0:
        tmp_output_path.unlink(missing_ok=True)
        raise RuntimeError("DTW preview export wrote zero frames.")

    tmp_output_path.replace(output_path)
    return {
        "sample": sample_name,
        "side": side,
        "output_file": str(output_path),
        "reference_file": str(reference_file),
        "event_file": str(event_file),
        "rgb_start_seconds": round(reference_start, 6),
        "rgb_end_seconds": round(reference_end, 6),
        "duration_seconds": round(duration, 6),
        "preview_fps": DTW_EXPORT_PREVIEW_FPS,
        "frames_written": written_count,
        "encoder": "opencv-mp4v",
        "render_curve": {
            "source": "smoothed_offset_curve",
            "smoothing_seconds": DTW_EXPORT_SMOOTHING_SECONDS,
            "points": _sample_offset_curve(render_offset_curve),
            "full_count": len(render_offset_curve),
            "monotonic_event_time": True,
            "event_frame_interpolation": "linear_blend",
        },
    }


def export_check_mailbox_day_rgb_event_dtw_alignment(
    alignment_input_path: Path | str = DEFAULT_DTW_CHECK_MAILBOX_OUTPUT_PATH,
    output_folder: Path | str = DEFAULT_EXPORT_OUTPUT_FOLDER,
) -> dict[str, Any]:
    """Export the check_mailbox day RGB/EVENT DTW drift-corrected preview."""
    alignment_input_path = Path(alignment_input_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    exported: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    result: dict[str, Any] | None = None

    try:
        result = _load_optical_flow_alignment(alignment_input_path)
        exported.append(_export_rgb_event_dtw_for_result(result=result, output_folder=output_folder))
    except Exception as exc:
        skipped.append(
            {
                "sample": "check_mailbox",
                "side": "day",
                "reason": str(exc),
            }
        )

    summary = {
        "exported_count": len(exported),
        "skipped_count": len(skipped),
        "exported": exported,
        "skipped": skipped,
    }
    summary_path = output_folder / "rgb_event_dtw_export_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    summary["summary_file"] = str(summary_path)
    if result is not None:
        result["export"] = summary
        with open(alignment_input_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
    return summary


def run_and_export_check_mailbox_day_rgb_event_dtw_alignment(
    dataset_folder: Path | str = "dataset",
    output_path: Path | str = DEFAULT_DTW_CHECK_MAILBOX_OUTPUT_PATH,
    plot_output_folder: Path | str | None = DEFAULT_PLOT_OUTPUT_FOLDER,
    output_folder: Path | str = DEFAULT_EXPORT_OUTPUT_FOLDER,
    window_seconds: float = 10.0,
    resize_width: int = 160,
) -> dict[str, Any]:
    """Run DTW alignment and export its drift-corrected preview."""
    result = run_check_mailbox_day_rgb_event_dtw_alignment(
        dataset_folder=dataset_folder,
        output_path=output_path,
        plot_output_folder=plot_output_folder,
        window_seconds=window_seconds,
        resize_width=resize_width,
    )
    export_summary = export_check_mailbox_day_rgb_event_dtw_alignment(
        alignment_input_path=output_path,
        output_folder=output_folder,
    )
    result["export"] = export_summary
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return result


def run_and_export_all_rgb_event_dtw_alignments(
    dataset_folder: Path | str = "dataset",
    alignment_output_folder: Path | str = ".",
    plot_output_folder: Path | str | None = DEFAULT_PLOT_OUTPUT_FOLDER,
    output_folder: Path | str = DEFAULT_EXPORT_OUTPUT_FOLDER,
    window_seconds: float = 10.0,
    resize_width: int = 160,
) -> dict[str, Any]:
    """Run DTW alignment and export drift-corrected previews for every RGB/EVENT pair."""
    dataset_folder = Path(dataset_folder)
    alignment_output_folder = Path(alignment_output_folder)
    plot_output_folder = Path(plot_output_folder) if plot_output_folder is not None else None
    output_folder = Path(output_folder)
    alignment_output_folder.mkdir(parents=True, exist_ok=True)
    output_folder.mkdir(parents=True, exist_ok=True)

    exported: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    discovered = _discover_rgb_event_dtw_pairs(dataset_folder)

    for pair in discovered:
        sample_name = str(pair["sample"])
        side = str(pair["side"])
        reference_file = Path(pair["reference_file"])
        event_file = Path(pair["event_file"])
        alignment_path = _dtw_alignment_output_path(alignment_output_folder, sample_name, side)

        if not bool(pair.get("complete")):
            skipped.append(
                {
                    "sample": sample_name,
                    "side": side,
                    "reference_file": str(reference_file),
                    "event_file": str(event_file),
                    "alignment_file": str(alignment_path),
                    "reason": f"Missing EVENT file: {event_file}",
                }
            )
            continue

        try:
            result = _run_rgb_event_dtw_alignment(
                sample_name=sample_name,
                split_folder_name=str(pair["split_folder_name"]),
                side=side,
                dataset_folder=dataset_folder,
                output_path=alignment_path,
                plot_output_folder=plot_output_folder,
                window_seconds=window_seconds,
                resize_width=resize_width,
                reference_file=reference_file,
                event_file=event_file,
            )
            export_item = _export_rgb_event_dtw_for_result(result=result, output_folder=output_folder)
            export_item["alignment_file"] = str(alignment_path)
            exported.append(export_item)

            result["export"] = {
                "exported_count": 1,
                "skipped_count": 0,
                "exported": [export_item],
                "skipped": [],
            }
            with open(alignment_path, "w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2, ensure_ascii=False)
        except Exception as exc:
            skipped.append(
                {
                    "sample": sample_name,
                    "side": side,
                    "reference_file": str(reference_file),
                    "event_file": str(event_file),
                    "alignment_file": str(alignment_path),
                    "reason": str(exc),
                }
            )

    summary = {
        "discovered_count": len(discovered),
        "exported_count": len(exported),
        "skipped_count": len(skipped),
        "exported": exported,
        "skipped": skipped,
    }
    summary_path = output_folder / "rgb_event_dtw_all_export_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    summary["summary_file"] = str(summary_path)
    return summary


def _export_rgb_event_feature_for_result(
    result: dict[str, Any],
    output_folder: Path,
) -> dict[str, Any]:
    output_path = output_folder / "check_mailbox_day_rgb_event_feature_aligned.mp4"

    reference_file = Path(str(result.get("reference_file", "")))
    reference_duration = float(result.get("reference_duration_seconds") or 0.0)
    alignment = result.get("alignment")
    if not isinstance(alignment, dict):
        raise ValueError("Feature result must include an alignment object.")

    event_file = Path(str(alignment.get("file") or result.get("target_file") or ""))
    event_duration = float(alignment.get("duration_seconds") or result.get("target_duration_seconds") or 0.0)
    offset_curve = alignment.get("smoothed_offset_curve") or alignment.get("offset_curve")
    if not isinstance(offset_curve, list) or not offset_curve:
        raise ValueError("Feature alignment must include a non-empty offset_curve.")

    if not reference_file.exists():
        raise FileNotFoundError(f"RGB reference file does not exist: {reference_file}")
    if not event_file.exists():
        raise FileNotFoundError(f"EVENT file does not exist: {event_file}")
    if reference_duration <= 0 or event_duration <= 0:
        raise ValueError("Feature export requires positive RGB and EVENT durations.")

    render_offset_curve = _smooth_dtw_offset_curve_for_export(offset_curve)
    reference_start = max(0.0, float(render_offset_curve[0]["reference_time_seconds"]))
    reference_end = min(reference_duration, float(render_offset_curve[-1]["reference_time_seconds"]))
    duration = max(0.0, reference_end - reference_start)
    if duration <= 0:
        raise ValueError("No positive RGB/EVENT feature-alignment overlap window is available.")

    rgb_cap = cv2.VideoCapture(str(reference_file))
    event_cap = cv2.VideoCapture(str(event_file))
    if not rgb_cap.isOpened():
        raise ValueError(f"Could not open RGB reference video: {reference_file}")
    if not event_cap.isOpened():
        rgb_cap.release()
        raise ValueError(f"Could not open EVENT video: {event_file}")
    reference_fps = float(rgb_cap.get(cv2.CAP_PROP_FPS) or 0.0)
    event_fps = float(event_cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if reference_fps <= 0 or event_fps <= 0:
        rgb_cap.release()
        event_cap.release()
        raise ValueError("Could not determine FPS for one or both videos.")

    output_folder.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    tmp_output_path.unlink(missing_ok=True)
    writer = cv2.VideoWriter(
        str(tmp_output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(DTW_EXPORT_PREVIEW_FPS),
        (EXPORT_PANEL_WIDTH * 2, EXPORT_PANEL_HEIGHT),
    )
    if not writer.isOpened():
        rgb_cap.release()
        event_cap.release()
        tmp_output_path.unlink(missing_ok=True)
        raise RuntimeError("Could not create OpenCV VideoWriter for feature preview.")

    frame_count = max(1, int(np.floor(duration * DTW_EXPORT_PREVIEW_FPS)))
    written_count = 0
    rgb_state: dict[str, Any] = {}
    event_state: dict[str, Any] = {}
    try:
        for frame_index in range(frame_count):
            rgb_time = reference_start + frame_index / float(DTW_EXPORT_PREVIEW_FPS)
            event_time = min(event_duration, max(0.0, _event_time_from_offset_curve(render_offset_curve, rgb_time)))
            offset = event_time - rgb_time
            rgb_frame_index = int(round(rgb_time * reference_fps))
            event_position = event_time * event_fps
            event_frame_index = int(np.floor(event_position))
            event_blend_weight = float(event_position - event_frame_index)

            rgb_panel = _prepare_preview_panel(_read_video_frame_forward(rgb_cap, rgb_frame_index, rgb_state))
            event_frame_a, event_frame_b = _read_video_frame_pair_forward(event_cap, event_frame_index, event_state)
            event_panel = _prepare_preview_panel(_blend_frames(event_frame_a, event_frame_b, event_blend_weight))
            _draw_label(rgb_panel, ["RGB", f"t={rgb_time:.2f}s"])
            _draw_label(event_panel, ["EVENT feature", f"t={event_time:.2f}s", f"offset={offset:+.3f}s"])
            writer.write(np.hstack([rgb_panel, event_panel]))
            written_count += 1
    finally:
        writer.release()
        rgb_cap.release()
        event_cap.release()

    if written_count == 0:
        tmp_output_path.unlink(missing_ok=True)
        raise RuntimeError("Feature preview export wrote zero frames.")

    tmp_output_path.replace(output_path)
    return {
        "sample": "check_mailbox",
        "side": "day",
        "output_file": str(output_path),
        "reference_file": str(reference_file),
        "event_file": str(event_file),
        "rgb_start_seconds": round(reference_start, 6),
        "rgb_end_seconds": round(reference_end, 6),
        "duration_seconds": round(duration, 6),
        "preview_fps": DTW_EXPORT_PREVIEW_FPS,
        "frames_written": written_count,
        "encoder": "opencv-mp4v",
        "render_curve": {
            "source": "smoothed_offset_curve",
            "smoothing_seconds": DTW_EXPORT_SMOOTHING_SECONDS,
            "points": _sample_offset_curve(render_offset_curve),
            "full_count": len(render_offset_curve),
            "monotonic_event_time": True,
            "event_frame_interpolation": "linear_blend",
        },
    }


def export_check_mailbox_day_rgb_event_feature_alignment(
    alignment_input_path: Path | str = DEFAULT_FEATURE_CHECK_MAILBOX_OUTPUT_PATH,
    output_folder: Path | str = DEFAULT_EXPORT_OUTPUT_FOLDER,
) -> dict[str, Any]:
    """Export the check_mailbox day RGB/EVENT feature-alignment preview."""
    alignment_input_path = Path(alignment_input_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    exported: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    result: dict[str, Any] | None = None

    try:
        result = _load_optical_flow_alignment(alignment_input_path)
        exported.append(_export_rgb_event_feature_for_result(result=result, output_folder=output_folder))
    except Exception as exc:
        skipped.append(
            {
                "sample": "check_mailbox",
                "side": "day",
                "reason": str(exc),
            }
        )

    summary = {
        "exported_count": len(exported),
        "skipped_count": len(skipped),
        "exported": exported,
        "skipped": skipped,
    }
    summary_path = output_folder / "rgb_event_feature_export_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    summary["summary_file"] = str(summary_path)
    if result is not None:
        result["export"] = summary
        with open(alignment_input_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
    return summary


def run_and_export_check_mailbox_day_rgb_event_feature_alignment(
    dataset_folder: Path | str = "dataset",
    output_path: Path | str = DEFAULT_FEATURE_CHECK_MAILBOX_OUTPUT_PATH,
    plot_output_folder: Path | str | None = DEFAULT_PLOT_OUTPUT_FOLDER,
    output_folder: Path | str = DEFAULT_EXPORT_OUTPUT_FOLDER,
    resize_width: int = FEATURE_ALIGNMENT_RESIZE_WIDTH,
) -> dict[str, Any]:
    """Run feature-based alignment and export its drift-corrected preview."""
    result = run_check_mailbox_day_rgb_event_feature_alignment(
        dataset_folder=dataset_folder,
        output_path=output_path,
        plot_output_folder=plot_output_folder,
        resize_width=resize_width,
    )
    export_summary = export_check_mailbox_day_rgb_event_feature_alignment(
        alignment_input_path=output_path,
        output_folder=output_folder,
    )
    result["export"] = export_summary
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return result


def _export_rgb_audio_cross_correlation_for_result(
    result: dict[str, Any],
    output_folder: Path,
    prefer_gpu: bool = True,
) -> dict[str, Any]:
    sample_name = str(result.get("sample") or "check_mailbox")
    side = str(result.get("side") or "day")
    output_path = output_folder / f"{sample_name}_{side}_rgb_audio_cross_correlation_aligned.mp4"
    reference_file = Path(str(result.get("reference_file", "")))
    audio_file = Path(str(result.get("audio_file", "")))
    reference_duration = float(result.get("reference_duration_seconds") or 0.0)
    audio_duration = float(result.get("audio_duration_seconds") or 0.0)
    reference_fps = float(result.get("reference_fps") or DTW_EXPORT_PREVIEW_FPS)
    alignment = result.get("alignment") if isinstance(result.get("alignment"), dict) else {}
    offset_seconds = alignment.get("offset_seconds")

    if offset_seconds is None:
        raise ValueError("RGB/AUDIO alignment must include offset_seconds.")
    offset_seconds = float(offset_seconds)
    if not reference_file.exists():
        raise FileNotFoundError(f"RGB reference file does not exist: {reference_file}")
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio_file}")
    if reference_duration <= 0 or audio_duration <= 0:
        raise ValueError("RGB/AUDIO export requires positive RGB and audio durations.")

    output_folder.mkdir(parents=True, exist_ok=True)
    if offset_seconds >= audio_duration:
        raise ValueError("Selected offset seeks beyond the end of the audio file.")

    preview_fps = min(max(reference_fps, 1.0), DTW_EXPORT_PREVIEW_FPS)
    audio_seek = max(0.0, offset_seconds)
    audio_delay_seconds = max(0.0, -offset_seconds)
    audio_label = _ffmpeg_escape_text(f"AUDIO offset {offset_seconds:+.3f}s")
    rgb_label = _ffmpeg_escape_text("RGB")
    summary_label = _ffmpeg_escape_text(
        f"RGB/AUDIO cross-correlation corr={float(alignment.get('peak_correlation') or 0.0):.3f}"
    )

    audio_filter = "[1:a]apad[a]"
    if audio_delay_seconds > 0:
        delay_ms = int(round(audio_delay_seconds * 1000.0))
        audio_filter = f"[1:a]adelay={delay_ms}:all=1,apad[a]"

    filter_complex = (
        f"[0:v]fps={preview_fps:.6f},"
        f"scale={EXPORT_PANEL_WIDTH * 2}:{EXPORT_PANEL_HEIGHT * 2}:force_original_aspect_ratio=decrease,"
        f"pad={EXPORT_PANEL_WIDTH * 2}:{EXPORT_PANEL_HEIGHT * 2}:(ow-iw)/2:(oh-ih)/2,"
        f"drawtext=text='{rgb_label}':x=12:y=12:fontsize={EXPORT_LABEL_FONT_SIZE}:fontcolor=white:"
        f"box=1:boxcolor=black@0.55:boxborderw=8,"
        f"drawtext=text='{audio_label}':x=12:y=44:fontsize={EXPORT_LABEL_FONT_SIZE}:fontcolor=white:"
        f"box=1:boxcolor=black@0.55:boxborderw=8,"
        f"drawtext=text='{summary_label}':x=12:y=76:fontsize={EXPORT_LABEL_FONT_SIZE}:fontcolor=white:"
        f"box=1:boxcolor=black@0.55:boxborderw=8,"
        "setsar=1,format=yuv420p[outv];"
        f"{audio_filter}"
    )
    command_prefix = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(reference_file),
    ]
    if audio_seek > 0:
        command_prefix.extend(["-ss", f"{audio_seek:.6f}"])
    command_prefix.extend(
        [
            "-i",
            str(audio_file),
            "-filter_complex",
            filter_complex,
            "-map",
            "[outv]",
            "-map",
            "[a]",
            "-t",
            f"{reference_duration:.6f}",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
        ]
    )
    encoder_used, error = _run_ffmpeg_with_optional_gpu(command_prefix, output_path, prefer_gpu=prefer_gpu)
    if encoder_used is None:
        raise RuntimeError(f"Failed to export aligned RGB/AUDIO video: {error}")

    reference_overlap, audio_overlap = _overlap_windows(reference_duration, audio_duration, offset_seconds)
    return {
        "sample": sample_name,
        "side": side,
        "output_file": str(output_path),
        "reference_file": str(reference_file),
        "audio_file": str(audio_file),
        "offset_seconds": round(offset_seconds, 6),
        "correlation": round(float(alignment.get("peak_correlation") or 0.0), 6),
        "overlap_ratio": round(float(alignment.get("overlap_ratio") or 0.0), 6),
        "overlap_reference_seconds": reference_overlap,
        "overlap_audio_seconds": audio_overlap,
        "duration_seconds": round(reference_duration, 6),
        "preview_fps": round(float(preview_fps), 6),
        "frames_written": int(round(reference_duration * preview_fps)),
        "encoder": encoder_used,
        "audio_encoder": "aac",
        "audio_shift": {
            "source_seek_seconds": round(audio_seek, 6),
            "output_delay_seconds": round(audio_delay_seconds, 6),
            "speed_warped": False,
        },
    }


def _run_rgb_audio_cross_correlation_alignment(
    sample_name: str,
    split_folder_name: str,
    side: str,
    dataset_folder: Path | str = "dataset",
    output_path: Path | str = DEFAULT_RGB_AUDIO_CHECK_MAILBOX_OUTPUT_PATH,
    plot_output_folder: Path | str | None = DEFAULT_PLOT_OUTPUT_FOLDER,
    output_folder: Path | str = DEFAULT_EXPORT_OUTPUT_FOLDER,
    resize_width: int = 160,
    max_lag_seconds: float = AUDIO_ALIGNMENT_MAX_LAG_SECONDS,
    prefer_gpu: bool = True,
    reference_file: Path | str | None = None,
    audio_file: Path | str | None = None,
    export_preview: bool = True,
    summary_file_name: str = "rgb_audio_cross_correlation_export_summary.json",
) -> dict[str, Any]:
    """Align RGB with its separate .m4a audio using one fixed offset."""
    dataset_folder = Path(dataset_folder)
    output_path = Path(output_path)
    plot_output_folder = Path(plot_output_folder) if plot_output_folder is not None else None
    output_folder = Path(output_folder)
    side = side.lower()
    if reference_file is None:
        reference_file = dataset_folder / split_folder_name / f"{sample_name}_{side}_rgb.mp4"
    else:
        reference_file = Path(reference_file)
    if audio_file is None:
        audio_file = dataset_folder / split_folder_name / f"{sample_name}_{side}.m4a"
    else:
        audio_file = Path(audio_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_folder.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "sample": sample_name,
        "side": side,
        "method": "raw_cross_correlation_rgb_optical_flow_audio_rms",
        "reference_modality": "rgb",
        "target_modality": "audio",
        "reference_file": str(reference_file),
        "audio_file": str(audio_file),
        "ignored_files": [str(reference_file.with_name(f"{sample_name}_{side}_rgb_with_audio.mp4"))],
        "speed_warped": False,
        "plot_file": None,
        "warnings": [],
    }
    exported: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    try:
        reference_metadata = _video_metadata(reference_file)
        if not bool(reference_metadata.get("opened")):
            raise ValueError(f"Could not open RGB reference video: {reference_file}")
        reference_fps = float(reference_metadata.get("fps") or 0.0)
        reference_duration = float(reference_metadata.get("duration_seconds") or 0.0)
        if reference_fps <= 0 or reference_duration <= 0:
            raise ValueError("RGB reference video has invalid FPS or duration.")

        reference_trace = _optical_flow_magnitude_trace(reference_file, resize_width=resize_width)
        audio_trace, audio_duration = _audio_energy_trace(audio_file, fps=reference_fps)
        alignment = _estimate_raw_cross_correlation_offset(
            reference_trace,
            audio_trace,
            fps=reference_fps,
            max_lag_seconds=max_lag_seconds,
        )
        reference_overlap, audio_overlap = _overlap_windows(
            reference_duration,
            audio_duration,
            alignment.get("offset_seconds"),
        )
        alignment["overlap_reference_seconds"] = reference_overlap
        alignment["overlap_audio_seconds"] = audio_overlap
        if alignment.get("candidate_offsets"):
            selected_frames = int(alignment.get("offset_frames") or 0)
            reference_overlap_trace, audio_overlap_trace = _lag_overlaps(
                _prepare_alignment_trace(reference_trace, reference_fps),
                _prepare_alignment_trace(audio_trace, reference_fps),
                selected_frames,
            )
            max_overlap = max(1, min(reference_trace.size, audio_trace.size))
            alignment["overlap_ratio"] = float(reference_overlap_trace.size / max_overlap)
            alignment["overlap_samples"] = int(reference_overlap_trace.size)
            alignment["selected_correlation_recomputed"] = _overlap_correlation(
                reference_overlap_trace,
                audio_overlap_trace,
            )
        result.update(
            {
                "reference_fps": reference_fps,
                "reference_frame_count": int(reference_metadata.get("frame_count") or 0),
                "reference_duration_seconds": reference_duration,
                "audio_duration_seconds": audio_duration,
                "reference_trace_count": int(reference_trace.size),
                "audio_trace_count": int(audio_trace.size),
                "alignment": alignment,
            }
        )

        if plot_output_folder is not None:
            plot_path = plot_output_folder / f"{sample_name}_{side}_rgb_audio_cross_correlation_activity_signal.png"
            try:
                _write_rgb_audio_activity_signal_plot(
                    plot_path,
                    f"RGB/audio cross-correlation: {sample_name} {side}",
                    reference_trace=reference_trace,
                    reference_fps=reference_fps,
                    audio_trace=audio_trace,
                    audio_fps=reference_fps,
                    alignment=alignment,
                )
                alignment["activity_plot_file"] = str(plot_path)
                result["plot_file"] = str(plot_path)
            except Exception as exc:
                warning = f"Could not write RGB/AUDIO activity plot: {exc}"
                result["warnings"].append(warning)
                alignment.setdefault("warnings", []).append(warning)

        if alignment.get("offset_seconds") is None:
            raise ValueError("No RGB/AUDIO offset could be estimated.")
        if export_preview:
            exported.append(
                _export_rgb_audio_cross_correlation_for_result(
                    result=result,
                    output_folder=output_folder,
                    prefer_gpu=prefer_gpu,
                )
            )
    except Exception as exc:
        skipped.append(
            {
                "sample": sample_name,
                "side": side,
                "reference_file": str(reference_file),
                "audio_file": str(audio_file),
                "reason": str(exc),
            }
        )

    export_summary = {
        "exported_count": len(exported),
        "skipped_count": len(skipped),
        "exported": exported,
        "skipped": skipped,
    }
    summary_path = output_folder / summary_file_name
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(export_summary, handle, indent=2)
    export_summary["summary_file"] = str(summary_path)
    result["export"] = export_summary
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return result


def run_and_export_check_mailbox_day_rgb_audio_cross_correlation_alignment(
    dataset_folder: Path | str = "dataset",
    output_path: Path | str = DEFAULT_RGB_AUDIO_CHECK_MAILBOX_OUTPUT_PATH,
    plot_output_folder: Path | str | None = DEFAULT_PLOT_OUTPUT_FOLDER,
    output_folder: Path | str = DEFAULT_EXPORT_OUTPUT_FOLDER,
    resize_width: int = 160,
    max_lag_seconds: float = AUDIO_ALIGNMENT_MAX_LAG_SECONDS,
    prefer_gpu: bool = True,
) -> dict[str, Any]:
    """Align check_mailbox day RGB with its separate .m4a audio using one fixed offset."""
    return _run_rgb_audio_cross_correlation_alignment(
        sample_name="check_mailbox",
        split_folder_name="check_mailbox_split",
        side="day",
        dataset_folder=dataset_folder,
        output_path=output_path,
        plot_output_folder=plot_output_folder,
        output_folder=output_folder,
        resize_width=resize_width,
        max_lag_seconds=max_lag_seconds,
        prefer_gpu=prefer_gpu,
        export_preview=True,
        summary_file_name="rgb_audio_cross_correlation_export_summary.json",
    )


def _mux_rgb_video_with_aligned_audio(
    reference_file: Path,
    audio_file: Path,
    output_file: Path,
    audio_offset_seconds: float,
    reference_duration_seconds: float,
    audio_duration_seconds: float,
) -> dict[str, Any]:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_file.with_name(f"{output_file.stem}.tmp{output_file.suffix}")
    tmp_output_path.unlink(missing_ok=True)

    audio_seek_seconds = max(0.0, audio_offset_seconds)
    audio_delay_seconds = max(0.0, -audio_offset_seconds)
    if audio_seek_seconds >= audio_duration_seconds:
        raise ValueError("Aligned audio seek starts beyond the end of the audio file.")

    audio_filter = "[1:a]apad[a]"
    if audio_delay_seconds > 0:
        delay_ms = int(round(audio_delay_seconds * 1000.0))
        audio_filter = f"[1:a]adelay={delay_ms}:all=1,apad[a]"

    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(reference_file)]
    if audio_seek_seconds > 0:
        command.extend(["-ss", f"{audio_seek_seconds:.6f}"])
    command.extend(
        [
            "-i",
            str(audio_file),
            "-filter_complex",
            audio_filter,
            "-map",
            "0:v:0",
            "-map",
            "[a]",
            "-t",
            f"{reference_duration_seconds:.6f}",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(tmp_output_path),
        ]
    )
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        tmp_output_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to mux RGB video with aligned audio: {completed.stderr.strip()}")

    tmp_output_path.replace(output_file)
    return {
        "output_file": str(output_file),
        "reference_file": str(reference_file),
        "audio_file": str(audio_file),
        "duration_seconds": round(reference_duration_seconds, 6),
        "audio_shift": {
            "source_seek_seconds": round(audio_seek_seconds, 6),
            "output_delay_seconds": round(audio_delay_seconds, 6),
            "speed_warped": False,
        },
        "video_stream": "rgb_video_copy",
        "audio_encoder": "aac",
    }


def _mux_rgb_video_segment_with_aligned_audio(
    reference_file: Path,
    audio_file: Path,
    output_file: Path,
    segment_start_seconds: float,
    duration_seconds: float,
    audio_offset_seconds: float,
) -> dict[str, Any]:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_file.with_name(f"{output_file.stem}.tmp{output_file.suffix}")
    tmp_output_path.unlink(missing_ok=True)

    audio_segment_start = segment_start_seconds + audio_offset_seconds
    audio_seek_seconds = max(0.0, audio_segment_start)
    audio_delay_seconds = max(0.0, -audio_segment_start)
    audio_filter = "[1:a]apad[a]"
    if audio_delay_seconds > 0:
        delay_ms = int(round(audio_delay_seconds * 1000.0))
        audio_filter = f"[1:a]adelay={delay_ms}:all=1,apad[a]"

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{segment_start_seconds:.6f}",
        "-t",
        f"{duration_seconds:.6f}",
        "-i",
        str(reference_file),
    ]
    if audio_seek_seconds > 0:
        command.extend(["-ss", f"{audio_seek_seconds:.6f}"])
    command.extend(
        [
            "-i",
            str(audio_file),
            "-filter_complex",
            audio_filter,
            "-map",
            "0:v:0",
            "-map",
            "[a]",
            "-t",
            f"{duration_seconds:.6f}",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(tmp_output_path),
        ]
    )
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        tmp_output_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to mux RGB segment with aligned audio: {completed.stderr.strip()}")

    tmp_output_path.replace(output_file)
    return {
        "output_file": str(output_file),
        "reference_file": str(reference_file),
        "audio_file": str(audio_file),
        "segment_start_seconds": round(segment_start_seconds, 6),
        "duration_seconds": round(duration_seconds, 6),
        "audio_shift": {
            "source_seek_seconds": round(audio_seek_seconds, 6),
            "output_delay_seconds": round(audio_delay_seconds, 6),
            "speed_warped": False,
        },
        "video_stream": "rgb_video_copy",
        "audio_encoder": "aac",
    }


def _export_one_aligned_rgb_with_audio(
    pair: dict[str, Any],
    resize_width: int,
    max_lag_seconds: float,
    overwrite: bool,
    plot_output_folder: Path | None,
) -> dict[str, Any]:
    reference_file = Path(str(pair["reference_file"]))
    audio_file = Path(str(pair["audio_file"]))
    output_file = Path(str(pair["output_file"]))
    plot_path = None
    if plot_output_folder is not None:
        plot_path = (
            plot_output_folder
            / str(pair.get("split_folder_name", "unknown_split"))
            / str(pair.get("segment_name", "unknown_segment"))
            / f"{pair.get('sample_name', output_file.stem)}_{pair.get('side', 'unknown')}_rgb_audio_cross_correlation_activity_signal.png"
        )

    output_exists = output_file.exists() and not overwrite
    if output_exists and (plot_path is None or plot_path.exists()):
        return {
            **pair,
            "output_file": str(output_file),
            "plot_file": str(plot_path) if plot_path else None,
            "status": "reused",
            "reason": "Output already exists.",
        }

    reference_metadata = _video_metadata(reference_file)
    if not bool(reference_metadata.get("opened")):
        raise ValueError(f"Could not open RGB reference video: {reference_file}")
    reference_fps = float(reference_metadata.get("fps") or 0.0)
    reference_duration = float(reference_metadata.get("duration_seconds") or 0.0)
    if reference_fps <= 0 or reference_duration <= 0:
        raise ValueError("RGB reference video has invalid FPS or duration.")

    reference_trace = _optical_flow_magnitude_trace(reference_file, resize_width=resize_width)
    audio_trace, audio_duration = _audio_energy_trace(audio_file, fps=reference_fps)
    alignment = _estimate_raw_cross_correlation_offset(
        reference_trace,
        audio_trace,
        fps=reference_fps,
        max_lag_seconds=max_lag_seconds,
    )
    if alignment.get("offset_seconds") is None:
        raise ValueError("No RGB/AUDIO offset could be estimated.")
    if alignment.get("candidate_offsets"):
        selected_frames = int(alignment.get("offset_frames") or 0)
        reference_overlap_trace, audio_overlap_trace = _lag_overlaps(
            _prepare_alignment_trace(reference_trace, reference_fps),
            _prepare_alignment_trace(audio_trace, reference_fps),
            selected_frames,
        )
        max_overlap = max(1, min(reference_trace.size, audio_trace.size))
        alignment["overlap_ratio"] = float(reference_overlap_trace.size / max_overlap)
        alignment["overlap_samples"] = int(reference_overlap_trace.size)

    plot_warning = None
    if plot_path is not None:
        try:
            _write_rgb_audio_activity_signal_plot(
                plot_path,
                (
                    "Aligned RGB/audio cross-correlation: "
                    f"{pair.get('sample_name')} {pair.get('side')} {pair.get('segment_name')}"
                ),
                reference_trace=reference_trace,
                reference_fps=reference_fps,
                audio_trace=audio_trace,
                audio_fps=reference_fps,
                alignment=alignment,
            )
            alignment["activity_plot_file"] = str(plot_path)
        except Exception as exc:
            plot_warning = f"Could not write RGB/AUDIO activity plot: {exc}"

    alignment_summary = {
        "offset_seconds": round(float(alignment["offset_seconds"]), 6),
        "offset_frames": alignment.get("offset_frames"),
        "peak_correlation": round(float(alignment.get("peak_correlation") or 0.0), 6),
        "confidence_label": alignment.get("confidence_label"),
        "overlap_ratio": (
            round(float(alignment.get("overlap_ratio")), 6)
            if alignment.get("overlap_ratio") is not None
            else (
                round(float(alignment["candidate_offsets"][0].get("overlap_ratio")), 6)
                if alignment.get("candidate_offsets")
                else None
            )
        ),
        "candidate_offsets": alignment.get("candidate_offsets", []),
        "activity_plot_file": str(plot_path) if plot_path and plot_path.exists() else None,
    }
    if plot_warning:
        alignment_summary["plot_warning"] = plot_warning

    if output_exists:
        return {
            **pair,
            "status": "reused",
            "reason": "Output already exists.",
            "reference_duration_seconds": round(reference_duration, 6),
            "audio_duration_seconds": round(audio_duration, 6),
            "reference_fps": round(reference_fps, 6),
            "alignment": alignment_summary,
            "plot_file": alignment_summary.get("activity_plot_file"),
        }

    mux = _mux_rgb_video_with_aligned_audio(
        reference_file=reference_file,
        audio_file=audio_file,
        output_file=output_file,
        audio_offset_seconds=float(alignment["offset_seconds"]),
        reference_duration_seconds=reference_duration,
        audio_duration_seconds=audio_duration,
    )
    return {
        **pair,
        "status": "exported",
        "reference_duration_seconds": round(reference_duration, 6),
        "audio_duration_seconds": round(audio_duration, 6),
        "reference_fps": round(reference_fps, 6),
        "alignment": alignment_summary,
        "plot_file": alignment_summary.get("activity_plot_file"),
        "mux": mux,
    }


def run_and_export_aligned_rgb_with_audio_segments(
    dataset_folder: Path | str = "aligned_dataset",
    summary_output_path: Path | str = "aligned_dataset/aligned_rgb_with_audio_export_summary.json",
    plot_output_folder: Path | str | None = DEFAULT_PLOT_OUTPUT_FOLDER / "aligned_rgb_audio",
    resize_width: int = 160,
    max_lag_seconds: float = AUDIO_ALIGNMENT_MAX_LAG_SECONDS,
    overwrite: bool = False,
    max_pairs: int | None = None,
    start_pair_index: int = 0,
    max_pair_groups: int | None = None,
    start_pair_group_index: int = 0,
) -> dict[str, Any]:
    """Create *_rgb_with_audio.mp4 files for aligned RGB/.m4a segment pairs."""
    dataset_folder = Path(dataset_folder)
    summary_output_path = Path(summary_output_path)
    plot_output_folder = Path(plot_output_folder) if plot_output_folder is not None else None
    all_discovered = discover_aligned_rgb_audio_segment_pairs(dataset_folder)
    discovered = all_discovered[start_pair_index:]
    selected_pair_groups: list[dict[str, str]] = []
    if max_pair_groups is not None:
        pair_group_keys: list[tuple[str, str]] = []
        for pair in all_discovered:
            group_key = (
                str(pair.get("split_folder_name", "")),
                str(pair.get("sample_name", "")),
            )
            if group_key not in pair_group_keys:
                pair_group_keys.append(group_key)
        selected_keys = pair_group_keys[start_pair_group_index:]
        selected_keys = selected_keys[:max_pair_groups]
        selected_key_set = set(selected_keys)
        discovered = [
            pair
            for pair in all_discovered
            if (
                str(pair.get("split_folder_name", "")),
                str(pair.get("sample_name", "")),
            )
            in selected_key_set
        ]
        selected_pair_groups = [
            {"split_folder_name": split_folder_name, "sample_name": sample_name}
            for split_folder_name, sample_name in selected_keys
        ]
    if max_pairs is not None:
        discovered = discovered[:max_pairs]
    exported: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for pair in discovered:
        try:
            result = _export_one_aligned_rgb_with_audio(
                pair,
                resize_width=resize_width,
                max_lag_seconds=max_lag_seconds,
                overwrite=overwrite,
                plot_output_folder=plot_output_folder,
            )
            exported.append(result)
        except Exception as exc:
            skipped.append({**pair, "reason": str(exc)})

    summary = {
        "dataset_folder": str(dataset_folder),
        "method": "rgb_optical_flow_audio_rms_cross_correlation_fixed_offset_mux",
        "output_naming": "*_rgb_with_audio.mp4",
        "plot_output_folder": str(plot_output_folder) if plot_output_folder else None,
        "total_discovered_count": len(all_discovered),
        "start_pair_index": start_pair_index,
        "max_pairs": max_pairs,
        "start_pair_group_index": start_pair_group_index,
        "max_pair_groups": max_pair_groups,
        "selected_pair_groups": selected_pair_groups,
        "discovered_count": len(discovered),
        "exported_count": len(exported),
        "skipped_count": len(skipped),
        "exported": exported,
        "skipped": skipped,
    }
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    summary["summary_file"] = str(summary_output_path)
    return summary


def _source_pair_group_key(pair: dict[str, Any]) -> tuple[str, str]:
    return (
        str(pair.get("split_folder_name", "")),
        str(pair.get("sample_name", "")),
    )


def _select_source_rgb_audio_pairs(
    discovered: list[dict[str, Any]],
    max_pair_groups: int | None,
    start_pair_group_index: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if max_pair_groups is None:
        return discovered, []

    pair_group_keys: list[tuple[str, str]] = []
    for pair in discovered:
        group_key = _source_pair_group_key(pair)
        if group_key not in pair_group_keys:
            pair_group_keys.append(group_key)

    selected_keys = pair_group_keys[start_pair_group_index:]
    selected_keys = selected_keys[:max_pair_groups]
    selected_key_set = set(selected_keys)
    selected_pairs = [pair for pair in discovered if _source_pair_group_key(pair) in selected_key_set]
    selected_groups = [
        {"split_folder_name": split_folder_name, "sample_name": sample_name}
        for split_folder_name, sample_name in selected_keys
    ]
    return selected_pairs, selected_groups


def _estimate_rgb_audio_alignment_for_source_pair(
    pair: dict[str, Any],
    resize_width: int,
    max_lag_seconds: float,
) -> dict[str, Any]:
    reference_file = Path(str(pair["reference_file"]))
    audio_file = Path(str(pair["audio_file"]))
    reference_metadata = _video_metadata(reference_file)
    if not bool(reference_metadata.get("opened")):
        raise ValueError(f"Could not open RGB source video: {reference_file}")
    reference_fps = float(reference_metadata.get("fps") or 0.0)
    reference_duration = float(reference_metadata.get("duration_seconds") or 0.0)
    if reference_fps <= 0 or reference_duration <= 0:
        raise ValueError("RGB source video has invalid FPS or duration.")

    reference_trace = _optical_flow_magnitude_trace(reference_file, resize_width=resize_width)
    audio_trace, audio_duration = _audio_energy_trace(audio_file, fps=reference_fps)
    alignment = _estimate_raw_cross_correlation_offset(
        reference_trace,
        audio_trace,
        fps=reference_fps,
        max_lag_seconds=max_lag_seconds,
    )
    if alignment.get("offset_seconds") is None:
        raise ValueError("No RGB/AUDIO offset could be estimated.")
    if alignment.get("candidate_offsets"):
        selected_frames = int(alignment.get("offset_frames") or 0)
        reference_overlap_trace, audio_overlap_trace = _lag_overlaps(
            _prepare_alignment_trace(reference_trace, reference_fps),
            _prepare_alignment_trace(audio_trace, reference_fps),
            selected_frames,
        )
        max_overlap = max(1, min(reference_trace.size, audio_trace.size))
        alignment["overlap_ratio"] = float(reference_overlap_trace.size / max_overlap)
        alignment["overlap_samples"] = int(reference_overlap_trace.size)

    return {
        "reference_fps": reference_fps,
        "reference_duration_seconds": reference_duration,
        "audio_duration_seconds": audio_duration,
        "reference_trace": reference_trace,
        "audio_trace": audio_trace,
        "alignment": alignment,
    }


def _write_source_rgb_audio_alignment_plot(
    plot_path: Path | None,
    pair: dict[str, Any],
    alignment_bundle: dict[str, Any],
) -> str | None:
    if plot_path is None:
        return None

    _write_rgb_audio_activity_signal_plot(
        plot_path,
        (
            "Source RGB/audio cross-correlation: "
            f"{pair.get('sample_name')} {pair.get('side')}"
        ),
        reference_trace=alignment_bundle["reference_trace"],
        reference_fps=float(alignment_bundle["reference_fps"]),
        audio_trace=alignment_bundle["audio_trace"],
        audio_fps=float(alignment_bundle["reference_fps"]),
        alignment=alignment_bundle["alignment"],
    )
    return str(plot_path)


def _source_rgb_audio_alignment_summary(alignment: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "offset_seconds": round(float(alignment["offset_seconds"]), 6),
        "offset_frames": alignment.get("offset_frames"),
        "peak_correlation": round(float(alignment.get("peak_correlation") or 0.0), 6),
        "confidence_label": alignment.get("confidence_label"),
        "overlap_ratio": (
            round(float(alignment.get("overlap_ratio")), 6)
            if alignment.get("overlap_ratio") is not None
            else None
        ),
        "candidate_offsets": alignment.get("candidate_offsets", []),
    }
    if alignment.get("confidence_label") == "low":
        summary["warnings"] = [
            "Low-confidence RGB/audio cross-correlation; selected offset is constrained by the conservative source-audio search window."
        ]
    return summary


def run_and_export_source_rgb_with_audio_segments_for_aligned_dataset(
    source_dataset_folder: Path | str = "dataset",
    aligned_dataset_folder: Path | str = "aligned_dataset",
    summary_output_path: Path | str = "aligned_dataset/source_rgb_with_audio_one_pair_export_summary.json",
    plot_output_folder: Path | str | None = DEFAULT_PLOT_OUTPUT_FOLDER / "aligned_rgb_audio",
    segment_seconds: float = 30.0,
    resize_width: int = 160,
    max_lag_seconds: float = SOURCE_RGB_AUDIO_ALIGNMENT_MAX_LAG_SECONDS,
    overwrite: bool = False,
    max_pair_groups: int | None = 1,
    start_pair_group_index: int = 0,
    verbose: bool = False,
) -> dict[str, Any]:
    """Create aligned_dataset 30s *_rgb_with_audio.mp4 segments from source dataset RGB/.m4a files."""
    source_dataset_folder = Path(source_dataset_folder)
    aligned_dataset_folder = Path(aligned_dataset_folder)
    summary_output_path = Path(summary_output_path)
    plot_output_folder = Path(plot_output_folder) if plot_output_folder is not None else None
    all_discovered = discover_source_rgb_audio_pairs(source_dataset_folder)
    discovered, selected_pair_groups = _select_source_rgb_audio_pairs(
        all_discovered,
        max_pair_groups=max_pair_groups,
        start_pair_group_index=start_pair_group_index,
    )
    exported: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for pair in discovered:
        try:
            alignment_bundle = _estimate_rgb_audio_alignment_for_source_pair(
                pair,
                resize_width=resize_width,
                max_lag_seconds=max_lag_seconds,
            )
            reference_duration = float(alignment_bundle["reference_duration_seconds"])
            full_segment_count = int(np.floor(reference_duration / segment_seconds))
            dropped_remainder = float(reference_duration - full_segment_count * segment_seconds)
            if full_segment_count <= 0:
                raise ValueError("Source RGB video has no full 30-second segments.")
            alignment_summary = _source_rgb_audio_alignment_summary(alignment_bundle["alignment"])

            plot_file = None
            if plot_output_folder is not None:
                plot_path = (
                    plot_output_folder
                    / str(pair["split_folder_name"])
                    / f"{pair['sample_name']}_{pair['side']}_rgb_audio_cross_correlation_activity_signal.png"
                )
                if overwrite or not plot_path.exists():
                    plot_file = _write_source_rgb_audio_alignment_plot(
                        plot_path,
                        pair,
                        alignment_bundle,
                    )
                    if verbose:
                        print(f"Plot written: {plot_file}")
                else:
                    plot_file = str(plot_path)
                    if verbose:
                        print(f"Plot exists: {plot_file}")

            for segment_index in range(full_segment_count):
                segment_name = f"Seg{segment_index + 1}"
                segment_start = segment_index * segment_seconds
                output_folder = (
                    aligned_dataset_folder
                    / str(pair["split_folder_name"])
                    / segment_name
                )
                output_file = (
                    output_folder
                    / f"{pair['sample_name']}_{pair['side']}_rgb_with_audio.mp4"
                )

                if output_file.exists() and not overwrite:
                    if verbose:
                        print(
                            "Reusing existing RGB-with-audio segment: "
                            f"{pair['split_folder_name']} {pair['sample_name']} {pair['side']} {segment_name} -> {output_file}"
                        )
                    exported.append(
                        {
                            **pair,
                            "segment": segment_name,
                            "segment_start_seconds": round(segment_start, 6),
                            "duration_seconds": round(segment_seconds, 6),
                            "output_file": str(output_file),
                            "plot_file": plot_file,
                            "status": "reused",
                            "reason": "Output already exists.",
                            "alignment": alignment_summary,
                        }
                    )
                    continue

                mux = _mux_rgb_video_segment_with_aligned_audio(
                    reference_file=Path(str(pair["reference_file"])),
                    audio_file=Path(str(pair["audio_file"])),
                    output_file=output_file,
                    segment_start_seconds=segment_start,
                    duration_seconds=segment_seconds,
                    audio_offset_seconds=float(alignment_bundle["alignment"]["offset_seconds"]),
                )
                if verbose:
                    print(
                        "Exported RGB-with-audio segment: "
                        f"{pair['split_folder_name']} {pair['sample_name']} {pair['side']} {segment_name} -> {output_file}"
                    )
                exported.append(
                    {
                        **pair,
                        "segment": segment_name,
                        "segment_start_seconds": round(segment_start, 6),
                        "duration_seconds": round(segment_seconds, 6),
                        "output_file": str(output_file),
                        "plot_file": plot_file,
                        "status": "exported",
                        "alignment": alignment_summary,
                        "mux": mux,
                    }
                )

            if dropped_remainder > 0:
                if verbose:
                    print(
                        "Dropped trailing remainder: "
                        f"{pair['split_folder_name']} {pair['sample_name']} {pair['side']} "
                        f"{dropped_remainder:.3f}s"
                    )
                skipped.append(
                    {
                        **pair,
                        "reason": "Dropped trailing remainder shorter than full segment duration.",
                        "dropped_remainder_seconds": round(dropped_remainder, 6),
                    }
                )
        except Exception as exc:
            skipped.append({**pair, "reason": str(exc)})

    summary = {
        "source_dataset_folder": str(source_dataset_folder),
        "aligned_dataset_folder": str(aligned_dataset_folder),
        "method": "source_rgb_audio_cross_correlation_fixed_offset_30s_segments",
        "segment_seconds": segment_seconds,
        "max_lag_seconds": max_lag_seconds,
        "output_naming": "*_rgb_with_audio.mp4",
        "plot_output_folder": str(plot_output_folder) if plot_output_folder else None,
        "total_discovered_count": len(all_discovered),
        "start_pair_group_index": start_pair_group_index,
        "max_pair_groups": max_pair_groups,
        "selected_pair_groups": selected_pair_groups,
        "overwrite": overwrite,
        "discovered_count": len(discovered),
        "exported_count": len(exported),
        "skipped_count": len(skipped),
        "exported": exported,
        "skipped": skipped,
    }
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    summary["summary_file"] = str(summary_output_path)
    return summary


def _mux_dtw_preview_with_aligned_audio(
    dtw_export: dict[str, Any],
    audio_result: dict[str, Any],
    output_folder: Path,
) -> dict[str, Any]:
    output_folder.mkdir(parents=True, exist_ok=True)
    sample_name = str(dtw_export.get("sample") or audio_result.get("sample") or "check_mailbox")
    side = str(dtw_export.get("side") or audio_result.get("side") or "day")
    output_path = output_folder / f"{sample_name}_{side}_rgb_event_dtw_with_aligned_audio.mp4"
    tmp_output_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    tmp_output_path.unlink(missing_ok=True)

    dtw_video_file = Path(str(dtw_export.get("output_file") or ""))
    audio_file = Path(str(audio_result.get("audio_file") or ""))
    audio_alignment = audio_result.get("alignment") if isinstance(audio_result.get("alignment"), dict) else {}
    audio_offset_seconds = audio_alignment.get("offset_seconds")
    if audio_offset_seconds is None:
        raise ValueError("RGB/AUDIO alignment must include offset_seconds for combined export.")
    audio_offset_seconds = float(audio_offset_seconds)
    audio_duration = float(audio_result.get("audio_duration_seconds") or 0.0)
    rgb_start_seconds = float(dtw_export.get("rgb_start_seconds") or 0.0)
    duration_seconds = float(dtw_export.get("duration_seconds") or 0.0)

    if not dtw_video_file.exists():
        raise FileNotFoundError(f"DTW preview video does not exist: {dtw_video_file}")
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio_file}")
    if duration_seconds <= 0:
        raise ValueError("Combined export requires a positive DTW preview duration.")
    if audio_duration <= 0:
        raise ValueError("Combined export requires a positive audio duration.")

    audio_time_at_preview_start = rgb_start_seconds + audio_offset_seconds
    audio_seek_seconds = max(0.0, audio_time_at_preview_start)
    audio_delay_seconds = max(0.0, -audio_time_at_preview_start)
    if audio_seek_seconds >= audio_duration:
        raise ValueError("Aligned audio seek starts beyond the end of the audio file.")

    audio_filter = "[1:a]apad[a]"
    if audio_delay_seconds > 0:
        delay_ms = int(round(audio_delay_seconds * 1000.0))
        audio_filter = f"[1:a]adelay={delay_ms}:all=1,apad[a]"

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(dtw_video_file),
    ]
    if audio_seek_seconds > 0:
        command.extend(["-ss", f"{audio_seek_seconds:.6f}"])
    command.extend(
        [
            "-i",
            str(audio_file),
            "-filter_complex",
            audio_filter,
            "-map",
            "0:v:0",
            "-map",
            "[a]",
            "-t",
            f"{duration_seconds:.6f}",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(tmp_output_path),
        ]
    )
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        tmp_output_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to mux DTW preview with aligned audio: {completed.stderr.strip()}")

    tmp_output_path.replace(output_path)
    return {
        "sample": sample_name,
        "side": side,
        "output_file": str(output_path),
        "dtw_video_file": str(dtw_video_file),
        "audio_file": str(audio_file),
        "duration_seconds": round(duration_seconds, 6),
        "rgb_start_seconds": round(rgb_start_seconds, 6),
        "audio_offset_seconds": round(audio_offset_seconds, 6),
        "audio_correlation": round(float(audio_alignment.get("peak_correlation") or 0.0), 6),
        "audio_shift": {
            "source_seek_seconds": round(audio_seek_seconds, 6),
            "output_delay_seconds": round(audio_delay_seconds, 6),
            "speed_warped": False,
        },
        "video_stream": "rgb_event_dtw_preview_copy",
        "audio_encoder": "aac",
    }


def _mux_silent_preview_with_aligned_audio(
    silent_video_file: Path,
    audio_result: dict[str, Any],
    output_path: Path,
    rgb_start_seconds: float,
    duration_seconds: float,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    tmp_output_path.unlink(missing_ok=True)

    audio_file = Path(str(audio_result.get("audio_file") or ""))
    audio_alignment = audio_result.get("alignment") if isinstance(audio_result.get("alignment"), dict) else {}
    audio_offset_seconds = audio_alignment.get("offset_seconds")
    if audio_offset_seconds is None:
        raise ValueError("RGB/AUDIO alignment must include offset_seconds for combined export.")
    audio_offset_seconds = float(audio_offset_seconds)
    audio_duration = float(audio_result.get("audio_duration_seconds") or 0.0)

    if not silent_video_file.exists():
        raise FileNotFoundError(f"Silent preview video does not exist: {silent_video_file}")
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio_file}")
    if duration_seconds <= 0:
        raise ValueError("Combined export requires a positive preview duration.")
    if audio_duration <= 0:
        raise ValueError("Combined export requires a positive audio duration.")

    audio_time_at_preview_start = rgb_start_seconds + audio_offset_seconds
    audio_seek_seconds = max(0.0, audio_time_at_preview_start)
    audio_delay_seconds = max(0.0, -audio_time_at_preview_start)
    if audio_seek_seconds >= audio_duration:
        raise ValueError("Aligned audio seek starts beyond the end of the audio file.")

    audio_filter = "[1:a]apad[a]"
    if audio_delay_seconds > 0:
        delay_ms = int(round(audio_delay_seconds * 1000.0))
        audio_filter = f"[1:a]adelay={delay_ms}:all=1,apad[a]"

    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(silent_video_file)]
    if audio_seek_seconds > 0:
        command.extend(["-ss", f"{audio_seek_seconds:.6f}"])
    command.extend(
        [
            "-i",
            str(audio_file),
            "-filter_complex",
            audio_filter,
            "-map",
            "0:v:0",
            "-map",
            "[a]",
            "-t",
            f"{duration_seconds:.6f}",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(tmp_output_path),
        ]
    )
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        tmp_output_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to mux preview with aligned audio: {completed.stderr.strip()}")

    tmp_output_path.replace(output_path)
    return {
        "output_file": str(output_path),
        "silent_video_file": str(silent_video_file),
        "audio_file": str(audio_file),
        "duration_seconds": round(duration_seconds, 6),
        "rgb_start_seconds": round(rgb_start_seconds, 6),
        "audio_offset_seconds": round(audio_offset_seconds, 6),
        "audio_correlation": round(float(audio_alignment.get("peak_correlation") or 0.0), 6),
        "audio_shift": {
            "source_seek_seconds": round(audio_seek_seconds, 6),
            "output_delay_seconds": round(audio_delay_seconds, 6),
            "speed_warped": False,
        },
        "video_stream": "rgb_event_ir_depth_preview_copy",
        "audio_encoder": "aac",
    }


def _export_rgb_event_ir_depth_dtw_audio_grid_preview(
    dtw_result: dict[str, Any],
    ir_alignment: dict[str, Any],
    depth_alignment: dict[str, Any],
    audio_result: dict[str, Any],
    output_folder: Path,
) -> dict[str, Any]:
    sample_name = str(dtw_result.get("sample") or "sample")
    side = str(dtw_result.get("side") or "day")
    silent_output_path = output_folder / f"{sample_name}_{side}_rgb_event_ir_depth_dtw_silent.mp4"
    final_output_path = output_folder / f"{sample_name}_{side}_rgb_event_ir_depth_dtw_with_aligned_audio.mp4"
    tmp_silent_path = silent_output_path.with_name(f"{silent_output_path.stem}.tmp{silent_output_path.suffix}")
    tmp_silent_path.unlink(missing_ok=True)

    reference_file = Path(str(dtw_result.get("reference_file", "")))
    reference_duration = float(dtw_result.get("reference_duration_seconds") or 0.0)
    dtw_alignment = dtw_result.get("alignment") if isinstance(dtw_result.get("alignment"), dict) else {}
    event_file = Path(str(dtw_alignment.get("file") or dtw_result.get("target_file") or ""))
    event_duration = float(dtw_alignment.get("duration_seconds") or dtw_result.get("target_duration_seconds") or 0.0)
    offset_curve = dtw_alignment.get("offset_curve")
    if not isinstance(offset_curve, list) or not offset_curve:
        raise ValueError("DTW alignment must include a non-empty offset_curve.")
    render_offset_curve = _smooth_dtw_offset_curve_for_export(offset_curve)

    ir_file = Path(str(ir_alignment.get("file") or ""))
    depth_file = Path(str(depth_alignment.get("file") or ""))
    ir_duration = float(ir_alignment.get("duration_seconds") or 0.0)
    depth_duration = float(depth_alignment.get("duration_seconds") or 0.0)
    ir_offset = ir_alignment.get("offset_seconds")
    depth_offset = depth_alignment.get("offset_seconds")
    if ir_offset is None:
        raise ValueError("IR alignment must include offset_seconds.")
    if depth_offset is None:
        raise ValueError("DEPTH alignment must include offset_seconds.")
    ir_offset = float(ir_offset)
    depth_offset = float(depth_offset)

    for label, file in (("RGB", reference_file), ("EVENT", event_file), ("IR", ir_file), ("DEPTH", depth_file)):
        if not file.exists():
            raise FileNotFoundError(f"{label} file does not exist: {file}")
    if reference_duration <= 0 or event_duration <= 0 or ir_duration <= 0 or depth_duration <= 0:
        raise ValueError("Combined visual export requires positive RGB/EVENT/IR/DEPTH durations.")

    reference_start = max(0.0, float(offset_curve[0]["reference_time_seconds"]), -ir_offset, -depth_offset)
    reference_end = min(
        reference_duration,
        float(offset_curve[-1]["reference_time_seconds"]),
        ir_duration - ir_offset,
        depth_duration - depth_offset,
    )
    duration = max(0.0, reference_end - reference_start)
    if duration <= 0:
        raise ValueError("No positive RGB/EVENT/IR/DEPTH overlap window is available.")

    rgb_cap = cv2.VideoCapture(str(reference_file))
    event_cap = cv2.VideoCapture(str(event_file))
    ir_cap = cv2.VideoCapture(str(ir_file))
    depth_cap = cv2.VideoCapture(str(depth_file))
    caps = [rgb_cap, event_cap, ir_cap, depth_cap]
    try:
        if not all(cap.isOpened() for cap in caps):
            raise ValueError("Could not open one or more RGB/EVENT/IR/DEPTH videos.")
        reference_fps = float(rgb_cap.get(cv2.CAP_PROP_FPS) or 0.0)
        event_fps = float(event_cap.get(cv2.CAP_PROP_FPS) or 0.0)
        ir_fps = float(ir_cap.get(cv2.CAP_PROP_FPS) or 0.0)
        depth_fps = float(depth_cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if reference_fps <= 0 or event_fps <= 0 or ir_fps <= 0 or depth_fps <= 0:
            raise ValueError("Could not determine FPS for one or more RGB/EVENT/IR/DEPTH videos.")

        output_folder.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(tmp_silent_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(DTW_EXPORT_PREVIEW_FPS),
            (EXPORT_PANEL_WIDTH * 2, EXPORT_PANEL_HEIGHT * 2),
        )
        if not writer.isOpened():
            tmp_silent_path.unlink(missing_ok=True)
            raise RuntimeError("Could not create OpenCV VideoWriter for 2x2 preview.")

        frame_count = max(1, int(np.floor(duration * DTW_EXPORT_PREVIEW_FPS)))
        written_count = 0
        rgb_state: dict[str, Any] = {}
        event_state: dict[str, Any] = {}
        ir_state: dict[str, Any] = {}
        depth_state: dict[str, Any] = {}
        try:
            for frame_index in range(frame_count):
                rgb_time = reference_start + frame_index / float(DTW_EXPORT_PREVIEW_FPS)
                event_time = min(event_duration, max(0.0, _event_time_from_offset_curve(render_offset_curve, rgb_time)))
                ir_time = min(ir_duration, max(0.0, rgb_time + ir_offset))
                depth_time = min(depth_duration, max(0.0, rgb_time + depth_offset))

                rgb_frame_index = int(round(rgb_time * reference_fps))
                event_position = event_time * event_fps
                event_frame_index = int(np.floor(event_position))
                event_blend_weight = float(event_position - event_frame_index)
                ir_frame_index = int(round(ir_time * ir_fps))
                depth_frame_index = int(round(depth_time * depth_fps))

                rgb_panel = _prepare_preview_panel(_read_video_frame_forward(rgb_cap, rgb_frame_index, rgb_state))
                event_a, event_b = _read_video_frame_pair_forward(event_cap, event_frame_index, event_state)
                event_panel = _prepare_preview_panel(_blend_frames(event_a, event_b, event_blend_weight))
                ir_panel = _prepare_preview_panel(_read_video_frame_forward(ir_cap, ir_frame_index, ir_state))
                depth_panel = _prepare_preview_panel(_read_video_frame_forward(depth_cap, depth_frame_index, depth_state))

                _draw_label(rgb_panel, ["RGB", f"t={rgb_time:.2f}s"])
                _draw_label(event_panel, ["EVENT DTW", f"t={event_time:.2f}s", f"offset={event_time - rgb_time:+.3f}s"])
                _draw_label(ir_panel, ["IR xcorr", f"t={ir_time:.2f}s", f"offset={ir_offset:+.3f}s"])
                _draw_label(depth_panel, ["DEPTH xcorr", f"t={depth_time:.2f}s", f"offset={depth_offset:+.3f}s"])
                writer.write(np.vstack([np.hstack([rgb_panel, event_panel]), np.hstack([ir_panel, depth_panel])]))
                written_count += 1
        finally:
            writer.release()

        if written_count == 0:
            tmp_silent_path.unlink(missing_ok=True)
            raise RuntimeError("2x2 preview export wrote zero frames.")
        tmp_silent_path.replace(silent_output_path)
    finally:
        for cap in caps:
            cap.release()

    silent_removed = False
    silent_remove_warning = None
    try:
        mux_item = _mux_silent_preview_with_aligned_audio(
            silent_video_file=silent_output_path,
            audio_result=audio_result,
            output_path=final_output_path,
            rgb_start_seconds=reference_start,
            duration_seconds=duration,
        )
    finally:
        try:
            silent_output_path.unlink(missing_ok=True)
            silent_removed = True
        except Exception as exc:
            silent_remove_warning = f"Could not remove silent intermediate preview: {exc}"
    mux_item["silent_intermediate_removed"] = silent_removed
    if silent_remove_warning is not None:
        mux_item.setdefault("warnings", []).append(silent_remove_warning)

    mux_item.update(
        {
            "sample": sample_name,
            "side": side,
            "reference_file": str(reference_file),
            "event_file": str(event_file),
            "ir_file": str(ir_file),
            "depth_file": str(depth_file),
            "rgb_start_seconds": round(reference_start, 6),
            "rgb_end_seconds": round(reference_end, 6),
            "duration_seconds": round(duration, 6),
            "preview_fps": DTW_EXPORT_PREVIEW_FPS,
            "frames_written": written_count,
            "layout": "2x2",
            "panels": {
                "top_left": "rgb",
                "top_right": "event_dtw",
                "bottom_left": "ir_cross_correlation",
                "bottom_right": "depth_cross_correlation",
            },
            "event_render_curve": {
                "source": "smoothed_offset_curve",
                "smoothing_seconds": DTW_EXPORT_SMOOTHING_SECONDS,
                "points": _sample_offset_curve(render_offset_curve),
                "full_count": len(render_offset_curve),
                "monotonic_event_time": True,
                "event_frame_interpolation": "linear_blend",
            },
            "ir_offset_seconds": round(ir_offset, 6),
            "depth_offset_seconds": round(depth_offset, 6),
        }
    )
    return mux_item


def _run_and_export_rgb_event_dtw_with_audio_alignment(
    sample_name: str,
    split_folder_name: str,
    side: str,
    dataset_folder: Path | str,
    dtw_output_path: Path | str,
    audio_output_path: Path | str,
    combined_summary_path: Path | str,
    plot_output_folder: Path | str | None,
    output_folder: Path | str,
    window_seconds: float,
    resize_width: int,
    reference_file: Path | str | None = None,
    event_file: Path | str | None = None,
    ir_file: Path | str | None = None,
    depth_file: Path | str | None = None,
    audio_file: Path | str | None = None,
    keep_intermediate_dtw_video: bool = True,
) -> dict[str, Any]:
    dataset_folder = Path(dataset_folder)
    dtw_output_path = Path(dtw_output_path)
    audio_output_path = Path(audio_output_path)
    combined_summary_path = Path(combined_summary_path)
    plot_output_folder = Path(plot_output_folder) if plot_output_folder is not None else None
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    side = side.lower()

    reference_file = Path(reference_file) if reference_file is not None else dataset_folder / split_folder_name / f"{sample_name}_{side}_rgb.mp4"
    event_file = Path(event_file) if event_file is not None else dataset_folder / split_folder_name / f"{sample_name}_{side}_event.mp4"
    ir_file = Path(ir_file) if ir_file is not None else dataset_folder / split_folder_name / f"{sample_name}_{side}_ir.mp4"
    depth_file = Path(depth_file) if depth_file is not None else dataset_folder / split_folder_name / f"{sample_name}_{side}_depth.mp4"
    audio_file = Path(audio_file) if audio_file is not None else dataset_folder / split_folder_name / f"{sample_name}_{side}.m4a"

    summary: dict[str, Any] = {
        "sample": sample_name,
        "side": side,
        "method": "rgb_event_dtw_ir_depth_cross_correlation_visual_with_rgb_audio_cross_correlation_audio",
        "dtw_alignment_file": str(dtw_output_path),
        "audio_alignment_file": str(audio_output_path),
        "source_rgb_file": str(reference_file),
        "source_event_file": str(event_file),
        "source_ir_file": str(ir_file),
        "source_depth_file": str(depth_file),
        "source_audio_file": str(audio_file),
        "ignored_files": [str(reference_file.with_name(f"{sample_name}_{side}_rgb_with_audio.mp4"))],
        "exported_count": 0,
        "skipped_count": 0,
        "exported": [],
        "skipped": [],
    }

    try:
        dtw_result = _run_rgb_event_dtw_alignment(
            sample_name=sample_name,
            split_folder_name=split_folder_name,
            side=side,
            dataset_folder=dataset_folder,
            output_path=dtw_output_path,
            plot_output_folder=plot_output_folder,
            window_seconds=window_seconds,
            resize_width=resize_width,
            reference_file=reference_file,
            event_file=event_file,
        )
        dtw_result["export"] = {
            "exported_count": 0,
            "skipped_count": 0,
            "exported": [],
            "skipped": [],
            "note": "EVENT DTW is rendered directly inside the final RGB/EVENT/IR/DEPTH video with aligned audio.",
        }
        with open(dtw_output_path, "w", encoding="utf-8") as handle:
            json.dump(dtw_result, handle, indent=2, ensure_ascii=False)

        dtw_alignment = dtw_result.get("alignment") if isinstance(dtw_result.get("alignment"), dict) else {}
        summary["dtw"] = {
            "offset_seconds": dtw_alignment.get("offset_seconds"),
            "start_offset_seconds": dtw_alignment.get("start_offset_seconds"),
            "end_offset_seconds": dtw_alignment.get("end_offset_seconds"),
            "offset_drift_seconds": dtw_alignment.get("offset_drift_seconds"),
            "dtw_path_length": dtw_alignment.get("dtw_path_length"),
            "plot_file": dtw_result.get("plot_file"),
            "export": dtw_result.get("export"),
        }

        reference_meta = _video_metadata(reference_file)
        if not reference_meta.get("opened"):
            raise ValueError(f"Could not open RGB reference video: {reference_file}")
        reference_trace = _motion_energy_trace(reference_file, resize_width=resize_width)
        ir_trace = _motion_energy_trace(ir_file, resize_width=resize_width)
        depth_trace = _motion_energy_trace(depth_file, resize_width=resize_width)
        ir_plot_path = (
            plot_output_folder / f"{sample_name}_{side}_rgb_ir_cross_correlation_activity_signal.png"
            if plot_output_folder is not None
            else None
        )
        depth_plot_path = (
            plot_output_folder / f"{sample_name}_{side}_rgb_depth_cross_correlation_activity_signal.png"
            if plot_output_folder is not None
            else None
        )
        ir_alignment = _estimate_modality_alignment(
            reference_file=reference_file,
            reference_meta=reference_meta,
            reference_trace=reference_trace,
            target_modality="ir",
            target_file=ir_file,
            target_trace=ir_trace,
            max_lag_seconds=LOW_CONFIDENCE_LARGE_VISUAL_OFFSET_SECONDS,
            plot_output_path=ir_plot_path,
        )
        depth_alignment = _estimate_modality_alignment(
            reference_file=reference_file,
            reference_meta=reference_meta,
            reference_trace=reference_trace,
            target_modality="depth",
            target_file=depth_file,
            target_trace=depth_trace,
            max_lag_seconds=LOW_CONFIDENCE_LARGE_VISUAL_OFFSET_SECONDS,
            plot_output_path=depth_plot_path,
        )
        summary["ir"] = {
            "offset_seconds": ir_alignment.get("offset_seconds"),
            "peak_correlation": ir_alignment.get("peak_correlation"),
            "confidence_label": ir_alignment.get("confidence_label"),
            "overlap_ratio": ir_alignment.get("overlap_ratio"),
            "plot_file": ir_alignment.get("activity_plot_file"),
            "warnings": ir_alignment.get("warnings", []),
        }
        summary["depth"] = {
            "offset_seconds": depth_alignment.get("offset_seconds"),
            "peak_correlation": depth_alignment.get("peak_correlation"),
            "confidence_label": depth_alignment.get("confidence_label"),
            "overlap_ratio": depth_alignment.get("overlap_ratio"),
            "plot_file": depth_alignment.get("activity_plot_file"),
            "warnings": depth_alignment.get("warnings", []),
        }
        if ir_alignment.get("offset_seconds") is None:
            raise ValueError("RGB/IR alignment did not produce an offset.")
        if depth_alignment.get("offset_seconds") is None:
            raise ValueError("RGB/DEPTH alignment did not produce an offset.")

        audio_result = _run_rgb_audio_cross_correlation_alignment(
            sample_name=sample_name,
            split_folder_name=split_folder_name,
            side=side,
            dataset_folder=dataset_folder,
            output_path=audio_output_path,
            plot_output_folder=plot_output_folder,
            output_folder=output_folder,
            resize_width=resize_width,
            prefer_gpu=False,
            reference_file=reference_file,
            audio_file=audio_file,
            export_preview=False,
            summary_file_name=f"{sample_name}_{side}_rgb_audio_cross_correlation_export_summary.json",
        )
        audio_alignment = audio_result.get("alignment") if isinstance(audio_result.get("alignment"), dict) else {}
        summary["audio"] = {
            "offset_seconds": audio_alignment.get("offset_seconds"),
            "peak_correlation": audio_alignment.get("peak_correlation"),
            "confidence_label": audio_alignment.get("confidence_label"),
            "overlap_ratio": audio_alignment.get("overlap_ratio"),
            "plot_file": audio_result.get("plot_file"),
            "export": audio_result.get("export"),
        }
        if audio_alignment.get("offset_seconds") is None:
            raise ValueError("RGB/AUDIO alignment did not produce an offset.")

        combined_export = _export_rgb_event_ir_depth_dtw_audio_grid_preview(
            dtw_result=dtw_result,
            ir_alignment=ir_alignment,
            depth_alignment=depth_alignment,
            audio_result=audio_result,
            output_folder=output_folder,
        )
        combined_export["dtw_alignment_file"] = str(dtw_output_path)
        combined_export["audio_alignment_file"] = str(audio_output_path)
        combined_export["ir_alignment"] = summary["ir"]
        combined_export["depth_alignment"] = summary["depth"]
        summary["exported"].append(combined_export)
    except Exception as exc:
        summary["skipped"].append(
            {
                "sample": sample_name,
                "side": side,
                "reference_file": str(reference_file),
                "event_file": str(event_file),
                "ir_file": str(ir_file),
                "depth_file": str(depth_file),
                "audio_file": str(audio_file),
                "reason": str(exc),
            }
        )

    summary["exported_count"] = len(summary["exported"])
    summary["skipped_count"] = len(summary["skipped"])
    combined_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(combined_summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    summary["summary_file"] = str(combined_summary_path)
    return summary


def run_and_export_check_mailbox_day_rgb_event_dtw_with_audio_alignment(
    dataset_folder: Path | str = "dataset",
    dtw_output_path: Path | str = DEFAULT_DTW_CHECK_MAILBOX_OUTPUT_PATH,
    audio_output_path: Path | str = DEFAULT_RGB_AUDIO_CHECK_MAILBOX_OUTPUT_PATH,
    plot_output_folder: Path | str | None = DEFAULT_PLOT_OUTPUT_FOLDER,
    output_folder: Path | str = DEFAULT_EXPORT_OUTPUT_FOLDER,
    window_seconds: float = 10.0,
    resize_width: int = 160,
) -> dict[str, Any]:
    """Export the check_mailbox day 2x2 RGB/EVENT/IR/DEPTH preview with aligned separate audio."""
    return _run_and_export_rgb_event_dtw_with_audio_alignment(
        sample_name="check_mailbox",
        split_folder_name="check_mailbox_split",
        side="day",
        dataset_folder=dataset_folder,
        dtw_output_path=dtw_output_path,
        audio_output_path=audio_output_path,
        combined_summary_path=(
            Path(output_folder) / "check_mailbox_day_rgb_event_ir_depth_dtw_with_aligned_audio_summary.json"
        ),
        plot_output_folder=plot_output_folder,
        output_folder=output_folder,
        window_seconds=window_seconds,
        resize_width=resize_width,
        keep_intermediate_dtw_video=False,
    )


def _bitrate_arg(bit_rate: int | None, fallback: str) -> str:
    if bit_rate is None or bit_rate <= 0:
        return fallback
    return str(int(bit_rate))


def _run_ffmpeg_exact(command: list[str], output_path: Path, description: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    tmp_output_path.unlink(missing_ok=True)
    completed = subprocess.run([*command, str(tmp_output_path)], check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        tmp_output_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to export {description}: {completed.stderr.strip()}")
    tmp_output_path.replace(output_path)


def _export_aligned_source_video_segment(
    source_file: Path,
    output_path: Path,
    source_start_seconds: float,
    duration_seconds: float,
    encoding_info: dict[str, Any],
    description: str,
) -> dict[str, Any]:
    bit_rate = _bitrate_arg(encoding_info.get("bit_rate"), "12000k")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{source_start_seconds:.6f}",
        "-t",
        f"{duration_seconds:.6f}",
        "-i",
        str(source_file),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-b:v",
        bit_rate,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    _run_ffmpeg_exact(command, output_path, description)
    return {
        "output_file": str(output_path),
        "source_file": str(source_file),
        "source_start_seconds": round(source_start_seconds, 6),
        "duration_seconds": round(duration_seconds, 6),
        "bit_rate": encoding_info.get("bit_rate"),
        "fps": encoding_info.get("fps"),
        "width": encoding_info.get("width"),
        "height": encoding_info.get("height"),
    }


def _export_aligned_audio_segment(
    audio_file: Path,
    output_path: Path,
    audio_start_seconds: float,
    duration_seconds: float,
    encoding_info: dict[str, Any],
) -> dict[str, Any]:
    bit_rate = _bitrate_arg(encoding_info.get("bit_rate"), "128k")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{audio_start_seconds:.6f}",
        "-t",
        f"{duration_seconds:.6f}",
        "-i",
        str(audio_file),
        "-vn",
        "-c:a",
        "aac",
        "-b:a",
        bit_rate,
        "-movflags",
        "+faststart",
    ]
    _run_ffmpeg_exact(command, output_path, "aligned audio segment")
    return {
        "output_file": str(output_path),
        "source_file": str(audio_file),
        "source_start_seconds": round(audio_start_seconds, 6),
        "duration_seconds": round(duration_seconds, 6),
        "bit_rate": encoding_info.get("bit_rate"),
        "sample_rate": encoding_info.get("sample_rate"),
        "channels": encoding_info.get("channels"),
    }


def _export_event_dtw_segment(
    dtw_result: dict[str, Any],
    output_path: Path,
    rgb_start_seconds: float,
    duration_seconds: float,
    reference_fps: float,
    event_encoding_info: dict[str, Any],
) -> dict[str, Any]:
    dtw_alignment = dtw_result.get("alignment") if isinstance(dtw_result.get("alignment"), dict) else {}
    event_file = Path(str(dtw_alignment.get("file") or dtw_result.get("target_file") or ""))
    event_duration = float(dtw_alignment.get("duration_seconds") or dtw_result.get("target_duration_seconds") or 0.0)
    offset_curve = dtw_alignment.get("offset_curve")
    if not isinstance(offset_curve, list) or not offset_curve:
        raise ValueError("DTW alignment must include a non-empty offset_curve.")
    if reference_fps <= 0:
        raise ValueError("EVENT segment export requires a positive RGB reference FPS.")

    render_offset_curve = _smooth_dtw_offset_curve_for_export(offset_curve)
    event_cap = cv2.VideoCapture(str(event_file))
    if not event_cap.isOpened():
        raise ValueError(f"Could not open EVENT video: {event_file}")
    event_fps = float(event_cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(event_encoding_info.get("width") or event_cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(event_encoding_info.get("height") or event_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if event_fps <= 0 or width <= 0 or height <= 0:
        event_cap.release()
        raise ValueError("Could not determine EVENT FPS or frame size.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output_path = output_path.with_name(f"{output_path.stem}.rawtmp{output_path.suffix}")
    raw_output_path.unlink(missing_ok=True)
    writer = cv2.VideoWriter(
        str(raw_output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(reference_fps),
        (width, height),
    )
    if not writer.isOpened():
        event_cap.release()
        raw_output_path.unlink(missing_ok=True)
        raise RuntimeError("Could not create OpenCV VideoWriter for EVENT DTW segment.")

    frame_count = max(1, int(round(duration_seconds * reference_fps)))
    event_state: dict[str, Any] = {}
    written_count = 0
    try:
        for frame_index in range(frame_count):
            rgb_time = rgb_start_seconds + frame_index / reference_fps
            event_time = min(event_duration, max(0.0, _event_time_from_offset_curve(render_offset_curve, rgb_time)))
            event_position = event_time * event_fps
            event_frame_index = int(np.floor(event_position))
            blend_weight = float(event_position - event_frame_index)
            event_a, event_b = _read_video_frame_pair_forward(event_cap, event_frame_index, event_state)
            frame = _blend_frames(event_a, event_b, blend_weight)
            if frame is None:
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            elif frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
            written_count += 1
    finally:
        writer.release()
        event_cap.release()

    if written_count == 0:
        raw_output_path.unlink(missing_ok=True)
        raise RuntimeError("EVENT DTW segment export wrote zero frames.")

    bit_rate = _bitrate_arg(event_encoding_info.get("bit_rate"), "56000k")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(raw_output_path),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-b:v",
        bit_rate,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    try:
        _run_ffmpeg_exact(command, output_path, "EVENT DTW segment")
    finally:
        raw_output_path.unlink(missing_ok=True)
    return {
        "output_file": str(output_path),
        "source_file": str(event_file),
        "rgb_start_seconds": round(rgb_start_seconds, 6),
        "duration_seconds": round(duration_seconds, 6),
        "frames_written": written_count,
        "fps": reference_fps,
        "source_fps": event_fps,
        "bit_rate": event_encoding_info.get("bit_rate"),
        "width": width,
        "height": height,
        "dtw_warped": True,
    }


def _build_aligned_dataset_side_alignment(
    sample_name: str,
    split_folder_name: str,
    side: str,
    dataset_folder: Path,
    alignment_output_folder: Path,
    plot_output_folder: Path | None,
    resize_width: int,
    window_seconds: float,
) -> dict[str, Any]:
    split_folder = dataset_folder / split_folder_name
    reference_file = split_folder / f"{sample_name}_{side}_rgb.mp4"
    event_file = split_folder / f"{sample_name}_{side}_event.mp4"
    ir_file = split_folder / f"{sample_name}_{side}_ir.mp4"
    depth_file = split_folder / f"{sample_name}_{side}_depth.mp4"
    audio_file = split_folder / f"{sample_name}_{side}.m4a"
    required_files = [reference_file, event_file, ir_file, depth_file, audio_file]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required {sample_name} {side} files: " + ", ".join(missing))

    dtw_path = _dtw_alignment_output_path(alignment_output_folder, sample_name, side)
    audio_path = _audio_alignment_output_path(alignment_output_folder, sample_name, side)
    dtw_result = _run_rgb_event_dtw_alignment(
        sample_name=sample_name,
        split_folder_name=split_folder_name,
        side=side,
        dataset_folder=dataset_folder,
        output_path=dtw_path,
        plot_output_folder=plot_output_folder,
        window_seconds=window_seconds,
        resize_width=resize_width,
        reference_file=reference_file,
        event_file=event_file,
    )

    reference_meta = _video_metadata(reference_file)
    if not reference_meta.get("opened"):
        raise ValueError(f"Could not open RGB reference video: {reference_file}")
    reference_trace = _motion_energy_trace(reference_file, resize_width=resize_width)
    ir_trace = _motion_energy_trace(ir_file, resize_width=resize_width)
    depth_trace = _motion_energy_trace(depth_file, resize_width=resize_width)
    ir_alignment = _estimate_modality_alignment(
        reference_file=reference_file,
        reference_meta=reference_meta,
        reference_trace=reference_trace,
        target_modality="ir",
        target_file=ir_file,
        target_trace=ir_trace,
        max_lag_seconds=LOW_CONFIDENCE_LARGE_VISUAL_OFFSET_SECONDS,
        plot_output_path=plot_output_folder / f"{sample_name}_{side}_rgb_ir_segment_export_activity_signal.png"
        if plot_output_folder is not None
        else None,
    )
    depth_alignment = _estimate_modality_alignment(
        reference_file=reference_file,
        reference_meta=reference_meta,
        reference_trace=reference_trace,
        target_modality="depth",
        target_file=depth_file,
        target_trace=depth_trace,
        max_lag_seconds=LOW_CONFIDENCE_LARGE_VISUAL_OFFSET_SECONDS,
        plot_output_path=plot_output_folder / f"{sample_name}_{side}_rgb_depth_segment_export_activity_signal.png"
        if plot_output_folder is not None
        else None,
    )
    if ir_alignment.get("offset_seconds") is None:
        raise ValueError(f"RGB/IR alignment did not produce an offset for {sample_name} {side}.")
    if depth_alignment.get("offset_seconds") is None:
        raise ValueError(f"RGB/DEPTH alignment did not produce an offset for {sample_name} {side}.")

    audio_result = _run_rgb_audio_cross_correlation_alignment(
        sample_name=sample_name,
        split_folder_name=split_folder_name,
        side=side,
        dataset_folder=dataset_folder,
        output_path=audio_path,
        plot_output_folder=plot_output_folder,
        output_folder=alignment_output_folder,
        resize_width=resize_width,
        prefer_gpu=False,
        reference_file=reference_file,
        audio_file=audio_file,
        export_preview=False,
        summary_file_name=f"{sample_name}_{side}_rgb_audio_cross_correlation_export_summary.json",
    )
    audio_alignment = audio_result.get("alignment") if isinstance(audio_result.get("alignment"), dict) else {}
    if audio_alignment.get("offset_seconds") is None:
        raise ValueError(f"RGB/AUDIO alignment did not produce an offset for {sample_name} {side}.")

    dtw_alignment = dtw_result.get("alignment") if isinstance(dtw_result.get("alignment"), dict) else {}
    offset_curve = dtw_alignment.get("offset_curve")
    if not isinstance(offset_curve, list) or not offset_curve:
        raise ValueError(f"EVENT DTW alignment did not produce an offset curve for {sample_name} {side}.")
    ir_offset = float(ir_alignment["offset_seconds"])
    depth_offset = float(depth_alignment["offset_seconds"])
    audio_offset = float(audio_alignment["offset_seconds"])
    reference_duration = float(dtw_result.get("reference_duration_seconds") or reference_meta.get("duration_seconds") or 0.0)
    event_duration = float(dtw_alignment.get("duration_seconds") or dtw_result.get("target_duration_seconds") or 0.0)
    ir_duration = float(ir_alignment.get("duration_seconds") or 0.0)
    depth_duration = float(depth_alignment.get("duration_seconds") or 0.0)
    audio_duration = float(audio_result.get("audio_duration_seconds") or 0.0)
    overlap_start = max(0.0, float(offset_curve[0]["reference_time_seconds"]), -ir_offset, -depth_offset, -audio_offset)
    overlap_end = min(
        reference_duration,
        float(offset_curve[-1]["reference_time_seconds"]),
        ir_duration - ir_offset,
        depth_duration - depth_offset,
        audio_duration - audio_offset,
    )
    overlap_duration = max(0.0, overlap_end - overlap_start)
    if overlap_duration < 30.0:
        raise ValueError(f"{sample_name} {side} has less than one full 30s aligned overlap segment.")

    return {
        "sample": sample_name,
        "side": side,
        "split_folder_name": split_folder_name,
        "files": {
            "rgb": reference_file,
            "event": event_file,
            "ir": ir_file,
            "depth": depth_file,
            "audio": audio_file,
        },
        "encoding": {
            "rgb": _source_video_encoding_info(reference_file),
            "event": _source_video_encoding_info(event_file),
            "ir": _source_video_encoding_info(ir_file),
            "depth": _source_video_encoding_info(depth_file),
            "audio": _source_audio_encoding_info(audio_file),
        },
        "reference_meta": reference_meta,
        "dtw_result": dtw_result,
        "dtw_alignment_file": str(dtw_path),
        "audio_result": audio_result,
        "audio_alignment_file": str(audio_path),
        "ir_alignment": ir_alignment,
        "depth_alignment": depth_alignment,
        "overlap_start_seconds": overlap_start,
        "overlap_end_seconds": overlap_end,
        "overlap_duration_seconds": overlap_duration,
        "segment_count": int(np.floor(overlap_duration / 30.0)),
        "dropped_remainder_seconds": float(overlap_duration - np.floor(overlap_duration / 30.0) * 30.0),
    }


def _export_aligned_dataset_split_segments(
    sample_name: str,
    split_folder_name: str,
    side_results: dict[str, dict[str, Any]],
    output_folder: Path,
    segment_seconds: float,
    summary_file_name: str = "aligned_segments_summary.json",
) -> dict[str, Any]:
    split_output_folder = output_folder / split_folder_name
    split_output_folder.mkdir(parents=True, exist_ok=True)
    skipped: list[dict[str, Any]] = []
    max_segment_count = max((item["segment_count"] for item in side_results.values()), default=0)
    exported_segments: list[dict[str, Any]] = []
    for segment_index in range(max_segment_count):
        segment_folder = split_output_folder / f"Seg{segment_index + 1}"
        segment_folder.mkdir(parents=True, exist_ok=True)
        segment_record: dict[str, Any] = {
            "segment": f"Seg{segment_index + 1}",
            "folder": str(segment_folder),
            "duration_seconds": segment_seconds,
            "sides": {},
        }
        for side, bundle in side_results.items():
            if segment_index >= int(bundle["segment_count"]):
                continue
            segment_start = float(bundle["overlap_start_seconds"]) + segment_index * segment_seconds
            files: dict[str, Path] = bundle["files"]
            encoding: dict[str, dict[str, Any]] = bundle["encoding"]
            ir_offset = float(bundle["ir_alignment"]["offset_seconds"])
            depth_offset = float(bundle["depth_alignment"]["offset_seconds"])
            audio_alignment = bundle["audio_result"].get("alignment", {})
            audio_offset = float(audio_alignment["offset_seconds"])
            reference_fps = float(bundle["reference_meta"]["fps"])
            side_record: dict[str, Any] = {
                "rgb_start_seconds": round(segment_start, 6),
                "rgb_end_seconds": round(segment_start + segment_seconds, 6),
                "outputs": {},
            }
            try:
                side_record["outputs"]["rgb"] = _export_aligned_source_video_segment(
                    source_file=files["rgb"],
                    output_path=segment_folder / f"{sample_name}_{side}_rgb.mp4",
                    source_start_seconds=segment_start,
                    duration_seconds=segment_seconds,
                    encoding_info=encoding["rgb"],
                    description=f"{sample_name} {side} RGB segment",
                )
                side_record["outputs"]["event"] = _export_event_dtw_segment(
                    dtw_result=bundle["dtw_result"],
                    output_path=segment_folder / f"{sample_name}_{side}_event.mp4",
                    rgb_start_seconds=segment_start,
                    duration_seconds=segment_seconds,
                    reference_fps=reference_fps,
                    event_encoding_info=encoding["event"],
                )
                side_record["outputs"]["ir"] = _export_aligned_source_video_segment(
                    source_file=files["ir"],
                    output_path=segment_folder / f"{sample_name}_{side}_ir.mp4",
                    source_start_seconds=segment_start + ir_offset,
                    duration_seconds=segment_seconds,
                    encoding_info=encoding["ir"],
                    description=f"{sample_name} {side} IR segment",
                )
                side_record["outputs"]["depth"] = _export_aligned_source_video_segment(
                    source_file=files["depth"],
                    output_path=segment_folder / f"{sample_name}_{side}_depth.mp4",
                    source_start_seconds=segment_start + depth_offset,
                    duration_seconds=segment_seconds,
                    encoding_info=encoding["depth"],
                    description=f"{sample_name} {side} DEPTH segment",
                )
                side_record["outputs"]["audio"] = _export_aligned_audio_segment(
                    audio_file=files["audio"],
                    output_path=segment_folder / f"{sample_name}_{side}.m4a",
                    audio_start_seconds=segment_start + audio_offset,
                    duration_seconds=segment_seconds,
                    encoding_info=encoding["audio"],
                )
                segment_record["sides"][side] = side_record
            except Exception as exc:
                skipped.append(
                    {
                        "sample": sample_name,
                        "side": side,
                        "segment": f"Seg{segment_index + 1}",
                        "stage": "export",
                        "reason": str(exc),
                    }
                )
        if segment_record["sides"]:
            exported_segments.append(segment_record)

    summary = {
        "sample": sample_name,
        "split_folder_name": split_folder_name,
        "output_folder": str(split_output_folder),
        "segment_seconds": segment_seconds,
        "exported_segment_count": len(exported_segments),
        "skipped_count": len(skipped),
        "sides": {
            side: {
                "segment_count": item["segment_count"],
                "overlap_start_seconds": round(float(item["overlap_start_seconds"]), 6),
                "overlap_end_seconds": round(float(item["overlap_end_seconds"]), 6),
                "overlap_duration_seconds": round(float(item["overlap_duration_seconds"]), 6),
                "dropped_remainder_seconds": round(float(item["dropped_remainder_seconds"]), 6),
                "dtw_alignment_file": item["dtw_alignment_file"],
                "audio_alignment_file": item["audio_alignment_file"],
                "source_files": {key: str(value) for key, value in item["files"].items()},
                "source_encoding": item["encoding"],
                "dtw": {
                    "start_offset_seconds": item["dtw_result"].get("alignment", {}).get("start_offset_seconds"),
                    "end_offset_seconds": item["dtw_result"].get("alignment", {}).get("end_offset_seconds"),
                    "offset_drift_seconds": item["dtw_result"].get("alignment", {}).get("offset_drift_seconds"),
                },
                "ir": {
                    "offset_seconds": item["ir_alignment"].get("offset_seconds"),
                    "peak_correlation": item["ir_alignment"].get("peak_correlation"),
                    "confidence_label": item["ir_alignment"].get("confidence_label"),
                },
                "depth": {
                    "offset_seconds": item["depth_alignment"].get("offset_seconds"),
                    "peak_correlation": item["depth_alignment"].get("peak_correlation"),
                    "confidence_label": item["depth_alignment"].get("confidence_label"),
                },
                "audio": {
                    "offset_seconds": item["audio_result"].get("alignment", {}).get("offset_seconds"),
                    "peak_correlation": item["audio_result"].get("alignment", {}).get("peak_correlation"),
                    "confidence_label": item["audio_result"].get("alignment", {}).get("confidence_label"),
                },
            }
            for side, item in side_results.items()
        },
        "segments": exported_segments,
        "skipped": skipped,
    }
    summary_path = split_output_folder / summary_file_name
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    summary["summary_file"] = str(summary_path)
    return summary


def run_and_export_cut_carrot_aligned_dataset_segments(
    dataset_folder: Path | str = "dataset",
    output_folder: Path | str = DEFAULT_ALIGNED_DATASET_FOLDER,
    alignment_output_folder: Path | str = ".",
    plot_output_folder: Path | str | None = DEFAULT_PLOT_OUTPUT_FOLDER,
    segment_seconds: float = 30.0,
    resize_width: int = 160,
    window_seconds: float = 10.0,
) -> dict[str, Any]:
    """Export cut_carrot day/night aligned modalities as separated 30-second dataset segments."""
    dataset_folder = Path(dataset_folder)
    output_folder = Path(output_folder)
    alignment_output_folder = Path(alignment_output_folder)
    plot_output_folder = Path(plot_output_folder) if plot_output_folder is not None else None
    output_folder.mkdir(parents=True, exist_ok=True)
    alignment_output_folder.mkdir(parents=True, exist_ok=True)

    side_results: dict[str, dict[str, Any]] = {}
    skipped: list[dict[str, Any]] = []
    for side in ("day", "night"):
        try:
            side_results[side] = _build_aligned_dataset_side_alignment(
                sample_name="cut_carrot",
                split_folder_name="cut_carrot_split",
                side=side,
                dataset_folder=dataset_folder,
                alignment_output_folder=alignment_output_folder,
                plot_output_folder=plot_output_folder,
                resize_width=resize_width,
                window_seconds=window_seconds,
            )
        except Exception as exc:
            skipped.append({"sample": "cut_carrot", "side": side, "stage": "alignment", "reason": str(exc)})

    summary = _export_aligned_dataset_split_segments(
        sample_name="cut_carrot",
        split_folder_name="cut_carrot_split",
        side_results=side_results,
        output_folder=output_folder,
        segment_seconds=segment_seconds,
    )
    summary["skipped"] = [*skipped, *summary.get("skipped", [])]
    summary["skipped_count"] = len(summary["skipped"])
    summary_path = Path(summary["summary_file"])
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump({key: value for key, value in summary.items() if key != "summary_file"}, handle, indent=2, ensure_ascii=False)
    return summary


def run_and_export_all_aligned_dataset_segments(
    dataset_folder: Path | str = "dataset",
    output_folder: Path | str = DEFAULT_ALIGNED_DATASET_FOLDER,
    alignment_output_folder: Path | str = ".",
    plot_output_folder: Path | str | None = DEFAULT_PLOT_OUTPUT_FOLDER,
    segment_seconds: float = 30.0,
    resize_width: int = 160,
    window_seconds: float = 10.0,
) -> dict[str, Any]:
    """Export every complete dataset sample as separated aligned 30-second segments."""
    dataset_folder = Path(dataset_folder)
    output_folder = Path(output_folder)
    alignment_output_folder = Path(alignment_output_folder)
    plot_output_folder = Path(plot_output_folder) if plot_output_folder is not None else None
    output_folder.mkdir(parents=True, exist_ok=True)
    alignment_output_folder.mkdir(parents=True, exist_ok=True)

    discovered = _discover_aligned_dataset_sets(dataset_folder)
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    skipped: list[dict[str, Any]] = []
    for item in discovered:
        sample_name = str(item["sample"])
        split_folder_name = str(item["split_folder_name"])
        side = str(item["side"])
        key = (split_folder_name, sample_name)
        grouped.setdefault(key, {"sample": sample_name, "split_folder_name": split_folder_name, "sides": {}})
        if not bool(item.get("complete")):
            skipped.append(
                {
                    "sample": sample_name,
                    "split_folder_name": split_folder_name,
                    "side": side,
                    "stage": "discovery",
                    "reason": "Missing required file(s): " + ", ".join(str(path) for path in item.get("missing", [])),
                }
            )
            continue
        try:
            grouped[key]["sides"][side] = _build_aligned_dataset_side_alignment(
                sample_name=sample_name,
                split_folder_name=split_folder_name,
                side=side,
                dataset_folder=dataset_folder,
                alignment_output_folder=alignment_output_folder,
                plot_output_folder=plot_output_folder,
                resize_width=resize_width,
                window_seconds=window_seconds,
            )
        except Exception as exc:
            skipped.append(
                {
                    "sample": sample_name,
                    "split_folder_name": split_folder_name,
                    "side": side,
                    "stage": "alignment",
                    "reason": str(exc),
                }
            )

    split_summaries: list[dict[str, Any]] = []
    exported_segment_count = 0
    for (split_folder_name, sample_name), group in sorted(grouped.items()):
        side_results = group.get("sides", {})
        if not side_results:
            continue
        split_summary = _export_aligned_dataset_split_segments(
            sample_name=sample_name,
            split_folder_name=split_folder_name,
            side_results=side_results,
            output_folder=output_folder,
            segment_seconds=segment_seconds,
            summary_file_name=f"{sample_name}_aligned_segments_summary.json",
        )
        skipped.extend(split_summary.get("skipped", []))
        exported_segment_count += int(split_summary.get("exported_segment_count") or 0)
        split_summaries.append(
            {
                "sample": sample_name,
                "split_folder_name": split_folder_name,
                "summary_file": split_summary.get("summary_file"),
                "output_folder": split_summary.get("output_folder"),
                "exported_segment_count": split_summary.get("exported_segment_count", 0),
                "skipped_count": split_summary.get("skipped_count", 0),
                "sides": split_summary.get("sides", {}),
            }
        )

    summary = {
        "output_folder": str(output_folder),
        "segment_seconds": segment_seconds,
        "discovered_count": len(discovered),
        "split_count": len(split_summaries),
        "exported_segment_count": exported_segment_count,
        "skipped_count": len(skipped),
        "splits": split_summaries,
        "skipped": skipped,
    }
    summary_path = output_folder / "aligned_dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    summary["summary_file"] = str(summary_path)
    return summary


def run_and_export_all_rgb_event_dtw_with_audio_alignments(
    dataset_folder: Path | str = "dataset",
    alignment_output_folder: Path | str = ".",
    plot_output_folder: Path | str | None = DEFAULT_PLOT_OUTPUT_FOLDER,
    output_folder: Path | str = DEFAULT_EXPORT_OUTPUT_FOLDER,
    window_seconds: float = 10.0,
    resize_width: int = 160,
) -> dict[str, Any]:
    """Run combined RGB/EVENT/IR/DEPTH visuals with fixed-offset audio for all complete samples."""
    dataset_folder = Path(dataset_folder)
    alignment_output_folder = Path(alignment_output_folder)
    plot_output_folder = Path(plot_output_folder) if plot_output_folder is not None else None
    output_folder = Path(output_folder)
    alignment_output_folder.mkdir(parents=True, exist_ok=True)
    output_folder.mkdir(parents=True, exist_ok=True)

    discovered = _discover_rgb_event_audio_triplets(dataset_folder)
    exported: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for triplet in discovered:
        sample_name = str(triplet["sample"])
        side = str(triplet["side"])
        reference_file = Path(triplet["reference_file"])
        event_file = Path(triplet["event_file"])
        ir_file = Path(triplet["ir_file"])
        depth_file = Path(triplet["depth_file"])
        audio_file = Path(triplet["audio_file"])
        dtw_path = _dtw_alignment_output_path(alignment_output_folder, sample_name, side)
        audio_path = _audio_alignment_output_path(alignment_output_folder, sample_name, side)
        combined_summary_path = (
            output_folder / f"{sample_name}_{side}_rgb_event_ir_depth_dtw_with_aligned_audio_summary.json"
        )

        if not bool(triplet.get("complete")):
            skipped.append(
                {
                    "sample": sample_name,
                    "side": side,
                    "reference_file": str(reference_file),
                    "event_file": str(event_file),
                    "ir_file": str(ir_file),
                    "depth_file": str(depth_file),
                    "audio_file": str(audio_file),
                    "dtw_alignment_file": str(dtw_path),
                    "audio_alignment_file": str(audio_path),
                    "reason": "Missing required file(s): " + ", ".join(str(item) for item in triplet.get("missing", [])),
                }
            )
            continue

        try:
            pair_summary = _run_and_export_rgb_event_dtw_with_audio_alignment(
                sample_name=sample_name,
                split_folder_name=str(triplet["split_folder_name"]),
                side=side,
                dataset_folder=dataset_folder,
                dtw_output_path=dtw_path,
                audio_output_path=audio_path,
                combined_summary_path=combined_summary_path,
                plot_output_folder=plot_output_folder,
                output_folder=output_folder,
                window_seconds=window_seconds,
                resize_width=resize_width,
                reference_file=reference_file,
                event_file=event_file,
                ir_file=ir_file,
                depth_file=depth_file,
                audio_file=audio_file,
                keep_intermediate_dtw_video=False,
            )
            if pair_summary.get("exported"):
                export_item = dict(pair_summary["exported"][0])
                export_item.update(
                    {
                        "dtw_alignment_file": str(dtw_path),
                        "audio_alignment_file": str(audio_path),
                        "pair_summary_file": str(combined_summary_path),
                        "source_rgb_file": str(reference_file),
                        "source_event_file": str(event_file),
                        "source_ir_file": str(ir_file),
                        "source_depth_file": str(depth_file),
                        "source_audio_file": str(audio_file),
                        "dtw": pair_summary.get("dtw"),
                        "ir": pair_summary.get("ir"),
                        "depth": pair_summary.get("depth"),
                        "audio": pair_summary.get("audio"),
                    }
                )
                exported.append(export_item)
            else:
                reason = "Combined export did not produce an output video."
                pair_skipped = pair_summary.get("skipped") or []
                if pair_skipped and isinstance(pair_skipped[0], dict):
                    reason = str(pair_skipped[0].get("reason") or reason)
                skipped.append(
                    {
                        "sample": sample_name,
                        "side": side,
                        "reference_file": str(reference_file),
                        "event_file": str(event_file),
                        "ir_file": str(ir_file),
                        "depth_file": str(depth_file),
                        "audio_file": str(audio_file),
                        "dtw_alignment_file": str(dtw_path),
                        "audio_alignment_file": str(audio_path),
                        "pair_summary_file": str(combined_summary_path),
                        "reason": reason,
                    }
                )
        except Exception as exc:
            skipped.append(
                {
                    "sample": sample_name,
                    "side": side,
                    "reference_file": str(reference_file),
                    "event_file": str(event_file),
                    "ir_file": str(ir_file),
                    "depth_file": str(depth_file),
                    "audio_file": str(audio_file),
                    "dtw_alignment_file": str(dtw_path),
                    "audio_alignment_file": str(audio_path),
                    "pair_summary_file": str(combined_summary_path),
                    "reason": str(exc),
                }
            )

    summary = {
        "discovered_count": len(discovered),
        "exported_count": len(exported),
        "skipped_count": len(skipped),
        "exported": exported,
        "skipped": skipped,
    }
    summary_path = output_folder / "rgb_event_ir_depth_dtw_with_audio_all_export_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
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
