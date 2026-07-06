"""Modality-aligned video frame sampling logic."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from annotation_feature.pipeline.utils import infer_recording_side

@dataclass
class MultimodalSamplingJob:
    segment_id: str
    split_dir: str
    segment_name: str
    side: str
    modality1: str
    modality2: str
    dir1: Path
    dir2: Path
    shared_indexes: tuple[int, ...]
    by_index1: dict[int, Path]
    by_index2: dict[int, Path]


ADAPTIVE_CHANGE_WEIGHT = 0.7
ADAPTIVE_COVERAGE_WEIGHT = 0.3

FRAME_CACHE_SUBDIRS = {
    "rgb": ".frames_cache",
    "ir": ".frames_cache_ir",
    "event": ".frames_cache_event",
}
SIDE_ORDER = ("day", "night", "unknown")


def frame_index(path: Path) -> int | None:
    match = re.search(r"frame_(\d+)", path.name)
    return int(match.group(1)) if match else None


def frames_by_index(frame_dir: Path, modality: str) -> dict[int, Path]:
    pattern = "frame_*_depth.png" if modality == "depth" else "frame_*.png"
    frames: dict[int, Path] = {}
    for path in sorted(frame_dir.glob(pattern)):
        index = frame_index(path)
        if index is not None:
            frames[index] = path
    return frames


def evenly_sample(values: list[int], count: int) -> list[int]:
    if count <= 0 or not values:
        return []
    if len(values) <= count:
        return values
    if count == 1:
        return [values[len(values) // 2]]
    last = len(values) - 1
    selected = [values[round(i * last / (count - 1))] for i in range(count)]
    return list(dict.fromkeys(selected))


def side_key(path: Path) -> str:
    return infer_recording_side(path.name) or "unknown"


def load_standard_frame_dirs(
    dataset_root: Path,
    split_dir: str,
    segment_name: str,
    modality: str,
) -> dict[str, Path]:
    cache_subdir = FRAME_CACHE_SUBDIRS[modality]
    base = dataset_root / cache_subdir / split_dir / segment_name
    if not base.exists():
        return {}
    dirs: dict[str, Path] = {}
    for path in sorted(base.iterdir()):
        if not path.is_dir():
            continue
        name = path.name.lower()
        if not name.endswith(f"_{modality}"):
            continue
        side = side_key(path)
        dirs.setdefault(side, path)
    return dirs


def load_depth_frame_dirs(dataset_root: Path, split_dir: str, segment_name: str) -> dict[str, Path]:
    base = dataset_root / ".frames_cache_marigold" / split_dir / segment_name
    if not base.exists():
        return {}
    dirs: dict[str, Path] = {}
    for pair_dir in sorted(base.iterdir()):
        if not pair_dir.is_dir():
            continue
        for side_dir in sorted(pair_dir.iterdir()):
            if not side_dir.is_dir():
                continue
            side = side_key(side_dir)
            if side == "unknown" and side_dir.name.lower() in {"day", "night"}:
                side = side_dir.name.lower()
            dirs.setdefault(side, side_dir)
    return dirs


def load_frame_dirs(
    dataset_root: Path,
    split_dir: str,
    segment_name: str,
    modality: str,
) -> dict[str, Path]:
    if modality == "depth":
        return load_depth_frame_dirs(dataset_root, split_dir, segment_name)
    if modality not in FRAME_CACHE_SUBDIRS:
        return {}
    return load_standard_frame_dirs(dataset_root, split_dir, segment_name, modality)


def pair_frame_dirs(
    helper_dirs: dict[str, Path],
    victim_dirs: dict[str, Path],
) -> list[tuple[str, Path, Path]]:
    sides = [side for side in SIDE_ORDER if side in helper_dirs and side in victim_dirs]
    sides.extend(
        sorted(
            side
            for side in set(helper_dirs) & set(victim_dirs)
            if side not in SIDE_ORDER
        )
    )
    return [(side, helper_dirs[side], victim_dirs[side]) for side in sides]


def load_thumbnail(path: Path, thumbnail_cache: dict[Path, np.ndarray]) -> np.ndarray:
    if path in thumbnail_cache:
        return thumbnail_cache[path]
    with Image.open(path) as img:
        img_gray = img.convert("L").resize((160, 90), Image.Resampling.BILINEAR)
        arr = np.array(img_gray, dtype=np.float32) / 255.0
        thumbnail_cache[path] = arr
        return arr


def local_temporal_change_score(t_index: int, shared_indexes: list[int], by_index: dict[int, Path], thumbnail_cache: dict[Path, np.ndarray]) -> float:
    try:
        idx = shared_indexes.index(t_index)
    except ValueError:
        return 0.0
    
    t_img = load_thumbnail(by_index[t_index], thumbnail_cache)
    
    d_prev = None
    if idx > 0:
        prev_idx = shared_indexes[idx - 1]
        prev_img = load_thumbnail(by_index[prev_idx], thumbnail_cache)
        d_prev = float(np.mean(np.abs(t_img - prev_img)))
        
    d_next = None
    if idx < len(shared_indexes) - 1:
        next_idx = shared_indexes[idx + 1]
        next_img = load_thumbnail(by_index[next_idx], thumbnail_cache)
        d_next = float(np.mean(np.abs(next_img - t_img)))
        
    if d_prev is not None and d_next is not None:
        return (d_prev + d_next) / 2.0
    elif d_prev is not None:
        return d_prev
    elif d_next is not None:
        return d_next
    else:
        return 0.0


def robust_normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    arr = np.array(scores, dtype=np.float64)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    
    eps = 1e-6
    if mad < eps:
        max_score = float(np.max(arr))
        min_score = float(np.min(arr))
        if max_score - min_score < eps:
            return [0.0] * len(scores)
        else:
            return [float((s - min_score) / (max_score - min_score)) for s in arr]
            
    z = (arr - median) / (1.4826 * mad + eps)
    z = np.clip(z, 0.0, 5.0)
    z_min = float(np.min(z))
    z_max = float(np.max(z))
    if z_max - z_min < eps:
        return [0.0] * len(scores)
    return [(float(v) - z_min) / (z_max - z_min) for v in z]


def select_adaptive_refinement_indexes(
    shared_indexes: list[int],
    by_index1: dict[int, Path],
    by_index2: dict[int, Path],
    num_uniform_frames: int,
    num_adaptive_frames: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    anchors = tuple(evenly_sample(shared_indexes, min(num_uniform_frames, len(shared_indexes))))
    if len(shared_indexes) <= num_uniform_frames or num_adaptive_frames <= 0:
        return anchors, (), anchors
        
    candidates = [idx for idx in shared_indexes if idx not in anchors]
    if not candidates:
        return anchors, (), anchors
        
    thumbnail_cache: dict[Path, np.ndarray] = {}
    
    scores1 = [local_temporal_change_score(idx, shared_indexes, by_index1, thumbnail_cache) for idx in candidates]
    scores2 = [local_temporal_change_score(idx, shared_indexes, by_index2, thumbnail_cache) for idx in candidates]
    
    norm1 = robust_normalize_scores(scores1)
    norm2 = robust_normalize_scores(scores2)
    
    multimodal_change_scores = []
    for c1, c2 in zip(norm1, norm2):
        mean_s = (c1 + c2) / 2.0
        max_s = max(c1, c2)
        multimodal_change_scores.append(0.5 * mean_s + 0.5 * max_s)
        
    position_by_index = {idx: pos for pos, idx in enumerate(shared_indexes)}
    selected_set = list(anchors)
    adaptive_frames = []
    
    for _ in range(min(num_adaptive_frames, len(candidates))):
        raw_coverage = []
        for cand in candidates:
            cand_pos = position_by_index[cand]
            dist = min(abs(cand_pos - position_by_index[sel]) for sel in selected_set)
            raw_coverage.append(dist)
            
        max_cov = max(raw_coverage)
        if max_cov > 0:
            norm_coverage = [r / max_cov for r in raw_coverage]
        else:
            norm_coverage = [0.0 for _ in raw_coverage]
            
        final_scores = [
            ADAPTIVE_CHANGE_WEIGHT * mc + ADAPTIVE_COVERAGE_WEIGHT * nc
            for mc, nc in zip(multimodal_change_scores, norm_coverage)
        ]
        
        best_candidate_idx = min(
            range(len(candidates)),
            key=lambda i: (
                -final_scores[i],
                -multimodal_change_scores[i],
                -norm_coverage[i],
                position_by_index[candidates[i]],
            )
        )
        
        chosen = candidates[best_candidate_idx]
        selected_set.append(chosen)
        adaptive_frames.append(chosen)
        
        candidates.pop(best_candidate_idx)
        multimodal_change_scores.pop(best_candidate_idx)
        
    return anchors, tuple(adaptive_frames), tuple(sorted(selected_set))



def select_paired_frames(
    dir1: Path,
    dir2: Path,
    modality1: str,
    modality2: str,
    sampling_strategy: str,
    num_uniform_frames: int,
    num_adaptive_frames: int,
) -> tuple[tuple[Path, ...], tuple[Path, ...], str, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    by_index1 = frames_by_index(dir1, modality1)
    by_index2 = frames_by_index(dir2, modality2)
    shared_indexes = sorted(set(by_index1) & set(by_index2))
    
    if len(shared_indexes) != 30:
        raise ValueError(f"Expected exactly 30 shared frames per segment, got {len(shared_indexes)}")
    
    candidate_frame_indexes = tuple(shared_indexes)
    
    if sampling_strategy == "uniform":
        selected_indexes = evenly_sample(list(candidate_frame_indexes), num_uniform_frames)
        anchors = tuple(selected_indexes)
        adaptive = ()
        selected = tuple(selected_indexes)
        strategy = "uniform"
    elif sampling_strategy == "uniform_adaptive":
        anchors, adaptive, selected = select_adaptive_refinement_indexes(
            list(candidate_frame_indexes), by_index1, by_index2, num_uniform_frames, num_adaptive_frames
        )
        strategy = "uniform_adaptive"
    else:
        raise ValueError(f"Unknown sampling_strategy: {sampling_strategy}")
        
    return (
        tuple(by_index1[index] for index in selected),
        tuple(by_index2[index] for index in selected),
        strategy,
        anchors,
        adaptive,
        selected,
    )

