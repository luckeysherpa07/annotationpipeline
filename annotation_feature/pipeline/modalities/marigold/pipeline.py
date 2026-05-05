"""Marigold depth estimation pipeline orchestrator."""

import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from annotation_feature.marigold_preprocessor import (
    get_cached_rgb_frames,
    list_cached_rgb_folders,
    preprocess_marigold_depth,
    resolve_cached_rgb_frame_from_folder,
    resolve_cached_rgb_pair_from_folder,
)


def _selected_frame_output_path(
    dataset_folder: Path,
    rgb_frames: Dict[str, Dict[str, List[Path]]],
) -> Path | None:
    """Return the Marigold output path for a one-frame option 20 selection."""
    for pair_key, frames_by_side in rgb_frames.items():
        for side in ("day", "night"):
            frames = frames_by_side.get(side, [])
            if len(frames) == 1:
                return (
                    dataset_folder
                    / ".frames_cache_marigold"
                    / pair_key
                    / side
                    / f"{frames[0].stem}_depth.png"
                )
    return None


def run_marigold_depth_estimation(
    test_mode: bool = False,
    dataset_folder: Path | str = "dataset",
    model_name: str = "prs-eth/marigold-depth-v1-1",
    device: str = "auto",
    test_pair_index: int = 0,
    selected_cache_folder: str | None = None,
    selected_frame: str | Path | None = None,
) -> Dict[str, Dict[str, List[Path]]]:
    """Estimate Marigold depth maps from cached RGB frames."""
    print("=" * 60)
    print(
        "TEST MODE: Marigold depth estimation"
        if test_mode
        else "PRODUCTION: Marigold depth estimation on all RGB frames"
    )
    print("=" * 60)

    dataset_folder = Path(dataset_folder)
    if not dataset_folder.exists():
        print("ERROR: Dataset folder not found!")
        print(f"Expected to find RGB frames cache in: {dataset_folder}")
        return {}

    print(f"Dataset directory: {dataset_folder}")

    if selected_cache_folder and selected_frame:
        rgb_frames = resolve_cached_rgb_frame_from_folder(
            selected_cache_folder,
            selected_frame=selected_frame,
            dataset_folder=dataset_folder,
        )
        if rgb_frames:
            print(f"Resolved frame '{Path(selected_frame).name}' from cache folder '{selected_cache_folder}'")
    elif selected_cache_folder:
        rgb_frames = resolve_cached_rgb_pair_from_folder(
            selected_cache_folder,
            dataset_folder=dataset_folder,
        )
        if rgb_frames:
            print(f"Resolved cache folder '{selected_cache_folder}' to {len(rgb_frames)} logical pair")
    else:
        rgb_frames = get_cached_rgb_frames(dataset_folder)
        if test_mode and rgb_frames:
            pair_items = list(rgb_frames.items())
            rgb_frames = dict(pair_items[test_pair_index:test_pair_index + 1])
            print(f"Selected Marigold test pair index: {test_pair_index}")

    if not rgb_frames:
        print("No depth maps were generated.")
        return {}

    expected_selected_output = None
    if selected_cache_folder and selected_frame:
        expected_selected_output = _selected_frame_output_path(dataset_folder, rgb_frames)

    marigold_frames = preprocess_marigold_depth(
        dataset_folder=dataset_folder,
        output_subdir=".frames_cache_marigold",
        model_name=model_name,
        device=device,
        rgb_frames=rgb_frames,
    )
    if not marigold_frames:
        print("No depth maps were generated.")
        return {}

    total_day = sum(len(frames.get("day", [])) for frames in marigold_frames.values())
    total_night = sum(len(frames.get("night", [])) for frames in marigold_frames.values())

    print("\nSummary:")
    print(f"  Video pairs processed: {len(marigold_frames)}")
    print(f"  Total day depth maps: {total_day}")
    print(f"  Total night depth maps: {total_night}")
    print(f"  Total depth maps: {total_day + total_night}")
    if expected_selected_output:
        print(f"  Expected selected-frame output: {expected_selected_output}")
        print(f"  Output exists: {'yes' if expected_selected_output.exists() else 'no'}")

    return marigold_frames


__all__ = [
    "list_cached_rgb_folders",
    "run_marigold_depth_estimation",
]
