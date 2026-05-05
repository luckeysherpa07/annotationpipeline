"""Preprocessor for reusing cached RGB/IR frames and estimating Marigold depth."""

import re
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def _build_flat_cache_pair_key(folder_name: str) -> str:
    """Build a logical pair key from a flat RGB/IR cache folder name."""
    stem = folder_name.lower()
    stem = re.sub(r"_(rgb|ir)$", "", stem)
    stem = stem.replace("_day", "")
    stem = stem.replace("_night", "")
    stem = re.sub(r"__+", "_", stem).strip("_")
    return stem


def _get_flat_cache_side(folder_name: str, cache_modality: str = "rgb") -> str | None:
    """Infer whether a flat RGB/IR cache folder contains day or night frames."""
    stem = folder_name.lower()
    modality_suffix = re.escape(cache_modality.lower())
    if re.search(rf"_day_{modality_suffix}$", stem):
        return "day"
    if re.search(rf"_night\d*_{modality_suffix}$", stem):
        return "night"
    return None


def _is_night_ir_cache_folder(folder: Path) -> bool:
    """Return whether a folder is a flat night IR frame cache with frame PNGs."""
    return (
        folder.is_dir()
        and _get_flat_cache_side(folder.name, cache_modality="ir") == "night"
        and any(folder.glob("frame_*.png"))
    )


def _expected_depth_outputs(frames: List[Path], output_dir: Path) -> List[Path]:
    """Build the expected Marigold output path for each cached frame."""
    return [output_dir / f"{frame.stem}_depth.png" for frame in frames]


def list_cached_frame_folders(
    dataset_folder: Path,
    cache_subdir: str = ".frames_cache",
    night_ir_only: bool = False,
) -> List[Path]:
    """List direct frame cache folders under the cache root."""
    cache_dir = dataset_folder / cache_subdir
    if not cache_dir.exists():
        return []
    folders = [item for item in cache_dir.iterdir() if item.is_dir()]
    if night_ir_only:
        folders = [item for item in folders if _is_night_ir_cache_folder(item)]
    return sorted(folders, key=lambda item: item.name.lower())


def list_cached_rgb_folders(
    dataset_folder: Path,
    cache_subdir: str = ".frames_cache",
) -> List[Path]:
    """List direct RGB cache folders under the cache root."""
    return list_cached_frame_folders(dataset_folder, cache_subdir=cache_subdir)


def list_cached_ir_night_folders(
    dataset_folder: Path,
    cache_subdir: str = ".frames_cache_ir",
) -> List[Path]:
    """List direct night IR cache folders under the cache root."""
    return list_cached_frame_folders(
        dataset_folder,
        cache_subdir=cache_subdir,
        night_ir_only=True,
    )


def get_cached_frames(
    dataset_folder: Path,
    cache_subdir: str = ".frames_cache",
    cache_modality: str = "rgb",
) -> Dict[str, Dict[str, List[Path]]]:
    """Retrieve already-extracted frames from flat or nested cache layouts."""
    cache_dir = dataset_folder / cache_subdir

    if not cache_dir.exists():
        print(f"ERROR: Cache directory not found: {cache_dir}")
        print(f"Please run {cache_modality.upper()} pipeline first to extract frames.")
        return {}

    paired_frames: Dict[str, Dict[str, List[Path]]] = {}

    for frame_dir in sorted(cache_dir.iterdir(), key=lambda item: item.name.lower()):
        if not frame_dir.is_dir():
            continue

        nested_day_frames = sorted(frame_dir.glob("day/frame_*.png"))
        nested_night_frames = sorted(frame_dir.glob("night/frame_*.png"))
        if nested_day_frames or nested_night_frames:
            paired_frames[frame_dir.name] = {
                "day": nested_day_frames,
                "night": nested_night_frames,
            }
            continue

        flat_frames = sorted(frame_dir.glob("frame_*.png"))
        if not flat_frames:
            continue

        side = _get_flat_cache_side(frame_dir.name, cache_modality=cache_modality)
        if side is None:
            print(
                f"WARNING: Skipping unrecognized {cache_modality.upper()} "
                f"cache folder: {frame_dir.name}"
            )
            continue

        pair_key = _build_flat_cache_pair_key(frame_dir.name)
        paired_frames.setdefault(pair_key, {"day": [], "night": []})
        paired_frames[pair_key][side] = flat_frames

    return paired_frames


def get_cached_rgb_frames(
    dataset_folder: Path,
    cache_subdir: str = ".frames_cache",
) -> Dict[str, Dict[str, List[Path]]]:
    """Retrieve already-extracted RGB frames from flat or nested cache layouts."""
    return get_cached_frames(
        dataset_folder,
        cache_subdir=cache_subdir,
        cache_modality="rgb",
    )


def get_cached_ir_frames(
    dataset_folder: Path,
    cache_subdir: str = ".frames_cache_ir",
) -> Dict[str, Dict[str, List[Path]]]:
    """Retrieve already-extracted IR frames from flat or nested cache layouts."""
    return get_cached_frames(
        dataset_folder,
        cache_subdir=cache_subdir,
        cache_modality="ir",
    )


def resolve_cached_pair_from_folder(
    selected_folder: str | Path,
    dataset_folder: Path,
    cache_subdir: str = ".frames_cache",
    cache_modality: str = "rgb",
) -> Dict[str, Dict[str, List[Path]]]:
    """Resolve one selected flat cache folder into its logical day/night pair."""
    cache_dir = dataset_folder / cache_subdir
    folder_name = Path(selected_folder).name
    folder_path = cache_dir / folder_name

    if not folder_path.exists() or not folder_path.is_dir():
        print(f"ERROR: {cache_modality.upper()} cache folder not found: {folder_path}")
        return {}

    all_pairs = get_cached_frames(
        dataset_folder,
        cache_subdir=cache_subdir,
        cache_modality=cache_modality,
    )

    nested_day_frames = sorted(folder_path.glob("day/frame_*.png"))
    nested_night_frames = sorted(folder_path.glob("night/frame_*.png"))
    if nested_day_frames or nested_night_frames:
        return {
            folder_name: {
                "day": nested_day_frames,
                "night": nested_night_frames,
            }
        }

    pair_key = _build_flat_cache_pair_key(folder_name)
    selected_pair = all_pairs.get(pair_key)
    if not selected_pair:
        print(f"ERROR: Could not resolve a logical pair from cache folder: {folder_name}")
        return {}

    if not selected_pair.get("day") or not selected_pair.get("night"):
        print(
            f"WARNING: Incomplete pair for '{folder_name}'. "
            "Marigold will process only the available side."
        )

    return {
        pair_key: {
            "day": selected_pair.get("day", []),
            "night": selected_pair.get("night", []),
        }
    }


def resolve_cached_rgb_pair_from_folder(
    selected_folder: str | Path,
    dataset_folder: Path,
    cache_subdir: str = ".frames_cache",
) -> Dict[str, Dict[str, List[Path]]]:
    """Resolve one selected flat RGB cache folder into its logical day/night pair."""
    return resolve_cached_pair_from_folder(
        selected_folder,
        dataset_folder=dataset_folder,
        cache_subdir=cache_subdir,
        cache_modality="rgb",
    )


def resolve_cached_ir_pair_from_folder(
    selected_folder: str | Path,
    dataset_folder: Path,
    cache_subdir: str = ".frames_cache_ir",
) -> Dict[str, Dict[str, List[Path]]]:
    """Resolve one selected flat IR cache folder into its logical day/night pair."""
    return resolve_cached_pair_from_folder(
        selected_folder,
        dataset_folder=dataset_folder,
        cache_subdir=cache_subdir,
        cache_modality="ir",
    )


def resolve_cached_ir_night_pair_from_folder(
    selected_folder: str | Path,
    dataset_folder: Path,
    cache_subdir: str = ".frames_cache_ir",
) -> Dict[str, Dict[str, List[Path]]]:
    """Resolve one selected flat night IR cache folder into a night-only pair."""
    cache_dir = dataset_folder / cache_subdir
    folder_name = Path(selected_folder).name
    folder_path = cache_dir / folder_name

    if not folder_path.exists() or not folder_path.is_dir():
        print(f"ERROR: IR cache folder not found: {folder_path}")
        return {}

    if _get_flat_cache_side(folder_name, cache_modality="ir") != "night":
        print(f"ERROR: Selected cache folder is not a night IR cache: {folder_name}")
        return {}

    night_frames = sorted(folder_path.glob("frame_*.png"))
    if not night_frames:
        print(f"ERROR: No frame_*.png files found in IR cache folder: {folder_path}")
        return {}

    pair_key = _build_flat_cache_pair_key(folder_name)
    return {
        pair_key: {
            "day": [],
            "night": night_frames,
        }
    }


def resolve_cached_rgb_frame_from_folder(
    selected_folder: str | Path,
    selected_frame: str | Path,
    dataset_folder: Path,
    cache_subdir: str = ".frames_cache",
) -> Dict[str, Dict[str, List[Path]]]:
    """Resolve one selected RGB frame into the logical Marigold day/night pair shape."""
    cache_dir = dataset_folder / cache_subdir
    folder_name = Path(selected_folder).name
    folder_path = cache_dir / folder_name

    if not folder_path.exists() or not folder_path.is_dir():
        print(f"ERROR: RGB cache folder not found: {folder_path}")
        return {}

    selected_frame_path = Path(selected_frame)
    if not selected_frame_path.is_absolute():
        candidate_paths = [
            folder_path / selected_frame_path,
            folder_path / selected_frame_path.name,
            folder_path / "day" / selected_frame_path.name,
            folder_path / "night" / selected_frame_path.name,
        ]
        selected_frame_path = next((path for path in candidate_paths if path.exists()), selected_frame_path)

    try:
        selected_frame_path = selected_frame_path.resolve(strict=True)
    except FileNotFoundError:
        print(f"ERROR: RGB frame not found: {selected_frame}")
        return {}

    try:
        selected_frame_path.relative_to(folder_path.resolve())
    except ValueError:
        print(f"ERROR: Selected frame is not inside cache folder '{folder_name}': {selected_frame_path}")
        return {}

    nested_day_frames = sorted(folder_path.glob("day/frame_*.png"))
    nested_night_frames = sorted(folder_path.glob("night/frame_*.png"))
    if nested_day_frames or nested_night_frames:
        side = selected_frame_path.parent.name.lower()
        if side not in {"day", "night"}:
            print(f"ERROR: Could not infer day/night side for frame: {selected_frame_path.name}")
            return {}
        return {
            folder_name: {
                "day": [selected_frame_path] if side == "day" else [],
                "night": [selected_frame_path] if side == "night" else [],
            }
        }

    side = _get_flat_cache_side(folder_name)
    if side is None:
        print(f"ERROR: Could not infer day/night side from cache folder: {folder_name}")
        return {}

    pair_key = _build_flat_cache_pair_key(folder_name)
    return {
        pair_key: {
            "day": [selected_frame_path] if side == "day" else [],
            "night": [selected_frame_path] if side == "night" else [],
        }
    }


def preprocess_marigold_depth(
    dataset_folder: Path,
    output_subdir: str = ".frames_cache_marigold",
    model_name: str = "prs-eth/marigold-depth-v1-1",
    device: str = "auto",
    rgb_frames: Dict[str, Dict[str, List[Path]]] | None = None,
    cache_frames: Dict[str, Dict[str, List[Path]]] | None = None,
    cache_subdir: str = ".frames_cache",
    cache_modality: str = "rgb",
) -> Dict[str, Dict[str, List[Path]]]:
    """Estimate Marigold depth maps from cached RGB/IR frames."""
    from annotation_feature.pipeline.modalities.marigold.marigold_depth_estimator import (
        get_depth_estimator,
    )

    cache_frames = cache_frames if cache_frames is not None else rgb_frames
    cache_frames = cache_frames or get_cached_frames(
        dataset_folder,
        cache_subdir=cache_subdir,
        cache_modality=cache_modality,
    )
    if not cache_frames:
        print(
            f"No cached {cache_modality.upper()} frames found. "
            "Exiting Marigold depth estimation."
        )
        return {}

    print(f"\nFound {len(cache_frames)} video pairs with cached {cache_modality.upper()} frames")
    estimator = get_depth_estimator(model_name=model_name, device=device)

    output_dir = dataset_folder / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    marigold_frames: Dict[str, Dict[str, List[Path]]] = {}

    for pair_key, frames in cache_frames.items():
        print(f"\n{'=' * 60}")
        print(f"Processing pair: {pair_key}")
        print(f"{'=' * 60}")

        pair_output_dir = output_dir / pair_key
        pair_output_dir.mkdir(parents=True, exist_ok=True)
        marigold_frames[pair_key] = {"day": [], "night": []}

        if frames["day"]:
            day_output_dir = pair_output_dir / "day"
            day_output_dir.mkdir(parents=True, exist_ok=True)
            expected_day_outputs = _expected_depth_outputs(frames["day"], day_output_dir)
            if all(output.exists() for output in expected_day_outputs):
                print(f"\nReusing {len(expected_day_outputs)} cached Marigold day depth maps:")
                for output in expected_day_outputs:
                    print(f"  {output}")
                marigold_frames[pair_key]["day"] = expected_day_outputs
            else:
                print(f"\nEstimating depth for {len(frames['day'])} day frames...")
                day_depths = estimator.estimate_depth_batch(
                    frames["day"], day_output_dir, save_format="png"
                )
                marigold_frames[pair_key]["day"] = day_depths
                print(f"Saved {len(day_depths)} day depth maps")

        if frames["night"]:
            night_output_dir = pair_output_dir / "night"
            night_output_dir.mkdir(parents=True, exist_ok=True)
            expected_night_outputs = _expected_depth_outputs(frames["night"], night_output_dir)
            if all(output.exists() for output in expected_night_outputs):
                print(f"\nReusing {len(expected_night_outputs)} cached Marigold night depth maps:")
                for output in expected_night_outputs:
                    print(f"  {output}")
                marigold_frames[pair_key]["night"] = expected_night_outputs
            else:
                print(f"\nEstimating depth for {len(frames['night'])} night frames...")
                night_depths = estimator.estimate_depth_batch(
                    frames["night"], night_output_dir, save_format="png"
                )
                marigold_frames[pair_key]["night"] = night_depths
                print(f"Saved {len(night_depths)} night depth maps")

        if not frames["day"] or not frames["night"]:
            print(
                "WARNING: Pair is incomplete. Generated Marigold depths only for the available side."
            )

    print(f"\n{'=' * 60}")
    print("Marigold depth estimation complete!")
    print(f"Processed {len(marigold_frames)} pairs")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 60}\n")

    return marigold_frames
