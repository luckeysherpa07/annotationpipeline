from pathlib import Path
import asyncio
import copy
import json
import os
import re
import sys
from typing import Dict

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from annotation_feature.demo_result import DEMO_RESULT
from annotation_feature.video_preprocessor import extract_frames, preprocess_videos
from annotation_feature.audio_preprocessor import preprocess_audio
from .client import create_gemini_client
from .utils import get_pair_key, infer_recording_side, is_modality_file, video_extensions, audio_extensions
from .modalities.rgb import run_parallel_pipeline
from .modalities.event import run_event_parallel_pipeline
from .modalities.depth import run_depth_parallel_pipeline
from .modalities.ir import run_ir_parallel_pipeline
from .modalities.audio import (
    format_audio_annotations,
    run_parallel_pipeline as run_audio_parallel_pipeline,
)
from annotation_feature.marigold_preprocessor import get_cached_marigold_depth_frames
from prompts.depth_prompts import DEPTH_PROMPTS
from prompts.event_prompts import EVENT_PROMPTS
from prompts.ir_prompts import IR_PROMPTS
from prompts.rgb_prompts import RGB_PROMPTS


ANNOTATION_PROMPT_KEYS = {
    "rgb": tuple(RGB_PROMPTS.keys()),
    "event": tuple(EVENT_PROMPTS.keys()),
    "ir": tuple(IR_PROMPTS.keys()),
    "depth": tuple(DEPTH_PROMPTS.keys()),
}

AUDIO_RGB_SOURCE_STEM_RE = re.compile(r"^(?P<sample>.+)_(?P<side>day|night\d*)_rgb$")
SKIPPED_MISSING_SIDE_STATUS = "skipped_missing_side"


def _load_existing_results(output_file: Path) -> Dict:
    if not output_file.exists():
        return {}

    try:
        with open(output_file, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"WARNING: Could not load existing results from {output_file}: {exc}")
        return {}

    if not isinstance(data, dict):
        print(f"WARNING: Existing results file is not a JSON object: {output_file}")
        return {}

    return {str(key).replace("\\", "/"): value for key, value in data.items()}


def _annotation_item_has_content(item: Dict) -> bool:
    if not isinstance(item, dict):
        return False
    return any(str(item.get(field, "")).strip() for field in ("caption", "question", "answer"))


def _annotation_item_complete(item: Dict) -> bool:
    if not isinstance(item, dict):
        return False
    return all(str(item.get(field, "")).strip() for field in ("caption", "question", "answer"))


def _annotations_complete(annotations: Dict, expected_keys: tuple[str, ...]) -> bool:
    if not isinstance(annotations, dict):
        return False
    return all(_annotation_item_complete(annotations.get(key, {})) for key in expected_keys)


def _annotations_have_content(annotations: Dict) -> bool:
    if not isinstance(annotations, dict):
        return False
    return any(_annotation_item_has_content(item) for item in annotations.values())


def _entry_complete(entry: Dict, expected_keys: tuple[str, ...]) -> bool:
    if not isinstance(entry, dict):
        return False
    if entry.get("status") == SKIPPED_MISSING_SIDE_STATUS:
        return True
    return _annotations_complete(entry.get("annotations", {}), expected_keys)


def _merge_annotations(existing_annotations: Dict, new_annotations: Dict) -> Dict:
    if not isinstance(existing_annotations, dict):
        existing_annotations = {}
    if not isinstance(new_annotations, dict):
        return copy.deepcopy(existing_annotations)

    if not _annotations_have_content(new_annotations) and _annotations_have_content(existing_annotations):
        return copy.deepcopy(existing_annotations)

    merged = copy.deepcopy(existing_annotations)
    for annotation_type, new_item in new_annotations.items():
        existing_item = merged.get(annotation_type, {})
        if _annotation_item_has_content(new_item) or not _annotation_item_has_content(existing_item):
            merged[annotation_type] = copy.deepcopy(new_item)
    return merged


def _write_results(output_file: Path, results: Dict) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)


def _result_entry_for_pair(dataset_folder: Path, pair_key: str, modality: str, annotations: Dict) -> Dict:
    night_file = None
    day_file = None
    for file in dataset_folder.rglob("*"):
        if not file.is_file() or file.suffix.lower() not in video_extensions:
            continue
        if not is_modality_file(file, modality):
            continue
        if get_pair_key(file) == pair_key:
            side = infer_recording_side(file)
            if side == "night":
                night_file = file
            elif side == "day":
                day_file = file

    return {
        "night_file": str(night_file) if night_file else None,
        "day_file": str(day_file) if day_file else None,
        "annotations": annotations,
    }


def _resume_filter_pairs(
    pairs_to_process: list[tuple[str, Dict[str, list]]],
    existing_results: Dict,
    expected_keys: tuple[str, ...],
    label: str,
) -> tuple[Dict[str, Dict[str, list]], int, int, int]:
    available_pairs = {
        pair_key: frames
        for pair_key, frames in pairs_to_process
        if frames.get("night") or frames.get("day")
    }
    complete_count = 0
    partial_count = 0
    empty_count = 0
    pending_pairs: Dict[str, Dict[str, list]] = {}

    for pair_key, frames in available_pairs.items():
        existing_entry = existing_results.get(pair_key)
        if _entry_complete(existing_entry, expected_keys):
            complete_count += 1
            continue

        existing_annotations = existing_entry.get("annotations", {}) if isinstance(existing_entry, dict) else {}
        if _annotations_have_content(existing_annotations):
            partial_count += 1
        else:
            empty_count += 1
        pending_pairs[pair_key] = frames

    print(
        f"Resume scan for {label}: {complete_count} complete skipped, "
        f"{partial_count} partial retry, {empty_count} empty/missing retry."
    )
    return pending_pairs, complete_count, partial_count, empty_count


def _audio_entry_complete(entry: Dict) -> bool:
    if not isinstance(entry, dict):
        return False

    annotations = entry.get("annotations", {})
    if not isinstance(annotations, dict):
        return False

    for section_name in ("audio_hia", "audio_chronological_caption"):
        section = annotations.get(section_name, {})
        if not isinstance(section, dict) or not str(section.get("caption", "")).strip():
            return False

    categories = annotations.get("categories", {})
    if not isinstance(categories, dict) or not categories:
        return False

    return all(_annotation_item_complete(item) for item in categories.values())


def _audio_annotations_have_content(annotations: Dict) -> bool:
    if not isinstance(annotations, dict):
        return False
    if _annotation_item_has_content(annotations.get("audio_hia", {})):
        return True
    if _annotation_item_has_content(annotations.get("audio_chronological_caption", {})):
        return True
    categories = annotations.get("categories", {})
    if isinstance(categories, dict):
        return _annotations_have_content(categories)
    return False


def _audio_source_pair_key(file: Path) -> str:
    match = AUDIO_RGB_SOURCE_STEM_RE.match(file.stem.lower())
    stem = match.group("sample") if match else file.stem.lower()
    return str(file.parent / stem)


def _discover_audio_rgb_videos(dataset_folder: Path) -> Dict[str, Dict[str, Path | None]]:
    rgb_videos: Dict[str, Dict[str, Path | None]] = {}

    for file in dataset_folder.rglob("*"):
        if not file.is_file() or file.suffix.lower() not in video_extensions:
            continue

        name = file.name.lower()
        stem = file.stem.lower()
        if not is_modality_file(file, "rgb") or "with_audio" in name:
            continue
        stem_match = AUDIO_RGB_SOURCE_STEM_RE.match(stem)
        if not stem_match:
            continue

        pair_key = _audio_source_pair_key(file)
        side = "night" if stem_match.group("side").startswith("night") else "day"
        rgb_videos.setdefault(pair_key, {"day": None, "night": None})

        if rgb_videos[pair_key][side] is not None:
            print(f"WARNING: Multiple {side} RGB source videos found for {pair_key}, using first one")
            continue

        rgb_videos[pair_key][side] = file

    return rgb_videos


def _load_or_extract_audio_hia_frames(day_rgb_video: Path, dataset_folder: Path) -> list[Path]:
    try:
        relative_parent = day_rgb_video.relative_to(dataset_folder).parent
    except ValueError:
        relative_parent = Path()
    frame_output_dir = dataset_folder / ".frames_cache_audio_hia" / relative_parent / day_rgb_video.stem
    cached_frames = sorted(frame_output_dir.glob("frame_*.png"))

    if cached_frames:
        print(f"Using cached HIA RGB frames for: {day_rgb_video.name} ({len(cached_frames)} frames)")
        return cached_frames

    print(f"Extracting HIA RGB frames from: {day_rgb_video.name}")
    frames = extract_frames(day_rgb_video, fps=1, output_dir=frame_output_dir)
    print(f"  Extracted {len(frames)} HIA RGB frames")
    return frames


def run(
    test_mode: bool = False,
    test_pair_index: int = 0,
    skip_api: bool = False,
    dataset_folder: Path | str = "dataset",
    output_file: Path | str = "rgb_qa_results.json",
):
    """
    Run the RGB annotation pipeline.

    Args:
        test_mode: If True, only process one video pair for testing
        test_pair_index: Which video pair to process in test mode (0 = first)
        skip_api: If True, skip Gemini API calls and use DEMO_RESULT instead
        dataset_folder: Dataset directory containing the source videos
        output_file: JSON path to write RGB QA results
    """
    if test_mode:
        print("=" * 50)
        print("TEST MODE: Processing only one RGB video pair")
        print("=" * 50)
        if skip_api:
            print("Gemini API calls disabled - using DEMO_RESULT data\n")

    dataset_folder = Path(dataset_folder)
    output_file = Path(output_file)
    results = _load_existing_results(output_file)

    if not dataset_folder.exists():
        print("ERROR: Dataset folder not found!")
        print(f"Expected to find videos in: {dataset_folder}")
        return results

    print(f"Dataset directory listing for {dataset_folder}:")
    print(os.listdir(dataset_folder))

    # Preprocess all videos and extract frames
    print("Preprocessing RGB videos...")
    paired_frames = preprocess_videos(dataset_folder, fps=1, video_type="rgb")
    print(f"Found {len(paired_frames)} video pairs\n")

    if len(paired_frames) == 0:
        print("ERROR: No video pairs found in dataset folder!")
        print(f"Expected to find videos in: {dataset_folder}")
        return results

    # In test mode, only process one pair
    if test_mode:
        pairs_to_process = list(paired_frames.items())[test_pair_index:test_pair_index + 1]
        print(f"Processing pair {test_pair_index} of {len(paired_frames)}:")
    else:
        pairs_to_process = list(paired_frames.items())

    expected_keys = ANNOTATION_PROMPT_KEYS["rgb"]
    available_pairs, _, _, _ = _resume_filter_pairs(
        pairs_to_process,
        results,
        expected_keys,
        "RGB",
    )

    if not available_pairs:
        print("No incomplete RGB pairs to process. Existing results are already complete for the selected pairs.")
        return results

    print(
        f"Processing {len(available_pairs)} batch pairs with up to 3 concurrent tasks and 4-second spacing..."
    )

    client = None
    if not skip_api:
        client = create_gemini_client()

    def checkpoint_pair(pair_key: str, annotation_results: Dict) -> None:
        existing_entry = results.get(pair_key, {})
        existing_annotations = existing_entry.get("annotations", {}) if isinstance(existing_entry, dict) else {}
        merged_annotations = _merge_annotations(existing_annotations, annotation_results)
        results[pair_key] = _result_entry_for_pair(dataset_folder, pair_key, "rgb", merged_annotations)
        _write_results(output_file, results)
        print(f"Checkpoint saved for: {pair_key}")

    batch_results = asyncio.run(
        run_parallel_pipeline(
            client,
            available_pairs,
            max_concurrent=3,
            delay_between_pairs=4,
            skip_api=skip_api,
            on_pair_complete=checkpoint_pair,
        )
    )

    for pair_key, frames in available_pairs.items():
        night_frames = frames.get("night") or []
        day_frames = frames.get("day") or []

        if not night_frames and not day_frames:
            print(f"Skipping {pair_key} - no frames found")
            continue

        file_results = batch_results.get(pair_key)
        if file_results is None:
            print(f"WARNING: No batch output for pair {pair_key}. Falling back to DEMO_RESULT.")
            file_results = copy.deepcopy(DEMO_RESULT)

        checkpoint_pair(pair_key, file_results)
        print(f"✓ Done: {pair_key}")

    # Save results to JSON file
    _write_results(output_file, results)

    print(f"\n" + "=" * 50)
    print(f"Results saved to: {output_file}")
    if test_mode:
        print("TEST MODE COMPLETE")
    print("=" * 50)
    return results


def run_event(
    test_mode: bool = False,
    test_pair_index: int = 0,
    skip_api: bool = False,
    dataset_folder: Path | str = "dataset",
    output_file: Path | str = "event_qa_results.json",
):
    """
    Run the EVENT annotation pipeline.

    Args:
        test_mode: If True, only process one video pair for testing
        test_pair_index: Which video pair to process in test mode (0 = first)
        skip_api: If True, skip Gemini API calls and return empty captions
        dataset_folder: Dataset directory containing the source videos
        output_file: JSON path to write EVENT QA results
    """
    if test_mode:
        print("=" * 50)
        print("TEST MODE: Processing only one EVENT video pair")
        print("=" * 50)
        if skip_api:
            print("Gemini API calls disabled - using demo captions\n")

    dataset_folder = Path(dataset_folder)
    output_file = Path(output_file)
    results = _load_existing_results(output_file)

    if not dataset_folder.exists():
        print("ERROR: Dataset folder not found!")
        print(f"Expected to find videos in: {dataset_folder}")
        return results

    print(f"Dataset directory listing for {dataset_folder}:")
    print(os.listdir(dataset_folder))

    # Preprocess all EVENT videos and extract frames
    print("Preprocessing EVENT videos...")
    paired_frames = preprocess_videos(dataset_folder, fps=1, video_type="event")
    print(f"Found {len(paired_frames)} event video pairs\n")

    if len(paired_frames) == 0:
        print("ERROR: No event video pairs found in dataset folder!")
        print(f"Expected to find videos with 'event' in filename in: {dataset_folder}")
        return results

    # In test mode, only process one pair
    if test_mode:
        pairs_to_process = list(paired_frames.items())[test_pair_index:test_pair_index + 1]
        print(f"Processing pair {test_pair_index} of {len(paired_frames)}:")
    else:
        pairs_to_process = list(paired_frames.items())

    expected_keys = ANNOTATION_PROMPT_KEYS["event"]
    available_pairs, _, _, _ = _resume_filter_pairs(
        pairs_to_process,
        results,
        expected_keys,
        "EVENT",
    )

    if not available_pairs:
        print("No incomplete EVENT pairs to process. Existing results are already complete for the selected pairs.")
        return results

    print(
        f"Processing {len(available_pairs)} event pairs with up to 3 concurrent tasks and 4-second spacing..."
    )

    client = None
    if not skip_api:
        client = create_gemini_client()

    def checkpoint_pair(pair_key: str, annotation_results: Dict) -> None:
        existing_entry = results.get(pair_key, {})
        existing_annotations = existing_entry.get("annotations", {}) if isinstance(existing_entry, dict) else {}
        merged_annotations = _merge_annotations(existing_annotations, annotation_results)
        results[pair_key] = _result_entry_for_pair(dataset_folder, pair_key, "event", merged_annotations)
        _write_results(output_file, results)
        print(f"Checkpoint saved for: {pair_key}")

    batch_results = asyncio.run(
        run_event_parallel_pipeline(
            client,
            available_pairs,
            max_concurrent=3,
            delay_between_pairs=4,
            skip_api=skip_api,
            on_pair_complete=checkpoint_pair,
        )
    )

    for pair_key, frames in available_pairs.items():
        night_frames = frames.get("night") or []
        day_frames = frames.get("day") or []

        if not night_frames and not day_frames:
            print(f"Skipping {pair_key} - no frames found")
            continue

        file_results = batch_results.get(pair_key)
        if file_results is None:
            print(f"WARNING: No batch output for pair {pair_key}. Using empty results.")
            from prompts.event_prompts import EVENT_PROMPTS
            file_results = {anno_type: {"caption": "", "question": "", "answer": ""} for anno_type in EVENT_PROMPTS.keys()}

        checkpoint_pair(pair_key, file_results)
        print(f"✓ Done: {pair_key}")

    _write_results(output_file, results)

    print(f"\n" + "=" * 50)
    print(f"Event QA results saved to: {output_file}")
    if test_mode:
        print("TEST MODE COMPLETE")
    print("=" * 50)
    return results


def run_depth(
    test_mode: bool = False,
    test_pair_index: int = 0,
    skip_api: bool = False,
    dataset_folder: Path | str = "dataset",
):
    """
    Run the DEPTH annotation pipeline.

    Args:
        test_mode: If True, only process one video pair for testing
        test_pair_index: Which video pair to process in test mode (0 = first)
        skip_api: If True, skip Gemini API calls and return demo results
        dataset_folder: Dataset directory containing the source videos
    """
    if test_mode:
        print("=" * 50)
        print("TEST MODE: Processing only one DEPTH video pair")
        print("=" * 50)
        if skip_api:
            print("Gemini API calls disabled - using demo results\n")

    client = None
    if not skip_api:
        client = create_gemini_client()

    dataset_folder = Path(dataset_folder)
    results = {}

    if not dataset_folder.exists():
        print("ERROR: Dataset folder not found!")
        print(f"Expected to find videos in: {dataset_folder}")
        return results

    print(f"Dataset directory listing for {dataset_folder}:")
    print(os.listdir(dataset_folder))

    # Preprocess all DEPTH videos and extract frames
    print("Preprocessing DEPTH videos...")
    paired_frames = preprocess_videos(dataset_folder, fps=1, video_type="depth")
    print(f"Found {len(paired_frames)} depth video pairs\n")

    if len(paired_frames) == 0:
        print("ERROR: No depth video pairs found in dataset folder!")
        print(f"Expected to find videos with 'depth' in filename in: {dataset_folder}")
        return results

    # In test mode, only process one pair
    if test_mode:
        pairs_to_process = list(paired_frames.items())[test_pair_index:test_pair_index + 1]
        print(f"Processing pair {test_pair_index} of {len(paired_frames)}:")
    else:
        pairs_to_process = list(paired_frames.items())

    available_pairs = {
        pair_key: frames
        for pair_key, frames in pairs_to_process
        if frames.get("night") or frames.get("day")
    }

    if not available_pairs:
        print("ERROR: No usable video frames found for selected pairs.")
        return results

    print(
        f"Processing {len(available_pairs)} depth pairs with up to 3 concurrent tasks and 4-second spacing..."
    )

    batch_results = asyncio.run(
        run_depth_parallel_pipeline(
            client,
            available_pairs,
            max_concurrent=3,
            delay_between_pairs=4,
            skip_api=skip_api,
        )
    )

    for pair_key, frames in pairs_to_process:
        night_frames = frames.get("night") or []
        day_frames = frames.get("day") or []

        if not night_frames and not day_frames:
            print(f"Skipping {pair_key} - no frames found")
            continue

        file_results = batch_results.get(pair_key)
        if file_results is None:
            print(f"WARNING: No batch output for pair {pair_key}. Using empty results.")
            from prompts.depth_prompts import DEPTH_PROMPTS
            file_results = {anno_type: {"caption": "", "question": "", "answer": ""} for anno_type in DEPTH_PROMPTS.keys()}

        night_file = None
        day_file = None
        for file in dataset_folder.rglob("*"):
            if not file.is_file() or file.suffix.lower() not in video_extensions:
                continue
            name = file.name.lower()
            if not is_modality_file(file, "depth"):
                continue
            if get_pair_key(file) == pair_key:
                if "night" in name:
                    night_file = file
                elif "day" in name:
                    day_file = file

        results[pair_key] = {
            "night_file": str(night_file) if night_file else None,
            "day_file": str(day_file) if day_file else None,
            "annotations": file_results,
        }
        print(f"✓ Done: {pair_key}")

    # Save results to JSON file at the project root
    output_file = Path("depth_qa_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n" + "=" * 50)
    print(f"Depth QA results saved to: {output_file}")
    if test_mode:
        print("TEST MODE COMPLETE")
    print("=" * 50)
    return results


def run_marigold_depth_qa(
    test_mode: bool = False,
    test_pair_index: int = 0,
    skip_api: bool = False,
    dataset_folder: Path | str = "dataset",
    cache_subdir: str = ".frames_cache_marigold",
    output_file: Path | str = "marigold_depth_qa_results.json",
):
    """
    Run the MARIGOLD DEPTH QA annotation pipeline.
    
    Uses Marigold-estimated depth frames from .frames_cache_marigold/ to generate
    depth-based QA annotations. Outputs results to marigold_depth_qa_results.json.

    Args:
        test_mode: If True, only process one video pair for testing
        test_pair_index: Which video pair to process in test mode (0 = first)
        skip_api: If True, skip Gemini API calls and return demo results
        dataset_folder: Dataset directory containing the source videos
        cache_subdir: Marigold depth cache directory name
        output_file: JSON path to write Marigold depth QA results
    """
    if test_mode:
        print("=" * 50)
        print("TEST MODE: Processing one MARIGOLD DEPTH QA pair")
        print("=" * 50)
        if skip_api:
            print("Gemini API calls disabled - using demo results\n")

    dataset_folder = Path(dataset_folder)
    output_file = Path(output_file)
    results = _load_existing_results(output_file)

    if not dataset_folder.exists():
        print("ERROR: Dataset folder not found!")
        print(f"Expected to find videos in: {dataset_folder}")
        return results

    print(f"Dataset directory: {dataset_folder}")

    # Load Marigold depth frames from cache
    print("Loading Marigold depth frames from cache...")
    cache_dir = dataset_folder / cache_subdir
    
    if not cache_dir.exists():
        print("ERROR: Marigold depth cache not found!")
        print(f"Expected to find depth maps at: {cache_dir}")
        print("Please run Marigold depth estimation first.")
        return results

    paired_frames = get_cached_marigold_depth_frames(
        dataset_folder,
        cache_subdir=cache_subdir,
    )

    print(f"Found {len(paired_frames)} Marigold depth video pairs\n")

    if len(paired_frames) == 0:
        print("ERROR: No Marigold depth frames found in cache!")
        print(f"Expected to find depth maps at: {cache_dir}")
        return results

    # In test mode, only process one pair
    if test_mode:
        pairs_to_process = list(paired_frames.items())[test_pair_index:test_pair_index + 1]
        print(f"Processing pair {test_pair_index} of {len(paired_frames)}:")
    else:
        pairs_to_process = list(paired_frames.items())

    expected_keys = ANNOTATION_PROMPT_KEYS["depth"]
    available_pairs, _, _, _ = _resume_filter_pairs(
        pairs_to_process,
        results,
        expected_keys,
        "Marigold depth",
    )

    if not available_pairs:
        print("No incomplete Marigold depth pairs to process. Existing results are already complete for the selected pairs.")
        return results

    print(
        f"Processing {len(available_pairs)} Marigold depth pairs with up to 3 concurrent tasks and 4-second spacing..."
    )

    client = None
    if not skip_api:
        client = create_gemini_client()

    def checkpoint_pair(pair_key: str, annotation_results: Dict) -> None:
        existing_entry = results.get(pair_key, {})
        existing_annotations = existing_entry.get("annotations", {}) if isinstance(existing_entry, dict) else {}
        merged_annotations = _merge_annotations(existing_annotations, annotation_results)
        frames = paired_frames.get(pair_key, {"day": [], "night": []})
        results[pair_key] = {
            "day_depth_count": len(frames.get("day", [])),
            "night_depth_count": len(frames.get("night", [])),
            "annotations": merged_annotations,
        }
        _write_results(output_file, results)
        print(f"Checkpoint saved for: {pair_key}")

    batch_results = asyncio.run(
        run_depth_parallel_pipeline(
            client,
            available_pairs,
            max_concurrent=3,
            delay_between_pairs=4,
            skip_api=skip_api,
            on_pair_complete=checkpoint_pair,
        )
    )

    for pair_key, frames in available_pairs.items():
        night_frames = frames.get("night") or []
        day_frames = frames.get("day") or []

        if not night_frames and not day_frames:
            print(f"Skipping {pair_key} - no frames found")
            continue

        file_results = batch_results.get(pair_key)
        if file_results is None:
            print(f"WARNING: No batch output for pair {pair_key}. Using empty results.")
            from prompts.depth_prompts import DEPTH_PROMPTS
            file_results = {anno_type: {"caption": "", "question": "", "answer": ""} for anno_type in DEPTH_PROMPTS.keys()}

        checkpoint_pair(pair_key, file_results)
        print(f"✓ Done: {pair_key}")

    _write_results(output_file, results)

    print(f"\n" + "=" * 50)
    print(f"Marigold Depth QA results saved to: {output_file}")
    if test_mode:
        print("TEST MODE COMPLETE")
    print("=" * 50)
    return results


def run_ir(
    test_mode: bool = False,
    test_pair_index: int = 0,
    skip_api: bool = False,
    dataset_folder: Path | str = "dataset",
    output_file: Path | str = "ir_qa_results.json",
    max_concurrent: int = 3,
    delay_between_pairs: int = 4,
):
    """
    Run the IR annotation pipeline.

    Args:
        test_mode: If True, only process one video pair for testing
        test_pair_index: Which video pair to process in test mode (0 = first)
        skip_api: If True, skip Gemini API calls and return demo results
        dataset_folder: Dataset directory containing the source videos
        output_file: JSON path to write IR QA results
        max_concurrent: Maximum concurrent Gemini calls
        delay_between_pairs: Delay between scheduling pair processing, in seconds
    """
    if test_mode:
        print("=" * 50)
        print("TEST MODE: Processing only one IR video pair")
        print("=" * 50)
        if skip_api:
            print("Gemini API calls disabled - using demo results\n")

    dataset_folder = Path(dataset_folder)
    output_file = Path(output_file)
    results = _load_existing_results(output_file)

    if not dataset_folder.exists():
        print("ERROR: Dataset folder not found!")
        print(f"Expected to find videos in: {dataset_folder}")
        return results

    print(f"Dataset directory listing for {dataset_folder}:")
    print(os.listdir(dataset_folder))

    print("Preprocessing IR videos...")
    paired_frames = preprocess_videos(dataset_folder, fps=1, video_type="ir")
    print(f"Found {len(paired_frames)} IR video pairs\n")

    if len(paired_frames) == 0:
        print("ERROR: No IR video pairs found in dataset folder!")
        print(f"Expected to find videos with 'ir' in filename in: {dataset_folder}")
        return results

    if test_mode:
        pairs_to_process = list(paired_frames.items())[test_pair_index:test_pair_index + 1]
        print(f"Processing pair {test_pair_index} of {len(paired_frames)}:")
    else:
        pairs_to_process = list(paired_frames.items())

    expected_keys = ANNOTATION_PROMPT_KEYS["ir"]
    available_pairs, _, _, _ = _resume_filter_pairs(
        pairs_to_process,
        results,
        expected_keys,
        "IR",
    )

    if not available_pairs:
        print("No incomplete IR pairs to process. Existing results are already complete for the selected pairs.")
        return results

    print(
        f"Processing {len(available_pairs)} IR pairs with up to {max_concurrent} "
        f"concurrent task(s) and {delay_between_pairs}-second spacing..."
    )

    client = None
    if not skip_api:
        client = create_gemini_client()

    def checkpoint_pair(pair_key: str, annotation_results: Dict) -> None:
        if annotation_results.get("status") == SKIPPED_MISSING_SIDE_STATUS:
            results[pair_key] = _result_entry_for_pair(dataset_folder, pair_key, "ir", {})
            results[pair_key].update(annotation_results)
            _write_results(output_file, results)
            print(f"Checkpoint saved for skipped pair: {pair_key}")
            return

        existing_entry = results.get(pair_key, {})
        existing_annotations = existing_entry.get("annotations", {}) if isinstance(existing_entry, dict) else {}
        merged_annotations = _merge_annotations(existing_annotations, annotation_results)
        results[pair_key] = _result_entry_for_pair(dataset_folder, pair_key, "ir", merged_annotations)
        _write_results(output_file, results)
        print(f"Checkpoint saved for: {pair_key}")

    batch_results = asyncio.run(
        run_ir_parallel_pipeline(
            client,
            available_pairs,
            max_concurrent=max_concurrent,
            delay_between_pairs=delay_between_pairs,
            skip_api=skip_api,
            on_pair_complete=checkpoint_pair,
        )
    )

    for pair_key, frames in available_pairs.items():
        night_frames = frames.get("night") or []
        day_frames = frames.get("day") or []

        if not night_frames and not day_frames:
            print(f"Skipping {pair_key} - no frames found")
            continue

        file_results = batch_results.get(pair_key)
        if file_results is None:
            print(f"WARNING: No batch output for pair {pair_key}. Using empty results.")
            from prompts.ir_prompts import IR_PROMPTS
            file_results = {anno_type: {"caption": "", "question": "", "answer": ""} for anno_type in IR_PROMPTS.keys()}

        checkpoint_pair(pair_key, file_results)
        print(f"âœ“ Done: {pair_key}")

    _write_results(output_file, results)

    print(f"\n" + "=" * 50)
    print(f"IR QA results saved to: {output_file}")
    if test_mode:
        print("TEST MODE COMPLETE")
    print("=" * 50)
    return results


def run_audio(
    test_mode: bool = False,
    test_pair_index: int = 0,
    skip_api: bool = False,
    dataset_folder: Path | str = "dataset",
    output_file: Path | str = "audio_qa_results.json",
):
    """
    Run the AUDIO annotation pipeline.

    Args:
        test_mode: If True, only process one audio-visual pair for testing
        test_pair_index: Which pair to process in test mode (0 = first)
        skip_api: If True, skip Gemini API calls and use demo results
        dataset_folder: Dataset directory containing source media files
        output_file: JSON file to write annotation results to
    """
    if test_mode:
        print("=" * 50)
        print("TEST MODE: Processing only one AUDIO-VISUAL pair")
        print("=" * 50)
        if skip_api:
            print("Gemini API calls disabled - using cascade demo results\n")

    dataset_folder = Path(dataset_folder)
    output_file = Path(output_file)
    results = _load_existing_results(output_file)

    if not dataset_folder.exists():
        print("ERROR: Dataset folder not found!")
        print(f"Expected to find media files in: {dataset_folder}")
        return results

    print(f"Dataset directory listing for {dataset_folder}:")
    print(os.listdir(dataset_folder))

    print("Discovering AUDIO with-audio media...")
    audio_pairs = preprocess_audio(dataset_folder)
    print(f"Found {len(audio_pairs)} with-audio media files\n")

    if len(audio_pairs) == 0:
        print("ERROR: No with-audio media files found in dataset folder!")
        print(f"Expected to find files ending in 'with_audio' in: {dataset_folder}")
        return results

    print("Discovering source RGB videos for HIA...")
    rgb_videos_dict = _discover_audio_rgb_videos(dataset_folder)
    print(f"Found {len(rgb_videos_dict)} RGB source video pairs\n")

    if test_mode:
        pairs_to_process = list(audio_pairs.items())[test_pair_index:test_pair_index + 1]
        print(f"Processing audio-visual pair {test_pair_index} of {len(audio_pairs)}:")
    else:
        pairs_to_process = list(audio_pairs.items())

    if not pairs_to_process:
        print("ERROR: No audio-visual pairs to process.")
        return results

    complete_count = 0
    partial_count = 0
    empty_count = 0
    pending_pairs = []
    for pair_key, audio_path in pairs_to_process:
        existing_entry = results.get(pair_key)
        if _audio_entry_complete(existing_entry):
            complete_count += 1
            continue

        existing_annotations = existing_entry.get("annotations", {}) if isinstance(existing_entry, dict) else {}
        if _audio_annotations_have_content(existing_annotations):
            partial_count += 1
        else:
            empty_count += 1
        pending_pairs.append((pair_key, audio_path))

    print(
        "Resume scan for AUDIO: "
        f"{complete_count} complete skipped, {partial_count} partial retry, "
        f"{empty_count} empty/missing retry."
    )

    if not pending_pairs:
        print("No incomplete AUDIO pairs remain. Skipping Gemini client creation and API calls.")
        print(f"Existing Audio QA results kept at: {output_file}")
        return results

    client = None
    if not skip_api:
        client = create_gemini_client()

    print(
        f"Processing {len(pending_pairs)} audio-visual pairs with up to 3 concurrent tasks and 4-second spacing..."
    )

    selected_audio_pairs = dict(pending_pairs)
    selected_rgb_videos = {}
    for pair_key in selected_audio_pairs.keys():
        rgb_videos = rgb_videos_dict.get(pair_key, {"day": None, "night": None}).copy()
        day_rgb_file = rgb_videos.get("day")

        if not skip_api and day_rgb_file:
            try:
                rgb_videos["day_frames"] = _load_or_extract_audio_hia_frames(
                    day_rgb_file,
                    dataset_folder,
                )
            except Exception as e:
                print(f"WARNING: Could not prepare HIA RGB frames for {pair_key}: {e}")
                rgb_videos["day_frames"] = []
        else:
            rgb_videos["day_frames"] = []

        selected_rgb_videos[pair_key] = rgb_videos

    def checkpoint_audio_pair(pair_key: str, file_results: Dict) -> None:
        audio_path = selected_audio_pairs.get(pair_key)
        rgb_videos = selected_rgb_videos.get(pair_key, {})
        day_rgb_file = rgb_videos.get("day") if isinstance(rgb_videos, dict) else None
        night_rgb_file = rgb_videos.get("night") if isinstance(rgb_videos, dict) else None

        results[pair_key] = {
            "audio_file": str(audio_path) if audio_path else None,
            "day_rgb_file": str(day_rgb_file) if day_rgb_file else None,
            "night_rgb_file": str(night_rgb_file) if night_rgb_file else None,
            "annotations": format_audio_annotations(file_results),
        }
        _write_results(output_file, results)
        print(f"Checkpoint saved: {pair_key}")

    batch_results = asyncio.run(
        run_audio_parallel_pipeline(
            client,
            selected_audio_pairs,
            selected_rgb_videos,
            max_concurrent=3,
            delay_between_pairs=4,
            skip_api=skip_api,
            on_pair_complete=checkpoint_audio_pair,
        )
    )

    for pair_key, audio_path in pending_pairs:
        if not audio_path:
            print(f"Skipping {pair_key} - no audio file found")
            continue

        file_results = batch_results.get(pair_key)
        if file_results is None:
            print(f"WARNING: No cascade output for pair {pair_key}. Using empty cascade result.")
            file_results = {"hia": "", "caption": "", "qa_pairs": []}

        checkpoint_audio_pair(pair_key, file_results)
        print(f"Done: {pair_key}")

    _write_results(output_file, results)

    print(f"\n" + "=" * 50)
    print(f"Audio QA results saved to: {output_file}")
    if test_mode:
        print("TEST MODE COMPLETE")
    print("=" * 50)
    return results
