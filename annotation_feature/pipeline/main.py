from pathlib import Path
import asyncio
import copy
import json
import os
import sys
from typing import Dict

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from annotation_feature.demo_result import DEMO_RESULT
from annotation_feature.video_preprocessor import preprocess_videos
from .client import create_gemini_client
from .utils import get_pair_key, video_extensions
from .qa_pipeline import run_parallel_pipeline
from .event_pipeline import run_event_parallel_pipeline


def run(
    test_mode: bool = False,
    test_pair_index: int = 0,
    skip_api: bool = False,
    dataset_folder: Path | str = "dataset",
):
    """
    Run the RGB annotation pipeline.

    Args:
        test_mode: If True, only process one video pair for testing
        test_pair_index: Which video pair to process in test mode (0 = first)
        skip_api: If True, skip Gemini API calls and use DEMO_RESULT instead
        dataset_folder: Dataset directory containing the source videos
    """
    if test_mode:
        print("=" * 50)
        print("TEST MODE: Processing only one RGB video pair")
        print("=" * 50)
        if skip_api:
            print("Gemini API calls disabled - using DEMO_RESULT data\n")

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

    available_pairs = {
        pair_key: frames
        for pair_key, frames in pairs_to_process
        if frames.get("night") or frames.get("day")
    }

    if not available_pairs:
        print("ERROR: No usable video frames found for selected pairs.")
        return results

    print(
        f"Processing {len(available_pairs)} batch pairs with up to 3 concurrent tasks and 4-second spacing..."
    )

    batch_results = asyncio.run(
        run_parallel_pipeline(
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
            print(f"WARNING: No batch output for pair {pair_key}. Falling back to DEMO_RESULT.")
            file_results = copy.deepcopy(DEMO_RESULT)

        night_file = None
        day_file = None
        for file in dataset_folder.rglob("*"):
            if not file.is_file() or file.suffix.lower() not in video_extensions:
                continue
            name = file.name.lower()
            if "rgb" not in name:
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
    output_file = Path("qa_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

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
):
    """
    Run the EVENT annotation pipeline.

    Args:
        test_mode: If True, only process one video pair for testing
        test_pair_index: Which video pair to process in test mode (0 = first)
        skip_api: If True, skip Gemini API calls and return empty captions
        dataset_folder: Dataset directory containing the source videos
    """
    if test_mode:
        print("=" * 50)
        print("TEST MODE: Processing only one EVENT video pair")
        print("=" * 50)
        if skip_api:
            print("Gemini API calls disabled - using demo captions\n")

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

    available_pairs = {
        pair_key: frames
        for pair_key, frames in pairs_to_process
        if frames.get("night") or frames.get("day")
    }

    if not available_pairs:
        print("ERROR: No usable video frames found for selected pairs.")
        return results

    print(
        f"Processing {len(available_pairs)} event pairs with up to 3 concurrent tasks and 4-second spacing..."
    )

    batch_results = asyncio.run(
        run_event_parallel_pipeline(
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
            print(f"WARNING: No batch output for pair {pair_key}. Using empty captions.")
            from prompts.event_prompts import EVENT_PROMPTS
            file_results = {anno_type: {"caption": ""} for anno_type in EVENT_PROMPTS.keys()}

        night_file = None
        day_file = None
        for file in dataset_folder.rglob("*"):
            if not file.is_file() or file.suffix.lower() not in video_extensions:
                continue
            name = file.name.lower()
            if "event" not in name:
                continue
            if get_pair_key(file) == pair_key:
                if "night" in name:
                    night_file = file
                elif "day" in name:
                    day_file = file

        results[pair_key] = {
            "night_file": str(night_file) if night_file else None,
            "day_file": str(day_file) if day_file else None,
            "captions": file_results,
        }
        print(f"✓ Done: {pair_key}")

    # Save results to JSON file at the project root
    output_file = Path("event_captions.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n" + "=" * 50)
    print(f"Event captions saved to: {output_file}")
    if test_mode:
        print("TEST MODE COMPLETE")
    print("=" * 50)
    return results


if __name__ == "__main__":
    run()
