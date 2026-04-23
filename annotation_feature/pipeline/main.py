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
from annotation_feature.audio_preprocessor import preprocess_audio
from .client import create_gemini_client
from .utils import get_pair_key, video_extensions, audio_extensions
from .modalities.rgb import run_parallel_pipeline
from .modalities.event import run_event_parallel_pipeline
from .modalities.depth import run_depth_parallel_pipeline
from .modalities.ir import run_ir_parallel_pipeline
from .modalities.audio import run_parallel_pipeline as run_audio_parallel_pipeline


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
    output_file = Path("rgb_qa_results.json")
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
            print(f"WARNING: No batch output for pair {pair_key}. Using empty results.")
            from prompts.event_prompts import EVENT_PROMPTS
            file_results = {anno_type: {"caption": "", "question": "", "answer": ""} for anno_type in EVENT_PROMPTS.keys()}

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
            "annotations": file_results,
        }
        print(f"✓ Done: {pair_key}")

    # Save results to JSON file at the project root
    output_file = Path("event_qa_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

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
            if "depth" not in name:
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


def run_ir(
    test_mode: bool = False,
    test_pair_index: int = 0,
    skip_api: bool = False,
    dataset_folder: Path | str = "dataset",
):
    """
    Run the IR annotation pipeline.

    Args:
        test_mode: If True, only process one video pair for testing
        test_pair_index: Which video pair to process in test mode (0 = first)
        skip_api: If True, skip Gemini API calls and return demo results
        dataset_folder: Dataset directory containing the source videos
    """
    if test_mode:
        print("=" * 50)
        print("TEST MODE: Processing only one IR video pair")
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

    available_pairs = {
        pair_key: frames
        for pair_key, frames in pairs_to_process
        if frames.get("night") or frames.get("day")
    }

    if not available_pairs:
        print("ERROR: No usable video frames found for selected pairs.")
        return results

    print(
        f"Processing {len(available_pairs)} IR pairs with up to 3 concurrent tasks and 4-second spacing..."
    )

    batch_results = asyncio.run(
        run_ir_parallel_pipeline(
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
            from prompts.ir_prompts import IR_PROMPTS
            file_results = {anno_type: {"caption": "", "question": "", "answer": ""} for anno_type in IR_PROMPTS.keys()}

        night_file = None
        day_file = None
        for file in dataset_folder.rglob("*"):
            if not file.is_file() or file.suffix.lower() not in video_extensions:
                continue
            name = file.name.lower()
            if "ir" not in name:
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
        print(f"âœ“ Done: {pair_key}")

    output_file = Path("ir_qa_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

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
):
    """
    Run the AUDIO annotation pipeline.

    Args:
        test_mode: If True, only process one audio pair for testing
        test_pair_index: Which audio pair to process in test mode (0 = first)
        skip_api: If True, skip Gemini API calls and use demo results
        dataset_folder: Dataset directory containing the source audio files
    """
    if test_mode:
        print("=" * 50)
        print("TEST MODE: Processing only one AUDIO file")
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
        print(f"Expected to find audio files in: {dataset_folder}")
        return results

    print(f"Dataset directory listing for {dataset_folder}:")
    print(os.listdir(dataset_folder))

    # Preprocess audio files - only night audio (day/night don't affect audio)
    print("Preprocessing AUDIO files...")
    audio_pairs = preprocess_audio(dataset_folder)
    print(f"Found {len(audio_pairs)} audio files\n")

    if len(audio_pairs) == 0:
        print("ERROR: No audio files found in dataset folder!")
        print(f"Expected to find .m4a, .mp3, .wav, .aac files with '_night' suffix in: {dataset_folder}")
        return results

    # In test mode, only process one audio file
    if test_mode:
        pairs_to_process = list(audio_pairs.items())[test_pair_index:test_pair_index + 1]
        print(f"Processing audio file {test_pair_index} of {len(audio_pairs)}:")
    else:
        pairs_to_process = list(audio_pairs.items())

    if not pairs_to_process:
        print("ERROR: No audio files to process.")
        return results

    print(
        f"Processing {len(pairs_to_process)} audio files with up to 3 concurrent tasks and 4-second spacing..."
    )

    batch_results = asyncio.run(
        run_audio_parallel_pipeline(
            client,
            dict(pairs_to_process),
            max_concurrent=3,
            delay_between_pairs=4,
            skip_api=skip_api,
        )
    )

    for pair_key, audio_path in pairs_to_process:
        if not audio_path:
            print(f"Skipping {pair_key} - no audio file found")
            continue

        file_results = batch_results.get(pair_key)
        if file_results is None:
            print(f"WARNING: No batch output for pair {pair_key}. Using demo results.")
            from prompts.audio_prompts import AUDIO_PROMPTS
            file_results = {anno_type: {"caption": "", "question": "", "answer": ""} for anno_type in AUDIO_PROMPTS.keys()}

        results[pair_key] = {
            "audio_file": str(audio_path),
            "annotations": file_results,
        }
        print(f"✓ Done: {pair_key}")

    # Save results to JSON file at the project root
    output_file = Path("audio_qa_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n" + "=" * 50)
    print(f"Audio QA results saved to: {output_file}")
    if test_mode:
        print("TEST MODE COMPLETE")
    print("=" * 50)
    return results
