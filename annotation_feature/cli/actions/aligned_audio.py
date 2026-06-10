"""Aligned Audio menu actions."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from annotation_feature.cli.menu import MenuAction
from annotation_feature.cli.actions.aligned_choices import (
    ALIGNED_AUDIO_EXPORT_ALL,
    ALIGNED_AUDIO_EXPORT_ONE,
    ALIGNED_AUDIO_REPAIR,
    ALIGNED_AUDIO_RUN,
    ALIGNED_AUDIO_TEST,
)
from annotation_feature.pipeline import run_audio, run_audio_repair
from annotation_feature.temporal_alignment import (
    run_and_export_aligned_rgb_with_audio_segments,
    run_and_export_source_rgb_with_audio_segments_for_aligned_dataset,
)

ALIGNED_AUDIO_MAX_CONCURRENT = 1
ALIGNED_AUDIO_DELAY_BETWEEN_PAIRS = 15


def _print_header(title: str) -> None:
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


def build_aligned_audio_actions(
    output_file: Path,
    confirm: Callable[[str], bool],
) -> dict[str, MenuAction]:
    """Build numeric menu choices for aligned Audio actions."""

    def run_export_one() -> None:
        _print_header("Running: export source-aligned RGB-with-audio 30s segment videos for all segments of 1 pair")
        print("Reads the first split/sample RGB + .m4a pair group from dataset/.")
        print("Computes full-source RGB/audio cross-correlation before cutting 30s segments.")
        print("Writes *_rgb_with_audio.mp4 files into matching aligned_dataset/<split>/SegN/ folders.")
        print("Writes one full-source RGB/audio activity plot per source side.")
        print("-" * 60)
        if confirm("Continue? (yes/no): "):
            summary = run_and_export_source_rgb_with_audio_segments_for_aligned_dataset(
                source_dataset_folder="dataset",
                aligned_dataset_folder="aligned_dataset",
                summary_output_path="aligned_dataset/source_rgb_with_audio_one_pair_export_summary.json",
                overwrite=False,
                max_pair_groups=1,
            )
            exported = summary.get("exported_count", 0)
            skipped = summary.get("skipped_count", 0)
            reused = sum(1 for item in summary.get("exported", []) if item.get("status") == "reused")
            print(
                f"One-pair-group source-aligned RGB-with-audio export complete: "
                f"{exported} exported/reused ({reused} reused), {skipped} skipped."
            )
            print(f"Summary: {summary.get('summary_file')}")
        else:
            print("Cancelled.")

    def run_export_all() -> None:
        _print_header("Running: export source-aligned RGB-with-audio 30s segment videos for all pairs")
        print("Reads source RGB + .m4a pairs from dataset/.")
        print("Computes full-source RGB/audio cross-correlation before cutting 30s segments.")
        print("Writes *_rgb_with_audio.mp4 files into matching aligned_dataset/<split>/SegN/ folders.")
        print("Writes one full-source RGB/audio activity plot per source side.")
        print("Existing *_rgb_with_audio.mp4 files will be replaced.")
        print("-" * 60)
        if confirm("Continue? (yes/no): "):
            summary = run_and_export_source_rgb_with_audio_segments_for_aligned_dataset(
                source_dataset_folder="dataset",
                aligned_dataset_folder="aligned_dataset",
                summary_output_path="aligned_dataset/source_rgb_with_audio_export_summary.json",
                overwrite=True,
                max_pair_groups=None,
                verbose=True,
            )
            exported = summary.get("exported_count", 0)
            skipped = summary.get("skipped_count", 0)
            print(
                f"Source-aligned RGB-with-audio export complete: "
                f"{exported} exported/replaced, {skipped} skipped."
            )
            print(f"Summary: {summary.get('summary_file')}")
        else:
            print("Cancelled.")

    def run_test() -> None:
        _print_header("Running: aligned AUDIO pipeline on 1 segment pair (real Gemini API calls)")
        print("WARNING: This will use Gemini API quota!")
        print("Reads existing *_rgb_with_audio.mp4 files from aligned_dataset/.")
        print("Run option 43 first if the needed with-audio segment videos are missing.")
        print(f"Writes {output_file}.")
        print("-" * 60)
        if confirm("Continue? (yes/no): "):
            run_audio(
                test_mode=True,
                skip_api=False,
                dataset_folder="aligned_dataset",
                output_file=str(output_file),
                max_concurrent=ALIGNED_AUDIO_MAX_CONCURRENT,
                delay_between_pairs=ALIGNED_AUDIO_DELAY_BETWEEN_PAIRS,
            )
        else:
            print("Cancelled.")

    def run_production() -> None:
        _print_header("Running: aligned AUDIO pipeline on all segment pairs (production)")
        print("WARNING: This will use Gemini API quota for each aligned segment pair!")
        print("Reads generated *_rgb_with_audio.mp4 files from aligned_dataset/.")
        print("Ensures RGB/audio cross-correlation plots exist before QA.")
        print(f"Writes {output_file}.")
        print(
            "Quota-friendly execution: "
            f"{ALIGNED_AUDIO_MAX_CONCURRENT} pair at a time, "
            f"{ALIGNED_AUDIO_DELAY_BETWEEN_PAIRS}-second spacing"
        )
        print("-" * 60)
        if confirm("Continue? (yes/no): "):
            export_summary = run_and_export_aligned_rgb_with_audio_segments(
                dataset_folder="aligned_dataset",
                summary_output_path="aligned_dataset/aligned_rgb_with_audio_export_summary.json",
                overwrite=False,
            )
            print(
                "Prepared aligned RGB-with-audio files/plots: "
                f"{export_summary.get('exported_count', 0)} exported/reused, "
                f"{export_summary.get('skipped_count', 0)} skipped."
            )
            run_audio(
                test_mode=False,
                skip_api=False,
                dataset_folder="aligned_dataset",
                output_file=str(output_file),
                max_concurrent=ALIGNED_AUDIO_MAX_CONCURRENT,
                delay_between_pairs=ALIGNED_AUDIO_DELAY_BETWEEN_PAIRS,
            )
        else:
            print("Cancelled.")

    def run_repair() -> None:
        _print_header("Running: repair aligned AUDIO flagged/incomplete entries")
        print("WARNING: This will use Gemini API quota for each repairable aligned segment pair!")
        print("Targets incomplete entries plus repairable audio quality flags.")
        print("Skips entries that only lack HIA source and preserves them with quality flags.")
        print(f"Writes {output_file}.")
        print(
            "Quota-friendly execution: "
            f"{ALIGNED_AUDIO_MAX_CONCURRENT} pair at a time, "
            f"{ALIGNED_AUDIO_DELAY_BETWEEN_PAIRS}-second spacing"
        )
        print("-" * 60)
        if confirm("Continue? (yes/no): "):
            run_audio_repair(
                skip_api=False,
                dataset_folder="aligned_dataset",
                output_file=str(output_file),
                max_concurrent=ALIGNED_AUDIO_MAX_CONCURRENT,
                delay_between_pairs=ALIGNED_AUDIO_DELAY_BETWEEN_PAIRS,
            )
        else:
            print("Cancelled.")

    actions = {
        ALIGNED_AUDIO_EXPORT_ONE: MenuAction(
            action_id="aligned.audio.export_one",
            title="Export source-aligned RGB-with-audio 30s segment videos for all segments of 1 pair",
            section="ALIGNED AUDIO PIPELINE",
            handler=run_export_one,
        ),
        ALIGNED_AUDIO_EXPORT_ALL: MenuAction(
            action_id="aligned.audio.export_all",
            title="Export source-aligned RGB-with-audio 30s segment videos for all pairs",
            section="ALIGNED AUDIO PIPELINE",
            handler=run_export_all,
        ),
        ALIGNED_AUDIO_TEST: MenuAction(
            action_id="aligned.audio.test",
            title="Test aligned AUDIO pipeline on 1 segment pair",
            section="ALIGNED AUDIO PIPELINE",
            handler=run_test,
        ),
        ALIGNED_AUDIO_RUN: MenuAction(
            action_id="aligned.audio.run",
            title="Run aligned AUDIO pipeline on all segment pairs",
            section="ALIGNED AUDIO PIPELINE",
            handler=run_production,
        ),
        ALIGNED_AUDIO_REPAIR: MenuAction(
            action_id="aligned.audio.repair",
            title="Repair aligned AUDIO flagged/incomplete entries",
            section="ALIGNED AUDIO PIPELINE",
            handler=run_repair,
        ),
    }
    actions["aligned.audio.export_one"] = actions[ALIGNED_AUDIO_EXPORT_ONE]
    actions["aligned.audio.export_all"] = actions[ALIGNED_AUDIO_EXPORT_ALL]
    actions["aligned.audio.test"] = actions[ALIGNED_AUDIO_TEST]
    actions["aligned.audio.run"] = actions[ALIGNED_AUDIO_RUN]
    actions["aligned.audio.repair"] = actions[ALIGNED_AUDIO_REPAIR]
    return actions
