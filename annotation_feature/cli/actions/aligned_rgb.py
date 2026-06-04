"""Aligned RGB menu actions."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from annotation_feature.cli.menu import MenuAction
from annotation_feature.pipeline import run, run_rgb_missing_section_repair


def _print_header(title: str) -> None:
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


def build_aligned_rgb_actions(
    output_file: Path,
    confirm: Callable[[str], bool],
) -> dict[str, MenuAction]:
    """Build numeric menu choices for aligned RGB actions."""

    def run_test() -> None:
        _print_header("Running: aligned RGB batch pipeline on 1 segment pair (real Gemini API calls)")
        print("WARNING: This will use Gemini API quota!")
        print("Reads RGB videos from aligned_dataset/.")
        print("Uses aligned_dataset/.frames_cache for extracted frames.")
        print(f"Writes {output_file}.")
        print("-" * 60)
        if confirm("Continue? (yes/no): "):
            run(
                test_mode=True,
                skip_api=False,
                dataset_folder="aligned_dataset",
                output_file=str(output_file),
            )
        else:
            print("Cancelled.")

    def run_production() -> None:
        _print_header("Running: aligned RGB batch pipeline on all segment pairs (production)")
        print("WARNING: This will use Gemini API quota for each aligned segment pair!")
        print("Reads RGB videos from aligned_dataset/.")
        print("Uses aligned_dataset/.frames_cache for extracted frames.")
        print(f"Writes {output_file}.")
        print("Quota-friendly execution: 1 pair at a time, 70-second spacing")
        print("-" * 60)
        if confirm("Continue? (yes/no): "):
            run(
                test_mode=False,
                skip_api=False,
                dataset_folder="aligned_dataset",
                output_file=str(output_file),
                max_concurrent=1,
                delay_between_pairs=70,
            )
        else:
            print("Cancelled.")

    def run_missing_section_repair() -> None:
        _print_header("Running: repair aligned RGB missing sections")
        print("WARNING: This will use Gemini API quota for each partial aligned RGB pair!")
        print("Reads RGB videos from aligned_dataset/.")
        print("Uses aligned_dataset/.frames_cache for extracted frames.")
        print(f"Repairs missing sections in {output_file}.")
        print("Quota-friendly execution: 1 pair at a time, 70-second spacing")
        print("-" * 60)
        if confirm("Continue? (yes/no): "):
            run_rgb_missing_section_repair(
                dataset_folder="aligned_dataset",
                output_file=str(output_file),
                skip_api=False,
                max_concurrent=1,
                delay_between_pairs=70,
            )
        else:
            print("Cancelled.")

    actions = {
        "37": MenuAction(
            action_id="aligned.rgb.test",
            title="Test aligned RGB batch pipeline on 1 segment pair",
            section="ALIGNED RGB PIPELINE",
            handler=run_test,
        ),
        "38": MenuAction(
            action_id="aligned.rgb.run",
            title="Run aligned RGB batch pipeline on all segment pairs",
            section="ALIGNED RGB PIPELINE",
            handler=run_production,
        ),
        "38r": MenuAction(
            action_id="aligned.rgb.repair",
            title="Repair aligned RGB missing sections",
            section="ALIGNED RGB PIPELINE",
            handler=run_missing_section_repair,
        ),
    }
    actions["aligned.rgb.repair"] = actions["38r"]
    return actions
