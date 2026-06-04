"""Aligned Event menu actions."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from annotation_feature.cli.menu import MenuAction
from annotation_feature.pipeline import run_event, run_event_missing_section_repair


def _print_header(title: str) -> None:
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


def build_aligned_event_actions(
    output_file: Path,
    confirm: Callable[[str], bool],
) -> dict[str, MenuAction]:
    """Build numeric menu choices for aligned Event actions."""

    def run_test() -> None:
        _print_header("Running: aligned EVENT batch pipeline on 1 segment pair (real Gemini API calls)")
        print("WARNING: This will use Gemini API quota!")
        print("Reads EVENT videos from aligned_dataset/.")
        print("Uses aligned_dataset/.frames_cache_event for extracted frames.")
        print(f"Writes {output_file}.")
        print("-" * 60)
        if confirm("Continue? (yes/no): "):
            run_event(
                test_mode=True,
                skip_api=False,
                dataset_folder="aligned_dataset",
                output_file=str(output_file),
            )
        else:
            print("Cancelled.")

    def run_production() -> None:
        _print_header("Running: aligned EVENT batch pipeline on all segment pairs (production)")
        print("WARNING: This will use Gemini API quota for each aligned segment pair!")
        print("Reads EVENT videos from aligned_dataset/.")
        print("Uses aligned_dataset/.frames_cache_event for extracted frames.")
        print(f"Writes {output_file}.")
        print("Quota-friendly execution: 1 pair at a time, 70-second spacing")
        print("-" * 60)
        if confirm("Continue? (yes/no): "):
            run_event(
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
        _print_header("Running: repair aligned EVENT missing sections")
        print("WARNING: This will use Gemini API quota for each partial aligned EVENT pair!")
        print("Reads EVENT videos from aligned_dataset/.")
        print("Uses aligned_dataset/.frames_cache_event for extracted frames.")
        print(f"Repairs missing sections in {output_file}.")
        print("Quota-friendly execution: 1 pair at a time, 70-second spacing")
        print("-" * 60)
        if confirm("Continue? (yes/no): "):
            run_event_missing_section_repair(
                dataset_folder="aligned_dataset",
                output_file=str(output_file),
                skip_api=False,
                max_concurrent=1,
                delay_between_pairs=70,
            )
        else:
            print("Cancelled.")

    actions = {
        "39": MenuAction(
            action_id="aligned.event.test",
            title="Test aligned EVENT batch pipeline on 1 segment pair",
            section="ALIGNED EVENT PIPELINE",
            handler=run_test,
        ),
        "40": MenuAction(
            action_id="aligned.event.run",
            title="Run aligned EVENT batch pipeline on all segment pairs",
            section="ALIGNED EVENT PIPELINE",
            handler=run_production,
        ),
        "40r": MenuAction(
            action_id="aligned.event.repair",
            title="Repair aligned EVENT missing sections",
            section="ALIGNED EVENT PIPELINE",
            handler=run_missing_section_repair,
        ),
    }
    actions["aligned.event.repair"] = actions["40r"]
    return actions
