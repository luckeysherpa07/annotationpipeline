"""Aligned Marigold depth menu actions."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from annotation_feature.cli.menu import MenuAction
from annotation_feature.pipeline import (
    run_aligned_marigold_depth_estimation,
    run_marigold_depth_missing_section_repair,
    run_marigold_depth_qa,
)


def _print_header(title: str) -> None:
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


def build_aligned_marigold_actions(
    output_file: Path,
    confirm: Callable[[str], bool],
) -> dict[str, MenuAction]:
    """Build menu choices for aligned Marigold depth actions."""

    def run_estimation() -> None:
        _print_header("Running: aligned Marigold depth estimation")
        print("Day depth maps are estimated from aligned RGB frames.")
        print("Night depth maps are estimated from aligned IR frames.")
        print("Writes aligned_dataset/.frames_cache_marigold.")
        print("-" * 60)
        if confirm("Continue? (yes/no): "):
            marigold_frames = run_aligned_marigold_depth_estimation(
                dataset_folder="aligned_dataset",
            )
            print(f"Resolved/generated aligned Marigold depth maps for {len(marigold_frames)} segment pair(s).")
        else:
            print("Cancelled.")

    def run_qa_only() -> None:
        _print_header("Running: aligned Marigold depth QA")
        print("WARNING: This will use Gemini API quota for each incomplete aligned Marigold pair!")
        print("Reads aligned_dataset/.frames_cache_marigold.")
        print(f"Writes {output_file}.")
        print("Quota-friendly QA execution: 1 pair at a time, 70-second spacing")
        print("-" * 60)
        if confirm("Continue? (yes/no): "):
            qa_results = run_marigold_depth_qa(
                test_mode=False,
                skip_api=False,
                dataset_folder="aligned_dataset",
                cache_subdir=".frames_cache_marigold",
                output_file=str(output_file),
                max_concurrent=1,
                delay_between_pairs=70,
            )
            print(f"Aligned Marigold depth QA results saved for {len(qa_results)} segment pair(s).")
        else:
            print("Cancelled.")

    def run_full() -> None:
        _print_header("Running: aligned Marigold depth estimation + QA")
        print("WARNING: QA will use Gemini API quota for each incomplete aligned Marigold pair!")
        print("Day depth maps are estimated from aligned RGB frames.")
        print("Night depth maps are estimated from aligned IR frames.")
        print(f"Writes aligned_dataset/.frames_cache_marigold and {output_file}.")
        print("Quota-friendly QA execution: 1 pair at a time, 70-second spacing")
        print("-" * 60)
        if confirm("Continue? (yes/no): "):
            marigold_frames = run_aligned_marigold_depth_estimation(
                dataset_folder="aligned_dataset",
            )
            print(f"Resolved/generated aligned Marigold depth maps for {len(marigold_frames)} segment pair(s).")
            qa_results = run_marigold_depth_qa(
                test_mode=False,
                skip_api=False,
                dataset_folder="aligned_dataset",
                cache_subdir=".frames_cache_marigold",
                output_file=str(output_file),
                max_concurrent=1,
                delay_between_pairs=70,
            )
            print(f"Aligned Marigold depth QA results saved for {len(qa_results)} segment pair(s).")
        else:
            print("Cancelled.")

    def run_missing_section_repair() -> None:
        _print_header("Running: repair aligned Marigold depth missing sections")
        print("WARNING: This will use Gemini API quota for each partial aligned Marigold depth pair!")
        print("Reads aligned_dataset/.frames_cache_marigold.")
        print(f"Repairs missing sections in {output_file}.")
        print("Quota-friendly execution: 1 pair at a time, 70-second spacing")
        print("-" * 60)
        if confirm("Continue? (yes/no): "):
            run_marigold_depth_missing_section_repair(
                dataset_folder="aligned_dataset",
                cache_subdir=".frames_cache_marigold",
                output_file=str(output_file),
                skip_api=False,
                max_concurrent=1,
                delay_between_pairs=70,
            )
        else:
            print("Cancelled.")

    actions = {
        "47": MenuAction(
            action_id="aligned.marigold_depth.run",
            title="Run aligned Marigold depth estimation + QA",
            section="ALIGNED MARIGOLD DEPTH PIPELINE",
            handler=run_full,
        ),
        "47e": MenuAction(
            action_id="aligned.marigold_depth.estimate",
            title="Run aligned Marigold depth estimation only",
            section="ALIGNED MARIGOLD DEPTH PIPELINE",
            handler=run_estimation,
        ),
        "47q": MenuAction(
            action_id="aligned.marigold_depth.qa",
            title="Run aligned Marigold depth QA only",
            section="ALIGNED MARIGOLD DEPTH PIPELINE",
            handler=run_qa_only,
        ),
        "47r": MenuAction(
            action_id="aligned.marigold_depth.repair",
            title="Repair aligned Marigold depth missing sections",
            section="ALIGNED MARIGOLD DEPTH PIPELINE",
            handler=run_missing_section_repair,
        ),
    }
    actions["aligned.marigold_depth.run"] = actions["47"]
    actions["aligned.marigold_depth.estimate"] = actions["47e"]
    actions["aligned.marigold_depth.qa"] = actions["47q"]
    actions["aligned.marigold_depth.repair"] = actions["47r"]
    return actions
