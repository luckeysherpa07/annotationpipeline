"""Aligned IR menu actions."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from annotation_feature.cli.menu import MenuAction
from annotation_feature.pipeline import run_ir, run_ir_missing_section_repair


def _print_header(title: str) -> None:
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


def build_aligned_ir_actions(
    output_file: Path,
    confirm: Callable[[str], bool],
) -> dict[str, MenuAction]:
    """Build numeric menu choices for aligned IR actions."""

    def run_test() -> None:
        _print_header("Running: aligned IR batch pipeline on 1 segment pair (real Gemini API calls)")
        print("WARNING: This will use Gemini API quota!")
        print("Reads IR videos from aligned_dataset/.")
        print("Uses aligned_dataset/.frames_cache_ir for extracted frames.")
        print(f"Writes {output_file}.")
        print("-" * 60)
        if confirm("Continue? (yes/no): "):
            run_ir(
                test_mode=True,
                skip_api=False,
                dataset_folder="aligned_dataset",
                output_file=str(output_file),
            )
        else:
            print("Cancelled.")

    def run_production() -> None:
        _print_header("Running: aligned IR batch pipeline on all segment pairs (production)")
        print("WARNING: This will use Gemini API quota for each aligned segment pair!")
        print("Reads IR videos from aligned_dataset/.")
        print("Uses aligned_dataset/.frames_cache_ir for extracted frames.")
        print(f"Writes {output_file}.")
        print("Quota-friendly execution: 1 pair at a time, 70-second spacing")
        print("-" * 60)
        if confirm("Continue? (yes/no): "):
            run_ir(
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
        _print_header("Running: repair aligned IR missing sections")
        print("WARNING: This will use Gemini API quota for each partial aligned IR pair!")
        print("Reads IR videos from aligned_dataset/.")
        print("Uses aligned_dataset/.frames_cache_ir for extracted frames.")
        print(f"Repairs missing sections in {output_file}.")
        print("Quota-friendly execution: 1 pair at a time, 70-second spacing")
        print("-" * 60)
        if confirm("Continue? (yes/no): "):
            run_ir_missing_section_repair(
                dataset_folder="aligned_dataset",
                output_file=str(output_file),
                skip_api=False,
                max_concurrent=1,
                delay_between_pairs=70,
            )
        else:
            print("Cancelled.")

    actions = {
        "41": MenuAction(
            action_id="aligned.ir.test",
            title="Test aligned IR batch pipeline on 1 segment pair",
            section="ALIGNED IR PIPELINE",
            handler=run_test,
        ),
        "42": MenuAction(
            action_id="aligned.ir.run",
            title="Run aligned IR batch pipeline on all segment pairs",
            section="ALIGNED IR PIPELINE",
            handler=run_production,
        ),
        "42r": MenuAction(
            action_id="aligned.ir.repair",
            title="Repair aligned IR missing sections",
            section="ALIGNED IR PIPELINE",
            handler=run_missing_section_repair,
        ),
    }
    actions["aligned.ir.repair"] = actions["42r"]
    return actions
