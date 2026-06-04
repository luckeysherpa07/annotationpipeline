"""Aligned QA quality menu actions."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from annotation_feature.cli.menu import MenuAction
from annotation_feature.qa_quality import evaluate_aligned_qa


def _print_header(title: str) -> None:
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


def build_aligned_qa_quality_actions(
    confirm: Callable[[str], bool],
    output_dir: Path | str = "outputs",
) -> dict[str, MenuAction]:
    """Build menu choices for aligned QA quality evaluation."""

    def run_evaluation() -> None:
        _print_header("Running: evaluate aligned QA quality")
        print("Reads qa_pairs/aligned/*.json.")
        print("Writes aligned QA quality report, item CSV, split-item exports, and cleaned QA exports.")
        print("This is rule-based and does not call Gemini.")
        print("-" * 60)
        if confirm("Continue? (yes/no): "):
            outputs = evaluate_aligned_qa(output_dir=output_dir)
            for label, path in outputs.items():
                print(f"{label}: {path}")
        else:
            print("Cancelled.")

    action = MenuAction(
        action_id="aligned.qa_quality.evaluate",
        title="Evaluate aligned QA quality",
        section="ALIGNED QA QUALITY",
        handler=run_evaluation,
    )
    return {
        "64": action,
        "aligned.qa_quality.evaluate": action,
    }
