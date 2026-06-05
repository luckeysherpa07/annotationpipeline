"""Aligned QA quality menu actions."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from annotation_feature.cli.menu import MenuAction
from annotation_feature.cli.actions.aligned_choices import (
    ALIGNED_QA_QUALITY_EVALUATE,
    ALIGNED_QA_QUALITY_LLM_EVAL,
)
from annotation_feature.qa_quality import evaluate_aligned_qa, run_aligned_qa_llm_evaluation


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

    def run_llm_evaluation() -> None:
        _print_header("Running: LLM-assisted aligned QA evaluation")
        print("Reads outputs/aligned_qa_cleaned_items.json.")
        print("Writes outputs/aligned_qa_llm_eval_results.json and outputs/aligned_qa_llm_eval_items.csv.")
        print("This calls Gemini and supports resume/checkpoint.")
        print("Default max items is 1000 for quota-controlled full evaluation; enter 0 to run all remaining items.")
        print("-" * 60)
        raw_limit = input("Max items to evaluate this run? (default 1000, 0 = all): ").strip()
        if not raw_limit:
            max_items = 1000
        else:
            try:
                parsed_limit = int(raw_limit)
            except ValueError:
                print("Invalid max items value.")
                return
            max_items = None if parsed_limit == 0 else max(0, parsed_limit)

        raw_batch_size = input("Batch size? (default 50): ").strip()
        if not raw_batch_size:
            batch_size = 50
        else:
            try:
                batch_size = max(1, int(raw_batch_size))
            except ValueError:
                print("Invalid batch size value.")
                return

        if confirm("Continue? (yes/no): "):
            outputs = run_aligned_qa_llm_evaluation(
                batch_size=batch_size,
                max_items=max_items,
            )
            for label, path in outputs.items():
                print(f"{label}: {path}")
        else:
            print("Cancelled.")

    quality_action = MenuAction(
        action_id="aligned.qa_quality.evaluate",
        title="Evaluate aligned QA quality",
        section="ALIGNED QA QUALITY",
        handler=run_evaluation,
    )
    llm_action = MenuAction(
        action_id="aligned.qa_quality.llm_eval",
        title="Run LLM-assisted aligned QA evaluation",
        section="ALIGNED QA QUALITY",
        handler=run_llm_evaluation,
    )
    return {
        ALIGNED_QA_QUALITY_EVALUATE: quality_action,
        ALIGNED_QA_QUALITY_LLM_EVAL: llm_action,
        "aligned.qa_quality.evaluate": quality_action,
        "aligned.qa_quality.llm_eval": llm_action,
    }
