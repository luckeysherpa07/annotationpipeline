"""Segmented QA quality menu actions."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from annotation_feature.cli.actions.aligned_choices import (
    SEGMENTED_QA_QUALITY_CLEAN,
    SEGMENTED_QA_QUALITY_EVALUATE,
    SEGMENTED_QA_QUALITY_LLM_EVAL,
)
from annotation_feature.cli.menu import MenuAction
from annotation_feature.qa_quality import (
    clean_segmented_qa_dataset,
    evaluate_segmented_qa,
    run_segmented_qa_llm_evaluation,
)


def _print_header(title: str) -> None:
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


def build_segmented_qa_quality_actions(
    confirm: Callable[[str], bool],
    output_dir: Path | str = "outputs",
) -> dict[str, MenuAction]:
    """Build menu choices for segmented QA quality evaluation."""
    output_dir = Path(output_dir)

    def run_evaluation() -> None:
        input_path = Path("segmented_normalized_evidence_units.json")
        _print_header("Running: evaluate segmented QA quality")
        print(f"Reads {input_path}.")
        print("Writes segmented QA quality report, item CSV, split exports, and cleaned QA exports.")
        print("Preserves segment bounds, day/night side, task label, modality, and source media.")
        print("This is rule-based and does not call Gemini.")
        print("-" * 60)
        if confirm("Continue? (yes/no): "):
            outputs = evaluate_segmented_qa(input_path=input_path, output_dir=output_dir)
            for label, path in outputs.items():
                print(f"{label}: {path}")
        else:
            print("Cancelled.")

    def run_llm_evaluation() -> None:
        input_path = output_dir / "segmented_qa_cleaned_items.json"
        output_json = output_dir / "segmented_qa_llm_eval_results.json"
        output_csv = output_dir / "segmented_qa_llm_eval_items.csv"
        _print_header("Running: LLM-assisted segmented QA evaluation")
        print(f"Reads {input_path}.")
        print(f"Writes {output_json} and {output_csv}.")
        print("Gemini checks caption support, modality fit, and semantic-segment consistency.")
        print("This supports resume/checkpoint.")
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
            outputs = run_segmented_qa_llm_evaluation(
                input_path=input_path,
                output_json=output_json,
                output_csv=output_csv,
                batch_size=batch_size,
                max_items=max_items,
            )
            for label, path in outputs.items():
                print(f"{label}: {path}")
        else:
            print("Cancelled.")

    def run_cleaner() -> None:
        input_path = output_dir / "segmented_qa_llm_eval_results.json"
        output_path = output_dir / "segmented_qa_valid_items.json"
        _print_header("Running: clean segmented QA dataset")
        print(f"Reads {input_path}.")
        print(f"Writes {output_path}.")
        print("Keeps only pass/low-risk, caption-supported, modality-appropriate, segment-consistent QA.")
        print("This is rule-based and does not call Gemini.")
        print("-" * 60)
        if confirm("Continue? (yes/no): "):
            result = clean_segmented_qa_dataset(
                input_path=input_path,
                output_path=output_path,
            )
            summary = result["summary"]
            print(
                "Cleaned segmented QA dataset: "
                f"{summary['total_valid']} valid, {summary['total_removed']} removed, "
                f"{summary['total_input']} input."
            )
            print(f"valid_json: {output_path}")
        else:
            print("Cancelled.")

    quality_action = MenuAction(
        action_id="segmented.qa_quality.evaluate",
        title="Evaluate segmented QA quality",
        section="SEGMENTED QA QUALITY",
        handler=run_evaluation,
    )
    llm_action = MenuAction(
        action_id="segmented.qa_quality.llm_eval",
        title="Run LLM-assisted segmented QA evaluation",
        section="SEGMENTED QA QUALITY",
        handler=run_llm_evaluation,
    )
    clean_action = MenuAction(
        action_id="segmented.qa_quality.clean",
        title="Clean segmented QA dataset",
        section="SEGMENTED QA QUALITY",
        handler=run_cleaner,
    )
    return {
        SEGMENTED_QA_QUALITY_EVALUATE: quality_action,
        SEGMENTED_QA_QUALITY_LLM_EVAL: llm_action,
        SEGMENTED_QA_QUALITY_CLEAN: clean_action,
        "segmented.qa_quality.evaluate": quality_action,
        "segmented.qa_quality.llm_eval": llm_action,
        "segmented.qa_quality.clean": clean_action,
    }
