"""Strict cleaner for LLM-evaluated segmented QA items."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .cleaner import _load_eval_items


DEFAULT_INPUT_PATH = Path("outputs/segmented_qa_llm_eval_results.json")
DEFAULT_OUTPUT_PATH = Path("outputs/segmented_qa_valid_items.json")
REQUIRED_TEXT_FIELDS = (
    "qa_id",
    "segment_id",
    "side",
    "modality",
    "section",
    "source_media",
    "question",
    "answer",
    "caption",
)
OUTPUT_FIELDS = (
    "qa_id",
    "segment_id",
    "source_prefix",
    "side",
    "task_label",
    "start_seconds",
    "end_seconds",
    "start_timestamp",
    "end_timestamp",
    "segment_confidence",
    "source_files",
    "source_media",
    "modality",
    "section",
    "question",
    "answer",
    "caption",
    "timestamp",
    "confidence",
)


def _is_valid(item: dict[str, Any]) -> bool:
    return (
        item.get("status") == "pass"
        and item.get("answerable_from_caption") is True
        and item.get("answer_matches_question") is True
        and item.get("caption_supports_answer") is True
        and item.get("modality_appropriate") is True
        and item.get("single_question") is True
        and item.get("segment_consistent") is True
        and item.get("hallucination_risk") == "low"
        and all(str(item.get(field, "")).strip() for field in REQUIRED_TEXT_FIELDS)
    )


def _project(item: dict[str, Any]) -> dict[str, Any]:
    return {field: item.get(field) for field in OUTPUT_FIELDS}


def clean_segmented_qa_dataset(
    input_path: Path | str = DEFAULT_INPUT_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    """Keep strict-valid segmented QA while preserving source media metadata."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    items = _load_eval_items(input_path)
    valid_qa = [_project(item) for item in items if _is_valid(item)]
    output = {
        "valid_qa": valid_qa,
        "summary": {
            "input_path": input_path.as_posix(),
            "total_input": len(items),
            "total_valid": len(valid_qa),
            "total_removed": len(items) - len(valid_qa),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()
    result = clean_segmented_qa_dataset(args.input, args.output)
    summary = result["summary"]
    print(
        f"Cleaned segmented QA dataset: {summary['total_valid']} valid, "
        f"{summary['total_removed']} removed, {summary['total_input']} input."
    )


if __name__ == "__main__":
    main()
