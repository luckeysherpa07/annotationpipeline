"""Strict cleaner for LLM-evaluated aligned QA items."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT_PATH = Path("outputs/aligned_qa_llm_eval_results.json")
DEFAULT_OUTPUT_PATH = Path("outputs/aligned_qa_valid_items.json")
REQUIRED_OUTPUT_FIELDS = ("qa_id", "modality", "section", "pair_key", "question", "answer", "caption")


def _load_eval_items(input_path: Path) -> list[dict[str, Any]]:
    with open(input_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    raw_items = payload.get("items") if isinstance(payload, dict) else None
    if isinstance(raw_items, dict):
        return [item for item in raw_items.values() if isinstance(item, dict)]
    if isinstance(raw_items, list):
        return [item for item in raw_items if isinstance(item, dict)]
    return []


def _has_required_output_fields(item: dict[str, Any]) -> bool:
    return all(str(item.get(field, "")).strip() for field in REQUIRED_OUTPUT_FIELDS)


def _is_valid_qa_item(item: dict[str, Any]) -> bool:
    return (
        item.get("status") == "pass"
        and item.get("answerable_from_caption") is True
        and item.get("answer_matches_question") is True
        and item.get("caption_supports_answer") is True
        and item.get("modality_appropriate") is True
        and item.get("single_question") is True
        and item.get("hallucination_risk") == "low"
        and _has_required_output_fields(item)
    )


def _project_valid_qa_item(item: dict[str, Any]) -> dict[str, str]:
    return {
        field: str(item.get(field, "")).strip()
        for field in REQUIRED_OUTPUT_FIELDS
    }


def clean_aligned_qa_dataset(
    input_path: Path | str = DEFAULT_INPUT_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    """Keep only strict-valid QA items from LLM-evaluated aligned QA results."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    items = _load_eval_items(input_path)
    valid_qa = [
        _project_valid_qa_item(item)
        for item in items
        if _is_valid_qa_item(item)
    ]
    output = {
        "valid_qa": valid_qa,
        "summary": {
            "total_input": len(items),
            "total_valid": len(valid_qa),
            "total_removed": len(items) - len(valid_qa),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
    return output


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = clean_aligned_qa_dataset(args.input, args.output)
    summary = result["summary"]
    print(
        "Cleaned aligned QA dataset: "
        f"{summary['total_valid']} valid, {summary['total_removed']} removed, "
        f"{summary['total_input']} input."
    )


if __name__ == "__main__":
    main()
