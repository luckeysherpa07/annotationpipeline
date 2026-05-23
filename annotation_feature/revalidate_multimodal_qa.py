"""Re-run local validation for generated implicit multimodal QA JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from annotation_feature.multimodal_qa_pipeline import _attach_validation, _build_distribution


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object in {path}")
    return data


def _save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def revalidate_multimodal_qa(input_path: Path | str, output_path: Path | str | None = None) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path is not None else input_path
    data = _load_json(input_path)
    qa_items = data.get("qa_items", [])
    if not isinstance(qa_items, list):
        raise ValueError("Input JSON must contain qa_items list")

    revalidated_items = []
    for item in qa_items:
        if isinstance(item, dict):
            revalidated_items.append(_attach_validation(item))
        else:
            revalidated_items.append(item)
    data["qa_items"] = revalidated_items

    valid_items = [item for item in revalidated_items if isinstance(item, dict)]
    passed = sum(
        1
        for item in valid_items
        if item.get("quality_control", {}).get("validation_status") == "passed"
    )
    metadata = data.setdefault("metadata", {})
    metadata["total_qa_items"] = len(valid_items)
    metadata["passed_validation"] = passed
    metadata["failed_validation"] = len(valid_items) - passed
    metadata["distribution"] = _build_distribution(valid_items)
    metadata["revalidated"] = True

    _save_json(data, output_path)
    return output_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input multimodal QA JSON.")
    parser.add_argument("--output", default=None, help="Output path. Defaults to overwriting --input.")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    output_path = revalidate_multimodal_qa(args.input, args.output)
    print(f"Revalidated multimodal QA file: {output_path}")


if __name__ == "__main__":
    main()
