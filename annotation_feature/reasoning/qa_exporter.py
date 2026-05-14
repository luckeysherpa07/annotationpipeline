"""Export grouped reasoning evidence into split question-answer pairs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .groups import MODALITIES, REASONING_GROUPS


NUMBERED_ITEM_RE = re.compile(r"(?:^|\s)(\d+)[.)]\s+")


def _as_text(value: Any) -> str:
    """Return value as text only when it is already a string."""
    return value if isinstance(value, str) else ""


def split_numbered_text(text: str) -> list[str]:
    """Split numbered text into items, trimming only numbering and whitespace."""
    text = _as_text(text).strip()
    if not text:
        return []

    matches = list(NUMBERED_ITEM_RE.finditer(text))
    if not matches:
        return [text]

    items: list[str] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        item = text[start:end].strip()
        if item:
            items.append(item)

    return items


def split_qa_pairs(question: str, answer: str) -> list[dict[str, str]]:
    """Split a unit's question and answer strings into paired Q/A records."""
    question_items = split_numbered_text(question)
    answer_items = split_numbered_text(answer)

    if not question_items and not answer_items:
        return []

    pair_count = max(len(question_items), len(answer_items), 1)
    question_fallback = _as_text(question).strip()
    answer_fallback = _as_text(answer).strip()
    pairs: list[dict[str, str]] = []

    for index in range(pair_count):
        question_text = (
            question_items[index]
            if index < len(question_items)
            else question_fallback
        )
        answer_text = (
            answer_items[index]
            if index < len(answer_items)
            else answer_fallback
        )

        if question_text or answer_text:
            pairs.append(
                {
                    "question": question_text,
                    "answer": answer_text,
                }
            )

    return pairs


def create_empty_grouped_qa() -> dict[str, list]:
    """Create an empty modality container for one reasoning group."""
    return {modality: [] for modality in MODALITIES}


def export_grouped_qa_pairs(grouped_data: dict) -> dict:
    """Convert grouped evidence data into nested split Q/A pairs."""
    if not isinstance(grouped_data, dict):
        return {}

    exported: dict[str, dict[str, Any]] = {}
    for sample_id in sorted(grouped_data):
        sample_data = grouped_data[sample_id]
        if not isinstance(sample_data, dict):
            sample_data = {}

        sample_output = {
            "source_keys": sample_data.get("source_keys", {})
            if isinstance(sample_data.get("source_keys", {}), dict)
            else {},
            "reasoning_groups": {
                group: create_empty_grouped_qa()
                for group in REASONING_GROUPS
            },
        }

        reasoning_groups = sample_data.get("reasoning_groups", {})
        if not isinstance(reasoning_groups, dict):
            exported[sample_id] = sample_output
            continue

        for group in REASONING_GROUPS:
            group_data = reasoning_groups.get(group, {})
            if not isinstance(group_data, dict):
                continue

            units = group_data.get("units", [])
            if not isinstance(units, list):
                continue

            for source_unit_index, unit in enumerate(units):
                if not isinstance(unit, dict):
                    continue

                modality = _as_text(unit.get("modality", ""))
                if modality not in MODALITIES:
                    continue

                pairs = split_qa_pairs(
                    _as_text(unit.get("question", "")),
                    _as_text(unit.get("answer", "")),
                )
                for pair_index, pair in enumerate(pairs, start=1):
                    sample_output["reasoning_groups"][group][modality].append(
                        {
                            "section": _as_text(unit.get("section", "")),
                            "question": pair["question"],
                            "answer": pair["answer"],
                            "caption": _as_text(unit.get("caption", "")),
                            "confidence": unit.get("confidence"),
                            "timestamp": unit.get("timestamp"),
                            "source_unit_index": source_unit_index,
                            "pair_index": pair_index,
                        }
                    )

        exported[sample_id] = sample_output

    return exported


def load_json_file(path: str | Path) -> dict:
    """Load a JSON object from disk, returning an empty dict when missing."""
    json_path = Path(path)
    if not json_path.exists():
        print(f"WARNING: Missing grouped evidence file: {json_path}")
        return {}

    try:
        with open(json_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {json_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object in {json_path}")

    return data


def save_json_file(data: dict, path: str | Path) -> None:
    """Save a dictionary to JSON, creating parent folders as needed."""
    output_path = Path(path)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def run_export_grouped_qa(
    input_path: str | Path = "grouped_evidence.json",
    output_path: str | Path = "grouped_qa_pairs.json",
) -> dict:
    """Load grouped evidence, split Q/A pairs, and save them to JSON."""
    grouped_data = load_json_file(input_path)
    print(f"Loaded grouped evidence from: {Path(input_path)}")

    exported = export_grouped_qa_pairs(grouped_data)
    save_json_file(exported, output_path)

    total_pairs = 0
    print(f"Exported samples: {len(exported)}")
    for sample_id in sorted(exported):
        sample_total = 0
        for group_data in exported[sample_id]["reasoning_groups"].values():
            sample_total += sum(len(records) for records in group_data.values())
        total_pairs += sample_total
        print(f"  {sample_id}: {sample_total} Q/A pairs")
    print(f"Total Q/A pairs: {total_pairs}")
    print(f"Grouped Q/A pairs written to: {Path(output_path)}")

    return exported
