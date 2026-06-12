"""Rule-based quality evaluation for normalized segmented QA items."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .aligned_evaluator import (
    DEPTH_FORBIDDEN_RE,
    FRAME_RE,
    IR_FORBIDDEN_RE,
    PROMPT_LEAK_RE,
    RGB_FORBIDDEN_RE,
    SHORT_ANSWER_RE,
    UNKNOWN_RE,
)
from .splitting import split_numbered_items, split_status


DEFAULT_INPUT_PATH = Path("segmented_normalized_evidence_units.json")
DEFAULT_OUTPUT_DIR = Path("outputs")


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level JSON object in {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            converted = dict(row)
            for key, value in converted.items():
                if isinstance(value, (dict, list)):
                    converted[key] = json.dumps(value, ensure_ascii=False)
            writer.writerow(converted)


def _clean_id_text(value: Any) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", str(value or "")).strip("_").lower()
    return cleaned or "unknown"


def _append_flag(flags: list[str], flag: str) -> None:
    if flag not in flags:
        flags.append(flag)


def _source_media_for_modality(segment: dict[str, Any], modality: str) -> str:
    source_files = segment.get("source_files", {})
    if not isinstance(source_files, dict):
        return ""
    direct = source_files.get(modality)
    if direct:
        return str(direct)
    videos = source_files.get("videos", {})
    if isinstance(videos, dict) and videos.get(modality):
        return str(videos[modality])
    if modality == "audio":
        return str(source_files.get("audio") or source_files.get("with_audio") or "")
    return ""


def _modality_mismatch(modality: str, question: str, answer: str) -> bool:
    text = f"{question} {answer}"
    if modality == "depth":
        return bool(DEPTH_FORBIDDEN_RE.search(text))
    if modality == "ir":
        return bool(IR_FORBIDDEN_RE.search(text))
    if modality == "rgb":
        return bool(RGB_FORBIDDEN_RE.search(text))
    return False


def _segment_metadata_flags(segment: dict[str, Any], modality: str) -> list[str]:
    flags: list[str] = []
    side = str(segment.get("side", "")).strip().lower()
    if side not in {"day", "night"}:
        flags.append("invalid_side")
    try:
        start_seconds = float(segment.get("start_seconds"))
        end_seconds = float(segment.get("end_seconds"))
        if start_seconds < 0 or end_seconds <= start_seconds:
            flags.append("invalid_segment_bounds")
    except (TypeError, ValueError):
        flags.append("invalid_segment_bounds")
    if not _source_media_for_modality(segment, modality):
        flags.append("missing_source_media")
    return flags


def _classify_unit(
    segment_id: str,
    segment: dict[str, Any],
    unit: dict[str, Any],
    duplicate_question: bool,
) -> dict[str, Any]:
    modality = str(unit.get("modality", "")).strip().lower()
    section = str(unit.get("section", "")).strip()
    caption = str(unit.get("caption") or unit.get("evidence") or "").strip()
    question = str(unit.get("question", "")).strip()
    answer = str(unit.get("answer", "")).strip()
    status, questions, answers = split_status(question, answer)
    flags = _segment_metadata_flags(segment, modality)

    if not modality or not section or not caption or not question or not answer:
        _append_flag(flags, "empty_field")
    if PROMPT_LEAK_RE.search(f"{caption} {question} {answer}"):
        _append_flag(flags, "prompt_leak")
    if status != "single":
        _append_flag(flags, "multi_question" if questions else "numbered_answer_list")
        if status == "count_mismatch":
            _append_flag(flags, "question_answer_count_mismatch")
    if duplicate_question:
        _append_flag(flags, "duplicate_question")
    if len(answer.split()) < 2 or SHORT_ANSWER_RE.match(answer):
        _append_flag(flags, "short_answer")
    if UNKNOWN_RE.search(f"{caption} {question} {answer}"):
        _append_flag(flags, "unknown_unclear")
    if _modality_mismatch(modality, question, answer):
        _append_flag(flags, "modality_mismatch")
    if FRAME_RE.search(question):
        _append_flag(flags, "frame_reference")

    hard_flags = {
        "empty_field",
        "prompt_leak",
        "invalid_side",
        "invalid_segment_bounds",
        "missing_source_media",
    }
    if any(flag in hard_flags for flag in flags):
        severity = "hard_fail"
        recommended_action = "regenerate_or_remove"
    elif "question_answer_count_mismatch" in flags:
        severity = "needs_transform"
        recommended_action = "review"
    elif status != "single":
        severity = "needs_transform"
        recommended_action = "split"
    elif flags:
        severity = "quality_warning"
        recommended_action = "review"
    else:
        severity = "pass"
        recommended_action = "keep"

    return {
        "segment_id": segment_id,
        "source_prefix": str(segment.get("source_prefix", "")),
        "side": str(segment.get("side", "")),
        "task_label": str(segment.get("task_label", "")),
        "start_seconds": segment.get("start_seconds"),
        "end_seconds": segment.get("end_seconds"),
        "start_timestamp": str(segment.get("start_timestamp", "")),
        "end_timestamp": str(segment.get("end_timestamp", "")),
        "segment_confidence": segment.get("confidence"),
        "source_files": segment.get("source_files", {}),
        "source_media": _source_media_for_modality(segment, modality),
        "modality": modality,
        "section": section,
        "caption": caption,
        "question": question,
        "answer": answer,
        "timestamp": unit.get("timestamp"),
        "confidence": unit.get("confidence"),
        "source_unit_index": unit.get("source_unit_index"),
        "pair_index": unit.get("pair_index"),
        "flags": flags,
        "severity": severity,
        "split_status": status,
        "question_count": len(questions) if questions else 1,
        "answer_count": len(answers) if answers else 1,
        "recommended_action": recommended_action,
    }


def _split_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        questions = split_numbered_items(item["question"])
        answers = split_numbered_items(item["answer"])
        if item["split_status"] == "aligned" and questions and answers:
            for index, (question, answer) in enumerate(zip(questions, answers), start=1):
                rows.append(
                    {
                        **item,
                        "qa_index": index,
                        "question": question,
                        "answer": answer,
                        "source_status": "split_from_numbered_list",
                    }
                )
        else:
            rows.append({**item, "qa_index": 1, "source_status": item["split_status"]})
    return rows


def _cleaned_rows(
    items: list[dict[str, Any]],
    input_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    for item in items:
        if item["severity"] == "hard_fail":
            excluded["hard_fail"] += 1
            continue

        questions = split_numbered_items(item["question"])
        answers = split_numbered_items(item["answer"])
        pairs: list[tuple[str, str]]
        transform: str
        if item["split_status"] == "aligned" and questions and answers:
            pairs = list(zip(questions, answers))
            transform = "split_multi_question"
        elif item["split_status"] == "single":
            pairs = [(item["question"], item["answer"])]
            transform = "none"
        else:
            excluded[f"unresolved_{item['split_status']}"] += 1
            continue

        for index, (question, answer) in enumerate(pairs, start=1):
            qa_id = "__".join(
                [
                    _clean_id_text(item["segment_id"]),
                    _clean_id_text(item["modality"]),
                    _clean_id_text(item["section"]),
                    str(item.get("source_unit_index", "unit")),
                    str(item.get("pair_index", "pair")),
                    f"{index:02d}",
                ]
            )
            rows.append(
                {
                    **{key: value for key, value in item.items() if key not in {"flags", "severity"}},
                    "qa_id": qa_id,
                    "question": question,
                    "answer": answer,
                    "source_question": item["question"],
                    "source_answer": item["answer"],
                    "source_severity": item["severity"],
                    "source_flags": list(item["flags"]),
                    "source_split_status": item["split_status"],
                    "transform": transform,
                    "split_index": index,
                    "split_count": len(pairs),
                }
            )

    metadata = {
        "source_file": input_path.as_posix(),
        "total_cleaned_items": len(rows),
        "excluded": dict(excluded),
        "transforms": dict(Counter(row["transform"] for row in rows)),
    }
    return rows, metadata


def evaluate_segmented_qa(
    input_path: Path | str = DEFAULT_INPUT_PATH,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Evaluate normalized segmented QA and export aligned-style quality artifacts."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    segments = _load_json(input_path)

    question_counts: Counter[str] = Counter()
    for segment in segments.values():
        if not isinstance(segment, dict):
            continue
        for unit in segment.get("evidence_units", []):
            if isinstance(unit, dict):
                question_counts[str(unit.get("question", "")).strip()] += 1

    items: list[dict[str, Any]] = []
    segment_counts: Counter[str] = Counter()
    for segment_id, segment in sorted(segments.items()):
        if not isinstance(segment, dict):
            segment_counts["invalid_segments"] += 1
            continue
        units = segment.get("evidence_units", [])
        if not isinstance(units, list):
            segment_counts["invalid_segments"] += 1
            continue
        segment_counts["segments"] += 1
        for unit in units:
            if not isinstance(unit, dict):
                continue
            question = str(unit.get("question", "")).strip()
            items.append(
                _classify_unit(
                    str(segment_id),
                    segment,
                    unit,
                    duplicate_question=bool(question and question_counts[question] > 1),
                )
            )

    split_rows = _split_rows(items)
    cleaned_rows, cleaned_metadata = _cleaned_rows(items, input_path)
    by_modality = Counter(item["modality"] for item in items)
    by_section = Counter(f"{item['modality']}:{item['section']}" for item in items)
    by_severity = Counter(item["severity"] for item in items)
    by_flag = Counter(flag for item in items for flag in item["flags"])

    report = {
        "summary": {
            "input_path": input_path.as_posix(),
            **dict(segment_counts),
            "qa_items": len(items),
            "by_modality": dict(by_modality),
            "severity": dict(by_severity),
            "flags": dict(by_flag),
        },
        "totals": {
            "qa_items": len(items),
            "by_section": dict(by_section),
            "split_items": len(split_rows),
            "cleaned_items": len(cleaned_rows),
            "cleaned_export": cleaned_metadata,
        },
    }

    report_path = output_dir / "segmented_qa_quality_report.json"
    items_csv_path = output_dir / "segmented_qa_quality_items.csv"
    split_json_path = output_dir / "segmented_qa_split_items.json"
    split_csv_path = output_dir / "segmented_qa_split_items.csv"
    cleaned_json_path = output_dir / "segmented_qa_cleaned_items.json"
    cleaned_csv_path = output_dir / "segmented_qa_cleaned_items.csv"

    _write_json(report_path, report)
    _write_csv(items_csv_path, items)
    _write_json(
        split_json_path,
        {"items": split_rows, "metadata": {"input_path": input_path.as_posix(), "total_items": len(split_rows)}},
    )
    _write_csv(split_csv_path, split_rows)
    _write_json(cleaned_json_path, {"items": cleaned_rows, "metadata": cleaned_metadata})
    _write_csv(cleaned_csv_path, cleaned_rows)

    return {
        "report": report_path,
        "items_csv": items_csv_path,
        "split_json": split_json_path,
        "split_csv": split_csv_path,
        "cleaned_json": cleaned_json_path,
        "cleaned_csv": cleaned_csv_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    for label, path in evaluate_segmented_qa(args.input, args.output_dir).items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
