"""Rule-based quality evaluation for aligned modality QA JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from annotation_feature.demo_result import DEMO_RESULT
from prompts.depth_prompts import DEPTH_PROMPTS
from prompts.event_prompts import EVENT_PROMPTS
from prompts.ir_prompts import IR_PROMPTS
from prompts.rgb_prompts import RGB_PROMPTS

from .splitting import split_numbered_items, split_status


DEFAULT_ALIGNED_QA_FILES = {
    "rgb": Path("qa_pairs/aligned/rgb_qa_results_aligned.json"),
    "ir": Path("qa_pairs/aligned/ir_qa_results_aligned.json"),
    "event": Path("qa_pairs/aligned/event_qa_results_aligned.json"),
    "audio": Path("qa_pairs/aligned/audio_qa_results_aligned.json"),
    "depth": Path("qa_pairs/aligned/marigold_depth_qa_results_aligned.json"),
}

EXPECTED_SECTIONS = {
    "rgb": tuple(RGB_PROMPTS.keys()),
    "ir": tuple(IR_PROMPTS.keys()),
    "event": tuple(EVENT_PROMPTS.keys()),
    "depth": tuple(DEPTH_PROMPTS.keys()),
}

PROMPT_LEAK_RE = re.compile(
    r"please work as a vqa assistant|generate exactly|caption prompt|question prompt|answering prompt",
    flags=re.I,
)
UNKNOWN_RE = re.compile(
    r"\b(unknown|unclear|not visible|cannot determine|can't determine|unable to determine|not enough information|not applicable|n/a)\b",
    flags=re.I,
)
FRAME_RE = re.compile(r"\b(frame|image|video|depth map|depth image)\b", flags=re.I)
SHORT_ANSWER_RE = re.compile(r"^(yes|no|none|zero|one|two|three|four|five)\.?$", flags=re.I)
DEPTH_FORBIDDEN_RE = re.compile(
    r"\b(color|colour|red|green|blue|yellow|white|black|lighting|illumination|light source|texture)\b",
    flags=re.I,
)
IR_FORBIDDEN_RE = re.compile(r"\b(depth map|depth image|rgb image)\b", flags=re.I)
RGB_FORBIDDEN_RE = re.compile(r"\b(thermal|infrared|depth map|depth image)\b", flags=re.I)


@dataclass
class QualityItem:
    modality: str
    source_file: str
    pair_key: str
    section: str
    caption: str
    question: str
    answer: str
    flags: list[str] = field(default_factory=list)
    severity: str = "pass"
    split_status: str = "single"
    question_count: int = 1
    answer_count: int = 1
    recommended_action: str = "keep"


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object in {path}")
    return data


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _item_complete(item: dict[str, Any]) -> bool:
    return isinstance(item, dict) and all(str(item.get(field, "")).strip() for field in ("caption", "question", "answer"))


def _caption_complete(item: dict[str, Any]) -> bool:
    return isinstance(item, dict) and bool(str(item.get("caption", "")).strip())


def _audio_sections(entry: dict[str, Any]) -> tuple[str, ...]:
    annotations = entry.get("annotations", {}) if isinstance(entry, dict) else {}
    categories = annotations.get("categories", {}) if isinstance(annotations, dict) else {}
    if not isinstance(categories, dict):
        return ()
    return tuple(section for section, qa in sorted(categories.items()) if isinstance(section, str) and isinstance(qa, dict))


def _expected_sections_for_entry(modality: str, entry: dict[str, Any]) -> tuple[str, ...]:
    if modality == "audio":
        return _audio_sections(entry)
    return EXPECTED_SECTIONS[modality]


def _entry_complete(modality: str, entry: dict[str, Any], expected_sections: tuple[str, ...]) -> bool:
    if not isinstance(entry, dict) or entry.get("status") == "skipped_missing_side":
        return False
    annotations = entry.get("annotations", {})
    if not isinstance(annotations, dict):
        return False
    if modality == "audio":
        categories = annotations.get("categories", {})
        return (
            bool(expected_sections)
            and isinstance(categories, dict)
            and _caption_complete(annotations.get("audio_hia", {}))
            and _caption_complete(annotations.get("audio_chronological_caption", {}))
            and all(_item_complete(categories.get(section, {})) for section in expected_sections)
        )
    return all(_item_complete(annotations.get(section, {})) for section in expected_sections)


def _modality_mismatch(modality: str, question: str, answer: str) -> bool:
    text = f"{question} {answer}"
    if modality == "depth":
        return bool(DEPTH_FORBIDDEN_RE.search(text))
    if modality == "ir":
        return bool(IR_FORBIDDEN_RE.search(text))
    if modality == "rgb":
        return bool(RGB_FORBIDDEN_RE.search(text))
    return False


def _append_flag(flags: list[str], flag: str) -> None:
    if flag not in flags:
        flags.append(flag)


def _classify_item(
    modality: str,
    section: str,
    qa: dict[str, Any],
    duplicate_question: bool,
    entry_quality_flags: set[str] | None = None,
) -> QualityItem:
    caption = str(qa.get("caption", "")).strip()
    question = str(qa.get("question", "")).strip()
    answer = str(qa.get("answer", "")).strip()
    status, questions, answers = split_status(question, answer)
    flags: list[str] = []
    qa_quality_flags = {
        str(flag).strip()
        for flag in qa.get("quality_flags", [])
        if str(flag).strip()
    }
    entry_quality_flags = entry_quality_flags or set()

    if not caption or not question or not answer:
        _append_flag(flags, "empty_field")
    if PROMPT_LEAK_RE.search(question):
        _append_flag(flags, "prompt_leak")
    if status != "single":
        _append_flag(flags, "multi_question" if questions else "numbered_answer_list")
        if status == "count_mismatch":
            _append_flag(flags, "question_answer_count_mismatch")
    if duplicate_question:
        _append_flag(flags, "duplicate_question")
    if len(answer.split()) < 2 or SHORT_ANSWER_RE.match(answer):
        _append_flag(flags, "short_answer")
    if UNKNOWN_RE.search(question) or UNKNOWN_RE.search(answer) or UNKNOWN_RE.search(caption):
        _append_flag(flags, "unknown_unclear")
    if _modality_mismatch(modality, question, answer):
        _append_flag(flags, "modality_mismatch")
    if FRAME_RE.search(question):
        _append_flag(flags, "frame_reference")
    if qa == DEMO_RESULT.get(section):
        _append_flag(flags, "demo_fallback")
    if not caption and ("empty_qa_caption" in qa_quality_flags or "has_empty_qa_caption" in entry_quality_flags):
        _append_flag(flags, "empty_qa_caption")
    if "demo_hia_fallback" in entry_quality_flags:
        _append_flag(flags, "demo_hia_fallback")
    if "missing_hia_source" in entry_quality_flags:
        _append_flag(flags, "missing_hia_source")

    hard_flags = {"empty_field", "prompt_leak", "demo_fallback"}
    transform_flags = {"multi_question", "numbered_answer_list", "question_answer_count_mismatch"}
    if any(flag in hard_flags for flag in flags):
        severity = "hard_fail"
        recommended_action = "regenerate_or_remove"
    elif "question_answer_count_mismatch" in flags:
        severity = "needs_transform"
        recommended_action = "review"
    elif any(flag in transform_flags for flag in flags):
        severity = "needs_transform"
        recommended_action = "split"
    elif flags:
        severity = "quality_warning"
        recommended_action = "review"
    else:
        severity = "pass"
        recommended_action = "keep"

    return QualityItem(
        modality=modality,
        source_file="",
        pair_key="",
        section=section,
        caption=caption,
        question=question,
        answer=answer,
        flags=flags,
        severity=severity,
        split_status=status,
        question_count=len(questions) if questions else 1,
        answer_count=len(answers) if answers else 1,
        recommended_action=recommended_action,
    )


def _iter_quality_items(modality: str, path: Path, data: dict[str, Any]) -> tuple[list[QualityItem], dict[str, Any]]:
    question_counts: Counter[str] = Counter()
    for entry in data.values():
        annotations = entry.get("annotations", {}) if isinstance(entry, dict) else {}
        if not isinstance(annotations, dict):
            continue
        expected_sections = _expected_sections_for_entry(modality, entry)
        qa_source = annotations.get("categories", {}) if modality == "audio" else annotations
        if not isinstance(qa_source, dict):
            continue
        for section in expected_sections:
            qa = qa_source.get(section)
            if isinstance(qa, dict):
                question_counts[str(qa.get("question", "")).strip()] += 1

    items: list[QualityItem] = []
    entry_counts = Counter()
    for pair_key, entry in sorted(data.items()):
        if not isinstance(entry, dict):
            entry_counts["invalid_entries"] += 1
            continue
        if entry.get("status") == "skipped_missing_side":
            entry_counts["skipped_entries"] += 1
            continue
        expected_sections = _expected_sections_for_entry(modality, entry)
        if _entry_complete(modality, entry, expected_sections):
            entry_counts["complete_entries"] += 1
        else:
            entry_counts["partial_or_empty_entries"] += 1

        annotations = entry.get("annotations", {})
        if not isinstance(annotations, dict):
            continue
        qa_source = annotations.get("categories", {}) if modality == "audio" else annotations
        if not isinstance(qa_source, dict):
            continue
        entry_quality_flags = {
            str(flag).strip()
            for flag in annotations.get("quality_flags", [])
            if str(flag).strip()
        }
        for section in expected_sections:
            qa = qa_source.get(section)
            if not isinstance(qa, dict):
                continue
            item = _classify_item(
                modality,
                section,
                qa,
                duplicate_question=question_counts[str(qa.get("question", "")).strip()] > 1,
                entry_quality_flags=entry_quality_flags,
            )
            item.source_file = path.as_posix()
            item.pair_key = str(pair_key)
            items.append(item)

    entry_counts["entries"] = len(data)
    return items, dict(entry_counts)


def _build_split_rows(items: list[QualityItem]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        questions = split_numbered_items(item.question)
        answers = split_numbered_items(item.answer)
        if item.split_status == "aligned" and questions and answers:
            for index, (question, answer) in enumerate(zip(questions, answers), start=1):
                row = asdict(item)
                row.update(
                    {
                        "qa_index": index,
                        "question": question,
                        "answer": answer,
                        "source_status": "split_from_numbered_list",
                    }
                )
                rows.append(row)
        else:
            row = asdict(item)
            row.update({"qa_index": 1, "source_status": item.split_status})
            rows.append(row)
    return rows


def _clean_qa_id_text(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", str(text)).strip("_").lower()
    return cleaned or "unknown"


def _build_cleaned_rows(items: list[QualityItem]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build flat benchmark-ready QA rows without mutating source QA JSON files."""
    rows: list[dict[str, Any]] = []
    excluded = Counter()

    for item in items:
        if item.severity == "hard_fail":
            excluded["hard_fail"] += 1
            continue

        base = {
            "modality": item.modality,
            "source_file": item.source_file,
            "pair_key": item.pair_key,
            "section": item.section,
            "caption": item.caption,
            "source_question": item.question,
            "source_answer": item.answer,
            "source_severity": item.severity,
            "source_flags": list(item.flags),
            "source_split_status": item.split_status,
        }

        questions = split_numbered_items(item.question)
        answers = split_numbered_items(item.answer)
        if item.split_status == "aligned" and questions and answers:
            for index, (question, answer) in enumerate(zip(questions, answers), start=1):
                qa_id = "__".join(
                    [
                        _clean_qa_id_text(item.modality),
                        _clean_qa_id_text(item.pair_key),
                        _clean_qa_id_text(item.section),
                        f"{index:02d}",
                    ]
                )
                rows.append(
                    {
                        **base,
                        "qa_id": qa_id,
                        "question": question,
                        "answer": answer,
                        "transform": "split_multi_question",
                        "split_index": index,
                        "split_count": len(questions),
                    }
                )
            continue

        if item.split_status != "single":
            excluded[f"unresolved_{item.split_status}"] += 1
            continue

        qa_id = "__".join(
            [
                _clean_qa_id_text(item.modality),
                _clean_qa_id_text(item.pair_key),
                _clean_qa_id_text(item.section),
                "01",
            ]
        )
        rows.append(
            {
                **base,
                "qa_id": qa_id,
                "question": item.question,
                "answer": item.answer,
                "transform": "none",
                "split_index": 1,
                "split_count": 1,
            }
        )

    metadata = {
        "total_cleaned_items": len(rows),
        "excluded": dict(excluded),
        "transforms": dict(Counter(row["transform"] for row in rows)),
    }
    return rows, metadata


def evaluate_aligned_qa(
    aligned_files: dict[str, Path] | None = None,
    output_dir: Path | str = "outputs",
) -> dict[str, Path]:
    """Evaluate aligned QA JSON files and export report, issue rows, split rows, and cleaned rows."""
    output_dir = Path(output_dir)
    aligned_files = aligned_files or DEFAULT_ALIGNED_QA_FILES

    all_items: list[QualityItem] = []
    summary: dict[str, Any] = {}
    by_flag: Counter[str] = Counter()
    by_section: Counter[str] = Counter()
    by_severity: Counter[str] = Counter()

    for modality, path in aligned_files.items():
        path = Path(path)
        if not path.exists():
            summary[modality] = {"missing_file": path.as_posix()}
            continue
        data = _load_json(path)
        items, entry_counts = _iter_quality_items(modality, path, data)
        all_items.extend(items)
        modality_flags = Counter(flag for item in items for flag in item.flags)
        modality_severity = Counter(item.severity for item in items)
        summary[modality] = {
            **entry_counts,
            "qa_items": len(items),
            "severity": dict(modality_severity),
            "flags": dict(modality_flags),
        }
        by_flag.update(modality_flags)
        by_severity.update(modality_severity)
        by_section.update(f"{item.modality}:{item.section}" for item in items)

    item_rows = []
    for item in all_items:
        row = asdict(item)
        row["flags"] = ";".join(item.flags)
        item_rows.append(row)

    split_rows = _build_split_rows(all_items)
    split_json_rows = []
    for row in split_rows:
        converted = dict(row)
        if isinstance(converted.get("flags"), list):
            converted["flags"] = list(converted["flags"])
        split_json_rows.append(converted)

    cleaned_rows, cleaned_metadata = _build_cleaned_rows(all_items)
    cleaned_csv_rows = []
    for row in cleaned_rows:
        converted = dict(row)
        if isinstance(converted.get("source_flags"), list):
            converted["source_flags"] = ";".join(converted["source_flags"])
        cleaned_csv_rows.append(converted)

    report = {
        "summary": summary,
        "totals": {
            "qa_items": len(all_items),
            "severity": dict(by_severity),
            "flags": dict(by_flag),
            "by_section": dict(by_section),
            "split_items": len(split_rows),
            "cleaned_items": len(cleaned_rows),
            "cleaned_export": cleaned_metadata,
        },
    }

    report_path = output_dir / "aligned_qa_quality_report.json"
    items_csv_path = output_dir / "aligned_qa_quality_items.csv"
    split_json_path = output_dir / "aligned_qa_split_items.json"
    split_csv_path = output_dir / "aligned_qa_split_items.csv"
    cleaned_json_path = output_dir / "aligned_qa_cleaned_items.json"
    cleaned_csv_path = output_dir / "aligned_qa_cleaned_items.csv"

    _write_json(report_path, report)
    _write_csv(items_csv_path, item_rows)
    _write_json(split_json_path, {"items": split_json_rows, "metadata": {"total_items": len(split_json_rows)}})
    _write_csv(split_csv_path, split_rows)
    _write_json(cleaned_json_path, {"items": cleaned_rows, "metadata": cleaned_metadata})
    _write_csv(cleaned_csv_path, cleaned_csv_rows)

    return {
        "report": report_path,
        "items_csv": items_csv_path,
        "split_json": split_json_path,
        "split_csv": split_csv_path,
        "cleaned_json": cleaned_json_path,
        "cleaned_csv": cleaned_csv_path,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    outputs = evaluate_aligned_qa(output_dir=args.output_dir)
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
