"""Export grouped reasoning evidence into split question-answer pairs."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from .groups import MODALITIES, REASONING_GROUPS
from .groups import group_all_evidence
from .normalizer import MODALITY_ORDER, normalize_all_modalities, normalize_sample_key


NUMBERED_ITEM_RE = re.compile(r"(?:^|\s)(\d+)[.)]\s+")
ANSWER_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
SEGMENTED_EVIDENCE_CSV_FIELDS = [
    "file_name",
    "task_label",
    "start_timestamp",
    "end_timestamp",
    "segment_confidence",
    "segment_id",
    "unit_index",
    "pair_index",
    "modality",
    "section",
    "question",
    "answer",
    "timestamp",
    "confidence",
]
SEGMENT_DETAIL_FIELDS = [
    "segment_id",
    "file_name",
    "source_prefix",
    "split_dir",
    "side",
    "task_label",
    "start_seconds",
    "end_seconds",
    "start_timestamp",
    "end_timestamp",
    "confidence",
    "evidence_modalities",
    "notes",
    "source_files",
]


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


def split_question_text(text: str) -> list[str]:
    """Split numbered or repeated question-mark text into question items."""
    text = _as_text(text).strip()
    if not text:
        return []

    numbered_items = split_numbered_text(text)
    if len(numbered_items) > 1:
        return numbered_items

    question_items = [
        f"{item.strip()}?"
        for item in text.split("?")
        if item.strip()
    ]
    return question_items if len(question_items) > 1 else [text]


def split_answer_text(text: str) -> list[str]:
    """Split numbered or sentence-delimited answer text into answer items."""
    text = _as_text(text).strip()
    if not text:
        return []

    numbered_items = split_numbered_text(text)
    if len(numbered_items) > 1:
        return numbered_items

    sentence_items = [
        item.strip()
        for item in ANSWER_SENTENCE_RE.split(text)
        if item.strip()
    ]
    return sentence_items if len(sentence_items) > 1 else [text]


def split_qa_pairs(question: str, answer: str) -> list[dict[str, str]]:
    """Split a unit's question and answer strings into paired Q/A records."""
    question_items = split_question_text(question)
    answer_items = (
        split_answer_text(answer)
        if len(question_items) > 1
        else split_numbered_text(answer)
    )

    if not question_items and not answer_items:
        return []

    pair_count = max(len(question_items), len(answer_items), 1)
    pairs: list[dict[str, str]] = []

    for index in range(pair_count):
        question_text = question_items[index] if index < len(question_items) else ""
        answer_text = answer_items[index] if index < len(answer_items) else ""

        if question_text or answer_text:
            pairs.append(
                {
                    "question": question_text,
                    "answer": answer_text,
                }
            )

    return pairs


def split_evidence_units(evidence_units: list[dict]) -> list[dict[str, Any]]:
    """Split evidence units with numbered Q/A text into one unit per pair."""
    split_units: list[dict[str, Any]] = []

    for source_unit_index, unit in enumerate(evidence_units):
        if not isinstance(unit, dict):
            continue

        pairs = split_qa_pairs(
            _as_text(unit.get("question", "")),
            _as_text(unit.get("answer", "")),
        )
        if not pairs:
            continue

        for pair_index, pair in enumerate(pairs, start=1):
            split_unit = dict(unit)
            split_unit["question"] = pair["question"]
            split_unit["answer"] = pair["answer"]
            split_unit["source_unit_index"] = source_unit_index
            split_unit["pair_index"] = pair_index
            split_units.append(split_unit)

    return split_units


def split_normalized_evidence_units(
    normalized_data: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Split each sample's evidence units into one Q/A pair per unit."""
    for sample_data in normalized_data.values():
        if not isinstance(sample_data, dict):
            continue

        evidence_units = sample_data.get("evidence_units", [])
        if not isinstance(evidence_units, list):
            continue

        sample_data["evidence_units"] = split_evidence_units(evidence_units)

    return normalized_data


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


def _source_file_candidates(source_files: Any) -> list[str]:
    """Return source-file paths in preferred display order."""
    if not isinstance(source_files, dict):
        return []

    candidates: list[str] = []
    value = source_files.get("with_audio")
    if isinstance(value, str) and value:
        candidates.append(value)

    videos = source_files.get("videos")
    if isinstance(videos, dict):
        for key in ("rgb", "event", "depth", "ir"):
            value = videos.get(key)
            if isinstance(value, str) and value:
                candidates.append(value)

    for key in MODALITY_ORDER:
        value = source_files.get(key)
        if isinstance(value, str) and value:
            candidates.append(value)

    value = source_files.get("audio")
    if isinstance(value, str) and value:
        candidates.append(value)

    return candidates


def _derive_file_name(source_files: Any, source_prefix: str = "") -> str:
    """Derive a stable file name from source files, falling back to source prefix."""
    for candidate in _source_file_candidates(source_files):
        return Path(candidate).name
    return source_prefix


def _normalize_task_segment_metadata(
    task_data: dict[str, Any],
    segment: dict[str, Any],
) -> dict[str, Any]:
    """Build export metadata for one task segment."""
    source_prefix = _as_text(task_data.get("source_prefix", ""))
    source_files = task_data.get("source_files", {})

    metadata = {
        "segment_id": _as_text(segment.get("segment_id", "")),
        "file_name": _derive_file_name(source_files, source_prefix),
        "source_prefix": source_prefix,
        "split_dir": _as_text(task_data.get("split_dir", "")),
        "side": _as_text(task_data.get("side", "")),
        "task_label": _as_text(segment.get("task_label", "")),
        "start_seconds": segment.get("start_seconds"),
        "end_seconds": segment.get("end_seconds"),
        "start_timestamp": _as_text(segment.get("start_timestamp", "")),
        "end_timestamp": _as_text(segment.get("end_timestamp", "")),
        "confidence": segment.get("confidence"),
        "evidence_modalities": segment.get("evidence_modalities", [])
        if isinstance(segment.get("evidence_modalities", []), list)
        else [],
        "notes": _as_text(segment.get("notes", "")),
        "source_files": source_files if isinstance(source_files, dict) else {},
    }

    return metadata


def collect_task_segment_metadata(
    segmented_folder: str | Path = "segmented_outputs",
) -> dict[str, dict[str, Any]]:
    """Collect rich segment details from task-segment suggestion files."""
    segmented_dir = Path(segmented_folder)
    metadata_by_sample: dict[str, dict[str, Any]] = {}

    for task_path in sorted((segmented_dir / "dataset").glob("**/*_task_segments.json")):
        task_data = load_json_file(task_path)
        segments = task_data.get("segments", [])
        if not isinstance(segments, list):
            continue

        for segment in segments:
            if not isinstance(segment, dict):
                continue

            segment_id = _as_text(segment.get("segment_id", ""))
            if not segment_id:
                continue

            metadata_by_sample[segment_id] = _normalize_task_segment_metadata(
                task_data,
                segment,
            )

    return metadata_by_sample


def run_export_segmented_normalized_evidence_csv(
    input_path: str | Path = "segmented_normalized_evidence_units.json",
    output_path: str | Path = "segmented_normalized_evidence_units.csv",
) -> int:
    """Export segmented normalized evidence units to a compact CSV."""
    normalized_data = load_json_file(input_path)
    print(f"Loaded segmented normalized evidence from: {Path(input_path)}")

    rows: list[dict[str, Any]] = []
    for segment_id in sorted(normalized_data):
        segment_data = normalized_data[segment_id]
        if not isinstance(segment_data, dict):
            continue

        evidence_units = segment_data.get("evidence_units", [])
        if not isinstance(evidence_units, list):
            continue

        for unit_index, unit in enumerate(evidence_units):
            if not isinstance(unit, dict):
                continue

            rows.append(
                {
                    "file_name": segment_data.get("file_name", ""),
                    "task_label": segment_data.get("task_label", ""),
                    "start_timestamp": segment_data.get("start_timestamp", ""),
                    "end_timestamp": segment_data.get("end_timestamp", ""),
                    "segment_confidence": segment_data.get("confidence", ""),
                    "segment_id": segment_id,
                    "unit_index": unit.get("source_unit_index", unit_index),
                    "pair_index": unit.get("pair_index", 1),
                    "modality": unit.get("modality", ""),
                    "section": unit.get("section", ""),
                    "question": unit.get("question", ""),
                    "answer": unit.get("answer", ""),
                    "timestamp": unit.get("timestamp") or "",
                    "confidence": unit.get("confidence", ""),
                }
            )

    output_csv_path = Path(output_path)
    if output_csv_path.parent != Path("."):
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEGMENTED_EVIDENCE_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Segmented normalized evidence CSV written to: {output_csv_path}")
    print(f"Total CSV rows: {len(rows)}")
    return len(rows)


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


def collect_segment_metadata(
    modality_paths: dict[str, str | Path],
    segmented_folder: str | Path = "segmented_outputs",
) -> dict[str, dict[str, Any]]:
    """Collect segment metadata from segmented modality result files."""
    metadata_by_sample: dict[str, dict[str, Any]] = collect_task_segment_metadata(
        segmented_folder
    )

    for modality in MODALITY_ORDER:
        path = modality_paths.get(modality)
        if path is None:
            continue

        data = load_json_file(path)
        for raw_key in sorted(data):
            sample_data = data[raw_key]
            if not isinstance(sample_data, dict):
                continue

            sample_id = normalize_sample_key(raw_key)
            sample_metadata = metadata_by_sample.setdefault(
                sample_id,
                {
                    "source_files": {},
                },
            )

            for field in ("source_prefix", "side"):
                if field not in sample_metadata and field in sample_data:
                    sample_metadata[field] = sample_data[field]

            segment = sample_data.get("segment")
            if isinstance(segment, dict):
                for field in (
                    "segment_id",
                    "task_label",
                    "start_seconds",
                    "end_seconds",
                    "start_timestamp",
                    "end_timestamp",
                ):
                    if field not in sample_metadata and field in segment:
                        sample_metadata[field] = segment[field]

            source_file = sample_data.get("source_file")
            if isinstance(source_file, str) and source_file:
                source_files = sample_metadata.setdefault("source_files", {})
                if isinstance(source_files, dict) and modality not in source_files:
                    source_files[modality] = source_file

            if "file_name" not in sample_metadata:
                sample_metadata["file_name"] = _derive_file_name(
                    sample_metadata.get("source_files", {}),
                    _as_text(sample_metadata.get("source_prefix", "")),
                )

    return metadata_by_sample


def attach_segment_metadata(
    exported: dict[str, dict[str, Any]],
    segment_metadata: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Attach collected segment metadata and keep large content blocks last."""
    for sample_id, sample_output in exported.items():
        if not isinstance(sample_output, dict):
            continue

        metadata = segment_metadata.get(sample_id, {})
        if not isinstance(metadata, dict):
            metadata = {}

        ordered_sample: dict[str, Any] = {}

        for field in SEGMENT_DETAIL_FIELDS:
            if field == "segment_id" and field not in metadata and field not in sample_output:
                ordered_sample[field] = sample_id
            elif field in metadata:
                ordered_sample[field] = metadata[field]
            elif field in sample_output:
                ordered_sample[field] = sample_output[field]

        if "source_keys" in sample_output:
            ordered_sample["source_keys"] = sample_output["source_keys"]

        content_fields = {"reasoning_groups", "evidence_units"}
        handled_fields = set(SEGMENT_DETAIL_FIELDS) | {"source_keys", "segment"} | content_fields
        for field, value in sample_output.items():
            if field not in handled_fields:
                ordered_sample[field] = value

        for field in ("reasoning_groups", "evidence_units"):
            if field in sample_output:
                ordered_sample[field] = sample_output[field]

        sample_output.clear()
        sample_output.update(ordered_sample)

    return exported


def run_export_segmented_grouped_qa(
    segmented_folder: str | Path = "segmented_outputs",
    normalized_output_path: str | Path = "segmented_normalized_evidence_units.json",
    output_path: str | Path = "segmented_grouped_qa_pairs.json",
) -> dict:
    """Export split grouped Q/A pairs from segmented modality result files."""
    segmented_dir = Path(segmented_folder)
    modality_paths = {
        modality: segmented_dir / f"segmented_{modality}_qa_results.json"
        for modality in MODALITY_ORDER
    }

    segment_metadata = collect_segment_metadata(modality_paths, segmented_dir)

    normalized_data = normalize_all_modalities(
        rgb_path=modality_paths["rgb"],
        event_path=modality_paths["event"],
        depth_path=modality_paths["depth"],
        ir_path=modality_paths["ir"],
        audio_path=modality_paths["audio"],
        output_path=normalized_output_path,
    )
    normalized_data = attach_segment_metadata(normalized_data, segment_metadata)
    normalized_data = split_normalized_evidence_units(normalized_data)
    save_json_file(normalized_data, normalized_output_path)

    grouped_data = group_all_evidence(normalized_data)
    exported = export_grouped_qa_pairs(grouped_data)
    exported = attach_segment_metadata(exported, segment_metadata)
    save_json_file(exported, output_path)

    total_pairs = 0
    print(f"Exported segmented samples: {len(exported)}")
    for sample_id in sorted(exported):
        sample_total = 0
        for group_data in exported[sample_id]["reasoning_groups"].values():
            sample_total += sum(len(records) for records in group_data.values())
        total_pairs += sample_total
        print(f"  {sample_id}: {sample_total} Q/A pairs")
    print(f"Total segmented Q/A pairs: {total_pairs}")
    print(f"Segmented grouped Q/A pairs written to: {Path(output_path)}")

    return exported
