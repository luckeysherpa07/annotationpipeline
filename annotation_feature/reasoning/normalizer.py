"""Normalize modality QA outputs into shared evidence units."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_RELIABILITY: dict[str, float] = {
    "rgb": 0.85,
    "event": 0.85,
    "depth": 0.65,
    "ir": 0.75,
    "audio": 0.80,
}

MODALITY_PATHS: dict[str, str] = {
    "rgb": "rgb_qa_results.json",
    "event": "event_qa_results.json",
    "depth": "depth_qa_results.json",
    "ir": "ir_qa_results.json",
    "audio": "audio_qa_results.json",
}

MODALITY_ORDER = ("rgb", "event", "depth", "ir", "audio")
MODALITY_SUFFIXES = ("_rgb", "_event", "_depth", "_ir", "_audio")
DEPTH_UNCERTAINTY_PHRASES = (
    "noise",
    "floating particles",
    "hard to distinguish",
    "uncertain",
    "not clear",
)


def normalize_sample_key(raw_key: str) -> str:
    """Return the core sample id for a modality result key."""
    normalized_key = str(raw_key or "").replace("\\", "/")
    sample_id = normalized_key.rsplit("/", 1)[-1]

    for suffix in MODALITY_SUFFIXES:
        if sample_id.endswith(suffix):
            sample_id = sample_id[: -len(suffix)]
            break

    return sample_id


def get_modality_confidence(
    modality: str,
    sample_id: str,
    raw_key: str,
    caption: str,
) -> float:
    """Compute a simple confidence score for one evidence unit."""
    confidence = DEFAULT_RELIABILITY.get(modality, 0.5)
    night_text = f"{sample_id} {raw_key}".lower()

    if "night" in night_text:
        if modality == "rgb":
            confidence = 0.45
        elif modality == "event":
            confidence = 0.90
        elif modality == "ir":
            confidence = 0.85

    if modality == "depth":
        caption_lower = caption.lower()
        if any(phrase in caption_lower for phrase in DEPTH_UNCERTAINTY_PHRASES):
            confidence = 0.35

    return confidence


def _as_text(value: Any) -> str:
    """Return value as text only when it is already a string."""
    return value if isinstance(value, str) else ""


def _build_evidence_unit(
    modality: str,
    section: str,
    caption: str,
    question: str,
    answer: str,
    timestamp: Any,
    sample_id: str,
    raw_key: str,
) -> dict[str, Any]:
    """Build one normalized evidence-unit dictionary."""
    return {
        "modality": modality,
        "section": section,
        "caption": caption,
        "question": question,
        "answer": answer,
        "timestamp": timestamp,
        "confidence": get_modality_confidence(modality, sample_id, raw_key, caption),
    }


def _extract_audio_evidence_units(
    annotations: dict[str, Any],
    sample_id: str,
    raw_key: str,
) -> list[dict[str, Any]]:
    """Extract audio evidence only from annotations.categories."""
    categories = annotations.get("categories", {})
    if not isinstance(categories, dict):
        return []

    evidence_units: list[dict[str, Any]] = []
    for category_name in sorted(categories):
        category_data = categories[category_name]
        if not isinstance(category_data, dict):
            continue

        caption = _as_text(category_data.get("caption", ""))
        question = _as_text(category_data.get("question", ""))
        answer = _as_text(category_data.get("answer", ""))
        if not (caption.strip() or question.strip() or answer.strip()):
            continue

        evidence_units.append(
            _build_evidence_unit(
                modality="audio",
                section=category_name,
                caption=caption,
                question=question,
                answer=answer,
                timestamp=category_data.get("timestamp"),
                sample_id=sample_id,
                raw_key=raw_key,
            )
        )

    return evidence_units


def extract_evidence_units_from_sample(
    sample_data: dict,
    modality: str,
    sample_id: str,
    raw_key: str,
) -> list[dict]:
    """Extract normalized section-level evidence units from one sample."""
    annotations = sample_data.get("annotations", {}) if isinstance(sample_data, dict) else {}
    if not isinstance(annotations, dict):
        return []

    if modality == "audio":
        return _extract_audio_evidence_units(annotations, sample_id, raw_key)

    evidence_units: list[dict[str, Any]] = []
    for section_name in sorted(annotations):
        section_data = annotations[section_name]
        if not isinstance(section_data, dict):
            continue

        caption = _as_text(section_data.get("caption", ""))
        if not caption.strip():
            continue

        evidence_units.append(
            _build_evidence_unit(
                modality=modality,
                section=section_name,
                caption=caption,
                question=_as_text(section_data.get("question", "")),
                answer=_as_text(section_data.get("answer", "")),
                timestamp=None,
                sample_id=sample_id,
                raw_key=raw_key,
            )
        )

    return evidence_units


def extract_evidence_units(data: dict, modality: str) -> dict:
    """Extract evidence units from all samples for one modality."""
    if not isinstance(data, dict):
        return {}

    results: dict[str, dict[str, Any]] = {}
    for raw_key in sorted(data):
        sample_data = data[raw_key]
        if not isinstance(sample_data, dict):
            continue

        sample_id = normalize_sample_key(raw_key)
        evidence_units = extract_evidence_units_from_sample(
            sample_data=sample_data,
            modality=modality,
            sample_id=sample_id,
            raw_key=raw_key,
        )

        sample_result = results.setdefault(
            sample_id,
            {
                "source_key": raw_key,
                "evidence_units": [],
            },
        )
        sample_result["evidence_units"].extend(evidence_units)

    return results


def load_json_file(path: str | Path) -> dict:
    """Load a JSON object from disk, returning an empty dict when missing."""
    json_path = Path(path)
    if not json_path.exists():
        print(f"WARNING: Missing modality JSON file: {json_path}")
        return {}

    try:
        with open(json_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {json_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object in {json_path}")

    return data


def normalize_all_modalities(
    rgb_path: str | Path = MODALITY_PATHS["rgb"],
    event_path: str | Path = MODALITY_PATHS["event"],
    depth_path: str | Path = MODALITY_PATHS["depth"],
    ir_path: str | Path = MODALITY_PATHS["ir"],
    audio_path: str | Path = MODALITY_PATHS["audio"],
    output_path: str | Path = "normalized_evidence_units.json",
) -> dict:
    """Normalize all modality outputs and save shared evidence units."""
    modality_paths = {
        "rgb": Path(rgb_path),
        "event": Path(event_path),
        "depth": Path(depth_path),
        "ir": Path(ir_path),
        "audio": Path(audio_path),
    }

    modality_results: dict[str, dict[str, Any]] = {}
    for modality in MODALITY_ORDER:
        path = modality_paths[modality]
        modality_results[modality] = load_json_file(path)
        print(f"Loaded {modality}: {path} ({len(modality_results[modality])} samples)")

    merged: dict[str, dict[str, Any]] = {}
    modality_sample_counts: dict[str, int] = {}

    for modality in MODALITY_ORDER:
        extracted = extract_evidence_units(modality_results[modality], modality)
        modality_sample_counts[modality] = len(extracted)

        for sample_id in sorted(extracted):
            sample_payload = extracted[sample_id]
            merged_sample = merged.setdefault(
                sample_id,
                {
                    "source_keys": {},
                    "evidence_units": [],
                },
            )
            merged_sample["source_keys"][modality] = sample_payload["source_key"]
            merged_sample["evidence_units"].extend(sample_payload["evidence_units"])

    final_output = {
        sample_id: {
            "source_keys": {
                modality: merged[sample_id]["source_keys"][modality]
                for modality in MODALITY_ORDER
                if modality in merged[sample_id]["source_keys"]
            },
            "evidence_units": merged[sample_id]["evidence_units"],
        }
        for sample_id in sorted(merged)
    }

    output_file = Path(output_path)
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(final_output, handle, indent=2, ensure_ascii=False)

    print("Samples per modality:")
    for modality in MODALITY_ORDER:
        print(f"  {modality}: {modality_sample_counts[modality]}")
    print(f"Total normalized samples: {len(final_output)}")
    print("Evidence units per sample:")
    for sample_id, sample_payload in final_output.items():
        print(f"  {sample_id}: {len(sample_payload['evidence_units'])}")
    print(f"Normalized evidence units written to: {output_file}")

    return final_output
