"""Group normalized evidence units into reasoning categories."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any


MODALITIES = ["rgb", "event", "depth", "ir", "audio"]

REASONING_GROUPS = [
    "object_reasoning",
    "spatial_reasoning",
    "temporal_sequence_reasoning",
    "action_motion_reasoning",
    "audio_visual_reasoning",
    "anomaly_uncertainty_reasoning",
    "ungrouped",
]

SECTION_GROUPS = {
    "object_reasoning": [
        "object_recognition",
        "event_object_recognition",
        "depth_object_recognition",
        "counting",
        "event_counting",
        "depth_counting",
    ],
    "spatial_reasoning": [
        "spatial_reasoning",
        "event_spatial_reasoning",
        "depth_spatial_reasoning",
        "navigation",
        "event_navigation",
        "depth_navigation",
        "audio_spatial_location",
    ],
    "temporal_sequence_reasoning": [
        "scene_sequence",
        "event_scene_sequence",
        "depth_scene_sequence",
        "dynamic_recognition",
        "event_dynamic_recognition",
        "depth_dynamic_recognition",
        "audio_temporal_attribute",
    ],
    "action_motion_reasoning": [
        "action",
        "event_action",
        "depth_action",
        "dynamic_counting",
        "event_dynamic_counting",
        "depth_dynamic_counting",
    ],
    "audio_visual_reasoning": [
        "audio_sound_characteristics",
        "audio_counting",
        "audio_temporal_attribute",
        "audio_spatial_location",
        "audio_sound_source_identification",
        "audio_inferential_causality",
        "audio_cross_modal_reasoning",
    ],
    "anomaly_uncertainty_reasoning": [
        "non_common",
        "event_non_common",
        "depth_non_common",
        "light_change",
        "light_recognition",
        "light_recongnition",
    ],
}


def normalize_section_name(section: str) -> str:
    """Normalize section names for deterministic group matching."""
    normalized = str(section or "").strip().lower()
    if normalized.startswith("audio_"):
        normalized = re.sub(r"_\d+$", "", normalized)
    return normalized


def find_reasoning_groups_for_section(section: str) -> list[str]:
    """Return every reasoning group that matches a section name."""
    normalized = normalize_section_name(section)
    groups = [
        group
        for group in REASONING_GROUPS
        if normalized in SECTION_GROUPS.get(group, [])
    ]
    return groups or ["ungrouped"]


def create_empty_group() -> dict:
    """Create an empty reasoning group with independent lists."""
    return {
        "evidence": {modality: [] for modality in MODALITIES},
        "units": [],
    }


def create_empty_reasoning_groups() -> dict:
    """Create every reasoning group with independent nested containers."""
    return {
        group: create_empty_group()
        for group in REASONING_GROUPS
    }


def group_evidence_units_for_sample(evidence_units: list[dict]) -> dict:
    """Group one sample's evidence units by reasoning category."""
    reasoning_groups = create_empty_reasoning_groups()

    for unit in evidence_units or []:
        if not isinstance(unit, dict):
            continue

        caption = unit.get("caption", "")
        if not isinstance(caption, str) or not caption.strip():
            continue

        modality = unit.get("modality", "")
        section = unit.get("section", "")
        target_groups = ["ungrouped"]
        if isinstance(modality, str) and modality in MODALITIES:
            target_groups = find_reasoning_groups_for_section(str(section or ""))

        for group in target_groups:
            if modality in MODALITIES:
                reasoning_groups[group]["evidence"][modality].append(caption)
            reasoning_groups[group]["units"].append(copy.deepcopy(unit))

    return reasoning_groups


def group_all_evidence(normalized_data: dict) -> dict:
    """Group all normalized samples into reasoning groups."""
    if not isinstance(normalized_data, dict):
        return {}

    grouped: dict[str, dict[str, Any]] = {}
    for sample_id in sorted(normalized_data):
        sample_data = normalized_data[sample_id]
        if not isinstance(sample_data, dict):
            sample_data = {}

        source_keys = sample_data.get("source_keys", {})
        evidence_units = sample_data.get("evidence_units", [])

        grouped[sample_id] = {
            "source_keys": copy.deepcopy(source_keys) if isinstance(source_keys, dict) else {},
            "reasoning_groups": group_evidence_units_for_sample(
                evidence_units if isinstance(evidence_units, list) else []
            ),
        }

    return grouped


def load_json_file(path: str | Path) -> dict:
    """Load a JSON object from disk, returning an empty dict when missing."""
    json_path = Path(path)
    if not json_path.exists():
        print(f"WARNING: Missing normalized evidence file: {json_path}")
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


def run_group_evidence(
    input_path: str | Path = "normalized_evidence_units.json",
    output_path: str | Path = "grouped_evidence.json",
) -> dict:
    """Load normalized evidence, group it, and save grouped evidence."""
    normalized_data = load_json_file(input_path)
    print(f"Loaded normalized evidence from: {Path(input_path)}")

    grouped_data = group_all_evidence(normalized_data)
    save_json_file(grouped_data, output_path)

    print(f"Grouped samples: {len(grouped_data)}")
    for sample_id in sorted(grouped_data):
        print(f"  {sample_id}:")
        reasoning_groups = grouped_data[sample_id]["reasoning_groups"]
        for group in REASONING_GROUPS:
            unit_count = len(reasoning_groups[group]["units"])
            print(f"    {group}: {unit_count}")
    print(f"Grouped evidence written to: {Path(output_path)}")

    return grouped_data
