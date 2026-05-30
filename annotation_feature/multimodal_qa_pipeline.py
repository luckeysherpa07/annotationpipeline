"""Generate implicit cross-modal QA benchmarks from normalized segmented evidence."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from annotation_feature.pipeline.client import create_gemini_client


NIGHT_COMBINATIONS = (
    ("event", "ir"),
    ("event", "audio"),
    ("event", "depth"),
    ("rgb", "ir"),
    ("rgb", "event"),
)

DAY_COMBINATIONS = (
    ("rgb", "audio"),
    ("rgb", "event"),
    ("rgb", "depth"),
    ("rgb", "ir"),
)

MULTIMODAL_QA_MODEL_NAME = "gemini-2.0-flash"
DEFAULT_API_DELAY_SECONDS = 8
DEFAULT_GEMINI_BATCH_SCOPE = "segment"
DEFAULT_MAX_BUNDLES_PER_GEMINI_CALL = 20
QA_STYLE = "implicit_cross_modal"


SECTION_PREFS: dict[str, dict[str, list[str]]] = {
    "context": {
        "rgb": ["action", "object_recognition", "scene_sequence", "navigation", "spatial_reasoning", "text_recognition"],
        "audio": ["audio_sound_source_identification", "audio_event_detection", "audio_ambient_environment_understanding"],
        "event": ["event_scene_sequence", "event_navigation", "event_object_recognition", "event_dynamic_recognition"],
        "depth": ["depth_scene_sequence", "depth_navigation", "depth_object_recognition", "depth_spatial_reasoning"],
        "ir": ["object_recognition", "scene_sequence", "navigation", "light_change", "dynamic_recognition"],
    },
    "decisive": {
        "rgb": ["text_recognition", "object_recognition", "light_change", "light_recognition", "action"],
        "audio": ["audio_sound_source_identification", "audio_event_detection", "audio_event_occurrence_detection", "audio_visual_correspondence"],
        "event": ["event_dynamic_recognition", "event_action", "event_navigation", "event_scene_sequence"],
        "depth": ["depth_spatial_reasoning", "depth_navigation", "depth_dynamic_recognition", "depth_scene_sequence"],
        "ir": ["light_change", "light_recognition", "object_recognition", "dynamic_recognition", "scene_sequence"],
    },
}


PAIR_ROLE_SPECS: dict[tuple[str, str], dict[str, Any]] = {
    ("rgb", "audio"): {
        "capability": "audio-visual implicit grounding",
        "directions": [
            {
                "context_modality": "rgb",
                "decisive_modality": "audio",
                "challenge_types": [
                    "audio_as_material_ground_truth",
                    "audio_as_action_confirmation",
                    "audio_as_sound_source_ground_truth",
                ],
            },
            {
                "context_modality": "audio",
                "decisive_modality": "rgb",
                "challenge_types": ["rgb_as_sound_source_localization", "rgb_as_object_ground_truth"],
            },
        ],
    },
    ("rgb", "event"): {
        "capability": "visible context with event-motion evidence",
        "directions": [
            {
                "context_modality": "rgb",
                "decisive_modality": "event",
                "challenge_types": ["event_as_motion_ground_truth", "event_as_action_phase_ground_truth"],
            },
            {
                "context_modality": "event",
                "decisive_modality": "rgb",
                "challenge_types": ["rgb_as_object_ground_truth", "rgb_as_color_ground_truth"],
            },
        ],
    },
    ("rgb", "depth"): {
        "capability": "visible context with spatial/depth evidence",
        "directions": [
            {
                "context_modality": "rgb",
                "decisive_modality": "depth",
                "challenge_types": ["depth_as_spatial_ground_truth", "depth_as_navigation_ground_truth"],
            },
            {
                "context_modality": "depth",
                "decisive_modality": "rgb",
                "challenge_types": ["rgb_as_object_ground_truth", "rgb_as_text_or_label_ground_truth"],
            },
        ],
    },
    ("rgb", "ir"): {
        "capability": "visible-light and IR complementarity",
        "directions": [
            {
                "context_modality": "ir",
                "decisive_modality": "rgb",
                "challenge_types": ["rgb_as_color_ground_truth", "rgb_as_text_or_label_ground_truth"],
            },
            {
                "context_modality": "rgb",
                "decisive_modality": "ir",
                "challenge_types": ["ir_as_low_light_ground_truth", "ir_as_visibility_ground_truth"],
            },
        ],
    },
    ("event", "ir"): {
        "capability": "low-light scene context with event-motion evidence",
        "directions": [
            {
                "context_modality": "ir",
                "decisive_modality": "event",
                "challenge_types": ["event_as_motion_ground_truth", "event_as_navigation_change_ground_truth"],
            },
            {
                "context_modality": "event",
                "decisive_modality": "ir",
                "challenge_types": ["ir_as_low_light_ground_truth", "ir_as_scene_structure_ground_truth"],
            },
        ],
    },
    ("event", "audio"): {
        "capability": "motion-sound implicit alignment",
        "directions": [
            {
                "context_modality": "event",
                "decisive_modality": "audio",
                "challenge_types": ["audio_as_sound_source_ground_truth", "audio_as_action_confirmation"],
            },
            {
                "context_modality": "audio",
                "decisive_modality": "event",
                "challenge_types": ["event_as_motion_ground_truth", "event_as_repetition_ground_truth"],
            },
        ],
    },
    ("event", "depth"): {
        "capability": "motion and spatial-layout implicit grounding",
        "directions": [
            {
                "context_modality": "depth",
                "decisive_modality": "event",
                "challenge_types": ["event_as_motion_ground_truth", "event_as_navigation_change_ground_truth"],
            },
            {
                "context_modality": "event",
                "decisive_modality": "depth",
                "challenge_types": ["depth_as_spatial_ground_truth", "depth_as_navigation_ground_truth"],
            },
        ],
    },
}


CHALLENGE_TO_DECISIVE = {
    "audio_as_material_ground_truth": "audio",
    "audio_as_action_confirmation": "audio",
    "audio_as_sound_source_ground_truth": "audio",
    "event_as_motion_ground_truth": "event",
    "event_as_action_phase_ground_truth": "event",
    "event_as_navigation_change_ground_truth": "event",
    "event_as_repetition_ground_truth": "event",
    "depth_as_spatial_ground_truth": "depth",
    "depth_as_navigation_ground_truth": "depth",
    "ir_as_low_light_ground_truth": "ir",
    "ir_as_visibility_ground_truth": "ir",
    "ir_as_scene_structure_ground_truth": "ir",
    "rgb_as_color_ground_truth": "rgb",
    "rgb_as_text_or_label_ground_truth": "rgb",
    "rgb_as_sound_source_localization": "rgb",
    "rgb_as_object_ground_truth": "rgb",
}


MODALITY_ROLE_LABELS = {
    "context": "context_object_or_event_grounding",
    "decisive": "decisive_answer_evidence",
}


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


def _canonical_pair(pair: tuple[str, str]) -> tuple[str, str]:
    if pair in PAIR_ROLE_SPECS:
        return pair
    reversed_pair = (pair[1], pair[0])
    if reversed_pair in PAIR_ROLE_SPECS:
        return reversed_pair
    return pair


def _segment_pairs(segment: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    side = str(segment.get("side") or "").lower()
    if side == "night":
        return NIGHT_COMBINATIONS
    if side == "day":
        return DAY_COMBINATIONS
    source_prefix = str(segment.get("source_prefix") or "").lower()
    return NIGHT_COMBINATIONS if "night" in source_prefix else DAY_COMBINATIONS


def _collect_evidence_by_modality(segment: dict[str, Any]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for unit in segment.get("evidence_units", []):
        if not isinstance(unit, dict):
            continue
        modality = str(unit.get("modality") or "").strip().lower()
        section = str(unit.get("section") or "").strip()
        if not modality or not section:
            continue
        caption = str(unit.get("caption") or unit.get("evidence") or "").strip()
        question = str(unit.get("question") or "").strip()
        answer = str(unit.get("answer") or "").strip()
        if not any((caption, question, answer)):
            continue
        grouped.setdefault(modality, {}).setdefault(section, []).append(
            {
                "evidence_id": (
                    f"{segment.get('segment_id')}::{modality}::{section}::"
                    f"{unit.get('source_unit_index', 'na')}::{unit.get('pair_index', 'na')}"
                ),
                "section": section,
                "caption": caption,
                "question": question,
                "answer": answer,
                "confidence": unit.get("confidence"),
                "timestamp": unit.get("timestamp"),
            }
        )
    return grouped


def _unit_text(unit: dict[str, Any]) -> str:
    pieces = []
    caption = str(unit.get("caption") or "").strip()
    question = str(unit.get("question") or "").strip()
    answer = str(unit.get("answer") or "").strip()
    if caption:
        pieces.append(caption)
    if question and answer:
        pieces.append(f"Existing QA: {question} Answer: {answer}")
    return " ".join(pieces).strip()


def _unit_payload(unit: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_id": unit.get("evidence_id"),
        "section": unit.get("section"),
        "caption": unit.get("caption"),
        "question": unit.get("question"),
        "answer": unit.get("answer"),
        "evidence_text": _unit_text(unit),
    }


def _select_units(
    evidence_by_modality: dict[str, dict[str, list[dict[str, Any]]]],
    modality: str,
    role: str,
    max_units: int = 3,
) -> tuple[list[str], str]:
    modality_sections = evidence_by_modality.get(modality, {})
    preferred_sections = SECTION_PREFS.get(role, {}).get(modality, [])
    selected_units: list[dict[str, Any]] = []
    source_sections: list[str] = []
    seen_texts: set[str] = set()

    def add_from_section(section: str) -> None:
        if len(selected_units) >= max_units:
            return
        for unit in modality_sections.get(section, []):
            text_key = _unit_text(unit)
            if not text_key or text_key in seen_texts:
                continue
            selected_units.append(unit)
            seen_texts.add(text_key)
            if section not in source_sections:
                source_sections.append(section)
            break

    for section in preferred_sections:
        add_from_section(section)
        if len(selected_units) >= max_units:
            break

    if not selected_units:
        for section in modality_sections:
            add_from_section(section)
            if len(selected_units) >= max_units:
                break

    evidence = " ".join(_unit_text(unit) for unit in selected_units if _unit_text(unit)).strip()
    return source_sections, evidence


def _select_candidate_units(
    evidence_by_modality: dict[str, dict[str, list[dict[str, Any]]]],
    modality: str,
    role: str,
    max_units: int = 8,
) -> list[dict[str, Any]]:
    modality_sections = evidence_by_modality.get(modality, {})
    preferred_sections = SECTION_PREFS.get(role, {}).get(modality, [])
    selected_units: list[dict[str, Any]] = []
    seen_texts: set[str] = set()

    def add_from_section(section: str) -> None:
        if len(selected_units) >= max_units:
            return
        for unit in modality_sections.get(section, []):
            text_key = _unit_text(unit)
            if not text_key or text_key in seen_texts:
                continue
            selected_units.append(unit)
            seen_texts.add(text_key)
            break

    for section in preferred_sections:
        add_from_section(section)
        if len(selected_units) >= max_units:
            break

    if len(selected_units) < max_units:
        for section in modality_sections:
            add_from_section(section)
            if len(selected_units) >= max_units:
                break

    return [_unit_payload(unit) for unit in selected_units]


def _challenge_supported(challenge_type: str, decisive_evidence: str) -> bool:
    text = decisive_evidence.lower()
    if challenge_type == "audio_as_material_ground_truth":
        return any(token in text for token in ("metal", "metallic", "clang", "click", "clank"))
    if challenge_type == "rgb_as_color_ground_truth":
        return any(token in text for token in ("green", "red", "white", "black", "orange", "blue", "yellow", "color", "coloured", "colored"))
    if challenge_type == "rgb_as_text_or_label_ground_truth":
        return any(token in text for token in ("text", "label", "sign", "number", "word", "display", "exit", "revolution", "game over"))
    if challenge_type.startswith("event_as"):
        return any(token in text for token in ("motion", "moving", "movement", "dynamic", "turn", "forward", "repeated", "rapid", "walking", "riding"))
    if challenge_type.startswith("depth_as"):
        return any(token in text for token in ("depth", "near", "far", "below", "above", "path", "spatial", "3d", "distance", "stairs", "closer"))
    if challenge_type.startswith("ir_as"):
        return any(token in text for token in ("light", "bright", "dark", "thermal", "night", "visible", "contrast", "illumination"))
    if challenge_type.startswith("audio_as"):
        return any(token in text for token in ("sound", "audio", "click", "footstep", "water", "wind", "door", "rustling", "pump"))
    return True


def _select_supported_challenges(direction: dict[str, Any], decisive_evidence: str) -> list[str]:
    candidates = list(direction.get("challenge_types", []))
    supported = [challenge for challenge in candidates if _challenge_supported(challenge, decisive_evidence)]
    return supported


def _build_role_bundles(
    segment: dict[str, Any],
    evidence_by_modality: dict[str, dict[str, list[dict[str, Any]]]],
    pair: tuple[str, str],
) -> list[dict[str, Any]]:
    pair = _canonical_pair(pair)
    spec = PAIR_ROLE_SPECS.get(pair)
    if not spec:
        return []
    if pair[0] not in evidence_by_modality or pair[1] not in evidence_by_modality:
        return []

    bundles: list[dict[str, Any]] = []
    for direction_index, direction in enumerate(spec.get("directions", []), start=1):
        context_modality = direction["context_modality"]
        decisive_modality = direction["decisive_modality"]
        if context_modality not in evidence_by_modality or decisive_modality not in evidence_by_modality:
            continue

        context_sections, context_evidence = _select_units(evidence_by_modality, context_modality, "context")
        decisive_sections, decisive_evidence = _select_units(evidence_by_modality, decisive_modality, "decisive")
        if not context_evidence or not decisive_evidence:
            continue
        context_candidates = _select_candidate_units(evidence_by_modality, context_modality, "context")
        decisive_candidates = _select_candidate_units(evidence_by_modality, decisive_modality, "decisive")

        challenge_types = _select_supported_challenges(direction, decisive_evidence)
        if not challenge_types:
            continue
        for challenge_index, challenge_type in enumerate(challenge_types, start=1):
            bundles.append(
                {
                    "segment_id": segment.get("segment_id"),
                    "source_prefix": segment.get("source_prefix"),
                    "side": segment.get("side"),
                    "task_label": segment.get("task_label"),
                    "start_seconds": segment.get("start_seconds"),
                    "end_seconds": segment.get("end_seconds"),
                    "start_timestamp": segment.get("start_timestamp"),
                    "end_timestamp": segment.get("end_timestamp"),
                    "modalities": list(pair),
                    "capability": spec["capability"],
                    "qa_style": QA_STYLE,
                    "context_modality": context_modality,
                    "decisive_modality": decisive_modality,
                    "challenge_type": challenge_type,
                    "role_direction_index": direction_index,
                    "challenge_index": challenge_index,
                    "evidence_by_modality": {
                        context_modality: {
                            "role": "context",
                            "role_description": MODALITY_ROLE_LABELS["context"],
                            "source_sections": context_sections,
                            "evidence": context_evidence,
                            "evidence_ids": [
                                candidate["evidence_id"]
                                for candidate in context_candidates
                                if candidate.get("evidence_text") and candidate["evidence_text"] in context_evidence
                            ],
                        },
                        decisive_modality: {
                            "role": "decisive",
                            "role_description": MODALITY_ROLE_LABELS["decisive"],
                            "source_sections": decisive_sections,
                            "evidence": decisive_evidence,
                            "evidence_ids": [
                                candidate["evidence_id"]
                                for candidate in decisive_candidates
                                if candidate.get("evidence_text") and candidate["evidence_text"] in decisive_evidence
                            ],
                        },
                    },
                    "evidence_candidates_by_modality": {
                        context_modality: context_candidates,
                        decisive_modality: decisive_candidates,
                    },
                }
            )
    return bundles


def _summarize_evidence(evidence: str, max_chars: int = 420) -> str:
    collapsed = re.sub(r"\s+", " ", evidence).strip()
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 3].rstrip() + "..."


def _infer_template_answer(challenge_type: str, bundle: dict[str, Any]) -> str:
    decisive = bundle["decisive_modality"]
    evidence = bundle["evidence_by_modality"][decisive]["evidence"]
    text = evidence.lower()
    task = str(bundle.get("task_label") or "the activity")

    if challenge_type == "audio_as_material_ground_truth":
        if any(token in text for token in ("metal", "metallic", "clang", "clank", "click")):
            return "It is likely made of metal or has a metal component."
        return "The material is suggested by the sound evidence, but it is not specific enough to name confidently."
    if challenge_type == "audio_as_action_confirmation":
        return "Yes, the sound evidence indicates that the visible interaction actually occurs."
    if challenge_type == "audio_as_sound_source_ground_truth":
        return "The sound is most likely produced by the object or action grounded in the scene context."
    if challenge_type == "event_as_motion_ground_truth":
        return "The subject or relevant object is moving rather than staying still."
    if challenge_type == "event_as_action_phase_ground_truth":
        return "The event evidence indicates an active motion phase rather than a static pause."
    if challenge_type == "event_as_navigation_change_ground_truth":
        return "The motion indicates a navigation change such as moving forward, turning, or transitioning between areas."
    if challenge_type == "event_as_repetition_ground_truth":
        return "The event evidence indicates repeated motion rather than a single isolated movement."
    if challenge_type == "depth_as_spatial_ground_truth":
        return "The answer depends on the spatial layout shown by depth, such as relative distance, height, or near/far structure."
    if challenge_type == "depth_as_navigation_ground_truth":
        return "The subject is moving through the available spatial path indicated by the depth layout."
    if challenge_type == "ir_as_low_light_ground_truth":
        return "The low-light structure remains interpretable through IR brightness or thermal contrast."
    if challenge_type == "ir_as_visibility_ground_truth":
        return "IR makes the relevant area or object more visible under the lighting conditions."
    if challenge_type == "ir_as_scene_structure_ground_truth":
        return "The IR evidence indicates the low-light scene structure where the movement occurs."
    if challenge_type == "rgb_as_color_ground_truth":
        if "green" in text:
            return "The relevant visible color clue is green."
        if "orange" in text:
            return "The relevant visible color clue is orange."
        if "white" in text:
            return "The relevant visible color clue is white."
        if "black" in text:
            return "The relevant visible color clue is black."
        return "The answer depends on a visible color or appearance clue in RGB."
    if challenge_type == "rgb_as_text_or_label_ground_truth":
        return "The answer depends on visible text, numbering, or labeling in RGB."
    if challenge_type == "rgb_as_sound_source_localization":
        return "The visible object or action in RGB localizes the likely source of the sound."
    if challenge_type == "rgb_as_object_ground_truth":
        return f"The RGB evidence identifies the relevant object or action in the '{task}' scene."
    return "The answer requires binding the context evidence to the decisive modality evidence."


def _template_question(bundle: dict[str, Any]) -> str:
    challenge_type = bundle["challenge_type"]
    task = str(bundle.get("task_label") or "this segment")
    if challenge_type == "audio_as_material_ground_truth":
        return "What material is the interacted object likely made of?"
    if challenge_type == "audio_as_action_confirmation":
        return f"Does the person actually manipulate an object during '{task}'?"
    if challenge_type == "audio_as_sound_source_ground_truth":
        return "What is the most likely source of the heard sound in the scene?"
    if challenge_type == "event_as_motion_ground_truth":
        return "Is the subject stationary or moving through the scene?"
    if challenge_type == "event_as_action_phase_ground_truth":
        return "Is the visible task in an active motion phase or a static pause?"
    if challenge_type == "event_as_navigation_change_ground_truth":
        return "Does the subject continue straight, turn, or transition to another area?"
    if challenge_type == "event_as_repetition_ground_truth":
        return "Does the motion pattern look repeated or like a single isolated movement?"
    if challenge_type == "depth_as_spatial_ground_truth":
        return "What spatial relationship is needed to understand the action or path?"
    if challenge_type == "depth_as_navigation_ground_truth":
        return "Is there a clear navigable path for the subject to move through?"
    if challenge_type == "ir_as_low_light_ground_truth":
        return "How can the scene still be interpreted under low-light conditions?"
    if challenge_type == "ir_as_visibility_ground_truth":
        return "Which part of the scene remains visible despite the lighting conditions?"
    if challenge_type == "ir_as_scene_structure_ground_truth":
        return "What kind of low-light environment is the movement occurring in?"
    if challenge_type == "rgb_as_color_ground_truth":
        return "What visible color clue helps identify the relevant object or sign?"
    if challenge_type == "rgb_as_text_or_label_ground_truth":
        return "What visible text, label, or number helps identify the relevant object?"
    if challenge_type == "rgb_as_sound_source_localization":
        return "Which visible object or action is most likely producing the heard sound?"
    if challenge_type == "rgb_as_object_ground_truth":
        return "Which visible object or action is relevant to the event being described?"
    return "What detail can be answered only by binding the scene context to the decisive evidence?"


def _single_modality_limits(bundle: dict[str, Any]) -> dict[str, str]:
    context = bundle["context_modality"]
    decisive = bundle["decisive_modality"]
    return {
        context: (
            f"{context} grounds the queried object, event, or scene, but it does not provide the answer-critical "
            f"evidence supplied by {decisive}."
        ),
        decisive: (
            f"{decisive} provides the answer-critical cue, but by itself it may not localize that cue to the "
            f"specific object, event, or scene grounded by {context}."
        ),
    }


def _why_multimodal(bundle: dict[str, Any]) -> str:
    context = bundle["context_modality"]
    decisive = bundle["decisive_modality"]
    return (
        f"{context} is needed to ground the queried object, event, or scene; "
        f"{decisive} is needed because it provides the decisive answer evidence."
    )


def _normalize_raw_qa(raw_item: dict[str, Any], bundle: dict[str, Any], index: int) -> dict[str, Any]:
    pair_label = "_".join(bundle["modalities"])
    qa_id = (
        f"{bundle['segment_id']}__{pair_label}__{bundle['context_modality']}_context__"
        f"{bundle['decisive_modality']}_decisive__{bundle['challenge_type']}__{index:03d}"
    )
    context = bundle["context_modality"]
    decisive = bundle["decisive_modality"]

    raw_evidence = raw_item.get("evidence_by_modality", {})
    evidence_by_modality: dict[str, dict[str, Any]] = {}
    for modality in bundle["modalities"]:
        source = bundle["evidence_by_modality"].get(modality, {})
        raw_modality_evidence = raw_evidence.get(modality) if isinstance(raw_evidence, dict) else None
        if isinstance(raw_modality_evidence, dict):
            evidence_text = str(raw_modality_evidence.get("evidence") or "").strip()
            source_sections = raw_modality_evidence.get("source_sections") or source.get("source_sections", [])
            evidence_ids = raw_modality_evidence.get("evidence_ids") or source.get("evidence_ids", [])
        else:
            evidence_text = str(raw_modality_evidence or "").strip()
            source_sections = source.get("source_sections", [])
            evidence_ids = source.get("evidence_ids", [])
        role = "context" if modality == context else "decisive"
        evidence_by_modality[modality] = {
            "role": role,
            "role_description": MODALITY_ROLE_LABELS[role],
            "source_sections": list(source_sections) if isinstance(source_sections, list) else [],
            "evidence_ids": list(evidence_ids) if isinstance(evidence_ids, list) else [],
            "evidence": evidence_text or source.get("evidence", ""),
        }

    return {
        "qa_id": qa_id,
        "segment_id": bundle.get("segment_id"),
        "source_prefix": bundle.get("source_prefix"),
        "side": bundle.get("side"),
        "task_label": bundle.get("task_label"),
        "start_seconds": bundle.get("start_seconds"),
        "end_seconds": bundle.get("end_seconds"),
        "start_timestamp": bundle.get("start_timestamp"),
        "end_timestamp": bundle.get("end_timestamp"),
        "modalities": bundle["modalities"],
        "qa_style": QA_STYLE,
        "capability": bundle["capability"],
        "context_modality": context,
        "decisive_modality": decisive,
        "challenge_type": bundle["challenge_type"],
        "question": str(raw_item.get("question") or _template_question(bundle)).strip(),
        "answer": str(raw_item.get("answer") or _infer_template_answer(bundle["challenge_type"], bundle)).strip(),
        "modality_roles": {
            context: MODALITY_ROLE_LABELS["context"],
            decisive: MODALITY_ROLE_LABELS["decisive"],
        },
        "evidence_by_modality": evidence_by_modality,
        "shared_object_or_event": str(raw_item.get("shared_object_or_event") or "").strip() or None,
        "context_constraint": str(raw_item.get("context_constraint") or "").strip() or None,
        "decisive_answer_cue": str(raw_item.get("decisive_answer_cue") or "").strip() or None,
        "why_context_is_needed": str(raw_item.get("why_context_is_needed") or "").strip() or None,
        "why_decisive_is_needed": str(raw_item.get("why_decisive_is_needed") or "").strip() or None,
        "why_decisive_alone_is_not_grounded": (
            str(raw_item.get("why_decisive_alone_is_not_grounded") or "").strip() or None
        ),
        "evidence_selection_rationale": str(raw_item.get("evidence_selection_rationale") or "").strip() or None,
        "why_multimodal": str(raw_item.get("why_multimodal") or _why_multimodal(bundle)).strip(),
        "single_modality_limits": raw_item.get("single_modality_limits")
        if isinstance(raw_item.get("single_modality_limits"), dict)
        else _single_modality_limits(bundle),
        "answerability_verification": {
            "status": "unverified",
            "schema_version": "context_decisive_v2",
            "context_only": None,
            "decisive_only_answer_cue": None,
            "decisive_only_grounding": None,
            "combined": None,
            "cross_modal_dependency": None,
            "verifier": None,
            "rationale": None,
        },
        "quality_control": {
            "question_mentions_modalities": _question_mentions_modalities(str(raw_item.get("question") or _template_question(bundle))),
            "context_grounding_required": True,
            "decisive_evidence_required": True,
            "cross_modal_binding_required": True,
            "ground_truth_confidence": float(raw_item.get("ground_truth_confidence") or 0.72),
            "validation_errors": [],
        },
        "selection": {
            "candidate_status": "candidate",
            "quality_status": "unreviewed",
            "benchmark_keep": None,
            "strict_keep": None,
            "keep": None,
            "review_notes": None,
        },
        "generation": {
            "mode": str(raw_item.get("_generation_mode") or "template"),
            "model_name": str(raw_item.get("_model_name") or ""),
            "source": "segmented_normalized_evidence_units",
        },
    }


def _generate_template_qa(bundle: dict[str, Any], item_index: int) -> dict[str, Any]:
    raw_item = {
        "_generation_mode": "template",
        "question": _template_question(bundle),
        "answer": _infer_template_answer(bundle["challenge_type"], bundle),
        "evidence_by_modality": bundle["evidence_by_modality"],
        "why_multimodal": _why_multimodal(bundle),
        "single_modality_limits": _single_modality_limits(bundle),
        "ground_truth_confidence": 0.72,
    }
    return _normalize_raw_qa(raw_item, bundle, item_index)


def _is_quota_or_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(token in text for token in ("quota", "rate limit", "rate_limit", "429", "resource_exhausted"))


async def _call_multimodal_gemini_with_retry(
    client,
    contents: list,
    max_retries: int = 3,
    model_name: str = MULTIMODAL_QA_MODEL_NAME,
) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=model_name,
                contents=contents,
            )
            return response.text
        except Exception as exc:
            if attempt == max_retries:
                raise
            wait_seconds = 30 * attempt if _is_quota_or_rate_limit_error(exc) else 2 * attempt
            print(
                f"    Implicit multimodal QA Gemini call failed on attempt {attempt}/{max_retries}; "
                f"retrying in {wait_seconds}s: {exc}"
            )
            await asyncio.sleep(wait_seconds)
    raise RuntimeError("Implicit multimodal QA Gemini call failed")


def _parse_json_object_response(text: str) -> dict[str, Any]:
    if not text:
        raise ValueError("Empty Gemini response")
    cleaned_text = text.strip()
    cleaned_text = re.sub(r"^```(?:json)?\s*", "", cleaned_text, flags=re.I)
    cleaned_text = re.sub(r"\s*```$", "", cleaned_text, flags=re.I)
    match = re.search(r"\{.*\}", cleaned_text, flags=re.S)
    if not match:
        raise ValueError("No JSON object found in Gemini response")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Gemini response must be a JSON object")
    return parsed


def _build_gemini_prompt(bundle: dict[str, Any]) -> str:
    context = bundle["context_modality"]
    decisive = bundle["decisive_modality"]
    prompt_payload = {
        "segment": {
            "segment_id": bundle.get("segment_id"),
            "side": bundle.get("side"),
            "task_label": bundle.get("task_label"),
            "start_seconds": bundle.get("start_seconds"),
            "end_seconds": bundle.get("end_seconds"),
        },
        "modalities": bundle["modalities"],
        "qa_style": QA_STYLE,
        "context_modality": context,
        "decisive_modality": decisive,
        "challenge_type": bundle["challenge_type"],
        "rule_selected_evidence_by_modality": bundle["evidence_by_modality"],
        "candidate_evidence_by_modality": bundle.get("evidence_candidates_by_modality", {}),
    }
    return "\n".join(
        [
            "You generate implicit cross-modal benchmark QA from existing evidence.",
            "First select the smallest relevant evidence set from candidate_evidence_by_modality.",
            "The selected context evidence and decisive evidence must refer to the same object, action, event, or scene.",
            "Return selected evidence_ids exactly as provided.",
            "Use the context modality to ground the queried object, event, or scene.",
            "Use the decisive modality to determine the answer.",
            "The question must include a target constraint that is grounded by the context evidence.",
            "The answer must rely on an answer cue supplied by the decisive evidence.",
            "Do not ask a question whose target and answer are both fully specified by decisive evidence alone.",
            "Prefer questions where decisive evidence gives the cue, but context evidence is needed to bind the cue to the asked target.",
            "Do not write questions like 'what RGB and audio evidence together...'.",
            "Do not explicitly mention modality names in the question unless there is no natural alternative.",
            "The question should sound like a natural question about the scene.",
            "The answer should be concise and supported by the decisive modality while grounded by the context modality.",
            "Do not invent facts not present in the evidence.",
            "",
            "Return ONLY valid JSON with this exact structure:",
            "{",
            '  "qa_items": [',
            "    {",
            '      "question": "...",',
            '      "answer": "...",',
            f'      "context_modality": "{context}",',
            f'      "decisive_modality": "{decisive}",',
            f'      "challenge_type": "{bundle["challenge_type"]}",',
            '      "evidence_by_modality": {',
            f'        "{context}": {{"source_sections": ["..."], "evidence_ids": ["..."], "evidence": "..."}},',
            f'        "{decisive}": {{"source_sections": ["..."], "evidence_ids": ["..."], "evidence": "..."}}',
            "      },",
            '      "shared_object_or_event": "...",',
            '      "context_constraint": "The target constraint in the question that comes from context evidence.",',
            '      "decisive_answer_cue": "The answer-critical cue that comes from decisive evidence.",',
            '      "why_context_is_needed": "...",',
            '      "why_decisive_is_needed": "...",',
            '      "why_decisive_alone_is_not_grounded": "...",',
            '      "evidence_selection_rationale": "...",',
            '      "why_multimodal": "...",',
            '      "single_modality_limits": {',
            f'        "{context}": "...",',
            f'        "{decisive}": "..."',
            "      },",
            '      "ground_truth_confidence": 0.0',
            "    }",
            "  ]",
            "}",
            "",
            "Evidence bundle:",
            json.dumps(prompt_payload, indent=2, ensure_ascii=False),
        ]
    )


def _build_gemini_batch_prompt(tasks: list[dict[str, Any]]) -> str:
    if not tasks:
        raise ValueError("Cannot build Gemini batch prompt without tasks")

    segment = tasks[0]["bundle"]
    bundle_payloads = []
    for task in tasks:
        bundle = task["bundle"]
        context = bundle["context_modality"]
        decisive = bundle["decisive_modality"]
        bundle_payloads.append(
            {
                "bundle_id": _planned_qa_id(bundle, task["index"]),
                "bundle_index": task["index"],
                "modalities": bundle["modalities"],
                "context_modality": context,
                "decisive_modality": decisive,
                "challenge_type": bundle["challenge_type"],
                "rule_selected_evidence_by_modality": bundle["evidence_by_modality"],
                "candidate_evidence_by_modality": bundle.get("evidence_candidates_by_modality", {}),
            }
        )

    prompt_payload = {
        "segment": {
            "segment_id": segment.get("segment_id"),
            "side": segment.get("side"),
            "task_label": segment.get("task_label"),
            "start_seconds": segment.get("start_seconds"),
            "end_seconds": segment.get("end_seconds"),
        },
        "qa_style": QA_STYLE,
        "bundles": bundle_payloads,
    }
    return "\n".join(
        [
            "You generate implicit cross-modal benchmark QA from existing evidence.",
            "You will receive multiple evidence bundles from the same segment.",
            "Generate exactly one QA item for each bundle.",
            "Each returned QA item must copy the corresponding bundle_id exactly.",
            "For each bundle, first select the smallest relevant evidence set from candidate_evidence_by_modality.",
            "The selected context evidence and decisive evidence must refer to the same object, action, event, or scene.",
            "Return selected evidence_ids exactly as provided.",
            "Use the context modality to ground the queried object, event, or scene.",
            "Use the decisive modality to determine the answer.",
            "Each question must include a target constraint that is grounded by the context evidence.",
            "Each answer must rely on an answer cue supplied by the decisive evidence.",
            "Do not ask questions whose target and answer are both fully specified by decisive evidence alone.",
            "Prefer questions where decisive evidence gives the cue, but context evidence is needed to bind the cue to the asked target.",
            "Do not write questions like 'what RGB and audio evidence together...'.",
            "Do not explicitly mention modality names in the question unless there is no natural alternative.",
            "The question should sound like a natural question about the scene.",
            "The answer should be concise and supported by the decisive modality while grounded by the context modality.",
            "Do not invent facts not present in the evidence.",
            "",
            "Return ONLY valid JSON with this exact structure:",
            "{",
            '  "qa_items": [',
            "    {",
            '      "bundle_id": "...",',
            '      "question": "...",',
            '      "answer": "...",',
            '      "context_modality": "...",',
            '      "decisive_modality": "...",',
            '      "challenge_type": "...",',
            '      "evidence_by_modality": {',
            '        "modality_name": {"source_sections": ["..."], "evidence_ids": ["..."], "evidence": "..."}',
            "      },",
            '      "shared_object_or_event": "...",',
            '      "context_constraint": "The target constraint in the question that comes from context evidence.",',
            '      "decisive_answer_cue": "The answer-critical cue that comes from decisive evidence.",',
            '      "why_context_is_needed": "...",',
            '      "why_decisive_is_needed": "...",',
            '      "why_decisive_alone_is_not_grounded": "...",',
            '      "evidence_selection_rationale": "...",',
            '      "why_multimodal": "...",',
            '      "single_modality_limits": {',
            '        "context_modality_name": "...",',
            '        "decisive_modality_name": "..."',
            "      },",
            '      "ground_truth_confidence": 0.0',
            "    }",
            "  ]",
            "}",
            "",
            "Evidence bundles:",
            json.dumps(prompt_payload, indent=2, ensure_ascii=False),
        ]
    )


async def _generate_gemini_qa(
    client,
    bundle: dict[str, Any],
    item_index: int,
    max_retries: int,
    model_name: str,
) -> list[dict[str, Any]]:
    response_text = await _call_multimodal_gemini_with_retry(
        client,
        [_build_gemini_prompt(bundle)],
        max_retries=max_retries,
        model_name=model_name,
    )
    parsed = _parse_json_object_response(response_text)
    raw_items = parsed.get("qa_items", [])
    if not isinstance(raw_items, list):
        raise ValueError("Gemini response qa_items must be a list")
    normalized = []
    for raw_item in raw_items[:1]:
        if not isinstance(raw_item, dict):
            continue
        raw_item["_generation_mode"] = "gemini"
        raw_item["_model_name"] = model_name
        normalized.append(_normalize_raw_qa(raw_item, bundle, item_index))
    return normalized


async def _generate_gemini_batch_qa(
    client,
    tasks: list[dict[str, Any]],
    max_retries: int,
    model_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    response_text = await _call_multimodal_gemini_with_retry(
        client,
        [_build_gemini_batch_prompt(tasks)],
        max_retries=max_retries,
        model_name=model_name,
    )
    parsed = _parse_json_object_response(response_text)
    raw_items = parsed.get("qa_items", [])
    if not isinstance(raw_items, list):
        raise ValueError("Gemini response qa_items must be a list")

    task_by_bundle_id = {_planned_qa_id(task["bundle"], task["index"]): task for task in tasks}
    normalized: list[dict[str, Any]] = []
    seen_bundle_ids: set[str] = set()
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue
        bundle_id = str(raw_item.get("bundle_id") or "").strip()
        task = task_by_bundle_id.get(bundle_id)
        if task is None or bundle_id in seen_bundle_ids:
            continue
        raw_item["_generation_mode"] = "gemini"
        raw_item["_model_name"] = model_name
        normalized.append(_normalize_raw_qa(raw_item, task["bundle"], task["index"]))
        seen_bundle_ids.add(bundle_id)

    missing_tasks = [task for bundle_id, task in task_by_bundle_id.items() if bundle_id not in seen_bundle_ids]
    return normalized, missing_tasks


def _question_mentions_modalities(question: str) -> bool:
    text = question.lower()
    patterns = (
        r"\brgb\b",
        r"\baudio\b",
        r"\bevent\s+stream\b",
        r"\bdepth\b",
        r"\bir\b",
        r"\binfrared\b",
        r"\bmodality\b",
        r"\bmodalities\b",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def _validate_qa_item(item: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    modalities = item.get("modalities")
    if not isinstance(modalities, list) or len(modalities) != 2:
        errors.append("modalities must contain exactly two entries")
        return errors

    pair = tuple(str(modality).lower() for modality in modalities)
    canonical_pair = _canonical_pair(pair)  # type: ignore[arg-type]
    side = str(item.get("side") or "").lower()
    if side == "night" and canonical_pair not in NIGHT_COMBINATIONS:
        errors.append(f"night QA uses unsupported modality pair: {pair}")
    if side == "day" and canonical_pair not in DAY_COMBINATIONS:
        errors.append(f"day QA uses unsupported modality pair: {pair}")

    context = str(item.get("context_modality") or "").lower()
    decisive = str(item.get("decisive_modality") or "").lower()
    if context not in pair:
        errors.append("context_modality must be one of the pair modalities")
    if decisive not in pair:
        errors.append("decisive_modality must be one of the pair modalities")
    if context == decisive:
        errors.append("context_modality and decisive_modality must be different")

    challenge_type = str(item.get("challenge_type") or "")
    expected_decisive = CHALLENGE_TO_DECISIVE.get(challenge_type)
    if expected_decisive and expected_decisive != decisive:
        errors.append(f"challenge_type {challenge_type!r} expects decisive modality {expected_decisive!r}")

    question = str(item.get("question") or "").strip()
    answer = str(item.get("answer") or "").strip()
    if not question:
        errors.append("question is empty")
    if not answer:
        errors.append("answer is empty")
    if _question_mentions_modalities(question):
        errors.append("question explicitly mentions modality names")

    evidence_by_modality = item.get("evidence_by_modality")
    if not isinstance(evidence_by_modality, dict):
        errors.append("evidence_by_modality must be an object")
    else:
        for modality in pair:
            evidence = evidence_by_modality.get(modality)
            if not isinstance(evidence, dict) or not str(evidence.get("evidence") or "").strip():
                errors.append(f"missing evidence for modality: {modality}")
        if isinstance(evidence_by_modality.get(context), dict) and evidence_by_modality[context].get("role") != "context":
            errors.append("context evidence role is incorrect")
        if isinstance(evidence_by_modality.get(decisive), dict) and evidence_by_modality[decisive].get("role") != "decisive":
            errors.append("decisive evidence role is incorrect")

    if not str(item.get("why_multimodal") or "").strip():
        errors.append("why_multimodal is empty")
    if item.get("qa_style") != QA_STYLE:
        errors.append(f"qa_style must be {QA_STYLE}")
    return errors


def _attach_validation(item: dict[str, Any]) -> dict[str, Any]:
    errors = _validate_qa_item(item)
    item.setdefault("quality_control", {})["validation_errors"] = errors
    item["quality_control"]["validation_status"] = "failed" if errors else "passed"
    return item


def _counter_to_dict(counter: Counter) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _build_distribution(qa_items: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    by_side: Counter = Counter()
    by_pair: Counter = Counter()
    by_direction: Counter = Counter()
    by_context_modality: Counter = Counter()
    by_decisive_modality: Counter = Counter()
    by_challenge_type: Counter = Counter()
    by_validation_status: Counter = Counter()
    by_task_label: Counter = Counter()

    for item in qa_items:
        side = str(item.get("side") or "unknown")
        modalities = item.get("modalities") if isinstance(item.get("modalities"), list) else []
        pair = "+".join(str(modality) for modality in modalities) if modalities else "unknown"
        context = str(item.get("context_modality") or "unknown")
        decisive = str(item.get("decisive_modality") or "unknown")
        challenge_type = str(item.get("challenge_type") or "unknown")
        validation_status = str(item.get("quality_control", {}).get("validation_status") or "unknown")
        task_label = str(item.get("task_label") or "unknown")

        by_side[side] += 1
        by_pair[pair] += 1
        by_direction[f"{context}->{decisive}"] += 1
        by_context_modality[context] += 1
        by_decisive_modality[decisive] += 1
        by_challenge_type[challenge_type] += 1
        by_validation_status[validation_status] += 1
        by_task_label[task_label] += 1

    return {
        "by_side": _counter_to_dict(by_side),
        "by_pair": _counter_to_dict(by_pair),
        "by_direction": _counter_to_dict(by_direction),
        "by_context_modality": _counter_to_dict(by_context_modality),
        "by_decisive_modality": _counter_to_dict(by_decisive_modality),
        "by_challenge_type": _counter_to_dict(by_challenge_type),
        "by_validation_status": _counter_to_dict(by_validation_status),
        "by_task_label": _counter_to_dict(by_task_label),
    }


def _build_output_payload(
    input_path: Path,
    generation_mode: str,
    qa_items: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    gemini_calls: int,
    planned_total: int,
    resumed_items: int,
    completed_bundles: int,
    max_concurrent_calls: int,
    gemini_batch_scope: str,
    max_bundles_per_gemini_call: int,
    model_name: str,
) -> dict[str, Any]:
    passed = sum(1 for item in qa_items if item.get("quality_control", {}).get("validation_status") == "passed")
    return {
        "metadata": {
            "source_file": str(input_path),
            "generation_mode": generation_mode,
            "qa_style": QA_STYLE,
            "generation_policy": {
                "candidate_generation": "one_candidate_per_available_role_direction_and_supported_challenge_type",
                "evidence_selection": {
                    "template": "rule_selected_evidence_from_section_preferences",
                    "gemini": "rule_retrieved_candidates_then_gemini_selects_related_evidence_ids",
                },
                "challenge_selection": "all_supported_challenge_types",
                "unsupported_challenge_types": "skipped",
                "direction_sampling": "balanced_candidate_coverage",
                "direction_weighting": False,
                "direction_quotas": False,
                "final_resampling": False,
                "selection_policy": "quality_filtering_only",
                "kept_after_filtering": "all_passing_items",
                "gemini_batch_scope": gemini_batch_scope,
                "max_bundles_per_gemini_call": max_bundles_per_gemini_call,
            },
            "benchmark_design": {
                "night": [list(pair) for pair in NIGHT_COMBINATIONS],
                "day": [list(pair) for pair in DAY_COMBINATIONS],
                "role_specs": {"__".join(pair): spec for pair, spec in PAIR_ROLE_SPECS.items()},
            },
            "run_status": {
                "planned_bundles": planned_total,
                "completed_bundles": completed_bundles,
                "resumed_items": resumed_items,
                "max_concurrent_calls": max_concurrent_calls,
                "gemini_batch_scope": gemini_batch_scope,
                "max_bundles_per_gemini_call": max_bundles_per_gemini_call,
            },
            "model_name": model_name,
            "total_qa_items": len(qa_items),
            "passed_validation": passed,
            "failed_validation": len(qa_items) - passed,
            "skipped_pairs": len(skipped),
            "gemini_calls": gemini_calls,
            "distribution": _build_distribution(qa_items),
        },
        "qa_items": sorted(qa_items, key=lambda item: str(item.get("qa_id") or "")),
        "skipped": skipped,
    }


def _load_resume_items(output_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not output_path.exists():
        return [], []
    try:
        existing = _load_json(output_path)
    except Exception as exc:
        print(f"WARNING: Could not read existing output for resume: {exc}")
        return [], []

    qa_items = existing.get("qa_items", [])
    skipped = existing.get("skipped", [])
    if not isinstance(qa_items, list):
        qa_items = []
    if not isinstance(skipped, list):
        skipped = []
    qa_items = [item for item in qa_items if isinstance(item, dict) and item.get("qa_id")]
    skipped = [item for item in skipped if isinstance(item, dict)]
    return qa_items, skipped


def _planned_qa_id(bundle: dict[str, Any], index: int) -> str:
    pair_label = "_".join(bundle["modalities"])
    return (
        f"{bundle['segment_id']}__{pair_label}__{bundle['context_modality']}_context__"
        f"{bundle['decisive_modality']}_decisive__{bundle['challenge_type']}__{index:03d}"
    )


def _chunked(items: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def _build_gemini_task_batches(
    planned_tasks: list[dict[str, Any]],
    existing_ids: set[str],
    batch_scope: str,
    max_bundles_per_call: int,
) -> list[list[dict[str, Any]]]:
    pending_tasks = [
        task
        for task in planned_tasks
        if _planned_qa_id(task["bundle"], task["index"]) not in existing_ids
    ]
    if batch_scope == "bundle":
        return [[task] for task in pending_tasks]

    grouped: dict[str, list[dict[str, Any]]] = {}
    for task in pending_tasks:
        grouped.setdefault(str(task["segment_id"]), []).append(task)

    batches: list[list[dict[str, Any]]] = []
    for task in planned_tasks:
        segment_id = str(task["segment_id"])
        if segment_id not in grouped:
            continue
        batches.extend(_chunked(grouped.pop(segment_id), max_bundles_per_call))
    return batches


async def _run_multimodal_qa_pipeline_async(
    input_path: Path,
    output_path: Path,
    generation_mode: str,
    test_mode: bool,
    delay_between_calls: int,
    max_concurrent_calls: int,
    max_retries: int,
    resume: bool,
    checkpoint_every: int,
    gemini_batch_scope: str,
    max_bundles_per_gemini_call: int,
    model_name: str,
) -> Path:
    data = _load_json(input_path)
    client = create_gemini_client() if generation_mode == "gemini" else None

    qa_items: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    segment_items = list(data.items())
    if test_mode:
        segment_items = segment_items[:2]

    planned_bundles_by_segment: dict[str, int] = {}
    planned_tasks: list[dict[str, Any]] = []
    planned_total = 0
    planning_skipped: list[dict[str, Any]] = []
    for segment_index, (segment_id, segment) in enumerate(segment_items, start=1):
        if not isinstance(segment, dict):
            planning_skipped.append({"segment_id": segment_id, "reason": "segment is not an object"})
            continue
        evidence_by_modality = _collect_evidence_by_modality(segment)
        segment_bundle_count = 0
        for pair in _segment_pairs(segment):
            bundles = _build_role_bundles(segment, evidence_by_modality, pair)
            if not bundles:
                planning_skipped.append({"segment_id": segment_id, "modalities": list(pair), "reason": "missing role evidence"})
                continue
            for bundle in bundles:
                planned_total += 1
                segment_bundle_count += 1
                planned_tasks.append(
                    {
                        "index": planned_total,
                        "segment_index": segment_index,
                        "segment_count": len(segment_items),
                        "segment_id": str(segment_id),
                        "pair": pair,
                        "bundle": bundle,
                    }
                )
        planned_bundles_by_segment[str(segment_id)] = segment_bundle_count

    if resume:
        resumed_qa_items, resumed_skipped = _load_resume_items(output_path)
        qa_items.extend(resumed_qa_items)
        skipped.extend(resumed_skipped)
    skipped.extend(planning_skipped)
    existing_ids = {str(item.get("qa_id")) for item in qa_items if item.get("qa_id")}
    resumed_items = len(existing_ids)

    print(
        f"Generating implicit multimodal QA: {len(segment_items)} segment(s), "
        f"~{planned_total} candidate bundle(s), mode={generation_mode}."
    )
    gemini_batches: list[list[dict[str, Any]]] = []
    if generation_mode == "gemini":
        gemini_batches = _build_gemini_task_batches(
            planned_tasks=planned_tasks,
            existing_ids=existing_ids,
            batch_scope=gemini_batch_scope,
            max_bundles_per_call=max_bundles_per_gemini_call,
        )
    if generation_mode == "gemini":
        print(
            f"Gemini generation will make up to {len(gemini_batches)} call(s) "
            f"for {sum(len(batch) for batch in gemini_batches)} pending bundle(s), "
            f"batch_scope={gemini_batch_scope}, max_bundles_per_call={max_bundles_per_gemini_call}, "
            f"model={model_name}, "
            f"with {delay_between_calls}s delay between calls and "
            f"max_concurrent_calls={max_concurrent_calls}."
        )
    if resumed_items:
        print(f"Resume enabled: found {resumed_items} existing QA item(s), matching bundles will be skipped.")

    gemini_calls = 0
    completed_bundles = sum(1 for task in planned_tasks if _planned_qa_id(task["bundle"], task["index"]) in existing_ids)
    checkpoint_counter = 0
    save_lock = asyncio.Lock()
    qa_lock = asyncio.Lock()
    segment_announced: set[str] = set()

    async def save_checkpoint() -> None:
        output = _build_output_payload(
            input_path=input_path,
            generation_mode=generation_mode,
            qa_items=qa_items,
            skipped=skipped,
            gemini_calls=gemini_calls,
            planned_total=planned_total,
            resumed_items=resumed_items,
            completed_bundles=completed_bundles,
            max_concurrent_calls=max_concurrent_calls,
            gemini_batch_scope=gemini_batch_scope,
            max_bundles_per_gemini_call=max_bundles_per_gemini_call,
            model_name=model_name,
        )
        _save_json(output, output_path)

    async def record_generated(generated: list[dict[str, Any]]) -> None:
        nonlocal checkpoint_counter
        async with qa_lock:
            for item in generated:
                qa_id = str(item.get("qa_id") or "")
                if qa_id and qa_id not in existing_ids:
                    qa_items.append(_attach_validation(item))
                    existing_ids.add(qa_id)
            checkpoint_counter += 1
            should_save = generation_mode == "gemini" and checkpoint_every > 0 and checkpoint_counter >= checkpoint_every
            if should_save:
                checkpoint_counter = 0
        if should_save:
            async with save_lock:
                await save_checkpoint()

    async def record_skip(task: dict[str, Any], exc: Exception) -> None:
        nonlocal checkpoint_counter
        bundle = task["bundle"]
        pair = task["pair"]
        async with qa_lock:
            skipped.append(
                {
                    "segment_id": task["segment_id"],
                    "modalities": list(pair),
                    "context_modality": bundle.get("context_modality"),
                    "decisive_modality": bundle.get("decisive_modality"),
                    "challenge_type": bundle.get("challenge_type"),
                    "reason": str(exc),
                }
            )
            checkpoint_counter += 1
            should_save = generation_mode == "gemini" and checkpoint_every > 0 and checkpoint_counter >= checkpoint_every
            if should_save:
                checkpoint_counter = 0
        if should_save:
            async with save_lock:
                await save_checkpoint()

    async def run_task(task: dict[str, Any]) -> None:
        nonlocal gemini_calls, completed_bundles
        bundle = task["bundle"]
        planned_id = _planned_qa_id(bundle, task["index"])
        segment_id = task["segment_id"]
        segment_key = f"{task['segment_index']}::{segment_id}"
        if segment_key not in segment_announced:
            segment_announced.add(segment_key)
            segment_planned = planned_bundles_by_segment.get(str(segment_id), 0)
            print(
                f"Segment [{task['segment_index']}/{task['segment_count']}] {segment_id}: "
                f"{segment_planned} candidate bundle(s)."
            )
        if planned_id in existing_ids:
            print(f"  Bundle [{task['index']}/{planned_total}] already exists, skipping: {planned_id}")
            return

        pair_label = "+".join(task["pair"])
        direction = f"{bundle.get('context_modality')}->{bundle.get('decisive_modality')}"
        print(
            f"  Bundle [{task['index']}/{planned_total}] "
            f"{pair_label} {direction} {bundle.get('challenge_type')}"
        )
        try:
            if generation_mode == "gemini":
                assert client is not None
                print(
                    f"    Calling Gemini for bundle [{task['index']}/{planned_total}] "
                    f"segment {segment_id}, pair {pair_label}, direction {direction}..."
                )
                generated = await _generate_gemini_qa(
                    client,
                    bundle,
                    task["index"],
                    max_retries=max_retries,
                    model_name=model_name,
                )
                async with qa_lock:
                    gemini_calls += 1
                    completed_bundles += 1
                    current_calls = gemini_calls
                print(f"    Gemini call complete [{current_calls}/{planned_total}].")
                if delay_between_calls > 0:
                    await asyncio.sleep(delay_between_calls)
            else:
                generated = [_generate_template_qa(bundle, task["index"])]
                async with qa_lock:
                    completed_bundles += 1
            await record_generated(generated)
        except Exception as exc:
            async with qa_lock:
                completed_bundles += 1
            print(f"WARNING: Failed QA generation for {segment_id} {task['pair']}: {exc}")
            await record_skip(task, exc)

    async def run_batch(batch: list[dict[str, Any]], batch_index: int, batch_count: int) -> None:
        nonlocal gemini_calls, completed_bundles
        if not batch:
            return
        segment_id = batch[0]["segment_id"]
        segment_key = f"{batch[0]['segment_index']}::{segment_id}"
        if segment_key not in segment_announced:
            segment_announced.add(segment_key)
            segment_planned = planned_bundles_by_segment.get(str(segment_id), 0)
            print(
                f"Segment [{batch[0]['segment_index']}/{batch[0]['segment_count']}] {segment_id}: "
                f"{segment_planned} candidate bundle(s)."
            )

        first_index = batch[0]["index"]
        last_index = batch[-1]["index"]
        print(
            f"  Gemini batch [{batch_index}/{batch_count}] segment {segment_id}: "
            f"{len(batch)} bundle(s), bundle index range [{first_index}-{last_index}]"
        )
        try:
            assert client is not None
            generated, missing_tasks = await _generate_gemini_batch_qa(
                client,
                batch,
                max_retries=max_retries,
                model_name=model_name,
            )
            async with qa_lock:
                gemini_calls += 1
                completed_bundles += len(generated)
                current_calls = gemini_calls
            print(
                f"    Gemini batch complete [{current_calls}/{batch_count}], "
                f"generated {len(generated)}/{len(batch)} QA item(s)."
            )
            await record_generated(generated)
            if missing_tasks:
                print(
                    f"    Gemini batch omitted {len(missing_tasks)} bundle(s); "
                    "falling back to single-bundle generation for those items."
                )
                for missing_task in missing_tasks:
                    await run_task(missing_task)
            if delay_between_calls > 0:
                await asyncio.sleep(delay_between_calls)
        except Exception as exc:
            print(
                f"WARNING: Failed Gemini batch for segment {segment_id}: {exc}. "
                "Falling back to single-bundle generation for this batch."
            )
            for task in batch:
                await run_task(task)

    if generation_mode == "gemini" and gemini_batch_scope == "segment":
        if max_concurrent_calls > 1:
            semaphore = asyncio.Semaphore(max_concurrent_calls)

            async def bounded_batch(batch_index: int, batch: list[dict[str, Any]]) -> None:
                async with semaphore:
                    await run_batch(batch, batch_index, len(gemini_batches))

            await asyncio.gather(
                *(bounded_batch(batch_index, batch) for batch_index, batch in enumerate(gemini_batches, start=1))
            )
        else:
            for batch_index, batch in enumerate(gemini_batches, start=1):
                await run_batch(batch, batch_index, len(gemini_batches))
    elif generation_mode == "gemini" and max_concurrent_calls > 1:
        semaphore = asyncio.Semaphore(max_concurrent_calls)

        async def bounded_run(task: dict[str, Any]) -> None:
            async with semaphore:
                await run_task(task)

        await asyncio.gather(*(bounded_run(task) for task in planned_tasks))
    else:
        for task in planned_tasks:
            await run_task(task)

    await save_checkpoint()
    return output_path


def run_multimodal_qa_pipeline(
    input_path: Path | str = "segmented_normalized_evidence_units.json",
    output_path: Path | str = "outputs/implicit_multimodal_qa_candidates.json",
    generation_mode: str = "template",
    test_mode: bool = False,
    delay_between_calls: int | None = None,
    max_concurrent_calls: int = 1,
    max_retries: int = 3,
    resume: bool = True,
    checkpoint_every: int = 1,
    gemini_batch_scope: str = DEFAULT_GEMINI_BATCH_SCOPE,
    max_bundles_per_gemini_call: int = DEFAULT_MAX_BUNDLES_PER_GEMINI_CALL,
    model_name: str = MULTIMODAL_QA_MODEL_NAME,
) -> Path:
    """Generate implicit cross-modal QA candidates from normalized evidence units."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    generation_mode = generation_mode.strip().lower()
    if generation_mode not in {"template", "gemini"}:
        raise ValueError("generation_mode must be 'template' or 'gemini'")
    resolved_batch_scope = gemini_batch_scope.strip().lower()
    if resolved_batch_scope not in {"bundle", "segment"}:
        raise ValueError("gemini_batch_scope must be 'bundle' or 'segment'")
    resolved_delay = DEFAULT_API_DELAY_SECONDS if generation_mode == "gemini" else 0
    if delay_between_calls is not None:
        resolved_delay = delay_between_calls
    resolved_concurrency = max(1, int(max_concurrent_calls))
    resolved_retries = max(1, int(max_retries))
    resolved_checkpoint_every = max(1, int(checkpoint_every))
    resolved_max_bundles = max(1, int(max_bundles_per_gemini_call))
    resolved_model_name = model_name.strip()
    if not resolved_model_name:
        raise ValueError("model_name must not be empty")
    return asyncio.run(
        _run_multimodal_qa_pipeline_async(
            input_path=input_path,
            output_path=output_path,
            generation_mode=generation_mode,
            test_mode=test_mode,
            delay_between_calls=resolved_delay,
            max_concurrent_calls=resolved_concurrency,
            max_retries=resolved_retries,
            resume=resume,
            checkpoint_every=resolved_checkpoint_every,
            gemini_batch_scope=resolved_batch_scope,
            max_bundles_per_gemini_call=resolved_max_bundles,
            model_name=resolved_model_name,
        )
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="segmented_normalized_evidence_units.json", help="Input normalized evidence JSON.")
    parser.add_argument("--output", default="outputs/implicit_multimodal_qa_candidates.json", help="Output QA candidates JSON.")
    parser.add_argument(
        "--generation-mode",
        choices=("template", "gemini"),
        default="template",
        help="Use deterministic templates or Gemini generation.",
    )
    parser.add_argument("--test-mode", action="store_true", help="Only process the first two segments.")
    parser.add_argument("--delay-between-calls", type=int, default=None, help="Delay between Gemini calls in seconds.")
    parser.add_argument(
        "--max-concurrent-calls",
        type=int,
        default=1,
        help="Maximum concurrent Gemini calls. Start with 2-4 if your quota allows it.",
    )
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retries for each Gemini call.")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not reuse QA items already present in the output file.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Save progress after this many Gemini bundle results.",
    )
    parser.add_argument(
        "--gemini-batch-scope",
        choices=("bundle", "segment"),
        default=DEFAULT_GEMINI_BATCH_SCOPE,
        help="Gemini batching scope. 'segment' sends multiple bundles from one segment in one call.",
    )
    parser.add_argument(
        "--max-bundles-per-gemini-call",
        type=int,
        default=DEFAULT_MAX_BUNDLES_PER_GEMINI_CALL,
        help="Maximum number of bundles in one Gemini call when --gemini-batch-scope segment is used.",
    )
    parser.add_argument(
        "--model-name",
        default=MULTIMODAL_QA_MODEL_NAME,
        help="Gemini model name used for QA generation.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    output_path = run_multimodal_qa_pipeline(
        input_path=args.input,
        output_path=args.output,
        generation_mode=args.generation_mode,
        test_mode=args.test_mode,
        delay_between_calls=args.delay_between_calls,
        max_concurrent_calls=args.max_concurrent_calls,
        max_retries=args.max_retries,
        resume=not args.no_resume,
        checkpoint_every=args.checkpoint_every,
        gemini_batch_scope=args.gemini_batch_scope,
        max_bundles_per_gemini_call=args.max_bundles_per_gemini_call,
        model_name=args.model_name,
    )
    print(f"Wrote implicit multimodal QA candidates to {output_path}")


if __name__ == "__main__":
    main()
