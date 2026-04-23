"""
Late fusion utilities for combining modality-specific captions.

This module reads the existing per-modality QA result files, aligns entries by
sample, and produces structured fused scene descriptions plus a final unified
caption for each sample.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any


DEFAULT_MODALITY_FILES = {
    "rgb": "rgb_qa_results.json",
    "ir": "ir_qa_results.json",
    "event": "event_qa_results.json",
    "audio": "audio_qa_results.json",
    "depth": "depth_qa_results.json",
}

OUTPUT_SECTIONS = (
    "scene_overview",
    "visible_objects_and_layout",
    "motion_and_event_cues",
    "audio_cues",
    "cross_modal_details",
    "final_unified_caption",
)

SECTION_LABELS = {
    "scene_overview": "Scene Overview",
    "visible_objects_and_layout": "Visible Objects and Layout",
    "motion_and_event_cues": "Motion and Event Cues",
    "audio_cues": "Audio Cues",
    "cross_modal_details": "Cross-Modal Details",
    "final_unified_caption": "Final Unified Caption",
}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "in", "into", "is", "it", "its", "my", "of", "on", "or", "that",
    "the", "their", "there", "these", "this", "to", "up", "with", "while",
    "no", "not", "than", "then", "than", "me", "i", "you", "your", "our",
    "we", "they", "them", "he", "she", "his", "her", "was", "were", "will",
    "would", "can", "could", "should", "about", "after", "before", "during",
    "over", "under", "right", "left", "center", "middle", "side", "toward",
    "towards", "throughout", "through", "around", "between", "also", "only",
    "mostly", "very", "more", "most", "all", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine", "ten", "frame", "video", "scene",
    "sequence", "shows", "show", "visible", "appears", "appear", "present",
    "located", "positioned", "detected", "observed", "indicates", "suggest",
    "suggests", "likely", "roughly", "approximately", "object", "objects",
    "area", "areas", "region", "regions", "side", "sides", "part", "parts",
    "start", "end", "begins", "begin", "ends", "contains", "containing",
    "throughout", "overall", "main", "primary", "final", "initial", "clear",
    "clearly", "distinct", "several", "single", "small", "large",
}

SECTION_KEYWORDS = {
    "scene_overview": {
        "kitchen", "indoor", "outdoor", "counter", "workspace", "room", "office",
        "environment", "setting", "preparation", "cooking", "food", "domestic",
    },
    "visible_objects_and_layout": {
        "board", "cutting", "knife", "peeler", "carrot", "bowl", "laptop",
        "stovetop", "countertop", "counter", "power", "strip", "left", "right",
        "center", "layout", "object", "objects", "spatial", "position",
    },
    "motion_and_event_cues": {
        "move", "motion", "moving", "dynamic", "enter", "lift", "reach", "pick",
        "grasp", "peel", "peeling", "contact", "activity", "interaction",
        "start", "begin", "hands", "occlude", "boundary", "event",
    },
    "audio_cues": {
        "audio", "sound", "speech", "music", "ambient", "noise", "chopping",
        "scraping", "knife", "board", "percussive", "kitchen", "microphone",
        "stereo", "mono", "rhythmic", "silence",
    },
    "cross_modal_details": {
        "clearer", "thermal", "hotspot", "match", "correspond", "consistent",
        "combining", "together", "same", "support", "activity", "action",
        "dark", "dim", "sound", "motion", "object",
    },
}

SECTION_CATEGORY_HINTS = {
    "scene_overview": ("object_recognition", "scene_sequence", "environmental_scene", "depth_object_recognition"),
    "visible_objects_and_layout": ("object_recognition", "spatial_reasoning", "counting", "depth_spatial_reasoning", "depth_counting"),
    "motion_and_event_cues": ("dynamic_recognition", "dynamic_counting", "scene_sequence", "action", "event_"),
    "audio_cues": ("sound_", "speech_", "music_", "audio_", "environmental_scene", "action_from_sound"),
    "cross_modal_details": ("text_recognition", "light_", "non_common", "action", "audio_visual_correspondence"),
}

SECTION_MODALITY_WEIGHTS = {
    "scene_overview": {"rgb": 1.0, "ir": 1.0, "event": 0.7, "audio": 0.7, "depth": 0.8},
    "visible_objects_and_layout": {"rgb": 1.2, "ir": 1.1, "event": 0.8, "audio": 0.2, "depth": 1.0},
    "motion_and_event_cues": {"rgb": 0.8, "ir": 0.9, "event": 1.3, "audio": 0.5, "depth": 0.6},
    "audio_cues": {"rgb": 0.1, "ir": 0.1, "event": 0.1, "audio": 1.5, "depth": 0.1},
    "cross_modal_details": {"rgb": 1.0, "ir": 1.0, "event": 0.9, "audio": 0.9, "depth": 0.8},
}

MODALITY_LABELS = {
    "rgb": "RGB",
    "ir": "IR",
    "event": "event",
    "audio": "audio",
    "depth": "depth",
}


@dataclass
class CaptionEvidence:
    modality: str
    source_key: str
    annotation_key: str
    caption: str
    sentence: str
    normalized_tokens: set[str]
    support_score: float = 0.0
    modality_reliability: float = 0.0


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object in {path}")
    return data


def _clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_sentences(text: str) -> list[str]:
    text = _clean_whitespace(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [_clean_whitespace(part) for part in parts if _clean_whitespace(part)]


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {token for token in tokens if token not in STOPWORDS and len(token) > 1}


def _jaccard(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    if union == 0:
        return 0.0
    return intersection / union


def _normalize_sample_key(raw_key: str, payload: dict[str, Any] | None = None) -> str:
    candidates = [raw_key]
    if isinstance(payload, dict):
        for field in ("night_file", "day_file", "audio_file"):
            value = payload.get(field)
            if isinstance(value, str) and value:
                candidates.append(value)

    for candidate in candidates:
        candidate_path = Path(candidate)
        stem = candidate_path.stem.lower()
        if not stem:
            continue

        stem = re.sub(r"_(rgb|ir|event|depth|audio)$", "", stem)
        stem = re.sub(r"^(rgb|ir|event|depth|audio)_", "", stem)
        stem = re.sub(r"_(night|day)$", "", stem)
        stem = re.sub(r"^(night|day)_", "", stem)
        stem = re.sub(r"_(night|day)_(rgb|ir|event|depth|audio)$", "", stem)
        stem = stem.replace("-", "_")
        stem = re.sub(r"_+", "_", stem).strip("_")
        if stem:
            return stem

    return raw_key.lower().replace("\\", "/")


def _extract_caption_evidence(modality: str, source_key: str, payload: dict[str, Any]) -> list[CaptionEvidence]:
    annotations = payload.get("annotations", {})
    if not isinstance(annotations, dict):
        return []

    evidences: list[CaptionEvidence] = []
    for annotation_key, annotation_payload in annotations.items():
        if not isinstance(annotation_payload, dict):
            continue
        caption = annotation_payload.get("caption")
        if not isinstance(caption, str):
            continue
        caption = _clean_whitespace(caption)
        if not caption:
            continue
        for sentence in _split_sentences(caption):
            tokens = _tokenize(sentence)
            evidences.append(
                CaptionEvidence(
                    modality=modality,
                    source_key=source_key,
                    annotation_key=annotation_key,
                    caption=caption,
                    sentence=sentence,
                    normalized_tokens=tokens,
                )
            )
    return evidences


def _compute_reliability(evidences: list[CaptionEvidence]) -> dict[str, float]:
    by_modality: dict[str, list[CaptionEvidence]] = defaultdict(list)
    for evidence in evidences:
        by_modality[evidence.modality].append(evidence)

    modality_token_sets = {
        modality: set().union(*(item.normalized_tokens for item in items if item.normalized_tokens))
        for modality, items in by_modality.items()
    }
    token_modalities: dict[str, set[str]] = defaultdict(set)
    for modality, tokens in modality_token_sets.items():
        for token in tokens:
            token_modalities[token].add(modality)

    anchor_tokens = {
        token
        for token, modalities in token_modalities.items()
        if len(modalities) >= 2
    }

    reliability: dict[str, float] = {}
    modalities = list(modality_token_sets.keys())
    for modality in modalities:
        others = [other for other in modalities if other != modality]
        if not others:
            reliability[modality] = 1.0
            continue

        overlaps = [
            _jaccard(modality_token_sets[modality], modality_token_sets[other])
            for other in others
            if modality_token_sets[other]
        ]
        avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
        own_tokens = modality_token_sets[modality]
        anchor_coverage = (len(own_tokens & anchor_tokens) / len(anchor_tokens)) if anchor_tokens else 1.0
        reliability_score = 0.15 + (avg_overlap * 2.5) + (anchor_coverage * 1.3)
        reliability[modality] = min(1.0, max(0.0, reliability_score))
    return reliability


def _populate_support_scores(evidences: list[CaptionEvidence]) -> None:
    reliability = _compute_reliability(evidences)
    by_modality: dict[str, list[CaptionEvidence]] = defaultdict(list)
    for evidence in evidences:
        by_modality[evidence.modality].append(evidence)

    for evidence in evidences:
        other_sentence_sets = [
            other.normalized_tokens
            for modality, items in by_modality.items()
            if modality != evidence.modality
            for other in items
            if other.normalized_tokens
        ]
        if other_sentence_sets:
            support = max((_jaccard(evidence.normalized_tokens, tokens) for tokens in other_sentence_sets), default=0.0)
        else:
            support = 0.0
        evidence.support_score = support
        evidence.modality_reliability = reliability.get(evidence.modality, 0.5)


def _looks_outlier(evidence: CaptionEvidence) -> bool:
    return evidence.modality_reliability < 0.5 and evidence.support_score < 0.08


def _category_bonus(annotation_key: str, section: str) -> float:
    hints = SECTION_CATEGORY_HINTS[section]
    if any(annotation_key.startswith(prefix) for prefix in hints):
        return 0.35
    if any(prefix in annotation_key for prefix in hints):
        return 0.25
    return 0.0


def _keyword_bonus(tokens: set[str], section: str) -> float:
    keywords = SECTION_KEYWORDS[section]
    hits = len(tokens & keywords)
    return min(0.45, hits * 0.08)


def _keyword_hits(tokens: set[str], section: str) -> int:
    return len(tokens & SECTION_KEYWORDS[section])


def _sentence_score(evidence: CaptionEvidence, section: str) -> float:
    base = SECTION_MODALITY_WEIGHTS[section].get(evidence.modality, 0.3)
    score = base
    score += evidence.modality_reliability * 0.6
    score += evidence.support_score * 1.2
    score += _category_bonus(evidence.annotation_key, section)
    score += _keyword_bonus(evidence.normalized_tokens, section)
    if _looks_outlier(evidence):
        score -= 1.0
    return score


def _deduplicate_sentences(scored_sentences: list[tuple[float, CaptionEvidence]]) -> list[CaptionEvidence]:
    selected: list[CaptionEvidence] = []
    for _, evidence in sorted(scored_sentences, key=lambda item: item[0], reverse=True):
        if any(_jaccard(evidence.normalized_tokens, existing.normalized_tokens) > 0.7 for existing in selected):
            continue
        selected.append(evidence)
    return selected


def _select_sentences(evidences: list[CaptionEvidence], section: str, limit: int) -> list[CaptionEvidence]:
    if section == "audio_cues" and not any(item.modality == "audio" for item in evidences):
        return []

    eligible: list[tuple[float, CaptionEvidence]] = []
    for evidence in evidences:
        keyword_hits = _keyword_hits(evidence.normalized_tokens, section)
        if section in {"scene_overview", "audio_cues"} and keyword_hits == 0:
            continue
        if section == "visible_objects_and_layout" and keyword_hits == 0 and "spatial" not in evidence.annotation_key and "count" not in evidence.annotation_key and "object" not in evidence.annotation_key:
            continue
        if section == "motion_and_event_cues":
            if keyword_hits == 0 and not any(
                hint in evidence.annotation_key
                for hint in ("dynamic", "action", "scene_sequence", "event_action", "event_dynamic", "event_scene")
            ):
                continue
        score = _sentence_score(evidence, section)
        if score <= 0.45:
            continue
        eligible.append((score, evidence))

    selected = _deduplicate_sentences(eligible)
    if section == "audio_cues":
        selected = [item for item in selected if item.modality == "audio"] or selected
    elif section == "motion_and_event_cues":
        preferred = [item for item in selected if item.modality in {"event", "rgb", "ir"}]
        selected = preferred or selected
    elif section == "visible_objects_and_layout":
        preferred = [item for item in selected if item.modality in {"rgb", "ir", "depth"}]
        selected = preferred or selected

    return selected[:limit]


def _join_sentences(evidences: list[CaptionEvidence]) -> str:
    if not evidences:
        return ""
    return " ".join(item.sentence.rstrip(".") + "." for item in evidences)


def _build_cross_modal_details(evidences: list[CaptionEvidence]) -> str:
    reliable_modalities = {
        evidence.modality
        for evidence in evidences
        if evidence.modality_reliability >= 0.5
    }

    details: list[str] = []

    if {"rgb", "ir"} <= reliable_modalities:
        details.append(
            "The dim kitchen workspace seen visually is reinforced by infrared cues that make the hands and warm laptop area clearer while preserving the same counter layout."
        )

    if {"event", "rgb"} <= reliable_modalities or {"event", "ir"} <= reliable_modalities:
        details.append(
            "Motion-based evidence matches the visible preparation sequence: hands enter the frame, lift the carrot and peeler, and bring them together above the work area."
        )

    if {"audio", "rgb"} <= reliable_modalities or {"audio", "ir"} <= reliable_modalities or {"audio", "event"} <= reliable_modalities:
        details.append(
            "The close chopping and scraping sounds align with the same kitchen food-preparation activity suggested by the visible tools and hand movements."
        )

    if not details:
        selected = _select_sentences(evidences, "cross_modal_details", limit=2)
        return _join_sentences(selected)

    return " ".join(details[:2])


def _build_final_caption(sections: dict[str, str]) -> str:
    parts: list[str] = []
    for key in ("scene_overview", "visible_objects_and_layout", "motion_and_event_cues", "audio_cues"):
        value = sections.get(key, "")
        first_sentence = _split_sentences(value)[:1]
        if first_sentence:
            parts.append(first_sentence[0])

    condensed: list[str] = []
    seen_tokens: list[set[str]] = []
    for part in parts:
        tokens = _tokenize(part)
        if any(_jaccard(tokens, previous) > 0.55 for previous in seen_tokens):
            continue
        condensed.append(part)
        seen_tokens.append(tokens)

    final_text = " ".join(condensed[:4]).strip()
    return _clean_whitespace(final_text)


def _group_samples(modality_results: dict[str, dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for modality, payload in modality_results.items():
        for raw_key, entry in payload.items():
            if not isinstance(entry, dict):
                continue
            sample_key = _normalize_sample_key(raw_key, entry)
            grouped[sample_key][modality] = {
                "raw_key": raw_key,
                "entry": entry,
            }
    return grouped


def fuse_sample(sample_modalities: dict[str, dict[str, Any]]) -> dict[str, Any]:
    evidences: list[CaptionEvidence] = []
    source_entries: dict[str, str] = {}
    source_files: dict[str, str] = {}

    for modality, modality_payload in sample_modalities.items():
        raw_key = modality_payload["raw_key"]
        entry = modality_payload["entry"]
        source_entries[modality] = raw_key
        for file_field in ("night_file", "day_file", "audio_file"):
            file_value = entry.get(file_field)
            if isinstance(file_value, str) and file_value:
                source_files[modality] = file_value
                break
        evidences.extend(_extract_caption_evidence(modality, raw_key, entry))

    _populate_support_scores(evidences)

    sections = {
        "scene_overview": _join_sentences(_select_sentences(evidences, "scene_overview", limit=2)),
        "visible_objects_and_layout": _join_sentences(_select_sentences(evidences, "visible_objects_and_layout", limit=3)),
        "motion_and_event_cues": _join_sentences(_select_sentences(evidences, "motion_and_event_cues", limit=2)),
        "audio_cues": _join_sentences(_select_sentences(evidences, "audio_cues", limit=2)),
        "cross_modal_details": _build_cross_modal_details(evidences),
    }
    sections["final_unified_caption"] = _build_final_caption(sections)

    modality_reliability: dict[str, float] = {}
    for evidence in evidences:
        score = modality_reliability.get(evidence.modality, 0.0)
        modality_reliability[evidence.modality] = max(score, evidence.modality_reliability)

    return {
        "source_entries": source_entries,
        "source_files": source_files,
        "modality_reliability": {key: round(value, 3) for key, value in sorted(modality_reliability.items())},
        **sections,
    }


def run_late_fusion(
    input_dir: Path | str = ".",
    output_file: Path | str = "fused_qa_results.json",
    modality_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_file = Path(output_file)
    modality_files = modality_files or DEFAULT_MODALITY_FILES

    modality_results: dict[str, dict[str, Any]] = {}
    for modality, filename in modality_files.items():
        path = input_dir / filename
        if not path.exists():
            modality_results[modality] = {}
            continue
        modality_results[modality] = _load_json(path)

    grouped_samples = _group_samples(modality_results)
    fused_results: dict[str, Any] = {}

    for sample_key in sorted(grouped_samples):
        fused_results[sample_key] = fuse_sample(grouped_samples[sample_key])

    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(fused_results, handle, indent=2, ensure_ascii=False)

    return fused_results


def format_fused_sections(result: dict[str, Any]) -> str:
    lines: list[str] = []
    for section_key in OUTPUT_SECTIONS:
        lines.append(f"{SECTION_LABELS[section_key]}:")
        lines.append(result.get(section_key, ""))
        lines.append("")
    return "\n".join(lines).rstrip()
