"""
Late fusion utilities for combining modality-specific captions.

This module reads the existing per-modality QA result files, aligns entries by
sample, and produces structured fused scene descriptions plus a final unified
caption for each sample.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
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

DEFAULT_SECTION_LIMITS = {
    "scene_overview": 2,
    "visible_objects_and_layout": 3,
    "motion_and_event_cues": 2,
    "audio_cues": 2,
}

RELAXED_SECTION_LIMITS = {
    "scene_overview": 4,
    "visible_objects_and_layout": 6,
    "motion_and_event_cues": 4,
    "audio_cues": 4,
}

DEFAULT_SECTION_QA_LIMITS = {
    "scene_overview": 2,
    "visible_objects_and_layout": 3,
    "motion_and_event_cues": 2,
    "audio_cues": 2,
    "cross_modal_details": 0,
    "final_unified_caption": 0,
}

RELAXED_SECTION_QA_LIMITS = {
    "scene_overview": 4,
    "visible_objects_and_layout": 6,
    "motion_and_event_cues": 4,
    "audio_cues": 4,
    "cross_modal_details": 0,
    "final_unified_caption": 0,
}

DEFAULT_MIN_SCORE = 0.45
RELAXED_MIN_SCORE = 0.2
DEFAULT_RELIABLE_QA_MIN_SCORE = 1.8
RELAXED_RELIABLE_QA_MIN_SCORE = 1.6
EXEMPT_RELIABLE_QA_MIN_SCORE = 1.7
REVIEW_RELIABLE_QA_MIN_SCORE = 2.0
DEFAULT_MIN_SUPPORT_SCORE = 0.08
STRICT_MIN_SUPPORT_SCORE = 0.12
STRICT_MIN_LEXICAL_SUPPORT_SCORE = 0.04
SUPPORTING_MODALITY_THRESHOLD = 0.3
STRICT_SUPPORTING_MODALITY_THRESHOLD = 0.25
ANSWER_REVIEW_WORD_LIMIT = 12
QUESTION_REVIEW_WORD_LIMIT = 35
SOFT_HIGH_SCORE = 2.2
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
SEMANTIC_SUPPORT_COSINE_FLOOR = 0.4
SEMANTIC_SUPPORT_COSINE_CEILING = 0.9
SEMANTIC_HIGH_REVIEW_THRESHOLD = 0.75
LEXICAL_LOW_REVIEW_THRESHOLD = 0.05
SEMANTIC_DUPLICATE_THRESHOLD = 0.9

PRIMARY_SECTIONS = (
    "scene_overview",
    "visible_objects_and_layout",
    "motion_and_event_cues",
    "audio_cues",
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
    "visible_objects_and_layout": ("object_recognition", "spatial_reasoning", "counting", "depth_spatial_reasoning",
                                   "depth_counting"),
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

VISUAL_MODALITIES = ("rgb", "ir", "depth")
FIELD_QA_SECTIONS = OUTPUT_SECTIONS
MODALITY_SORT_ORDER = {
    "rgb": 0,
    "ir": 1,
    "depth": 2,
    "event": 3,
    "audio": 4,
}

QA_SECTIONS = (
    "scene_and_context",
    "objects_and_attributes",
    "spatial_and_layout",
    "motion_and_action",
    "temporal_sequence",
    "counting",
    "text_and_symbols",
    "audio_understanding",
    "anomaly_and_safety",
    "others",
)

CATEGORY_SECTION_MAP = {
    "scene_sequence": "temporal_sequence",
    "event_scene_sequence": "temporal_sequence",
    "depth_scene_sequence": "temporal_sequence",
    "light_recongnition": "scene_and_context",
    "light_recognition": "scene_and_context",
    "light_change": "scene_and_context",
    "object_recognition": "objects_and_attributes",
    "event_object_recognition": "objects_and_attributes",
    "depth_object_recognition": "objects_and_attributes",
    "spatial_reasoning": "spatial_and_layout",
    "event_spatial_reasoning": "spatial_and_layout",
    "depth_spatial_reasoning": "spatial_and_layout",
    "navigation": "spatial_and_layout",
    "event_navigation": "spatial_and_layout",
    "depth_navigation": "spatial_and_layout",
    "action": "motion_and_action",
    "dynamic_recognition": "motion_and_action",
    "dynamic_counting": "motion_and_action",
    "event_action": "motion_and_action",
    "event_dynamic_recognition": "motion_and_action",
    "event_dynamic_counting": "motion_and_action",
    "depth_action": "motion_and_action",
    "depth_dynamic_recognition": "motion_and_action",
    "depth_dynamic_counting": "motion_and_action",
    "counting": "counting",
    "event_counting": "counting",
    "depth_counting": "counting",
    "text_recognition": "text_and_symbols",
    "audio_hia": "audio_understanding",
    "audio_chronological_caption": "audio_understanding",
    "non_common": "anomaly_and_safety",
    "event_non_common": "anomaly_and_safety",
    "depth_non_common": "anomaly_and_safety",
}

CATEGORY_SECTIONS_MAP = {
    "scene_sequence": ("temporal_sequence", "motion_and_action"),
    "event_scene_sequence": ("temporal_sequence", "motion_and_action"),
    "depth_scene_sequence": ("temporal_sequence", "spatial_and_layout"),
    "light_recongnition": ("scene_and_context",),
    "light_recognition": ("scene_and_context",),
    "light_change": ("scene_and_context",),
    "object_recognition": ("objects_and_attributes",),
    "event_object_recognition": ("objects_and_attributes",),
    "depth_object_recognition": ("objects_and_attributes",),
    "spatial_reasoning": ("spatial_and_layout",),
    "event_spatial_reasoning": ("spatial_and_layout",),
    "depth_spatial_reasoning": ("spatial_and_layout",),
    "navigation": ("spatial_and_layout", "motion_and_action"),
    "event_navigation": ("spatial_and_layout", "motion_and_action"),
    "depth_navigation": ("spatial_and_layout", "motion_and_action"),
    "action": ("motion_and_action",),
    "dynamic_recognition": ("motion_and_action",),
    "dynamic_counting": ("motion_and_action", "counting"),
    "event_action": ("motion_and_action",),
    "event_dynamic_recognition": ("motion_and_action",),
    "event_dynamic_counting": ("motion_and_action", "counting"),
    "depth_action": ("motion_and_action",),
    "depth_dynamic_recognition": ("motion_and_action",),
    "depth_dynamic_counting": ("motion_and_action", "counting"),
    "counting": ("counting",),
    "event_counting": ("counting",),
    "depth_counting": ("counting",),
    "text_recognition": ("text_and_symbols", "objects_and_attributes"),
    "audio_hia": ("audio_understanding",),
    "audio_chronological_caption": ("audio_understanding", "temporal_sequence"),
    "non_common": ("anomaly_and_safety",),
    "event_non_common": ("anomaly_and_safety",),
    "depth_non_common": ("anomaly_and_safety",),
}

MODALITY_WEIGHT_PROFILES = {
    "visual_object": {"rgb": 1.2, "ir": 1.1, "event": 0.8, "audio": 0.2, "depth": 1.0},
    "visual_spatial": {"rgb": 1.1, "ir": 1.0, "event": 0.7, "audio": 0.1, "depth": 1.2},
    "visual_count": {"rgb": 1.1, "ir": 1.0, "event": 0.8, "audio": 0.1, "depth": 1.1},
    "visual_text": {"rgb": 1.4, "ir": 1.0, "event": 0.2, "audio": 0.0, "depth": 0.2},
    "visual_light": {"rgb": 1.0, "ir": 1.2, "event": 0.4, "audio": 0.0, "depth": 0.2},
    "visual_motion": {"rgb": 0.8, "ir": 0.9, "event": 1.3, "audio": 0.4, "depth": 0.6},
    "temporal": {"rgb": 0.8, "ir": 0.8, "event": 1.2, "audio": 0.5, "depth": 0.6},
    "audio": {"rgb": 0.1, "ir": 0.1, "event": 0.1, "audio": 1.5, "depth": 0.1},
    "anomaly": {"rgb": 1.0, "ir": 1.0, "event": 0.9, "audio": 0.6, "depth": 0.8},
    "default": {"rgb": 0.8, "ir": 0.8, "event": 0.8, "audio": 0.5, "depth": 0.8},
}

DEFAULT_CATEGORY_RELIABILITY_PROFILE = {
    "gate": "support_required",
    "modality_weight_profile": "default",
    "support_weights": {"lexical": 0.5, "semantic": 0.5},
    "requires_lexical_support": False,
}

CATEGORY_RELIABILITY_PROFILES = {
    "scene_sequence": {
        "gate": "support_soft",
        "modality_weight_profile": "temporal",
        "support_weights": {"lexical": 0.4, "semantic": 0.6},
        "requires_lexical_support": False,
    },
    "event_scene_sequence": {
        "gate": "support_soft",
        "modality_weight_profile": "temporal",
        "support_weights": {"lexical": 0.4, "semantic": 0.6},
        "requires_lexical_support": False,
    },
    "depth_scene_sequence": {
        "gate": "support_soft",
        "modality_weight_profile": "temporal",
        "support_weights": {"lexical": 0.4, "semantic": 0.6},
        "requires_lexical_support": False,
    },
    "light_recongnition": {
        "gate": "support_exempt",
        "modality_weight_profile": "visual_light",
        "support_weights": {"lexical": 0.45, "semantic": 0.55},
        "requires_lexical_support": False,
    },
    "light_recognition": {
        "gate": "support_exempt",
        "modality_weight_profile": "visual_light",
        "support_weights": {"lexical": 0.45, "semantic": 0.55},
        "requires_lexical_support": False,
    },
    "light_change": {
        "gate": "support_exempt",
        "modality_weight_profile": "visual_light",
        "support_weights": {"lexical": 0.45, "semantic": 0.55},
        "requires_lexical_support": False,
    },
    "object_recognition": {
        "gate": "support_required",
        "modality_weight_profile": "visual_object",
        "support_weights": {"lexical": 0.55, "semantic": 0.45},
        "requires_lexical_support": False,
    },
    "event_object_recognition": {
        "gate": "support_required",
        "modality_weight_profile": "visual_object",
        "support_weights": {"lexical": 0.55, "semantic": 0.45},
        "requires_lexical_support": False,
    },
    "depth_object_recognition": {
        "gate": "support_required",
        "modality_weight_profile": "visual_object",
        "support_weights": {"lexical": 0.55, "semantic": 0.45},
        "requires_lexical_support": False,
    },
    "spatial_reasoning": {
        "gate": "support_required",
        "modality_weight_profile": "visual_spatial",
        "support_weights": {"lexical": 0.6, "semantic": 0.4},
        "requires_lexical_support": True,
    },
    "event_spatial_reasoning": {
        "gate": "support_required",
        "modality_weight_profile": "visual_spatial",
        "support_weights": {"lexical": 0.6, "semantic": 0.4},
        "requires_lexical_support": True,
    },
    "depth_spatial_reasoning": {
        "gate": "support_required",
        "modality_weight_profile": "visual_spatial",
        "support_weights": {"lexical": 0.6, "semantic": 0.4},
        "requires_lexical_support": True,
    },
    "navigation": {
        "gate": "support_required",
        "modality_weight_profile": "visual_spatial",
        "support_weights": {"lexical": 0.6, "semantic": 0.4},
        "requires_lexical_support": True,
    },
    "event_navigation": {
        "gate": "support_required",
        "modality_weight_profile": "visual_spatial",
        "support_weights": {"lexical": 0.6, "semantic": 0.4},
        "requires_lexical_support": True,
    },
    "depth_navigation": {
        "gate": "support_required",
        "modality_weight_profile": "visual_spatial",
        "support_weights": {"lexical": 0.6, "semantic": 0.4},
        "requires_lexical_support": True,
    },
    "action": {
        "gate": "support_soft",
        "modality_weight_profile": "visual_motion",
        "support_weights": {"lexical": 0.35, "semantic": 0.65},
        "requires_lexical_support": False,
    },
    "dynamic_recognition": {
        "gate": "support_soft",
        "modality_weight_profile": "visual_motion",
        "support_weights": {"lexical": 0.35, "semantic": 0.65},
        "requires_lexical_support": False,
    },
    "dynamic_counting": {
        "gate": "support_soft",
        "modality_weight_profile": "visual_motion",
        "support_weights": {"lexical": 0.55, "semantic": 0.45},
        "requires_lexical_support": True,
    },
    "event_action": {
        "gate": "support_soft",
        "modality_weight_profile": "visual_motion",
        "support_weights": {"lexical": 0.35, "semantic": 0.65},
        "requires_lexical_support": False,
    },
    "event_dynamic_recognition": {
        "gate": "support_soft",
        "modality_weight_profile": "visual_motion",
        "support_weights": {"lexical": 0.35, "semantic": 0.65},
        "requires_lexical_support": False,
    },
    "event_dynamic_counting": {
        "gate": "support_soft",
        "modality_weight_profile": "visual_motion",
        "support_weights": {"lexical": 0.55, "semantic": 0.45},
        "requires_lexical_support": True,
    },
    "depth_action": {
        "gate": "support_soft",
        "modality_weight_profile": "visual_motion",
        "support_weights": {"lexical": 0.35, "semantic": 0.65},
        "requires_lexical_support": False,
    },
    "depth_dynamic_recognition": {
        "gate": "support_soft",
        "modality_weight_profile": "visual_motion",
        "support_weights": {"lexical": 0.35, "semantic": 0.65},
        "requires_lexical_support": False,
    },
    "depth_dynamic_counting": {
        "gate": "support_soft",
        "modality_weight_profile": "visual_motion",
        "support_weights": {"lexical": 0.55, "semantic": 0.45},
        "requires_lexical_support": True,
    },
    "counting": {
        "gate": "support_required",
        "modality_weight_profile": "visual_count",
        "support_weights": {"lexical": 0.7, "semantic": 0.3},
        "requires_lexical_support": True,
    },
    "event_counting": {
        "gate": "support_required",
        "modality_weight_profile": "visual_count",
        "support_weights": {"lexical": 0.7, "semantic": 0.3},
        "requires_lexical_support": True,
    },
    "depth_counting": {
        "gate": "support_required",
        "modality_weight_profile": "visual_count",
        "support_weights": {"lexical": 0.7, "semantic": 0.3},
        "requires_lexical_support": True,
    },
    "text_recognition": {
        "gate": "support_exempt",
        "modality_weight_profile": "visual_text",
        "support_weights": {"lexical": 0.65, "semantic": 0.35},
        "requires_lexical_support": True,
    },
    "audio_hia": {
        "gate": "support_exempt",
        "modality_weight_profile": "audio",
        "support_weights": {"lexical": 0.3, "semantic": 0.7},
        "requires_lexical_support": False,
    },
    "audio_chronological_caption": {
        "gate": "support_exempt",
        "modality_weight_profile": "audio",
        "support_weights": {"lexical": 0.3, "semantic": 0.7},
        "requires_lexical_support": False,
    },
    "non_common": {
        "gate": "review_recommended",
        "modality_weight_profile": "anomaly",
        "support_weights": {"lexical": 0.5, "semantic": 0.5},
        "requires_lexical_support": False,
    },
    "event_non_common": {
        "gate": "review_recommended",
        "modality_weight_profile": "anomaly",
        "support_weights": {"lexical": 0.5, "semantic": 0.5},
        "requires_lexical_support": False,
    },
    "depth_non_common": {
        "gate": "review_recommended",
        "modality_weight_profile": "anomaly",
        "support_weights": {"lexical": 0.5, "semantic": 0.5},
        "requires_lexical_support": False,
    },
}

SECTION_FACT_TYPES = {
    "scene_overview": {"scene", "environment", "object_presence", "lighting"},
    "visible_objects_and_layout": {"object_presence", "spatial_relation", "count", "non_common"},
    "motion_and_event_cues": {"action", "motion", "temporal", "verification"},
    "audio_cues": {"sound", "audio_event", "audio_causality"},
    "cross_modal_details": {"cross_modal"},
    "final_unified_caption": {
        "scene",
        "environment",
        "object_presence",
        "spatial_relation",
        "count",
        "lighting",
        "action",
        "motion",
        "temporal",
        "sound",
        "audio_event",
        "audio_causality",
        "verification",
        "cross_modal",
        "non_common",
    },
}

FACT_SECTION_PRIORITIES = {
    "scene_overview": ("scene", "environment", "lighting", "object_presence", "count"),
    "visible_objects_and_layout": ("spatial_relation", "object_presence", "count", "non_common"),
    "motion_and_event_cues": ("action", "motion", "temporal", "verification"),
    "audio_cues": ("sound", "audio_event", "audio_causality"),
    "cross_modal_details": ("cross_modal",),
    "final_unified_caption": (
        "cross_modal",
        "scene",
        "environment",
        "object_presence",
        "spatial_relation",
        "count",
        "lighting",
        "action",
        "motion",
        "temporal",
        "sound",
        "audio_event",
        "audio_causality",
        "verification",
        "non_common",
    ),
}


@dataclass
class SemanticFact:
    modality: str
    section: str
    fact_type: str
    entity: str
    attribute: str
    value: str
    confidence: float
    source_question: str
    source_answer: str
    source_caption: str
    source_key: str
    annotation_key: str
    support_score: float = 0.0
    modality_reliability: float = 0.0


@dataclass
class CaptionEvidence:
    modality: str
    source_key: str
    annotation_key: str
    question: str
    answer: str
    caption: str
    sentence: str
    normalized_tokens: set[str]
    qa_index: int | None = None
    support_score: float = 0.0
    lexical_support_score: float = 0.0
    semantic_support_score: float = 0.0
    semantic_support_raw_cosine: float = 0.0
    support_fusion_weights: dict[str, float] = field(default_factory=dict)
    lexical_support_match: dict[str, Any] | None = None
    semantic_support_match: dict[str, Any] | None = None
    support_by_modality: dict[str, float] = field(default_factory=dict)
    supporting_modalities: list[str] = field(default_factory=list)
    supporting_modality_count: int = 0
    modality_reliability: float = 0.0
    semantic_facts: list[SemanticFact] = field(default_factory=list)
    original_qa: dict[str, str] | None = None


def _serialize_semantic_fact(fact: SemanticFact) -> dict[str, Any]:
    return asdict(fact)


def _is_yes_no_answer(text: str) -> bool:
    normalized = _clean_whitespace(text).lower()
    return normalized in {"yes", "no", "true", "false"}


def _looks_like_question(text: str) -> bool:
    normalized = _clean_whitespace(text).lower()
    return normalized.startswith(("what ", "where ", "when ", "which ", "how ", "why ", "is ", "are ", "do ", "does ", "did ", "was ", "were "))


def _first_answer_fragment(text: str) -> str:
    cleaned = _clean_whitespace(text).strip(" .,:;\n\t")
    if not cleaned:
        return ""

    parts = [part.strip(" .,:;\n\t") for part in re.split(r"\s*\d+\.\s*", cleaned) if part.strip(" .,:;\n\t")]
    if parts:
        return parts[0]

    return cleaned


def _infer_fact_type(section: str, question: str, answer: str, caption: str) -> str:
    text = f"{section} {question} {answer} {caption}".lower()
    question_lower = question.lower().strip()
    section_lower = section.lower()

    if "audio_" in section_lower or section_lower.startswith("audio"):
        if any(word in section_lower for word in ("chronological", "temporal")):
            return "audio_event"
        if any(word in section_lower for word in ("source", "causality", "inferential")):
            return "audio_causality"
        return "sound"

    if "object_recognition" in section_lower:
        return "object_presence"

    if any(word in section_lower for word in ("spatial_reasoning", "navigation")):
        return "spatial_relation"

    if "counting" in section_lower:
        return "count"

    if any(word in section_lower for word in ("dynamic_recognition", "scene_sequence")):
        return "motion"

    if "dynamic_counting" in section_lower:
        return "count"

    if "action" in section_lower:
        return "action"

    if any(word in section_lower for word in ("light_recongnition", "light_recognition", "light_change")):
        return "lighting"

    if "text_recognition" in section_lower:
        return "object_presence"

    if "non_common" in section_lower:
        return "non_common"

    if "scene_sequence" in section_lower:
        return "motion"

    if section.startswith("audio_") or "sound" in text or "audio" in text or "scraping" in text or "thwack" in text:
        if any(word in text for word in ("because", "caused", "cause", "result", "due to")):
            return "audio_causality"
        if any(word in text for word in ("characteristics", "texture", "tempo", "source", "identification")):
            return "audio_event"
        return "sound"

    if any(word in text for word in ("light", "illumination", "brightness", "shadow", "luminous")):
        return "lighting"

    if question_lower.startswith(("how many", "how much")) or re.search(r"\b\d+\b", answer):
        return "count"

    if question_lower.startswith(("where", "which side", "in which direction")) or any(
        phrase in text for phrase in ("to the left", "to the right", "on the", "in front", "behind", "above", "below")
    ):
        return "spatial_relation"

    if question_lower.startswith(("what action", "what is happening", "what occurs", "what does", "what do")):
        return "action"

    if any(word in text for word in ("enter", "lift", "grasp", "pick up", "move", "motion", "moving", "sequence")):
        return "motion"

    if question_lower.startswith(("when", "at what frame")) or "frame_" in text or re.search(r"\bframe\b", text):
        return "temporal"

    if _is_yes_no_answer(answer) or question_lower.startswith(("is ", "are ", "was ", "were ", "did ", "does ", "do ")):
        return "verification"

    if any(word in text for word in ("setting", "scene", "workspace", "kitchen", "room", "environment")):
        return "scene"

    if any(word in text for word in ("object", "device", "tool", "item", "carrot", "bowl", "knife", "peeler", "laptop", "hands")):
        return "object_presence"

    if any(word in text for word in ("match", "correspond", "consistent", "align", "reinforced", "cross modal")):
        return "cross_modal"

    if any(word in text for word in ("unsafe", "odd", "wrong", "uncommon", "non common", "impossible")):
        return "non_common"

    return "environment"


def _infer_fact_entity(section: str, question: str, answer: str, fact_type: str) -> str:
    cleaned_answer = _first_answer_fragment(answer)
    if not cleaned_answer:
        return ""
    if fact_type == "count":
        count_question = _clean_whitespace(question)
        match = re.search(r"how many\s+(.+?)(?:\s+are\b|\s+is\b|\s+do\b|\s+does\b|\s+appear\b|\s+visible\b|\s+present\b|\s+in\b|\s+on\b|\?|$)", count_question, flags=re.IGNORECASE)
        if match:
            entity = _clean_whitespace(match.group(1))
            entity = re.split(r"\b(?:are|is|do|does|move|moves|enter|enters|appear|appears|start|starts|begin|begins|in|on|at|with|during|while|to)\b", entity, maxsplit=1, flags=re.IGNORECASE)[0]
            return _clean_whitespace(entity)
        return cleaned_answer

    if fact_type == "temporal":
        return _clean_whitespace(question)

    if fact_type == "verification":
        return cleaned_answer

    if fact_type == "spatial_relation":
        if any(token in cleaned_answer.lower() for token in ("left", "right", "above", "below", "in front", "behind", "center", "middle")):
            return _clean_whitespace(question)
        return cleaned_answer

    if fact_type in {"sound", "audio_event", "audio_causality", "action", "motion", "scene", "environment", "object_presence", "lighting", "non_common"}:
        return cleaned_answer

    return cleaned_answer


def _infer_fact_attribute(section: str, question: str, fact_type: str) -> str:
    question = _clean_whitespace(question)
    if fact_type == "spatial_relation":
        if " of the " in question.lower():
            return "relative_location"
        return "location"
    if fact_type == "count":
        return "count"
    if fact_type in {"sound", "audio_event", "audio_causality"}:
        return "audio"
    if fact_type == "lighting":
        return "illumination"
    if fact_type == "verification":
        return "truth_value"
    if fact_type in {"action", "motion"}:
        return "motion"
    if fact_type == "temporal":
        return "time"
    if fact_type == "scene":
        return "scene_type"
    if fact_type == "non_common":
        return "anomaly"
    return "fact"


def _infer_fact_value(answer: str, caption: str, fact_type: str) -> str:
    cleaned_answer = _first_answer_fragment(answer)
    if cleaned_answer:
        if fact_type == "verification":
            return cleaned_answer.lower().capitalize()
        return cleaned_answer

    cleaned_caption = _clean_whitespace(caption)
    if fact_type == "scene":
        return cleaned_caption
    if fact_type == "count":
        match = re.search(r"\b\d+\b", cleaned_caption)
        if match:
            return match.group(0)
    return cleaned_caption


def _build_semantic_fact(
        modality: str,
        source_key: str,
        annotation_key: str,
        section: str,
        caption: str,
        question: str,
        answer: str,
        confidence: float,
        support_score: float,
        modality_reliability: float,
) -> SemanticFact:
    fact_type = _infer_fact_type(section, question, answer, caption)
    entity = _infer_fact_entity(section, question, answer, fact_type)
    attribute = _infer_fact_attribute(section, question, fact_type)
    value = _infer_fact_value(answer, caption, fact_type)
    if fact_type == "cross_modal":
        attribute = "agreement"
    return SemanticFact(
        modality=modality,
        section=section,
        fact_type=fact_type,
        entity=entity,
        attribute=attribute,
        value=value,
        confidence=confidence,
        source_question=_clean_whitespace(question),
        source_answer=_clean_whitespace(answer),
        source_caption=_clean_whitespace(caption),
        source_key=source_key,
        annotation_key=annotation_key,
        support_score=support_score,
        modality_reliability=modality_reliability,
    )


def _fact_section_score(fact: SemanticFact, section: str) -> float:
    score = fact.confidence * 0.8 + fact.modality_reliability * 0.6 + fact.support_score * 0.5
    if fact.fact_type in SECTION_FACT_TYPES.get(section, set()):
        score += 0.7
    if fact.fact_type in FACT_SECTION_PRIORITIES.get(section, ()):  # type: ignore[arg-type]
        score += 0.35
    if section == "cross_modal_details" and len(fact.entity.split()) > 1:
        score += 0.1
    return score


def _format_fact_clause(fact: SemanticFact) -> str:
    modality = fact.modality.upper()
    entity = fact.entity.strip()
    value = fact.value.strip()

    if fact.fact_type == "spatial_relation":
        if entity and value and not _looks_like_question(entity):
            return f"{modality} places {entity} {value}"
        if value:
            return f"{modality} places {value}"
    if fact.fact_type == "count":
        if entity and value and not _looks_like_question(entity):
            return f"{modality} counts {value} {entity}"
        return f"{modality} counts {value}"
    if fact.fact_type == "object_presence":
        return f"{modality} grounds {value or entity}"
    if fact.fact_type in {"action", "motion"}:
        return f"{modality} captures {value or entity}"
    if fact.fact_type in {"sound", "audio_event", "audio_causality"}:
        return f"{modality} captures {value or entity}"
    if fact.fact_type == "lighting":
        return f"{modality} describes {value or entity}"
    if fact.fact_type == "verification":
        return f"{modality} confirms {value or entity}"
    if fact.fact_type == "temporal":
        return f"{modality} anchors the event at {value or entity}"
    if fact.fact_type == "non_common":
        return f"{modality} flags {value or entity}"
    if fact.fact_type == "scene":
        return f"{modality} sets the scene as {value or entity}"
    if fact.fact_type == "cross_modal":
        return f"{modality} links {value or entity}"
    if entity and value and entity.lower() != value.lower() and not _looks_like_question(entity):
        return f"{modality} grounds {entity} as {value}"
    return f"{modality} notes {value or entity}"


def _fact_to_qa(fact: SemanticFact, section: str) -> tuple[str, str]:
    entity = fact.entity.strip()
    value = fact.value.strip()

    if section == "scene_overview":
        if value:
            return "What scene context is grounded by the semantic facts?", value
        return "What scene context is grounded by the semantic facts?", entity or fact.source_answer

    if section == "visible_objects_and_layout":
        if fact.fact_type == "spatial_relation" and entity and value:
            return f"Where is {entity} located?", value
        if fact.fact_type == "count" and entity and value:
            return f"How many {entity} are identified?", value
        return "What object or layout fact is grounded?", value or entity

    if section == "motion_and_event_cues":
        return "What motion or event cue is grounded?", value or entity or fact.source_answer

    if section == "audio_cues":
        return "What audio cue is grounded?", value or entity or fact.source_answer

    if section == "cross_modal_details":
        return "What cross-modal detail is supported by the facts?", value or entity or fact.source_answer

    if section == "final_unified_caption":
        return "What unified scene does the semantic graph describe?", value or entity or fact.source_answer

    return "What semantic fact is grounded?", value or entity or fact.source_answer


def _choose_facts_for_section(
        semantic_facts: list[SemanticFact],
        section: str,
        limit: int,
) -> list[SemanticFact]:
    allowed_types = SECTION_FACT_TYPES.get(section, set())
    scoped_facts = [fact for fact in semantic_facts if not allowed_types or fact.fact_type in allowed_types]
    if not scoped_facts:
        scoped_facts = semantic_facts

    seen: set[tuple[str, str, str, str, str]] = set()
    scored_facts: list[tuple[float, SemanticFact]] = []

    for fact in scoped_facts:
        score = _fact_section_score(fact, section)
        if score <= 0:
            continue
        key = (fact.modality, fact.fact_type, fact.entity, fact.attribute, fact.value)
        if key in seen:
            continue
        seen.add(key)
        scored_facts.append((score, fact))

    scored_facts.sort(key=lambda item: (-item[0], -item[1].confidence, -item[1].modality_reliability, item[1].modality))
    selected = [fact for _, fact in scored_facts[:limit]]
    if section == "cross_modal_details":
        return [fact for fact in selected if fact.fact_type == "cross_modal"] or selected
    return selected


def _summarize_facts(semantic_facts: list[SemanticFact], section: str, limit: int) -> str:
    selected = _choose_facts_for_section(semantic_facts, section, limit)
    if not selected:
        return ""

    clauses = [_format_fact_clause(fact) for fact in selected if _format_fact_clause(fact)]
    clauses = list(dict.fromkeys(clauses))
    if section == "cross_modal_details" and len(clauses) >= 2:
        return f"{clauses[0]} while {clauses[1].lower()}"
    return ". ".join(clause.rstrip(".") for clause in clauses[:limit]).strip()


def _facts_from_evidences(evidences: list[CaptionEvidence]) -> list[SemanticFact]:
    facts: list[SemanticFact] = []
    seen: set[tuple[str, str, str, str, str, str]] = set()
    for evidence in evidences:
        for fact in evidence.semantic_facts:
            key = (fact.modality, fact.section, fact.fact_type, fact.entity, fact.attribute, fact.value)
            if key in seen:
                continue
            seen.add(key)
            facts.append(fact)
    return facts


def _trim_answer(text: str, max_words: int = 8) -> str:
    """Trim answer to complete phrases with intelligent sentence boundary detection.

    Args:
        text: The answer text to trim
        max_words: Maximum number of words allowed (default 8 for balanced VQA compliance)
                  Respects VQA requirement of "concise and evaluable" while allowing
                  complete noun phrases. Rationale:
                  - Event/Depth original spec: 1-5 words (strict VQA)
                  - Spatial relations need: 3-6 words (2-word object + 3-4 words context)
                  - Multi-word objects: up to 8 words (e.g., "empty white bowl on counter")
                  - Compromise: 8 words balances completeness with conciseness

    Strategy:
    1. Try to find a natural sentence boundary (period, exclamation mark, question mark)
    2. If found, take up to that boundary (or up to max_words, whichever is shorter)
    3. Otherwise, try phrase separators (comma, "and", etc.)
    4. As fallback, apply word-count truncation and remove trailing prepositions/articles

    This prevents incomplete truncations like "To the right of the cutting" when
    the full phrase should be "To the right of the cutting board" (staying within limits).
    """
    text = _clean_whitespace(text.strip(" .,:;"))
    if not text:
        return ""

    # Remove leading articles
    text = re.sub(r"^(?:a|an|the)\s+", "", text, flags=re.IGNORECASE)

    # First, try to find a natural sentence boundary (period, exclamation, question mark)
    # Look for these within max_words to keep answers reasonably concise
    sentence_boundary_match = re.search(r'([^.!?]*[.!?])', text)
    if sentence_boundary_match:
        candidate = sentence_boundary_match.group(1).strip(" .!?,;")
        words_in_candidate = len(candidate.split())
        if words_in_candidate <= max_words:
            return candidate

    # Second, try phrase boundary separators (ordered by priority)
    phrase_separators = (
        ", followed by", ". ", ",",
        " and ", " or ", " while ", " as ", " because ", " which ",
    )
    for separator in phrase_separators:
        if separator in text:
            text = text.split(separator, 1)[0].strip()
            break

    # Finally, apply word-count truncation if necessary
    words = text.split()
    if len(words) > max_words:
        words = words[:max_words]

        # Remove trailing prepositions/articles to avoid incomplete phrases
        # (e.g., "power strip with 3 outlets sits on the" â†’ "power strip with 3 outlets sits on")
        trailing_preps = {"on", "in", "at", "to", "from", "by", "of", "with", "for",
                          "a", "an", "the", "and", "or", "as", "but"}
        while words and words[-1].lower() in trailing_preps:
            words.pop()

        # Ensure we still have content
        if not words:
            words = text.split()[:max_words]

    result = " ".join(words)
    return result.strip(" .,:;")


def _best_visual_modality(
        source_entries: dict[str, str],
        modality_reliability: dict[str, float],
) -> str:
    available_visual = [modality for modality in VISUAL_MODALITIES if modality in source_entries]
    if "rgb" in available_visual:
        return "rgb"
    if available_visual:
        return max(
            available_visual,
            key=lambda modality: (modality_reliability.get(modality, 0.0), -MODALITY_SORT_ORDER.get(modality, 99)),
        )
    if source_entries:
        return max(
            source_entries,
            key=lambda modality: (modality_reliability.get(modality, 0.0), -MODALITY_SORT_ORDER.get(modality, 99)),
        )
    return "fused"


def _primary_category_for_section(
        section: str,
        source_entries: dict[str, str],
        modality_reliability: dict[str, float],
) -> str:
    if section in {"scene_overview", "cross_modal_details", "final_unified_caption"}:
        return "fused"
    if section == "visible_objects_and_layout":
        return _best_visual_modality(source_entries, modality_reliability)
    if section == "motion_and_event_cues":
        if "event" in source_entries:
            return "event"
        return _best_visual_modality(source_entries, modality_reliability)
    if section == "audio_cues":
        if "audio" in source_entries:
            return "audio"
        if source_entries:
            return max(
                source_entries,
                key=lambda modality: (modality_reliability.get(modality, 0.0), -MODALITY_SORT_ORDER.get(modality, 99)),
            )
    return "fused"


def _supporting_modalities(
        source_entries: dict[str, str],
        modality_reliability: dict[str, float],
) -> list[str]:
    return sorted(
        source_entries.keys(),
        key=lambda modality: (-modality_reliability.get(modality, 0.0), MODALITY_SORT_ORDER.get(modality, 99),
                              modality),
    )


def _extract_scene_answer(caption: str) -> str:
    patterns = (
        r"(?:view|shot) of the ([^.]+?)(?: prepared for| setup|\.|,)",
        r"shows (?:a|an) ([^.]+?)(?: with|\.|,)",
        r"shows (?:the )?([^.]+? room)(?: with|\.|,)",
        r"shows (?:the )?([^.]+? workspace)(?: with|\.|,)",
        r"shows (?:the )?([^.]+? counter)(?: prepared|\.|,)",
    )
    for pattern in patterns:
        match = re.search(pattern, caption, flags=re.IGNORECASE)
        if match:
            return _trim_answer(match.group(1))
    if "kitchen" in caption.lower():
        return "kitchen counter"
    if "office" in caption.lower():
        return "office room"
    return ""


def _extract_layout_qa(caption: str) -> tuple[str, str]:
    directional_patterns = (
        r"To the (right|left) of the ([^.]+?) (?:is|sits|stands|lies|are)\s+(?:a|an|the) ([^.]+?)(?:[.,]|$)",
        r"The ([^.]+?) is located to the (right|left) of the ([^.]+?)(?:[.,]|$)",
    )
    for pattern in directional_patterns:
        match = re.search(pattern, caption, flags=re.IGNORECASE)
        if not match:
            continue
        groups = [group.strip() for group in match.groups()]
        if "located" not in pattern:
            direction, reference, answer = groups
        else:
            answer, direction, reference = groups
        return f"What is to the {direction} of the {reference}?", _trim_answer(answer)

    match = re.search(r"with (\d+) outlets", caption, flags=re.IGNORECASE)
    if match and "power strip" in caption.lower():
        return "How many outlets are on the power strip?", _trim_answer(match.group(1))

    match = re.search(r"Resting on the cutting board are ([^.]+?)(?:[.,]|$)", caption, flags=re.IGNORECASE)
    if match:
        items_text = match.group(1)
        items = [item.strip() for item in re.split(r",| and ", items_text) if item.strip()]
        if items:
            return "What is on the cutting board?", _trim_answer(items[0])

    return "What object is explicitly mentioned in the layout?", _trim_answer(caption)


def _extract_motion_qa(caption: str) -> tuple[str, str]:
    lower_caption = caption.lower()
    if "no independent dynamic entities" in lower_caption:
        return "Are any independent dynamic entities recognized?", "No"

    match = re.search(r"At approximately frame ([0-9]+), ([^.]+?) enter[s]? the frame", caption, flags=re.IGNORECASE)
    if match:
        frame, actor = match.groups()
        return f"What enters the frame at approximately frame {frame}?", _trim_answer(actor)

    match = re.search(r"The sequence begins with ([^.]+?)(?:[.,]|$)", caption, flags=re.IGNORECASE)
    if match:
        return "What does the sequence begin with?", _trim_answer(match.group(1))

    match = re.search(r"([^.]+?) enter[s]? the scene", caption, flags=re.IGNORECASE)
    if match:
        return "What enters the scene?", _trim_answer(match.group(1))

    return "What motion cue is described?", _trim_answer(caption)


def _extract_audio_qa(caption: str) -> tuple[str, str]:
    match = re.search(r"audio changes from [^.]* to ([^.]+?)(?:[.,]|$)", caption, flags=re.IGNORECASE)
    if match:
        return "What does the audio change to?", _trim_answer(match.group(1))

    match = re.search(r"sound of ([^.]+?)(?:,| followed by|\.|$)", caption, flags=re.IGNORECASE)
    if match:
        return "What sound is described?", _trim_answer(match.group(1))

    match = re.search(r"audio features ([^.]+?)(?:[.,]|$)", caption, flags=re.IGNORECASE)
    if match:
        return "What sound is featured in the audio?", _trim_answer(match.group(1))

    return "What audio cue is described?", _trim_answer(caption)


def _extract_cross_modal_qa(caption: str) -> tuple[str, str]:
    match = re.search(r"reinforced by ([^.]+?) that make", caption, flags=re.IGNORECASE)
    if match:
        return "Which modality makes the scene clearer?", _trim_answer(match.group(1))

    match = re.search(r"Motion-based evidence matches ([^.]+?)(?:[.,]|$)", caption, flags=re.IGNORECASE)
    if match:
        return "What sequence do the motion cues match?", _trim_answer(match.group(1))

    match = re.search(r"sounds align with ([^.]+?)(?:[.,]|$)", caption, flags=re.IGNORECASE)
    if match:
        return "What activity do the sounds align with?", _trim_answer(match.group(1))

    return "What cross-modal detail is stated?", _trim_answer(caption)


def _extract_final_qa(caption: str) -> tuple[str, str]:
    answer = _extract_scene_answer(caption)
    if answer:
        return "What setting is described in the unified caption?", answer

    match = re.search(r"At approximately frame ([0-9]+), ([^.]+?) enter[s]? the frame", caption, flags=re.IGNORECASE)
    if match:
        frame, actor = match.groups()
        return f"What enters the frame at approximately frame {frame}?", _trim_answer(actor)

    return "What does the unified caption describe?", _trim_answer(caption)


def _generate_field_qa(
        section: str,
        caption: str,
        source_entries: dict[str, str],
        modality_reliability: dict[str, float],
    semantic_facts: list[SemanticFact] | None = None,
) -> dict[str, Any]:
    category = _primary_category_for_section(section, source_entries, modality_reliability)
    qa = {
        "caption": caption,
        "question": "",
        "answer": "",
        "category": category,
        "supporting_modalities": _supporting_modalities(source_entries, modality_reliability),
        "qa_source": "generated_from_caption",
    }
    if not caption:
        return qa

    if semantic_facts:
        selected_fact = _choose_facts_for_section(semantic_facts, section, limit=1)
        if selected_fact:
            question, answer = _fact_to_qa(selected_fact[0], section)
            qa["question"] = _clean_whitespace(question)
            qa["answer"] = _clean_whitespace(answer or selected_fact[0].value or selected_fact[0].source_answer)
            qa["semantic_fact"] = _serialize_semantic_fact(selected_fact[0])
            qa["fact_type"] = selected_fact[0].fact_type
            qa["source_question"] = selected_fact[0].source_question
            qa["source_answer"] = selected_fact[0].source_answer
            qa["source_caption"] = selected_fact[0].source_caption
            qa["qa_source"] = "semantic_fact"
            return qa

    if section == "scene_overview":
        answer = _extract_scene_answer(caption)
        qa["question"] = "What setting is described in the scene overview?"
        qa["answer"] = answer or _trim_answer(caption)
        return qa

    builders = {
        "visible_objects_and_layout": _extract_layout_qa,
        "motion_and_event_cues": _extract_motion_qa,
        "audio_cues": _extract_audio_qa,
        "cross_modal_details": _extract_cross_modal_qa,
        "final_unified_caption": _extract_final_qa,
    }
    question, answer = builders.get(section, _extract_final_qa)(caption)
    qa["question"] = _clean_whitespace(question)
    qa["answer"] = answer or _trim_answer(caption)
    return qa


def _qa_from_evidence(section: str, evidence: CaptionEvidence) -> dict[str, Any]:
    qa = {
        "caption": evidence.caption or evidence.sentence,
        "question": evidence.question,
        "answer": evidence.answer,
        "category": evidence.annotation_key,
        "supporting_modalities": [evidence.modality],
        "qa_source": "original_modality",
    }
    if qa["question"] and qa["answer"]:
        return qa

    generated = _generate_field_qa(
        _legacy_section_for_qa_section(section),
        evidence.caption or evidence.sentence,
        {evidence.modality: evidence.source_key},
        {evidence.modality: evidence.modality_reliability},
    )
    generated["qa_source"] = "caption_fallback"
    return generated


def _section_for_category(category: str) -> str:
    return _sections_for_category(category)[0]


def _sections_for_category(category: str) -> list[str]:
    sections = CATEGORY_SECTIONS_MAP.get(category)
    if sections is None:
        sections = (CATEGORY_SECTION_MAP.get(category, "others"),)
    valid_sections = [section for section in sections if section in QA_SECTIONS]
    return list(dict.fromkeys(valid_sections)) or ["others"]


def _reliability_profile_for_category(category: str) -> dict[str, Any]:
    profile = CATEGORY_RELIABILITY_PROFILES.get(category, DEFAULT_CATEGORY_RELIABILITY_PROFILE)
    return {
        **DEFAULT_CATEGORY_RELIABILITY_PROFILE,
        **profile,
    }


def _support_weights_for_category(category: str, *, semantic_enabled: bool) -> dict[str, float]:
    if not semantic_enabled:
        return {"lexical": 1.0, "semantic": 0.0}
    profile = _reliability_profile_for_category(category)
    weights = profile.get("support_weights", DEFAULT_CATEGORY_RELIABILITY_PROFILE["support_weights"])
    return {
        "lexical": float(weights.get("lexical", 0.5)),
        "semantic": float(weights.get("semantic", 0.5)),
    }


def _modality_weights_for_category(category: str) -> dict[str, float]:
    profile = _reliability_profile_for_category(category)
    profile_name = str(profile.get("modality_weight_profile", "default"))
    return MODALITY_WEIGHT_PROFILES.get(profile_name, MODALITY_WEIGHT_PROFILES["default"])


def _normalize_semantic_cosine(cosine: float) -> float:
    normalized = (cosine - SEMANTIC_SUPPORT_COSINE_FLOOR) / (
        SEMANTIC_SUPPORT_COSINE_CEILING - SEMANTIC_SUPPORT_COSINE_FLOOR
    )
    return min(1.0, max(0.0, normalized))


def _evidence_support_text(evidence: CaptionEvidence) -> str:
    return _clean_whitespace(" ".join(
        part for part in (evidence.question, evidence.answer) if part
    )) or evidence.caption or evidence.sentence


def _support_match_record(evidence: CaptionEvidence, score: float) -> dict[str, Any]:
    sections = _sections_for_category(evidence.annotation_key)
    primary_section = sections[0]
    return {
        "source_modality": evidence.modality,
        "category": evidence.annotation_key,
        "section": primary_section,
        "primary_section": primary_section,
        "sections": sections,
        "source_key": evidence.source_key,
        "annotation_key": evidence.annotation_key,
        "qa_index": evidence.qa_index,
        "question": evidence.question,
        "answer": evidence.answer,
        "score": round(score, 3),
    }


_EMBEDDING_MODEL_CACHE: Any | None = None
_EMBEDDING_MODEL_LOAD_FAILED = False


def _load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> Any | None:
    global _EMBEDDING_MODEL_CACHE, _EMBEDDING_MODEL_LOAD_FAILED
    if _EMBEDDING_MODEL_CACHE is not None:
        return _EMBEDDING_MODEL_CACHE
    if _EMBEDDING_MODEL_LOAD_FAILED:
        return None
    try:
        from sentence_transformers import SentenceTransformer
        _EMBEDDING_MODEL_CACHE = SentenceTransformer(model_name)
        return _EMBEDDING_MODEL_CACHE
    except Exception:
        _EMBEDDING_MODEL_LOAD_FAILED = True
        return None


def _compute_embeddings(evidences: list[CaptionEvidence], *, model_name: str = DEFAULT_EMBEDDING_MODEL) -> list[Any] | None:
    model = _load_embedding_model(model_name)
    if model is None:
        return None
    texts = [_evidence_support_text(evidence) for evidence in evidences]
    try:
        return list(model.encode(texts, normalize_embeddings=True))
    except Exception:
        return None


def _embedding_similarity(embedding_a: Any, embedding_b: Any) -> float:
    try:
        return float(embedding_a @ embedding_b)
    except Exception:
        return sum(float(a) * float(b) for a, b in zip(embedding_a, embedding_b))


def _compute_text_embeddings(texts: list[str], *, model_name: str = DEFAULT_EMBEDDING_MODEL) -> list[Any] | None:
    model = _load_embedding_model(model_name)
    if model is None:
        return None
    try:
        return list(model.encode(texts, normalize_embeddings=True))
    except Exception:
        return None


def _qa_duplicate_text(evidence: CaptionEvidence) -> str:
    return _clean_whitespace(" ".join(
        part for part in (evidence.question, evidence.answer) if part
    )) or evidence.caption or evidence.sentence


def _has_duplicate_conflict(current: CaptionEvidence, previous: CaptionEvidence) -> bool:
    current_text = f"{current.question} {current.answer}"
    previous_text = f"{previous.question} {previous.answer}"
    return (
        _has_token_conflict(current_text, previous_text, _number_tokens)
        or _has_token_conflict(current_text, previous_text, _direction_tokens)
        or _has_token_conflict(current_text, previous_text, _polarity_tokens)
    )


def _answers_compatible_for_duplicate(current: CaptionEvidence, previous: CaptionEvidence) -> bool:
    current_answer = _clean_whitespace(current.answer).lower()
    previous_answer = _clean_whitespace(previous.answer).lower()
    if not current_answer or not previous_answer:
        return False
    if current_answer == previous_answer:
        return True
    if _is_yes_no_answer(current_answer) or _is_yes_no_answer(previous_answer):
        return False

    current_tokens = _tokenize(current_answer)
    previous_tokens = _tokenize(previous_answer)
    if not current_tokens or not previous_tokens:
        return False
    if _jaccard(current_tokens, previous_tokens) >= 0.7:
        return True
    shorter, longer = sorted((current_answer, previous_answer), key=len)
    return len(shorter.split()) >= 2 and shorter in longer


def _benchmark_qa_from_evidence(
        evidence: CaptionEvidence,
        *,
        section: str,
        fusion_score: float,
        selection_reason: str,
        reliability_gate: str,
        review_recommended: bool,
) -> dict[str, Any]:
    sections = _sections_for_category(evidence.annotation_key)
    primary_section = section if section in sections else sections[0]
    profile = _reliability_profile_for_category(evidence.annotation_key)
    qa = _qa_from_evidence(section, evidence)
    qa["section"] = primary_section
    qa["primary_section"] = primary_section
    qa["sections"] = sections
    qa["category"] = evidence.annotation_key
    qa["source_modality"] = evidence.modality
    qa["source_key"] = evidence.source_key
    qa["annotation_key"] = evidence.annotation_key
    qa["qa_index"] = evidence.qa_index
    qa["support_score"] = round(evidence.support_score, 3)
    qa["lexical_support_score"] = round(evidence.lexical_support_score, 3)
    qa["semantic_support_score"] = round(evidence.semantic_support_score, 3)
    qa["semantic_support_raw_cosine"] = round(evidence.semantic_support_raw_cosine, 3)
    qa["support_fusion_weights"] = {
        key: round(value, 3)
        for key, value in evidence.support_fusion_weights.items()
    }
    qa["lexical_support_match"] = evidence.lexical_support_match
    qa["semantic_support_match"] = evidence.semantic_support_match
    qa["support_by_modality"] = {
        key: round(value, 3)
        for key, value in evidence.support_by_modality.items()
    }
    qa["supporting_modalities"] = evidence.supporting_modalities
    qa["supporting_modality_count"] = evidence.supporting_modality_count
    qa["modality_reliability"] = round(evidence.modality_reliability, 3)
    qa["modality_weight_profile"] = profile["modality_weight_profile"]
    qa["requires_lexical_support"] = bool(profile.get("requires_lexical_support", False))
    qa["fusion_score"] = round(fusion_score, 3)
    qa["is_outlier"] = _looks_outlier(evidence)
    qa["selection_reason"] = selection_reason
    qa["reliability_gate"] = reliability_gate
    qa["review_recommended"] = review_recommended
    qa["review_priority"] = "none"
    qa["support_explanation"] = _support_explanation(evidence, reliability_gate)
    qa["source_caption"] = evidence.caption
    if evidence.original_qa and evidence.qa_index is not None:
        qa["source_question_block"] = evidence.original_qa["question"]
        qa["source_answer_block"] = evidence.original_qa["answer"]
    return qa


def _drop_record(
        evidence: CaptionEvidence,
        *,
        section: str,
        score: float,
        reason: str,
        reliability_gate: str = "",
        review_recommended: bool = False,
) -> dict[str, Any]:
    sections = _sections_for_category(evidence.annotation_key)
    primary_section = section if section in sections else sections[0]
    profile = _reliability_profile_for_category(evidence.annotation_key)
    return {
        "reason": reason,
        "section": primary_section,
        "primary_section": primary_section,
        "sections": sections,
        "category": evidence.annotation_key,
        "source_modality": evidence.modality,
        "source_key": evidence.source_key,
        "annotation_key": evidence.annotation_key,
        "qa_index": evidence.qa_index,
        "question": evidence.question,
        "answer": evidence.answer,
        "caption": evidence.caption,
        "support_score": round(evidence.support_score, 3),
        "lexical_support_score": round(evidence.lexical_support_score, 3),
        "semantic_support_score": round(evidence.semantic_support_score, 3),
        "semantic_support_raw_cosine": round(evidence.semantic_support_raw_cosine, 3),
        "lexical_support_match": evidence.lexical_support_match,
        "semantic_support_match": evidence.semantic_support_match,
        "support_by_modality": {
            key: round(value, 3)
            for key, value in evidence.support_by_modality.items()
        },
        "supporting_modalities": evidence.supporting_modalities,
        "supporting_modality_count": evidence.supporting_modality_count,
        "modality_reliability": round(evidence.modality_reliability, 3),
        "modality_weight_profile": profile["modality_weight_profile"],
        "requires_lexical_support": bool(profile.get("requires_lexical_support", False)),
        "fusion_score": round(score, 3),
        "is_outlier": _looks_outlier(evidence),
        "reliability_gate": reliability_gate,
        "review_recommended": review_recommended,
        "support_explanation": _support_explanation(evidence, reliability_gate),
    }


def _deduplicate_qas(
        scored_items: list[tuple[float, CaptionEvidence, str]],
) -> tuple[list[tuple[float, CaptionEvidence, str]], list[dict[str, Any]]]:
    selected: list[tuple[float, CaptionEvidence, str]] = []
    seen_exact: set[tuple[str, str]] = set()
    seen_tokens: list[set[str]] = []
    dropped: list[dict[str, Any]] = []
    sorted_items = sorted(scored_items, key=lambda item: item[0], reverse=True)
    duplicate_texts = [_qa_duplicate_text(evidence) for _, evidence, _ in sorted_items]
    embeddings = _compute_text_embeddings(duplicate_texts) if duplicate_texts else None
    selected_embedding_items: list[tuple[Any, CaptionEvidence, str, float]] = []

    for item_index, (score, evidence, section) in enumerate(sorted_items):
        question_key = _clean_whitespace(evidence.question).lower()
        answer_key = _clean_whitespace(evidence.answer).lower()
        exact_key = (question_key, answer_key)
        if question_key and answer_key and exact_key in seen_exact:
            dropped.append(_drop_record(evidence, section=section, score=score, reason="duplicate_exact_qa"))
            continue
        tokens = _tokenize(f"{evidence.question} {evidence.answer}")
        if tokens and any(_jaccard(tokens, previous) > 0.85 for previous in seen_tokens):
            dropped.append(_drop_record(evidence, section=section, score=score, reason="duplicate_similar_qa"))
            continue
        if embeddings is not None:
            current_embedding = embeddings[item_index]
            duplicate_match: tuple[float, CaptionEvidence, str, float] | None = None
            for previous_embedding, previous_evidence, previous_section, previous_score in selected_embedding_items:
                similarity = _embedding_similarity(current_embedding, previous_embedding)
                if similarity < SEMANTIC_DUPLICATE_THRESHOLD:
                    continue
                if _has_duplicate_conflict(evidence, previous_evidence):
                    continue
                if not _answers_compatible_for_duplicate(evidence, previous_evidence):
                    continue
                if duplicate_match is None or similarity > duplicate_match[0]:
                    duplicate_match = (similarity, previous_evidence, previous_section, previous_score)
            if duplicate_match is not None:
                similarity, previous_evidence, previous_section, previous_score = duplicate_match
                drop = _drop_record(evidence, section=section, score=score, reason="duplicate_semantic_qa")
                drop["duplicate_of"] = {
                    "section": previous_section,
                    "category": previous_evidence.annotation_key,
                    "source_modality": previous_evidence.modality,
                    "source_key": previous_evidence.source_key,
                    "qa_index": previous_evidence.qa_index,
                    "question": previous_evidence.question,
                    "answer": previous_evidence.answer,
                    "fusion_score": round(previous_score, 3),
                    "semantic_duplicate_score": round(similarity, 3),
                }
                dropped.append(drop)
                continue
        selected.append((score, evidence, section))
        if embeddings is not None:
            selected_embedding_items.append((embeddings[item_index], evidence, section, score))
        if question_key and answer_key:
            seen_exact.add(exact_key)
        if tokens:
            seen_tokens.append(tokens)
    return selected, dropped


def _qa_gate_for_category(category: str) -> tuple[str, bool]:
    gate = str(_reliability_profile_for_category(category).get("gate", "support_required"))
    return gate, gate == "review_recommended"


def _qa_gate_decision(
        evidence: CaptionEvidence,
        *,
        score: float,
        min_score: float,
) -> tuple[bool, str, str, bool]:
    category = evidence.annotation_key
    gate, review_recommended = _qa_gate_for_category(category)
    support = evidence.support_score

    if gate == "support_exempt":
        threshold = min(EXEMPT_RELIABLE_QA_MIN_SCORE, min_score)
        if score >= threshold:
            return True, "passed_support_exempt_gate", gate, review_recommended
        return False, "score_gate", gate, review_recommended

    if gate == "support_soft":
        if score < min_score:
            return False, "score_gate", gate, review_recommended
        if support >= DEFAULT_MIN_SUPPORT_SCORE or score >= SOFT_HIGH_SCORE:
            return True, "passed_support_soft_gate", gate, review_recommended
        return False, "low_cross_modal_support", gate, review_recommended

    if gate == "review_recommended":
        if score >= REVIEW_RELIABLE_QA_MIN_SCORE:
            return True, "passed_review_gate", gate, review_recommended
        return False, "score_gate", gate, review_recommended

    if score < min_score:
        return False, "score_gate", gate, review_recommended
    if support < STRICT_MIN_SUPPORT_SCORE:
        return False, "low_cross_modal_support", gate, review_recommended
    return True, "passed_support_required_gate", gate, review_recommended


def _number_tokens(text: str) -> set[str]:
    number_words = {
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    }
    tokens = set(re.findall(r"\b\d+\b", text.lower()))
    tokens.update(token for token in re.findall(r"[a-z]+", text.lower()) if token in number_words)
    return tokens


def _direction_tokens(text: str) -> set[str]:
    directions = {"left", "right", "above", "below", "front", "behind", "before", "after"}
    return {token for token in re.findall(r"[a-z]+", text.lower()) if token in directions}


def _polarity_tokens(text: str) -> set[str]:
    text = text.lower()
    tokens: set[str] = set()
    if re.search(r"\b(?:yes|true|visible|present)\b", text):
        tokens.add("positive")
    if re.search(r"\b(?:no|false|not|none|absent|invisible|without)\b", text):
        tokens.add("negative")
    return tokens


def _has_token_conflict(current: str, other: str, extractor: Any) -> bool:
    current_tokens = extractor(current)
    other_tokens = extractor(other)
    return bool(current_tokens and other_tokens and current_tokens != other_tokens)


def _category_mismatch_reasons(evidence: CaptionEvidence) -> list[str]:
    category = evidence.annotation_key
    qa_text = f"{evidence.question} {evidence.answer}".lower()
    question = evidence.question.lower().strip()
    reasons: list[str] = []

    counting_categories = {"counting", "event_counting", "depth_counting", "dynamic_counting",
                           "event_dynamic_counting", "depth_dynamic_counting"}
    spatial_categories = {"spatial_reasoning", "event_spatial_reasoning", "depth_spatial_reasoning",
                          "navigation", "event_navigation", "depth_navigation"}
    text_categories = {"text_recognition"}
    audio_categories = {"audio_hia", "audio_chronological_caption"}
    motion_categories = {"action", "dynamic_recognition", "dynamic_counting", "event_action",
                         "event_dynamic_recognition", "event_dynamic_counting", "depth_action",
                         "depth_dynamic_recognition", "depth_dynamic_counting", "scene_sequence",
                         "event_scene_sequence", "depth_scene_sequence"}

    if question.startswith(("how many", "how much")) and category not in counting_categories:
        reasons.append("category_mismatch_counting_hint")
    if question.startswith(("where", "which side", "in which direction")) and category not in spatial_categories:
        reasons.append("category_mismatch_spatial_hint")
    if any(word in qa_text for word in ("brand", "logo", "word", "text", "written", "sticker")) and category not in text_categories:
        reasons.append("category_mismatch_text_hint")
    if question.startswith(("what sound", "what audio", "which sound", "which audio")) and category not in audio_categories:
        reasons.append("category_mismatch_audio_hint")
    if question.startswith(("what action", "what is happening", "what occurs", "what does", "what do", "when")) and category not in motion_categories:
        reasons.append("category_mismatch_motion_hint")

    return reasons


def _answer_quality_reasons(evidence: CaptionEvidence) -> list[str]:
    question = _clean_whitespace(evidence.question).lower()
    answer = _clean_whitespace(evidence.answer).lower()
    if not answer:
        return []

    reasons: list[str] = []
    generic_answers = {"it", "this", "that", "there", "object", "thing", "something", "unknown", "unclear", "none"}
    uncertainty_terms = ("not sure", "unclear", "cannot tell", "can't tell", "unknown", "maybe", "possibly")

    if answer in generic_answers and not _is_yes_no_answer(answer):
        reasons.append("generic_answer")
    if any(term in answer for term in uncertainty_terms):
        reasons.append("uncertain_answer")
    if question and (answer == question or answer in question and len(answer.split()) >= 5):
        reasons.append("answer_repeats_question")

    return reasons


def _review_priority(review_reasons: list[str]) -> str:
    if not review_reasons:
        return "none"
    high_priority = {
        "possible_numeric_conflict",
        "possible_direction_conflict",
        "possible_negation_conflict",
        "no_supporting_modality",
        "answer_repeats_question",
        "uncertain_answer",
    }
    if any(reason in high_priority or reason.startswith("category_mismatch_") for reason in review_reasons):
        return "high"
    medium_priority = {
        "strict_category_low_lexical",
        "unparsed_numbered_block",
        "multi_clause_answer",
        "long_answer",
        "long_question",
        "semantic_high_lexical_low",
        "generic_answer",
    }
    if any(reason in medium_priority for reason in review_reasons):
        return "medium"
    return "low"


def _support_level(score: float) -> str:
    if score >= 0.45:
        return "strong"
    if score >= STRICT_MIN_SUPPORT_SCORE:
        return "moderate"
    if score >= DEFAULT_MIN_SUPPORT_SCORE:
        return "weak"
    return "very_weak"


def _support_explanation(evidence: CaptionEvidence, gate: str) -> dict[str, Any]:
    profile = _reliability_profile_for_category(evidence.annotation_key)
    best_support_modality = ""
    best_support_score = 0.0
    if evidence.support_by_modality:
        best_support_modality, best_support_score = max(
            evidence.support_by_modality.items(),
            key=lambda item: item[1],
        )
    best_match = evidence.semantic_support_match or evidence.lexical_support_match or {}
    return {
        "gate": gate,
        "modality_weight_profile": profile["modality_weight_profile"],
        "support_level": _support_level(evidence.support_score),
        "best_support_modality": best_support_modality,
        "best_support_score": round(best_support_score, 3),
        "best_match_modality": best_match.get("source_modality", "") if isinstance(best_match, dict) else "",
        "best_match_category": best_match.get("category", "") if isinstance(best_match, dict) else "",
        "supporting_modality_count": evidence.supporting_modality_count,
        "requires_lexical_support": bool(profile.get("requires_lexical_support", False)),
        "lexical_support_score": round(evidence.lexical_support_score, 3),
        "semantic_support_score": round(evidence.semantic_support_score, 3),
    }


def _review_reasons_for_evidence(evidence: CaptionEvidence, gate: str) -> list[str]:
    profile = _reliability_profile_for_category(evidence.annotation_key)
    reasons: list[str] = []
    if gate == "review_recommended":
        reasons.append("anomaly_category")
    if not (evidence.question and evidence.answer):
        reasons.append("caption_fallback")
    if len(evidence.answer.split()) > ANSWER_REVIEW_WORD_LIMIT:
        reasons.append("long_answer")
    if len(evidence.question.split()) > QUESTION_REVIEW_WORD_LIMIT:
        reasons.append("long_question")
    if _split_numbered_items(evidence.question) or _split_numbered_items(evidence.answer):
        reasons.append("unparsed_numbered_block")
    clause_count = len(re.findall(r"\b(?:and|or|while|because|then|followed by)\b", evidence.answer.lower()))
    if clause_count >= 2:
        reasons.append("multi_clause_answer")
    if evidence.semantic_support_score >= SEMANTIC_HIGH_REVIEW_THRESHOLD and evidence.lexical_support_score < LEXICAL_LOW_REVIEW_THRESHOLD:
        reasons.append("semantic_high_lexical_low")
    if profile.get("requires_lexical_support", False) and evidence.lexical_support_score < STRICT_MIN_LEXICAL_SUPPORT_SCORE:
        reasons.append("strict_category_low_lexical")
    if gate == "support_required" and evidence.supporting_modality_count < 1:
        reasons.append("no_supporting_modality")
    reasons.extend(_category_mismatch_reasons(evidence))
    reasons.extend(_answer_quality_reasons(evidence))

    match = evidence.semantic_support_match or evidence.lexical_support_match
    if match and isinstance(match, dict):
        other_answer = str(match.get("answer", ""))
        if _has_token_conflict(evidence.answer, other_answer, _number_tokens):
            reasons.append("possible_numeric_conflict")
        if _has_token_conflict(evidence.answer, other_answer, _direction_tokens):
            reasons.append("possible_direction_conflict")
        if _has_token_conflict(evidence.answer, other_answer, _polarity_tokens):
            reasons.append("possible_negation_conflict")

    return list(dict.fromkeys(reasons))


def _select_reliable_qas(
        evidences: list[CaptionEvidence],
        *,
        min_score: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    scored_items: list[tuple[float, CaptionEvidence, str, str, bool, str]] = []
    review_items: list[tuple[float, CaptionEvidence, str, str, bool, str, list[str]]] = []
    dropped: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = defaultdict(int)
    review_reason_counts: dict[str, int] = defaultdict(int)
    review_priority_counts: dict[str, int] = defaultdict(int)
    for evidence in evidences:
        section = _section_for_category(evidence.annotation_key)
        score = _qa_reliability_score(evidence)
        if evidence.qa_index is not None:
            score += 0.1
        if evidence.question and evidence.answer:
            score += 0.15
        if section == "others":
            score -= 0.2
        if _looks_outlier(evidence):
            gate, review_recommended = _qa_gate_for_category(evidence.annotation_key)
            dropped.append(_drop_record(
                evidence,
                section=section,
                score=score,
                reason="outlier",
                reliability_gate=gate,
                review_recommended=review_recommended,
            ))
            reason_counts["outlier"] += 1
            continue
        keep, reason, gate, review_recommended = _qa_gate_decision(evidence, score=score, min_score=min_score)
        if not keep:
            dropped.append(_drop_record(
                evidence,
                section=section,
                score=score,
                reason=reason,
                reliability_gate=gate,
                review_recommended=review_recommended,
            ))
            reason_counts[reason] += 1
            continue
        review_reasons = _review_reasons_for_evidence(evidence, gate)
        if review_reasons:
            for review_reason in review_reasons:
                review_reason_counts[review_reason] += 1
            review_priority_counts[_review_priority(review_reasons)] += 1
            review_items.append((score, evidence, section, reason, True, gate, review_reasons))
            continue
        scored_items.append((score, evidence, section, reason, review_recommended, gate))

    dedup_input = [(score, evidence, section) for score, evidence, section, _, _, _ in scored_items]
    deduped_items, dedup_drops = _deduplicate_qas(dedup_input)
    selected_keys = {
        (id(evidence), section)
        for _, evidence, section in deduped_items
    }
    dropped.extend(dedup_drops)
    for item in dedup_drops:
        reason_counts[item["reason"]] += 1

    metadata_by_key = {
        (id(evidence), section): (selection_reason, review_recommended, gate)
        for _, evidence, section, selection_reason, review_recommended, gate in scored_items
    }
    selected = [
        _benchmark_qa_from_evidence(
            evidence,
            section=section,
            fusion_score=score,
            selection_reason=metadata_by_key[(id(evidence), section)][0],
            review_recommended=metadata_by_key[(id(evidence), section)][1],
            reliability_gate=metadata_by_key[(id(evidence), section)][2],
        )
        for score, evidence, section in deduped_items
        if (id(evidence), section) in selected_keys
    ]
    review_recommended_qas = [
        {
            **_benchmark_qa_from_evidence(
                evidence,
                section=section,
                fusion_score=score,
                selection_reason=selection_reason,
                review_recommended=True,
                reliability_gate=gate,
            ),
            "review_reasons": review_reasons,
            "review_priority": _review_priority(review_reasons),
        }
        for score, evidence, section, selection_reason, _, gate, review_reasons in sorted(
            review_items,
            key=lambda item: item[0],
            reverse=True,
        )
    ]
    diagnostics = {
        "total_candidates": len(evidences),
        "passed_score_and_outlier_filters": len(scored_items) + len(review_items),
        "selected_count": len(selected),
        "review_count": len(review_recommended_qas),
        "dropped_count": len(dropped),
        "drop_reason_counts": dict(sorted(reason_counts.items())),
        "review_reason_counts": dict(sorted(review_reason_counts.items())),
        "review_priority_counts": dict(sorted(review_priority_counts.items())),
        "dropped_qas": dropped,
        "review_recommended_qas": review_recommended_qas,
    }
    return selected, review_recommended_qas, diagnostics


def _legacy_section_for_qa_section(section: str) -> str:
    if section in {"scene_and_context", "temporal_sequence"}:
        return "scene_overview"
    if section in {"objects_and_attributes", "spatial_and_layout", "text_and_symbols", "counting", "anomaly_and_safety", "others"}:
        return "visible_objects_and_layout"
    if section == "motion_and_action":
        return "motion_and_event_cues"
    if section == "audio_understanding":
        return "audio_cues"
    return "visible_objects_and_layout"


def _group_qas_by_section(qas: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = {section: [] for section in QA_SECTIONS}
    for qa in qas:
        sections = qa.get("sections")
        if not isinstance(sections, list) or not sections:
            sections = [qa.get("section", "others")]
        for section in sections:
            section_key = section if section in QA_SECTIONS else "others"
            grouped.setdefault(section_key, []).append(qa)
    return grouped


def _build_modality_support_summary(
        evidences: list[CaptionEvidence],
        selected_qas: list[dict[str, Any]],
        review_qas: list[dict[str, Any]],
        dropped_qas: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    selected_counts = Counter(str(qa.get("source_modality", "unknown")) for qa in selected_qas)
    review_counts = Counter(str(qa.get("source_modality", "unknown")) for qa in review_qas)
    dropped_counts = Counter(str(qa.get("source_modality", "unknown")) for qa in dropped_qas)
    by_modality: dict[str, list[CaptionEvidence]] = defaultdict(list)
    for evidence in evidences:
        by_modality[evidence.modality].append(evidence)

    for modality, items in sorted(by_modality.items()):
        support_values = [item.support_score for item in items]
        lexical_values = [item.lexical_support_score for item in items]
        semantic_values = [item.semantic_support_score for item in items]
        supporting_counts = [item.supporting_modality_count for item in items]
        avg_support = sum(support_values) / len(support_values) if support_values else 0.0
        avg_supporting_count = sum(supporting_counts) / len(supporting_counts) if supporting_counts else 0.0
        needs_review = avg_support < STRICT_MIN_SUPPORT_SCORE or avg_supporting_count < 0.5
        summary[modality] = {
            "candidate_count": len(items),
            "selected_count": selected_counts.get(modality, 0),
            "review_count": review_counts.get(modality, 0),
            "dropped_count": dropped_counts.get(modality, 0),
            "avg_support_score": round(avg_support, 3),
            "avg_lexical_support_score": round(sum(lexical_values) / len(lexical_values), 3) if lexical_values else 0.0,
            "avg_semantic_support_score": round(sum(semantic_values) / len(semantic_values), 3) if semantic_values else 0.0,
            "avg_supporting_modality_count": round(avg_supporting_count, 3),
            "needs_review": needs_review,
        }
    return summary


def _build_field_qas(
        sections: dict[str, str],
        source_entries: dict[str, str],
        modality_reliability: dict[str, float],
        semantic_facts: list[SemanticFact] | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        section: _generate_field_qa(
            section,
            sections.get(section, ""),
            source_entries,
            modality_reliability,
            semantic_facts,
        )
        for section in FIELD_QA_SECTIONS
    }


def _build_section_evidence_qas(
        sections: dict[str, str],
        selected_by_section: dict[str, list[CaptionEvidence]],
        source_entries: dict[str, str],
        modality_reliability: dict[str, float],
        qa_limits: dict[str, int],
) -> dict[str, list[dict[str, Any]]]:
    section_qas: dict[str, list[dict[str, Any]]] = {}

    for section in OUTPUT_SECTIONS:
        limit = qa_limits.get(section, 1)
        if limit <= 0:
            section_qas[section] = []
            continue

        if section == "final_unified_caption":
            caption = sections.get(section, "")
            qa = _generate_field_qa(section, caption, source_entries, modality_reliability)
            if qa.get("answer"):
                qa["source_modality"] = "fused"
                qa["annotation_key"] = "fused_final_unified_caption"
                qa["support_score"] = 0.0
                qa["modality_reliability"] = 1.0
                qa["fusion_score"] = None
                qa["is_outlier"] = False
                qa["selection_reason"] = "selected_summary"
                section_qas[section] = [qa]
            else:
                section_qas[section] = []
            continue

        selected = selected_by_section.get(section, [])[:limit]
        items: list[dict[str, Any]] = []
        for evidence in selected:
            qa = _qa_from_evidence(section, evidence)

            qa["source_modality"] = evidence.modality
            qa["annotation_key"] = evidence.annotation_key
            qa["qa_index"] = evidence.qa_index
            qa["support_score"] = round(evidence.support_score, 3)
            qa["modality_reliability"] = round(evidence.modality_reliability, 3)
            qa["fusion_score"] = round(_sentence_score(evidence, section), 3)
            qa["is_outlier"] = _looks_outlier(evidence)
            qa["selection_reason"] = "selected_topk"
            qa["source_caption"] = evidence.caption
            if evidence.original_qa and evidence.qa_index is not None:
                qa["source_question_block"] = evidence.original_qa["question"]
                qa["source_answer_block"] = evidence.original_qa["answer"]
            items.append(qa)

        if not items:
            caption = sections.get(section, "")
            fallback = _generate_field_qa(section, caption, source_entries, modality_reliability)
            if fallback.get("answer"):
                fallback["source_modality"] = "fused"
                fallback["annotation_key"] = f"fused_{section}_fallback"
                fallback["support_score"] = 0.0
                fallback["modality_reliability"] = 1.0
                fallback["fusion_score"] = None
                fallback["is_outlier"] = False
                fallback["selection_reason"] = "fallback_no_selected_evidence"
                items = [fallback]
            else:
                items = []

        section_qas[section] = items

    return section_qas


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


def _split_numbered_items(text: str) -> list[tuple[int, str]]:
    text = _clean_whitespace(text)
    if not text:
        return []

    matches = list(re.finditer(r"(?:^|\s)(\d+)\.\s*", text))
    if len(matches) < 2:
        return []

    items: list[tuple[int, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        value = _clean_whitespace(text[start:end].strip(" .;"))
        if value:
            items.append((int(match.group(1)), value))
    return items


def _split_qa_pairs(question: str, answer: str) -> list[tuple[int | None, str, str]]:
    question_items = _split_numbered_items(question)
    answer_items = _split_numbered_items(answer)
    if not question_items or not answer_items:
        return [(None, question, answer)] if question and answer else []

    answers_by_index = {index: value for index, value in answer_items}
    pairs = [
        (index, item_question, answers_by_index[index])
        for index, item_question in question_items
        if index in answers_by_index
    ]
    if pairs:
        return pairs

    if len(question_items) == len(answer_items):
        return [
            (question_item[0], question_item[1], answer_item[1])
            for question_item, answer_item in zip(question_items, answer_items)
        ]

    return [(None, question, answer)] if question and answer else []


def _extract_caption_evidence(modality: str, source_key: str, payload: dict[str, Any]) -> list[CaptionEvidence]:
    annotations = payload.get("annotations", {})
    if not isinstance(annotations, dict):
        return []

    evidences: list[CaptionEvidence] = []
    for annotation_key, annotation_payload in annotations.items():
        if not isinstance(annotation_payload, dict):
            continue

        original_question = annotation_payload.get("question")
        original_answer = annotation_payload.get("answer")
        question_text = _clean_whitespace(str(original_question)) if original_question is not None else ""
        answer_text = _clean_whitespace(str(original_answer)) if original_answer is not None else ""

        caption = annotation_payload.get("caption")
        caption_text = _clean_whitespace(caption) if isinstance(caption, str) else ""
        if not caption_text and not (question_text or answer_text):
            continue

        original_qa = None
        if question_text and answer_text:
            original_qa = {
                "question": question_text,
                "answer": answer_text,
            }

        qa_pairs = _split_qa_pairs(question_text, answer_text)
        if not qa_pairs:
            qa_pairs = [(None, question_text, answer_text)]

        for qa_index, item_question, item_answer in qa_pairs:
            evidence_text = _clean_whitespace(" ".join(
                part for part in (item_question, item_answer) if part
            ))
            if not evidence_text:
                evidence_text = caption_text
            sentence = caption_text or evidence_text
            tokens = _tokenize(evidence_text)
            evidences.append(
                CaptionEvidence(
                    modality=modality,
                    source_key=source_key,
                    annotation_key=annotation_key,
                    question=item_question,
                    answer=item_answer,
                    caption=caption_text,
                    sentence=sentence,
                    normalized_tokens=tokens,
                    qa_index=qa_index,
                    original_qa=original_qa,
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

    embeddings = _compute_embeddings(evidences)
    semantic_enabled = embeddings is not None
    evidence_indices = {id(evidence): index for index, evidence in enumerate(evidences)}

    for evidence in evidences:
        other_evidences = [
            other
            for modality, items in by_modality.items()
            if modality != evidence.modality
            for other in items
        ]
        lexical_score = 0.0
        lexical_match: CaptionEvidence | None = None
        for other in other_evidences:
            if not other.normalized_tokens:
                continue
            score = _jaccard(evidence.normalized_tokens, other.normalized_tokens)
            if score > lexical_score:
                lexical_score = score
                lexical_match = other

        semantic_score = 0.0
        semantic_raw_cosine = 0.0
        semantic_match: CaptionEvidence | None = None
        support_by_modality: dict[str, float] = {}
        if semantic_enabled and embeddings is not None:
            evidence_embedding = embeddings[evidence_indices[id(evidence)]]
            for other in other_evidences:
                raw_cosine = _embedding_similarity(evidence_embedding, embeddings[evidence_indices[id(other)]])
                score = _normalize_semantic_cosine(raw_cosine)
                if score > semantic_score:
                    semantic_score = score
                    semantic_raw_cosine = raw_cosine
                    semantic_match = other

        weights = _support_weights_for_category(evidence.annotation_key, semantic_enabled=semantic_enabled)
        support = weights["lexical"] * lexical_score + weights["semantic"] * semantic_score
        for modality, items in by_modality.items():
            if modality == evidence.modality:
                continue
            modality_best = 0.0
            for other in items:
                lexical = _jaccard(evidence.normalized_tokens, other.normalized_tokens) if other.normalized_tokens else 0.0
                semantic = 0.0
                if semantic_enabled and embeddings is not None:
                    raw_cosine = _embedding_similarity(
                        embeddings[evidence_indices[id(evidence)]],
                        embeddings[evidence_indices[id(other)]],
                    )
                    semantic = _normalize_semantic_cosine(raw_cosine)
                modality_best = max(modality_best, weights["lexical"] * lexical + weights["semantic"] * semantic)
            support_by_modality[modality] = modality_best
        supporting_modalities = sorted(
            modality
            for modality, score in support_by_modality.items()
            if score >= SUPPORTING_MODALITY_THRESHOLD
        )
        evidence.lexical_support_score = lexical_score
        evidence.semantic_support_score = semantic_score
        evidence.semantic_support_raw_cosine = semantic_raw_cosine
        evidence.support_score = support
        evidence.support_fusion_weights = dict(weights)
        evidence.lexical_support_match = _support_match_record(lexical_match, lexical_score) if lexical_match else None
        evidence.semantic_support_match = (
            _support_match_record(semantic_match, semantic_raw_cosine) if semantic_match else None
        )
        evidence.support_by_modality = support_by_modality
        evidence.supporting_modalities = supporting_modalities
        evidence.supporting_modality_count = len(supporting_modalities)
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


def _qa_reliability_score(evidence: CaptionEvidence) -> float:
    weights = _modality_weights_for_category(evidence.annotation_key)
    score = weights.get(evidence.modality, 0.3)
    score += evidence.modality_reliability * 0.6
    score += evidence.support_score * 1.2
    if evidence.annotation_key not in CATEGORY_RELIABILITY_PROFILES:
        score -= 0.15
    if _looks_outlier(evidence):
        score -= 1.0
    return score


def _section_route_bonus(evidence: CaptionEvidence, section: str) -> float:
    annotation_key = evidence.annotation_key.lower()
    tokens = evidence.normalized_tokens
    qa_text = f"{evidence.question} {evidence.answer}".lower()
    question = evidence.question.lower().strip()

    if section == "scene_overview":
        bonus = 0.0
        if any(hint in annotation_key for hint in ("overview", "setting", "environment", "context")):
            bonus += 0.6
        if any(phrase in qa_text for phrase in ("setting", "environment", "scene context", "overall scene")):
            bonus += 0.4
        bonus += min(0.25, 0.05 * len(tokens & SECTION_KEYWORDS[section]))
        if question.startswith(("where", "how many", "what action", "what tool", "what object", "is ", "are ", "was ")):
            bonus -= 0.45
        if any(hint in annotation_key for hint in ("layout", "spatial", "count", "object", "motion", "event", "audio", "sound", "action", "dynamic")):
            bonus -= 0.3
        return bonus

    if section == "visible_objects_and_layout":
        bonus = 0.0
        if any(hint in annotation_key for hint in ("layout", "spatial", "count", "object")):
            bonus += 0.55
        if question.startswith(("where", "how many", "which object", "what object", "what item", "what brand", "what logo")):
            bonus += 0.45
        bonus += min(0.35, 0.08 * len(tokens & SECTION_KEYWORDS[section]))
        if any(hint in annotation_key for hint in ("scene", "overview", "audio", "sound", "motion", "event")):
            bonus -= 0.15
        return bonus

    if section == "motion_and_event_cues":
        bonus = 0.0
        if any(
                hint in annotation_key
                for hint in
                ("dynamic", "action", "scene_sequence", "event_action", "event_dynamic", "event_scene", "motion")
        ):
            bonus += 0.7
        if question.startswith(("what action", "what tool do", "what am i doing", "what do i", "what does", "when")):
            bonus += 0.45
        if any(token in qa_text for token in ("peel", "slice", "pick up", "put down", "grasp", "lift", "enter", "move")):
            bonus += 0.25
        bonus += min(0.35, 0.08 * len(tokens & SECTION_KEYWORDS[section]))
        if not any(token in tokens for token in
                   ("move", "motion", "moving", "enter", "lift", "reach", "pick", "grasp", "peel", "contact",
                    "activity", "interaction", "hands")):
            bonus -= 0.2
        return bonus

    if section == "audio_cues":
        bonus = 0.0
        if any(hint in annotation_key for hint in ("audio", "sound", "speech", "music", "noise")):
            bonus += 0.75
        bonus += min(0.4, 0.1 * len(tokens & SECTION_KEYWORDS[section]))
        if "audio" not in tokens and not any(token in tokens for token in
                                             ("sound", "speech", "music", "noise", "silence", "scraping", "chopping",
                                              "rhythmic")):
            bonus -= 0.3
        return bonus

    return 0.0


def _deduplicate_sentences(scored_sentences: list[tuple[float, CaptionEvidence]]) -> list[CaptionEvidence]:
    selected: list[CaptionEvidence] = []
    for _, evidence in sorted(scored_sentences, key=lambda item: item[0], reverse=True):
        if any(_jaccard(evidence.normalized_tokens, existing.normalized_tokens) > 0.7 for existing in selected):
            continue
        selected.append(evidence)
    return selected


def _select_sentences(
        evidences: list[CaptionEvidence],
        section: str,
        limit: int,
        *,
        min_score: float = DEFAULT_MIN_SCORE,
        relax_filters: bool = False,
        section_diagnostics: dict[str, Any] | None = None,
) -> list[CaptionEvidence]:
    diagnostics: dict[str, Any] | None = None
    if section_diagnostics is not None:
        diagnostics = {
            "total_candidates": 0,
            "dropped_keyword_gate": 0,
            "dropped_annotation_gate": 0,
            "dropped_score_gate": 0,
            "outlier_count": 0,
            "selected_after_dedup": 0,
            "selected_after_preference": 0,
            "selected_after_limit": 0,
            "min_score": min_score,
            "relax_filters": relax_filters,
            "outlier_examples": [],
            "dropped_examples": [],
        }

    if section == "audio_cues" and not any(item.modality == "audio" for item in evidences):
        if diagnostics is not None:
            diagnostics["total_candidates"] = len(evidences)
            if section_diagnostics is not None:
                section_diagnostics[section] = diagnostics
        return []

    eligible: list[tuple[float, CaptionEvidence]] = []
    for evidence in evidences:
        if diagnostics is not None:
            diagnostics["total_candidates"] += 1
        keyword_hits = _keyword_hits(evidence.normalized_tokens, section)
        if not relax_filters:
            if section in {"scene_overview", "audio_cues"} and keyword_hits == 0:
                if diagnostics is not None:
                    diagnostics["dropped_keyword_gate"] += 1
                    if len(diagnostics["dropped_examples"]) < 20:
                        diagnostics["dropped_examples"].append(
                            {
                                "reason": "keyword_gate",
                                "modality": evidence.modality,
                                "annotation_key": evidence.annotation_key,
                                "sentence": evidence.sentence,
                            }
                        )
                continue
            if section == "visible_objects_and_layout" and keyword_hits == 0 and "spatial" not in evidence.annotation_key and "count" not in evidence.annotation_key and "object" not in evidence.annotation_key:
                if diagnostics is not None:
                    diagnostics["dropped_annotation_gate"] += 1
                    if len(diagnostics["dropped_examples"]) < 20:
                        diagnostics["dropped_examples"].append(
                            {
                                "reason": "annotation_gate",
                                "modality": evidence.modality,
                                "annotation_key": evidence.annotation_key,
                                "sentence": evidence.sentence,
                            }
                        )
                continue
            if section == "motion_and_event_cues":
                if keyword_hits == 0 and not any(
                        hint in evidence.annotation_key
                        for hint in
                        ("dynamic", "action", "scene_sequence", "event_action", "event_dynamic", "event_scene")
                ):
                    if diagnostics is not None:
                        diagnostics["dropped_annotation_gate"] += 1
                        if len(diagnostics["dropped_examples"]) < 20:
                            diagnostics["dropped_examples"].append(
                                {
                                    "reason": "annotation_gate",
                                    "modality": evidence.modality,
                                    "annotation_key": evidence.annotation_key,
                                    "sentence": evidence.sentence,
                                }
                            )
                    continue
        score = _sentence_score(evidence, section)
        is_outlier = _looks_outlier(evidence)
        if diagnostics is not None and is_outlier:
            diagnostics["outlier_count"] += 1
            if len(diagnostics["outlier_examples"]) < 5:
                diagnostics["outlier_examples"].append(
                    {
                        "modality": evidence.modality,
                        "annotation_key": evidence.annotation_key,
                        "score": round(score, 3),
                        "support_score": round(evidence.support_score, 3),
                        "modality_reliability": round(evidence.modality_reliability, 3),
                        "sentence": evidence.sentence,
                    }
                )
        if score <= min_score:
            if diagnostics is not None:
                diagnostics["dropped_score_gate"] += 1
                if len(diagnostics["dropped_examples"]) < 20:
                    diagnostics["dropped_examples"].append(
                        {
                            "reason": "score_gate",
                            "modality": evidence.modality,
                            "annotation_key": evidence.annotation_key,
                            "sentence": evidence.sentence,
                            "score": round(score, 3),
                            "support_score": round(evidence.support_score, 3),
                            "modality_reliability": round(evidence.modality_reliability, 3),
                            "is_outlier": is_outlier,
                        }
                    )
            continue
        eligible.append((score, evidence))

    selected = _deduplicate_sentences(eligible)
    if diagnostics is not None:
        diagnostics["selected_after_dedup"] = len(selected)
    if section == "audio_cues":
        selected = [item for item in selected if item.modality == "audio"] or selected
    elif section == "motion_and_event_cues":
        preferred = [item for item in selected if item.modality in {"event", "rgb", "ir"}]
        selected = preferred or selected
    elif section == "visible_objects_and_layout":
        preferred = [item for item in selected if item.modality in {"rgb", "ir", "depth"}]
        selected = preferred or selected

    if diagnostics is not None:
        diagnostics["selected_after_preference"] = len(selected)
        diagnostics["selected_after_limit"] = min(len(selected), limit)
        if section_diagnostics is not None:
            section_diagnostics[section] = diagnostics

    return selected[:limit]


def _assign_evidences_to_sections(
        evidences: list[CaptionEvidence],
        *,
        min_score: float,
        relax_filters: bool,
        section_diagnostics: dict[str, Any] | None = None,
) -> dict[str, list[CaptionEvidence]]:
    selected_by_section: dict[str, list[tuple[float, CaptionEvidence]]] = {section: [] for section in PRIMARY_SECTIONS}

    for evidence in evidences:
        scored_sections: list[tuple[float, str]] = []
        for section in PRIMARY_SECTIONS:
            score = _sentence_score(evidence, section) + _section_route_bonus(evidence, section)
            scored_sections.append((score, section))

        scored_sections.sort(reverse=True)
        best_score, best_section = scored_sections[0]
        second_best_score = scored_sections[1][0] if len(scored_sections) > 1 else float("-inf")

        if best_score <= min_score:
            continue
        if not relax_filters and best_score - second_best_score < 0.02:
            if evidence.modality == "audio":
                best_section = "audio_cues"
            elif evidence.modality == "event":
                best_section = "motion_and_event_cues"
            elif evidence.modality in {"rgb", "ir", "depth"}:
                best_section = "visible_objects_and_layout"

        selected_by_section[best_section].append((best_score, evidence))

    if section_diagnostics is not None:
        for section, scored_items in selected_by_section.items():
            section_diagnostics[section] = {
                "total_candidates": len(evidences),
                "assigned_after_route": len(scored_items),
                "selected_after_limit": len(scored_items),
                "min_score": min_score,
                "relax_filters": relax_filters,
                "dropped_examples": [],
            }

    return {
        section: _deduplicate_sentences(scored_items)
        for section, scored_items in selected_by_section.items()
    }


def _join_sentences(evidences: list[CaptionEvidence]) -> str:
    if not evidences:
        return ""
    return " ".join(item.sentence.rstrip(".") + "." for item in evidences)


def _join_evidence_captions(evidences: list[CaptionEvidence], limit: int) -> str:
    selected: list[CaptionEvidence] = []
    seen_tokens: list[set[str]] = []
    for evidence in evidences:
        tokens = evidence.normalized_tokens
        if tokens and any(_jaccard(tokens, previous) > 0.65 for previous in seen_tokens):
            continue
        selected.append(evidence)
        seen_tokens.append(tokens)
        if len(selected) >= limit:
            break
    return _join_sentences(selected)


def _build_cross_modal_details(
        evidences: list[CaptionEvidence],
        *,
        min_score: float = DEFAULT_MIN_SCORE,
        relax_filters: bool = False,
) -> str:
    reliable_modalities = {
        evidence.modality
        for evidence in evidences
        if evidence.modality_reliability >= 0.5
    }

    details: list[str] = []

    if {"rgb", "ir"} <= reliable_modalities:
        visual_selected = _select_sentences(
            evidences,
            "visible_objects_and_layout",
            limit=1,
            min_score=min_score,
            relax_filters=relax_filters,
        )
        if visual_selected:
            details.append(
                f"RGB and IR cues agree on the visual layout: {visual_selected[0].sentence.rstrip('.')}."
            )

    if {"event", "rgb"} <= reliable_modalities or {"event", "ir"} <= reliable_modalities:
        motion_selected = _select_sentences(
            evidences,
            "motion_and_event_cues",
            limit=1,
            min_score=min_score,
            relax_filters=relax_filters,
        )
        if motion_selected:
            details.append(
                f"Event evidence is consistent with the observed motion: {motion_selected[0].sentence.rstrip('.')}."
            )

    if {"audio", "rgb"} <= reliable_modalities or {"audio", "ir"} <= reliable_modalities or {"audio",
                                                                                             "event"} <= reliable_modalities:
        audio_selected = _select_sentences(
            evidences,
            "audio_cues",
            limit=1,
            min_score=min_score,
            relax_filters=relax_filters,
        )
        if audio_selected:
            details.append(
                f"Audio evidence matches the rest of the sample: {audio_selected[0].sentence.rstrip('.')}."
            )

    if not details:
        selected = _select_sentences(
            evidences,
            "cross_modal_details",
            limit=2,
            min_score=min_score,
            relax_filters=relax_filters,
        )
        return _join_sentences(selected)

    return " ".join(details[:2])


def _build_final_caption(sections: dict[str, str]) -> str:
    parts: list[str] = []
    for key in ("scene_overview", "visible_objects_and_layout", "motion_and_event_cues", "audio_cues",
                "cross_modal_details"):
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


def fuse_sample(
        sample_modalities: dict[str, dict[str, Any]],
        *,
        relax_filters: bool = False,
        collect_diagnostics: bool = False,
) -> dict[str, Any]:
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

    section_limits = RELAXED_SECTION_LIMITS if relax_filters else DEFAULT_SECTION_LIMITS
    qa_limits = RELAXED_SECTION_QA_LIMITS if relax_filters else DEFAULT_SECTION_QA_LIMITS
    min_score = RELAXED_MIN_SCORE if relax_filters else DEFAULT_MIN_SCORE
    reliable_qa_min_score = RELAXED_RELIABLE_QA_MIN_SCORE if relax_filters else DEFAULT_RELIABLE_QA_MIN_SCORE
    section_diagnostics: dict[str, Any] | None = {} if collect_diagnostics else None

    selected_by_section = _assign_evidences_to_sections(
        evidences,
        min_score=min_score,
        relax_filters=relax_filters,
        section_diagnostics=section_diagnostics,
    )

    sections = {}
    for section in PRIMARY_SECTIONS:
        sections[section] = _join_evidence_captions(selected_by_section[section], section_limits[section])
    sections["cross_modal_details"] = _build_cross_modal_details(
        evidences,
        min_score=min_score,
        relax_filters=relax_filters,
    )
    sections["final_unified_caption"] = _build_final_caption(sections)

    modality_reliability: dict[str, float] = {}
    for evidence in evidences:
        score = modality_reliability.get(evidence.modality, 0.0)
        modality_reliability[evidence.modality] = max(score, evidence.modality_reliability)

    rounded_reliability = {key: round(value, 3) for key, value in sorted(modality_reliability.items())}
    selected_reliable_qas, review_recommended_qas, qa_selection_diagnostics = _select_reliable_qas(
        evidences,
        min_score=reliable_qa_min_score,
    )
    semantic_support_enabled = any(
        evidence.support_fusion_weights.get("semantic", 0.0) > 0
        for evidence in evidences
    )
    modality_support_summary = _build_modality_support_summary(
        evidences,
        selected_reliable_qas,
        review_recommended_qas,
        qa_selection_diagnostics.get("dropped_qas", []),
    )
    qa_selection_diagnostics["support_scoring"] = {
        "lexical_enabled": True,
        "semantic_enabled": semantic_support_enabled,
        "embedding_model": DEFAULT_EMBEDDING_MODEL if semantic_support_enabled else None,
    }
    qa_selection_diagnostics["modality_support_summary"] = modality_support_summary
    qas_by_section = _group_qas_by_section(selected_reliable_qas)
    review_qas_by_section = _group_qas_by_section(review_recommended_qas)

    fused_sample = {
        "source_entries": source_entries,
        "source_files": source_files,
        "modality_reliability": rounded_reliability,
        **sections,
        "selected_reliable_qas": selected_reliable_qas,
        "review_recommended_qas": review_recommended_qas,
        "qas_by_section": qas_by_section,
        "review_qas_by_section": review_qas_by_section,
    }

    if collect_diagnostics:
        diagnostic_facts = [
            _build_semantic_fact(
                modality=evidence.modality,
                source_key=evidence.source_key,
                annotation_key=evidence.annotation_key,
                section=evidence.annotation_key,
                caption=evidence.caption,
                question=evidence.question,
                answer=evidence.answer,
                confidence=0.85,
                support_score=evidence.support_score,
                modality_reliability=evidence.modality_reliability,
            )
            for evidence in evidences
        ]
        fused_sample["fusion_diagnostics"] = {
            "relax_filters": relax_filters,
            "min_score": min_score,
            "reliable_qa_min_score": reliable_qa_min_score,
            "section_limits": section_limits,
            "sections": section_diagnostics,
            "qa_selection": qa_selection_diagnostics,
            "semantic_facts": [_serialize_semantic_fact(fact) for fact in diagnostic_facts],
            "semantic_scene_graph": {
                section: [
                    _serialize_semantic_fact(fact)
                    for fact in _choose_facts_for_section(diagnostic_facts, section, limit=qa_limits.get(section, 1))
                ]
                for section in OUTPUT_SECTIONS
            },
        }

    return fused_sample


def run_late_fusion(
        input_dir: Path | str = ".",
        output_file: Path | str = "fused_qa_results.json",
        modality_files: dict[str, str] | None = None,
        relax_filters: bool = False,
        collect_diagnostics: bool = False,
        diagnostics_output_file: Path | str = "fusion_diagnostics.json",
        generate_analysis: bool = True,
        analysis_output_json: Path | str = "fusion_qa_stats.json",
        analysis_output_csv: Path | str = "fusion_qa_rows.csv",
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
    diagnostics_results: dict[str, Any] = {}

    for sample_key in sorted(grouped_samples):
        sample_result = fuse_sample(
            grouped_samples[sample_key],
            relax_filters=relax_filters,
            collect_diagnostics=collect_diagnostics,
        )
        if collect_diagnostics:
            diagnostics_results[sample_key] = sample_result.pop("fusion_diagnostics", {})
        fused_results[sample_key] = sample_result

    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(fused_results, handle, indent=2, ensure_ascii=False)

    if collect_diagnostics:
        diagnostics_output_file = Path(diagnostics_output_file)
        with open(diagnostics_output_file, "w", encoding="utf-8") as handle:
            json.dump(diagnostics_results, handle, indent=2, ensure_ascii=False)

    if generate_analysis:
        from annotation_feature.analyze_fusion_outputs import build_reports

        diagnostics_path = Path(diagnostics_output_file) if collect_diagnostics else Path("__missing_fusion_diagnostics__.json")
        build_reports(
            fused_results_path=output_file,
            diagnostics_path=diagnostics_path,
            output_json_path=Path(analysis_output_json),
            output_csv_path=Path(analysis_output_csv),
        )

    return fused_results


def format_fused_sections(result: dict[str, Any]) -> str:
    lines: list[str] = []
    for section_key in OUTPUT_SECTIONS:
        lines.append(f"{SECTION_LABELS[section_key]}:")
        lines.append(result.get(section_key, ""))
        lines.append("")
    return "\n".join(lines).rstrip()
