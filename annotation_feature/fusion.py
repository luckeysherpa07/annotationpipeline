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
    "cross_modal_details": 2,
    "final_unified_caption": 1,
}

RELAXED_SECTION_QA_LIMITS = {
    "scene_overview": 4,
    "visible_objects_and_layout": 6,
    "motion_and_event_cues": 4,
    "audio_cues": 4,
    "cross_modal_details": 3,
    "final_unified_caption": 1,
}

DEFAULT_MIN_SCORE = 0.45
RELAXED_MIN_SCORE = 0.2

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

VISUAL_MODALITIES = ("rgb", "ir", "depth")
FIELD_QA_SECTIONS = OUTPUT_SECTIONS
MODALITY_SORT_ORDER = {
    "rgb": 0,
    "ir": 1,
    "depth": 2,
    "event": 3,
    "audio": 4,
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
    original_qa: dict[str, str] | None = None  # Preserve original modality Q/A as fallback


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
        # (e.g., "power strip with 3 outlets sits on the" → "power strip with 3 outlets sits on")
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
        key=lambda modality: (-modality_reliability.get(modality, 0.0), MODALITY_SORT_ORDER.get(modality, 99), modality),
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
) -> dict[str, Any]:
    category = _primary_category_for_section(section, source_entries, modality_reliability)
    qa = {
        "caption": caption,
        "question": "",
        "answer": "",
        "category": category,
        "supporting_modalities": _supporting_modalities(source_entries, modality_reliability),
    }
    if not caption:
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


def _build_field_qas(
    sections: dict[str, str],
    source_entries: dict[str, str],
    modality_reliability: dict[str, float],
) -> dict[str, dict[str, Any]]:
    return {
        section: _generate_field_qa(section, sections.get(section, ""), source_entries, modality_reliability)
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
                qa["drop_reason"] = "selected_summary"
                section_qas[section] = [qa]
            else:
                section_qas[section] = []
            continue

        selected = selected_by_section.get(section, [])[:limit]
        items: list[dict[str, Any]] = []
        for evidence in selected:
            qa = _generate_field_qa(section, evidence.sentence, source_entries, modality_reliability)
            
            # Fallback to original modality Q/A if extraction is poor quality
            # (empty answer or extracted answer appears incomplete)
            extracted_answer = qa.get("answer", "").strip()
            word_count = len(extracted_answer.split()) if extracted_answer else 0
            ends_with_preposition = extracted_answer.endswith(
                (" of the", " of a", " of an", " the", " a", " an", " to", " in", " on")
            )
            
            # Use original Q/A if:
            # 1. No extracted answer at all, OR
            # 2. Answer is extremely short (< 2 words), OR
            # 3. Answer ends with incomplete preposition (e.g., "To the" without the object)
            should_use_original = evidence.original_qa and (
                not extracted_answer 
                or word_count < 2  # Extremely short (less than 2 words)
                or ends_with_preposition  # Ends with incomplete preposition
            )
            
            if should_use_original:
                qa["question"] = evidence.original_qa["question"]
                qa["answer"] = evidence.original_qa["answer"]
                qa["qa_source"] = "original_modality"  # Mark that we used original Q/A
            else:
                qa["qa_source"] = "extracted"  # Mark that we used extracted Q/A
            
            qa["source_modality"] = evidence.modality
            qa["annotation_key"] = evidence.annotation_key
            qa["support_score"] = round(evidence.support_score, 3)
            qa["modality_reliability"] = round(evidence.modality_reliability, 3)
            qa["fusion_score"] = round(_sentence_score(evidence, section), 3)
            qa["is_outlier"] = _looks_outlier(evidence)
            qa["drop_reason"] = "selected_topk"
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
                fallback["drop_reason"] = "fallback_no_selected_evidence"
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
        
        # Extract original Q/A pair from this annotation as fallback
        original_qa = None
        original_question = annotation_payload.get("question")
        original_answer = annotation_payload.get("answer")
        if original_question and original_answer:
            original_qa = {
                "question": _clean_whitespace(str(original_question)),
                "answer": _clean_whitespace(str(original_answer)),
            }
        
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


def _section_route_bonus(evidence: CaptionEvidence, section: str) -> float:
    annotation_key = evidence.annotation_key.lower()
    tokens = evidence.normalized_tokens

    if section == "scene_overview":
        bonus = 0.0
        if any(hint in annotation_key for hint in ("scene", "overview", "setting", "environment", "context")):
            bonus += 0.6
        bonus += min(0.25, 0.05 * len(tokens & SECTION_KEYWORDS[section]))
        if any(hint in annotation_key for hint in ("layout", "spatial", "object", "motion", "event", "audio", "sound")):
            bonus -= 0.3
        return bonus

    if section == "visible_objects_and_layout":
        bonus = 0.0
        if any(hint in annotation_key for hint in ("layout", "spatial", "count", "object")):
            bonus += 0.55
        bonus += min(0.35, 0.08 * len(tokens & SECTION_KEYWORDS[section]))
        if any(hint in annotation_key for hint in ("scene", "overview", "audio", "sound", "motion", "event")):
            bonus -= 0.15
        return bonus

    if section == "motion_and_event_cues":
        bonus = 0.0
        if any(
            hint in annotation_key
            for hint in ("dynamic", "action", "scene_sequence", "event_action", "event_dynamic", "event_scene", "motion")
        ):
            bonus += 0.7
        bonus += min(0.35, 0.08 * len(tokens & SECTION_KEYWORDS[section]))
        if not any(token in tokens for token in ("move", "motion", "moving", "enter", "lift", "reach", "pick", "grasp", "peel", "contact", "activity", "interaction", "hands")):
            bonus -= 0.2
        return bonus

    if section == "audio_cues":
        bonus = 0.0
        if any(hint in annotation_key for hint in ("audio", "sound", "speech", "music", "noise")):
            bonus += 0.75
        bonus += min(0.4, 0.1 * len(tokens & SECTION_KEYWORDS[section]))
        if "audio" not in tokens and not any(token in tokens for token in ("sound", "speech", "music", "noise", "silence", "scraping", "chopping", "rhythmic")):
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
                    for hint in ("dynamic", "action", "scene_sequence", "event_action", "event_dynamic", "event_scene")
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

    if {"audio", "rgb"} <= reliable_modalities or {"audio", "ir"} <= reliable_modalities or {"audio", "event"} <= reliable_modalities:
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
    for key in ("scene_overview", "visible_objects_and_layout", "motion_and_event_cues", "audio_cues", "cross_modal_details"):
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
    section_diagnostics: dict[str, Any] | None = {} if collect_diagnostics else None

    selected_by_section = _assign_evidences_to_sections(
        evidences,
        min_score=min_score,
        relax_filters=relax_filters,
        section_diagnostics=section_diagnostics,
    )

    sections = {
        "scene_overview": _join_sentences(selected_by_section["scene_overview"]),
        "visible_objects_and_layout": _join_sentences(selected_by_section["visible_objects_and_layout"]),
        "motion_and_event_cues": _join_sentences(selected_by_section["motion_and_event_cues"]),
        "audio_cues": _join_sentences(selected_by_section["audio_cues"]),
        "cross_modal_details": _build_cross_modal_details(
            evidences,
            min_score=min_score,
            relax_filters=relax_filters,
        ),
    }
    sections["final_unified_caption"] = _build_final_caption(sections)

    modality_reliability: dict[str, float] = {}
    for evidence in evidences:
        score = modality_reliability.get(evidence.modality, 0.0)
        modality_reliability[evidence.modality] = max(score, evidence.modality_reliability)

    rounded_reliability = {key: round(value, 3) for key, value in sorted(modality_reliability.items())}
    section_evidence_qas = _build_section_evidence_qas(
        sections,
        selected_by_section,
        source_entries,
        rounded_reliability,
        qa_limits,
    )

    fused_sample = {
        "source_entries": source_entries,
        "source_files": source_files,
        "modality_reliability": rounded_reliability,
        **sections,
        "section_evidence_qas": section_evidence_qas,
    }

    if collect_diagnostics:
        fused_sample["fusion_diagnostics"] = {
            "relax_filters": relax_filters,
            "min_score": min_score,
            "section_limits": section_limits,
            "sections": section_diagnostics,
        }

    return fused_sample


def run_late_fusion(
    input_dir: Path | str = ".",
    output_file: Path | str = "fused_qa_results.json",
    modality_files: dict[str, str] | None = None,
    relax_filters: bool = False,
    collect_diagnostics: bool = False,
    diagnostics_output_file: Path | str = "fusion_diagnostics.json",
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

    return fused_results


def format_fused_sections(result: dict[str, Any]) -> str:
    lines: list[str] = []
    for section_key in OUTPUT_SECTIONS:
        lines.append(f"{SECTION_LABELS[section_key]}:")
        lines.append(result.get(section_key, ""))
        lines.append("")
    return "\n".join(lines).rstrip()
