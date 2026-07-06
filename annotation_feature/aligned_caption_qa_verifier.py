"""Verify cross-modal disambiguation MCQ QA using caption evidence."""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from annotation_feature.pipeline.client import create_gemini_client
from annotation_feature.pipeline.gemini_retry import call_with_retry_async
from annotation_feature.pipeline.utils import build_image_parts


DEFAULT_INPUT_PATH = Path("outputs/aligned_cross_modal_disambiguation_qa_mcq_5folders_gemini_cleaned.json")
DEFAULT_OUTPUT_PATH = Path("outputs/aligned_cross_modal_disambiguation_qa_mcq_5folders_verified_gemini.json")
DEFAULT_MODEL_NAME = "gemini-3.1-flash-lite"
VERIFICATION_SCHEMA_VERSION = "cross_modal_disambiguation_qa_verification_v2"
CHOICE_LABELS = ("A", "B", "C", "D")
MODALITY_NAMES = (
    "rgb",
    "event",
    "ir",
    "infrared",
    "depth",
    "audio",
    "thermal",
    "grayscale",
    "greyscale",
    "monochrome",
)
TOKEN_STOPWORDS = {
    "about",
    "across",
    "after",
    "alone",
    "along",
    "also",
    "and",
    "are",
    "because",
    "before",
    "being",
    "best",
    "based",
    "between",
    "caption",
    "camera",
    "choice",
    "could",
    "correct",
    "decisive",
    "describe",
    "doing",
    "does",
    "for",
    "from",
    "ground",
    "image",
    "into",
    "near",
    "only",
    "option",
    "person",
    "scene",
    "should",
    "shows",
    "that",
    "the",
    "their",
    "there",
    "this",
    "they",
    "through",
    "too",
    "toward",
    "using",
    "very",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
}
WEAK_ACTION_PATTERN = re.compile(
    r"\b(touch(?:ing)?|near|reach(?:ing)?|approach(?:ing)?|interact(?:ing)?|hold(?:ing)?|move(?:ing)?|moving)\b",
    re.I,
)
TARGET_BOUND_QUESTION_PATTERN = re.compile(
    r"\b(doing with|walking toward|approaching|interacting with|what is the .*object|which activity|relationship to)\b",
    re.I,
)

VERIFICATION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verifications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "qa_id": {"type": "string"},
                    "status": {"type": "string", "enum": ["keep", "reject", "review"]},
                    "is_answer_correct": {"type": "boolean"},
                    "is_single_choice": {"type": "boolean"},
                    "decisive_alone_ambiguous": {"type": "boolean"},
                    "context_disambiguates": {"type": "boolean"},
                    "answer_not_visible_from_decisive_alone": {"type": "boolean"},
                    "decisive_only_leakage": {
                        "type": "string",
                        "enum": ["none", "target_only", "partial", "answer_level"],
                    },
                    "leakage_is_fatal": {"type": "boolean"},
                    "distractors_plausible_from_decisive_alone": {"type": "boolean"},
                    "question_avoids_modality_words": {"type": "boolean"},
                    "failure_reasons": {"type": "array", "items": {"type": "string"}},
                    "rationale": {"type": "string"},
                },
                "required": [
                    "qa_id",
                    "status",
                    "is_answer_correct",
                    "is_single_choice",
                    "decisive_alone_ambiguous",
                    "context_disambiguates",
                    "answer_not_visible_from_decisive_alone",
                    "decisive_only_leakage",
                    "leakage_is_fatal",
                    "distractors_plausible_from_decisive_alone",
                    "question_avoids_modality_words",
                    "failure_reasons",
                    "rationale",
                ],
            },
        }
    },
    "required": ["verifications"],
}

VISUAL_VERIFICATION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "qa_id": {"type": "string"},
        "visual_status": {"type": "string", "enum": ["keep", "reject", "review"]},
        "captions_match_visual_evidence": {"type": "boolean"},
        "decisive_visual_alone_ambiguous": {"type": "boolean"},
        "context_visual_disambiguates": {"type": "boolean"},
        "answer_supported_by_visual_context": {"type": "boolean"},
        "decisive_visual_leaks_answer": {"type": "boolean"},
        "visual_failure_reasons": {"type": "array", "items": {"type": "string"}},
        "visual_rationale": {"type": "string"},
    },
    "required": [
        "qa_id",
        "visual_status",
        "captions_match_visual_evidence",
        "decisive_visual_alone_ambiguous",
        "context_visual_disambiguates",
        "answer_supported_by_visual_context",
        "decisive_visual_leaks_answer",
        "visual_failure_reasons",
        "visual_rationale",
    ],
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


def _parse_json_response(text: str) -> dict[str, Any]:
    if not text:
        raise ValueError("Empty Gemini response")
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.I)
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if not match:
        raise ValueError("No JSON object found in Gemini response")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Gemini response must be a JSON object")
    return parsed


def _question_mentions_modalities(question: str) -> bool:
    lowered = question.lower()
    return any(re.search(rf"\b{re.escape(name)}\b", lowered) for name in MODALITY_NAMES)


def _rule_failures(item: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    question = str(item.get("question") or "").strip()
    choices = item.get("choices")
    answer_choice = str(item.get("answer_choice") or "").strip().upper()
    answer = str(item.get("answer") or "").strip()
    if item.get("question_type") != "single_choice":
        failures.append("question_type is not single_choice")
    if not question:
        failures.append("question is empty")
    if _question_mentions_modalities(question):
        failures.append("question mentions modality names")
    if not isinstance(choices, dict):
        failures.append("choices is not an object")
    else:
        values = [str(choices.get(label) or "").strip() for label in CHOICE_LABELS]
        if any(not value for value in values):
            failures.append("choices must contain non-empty A/B/C/D")
        if len({value.lower() for value in values}) != 4:
            failures.append("choices are not four distinct options")
        if answer_choice not in CHOICE_LABELS:
            failures.append("answer_choice is not A/B/C/D")
        elif str(choices.get(answer_choice) or "").strip() != answer:
            failures.append("answer does not match choices[answer_choice]")
    caption_context = item.get("caption_context")
    if not isinstance(caption_context, dict):
        failures.append("caption_context is missing")
    else:
        observation = caption_context.get("ambiguity_event")
        if not isinstance(observation, dict):
            failures.append("ambiguity_event is missing")
    return failures


def _verification_payload(item: dict[str, Any]) -> dict[str, Any]:
    caption_context = item.get("caption_context") or {}
    observation = caption_context.get("ambiguity_event") or {}
    return {
        "qa_id": item.get("qa_id"),
        "question": item.get("question"),
        "choices": item.get("choices"),
        "answer_choice": item.get("answer_choice"),
        "answer": item.get("answer"),
        "context_modality": item.get("context_modality"),
        "decisive_modality": item.get("decisive_modality"),
        "helper_modality_analysis": caption_context.get("helper_modality_analysis"),
        "victim_modality_analysis": caption_context.get("victim_modality_analysis"),
        "global_scene": caption_context.get("global_scene"),
        "ambiguity_event": {
            "low_confidence_observation": observation.get("low_confidence_observation"),
            "why_victim_cannot_resolve": observation.get("why_victim_cannot_resolve"),
            "candidate_hypotheses": observation.get("candidate_hypotheses"),
            "why_helper_can_resolve": observation.get("why_helper_can_resolve"),
            "helper_discriminative_evidence": observation.get("helper_discriminative_evidence"),
            "fusion_conclusion": observation.get("fusion_conclusion"),
            "missing_attribute_type": observation.get("missing_attribute_type"),
        },
    }


def _build_prompt(items: list[dict[str, Any]]) -> str:
    payloads = [_verification_payload(item) for item in items]
    return "\n".join(
        [
            "You verify multiple-choice QA items for a cross-modal disambiguation benchmark.",
            "Use only the provided caption evidence. Do not use external knowledge.",
            "The benchmark goal: the victim modality alone should leave the answer ambiguous, and the helper modality should disambiguate the correct option.",
            "Treat victim_modality_analysis.detailed_caption as the information available to a model that only sees the victim (decisive) modality.",
            "If the question can be answered by comparing victim_modality_analysis.detailed_caption with the choices, reject the item.",
            "Judge leakage carefully. Mentioning the target object/location alone is not always fatal, BUT target leakage can become fatal depending on the choices.",
            "If victim_modality_analysis.detailed_caption mentions the target object/location and this lets a model eliminate most or all distractors because they involve clearly different targets, treat the leakage as fatal and reject.",
            "Reject only when victim_modality_analysis.detailed_caption already contains answer-level evidence: the final action, state, relation, object identity, or enough wording to choose the correct option.",
            "Use target_only only when victim_modality_analysis.detailed_caption names the object/location but still does not reveal the required action, state, phase, or relation, and the distractors remain plausible under the same target/location.",
            "Use partial when victim_modality_analysis.detailed_caption hints at the answer but still leaves a meaningful ambiguity that helper context resolves.",
            "Examples: 'touching the lid' does not by itself prove 'removing or lifting the lid'; 'hand near the control panel' does not by itself prove 'operating the controls'. These are not fatal unless the question only asks for the target.",
            "If the question asks object identity and victim_modality_analysis.detailed_caption already names that object, use answer_level leakage and reject.",
            "Be stricter than the generator, but do not reject just because the victim-only caption mentions the target involved in the answer.",
            "Mark an item keep only if all of these are true:",
            "1. The listed answer choice is supported by the fusion_conclusion or helper_discriminative_evidence.",
            "2. The question is single-choice and has exactly one correct option.",
            "3. The victim-only caption/observation leaves at least two plausible options.",
            "4. The helper-only caption or helper_discriminative_evidence resolves the ambiguity.",
            "5. Each distractor is plausible under victim-only ambiguity, including target/location information, but not supported after helper context is added.",
            "6. The question does not explicitly mention modality names.",
            "Use review for borderline cases. Use reject for clear failures.",
            "",
            "Return ONLY valid JSON with this exact structure:",
            "{",
            '  "verifications": [',
            "    {",
            '      "qa_id": "must exactly match one input qa_id",',
            '      "status": "keep|reject|review",',
            '      "is_answer_correct": true,',
            '      "is_single_choice": true,',
            '      "decisive_alone_ambiguous": true,',
            '      "context_disambiguates": true,',
            '      "answer_not_visible_from_decisive_alone": true,',
            '      "decisive_only_leakage": "none|target_only|partial|answer_level",',
            '      "leakage_is_fatal": true,',
            '      "distractors_plausible_from_decisive_alone": true,',
            '      "question_avoids_modality_words": true,',
            '      "failure_reasons": ["optional reason(s) if status is reject"],',
            '      "rationale": "Reasoning for the evaluation"',
            "    }",
            "  ]",
            "}",
            "",
            "Input QA items:",
            json.dumps(payloads, indent=2, ensure_ascii=False),
        ]
    )


def _template_verification(item: dict[str, Any]) -> dict[str, Any]:
    failures = _rule_failures(item)
    return {
        "qa_id": item.get("qa_id"),
        "status": "keep" if not failures else "reject",
        "is_answer_correct": not failures,
        "is_single_choice": item.get("question_type") == "single_choice",
        "decisive_alone_ambiguous": not failures,
        "context_disambiguates": not failures,
        "answer_not_visible_from_decisive_alone": not failures,
        "decisive_only_leakage": "none",
        "leakage_is_fatal": False,
        "distractors_plausible_from_decisive_alone": not failures,
        "question_avoids_modality_words": not _question_mentions_modalities(str(item.get("question") or "")),
        "failure_reasons": failures,
        "rationale": "Rule-only template verification.",
    }


def _normalize_verification(raw: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    qa_id = str(raw.get("qa_id") or item.get("qa_id") or "").strip()
    if qa_id != item.get("qa_id"):
        raise ValueError(f"Expected qa_id {item.get('qa_id')!r}, got {qa_id!r}")
    status = str(raw.get("status") or "review").strip().lower()
    if status not in {"keep", "reject", "review"}:
        status = "review"
    failure_reasons = raw.get("failure_reasons")
    if not isinstance(failure_reasons, list):
        failure_reasons = []
    rule_failures = _rule_failures(item)
    if rule_failures:
        status = "reject"
        failure_reasons = list(dict.fromkeys([*failure_reasons, *rule_failures]))
    decisive_only_leakage = str(raw.get("decisive_only_leakage") or "none").strip().lower()
    if decisive_only_leakage not in {"none", "target_only", "partial", "answer_level"}:
        decisive_only_leakage = "partial" if decisive_only_leakage else "none"
    legacy_contains_answer = bool(raw.get("decisive_only_caption_contains_answer"))
    leakage_is_fatal = bool(raw.get("leakage_is_fatal")) or decisive_only_leakage == "answer_level" or legacy_contains_answer
    if legacy_contains_answer and decisive_only_leakage == "none":
        decisive_only_leakage = "answer_level"
    if leakage_is_fatal:
        status = "reject"
        failure_reasons = list(
            dict.fromkeys([*failure_reasons, "victim_modality_analysis contains answer-level evidence"])
        )
    critical_checks = {
        "is_answer_correct": bool(raw.get("is_answer_correct")),
        "is_single_choice": bool(raw.get("is_single_choice")),
        "decisive_alone_ambiguous": bool(raw.get("decisive_alone_ambiguous")),
        "context_disambiguates": bool(raw.get("context_disambiguates")),
        "answer_not_visible_from_decisive_alone": bool(raw.get("answer_not_visible_from_decisive_alone")),
        "distractors_plausible_from_decisive_alone": bool(raw.get("distractors_plausible_from_decisive_alone")),
        "question_avoids_modality_words": bool(raw.get("question_avoids_modality_words"))
        and not _question_mentions_modalities(str(item.get("question") or "")),
    }
    if status == "keep":
        failed_checks = [name for name, passed in critical_checks.items() if not passed]
        if failed_checks:
            status = "reject"
            failure_reasons = list(dict.fromkeys([*failure_reasons, *failed_checks]))
    return {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "qa_id": qa_id,
        "status": status,
        "is_answer_correct": critical_checks["is_answer_correct"],
        "is_single_choice": critical_checks["is_single_choice"],
        "decisive_alone_ambiguous": critical_checks["decisive_alone_ambiguous"],
        "context_disambiguates": critical_checks["context_disambiguates"],
        "answer_not_visible_from_decisive_alone": critical_checks["answer_not_visible_from_decisive_alone"],
        "decisive_only_leakage": decisive_only_leakage,
        "leakage_is_fatal": leakage_is_fatal,
        "distractors_plausible_from_decisive_alone": critical_checks["distractors_plausible_from_decisive_alone"],
        "question_avoids_modality_words": critical_checks["question_avoids_modality_words"],
        "failure_reasons": [str(reason).strip() for reason in failure_reasons if str(reason).strip()],
        "rationale": str(raw.get("rationale") or "").strip(),
    }


async def _call_gemini_verify_batch(
    client,
    items: list[dict[str, Any]],
    model_name: str,
    max_retries: int,
) -> dict[str, dict[str, Any]]:
    prompt = _build_prompt(items)
    response = await call_with_retry_async(
        lambda: client.models.generate_content(
            model=model_name,
            contents=[prompt],
            config={
                "response_mime_type": "application/json",
                "response_schema": VERIFICATION_RESPONSE_SCHEMA,
            },
        ),
        max_attempts=max_retries,
        label="QA verifier Gemini call",
    )
    parsed = _parse_json_response(response.text)
    raw_verifications = parsed.get("verifications")
    if not isinstance(raw_verifications, list):
        raise ValueError("Gemini response verifications must be a list")
    expected = {str(item.get("qa_id")): item for item in items}
    normalized: dict[str, dict[str, Any]] = {}
    for raw in raw_verifications:
        if not isinstance(raw, dict):
            continue
        qa_id = str(raw.get("qa_id") or "").strip()
        item = expected.get(qa_id)
        if item is None or qa_id in normalized:
            continue
        normalized[qa_id] = _normalize_verification(raw, item)
    missing = [qa_id for qa_id in expected if qa_id not in normalized]
    if missing:
        raise ValueError(f"Gemini response missing verification for qa_id(s): {', '.join(missing)}")
    return normalized


def _chunk(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _normalize_token(token: str) -> str:
    token = token.lower()
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]
    return token


def _content_tokens(text: Any) -> set[str]:
    tokens: set[str] = set()
    for token in re.findall(r"[A-Za-z][A-Za-z'-]*", str(text or "").lower()):
        normalized = _normalize_token(token.strip("'"))
        if len(normalized) < 3 or normalized in TOKEN_STOPWORDS:
            continue
        tokens.add(normalized)
    return tokens


def _choice_texts(item: dict[str, Any]) -> dict[str, str]:
    choices = item.get("choices")
    if not isinstance(choices, dict):
        return {}
    return {label: str(choices.get(label) or "") for label in CHOICE_LABELS}


def _compute_visual_risk(item: dict[str, Any]) -> dict[str, Any]:
    caption_context = item.get("caption_context") or {}
    victim_analysis = caption_context.get("victim_modality_analysis") or {}
    decisive_caption = str(victim_analysis.get("detailed_caption") or "")
    decisive_tokens = _content_tokens(decisive_caption)
    answer_tokens = _content_tokens(item.get("answer"))
    question = str(item.get("question") or "")
    verification = item.get("verification") or {}
    rationale = str(verification.get("rationale") or "")
    choices = _choice_texts(item)
    answer_choice = str(item.get("answer_choice") or "").strip().upper()

    score = 0
    reasons: list[str] = []

    answer_overlap = sorted(answer_tokens & decisive_tokens)
    if len(answer_overlap) >= 2:
        score += 3
        reasons.append(f"answer overlaps decisive-only caption: {', '.join(answer_overlap[:8])}")

    choice_overlaps = {
        label: len(_content_tokens(text) & decisive_tokens)
        for label, text in choices.items()
    }
    correct_overlap = choice_overlaps.get(answer_choice, 0)
    wrong_overlaps = [count for label, count in choice_overlaps.items() if label != answer_choice]
    if correct_overlap >= 2 and wrong_overlaps and correct_overlap >= max(wrong_overlaps) + 2:
        score += 3
        reasons.append(
            f"correct choice has much higher decisive-caption overlap ({correct_overlap} vs {max(wrong_overlaps)})"
        )

    answer_choice_tokens = _content_tokens(choices.get(answer_choice, ""))
    wrong_choice_tokens = [
        _content_tokens(text)
        for label, text in choices.items()
        if label != answer_choice
    ]
    if answer_choice_tokens and wrong_choice_tokens:
        max_wrong_shared_with_answer = max((len(answer_choice_tokens & tokens) for tokens in wrong_choice_tokens), default=0)
        if correct_overlap >= 2 and max_wrong_shared_with_answer <= 1:
            score += 2
            reasons.append("correct and distractor choices appear to involve separated targets")

    if TARGET_BOUND_QUESTION_PATTERN.search(question):
        score += 2
        reasons.append("question uses a target-bound structure")

    if WEAK_ACTION_PATTERN.search(decisive_caption) and decisive_tokens:
        score += 2
        reasons.append("decisive-only caption contains weak action with a specific target")

    rationale_lower = rationale.lower()
    if str(verification.get("decisive_only_leakage") or "").lower() == "none" and any(
        marker in rationale_lower
        for marker in ("ambiguous", "does not confirm", "only shows", "touching", "near")
    ):
        score += 1
        reasons.append("text verifier rationale describes a borderline ambiguity while leakage is none")

    return {
        "score": score,
        "reasons": reasons,
        "answer_decisive_overlap": answer_overlap,
        "choice_decisive_overlap": choice_overlaps,
    }


def _should_visual_verify(item: dict[str, Any]) -> bool:
    verification = item.get("verification") or {}
    status = str(verification.get("status") or "").strip().lower()
    leakage = str(verification.get("decisive_only_leakage") or "").strip().lower()
    return status == "review" or leakage in {"target_only", "partial"}


def _limited_existing_paths(paths: Any, max_count: int) -> list[Path]:
    if not isinstance(paths, list):
        return []
    selected: list[Path] = []
    for value in paths:
        path = Path(str(value))
        if path.exists():
            selected.append(path)
        if len(selected) >= max_count:
            break
    return selected


def _normalize_image_for_gemini(path: Path) -> str:
    with Image.open(path) as image:
        if image.mode in {"I", "I;16", "I;16B", "I;16L", "F"}:
            extrema = image.getextrema()
            if isinstance(extrema[0], tuple):
                image = image.convert("RGB")
            else:
                min_value, max_value = extrema
                if max_value <= min_value:
                    image = Image.new("L", image.size, 0)
                else:
                    scale = 255.0 / (max_value - min_value)
                    pixels = [
                        max(0, min(255, int((float(value) - min_value) * scale)))
                        for value in image.getdata()
                    ]
                    normalized = Image.new("L", image.size)
                    normalized.putdata(pixels)
                    image = normalized
                image = image.convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
    return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")


def _encode_visual_frames(frame_paths: list[Path]) -> list[str]:
    encoded: list[str] = []
    for frame_path in frame_paths:
        try:
            encoded.append(_normalize_image_for_gemini(frame_path))
        except Exception as exc:
            print(f"WARNING: Failed to normalize visual verifier frame {frame_path}: {exc}")
    return encoded


def _visual_prompt(item: dict[str, Any]) -> str:
    return "\n".join(
        [
            "You are a visual verifier for a cross-modal disambiguation QA benchmark.",
            "Use the provided images together with the QA text. Do not use external knowledge.",
            "Image groups are provided in this order when available: composite frames, decisive-only frames, context-only frames.",
            "Composite frames place context on the left and decisive on the right.",
            "Verify whether the text-only captions are faithful enough to the images, whether the decisive-only visual evidence is genuinely ambiguous, and whether adding context visually disambiguates the correct answer.",
            "Reject if the decisive-only images alone visibly reveal the correct answer, if the captions hallucinate important visual evidence, if the answer is not visually supported, or if distractors are not plausible under decisive-only visual evidence.",
            "Use review for borderline cases where the images are too unclear to make a confident decision.",
            "",
            "Return ONLY valid JSON matching the requested schema.",
            "",
            "QA item:",
            json.dumps(_verification_payload(item), indent=2, ensure_ascii=False),
            "",
            "Current text verification:",
            json.dumps(item.get("verification") or {}, indent=2, ensure_ascii=False),
        ]
    )


def _normalize_visual_verification(raw: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    qa_id = str(raw.get("qa_id") or item.get("qa_id") or "").strip()
    if qa_id != item.get("qa_id"):
        raise ValueError(f"Expected visual qa_id {item.get('qa_id')!r}, got {qa_id!r}")
    status = str(raw.get("visual_status") or "review").strip().lower()
    if status not in {"keep", "reject", "review"}:
        status = "review"
    failure_reasons = raw.get("visual_failure_reasons")
    if not isinstance(failure_reasons, list):
        failure_reasons = []
    checks = {
        "captions_match_visual_evidence": bool(raw.get("captions_match_visual_evidence")),
        "decisive_visual_alone_ambiguous": bool(raw.get("decisive_visual_alone_ambiguous")),
        "context_visual_disambiguates": bool(raw.get("context_visual_disambiguates")),
        "answer_supported_by_visual_context": bool(raw.get("answer_supported_by_visual_context")),
        "decisive_visual_leaks_answer": bool(raw.get("decisive_visual_leaks_answer")),
    }
    if checks["decisive_visual_leaks_answer"]:
        status = "reject"
        failure_reasons = list(dict.fromkeys([*failure_reasons, "decisive-only visual evidence reveals the answer"]))
    if status == "keep":
        failed_checks = [
            name
            for name in (
                "captions_match_visual_evidence",
                "decisive_visual_alone_ambiguous",
                "context_visual_disambiguates",
                "answer_supported_by_visual_context",
            )
            if not checks[name]
        ]
        if failed_checks:
            status = "reject"
            failure_reasons = list(dict.fromkeys([*failure_reasons, *failed_checks]))
    return {
        "qa_id": qa_id,
        "visual_status": status,
        **checks,
        "visual_failure_reasons": [str(reason).strip() for reason in failure_reasons if str(reason).strip()],
        "visual_rationale": str(raw.get("visual_rationale") or "").strip(),
    }


def _apply_visual_verification(item: dict[str, Any], visual: dict[str, Any]) -> None:
    item["visual_verification"] = visual
    verification = item.get("verification")
    if not isinstance(verification, dict):
        return
    failure_reasons = verification.get("failure_reasons")
    if not isinstance(failure_reasons, list):
        failure_reasons = []
    visual_status = visual.get("visual_status")
    if visual_status == "reject":
        verification["status"] = "reject"
        verification["failure_reasons"] = list(
            dict.fromkeys([*failure_reasons, "visual verifier rejected the item", *visual.get("visual_failure_reasons", [])])
        )
    elif visual_status == "review" and verification.get("status") == "keep":
        verification["status"] = "review"
        verification["failure_reasons"] = list(
            dict.fromkeys([*failure_reasons, "visual verifier marked the item as review"])
        )


async def _call_gemini_visual_verify(
    client,
    item: dict[str, Any],
    model_name: str,
    max_retries: int,
    max_visual_frames_per_kind: int,
) -> dict[str, Any]:
    input_frames = item.get("input_frames") or {}
    if not isinstance(input_frames, dict):
        raise ValueError(f"Input frames missing for visual verifier: {item.get('qa_id')}")
    frame_groups = [
        ("Composite frames: left=context, right=decisive.", input_frames.get("composite_frames")),
        ("Decisive-only frames.", input_frames.get("decisive_frames")),
        ("Context-only frames.", input_frames.get("context_frames")),
    ]
    contents: list[Any] = []
    found_any_frame = False
    for label, paths in frame_groups:
        selected = _limited_existing_paths(paths, max_visual_frames_per_kind)
        if not selected:
            continue
        encoded = _encode_visual_frames(selected)
        if not encoded:
            continue
        found_any_frame = True
        contents.append(label)
        contents.extend(build_image_parts(encoded))
    if not found_any_frame:
        raise ValueError(f"No readable frames found for visual verifier: {item.get('qa_id')}")
    contents.append(_visual_prompt(item))
    response = await call_with_retry_async(
        lambda: client.models.generate_content(
            model=model_name,
            contents=contents,
            config={
                "response_mime_type": "application/json",
                "response_schema": VISUAL_VERIFICATION_RESPONSE_SCHEMA,
            },
        ),
        max_attempts=max_retries,
        label="Visual QA verifier Gemini call",
    )
    return _normalize_visual_verification(_parse_json_response(response.text), item)


def _load_resume(output_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not output_path.exists():
        return [], []
    data = _load_json(output_path)
    qa_items = data.get("qa_items") if isinstance(data.get("qa_items"), list) else []
    skipped = data.get("skipped") if isinstance(data.get("skipped"), list) else []
    return list(qa_items), list(skipped)


def _build_output_payload(
    input_path: Path,
    output_path: Path,
    model_name: str,
    verification_mode: str,
    visual_verification_mode: str,
    batch_size: int,
    visual_keep_top_n: int,
    visual_keep_risk_threshold: int,
    planned_items: int,
    gemini_calls: int,
    visual_gemini_calls: int,
    qa_items: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
) -> dict[str, Any]:
    statuses = Counter((item.get("verification") or {}).get("status", "missing") for item in qa_items)
    visual_statuses = Counter(
        (item.get("visual_verification") or {}).get("visual_status", "missing")
        for item in qa_items
        if item.get("visual_verification")
    )
    return {
        "metadata": {
            "task": "cross_modal_disambiguation_qa_verification",
            "schema_version": VERIFICATION_SCHEMA_VERSION,
            "input": input_path.as_posix(),
            "output": output_path.as_posix(),
            "verification_mode": verification_mode,
            "visual_verification_mode": visual_verification_mode,
            "model_name": model_name,
            "max_qa_items_per_gemini_call": batch_size,
            "visual_keep_top_n": visual_keep_top_n,
            "visual_keep_risk_threshold": visual_keep_risk_threshold,
            "planned_qa_items": planned_items,
            "completed_qa_items": len(qa_items),
            "kept_items": statuses.get("keep", 0),
            "review_items": statuses.get("review", 0),
            "rejected_items": statuses.get("reject", 0),
            "missing_verification": statuses.get("missing", 0),
            "skipped_items": len(skipped),
            "gemini_calls": gemini_calls,
            "visual_gemini_calls": visual_gemini_calls,
            "status_distribution": dict(sorted(statuses.items())),
            "visual_status_distribution": dict(sorted(visual_statuses.items())),
            "by_direction_kept": dict(
                sorted(
                    Counter(
                        f"{item.get('context_modality')}->{item.get('decisive_modality')}"
                        for item in qa_items
                        if (item.get("verification") or {}).get("status") == "keep"
                    ).items()
                )
            ),
        },
        "qa_items": sorted(qa_items, key=lambda item: str(item.get("qa_id") or "")),
        "skipped": skipped,
    }


async def run_verifier_async(
    input_path: Path,
    output_path: Path,
    model_name: str,
    verification_mode: str,
    visual_verification_mode: str,
    max_qa_items_per_gemini_call: int,
    max_visual_frames_per_kind: int,
    visual_keep_top_n: int,
    visual_keep_risk_threshold: int,
    limit: int | None,
    max_retries: int,
    delay_between_calls: int,
    checkpoint_every: int,
    resume: bool,
) -> Path:
    data = _load_json(input_path)
    source_items = data.get("qa_items")
    if not isinstance(source_items, list):
        raise ValueError("Input must contain qa_items list")
    if limit is not None:
        source_items = source_items[:limit]
    completed_items, skipped = _load_resume(output_path) if resume else ([], [])
    completed_ids = {str(item.get("qa_id")) for item in completed_items if item.get("qa_id")}
    pending_items = [item for item in source_items if str(item.get("qa_id")) not in completed_ids]
    batch_size = max(1, max_qa_items_per_gemini_call if verification_mode == "gemini" else 1)
    batches = _chunk(pending_items, batch_size)
    client = create_gemini_client() if "gemini" in {verification_mode, visual_verification_mode} else None
    gemini_calls = 0
    visual_gemini_calls = 0
    checkpoint_counter = 0

    print(
        f"Verifying QA: {len(source_items)} planned item(s), {len(pending_items)} pending, "
        f"{len(batches)} batch(es), mode={verification_mode}, visual_mode={visual_verification_mode}, "
        f"model={model_name}, batch_size={batch_size}."
    )

    def save_checkpoint() -> None:
        _save_json(
            _build_output_payload(
                input_path=input_path,
                output_path=output_path,
                model_name=model_name,
                verification_mode=verification_mode,
                visual_verification_mode=visual_verification_mode,
                batch_size=batch_size,
                visual_keep_top_n=visual_keep_top_n,
                visual_keep_risk_threshold=visual_keep_risk_threshold,
                planned_items=len(source_items),
                gemini_calls=gemini_calls,
                visual_gemini_calls=visual_gemini_calls,
                qa_items=completed_items,
                skipped=skipped,
            ),
            output_path,
        )

    for batch_index, batch in enumerate(batches, start=1):
        label = ", ".join(str(item.get("qa_id")) for item in batch)
        print(f"  Verify batch [{batch_index}/{len(batches)}] {len(batch)} item(s): {label}")
        try:
            if verification_mode == "gemini":
                assert client is not None
                verifications = await _call_gemini_verify_batch(
                    client,
                    batch,
                    model_name=model_name,
                    max_retries=max_retries,
                )
                gemini_calls += 1
            else:
                verifications = {
                    str(item.get("qa_id")): _normalize_verification(_template_verification(item), item)
                    for item in batch
                }
            for item in batch:
                copied = dict(item)
                copied["verification"] = verifications[str(item.get("qa_id"))]
                copied["visual_risk"] = _compute_visual_risk(copied)
                if visual_verification_mode == "gemini" and _should_visual_verify(copied):
                    try:
                        assert client is not None
                        print(f"    Visual verify {copied.get('qa_id')}")
                        visual = await _call_gemini_visual_verify(
                            client,
                            copied,
                            model_name=model_name,
                            max_retries=max_retries,
                            max_visual_frames_per_kind=max_visual_frames_per_kind,
                        )
                        visual_gemini_calls += 1
                        _apply_visual_verification(copied, visual)
                        if delay_between_calls > 0:
                            await asyncio.sleep(delay_between_calls)
                    except Exception as exc:
                        copied["visual_verification_error"] = str(exc)
                        skipped.append({"qa_id": copied.get("qa_id"), "stage": "visual_verification", "reason": str(exc)})
                completed_items.append(copied)
        except Exception as exc:
            for item in batch:
                skipped.append({"qa_id": item.get("qa_id"), "reason": str(exc)})
            print(f"WARNING: Verification failed for batch containing {label}: {exc}")

        checkpoint_counter += 1
        if checkpoint_every > 0 and checkpoint_counter >= checkpoint_every:
            checkpoint_counter = 0
            save_checkpoint()
        if verification_mode == "gemini" and delay_between_calls > 0 and batch_index < len(batches):
            await asyncio.sleep(delay_between_calls)

    if visual_verification_mode == "gemini" and visual_keep_top_n > 0:
        candidates = [
            item
            for item in completed_items
            if not item.get("visual_verification")
            and not item.get("visual_verification_error")
            and (item.get("verification") or {}).get("status") == "keep"
            and str((item.get("verification") or {}).get("decisive_only_leakage") or "").lower() == "none"
            and int((item.get("visual_risk") or {}).get("score") or 0) >= visual_keep_risk_threshold
        ]
        candidates.sort(
            key=lambda item: (
                -int((item.get("visual_risk") or {}).get("score") or 0),
                str(item.get("qa_id") or ""),
            )
        )
        for item in candidates[:visual_keep_top_n]:
            try:
                assert client is not None
                print(
                    f"    Visual verify high-risk keep {item.get('qa_id')} "
                    f"(risk={int((item.get('visual_risk') or {}).get('score') or 0)})"
                )
                visual = await _call_gemini_visual_verify(
                    client,
                    item,
                    model_name=model_name,
                    max_retries=max_retries,
                    max_visual_frames_per_kind=max_visual_frames_per_kind,
                )
                visual_gemini_calls += 1
                _apply_visual_verification(item, visual)
                if delay_between_calls > 0:
                    await asyncio.sleep(delay_between_calls)
            except Exception as exc:
                item["visual_verification_error"] = str(exc)
                skipped.append({"qa_id": item.get("qa_id"), "stage": "visual_verification", "reason": str(exc)})

    save_checkpoint()
    print(f"Wrote verified QA output to {output_path}")
    return output_path


def run_verifier(
    input_path: Path | str = DEFAULT_INPUT_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
    verification_mode: str = "template",
    visual_verification_mode: str = "none",
    max_qa_items_per_gemini_call: int = 8,
    max_visual_frames_per_kind: int = 2,
    visual_keep_top_n: int = 0,
    visual_keep_risk_threshold: int = 3,
    limit: int | None = None,
    max_retries: int = 3,
    delay_between_calls: int = 5,
    checkpoint_every: int = 1,
    resume: bool = True,
) -> Path:
    return asyncio.run(
        run_verifier_async(
            input_path=Path(input_path),
            output_path=Path(output_path),
            model_name=model_name,
            verification_mode=verification_mode,
            visual_verification_mode=visual_verification_mode,
            max_qa_items_per_gemini_call=max(1, max_qa_items_per_gemini_call),
            max_visual_frames_per_kind=max(1, max_visual_frames_per_kind),
            visual_keep_top_n=max(0, visual_keep_top_n),
            visual_keep_risk_threshold=max(0, visual_keep_risk_threshold),
            limit=limit,
            max_retries=max(1, max_retries),
            delay_between_calls=max(0, delay_between_calls),
            checkpoint_every=max(0, checkpoint_every),
            resume=resume,
        )
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--verification-mode", choices=("template", "gemini"), default="template")
    parser.add_argument("--visual-verification-mode", choices=("none", "gemini"), default="none")
    parser.add_argument("--max-qa-items-per-gemini-call", type=int, default=8)
    parser.add_argument("--max-visual-frames-per-kind", type=int, default=2)
    parser.add_argument("--visual-keep-top-n", type=int, default=0)
    parser.add_argument("--visual-keep-risk-threshold", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--delay-between-calls", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    run_verifier(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model_name,
        verification_mode=args.verification_mode,
        visual_verification_mode=args.visual_verification_mode,
        max_qa_items_per_gemini_call=args.max_qa_items_per_gemini_call,
        max_visual_frames_per_kind=args.max_visual_frames_per_kind,
        visual_keep_top_n=args.visual_keep_top_n,
        visual_keep_risk_threshold=args.visual_keep_risk_threshold,
        limit=args.limit,
        max_retries=args.max_retries,
        delay_between_calls=args.delay_between_calls,
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
