"""Generate QA candidates from cross-modal disambiguation captions."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from annotation_feature.pipeline.client import create_gemini_client
from annotation_feature.pipeline.gemini_retry import call_with_retry_async


DEFAULT_INPUT_PATH = Path("outputs/aligned_cross_modal_disambiguation_captions_all_directions_5folders_batch2_gemini.json")
DEFAULT_OUTPUT_PATH = Path("outputs/aligned_cross_modal_disambiguation_qa_candidates_gemini.json")
DEFAULT_MODEL_NAME = "gemini-3.1-flash-lite"
QA_SCHEMA_VERSION = "cross_modal_disambiguation_qa_mcq_v1"
QA_STYLE = "cross_modal_disambiguation"
LOW_ENTROPY_ANSWERS = {
    "yes",
    "no",
    "true",
    "false",
    "none",
    "unknown",
    "one",
    "two",
    "three",
    "1",
    "2",
    "3",
}
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
BLANK_SENSOR_PATTERNS = (
    "completely blank",
    "featureless white",
    "featureless black",
    "no visual data",
    "no identifying features",
    "uniform white",
    "uniform black",
)
CHOICE_LABELS = ("A", "B", "C", "D")

QA_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "qa_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "question_type": {"type": "string", "enum": ["single_choice"]},
                    "question": {"type": "string"},
                    "choices": {
                        "type": "object",
                        "properties": {
                            "A": {"type": "string"},
                            "B": {"type": "string"},
                            "C": {"type": "string"},
                            "D": {"type": "string"},
                        },
                        "required": ["A", "B", "C", "D"],
                    },
                    "answer_choice": {"type": "string", "enum": ["A", "B", "C", "D"]},
                    "answer": {"type": "string"},
                    "possible_answers_from_victim_only": {"type": "array", "items": {"type": "string"}},
                    "why_victim_alone_is_ambiguous": {"type": "string"},
                    "helper_disambiguating_evidence": {"type": "string"},
                    "ground_truth_source": {"type": "string"},
                    "answer_type": {
                        "type": "string",
                        "enum": [
                            "object_action_relation",
                            "action_phase",
                            "spatial_relation",
                            "motion_interaction",
                            "text_semantic_identity",
                            "visibility_or_state",
                            "other",
                        ],
                    },
                    "question_difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                    "ground_truth_confidence": {"type": "number"},
                },
                "required": [
                    "task_id",
                    "question_type",
                    "question",
                    "choices",
                    "answer_choice",
                    "answer",
                    "possible_answers_from_victim_only",
                    "why_victim_alone_is_ambiguous",
                    "helper_disambiguating_evidence",
                    "ground_truth_source",
                    "answer_type",
                    "question_difficulty",
                    "ground_truth_confidence",
                ],
            },
        }
    },
    "required": ["qa_items"],
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


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None


def _looks_like_blank_sensor_caption(caption: Any) -> bool:
    lowered = str(caption or "").lower()
    return any(pattern in lowered for pattern in BLANK_SENSOR_PATTERNS)


def _question_mentions_modalities(question: str) -> bool:
    lowered = question.lower()
    return any(re.search(rf"\b{re.escape(name)}\b", lowered) for name in MODALITY_NAMES)


def _looks_low_entropy_answer(answer: str) -> bool:
    normalized = re.sub(r"[^a-z0-9 ]+", "", answer.lower()).strip()
    if normalized in LOW_ENTROPY_ANSWERS:
        return True
    if re.fullmatch(r"\d+(?:\.\d+)?(?:\s*(?:degrees?|c|°c|%))?", normalized):
        return True
    words = normalized.split()
    return len(words) == 1 and normalized in {"red", "blue", "green", "black", "white", "yellow", "orange"}


def _normalize_choice_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _coerce_choices(raw_choices: Any) -> dict[str, str]:
    if isinstance(raw_choices, dict):
        normalized_choices = {str(key).strip().upper(): value for key, value in raw_choices.items()}
        return {
            label: _normalize_choice_text(normalized_choices.get(label))
            for label in CHOICE_LABELS
        }
    if isinstance(raw_choices, list):
        return {
            label: _normalize_choice_text(raw_choices[index] if index < len(raw_choices) else "")
            for index, label in enumerate(CHOICE_LABELS)
        }
    return {label: "" for label in CHOICE_LABELS}


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = _normalize_choice_text(value)
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        deduped.append(normalized)
        seen.add(key)
    return deduped


def _generic_template_distractors(answer_type: str) -> list[str]:
    pools = {
        "object_action_relation": [
            "The person is touching the same object without changing it.",
            "The person is reaching toward a nearby object instead.",
            "The interaction involves the object but a different action.",
        ],
        "action_phase": [
            "The person is preparing to start the action.",
            "The person has already finished the action.",
            "The person is pausing between two related steps.",
        ],
        "spatial_relation": [
            "The person is beside the object rather than directly interacting with it.",
            "The cue marks a nearby boundary instead of the target location.",
            "The object is in the background rather than at the interaction point.",
        ],
        "motion_interaction": [
            "The motion comes from the person moving past the object.",
            "The object is stationary while the hand moves nearby.",
            "The motion corresponds to a different nearby interaction.",
        ],
        "text_semantic_identity": [
            "The visible marking belongs to a different sign or label.",
            "The text-like cue is a reflection or surface pattern.",
            "The cue indicates a generic label rather than the specific identity.",
        ],
        "visibility_or_state": [
            "The object is present but not in the stated state.",
            "The cue is caused by lighting or reflection rather than the object state.",
            "The scene does not resolve whether the object is active or inactive.",
        ],
    }
    return pools.get(answer_type, []) + [
        "The scene indicates a related but different object.",
        "The person is interacting with the scene in a different way.",
        "The cue is insufficient to identify the exact interaction.",
    ]


def _build_template_choices(
    answer: str,
    possible_answers: list[str],
    observation: dict[str, Any],
    answer_type: str = "other",
) -> tuple[dict[str, str], str]:
    distractors = [value for value in _dedupe_preserve_order(possible_answers) if value.lower() != answer.lower()]
    observation_text = _normalize_choice_text(
        observation.get("low_confidence_observation") or observation.get("decisive_observation")
    )
    if observation_text:
        distractors.extend(
            [
                f"The cue only shows {observation_text.lower()}, not the resolved interaction.",
                f"The same cue could refer to another interpretation of {observation_text.lower()}.",
            ]
        )
    distractors.extend(_generic_template_distractors(answer_type))
    distractors = [value for value in _dedupe_preserve_order(distractors) if value.lower() != answer.lower()]
    while len(distractors) < 3:
        distractors.append(f"The scene suggests an alternative interpretation {len(distractors) + 1}.")
    choices = {
        "A": distractors[0],
        "B": answer,
        "C": distractors[1],
        "D": distractors[2],
    }
    return choices, "B"


def _hypothesis_texts(observation: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for item in observation.get("candidate_hypotheses") or []:
        if isinstance(item, dict):
            text = str(item.get("hypothesis") or "").strip()
        else:
            text = str(item or "").strip()
        if text:
            values.append(text)
    return values


def _shuffle_choices(
    choices: dict[str, str],
    answer_choice: str,
    seed: str,
) -> tuple[dict[str, str], str]:
    if answer_choice not in CHOICE_LABELS:
        return choices, answer_choice
    values = [(label, choices[label]) for label in CHOICE_LABELS]
    ranked = sorted(
        values,
        key=lambda pair: hashlib.sha256(f"{seed}:{pair[0]}".encode("utf-8")).hexdigest(),
    )
    shuffled = {label: ranked[index][1] for index, label in enumerate(CHOICE_LABELS)}
    answer_text = choices[answer_choice]
    shuffled_answer = next(label for label, value in shuffled.items() if value == answer_text)
    return shuffled, shuffled_answer


def _validation_errors(item: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    question = str(item.get("question") or "").strip()
    answer = str(item.get("answer") or "").strip()
    choices = item.get("choices")
    answer_choice = str(item.get("answer_choice") or "").strip().upper()
    if not question:
        errors.append("question is empty")
    if not answer:
        errors.append("answer is empty")
    if item.get("question_type") != "single_choice":
        errors.append("question_type must be single_choice")
    if _question_mentions_modalities(question):
        errors.append("question explicitly mentions modality names")
    if _looks_low_entropy_answer(answer):
        errors.append("answer looks low-entropy")
    if not isinstance(choices, dict):
        errors.append("choices must be an object")
    else:
        missing = [label for label in CHOICE_LABELS if not str(choices.get(label) or "").strip()]
        if missing:
            errors.append(f"choices missing option(s): {', '.join(missing)}")
        extra = [label for label in choices if label not in CHOICE_LABELS]
        if extra:
            errors.append(f"choices contains unexpected option(s): {', '.join(extra)}")
        choice_values = [_normalize_choice_text(choices.get(label)) for label in CHOICE_LABELS]
        if len({value.lower() for value in choice_values if value}) != 4:
            errors.append("choices must contain four distinct non-empty options")
        if answer_choice not in CHOICE_LABELS:
            errors.append("answer_choice must be one of A, B, C, D")
        elif _normalize_choice_text(choices.get(answer_choice)).lower() != answer.lower():
            errors.append("answer must exactly match choices[answer_choice]")
    possible_answers = item.get("possible_answers_from_victim_only")
    if not isinstance(possible_answers, list) or len(possible_answers) < 2:
        errors.append("possible_answers_from_victim_only must include at least two options")
    if not str(item.get("helper_disambiguating_evidence") or "").strip():
        errors.append("helper_disambiguating_evidence is empty")
    if not str(item.get("why_victim_alone_is_ambiguous") or "").strip():
        errors.append("why_victim_alone_is_ambiguous is empty")
    if not str(item.get("ground_truth_source") or "").strip():
        errors.append("ground_truth_source is empty")
    return errors


def _attach_validation(item: dict[str, Any]) -> dict[str, Any]:
    errors = _validation_errors(item)
    item["quality_control"] = {
        "validation_status": "passed" if not errors else "failed",
        "validation_errors": errors,
    }
    return item


def _iter_valid_observation_tasks(caption_data: dict[str, Any], limit: int | None = None) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for caption_item in caption_data.get("items", []):
        if not isinstance(caption_item, dict):
            continue
        caption = caption_item.get("caption")
        if not isinstance(caption, dict):
            continue
        observations = caption.get("ambiguity_events")
        if not isinstance(observations, list):
            continue
        for obs_index, observation in enumerate(observations, start=1):
            if not isinstance(observation, dict):
                continue
            if observation.get("qa_potential") not in {"high", "medium"}:
                continue
            if not str(observation.get("fusion_conclusion") or "").strip():
                continue
            if not str(observation.get("helper_discriminative_evidence") or "").strip():
                continue
            if len(_hypothesis_texts(observation)) < 2:
                continue
            task = {
                "task_id": f"{caption_item.get('caption_id')}__event{obs_index}",
                "caption_item": caption_item,
                "observation": observation,
                "observation_index": obs_index,
            }
            tasks.append(task)
            if limit is not None and len(tasks) >= limit:
                return tasks
    return tasks


def _task_prompt_payload(task: dict[str, Any]) -> dict[str, Any]:
    caption_item = task["caption_item"]
    caption = caption_item["caption"]
    observation = task["observation"]
    return {
        "task_id": task["task_id"],
        "source_caption_id": caption_item.get("caption_id"),
        "segment_id": caption_item.get("segment_id"),
        "split_dir": caption_item.get("split_dir"),
        "side": caption_item.get("side"),
        "helper_modality": caption_item.get("helper_modality"),
        "victim_modality": caption_item.get("victim_modality"),
        "global_scene": caption.get("global_scene"),
        "helper_modality_analysis": caption.get("helper_modality_analysis"),
        "victim_modality_analysis": caption.get("victim_modality_analysis"),
        "ambiguity_event": observation,
    }


def _build_prompt(tasks: list[dict[str, Any]], qa_per_observation: int) -> str:
    payloads = [_task_prompt_payload(task) for task in tasks]
    return "\n".join(
        [
            "You generate benchmark QA from cross-modal disambiguation captions.",
            "Each input task describes one v3 ambiguity_event where the victim modality alone is ambiguous and the helper modality disambiguates it.",
            f"Generate exactly {qa_per_observation} QA item(s) for each task_id.",
            "The QA must test whether a model can use helper evidence to disambiguate the victim cue.",
            "Use ambiguity_event.fusion_conclusion as the ground-truth answer source.",
            "Use ambiguity_event.candidate_hypotheses as strong distractors when possible.",
            "Use target_entity and approx_time_range to make the question referentially precise when useful.",
            "Do not invent facts beyond the supplied evidence graph and ambiguity event.",
            "Do not ask yes/no questions, counting questions, anomaly questions, or questions whose answer is only a number, truth value, or single color word.",
            "Do not explicitly mention modality names such as RGB, event, IR, depth, or audio in the question.",
            "The question should sound like a natural question about the scene.",
            "Generate a four-option single-choice question.",
            "The correct option must be the concise ground-truth answer.",
            "Use plausible victim-only interpretations as strong distractors when possible.",
            "All four choices must be distinct scene-level answers, not A/B/C/D labels embedded in text.",
            "The victim-only view should leave at least two plausible answers; list those possible answers separately.",
            "The helper evidence must remove that ambiguity.",
            "",
            "Return ONLY valid JSON with this exact structure:",
            "{",
            '  "qa_items": [',
            "    {",
            '      "task_id": "must exactly match one input task_id",',
            '      "question_type": "single_choice",',
            '      "question": "...",',
            '      "choices": {',
            '        "A": "...",',
            '        "B": "...",',
            '        "C": "...",',
            '        "D": "..."',
            "      },",
            '      "answer_choice": "A|B|C|D",',
            '      "answer": "...",',
            '      "possible_answers_from_victim_only": ["...", "..."],',
            '      "why_victim_alone_is_ambiguous": "...",',
            '      "helper_disambiguating_evidence": "...",',
            '      "ground_truth_source": "Quote or paraphrase the fusion_conclusion used as answer source.",',
            '      "answer_type": "object_action_relation|action_phase|spatial_relation|motion_interaction|text_semantic_identity|visibility_or_state|other",',
            '      "question_difficulty": "easy|medium|hard",',
            '      "ground_truth_confidence": 0.0',
            "    }",
            "  ]",
            "}",
            "",
            "Input tasks:",
            json.dumps(payloads, indent=2, ensure_ascii=False),
        ]
    )


def _template_qa(task: dict[str, Any], qa_index: int) -> dict[str, Any]:
    caption_item = task["caption_item"]
    observation = task["observation"]
    question = "What is the ambiguous action or interaction actually referring to in the scene?"
    answer = str(observation.get("fusion_conclusion") or "").strip()
    choices, answer_choice = _build_template_choices(
        answer,
        _hypothesis_texts(observation),
        observation,
    )
    return _normalize_raw_qa(
        {
            "task_id": task["task_id"],
            "question_type": "single_choice",
            "question": question,
            "choices": choices,
            "answer_choice": answer_choice,
            "answer": answer,
            "possible_answers_from_victim_only": _hypothesis_texts(observation),
            "why_victim_alone_is_ambiguous": observation.get("why_victim_cannot_resolve"),
            "helper_disambiguating_evidence": observation.get("helper_discriminative_evidence"),
            "ground_truth_source": observation.get("fusion_conclusion"),
            "answer_type": "other",
            "question_difficulty": "medium",
            "ground_truth_confidence": 0.7,
        },
        task,
        qa_index,
        generation_mode="template",
        model_name=None,
    )


def _normalize_raw_qa(
    raw: dict[str, Any],
    task: dict[str, Any],
    qa_index: int,
    generation_mode: str,
    model_name: str | None,
) -> dict[str, Any]:
    caption_item = task["caption_item"]
    caption = caption_item["caption"]
    observation = task["observation"]
    qa_id = f"{task['task_id']}__qa{qa_index}"
    choices = _coerce_choices(raw.get("choices"))
    answer_choice = str(raw.get("answer_choice") or "").strip().upper()
    answer = _normalize_choice_text(raw.get("answer") or observation.get("fusion_conclusion") or "")
    if answer_choice not in CHOICE_LABELS:
        for label in CHOICE_LABELS:
            if choices.get(label, "").lower() == answer.lower():
                answer_choice = label
                break
    if answer_choice in CHOICE_LABELS and all(choices.get(label) for label in CHOICE_LABELS):
        choices, answer_choice = _shuffle_choices(choices, answer_choice, qa_id)
    if answer_choice in CHOICE_LABELS and choices.get(answer_choice):
        answer = choices[answer_choice]
    item = {
        "qa_id": qa_id,
        "schema_version": QA_SCHEMA_VERSION,
        "qa_style": QA_STYLE,
        "question_type": "single_choice",
        "generation_mode": generation_mode,
        "model_name": model_name,
        "source_caption_id": caption_item.get("caption_id"),
        "source_observation_index": task["observation_index"],
        "segment_id": caption_item.get("segment_id"),
        "split_dir": caption_item.get("split_dir"),
        "segment_name": caption_item.get("segment_name"),
        "side": caption_item.get("side"),
        "helper_modality": caption_item.get("helper_modality"),
        "victim_modality": caption_item.get("victim_modality"),
        "question": str(raw.get("question") or "").strip(),
        "choices": choices,
        "answer_choice": answer_choice,
        "answer": answer,
        "possible_answers_from_victim_only": [
            str(value).strip()
            for value in raw.get("possible_answers_from_victim_only", _hypothesis_texts(observation))
            if str(value).strip()
        ],
        "why_victim_alone_is_ambiguous": str(
            raw.get("why_victim_alone_is_ambiguous") or observation.get("why_victim_cannot_resolve") or ""
        ).strip(),
        "helper_disambiguating_evidence": str(
            raw.get("helper_disambiguating_evidence") or observation.get("helper_discriminative_evidence") or ""
        ).strip(),
        "ground_truth_source": str(raw.get("ground_truth_source") or observation.get("fusion_conclusion") or "").strip(),
        "answer_type": str(raw.get("answer_type") or "other").strip(),
        "question_difficulty": str(raw.get("question_difficulty") or "medium").strip(),
        "ground_truth_confidence": float(raw.get("ground_truth_confidence") or 0.75),
        "caption_context": {
            "global_scene": caption.get("global_scene"),
            "helper_modality_analysis": caption.get("helper_modality_analysis"),
            "victim_modality_analysis": caption.get("victim_modality_analysis"),
            "ambiguity_event": observation,
        },
        "input_frames": {
            "helper_frames": caption_item.get("helper_frames") or [],
            "victim_frames": caption_item.get("victim_frames") or [],
            "composite_frames": caption_item.get("composite_frames") or [],
        },
        "answerability_verification": {
            "status": "unverified",
            "victim_alone_should_be_ambiguous": True,
            "helper_should_disambiguate": True,
        },
    }
    return _attach_validation(item)


async def _call_gemini_batch(
    client,
    tasks: list[dict[str, Any]],
    qa_per_observation: int,
    model_name: str,
    max_retries: int,
) -> dict[str, list[dict[str, Any]]]:
    prompt = _build_prompt(tasks, qa_per_observation)

    response_text = await call_with_retry_async(
        lambda: client.models.generate_content(
            model=model_name,
            contents=[prompt],
            config={
                "response_mime_type": "application/json",
                "response_schema": QA_RESPONSE_SCHEMA,
            },
        ),
        max_attempts=max_retries,
        label="Caption QA Gemini call",
    )
    parsed = _parse_json_response(response_text.text)
    raw_items = parsed.get("qa_items")
    if not isinstance(raw_items, list):
        raise ValueError("Gemini response qa_items must be a list")
    tasks_by_id = {task["task_id"]: task for task in tasks}
    grouped: dict[str, list[dict[str, Any]]] = {task["task_id"]: [] for task in tasks}
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        task_id = str(raw.get("task_id") or "").strip()
        task = tasks_by_id.get(task_id)
        if task is None:
            continue
        if len(grouped[task_id]) >= qa_per_observation:
            continue
        grouped[task_id].append(raw)
    missing = [task_id for task_id, values in grouped.items() if len(values) != qa_per_observation]
    if missing:
        raise ValueError(f"Gemini response missing QA items for task_id(s): {', '.join(missing)}")
    return grouped


def _chunk(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


async def _call_gemini_batch_with_split_fallback(
    client,
    tasks: list[dict[str, Any]],
    qa_per_observation: int,
    model_name: str,
    max_retries: int,
) -> tuple[dict[str, list[dict[str, Any]]], int, list[dict[str, Any]]]:
    try:
        return (
            await _call_gemini_batch(
                client,
                tasks,
                qa_per_observation=qa_per_observation,
                model_name=model_name,
                max_retries=max_retries,
            ),
            1,
            [],
        )
    except Exception as exc:
        if len(tasks) == 1:
            task = tasks[0]
            return (
                {},
                1,
                [
                    {
                        "task_id": task["task_id"],
                        "source_caption_id": task["caption_item"].get("caption_id"),
                        "source_observation_index": task["observation_index"],
                        "reason": str(exc),
                    }
                ],
            )
        grouped: dict[str, list[dict[str, Any]]] = {}
        calls = 1
        skipped: list[dict[str, Any]] = []
        print(f"WARNING: QA batch failed; retrying {len(tasks)} observation(s) individually: {exc}")
        for task in tasks:
            task_grouped, task_calls, task_skipped = await _call_gemini_batch_with_split_fallback(
                client,
                [task],
                qa_per_observation=qa_per_observation,
                model_name=model_name,
                max_retries=max_retries,
            )
            calls += task_calls
            grouped.update(task_grouped)
            skipped.extend(task_skipped)
        return grouped, calls, skipped


def _build_output_payload(
    input_path: Path,
    output_path: Path,
    model_name: str,
    generation_mode: str,
    qa_per_observation: int,
    max_observations_per_gemini_call: int,
    planned_tasks: int,
    gemini_calls: int,
    qa_items: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
) -> dict[str, Any]:
    passed = sum(1 for item in qa_items if item.get("quality_control", {}).get("validation_status") == "passed")
    return {
        "metadata": {
            "task": "cross_modal_disambiguation_qa_generation",
            "schema_version": QA_SCHEMA_VERSION,
            "input": input_path.as_posix(),
            "output": output_path.as_posix(),
            "generation_mode": generation_mode,
            "model_name": model_name,
            "qa_per_observation": qa_per_observation,
            "max_observations_per_gemini_call": max_observations_per_gemini_call,
            "planned_observation_tasks": planned_tasks,
            "completed_qa_items": len(qa_items),
            "passed_validation": passed,
            "failed_validation": len(qa_items) - passed,
            "skipped_tasks": len(skipped),
            "gemini_calls": gemini_calls,
            "distribution": {
                "by_direction": dict(
                    sorted(Counter(f"{item.get('helper_modality')}->{item.get('victim_modality')}" for item in qa_items).items())
                ),
                "by_split_dir": dict(sorted(Counter(str(item.get("split_dir") or "") for item in qa_items).items())),
            },
        },
        "qa_items": sorted(qa_items, key=lambda item: str(item.get("qa_id") or "")),
        "skipped": skipped,
    }


def _load_resume(output_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not output_path.exists():
        return [], []
    data = _load_json(output_path)
    qa_items = data.get("qa_items") if isinstance(data.get("qa_items"), list) else []
    skipped = data.get("skipped") if isinstance(data.get("skipped"), list) else []
    return list(qa_items), list(skipped)


async def run_caption_qa_pipeline_async(
    input_path: Path,
    output_path: Path,
    model_name: str,
    generation_mode: str,
    qa_per_observation: int,
    max_observations_per_gemini_call: int,
    limit: int | None,
    max_retries: int,
    delay_between_calls: int,
    checkpoint_every: int,
    resume: bool,
) -> Path:
    caption_data = _load_json(input_path)
    tasks = _iter_valid_observation_tasks(caption_data, limit=limit)
    qa_items, skipped = _load_resume(output_path) if resume else ([], [])
    existing_counts = Counter(
        f"{item.get('source_caption_id')}__obs{item.get('source_observation_index')}"
        for item in qa_items
    )
    pending_tasks = [task for task in tasks if existing_counts.get(task["task_id"], 0) < qa_per_observation]
    batch_size = max(1, max_observations_per_gemini_call if generation_mode == "gemini" else 1)
    batches = _chunk(pending_tasks, batch_size)
    client = create_gemini_client() if generation_mode == "gemini" else None
    gemini_calls = 0
    checkpoint_counter = 0

    print(
        f"Generating QA from captions: {len(tasks)} observation task(s), "
        f"{len(pending_tasks)} pending, {len(batches)} batch(es), "
        f"mode={generation_mode}, model={model_name}, batch_size={batch_size}."
    )

    def save_checkpoint() -> None:
        _save_json(
            _build_output_payload(
                input_path=input_path,
                output_path=output_path,
                model_name=model_name,
                generation_mode=generation_mode,
                qa_per_observation=qa_per_observation,
                max_observations_per_gemini_call=batch_size,
                planned_tasks=len(tasks),
                gemini_calls=gemini_calls,
                qa_items=qa_items,
                skipped=skipped,
            ),
            output_path,
        )

    for batch_index, task_batch in enumerate(batches, start=1):
        label = ", ".join(task["task_id"] for task in task_batch)
        print(f"  QA batch [{batch_index}/{len(batches)}] {len(task_batch)} observation(s): {label}")
        try:
            if generation_mode == "gemini":
                assert client is not None
                grouped_raw, call_count, skipped_tasks = await _call_gemini_batch_with_split_fallback(
                    client,
                    task_batch,
                    qa_per_observation=qa_per_observation,
                    model_name=model_name,
                    max_retries=max_retries,
                )
                gemini_calls += call_count
                skipped.extend(skipped_tasks)
                for task in task_batch:
                    existing_count = existing_counts.get(task["task_id"], 0)
                    for offset, raw in enumerate(grouped_raw.get(task["task_id"], []), start=1):
                        qa_index = existing_count + offset
                        qa_items.append(
                            _normalize_raw_qa(
                                raw,
                                task,
                                qa_index,
                                generation_mode="gemini",
                                model_name=model_name,
                            )
                        )
                    existing_counts[task["task_id"]] += len(grouped_raw.get(task["task_id"], []))
            else:
                for task in task_batch:
                    existing_count = existing_counts.get(task["task_id"], 0)
                    needed = qa_per_observation - existing_count
                    for offset in range(1, needed + 1):
                        qa_index = existing_count + offset
                        qa_items.append(_template_qa(task, qa_index))
                    existing_counts[task["task_id"]] += needed
        except Exception as exc:
            for task in task_batch:
                skipped.append(
                    {
                        "task_id": task["task_id"],
                        "source_caption_id": task["caption_item"].get("caption_id"),
                        "source_observation_index": task["observation_index"],
                        "reason": str(exc),
                    }
                )
            print(f"WARNING: QA generation failed for batch containing {label}: {exc}")

        checkpoint_counter += 1
        if checkpoint_every > 0 and checkpoint_counter >= checkpoint_every:
            checkpoint_counter = 0
            save_checkpoint()
        if generation_mode == "gemini" and delay_between_calls > 0 and batch_index < len(batches):
            await asyncio.sleep(delay_between_calls)

    save_checkpoint()
    print(f"Wrote caption QA output to {output_path}")
    return output_path


def run_caption_qa_pipeline(
    input_path: Path | str = DEFAULT_INPUT_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
    generation_mode: str = "template",
    qa_per_observation: int = 1,
    max_observations_per_gemini_call: int = 1,
    limit: int | None = None,
    max_retries: int = 3,
    delay_between_calls: int = 5,
    checkpoint_every: int = 1,
    resume: bool = True,
) -> Path:
    return asyncio.run(
        run_caption_qa_pipeline_async(
            input_path=Path(input_path),
            output_path=Path(output_path),
            model_name=model_name,
            generation_mode=generation_mode,
            qa_per_observation=max(1, qa_per_observation),
            max_observations_per_gemini_call=max(1, max_observations_per_gemini_call),
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
    parser.add_argument("--generation-mode", choices=("template", "gemini"), default="template")
    parser.add_argument("--qa-per-observation", type=int, default=1)
    parser.add_argument("--max-observations-per-gemini-call", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--delay-between-calls", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    run_caption_qa_pipeline(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model_name,
        generation_mode=args.generation_mode,
        qa_per_observation=args.qa_per_observation,
        max_observations_per_gemini_call=args.max_observations_per_gemini_call,
        limit=args.limit,
        max_retries=args.max_retries,
        delay_between_calls=args.delay_between_calls,
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
