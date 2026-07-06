"""Generate pairwise ambiguity events from segment-level global evidence."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from annotation_feature.aligned_global_evidence_pipeline import GLOBAL_EVIDENCE_SCHEMA_VERSION
from annotation_feature.aligned_multimodal_caption_pipeline import (
    ALLOWED_MISSING_ATTRIBUTE_TYPES,
    ALLOWED_QA_POTENTIAL,
    CAPTION_SCHEMA_VERSION,
    DEFAULT_COMPOSITE_ROOT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_INPUT_PATH,
    DEFAULT_MODEL_NAME,
    CaptionTask,
    _compose_frame,
    _frame_index,
    _parse_json_response,
    _require_list,
    _require_object,
    _require_string,
    _safe_name,
    build_caption_tasks,
)
from annotation_feature.pipeline.client import create_gemini_client
from annotation_feature.pipeline.utils import build_image_parts


DEFAULT_GLOBAL_EVIDENCE_PATH = Path("outputs/aligned_global_evidence_v3_gemini.json")
DEFAULT_OUTPUT_PATH = Path("outputs/aligned_pairwise_ambiguity_v3_gemini.json")
PAIRWISE_AMBIGUITY_SCHEMA_VERSION = "aligned_pairwise_ambiguity_v3"
ENTITY_CATEGORY_CONFLICTS = {
    "tea_infuser": ("teapot", "kettle"),
    "infuser": ("teapot", "kettle"),
    "teapot": ("infuser", "kettle"),
    "kettle": ("infuser", "teapot"),
    "mailbox": ("shoe", "floor", "window"),
    "person": ("mailbox", "shoe", "window"),
}


@dataclass(frozen=True)
class PairwiseAmbiguityTask:
    caption_id: str
    segment_id: str
    split_dir: str
    segment_name: str
    side: str
    helper_modality: str
    victim_modality: str
    helper_frame_dir: Path
    victim_frame_dir: Path
    helper_frames: tuple[Path, ...]
    victim_frames: tuple[Path, ...]
    composite_frames: tuple[Path, ...]

    @classmethod
    def from_caption_task(
        cls,
        task: CaptionTask,
        composite_root: Path,
        write_composites: bool,
    ) -> "PairwiseAmbiguityTask":
        caption_id = task.caption_id.replace(
            f"__{task.context_modality}_context__{task.decisive_modality}_decisive",
            f"__{task.context_modality}_helper__{task.decisive_modality}_victim",
        )
        output_dir = (
            composite_root
            / _safe_name(task.split_dir)
            / _safe_name(task.segment_name)
            / _safe_name(task.side)
            / f"{task.context_modality}_helper__{task.decisive_modality}_victim"
        )
        composite_frames: list[Path] = []
        for index, (helper_frame, victim_frame) in enumerate(zip(task.context_frames, task.decisive_frames), start=1):
            frame_number = _frame_index(helper_frame)
            suffix = f"{frame_number:06d}" if frame_number is not None else f"{index:03d}"
            output_path = output_dir / f"frame_{suffix}.png"
            if write_composites:
                _compose_frame(
                    helper_frame,
                    victim_frame,
                    task.context_modality,
                    task.decisive_modality,
                    output_path,
                )
            composite_frames.append(output_path)
        return cls(
            caption_id=caption_id,
            segment_id=task.segment_id,
            split_dir=task.split_dir,
            segment_name=task.segment_name,
            side=task.side,
            helper_modality=task.context_modality,
            victim_modality=task.decisive_modality,
            helper_frame_dir=task.context_frame_dir,
            victim_frame_dir=task.decisive_frame_dir,
            helper_frames=task.context_frames,
            victim_frames=task.decisive_frames,
            composite_frames=tuple(composite_frames),
        )


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


def _evidence_id(segment_id: str, side: str) -> str:
    return "__".join(
        [
            "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in segment_id).strip("_").lower(),
            "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in side).strip("_").lower(),
            "global_evidence",
        ]
    )


def _load_global_evidence_map(path: Path) -> dict[str, dict[str, Any]]:
    data = _load_json(path)
    metadata = data.get("metadata") or {}
    if metadata.get("schema_version") != GLOBAL_EVIDENCE_SCHEMA_VERSION:
        raise ValueError(
            f"Expected global evidence schema_version {GLOBAL_EVIDENCE_SCHEMA_VERSION!r}, "
            f"got {metadata.get('schema_version')!r}"
        )
    evidence_by_id: dict[str, dict[str, Any]] = {}
    for item in data.get("items", []):
        if not isinstance(item, dict):
            continue
        evidence = item.get("evidence")
        evidence_id = str(item.get("evidence_id") or "")
        if evidence_id and isinstance(evidence, dict):
            evidence_by_id[evidence_id] = item
    return evidence_by_id


def _encode_images(paths: tuple[Path, ...]) -> list[str]:
    encoded: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        with open(path, "rb") as handle:
            encoded.append(base64.standard_b64encode(handle.read()).decode("utf-8"))
    return encoded


def _global_payload_for_prompt(global_item: dict[str, Any], task: PairwiseAmbiguityTask) -> dict[str, Any]:
    evidence = global_item["evidence"]
    observations = evidence.get("modality_observations") or {}
    return {
        "evidence_id": global_item.get("evidence_id"),
        "segment_id": global_item.get("segment_id"),
        "side": global_item.get("side"),
        "global_scene": evidence.get("global_scene"),
        "victim_modality": task.victim_modality,
        "helper_modality": task.helper_modality,
        "victim_modality_observation": observations.get(task.victim_modality),
        "helper_modality_observation": observations.get(task.helper_modality),
    }


def _build_pairwise_prompt(task: PairwiseAmbiguityTask, global_item: dict[str, Any]) -> str:
    frame_names = ", ".join(path.name for path in task.composite_frames)
    payload = _global_payload_for_prompt(global_item, task)
    return "\n".join(
        [
            "You are a pairwise cross-modal ambiguity agent.",
            "You will receive side-by-side composite frames and a segment-level global evidence graph.",
            f"Left side is the helper modality: {task.helper_modality}.",
            f"Right side is the victim modality: {task.victim_modality}.",
            "Do not regenerate global_scene or per-modality observations. Use the supplied global evidence as fixed context.",
            "Your only job is to enumerate valid ambiguity_events where the victim modality alone supports multiple hypotheses and the helper modality provides independent discriminative evidence.",
            "Every ambiguity_event.target_entity must exactly match one entity_id from global_scene.physical_entities.",
            "The fusion_conclusion must preserve the target_entity identity from global_scene.",
            "Do not silently relabel the target as a different object category. For example, if target_entity is tea_infuser, do not conclude it is a teapot or kettle unless the conclusion explicitly states this as a corrected global-evidence label.",
            "Reject cases where the victim modality already resolves the final fact, where helper evidence merely repeats the victim cue, or where the modalities refer to different targets.",
            "Prefer high-entropy object identity, action phase, spatial relation, motion trend, interaction, location, or text/semantic identity facts.",
            "Avoid yes/no facts, simple counting, single-color answers, and generic display/readout questions when stronger physical interaction ambiguity exists.",
            "Return ONLY valid JSON with this exact structure:",
            "{",
            f'  "schema_version": "{PAIRWISE_AMBIGUITY_SCHEMA_VERSION}",',
            '  "ambiguity_events": [',
            "    {",
            '      "target_entity": "entity_id from global_scene.physical_entities",',
            '      "approx_time_range": "early sampled frame|middle sampled frame|late sampled frame|specific frame names",',
            f'      "victim_modality": "{task.victim_modality}",',
            f'      "helper_modality": "{task.helper_modality}",',
            '      "low_confidence_observation": "What the victim modality shows by itself.",',
            '      "why_victim_cannot_resolve": "Specific reason the victim modality cannot uniquely interpret the cue.",',
            '      "candidate_hypotheses": [{"hypothesis": "...", "support_from_victim": "..."}],',
            '      "why_helper_can_resolve": "Specific independent feature from the helper modality.",',
            '      "helper_discriminative_evidence": "Concrete cue that eliminates at least one hypothesis.",',
            '      "eliminated_hypotheses": [{"hypothesis": "...", "why_eliminated": "..."}],',
            '      "fusion_conclusion": "Final physical fact after combining both modalities.",',
            '      "missing_attribute_type": "existence|target_category|spatial_distance|surface_attribute|motion_trend",',
            '      "qa_potential": "high|medium|low"',
            "    }",
            "  ],",
            '  "rejected_observations": [{"observation": "...", "reason": "..."}]',
            "}",
            "If no valid ambiguity exists, return an empty ambiguity_events list and explain why in rejected_observations.",
            f"Caption task: {task.caption_id}.",
            f"Composite frames ({len(task.composite_frames)} images): {frame_names}",
            "Fixed global evidence payload:",
            json.dumps(payload, ensure_ascii=False, indent=2),
        ]
    )


def _validate_pairwise_schema(
    parsed: dict[str, Any],
    task: PairwiseAmbiguityTask,
    global_item: dict[str, Any],
) -> dict[str, Any]:
    required_fields = ("schema_version", "ambiguity_events", "rejected_observations")
    missing = [field for field in required_fields if field not in parsed]
    if missing:
        raise ValueError(f"Gemini response missing required pairwise field(s): {', '.join(missing)}")
    if parsed["schema_version"] != PAIRWISE_AMBIGUITY_SCHEMA_VERSION:
        raise ValueError(
            f"Gemini response schema_version must be {PAIRWISE_AMBIGUITY_SCHEMA_VERSION!r}, "
            f"got {parsed['schema_version']!r}"
        )
    global_scene = _require_object(global_item["evidence"].get("global_scene"), "global_scene")
    entities = _require_list(global_scene.get("physical_entities"), "global_scene.physical_entities")
    entity_by_id = {
        str(entity.get("entity_id")): entity
        for entity in entities
        if isinstance(entity, dict) and entity.get("entity_id")
    }
    entity_ids = set(entity_by_id)
    events = _require_list(parsed["ambiguity_events"], "ambiguity_events")
    for index, event in enumerate(events, start=1):
        if not isinstance(event, dict):
            raise ValueError(f"ambiguity_events[{index}] must be an object")
        target_entity = _require_string(event.get("target_entity"), f"ambiguity_events[{index}].target_entity")
        if target_entity not in entity_ids:
            raise ValueError(f"ambiguity_events[{index}].target_entity must match a global_scene entity_id")
        if event.get("victim_modality") != task.victim_modality:
            raise ValueError(f"ambiguity_events[{index}].victim_modality must be {task.victim_modality!r}")
        if event.get("helper_modality") != task.helper_modality:
            raise ValueError(f"ambiguity_events[{index}].helper_modality must be {task.helper_modality!r}")
        for key in (
            "approx_time_range",
            "low_confidence_observation",
            "why_victim_cannot_resolve",
            "why_helper_can_resolve",
            "helper_discriminative_evidence",
            "fusion_conclusion",
        ):
            _require_string(event.get(key), f"ambiguity_events[{index}].{key}")
        _validate_target_identity_consistency(event, entity_by_id[target_entity], index)
        missing_type = event.get("missing_attribute_type")
        if missing_type not in ALLOWED_MISSING_ATTRIBUTE_TYPES:
            raise ValueError(
                f"ambiguity_events[{index}].missing_attribute_type must be one of "
                f"{sorted(ALLOWED_MISSING_ATTRIBUTE_TYPES)}, got {missing_type!r}"
            )
        if event.get("qa_potential") not in ALLOWED_QA_POTENTIAL:
            raise ValueError(f"ambiguity_events[{index}].qa_potential must be high, medium, or low")
        hypotheses = _require_list(event.get("candidate_hypotheses"), f"ambiguity_events[{index}].candidate_hypotheses")
        if len(hypotheses) < 2:
            raise ValueError(f"ambiguity_events[{index}].candidate_hypotheses must include at least two hypotheses")
        for hyp_index, hypothesis in enumerate(hypotheses, start=1):
            if not isinstance(hypothesis, dict):
                raise ValueError(f"ambiguity_events[{index}].candidate_hypotheses[{hyp_index}] must be an object")
            _require_string(hypothesis.get("hypothesis"), f"ambiguity_events[{index}].candidate_hypotheses[{hyp_index}].hypothesis")
            _require_string(
                hypothesis.get("support_from_victim"),
                f"ambiguity_events[{index}].candidate_hypotheses[{hyp_index}].support_from_victim",
            )
        eliminated = _require_list(event.get("eliminated_hypotheses"), f"ambiguity_events[{index}].eliminated_hypotheses")
        if not eliminated:
            raise ValueError(f"ambiguity_events[{index}].eliminated_hypotheses must not be empty")
        for elim_index, hypothesis in enumerate(eliminated, start=1):
            if not isinstance(hypothesis, dict):
                raise ValueError(f"ambiguity_events[{index}].eliminated_hypotheses[{elim_index}] must be an object")
            _require_string(hypothesis.get("hypothesis"), f"ambiguity_events[{index}].eliminated_hypotheses[{elim_index}].hypothesis")
            _require_string(
                hypothesis.get("why_eliminated"),
                f"ambiguity_events[{index}].eliminated_hypotheses[{elim_index}].why_eliminated",
            )
    _require_list(parsed["rejected_observations"], "rejected_observations")
    return parsed


def _validate_target_identity_consistency(event: dict[str, Any], entity: dict[str, Any], index: int) -> None:
    entity_id = str(entity.get("entity_id") or "").lower()
    category = str(entity.get("category") or "").lower()
    conclusion = str(event.get("fusion_conclusion") or "").lower()
    explicit_correction_markers = (
        "corrected",
        "rather than",
        "instead of",
        "not a",
        "not the",
        "initial label",
        "global evidence label",
    )
    if any(marker in conclusion for marker in explicit_correction_markers):
        return
    target_tokens = set(re.findall(r"[a-z]+", f"{entity_id} {category}"))
    conflict_terms: set[str] = set()
    for token in target_tokens:
        conflict_terms.update(ENTITY_CATEGORY_CONFLICTS.get(token, ()))
    present_conflicts = sorted(term for term in conflict_terms if re.search(rf"\b{re.escape(term)}\b", conclusion))
    if present_conflicts:
        raise ValueError(
            f"ambiguity_events[{index}].fusion_conclusion appears to relabel target_entity "
            f"{entity.get('entity_id')!r} ({entity.get('category')!r}) as conflicting category/categories "
            f"{present_conflicts}; state an explicit correction or keep the target identity stable"
        )


async def _call_gemini_pairwise(
    client,
    task: PairwiseAmbiguityTask,
    global_item: dict[str, Any],
    model_name: str,
    max_retries: int,
) -> dict[str, Any]:
    encoded = _encode_images(task.composite_frames)
    if not encoded:
        raise ValueError(f"No composite frames found for Gemini call: {task.caption_id}")
    contents = build_image_parts(encoded) + [_build_pairwise_prompt(task, global_item)]
    for attempt in range(1, max_retries + 1):
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=model_name,
                contents=contents,
            )
            return _validate_pairwise_schema(_parse_json_response(response.text), task, global_item)
        except Exception as exc:
            if attempt == max_retries:
                raise
            wait_seconds = 30 * attempt if "429" in str(exc) or "quota" in str(exc).lower() else 2 * attempt
            print(
                f"    Pairwise ambiguity Gemini call failed on attempt {attempt}/{max_retries}; "
                f"retrying in {wait_seconds}s: {exc}"
            )
            await asyncio.sleep(wait_seconds)
    raise RuntimeError("Gemini pairwise ambiguity call failed")


def _analysis_for_caption(global_item: dict[str, Any], modality: str) -> dict[str, Any]:
    analysis = dict((global_item["evidence"].get("modality_observations") or {}).get(modality) or {})
    analysis["modality"] = modality
    return analysis


def _caption_from_pairwise(
    task: PairwiseAmbiguityTask,
    global_item: dict[str, Any],
    pairwise: dict[str, Any],
) -> dict[str, Any]:
    evidence = global_item["evidence"]
    return {
        "schema_version": CAPTION_SCHEMA_VERSION,
        "global_scene": evidence.get("global_scene"),
        "helper_modality_analysis": _analysis_for_caption(global_item, task.helper_modality),
        "victim_modality_analysis": _analysis_for_caption(global_item, task.victim_modality),
        "ambiguity_events": pairwise.get("ambiguity_events") or [],
        "rejected_observations": pairwise.get("rejected_observations") or [],
    }


def _template_pairwise() -> dict[str, Any]:
    return {
        "schema_version": PAIRWISE_AMBIGUITY_SCHEMA_VERSION,
        "ambiguity_events": [],
        "rejected_observations": [
            {"observation": "", "reason": "template mode; Gemini was not called"}
        ],
    }


def _task_to_item(
    task: PairwiseAmbiguityTask,
    global_item: dict[str, Any],
    status: str,
    pairwise: dict[str, Any] | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    pairwise = pairwise or _template_pairwise()
    return {
        "caption_id": task.caption_id,
        "source_global_evidence_id": global_item.get("evidence_id"),
        "segment_id": task.segment_id,
        "split_dir": task.split_dir,
        "segment_name": task.segment_name,
        "side": task.side,
        "helper_modality": task.helper_modality,
        "victim_modality": task.victim_modality,
        "helper_frame_dir": task.helper_frame_dir.as_posix(),
        "victim_frame_dir": task.victim_frame_dir.as_posix(),
        "helper_frames": [path.as_posix() for path in task.helper_frames],
        "victim_frames": [path.as_posix() for path in task.victim_frames],
        "composite_frames": [path.as_posix() for path in task.composite_frames],
        "status": status,
        "reason": reason,
        "pairwise_ambiguity": pairwise,
        "caption": _caption_from_pairwise(task, global_item, pairwise),
    }


def _has_complementarity(task: PairwiseAmbiguityTask, global_item: dict[str, Any]) -> bool:
    observations = global_item["evidence"].get("modality_observations") or {}
    victim = observations.get(task.victim_modality) or {}
    helper = task.helper_modality
    for missing in victim.get("missing_key_attributes") or []:
        if not isinstance(missing, dict):
            continue
        recoverable_from = {str(value) for value in missing.get("recoverable_from") or []}
        if helper in recoverable_from:
            return True
    return False


def _build_output_payload(
    input_path: Path,
    global_evidence_path: Path,
    dataset_root: Path,
    composite_root: Path,
    output_path: Path,
    model_name: str,
    generation_mode: str,
    num_frames: int,
    prefilter: bool,
    items: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    planned_total: int,
    gemini_calls: int,
) -> dict[str, Any]:
    return {
        "metadata": {
            "task": "aligned_pairwise_ambiguity_generation",
            "schema_version": PAIRWISE_AMBIGUITY_SCHEMA_VERSION,
            "caption_schema_version": CAPTION_SCHEMA_VERSION,
            "input": input_path.as_posix(),
            "global_evidence_input": global_evidence_path.as_posix(),
            "output": output_path.as_posix(),
            "dataset_root": dataset_root.as_posix(),
            "composite_root": composite_root.as_posix(),
            "generation_mode": generation_mode,
            "model_name": model_name,
            "num_frames": num_frames,
            "prefilter": prefilter,
            "planned_items": planned_total,
            "completed_items": len(items),
            "skipped_items": len(skipped),
            "gemini_calls": gemini_calls,
        },
        "items": items,
        "skipped": skipped,
    }


def _load_resume(output_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not output_path.exists():
        return [], []
    try:
        data = _load_json(output_path)
    except Exception as exc:
        print(f"WARNING: Could not load existing pairwise ambiguity output for resume: {exc}")
        return [], []
    items = data.get("items") if isinstance(data.get("items"), list) else []
    skipped = data.get("skipped") if isinstance(data.get("skipped"), list) else []
    return list(items), list(skipped)


async def run_pairwise_ambiguity_pipeline_async(
    input_path: Path,
    global_evidence_path: Path,
    output_path: Path,
    dataset_root: Path,
    composite_root: Path,
    model_name: str,
    generation_mode: str,
    num_frames: int,
    pairs: str | None,
    directions: str | None,
    limit: int | None,
    limit_scenes: int | None,
    limit_scene_folders: int | None,
    prefilter: bool,
    max_retries: int,
    delay_between_calls: int,
    checkpoint_every: int,
    resume: bool,
) -> Path:
    global_evidence = _load_global_evidence_map(global_evidence_path)
    raw_tasks, skipped = build_caption_tasks(
        input_path=input_path,
        dataset_root=dataset_root,
        composite_root=composite_root,
        num_frames=num_frames,
        allowed_pairs=None,
        allowed_directions=None,
        limit=limit,
        limit_scenes=limit_scenes,
        limit_scene_folders=limit_scene_folders,
        write_composites=False,
    )
    tasks = [
        PairwiseAmbiguityTask.from_caption_task(task, composite_root=composite_root, write_composites=True)
        for task in raw_tasks
    ]
    if pairs or directions:
        from annotation_feature.aligned_multimodal_caption_pipeline import _parse_pairs

        allowed_pairs = _parse_pairs(pairs)
        allowed_directions = _parse_pairs(directions)
        if allowed_pairs is not None:
            tasks = [
                task
                for task in tasks
                if (task.helper_modality, task.victim_modality) in allowed_pairs
                or (task.victim_modality, task.helper_modality) in allowed_pairs
            ]
        if allowed_directions is not None:
            tasks = [
                task
                for task in tasks
                if (task.helper_modality, task.victim_modality) in allowed_directions
            ]

    paired_tasks: list[tuple[PairwiseAmbiguityTask, dict[str, Any]]] = []
    for task in tasks:
        evidence_id = _evidence_id(task.segment_id, task.side)
        global_item = global_evidence.get(evidence_id)
        if global_item is None:
            skipped.append(
                {
                    "caption_id": task.caption_id,
                    "segment_id": task.segment_id,
                    "side": task.side,
                    "reason": f"missing global evidence item {evidence_id}",
                }
            )
            continue
        modalities = set(global_item.get("modalities") or [])
        if task.helper_modality not in modalities or task.victim_modality not in modalities:
            skipped.append(
                {
                    "caption_id": task.caption_id,
                    "segment_id": task.segment_id,
                    "side": task.side,
                    "helper_modality": task.helper_modality,
                    "victim_modality": task.victim_modality,
                    "reason": "task modalities are not both present in global evidence",
                }
            )
            continue
        if prefilter and not _has_complementarity(task, global_item):
            skipped.append(
                {
                    "caption_id": task.caption_id,
                    "segment_id": task.segment_id,
                    "side": task.side,
                    "helper_modality": task.helper_modality,
                    "victim_modality": task.victim_modality,
                    "reason": "prefilter: victim missing attributes are not recoverable from helper modality",
                }
            )
            continue
        paired_tasks.append((task, global_item))

    existing_items, existing_skipped = _load_resume(output_path) if resume else ([], [])
    items = existing_items
    skipped = existing_skipped + skipped
    existing_ids = {str(item.get("caption_id")) for item in items if item.get("caption_id")}
    pending = [(task, global_item) for task, global_item in paired_tasks if task.caption_id not in existing_ids]

    client = create_gemini_client() if generation_mode == "gemini" else None
    gemini_calls = 0
    checkpoint_counter = 0

    print(
        f"Generating pairwise ambiguity: {len(paired_tasks)} planned item(s), "
        f"{len(pending)} pending, mode={generation_mode}, model={model_name}, prefilter={prefilter}."
    )

    def save_checkpoint() -> None:
        _save_json(
            _build_output_payload(
                input_path=input_path,
                global_evidence_path=global_evidence_path,
                dataset_root=dataset_root,
                composite_root=composite_root,
                output_path=output_path,
                model_name=model_name,
                generation_mode=generation_mode,
                num_frames=num_frames,
                prefilter=prefilter,
                items=items,
                skipped=skipped,
                planned_total=len(paired_tasks),
                gemini_calls=gemini_calls,
            ),
            output_path,
        )

    for index, (task, global_item) in enumerate(pending, start=1):
        print(f"  Pairwise ambiguity [{index}/{len(pending)}]: {task.caption_id}")
        try:
            if generation_mode == "gemini":
                assert client is not None
                pairwise = await _call_gemini_pairwise(client, task, global_item, model_name, max_retries=max_retries)
                gemini_calls += 1
                status = "generated"
            else:
                pairwise = _template_pairwise()
                status = "template"
            items.append(_task_to_item(task, global_item, status=status, pairwise=pairwise))
        except Exception as exc:
            skipped.append(
                {
                    "caption_id": task.caption_id,
                    "segment_id": task.segment_id,
                    "side": task.side,
                    "helper_modality": task.helper_modality,
                    "victim_modality": task.victim_modality,
                    "reason": str(exc),
                }
            )
            print(f"WARNING: Pairwise ambiguity generation failed for {task.caption_id}: {exc}")

        checkpoint_counter += 1
        if checkpoint_every > 0 and checkpoint_counter >= checkpoint_every:
            checkpoint_counter = 0
            save_checkpoint()
        if generation_mode == "gemini" and delay_between_calls > 0 and index < len(pending):
            await asyncio.sleep(delay_between_calls)

    save_checkpoint()
    print(f"Wrote pairwise ambiguity output to {output_path}")
    return output_path


def run_pairwise_ambiguity_pipeline(
    input_path: Path | str = DEFAULT_INPUT_PATH,
    global_evidence_path: Path | str = DEFAULT_GLOBAL_EVIDENCE_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    dataset_root: Path | str = DEFAULT_DATASET_ROOT,
    composite_root: Path | str = DEFAULT_COMPOSITE_ROOT,
    model_name: str = DEFAULT_MODEL_NAME,
    generation_mode: str = "template",
    num_frames: int = 6,
    pairs: str | None = None,
    directions: str | None = None,
    limit: int | None = None,
    limit_scenes: int | None = None,
    limit_scene_folders: int | None = None,
    prefilter: bool = False,
    max_retries: int = 3,
    delay_between_calls: int = 5,
    checkpoint_every: int = 1,
    resume: bool = True,
) -> Path:
    return asyncio.run(
        run_pairwise_ambiguity_pipeline_async(
            input_path=Path(input_path),
            global_evidence_path=Path(global_evidence_path),
            output_path=Path(output_path),
            dataset_root=Path(dataset_root),
            composite_root=Path(composite_root),
            model_name=model_name,
            generation_mode=generation_mode,
            num_frames=num_frames,
            pairs=pairs,
            directions=directions,
            limit=limit,
            limit_scenes=limit_scenes,
            limit_scene_folders=limit_scene_folders,
            prefilter=prefilter,
            max_retries=max_retries,
            delay_between_calls=delay_between_calls,
            checkpoint_every=checkpoint_every,
            resume=resume,
        )
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--global-evidence", default=str(DEFAULT_GLOBAL_EVIDENCE_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--composite-root", default=str(DEFAULT_COMPOSITE_ROOT))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--generation-mode",
        choices=("template", "gemini"),
        default="template",
        help="Use template to build pairwise items without calling Gemini.",
    )
    parser.add_argument("--num-frames", type=int, default=6)
    parser.add_argument("--pairs", default=None)
    parser.add_argument("--directions", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit-scenes", "--limit-segments", dest="limit_scenes", type=int, default=None)
    parser.add_argument("--limit-scene-folders", "--limit-split-dirs", dest="limit_scene_folders", type=int, default=None)
    parser.add_argument("--prefilter", action="store_true")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--delay-between-calls", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    run_pairwise_ambiguity_pipeline(
        input_path=args.input,
        global_evidence_path=args.global_evidence,
        output_path=args.output,
        dataset_root=args.dataset_root,
        composite_root=args.composite_root,
        model_name=args.model_name,
        generation_mode=args.generation_mode,
        num_frames=max(1, args.num_frames),
        pairs=args.pairs,
        directions=args.directions,
        limit=args.limit,
        limit_scenes=args.limit_scenes,
        limit_scene_folders=args.limit_scene_folders,
        prefilter=args.prefilter,
        max_retries=max(1, args.max_retries),
        delay_between_calls=max(0, args.delay_between_calls),
        checkpoint_every=max(0, args.checkpoint_every),
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
