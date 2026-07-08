"""Generate segment-level global evidence graphs from aligned frame caches."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from annotation_feature.aligned_caption_schema import ALLOWED_MISSING_ATTRIBUTE_TYPES
from annotation_feature.aligned_caption_validation import _require_list, _require_object, _require_string
from annotation_feature.aligned_multimodal_caption_pipeline import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_INPUT_PATH,
    DEFAULT_MODEL_NAME,
    _parse_json_response,
    _safe_name,
)
from annotation_feature.aligned_multimodal_evidence import VISUAL_PAIRS
from annotation_feature.aligned_multimodal_sampling import (
    SIDE_ORDER,
    evenly_sample,
    frame_index,
    frames_by_index,
    load_frame_dirs,
)
from annotation_feature.pipeline.client import create_gemini_client
from annotation_feature.pipeline.utils import build_image_parts


DEFAULT_OUTPUT_PATH = Path("outputs/aligned_global_evidence_v3_gemini.json")
DEFAULT_CONTACT_SHEET_ROOT = Path("outputs/global_evidence_frames")
GLOBAL_EVIDENCE_SCHEMA_VERSION = "aligned_global_evidence_v3"
VISUAL_MODALITIES = ("rgb", "event", "depth", "ir")
SAMPLING_STRATEGIES = ("uniform", "event_activity", "hybrid_event")
MIN_DETAILED_CAPTION_WORDS = 45
MIN_SCENE_SUMMARY_WORDS = 55
MIN_FRAME_DETAIL_WORDS = 12
MIN_OBSERVABLE_FACTS = 3
GENERIC_SENSOR_EXPLANATION_PATTERNS = (
    re.compile(r"\bevent cameras?\s+(capture|detect|record|respond)", re.I),
    re.compile(r"\b(depth|rgb|infrared|ir)\s+(camera|sensor)s?\s+(capture|detect|record|measure)", re.I),
    re.compile(r"\bthis modality\s+(captures|detects|records|measures)", re.I),
    re.compile(r"\bdesigned to\s+(capture|detect|record|measure)", re.I),
)


@dataclass(frozen=True)
class GlobalEvidenceTask:
    evidence_id: str
    segment_id: str
    split_dir: str
    segment_name: str
    side: str
    modalities: tuple[str, ...]
    frame_dirs: dict[str, Path]
    frames_by_modality: dict[str, tuple[Path, ...]]
    sharedframe_indexes: tuple[int, ...]
    contact_sheet: Path


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


def _load_font(size: int = 22) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_width, target_height = size
    image = image.convert("RGB")
    scale = min(target_width / image.width, target_height / image.height)
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    resized = image.resize((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (0, 0, 0))
    canvas.paste(resized, ((target_width - width) // 2, (target_height - height) // 2))
    return canvas


def _draw_label(image: Image.Image, label: str, xy: tuple[int, int]) -> None:
    draw = ImageDraw.Draw(image)
    font = _load_font()
    x, y = xy
    padding = 6
    bbox = draw.textbbox((x, y), label, font=font)
    draw.rectangle(
        (
            bbox[0] - padding,
            bbox[1] - padding,
            bbox[2] + padding,
            bbox[3] + padding,
        ),
        fill=(0, 0, 0),
    )
    draw.text((x, y), label, fill=(255, 255, 255), font=font)


def _compose_contact_sheet(
    task_id: str,
    modalities: tuple[str, ...],
    frames_by_modality: dict[str, tuple[Path, ...]],
    sharedframe_indexes: tuple[int, ...],
    output_path: Path,
    cell_size: tuple[int, int] = (360, 260),
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cell_width, cell_height = cell_size
    rows = max(1, len(sharedframe_indexes))
    cols = len(modalities)
    header_height = 42
    canvas = Image.new(
        "RGB",
        (cell_width * cols, header_height + cell_height * rows),
        (18, 18, 18),
    )
    draw = ImageDraw.Draw(canvas)
    font = _load_font()
    for col, modality in enumerate(modalities):
        x = col * cell_width
        draw.rectangle((x, 0, x + cell_width, header_height), fill=(38, 38, 38))
        draw.text((x + 10, 10), modality.upper(), fill=(255, 255, 255), font=font)
    for row, frame_index in enumerate(sharedframe_indexes):
        y = header_height + row * cell_height
        for col, modality in enumerate(modalities):
            x = col * cell_width
            frames = frames_by_modality[modality]
            frame = frames[row]
            with Image.open(frame) as raw:
                fitted = _fit_image(raw, cell_size)
            canvas.paste(fitted, (x, y))
            _draw_label(canvas, f"{modality.upper()} frame_{frame_index:06d}", (x + 10, y + 10))
    _draw_label(canvas, task_id, (10, canvas.height - 34))
    canvas.save(output_path)
    return output_path


def _available_pairs(modalities: set[str]) -> list[tuple[str, str]]:
    return [
        (first, second)
        for first, second in VISUAL_PAIRS
        if first in modalities and second in modalities
    ]


def _evidence_id(segment_id: str, side: str) -> str:
    return "__".join([_safe_name(segment_id).lower(), _safe_name(side).lower(), "global_evidence"])


def _event_activity_score(path: Path) -> float:
    try:
        with Image.open(path) as image:
            gray = image.convert("L")
            arr = np.asarray(gray)
    except Exception:
        return 0.0
    if arr.size == 0:
        return 0.0
    return float(np.count_nonzero(arr)) / float(arr.size)


def _sample_by_event_activity(
    shared_indexes: list[int],
    event_by_index: dict[int, Path],
    count: int,
) -> list[int]:
    if count <= 0 or not shared_indexes:
        return []
    if len(shared_indexes) <= count:
        return shared_indexes

    selected: set[int] = {shared_indexes[0], shared_indexes[-1]}
    remaining_count = max(0, count - len(selected))
    if remaining_count <= 0:
        return sorted(selected)

    activity_scores = [
        (index, _event_activity_score(event_by_index.get(index, Path())))
        for index in shared_indexes[1:-1]
    ]
    activity_scores.sort(key=lambda item: item[1], reverse=True)
    for index, _score in activity_scores[:remaining_count]:
        selected.add(index)
    return sorted(selected)


def _hybrid_event_sample(
    shared_indexes: list[int],
    event_by_index: dict[int, Path],
    count: int,
) -> list[int]:
    if count <= 0 or not shared_indexes:
        return []
    if len(shared_indexes) <= count:
        return shared_indexes

    selected: set[int] = {shared_indexes[0], shared_indexes[-1]}
    remaining_count = max(0, count - len(selected))
    if remaining_count <= 0:
        return sorted(selected)

    uniform_count = max(1, remaining_count // 2)
    event_count = remaining_count - uniform_count

    for index in evenly_sample(shared_indexes[1:-1], uniform_count):
        selected.add(index)

    if event_count > 0:
        activity_scores = [
            (index, _event_activity_score(event_by_index.get(index, Path())))
            for index in shared_indexes[1:-1]
            if index not in selected
        ]
        activity_scores.sort(key=lambda item: item[1], reverse=True)
        for index, _score in activity_scores[:event_count]:
            selected.add(index)

    if len(selected) < count:
        for index in evenly_sample(shared_indexes, count):
            selected.add(index)
            if len(selected) >= count:
                break
    return sorted(selected)


def _select_shared_indexes(
    shared_indexes: list[int],
    by_modality: dict[str, dict[int, Path]],
    num_frames: int,
    sampling_strategy: str,
) -> tuple[int, ...]:
    if sampling_strategy not in SAMPLING_STRATEGIES:
        raise ValueError(f"sampling_strategy must be one of {SAMPLING_STRATEGIES}, got {sampling_strategy!r}")
    if sampling_strategy == "uniform" or "event" not in by_modality:
        return tuple(evenly_sample(shared_indexes, num_frames))
    event_by_index = by_modality["event"]
    if sampling_strategy == "event_activity":
        return tuple(_sample_by_event_activity(shared_indexes, event_by_index, num_frames))
    return tuple(_hybrid_event_sample(shared_indexes, event_by_index, num_frames))


def _collect_shared_frames(
    frame_dirs: dict[str, Path],
    num_frames: int,
    sampling_strategy: str = "uniform",
) -> tuple[dict[str, tuple[Path, ...]], tuple[int, ...]]:
    by_modality = {
        modality: frames_by_index(frame_dir, modality)
        for modality, frame_dir in frame_dirs.items()
    }
    shared_indexes: set[int] | None = None
    for values in by_modality.values():
        indexes = set(values)
        shared_indexes = indexes if shared_indexes is None else shared_indexes & indexes
    selected = _select_shared_indexes(
        sorted(shared_indexes or []),
        by_modality,
        num_frames,
        sampling_strategy,
    )
    frames_by_modality = {
        modality: tuple(indexed[index] for index in selected)
        for modality, indexed in by_modality.items()
    }
    return frames_by_modality, selected


def build_global_evidence_tasks(
    input_path: Path,
    dataset_root: Path,
    contact_sheet_root: Path,
    num_frames: int,
    sampling_strategy: str = "uniform",
    limit: int | None = None,
    limit_scenes: int | None = None,
    limit_scene_folders: int | None = None,
    write_contact_sheets: bool = True,
) -> tuple[list[GlobalEvidenceTask], list[dict[str, Any]]]:
    if sampling_strategy not in SAMPLING_STRATEGIES:
        raise ValueError(f"sampling_strategy must be one of {SAMPLING_STRATEGIES}, got {sampling_strategy!r}")
    data = _load_json(input_path)
    segments = data.get("segments")
    if not isinstance(segments, dict):
        raise ValueError(f"Expected {input_path} to contain a segments object")

    tasks: list[GlobalEvidenceTask] = []
    skipped: list[dict[str, Any]] = []
    selected_scenes: set[str] = set()
    selected_scene_folders: set[str] = set()
    for segment_id, segment in sorted(segments.items()):
        if limit_scenes is not None and len(selected_scenes) >= limit_scenes:
            break
        if not isinstance(segment, dict):
            continue
        split_dir = str(segment.get("split_dir") or "")
        segment_name = str(segment.get("segment_name") or "")
        if (
            limit_scene_folders is not None
            and split_dir not in selected_scene_folders
            and len(selected_scene_folders) >= limit_scene_folders
        ):
            break
        if not split_dir or not segment_name:
            skipped.append({"segment_id": segment_id, "reason": "missing split_dir or segment_name"})
            continue

        modality_dirs: dict[str, dict[str, Path]] = {
            modality: load_frame_dirs(dataset_root, split_dir, segment_name, modality)
            for modality in VISUAL_MODALITIES
        }
        sides = [
            side
            for side in SIDE_ORDER
            if sum(1 for dirs in modality_dirs.values() if side in dirs) >= 2
        ]
        sides.extend(
            sorted(
                side
                for side in set().union(*(set(dirs) for dirs in modality_dirs.values()))
                if side not in SIDE_ORDER and sum(1 for dirs in modality_dirs.values() if side in dirs) >= 2
            )
        )
        task_count_before_segment = len(tasks)
        for side in sides:
            frame_dirs = {
                modality: dirs[side]
                for modality, dirs in modality_dirs.items()
                if side in dirs
            }
            modalities = tuple(modality for modality in VISUAL_MODALITIES if modality in frame_dirs)
            if len(_available_pairs(set(modalities))) == 0:
                skipped.append(
                    {
                        "segment_id": segment_id,
                        "side": side,
                        "modalities": list(modalities),
                        "reason": "fewer than two supported visual modalities with a configured pair",
                    }
                )
                continue
            frames_by_modality, selected_indexes = _collect_shared_frames(
                frame_dirs,
                num_frames,
                sampling_strategy=sampling_strategy,
            )
            if not selected_indexes:
                skipped.append(
                    {
                        "segment_id": segment_id,
                        "side": side,
                        "modalities": list(modalities),
                        "reason": "no shared frame indexes across available modalities",
                    }
                )
                continue
            evidence_id = _evidence_id(str(segment_id), side)
            contact_sheet = (
                contact_sheet_root
                / _safe_name(split_dir)
                / _safe_name(segment_name)
                / _safe_name(side)
                / "global_evidence_contact_sheet.png"
            )
            task = GlobalEvidenceTask(
                evidence_id=evidence_id,
                segment_id=str(segment_id),
                split_dir=split_dir,
                segment_name=segment_name,
                side=side,
                modalities=modalities,
                frame_dirs=frame_dirs,
                frames_by_modality=frames_by_modality,
                sharedframe_indexes=selected_indexes,
                contact_sheet=contact_sheet,
            )
            if write_contact_sheets:
                _compose_contact_sheet(
                    task_id=evidence_id,
                    modalities=modalities,
                    frames_by_modality=frames_by_modality,
                    sharedframe_indexes=selected_indexes,
                    output_path=contact_sheet,
                )
            tasks.append(task)
            if limit is not None and len(tasks) >= limit:
                return tasks, skipped
        if len(tasks) > task_count_before_segment:
            selected_scenes.add(str(segment_id))
            selected_scene_folders.add(split_dir)
    return tasks, skipped


def _build_global_prompt(task: GlobalEvidenceTask) -> str:
    frame_indexes = ", ".join(f"frame_{index:06d}" for index in task.sharedframe_indexes)
    frame_keys = ", ".join(f'"frame_{index:06d}"' for index in task.sharedframe_indexes)
    frame_detail_example = ", ".join(
        f'"frame_{index:06d}": "..."'
        for index in task.sharedframe_indexes
    )
    modalities = ", ".join(task.modalities)
    return "\n".join(
        [
            "You are a segment-level global evidence graph assistant.",
            "You will receive one contact sheet for a single aligned video segment.",
            "Columns are modalities and rows are shared sampled time indexes.",
            f"Segment: {task.segment_id}; side: {task.side}.",
            f"Modalities present: {modalities}.",
            f"Sampled frame indexes: {frame_indexes}.",
            "Your task is to produce only global physical facts and per-modality evidence.",
            "Do not enumerate pairwise ambiguity events here. Pairwise ambiguity will be generated by a separate agent later.",
            "CRITICAL INSTRUCTION: do not summarize and do not explain sensor theory.",
            "Describe the actual pixels, objects, surfaces, text, poses, edges, motion traces, occlusions, and spatial layout in THIS specific segment.",
            "You must trace the scene chronologically across every sampled frame index. For each frame, state what is present, where it is, what changed since the previous sampled frame, and what remains ambiguous.",
            "Do not write generic statements such as 'event cameras capture temporal changes' or 'depth sensors measure distance'. If a modality is sparse, blank, noisy, or edge-only, describe the exact sparse/blank/noisy/edge pattern in the shown frames.",
            "Use dense, auditable physical evidence: colors, material texture, printed text or numbers, reflections, object parts, hand/body pose, contact state, relative position, motion direction, and occlusion when visible.",
            "scene_summary must be a dense paragraph, not a one-sentence overview.",
            "Each modality detailed_caption must be a dense paragraph grounded in the sampled frames.",
            f"Every frame_by_frame_details object must contain exactly these keys: {frame_keys}.",
            f"observable_facts must contain at least {MIN_OBSERVABLE_FACTS} concrete facts per modality; each fact should be specific enough to support a later QA item.",
            "Global scene fields must describe physical facts, not sensor artifacts. Avoid words such as blurry, noisy, pixel, infrared, thermal, point cloud, visible, invisible in global_scene.",
            "The physical_entities list must cover every main actor and task-relevant object mentioned in scene_summary or temporal_progression.",
            "If a person, hand, arm, body part, vehicle, animal, or moving actor is involved in the action, include it as a separate physical entity with a stable entity_id.",
            "Use stable entity_id values that can be reused later by pairwise ambiguity agents, such as person_at_mailbox, right_hand, kettle_on_heater, mailbox_panel.",
            "Each modality_observations entry should describe only what that modality supports by itself, including concrete observable facts, sensor-specific cues, limitations, and missing key attributes.",
            "Each missing_key_attributes item must use one of exactly five attribute_type values: existence, target_category, spatial_distance, surface_attribute, motion_trend.",
            "Each missing_key_attributes item must name a concrete missing scene attribute, not a generic phrase such as context, background context, visual information, details, or information.",
            "Each recoverable_from list may contain only modality names that are present in this task.",
            "Return ONLY valid JSON with this exact structure:",
            "{",
            f'  "schema_version": "{GLOBAL_EVIDENCE_SCHEMA_VERSION}",',
            '  "global_scene": {',
            '    "scene_summary": "Detailed physical scene summary independent of sensor artifacts.",',
            '    "physical_entities": [',
            '      {"entity_id": "stable_snake_case_id", "category": "...", "appearance_or_state": "...", "location": "...", "motion_or_action": "...", "spatial_relations": ["..."]}',
            "    ],",
            '    "environment": "Objective environment or recording condition if evident.",',
            '    "temporal_progression": "Dense chronological account of how the scene/action changes across sampled frames.",',
            f'    "frame_by_frame_details": {{{frame_detail_example}}}',
            "  },",
            '  "modality_observations": {',
            '    "rgb": {"detailed_caption": "...", "frame_by_frame_details": {"frame_000000": "..."}, "observable_facts": ["..."], "sensor_specific_cues": ["..."], "sensor_limitations": ["..."], "missing_key_attributes": [{"attribute_type": "existence|target_category|spatial_distance|surface_attribute|motion_trend", "missing_attribute": "...", "why_missing": "...", "recoverable_from": ["..."]}]}',
            "  },",
            '  "rejected_modalities": [{"modality": "...", "reason": "..."}]',
            "}",
            "Include exactly one key in modality_observations for every modality listed as present.",
        ]
    )


def _encode_image(path: Path) -> str:
    with open(path, "rb") as handle:
        return base64.standard_b64encode(handle.read()).decode("utf-8")


def _ensure_contact_sheet(task: GlobalEvidenceTask) -> Path:
    if not task.contact_sheet.exists():
        _compose_contact_sheet(
            task_id=task.evidence_id,
            modalities=task.modalities,
            frames_by_modality=task.frames_by_modality,
            sharedframe_indexes=task.sharedframe_indexes,
            output_path=task.contact_sheet,
        )
    return task.contact_sheet


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def _validate_min_words(text: Any, field: str, minimum: int) -> str:
    value = _require_string(text, field)
    if _word_count(value) < minimum:
        raise ValueError(f"{field} is too short; expected at least {minimum} words")
    return value


def _validate_no_generic_sensor_explanation(text: str, field: str) -> None:
    for pattern in GENERIC_SENSOR_EXPLANATION_PATTERNS:
        if pattern.search(text):
            raise ValueError(f"{field} contains generic sensor-theory wording instead of segment-specific evidence")


def _validate_frame_by_frame_details(value: Any, field: str, frame_indexes: tuple[int, ...]) -> None:
    details = _require_object(value, field)
    expected = {f"frame_{index:06d}" for index in frame_indexes}
    actual = set(details)
    if actual != expected:
        raise ValueError(f"{field} keys must be {sorted(expected)}, got {sorted(actual)}")
    for key in sorted(expected):
        text = _validate_min_words(details.get(key), f"{field}.{key}", MIN_FRAME_DETAIL_WORDS)
        _validate_no_generic_sensor_explanation(text, f"{field}.{key}")


def _validate_missing_key_attributes(values: Any, field: str, allowed_modalities: set[str]) -> None:
    for index, item in enumerate(_require_list(values, field), start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{field}[{index}] must be an object")
        attribute_type = item.get("attribute_type")
        if attribute_type not in ALLOWED_MISSING_ATTRIBUTE_TYPES:
            raise ValueError(
                f"{field}[{index}].attribute_type must be one of "
                f"{sorted(ALLOWED_MISSING_ATTRIBUTE_TYPES)}, got {attribute_type!r}"
            )
        for key in ("missing_attribute", "why_missing"):
            _require_string(item.get(key), f"{field}[{index}].{key}")
        missing_attribute = str(item.get("missing_attribute") or "").strip().lower()
        generic_missing_values = {
            "context",
            "background context",
            "static background context",
            "visual information",
            "details",
            "information",
            "scene context",
        }
        if missing_attribute in generic_missing_values:
            raise ValueError(
                f"{field}[{index}].missing_attribute is too generic: {item.get('missing_attribute')!r}"
            )
        recoverable_from = item.get("recoverable_from")
        if not isinstance(recoverable_from, list) or not recoverable_from:
            raise ValueError(f"{field}[{index}].recoverable_from must be a list")
        invalid = [
            str(modality)
            for modality in recoverable_from
            if str(modality) not in allowed_modalities
        ]
        if invalid:
            raise ValueError(
                f"{field}[{index}].recoverable_from contains unavailable modality/modalities: {invalid}; "
                f"available modalities are {sorted(allowed_modalities)}"
            )


def _validate_global_evidence_schema(parsed: dict[str, Any], task: GlobalEvidenceTask) -> dict[str, Any]:
    required_fields = (
        "schema_version",
        "global_scene",
        "modality_observations",
        "rejected_modalities",
    )
    missing = [field for field in required_fields if field not in parsed]
    if missing:
        raise ValueError(f"Gemini response missing required global evidence field(s): {', '.join(missing)}")
    if parsed["schema_version"] != GLOBAL_EVIDENCE_SCHEMA_VERSION:
        raise ValueError(
            f"Gemini response schema_version must be {GLOBAL_EVIDENCE_SCHEMA_VERSION!r}, "
            f"got {parsed['schema_version']!r}"
        )
    global_scene = _require_object(parsed["global_scene"], "global_scene")
    scene_summary = _validate_min_words(global_scene.get("scene_summary"), "global_scene.scene_summary", MIN_SCENE_SUMMARY_WORDS)
    _validate_no_generic_sensor_explanation(scene_summary, "global_scene.scene_summary")
    _require_string(global_scene.get("environment"), "global_scene.environment")
    temporal_progression = _validate_min_words(global_scene.get("temporal_progression"), "global_scene.temporal_progression", MIN_FRAME_DETAIL_WORDS * max(1, len(task.sharedframe_indexes)))
    _validate_no_generic_sensor_explanation(temporal_progression, "global_scene.temporal_progression")
    _validate_frame_by_frame_details(
        global_scene.get("frame_by_frame_details"),
        "global_scene.frame_by_frame_details",
        task.sharedframe_indexes,
    )
    entities = _require_list(global_scene.get("physical_entities"), "global_scene.physical_entities")
    if not entities:
        raise ValueError("global_scene.physical_entities must not be empty")
    for index, entity in enumerate(entities, start=1):
        if not isinstance(entity, dict):
            raise ValueError(f"global_scene.physical_entities[{index}] must be an object")
        for key in ("entity_id", "category", "appearance_or_state", "location", "motion_or_action"):
            _require_string(entity.get(key), f"global_scene.physical_entities[{index}].{key}")
        _require_list(entity.get("spatial_relations"), f"global_scene.physical_entities[{index}].spatial_relations")
    global_text = " ".join(
        str(global_scene.get(key) or "").lower()
        for key in ("scene_summary", "temporal_progression")
    )
    entity_text = " ".join(
        " ".join(str(entity.get(key) or "").lower() for key in ("entity_id", "category"))
        for entity in entities
        if isinstance(entity, dict)
    )
    actor_tokens = ("person", "human", "hand", "arm", "body", "vehicle", "animal")
    if any(token in global_text for token in actor_tokens) and not any(token in entity_text for token in actor_tokens):
        raise ValueError(
            "global_scene mentions a person/hand/actor but physical_entities does not include a corresponding actor entity"
        )

    observations = _require_object(parsed["modality_observations"], "modality_observations")
    expected = set(task.modalities)
    actual = set(observations)
    if actual != expected:
        raise ValueError(f"modality_observations keys must be {sorted(expected)}, got {sorted(actual)}")
    for modality in task.modalities:
        analysis = _require_object(observations.get(modality), f"modality_observations.{modality}")
        detailed_caption = _validate_min_words(
            analysis.get("detailed_caption"),
            f"modality_observations.{modality}.detailed_caption",
            MIN_DETAILED_CAPTION_WORDS,
        )
        _validate_no_generic_sensor_explanation(detailed_caption, f"modality_observations.{modality}.detailed_caption")
        _validate_frame_by_frame_details(
            analysis.get("frame_by_frame_details"),
            f"modality_observations.{modality}.frame_by_frame_details",
            task.sharedframe_indexes,
        )
        for key in ("observable_facts", "sensor_specific_cues", "sensor_limitations"):
            values = _require_list(analysis.get(key), f"modality_observations.{modality}.{key}")
            if not values:
                raise ValueError(f"modality_observations.{modality}.{key} must not be empty")
            if key == "observable_facts" and len(values) < MIN_OBSERVABLE_FACTS:
                raise ValueError(
                    f"modality_observations.{modality}.{key} must contain at least {MIN_OBSERVABLE_FACTS} facts"
                )
            for fact_index, value in enumerate(values, start=1):
                text = _require_string(value, f"modality_observations.{modality}.{key}[{fact_index}]")
                _validate_no_generic_sensor_explanation(text, f"modality_observations.{modality}.{key}[{fact_index}]")
        _validate_missing_key_attributes(
            analysis.get("missing_key_attributes"),
            f"modality_observations.{modality}.missing_key_attributes",
            expected,
        )
    _require_list(parsed["rejected_modalities"], "rejected_modalities")
    return parsed


async def _call_gemini_global_evidence(
    client,
    task: GlobalEvidenceTask,
    model_name: str,
    max_retries: int,
) -> dict[str, Any]:
    contact_sheet = _ensure_contact_sheet(task)
    contents = build_image_parts([_encode_image(contact_sheet)]) + [_build_global_prompt(task)]
    for attempt in range(1, max_retries + 1):
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=model_name,
                contents=contents,
            )
            return _validate_global_evidence_schema(_parse_json_response(response.text), task)
        except Exception as exc:
            if attempt == max_retries:
                raise
            wait_seconds = 30 * attempt if "429" in str(exc) or "quota" in str(exc).lower() else 2 * attempt
            print(
                f"    Global evidence Gemini call failed on attempt {attempt}/{max_retries}; "
                f"retrying in {wait_seconds}s: {exc}"
            )
            await asyncio.sleep(wait_seconds)
    raise RuntimeError("Gemini global evidence call failed")


def _template_global_evidence(task: GlobalEvidenceTask) -> dict[str, Any]:
    return {
        "schema_version": GLOBAL_EVIDENCE_SCHEMA_VERSION,
        "global_scene": {
            "scene_summary": "Template mode placeholder; Gemini was not called.",
            "physical_entities": [
                {
                    "entity_id": "unresolved_target",
                    "category": "unknown",
                    "appearance_or_state": "Template mode placeholder.",
                    "location": "unknown",
                    "motion_or_action": "unknown",
                    "spatial_relations": [],
                }
            ],
            "environment": "unknown",
            "temporal_progression": "Template mode placeholder; no visual reasoning was performed.",
            "frame_by_frame_details": {
                f"frame_{index:06d}": "Template mode placeholder; no frame-level visual reasoning was performed."
                for index in task.sharedframe_indexes
            },
        },
        "modality_observations": {
            modality: {
                "detailed_caption": "Template mode placeholder; Gemini was not called.",
                "frame_by_frame_details": {
                    f"frame_{index:06d}": "Template mode placeholder; no frame-level visual reasoning was performed."
                    for index in task.sharedframe_indexes
                },
                "observable_facts": ["Template mode placeholder."],
                "sensor_specific_cues": ["Template mode placeholder."],
                "sensor_limitations": ["Template mode placeholder."],
                "missing_key_attributes": [],
            }
            for modality in task.modalities
        },
        "rejected_modalities": [],
    }


def _task_to_item(
    task: GlobalEvidenceTask,
    status: str,
    evidence: dict[str, Any] | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "evidence_id": task.evidence_id,
        "segment_id": task.segment_id,
        "split_dir": task.split_dir,
        "segment_name": task.segment_name,
        "side": task.side,
        "modalities": list(task.modalities),
        "frame_dirs": {modality: path.as_posix() for modality, path in task.frame_dirs.items()},
        "frames_by_modality": {
            modality: [path.as_posix() for path in frames]
            for modality, frames in task.frames_by_modality.items()
        },
        "sharedframe_indexes": list(task.sharedframe_indexes),
        "contact_sheet": task.contact_sheet.as_posix(),
        "status": status,
        "reason": reason,
        "evidence": evidence,
    }


def _build_output_payload(
    input_path: Path,
    dataset_root: Path,
    contact_sheet_root: Path,
    output_path: Path,
    model_name: str,
    generation_mode: str,
    num_frames: int,
    sampling_strategy: str,
    items: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    planned_total: int,
    gemini_calls: int,
) -> dict[str, Any]:
    return {
        "metadata": {
            "task": "aligned_global_evidence_generation",
            "schema_version": GLOBAL_EVIDENCE_SCHEMA_VERSION,
            "input": input_path.as_posix(),
            "output": output_path.as_posix(),
            "dataset_root": dataset_root.as_posix(),
            "contact_sheet_root": contact_sheet_root.as_posix(),
            "generation_mode": generation_mode,
            "model_name": model_name,
            "num_frames": num_frames,
            "sampling_strategy": sampling_strategy,
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
        print(f"WARNING: Could not load existing global evidence output for resume: {exc}")
        return [], []
    items = data.get("items") if isinstance(data.get("items"), list) else []
    skipped = data.get("skipped") if isinstance(data.get("skipped"), list) else []
    return list(items), list(skipped)


async def run_global_evidence_pipeline_async(
    input_path: Path,
    output_path: Path,
    dataset_root: Path,
    contact_sheet_root: Path,
    model_name: str,
    generation_mode: str,
    num_frames: int,
    sampling_strategy: str,
    limit: int | None,
    limit_scenes: int | None,
    limit_scene_folders: int | None,
    max_retries: int,
    delay_between_calls: int,
    checkpoint_every: int,
    resume: bool,
) -> Path:
    tasks, skipped = build_global_evidence_tasks(
        input_path=input_path,
        dataset_root=dataset_root,
        contact_sheet_root=contact_sheet_root,
        num_frames=num_frames,
        sampling_strategy=sampling_strategy,
        limit=limit,
        limit_scenes=limit_scenes,
        limit_scene_folders=limit_scene_folders,
        write_contact_sheets=False,
    )
    existing_items, existing_skipped = _load_resume(output_path) if resume else ([], [])
    items = existing_items
    skipped = existing_skipped + skipped
    existing_ids = {str(item.get("evidence_id")) for item in items if item.get("evidence_id")}
    pending_tasks = [task for task in tasks if task.evidence_id not in existing_ids]

    client = create_gemini_client() if generation_mode == "gemini" else None
    gemini_calls = 0
    checkpoint_counter = 0

    print(
        f"Generating global evidence: {len(tasks)} planned item(s), "
        f"{len(pending_tasks)} pending, mode={generation_mode}, model={model_name}."
    )

    def save_checkpoint() -> None:
        _save_json(
            _build_output_payload(
                input_path=input_path,
                dataset_root=dataset_root,
                contact_sheet_root=contact_sheet_root,
                output_path=output_path,
                model_name=model_name,
                generation_mode=generation_mode,
                num_frames=num_frames,
                sampling_strategy=sampling_strategy,
                items=items,
                skipped=skipped,
                planned_total=len(tasks),
                gemini_calls=gemini_calls,
            ),
            output_path,
        )

    for index, task in enumerate(pending_tasks, start=1):
        print(f"  Global evidence [{index}/{len(pending_tasks)}]: {task.evidence_id}")
        try:
            if generation_mode == "gemini":
                assert client is not None
                _ensure_contact_sheet(task)
                evidence = await _call_gemini_global_evidence(client, task, model_name, max_retries=max_retries)
                gemini_calls += 1
                status = "generated"
            else:
                _ensure_contact_sheet(task)
                evidence = _template_global_evidence(task)
                status = "template"
            items.append(_task_to_item(task, status=status, evidence=evidence))
        except Exception as exc:
            skipped.append(
                {
                    "evidence_id": task.evidence_id,
                    "segment_id": task.segment_id,
                    "side": task.side,
                    "modalities": list(task.modalities),
                    "reason": str(exc),
                }
            )
            print(f"WARNING: Global evidence generation failed for {task.evidence_id}: {exc}")

        checkpoint_counter += 1
        if checkpoint_every > 0 and checkpoint_counter >= checkpoint_every:
            checkpoint_counter = 0
            save_checkpoint()
        if generation_mode == "gemini" and delay_between_calls > 0 and index < len(pending_tasks):
            await asyncio.sleep(delay_between_calls)

    save_checkpoint()
    print(f"Wrote global evidence output to {output_path}")
    return output_path


def run_global_evidence_pipeline(
    input_path: Path | str = DEFAULT_INPUT_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    dataset_root: Path | str = DEFAULT_DATASET_ROOT,
    contact_sheet_root: Path | str = DEFAULT_CONTACT_SHEET_ROOT,
    model_name: str = DEFAULT_MODEL_NAME,
    generation_mode: str = "template",
    num_frames: int = 6,
    sampling_strategy: str = "uniform",
    limit: int | None = None,
    limit_scenes: int | None = None,
    limit_scene_folders: int | None = None,
    max_retries: int = 3,
    delay_between_calls: int = 5,
    checkpoint_every: int = 1,
    resume: bool = True,
) -> Path:
    return asyncio.run(
        run_global_evidence_pipeline_async(
            input_path=Path(input_path),
            output_path=Path(output_path),
            dataset_root=Path(dataset_root),
            contact_sheet_root=Path(contact_sheet_root),
            model_name=model_name,
            generation_mode=generation_mode,
            num_frames=num_frames,
            sampling_strategy=sampling_strategy,
            limit=limit,
            limit_scenes=limit_scenes,
            limit_scene_folders=limit_scene_folders,
            max_retries=max_retries,
            delay_between_calls=delay_between_calls,
            checkpoint_every=checkpoint_every,
            resume=resume,
        )
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--contact-sheet-root", default=str(DEFAULT_CONTACT_SHEET_ROOT))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--generation-mode",
        choices=("template", "gemini"),
        default="template",
        help="Use template to build contact sheets without calling Gemini.",
    )
    parser.add_argument("--num-frames", type=int, default=6)
    parser.add_argument(
        "--sampling-strategy",
        choices=SAMPLING_STRATEGIES,
        default="uniform",
        help=(
            "Frame selection strategy. uniform preserves the old behavior; "
            "event_activity keeps endpoints and highest Event activity frames; "
            "hybrid_event combines endpoints, uniform coverage, and Event activity peaks."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--limit-scenes",
        "--limit-segments",
        dest="limit_scenes",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--limit-scene-folders",
        "--limit-split-dirs",
        dest="limit_scene_folders",
        type=int,
        default=None,
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--delay-between-calls", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    run_global_evidence_pipeline(
        input_path=args.input,
        output_path=args.output,
        dataset_root=args.dataset_root,
        contact_sheet_root=args.contact_sheet_root,
        model_name=args.model_name,
        generation_mode=args.generation_mode,
        num_frames=max(1, args.num_frames),
        sampling_strategy=args.sampling_strategy,
        limit=args.limit,
        limit_scenes=args.limit_scenes,
        limit_scene_folders=args.limit_scene_folders,
        max_retries=max(1, args.max_retries),
        delay_between_calls=max(0, args.delay_between_calls),
        checkpoint_every=max(0, args.checkpoint_every),
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()


