"""Generate cross-modal disambiguation captions from aligned frame caches."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import hashlib
import os
import re
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from PIL import Image, ImageDraw, ImageFont

try:
    from google.genai import types as genai_types
except ImportError:
    genai_types = None

from annotation_feature.pipeline.client import create_gemini_client
from annotation_feature.pipeline.utils import build_image_parts, infer_recording_side

from annotation_feature.aligned_multimodal_sampling import (
    MultimodalSamplingJob,
    load_frame_dirs,
    pair_frame_dirs,
    select_paired_frames,
    frame_index,
    frames_by_index,
    ADAPTIVE_CHANGE_WEIGHT,
    ADAPTIVE_COVERAGE_WEIGHT
)

SELECTION_ALGORITHM_VERSION = "uniform_adaptive_v1"

def build_selection_config_fingerprint(
    num_uniform_frames: int,
    num_adaptive_frames: int,
) -> str:
    payload = {
        "algorithm": "uniform_adaptive",
        "algorithm_version": SELECTION_ALGORITHM_VERSION,
        "num_uniform_frames": num_uniform_frames,
        "num_adaptive_frames": num_adaptive_frames,
        "adaptive_change_weight": ADAPTIVE_CHANGE_WEIGHT,
        "adaptive_coverage_weight": ADAPTIVE_COVERAGE_WEIGHT,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


DEFAULT_INPUT_PATH = Path("outputs/aligned_multimodal_visual_evidence_units_filtered.json")
DEFAULT_OUTPUT_PATH = Path("outputs/aligned_cross_modal_disambiguation_captions_gemini.json")
DEFAULT_COMPOSITE_ROOT = Path("outputs/composite_frames")
DEFAULT_DATASET_ROOT = Path("aligned_dataset")
DEFAULT_MODEL_NAME = "gemini-3.1-flash-lite"
CAPTION_SCHEMA_VERSION = "cross_modal_disambiguation_caption_v7"
ALLOWED_MISSING_ATTRIBUTE_TYPES = {
    "existence",
    "target_category",
    "spatial_distance",
    "surface_attribute",
    "motion_trend",
}
ALLOWED_GAIN_RATINGS = {"low", "medium", "high"}
ALLOWED_FOCUS_REASONS = {
    "cross_modal_complementarity", "fusion_gain", "temporal_change", 
    "interaction", "occlusion_change", "spatial_transition", "joint_fusion", "ambiguity_resolution"
}
FOCUS_REASON_SECTION_MAP = {
    "cross_modal_complementarity": {"cross_modal_evidence_links"},
    "fusion_gain": {"information_gain"},
    "temporal_change": {"reasoning_events"},
    "interaction": {"reasoning_events"},
    "occlusion_change": {"reasoning_events"},
    "spatial_transition": {"reasoning_events"},
    "joint_fusion": {"reasoning_events", "information_gain"},
    "ambiguity_resolution": {"ambiguity_events"}
}
FOCUS_REASON_EVENT_TYPE_MAP = {
    "temporal_change": {"temporal_change"},
    "interaction": {"interaction"},
    "occlusion_change": {"occlusion_change"},
    "spatial_transition": {"spatial_transition"},
    "joint_fusion": {"joint_fusion"},
}
ALLOWED_QA_REASONING_PATTERNS = {
    "cross_modal_disambiguation", "temporal_integration", "occlusion_reasoning", 
    "interaction_reasoning", "spatial_transition", "hypothesis_elimination", 
    "multi_hop_composition", "joint_fusion"
}
ALLOWED_REASONING_EVENT_TYPES = {
    "temporal_change",
    "interaction",
    "occlusion_change",
    "spatial_transition",
    "joint_fusion",
}
ALLOWED_AMBIGUITY_DIRECTIONS = {"video1_resolves_video2", "video2_resolves_video1"}
FORBIDDEN_GLOBAL_SCENE_WORDS = re.compile(
    r"\b(modality|thermal|rgb|event|depth|infrared|ir|visible|invisible|"
    r"blurry|noisy|pixels?|grayscale|greyscale|heat|edge|edge-based|sparse|contrast|"
    r"monochrome|overexposed|saturated|contour|silhouette)\b", re.I
)
MIN_DETAILED_CAPTION_WORDS = 30
MIN_SCENE_SUMMARY_WORDS = 20
MIN_FRAME_DETAIL_WORDS = 8
GENERIC_SENSOR_EXPLANATION_PATTERNS = (
    re.compile(r"\bevent cameras?\s+(capture|detect|record|respond)", re.I),
    re.compile(r"\b(depth|rgb|infrared|ir)\s+(camera|sensor)s?\s+(capture|detect|record|measure)", re.I),
    re.compile(r"\bthis modality\s+(captures|detects|records|measures)", re.I),
    re.compile(r"\bdesigned to\s+(capture|detect|record|measure)", re.I),
)
VISUAL_PAIRS = (
    ("rgb", "event"),
    ("rgb", "depth"),
    ("rgb", "ir"),
    ("event", "ir"),
    ("event", "depth"),
)
MODALITY_CAPABILITIES = {
    "rgb":   {"color": True,  "thermal": False, "structure_edge": False, "depth": False},
    "event": {"color": False, "thermal": False, "structure_edge": True,  "depth": False},
    "ir":    {"color": False, "thermal": True,  "structure_edge": False, "depth": False},
    "depth": {"color": False, "thermal": False, "structure_edge": False, "depth": True},
}

def build_modality_constraint_block(mod1: str, mod2: str) -> str:
    h = MODALITY_CAPABILITIES.get(mod1, {"color": True, "thermal": True, "structure_edge": True, "depth": True})
    v = MODALITY_CAPABILITIES.get(mod2, {"color": True, "thermal": True, "structure_edge": True, "depth": True})
    
    lines = []
    for attr, cap_name in [
        ("color/paint", "color"), 
        ("thermal/heat", "thermal"), 
        ("structural edges/motion boundaries", "structure_edge"), 
        ("metric depth/distance", "depth")
    ]:
        h_can = h[cap_name]
        v_can = v[cap_name]
        if not h_can and not v_can:
            lines.append(f"- {attr}: NEITHER modality can provide this. Do NOT list it as recoverable_evidence_refs from either video.")
        elif h_can and not v_can:
            lines.append(f"- {attr}: Only video1 ({mod1}) can provide this.")
        elif not h_can and v_can:
            lines.append(f"- {attr}: Only video2 ({mod2}) can provide this.")
        else:
            lines.append(f"- {attr}: Both modalities can provide this.")
    
    return "MODALITY CAPABILITY CONSTRAINTS:\n" + "\n".join(lines)

def _normalize_license_plates(data: Any) -> Any:
    if isinstance(data, str):
        pattern = r'\b([A-Z]{1,3})[\s\-]+([A-Z]{1,2})[\s\-]*(\d{1,4})\b'
        return re.sub(pattern, r'\1-\2 \3', data)
    elif isinstance(data, list):
        return [_normalize_license_plates(item) for item in data]
    elif isinstance(data, dict):
        return {k: _normalize_license_plates(v) for k, v in data.items()}
    return data


@dataclass(frozen=True)
class CaptionTask:
    caption_id: str
    segment_id: str
    split_dir: str
    segment_name: str
    side: str
    modality1: str
    modality2: str
    frame_dir1: Path
    frame_dir2: Path
    frames1: tuple[Path, ...]
    frames2: tuple[Path, ...]
    composite_frames: tuple[Path, ...]
    sampling_strategy: str
    uniform_anchor_indexes: tuple[int, ...]
    adaptive_frame_indexes: tuple[int, ...]
    selected_frame_indexes: tuple[int, ...]
    candidate_frame_indexes: tuple[int, ...]
    selection_config_fingerprint: str





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


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")


_FONT_CACHE: dict[int, ImageFont.ImageFont] = {}


def _load_font(size: int = 28) -> ImageFont.ImageFont:
    if size not in _FONT_CACHE:
        try:
            _FONT_CACHE[size] = ImageFont.truetype("arial.ttf", size=size)
        except OSError:
            _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def _resize_to_height(image: Image.Image, height: int) -> Image.Image:
    if image.height == height:
        return image
    width = max(1, round(image.width * height / image.height))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _draw_label(image: Image.Image, label: str, xy: tuple[int, int]) -> None:
    draw = ImageDraw.Draw(image)
    font = _load_font()
    x, y = xy
    padding = 10
    bbox = draw.textbbox((x, y), label, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    draw.rectangle(
        (x, y, x + width + padding * 2, y + height + padding * 2),
        fill=(0, 0, 0),
    )
    draw.text((x + padding, y + padding), label, fill=(255, 255, 255), font=font)


def _compose_frame(
    frame1: Path,
    frame2: Path,
    modality1: str,
    modality2: str,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(frame1) as left_raw, Image.open(frame2) as right_raw:
        left = left_raw.convert("RGB")
        right = right_raw.convert("RGB")
        target_height = min(left.height, right.height)
        left = _resize_to_height(left, target_height)
        right = _resize_to_height(right, target_height)
        canvas = Image.new("RGB", (left.width + right.width, target_height), (0, 0, 0))
        canvas.paste(left, (0, 0))
        canvas.paste(right, (left.width, 0))
        _draw_label(canvas, f"LEFT: {modality1.upper()}", (0, 0))
        _draw_label(canvas, f"RIGHT: {modality2.upper()}", (left.width, 0))
        canvas.save(output_path)
    return output_path


def _ensure_composite_frame(
    frame1: Path,
    frame2: Path,
    modality1: str,
    modality2: str,
    output_path: Path,
) -> Path:
    if not output_path.exists():
        _compose_frame(
            frame1,
            frame2,
            modality1,
            modality2,
            output_path,
        )
    return output_path


def _ensure_composite_frames(task: CaptionTask) -> None:
    for frame1, frame2, composite_frame in zip(
        task.frames1,
        task.frames2,
        task.composite_frames,
    ):
        _ensure_composite_frame(
            frame1,
            frame2,
            task.modality1,
            task.modality2,
            composite_frame,
        )


def _caption_id(segment_id: str, side: str, modality1: str, modality2: str) -> str:
    return "__".join(
        [
            _safe_name(segment_id).lower(),
            _safe_name(side).lower(),
            modality1,
            modality2,
        ]
    )


def _parse_sides(values: str | None) -> set[str] | None:
    if not values:
        return None
    return {val.strip().lower() for val in values.split(",") if val.strip()}


def _parse_pairs(values: str | None) -> set[tuple[str, str]] | None:
    if not values:
        return None
    pairs: set[tuple[str, str]] = set()
    for chunk in values.split(","):
        token = chunk.strip().lower()
        if not token:
            continue
        if "+" in token:
            first, second = token.split("+", 1)
        elif "->" in token:
            first, second = token.split("->", 1)
        else:
            raise ValueError(f"Invalid pair/direction token: {chunk}")
        pairs.add((first.strip(), second.strip()))
    return pairs


def _directions_for_pair(pair: list[str] | tuple[str, str], allowed: set[tuple[str, str]] | None) -> list[tuple[str, str]]:
    first, second = str(pair[0]).lower(), str(pair[1]).lower()
    # v5 schema captures bidirectional reasoning in a single call (video1 helps video2 AND
    # video2 helps video1 are both present in the output). So by default we only generate one
    # canonical task per pair (the ordering from the input data) to avoid calling Gemini twice
    # on the same pair. Use --directions to explicitly override the ordering.
    if allowed is None:
        return [(first, second)]
    return [(d_first, d_second) for d_first, d_second in [(first, second), (second, first)] if (d_first, d_second) in allowed]


def build_caption_tasks(
    input_path: Path,
    dataset_root: Path,
    composite_root: Path,
    sampling_strategy: str,
    num_uniform_frames: int,
    num_adaptive_frames: int,
    existing_items: list[dict[str, Any]],
    allowed_pairs: set[tuple[str, str]] | None = None,
    allowed_directions: set[tuple[str, str]] | None = None,
    allowed_sides: set[str] | None = None,
    limit: int | None = None,
    limit_scenes: int | None = None,
    limit_scene_folders: int | None = None,
    write_composites: bool = True,
) -> tuple[list[CaptionTask], list[dict[str, Any]], int]:
    data = _load_json(input_path)
    segments = data.get("segments")
    if not isinstance(segments, dict):
        raise ValueError(f"Expected {input_path} to contain a segments object")

    skipped: list[dict[str, Any]] = []
    all_jobs: list[MultimodalSamplingJob] = []
    
    for segment_id, segment in sorted(segments.items()):
        if not isinstance(segment, dict):
            continue
            
        split_dir = str(segment.get("split_dir") or "")
        segment_name = str(segment.get("segment_name") or "")
        if not split_dir or not segment_name:
            skipped.append({"segment_id": segment_id, "reason": "missing split_dir or segment_name"})
            continue
            
        pairs = segment.get("modality_pairs") or []
        for pair in pairs:
            if not isinstance(pair, list | tuple) or len(pair) != 2:
                continue
            
            pair_tuple = (str(pair[0]).lower(), str(pair[1]).lower())
            if allowed_pairs is not None and pair_tuple not in allowed_pairs and pair_tuple[::-1] not in allowed_pairs:
                continue
            pair_directions = _directions_for_pair(pair, allowed_directions)
                
            for modality1, modality2 in pair_directions:
                dirs1 = load_frame_dirs(dataset_root, split_dir, segment_name, modality1)
                dirs2 = load_frame_dirs(dataset_root, split_dir, segment_name, modality2)
                if not dirs1 or not dirs2:
                    skipped.append({
                        "segment_id": segment_id, "split_dir": split_dir, "segment_name": segment_name,
                        "modality1": modality1, "modality2": modality2, "reason": "missing frame cache directory",
                    })
                    continue
                for side, dir1, dir2 in pair_frame_dirs(dirs1, dirs2):
                    if allowed_sides is not None and side.lower() not in allowed_sides:
                        continue
                        
                    by_index1 = frames_by_index(dir1, modality1)
                    by_index2 = frames_by_index(dir2, modality2)
                    shared_indexes = tuple(sorted(set(by_index1) & set(by_index2)))
                    
                    if not shared_indexes:
                        skipped.append({
                            "segment_id": segment_id, "side": side, "modality1": modality1, "modality2": modality2,
                            "frame_dir1": dir1.as_posix(), "frame_dir2": dir2.as_posix(), "reason": "no shared frame indexes",
                        })
                        continue
                    
                    if len(shared_indexes) != 30:
                        skipped.append({
                            "segment_id": segment_id, "side": side, "modality1": modality1, "modality2": modality2,
                            "frame_dir1": dir1.as_posix(), "frame_dir2": dir2.as_posix(), "reason": f"expected 30 frames, got {len(shared_indexes)}",
                        })
                        continue

                    all_jobs.append(MultimodalSamplingJob(
                        segment_id=segment_id, split_dir=split_dir, segment_name=segment_name,
                        side=side, modality1=modality1, modality2=modality2, dir1=dir1, dir2=dir2,
                        shared_indexes=shared_indexes, by_index1=by_index1, by_index2=by_index2,
                    ))

    selection_config_fingerprint = build_selection_config_fingerprint(num_uniform_frames, num_adaptive_frames)

    completed_keys = {
        (item.get("caption_id"), item.get("selection_config_fingerprint"))
        for item in existing_items
    }

    selected_jobs = []
    selected_scenes = set()
    selected_scene_folders = set()
    
    for job in all_jobs:
        if limit_scenes is not None and len(selected_scenes) >= limit_scenes and job.segment_id not in selected_scenes:
            continue
        if limit_scene_folders is not None and len(selected_scene_folders) >= limit_scene_folders and job.split_dir not in selected_scene_folders:
            continue
            
        selected_jobs.append(job)
        selected_scenes.add(job.segment_id)
        selected_scene_folders.add(job.split_dir)

    if limit is not None:
        selected_jobs = selected_jobs[:limit]

    total_selected_jobs = len(selected_jobs)

    pending_jobs = []
    for job in selected_jobs:
        caption_id = _caption_id(str(job.segment_id), job.side, job.modality1, job.modality2)
        if (caption_id, selection_config_fingerprint) not in completed_keys:
            pending_jobs.append(job)



    tasks: list[CaptionTask] = []
    
    for job in pending_jobs:
        frames1, frames2, strategy, anchors, adaptive, selected = select_paired_frames(
            job.dir1,
            job.dir2,
            job.modality1,
            job.modality2,
            sampling_strategy,
            num_uniform_frames,
            num_adaptive_frames,
        )

        caption_id = _caption_id(str(job.segment_id), job.side, job.modality1, job.modality2)
        output_dir = (
            composite_root
            / _safe_name(job.split_dir)
            / _safe_name(job.segment_name)
            / _safe_name(job.side)
            / f"{job.modality1}__{job.modality2}"
        )
        composite_frames: list[Path] = []
        for index, (f1, f2) in enumerate(zip(frames1, frames2), start=1):
            frame_number = frame_index(f1)
            suffix = f"{frame_number:06d}" if frame_number is not None else f"{index:03d}"
            output_path = output_dir / f"frame_{suffix}.png"
            if write_composites:
                _compose_frame(
                    f1,
                    f2,
                    job.modality1,
                    job.modality2,
                    output_path,
                )
            composite_frames.append(output_path)
            
        tasks.append(
            CaptionTask(
                caption_id=caption_id,
                segment_id=str(job.segment_id),
                split_dir=job.split_dir,
                segment_name=job.segment_name,
                side=job.side,
                modality1=job.modality1,
                modality2=job.modality2,
                frame_dir1=job.dir1,
                frame_dir2=job.dir2,
                frames1=frames1,
                frames2=frames2,
                composite_frames=tuple(composite_frames),
                sampling_strategy=strategy,
                uniform_anchor_indexes=tuple(anchors),
                adaptive_frame_indexes=tuple(adaptive),
                selected_frame_indexes=tuple(selected),
                candidate_frame_indexes=tuple(job.shared_indexes),
                selection_config_fingerprint=selection_config_fingerprint,
            )
        )
        
    return tasks, skipped, total_selected_jobs


def _build_caption_prompt(task: CaptionTask) -> str:
    frame_keys = ", ".join(f'"{path.stem}"' for path in task.composite_frames)
    frame_names = ", ".join(path.name for path in task.composite_frames)
    example_frame_key = task.composite_frames[0].stem if task.composite_frames else "frame_key_from_valid_list"
    constraint_block = build_modality_constraint_block(task.modality1, task.modality2)
    return "\n".join(
        [
            "You are an expert multimodal perception analyst.",
            "You will receive multiple synchronized composite frames sampled from one aligned video segment.",
            f"Video 1 (left): {task.modality1}.",
            f"Video 2 (right): {task.modality2}.",
            "These two videos observe the same physical scene using different sensing modalities.",
            constraint_block,
            "Neither video is considered the reference or the ground truth.",
            "The goal is not ordinary captioning. Build a dense bidirectional multimodal evidence graph that maximizes reasoning-relevant information.",
            "Only use evidence directly observable in the supplied frames. Do not invent objects, future events, intentions, identities, unreadable text, or unsupported actions.",
            "Always distinguish between physical reality, video observations, and reasoning uncertainty. Do not mix these concepts.",
            "CRITICAL RULE for global_scene: You must describe the physical world as if you are standing there. NEVER mention the camera, the sensor type, or image quality artifacts. Do NOT use ANY of the following words (case-insensitive, including plural forms) in global_scene.scene_summary or global_scene.temporal_progression: modality, thermal, rgb, event, depth, infrared, ir, visible, invisible, blurry, noisy, pixel, pixels, grayscale, heat, edge, sparse, contrast, monochrome, overexposed, saturated, contour, silhouette. This is an exact blocklist, not a suggestion list. Any match causes rejection. The global_scene.scene_summary must be a detailed paragraph covering: which entities are present and their appearance, their spatial layout, the environment/setting, and any ongoing actions. Do NOT write a single sentence — write a full descriptive paragraph. Trace the scene chronologically. global_scene.temporal_progression must also strictly follow the blocklist.",            "ENTITY SELECTION: physical_entities should include entities central to the scene action or where modalities differ.",
            "DEEP REASONING ANALYSIS: When analyzing the scene, you MUST follow these paradigms to support difficult QA generation: (1) Information Atoms: Must contain directly observable, source-local facts. Each atom should express one minimal factual claim grounded in its referenced frames. Do not place intentions, causal explanations, fusion conclusions, or multi-step inferences inside atoms; those belong in reasoning_events or ambiguity_events. (2) Visibility & Occlusion: Track entity occlusion states chronologically. (3) Interaction Graph: Build human-object and object-object causality. (4) QA-Relevant Details: Focus on non-obvious discriminative features that require cross-modal thinking. (5) UNCERTAINTY HONESTY GUARDRAIL: Do NOT hallucinate physical attributes. Honestly record uncertainty. Do not guess.",
            "FIELD RULES (violations cause rejection):",
            "1. GLOBAL NAMESPACES & REFERENCE IDs: All referenceable structures must use exact ID prefixes and be globally unique across the entire JSON to prevent collisions. information_atoms must use 'v1_atom_' or 'v2_atom_'. reasoning_events must use 'evt_'. ambiguity_events must use 'amb_'. qa_relevant_details must use 'qa_detail_'.",
            "2. SINGLE PROVENANCE TRUTH: Information atoms are the ONLY structures that contain frame_keys. reasoning_events and ambiguity_events must point strictly to atom IDs to indicate their frame source.",
            "3. NO SELF-REFERENCE: qa_relevant_details.supporting_refs MUST NOT reference another qa_detail. It can only reference v1_atom_, v2_atom_, evt_, and amb_.",
            "4. MISSING KEY ATTRIBUTES: The recoverable_evidence_refs list MUST ONLY contain atom IDs, and it must not be empty. If an attribute is missing from video1, its recovery MUST reference at least one v2_atom. If missing from video2, it MUST reference at least one v1_atom.",
            "5. CONDITIONAL EVIDENCE PROFILE: evidence_profile fields (identity_evidence, observable_attributes, spatial_context) must be completely omitted from the JSON if there is no meaningful non-dynamic evidence for them. DO NOT return empty lists or empty strings.",
            f"VALID FRAME KEYS: [{frame_keys}]. information_atoms[].frame_keys MUST choose only from these exact values.",
            "6. REASON-DRIVEN COVERAGE: DO NOT force every entity into cross_modal_evidence_links or information_gain. You must only include an entity in a section if its focus_reasons justify it. Static occluders should only be in occlusion_change if their occlusion state actually changes.",
            "7. CRITICAL RULE for detailed_caption & global_scene: Describe the physical world as if you are standing there. NEVER mention the camera, the sensor type, or image quality artifacts. You MUST STRICTLY AVOID these exact words: modality, thermal, rgb, event, depth, infrared, ir, visible, invisible, blurry, noisy, pixel, pixels, grayscale, heat, edge, sparse, contrast, monochrome, overexposed, saturated, contour, silhouette.",
            "8. ambiguity_events[].candidate_hypotheses must include AT LEAST TWO distinct hypotheses — never provide only one.",
            "9. NEVER use generic sensor-theory wording like 'this modality captures' or 'event cameras detect'. Describe ONLY specific evidence from the current frames.",
            "UNCERTAIN OBSERVATIONS: For BOTH videos independently, identify observations that are genuinely ambiguous.",
            "MISSING INFORMATION: List attributes that cannot be determined from each individual video and which cross-modal atoms recover them.",
            "CROSS-MODAL EVIDENCE LINKS: Jointly analyze both videos. For entities where focus_reason includes cross_modal_complementarity, identify shared and unique evidence.",
            "INFORMATION GAIN: For entities where focus_reason includes fusion_gain or joint_fusion, explain what each video can/cannot determine and the fusion reveal.",
            "AMBIGUITY RESOLUTION: Actively search for ambiguities in BOTH directions. You MUST check both directions independently and report any valid events.",
            "REASONING EVENTS: Document dynamic changes (temporal_change, interaction, occlusion_change, spatial_transition, joint_fusion). These MUST be supported by atom references.",
            "QA RELEVANT DETAILS: Document facts that are particularly useful for downstream QA by pointing to the relevant atoms/events/ambiguities.",
            "Return ONLY valid JSON with this exact structure:",
            "{",
            f'  "schema_version": "{CAPTION_SCHEMA_VERSION}",',
            '  "global_scene": {',
            '    "scene_summary": "Detailed physical-scene summary independent of sensor artifacts.",',
            '    "physical_entities": [',
            '      {',
            '        "entity_id": "stable_snake_case_id",',
            '        "category": "...",',
            '        "evidence_profile": {',
            '          "identity_evidence": ["Why do we think it is this category?"],',
            '          "observable_attributes": ["Current observable non-pure-appearance attributes/states (exclude dynamic changes)."],',
            '          "spatial_context": ["Base location or stable contextual placement."]',
            '        }',
            '      }',
            '    ],',
            '    "reasoning_focus_entities": [',
            '      {"entity_id": "pedestrian_1", "focus_reasons": ["temporal_change", "occlusion_change"]}',
            '    ],',
            '    "environment": "Objective environment or recording condition if evident.",',
            '    "temporal_progression": "Dense chronological account of how the scene/action changes across supplied frames."',
            '  },',
            '  "video1_analysis": {',
            f'    "modality": "{task.modality1}",',
            '    "detailed_caption": "Detailed caption using only Video 1 (LEFT).",',
            '    "information_atoms": [',
            f'      {{"atom_id": "v1_atom_001", "frame_keys": ["{example_frame_key}"], "fact": "Discrete atomic observation. Must be a direct factual claim, NO inferences or causality."}}',
            '    ],',
            '    "sensor_specific_cues": ["Imaging/measurement cues from this modality."],',
            '    "sensor_limitations": ["Specific limitations that affect interpretation."],',
            '    "uncertain_observations": [{"observed_evidence": "...", "hypotheses": [{"hypothesis": "...", "confidence": "low"}], "missing_evidence": "..."}],',
            '    "missing_key_attributes": [{"attribute_type": "existence", "missing_attribute": "...", "why_missing": "...", "recoverable_evidence_refs": ["v2_atom_001"]}]',
            '  },',
            '  "video2_analysis": {',
            f'    "modality": "{task.modality2}",',
            '    "detailed_caption": "Detailed caption using only Video 2 (RIGHT).",',
            '    "information_atoms": [',
            f'      {{"atom_id": "v2_atom_001", "frame_keys": ["{example_frame_key}"], "fact": "Discrete atomic observation. Must be a direct factual claim, NO inferences or causality."}}',
            '    ],',
            '    "sensor_specific_cues": ["Imaging/measurement cues from this modality."],',
            '    "sensor_limitations": ["Specific limitations that affect interpretation."],',
            '    "uncertain_observations": [{"observed_evidence": "...", "hypotheses": [{"hypothesis": "...", "confidence": "low"}], "missing_evidence": "..."}],',
            '    "missing_key_attributes": [{"attribute_type": "existence", "missing_attribute": "...", "why_missing": "...", "recoverable_evidence_refs": ["v1_atom_001"]}]',
            '  },',
            '  "cross_modal_evidence_links": [',
            '    {"entity_id": "...", "shared_evidence": "...", "unique_to_video1": "...", "unique_to_video2": "...", "how_video1_improves_video2": "...", "how_video2_improves_video1": "..."}',
            '  ],',
            '  "information_gain": [',
            '    {"entity_id": "...", "gain_rating": "low", "video1_can_determine": ["..."], "video1_cannot_determine": ["..."], "video2_can_determine": ["..."], "video2_cannot_determine": ["..."], "fusion_additionally_reveals": ["..."]}',
            '  ],',
            '  "reasoning_events": [',
            '    {',
            '      "event_id": "evt_001",',
            '      "event_type": "temporal_change",',
            '      "participating_entities": ["entity_id"],',
            '      "supporting_atom_refs": ["v1_atom_001", "v2_atom_001"],',
            '      "description": "Non-trivial inference or dynamic behavior based on the supporting atoms."',
            '    }',
            '  ],',
            '  "ambiguity_events": [',
            '    {',
            '      "ambiguity_id": "amb_001",',
            '      "target_entity": "entity_id",',
            '      "direction": "video1_resolves_video2",',
            '      "ambiguous_video": "video2",',
            '      "resolving_video": "video1",',
            '      "low_confidence_observation": "What the ambiguous video shows by itself.",',
            '      "why_ambiguous_video_cannot_resolve": "Specific reason the ambiguous video cannot uniquely interpret the cue.",',
            '      "candidate_hypotheses": [{"hypothesis": "hypothesis 1...", "support_from_resolving": "..."}, {"hypothesis": "hypothesis 2...", "support_from_resolving": "..."}],',
            '      "resolving_discriminative_evidence": "Concrete cue from the resolving video that eliminates at least one hypothesis.",',
            '      "eliminated_hypotheses": [{"hypothesis": "...", "why_eliminated": "..."}],',
            '      "fusion_conclusion": "Final physical fact after combining both modalities.",',
            '      "missing_attribute_type": "existence",',
            '      "ambiguous_evidence_refs": ["v1_atom_001"],',
            '      "resolving_evidence_refs": ["v2_atom_001"]',
            '    }',
            '  ],',
            '  "qa_relevant_details": [',
            '    {',
            '      "detail_id": "qa_detail_001",',
            '      "reasoning_pattern": "cross_modal_disambiguation",',
            '      "supporting_refs": ["evt_001", "amb_001"],',
            '      "why_question_worthy": "Why this grounded fact structure makes a good downstream question."',
            '    }',
            '  ],',
            '  "rejected_observations": [',
            '    {"observation": "...", "reason": "Why this was not a valid ambiguity_event."}',
            '  ]',
            '}',
            "ALLOWED enum values for fields:",
            "- confidence: high, medium, low",
            "- attribute_type / missing_attribute_type: existence, target_category, spatial_distance, surface_attribute, motion_trend",
            "- gain_rating: high, medium, low",
            "- event_type: temporal_change, interaction, occlusion_change, spatial_transition, joint_fusion",
            "- direction: video1_resolves_video2, video2_resolves_video1",
            "- ambiguous_video / resolving_video: video1, video2",
            "- reasoning_pattern: cross_modal_disambiguation, temporal_integration, occlusion_reasoning, interaction_reasoning, spatial_transition, hypothesis_elimination, multi_hop_composition, joint_fusion",
            "FOCUS REASON NOTE: The example reasoning_focus_entities is illustrative only. Select only genuinely supported reasons and never list all enum values merely because they are allowed.",
            "Only include an item in ambiguity_events when one video genuinely disambiguates the other.",
            "If no valid ambiguity_event exists, return an empty ambiguity_events list and explain each rejected case in rejected_observations.",
            f"Segment: {task.segment_id}; side: {task.side}.",
            f"Composite frames ({len(task.composite_frames)} images): {frame_names}",
        ]
    )



def _encode_images(paths: tuple[Path, ...]) -> list[str]:
    encoded: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        with open(path, "rb") as handle:
            encoded.append(base64.standard_b64encode(handle.read()).decode("utf-8"))
    return encoded


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


def _require_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Gemini response field {field} must be an object")
    return value


def _require_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"Gemini response field {field} must be a list")
    return value


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Gemini response field {field} must be a non-empty string")
    return value


def _validate_string_list(value: Any, field: str, *, allow_empty: bool = True) -> list[str]:
    items = _require_list(value, field)
    if not allow_empty and not items:
        raise ValueError(f"{field} must not be empty")
    for i, item in enumerate(items, start=1):
        _require_string(item, f"{field}[{i}]")
    return items


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


def _validate_uncertain_observations(values: Any, field: str) -> None:
    for index, item in enumerate(_require_list(values, field), start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{field}[{index}] must be an object")
        _require_string(item.get("observed_evidence"), f"{field}[{index}].observed_evidence")
        _require_string(item.get("missing_evidence"), f"{field}[{index}].missing_evidence")
        hypotheses = _require_list(item.get("hypotheses"), f"{field}[{index}].hypotheses")
        if len(hypotheses) < 2:
            raise ValueError(f"{field}[{index}].hypotheses must contain at least 2 hypotheses")
        normalized_hyps = {h.get("hypothesis", "").strip().casefold() for h in hypotheses if isinstance(h, dict)}
        if len(normalized_hyps) < 2:
            raise ValueError(f"{field}[{index}].hypotheses must contain at least 2 distinct hypotheses")
        for hyp_index, hyp in enumerate(hypotheses, start=1):
            if not isinstance(hyp, dict):
                raise ValueError(f"{field}[{index}].hypotheses[{hyp_index}] must be an object")
            _require_string(hyp.get("hypothesis"), f"{field}[{index}].hypotheses[{hyp_index}].hypothesis")
            conf = hyp.get("confidence")
            if conf not in ALLOWED_GAIN_RATINGS:
                raise ValueError(f"{field}[{index}].hypotheses[{hyp_index}].confidence must be high, medium, or low")

def _validate_cross_modal_evidence_links(values: Any, entity_ids: set[str], entity_focus_reasons: dict[str, set[str]], field: str) -> None:
    seen_entities = set()
    links = _require_list(values, field)
    for index, item in enumerate(links, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{field}[{index}] must be an object")
        entity_id = _require_string(item.get("entity_id"), f"{field}[{index}].entity_id")
        if entity_id in seen_entities:
            raise ValueError(f"{field} contains duplicate entry for entity_id: {entity_id}")
        seen_entities.add(entity_id)
        if entity_id not in entity_ids:
            raise ValueError(f"{field}[{index}].entity_id must match a global_scene entity_id")
        if "cross_modal_complementarity" not in entity_focus_reasons.get(entity_id, set()):
            raise ValueError(f"{field}[{index}] entity {entity_id} lacks 'cross_modal_complementarity' focus reason")
        for key in ("shared_evidence", "unique_to_video1", "unique_to_video2", "how_video1_improves_video2", "how_video2_improves_video1"):
            _require_string(item.get(key), f"{field}[{index}].{key}")

def _validate_information_gain(values: Any, entity_ids: set[str], entity_focus_reasons: dict[str, set[str]], field: str) -> None:
    seen_entities = set()
    gains = _require_list(values, field)
    for index, item in enumerate(gains, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{field}[{index}] must be an object")
        entity_id = _require_string(item.get("entity_id"), f"{field}[{index}].entity_id")
        if entity_id in seen_entities:
            raise ValueError(f"{field} contains duplicate entry for entity_id: {entity_id}")
        seen_entities.add(entity_id)
        if entity_id not in entity_ids:
            raise ValueError(f"{field}[{index}].entity_id must match a global_scene entity_id")
        reasons = entity_focus_reasons.get(entity_id, set())
        if "fusion_gain" not in reasons and "joint_fusion" not in reasons:
            raise ValueError(f"{field}[{index}].entity_id {entity_id} lacks 'fusion_gain' or 'joint_fusion' focus_reason")
            
        for key in ("video1_can_determine", "video1_cannot_determine", "video2_can_determine", "video2_cannot_determine", "fusion_additionally_reveals"):
            _validate_string_list(item.get(key), f"{field}[{index}].{key}", allow_empty=(key != "fusion_additionally_reveals"))
            
        rating = _require_string(item.get("gain_rating"), f"{field}[{index}].gain_rating")
        if rating not in ALLOWED_GAIN_RATINGS:
            raise ValueError(f"{field}[{index}].gain_rating must be high, medium, or low")

def _validate_caption_schema(parsed: dict[str, Any], valid_frame_keys: set[str], expected_modality1: str, expected_modality2: str) -> dict[str, Any]:
    if not valid_frame_keys:
        raise ValueError("valid_frame_keys must not be empty for validation")

    atom_frame_keys: dict[str, set[str]] = {}

    required_fields = (
        "schema_version",
        "global_scene",
        "video1_analysis",
        "video2_analysis",
        "cross_modal_evidence_links",
        "information_gain",
        "reasoning_events",
        "ambiguity_events",
        "qa_relevant_details",
        "rejected_observations",
    )
    missing = [field for field in required_fields if field not in parsed]
    if missing:
        raise ValueError(f"Gemini response missing required caption field(s): {', '.join(missing)}")
        
    unexpected_fields = set(parsed.keys()) - set(required_fields)
    if unexpected_fields:
        raise ValueError(f"Gemini response contains unknown top-level fields: {', '.join(sorted(unexpected_fields))}. Allowed fields are only: {', '.join(required_fields)}")

    if parsed["schema_version"] != CAPTION_SCHEMA_VERSION:
        raise ValueError(
            f"Gemini response schema_version must be {CAPTION_SCHEMA_VERSION!r}, "
            f"got {parsed['schema_version']!r}"
        )
        
    evidence_namespace: set[str] = set()

    def _register_evidence_id(eid: str) -> None:
        if eid in evidence_namespace:
            raise ValueError(f"Duplicate evidence ID found: {eid}. Evidence IDs must be globally unique.")
        evidence_namespace.add(eid)

    global_scene = _require_object(parsed["global_scene"], "global_scene")
    scene_summary = _validate_min_words(global_scene.get("scene_summary"), "global_scene.scene_summary", MIN_SCENE_SUMMARY_WORDS)
    if FORBIDDEN_GLOBAL_SCENE_WORDS.search(scene_summary):
        raise ValueError("global_scene.scene_summary contains forbidden sensor-quality words. EXACT BLOCKLIST: modality, thermal, rgb, event, depth, infrared, ir, visible, invisible, blurry, noisy, pixel, pixels, grayscale, heat, edge, sparse, contrast, monochrome, overexposed, saturated, contour, silhouette. REMOVE THESE WORDS!")
    _validate_no_generic_sensor_explanation(scene_summary, "global_scene.scene_summary")
    _require_string(global_scene.get("environment"), "global_scene.environment")
    temporal_progression = _validate_min_words(global_scene.get("temporal_progression"), "global_scene.temporal_progression", MIN_FRAME_DETAIL_WORDS)
    if FORBIDDEN_GLOBAL_SCENE_WORDS.search(temporal_progression):
        raise ValueError("global_scene.temporal_progression contains forbidden sensor-quality words. EXACT BLOCKLIST: modality, thermal, rgb, event, depth, infrared, ir, visible, invisible, blurry, noisy, pixel, pixels, grayscale, heat, edge, sparse, contrast, monochrome, overexposed, saturated, contour, silhouette. REMOVE THESE WORDS!")
    _validate_no_generic_sensor_explanation(temporal_progression, "global_scene.temporal_progression")
    
    physical_entities = _require_list(global_scene.get("physical_entities"), "global_scene.physical_entities")
    if not physical_entities:
        raise ValueError("global_scene.physical_entities must not be empty")
    entity_ids: set[str] = set()
    for index, entity in enumerate(physical_entities, start=1):
        if not isinstance(entity, dict):
            raise ValueError(f"global_scene.physical_entities[{index}] must be an object")
        entity_id = _require_string(entity.get("entity_id"), f"global_scene.physical_entities[{index}].entity_id")
        if entity_id in entity_ids:
            raise ValueError(f"Duplicate entity_id: {entity_id}")
        entity_ids.add(entity_id)
        _require_string(entity.get("category"), f"global_scene.physical_entities[{index}].category")
        if "evidence_profile" in entity:
            prof = _require_object(entity.get("evidence_profile"), f"global_scene.physical_entities[{index}].evidence_profile")
            if not prof:
                raise ValueError("evidence_profile must not be empty if present.")
            for prof_key in ("identity_evidence", "observable_attributes", "spatial_context"):
                if prof_key in prof:
                    ev_list = _require_list(prof[prof_key], f"global_scene.physical_entities[{index}].evidence_profile.{prof_key}")
                    if not ev_list:
                        raise ValueError(f"evidence_profile.{prof_key} must not be empty if present.")
                    for j, s in enumerate(ev_list, start=1):
                        _require_string(s, f"evidence_profile.{prof_key}[{j}]")

    focus_entities = _require_list(global_scene.get("reasoning_focus_entities"), "global_scene.reasoning_focus_entities")
    entity_focus_reasons: dict[str, set[str]] = {}
    for index, fe in enumerate(focus_entities, start=1):
        if not isinstance(fe, dict):
            raise ValueError(f"global_scene.reasoning_focus_entities[{index}] must be an object")
        fe_id = _require_string(fe.get("entity_id"), f"global_scene.reasoning_focus_entities[{index}].entity_id")
        if fe_id in entity_focus_reasons:
            raise ValueError(f"Duplicate reasoning_focus_entities entity_id: {fe_id}")
        if fe_id not in entity_ids:
            raise ValueError(f"reasoning_focus_entities[{index}] entity_id {fe_id} does not exist in physical_entities")
        reasons = _require_list(fe.get("focus_reasons"), f"global_scene.reasoning_focus_entities[{index}].focus_reasons")
        if not reasons:
            raise ValueError(f"global_scene.reasoning_focus_entities[{index}].focus_reasons must not be empty")
        reasons_set = set()
        for r in reasons:
            if r not in ALLOWED_FOCUS_REASONS:
                raise ValueError(f"Invalid focus_reason: {r}")
            reasons_set.add(r)
        entity_focus_reasons[fe_id] = reasons_set
        
    actual_section_entities = {
        "cross_modal_evidence_links": set(),
        "information_gain": set(),
        "reasoning_events": set(),
        "ambiguity_events": set()
    }
    for link in parsed.get("cross_modal_evidence_links", []):
        if isinstance(link, dict) and link.get("entity_id"):
            actual_section_entities["cross_modal_evidence_links"].add(link["entity_id"])
    for gain in parsed.get("information_gain", []):
        if isinstance(gain, dict) and gain.get("entity_id"):
            actual_section_entities["information_gain"].add(gain["entity_id"])
    for event in parsed.get("reasoning_events", []):
        if isinstance(event, dict):
            for ent in event.get("participating_entities", []):
                actual_section_entities["reasoning_events"].add(ent)
    for amb in parsed.get("ambiguity_events", []):
        if isinstance(amb, dict) and amb.get("target_entity"):
            actual_section_entities["ambiguity_events"].add(amb["target_entity"])

    for fe_id, reasons in entity_focus_reasons.items():
        expected_sections = set()
        for r in reasons:
            expected_sections.update(FOCUS_REASON_SECTION_MAP.get(r, set()))
        for section in expected_sections:
            if fe_id not in actual_section_entities.get(section, set()):
                raise ValueError(f"Entity {fe_id} declared focus reasons requiring {section}, but it is missing from that section.")

    modality1 = parsed.get("video1_analysis", {}).get("modality", "")
    if modality1 != expected_modality1:
        raise ValueError(f"video1_analysis.modality {modality1!r} does not match expected {expected_modality1!r}")
    modality2 = parsed.get("video2_analysis", {}).get("modality", "")
    if modality2 != expected_modality2:
        raise ValueError(f"video2_analysis.modality {modality2!r} does not match expected {expected_modality2!r}")

    def _validate_v7_video_analysis(parsed: dict[str, Any], field: str, atom_prefix: str) -> None:
        analysis = _require_object(parsed.get(field), field)
        _require_string(analysis.get("modality"), f"{field}.modality")
        detailed_caption = _validate_min_words(
            analysis.get("detailed_caption"),
            f"{field}.detailed_caption",
            MIN_DETAILED_CAPTION_WORDS,
        )
        if FORBIDDEN_GLOBAL_SCENE_WORDS.search(detailed_caption):
            raise ValueError(f"{field}.detailed_caption contains forbidden sensor-quality words. EXACT BLOCKLIST: modality, thermal, rgb, event, depth, infrared, ir, visible, invisible, blurry, noisy, pixel, pixels, grayscale, heat, edge, sparse, contrast, monochrome, overexposed, saturated, contour, silhouette. REMOVE THESE WORDS!")
        _validate_no_generic_sensor_explanation(detailed_caption, f"{field}.detailed_caption")
        
        atoms = _require_list(analysis.get("information_atoms"), f"{field}.information_atoms")
        if not atoms:
            raise ValueError(f"{field}.information_atoms must not be empty")
            
        for i, atom in enumerate(atoms, start=1):
            if not isinstance(atom, dict):
                raise ValueError(f"{field}.information_atoms[{i}] must be an object")
            atom_id = _require_string(atom.get("atom_id"), f"{field}.information_atoms[{i}].atom_id")
            if not atom_id.startswith(atom_prefix):
                raise ValueError(f"atom_id {atom_id} must start with {atom_prefix}")
            _register_evidence_id(atom_id)
            f_keys = _require_list(atom.get("frame_keys"), f"{field}.information_atoms[{i}].frame_keys")
            if not f_keys:
                raise ValueError(f"{field}.information_atoms[{i}].frame_keys cannot be empty")
            for fk in f_keys:
                if fk not in valid_frame_keys:
                    raise ValueError(f"Unknown frame_key '{fk}' in {atom_id}")
            atom_frame_keys[atom_id] = set(f_keys)
            _require_string(atom.get("fact"), f"{field}.information_atoms[{i}].fact")

        for key in ("sensor_specific_cues", "sensor_limitations"):
            values = _require_list(analysis.get(key), f"{field}.{key}")
            for value_index, value in enumerate(values, start=1):
                text = _require_string(value, f"{field}.{key}[{value_index}]")
                _validate_no_generic_sensor_explanation(text, f"{field}.{key}[{value_index}]")
        _validate_uncertain_observations(analysis.get("uncertain_observations"), f"{field}.uncertain_observations")
        
        missing_attrs = _require_list(analysis.get("missing_key_attributes"), f"{field}.missing_key_attributes")
        for i, attr in enumerate(missing_attrs, start=1):
            if not isinstance(attr, dict):
                raise ValueError(f"{field}.missing_key_attributes[{i}] must be an object")
            attr_type = attr.get("attribute_type")
            if attr_type not in ALLOWED_MISSING_ATTRIBUTE_TYPES:
                raise ValueError(f"{field}.missing_key_attributes[{i}].attribute_type invalid: {attr_type}")
            _require_string(attr.get("missing_attribute"), f"{field}.missing_key_attributes[{i}].missing_attribute")
            _require_string(attr.get("why_missing"), f"{field}.missing_key_attributes[{i}].why_missing")
            rec_refs = _require_list(attr.get("recoverable_evidence_refs"), f"{field}.missing_key_attributes[{i}].recoverable_evidence_refs")
            if not rec_refs:
                raise ValueError(f"{field}.missing_key_attributes[{i}].recoverable_evidence_refs must not be empty")
            # Defer cross-modal rule check until all atoms are registered

    _validate_v7_video_analysis(parsed, "video1_analysis", "v1_atom_")
    _validate_v7_video_analysis(parsed, "video2_analysis", "v2_atom_")
    
    _validate_cross_modal_evidence_links(parsed.get("cross_modal_evidence_links"), entity_ids, entity_focus_reasons, "cross_modal_evidence_links")
    _validate_information_gain(parsed.get("information_gain"), entity_ids, entity_focus_reasons, "information_gain")

    events = _require_list(parsed["reasoning_events"], "reasoning_events")
    for index, event in enumerate(events, start=1):
        if not isinstance(event, dict):
            raise ValueError(f"reasoning_events[{index}] must be an object")
        evt_id = _require_string(event.get("event_id"), f"reasoning_events[{index}].event_id")
        if not evt_id.startswith("evt_"):
            raise ValueError(f"reasoning_events[{index}].event_id must start with evt_")
        _register_evidence_id(evt_id)
        evt_type = _require_string(event.get("event_type"), f"reasoning_events[{index}].event_type")
        if evt_type not in ALLOWED_REASONING_EVENT_TYPES:
            raise ValueError(f"reasoning_events[{index}].event_type {evt_type} is not a valid reasoning event type")
        part_ents = _require_list(event.get("participating_entities"), f"reasoning_events[{index}].participating_entities")
        if not part_ents:
            raise ValueError(f"reasoning_events[{index}].participating_entities must not be empty")
        for pe in part_ents:
            if pe not in entity_ids:
                raise ValueError(f"reasoning_events[{index}] entity {pe} not found in physical_entities")
        
        compatible_focus_participants = [
            pe for pe in part_ents
            if any(
                evt_type in FOCUS_REASON_EVENT_TYPE_MAP.get(r, set())
                for r in entity_focus_reasons.get(pe, set())
            )
        ]
        if not compatible_focus_participants:
            raise ValueError(f"reasoning_events[{index}] event_type {evt_type} requires at least one participating_entity to have a compatible focus reason")
        
        atom_refs = _require_list(event.get("supporting_atom_refs"), f"reasoning_events[{index}].supporting_atom_refs")
        if not atom_refs:
            raise ValueError(f"reasoning_events[{index}] must have supporting_atom_refs")
        for ref in atom_refs:
            if not (ref.startswith("v1_atom_") or ref.startswith("v2_atom_")):
                raise ValueError(f"reasoning_events[{index}].supporting_atom_refs must only point to atoms. Invalid: {ref}")
            if ref not in evidence_namespace:
                raise ValueError(f"reasoning_events[{index}] references unknown atom: {ref}")
                
        dynamic_event_types = {"temporal_change", "occlusion_change", "spatial_transition"}
        if evt_type in dynamic_event_types:
            supporting_frames = set()
            for ref in atom_refs:
                supporting_frames.update(atom_frame_keys.get(ref, set()))
            if len(supporting_frames) < 2:
                raise ValueError(f"reasoning_events[{index}] {evt_type} requires evidence spanning at least 2 distinct frames")
                
        if evt_type == "joint_fusion":
            has_v1 = any(r.startswith("v1_atom_") for r in atom_refs)
            has_v2 = any(r.startswith("v2_atom_") for r in atom_refs)
            if not (has_v1 and has_v2):
                raise ValueError(f"reasoning_events[{index}] joint_fusion requires at least one V1 atom and one V2 atom")
        _require_string(event.get("description"), f"reasoning_events[{index}].description")

    for fe_id, reasons in entity_focus_reasons.items():
        for r in reasons:
            allowed_event_types = FOCUS_REASON_EVENT_TYPE_MAP.get(r)
            if not allowed_event_types:
                continue
            
            matched = any(
                isinstance(event, dict)
                and event.get("event_type") in allowed_event_types
                and fe_id in event.get("participating_entities", [])
                for event in events
            )
            
            if not matched:
                raise ValueError(f"Entity {fe_id} declares focus reason {r!r} but has no compatible reasoning_event")

    ambiguities = _require_list(parsed["ambiguity_events"], "ambiguity_events")
    for index, event in enumerate(ambiguities, start=1):
        if not isinstance(event, dict):
            raise ValueError(f"ambiguity_events[{index}] must be an object")
        amb_id = _require_string(event.get("ambiguity_id"), f"ambiguity_events[{index}].ambiguity_id")
        if not amb_id.startswith("amb_"):
            raise ValueError(f"ambiguity_events[{index}].ambiguity_id must start with amb_")
        _register_evidence_id(amb_id)
        target = _require_string(event.get("target_entity"), f"ambiguity_events[{index}].target_entity")
        if target not in entity_ids:
            raise ValueError(f"ambiguity_events[{index}] target_entity {target} not found in physical_entities")
        if "ambiguity_resolution" not in entity_focus_reasons.get(target, set()):
            raise ValueError(f"ambiguity_events[{index}] target {target} lacks ambiguity_resolution focus_reason")
        
        direction = event.get("direction")
        if direction not in ALLOWED_AMBIGUITY_DIRECTIONS:
            raise ValueError(f"ambiguity_events[{index}].direction must be video1_resolves_video2 or video2_resolves_video1")
        
        amb_video = _require_string(event.get("ambiguous_video"), f"ambiguity_events[{index}].ambiguous_video")
        res_video = _require_string(event.get("resolving_video"), f"ambiguity_events[{index}].resolving_video")
        if direction == "video1_resolves_video2":
            if amb_video != "video2" or res_video != "video1":
                raise ValueError(f"ambiguity_events[{index}] direction {direction} contradicts video fields")
        elif direction == "video2_resolves_video1":
            if amb_video != "video1" or res_video != "video2":
                raise ValueError(f"ambiguity_events[{index}] direction {direction} contradicts video fields")

        for key in (
            "low_confidence_observation", "why_ambiguous_video_cannot_resolve", 
            "resolving_discriminative_evidence", "fusion_conclusion",
        ):
            _require_string(event.get(key), f"ambiguity_events[{index}].{key}")
            
        missing_type = event.get("missing_attribute_type")
        if missing_type not in ALLOWED_MISSING_ATTRIBUTE_TYPES:
            raise ValueError(f"ambiguity_events[{index}].missing_attribute_type invalid: {missing_type}")

        hypotheses = _require_list(event.get("candidate_hypotheses"), f"ambiguity_events[{index}].candidate_hypotheses")
        if len(hypotheses) < 2:
            raise ValueError(f"ambiguity_events[{index}].candidate_hypotheses must include at least two hypotheses")
            
        normalized_hyps = {h.get("hypothesis", "").strip().casefold() for h in hypotheses if isinstance(h, dict)}
        if len(normalized_hyps) < 2:
            raise ValueError(f"ambiguity_events[{index}].candidate_hypotheses must contain at least two distinct hypotheses")
            
        for hyp_index, hypothesis in enumerate(hypotheses, start=1):
            if not isinstance(hypothesis, dict):
                raise ValueError(f"ambiguity_events[{index}].candidate_hypotheses[{hyp_index}] must be an object")
            _require_string(hypothesis.get("hypothesis"), f"ambiguity_events[{index}].candidate_hypotheses[{hyp_index}].hypothesis")
            _require_string(hypothesis.get("support_from_resolving"), f"ambiguity_events[{index}].candidate_hypotheses[{hyp_index}].support_from_resolving")
            
        eliminated = _require_list(event.get("eliminated_hypotheses"), f"ambiguity_events[{index}].eliminated_hypotheses")
        if not eliminated:
            raise ValueError(f"ambiguity_events[{index}].eliminated_hypotheses must not be empty")
            
        candidate_names = {h["hypothesis"].strip().casefold() for h in hypotheses if isinstance(h, dict) and "hypothesis" in h}
            
        for elim_index, hypothesis in enumerate(eliminated, start=1):
            if not isinstance(hypothesis, dict):
                raise ValueError(f"ambiguity_events[{index}].eliminated_hypotheses[{elim_index}] must be an object")
            elim_name = _require_string(hypothesis.get("hypothesis"), f"ambiguity_events[{index}].eliminated_hypotheses[{elim_index}].hypothesis")
            if elim_name.strip().casefold() not in candidate_names:
                raise ValueError(f"ambiguity_events[{index}] eliminated hypothesis must appear in candidate_hypotheses")
            _require_string(hypothesis.get("why_eliminated"), f"ambiguity_events[{index}].eliminated_hypotheses[{elim_index}].why_eliminated")
        
        amb_refs = _require_list(event.get("ambiguous_evidence_refs"), f"ambiguity_events[{index}].ambiguous_evidence_refs")
        if not amb_refs:
            raise ValueError(f"ambiguity_events[{index}].ambiguous_evidence_refs must not be empty")
        res_refs = _require_list(event.get("resolving_evidence_refs"), f"ambiguity_events[{index}].resolving_evidence_refs")
        if not res_refs:
            raise ValueError(f"ambiguity_events[{index}].resolving_evidence_refs must not be empty")
            
        if direction == "video1_resolves_video2":
            if not all(r.startswith("v2_atom_") for r in amb_refs):
                raise ValueError(f"ambiguity_events[{index}] direction {direction} requires ambiguous_evidence_refs to be v2_atom_")
            if not all(r.startswith("v1_atom_") for r in res_refs):
                raise ValueError(f"ambiguity_events[{index}] direction {direction} requires resolving_evidence_refs to be v1_atom_")
        elif direction == "video2_resolves_video1":
            if not all(r.startswith("v1_atom_") for r in amb_refs):
                raise ValueError(f"ambiguity_events[{index}] direction {direction} requires ambiguous_evidence_refs to be v1_atom_")
            if not all(r.startswith("v2_atom_") for r in res_refs):
                raise ValueError(f"ambiguity_events[{index}] direction {direction} requires resolving_evidence_refs to be v2_atom_")

        for ref in amb_refs + res_refs:
            if not (ref.startswith("v1_atom_") or ref.startswith("v2_atom_")):
                raise ValueError(f"ambiguity_events[{index}] evidence_refs must only point to atoms. Invalid: {ref}")
            if ref not in evidence_namespace:
                raise ValueError(f"ambiguity_events[{index}] references unknown atom: {ref}")

    QA_PATTERN_EVENT_TYPE_MAP = {
        "temporal_integration": {"temporal_change"},
        "occlusion_reasoning": {"occlusion_change"},
        "interaction_reasoning": {"interaction"},
        "spatial_transition": {"spatial_transition"},
        "joint_fusion": {"joint_fusion"},
    }

    qa_details = _require_list(parsed["qa_relevant_details"], "qa_relevant_details")
    for index, qa in enumerate(qa_details, start=1):
        if not isinstance(qa, dict):
            raise ValueError(f"qa_relevant_details[{index}] must be an object")
        qa_id = _require_string(qa.get("detail_id"), f"qa_relevant_details[{index}].detail_id")
        if not qa_id.startswith("qa_detail_"):
            raise ValueError(f"qa_relevant_details[{index}].detail_id must start with qa_detail_")
        _register_evidence_id(qa_id)
        pat = _require_string(qa.get("reasoning_pattern"), f"qa_relevant_details[{index}].reasoning_pattern")
        if pat not in ALLOWED_QA_REASONING_PATTERNS:
            raise ValueError(f"qa_relevant_details[{index}].reasoning_pattern invalid: {pat}")
        refs = _require_list(qa.get("supporting_refs"), f"qa_relevant_details[{index}].supporting_refs")
        if not refs:
            raise ValueError(f"qa_relevant_details[{index}].supporting_refs must not be empty")
        for ref in refs:
            if ref.startswith("qa_detail_"):
                raise ValueError(f"qa_relevant_details[{index}] illegally references another qa_detail: {ref}")
            if ref not in evidence_namespace:
                raise ValueError(f"qa_relevant_details[{index}] references unknown ID: {ref}")
                
        expected_types = QA_PATTERN_EVENT_TYPE_MAP.get(pat)
        if expected_types:
            referenced_event_types = {
                evt["event_type"] for evt in events if evt["event_id"] in refs
            }
            if not (referenced_event_types & expected_types):
                raise ValueError(f"qa_relevant_details[{index}] pattern '{pat}' requires at least one supporting event of type: {', '.join(expected_types)}")
            
        def _resolve_atoms(ref_id: str) -> set[str]:
            if ref_id.startswith("v1_atom_") or ref_id.startswith("v2_atom_"):
                return {ref_id}
            atoms = set()
            if ref_id.startswith("evt_"):
                evt = next(e for e in events if e.get("event_id") == ref_id)
                atoms.update(evt.get("supporting_atom_refs", []))
            elif ref_id.startswith("amb_"):
                amb = next(a for a in ambiguities if a.get("ambiguity_id") == ref_id)
                atoms.update(amb.get("ambiguous_evidence_refs", []))
                atoms.update(amb.get("resolving_evidence_refs", []))
            return atoms

        resolved_atoms = set()
        for ref in refs:
            resolved_atoms.update(_resolve_atoms(ref))
            
        if pat == "multi_hop_composition" and len(resolved_atoms) < 2:
            raise ValueError(f"qa_relevant_details[{index}] multi_hop_composition requires at least 2 underlying atoms")
        if pat in ("cross_modal_disambiguation", "hypothesis_elimination"):
            if not any(r.startswith("amb_") for r in refs):
                raise ValueError(f"qa_relevant_details[{index}] {pat} MUST reference at least one amb_ event directly")
        if pat == "joint_fusion":
            has_v1 = any(a.startswith("v1_atom_") for a in resolved_atoms)
            has_v2 = any(a.startswith("v2_atom_") for a in resolved_atoms)
            if not (has_v1 and has_v2):
                raise ValueError(f"qa_relevant_details[{index}] joint_fusion requires at least one V1 atom and one V2 atom in its resolved tree")
        
        _require_string(qa.get("why_question_worthy"), f"qa_relevant_details[{index}].why_question_worthy")

    rejected = _require_list(parsed["rejected_observations"], "rejected_observations")
    for index, item in enumerate(rejected, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"rejected_observations[{index}] must be an object")
        _require_string(item.get("observation"), f"rejected_observations[{index}].observation")
        _require_string(item.get("reason"), f"rejected_observations[{index}].reason")

    def _check_missing_attrs(analysis_key: str, required_prefix: str, ref_modality: str):
        analysis = parsed.get(analysis_key, {})
        for i, attr in enumerate(analysis.get("missing_key_attributes", []), start=1):
            refs = attr.get("recoverable_evidence_refs", [])
            has_required = False
            for ref in refs:
                if not (ref.startswith("v1_atom_") or ref.startswith("v2_atom_")):
                    raise ValueError(f"{analysis_key}.missing_key_attributes[{i}] recoverable_evidence_refs MUST only reference atoms. Invalid: {ref}")
                if ref not in evidence_namespace:
                    raise ValueError(f"{analysis_key}.missing_key_attributes[{i}] references unknown atom: {ref}")
                if ref.startswith(required_prefix):
                    has_required = True
            if not has_required:
                raise ValueError(f"{analysis_key}.missing_key_attributes[{i}] MUST reference at least one {required_prefix} atom to prove cross-modal recovery")
            
            attr_type = attr.get("attribute_type")
            if attr_type == "surface_attribute" and "color" in attr.get("missing_attribute", "").lower():
                cap = MODALITY_CAPABILITIES.get(ref_modality, {})
                if not cap.get("color", True):
                    raise ValueError(
                        f"{analysis_key}.missing_key_attributes[{i}]: "
                        f"'{attr.get('missing_attribute')}' marked recoverable from {ref_modality}, "
                        f"but {ref_modality} has no color capability."
                    )

    _check_missing_attrs("video1_analysis", "v2_atom_", modality2)
    _check_missing_attrs("video2_analysis", "v1_atom_", modality1)
    
    has_reasoning_content = any([
        parsed.get("cross_modal_evidence_links"),
        parsed.get("information_gain"),
        parsed.get("reasoning_events"),
        parsed.get("ambiguity_events"),
        parsed.get("qa_relevant_details"),
    ])
    if not has_reasoning_content:
        raise ValueError("Caption contains no reasoning-relevant graph content (all evidence/reasoning sections are empty).")
        
    parsed = _normalize_license_plates(parsed)
    return parsed


async def _call_gemini_caption(client, task: CaptionTask, model_name: str, max_retries: int) -> dict[str, Any]:
    _ensure_composite_frames(task)
    encoded = _encode_images(task.composite_frames)
    if not encoded:
        raise ValueError("No composite frames found for Gemini call")
    base_contents = build_image_parts(encoded) + [_build_caption_prompt(task)]
    contents = base_contents
    raw_text = None
    for attempt in range(1, max_retries + 1):
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=model_name,
                contents=contents,
            )
            raw_text = response.text
            valid_frame_keys = {path.stem for path in task.composite_frames}
            return _validate_caption_schema(_parse_json_response(raw_text), valid_frame_keys, task.modality1, task.modality2)
        except Exception as exc:
            exc_str = str(exc).lower()
            # Quota / rate-limit errors are permanent for the current key — don't waste retries
            if "429" in exc_str or "quota" in exc_str:
                raise
            if attempt == max_retries:
                raise
            wait_seconds = 2 * attempt
            print(
                f"    Caption Gemini call failed on attempt {attempt}/{max_retries}; "
                f"retrying in {wait_seconds}s: {exc}"
            )
            await asyncio.sleep(wait_seconds)
            # Error-guided retry: tell the model exactly what went wrong
            previous_context = ""
            if raw_text:
                previous_context = f"\n\nHere is your previous invalid response:\n```json\n{raw_text}\n```\n\n"
            error_feedback = (
                f"{previous_context}"
                f"Your previous response was REJECTED due to this validation error: [{exc}]. "
                f"Please fix ONLY this specific issue, maintain strict semantic consistency "
                f"with all other fields, and return the complete updated JSON."
            )
            contents = base_contents + [error_feedback]
    raise RuntimeError("Gemini caption call failed")


def _task_metadata(task: CaptionTask) -> dict[str, Any]:
    return {
        "caption_id": task.caption_id,
        "segment_id": task.segment_id,
        "split_dir": task.split_dir,
        "segment_name": task.segment_name,
        "side": task.side,
        "modality1": task.modality1,
        "modality2": task.modality2,
        "frame_dir1": task.frame_dir1.as_posix(),
        "frame_dir2": task.frame_dir2.as_posix(),
        "frames1": [path.as_posix() for path in task.frames1],
        "frames2": [path.as_posix() for path in task.frames2],
        "composite_frames": [f.name for f in task.composite_frames],
        "sampling_strategy": task.sampling_strategy,
        "uniform_anchor_indexes": list(task.uniform_anchor_indexes),
        "num_selected_frames": len(task.selected_frame_indexes),
    }

def _task_to_item(task: CaptionTask, status: str, caption: dict[str, Any] | None = None, reason: str | None = None) -> dict[str, Any]:
    item = _task_metadata(task)
    item.update({
        "status": status,
        "reason": reason,
        "caption": caption,
    })
    return item


def _template_caption(task: CaptionTask) -> dict[str, Any]:
    target_entity = "unresolved_target"
    return {
        "schema_version": CAPTION_SCHEMA_VERSION,
        "global_scene": {
            "scene_summary": "Template mode placeholder for scene summary. " * 5,
            "physical_entities": [
                {
                    "entity_id": target_entity,
                    "category": "unknown",
                    "evidence_profile": {
                        "identity_evidence": ["Placeholder evidence"],
                        "observable_attributes": ["Placeholder attribute"],
                        "spatial_context": ["Placeholder context"]
                    }
                }
            ],
            "reasoning_focus_entities": [
                {
                    "entity_id": target_entity,
                    "focus_reasons": ["joint_fusion", "ambiguity_resolution"]
                }
            ],
            "environment": "unknown",
            "temporal_progression": "Template mode placeholder for temporal progression. " * 3,
        },
        "video1_analysis": {
            "modality": task.modality1,
            "detailed_caption": "Template mode placeholder for detailed caption. " * 6,
            "information_atoms": [
                {
                    "atom_id": "v1_atom_001",
                    "frame_keys": [task.composite_frames[0].stem] if task.composite_frames else [],
                    "fact": "Template placeholder fact"
                }
            ],
            "sensor_specific_cues": ["Template mode placeholder."],
            "sensor_limitations": ["Template mode placeholder."],
            "uncertain_observations": [
                {
                    "observed_evidence": "Template placeholder",
                    "hypotheses": [{"hypothesis": "Placeholder 1", "confidence": "low"}, {"hypothesis": "Placeholder 2", "confidence": "low"}],
                    "missing_evidence": "Placeholder"
                }
            ],
            "missing_key_attributes": [
                {
                    "attribute_type": "existence",
                    "missing_attribute": "Template placeholder",
                    "why_missing": "Template placeholder",
                    "recoverable_evidence_refs": ["v2_atom_001"]
                }
            ],
        },
        "video2_analysis": {
            "modality": task.modality2,
            "detailed_caption": "Template mode placeholder for detailed caption. " * 6,
            "information_atoms": [
                {
                    "atom_id": "v2_atom_001",
                    "frame_keys": [task.composite_frames[0].stem] if task.composite_frames else [],
                    "fact": "Template placeholder fact"
                }
            ],
            "sensor_specific_cues": ["Template mode placeholder."],
            "sensor_limitations": ["Template mode placeholder."],
            "uncertain_observations": [
                {
                    "observed_evidence": "Template placeholder",
                    "hypotheses": [{"hypothesis": "Placeholder 1", "confidence": "low"}, {"hypothesis": "Placeholder 2", "confidence": "low"}],
                    "missing_evidence": "Placeholder"
                }
            ],
            "missing_key_attributes": [
                {
                    "attribute_type": "existence",
                    "missing_attribute": "Template placeholder",
                    "why_missing": "Template placeholder",
                    "recoverable_evidence_refs": ["v1_atom_001"]
                }
            ],
        },
        "cross_modal_evidence_links": [],
        "information_gain": [
            {
                "entity_id": target_entity,
                "video1_can_determine": ["Placeholder"],
                "video1_cannot_determine": ["Placeholder"],
                "video2_can_determine": ["Placeholder"],
                "video2_cannot_determine": ["Placeholder"],
                "fusion_additionally_reveals": ["Placeholder"],
                "gain_rating": "low",
            }
        ],
        "reasoning_events": [
            {
                "event_id": "evt_001",
                "event_type": "joint_fusion",
                "participating_entities": [target_entity],
                "supporting_atom_refs": ["v1_atom_001", "v2_atom_001"],
                "description": "Template placeholder event"
            }
        ],
        "ambiguity_events": [
            {
                "ambiguity_id": "amb_001",
                "target_entity": target_entity,
                "direction": "video1_resolves_video2",
                "ambiguous_video": "video2",
                "resolving_video": "video1",
                "low_confidence_observation": "Placeholder",
                "why_ambiguous_video_cannot_resolve": "Placeholder",
                "candidate_hypotheses": [
                    {"hypothesis": "Placeholder 1", "support_from_resolving": "Placeholder"},
                    {"hypothesis": "Placeholder 2", "support_from_resolving": "Placeholder"}
                ],
                "resolving_discriminative_evidence": "Placeholder",
                "eliminated_hypotheses": [{"hypothesis": "Placeholder 1", "why_eliminated": "Placeholder"}],
                "fusion_conclusion": "Placeholder",
                "missing_attribute_type": "existence",
                "ambiguous_evidence_refs": ["v2_atom_001"],
                "resolving_evidence_refs": ["v1_atom_001"]
            }
        ],
        "qa_relevant_details": [
            {
                "detail_id": "qa_detail_001",
                "reasoning_pattern": "joint_fusion",
                "supporting_refs": ["evt_001", "amb_001"],
                "why_question_worthy": "Placeholder"
            }
        ],
        "rejected_observations": [
            {
                "observation": "No rejected observation was evaluated in template mode.",
                "reason": "Gemini was not called in template mode."
            }
        ],
    }


def _batch_state_path(output_path: Path | str) -> Path:
    out = Path(output_path)
    return out.with_name(f".{out.stem}_batch_state.json")

def _build_batch_request(task: CaptionTask, model_name: str, req_id: str) -> "genai_types.InlinedRequest":
    if genai_types is None:
        raise ImportError("google.genai is not installed")
    _ensure_composite_frames(task)
    encoded = _encode_images(task.composite_frames)
    if not encoded:
        raise ValueError("No composite frames found for Gemini batch call")
    prompt_part = genai_types.Part.from_text(text=_build_caption_prompt(task))
    base_contents = build_image_parts(encoded) + [prompt_part]
    return genai_types.InlinedRequest(
        model=model_name,
        contents=[genai_types.Content(role="user", parts=base_contents)],
        metadata={"id": req_id}
    )

def _submit_batch(client, tasks: list[CaptionTask], model_name: str, output_path: Path | str, api_key_source: str, planned_total: int) -> None:
    if not tasks:
        print("No pending tasks to submit in batch mode.")
        return
    print(f"Building batch request for {len(tasks)} tasks...")
    requests = [_build_batch_request(t, model_name, str(i)) for i, t in enumerate(tasks)]
    
    MAX_ROTATIONS = 5
    job = None
    for attempt in range(MAX_ROTATIONS + 1):
        print(f"Submitting batch job to Gemini API (Attempt {attempt+1}/{MAX_ROTATIONS+1})...")
        try:
            job = client.batches.create(model=model_name, src=requests)
            break
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str or "resource exhausted" in err_str or "resource_exhausted" in err_str or "resourceexhausted" in err_str:
                if attempt == MAX_ROTATIONS:
                    print(f"FATAL: Exceeded maximum API key rotations during batch submission: {e}")
                    raise
                print(f"WARNING: Quota exhausted or rate limit hit submitting batch. Rotating API key...")
                try:
                    from annotation_feature.pipeline.client import rotate_gemini_client
                    client = rotate_gemini_client()
                except Exception as rot_e:
                    print(f"FATAL: All API keys exhausted or failed to rotate: {rot_e}")
                    raise
            else:
                raise

    print(f"Batch job submitted successfully! Job Name: {job.name}")
    
    state_path = _batch_state_path(output_path)
    state = {
        "job_name": job.name,
        "model_name": model_name,
        "api_key_source": api_key_source,
        "output_path": str(output_path),
        "planned_total": planned_total,
        "submitted_at": datetime.datetime.now().isoformat(),
        "pending_tasks": [_task_metadata(t) for t in tasks]
    }
    _save_json(state, state_path)
    print(f"Saved batch state to {state_path}")
    print("You can exit the script now. Check status later using --fetch-batch.")

def _fetch_batch(client, output_path: Path | str, batch_state_path: Path | str | None = None) -> None:
    out_path = Path(output_path)
    state_path = Path(batch_state_path) if batch_state_path else _batch_state_path(out_path)
    if not state_path.exists():
        print(f"Batch state file not found: {state_path}. Cannot fetch.")
        return
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
        
    saved_api_key_source = state.get("api_key_source", "list")
    if client is None:
        client = create_gemini_client(api_key_source=saved_api_key_source)
            
    if "fetched_at" in state:
        print(f"Batch state file {state_path} was already fetched at {state['fetched_at']}.")
        return

    job_name = state["job_name"]
    print(f"Checking status for batch job: {job_name}")
    job = client.batches.get(name=job_name)
    
    if job.state.name in ("JOB_STATE_PENDING", "JOB_STATE_RUNNING", "JOB_STATE_CANCELLING", "JOB_STATE_PARTIALLY_SUCCEEDED"):
        print(f"Job is still running. Current state: {job.state.name}")
        return
    elif job.state.name in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED"):
        print(f"Job finished with state: {job.state.name}")
        print(f"Error details: {job.error}")
        return
        
    print(f"Job succeeded! Fetching results...")
    
    existing_items, existing_skipped = _load_resume(out_path)
    items = existing_items
    skipped = existing_skipped
    existing_keys = {
        (str(item.get("caption_id")), item.get("selection_config_fingerprint"))
        for item in items 
        if item.get("caption_id")
    }
    
    pending_tasks_list = state.get("pending_tasks", [])
    pending_tasks_map = {str(i): t for i, t in enumerate(pending_tasks_list)}
    
    seen_in_this_fetch = set()
    
    for resp_entry in job.dest.inlined_responses:
        req_id = resp_entry.metadata.get("id") if resp_entry.metadata else None
        if req_id is None:
            continue
        
        task_dict = pending_tasks_map.get(req_id)
        if not task_dict:
            continue
            
        caption_id = task_dict.get("caption_id")
        fingerprint = task_dict.get("selection_config_fingerprint")
        selection_key = (str(caption_id), str(fingerprint))
        
        if not caption_id or selection_key in existing_keys:
            continue
            
        if selection_key in seen_in_this_fetch:
            continue
        seen_in_this_fetch.add(selection_key)
        base_item = task_dict.copy()
        
        if resp_entry.error:
            base_item["reason"] = f"Batch API Error: {resp_entry.error}"
            skipped[:] = [item for item in skipped if item.get("caption_id") != caption_id]
            skipped.append(base_item)
            continue
            
        try:
            raw_text = "".join(part.text for part in resp_entry.response.candidates[0].content.parts)
            valid_frame_keys = {Path(p).stem for p in task_dict.get("composite_frames", [])}
            caption = _validate_caption_schema(_parse_json_response(raw_text), valid_frame_keys, task_dict.get("modality1", ""), task_dict.get("modality2", ""))
            base_item["status"] = "generated_batch"
            base_item["caption"] = caption
            skipped[:] = [item for item in skipped if item.get("caption_id") != caption_id]
            items[:] = [item for item in items if item.get("caption_id") != caption_id]
            items.append(base_item)
        except Exception as exc:
            base_item["reason"] = f"Validation Error: {exc}"
            skipped[:] = [item for item in skipped if item.get("caption_id") != caption_id]
            skipped.append(base_item)
            
    _save_json(
        _build_output_payload(
            input_path=Path("batch_mode"),
            dataset_root=Path("batch_mode"),
            composite_root=Path("batch_mode"),
            output_path=out_path,
            model_name=state.get("model_name", "unknown"),
            generation_mode="batch",
            items=items,
            skipped=skipped,
            planned_total=state.get("planned_total", len(items) + len(skipped)),
            gemini_calls=len(job.dest.inlined_responses),
        ),
        out_path,
    )
    
    state["fetched_at"] = datetime.datetime.now().isoformat()
    _save_json(state, state_path)
    
    print(f"Wrote batch fetched results to {out_path}")


def _build_output_payload(
    input_path: Path,
    dataset_root: Path,
    composite_root: Path,
    output_path: Path,
    model_name: str,
    generation_mode: str,
    items: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    planned_total: int,
    gemini_calls: int,
) -> dict[str, Any]:
    return {
        "metadata": {
            "task": "cross_modal_disambiguation_caption",
            "schema_version": CAPTION_SCHEMA_VERSION,
            "input": input_path.as_posix(),
            "output": output_path.as_posix(),
            "dataset_root": dataset_root.as_posix(),
            "composite_root": composite_root.as_posix(),
            "generation_mode": generation_mode,
            "model_name": model_name,
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
        print(f"WARNING: Could not load existing caption output for resume: {exc}")
        return [], []
    items = data.get("items") if isinstance(data.get("items"), list) else []
    skipped = data.get("skipped") if isinstance(data.get("skipped"), list) else []
    
    # Filter out stale "llm semantic selection failed" errors from legacy pipeline
    filtered_skipped = []
    for item in skipped:
        reason = str(item.get("reason", ""))
        if "llm semantic selection failed" in reason or "additional_properties" in reason:
            continue
        filtered_skipped.append(item)
        
    return list(items), filtered_skipped


async def run_caption_pipeline_async(
    input_path: Path,
    output_path: Path,
    dataset_root: Path,
    composite_root: Path,
    model_name: str,
    generation_mode: str,
    api_key_source: str,
    num_uniform_frames: int,
    num_adaptive_frames: int,
    pairs: str | None,
    directions: str | None,
    sides: str | None,
    limit: int | None,
    limit_scenes: int | None,
    limit_scene_folders: int | None,
    max_retries: int,
    delay_between_calls: int,
    checkpoint_every: int,
    resume: bool,
) -> Path:
    allowed_pairs = _parse_pairs(pairs)
    allowed_directions = _parse_pairs(directions)
    allowed_sides = _parse_sides(sides)
    
    client = create_gemini_client(api_key_source=api_key_source) if generation_mode in ("gemini", "batch") else None
    
    existing_items, existing_skipped = _load_resume(output_path) if resume else ([], [])
    items = existing_items
    skipped = existing_skipped
    
    tasks, new_skipped, total_selected_jobs = build_caption_tasks(
        input_path=input_path,
        dataset_root=dataset_root,
        composite_root=composite_root,
        sampling_strategy="uniform_adaptive",
        num_uniform_frames=num_uniform_frames,
        num_adaptive_frames=num_adaptive_frames,
        existing_items=existing_items,
        allowed_pairs=allowed_pairs,
        allowed_directions=allowed_directions,
        limit=limit,
        limit_scenes=limit_scenes,
        limit_scene_folders=limit_scene_folders,
        allowed_sides=allowed_sides,
        write_composites=False,
    )
    
    seen_skipped = set()
    deduped_skipped = []
    for s in skipped + new_skipped:
        key = (s.get("segment_id"), s.get("side"), s.get("modality1"), s.get("modality2"), s.get("reason"), s.get("caption_id"))
        if key not in seen_skipped:
            seen_skipped.add(key)
            deduped_skipped.append(s)
    skipped = deduped_skipped
    pending_tasks = tasks # Pending tasks are already filtered inside build_caption_tasks

    # client already created if needed
    gemini_calls = 0
    checkpoint_counter = 0

    print(
        f"Generating cross-modal captions: {len(tasks)} planned item(s), "
        f"{len(pending_tasks)} pending, mode={generation_mode}, model={model_name}."
    )
    
    if generation_mode == "batch":
        assert client is not None
        _submit_batch(client, pending_tasks, model_name, output_path, api_key_source, total_selected_jobs)
        return output_path

    def save_checkpoint() -> None:
        _save_json(
            _build_output_payload(
                input_path=input_path,
                dataset_root=dataset_root,
                composite_root=composite_root,
                output_path=output_path,
                model_name=model_name,
                generation_mode=generation_mode,
                items=items,
                skipped=skipped,
                planned_total=total_selected_jobs,
                gemini_calls=gemini_calls,
            ),
            output_path,
        )

    task_index = 0
    MAX_KEY_ROTATIONS = 5
    rotation_attempts = 0

    while task_index < len(pending_tasks):
        task = pending_tasks[task_index]
        print(
            f"  Caption item [{task_index + 1}/{len(pending_tasks)}] "
            f"{task.caption_id}"
        )
        try:
            if generation_mode == "gemini":
                assert client is not None
                caption = await _call_gemini_caption(client, task, model_name, max_retries=max_retries)
                gemini_calls += 1
                status = "generated"
            else:
                _ensure_composite_frames(task)
                valid_frame_keys = {path.stem for path in task.composite_frames}
                caption = _validate_caption_schema(_template_caption(task), valid_frame_keys, task.modality1, task.modality2)
                status = "template"
            skipped[:] = [item for item in skipped if item.get("caption_id") != task.caption_id]
            items[:] = [item for item in items if item.get("caption_id") != task.caption_id]
            items.append(_task_to_item(task, status=status, caption=caption))
            rotation_attempts = 0
        except Exception as exc:
            exc_str = str(exc).lower()
            if "429" in exc_str or "quota" in exc_str:
                rotation_attempts += 1
                if rotation_attempts > MAX_KEY_ROTATIONS:
                    raise RuntimeError("Exceeded maximum API-key rotation attempts")
                print(f"WARNING: Quota exhausted or rate limit hit. Attempting to rotate API key...")
                try:
                    client = create_gemini_client(api_key_source=api_key_source)
                    # Retry the current task seamlessly
                    continue
                except Exception as rotate_exc:
                    print(f"FATAL: All API keys exhausted or failed to rotate: {rotate_exc}")
                    break
            skipped[:] = [item for item in skipped if item.get("caption_id") != task.caption_id]
            skipped.append(
                {
                    "caption_id": task.caption_id,
                    "segment_id": task.segment_id,
                    "side": task.side,
                    "modality1": task.modality1,
                    "modality2": task.modality2,
                    "reason": str(exc),
                }
            )
            print(f"WARNING: Caption generation failed for {task.caption_id}: {exc}")

        checkpoint_counter += 1
        if checkpoint_every > 0 and checkpoint_counter >= checkpoint_every:
            checkpoint_counter = 0
            save_checkpoint()
        if generation_mode == "gemini" and delay_between_calls > 0 and task_index < len(pending_tasks) - 1:
            await asyncio.sleep(delay_between_calls)
            
        task_index += 1

    save_checkpoint()
    print(f"Wrote cross-modal caption output to {output_path}")
    return output_path


def run_caption_pipeline(
    input_path: Path | str = DEFAULT_INPUT_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    dataset_root: Path | str = DEFAULT_DATASET_ROOT,
    composite_root: Path | str = DEFAULT_COMPOSITE_ROOT,
    model_name: str = DEFAULT_MODEL_NAME,
    generation_mode: str = "template",
    api_key_source: str = "list",
    num_uniform_frames: int = 8,
    num_adaptive_frames: int = 2,
    pairs: str | None = None,
    directions: str | None = None,
    sides: str | None = None,
    limit: int | None = None,
    limit_scenes: int | None = None,
    limit_scene_folders: int | None = None,
    max_retries: int = 3,
    delay_between_calls: int = 5,
    checkpoint_every: int = 1,
    resume: bool = True,
) -> Path:
    return asyncio.run(
        run_caption_pipeline_async(
            input_path=Path(input_path),
            output_path=Path(output_path),
            dataset_root=Path(dataset_root),
            composite_root=Path(composite_root),
            model_name=model_name,
            generation_mode=generation_mode,
            api_key_source=api_key_source,
            num_uniform_frames=num_uniform_frames,
            num_adaptive_frames=num_adaptive_frames,
            pairs=pairs,
            directions=directions,
            sides=sides,
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
    parser.add_argument("--composite-root", default=str(DEFAULT_COMPOSITE_ROOT))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--generation-mode",
        choices=("template", "gemini", "batch"),
        default="template",
        help="Use template to build composite frames without calling Gemini. Use batch for async batch API.",
    )
    parser.add_argument("--fetch-batch", action="store_true", help="Fetch results for a pending batch job instead of creating tasks.")
    parser.add_argument("--batch-state", default=None, help="Path to batch state file (optional).")
    parser.add_argument("--api-key-source", choices=("env", "list"), default="list", help="Source for Gemini API keys.")
    parser.add_argument("--num-uniform-frames", type=int, default=8)
    parser.add_argument("--num-adaptive-frames", type=int, default=2)
    parser.add_argument(
        "--pairs",
        default=None,
        help="Comma-separated modality pairs such as rgb+depth,rgb+event. Defaults to input modality_pairs.",
    )
    parser.add_argument(
        "--sides",
        default=None,
        help="Comma-separated sides to process, such as day,night,aligned. Defaults to all available.",
    )
    parser.add_argument(
        "--directions",
        default=None,
        help="Comma-separated modality1->modality2 directions such as rgb->depth,event->ir. Defaults to the canonical ordering from input modality_pairs.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--limit-scenes",
        "--limit-segments",
        dest="limit_scenes",
        type=int,
        default=None,
        help="Limit the number of unique segment_id scenes. Unlike --limit, this keeps all matching items in each selected scene.",
    )
    parser.add_argument(
        "--limit-scene-folders",
        "--limit-split-dirs",
        dest="limit_scene_folders",
        type=int,
        default=None,
        help="Limit the number of top-level aligned_dataset scene folders/split_dir values, such as brew_tea_split.",
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--delay-between-calls", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    
    return parser


def main() -> None:
    import sys
    import asyncio
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    args = _build_arg_parser().parse_args()
    
    if args.fetch_batch:
        _fetch_batch(None, args.output, args.batch_state)
        return

    run_caption_pipeline(
        input_path=args.input,
        output_path=args.output,
        dataset_root=args.dataset_root,
        composite_root=args.composite_root,
        model_name=args.model_name,
        generation_mode=args.generation_mode,
        api_key_source=args.api_key_source,
        num_uniform_frames=max(1, args.num_uniform_frames),
        num_adaptive_frames=max(0, args.num_adaptive_frames),
        pairs=args.pairs,
        directions=args.directions,
        sides=args.sides,
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
