"""Generate cross-modal disambiguation captions from aligned frame caches."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from annotation_feature.pipeline.client import create_gemini_client
from annotation_feature.pipeline.utils import build_image_parts, infer_recording_side


DEFAULT_INPUT_PATH = Path("outputs/aligned_multimodal_visual_evidence_units_filtered.json")
DEFAULT_OUTPUT_PATH = Path("outputs/aligned_cross_modal_disambiguation_captions_gemini.json")
DEFAULT_COMPOSITE_ROOT = Path("outputs/composite_frames")
DEFAULT_DATASET_ROOT = Path("aligned_dataset")
DEFAULT_MODEL_NAME = "gemini-3.1-flash-lite"
CAPTION_SCHEMA_VERSION = "cross_modal_disambiguation_caption_v5"
ALLOWED_MISSING_ATTRIBUTE_TYPES = {
    "existence",
    "target_category",
    "spatial_distance",
    "surface_attribute",
    "motion_trend",
}
ALLOWED_QA_POTENTIAL = {"high", "medium", "low"}
ALLOWED_QUESTION_TYPES = {
    "object_identity", "attribute_reasoning", "temporal_reasoning",
    "spatial_reasoning", "interaction_reasoning", "cross_modal_reasoning",
    "counterfactual_reasoning",
}
ALLOWED_GAIN_RATINGS = {"low", "medium", "high"}
ALLOWED_AMBIGUITY_DIRECTIONS = {"video1_resolves_video2", "video2_resolves_video1"}
FORBIDDEN_GLOBAL_SCENE_WORDS = re.compile(
    r"\b(modality|thermal|rgb|event|depth|infrared|ir|visible|invisible|"
    r"blurry|noisy|pixels?|grayscale|greyscale|heat|edge|edge-based|sparse|contrast|"
    r"monochrome|overexposed|saturated|contour|silhouette)\b", re.I
)
MIN_DETAILED_CAPTION_WORDS = 30
MIN_SCENE_SUMMARY_WORDS = 20
MIN_FRAME_DETAIL_WORDS = 8
MIN_OBSERVABLE_FACTS = 3
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
FRAME_CACHE_SUBDIRS = {
    "rgb": ".frames_cache",
    "ir": ".frames_cache_ir",
    "event": ".frames_cache_event",
}
SIDE_ORDER = ("day", "night", "unknown")


@dataclass(frozen=True)
class CaptionTask:
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


def _frame_index(path: Path) -> int | None:
    match = re.search(r"frame_(\d+)", path.name)
    return int(match.group(1)) if match else None


def _frames_by_index(frame_dir: Path, modality: str) -> dict[int, Path]:
    pattern = "frame_*_depth.png" if modality == "depth" else "frame_*.png"
    frames: dict[int, Path] = {}
    for path in sorted(frame_dir.glob(pattern)):
        index = _frame_index(path)
        if index is not None:
            frames[index] = path
    return frames


def _evenly_sample(values: list[int], count: int) -> list[int]:
    if count <= 0 or not values:
        return []
    if len(values) <= count:
        return values
    if count == 1:
        return [values[len(values) // 2]]
    last = len(values) - 1
    selected = [values[round(i * last / (count - 1))] for i in range(count)]
    return list(dict.fromkeys(selected))


def _side_key(path: Path) -> str:
    return infer_recording_side(path.name) or "unknown"


def _load_standard_frame_dirs(
    dataset_root: Path,
    split_dir: str,
    segment_name: str,
    modality: str,
) -> dict[str, Path]:
    cache_subdir = FRAME_CACHE_SUBDIRS[modality]
    base = dataset_root / cache_subdir / split_dir / segment_name
    if not base.exists():
        return {}
    dirs: dict[str, Path] = {}
    for path in sorted(base.iterdir()):
        if not path.is_dir():
            continue
        name = path.name.lower()
        if not name.endswith(f"_{modality}"):
            continue
        side = _side_key(path)
        dirs.setdefault(side, path)
    return dirs


def _load_depth_frame_dirs(dataset_root: Path, split_dir: str, segment_name: str) -> dict[str, Path]:
    base = dataset_root / ".frames_cache_marigold" / split_dir / segment_name
    if not base.exists():
        return {}
    dirs: dict[str, Path] = {}
    for pair_dir in sorted(base.iterdir()):
        if not pair_dir.is_dir():
            continue
        for side_dir in sorted(pair_dir.iterdir()):
            if not side_dir.is_dir():
                continue
            side = _side_key(side_dir)
            if side == "unknown" and side_dir.name.lower() in {"day", "night"}:
                side = side_dir.name.lower()
            dirs.setdefault(side, side_dir)
    return dirs


def _load_frame_dirs(
    dataset_root: Path,
    split_dir: str,
    segment_name: str,
    modality: str,
) -> dict[str, Path]:
    if modality == "depth":
        return _load_depth_frame_dirs(dataset_root, split_dir, segment_name)
    if modality not in FRAME_CACHE_SUBDIRS:
        return {}
    return _load_standard_frame_dirs(dataset_root, split_dir, segment_name, modality)


def _pair_frame_dirs(
    helper_dirs: dict[str, Path],
    victim_dirs: dict[str, Path],
) -> list[tuple[str, Path, Path]]:
    sides = [side for side in SIDE_ORDER if side in helper_dirs and side in victim_dirs]
    sides.extend(
        sorted(
            side
            for side in set(helper_dirs) & set(victim_dirs)
            if side not in SIDE_ORDER
        )
    )
    return [(side, helper_dirs[side], victim_dirs[side]) for side in sides]


def _select_paired_frames(
    helper_dir: Path,
    victim_dir: Path,
    helper_modality: str,
    victim_modality: str,
    num_frames: int,
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    helper_by_index = _frames_by_index(helper_dir, helper_modality)
    victim_by_index = _frames_by_index(victim_dir, victim_modality)
    shared_indexes = sorted(set(helper_by_index) & set(victim_by_index))
    selected_indexes = _evenly_sample(shared_indexes, num_frames)
    return (
        tuple(helper_by_index[index] for index in selected_indexes),
        tuple(victim_by_index[index] for index in selected_indexes),
    )


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
    context_frame: Path,
    decisive_frame: Path,
    helper_modality: str,
    victim_modality: str,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(context_frame) as left_raw, Image.open(decisive_frame) as right_raw:
        left = left_raw.convert("RGB")
        right = right_raw.convert("RGB")
        target_height = min(left.height, right.height)
        left = _resize_to_height(left, target_height)
        right = _resize_to_height(right, target_height)
        canvas = Image.new("RGB", (left.width + right.width, target_height), (0, 0, 0))
        canvas.paste(left, (0, 0))
        canvas.paste(right, (left.width, 0))
        _draw_label(canvas, f"LEFT: {helper_modality.upper()} helper", (0, 0))
        _draw_label(canvas, f"RIGHT: {victim_modality.upper()} victim", (left.width, 0))
        canvas.save(output_path)
    return output_path


def _ensure_composite_frame(
    context_frame: Path,
    decisive_frame: Path,
    helper_modality: str,
    victim_modality: str,
    output_path: Path,
) -> Path:
    if not output_path.exists():
        _compose_frame(
            context_frame,
            decisive_frame,
            helper_modality,
            victim_modality,
            output_path,
        )
    return output_path


def _ensure_composite_frames(task: CaptionTask) -> None:
    for context_frame, decisive_frame, composite_frame in zip(
        task.helper_frames,
        task.victim_frames,
        task.composite_frames,
    ):
        _ensure_composite_frame(
            context_frame,
            decisive_frame,
            task.helper_modality,
            task.victim_modality,
            composite_frame,
        )


def _caption_id(segment_id: str, side: str, helper_modality: str, victim_modality: str) -> str:
    return "__".join(
        [
            _safe_name(segment_id).lower(),
            _safe_name(side).lower(),
            f"{helper_modality}_helper",
            f"{victim_modality}_victim",
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
    num_frames: int,
    allowed_pairs: set[tuple[str, str]] | None = None,
    allowed_directions: set[tuple[str, str]] | None = None,
    allowed_sides: set[str] | None = None,
    limit: int | None = None,
    limit_scenes: int | None = None,
    limit_scene_folders: int | None = None,
    write_composites: bool = True,
) -> tuple[list[CaptionTask], list[dict[str, Any]]]:
    data = _load_json(input_path)
    segments = data.get("segments")
    if not isinstance(segments, dict):
        raise ValueError(f"Expected {input_path} to contain a segments object")

    tasks: list[CaptionTask] = []
    skipped: list[dict[str, Any]] = []
    selected_scenes: set[str] = set()
    selected_scene_folders: set[str] = set()
    for segment_id, segment in sorted(segments.items()):
        if limit_scenes is not None and len(selected_scenes) >= limit_scenes:
            break
        if not isinstance(segment, dict):
            continue
        task_count_before_segment = len(tasks)
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
        pairs = segment.get("modality_pairs") or []
        for pair in pairs:
            if not isinstance(pair, list | tuple) or len(pair) != 2:
                continue
            pair_tuple = (str(pair[0]).lower(), str(pair[1]).lower())
            if allowed_pairs is not None and pair_tuple not in allowed_pairs and pair_tuple[::-1] not in allowed_pairs:
                continue
            for helper_modality, victim_modality in _directions_for_pair(pair, allowed_directions):
                helper_dirs = _load_frame_dirs(dataset_root, split_dir, segment_name, helper_modality)
                victim_dirs = _load_frame_dirs(dataset_root, split_dir, segment_name, victim_modality)
                if not helper_dirs or not victim_dirs:
                    skipped.append(
                        {
                            "segment_id": segment_id,
                            "split_dir": split_dir,
                            "segment_name": segment_name,
                            "helper_modality": helper_modality,
                            "victim_modality": victim_modality,
                            "reason": "missing frame cache directory",
                        }
                    )
                    continue
                for side, helper_dir, victim_dir in _pair_frame_dirs(helper_dirs, victim_dirs):
                    if allowed_sides is not None and side.lower() not in allowed_sides:
                        continue
                    helper_frames, victim_frames = _select_paired_frames(
                        helper_dir,
                        victim_dir,
                        helper_modality,
                        victim_modality,
                        num_frames,
                    )
                    if not helper_frames:
                        skipped.append(
                            {
                                "segment_id": segment_id,
                                "side": side,
                                "helper_modality": helper_modality,
                                "victim_modality": victim_modality,
                                "helper_frame_dir": helper_dir.as_posix(),
                                "victim_frame_dir": victim_dir.as_posix(),
                                "reason": "no shared frame indexes",
                            }
                        )
                        continue
                    caption_id = _caption_id(str(segment_id), side, helper_modality, victim_modality)
                    output_dir = (
                        composite_root
                        / _safe_name(split_dir)
                        / _safe_name(segment_name)
                        / _safe_name(side)
                        / f"{helper_modality}_helper__{victim_modality}_victim"
                    )
                    composite_frames: list[Path] = []
                    for index, (context_frame, decisive_frame) in enumerate(zip(helper_frames, victim_frames), start=1):
                        frame_number = _frame_index(context_frame)
                        suffix = f"{frame_number:06d}" if frame_number is not None else f"{index:03d}"
                        output_path = output_dir / f"frame_{suffix}.png"
                        if write_composites:
                            _compose_frame(
                                context_frame,
                                decisive_frame,
                                helper_modality,
                                victim_modality,
                                output_path,
                            )
                        composite_frames.append(output_path)
                    tasks.append(
                        CaptionTask(
                            caption_id=caption_id,
                            segment_id=str(segment_id),
                            split_dir=split_dir,
                            segment_name=segment_name,
                            side=side,
                            helper_modality=helper_modality,
                            victim_modality=victim_modality,
                            helper_frame_dir=helper_dir,
                            victim_frame_dir=victim_dir,
                            helper_frames=helper_frames,
                            victim_frames=victim_frames,
                            composite_frames=tuple(composite_frames),
                        )
                    )
                    if limit is not None and len(tasks) >= limit:
                        return tasks, skipped
        if len(tasks) > task_count_before_segment:
            selected_scenes.add(str(segment_id))
            selected_scene_folders.add(split_dir)
    return tasks, skipped


def _build_caption_prompt(task: CaptionTask) -> str:
    frame_names = ", ".join(path.name for path in task.composite_frames)
    frame_keys = ", ".join(f'"{path.stem}"' for path in task.composite_frames)
    return "\n".join(
        [
            "You are an expert multimodal perception analyst.",
            "You will receive multiple synchronized composite frames sampled from one aligned video segment.",
            f"Video 1 (left): {task.helper_modality} modality.",
            f"Video 2 (right): {task.victim_modality} modality.",
            "These two videos observe the same physical scene using different sensing modalities.",
            "Neither video is considered the reference or the ground truth.",
            "The goal is not ordinary captioning. Build a dense bidirectional multimodal evidence graph that maximizes reasoning-relevant information.",
            "Only use evidence directly observable in the supplied frames. Do not invent objects, future events, intentions, identities, unreadable text, or unsupported actions.",
            "Always distinguish between physical reality, video observations, and reasoning uncertainty. Do not mix these concepts.",
            "CRITICAL INSTRUCTION: First write the GLOBAL PHYSICAL SCENE. Do NOT use ANY of the following words (case-insensitive, including plural forms) in global_scene.scene_summary or global_scene.temporal_progression: modality, thermal, rgb, event, depth, infrared, ir, visible, invisible, blurry, noisy, pixel, pixels, grayscale, heat, edge, sparse, contrast. This is an exact blocklist, not a suggestion list. Any match causes rejection. The global_scene.scene_summary must be a detailed paragraph covering: which entities are present and their appearance, their spatial layout, the environment/setting, and any ongoing actions. Do NOT write a single sentence — write a full descriptive paragraph. Trace the scene chronologically. global_scene.temporal_progression must also strictly follow the blocklist.",
            "FIELD RULES (violations cause rejection): (1) every missing_key_attributes[].recoverable_from must be a non-empty list — always include at least one recoverable source string. (2) every information_gain[].entity_id, ambiguity_events[].target_entity, AND cross_modal_evidence_links[].entity_id must exactly match an entity_id from global_scene.physical_entities. (3) detailed_caption for each video must be a full descriptive paragraph, not a single sentence. (4) detailed_caption must describe WHAT IS PHYSICALLY HAPPENING in the scene (objects, motion, actions, spatial layout), not HOW the sensor captures it. Do NOT use words like monochrome, greyscale, thermal, edge-based, overexposed, saturated, blurry, contour, silhouette, ir, or pixel in detailed_caption. Those words belong in sensor_specific_cues and sensor_limitations instead. (5) fusion_additionally_reveals must be a list of descriptive observation strings ONLY. Do NOT include rating words like 'low', 'medium', or 'high' inside this list — those belong in the separate gain_rating field. (6) ambiguity_events[].candidate_hypotheses must include AT LEAST TWO distinct hypotheses — never provide only one. (7) NEVER use generic sensor-theory wording like 'this modality captures', 'event cameras detect', or 'designed to measure' in any video analysis fields. Describe ONLY specific evidence from the current frames. (8) supported_question_types MUST only use values from the provided template list. Do NOT use attribute_type values (like surface_attribute, motion_trend) as question types. (9) In missing_key_attributes[].why_missing, explain the specific physical reason the attribute is unobservable in this exact segment, NOT generic sensor capabilities (e.g., say 'The sunlit brick and asphalt appear visually identical in color here' instead of 'Color cameras do not capture thermal energy'). (10) frame_by_frame_analysis[].frame_key must be exactly the frame name WITHOUT the file extension (e.g., 'frame_000000', NOT 'frame_000000.png'). (11) cross_modal_evidence_links and information_gain must include an entry for EVERY entity listed in global_scene.physical_entities. Do not skip any entities.",
            "UNCERTAIN OBSERVATIONS: For BOTH videos independently, identify observations that are genuinely ambiguous (observed evidence, multiple plausible hypotheses, confidence for each hypothesis, missing evidence).",
            "MISSING INFORMATION: List attributes that cannot be determined from each individual video (existence, target_category, spatial_distance, surface_attribute, motion_trend) and whether they can be recovered after combining both.",
            "CROSS-MODAL EVIDENCE LINKS: Jointly analyze both videos. For EVERY entity listed in global_scene.physical_entities, identify evidence shared, unique to Video 1, unique to Video 2, and exactly how one improves understanding of the other.",
            "INFORMATION GAIN: For EVERY entity listed in global_scene.physical_entities, explain what Video 1 alone can/cannot determine, what Video 2 alone can/cannot determine, and what the combination additionally reveals. Each entity must have exactly one information_gain entry. Rate gain as low/medium/high.",
            "AMBIGUITY RESOLUTION: Actively search for ambiguities in BOTH directions. Direction A (video1_resolves_video2): Does Video 1 resolve something Video 2 cannot determine alone? Direction B (video2_resolves_video1): Does Video 2 resolve something Video 1 cannot determine alone? For example, an event camera may reveal rapid motion that a depth camera cannot capture; an IR camera may reveal heat signatures that RGB cannot distinguish. You MUST check both directions independently and report any valid events found in either.",
            "QUESTION-WORTHINESS: Estimate usefulness for generating difficult reasoning questions. Provide difficulty, qa_potential, and supported_question_types.",
            "FRAME-BY-FRAME ANALYSIS: Describe newly appearing/disappearing entities, motion/interaction changes, and newly introduced/resolved uncertainty across frames.",
            "Return ONLY valid JSON with this exact structure:",
            "{",
            f'  "schema_version": "{CAPTION_SCHEMA_VERSION}",',
            '  "global_scene": {',
            '    "scene_summary": "Detailed physical-scene summary independent of sensor artifacts.",',
            '    "physical_entities": [',
            '      {"entity_id": "stable_snake_case_id", "category": "...", "appearance_or_state": "...", "location": "...", "motion_or_action": "...", "spatial_relations": ["..."]}',
            '    ],',
            '    "environment": "Objective environment or recording condition if evident.",',
            '    "temporal_progression": "Dense chronological account of how the scene/action changes across supplied frames."',
            '  },',
            '  "video1_analysis": {',
            f'    "modality": "{task.helper_modality}",',
            '    "detailed_caption": "Detailed caption using only Video 1 (LEFT).",',
            '    "observable_facts": ["Concrete fact 1", "Concrete fact 2", "Concrete fact 3"],',
            '    "sensor_specific_cues": ["Imaging/measurement cues from this modality."],',
            '    "sensor_limitations": ["Specific limitations that affect interpretation."],',
            '    "uncertain_observations": [{"observed_evidence": "...", "hypotheses": [{"hypothesis": "...", "confidence": "high|medium|low"}], "missing_evidence": "..."}],',
            '    "missing_key_attributes": [{"attribute_type": "existence|target_category|spatial_distance|surface_attribute|motion_trend", "missing_attribute": "...", "why_missing": "...", "recoverable_from": ["video1_analysis.observable_facts"]}]',
            '  },',
            '  "video2_analysis": {',
            f'    "modality": "{task.victim_modality}",',
            '    "detailed_caption": "Detailed caption using only Video 2 (RIGHT).",',
            '    "observable_facts": ["Concrete fact 1", "Concrete fact 2", "Concrete fact 3"],',
            '    "sensor_specific_cues": ["Imaging/measurement cues from this modality."],',
            '    "sensor_limitations": ["Specific limitations that affect interpretation."],',
            '    "uncertain_observations": [{"observed_evidence": "...", "hypotheses": [{"hypothesis": "...", "confidence": "high|medium|low"}], "missing_evidence": "..."}],',
            '    "missing_key_attributes": [{"attribute_type": "existence|target_category|spatial_distance|surface_attribute|motion_trend", "missing_attribute": "...", "why_missing": "...", "recoverable_from": ["video2_analysis.observable_facts"]}]',
            '  },',
            '  "cross_modal_evidence_links": [',
            '    {"entity_id": "...", "shared_evidence": "...", "unique_to_video1": "...", "unique_to_video2": "...", "how_video1_improves_video2": "...", "how_video2_improves_video1": "..."}',
            '  ],',
            '  "information_gain": [',
            '    {"entity_id": "...", "gain_rating": "low|medium|high", "video1_can_determine": ["..."], "video1_cannot_determine": ["..."], "video2_can_determine": ["..."], "video2_cannot_determine": ["..."], "fusion_additionally_reveals": ["..."]}',
            '  ],',
            '  "ambiguity_events": [',
            '    {',
            '      "target_entity": "entity_id from global_scene.physical_entities",',
            '      "approx_time_range": "early sampled frame|middle sampled frame|late sampled frame|specific frame names",',
            '      "direction": "video1_resolves_video2|video2_resolves_video1",',
            '      "ambiguous_video": "video1|video2",',
            '      "resolving_video": "video2|video1",',
            '      "low_confidence_observation": "What the ambiguous video shows by itself.",',
            '      "why_ambiguous_video_cannot_resolve": "Specific reason the ambiguous video cannot uniquely interpret the cue.",',
            '      "candidate_hypotheses": [{"hypothesis": "hypothesis 1...", "support_from_victim": "..."}, {"hypothesis": "hypothesis 2...", "support_from_victim": "..."}],',
            '      "resolving_discriminative_evidence": "Concrete cue from the resolving video that eliminates at least one hypothesis.",',
            '      "eliminated_hypotheses": [{"hypothesis": "...", "why_eliminated": "..."}],',
            '      "fusion_conclusion": "Final physical fact after combining both modalities.",',
            '      "missing_attribute_type": "existence|target_category|spatial_distance|surface_attribute|motion_trend",',
            '      "question_worthiness": {"difficulty": "easy|medium|hard", "qa_potential": "high|medium|low", "supported_question_types": ["object_identity|attribute_reasoning|temporal_reasoning|spatial_reasoning|interaction_reasoning|cross_modal_reasoning|counterfactual_reasoning"]}',
            '    }',
            '  ],',
            '  "frame_by_frame_analysis": [',
            '    {"frame_key": "frame_000000", "newly_appearing_entities": ["..."], "disappearing_entities": ["..."], "motion_changes": "...", "interaction_changes": "...", "newly_introduced_uncertainty": "...", "resolved_uncertainty": "..."}',
            '  ],',
            '  "rejected_observations": [',
            '    {"observation": "...", "reason": "Why this was not a valid ambiguity_event."}',
            '  ]',
            '}',
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
        if not hypotheses:
            raise ValueError(f"{field}[{index}].hypotheses must not be empty")
        for hyp_index, hyp in enumerate(hypotheses, start=1):
            if not isinstance(hyp, dict):
                raise ValueError(f"{field}[{index}].hypotheses[{hyp_index}] must be an object")
            _require_string(hyp.get("hypothesis"), f"{field}[{index}].hypotheses[{hyp_index}].hypothesis")
            conf = hyp.get("confidence")
            if conf not in ALLOWED_GAIN_RATINGS:
                raise ValueError(f"{field}[{index}].hypotheses[{hyp_index}].confidence must be high, medium, or low")

def _validate_missing_key_attributes(values: Any, field: str) -> None:
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
        recoverable_from = item.get("recoverable_from")
        if not isinstance(recoverable_from, list) or not recoverable_from:
            raise ValueError(f"{field}[{index}].recoverable_from must be a non-empty list")

def _validate_video_analysis(parsed: dict[str, Any], field: str) -> None:
    analysis = _require_object(parsed.get(field), field)
    _require_string(analysis.get("modality"), f"{field}.modality")
    detailed_caption = _validate_min_words(
        analysis.get("detailed_caption"),
        f"{field}.detailed_caption",
        MIN_DETAILED_CAPTION_WORDS,
    )
    if FORBIDDEN_GLOBAL_SCENE_WORDS.search(detailed_caption):
        raise ValueError(f"{field}.detailed_caption contains forbidden sensor-quality words")
    _validate_no_generic_sensor_explanation(detailed_caption, f"{field}.detailed_caption")
    for key in ("observable_facts", "sensor_specific_cues", "sensor_limitations"):
        values = _require_list(analysis.get(key), f"{field}.{key}")
        if not values:
            raise ValueError(f"{field}.{key} must not be empty")
        if key == "observable_facts" and len(values) < MIN_OBSERVABLE_FACTS:
            raise ValueError(f"{field}.{key} must contain at least {MIN_OBSERVABLE_FACTS} facts")
        for value_index, value in enumerate(values, start=1):
            text = _require_string(value, f"{field}.{key}[{value_index}]")
            _validate_no_generic_sensor_explanation(text, f"{field}.{key}[{value_index}]")
    _validate_uncertain_observations(analysis.get("uncertain_observations"), f"{field}.uncertain_observations")
    _validate_missing_key_attributes(analysis.get("missing_key_attributes"), f"{field}.missing_key_attributes")

def _validate_cross_modal_evidence_links(values: Any, entity_ids: set[str], field: str) -> None:
    links = _require_list(values, field)
    if not links:
        raise ValueError(f"{field} must not be empty")
    seen_entities = set()
    for index, item in enumerate(links, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{field}[{index}] must be an object")
        entity_id = _require_string(item.get("entity_id"), f"{field}[{index}].entity_id")
        if entity_id not in entity_ids:
            raise ValueError(f"{field}[{index}].entity_id must match a global_scene entity_id")
        seen_entities.add(entity_id)
        for key in ("shared_evidence", "unique_to_video1", "unique_to_video2", "how_video1_improves_video2", "how_video2_improves_video1"):
            _require_string(item.get(key), f"{field}[{index}].{key}")
    missing_entities = entity_ids - seen_entities
    if missing_entities:
        raise ValueError(f"{field} is missing entries for these entities: {', '.join(sorted(missing_entities))}")

def _validate_information_gain(values: Any, entity_ids: set[str], field: str) -> None:
    gains = _require_list(values, field)
    if not gains:
        raise ValueError(f"{field} must not be empty")
    seen_entities = set()
    for index, item in enumerate(gains, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{field}[{index}] must be an object")
        entity_id = _require_string(item.get("entity_id"), f"{field}[{index}].entity_id")
        if entity_id not in entity_ids:
            raise ValueError(f"{field}[{index}].entity_id must match a global_scene entity_id")
        seen_entities.add(entity_id)
        for key in ("video1_can_determine", "video1_cannot_determine", "video2_can_determine", "video2_cannot_determine", "fusion_additionally_reveals"):
            _require_list(item.get(key), f"{field}[{index}].{key}")
        rating = item.get("gain_rating")
        if rating not in ALLOWED_GAIN_RATINGS:
            raise ValueError(f"{field}[{index}].gain_rating must be high, medium, or low")
    missing_entities = entity_ids - seen_entities
    if missing_entities:
        raise ValueError(f"{field} is missing entries for these entities: {', '.join(sorted(missing_entities))}")

def _validate_question_worthiness(value: Any, field: str) -> None:
    qw = _require_object(value, field)
    _require_string(qw.get("difficulty"), f"{field}.difficulty")
    qa_pot = qw.get("qa_potential")
    if qa_pot not in ALLOWED_QA_POTENTIAL:
        raise ValueError(f"{field}.qa_potential must be high, medium, or low")
    q_types = _require_list(qw.get("supported_question_types"), f"{field}.supported_question_types")
    for qt in q_types:
        if qt not in ALLOWED_QUESTION_TYPES:
            raise ValueError(f"{field}.supported_question_types contains invalid type: {qt}")

def _validate_frame_by_frame_analysis(values: Any, field: str) -> None:
    frames = _require_list(values, field)
    if not frames:
        raise ValueError(f"{field} must not be empty")
    for index, item in enumerate(frames, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{field}[{index}] must be an object")
        _require_string(item.get("frame_key"), f"{field}[{index}].frame_key")
        _require_list(item.get("newly_appearing_entities"), f"{field}[{index}].newly_appearing_entities")
        _require_list(item.get("disappearing_entities"), f"{field}[{index}].disappearing_entities")
        for key in ("motion_changes", "interaction_changes", "newly_introduced_uncertainty", "resolved_uncertainty"):
            _require_string(item.get(key), f"{field}[{index}].{key}")

def _validate_caption_schema(parsed: dict[str, Any]) -> dict[str, Any]:
    required_fields = (
        "schema_version",
        "global_scene",
        "video1_analysis",
        "video2_analysis",
        "cross_modal_evidence_links",
        "information_gain",
        "ambiguity_events",
        "frame_by_frame_analysis",
        "rejected_observations",
    )
    missing = [field for field in required_fields if field not in parsed]
    if missing:
        raise ValueError(f"Gemini response missing required caption field(s): {', '.join(missing)}")
    if parsed["schema_version"] != CAPTION_SCHEMA_VERSION:
        raise ValueError(
            f"Gemini response schema_version must be {CAPTION_SCHEMA_VERSION!r}, "
            f"got {parsed['schema_version']!r}"
        )
    global_scene = _require_object(parsed["global_scene"], "global_scene")
    scene_summary = _validate_min_words(global_scene.get("scene_summary"), "global_scene.scene_summary", MIN_SCENE_SUMMARY_WORDS)
    if FORBIDDEN_GLOBAL_SCENE_WORDS.search(scene_summary):
        raise ValueError("global_scene.scene_summary contains forbidden sensor-quality words")
    _validate_no_generic_sensor_explanation(scene_summary, "global_scene.scene_summary")
    _require_string(global_scene.get("environment"), "global_scene.environment")
    temporal_progression = _validate_min_words(global_scene.get("temporal_progression"), "global_scene.temporal_progression", MIN_FRAME_DETAIL_WORDS)
    if FORBIDDEN_GLOBAL_SCENE_WORDS.search(temporal_progression):
        raise ValueError("global_scene.temporal_progression contains forbidden sensor-quality words")
    _validate_no_generic_sensor_explanation(temporal_progression, "global_scene.temporal_progression")
    physical_entities = _require_list(global_scene.get("physical_entities"), "global_scene.physical_entities")
    if not physical_entities:
        raise ValueError("global_scene.physical_entities must not be empty")
    entity_ids: set[str] = set()
    for index, entity in enumerate(physical_entities, start=1):
        if not isinstance(entity, dict):
            raise ValueError(f"global_scene.physical_entities[{index}] must be an object")
        entity_id = _require_string(entity.get("entity_id"), f"global_scene.physical_entities[{index}].entity_id")
        entity_ids.add(entity_id)
        for key in ("category", "appearance_or_state", "location", "motion_or_action"):
            _require_string(entity.get(key), f"global_scene.physical_entities[{index}].{key}")
        _require_list(entity.get("spatial_relations"), f"global_scene.physical_entities[{index}].spatial_relations")

    _validate_video_analysis(parsed, "video1_analysis")
    _validate_video_analysis(parsed, "video2_analysis")
    
    _validate_cross_modal_evidence_links(parsed.get("cross_modal_evidence_links"), entity_ids, "cross_modal_evidence_links")
    _validate_information_gain(parsed.get("information_gain"), entity_ids, "information_gain")

    ambiguity_events = _require_list(parsed["ambiguity_events"], "ambiguity_events")
    for index, event in enumerate(ambiguity_events, start=1):
        if not isinstance(event, dict):
            raise ValueError(f"ambiguity_events[{index}] must be an object")
        target_entity = _require_string(event.get("target_entity"), f"ambiguity_events[{index}].target_entity")
        if target_entity not in entity_ids:
            raise ValueError(f"ambiguity_events[{index}].target_entity must match a global_scene entity_id")
        direction = event.get("direction")
        if direction not in ALLOWED_AMBIGUITY_DIRECTIONS:
            raise ValueError(f"ambiguity_events[{index}].direction must be video1_resolves_video2 or video2_resolves_video1")
        for key in (
            "approx_time_range",
            "ambiguous_video",
            "resolving_video",
            "low_confidence_observation",
            "why_ambiguous_video_cannot_resolve",
            "resolving_discriminative_evidence",
            "fusion_conclusion",
        ):
            _require_string(event.get(key), f"ambiguity_events[{index}].{key}")
        missing_type = event.get("missing_attribute_type")
        if missing_type not in ALLOWED_MISSING_ATTRIBUTE_TYPES:
            raise ValueError(
                f"ambiguity_events[{index}].missing_attribute_type must be one of "
                f"{sorted(ALLOWED_MISSING_ATTRIBUTE_TYPES)}, got {missing_type!r}"
            )
        _validate_question_worthiness(event.get("question_worthiness"), f"ambiguity_events[{index}].question_worthiness")
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
    
    _validate_frame_by_frame_analysis(parsed.get("frame_by_frame_analysis"), "frame_by_frame_analysis")
    _require_list(parsed["rejected_observations"], "rejected_observations")
    return parsed


async def _call_gemini_caption(client, task: CaptionTask, model_name: str, max_retries: int) -> dict[str, Any]:
    _ensure_composite_frames(task)
    encoded = _encode_images(task.composite_frames)
    if not encoded:
        raise ValueError("No composite frames found for Gemini call")
    base_contents = build_image_parts(encoded) + [_build_caption_prompt(task)]
    contents = base_contents
    for attempt in range(1, max_retries + 1):
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=model_name,
                contents=contents,
            )
            return _validate_caption_schema(_parse_json_response(response.text))
        except Exception as exc:
            if attempt == max_retries:
                raise
            wait_seconds = 30 * attempt if "429" in str(exc) or "quota" in str(exc).lower() else 2 * attempt
            print(
                f"    Caption Gemini call failed on attempt {attempt}/{max_retries}; "
                f"retrying in {wait_seconds}s: {exc}"
            )
            await asyncio.sleep(wait_seconds)
            # Error-guided retry: tell the model exactly what went wrong
            error_feedback = (
                f"Your previous response was REJECTED due to this validation error: [{exc}]. "
                f"Fix ONLY that specific issue and return the corrected full JSON."
            )
            contents = base_contents + [error_feedback]
    raise RuntimeError("Gemini caption call failed")


def _task_to_item(task: CaptionTask, status: str, caption: dict[str, Any] | None = None, reason: str | None = None) -> dict[str, Any]:
    return {
        "caption_id": task.caption_id,
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
        "caption": caption,
    }


def _template_caption(task: CaptionTask) -> dict[str, Any]:
    target_entity = "unresolved_target"
    return {
        "schema_version": CAPTION_SCHEMA_VERSION,
        "global_scene": {
            "scene_summary": "Template mode placeholder; Gemini was not called.",
            "physical_entities": [
                {
                    "entity_id": target_entity,
                    "category": "unknown",
                    "appearance_or_state": "Template mode placeholder.",
                    "location": "unknown",
                    "motion_or_action": "unknown",
                    "spatial_relations": [],
                }
            ],
            "environment": "unknown",
            "temporal_progression": "Template mode placeholder; no visual reasoning was performed.",
        },
        "video1_analysis": {
            "modality": task.helper_modality,
            "detailed_caption": "Template mode placeholder; Gemini was not called.",
            "observable_facts": ["Template mode placeholder.", "Fact 2", "Fact 3"],
            "sensor_specific_cues": ["Template mode placeholder."],
            "sensor_limitations": ["Template mode placeholder."],
            "uncertain_observations": [
                {
                    "observed_evidence": "Template placeholder",
                    "hypotheses": [{"hypothesis": "Placeholder", "confidence": "low"}],
                    "missing_evidence": "Placeholder"
                }
            ],
            "missing_key_attributes": [],
        },
        "video2_analysis": {
            "modality": task.victim_modality,
            "detailed_caption": "Template mode placeholder; Gemini was not called.",
            "observable_facts": ["Template mode placeholder.", "Fact 2", "Fact 3"],
            "sensor_specific_cues": ["Template mode placeholder."],
            "sensor_limitations": ["Template mode placeholder."],
            "uncertain_observations": [
                {
                    "observed_evidence": "Template placeholder",
                    "hypotheses": [{"hypothesis": "Placeholder", "confidence": "low"}],
                    "missing_evidence": "Placeholder"
                }
            ],
            "missing_key_attributes": [],
        },
        "cross_modal_evidence_links": [
            {
                "entity_id": target_entity,
                "shared_evidence": "Placeholder",
                "unique_to_video1": "Placeholder",
                "unique_to_video2": "Placeholder",
                "how_video1_improves_video2": "Placeholder",
                "how_video2_improves_video1": "Placeholder",
            }
        ],
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
        "ambiguity_events": [],
        "frame_by_frame_analysis": [
            {
                "frame_key": path.stem,
                "newly_appearing_entities": [],
                "disappearing_entities": [],
                "motion_changes": "Placeholder",
                "interaction_changes": "Placeholder",
                "newly_introduced_uncertainty": "Placeholder",
                "resolved_uncertainty": "Placeholder"
            }
            for path in task.composite_frames
        ],
        "rejected_observations": [
            {"observation": "", "reason": "template mode; Gemini was not called"}
        ],
    }


def _build_output_payload(
    input_path: Path,
    dataset_root: Path,
    composite_root: Path,
    output_path: Path,
    model_name: str,
    generation_mode: str,
    num_frames: int,
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
            "num_frames": num_frames,
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
    return list(items), list(skipped)


async def run_caption_pipeline_async(
    input_path: Path,
    output_path: Path,
    dataset_root: Path,
    composite_root: Path,
    model_name: str,
    generation_mode: str,
    num_frames: int,
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
    tasks, skipped = build_caption_tasks(
        input_path=input_path,
        dataset_root=dataset_root,
        composite_root=composite_root,
        num_frames=num_frames,
        allowed_pairs=allowed_pairs,
        allowed_directions=allowed_directions,
        limit=limit,
        limit_scenes=limit_scenes,
        limit_scene_folders=limit_scene_folders,
        allowed_sides=allowed_sides,
        write_composites=False,
    )
    existing_items, existing_skipped = _load_resume(output_path) if resume else ([], [])
    items = existing_items
    skipped = existing_skipped + skipped
    existing_ids = {str(item.get("caption_id")) for item in items if item.get("caption_id")}
    pending_tasks = [task for task in tasks if task.caption_id not in existing_ids]

    client = create_gemini_client() if generation_mode == "gemini" else None
    gemini_calls = 0
    checkpoint_counter = 0

    print(
        f"Generating cross-modal captions: {len(tasks)} planned item(s), "
        f"{len(pending_tasks)} pending, mode={generation_mode}, model={model_name}."
    )

    def save_checkpoint() -> None:
        _save_json(
            _build_output_payload(
                input_path=input_path,
                dataset_root=dataset_root,
                composite_root=composite_root,
                output_path=output_path,
                model_name=model_name,
                generation_mode=generation_mode,
                num_frames=num_frames,
                items=items,
                skipped=skipped,
                planned_total=len(tasks),
                gemini_calls=gemini_calls,
            ),
            output_path,
        )

    for index, task in enumerate(pending_tasks, start=1):
        print(
            f"  Caption item [{index}/{len(pending_tasks)}] "
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
                caption = _template_caption(task)
                status = "template"
            items.append(_task_to_item(task, status=status, caption=caption))
        except Exception as exc:
            exc_str = str(exc).lower()
            if "429" in exc_str or "quota" in exc_str:
                print(f"FATAL: Quota exhausted or rate limit hit. Stopping execution to preserve state: {exc}")
                break
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
            print(f"WARNING: Caption generation failed for {task.caption_id}: {exc}")

        checkpoint_counter += 1
        if checkpoint_every > 0 and checkpoint_counter >= checkpoint_every:
            checkpoint_counter = 0
            save_checkpoint()
        if generation_mode == "gemini" and delay_between_calls > 0 and index < len(pending_tasks):
            await asyncio.sleep(delay_between_calls)

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
    num_frames: int = 6,
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
            num_frames=num_frames,
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
        choices=("template", "gemini"),
        default="template",
        help="Use template to build composite frames without calling Gemini.",
    )
    parser.add_argument("--num-frames", type=int, default=6)
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
        help="Comma-separated helper->victim directions such as rgb->depth,event->ir. Defaults to both directions.",
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
    args = _build_arg_parser().parse_args()
    run_caption_pipeline(
        input_path=args.input,
        output_path=args.output,
        dataset_root=args.dataset_root,
        composite_root=args.composite_root,
        model_name=args.model_name,
        generation_mode=args.generation_mode,
        num_frames=max(1, args.num_frames),
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
