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
CAPTION_SCHEMA_VERSION = "cross_modal_disambiguation_caption_v2"
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
    context_modality: str
    decisive_modality: str
    context_frame_dir: Path
    decisive_frame_dir: Path
    context_frames: tuple[Path, ...]
    decisive_frames: tuple[Path, ...]
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
    context_dirs: dict[str, Path],
    decisive_dirs: dict[str, Path],
) -> list[tuple[str, Path, Path]]:
    sides = [side for side in SIDE_ORDER if side in context_dirs and side in decisive_dirs]
    sides.extend(
        sorted(
            side
            for side in set(context_dirs) & set(decisive_dirs)
            if side not in SIDE_ORDER
        )
    )
    return [(side, context_dirs[side], decisive_dirs[side]) for side in sides]


def _select_paired_frames(
    context_dir: Path,
    decisive_dir: Path,
    context_modality: str,
    decisive_modality: str,
    num_frames: int,
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    context_by_index = _frames_by_index(context_dir, context_modality)
    decisive_by_index = _frames_by_index(decisive_dir, decisive_modality)
    shared_indexes = sorted(set(context_by_index) & set(decisive_by_index))
    selected_indexes = _evenly_sample(shared_indexes, num_frames)
    return (
        tuple(context_by_index[index] for index in selected_indexes),
        tuple(decisive_by_index[index] for index in selected_indexes),
    )


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")


def _load_font(size: int = 28) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _resize_to_height(image: Image.Image, height: int) -> Image.Image:
    if image.height == height:
        return image
    width = max(1, round(image.width * height / image.height))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _draw_label(image: Image.Image, label: str) -> None:
    draw = ImageDraw.Draw(image)
    font = _load_font()
    padding = 10
    bbox = draw.textbbox((0, 0), label, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    draw.rectangle(
        (0, 0, width + padding * 2, height + padding * 2),
        fill=(0, 0, 0),
    )
    draw.text((padding, padding), label, fill=(255, 255, 255), font=font)


def _compose_frame(
    context_frame: Path,
    decisive_frame: Path,
    context_modality: str,
    decisive_modality: str,
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
        _draw_label(canvas, f"LEFT: {context_modality.upper()} context")
        right_label = Image.new("RGB", (right.width, target_height), (0, 0, 0))
        right_label.paste(right, (0, 0))
        _draw_label(right_label, f"RIGHT: {decisive_modality.upper()} decisive")
        canvas.paste(right_label, (left.width, 0))
        canvas.save(output_path)
    return output_path


def _caption_id(segment_id: str, side: str, context_modality: str, decisive_modality: str) -> str:
    return "__".join(
        [
            _safe_name(segment_id).lower(),
            _safe_name(side).lower(),
            f"{context_modality}_context",
            f"{decisive_modality}_decisive",
        ]
    )


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
    directions = [(first, second), (second, first)]
    if allowed is None:
        return directions
    return [direction for direction in directions if direction in allowed]


def build_caption_tasks(
    input_path: Path,
    dataset_root: Path,
    composite_root: Path,
    num_frames: int,
    allowed_pairs: set[tuple[str, str]] | None = None,
    allowed_directions: set[tuple[str, str]] | None = None,
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
            for context_modality, decisive_modality in _directions_for_pair(pair, allowed_directions):
                context_dirs = _load_frame_dirs(dataset_root, split_dir, segment_name, context_modality)
                decisive_dirs = _load_frame_dirs(dataset_root, split_dir, segment_name, decisive_modality)
                if not context_dirs or not decisive_dirs:
                    skipped.append(
                        {
                            "segment_id": segment_id,
                            "split_dir": split_dir,
                            "segment_name": segment_name,
                            "context_modality": context_modality,
                            "decisive_modality": decisive_modality,
                            "reason": "missing frame cache directory",
                        }
                    )
                    continue
                for side, context_dir, decisive_dir in _pair_frame_dirs(context_dirs, decisive_dirs):
                    context_frames, decisive_frames = _select_paired_frames(
                        context_dir,
                        decisive_dir,
                        context_modality,
                        decisive_modality,
                        num_frames,
                    )
                    if not context_frames:
                        skipped.append(
                            {
                                "segment_id": segment_id,
                                "side": side,
                                "context_modality": context_modality,
                                "decisive_modality": decisive_modality,
                                "context_frame_dir": context_dir.as_posix(),
                                "decisive_frame_dir": decisive_dir.as_posix(),
                                "reason": "no shared frame indexes",
                            }
                        )
                        continue
                    caption_id = _caption_id(str(segment_id), side, context_modality, decisive_modality)
                    output_dir = (
                        composite_root
                        / _safe_name(split_dir)
                        / _safe_name(segment_name)
                        / _safe_name(side)
                        / f"{context_modality}_context__{decisive_modality}_decisive"
                    )
                    composite_frames: list[Path] = []
                    for index, (context_frame, decisive_frame) in enumerate(zip(context_frames, decisive_frames), start=1):
                        frame_number = _frame_index(context_frame)
                        suffix = f"{frame_number:06d}" if frame_number is not None else f"{index:03d}"
                        output_path = output_dir / f"frame_{suffix}.png"
                        if write_composites:
                            _compose_frame(
                                context_frame,
                                decisive_frame,
                                context_modality,
                                decisive_modality,
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
                            context_modality=context_modality,
                            decisive_modality=decisive_modality,
                            context_frame_dir=context_dir,
                            decisive_frame_dir=decisive_dir,
                            context_frames=context_frames,
                            decisive_frames=decisive_frames,
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
    return "\n".join(
        [
            "You are a cross-modal video captioning assistant.",
            "You will receive side-by-side composite frames from one aligned video segment.",
            f"Left side is the context modality: {task.context_modality}.",
            f"Right side is the decisive modality: {task.decisive_modality}.",
            "The goal is not ordinary captioning. Find observations where the decisive modality alone contains an ambiguous cue, and the context modality disambiguates it.",
            "First describe what is visible from each side alone. Then use those one-sided captions to decide whether a true cross-modal disambiguation exists.",
            "Only use cues visible in the supplied frames. Do not invent objects, actions, text, or intentions.",
            "The context and decisive observations must refer to the same scene, object, action, or event.",
            "Reject observations if the decisive side is already unambiguous, if the context side does not help, or if the two sides do not refer to the same target.",
            "Reject observations if the decisive-only caption already contains the final disambiguated fact or enough information to answer the likely QA target.",
            "Reject observations where the context modality only confirms, repeats, or labels a fact already directly visible in the decisive modality.",
            "A valid observation must have at least two plausible interpretations from the decisive side alone, and the context side must remove that ambiguity.",
            "The context_only_caption and decisive_only_caption are mandatory because they will be used later to filter false disambiguation cases.",
            "Write the one-sided captions before deciding the cross-modal caption. If the decisive-only caption already states the final fact, reject that observation.",
            "Prefer high-entropy facts about object identity, action phase, spatial relation, motion, interaction, location, or text/semantic identity.",
            "Avoid yes/no facts, simple counting, single-color answers, and anomaly/non-common-object judgments.",
            "Return ONLY valid JSON with this exact structure:",
            "{",
            f'  "schema_version": "{CAPTION_SCHEMA_VERSION}",',
            f'  "context_only_caption": "Caption using only the LEFT {task.context_modality} context side. Do not mention the right side.",',
            f'  "decisive_only_caption": "Caption using only the RIGHT {task.decisive_modality} decisive side. Do not mention the left side.",',
            '  "segment_caption": "Brief caption of the shared scene/action across both modalities.",',
            '  "decisive_only_ambiguous_observations": [',
            "    {",
            '      "decisive_observation": "What the decisive modality shows by itself.",',
            '      "why_ambiguous_without_context": "Why that cue cannot be uniquely interpreted from the decisive side alone.",',
            '      "possible_interpretations": ["...", "..."],',
            '      "context_disambiguating_cue": "Specific cue from the context modality that resolves the ambiguity.",',
            '      "disambiguated_fact": "The fact that becomes clear only after combining both modalities.",',
            '      "decisive_alone_contains_final_fact": false,',
            '      "context_role": "disambiguates|confirms_only|irrelevant",',
            '      "qa_potential": "high|medium|low"',
            "    }",
            "  ],",
            '  "cross_modal_caption": "A concise caption focused on context-assisted disambiguation.",',
            '  "rejected_observations": [',
            '    {"observation": "...", "reason": "..."}',
            "  ]",
            "}",
            "Only include an item in decisive_only_ambiguous_observations when decisive_alone_contains_final_fact is false and context_role is disambiguates.",
            "If the context_only_caption and decisive_only_caption both independently support the same final fact, reject it as confirms_only.",
            "If no valid ambiguity/disambiguation observation exists, return an empty decisive_only_ambiguous_observations list and explain why in rejected_observations.",
            f"Segment: {task.segment_id}; side: {task.side}.",
            f"Composite frames ({len(task.composite_frames)} images): {frame_names}",
        ]
    )


def _build_batch_caption_prompt(tasks: list[CaptionTask]) -> str:
    item_specs: list[str] = []
    image_index = 1
    for task in tasks:
        frame_range = f"images {image_index}-{image_index + len(task.composite_frames) - 1}"
        image_index += len(task.composite_frames)
        item_specs.append(
            "\n".join(
                [
                    "{",
                    f'  "caption_id": "{task.caption_id}",',
                    f'  "segment_id": "{task.segment_id}",',
                    f'  "side": "{task.side}",',
                    f'  "context_modality": "{task.context_modality}",',
                    f'  "decisive_modality": "{task.decisive_modality}",',
                    f'  "composite_images": "{frame_range}",',
                    f'  "frame_names": "{", ".join(path.name for path in task.composite_frames)}"',
                    "}",
                ]
            )
        )
    return "\n".join(
        [
            "You are a cross-modal video captioning assistant.",
            "You will receive side-by-side composite frames for multiple independent caption items.",
            "For every item, the left side of each composite image is the context modality and the right side is the decisive modality.",
            "Process each item independently. Do not mix observations across caption_id values.",
            "The goal is not ordinary captioning. Find observations where the decisive modality alone contains an ambiguous cue, and the context modality disambiguates it.",
            "First describe what is visible from each side alone. Then use those one-sided captions to decide whether a true cross-modal disambiguation exists.",
            "Only use cues visible in the supplied frames. Do not invent objects, actions, text, or intentions.",
            "The context and decisive observations must refer to the same scene, object, action, or event.",
            "Reject observations if the decisive side is already unambiguous, if the context side does not help, or if the two sides do not refer to the same target.",
            "Reject observations if the decisive-only caption already contains the final disambiguated fact or enough information to answer the likely QA target.",
            "Reject observations where the context modality only confirms, repeats, or labels a fact already directly visible in the decisive modality.",
            "A valid observation must have at least two plausible interpretations from the decisive side alone, and the context side must remove that ambiguity.",
            "The context_only_caption and decisive_only_caption are mandatory because they will be used later to filter false disambiguation cases.",
            "Prefer high-entropy facts about object identity, action phase, spatial relation, motion, interaction, location, or text/semantic identity.",
            "Avoid yes/no facts, simple counting, single-color answers, and anomaly/non-common-object judgments.",
            "Return ONLY valid JSON with this exact top-level structure:",
            "{",
            '  "items": [',
            "    {",
            '      "caption_id": "must exactly match one input caption_id",',
            f'      "schema_version": "{CAPTION_SCHEMA_VERSION}",',
            '      "context_only_caption": "Caption using only the LEFT context side. Do not mention the right side.",',
            '      "decisive_only_caption": "Caption using only the RIGHT decisive side. Do not mention the left side.",',
            '      "segment_caption": "Brief caption of the shared scene/action across both modalities.",',
            '      "decisive_only_ambiguous_observations": [',
            "        {",
            '          "decisive_observation": "What the decisive modality shows by itself.",',
            '          "why_ambiguous_without_context": "Why that cue cannot be uniquely interpreted from the decisive side alone.",',
            '          "possible_interpretations": ["...", "..."],',
            '          "context_disambiguating_cue": "Specific cue from the context modality that resolves the ambiguity.",',
            '          "disambiguated_fact": "The fact that becomes clear only after combining both modalities.",',
            '          "decisive_alone_contains_final_fact": false,',
            '          "context_role": "disambiguates|confirms_only|irrelevant",',
            '          "qa_potential": "high|medium|low"',
            "        }",
            "      ],",
            '      "cross_modal_caption": "A concise caption focused on context-assisted disambiguation.",',
            '      "rejected_observations": [',
            '        {"observation": "...", "reason": "..."}',
            "      ]",
            "    }",
            "  ]",
            "}",
            "Return exactly one item for every input caption_id, in the same order as listed below.",
            "Only include an observation in decisive_only_ambiguous_observations when decisive_alone_contains_final_fact is false and context_role is disambiguates.",
            "If the context_only_caption and decisive_only_caption both independently support the same final fact, reject it as confirms_only.",
            "If no valid ambiguity/disambiguation observation exists for an item, return an empty decisive_only_ambiguous_observations list and explain why in rejected_observations.",
            "Input items:",
            "\n\n".join(item_specs),
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


def _validate_caption_schema(parsed: dict[str, Any]) -> dict[str, Any]:
    required_fields = (
        "schema_version",
        "context_only_caption",
        "decisive_only_caption",
        "segment_caption",
        "decisive_only_ambiguous_observations",
        "cross_modal_caption",
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
    for field in ("context_only_caption", "decisive_only_caption", "segment_caption", "cross_modal_caption"):
        if not isinstance(parsed[field], str):
            raise ValueError(f"Gemini response field {field} must be a string")
    for field in ("decisive_only_ambiguous_observations", "rejected_observations"):
        if not isinstance(parsed[field], list):
            raise ValueError(f"Gemini response field {field} must be a list")
    for index, observation in enumerate(parsed["decisive_only_ambiguous_observations"], start=1):
        if not isinstance(observation, dict):
            raise ValueError(f"Ambiguous observation #{index} must be an object")
        if observation.get("decisive_alone_contains_final_fact") is not False:
            raise ValueError(
                "Ambiguous observations must set decisive_alone_contains_final_fact to false"
            )
        if observation.get("context_role") != "disambiguates":
            raise ValueError("Ambiguous observations must set context_role to disambiguates")
        interpretations = observation.get("possible_interpretations")
        if not isinstance(interpretations, list) or len(interpretations) < 2:
            raise ValueError("Ambiguous observations must include at least two possible_interpretations")
    return parsed


async def _call_gemini_caption(client, task: CaptionTask, model_name: str, max_retries: int) -> dict[str, Any]:
    encoded = _encode_images(task.composite_frames)
    if not encoded:
        raise ValueError("No composite frames found for Gemini call")
    contents = build_image_parts(encoded) + [_build_caption_prompt(task)]
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
    raise RuntimeError("Gemini caption call failed")


def _validate_batch_caption_schema(parsed: dict[str, Any], tasks: list[CaptionTask]) -> dict[str, dict[str, Any]]:
    items = parsed.get("items")
    if not isinstance(items, list):
        raise ValueError("Batch Gemini response must contain an items list")
    expected_ids = [task.caption_id for task in tasks]
    if len(items) != len(expected_ids):
        raise ValueError(f"Batch Gemini response must contain {len(expected_ids)} item(s), got {len(items)}")
    captions_by_id: dict[str, dict[str, Any]] = {}
    for expected_id, item in zip(expected_ids, items):
        if not isinstance(item, dict):
            raise ValueError("Each batch caption item must be an object")
        caption_id = item.get("caption_id")
        if caption_id != expected_id:
            raise ValueError(f"Expected caption_id {expected_id!r}, got {caption_id!r}")
        caption = dict(item)
        caption.pop("caption_id", None)
        captions_by_id[expected_id] = _validate_caption_schema(caption)
    return captions_by_id


def _chunk_tasks(tasks: list[CaptionTask], batch_size: int) -> list[list[CaptionTask]]:
    return [tasks[index : index + batch_size] for index in range(0, len(tasks), batch_size)]


async def _call_gemini_caption_batch(
    client,
    tasks: list[CaptionTask],
    model_name: str,
    max_retries: int,
) -> dict[str, dict[str, Any]]:
    encoded: list[str] = []
    for task in tasks:
        task_encoded = _encode_images(task.composite_frames)
        if not task_encoded:
            raise ValueError(f"No composite frames found for Gemini call: {task.caption_id}")
        encoded.extend(task_encoded)
    contents = build_image_parts(encoded) + [_build_batch_caption_prompt(tasks)]
    for attempt in range(1, max_retries + 1):
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=model_name,
                contents=contents,
            )
            return _validate_batch_caption_schema(_parse_json_response(response.text), tasks)
        except Exception as exc:
            if attempt == max_retries:
                raise
            wait_seconds = 30 * attempt if "429" in str(exc) or "quota" in str(exc).lower() else 2 * attempt
            print(
                f"    Batch caption Gemini call failed on attempt {attempt}/{max_retries}; "
                f"retrying in {wait_seconds}s: {exc}"
            )
            await asyncio.sleep(wait_seconds)
    raise RuntimeError("Batch Gemini caption call failed")


def _task_to_item(task: CaptionTask, status: str, caption: dict[str, Any] | None = None, reason: str | None = None) -> dict[str, Any]:
    return {
        "caption_id": task.caption_id,
        "segment_id": task.segment_id,
        "split_dir": task.split_dir,
        "segment_name": task.segment_name,
        "side": task.side,
        "context_modality": task.context_modality,
        "decisive_modality": task.decisive_modality,
        "context_frame_dir": task.context_frame_dir.as_posix(),
        "decisive_frame_dir": task.decisive_frame_dir.as_posix(),
        "context_frames": [path.as_posix() for path in task.context_frames],
        "decisive_frames": [path.as_posix() for path in task.decisive_frames],
        "composite_frames": [path.as_posix() for path in task.composite_frames],
        "status": status,
        "reason": reason,
        "caption": caption,
    }


def _build_output_payload(
    input_path: Path,
    dataset_root: Path,
    composite_root: Path,
    output_path: Path,
    model_name: str,
    generation_mode: str,
    num_frames: int,
    max_caption_items_per_gemini_call: int,
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
            "max_caption_items_per_gemini_call": max_caption_items_per_gemini_call,
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
    limit: int | None,
    limit_scenes: int | None,
    limit_scene_folders: int | None,
    max_retries: int,
    delay_between_calls: int,
    checkpoint_every: int,
    max_caption_items_per_gemini_call: int,
    resume: bool,
) -> Path:
    allowed_pairs = _parse_pairs(pairs)
    allowed_directions = _parse_pairs(directions)
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
        write_composites=True,
    )
    existing_items, existing_skipped = _load_resume(output_path) if resume else ([], [])
    items = existing_items
    skipped = existing_skipped + skipped
    existing_ids = {str(item.get("caption_id")) for item in items if item.get("caption_id")}
    pending_tasks = [task for task in tasks if task.caption_id not in existing_ids]
    batch_size = max(1, max_caption_items_per_gemini_call if generation_mode == "gemini" else 1)
    pending_batches = _chunk_tasks(pending_tasks, batch_size)

    client = create_gemini_client() if generation_mode == "gemini" else None
    gemini_calls = 0
    checkpoint_counter = 0

    print(
        f"Generating cross-modal captions: {len(tasks)} planned item(s), "
        f"{len(pending_tasks)} pending, {len(pending_batches)} batch(es), "
        f"mode={generation_mode}, model={model_name}, batch_size={batch_size}."
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
                max_caption_items_per_gemini_call=batch_size,
                items=items,
                skipped=skipped,
                planned_total=len(tasks),
                gemini_calls=gemini_calls,
            ),
            output_path,
        )

    for index, task_batch in enumerate(pending_batches, start=1):
        batch_label = ", ".join(task.caption_id for task in task_batch)
        print(
            f"  Caption batch [{index}/{len(pending_batches)}] "
            f"{len(task_batch)} item(s): {batch_label}"
        )
        try:
            if generation_mode == "gemini":
                assert client is not None
                if len(task_batch) == 1:
                    task = task_batch[0]
                    caption = await _call_gemini_caption(client, task, model_name, max_retries=max_retries)
                    captions_by_id = {task.caption_id: caption}
                else:
                    captions_by_id = await _call_gemini_caption_batch(
                        client,
                        task_batch,
                        model_name,
                        max_retries=max_retries,
                    )
                gemini_calls += 1
                status = "generated"
            else:
                captions_by_id = {
                    task.caption_id: {
                        "schema_version": CAPTION_SCHEMA_VERSION,
                        "context_only_caption": "",
                        "decisive_only_caption": "",
                        "segment_caption": "",
                        "decisive_only_ambiguous_observations": [],
                        "cross_modal_caption": "",
                        "rejected_observations": [
                            {"observation": "", "reason": "template mode; Gemini was not called"}
                        ],
                    }
                    for task in task_batch
                }
                status = "template"
            for task in task_batch:
                items.append(_task_to_item(task, status=status, caption=captions_by_id[task.caption_id]))
        except Exception as exc:
            for task in task_batch:
                skipped.append(
                    {
                        "caption_id": task.caption_id,
                        "segment_id": task.segment_id,
                        "side": task.side,
                        "context_modality": task.context_modality,
                        "decisive_modality": task.decisive_modality,
                        "reason": str(exc),
                    }
                )
            print(f"WARNING: Caption generation failed for batch containing {batch_label}: {exc}")

        checkpoint_counter += 1
        if checkpoint_every > 0 and checkpoint_counter >= checkpoint_every:
            checkpoint_counter = 0
            save_checkpoint()
        if generation_mode == "gemini" and delay_between_calls > 0 and index < len(pending_batches):
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
    limit: int | None = None,
    limit_scenes: int | None = None,
    limit_scene_folders: int | None = None,
    max_retries: int = 3,
    delay_between_calls: int = 5,
    checkpoint_every: int = 1,
    max_caption_items_per_gemini_call: int = 1,
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
            limit=limit,
            limit_scenes=limit_scenes,
            limit_scene_folders=limit_scene_folders,
            max_retries=max_retries,
            delay_between_calls=delay_between_calls,
            checkpoint_every=checkpoint_every,
            max_caption_items_per_gemini_call=max_caption_items_per_gemini_call,
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
        "--directions",
        default=None,
        help="Comma-separated context->decisive directions such as rgb->depth,event->ir. Defaults to both directions.",
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
    parser.add_argument(
        "--max-caption-items-per-gemini-call",
        type=int,
        default=1,
        help="Batch this many caption items into one Gemini request. Use 1 for safest parsing.",
    )
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
        limit=args.limit,
        limit_scenes=args.limit_scenes,
        limit_scene_folders=args.limit_scene_folders,
        max_retries=max(1, args.max_retries),
        delay_between_calls=max(0, args.delay_between_calls),
        checkpoint_every=max(0, args.checkpoint_every),
        max_caption_items_per_gemini_call=max(1, args.max_caption_items_per_gemini_call),
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
