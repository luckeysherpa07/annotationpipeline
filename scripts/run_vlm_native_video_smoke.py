#!/usr/bin/env python3
"""Run reproducible native-video smoke tests for local 8B VLMs."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from annotation_feature.qa_quality.benchmark import (
    DEFAULT_FRAME_CACHE_ROOT,
    DEFAULT_INPUT_PATH,
    InternVLFrameAnswerAdapter,
    QwenVLFrameAnswerAdapter,
    _internvl_image_to_tensor,
    load_valid_qa_items,
)

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None

try:
    from decord import VideoReader, cpu
except ImportError:
    VideoReader = None
    cpu = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from qwen_vl_utils import process_vision_info as process_qwen_vision_info
except ImportError:
    process_qwen_vision_info = None

try:
    from molmo_utils import process_vision_info as process_molmo_vision_info
except ImportError:
    process_molmo_vision_info = None

try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
except ImportError:
    AutoModelForImageTextToText = None
    AutoProcessor = None


DEFAULT_EXPERIMENT_DIR = Path("outputs/benchmarks/vlm_native_video")
DEFAULT_VIDEO_MANIFEST = DEFAULT_EXPERIMENT_DIR / "video_manifest.json"
DEFAULT_QWEN_VL_8B = Path("models/qwen/Qwen3-VL-8B-Instruct")
DEFAULT_INTERNVL_8B = Path("models/internvl/InternVL3-8B")
DEFAULT_MOLMO2_8B = Path("models/molmo2/Molmo2-8B")
DEFAULT_MODALITIES = ("rgb", "ir", "event", "depth")
GENERATION_CONFIG = {"max_new_tokens": 128, "do_sample": False}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".mpeg", ".mpg"}


class VideoPairLRUCache:
    """Keep a bounded number of prepared video pairs in CPU memory."""

    def __init__(self, capacity: int) -> None:
        self.capacity = max(0, int(capacity))
        self.values: OrderedDict[tuple[Any, ...], Any] = OrderedDict()

    def get(self, key: tuple[Any, ...]) -> tuple[Any, bool]:
        if key in self.values:
            self.values.move_to_end(key)
            return self.values[key], True
        return None, False

    def put(self, key: tuple[Any, ...], value: Any) -> Any:
        if self.capacity == 0:
            return value
        self.values[key] = value
        self.values.move_to_end(key)
        while len(self.values) > self.capacity:
            self.values.popitem(last=False)
        return value


def _video_cache_token(path: Path) -> tuple[str, int, int]:
    resolved = path.resolve()
    stat = resolved.stat()
    return resolved.as_posix(), int(stat.st_size), int(stat.st_mtime_ns)


def _model_input_device(model: Any) -> Any | None:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except Exception:
        return None


def _move_inputs_to_device(inputs: Any, device: Any) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    if isinstance(inputs, dict):
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
    return inputs


def _json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            converted = dict(row)
            for key, value in converted.items():
                if isinstance(value, (dict, list)):
                    converted[key] = json.dumps(value, ensure_ascii=False)
            writer.writerow(converted)


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")


def _segment_folder(item: dict[str, Any], dataset_root: Path) -> Path:
    parts = Path(str(item.get("pair_key", ""))).parts
    if parts and parts[0] == dataset_root.name:
        parts = parts[1:]
    if len(parts) > 1:
        parts = parts[:-1]
    return dataset_root.joinpath(*parts)


def _is_modality_video(path: Path, modality: str) -> bool:
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        return False
    stem = path.stem.lower()
    tokens = stem.split("_")
    if modality == "rgb" and "with_audio" in stem:
        return False
    return modality in tokens


def _video_side(path: Path) -> str:
    stem = path.stem.lower()
    if "no_light" in stem or "night" in stem:
        return "night"
    if "with_light" in stem or "day" in stem:
        return "day"
    return "unknown"


def resolve_video_pair(item: dict[str, Any], dataset_root: Path) -> tuple[Path, Path] | None:
    modality = str(item.get("modality", "")).strip().lower()
    folder = _segment_folder(item, dataset_root)
    if not folder.is_dir():
        return None
    candidates = sorted(
        path for path in folder.iterdir()
        if path.is_file() and _is_modality_video(path, modality)
    )
    day = [path for path in candidates if _video_side(path) == "day"]
    night = [path for path in candidates if _video_side(path) == "night"]
    if not day or not night:
        return None
    return day[0], night[0]


def probe_video(path: Path) -> dict[str, Any]:
    if cv2 is None:
        raise RuntimeError("OpenCV is required to inspect video metadata.")
    capture = cv2.VideoCapture(path.as_posix())
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()
    return {
        "path": path.as_posix(),
        "fps": round(fps, 6),
        "frame_count": frame_count,
        "duration_seconds": round(frame_count / fps, 6) if fps > 0 else 0.0,
        "width": width,
        "height": height,
        "size_bytes": path.stat().st_size,
    }


def build_video_manifest(
    input_path: Path,
    output_path: Path,
    dataset_root: Path,
    modalities: tuple[str, ...],
    items_per_modality: int | None,
) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)
    skipped_no_pair: dict[str, int] = defaultdict(int)
    metadata_cache: dict[str, dict[str, Any]] = {}
    wanted = set(modalities)

    for item in load_valid_qa_items(input_path):
        modality = str(item.get("modality", "")).lower()
        if modality not in wanted:
            continue
        if items_per_modality is not None and counts[modality] >= items_per_modality:
            continue
        pair = resolve_video_pair(item, dataset_root)
        if pair is None:
            skipped_no_pair[modality] += 1
            continue
        day_path, night_path = pair
        for path in pair:
            metadata_cache.setdefault(path.as_posix(), probe_video(path))
        selected.append(
            {
                "qa_id": item["qa_id"],
                "modality": item["modality"],
                "section": item["section"],
                "pair_key": item["pair_key"],
                "question": item["question"],
                "ground_truth_answer": item["answer"],
                "day_video": metadata_cache[day_path.as_posix()],
                "night_video": metadata_cache[night_path.as_posix()],
            }
        )
        counts[modality] += 1
        if (
            items_per_modality is not None
            and all(counts[name] >= items_per_modality for name in wanted)
        ):
            break

    if items_per_modality is not None:
        missing = {
            modality: items_per_modality - counts[modality]
            for modality in wanted
            if counts[modality] < items_per_modality
        }
        if missing:
            raise RuntimeError(f"Insufficient day/night video pairs: {missing}")

    metadata = {
        "manifest_type": "aligned_native_day_night_video_v1",
        "input_path": input_path.as_posix(),
        "dataset_root": dataset_root.as_posix(),
        "modalities": list(modalities),
        "items_per_modality": items_per_modality or 0,
        "selected_items": len(selected),
        "counts_by_modality": dict(sorted(counts.items())),
        "skipped_no_video_pair_by_modality": dict(sorted(skipped_no_pair.items())),
        "video_order": "day_then_night",
    }
    payload = {"metadata": metadata, "items": selected}
    payload["metadata"]["manifest_sha256"] = _json_hash(payload["items"])
    _write_json(output_path, payload)
    return payload


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError(f"Invalid video manifest: {path}")
    return payload


def build_native_video_prompt(item: dict[str, Any]) -> str:
    return "\n".join(
        [
            "Answer the question using only the two provided videos.",
            "Video 1 is the daytime/with-light observation.",
            "Video 2 is the nighttime/no-light observation.",
            "Do not use captions, hidden metadata, audio, or outside knowledge.",
            "Return only a concise answer. Do not include explanation.",
            f"Modality: {item.get('modality', '')}",
            f"Section: {item.get('section', '')}",
            "",
            "Question:",
            str(item.get("question", "")).strip(),
        ]
    )


class QwenNativeVideoAdapter(QwenVLFrameAnswerAdapter):
    provider = "qwen_vl"
    quantization = "4bit_nf4"
    video_processing = "qwen_vl_utils_native_video_path"
    def __init__(
        self,
        model_name: str,
        max_tokens: int = 128,
        video_cache_pairs: int = 4,
    ):
        self.last_input_stats: dict[str, Any] = {}
        self.video_cache_pairs = max(0, int(video_cache_pairs))
        self.cpu_preprocess_cache = f"lru_video_pairs={self.video_cache_pairs}"
        self._video_pair_cache = VideoPairLRUCache(self.video_cache_pairs)
        super().__init__(model_name=model_name, max_tokens=max_tokens)

    def _prepare_video_pair(
        self,
        messages: list[dict[str, Any]],
        day_path: Path,
        night_path: Path,
    ) -> tuple[dict[str, Any], bool, float]:
        if process_qwen_vision_info is None:
            raise RuntimeError("qwen-vl-utils is required.")
        cache_key = (
            _video_cache_token(day_path),
            _video_cache_token(night_path),
        )
        cached, cache_hit = self._video_pair_cache.get(cache_key)
        started = time.perf_counter()
        if cache_hit:
            return cached, True, time.perf_counter() - started
        vision_result = process_qwen_vision_info(
            messages,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if len(vision_result) == 3:
            image_inputs, video_inputs, video_kwargs = vision_result
        else:
            image_inputs, video_inputs = vision_result
            video_kwargs = {}
        videos: list[Any] | None = None
        video_metadata: list[Any] | None = None
        if video_inputs:
            videos, video_metadata = map(list, zip(*video_inputs))
        prepared = {
            "image_inputs": image_inputs,
            "videos": videos,
            "video_metadata": video_metadata,
            "video_kwargs": video_kwargs or {},
        }
        self._video_pair_cache.put(cache_key, prepared)
        return prepared, False, time.perf_counter() - started

    def answer(self, item: dict[str, Any], day_path: Path, night_path: Path) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Video 1: daytime/with-light."},
                    {"type": "video", "video": day_path.resolve().as_posix()},
                    {"type": "text", "text": "Video 2: nighttime/no-light."},
                    {"type": "video", "video": night_path.resolve().as_posix()},
                    {"type": "text", "text": build_native_video_prompt(item)},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prepared, cache_hit, preprocess_seconds = self._prepare_video_pair(
            messages,
            day_path,
            night_path,
        )
        videos = prepared["videos"]
        video_metadata = prepared["video_metadata"]
        video_kwargs = prepared["video_kwargs"]
        inputs = self.processor(
            text=[text],
            images=prepared["image_inputs"],
            videos=videos,
            video_metadata=video_metadata,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        self.last_input_stats = {
            "cpu_preprocess_cache_hit": cache_hit,
            "cpu_preprocess_seconds": round(preprocess_seconds, 6),
            "input_tokens": int(inputs["input_ids"].shape[-1]),
            "video_count": len(videos or []),
            "video_kwargs": _json_safe(video_kwargs),
            "video_metadata": _json_safe(video_metadata or []),
            "pixel_values_videos_shape": _shape_of(inputs.get("pixel_values_videos")),
        }
        inputs = _move_inputs_to_device(inputs, _model_input_device(self.model))
        input_length = int(inputs["input_ids"].shape[-1])
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
            )
        new_tokens = generated[0, input_length:]
        return str(self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)).strip()


class InternVLNativeVideoAdapter:
    provider = "internvl"
    quantization = "8bit"

    def __init__(
        self,
        model_name: str,
        max_tokens: int = 128,
        segments_per_video: int = 32,
        image_size: int = 448,
        video_cache_pairs: int = 4,
    ):
        self.max_tokens = max(1, int(max_tokens))
        self.segments_per_video = max(1, int(segments_per_video))
        self.image_size = max(1, int(image_size))
        self.video_processing = (
            f"official_uniform_segments;segments_per_video={self.segments_per_video};"
            "max_num_tiles_per_frame=1"
        )
        self.last_input_stats: dict[str, Any] = {}
        self.video_cache_pairs = max(0, int(video_cache_pairs))
        self.cpu_preprocess_cache = f"lru_video_pairs={self.video_cache_pairs}"
        self._video_pair_cache = VideoPairLRUCache(self.video_cache_pairs)
        self.frame_adapter = InternVLFrameAnswerAdapter(
            model_name=model_name,
            max_tokens=max_tokens,
            image_size=image_size,
            max_num_tiles=1,
        )
        self.model = self.frame_adapter.model
        self.tokenizer = self.frame_adapter.tokenizer

    @staticmethod
    def _indices(frame_count: int, segments: int) -> list[int]:
        if frame_count <= 0:
            raise RuntimeError("Video has no frames.")
        end_index = frame_count - 1
        segment_size = float(end_index) / segments
        return [
            min(end_index, int((segment_size / 2) + np.round(segment_size * index)))
            for index in range(segments)
        ]

    def _load_video(self, path: Path) -> tuple[Any, list[int], list[int], float]:
        if VideoReader is None or cpu is None or np is None or Image is None:
            raise RuntimeError("InternVL native video requires decord, NumPy, and Pillow.")
        reader = VideoReader(path.as_posix(), ctx=cpu(0), num_threads=1)
        indices = self._indices(len(reader), self.segments_per_video)
        tensors = [
            _internvl_image_to_tensor(
                Image.fromarray(reader[index].asnumpy()).convert("RGB"),
                self.image_size,
            )
            for index in indices
        ]
        return (
            torch.stack(tensors).to(torch.bfloat16),
            [1] * len(tensors),
            indices,
            float(reader.get_avg_fps()),
        )

    def _load_video_pair(
        self,
        day_path: Path,
        night_path: Path,
    ) -> tuple[tuple[Any, ...], bool, float]:
        cache_key = (
            _video_cache_token(day_path),
            _video_cache_token(night_path),
            self.segments_per_video,
            self.image_size,
        )
        cached, cache_hit = self._video_pair_cache.get(cache_key)
        started = time.perf_counter()
        if cache_hit:
            return cached, True, time.perf_counter() - started
        prepared = (*self._load_video(day_path), *self._load_video(night_path))
        self._video_pair_cache.put(cache_key, prepared)
        return prepared, False, time.perf_counter() - started

    def answer(self, item: dict[str, Any], day_path: Path, night_path: Path) -> str:
        prepared, cache_hit, preprocess_seconds = self._load_video_pair(
            day_path,
            night_path,
        )
        (
            day_values,
            day_patches,
            day_indices,
            day_fps,
            night_values,
            night_patches,
            night_indices,
            night_fps,
        ) = prepared
        pixel_values = torch.cat([day_values, night_values], dim=0)
        num_patches_list = [*day_patches, *night_patches]
        device = _model_input_device(self.model)
        pixel_values = pixel_values.to(device)
        day_prefix = "".join(
            f"DayFrame{index + 1}: <image>\n" for index in range(len(day_patches))
        )
        night_prefix = "".join(
            f"NightFrame{index + 1}: <image>\n" for index in range(len(night_patches))
        )
        prompt = f"{day_prefix}{night_prefix}{build_native_video_prompt(item)}"
        self.last_input_stats = {
            "cpu_preprocess_cache_hit": cache_hit,
            "cpu_preprocess_seconds": round(preprocess_seconds, 6),
            "segments_per_video": self.segments_per_video,
            "day_frame_indices": day_indices,
            "night_frame_indices": night_indices,
            "day_timestamps_seconds": [round(index / day_fps, 6) for index in day_indices],
            "night_timestamps_seconds": [round(index / night_fps, 6) for index in night_indices],
            "total_visual_tiles": int(pixel_values.shape[0]),
            "pixel_values_shape": list(pixel_values.shape),
        }
        answer = self.model.chat(
            self.tokenizer,
            pixel_values,
            prompt,
            {"max_new_tokens": self.max_tokens, "do_sample": False},
            num_patches_list=num_patches_list,
            history=None,
            return_history=False,
        )
        return str(answer[0] if isinstance(answer, tuple) else answer).strip()


class Molmo2NativeVideoAdapter:
    provider = "molmo2"
    quantization = "bfloat16"
    video_processing = "molmo2_official_sampling;day_night_temporal_concatenation"

    def __init__(
        self,
        model_name: str,
        max_tokens: int = 128,
        max_fps: float = 1.0,
        video_cache_pairs: int = 4,
    ):
        self.max_tokens = max(1, int(max_tokens))
        self.max_fps = max(0.1, float(max_fps))
        self.video_processing = (
            "molmo2_official_sampling;day_night_temporal_concatenation;"
            f"max_fps_per_source_video={self.max_fps:g}"
        )
        self.last_input_stats: dict[str, Any] = {}
        self.video_cache_pairs = max(0, int(video_cache_pairs))
        self.cpu_preprocess_cache = f"lru_video_pairs={self.video_cache_pairs}"
        self._video_pair_cache = VideoPairLRUCache(self.video_cache_pairs)
        if (
            torch is None
            or np is None
            or process_molmo_vision_info is None
            or AutoProcessor is None
            or AutoModelForImageTextToText is None
        ):
            raise RuntimeError(
                "Molmo2 native video requires PyTorch, NumPy, molmo-utils, and Transformers."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True,
            use_fast=False,
            dtype="auto",
            device_map="auto",
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        ).eval()

    def _prepare_video_pair(
        self,
        day_path: Path,
        night_path: Path,
    ) -> tuple[dict[str, Any], bool, float]:
        cache_key = (
            _video_cache_token(day_path),
            _video_cache_token(night_path),
            self.max_fps,
        )
        cached, cache_hit = self._video_pair_cache.get(cache_key)
        started = time.perf_counter()
        if cache_hit:
            return cached, True, time.perf_counter() - started
        source_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": day_path.resolve().as_posix(),
                        "max_fps": self.max_fps,
                    },
                    {
                        "type": "video",
                        "video": night_path.resolve().as_posix(),
                        "max_fps": self.max_fps,
                    },
                ],
            }
        ]
        _, sampled_videos, video_kwargs = process_molmo_vision_info(source_messages)
        if sampled_videos is None or len(sampled_videos) != 2:
            raise RuntimeError("Molmo2 preprocessing did not return both source videos.")
        (day_frames, day_metadata), (night_frames, night_metadata) = sampled_videos
        if day_frames.shape[1:] != night_frames.shape[1:]:
            raise RuntimeError(
                "Molmo2 day/night sampled video shapes do not match: "
                f"{day_frames.shape} vs {night_frames.shape}"
            )

        combined_video = np.concatenate([day_frames, night_frames], axis=0)
        sampling_fps = self.max_fps
        combined_metadata = {
            "total_num_frames": int(combined_video.shape[0]),
            "fps": sampling_fps,
            "duration": float((combined_video.shape[0] - 1) / sampling_fps),
            "video_backend": "molmo_utils_day_night_concat",
            "frames_indices": np.arange(combined_video.shape[0], dtype=np.float64),
            "height": int(combined_video.shape[1]),
            "width": int(combined_video.shape[2]),
        }
        prepared = {
            "combined_video": combined_video,
            "combined_metadata": combined_metadata,
            "video_kwargs": video_kwargs,
            "day_frames": len(day_frames),
            "night_frames": len(night_frames),
            "day_metadata": day_metadata,
            "night_metadata": night_metadata,
        }
        self._video_pair_cache.put(cache_key, prepared)
        return prepared, False, time.perf_counter() - started

    def answer(self, item: dict[str, Any], day_path: Path, night_path: Path) -> str:
        prepared, cache_hit, preprocess_seconds = self._prepare_video_pair(
            day_path,
            night_path,
        )
        combined_video = prepared["combined_video"]
        combined_metadata = prepared["combined_metadata"]
        day_frame_count = prepared["day_frames"]
        night_frame_count = prepared["night_frames"]
        sampling_fps = self.max_fps
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"The single video contains {day_frame_count} daytime/with-light "
                            f"frames followed by {night_frame_count} nighttime/no-light frames."
                        ),
                    },
                    {"type": "video", "video": day_path.resolve().as_posix()},
                    {"type": "text", "text": build_native_video_prompt(item)},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=text,
            videos=[combined_video],
            video_metadata=[combined_metadata],
            padding=True,
            return_tensors="pt",
            **prepared["video_kwargs"],
        )
        self.last_input_stats = {
            "cpu_preprocess_cache_hit": cache_hit,
            "cpu_preprocess_seconds": round(preprocess_seconds, 6),
            "input_tokens": int(inputs["input_ids"].shape[-1]),
            "pixel_values_videos_shape": _shape_of(inputs.get("pixel_values_videos")),
            "source_video_count": 2,
            "model_video_count": 1,
            "day_sampled_frames": int(day_frame_count),
            "night_sampled_frames": int(night_frame_count),
            "combined_sampled_frames": int(len(combined_video)),
            "max_fps_per_source_video": self.max_fps,
            "combined_timeline_fps": sampling_fps,
            "night_start_seconds": round(day_frame_count / sampling_fps, 6),
            "day_source_metadata": _json_safe(prepared["day_metadata"]),
            "night_source_metadata": _json_safe(prepared["night_metadata"]),
        }
        inputs = _move_inputs_to_device(inputs, _model_input_device(self.model))
        input_length = int(inputs["input_ids"].shape[-1])
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
            )
        new_tokens = generated[0, input_length:]
        return str(self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)).strip()


def _shape_of(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    return list(shape) if shape is not None else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _is_oom(exc: Exception) -> bool:
    return "out of memory" in str(exc).lower() or (
        torch is not None and isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", RuntimeError))
    )


def _is_context_overflow(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "maximum context",
            "max_position_embeddings",
            "sequence length",
            "context length",
            "token indices sequence length",
        )
    )


def _load_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    items = payload.get("items", {}) if isinstance(payload, dict) else {}
    return items if isinstance(items, dict) else {}


def _save_results(
    path: Path,
    results: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    _write_json(path, {"items": results, "metadata": metadata})
    _write_csv(path.with_suffix(".csv"), list(results.values()))


def _selected_manifest_items(
    manifest: dict[str, Any],
    max_items_per_modality: int | None,
) -> list[dict[str, Any]]:
    if max_items_per_modality is None:
        return list(manifest["items"])
    counts: dict[str, int] = defaultdict(int)
    selected: list[dict[str, Any]] = []
    for item in manifest["items"]:
        modality = str(item.get("modality", "")).lower()
        if counts[modality] >= max_items_per_modality:
            continue
        selected.append(item)
        counts[modality] += 1
    return selected


def run_model(
    label: str,
    model_name: str,
    adapter: Any,
    manifest: dict[str, Any],
    output_dir: Path,
    resume: bool,
    max_items_per_modality: int | None,
    soft_timeout_seconds: float,
) -> None:
    output_path = output_dir / f"{label}_{_safe_name(Path(model_name).name)}.json"
    results = _load_results(output_path) if resume else {}
    rows = _selected_manifest_items(manifest, max_items_per_modality)
    for item in rows:
        qa_id = str(item["qa_id"])
        if results.get(qa_id, {}).get("status") == "answered":
            continue
        day_path = Path(item["day_video"]["path"])
        night_path = Path(item["night_video"]["path"])
        missing = [path.as_posix() for path in (day_path, night_path) if not path.is_file()]
        if missing:
            raise RuntimeError(f"Manifest contains missing videos for {qa_id}: {missing}")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            baseline_bytes = torch.cuda.memory_allocated()
        else:
            baseline_bytes = 0
        started = time.perf_counter()
        answer = ""
        status = "failed"
        reason = ""
        try:
            answer = adapter.answer(item, day_path, night_path)
            status = "answered" if answer else "failed"
            reason = "" if answer else "Empty model answer"
        except Exception as exc:
            if _is_oom(exc):
                status = "oom"
            elif _is_context_overflow(exc):
                status = "context_overflow"
            else:
                status = "failed"
            reason = f"{type(exc).__name__}: {exc}"
        latency = time.perf_counter() - started
        if status == "answered" and soft_timeout_seconds > 0 and latency > soft_timeout_seconds:
            status = "soft_timeout"
            reason = (
                f"Completed in {latency:.2f}s, exceeding the configured "
                f"{soft_timeout_seconds:.2f}s soft timeout."
            )
        peak_bytes = (
            torch.cuda.max_memory_allocated()
            if torch is not None and torch.cuda.is_available()
            else 0
        )
        results[qa_id] = {
            "qa_id": qa_id,
            "provider": label,
            "model_name": model_name,
            "modality": item["modality"],
            "section": item["section"],
            "pair_key": item["pair_key"],
            "question": item["question"],
            "ground_truth_answer": item["ground_truth_answer"],
            "model_answer": answer,
            "status": status,
            "reason": reason,
            "day_video": item["day_video"],
            "night_video": item["night_video"],
            "latency_seconds": round(latency, 4),
            "baseline_gpu_gb": round(baseline_bytes / 1024**3, 3),
            "peak_gpu_gb": round(peak_bytes / 1024**3, 3),
            "incremental_peak_gpu_gb": round(
                max(0, peak_bytes - baseline_bytes) / 1024**3,
                3,
            ),
            "input_stats": _json_safe(getattr(adapter, "last_input_stats", {}) or {}),
            "generation_config": dict(GENERATION_CONFIG),
        }
        metadata = {
            "benchmark_type": "native_day_night_video_answer_generation_v1",
            "provider": label,
            "model_name": model_name,
            "quantization": getattr(adapter, "quantization", "adapter_default"),
            "video_processing": getattr(adapter, "video_processing", "adapter_default"),
            "cpu_preprocess_cache": getattr(adapter, "cpu_preprocess_cache", "disabled"),
            "video_manifest_sha256": manifest["metadata"]["manifest_sha256"],
            "total_manifest_items": len(manifest["items"]),
            "run_item_limit_per_modality": max_items_per_modality or 0,
            "run_items": len(rows),
            "attempted_items": len(results),
            "status_counts": dict(Counter(row["status"] for row in results.values())),
            "generation_config": dict(GENERATION_CONFIG),
            "soft_timeout_seconds": soft_timeout_seconds,
            "resume": resume,
        }
        _save_results(output_path, results, metadata)
        print(
            f"{label} {qa_id}: {status}, latency={latency:.2f}s, "
            f"peak={peak_bytes / 1024**3:.2f}GB"
        )


def clear_cuda() -> None:
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def assert_gpu_is_available(max_used_memory_mib: int, allow_busy_gpu: bool) -> None:
    if allow_busy_gpu:
        return
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        used_values = [
            int(line.strip())
            for line in completed.stdout.splitlines()
            if line.strip()
        ]
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise RuntimeError(f"Could not verify current GPU memory use: {exc}") from exc
    if not used_values:
        raise RuntimeError("nvidia-smi did not report any GPU memory values.")
    used_mib = max(used_values)
    if used_mib > max_used_memory_mib:
        raise RuntimeError(
            f"GPU is already using {used_mib} MiB, above the allowed startup threshold "
            f"of {max_used_memory_mib} MiB. Finish the existing GPU task first, or pass "
            "--allow-busy-gpu only when concurrent use is intentional."
        )


def _adapter_for(
    label: str,
    model_name: str,
    internvl_segments_per_video: int,
    molmo2_max_fps: float,
    video_cache_pairs: int,
) -> Any:
    if label == "qwen_vl":
        return QwenNativeVideoAdapter(
            model_name=model_name,
            video_cache_pairs=video_cache_pairs,
        )
    if label == "internvl":
        return InternVLNativeVideoAdapter(
            model_name=model_name,
            segments_per_video=internvl_segments_per_video,
            video_cache_pairs=video_cache_pairs,
        )
    if label == "molmo2":
        return Molmo2NativeVideoAdapter(
            model_name=model_name,
            max_fps=molmo2_max_fps,
            video_cache_pairs=video_cache_pairs,
        )
    raise ValueError(f"Unsupported model label: {label}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--dataset-root", default=str(DEFAULT_FRAME_CACHE_ROOT))
    parser.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT_DIR))
    parser.add_argument("--video-manifest", default=str(DEFAULT_VIDEO_MANIFEST))
    parser.add_argument(
        "--items-per-modality",
        type=int,
        default=0,
        help="Manifest items per modality; 0 includes every eligible visual QA item.",
    )
    parser.add_argument("--modalities", default=",".join(DEFAULT_MODALITIES))
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--rebuild-video-manifest", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--max-items-per-modality",
        type=int,
        default=1,
        help="Inference items per modality; 0 runs the complete manifest.",
    )
    parser.add_argument(
        "--soft-timeout-seconds",
        type=float,
        default=300.0,
        help="Mark completed calls exceeding this duration as soft_timeout; 0 disables.",
    )
    parser.add_argument(
        "--internvl-segments-per-video",
        type=int,
        default=12,
        help="InternVL official uniform temporal segments per input video.",
    )
    parser.add_argument(
        "--molmo2-max-fps",
        type=float,
        default=1.0,
        help="Molmo2 official maximum sampling FPS for each source video.",
    )
    parser.add_argument(
        "--video-cache-pairs",
        type=int,
        default=4,
        help="CPU LRU cache capacity for decoded/preprocessed video pairs; 0 disables.",
    )
    parser.add_argument(
        "--max-startup-gpu-memory-mib",
        type=int,
        default=1024,
        help="Refuse to load a model when current GPU use exceeds this value.",
    )
    parser.add_argument(
        "--allow-busy-gpu",
        action="store_true",
        help="Disable the startup GPU memory guard.",
    )
    parser.add_argument("--models", default="qwen_vl,internvl,molmo2")
    parser.add_argument("--qwen-vl-model", default=str(DEFAULT_QWEN_VL_8B))
    parser.add_argument("--internvl-model", default=str(DEFAULT_INTERNVL_8B))
    parser.add_argument("--molmo2-model", default=str(DEFAULT_MOLMO2_8B))
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    input_path = Path(args.input)
    dataset_root = Path(args.dataset_root)
    experiment_dir = Path(args.experiment_dir)
    manifest_path = Path(args.video_manifest)
    modalities = tuple(
        part.strip().lower()
        for part in args.modalities.split(",")
        if part.strip()
    )
    items_per_modality = (
        None if args.items_per_modality == 0 else max(1, args.items_per_modality)
    )
    max_run_items = (
        None
        if args.max_items_per_modality == 0
        else max(1, args.max_items_per_modality)
    )

    if args.rebuild_video_manifest or not manifest_path.exists():
        manifest = build_video_manifest(
            input_path=input_path,
            output_path=manifest_path,
            dataset_root=dataset_root,
            modalities=modalities,
            items_per_modality=items_per_modality,
        )
    else:
        manifest = load_manifest(manifest_path)
        expected = {
            "modalities": list(modalities),
            "items_per_modality": items_per_modality or 0,
        }
        for key, value in expected.items():
            actual = manifest.get("metadata", {}).get(key)
            if actual != value:
                raise RuntimeError(
                    f"Existing video manifest {key}={actual!r}, requested {value!r}; "
                    "use --rebuild-video-manifest."
                )
    print(f"video manifest: {manifest_path}")
    print(json.dumps(manifest["metadata"], indent=2, ensure_ascii=False))
    if args.build_only:
        return

    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; manifest built but inference cannot run.")
    assert_gpu_is_available(
        max_used_memory_mib=max(0, args.max_startup_gpu_memory_mib),
        allow_busy_gpu=args.allow_busy_gpu,
    )

    model_paths = {
        "qwen_vl": args.qwen_vl_model,
        "internvl": args.internvl_model,
        "molmo2": args.molmo2_model,
    }
    requested = [part.strip().lower() for part in args.models.split(",") if part.strip()]
    for label in requested:
        if label not in model_paths:
            raise ValueError(f"Unsupported model label: {label}")
        adapter = _adapter_for(
            label,
            model_paths[label],
            internvl_segments_per_video=max(1, args.internvl_segments_per_video),
            molmo2_max_fps=max(0.1, args.molmo2_max_fps),
            video_cache_pairs=max(0, args.video_cache_pairs),
        )
        try:
            run_model(
                label=label,
                model_name=model_paths[label],
                adapter=adapter,
                manifest=manifest,
                output_dir=experiment_dir / "results",
                resume=not args.no_resume,
                max_items_per_modality=max_run_items,
                soft_timeout_seconds=max(0.0, args.soft_timeout_seconds),
            )
        finally:
            del adapter
            clear_cuda()


if __name__ == "__main__":
    main()
