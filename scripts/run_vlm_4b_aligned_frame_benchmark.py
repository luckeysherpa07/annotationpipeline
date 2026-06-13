#!/usr/bin/env python3
"""Run the fixed 8-frame aligned QA benchmark on local 4B VLMs.

This script reuses the exact frame manifest produced for the 8B benchmark so
the 4B runs consume the same per-item frame paths and model settings.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from annotation_feature.qa_quality.benchmark import (
    DEFAULT_INPUT_PATH,
    InternVLFrameAnswerAdapter,
    QwenVLFrameAnswerAdapter,
    load_valid_qa_items,
)

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from transformers.processing_utils import ProcessorMixin
except ImportError:
    AutoModelForImageTextToText = None
    AutoProcessor = None
    ProcessorMixin = None

try:
    import torch
except ImportError:
    torch = None


DEFAULT_EXPERIMENT_DIR = Path("outputs/benchmarks/vlm_8frame_aligned_4b")
DEFAULT_FRAME_MANIFEST = Path("outputs/benchmarks/vlm_8frame_aligned/manifests/frames_8.json")
DEFAULT_FRAME_MANIFEST_SHA256 = "ce1b15ad21ec8e429b71a7b71e5e2ab4d08453ec74ce76409efde3d9082ce8b3"
DEFAULT_QWEN_VL_4B = Path("models/qwen/Qwen3-VL-4B-Instruct")
DEFAULT_INTERNVL_4B = Path("models/internvl/InternVL3_5-4B-Instruct")
DEFAULT_MOLMO2_4B = Path("models/molmo2/Molmo2-4B")
DEFAULT_FRAME_COUNT = 8
DEFAULT_FRAMES_PER_SIDE = 4
DEFAULT_FRAME_ORDER = "day_then_night"
DEFAULT_SAMPLING_ALGORITHM = "referenced_or_nearest_then_stratified_uniform_v2"
GENERATION_CONFIG = {"max_new_tokens": 128, "do_sample": False}
MOLMO2_LEGACY_PROCESSOR_OPTIONS = {
    "image_use_col_tokens",
    "use_single_crop_col_tokens",
    "use_single_crop_start_token",
    "video_use_col_tokens",
    "use_frame_special_tokens",
    "time_mode",
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

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
    import re

    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value).strip()).strip("_")
    return cleaned or "model"


def _save_frame_answer_outputs(
    output_json: Path,
    output_csv: Path,
    results_by_id: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    payload = {"results": results_by_id, "metadata": metadata}
    _write_json(output_json, payload)
    _write_csv(output_csv, list(results_by_id.values()))


def _json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_existing_path(path: Path, *, base_dir: Path = REPO_ROOT) -> Path:
    if path.is_absolute():
        return path
    if path.is_file():
        return path
    return base_dir / path


def manifest_items(input_path: Path, manifest: dict[str, Any]) -> list[tuple[dict[str, Any], list[Path]]]:
    source_by_id = {str(item["qa_id"]): item for item in load_valid_qa_items(input_path)}
    resolved: list[tuple[dict[str, Any], list[Path]]] = []
    for entry in manifest["items"]:
        item = source_by_id.get(str(entry.get("qa_id", "")))
        frame_paths = [
            _resolve_existing_path(Path(path))
            for path in entry.get("frame_paths", [])
        ]
        if item is None:
            raise RuntimeError(f"QA item missing from source: {entry.get('qa_id')}")
        missing = [path.as_posix() for path in frame_paths if not path.is_file()]
        if missing:
            raise RuntimeError(f"Manifest contains missing frames for {item['qa_id']}: {missing}")
        resolved.append((item, frame_paths))
    return resolved


def _load_manifest(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError(f"Invalid frame manifest: {path}")
    return payload


def _validate_manifest(
    manifest: dict[str, Any],
    manifest_path: Path,
    expected_sha256: str,
) -> None:
    metadata = manifest.get("metadata", {}) if isinstance(manifest, dict) else {}
    if not isinstance(metadata, dict):
        raise ValueError(f"Frame manifest metadata is invalid: {manifest_path}")

    actual_hash = _json_hash(manifest["items"])
    checks = {
        "manifest_sha256": actual_hash,
        "total_frames": DEFAULT_FRAME_COUNT,
        "frames_per_side": DEFAULT_FRAMES_PER_SIDE,
        "frame_order": DEFAULT_FRAME_ORDER,
        "sampling_algorithm": DEFAULT_SAMPLING_ALGORITHM,
    }
    for key, expected in checks.items():
        actual = metadata.get(key)
        if actual != expected:
            raise RuntimeError(
                f"Frame manifest {key}={actual!r} does not match expected {expected!r}; "
                f"use the fixed 8-frame manifest at {manifest_path}."
            )
    if actual_hash != expected_sha256:
        raise RuntimeError(
            f"Frame manifest content hash {actual_hash!r} does not match expected "
            f"{expected_sha256!r}: {manifest_path}"
        )

    seen_qa_ids: set[str] = set()
    for index, entry in enumerate(manifest["items"]):
        if not isinstance(entry, dict):
            raise ValueError(f"Frame manifest item {index} is not an object: {manifest_path}")
        qa_id = str(entry.get("qa_id", "")).strip()
        frame_paths = entry.get("frame_paths")
        if not qa_id:
            raise ValueError(f"Frame manifest item {index} has no qa_id: {manifest_path}")
        if qa_id in seen_qa_ids:
            raise ValueError(f"Frame manifest contains duplicate qa_id {qa_id!r}: {manifest_path}")
        seen_qa_ids.add(qa_id)
        if not isinstance(frame_paths, list) or len(frame_paths) != DEFAULT_FRAME_COUNT:
            raise ValueError(
                f"Frame manifest item {qa_id!r} must contain exactly "
                f"{DEFAULT_FRAME_COUNT} frame paths."
            )
        sides = [_frame_side(Path(path)) for path in frame_paths]
        expected_sides = ["day"] * DEFAULT_FRAMES_PER_SIDE + ["night"] * DEFAULT_FRAMES_PER_SIDE
        if sides != expected_sides:
            raise ValueError(
                f"Frame manifest item {qa_id!r} does not follow {DEFAULT_FRAME_ORDER}: {sides}"
            )


def _frame_side(path: Path) -> str:
    text = path.as_posix().lower()
    if "/day/" in text or "_day_" in text or "with_light" in text:
        return "day"
    if "/night/" in text or "_night" in text or "no_light" in text or "cloudy_no_light" in text:
        return "night"
    return "unknown"


def _result_output_paths(output_dir: Path, model_name: str) -> tuple[Path, Path]:
    safe_name = _safe_name(Path(model_name).name)
    return (
        output_dir / f"aligned_qa_frame_answers_{safe_name}.json",
        output_dir / f"aligned_qa_frame_answers_{safe_name}.csv",
    )


def _load_molmo2_processor(model_name: str) -> Any:
    kwargs = {
        "trust_remote_code": True,
        "local_files_only": True,
        "use_fast": False,
    }
    try:
        processor = AutoProcessor.from_pretrained(model_name, **kwargs)
    except TypeError as exc:
        if (
            ProcessorMixin is None
            or "Unexpected keyword argument" not in str(exc)
            or not any(option in str(exc) for option in MOLMO2_LEGACY_PROCESSOR_OPTIONS)
        ):
            raise
    else:
        return _ensure_molmo2_chat_template(processor, model_name)

    original_init = ProcessorMixin.__init__

    def compatible_init(processor: Any, *args: Any, **init_kwargs: Any) -> None:
        legacy_options = {
            key: init_kwargs.pop(key)
            for key in tuple(init_kwargs)
            if key in MOLMO2_LEGACY_PROCESSOR_OPTIONS
        }
        original_init(processor, *args, **init_kwargs)
        for key, value in legacy_options.items():
            setattr(processor, key, value)

    ProcessorMixin.__init__ = compatible_init
    try:
        processor = AutoProcessor.from_pretrained(model_name, **kwargs)
    finally:
        ProcessorMixin.__init__ = original_init
    return _ensure_molmo2_chat_template(processor, model_name)


def _ensure_molmo2_chat_template(processor: Any, model_name: str) -> Any:
    if getattr(processor, "chat_template", None):
        return processor
    template_path = Path(model_name) / "chat_template.jinja"
    if not template_path.is_file():
        raise RuntimeError(f"Molmo2 chat template is missing: {template_path}")
    processor.chat_template = template_path.read_text(encoding="utf-8")
    return processor


class Molmo2FrameAnswerAdapter:
    """Local Molmo2 adapter for independent image-frame input."""

    provider = "molmo2"
    quantization = "bfloat16"

    def __init__(self, model_name: str, max_tokens: int = 128, require_cuda: bool = True):
        self.model_name = model_name
        self.max_tokens = max(1, int(max_tokens))
        self.last_input_stats: dict[str, Any] = {}
        if torch is None or Image is None:
            raise RuntimeError("Molmo2 requires PyTorch and Pillow.")
        if require_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        if AutoProcessor is None or AutoModelForImageTextToText is None:
            raise RuntimeError("Molmo2 requires Transformers.")
        self.processor = _load_molmo2_processor(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        ).eval()

    def answer(self, item: dict[str, Any], frame_paths: list[Path]) -> str:
        images = []
        for path in frame_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        half = len(frame_paths) // 2
        prompt = "\n".join(
            [
                "Answer using only the provided image frames.",
                f"Frames 1-{half} are daytime/with-light observations.",
                f"Frames {half + 1}-{len(frame_paths)} are nighttime/no-light observations.",
                "Return only a concise answer. Do not include explanation.",
                f"Modality: {item.get('modality', '')}",
                f"Section: {item.get('section', '')}",
                "",
                "Question:",
                str(item.get("question", "")).strip(),
            ]
        )
        content: list[dict[str, Any]] = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
        try:
            inputs = self.processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            self.last_input_stats = {
                "input_tokens": int(inputs["input_ids"].shape[-1]),
                "pixel_values_shape": list(inputs["pixel_values"].shape),
            }
            device = getattr(self.model, "device", None)
            if device is None:
                try:
                    device = next(self.model.parameters()).device
                except Exception:
                    device = None
            inputs = {
                key: value.to(device) if device is not None and hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            input_length = int(inputs["input_ids"].shape[-1])
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    do_sample=False,
                )
            new_tokens = generated[0, input_length:]
            return str(
                self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
            ).strip()
        finally:
            for image in images:
                image.close()


def _load_results(
    path: Path,
    *,
    model_name: str,
    manifest_sha256: str,
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise RuntimeError(f"Existing result metadata is invalid: {path}")
    expected = {
        "model_name": model_name,
        "frame_manifest_sha256": manifest_sha256,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"Existing result {key}={metadata.get(key)!r}, requested {value!r}: {path}. "
                "Use --no-resume or a different --experiment-dir."
            )
    items = payload.get("results")
    if isinstance(items, dict):
        return {str(qa_id): result for qa_id, result in items.items() if isinstance(result, dict)}
    return {}


def _completed(result: dict[str, Any]) -> bool:
    reason = str(result.get("reason", ""))
    if reason.startswith("Frame answer call failed:"):
        return False
    return result.get("status") == "answered" and bool(str(result.get("model_answer", "")).strip())


def _is_oom(exc: Exception) -> bool:
    return "out of memory" in str(exc).lower() or (
        torch is not None and isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", RuntimeError))
    )


def _run_fixed_frame_model(
    label: str,
    model_name: str,
    adapter: Any,
    input_path: Path,
    frame_manifest: dict[str, Any],
    frame_manifest_path: Path,
    output_dir: Path,
    resume: bool,
    max_items_per_modality: int | None = None,
) -> None:
    output_json, output_csv = _result_output_paths(output_dir, model_name)
    results = (
        _load_results(
            output_json,
            model_name=model_name,
            manifest_sha256=frame_manifest["metadata"]["manifest_sha256"],
        )
        if resume
        else {}
    )
    all_rows = manifest_items(input_path, frame_manifest)

    if max_items_per_modality is None:
        rows = all_rows
    else:
        selected_counts: dict[str, int] = defaultdict(int)
        rows = []
        for item, frame_paths in all_rows:
            modality = str(item.get("modality", "")).lower()
            if selected_counts[modality] >= max_items_per_modality:
                continue
            rows.append((item, frame_paths))
            selected_counts[modality] += 1

    for item, frame_paths in rows:
        if _completed(results.get(item["qa_id"], {})):
            continue

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
            answer = adapter.answer(item, frame_paths)
            status = "answered" if answer else "failed"
            reason = "" if answer else "Frame answer call failed: empty model answer"
        except Exception as exc:
            status = "oom" if _is_oom(exc) else "failed"
            reason = f"Frame answer call failed: {type(exc).__name__}: {exc}"

        latency = time.perf_counter() - started
        peak_bytes = torch.cuda.max_memory_allocated() if torch is not None and torch.cuda.is_available() else 0
        results[item["qa_id"]] = {
            "qa_id": item["qa_id"],
            "provider": label,
            "model_name": model_name,
            "modality": item["modality"],
            "section": item["section"],
            "pair_key": item["pair_key"],
            "question": item["question"],
            "ground_truth_answer": item["answer"],
            "model_answer": answer,
            "status": status,
            "reason": reason,
            "frame_count": len(frame_paths),
            "day_frame_count": sum(_frame_side(path) == "day" for path in frame_paths),
            "night_frame_count": sum(_frame_side(path) == "night" for path in frame_paths),
            "frame_paths": [path.as_posix() for path in frame_paths],
            "latency_seconds": round(latency, 4),
            "baseline_gpu_gb": round(baseline_bytes / 1024**3, 3),
            "peak_gpu_gb": round(peak_bytes / 1024**3, 3),
            "incremental_peak_gpu_gb": round(max(0, peak_bytes - baseline_bytes) / 1024**3, 3),
            "input_stats": dict(getattr(adapter, "last_input_stats", {}) or {}),
            "generation_config": dict(GENERATION_CONFIG),
        }

        metadata = {
            "benchmark_type": "fixed_8frame_aligned_4b_frame_input_v1",
            "reference_benchmark": "outputs/benchmarks/vlm_8frame_aligned",
            "provider": label,
            "model_name": model_name,
            "quantization": getattr(adapter, "quantization", "adapter_default"),
            "frame_manifest_path": frame_manifest_path.as_posix(),
            "frame_manifest_sha256": frame_manifest["metadata"]["manifest_sha256"],
            "frame_manifest_total_frames": frame_manifest["metadata"]["total_frames"],
            "frame_manifest_frames_per_side": frame_manifest["metadata"]["frames_per_side"],
            "frame_manifest_order": frame_manifest["metadata"]["frame_order"],
            "frame_manifest_sampling_algorithm": frame_manifest["metadata"]["sampling_algorithm"],
            "total_manifest_items": len(all_rows),
            "run_item_limit_per_modality": max_items_per_modality or 0,
            "run_items": len(rows),
            "attempted_items": len(results),
            "status_counts": dict(Counter(row["status"] for row in results.values())),
            "generation_config": dict(GENERATION_CONFIG),
            "resume": resume,
        }
        _save_frame_answer_outputs(output_json, output_csv, results, metadata)
        print(
            f"{label} {item['qa_id']}: {status}, frames={len(frame_paths)}, "
            f"latency={latency:.2f}s, peak={peak_bytes / 1024**3:.2f}GB"
        )


def _adapter_for(label: str, model_name: str) -> Any:
    if label == "qwen_vl":
        adapter = QwenVLFrameAnswerAdapter(model_name=model_name)
        adapter.quantization = "4bit_nf4"
        return adapter
    if label == "internvl":
        adapter = InternVLFrameAnswerAdapter(model_name=model_name, max_num_tiles=1)
        adapter.quantization = "8bit;max_num_tiles_per_frame=1"
        return adapter
    if label == "molmo2":
        adapter = Molmo2FrameAnswerAdapter(model_name=model_name)
        adapter.quantization = "bfloat16"
        return adapter
    raise ValueError(f"Unsupported model label: {label}")


def clear_cuda() -> None:
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--frame-manifest", default=str(DEFAULT_FRAME_MANIFEST))
    parser.add_argument("--expected-frame-manifest-sha256", default=DEFAULT_FRAME_MANIFEST_SHA256)
    parser.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT_DIR))
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--max-items-per-modality",
        type=int,
        default=0,
        help="Limit inference to N items per modality; 0 runs the full manifest.",
    )
    parser.add_argument("--models", default="qwen_vl,internvl,molmo2")
    parser.add_argument("--qwen-vl-model", default=str(DEFAULT_QWEN_VL_4B))
    parser.add_argument("--internvl-model", default=str(DEFAULT_INTERNVL_4B))
    parser.add_argument("--molmo2-model", default=str(DEFAULT_MOLMO2_4B))
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    input_path = Path(args.input)
    frame_manifest_path = Path(args.frame_manifest)
    experiment_dir = Path(args.experiment_dir)
    max_items_per_modality = None if args.max_items_per_modality == 0 else max(1, args.max_items_per_modality)
    requested = [part.strip().lower() for part in args.models.split(",") if part.strip()]

    input_path = _resolve_existing_path(input_path)
    frame_manifest_path = _resolve_existing_path(frame_manifest_path)
    frame_manifest = _load_manifest(frame_manifest_path)
    _validate_manifest(frame_manifest, frame_manifest_path, str(args.expected_frame_manifest_sha256).strip())

    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; the fixed-frame manifest was loaded but inference cannot run.")

    model_paths = {
        "qwen_vl": args.qwen_vl_model,
        "internvl": args.internvl_model,
        "molmo2": args.molmo2_model,
    }

    for label in requested:
        if label not in model_paths:
            raise ValueError(f"Unsupported model label: {label}")

        model_name = model_paths[label]
        model_path = _resolve_existing_path(Path(model_name))
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist for {label}: {model_name}")
        model_name = model_path.as_posix()
        adapter = None
        try:
            adapter = _adapter_for(label, model_name)
            _run_fixed_frame_model(
                label=label,
                model_name=model_name,
                adapter=adapter,
                input_path=input_path,
                frame_manifest=frame_manifest,
                frame_manifest_path=frame_manifest_path,
                output_dir=experiment_dir,
                resume=not args.no_resume,
                max_items_per_modality=max_items_per_modality,
            )
        finally:
            if adapter is not None:
                del adapter
            clear_cuda()


if __name__ == "__main__":
    main()
