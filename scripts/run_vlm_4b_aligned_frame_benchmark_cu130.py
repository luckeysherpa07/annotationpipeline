#!/usr/bin/env python3
"""Run the fixed 8-frame 4B VLM benchmark in the local CUDA 13 environment.

This runner deliberately keeps the CUDA 12.6 / Transformers 5 compatibility
runner unchanged. It reuses that runner's manifest validation, result format,
resume behavior, and frame caches, while selecting model loading paths that
have been verified in this machine's primary .venv.
"""

from __future__ import annotations

import hashlib
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if SCRIPT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, SCRIPT_DIR.as_posix())

import run_vlm_4b_aligned_frame_benchmark as shared


DEFAULT_EXPERIMENT_DIR = Path("outputs/benchmarks/vlm_8frame_aligned_4b")
DEFAULT_QWEN_VL_4B = Path("models/qwen/Qwen3-VL-4B-Instruct")
DEFAULT_INTERNVL_4B = Path("models/internvl/InternVL3_5-4B-Instruct")
DEFAULT_MOLMO2_4B = Path("models/molmo2/Molmo2-4B")
DEFAULT_CHECKPOINT_EVERY_ITEMS = 25
_shared_save_outputs = shared._save_frame_answer_outputs


def _package_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
        return "unavailable"


def _runtime_environment() -> dict[str, Any]:
    gpu_name = "unavailable"
    if shared.torch is not None and shared.torch.cuda.is_available():
        try:
            gpu_name = str(shared.torch.cuda.get_device_name(0))
        except Exception:
            gpu_name = "visible"
    return {
        "runner": Path(__file__).name,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pytorch": str(getattr(shared.torch, "__version__", "unavailable")),
        "cuda_runtime": str(getattr(getattr(shared.torch, "version", None), "cuda", None)),
        "transformers": _package_version("transformers"),
        "bitsandbytes": _package_version("bitsandbytes"),
        "gpu": gpu_name,
    }


@lru_cache(maxsize=8)
def _model_index_sha256(model_name: str) -> str | None:
    index_path = Path(model_name) / "model.safetensors.index.json"
    if not index_path.is_file():
        return None
    return hashlib.sha256(index_path.read_bytes()).hexdigest()


def _save_frame_answer_outputs(
    output_json: Path,
    output_csv: Path,
    results_by_id: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    metadata = dict(metadata)
    metadata["benchmark_type"] = "fixed_8frame_aligned_4b_frame_input_cu130_v1"
    metadata["runtime_environment"] = _runtime_environment()
    metadata["model_index_sha256"] = _model_index_sha256(
        str(metadata.get("model_name", ""))
    )
    temporary_json = output_json.with_name(f".{output_json.name}.tmp")
    temporary_csv = output_csv.with_name(f".{output_csv.name}.tmp")
    try:
        _shared_save_outputs(
            temporary_json,
            temporary_csv,
            results_by_id,
            metadata,
        )
        os.replace(temporary_json, output_json)
        os.replace(temporary_csv, output_csv)
    finally:
        temporary_json.unlink(missing_ok=True)
        temporary_csv.unlink(missing_ok=True)


def _result_metadata(
    *,
    label: str,
    model_name: str,
    adapter: Any,
    frame_manifest: dict[str, Any],
    frame_manifest_path: Path,
    all_rows: list[tuple[dict[str, Any], list[Path]]],
    rows: list[tuple[dict[str, Any], list[Path]]],
    results: dict[str, dict[str, Any]],
    resume: bool,
    max_items_per_modality: int | None,
    checkpoint_every_items: int,
) -> dict[str, Any]:
    return {
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
        "frame_manifest_sampling_algorithm": frame_manifest["metadata"][
            "sampling_algorithm"
        ],
        "total_manifest_items": len(all_rows),
        "run_item_limit_per_modality": max_items_per_modality or 0,
        "run_items": len(rows),
        "attempted_items": len(results),
        "status_counts": dict(Counter(row["status"] for row in results.values())),
        "frame_cache_level": getattr(adapter, "frame_cache_level", "none"),
        "frame_cache_capacity_sets": 1,
        "frame_cache_hits": int(getattr(adapter, "frame_cache_hits", 0)),
        "frame_cache_misses": int(getattr(adapter, "frame_cache_misses", 0)),
        "generation_config": dict(shared.GENERATION_CONFIG),
        "checkpoint_every_items": checkpoint_every_items,
        "resume": resume,
    }


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
    output_json, output_csv = shared._result_output_paths(output_dir, model_name)
    results = (
        shared._load_results(
            output_json,
            model_name=model_name,
            manifest_sha256=frame_manifest["metadata"]["manifest_sha256"],
        )
        if resume
        else {}
    )
    all_rows = shared.manifest_items(input_path, frame_manifest)

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

    rows = shared._group_rows_by_frame_set(rows)
    checkpoint_every_items = max(
        1,
        int(os.environ.get("VLM_CHECKPOINT_EVERY_ITEMS", DEFAULT_CHECKPOINT_EVERY_ITEMS)),
    )
    unsaved_items = 0

    def save_checkpoint() -> None:
        metadata = _result_metadata(
            label=label,
            model_name=model_name,
            adapter=adapter,
            frame_manifest=frame_manifest,
            frame_manifest_path=frame_manifest_path,
            all_rows=all_rows,
            rows=rows,
            results=results,
            resume=resume,
            max_items_per_modality=max_items_per_modality,
            checkpoint_every_items=checkpoint_every_items,
        )
        _save_frame_answer_outputs(output_json, output_csv, results, metadata)

    try:
        for item, frame_paths in rows:
            if shared._completed(results.get(item["qa_id"], {})):
                continue

            if shared.torch is not None and shared.torch.cuda.is_available():
                shared.torch.cuda.empty_cache()
                shared.torch.cuda.reset_peak_memory_stats()
                baseline_bytes = shared.torch.cuda.memory_allocated()
            else:
                baseline_bytes = 0

            started = time.perf_counter()
            answer = ""
            status = "failed"
            reason = ""
            cache_hits_before = int(getattr(adapter, "frame_cache_hits", 0))
            try:
                answer = adapter.answer(item, frame_paths)
                status = "answered" if answer else "failed"
                reason = "" if answer else "Frame answer call failed: empty model answer"
            except Exception as exc:
                status = "oom" if shared._is_oom(exc) else "failed"
                reason = f"Frame answer call failed: {type(exc).__name__}: {exc}"

            latency = time.perf_counter() - started
            peak_bytes = (
                shared.torch.cuda.max_memory_allocated()
                if shared.torch is not None and shared.torch.cuda.is_available()
                else 0
            )
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
                "day_frame_count": sum(
                    shared._frame_side(path) == "day" for path in frame_paths
                ),
                "night_frame_count": sum(
                    shared._frame_side(path) == "night" for path in frame_paths
                ),
                "frame_paths": [path.as_posix() for path in frame_paths],
                "latency_seconds": round(latency, 4),
                "baseline_gpu_gb": round(baseline_bytes / 1024**3, 3),
                "peak_gpu_gb": round(peak_bytes / 1024**3, 3),
                "incremental_peak_gpu_gb": round(
                    max(0, peak_bytes - baseline_bytes) / 1024**3,
                    3,
                ),
                "input_stats": dict(getattr(adapter, "last_input_stats", {}) or {}),
                "frame_cache_hit": (
                    int(getattr(adapter, "frame_cache_hits", 0)) > cache_hits_before
                ),
                "generation_config": dict(shared.GENERATION_CONFIG),
            }
            unsaved_items += 1
            if unsaved_items >= checkpoint_every_items:
                save_checkpoint()
                unsaved_items = 0

            print(
                f"{label} {item['qa_id']}: {status}, frames={len(frame_paths)}, "
                f"latency={latency:.2f}s, peak={peak_bytes / 1024**3:.2f}GB"
            )
    finally:
        if unsaved_items:
            save_checkpoint()


def _model_input_device(model: Any) -> Any | None:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except Exception:
        return None


class LocalMolmo2FrameAnswerAdapter(shared._SingleRGBFrameSetCache):
    """Molmo2 frame adapter for Transformers 4.57 and CUDA 13."""

    provider = "molmo2"
    quantization = "bfloat16"

    def __init__(self, model_name: str, max_tokens: int = 128):
        self._init_frame_cache()
        self.model_name = model_name
        self.max_tokens = max(1, int(max_tokens))
        self.last_input_stats: dict[str, Any] = {}

        if shared.torch is None or shared.Image is None:
            raise RuntimeError("Molmo2 requires PyTorch and Pillow.")
        if not shared.torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        if shared.AutoProcessor is None or shared.AutoModelForImageTextToText is None:
            raise RuntimeError("Molmo2 requires Transformers.")

        self.processor = shared.AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True,
            use_fast=False,
        )
        self.model = shared.AutoModelForImageTextToText.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True,
            dtype=shared.torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        ).eval()

    def answer(self, item: dict[str, Any], frame_paths: list[Path]) -> str:
        images = self._rgb_frames(frame_paths)
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
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
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

        device = _model_input_device(self.model)
        inputs = {
            key: value.to(device) if device is not None and hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        input_length = int(inputs["input_ids"].shape[-1])
        with shared.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
            )
        new_tokens = generated[0, input_length:]
        return str(
            self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        ).strip()


def _adapter_for(label: str, model_name: str) -> Any:
    if label == "qwen_vl":
        adapter = shared.CachedQwenVLFrameAnswerAdapter(model_name=model_name)
        adapter.quantization = "4bit_nf4"
        return adapter
    if label == "internvl":
        adapter = shared.CachedInternVLFrameAnswerAdapter(
            model_name=model_name,
            max_num_tiles=1,
        )
        adapter.quantization = "8bit;max_num_tiles_per_frame=1"
        return adapter
    if label == "molmo2":
        return LocalMolmo2FrameAnswerAdapter(model_name=model_name)
    raise ValueError(f"Unsupported model label: {label}")


def main() -> None:
    shared.__doc__ = __doc__
    shared.DEFAULT_EXPERIMENT_DIR = DEFAULT_EXPERIMENT_DIR
    shared.DEFAULT_QWEN_VL_4B = DEFAULT_QWEN_VL_4B
    shared.DEFAULT_INTERNVL_4B = DEFAULT_INTERNVL_4B
    shared.DEFAULT_MOLMO2_4B = DEFAULT_MOLMO2_4B
    shared._adapter_for = _adapter_for
    shared._save_frame_answer_outputs = _save_frame_answer_outputs
    shared._run_fixed_frame_model = _run_fixed_frame_model
    shared.main()


if __name__ == "__main__":
    main()
