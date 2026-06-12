#!/usr/bin/env python3
"""Run reproducible multi-model frame-count smoke tests on aligned QA."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
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
    load_valid_qa_items,
    resolve_frame_cache_candidates,
)

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
except ImportError:
    AutoModelForImageTextToText = None
    AutoProcessor = None
    BitsAndBytesConfig = None


DEFAULT_EXPERIMENT_DIR = Path("outputs/benchmarks/vlm_frame_count_smoke")
DEFAULT_QA_MANIFEST = DEFAULT_EXPERIMENT_DIR / "fixed_qa_manifest.json"
DEFAULT_QWEN_VL_8B = Path("models/qwen/Qwen3-VL-8B-Instruct")
DEFAULT_INTERNVL_8B = Path("models/internvl/InternVL3-8B")
DEFAULT_MOLMO2_8B = Path("models/molmo2/Molmo2-8B")
DEFAULT_MODALITIES = ("rgb", "ir", "event", "depth")
DEFAULT_FRAME_COUNTS = (4, 6, 8)
GENERATION_CONFIG = {"max_new_tokens": 128, "do_sample": False}


def _model_input_device(model: Any) -> Any | None:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except Exception:
        return None


def _frame_number(path: Path) -> int:
    match = re.search(r"frame_(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def _frame_side(path: Path) -> str:
    text = path.as_posix().lower()
    if "/day/" in text or "_day_" in text or "with_light" in text:
        return "day"
    if (
        "/night/" in text
        or "_night" in text
        or "no_light" in text
        or "cloudy_no_light" in text
    ):
        return "night"
    return "unknown"


def _dedupe_sorted(paths: list[Path]) -> list[Path]:
    by_path = {path.as_posix(): path for path in paths}
    return sorted(by_path.values(), key=lambda path: (_frame_number(path), path.as_posix()))


def _uniform_sample(paths: list[Path], count: int) -> list[Path]:
    paths = _dedupe_sorted(paths)
    if count <= 0 or len(paths) <= count:
        return paths
    if count == 1:
        return [paths[len(paths) // 2]]
    indices = [round(index * (len(paths) - 1) / (count - 1)) for index in range(count)]
    return [paths[index] for index in indices]


def _referenced_frame_numbers(question: str) -> list[int]:
    return [
        int(match.group(1))
        for match in re.finditer(r"\bframe[_\s-]*(\d{3,6})\b", question, flags=re.I)
    ]


def _sample_side(paths: list[Path], question: str, count: int) -> list[Path]:
    """Select referenced frames first, then uniformly fill the remaining side quota."""
    paths = _dedupe_sorted(paths)
    wanted = set(_referenced_frame_numbers(question))
    exact = [path for path in paths if _frame_number(path) in wanted][:count]
    exact_keys = {path.as_posix() for path in exact}
    remaining = [path for path in paths if path.as_posix() not in exact_keys]
    return [*exact, *_uniform_sample(remaining, count - len(exact))]


def _split_candidates_by_side(item: dict[str, Any], frame_cache_root: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {"day": [], "night": [], "unknown": []}
    for path in resolve_frame_cache_candidates(item, frame_cache_root=frame_cache_root):
        grouped[_frame_side(path)].append(path)
    return {side: _dedupe_sorted(paths) for side, paths in grouped.items()}


def _json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def build_fixed_qa_manifest(
    input_path: Path,
    output_path: Path,
    frame_cache_root: Path,
    modalities: tuple[str, ...],
    items_per_modality: int,
    maximum_frames_per_side: int,
) -> dict[str, Any]:
    """Select one fixed QA set that supports every requested frame-count tier."""
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)
    wanted = set(modalities)
    for item in load_valid_qa_items(input_path):
        modality = str(item.get("modality", "")).lower()
        if modality not in wanted or counts[modality] >= items_per_modality:
            continue
        grouped = _split_candidates_by_side(item, frame_cache_root)
        if (
            len(grouped["day"]) < maximum_frames_per_side
            or len(grouped["night"]) < maximum_frames_per_side
        ):
            continue
        selected.append(
            {
                "qa_id": item["qa_id"],
                "modality": item["modality"],
                "section": item["section"],
                "pair_key": item["pair_key"],
                "question": item["question"],
                "ground_truth_answer": item["answer"],
                "available_day_frames": len(grouped["day"]),
                "available_night_frames": len(grouped["night"]),
            }
        )
        counts[modality] += 1
        if all(counts[modality] >= items_per_modality for modality in wanted):
            break

    missing = {modality: items_per_modality - counts[modality] for modality in wanted if counts[modality] < items_per_modality}
    if missing:
        raise RuntimeError(f"Could not build fixed QA manifest; insufficient balanced frame coverage: {missing}")

    metadata = {
        "manifest_type": "fixed_aligned_qa_selection_v1",
        "input_path": input_path.as_posix(),
        "frame_cache_root": frame_cache_root.as_posix(),
        "modalities": list(modalities),
        "items_per_modality": items_per_modality,
        "selected_items": len(selected),
        "counts_by_modality": dict(sorted(counts.items())),
        "minimum_frames_required_per_side": maximum_frames_per_side,
    }
    payload = {"metadata": metadata, "items": selected}
    payload["metadata"]["manifest_sha256"] = _json_hash(payload["items"])
    _write_json(output_path, payload)
    return payload


def load_manifest(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError(f"Invalid manifest: {path}")
    return payload


def build_frame_manifest(
    qa_manifest: dict[str, Any],
    output_path: Path,
    input_path: Path,
    frame_cache_root: Path,
    total_frames: int,
    anchor_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if total_frames < 2 or total_frames % 2:
        raise ValueError("Total frame count must be an even integer of at least 2")
    frames_per_side = total_frames // 2
    source_by_id = {str(item["qa_id"]): item for item in load_valid_qa_items(input_path)}
    anchor_by_id = (
        {str(item["qa_id"]): item for item in anchor_manifest["items"]}
        if anchor_manifest is not None
        else {}
    )
    rows: list[dict[str, Any]] = []
    for fixed in qa_manifest["items"]:
        item = source_by_id.get(str(fixed["qa_id"]))
        if item is None:
            raise RuntimeError(f"QA item disappeared from source data: {fixed['qa_id']}")
        if anchor_manifest is None:
            grouped = _split_candidates_by_side(item, frame_cache_root)
        else:
            anchor = anchor_by_id.get(str(item["qa_id"]))
            if anchor is None:
                raise RuntimeError(f"QA item missing from anchor manifest: {item['qa_id']}")
            grouped = {
                "day": [Path(path) for path in anchor["day_frames"]],
                "night": [Path(path) for path in anchor["night_frames"]],
            }
        day = _sample_side(grouped["day"], item["question"], frames_per_side)
        night = _sample_side(grouped["night"], item["question"], frames_per_side)
        if len(day) != frames_per_side or len(night) != frames_per_side:
            raise RuntimeError(f"Insufficient balanced frames for {item['qa_id']}")
        rows.append(
            {
                **fixed,
                "day_frames": [path.as_posix() for path in day],
                "night_frames": [path.as_posix() for path in night],
                "frame_paths": [path.as_posix() for path in [*day, *night]],
            }
        )

    metadata = {
        "manifest_type": "balanced_frame_sampling_v1",
        "qa_manifest_sha256": qa_manifest["metadata"]["manifest_sha256"],
        "sampling_algorithm": "referenced_then_stratified_uniform_v1",
        "anchor_manifest_sha256": (
            anchor_manifest["metadata"]["manifest_sha256"]
            if anchor_manifest is not None
            else None
        ),
        "total_frames": total_frames,
        "frames_per_side": frames_per_side,
        "frame_order": "day_then_night",
        "selected_items": len(rows),
    }
    payload = {"metadata": metadata, "items": rows}
    payload["metadata"]["manifest_sha256"] = _json_hash(payload["items"])
    _write_json(output_path, payload)
    return payload


def manifest_items(input_path: Path, manifest: dict[str, Any]) -> list[tuple[dict[str, Any], list[Path]]]:
    source_by_id = {str(item["qa_id"]): item for item in load_valid_qa_items(input_path)}
    resolved: list[tuple[dict[str, Any], list[Path]]] = []
    for entry in manifest["items"]:
        item = source_by_id.get(str(entry.get("qa_id", "")))
        frame_paths = [Path(path) for path in entry.get("frame_paths", [])]
        if item is None:
            raise RuntimeError(f"QA item missing from source: {entry.get('qa_id')}")
        missing = [path.as_posix() for path in frame_paths if not path.is_file()]
        if missing:
            raise RuntimeError(f"Manifest contains missing frames for {item['qa_id']}: {missing}")
        resolved.append((item, frame_paths))
    return resolved


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
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True,
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

    def answer(self, item: dict[str, Any], frame_paths: list[Path]) -> str:
        images = [Image.open(path).convert("RGB") for path in frame_paths]
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
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        input_length = int(inputs["input_ids"].shape[-1])
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
            )
        new_tokens = generated[0, input_length:]
        return str(self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)).strip()


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")


def _load_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    items = payload.get("items", {}) if isinstance(payload, dict) else {}
    if isinstance(items, list):
        return {str(item.get("qa_id")): item for item in items if isinstance(item, dict) and item.get("qa_id")}
    return items if isinstance(items, dict) else {}


def _save_results(path: Path, results: dict[str, dict[str, Any]], metadata: dict[str, Any]) -> None:
    _write_json(path, {"items": results, "metadata": metadata})
    _write_csv(path.with_suffix(".csv"), list(results.values()))


def _is_oom(exc: Exception) -> bool:
    return "out of memory" in str(exc).lower() or (
        torch is not None and isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", RuntimeError))
    )


def run_model(
    label: str,
    model_name: str,
    adapter: Any,
    input_path: Path,
    frame_manifest: dict[str, Any],
    output_dir: Path,
    resume: bool,
) -> None:
    safe_model = _safe_name(Path(model_name).name)
    output_json = output_dir / f"{label}_{safe_model}.json"
    results = _load_results(output_json) if resume else {}
    rows = manifest_items(input_path, frame_manifest)
    for item, frame_paths in rows:
        if results.get(item["qa_id"], {}).get("status") == "answered":
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
            reason = "" if answer else "Empty model answer"
        except Exception as exc:
            status = "oom" if _is_oom(exc) else "failed"
            reason = f"{type(exc).__name__}: {exc}"
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
            "benchmark_type": "shared_balanced_frame_count_smoke_v1",
            "provider": label,
            "model_name": model_name,
            "quantization": getattr(adapter, "quantization", "adapter_default"),
            "frame_manifest_sha256": frame_manifest["metadata"]["manifest_sha256"],
            "total_frames": frame_manifest["metadata"]["total_frames"],
            "frames_per_side": frame_manifest["metadata"]["frames_per_side"],
            "total_items": len(rows),
            "attempted_items": len(results),
            "status_counts": dict(Counter(row["status"] for row in results.values())),
            "generation_config": dict(GENERATION_CONFIG),
            "resume": resume,
        }
        _save_results(output_json, results, metadata)
        print(
            f"{label} {item['qa_id']}: {status}, frames={len(frame_paths)}, "
            f"latency={latency:.2f}s, peak={peak_bytes / 1024**3:.2f}GB"
        )


def clear_cuda() -> None:
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _parse_frame_counts(raw: str) -> tuple[int, ...]:
    values = tuple(sorted({int(part.strip()) for part in raw.split(",") if part.strip()}))
    if not values or any(value < 2 or value % 2 for value in values):
        raise ValueError("Frame counts must be comma-separated even integers of at least 2")
    return values


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
        return Molmo2FrameAnswerAdapter(model_name=model_name)
    raise ValueError(f"Unsupported model label: {label}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--frame-cache-root", default=str(DEFAULT_FRAME_CACHE_ROOT))
    parser.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT_DIR))
    parser.add_argument("--qa-manifest", default=str(DEFAULT_QA_MANIFEST))
    parser.add_argument("--items-per-modality", type=int, default=5)
    parser.add_argument("--frame-counts", default=",".join(map(str, DEFAULT_FRAME_COUNTS)))
    parser.add_argument("--modalities", default=",".join(DEFAULT_MODALITIES))
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--rebuild-qa-manifest", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--models", default="qwen_vl,internvl,molmo2")
    parser.add_argument("--qwen-vl-model", default=str(DEFAULT_QWEN_VL_8B))
    parser.add_argument("--internvl-model", default=str(DEFAULT_INTERNVL_8B))
    parser.add_argument("--molmo2-model", default=str(DEFAULT_MOLMO2_8B))
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    input_path = Path(args.input)
    frame_cache_root = Path(args.frame_cache_root)
    experiment_dir = Path(args.experiment_dir)
    qa_manifest_path = Path(args.qa_manifest)
    modalities = tuple(part.strip().lower() for part in args.modalities.split(",") if part.strip())
    frame_counts = _parse_frame_counts(args.frame_counts)
    maximum_frames_per_side = max(frame_counts) // 2

    if args.rebuild_qa_manifest or not qa_manifest_path.exists():
        qa_manifest = build_fixed_qa_manifest(
            input_path=input_path,
            output_path=qa_manifest_path,
            frame_cache_root=frame_cache_root,
            modalities=modalities,
            items_per_modality=max(1, args.items_per_modality),
            maximum_frames_per_side=maximum_frames_per_side,
        )
    else:
        qa_manifest = load_manifest(qa_manifest_path)
        expected = {
            "modalities": list(modalities),
            "items_per_modality": max(1, args.items_per_modality),
        }
        for key, value in expected.items():
            if qa_manifest.get("metadata", {}).get(key) != value:
                raise RuntimeError(
                    f"Existing QA manifest {key}={qa_manifest.get('metadata', {}).get(key)!r}, "
                    f"requested {value!r}; use --rebuild-qa-manifest."
                )
        available_per_side = int(
            qa_manifest.get("metadata", {}).get("minimum_frames_required_per_side", 0)
        )
        if available_per_side < maximum_frames_per_side:
            raise RuntimeError(
                f"Existing QA manifest supports {available_per_side} frames per side, "
                f"but this run needs {maximum_frames_per_side}; use --rebuild-qa-manifest."
            )

    frame_manifests: dict[int, dict[str, Any]] = {}
    maximum_frame_count = max(frame_counts)
    build_order = (
        maximum_frame_count,
        *(count for count in frame_counts if count != maximum_frame_count),
    )
    anchor_manifest: dict[str, Any] | None = None
    for frame_count in build_order:
        path = experiment_dir / "manifests" / f"frames_{frame_count}.json"
        frame_manifests[frame_count] = build_frame_manifest(
            qa_manifest=qa_manifest,
            output_path=path,
            input_path=input_path,
            frame_cache_root=frame_cache_root,
            total_frames=frame_count,
            anchor_manifest=anchor_manifest,
        )
        if frame_count == maximum_frame_count:
            anchor_manifest = frame_manifests[frame_count]
        print(f"frame manifest {frame_count}: {path}")
    print(f"fixed QA manifest: {qa_manifest_path}")
    if args.build_only:
        return

    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; manifests were built but inference cannot run.")

    model_paths = {
        "qwen_vl": args.qwen_vl_model,
        "internvl": args.internvl_model,
        "molmo2": args.molmo2_model,
    }
    requested = [part.strip().lower() for part in args.models.split(",") if part.strip()]
    for label in requested:
        if label not in model_paths:
            raise ValueError(f"Unsupported model label: {label}")
        model_name = model_paths[label]
        adapter = _adapter_for(label, model_name)
        try:
            for frame_count in sorted(frame_counts, reverse=True):
                run_model(
                    label=label,
                    model_name=model_name,
                    adapter=adapter,
                    input_path=input_path,
                    frame_manifest=frame_manifests[frame_count],
                    output_dir=experiment_dir / f"frames_{frame_count}",
                    resume=not args.no_resume,
                )
        finally:
            del adapter
            clear_cuda()


if __name__ == "__main__":
    main()
