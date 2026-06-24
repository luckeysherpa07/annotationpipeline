#!/usr/bin/env python3
"""Run same-question cross-modality frame-input VLM benchmark.

Each source QA item is expanded across requested input modalities. The original
question and answer stay fixed, while the frame input modality changes.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
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
    load_valid_qa_items,
    resolve_frame_cache_candidates,
)
from scripts.run_vlm_4b_aligned_frame_benchmark import _adapter_for

try:
    import torch
except ImportError:
    torch = None


DEFAULT_EXPERIMENT_DIR = Path("outputs/benchmarks/vlm_cross_modality_8frame")
DEFAULT_MANIFEST = DEFAULT_EXPERIMENT_DIR / "cross_modality_frame_manifest.json"
DEFAULT_INPUT_MODALITIES = ("rgb", "ir", "event", "depth")
DEFAULT_SOURCE_MODALITIES = ("rgb", "ir", "event", "depth")
DEFAULT_FRAME_COUNT = 8
DEFAULT_FRAMES_PER_SIDE = 4
GENERATION_CONFIG = {"max_new_tokens": 128, "do_sample": False}


def _json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            converted = dict(row)
            for key, value in converted.items():
                if isinstance(value, (dict, list)):
                    converted[key] = json.dumps(value, ensure_ascii=False)
            writer.writerow(converted)


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value).strip()).strip("_") or "model"


def _frame_number(path: Path) -> int:
    match = re.search(r"frame_(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def _frame_side(path: Path) -> str:
    text = path.as_posix().lower()
    if "/day/" in text or "_day_" in text or "with_light" in text:
        return "day"
    if "/night/" in text or "_night" in text or "no_light" in text:
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
    paths = _dedupe_sorted(paths)
    selected: list[Path] = []
    selected_keys: set[str] = set()
    for frame_number in _referenced_frame_numbers(question):
        candidates = [path for path in paths if path.as_posix() not in selected_keys]
        if not candidates or len(selected) >= count:
            break
        nearest = min(
            candidates,
            key=lambda path: (abs(_frame_number(path) - frame_number), _frame_number(path)),
        )
        selected.append(nearest)
        selected_keys.add(nearest.as_posix())
    remaining = [path for path in paths if path.as_posix() not in selected_keys]
    return [*selected, *_uniform_sample(remaining, count - len(selected))]


def _replace_pair_key_modality(pair_key: str, input_modality: str) -> str:
    parts = list(Path(str(pair_key)).parts)
    if not parts:
        return input_modality
    parts[-1] = input_modality
    return Path(*parts).as_posix()


def _frame_candidates_by_side(
    source_item: dict[str, Any],
    input_modality: str,
    frame_cache_root: Path,
) -> dict[str, list[Path]]:
    lookup_item = {
        **source_item,
        "modality": input_modality,
        "pair_key": _replace_pair_key_modality(str(source_item["pair_key"]), input_modality),
    }
    grouped: dict[str, list[Path]] = {"day": [], "night": [], "unknown": []}
    for path in resolve_frame_cache_candidates(lookup_item, frame_cache_root=frame_cache_root):
        grouped[_frame_side(path)].append(path)
    return {side: _dedupe_sorted(paths) for side, paths in grouped.items()}


def _cross_qa_id(source_qa_id: str, input_modality: str) -> str:
    return f"{source_qa_id}__input_{input_modality}"


def build_cross_modality_manifest(
    *,
    input_path: Path,
    output_path: Path,
    frame_cache_root: Path,
    source_modalities: tuple[str, ...],
    input_modalities: tuple[str, ...],
    frames_per_side: int,
    items_per_source_modality: int | None,
    strict_all_input_modalities: bool,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    selected_source_counts: dict[str, int] = defaultdict(int)
    skipped_source_counts: dict[str, int] = defaultdict(int)
    skipped_input_counts: dict[str, int] = defaultdict(int)
    wanted_sources = set(source_modalities)

    for source_item in load_valid_qa_items(input_path):
        source_modality = str(source_item.get("modality", "")).lower()
        if source_modality not in wanted_sources:
            continue
        if (
            items_per_source_modality is not None
            and selected_source_counts[source_modality] >= items_per_source_modality
        ):
            continue

        sampled_by_input: dict[str, dict[str, list[Path]]] = {}
        missing_inputs: list[str] = []
        for input_modality in input_modalities:
            grouped = _frame_candidates_by_side(source_item, input_modality, frame_cache_root)
            if len(grouped["day"]) < frames_per_side or len(grouped["night"]) < frames_per_side:
                missing_inputs.append(input_modality)
                skipped_input_counts[input_modality] += 1
                continue
            day = _sample_side(grouped["day"], str(source_item["question"]), frames_per_side)
            night = _sample_side(grouped["night"], str(source_item["question"]), frames_per_side)
            sampled_by_input[input_modality] = {"day": day, "night": night}

        if strict_all_input_modalities and missing_inputs:
            skipped_source_counts[f"{source_modality}:missing_{','.join(missing_inputs)}"] += 1
            continue
        if not sampled_by_input:
            skipped_source_counts[f"{source_modality}:missing_all_inputs"] += 1
            continue

        selected_source_counts[source_modality] += 1
        for input_modality in input_modalities:
            sampled = sampled_by_input.get(input_modality)
            if sampled is None:
                continue
            day = sampled["day"]
            night = sampled["night"]
            input_pair_key = _replace_pair_key_modality(str(source_item["pair_key"]), input_modality)
            rows.append(
                {
                    "qa_id": _cross_qa_id(str(source_item["qa_id"]), input_modality),
                    "source_qa_id": source_item["qa_id"],
                    "source_modality": source_modality,
                    "input_modality": input_modality,
                    "modality": input_modality,
                    "section": source_item["section"],
                    "source_section": source_item["section"],
                    "source_pair_key": source_item["pair_key"],
                    "pair_key": input_pair_key,
                    "question": source_item["question"],
                    "ground_truth_answer": source_item["answer"],
                    "caption": source_item.get("caption", ""),
                    "day_frames": [path.as_posix() for path in day],
                    "night_frames": [path.as_posix() for path in night],
                    "frame_paths": [path.as_posix() for path in [*day, *night]],
                }
            )

    metadata = {
        "manifest_type": "same_question_cross_modality_frame_sampling_v1",
        "input_path": input_path.as_posix(),
        "frame_cache_root": frame_cache_root.as_posix(),
        "source_modalities": list(source_modalities),
        "input_modalities": list(input_modalities),
        "items_per_source_modality": items_per_source_modality or 0,
        "strict_all_input_modalities": strict_all_input_modalities,
        "total_frames": frames_per_side * 2,
        "frames_per_side": frames_per_side,
        "frame_order": "day_then_night",
        "sampling_algorithm": "referenced_or_nearest_then_stratified_uniform_v2",
        "selected_source_questions": sum(selected_source_counts.values()),
        "selected_items": len(rows),
        "selected_source_counts": dict(sorted(selected_source_counts.items())),
        "expanded_counts_by_input_modality": dict(
            sorted(Counter(row["input_modality"] for row in rows).items())
        ),
        "expanded_counts_by_source_modality": dict(
            sorted(Counter(row["source_modality"] for row in rows).items())
        ),
        "skipped_source_counts": dict(sorted(skipped_source_counts.items())),
        "skipped_insufficient_input_counts": dict(sorted(skipped_input_counts.items())),
    }
    payload = {"metadata": metadata, "items": rows}
    payload["metadata"]["manifest_sha256"] = _json_hash(payload["items"])
    _write_json(output_path, payload)
    return payload


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError(f"Invalid cross-modality manifest: {path}")
    return payload


def _resolve_frame_paths(entry: dict[str, Any]) -> list[Path]:
    paths = [Path(path) for path in entry.get("frame_paths", [])]
    missing = [path.as_posix() for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Manifest entry {entry.get('qa_id')} has missing frames: {missing}")
    return paths


def _load_results(path: Path, *, model_name: str, manifest_sha256: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        return {}
    for key, expected in {"model_name": model_name, "frame_manifest_sha256": manifest_sha256}.items():
        if metadata.get(key) != expected:
            raise RuntimeError(
                f"Existing result {key}={metadata.get(key)!r}, requested {expected!r}: {path}. "
                "Use --no-resume or a different --experiment-dir."
            )
    results = payload.get("results")
    return (
        {str(key): value for key, value in results.items() if isinstance(value, dict)}
        if isinstance(results, dict)
        else {}
    )


def _save_results(path: Path, results: dict[str, dict[str, Any]], metadata: dict[str, Any]) -> None:
    _write_json(path, {"results": results, "metadata": metadata})
    _write_csv(path.with_suffix(".csv"), list(results.values()))


def _completed(result: dict[str, Any]) -> bool:
    return result.get("status") == "answered" and bool(str(result.get("model_answer", "")).strip())


def _is_oom(exc: Exception) -> bool:
    return "out of memory" in str(exc).lower() or (
        torch is not None and isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", RuntimeError))
    )


def _group_rows_by_frame_set(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(entries, key=lambda row: tuple(row.get("frame_paths", [])))


def run_model(
    *,
    label: str,
    model_name: str,
    adapter: Any,
    manifest: dict[str, Any],
    manifest_path: Path,
    output_dir: Path,
    resume: bool,
    max_items_per_input_modality: int | None,
) -> None:
    safe_model = _safe_name(Path(model_name).name)
    output_json = output_dir / f"{label}_{safe_model}.json"
    results = (
        _load_results(
            output_json,
            model_name=model_name,
            manifest_sha256=manifest["metadata"]["manifest_sha256"],
        )
        if resume
        else {}
    )

    rows = list(manifest["items"])
    if max_items_per_input_modality is not None:
        selected: list[dict[str, Any]] = []
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            input_modality = str(row.get("input_modality", row.get("modality", ""))).lower()
            if counts[input_modality] >= max_items_per_input_modality:
                continue
            selected.append(row)
            counts[input_modality] += 1
        rows = selected
    rows = _group_rows_by_frame_set(rows)

    for entry in rows:
        qa_id = str(entry["qa_id"])
        if _completed(results.get(qa_id, {})):
            continue

        frame_paths = _resolve_frame_paths(entry)
        item_for_model = {
            "qa_id": qa_id,
            "modality": entry["input_modality"],
            "section": entry["source_section"],
            "pair_key": entry["pair_key"],
            "question": entry["question"],
            "answer": entry["ground_truth_answer"],
        }

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
        cache_hits_before = int(getattr(adapter, "frame_cache_hits", 0))
        try:
            answer = adapter.answer(item_for_model, frame_paths)
            status = "answered" if answer else "failed"
            reason = "" if answer else "Frame answer call failed: empty model answer"
        except Exception as exc:
            status = "oom" if _is_oom(exc) else "failed"
            reason = f"Frame answer call failed: {type(exc).__name__}: {exc}"

        latency = time.perf_counter() - started
        peak_bytes = torch.cuda.max_memory_allocated() if torch is not None and torch.cuda.is_available() else 0
        results[qa_id] = {
            "qa_id": qa_id,
            "source_qa_id": entry["source_qa_id"],
            "provider": label,
            "model_name": model_name,
            "modality": entry["input_modality"],
            "input_modality": entry["input_modality"],
            "source_modality": entry["source_modality"],
            "section": entry["source_section"],
            "source_section": entry["source_section"],
            "pair_key": entry["pair_key"],
            "source_pair_key": entry["source_pair_key"],
            "question": entry["question"],
            "ground_truth_answer": entry["ground_truth_answer"],
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
            "frame_cache_hit": int(getattr(adapter, "frame_cache_hits", 0)) > cache_hits_before,
            "generation_config": dict(GENERATION_CONFIG),
        }

        metadata = {
            "benchmark_type": "same_question_cross_modality_8frame_input_v1",
            "provider": label,
            "model_name": model_name,
            "quantization": getattr(adapter, "quantization", "adapter_default"),
            "input_path": manifest["metadata"]["input_path"],
            "frame_manifest_path": manifest_path.as_posix(),
            "frame_manifest_sha256": manifest["metadata"]["manifest_sha256"],
            "frame_manifest_total_frames": manifest["metadata"]["total_frames"],
            "frame_manifest_frames_per_side": manifest["metadata"]["frames_per_side"],
            "frame_manifest_order": manifest["metadata"]["frame_order"],
            "frame_manifest_sampling_algorithm": manifest["metadata"]["sampling_algorithm"],
            "total_manifest_items": len(manifest["items"]),
            "run_item_limit_per_input_modality": max_items_per_input_modality or 0,
            "run_items": len(rows),
            "attempted_items": len(results),
            "status_counts": dict(Counter(row["status"] for row in results.values())),
            "frame_cache_level": getattr(adapter, "frame_cache_level", "none"),
            "frame_cache_hits": int(getattr(adapter, "frame_cache_hits", 0)),
            "frame_cache_misses": int(getattr(adapter, "frame_cache_misses", 0)),
            "generation_config": dict(GENERATION_CONFIG),
            "resume": resume,
        }
        _save_results(output_json, results, metadata)
        print(
            f"{label} {qa_id}: {status}, source={entry['source_modality']}, "
            f"input={entry['input_modality']}, latency={latency:.2f}s, "
            f"peak={peak_bytes / 1024**3:.2f}GB"
        )


def _parse_modalities(raw: str) -> tuple[str, ...]:
    values = tuple(item.strip().lower() for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("At least one modality is required.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT_DIR))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--frame-cache-root", default=str(DEFAULT_FRAME_CACHE_ROOT))
    parser.add_argument("--source-modalities", default=",".join(DEFAULT_SOURCE_MODALITIES))
    parser.add_argument("--input-modalities", default=",".join(DEFAULT_INPUT_MODALITIES))
    parser.add_argument("--frames-per-side", type=int, default=DEFAULT_FRAMES_PER_SIDE)
    parser.add_argument("--items-per-source-modality", type=int, default=0)
    parser.add_argument(
        "--allow-partial-input-coverage",
        action="store_true",
        help="Keep a source question if at least one requested input modality has enough frames.",
    )
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument("--build-manifest-only", action="store_true")
    parser.add_argument("--models", default="qwen_vl,internvl,molmo2")
    parser.add_argument("--qwen-vl-model", default="models/qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--internvl-model", default="models/internvl/InternVL3_5-4B-Instruct")
    parser.add_argument("--molmo2-model", default="models/molmo2/Molmo2-4B")
    parser.add_argument("--max-items-per-input-modality", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    experiment_dir = Path(args.experiment_dir)
    manifest_path = Path(args.manifest)
    input_path = Path(args.input)
    frame_cache_root = Path(args.frame_cache_root)
    source_modalities = _parse_modalities(args.source_modalities)
    input_modalities = _parse_modalities(args.input_modalities)
    items_per_source_modality = (
        None if args.items_per_source_modality == 0 else max(1, args.items_per_source_modality)
    )
    max_items_per_input_modality = (
        None if args.max_items_per_input_modality == 0 else max(1, args.max_items_per_input_modality)
    )

    if args.rebuild_manifest or not manifest_path.exists():
        manifest = build_cross_modality_manifest(
            input_path=input_path,
            output_path=manifest_path,
            frame_cache_root=frame_cache_root,
            source_modalities=source_modalities,
            input_modalities=input_modalities,
            frames_per_side=max(1, args.frames_per_side),
            items_per_source_modality=items_per_source_modality,
            strict_all_input_modalities=not args.allow_partial_input_coverage,
        )
    else:
        manifest = load_manifest(manifest_path)

    print(f"cross-modality manifest: {manifest_path}")
    print(json.dumps(manifest["metadata"], indent=2, ensure_ascii=False))
    if args.build_manifest_only:
        return

    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; manifest built but inference cannot run.")

    model_paths = {
        "qwen_vl": args.qwen_vl_model,
        "internvl": args.internvl_model,
        "molmo2": args.molmo2_model,
    }
    requested_models = [item.strip() for item in args.models.split(",") if item.strip()]
    for label in requested_models:
        if label not in model_paths:
            raise ValueError(f"Unsupported model label {label!r}; choose from {sorted(model_paths)}")

    experiment_dir.mkdir(parents=True, exist_ok=True)
    for label in requested_models:
        model_name = model_paths[label]
        adapter = _adapter_for(label, model_name)
        try:
            run_model(
                label=label,
                model_name=model_name,
                adapter=adapter,
                manifest=manifest,
                manifest_path=manifest_path,
                output_dir=experiment_dir,
                resume=not args.no_resume,
                max_items_per_input_modality=max_items_per_input_modality,
            )
        finally:
            clear_frame_cache = getattr(adapter, "clear_frame_cache", None)
            if callable(clear_frame_cache):
                clear_frame_cache()
            del adapter
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
