#!/usr/bin/env python3
"""Run and summarize a fixed Pass 1 development evaluation set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from annotation_feature.aligned_multimodal_caption_pipeline import (  # noqa: E402
    DEFAULT_COMPOSITE_ROOT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_MAX_TRANSPORT_RETRIES,
)
from annotation_feature.aligned_multimodal_caption_two_pass_pipeline import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    run_caption_pipeline_pass1,
)

DEFAULT_MANIFEST = REPO_ROOT / "evaluation" / "pass1_dev_manifest.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "pass1_evaluation"
BASELINE_PROMPT_VERSION = "pass1_prompt_v1.1"
RUN_FILE_PATTERN = re.compile(r"^run_(\d{3,})\.json$")
FROZEN_BASELINE_FILES = (
    "annotation_feature/aligned_caption_pass1_prompt.py",
    "annotation_feature/aligned_caption_schema.py",
    "annotation_feature/aligned_caption_pass1_validation.py",
    "annotation_feature/aligned_caption_validation.py",
    "annotation_feature/aligned_multimodal_caption_two_pass_pipeline.py",
    "tests/test_pass1_prompt_rules.py",
)

PIPELINE_DEFAULTS: dict[str, Any] = {
    "dataset_root": DEFAULT_DATASET_ROOT.as_posix(),
    "composite_root": DEFAULT_COMPOSITE_ROOT.as_posix(),
    "model_name": DEFAULT_MODEL_NAME,
    "generation_mode": "gemini",
    "api_key_source": "list",
    "num_uniform_frames": 8,
    "num_adaptive_frames": 2,
    "pairs": None,
    "directions": None,
    "sides": None,
    "limit": None,
    "limit_scenes": None,
    "limit_scene_folders": None,
    "target_paths": None,
    "max_retries": 3,
    "max_transport_retries": DEFAULT_MAX_TRANSPORT_RETRIES,
    "delay_between_calls": 0,
    "checkpoint_every": 1,
}

SUMMARY_FIELDS = [
    "evaluation_set",
    "prompt_version",
    "sample_id",
    "run_id",
    "status",
    "completed_item_count",
    "skipped_item_count",
    "first_validation_attempt_success",
    "validation_attempt_count",
    "api_call_count",
    "transport_retry_count",
    "final_error_category",
    "final_error_categories",
    "validation_warning_count",
    "retry_error_categories",
    "physical_entity_count",
    "video1_atom_count",
    "video2_atom_count",
    "video1_uncertainty_count",
    "video2_uncertainty_count",
    "video1_missing_target_count",
    "video2_missing_target_count",
    "model_name",
    "input_path",
    "git_commit",
    "git_dirty",
    "raw_output_path",
]


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("._-")
    if not safe:
        raise ValueError(f"Value cannot be converted to a safe path component: {value!r}")
    return safe


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _display_path(path: str | Path) -> str:
    resolved = _repo_path(path)
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _validate_pipeline_config(config: dict[str, Any], field: str) -> None:
    unknown = sorted(set(config) - set(PIPELINE_DEFAULTS))
    if unknown:
        raise ValueError(f"{field} contains unsupported keys: {', '.join(unknown)}")


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = _repo_path(path)
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("Evaluation manifest must be a JSON object")

    evaluation_set = _require_nonempty_string(manifest.get("evaluation_set"), "evaluation_set")
    prompt_version = _require_nonempty_string(manifest.get("prompt_version"), "prompt_version")
    if prompt_version != BASELINE_PROMPT_VERSION:
        raise ValueError(
            f"This runner freezes {BASELINE_PROMPT_VERSION}; manifest requested {prompt_version!r}"
        )
    runs_per_sample = manifest.get("runs_per_sample")
    if not isinstance(runs_per_sample, int) or isinstance(runs_per_sample, bool) or runs_per_sample < 1:
        raise ValueError("runs_per_sample must be a positive integer")

    pipeline = manifest.get("pipeline", {})
    if not isinstance(pipeline, dict):
        raise ValueError("pipeline must be an object when present")
    _validate_pipeline_config(pipeline, "pipeline")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("samples must be a non-empty list")

    seen_ids: set[str] = set()
    normalized_samples: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        field = f"samples[{index}]"
        if not isinstance(sample, dict):
            raise ValueError(f"{field} must be an object")
        sample_id = _require_nonempty_string(sample.get("sample_id"), f"{field}.sample_id")
        safe_sample_id = _safe_name(sample_id)
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        if safe_sample_id in {_safe_name(value) for value in seen_ids}:
            raise ValueError(f"sample_id path collision after normalization: {sample_id}")
        seen_ids.add(sample_id)
        input_path = _require_nonempty_string(sample.get("input"), f"{field}.input")
        tags = sample.get("tags", [])
        if not isinstance(tags, list) or any(not isinstance(tag, str) or not tag.strip() for tag in tags):
            raise ValueError(f"{field}.tags must be a string list")
        sample_pipeline = sample.get("pipeline", {})
        if not isinstance(sample_pipeline, dict):
            raise ValueError(f"{field}.pipeline must be an object when present")
        _validate_pipeline_config(sample_pipeline, f"{field}.pipeline")
        normalized_samples.append({
            **sample,
            "sample_id": sample_id,
            "input": input_path,
            "tags": tags,
            "pipeline": sample_pipeline,
        })

    return {
        **manifest,
        "evaluation_set": evaluation_set,
        "prompt_version": prompt_version,
        "runs_per_sample": runs_per_sample,
        "pipeline": pipeline,
        "samples": normalized_samples,
        "manifest_path": manifest_path,
    }


def next_run_path(sample_dir: Path) -> tuple[str, Path, Path]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    run_number = 1
    while True:
        run_id = f"run_{run_number:03d}"
        output_path = sample_dir / f"{run_id}.json"
        lock_path = sample_dir / f"{run_id}.lock"
        if output_path.exists() or lock_path.exists():
            run_number += 1
            continue
        try:
            lock_path.touch(exist_ok=False)
        except FileExistsError:
            run_number += 1
            continue
        return run_id, output_path, lock_path


def _git_fingerprint() -> dict[str, Any]:
    result: dict[str, Any] = {"commit": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        result = {"commit": commit or None, "dirty": bool(status.strip())}
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _baseline_fingerprint() -> dict[str, Any]:
    files = {
        relative_path: _sha256_file(REPO_ROOT / relative_path)
        for relative_path in FROZEN_BASELINE_FILES
    }
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "combined_sha256": hashlib.sha256(encoded).hexdigest(),
        "files": files,
    }


def _item_diagnostics(item: dict[str, Any]) -> dict[str, Any]:
    diagnostics = item.get("diagnostics")
    return diagnostics if isinstance(diagnostics, dict) else item


def _integer(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _retry_categories(items: Iterable[dict[str, Any]]) -> list[str]:
    categories: list[str] = []
    for item in items:
        history = _item_diagnostics(item).get("retry_history", [])
        for entry in _list(history):
            if isinstance(entry, dict):
                category = entry.get("category") or entry.get("type")
                if isinstance(category, str) and category.strip():
                    categories.append(category.strip())
    return categories


def _evidence_counts(items: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = Counter({
        "physical_entity_count": 0,
        "video1_atom_count": 0,
        "video2_atom_count": 0,
        "video1_uncertainty_count": 0,
        "video2_uncertainty_count": 0,
        "video1_missing_target_count": 0,
        "video2_missing_target_count": 0,
    })
    for item in items:
        evidence = item.get("evidence")
        if not isinstance(evidence, dict):
            continue
        global_scene = evidence.get("global_scene")
        if isinstance(global_scene, dict):
            counts["physical_entity_count"] += len(_list(global_scene.get("physical_entities")))
        for source_number in (1, 2):
            analysis = evidence.get(f"video{source_number}_analysis")
            if not isinstance(analysis, dict):
                continue
            counts[f"video{source_number}_atom_count"] += len(_list(analysis.get("information_atoms")))
            counts[f"video{source_number}_uncertainty_count"] += len(_list(analysis.get("uncertain_observations")))
            counts[f"video{source_number}_missing_target_count"] += len(_list(analysis.get("missing_key_attributes")))
    return dict(counts)


def extract_run_metrics(
    payload: dict[str, Any],
    *,
    sample_id: str,
    run_id: str,
    evaluation_set: str,
    prompt_version: str,
    raw_output_path: str | Path,
) -> dict[str, Any]:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    items = [item for item in _list(payload.get("items")) if isinstance(item, dict)]
    skipped = [item for item in _list(payload.get("skipped")) if isinstance(item, dict)]
    all_records = [*items, *skipped]

    completed_count = len(items)
    skipped_count = len(skipped)
    evaluation_error = metadata.get("evaluation_error")
    has_runner_error = isinstance(evaluation_error, dict)
    if completed_count and not skipped_count and not has_runner_error:
        status = "success"
    elif completed_count:
        status = "partial"
    else:
        status = "failed"

    first_attempt_values = [
        value
        for record in all_records
        for value in [_item_diagnostics(record).get("first_validation_attempt_success")]
        if isinstance(value, bool)
    ]
    first_attempt_success: bool | None = None
    if first_attempt_values:
        first_attempt_success = all(first_attempt_values)

    final_error_categories = [
        str(record["final_error_category"])
        for record in skipped
        if record.get("final_error_category")
    ]
    if not final_error_categories and not completed_count:
        initialization_category = metadata.get("initialization_error_category")
        if isinstance(initialization_category, str) and initialization_category.strip():
            final_error_categories.append(initialization_category.strip())
        elif has_runner_error:
            final_error_categories.append("evaluation_runner_error")
        else:
            final_error_categories.append("no_completed_items")
    retry_categories = _retry_categories(all_records)
    evidence_counts = _evidence_counts(all_records)
    fingerprint = metadata.get("reproducibility")
    fingerprint = fingerprint if isinstance(fingerprint, dict) else {}
    git_info = fingerprint.get("git") if isinstance(fingerprint.get("git"), dict) else {}

    metrics: dict[str, Any] = {
        "evaluation_set": evaluation_set,
        "prompt_version": prompt_version,
        "sample_id": sample_id,
        "run_id": run_id,
        "status": status,
        "completed_item_count": completed_count,
        "skipped_item_count": skipped_count,
        "first_validation_attempt_success": first_attempt_success,
        "validation_attempt_count": sum(
            _integer(_item_diagnostics(record).get("validation_attempts")) for record in all_records
        ),
        "api_call_count": sum(
            _integer(_item_diagnostics(record).get("api_calls")) for record in all_records
        ),
        "transport_retry_count": sum(
            _integer(_item_diagnostics(record).get("transport_retries")) for record in all_records
        ),
        "final_error_category": final_error_categories[0] if len(final_error_categories) == 1 else None,
        "final_error_categories": final_error_categories,
        "validation_warning_count": sum(len(_list(record.get("validation_warnings"))) for record in all_records),
        "retry_error_categories": retry_categories,
        "model_name": metadata.get("model_name") or fingerprint.get("model_name"),
        "input_path": metadata.get("input") or fingerprint.get("input_path"),
        "git_commit": git_info.get("commit"),
        "git_dirty": git_info.get("dirty"),
        "raw_output_path": Path(raw_output_path).as_posix(),
        **evidence_counts,
    }
    return metrics


def _numeric_mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [row.get(field) for row in rows]
    numeric = [float(value) for value in values if isinstance(value, int | float) and not isinstance(value, bool)]
    return mean(numeric) if numeric else None


def _counter_values(rows: Iterable[dict[str, Any]], field: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        values = row.get(field)
        if isinstance(values, list):
            counter.update(str(value) for value in values if value)
        elif values:
            counter[str(values)] += 1
    return dict(sorted(counter.items()))


def _aggregate_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    successful = sum(row.get("status") == "success" for row in rows)
    partial = sum(row.get("status") == "partial" for row in rows)
    failed = total - successful
    first_attempt_values = [
        row["first_validation_attempt_success"]
        for row in rows
        if isinstance(row.get("first_validation_attempt_success"), bool)
    ]
    return {
        "total_runs": total,
        "successful_runs": successful,
        "partial_runs": partial,
        "failed_runs": failed,
        "success_rate": successful / total if total else None,
        "first_attempt_success_rate": (
            sum(first_attempt_values) / len(first_attempt_values) if first_attempt_values else None
        ),
        "mean_validation_attempts": _numeric_mean(rows, "validation_attempt_count"),
        "mean_api_calls": _numeric_mean(rows, "api_call_count"),
        "error_category_counts": _counter_values(rows, "final_error_categories"),
        "retry_error_category_counts": _counter_values(rows, "retry_error_categories"),
        "warning_count": sum(_integer(row.get("validation_warning_count")) for row in rows),
    }


def aggregate_metrics(rows: list[dict[str, Any]], evaluation_set: str, prompt_version: str) -> dict[str, Any]:
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sample[str(row.get("sample_id", "unknown"))].append(row)
    return {
        "evaluation_set": evaluation_set,
        "prompt_version": prompt_version,
        "metric_definitions": {
            "success": "At least one completed item and no skipped items or evaluation-runner error.",
            "partial": "At least one completed item plus skipped items or an evaluation-runner error.",
            "failed": "No completed items; partial runs are included in failed_runs.",
            "first_attempt_success_rate": "Mean of non-null run-level booleans; a run is true only when all recorded items succeeded validation on their first attempt.",
            "mean_validation_attempts": "Mean of per-run validation-attempt totals.",
            "mean_api_calls": "Mean of per-run API-call totals.",
        },
        **_aggregate_group(rows),
        "per_sample": {
            sample_id: _aggregate_group(sample_rows)
            for sample_id, sample_rows in sorted(by_sample.items())
        },
        "runs": rows,
    }


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return ""
    return value


def write_reports(evaluation_dir: Path, rows: list[dict[str, Any]], evaluation_set: str, prompt_version: str) -> None:
    aggregate = aggregate_metrics(rows, evaluation_set, prompt_version)
    _json_dump(evaluation_dir / "summary.json", aggregate)
    with open(evaluation_dir / "summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in SUMMARY_FIELDS})


def _load_run_rows(evaluation_dir: Path, evaluation_set: str, prompt_version: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for output_path in sorted(evaluation_dir.glob("*/run_*.json")):
        match = RUN_FILE_PATTERN.match(output_path.name)
        if not match:
            continue
        try:
            with open(output_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            output_metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            evaluation = output_metadata.get("evaluation") if isinstance(output_metadata.get("evaluation"), dict) else {}
            sample_id = str(evaluation.get("sample_id") or output_path.parent.name)
            run_id = str(evaluation.get("run_id") or output_path.stem)
            rows.append(extract_run_metrics(
                payload,
                sample_id=sample_id,
                run_id=run_id,
                evaluation_set=str(evaluation.get("evaluation_set") or evaluation_set),
                prompt_version=str(evaluation.get("prompt_version") or prompt_version),
                raw_output_path=output_path.relative_to(REPO_ROOT) if output_path.is_relative_to(REPO_ROOT) else output_path,
            ))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            print(f"WARNING: Could not summarize {output_path}: {exc}")
    return rows


def _pipeline_config(manifest: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    config = {**PIPELINE_DEFAULTS, **manifest.get("pipeline", {}), **sample.get("pipeline", {})}
    config["dataset_root"] = _repo_path(config["dataset_root"])
    config["composite_root"] = _repo_path(config["composite_root"])
    return config


def _annotate_output(
    payload: dict[str, Any],
    *,
    manifest: dict[str, Any],
    sample: dict[str, Any],
    run_id: str,
    config: dict[str, Any],
    output_path: Path,
    git_info: dict[str, Any],
    baseline_fingerprint: dict[str, Any],
    runner_error: Exception | None,
) -> dict[str, Any]:
    items = [item for item in _list(payload.get("items")) if isinstance(item, dict)]
    skipped = [item for item in _list(payload.get("skipped")) if isinstance(item, dict)]
    selection = [
        {
            "caption_id": item.get("caption_id"),
            "selection_config_fingerprint": item.get("selection_config_fingerprint"),
            "sampling_strategy": item.get("sampling_strategy"),
            "selected_frame_indexes": item.get("selected_frame_indexes"),
        }
        for item in [*items, *skipped]
    ]
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        payload["metadata"] = metadata
    metadata["evaluation"] = {
        "evaluation_set": manifest["evaluation_set"],
        "prompt_version": manifest["prompt_version"],
        "sample_id": sample["sample_id"],
        "run_id": run_id,
        "tags": sample.get("tags", []),
    }
    metadata["reproducibility"] = {
        "prompt_version": manifest["prompt_version"],
        "model_name": config["model_name"],
        "input_path": _display_path(sample["input"]),
        "input_sha256": _sha256_file(_repo_path(sample["input"])),
        "pipeline_config": {
            key: value.as_posix() if isinstance(value, Path) else value
            for key, value in config.items()
        },
        "selection": selection,
        "git": git_info,
        "frozen_baseline": baseline_fingerprint,
    }
    if runner_error is not None:
        metadata["evaluation_error"] = {
            "category": type(runner_error).__name__,
            "message": str(runner_error),
        }
        if not items and not skipped:
            payload["skipped"] = [{
                "status": "failed",
                "reason": str(runner_error),
                "final_error_category": "evaluation_runner_error",
                "evidence": None,
                "validation_warnings": [],
                "api_calls": 0,
                "validation_attempts": 0,
                "transport_retries": 0,
                "first_validation_attempt_success": False,
                "retry_history": [],
            }]
    _json_dump(output_path, payload)
    return payload


def run_evaluation(manifest_path: str | Path, output_root: str | Path) -> Path:
    manifest = load_manifest(manifest_path)
    evaluation_dir = _repo_path(output_root) / _safe_name(manifest["evaluation_set"])
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    git_info = _git_fingerprint()
    baseline_fingerprint = _baseline_fingerprint()

    for sample in manifest["samples"]:
        sample_dir = evaluation_dir / _safe_name(sample["sample_id"])
        config = _pipeline_config(manifest, sample)
        input_path = _repo_path(sample["input"])
        for _ in range(manifest["runs_per_sample"]):
            run_id, output_path, lock_path = next_run_path(sample_dir)
            payload: dict[str, Any] = {"metadata": {}, "items": [], "skipped": []}
            runner_error: Exception | None = None
            try:
                run_caption_pipeline_pass1(
                    input_path=input_path,
                    output_path=output_path,
                    resume=False,
                    **config,
                )
                with open(output_path, "r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                if not isinstance(loaded, dict):
                    raise ValueError("Pass 1 pipeline output must be a JSON object")
                payload = loaded
            except Exception as exc:
                runner_error = exc
                if output_path.exists():
                    try:
                        with open(output_path, "r", encoding="utf-8") as handle:
                            loaded = json.load(handle)
                        if isinstance(loaded, dict):
                            payload = loaded
                    except (OSError, ValueError, TypeError, json.JSONDecodeError):
                        pass
                print(f"WARNING: Evaluation run {sample['sample_id']}/{run_id} failed: {exc}")
            finally:
                _annotate_output(
                    payload,
                    manifest=manifest,
                    sample=sample,
                    run_id=run_id,
                    config=config,
                    output_path=output_path,
                    git_info=git_info,
                    baseline_fingerprint=baseline_fingerprint,
                    runner_error=runner_error,
                )
                lock_path.unlink(missing_ok=True)

            rows = _load_run_rows(evaluation_dir, manifest["evaluation_set"], manifest["prompt_version"])
            write_reports(evaluation_dir, rows, manifest["evaluation_set"], manifest["prompt_version"])

    print(f"Wrote Pass 1 evaluation to {evaluation_dir}")
    return evaluation_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST.as_posix())
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT.as_posix())
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    run_evaluation(args.manifest, args.output_root)


if __name__ == "__main__":
    main()
