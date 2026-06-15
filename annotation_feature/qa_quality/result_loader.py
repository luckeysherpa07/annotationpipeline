"""Load heterogeneous VLM answer files into one evaluation schema."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class EvaluationRecord:
    record_id: str
    source_path: str
    qa_id: str
    provider: str
    model_name: str
    input_type: str
    benchmark_type: str
    modality: str
    section: str
    pair_key: str
    question: str
    ground_truth_answer: str
    model_answer: str
    status: str
    reason: str
    latency_seconds: float | None
    baseline_gpu_gb: float | None
    peak_gpu_gb: float | None
    incremental_peak_gpu_gb: float | None
    source_metadata: dict[str, Any]
    frame_count: int | None = None
    max_frames_per_item: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _input_type(metadata: dict[str, Any], row: dict[str, Any]) -> str:
    benchmark_type = str(metadata.get("benchmark_type", "")).lower()
    if "video" in benchmark_type or "day_video" in row or "night_video" in row:
        return "video"
    if "frame" in benchmark_type or "frame_paths" in row:
        return "frame"
    return "unknown"


def _rows_from_payload(payload: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("Result payload must be a JSON object")
    metadata = payload.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    rows = payload.get("results", payload.get("items"))
    if isinstance(rows, dict):
        return [row for row in rows.values() if isinstance(row, dict)], metadata
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)], metadata
    raise ValueError("Result payload does not contain a results/items collection")


def _resolve_metadata_path(result_path: Path, raw_path: Any) -> Path | None:
    if not raw_path:
        return None
    candidate = Path(str(raw_path)).expanduser()
    candidates = [candidate] if candidate.is_absolute() else [
        Path.cwd() / candidate,
        result_path.parent / candidate,
    ]
    for resolved in candidates:
        if resolved.is_file():
            return resolved.resolve()
    return None


def _qa_lookup_from_metadata(
    result_path: Path,
    metadata: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], str]:
    source_path = _resolve_metadata_path(result_path, metadata.get("input_path"))
    if source_path is None:
        return {}, ""
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}, ""

    if isinstance(payload, dict):
        rows = payload.get("valid_qa", payload.get("results", payload.get("items")))
    else:
        rows = payload
    if isinstance(rows, dict):
        iterable = rows.values()
    elif isinstance(rows, list):
        iterable = rows
    else:
        return {}, ""

    lookup = {
        str(row.get("qa_id") or row.get("id")): row
        for row in iterable
        if isinstance(row, dict) and (row.get("qa_id") or row.get("id"))
    }
    return lookup, source_path.as_posix()


def load_result_file(path: Path | str) -> list[EvaluationRecord]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows, metadata = _rows_from_payload(payload)
    qa_lookup, qa_source_path = _qa_lookup_from_metadata(path, metadata)
    benchmark_type = str(metadata.get("benchmark_type", "unknown"))
    records: list[EvaluationRecord] = []
    for index, row in enumerate(rows):
        qa_id = str(row.get("qa_id") or row.get("id") or f"row-{index}")
        source_qa = qa_lookup.get(qa_id, {})
        provider = str(row.get("provider") or metadata.get("provider") or "unknown")
        model_name = str(row.get("model_name") or metadata.get("model_name") or "unknown")
        identity = "\n".join((path.resolve().as_posix(), qa_id, provider, model_name))
        record_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        record_metadata = dict(metadata)
        if qa_source_path:
            record_metadata["ground_truth_source_path"] = qa_source_path
        records.append(
            EvaluationRecord(
                record_id=record_id,
                source_path=path.as_posix(),
                qa_id=qa_id,
                provider=provider,
                model_name=model_name,
                input_type=_input_type(metadata, row),
                benchmark_type=benchmark_type,
                modality=str(row.get("modality") or "unknown").lower(),
                section=str(row.get("section") or "unknown").lower(),
                pair_key=str(row.get("pair_key") or ""),
                question=str(row.get("question") or ""),
                ground_truth_answer=str(
                    row.get("ground_truth_answer")
                    or row.get("gold_answer")
                    or row.get("answer")
                    or source_qa.get("ground_truth_answer")
                    or source_qa.get("gold_answer")
                    or source_qa.get("answer")
                    or ""
                ),
                model_answer=str(row.get("model_answer") or ""),
                status=str(row.get("status") or "unknown").lower(),
                reason=str(row.get("reason") or ""),
                latency_seconds=_optional_float(row.get("latency_seconds")),
                baseline_gpu_gb=_optional_float(row.get("baseline_gpu_gb")),
                peak_gpu_gb=_optional_float(row.get("peak_gpu_gb")),
                incremental_peak_gpu_gb=_optional_float(
                    row.get("incremental_peak_gpu_gb")
                ),
                source_metadata=record_metadata,
                frame_count=_optional_int(
                    row.get("frame_count")
                    or (
                        len(row["frame_paths"])
                        if isinstance(row.get("frame_paths"), list)
                        else None
                    )
                ),
                max_frames_per_item=_optional_int(
                    metadata.get("max_frames_per_item")
                    or metadata.get("max_frames")
                ),
            )
        )
    return records


def discover_result_files(inputs: Iterable[Path | str]) -> list[Path]:
    discovered: set[Path] = set()
    for raw in inputs:
        path = Path(raw)
        if path.is_file() and path.suffix.lower() == ".json":
            discovered.add(path)
        elif path.is_dir():
            for candidate in path.rglob("*.json"):
                name = candidate.name.lower()
                if "manifest" in name or name in {"summary.json", "per_item_scores.json"}:
                    continue
                discovered.add(candidate)
        else:
            raise FileNotFoundError(f"Evaluation input does not exist: {path}")
    return sorted(discovered)


def load_evaluation_records(
    inputs: Iterable[Path | str],
) -> tuple[list[EvaluationRecord], list[dict[str, str]]]:
    records: list[EvaluationRecord] = []
    skipped: list[dict[str, str]] = []
    for path in discover_result_files(inputs):
        try:
            loaded = load_result_file(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            skipped.append({"path": path.as_posix(), "reason": str(exc)})
            continue
        if not loaded:
            skipped.append({"path": path.as_posix(), "reason": "No result rows"})
            continue
        records.extend(loaded)
    return records, skipped
