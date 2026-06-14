"""Reference-guided Gemini judge for generated VLM answers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

from annotation_feature.pipeline.client import create_gemini_client
from annotation_feature.pipeline.gemini_retry import (
    call_with_retry,
    is_service_unavailable_error,
)

from .benchmark import (
    DEFAULT_GEMINI_API_KEY_LIST_PATH,
    is_quota_error,
    load_api_keys,
)
from .result_loader import EvaluationRecord


DEFAULT_JUDGE_MODEL = "gemini-3.1-flash-lite"
JUDGE_PROMPT_VERSION = "reference_guided_vlm_answer_judge_v1"
JUDGE_CACHE_SCHEMA_VERSION = 3
VALID_LABELS = {"correct", "partially_correct", "incorrect", "unjudgeable"}
LABEL_SCORES = {
    "correct": 1.0,
    "partially_correct": 0.5,
    "incorrect": 0.0,
    "unjudgeable": None,
}


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(text).strip(), flags=re.I)
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if not match:
        raise ValueError("Judge response does not contain a JSON object")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("Judge response must be a JSON object")
    return payload


def build_judge_prompt(records: Iterable[EvaluationRecord]) -> str:
    items = [
        {
            "record_id": record.record_id,
            "modality": record.modality,
            "section": record.section,
            "question": record.question,
            "ground_truth_answer": record.ground_truth_answer,
            "candidate_answer": record.model_answer,
        }
        for record in records
    ]
    return (
        "You are a strict reference-guided evaluator for open-ended visual and video "
        "question answering. You do not have access to the media. Judge only whether "
        "the candidate answer correctly answers the question relative to the reference. "
        "Do not infer correctness from model identity; model identity is intentionally hidden.\n\n"
        "Return ONLY valid JSON:\n"
        '{"items":[{"record_id":"...","label":"correct|partially_correct|incorrect|unjudgeable",'
        '"reason":"brief reason","error_type":"none|missing_detail|extra_incorrect_detail|'
        'contradiction|wrong_entity|wrong_count|wrong_order|too_vague|repetition|other"}]}\n\n'
        "Rules:\n"
        "- correct: semantically equivalent; harmless wording, articles, plurality, or synonyms are allowed.\n"
        "- partially_correct: contains an important correct part but misses required detail, or a multi-part answer is incomplete.\n"
        "- incorrect: wrong, contradictory, unsupported by the reference, or fails the requested answer type.\n"
        "- unjudgeable: the reference is insufficient/ambiguous or both answers could be valid without media.\n"
        "- For yes/no and counts, require the same polarity or number.\n"
        "- For lists, check required members; for sequences, check event order.\n"
        "- Penalize repeated text only when it obscures or changes the answer.\n"
        "- Keep each reason under 25 words.\n\n"
        f"Prompt version: {JUDGE_PROMPT_VERSION}\n"
        f"Items:\n{json.dumps(items, ensure_ascii=False)}"
    )


def judge_prompt_sha256() -> str:
    template = build_judge_prompt([])
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def evaluation_input_sha256(record: EvaluationRecord) -> str:
    payload = {
        "qa_id": record.qa_id,
        "modality": record.modality,
        "section": record.section,
        "question": record.question,
        "ground_truth_answer": record.ground_truth_answer,
        "model_answer": record.model_answer,
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _is_reusable_judgment(item: Any, input_sha256: str) -> bool:
    return (
        isinstance(item, dict)
        and item.get("evaluation_input_sha256") == input_sha256
        and item.get("reason") != "Judge omitted this item."
    )


def _normalize_judgment(
    raw: dict[str, Any],
    record: EvaluationRecord,
    judge_model: str,
) -> dict[str, Any]:
    label = str(raw.get("label", "unjudgeable")).strip().lower()
    if label not in VALID_LABELS:
        label = "unjudgeable"
    error_type = str(raw.get("error_type", "other")).strip().lower() or "other"
    return {
        "record_id": record.record_id,
        "qa_id": record.qa_id,
        "label": label,
        "score": LABEL_SCORES[label],
        "reason": str(raw.get("reason", "")).strip(),
        "error_type": error_type,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "judge_model": judge_model,
        "evaluation_input_sha256": evaluation_input_sha256(record),
        "model_name": record.model_name,
        "provider": record.provider,
        "source_path": record.source_path,
    }


def _judge_batch(client: Any, model_name: str, batch: list[EvaluationRecord]) -> list[dict[str, Any]]:
    response = client.models.generate_content(
        model=model_name,
        contents=[build_judge_prompt(batch)],
    )
    payload = _parse_json_object(str(getattr(response, "text", "")))
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("Judge response is missing the items list")
    requested_ids = {record.record_id for record in batch}
    returned_ids = [
        str(item.get("record_id"))
        for item in raw_items
        if isinstance(item, dict) and item.get("record_id")
    ]
    returned_counts = Counter(returned_ids)
    missing_ids = sorted(requested_ids - returned_counts.keys())
    extra_ids = sorted(returned_counts.keys() - requested_ids)
    duplicate_ids = sorted(
        record_id for record_id, count in returned_counts.items() if count > 1
    )
    diagnostics = (
        ("missing", missing_ids),
        ("unexpected", extra_ids),
        ("duplicate", duplicate_ids),
    )
    for label, record_ids in diagnostics:
        if record_ids:
            sample = ", ".join(record_ids[:5])
            suffix = "" if len(record_ids) <= 5 else ", ..."
            print(
                f"WARNING: Gemini judge response has {len(record_ids)} {label} "
                f"record_id(s): {sample}{suffix}",
                file=sys.stderr,
                flush=True,
            )
    by_id = {
        str(item.get("record_id")): item
        for item in raw_items
        if isinstance(item, dict) and item.get("record_id")
    }
    return [
        _normalize_judgment(
            by_id.get(
                record.record_id,
                {
                    "label": "unjudgeable",
                    "reason": "Judge omitted this item.",
                    "error_type": "other",
                },
            ),
            record,
            model_name,
        )
        for record in batch
    ]


def run_llm_judge(
    records: list[EvaluationRecord],
    output_path: Path | str,
    *,
    model_name: str = DEFAULT_JUDGE_MODEL,
    batch_size: int = 100,
    checkpoint_every_batches: int = 1,
    delay_seconds: float = 0.0,
    max_retries: int = 3,
    retry_delay_seconds: float = 2.0,
    service_unavailable_max_retries: int = 8,
    service_unavailable_retry_delay_seconds: float = 15.0,
    max_items: int | None = None,
    api_key_list_path: Path | str = DEFAULT_GEMINI_API_KEY_LIST_PATH,
    client_factory: Callable[[str | None], Any] = create_gemini_client,
    retry_sleep: Callable[[float], None] = time.sleep,
) -> dict[str, dict[str, Any]]:
    output_path = Path(output_path)
    existing: dict[str, Any] = {}
    legacy_judge_model = ""
    if output_path.exists():
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        legacy_judge_model = str(
            metadata.get("active_judge_model")
            or metadata.get("judge_model")
            or ""
        ).strip()
        expected = {
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "judge_prompt_sha256": judge_prompt_sha256(),
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise RuntimeError(
                    f"Existing judge cache {key}={metadata.get(key)!r}, requested {value!r}"
                )
        existing = payload.get("items", {}) if isinstance(payload, dict) else {}
        if not isinstance(existing, dict):
            existing = {}
        if legacy_judge_model:
            for item in existing.values():
                if isinstance(item, dict) and not item.get("judge_model"):
                    item["judge_model"] = legacy_judge_model

    reusable_by_input = {
        str(item.get("evaluation_input_sha256")): item
        for item in existing.values()
        if isinstance(item, dict)
        and item.get("evaluation_input_sha256")
        and item.get("reason") != "Judge omitted this item."
    }
    for record in records:
        input_sha256 = evaluation_input_sha256(record)
        cached = existing.get(record.record_id)
        if not _is_reusable_judgment(cached, input_sha256):
            cached = reusable_by_input.get(input_sha256)
        if _is_reusable_judgment(cached, input_sha256):
            existing[record.record_id] = {
                **cached,
                "record_id": record.record_id,
                "qa_id": record.qa_id,
                "model_name": record.model_name,
                "provider": record.provider,
                "source_path": record.source_path,
            }

    pending = [
        record
        for record in records
        if record.status == "answered"
        and record.model_answer.strip()
        and not _is_reusable_judgment(
            existing.get(record.record_id),
            evaluation_input_sha256(record),
        )
    ]
    if max_items is not None:
        pending = pending[: max(0, int(max_items))]

    stopped_reason: str | None = None
    keys = load_api_keys(api_key_list_path)

    def judge_model_counts() -> dict[str, int]:
        counts = Counter(
            str(item.get("judge_model") or "unknown")
            for item in existing.values()
            if isinstance(item, dict)
        )
        return dict(sorted(counts.items()))

    def cache_metadata() -> dict[str, Any]:
        model_counts = judge_model_counts()
        return {
            "cache_schema_version": JUDGE_CACHE_SCHEMA_VERSION,
            "active_judge_model": model_name,
            "judge_models": sorted(model_counts),
            "judge_model_counts": model_counts,
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "judge_prompt_sha256": judge_prompt_sha256(),
            "evaluated_items": evaluated_input_count(),
            "cached_items": len(existing),
            "total_input_records": len(records),
            "batch_size": batch_size,
            "max_retries": max_retries,
            "retry_delay_seconds": retry_delay_seconds,
            "service_unavailable_max_retries": service_unavailable_max_retries,
            "service_unavailable_retry_delay_seconds": (
                service_unavailable_retry_delay_seconds
            ),
            "key_rotation_enabled": bool(keys),
            "keys_available": len(keys),
            "exhausted_key_count": exhausted_keys,
        }

    def evaluated_input_count() -> int:
        return sum(
            _is_reusable_judgment(
                existing.get(record.record_id),
                evaluation_input_sha256(record),
            )
            for record in records
        )

    if not pending:
        exhausted_keys = 0
        metadata = cache_metadata()
        _atomic_write_json(output_path, {"items": existing, "metadata": metadata})
        return existing
    key_index = 0
    client = client_factory(keys[0] if keys else None)
    exhausted_keys = 0
    completed_batches = 0

    def checkpoint(reason: str | None = None) -> None:
        metadata = cache_metadata()
        if reason:
            metadata["stopped_reason"] = reason
        _atomic_write_json(output_path, {"items": existing, "metadata": metadata})

    try:
        for start in range(0, len(pending), max(1, batch_size)):
            batch = pending[start : start + max(1, batch_size)]
            while True:
                try:
                    judgments = call_with_retry(
                        lambda: _judge_batch(client, model_name, batch),
                        max_attempts=max(1, max_retries),
                        attempt_limit=lambda exc: (
                            max(1, service_unavailable_max_retries)
                            if is_service_unavailable_error(exc)
                            else max(1, max_retries)
                        ),
                        retry_if=lambda exc: not is_quota_error(exc),
                        wait_seconds=lambda exc, attempt: (
                            max(
                                0.0,
                                service_unavailable_retry_delay_seconds
                                if is_service_unavailable_error(exc)
                                else retry_delay_seconds,
                            )
                            * attempt
                        ),
                        label=f"Gemini judge batch ({len(batch)} item(s))",
                        sleep=retry_sleep,
                    )
                    break
                except Exception as exc:
                    if not is_quota_error(exc) or not keys or key_index + 1 >= len(keys):
                        stopped_reason = (
                            "quota_or_rate_limit" if is_quota_error(exc) else "judge_error"
                        )
                        checkpoint(stopped_reason)
                        raise
                    exhausted_keys += 1
                    key_index += 1
                    client = client_factory(keys[key_index])
            for judgment in judgments:
                existing[judgment["record_id"]] = judgment
            completed_batches += 1
            if completed_batches % max(1, checkpoint_every_batches) == 0:
                checkpoint()
            print(f"LLM judge checkpoint: {len(existing)} item(s)")
            if delay_seconds > 0 and start + batch_size < len(pending):
                time.sleep(delay_seconds)
    finally:
        checkpoint(stopped_reason)
    return existing
