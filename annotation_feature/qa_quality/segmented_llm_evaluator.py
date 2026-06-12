"""LLM-assisted quality evaluation for cleaned segmented QA items."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any

DEFAULT_INPUT_PATH = Path("outputs/segmented_qa_cleaned_items.json")
DEFAULT_OUTPUT_JSON = Path("outputs/segmented_qa_llm_eval_results.json")
DEFAULT_OUTPUT_CSV = Path("outputs/segmented_qa_llm_eval_items.csv")
LLM_EVAL_MODEL_NAME = "gemini-3.1-flash-lite"
DEFAULT_SELECTION_STRATEGY = "balanced_modality"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


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


def _load_cleaned_items(input_path: Path) -> list[dict[str, Any]]:
    with open(input_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise ValueError(f"Expected {input_path} to contain an object with an items list")
    return [item for item in items if isinstance(item, dict) and item.get("qa_id")]


def _load_existing_results(output_json: Path) -> dict[str, Any]:
    if not output_json.exists():
        return {"items": {}, "metadata": {}}
    try:
        with open(output_json, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARNING: Could not load existing segmented LLM results from {output_json}: {exc}")
        return {"items": {}, "metadata": {}}
    if not isinstance(payload, dict):
        return {"items": {}, "metadata": {}}
    items = payload.get("items")
    if isinstance(items, list):
        items = {
            str(item.get("qa_id")): item
            for item in items
            if isinstance(item, dict) and item.get("qa_id")
        }
    return {
        "items": items if isinstance(items, dict) else {},
        "metadata": payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {},
    }


def _select_pending_items(
    items: list[dict[str, Any]],
    max_items: int | None,
    selection_strategy: str,
) -> list[dict[str, Any]]:
    if selection_strategy == "sequential":
        return list(items) if max_items is None else items[:max(0, int(max_items))]
    if selection_strategy != "balanced_modality":
        raise ValueError(f"Unsupported selection strategy: {selection_strategy}")
    if max_items is None or max_items >= len(items):
        return list(items)
    if max_items <= 0:
        return []

    grouped: dict[str, list[dict[str, Any]]] = {}
    modality_order: list[str] = []
    for item in items:
        modality = str(item.get("modality") or "unknown")
        if modality not in grouped:
            grouped[modality] = []
            modality_order.append(modality)
        grouped[modality].append(item)

    selected: list[dict[str, Any]] = []
    offsets = {modality: 0 for modality in modality_order}
    while len(selected) < max_items:
        made_progress = False
        for modality in modality_order:
            offset = offsets[modality]
            bucket = grouped[modality]
            if offset >= len(bucket):
                continue
            selected.append(bucket[offset])
            offsets[modality] += 1
            made_progress = True
            if len(selected) >= max_items:
                break
        if not made_progress:
            break
    return selected


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(text or "").strip(), flags=re.I)
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if not match:
        raise ValueError("No JSON object found in segmented QA evaluation response")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Segmented QA evaluation response must be an object")
    return parsed


def _build_prompt(batch: list[dict[str, Any]]) -> str:
    compact = [
        {
            "qa_id": item.get("qa_id"),
            "segment_id": item.get("segment_id"),
            "side": item.get("side"),
            "task_label": item.get("task_label"),
            "start_seconds": item.get("start_seconds"),
            "end_seconds": item.get("end_seconds"),
            "modality": item.get("modality"),
            "section": item.get("section"),
            "caption": item.get("caption"),
            "question": item.get("question"),
            "answer": item.get("answer"),
        }
        for item in batch
    ]
    return (
        "You are evaluating generated QA items for semantic video segments. Use only the supplied segment "
        "metadata, caption, question, answer, modality, and section. Do not assume access to source media.\n\n"
        "Return ONLY valid JSON:\n"
        '{"items":[{"qa_id":"...","status":"pass|review|reject",'
        '"answerable_from_caption":true,"answer_matches_question":true,'
        '"caption_supports_answer":true,"modality_appropriate":true,'
        '"single_question":true,"segment_consistent":true,'
        '"hallucination_risk":"low|medium|high","reason":"short explanation"}]}\n\n'
        "Rules:\n"
        "- pass requires a single, caption-supported QA appropriate to the named modality and semantic segment.\n"
        "- segment_consistent means the QA is plausibly scoped to the task label and this segment rather than "
        "another recording side or unrelated task.\n"
        "- RGB may use visible appearance/color/text; depth should use geometry/layout/distance; IR should use "
        "infrared-visible structure; event should use motion/change; audio should use audible evidence or "
        "explicitly caption-grounded audio-visual reasoning.\n"
        "- review minor ambiguity; reject contradiction, unsupported answers, modality mismatch, unrelated "
        "segment content, or high hallucination risk. Keep reason under 30 words.\n\n"
        f"Items:\n{json.dumps(compact, ensure_ascii=False)}"
    )


def _normalized_result(raw: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    status = str(raw.get("status", "review")).strip().lower()
    if status not in {"pass", "review", "reject"}:
        status = "review"
    risk = str(raw.get("hallucination_risk", "medium")).strip().lower()
    if risk not in {"low", "medium", "high"}:
        risk = "medium"
    preserved_fields = (
        "qa_id",
        "segment_id",
        "source_prefix",
        "side",
        "task_label",
        "start_seconds",
        "end_seconds",
        "start_timestamp",
        "end_timestamp",
        "segment_confidence",
        "source_files",
        "source_media",
        "modality",
        "section",
        "question",
        "answer",
        "caption",
        "timestamp",
        "confidence",
    )
    return {
        **{field: source.get(field) for field in preserved_fields},
        "status": status,
        "answerable_from_caption": bool(raw.get("answerable_from_caption")),
        "answer_matches_question": bool(raw.get("answer_matches_question")),
        "caption_supports_answer": bool(raw.get("caption_supports_answer")),
        "modality_appropriate": bool(raw.get("modality_appropriate")),
        "single_question": bool(raw.get("single_question")),
        "segment_consistent": bool(raw.get("segment_consistent")),
        "hallucination_risk": risk,
        "reason": str(raw.get("reason", "")).strip(),
        "source_transform": source.get("transform"),
        "source_severity": source.get("source_severity"),
        "source_flags": source.get("source_flags", []),
    }


def _evaluate_batch(client: Any, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    response = client.models.generate_content(
        model=LLM_EVAL_MODEL_NAME,
        contents=[_build_prompt(batch)],
    )
    parsed = _parse_json_object(str(getattr(response, "text", "")))
    raw_items = parsed.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("LLM response missing items list")
    by_id = {
        str(item.get("qa_id")): item
        for item in raw_items
        if isinstance(item, dict) and item.get("qa_id")
    }
    results = []
    for source in batch:
        raw = by_id.get(str(source.get("qa_id")), {})
        if not raw:
            raw = {
                "status": "review",
                "single_question": True,
                "hallucination_risk": "medium",
                "reason": "Missing item in LLM response.",
            }
        results.append(_normalized_result(raw, source))
    return results


def run_segmented_qa_llm_evaluation(
    input_path: Path | str = DEFAULT_INPUT_PATH,
    output_json: Path | str = DEFAULT_OUTPUT_JSON,
    output_csv: Path | str = DEFAULT_OUTPUT_CSV,
    batch_size: int = 50,
    max_items: int | None = None,
    delay_between_batches: int = 5,
    selection_strategy: str = DEFAULT_SELECTION_STRATEGY,
    client: Any | None = None,
) -> dict[str, Path]:
    """Run resumable LLM quality evaluation on cleaned segmented QA."""
    input_path = Path(input_path)
    output_json = Path(output_json)
    output_csv = Path(output_csv)
    batch_size = max(1, int(batch_size))
    cleaned_items = _load_cleaned_items(input_path)
    existing = _load_existing_results(output_json)
    results_by_id = dict(existing["items"])
    pending_all = [item for item in cleaned_items if str(item.get("qa_id")) not in results_by_id]
    pending = _select_pending_items(pending_all, max_items, selection_strategy)

    print(
        f"Segmented LLM QA evaluation resume scan: {len(results_by_id)} complete skipped, "
        f"{len(pending)} pending selected, {len(cleaned_items)} cleaned total, "
        f"selection_strategy={selection_strategy}."
    )
    if not pending:
        _write_csv(output_csv, list(results_by_id.values()))
        return {"llm_eval_json": output_json, "llm_eval_csv": output_csv}

    if client is None:
        from annotation_feature.pipeline.client import create_gemini_client

        llm_client = create_gemini_client()
    else:
        llm_client = client
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        print(f"Evaluating segmented LLM batch {start // batch_size + 1}: {len(batch)} item(s)")
        try:
            evaluated = _evaluate_batch(llm_client, batch)
        except Exception as exc:
            print(f"ERROR: Segmented LLM evaluation batch failed: {exc}")
            if delay_between_batches > 0 and start + batch_size < len(pending):
                time.sleep(delay_between_batches)
            continue
        for item in evaluated:
            results_by_id[str(item["qa_id"])] = item
        metadata = {
            "model": LLM_EVAL_MODEL_NAME,
            "input_path": input_path.as_posix(),
            "total_cleaned_items": len(cleaned_items),
            "evaluated_items": len(results_by_id),
            "pending_items": max(0, len(cleaned_items) - len(results_by_id)),
            "selection_strategy": selection_strategy,
        }
        _write_json(output_json, {"items": results_by_id, "metadata": metadata})
        _write_csv(output_csv, list(results_by_id.values()))
        print(f"Checkpoint saved: {len(results_by_id)} segmented evaluated item(s)")
        if delay_between_batches > 0 and start + batch_size < len(pending):
            time.sleep(delay_between_batches)
    return {"llm_eval_json": output_json, "llm_eval_csv": output_csv}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--delay-between-batches", type=int, default=5)
    args = parser.parse_args()
    outputs = run_segmented_qa_llm_evaluation(
        input_path=args.input,
        output_json=args.output_json,
        output_csv=args.output_csv,
        batch_size=args.batch_size,
        max_items=args.max_items,
        delay_between_batches=args.delay_between_batches,
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
