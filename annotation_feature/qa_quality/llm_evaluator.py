"""LLM-assisted caption-grounded evaluation for cleaned aligned QA items."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any

from annotation_feature.pipeline.client import create_gemini_client


DEFAULT_INPUT_PATH = Path("outputs/aligned_qa_cleaned_items.json")
DEFAULT_OUTPUT_JSON = Path("outputs/aligned_qa_llm_eval_results.json")
DEFAULT_OUTPUT_CSV = Path("outputs/aligned_qa_llm_eval_items.csv")
LLM_EVAL_MODEL_NAME = "gemini-3.1-flash-lite"
DEFAULT_SELECTION_STRATEGY = "balanced_modality"


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_json_response(text: str) -> dict[str, Any]:
    if not text:
        raise ValueError("Empty response text")
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.I)
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if not match:
        raise ValueError("No JSON object found in response")
    return json.loads(match.group(0))


def _load_cleaned_items(input_path: Path) -> list[dict[str, Any]]:
    payload = _load_json(input_path)
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise ValueError(f"Expected {input_path} to contain an object with an items list")
    return [item for item in items if isinstance(item, dict) and item.get("qa_id")]


def _load_existing_results(output_json: Path) -> dict[str, Any]:
    if not output_json.exists():
        return {"items": {}, "metadata": {}}
    try:
        payload = _load_json(output_json)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARNING: Could not load existing LLM eval results from {output_json}: {exc}")
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
    if not isinstance(items, dict):
        items = {}
    return {"items": items, "metadata": payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}}


def _build_prompt(batch: list[dict[str, Any]]) -> str:
    compact_items = []
    for item in batch:
        compact_items.append(
            {
                "qa_id": item.get("qa_id"),
                "modality": item.get("modality"),
                "section": item.get("section"),
                "caption": item.get("caption"),
                "question": item.get("question"),
                "answer": item.get("answer"),
            }
        )

    return (
        "You are evaluating generated video QA items using only the provided caption, question, answer, "
        "modality, and section. Do not assume access to the original video frames.\n"
        "For each item, judge caption-grounded semantic quality.\n\n"
        "Return ONLY valid JSON with this exact shape:\n"
        "{\n"
        '  "items": [\n'
        "    {\n"
        '      "qa_id": "...",\n'
        '      "status": "pass|review|reject",\n'
        '      "answerable_from_caption": true,\n'
        '      "answer_matches_question": true,\n'
        '      "caption_supports_answer": true,\n'
        '      "modality_appropriate": true,\n'
        '      "single_question": true,\n'
        '      "hallucination_risk": "low|medium|high",\n'
        '      "reason": "short explanation"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Decision rules:\n"
        "- pass: answer is supported by the caption, answers the question, is modality-appropriate, and is a single QA.\n"
        "- review: minor ambiguity, weak caption support, generic answer, or possible modality mismatch.\n"
        "- reject: answer contradicts the caption, cannot be inferred from caption, is not a real QA, or has high hallucination risk.\n"
        "- Modality appropriateness: RGB can use visible appearance/color/text; depth should focus on geometry/layout/distance; "
        "IR should focus on infrared/low-light visible structure; event should focus on motion/change cues; "
        "audio should focus on sound events, source identity, temporal sound structure, speech, or explicitly caption-grounded cross-modal audio-visual reasoning.\n"
        "- Keep reason under 30 words.\n\n"
        "Items to evaluate:\n"
        f"{json.dumps(compact_items, ensure_ascii=False)}"
    )


def _normalize_eval_item(raw: dict[str, Any], source_item: dict[str, Any]) -> dict[str, Any]:
    status = str(raw.get("status", "review")).strip().lower()
    if status not in {"pass", "review", "reject"}:
        status = "review"
    hallucination_risk = str(raw.get("hallucination_risk", "medium")).strip().lower()
    if hallucination_risk not in {"low", "medium", "high"}:
        hallucination_risk = "medium"

    return {
        "qa_id": source_item.get("qa_id"),
        "modality": source_item.get("modality"),
        "pair_key": source_item.get("pair_key"),
        "section": source_item.get("section"),
        "question": source_item.get("question"),
        "answer": source_item.get("answer"),
        "caption": source_item.get("caption"),
        "status": status,
        "answerable_from_caption": bool(raw.get("answerable_from_caption")),
        "answer_matches_question": bool(raw.get("answer_matches_question")),
        "caption_supports_answer": bool(raw.get("caption_supports_answer")),
        "modality_appropriate": bool(raw.get("modality_appropriate")),
        "single_question": bool(raw.get("single_question")),
        "hallucination_risk": hallucination_risk,
        "reason": str(raw.get("reason", "")).strip(),
        "source_transform": source_item.get("transform"),
        "source_severity": source_item.get("source_severity"),
        "source_flags": source_item.get("source_flags", []),
    }


def _evaluate_batch(client, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prompt = _build_prompt(batch)
    response = client.models.generate_content(
        model=LLM_EVAL_MODEL_NAME,
        contents=[prompt],
    )
    parsed = _parse_json_response(response.text)
    raw_items = parsed.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("LLM response missing items list")

    raw_by_id = {
        str(item.get("qa_id")): item
        for item in raw_items
        if isinstance(item, dict) and item.get("qa_id")
    }
    normalized = []
    for source_item in batch:
        raw = raw_by_id.get(str(source_item.get("qa_id")))
        if not isinstance(raw, dict):
            raw = {
                "status": "review",
                "answerable_from_caption": False,
                "answer_matches_question": False,
                "caption_supports_answer": False,
                "modality_appropriate": False,
                "single_question": True,
                "hallucination_risk": "medium",
                "reason": "Missing item in LLM response.",
            }
        normalized.append(_normalize_eval_item(raw, source_item))
    return normalized


def _select_balanced_by_modality(items: list[dict[str, Any]], max_items: int | None) -> list[dict[str, Any]]:
    """Select pending items with a roughly even modality distribution."""
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


def _select_pending_items(
    items: list[dict[str, Any]],
    max_items: int | None,
    selection_strategy: str,
) -> list[dict[str, Any]]:
    if selection_strategy == "sequential":
        if max_items is None:
            return list(items)
        return items[: max(0, int(max_items))]
    if selection_strategy == "balanced_modality":
        return _select_balanced_by_modality(items, max_items)
    raise ValueError(f"Unsupported selection strategy: {selection_strategy}")


def run_aligned_qa_llm_evaluation(
    input_path: Path | str = DEFAULT_INPUT_PATH,
    output_json: Path | str = DEFAULT_OUTPUT_JSON,
    output_csv: Path | str = DEFAULT_OUTPUT_CSV,
    batch_size: int = 50,
    max_items: int | None = None,
    delay_between_batches: int = 5,
    selection_strategy: str = DEFAULT_SELECTION_STRATEGY,
) -> dict[str, Path]:
    """Run caption-grounded LLM evaluation on cleaned aligned QA items."""
    input_path = Path(input_path)
    output_json = Path(output_json)
    output_csv = Path(output_csv)
    batch_size = max(1, int(batch_size))

    cleaned_items = _load_cleaned_items(input_path)
    existing = _load_existing_results(output_json)
    results_by_id: dict[str, dict[str, Any]] = dict(existing["items"])

    pending_all = [item for item in cleaned_items if str(item.get("qa_id")) not in results_by_id]
    pending = _select_pending_items(pending_all, max_items, selection_strategy)

    print(
        f"LLM QA evaluation resume scan: {len(results_by_id)} complete skipped, "
        f"{len(pending)} pending selected, {len(cleaned_items)} cleaned total, "
        f"selection_strategy={selection_strategy}."
    )
    if not pending:
        _write_csv(output_csv, list(results_by_id.values()))
        return {"llm_eval_json": output_json, "llm_eval_csv": output_csv}

    client = create_gemini_client()
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        print(f"Evaluating LLM batch {start // batch_size + 1}: {len(batch)} item(s)")
        try:
            evaluated = _evaluate_batch(client, batch)
        except Exception as exc:
            print(f"ERROR: LLM evaluation batch failed: {exc}")
            print("Skipping checkpoint for this batch; it will remain pending for resume.")
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
        print(f"Checkpoint saved: {len(results_by_id)} evaluated item(s)")

        if delay_between_batches > 0 and start + batch_size < len(pending):
            time.sleep(delay_between_batches)

    return {"llm_eval_json": output_json, "llm_eval_csv": output_csv}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--delay-between-batches", type=int, default=5)
    parser.add_argument(
        "--selection-strategy",
        choices=("balanced_modality", "sequential"),
        default=DEFAULT_SELECTION_STRATEGY,
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    outputs = run_aligned_qa_llm_evaluation(
        input_path=args.input,
        output_json=args.output_json,
        output_csv=args.output_csv,
        batch_size=args.batch_size,
        max_items=args.max_items,
        delay_between_batches=args.delay_between_batches,
        selection_strategy=args.selection_strategy,
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
