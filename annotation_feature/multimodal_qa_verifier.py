"""Verify implicit cross-modal QA candidates with Gemini."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from annotation_feature.pipeline.client import create_gemini_client


VERIFIER_MODEL_NAME = "gemini-2.0-flash"
DEFAULT_DELAY_SECONDS = 8

ANSWERABILITY_LABELS = {"unanswerable", "partial", "answerable"}
DEPENDENCY_LABELS = {"none", "weak", "strong"}
HALLUCINATION_LABELS = {"pass", "fail"}
QUALITY_STATUS = {"accepted", "rejected", "needs_revision"}


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object in {path}")
    return data


def _save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def _is_quota_or_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(token in text for token in ("quota", "rate limit", "rate_limit", "429", "resource_exhausted"))


async def _call_verifier_gemini_with_retry(
    client,
    contents: list,
    max_retries: int = 3,
    model_name: str = VERIFIER_MODEL_NAME,
) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=model_name,
                contents=contents,
            )
            return response.text
        except Exception as exc:
            if attempt == max_retries:
                raise
            wait_seconds = 30 * attempt if _is_quota_or_rate_limit_error(exc) else 2 * attempt
            print(
                f"    QA verifier Gemini call failed on attempt {attempt}/{max_retries}; "
                f"retrying in {wait_seconds}s: {exc}"
            )
            await asyncio.sleep(wait_seconds)
    raise RuntimeError("QA verifier Gemini call failed")


def _parse_json_object_response(text: str) -> dict[str, Any]:
    if not text:
        raise ValueError("Empty Gemini response")
    cleaned_text = text.strip()
    cleaned_text = re.sub(r"^```(?:json)?\s*", "", cleaned_text, flags=re.I)
    cleaned_text = re.sub(r"\s*```$", "", cleaned_text, flags=re.I)
    match = re.search(r"\{.*\}", cleaned_text, flags=re.S)
    if not match:
        raise ValueError("No JSON object found in Gemini response")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Gemini response must be a JSON object")
    return parsed


def _pair_label(item: dict[str, Any]) -> str:
    modalities = item.get("modalities")
    if isinstance(modalities, list) and modalities:
        return "+".join(str(modality) for modality in modalities)
    return "unknown"


def _direction_label(item: dict[str, Any]) -> str:
    return f"{item.get('context_modality') or 'unknown'}->{item.get('decisive_modality') or 'unknown'}"


def _item_matches_filters(
    item: dict[str, Any],
    side: str | None,
    pair: str | None,
    challenge_type: str | None,
) -> bool:
    if side and str(item.get("side") or "").lower() != side.lower():
        return False
    if pair and _pair_label(item).lower() != pair.lower():
        return False
    if challenge_type and str(item.get("challenge_type") or "") != challenge_type:
        return False
    return True


def _is_eligible_for_verification(
    item: dict[str, Any],
    resume: bool,
    side: str | None,
    pair: str | None,
    challenge_type: str | None,
) -> bool:
    if not _item_matches_filters(item, side=side, pair=pair, challenge_type=challenge_type):
        return False
    if item.get("quality_control", {}).get("validation_status") != "passed":
        return False
    verification = item.get("answerability_verification", {})
    if resume and isinstance(verification, dict) and verification.get("status") == "verified":
        return False
    return True


def _build_verifier_payload(item: dict[str, Any]) -> dict[str, Any]:
    context = item.get("context_modality")
    decisive = item.get("decisive_modality")
    evidence_by_modality = item.get("evidence_by_modality", {})
    return {
        "qa_id": item.get("qa_id"),
        "segment_id": item.get("segment_id"),
        "side": item.get("side"),
        "task_label": item.get("task_label"),
        "question": item.get("question"),
        "answer": item.get("answer"),
        "modalities": item.get("modalities"),
        "context_modality": context,
        "decisive_modality": decisive,
        "challenge_type": item.get("challenge_type"),
        "context_evidence": evidence_by_modality.get(context, {}) if isinstance(evidence_by_modality, dict) else {},
        "decisive_evidence": evidence_by_modality.get(decisive, {}) if isinstance(evidence_by_modality, dict) else {},
        "why_multimodal_claim": item.get("why_multimodal"),
    }


def _build_verifier_prompt(item: dict[str, Any], model_name: str) -> str:
    payload = _build_verifier_payload(item)
    return "\n".join(
        [
            "You are verifying whether an implicit cross-modal QA is valid.",
            "Use ONLY the provided evidence. Do not use external knowledge or unstated commonsense guesses.",
            "A condition is answerable only if the evidence explicitly supports the answer.",
            "Evaluate whether the context modality grounds the queried object/event/scene and whether the decisive modality supports the answer.",
            "",
            "Definitions:",
            "- context_only: answerability using only context_evidence.",
            "- decisive_only: answerability using only decisive_evidence.",
            "- combined: answerability using both evidence sources together.",
            "- cross_modal_dependency: whether the QA truly requires binding context evidence to decisive evidence.",
            "",
            "Return ONLY valid JSON with this exact structure:",
            "{",
            '  "answerability_verification": {',
            '    "status": "verified",',
            '    "context_only": {"label": "unanswerable|partial|answerable", "rationale": "..."},',
            '    "decisive_only": {"label": "unanswerable|partial|answerable", "rationale": "..."},',
            '    "combined": {"label": "unanswerable|partial|answerable", "rationale": "..."},',
            '    "cross_modal_dependency": {"label": "none|weak|strong", "rationale": "..."},',
            '    "hallucination_check": {"label": "pass|fail", "rationale": "..."},',
            f'    "verifier": "{model_name}"',
            "  },",
            '  "selection": {',
            '    "candidate_status": "reviewed",',
            '    "quality_status": "accepted|rejected|needs_revision",',
            '    "keep": true,',
            '    "review_notes": "..."',
            "  }",
            "}",
            "",
            "QA candidate:",
            json.dumps(payload, indent=2, ensure_ascii=False),
        ]
    )


def _label(value: Any, allowed: set[str], default: str) -> str:
    label = str(value or "").strip().lower()
    return label if label in allowed else default


def _normalize_label_record(value: Any, allowed: set[str], default: str) -> dict[str, str]:
    if isinstance(value, dict):
        return {
            "label": _label(value.get("label"), allowed, default),
            "rationale": str(value.get("rationale") or "").strip(),
        }
    return {"label": default, "rationale": ""}


def _rule_keep(verification: dict[str, Any]) -> bool:
    context_only = verification.get("context_only", {}).get("label")
    decisive_only = verification.get("decisive_only", {}).get("label")
    combined = verification.get("combined", {}).get("label")
    dependency = verification.get("cross_modal_dependency", {}).get("label")
    hallucination = verification.get("hallucination_check", {}).get("label")
    return (
        context_only != "answerable"
        and decisive_only != "answerable"
        and combined == "answerable"
        and dependency == "strong"
        and hallucination == "pass"
    )


def _normalize_verifier_response(parsed: dict[str, Any], model_name: str) -> dict[str, Any]:
    raw_verification = parsed.get("answerability_verification", {})
    if not isinstance(raw_verification, dict):
        raw_verification = {}
    verification = {
        "status": "verified",
        "context_only": _normalize_label_record(raw_verification.get("context_only"), ANSWERABILITY_LABELS, "unanswerable"),
        "decisive_only": _normalize_label_record(raw_verification.get("decisive_only"), ANSWERABILITY_LABELS, "unanswerable"),
        "combined": _normalize_label_record(raw_verification.get("combined"), ANSWERABILITY_LABELS, "unanswerable"),
        "cross_modal_dependency": _normalize_label_record(
            raw_verification.get("cross_modal_dependency"),
            DEPENDENCY_LABELS,
            "none",
        ),
        "hallucination_check": _normalize_label_record(
            raw_verification.get("hallucination_check"),
            HALLUCINATION_LABELS,
            "fail",
        ),
        "verifier": str(raw_verification.get("verifier") or model_name),
    }

    raw_selection = parsed.get("selection", {})
    if not isinstance(raw_selection, dict):
        raw_selection = {}
    raw_keep = raw_selection.get("keep")
    rule_keep = _rule_keep(verification)
    selection = {
        "candidate_status": "reviewed",
        "quality_status": _label(
            raw_selection.get("quality_status"),
            QUALITY_STATUS,
            "accepted" if rule_keep else "rejected",
        ),
        "keep": rule_keep,
        "verifier_keep_raw": raw_keep if isinstance(raw_keep, bool) else None,
        "review_notes": str(raw_selection.get("review_notes") or "").strip(),
    }
    if not rule_keep and selection["quality_status"] == "accepted":
        selection["quality_status"] = "rejected"
    return {
        "answerability_verification": verification,
        "selection": selection,
    }


async def _verify_item(client, item: dict[str, Any], model_name: str) -> dict[str, Any]:
    response_text = await _call_verifier_gemini_with_retry(
        client,
        [_build_verifier_prompt(item, model_name)],
        model_name=model_name,
    )
    parsed = _parse_json_object_response(response_text)
    return _normalize_verifier_response(parsed, model_name)


def _mark_verification_failed(item: dict[str, Any], error: Exception, model_name: str) -> None:
    item["answerability_verification"] = {
        "status": "verification_failed",
        "error": str(error),
        "verifier": model_name,
    }
    item["selection"] = {
        "candidate_status": "candidate",
        "quality_status": "unreviewed",
        "keep": None,
        "review_notes": f"Verification failed: {error}",
    }


def _counter_to_dict(counter: Counter) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _build_distribution(qa_items: list[dict[str, Any]], accepted_only: bool = False) -> dict[str, dict[str, int]]:
    by_pair: Counter = Counter()
    by_direction: Counter = Counter()
    by_challenge_type: Counter = Counter()
    by_side: Counter = Counter()
    by_status: Counter = Counter()
    by_keep: Counter = Counter()
    for item in qa_items:
        if accepted_only and item.get("selection", {}).get("keep") is not True:
            continue
        by_pair[_pair_label(item)] += 1
        by_direction[_direction_label(item)] += 1
        by_challenge_type[str(item.get("challenge_type") or "unknown")] += 1
        by_side[str(item.get("side") or "unknown")] += 1
        by_status[str(item.get("selection", {}).get("quality_status") or "unknown")] += 1
        by_keep[str(item.get("selection", {}).get("keep"))] += 1
    return {
        "by_pair": _counter_to_dict(by_pair),
        "by_direction": _counter_to_dict(by_direction),
        "by_challenge_type": _counter_to_dict(by_challenge_type),
        "by_side": _counter_to_dict(by_side),
        "by_quality_status": _counter_to_dict(by_status),
        "by_keep": _counter_to_dict(by_keep),
    }


def _verification_summary(qa_items: list[dict[str, Any]], eligible_count: int, verified_count: int, failed_count: int) -> dict[str, int]:
    accepted = sum(1 for item in qa_items if item.get("selection", {}).get("keep") is True)
    rejected = sum(1 for item in qa_items if item.get("selection", {}).get("quality_status") == "rejected")
    needs_revision = sum(1 for item in qa_items if item.get("selection", {}).get("quality_status") == "needs_revision")
    return {
        "total_items": len(qa_items),
        "eligible_items": eligible_count,
        "verified_items": verified_count,
        "accepted_items": accepted,
        "rejected_items": rejected,
        "needs_revision_items": needs_revision,
        "verification_failed_items": failed_count,
    }


async def _run_verifier_async(
    input_path: Path,
    output_path: Path,
    limit: int | None,
    resume: bool,
    side: str | None,
    pair: str | None,
    challenge_type: str | None,
    delay_between_calls: int,
    model_name: str,
) -> Path:
    data = _load_json(input_path)
    qa_items = data.get("qa_items", [])
    if not isinstance(qa_items, list):
        raise ValueError("Input JSON must contain qa_items list")

    output_data = copy.deepcopy(data)
    output_items = output_data["qa_items"]
    eligible_indices = [
        index
        for index, item in enumerate(output_items)
        if isinstance(item, dict)
        and _is_eligible_for_verification(
            item,
            resume=resume,
            side=side,
            pair=pair,
            challenge_type=challenge_type,
        )
    ]
    if limit is not None:
        eligible_indices = eligible_indices[:limit]

    client = create_gemini_client()
    verified_count = 0
    failed_count = 0
    for run_index, item_index in enumerate(eligible_indices, start=1):
        item = output_items[item_index]
        print(f"Verifying [{run_index}/{len(eligible_indices)}]: {item.get('qa_id')}")
        try:
            result = await _verify_item(client, item, model_name)
            item["answerability_verification"] = result["answerability_verification"]
            item["selection"] = result["selection"]
            verified_count += 1
        except Exception as exc:
            print(f"WARNING: Verification failed for {item.get('qa_id')}: {exc}")
            _mark_verification_failed(item, exc, model_name)
            failed_count += 1
        if run_index < len(eligible_indices) and delay_between_calls > 0:
            await asyncio.sleep(delay_between_calls)

    output_data.setdefault("metadata", {})["verification"] = {
        "verifier": model_name,
        "source_file": str(input_path),
        "resume": resume,
        "filters": {
            "limit": limit,
            "side": side,
            "pair": pair,
            "challenge_type": challenge_type,
        },
        "summary": _verification_summary(output_items, len(eligible_indices), verified_count, failed_count),
        "distribution_all_items": _build_distribution(output_items, accepted_only=False),
        "distribution_accepted_items": _build_distribution(output_items, accepted_only=True),
    }
    _save_json(output_data, output_path)
    return output_path


def run_multimodal_qa_verifier(
    input_path: Path | str = "outputs/implicit_multimodal_qa_candidates_template.json",
    output_path: Path | str = "outputs/implicit_multimodal_qa_verified.json",
    limit: int | None = None,
    resume: bool = True,
    side: str | None = None,
    pair: str | None = None,
    challenge_type: str | None = None,
    delay_between_calls: int = DEFAULT_DELAY_SECONDS,
    model_name: str = VERIFIER_MODEL_NAME,
) -> Path:
    """Verify implicit cross-modal QA candidates using Gemini."""
    resolved_model_name = model_name.strip()
    if not resolved_model_name:
        raise ValueError("model_name must not be empty")
    return asyncio.run(
        _run_verifier_async(
            input_path=Path(input_path),
            output_path=Path(output_path),
            limit=limit,
            resume=resume,
            side=side,
            pair=pair,
            challenge_type=challenge_type,
            delay_between_calls=delay_between_calls,
            model_name=resolved_model_name,
        )
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="outputs/implicit_multimodal_qa_candidates_template.json")
    parser.add_argument("--output", default="outputs/implicit_multimodal_qa_verified.json")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of eligible QA items to verify.")
    parser.add_argument("--no-resume", action="store_true", help="Re-verify items even if already verified.")
    parser.add_argument("--side", choices=("day", "night"), default=None)
    parser.add_argument("--pair", default=None, help="Filter by pair label such as rgb+audio.")
    parser.add_argument("--challenge-type", default=None)
    parser.add_argument("--delay-between-calls", type=int, default=DEFAULT_DELAY_SECONDS)
    parser.add_argument("--model-name", default=VERIFIER_MODEL_NAME, help="Gemini model name used for verification.")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    output_path = run_multimodal_qa_verifier(
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
        resume=not args.no_resume,
        side=args.side,
        pair=args.pair,
        challenge_type=args.challenge_type,
        delay_between_calls=args.delay_between_calls,
        model_name=args.model_name,
    )
    print(f"Wrote verified implicit multimodal QA to {output_path}")


if __name__ == "__main__":
    main()
