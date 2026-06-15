#!/usr/bin/env python3
"""Evaluate saved VLM answers with deterministic metrics and an optional judge."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from annotation_feature.qa_quality.answer_judge import (
    DEFAULT_JUDGE_MODEL,
    run_llm_judge,
)
from annotation_feature.qa_quality.benchmark import DEFAULT_GEMINI_API_KEY_LIST_PATH
from annotation_feature.qa_quality.evaluation_report import (
    score_records,
    write_evaluation_outputs,
)
from annotation_feature.qa_quality.result_loader import load_evaluation_records


def _load_judgments(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items", {}) if isinstance(payload, dict) else {}
    return items if isinstance(items, dict) else {}


def validate_required_frame_count(records: list, expected_frame_count: int) -> None:
    expected = max(0, int(expected_frame_count))
    mismatched = [
        record
        for record in records
        if record.status == "answered" and record.frame_count != expected
    ]
    if mismatched:
        examples = ", ".join(
            f"{record.source_path}:{record.qa_id}={record.frame_count!r}"
            for record in mismatched[:5]
        )
        raise RuntimeError(
            f"{len(mismatched)} answered record(s) do not use "
            f"{expected} frames. Examples: {examples}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more result JSON files or directories containing result JSON files.",
    )
    parser.add_argument("--output", required=True, help="Evaluation output directory.")
    parser.add_argument(
        "--metrics",
        default="deterministic",
        help="Comma-separated stages: deterministic,llm_judge.",
    )
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument(
        "--require-frame-count",
        type=int,
        default=None,
        help="Reject inputs whose answered records do not use this frame count.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-batch-size", type=int, default=100)
    parser.add_argument("--judge-checkpoint-every-batches", type=int, default=1)
    parser.add_argument("--judge-delay-seconds", type=float, default=0.0)
    parser.add_argument("--judge-max-retries", type=int, default=3)
    parser.add_argument("--judge-retry-delay-seconds", type=float, default=2.0)
    parser.add_argument(
        "--judge-service-unavailable-max-retries",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--judge-service-unavailable-retry-delay-seconds",
        type=float,
        default=15.0,
    )
    parser.add_argument("--judge-max-items", type=int, default=None)
    parser.add_argument(
        "--api-key-list",
        default=str(DEFAULT_GEMINI_API_KEY_LIST_PATH),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    requested_metrics = {
        item.strip().lower() for item in args.metrics.split(",") if item.strip()
    }
    unsupported = requested_metrics - {"deterministic", "llm_judge"}
    if unsupported:
        raise ValueError(f"Unsupported metric stages: {sorted(unsupported)}")

    records, skipped = load_evaluation_records(args.input)
    if args.require_frame_count is not None:
        validate_required_frame_count(records, args.require_frame_count)
    if args.max_records is not None:
        records = records[: max(0, args.max_records)]
    if not records:
        raise RuntimeError("No VLM answer records were loaded.")
    print(f"Loaded {len(records)} answer record(s); skipped {len(skipped)} file(s).")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    judge_cache_path = output_dir / "llm_judge_cache.json"
    if "llm_judge" in requested_metrics:
        run_llm_judge(
            records,
            judge_cache_path,
            model_name=args.judge_model,
            batch_size=max(1, args.judge_batch_size),
            checkpoint_every_batches=max(1, args.judge_checkpoint_every_batches),
            delay_seconds=max(0.0, args.judge_delay_seconds),
            max_retries=max(1, args.judge_max_retries),
            retry_delay_seconds=max(0.0, args.judge_retry_delay_seconds),
            service_unavailable_max_retries=max(
                1,
                args.judge_service_unavailable_max_retries,
            ),
            service_unavailable_retry_delay_seconds=max(
                0.0,
                args.judge_service_unavailable_retry_delay_seconds,
            ),
            max_items=args.judge_max_items,
            api_key_list_path=args.api_key_list,
        )
    judgments = _load_judgments(judge_cache_path)
    rows = score_records(records, judgments=judgments)
    outputs = write_evaluation_outputs(
        output_dir,
        rows,
        skipped_inputs=skipped,
        bootstrap_samples=max(1, args.bootstrap_samples),
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")
    if judgments:
        print(f"llm_judge_cache: {judge_cache_path}")


if __name__ == "__main__":
    main()
