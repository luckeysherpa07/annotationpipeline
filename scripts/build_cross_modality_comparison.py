#!/usr/bin/env python3
"""Build reports for same-question cross-modality VLM results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_RESULTS = Path("outputs/benchmarks/vlm_cross_modality_8frame")
DEFAULT_OUTPUT = Path("outputs/evaluations/vlm_cross_modality_comparison")
METRICS = (
    "task_aware_score",
    "token_f1",
    "bleu_4",
    "rouge_l_f1",
    "meteor",
    "judge_strict_accuracy",
    "judge_soft_accuracy",
)


def discover_json(inputs: Iterable[Path]) -> list[Path]:
    discovered: set[Path] = set()
    for path in inputs:
        if path.is_file() and path.suffix.lower() == ".json":
            if "manifest" not in path.name.lower():
                discovered.add(path)
        elif path.is_dir():
            for candidate in path.rglob("*.json"):
                name = candidate.name.lower()
                if "manifest" in name or name in {"summary.json", "per_item_scores.json"}:
                    continue
                discovered.add(candidate)
        else:
            raise FileNotFoundError(f"Missing result input: {path}")
    return sorted(discovered)


def load_result_rows(inputs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in discover_json(inputs):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        metadata = payload.get("metadata", {})
        raw_rows = payload.get("results", payload.get("items", {}))
        if isinstance(raw_rows, dict):
            iterable = raw_rows.values()
        elif isinstance(raw_rows, list):
            iterable = raw_rows
        else:
            continue
        for raw in iterable:
            if not isinstance(raw, dict):
                continue
            row = dict(raw)
            row["result_path"] = path.as_posix()
            row["benchmark_type"] = row.get("benchmark_type") or metadata.get("benchmark_type", "")
            row["input_modality"] = str(
                row.get("input_modality") or row.get("modality") or "unknown"
            ).lower()
            row["source_modality"] = str(row.get("source_modality") or "unknown").lower()
            row["source_section"] = str(row.get("source_section") or row.get("section") or "unknown").lower()
            rows.append(row)
    return rows


def discover_score_csv(inputs: Iterable[Path]) -> list[Path]:
    discovered: set[Path] = set()
    for path in inputs:
        if path.is_file() and path.name == "per_item_scores.csv":
            discovered.add(path)
        elif path.is_dir():
            discovered.update(path.rglob("per_item_scores.csv"))
    return sorted(discovered)


def load_score_rows(inputs: list[Path]) -> dict[tuple[str, str, str], dict[str, str]]:
    scores: dict[tuple[str, str, str], dict[str, str]] = {}
    for path in discover_score_csv(inputs):
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (
                    str(row.get("provider", "")),
                    str(row.get("model_name", "")),
                    str(row.get("qa_id", "")),
                )
                scores[key] = row
    return scores


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def mean(values: Iterable[Any]) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return statistics.mean(numeric) if numeric else None


def enrich_with_scores(
    rows: list[dict[str, Any]],
    scores: dict[tuple[str, str, str], dict[str, str]],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        score = scores.get(
            (
                str(row.get("provider", "")),
                str(row.get("model_name", "")),
                str(row.get("qa_id", "")),
            ),
            {},
        )
        merged = dict(row)
        for field in ("task_aware_score", "token_f1", "bleu_4", "rouge_l_f1", "meteor", "judge_score"):
            value = optional_float(score.get(field))
            if value is not None:
                merged[field] = value
        if score.get("judge_label"):
            merged["judge_label"] = score["judge_label"]
        enriched.append(merged)
    return enriched


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answered = [
        row
        for row in rows
        if row.get("status") == "answered" and str(row.get("model_answer", "")).strip()
    ]
    judged = [row for row in answered if str(row.get("judge_label", "")).strip()]
    strict_values = [
        float(row.get("judge_label") == "correct")
        for row in judged
        if row.get("judge_label") != "unjudgeable"
    ]
    soft_values = [
        value
        for row in judged
        if (value := optional_float(row.get("judge_score"))) is not None
    ]
    summary = {
        "total": len(rows),
        "answered": len(answered),
        "answer_rate": len(answered) / len(rows) if rows else 0.0,
        "judge_evaluated": len(judged),
        "judge_strict_accuracy": mean(strict_values),
        "judge_soft_accuracy": mean(soft_values),
        "status_counts": dict(Counter(str(row.get("status", "unknown")) for row in rows)),
    }
    for metric in METRICS[:-2]:
        summary[metric] = mean(
            value
            for row in answered
            if (value := optional_float(row.get(metric))) is not None
        )
    return summary


def grouped_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(key, "unknown")) for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for key_values, values in sorted(grouped.items()):
        row = dict(zip(keys, key_values))
        row.update(summarize(values))
        output.append(row)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    return "" if value is None else f"{float(value):.4f}"


def write_report(path: Path, input_rows: list[dict[str, Any]], matrix_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Cross-Modality VLM Comparison",
        "",
        "Same questions are evaluated with different input modalities. "
        "`source_modality` is where the QA was generated; `input_modality` is the frames shown to the model.",
        "",
        "## By Input Modality",
        "",
        "| Provider | Model | Input modality | N | Answered | Judge strict | Judge soft | Task-aware |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in input_rows:
        lines.append(
            f"| {row['provider']} | {Path(row['model_name']).name} | {row['input_modality']} | "
            f"{row['total']} | {row['answered']} | {fmt(row.get('judge_strict_accuracy'))} | "
            f"{fmt(row.get('judge_soft_accuracy'))} | {fmt(row.get('task_aware_score'))} |"
        )

    lines.extend(
        [
            "",
            "## Source By Input Matrix",
            "",
            "| Provider | Model | Source modality | Input modality | N | Judge strict | Judge soft | Task-aware |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in matrix_rows:
        lines.append(
            f"| {row['provider']} | {Path(row['model_name']).name} | {row['source_modality']} | "
            f"{row['input_modality']} | {row['total']} | {fmt(row.get('judge_strict_accuracy'))} | "
            f"{fmt(row.get('judge_soft_accuracy'))} | {fmt(row.get('task_aware_score'))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", type=Path, default=[DEFAULT_RESULTS])
    parser.add_argument(
        "--scores",
        nargs="*",
        type=Path,
        default=[],
        help="Evaluation directories or per_item_scores.csv files to merge judge/metric columns.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = load_result_rows(args.results)
    if not rows:
        raise RuntimeError("No cross-modality result rows were loaded.")
    scores = load_score_rows(args.scores) if args.scores else {}
    rows = enrich_with_scores(rows, scores)

    input_rows = grouped_rows(rows, ("provider", "model_name", "input_modality"))
    matrix_rows = grouped_rows(rows, ("provider", "model_name", "source_modality", "input_modality"))
    section_rows = grouped_rows(
        rows,
        ("provider", "model_name", "source_modality", "source_section", "input_modality"),
    )

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "input_modality_summary.csv", input_rows)
    write_csv(args.output / "source_input_modality_matrix.csv", matrix_rows)
    write_csv(args.output / "source_section_input_modality_matrix.csv", section_rows)
    write_report(args.output / "cross_modality_report.md", input_rows, matrix_rows)
    print(f"report: {args.output / 'cross_modality_report.md'}")


if __name__ == "__main__":
    main()
