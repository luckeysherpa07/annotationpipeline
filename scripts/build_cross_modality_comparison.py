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
PAIR_MODALITIES = ("rgb", "ir")


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


def semantic_section(section: Any) -> str:
    """Merge source-specific section prefixes into one QA-type label."""
    normalized = str(section or "unknown").lower()
    for prefix in ("event_", "depth_"):
        if normalized.startswith(prefix):
            return normalized[len(prefix) :]
    return normalized


def source_qa_key(row: dict[str, Any]) -> str:
    source_qa_id = str(row.get("source_qa_id", "")).strip()
    if source_qa_id:
        return source_qa_id
    return str(row.get("qa_id", "")).split("__input_", maxsplit=1)[0]


def build_rgb_ir_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join RGB and IR scores for each model's identical source QA."""
    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        modality = str(row.get("input_modality", "")).lower()
        if modality not in PAIR_MODALITIES:
            continue
        key = (
            str(row.get("provider", "")),
            str(row.get("model_name", "")),
            source_qa_key(row),
        )
        grouped[key][modality] = row

    pairs: list[dict[str, Any]] = []
    for (provider, model_name, qa_id), modalities in grouped.items():
        rgb = modalities.get("rgb")
        ir = modalities.get("ir")
        if rgb is None or ir is None:
            continue
        pairs.append(
            {
                "provider": provider,
                "model_name": model_name,
                "source_qa_id": qa_id,
                "source_modality": str(rgb.get("source_modality", "unknown")).lower(),
                "source_section": str(rgb.get("source_section", "unknown")).lower(),
                "semantic_section": semantic_section(rgb.get("source_section")),
                "rgb_judge_label": str(rgb.get("judge_label", "")).lower(),
                "ir_judge_label": str(ir.get("judge_label", "")).lower(),
                "rgb_judge_score": optional_float(rgb.get("judge_score")),
                "ir_judge_score": optional_float(ir.get("judge_score")),
            }
        )
    return pairs


def summarize_rgb_ir_pairs(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    judged = [
        pair
        for pair in pairs
        if pair["rgb_judge_label"] and pair["ir_judge_label"]
        and pair["rgb_judge_label"] != "unjudgeable"
        and pair["ir_judge_label"] != "unjudgeable"
    ]
    rgb_strict = [float(pair["rgb_judge_label"] == "correct") for pair in judged]
    ir_strict = [float(pair["ir_judge_label"] == "correct") for pair in judged]
    rgb_soft = [pair["rgb_judge_score"] for pair in judged if pair["rgb_judge_score"] is not None]
    ir_soft = [pair["ir_judge_score"] for pair in judged if pair["ir_judge_score"] is not None]
    rgb_accuracy = mean(rgb_strict)
    ir_accuracy = mean(ir_strict)
    rgb_soft_accuracy = mean(rgb_soft)
    ir_soft_accuracy = mean(ir_soft)
    return {
        "paired_total": len(pairs),
        "paired_judgeable": len(judged),
        "paired_excluded": len(pairs) - len(judged),
        "rgb_strict_accuracy": rgb_accuracy,
        "ir_strict_accuracy": ir_accuracy,
        "rgb_minus_ir_strict_pp": (
            (rgb_accuracy - ir_accuracy) * 100
            if rgb_accuracy is not None and ir_accuracy is not None
            else None
        ),
        "rgb_soft_accuracy": rgb_soft_accuracy,
        "ir_soft_accuracy": ir_soft_accuracy,
        "rgb_minus_ir_soft_pp": (
            (rgb_soft_accuracy - ir_soft_accuracy) * 100
            if rgb_soft_accuracy is not None and ir_soft_accuracy is not None
            else None
        ),
        "both_correct": sum(
            pair["rgb_judge_label"] == "correct" and pair["ir_judge_label"] == "correct"
            for pair in judged
        ),
        "rgb_only_correct": sum(
            pair["rgb_judge_label"] == "correct" and pair["ir_judge_label"] != "correct"
            for pair in judged
        ),
        "ir_only_correct": sum(
            pair["ir_judge_label"] == "correct" and pair["rgb_judge_label"] != "correct"
            for pair in judged
        ),
        "both_not_correct": sum(
            pair["rgb_judge_label"] != "correct" and pair["ir_judge_label"] != "correct"
            for pair in judged
        ),
    }


def grouped_rgb_ir_pairs(
    pairs: list[dict[str, Any]], keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        grouped[tuple(str(pair.get(key, "unknown")) for key in keys)].append(pair)
    rows: list[dict[str, Any]] = []
    for key_values, values in sorted(grouped.items()):
        row = dict(zip(keys, key_values))
        row.update(summarize_rgb_ir_pairs(values))
        rows.append(row)
    return rows


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


def write_rgb_ir_pair_report(
    path: Path,
    overall_rows: list[dict[str, Any]],
    section_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# RGB vs IR Paired LLM-Judge Comparison",
        "",
        "Each pair contains the RGB and IR answers for the same model and `source_qa_id`. "
        "Pairs with an `unjudgeable` or missing label are excluded from strict and soft accuracy.",
        "",
        "## Overall",
        "",
        "| Model | N | RGB strict | IR strict | RGB - IR (pp) | RGB-only correct | IR-only correct |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall_rows:
        lines.append(
            f"| {Path(row['model_name']).name} | {row['paired_judgeable']} | "
            f"{fmt(row.get('rgb_strict_accuracy'))} | {fmt(row.get('ir_strict_accuracy'))} | "
            f"{fmt(row.get('rgb_minus_ir_strict_pp'))} | {row['rgb_only_correct']} | "
            f"{row['ir_only_correct']} |"
        )
    lines.extend(
        [
            "",
            "## By QA Section",
            "",
            "`semantic_section` removes the `event_` and `depth_` prefixes, so sections with the same QA "
            "purpose are combined across their source modalities.",
            "",
            "| Model | QA section | N | RGB strict | IR strict | RGB - IR (pp) | RGB-only correct | IR-only correct |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in section_rows:
        lines.append(
            f"| {Path(row['model_name']).name} | {row['semantic_section']} | {row['paired_judgeable']} | "
            f"{fmt(row.get('rgb_strict_accuracy'))} | {fmt(row.get('ir_strict_accuracy'))} | "
            f"{fmt(row.get('rgb_minus_ir_strict_pp'))} | {row['rgb_only_correct']} | "
            f"{row['ir_only_correct']} |"
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
    rgb_ir_pairs = build_rgb_ir_pairs(rows)
    rgb_ir_overall_rows = grouped_rgb_ir_pairs(rgb_ir_pairs, ("provider", "model_name"))
    rgb_ir_section_rows = grouped_rgb_ir_pairs(
        rgb_ir_pairs,
        ("provider", "model_name", "semantic_section"),
    )

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "input_modality_summary.csv", input_rows)
    write_csv(args.output / "source_input_modality_matrix.csv", matrix_rows)
    write_csv(args.output / "source_section_input_modality_matrix.csv", section_rows)
    write_csv(args.output / "rgb_ir_paired_overall.csv", rgb_ir_overall_rows)
    write_csv(args.output / "rgb_ir_paired_by_section.csv", rgb_ir_section_rows)
    write_report(args.output / "cross_modality_report.md", input_rows, matrix_rows)
    write_rgb_ir_pair_report(
        args.output / "rgb_ir_paired_report.md",
        rgb_ir_overall_rows,
        rgb_ir_section_rows,
    )
    print(f"report: {args.output / 'cross_modality_report.md'}")


if __name__ == "__main__":
    main()
