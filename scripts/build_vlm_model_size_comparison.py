#!/usr/bin/env python3
"""Build controlled VLM model and parameter-size comparison tables."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


DEFAULT_INPUTS = (
    Path("outputs/evaluations/vlm_8frame_aligned_4b/answer_metrics_summary.csv"),
    Path("outputs/evaluations/vlm_8frame_aligned_8b/answer_metrics_summary.csv"),
    Path("outputs/evaluations/vlm_native_video_4b/answer_metrics_summary.csv"),
    Path("outputs/evaluations/vlm_native_video_8b/answer_metrics_summary.csv"),
)
DEFAULT_OUTPUT = Path("outputs/evaluations/vlm_model_size_comparison")
DEFAULT_EVALUATION_DIRS = (
    Path("outputs/evaluations/vlm_8frame_aligned_4b"),
    Path("outputs/evaluations/vlm_8frame_aligned_8b"),
    Path("outputs/evaluations/vlm_native_video_4b"),
    Path("outputs/evaluations/vlm_native_video_8b"),
)
METRICS = (
    "bleu_4",
    "rouge_l_f1",
    "meteor",
    "judge_strict_accuracy",
    "judge_soft_accuracy",
)
DETAILED_METRICS = (
    "task_aware_score",
    "token_f1",
    "bleu_4",
    "rouge_l_f1",
    "meteor",
    "judge_strict_accuracy",
    "judge_soft_accuracy",
)


def model_label(model_name: str) -> str:
    return model_name.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def model_family(label: str) -> str:
    if label.startswith("InternVL3_5"):
        return "InternVL3.5"
    if label.startswith("Molmo2"):
        return "Molmo2"
    if label.startswith("Qwen3-VL"):
        return "Qwen3-VL"
    return re.sub(r"-(?:4B|8B)(?:-Instruct)?$", "", label)


def size_tier(label: str) -> str:
    match = re.search(r"-(4B|8B)(?:-|$)", label)
    if not match:
        raise ValueError(f"Cannot determine size tier from model name: {label}")
    return match.group(1)


def input_label(row: dict[str, str]) -> str:
    if row["input_type"] == "frame":
        counts = row.get("frame_counts", "").strip("[] ")
        return f"frame {counts}" if counts else "frame"
    return row["input_type"]


def read_rows(paths: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            for raw in csv.DictReader(handle):
                label = model_label(raw["model_name"])
                row: dict[str, object] = {
                    "input": input_label(raw),
                    "size": size_tier(label),
                    "model_family": model_family(label),
                    "model": label,
                    "answer_rate": float(raw["answer_rate"]),
                }
                row.update({metric: float(raw[metric]) for metric in METRICS})
                rows.append(row)
    return sorted(rows, key=lambda row: (str(row["input"]), str(row["size"]), str(row["model_family"])))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_size_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    indexed = {(row["input"], row["model_family"], row["size"]): row for row in rows}
    comparisons: list[dict[str, object]] = []
    for input_name, family in sorted({(str(row["input"]), str(row["model_family"])) for row in rows}):
        row_4b = indexed.get((input_name, family, "4B"))
        row_8b = indexed.get((input_name, family, "8B"))
        if not row_4b or not row_8b:
            continue
        comparison: dict[str, object] = {
            "input": input_name,
            "model_family": family,
            "model_4b": row_4b["model"],
            "model_8b": row_8b["model"],
        }
        for metric in METRICS:
            value_4b = float(row_4b[metric])
            value_8b = float(row_8b[metric])
            comparison[f"{metric}_4b"] = value_4b
            comparison[f"{metric}_8b"] = value_8b
            comparison[f"{metric}_delta_8b_minus_4b"] = value_8b - value_4b
        comparisons.append(comparison)
    return comparisons


def optional_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    return float(value)


def semantic_section(section: str) -> str:
    return re.sub(r"^(?:depth|event)_", "", section)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def read_detailed_rows(evaluation_dirs: list[Path]) -> list[dict[str, object]]:
    """Aggregate per-item scores by model, modality, and modality-section."""
    records: list[dict[str, str]] = []
    for directory in evaluation_dirs:
        path = directory / "per_item_scores.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing detailed evaluation input: {path}")
        with path.open(encoding="utf-8", newline="") as handle:
            records.extend(csv.DictReader(handle))

    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for raw in records:
        label = model_label(raw["model_name"])
        input_name = "frame " + raw["frame_count"] if raw["input_type"] == "frame" else raw["input_type"]
        base = (input_name, size_tier(label), model_family(label), label)
        grouped[(*base, raw["modality"], "", "")].append(raw)
        grouped[(*base, raw["modality"], semantic_section(raw["section"]), raw["section"])].append(raw)

    rows: list[dict[str, object]] = []
    for key, items in grouped.items():
        input_name, size, family, model, modality, section, section_raw = key
        answered = [item for item in items if item["status"] == "answered"]
        judged = [item for item in answered if item.get("judge_score", "").strip()]
        strict = [1.0 if item.get("judge_label") == "correct" else 0.0 for item in judged]
        soft = [float(item["judge_score"]) for item in judged]
        row: dict[str, object] = {
            "input": input_name,
            "size": size,
            "model_family": family,
            "model": model,
            "modality": modality,
            "section": section or "ALL",
            "section_raw": section_raw or "ALL",
            "total": len(items),
            "answered": len(answered),
            "answer_rate": len(answered) / len(items),
            "judge_evaluated": len(judged),
            "judge_strict_accuracy": mean(strict),
            "judge_soft_accuracy": mean(soft),
        }
        for metric in DETAILED_METRICS[:-2]:
            values = [value for item in answered if (value := optional_float(item.get(metric))) is not None]
            row[metric] = mean(values)
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            str(row["input"]), str(row["size"]), str(row["model_family"]),
            str(row["modality"]), str(row["section"]), str(row["section_raw"]),
        ),
    )


def build_detailed_size_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    indexed = {
        (row["input"], row["model_family"], row["modality"], row["section"], row["size"]): row
        for row in rows
    }
    comparisons: list[dict[str, object]] = []
    dimensions = {
        (str(row["input"]), str(row["model_family"]), str(row["modality"]), str(row["section"]))
        for row in rows
    }
    for input_name, family, modality, section in sorted(dimensions):
        row_4b = indexed.get((input_name, family, modality, section, "4B"))
        row_8b = indexed.get((input_name, family, modality, section, "8B"))
        if not row_4b or not row_8b:
            continue
        comparison: dict[str, object] = {
            "input": input_name,
            "model_family": family,
            "modality": modality,
            "section": section,
            "section_raw_4b": row_4b["section_raw"],
            "section_raw_8b": row_8b["section_raw"],
            "total_4b": row_4b["total"],
            "total_8b": row_8b["total"],
        }
        for metric in DETAILED_METRICS:
            value_4b = float(row_4b[metric])
            value_8b = float(row_8b[metric])
            comparison[f"{metric}_4b"] = value_4b
            comparison[f"{metric}_8b"] = value_8b
            comparison[f"{metric}_delta_8b_minus_4b"] = value_8b - value_4b
        comparisons.append(comparison)
    return comparisons


def write_detailed_report(path: Path, rows: list[dict[str, object]]) -> None:
    modality_rows = [row for row in rows if row["section"] == "ALL"]
    section_rows = [row for row in rows if row["section"] != "ALL"]
    lines = [
        "# Detailed VLM Comparison",
        "",
        "The tables use blinded LLM-judge accuracy as the headline semantic metric. "
        "`N` is shown because modality-section groups have different sample counts.",
        "",
        "## Modality Comparison",
        "",
        "| Input | Size | Model | Modality | N | Judge strict | Judge soft | Task-aware |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in modality_rows:
        lines.append(
            f"| {row['input']} | {row['size']} | {row['model']} | {row['modality']} | "
            f"{row['total']} | {fmt(row['judge_strict_accuracy'])} | "
            f"{fmt(row['judge_soft_accuracy'])} | {fmt(row['task_aware_score'])} |"
        )

    lines.extend([
        "",
        "## Best Configuration By Modality And Section",
        "",
        "This is a descriptive maximum across all compared input types, sizes, and models, not a paired significance test.",
        "",
        "| Modality | Section | Best input | Size | Model | N | Judge strict | Judge soft |",
        "|---|---|---|---:|---|---:|---:|---:|",
    ])
    best: dict[tuple[str, str], dict[str, object]] = {}
    for row in section_rows:
        key = (str(row["modality"]), str(row["section"]))
        if key not in best or float(row["judge_strict_accuracy"]) > float(best[key]["judge_strict_accuracy"]):
            best[key] = row
    for modality, section in sorted(best):
        row = best[(modality, section)]
        lines.append(
            f"| {modality} | {section} | {row['input']} | {row['size']} | {row['model']} | "
            f"{row['total']} | {fmt(row['judge_strict_accuracy'])} | {fmt(row['judge_soft_accuracy'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: object) -> str:
    return f"{float(value):.4f}"


def write_report(path: Path, model_rows: list[dict[str, object]], size_rows: list[dict[str, object]]) -> None:
    lines = [
        "# VLM Model and Size Comparison",
        "",
        "## Same Input And Size, Different Models",
        "",
        "| Input | Size | Model | Judge strict | Judge soft | BLEU-4 | ROUGE-L | METEOR |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in model_rows:
        lines.append(
            f"| {row['input']} | {row['size']} | {row['model']} | "
            f"{fmt(row['judge_strict_accuracy'])} | {fmt(row['judge_soft_accuracy'])} | "
            f"{fmt(row['bleu_4'])} | {fmt(row['rouge_l_f1'])} | {fmt(row['meteor'])} |"
        )
    lines.extend(
        [
            "",
            "## Same Input And Model, Different Sizes",
            "",
            "Deltas are calculated as `8B - 4B`.",
            "",
            "| Input | Model family | 4B strict | 8B strict | Strict delta | 4B soft | 8B soft | Soft delta |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in size_rows:
        lines.append(
            f"| {row['input']} | {row['model_family']} | "
            f"{fmt(row['judge_strict_accuracy_4b'])} | {fmt(row['judge_strict_accuracy_8b'])} | "
            f"{fmt(row['judge_strict_accuracy_delta_8b_minus_4b'])} | "
            f"{fmt(row['judge_soft_accuracy_4b'])} | {fmt(row['judge_soft_accuracy_8b'])} | "
            f"{fmt(row['judge_soft_accuracy_delta_8b_minus_4b'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, default=list(DEFAULT_INPUTS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--evaluation-dir", nargs="+", type=Path, default=list(DEFAULT_EVALUATION_DIRS),
        help="Evaluation directories containing per_item_scores.csv for detailed aggregation.",
    )
    args = parser.parse_args()

    rows = read_rows(args.input)
    size_rows = build_size_rows(rows)
    detailed_rows = read_detailed_rows(args.evaluation_dir)
    detailed_size_rows = build_detailed_size_rows(detailed_rows)
    args.output.mkdir(parents=True, exist_ok=True)

    model_fields = ["input", "size", "model_family", "model", "answer_rate", *METRICS]
    size_fields = ["input", "model_family", "model_4b", "model_8b"]
    for metric in METRICS:
        size_fields.extend((f"{metric}_4b", f"{metric}_8b", f"{metric}_delta_8b_minus_4b"))

    write_csv(args.output / "same_size_model_comparison.csv", rows, model_fields)
    write_csv(args.output / "same_model_size_comparison.csv", size_rows, size_fields)
    write_report(args.output / "comparison_report.md", rows, size_rows)

    detailed_fields = [
        "input", "size", "model_family", "model", "modality", "section", "section_raw",
        "total", "answered", "answer_rate", "judge_evaluated", *DETAILED_METRICS,
    ]
    write_csv(
        args.output / "modality_comparison.csv",
        [row for row in detailed_rows if row["section"] == "ALL"],
        detailed_fields,
    )
    write_csv(
        args.output / "modality_section_comparison.csv",
        [row for row in detailed_rows if row["section"] != "ALL"],
        detailed_fields,
    )
    detailed_size_fields = [
        "input", "model_family", "modality", "section", "section_raw_4b", "section_raw_8b",
        "total_4b", "total_8b",
    ]
    for metric in DETAILED_METRICS:
        detailed_size_fields.extend((f"{metric}_4b", f"{metric}_8b", f"{metric}_delta_8b_minus_4b"))
    write_csv(
        args.output / "modality_section_size_comparison.csv",
        detailed_size_rows,
        detailed_size_fields,
    )
    write_detailed_report(args.output / "detailed_comparison_report.md", detailed_rows)


if __name__ == "__main__":
    main()
