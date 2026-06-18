#!/usr/bin/env python3
"""Build controlled VLM model and parameter-size comparison tables."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


DEFAULT_INPUTS = (
    Path("outputs/evaluations/vlm_8frame_aligned_4b/answer_metrics_summary.csv"),
    Path("outputs/evaluations/vlm_8frame_aligned_8b/answer_metrics_summary.csv"),
    Path("outputs/evaluations/vlm_native_video_4b/answer_metrics_summary.csv"),
    Path("outputs/evaluations/vlm_native_video_8b/answer_metrics_summary.csv"),
)
DEFAULT_OUTPUT = Path("outputs/evaluations/vlm_model_size_comparison")
METRICS = (
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
    args = parser.parse_args()

    rows = read_rows(args.input)
    size_rows = build_size_rows(rows)
    args.output.mkdir(parents=True, exist_ok=True)

    model_fields = ["input", "size", "model_family", "model", "answer_rate", *METRICS]
    size_fields = ["input", "model_family", "model_4b", "model_8b"]
    for metric in METRICS:
        size_fields.extend((f"{metric}_4b", f"{metric}_8b", f"{metric}_delta_8b_minus_4b"))

    write_csv(args.output / "same_size_model_comparison.csv", rows, model_fields)
    write_csv(args.output / "same_model_size_comparison.csv", size_rows, size_fields)
    write_report(args.output / "comparison_report.md", rows, size_rows)


if __name__ == "__main__":
    main()
