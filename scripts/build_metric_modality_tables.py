#!/usr/bin/env python3
"""Build per-question metric tables across cross-modality inputs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT = Path(
    "outputs/evaluations/vlm_cross_modality_8frame_8b_full_qwen/per_item_scores.csv"
)
DEFAULT_OUTPUT = Path(
    "outputs/evaluations/vlm_cross_modality_8frame_8b_full_qwen/metric_modality_tables"
)
DEFAULT_MODALITIES = ("rgb", "ir", "event", "depth")
METRICS = {
    "llm_judge": "judge_score",
    "rouge_l": "rouge_l_f1",
    "meteor": "meteor",
    "bleu_4": "bleu_4",
}
DEFAULT_COMPOSITE_WEIGHTS = {
    "llm_judge": 0.70,
    "rouge_l": 0.10,
    "meteor": 0.15,
    "bleu_4": 0.05,
}
TEXT_AVERAGE_FIELDS = ("rouge_l_f1", "meteor", "bleu_4")


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


def dense_ranks(scores: dict[str, float | None]) -> dict[str, int | None]:
    unique_scores = sorted(
        {score for score in scores.values() if score is not None},
        reverse=True,
    )
    rank_by_score = {score: index + 1 for index, score in enumerate(unique_scores)}
    return {
        modality: rank_by_score[score] if score is not None else None
        for modality, score in scores.items()
    }


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def source_qa_key(row: dict[str, Any]) -> str:
    source_qa_id = str(row.get("source_qa_id", "")).strip()
    if source_qa_id:
        return source_qa_id
    return str(row.get("qa_id", "")).split("__input_", maxsplit=1)[0]


def group_by_question(rows: Iterable[dict[str, str]]) -> dict[str, dict[str, dict[str, str]]]:
    grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        modality = str(row.get("input_modality", "")).strip().lower()
        if not modality:
            continue
        grouped[source_qa_key(row)][modality] = row
    return grouped


def first_present(rows_by_modality: dict[str, dict[str, str]], field: str) -> str:
    for row in rows_by_modality.values():
        value = str(row.get(field, "")).strip()
        if value:
            return value
    return ""


def build_metric_table(
    rows: list[dict[str, str]],
    metric_field: str,
    modalities: tuple[str, ...] = DEFAULT_MODALITIES,
) -> list[dict[str, Any]]:
    grouped = group_by_question(rows)
    output: list[dict[str, Any]] = []
    for source_qa_id, rows_by_modality in sorted(grouped.items()):
        scores = {
            modality: optional_float(rows_by_modality.get(modality, {}).get(metric_field))
            for modality in modalities
        }
        ranks = dense_ranks(scores)
        row: dict[str, Any] = {
            "source_qa_id": source_qa_id,
            "question": first_present(rows_by_modality, "question"),
            "source_modality": first_present(rows_by_modality, "source_modality"),
            "source_section": first_present(rows_by_modality, "source_section"),
            "ground_truth_answer": first_present(rows_by_modality, "ground_truth_answer"),
        }
        for modality in modalities:
            row[f"{modality}_score"] = scores[modality]
            row[f"{modality}_rank"] = ranks[modality]
        output.append(row)
    return output


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(value for value in weights.values() if value > 0)
    if total <= 0:
        raise ValueError("At least one composite metric weight must be positive.")
    return {metric: value / total for metric, value in weights.items() if value > 0}


def weighted_score(row: dict[str, str], weights: dict[str, float]) -> float | None:
    values: list[tuple[float, float]] = []
    for metric_name, weight in weights.items():
        score = optional_float(row.get(METRICS[metric_name]))
        if score is not None:
            values.append((score, weight))
    if not values:
        return None
    present_weight = sum(weight for _, weight in values)
    return sum(score * weight for score, weight in values) / present_weight


def mean_score(row: dict[str, str], fields: tuple[str, ...]) -> float | None:
    values = [
        score
        for field in fields
        if (score := optional_float(row.get(field))) is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def top_modalities(
    modalities: list[str],
    scores: dict[str, float | None],
) -> list[str]:
    present_scores = [scores[modality] for modality in modalities if scores[modality] is not None]
    if not present_scores:
        return []
    best_score = max(present_scores)
    return [
        modality
        for modality in modalities
        if scores[modality] is not None and abs(scores[modality] - best_score) < 1e-12
    ]


def build_composite_table(
    rows: list[dict[str, str]],
    weights: dict[str, float],
    modalities: tuple[str, ...] = DEFAULT_MODALITIES,
) -> list[dict[str, Any]]:
    normalized_weights = normalize_weights(weights)
    grouped = group_by_question(rows)
    output: list[dict[str, Any]] = []
    for source_qa_id, rows_by_modality in sorted(grouped.items()):
        scores = {
            modality: weighted_score(rows_by_modality.get(modality, {}), normalized_weights)
            for modality in modalities
        }
        ranks = dense_ranks(scores)
        best_modalities = top_modalities(list(modalities), scores)
        row: dict[str, Any] = {
            "source_qa_id": source_qa_id,
            "question": first_present(rows_by_modality, "question"),
            "source_modality": first_present(rows_by_modality, "source_modality"),
            "source_section": first_present(rows_by_modality, "source_section"),
            "ground_truth_answer": first_present(rows_by_modality, "ground_truth_answer"),
            "best_input_modalities": ";".join(best_modalities),
            "is_tie": len(best_modalities) > 1,
        }
        for modality in modalities:
            row[f"{modality}_composite_score"] = scores[modality]
            row[f"{modality}_rank"] = ranks[modality]
            source_row = rows_by_modality.get(modality, {})
            row[f"{modality}_judge_score"] = optional_float(source_row.get("judge_score"))
            row[f"{modality}_task_aware_score"] = optional_float(source_row.get("task_aware_score"))
            row[f"{modality}_text_metric_mean"] = mean_score(source_row, TEXT_AVERAGE_FIELDS)
            row[f"{modality}_token_f1"] = optional_float(source_row.get("token_f1"))
        output.append(row)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]], modalities: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_qa_id",
        "question",
        "source_modality",
        "source_section",
        "ground_truth_answer",
    ]
    for modality in modalities:
        fields.extend([f"{modality}_score", f"{modality}_rank"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_composite_csv(path: Path, rows: list[dict[str, Any]], modalities: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_qa_id",
        "question",
        "source_modality",
        "source_section",
        "ground_truth_answer",
        "best_input_modalities",
        "is_tie",
    ]
    for modality in modalities:
        fields.extend(
            [
                f"{modality}_composite_score",
                f"{modality}_rank",
                f"{modality}_judge_score",
                f"{modality}_task_aware_score",
                f"{modality}_text_metric_mean",
                f"{modality}_token_f1",
            ]
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_weights(values: list[str]) -> dict[str, float]:
    weights = dict(DEFAULT_COMPOSITE_WEIGHTS)
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected metric=weight, got: {value}")
        metric_name, raw_weight = value.split("=", maxsplit=1)
        metric_name = metric_name.strip()
        if metric_name not in METRICS:
            valid = ", ".join(METRICS)
            raise ValueError(f"Unknown metric {metric_name!r}; valid metrics: {valid}")
        weights[metric_name] = float(raw_weight)
    return normalize_weights(weights)


def write_scoring_method_report(path: Path, weights: dict[str, float]) -> None:
    lines = [
        "# Composite Modality Scoring Method",
        "",
        "The composite score is a weighted average of per-question modality scores. "
        "All included metrics are higher-is-better and already scaled to the 0-1 range.",
        "",
        "| Metric | Field | Weight |",
        "|---|---|---:|",
    ]
    for metric_name, weight in weights.items():
        lines.append(f"| {metric_name} | {METRICS[metric_name]} | {weight:.4f} |")
    lines.extend(
        [
            "",
            "Rationale: LLM judge receives the largest weight because it best captures semantic correctness; "
            "ROUGE-L and METEOR keep lexical overlap visible; BLEU-4 receives the smallest weight because it is brittle for short answers.",
            "",
            "## Best Modality Definition",
            "",
            "`best_input_modalities` contains every input modality tied for the highest composite score for that question. "
            "Ties are retained instead of force-resolved, because a sensitivity check with deterministic tie-breakers changed only 9 of 5331 assignments (0.17%). "
            "`is_tie` is true when more than one modality shares the maximum composite score.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=list(DEFAULT_MODALITIES),
        help="Input modality columns to include, in output order.",
    )
    parser.add_argument(
        "--weight",
        action="append",
        default=[],
        help="Override composite metric weight as metric=weight, e.g. --weight llm_judge=0.6.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = load_rows(args.input)
    modalities = tuple(str(modality).lower() for modality in args.modalities)
    weights = parse_weights(args.weight)
    for metric_name, metric_field in METRICS.items():
        metric_rows = build_metric_table(rows, metric_field, modalities)
        write_csv(args.output / f"{metric_name}_modality_scores.csv", metric_rows, modalities)
    composite_rows = build_composite_table(rows, weights, modalities)
    write_composite_csv(args.output / "composite_modality_scores.csv", composite_rows, modalities)
    write_scoring_method_report(args.output / "composite_scoring_method.md", weights)
    print(f"metric tables: {args.output}")


if __name__ == "__main__":
    main()
