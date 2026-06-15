"""Score VLM answers and write reproducible evaluation reports."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .answer_metrics import deterministic_metrics
from .metric_router import route_metric
from .result_loader import EvaluationRecord


SCORE_FIELDS = (
    "normalized_exact_match",
    "token_f1",
    "rouge_l_f1",
    "anls",
    "character_f1",
    "task_aware_score",
)


def _model_key(record: EvaluationRecord) -> str:
    model_short = Path(record.model_name).name or record.model_name
    source_tag = hashlib.sha256(record.source_path.encode("utf-8")).hexdigest()[:8]
    return f"{record.provider}:{model_short}:{record.input_type}:{source_tag}"


def _frame_configuration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame_counts = sorted(
        {
            int(row["frame_count"])
            for row in rows
            if isinstance(row.get("frame_count"), int)
        }
    )
    configured_maxima = sorted(
        {
            int(row["max_frames_per_item"])
            for row in rows
            if isinstance(row.get("max_frames_per_item"), int)
        }
    )
    return {
        "frame_counts": frame_counts,
        "max_frames_per_item": (
            configured_maxima[0] if len(configured_maxima) == 1 else configured_maxima
        ),
    }


def score_records(
    records: Iterable[EvaluationRecord],
    judgments: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    judgments = judgments or {}
    scored: list[dict[str, Any]] = []
    for record in records:
        metrics = deterministic_metrics(
            record.ground_truth_answer,
            record.model_answer,
        )
        routed = route_metric(
            record.section,
            record.question,
            record.ground_truth_answer,
            metrics,
        )
        judgment = judgments.get(record.record_id, {})
        row = record.to_dict()
        row.pop("source_metadata", None)
        row["model_key"] = _model_key(record)
        row.update(metrics)
        row.update(
            {
                "task_aware_metric": routed["metric"],
                "task_aware_score": routed["score"],
                "task_aware_reason": routed["reason"],
                "judge_label": judgment.get("label"),
                "judge_score": judgment.get("score"),
                "judge_reason": judgment.get("reason"),
                "judge_error_type": judgment.get("error_type"),
                "judge_model": judgment.get("judge_model"),
            }
        )
        scored.append(row)
    return scored


def _mean(values: Iterable[Any]) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return statistics.mean(numeric) if numeric else None


def _percentile(values: Iterable[Any], percentile: float) -> float | None:
    numeric = sorted(float(value) for value in values if isinstance(value, (int, float)))
    if not numeric:
        return None
    index = round((len(numeric) - 1) * percentile)
    return numeric[index]


def bootstrap_mean_ci(
    values: Iterable[Any],
    *,
    samples: int = 1000,
    seed: int = 20260614,
) -> dict[str, float | None]:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    if not numeric:
        return {"mean": None, "low": None, "high": None}
    if len(numeric) == 1:
        return {"mean": numeric[0], "low": numeric[0], "high": numeric[0]}
    rng = random.Random(seed)
    means = [
        statistics.mean(rng.choices(numeric, k=len(numeric)))
        for _ in range(max(1, samples))
    ]
    means.sort()
    return {
        "mean": statistics.mean(numeric),
        "low": means[int(0.025 * (len(means) - 1))],
        "high": means[int(0.975 * (len(means) - 1))],
    }


def _group_summary(
    rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    answered = [
        row
        for row in rows
        if row.get("status") == "answered" and str(row.get("model_answer", "")).strip()
    ]
    summary: dict[str, Any] = {
        "total": len(rows),
        "answered": len(answered),
        "answer_rate": len(answered) / len(rows) if rows else 0.0,
        "status_counts": dict(Counter(str(row.get("status")) for row in rows)),
        "repetition_rate": _mean(bool(row.get("repetition_flag")) for row in answered),
        "conciseness_violation_rate": _mean(
            bool(row.get("conciseness_violation")) for row in answered
        ),
        "latency_mean_seconds": _mean(row.get("latency_seconds") for row in rows),
        "latency_median_seconds": _percentile(
            (row.get("latency_seconds") for row in rows), 0.5
        ),
        "latency_p95_seconds": _percentile(
            (row.get("latency_seconds") for row in rows), 0.95
        ),
        "throughput_qa_per_hour": None,
        "peak_gpu_gb_max": max(
            (
                float(row["peak_gpu_gb"])
                for row in rows
                if isinstance(row.get("peak_gpu_gb"), (int, float))
            ),
            default=None,
        ),
        "incremental_peak_gpu_gb_max": max(
            (
                float(row["incremental_peak_gpu_gb"])
                for row in rows
                if isinstance(row.get("incremental_peak_gpu_gb"), (int, float))
            ),
            default=None,
        ),
    }
    total_latency = sum(
        float(row["latency_seconds"])
        for row in rows
        if isinstance(row.get("latency_seconds"), (int, float))
    )
    if total_latency > 0:
        summary["throughput_qa_per_hour"] = len(answered) * 3600 / total_latency
    for field in SCORE_FIELDS:
        summary[field] = _mean(row.get(field) for row in answered)

    judged = [row for row in answered if row.get("judge_label")]
    judge_counts = Counter(str(row.get("judge_label")) for row in judged)
    judge_scores = [
        row.get("judge_score")
        for row in judged
        if isinstance(row.get("judge_score"), (int, float))
    ]
    strict_values = [
        float(row.get("judge_label") == "correct")
        for row in judged
        if row.get("judge_label") != "unjudgeable"
    ]
    summary.update(
        {
            "judge_evaluated": len(judged),
            "judge_label_counts": dict(judge_counts),
            "judge_strict_accuracy": _mean(strict_values),
            "judge_soft_accuracy": _mean(judge_scores),
            "judge_unjudgeable_rate": (
                judge_counts.get("unjudgeable", 0) / len(judged) if judged else None
            ),
            "task_aware_score_ci95": bootstrap_mean_ci(
                (row.get("task_aware_score") for row in answered),
                samples=bootstrap_samples,
            ),
            "token_f1_ci95": bootstrap_mean_ci(
                (row.get("token_f1") for row in answered),
                samples=bootstrap_samples,
            ),
            "judge_strict_accuracy_ci95": bootstrap_mean_ci(
                strict_values,
                samples=bootstrap_samples,
            ),
        }
    )
    return summary


def _macro_average(group_summaries: dict[str, dict[str, Any]], field: str) -> float | None:
    return _mean(summary.get(field) for summary in group_summaries.values())


def summarize_scores(
    rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int = 1000,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[str(row["model_key"])].append(row)

    summary_models: dict[str, Any] = {}
    modality_rows: list[dict[str, Any]] = []
    section_rows: list[dict[str, Any]] = []
    for model_key, model_rows in sorted(by_model.items()):
        by_modality: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_section: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in model_rows:
            by_modality[str(row.get("modality", "unknown"))].append(row)
            by_section[str(row.get("section", "unknown"))].append(row)
        modality_summaries = {
            key: _group_summary(value, bootstrap_samples=bootstrap_samples)
            for key, value in sorted(by_modality.items())
        }
        section_summaries = {
            key: _group_summary(value, bootstrap_samples=bootstrap_samples)
            for key, value in sorted(by_section.items())
        }
        overall = _group_summary(model_rows, bootstrap_samples=bootstrap_samples)
        overall["modality_macro_task_aware_score"] = _macro_average(
            modality_summaries, "task_aware_score"
        )
        overall["section_macro_task_aware_score"] = _macro_average(
            section_summaries, "task_aware_score"
        )
        overall["modality_macro_judge_strict_accuracy"] = _macro_average(
            modality_summaries, "judge_strict_accuracy"
        )
        overall["section_macro_judge_strict_accuracy"] = _macro_average(
            section_summaries, "judge_strict_accuracy"
        )
        frame_configuration = _frame_configuration(model_rows)
        summary_models[model_key] = {
            "identity": {
                key: model_rows[0].get(key)
                for key in (
                    "provider",
                    "model_name",
                    "input_type",
                    "benchmark_type",
                    "source_path",
                )
            }
            | frame_configuration,
            "overall": overall,
            "by_modality": modality_summaries,
            "by_section": section_summaries,
        }
        for modality, values in modality_summaries.items():
            modality_rows.append(
                {
                    "model_key": model_key,
                    "modality": modality,
                    **frame_configuration,
                    **values,
                }
            )
        for section, values in section_summaries.items():
            section_rows.append(
                {
                    "model_key": model_key,
                    "section": section,
                    **frame_configuration,
                    **values,
                }
            )
    return {"models": summary_models}, modality_rows, section_rows


def _binomial_two_sided_p_value(successes: int, trials: int) -> float:
    if trials == 0:
        return 1.0
    tail_end = min(successes, trials - successes)
    log_probabilities = [
        math.lgamma(trials + 1)
        - math.lgamma(index + 1)
        - math.lgamma(trials - index + 1)
        - trials * math.log(2.0)
        for index in range(tail_end + 1)
    ]
    maximum = max(log_probabilities)
    probability = math.exp(maximum) * sum(
        math.exp(value - maximum) for value in log_probabilities
    )
    return min(1.0, 2 * probability)


def pairwise_judge_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("judge_label") in {"correct", "partially_correct", "incorrect"}:
            by_model[str(row["model_key"])][str(row["qa_id"])] = row
    models = sorted(by_model)
    comparisons: list[dict[str, Any]] = []
    for left_index, left in enumerate(models):
        for right in models[left_index + 1 :]:
            shared_ids = sorted(set(by_model[left]) & set(by_model[right]))
            if not shared_ids:
                continue
            left_only_correct = 0
            right_only_correct = 0
            disagreements = 0
            for qa_id in shared_ids:
                left_row = by_model[left][qa_id]
                right_row = by_model[right][qa_id]
                left_correct = left_row.get("judge_label") == "correct"
                right_correct = right_row.get("judge_label") == "correct"
                left_only_correct += int(left_correct and not right_correct)
                right_only_correct += int(right_correct and not left_correct)
                disagreements += int(left_row.get("judge_label") != right_row.get("judge_label"))
            discordant = left_only_correct + right_only_correct
            comparisons.append(
                {
                    "model_a": left,
                    "model_b": right,
                    "shared_items": len(shared_ids),
                    "label_disagreement_rate": disagreements / len(shared_ids),
                    "model_a_only_correct": left_only_correct,
                    "model_b_only_correct": right_only_correct,
                    "mcnemar_exact_p_value": _binomial_two_sided_p_value(
                        left_only_correct,
                        discordant,
                    ),
                }
            )
    return comparisons


def _flatten_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
        for key, value in row.items()
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_flatten_for_csv(row) for row in rows)


def _write_report_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# VLM Answer Evaluation Report",
        "",
        "| Model | Input | Answer rate | Task-aware | Token F1 | Judge strict | Judge soft | P95 latency | Peak GPU |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_key, model in summary.get("models", {}).items():
        identity = model["identity"]
        values = model["overall"]
        format_value = lambda value: "" if value is None else f"{value:.4f}"
        frame_counts = identity.get("frame_counts") or []
        input_label = str(identity.get("input_type", ""))
        if frame_counts:
            input_label += f" ({'/'.join(str(value) for value in frame_counts)} frames)"
        lines.append(
            "| "
            + " | ".join(
                (
                    Path(str(identity.get("model_name", model_key))).name,
                    input_label,
                    format_value(values.get("answer_rate")),
                    format_value(values.get("task_aware_score")),
                    format_value(values.get("token_f1")),
                    format_value(values.get("judge_strict_accuracy")),
                    format_value(values.get("judge_soft_accuracy")),
                    format_value(values.get("latency_p95_seconds")),
                    format_value(values.get("peak_gpu_gb_max")),
                )
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_evaluation_outputs(
    output_dir: Path | str,
    rows: list[dict[str, Any]],
    *,
    skipped_inputs: list[dict[str, str]] | None = None,
    bootstrap_samples: int = 1000,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, modality_rows, section_rows = summarize_scores(
        rows,
        bootstrap_samples=bootstrap_samples,
    )
    summary["pairwise_judge_comparisons"] = pairwise_judge_comparisons(rows)
    summary["skipped_inputs"] = skipped_inputs or []
    failures = [
        row
        for row in rows
        if row.get("status") != "answered" or not str(row.get("model_answer", "")).strip()
    ]
    paths = {
        "per_item_json": output_dir / "per_item_scores.json",
        "per_item_csv": output_dir / "per_item_scores.csv",
        "summary_json": output_dir / "summary.json",
        "summary_csv": output_dir / "summary.csv",
        "modality_csv": output_dir / "modality_scores.csv",
        "section_csv": output_dir / "section_scores.csv",
        "failures_csv": output_dir / "failures.csv",
        "report_md": output_dir / "report.md",
    }
    paths["per_item_json"].write_text(
        json.dumps({"items": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_csv(paths["per_item_csv"], rows)
    paths["summary_json"].write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary_rows = [
        {"model_key": key, **value["identity"], **value["overall"]}
        for key, value in summary["models"].items()
    ]
    _write_csv(paths["summary_csv"], summary_rows)
    _write_csv(paths["modality_csv"], modality_rows)
    _write_csv(paths["section_csv"], section_rows)
    _write_csv(paths["failures_csv"], failures)
    _write_report_markdown(paths["report_md"], summary)
    return paths
