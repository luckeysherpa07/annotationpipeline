from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object in {path}")
    return data


def build_reports(
    fused_results_path: Path,
    diagnostics_path: Path,
    output_json_path: Path,
    output_csv_path: Path,
) -> dict[str, Any]:
    fused = _load_json(fused_results_path)
    diagnostics = _load_json(diagnostics_path) if diagnostics_path.exists() else {}

    rows: list[dict[str, Any]] = []
    section_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    modality_counts: Counter[str] = Counter()
    outlier_counts: Counter[str] = Counter()
    selection_reason_counts: Counter[str] = Counter()

    for sample_key, sample_payload in fused.items():
        selected_qas = sample_payload.get("selected_reliable_qas", [])
        if not isinstance(selected_qas, list):
            continue

        for index, qa in enumerate(selected_qas, start=1):
            if not isinstance(qa, dict):
                continue
            section = str(qa.get("section", "unknown"))
            category = str(qa.get("category", qa.get("annotation_key", "unknown")))
            source_modality = qa.get("source_modality", "unknown")
            is_outlier = bool(qa.get("is_outlier", False))
            selection_reason = str(qa.get("selection_reason", "unknown"))
            section_counts[section] += 1
            category_counts[category] += 1
            modality_counts[str(source_modality)] += 1
            outlier_counts["outlier" if is_outlier else "not_outlier"] += 1
            selection_reason_counts[selection_reason] += 1

            rows.append(
                {
                    "sample_key": sample_key,
                    "row_index": index,
                    "section": section,
                    "category": category,
                    "source_modality": source_modality,
                    "annotation_key": qa.get("annotation_key", ""),
                    "qa_index": qa.get("qa_index"),
                    "qa_source": qa.get("qa_source", ""),
                    "fusion_score": qa.get("fusion_score"),
                    "support_score": qa.get("support_score"),
                    "modality_reliability": qa.get("modality_reliability"),
                    "is_outlier": is_outlier,
                    "selection_reason": selection_reason,
                    "question": qa.get("question", ""),
                    "answer": qa.get("answer", ""),
                }
            )

    dropped_reason_counts: Counter[str] = Counter()
    dropped_examples_by_reason: dict[str, list[dict[str, Any]]] = defaultdict(list)
    qa_selection_by_sample: dict[str, Any] = {}
    for sample_key, sample_diag in diagnostics.items():
        qa_selection = sample_diag.get("qa_selection", {}) if isinstance(sample_diag, dict) else {}
        if not isinstance(qa_selection, dict):
            continue
        qa_selection_by_sample[sample_key] = {
            key: qa_selection.get(key)
            for key in ("total_candidates", "passed_score_and_outlier_filters", "selected_count", "dropped_count", "drop_reason_counts")
        }
        dropped_qas = qa_selection.get("dropped_qas", [])
        if not isinstance(dropped_qas, list):
            continue
        for example in dropped_qas:
            if not isinstance(example, dict):
                continue
            reason = str(example.get("reason", "unknown"))
            dropped_reason_counts[reason] += 1
            enriched = {"sample_key": sample_key, **example}
            if len(dropped_examples_by_reason[reason]) < 50:
                dropped_examples_by_reason[reason].append(enriched)

    report = {
        "meta": {
            "fused_results": str(fused_results_path),
            "diagnostics": str(diagnostics_path),
            "total_samples": len(fused),
            "total_qa_rows": len(rows),
        },
        "summary": {
            "section_counts": dict(section_counts),
            "category_counts": dict(category_counts),
            "source_modality_counts": dict(modality_counts),
            "outlier_counts": dict(outlier_counts),
            "selection_reason_counts": dict(selection_reason_counts),
            "dropped_reason_counts_from_diagnostics": dict(dropped_reason_counts),
        },
        "qa_selection_by_sample": qa_selection_by_sample,
        "dropped_examples_by_reason": dict(dropped_examples_by_reason),
        "rows": rows,
    }

    with open(output_json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    fieldnames = [
        "sample_key",
        "row_index",
        "section",
        "category",
        "qa_source",
        "qa_index",
        "source_modality",
        "annotation_key",
        "fusion_score",
        "support_score",
        "modality_reliability",
        "is_outlier",
        "selection_reason",
        "question",
        "answer",
    ]
    with open(output_csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze selected_reliable_qas and export fusion QA stats.")
    parser.add_argument("--fused", default="fused_qa_results.json", help="Path to fused results JSON")
    parser.add_argument("--diagnostics", default="fusion_diagnostics.json", help="Path to diagnostics JSON")
    parser.add_argument("--out-json", default="fusion_qa_stats.json", help="Output aggregated stats JSON")
    parser.add_argument("--out-csv", default="fusion_qa_rows.csv", help="Output flattened QA rows CSV")
    args = parser.parse_args()

    report = build_reports(
        fused_results_path=Path(args.fused),
        diagnostics_path=Path(args.diagnostics),
        output_json_path=Path(args.out_json),
        output_csv_path=Path(args.out_csv),
    )

    print("total_samples", report["meta"]["total_samples"])
    print("total_qa_rows", report["meta"]["total_qa_rows"])
    print("section_counts", report["summary"]["section_counts"])
    print("category_counts", report["summary"]["category_counts"])
    print("source_modality_counts", report["summary"]["source_modality_counts"])
    print("outlier_counts", report["summary"]["outlier_counts"])
    print("dropped_reason_counts", report["summary"]["dropped_reason_counts_from_diagnostics"])


if __name__ == "__main__":
    main()
