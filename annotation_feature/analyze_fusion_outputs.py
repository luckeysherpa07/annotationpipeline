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
    modality_counts: Counter[str] = Counter()
    outlier_counts: Counter[str] = Counter()
    drop_reason_counts: Counter[str] = Counter()

    for sample_key, sample_payload in fused.items():
        section_qas = sample_payload.get("section_evidence_qas", {})
        if not isinstance(section_qas, dict):
            continue

        for section, qa_items in section_qas.items():
            if not isinstance(qa_items, list):
                continue
            section_counts[section] += len(qa_items)

            for index, qa in enumerate(qa_items, start=1):
                if not isinstance(qa, dict):
                    continue
                source_modality = qa.get("source_modality", "unknown")
                is_outlier = bool(qa.get("is_outlier", False))
                drop_reason = str(qa.get("drop_reason", "unknown"))
                modality_counts[str(source_modality)] += 1
                outlier_counts["outlier" if is_outlier else "not_outlier"] += 1
                drop_reason_counts[drop_reason] += 1

                rows.append(
                    {
                        "sample_key": sample_key,
                        "section": section,
                        "qa_index": index,
                        "source_modality": source_modality,
                        "annotation_key": qa.get("annotation_key", ""),
                        "fusion_score": qa.get("fusion_score"),
                        "support_score": qa.get("support_score"),
                        "modality_reliability": qa.get("modality_reliability"),
                        "is_outlier": is_outlier,
                        "drop_reason": drop_reason,
                        "question": qa.get("question", ""),
                        "answer": qa.get("answer", ""),
                    }
                )

    dropped_reason_counts: Counter[str] = Counter()
    dropped_examples_by_section: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample_key, sample_diag in diagnostics.items():
        sections = sample_diag.get("sections", {}) if isinstance(sample_diag, dict) else {}
        if not isinstance(sections, dict):
            continue
        for section, section_diag in sections.items():
            if not isinstance(section_diag, dict):
                continue
            dropped_examples = section_diag.get("dropped_examples", [])
            if not isinstance(dropped_examples, list):
                continue
            for example in dropped_examples:
                if not isinstance(example, dict):
                    continue
                reason = str(example.get("reason", "unknown"))
                dropped_reason_counts[reason] += 1
                enriched = {"sample_key": sample_key, "section": section, **example}
                if len(dropped_examples_by_section[section]) < 50:
                    dropped_examples_by_section[section].append(enriched)

    report = {
        "meta": {
            "fused_results": str(fused_results_path),
            "diagnostics": str(diagnostics_path),
            "total_samples": len(fused),
            "total_qa_rows": len(rows),
        },
        "summary": {
            "section_counts": dict(section_counts),
            "source_modality_counts": dict(modality_counts),
            "outlier_counts": dict(outlier_counts),
            "selected_drop_reason_counts": dict(drop_reason_counts),
            "dropped_reason_counts_from_diagnostics": dict(dropped_reason_counts),
        },
        "dropped_examples_by_section": dict(dropped_examples_by_section),
        "rows": rows,
    }

    with open(output_json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    fieldnames = [
        "sample_key",
        "section",
        "qa_index",
        "source_modality",
        "annotation_key",
        "fusion_score",
        "support_score",
        "modality_reliability",
        "is_outlier",
        "drop_reason",
        "question",
        "answer",
    ]
    with open(output_csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze section_evidence_qas and export fusion QA stats.")
    parser.add_argument("--fused", default="fused_qa_results.json", help="Path to fused results JSON")
    parser.add_argument("--diagnostics", default="fusion_diagnostics.json", help="Path to diagnostics JSON")
    parser.add_argument("--out-json", default="fusion_section_qa_stats.json", help="Output aggregated stats JSON")
    parser.add_argument("--out-csv", default="fusion_section_qa_rows.csv", help="Output flattened QA rows CSV")
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
    print("source_modality_counts", report["summary"]["source_modality_counts"])
    print("outlier_counts", report["summary"]["outlier_counts"])


if __name__ == "__main__":
    main()
