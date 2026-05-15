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


def _qa_sections(qa: dict[str, Any], fallback_section: str) -> list[str]:
    sections = qa.get("sections")
    if isinstance(sections, list) and sections:
        return [str(section) for section in sections]
    return [fallback_section]


def build_reports(
    fused_results_path: Path,
    diagnostics_path: Path,
    output_json_path: Path,
    output_csv_path: Path,
) -> dict[str, Any]:
    fused = _load_json(fused_results_path)
    diagnostics = _load_json(diagnostics_path) if diagnostics_path.exists() else {}

    rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    section_counts: Counter[str] = Counter()
    review_section_counts: Counter[str] = Counter()
    multi_section_counts: Counter[str] = Counter()
    review_multi_section_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    review_category_counts: Counter[str] = Counter()
    modality_counts: Counter[str] = Counter()
    review_modality_counts: Counter[str] = Counter()
    outlier_counts: Counter[str] = Counter()
    selection_reason_counts: Counter[str] = Counter()
    reliability_gate_counts: Counter[str] = Counter()
    review_counts: Counter[str] = Counter()
    review_priority_counts: Counter[str] = Counter()

    for sample_key, sample_payload in fused.items():
        selected_qas = sample_payload.get("selected_reliable_qas", [])
        review_qas = sample_payload.get("review_recommended_qas", [])
        if not isinstance(selected_qas, list):
            selected_qas = []
        if not isinstance(review_qas, list):
            review_qas = []

        for status, qa_items in (("selected", selected_qas), ("review", review_qas)):
            target_rows = rows if status == "selected" else review_rows
            for index, qa in enumerate(qa_items, start=1):
                if not isinstance(qa, dict):
                    continue
                section = str(qa.get("section", "unknown"))
                primary_section = str(qa.get("primary_section", section))
                sections = _qa_sections(qa, primary_section)
                category = str(qa.get("category", qa.get("annotation_key", "unknown")))
                source_modality = qa.get("source_modality", "unknown")
                is_outlier = bool(qa.get("is_outlier", False))
                selection_reason = str(qa.get("selection_reason", "unknown"))
                reliability_gate = str(qa.get("reliability_gate", "unknown"))
                review_recommended = bool(qa.get("review_recommended", False))
                review_priority = str(qa.get("review_priority", "none"))
                support_explanation = qa.get("support_explanation", {})
                if not isinstance(support_explanation, dict):
                    support_explanation = {}
                if status == "selected":
                    section_counts[primary_section] += 1
                    multi_section_counts.update(sections)
                    category_counts[category] += 1
                    modality_counts[str(source_modality)] += 1
                    outlier_counts["outlier" if is_outlier else "not_outlier"] += 1
                    selection_reason_counts[selection_reason] += 1
                    reliability_gate_counts[reliability_gate] += 1
                    review_counts["review_recommended" if review_recommended else "no_review_flag"] += 1
                else:
                    review_section_counts[primary_section] += 1
                    review_multi_section_counts.update(sections)
                    review_category_counts[category] += 1
                    review_modality_counts[str(source_modality)] += 1
                    review_priority_counts[review_priority] += 1

                target_rows.append(
                    {
                        "sample_key": sample_key,
                        "status": status,
                        "row_index": index,
                        "section": primary_section,
                        "primary_section": primary_section,
                        "sections": ";".join(sections),
                        "category": category,
                        "source_modality": source_modality,
                        "annotation_key": qa.get("annotation_key", ""),
                        "qa_index": qa.get("qa_index"),
                        "qa_source": qa.get("qa_source", ""),
                        "fusion_score": qa.get("fusion_score"),
                        "support_score": qa.get("support_score"),
                        "lexical_support_score": qa.get("lexical_support_score"),
                        "semantic_support_score": qa.get("semantic_support_score"),
                        "semantic_support_raw_cosine": qa.get("semantic_support_raw_cosine"),
                        "supporting_modality_count": qa.get("supporting_modality_count"),
                        "supporting_modalities": ";".join(qa.get("supporting_modalities", [])) if isinstance(qa.get("supporting_modalities"), list) else "",
                        "modality_reliability": qa.get("modality_reliability"),
                        "modality_weight_profile": qa.get("modality_weight_profile", ""),
                        "requires_lexical_support": qa.get("requires_lexical_support", ""),
                        "is_outlier": is_outlier,
                        "selection_reason": selection_reason,
                        "reliability_gate": reliability_gate,
                        "review_recommended": review_recommended,
                        "review_priority": review_priority,
                        "review_reasons": ";".join(qa.get("review_reasons", [])) if isinstance(qa.get("review_reasons"), list) else "",
                        "support_level": support_explanation.get("support_level", ""),
                        "best_support_modality": support_explanation.get("best_support_modality", ""),
                        "best_support_score": support_explanation.get("best_support_score", ""),
                        "best_match_modality": support_explanation.get("best_match_modality", ""),
                        "best_match_category": support_explanation.get("best_match_category", ""),
                        "question": qa.get("question", ""),
                        "answer": qa.get("answer", ""),
                    }
                )
    dropped_reason_counts: Counter[str] = Counter()
    review_reason_counts: Counter[str] = Counter()
    dropped_examples_by_reason: dict[str, list[dict[str, Any]]] = defaultdict(list)
    qa_selection_by_sample: dict[str, Any] = {}
    modality_support_by_sample: dict[str, Any] = {}
    for sample_key, sample_diag in diagnostics.items():
        qa_selection = sample_diag.get("qa_selection", {}) if isinstance(sample_diag, dict) else {}
        if not isinstance(qa_selection, dict):
            continue
        qa_selection_by_sample[sample_key] = {
            key: qa_selection.get(key)
            for key in (
                "total_candidates",
                "passed_score_and_outlier_filters",
                "selected_count",
                "review_count",
                "dropped_count",
                "drop_reason_counts",
                "review_reason_counts",
                "support_scoring",
                "review_priority_counts",
            )
        }
        modality_support_by_sample[sample_key] = qa_selection.get("modality_support_summary", {})
        for reason, count in (qa_selection.get("review_reason_counts", {}) or {}).items():
            review_reason_counts[str(reason)] += int(count)
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
            "total_review_rows": len(review_rows),
        },
        "summary": {
            "section_counts": dict(section_counts),
            "review_section_counts": dict(review_section_counts),
            "multi_section_counts": dict(multi_section_counts),
            "review_multi_section_counts": dict(review_multi_section_counts),
            "category_counts": dict(category_counts),
            "review_category_counts": dict(review_category_counts),
            "source_modality_counts": dict(modality_counts),
            "review_source_modality_counts": dict(review_modality_counts),
            "outlier_counts": dict(outlier_counts),
            "selection_reason_counts": dict(selection_reason_counts),
            "reliability_gate_counts": dict(reliability_gate_counts),
            "review_recommended_counts": dict(review_counts),
            "review_priority_counts": dict(review_priority_counts),
            "dropped_reason_counts_from_diagnostics": dict(dropped_reason_counts),
            "review_reason_counts_from_diagnostics": dict(review_reason_counts),
        },
        "qa_selection_by_sample": qa_selection_by_sample,
        "modality_support_by_sample": modality_support_by_sample,
        "dropped_examples_by_reason": dict(dropped_examples_by_reason),
        "rows": rows,
        "review_rows": review_rows,
    }

    with open(output_json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    fieldnames = [
        "sample_key",
        "status",
        "row_index",
        "section",
        "primary_section",
        "sections",
        "category",
        "qa_source",
        "qa_index",
        "source_modality",
        "annotation_key",
        "fusion_score",
        "support_score",
        "lexical_support_score",
        "semantic_support_score",
        "semantic_support_raw_cosine",
        "supporting_modality_count",
        "supporting_modalities",
        "modality_reliability",
        "modality_weight_profile",
        "requires_lexical_support",
        "is_outlier",
        "selection_reason",
        "reliability_gate",
        "review_recommended",
        "review_priority",
        "review_reasons",
        "support_level",
        "best_support_modality",
        "best_support_score",
        "best_match_modality",
        "best_match_category",
        "question",
        "answer",
    ]
    with open(output_csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows + review_rows)

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
    print("total_review_rows", report["meta"]["total_review_rows"])
    print("section_counts", report["summary"]["section_counts"])
    print("multi_section_counts", report["summary"]["multi_section_counts"])
    print("category_counts", report["summary"]["category_counts"])
    print("source_modality_counts", report["summary"]["source_modality_counts"])
    print("outlier_counts", report["summary"]["outlier_counts"])
    print("reliability_gate_counts", report["summary"]["reliability_gate_counts"])
    print("review_recommended_counts", report["summary"]["review_recommended_counts"])
    print("review_priority_counts", report["summary"]["review_priority_counts"])
    print("dropped_reason_counts", report["summary"]["dropped_reason_counts_from_diagnostics"])
    print("review_reason_counts", report["summary"]["review_reason_counts_from_diagnostics"])


if __name__ == "__main__":
    main()
