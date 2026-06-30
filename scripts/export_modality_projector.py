#!/usr/bin/env python3
"""Export modality-score question vectors for TensorFlow Projector."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(
    "outputs/evaluations/vlm_cross_modality_8frame_8b_full_qwen/"
    "metric_modality_tables/composite_modality_scores.csv"
)
DEFAULT_OUTPUT = Path(
    "outputs/evaluations/vlm_cross_modality_8frame_8b_full_qwen/"
    "metric_modality_tables/projector"
)
DEFAULT_SIMPLIFIED_OUTPUT = Path(
    "outputs/evaluations/vlm_cross_modality_8frame_8b_full_qwen/"
    "metric_modality_tables/projector_best_group"
)
DEFAULT_UNIQUE_OUTPUT = Path(
    "outputs/evaluations/vlm_cross_modality_8frame_8b_full_qwen/"
    "metric_modality_tables/projector_unique_best"
)
MODALITIES = ("rgb", "ir", "event", "depth")
FEATURE_SUFFIXES = (
    "composite_score",
    "judge_score",
    "task_aware_score",
    "text_metric_mean",
    "token_f1",
)
METADATA_FIELDS = (
    "source_qa_id",
    "question",
    "source_modality",
    "source_section",
    "ground_truth_answer",
    "best_input_modalities",
    "is_tie",
)
SIMPLIFIED_METADATA_FIELDS = METADATA_FIELDS + ("best_group",)


def sanitize_tsv(value: Any) -> str:
    return str(value or "").replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def optional_float_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "0"
    try:
        return f"{float(text):.10g}"
    except ValueError:
        return "0"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def best_group(best_input_modalities: str) -> str:
    modalities = [item for item in best_input_modalities.split(";") if item]
    if len(modalities) == 1:
        return modalities[0]
    if set(modalities) == set(MODALITIES):
        return "all_modalities"
    return "multi_best"


def feature_fields() -> list[str]:
    return [f"{modality}_{suffix}" for modality in MODALITIES for suffix in FEATURE_SUFFIXES]


def write_vectors(path: Path, rows: list[dict[str, str]]) -> None:
    fields = feature_fields()
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write("\t".join(optional_float_text(row.get(field)) for field in fields) + "\n")


def write_metadata(path: Path, rows: list[dict[str, str]], simplified: bool = False) -> None:
    fields = list(SIMPLIFIED_METADATA_FIELDS if simplified else METADATA_FIELDS)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(fields) + "\n")
        for row in rows:
            if simplified:
                row = dict(row)
                row["best_group"] = best_group(row.get("best_input_modalities", ""))
            handle.write("\t".join(sanitize_tsv(row.get(field)) for field in fields) + "\n")


def write_readme(
    path: Path,
    input_path: Path,
    rows: list[dict[str, str]],
    simplified: bool = False,
    unique_only: bool = False,
) -> None:
    color_fields = [
        "`best_group`" if simplified else "`best_input_modalities`",
        "`best_input_modalities`" if simplified else "`source_modality`",
        "`source_modality`" if simplified else "`source_section`",
        "`source_section`" if simplified else "`is_tie`",
        "`is_tie`" if simplified else "",
    ]
    color_fields = [field for field in color_fields if field]
    lines = [
        "# TensorFlow Projector Export",
        "",
        "Load these files at https://projector.tensorflow.org/ using `Load data from your computer`.",
        "",
        "- Step 1 vectors: `projector_vectors.tsv`",
        "- Step 2 metadata: `projector_metadata.tsv`",
        "",
        f"Source CSV: `{input_path.as_posix()}`",
        f"Rows/questions: {len(rows)}",
        f"Vector dimensions: {len(feature_fields())}",
        "",
        "Vector feature order:",
        "",
    ]
    lines.extend(f"- `{field}`" for field in feature_fields())
    lines.extend(
        [
            "",
            "Suggested coloring fields in Projector:",
            "",
        ]
    )
    lines.extend(f"- {field}" for field in color_fields)
    if simplified:
        lines.extend(
            [
                "",
                "`best_group` simplifies best-input combinations into `rgb`, `ir`, `event`, `depth`, "
                "`multi_best`, or `all_modalities` for cleaner coloring.",
            ]
        )
    if unique_only:
        lines.extend(
            [
                "",
                "This export includes only rows with `is_tie = False`, so `best_input_modalities` has four possible values: `rgb`, `ir`, `event`, and `depth`.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_projector(
    output: Path,
    input_path: Path,
    rows: list[dict[str, str]],
    simplified: bool,
    unique_only: bool = False,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_vectors(output / "projector_vectors.tsv", rows)
    write_metadata(output / "projector_metadata.tsv", rows, simplified=simplified)
    write_readme(output / "README.md", input_path, rows, simplified=simplified, unique_only=unique_only)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--simplified-output", type=Path, default=DEFAULT_SIMPLIFIED_OUTPUT)
    parser.add_argument("--unique-output", type=Path, default=DEFAULT_UNIQUE_OUTPUT)
    parser.add_argument(
        "--skip-simplified",
        action="store_true",
        help="Only export the full best_input_modalities metadata version.",
    )
    parser.add_argument(
        "--skip-unique",
        action="store_true",
        help="Do not export the is_tie=False unique-best-only version.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = load_rows(args.input)
    export_projector(args.output, args.input, rows, simplified=False)
    if not args.skip_simplified:
        export_projector(args.simplified_output, args.input, rows, simplified=True)
    if not args.skip_unique:
        unique_rows = [row for row in rows if str(row.get("is_tie", "")).lower() == "false"]
        export_projector(args.unique_output, args.input, unique_rows, simplified=False, unique_only=True)
    print(f"projector export: {args.output}")
    if not args.skip_simplified:
        print(f"simplified projector export: {args.simplified_output}")
    if not args.skip_unique:
        print(f"unique-best projector export: {args.unique_output}")


if __name__ == "__main__":
    main()
