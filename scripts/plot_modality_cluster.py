#!/usr/bin/env python3
"""Plot clustered per-question modality score heatmaps."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.cluster.hierarchy import leaves_list, linkage


DEFAULT_INPUT = Path(
    "outputs/evaluations/vlm_cross_modality_8frame_8b_full_qwen/"
    "metric_modality_tables/composite_modality_scores.csv"
)
DEFAULT_OUTPUT = Path(
    "outputs/evaluations/vlm_cross_modality_8frame_8b_full_qwen/"
    "metric_modality_tables/modality_best_cluster_heatmap.png"
)
MODALITIES = ("rgb", "ir", "event", "depth")
CLUSTER_FEATURE_SUFFIXES = (
    "composite_score",
    "judge_score",
    "task_aware_score",
    "text_metric_mean",
    "token_f1",
)
HEATMAP_SUFFIX = "composite_score"
BASE_COLORS = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#B279A2",
    "#E45756",
    "#72B7B2",
    "#EECA3B",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
    "#8CD17D",
    "#B6992D",
    "#499894",
    "#D37295",
    "#A0CBE8",
    "#FABFD2",
]


def optional_float(value: Any) -> float:
    try:
        if value is None or str(value).strip() == "":
            return 0.0
        return float(value)
    except ValueError:
        return 0.0


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def feature_vector(row: dict[str, str]) -> list[float]:
    values: list[float] = []
    for modality in MODALITIES:
        for suffix in CLUSTER_FEATURE_SUFFIXES:
            values.append(optional_float(row.get(f"{modality}_{suffix}")))
    return values


def heatmap_vector(row: dict[str, str]) -> list[float]:
    return [optional_float(row.get(f"{modality}_{HEATMAP_SUFFIX}")) for modality in MODALITIES]


def best_group_order(rows: list[dict[str, str]]) -> list[str]:
    counts = Counter(row["best_input_modalities"] for row in rows)
    preferred = ["rgb", "ir", "event", "depth"]
    groups = list(counts)
    single = [group for group in preferred if group in counts]
    multi = sorted(
        [group for group in groups if group not in single],
        key=lambda group: (-counts[group], group.count(";"), group),
    )
    return single + multi


def clustered_indices(rows: list[dict[str, str]]) -> list[int]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["best_input_modalities"]].append(index)

    ordered: list[int] = []
    features = np.array([feature_vector(row) for row in rows], dtype=float)
    for group in best_group_order(rows):
        indices = grouped[group]
        if len(indices) <= 2:
            ordered.extend(indices)
            continue
        group_features = features[indices]
        # Cluster only inside each best-modality group so the visual keeps same winners together.
        leaves = leaves_list(linkage(group_features, method="average", metric="euclidean"))
        ordered.extend(indices[int(leaf)] for leaf in leaves)
    return ordered


def color_map_for_groups(rows: list[dict[str, str]]) -> tuple[dict[str, int], ListedColormap]:
    groups = best_group_order(rows)
    group_to_id = {group: index for index, group in enumerate(groups)}
    repeats = (len(groups) // len(BASE_COLORS)) + 1
    colors = (BASE_COLORS * repeats)[: len(groups)]
    return group_to_id, ListedColormap(colors)


def plot_heatmap(rows: list[dict[str, str]], output: Path) -> None:
    if not rows:
        raise RuntimeError("No rows loaded for plotting.")
    ordered_indices = clustered_indices(rows)
    ordered_rows = [rows[index] for index in ordered_indices]
    heatmap = np.array([heatmap_vector(row) for row in ordered_rows], dtype=float)
    group_to_id, group_cmap = color_map_for_groups(rows)
    group_ids = np.array(
        [[group_to_id[row["best_input_modalities"]]] for row in ordered_rows],
        dtype=float,
    )

    group_counts = Counter(row["best_input_modalities"] for row in ordered_rows)
    group_order = best_group_order(rows)
    boundary_positions: list[int] = []
    running = 0
    for group in group_order:
        running += group_counts[group]
        boundary_positions.append(running - 0.5)

    height = max(10, min(28, len(rows) / 220))
    fig = plt.figure(figsize=(13, height), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[0.35, 4.0, 1.7])
    group_ax = fig.add_subplot(grid[0, 0])
    heat_ax = fig.add_subplot(grid[0, 1])
    legend_ax = fig.add_subplot(grid[0, 2])

    group_ax.imshow(group_ids, aspect="auto", interpolation="nearest", cmap=group_cmap)
    group_ax.set_xticks([0])
    group_ax.set_xticklabels(["best"])
    group_ax.set_yticks([])
    group_ax.set_title("Best\nmodality", fontsize=9)

    image = heat_ax.imshow(heatmap, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=1)
    heat_ax.set_xticks(range(len(MODALITIES)))
    heat_ax.set_xticklabels(MODALITIES)
    heat_ax.set_yticks([])
    heat_ax.set_title("Composite scores per question, grouped by best input modality")
    heat_ax.set_xlabel("Input modality")
    for boundary in boundary_positions[:-1]:
        group_ax.axhline(boundary, color="white", linewidth=0.25, alpha=0.7)
        heat_ax.axhline(boundary, color="white", linewidth=0.25, alpha=0.7)
    fig.colorbar(image, ax=heat_ax, fraction=0.025, pad=0.02, label="Composite score")

    legend_ax.axis("off")
    legend_lines = ["Best modality groups", ""]
    for group in group_order:
        legend_lines.append(f"{group}: {group_counts[group]}")
    legend_ax.text(0, 1, "\n".join(legend_lines), va="top", ha="left", fontsize=8)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = load_rows(args.input)
    plot_heatmap(rows, args.output)
    print(f"cluster heatmap: {args.output}")


if __name__ == "__main__":
    main()
