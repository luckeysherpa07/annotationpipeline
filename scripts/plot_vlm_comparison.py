#!/usr/bin/env python3
"""Create slide-ready figures from the VLM comparison outputs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter


ROOT = Path("outputs/evaluations")
COMPARISON = ROOT / "vlm_model_size_comparison"
OUTPUT = COMPARISON / "figures"
FAMILIES = ("InternVL3.5", "Molmo2", "Qwen3-VL")
MODALITIES = ("rgb", "ir", "event", "depth")
COLORS = {
    "InternVL3.5": "#009E73",
    "Molmo2": "#D55E00",
    "Qwen3-VL": "#0072B2",
    "4B": "#79BCE8",
    "8B": "#174A7E",
}
LEXICAL_COLORS = {
    "BLEU-4": "#4E79A7",
    "ROUGE-L": "#F28E2B",
    "METEOR": "#59A14F",
}
EVALUATION_DIRS = (
    ROOT / "vlm_8frame_aligned_4b",
    ROOT / "vlm_8frame_aligned_8b",
    ROOT / "vlm_native_video_4b",
    ROOT / "vlm_native_video_8b",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)


def plot_overall() -> None:
    rows = read_csv(COMPARISON / "same_size_model_comparison.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 6.6), sharey=True)
    x = np.arange(len(FAMILIES))
    width = 0.34
    for ax, input_name, title in zip(axes, ("frame 8", "video"), ("Eight-frame input", "Native-video input")):
        for offset, size in ((-width / 2, "4B"), (width / 2, "8B")):
            values = [
                float(next(row["judge_strict_accuracy"] for row in rows
                           if row["input"] == input_name and row["size"] == size
                           and row["model_family"] == family))
                for family in FAMILIES
            ]
            bars = ax.bar(x + offset, values, width, label=size, color=COLORS[size])
            ax.bar_label(bars, labels=[f"{value:.1%}" for value in values], padding=4, fontsize=10)
        ax.set_title(title, fontsize=15, fontweight="bold", pad=14)
        ax.set_xticks(x, FAMILIES)
        ax.set_ylim(0.0, 0.65)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        style_axis(ax)
    axes[0].set_ylabel("LLM-judge strict accuracy", fontsize=12)
    axes[1].legend(frameon=False, loc="upper left", ncols=2)
    fig.suptitle("Overall QA Accuracy Across Models", fontsize=21, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "01_overall_model_comparison")


def heatmap(ax: plt.Axes, values: np.ndarray, row_labels: list[str], column_labels: list[str],
            vmin: float, vmax: float, annotate_n: np.ndarray | None = None) -> None:
    cmap = LinearSegmentedColormap.from_list("slide_blue", ("#F4F8FC", "#8FC4E8", "#0B5A8C"))
    image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(column_labels)), column_labels)
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.tick_params(length=0)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if np.isnan(value):
                text = "–"
            else:
                text = f"{value:.1%}"
                if annotate_n is not None:
                    text += f"\nN={int(annotate_n[i, j])}"
            color = "white" if not np.isnan(value) and value > (vmin + vmax) / 2 + 0.02 else "#172B3A"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)
    return image


def plot_modality_heatmap() -> None:
    rows = read_csv(COMPARISON / "modality_comparison.csv")
    configs = [(input_name, size, family) for input_name in ("frame 8", "video")
               for size in ("4B", "8B") for family in FAMILIES]
    labels = [f"{'8 frames' if inp == 'frame 8' else 'Video'} | {size} | {family}"
              for inp, size, family in configs]
    values = np.full((len(configs), len(MODALITIES)), np.nan)
    for i, (input_name, size, family) in enumerate(configs):
        for j, modality in enumerate(MODALITIES):
            row = next(row for row in rows if row["input"] == input_name and row["size"] == size
                       and row["model_family"] == family and row["modality"] == modality)
            values[i, j] = float(row["judge_strict_accuracy"])
    fig, ax = plt.subplots(figsize=(10.8, 8.2))
    image = heatmap(ax, values, labels, [item.upper() for item in MODALITIES], 0.34, 0.62)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    colorbar.set_label("Strict accuracy", rotation=270, labelpad=18)
    ax.set_title("Model Performance Across Sensor Modalities", fontsize=20, fontweight="bold", pad=18)
    fig.tight_layout()
    save(fig, "02_modality_performance_heatmap")


def plot_scaling() -> None:
    rows = read_csv(COMPARISON / "same_model_size_comparison.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 6.6), sharey=True)
    for ax, input_name, title in zip(axes, ("frame 8", "video"), ("Eight-frame input", "Native-video input")):
        for family in FAMILIES:
            row = next(row for row in rows if row["input"] == input_name and row["model_family"] == family)
            values = [float(row["judge_strict_accuracy_4b"]), float(row["judge_strict_accuracy_8b"])]
            delta = float(row["judge_strict_accuracy_delta_8b_minus_4b"])
            ax.plot((0, 1), values, marker="o", markersize=9, linewidth=2.5,
                    color=COLORS[family], label=family)
            ax.text(1.04, values[1], f"{delta * 100:+.1f} pp", va="center", fontsize=10, color=COLORS[family])
        ax.set_xlim(-0.15, 1.42)
        ax.set_xticks((0, 1), ("4B", "8B"))
        ax.set_ylim(0.43, 0.60)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_title(title, fontsize=15, fontweight="bold", pad=14)
        style_axis(ax)
    axes[0].set_ylabel("LLM-judge strict accuracy", fontsize=12)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Effect of Scaling from 4B to 8B", fontsize=21, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "03_model_scaling_effect")


def plot_section_heatmap() -> None:
    rows = [row for row in read_csv(COMPARISON / "modality_section_comparison.csv")
            if row["input"] == "frame 8" and row["size"] == "8B" and row["model_family"] == "Qwen3-VL"]
    common_sections = (
        "action", "counting", "dynamic_counting", "dynamic_recognition", "navigation",
        "non_common", "object_recognition", "scene_sequence", "spatial_reasoning",
    )
    labels = ("Action", "Counting", "Dynamic\ncounting", "Dynamic\nrecognition", "Navigation",
              "Non-common", "Object\nrecognition", "Scene\nsequence", "Spatial\nreasoning")
    values = np.full((len(MODALITIES), len(common_sections)), np.nan)
    counts = np.zeros_like(values)
    for i, modality in enumerate(MODALITIES):
        for j, section in enumerate(common_sections):
            match = next((row for row in rows if row["modality"] == modality and row["section"] == section), None)
            if match:
                values[i, j] = float(match["judge_strict_accuracy"])
                counts[i, j] = int(match["total"])
    fig, ax = plt.subplots(figsize=(13.33, 5.8))
    image = heatmap(ax, values, [item.upper() for item in MODALITIES], list(labels), 0.15, 1.0, counts)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.025)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    colorbar.set_label("Strict accuracy", rotation=270, labelpad=18)
    ax.set_title("Task-Specific Performance Across Modalities", fontsize=20, fontweight="bold", pad=18)
    ax.set_xlabel("Qwen3-VL-8B, eight-frame input", fontsize=11, labelpad=15, color="#555555")
    fig.tight_layout()
    save(fig, "04_modality_section_heatmap")


def plot_efficiency() -> None:
    rows: list[dict[str, str]] = []
    for directory in EVALUATION_DIRS:
        rows.extend(read_csv(directory / "summary.csv"))
    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    markers = {"frame": "o", "video": "s"}
    for row in rows:
        family = {"internvl": "InternVL3.5", "molmo2": "Molmo2", "qwen_vl": "Qwen3-VL"}[row["provider"]]
        size = "8B" if "8B" in row["model_name"] else "4B"
        x = float(row["latency_mean_seconds"])
        y = float(row["judge_strict_accuracy"])
        ax.scatter(x, y, s=95 if size == "8B" else 65, marker=markers[row["input_type"]],
                   color=COLORS[family], edgecolor="white", linewidth=0.8, zorder=3)
        offset = (6, 5)
        if row["input_type"] == "video" and family == "InternVL3.5":
            offset = (-94, 5) if size == "4B" else (8, 5)
        ax.annotate(f"{family} {size}", (x, y), xytext=offset, textcoords="offset points", fontsize=9)
    ax.set_xlabel("Mean latency per QA item (seconds)", fontsize=12)
    ax.set_ylabel("LLM-judge strict accuracy", fontsize=12)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0.43, 0.60)
    style_axis(ax)
    handles = [
        plt.Line2D([], [], color="#555555", marker="o", linestyle="None", markersize=8, label="8 frames"),
        plt.Line2D([], [], color="#555555", marker="s", linestyle="None", markersize=8, label="Native video"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    ax.set_title("Accuracy–Efficiency Trade-off", fontsize=20, fontweight="bold", pad=18)
    fig.text(0.99, 0.01, "Native-video points include model-specific adapter costs.",
             ha="right", fontsize=9, color="#666666")
    fig.tight_layout()
    save(fig, "05_accuracy_efficiency_tradeoff")


def _short_model_label(model: str, size: str) -> str:
    if "InternVL" in model:
        return f"InternVL {size}"
    if "Molmo2" in model:
        return f"Molmo2 {size}"
    if "Qwen3" in model:
        return f"Qwen3-VL {size}"
    return f"{model} {size}"


def plot_lexical_diagnostics() -> None:
    rows = [
        row for row in read_csv(COMPARISON / "same_size_model_comparison.csv")
        if row["input"] == "frame 8"
    ]
    labels: list[str] = []
    values = {"BLEU-4": [], "ROUGE-L": [], "METEOR": []}
    for size in ("4B", "8B"):
        for family in FAMILIES:
            row = next(row for row in rows if row["size"] == size and row["model_family"] == family)
            labels.append(_short_model_label(row["model"], size))
            values["BLEU-4"].append(float(row["bleu_4"]))
            values["ROUGE-L"].append(float(row["rouge_l_f1"]))
            values["METEOR"].append(float(row["meteor"]))

    y = np.arange(len(labels))
    bar_height = 0.22
    offsets = (-bar_height, 0.0, bar_height)

    fig, ax = plt.subplots(figsize=(11.6, 6.8))
    for (metric, metric_values), offset in zip(values.items(), offsets):
        bars = ax.barh(
            y + offset,
            metric_values,
            height=bar_height,
            label=metric,
            color=LEXICAL_COLORS[metric],
        )
        ax.bar_label(bars, labels=[f"{value:.1%}" for value in metric_values], padding=3, fontsize=9)

    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 0.55)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Lexical similarity score", fontsize=12)
    ax.set_title("Supplementary Lexical Similarity Metrics", fontsize=20, fontweight="bold", pad=18)
    ax.legend(frameon=False, loc="lower right", ncols=3)
    style_axis(ax)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.8, alpha=0.7)
    ax.grid(axis="y", visible=False)
    fig.text(
        0.99,
        0.01,
        "Fixed eight-frame input only. Lexical overlap is diagnostic and does not fully measure semantic correctness.",
        ha="right",
        fontsize=9,
        color="#666666",
    )
    fig.tight_layout()
    save(fig, "06_lexical_diagnostics_8frame")


def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelcolor": "#172B3A",
        "xtick.color": "#172B3A",
        "ytick.color": "#172B3A",
        "text.color": "#172B3A",
    })
    plot_overall()
    plot_modality_heatmap()
    plot_scaling()
    plot_section_heatmap()
    plot_efficiency()
    plot_lexical_diagnostics()


if __name__ == "__main__":
    main()
