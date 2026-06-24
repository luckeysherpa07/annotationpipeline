#!/usr/bin/env python3
"""Generate presentation charts for the 30-frame aligned 4B benchmark."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "outputs/evaluations/vlm_30frame_aligned_4b"
OUT_DIR = ROOT / "outputs/presentation"

WIDTH = 1920
HEIGHT = 1080
COLORS = {
    "Qwen3-VL-4B": "#2563EB",
    "InternVL2.5-4B": "#F59E0B",
    "Molmo2-4B": "#10B981",
}


def model_label(model_name: str) -> str:
    if "Qwen3-VL" in model_name:
        return "Qwen3-VL-4B"
    if "InternVL2_5" in model_name:
        return "InternVL2.5-4B"
    if "Molmo2" in model_name:
        return "Molmo2-4B"
    return model_name


def svg_start(title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="1920" height="1080" fill="#F7F9FC"/>',
        '<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#172033}</style>',
        f'<text x="110" y="105" font-size="52" font-weight="700">{title}</text>',
        f'<text x="110" y="155" font-size="25" fill="#64748B">{subtitle}</text>',
    ]


def add_legend(parts: list[str], models: list[str], y: int = 205) -> None:
    x = 112
    for model in models:
        parts.append(f'<rect x="{x}" y="{y - 19}" width="28" height="28" rx="6" fill="{COLORS[model]}"/>')
        parts.append(f'<text x="{x + 40}" y="{y + 3}" font-size="24">{model}</text>')
        x += 290


def add_axes(parts: list[str], left: int, top: int, width: int, height: int, maximum: float) -> None:
    for tick in range(0, 7):
        value = maximum * tick / 6
        y = top + height - height * tick / 6
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + width}" y2="{y:.1f}" stroke="#D9E1EC" stroke-width="2"/>')
        parts.append(f'<text x="{left - 22}" y="{y + 8:.1f}" text-anchor="end" font-size="22" fill="#64748B">{value:.1f}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" stroke="#94A3B8" stroke-width="3"/>')
    parts.append(f'<line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" stroke="#94A3B8" stroke-width="3"/>')


def make_main_results() -> Path:
    with (EVAL_DIR / "answer_metrics_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_model = {model_label(row["model_name"]): row for row in rows}
    models = ["Qwen3-VL-4B", "InternVL2.5-4B", "Molmo2-4B"]
    metrics = [
        ("BLEU-4", "bleu_4"),
        ("ROUGE-L", "rouge_l_f1"),
        ("METEOR", "meteor"),
        ("Judge strict", "judge_strict_accuracy"),
    ]

    parts = svg_start(
        "30-Frame Aligned 4B Benchmark — Main Results",
        "Automatic answer metrics and LLM-as-a-judge accuracy · higher is better",
    )
    add_legend(parts, models)
    left, top, chart_w, chart_h = 150, 285, 1620, 610
    maximum = 0.7
    add_axes(parts, left, top, chart_w, chart_h, maximum)

    group_w = chart_w / len(metrics)
    bar_w = 72
    gap = 18
    for group_index, (metric_label, field) in enumerate(metrics):
        center = left + group_w * (group_index + 0.5)
        start_x = center - (3 * bar_w + 2 * gap) / 2
        for model_index, model in enumerate(models):
            value = float(by_model[model][field])
            bar_h = chart_h * value / maximum
            x = start_x + model_index * (bar_w + gap)
            y = top + chart_h - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" rx="10" fill="{COLORS[model]}"/>')
            parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 13:.1f}" text-anchor="middle" font-size="21" font-weight="700">{value:.3f}</text>')
        parts.append(f'<text x="{center:.1f}" y="{top + chart_h + 48}" text-anchor="middle" font-size="27" font-weight="600">{metric_label}</text>')

    parts.extend([
        '<rect x="1235" y="177" width="535" height="82" rx="18" fill="#E8F0FF"/>',
        '<text x="1502" y="212" text-anchor="middle" font-size="23" font-weight="700" fill="#1D4ED8">Best overall: Qwen3-VL-4B</text>',
        '<text x="1502" y="243" text-anchor="middle" font-size="19" fill="#475569">Highest BLEU, ROUGE-L, and judge accuracy</text>',
        '</svg>',
    ])
    output = OUT_DIR / "vlm_30frame_main_results.svg"
    output.write_text("\n".join(parts), encoding="utf-8")
    return output


def make_modality_results() -> Path:
    with (EVAL_DIR / "answer_metrics_modality_scores.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    values: dict[tuple[str, str], float] = {}
    for row in rows:
        model_key = row["model_key"]
        if model_key.startswith("qwen_vl:"):
            model = "Qwen3-VL-4B"
        elif model_key.startswith("internvl:"):
            model = "InternVL2.5-4B"
        else:
            model = "Molmo2-4B"
        values[(model, row["modality"])] = float(row["judge_strict_accuracy"])

    models = ["Qwen3-VL-4B", "InternVL2.5-4B", "Molmo2-4B"]
    modalities = [("RGB", "rgb"), ("IR", "ir"), ("Event", "event"), ("Depth", "depth")]
    parts = svg_start(
        "Performance by Sensor Modality",
        "Strict LLM-judge accuracy with 30 aligned frames · higher is better",
    )
    add_legend(parts, models)
    left, top, chart_w, chart_h = 150, 285, 1620, 610
    maximum = 0.7
    add_axes(parts, left, top, chart_w, chart_h, maximum)

    group_w = chart_w / len(modalities)
    ir_x = left + group_w
    parts.append(f'<rect x="{ir_x + 18:.1f}" y="{top}" width="{group_w - 36:.1f}" height="{chart_h}" rx="18" fill="#FEF3C7" opacity="0.48"/>')
    bar_w = 72
    gap = 18
    for group_index, (modality_label, modality_key) in enumerate(modalities):
        center = left + group_w * (group_index + 0.5)
        start_x = center - (3 * bar_w + 2 * gap) / 2
        for model_index, model in enumerate(models):
            value = values[(model, modality_key)]
            bar_h = chart_h * value / maximum
            x = start_x + model_index * (bar_w + gap)
            y = top + chart_h - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" rx="10" fill="{COLORS[model]}"/>')
            parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 13:.1f}" text-anchor="middle" font-size="21" font-weight="700">{value:.3f}</text>')
        parts.append(f'<text x="{center:.1f}" y="{top + chart_h + 48}" text-anchor="middle" font-size="28" font-weight="600">{modality_label}</text>')

    parts.extend([
        '<rect x="1050" y="177" width="720" height="82" rx="18" fill="#FFF4D6"/>',
        '<text x="1410" y="212" text-anchor="middle" font-size="23" font-weight="700" fill="#92400E">IR is the strongest modality for all three models</text>',
        '<text x="1410" y="243" text-anchor="middle" font-size="19" fill="#475569">Depth remains the hardest modality for Qwen and InternVL</text>',
        '</svg>',
    ])
    output = OUT_DIR / "vlm_30frame_modality_performance.svg"
    output.write_text("\n".join(parts), encoding="utf-8")
    return output


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in (make_main_results(), make_modality_results()):
        print(path)


if __name__ == "__main__":
    main()
