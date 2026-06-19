#!/usr/bin/env python3
"""Generate a QA type distribution SVG figure from aligned QA records.

The figure is intentionally dependency-free so it can run in lightweight
environments without matplotlib. It produces a report-ready SVG with:

- visual modality pie chart
- separate RGB, IR, Event, and Depth QA type bars
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from html import escape
from pathlib import Path


VISUAL_MODALITIES = ("rgb", "ir", "event", "depth")
MODALITY_LABELS = {
    "rgb": "RGB",
    "ir": "IR",
    "event": "Event",
    "depth": "Depth",
}

PIE_COLORS = {
    "rgb": "#8ecae6",
    "ir": "#ffdd8a",
    "event": "#f3cfda",
    "depth": "#b9dfbf",
}

SECTION_LABELS = {
    "object_recognition": "object recognition",
    "spatial_reasoning": "spatial reasoning",
    "text_recognition": "text recognition",
    "scene_sequence": "scene sequence",
    "light_recognition": "lighting recognition",
    "light_change": "lighting change",
    "counting": "counting static",
    "dynamic_counting": "counting dynamic",
    "dynamic_recognition": "dynamic recognition",
    "action": "action recognition",
    "navigation": "navigation",
    "non_common": "non common",
}


def load_items(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and isinstance(data.get("valid_qa"), list):
        return data["valid_qa"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Could not find QA list in {path}")


def normalize_section(section: str) -> str:
    for prefix in ("event_", "depth_"):
        if section.startswith(prefix):
            return section[len(prefix) :]
    return section


def count_by_section(items: list[dict], modalities: set[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for item in items:
        if item.get("modality") in modalities:
            counts[normalize_section(str(item.get("section", "")))] += 1
    return counts


def text(x: float, y: float, value: str, size: int = 18, anchor: str = "start", weight: str = "400", rotate: int | None = None) -> str:
    transform = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate is not None else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="DejaVu Sans, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="#141414"{transform}>{escape(value)}</text>'
    )


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", rx: float = 0, opacity: float = 1.0) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'rx="{rx:.1f}" fill="{fill}" stroke="{stroke}" opacity="{opacity:.3f}" />'
    )


def bar_panel(
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    counts: Counter[str],
    fill: str,
    stroke: str,
    bar_fill: str,
    *,
    max_items: int = 12,
    compact: bool = False,
) -> str:
    parts = [rect(x, y, w, h, fill, stroke, rx=34)]
    parts.append(text(x + 24, y + 42, title, 22 if not compact else 20, weight="700"))

    items = counts.most_common(max_items)
    if not items:
        return "\n".join(parts)

    label_w = 168 if not compact else 132
    left = x + 24
    top = y + (72 if not compact else 66)
    row_h = (h - (112 if not compact else 104)) / max(len(items), 1)
    bar_x = left + label_w
    bar_max_w = w - label_w - 64
    max_count = max(value for _, value in items)

    for index, (section, value) in enumerate(items):
        cy = top + index * row_h
        label = SECTION_LABELS.get(section, section.replace("_", " "))
        size = 14 if not compact else 12
        parts.append(text(bar_x - 12, cy + row_h * 0.58, label, size, anchor="end"))
        bar_w = max(4, bar_max_w * value / max_count)
        parts.append(rect(bar_x, cy + row_h * 0.18, bar_w, row_h * 0.64, bar_fill, rx=4))
        parts.append(text(bar_x + bar_w - 8, cy + row_h * 0.58, str(value), size, anchor="end"))

    axis_y = y + h - 36
    parts.append(f'<line x1="{bar_x:.1f}" y1="{axis_y:.1f}" x2="{(bar_x + bar_max_w):.1f}" y2="{axis_y:.1f}" stroke="#222" stroke-width="1"/>')
    for tick in range(0, max_count + 1, max(1, math.ceil(max_count / 4 / 50) * 50)):
        tx = bar_x + bar_max_w * tick / max_count
        parts.append(f'<line x1="{tx:.1f}" y1="{axis_y:.1f}" x2="{tx:.1f}" y2="{axis_y + 6:.1f}" stroke="#222" stroke-width="1"/>')
        parts.append(text(tx, axis_y + 24, str(tick), 11 if compact else 12, anchor="middle"))
    return "\n".join(parts)


def pie_path(cx: float, cy: float, r: float, start: float, end: float) -> str:
    start_x = cx + r * math.cos(start)
    start_y = cy + r * math.sin(start)
    end_x = cx + r * math.cos(end)
    end_y = cy + r * math.sin(end)
    large = 1 if end - start > math.pi else 0
    return (
        f"M {cx:.3f} {cy:.3f} L {start_x:.3f} {start_y:.3f} "
        f"A {r:.3f} {r:.3f} 0 {large} 1 {end_x:.3f} {end_y:.3f} Z"
    )


def pie_chart(x: float, y: float, r: float, counts: Counter[str]) -> str:
    cx = x + r
    cy = y + r
    total = sum(counts[m] for m in VISUAL_MODALITIES)
    start = -math.pi / 2
    parts: list[str] = []

    for modality in VISUAL_MODALITIES:
        value = counts[modality]
        angle = 2 * math.pi * value / total
        end = start + angle
        parts.append(f'<path d="{pie_path(cx, cy, r, start, end)}" fill="{PIE_COLORS[modality]}" stroke="none"/>')

        mid = (start + end) / 2
        label_r = r * 0.58
        lx = cx + label_r * math.cos(mid)
        ly = cy + label_r * math.sin(mid)
        pct = 100 * value / total
        parts.append(text(lx, ly - 7, MODALITY_LABELS[modality], 24, anchor="middle", weight="700"))
        parts.append(text(lx, ly + 22, f"{value}", 17, anchor="middle"))
        parts.append(text(lx, ly + 45, f"{pct:.1f}%", 15, anchor="middle"))
        start = end

    parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="none" stroke="#ffffff" stroke-width="2"/>')
    return "\n".join(parts)


def qa_type_stacked_panel(
    x: float,
    y: float,
    w: float,
    h: float,
    modality_section_counts: dict[str, Counter[str]],
) -> str:
    totals: Counter[str] = Counter()
    for counts in modality_section_counts.values():
        totals.update(counts)

    preferred_order = [
        "scene_sequence",
        "counting",
        "dynamic_counting",
        "navigation",
        "action",
        "text_recognition",
        "spatial_reasoning",
        "dynamic_recognition",
        "object_recognition",
        "light_change",
        "light_recognition",
        "non_common",
    ]
    sections = [section for section in preferred_order if totals.get(section, 0)]
    sections.extend(section for section, _ in totals.most_common() if section not in sections)

    parts = [rect(x, y, w, h, "#f8fbff", "#a7bddb", rx=34)]
    parts.append(text(x + 28, y + 44, "QA types by modality", 24, weight="700"))

    legend_x = x + 360
    legend_y = y + 27
    for index, modality in enumerate(VISUAL_MODALITIES):
        lx = legend_x + index * 120
        parts.append(rect(lx, legend_y, 18, 18, PIE_COLORS[modality], rx=3))
        parts.append(text(lx + 28, legend_y + 15, MODALITY_LABELS[modality], 14))

    label_w = 180
    bar_x = x + 215
    top = y + 82
    row_h = (h - 130) / max(len(sections), 1)
    bar_max_w = w - 310
    max_total = max(totals.values()) if totals else 1

    for index, section in enumerate(sections):
        cy = top + index * row_h
        label = SECTION_LABELS.get(section, section.replace("_", " "))
        parts.append(text(x + label_w, cy + row_h * 0.58, label, 13, anchor="end"))

        running_x = bar_x
        section_total = totals[section]
        total_bar_w = bar_max_w * section_total / max_total
        for modality in VISUAL_MODALITIES:
            value = modality_section_counts[modality].get(section, 0)
            if not value:
                continue
            segment_w = total_bar_w * value / section_total
            parts.append(rect(running_x, cy + row_h * 0.18, segment_w, row_h * 0.62, PIE_COLORS[modality], rx=3))
            if segment_w >= 30:
                parts.append(text(running_x + segment_w / 2, cy + row_h * 0.58, str(value), 11, anchor="middle"))
            running_x += segment_w

        parts.append(text(x + w - 32, cy + row_h * 0.58, str(section_total), 12, anchor="end", weight="700"))

    axis_y = y + h - 36
    parts.append(f'<line x1="{bar_x:.1f}" y1="{axis_y:.1f}" x2="{(bar_x + bar_max_w):.1f}" y2="{axis_y:.1f}" stroke="#222" stroke-width="1"/>')
    for tick in range(0, max_total + 1, max(1, math.ceil(max_total / 5 / 100) * 100)):
        tx = bar_x + bar_max_w * tick / max_total
        parts.append(f'<line x1="{tx:.1f}" y1="{axis_y:.1f}" x2="{tx:.1f}" y2="{axis_y + 6:.1f}" stroke="#222" stroke-width="1"/>')
        parts.append(text(tx, axis_y + 24, str(tick), 11, anchor="middle"))
    parts.append(text(bar_x + bar_max_w, y + h - 8, "bar end label = total QA count", 11, anchor="end"))
    return "\n".join(parts)


def build_svg(items: list[dict]) -> str:
    modality_counts = Counter(item.get("modality") for item in items if item.get("modality") in VISUAL_MODALITIES)
    rgb_counts = count_by_section(items, {"rgb"})
    ir_counts = count_by_section(items, {"ir"})
    event_counts = count_by_section(items, {"event"})
    depth_counts = count_by_section(items, {"depth"})
    visual_total = sum(modality_counts.values())

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900">',
        rect(0, 0, 1600, 900, "#ffffff"),
        text(800, 44, "Aligned QA Distribution by Modality and Type", 28, anchor="middle", weight="700"),
        text(800, 76, f"Visual benchmark QA items: {visual_total}", 18, anchor="middle"),
        rect(52, 110, 540, 690, "#f7fbff", "#a7bddb", rx=34),
        text(88, 158, "Visual modalities", 24, weight="700"),
        pie_chart(100, 190, 220, modality_counts),
        qa_type_stacked_panel(
            640,
            110,
            908,
            690,
            {
                "rgb": rgb_counts,
                "ir": ir_counts,
                "event": event_counts,
                "depth": depth_counts,
            },
        ),
    ]

    parts.append(text(800, 850, "Generated from outputs/aligned_qa_valid_items.json; audio is omitted here because the native-video VLM benchmark is visual-only.", 14, anchor="middle"))
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/aligned_qa_valid_items.json", type=Path)
    parser.add_argument("--output", default="outputs/figures/qa_type_distribution_visual.svg", type=Path)
    args = parser.parse_args()

    items = load_items(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_svg(items))
    print(args.output)


if __name__ == "__main__":
    main()
