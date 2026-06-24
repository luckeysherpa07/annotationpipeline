#!/usr/bin/env python3
"""Generate tables comparing visual QA types and modality evidence focus."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from scripts.make_gemini_annotation_pipeline_figure import _text
except ModuleNotFoundError:
    from make_gemini_annotation_pipeline_figure import _text  # type: ignore[no-redef]


MODALITIES = ("RGB", "IR", "Event", "Depth")
COLORS = {
    "RGB": "#9fd0f3",
    "IR": "#f2b6c8",
    "Event": "#cbbaf7",
    "Depth": "#b9dfbf",
}

QA_ROWS = (
    ("Object recognition", True, True, True, True),
    ("Spatial reasoning", True, True, True, True),
    ("Text recognition", True, True, False, False),
    ("Scene sequence", True, True, True, True),
    ("Light recognition", True, True, False, False),
    ("Light change", True, True, False, False),
    ("Counting", True, True, True, True),
    ("Dynamic counting", True, True, True, True),
    ("Dynamic recognition", True, True, True, True),
    ("Navigation", True, True, True, True),
    ("Action recognition", True, True, True, True),
    ("Non-common / implausible scenes", True, True, True, True),
)

EMPHASIS_ROWS = (
    ("RGB", "Objects, colors, textures, visible scenes, and illumination"),
    ("IR", "Thermal contrast, shapes, hotspots, and IR scenes"),
    ("Event", "Motion, event activity, temporal changes, and moving boundaries"),
    ("Depth", "Distance, 3D structure, spatial ordering, and geometry"),
)


def build_svg() -> str:
    width, height = 1500, 1060
    table_x, table_y = 65, 115
    label_w, col_w = 610, 190
    header_h, row_h = 58, 45
    table_w = label_w + 4 * col_w
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="1500" height="1060" fill="#ffffff" />',
        _text(750, 48, "Visual QA Types Across Modalities", 30, weight="700"),
        _text(750, 78, "Configured prompt categories for RGB, IR, Event, and Depth", 16, weight="600", fill="#657083"),
        f'<rect x="{table_x}" y="{table_y}" width="{table_w}" height="{header_h + (len(QA_ROWS) + 1) * row_h}" rx="18" fill="#ffffff" stroke="#536b87" stroke-width="2" />',
        f'<path d="M {table_x} {table_y + header_h} H {table_x + table_w}" stroke="#536b87" stroke-width="2" />',
        f'<rect x="{table_x}" y="{table_y}" width="{label_w}" height="{header_h}" rx="18" fill="#e8eef6" />',
        _text(table_x + 26, table_y + 37, "QA type", 18, anchor="start", weight="700"),
    ]

    for index, modality in enumerate(MODALITIES):
        x = table_x + label_w + index * col_w
        parts.append(f'<rect x="{x}" y="{table_y}" width="{col_w}" height="{header_h}" fill="{COLORS[modality]}" />')
        parts.append(_text(x + col_w / 2, table_y + 37, modality, 18, weight="700"))

    all_rows = list(QA_ROWS) + [("Total configured types", "12", "12", "9", "9")]
    for row_index, row in enumerate(all_rows):
        y = table_y + header_h + row_index * row_h
        fill = "#f4f7fb" if row_index % 2 == 0 else "#ffffff"
        if row_index == len(QA_ROWS):
            fill = "#fff3cf"
        parts.append(f'<rect x="{table_x}" y="{y}" width="{table_w}" height="{row_h}" fill="{fill}" />')
        parts.append(_text(table_x + 26, y + 29, str(row[0]), 15, anchor="start", weight="700" if row_index == len(QA_ROWS) else "600"))
        for col_index, value in enumerate(row[1:]):
            cx = table_x + label_w + col_index * col_w + col_w / 2
            if isinstance(value, bool):
                if value:
                    color = COLORS[MODALITIES[col_index]]
                    parts.append(f'<circle cx="{cx}" cy="{y + row_h / 2}" r="12" fill="{color}" stroke="#355070" stroke-width="1.5" />')
                    parts.append(_text(cx, y + 28, "✓", 15, weight="700"))
                else:
                    parts.append(_text(cx, y + 29, "—", 18, weight="600", fill="#9aa3af"))
            else:
                parts.append(_text(cx, y + 29, str(value), 17, weight="700"))

        parts.append(f'<path d="M {table_x} {y + row_h} H {table_x + table_w}" stroke="#c9d1dc" stroke-width="1" />')

    for boundary in range(5):
        x = table_x + label_w + boundary * col_w
        parts.append(f'<path d="M {x} {table_y} V {table_y + header_h + len(all_rows) * row_h}" stroke="#9aa8b9" stroke-width="1" />')

    emphasis_y = 830
    emphasis_h = 175
    parts.extend(
        [
            _text(65, 806, "Modality evidence emphasis", 21, anchor="start", weight="700"),
            f'<rect x="65" y="{emphasis_y}" width="1370" height="{emphasis_h}" rx="18" fill="#ffffff" stroke="#536b87" stroke-width="2" />',
        ]
    )
    card_w = 1370 / 4
    for index, (modality, description) in enumerate(EMPHASIS_ROWS):
        x = 65 + index * card_w
        if index:
            parts.append(f'<path d="M {x} {emphasis_y} V {emphasis_y + emphasis_h}" stroke="#9aa8b9" stroke-width="1" />')
        parts.append(f'<rect x="{x + 12}" y="{emphasis_y + 12}" width="80" height="32" rx="9" fill="{COLORS[modality]}" />')
        parts.append(_text(x + 52, emphasis_y + 34, modality, 14, weight="700"))
        words = description.split()
        lines: list[str] = []
        current: list[str] = []
        for word in words:
            if len(" ".join([*current, word])) > 27 and current:
                lines.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            lines.append(" ".join(current))
        for line_index, line in enumerate(lines[:3]):
            parts.append(_text(x + 18, emphasis_y + 76 + line_index * 27, line, 18, anchor="start", weight="700", fill="#455064"))

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/figures/visual_qa_types_by_modality.svg"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_svg(), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
