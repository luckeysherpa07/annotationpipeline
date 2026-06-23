#!/usr/bin/env python3
"""Generate the aligned Marigold estimation and paired Depth QA workflow SVG."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from scripts.make_gemini_annotation_pipeline_figure import _agent, _gemini_mark, _multiline, _text
except ModuleNotFoundError:
    from make_gemini_annotation_pipeline_figure import _agent, _gemini_mark, _multiline, _text  # type: ignore[no-redef]


def _flow(path: str) -> str:
    return f'<path d="{path}" fill="none" stroke="#25344a" stroke-width="3" marker-end="url(#arrow)" />'


def _source_card(x: float, y: float, side: str, modality: str) -> str:
    thermal = modality == "IR"
    sky = "#32164f" if thermal else "#b9def7"
    accent = "#ff9f43" if thermal else "#ffd34d"
    return "\n".join(
        [
            f'<rect x="{x}" y="{y}" width="210" height="130" rx="18" fill="#eef2f7" stroke="#536b87" stroke-width="2" />',
            f'<rect x="{x + 14}" y="{y + 14}" width="84" height="84" rx="6" fill="{sky}" />',
            f'<circle cx="{x + 80}" cy="{y + 31}" r="9" fill="{accent}" />',
            f'<path d="M {x + 23} {y + 89} L {x + 56} {y + 43} L {x + 90} {y + 89} Z" fill="#d8d3cc" />',
            f'<rect x="{x + 50}" y="{y + 67}" width="18" height="22" fill="#8b6f62" />',
            _multiline(x + 151, y + 49, [side, f"{modality} frames"], 16, "700"),
            _text(x + 105, y + 119, "Aligned 30-second segment", 11, weight="600", fill="#5b6878"),
        ]
    )


def _depth_card(x: float, y: float, side: str) -> str:
    return "\n".join(
        [
            f'<rect x="{x}" y="{y}" width="210" height="130" rx="18" fill="#edf3fb" stroke="#536b87" stroke-width="2" />',
            f'<rect x="{x + 14}" y="{y + 14}" width="84" height="84" rx="6" fill="url(#depthGradient)" />',
            f'<circle cx="{x + 56}" cy="{y + 41}" r="14" fill="none" stroke="#fff" stroke-width="3" />',
            f'<path d="M {x + 23} {y + 89} Q {x + 56} {y + 54} {x + 89} {y + 89}" fill="none" stroke="#fff" stroke-width="3" />',
            _multiline(x + 151, y + 49, [side, "depth maps"], 16, "700"),
            _text(x + 105, y + 119, "16-bit Marigold PNG", 11, weight="600", fill="#5b6878"),
        ]
    )


def _paired_depth_card(x: float, y: float) -> str:
    parts = [
        f'<rect x="{x}" y="{y}" width="220" height="220" rx="22" fill="#edf3fb" stroke="#536b87" stroke-width="2" />',
        _text(x + 110, y + 31, "Paired depth maps", 17, weight="700"),
    ]
    for index, side in enumerate(("Day", "Night")):
        row_y = y + 48 + index * 78
        parts.extend(
            [
                f'<rect x="{x + 18}" y="{row_y}" width="70" height="64" rx="5" fill="url(#depthGradient)" />',
                f'<circle cx="{x + 53}" cy="{row_y + 21}" r="11" fill="none" stroke="#fff" stroke-width="3" />',
                f'<path d="M {x + 25} {row_y + 57} Q {x + 53} {row_y + 29} {x + 81} {row_y + 57}" fill="none" stroke="#fff" stroke-width="3" />',
                _multiline(x + 151, row_y + 28, [side, "depth maps"], 14, "700"),
            ]
        )
    parts.append(_text(x + 110, y + 210, "16-bit Marigold PNG", 11, weight="600", fill="#5b6878"))
    return "\n".join(parts)


def build_svg() -> str:
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="610" viewBox="0 0 1800 610">',
        """<defs>
  <linearGradient id="geminiGradient" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#4285f4"/><stop offset="0.48" stop-color="#8b5cf6"/>
    <stop offset="1" stop-color="#e86aa6"/>
  </linearGradient>
  <linearGradient id="depthGradient" x1="0" y1="1" x2="1" y2="0">
    <stop offset="0" stop-color="#29115f"/><stop offset="0.48" stop-color="#286bc0"/>
    <stop offset="1" stop-color="#f2d14b"/>
  </linearGradient>
  <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
    <path d="M0,0 L0,6 L9,3 z" fill="#25344a"/>
  </marker>
</defs>""",
        '<rect width="1800" height="610" fill="#ffffff" />',
        _text(900, 45, "Aligned Marigold Depth Estimation and QA Pipeline", 29, weight="700"),
        '<rect x="15" y="76" width="1770" height="500" rx="38" fill="#fff7e3" stroke="#d39b29" stroke-width="2" stroke-dasharray="7 6" />',
        _text(130, 108, "1 · ALIGNED SOURCE FRAMES", 13, weight="700", fill="#536b87"),
        _source_card(25, 135, "Day", "RGB"),
        _source_card(25, 335, "Night", "IR"),
        _flow("M 235 200 H 275 V 275 H 315"),
        _flow("M 235 400 H 275 V 305 H 315"),
        _text(440, 108, "2 · DEPTH ESTIMATION", 13, weight="700", fill="#7255b2"),
        '<rect x="315" y="205" width="250" height="170" rx="24" fill="#eee7ff" stroke="#7255b2" stroke-width="2.5" />',
        _gemini_mark(365, 255, 0.9),
        _multiline(455, 245, ["Marigold", "Depth Estimation"], 20, "700"),
        _text(440, 323, "prs-eth/marigold-depth-v1-1", 11, weight="600", fill="#655979"),
        _text(440, 350, "Existing maps are reused", 12, weight="700", fill="#6545ad"),
        _flow("M 565 290 H 625"),
        _text(730, 108, "3 · PAIRED DEPTH MAPS", 13, weight="700", fill="#3974a8"),
        _paired_depth_card(625, 180),
        _flow("M 845 290 H 890"),
        '<rect x="875" y="91" width="660" height="395" rx="30" fill="#f7f1ff" fill-opacity=".78" stroke="#7c5cc4" stroke-width="2.5" stroke-dasharray="8 6" />',
        '<rect x="1003" y="85" width="405" height="31" rx="15" fill="#ffffff" stroke="#7c5cc4" stroke-width="2" />',
        _text(1206, 107, "ONE GEMINI MEGA-PROMPT · ONE API CALL", 15, weight="700", fill="#6545ad"),
        _text(1205, 142, "4 · PAIRED DAY/NIGHT DEPTH QA", 13, weight="700", fill="#7255b2"),
        _agent(900, 180, "#cfe7ff", "Caption"),
        _multiline(995, 321, ["1 · Caption paired", "depth-stream evidence"], 13, "700"),
        _flow("M 1090 238 H 1110"),
        _agent(1110, 180, "#cceca0", "Question"),
        _multiline(1205, 321, ["2 · Generate a depth-oriented", "question from the caption"], 13, "700"),
        _flow("M 1300 238 H 1320"),
        _agent(1320, 180, "#ffc9d3", "Answering"),
        _multiline(1415, 321, ["3 · Answer using paired", "day/night depth maps"], 13, "700"),
        _flow("M 1510 238 H 1550"),
        '<rect x="1550" y="174" width="205" height="142" rx="22" fill="#d8efcf" stroke="#579353" stroke-width="2" />',
        '<path d="M 1590 195 H 1715 V 286 H 1590 Z" fill="#ffffff" stroke="#579353" stroke-width="2" />',
        '<path d="M 1605 218 H 1700 M 1605 239 H 1684 M 1605 260 H 1694" stroke="#74a36f" stroke-width="5" />',
        _text(1652, 347, "Depth QA pairs", 18, weight="700"),
        '<rect x="310" y="521" width="1180" height="35" rx="16" fill="#ffffff" stroke="#c29a48" stroke-width="1.3" />',
        _text(900, 544, "Prompt focus: depth structure, distance, spatial relations, scene layout, counting, navigation, and actions", 14, weight="700", fill="#5f4b26"),
        "</svg>",
    ]
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/figures/aligned_marigold_depth_qa_pipeline.svg"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_svg(), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
