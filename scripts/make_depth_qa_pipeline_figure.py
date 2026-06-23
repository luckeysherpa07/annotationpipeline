#!/usr/bin/env python3
"""Generate the segmented Marigold Depth QA pipeline SVG used by option 52."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from scripts.make_gemini_annotation_pipeline_figure import _gemini_mark, _multiline, _text
except ModuleNotFoundError:
    from make_gemini_annotation_pipeline_figure import _gemini_mark, _multiline, _text  # type: ignore[no-redef]


def _arrow(x1: float, y: float, x2: float) -> str:
    return f'<path d="M {x1} {y} H {x2}" fill="none" stroke="#25344a" stroke-width="3" marker-end="url(#arrow)" />'


def _depth_map(x: float, y: float, side: str) -> str:
    tint = "#e7f2ff" if side == "DAY" else "#eee6ff"
    return "\n".join(
        [
            f'<rect x="{x}" y="{y}" width="250" height="116" rx="20" fill="{tint}" stroke="#4f6382" stroke-width="2" />',
            f'<rect x="{x + 16}" y="{y + 17}" width="74" height="72" rx="5" fill="url(#depthGradient)" />',
            f'<circle cx="{x + 53}" cy="{y + 41}" r="13" fill="none" stroke="#fff" stroke-width="3" />',
            f'<path d="M {x + 24} {y + 81} Q {x + 53} {y + 52} {x + 82} {y + 81}" fill="none" stroke="#fff" stroke-width="3" />',
            _multiline(x + 171, y + 43, [f"Cached {side} Marigold", "depth maps"], 15, "700"),
            _text(x + 171, y + 91, "frame_*_depth.png", 11, weight="600", fill="#526176"),
        ]
    )


def _source_box(x: float, y: float, side: str) -> str:
    fill = "#dcecff" if side == "DAY" else "#eadcff"
    return "\n".join(
        [
            f'<rect x="{x}" y="{y}" width="215" height="116" rx="20" fill="{fill}" stroke="#536b87" stroke-width="2" />',
            _text(x + 107, y + 34, f"{side} recording side", 17, weight="700"),
            _multiline(x + 107, y + 64, ["Task-segment manifest", "start/end timestamps"], 14, "600"),
        ]
    )


def _gemini_box(x: float, y: float, side: str) -> str:
    return "\n".join(
        [
            f'<rect x="{x}" y="{y}" width="310" height="116" rx="20" fill="#f8efff" stroke="#7656bb" stroke-width="2.5" />',
            _gemini_mark(x + 42, y + 35, 0.7),
            _text(x + 180, y + 31, f"Segmented {side} Depth QA", 17, weight="700"),
            _multiline(x + 180, y + 60, ["Caption → Question → Answer", "one call per source-side batch"], 14, "600"),
            _text(x + 180, y + 103, "Output keyed by segment_id", 11, weight="700", fill="#6545ad"),
        ]
    )


def _output_box(x: float, y: float, side: str) -> str:
    fill = "#d8efcf" if side == "DAY" else "#dcd8f4"
    return "\n".join(
        [
            f'<rect x="{x}" y="{y}" width="220" height="116" rx="20" fill="{fill}" stroke="#579353" stroke-width="2" />',
            _multiline(x + 110, y + 44, [f"{side} segment", "Depth QA JSON"], 18, "700"),
            _text(x + 110, y + 96, "caption · question · answer", 11, weight="600", fill="#435349"),
        ]
    )


def build_svg() -> str:
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="590" viewBox="0 0 1400 590">',
        """<defs>
  <linearGradient id="geminiGradient" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#4285f4"/><stop offset="0.48" stop-color="#8b5cf6"/>
    <stop offset="1" stop-color="#e86aa6"/>
  </linearGradient>
  <linearGradient id="depthGradient" x1="0" y1="1" x2="1" y2="0">
    <stop offset="0" stop-color="#29115f"/><stop offset="0.5" stop-color="#286bc0"/>
    <stop offset="1" stop-color="#f2d14b"/>
  </linearGradient>
  <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
    <path d="M0,0 L0,6 L9,3 z" fill="#25344a"/>
  </marker>
</defs>""",
        '<rect width="1400" height="590" fill="#ffffff" />',
        _text(700, 43, "Option 52 · Segmented Marigold Depth QA Pipeline", 27, weight="700"),
        _text(700, 72, "Day and night recording sides are processed independently", 17, weight="600", fill="#5a6473"),
        '<rect x="25" y="95" width="1350" height="410" rx="34" fill="#fff7e3" stroke="#d39b29" stroke-width="2" stroke-dasharray="7 6" />',
        _text(52, 126, "DAY LANE", 14, anchor="start", weight="700", fill="#3974a8"),
        _source_box(50, 142, "DAY"),
        _arrow(265, 200, 305),
        _depth_map(305, 142, "DAY"),
        _arrow(555, 200, 600),
        _gemini_box(600, 142, "DAY"),
        _arrow(910, 200, 955),
        _output_box(955, 142, "DAY"),
        '<line x1="50" y1="295" x2="1350" y2="295" stroke="#c8ad72" stroke-width="2" stroke-dasharray="8 7" />',
        _text(52, 326, "NIGHT LANE", 14, anchor="start", weight="700", fill="#7550a1"),
        _source_box(50, 342, "NIGHT"),
        _arrow(265, 400, 305),
        _depth_map(305, 342, "NIGHT"),
        _arrow(555, 400, 600),
        _gemini_box(600, 342, "NIGHT"),
        _arrow(910, 400, 955),
        _output_box(955, 342, "NIGHT"),
        '<rect x="250" y="523" width="900" height="44" rx="18" fill="#ffe1df" stroke="#d16666" stroke-width="1.5" />',
        _text(700, 551, "No day-to-night grounding or cross-side pairing · Option 52 does not estimate depth maps", 15, weight="700", fill="#7a3030"),
        "</svg>",
    ]
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/figures/depth_qa_pipeline.svg"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_svg(), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
