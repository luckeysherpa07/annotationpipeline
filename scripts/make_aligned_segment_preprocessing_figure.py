#!/usr/bin/env python3
"""Generate the alignment, 30-second segmentation, and frame-cache workflow SVG."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from scripts.make_gemini_annotation_pipeline_figure import _multiline, _text
except ModuleNotFoundError:
    from make_gemini_annotation_pipeline_figure import _multiline, _text  # type: ignore[no-redef]


def _arrow(x1: float, y: float, x2: float) -> str:
    return f'<path d="M {x1} {y} H {x2}" fill="none" stroke="#27364b" stroke-width="3" marker-end="url(#arrow)" />'


def _source_media(x: float, y: float) -> str:
    labels = (("RGB", "#9fd0f3"), ("Event", "#cbbaf7"), ("IR", "#f2b6c8"), ("Depth", "#b9dfbf"))
    parts = [f'<rect x="{x}" y="{y}" width="220" height="150" rx="22" fill="#e8eef6" stroke="#536b87" stroke-width="2" />']
    for index, (label, color) in enumerate(labels):
        row = index // 2
        col = index % 2
        bx = x + 17 + col * 96
        by = y + 17 + row * 43
        width = 90
        parts.append(f'<rect x="{bx}" y="{by}" width="{width}" height="34" rx="8" fill="{color}" stroke="#526176" stroke-width="1.5" />')
        parts.append(_text(bx + width / 2, by + 23, label, 13, weight="700"))
    parts.append(_text(x + 110, y + 136, "Source modality media", 16, weight="700"))
    return "\n".join(parts)


def _cache_folders(x: float, y: float) -> str:
    labels = (("RGB", "#9fd0f3"), ("IR", "#f2b6c8"), ("Event", "#cbbaf7"), ("Depth", "#b9dfbf"))
    parts: list[str] = []
    for index, (label, color) in enumerate(labels):
        fx = x + index * 67
        parts.append(f'<path d="M {fx} {y + 10} H {fx + 20} L {fx + 28} {y} H {fx + 59} V {y + 62} H {fx} Z" fill="{color}" stroke="#355070" stroke-width="1.7" />')
        parts.append(_text(fx + 30, y + 39, label, 10 if label != "Event" else 9, weight="700"))
    return "\n".join(parts)


def build_svg() -> str:
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="420" viewBox="0 0 1600 420">',
        """<defs>
  <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
    <path d="M0,0 L0,6 L9,3 z" fill="#27364b"/>
  </marker>
</defs>""",
        '<rect width="1600" height="420" fill="#ffffff" />',
        _text(800, 47, "Aligned Segment and Frame-Cache Preprocessing", 29, weight="700"),
        '<rect x="18" y="80" width="1564" height="300" rx="36" fill="#fff7e3" stroke="#d39b29" stroke-width="2" stroke-dasharray="7 6" />',
        _source_media(42, 166),
        _arrow(262, 241, 300),
        '<rect x="300" y="132" width="300" height="218" rx="24" fill="#eee7ff" stroke="#7255b2" stroke-width="2" />',
        _text(450, 169, "Temporal alignment", 20, weight="700"),
        '<rect x="325" y="187" width="250" height="48" rx="12" fill="#dcd0fa" />',
        _text(450, 218, "Event → RGB: windowed DTW", 15, weight="700"),
        '<rect x="325" y="246" width="250" height="70" rx="12" fill="#f6f1ff" />',
        _multiline(450, 270, ["IR and Depth → RGB:", "fixed-offset cross-correlation"], 14, "700"),
        _text(450, 339, "Day/night sides aligned independently", 11, weight="600", fill="#655979"),
        _arrow(600, 241, 635),
        '<rect x="635" y="153" width="225" height="176" rx="22" fill="#e5f3f4" stroke="#3f8990" stroke-width="2" />',
        _text(747, 189, "Common overlap", 19, weight="700"),
        '<line x1="669" y1="222" x2="827" y2="222" stroke="#516779" stroke-width="5" />',
        '<line x1="689" y1="243" x2="817" y2="243" stroke="#7656bb" stroke-width="5" />',
        '<line x1="703" y1="264" x2="805" y2="264" stroke="#d98645" stroke-width="5" />',
        '<line x1="703" y1="286" x2="805" y2="286" stroke="#3b9b72" stroke-width="8" />',
        _text(754, 316, "Latest start → earliest end", 11, weight="700", fill="#3d5f57"),
        _arrow(860, 241, 895),
        '<rect x="895" y="153" width="250" height="176" rx="22" fill="#fff0c7" stroke="#b88628" stroke-width="2" />',
        _text(1020, 188, "30-second segmentation", 19, weight="700"),
        '<rect x="920" y="211" width="58" height="70" rx="8" fill="#ffffff" stroke="#a77c27" stroke-width="2" />',
        '<rect x="991" y="211" width="58" height="70" rx="8" fill="#ffffff" stroke="#a77c27" stroke-width="2" />',
        '<rect x="1062" y="211" width="58" height="70" rx="8" fill="#ffffff" stroke="#a77c27" stroke-width="2" />',
        _text(949, 253, "Seg1", 13, weight="700"),
        _text(1020, 253, "Seg2", 13, weight="700"),
        _text(1091, 253, "SegN", 13, weight="700"),
        _text(1020, 310, "Exact full segments; remainder recorded", 11, weight="600", fill="#6f592b"),
        _arrow(1145, 241, 1180),
        '<rect x="1180" y="132" width="365" height="218" rx="24" fill="#e5f0ff" stroke="#4f78a8" stroke-width="2" />',
        _text(1362, 169, "Frame extraction and caching", 19, weight="700"),
        _multiline(1362, 202, ["Extract PNG frames at 1 FPS", "Reuse existing modality caches"], 15, "700"),
        _cache_folders(1228, 253),
        _text(1362, 337, "Ready for segmented QA", 14, weight="700", fill="#315b88"),
        "</svg>",
    ]
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/figures/aligned_segment_preprocessing.svg"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_svg(), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
