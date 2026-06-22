#!/usr/bin/env python3
"""Generate a standalone frame-cache and segment-timestamp workflow SVG."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from scripts.make_gemini_annotation_pipeline_figure import _multiline, _text
except ModuleNotFoundError:
    from make_gemini_annotation_pipeline_figure import _multiline, _text  # type: ignore[no-redef]


def _flow(path: str) -> str:
    return f'<path d="{path}" fill="none" stroke="#27364b" stroke-width="3" marker-end="url(#arrow)" />'


def build_svg() -> str:
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="430" viewBox="0 0 1400 430">',
        """<defs>
  <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
    <path d="M0,0 L0,6 L9,3 z" fill="#27364b"/>
  </marker>
</defs>""",
        '<rect width="1400" height="430" fill="#ffffff" />',
        _text(700, 45, "Preprocessing for Segmented Modality QA", 28, weight="700"),
        '<rect x="26" y="78" width="1348" height="320" rx="34" fill="#fff7e3" stroke="#d39b29" stroke-width="2" stroke-dasharray="7 6" />',
        '<rect x="55" y="173" width="190" height="118" rx="20" fill="#e8eef6" stroke="#536b87" stroke-width="2" />',
        '<rect x="80" y="193" width="140" height="64" rx="8" fill="#27364b" />',
        '<polygon points="95,245 130,207 168,245" fill="#d6d2c9" />',
        '<circle cx="196" cy="207" r="10" fill="#ffd45a" />',
        _text(150, 281, "Source modality media", 15, weight="700"),
        _flow("M 245 232 H 294 V 148 H 330"),
        _flow("M 245 232 H 294 V 318 H 330"),
        '<rect x="330" y="100" width="280" height="96" rx="20" fill="#dcecff" stroke="#4f78a8" stroke-width="2" />',
        _multiline(470, 135, ["Extract PNG frames at 1 FPS", "and reuse existing caches"], 17, "700"),
        '<rect x="330" y="270" width="280" height="96" rx="20" fill="#fff0c7" stroke="#b88628" stroke-width="2" />',
        _multiline(470, 305, ["Generate semantic segment", "start/end timestamps"], 17, "700"),
        _flow("M 610 148 H 665"),
        _flow("M 610 318 H 665"),
        '<rect x="665" y="91" width="360" height="114" rx="22" fill="#f1f7ff" stroke="#4f78a8" stroke-width="2" />',
        '<path d="M 682 125 H 703 L 712 115 H 750 V 177 H 682 Z" fill="#9fd0f3" stroke="#355070" stroke-width="2" />',
        '<path d="M 766 125 H 787 L 796 115 H 834 V 177 H 766 Z" fill="#f2b6c8" stroke="#355070" stroke-width="2" />',
        '<path d="M 850 125 H 871 L 880 115 H 918 V 177 H 850 Z" fill="#cbbaf7" stroke="#355070" stroke-width="2" />',
        '<path d="M 934 125 H 955 L 964 115 H 1002 V 177 H 934 Z" fill="#b9dfbf" stroke="#355070" stroke-width="2" />',
        _text(716, 155, "RGB", 13, weight="700"),
        _text(800, 155, "IR", 13, weight="700"),
        _text(884, 155, "Event", 12, weight="700"),
        _text(968, 155, "Depth", 12, weight="700"),
        _text(845, 196, "Reusable modality frame caches", 15, weight="700"),
        '<rect x="665" y="261" width="360" height="114" rx="22" fill="#fffaf0" stroke="#b88628" stroke-width="2" />',
        '<path d="M 700 278 H 990 V 347 H 700 Z" fill="#ffffff" stroke="#8c6b28" stroke-width="2" />',
        '<path d="M 720 299 H 965 M 720 317 H 930 M 720 335 H 950" stroke="#c39842" stroke-width="5" />',
        _text(845, 367, "Task-segment JSON manifest", 15, weight="700"),
        _flow("M 1025 148 H 1080 V 232 H 1115"),
        _flow("M 1025 318 H 1080 V 232 H 1115"),
        '<rect x="1115" y="169" width="225" height="126" rx="22" fill="#d8efcf" stroke="#579353" stroke-width="2" />',
        _multiline(1227, 205, ["Cached evidence", "+ segment timestamps", "ready for QA pipeline"], 17, "700"),
        "</svg>",
    ]
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/figures/segmented_qa_preprocessing.svg"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_svg(), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
