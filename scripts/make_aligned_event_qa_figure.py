#!/usr/bin/env python3
"""Generate the aligned paired Event QA workflow SVG used by option 40."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from scripts.make_gemini_annotation_pipeline_figure import _agent, _multiline, _text
except ModuleNotFoundError:
    from make_gemini_annotation_pipeline_figure import _agent, _multiline, _text  # type: ignore[no-redef]


def _flow(path: str) -> str:
    return f'<path d="{path}" fill="none" stroke="#25344a" stroke-width="3" marker-end="url(#arrow)" />'


def _event_thumbnail(x: float, y: float, width: float = 84, height: float = 84) -> str:
    return "\n".join(
        [
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="6" fill="#20133f" />',
            f'<circle cx="{x + 19}" cy="{y + 19}" r="5" fill="#ff5470" />',
            f'<circle cx="{x + 57}" cy="{y + 27}" r="4" fill="#58c7ff" />',
            f'<circle cx="{x + 29}" cy="{y + 55}" r="6" fill="#ff5470" />',
            f'<circle cx="{x + 68}" cy="{y + 67}" r="5" fill="#58c7ff" />',
            f'<path d="M {x + 13} {y + 75} L {x + 39} {y + 34} L {x + 74} {y + 75}" fill="none" stroke="#b8a8ff" stroke-width="4" />',
        ]
    )


def _source_card(x: float, y: float, side: str) -> str:
    return "\n".join(
        [
            f'<rect x="{x}" y="{y}" width="210" height="130" rx="18" fill="#eef2f7" stroke="#536b87" stroke-width="2" />',
            _event_thumbnail(x + 14, y + 14),
            _multiline(x + 151, y + 49, [side, "Event video"], 16, "700"),
            _text(x + 105, y + 119, "Aligned 30-second segment", 11, weight="600", fill="#5b6878"),
        ]
    )


def _paired_event_card(x: float, y: float) -> str:
    parts = [
        f'<rect x="{x}" y="{y}" width="220" height="220" rx="22" fill="#edf3fb" stroke="#536b87" stroke-width="2" />',
        _text(x + 110, y + 31, "Paired Event frames", 17, weight="700"),
    ]
    for index, side in enumerate(("Day", "Night")):
        row_y = y + 48 + index * 78
        parts.append(_event_thumbnail(x + 18, row_y, 70, 64))
        parts.append(_multiline(x + 151, row_y + 28, [side, "Event frames"], 14, "700"))
    parts.append(_text(x + 110, y + 210, "Cached PNG frames · 1 FPS", 11, weight="600", fill="#5b6878"))
    return "\n".join(parts)


def build_svg() -> str:
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1450" height="530" viewBox="0 0 1450 530">',
        """<defs>
  <linearGradient id="geminiGradient" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#4285f4"/><stop offset="0.48" stop-color="#8b5cf6"/>
    <stop offset="1" stop-color="#e86aa6"/>
  </linearGradient>
  <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
    <path d="M0,0 L0,6 L9,3 z" fill="#25344a"/>
  </marker>
</defs>""",
        '<rect width="1450" height="530" fill="#ffffff" />',
        _text(725, 43, "Aligned Event QA Pipeline", 29, weight="700"),
        '<rect x="15" y="72" width="1420" height="430" rx="38" fill="#fff7e3" stroke="#d39b29" stroke-width="2" stroke-dasharray="7 6" />',
        _text(145, 105, "1 · PAIRED EVENT FRAMES", 13, weight="700", fill="#3974a8"),
        _paired_event_card(35, 145),
        _flow("M 255 255 H 405"),
        '<rect x="290" y="87" width="875" height="350" rx="30" fill="#f7f1ff" fill-opacity=".78" stroke="#7c5cc4" stroke-width="2.5" stroke-dasharray="8 6" />',
        '<rect x="523" y="81" width="405" height="31" rx="15" fill="#ffffff" stroke="#7c5cc4" stroke-width="2" />',
        _text(726, 103, "ONE GEMINI MEGA-PROMPT · ONE API CALL", 15, weight="700", fill="#6545ad"),
        _text(727, 138, "2 · PAIRED DAY/NIGHT EVENT QA", 13, weight="700", fill="#7255b2"),
        _agent(405, 197, "#cfe7ff", "Caption"),
        _multiline(500, 338, ["1 · Caption paired", "event-stream evidence"], 13, "700"),
        _flow("M 595 255 H 635"),
        _agent(635, 197, "#cceca0", "Question"),
        _multiline(730, 338, ["2 · Generate an event-oriented", "question from the caption"], 13, "700"),
        _flow("M 825 255 H 865"),
        _agent(865, 197, "#ffc9d3", "Answering"),
        _multiline(960, 338, ["3 · Answer using paired", "day/night Event frames"], 13, "700"),
        _flow("M 1055 255 H 1195"),
        '<rect x="1195" y="184" width="205" height="142" rx="22" fill="#d8efcf" stroke="#579353" stroke-width="2" />',
        '<path d="M 1235 205 H 1360 V 296 H 1235 Z" fill="#ffffff" stroke="#579353" stroke-width="2" />',
        '<path d="M 1250 228 H 1345 M 1250 249 H 1329 M 1250 270 H 1339" stroke="#74a36f" stroke-width="5" />',
        _text(1297, 357, "Event QA pairs", 18, weight="700"),
        '<rect x="210" y="452" width="1030" height="35" rx="16" fill="#ffffff" stroke="#c29a48" stroke-width="1.3" />',
        _text(725, 475, "Prompt focus: event activity, motion, temporal changes, counting, navigation, actions, and unusual events", 14, weight="700", fill="#5f4b26"),
        "</svg>",
    ]
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/figures/aligned_event_qa_pipeline.svg"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_svg(), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
