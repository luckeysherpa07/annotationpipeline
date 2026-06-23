#!/usr/bin/env python3
"""Generate a presentation-ready RGB/IR QA workflow figure."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path


def _text(
    x: float,
    y: float,
    value: str,
    size: int = 18,
    *,
    anchor: str = "middle",
    weight: str = "400",
    fill: str = "#171923",
) -> str:
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-family="DejaVu Sans, Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}">{escape(value)}</text>'
    )


def _multiline(x: float, y: float, lines: list[str], size: int = 18, weight: str = "400") -> str:
    spans = "".join(
        f'<tspan x="{x}" dy="{0 if index == 0 else size * 1.25}">{escape(line)}</tspan>'
        for index, line in enumerate(lines)
    )
    return (
        f'<text x="{x}" y="{y}" text-anchor="middle" '
        f'font-family="DejaVu Sans, Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="#171923">{spans}</text>'
    )


def _gemini_mark(cx: float, cy: float, scale: float = 1.0) -> str:
    """A compact four-point Gemini-style sparkle mark."""
    points = [
        (cx, cy - 24 * scale),
        (cx + 7 * scale, cy - 7 * scale),
        (cx + 24 * scale, cy),
        (cx + 7 * scale, cy + 7 * scale),
        (cx, cy + 24 * scale),
        (cx - 7 * scale, cy + 7 * scale),
        (cx - 24 * scale, cy),
        (cx - 7 * scale, cy - 7 * scale),
    ]
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polygon points="{coords}" fill="url(#geminiGradient)" />'


def _agent(x: float, y: float, color: str, label: str) -> str:
    return "\n".join(
        [
            f'<rect x="{x}" y="{y}" width="190" height="116" rx="22" fill="{color}" />',
            _gemini_mark(x + 58, y + 49, 0.75),
            '<rect x="{:.1f}" y="{:.1f}" width="42" height="32" rx="5" fill="#fff" stroke="#355070" stroke-width="2" />'.format(x + 88, y + 32),
            '<circle cx="{:.1f}" cy="{:.1f}" r="4" fill="#34a853" />'.format(x + 99, y + 43),
            '<circle cx="{:.1f}" cy="{:.1f}" r="4" fill="#fbbc04" />'.format(x + 111, y + 43),
            '<path d="M {:.1f} {:.1f} h 25" stroke="#4285f4" stroke-width="3" />'.format(x + 96, y + 54),
            _multiline(x + 95, y + 86, [label, "Agent"], 17, "700"),
        ]
    )


def _image_card(x: float, y: float, day: bool, thermal: bool = False) -> str:
    sky = ("#743c8f" if day else "#26133f") if thermal else ("#b9def7" if day else "#111d35")
    window = "#ffb347" if thermal else ("#ffd166" if not day else "#d9f0ff")
    return "\n".join(
        [
            f'<rect x="{x}" y="{y}" width="135" height="128" rx="4" fill="{sky}" stroke="#252b3a" stroke-width="3" />',
            f'<circle cx="{x + 105}" cy="{y + 24}" r="11" fill="{("#ffd34d" if day else "#f4f1c9")}" />',
            f'<path d="M {x + 12} {y + 112} L {x + 68} {y + 36} L {x + 123} {y + 112} Z" fill="#d9d5cf" />',
            f'<rect x="{x + 55}" y="{y + 78}" width="28" height="34" fill="#8b6f62" />',
            f'<rect x="{x + 60}" y="{y + 50}" width="18" height="20" fill="{window}" />',
            f'<path d="M {x + 6} {y + 113} H {x + 129}" stroke="#3b5139" stroke-width="9" />',
        ]
    )


def _arrow(x1: float, y1: float, x2: float, y2: float) -> str:
    return f'<path d="M {x1} {y1} H {x2}" fill="none" stroke="#171923" stroke-width="3" marker-end="url(#arrow)" />'


def _phase(x: float, y: float, width: float, fill: str, label: str) -> str:
    points = f"{x},{y} {x + width - 18},{y} {x + width},{y + 20} {x + width - 18},{y + 40} {x},{y + 40} {x + 18},{y + 20}"
    return f'<polygon points="{points}" fill="{fill}" stroke="#8a6b75" stroke-width="1.5" />\n{_text(x + width / 2, y + 27, label, 15, weight="700")}'


def build_svg(modality: str = "rgb") -> str:
    if modality not in {"rgb", "ir"}:
        raise ValueError("modality must be 'rgb' or 'ir'")
    is_ir = modality == "ir"
    modality_label = "IR" if is_ir else "RGB"
    title = f"Gemini-Powered {modality_label} QA Annotation"
    input_lines = ["Paired Night IR media"] if is_ir else ["Night RGB media"]
    caption_label = "1 · Analyze and caption IR stream" if is_ir else "1 · Caption NIGHT RGB frames"
    question_lines = (
        ["2 · Generate an IR-oriented", "question from the caption"]
        if is_ir
        else ["2 · Generate a night-oriented", "question from the caption"]
    )
    evidence_lines = ["Paired Day IR evidence", "supports grounded answering"] if is_ir else ["DAY RGB reference", "provides clearer evidence"]
    answer_lines = ["3 · Answer using paired", "IR-stream evidence"] if is_ir else ["3 · DAY-grounded answer", "to the night-oriented question"]
    output_lines = ["Paired DAY/NIGHT IR", "with grounded QA pairs"] if is_ir else ["NIGHT RGB images/video", "with grounded QA pairs"]
    footer = (
        "Prompt focus: thermal contrast, shapes, hotspots, and IR scenes"
        if is_ir
        else "Prompt focus: objects, colors, textures, and visible scenes"
    )
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="430" viewBox="0 0 1600 430">',
        """<defs>
  <linearGradient id="geminiGradient" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#4285f4"/><stop offset="0.48" stop-color="#8b5cf6"/>
    <stop offset="1" stop-color="#e86aa6"/>
  </linearGradient>
  <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
    <path d="M0,0 L0,6 L9,3 z" fill="#171923"/>
  </marker>
</defs>""",
        '<rect width="1600" height="430" fill="#ffffff" />',
        '<rect x="8" y="54" width="1584" height="356" rx="38" fill="#fff4d8" stroke="#d39b29" stroke-width="2" stroke-dasharray="7 6" />',
        _gemini_mark(510, 27, 0.65),
        _text(800, 38, title, 27, weight="700"),
        '<rect x="202" y="62" width="895" height="207" rx="28" fill="#f7f1ff" fill-opacity="0.62" stroke="#7c5cc4" stroke-width="2.5" stroke-dasharray="8 6" />',
        '<rect x="447" y="57" width="405" height="31" rx="15" fill="#ffffff" stroke="#7c5cc4" stroke-width="2" />',
        _text(650, 79, "ONE GEMINI MEGA-PROMPT · ONE API CALL", 15, weight="700", fill="#6545ad"),
        _image_card(38, 92, False, thermal=is_ir),
        _multiline(106, 239, input_lines, 15, "600"),
        _arrow(173, 139, 224, 139),
        _agent(224, 94, "#cfe7ff", "Caption"),
        _text(319, 229, caption_label, 14, weight="700"),
        _arrow(414, 139, 459, 139),
        _agent(459, 94, "#cceca0", "Question"),
        _multiline(554, 229, question_lines, 14, "700"),
        _arrow(649, 139, 700, 139),
        _image_card(700, 92, True, thermal=is_ir),
        _multiline(768, 239, evidence_lines, 13, "700"),
        _arrow(835, 139, 884, 139),
        _agent(884, 94, "#ffc9d3", "Answering"),
        _multiline(979, 229, answer_lines, 13, "700"),
        _arrow(1074, 139, 1127, 139),
        '<circle cx="1174" cy="139" r="46" fill="#e6e9ef" stroke="#e05a67" stroke-width="5" />',
        '<circle cx="1174" cy="119" r="13" fill="#55708d" />',
        '<path d="M 1149 164 Q 1174 132 1199 164" fill="#55708d" />',
        _text(1174, 207, "Human refinement", 15, weight="700"),
        _arrow(1220, 139, 1270, 139),
        '<rect x="1270" y="93" width="118" height="126" rx="10" fill="#2e7ab8" stroke="#1c4b70" stroke-width="3" />',
        '<rect x="1285" y="108" width="88" height="18" rx="4" fill="#f6ba35" />',
        '<rect x="1285" y="137" width="38" height="64" rx="4" fill="#123c60" />',
        '<rect x="1335" y="137" width="38" height="64" rx="4" fill="#123c60" />',
        '<rect x="1405" y="105" width="153" height="65" rx="12" fill="#bcefdc" stroke="#30977d" stroke-width="2" />',
        _text(1482, 132, "Q", 24, weight="700", fill="#146b61"),
        _text(1482, 157, "+ A", 21, weight="700", fill="#146b61"),
        _multiline(1414, 239, output_lines, 15, "700"),
        _phase(28, 326, 310, "#ffd0c5", "IR-stream captioning" if is_ir else "Night-time captioning"),
        _phase(328, 326, 390, "#d7e8ff", "IR question generation" if is_ir else "Night-time question generation"),
        _phase(708, 326, 360, "#d4efcb", "Paired IR answer synthesis" if is_ir else "Day-augmented answer synthesis"),
        _phase(1058, 326, 514, "#e7d9ef", "Human refinement and QA export"),
        _text(800, 397, footer, 15, weight="700", fill="#4c3d2f"),
        "</svg>",
    ]
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=("rgb", "ir"), default="rgb")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    output = args.output or Path(f"outputs/figures/{args.modality}_qa_pipeline.svg")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_svg(args.modality), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
