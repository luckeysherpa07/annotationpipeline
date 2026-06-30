"""Convert aligned QA split items into multimodal QA evidence units."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT_PATH = Path("outputs/aligned_qa_split_items.json")
DEFAULT_OUTPUT_PATH = Path("outputs/aligned_multimodal_visual_evidence_units.json")
VISUAL_MODALITIES = ("rgb", "event", "depth", "ir")
MODALITY_EXTENSIONS = {
    "rgb": (".mp4", ".avi", ".mov", ".mkv"),
    "event": (".mp4", ".avi", ".mov", ".mkv"),
    "depth": (".mp4", ".avi", ".mov", ".mkv"),
    "ir": (".mp4", ".avi", ".mov", ".mkv"),
}
VISUAL_PAIRS = (
    ("rgb", "event"),
    ("rgb", "depth"),
    ("rgb", "ir"),
    ("event", "ir"),
    ("event", "depth"),
)
EXCLUDED_SECTION_TOKENS = ("counting", "non_common")
LOW_ENTROPY_ANSWERS = {
    "yes",
    "no",
    "true",
    "false",
    "none",
    "nothing",
    "no one",
    "nobody",
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
}
SINGLE_COLOR_ANSWERS = {
    "black",
    "white",
    "red",
    "green",
    "blue",
    "yellow",
    "orange",
    "purple",
    "pink",
    "brown",
    "gray",
    "grey",
    "silver",
    "gold",
}
YES_NO_QUESTION_PREFIXES = (
    "is ",
    "are ",
    "was ",
    "were ",
    "do ",
    "does ",
    "did ",
    "can ",
    "could ",
    "would ",
    "should ",
    "has ",
    "have ",
    "had ",
)


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object in {path}")
    return data


def _save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def _segment_from_pair_key(pair_key: str) -> tuple[str, str, str] | None:
    parts = pair_key.replace("\\", "/").split("/")
    for index, part in enumerate(parts[:-1]):
        if part.endswith("_split") and parts[index + 1].startswith("Seg"):
            split_name = part
            segment_name = parts[index + 1]
            activity = split_name.removesuffix("_split")
            return split_name, segment_name, activity
    return None


def _task_label(activity: str, segment_name: str) -> str:
    label = activity.replace("_", " ")
    match = re.search(r"(\d+)$", segment_name)
    if match:
        return f"{label} segment {match.group(1)}"
    return f"{label} {segment_name}".strip()


def _available_pairs(modalities: set[str]) -> list[list[str]]:
    return [
        [first, second]
        for first, second in VISUAL_PAIRS
        if first in modalities and second in modalities
    ]


def _media_candidates(split_name: str, segment_name: str, modality: str) -> list[str]:
    segment_dir = Path("aligned_dataset") / split_name / segment_name
    extensions = MODALITY_EXTENSIONS.get(modality, ())
    if not segment_dir.exists():
        return []
    candidates = []
    for path in sorted(segment_dir.iterdir()):
        if not path.is_file():
            continue
        name = path.name.lower()
        if path.suffix.lower() not in extensions:
            continue
        if name.endswith(f"_{modality}{path.suffix.lower()}"):
            candidates.append(path.as_posix())
    return candidates


def _frame_cache_dir(split_name: str, segment_name: str) -> str | None:
    path = Path("aligned_dataset") / ".frames_cache" / split_name / segment_name
    return path.as_posix() if path.exists() else None


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()).strip(" .?!,;:\"'")


def _is_excluded_section(section: str) -> bool:
    normalized = section.strip().lower()
    return any(token in normalized for token in EXCLUDED_SECTION_TOKENS)


def _is_low_entropy_answer(answer: str) -> bool:
    normalized = _normalize_text(answer)
    if not normalized:
        return True
    if normalized in LOW_ENTROPY_ANSWERS or normalized in SINGLE_COLOR_ANSWERS:
        return True
    if re.fullmatch(r"\d+(?:\.\d+)?", normalized):
        return True
    if re.fullmatch(r"(?:about|approximately|around)?\s*\d+\s*(?:objects?|items?|people|times?)?", normalized):
        return True
    return False


def _is_yes_no_question(question: str) -> bool:
    normalized = _normalize_text(question)
    return normalized.startswith(YES_NO_QUESTION_PREFIXES)


def convert_aligned_split_items(
    input_path: Path | str = DEFAULT_INPUT_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    min_modalities: int = 2,
    filter_low_entropy: bool = True,
) -> Path:
    """Write aligned visual evidence units in multimodal QA pipeline format."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    raw_data = _load_json(input_path)
    items = raw_data.get("items")
    if not isinstance(items, list):
        raise ValueError(f"Expected {input_path} to contain an items list")

    grouped: dict[str, dict[str, Any]] = {}
    modality_sets: dict[str, set[str]] = defaultdict(set)
    source_files: dict[str, dict[str, str]] = defaultdict(dict)
    media_by_segment: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    excluded_counts: dict[str, int] = defaultdict(int)

    for item_index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        modality = str(item.get("modality") or "").strip().lower()
        if modality not in VISUAL_MODALITIES:
            excluded_counts["excluded_modality"] += 1
            continue
        segment_info = _segment_from_pair_key(str(item.get("pair_key") or ""))
        if segment_info is None:
            excluded_counts["missing_segment_key"] += 1
            continue
        split_name, segment_name, activity = segment_info
        segment_id = f"{split_name}__{segment_name}"
        caption = str(item.get("caption") or "").strip()
        question = str(item.get("question") or "").strip()
        answer = str(item.get("answer") or "").strip()
        section = str(item.get("section") or "").strip()
        if not section or not any((caption, question, answer)):
            excluded_counts["empty_evidence"] += 1
            continue
        if filter_low_entropy and _is_excluded_section(section):
            excluded_counts["excluded_section"] += 1
            continue
        if filter_low_entropy and _is_low_entropy_answer(answer):
            excluded_counts["low_entropy_answer"] += 1
            continue
        if filter_low_entropy and _is_yes_no_question(question):
            excluded_counts["yes_no_question"] += 1
            continue

        segment = grouped.setdefault(
            segment_id,
            {
                "segment_id": segment_id,
                "source_prefix": activity,
                "split_dir": split_name,
                "segment_name": segment_name,
                "side": "aligned",
                "task_label": _task_label(activity, segment_name),
                "source_files": {},
                "media_by_modality": {},
                "source_keys": {},
                "evidence_units": [],
            },
        )
        unit_index = len(segment["evidence_units"])
        segment["evidence_units"].append(
            {
                "modality": modality,
                "section": section,
                "caption": caption,
                "question": question,
                "answer": answer,
                "evidence": caption,
                "timestamp": None,
                "confidence": None,
                "source_unit_index": item_index,
                "pair_index": int(item.get("qa_index") or unit_index + 1),
                "source_pair_key": item.get("pair_key"),
                "source_file": item.get("source_file"),
            }
        )
        modality_sets[segment_id].add(modality)
        source_files[segment_id][modality] = str(item.get("source_file") or "")
        media_by_segment[segment_id][modality] = {
            "pair_key": item.get("pair_key"),
            "videos": _media_candidates(split_name, segment_name, modality),
            "frame_cache_dir": _frame_cache_dir(split_name, segment_name),
        }
        segment["source_keys"][modality] = item.get("pair_key")

    output: dict[str, dict[str, Any]] = {}
    for segment_id in sorted(grouped):
        modalities = modality_sets[segment_id]
        pairs = _available_pairs(modalities)
        if len(modalities) < min_modalities or not pairs:
            excluded_counts["insufficient_visual_modalities_segment"] += 1
            continue
        segment = grouped[segment_id]
        segment["evidence_modalities"] = sorted(modalities)
        segment["modality_pairs"] = pairs
        segment["source_files"] = {
            modality: path
            for modality, path in sorted(source_files[segment_id].items())
            if path
        }
        segment["media_by_modality"] = {
            modality: payload
            for modality, payload in sorted(media_by_segment[segment_id].items())
        }
        output[segment_id] = segment

    _save_json(
        {
            "metadata": {
                "source_file": str(input_path),
                "modality_filter": list(VISUAL_MODALITIES),
                "excluded_modalities": ["audio"],
                "filter_low_entropy": filter_low_entropy,
                "excluded_section_tokens": list(EXCLUDED_SECTION_TOKENS),
                "low_entropy_answer_policy": [
                    "yes_no_true_false",
                    "none_zero_small_number_words",
                    "pure_numeric",
                    "single_color_word",
                    "yes_no_question_prefix",
                ],
                "excluded_counts": dict(sorted(excluded_counts.items())),
                "supported_pairs": [list(pair) for pair in VISUAL_PAIRS],
                "segments": len(output),
            },
            "segments": output,
        },
        output_path,
    )
    return output_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument(
        "--min-modalities",
        type=int,
        default=2,
        help="Drop segments with fewer than this many visual modalities.",
    )
    parser.add_argument(
        "--no-low-entropy-filter",
        action="store_true",
        help="Keep counting, non-common, yes/no, numeric, and single-color QA evidence.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    output_path = convert_aligned_split_items(
        input_path=args.input,
        output_path=args.output,
        min_modalities=max(2, args.min_modalities),
        filter_low_entropy=not args.no_low_entropy_filter,
    )
    print(f"Wrote aligned visual evidence units to {output_path}")


if __name__ == "__main__":
    main()
