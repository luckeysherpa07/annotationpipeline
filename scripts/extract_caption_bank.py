"""Extract unique captions from flattened QA items.

The cleaned QA file is QA-level, so the same caption can appear multiple times.
This script creates caption-level JSON/CSV files for easier manual inspection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


DEFAULT_INPUT = Path("outputs/aligned_qa_cleaned_items.json")
DEFAULT_OUTPUT_JSON = Path("outputs/caption_bank.json")
DEFAULT_OUTPUT_CSV = Path("outputs/caption_bank.csv")


def normalize_caption(caption: str) -> str:
    return " ".join((caption or "").strip().split())


def load_items(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return data["items"]
    raise ValueError(f"Unsupported QA item format in {path}")


def build_caption_bank(items: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, str], dict] = {}

    for item in items:
        caption = normalize_caption(str(item.get("caption") or ""))
        if not caption:
            continue

        modality = str(item.get("modality") or "")
        pair_key = str(item.get("pair_key") or "")
        source_file = str(item.get("source_file") or "")
        key = (caption, modality, pair_key, source_file)

        if key not in grouped:
            caption_id_base = "|".join(key)
            caption_id = hashlib.sha1(caption_id_base.encode("utf-8")).hexdigest()[:12]
            grouped[key] = {
                "caption_id": caption_id,
                "caption": caption,
                "modality": modality,
                "pair_key": pair_key,
                "source_file": source_file,
                "sections": set(),
                "qa_ids": [],
                "questions": [],
                "answers": [],
            }

        record = grouped[key]
        section = str(item.get("section") or "")
        qa_id = str(item.get("qa_id") or "")
        question = str(item.get("question") or "")
        answer = str(item.get("answer") or "")
        if section:
            record["sections"].add(section)
        if qa_id:
            record["qa_ids"].append(qa_id)
        if question:
            record["questions"].append(question)
        if answer:
            record["answers"].append(answer)

    bank = []
    for record in grouped.values():
        qa_ids = sorted(set(record["qa_ids"]))
        questions = sorted(set(record["questions"]))
        answers = sorted(set(record["answers"]))
        bank.append(
            {
                "caption_id": record["caption_id"],
                "modality": record["modality"],
                "pair_key": record["pair_key"],
                "source_file": record["source_file"],
                "sections": sorted(record["sections"]),
                "num_existing_qa": len(qa_ids),
                "existing_qa_ids": qa_ids,
                "existing_questions": questions,
                "existing_answers": answers,
                "caption": record["caption"],
            }
        )

    return sorted(bank, key=lambda r: (r["modality"], r["pair_key"], r["caption_id"]))


def write_json(path: Path, bank: list[dict], source: Path, total_items: int) -> None:
    payload = {
        "metadata": {
            "source_file": str(source),
            "total_qa_items": total_items,
            "unique_captions": len(bank),
        },
        "captions": bank,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, bank: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "caption_id",
        "modality",
        "pair_key",
        "source_file",
        "sections",
        "num_existing_qa",
        "caption",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in bank:
            writer.writerow(
                {
                    "caption_id": record["caption_id"],
                    "modality": record["modality"],
                    "pair_key": record["pair_key"],
                    "source_file": record["source_file"],
                    "sections": ";".join(record["sections"]),
                    "num_existing_qa": record["num_existing_qa"],
                    "caption": record["caption"],
                }
            )


def summarize(bank: list[dict], total_items: int) -> None:
    by_modality: dict[str, int] = defaultdict(int)
    qa_by_modality: dict[str, int] = defaultdict(int)
    for record in bank:
        by_modality[record["modality"]] += 1
        qa_by_modality[record["modality"]] += int(record["num_existing_qa"])

    print(f"QA items: {total_items}")
    print(f"Unique captions: {len(bank)}")
    print("Unique captions by modality:")
    for modality in sorted(by_modality):
        print(f"  {modality}: {by_modality[modality]} captions, {qa_by_modality[modality]} QA links")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    args = parser.parse_args()

    items = load_items(args.input)
    bank = build_caption_bank(items)
    write_json(args.output_json, bank, args.input, len(items))
    write_csv(args.output_csv, bank)
    summarize(bank, len(items))
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
