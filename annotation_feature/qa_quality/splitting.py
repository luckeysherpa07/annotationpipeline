"""Helpers for detecting and splitting numbered QA lists."""

from __future__ import annotations

import re


NUMBERED_ITEM_RE = re.compile(
    r"(?:^|\n|\s)(?P<index>\d{1,2})[\.)]\s+(?P<text>.*?)(?=(?:\n|\s)\d{1,2}[\.)]\s+|$)",
    flags=re.S,
)


def split_numbered_items(text: str) -> list[str]:
    """Return numbered-list items from text, or [] when no list is detected."""
    cleaned = str(text or "").strip()
    if not cleaned:
        return []

    matches = list(NUMBERED_ITEM_RE.finditer(cleaned))
    if len(matches) < 2:
        return []

    items = []
    for match in matches:
        item = re.sub(r"\s+", " ", match.group("text")).strip(" ;")
        if item:
            items.append(item)
    return items


def numbered_item_count(text: str) -> int:
    return len(split_numbered_items(text))


def split_status(question: str, answer: str) -> tuple[str, list[str], list[str]]:
    """Classify whether question/answer can be split into aligned items."""
    questions = split_numbered_items(question)
    answers = split_numbered_items(answer)

    if not questions and not answers:
        return "single", questions, answers
    if questions and answers and len(questions) == len(answers):
        return "aligned", questions, answers
    return "count_mismatch", questions, answers
