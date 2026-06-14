"""Deterministic answer metrics for open-ended visual question answering."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Any


ARTICLES = {"a", "an", "the"}
NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
YES_WORDS = {"yes", "true", "yeah", "yep"}
NO_WORDS = {"no", "false", "nope"}


def normalize_text(text: str, *, remove_articles: bool = True) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = text.replace("’", "'")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = text.split()
    if remove_articles:
        tokens = [token for token in tokens if token not in ARTICLES]
    return " ".join(tokens)


def normalized_tokens(text: str) -> list[str]:
    return normalize_text(text).split()


def normalized_exact_match(reference: str, candidate: str) -> float:
    return float(normalize_text(reference) == normalize_text(candidate))


def token_prf(reference: str, candidate: str) -> dict[str, float]:
    reference_tokens = normalized_tokens(reference)
    candidate_tokens = normalized_tokens(candidate)
    if not reference_tokens and not candidate_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not reference_tokens or not candidate_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    overlap = sum((Counter(reference_tokens) & Counter(candidate_tokens)).values())
    precision = overlap / len(candidate_tokens)
    recall = overlap / len(reference_tokens)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _lcs_length(left: list[str], right: list[str]) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = [0] * (len(left) + 1)
    for right_token in right:
        current = [0]
        for index, left_token in enumerate(left, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(reference: str, candidate: str) -> float:
    reference_tokens = normalized_tokens(reference)
    candidate_tokens = normalized_tokens(candidate)
    if not reference_tokens and not candidate_tokens:
        return 1.0
    if not reference_tokens or not candidate_tokens:
        return 0.0
    lcs = _lcs_length(reference_tokens, candidate_tokens)
    precision = lcs / len(candidate_tokens)
    recall = lcs / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def levenshtein_distance(left: str, right: str) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for row, right_char in enumerate(right, start=1):
        current = [row]
        for column, left_char in enumerate(left, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def anls(reference: str, candidate: str, threshold: float = 0.5) -> float:
    reference_norm = normalize_text(reference, remove_articles=False)
    candidate_norm = normalize_text(candidate, remove_articles=False)
    maximum = max(len(reference_norm), len(candidate_norm))
    if maximum == 0:
        return 1.0
    similarity = 1.0 - levenshtein_distance(reference_norm, candidate_norm) / maximum
    return similarity if similarity >= threshold else 0.0


def character_f1(reference: str, candidate: str) -> float:
    reference_chars = list(normalize_text(reference, remove_articles=False).replace(" ", ""))
    candidate_chars = list(normalize_text(candidate, remove_articles=False).replace(" ", ""))
    if not reference_chars and not candidate_chars:
        return 1.0
    if not reference_chars or not candidate_chars:
        return 0.0
    overlap = sum((Counter(reference_chars) & Counter(candidate_chars)).values())
    precision = overlap / len(candidate_chars)
    recall = overlap / len(reference_chars)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def parse_boolean(text: str) -> bool | None:
    tokens = normalized_tokens(text)
    if not tokens:
        return None
    if tokens[0] in YES_WORDS:
        return True
    if tokens[0] in NO_WORDS:
        return False
    return None


def boolean_accuracy(reference: str, candidate: str) -> float | None:
    expected = parse_boolean(reference)
    predicted = parse_boolean(candidate)
    if expected is None:
        return None
    return float(predicted == expected)


def parse_number(text: str) -> float | None:
    normalized = normalize_text(text, remove_articles=False)
    match = re.search(r"(?<!\w)-?\d+(?:\.\d+)?", normalized)
    if match:
        return float(match.group(0))
    for token in normalized.split():
        if token in NUMBER_WORDS:
            return float(NUMBER_WORDS[token])
    return None


def numeric_accuracy(reference: str, candidate: str) -> float | None:
    expected = parse_number(reference)
    predicted = parse_number(candidate)
    if expected is None:
        return None
    return float(predicted is not None and math.isclose(expected, predicted))


def _concepts(text: str) -> list[str]:
    raw = unicodedata.normalize("NFKC", str(text)).lower()
    parts = re.split(r"\s*(?:,|;|/|\band\b|\bthen\b)\s*", raw)
    return [normalized for part in parts if (normalized := normalize_text(part))]


def set_f1(reference: str, candidate: str) -> float:
    expected = set(_concepts(reference))
    predicted = set(_concepts(candidate))
    if not expected and not predicted:
        return 1.0
    if not expected or not predicted:
        return 0.0
    overlap = len(expected & predicted)
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def sequence_order_score(reference: str, candidate: str) -> float:
    expected = _concepts(reference)
    predicted = _concepts(candidate)
    if not expected and not predicted:
        return 1.0
    if not expected or not predicted:
        return 0.0
    return _lcs_length(expected, predicted) / len(expected)


def repetition_score(text: str) -> dict[str, Any]:
    tokens = normalized_tokens(text)
    if len(tokens) < 4:
        return {"ratio": 0.0, "flag": False}
    unigrams = Counter(tokens)
    excess = sum(max(0, count - 2) for count in unigrams.values())
    bigrams = Counter(zip(tokens, tokens[1:]))
    repeated_bigrams = sum(max(0, count - 1) for count in bigrams.values())
    ratio = min(1.0, (excess + repeated_bigrams) / max(1, len(tokens)))
    return {"ratio": ratio, "flag": ratio >= 0.3}


def conciseness_violation(reference: str, candidate: str) -> bool:
    candidate_length = len(str(candidate).strip())
    reference_length = len(str(reference).strip())
    return candidate_length > 100 and candidate_length > max(20, reference_length * 4)


def deterministic_metrics(reference: str, candidate: str) -> dict[str, Any]:
    token = token_prf(reference, candidate)
    repetition = repetition_score(candidate)
    return {
        "normalized_exact_match": normalized_exact_match(reference, candidate),
        "token_precision": token["precision"],
        "token_recall": token["recall"],
        "token_f1": token["f1"],
        "rouge_l_f1": rouge_l_f1(reference, candidate),
        "anls": anls(reference, candidate),
        "character_f1": character_f1(reference, candidate),
        "boolean_accuracy": boolean_accuracy(reference, candidate),
        "numeric_accuracy": numeric_accuracy(reference, candidate),
        "set_f1": set_f1(reference, candidate),
        "sequence_order_score": sequence_order_score(reference, candidate),
        "repetition_ratio": repetition["ratio"],
        "repetition_flag": repetition["flag"],
        "conciseness_violation": conciseness_violation(reference, candidate),
        "answer_length_chars": len(str(candidate)),
    }
