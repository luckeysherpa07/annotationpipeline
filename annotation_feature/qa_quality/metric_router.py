"""Choose a task-aware primary deterministic metric for each QA item."""

from __future__ import annotations

from typing import Any

from .answer_metrics import parse_boolean, parse_number


def route_metric(
    section: str,
    question: str,
    ground_truth_answer: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    section = str(section).lower()
    question = str(question).lower()
    if parse_boolean(ground_truth_answer) is not None:
        name = "boolean_accuracy"
        reason = "Ground truth is a boolean answer."
    elif "count" in section or "how many" in question:
        name = "numeric_accuracy" if parse_number(ground_truth_answer) is not None else "token_f1"
        reason = "Counting question."
    elif any(marker in section for marker in ("text", "ocr", "reading")):
        name = "anls"
        reason = "Text-recognition answer."
    elif any(marker in section for marker in ("sequence", "navigation", "order", "temporal")):
        name = "sequence_order_score"
        reason = "Ordered or temporal answer."
    elif any(separator in ground_truth_answer.lower() for separator in (",", ";", " and ")):
        name = "set_f1"
        reason = "Ground truth contains multiple concepts."
    else:
        name = "token_f1"
        reason = "General open-ended short answer."
    value = metrics.get(name)
    if value is None:
        name = "token_f1"
        value = metrics.get(name, 0.0)
        reason += " Routed metric was unavailable; token F1 used."
    return {"metric": name, "score": float(value), "reason": reason}
