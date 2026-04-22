"""Shared utilities for caption generation across modalities."""
from .caption_generator import (
    get_caption_from_gemini,
    get_question_from_gemini,
    get_answer_from_gemini,
)

__all__ = [
    "get_caption_from_gemini",
    "get_question_from_gemini",
    "get_answer_from_gemini",
]
