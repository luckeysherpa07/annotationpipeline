"""Shared caption generation utilities for all modalities.

This module contains functions for generating captions, questions, and answers
using the Gemini API. These functions are used by all annotation modalities (RGB, Event, Depth).
"""
from pathlib import Path
from typing import List

try:
    from google.genai import types
except ImportError:
    types = None

from ..utils import encode_frames_to_base64, build_image_parts


def get_caption_from_gemini(client, frame_paths: list, caption_prompt: str) -> str:
    """Generate a caption for video frames using Gemini API.

    Args:
        client: Gemini client instance
        frame_paths: List of Path objects to frame images
        caption_prompt: The prompt to send to the API

    Returns:
        The caption text from the API response
        
    Raises:
        ValueError: If no frames are provided
    """
    if not frame_paths:
        raise ValueError("No frames provided for captioning")

    # Build image content blocks (using first 10 frames to avoid context limits)
    frames_to_use = frame_paths[:10]
    encoded_frames = encode_frames_to_base64(frames_to_use)

    image_parts = build_image_parts(encoded_frames)

    contents = image_parts + [caption_prompt]

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=contents,
    )

    return response.text


def get_question_from_gemini(client, caption: str, question_prompt: str) -> str:
    """Generate a question from a caption using Gemini API.

    Args:
        client: Gemini client instance
        caption: The caption text to generate a question from
        question_prompt: The prompt to send to the API

    Returns:
        The question text from the API response
    """
    contents = [f"{caption}\n\n{question_prompt}"]

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=contents,
    )

    return response.text


def get_answer_from_gemini(client, frame_paths: list, question: str, answering_prompt: str) -> str:
    """Generate an answer for a question based on video frames using Gemini API.

    Args:
        client: Gemini client instance
        frame_paths: List of Path objects to frame images
        question: The question to answer
        answering_prompt: The prompt to send to the API

    Returns:
        The answer text from the API response
        
    Raises:
        ValueError: If no frames are provided
    """
    if not frame_paths:
        raise ValueError("No frames provided for answering")

    # Build image content blocks (using first 10 frames to avoid context limits)
    frames_to_use = frame_paths[:10]
    encoded_frames = encode_frames_to_base64(frames_to_use)

    image_parts = build_image_parts(encoded_frames)

    contents = image_parts + [f"Question: {question}\n\n{answering_prompt}"]

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=contents,
    )

    return response.text
