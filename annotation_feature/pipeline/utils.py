from pathlib import Path
import base64
from typing import List

try:
    from google.genai import types
except ImportError:
    types = None

video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".mpeg", ".mpg"}


def get_pair_key(file: Path) -> str:
    """
    Build a shared key for matching day/night RGB videos from the same scene.
    """
    stem = file.stem.lower()
    for token in ("_night", "_day", "night", "day"):
        stem = stem.replace(token, "")
    stem = stem.replace("__", "_").strip("_")
    return str(file.parent / stem)


def encode_frames_to_base64(frame_paths: list) -> list:
    """
    Encode image frames to base64 for API transmission.

    Args:
        frame_paths: List of Path objects to image files

    Returns:
        List of base64 encoded image strings
    """
    encoded_frames = []
    for frame_path in frame_paths:
        if not frame_path.exists():
            continue
        with open(frame_path, "rb") as f:
            encoded = base64.standard_b64encode(f.read()).decode("utf-8")
            encoded_frames.append(encoded)
    return encoded_frames


def build_image_parts(encoded_frames: list[str]) -> list:
    return [
        types.Part.from_bytes(data=base64.b64decode(encoded), mime_type="image/png")
        for encoded in encoded_frames
    ]