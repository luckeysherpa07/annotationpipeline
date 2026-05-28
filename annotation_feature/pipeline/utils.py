from pathlib import Path
import base64
import re
from typing import List

try:
    from google.genai import types
except ImportError:
    types = None

video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".mpeg", ".mpg"}
audio_extensions = {".m4a", ".mp3", ".wav", ".aac", ".flac"}


SIDE_ALIAS_PATTERNS = {
    "night": (
        r"night\d*",
        r"cloudy_no_light",
        r"no_light",
    ),
    "day": (
        r"day",
        r"with_light",
    ),
}


def _stem_from_name(value: str | Path) -> str:
    return Path(value).stem.lower()


def _contains_side_alias(stem: str, alias_pattern: str) -> bool:
    return re.search(rf"(?:^|_){alias_pattern}(?:_|$)", stem) is not None


def infer_recording_side(value: str | Path) -> str | None:
    """Infer whether a media/cache name represents the day or night side."""
    stem = _stem_from_name(value)
    for side, alias_patterns in SIDE_ALIAS_PATTERNS.items():
        for alias_pattern in alias_patterns:
            if _contains_side_alias(stem, alias_pattern):
                return side
    return None


def _remove_side_aliases(stem: str) -> str:
    for alias_patterns in SIDE_ALIAS_PATTERNS.values():
        for alias_pattern in alias_patterns:
            stem = re.sub(rf"(?:^|_){alias_pattern}(?=_|$)", "_", stem)
    return re.sub(r"__+", "_", stem).strip("_")


def get_pair_key(file: Path) -> str:
    """
    Build a shared key for matching day/night RGB videos from the same scene.
    """
    stem = file.stem.lower()
    stem = _remove_side_aliases(stem)
    return (file.parent / stem).as_posix()


def is_modality_file(file: Path, modality: str) -> bool:
    """
    Return True when a media filename has the modality as its own underscore token.
    This avoids false IR matches in names like walk_upstairs_day_rgb.mp4.
    """
    tokens = file.stem.lower().split("_")
    if modality.lower() != "audio" and "with_audio" in file.stem.lower():
        return False
    return modality.lower() in tokens


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


def encode_audio_to_base64(audio_path: Path) -> str:
    """
    Encode an audio file to base64 for API transmission.

    Args:
        audio_path: Path to the audio file

    Returns:
        Base64 encoded audio string, or empty string if file cannot be read
    """
    if not audio_path.exists():
        return ""
    try:
        with open(audio_path, "rb") as f:
            encoded = base64.standard_b64encode(f.read()).decode("utf-8")
            return encoded
    except Exception as e:
        print(f"ERROR: Failed to encode audio file {audio_path}: {e}")
        return ""


def build_audio_part(encoded_audio: str, mime_type: str = "audio/mp4") -> list:
    """
    Convert base64 encoded audio to Gemini API audio part.

    Args:
        encoded_audio: Base64 encoded audio string
        mime_type: MIME type of the audio (default: audio/mp4 for .m4a files)

    Returns:
        List containing a single audio Part for the API
    """
    if not encoded_audio:
        return []
    try:
        return [types.Part.from_bytes(data=base64.b64decode(encoded_audio), mime_type=mime_type)]
    except Exception as e:
        print(f"ERROR: Failed to build audio part: {e}")
        return []
