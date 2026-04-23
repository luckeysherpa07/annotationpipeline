from pathlib import Path
import base64
from typing import List

try:
    from google.genai import types
except ImportError:
    types = None

video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".mpeg", ".mpg"}
audio_extensions = {".m4a", ".mp3", ".wav", ".aac", ".flac"}


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