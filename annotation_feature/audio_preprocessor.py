from pathlib import Path
import re
from typing import Dict


WITH_AUDIO_STEM_RE = re.compile(r"^(?P<sample>.+)_(?P<side>day|night\d*)_(?:rgb_)?with_audio$")
RGB_SOURCE_STEM_RE = re.compile(r"^(?P<sample>.+)_(?P<side>day|night\d*)_rgb$")


def _audio_pair_key(file: Path) -> str:
    match = WITH_AUDIO_STEM_RE.match(file.stem.lower())
    stem = match.group("sample") if match else file.stem.lower()
    return str(file.parent / stem)


def _rgb_source_pair_key(file: Path) -> str:
    match = RGB_SOURCE_STEM_RE.match(file.stem.lower())
    stem = match.group("sample") if match else file.stem.lower()
    return str(file.parent / stem)


def preprocess_audio(dataset_folder: Path) -> Dict[str, Path]:
    """
    Discover and organize video/audio media files ending in "with_audio".

    Prefer night media when both day and night versions exist.
    
    Args:
        dataset_folder: Path to the dataset folder containing media files
        
    Returns:
        Dictionary mapping pair keys to selected with-audio media file paths
    """
    media_extensions = {
        ".m4a",
        ".mp3",
        ".wav",
        ".aac",
        ".flac",
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".wmv",
        ".flv",
        ".mpeg",
        ".mpg",
    }
    candidates: Dict[str, Dict[str, Path]] = {}
    rgb_source_pairs = set()
    
    for file in dataset_folder.rglob("*"):
        if not file.is_file() or file.suffix.lower() not in media_extensions:
            continue
        
        name = file.name.lower()

        if "rgb" in name and "with_audio" not in name:
            if RGB_SOURCE_STEM_RE.match(file.stem.lower()):
                rgb_source_pairs.add(_rgb_source_pair_key(file))

        if "with_audio" not in file.stem.lower():
            continue

        stem_match = WITH_AUDIO_STEM_RE.match(file.stem.lower())
        if not stem_match:
            continue

        pair_key = _audio_pair_key(file)
        side = "night" if stem_match.group("side").startswith("night") else "day"
        candidates.setdefault(pair_key, {})

        if side in candidates[pair_key]:
            print(f"WARNING: Multiple {side} with-audio files found for {pair_key}, using first one")
            continue

        candidates[pair_key][side] = file

    audio_pairs: Dict[str, Path] = {}
    for pair_key, sides in candidates.items():
        selected = sides.get("night") or sides.get("day")
        if selected is None:
            continue
        audio_pairs[pair_key] = selected
        print(f"Found with-audio media: {pair_key} -> {selected.name}")

    for pair_key in sorted(rgb_source_pairs - set(audio_pairs.keys())):
        print(f"WARNING: No with-audio media found for RGB source pair {pair_key}")
    
    return audio_pairs
