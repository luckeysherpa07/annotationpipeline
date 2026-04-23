from pathlib import Path
from typing import Dict


def preprocess_audio(dataset_folder: Path) -> Dict[str, Path]:
    """
    Discover and organize audio files in the dataset.
    
    Since day and night audio are equivalent, only use night audio files.
    
    Args:
        dataset_folder: Path to the dataset folder containing audio files
        
    Returns:
        Dictionary mapping pair keys (e.g., "wash_hands") to their night audio file paths
    """
    audio_extensions = {".m4a", ".mp3", ".wav", ".aac", ".flac"}
    audio_pairs = {}
    
    for file in dataset_folder.rglob("*"):
        if not file.is_file() or file.suffix.lower() not in audio_extensions:
            continue
        
        name = file.name.lower()
        
        # Only process night audio files
        if "_night" not in name:
            continue
        
        # Build pair key by removing _night suffix
        pair_key = file.stem.lower().replace("_night", "")
        
        # Verify this is an audio file (not a different modality)
        if any(modality in name for modality in ["_depth", "_event", "_ir", "_rgb"]):
            continue
        
        if pair_key in audio_pairs:
            print(f"WARNING: Multiple night audio files found for {pair_key}, using first one")
            continue
        
        audio_pairs[pair_key] = file
        print(f"Found audio: {pair_key} -> {file.name}")
    
    return audio_pairs
