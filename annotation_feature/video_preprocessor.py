from pathlib import Path
import json
import cv2
from typing import List, Tuple, Dict


def extract_frames(video_path: Path, fps: int = 1, output_dir: Path = None) -> List[Path]:
    """
    Extract frames from a video at the specified frames per second rate.
    
    Args:
        video_path: Path to the video file
        fps: Frames per second to extract (1 = 1 frame per second)
        output_dir: Directory to save frames. If None, creates based on video name
        
    Returns:
        List of paths to extracted frame images
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Set default output directory
    if output_dir is None:
        output_dir = video_path.parent / ".frames_cache" / video_path.stem
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30  # Default fallback
    
    frame_interval = int(video_fps / fps)  # Extract every N frames
    frame_count = 0
    extracted_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract frame at specified interval
        if frame_count % frame_interval == 0:
            frame_path = output_dir / f"frame_{frame_count:06d}.png"
            cv2.imwrite(str(frame_path), frame)
            extracted_frames.append(frame_path)
        
        frame_count += 1
    
    cap.release()
    
    return extracted_frames


def preprocess_videos(dataset_folder: Path, fps: int = 1, video_type: str = "rgb") -> Dict[str, Dict[str, List[Path]]]:
    """
    Preprocess all videos in the dataset, extracting frames and caching them.
    
    Args:
        dataset_folder: Path to the dataset folder containing videos
        fps: Frames per second to extract
        video_type: Type of video to process ("rgb" or "event")
        
    Returns:
        Dictionary mapping pair keys to their night/day frame lists
    """
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".mpeg", ".mpg"}
    cache_subdir = f".frames_cache_{video_type}" if video_type != "rgb" else ".frames_cache"
    cache_dir = dataset_folder / cache_subdir
    
    # Track which videos have been processed
    processed_videos = {}
    paired_frames = {}
    
    for file in dataset_folder.rglob("*"):
        if not file.is_file() or file.suffix.lower() not in video_extensions:
            continue
        
        name = file.name.lower()
        if video_type not in name:
            continue
        
        # Build pair key (same logic as in pipeline.py)
        from annotation_feature.pipeline.utils import get_pair_key
        pair_key = get_pair_key(file)
        
        # Check if frames already cached
        frame_output_dir = cache_dir / f"{file.stem}"
        cached_frames = sorted(frame_output_dir.glob("frame_*.png"))
        
        if cached_frames:
            # Use cached frames
            frames = cached_frames
            print(f"Using cached frames for: {file.name} ({len(frames)} frames)")
        else:
            # Extract new frames
            print(f"Extracting frames from: {file.name}")
            frames = extract_frames(file, fps=fps, output_dir=frame_output_dir)
            print(f"  Extracted {len(frames)} frames")
        
        # Store frames by pair and day/night
        if pair_key not in paired_frames:
            paired_frames[pair_key] = {"night": None, "day": None}
        
        if "night" in name:
            paired_frames[pair_key]["night"] = frames
        elif "day" in name:
            paired_frames[pair_key]["day"] = frames
    
    return paired_frames


def load_preprocessed_frames(pair_key: str, dataset_folder: Path) -> Tuple[List[Path], List[Path]]:
    """
    Load preprocessed (cached) frames for a video pair.
    
    Args:
        pair_key: The pair key for the video (e.g., 'dataset/scene_name')
        dataset_folder: Path to the dataset folder
        
    Returns:
        Tuple of (night_frames, day_frames), each as list of Path objects
    """
    cache_dir = dataset_folder / ".frames_cache"
    
    # Try to find corresponding frame directories
    night_frames = []
    day_frames = []
    
    # Search in cache for matching frames
    if cache_dir.exists():
        for frame_dir in cache_dir.iterdir():
            if frame_dir.is_dir():
                frames = sorted(frame_dir.glob("frame_*.png"))
                if frames:
                    # Heuristic: check if directory name contains "night" or "day"
                    dir_name = frame_dir.name.lower()
                    if "night" in dir_name:
                        night_frames = frames
                    elif "day" in dir_name:
                        day_frames = frames
    
    return night_frames, day_frames


def save_frame_manifest(results: Dict, dataset_folder: Path) -> None:
    """
    Save a manifest of extracted frames for audit/debugging purposes.
    
    Args:
        results: The results dictionary containing annotation data
        dataset_folder: Path to the dataset folder
    """
    manifest_path = dataset_folder.parent / "frames_manifest.json"
    
    # Convert frame paths to strings for JSON serialization
    manifest = {}
    for pair_key, data in results.items():
        if isinstance(data, dict) and "annotations" in data:
            manifest[pair_key] = {
                "night_file": data.get("night_file"),
                "day_file": data.get("day_file"),
            }
    
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print(f"Frame manifest saved to: {manifest_path}")
