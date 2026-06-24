#!/usr/bin/env python3
"""Create an MP4/GIF collage from one aligned multimodal segment directory."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


DEFAULT_SEGMENT = Path("aligned_dataset/cut_carrot_split/Seg1")
DEFAULT_MODALITIES = ("rgb", "ir", "event", "depth")


def even(value: float) -> int:
    return max(2, round(value / 2) * 2)


def ffmpeg_filter_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace(":", "\\:")


def discover_prefix(segment_dir: Path) -> str:
    matches = sorted(segment_dir.glob("*_day_rgb.mp4"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one '*_day_rgb.mp4' in {segment_dir}, found {len(matches)}."
        )
    return matches[0].name.removesuffix("_day_rgb.mp4")


def video_path(segment_dir: Path, prefix: str, condition: str, modality: str) -> Path:
    path = segment_dir / f"{prefix}_{condition}_{modality}.mp4"
    if not path.exists():
        raise FileNotFoundError(f"Missing input video: {path}")
    return path


def grid_shape(condition_count: int, modality_count: int) -> tuple[int, int]:
    if condition_count == 2:
        return modality_count, 2
    if modality_count == 4:
        return 2, 2
    return modality_count, 1


def build_filter(
    inputs: list[tuple[str, str]],
    columns: int,
    tile_width: int,
    tile_height: int,
    fps: int,
    font_file: Path,
    show_condition_in_first_cell: bool,
) -> str:
    filters: list[str] = []
    font = ffmpeg_filter_path(font_file)
    for index, (condition, modality) in enumerate(inputs):
        label = modality.upper()
        if show_condition_in_first_cell and index % columns == 0:
            label = f"{condition.upper()} / {label}"
        filters.append(
            f"[{index}:v]fps={fps},"
            f"scale={tile_width}:{tile_height + 2}:force_original_aspect_ratio=increase,"
            f"crop={tile_width}:{tile_height},setsar=1,"
            "drawbox=x=0:y=0:w=iw:h=ih:color=white@0.65:t=2,"
            f"drawtext=fontfile='{font}':text='{label}':x=14:y=12:fontsize=25:"
            "fontcolor=white:box=1:boxcolor=black@0.60:boxborderw=7"
            f"[v{index}]"
        )

    labels = "".join(f"[v{index}]" for index in range(len(inputs)))
    layout = "|".join(
        f"{(index % columns) * tile_width}_{(index // columns) * tile_height}"
        for index in range(len(inputs))
    )
    filters.append(
        f"{labels}xstack=inputs={len(inputs)}:layout={layout}:fill=black,format=yuv420p[out]"
    )
    return ";".join(filters)


def run(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment-dir", type=Path, default=DEFAULT_SEGMENT)
    parser.add_argument(
        "--conditions", nargs="+", choices=("day", "night"), default=["day", "night"],
        help="One condition creates a 2x2 four-modality view; both create a 2x4 view.",
    )
    parser.add_argument(
        "--modalities", nargs="+", choices=DEFAULT_MODALITIES, default=list(DEFAULT_MODALITIES)
    )
    parser.add_argument("--day-start", type=float, default=0.0)
    parser.add_argument("--night-start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--fps", type=int, default=12, help="MP4 frame rate.")
    parser.add_argument("--gif-fps", type=int, default=8)
    parser.add_argument("--tile-width", type=int, default=None)
    parser.add_argument("--gif-width", type=int, default=1280)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/visualizations"))
    parser.add_argument("--name", help="Output basename without an extension.")
    parser.add_argument("--skip-gif", action="store_true")
    args = parser.parse_args()

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg and ffprobe must be available on PATH.")
    if args.duration <= 0 or args.fps <= 0 or args.gif_fps <= 0:
        raise ValueError("Duration and frame rates must be positive.")

    segment_dir = args.segment_dir.resolve()
    prefix = discover_prefix(segment_dir)
    conditions = list(dict.fromkeys(args.conditions))
    modalities = list(dict.fromkeys(args.modalities))
    columns, rows = grid_shape(len(conditions), len(modalities))
    tile_width = args.tile_width or (400 if len(conditions) == 2 else 640)
    tile_height = even(tile_width * 9 / 16)

    ordered_inputs = [(condition, modality) for condition in conditions for modality in modalities]
    starts = {"day": args.day_start, "night": args.night_start}
    ffmpeg_command = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y"]
    for condition, modality in ordered_inputs:
        ffmpeg_command.extend(
            [
                "-ss", str(starts[condition]),
                "-t", str(args.duration),
                "-i", str(video_path(segment_dir, prefix, condition, modality)),
            ]
        )

    font_file = Path("C:/Windows/Fonts/arial.ttf")
    if not font_file.exists():
        raise FileNotFoundError(f"Label font not found: {font_file}")
    filter_graph = build_filter(
        ordered_inputs,
        columns,
        tile_width,
        tile_height,
        args.fps,
        font_file,
        len(conditions) == 2,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    condition_label = "_".join(conditions)
    name = args.name or f"{prefix}_{condition_label}_multimodal"
    mp4_path = args.output_dir / f"{name}.mp4"
    gif_path = args.output_dir / f"{name}.gif"
    preview_path = args.output_dir / f"{name}_preview.jpg"

    ffmpeg_command.extend(
        [
            "-filter_complex", filter_graph,
            "-map", "[out]", "-an",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(mp4_path),
        ]
    )
    run(ffmpeg_command)

    run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
            "-ss", str(args.duration / 2), "-i", str(mp4_path),
            "-frames:v", "1", "-update", "1", str(preview_path),
        ]
    )
    if not args.skip_gif:
        gif_filter = (
            f"fps={args.gif_fps},scale={even(args.gif_width)}:-2:flags=lanczos,split[s0][s1];"
            "[s0]palettegen=max_colors=192:stats_mode=diff[p];"
            "[s1][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle"
        )
        run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
                "-i", str(mp4_path), "-filter_complex", gif_filter,
                "-loop", "0", str(gif_path),
            ]
        )

    print(f"Created MP4: {mp4_path}")
    print(f"Created preview: {preview_path}")
    if not args.skip_gif:
        print(f"Created GIF: {gif_path}")


if __name__ == "__main__":
    main()
