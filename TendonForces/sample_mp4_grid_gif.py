#!/usr/bin/env python3
"""
Randomly sample MP4 simulation videos from a dataset folder and build one grid GIF.

Default behavior:
  - sample 32 MP4 files recursively
  - arrange them as 4 rows x 8 columns
  - save individual GIFs plus one combined GIF

Example:
  python TendonForces/sample_mp4_grid_gif.py \
      --dataset_dir "/full/path/to/TendonForces/runs/exp3" \
      --output_dir "/full/path/to/slide_gifs" \
      --seed 0

If your videos are under a verification folder:
  python TendonForces/sample_mp4_grid_gif.py \
      --dataset_dir "/full/path/to/generator/runs/sample_exp3/sim_verification_bending_only" \
      --output_dir "/full/path/to/slide_gifs"
"""

import argparse
import math
import os
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from moviepy.editor import VideoFileClip, clips_array
from PIL import Image


# Pillow >= 10 removed Image.ANTIALIAS; MoviePy 1.0.3 still expects it.
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.Resampling.LANCZOS


VIDEO_EXTS = {".mp4"}


def find_videos(dataset_dir: str) -> List[Path]:
    root = Path(dataset_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {root}")

    videos = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS
    ]
    return sorted(videos)


def center_crop_clip(clip: VideoFileClip, crop_w: int, crop_h: int):
    """Center-crop a clip to crop_w x crop_h, clamped to the clip's size."""
    w, h = clip.size
    crop_w = min(int(crop_w), int(w))
    crop_h = min(int(crop_h), int(h))
    x1 = max(0, int((w - crop_w) / 2))
    y1 = max(0, int((h - crop_h) / 2))
    x2 = x1 + crop_w
    y2 = y1 + crop_h
    return clip.crop(x1=x1, y1=y1, x2=x2, y2=y2)


def pad_sequence(seq: Sequence, target_len: int):
    """Pad a sequence with blank clips later in the grid creation step."""
    return list(seq) + [None] * max(0, target_len - len(seq))


def choose_videos(videos: Sequence[Path], n: int, seed: int) -> List[Path]:
    if n <= 0:
        raise ValueError("--n must be positive")
    if len(videos) == 0:
        raise FileNotFoundError("No MP4 files found in the dataset directory.")

    rng = random.Random(seed)
    if len(videos) < n:
        print(
            f"[WARN] Requested n={n}, but only found {len(videos)} MP4 files. "
            f"Using all available videos."
        )
        chosen = list(videos)
        rng.shuffle(chosen)
        return chosen

    return rng.sample(list(videos), n)


def make_blank_clip(width: int, height: int, duration: float):
    """Create a black silent clip used when n is smaller than rows*cols."""
    from moviepy.editor import ColorClip

    return ColorClip(size=(width, height), color=(0, 0, 0), duration=duration)


def convert_one_to_gif(
    video_path: Path,
    output_path: Path,
    crop_w: int,
    crop_h: int,
    fps: int,
    max_duration: float,
) -> None:
    """Save one cropped MP4 as an individual GIF."""
    clip = VideoFileClip(str(video_path))
    try:
        duration = min(float(clip.duration), float(max_duration))
        clip = clip.subclip(0, duration)
        clip = center_crop_clip(clip, crop_w, crop_h)
        clip.write_gif(str(output_path), fps=fps, program="ffmpeg")
    finally:
        clip.close()


def load_grid_clips(
    videos: Sequence[Path],
    crop_w: int,
    crop_h: int,
    fps: int,
    max_duration: float,
) -> Tuple[List, float]:
    """Load and standardize clips for the combined grid GIF."""
    raw_clips = [VideoFileClip(str(path)) for path in videos]
    if not raw_clips:
        raise ValueError("No clips to load.")

    duration = min(min(float(c.duration) for c in raw_clips), float(max_duration))
    grid_clips = []

    for clip in raw_clips:
        c = clip.subclip(0, duration)
        c = center_crop_clip(c, crop_w, crop_h)
        c = c.resize((crop_w, crop_h))
        grid_clips.append(c)

    return grid_clips, duration


def close_clips(clips: Iterable) -> None:
    for clip in clips:
        if clip is not None:
            clip.close()


def write_selected_list(output_path: Path, selected: Sequence[Path]) -> None:
    with output_path.open("w") as f:
        for idx, path in enumerate(selected):
            f.write(f"{idx:03d}\t{path}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample MP4 files from a dataset directory, convert each to GIF, "
            "and concatenate them into a single grid GIF for slides."
        )
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Full path to dataset/output directory containing MP4 files. Searched recursively.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to save GIFs. Defaults to <dataset_dir>/sampled_gif_grid.",
    )
    parser.add_argument("--n", type=int, default=32, help="Number of videos to sample.")
    parser.add_argument("--cols", type=int, default=8, help="Number of GIFs per row.")
    parser.add_argument("--rows", type=int, default=4, help="Number of rows in the final grid.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for repeatable sampling.")
    parser.add_argument("--fps", type=int, default=8, help="GIF frames per second.")
    parser.add_argument(
        "--max_duration",
        type=float,
        default=5.0,
        help="Maximum seconds to keep from each video.",
    )
    parser.add_argument("--crop_w", type=int, default=320, help="Per-video crop/output width.")
    parser.add_argument("--crop_h", type=int, default=240, help="Per-video crop/output height.")
    parser.add_argument(
        "--grid_name",
        type=str,
        default="sampled_grid.gif",
        help="Filename for the combined grid GIF.",
    )
    parser.add_argument(
        "--skip_individual_gifs",
        action="store_true",
        help="Only write the combined grid GIF, not the individual GIF files.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else dataset_dir / "sampled_gif_grid"
    )
    individual_dir = output_dir / "individual_gifs"
    output_dir.mkdir(parents=True, exist_ok=True)
    individual_dir.mkdir(parents=True, exist_ok=True)

    target_grid_count = args.rows * args.cols
    if args.n != target_grid_count:
        print(
            f"[INFO] n={args.n}, but rows*cols={target_grid_count}. "
            f"The grid will contain {target_grid_count} cells; empty cells are black if needed."
        )

    videos = find_videos(str(dataset_dir))
    selected = choose_videos(videos, args.n, args.seed)

    selected_list_path = output_dir / "selected_videos.txt"
    write_selected_list(selected_list_path, selected)

    print(f"[FOUND] {len(videos)} MP4 files")
    print(f"[SELECTED] {len(selected)} videos")
    print(f"[SAVED] selected list: {selected_list_path}")

    if not args.skip_individual_gifs:
        for idx, video_path in enumerate(selected):
            gif_path = individual_dir / f"{idx:03d}_{video_path.stem}.gif"
            print(f"[GIF {idx + 1}/{len(selected)}] {video_path.name} -> {gif_path.name}")
            convert_one_to_gif(
                video_path=video_path,
                output_path=gif_path,
                crop_w=args.crop_w,
                crop_h=args.crop_h,
                fps=args.fps,
                max_duration=args.max_duration,
            )

    grid_clips = []
    blanks = []
    grid = None
    try:
        grid_source_videos = selected[:target_grid_count]
        grid_clips, duration = load_grid_clips(
            videos=grid_source_videos,
            crop_w=args.crop_w,
            crop_h=args.crop_h,
            fps=args.fps,
            max_duration=args.max_duration,
        )

        padded = pad_sequence(grid_clips, target_grid_count)
        for idx, clip in enumerate(padded):
            if clip is None:
                blank = make_blank_clip(args.crop_w, args.crop_h, duration)
                blanks.append(blank)
                padded[idx] = blank

        clip_rows = [
            padded[row_idx * args.cols : (row_idx + 1) * args.cols]
            for row_idx in range(args.rows)
        ]
        grid = clips_array(clip_rows)

        grid_path = output_dir / args.grid_name
        print(f"[GRID] writing {args.rows}x{args.cols} GIF: {grid_path}")
        grid.write_gif(str(grid_path), fps=args.fps, program="ffmpeg")
        print(f"[DONE] Combined GIF saved to: {grid_path}")
    finally:
        if grid is not None:
            grid.close()
        close_clips(grid_clips)
        close_clips(blanks)


if __name__ == "__main__":
    main()

