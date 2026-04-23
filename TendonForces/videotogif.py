import os
import random
from pathlib import Path

from moviepy.editor import VideoFileClip, clips_array

from PIL import Image

# Fix for Pillow >= 10
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.Resampling.LANCZOS


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def find_videos(folder):
    files = []
    for root, _, filenames in os.walk(folder):
        for f in filenames:
            if Path(f).suffix.lower() in VIDEO_EXTS:
                files.append(os.path.join(root, f))
    return sorted(files)


def center_crop_clip(clip, crop_w, crop_h):
    w, h = clip.size
    x1 = max(0, int((w - crop_w) / 2))
    y1 = max(0, int((h - crop_h) / 2))
    x2 = x1 + crop_w
    y2 = y1 + crop_h
    return clip.crop(x1=x1, y1=y1, x2=x2, y2=y2)


def main():
    input_dir = "TendonForces/runs/exp15"
    output_path = os.path.join(input_dir, "combined_8.gif")

    videos = find_videos(input_dir)

    if len(videos) < 8:
        raise ValueError(f"Need at least 8 videos, found {len(videos)}")

    chosen = random.sample(videos, 8)

    print("Selected videos:")
    for v in chosen:
        print("  ", os.path.basename(v))

    clips = [VideoFileClip(v) for v in chosen]

    # ---- keep original FPS ----
    fps = clips[0].fps

    # ---- match duration ----
    min_duration = min(c.duration for c in clips)
    clips = [c.subclip(0, min_duration) for c in clips]

    # ---- crop (NO resizing) ----
    min_w = min(c.w for c in clips)
    min_h = min(c.h for c in clips)

    clips = [center_crop_clip(c, 320, 240) for c in clips]

    # ---- 4 x 2 grid ----
    grid = clips_array([
        clips[:4],
        clips[4:]
    ])

    # ---- export GIF ----
    grid.write_gif(
        output_path,
        fps=fps,
        program="ffmpeg"  # better quality than default
    )

    print(f"\nSaved GIF to: {output_path}")

    for c in clips:
        c.close()


if __name__ == "__main__":
    main()