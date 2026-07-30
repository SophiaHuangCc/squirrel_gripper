# import os
# import random
# from pathlib import Path

# from moviepy.editor import VideoFileClip, clips_array

# from PIL import Image

# # Fix for Pillow >= 10
# if not hasattr(Image, "ANTIALIAS"):
#     Image.ANTIALIAS = Image.Resampling.LANCZOS


# VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


# def find_videos(folder):
#     files = []
#     for root, _, filenames in os.walk(folder):
#         for f in filenames:
#             if Path(f).suffix.lower() in VIDEO_EXTS:
#                 files.append(os.path.join(root, f))
#     return sorted(files)


# def center_crop_clip(clip, crop_w, crop_h):
#     w, h = clip.size
#     x1 = max(0, int((w - crop_w) / 2))
#     y1 = max(0, int((h - crop_h) / 2))
#     x2 = x1 + crop_w
#     y2 = y1 + crop_h
#     return clip.crop(x1=x1, y1=y1, x2=x2, y2=y2)


# def main():
#     input_dir = "/home/real/Desktop/SquirrelGripper/ws/squirrel_gripper/optimizer_test_run/disturbance_contact_cylinder/finger_0"
#     output_path = os.path.join(input_dir, "disturbance_contact_0.gif")

#     videos = find_videos(input_dir)

#     if len(videos) < 1:
#         raise ValueError(f"Need at least 8 videos, found {len(videos)}")

#     chosen = random.sample(videos, 1)

#     print("Selected videos:")
#     for v in chosen:
#         print("  ", os.path.basename(v))

#     clips = [VideoFileClip(v) for v in chosen]

#     # ---- keep original FPS ----
#     fps = clips[0].fps

#     # ---- match duration ----
#     min_duration = min(c.duration for c in clips)
#     clips = [c.subclip(0, min_duration) for c in clips]

#     # ---- crop (NO resizing) ----
#     min_w = min(c.w for c in clips)
#     min_h = min(c.h for c in clips)

#     clips = [center_crop_clip(c, 320, 240) for c in clips]

#     # ---- 4 x 2 grid ----
#     grid = clips_array([
#         clips[:4],
#         clips[4:]
#     ])

#     # ---- export GIF ----
#     grid.write_gif(
#         output_path,
#         fps=fps,
#         program="ffmpeg"  # better quality than default
#     )

#     print(f"\nSaved GIF to: {output_path}")

#     for c in clips:
#         c.close()


# if __name__ == "__main__":
#     main()

import os
from moviepy.editor import VideoFileClip
from PIL import Image

# Fix for Pillow >= 10
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.Resampling.LANCZOS


def center_crop_clip(clip, crop_w, crop_h):
    w, h = clip.size
    crop_w = min(crop_w, w)
    crop_h = min(crop_h, h)

    x1 = max(0, int((w - crop_w) / 2))
    y1 = max(0, int((h - crop_h) / 2))
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    return clip.crop(x1=x1, y1=y1, x2=x2, y2=y2)


def mp4_to_cropped_gif(video_path, output_path=None, crop_w=320, crop_h=240, fps=None):
    video_path = os.path.abspath(video_path)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if output_path is None:
        output_path = os.path.splitext(video_path)[0] + "_cropped.gif"

    clip = VideoFileClip(video_path)

    if fps is None:
        fps = clip.fps

    cropped = center_crop_clip(clip, crop_w, crop_h)

    cropped.write_gif(
        output_path,
        fps=fps,
        program="ffmpeg"
    )

    cropped.close()
    clip.close()

    print(f"Saved GIF to: {output_path}")


if __name__ == "__main__":
    video_path = "/Users/sophiahuang/Desktop/SquirrelGripper/docs/output_20260728_225016_opt_0.mp4"

    mp4_to_cropped_gif(
        video_path,
        crop_w=320,
        crop_h=240,
        fps=None,   # lower fps makes smaller GIF; set None to use original
    )