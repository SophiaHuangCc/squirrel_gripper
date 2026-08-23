#!/usr/bin/env python3
"""
Create a 2x4 summary GIF montage from the sample_exp3 verification videos.

The script scans the sample_exp3/sim_verification_bending_only folder, reads the
per-finger .npz metrics for each finger_0, finger_1, ..., and builds:
  1) one annotated GIF per selected finger design, and
  2) one combined 2-row x 4-column GIF montage of the top-N designs.

Example:
    python generate_sample_exp3_finger_gifs.py \
        --input-dir "docs/sample_exp3/sim_verification_bending_only" \
        --output-dir "docs/sample_exp3/sim_verification_bending_only/summary_gifs" \
        --top-n 8
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from moviepy.editor import CompositeVideoClip, TextClip, VideoFileClip, clips_array


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_input = repo_root / "docs" / "sample_exp3" / "sim_verification_bending_only"
    default_output = default_input / "summary_gifs"

    parser = argparse.ArgumentParser(description="Build summary GIFs from sample_exp3 finger videos")
    parser.add_argument("--input-dir", type=str, default=str(default_input), help="Folder containing finger_0 ... finger_N")
    parser.add_argument("--output-dir", type=str, default=str(default_output), help="Where the GIFs are written")
    parser.add_argument("--top-n", type=int, default=8, help="Number of designs to include in the final montage")
    parser.add_argument("--rows", type=int, default=2, help="Number of montage rows")
    parser.add_argument("--cols", type=int, default=4, help="Number of montage columns")
    parser.add_argument("--max-sec", type=float, default=3.0, help="How many seconds of each video to use")
    parser.add_argument("--sort-by", type=str, default="combined_score", help="Metric to rank by")
    return parser.parse_args()


def load_finger_metrics(folder: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    npz_path = folder / f"{folder.name}.npz"

    if npz_path.exists():
        try:
            with np.load(npz_path, allow_pickle=True) as data:
                if "metric" in data.files:
                    value = data["metric"]
                    if value.shape == ():
                        item = value.item()
                        if isinstance(item, dict):
                            for key, val in item.items():
                                if isinstance(val, (int, float, np.number)):
                                    metrics[key] = float(val)
        except Exception:
            pass

    sweep_path = folder / "sweep_summary.csv"
    if sweep_path.exists() and not metrics:
        with sweep_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                for key in ["angular_span", "num_contacts"]:
                    if key in row and row[key] not in {"", None}:
                        try:
                            metrics[key] = float(row[key])
                        except ValueError:
                            pass
                if "E" in row:
                    try:
                        metrics["E"] = float(row["E"])
                    except ValueError:
                        pass
                if "tension" in row:
                    try:
                        metrics["tension"] = float(row["tension"])
                    except ValueError:
                        pass

    return metrics


def find_video_path(folder: Path) -> Path | None:
    candidates = sorted(folder.glob("*.mp4"))
    if candidates:
        return candidates[0]
    for path in sorted(folder.rglob("*.mp4")):
        if path.is_file():
            return path
    return None


def build_caption_text(folder: Path, metrics: Dict[str, float]) -> str:
    lines = [folder.name]

    def add_line(label: str, key: str, fmt: str = ".3f") -> None:
        if key in metrics:
            lines.append(f"{label}: {metrics[key]:{fmt}}")

    add_line("contacts", "num_contacts")
    add_line("disturbance", "disturbance_resistance_score")
    add_line("angular span", "angular_span")
    add_line("curl time", "curl_time")
    add_line("curl speed", "curl_speed_score")
    add_line("combined", "combined_score")
    add_line("tension", "tension")
    return "\n".join(lines)


def make_annotated_gif(video_path: Path, output_gif: Path, caption: str, max_sec: float) -> None:
    clip = VideoFileClip(str(video_path))
    duration = min(max_sec, clip.duration)
    clip = clip.subclip(0, duration)

    txt = TextClip(
        caption,
        font="Arial-Bold",
        fontsize=20,
        color="white",
        method="caption",
        size=(clip.size[0], 120),
    )
    txt = txt.set_position(("center", 0.02)).set_duration(duration)
    composite = CompositeVideoClip([clip, txt])
    composite.write_gif(str(output_gif), fps=8, program="ffmpeg")
    clip.close()
    txt.close()
    composite.close()


def build_grid_montage(clips: List[VideoFileClip], output_gif: Path, rows: int, cols: int) -> None:
    if len(clips) == 0:
        raise ValueError("No clips available for the montage")

    max_items = rows * cols
    clips = clips[:max_items]

    # Make every clip the same size and duration.
    duration = min(clip.duration for clip in clips)
    resized_clips = []
    for clip in clips:
        clip = clip.subclip(0, duration)
        clip = clip.resize(width=320)
        resized_clips.append(clip)

    rows_list = []
    for r in range(rows):
        start = r * cols
        end = start + cols
        row = resized_clips[start:end]
        if len(row) < cols:
            row = row + [None] * (cols - len(row))
        rows_list.append(row)

    grid_clip = clips_array(rows_list)
    grid_clip.write_gif(str(output_gif), fps=8, program="ffmpeg")
    grid_clip.close()
    for clip in resized_clips:
        clip.close()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    finger_dirs = sorted([p for p in input_dir.glob("finger_*") if p.is_dir()])
    if not finger_dirs:
        raise FileNotFoundError(f"No finger_* folders found under {input_dir}")

    entries: List[Tuple[float, Path, Dict[str, float]]] = []
    for folder in finger_dirs:
        metrics = load_finger_metrics(folder)
        video_path = find_video_path(folder)
        if video_path is None:
            continue

        sort_value = metrics.get(args.sort_by, -1e9)
        if args.sort_by == "combined_score" and "combined_score" not in metrics:
            sort_value = metrics.get("num_contacts", -1e9) + metrics.get("disturbance_resistance_score", -1e9) * 100
        entries.append((sort_value, folder, metrics))

    if not entries:
        raise RuntimeError("No usable finger folders with videos were found")

    entries.sort(key=lambda item: item[0], reverse=True)
    top_entries = entries[: max(1, args.top_n)]

    clipped_paths: List[VideoFileClip] = []
    for rank, (score, folder, metrics) in enumerate(top_entries, start=1):
        video_path = find_video_path(folder)
        if video_path is None:
            continue

        caption = build_caption_text(folder, metrics)
        out_gif = output_dir / f"{folder.name}_rank{rank:02d}.gif"
        make_annotated_gif(video_path, out_gif, caption, args.max_sec)
        print(f"Saved {out_gif}")

        clip = VideoFileClip(str(out_gif))
        clipped_paths.append(clip)

    if len(clipped_paths) == 0:
        raise RuntimeError("No GIFs were generated")

    montage_out = output_dir / f"top{len(clipped_paths)}_montage_{args.rows}x{args.cols}.gif"
    build_grid_montage(clipped_paths, montage_out, rows=args.rows, cols=args.cols)
    print(f"Saved combined montage: {montage_out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
