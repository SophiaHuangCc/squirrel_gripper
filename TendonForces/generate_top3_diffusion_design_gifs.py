#!/usr/bin/env python3
"""
Create GIFs for the top 3 diffusion-style designs from the sample_exp3
verification folders, using the same metric fields that the previous
comparison workflow extracted from the per-finger verification outputs.

This script does not run any diffusion or optimization pipeline. It simply:
  1. scans finger_* folders under the sample verification directory,
  2. reads the stored verification metrics from each finger's .npz,
  3. ranks them by a composite score,
  4. creates one GIF per top-3 design with the summary metrics overlaid.

Example:
    python generate_top3_diffusion_design_gifs.py \
        --input-dir "docs/sample_exp3/sim_verification_bending_only" \
        --output-dir "docs/sample_exp3/sim_verification_bending_only/top3_diffusion_gifs"
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_input = repo_root / "docs" / "sample_exp3" / "sim_verification_bending_only"
    default_output = default_input / "top3_diffusion_gifs"

    parser = argparse.ArgumentParser(description="Make GIFs for the top 3 diffusion-style designs")
    parser.add_argument("--input-dir", type=str, default=str(default_input), help="Folder containing finger_0 ... finger_N")
    parser.add_argument("--output-dir", type=str, default=str(default_output), help="Where GIFs are written")
    parser.add_argument("--max-sec", type=float, default=3.0, help="How many seconds of each video to use")
    return parser.parse_args()


def read_metrics(folder: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    npz_path = folder / f"{folder.name}.npz"
    if npz_path.exists():
        try:
            with np.load(npz_path, allow_pickle=True) as data:
                if "metric" in data.files:
                    item = data["metric"].item()
                    if isinstance(item, dict):
                        for key, value in item.items():
                            if isinstance(value, (int, float, np.number)):
                                metrics[str(key)] = float(value)
        except Exception:
            pass

    sweep_path = folder / "sweep_summary.csv"
    if sweep_path.exists() and not metrics:
        with sweep_path.open(newline="") as fh:
            row = next(csv.DictReader(fh), None)
            if row is not None:
                for key in ["num_contacts", "angular_span", "tension"]:
                    if key in row:
                        try:
                            metrics[key] = float(row[key])
                        except ValueError:
                            pass

    return metrics


def find_video_path(folder: Path) -> Path | None:
    for path in sorted(folder.glob("*.mp4")):
        if path.is_file():
            return path
    for path in sorted(folder.rglob("*.mp4")):
        if path.is_file():
            return path
    return None


def rank_score(metrics: Dict[str, float]) -> float:
    disturbance = float(metrics.get("disturbance_resistance_score", 0.0))
    contacts = float(metrics.get("num_contacts", 0.0))
    span = float(metrics.get("angular_span", 0.0))
    return disturbance + 0.1 * contacts + 0.5 * span


def build_caption(folder: Path, metrics: Dict[str, float]) -> str:
    lines = [folder.name]
    for label, key in [
        ("contacts", "num_contacts"),
        ("disturbance", "disturbance_resistance_score"),
        ("angular span", "angular_span"),
        ("curl time", "curl_time"),
        ("curl speed", "curl_speed_score"),
        ("combined", "combined_score"),
    ]:
        if key in metrics:
            lines.append(f"{label}: {metrics[key]:.3f}")
    return "\n".join(lines)


def print_ranked_results(entries: List[Tuple[float, Path, Dict[str, float]]], output_dir: Path) -> None:
    print("\nTop 3 diffusion-style designs:")
    for rank, (score, folder, metrics) in enumerate(entries[:3], start=1):
        print(f"\n[{rank}] {folder.name}")
        print(f"  score: {score:.4f}")
        for label, key in [
            ("contacts", "num_contacts"),
            ("disturbance", "disturbance_resistance_score"),
            ("angular span", "angular_span"),
            ("curl time", "curl_time"),
            ("curl speed", "curl_speed_score"),
            ("combined", "combined_score"),
        ]:
            if key in metrics:
                print(f"  {label}: {metrics[key]:.3f}")
        print(f"  video: {folder / (folder.name + '.mp4')}")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    entries: List[Tuple[float, Path, Dict[str, float]]] = []
    for folder in sorted(input_dir.glob("finger_*")):
        if not folder.is_dir():
            continue
        video_path = find_video_path(folder)
        if video_path is None:
            continue
        metrics = read_metrics(folder)
        entries.append((rank_score(metrics), folder, metrics))

    if not entries:
        raise RuntimeError("No finger folders with videos were found")

    entries.sort(key=lambda item: item[0], reverse=True)
    top3 = entries[:3]
    print_ranked_results(top3, output_dir)
    print(f"\nResults written to terminal only; no GIFs were generated.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
