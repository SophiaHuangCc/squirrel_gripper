#!/usr/bin/env python3
"""
Select one good and one bad grasp example from a TendonForces dataset and
render their original simulated movement as GIFs.

This script is designed for presentation slides explaining evaluation metrics.
It selects examples using metrics already saved in `master_log_*.npz`:

Good grasp:
  high contact count + high angular span + high disturbance resistance

Bad grasp:
  low contact count + low angular span + low disturbance resistance

It renders directly from the stored rod `position` array, so it works even when
the dataset was generated with --data_only and no MP4 videos were saved.

Example:
  python TendonForces/select_good_bad_grasp_examples.py \
      --dataset_dir "TendonForces/runs/exp3/train" \
      --output_dir "docs/good_bad_grasp_examples"

Outputs:
  docs/good_bad_grasp_examples/
    good_grasp.gif
    bad_grasp.gif
    good_bad_comparison.gif
    good_grasp_first_frame.png
    bad_grasp_first_frame.png
    selected_examples.csv
    selected_examples.json
"""

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class GraspExample:
    path: Path
    metrics: Dict[str, float]
    good_score: float
    bad_score: float


def scalar_from_npz(data, keys: Sequence[str], default: float = 0.0) -> float:
    for key in keys:
        if key not in data:
            continue
        value = np.asarray(data[key])
        if value.size == 0:
            continue
        item = value.reshape(-1)[0]
        try:
            return float(item)
        except (TypeError, ValueError):
            return float(str(item))
    return float(default)


def array_from_npz(data, keys: Sequence[str]):
    for key in keys:
        if key in data:
            return np.asarray(data[key])
    return None


def find_npz_files(dataset_dir: Path) -> List[Path]:
    files = sorted(dataset_dir.rglob("*.npz"))
    master_logs = [p for p in files if p.name.startswith("master_log_")]
    return master_logs if master_logs else files


def normalize_contact(num_contacts: float, n_elements: float) -> float:
    return float(np.clip(np.log1p(num_contacts) / np.log1p(max(n_elements, 1.0)), 0.0, 1.0))


def normalize_span(angular_span: float) -> float:
    return float(np.clip(angular_span / 180.0, 0.0, 1.0))


def load_metrics(path: Path) -> GraspExample:
    with np.load(path, allow_pickle=True) as data:
        n_elements = scalar_from_npz(data, ["n_elements", "arg_n_elements"], 100.0)
        num_contacts = scalar_from_npz(data, ["num_contacts"], 0.0)
        angular_span = scalar_from_npz(data, ["angular_span"], 0.0)
        disturbance = scalar_from_npz(data, ["disturbance_resistance_score"], 0.0)
        approach_deg = scalar_from_npz(data, ["arg_approach_deg", "approach_deg"], 45.0)
        cyl_radius = scalar_from_npz(data, ["cyl_radius", "arg_cyl_rad"], 0.03)

    contact_norm = normalize_contact(num_contacts, n_elements)
    span_norm = normalize_span(angular_span)
    disturbance_norm = float(np.clip(disturbance, 0.0, 1.0))

    # Equal-weight metric score for the three metrics shown on the slide.
    good_score = contact_norm + span_norm + disturbance_norm
    bad_score = -good_score

    return GraspExample(
        path=path,
        metrics={
            "num_contacts": num_contacts,
            "contact_norm": contact_norm,
            "angular_span": angular_span,
            "angular_span_norm": span_norm,
            "disturbance_resistance_score": disturbance,
            "disturbance_norm": disturbance_norm,
            "n_elements": n_elements,
            "approach_deg": approach_deg,
            "cyl_radius": cyl_radius,
        },
        good_score=good_score,
        bad_score=bad_score,
    )


def has_motion(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=True) as data:
            pos = array_from_npz(data, ["position", "positions"])
            return pos is not None and np.asarray(pos).ndim == 3 and np.asarray(pos).shape[0] >= 2
    except Exception:
        return False


def select_examples(examples: Sequence[GraspExample]) -> Tuple[GraspExample, GraspExample]:
    usable = [ex for ex in examples if has_motion(ex.path)]
    if not usable:
        raise ValueError("No examples with usable `position` trajectories were found.")

    good = max(usable, key=lambda ex: ex.good_score)

    # Prefer genuinely bad examples: low contact + low span + low disturbance.
    # Exclude the selected good sample.
    remaining = [ex for ex in usable if ex.path != good.path]
    bad = min(remaining, key=lambda ex: ex.good_score)

    return good, bad


def parse_vertebra_nodes(data):
    arr = array_from_npz(data, ["vertebra_nodes"])
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.dtype.kind in {"U", "S", "O"}:
        text = str(arr.reshape(-1)[0])
        return np.asarray([int(x) for x in text.split(",") if x.strip()], dtype=int)
    return arr.astype(int).reshape(-1)


def cylinder_center_and_radius(data, fallback_radius: float) -> Tuple[np.ndarray, float]:
    center = array_from_npz(data, ["cyl_position", "cylinder_position"])
    if center is None:
        center_arr = np.asarray([0.0, 0.0, 0.0], dtype=float)
    else:
        center_arr = np.asarray(center, dtype=float)
        if center_arr.ndim >= 2:
            center_arr = center_arr.reshape(3, -1)[:, 0]
        else:
            center_arr = center_arr.reshape(-1)[:3]
    radius = scalar_from_npz(data, ["cyl_radius", "arg_cyl_rad"], fallback_radius)
    return center_arr, radius


def get_times(data, n_frames: int, final_time: float) -> np.ndarray:
    times = array_from_npz(data, ["time"])
    if times is not None:
        times = np.asarray(times, dtype=float).reshape(-1)
        if len(times) == n_frames:
            return times
    return np.linspace(0.0, final_time, n_frames)


def sample_frame_indices(total_frames: int, num_frames: int) -> np.ndarray:
    if total_frames <= num_frames:
        return np.arange(total_frames)
    return np.linspace(0, total_frames - 1, num_frames).round().astype(int)


def set_equal_aspect_xz(ax, xs: Sequence[float], zs: Sequence[float], pad: float = 0.02) -> None:
    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    z_min, z_max = float(np.min(zs)), float(np.max(zs))
    x_mid = 0.5 * (x_min + x_max)
    z_mid = 0.5 * (z_min + z_max)
    span = max(x_max - x_min, z_max - z_min, 1e-3) + 2 * pad
    ax.set_xlim(x_mid - span / 2, x_mid + span / 2)
    ax.set_ylim(z_mid - span / 2, z_mid + span / 2)
    ax.set_aspect("equal", adjustable="box")


def fig_to_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return rgba[:, :, :3].copy()


def render_frame(
    pos: np.ndarray,
    center: np.ndarray,
    cyl_radius: float,
    base_radius: float,
    vertebra_nodes,
    title: str,
    label: str,
    axis_bounds: Tuple[float, float, float, float],
) -> np.ndarray:
    x = pos[0]
    z = pos[2]

    fig, ax = plt.subplots(figsize=(5.2, 4.0), dpi=120)

    circle = plt.Circle(
        (center[0], center[2]),
        cyl_radius,
        fill=False,
        color="black",
        linewidth=2.5,
    )
    ax.add_patch(circle)

    radial = np.sqrt((x - center[0]) ** 2 + (z - center[2]) ** 2)
    contact_idx = np.where(radial <= cyl_radius + base_radius)[0]

    ax.plot(x, z, "-", color="#1f77b4", linewidth=3)
    ax.scatter(x[0], z[0], s=45, color="black", zorder=5)
    ax.scatter(x[-1], z[-1], s=45, color="#1f77b4", zorder=5)

    if contact_idx.size:
        ax.scatter(x[contact_idx], z[contact_idx], s=35, color="#ff7f0e", zorder=6, label="contact")

    if vertebra_nodes is not None and len(vertebra_nodes):
        valid = np.clip(np.asarray(vertebra_nodes, dtype=int), 0, len(x) - 1)
        ax.scatter(x[valid], z[valid], s=55, color="#b22222", zorder=7, label="vertebrae")

    ax.set_title(title, fontsize=12)
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox=dict(facecolor="white", alpha=0.86, edgecolor="none"),
    )

    ax.set_xlim(axis_bounds[0], axis_bounds[1])
    ax.set_ylim(axis_bounds[2], axis_bounds[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.22)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    fig.tight_layout()
    img = fig_to_rgb(fig)
    plt.close(fig)
    return img


def compute_axis_bounds(positions: np.ndarray, center: np.ndarray, cyl_radius: float, pad: float = 0.025):
    x = positions[:, 0, :].reshape(-1)
    z = positions[:, 2, :].reshape(-1)
    xs = np.concatenate([x, [center[0] - cyl_radius, center[0] + cyl_radius]])
    zs = np.concatenate([z, [center[2] - cyl_radius, center[2] + cyl_radius]])
    x_min, x_max = float(xs.min()), float(xs.max())
    z_min, z_max = float(zs.min()), float(zs.max())
    x_mid = 0.5 * (x_min + x_max)
    z_mid = 0.5 * (z_min + z_max)
    span = max(x_max - x_min, z_max - z_min, 1e-3) + 2 * pad
    return (
        x_mid - span / 2,
        x_mid + span / 2,
        z_mid - span / 2,
        z_mid + span / 2,
    )


def render_example_gif(
    example: GraspExample,
    output_path: Path,
    first_frame_path: Path,
    title: str,
    fps: int = 8,
    num_frames: int = 48,
    max_duration: float = 5.0,
) -> List[np.ndarray]:
    with np.load(example.path, allow_pickle=True) as data:
        positions = np.asarray(array_from_npz(data, ["position", "positions"]), dtype=float)
        center, cyl_radius = cylinder_center_and_radius(data, example.metrics["cyl_radius"])
        vertebra_nodes = parse_vertebra_nodes(data)
        base_radius = scalar_from_npz(data, ["base_radius", "arg_base_rad"], 0.005)
        final_time = scalar_from_npz(data, ["arg_final_time", "final_time"], max_duration)
        times = get_times(data, positions.shape[0], final_time)

    valid_until = np.searchsorted(times, min(max_duration, float(times[-1])), side="right")
    positions = positions[: max(valid_until, 2)]
    times = times[: positions.shape[0]]

    frame_indices = sample_frame_indices(positions.shape[0], num_frames)
    axis_bounds = compute_axis_bounds(positions, center, cyl_radius)

    metric_label = (
        f"contacts={example.metrics['num_contacts']:.0f}\n"
        f"span={example.metrics['angular_span']:.1f}°\n"
        f"disturbance={example.metrics['disturbance_resistance_score']:.2f}\n"
        f"t={0:.2f}s"
    )
    first_img = render_frame(
        positions[frame_indices[0]],
        center,
        cyl_radius,
        base_radius,
        vertebra_nodes,
        title,
        metric_label,
        axis_bounds,
    )
    imageio.imwrite(first_frame_path, first_img)

    frames = []
    for frame_idx in frame_indices:
        label = (
            f"contacts={example.metrics['num_contacts']:.0f}\n"
            f"span={example.metrics['angular_span']:.1f}°\n"
            f"disturbance={example.metrics['disturbance_resistance_score']:.2f}\n"
            f"t={times[frame_idx]:.2f}s"
        )
        frames.append(
            render_frame(
                positions[frame_idx],
                center,
                cyl_radius,
                base_radius,
                vertebra_nodes,
                title,
                label,
                axis_bounds,
            )
        )

    imageio.mimsave(output_path, frames, fps=fps)
    return frames


def combine_side_by_side(left_frames: List[np.ndarray], right_frames: List[np.ndarray], output_path: Path, fps: int):
    n = min(len(left_frames), len(right_frames))
    combined = []
    for i in range(n):
        left = left_frames[i]
        right = right_frames[i]
        if left.shape[0] != right.shape[0]:
            h = min(left.shape[0], right.shape[0])
            left = left[:h]
            right = right[:h]
        combined.append(np.concatenate([left, right], axis=1))
    imageio.mimsave(output_path, combined, fps=fps)


def write_summaries(output_dir: Path, good: GraspExample, bad: GraspExample) -> None:
    rows = [("good", good), ("bad", bad)]

    with (output_dir / "selected_examples.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        metric_keys = list(good.metrics.keys())
        writer.writerow(["label", "good_score", "npz_path", *metric_keys])
        for label, ex in rows:
            writer.writerow([label, ex.good_score, str(ex.path), *[ex.metrics[k] for k in metric_keys]])

    payload = {
        label: {
            "npz_path": str(ex.path),
            "good_score": ex.good_score,
            "metrics": ex.metrics,
        }
        for label, ex in rows
    }
    (output_dir / "selected_examples.json").write_text(json.dumps(payload, indent=2))


def print_metric_ranges(examples: Sequence[GraspExample]) -> None:
    print("[METRIC RANGES]")
    for key in ["num_contacts", "angular_span", "disturbance_resistance_score"]:
        arr = np.asarray([ex.metrics[key] for ex in examples], dtype=float)
        print(
            f"  {key}: min={np.nanmin(arr):.4g}, "
            f"mean={np.nanmean(arr):.4g}, max={np.nanmax(arr):.4g}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Select good/bad grasp examples from npz metrics and render movement GIFs."
    )
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="docs/good_bad_grasp_examples")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--num_frames", type=int, default=48)
    parser.add_argument("--max_duration", type=float, default=5.0)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = find_npz_files(dataset_dir)
    if not files:
        raise FileNotFoundError(f"No npz files found under: {dataset_dir}")

    examples = [load_metrics(path) for path in files]
    print(f"[FOUND] {len(examples)} npz files")
    print_metric_ranges(examples)

    good, bad = select_examples(examples)
    print("\n[SELECTED GOOD]")
    print(f"  {good.path}")
    print(f"  score={good.good_score:.4f} metrics={good.metrics}")
    print("\n[SELECTED BAD]")
    print(f"  {bad.path}")
    print(f"  score={bad.good_score:.4f} metrics={bad.metrics}")

    good_frames = render_example_gif(
        good,
        output_path=output_dir / "good_grasp.gif",
        first_frame_path=output_dir / "good_grasp_first_frame.png",
        title="Good grasp: high contact/span/disturbance",
        fps=args.fps,
        num_frames=args.num_frames,
        max_duration=args.max_duration,
    )
    bad_frames = render_example_gif(
        bad,
        output_path=output_dir / "bad_grasp.gif",
        first_frame_path=output_dir / "bad_grasp_first_frame.png",
        title="Bad grasp: low contact/span/disturbance",
        fps=args.fps,
        num_frames=args.num_frames,
        max_duration=args.max_duration,
    )
    combine_side_by_side(
        good_frames,
        bad_frames,
        output_path=output_dir / "good_bad_comparison.gif",
        fps=args.fps,
    )
    write_summaries(output_dir, good, bad)

    print(f"\n[DONE] Wrote outputs to: {output_dir}")
    print(f"  good: {output_dir / 'good_grasp.gif'}")
    print(f"  bad:  {output_dir / 'bad_grasp.gif'}")
    print(f"  side-by-side: {output_dir / 'good_bad_comparison.gif'}")


if __name__ == "__main__":
    main()
