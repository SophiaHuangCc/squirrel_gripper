#!/usr/bin/env python3
"""
Select diverse task / initial-condition examples from a TendonForces dataset.

This is meant for presentation slides. It reads simulation `master_log_*.npz`
files, selects examples that are different in task / initialization variables,
and saves the first simulation frame as PNG images.

Default variables used for diversity:
  - approach angle
  - cylinder radius
  - landing height
  - landing speed
  - initial x gap

Example:
  python TendonForces/select_task_init_examples.py \
      --dataset_dir "TendonForces/runs/exp3/train" \
      --output_dir "docs/task_init_examples" \
      --n 4

Outputs:
  docs/task_init_examples/
    example_00.png
    example_01.png
    ...
    task_init_examples_grid.png
    selected_examples.csv
    selected_examples.json
"""

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


TASK_INIT_KEYS = [
    "approach_deg",
    "cyl_radius",
    "landing_height",
    "landing_speed",
    "initial_x_gap",
]


@dataclass
class Example:
    path: Path
    values: Dict[str, float]
    score: float = 0.0


def scalar_from_npz(data, keys: Sequence[str], default: float = float("nan")) -> float:
    """Read first available scalar from an npz using candidate key names."""
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
    # Prefer full master logs over helper summaries if both exist.
    master_logs = [p for p in files if p.name.startswith("master_log_")]
    return master_logs if master_logs else files


def read_example(path: Path) -> Example:
    with np.load(path, allow_pickle=True) as data:
        values = {
            "approach_deg": scalar_from_npz(data, ["arg_approach_deg", "approach_deg"], 45.0),
            "cyl_radius": scalar_from_npz(data, ["cyl_radius", "arg_cyl_rad"], 0.03),
            "landing_height": scalar_from_npz(data, ["arg_landing_height", "landing_height"], 0.0),
            "landing_speed": scalar_from_npz(data, ["arg_landing_speed", "landing_speed"], 0.0),
            "initial_x_gap": scalar_from_npz(data, ["arg_initial_x_gap", "initial_x_gap"], 0.0),
        }
    return Example(path=path, values=values)


def finite_range(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return 0.0, 1.0
    lo = float(arr.min())
    hi = float(arr.max())
    if abs(hi - lo) < 1e-12:
        hi = lo + 1.0
    return lo, hi


def normalized_vector(example: Example, ranges: Dict[str, Tuple[float, float]]) -> np.ndarray:
    vec = []
    for key in TASK_INIT_KEYS:
        lo, hi = ranges[key]
        value = example.values.get(key, float("nan"))
        if not np.isfinite(value):
            value = lo
        vec.append((value - lo) / (hi - lo))
    return np.asarray(vec, dtype=float)


def diversity_distance(a: Example, b: Example, ranges: Dict[str, Tuple[float, float]]) -> float:
    va = normalized_vector(a, ranges)
    vb = normalized_vector(b, ranges)
    return float(np.linalg.norm(va - vb))


def unique_signature(example: Example, ndigits: int = 5) -> Tuple[float, ...]:
    return tuple(round(float(example.values[key]), ndigits) for key in TASK_INIT_KEYS)


def select_diverse_examples(examples: List[Example], n: int) -> List[Example]:
    """
    Greedy farthest-point selection over task/init variables.

    First, collapse exact duplicate task/init signatures so we do not select
    repeated initial conditions. Then:
      1. start from the lowest approach angle example,
      2. repeatedly add the example farthest from the current selected set.
    """
    if n <= 0:
        raise ValueError("--n must be positive")
    if not examples:
        raise ValueError("No examples available.")

    # Keep one representative per unique task/init signature.
    by_signature: Dict[Tuple[float, ...], Example] = {}
    for ex in examples:
        by_signature.setdefault(unique_signature(ex), ex)
    unique_examples = list(by_signature.values())

    if len(unique_examples) < n:
        print(
            f"[WARN] Requested n={n}, but only found {len(unique_examples)} unique "
            "task/init combinations. Using all unique combinations."
        )
        n = len(unique_examples)

    ranges = {
        key: finite_range([ex.values[key] for ex in unique_examples])
        for key in TASK_INIT_KEYS
    }

    # Start from smallest approach, then smallest radius/gap as tie breakers.
    selected = [
        min(
            unique_examples,
            key=lambda ex: (
                ex.values["approach_deg"],
                ex.values["cyl_radius"],
                ex.values["landing_height"],
                ex.values["initial_x_gap"],
            ),
        )
    ]
    remaining = [ex for ex in unique_examples if ex is not selected[0]]

    while len(selected) < n and remaining:
        best = max(
            remaining,
            key=lambda ex: min(diversity_distance(ex, s, ranges) for s in selected),
        )
        selected.append(best)
        remaining.remove(best)

    return selected


def parse_vertebra_nodes(data) -> Optional[np.ndarray]:
    arr = array_from_npz(data, ["vertebra_nodes"])
    if arr is None:
        return None
    if arr.dtype.kind in {"U", "S", "O"}:
        text = str(arr.reshape(-1)[0])
        return np.asarray([int(x) for x in text.split(",") if x.strip()], dtype=int)
    return np.asarray(arr, dtype=int).reshape(-1)


def get_initial_positions(data) -> np.ndarray:
    pos = array_from_npz(data, ["position", "positions"])
    if pos is None:
        raise KeyError("No `position` array found in npz.")
    pos = np.asarray(pos)
    if pos.ndim != 3:
        raise ValueError(f"Expected position shape (T, 3, N), got {pos.shape}")
    return pos[0]


def cylinder_center_and_radius(data, fallback_radius: float) -> Tuple[np.ndarray, float]:
    center = array_from_npz(data, ["cyl_position", "cylinder_position"])
    if center is None:
        # Reasonable visual fallback; the exact value should normally be logged.
        center_arr = np.asarray([0.0, 0.0, 0.0], dtype=float)
    else:
        center_arr = np.asarray(center, dtype=float)
        if center_arr.ndim >= 2:
            center_arr = center_arr.reshape(3, -1)[:, 0]
        else:
            center_arr = center_arr.reshape(-1)[:3]
    radius = scalar_from_npz(data, ["cyl_radius", "arg_cyl_rad"], fallback_radius)
    return center_arr, radius


def set_equal_aspect_xz(ax, xs: Sequence[float], zs: Sequence[float], pad: float = 0.02) -> None:
    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    z_min, z_max = float(np.min(zs)), float(np.max(zs))
    x_mid = 0.5 * (x_min + x_max)
    z_mid = 0.5 * (z_min + z_max)
    span = max(x_max - x_min, z_max - z_min, 1e-3) + 2 * pad
    ax.set_xlim(x_mid - span / 2, x_mid + span / 2)
    ax.set_ylim(z_mid - span / 2, z_mid + span / 2)
    ax.set_aspect("equal", adjustable="box")


def render_first_frame_png(example: Example, output_path: Path, dpi: int = 180) -> None:
    with np.load(example.path, allow_pickle=True) as data:
        pos0 = get_initial_positions(data)
        center, cyl_radius = cylinder_center_and_radius(data, example.values["cyl_radius"])
        vertebra_nodes = parse_vertebra_nodes(data)

    x = pos0[0]
    z = pos0[2]

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot(x, z, "-", color="#1f77b4", linewidth=3, label="finger centerline")
    ax.scatter(x[0], z[0], s=45, color="black", zorder=4, label="base")
    ax.scatter(x[-1], z[-1], s=45, color="#1f77b4", zorder=4, label="tip")

    if vertebra_nodes is not None and vertebra_nodes.size:
        valid = np.clip(vertebra_nodes, 0, len(x) - 1)
        ax.scatter(x[valid], z[valid], s=60, color="#b22222", zorder=5, label="vertebrae")

    circle = plt.Circle(
        (center[0], center[2]),
        cyl_radius,
        fill=False,
        color="black",
        linewidth=2.5,
        label="cylinder",
    )
    ax.add_patch(circle)

    label = (
        f"approach={example.values['approach_deg']:.0f}°\n"
        f"cyl R={example.values['cyl_radius']:.3f} m\n"
        f"h={example.values['landing_height']:.3f} m, "
        f"v={example.values['landing_speed']:.2f} m/s\n"
        f"x_gap={example.values['initial_x_gap']:.3f} m"
    )
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
    )

    xs = np.concatenate([x, [center[0] - cyl_radius, center[0] + cyl_radius]])
    zs = np.concatenate([z, [center[2] - cyl_radius, center[2] + cyl_radius]])
    set_equal_aspect_xz(ax, xs, zs)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title("Initial frame")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def write_summary_csv(path: Path, selected: Sequence[Example]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", *TASK_INIT_KEYS, "npz_path"])
        for idx, ex in enumerate(selected):
            writer.writerow([idx, *[ex.values[k] for k in TASK_INIT_KEYS], str(ex.path)])


def write_summary_json(path: Path, selected: Sequence[Example]) -> None:
    payload = []
    for idx, ex in enumerate(selected):
        payload.append(
            {
                "idx": idx,
                "values": ex.values,
                "npz_path": str(ex.path),
            }
        )
    path.write_text(json.dumps(payload, indent=2))


def make_grid(image_paths: Sequence[Path], output_path: Path, cols: int = 4, dpi: int = 180) -> None:
    if not image_paths:
        return
    rows = int(math.ceil(len(image_paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows))
    axes_arr = np.asarray(axes).reshape(rows, cols)

    for ax in axes_arr.reshape(-1):
        ax.axis("off")

    for idx, image_path in enumerate(image_paths):
        ax = axes_arr[idx // cols, idx % cols]
        img = plt.imread(image_path)
        ax.imshow(img)
        ax.axis("off")

    fig.tight_layout(pad=0.2)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def print_available_variation(examples: Sequence[Example]) -> None:
    print("[AVAILABLE TASK/INIT VALUES]")
    for key in TASK_INIT_KEYS:
        vals = sorted({round(float(ex.values[key]), 6) for ex in examples if np.isfinite(ex.values[key])})
        preview = vals[:12]
        suffix = "" if len(vals) <= 12 else f" ... ({len(vals)} unique)"
        print(f"  {key}: {preview}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select diverse task / initial-condition examples from TendonForces "
            "npz logs and render first-frame PNGs for presentation slides."
        )
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Full or relative path to dataset directory, e.g. TendonForces/runs/exp3/train.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="docs/task_init_examples",
        help="Directory for PNGs and summaries.",
    )
    parser.add_argument("--n", type=int, default=4, help="Number of examples to select.")
    parser.add_argument("--grid_cols", type=int, default=4, help="Columns in output grid PNG.")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = find_npz_files(dataset_dir)
    if not files:
        raise FileNotFoundError(f"No npz files found under: {dataset_dir}")

    examples = [read_example(path) for path in files]
    print(f"[FOUND] {len(examples)} npz files")
    print_available_variation(examples)

    selected = select_diverse_examples(examples, args.n)
    print("\n[SELECTED]")
    image_paths = []
    for idx, ex in enumerate(selected):
        print(
            f"  {idx}: "
            + ", ".join(f"{k}={ex.values[k]:.6g}" for k in TASK_INIT_KEYS)
            + f" | {ex.path.name}"
        )
        out_png = output_dir / f"example_{idx:02d}.png"
        render_first_frame_png(ex, out_png, dpi=args.dpi)
        image_paths.append(out_png)

    write_summary_csv(output_dir / "selected_examples.csv", selected)
    write_summary_json(output_dir / "selected_examples.json", selected)
    make_grid(image_paths, output_dir / "task_init_examples_grid.png", cols=args.grid_cols, dpi=args.dpi)

    print(f"\n[DONE] Wrote outputs to: {output_dir}")
    print(f"  grid: {output_dir / 'task_init_examples_grid.png'}")
    print(f"  csv:  {output_dir / 'selected_examples.csv'}")


if __name__ == "__main__":
    main()

