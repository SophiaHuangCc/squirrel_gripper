"""Plot existing local pose perturbations; performs no new simulations."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(
        "outputs/pose_pilot_small_steps/pose_results.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = json.loads(args.input.read_text())
    anchors = sorted({r["anchor"] for r in rows if r["status"] == "ok"})
    if not anchors:
        raise ValueError("No successful anchors to plot")
    fig, axes = plt.subplots(len(anchors), 2, figsize=(11, 3 * len(anchors)), squeeze=False)
    for axrow, anchor in zip(axes, anchors):
        group = [r for r in rows if r["anchor"] == anchor and r["status"] == "ok"]
        base = next(r for r in group if r["variant"] == "base")
        for ax, field, label in zip(axrow,
                ("predicted_loss_m2", "simulated_loss_m2"), ("Predicted", "Simulator")):
            for variant, color in (("descent", "C0"), ("random", "C1")):
                points = [(0., 0.)]
                for row in group:
                    if row["variant"] == variant or (variant == "descent" and row["variant"] == "reverse"):
                        sign = -1 if row["variant"] == "reverse" else 1
                        points.append((sign * row["step"], (row[field] - base[field]) * 1e6))
                points.sort()
                ax.plot(*zip(*points), "o-", color=color, label=variant)
            ax.axhline(0, color="gray", linewidth=.7)
            ax.set_title(f"{base['source_method']} / {anchor[:8]} — {label}")
            ax.set_xlabel("Signed requested step in normalized design coordinates")
            ax.set_ylabel("Change in surface-fit loss (mm²)\nNegative = improvement")
            ax.legend()
            ax.grid(alpha=.2)
    fig.suptitle("Clean-model local pose test: anchor source is not a running sampler", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .97))
    output = args.output or args.input.with_name("pose_loss_slices.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
