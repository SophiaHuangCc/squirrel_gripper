"""Compare two timestep diagnostic CSV files row by row."""
import argparse
import csv
from pathlib import Path


METRICS = (
    "utility_mae", "contact_coverage_mae", "disturbance_mae",
    "angular_span_mae", "direction_sign_accuracy", "direction_pearson",
    "mean_gradient_norm", "scheduler_scaled_gradient_norm",
)


def load(path):
    with path.open(newline="") as handle:
        return {
            (row["model"], int(row["diffusion_timestep"])): row
            for row in csv.DictReader(handle)
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline, candidate = load(args.baseline), load(args.candidate)
    if baseline.keys() != candidate.keys():
        raise ValueError("Diagnostics do not contain identical model/timestep rows")
    fields = ["model", "diffusion_timestep"]
    for metric in METRICS:
        fields += [f"baseline_{metric}", f"fixed10_{metric}", f"fixed10_minus_baseline_{metric}"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model, timestep in sorted(baseline):
            row = {"model": model, "diffusion_timestep": timestep}
            for metric in METRICS:
                old = float(baseline[(model, timestep)][metric])
                new = float(candidate[(model, timestep)][metric])
                row[f"baseline_{metric}"] = old
                row[f"fixed10_{metric}"] = new
                row[f"fixed10_minus_baseline_{metric}"] = new - old
            writer.writerow(row)
    print(f"[DONE] {args.output}")


if __name__ == "__main__":
    main()
