"""Compare held-out DGDM diagnostics from controlled depth experiment arms."""
import argparse
import csv
from pathlib import Path


KEYS = (
    "utility_mae", "contact_coverage_mae", "disturbance_mae",
    "angular_span_mae", "direction_sign_accuracy", "direction_pearson",
    "scheduler_scaled_gradient_norm",
)


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def row_key(row):
    return row["model"], int(row["diffusion_timestep"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shallow", type=Path, required=True)
    parser.add_argument("--deep", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    shallow = {row_key(row): row for row in read_rows(args.shallow)}
    deep = {row_key(row): row for row in read_rows(args.deep)}
    if shallow.keys() != deep.keys():
        raise ValueError("Diagnostic arms do not contain identical model/timestep rows")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "diffusion_timestep"]
    for key in KEYS:
        fields.extend((f"shallow_{key}", f"deep_{key}", f"deep_minus_shallow_{key}"))
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model, timestep in sorted(shallow):
            out = {"model": model, "diffusion_timestep": timestep}
            for key in KEYS:
                a = float(shallow[(model, timestep)][key])
                b = float(deep[(model, timestep)][key])
                out.update({f"shallow_{key}": a, f"deep_{key}": b,
                            f"deep_minus_shallow_{key}": b - a})
            writer.writerow(out)
    print(f"[DONE] {args.output}")


if __name__ == "__main__":
    main()
