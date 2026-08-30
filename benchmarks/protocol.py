"""Expand benchmark scenarios and aggregate simulator metrics consistently."""

import argparse
import itertools
import json
import math
import statistics
from pathlib import Path


DEFAULT_CONFIG = Path(__file__).with_name("scenarios_v1.json")


def load_config(path=DEFAULT_CONFIG):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def expand_core_scenarios(config):
    """Return deterministic scenario cells with nominal values filled in."""
    nominal = dict(config["nominal"])
    fixed = dict(config["fixed_simulation"])
    cells = []
    for family in config["families"]:
        grid = family.get("grid", {})
        keys = list(grid)
        combinations = itertools.product(*(grid[key] for key in keys)) if keys else [()]
        for cell_index, values in enumerate(combinations):
            params = {**fixed, **nominal, **dict(zip(keys, values))}
            cells.append(
                {
                    "scenario_id": f"{family['id']}:{cell_index:02d}",
                    "family": family["id"],
                    "params": params,
                }
            )
    return cells


def normalized_metrics(metric):
    n_elements = max(float(metric.get("n_elements", 100.0)), 1.0)
    return {
        "disturbance_resistance_score": min(max(float(metric["disturbance_resistance_score"]), 0.0), 1.0),
        "contact_coverage_norm": min(
            max(math.log1p(float(metric["num_contacts"])) / math.log1p(n_elements), 0.0),
            1.0,
        ),
        "angular_span_norm": min(max(float(metric["angular_span"]) / 180.0, 0.0), 1.0),
    }


def utility(metric, weights):
    values = normalized_metrics(metric)
    return sum(float(weights[name]) * values[name] for name in weights)


def aggregate_records(records, config):
    """Aggregate already simulated records; dynamics predictions are not accepted."""
    weights = config["evaluation"]["utility_weights"]
    scored = [{**record, "utility": utility(record["metrics"], weights)} for record in records]
    if not scored:
        raise ValueError("Cannot aggregate an empty benchmark result set")

    values = sorted(record["utility"] for record in scored)
    cvar_fraction = float(config["evaluation"]["cvar_fraction"])
    tail_count = max(1, math.ceil(cvar_fraction * len(values)))
    by_family = {}
    for record in scored:
        by_family.setdefault(record["family"], []).append(record["utility"])
    family_means = {
        family: sum(family_values) / len(family_values)
        for family, family_values in by_family.items()
    }
    return {
        "num_rollouts": len(values),
        "mean_utility": sum(values) / len(values),
        "median_utility": statistics.median(values),
        "std_utility": statistics.pstdev(values),
        "cvar20_utility": sum(values[:tail_count]) / tail_count,
        "worst_cell_utility": values[0],
        "worst_family_utility": min(family_means.values()),
        "family_mean_utility": family_means,
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect Squirrel Benchmark V1 scenarios.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--json", action="store_true", help="Print expanded cells as JSON")
    args = parser.parse_args()
    config = load_config(args.config)
    cells = expand_core_scenarios(config)
    if args.json:
        print(json.dumps(cells, indent=2))
        return
    counts = {}
    for cell in cells:
        counts[cell["family"]] = counts.get(cell["family"], 0) + 1
    print(f"Core scenarios: {len(cells)}")
    for family, count in counts.items():
        print(f"  {family}: {count}")


if __name__ == "__main__":
    main()
