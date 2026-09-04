"""Expand benchmark scenarios and aggregate simulator metrics consistently."""

import argparse
import itertools
import json
import math
import statistics
from pathlib import Path


DEFAULT_CONFIG = Path(__file__).with_name("scenarios_v2.json")


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


def expand_physical_conditions(cells, config):
    """Expand each scenario into configured physical initial-condition variants.

    These variants are repeated evaluations of the same design/scenario target,
    not additional design candidates. Offsets use simulator-native units.
    """
    robustness = config.get("evaluation", {}).get("physical_condition_ensemble", {})
    if not robustness.get("enabled", False):
        return cells
    variants = robustness.get("variants", [])
    if not variants:
        raise ValueError("Enabled physical_condition_ensemble requires nonempty variants")
    expanded = []
    allowed = {"landing_height", "landing_speed", "initial_x_gap"}
    for cell in cells:
        for index, variant in enumerate(variants):
            name = str(variant.get("id", f"ic{index:02d}"))
            offsets = dict(variant.get("offsets", {}))
            unknown = set(offsets) - allowed
            if unknown:
                raise ValueError(
                    f"Unsupported physical-condition offsets {sorted(unknown)}; "
                    f"allowed keys are {sorted(allowed)}"
                )
            params = dict(cell["params"])
            for key, offset in offsets.items():
                params[key] = float(params[key]) + float(offset)
            expanded.append({
                **cell,
                "scenario_id": f"{cell['scenario_id']}@{name}",
                "base_scenario_id": cell["scenario_id"],
                "physical_condition_id": name,
                "params": params,
            })
    return expanded


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
    component_names = tuple(weights)
    component_means = {
        name: sum(normalized_metrics(record["metrics"])[name] for record in scored) / len(scored)
        for name in component_names
    }
    raw_metric_means = {
        "num_contacts": sum(float(record["metrics"]["num_contacts"]) for record in scored) / len(scored),
        "disturbance_resistance_score": sum(
            float(record["metrics"]["disturbance_resistance_score"]) for record in scored
        ) / len(scored),
        "angular_span_deg": sum(float(record["metrics"]["angular_span"]) for record in scored) / len(scored),
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
        "component_mean": component_means,
        "raw_metric_mean": raw_metric_means,
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
