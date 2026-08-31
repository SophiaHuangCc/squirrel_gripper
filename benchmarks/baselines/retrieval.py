"""Nearest-scenario retrieval baseline using only observed simulator archives."""

import argparse
from pathlib import Path

import numpy as np

from benchmarks.candidates import save_candidates
from benchmarks.protocol import DEFAULT_CONFIG, expand_core_scenarios, load_config
from dynamics.dataloader import DynamicsDataset
from generator.dataloader import model_norm_to_physical
ENV_KEYS = ("approach_deg", "cyl_rad", "landing_approach_deg", "initial_x_gap", "landing_height")
ENV_SCALES = np.asarray([90.0, 0.05, 90.0, 0.20, 0.10], dtype=np.float32)


def scalar(data, key, default):
    if key not in data:
        return float(default)
    value = np.asarray(data[key]).reshape(-1)
    return float(default) if value.size == 0 else float(value[0])


def archive_environment(path):
    with np.load(path, allow_pickle=True) as data:
        return np.asarray(
            [
                scalar(data, "arg_approach_deg", 45.0),
                scalar(data, "cyl_radius", scalar(data, "arg_cyl_rad", 0.025)),
                scalar(data, "arg_landing_approach_deg", 45.0),
                scalar(data, "arg_initial_x_gap", 0.12),
                scalar(data, "arg_landing_height", 0.04),
            ],
            dtype=np.float32,
        )


def target_cells(config, scenario_id=None, family=None, generalist=False):
    cells = expand_core_scenarios(config)
    if scenario_id:
        cells = [cell for cell in cells if cell["scenario_id"] == scenario_id]
    elif family:
        cells = [cell for cell in cells if cell["family"] == family]
    elif not generalist:
        default_id = config.get("default_target_scenario_id")
        if default_id:
            cells = [cell for cell in cells if cell["scenario_id"] == default_id]
        else:
            cells = [cell for cell in cells if cell["family"] == "nominal"]
    if not cells:
        raise ValueError("Retrieval target selected zero benchmark cells")
    return np.asarray([[cell["params"][key] for key in ENV_KEYS] for cell in cells], dtype=np.float32)


def retrieve(dataset_dir, config, num_candidates, scenario_id=None, family=None, generalist=False):
    dataset = DynamicsDataset(str(dataset_dir))
    targets = target_cells(
        config, scenario_id=scenario_id, family=family, generalist=generalist
    )
    rows = []
    weights = config["evaluation"]["utility_weights"]
    for index, path in enumerate(dataset.files):
        item = dataset[index]
        # Use the shared model-to-physical conversion so retrieval cannot drift
        # from the dynamics/generator design contract (currently 16D and
        # including base_thickness).
        design = model_norm_to_physical(item["design_params"]).numpy()
        environment = archive_environment(path)
        distances = np.linalg.norm((targets - environment[None, :]) / ENV_SCALES, axis=1)
        scenario_distance = float(np.mean(distances))
        target = item["target_metrics"].numpy()
        observed_utility = float(
            float(weights["contact_coverage_norm"]) * target[0]
            + float(weights["disturbance_resistance_score"]) * target[1]
            + float(weights["angular_span_norm"]) * target[2]
        )
        rows.append((scenario_distance, -observed_utility, path, design, observed_utility))
    rows.sort(key=lambda row: (row[0], row[1], row[2]))
    selected = []
    seen = set()
    for distance, _, path, design, observed_utility in rows:
        signature = tuple(np.round(design, 7))
        if signature in seen:
            continue
        seen.add(signature)
        selected.append((design, distance, path, observed_utility))
        if len(selected) >= num_candidates:
            break
    if not selected:
        raise ValueError(f"No retrievable designs found in {dataset_dir}")
    return selected


def main():
    parser = argparse.ArgumentParser(description="Retrieve designs from nearest observed scenarios.")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--num_candidates", type=int, default=16)
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--scenario_id", type=str, default=None)
    target.add_argument("--family", type=str, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    selected = retrieve(
        args.data_dir, config, args.num_candidates,
        scenario_id=args.scenario_id, family=args.family,
    )
    designs = np.asarray([row[0] for row in selected], dtype=np.float32)
    scores = np.asarray([-row[1] for row in selected], dtype=np.float32)
    metadata = {
        "target_scenario_id": args.scenario_id,
        "target_family": args.family,
        "sources": [row[2] for row in selected],
        "observed_utilities": [row[3] for row in selected],
    }
    save_candidates(args.output, designs, method="retrieval", seed=0, scores=scores, metadata=metadata)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
