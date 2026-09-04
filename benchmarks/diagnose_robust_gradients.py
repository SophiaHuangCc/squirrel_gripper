"""Simulator-backed gradient diagnostics for physical-condition ensembles."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from benchmarks.baselines.surrogate_search import (
    _scenario_tensors, load_surrogate, select_target_cells,
)
from generator.dataloader import (
    DESIGN_NAMES, DesignBounds, enforce_fixed_design_unit,
    physical_to_diffusion, physical_to_model_norm, variable_design_mask,
)


def parse_int_list(text):
    return tuple(int(value.strip()) for value in text.split(",") if value.strip())


def pearson(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def directional_quality(designs, gradients, utilities):
    predicted, observed, pair_rows = [], [], []
    for i in range(len(designs)):
        distances = np.linalg.norm(designs - designs[i], axis=1)
        distances[i] = np.inf
        j = int(np.argmin(distances))
        predicted_change = float(np.dot(gradients[i], designs[j] - designs[i]))
        observed_change = float(utilities[j] - utilities[i])
        predicted.append(predicted_change)
        observed.append(observed_change)
        pair_rows.append({
            "source_index": i, "neighbor_index": j,
            "predicted_directional_change": predicted_change,
            "observed_utility_change": observed_change,
            "sign_correct": (
                np.sign(predicted_change) == np.sign(observed_change)
                if abs(predicted_change) > 1e-10 and abs(observed_change) > 1e-10
                else None
            ),
        })
    predicted, observed = np.asarray(predicted), np.asarray(observed)
    valid = (np.abs(predicted) > 1e-10) & (np.abs(observed) > 1e-10)
    sign = (
        float(np.mean(np.sign(predicted[valid]) == np.sign(observed[valid])))
        if valid.any() else math.nan
    )
    return (len(predicted), sign, pearson(predicted, observed)), pair_rows


def load_robust_rollouts(root, expected_conditions):
    grouped = defaultdict(list)
    for path in sorted(root.rglob("benchmark_result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") != "ok" or "@" not in result.get("scenario_id", ""):
            continue
        job_path = path.with_name("benchmark_job.json")
        if not job_path.exists():
            continue
        job = json.loads(job_path.read_text(encoding="utf-8"))
        base = result["scenario_id"].split("@", 1)[0]
        key = (base, result.get("method", ""), int(result.get("seed", 0)),
               result.get("candidate_id", ""))
        grouped[key].append((result, job, path))
    rows = []
    for key, records in grouped.items():
        if len(records) != expected_conditions:
            continue
        nominal = next(
            ((result, job, path) for result, job, path in records
             if result["scenario_id"].endswith("@nominal")),
            None,
        )
        if nominal is None:
            continue
        utilities = [float(result["utility"]) for result, _, _ in records]
        components = [result["normalized_metrics"] for result, _, _ in records]
        nominal_result = nominal[0]
        nominal_components = nominal_result["normalized_metrics"]
        design_physical = np.asarray(records[0][1]["design_params"], dtype=np.float32)
        rows.append({
            "base_scenario_id": key[0], "method": key[1], "seed": key[2],
            "candidate_id": key[3],
            "source_result": str(nominal[2].resolve()),
            "design_physical": design_physical,
            **{
                f"design_{name}": float(value)
                for name, value in zip(DESIGN_NAMES, design_physical)
            },
            "robust_utility": float(np.mean(utilities)),
            "worst_utility": float(np.min(utilities)),
            "std_utility": float(np.std(utilities)),
            "robust_contact": float(np.mean([x["contact_coverage_norm"] for x in components])),
            "std_contact": float(np.std([x["contact_coverage_norm"] for x in components])),
            "robust_disturbance": float(np.mean([x["disturbance_resistance_score"] for x in components])),
            "std_disturbance": float(np.std([x["disturbance_resistance_score"] for x in components])),
            "robust_angular_span": float(np.mean([x["angular_span_norm"] for x in components])),
            "std_angular_span": float(np.std([x["angular_span_norm"] for x in components])),
            "nominal_utility": float(nominal_result["utility"]),
            "nominal_contact": float(nominal_components["contact_coverage_norm"]),
            "nominal_disturbance": float(nominal_components["disturbance_resistance_score"]),
            "nominal_angular_span": float(nominal_components["angular_span_norm"]),
        })
    return rows


def evaluate(model, model_design, physical_design, task, init, targets, weights, timestep):
    count, conditions = len(model_design), len(task)
    design = model_design.detach().clone().requires_grad_(True)
    design_batch = design.repeat_interleave(conditions, dim=0)
    task_batch = task.repeat(count, 1)
    init_batch = init.repeat(count, 1)
    time_batch = torch.full(
        (count * conditions,), float(timestep), dtype=design.dtype, device=design.device
    )
    raw = model(task_batch, design_batch, init_batch, time_batch)
    prediction = raw.clamp(0.0, 1.0).reshape(count, conditions, 3).mean(dim=1)
    weight_tensor = torch.tensor(weights, dtype=design.dtype, device=design.device)
    predicted_utility = prediction @ weight_tensor
    gradient = torch.autograd.grad(predicted_utility.sum(), design)[0]
    gradient = gradient * variable_design_mask(DesignBounds.defaults(), design.device)
    target_tensor = torch.as_tensor(targets, dtype=design.dtype, device=design.device)
    errors = prediction - target_tensor[:, 1:]
    (n_pairs, sign, correlation), pair_rows = directional_quality(
        design.detach().cpu().numpy(), gradient.detach().cpu().numpy(), targets[:, 0]
    )
    return {
        "num_designs": count,
        "num_physical_conditions": conditions,
        "utility_mae": float(np.mean(np.abs(predicted_utility.detach().cpu().numpy() - targets[:, 0]))),
        "utility_bias": float(np.mean(predicted_utility.detach().cpu().numpy() - targets[:, 0])),
        "contact_coverage_mae": float(errors[:, 0].abs().mean()),
        "disturbance_mae": float(errors[:, 1].abs().mean()),
        "angular_span_mae": float(errors[:, 2].abs().mean()),
        "mean_gradient_norm": float(torch.linalg.vector_norm(gradient, dim=1).mean()),
        "num_direction_pairs": n_pairs,
        "direction_sign_accuracy": sign,
        "direction_pearson": correlation,
    }, pair_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark_dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--clean_checkpoint", type=Path, required=True)
    parser.add_argument("--noisy_checkpoint", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--timesteps", default="0,3,6,9,12")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from benchmarks.protocol import load_config
    config = load_config(args.config)
    variants = config["evaluation"]["physical_condition_ensemble"]["variants"]
    rollouts = load_robust_rollouts(args.benchmark_dir, len(variants))
    if not rollouts:
        raise ValueError("No complete robust candidate groups found")
    device = torch.device(args.device)
    weights_map = config["evaluation"]["utility_weights"]
    weights = (
        weights_map["contact_coverage_norm"],
        weights_map["disturbance_resistance_score"],
        weights_map["angular_span_norm"],
    )
    clean = load_surrogate(args.clean_checkpoint, device=args.device)
    noisy = (
        load_surrogate(args.noisy_checkpoint, device=args.device, expected_noise_conditioned=True)
        if args.noisy_checkpoint else None
    )
    bounds = DesignBounds.defaults()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    output_rows = []
    direction_rows = []
    for scenario in sorted({row["base_scenario_id"] for row in rollouts}):
        subset = [row for row in rollouts if row["base_scenario_id"] == scenario]
        physical = torch.tensor(
            np.stack([row["design_physical"] for row in subset]), device=device
        )
        robust_targets = np.asarray([
            [row["robust_utility"], row["robust_contact"],
             row["robust_disturbance"], row["robust_angular_span"]]
            for row in subset
        ], dtype=np.float32)
        nominal_targets = np.asarray([
            [row["nominal_utility"], row["nominal_contact"],
             row["nominal_disturbance"], row["nominal_angular_span"]]
            for row in subset
        ], dtype=np.float32)
        cells = select_target_cells(config, scenario_id=scenario)
        task, init = _scenario_tensors(cells, device)
        nominal_indices = [
            index for index, cell in enumerate(cells)
            if cell.get("physical_condition_id") == "nominal"
        ]
        if len(nominal_indices) != 1:
            raise ValueError(f"Expected exactly one nominal condition for {scenario}")
        nominal_task = task[nominal_indices]
        nominal_init = init[nominal_indices]
        clean_design = physical_to_model_norm(physical)
        for aggregation, eval_task, eval_init, targets in (
            ("nominal", nominal_task, nominal_init, nominal_targets),
            ("robust_mean", task, init, robust_targets),
        ):
            clean_result, pairs = evaluate(
                clean, clean_design, physical, eval_task, eval_init, targets, weights, 0.0
            )
            output_rows.append({"model": "clean", "aggregation": aggregation,
                                "base_scenario_id": scenario,
                                "diffusion_timestep": 0, "noise_std": 0.0,
                                "scheduler_scaled_gradient_norm": 0.0, **clean_result})
            for pair in pairs:
                source = subset[pair.pop("source_index")]
                neighbor = subset[pair.pop("neighbor_index")]
                direction_rows.append({
                    "model": "clean", "aggregation": aggregation,
                    "base_scenario_id": scenario, "diffusion_timestep": 0,
                    "source_method": source["method"],
                    "source_candidate_id": source["candidate_id"],
                    "neighbor_method": neighbor["method"],
                    "neighbor_candidate_id": neighbor["candidate_id"],
                    **pair,
                })
        if noisy is None:
            continue
        steps = noisy.num_train_timesteps
        scheduler = DDIMScheduler(num_train_timesteps=steps, beta_schedule="squaredcos_cap_v2",
                                  clip_sample=True, prediction_type="epsilon")
        clean_unit = physical_to_diffusion(physical, bounds).clamp(-1, 1)
        shared_noise = torch.randn((1, clean_unit.shape[1]), generator=generator,
                                   device=device).expand_as(clean_unit)
        for timestep in parse_int_list(args.timesteps):
            t = torch.full((len(clean_unit),), timestep, dtype=torch.long, device=device)
            noisy_design = enforce_fixed_design_unit(
                scheduler.add_noise(clean_unit, shared_noise, t), bounds
            )
            noise_std = float((1.0 - scheduler.alphas_cumprod[timestep]).sqrt())
            for aggregation, eval_task, eval_init, targets in (
                ("nominal", nominal_task, nominal_init, nominal_targets),
                ("robust_mean", task, init, robust_targets),
            ):
                result, pairs = evaluate(
                    noisy, noisy_design, physical, eval_task, eval_init, targets, weights,
                    timestep / steps,
                )
                output_rows.append({
                    "model": "noise_conditioned", "aggregation": aggregation,
                    "base_scenario_id": scenario,
                    "diffusion_timestep": timestep, "noise_std": noise_std,
                    "scheduler_scaled_gradient_norm": noise_std * result["mean_gradient_norm"],
                    **result,
                })
                for pair in pairs:
                    source = subset[pair.pop("source_index")]
                    neighbor = subset[pair.pop("neighbor_index")]
                    direction_rows.append({
                        "model": "noise_conditioned", "aggregation": aggregation,
                        "base_scenario_id": scenario, "diffusion_timestep": timestep,
                        "source_method": source["method"],
                        "source_candidate_id": source["candidate_id"],
                        "neighbor_method": neighbor["method"],
                        "neighbor_candidate_id": neighbor["candidate_id"],
                        **pair,
                    })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output_dir / "robust_candidate_metrics.csv"
    candidate_fields = [
        "base_scenario_id", "method", "seed", "candidate_id", "source_result",
        *[f"design_{name}" for name in DESIGN_NAMES], "robust_utility",
        "worst_utility", "std_utility", "robust_contact", "robust_disturbance",
        "robust_angular_span", "std_contact", "std_disturbance", "std_angular_span",
    ]
    with candidate_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=candidate_fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rollouts)
    best_rows = []
    best_groups = defaultdict(list)
    for row in rollouts:
        best_groups[(row["base_scenario_id"], row["method"])].append(row)
    for (scenario, method), rows in sorted(best_groups.items()):
        winner = max(rows, key=lambda row: row["robust_utility"])
        best_rows.append({key: winner[key] for key in candidate_fields})
    best_path = args.output_dir / "best_robust_design_per_method.csv"
    with best_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=candidate_fields)
        writer.writeheader(); writer.writerows(best_rows)
    criterion_rows = []
    criteria = {
        "combined_utility": "robust_utility",
        "contact": "robust_contact",
        "disturbance": "robust_disturbance",
        "angular_span": "robust_angular_span",
        "worst_case_utility": "worst_utility",
    }
    for scenario in sorted({row["base_scenario_id"] for row in rollouts}):
        scenario_rows = [row for row in rollouts if row["base_scenario_id"] == scenario]
        for criterion, value_key in criteria.items():
            winner = max(scenario_rows, key=lambda row: row[value_key])
            criterion_rows.append({
                "base_scenario_id": scenario, "criterion": criterion,
                "selected_value": winner[value_key],
                **{key: winner[key] for key in candidate_fields},
            })
    criterion_path = args.output_dir / "best_robust_designs_by_criterion.csv"
    with criterion_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(criterion_rows[0]))
        writer.writeheader(); writer.writerows(criterion_rows)
    path = args.output_dir / "robust_gradient_diagnostics.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader(); writer.writerows(output_rows)
    direction_path = args.output_dir / "robust_direction_pairs.csv"
    with direction_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(direction_rows[0]))
        writer.writeheader(); writer.writerows(direction_rows)
    manifest = {
        "benchmark_dir": str(args.benchmark_dir.resolve()),
        "config": str(args.config.resolve()),
        "clean_checkpoint": str(args.clean_checkpoint.resolve()),
        "noisy_checkpoint": str(args.noisy_checkpoint.resolve()) if args.noisy_checkpoint else None,
        "physical_conditions_per_design": len(variants),
        "complete_design_groups": len(rollouts),
        "direction_test": "nearest generated design using simulator robust-mean utility",
    }
    (args.output_dir / "robust_gradient_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("\n[ROBUST CANDIDATE MEAN +/- STD; BEST COMBINED PER METHOD]")
    print(
        f"{'SCENARIO':22} {'METHOD':24} "
        f"{'UTILITY':17} {'CONTACT':17} {'DISTURB':17} {'ANGULAR':17}"
    )
    for row in best_rows:
        def metric(mean_key, std_key):
            return f"{row[mean_key]:.4f} +/- {row[std_key]:.4f}"
        print(
            f"{row['base_scenario_id']:22} {row['method']:24} "
            f"{metric('robust_utility', 'std_utility'):17} "
            f"{metric('robust_contact', 'std_contact'):17} "
            f"{metric('robust_disturbance', 'std_disturbance'):17} "
            f"{metric('robust_angular_span', 'std_angular_span'):17}"
        )
    print("\n[BEST DESIGNS BY CRITERION]")
    for row in criterion_rows:
        print(
            f"{row['base_scenario_id']} {row['criterion']:20} "
            f"method={row['method']:24} candidate={row['candidate_id']} "
            f"value={row['selected_value']:.4f}"
        )
    print(f"[ROBUST GRADIENT DIAGNOSTICS] {path}")
    for row in output_rows:
        print(row)


if __name__ == "__main__":
    main()
