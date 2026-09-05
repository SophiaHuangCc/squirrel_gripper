"""Diagnose clean and diffusion-noise-conditioned dynamics guidance.

This is deliberately simulator-label based: in addition to prediction errors it
checks whether a model gradient points from a validation design toward another
same-environment design with higher measured utility.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from benchmarks.baselines.surrogate_search import load_surrogate
from benchmarks.protocol import load_config
from dynamics.dataloader import DynamicsDataset
from dynamics.pose_targets import pose_joint_angles_deg, surrogate_metrics
from generator.dataloader import (
    DesignBounds,
    enforce_fixed_design_unit,
    model_norm_to_physical,
    physical_to_diffusion,
    variable_design_mask,
)


METRICS = ("contact_coverage", "disturbance", "angular_span")


def parse_int_list(text):
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values or any(value < 0 for value in values):
        raise ValueError("--timesteps must contain nonnegative integers")
    return values


def pearson(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def directional_quality(designs, gradients, utility, environment, max_pairs=2000):
    """Compare local gradient predictions with simulator-labeled utility differences."""
    groups = defaultdict(list)
    for index, key in enumerate(environment):
        groups[key].append(index)
    predicted, observed = [], []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        group_designs = designs[indices]
        for local_i, index_i in enumerate(indices):
            distances = np.linalg.norm(group_designs - group_designs[local_i], axis=1)
            distances[local_i] = np.inf
            index_j = indices[int(np.argmin(distances))]
            predicted.append(float(np.dot(gradients[index_i], designs[index_j] - designs[index_i])))
            observed.append(float(utility[index_j] - utility[index_i]))
            if len(predicted) >= max_pairs:
                break
        if len(predicted) >= max_pairs:
            break
    predicted, observed = np.asarray(predicted), np.asarray(observed)
    valid = (np.abs(predicted) > 1e-10) & (np.abs(observed) > 1e-10)
    sign_accuracy = float(np.mean(np.sign(predicted[valid]) == np.sign(observed[valid]))) if valid.any() else math.nan
    return {"num_direction_pairs": int(len(predicted)),
            "direction_sign_accuracy": sign_accuracy,
            "direction_pearson": pearson(predicted, observed)}


def collect(dataset, max_samples):
    count = min(len(dataset), max_samples or len(dataset))
    rows = [dataset[index] for index in range(count)]
    return {key: torch.stack([row[key] for row in rows]) for key in rows[0]}


def evaluate_model(model, design, task, init, target, weights, environment, target_pose=None):
    design = design.detach().clone().requires_grad_(True)
    timestep = evaluate_model.timestep.expand(len(design))
    prediction_raw = model(task, design, init, timestep)
    prediction = surrogate_metrics(model, prediction_raw, task).clamp(0.0, 1.0)
    weight_tensor = torch.tensor(weights, dtype=design.dtype, device=design.device)
    predicted_utility = (prediction * weight_tensor).sum(dim=1)
    true_utility = (target * weight_tensor).sum(dim=1)
    gradient = torch.autograd.grad(predicted_utility.sum(), design)[0]
    gradient = gradient * variable_design_mask(DesignBounds.defaults(), design.device)
    error = prediction - target
    gradient_np = gradient.detach().cpu().numpy()
    design_np = design.detach().cpu().numpy()
    output = {
        "num_samples": len(design),
        "utility_mae": float((predicted_utility - true_utility).abs().mean()),
        "utility_bias": float((predicted_utility - true_utility).mean()),
        "prediction_out_of_range_fraction": (float(((prediction_raw < 0) | (prediction_raw > 1)).float().mean())
                                             if getattr(model, "target_representation", "metrics") == "metrics"
                                             else 0.0),
        "mean_gradient_norm": float(torch.linalg.vector_norm(gradient, dim=1).mean()),
        "near_zero_gradient_fraction": float((torch.linalg.vector_norm(gradient, dim=1) < 1e-8).float().mean()),
    }
    for column, name in enumerate(METRICS):
        output[f"{name}_mae"] = float(error[:, column].abs().mean())
        output[f"{name}_bias"] = float(error[:, column].mean())
    if getattr(model, "target_representation", "metrics") == "pose_keypoints" and target_pose is not None:
        pose_error = prediction_raw - target_pose
        output["pose_keypoint_mae_mm"] = float(pose_error.abs().mean() * model.pose_scale_m * 1000.0)
        output["pose_tip_mae_mm"] = float(torch.linalg.vector_norm(
            pose_error.reshape(-1, 5, 2)[:, -1], dim=-1
        ).mean() * model.pose_scale_m * 1000.0)
        _, pred_bends = pose_joint_angles_deg(prediction_raw)
        _, true_bends = pose_joint_angles_deg(target_pose)
        angle_delta = torch.atan2(
            torch.sin((pred_bends - true_bends) * torch.pi / 180.0),
            torch.cos((pred_bends - true_bends) * torch.pi / 180.0),
        )
        output["pose_joint_angle_mae_deg"] = float(angle_delta.abs().mean() * 180.0 / torch.pi)
    output.update(directional_quality(
        design_np, gradient_np, true_utility.detach().cpu().numpy(), environment
    ))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True,
                        help="Held-out simulator NPZ directory; do not use training data.")
    parser.add_argument("--clean_checkpoint", type=Path, required=True)
    parser.add_argument("--noisy_checkpoint", type=Path, required=True)
    parser.add_argument("--diffusion_checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--timesteps", default="0,10,25,50,75,90,99")
    parser.add_argument("--max_samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    parser.add_argument("--wandb_project", default="")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_mode", choices=("online", "offline", "disabled"),
                        default="online")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = load_config(args.config)
    weight_map = config["evaluation"]["utility_weights"]
    weights = (weight_map["contact_coverage_norm"],
               weight_map["disturbance_resistance_score"],
               weight_map["angular_span_norm"])
    dataset = DynamicsDataset(args.data_dir, target_representation="pose_keypoints")
    if not len(dataset):
        raise ValueError(f"No validation NPZ files found under {args.data_dir}")
    batch = collect(dataset, args.max_samples)
    device = torch.device(args.device)
    task = batch["task_params"].to(device)
    clean_model_design = batch["design_params"].to(device)
    init = batch["init_config"].to(device)
    target = batch["target_metrics"].to(device)
    target_pose = batch["target_pose"].to(device)
    environment = [tuple(np.round(row, 7)) for row in torch.cat((task, init), dim=1).cpu().numpy()]

    clean_model = load_surrogate(args.clean_checkpoint, device=args.device)
    noisy_model = load_surrogate(args.noisy_checkpoint, device=args.device,
                                 expected_noise_conditioned=True)
    checkpoint = torch.load(args.diffusion_checkpoint, map_location="cpu")
    prior_steps = int(checkpoint.get("args", {}).get("num_train_timesteps", 100))
    if prior_steps != noisy_model.num_train_timesteps:
        raise ValueError(f"Prior has {prior_steps} timesteps but noisy dynamics has "
                         f"{noisy_model.num_train_timesteps}")
    bounds_path = args.diffusion_checkpoint.parent / "design_bounds.npz"
    source_bounds = (
        DesignBounds.from_npz(str(bounds_path))
        if bounds_path.exists() else DesignBounds.defaults()
    )
    # DesignBounds is intentionally frozen.  Construct a device-local copy
    # rather than mutating the checkpoint-loaded instance.
    bounds = DesignBounds(
        lo=source_bounds.lo.to(device),
        hi=source_bounds.hi.to(device),
    )
    clean_unit = physical_to_diffusion(
        model_norm_to_physical(clean_model_design), bounds
    ).clamp(-1, 1)
    scheduler = DDIMScheduler(num_train_timesteps=prior_steps,
                              beta_schedule="squaredcos_cap_v2", clip_sample=True,
                              prediction_type="epsilon")
    shared_noise = torch.randn((1, clean_unit.shape[1]), device=device).expand_as(clean_unit)

    records = []
    evaluate_model.timestep = torch.tensor(0.0, device=device)
    records.append({"model": "clean", "diffusion_timestep": 0,
                    "noise_std": 0.0, "scheduler_scaled_gradient_norm": 0.0,
                    **evaluate_model(clean_model, clean_model_design, task, init,
                                     target, weights, environment, target_pose)})
    for timestep in parse_int_list(args.timesteps):
        if timestep >= prior_steps:
            raise ValueError(f"Timestep {timestep} is outside [0, {prior_steps - 1}]")
        t = torch.full((len(clean_unit),), timestep, dtype=torch.long, device=device)
        noisy_design = enforce_fixed_design_unit(
            scheduler.add_noise(clean_unit, shared_noise, t), bounds
        )
        evaluate_model.timestep = torch.tensor(timestep / prior_steps, device=device)
        noise_std = float((1.0 - scheduler.alphas_cumprod[timestep]).sqrt())
        row = {"model": "noise_conditioned", "diffusion_timestep": timestep,
               "noise_std": noise_std,
                        **evaluate_model(noisy_model, noisy_design, task, init,
                                         target, weights, environment, target_pose)}
        row["scheduler_scaled_gradient_norm"] = noise_std * row["mean_gradient_norm"]
        records.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "timestep_diagnostics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    manifest = {
        "data_dir": str(args.data_dir.resolve()), "validation_samples": len(target),
        "clean_checkpoint": str(args.clean_checkpoint.resolve()),
        "noisy_checkpoint": str(args.noisy_checkpoint.resolve()),
        "diffusion_checkpoint": str(args.diffusion_checkpoint.resolve()),
        "timesteps": list(parse_int_list(args.timesteps)), "seed": args.seed,
        "direction_test": "nearest other design with identical normalized task and initial condition",
        "limitations": [
            "Direction tests use finite differences between observed designs, not local simulator derivatives.",
            "A model can pass prediction tests but still fail during iterative denoising.",
        ],
    }
    (args.output_dir / "diagnostic_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    if args.wandb_project:
        import wandb
        run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=args.wandb_run_name, mode=args.wandb_mode,
            config=manifest,
        )
        run.define_metric("diagnostic_timestep")
        run.define_metric("noise_conditioned/*", step_metric="diagnostic_timestep")
        run.log({"timestep_diagnostics": wandb.Table(
            columns=list(records[0]),
            data=[[row[key] for key in records[0]] for row in records],
        )})
        clean_row = records[0]
        run.log({f"clean/{key}": value for key, value in clean_row.items()
                 if isinstance(value, (int, float))})
        for row in records[1:]:
            payload = {f"noise_conditioned/{key}": value for key, value in row.items()
                       if isinstance(value, (int, float))}
            payload["diagnostic_timestep"] = int(row["diffusion_timestep"])
            run.log(payload)
        run.finish()
    print(f"[DIAGNOSTICS] {csv_path}")
    for row in records:
        print(row)


if __name__ == "__main__":
    main()
