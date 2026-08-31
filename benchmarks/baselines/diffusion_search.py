"""Seeded conditional/unconditional diffusion and dynamics-guided baselines."""

from pathlib import Path

import numpy as np
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from benchmarks.baselines.surrogate_search import SearchResult, rank_designs, select_target_cells
from generator.dataloader import DesignBounds
from generator.diffusion import SquirrelDesignDiffusion, make_condition_batch
from generator.diffusion_utils import ConditionalUnet1D


def load_diffusion(
    checkpoint_path, device="cpu", num_inference_steps=20, expected_conditioning=None,
):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_args = checkpoint.get("args", {})
    conditioning_mode = checkpoint_args.get("conditioning", "conditional")
    if expected_conditioning is not None and conditioning_mode != expected_conditioning:
        raise ValueError(
            f"Diffusion checkpoint {checkpoint_path} is {conditioning_mode!r}, but "
            f"{expected_conditioning!r} conditioning was requested. Train/use the matching checkpoint."
        )
    bounds_path = checkpoint_path.parent / "design_bounds.npz"
    bounds = DesignBounds.from_npz(str(bounds_path)) if bounds_path.exists() else DesignBounds.defaults()
    network = ConditionalUnet1D(
        input_dim=1, global_cond_dim=9, down_dims=[128, 256],
        diffusion_step_embed_dim=32,
    )
    scheduler = DDIMScheduler(
        num_train_timesteps=int(checkpoint_args.get("num_train_timesteps", 100)),
        beta_schedule="squaredcos_cap_v2", clip_sample=True, prediction_type="epsilon",
    )
    model = SquirrelDesignDiffusion(
        noise_pred_net=network, noise_scheduler=scheduler, bounds=bounds,
        num_inference_steps=num_inference_steps,
        conditioning_mode=conditioning_mode,
    ).to(device)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
    if "ema" in checkpoint:
        model.ema.load_state_dict(checkpoint["ema"], torch.device(device))
        model.copy_ema_to_model()
    model.eval()
    return model


def diffusion_search(
    diffusion_model, dynamics_model, config, num_candidates, num_samples, seed,
    batch_size=256, guidance_scale=0.0, num_inference_steps=20,
    target_contacts=0.8, target_disturbance=0.8, target_angular_span=0.8,
    scenario_id=None, family=None, generalist=False, device="cpu",
):
    cells = select_target_cells(config, scenario_id, family, generalist)
    if num_samples < num_candidates:
        raise ValueError("diffusion num_samples must be >= candidate_budget")
    # The diffusion network was trained with one condition vector.  For a
    # family/generalist task, use the centroid of the selected scenario set as
    # its proposal context, then rank every proposal over the complete set.
    # DGDM additionally differentiates the mean utility over every selected
    # scenario during every denoising step.
    keys = (
        "approach_deg", "landing_approach_deg", "cyl_rad", "landing_height",
        "landing_speed", "initial_x_gap",
    )
    params = {
        key: float(np.mean([cell["params"][key] for cell in cells]))
        for key in keys
    }
    guidance_task = torch.tensor(
        [[cell["params"]["approach_deg"] / 90.0,
          cell["params"]["landing_approach_deg"] / 90.0,
          cell["params"]["cyl_rad"] / 0.05] for cell in cells],
        dtype=torch.float32, device=device,
    )
    guidance_init = torch.tensor(
        [[cell["params"]["landing_height"] / 0.10,
          cell["params"]["landing_speed"] / 1.0,
          cell["params"]["initial_x_gap"] / 0.10] for cell in cells],
        dtype=torch.float32, device=device,
    )
    diffusion_model.num_inference_steps = int(num_inference_steps)
    generator = torch.Generator(device=device).manual_seed(seed)
    generated = []
    remaining = int(num_samples)
    while remaining:
        current = min(int(batch_size), remaining)
        condition = make_condition_batch(
            batch_size=current,
            approach_deg=params["approach_deg"],
            landing_approach_deg=params["landing_approach_deg"],
            cyl_radius=params["cyl_rad"],
            landing_height=params["landing_height"], landing_speed=params["landing_speed"],
            initial_x_gap=params["initial_x_gap"], target_contacts=target_contacts,
            target_disturbance=target_disturbance, target_angular_span=target_angular_span,
            device=torch.device(device),
        )
        output = diffusion_model.sample(
            cond=condition,
            dynamics_model=dynamics_model if guidance_scale > 0 else None,
            guidance_scale=float(guidance_scale), guidance_objective="benchmark_utility",
            generator=generator,
            guidance_task_params=guidance_task if guidance_scale > 0 else None,
            guidance_init_config=guidance_init if guidance_scale > 0 else None,
            guidance_weights=config["evaluation"]["utility_weights"],
        )
        generated.append(output["design_physical"].detach().cpu().numpy())
        remaining -= current
    pool = np.concatenate(generated, axis=0)
    ranked = rank_designs(
        dynamics_model, pool, config, scenario_id=scenario_id, family=family,
        generalist=generalist, device=device,
    )
    denoising_evaluations = num_samples * num_inference_steps
    guidance_evaluations = (
        num_samples * num_inference_steps * len(cells) if guidance_scale > 0 else 0
    )
    return SearchResult(
        designs=ranked.designs[:num_candidates], scores=ranked.scores[:num_candidates],
        model_evaluations=ranked.model_evaluations + denoising_evaluations + guidance_evaluations,
        target_scenario_ids=ranked.target_scenario_ids,
    )
