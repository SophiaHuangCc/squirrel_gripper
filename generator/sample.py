import argparse
import os
import sys
from os.path import join as pjoin

BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)
sys.path.insert(0, pjoin(BASEPATH, ".."))

import numpy as np
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from dynamics.profile_forward_2d import ProfileForward2DModel
from generator.dataloader import DesignBounds
from generator.diffusion import SquirrelDesignDiffusion, make_condition_batch
from generator.diffusion_utils import ConditionalUnet1D


def parse_args():
    parser = argparse.ArgumentParser(description="Sample squirrel finger designs from diffusion generator.")
    parser.add_argument("--diffusion_checkpoint_path", type=str, required=True)
    parser.add_argument("--dynamics_checkpoint_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="generator/runs/sample")
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_train_timesteps", type=int, default=100)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=0.0)
    parser.add_argument("--guidance_objective", type=str, default="disturbance_contact_span")
    parser.add_argument("--top_k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--approach_deg", type=float, default=45.0)
    parser.add_argument("--landing_approach_deg", type=float, default=45.0)
    parser.add_argument("--cyl_rad", type=float, default=0.03)
    parser.add_argument("--landing_height", type=float, default=0.04)
    parser.add_argument("--landing_speed", type=float, default=0.0)
    parser.add_argument("--initial_x_gap", type=float, default=0.06)
    parser.add_argument("--target_contacts", type=float, default=0.6)
    parser.add_argument("--target_disturbance", type=float, default=0.8)
    parser.add_argument("--target_angular_span", type=float, default=0.8)
    return parser.parse_args()


def load_diffusion(args, device):
    ckpt = torch.load(args.diffusion_checkpoint_path, map_location=device)
    ckpt_args = ckpt.get("args", {})
    bounds_path = os.path.join(os.path.dirname(args.diffusion_checkpoint_path), "design_bounds.npz")
    bounds = DesignBounds.from_npz(bounds_path) if os.path.exists(bounds_path) else DesignBounds.defaults()
    unet = ConditionalUnet1D(
        input_dim=1,
        global_cond_dim=9,
        down_dims=[128, 256],
        diffusion_step_embed_dim=32,
    )
    scheduler = DDIMScheduler(
        num_train_timesteps=int(ckpt_args.get("num_train_timesteps", args.num_train_timesteps)),
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    model = SquirrelDesignDiffusion(
        noise_pred_net=unet,
        noise_scheduler=scheduler,
        bounds=bounds,
        num_inference_steps=args.num_inference_steps,
    ).to(device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    if "ema" in ckpt:
        model.ema.load_state_dict(ckpt["ema"], device)
    model.eval()
    model.copy_ema_to_model()
    return model


def load_dynamics(path, device):
    if path is None:
        return None
    model = ProfileForward2DModel(W=256, task_ch=3, design_ch=16, init_ch=3, output_ch=3).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def objective_from_pred(pred, objective):
    contacts = pred[:, 0]
    disturbance = pred[:, 1]
    angular_span = pred[:, 2]
    if objective == "disturbance":
        return disturbance
    if objective == "contact":
        return contacts
    if objective == "angular_span":
        return angular_span
    if objective == "disturbance_contact_span":
        return disturbance + 0.1 * contacts + 0.5 * angular_span
    raise ValueError(f"Unknown objective: {objective}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    diffusion = load_diffusion(args, device)
    dynamics_model = load_dynamics(args.dynamics_checkpoint_path, device)

    all_design_physical = []
    all_design_norm = []
    all_pred = []
    generator = torch.Generator(device=device).manual_seed(args.seed)

    remaining = args.num_samples
    while remaining > 0:
        current_bs = min(args.batch_size, remaining)
        cond = make_condition_batch(
            batch_size=current_bs,
            approach_deg=args.approach_deg,
            landing_approach_deg=args.landing_approach_deg,
            cyl_radius=args.cyl_rad,
            landing_height=args.landing_height,
            landing_speed=args.landing_speed,
            initial_x_gap=args.initial_x_gap,
            target_contacts=args.target_contacts,
            target_disturbance=args.target_disturbance,
            target_angular_span=args.target_angular_span,
            device=device,
        )
        out = diffusion.sample(
            cond=cond,
            dynamics_model=dynamics_model,
            guidance_scale=args.guidance_scale,
            guidance_objective=args.guidance_objective,
            generator=generator,
        )
        all_design_physical.append(out["design_physical"].detach().cpu().numpy())
        all_design_norm.append(out["design_norm"].detach().cpu().numpy())

        if dynamics_model is not None:
            with torch.no_grad():
                timesteps = torch.zeros(current_bs, dtype=torch.float32, device=device)
                pred = dynamics_model(cond[:, 0:3], out["design_norm"], cond[:, 3:6], timesteps)
            all_pred.append(pred.detach().cpu().numpy())
        remaining -= current_bs

    design_physical = np.concatenate(all_design_physical, axis=0)
    design_norm = np.concatenate(all_design_norm, axis=0)
    pred_metrics = np.concatenate(all_pred, axis=0) if all_pred else None
    task_params = np.tile(
        np.asarray([[args.approach_deg, args.landing_approach_deg, args.cyl_rad]], dtype=np.float32),
        (design_physical.shape[0], 1),
    )

    if pred_metrics is not None:
        scores = objective_from_pred(pred_metrics, args.guidance_objective)
        top_ids = np.argsort(scores)[-min(args.top_k, len(scores)):][::-1]
    else:
        scores = None
        top_ids = np.arange(min(args.top_k, design_physical.shape[0]))

    output_path = os.path.join(args.save_dir, "generated_candidates.npz")
    np.savez_compressed(
        output_path,
        design_params=design_physical,
        design_params_norm=design_norm,
        task_params=task_params,
        pred_metrics=pred_metrics,
        scores=scores,
        top_ids=top_ids,
        top_design_params=design_physical[top_ids],
        top_task_params=task_params[top_ids],
    )
    print("[SAVED]", os.path.abspath(output_path))
    print("[TOP IDS]", top_ids.tolist())
    if scores is not None:
        print("[TOP SCORES]", scores[top_ids].tolist())


if __name__ == "__main__":
    main()
