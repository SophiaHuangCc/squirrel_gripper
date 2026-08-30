import argparse
import os
import sys
from os.path import join as pjoin

import numpy as np

BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)
sys.path.insert(0, pjoin(BASEPATH, ".."))

from dynamics.sim_test_mj import sim_test_batch


def add_finger_runtime_args(parser):
    parser.add_argument("--E", type=float, default=2e7)
    parser.add_argument("--damping", type=float, default=1.0)
    parser.add_argument("--n_elements", type=int, default=100)
    parser.add_argument("--final_time", type=float, default=5.0)
    parser.add_argument("--k_contact", type=float, default=4000.0)
    parser.add_argument("--max_penetration_warn", type=float, default=0.002)
    parser.add_argument("--nu_contact", type=float, default=30.0)
    parser.add_argument("--mu_contact", type=float, default=0.8)
    parser.add_argument("--vel_damp_contact", type=int, default=90)
    parser.add_argument("--poisson_nu", type=float, default=0.4)
    parser.add_argument("--v_mass", type=float, default=0.002)
    parser.add_argument("--body_mass", type=float, default=0.5)
    parser.add_argument(
        "--joint_stiffness_mode",
        choices=["full_material", "bending_only"],
        default="bending_only",
    )
    parser.add_argument(
        "--distal_tendon_anchor",
        choices=["none", "tip"],
        default="none",
    )
    parser.add_argument("--distal_tendon_anchor_node", type=int, default=-1)
    parser.add_argument(
        "--joint_lengths", type=str, default=None,
        help="Legacy override only; generated From Links joint lengths are used by default.",
    )
    parser.add_argument("--landing_speed", type=float, default=0.0)
    parser.add_argument("--landing_height", type=float, default=0.04)
    parser.add_argument("--initial_x_gap", type=float, default=0.06)
    parser.add_argument("--landing_approach_deg", type=float, default=30.0)
    parser.add_argument(
        "--prescribed_stop_at_contact",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--prescribed_contact_margin", type=float, default=-0.005)
    parser.add_argument("--min_tension", type=float, default=0.1)
    parser.add_argument("--max_tension", type=float, default=20.0)
    parser.add_argument("--disturbance_force_mag", type=float, default=5.0)
    parser.add_argument("--disturbance_base_nodes", type=int, default=5)
    parser.add_argument("--disturbance_steps", type=int, default=100)
    parser.add_argument("--disturbance_dt_scale", type=float, default=1.0)
    parser.add_argument("--continuous_disturbance_metric", action="store_true")


def collect_finger_runtime_args(args):
    keys = [
        "E",
        "damping",
        "n_elements",
        "final_time",
        "k_contact",
        "max_penetration_warn",
        "nu_contact",
        "mu_contact",
        "vel_damp_contact",
        "poisson_nu",
        "v_mass",
        "body_mass",
        "joint_stiffness_mode",
        "distal_tendon_anchor",
        "distal_tendon_anchor_node",
        "joint_lengths",
        "landing_speed",
        "landing_height",
        "initial_x_gap",
        "landing_approach_deg",
        "prescribed_stop_at_contact",
        "prescribed_contact_margin",
        "min_tension",
        "max_tension",
        "disturbance_force_mag",
        "disturbance_base_nodes",
        "disturbance_steps",
        "disturbance_dt_scale",
        "continuous_disturbance_metric",
    ]
    return {key: getattr(args, key) for key in keys}


def objective_score(pred_metrics, objective):
    contacts = pred_metrics[:, 0]
    disturbance = pred_metrics[:, 1]
    angular_span = pred_metrics[:, 2]

    if objective == "disturbance":
        return disturbance
    if objective == "contact":
        return contacts
    if objective == "angular_span":
        return angular_span
    if objective == "disturbance_contact":
        return disturbance + 0.1 * contacts
    if objective == "disturbance_span":
        return disturbance + 0.5 * angular_span
    if objective == "disturbance_contact_span":
        return disturbance + 0.1 * contacts + 0.5 * angular_span
    raise ValueError(f"Unknown objective: {objective}")


def main():
    parser = argparse.ArgumentParser(
        description="Render/verify final diffusion-generated squirrel finger candidates."
    )
    parser.add_argument(
        "--candidate_path",
        type=str,
        default="generator/runs/sample_exp3/generated_candidates.npz",
        help="Path to generator/sample.py output.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to save rendered simulations and verification_results.npz.",
    )
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--num_cpus", type=int, default=4)
    parser.add_argument("--render", action="store_true")
    add_finger_runtime_args(parser)
    parser.add_argument(
        "--objective",
        type=str,
        default="disturbance_contact_span",
        choices=[
            "disturbance",
            "contact",
            "angular_span",
            "disturbance_contact",
            "disturbance_span",
            "disturbance_contact_span",
        ],
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.candidate_path), "sim_verification")
    os.makedirs(args.output_dir, exist_ok=True)

    data = np.load(args.candidate_path, allow_pickle=True)

    design_params = data["design_params"]
    task_params = data["task_params"]

    if "top_ids" in data:
        top_ids = data["top_ids"][: args.top_k]
    elif "pred_metrics" in data and data["pred_metrics"] is not None:
        scores = objective_score(data["pred_metrics"], args.objective)
        top_ids = np.argsort(scores)[-args.top_k:][::-1]
    else:
        top_ids = np.arange(min(args.top_k, design_params.shape[0]))

    selected_designs = design_params[top_ids]
    selected_tasks = task_params[top_ids]
    selected_preds = data["pred_metrics"][top_ids] if "pred_metrics" in data and data["pred_metrics"] is not None else None
    selected_scores = data["scores"][top_ids] if "scores" in data and data["scores"] is not None else None

    np.savez_compressed(
        os.path.join(args.output_dir, "selected_generated_candidates.npz"),
        selected_ids=top_ids,
        design_params=selected_designs,
        task_params=selected_tasks,
        pred_metrics=selected_preds,
        pred_scores=selected_scores,
    )

    print(f"[GEN EVAL] Candidate file: {args.candidate_path}")
    print(f"[GEN EVAL] Selected ids: {top_ids}")
    print(f"[GEN EVAL] Saving verification to: {args.output_dir}")
    sim_params = collect_finger_runtime_args(args)
    print(f"[GEN EVAL] finger.py sim params: {sim_params}")

    metrics, save_dirs = sim_test_batch(
        design_params=selected_designs,
        task_params=selected_tasks,
        save_dir=args.output_dir,
        num_cpus=args.num_cpus,
        render=args.render,
        sim_params=sim_params,
    )

    np.savez_compressed(
        os.path.join(args.output_dir, "verification_results.npz"),
        selected_ids=top_ids,
        pred_metrics=selected_preds,
        pred_scores=selected_scores,
        sim_metrics=np.asarray(metrics, dtype=object),
        sim_dirs=np.asarray(save_dirs, dtype=object),
    )

    print(f"[DONE] Saved verification results to {args.output_dir}")


if __name__ == "__main__":
    main()
