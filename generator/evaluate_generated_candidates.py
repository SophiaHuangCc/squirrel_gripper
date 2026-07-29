import argparse
import os
import sys
from os.path import join as pjoin

import numpy as np

BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)
sys.path.insert(0, pjoin(BASEPATH, ".."))

from dynamics.sim_test_mj import sim_test_batch


def objective_score(pred_metrics, objective):
    contacts = pred_metrics[:, 0]
    disturbance = pred_metrics[:, 1]
    angular_span = pred_metrics[:, 2]
    curl_speed = pred_metrics[:, 3]

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
    if objective == "curl_speed":
        return curl_speed
    if objective == "disturbance_contact_span_speed":
        gate = 1.0 / (1.0 + np.exp(-(contacts - 0.3) / 0.05))
        return disturbance + 0.1 * contacts + 0.5 * angular_span + 0.1 * curl_speed * gate
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
    parser.add_argument(
        "--damping",
        type=float,
        default=1.0,
        help="Internal damping passed to TendonForces/finger.py.",
    )
    parser.add_argument(
        "--nu_contact",
        type=float,
        default=30.0,
        help="Contact damping passed to TendonForces/finger.py.",
    )
    parser.add_argument(
        "--vel_damp_contact",
        type=int,
        default=90,
        help="Velocity contact damping passed to TendonForces/finger.py.",
    )
    parser.add_argument(
        "--k_contact",
        type=float,
        default=4000.0,
        help="Contact stiffness passed to TendonForces/finger.py.",
    )
    parser.add_argument(
        "--final_time",
        type=float,
        default=5.0,
        help="Simulation final time passed to TendonForces/finger.py.",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="disturbance_contact_span_speed",
        choices=[
            "disturbance",
            "contact",
            "angular_span",
            "disturbance_contact",
            "disturbance_span",
            "disturbance_contact_span",
            "curl_speed",
            "disturbance_contact_span_speed",
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
    sim_params = {
        "damping": args.damping,
        "nu_contact": args.nu_contact,
        "vel_damp_contact": args.vel_damp_contact,
        "k_contact": args.k_contact,
        "final_time": args.final_time,
    }
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
