import os
import sys
from os.path import join as pjoin

import argparse
import numpy as np
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)
sys.path.insert(0, pjoin(BASEPATH, '..'))
from dynamics.sim_test_mj import sim_test_batch


def objective_score(pred_metrics, objective):
    contacts = pred_metrics[:, 0]
    disturbance = pred_metrics[:, 1]
    angular_span = pred_metrics[:, 2]

    if objective == "disturbance":
        return disturbance
    elif objective == "contact":
        return contacts
    elif objective == "angular_span":
        return angular_span
    elif objective == "disturbance_contact":
        return disturbance + 0.1 * contacts
    elif objective == "disturbance_span":
        return disturbance + 0.5 * angular_span
    elif objective == "disturbance_contact_span":
        return disturbance + 0.1 * contacts + 0.5 * angular_span
    else:
        raise ValueError(f"Unknown objective: {objective}")


def evaluate_one_objective(args, objective):
    candidate_path = os.path.join(
        args.optimization_dir,
        f"{objective}_surrogate_only",
        "optimized_candidates.npz",
    )

    if not os.path.exists(candidate_path):
        print(f"[SKIP] Missing {candidate_path}")
        return

    data = np.load(candidate_path, allow_pickle=True)

    design_params = data["design_params"]
    task_params = data["task_params"]
    pred_metrics = data["pred_metrics"]

    scores = objective_score(pred_metrics, objective)
    top_ids = np.argsort(scores)[-args.top_k:][::-1]

    selected_designs = design_params[top_ids]
    selected_tasks = task_params[top_ids]
    selected_preds = pred_metrics[top_ids]

    save_dir = os.path.join(
        args.output_dir,
        f"{objective}_verified_top{args.top_k}",
    )
    os.makedirs(save_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(save_dir, "selected_candidates.npz"),
        selected_ids=top_ids,
        design_params=selected_designs,
        task_params=selected_tasks,
        pred_metrics=selected_preds,
        pred_scores=scores[top_ids],
    )

    print(f"\n[EVAL] {objective}")
    print(f"Candidate file: {candidate_path}")
    print(f"Selected ids: {top_ids}")
    print(f"Saving verification to: {save_dir}")

    metrics, save_dirs = sim_test_batch(
        design_params=selected_designs,
        task_params=selected_tasks,
        save_dir=save_dir,
        num_cpus=args.num_cpus,
        render=args.render,
    )

    np.savez_compressed(
        os.path.join(save_dir, "verification_results.npz"),
        selected_ids=top_ids,
        pred_metrics=selected_preds,
        pred_scores=scores[top_ids],
        sim_metrics=np.asarray(metrics, dtype=object),
        sim_dirs=np.asarray(save_dirs, dtype=object),
    )

    print(f"[DONE] Saved verification results to {save_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimization_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--num_cpus", type=int, default=4)
    parser.add_argument("--render", action="store_true")

    parser.add_argument(
        "--objectives",
        type=str,
        default="disturbance,disturbance_contact,contact,angular_span,disturbance_span,disturbance_contact_span",
    )

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.optimization_dir, "sim_verification")

    objectives = [x.strip() for x in args.objectives.split(",") if x.strip()]

    for objective in objectives:
        evaluate_one_objective(args, objective)


if __name__ == "__main__":
    main()