"""Small, resumable simulator experiment for local pose guidance (no retraining)."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from benchmarks.baselines.surrogate_search import _scenario_tensors, load_surrogate
from benchmarks.run_sim_benchmark import execute_job, json_safe, stable_run_id
from dynamics.pose_targets import pose_target_from_npz
from generator.dataloader import (DesignBounds, diffusion_to_physical,
    physical_to_diffusion, physical_to_model_norm, project_physical_design,
    variable_design_mask)


def pose_loss(pose, radius, clearance, scale, target=None):
    """Squared distance in m², excluding the approach/base keypoint.

    Surface fit is a geometric diagnostic, not a force-closure objective.
    An optional fixed reference pose defines a pose-matching experiment instead.
    """
    points = pose.reshape(-1, 5, 2)[:, 1:] * scale
    if target is not None:
        return (points - target.reshape(-1, 5, 2)[:, 1:] * scale).square().sum(-1).mean(-1)
    return (torch.linalg.vector_norm(points, dim=-1) - radius - clearance).square().mean(-1)


def change_stats(predicted, observed, floor_m):
    pn, sn = float(np.linalg.norm(predicted)), float(np.linalg.norm(observed))
    resolved = pn > floor_m and sn > floor_m
    return {"predicted_change_norm_mm": pn * 1000,
            "simulated_change_norm_mm": sn * 1000,
            "change_rmse_mm": float(np.sqrt(np.mean((predicted-observed)**2)) * 1000),
            "direction_resolved": resolved,
            "change_cosine": float(np.dot(predicted, observed)/(pn*sn)) if resolved else None}


def write_json(path, value):
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True, help="Existing benchmark run tree")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scenario", default="approach_radius:00@nominal")
    p.add_argument("--methods", default="conditional_diffusion,pose_dgdm_gs1")
    p.add_argument("--anchors", type=int, default=1, help="Random anchors per method; never ranked by outcome")
    p.add_argument("--steps", default="0.01,0.03", help="L2 steps in [-1,1] design coordinates")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--timeout", type=int, default=1200, help="Per simulation timeout in seconds")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--device", default="cpu")
    p.add_argument("--noise-conditioned", action="store_true", help="Evaluate a noisy checkpoint at t=0 only")
    p.add_argument("--clearance-m", type=float, default=0.01, help="Fixed centerline clearance from branch surface")
    p.add_argument("--target-npz", type=Path, help="Optional same-scenario reference pose, fixed before evaluation")
    p.add_argument("--pose-floor-mm", type=float, default=0.01)
    p.add_argument("--loss-floor-m2", type=float, default=1e-8)
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    steps = [float(x) for x in args.steps.split(",")]
    if args.anchors < 1 or args.workers < 1 or args.timeout <= 0 or not steps or min(steps) <= 0:
        p.error("anchors, workers, timeout and steps must be positive")
    if args.pose_floor_mm <= 0 or args.loss_floor_m2 < 0 or args.clearance_m < 0:
        p.error("Invalid tolerance or clearance")
    args.output.mkdir(parents=True, exist_ok=True)
    model = load_surrogate(args.checkpoint, device=args.device,
                           expected_noise_conditioned=args.noise_conditioned)
    if model.target_representation != "pose_keypoints":
        raise ValueError("This experiment requires a pose-keypoint checkpoint")
    scale = model.pose_scale_m
    bounds = DesignBounds.defaults()
    target = None
    if args.target_npz:
        with np.load(args.target_npz) as data:
            target = torch.tensor(pose_target_from_npz(data, scale), device=args.device)
    rng = np.random.default_rng(args.seed)
    methods = args.methods.split(",")
    sources = {method: [] for method in methods}
    for path in sorted(args.source.rglob("benchmark_result.json")):
        result = json.loads(path.read_text())
        method = result.get("method")
        if method in sources and result.get("scenario_id") == args.scenario and result.get("status") == "ok":
            sources[method].append(path)
    jobs, experiments = [], []
    for method, paths in sources.items():
        if len(paths) < args.anchors:
            raise ValueError(f"Need {args.anchors} successful source designs for {method}/{args.scenario}; found {len(paths)}")
        for index in rng.choice(len(paths), args.anchors, replace=False):
            source = paths[int(index)]
            source_job = json.loads(source.with_name("benchmark_job.json").read_text())
            scenario = source_job["scenario"]
            task, init = _scenario_tensors([scenario], args.device)
            physical = torch.tensor(source_job["design_params"], device=args.device).unsqueeze(0)
            unit = physical_to_diffusion(project_physical_design(physical, bounds), bounds).detach().requires_grad_(True)

            def predict(u):
                physical = project_physical_design(diffusion_to_physical(u, bounds), bounds)
                coords = physical_to_diffusion(physical, bounds) if model.design_coordinates == "diffusion_unit" else physical_to_model_norm(physical)
                return model(task, coords, init, torch.zeros(1, device=args.device))

            def loss(pred):
                return pose_loss(pred, scenario["params"]["cyl_rad"], args.clearance_m, scale, target)

            pred = predict(unit)
            grad = torch.autograd.grad(loss(pred).sum(), unit)[0] * variable_design_mask(bounds, args.device)
            norm = torch.linalg.vector_norm(grad)
            if not torch.isfinite(norm) or norm < 1e-12:
                raise ValueError(f"Unusable pose gradient for {source}")
            descent = -grad / norm
            random = torch.tensor(rng.normal(size=unit.shape), dtype=unit.dtype, device=args.device)
            random *= variable_design_mask(bounds, args.device)
            random /= torch.linalg.vector_norm(random)
            anchor_id = hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:12]
            variants = [("base", 0., torch.zeros_like(unit))]
            variants += [(name, step, direction) for step in steps
                         for name, direction in (("descent", descent), ("reverse", -descent), ("random", random))]
            for name, step, direction in variants:
                u = (unit.detach() + step * direction).clamp(-1, 1)
                design = project_physical_design(diffusion_to_physical(u, bounds), bounds)
                effective = physical_to_diffusion(design, bounds)
                prediction = predict(effective).detach()
                cid = f"{anchor_id}_{name}_{step:g}"
                design_list = design[0].detach().cpu().tolist()
                rid = stable_run_id("pose_pilot", args.seed, cid, args.scenario, design_list, scenario["params"])
                job = dict(run_id=rid, method="pose_pilot", seed=args.seed, candidate_id=cid,
                           scenario_id=args.scenario, family=scenario["family"], scenario=scenario,
                           design_params=design_list, run_dir=str((args.output / "runs" / rid).resolve()))
                jobs.append(job)
                experiments.append(dict(run_id=rid, anchor=anchor_id, source=str(source.resolve()),
                    source_method=method, variant=name, step=step,
                    effective_step_norm=float(torch.linalg.vector_norm(effective-unit.detach())),
                    predicted_pose=prediction.cpu().flatten().tolist(), predicted_loss_m2=float(loss(prediction)),
                    radius_m=scenario["params"]["cyl_rad"]))
    manifest = dict(args={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        target_pose=None if target is None else target.cpu().tolist(),
        pose_scale_m=scale, experiments=experiments, jobs=jobs)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        # Preparing then executing, or changing parallelism, is safe. Scientific changes need a new directory.
        for obj in (old, manifest):
            for key in ("prepare_only", "workers", "timeout"):
                obj["args"].pop(key, None)
        if old != manifest:
            raise ValueError("Experiment changed; use a new output directory")
    write_json(manifest_path, manifest)
    print(f"Prepared {len(jobs)} simulations; manifest: {manifest_path}", flush=True)
    if args.prepare_only:
        return
    results = {}
    weights = {"contact_coverage_norm": .2, "disturbance_resistance_score": .45, "angular_span_norm": .35}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute_job, j, weights, args.timeout, sys.executable, False, False): j for j in jobs}
        for future in as_completed(futures):
            result = future.result()
            results[result["run_id"]] = result
            print(f"[{len(results)}/{len(jobs)}] {result['candidate_id']}: {result['status']} ({result.get('elapsed_seconds', 0):.1f}s)", flush=True)
    rows, bases = [], {}
    for exp in experiments:
        result = results[exp["run_id"]]
        row = {**exp, "status": result["status"]}
        if result["status"] == "ok":
            with np.load(result["master_log_path"]) as data:
                sim = pose_target_from_npz(data, scale)
            row["simulated_pose"] = sim.tolist()
            row["simulated_loss_m2"] = float(pose_loss(torch.tensor(sim), exp["radius_m"], args.clearance_m,
                                                       scale, None if target is None else target.cpu()))
            row["absolute_pose_rmse_mm"] = float(np.sqrt(np.mean((sim-np.array(exp["predicted_pose"]))**2))*scale*1000)
        if exp["variant"] == "base":
            bases[exp["anchor"]] = row
        rows.append(row)
    for row in rows:
        base = bases[row["anchor"]]
        if row["variant"] == "base" or row["status"] != "ok" or base["status"] != "ok":
            continue
        dp = (np.array(row["predicted_pose"])-base["predicted_pose"]) * scale
        ds = (np.array(row["simulated_pose"])-base["simulated_pose"]) * scale
        row.update(change_stats(dp, ds, args.pose_floor_mm / 1000))
        row["predicted_loss_change_m2"] = row["predicted_loss_m2"]-base["predicted_loss_m2"]
        row["simulated_loss_change_m2"] = row["simulated_loss_m2"]-base["simulated_loss_m2"]
        row["simulated_loss_improved"] = row["simulated_loss_change_m2"] < -args.loss_floor_m2
        row["predicted_loss_improved"] = row["predicted_loss_change_m2"] < -args.loss_floor_m2
        row["loss_direction_resolved"] = min(abs(row["predicted_loss_change_m2"]), abs(row["simulated_loss_change_m2"])) > args.loss_floor_m2
        row["loss_sign_correct"] = bool(np.sign(row["predicted_loss_change_m2"]) == np.sign(row["simulated_loss_change_m2"])) if row["loss_direction_resolved"] else None
    summaries = []
    for method in methods:
        for step in steps:
            for variant in ("descent", "reverse", "random"):
                group = [r for r in rows if r["source_method"] == method and r["step"] == step and r["variant"] == variant]
                valid = [r for r in group if "simulated_loss_improved" in r]
                cosines = [r["change_cosine"] for r in valid if r["change_cosine"] is not None]
                summaries.append(dict(method=method, step=step, variant=variant, attempted=len(group),
                    valid_pairs=len(valid), resolved_directions=len(cosines),
                    mean_cosine=np.mean(cosines) if cosines else None,
                    improvement_fraction=np.mean([r["simulated_loss_improved"] for r in valid]) if valid else None))
    write_json(args.output / "pose_results.json", rows)
    write_json(args.output / "summary.json", summaries)
    print(json.dumps(json_safe(summaries), indent=2))


if __name__ == "__main__":
    main()
