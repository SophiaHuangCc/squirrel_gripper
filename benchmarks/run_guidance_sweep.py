"""Paired guidance-scale sweep for the current conditional DGDM adaptation."""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

from benchmarks.candidates import save_candidates
from benchmarks.protocol import DEFAULT_CONFIG, load_config


DEFAULT_SCALES = (0.0, 0.1, 1.0, 2.0, 10.0)


def parse_float_list(text):
    values = tuple(float(value.strip()) for value in text.split(",") if value.strip())
    if not values:
        raise ValueError("At least one guidance scale is required")
    if any(value < 0 for value in values):
        raise ValueError("Guidance scales must be nonnegative")
    if len(set(values)) != len(values):
        raise ValueError("Guidance scales must be unique")
    return values


def parse_int_list(text, defaults):
    return tuple(int(value.strip()) for value in text.split(",") if value.strip()) if text else tuple(defaults)


def scale_slug(scale):
    return f"{float(scale):g}".replace("-", "m").replace(".", "p")


def method_name(scale):
    # This path uses the task/target-conditioned generator. Keep its name
    # distinct from the separate unconditional DGDM implementation.
    return f"conditional_dgdm_gs{scale_slug(scale)}"


def run_benchmark(args, candidate_path, method, seed):
    output = args.output_dir / "runs" / f"{method}_s{seed}"
    command = [
        sys.executable, "-m", "benchmarks.run_sim_benchmark",
        "--candidates", str(candidate_path), "--output_dir", str(output),
        "--config", str(args.config), "--top_k", str(args.benchmark_top_k),
        "--num_workers", str(args.num_workers), "--timeout", str(args.timeout),
        "--python", sys.executable,
    ]
    if args.families:
        command.extend(("--families", args.families))
    if args.render:
        command.append("--render")
    if args.dry_run:
        command.append("--dry_run")
    print("[RUN]", " ".join(command))
    subprocess.run(command, check=True)


def main():
    # Delay optional ML imports so helper functions can be tested without the
    # diffusion training environment installed.
    from benchmarks.baselines.diffusion_search import diffusion_search, load_diffusion
    from benchmarks.baselines.surrogate_search import load_surrogate

    parser = argparse.ArgumentParser(
        description="Generate and optionally simulate a paired conditional-DGDM guidance-scale sweep."
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--diffusion_checkpoint", type=Path, required=True)
    parser.add_argument("--dynamics_checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scales", default=",".join(str(x) for x in DEFAULT_SCALES))
    parser.add_argument("--seeds", default="", help="Defaults to method_seeds in the benchmark config")
    parser.add_argument("--candidate_budget", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--target_contacts", type=float, default=0.8)
    parser.add_argument("--target_disturbance", type=float, default=0.8)
    parser.add_argument("--target_angular_span", type=float, default=0.8)
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--target_scenario_id")
    target.add_argument("--target_family")
    target.add_argument("--generalist", action="store_true")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cuda")
    parser.add_argument("--run_benchmark", action="store_true")
    parser.add_argument("--benchmark_top_k", type=int, default=1)
    parser.add_argument("--families", default="")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    scales = parse_float_list(args.scales)
    seeds = parse_int_list(args.seeds, config["evaluation"]["method_seeds"])
    budget = args.candidate_budget or int(config["evaluation"]["candidate_budget"])
    if budget < 1 or args.num_samples < budget:
        parser.error("num_samples must be at least candidate_budget, and both must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir = args.output_dir / "candidates"
    candidate_dir.mkdir(exist_ok=True)
    surrogate = load_surrogate(args.dynamics_checkpoint, device=args.device)
    diffusion = load_diffusion(
        args.diffusion_checkpoint, device=args.device,
        num_inference_steps=args.inference_steps,
    )
    manifest = {
        "variant": "task_and_target_conditioned_diffusion_with_dynamics_guidance",
        "paired_initial_noise": True,
        "scales": scales, "seeds": seeds, "candidate_budget": budget,
        "num_samples": args.num_samples, "inference_steps": args.inference_steps,
        "diffusion_checkpoint": str(args.diffusion_checkpoint.resolve()),
        "dynamics_checkpoint": str(args.dynamics_checkpoint.resolve()),
        "runs": [],
    }
    proposal_times = []
    for seed in seeds:
        for scale in scales:
            method = method_name(scale)
            started = time.perf_counter()
            result = diffusion_search(
                diffusion, surrogate, config, budget, args.num_samples, seed,
                batch_size=args.batch_size, guidance_scale=scale,
                num_inference_steps=args.inference_steps,
                target_contacts=args.target_contacts,
                target_disturbance=args.target_disturbance,
                target_angular_span=args.target_angular_span,
                scenario_id=args.target_scenario_id, family=args.target_family,
                generalist=args.generalist, device=args.device,
            )
            elapsed = time.perf_counter() - started
            path = candidate_dir / f"{method}_s{seed}.npz"
            metadata = {
                "variant": "conditional_dgdm", "guidance_scale": scale,
                "paired_noise_seed": seed, "num_samples": args.num_samples,
                "candidate_budget": budget, "num_inference_steps": args.inference_steps,
                "model_evaluations": result.model_evaluations,
                "target_scenario_ids": result.target_scenario_ids,
                "proposal_elapsed_seconds": elapsed,
                "selection_rule": "surrogate_benchmark_utility",
                "diffusion_checkpoint": str(args.diffusion_checkpoint.resolve()),
                "dynamics_checkpoint": str(args.dynamics_checkpoint.resolve()),
            }
            save_candidates(path, result.designs, method, seed, scores=result.scores, metadata=metadata)
            manifest["runs"].append({"method": method, "scale": scale, "seed": seed,
                                     "candidate_file": str(path.resolve()), **metadata})
            proposal_times.append({
                "method": method,
                "seed": seed,
                "guidance_scale": scale,
                "proposal_elapsed_seconds": elapsed,
                "candidate_file": str(path.resolve()),
            })
            print(f"[GENERATED] {method} seed={seed} seconds={elapsed:.2f}")
            if args.run_benchmark:
                run_benchmark(args, path, method, seed)
    manifest_path = args.output_dir / "guidance_sweep_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (args.output_dir / "proposal_times.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(proposal_times[0]))
        writer.writeheader()
        writer.writerows(proposal_times)
    print("[MANIFEST]", manifest_path.resolve())
    if args.run_benchmark and not args.dry_run:
        print("[NEXT] summarize with:")
        print(f"{sys.executable} -m benchmarks.summarize "
              f"{args.output_dir}/runs/*/records.jsonl --output_dir {args.output_dir}/summary")


if __name__ == "__main__":
    main()
