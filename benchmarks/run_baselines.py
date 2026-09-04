"""Generate common baseline candidates and optionally run the simulation suite."""

import argparse
import concurrent.futures
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

from benchmarks.baselines.random_search import sample_feasible_designs
from benchmarks.baselines.reference import reference_design
from benchmarks.baselines.retrieval import retrieve
from benchmarks.baselines.surrogate_search import (
    adam_search, cma_es_search, load_surrogate, rank_designs, select_target_cells,
)
from benchmarks.candidates import load_candidates, save_candidates
from benchmarks.protocol import DEFAULT_CONFIG, expand_core_scenarios, load_config


UTILITY_PROFILES = {
    "contact_only": {
        "disturbance_resistance_score": 0.0,
        "contact_coverage_norm": 1.0,
        "angular_span_norm": 0.0,
    },
    "disturbance_only": {
        "disturbance_resistance_score": 1.0,
        "contact_coverage_norm": 0.0,
        "angular_span_norm": 0.0,
    },
}


def parse_adapter(spec):
    if "=" not in spec:
        raise ValueError("--adapt must be METHOD=PATH")
    method, path = spec.split("=", 1)
    return method.strip(), Path(path).expanduser()


def run_checked(command):
    print("[RUN]", " ".join(str(part) for part in command))
    subprocess.run(command, check=True)


def replace_cli_option(argv, flag, value):
    argv = list(argv)
    if flag in argv:
        index = argv.index(flag)
        if index + 1 >= len(argv):
            raise ValueError(f"{flag} is missing its value")
        argv[index + 1] = str(value)
    else:
        argv.extend([flag, str(value)])
    return argv


def apply_utility_override(config, profile, custom_weights):
    if profile != "combined" and custom_weights is not None:
        raise ValueError("Choose either --utility_profile or --utility_weights, not both")
    if custom_weights is not None:
        values = [float(value) for value in custom_weights.split(",") if value.strip()]
        if len(values) != 3:
            raise ValueError("--utility_weights must be D,C,A")
        if any(value < 0 for value in values) or abs(sum(values) - 1.0) > 1e-6:
            raise ValueError("utility weights must be nonnegative and sum to 1")
        weights = {
            "disturbance_resistance_score": values[0],
            "contact_coverage_norm": values[1],
            "angular_span_norm": values[2],
        }
    elif profile in UTILITY_PROFILES:
        weights = UTILITY_PROFILES[profile]
    else:
        weights = config["evaluation"]["utility_weights"]
    config["evaluation"]["utility_weights"] = dict(weights)
    config["evaluation"]["utility_profile"] = profile if custom_weights is None else "custom"
    return weights


def write_specialist_sweep_summary(output_dir, sweep_rows):
    rows = []
    for sweep in sweep_rows:
        path = Path(sweep["output_dir"]) / "summary" / "method_summary.csv"
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                rows.append({
                    "target_scenario_id": sweep["scenario_id"],
                    "target_approach_deg": sweep["params"]["approach_deg"],
                    "target_cyl_rad": sweep["params"]["cyl_rad"],
                    **row,
                })
    if not rows:
        return
    fields = list(rows[0])
    with open(output_dir / "specialist_method_summary.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    grouped = {}
    for row in rows:
        grouped.setdefault(row["method"], []).append(row)
    aggregate = []
    for method, method_rows in sorted(grouped.items()):
        values = [float(row["best_mean_utility"]) for row in method_rows]
        aggregate.append({
            "method": method,
            "num_target_seed_rows": len(values),
            "num_specialist_targets": len({row["target_scenario_id"] for row in method_rows}),
            "num_method_seeds": len({row["seed"] for row in method_rows}),
            "mean_target_utility": sum(values) / len(values),
            "std_target_utility": statistics.pstdev(values),
        })
    with open(output_dir / "specialist_method_aggregate.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)


def main():
    parser = argparse.ArgumentParser(description="Prepare and run Squirrel Benchmark V1 baselines.")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--methods", type=str, default="reference,random,retrieval")
    parser.add_argument("--candidate_budget", type=int, default=None)
    parser.add_argument("--seeds", type=str, default="", help="Comma-separated random seeds")
    parser.add_argument("--retrieval_data_dir", type=Path, default=None)
    parser.add_argument("--retrieval_scenario_id", type=str, default=None)
    parser.add_argument("--retrieval_family", type=str, default=None)
    parser.add_argument("--dynamics_checkpoint", type=Path, default=None)
    parser.add_argument(
        "--dgdm_dynamics_checkpoint", type=Path, default=None,
        help=("Noise-and-timestep-conditioned dynamics checkpoint trained with "
              "dynamics/main.py --use_design_noise; used only for DGDM guidance."),
    )
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--target_scenario_id", type=str, default=None)
    parser.add_argument("--target_family", type=str, default=None)
    parser.add_argument(
        "--generalist", action="store_true",
        help="Optimize mean surrogate utility over all core scenarios.",
    )
    parser.add_argument(
        "--utility_profile",
        choices=("combined", "contact_only", "disturbance_only"),
        default="combined",
        help="Selection and evaluation objective; combined uses the config weights.",
    )
    parser.add_argument(
        "--utility_weights", type=str, default=None, metavar="D,C,A",
        help="Custom nonnegative disturbance, contact, angular weights summing to one.",
    )
    parser.add_argument("--adam_steps", type=int, default=300)
    parser.add_argument("--adam_lr", type=float, default=0.03)
    parser.add_argument("--cma_generations", type=int, default=100)
    parser.add_argument("--cma_popsize", type=int, default=32)
    parser.add_argument("--cma_sigma", type=float, default=0.5)
    parser.add_argument("--diffusion_checkpoint", type=Path, default=None)
    parser.add_argument(
        "--unconditional_diffusion_checkpoint", type=Path, default=None,
        help="Checkpoint trained with generator/train.py --conditioning unconditional.",
    )
    parser.add_argument("--diffusion_num_samples", type=int, default=256)
    parser.add_argument("--diffusion_batch_size", type=int, default=256)
    parser.add_argument("--diffusion_inference_steps", type=int, default=20)
    parser.add_argument("--dgdm_guidance_scale", type=float, default=0.1)
    parser.add_argument(
        "--dgdm_guidance_timesteps", type=str, default="",
        help=(
            "Optional comma-separated DDIM training-timestep indices at which "
            "guidance is injected, e.g. 0,3,6 for late-step-only guidance."
        ),
    )
    parser.add_argument(
        "--dgdm_method_label", type=str, default="dgdm",
        help=(
            "Method label stored in candidate/results files for a DGDM run. "
            "Use distinct labels such as dgdm_gs0p1 when comparing scales."
        ),
    )
    parser.add_argument("--target_contacts", type=float, default=0.8)
    parser.add_argument("--target_disturbance", type=float, default=0.8)
    parser.add_argument("--target_angular_span", type=float, default=0.8)
    parser.add_argument(
        "--random_pool_size", type=int, default=256,
        help="Feasible samples proposed before surrogate ranking in random_search.",
    )
    parser.add_argument(
        "--surrogate_eval_budget", type=int, default=None,
        help="Optional equal budget of candidate-scenario surrogate predictions per method.",
    )
    parser.add_argument(
        "--adapt", action="append", default=[], metavar="METHOD=PATH",
        help="Adapt an existing generator/optimizer candidate NPZ to the common schema",
    )
    parser.add_argument("--run_benchmark", action="store_true")
    parser.add_argument(
        "--benchmark_top_k", type=int, default=None,
        help=(
            "Number of generated candidates to verify in the simulator. "
            "The default verifies the complete saved candidate pool."
        ),
    )
    parser.add_argument(
        "--evaluation_scope", choices=("auto", "target", "all"), default="auto",
        help=(
            "auto evaluates a specialist only on its selection target and a generalist on all cells; "
            "target always evaluates selection cells; all measures transfer over the complete grid"
        ),
    )
    parser.add_argument("--families", type=str, default="")
    parser.add_argument(
        "--num_workers", type=int, default=1,
        help=(
            "Maximum concurrent candidate benchmark subprocesses. Each subprocess "
            "runs one rollout at a time, preventing nested worker oversubscription."
        ),
    )
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if not args.dgdm_method_label or not all(
        char.isalnum() or char in "_.-" for char in args.dgdm_method_label
    ):
        raise ValueError("--dgdm_method_label may contain only letters, numbers, _, -, and .")

    source_config = args.config.resolve()
    config = load_config(source_config)
    weights = apply_utility_override(config, args.utility_profile, args.utility_weights)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    effective_config = args.output_dir / "effective_config.json"
    effective_config.write_text(json.dumps(config, indent=2), encoding="utf-8")
    args.config = effective_config.resolve()
    print(f"[UTILITY] D={weights['disturbance_resistance_score']:.3f} "
          f"C={weights['contact_coverage_norm']:.3f} A={weights['angular_span_norm']:.3f}")
    print(f"[CONFIG] source={source_config} effective={args.config}")

    if args.target_scenario_id == "all":
        if args.generalist or args.target_family is not None:
            raise ValueError("--target_scenario_id all cannot be combined with another target mode")
        cells = expand_core_scenarios(config)
        sweep_root = args.output_dir / "specialists"
        sweep_rows = []
        for cell in cells:
            scenario_id = cell["scenario_id"]
            cell_dir = sweep_root / scenario_id.replace(":", "-")
            child_argv = list(sys.argv[1:])
            child_argv = replace_cli_option(child_argv, "--output_dir", cell_dir)
            child_argv = replace_cli_option(child_argv, "--config", args.config)
            child_argv = replace_cli_option(child_argv, "--target_scenario_id", scenario_id)
            run_checked([sys.executable, "-m", "benchmarks.run_baselines", *child_argv])
            sweep_rows.append({
                "scenario_id": scenario_id,
                "params": cell["params"],
                "output_dir": str(cell_dir.resolve()),
            })
        (args.output_dir / "specialist_sweep.json").write_text(
            json.dumps({
                "source_config": str(source_config),
                "effective_config": str(args.config),
                "utility_weights": weights,
                "num_specialist_targets": len(cells),
                "specialists": sweep_rows,
            }, indent=2),
            encoding="utf-8",
        )
        if args.run_benchmark and not args.dry_run:
            write_specialist_sweep_summary(args.output_dir, sweep_rows)
        print(f"[SPECIALIST SWEEP] completed {len(cells)} targets under {sweep_root}")
        return
    if sum((args.target_scenario_id is not None, args.target_family is not None, args.generalist)) > 1:
        raise ValueError("Choose only one of --target_scenario_id, --target_family, or --generalist")
    if int(config.get("schema_version", 1)) >= 2 and (
        args.target_family is not None or args.retrieval_family is not None
    ):
        raise ValueError(
            "Family specialists are not part of the V2 approach/radius protocol. "
            "Use --target_scenario_id for a specialist or --generalist for the complete grid."
        )
    selected_target_cells = select_target_cells(
        config, args.target_scenario_id, args.target_family, args.generalist
    )
    budget = args.candidate_budget or int(config["evaluation"]["candidate_budget"])
    seeds = (
        [int(value) for value in args.seeds.split(",") if value.strip()]
        if args.seeds
        else [int(value) for value in config["evaluation"]["method_seeds"]]
    )
    methods = {value.strip() for value in args.methods.split(",") if value.strip()}
    allowed_methods = {
        "reference", "random", "random_search", "retrieval", "adam", "cma_es",
        "conditional_diffusion", "dgdm", "unconditional_diffusion", "unconditional_dgdm",
    }
    unknown_methods = methods - allowed_methods
    if unknown_methods:
        raise ValueError(f"Unknown --methods values: {sorted(unknown_methods)}")
    candidate_dir = args.output_dir / "candidates"
    candidate_dir.mkdir(exist_ok=True)
    candidate_files = []
    proposal_times = []

    def record_proposal_time(method, seed, started, path):
        elapsed = time.perf_counter() - started
        proposal_times.append({
            "method": method,
            "seed": int(seed),
            "proposal_elapsed_seconds": elapsed,
            "candidate_file": str(path.resolve()),
        })
        return elapsed

    if "reference" in methods:
        started = time.perf_counter()
        path = candidate_dir / "reference_s0.npz"
        save_candidates(path, reference_design(), "reference", seed=0, candidate_ids=["manufactured_runsh"])
        candidate_files.append(path)
        record_proposal_time("reference", 0, started, path)

    if "random" in methods:
        for seed in seeds:
            started = time.perf_counter()
            path = candidate_dir / f"random_s{seed}.npz"
            save_candidates(path, sample_feasible_designs(budget, seed), "random", seed=seed)
            candidate_files.append(path)
            record_proposal_time("random", seed, started, path)

    if "retrieval" in methods:
        started = time.perf_counter()
        if args.retrieval_data_dir is None:
            raise ValueError("--retrieval_data_dir is required when retrieval is enabled")
        selected = retrieve(
            args.retrieval_data_dir, config, budget,
            scenario_id=args.retrieval_scenario_id or args.target_scenario_id,
            family=args.retrieval_family or args.target_family,
            generalist=args.generalist,
        )
        path = candidate_dir / "retrieval_s0.npz"
        save_candidates(
            path,
            [row[0] for row in selected],
            "retrieval",
            seed=0,
            scores=[-row[1] for row in selected],
            metadata={
                "sources": [row[2] for row in selected],
                "observed_utilities": [row[3] for row in selected],
                "target_scenario_id": args.retrieval_scenario_id or args.target_scenario_id,
                "target_family": args.retrieval_family or args.target_family,
                "generalist": args.generalist,
            },
        )
        candidate_files.append(path)
        record_proposal_time("retrieval", 0, started, path)

    search_methods = methods.intersection({"random_search", "adam", "cma_es"})
    if search_methods:
        if args.dynamics_checkpoint is None:
            raise ValueError("--dynamics_checkpoint is required for random_search, Adam, or CMA-ES")
        surrogate = load_surrogate(args.dynamics_checkpoint, device=args.device)
        target_count = len(select_target_cells(
            config, args.target_scenario_id, args.target_family, args.generalist
        ))
        random_pool_size = args.random_pool_size
        adam_steps = args.adam_steps
        cma_generations = args.cma_generations
        if args.surrogate_eval_budget is not None:
            if args.surrogate_eval_budget < budget * target_count:
                raise ValueError(
                    "--surrogate_eval_budget must cover at least candidate_budget x target scenarios"
                )
            random_pool_size = max(budget, args.surrogate_eval_budget // target_count)
            adam_steps = max(1, args.surrogate_eval_budget // (budget * target_count))
            effective_popsize = max(args.cma_popsize, budget)
            cma_generations = max(1, args.surrogate_eval_budget // (effective_popsize * target_count))
        for seed in seeds:
            if "random_search" in search_methods:
                started = time.perf_counter()
                pool_size = max(random_pool_size, budget)
                pool = sample_feasible_designs(pool_size, seed)
                result = rank_designs(
                    surrogate, pool, config,
                    scenario_id=args.target_scenario_id, family=args.target_family,
                    generalist=args.generalist, device=args.device,
                )
                path = candidate_dir / f"random_search_s{seed}.npz"
                save_candidates(
                    path, result.designs[:budget], "random_search", seed=seed,
                    scores=result.scores[:budget], metadata={
                        "dynamics_checkpoint": str(args.dynamics_checkpoint.resolve()),
                        "proposal_pool_size": pool_size,
                        "model_evaluations": result.model_evaluations,
                        "target_scenario_ids": result.target_scenario_ids,
                        "selection_rule": "surrogate_mean_utility",
                        "proposal_elapsed_seconds": time.perf_counter() - started,
                    },
                )
                candidate_files.append(path)
                record_proposal_time("random_search", seed, started, path)
            if "adam" in search_methods:
                started = time.perf_counter()
                result = adam_search(
                    surrogate, config, budget, seed,
                    num_steps=adam_steps, learning_rate=args.adam_lr,
                    scenario_id=args.target_scenario_id, family=args.target_family,
                    generalist=args.generalist, device=args.device,
                )
                path = candidate_dir / f"adam_s{seed}.npz"
                save_candidates(
                    path, result.designs, "adam", seed=seed, scores=result.scores,
                    metadata={
                        "dynamics_checkpoint": str(args.dynamics_checkpoint.resolve()),
                        "model_evaluations": result.model_evaluations,
                        "target_scenario_ids": result.target_scenario_ids,
                        "selection_rule": "surrogate_mean_utility",
                        "proposal_elapsed_seconds": time.perf_counter() - started,
                    },
                )
                candidate_files.append(path)
                record_proposal_time("adam", seed, started, path)
            if "cma_es" in search_methods:
                started = time.perf_counter()
                result = cma_es_search(
                    surrogate, config, budget, seed,
                    num_generations=cma_generations, popsize=args.cma_popsize,
                    sigma=args.cma_sigma, scenario_id=args.target_scenario_id,
                    family=args.target_family, generalist=args.generalist, device=args.device,
                )
                path = candidate_dir / f"cma_es_s{seed}.npz"
                save_candidates(
                    path, result.designs, "cma_es", seed=seed, scores=result.scores,
                    metadata={
                        "dynamics_checkpoint": str(args.dynamics_checkpoint.resolve()),
                        "model_evaluations": result.model_evaluations,
                        "target_scenario_ids": result.target_scenario_ids,
                        "selection_rule": "surrogate_mean_utility",
                        "proposal_elapsed_seconds": time.perf_counter() - started,
                    },
                )
                candidate_files.append(path)
                record_proposal_time("cma_es", seed, started, path)

    diffusion_methods = methods.intersection({
        "conditional_diffusion", "dgdm", "unconditional_diffusion", "unconditional_dgdm",
    })
    if diffusion_methods:
        conditional_methods = diffusion_methods.intersection({"conditional_diffusion", "dgdm"})
        unconditional_methods = diffusion_methods.intersection({
            "unconditional_diffusion", "unconditional_dgdm",
        })
        if conditional_methods and args.diffusion_checkpoint is None:
            raise ValueError("--diffusion_checkpoint is required for conditional diffusion methods")
        if unconditional_methods and args.unconditional_diffusion_checkpoint is None:
            raise ValueError(
                "--unconditional_diffusion_checkpoint is required for unconditional diffusion methods"
            )
        if args.dynamics_checkpoint is None:
            raise ValueError("--dynamics_checkpoint is required to rank diffusion candidates")
        guided_methods = diffusion_methods.intersection({"dgdm", "unconditional_dgdm"})
        if guided_methods and args.dgdm_dynamics_checkpoint is None:
            raise ValueError(
                "--dgdm_dynamics_checkpoint is required for DGDM methods. Keep "
                "--dynamics_checkpoint as the clean model used for final ranking."
            )
        from benchmarks.baselines.diffusion_search import diffusion_search, load_diffusion

        if not search_methods:
            surrogate = load_surrogate(args.dynamics_checkpoint, device=args.device)
        guidance_surrogate = (
            load_surrogate(
                args.dgdm_dynamics_checkpoint, device=args.device,
                expected_noise_conditioned=True,
            )
            if guided_methods else None
        )
        diffusion_models = {}
        if conditional_methods:
            diffusion_models["conditional"] = load_diffusion(
                args.diffusion_checkpoint, device=args.device,
                num_inference_steps=args.diffusion_inference_steps,
                expected_conditioning="conditional",
            )
        if unconditional_methods:
            diffusion_models["unconditional"] = load_diffusion(
                args.unconditional_diffusion_checkpoint, device=args.device,
                num_inference_steps=args.diffusion_inference_steps,
                expected_conditioning="unconditional",
            )
        for seed in seeds:
            for method in sorted(diffusion_methods):
                started = time.perf_counter()
                output_method = args.dgdm_method_label if method == "dgdm" else method
                conditioning_mode = (
                    "unconditional" if method.startswith("unconditional_") else "conditional"
                )
                guided = method in {"dgdm", "unconditional_dgdm"}
                guidance_scale = args.dgdm_guidance_scale if guided else 0.0
                guidance_timesteps = (
                    tuple(int(value.strip()) for value in args.dgdm_guidance_timesteps.split(",")
                          if value.strip())
                    if guided and args.dgdm_guidance_timesteps else None
                )
                checkpoint_path = (
                    args.unconditional_diffusion_checkpoint
                    if conditioning_mode == "unconditional"
                    else args.diffusion_checkpoint
                )
                result = diffusion_search(
                    diffusion_models[conditioning_mode], surrogate, config, budget,
                    num_samples=args.diffusion_num_samples, seed=seed,
                    batch_size=args.diffusion_batch_size, guidance_scale=guidance_scale,
                    num_inference_steps=args.diffusion_inference_steps,
                    target_contacts=args.target_contacts,
                    target_disturbance=args.target_disturbance,
                    target_angular_span=args.target_angular_span,
                    scenario_id=args.target_scenario_id, family=args.target_family,
                    generalist=args.generalist, device=args.device,
                    guidance_dynamics_model=guidance_surrogate if guided else None,
                    guidance_timesteps=guidance_timesteps,
                )
                path = candidate_dir / f"{output_method}_s{seed}.npz"
                save_candidates(
                    path, result.designs, output_method, seed=seed, scores=result.scores,
                    metadata={
                        "base_method": method,
                        "diffusion_checkpoint": str(checkpoint_path.resolve()),
                        "conditioning_mode": conditioning_mode,
                        "dynamics_checkpoint": str(args.dynamics_checkpoint.resolve()),
                        "dgdm_dynamics_checkpoint": (
                            str(args.dgdm_dynamics_checkpoint.resolve()) if guided else None
                        ),
                        "guidance_scale": guidance_scale,
                        "guidance_timesteps": guidance_timesteps,
                        "num_samples": args.diffusion_num_samples,
                        "num_inference_steps": args.diffusion_inference_steps,
                        "model_evaluations": result.model_evaluations,
                        "target_scenario_ids": result.target_scenario_ids,
                        "selection_rule": "surrogate_benchmark_utility",
                        "proposal_conditioning": (
                            "none" if conditioning_mode == "unconditional" else
                            ("scenario_set_centroid" if len(result.target_scenario_ids) > 1
                             else "exact_scenario")
                        ),
                        "guidance_aggregation": (
                            "mean_over_target_scenarios" if guidance_scale > 0
                            else "none"
                        ),
                        "proposal_elapsed_seconds": time.perf_counter() - started,
                    },
                )
                candidate_files.append(path)
                record_proposal_time(output_method, seed, started, path)

    for spec in args.adapt:
        started = time.perf_counter()
        method, source = parse_adapter(spec)
        adapted = load_candidates(source, method=method, top_k=budget)
        path = candidate_dir / f"{method}_s{adapted['seed']}.npz"
        save_candidates(
            path, adapted["design_params"], method, seed=adapted["seed"],
            candidate_ids=adapted["candidate_ids"], scores=adapted["selection_scores"],
            metadata={"adapted_from": str(source.resolve())},
        )
        candidate_files.append(path)
        record_proposal_time(method, adapted["seed"], started, path)

    if not candidate_files:
        raise ValueError("No candidate files were generated")
    print("[CANDIDATES]")
    for path in candidate_files:
        print(path.resolve())
    if proposal_times:
        with open(args.output_dir / "proposal_times.csv", "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(proposal_times[0]))
            writer.writeheader()
            writer.writerows(proposal_times)
    if not args.run_benchmark:
        return

    benchmark_jobs = []
    for path in candidate_files:
        loaded = load_candidates(path)
        run_dir = args.output_dir / "runs" / f"{loaded['method']}_s{loaded['seed']}"
        command = [
            sys.executable, "-m", "benchmarks.run_sim_benchmark",
            "--candidates", str(path),
            "--output_dir", str(run_dir),
            "--config", str(args.config),
            "--top_k", str(
                args.benchmark_top_k
                if args.benchmark_top_k is not None
                else len(loaded["design_params"])
            ),
            # Parallelism is managed here across candidate/method/seed files.
            # A specialist child contains only one rollout, so assigning the
            # complete worker pool inside that child leaves the other CPUs idle.
            "--num_workers", "1",
            "--timeout", str(args.timeout),
        ]
        if args.families:
            command.extend(["--families", args.families])
        if args.evaluation_scope == "target" or (
            args.evaluation_scope == "auto" and not args.generalist
        ):
            command.extend([
                "--scenario_ids",
                ",".join(cell["scenario_id"] for cell in selected_target_cells),
            ])
        if args.render:
            command.append("--render")
        if args.dry_run:
            command.append("--dry_run")
        benchmark_jobs.append((command, run_dir / "records.jsonl"))

    worker_count = max(1, int(args.num_workers))
    print(
        f"[BENCHMARK PARALLELISM] {len(benchmark_jobs)} candidate groups, "
        f"up to {min(worker_count, len(benchmark_jobs))} concurrent subprocesses"
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_record = {
            executor.submit(run_checked, command): record_path
            for command, record_path in benchmark_jobs
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(future_to_record), start=1
        ):
            future.result()
            print(f"[BENCHMARK GROUP {completed}/{len(benchmark_jobs)}] complete")

    record_files = (
        [] if args.dry_run else [record_path for _, record_path in benchmark_jobs]
    )

    if record_files:
        run_checked(
            [
                sys.executable, "-m", "benchmarks.summarize",
                *[str(path) for path in record_files],
                "--output_dir", str(args.output_dir / "summary"),
                "--config", str(args.config),
            ]
        )


if __name__ == "__main__":
    main()
