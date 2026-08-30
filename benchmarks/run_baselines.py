"""Generate common baseline candidates and optionally run the simulation suite."""

import argparse
import subprocess
import sys
from pathlib import Path

from benchmarks.baselines.random_search import sample_feasible_designs
from benchmarks.baselines.reference import reference_design
from benchmarks.baselines.retrieval import retrieve
from benchmarks.baselines.surrogate_search import (
    adam_search, cma_es_search, load_surrogate, rank_designs, select_target_cells,
)
from benchmarks.candidates import load_candidates, save_candidates
from benchmarks.protocol import DEFAULT_CONFIG, load_config


def parse_adapter(spec):
    if "=" not in spec:
        raise ValueError("--adapt must be METHOD=PATH")
    method, path = spec.split("=", 1)
    return method.strip(), Path(path).expanduser()


def run_checked(command):
    print("[RUN]", " ".join(str(part) for part in command))
    subprocess.run(command, check=True)


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
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--target_scenario_id", type=str, default=None)
    parser.add_argument("--target_family", type=str, default=None)
    parser.add_argument(
        "--generalist", action="store_true",
        help="Optimize mean surrogate utility over all core scenarios.",
    )
    parser.add_argument("--adam_steps", type=int, default=300)
    parser.add_argument("--adam_lr", type=float, default=0.03)
    parser.add_argument("--cma_generations", type=int, default=100)
    parser.add_argument("--cma_popsize", type=int, default=32)
    parser.add_argument("--cma_sigma", type=float, default=0.5)
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
    parser.add_argument("--benchmark_top_k", type=int, default=1)
    parser.add_argument("--families", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    budget = args.candidate_budget or int(config["evaluation"]["candidate_budget"])
    seeds = (
        [int(value) for value in args.seeds.split(",") if value.strip()]
        if args.seeds
        else [int(value) for value in config["evaluation"]["method_seeds"]]
    )
    methods = {value.strip() for value in args.methods.split(",") if value.strip()}
    allowed_methods = {"reference", "random", "random_search", "retrieval", "adam", "cma_es"}
    unknown_methods = methods - allowed_methods
    if unknown_methods:
        raise ValueError(f"Unknown --methods values: {sorted(unknown_methods)}")
    if sum((args.target_scenario_id is not None, args.target_family is not None, args.generalist)) > 1:
        raise ValueError("Choose only one of --target_scenario_id, --target_family, or --generalist")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir = args.output_dir / "candidates"
    candidate_dir.mkdir(exist_ok=True)
    candidate_files = []

    if "reference" in methods:
        path = candidate_dir / "reference_s0.npz"
        save_candidates(path, reference_design(), "reference", seed=0, candidate_ids=["manufactured_runsh"])
        candidate_files.append(path)

    if "random" in methods:
        for seed in seeds:
            path = candidate_dir / f"random_s{seed}.npz"
            save_candidates(path, sample_feasible_designs(budget, seed), "random", seed=seed)
            candidate_files.append(path)

    if "retrieval" in methods:
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
                    },
                )
                candidate_files.append(path)
            if "adam" in search_methods:
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
                    },
                )
                candidate_files.append(path)
            if "cma_es" in search_methods:
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
                    },
                )
                candidate_files.append(path)

    for spec in args.adapt:
        method, source = parse_adapter(spec)
        adapted = load_candidates(source, method=method, top_k=budget)
        path = candidate_dir / f"{method}_s{adapted['seed']}.npz"
        save_candidates(
            path, adapted["design_params"], method, seed=adapted["seed"],
            candidate_ids=adapted["candidate_ids"], scores=adapted["selection_scores"],
            metadata={"adapted_from": str(source.resolve())},
        )
        candidate_files.append(path)

    if not candidate_files:
        raise ValueError("No candidate files were generated")
    print("[CANDIDATES]")
    for path in candidate_files:
        print(path.resolve())
    if not args.run_benchmark:
        return

    record_files = []
    for path in candidate_files:
        loaded = load_candidates(path)
        run_dir = args.output_dir / "runs" / f"{loaded['method']}_s{loaded['seed']}"
        command = [
            sys.executable, "-m", "benchmarks.run_sim_benchmark",
            "--candidates", str(path),
            "--output_dir", str(run_dir),
            "--config", str(args.config),
            "--top_k", str(args.benchmark_top_k),
            "--num_workers", str(args.num_workers),
            "--timeout", str(args.timeout),
        ]
        if args.families:
            command.extend(["--families", args.families])
        if args.render:
            command.append("--render")
        if args.dry_run:
            command.append("--dry_run")
        run_checked(command)
        if not args.dry_run:
            record_files.append(run_dir / "records.jsonl")

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
