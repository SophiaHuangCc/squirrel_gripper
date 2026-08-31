"""Render the best simulator-evaluated seed/candidate from each benchmark method."""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from benchmarks.candidates import load_candidates, save_candidates
from benchmarks.protocol import DEFAULT_CONFIG


def best_method_rows(summary_path):
    with open(summary_path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No method rows found in {summary_path}")
    best = {}
    for row in rows:
        method = row["method"]
        score = float(row["best_mean_utility"])
        if method not in best or score > float(best[method]["best_mean_utility"]):
            best[method] = row
    return best


def main():
    parser = argparse.ArgumentParser(
        description="Render the best full-simulator seed from each benchmark method."
    )
    parser.add_argument("--benchmark_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--methods", type=str, default="",
        help="Optional comma-separated subset; default renders every summarized method.",
    )
    parser.add_argument(
        "--scenario_ids", type=str, default="approach_radius:12",
        help="Comma-separated scenes to render; use an empty string to render all 25.",
    )
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    benchmark_dir = args.benchmark_dir.resolve()
    output_dir = (args.output_dir or benchmark_dir / "best_seed_visualizations").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = best_method_rows(benchmark_dir / "summary" / "method_summary.csv")
    requested = {value.strip() for value in args.methods.split(",") if value.strip()}
    if requested:
        missing = requested - set(rows)
        if missing:
            raise ValueError(f"Methods absent from method_summary.csv: {sorted(missing)}")
        rows = {method: row for method, row in rows.items() if method in requested}

    for method, row in sorted(rows.items()):
        seed = int(row["seed"])
        candidate_id = row["best_candidate_id"]
        source = benchmark_dir / "candidates" / f"{method}_s{seed}.npz"
        loaded = load_candidates(source)
        matches = [
            index for index, value in enumerate(loaded["candidate_ids"])
            if str(value) == candidate_id
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected candidate {candidate_id!r} exactly once in {source}; found {len(matches)}"
            )
        index = matches[0]
        candidate_path = output_dir / "candidates" / f"{method}_best_s{seed}.npz"
        score = None
        if loaded["selection_scores"] is not None:
            score = [loaded["selection_scores"][index]]
        save_candidates(
            candidate_path,
            loaded["design_params"][index],
            method=method,
            seed=seed,
            candidate_ids=[candidate_id],
            scores=score,
            metadata={
                "selected_from": str(source),
                "selection_basis": "highest best_mean_utility in method_summary.csv",
                "simulator_mean_utility": float(row["best_mean_utility"]),
            },
        )
        command = [
            sys.executable, "-m", "benchmarks.run_sim_benchmark",
            "--candidates", str(candidate_path),
            "--output_dir", str(output_dir / "runs" / f"{method}_best_s{seed}"),
            "--config", str(args.config),
            "--top_k", "1",
            "--num_workers", str(args.num_workers),
            "--timeout", str(args.timeout),
            "--render",
        ]
        if args.scenario_ids:
            command.extend(["--scenario_ids", args.scenario_ids])
        if args.dry_run:
            command.append("--dry_run")
        print(
            f"[BEST] method={method} seed={seed} candidate={candidate_id} "
            f"mean_utility={float(row['best_mean_utility']):.6f}"
        )
        print("[RUN]", " ".join(command))
        subprocess.run(command, check=True)

    print(f"[SAVED] {output_dir}")


if __name__ == "__main__":
    main()
