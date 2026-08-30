"""Resumable evaluation of fixed From Links designs across scenario families."""

import argparse
import concurrent.futures
import csv
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from benchmarks.candidates import load_candidates
from benchmarks.protocol import DEFAULT_CONFIG, expand_core_scenarios, load_config, normalized_metrics, utility
from dynamics.utils import design_to_dict


REPO_ROOT = Path(__file__).resolve().parents[1]
FINGER_DIR = REPO_ROOT / "TendonForces"
FINGER_SCRIPT = FINGER_DIR / "finger.py"
RESERVED_SCENARIO_KEYS = {
    "base_len", "base_rad", "tension", "ankle_wrap_radius", "ankle_stiffness",
    "joint_softness", "joint_lengths", "link_lengths", "output_dir", "suffix",
}


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def read_latest_npz(run_dir):
    files = glob.glob(str(Path(run_dir) / "**" / "master_log_*.npz"), recursive=True)
    if not files:
        raise FileNotFoundError(f"No master_log NPZ found in {run_dir}")
    return max(files, key=os.path.getmtime)


def read_metric_from_npz(path):
    def scalar(data, key, default):
        value = np.asarray(data.get(key, [default])).reshape(-1)
        return float(default) if value.size == 0 else float(value[0])

    with np.load(path, allow_pickle=True) as data:
        metric = {
            "num_contacts": scalar(data, "num_contacts", 0.0),
            "disturbance_resistance_score": scalar(data, "disturbance_resistance_score", 0.0),
            "angular_span": scalar(data, "angular_span", 0.0),
            "n_elements": scalar(data, "n_elements", 100.0),
        }
        for key in ("max_overlap_overall", "total_energy"):
            if key in data:
                metric[key] = scalar(data, key, 0.0)
    return metric


def append_arg(command, key, value):
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            command.append(f"--{key}")
        return
    command.extend([f"--{key}", str(value)])


def stable_run_id(method, seed, candidate_id, scenario_id, design, scenario_params):
    readable = f"{method}_s{seed}_{candidate_id}_{scenario_id}".replace(":", "-").replace("/", "-")
    identity = {
        "method": method,
        "seed": int(seed),
        "candidate_id": str(candidate_id),
        "design": [float(value) for value in design],
        "scenario_id": scenario_id,
        "scenario_params": scenario_params,
    }
    digest = hashlib.sha1(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    return f"{readable[:100]}_{digest}"


def build_command(design, scenario, run_dir, run_id, python_executable):
    overlap = RESERVED_SCENARIO_KEYS.intersection(scenario["params"])
    if overlap:
        raise ValueError(f"Scenario may not override design fields: {sorted(overlap)}")
    decoded = design_to_dict(design)
    params = scenario["params"]
    command = [
        python_executable, str(FINGER_SCRIPT),
        "--approach_deg", str(params["approach_deg"]),
        "--cyl_rad", str(params["cyl_rad"]),
        "--v_mode", "from_links",
        "--link_lengths", decoded["link_lengths_str"],
        "--joint_lengths", decoded["joint_lengths_str"],
        "--joint_softness", decoded["joint_softness_str"],
        "--base_rad", str(decoded["base_rad"]),
        "--base_len", str(decoded["base_len"]),
        "--tension", str(decoded["tension"]),
        "--ankle_wrap_radius", str(decoded["ankle_wrap_radius"]),
        "--ankle_stiffness", str(decoded["ankle_stiffness"]),
        "--output_dir", str(run_dir),
        "--suffix", run_id,
    ]
    for key, value in params.items():
        if key in {"approach_deg", "cyl_rad", "v_mode"}:
            continue
        append_arg(command, key, value)
    return command, decoded


def execute_job(job, weights, timeout, python_executable, render):
    run_dir = Path(job["run_dir"])
    result_path = run_dir / "benchmark_result.json"
    if result_path.exists():
        with open(result_path, "r", encoding="utf-8") as stream:
            cached = json.load(stream)
        if cached.get("status") == "ok":
            cached["cached"] = True
            return cached
    run_dir.mkdir(parents=True, exist_ok=True)
    command, decoded = build_command(
        np.asarray(job["design_params"], dtype=np.float32), job["scenario"], run_dir,
        job["run_id"], python_executable,
    )
    if not render:
        command.append("--disable_video_plots")
    with open(run_dir / "benchmark_job.json", "w", encoding="utf-8") as stream:
        json.dump({**job, "decoded_design": decoded, "command": command}, stream, indent=2)

    started = time.time()
    record = {
        key: job[key]
        for key in ("run_id", "method", "seed", "candidate_id", "scenario_id", "family")
    }
    record["selection_score"] = job.get("selection_score")
    record["scenario_params"] = job["scenario"]["params"]
    record["status"] = "error"
    try:
        completed = subprocess.run(
            command, cwd=FINGER_DIR, capture_output=True, text=True, timeout=timeout,
        )
        (run_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (run_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"finger.py exited {completed.returncode}: {completed.stderr[-1500:]}")
        master_log = read_latest_npz(run_dir)
        metrics = read_metric_from_npz(master_log)
        record.update(
            status="ok",
            metrics=metrics,
            normalized_metrics=normalized_metrics(metrics),
            utility=utility(metrics, weights),
            master_log_path=str(Path(master_log).resolve()),
        )
    except subprocess.TimeoutExpired as exc:
        record.update(status="timeout", error=f"Timed out after {timeout}s: {exc}")
    except Exception as exc:  # persist failures so large studies can finish
        record.update(status="error", error=str(exc))
    record["elapsed_seconds"] = time.time() - started
    record = json_safe(record)
    temp_path = result_path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp_path, result_path)
    return record


def write_rollout_index(records, output_dir):
    fields = [
        "run_id", "method", "seed", "candidate_id", "scenario_id", "family", "status",
        "utility", "elapsed_seconds", "error", "master_log_path",
    ]
    with open(output_dir / "rollouts.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fields})
    with open(output_dir / "records.jsonl", "w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate candidate designs over Squirrel Benchmark V1.")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--families", type=str, default="", help="Comma-separated family subset")
    parser.add_argument("--scenario_ids", type=str, default="", help="Comma-separated exact scenario IDs")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_rollouts", type=int, default=None, help="Debug-only truncation")
    args = parser.parse_args()
    # Keep a virtualenv's python symlink intact; resolving the symlink would
    # launch the base interpreter and lose the virtualenv site-packages.
    python_executable = os.path.abspath(os.path.expanduser(args.python))

    config = load_config(args.config)
    candidates = load_candidates(args.candidates, method=args.method, seed=args.seed, top_k=args.top_k)
    scenarios = expand_core_scenarios(config)
    requested_families = {x.strip() for x in args.families.split(",") if x.strip()}
    requested_ids = {x.strip() for x in args.scenario_ids.split(",") if x.strip()}
    if requested_families:
        scenarios = [cell for cell in scenarios if cell["family"] in requested_families]
    if requested_ids:
        scenarios = [cell for cell in scenarios if cell["scenario_id"] in requested_ids]
    if not scenarios:
        raise ValueError("Scenario filters selected zero cells")

    output_dir = args.output_dir.resolve()
    runs_dir = output_dir / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    selection_scores = candidates["selection_scores"]
    if selection_scores is None:
        selection_scores = [None] * len(candidates["design_params"])
    for candidate_id, design, selection_score in zip(
        candidates["candidate_ids"], candidates["design_params"], selection_scores
    ):
        for scenario in scenarios:
            run_id = stable_run_id(
                candidates["method"], candidates["seed"], candidate_id,
                scenario["scenario_id"], design, scenario["params"],
            )
            jobs.append(
                {
                    "run_id": run_id,
                    "method": candidates["method"],
                    "seed": candidates["seed"],
                    "candidate_id": str(candidate_id),
                    "selection_score": None if selection_score is None else float(selection_score),
                    "design_params": design.tolist(),
                    "scenario_id": scenario["scenario_id"],
                    "family": scenario["family"],
                    "scenario": scenario,
                    "run_dir": str(runs_dir / run_id),
                }
            )
    if args.max_rollouts is not None:
        jobs = jobs[: args.max_rollouts]
    manifest = {
        "config": str(args.config.resolve()),
        "candidate_source": candidates["source_path"],
        "method": candidates["method"],
        "seed": candidates["seed"],
        "num_candidates": len(candidates["design_params"]),
        "num_scenarios": len(scenarios),
        "num_rollouts": len(jobs),
        "proposal_metadata": candidates["metadata"],
        "jobs": jobs,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[BENCHMARK] {len(jobs)} rollouts = {len(candidates['design_params'])} designs x {len(scenarios)} scenarios")
    if args.dry_run:
        print(f"[DRY RUN] Manifest: {output_dir / 'manifest.json'}")
        return

    weights = config["evaluation"]["utility_weights"]
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as executor:
        futures = [
            executor.submit(execute_job, job, weights, args.timeout, python_executable, args.render)
            for job in jobs
        ]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            print(f"[{completed}/{len(futures)}] {record['run_id']} -> {record['status']}")
            write_rollout_index(sorted(records, key=lambda item: item["run_id"]), output_dir)
    print(f"[DONE] {output_dir / 'records.jsonl'}")


if __name__ == "__main__":
    main()
