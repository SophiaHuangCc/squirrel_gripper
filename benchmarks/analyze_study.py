"""Aggregate specialist or generalist benchmark studies without browsing trees.

The input may be the root containing combined/contact_only/disturbance_only,
one objective directory, or one specialist scenario directory.  Results are
discovered from benchmark_result.json files so interrupted/resumed studies are
handled naturally.
"""

import argparse
import concurrent.futures
import csv
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmarks.candidates import save_candidates


OBJECTIVE_NAMES = {"combined", "contact_only", "disturbance_only"}


def objective_name(path, root):
    for parent in (path.parent, *path.parents):
        if parent.name in OBJECTIVE_NAMES:
            return parent.name
        if parent == root.parent:
            break
    return root.name if root.name in OBJECTIVE_NAMES else "unknown"


def nearest_config(path, stop):
    for parent in (path.parent, *path.parents):
        candidate = parent / "effective_config.json"
        if candidate.exists():
            return candidate
        if parent == stop.parent:
            break
    return None


def read_rows(root):
    rows = []
    # records.jsonl is rewritten by each current benchmark group and therefore
    # excludes obsolete run directories left by a rerun with new candidates.
    # Fall back to recursive result discovery only for legacy outputs.
    indexed_paths = set()
    for index_path in sorted(root.rglob("records.jsonl")):
        if "study_analysis" in index_path.parts:
            continue
        with index_path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                run_id = record.get("run_id")
                if run_id:
                    candidate = index_path.parent / "runs" / run_id / "benchmark_result.json"
                    if candidate.exists():
                        indexed_paths.add(candidate.resolve())
    result_paths = sorted(indexed_paths) if indexed_paths else [
        path for path in sorted(root.rglob("benchmark_result.json"))
        if "study_analysis" not in path.parts
    ]
    for result_path in result_paths:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "ok":
            continue
        job_path = result_path.with_name("benchmark_job.json")
        job = json.loads(job_path.read_text(encoding="utf-8")) if job_path.exists() else {}
        metrics = result.get("metrics", {})
        normalized = result.get("normalized_metrics", {})
        selection = result.get("selection_score")
        actual = result.get("utility")
        contact = normalized.get("contact_coverage_norm")
        disturbance = normalized.get("disturbance_resistance_score")
        angular = normalized.get("angular_span_norm")
        positive_work = metrics.get("tendon_actuator_work_positive_j")
        combined = (
            None if None in (contact, disturbance, angular)
            else 0.20 * float(contact) + 0.45 * float(disturbance) + 0.35 * float(angular)
        )
        row = {
            "objective": objective_name(result_path, root),
            "scenario_id": result.get("scenario_id", ""),
            "method": result.get("method", ""),
            "seed": int(result.get("seed", 0)),
            "candidate_id": result.get("candidate_id", ""),
            "selection_score": selection,
            "simulator_utility": actual,
            # Counterfactual rescoring makes designs generated under different
            # objectives comparable without confusing their native scores.
            "combined_utility": combined,
            "contact_only_utility": contact,
            "disturbance_only_utility": disturbance,
            "selection_minus_simulator": (
                None if selection is None or actual is None else float(selection) - float(actual)
            ),
            "selection_out_of_range": (
                selection is not None and not 0.0 <= float(selection) <= 1.0
            ),
            "num_contacts": metrics.get("num_contacts"),
            "contact_coverage_norm": contact,
            "disturbance_resistance": metrics.get("disturbance_resistance_score"),
            "angular_span_deg": metrics.get("angular_span"),
            "angular_span_norm": angular,
            "total_energy_j": metrics.get("total_energy"),
            "tendon_displacement_m": metrics.get("tendon_displacement_m"),
            "tendon_work_positive_j": positive_work,
            "tendon_work_net_j": metrics.get("tendon_actuator_work_net_j"),
            "combined_utility_per_joule": (
                None if combined is None or positive_work is None or float(positive_work) <= 1e-12
                else float(combined) / float(positive_work)
            ),
            "simulation_seconds": result.get("elapsed_seconds"),
            "result_path": str(result_path.resolve()),
            "master_log_path": result.get("master_log_path", ""),
            "config_path": str(nearest_config(result_path, root) or ""),
            "design_params": job.get("design_params"),
        }
        rows.append(row)
    return rows


def mean(values):
    values = [float(value) for value in values if value is not None]
    return float(np.mean(values)) if values else math.nan


def std(values):
    values = [float(value) for value in values if value is not None]
    return float(np.std(values)) if values else math.nan


def grouped_summary(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for key, group in sorted(groups.items()):
        output.append({
            **dict(zip(keys, key)),
            "num_rollouts": len(group),
            "num_seeds": len({row["seed"] for row in group}),
            "mean_utility": mean(row["simulator_utility"] for row in group),
            "std_utility": std(row["simulator_utility"] for row in group),
            "mean_combined_utility": mean(row["combined_utility"] for row in group),
            "mean_contact_only_utility": mean(row["contact_only_utility"] for row in group),
            "mean_disturbance_only_utility": mean(
                row["disturbance_only_utility"] for row in group
            ),
            "mean_contacts": mean(row["num_contacts"] for row in group),
            "mean_contact_coverage": mean(row["contact_coverage_norm"] for row in group),
            "std_contact_coverage": std(row["contact_coverage_norm"] for row in group),
            "mean_disturbance": mean(row["disturbance_resistance"] for row in group),
            "std_disturbance": std(row["disturbance_resistance"] for row in group),
            "mean_angular_span_deg": mean(row["angular_span_deg"] for row in group),
            "std_angular_span_deg": std(row["angular_span_deg"] for row in group),
            "mean_angular_span_norm": mean(row["angular_span_norm"] for row in group),
            "std_angular_span_norm": std(row["angular_span_norm"] for row in group),
            "mean_simulation_seconds": mean(row["simulation_seconds"] for row in group),
        })
    return output


def select_best(rows, keys, value_key="simulator_utility"):
    groups = defaultdict(list)
    for row in rows:
        if row.get(value_key) is not None:
            groups[tuple(row[key] for key in keys)].append(row)
    return [max(group, key=lambda row: float(row[value_key]))
            for _, group in sorted(groups.items())]


def design_key(row):
    design = row.get("design_params")
    return tuple(round(float(value), 7) for value in design) if design is not None else ()


def generalist_candidate_summary(rows):
    """One row per fixed generalist design, aggregated across scenario cells."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["objective"], row["method"], row["seed"],
                row["candidate_id"], design_key(row))].append(row)
    output = []
    for (objective, method, seed, candidate_id, _), group in sorted(groups.items()):
        scenario_ids = sorted({row["scenario_id"] for row in group})
        if len(scenario_ids) < 2:
            continue
        utilities = sorted(float(row["simulator_utility"]) for row in group
                           if row["simulator_utility"] is not None)
        tail_count = max(1, math.ceil(0.2 * len(utilities)))
        representative = dict(group[0])
        representative.update({
            "objective": objective,
            "method": method,
            "seed": seed,
            "candidate_id": candidate_id,
            "num_scenarios": len(scenario_ids),
            "scenario_ids": ",".join(scenario_ids),
            "mean_utility": mean(utilities),
            "std_across_scenarios": std(utilities),
            "cvar20_utility": mean(utilities[:tail_count]),
            "worst_scenario_utility": min(utilities),
            "best_scenario_utility": max(utilities),
            "mean_combined_utility": mean(row["combined_utility"] for row in group),
            "mean_contact_only_utility": mean(row["contact_only_utility"] for row in group),
            "mean_disturbance_only_utility": mean(
                row["disturbance_only_utility"] for row in group
            ),
            "mean_contacts": mean(row["num_contacts"] for row in group),
            "mean_angular_span_deg": mean(row["angular_span_deg"] for row in group),
        })
        output.append(representative)
    return output


def generalist_method_summary(candidate_rows):
    groups = defaultdict(list)
    for row in candidate_rows:
        groups[(row["objective"], row["method"])].append(row)
    output = []
    for (objective, method), group in sorted(groups.items()):
        values = [row["mean_utility"] for row in group]
        cvars = [row["cvar20_utility"] for row in group]
        output.append({
            "objective": objective,
            "method": method,
            "num_seeds": len({row["seed"] for row in group}),
            "num_generalist_designs": len(group),
            "mean_utility_across_seeds": mean(values),
            "std_utility_across_seeds": std(values),
            "mean_cvar20_across_seeds": mean(cvars),
            "mean_worst_scenario_across_seeds": mean(
                row["worst_scenario_utility"] for row in group
            ),
        })
    return output


def read_proposal_times(root):
    rows = []
    for path in sorted(root.rglob("proposal_times.csv")):
        # Analysis writes a normalized proposal_times.csv beneath
        # study_analysis.  Do not ingest that derived file on a later analysis
        # pass: its schema contains proposal_seconds rather than the raw
        # proposal_elapsed_seconds field.
        if "study_analysis" in path.parts:
            continue
        objective = objective_name(path, root)
        scenario = next(
            (part.replace("approach_radius-", "approach_radius:")
             for part in path.parts if part.startswith("approach_radius-")),
            "",
        )
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                rows.append({
                    "objective": objective,
                    "scenario_id": scenario,
                    "method": row["method"],
                    "seed": int(row["seed"]),
                    "proposal_seconds": float(row["proposal_elapsed_seconds"]),
                    "candidate_file": row.get("candidate_file", ""),
                })
    return rows


def timing_summary(proposal_rows, rollout_rows):
    proposal = grouped_summary_generic(
        proposal_rows, ("objective", "method"), "proposal_seconds"
    )
    simulation = grouped_summary_generic(
        rollout_rows, ("objective", "method"), "simulation_seconds"
    )
    merged = {}
    for row in proposal:
        merged[(row["objective"], row["method"])] = {
            "objective": row["objective"], "method": row["method"],
            "num_proposals": row["count"],
            "mean_proposal_seconds": row["mean"],
            "std_proposal_seconds": row["std"],
            "median_proposal_seconds": row["median"],
            "min_proposal_seconds": row["min"],
            "max_proposal_seconds": row["max"],
        }
    for row in simulation:
        target = merged.setdefault((row["objective"], row["method"]), {
            "objective": row["objective"], "method": row["method"],
        })
        target.update(num_simulations=row["count"],
                      mean_simulation_seconds=row["mean"],
                      std_simulation_seconds=row["std"],
                      median_simulation_seconds=row["median"])
    return list(merged.values())


def grouped_summary_generic(rows, keys, value_key):
    groups = defaultdict(list)
    for row in rows:
        if row.get(value_key) is not None:
            groups[tuple(row[key] for key in keys)].append(float(row[value_key]))
    return [{**dict(zip(keys, key)), "count": len(values),
             "mean": mean(values), "std": std(values),
             "median": float(np.median(values)), "min": min(values), "max": max(values)}
            for key, values in sorted(groups.items())]


def write_csv(path, rows, exclude=()):
    if not rows:
        return
    fields = []
    for row in rows:
        for field in row:
            if field not in exclude and field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def render_rows(rows, output_dir, num_workers, timeout, dry_run, measure_energy):
    commands = []
    for row in rows:
        if row["design_params"] is None or not row["config_path"]:
            print(f"[WARN] cannot render {row['result_path']}: design/config unavailable")
            continue
        tag = f"{row['objective']}_{row['scenario_id'].replace(':', '-')}_{row['method']}_s{row['seed']}"
        candidate = output_dir / "candidates" / f"{tag}.npz"
        save_candidates(
            candidate, row["design_params"], row["method"], seed=row["seed"],
            candidate_ids=[row["candidate_id"]], scores=[row["selection_score"]]
            if row["selection_score"] is not None else None,
            metadata={"selected_by": "highest full-simulator utility", "source": row["result_path"]},
        )
        command = [
            sys.executable, "-m", "benchmarks.run_sim_benchmark",
            "--candidates", str(candidate), "--output_dir", str(output_dir / "runs" / tag),
            "--config", row["config_path"], "--scenario_ids", row["scenario_id"],
            "--top_k", "1", "--num_workers", "1", "--timeout", str(timeout),
            "--render",
        ]
        if dry_run:
            command.append("--dry_run")
        if measure_energy:
            command.append("--measure_energy")
        print("[RENDER]", " ".join(command))
        commands.append(command)
    run_render_commands(commands, num_workers)


def run_render_commands(commands, num_workers):
    """Run independent render groups concurrently without nested worker pools."""
    total = len(commands)
    if not total:
        print("[RENDER PROGRESS] no render commands selected", flush=True)
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(num_workers))) as executor:
        futures = [executor.submit(subprocess.run, command, check=True) for command in commands]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            future.result()
            print(f"[RENDER PROGRESS] completed={completed}/{total}", flush=True)


def render_generalists(rows, output_dir, num_workers, timeout, dry_run, measure_energy):
    """Render each selected fixed generalist design over its complete scenario set."""
    commands = []
    for row in rows:
        if row["design_params"] is None or not row["config_path"]:
            print(f"[WARN] cannot render generalist {row['candidate_id']}: design/config unavailable")
            continue
        tag = f"{row['objective']}_{row['method']}_s{row['seed']}_generalist"
        candidate = output_dir / "candidates" / f"{tag}.npz"
        save_candidates(
            candidate, row["design_params"], row["method"], seed=row["seed"],
            candidate_ids=[row["candidate_id"]], scores=[row["selection_score"]]
            if row["selection_score"] is not None else None,
            metadata={
                "selected_by": "highest mean full-simulator utility across scenarios",
                "source": row["result_path"],
                "scenario_ids": row["scenario_ids"].split(","),
            },
        )
        command = [
            sys.executable, "-m", "benchmarks.run_sim_benchmark",
            "--candidates", str(candidate), "--output_dir", str(output_dir / "runs" / tag),
            "--config", row["config_path"], "--scenario_ids", row["scenario_ids"],
            "--top_k", "1", "--num_workers", "1", "--timeout", str(timeout),
            "--render",
        ]
        if dry_run:
            command.append("--dry_run")
        if measure_energy:
            command.append("--measure_energy")
        print("[RENDER GENERALIST]", " ".join(command))
        commands.append(command)
    run_render_commands(commands, num_workers)


def main():
    parser = argparse.ArgumentParser(description="Analyze a completed multi-objective benchmark study.")
    parser.add_argument("--study_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument(
        "--protocol", choices=("auto", "specialist", "generalist"), default="auto",
        help="Analysis protocol. Auto writes generalist tables when multi-scenario designs exist.",
    )
    parser.add_argument("--objectives", type=str, default="",
                        help="Optional comma-separated objective subset.")
    parser.add_argument("--scenario_ids", type=str, default="",
                        help="Optional comma-separated scenario subset.")
    parser.add_argument("--render_best_overall", action="store_true",
                        help="Render one best full-simulator design per objective and scenario.")
    parser.add_argument(
        "--render_best_per_method", action="store_true",
        help="Render the simulator-best seed for every method/objective/scenario.",
    )
    parser.add_argument(
        "--render_best_generalist", action="store_true",
        help="Render the one generalist with highest mean simulator utility per objective.",
    )
    parser.add_argument(
        "--render_best_generalist_per_method", action="store_true",
        help="Render the best mean-utility generalist seed from every method/objective.",
    )
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--measure_energy", action="store_true",
        help="Measure tendon displacement/work during final-design render reruns.",
    )
    args = parser.parse_args()
    root = args.study_dir.resolve()
    output = (args.output_dir or root / "study_analysis").resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = read_rows(root)
    requested_objectives = {value.strip() for value in args.objectives.split(",") if value.strip()}
    requested_scenarios = {value.strip() for value in args.scenario_ids.split(",") if value.strip()}
    if requested_objectives:
        rows = [row for row in rows if row["objective"] in requested_objectives]
    if requested_scenarios:
        rows = [row for row in rows if row["scenario_id"] in requested_scenarios]
    if not rows:
        raise ValueError(f"No successful benchmark_result.json files found under {root}")

    method_scenario = grouped_summary(rows, ("objective", "scenario_id", "method"))
    method_overall = grouped_summary(rows, ("objective", "method"))
    best_per_method = select_best(rows, ("objective", "scenario_id", "method"))
    best_overall = select_best(rows, ("objective", "scenario_id"))
    best_contact = select_best(
        rows, ("objective", "scenario_id"), value_key="contact_only_utility"
    )
    best_disturbance = select_best(
        rows, ("objective", "scenario_id"), value_key="disturbance_only_utility"
    )
    best_angular = select_best(
        rows, ("objective", "scenario_id"), value_key="angular_span_norm"
    )
    best_contact_per_method = select_best(
        rows, ("objective", "scenario_id", "method"), value_key="contact_only_utility"
    )
    best_disturbance_per_method = select_best(
        rows, ("objective", "scenario_id", "method"), value_key="disturbance_only_utility"
    )
    best_angular_per_method = select_best(
        rows, ("objective", "scenario_id", "method"), value_key="angular_span_norm"
    )
    generalist_candidates = generalist_candidate_summary(rows)
    if args.protocol == "generalist" and not generalist_candidates:
        raise ValueError(
            "No fixed design evaluated on multiple scenarios was found. Point --study_dir "
            "at a completed generalist study and do not filter it to one scenario."
        )
    generalist_methods = generalist_method_summary(generalist_candidates)
    best_generalist_per_method = select_best(
        generalist_candidates, ("objective", "method"), value_key="mean_utility"
    ) if generalist_candidates else []
    best_generalist = select_best(
        generalist_candidates, ("objective",), value_key="mean_utility"
    ) if generalist_candidates else []
    proposal_rows = read_proposal_times(root)
    if requested_objectives:
        proposal_rows = [row for row in proposal_rows if row["objective"] in requested_objectives]
    if requested_scenarios:
        proposal_rows = [row for row in proposal_rows if row["scenario_id"] in requested_scenarios]
    timings = timing_summary(proposal_rows, rows)
    calibration = grouped_summary_generic(
        [row for row in rows if row["selection_minus_simulator"] is not None],
        ("objective", "method"), "selection_minus_simulator",
    )
    for row in calibration:
        matching = [value for value in rows if value["objective"] == row["objective"]
                    and value["method"] == row["method"]]
        row["num_out_of_range_selection_scores"] = sum(
            bool(value["selection_out_of_range"]) for value in matching
        )

    write_csv(output / "all_rollouts.csv", rows, exclude=("design_params",))
    write_csv(output / "method_by_scenario.csv", method_scenario)
    write_csv(output / "method_overall.csv", method_overall)
    write_csv(output / "best_per_method_scenario.csv", best_per_method, exclude=("design_params",))
    write_csv(output / "best_overall_per_scenario.csv", best_overall, exclude=("design_params",))
    write_csv(output / "best_contact_per_scenario.csv", best_contact, exclude=("design_params",))
    write_csv(output / "best_disturbance_per_scenario.csv", best_disturbance, exclude=("design_params",))
    write_csv(output / "best_angular_span_per_scenario.csv", best_angular, exclude=("design_params",))
    write_csv(output / "best_contact_per_method_scenario.csv", best_contact_per_method,
              exclude=("design_params",))
    write_csv(output / "best_disturbance_per_method_scenario.csv", best_disturbance_per_method,
              exclude=("design_params",))
    write_csv(output / "best_angular_span_per_method_scenario.csv", best_angular_per_method,
              exclude=("design_params",))
    write_csv(output / "proposal_times.csv", proposal_rows)
    write_csv(output / "timing_summary.csv", timings)
    write_csv(output / "surrogate_gap_summary.csv", calibration)
    if args.protocol in {"auto", "generalist"} and generalist_candidates:
        write_csv(
            output / "generalist_candidate_summary.csv", generalist_candidates,
            exclude=("design_params",),
        )
        write_csv(output / "generalist_method_summary.csv", generalist_methods)
        write_csv(
            output / "best_generalist_per_method.csv", best_generalist_per_method,
            exclude=("design_params",),
        )
        write_csv(
            output / "best_generalist_overall.csv", best_generalist,
            exclude=("design_params",),
        )
    manifest = {
        "study_dir": str(root), "successful_rollouts": len(rows),
        "objectives": sorted({row["objective"] for row in rows}),
        "scenarios": sorted({row["scenario_id"] for row in rows}),
        "methods": sorted({row["method"] for row in rows}),
        "protocol": args.protocol,
        "generalist_designs_found": len(generalist_candidates),
        "selection_warning": (
            "Selection scores are surrogate predictions; simulator utility is authoritative. "
            "New candidate generation clamps predicted C/D/A to [0,1]. Existing candidates "
            "were selected by the code version used to create them."
        ),
    }
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[ANALYSIS] {len(rows)} successful rollouts -> {output}")
    if args.render_best_overall:
        visualization_dir = output / "best_overall_visualizations"
        render_rows(best_overall, visualization_dir,
                    args.num_workers, args.timeout, args.dry_run, args.measure_energy)
        if args.measure_energy and not args.dry_run:
            energy_rows = read_rows(visualization_dir)
            write_csv(
                output / "final_design_energy.csv", energy_rows,
                exclude=("design_params",),
            )
    if args.render_best_per_method:
        visualization_dir = output / "best_per_method_visualizations"
        render_rows(best_per_method, visualization_dir,
                    args.num_workers, args.timeout, args.dry_run, args.measure_energy)
        if args.measure_energy and not args.dry_run:
            energy_rows = read_rows(visualization_dir)
            write_csv(
                output / "per_method_final_design_energy.csv", energy_rows,
                exclude=("design_params",),
            )
    if args.render_best_generalist:
        visualization_dir = output / "best_generalist_visualizations"
        render_generalists(best_generalist, visualization_dir,
                           args.num_workers, args.timeout, args.dry_run, args.measure_energy)
        if args.measure_energy and not args.dry_run:
            write_csv(
                output / "best_generalist_energy.csv", read_rows(visualization_dir),
                exclude=("design_params",),
            )
    if args.render_best_generalist_per_method:
        visualization_dir = output / "best_generalist_per_method_visualizations"
        render_generalists(best_generalist_per_method, visualization_dir,
                           args.num_workers, args.timeout, args.dry_run, args.measure_energy)
        if args.measure_energy and not args.dry_run:
            write_csv(
                output / "generalist_per_method_energy.csv", read_rows(visualization_dir),
                exclude=("design_params",),
            )


if __name__ == "__main__":
    main()
