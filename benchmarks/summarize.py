"""Summarize benchmark rollout records into reproducible tables and plots."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmarks.protocol import DEFAULT_CONFIG, aggregate_records, load_config


def load_records(paths):
    records = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    records.append(json.loads(line))
    return records


def bootstrap_mean_ci(values, seed=0, iterations=2000, confidence=0.95):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(iterations, values.size))
    means = values[indices].mean(axis=1)
    alpha = 0.5 * (1.0 - confidence)
    return tuple(float(x) for x in np.quantile(means, [alpha, 1.0 - alpha]))


def summarize_candidates(records, config, bootstrap_iterations):
    grouped = defaultdict(list)
    all_jobs = defaultdict(int)
    for record in records:
        key = (record["method"], int(record["seed"]), record["candidate_id"])
        all_jobs[key] += 1
        if record.get("status") == "ok":
            grouped[key].append(record)

    rows = []
    for key in sorted(all_jobs):
        method, seed, candidate_id = key
        successful = grouped.get(key, [])
        row = {
            "method": method,
            "seed": seed,
            "candidate_id": candidate_id,
            "rollouts_total": all_jobs[key],
            "rollouts_ok": len(successful),
            "failure_rate": 1.0 - len(successful) / all_jobs[key],
        }
        if successful:
            summary = aggregate_records(successful, config)
            row["selection_score"] = successful[0].get("selection_score")
            low, high = bootstrap_mean_ci(
                [record["utility"] for record in successful],
                seed=seed,
                iterations=bootstrap_iterations,
            )
            row.update(summary)
            row["mean_utility_ci_low"] = low
            row["mean_utility_ci_high"] = high
            row["family_mean_utility"] = json.dumps(summary["family_mean_utility"], sort_keys=True)
            row["component_mean"] = json.dumps(summary["component_mean"], sort_keys=True)
            row["mean_num_contacts"] = summary["raw_metric_mean"]["num_contacts"]
            row["mean_disturbance_resistance"] = summary["raw_metric_mean"]["disturbance_resistance_score"]
            row["mean_angular_span_deg"] = summary["raw_metric_mean"]["angular_span_deg"]
            row["raw_metric_mean"] = json.dumps(summary["raw_metric_mean"], sort_keys=True)
        rows.append(row)
    return rows


def summarize_methods(candidate_rows):
    grouped = defaultdict(list)
    for row in candidate_rows:
        if "mean_utility" in row:
            grouped[(row["method"], row["seed"])].append(row)
    rows = []
    for (method, seed), candidates in sorted(grouped.items()):
        best = max(candidates, key=lambda item: item["mean_utility"])
        rows.append(
            {
                "method": method,
                "seed": seed,
                "num_candidates": len(candidates),
                "best_candidate_id": best["candidate_id"],
                "best_mean_utility": best["mean_utility"],
                "best_cvar20_utility": best["cvar20_utility"],
                "best_worst_family_utility": best["worst_family_utility"],
                "candidate_mean_utility": float(np.mean([x["mean_utility"] for x in candidates])),
                "best_mean_num_contacts": best["mean_num_contacts"],
                "best_mean_disturbance_resistance": best["mean_disturbance_resistance"],
                "best_mean_angular_span_deg": best["mean_angular_span_deg"],
                "total_failure_rate": float(
                    sum(x["failure_rate"] * x["rollouts_total"] for x in candidates)
                    / sum(x["rollouts_total"] for x in candidates)
                ),
            }
        )
    return rows


def aggregate_method_seeds(method_rows):
    grouped = defaultdict(list)
    for row in method_rows:
        grouped[row["method"]].append(row)
    rows = []
    for method, seeds in sorted(grouped.items()):
        means = np.asarray([row["best_mean_utility"] for row in seeds], dtype=float)
        cvars = np.asarray([row["best_cvar20_utility"] for row in seeds], dtype=float)
        worst = np.asarray([row["best_worst_family_utility"] for row in seeds], dtype=float)
        rows.append({
            "method": method,
            "num_method_seeds": len(seeds),
            "mean_utility_across_seeds": float(means.mean()),
            "std_utility_across_seeds": float(means.std()),
            "mean_cvar20_across_seeds": float(cvars.mean()),
            "mean_worst_family_across_seeds": float(worst.mean()),
            "mean_failure_rate_across_seeds": float(np.mean([row["total_failure_rate"] for row in seeds])),
        })
    return rows


def surrogate_calibration(candidate_rows):
    """Measure whether pre-simulation scores rank simulator outcomes correctly."""
    grouped = defaultdict(list)
    for row in candidate_rows:
        if row.get("selection_score") is not None and "mean_utility" in row:
            grouped[row["method"]].append(row)
    rows = []
    for method, candidates in sorted(grouped.items()):
        predicted = np.asarray([row["selection_score"] for row in candidates], dtype=float)
        actual = np.asarray([row["mean_utility"] for row in candidates], dtype=float)
        pearson = float("nan")
        spearman = float("nan")
        if len(candidates) >= 2 and predicted.std() > 0 and actual.std() > 0:
            pearson = float(np.corrcoef(predicted, actual)[0, 1])
            predicted_rank = np.argsort(np.argsort(predicted))
            actual_rank = np.argsort(np.argsort(actual))
            spearman = float(np.corrcoef(predicted_rank, actual_rank)[0, 1])
        rows.append({
            "method": method,
            "num_candidates": len(candidates),
            "predicted_actual_pearson": pearson,
            "predicted_actual_spearman": spearman,
            "mean_selection_score": float(predicted.mean()),
            "mean_simulator_utility": float(actual.mean()),
        })
    return rows


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_method_summary(rows, path):
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib unavailable; skipping comparison plot")
        return
    labels = [f"{row['method']}\nseed {row['seed']}" for row in rows]
    x = np.arange(len(rows))
    width = 0.25
    figure, axis = plt.subplots(figsize=(max(8, len(rows) * 1.2), 5))
    axis.bar(x - width, [r["best_mean_utility"] for r in rows], width, label="Mean")
    axis.bar(x, [r["best_cvar20_utility"] for r in rows], width, label="Worst 20% CVaR")
    axis.bar(x + width, [r["best_worst_family_utility"] for r in rows], width, label="Worst family")
    axis.set_xticks(x, labels, rotation=20, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Normalized benchmark utility")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Summarize Squirrel Benchmark V1 results.")
    parser.add_argument("records", nargs="+", type=Path, help="One or more records.jsonl files")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--bootstrap_iterations", type=int, default=2000)
    args = parser.parse_args()
    config = load_config(args.config)
    records = load_records(args.records)
    candidate_rows = summarize_candidates(records, config, args.bootstrap_iterations)
    method_rows = summarize_methods(candidate_rows)
    method_aggregate_rows = aggregate_method_seeds(method_rows)
    calibration_rows = surrogate_calibration(candidate_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "candidate_summary.csv", candidate_rows)
    write_csv(args.output_dir / "method_summary.csv", method_rows)
    write_csv(args.output_dir / "method_aggregate.csv", method_aggregate_rows)
    write_csv(args.output_dir / "surrogate_calibration.csv", calibration_rows)
    plot_method_summary(method_rows, args.output_dir / "method_comparison.png")
    summary = {
        "records": [str(path.resolve()) for path in args.records],
        "rollouts_total": len(records),
        "rollouts_ok": sum(record.get("status") == "ok" for record in records),
        "candidate_rows": candidate_rows,
        "method_rows": method_rows,
        "method_aggregate_rows": method_aggregate_rows,
        "surrogate_calibration_rows": calibration_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[SUMMARY] {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
