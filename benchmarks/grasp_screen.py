"""Exploratory simulator-metric grasp pass rates; not a physical retention test."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

DIRECTIONS = ("drag_left", "drag_right", "drag_down")
ALLOWED = {"min_contacts", "min_span_deg", "min_direction_support", "min_opposing_force_n"}


def screen(record, thresholds):
    """Missing/nonfinite measurements never pass; weakest direction governs."""
    if record.get("status") != "ok":
        return {"pass": False, "reason": "simulation_" + record.get("status", "missing")}
    m = record.get("metrics", {})
    checks = [("num_contacts", thresholds["min_contacts"]),
              ("angular_span", thresholds["min_span_deg"])]
    for direction in DIRECTIONS:
        if "min_direction_support" in thresholds:
            checks.append((f"disturbance_{direction}_directional_support_score", thresholds["min_direction_support"]))
        if "min_opposing_force_n" in thresholds:
            checks.append((f"disturbance_{direction}_opposing_force", thresholds["min_opposing_force_n"]))
    missing = [k for k, _ in checks if not isinstance(m.get(k), (int, float)) or not math.isfinite(m[k])]
    failed = [k for k, value in checks if k not in missing and m[k] < value]
    return {"pass": not missing and not failed, "reason": "missing:" + ",".join(missing) if missing else ",".join(failed) or "pass"}


def summarize(rows, rng):
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["profile"], r["method"])].append(r)
    output = []
    for (profile, method), group in sorted(grouped.items()):
        seeds = defaultdict(list)
        for r in group:
            seeds[r["seed"]].append(float(r["pass"]))
        means = np.array([np.mean(values) for values in seeds.values()])
        ci = None
        if len(means) >= 2:
            boot = rng.choice(means, size=(2000, len(means)), replace=True).mean(1)
            ci = np.quantile(boot, [.025, .975]).tolist()
        output.append(dict(profile=profile, method=method, trials=len(group),
            successes=sum(r["pass"] for r in group), trial_pass_rate=float(np.mean([r["pass"] for r in group])),
            seeds=len(means), seed_mean_pass_rate=float(means.mean()), seed_bootstrap_95ci=ci))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--thresholds", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--baseline", default="conditional_diffusion")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--labels", type=Path, help="Independent retention labels CSV: source,retention_success,split")
    args = p.parse_args()
    spec = json.loads(args.thresholds.read_text())
    for name, t in spec["profiles"].items():
        if set(t) - ALLOWED or not {"min_contacts", "min_span_deg"} <= set(t):
            raise ValueError(f"Invalid threshold keys for {name}")
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in t.values()):
            raise ValueError("Thresholds must be finite nonnegative numbers")
        if t.get("min_direction_support", 0) > 1 or t["min_span_deg"] > 360:
            raise ValueError("Direction support is in [0,1]; span is in degrees [0,360]")
    args.output.mkdir(parents=True, exist_ok=True)
    records, seen = [], set()
    for path in sorted(args.source.rglob("benchmark_result.json")):
        r = json.loads(path.read_text())
        identity = tuple(r.get(k) for k in ("method", "seed", "candidate_id", "scenario_id"))
        if identity in seen:
            raise ValueError(f"Duplicate trial identity {identity}; narrow --source to one protocol")
        seen.add(identity)
        r["source"] = str(path.resolve())
        records.append(r)
    if not records:
        raise ValueError("No benchmark results found")
    rows = []
    for r in records:
        for profile, thresholds in spec["profiles"].items():
            rows.append({**{k: r.get(k) for k in ("method", "seed", "candidate_id", "scenario_id", "source")},
                         "profile": profile, **screen(r, thresholds)})
    # Selection uses nominal simulator utility only, independently of threshold profiles.
    # Perturbed conditions supply the exploratory evaluation of that selected design.
    groups = defaultdict(list)
    for r in records:
        base, _, condition = r["scenario_id"].partition("@")
        groups[(r["method"], r["seed"], base)].append(r)
    selected_keys, selection = set(), []
    for (method, seed, base), group in sorted(groups.items()):
        nominal = [r for r in group if r["scenario_id"].endswith("@nominal") and r["status"] == "ok"
                   and isinstance(r.get("utility"), (float, int)) and math.isfinite(r["utility"])]
        winner = max(nominal, key=lambda r: (r["utility"], r["candidate_id"])) if nominal else None
        selection.append(dict(method=method, seed=seed, scenario=base,
            candidates=len({r["candidate_id"] for r in group}),
            nominal_successful_candidates=len(nominal), selected=winner["candidate_id"] if winner else None))
        if winner:
            selected_keys.add((method, seed, base, winner["candidate_id"]))
    heldout = [r for r in rows if "@" in r["scenario_id"] and not r["scenario_id"].endswith("@nominal")
               and (r["method"], r["seed"], r["scenario_id"].split("@")[0], r["candidate_id"]) in selected_keys]
    # Explicitly fail missing selected-design trials in conditions observed for this scenario.
    conditions = defaultdict(set)
    for r in records:
        base, _, condition = r["scenario_id"].partition("@")
        if condition and condition != "nominal":
            conditions[base].add(condition)
    present = {(r["method"], r["seed"], r["scenario_id"], r["profile"]) for r in heldout}
    for s in selection:
        for condition in sorted(conditions[s["scenario"]]):
            scenario = s["scenario"] + "@" + condition
            for profile in spec["profiles"]:
                if (s["method"], s["seed"], scenario, profile) not in present:
                    heldout.append(dict(method=s["method"], seed=s["seed"], scenario_id=scenario,
                        candidate_id=s["selected"], source="", profile=profile,
                        **{"pass": False, "reason": "missing_selected_trial_or_no_nominal_winner"}))
    rng = np.random.default_rng(args.seed)
    pool_summary, selected_summary = summarize(rows, rng), summarize(heldout, rng)
    for summary in (pool_summary, selected_summary):
        for row in summary:
            baseline = next((r for r in summary if r["profile"] == row["profile"] and r["method"] == args.baseline), None)
            if baseline:
                b = baseline["seed_mean_pass_rate"]
                row["difference_percentage_points_vs_baseline"] = 100*(row["seed_mean_pass_rate"]-b)
                row["relative_improvement_percent_vs_baseline"] = 100*(row["seed_mean_pass_rate"]-b)/b if b else None
    with (args.output / "trials.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / "selected_trials.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(heldout)
    # Blank labels must be supplied by a genuinely separate retention experiment.
    template = args.output / "retention_labels_template.csv"
    if not template.exists():
        with template.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=["source", "method", "retention_success", "split"])
            writer.writeheader()
            for method in sorted({r["method"] for r in records}):
                subset = [r for r in records if r["method"] == method and r["status"] == "ok"]
                for index in rng.choice(len(subset), min(4, len(subset)), replace=False):
                    r = subset[int(index)]
                    writer.writerow(dict(source=r["source"], method=method, retention_success="", split="calibration"))
    calibration = []
    if args.labels:
        labels = {}
        for label in csv.DictReader(args.labels.open()):
            if not label["retention_success"].strip():
                continue
            if label["retention_success"] not in ("0", "1") or label["split"] not in ("calibration", "validation"):
                raise ValueError("Labels require retention_success 0/1 and split calibration/validation")
            if label["source"] in labels:
                raise ValueError("Duplicate retention label")
            labels[label["source"]] = label
        if set(labels) - {r["source"] for r in records}:
            raise ValueError("Retention labels contain sources outside this report")
        for profile in spec["profiles"]:
            for split in ("calibration", "validation"):
                pairs = [(int(r["pass"]), int(labels[r["source"]]["retention_success"]))
                         for r in rows if r["profile"] == profile and r["source"] in labels
                         and labels[r["source"]]["split"] == split]
                tp = sum(a == 1 and b == 1 for a,b in pairs)
                fp = sum(a == 1 and b == 0 for a,b in pairs)
                tn = sum(a == 0 and b == 0 for a,b in pairs)
                fn = sum(a == 0 and b == 1 for a,b in pairs)
                calibration.append(dict(profile=profile, split=split, labeled_trials=len(pairs),
                    true_positive=tp, false_positive=fp, true_negative=tn, false_negative=fn,
                    precision=tp/(tp+fp) if tp+fp else None, recall=tp/(tp+fn) if tp+fn else None))
    report = dict(endpoint="exploratory geometric/load-proxy pass rate; not validated retention",
        thresholds=spec, thresholds_sha256=hashlib.sha256(args.thresholds.read_bytes()).hexdigest(),
        source=str(args.source.resolve()), baseline=args.baseline,
        independent_retention_calibration=calibration,
        labels_source=str(args.labels.resolve()) if args.labels else None,
        limitations=["Missing measurements and completed error/timeout records fail. Unrun jobs are not present in the denominator.",
            "Raw contact counts require consistent simulator discretization.",
            "Projected opposing force is not full force/torque equilibrium or sustained retention.",
            "Pool rates describe screened candidate pools, not all raw generated samples.",
            "Selected-design report chooses on nominal utility and evaluates recorded nonnominal conditions; these are exploratory, not fresh test data.",
            "Check selection_budgets for unequal candidate budgets and absent conditions before comparison.",
            "CI resamples seeds, not correlated rollouts. No CI with one seed; very few seeds cannot establish significance.",
            "Method differences are descriptive, unpaired, and require matched coverage for interpretation."],
        pool_summary=pool_summary, nominal_selected_perturbation_summary=selected_summary,
        selection_budgets=selection)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(pool_summary, indent=2))


if __name__ == "__main__":
    main()
