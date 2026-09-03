"""Measure surrogate-selection regret and oracle quality in candidate pools."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.study_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    for result_path in sorted(root.rglob("benchmark_result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "ok":
            continue
        result["_result_path"] = str(result_path.resolve())
        manifest_path = result_path
        while manifest_path != root and not (manifest_path / "manifest.json").exists():
            manifest_path = manifest_path.parent
        manifest_file = manifest_path / "manifest.json"
        if not manifest_file.exists():
            continue
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        source = manifest.get("candidate_source", "")
        key = (
            result.get("scenario_id", ""), result.get("method", ""),
            int(result.get("seed", 0)), source,
        )
        groups[key].append(result)

    summary = []
    candidates = []
    for (scenario, method, seed, source), rows in sorted(groups.items()):
        by_selection = sorted(
            rows,
            key=lambda row: float("-inf") if row.get("selection_score") is None
            else float(row["selection_score"]),
            reverse=True,
        )
        by_simulator = sorted(rows, key=lambda row: float(row["utility"]), reverse=True)
        selected = by_selection[0]
        oracle = by_simulator[0]
        selected_utility = float(selected["utility"])
        oracle_utility = float(oracle["utility"])
        simulator_rank_of_selected = next(
            i for i, row in enumerate(by_simulator, 1)
            if row.get("candidate_id") == selected.get("candidate_id")
        )
        summary.append({
            "scenario_id": scenario,
            "method": method,
            "seed": seed,
            "num_candidates_simulated": len(rows),
            "selected_candidate_id": selected.get("candidate_id", ""),
            "selected_utility": selected_utility,
            "oracle_candidate_id": oracle.get("candidate_id", ""),
            "oracle_utility": oracle_utility,
            "selection_regret": oracle_utility - selected_utility,
            "simulator_rank_of_selected": simulator_rank_of_selected,
            "candidate_source": source,
        })
        for simulator_rank, row in enumerate(by_simulator, 1):
            metrics = row.get("normalized_metrics", {})
            candidates.append({
                "scenario_id": scenario,
                "method": method,
                "seed": seed,
                "candidate_id": row.get("candidate_id", ""),
                "selection_score": row.get("selection_score"),
                "simulator_utility": row.get("utility"),
                "simulator_rank": simulator_rank,
                "contact_coverage_norm": metrics.get("contact_coverage_norm"),
                "disturbance_resistance_score": metrics.get("disturbance_resistance_score"),
                "angular_span_norm": metrics.get("angular_span_norm"),
                "result_path": row.get("_result_path", ""),
            })

    write_csv(output / "candidate_pool_selection_regret.csv", summary)
    write_csv(output / "all_candidate_metrics.csv", candidates)
    winners = []
    by_scenario = defaultdict(list)
    for row in summary:
        by_scenario[row["scenario_id"]].append(row)
    for scenario, rows in sorted(by_scenario.items()):
        ranked = sorted(rows, key=lambda row: row["oracle_utility"], reverse=True)
        winner = ranked[0]
        runner_up = ranked[1] if len(ranked) > 1 else winner
        winners.append({
            "scenario_id": scenario,
            "winning_method": winner["method"],
            "winning_seed": winner["seed"],
            "winning_candidate_id": winner["oracle_candidate_id"],
            "winning_utility": winner["oracle_utility"],
            "runner_up_method": runner_up["method"],
            "runner_up_utility": runner_up["oracle_utility"],
            "absolute_margin": winner["oracle_utility"] - runner_up["oracle_utility"],
        })
    write_csv(output / "oracle_winners.csv", winners)
    print(f"[ANALYSIS] pools={len(summary)} -> {output}")


if __name__ == "__main__":
    main()
