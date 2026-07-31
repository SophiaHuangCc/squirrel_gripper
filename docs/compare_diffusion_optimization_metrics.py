#!/usr/bin/env python3
"""
Build a slide-ready metric table comparing verified optimization and diffusion results.

This script does not run new simulations. It reads the outputs produced by:

  optimization/evaluate_optimized_candidates.py
  generator/evaluate_generated_candidates.py

and extracts the actual finger.py verification metrics from each verified
`finger_*` folder. It then selects one representative candidate per method and
writes both CSV and Markdown tables.

Example:
  python docs/compare_diffusion_optimization_metrics.py \
      --optimization_dir "optimization/runs/exp20/sim_verification/disturbance_contact_span_speed_verified_top10" \
      --diffusion_dir "generator/runs/sample_exp3/sim_verification" \
      --output_dir "docs/method_comparison_metrics"

If you do not remember the exact folders, try:
  python docs/compare_diffusion_optimization_metrics.py \
      --auto \
      --optimization_hint exp20 \
      --output_dir "docs/method_comparison_metrics"
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


METRIC_KEYS = [
    "num_contacts",
    "contact_norm",
    "angular_span",
    "angular_span_norm",
    "disturbance_resistance_score",
    "curl_speed_score",
    "curl_time",
    "raw_pipeline_score",
    "normalized_slide_score",
]


@dataclass
class CandidateRecord:
    method: str
    verification_dir: Path
    finger_dir: Path
    finger_rank: int
    selected_id: Optional[int]
    pred_metrics: Optional[np.ndarray]
    pred_score: Optional[float]
    metrics: Dict[str, float]
    design: Dict[str, Any]
    master_log_path: Optional[Path]


def scalar_from_npz(data: np.lib.npyio.NpzFile, keys: Sequence[str], default: float = 0.0) -> float:
    for key in keys:
        if key not in data:
            continue
        value = np.asarray(data[key])
        if value.size == 0:
            continue
        item = value.reshape(-1)[0]
        try:
            return float(item)
        except (TypeError, ValueError):
            try:
                return float(str(item))
            except ValueError:
                return float(default)
    return float(default)


def safe_float(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return default
    try:
        arr = np.asarray(value)
        if arr.size:
            value = arr.reshape(-1)[0]
        return float(value)
    except Exception:
        return default


def normalize_contact(num_contacts: float, n_elements: float) -> float:
    return float(np.clip(np.log1p(num_contacts) / np.log1p(max(n_elements, 1.0)), 0.0, 1.0))


def normalize_span(angular_span: float) -> float:
    return float(np.clip(angular_span / 180.0, 0.0, 1.0))


def compute_scores(metrics: Dict[str, float]) -> Dict[str, float]:
    n_elements = max(float(metrics.get("n_elements", 100.0)), 1.0)
    contact_norm = normalize_contact(metrics.get("num_contacts", 0.0), n_elements)
    angular_span_norm = normalize_span(metrics.get("angular_span", 0.0))
    disturbance = float(np.clip(metrics.get("disturbance_resistance_score", 0.0), 0.0, 1.0))
    curl_speed = float(np.clip(metrics.get("curl_speed_score", 0.0), 0.0, 1.0))

    # This mirrors the raw score currently saved by sim_test_mj.py / metrics.py.
    # It is useful for tracing exactly what the optimization code was ranking.
    raw_pipeline_score = (
        metrics.get("disturbance_resistance_score", 0.0)
        + 0.1 * metrics.get("num_contacts", 0.0)
        + 0.5 * metrics.get("angular_span", 0.0)
    )

    # This is better for slides because all terms are comparable ranges.
    normalized_slide_score = disturbance + 0.1 * contact_norm + 0.5 * angular_span_norm

    # Optional speed-aware score; useful if you decide to put curl speed back in.
    speed_gate = 1.0 / (1.0 + np.exp(-(contact_norm - 0.3) / 0.05))
    normalized_slide_score_with_speed = normalized_slide_score + 0.1 * curl_speed * speed_gate

    return {
        "contact_norm": contact_norm,
        "angular_span_norm": angular_span_norm,
        "raw_pipeline_score": float(raw_pipeline_score),
        "normalized_slide_score": float(normalized_slide_score),
        "normalized_slide_score_with_speed": float(normalized_slide_score_with_speed),
    }


def read_metrics_from_master_log(npz_path: Path) -> Dict[str, float]:
    with np.load(npz_path, allow_pickle=True) as data:
        metrics = {
            "num_contacts": scalar_from_npz(data, ["num_contacts"], 0.0),
            "disturbance_resistance_score": scalar_from_npz(
                data, ["disturbance_resistance_score"], 0.0
            ),
            "angular_span": scalar_from_npz(data, ["angular_span"], 0.0),
            "curl_time": scalar_from_npz(data, ["curl_time"], float("nan")),
            "curl_speed_score": scalar_from_npz(data, ["curl_speed_score"], 0.0),
            "n_elements": scalar_from_npz(data, ["n_elements", "arg_n_elements"], 100.0),
        }
    metrics.update(compute_scores(metrics))
    return metrics


def latest_master_log(finger_dir: Path) -> Optional[Path]:
    files = sorted(
        finger_dir.rglob("master_log_*.npz"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def read_design_json(finger_dir: Path) -> Dict[str, Any]:
    path = finger_dir / "design.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def finger_rank_from_dir(path: Path) -> Optional[int]:
    match = re.search(r"finger_?(\d+)$", path.name)
    if match:
        return int(match.group(1))
    return None


def unpack_object_array(arr: Any) -> List[Any]:
    if arr is None:
        return []
    arr = np.asarray(arr, dtype=object)
    return [arr.reshape(-1)[i] for i in range(arr.size)]


def load_selection_metadata(verification_dir: Path) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "selected_ids": [],
        "pred_metrics": [],
        "pred_scores": [],
    }

    verification_npz = verification_dir / "verification_results.npz"
    selected_npzs = [
        verification_dir / "selected_candidates.npz",
        verification_dir / "selected_generated_candidates.npz",
    ]

    paths = [verification_npz, *selected_npzs]
    for path in paths:
        if not path.exists():
            continue
        try:
            with np.load(path, allow_pickle=True) as data:
                if "selected_ids" in data and not metadata["selected_ids"]:
                    metadata["selected_ids"] = [int(x) for x in np.asarray(data["selected_ids"]).reshape(-1)]
                if "pred_metrics" in data and not metadata["pred_metrics"]:
                    pred = np.asarray(data["pred_metrics"])
                    if pred.size and pred.dtype != object:
                        metadata["pred_metrics"] = [pred[i] for i in range(pred.shape[0])]
                if "pred_scores" in data and not metadata["pred_scores"]:
                    scores = np.asarray(data["pred_scores"]).reshape(-1)
                    if scores.size:
                        metadata["pred_scores"] = [safe_float(x) for x in scores]
        except Exception as exc:
            print(f"[WARN] Could not read metadata from {path}: {exc}")

    return metadata


def find_finger_dirs(verification_dir: Path) -> List[Path]:
    candidates = [
        p for p in verification_dir.iterdir()
        if p.is_dir() and finger_rank_from_dir(p) is not None
    ]
    return sorted(candidates, key=lambda p: finger_rank_from_dir(p) or 0)


def load_records(method: str, verification_dir: Path) -> List[CandidateRecord]:
    verification_dir = verification_dir.expanduser().resolve()
    if not verification_dir.exists():
        raise FileNotFoundError(f"{method} verification folder does not exist: {verification_dir}")

    metadata = load_selection_metadata(verification_dir)
    finger_dirs = find_finger_dirs(verification_dir)
    if not finger_dirs:
        raise FileNotFoundError(
            f"No finger_* folders found in {verification_dir}. "
            "Make sure this points to a simulation verification output folder."
        )

    records: List[CandidateRecord] = []
    for finger_dir in finger_dirs:
        rank = finger_rank_from_dir(finger_dir)
        if rank is None:
            continue

        master_log = latest_master_log(finger_dir)
        if master_log is None:
            print(f"[WARN] Skipping {finger_dir}: no master_log_*.npz")
            continue

        metrics = read_metrics_from_master_log(master_log)
        design = read_design_json(finger_dir)

        selected_id = None
        if rank < len(metadata["selected_ids"]):
            selected_id = metadata["selected_ids"][rank]

        pred_metrics = None
        if rank < len(metadata["pred_metrics"]):
            pred_metrics = np.asarray(metadata["pred_metrics"][rank], dtype=float)

        pred_score = None
        if rank < len(metadata["pred_scores"]):
            pred_score = safe_float(metadata["pred_scores"][rank], None)

        records.append(
            CandidateRecord(
                method=method,
                verification_dir=verification_dir,
                finger_dir=finger_dir,
                finger_rank=rank,
                selected_id=selected_id,
                pred_metrics=pred_metrics,
                pred_score=pred_score,
                metrics=metrics,
                design=design,
                master_log_path=master_log,
            )
        )

    return records


def find_verification_dirs(root: Path, hint: str = "") -> List[Path]:
    root = root.expanduser().resolve()
    if not root.exists():
        return []

    files = sorted(root.rglob("verification_results.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    dirs = [p.parent for p in files]

    if hint:
        hint_lower = hint.lower()
        hinted = [d for d in dirs if hint_lower in str(d).lower()]
        if hinted:
            return hinted
    return dirs


def choose_best(records: Sequence[CandidateRecord], select_by: str) -> CandidateRecord:
    if not records:
        raise ValueError("No candidate records to choose from.")
    if select_by == "raw_pipeline_score":
        return max(records, key=lambda r: r.metrics["raw_pipeline_score"])
    if select_by == "normalized_slide_score":
        return max(records, key=lambda r: r.metrics["normalized_slide_score"])
    if select_by == "normalized_slide_score_with_speed":
        return max(records, key=lambda r: r.metrics["normalized_slide_score_with_speed"])
    if select_by == "finger_rank":
        return min(records, key=lambda r: r.finger_rank)
    if select_by == "pred_score":
        valid = [r for r in records if r.pred_score is not None and not math.isnan(float(r.pred_score))]
        if valid:
            return max(valid, key=lambda r: float(r.pred_score))
        print("[WARN] No pred_score found; falling back to finger_rank.")
        return min(records, key=lambda r: r.finger_rank)
    raise ValueError(f"Unknown --select_by: {select_by}")


def design_summary(design: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "v_list",
        "joint_softness",
        "link_lengths",
        "base_rad",
        "base_len",
        "tension",
        "ankle_wrap_radius",
        "ankle_stiffness",
        "approach_deg",
        "cyl_rad",
    ]
    out = {}
    for key in keys:
        if key in design:
            out[key] = design[key]
    sim_params = design.get("sim_params")
    if isinstance(sim_params, dict):
        for key in ["joint_stiffness_mode", "damping", "nu_contact", "vel_damp_contact", "k_contact", "final_time"]:
            if key in sim_params:
                out[f"sim_{key}"] = sim_params[key]
    return out


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if abs(value) >= 100:
            return f"{value:.1f}"
        return f"{value:.3f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, np.ndarray):
        return np.array2string(value, precision=3, separator=", ")
    if isinstance(value, list):
        return json.dumps(value)
    return str(value)


def candidate_to_row(record: CandidateRecord) -> Dict[str, Any]:
    m = record.metrics
    row: Dict[str, Any] = {
        "method": record.method,
        "finger_rank": record.finger_rank,
        "selected_id": record.selected_id,
        "pred_score": record.pred_score,
        "num_contacts": m.get("num_contacts"),
        "contact_norm": m.get("contact_norm"),
        "angular_span_deg": m.get("angular_span"),
        "angular_span_norm": m.get("angular_span_norm"),
        "disturbance_resistance_score": m.get("disturbance_resistance_score"),
        "curl_speed_score": m.get("curl_speed_score"),
        "curl_time_s": m.get("curl_time"),
        "raw_pipeline_score": m.get("raw_pipeline_score"),
        "normalized_slide_score": m.get("normalized_slide_score"),
        "normalized_slide_score_with_speed": m.get("normalized_slide_score_with_speed"),
        "finger_dir": str(record.finger_dir),
        "master_log_path": "" if record.master_log_path is None else str(record.master_log_path),
    }
    for key, value in design_summary(record.design).items():
        row[f"design_{key}"] = value
    return row


def write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def write_markdown(selected_rows: Sequence[Dict[str, Any]], path: Path) -> None:
    columns = [
        ("Metric", "metric"),
        ("Optimization", "optimization"),
        ("Diffusion", "diffusion"),
    ]

    by_method = {row["method"]: row for row in selected_rows}
    opt = by_method.get("Optimization", {})
    diff = by_method.get("Diffusion", {})

    metric_rows = [
        ("Selected candidate", "finger_rank"),
        ("Verified final score, normalized", "normalized_slide_score"),
        ("Verified final score, pipeline/raw", "raw_pipeline_score"),
        ("Contact count", "num_contacts"),
        ("Contact count, normalized", "contact_norm"),
        ("Angular span (deg)", "angular_span_deg"),
        ("Angular span, normalized", "angular_span_norm"),
        ("Disturbance resistance", "disturbance_resistance_score"),
        ("Curl speed score", "curl_speed_score"),
        ("Curl time (s)", "curl_time_s"),
        ("Predicted score", "pred_score"),
    ]

    lines = []
    lines.append("| " + " | ".join(c[0] for c in columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for label, key in metric_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    format_value(opt.get(key)),
                    format_value(diff.get(key)),
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("Notes:")
    lines.append("")
    lines.append(
        "- `normalized` score uses disturbance + 0.1*normalized_contacts + 0.5*normalized_angular_span."
    )
    lines.append(
        "- `pipeline/raw` score mirrors the current simulator objective using raw contact count and angular span in degrees; it can be useful for debugging but may over-emphasize angular span."
    )

    path.write_text("\n".join(lines) + "\n")


def print_available(label: str, dirs: Sequence[Path]) -> None:
    print(f"[{label}] found {len(dirs)} verification dirs")
    for i, d in enumerate(dirs[:10]):
        print(f"  {i}: {d}")
    if len(dirs) > 10:
        print(f"  ... {len(dirs) - 10} more")


def resolve_dirs(args) -> Tuple[Path, Path]:
    opt_dir = Path(args.optimization_dir).expanduser().resolve() if args.optimization_dir else None
    diff_dir = Path(args.diffusion_dir).expanduser().resolve() if args.diffusion_dir else None

    if args.auto or opt_dir is None:
        opt_candidates = find_verification_dirs(Path(args.optimization_root), args.optimization_hint)
        print_available("optimization", opt_candidates)
        if opt_dir is None:
            if not opt_candidates:
                raise FileNotFoundError(
                    "Could not auto-find optimization verification_results.npz. "
                    "Pass --optimization_dir explicitly."
                )
            opt_dir = opt_candidates[0]

    if args.auto or diff_dir is None:
        diff_candidates = find_verification_dirs(Path(args.diffusion_root), args.diffusion_hint)
        print_available("diffusion", diff_candidates)
        if diff_dir is None:
            if not diff_candidates:
                raise FileNotFoundError(
                    "Could not auto-find diffusion verification_results.npz. "
                    "Pass --diffusion_dir explicitly."
                )
            diff_dir = diff_candidates[0]

    assert opt_dir is not None
    assert diff_dir is not None
    return opt_dir, diff_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract verified optimization/diffusion metrics into a slide-ready comparison table."
    )
    parser.add_argument("--optimization_dir", type=str, default="")
    parser.add_argument("--diffusion_dir", type=str, default="")
    parser.add_argument("--optimization_root", type=str, default="optimization")
    parser.add_argument("--diffusion_root", type=str, default="generator")
    parser.add_argument("--optimization_hint", type=str, default="exp20")
    parser.add_argument("--diffusion_hint", type=str, default="sample_exp3")
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--output_dir", type=str, default="docs/method_comparison_metrics")
    parser.add_argument(
        "--select_by",
        choices=[
            "finger_rank",
            "pred_score",
            "raw_pipeline_score",
            "normalized_slide_score",
            "normalized_slide_score_with_speed",
        ],
        default="finger_rank",
        help=(
            "Which candidate to report per method. "
            "finger_rank uses finger_0, matching top-k order from the evaluator."
        ),
    )
    args = parser.parse_args()

    optimization_dir, diffusion_dir = resolve_dirs(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[USING] optimization_dir={optimization_dir}")
    print(f"[USING] diffusion_dir={diffusion_dir}")

    opt_records = load_records("Optimization", optimization_dir)
    diff_records = load_records("Diffusion", diffusion_dir)

    opt_selected = choose_best(opt_records, args.select_by)
    diff_selected = choose_best(diff_records, args.select_by)

    all_rows = [candidate_to_row(r) for r in [*opt_records, *diff_records]]
    selected_rows = [candidate_to_row(opt_selected), candidate_to_row(diff_selected)]

    write_csv(all_rows, output_dir / "all_verified_candidates.csv")
    write_csv(selected_rows, output_dir / "selected_method_comparison.csv")
    write_markdown(selected_rows, output_dir / "selected_method_comparison.md")

    payload = {
        "selection_rule": args.select_by,
        "optimization_dir": str(optimization_dir),
        "diffusion_dir": str(diffusion_dir),
        "selected": selected_rows,
    }
    (output_dir / "selected_method_comparison.json").write_text(json.dumps(payload, indent=2))

    print("\n[SELECTED FOR SLIDE]")
    for row in selected_rows:
        print(
            f"  {row['method']}: finger_{row['finger_rank']} "
            f"norm_score={row['normalized_slide_score']:.3f} "
            f"raw_score={row['raw_pipeline_score']:.3f} "
            f"contacts={row['num_contacts']:.0f} "
            f"span={row['angular_span_deg']:.1f} "
            f"disturbance={row['disturbance_resistance_score']:.3f}"
        )

    print(f"\n[DONE] wrote:")
    print(f"  {output_dir / 'selected_method_comparison.md'}")
    print(f"  {output_dir / 'selected_method_comparison.csv'}")
    print(f"  {output_dir / 'all_verified_candidates.csv'}")


if __name__ == "__main__":
    main()
