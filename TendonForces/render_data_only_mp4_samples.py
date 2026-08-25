#!/usr/bin/env python3
"""Randomly re-render MP4 videos from a data-only TendonForces dataset.

The data-only runs save numeric ``master_log_*.npz`` files but skip MP4
generation.  This script samples a subset of those NPZ files, reconstructs the
original ``finger.py`` command from the stored ``arg_*`` entries, removes
``--data_only`` / ``--disable_video_plots``, and reruns the selected cases into
a new output folder.

Example:
    cd TendonForces
    python render_data_only_mp4_samples.py \
        --dataset_dir runs/exp_new/train \
        --output_dir runs/exp_new/rendered_mp4_samples \
        --n 100 \
        --num_workers 4
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import math
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError:  # allow --help to work outside the simulation venv
    np = None


BOOLEAN_OPTIONAL_KEYS = {
    "auto_contact_stiffness",
    "prescribed_stop_at_contact",
    "force_driven_stabilize",
    "force_driven_lock_base_xy",
    "force_driven_z_stabilize",
    "force_driven_rot_stabilize",
}

SKIP_ARG_KEYS = {
    # These two are the reason the source run did not produce videos.
    "data_only",
    "disable_video_plots",
    # The output destination/suffix should be controlled by this script.
    "output_dir",
    "suffix",
    # Avoid interactive/extra diagnostics unless explicitly requested later.
    "debug",
    "torque_debug",
    "confirm_tendon",
}


def scalarize(value: Any) -> Any:
    """Convert a saved NPZ value into a Python scalar/string/list if possible."""
    if np is None:
        raise ModuleNotFoundError(
            "This script needs numpy to read .npz files. Run it inside the same "
            "environment you use for finger.py."
        )
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return arr.tolist()


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == "" or value.strip().lower() == "none"
    if isinstance(value, float):
        return math.isnan(value)
    return False


def stringify(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(stringify(v) for v in value)
    if np is not None and isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def find_npz_files(dataset_dir: Path) -> list[Path]:
    master_logs = sorted(dataset_dir.rglob("master_log_*.npz"))
    if master_logs:
        return master_logs
    return sorted(dataset_dir.rglob("*.npz"))


def sample_npz_files(npz_files: list[Path], n: int, seed: int) -> list[Path]:
    if not npz_files:
        raise FileNotFoundError("No .npz files found in the dataset folder.")
    rng = random.Random(seed)
    if len(npz_files) <= n:
        print(
            f"[WARN] Requested n={n}, but only found {len(npz_files)} NPZ files. "
            "Rendering all of them."
        )
        selected = list(npz_files)
    else:
        selected = rng.sample(npz_files, n)
    return sorted(selected)


def load_args_from_npz(npz_path: Path) -> dict[str, Any]:
    if np is None:
        raise ModuleNotFoundError(
            "This script needs numpy to read .npz files. Run it inside the same "
            "environment you use for finger.py."
        )
    args: dict[str, Any] = {}
    with np.load(npz_path, allow_pickle=True) as data:
        for key in data.files:
            if key.startswith("arg_"):
                args[key[len("arg_") :]] = scalarize(data[key])
    if not args:
        raise ValueError(f"No arg_* entries found in {npz_path}")
    return args


def build_finger_command(
    npz_path: Path,
    output_dir: Path,
    sample_index: int,
    python_executable: str,
    finger_py: Path,
) -> list[str]:
    saved_args = load_args_from_npz(npz_path)
    source_suffix = str(saved_args.get("suffix", npz_path.stem))
    render_suffix = f"rerender_{sample_index:03d}_{source_suffix}"

    cmd = [python_executable, str(finger_py)]
    for key in sorted(saved_args):
        if key in SKIP_ARG_KEYS:
            continue
        value = saved_args[key]
        if is_missing(value):
            continue

        if isinstance(value, (bool, np.bool_)):
            if key in BOOLEAN_OPTIONAL_KEYS:
                cmd.append(f"--{key}" if bool(value) else f"--no-{key}")
            elif bool(value):
                cmd.append(f"--{key}")
            continue

        cmd.extend([f"--{key}", stringify(value)])

    cmd.extend(["--output_dir", str(output_dir), "--suffix", render_suffix])
    return cmd


def render_one(task: tuple[int, str, str, str, str, bool]) -> dict[str, Any]:
    idx, npz_str, output_dir_str, python_executable, finger_py_str, dry_run = task
    npz_path = Path(npz_str)
    output_dir = Path(output_dir_str)
    finger_py = Path(finger_py_str)
    cmd = build_finger_command(
        npz_path=npz_path,
        output_dir=output_dir,
        sample_index=idx,
        python_executable=python_executable,
        finger_py=finger_py,
    )

    if dry_run:
        return {
            "ok": True,
            "npz": str(npz_path),
            "cmd": cmd,
            "stdout_tail": "",
            "stderr_tail": "",
        }

    proc = subprocess.run(
        cmd,
        cwd=str(finger_py.parent),
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "ok": proc.returncode == 0,
        "npz": str(npz_path),
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Randomly sample data-only NPZ runs and regenerate MP4 videos."
    )
    parser.add_argument(
        "--dataset_dir",
        required=True,
        help="Dataset folder containing data-only .npz files, e.g. runs/exp4/train.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Where to save rendered MP4s. Default: <dataset_dir>/rendered_mp4_samples.",
    )
    parser.add_argument("--n", type=int, default=100, help="Number of samples to render.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel finger.py render subprocesses.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for finger.py. Default: current Python.",
    )
    parser.add_argument(
        "--finger_py",
        default=None,
        help="Path to finger.py. Default: sibling file next to this script.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Write selected files and commands, but do not run simulations.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else dataset_dir / "rendered_mp4_samples"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    finger_py = (
        Path(args.finger_py).expanduser().resolve()
        if args.finger_py
        else Path(__file__).resolve().parent / "finger.py"
    )
    if not finger_py.exists():
        raise FileNotFoundError(f"finger.py not found: {finger_py}")

    npz_files = find_npz_files(dataset_dir)
    selected = sample_npz_files(npz_files, n=args.n, seed=args.seed)

    selected_list_path = output_dir / "selected_npz.txt"
    with selected_list_path.open("w") as f:
        for path in selected:
            f.write(str(path) + "\n")

    tasks = [
        (
            idx,
            str(npz_path),
            str(output_dir),
            args.python,
            str(finger_py),
            args.dry_run,
        )
        for idx, npz_path in enumerate(selected)
    ]

    print(f"[FOUND] {len(npz_files)} NPZ files")
    print(f"[SELECTED] {len(selected)} files")
    print(f"[OUTPUT] {output_dir}")
    print(f"[SELECTED LIST] {selected_list_path}")
    print(f"[WORKERS] {max(1, args.num_workers)}")

    results: list[dict[str, Any]] = []
    printed_first_failure = False
    with futures.ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as executor:
        future_to_idx = {executor.submit(render_one, task): task[0] for task in tasks}
        for future in futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "ok": False,
                    "npz": tasks[idx][1],
                    "cmd": [],
                    "returncode": None,
                    "stdout_tail": "",
                    "stderr_tail": repr(exc),
                }
            results.append(result)
            status = "OK" if result["ok"] else "FAIL"
            print(f"[{status}] {idx + 1:03d}/{len(tasks):03d} {Path(result['npz']).name}")
            if (not result["ok"]) and (not printed_first_failure):
                printed_first_failure = True
                print("\n[FIRST FAILURE DEBUG]")
                print("Command:")
                print(" ".join(result.get("cmd", [])))
                if result.get("returncode") is not None:
                    print(f"Return code: {result['returncode']}")
                stdout_tail = result.get("stdout_tail") or ""
                stderr_tail = result.get("stderr_tail") or ""
                if stdout_tail:
                    print("\nstdout tail:")
                    print(stdout_tail)
                if stderr_tail:
                    print("\nstderr tail:")
                    print(stderr_tail)
                print("[END FIRST FAILURE DEBUG]\n")

    results = sorted(results, key=lambda r: r["npz"])
    summary = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "n_requested": args.n,
        "n_selected": len(selected),
        "seed": args.seed,
        "num_workers": max(1, args.num_workers),
        "dry_run": args.dry_run,
        "successes": sum(1 for r in results if r["ok"]),
        "failures": sum(1 for r in results if not r["ok"]),
        "results": results,
    }
    summary_path = output_dir / "render_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"[SUMMARY] {summary_path}")
    print(f"[DONE] successes={summary['successes']} failures={summary['failures']}")


if __name__ == "__main__":
    main()
