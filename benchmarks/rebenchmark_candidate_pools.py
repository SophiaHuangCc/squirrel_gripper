"""Re-simulate every saved candidate without regenerating proposals."""

import argparse
import re
import subprocess
import sys
from pathlib import Path


SCENARIO_RE = re.compile(r"^approach_radius-(\d+)$")


def nearest(path: Path, name: str, stop: Path) -> Path | None:
    for parent in (path.parent, *path.parents):
        candidate = parent / name
        if candidate.exists():
            return candidate
        if parent == stop:
            break
    return None


def scenario_id(path: Path, stop: Path) -> str | None:
    for parent in (path.parent, *path.parents):
        match = SCENARIO_RE.match(parent.name)
        if match:
            return f"approach_radius:{match.group(1)}"
        if parent == stop:
            break
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate all candidates already produced by a specialist study."
    )
    parser.add_argument("--study_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--top_k", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    root = args.study_dir.resolve()
    output = args.output_dir.resolve()
    candidate_files = sorted(root.rglob("candidates/*.npz"))
    if not candidate_files:
        raise ValueError(f"No candidate NPZ files found under {root}")

    commands = []
    for candidate in candidate_files:
        scenario = scenario_id(candidate, root)
        config = nearest(candidate, "effective_config.json", root)
        if scenario is None or config is None:
            print(f"[SKIP] cannot resolve scenario/config for {candidate}")
            continue
        relative_group = candidate.parent.parent.relative_to(root)
        group_output = output / relative_group / candidate.stem
        command = [
            sys.executable, "-m", "benchmarks.run_sim_benchmark",
            "--candidates", str(candidate),
            "--output_dir", str(group_output),
            "--config", str(config),
            "--scenario_ids", scenario,
            "--top_k", str(args.top_k),
            "--num_workers", str(args.num_workers),
            "--timeout", str(args.timeout),
        ]
        if args.render:
            command.append("--render")
        if args.dry_run:
            command.append("--dry_run")
        commands.append(command)

    print(
        f"[POOL REBENCHMARK] groups={len(commands)} top_k={args.top_k} "
        f"maximum_rollouts={len(commands) * args.top_k}"
    )
    for index, command in enumerate(commands, start=1):
        print(f"[GROUP {index}/{len(commands)}] {' '.join(command)}", flush=True)
        subprocess.run(command, check=True)
    print(f"[DONE] {output}")


if __name__ == "__main__":
    main()
