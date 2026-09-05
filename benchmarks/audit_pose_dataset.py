"""Check whether a simulator dataset contains valid pose-supervision fields."""

import argparse
from pathlib import Path
import numpy as np

from dynamics.pose_targets import pose_target_from_npz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--max_errors", type=int, default=20)
    args = parser.parse_args()
    failed = []
    failure_count = 0
    total = 0
    for root in args.paths:
        files = sorted(root.rglob("*.npz"))
        for path in files:
            total += 1
            try:
                with np.load(path, allow_pickle=True) as data:
                    target = pose_target_from_npz(data)
                if target.shape != (10,):
                    raise ValueError(f"target shape {target.shape}")
            except Exception as exc:
                failure_count += 1
                if len(failed) < args.max_errors:
                    failed.append((path, str(exc)))
    print(f"[POSE DATA AUDIT] total={total} valid={total-failure_count} failures={failure_count}")
    for path, error in failed:
        print(f"[INVALID] {path}: {error}")
    if failure_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
