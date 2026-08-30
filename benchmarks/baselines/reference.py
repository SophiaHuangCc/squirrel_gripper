"""Manufactured run.sh finger baseline."""

import argparse
from pathlib import Path

import numpy as np

from benchmarks.candidates import save_candidates


def reference_design():
    base_e = 6.74e6
    joint_e = np.asarray([0.10, 0.08, 0.06], dtype=np.float32) * 1e6
    return np.asarray(
        [
            *(joint_e / base_e),
            0.066, 0.020, 0.024, 0.030,
            0.020, 0.020, 0.020,
            0.010,
            0.200,
            14.7,
            0.030,
            500.0,
        ],
        dtype=np.float32,
    )


def main():
    parser = argparse.ArgumentParser(description="Write the manufactured reference candidate.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    save_candidates(
        args.output, reference_design(), method="reference", seed=0,
        candidate_ids=["manufactured_runsh"],
        metadata={"description": "6.6/2/2.4/3 cm links, 2 cm joints, 0.1/0.08/0.06 MPa joint E"},
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()

