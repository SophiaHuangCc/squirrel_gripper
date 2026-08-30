"""Feasible uniform random candidate baseline."""

import argparse
from pathlib import Path

import numpy as np

from benchmarks.candidates import save_candidates


def sample_feasible_designs(num_candidates, seed):
    rng = np.random.default_rng(seed)
    designs = []
    for _ in range(num_candidates):
        joint_stiffness = np.exp(rng.uniform(np.log(5e-4), np.log(5e-2), size=3))
        joint_lengths = rng.uniform(0.005, 0.030, size=3)
        base_length = rng.uniform(max(0.15, joint_lengths.sum() + 0.04), 0.30)
        available = base_length - joint_lengths.sum()
        residual = available - 4 * 0.01
        links = 0.01 + rng.dirichlet(np.ones(4)) * residual
        design = np.concatenate(
            [
                joint_stiffness,
                links,
                joint_lengths,
                [
                    rng.uniform(0.01025, 0.013),
                    base_length,
                    rng.uniform(1.0, 30.0),
                    rng.uniform(0.015, 0.035),
                    rng.uniform(300.0, 700.0),
                ],
            ]
        )
        designs.append(design)
    return np.asarray(designs, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Generate feasible random From Links candidates.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num_candidates", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    designs = sample_feasible_designs(args.num_candidates, args.seed)
    save_candidates(args.output, designs, method="random", seed=args.seed)
    print(args.output.resolve())


if __name__ == "__main__":
    main()

