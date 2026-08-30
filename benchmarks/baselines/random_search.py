"""Feasible uniform random candidate baseline."""

import argparse
from pathlib import Path

import numpy as np

from benchmarks.candidates import save_candidates
from generator.dataloader import DesignBounds


GEOMETRIES = np.asarray(
    [
        [0.066, 0.020, 0.024, 0.030],
        [0.040, 0.015, 0.015, 0.020], [0.048, 0.012, 0.013, 0.017],
        [0.035, 0.015, 0.018, 0.022], [0.055, 0.025, 0.030, 0.030],
        [0.075, 0.018, 0.022, 0.025], [0.058, 0.018, 0.029, 0.035],
        [0.075, 0.035, 0.040, 0.040], [0.100, 0.025, 0.030, 0.035],
        [0.075, 0.025, 0.040, 0.050], [0.090, 0.045, 0.050, 0.055],
        [0.130, 0.030, 0.035, 0.045], [0.085, 0.035, 0.055, 0.065],
    ],
    dtype=np.float32,
)


def sample_feasible_designs(num_candidates, seed):
    rng = np.random.default_rng(seed)
    bounds = DesignBounds.defaults()
    lo = bounds.lo.numpy()
    hi = bounds.hi.numpy()
    designs = []
    for _ in range(num_candidates):
        joint_stiffness = np.exp(rng.uniform(np.log(lo[:3]), np.log(hi[:3])))
        joint_lengths = lo[7:10].copy()
        links = GEOMETRIES[int(rng.integers(0, len(GEOMETRIES)))].copy()
        base_length = float(links.sum() + joint_lengths.sum())
        design = np.concatenate(
            [
                joint_stiffness,
                links,
                joint_lengths,
                [
                    lo[10],
                    rng.uniform(lo[11], hi[11]),
                    base_length,
                    rng.uniform(lo[13], hi[13]),
                    lo[14],
                    lo[15],
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
