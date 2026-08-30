import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from benchmarks.baselines.random_search import sample_feasible_designs
from benchmarks.baselines.reference import reference_design
from benchmarks.baselines.surrogate_search import adam_search, select_target_cells
from benchmarks.candidates import load_candidates, save_candidates, validate_designs
from benchmarks.protocol import aggregate_records, expand_core_scenarios, load_config


class BenchmarkTests(unittest.TestCase):
    def test_scenario_count_and_families(self):
        cells = expand_core_scenarios(load_config())
        self.assertEqual(len(cells), 28)
        counts = {family: sum(cell["family"] == family for cell in cells) for family in {c["family"] for c in cells}}
        self.assertEqual(counts, {"nominal": 1, "orientation": 9, "branch_offset": 9, "landing_severity": 9})

    def test_reference_is_valid_from_links_design(self):
        design = validate_designs(reference_design())
        self.assertEqual(design.shape, (1, 15))
        self.assertAlmostEqual(float(design[0, 3:10].sum()), float(design[0, 11]), places=6)

    def test_random_designs_are_reproducible_and_feasible(self):
        first = sample_feasible_designs(16, seed=7)
        second = sample_feasible_designs(16, seed=7)
        np.testing.assert_array_equal(first, second)
        validate_designs(first)
        self.assertTrue(np.all(first[:, 3:7] >= 0.01))

    def test_candidate_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidates.npz"
            designs = sample_feasible_designs(3, seed=2)
            save_candidates(path, designs, method="random", seed=2)
            loaded = load_candidates(path, top_k=2)
            self.assertEqual(loaded["design_params"].shape, (2, 15))
            self.assertEqual(loaded["method"], "random")
            self.assertEqual(loaded["seed"], 2)

    def test_aggregation_penalizes_lower_tail(self):
        config = load_config()
        records = []
        for index, score in enumerate([0.2, 0.5, 0.8, 1.0, 1.0]):
            records.append(
                {
                    "family": "nominal",
                    "metrics": {
                        "disturbance_resistance_score": score,
                        "num_contacts": 100,
                        "angular_span": 180,
                        "n_elements": 100,
                    },
                }
            )
        summary = aggregate_records(records, config)
        self.assertLess(summary["cvar20_utility"], summary["mean_utility"])
        self.assertEqual(summary["num_rollouts"], 5)

    def test_target_selection_distinguishes_specialist_and_generalist(self):
        config = load_config()
        self.assertEqual(len(select_target_cells(config)), 1)
        self.assertEqual(len(select_target_cells(config, family="orientation")), 9)
        self.assertEqual(len(select_target_cells(config, generalist=True)), 28)

    def test_adam_search_returns_ranked_feasible_candidates(self):
        class SmoothFakeSurrogate(torch.nn.Module):
            def forward(self, task_params, design_params, init_config, timesteps):
                base = torch.sigmoid(design_params[:, 12:13])
                return torch.cat([base, 0.5 * base, 0.25 * base], dim=1)

        result = adam_search(
            SmoothFakeSurrogate(), load_config(), num_candidates=3, seed=4,
            num_steps=2, learning_rate=0.01, device="cpu",
        )
        validate_designs(result.designs)
        self.assertEqual(result.designs.shape, (3, 15))
        self.assertTrue(np.all(result.scores[:-1] >= result.scores[1:]))
        self.assertEqual(result.model_evaluations, 6)


if __name__ == "__main__":
    unittest.main()
