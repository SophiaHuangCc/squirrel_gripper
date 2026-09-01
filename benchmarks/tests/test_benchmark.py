import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from benchmarks.baselines.random_search import sample_feasible_designs
from benchmarks.baselines.reference import reference_design
from benchmarks.baselines.surrogate_search import _candidate_scores, adam_search, select_target_cells
from benchmarks.candidates import load_candidates, save_candidates, validate_designs
from benchmarks.protocol import aggregate_records, expand_core_scenarios, load_config
from benchmarks.run_guidance_sweep import method_name, parse_float_list
from dynamics.trainer import Trainer
from generator.dataloader import (
    DESIGN_MODEL_SCALES, DesignBounds, model_norm_to_physical,
    project_physical_design,
)


class BenchmarkTests(unittest.TestCase):
    def test_guidance_sweep_labels_are_distinct_and_parseable(self):
        scales = parse_float_list("0,0.1,1,2,10")
        self.assertEqual(scales, (0.0, 0.1, 1.0, 2.0, 10.0))
        self.assertEqual(len({method_name(scale) for scale in scales}), len(scales))
        self.assertEqual(method_name(0.1), "conditional_dgdm_gs0p1")

    def test_scenario_count_and_families(self):
        config = load_config()
        cells = expand_core_scenarios(config)
        self.assertEqual(len(cells), 25)
        counts = {family: sum(cell["family"] == family for cell in cells) for family in {c["family"] for c in cells}}
        self.assertEqual(counts, {"approach_radius": 25})
        self.assertEqual(
            [cell["params"]["approach_deg"] for cell in cells[::5]],
            [5.0, 25.0, 45.0, 65.0, 85.0],
        )
        self.assertEqual(
            [cell["params"]["cyl_rad"] for cell in cells[:5]],
            [0.015, 0.020, 0.025, 0.030, 0.035],
        )
        self.assertEqual(
            config["evaluation"]["utility_weights"],
            {
                "disturbance_resistance_score": 0.55,
                "contact_coverage_norm": 0.35,
                "angular_span_norm": 0.10,
            },
        )

    def test_compact_scenario_grid_is_center_and_boundaries(self):
        compact = load_config(Path(__file__).parents[1] / "scenarios_v2_compact.json")
        cells = expand_core_scenarios(compact)
        self.assertEqual(len(cells), 9)
        self.assertEqual(compact["default_target_scenario_id"], "approach_radius:04")

    def test_reference_is_valid_from_links_design(self):
        design = validate_designs(reference_design())
        self.assertEqual(design.shape, (1, 16))
        self.assertAlmostEqual(float(design[0, 3:10].sum()), float(design[0, 12]), places=6)

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
            self.assertEqual(loaded["design_params"].shape, (2, 16))
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

    def test_raw_angular_span_is_reported_above_utility_cap(self):
        summary = aggregate_records(
            [{
                "family": "approach_radius",
                "metrics": {
                    "disturbance_resistance_score": 0.5,
                    "num_contacts": 20,
                    "angular_span": 240.0,
                    "n_elements": 100,
                },
            }],
            load_config(),
        )
        self.assertEqual(summary["component_mean"]["angular_span_norm"], 1.0)
        self.assertEqual(summary["raw_metric_mean"]["angular_span_deg"], 240.0)

    def test_target_selection_distinguishes_specialist_and_generalist(self):
        config = load_config()
        default = select_target_cells(config)
        self.assertEqual(len(default), 1)
        self.assertEqual(default[0]["scenario_id"], "approach_radius:12")
        self.assertEqual(len(select_target_cells(config, family="approach_radius")), 25)
        self.assertEqual(len(select_target_cells(config, generalist=True)), 25)

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
        self.assertEqual(result.designs.shape, (3, 16))
        self.assertTrue(np.all(result.scores[:-1] >= result.scores[1:]))
        self.assertEqual(result.model_evaluations, 6)

    def test_surrogate_scores_obey_normalized_metric_bounds(self):
        class OutOfRangeSurrogate(torch.nn.Module):
            def forward(self, task_params, design_params, init_config, timesteps):
                return torch.tensor([[2.0, -1.0, 0.5]], dtype=torch.float32).repeat(
                    design_params.shape[0], 1
                )

        score = _candidate_scores(
            OutOfRangeSurrogate(), torch.zeros(1, 16), torch.zeros(1, 3),
            torch.zeros(1, 3), {
                "contact_coverage_norm": 1.0,
                "disturbance_resistance_score": 0.0,
                "angular_span_norm": 0.0,
            },
        )
        self.assertAlmostEqual(float(score.item()), 1.0)

    def test_standard_dynamics_trainer_does_not_require_diffusers(self):
        args = SimpleNamespace(
            device="cpu", task_dim=3, design_dim=16, init_dim=3,
            output_dim=3, hidden_dim=16, lr=1e-3, use_design_noise=False,
        )
        trainer = Trainer(args)
        trainer.create_model()
        self.assertIsNone(trainer.noise_scheduler)

    def test_geometry_projection_preserves_each_link_bound(self):
        bounds = DesignBounds.defaults()
        raw = bounds.hi.repeat(3, 1)
        raw[:, 3:7] = torch.tensor([
            [1.0, -1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0, -1.0],
        ])
        projected = project_physical_design(raw, bounds)
        self.assertTrue(torch.all(projected[:, 3:7] >= bounds.lo[3:7] - 1e-6))
        self.assertTrue(torch.all(projected[:, 3:7] <= bounds.hi[3:7] + 1e-6))
        np.testing.assert_allclose(
            projected[:, 3:10].sum(dim=1).numpy(),
            projected[:, 12].numpy(), atol=1e-5,
        )

    def test_shared_design_conversion_includes_base_thickness(self):
        physical = model_norm_to_physical(torch.ones(16))
        self.assertEqual(tuple(physical.shape), (16,))
        self.assertAlmostEqual(float(physical[11]), float(DESIGN_MODEL_SCALES[11]))

    def test_tendon_path_length_uses_routing_points_and_distal_anchor(self):
        try:
            from TendonForces.TendonForces import TendonForces
        except ModuleNotFoundError as exc:
            if exc.name == "elastica":
                self.skipTest("PyElastica is not installed in this test environment")
            raise
        positions = np.array([
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ])
        directors = np.repeat(np.eye(3)[:, :, None], 3, axis=2)
        length = TendonForces.get_path_length(
            positions, directors, np.array([1, 2]), np.array([0.0, 0.0, 0.1]), 3
        )
        self.assertAlmostEqual(float(length), 3.0)


if __name__ == "__main__":
    unittest.main()
