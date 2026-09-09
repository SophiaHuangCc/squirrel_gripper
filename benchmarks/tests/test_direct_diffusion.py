"""Diffusion proposals must reach simulation without overgeneration or ranking."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from benchmarks.baselines.diffusion_search import diffusion_search
from benchmarks.candidates import load_candidates, save_candidates
from benchmarks.protocol import load_config
from generator.dataloader import DesignBounds, diffusion_to_physical, project_physical_design


class RecordingSampler:
    def __init__(self):
        self.calls = []
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=15)

    def sample(self, cond, **kwargs):
        offset = sum(len(c[0]) for c in self.calls)
        self.calls.append((cond, kwargs))
        unit = torch.zeros(len(cond), 16)
        unit[:, 0] = torch.arange(offset, offset+len(cond)) / 10
        bounds = DesignBounds.defaults()
        return {"design_physical": project_physical_design(diffusion_to_physical(unit, bounds), bounds)}


class DirectDiffusionTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config(Path(__file__).resolve().parents[1] / "scenarios_v5_robust_four.json")

    def test_exact_count_generation_order_and_no_ranking(self):
        sampler = RecordingSampler()
        with patch("benchmarks.baselines.surrogate_search.rank_designs", side_effect=AssertionError("Ranking forbidden")):
            result = diffusion_search(sampler, self.config, 5, seed=0, batch_size=2,
                                      num_inference_steps=5, scenario_id="approach_radius:00")
        self.assertEqual([len(c[0]) for c in sampler.calls], [2, 2, 1])
        self.assertEqual(len(result.designs), 5)
        self.assertIsNone(result.scores)
        self.assertEqual(result.model_evaluations, 25)
        self.assertEqual(len(result.target_scenario_ids), 5)
        self.assertTrue(all(c[1]["dynamics_model"] is None for c in sampler.calls))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidates.npz"
            save_candidates(path, result.designs, "conditional_diffusion", scores=result.scores)
            loaded = load_candidates(path)
            self.assertIsNone(loaded["selection_scores"])
            self.assertTrue((loaded["design_params"] == result.designs).all())

    def test_guidance_sees_actual_conditions_and_counts_its_extra_work(self):
        sampler = RecordingSampler()
        guidance = type("Guidance", (), {"num_train_timesteps": 15})()
        result = diffusion_search(sampler, self.config, 4, seed=0, batch_size=4,
            num_inference_steps=5, generalist=True, guidance_scale=1.,
            guidance_dynamics_model=guidance, guidance_timesteps=(0, 3, 6))
        cond, kwargs = sampler.calls[0]
        self.assertAlmostEqual(float(cond[0, 2]), .025/.05)
        self.assertEqual(kwargs["guidance_task_params"].shape, (20, 3))
        self.assertIs(kwargs["dynamics_model"], guidance)
        self.assertEqual(result.model_evaluations, 4*5 + 4*3*20)
        self.assertIsNone(result.scores)


if __name__ == "__main__":
    unittest.main()
