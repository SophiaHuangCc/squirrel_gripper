"""Condition masking must hold in training/sampling without disabling guidance."""
import unittest

import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from generator.diffusion import SquirrelDesignDiffusion


class TinyDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.1))

    def forward(self, sample, timestep, global_cond):
        return self.weight * sample + global_cond.sum(-1)[:, None, None]


class TinyDynamics(torch.nn.Module):
    noise_conditioned = True
    design_coordinates = "diffusion_unit"
    num_train_timesteps = 15
    target_representation = "metrics"

    def forward(self, task, design, init, timestep):
        return .5 + .1 * design[:, :3]


class UnconditionalDiffusionTest(unittest.TestCase):
    def setUp(self):
        self.model = SquirrelDesignDiffusion(TinyDenoiser(), DDIMScheduler(
            num_train_timesteps=15, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon"),
            num_inference_steps=5, conditioning_mode="unconditional")

    def test_training_loss_ignores_conditions(self):
        batch = dict(design_unit=torch.zeros(2, 16, 1), cond=torch.zeros(2, 9))
        torch.manual_seed(31)
        first = self.model.training_loss(batch)
        batch["cond"] = torch.ones(2, 9)
        torch.manual_seed(31)
        second = self.model.training_loss(batch)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        second.backward()
        self.assertTrue(torch.isfinite(self.model.noise_pred_net.weight.grad))

    def test_context_only_keeps_environment_but_masks_targets_in_training(self):
        self.model.conditioning_mode = "context_only"
        condition = torch.arange(18, dtype=torch.float32).reshape(2, 9) / 100
        original = condition.clone()
        masked = self.model._network_condition(condition)
        torch.testing.assert_close(masked[:, :6], original[:, :6])
        self.assertTrue((masked[:, 6:] == 0).all())
        torch.testing.assert_close(condition, original)
        def loss(c):
            torch.manual_seed(31)
            return self.model.training_loss(dict(design_unit=torch.zeros(2, 16, 1), cond=c))
        changed_targets = condition.clone()
        changed_targets[:, 6:] = .8
        torch.testing.assert_close(loss(condition), loss(changed_targets), rtol=0, atol=0)
        changed_environment = condition.clone()
        changed_environment[:, 0] += .2
        self.assertNotEqual(float(loss(condition)), float(loss(changed_environment)))

    def test_context_only_sampling_ignores_requested_performance(self):
        self.model.conditioning_mode = "context_only"
        a = torch.zeros(2, 9)
        b = a.clone()
        b[:, 6:] = .8
        def sample(c):
            return self.model.sample(c, generator=torch.Generator().manual_seed(31))["design_physical"]
        torch.testing.assert_close(sample(a), sample(b), rtol=0, atol=0)

    def test_sampling_ignores_conditions_but_guidance_changes_designs(self):
        def sample(cond, guided=False):
            return self.model.sample(cond, generator=torch.Generator().manual_seed(31),
                dynamics_model=TinyDynamics() if guided else None,
                guidance_scale=1. if guided else 0., guidance_objective="contact"
            )["design_physical"]
        first = sample(torch.zeros(2, 9))
        second = sample(torch.ones(2, 9))
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        guided = sample(torch.ones(2, 9), True)
        self.assertTrue(torch.isfinite(guided).all())
        self.assertFalse(torch.equal(first, guided))


if __name__ == "__main__":
    unittest.main()
