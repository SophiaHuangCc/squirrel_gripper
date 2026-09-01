import unittest
import torch

from dgdm.data import PROFILE_CHANNELS
from dgdm.guidance import ProfileTarget, ScenarioBatch, aggregate_profile_score
from dgdm.models import InteractionProfileModel, masked_profile_loss


class DGDMCoreTest(unittest.TestCase):
    def test_profile_gradient_aggregates_scenarios(self):
        model = InteractionProfileModel(profile_steps=8, channels=len(PROFILE_CHANNELS), width=16)
        designs = torch.randn(3, 16, requires_grad=True)
        scenarios = ScenarioBatch(torch.zeros(4, 8), torch.tensor([1.0, 2.0, 3.0, 4.0]))
        target = ProfileTarget.from_dict({"channels": {"contact_fraction": 0.5}}, steps=8)
        score = aggregate_profile_score(model, designs, scenarios, target)
        self.assertEqual(tuple(score.shape), (3,))
        score.sum().backward()
        self.assertTrue(torch.isfinite(designs.grad).all())
        self.assertGreater(designs.grad.abs().sum().item(), 0.0)

    def test_profile_model_accepts_diffusion_timestep(self):
        model = InteractionProfileModel(profile_steps=8, channels=len(PROFILE_CHANNELS), width=16)
        designs = torch.randn(3, 16, requires_grad=True)
        scenarios = ScenarioBatch(torch.zeros(2, 8))
        target = ProfileTarget.from_dict({"channels": {"contact_fraction": 0.5}}, steps=8)
        score = aggregate_profile_score(
            model, designs, scenarios, target, torch.tensor([0.0, 0.5, 0.99])
        )
        self.assertEqual(tuple(score.shape), (3,))
        score.sum().backward()
        self.assertTrue(torch.isfinite(designs.grad).all())

    def test_masked_loss_ignores_unobserved_values(self):
        pred = torch.tensor([[[2.0, 100.0]]])
        target = torch.zeros_like(pred)
        mask = torch.tensor([[[1.0, 0.0]]])
        self.assertEqual(masked_profile_loss(pred, target, mask).item(), 4.0)

    def test_unknown_target_channel_is_rejected(self):
        with self.assertRaises(ValueError):
            ProfileTarget.from_dict({"channels": {"not_a_signal": 1.0}}, steps=8)


if __name__ == "__main__":
    unittest.main()
