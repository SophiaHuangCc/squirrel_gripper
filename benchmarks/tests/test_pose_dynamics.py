import unittest

import torch

from dynamics.pose_targets import (
    POSE_OUTPUT_DIM, pose_geometric_metrics, pose_joint_angles_deg,
)


class PoseDynamicsTest(unittest.TestCase):
    def test_pose_metrics_are_finite_and_differentiable(self):
        theta = torch.linspace(0.0, torch.pi, 5)
        pose = torch.stack((0.25 * torch.cos(theta), 0.25 * torch.sin(theta)), dim=-1)
        pose = pose.reshape(1, POSE_OUTPUT_DIM).requires_grad_(True)
        task = torch.tensor([[0.0, 0.0, 0.5]])  # radius=.025 m = .25 pose units
        metrics = pose_geometric_metrics(pose, task)
        self.assertEqual(tuple(metrics.shape), (1, 3))
        self.assertTrue(torch.isfinite(metrics).all())
        metrics.sum().backward()
        self.assertTrue(torch.isfinite(pose.grad).all())

    def test_joint_angles_have_expected_shapes(self):
        pose = torch.tensor([[0., 0., 1., 0., 1., 1., 0., 1., 0., 2.]])
        headings, bends = pose_joint_angles_deg(pose)
        self.assertEqual(tuple(headings.shape), (1, 4))
        self.assertEqual(tuple(bends.shape), (1, 3))


if __name__ == "__main__":
    unittest.main()
