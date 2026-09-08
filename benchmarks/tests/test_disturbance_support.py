import unittest
import numpy as np
from TendonForces.disturbance_metrics import (
    directional_contact_support_score, directional_projected_force_stats,
)


class DirectionalContactSupportTest(unittest.TestCase):
    def test_no_contacts_score_zero(self):
        self.assertEqual(directional_contact_support_score([], [1, 0, 0]), 0.0)

    def test_opposing_normal_supports_direction(self):
        score = directional_contact_support_score([[-1, 0, 0]], [1, 0, 0])
        self.assertAlmostEqual(score, 1.0)

    def test_surrounding_normals_support_fixed_directions(self):
        normals = np.array([[1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1]])
        for disturbance in ([1, 0, 0], [-1, 0, 0], [0, 0, -1]):
            self.assertAlmostEqual(
                directional_contact_support_score(normals, disturbance), 1.0
            )

    def test_one_sided_contact_does_not_support_required_reaction(self):
        self.assertAlmostEqual(
            directional_contact_support_score([[1, 0, 0]], [1, 0, 0]), 0.0
        )

    def test_projected_force_preserves_useful_components(self):
        forces = np.array([[2.0, 0.0, 1.0], [2.0, 0.0, -1.0]])
        stats = directional_projected_force_stats(forces, [-1.0, 0.0, 0.0], 4.0)
        self.assertAlmostEqual(stats["opposing_force"], 4.0)
        self.assertAlmostEqual(stats["mean_opposing_force"], 2.0)
        self.assertAlmostEqual(stats["force_score"], 0.5)


if __name__ == "__main__":
    unittest.main()
