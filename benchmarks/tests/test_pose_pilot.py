import unittest
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import torch
from benchmarks.pose_pilot import change_stats, pose_loss
from benchmarks.grasp_screen import screen, DIRECTIONS


class PosePilotTest(unittest.TestCase):
    def test_local_response_detects_reversed_and_unresolved_changes(self):
        delta = np.array([.001, .002])
        self.assertAlmostEqual(change_stats(delta, -delta, 1e-5)["change_cosine"], -1)
        self.assertIsNone(change_stats(delta, np.zeros(2), 1e-5)["change_cosine"])

    def test_pose_loss_descent_on_analytic_radial_model(self):
        # Independent analytic simulator p(d)=(d,0) for all keypoints.
        d = torch.tensor(.04, requires_grad=True)
        def simulate(value):
            return torch.stack((value, value*0)).repeat(5)
        before = pose_loss(simulate(d), .02, .01, 1.)
        grad = torch.autograd.grad(before.sum(), d)[0]
        self.assertLess(float(pose_loss(simulate(d-.1*grad), .02, .01, 1.)), float(before))
        self.assertGreater(float(pose_loss(simulate(d+.1*grad), .02, .01, 1.)), float(before))

    def test_weakest_direction_missing_and_failure_cannot_pass(self):
        t = dict(min_contacts=10, min_span_deg=180, min_direction_support=.9, min_opposing_force_n=5)
        m = dict(num_contacts=10, angular_span=180)
        for direction in DIRECTIONS:
            m[f"disturbance_{direction}_directional_support_score"] = 1.
            m[f"disturbance_{direction}_opposing_force"] = 5.
        r = dict(status="ok", metrics=m)
        self.assertTrue(screen(r,t)["pass"])
        m["disturbance_drag_down_opposing_force"] = 4.9
        self.assertFalse(screen(r,t)["pass"])
        del m["disturbance_drag_down_opposing_force"]
        self.assertIn("missing", screen(r,t)["reason"])
        self.assertFalse(screen(dict(status="timeout"), t)["pass"])

    def test_report_uses_nominal_selection_and_counts_missing_selected_condition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            records = [
                dict(candidate_id="winner", scenario_id="s@nominal", utility=1., status="ok"),
                dict(candidate_id="other", scenario_id="s@nominal", utility=.5, status="ok"),
                dict(candidate_id="other", scenario_id="s@offset", utility=1., status="ok"),
                dict(candidate_id="third", scenario_id="s@offset", status="timeout"),
            ]
            for i, record in enumerate(records):
                folder = source / str(i)
                folder.mkdir()
                (folder / "benchmark_result.json").write_text(json.dumps(dict(
                    method="conditional_diffusion", seed=0,
                    metrics=dict(num_contacts=10, angular_span=180), **record)))
            thresholds = root / "thresholds.json"
            thresholds.write_text(json.dumps(dict(profiles=dict(geometry=dict(min_contacts=10, min_span_deg=180)))))
            subprocess.run([sys.executable, "-m", "benchmarks.grasp_screen", "--source", str(source),
                "--thresholds", str(thresholds), "--output", str(root / "report")], check=True, capture_output=True)
            report = json.loads((root / "report/report.json").read_text())
            self.assertEqual(report["pool_summary"][0]["trial_pass_rate"], .75)
            self.assertIsNone(report["pool_summary"][0]["seed_bootstrap_95ci"])
            self.assertEqual(report["nominal_selected_perturbation_summary"][0]["trial_pass_rate"], 0.)
            self.assertEqual(report["selection_budgets"][0]["selected"], "winner")


if __name__ == "__main__":
    unittest.main()
