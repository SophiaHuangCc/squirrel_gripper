import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.compare_conditioning_runs import main


class ConditioningComparisonTest(unittest.TestCase):
    def test_guidance_gain_is_separate_from_absolute_prior_quality(self):
        def collected(root, mode):
            base, gain = (.6, .01) if mode == 'conditional' else (.4, .03)
            manifests = {method: dict(seed=0, num_candidates=16, num_scenarios=5,
                num_rollouts=80, proposal_metadata={}, jobs=[])
                for method in ('conditional_diffusion', 'dgdm')}
            values = {'conditional_diffusion': [base]*16, 'dgdm': [base+gain]*16}
            rows = [dict(prior=mode, guidance_scale=scale, mean_v20_utility=base+scale*gain,
                         std_design_mean_utility=0.,p10_design_mean_utility=base+scale*gain,
                         min_design_mean_utility=base+scale*gain,design_diversity_rms=.1)
                    for scale in (0, 1)]
            return {}, manifests, rows, values
        with tempfile.TemporaryDirectory() as directory:
            with patch('benchmarks.compare_conditioning_runs.collect', side_effect=collected), \
                 patch('sys.argv', ['compare', '--context', directory]), \
                 patch('builtins.print'):
                main()
            report = json.loads((Path(directory)/'conditioning_comparison.json').read_text())
            self.assertAlmostEqual(report['difference_in_guidance_gain'], .02)
            self.assertAlmostEqual(report['rows'][3]['mean_v20_utility'], .43)
            self.assertEqual(report['guidance_effect']['context_only']['dgdm_wins'], 16)


if __name__ == '__main__':
    unittest.main()
