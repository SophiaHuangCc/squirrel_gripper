import unittest
import numpy as np
from benchmarks.design_spread import build_spread, geometry_stats


def examples(conditions=2):
    rows=[]
    for scene, scores in [('s0', [.8,.4]), ('s1',[.6,.2])]:
        for candidate, score in enumerate(scores):
            for condition in range(conditions):
                rows.append(dict(objective='combined',method='dgdm',scenario_id=f'{scene}@{condition}',
                    seed=0,candidate_id=str(candidate),status='ok',simulator_utility=score,
                    num_contacts=10*score,angular_span_deg=100*score,design_params=[.2+.4*candidate,.5]))
    return rows


class DesignSpreadTest(unittest.TestCase):
    def test_winners_and_pool_have_distinct_spread(self):
        r=build_spread(examples(),np.zeros(2),np.ones(2))['summary'][0]
        self.assertAlmostEqual(r['selected_mean_utility'],.7)
        self.assertAlmostEqual(r['selected_std_utility'],.1)
        self.assertAlmostEqual(r['pool_mean_utility'],.5)
        self.assertAlmostEqual(r['pool_min_utility'],.2)
        self.assertEqual(r['complete_designs'],4)

    def test_repeated_initial_conditions_do_not_inflate_diversity_or_count(self):
        a=build_spread(examples(2),np.zeros(2),np.ones(2))['summary'][0]
        b=build_spread(examples(5),np.zeros(2),np.ones(2))['summary'][0]
        self.assertEqual(a['design_diversity_rms'],b['design_diversity_rms'])
        self.assertEqual(a['complete_designs'],b['complete_designs'])
        self.assertEqual(a['pool_std_utility'],b['pool_std_utility'])

    def test_failed_candidate_is_counted_and_excluded_from_selection(self):
        rows=examples()
        rows[0].update(status='timeout',simulator_utility=None)
        r=build_spread(rows,np.zeros(2),np.ones(2))['summary'][0]
        self.assertEqual(r['incomplete_designs'],1)
        self.assertEqual(r['unsuccessful_or_invalid_trials'],1)
        self.assertAlmostEqual(r['selected_mean_utility'],.5)

    def test_geometry_is_scale_normalized_and_ignores_fixed_dimensions(self):
        lo=np.array([0.,0.,3.]);hi=np.array([1.,100.,3.])
        a=geometry_stats([[.2,20.,3.],[.6,60.,3.]],lo,hi)
        self.assertAlmostEqual(a['pairwise_rms'],.4)
        identical=geometry_stats([[.2,20.,3.],[.2,20.,3.]],lo,hi)
        self.assertEqual(identical['pairwise_rms'],0)
        self.assertEqual(identical['near_duplicate_pair_fraction'],1)
        self.assertIsNone(geometry_stats([[.2,20.,3.]],lo,hi)['pairwise_rms'])


if __name__=='__main__':unittest.main()
