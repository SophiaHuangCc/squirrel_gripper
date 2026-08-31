import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from generator.dataloader import DesignBounds
from pareto_diffusion.core import (
    build_preference_pairs, crowding_distance, hypervolume,
    inverted_generational_distance, non_dominated_sort, summarize_front,
)
from pareto_diffusion.data import ObjectiveTable, PreferencePairDataset, load_table, save_table, split_for_id
from pareto_diffusion.model import PreferenceClassifier
from pareto_diffusion.sampler import preference_scores


class ParetoCoreTests(unittest.TestCase):
    def test_sort_and_crowding(self):
        values=np.asarray([[1,0,0],[0,1,0],[0,0,1],[.2,.2,.2],[.1,.1,.1]],float)
        ranks=non_dominated_sort(values)
        np.testing.assert_array_equal(ranks,np.asarray([0,0,0,0,1]))
        crowd=crowding_distance(values,ranks)
        self.assertTrue(np.isinf(crowd[:3]).all())
        pairs=build_preference_pairs(ranks,crowd,seed=4)
        self.assertTrue(set(pairs[:,2])=={0.0,1.0})

    def test_constraints_rank_feasible_first(self):
        values=np.asarray([[.1,.1,.1],[1,1,1]])
        ranks=non_dominated_sort(values,feasible=[True,False],violation=[0,.2])
        np.testing.assert_array_equal(ranks,[0,1])

    def test_hypervolume_and_igd(self):
        values=np.asarray([[1,.5,.5],[.5,1,.5]])
        self.assertAlmostEqual(hypervolume(values),.375)
        self.assertEqual(inverted_generational_distance(values,values),0.0)
        summary,ids=summarize_front(values)
        self.assertEqual(summary["num_non_dominated"],2); self.assertEqual(len(ids),2)

    def test_design_disjoint_split_is_stable(self):
        self.assertEqual(split_for_id("abc",seed=7),split_for_id("abc",seed=7))

    def test_table_round_trip_and_pairs(self):
        bounds=DesignBounds.defaults(); designs=[]
        for fraction in (.2,.4,.6,.8):
            design=(bounds.lo+fraction*(bounds.hi-bounds.lo)).numpy()
            # Restore geometry constraint for realism; pair dataset only needs bounds here.
            designs.append(design)
        table=ObjectiveTable(np.stack(designs),np.asarray([[.1,.8,.2],[.4,.5,.3],[.7,.2,.4],[.3,.3,.9]]),
            np.ones(4,bool),np.zeros(4),np.asarray(["a","b","c","d"]),np.asarray(["train"]*4),np.ones(4),{})
        with tempfile.TemporaryDirectory() as directory:
            path=save_table(Path(directory)/"table.npz",table); loaded=load_table(path)
            dataset=PreferencePairDataset(loaded,"train",bounds,max_pairs=20)
            self.assertGreater(len(dataset),0); self.assertEqual(dataset[0]["design_a"].shape,(16,))

    def test_classifier_is_antisymmetric_without_dropout(self):
        model=PreferenceClassifier(16,width=32,time_dim=16,dropout=0)
        a,b=torch.randn(5,16),torch.randn(5,16); t=torch.arange(5)
        torch.testing.assert_close(model(a,b,t),-model(b,a,t))
        score=preference_scores(model,a,b[:3],torch.tensor(2)); self.assertEqual(score.shape,(5,))


if __name__=="__main__": unittest.main()
