from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from genesis_core.intervention import orthogonal_recolor,recolor,center,diagnostics
from genesis_core import experiments


class ConstructionTests(unittest.TestCase):
    def test_algebra_and_covariance_only_control(self):
        rng=np.random.default_rng(8);z=rng.normal(size=(100,3));s=rng.normal(size=(100,5))+z@rng.normal(size=(3,5));target=np.arange(1.,6.)
        transformed,energy=orthogonal_recolor(z,s,target)
        np.testing.assert_allclose(center(z).T@transformed,0,atol=1e-11)
        np.testing.assert_allclose(transformed.T@transformed/99,np.diag(target),atol=1e-11)
        self.assertAlmostEqual(energy['projection_energy_fraction']+energy['residual_energy_fraction'],1,places=12)
        self.assertGreater(diagnostics(z,recolor(s,target),target)['cross_moment_max_abs'],.01)

    def test_singular_and_near_singular_rejected(self):
        rng=np.random.default_rng(9);z=rng.normal(size=(80,3));s=rng.normal(size=(80,5))
        for scale in (0,1e-8):
            bad=s.copy();bad[:,-1]*=scale
            with self.assertRaises(ValueError):orthogonal_recolor(z,bad,np.ones(5))
        with self.assertRaises(ValueError):orthogonal_recolor(np.ones((80,3)),s,np.ones(5))
        with self.assertRaises(ValueError):orthogonal_recolor(z[:6],s[:6],np.ones(5))

    def test_all_historical_shared_components(self):
        result=experiments.construction()
        self.assertEqual(len(result['cohorts']),16)
        self.assertEqual({r['n'] for r in result['cohorts']},{512,768})
        self.assertLess(result['maximum_historical_scalar_error'],1e-9)
        self.assertTrue(all(r['variants']['cross_degree']['covariance_max_abs_error']<5e-11 for r in result['cohorts']))

    def test_all_paired_records_and_distinct_n256_branch(self):
        r=experiments.historical_records()
        self.assertEqual(r['matched_readout_exact_comparisons'],330)
        self.assertEqual(r['cell_ablation_exact_paired_deltas'],120)
        self.assertEqual(len(r['cross_degree_between_cohort_RSA']),12)
        self.assertTrue(all(x['RSA_mean_change']<0 for x in r['cross_degree_between_cohort_RSA']))
        a,b=[x for x in r['size_specific_validation'] if x['n']==256]
        self.assertEqual((a['all_repair_pass_cohorts'],b['all_repair_pass_cohorts']),(6,7))
        self.assertTrue(r['matrix_parity_pending'])


if __name__=='__main__':unittest.main()
