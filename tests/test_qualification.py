from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from genesis_core import entropy,experiments
from genesis_core.data import load
from genesis_core.intervals import check_cover,inspect_leaves,inspect_ridge
from genesis_core.qualification import saved_bounds,check_capture


class QualificationTests(unittest.TestCase):
    def test_exact_moment_identity_and_log_sqrt_controls(self):
        self.assertEqual(entropy.moment_identity_controls()['enumerated_fixture_samples'],117)
        self.assertEqual(entropy.sqrt_box(F(9,4)),(F(3,2),F(3,2)))
        lo,hi=entropy.log_box(F(2));self.assertTrue(lo<hi)
        with self.assertRaises(AssertionError):entropy.log_box(0)

    def test_all_laws_keep_separate_scope(self):
        result=experiments.entropy_bounds(True)
        self.assertEqual(len(result['bounds']),14)
        self.assertTrue(all(r['log_margin']>0 for r in result['bounds']))
        self.assertEqual(sum(r['role']=='FROZEN_SELECTED_MODEL_BOUND' for r in result['bounds']),2)

    def test_full_saved_coverage_and_svd(self):
        result=saved_bounds()
        self.assertEqual(result['construction_interval']['cover']['cells'],2048)
        self.assertEqual(result['ridge']['fold_cell_bounds'],96)
        self.assertEqual(sum(r['values_checked'] for r in result['captured_svd']),3072)
        self.assertGreater(min(r['target_to_bound_ratio'] for r in result['captured_svd']),26000)
        self.assertFalse(result['whole_interval_SVD_solver_certified'])

    def test_interval_gap_overlap_and_missing_leaf_rejected(self):
        for cells in ([[.015,.017],[.018,.02]],[[.015,.018],[.017,.02]],[[.016,.02]]):
            with self.assertRaises(ValueError):check_cover(cells)
        data=load('qualification')
        with self.assertRaises(ValueError):inspect_leaves(data['f1_leaves'][1:])
        bad=deepcopy(data['ridge_cells']);bad[0]['folds'].pop()
        with self.assertRaises(ValueError):inspect_ridge(bad)

    def test_svd_mutations_rejected(self):
        original=load('qualification')['captures'][0]
        edits=[lambda r:r['values_hex'].pop(),
               lambda r:r['saved']['relative_upper'].update(numerator='0'),
               lambda r:r['saved'].update(values_checked=210),
               lambda r:r['saved']['input_error'].update(numerator='0'),
               lambda r:r['boxes'][0]['center_root_upper'].update(numerator='0'),
               lambda r:r['saved']['ideal_norm_root_lower'].update(numerator='9'*120)]
        for edit in edits:
            modified=deepcopy(original);edit(modified)
            with self.assertRaises(ValueError):check_capture(modified)


if __name__=='__main__':unittest.main()
