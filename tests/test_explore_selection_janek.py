"""E-sel (plan 4.4): tiny-constrained selection on A, matched-recall FP. Exploratory analysis only."""
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from mixab import explore_selection_janek as ex 
from mixab import reselect_split_janek as rs 
from tests.test_reselect_split_janek import SETTINGS, SOURCES, SPLIT, make_counts, save_npz


def model(tmp, name, alpha, counts):
    return dict(name=name, alpha=alpha, npz=save_npz(Path(tmp)/f'{name}.npz', counts))


class ConstrainedSelectionTest(unittest.TestCase):
    def test_constraint_excludes_high_score_low_tiny_setting(self):
        """Like v1: B-Clean wins the unconstrained R1 objective only at a setting with low tiny recall."""
        with TemporaryDirectory() as tmp:
            # B-Clean: best objective at setting 0 but tiny 0/2 there; setting 1 is feasible but weaker.
            bc = make_counts(lambda s, c, k: 9 if k == 0 else 6)
            bc[:, 1, 0, 4] = 0
            # B1: objective 7 everywhere, tiny 1/2 -> t_ref = 1/2.
            b1 = make_counts(lambda s, c, k: 7)
            # alpha 1/2: objective 8 at setting 2, tiny 1/2 (feasible).
            mid = make_counts(lambda s, c, k: 8 if k == 2 else 5)
            models = [model(tmp, 'B-Clean', '0', bc), model(tmp, 'merge_1_2', '1/2', mid), model(tmp, 'B1', '1', b1)]
            r1 = rs.run(models, {}, {}, Path(tmp)/'v1')['decision']
            self.assertEqual(r1['alpha_star'], '0')                      # unconstrained R1 picks B-Clean
            out = ex.analyse(models)
            d = out['decision']
            self.assertAlmostEqual(d['t_ref_tiny_A_B1'], .5)
            self.assertAlmostEqual(d['tiny_floor_A'], .47)
            self.assertEqual(out['selection']['B-Clean']['setting_key'], 'base/score_0.40')   # setting 0 excluded
            self.assertEqual(d['alpha_star'], '1/2')
            self.assertEqual(d['baseline_settings']['B-Clean'], 'base/score_0.30')          # B-Clean at its R1 setting
            self.assertAlmostEqual(d['mean_hmean_B_alpha_star'], .8)

    def test_margin_is_against_bclean_at_r1_and_b1_constrained(self):
        with TemporaryDirectory() as tmp:
            bc = make_counts(lambda s, c, k: 9 if k == 0 else 6)
            bc[:, 1, 0, 4] = 0
            b1 = make_counts(lambda s, c, k: 7)
            mid = make_counts(lambda s, c, k: 8 if k == 2 else 5)
            models = [model(tmp, 'B-Clean', '0', bc), model(tmp, 'merge_1_2', '1/2', mid), model(tmp, 'B1', '1', b1)]
            d = ex.analyse(models)['decision']
            self.assertAlmostEqual(d['baseline_mean_hmean_B']['B-Clean'], .9)
            self.assertAlmostEqual(d['baseline_mean_hmean_B']['B1'], .7)
            self.assertFalse(d['passes_margin_exploratory'])
            self.assertTrue(d['passes_tiny_exploratory'])

    def test_model_without_feasible_setting_is_excluded_and_ties_follow_r1(self):
        with TemporaryDirectory() as tmp:
            bc = make_counts(lambda s, c, k: 9)
            bc[..., 4] = 0                                              # never feasible
            b1 = make_counts(lambda s, c, k: 7)
            mid = make_counts(lambda s, c, k: 7)                         # ties B1 everywhere
            models = [model(tmp, 'B-Clean', '0', bc), model(tmp, 'merge_1_2', '1/2', mid), model(tmp, 'B1', '1', b1)]
            out = ex.analyse(models)
            self.assertIsNone(out['selection']['B-Clean']['setting_index'])
            self.assertEqual(out['selection']['B-Clean']['feasible_settings'], 0)
            self.assertEqual(out['decision']['alpha_star'], '1')        # tie: endpoint before interior alpha
            self.assertEqual(out['selection']['B1']['setting_key'], 'base/score_0.30')   # native tie-break

    def test_matched_recall_fp(self):
        with TemporaryDirectory() as tmp:
            b1 = make_counts(lambda s, c, k: 7)
            mid = make_counts(lambda s, c, k: 7 if k in (1, 3) else 5)   # recall .5 at k 0, 2
            mid[:, 1, 3, 2] = 8                                          # fewer predictions -> fewer FP at k 3
            bc = make_counts(lambda s, c, k: 6)
            models = [model(tmp, 'B-Clean', '0', bc), model(tmp, 'merge_1_2', '1/2', mid), model(tmp, 'B1', '1', b1)]
            m = ex.analyse(models)['matched_recall']
            self.assertAlmostEqual(m['recall_floor_A'], .7)
            self.assertIsNone(m['models']['B-Clean'])                   # never reaches B1's recall
            self.assertEqual(m['models']['merge_1_2']['setting'], 'width_3/score_0.40')
            self.assertEqual(m['models']['merge_1_2']['lr2_fp_A'], 1*sum(s == 'A' for s in SPLIT))

    def test_exact_floor_boundary(self):
        self.assertEqual(ex.tiny_lr2(np.array([[0, 0, 0, 0, 0], [0, 0, 0, 100, 47]])), Fraction(47, 100))
        self.assertIsNone(ex.tiny_lr2(np.zeros((2, 5), dtype=np.int64)))


if __name__ == '__main__':
    unittest.main()
