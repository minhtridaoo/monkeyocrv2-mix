"""Phase 0 (CS-B1) and the Kaggle latency helpers: CPU tests of the pure functions."""
from pathlib import Path
import unittest

import numpy as np

from mixab import csb1_latency_janek as lat 
from mixab import dualagree_janek as da 
from mixab import phase0_janek as p0 
from mixab.monkey_janek import monkey_root

ROOT = monkey_root(required=False)
BUNDLE = (ROOT or Path('.'))/'docs/handoff/artifacts/pr_quad_audit_v1'
HAVE_BUNDLE = (BUNDLE/'data/validation_remaining.jsonl.gz').is_file()
try:
    import cv2  # noqa: F401
    import shapely  # noqa: F401
    HAVE_GEOM = True
except ImportError:
    HAVE_GEOM = False


def quad(x0, y0, x1, y1):
    return [x0, y0, x1, y0, x1, y1, x0, y1]


class Phase0PureTest(unittest.TestCase):
    def test_net_contributions_sum_to_exact_delta_hmean(self):
        rng = np.random.default_rng(0)
        base = rng.integers(0, 20, (50, 3)); base[:, 1] += 20; base[:, 2] += base[:, 0]
        new = base.copy(); new[:, 0] += rng.integers(0, 3, 50); new[:, 2] += rng.integers(0, 4, 50)
        new[:, 0] = np.minimum(new[:, 0], np.minimum(new[:, 1], new[:, 2]))
        h_b = p0.hmean(base.sum(0))
        net = p0.net_contributions(new[:, 0] - base[:, 0], new[:, 2] - base[:, 2], h_b)
        denom = new[:, 1].sum() + new[:, 2].sum()
        self.assertAlmostEqual(sum(net)/denom, p0.hmean(new.sum(0)) - h_b, places=12)

    def test_top_share(self):
        self.assertAlmostEqual(p0.top_share([5, 3, 1, 1], k=1), .5)
        self.assertIsNone(p0.top_share([1, -2], k=1))

    def test_gt_diagnosis_follows_tuan_box_order(self):
        d = p0.gt_diagnosis
        self.assertEqual(d([.6], [.9], False, .5), 'ambiguous_matching')
        self.assertEqual(d([.3], [.6], False, .5), 'score_filtering')
        self.assertEqual(d([.3], [.2], False, .5), 'geometry')
        self.assertEqual(d([], [.2], False, .5), 'ambiguous_score_and_geometry')
        self.assertEqual(d([], [], True, .5), 'only_box_removed_by_roi_ignore')
        self.assertEqual(d([], [], False, .5), 'no_box_in_dump')

    def test_score_bucket_and_size_bin(self):
        self.assertEqual(p0.score_bucket(.55), '0.5-0.6')
        self.assertEqual(p0.score_bucket(.95), '>=0.9')
        self.assertEqual(p0.score_bucket(.75, (.8, .9), .7), '0.7-0.8')
        self.assertEqual(p0.size_bin(3.99), '<4')
        self.assertEqual(p0.size_bin(16), '>=16')

    def test_allocate_minimum_then_proportional_and_capped(self):
        q = p0.allocate({'a': 100, 'b': 10, 'c': 2, 'd': 0, 'e': 50}, 30, minimum=4)
        self.assertEqual(sum(q.values()), 30)
        self.assertEqual((q['c'], q['d']), (2, 0))
        self.assertGreaterEqual(q['b'], 4)
        self.assertGreater(q['a'], q['e'])
        q = p0.equal_allocate({'a': 1, 'b': 10, 'c': 2, 'd': 5}, 16)
        self.assertEqual(sum(q.values()), 16)
        self.assertEqual(q['a'], 1)

    def test_sample_strata_is_deterministic_and_weighted(self):
        cases = [dict(case_id=f'x{i}', s='A' if i < 30 else 'B') for i in range(40)]
        a = p0.sample_strata(cases, lambda c: c['s'], {'A': 3, 'B': 2})
        b = p0.sample_strata(list(reversed(cases)), lambda c: c['s'], {'A': 3, 'B': 2})
        self.assertEqual([c['case_id'] for c in a], [c['case_id'] for c in b])
        self.assertEqual(sum(c['stratum'] == 'A' for c in a), 3)
        w = {c['stratum']: c['population_weight'] for c in a}
        self.assertEqual(w, {'A': 10.0, 'B': 5.0})

    @unittest.skipUnless(HAVE_GEOM, 'needs cv2 + shapely')
    def test_short_edge_lr2(self):
        self.assertAlmostEqual(p0.short_edge_lr2(quad(0, 0, 40, 8), 100), 4.0)
        self.assertAlmostEqual(p0.short_edge_lr2(quad(0, 0, 40, 8), 101), 8*50/101)


class LatencyPureTest(unittest.TestCase):
    def test_select_sources_split_is_fixed_and_disjoint(self):
        ids = [f'{i:016x}' for i in range(500)]
        w, t = lat.select_sources(ids, 50, 10)
        w2, t2 = lat.select_sources(list(reversed(ids)), 50, 10)
        self.assertEqual((w, t), (w2, t2))
        self.assertEqual((len(w), len(t)), (10, 50))
        self.assertFalse(set(w) & set(t))

    def test_arm_order_alternates_and_mirrors(self):
        self.assertEqual(lat.arm_order(0, 0), ('B1', 'CSB1'))
        self.assertEqual(lat.arm_order(1, 0), ('CSB1', 'B1'))
        self.assertEqual(lat.arm_order(0, 1), ('CSB1', 'B1'))
        self.assertEqual(lat.arm_order(1, 1), ('B1', 'CSB1'))

    def test_paired_summary(self):
        b = [100, 110, 90, 105, 95]
        c = [190, 200, 180, 210, 185]
        r = lat.paired_summary(b, c, n_boot=200)
        self.assertAlmostEqual(r['ratio_of_medians'], 190/100)
        self.assertAlmostEqual(r['mean_extra_ms'], np.mean(np.array(c) - np.array(b)))
        self.assertLessEqual(r['ratio_of_medians_ci95'][0], r['ratio_of_medians'])

    @unittest.skipUnless(HAVE_GEOM, 'needs shapely')
    def test_fuse_thresholded_equals_dump_fusion(self):
        """Fusion on postprocessor-filtered boxes == dualagree.fuse_locked on the raw dump entries."""
        e_l = dict(polygons=[quad(0, 0, 10, 10), quad(20, 0, 30, 10), quad(50, 50, 60, 60)], raw=[.9, .55, .45], packed=[0]*3)
        e_c = dict(polygons=[quad(0, 0, 10, 11), quad(21, 0, 31, 10), quad(50, 50, 60, 60), quad(80, 0, 90, 10)],
                   raw=[.8, .65, .75, .71], packed=[0]*4)
        ref = da.fuse_locked(e_l, e_c)
        kept_l = [p for p, r in zip(e_l['polygons'], e_l['raw']) if r >= .5]
        kept_c = [p for p, r in zip(e_c['polygons'], e_c['raw']) if r >= .7]
        self.assertEqual(lat.fuse_thresholded(kept_l, kept_c), ref)
        # B1 box below .5 does not block the B-Clean box at the same place; low B-Clean box never added
        self.assertIn(quad(50, 50, 60, 60), ref)
        self.assertNotIn(quad(21, 0, 31, 10), ref)
        self.assertEqual(lat.dump_expectation(e_l, e_c), (2, len(ref)))


@unittest.skipUnless(HAVE_BUNDLE and HAVE_GEOM, 'needs the MONKEY_V2 bundle')
class Phase0BundleTest(unittest.TestCase):
    def test_image_records_counts_equal_pool_scorer_on_a_real_image(self):
        rows, policy, lr2, geoms, scores = da._context(BUNDLE)
        source = sorted(rows)[3]
        row = rows[source]
        gts = [g for g in row['gt_clean']['valid']][:3]
        from shapely.geometry import shape
        polys = [list(np.asarray(shape(g['geometry']).exterior.coords)[:-1].reshape(-1)) for g in gts]
        e_l = dict(polygons=polys[:2] + [quad(1, 1, 30, 12)], raw=[.9, .45, .6], packed=[.9, .45, .6])
        e_c = dict(polygons=polys[1:] + [quad(1, 1, 30, 12)], raw=[.8, .75, .9], packed=[.8, .75, .9])
        rec = p0.image_records('lr2::' + source, source, row, policy, e_l, e_c, {'epsilon_005': e_l})
        pool = da.PoolScorer(e_l, e_c, row, policy)
        self.assertEqual(rec['rows']['B1'], pool.fused(*p0.BASE))
        self.assertEqual(rec['rows']['CSB1'], pool.fused(*p0.CSB1))
        self.assertEqual(rec['rows']['DA1'], pool.fused(*p0.DUALAGREE_V1))


if __name__ == '__main__':
    unittest.main()
