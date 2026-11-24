"""DualAgree v1: CPU tests for dump shards, pairing, fusion rule and scoring equivalence with Tuan's matcher."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import numpy as np

from mixab import dualagree as da
from mixab.monkey import monkey_root

ROOT = monkey_root(required=False)
BUNDLE = (ROOT or Path('.'))/'docs/handoff/artifacts/pr_quad_audit_v1'
HAVE_BUNDLE = (BUNDLE/'data/validation_remaining.jsonl.gz').is_file()
try:
    import shapely  # noqa: F401
    HAVE_SHAPELY = True
except ImportError:
    HAVE_SHAPELY = False
INF = da.INF


def quad(x0, y0, x1, y1):
    return [x0, y0, x1, y0, x1, y1, x0, y1]


class ShardTest(unittest.TestCase):
    def test_roundtrip_keeps_order_floats_and_empty_entries(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'shard_0000.npz'
            recs = [('s1', 0, [quad(0, 0, 1, 1), quad(2, 2, 3, 3)], [0.25, 0.9], [0.25, 0.8999999761581421]),
                    ('s2', 1, [quad(5, 5, 6.123456789012, 6)], [0.123456789012345678 + .2], [0.3])]
            da.write_shard(path, 'id-x', ['s1', 's2', 's3'], recs)
            self.assertEqual(da.shard_identity(path), 'id-x')
            out = da.read_shard(path, 2)
            self.assertEqual(set(out), {'s1', 's2', 's3'})
            self.assertEqual(out['s1'][0]['raw'], [0.25, 0.9])
            self.assertEqual(out['s1'][0]['polygons'][1], quad(2, 2, 3, 3))
            self.assertEqual(out['s2'][1]['polygons'][0][4], 6.123456789012)
            self.assertEqual(out['s2'][1]['raw'][0], 0.123456789012345678 + .2)
            self.assertEqual(out['s1'][1], dict(polygons=[], raw=[], packed=[]))
            self.assertEqual(out['s3'][0]['raw'], [])
            with self.assertRaises(ValueError):
                da.load_model_dump(tmp, 2, identity_sha='other')


class PairingTest(unittest.TestCase):
    def test_greedy_is_one_to_one_by_iou_with_theta(self):
        iou = np.array([[.9, .6], [.8, .55], [.2, .4]])
        ov = da.sorted_overlaps(iou)
        self.assertEqual(da.greedy_pairs(ov, [True]*3, [True]*2, .5), [(0, 0), (1, 1)])
        self.assertEqual(da.greedy_pairs(ov, [False, True, True], [True]*2, .5), [(1, 0)])
        self.assertEqual(da.greedy_pairs(ov, [True]*3, [True]*2, .85), [(0, 0)])
        self.assertEqual(da.greedy_pairs(ov, [True]*3, [True]*2, .3), [(0, 0), (1, 1), (2, 1)][:2])


class FuseRuleTest(unittest.TestCase):
    def setUp(self):
        self.raw_l = [.9, .45, .25]
        self.raw_c = [.7, .65, .3]
        iou = np.array([[.8, 0, 0], [0, .6, 0], [0, 0, 0]])
        self.ov = da.sorted_overlaps(iou)

    def test_b1_alone_and_bclean_alone(self):
        for t in (.2, .3, .5):
            kl, kc = da.fuse_keep(self.raw_l, self.raw_c, self.ov, t, INF, INF, .5)
            self.assertEqual(kl, {i for i, r in enumerate(self.raw_l) if r >= t})
            self.assertEqual(kc, set())
            kl, kc = da.fuse_keep(self.raw_l, self.raw_c, self.ov, INF, t, INF, .5)
            self.assertEqual((kl, kc), (set(), {j for j, r in enumerate(self.raw_c) if r >= t}))

    def test_pair_emits_one_box_and_agreement_rescues_b1(self):
        # pair (1,1): raw_l .45 < t_l .5 but both >= t_a .4 -> keep B1 poly, drop B-Clean duplicate
        kl, kc = da.fuse_keep(self.raw_l, self.raw_c, self.ov, .5, .6, .4, .5)
        self.assertEqual(kl, {0, 1})
        self.assertEqual(kc, set())           # j=0 and j=1 are paired; j=2 (.3) < t_c

    def test_pair_falls_back_to_bclean_when_b1_too_weak(self):
        kl, kc = da.fuse_keep(self.raw_l, self.raw_c, self.ov, .5, .6, INF, .5)
        self.assertEqual(kl, {0})
        self.assertEqual(kc, {1})             # pair (1,1): B1 .45 < .5, B-Clean .65 >= .6
        kl, kc = da.fuse_keep(self.raw_l, self.raw_c, self.ov, .5, .25, INF, .5)
        self.assertEqual(kc, {1, 2})          # unpaired j=2 kept, j=0 suppressed by its B1 partner

    def test_grid_and_family(self):
        g = da.grid([.2, .3])
        self.assertEqual(len(g), 2*(3*3*3 - 2))
        self.assertEqual(len({c for c in g if da.family(*c[:3]) == 'B1_alone'}), 2)
        self.assertEqual(da.family(.3, INF, .4), 'B1+agree')
        self.assertEqual(da.family(.3, .4, INF), 'B1+BConly')


@unittest.skipUnless(HAVE_BUNDLE and HAVE_SHAPELY, 'needs MONKEY_V2 bundle and shapely')
class MatcherEquivalenceTest(unittest.TestCase):
    """PoolScorer must reproduce Tuan's CandidateMatcher exactly."""

    @classmethod
    def setUpClass(cls):
        from mixab import merge_audit as ma
        from mixab.monkey import use_monkey
        use_monkey()
        rows = ma.load_rows(BUNDLE)
        cls.policy = json.loads((BUNDLE/'provenance/textvqa_evaluation_protocol.json').read_text(encoding='utf-8'))
        cls.row = next(r for r in rows.values() if len(r['gt_clean']['valid']) >= 6)
        rng = np.random.default_rng(0)
        polys_l, polys_c = [], []
        for e in cls.row['gt_clean']['valid'][:6]:
            xs = [p[0] for p in e['geometry']['coordinates'][0]]; ys = [p[1] for p in e['geometry']['coordinates'][0]]
            box = (min(xs), min(ys), max(xs), max(ys))
            polys_l.append(quad(*box))
            polys_c.append(quad(box[0] + 1, box[1], box[2] + 1, box[3]))
        polys_l.append(quad(1, 1, 30, 12)); polys_c.append(quad(40, 40, 90, 60))   # likely FPs
        polys_l.append([1, 2, 3]); polys_c.append(quad(2, 2, 2, 9))                  # invalid / degenerate
        n = len(polys_l)
        cls.l = dict(polygons=polys_l, raw=[float(x) for x in rng.uniform(.2, 1, n)], packed=None)
        cls.c = dict(polygons=polys_c, raw=[float(x) for x in rng.uniform(.2, 1, n)], packed=None)
        cls.l['packed'], cls.c['packed'] = list(cls.l['raw']), list(cls.c['raw'])

    def test_subset_scoring_equals_matcher(self):
        from mixab import merge_audit as ma
        from rev4.decision_core import CandidateMatcher
        pool = da.PoolScorer(self.l, self.c, self.row, self.policy)
        union = CandidateMatcher(dict(polygons=self.l['polygons'] + self.c['polygons'],
                                      scores=self.l['packed'] + self.c['packed']),
                                 self.l['raw'] + self.c['raw'], self.row, self.policy)
        only_l = CandidateMatcher(dict(polygons=self.l['polygons'], scores=self.l['packed']),
                                  self.l['raw'], self.row, self.policy)
        n = pool.n_l
        for cut in (.2, .4, .6, .8):
            keep_all = {p for p, r in enumerate(self.l['raw'] + self.c['raw']) if r >= cut}
            self.assertEqual(pool.score({p for p in keep_all if p < n}, {p - n for p in keep_all if p >= n}),
                             ma.count_row(union.score(cut)))
            self.assertEqual(pool.fused(cut, INF, INF, .5), ma.count_row(only_l.score(cut)))

    def test_fusion_adds_matches_without_duplicates(self):
        pool = da.PoolScorer(self.l, self.c, self.row, self.policy)
        both = pool.fused(.2, .2, .2, .5)
        alone = pool.fused(.2, INF, INF, .5)
        self.assertGreaterEqual(both[0], alone[0])
        self.assertLessEqual(both[2], alone[2] + 2)   # duplicates of paired boxes are suppressed


class LockedVariantTest(unittest.TestCase):
    @unittest.skipUnless(HAVE_SHAPELY, 'needs shapely')
    def test_fuse_locked_adds_only_unmatched_confident_bclean_boxes(self):
        l = dict(polygons=[quad(0, 0, 10, 5), quad(50, 0, 60, 5)], raw=[.6, .4], packed=[.6, .4])
        c = dict(polygons=[quad(0, 0, 10, 5), quad(100, 0, 110, 5), quad(200, 0, 210, 5), quad(50, 0, 60, 5)],
                 raw=[.9, .8, .5, .75], packed=[.9, .8, .5, .75])
        out = da.fuse_locked(l, c)
        # B1 box 0 kept (.6>=.5); B1 box 1 dropped (.4); B-Clean: 0 duplicate of B1 0, 1 added (.8),
        # 2 too weak (.5<.7), 3 added: its only B1 overlap (.4) is below t_l, so it never enters pairing
        self.assertEqual(out, [quad(0, 0, 10, 5), quad(100, 0, 110, 5), quad(50, 0, 60, 5)])

    def test_summarize_ms(self):
        s = da.summarize_ms([1, 2, 3, 4, 100])
        self.assertEqual((s['n'], s['median']), (5, 3.0))
        self.assertAlmostEqual(s['mean'], 22.0)


if __name__ == '__main__':
    unittest.main()
