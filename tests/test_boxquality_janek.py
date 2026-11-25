"""Box-quality scorer: CPU tests of the selection rule, the features (vs Tuan's native score), the logistic model,
cutoff choice, bootstrap, outcome labels, per-member zip fetch and evaluator-mirroring labels."""
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from mixab import boxquality_janek as bq 
from mixab.monkey_janek import monkey_root

try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False
ROOT = monkey_root(required=False)
POLICY = (ROOT or Path('.'))/'docs/handoff/artifacts/pr_quad_audit_v1/provenance/textvqa_evaluation_protocol.json'


def native_bbox_score(score_map, poly_pts):
    """Verbatim logic of installed_db_postprocessor._get_bbox_score (provenance file, lines ~127-153)."""
    h, w = score_map.shape[:2]
    poly_pts = poly_pts.copy()
    xmin = np.clip(np.floor(poly_pts[:, 0].min()).astype(np.int32), 0, w - 1)
    xmax = np.clip(np.ceil(poly_pts[:, 0].max()).astype(np.int32), 0, w - 1)
    ymin = np.clip(np.floor(poly_pts[:, 1].min()).astype(np.int32), 0, h - 1)
    ymax = np.clip(np.ceil(poly_pts[:, 1].max()).astype(np.int32), 0, h - 1)
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    poly_pts[:, 0] = poly_pts[:, 0] - xmin
    poly_pts[:, 1] = poly_pts[:, 1] - ymin
    cv2.fillPoly(mask, poly_pts.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(score_map[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


class SelectionTest(unittest.TestCase):
    def test_roles_counts_disjoint_and_deterministic(self):
        ids = [dict(source_image_id=f'{i:016x}', clean_relpath=f'train_images/{i:016x}.jpg', clean_sha256='0'*64) for i in range(2500)]
        a = bq.select_train_sources(ids)
        b = bq.select_train_sources(list(reversed(ids)))
        self.assertEqual(a, b)
        roles = [x['role'] for x in a]
        self.assertEqual((roles.count('smoke'), roles.count('fit'), roles.count('calibration')), (10, 1600, 400))
        self.assertEqual(len({x['source_image_id'] for x in a}), 2010)


@unittest.skipUnless(HAVE_CV2, 'needs cv2')
class FeatureTest(unittest.TestCase):
    def test_region_mean_equals_native_score_exactly(self):
        rng = np.random.default_rng(0)
        m = rng.random((120, 160)).astype(np.float32)
        for _ in range(50):
            c = rng.integers(10, 110, 2)
            pts = (c + rng.integers(-9, 10, (6, 2))).astype(np.int32)
            st = bq.region_stats(m, pts)
            self.assertEqual(st['mean'], native_bbox_score(m, pts))
            self.assertLessEqual(st['q10'], st['q50']); self.assertLessEqual(st['q50'], st['q90'])
            self.assertLessEqual(st['q90'], st['pmax'] + 1e-12)

    def test_quad_features_constant_map_and_shape(self):
        m = np.full((100, 100), .7, dtype=np.float32)
        poly = np.array([[20, 40], [60, 40], [60, 50], [20, 50]], dtype=np.int32)
        quad = np.array([[18, 38], [62, 38], [62, 52], [18, 52]], dtype=np.float32)
        f = bq.quad_features(m, poly, quad, 100)
        self.assertAlmostEqual(f['quad_mean'], .7, places=6)
        self.assertAlmostEqual(f['ring_mean'], .7, places=6)
        self.assertAlmostEqual(f['fill_quad'], 400/616, places=6)
        self.assertAlmostEqual(f['fill_hull'], 1., places=6)
        self.assertAlmostEqual(f['short_norm'], .14, places=6)
        self.assertEqual(len(bq.feature_vector(bq.region_stats(m, poly), f)), len(bq.FEATURES))

    def test_ring_detects_low_probability_border(self):
        m = np.zeros((100, 100), dtype=np.float32)
        m[40:51, 20:61] = 1.
        poly = np.array([[20, 40], [60, 40], [60, 50], [20, 50]], dtype=np.int32)
        quad = np.array([[15, 35], [65, 35], [65, 55], [15, 55]], dtype=np.float32)
        f = bq.quad_features(m, poly, quad, 100)
        self.assertLess(f['ring_mean'], .05)
        self.assertGreater(f['quad_mean'], .3)


class ModelTest(unittest.TestCase):
    def test_logistic_recovers_sign_and_is_deterministic(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(4000, 3))
        y = (rng.random(4000) < 1/(1 + np.exp(-(2*X[:, 0] - X[:, 2])))).astype(float)
        w = np.ones(4000)
        b1 = bq.fit_logistic(X, y, w)
        b2 = bq.fit_logistic(X, y, w)
        self.assertTrue(np.array_equal(b1, b2))
        self.assertGreater(b1[1], 1.); self.assertLess(b1[3], -.5); self.assertLess(abs(b1[2]), .3)

    def test_score_only_logistic_is_monotone_in_raw(self):
        rng = np.random.default_rng(2)
        r = rng.random(3000)
        y = (rng.random(3000) < r).astype(float)
        mu, sd = bq.standardize_fit(r[:, None], np.ones(3000))
        b = bq.fit_logistic((r[:, None] - mu)/sd, y, np.ones(3000))
        p = bq.predict(b, (np.sort(r)[:, None] - mu)/sd)
        self.assertTrue(np.all(np.diff(p) >= 0))

    def test_choose_cut_tiny_constraint_and_tie(self):
        def rb(tp, pred, tiny_tp):
            a = np.array([[tp, 100, pred, 10, tiny_tp]])
            return dict(clean=a, lr2=a)
        per = [(.5, rb(80, 100, 5)), (.6, rb(80, 100, 5)), (.7, rb(85, 100, 1))]
        best = bq.choose_cut(per, tiny_ref=.5)          # .7 violates tiny (10% < 50% - 3 pp); .5/.6 tie -> .6
        self.assertEqual(best[1], .6)

    def test_paired_boot_identity_and_labels(self):
        a = dict(clean=np.array([[5, 10, 8, 1, 1]]*20), lr2=np.array([[4, 10, 9, 1, 0]]*20))
        r = bq.paired_boot_mean(a, a)
        self.assertEqual(r['delta_pp'], 0.); self.assertEqual(r['ci95_pp'], [0., 0.])
        self.assertEqual(bq.label_outcome(.5, .1, False), 'INVALID_CONTROL')
        self.assertEqual(bq.label_outcome(-.1, -.5, True), 'NO_SIGNAL')
        self.assertEqual(bq.label_outcome(.5, -.1, True), 'INCONCLUSIVE')
        self.assertEqual(bq.label_outcome(.2, .05, True), 'POSITIVE_BELOW_GATE')
        self.assertEqual(bq.label_outcome(.4, .05, True), 'PASS_CALIBRATION_SCREEN')


class FetchTest(unittest.TestCase):
    def _zip(self):
        buf = io.BytesIO()
        payload = {'train_images/a.jpg': b'A'*5000, 'train_images/b.jpg': bytes(range(256))*40}
        with zipfile.ZipFile(buf, 'w') as zf:
            zf.writestr(zipfile.ZipInfo('train_images/a.jpg'), payload['train_images/a.jpg'], compress_type=zipfile.ZIP_DEFLATED)
            zf.writestr(zipfile.ZipInfo('train_images/b.jpg'), payload['train_images/b.jpg'], compress_type=zipfile.ZIP_STORED)
        return buf.getvalue(), payload

    def test_fetch_members_by_range(self):
        blob, payload = self._zip()

        class Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def opener(req):
            lo, hi = map(int, req.headers['Range'].split('=')[1].split('-'))
            return Resp(blob[lo:hi + 1])
        orig = bq.RangeFile.__init__

        def patched(self, url, size, opener_=None):
            orig(self, url, size, opener)
        bq.RangeFile.__init__ = patched
        try:
            with tempfile.TemporaryDirectory() as d:
                items = [(k, hashlib.sha256(v).hexdigest()) for k, v in payload.items()]
                r = bq.fetch_members(items, d, url='http://x', size=len(blob))
                self.assertEqual(r['ok'], 2)
                for k, v in payload.items():
                    self.assertEqual((Path(d)/k).read_bytes(), v)
                with self.assertRaises(ValueError):
                    bq.fetch_members([('train_images/a.jpg', '0'*64)], d + '/x', url='http://x', size=len(blob))
        finally:
            bq.RangeFile.__init__ = orig


@unittest.skipUnless(HAVE_CV2 and POLICY.is_file(), 'needs MONKEY_V2 bundle + cv2')
class LabelTest(unittest.TestCase):
    def test_labels_mirror_evaluator_filters(self):
        from mixab.monkey_janek import use_monkey
        use_monkey()
        from shapely.geometry import box, mapping
        policy = json.loads(POLICY.read_text(encoding='utf-8'))
        row = dict(clean_size=[200, 100], roi_clean=mapping(box(0, 0, 200, 100)),
                   gt_clean=dict(valid=[dict(ann_id='g1', geometry=mapping(box(10, 10, 60, 30)))],
                                 ignored=[dict(ann_id='g2', geometry=mapping(box(100, 10, 150, 30)), ignore_reason='unreadable')]))

        def q(x0, y0, x1, y1):
            return [x0, y0, x1, y0, x1, y1, x0, y1]
        rec = dict(polygons=[q(11, 10, 60, 30), q(10, 50, 60, 70), q(100, 10, 150, 30)], raw=[.9, .8, .7], packed=[.9, .8, .7])
        lab, m = bq.task_labels(rec, row, policy)
        self.assertEqual(lab, [1, 0, None])         # match / counted FP / dropped by ignore filter


if __name__ == '__main__':
    unittest.main()
