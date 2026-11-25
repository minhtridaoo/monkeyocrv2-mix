import unittest

import numpy as np

from mixab import refine_janek as rf


class TestRefine(unittest.TestCase):
    def test_crop_of_linear_map(self):
        H, W = 60, 90
        yy, xx = np.mgrid[0:H, 0:W].astype(float)
        pmap = (.01*xx + .02*yy).astype(np.float32)
        q = np.array([[30, 20], [60, 20], [60, 30], [30, 30]], float)
        c = rf.sample_crop(pmap, q)
        self.assertEqual(c.shape, (rf.GRID_T, rf.GRID_S))
        S, T = rf.grid_st()
        x, y = 30 + 30*S, 20 + 10*T
        np.testing.assert_allclose(c, .01*x + .02*y, atol=1e-5)

    def test_crop_outside_is_zero(self):
        pmap = np.ones((10, 10), np.float32)
        q = np.array([[100, 100], [120, 100], [120, 110], [100, 110]], float)
        self.assertEqual(float(np.abs(rf.sample_crop(pmap, q)).max()), 0.)

    def test_zero_init_model(self):
        import torch  # noqa: F401
        m = rf.make_model()
        X = np.random.default_rng(0).normal(size=(7, rf.IN_DIM)).astype(np.float32)
        out = rf.predict_offsets(m, X, dict(mean=np.zeros(rf.IN_DIM, np.float32), std=np.ones(rf.IN_DIM, np.float32)))
        self.assertEqual(out.shape, (7, 4))
        self.assertEqual(float(np.abs(out).max()), 0.)

    def test_eligible_classes(self):
        # box 0: clean one-to-one; box 1: duplicate of GT0 with lower IoU; box 2: merge; box 3: no overlap
        ious = {0: [.7, 0., 0.], 1: [.4, 0., 0.], 2: [0., .3, .3], 3: [0., 0., 0.]}
        covs = {0: [.9, 0., 0.], 1: [.5, 0., 0.], 2: [0., .9, .9], 3: [0., 0., 0.]}
        pairs, cls = rf.eligible(ious, covs)
        self.assertEqual(pairs, {0: 0})
        self.assertEqual(cls, dict(eligible_tp=1, duplicate_not_best=1, merge=1, no_gt_overlap=1))

    def test_fit_finite_and_learns(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(600, rf.IN_DIM)).astype(np.float32)
        Y = np.clip(.1*X[:, :4], -.5, .5).astype(np.float32)
        saved = []
        old = dict(rf.TRAIN)
        rf.TRAIN.update(epochs=10, batch=128)
        try:
            m, norm, hist = rf.fit(X, Y, np.ones(len(X)), ckpt_cb=lambda ep, *_: saved.append(ep), log=lambda *_: None)
        finally:
            rf.TRAIN.clear(); rf.TRAIN.update(old)
        self.assertTrue(all(np.isfinite(h['loss']) for h in hist))
        self.assertLess(hist[-1]['loss'], hist[0]['loss'])
        self.assertEqual(saved, [5, 10])


if __name__ == '__main__':
    unittest.main()
