import unittest

import numpy as np

from mixab import refine_geom_janek as rg


def rot_rect(cx, cy, w, h, ang):
    c, s = np.cos(ang), np.sin(ang)
    base = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
    return base @ np.array([[c, s], [-s, c]]) + [cx, cy]


class TestRefineGeom(unittest.TestCase):
    def test_zero_offset_exact(self):
        q = rot_rect(100, 50, 80, 20, .3)
        out, st = rg.apply_offsets(q, np.zeros(4))
        self.assertEqual(st, 'ok')
        self.assertEqual(out, q.reshape(-1).tolist())

    def test_zero_offset_exact_nonparallelogram(self):
        q = np.array([[0, 0], [10, 1], [11, 6], [-1, 5]], float)
        out, _ = rg.apply_offsets(q, np.zeros(4))
        self.assertEqual(out, q.reshape(-1).tolist())

    def test_target_roundtrip(self):
        q = rot_rect(100, 50, 80, 20, .3)
        g = rot_rect(95, 52, 50, 16, .3)
        d = rg.target_offsets(q, g)
        out, st = rg.apply_offsets(q, d, bound=None)
        self.assertEqual(st, 'ok')
        np.testing.assert_allclose(np.asarray(out).reshape(-1, 2), g, atol=1e-9)

    def test_bound_and_orientation(self):
        rng = np.random.default_rng(0)
        q = rot_rect(0, 0, 40, 10, -.7)
        a0 = rg.signed_area(q)
        for _ in range(500):
            d = rng.normal(0, 1, 4)
            out, st = rg.apply_offsets(q, d)
            p = np.asarray(out).reshape(-1, 2)
            self.assertGreater(rg.signed_area(p)*a0, 0)
            # convex, same winding at every corner -> no self-intersection
            cr = [np.cross(p[(k+1) % 4] - p[k], p[(k+2) % 4] - p[(k+1) % 4]) for k in range(4)]
            self.assertTrue(all(np.sign(x) == np.sign(cr[0]) for x in cr))

    def test_fallbacks(self):
        q = rot_rect(0, 0, 40, 10, 0)
        self.assertEqual(rg.apply_offsets(q, [np.nan, 0, 0, 0])[1], 'fallback_nonfinite')
        self.assertEqual(rg.apply_offsets(q, [.5, -.5, 0, 0])[1], 'fallback_collapse')
        self.assertEqual(rg.apply_offsets(None, [0, 0, 0, 0])[1], 'fallback_not_quad')
        self.assertIsNone(rg.quad([1, 2, 3, 4, 5, 6]))

    def test_uniform_is_dilation(self):
        q = rot_rect(0, 0, 40, 10, 0)
        out, _ = rg.apply_offsets(q, rg.uniform_offsets(q, 2.), bound=None)
        p = np.asarray(out).reshape(-1, 2)
        self.assertAlmostEqual(abs(rg.signed_area(p)), 44*14)


if __name__ == '__main__':
    unittest.main()
