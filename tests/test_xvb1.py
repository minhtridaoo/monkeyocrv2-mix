"""XV-B1: CPU tests of the locked removal rule and the outcome label."""
import unittest

from mixab import xvb1

try:
    import cv2  # noqa: F401
    import shapely  # noqa: F401
    HAVE_GEOM = True
except ImportError:
    HAVE_GEOM = False


def quad(x0, y0, x1, y1):
    return [x0, y0, x1, y0, x1, y1, x0, y1]


@unittest.skipUnless(HAVE_GEOM, 'needs cv2 + shapely')
class RemovableTest(unittest.TestCase):
    H = 1000          # clean height 1000 -> LR2 factor 0.5

    def test_large_unconfirmed_removed_small_unconfirmed_kept(self):
        large = quad(0, 0, 200, 20)       # short side 20 clean = 10 LR2 -> eligible
        small = quad(300, 0, 500, 14)     # 14 clean = 7 LR2 -> exempt
        self.assertEqual(xvb1.removable([large, small], [], self.H), {0})

    def test_threshold_is_inclusive_at_8_lr2(self):
        self.assertEqual(xvb1.removable([quad(0, 0, 200, 16)], [], self.H), {0})      # exactly 8.0
        self.assertEqual(xvb1.removable([quad(0, 0, 200, 15.9)], [], self.H), set())

    def test_confirmed_by_any_overlapping_evidence_is_kept(self):
        large = quad(0, 0, 200, 20)
        self.assertEqual(xvb1.removable([large], [quad(10, 0, 210, 20)], self.H), set())     # IoU ~ .90
        self.assertEqual(xvb1.removable([large], [quad(150, 0, 600, 20)], self.H), {0})      # IoU ~ .09

    def test_iou_cut_is_inclusive_and_many_to_one(self):
        large = quad(0, 0, 100, 20)
        half = quad(0, 0, 30, 20)         # IoU = 0.30 exactly
        self.assertEqual(xvb1.removable([large], [half], self.H), set())
        self.assertEqual(xvb1.removable([large], [quad(0, 0, 29, 20)], self.H), {0})
        far = quad(0, 0, 100, 20)         # one evidence box confirms two parents
        two = [quad(0, 0, 100, 20), quad(0, 2, 100, 22)]
        self.assertEqual(xvb1.removable(two, [far], self.H), set())

    def test_degenerate_polygon_is_kept(self):
        self.assertEqual(xvb1.removable([[0, 0, 0, 0, 0, 0]], [], self.H), set())


class LabelTest(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(xvb1.label(.4, .1, -1., 10, 20)[0], 'PASS_SCREEN')
        self.assertEqual(xvb1.label(-.1, -.3, 0., 10, 20)[0], 'NEGATIVE')
        self.assertEqual(xvb1.label(0., -.3, 0., 10, 20)[0], 'NEGATIVE')
        n, f = xvb1.label(.2, .05, 0., 10, 20)
        self.assertEqual(n, 'POSITIVE_BELOW_GATE')
        self.assertFalse(f['c1_gain_ge_0p3_and_ci_gt0'])
        n, f = xvb1.label(.5, -.1, 0., 10, 20)           # CI includes 0
        self.assertEqual(n, 'POSITIVE_BELOW_GATE')
        n, f = xvb1.label(.5, .1, -5., 10, 20)           # tiny too low
        self.assertEqual((n, f['c2_tiny_ge_minus_3']), ('POSITIVE_BELOW_GATE', False))
        n, f = xvb1.label(.5, .1, 0., 30, 20)            # FP not lower
        self.assertEqual((n, f['c3_fp_lower']), ('POSITIVE_BELOW_GATE', False))


if __name__ == '__main__':
    unittest.main()
