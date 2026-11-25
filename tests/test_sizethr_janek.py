"""Size-aware threshold for B1: CPU tests."""
import unittest

from mixab import sizethr_janek as st 

try:
    import shapely  # noqa: F401
    HAVE_SHAPELY = True
except ImportError:
    HAVE_SHAPELY = False


class SizeRuleTest(unittest.TestCase):
    def test_keep_by_size_and_single_threshold_special_case(self):
        raw = [.25, .45, .65, .35]
        sizes = [2., 3., 10., 20.]
        self.assertEqual(st.keep_by_size(raw, sizes, 4, .2, .5), {0, 1, 2})
        self.assertEqual(st.keep_by_size(raw, sizes, 4, .5, .3), {2, 3})
        for t in (.2, .3, .5):
            self.assertEqual(st.keep_by_size(raw, sizes, 8, t, t), {i for i, r in enumerate(raw) if r >= t})

    def test_grid_contains_every_single_threshold(self):
        g = st.grid([.2, .3])
        self.assertEqual(len(g), len(st.GEOMS)*len(st.CUTS)*4)
        self.assertTrue(all((geo, c, s, s) in g for geo in st.GEOMS for c in st.CUTS for s in (.2, .3)))

    def test_native_factor_matches_matcher_gt_scaling(self):
        self.assertEqual(st.native_factor(dict(clean_size=[1024, 768])), 384/768)
        self.assertEqual(st.native_factor(dict(clean_size=[1024, 767])), 383/767)

    @unittest.skipUnless(HAVE_SHAPELY, 'needs shapely')
    def test_short_edge_of_rotated_and_degenerate_boxes(self):
        self.assertAlmostEqual(st.short_edge([0, 0, 40, 0, 40, 10, 0, 10]), 10.)
        self.assertAlmostEqual(st.short_edge([0, 0, 30, 30, 25, 35, -5, 5]), 50**.5, places=6)
        self.assertEqual(st.short_edge([1, 2, 3]), 0.)


if __name__ == '__main__':
    unittest.main()
