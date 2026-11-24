"""MSB1: CPU tests of the split rule, the subset bookkeeping and the exactness of the Dinkelbach ceiling."""
from itertools import product
import unittest

from mixab import msb1

try:
    from shapely.geometry import box
    HAVE_SHAPELY = True
except ImportError:
    HAVE_SHAPELY = False


@unittest.skipUnless(HAVE_SHAPELY, 'needs shapely')
class AssignTest(unittest.TestCase):
    def test_assignment_containment_and_best_parent(self):
        parents = [box(0, 0, 100, 10), box(90, 0, 200, 10), None]
        children = [box(5, 1, 40, 9),          # inside parent 0
                    box(50, 1, 85, 9),         # inside parent 0
                    box(92, 1, 120, 9),        # inside parent 1 only
                    box(80, 1, 110, 9),        # 1/3 in p0, 2/3 in p1 -> frac < .8 for both -> unassigned
                    box(300, 0, 310, 5)]       # nowhere
        a = msb1.assign_children(parents, children)
        self.assertEqual(a, {0: [0, 1], 1: [2]})
        self.assertEqual(msb1.split_candidates(a), {0: [0, 1]})

    def test_child_in_two_parents_goes_to_largest_fraction(self):
        parents = [box(0, 0, 100, 10), box(10, 0, 60, 10)]
        children = [box(12, 1, 58, 9), box(70, 1, 90, 9)]
        a = msb1.assign_children(parents, children)
        # child 0 is fully inside both parents (frac 1.0 = 1.0): tie -> lower parent index 0
        self.assertEqual(a, {0: [0, 1]})
        parents = [box(0, 0, 50, 10), box(0, 0, 20, 10)]
        a = msb1.assign_children(parents, [box(0, 1, 30, 9)])   # frac 1.0 in p0, .67 in p1
        self.assertEqual(a, {0: [0]})


class BookkeepingTest(unittest.TestCase):
    def test_kept_set(self):
        kept = msb1.kept_set([0, 1, 2], {1: [0, 3]}, {1}, n_parent_pool=5)
        self.assertEqual(kept, {0, 2, 5, 8})
        self.assertEqual(msb1.kept_set([0, 1, 2], {1: [0, 3]}, set(), 5), {0, 1, 2})

    def test_dinkelbach_equals_brute_force(self):
        options = [[(3, 5), (4, 7), (4, 6)], [(1, 2), (2, 2)], [(0, 1), (5, 9), (3, 3)]]
        gt = 20
        h, pick = msb1.dinkelbach(options, gt)
        best = max(2*sum(o[k][0] for o, k in zip(options, ks))/(gt + sum(o[k][1] for o, k in zip(options, ks)))
                   for ks in product(*[range(len(o)) for o in options]))
        self.assertAlmostEqual(h, best, places=12)


if __name__ == '__main__':
    unittest.main()
