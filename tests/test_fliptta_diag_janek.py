"""Flip-TTA diagnosis: CPU tests of the pure helpers (pairing, bins)."""
import unittest

import numpy as np

from mixab import fliptta_diag_janek as fd 


class DiagHelpersTest(unittest.TestCase):
    def test_greedy_pairs_one_to_one_by_iou(self):
        iou = np.array([[.9, .6], [.7, .2]])
        self.assertEqual([(i, j) for i, j, _ in fd.greedy_pairs(iou, [0, 1], [0, 1])], [(0, 0)])
        self.assertEqual([(i, j) for i, j, _ in fd.greedy_pairs(iou, [0, 1], [0, 1], thr=.15)], [(0, 0), (1, 1)])

    def test_bins(self):
        self.assertEqual(fd.sbin(.6), '0.60-0.65')
        self.assertEqual(fd.sbin(.95), '0.80-1.00')
        self.assertEqual(fd.sbin(.5), '<0.60')
        self.assertEqual(fd.abin(2.), '0-3')
        self.assertEqual(fd.abin(20.), '>=12')

    def test_aspect_of_rectangle(self):
        self.assertAlmostEqual(fd.aspect([0, 0, 40, 0, 40, 10, 0, 10]), 4., places=3)


if __name__ == '__main__':
    unittest.main()
