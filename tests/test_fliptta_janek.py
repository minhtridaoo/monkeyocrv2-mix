"""Flip-TTA: CPU tests of the flip/unflip/average maths, the alignment checks, sample selection and the outcome label."""
import unittest

from mixab import fliptta_janek as ft 

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, 'needs torch')
class FlipMathTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.hw = (5, 7)                                   # content 5 x 7 inside a 8 x 10 padded grid
        self.x = torch.zeros(1, 3, 8, 10)
        self.x[..., :5, :7] = torch.randn(1, 3, 5, 7)

    def test_flip_only_touches_content_and_reverses_columns(self):
        y = ft.flip_inputs(self.x, self.hw)
        self.assertTrue(torch.equal(y[..., :5, :7], self.x[..., :5, :7].flip(-1)))
        self.assertTrue(torch.equal(y[..., 5:, :], self.x[..., 5:, :]))        # bottom padding untouched
        self.assertTrue(torch.equal(y[..., :, 7:], self.x[..., :, 7:]))        # right padding untouched
        self.assertFalse(torch.equal(y, self.x))

    def test_flip_twice_is_identity_and_input_not_mutated(self):
        before = self.x.clone()
        y = ft.flip_inputs(ft.flip_inputs(self.x, self.hw), self.hw)
        self.assertTrue(torch.equal(y, before))
        self.assertTrue(torch.equal(self.x, before))

    def test_unflip_pmap_inverts_flip_for_a_pixel_equivariant_model(self):
        # a "model" that is exactly flip-equivariant (pixelwise): P(flip(x)) == flip(P(x)) -> P1b == P0 -> mean == P0
        p0 = torch.sigmoid(self.x.mean(1))                      # [1, 8, 10]
        p1 = torch.sigmoid(ft.flip_inputs(self.x, self.hw).mean(1))
        p1b = ft.unflip_pmap(p1, self.hw)
        self.assertTrue(torch.allclose(p1b, p0))
        self.assertTrue(torch.allclose(ft.average_pmaps(p0, p1b, self.hw), p0))

    def test_misaligned_without_unflip_differs(self):
        p0 = torch.sigmoid(self.x.mean(1) + torch.arange(10.)/10)        # not left-right symmetric
        p1 = ft.flip_inputs(p0.unsqueeze(1), self.hw).squeeze(1)         # P of the flipped image for an equivariant model
        self.assertFalse(torch.allclose(p1[..., :5, :7], p0[..., :5, :7]))
        self.assertTrue(torch.allclose(ft.unflip_pmap(p1, self.hw), p0))

    def test_average_keeps_padding_from_p0(self):
        p0 = torch.rand(1, 8, 10)
        p1b = torch.rand(1, 8, 10)
        m = ft.average_pmaps(p0, p1b, self.hw)
        self.assertTrue(torch.allclose(m[..., :5, :7], (p0[..., :5, :7] + p1b[..., :5, :7])/2))
        self.assertTrue(torch.equal(m[..., 5:, :], p0[..., 5:, :]))
        self.assertTrue(torch.equal(m[..., :, 7:], p0[..., :, 7:]))

    def test_padding_constant_check(self):
        self.assertTrue(ft.padding_is_constant(self.x, self.hw))
        bad = self.x.clone()
        bad[..., 6, 8] = 1.
        self.assertFalse(ft.padding_is_constant(bad, self.hw))
        self.assertTrue(ft.padding_is_constant(self.x[..., :5, :7], self.hw))      # no padding at all

    def test_mask_iou(self):
        a = torch.zeros(1, 8, 10); b = torch.zeros(1, 8, 10)
        self.assertIsNone(ft.mask_iou(a, b, self.hw))
        a[..., 0, :4] = .9; b[..., 0, 2:6] = .9
        self.assertAlmostEqual(ft.mask_iou(a, b, self.hw), 2/6)

    def test_alignment_checks_and_gate(self):
        p0 = torch.rand(1, 8, 10); p0[..., 4:, :] = 0
        p0[..., :5, :7] = (torch.arange(7.)/7).expand(5, 7)             # left-right asymmetric
        p1 = ft.flip_inputs(p0.unsqueeze(1), self.hw).squeeze(1)       # equivariant model
        p1b = ft.unflip_pmap(p1, self.hw)
        c = ft.alignment_checks(self.x, p0, p1, p1b, self.hw)
        self.assertTrue(c['pad_constant'] and c['flip_twice_identity'] and c['pmap_shape_finite'])
        self.assertEqual(c['iou_aligned'], 1.0)
        self.assertLess(c['iou_control'], 1.0)
        self.assertEqual(c['mean_abs_diff'], 0.0)
        c['same_metainfo'] = True
        self.assertTrue(ft.alignment_gate([c, c])['passed'])
        c2 = dict(c, same_metainfo=False)
        self.assertFalse(ft.alignment_gate([c, c2])['passed'])
        c3 = dict(c, iou_aligned=.2, iou_control=.4)
        self.assertFalse(ft.alignment_gate([c3])['passed'])


class SelectionAndLabelTest(unittest.TestCase):
    def test_select_sources_is_split_a_deterministic_and_seed_specific(self):
        from mixab import reselect_split_janek as rs, zoom_janek as zoom 
        ids = [f'{i:016x}' for i in range(3000)]
        a = ft.select_sources(ids, 50)
        self.assertEqual(a, ft.select_sources(list(reversed(ids)), 50))
        self.assertEqual(len(a), 50)
        self.assertTrue(all(rs.split_of(s) == 'A' for s in a))
        self.assertNotEqual(a, zoom.select_sources(ids, 50))           # different seed key

    def test_labels(self):
        L = ft.label
        self.assertEqual(L(False, True, .5, .2, 0., 0.)[0], 'INCONCLUSIVE_ALIGNMENT')
        self.assertEqual(L(True, False, .5, .2, 0., 0.)[0], 'INCONCLUSIVE_ALIGNMENT')
        self.assertEqual(L(True, True, -.1, -.5, 0., 0.)[0], 'NEGATIVE')
        self.assertEqual(L(True, True, 0., -.5, 0., 0.)[0], 'NEGATIVE')
        self.assertEqual(L(True, True, .6, -.1, 0., 0.)[0], 'INCONCLUSIVE')            # CI contains 0
        self.assertEqual(L(True, True, .5, .1, -.1, -1.)[0], 'PASS_SCREEN')
        n, f = L(True, True, .2, .05, 0., 0.)
        self.assertEqual(n, 'POSITIVE_BELOW_GATE')
        self.assertFalse(f['c1_lr2_gain_ge_0p3_and_ci_gt0'])
        n, f = L(True, True, .5, .1, -.5, 0.)
        self.assertEqual((n, f['c2_clean_ge_minus_0p3']), ('POSITIVE_BELOW_GATE', False))
        n, f = L(True, True, .5, .1, 0., -4.)
        self.assertEqual((n, f['c3_lr2_tiny_ge_minus_3']), ('POSITIVE_BELOW_GATE', False))


if __name__ == '__main__':
    unittest.main()
