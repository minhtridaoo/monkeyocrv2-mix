"""Mix A-core A1: endpoint copies, interpolation arithmetic, BN policy and checkpoint I/O."""
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
from torch import nn

from mixab import read_json 
from mixab import weight_merge_janek as wm 
from mixab.monkey_janek import monkey_root

_ROOT = monkey_root(required=False)
REAL_B1 = (_ROOT or Path('.'))/'docs/handoff/weights/b1-inference.pth'


def tiny_state(seed, bn_batches=7):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.ReLU(), nn.Conv2d(4, 1, 1))
    state = model.state_dict()
    state['1.running_var'] = torch.rand(4) + 0.5
    state['1.running_mean'] = torch.randn(4)
    state['1.num_batches_tracked'] = torch.tensor(bn_batches)
    return {k: v.clone() for k, v in state.items()}


def fresh_model():
    return nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.ReLU(), nn.Conv2d(4, 1, 1))


class InterpolationTest(unittest.TestCase):
    def setUp(self):
        self.bclean, self.b1 = tiny_state(0, bn_batches=5), tiny_state(1, bn_batches=9)

    def test_endpoints_are_exact_copies_not_arithmetic(self):
        for alpha, source in ((Fraction(0), self.bclean), (Fraction(1), self.b1)):
            merged, report = wm.interpolate(self.bclean, self.b1, alpha)
            self.assertEqual(report['mode'], 'endpoint_copy')
            self.assertTrue(wm.compare(merged, source)['equal'])
            # integer buffer keeps the endpoint's own value (not B1's) at alpha = 0
            self.assertEqual(int(merged['1.num_batches_tracked']), int(source['1.num_batches_tracked']))
            # a copy, not an alias of the source tensors
            merged['0.weight'].add_(1)
            self.assertFalse(torch.equal(merged['0.weight'], source['0.weight']))

    def test_interior_alpha_formula_and_integer_policy(self):
        alpha = Fraction(1, 4)
        merged, report = wm.interpolate(self.bclean, self.b1, alpha)
        self.assertEqual(report['mode'], 'linear')
        for key, value in merged.items():
            if value.is_floating_point():
                expected = (0.75*self.bclean[key].double() + 0.25*self.b1[key].double()).to(value.dtype)
                self.assertTrue(torch.equal(value, expected), key)
                self.assertEqual(value.dtype, self.bclean[key].dtype)
        self.assertEqual(int(merged['1.num_batches_tracked']), 9)
        self.assertEqual(report['non_floating_keys'], ['1.num_batches_tracked'])
        self.assertEqual(report['non_floating_differ'], ['1.num_batches_tracked'])

    def test_self_merge_is_identity(self):
        for text in wm.DEFAULT_ALPHAS:
            merged, _ = wm.interpolate(self.b1, self.b1, wm.parse_alpha(text))
            self.assertTrue(wm.compare(merged, self.b1)['equal'], text)

    def test_merged_state_loads_strictly_and_runs(self):
        merged, _ = wm.interpolate(self.bclean, self.b1, Fraction(1, 2))
        wm.validate(merged)
        model = fresh_model()
        model.load_state_dict(merged, strict=True)
        output = model.eval()(torch.randn(2, 3, 8, 8))
        self.assertTrue(bool(torch.isfinite(output).all()))

    def test_rejects_incompatible_checkpoints(self):
        missing = dict(self.b1)
        missing.pop('3.bias')
        with self.assertRaisesRegex(ValueError, 'Key sets differ'):
            wm.interpolate(self.bclean, missing, Fraction(1, 2))
        reshaped = dict(self.b1, **{'3.bias': torch.zeros(2)})
        with self.assertRaisesRegex(ValueError, 'Shape differs'):
            wm.interpolate(self.bclean, reshaped, Fraction(1, 2))
        retyped = dict(self.b1, **{'3.bias': self.b1['3.bias'].double()})
        with self.assertRaisesRegex(ValueError, 'Dtype differs'):
            wm.interpolate(self.bclean, retyped, Fraction(1, 2))

    def test_validation_rejects_bad_values(self):
        bad = dict(self.b1, **{'1.running_var': torch.tensor([1., 0., 1., 1.])})
        with self.assertRaisesRegex(ValueError, 'running_var'):
            wm.validate(bad)
        bad = dict(self.b1, **{'0.bias': torch.tensor([float('nan')]*4)})
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            wm.validate(bad)

    def test_alpha_range_and_tags(self):
        with self.assertRaises(ValueError):
            wm.parse_alpha('1.5')
        with self.assertRaises(ValueError):
            wm.parse_alpha('-0.25')
        self.assertEqual([wm.alpha_tag(wm.parse_alpha(a)) for a in wm.DEFAULT_ALPHAS],
                         ['alpha_0p00', 'alpha_0p25', 'alpha_0p50', 'alpha_0p75', 'alpha_1p00'])


class CheckpointIOTest(unittest.TestCase):
    def test_mmengine_wrapper_and_module_prefix(self):
        state = tiny_state(2)
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'iter_10000.pth'
            torch.save(dict(state_dict={f'module.{k}': v for k, v in state.items()}, meta=dict(iter=10000)), path)
            loaded, info = wm.load_state_dict(path)
            self.assertEqual(info['wrapper'], 'mmengine_state_dict')
            self.assertTrue(info['stripped_module_prefix'])
            self.assertTrue(wm.compare(loaded, state)['equal'])

    def test_build_writes_endpoints_manifest_and_refuses_overwrite(self):
        bclean, b1 = tiny_state(3, 4), tiny_state(4, 8)
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            torch.save(bclean, tmp/'bclean.pth')
            torch.save(dict(state_dict=b1, meta={}), tmp/'b1.pth')
            rows = wm.build(tmp/'bclean.pth', tmp/'b1.pth', wm.DEFAULT_ALPHAS, tmp/'out')
            self.assertEqual([r['alpha'] for r in rows], ['0', '1/4', '1/2', '3/4', '1'])
            first, _ = wm.load_state_dict(tmp/'out/alpha_0p00.pth')
            last, _ = wm.load_state_dict(tmp/'out/alpha_1p00.pth')
            self.assertTrue(wm.compare(first, bclean)['equal'])
            self.assertTrue(wm.compare(last, b1)['equal'])
            manifest = read_json(tmp/'out/merge_manifest.json')
            self.assertEqual(len(manifest['models']), 5)
            self.assertIn('PENDING_REFERENCE_CHECK', manifest['policy']['verification'])
            self.assertEqual(manifest['endpoints']['b1']['wrapper'], 'mmengine_state_dict')
            with self.assertRaisesRegex(ValueError, 'Refusing to overwrite'):
                wm.build(tmp/'bclean.pth', tmp/'b1.pth', ['0.5'], tmp/'out')


@unittest.skipUnless(REAL_B1.is_file() and REAL_B1.stat().st_size > 1_000_000, 'B1 export not present locally')
class RealB1SmokeTest(unittest.TestCase):
    def test_real_b1_loads_validates_and_self_merges_exactly(self):
        b1, info = wm.load_state_dict(REAL_B1)
        self.assertEqual(info['wrapper'], 'raw_state_dict')
        self.assertEqual(info['sha256'], '9c3814e494a73a0a6089ba5a6585f079ea8f5a6defdc9347213bb98a64337147')
        checks = wm.validate(b1)
        self.assertGreater(checks['running_var_tensors'], 0)
        merged, _ = wm.interpolate(b1, b1, Fraction(1, 2))
        self.assertTrue(wm.compare(merged, b1)['equal'])


if __name__ == '__main__':
    unittest.main()
