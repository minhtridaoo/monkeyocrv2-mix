"""Mix B preparation: zero-init identity, gate start, gradient flow, param parity, shared init."""
import unittest

import torch
from torch import nn

from mixab import fsr_janek as fsr 

C = 16      # small channel count for CPU tests; the real FPNC output is 256


def small_body():
    return nn.Sequential(nn.Conv2d(C, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(), nn.Conv2d(4, C, 1))


def branches(state=None):
    state = state or fsr.shared_init_state(C, small_body, seed=7)
    out = {}
    for mode in fsr.MODES:
        adapter = fsr.ResidualAdapter(mode, C, small_body())
        fsr.load_shared_state(adapter, state)
        out[mode] = adapter
    return out, state


class FsrTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(2, C, 9, 11)

    def test_identity_at_init_in_train_and_eval(self):
        adapters, _ = branches()
        for mode, adapter in adapters.items():
            for train in (True, False):
                adapter.train(train)
                self.assertTrue(torch.equal(adapter(self.x), self.x), (mode, train))

    def test_gates_start_at_half_and_both_gated_branches_feed_half_x(self):
        adapters, _ = branches()
        a_low, a_high = adapters['fsr'].gates(self.x)
        self.assertTrue(torch.equal(a_low, torch.full_like(a_low, .5)))
        self.assertTrue(torch.equal(a_high, torch.full_like(a_high, .5)))
        self.assertTrue(torch.allclose(adapters['gate_only'].mix(self.x), .5*self.x))
        self.assertTrue(torch.allclose(adapters['fsr'].mix(self.x), .5*self.x, atol=1e-6))

    def test_fsr_rewrite_identity(self):
        adapter = fsr.ResidualAdapter('fsr', C, small_body())
        nn.init.normal_(adapter.gate.weight)
        a_low, a_high = adapter.gates(self.x)
        low, _ = adapter.bands(self.x)
        rewritten = a_high*self.x + (a_low - a_high)*low
        self.assertTrue(torch.allclose(adapter.mix(self.x), rewritten, atol=1e-6))

    def test_gradient_policy_first_step_then_after_updates(self):
        adapters, _ = branches()
        for mode, adapter in adapters.items():
            adapter.train()
            opt = torch.optim.SGD(adapter.parameters(), lr=.1)
            loss = (adapter(self.x) - 1).pow(2).mean()
            loss.backward()
            last = fsr.last_conv(adapter.body)
            self.assertGreater(float(last.weight.grad.abs().sum()), 0, mode)
            first = adapter.body[0]
            self.assertEqual(float(first.weight.grad.abs().sum()), 0., mode)   # blocked by zero last layer
            for _ in range(5):
                opt.step()
                opt.zero_grad()
                (adapter(self.x) - 1).pow(2).mean().backward()
            self.assertGreater(float(first.weight.grad.abs().sum()), 0, mode)
            if mode != 'residual':
                self.assertGreater(float(adapter.gate.weight.grad.abs().sum()), 0, mode)

    def test_parameter_parity(self):
        adapters, _ = branches()
        counts = {m: fsr.trainable_parameters(a) for m, a in adapters.items()}
        self.assertEqual(counts['fsr'], counts['gate_only'])
        self.assertEqual(counts['fsr'] - counts['residual'], C*2 + 2)
        self.assertEqual(sum(p.numel() for p in fsr.ResidualAdapter('fsr', 256).gate.parameters()), 514)

    def test_shared_init_and_strict_key_policy(self):
        adapters, state = branches()
        body_keys = [k for k in state if k.startswith('body.')]
        for mode, adapter in adapters.items():
            own = adapter.state_dict()
            for key in body_keys:
                self.assertTrue(torch.equal(own[key], state[key]), (mode, key))
        bad = dict(state, extra=torch.zeros(1))
        with self.assertRaisesRegex(ValueError, 'Unexpected keys'):
            fsr.load_shared_state(fsr.ResidualAdapter('residual', C, small_body()), bad)
        short = {k: v for k, v in state.items() if k != 'gate.bias'}
        with self.assertRaisesRegex(ValueError, 'lacks adapter keys'):
            fsr.load_shared_state(fsr.ResidualAdapter('fsr', C, small_body()), short)
        info = fsr.load_shared_state(fsr.ResidualAdapter('residual', C, small_body()), state)
        self.assertEqual(info['skipped'], sorted(fsr.GATE_KEYS))

    def test_rejects_unknown_mode_and_wrong_body(self):
        with self.assertRaises(ValueError):
            fsr.ResidualAdapter('freq', C, small_body())
        with self.assertRaisesRegex(ValueError, 'output channels'):
            fsr.ResidualAdapter('fsr', C, nn.Sequential(nn.Conv2d(C, 3, 1)))


if __name__ == '__main__':
    unittest.main()
