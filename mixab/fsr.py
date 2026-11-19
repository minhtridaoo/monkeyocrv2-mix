"""Mix B preparation: residual / gate-only / FSR adapters after FPNC (Ke_hoach... section 5.2).

    F' = F + R(Z)
    residual : Z = F
    gate_only: Z = 0.5 * (A_l + A_h) * F
    fsr      : Z = A_l * L + A_h * H,   L = AvgPool3x3(F), H = F - L
    [A_l, A_h] = sigmoid(G(F)),  G = Conv1x1(C -> 2)

Locked by the plan: G has zero weight and bias (so A_l = A_h = 0.5 at init and gate_only/fsr both
feed 0.5*F to R); the last layer of R has zero weight and bias (so F' = F at init for every branch);
all branches load one shared init state; the residual branch may skip only the listed G keys.

NOT locked yet (must be agreed with Tuấn before any training): the body R (`default_body` is only a
proposal from the earlier plan), AvgPool padding (here stride 1, padding 1, count_include_pad=False so
borders are not darkened), scheduler and training recipe. This module is pure PyTorch; the MMOCR/FPNC
integration and the GPU probe live elsewhere.
"""
from torch import nn
import torch.nn.functional as nnf

from mixab import require

MODES = ('residual', 'gate_only', 'fsr')
GATE_KEYS = ('gate.weight', 'gate.bias')


def default_body(channels=256, hidden=64):
    """Proposal only (not agreed): Conv3x3 C->hidden, BN, ReLU, Conv1x1 hidden->C."""
    return nn.Sequential(nn.Conv2d(channels, hidden, 3, padding=1), nn.BatchNorm2d(hidden),
                         nn.ReLU(inplace=False), nn.Conv2d(hidden, channels, 1))


def last_conv(body):
    convs = [m for m in body.modules() if isinstance(m, nn.Conv2d)]
    require(convs, 'Residual body must contain a Conv2d output layer')
    return convs[-1]


class ResidualAdapter(nn.Module):
    def __init__(self, mode, channels=256, body=None):
        super().__init__()
        require(mode in MODES, f'mode must be one of {MODES}')
        self.mode = mode
        self.body = body if body is not None else default_body(channels)
        out = last_conv(self.body)
        require(out.out_channels == channels, 'Body output channels must equal FPNC channels')
        nn.init.zeros_(out.weight)
        if out.bias is not None:
            nn.init.zeros_(out.bias)
        if mode != 'residual':
            self.gate = nn.Conv2d(channels, 2, 1)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)

    def bands(self, x):
        low = nnf.avg_pool2d(x, 3, stride=1, padding=1, count_include_pad=False)
        return low, x - low

    def gates(self, x):
        a = self.gate(x).sigmoid()
        return a[:, :1], a[:, 1:]

    def mix(self, x):
        if self.mode == 'residual':
            return x
        a_low, a_high = self.gates(x)
        if self.mode == 'gate_only':
            return 0.5*(a_low + a_high)*x
        low, high = self.bands(x)
        return a_low*low + a_high*high

    def forward(self, x):
        return x + self.body(self.mix(x))


def shared_init_state(channels=256, body_factory=None, seed=0):
    """One init state for all branches: body (seeded) + zero gate. DBHead/B1 weights are added later."""
    import torch

    torch.manual_seed(seed)
    reference = ResidualAdapter('fsr', channels, body_factory() if body_factory else None)
    return {k: v.detach().clone() for k, v in reference.state_dict().items()}


def load_shared_state(adapter, state):
    """Load the shared init; only the gate keys may be absent from a residual adapter."""
    own = adapter.state_dict()
    missing = sorted(set(own) - set(state))
    unexpected = sorted(set(state) - set(own))
    allowed = set(GATE_KEYS) if adapter.mode == 'residual' else set()
    require(not missing, f'Init state lacks adapter keys: {missing}')
    require(set(unexpected) <= allowed, f'Unexpected keys for {adapter.mode}: {sorted(set(unexpected) - allowed)}')
    for key in own:
        require(own[key].shape == state[key].shape, f'Shape differs for {key}')
    adapter.load_state_dict({k: state[k] for k in own}, strict=True)
    return dict(skipped=unexpected)


def trainable_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
