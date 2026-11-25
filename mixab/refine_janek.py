"""Learned box refinement on frozen B1 + frozen box-quality v1 (one variant, locked before any A number).

Pipeline: image -> B1 (frozen, one forward) -> P-map + mask_040 candidates -> v1 scorer keeps boxes (unchanged)
          -> for each kept box: P-map crop on a fixed grid in the box frame (with context) -> small MLP -> 4 edge offsets
          -> refined quad (same frame/angle, same score, same box set) -> official evaluator.
An adaptation of bounding-box regression (offsets normalized by the box's own sides). Not a new architecture.

Locked design (from geometry + TRAIN only; nothing chosen from A):
  grid          GRID_T x GRID_S = 12 x 40 samples over s, t in [-MARGIN, 1 + MARGIN], MARGIN = .5 (= offset bound)
  extra inputs  log short side, log long side (tensor px), log aspect, raw score (4 scalars)
  model         MLP 484 -> 256 -> 256 -> 4, ReLU, last layer zero-initialised (initial output = original box)
  offsets       (left, right, top, bottom) in side units, clamp +-BOUND = .5 (refine_geom_janek)
  loss          Smooth-L1 (beta .05) on targets clamped to +-.5 (no pair is dropped: on TRAIN 26-30 % of the
                non-TP one-to-one pairs need > 1 side; dropping them would remove a quarter of the useful signal)
  weights       uniform per pair (box-regression standard); TP pairs teach stability, non-TP pairs teach the fix
  optimiser     AdamW lr 1e-3, wd 1e-4, batch 512, 30 epochs, cosine, seed 431; checkpoints at epochs 5,10,...,30
  selection     official-evaluator mean Hmean (Clean, LR2) on TRAIN-calibration; tie -> earlier; must beat
                offset-0 (= v1) on calibration, else NO_SIGNAL and A is not evaluated
"""
import math

import numpy as np

from mixab import refine_geom_janek as rg

GRID_T, GRID_S, MARGIN = 12, 40, .5
N_EXTRA = 4
IN_DIM = GRID_T*GRID_S + N_EXTRA
HIDDEN = 256
BETA = .05
TRAIN = dict(lr=1e-3, wd=1e-4, batch=512, epochs=30, seed=431, ckpt_every=5)

# eligibility (identical to scripts/refine_oracle_janek.py)
MIN_IOU, AMBIG_RATIO, MERGE_COVER = .1, 2., .5


def grid_st():
    s = np.linspace(-MARGIN, 1 + MARGIN, GRID_S)
    t = np.linspace(-MARGIN, 1 + MARGIN, GRID_T)
    return np.meshgrid(s, t)            # (T, S) each


def sample_crop(pmap, q):
    """Bilinear samples of the P-map (H x W, network space) at the box-frame grid of quad q (network space).
    Outside the map -> 0 (pad region of B1 input is also near 0)."""
    S, T = grid_st()
    pts = rg.bilinear(q, S, T)          # (T, S, 2) x,y
    x, y = pts[..., 0], pts[..., 1]
    H, W = pmap.shape
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    out = np.zeros_like(x, dtype=np.float32)
    for dx, dy, w in ((0, 0, (1 - fx)*(1 - fy)), (1, 0, fx*(1 - fy)), (0, 1, (1 - fx)*fy), (1, 1, fx*fy)):
        xi, yi = x0 + dx, y0 + dy
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        v = np.zeros_like(out)
        v[ok] = pmap[yi[ok], xi[ok]]
        out += (w*v).astype(np.float32)
    return out


def extras(q, raw):
    la, lb = np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[3] - q[0])
    short, long_ = max(min(la, lb), 1e-3), max(la, lb, 1e-3)
    return np.array([math.log(short), math.log(long_), math.log(long_/short), raw], dtype=np.float32)


def eligible(ious, covs):
    """ious/covs: dict box -> list over GT. Returns ({box: gt}, Counter-like dict of classes). Mirrors the oracle."""
    best_box = {}
    n_gt = len(next(iter(ious.values()))) if ious else 0
    for gi in range(n_gt):
        cand = [(ious[j][gi], -j) for j in ious if ious[j][gi] > 0]
        if cand:
            best_box[gi] = -max(cand)[1]
    out, cls = {}, {}
    for j, v in ious.items():
        if not n_gt or max(v) <= 0:
            k = 'no_gt_overlap'
        else:
            order = np.argsort(v)[::-1]
            g1 = int(order[0]); b1 = v[g1]; b2 = v[int(order[1])] if len(order) > 1 else 0.
            if sum(x >= MERGE_COVER for x in covs[j]) >= 2:
                k = 'merge'
            elif b1 < MIN_IOU:
                k = 'weak_overlap'
            elif b2*AMBIG_RATIO > b1:
                k = 'ambiguous'
            elif best_box.get(g1) != j:
                k = 'duplicate_not_best'
            else:
                out[j] = g1
                k = 'eligible_tp' if b1 >= .5 else 'eligible_nontp'
        cls[k] = cls.get(k, 0) + 1
    return out, cls


def make_model():
    import torch
    from torch import nn
    m = nn.Sequential(nn.Linear(IN_DIM, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 4))
    nn.init.zeros_(m[-1].weight); nn.init.zeros_(m[-1].bias)
    return m


def predict_offsets(model, X, norm):
    import torch
    with torch.inference_mode():
        Z = torch.as_tensor((X - norm['mean'])/norm['std'], dtype=torch.float32)
        return model(Z).numpy().astype(np.float64)


def fit(X, Y, w, ckpt_cb=None, log=print):
    """X (n, IN_DIM), Y (n, 4) clamped targets, w (n,) weights. Deterministic on CPU."""
    import torch
    torch.manual_seed(TRAIN['seed'])
    rng = np.random.default_rng(TRAIN['seed'])
    mean = np.average(X, 0, weights=w).astype(np.float32)
    std = np.sqrt(np.average((X - mean)**2, 0, weights=w)).astype(np.float32) + 1e-6
    norm = dict(mean=mean, std=std)
    Xt = torch.as_tensor((X - mean)/std, dtype=torch.float32)
    Yt = torch.as_tensor(Y, dtype=torch.float32)
    Wt = torch.as_tensor(w/w.mean(), dtype=torch.float32)
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=TRAIN['lr'], weight_decay=TRAIN['wd'])
    steps = TRAIN['epochs']*math.ceil(len(X)/TRAIN['batch'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    hist = []
    for ep in range(1, TRAIN['epochs'] + 1):
        perm = rng.permutation(len(X))
        tot = 0.
        for k in range(0, len(X), TRAIN['batch']):
            ix = torch.as_tensor(perm[k:k + TRAIN['batch']])
            pred = model(Xt[ix])
            l = (torch.nn.functional.smooth_l1_loss(pred, Yt[ix], beta=BETA, reduction='none').sum(1)*Wt[ix]).mean()
            if not torch.isfinite(l):
                raise RuntimeError(f'non-finite loss at epoch {ep}')
            opt.zero_grad(); l.backward(); opt.step(); sched.step()
            tot += l.item()*len(ix)
        hist.append(dict(epoch=ep, loss=tot/len(X)))
        log(f'epoch {ep} loss {tot/len(X):.5f}')
        if ckpt_cb and ep % TRAIN['ckpt_every'] == 0:
            ckpt_cb(ep, model, norm)
    return model, norm, hist
