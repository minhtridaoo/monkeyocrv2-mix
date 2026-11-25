"""CPU diagnosis after Mix A v1 (does NOT modify v1 outputs). Uses only A100 A2r counts + bundle metadata.

Q1  Frontier: does any interpolated alpha reach an LR2 (recall, precision) point that neither endpoint
    reaches at any of the 72 settings (on split A)? If not, interpolation only moves along the existing
    trade-off curve.
Q2  Linearity: are per-bin recalls of interior alphas above the straight line between endpoints (on B)?
Q3  Routing upper bound: B-Clean on Clean tasks + B1 on LR2 tasks ("condition oracle"), each with its
    own setting chosen on A for that condition; evaluated on B. This is an UPPER BOUND that assumes
    the condition is known.
Q4  Realistic router: route each image by its native max side (<= T -> B1, else B-Clean); T and the two
    settings are chosen on A, then evaluated on B. Image sizes from the locked GT manifest.
Everything on B is EXPLORATORY (B already seen). Output: docs/handoff/artifacts/janek/diag_v2_cpu/diagnosis_janek.json
"""
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mixab import reselect_split_janek as rs # noqa: E402
from mixab.merge_audit_janek import load_rows  # noqa: E402
from mixab.monkey_janek import monkey_root  # noqa: E402

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('H:/My Drive/CS431_MixA/runs/merge_audit_v1_A100-SXM4-40GB')
MODELS = [('B-Clean', '0'), ('merge_a1_4', '1/4'), ('merge_a1_2', '1/2'), ('merge_a3_4', '3/4'), ('B1', '1')]


def hmean(t):
    return float(rs.exact_hmean(*t[:3]))


def main():
    sys.stdout.reconfigure(errors='backslashreplace')
    data = {n: rs.load_counts(RUN/f'{n}_source_counts.npz') for n, _ in MODELS}
    sources, settings = data['B-Clean']['source_ids'], data['B-Clean']['settings']
    split = np.array([rs.split_of(s) for s in sources])
    A, B = np.where(split == 'A')[0], np.where(split == 'B')[0]
    C = {n: d['counts'] for n, d in data.items()}
    out = {}

    # Q1 frontier on A, LR2 condition
    def points(n, idx, cond=1):
        tot = C[n][idx].sum(axis=0)[cond]            # [setting, fields]
        return [(t[0]/t[1], t[0]/max(t[2], 1)) for t in tot]
    ends = points('B-Clean', A) + points('B1', A)
    q1 = {}
    for n, a in MODELS[1:4]:
        beyond = 0
        for r, p in points(n, A):
            if not any(re >= r - 1e-12 and pe >= p - 1e-12 for re, pe in ends):
                beyond += 1
        q1[n] = dict(settings_not_dominated_by_any_endpoint_setting=beyond, of=len(settings))
    for cond, name in ((0, 'clean'), (1, 'lr2')):
        ends_c = points('B-Clean', A, cond) + points('B1', A, cond)
        for n, a in MODELS[1:4]:
            q1.setdefault(n, {})[f'{name}_not_dominated'] = sum(
                not any(re >= r - 1e-12 and pe >= p - 1e-12 for re, pe in ends_c) for r, p in points(n, A, cond))
    out['Q1_frontier_A'] = q1

    # Q2 linearity on B at each model's R1 setting (from v1 selection)
    sel = json.loads((ROOT/'docs/handoff/artifacts/janek/mix_a_v1/selection_janek.json').read_text(encoding='utf-8'))['selection']
    q2 = {}
    for cond, cname in ((0, 'clean'), (1, 'lr2')):
        for bi, bname in enumerate(rs.BINS):
            g, t = rs.bin_fields(11)[bi]
            vals = {}
            for n, a in MODELS:
                tot = C[n][B][:, cond, sel[n]['setting_index'], :].sum(axis=0)
                vals[float(Fraction(a))] = tot[t]/tot[g] if tot[g] else None
            line = {x: (1 - x)*vals[0.0] + x*vals[1.0] for x in vals}
            q2[f'{cname}_{bname}'] = {str(x): dict(recall=round(vals[x], 4), above_line_pp=round((vals[x] - line[x])*100, 2))
                                      for x in vals}
    out['Q2_bin_linearity_B'] = q2

    # helpers for routing
    def best_setting(n, idx, cond, mask=None):
        best, bk = None, None
        for k in range(len(settings)):
            m = C[n][idx, cond, k, :]
            if mask is not None:
                m = m[mask]
            v = Fraction(2*int(m[:, 0].sum()), int(m[:, 1].sum() + m[:, 2].sum()))
            if best is None or v > best:
                best, bk = v, k
        return bk

    def routed_total(idx, cond, route_b1, k_clean_model, k_b1):
        """route_b1: bool array over idx for this condition."""
        a = C['B-Clean'][idx, cond, k_clean_model, :][~route_b1].sum(axis=0)
        b = C['B1'][idx, cond, k_b1, :][route_b1].sum(axis=0)
        return a + b

    def summary(totals):
        res = {c: dict(hmean=hmean(totals[i]), fp=int(totals[i][2] - totals[i][0]),
                       tiny=float(totals[i][4]/totals[i][3]) if totals[i][3] else None) for i, c in enumerate(rs.CONDITIONS)}
        res['mean_hmean'] = (res['clean']['hmean'] + res['lr2']['hmean'])/2
        return res

    # single-model references on B at R1 settings
    out['single_B'] = {n: summary([C[n][B][:, c, sel[n]['setting_index'], :].sum(axis=0) for c in (0, 1)]) for n, _ in MODELS}

    # Q3 condition oracle: B-Clean for clean (setting best on A clean), B1 for lr2 (best on A lr2)
    k_bc_clean = best_setting('B-Clean', A, 0)
    k_b1_lr2 = best_setting('B1', A, 1)
    tot = [C['B-Clean'][B][:, 0, k_bc_clean, :].sum(axis=0), C['B1'][B][:, 1, k_b1_lr2, :].sum(axis=0)]
    out['Q3_condition_oracle_B'] = dict(settings=dict(bclean_clean=rs.setting_key(settings[k_bc_clean]),
                                                       b1_lr2=rs.setting_key(settings[k_b1_lr2])), **summary(tot))

    # Q4 size router; sizes from GT manifest (clean_size); lr2 = max(1, side//2)
    rows = load_rows(monkey_root()/'docs/handoff/artifacts/pr_quad_audit_v1')
    maxside = np.zeros((len(sources), 2), dtype=np.int64)
    for i, s in enumerate(sources):
        w, h = rows[s]['clean_size']
        maxside[i, 0] = max(w, h)
        maxside[i, 1] = max(max(1, w//2), max(1, h//2))
    thresholds = sorted(set(np.unique(maxside).tolist()))
    cand = [t for t in thresholds if t % 32 == 0] or thresholds
    best = None
    # KNOWN FLAW (review 22/11): settings below are chosen per (expert x TRUE condition) -> label leak.
    # A deployed router only knows its own decision. Use scripts/p0_fixed_router_eval_janek.py (one setting per expert).
    for T in cand:
        tot_a = []
        for c in (0, 1):
            route = maxside[A, c] <= T
            kb = best_setting('B-Clean', A, c, ~route) if (~route).any() else 0
            k1 = best_setting('B1', A, c, route) if route.any() else 0
            tot_a.append(routed_total(A, c, route, kb, k1))
        v = sum(Fraction(2*int(t[0]), int(t[1] + t[2])) for t in tot_a)
        if best is None or v > best[0]:
            best = (v, T)
    T = best[1]
    tot_b, frac, ks = [], {}, {}
    for c, cname in ((0, 'clean'), (1, 'lr2')):
        ra, rb = maxside[A, c] <= T, maxside[B, c] <= T
        kb = best_setting('B-Clean', A, c, ~ra) if (~ra).any() else 0
        k1 = best_setting('B1', A, c, ra) if ra.any() else 0
        ks[cname] = dict(bclean=rs.setting_key(settings[kb]), b1=rs.setting_key(settings[k1]))
        tot_b.append(routed_total(B, c, rb, kb, k1))
        frac[cname] = float(rb.mean())
    out['Q4_size_router_B'] = dict(threshold_maxside=int(T), fraction_routed_to_B1=frac, settings=ks, **summary(tot_b))
    out['size_overlap'] = dict(clean_maxside_quantiles=np.quantile(maxside[:, 0], [0, .05, .5, .95, 1]).tolist(),
                               lr2_maxside_quantiles=np.quantile(maxside[:, 1], [0, .05, .5, .95, 1]).tolist())
    out['label'] = 'EXPLORATORY CPU diagnosis on development data already seen; not a method result'
    dest = ROOT/'docs/handoff/artifacts/janek/diag_v2_cpu'
    dest.mkdir(parents=True, exist_ok=True)
    (dest/'diagnosis_janek.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
