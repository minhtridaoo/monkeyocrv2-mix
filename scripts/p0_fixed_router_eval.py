"""P0 (fixed) — CPU, counts only. Fixes the label leak in diagnose_tradeoff_v2.py Q4.

Deployment rule: a router sees only the image; each expert gets ONE post-processing setting, locked on A
and applied to every image routed to it, whatever its true condition. True condition labels are used only
to compute the metric (Hmean per condition, then mean), never to pick a setting per image.

Outputs (results/diag_v2_cpu/p0_fixed.json):
  singles        each model at its R1 setting (from v1 selection) — A and B
  oracle_cond    B-Clean on Clean, B1 on LR2, setting per expert locked on A (condition known: reference)
  size_router    (T, k_bclean, k_b1) jointly chosen on A; routing by native max side
  postproc_ctrl  ONE detector, setting chosen per condition on A (separates threshold gain from expert gain)
  margin_A/B     oracle and router minus best single; target = best single + 0.5 pp
  boot_B         paired bootstrap over sources (Clean/LR2 of a source kept together), 2000 draws
Everything on B is EXPLORATORY (B already seen).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mixab import reselect_split as rs  # noqa: E402
from mixab.merge_audit import load_rows  # noqa: E402
from mixab.monkey import monkey_root  # noqa: E402

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('H:/My Drive/CS431_MixA/runs/merge_audit_v1_A100-SXM4-40GB')
MODELS = ['B-Clean', 'merge_a1_4', 'merge_a1_2', 'merge_a3_4', 'B1']


def hm(t):
    """Hmean from summed fields [match, gt, det, ...]; works on arrays with fields on last axis."""
    t = np.asarray(t, dtype=np.float64)
    den = t[..., 1] + t[..., 2]
    return np.where(den > 0, 2*t[..., 0]/np.maximum(den, 1), 0.0)


def main():
    sys.stdout.reconfigure(errors='backslashreplace')
    data = {n: rs.load_counts(RUN/f'{n}_source_counts.npz') for n in MODELS}
    sources, settings = data['B-Clean']['source_ids'], data['B-Clean']['settings']
    for n in MODELS:
        assert list(data[n]['source_ids']) == list(sources)
    C = {n: data[n]['counts'].astype(np.int64) for n in MODELS}   # [src, cond, setting, field]
    split = np.array([rs.split_of(s) for s in sources])
    idx = {'A': np.where(split == 'A')[0], 'B': np.where(split == 'B')[0]}
    sel = json.loads((ROOT/'results/mix_a_v1/selection.json').read_text(encoding='utf-8'))['selection']
    key = lambda k: rs.setting_key(settings[k])  # noqa: E731

    rows = load_rows(monkey_root()/'docs/handoff/artifacts/pr_quad_audit_v1')
    side = np.zeros((len(sources), 2), dtype=np.int64)
    for i, s in enumerate(sources):
        w, h = rows[s]['clean_size']
        side[i, 0] = max(w, h)
        side[i, 1] = max(max(1, w//2), max(1, h//2))   # assumption: native LR2 = floor half (not read from files)

    def mean_of(per_cond_totals):
        h = [float(hm(per_cond_totals[c])) for c in (0, 1)]
        return dict(clean=h[0], lr2=h[1], mean=(h[0] + h[1])/2)

    def single(n, ix, k):
        return mean_of([C[n][ix, c, k].sum(0) for c in (0, 1)])

    def routed(ix, route, kb, k1):
        """route[ix_pos, cond] True -> B1. Same setting per expert for both conditions."""
        tot = []
        for c in (0, 1):
            r = route[:, c]
            tot.append(C['B-Clean'][ix, c, kb][~r].sum(0) + C['B1'][ix, c, k1][r].sum(0))
        return tot

    def joint_select(ix, route):
        """Pick (kb, k1) maximising mean Hmean on ix; one setting per expert, pooled over conditions."""
        Pb = [C['B-Clean'][ix, c][~route[:, c]].sum(0) for c in (0, 1)]   # [setting, field]
        P1 = [C['B1'][ix, c][route[:, c]].sum(0) for c in (0, 1)]
        score = sum(hm(Pb[c][:, None, :] + P1[c][None, :, :]) for c in (0, 1))/2   # [kb, k1]
        kb, k1 = np.unravel_index(np.argmax(score), score.shape)
        return int(kb), int(k1), float(score[kb, k1])

    A, B = idx['A'], idx['B']
    out = {'label': 'EXPLORATORY CPU; settings locked on A; B already seen', 'run': str(RUN)}

    out['singles'] = {n: {s: single(n, idx[s], sel[n]['setting_index']) for s in 'AB'} | {'setting': key(sel[n]['setting_index'])}
                      for n in MODELS}
    best_single = {s: max(out['singles'][n][s]['mean'] for n in MODELS) for s in 'AB'}
    best_single_name = {s: max(MODELS, key=lambda n: out['singles'][n][s]['mean']) for s in 'AB'}

    # oracle: condition known; still ONE setting per expert (each expert only ever sees one condition)
    oracle_route = np.zeros((len(sources), 2), dtype=bool)
    oracle_route[:, 1] = True
    kb, k1, _ = joint_select(A, oracle_route[A])
    out['oracle_cond'] = dict(settings=dict(bclean=key(kb), b1=key(k1)),
                              **{s: mean_of(routed(idx[s], oracle_route[idx[s]], kb, k1)) for s in 'AB'})

    # size router: choose T, kb, k1 jointly on A
    cands = sorted({int(t) for t in np.unique(side) if t % 32 == 0})
    best = None
    for T in cands:
        r = side[A] <= T
        kb_, k1_, v = joint_select(A, r)
        if best is None or v > best[0]:
            best = (v, T, kb_, k1_)
    _, T, kb, k1 = best
    route = side <= T
    out['size_router'] = dict(T=T, settings=dict(bclean=key(kb), b1=key(k1)),
                              routed_to_b1={s: dict(clean=float(route[idx[s], 0].mean()), lr2=float(route[idx[s], 1].mean()))
                                            for s in 'AB'},
                              **{s: mean_of(routed(idx[s], route[idx[s]], kb, k1)) for s in 'AB'})
    out['size_separation'] = dict(clean_min=int(side[:, 0].min()), clean_max=int(side[:, 0].max()),
                                  lr2_min=int(side[:, 1].min()), lr2_max=int(side[:, 1].max()),
                                  disjoint=bool(side[:, 1].max() < side[:, 0].min()))

    # post-processing control: ONE detector, setting chosen per condition on A (condition known)
    ctrl = {}
    for n in MODELS:
        ks = []
        for c in (0, 1):
            ks.append(int(np.argmax(hm(C[n][A, c].sum(0)))))
        ctrl[n] = dict(settings=dict(clean=key(ks[0]), lr2=key(ks[1])),
                       **{s: mean_of([C[n][idx[s], c, ks[c]].sum(0) for c in (0, 1)]) for s in 'AB'})
    out['postproc_ctrl'] = ctrl
    best_ctrl = {s: max(ctrl[n][s]['mean'] for n in MODELS) for s in 'AB'}

    out['margin'] = {s: dict(best_single=best_single[s], best_single_model=best_single_name[s],
                             target_plus_0p5=best_single[s] + 0.005,
                             oracle_minus_best_single_pp=(out['oracle_cond'][s]['mean'] - best_single[s])*100,
                             router_minus_best_single_pp=(out['size_router'][s]['mean'] - best_single[s])*100,
                             oracle_minus_best_postproc_ctrl_pp=(out['oracle_cond'][s]['mean'] - best_ctrl[s])*100,
                             best_postproc_ctrl=best_ctrl[s])
                     for s in 'AB'}

    # paired bootstrap on B over sources: oracle vs best single (B-chosen name is exploratory; use A-best name)
    rng = np.random.default_rng(431)
    nA = best_single_name['A']
    kS = sel[nA]['setting_index']
    kb_o = [i for i in range(len(settings)) if key(i) == out['oracle_cond']['settings']['bclean']][0]
    k1_o = [i for i in range(len(settings)) if key(i) == out['oracle_cond']['settings']['b1']][0]
    Bc = C['B-Clean'][B, 0, kb_o]; Bl = C['B1'][B, 1, k1_o]
    Sc = C[nA][B, 0, kS]; Sl = C[nA][B, 1, kS]
    diffs = []
    for _ in range(2000):
        j = rng.integers(0, len(B), len(B))
        o = (hm(Bc[j].sum(0)) + hm(Bl[j].sum(0)))/2
        s_ = (hm(Sc[j].sum(0)) + hm(Sl[j].sum(0)))/2
        diffs.append(float(o - s_)*100)
    out['boot_B_oracle_vs_A_best_single'] = dict(single=nA, mean_pp=float(np.mean(diffs)),
                                                  ci95_pp=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                                                  p_le_0p5=float(np.mean(np.array(diffs) <= 0.5)))

    dest = ROOT/'results/diag_v2_cpu'
    dest.mkdir(parents=True, exist_ok=True)
    (dest/'p0_fixed.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
