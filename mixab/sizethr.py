"""Size-aware score threshold for B1 alone (1 model, 1x forward), from the DualAgree dump. CPU only.

Rule: for each B1 candidate, s = short edge of its minimum rotated rectangle, converted to native-LR2
pixels with the same factor Tuan's matcher uses for GT heights (max(1, H//2)/H). Keep the candidate if
    raw >= t_small  when s < cut,   raw >= t_large  otherwise.
t_small = t_large is exactly a single global threshold, so the 72-setting baseline is inside the family.

Data limits (dump v1): only candidates ACCEPTED by Tuan's postprocessor are stored (min width 5, or 3 for
width_3), so a smaller min width or per-box unclip cannot be tested; only LR2 images were dumped, so the
rule's effect on Clean images is NOT measured (a 1-model claim needs a Clean dump of B1).

Pre-registered (22/11, before the full dump existed), chosen on split A only:
  grid    geometry in {epsilon_005, width_3} x cut in CUTS x (t_small, t_large) in SCORES^2
  base    best of B1's 72 single settings recomputed from the SAME dump (same GPU numerics)
  pass    on A: LR2 Hmean >= base + 0.30 pp; tiny (<4 px) recall >= base - 3 pp;
          FP not above the lowest FP of any of the 72 settings reaching the same recall
Split B is exploratory (already seen). Output: <output>/size_decision.json, size_sweep.npz
"""
import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

from mixab import read_json, require, save_json
from mixab import dualagree as da
from mixab import merge_audit as ma

GEOMS = ('epsilon_005', 'width_3')
CUTS = (3, 4, 5, 6, 8, 10, 12, 16)
EMPTY = dict(polygons=[], raw=[], packed=[])


def short_edge(coords):
    import numpy as np
    poly = da.polygon_of(coords)
    if poly is None:
        return 0.
    c = np.asarray(poly.minimum_rotated_rectangle.exterior.coords)[:4]
    return float(min(np.linalg.norm(c[0] - c[1]), np.linalg.norm(c[1] - c[2])))


def native_factor(row):
    h = row['clean_size'][1]
    return max(1, h//2)/h


def keep_by_size(raw, sizes, cut, t_small, t_large):
    return {i for i, (r, s) in enumerate(zip(raw, sizes)) if r >= (t_small if s < cut else t_large)}


def grid(scores):
    return [(g, c, ts, tl) for g in GEOMS for c in CUTS for ts in scores for tl in scores]


def _h(t):
    return Fraction(2*int(t[0]), int(t[1] + t[2])) if (t[1] + t[2]) else Fraction(0)


def analyze(a):
    import numpy as np

    from mixab import reselect_split as rs
    rows, policy, lr2, geoms, scores = da._context(a.bundle)
    out_dir = Path(a.output)
    require(not (out_dir/'size_decision.json').exists(), f'Refusing to overwrite {out_dir}/size_decision.json')
    out_dir.mkdir(parents=True, exist_ok=True)
    ident = read_json(Path(a.dump)/'dump_identity.json')
    dump = da.load_model_dump(Path(a.dump)/'B1', len(geoms), ident['identity_sha256'])
    sids = sorted(dump, key=lambda s: lr2[s])
    sources = [lr2[s] for s in sids]
    split = np.array([rs.split_of(s) for s in sources])
    A, B = split == 'A', split == 'B'
    # baseline: 72 single settings from the same dump
    base = np.zeros((len(geoms)*len(scores), len(sids), ma.FIELDS), dtype=np.int64)
    scorers = {}
    for i, sid in enumerate(sids):
        row = rows[lr2[sid]]
        f = native_factor(row)
        for gi, g in enumerate(geoms):
            e = dump[sid][gi]
            pool = da.PoolScorer(e, EMPTY, row, policy)
            for si, s in enumerate(scores):
                base[gi*len(scores) + si, i] = pool.score({k for k, r in enumerate(e['raw']) if r >= s}, set())
            if g in GEOMS:
                scorers[(g, i)] = (pool, e['raw'], [short_edge(p)*f for p in e['polygons']])
    combos = grid(scores)
    per = np.zeros((len(combos), len(sids), ma.FIELDS), dtype=np.int64)
    for k, (g, c, ts, tl) in enumerate(combos):
        for i in range(len(sids)):
            pool, raw, sizes = scorers[(g, i)]
            per[k, i] = pool.score(keep_by_size(raw, sizes, c, ts, tl), set())
        if k % 128 == 0:
            print(f'combo {k + 1}/{len(combos)}', flush=True)
    names = [f'{g}/score_{s:.2f}' for g in geoms for s in scores]

    def stats(t):
        tp, gt, pred, tg, tt = (int(x) for x in t[:5])
        return dict(tp=tp, gt=gt, pred=pred, fp=pred - tp, recall=tp/gt if gt else 0., precision=tp/pred if pred else 0.,
                    hmean=float(_h(t)), tiny_recall=tt/tg if tg else None)
    totA, totB = per[:, A].sum(1), per[:, B].sum(1)
    bA, bB = base[:, A].sum(1), base[:, B].sum(1)
    kb = max(range(len(names)), key=lambda k: _h(bA[k]))
    ks = max(range(len(combos)), key=lambda k: (_h(totA[k]), combos[k][2] == combos[k][3]))   # tie -> single threshold

    def min_fp(tot, recall):
        ok = [int(t[2] - t[0]) for t in tot if t[1] and t[0]/t[1] >= recall]
        return min(ok) if ok else None
    res = dict(label='Size-aware threshold, B1 alone, LR2 only. Chosen on A; B exploratory.',
               dump_gpu=ident.get('gpu'), probe=ident.get('probe_sources'),
               verification_note='internal comparison within one dump; Clean NOT evaluated',
               images=dict(A=int(A.sum()), B=int(B.sum())), grid_size=len(combos),
               base_setting=names[kb], locked=dict(geometry=combos[ks][0], cut_px=combos[ks][1],
                                                     t_small=combos[ks][2], t_large=combos[ks][3]))
    for s, tot, bt, m in (('A', totA, bA, A), ('B', totB, bB, B)):
        best, b0 = stats(tot[ks]), stats(bt[kb])
        res[s] = dict(size_aware=best, base=b0, delta_lr2_pp=(best['hmean'] - b0['hmean'])*100,
                      delta_tiny_pp=None if best['tiny_recall'] is None or b0['tiny_recall'] is None
                      else (best['tiny_recall'] - b0['tiny_recall'])*100,
                      base72_min_fp_at_recall=min_fp(bt, best['recall']))
    rA = res['A']
    res['rules_A'] = dict(gain_ge_0p3=rA['delta_lr2_pp'] >= .3,
                          tiny_ge_base_minus_3=rA['delta_tiny_pp'] is not None and rA['delta_tiny_pp'] >= -3,
                          fp_not_above_at_same_recall=rA['base72_min_fp_at_recall'] is None
                          or rA['size_aware']['fp'] <= rA['base72_min_fp_at_recall'])
    res['pass_A'] = all(res['rules_A'].values())
    if B.any():
        rng = np.random.default_rng(431)
        idx = np.where(B)[0]
        diffs = []
        for _ in range(2000):
            j = rng.choice(idx, len(idx))
            diffs.append(float(_h(per[ks, j].sum(0)) - _h(base[kb, j].sum(0)))*100)
        res['boot_B_pp'] = dict(mean=float(np.mean(diffs)), ci95=[float(np.percentile(diffs, 2.5)),
                                                                 float(np.percentile(diffs, 97.5))])
    np.savez_compressed(out_dir/'size_sweep.npz', combos=np.asarray([[GEOMS.index(c[0]), c[1], c[2], c[3]] for c in combos]),
                        A=totA, B=totB, base_names=np.asarray(names), base_A=bA, base_B=bB)
    save_json(out_dir/'size_decision.json', res)
    print(json.dumps(res, indent=1))


def main():
    sys.stdout.reconfigure(errors='backslashreplace')
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dump', required=True, type=Path)
    p.add_argument('--bundle', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    analyze(p.parse_args())


if __name__ == '__main__':
    main()
