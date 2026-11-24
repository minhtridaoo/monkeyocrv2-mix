"""P0b — CPU, counts only. Ceiling of ANY per-image router between B-Clean and B1.

Settings are locked per expert (same as p0_fixed oracle: chosen on A). For each condition, each image picks
the expert that maximises corpus Hmean = 2*sum(match) / (sum(gt) + sum(det)). This 0/1 fractional problem is
solved exactly with Dinkelbach: for fixed h, pick per image argmax(2*match - h*det); iterate to convergence.
It uses GT, so it is an UPPER BOUND, not a method. Output: results/diag_v2_cpu/p0b_per_image_oracle.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mixab import reselect_split as rs  # noqa: E402

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('H:/My Drive/CS431_MixA/runs/merge_audit_v1_A100-SXM4-40GB')


def best_choice(M, G, D):
    """M, D: [n_img, 2 experts]; G: [n_img] (same GT for both). Returns choice [n_img], hmean."""
    h = 0.0
    for _ in range(100):
        ch = np.argmax(2*M - h*D, axis=1)
        m = M[np.arange(len(ch)), ch].sum(); d = D[np.arange(len(ch)), ch].sum()
        h_new = 2*m/(G.sum() + d)
        if abs(h_new - h) < 1e-15:
            break
        h = h_new
    return ch, float(h_new)


def main():
    p0 = json.loads((ROOT/'results/diag_v2_cpu/p0_fixed.json').read_text(encoding='utf-8'))
    kb_key, k1_key = p0['oracle_cond']['settings']['bclean'], p0['oracle_cond']['settings']['b1']
    a = rs.load_counts(RUN/'B-Clean_source_counts.npz'); b = rs.load_counts(RUN/'B1_source_counts.npz')
    assert list(a['source_ids']) == list(b['source_ids'])
    keys = [rs.setting_key(s) for s in a['settings']]
    kb, k1 = keys.index(kb_key), keys.index(k1_key)
    split = np.array([rs.split_of(s) for s in a['source_ids']])
    out = {'label': 'UPPER BOUND using GT; settings locked on A', 'settings': dict(bclean=kb_key, b1=k1_key)}
    for s in 'AB':
        ix = np.where(split == s)[0]
        res = {}
        for c, cname in ((0, 'clean'), (1, 'lr2')):
            X = np.stack([a['counts'][ix, c, kb], b['counts'][ix, c, k1]], axis=1).astype(np.float64)  # [n, 2, F]
            M, D, G = X[:, :, 0], X[:, :, 2], X[:, 0, 1]
            assert np.array_equal(X[:, 0, 1], X[:, 1, 1])
            ch, h = best_choice(M, G, D)
            tie = (M[:, 0] == M[:, 1]) & (D[:, 0] == D[:, 1])
            res[cname] = dict(hmean=h, frac_to_b1=float(ch.mean()), frac_identical_counts=float(tie.mean()),
                              frac_to_b1_among_nonidentical=float(ch[~tie].mean()) if (~tie).any() else None,
                              cond_oracle_hmean=p0['oracle_cond'][s][cname])
        res['mean'] = (res['clean']['hmean'] + res['lr2']['hmean'])/2
        res['gain_over_cond_oracle_pp'] = (res['mean'] - p0['oracle_cond'][s]['mean'])*100
        res['gain_over_best_single_pp'] = (res['mean'] - p0['margin'][s]['best_single'])*100
        out[s] = res
    dest = ROOT/'results/diag_v2_cpu/p0b_per_image_oracle.json'
    dest.write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
