"""EXPLORATORY follow-up (post-hoc, after seeing decision.json): paired source bootstrap for the
"B1 + B-Clean-only boxes" variant vs B1 alone, on A and B. The variant was NOT the pre-registered pick
(that pick failed the tiny rule); it is the best combo on A among those passing the tiny rule.
Output: results/dualagree_v1_T4_followup.json
"""
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mixab import dualagree as da  # noqa: E402
from mixab import reselect_split as rs  # noqa: E402

DUMP = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('H:/My Drive/CS431_MixA/runs/dualagree_dump_v1_T4')
BUNDLE = ROOT.parents[1]/'MONKEY_V2/docs/handoff/artifacts/pr_quad_audit_v1'
VARIANT = (.5, .7, da.INF, .5)        # t_l, t_c, t_a, theta
BASE = (.5, da.INF, da.INF, .3)


def h(t):
    return float(Fraction(2*int(t[0]), int(t[1] + t[2])))


def main():
    rows, policy, lr2, geoms, _ = da._context(BUNDLE)
    ident = json.loads((DUMP/'dump_identity.json').read_text(encoding='utf-8'))['identity_sha256']
    dl = da.load_model_dump(DUMP/'B1', len(geoms), ident)
    dc = da.load_model_dump(DUMP/'B-Clean', len(geoms), ident)
    gl, gc = geoms.index(da.GEOM_L), geoms.index(da.GEOM_C)
    sids = sorted(dl, key=lambda s: lr2[s])
    split = np.array([rs.split_of(lr2[s]) for s in sids])
    v, b = [], []
    for s in sids:
        pool = da.PoolScorer(dl[s][gl], dc[s][gc], rows[lr2[s]], policy)
        v.append(pool.fused(*VARIANT)); b.append(pool.fused(*BASE))
    v, b = np.array(v), np.array(b)
    rng = np.random.default_rng(431)
    out = dict(label='EXPLORATORY post-hoc; T4 dump (PENDING_REFERENCE_CHECK)', variant=dict(t_l=.5, t_c=.7, theta=.5),
               base=dict(t_l=.5))
    for name in ('A', 'B'):
        idx = np.where(split == name)[0]
        d = []
        for _ in range(2000):
            j = rng.choice(idx, len(idx))
            d.append((h(v[j].sum(0)) - h(b[j].sum(0)))*100)
        tv, tb = v[idx].sum(0), b[idx].sum(0)
        out[name] = dict(delta_pp=(h(tv) - h(tb))*100, ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                         tp=[int(tv[0]), int(tb[0])], fp=[int(tv[2] - tv[0]), int(tb[2] - tb[0])],
                         tiny=[float(tv[4]/tv[3]), float(tb[4]/tb[3])],
                         sources_improved=int(((2*v[idx, 0] - (v[idx, 1] + v[idx, 2])) > (2*b[idx, 0] - (b[idx, 1] + b[idx, 2]))).sum()))
    (ROOT/'results/dualagree_v1_T4_followup.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
