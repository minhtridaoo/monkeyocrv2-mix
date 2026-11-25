"""Descriptive control for the Flip-TTA diagnosis (post hoc, NOT a configuration choice): compare T(.60) with F at the
cutoff whose summed recall (or FP) is closest to T's, on a fine grid .580-.700 step .002, per condition. Paired bootstrap
over sources (2,000, seed 431). Output: docs/handoff/artifacts/janek/fliptta_diag_v1/matched_control_janek.json"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np  # noqa: E402
from mixab import dualagree_janek as da, fliptta_janek as ft, merge_audit_janek as ma, phase0_janek as p0 # noqa: E402
from mixab.monkey_janek import monkey_root  # noqa: E402
rows, policy, *_ = da._context(monkey_root()/'docs/handoff/artifacts/pr_quad_audit_v1')
from rev4.decision_core import CandidateMatcher  # noqa: E402
recs = ft.load_records(ROOT/'runtime/kaggle_downloads/fliptta_pilot_1/out/fliptta_infer')
grid = [round(.58 + .002*k, 3) for k in range(61)]
out = {}
for cond in ('lr2', 'clean'):
    rr = sorted((r for r in recs if r['condition'] == cond), key=lambda r: r['source'])
    Fg = np.zeros((len(rr), len(grid), 11), dtype=np.int64); T = np.zeros((len(rr), 11), dtype=np.int64)
    for i, r in enumerate(rr):
        mF = CandidateMatcher(dict(polygons=r['F']['polygons'], scores=r['F']['packed']), r['F']['raw'], rows[r['source']], policy)
        for k, c in enumerate(grid):
            Fg[i, k] = ma.count_row(mF.score(c))
        e = r['T']
        T[i] = ma.count_row(CandidateMatcher(dict(polygons=e['polygons'], scores=e['packed']), e['raw'], rows[r['source']], policy).score(.6))
    tT = T.sum(0); recT, fpT = tT[0]/tT[1], tT[2] - tT[0]
    tot = Fg.sum(0)
    rec = tot[:, 0]/tot[:, 1]; fp = tot[:, 2] - tot[:, 0]
    k_r = int(np.argmin(np.abs(rec - recT))); k_f = int(np.argmin(np.abs(fp - fpT)))
    out[cond] = dict(T60=p0.totals(tT), T_recall=float(recT),
                     F_recall_matched=dict(cut=grid[k_r], F=p0.totals(tot[k_r]), recall=float(rec[k_r]), T_minus_F=p0.paired_boot(T, Fg[:, k_r])),
                     F_fp_matched=dict(cut=grid[k_f], F=p0.totals(tot[k_f]), recall=float(rec[k_f]), T_minus_F=p0.paired_boot(T, Fg[:, k_f])),
                     F_best_on_grid=dict(cut=grid[int(np.argmax([p0.hmean(t) for t in tot]))], hmean=max(p0.hmean(t) for t in tot)),
                     F_at_060=p0.totals(tot[grid.index(.6)]))
Path(ROOT/'docs/handoff/artifacts/janek/fliptta_diag_v1/matched_control_janek.json').write_text(json.dumps(dict(
    label='Post-hoc descriptive control on the 300-source pilot sample (already used): T(.60) vs F at a recall- or FP-matched '
          'cutoff. Not a configuration choice; F .40/.60 remains the locked baseline.', grid=[grid[0], grid[-1], .002], **out),
    indent=1), encoding='utf-8')
print(json.dumps(out, indent=1))
