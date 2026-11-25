"""Bounded geometric-correction ORACLE on development A (CPU, existing artifacts). DIAGNOSTIC ONLY: uses GT.

Locked before computing (no A number seen):
  * Arm v1 = box-quality v1 FULL (frozen scorer, frozen cut). Its kept set and scores are NOT changed.
  * A kept box j is 'one-to-one eligible' iff (after the evaluator's ROI/ignore filter):
      - j covers >= 50 % of at most one valid GT            (not a merge box)
      - best GT g* has IoU(j, g*) >= .10 and IoU(j, g*) >= 2 x second-best IoU of j   (unambiguous)
      - j is the best-IoU kept box for g*                   (mutual best; duplicates are not eligible)
  * Eligible box -> refined in its own quad frame (keep angle), offsets = GT extent in the frame, clamped to
    |offset| <= BOUND side lengths (primary .5; .25 / 1.0 / none = sensitivity, reported, never substituted).
  * Nothing else changes: no deletion, no new box, no split, scores unchanged. Matching recomputed per image by the
    official evaluator. Contrast arm 'uniform' = best per-box single pixel offset on all edges (unclip-like).
Outputs: docs/handoff/artifacts/janek/refine_oracle_20261125/refine_oracle_janek.json
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding='utf-8')
from mixab import boxquality_devA_janek as dv  # noqa: E402
from mixab import boxquality_janek as bq  # noqa: E402
from mixab import refine_geom_janek as rg  # noqa: E402
from mixab import read_json, save_json  # noqa: E402
from mixab.monkey_janek import monkey_root, use_monkey  # noqa: E402

RUN = ROOT/'runtime/kaggle_downloads/boxquality_devA_T4_20261125-084951/out'
ART = ROOT/'docs/handoff/artifacts/janek/boxquality_v1_T4'
OUT = ROOT/'docs/handoff/artifacts/janek/refine_oracle_20261125'
BOUNDS = {'b0.50': .5, 'b0.25': .25, 'b1.00': 1., 'none': None}
PRIMARY = 'b0.50'
UNIFORM_GRID = [k/10 for k in range(-5, 6)]   # pixel offset = k/10 x short side
GATE = dict(mean=79.0019, clean=80.2202)


def H(tp, gt, pred):
    return 2*tp/(gt + pred) if gt + pred else 0.


def main():
    use_monkey()
    import numpy as np
    from shapely.geometry import Polygon, shape
    from lowres.evaluate_shared import evaluate_image
    from rev4.paired_eda_core import filtered_geometry
    bundle = monkey_root()/'docs/handoff/artifacts/pr_quad_audit_v1'
    rows, A, _ = dv._context(bundle)
    policy = read_json(bundle/'provenance/textvqa_evaluation_protocol.json')
    model = dv.load_model(ART/'model_janek.npz', ART/'boxquality_fit_janek.json')
    recs = {(r['source'], r['condition']): r for r in bq.load_records(RUN/'extract')}

    def poly(p):
        g = Polygon(np.asarray(p, float).reshape(-1, 2))
        return g if g.is_valid else g.buffer(0)

    def iou(a, b):
        inter = a.intersection(b).area
        return inter/(a.area + b.area - inter) if inter else 0.

    res = dict(label='Bounded geometric-correction ORACLE (uses GT; diagnostic upper bound, NOT a method). Split A only.',
               rule=__doc__, sources=len(A), conditions={})
    for c in dv.CONDS:
        tot = defaultdict(lambda: np.zeros(3, dtype=np.int64))
        gained, lost = Counter(), Counter()
        boxcls = Counter()
        status = Counter()
        tp_iou_bins = Counter()
        fixable_gt = Counter()
        per_source = {}
        for s in A:
            row, r = rows[s], recs[(s, c)]
            K = dv.kept_indices(r['raw'], dv.full_scores(model, r['features']), model)['FULL']
            polys = [r['polygons'][i] for i in K]
            scores = [r['packed'][i] for i in K]
            base = evaluate_image(row['gt_clean'], row['roi_clean'], dict(polygons=polys, scores=scores), policy)
            tot['v1'] += base['counts'][:3]
            m0 = set(base['matched_ann_ids'])
            shapes, _ = filtered_geometry(dict(polygons=polys, scores=scores), row, policy)
            gts = [shape(e['geometry']) for e in row['gt_clean']['valid']]
            ann = [str(e['ann_id']) for e in row['gt_clean']['valid']]
            ious = {j: [iou(g, p) if g.intersects(p) else 0. for g in gts] for j, p in shapes.items()}
            covs = {j: [(g.intersection(p).area/g.area) if g.intersects(p) else 0. for g in gts] for j, p in shapes.items()}
            best_box = {}
            for gi in range(len(gts)):
                cand = [(ious[j][gi], -j) for j in shapes if ious[j][gi] > 0]
                if cand:
                    best_box[gi] = -max(cand)[1]
            elig = {}
            for j in shapes:
                v = ious[j]
                if not gts or max(v) <= 0:
                    boxcls['no_gt_overlap'] += 1
                    continue
                order = np.argsort(v)[::-1]
                g1 = int(order[0]); b1 = v[g1]; b2 = v[int(order[1])] if len(order) > 1 else 0.
                if sum(x >= .5 for x in covs[j]) >= 2:
                    boxcls['merge'] += 1
                elif b1 < .1:
                    boxcls['weak_overlap'] += 1
                elif b2*2 > b1:
                    boxcls['ambiguous'] += 1
                elif best_box.get(g1) != j:
                    boxcls['duplicate_not_best'] += 1
                else:
                    elig[j] = g1
                    boxcls['eligible_tp' if b1 >= .5 else 'eligible_nontp'] += 1
                    if b1 >= .5:
                        tp_iou_bins['.5-.6' if b1 < .6 else '.6-.7' if b1 < .7 else '>=.7'] += 1
                    else:
                        fixable_gt['iou<.5 one-to-one'] += 1
            per_source[s] = {}
            for name, bound in list(BOUNDS.items()) + [('uniform', 'uniform')]:
                new = list(polys)
                for j, gi in elig.items():
                    q = rg.quad(polys[j])
                    if q is None:
                        status[(name, 'fallback_not_quad')] += 1
                        continue
                    gpts = np.asarray(gts[gi].exterior.coords)
                    if bound == 'uniform':
                        short = min(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[3] - q[0]))
                        best = (ious[j][gi], polys[j])
                        for k in UNIFORM_GRID:
                            p2, st = rg.apply_offsets(q, rg.uniform_offsets(q, k*short), bound=None)
                            if st == 'ok':
                                v2 = iou(gts[gi], poly(p2))
                                if v2 > best[0] + 1e-12:
                                    best = (v2, p2)
                        new[j] = best[1]
                        status[(name, 'ok')] += 1
                        continue
                    d = rg.target_offsets(q, gpts)
                    if d is None:
                        status[(name, 'fallback_degenerate')] += 1
                        continue
                    new[j], st = rg.apply_offsets(q, d, bound=bound)
                    status[(name, st)] += 1
                o = evaluate_image(row['gt_clean'], row['roi_clean'], dict(polygons=new, scores=scores), policy)
                tot[name] += o['counts'][:3]
                m1 = set(o['matched_ann_ids'])
                gained[name] += len(m1 - m0); lost[name] += len(m0 - m1)
                per_source[s][name] = [int(x) for x in o['counts'][:3]]
                if o['counts'][2] != base['counts'][2]:
                    status[(name, 'pred_count_changed_by_roi_ignore')] += o['counts'][2] - base['counts'][2]
            per_source[s]['v1'] = [int(x) for x in base['counts'][:3]]
        res['conditions'][c] = dict(
            arms={k: dict(tp=int(v[0]), gt=int(v[1]), pred=int(v[2]), fp=int(v[2] - v[0]), hmean=H(*v)*100) for k, v in tot.items()},
            gt_gained=dict(gained), gt_lost=dict(lost), kept_box_classes=dict(boxcls),
            eligible_tp_iou_bins=dict(tp_iou_bins), status={f'{a}|{b}': n for (a, b), n in status.items()})
        res['conditions'][c]['_per_source'] = per_source
        print(c, json.dumps({k: round(v['hmean'], 3) for k, v in res['conditions'][c]['arms'].items()}), dict(boxcls), flush=True)
    C, L = res['conditions']['clean']['arms'], res['conditions']['lr2']['arms']
    res['summary'] = {k: dict(clean=C[k]['hmean'], lr2=L[k]['hmean'], mean=(C[k]['hmean'] + L[k]['hmean'])/2,
                              clean_vs_gate=C[k]['hmean'] - GATE['clean'], mean_vs_gate=(C[k]['hmean'] + L[k]['hmean'])/2 - GATE['mean'])
                      for k in C}
    res['primary'] = PRIMARY
    # paired bootstrap by source for primary - v1 (mean of Clean/LR2 Hmean)
    rng = np.random.default_rng(431)
    src = list(A)
    arr = {k: np.array([[res['conditions'][c]['_per_source'][s][k] for c in dv.CONDS] for s in src]) for k in ('v1', PRIMARY)}
    def mean_h(a):
        t = a.sum(0)
        return np.mean([H(*t[i]) for i in range(2)])*100
    obs = mean_h(arr[PRIMARY]) - mean_h(arr['v1'])
    bs = []
    for _ in range(2000):
        ix = rng.integers(0, len(src), len(src))
        bs.append(mean_h(arr[PRIMARY][ix]) - mean_h(arr['v1'][ix]))
    res['primary_minus_v1_mean'] = dict(delta=obs, ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
    OUT.mkdir(parents=True, exist_ok=True)
    save_json(OUT/'refine_oracle_janek.json', res)
    print(json.dumps(res['summary'], indent=1), json.dumps(res['primary_minus_v1_mean']))


if __name__ == '__main__':
    main()
