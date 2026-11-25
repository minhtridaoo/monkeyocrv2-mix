"""TRAIN-side statistics for learned box refinement (CPU, box-quality pilot extraction; NO P-map, NO A/B rows).
For the frozen v1 FULL kept set on TRAIN fit/calibration: eligible one-to-one pairs, target-offset distribution
(to check the locked bound .5 / drop rule 1.0 against TRAIN), and the same bounded oracle on TRAIN-calibration.
Output: docs/handoff/artifacts/janek/refine_oracle_20261125/train_stats_janek.json
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
from mixab import refine_janek as rf  # noqa: E402
from mixab import read_json, save_json  # noqa: E402
from mixab.monkey_janek import monkey_root, use_monkey  # noqa: E402

RUN = ROOT/'runtime/kaggle_downloads/boxquality_T4_20261124-224054/out/extract'
ROWS = ROOT/'runtime/kaggle_upload/cs431-boxquality/inputs/train_rows.jsonl.gz'
ART = ROOT/'docs/handoff/artifacts/janek/boxquality_v1_T4'
OUT = ROOT/'docs/handoff/artifacts/janek/refine_oracle_20261125'


def main():
    use_monkey()
    import numpy as np
    from shapely.geometry import shape
    from lowres.evaluate_shared import evaluate_image
    from rev4.paired_eda_core import filtered_geometry
    policy = read_json(monkey_root()/'docs/handoff/artifacts/pr_quad_audit_v1/provenance/textvqa_evaluation_protocol.json')
    rows = bq.load_rows(ROWS)
    model = dv.load_model(ART/'model_janek.npz', ART/'boxquality_fit_janek.json')
    recs = bq.load_records(RUN)
    cls = defaultdict(Counter)
    tg = defaultdict(list)
    tot = defaultdict(lambda: np.zeros(3, dtype=np.int64))
    for r in recs:
        s, c = r['source'], r['condition']
        row = rows[s]
        fold = row['role']
        if fold not in ('fit', 'calibration'):
            continue
        K = dv.kept_indices(r['raw'], dv.full_scores(model, r['features']), model)['FULL']
        polys = [r['polygons'][i] for i in K]
        scores = [r['packed'][i] for i in K]
        shapes, _ = filtered_geometry(dict(polygons=polys, scores=scores), row, policy)
        gts = [shape(e['geometry']) for e in row['gt_clean']['valid']]

        def iou(a, b):
            inter = a.intersection(b).area
            return inter/(a.area + b.area - inter) if inter else 0.
        ious = {j: [iou(g, p) if g.intersects(p) else 0. for g in gts] for j, p in shapes.items()}
        covs = {j: [(g.intersection(p).area/g.area) if g.intersects(p) else 0. for g in gts] for j, p in shapes.items()}
        pairs, k = rf.eligible(ious, covs)
        for a, n in k.items():
            cls[f'{fold}|{c}'][a] += n
        new = list(polys)
        for j, gi in pairs.items():
            q = rg.quad(polys[j])
            d = None if q is None else rg.target_offsets(q, np.asarray(gts[gi].exterior.coords))
            if d is None:
                cls[f'{fold}|{c}']['target_fallback'] += 1
                continue
            tg[f'{fold}|{c}|{"tp" if ious[j][gi] >= .5 else "nontp"}'].append(d.tolist())
            if fold == 'calibration':
                new[j], _ = rg.apply_offsets(q, d)
        if fold == 'calibration':
            for arm, P in (('v1', polys), ('oracle_b0.50', new)):
                o = evaluate_image(row['gt_clean'], row['roi_clean'], dict(polygons=P, scores=scores), policy)
                tot[f'{c}|{arm}'] += o['counts'][:3]
    res = dict(label='TRAIN-only statistics (fit + calibration), v1 FULL kept set; oracle uses GT (diagnostic).',
               classes={k: dict(v) for k, v in cls.items()}, targets={})
    for k, v in sorted(tg.items()):
        a = np.abs(np.asarray(v))
        res['targets'][k] = dict(n=len(v), median_abs=np.median(a, 0).tolist(), p90_abs=np.percentile(a, 90, 0).tolist(),
                                 frac_any_gt_0_5=float((a.max(1) > .5).mean()), frac_any_gt_1=float((a.max(1) > 1.).mean()))
    res['calibration_oracle'] = {k: dict(tp=int(v[0]), gt=int(v[1]), pred=int(v[2]), hmean=2*v[0]/(v[1] + v[2])*100) for k, v in tot.items()}
    save_json(OUT/'train_stats_janek.json', res)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
