"""Descriptive (post-hoc, does not change the locked label): per-source net contribution of FULL vs F on A; share of the top sources and of the dense-text page. Run from the workspace root."""
import json, sys
from pathlib import Path
ROOT = Path.cwd(); sys.path.insert(0, str(ROOT))
from mixab import boxquality_devA_janek as dv, boxquality_janek as bq, phase0_janek as p0, read_json
from mixab.monkey_janek import monkey_root, use_monkey
use_monkey()
from lowres.evaluate_shared import evaluate_image
B = monkey_root()/'docs/handoff/artifacts/pr_quad_audit_v1'
rows, A, _ = dv._context(B); policy = read_json(B/'provenance/textvqa_evaluation_protocol.json')
m = dv.load_model(ROOT/'docs/handoff/artifacts/janek/boxquality_v1_T4/model_janek.npz', ROOT/'docs/handoff/artifacts/janek/boxquality_v1_T4/boxquality_fit_janek.json')
recs = {(r['source'], r['condition']): r for r in bq.load_records(ROOT/'runtime/kaggle_downloads/boxquality_devA_T4_20261125-084951/out/extract')}
out = {}
for c in dv.CONDS:
    rowsF, rowsU = [], []
    for s in A:
        r = recs[(s, c)]; k = dv.kept_indices(r['raw'], dv.full_scores(m, r['features']), m)
        res = []
        for arm in ('F', 'FULL'):
            idx = k[arm]
            o = evaluate_image(rows[s]['gt_clean'], rows[s]['roi_clean'], dict(polygons=[r['polygons'][i] for i in idx], scores=[r['packed'][i] for i in idx]), policy)
            res.append(o['counts'][:3])
        rowsF.append(res[0]); rowsU.append(res[1])
    hF = bq.hmean([sum(x[i] for x in rowsF) for i in range(3)])
    net = p0.net_contributions([u[0]-f[0] for f, u in zip(rowsF, rowsU)], [u[2]-f[2] for f, u in zip(rowsF, rowsU)], hF)
    order = sorted(range(len(A)), key=lambda i: -net[i])
    page = A.index('6e340fcbb1e0a877') if '6e340fcbb1e0a877' in A else None
    keep = [i for i in range(len(A)) if i != page]
    d_all = (bq.hmean([sum(rowsU[i][j] for i in range(len(A))) for j in range(3)]) - hF)*100
    d_wo = (bq.hmean([sum(rowsU[i][j] for i in keep) for j in range(3)]) - bq.hmean([sum(rowsF[i][j] for i in keep) for j in range(3)]))*100
    out[c] = dict(delta_pp=d_all, delta_without_dense_page_pp=d_wo, dense_page_net_share=net[page]/sum(net) if page is not None else None,
                  top10=[dict(source=A[i], dtp=rowsU[i][0]-rowsF[i][0], dpred=rowsU[i][2]-rowsF[i][2], gt=rowsF[i][1]) for i in order[:10]],
                  top10_share=sum(net[i] for i in order[:10])/sum(net), bottom5=[dict(source=A[i], dtp=rowsU[i][0]-rowsF[i][0], dpred=rowsU[i][2]-rowsF[i][2]) for i in order[-5:]])
Path(ROOT/'docs/handoff/artifacts/janek/boxquality_devA_20261125').mkdir(parents=True, exist_ok=True)
json.dump(dict(label='Descriptive concentration of FULL - F on A (post-hoc, does not change the locked label).', **out),
          open(ROOT/'docs/handoff/artifacts/janek/boxquality_devA_20261125/concentration_janek.json', 'w', encoding='utf-8'), indent=1)
for c, v in out.items():
    print(c, round(v['delta_pp'], 3), 'w/o page', round(v['delta_without_dense_page_pp'], 3), 'page share', v['dense_page_net_share'], 'top10', round(v['top10_share'], 3))
    print('  top', [(x['source'][:8], x['dtp'], x['dpred'], x['gt']) for x in v['top10']])
    print('  bottom', [(x['source'][:8], x['dtp'], x['dpred']) for x in v['bottom5']])
