"""ROI/ignore audit of the box-quality TRAIN label rule (spec docs/research/janek/boxquality_devA_spec_20261125_janek.md §8). CPU.

The scorer was labelled with ROI = full canvas (mixab.boxquality_janek.train_row) because the historical TRAIN ROI
(perspective field of view, job 8323 metadata) is HPC-only. B1 itself was trained from the frozen TRAIN manifest whose
ROI is that historical field of view; the valid/ignore rule is the same gt_for_source + DEFAULT_SHARED_CONFIG.

The TRAIN manifest cannot be compared directly here, so this ESTIMATES the candidate-label disagreement on development
split A, where the locked ROI exists: B1 mask_040 candidates (raw >= .20) from the existing T4 dumps (Clean + LR2) are
labelled with the scorer's own rule (task_labels) twice, with the locked row and with a full-canvas row rebuilt by the
vendored gt_for_source. Reads A labels only to measure label noise; no scorer, cutoff or model is touched.

Materiality (locked in the spec): weighted disagreement >= 2.0 % -> report to the user BEFORE running the notebook.
Run from the workspace root:  python scripts/boxquality_devA_roi_audit_janek.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding='utf-8')
from mixab import boxquality_janek as bq  # noqa: E402
from mixab import dualagree_janek as da  # noqa: E402
from mixab import merge_audit_janek as ma  # noqa: E402
from mixab import read_json, save_json  # noqa: E402
from mixab import reselect_split_janek as rs  # noqa: E402
from mixab.monkey_janek import monkey_root, use_monkey  # noqa: E402
from mixab.vendor import shared_manifest_janek as sm  # noqa: E402

TEXTOCR = ROOT/'runtime/external/textocr/TextOCR_0.1_train.json'
DUMPS = dict(lr2=ROOT/'runtime/phase0_inputs/dualagree_dump_v1_T4',
             clean=ROOT/'runtime/kaggle_downloads/csb1_clean_T4_20261123-110923/out/clean_dump')
OUT = ROOT/'docs/handoff/artifacts/janek/boxquality_devA_20261125/roi_audit_janek.json'
THRESHOLD = .02


def name(v):
    return {1: 'pos', 0: 'neg', None: 'dropped'}[v]


def main():
    use_monkey()
    from rev4.decision_plan import GEOMETRIES
    bundle = monkey_root()/'docs/handoff/artifacts/pr_quad_audit_v1'
    assert bq.sha256_file(TEXTOCR) == bq.TEXTOCR_TRAIN_SHA256, 'TextOCR SHA differs'
    rows = ma.load_rows(bundle)
    policy = read_json(bundle/'provenance/textvqa_evaluation_protocol.json')
    A = sorted(s for s in rows if rs.split_of(s) == 'A')
    textocr, _ = sm.normalize_textocr(json.loads(TEXTOCR.read_text(encoding='utf-8')))
    gi = [g[0] for g in GEOMETRIES].index(bq.GEOM[0])
    res = dict(label='ROI/ignore audit of the TRAIN label rule, estimated on development A (T4 dumps). Label noise only.',
               spec='docs/research/janek/boxquality_devA_spec_20261125_janek.md §8', sources=len(A), threshold=THRESHOLD,
               conditions={})
    tot_w = tot_dis = 0.
    for cond, folder in DUMPS.items():
        ident = read_json(folder/'dump_identity.json')
        assert ident['condition'] == cond, ident['condition']
        dump = da.load_model_dump(folder/'B1', len(GEOMETRIES), ident['identity_sha256'])
        table, w_all, w_dis, tasks_changed, cands = Counter(), 0., 0., 0, 0
        for s in A:
            e = dump[f'{cond}::{s}'][gi]
            rec = dict(polygons=e['polygons'], packed=e['packed'], raw=e['raw'])
            lock, _ = bq.task_labels(rec, rows[s], policy)
            W, H = rows[s]['clean_size']
            full, _ = bq.task_labels(rec, bq.train_row(s, textocr, W, H, sm), policy)
            used = [i for i, v in enumerate(full) if v is not None]       # the fit weighted the full-canvas rule
            wt = .5/len(used) if used else 0.
            changed = False
            for i in range(len(rec['raw'])):
                table[f'{name(full[i])}(full)->{name(lock[i])}(locked)'] += 1
                if full[i] is not None:
                    w_all += wt
                    if lock[i] != full[i]:
                        w_dis += wt
                if lock[i] != full[i]:
                    changed = True
            cands += len(rec['raw'])
            tasks_changed += changed
        res['conditions'][cond] = dict(candidates=cands, contingency=dict(sorted(table.items())),
                                       tasks_with_any_change=tasks_changed, weighted_disagreement=w_dis/w_all,
                                       dump_gpu=ident.get('gpu'))
        tot_w += w_all; tot_dis += w_dis
        print(cond, json.dumps(res['conditions'][cond], indent=1), flush=True)
    res['weighted_disagreement_overall'] = tot_dis/tot_w
    res['material'] = res['weighted_disagreement_overall'] >= THRESHOLD
    res['note'] = ('Weight = fit weighting under the full-canvas rule (0.5 per task over its labelled candidates). '
                   'Disagreement = a labelled candidate whose label under the locked ROI is different (1<->0) or '
                   'dropped by the locked ROI/ignore filter. Estimate for TRAIN by inference (same generator); not '
                   'verified on TRAIN, whose historical ROI is HPC-only.')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_json(OUT, res)
    print(json.dumps({k: res[k] for k in ('weighted_disagreement_overall', 'material')}, indent=1))


if __name__ == '__main__':
    main()
