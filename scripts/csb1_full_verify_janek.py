"""CS-B1@0.7 verification on the full 2,262-source remaining-validation set (LR2), from EXISTING artifacts only.

Light CPU: sums cached per-source count rows (runtime/phase0_cache/phase0_records.jsonl.gz, produced by
mixab.phase0_janek stats from the T4 dump) and the 72-setting T4 recompute (t4_counts_B1.npz). No new inference.
Checks: source IDs (A u B = bundle cohort = dump = A2r npz), configuration identity, per-source counts reproduce
the published LR2 totals of B1 at Tuan's .40/.60. Reports full-set Hmean from SUMMED counts and paired bootstrap
CIs over all 2,262 sources, vs (a) B1 epsilon_005/0.50 (the per-split baseline used before) and (b) B1 .40/.60
(Tuan's calibrated operating point, mask .40/score .60/eps .01). Output: docs/handoff/artifacts/janek/csb1_full_v1_T4/csb1_full_janek.json
"""
import csv
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np  # noqa: E402
from mixab import merge_audit_janek as ma, phase0_janek as p0, reselect_split_janek as rs # noqa: E402
from mixab.monkey_janek import monkey_root  # noqa: E402

BUNDLE = monkey_root()/'docs/handoff/artifacts/pr_quad_audit_v1'
CACHE = ROOT/'runtime/phase0_cache'
INP = ROOT/'runtime/phase0_inputs'
OUT = ROOT/'docs/handoff/artifacts/janek/csb1_full_v1_T4'


def boot(new, old, seed=431, n=2000):
    return p0.paired_boot(new, old, seed=seed, n=n)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (BUNDLE/'data/source_cohort.csv').open(encoding='utf-8', newline='') as f:
        cohort = sorted(r['source_id'] if 'source_id' in r else r[next(iter(r))] for r in csv.DictReader(f))
    rows = ma.load_rows(BUNDLE)
    recs = [json.loads(l) for l in gzip.open(CACHE/'phase0_records.jsonl.gz', 'rt', encoding='utf-8')]
    srcs = sorted(r['source'] for r in recs)
    a2r = rs.load_counts(INP/'a2r/B1_source_counts.npz')
    ident = json.loads((INP/'dualagree_dump_v1_T4/dump_identity.json').read_text(encoding='utf-8'))
    split = {s: rs.split_of(s) for s in srcs}
    checks = dict(
        cohort_sources=len(cohort), cohort_equals_gt_manifest=sorted(rows) == cohort,
        records_sources=len(srcs), records_unique=len(set(srcs)) == len(srcs),
        records_equal_cohort=srcs == cohort, a2r_equal_cohort=sorted(a2r['source_ids']) == cohort,
        split_A=sum(v == 'A' for v in split.values()), split_B=sum(v == 'B' for v in split.values()),
        dump=dict(gpu=ident['gpu'], monkey_v2=ident['monkey_v2']['commit'], gt_manifest_sha256=ident['gt_manifest_sha256'],
                  bundle_manifest_sha256=ident['bundle_manifest_sha256'], postprocessor_sha256=ident['installed_db_postprocessor_sha256'],
                  merge_manifest_sha256=ident['merge_manifest_sha256'], model_config_sha256=ident['model_config_sha256'],
                  torch=ident['torch'], condition=ident['condition']))
    by = {r['source']: r for r in recs}
    order = cohort
    base = np.array([by[s]['rows']['B1'] for s in order])
    cs = np.array([by[s]['rows']['CSB1'] for s in order])
    with np.load(CACHE/'t4_counts_B1.npz') as z:
        idx = {s: i for i, s in enumerate(z['source_ids'])}
        c72 = z['counts'][[idx[s] for s in order]]
    keys = [rs.setting_key(x) for x in a2r['settings']]
    k_eps, k_tuan = keys.index(p0.SET_B1), keys.index(p0.SET_TUAN)
    require_same = (c72[:, k_eps] == base).all()
    tuan_t4 = c72[:, k_tuan]
    aidx = {s: i for i, s in enumerate(a2r['source_ids'])}
    tuan_a100 = a2r['counts'][[aidx[s] for s in order], 1, k_tuan]
    checks['b1_eps005_rows_equal_72grid_recompute'] = bool(require_same)
    checks['b1_040_060_a100_full_hmean_pct'] = p0.hmean(tuan_a100.sum(0))*100
    checks['b1_040_060_a100_full_fp'] = int(tuan_a100.sum(0)[2] - tuan_a100.sum(0)[0])
    checks['note_tuan_published'] = 'Tuan handoff.md:56 B1 .40/.60 LR2 = 76,90%, FP 11.997 (2,262 sources)'
    res = dict(label='CS-B1@0.7 on the full 2,262 remaining-validation sources (LR2 only), development data already '
                     'seen many times; T4 dump -> PENDING_REFERENCE_CHECK. Hmean from summed counts.',
               checks=checks, rows={})
    for name, m in (('ALL', np.ones(len(order), bool)), ('A', np.array([split[s] == 'A' for s in order])),
                    ('B', np.array([split[s] == 'B' for s in order]))):
        t_cs, t_b, t_tu = cs[m].sum(0), base[m].sum(0), tuan_t4[m].sum(0)
        res['rows'][name] = dict(
            sources=int(m.sum()), csb1=p0.totals(t_cs), b1_eps005_050=p0.totals(t_b), b1_040_060_t4=p0.totals(t_tu),
            csb1_minus_b1_eps005=boot(cs[m], base[m]), csb1_minus_b1_040_060=boot(cs[m], tuan_t4[m]),
            tiny_pp_vs_eps005=(p0.totals(t_cs)['tiny_recall'] - p0.totals(t_b)['tiny_recall'])*100,
            tiny_pp_vs_040_060=(p0.totals(t_cs)['tiny_recall'] - p0.totals(t_tu)['tiny_recall'])*100)
    (OUT/'csb1_full_janek.json').write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
