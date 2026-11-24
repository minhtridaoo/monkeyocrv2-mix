"""CS-B1@0.7 on CLEAN images + router-free system (CPU; runs on Kaggle after the Clean dump).
Spec: docs/csb1_clean_check_spec_20261123.md. Rule unchanged: mixab.dualagree.LOCKED (B1 epsilon_005 >= .5 plus
B-Clean mask_040 >= .7 boxes not overlapping a kept B1 box at IoU >= .5).

Inputs: the new Clean dump (dualagree dump --condition clean) and the existing LR2 T4 dump. Every comparison uses
counts recomputed from the SAME dumps (same GPU numerics) with Tuan's CandidateMatcher.
Reports, on ALL 2,262 sources and on A/B: Clean and LR2 Hmean, TP/FP/FN, tiny recall, paired bootstrap CIs, and
the router-free system mean (CS-B1 applied to every image) versus single models at their published setting.
"""
import argparse
import json
import sys
from pathlib import Path

from mixab import read_json, require, save_json
from mixab import dualagree as da
from mixab import phase0 as p0

SINGLES = {'B1 .40/.60': ('B1', 'mask_040', .6), 'B1 eps005/.50': ('B1', 'epsilon_005', .5),
           'B-Clean .40/.60': ('B-Clean', 'mask_040', .6)}


def condition_rows(dump_dir, prefix, rows, policy, geoms):
    """{source: dict(CSB1=row, <single>=row, ...)} for one condition's dump."""
    from rev4.decision_core import CandidateMatcher
    from mixab import merge_audit as ma
    ident = read_json(Path(dump_dir)/'dump_identity.json')['identity_sha256']
    dl = da.load_model_dump(Path(dump_dir)/'B1', len(geoms), ident)
    dc = da.load_model_dump(Path(dump_dir)/'B-Clean', len(geoms), ident)
    require(set(dl) == set(dc), 'B1 and B-Clean dumps differ')
    gl, gc = geoms.index(da.GEOM_L), geoms.index(da.GEOM_C)
    out = {}
    for sid in dl:
        require(sid.startswith(prefix + '::'), f'Unexpected sample id {sid}')
        src = sid.split('::', 1)[1]
        row, r = rows[src], {}
        pool = da.PoolScorer(dl[sid][gl], dc[sid][gc], row, policy)
        r['CSB1'] = pool.fused(*p0.CSB1)
        for name, (model, geom, cut) in SINGLES.items():
            e = (dl if model == 'B1' else dc)[sid][geoms.index(geom)]
            r[name] = ma.count_row(CandidateMatcher(dict(polygons=e['polygons'], scores=e['packed']), e['raw'], row, policy).score(cut))
        out[src] = r
    return out, read_json(Path(dump_dir)/'dump_identity.json')


def evaluate(a):
    import numpy as np
    from mixab import reselect_split as rs
    rows, policy, lr2, geoms, scores = da._context(a.bundle)
    clean, id_c = condition_rows(a.clean_dump, 'clean', rows, policy, geoms)
    low, id_l = condition_rows(a.lr2_dump, 'lr2', rows, policy, geoms)
    require(sorted(clean) == sorted(low) == sorted(rows), 'Both dumps must cover all 2,262 sources')
    srcs = sorted(rows)
    split = np.array([rs.split_of(s) for s in srcs])
    arms = ['CSB1'] + list(SINGLES)
    res = dict(label='CS-B1@0.7 Clean + router-free system, development data (already seen); T4 -> PENDING_REFERENCE_CHECK',
               dumps=dict(clean=dict(gpu=id_c['gpu'], condition=id_c['condition'], identity=id_c['identity_sha256']),
                          lr2=dict(gpu=id_l['gpu'], condition=id_l['condition'], identity=id_l['identity_sha256'])),
               sources=len(srcs), splits={})
    for name, m in (('ALL', np.ones(len(srcs), bool)), ('A', split == 'A'), ('B', split == 'B')):
        sub = [s for s, k in zip(srcs, m) if k]
        C = {arm: np.array([clean[s][arm] for s in sub]) for arm in arms}
        L = {arm: np.array([low[s][arm] for s in sub]) for arm in arms}
        entry = dict(sources=len(sub), clean={}, lr2={}, mean={})
        for arm in arms:
            entry['clean'][arm] = p0.totals(C[arm].sum(0))
            entry['lr2'][arm] = p0.totals(L[arm].sum(0))
            entry['mean'][arm] = (entry['clean'][arm]['hmean'] + entry['lr2'][arm]['hmean'])/2
        for ref in SINGLES:
            entry[f'CSB1_minus_{ref}'] = dict(clean=p0.paired_boot(C['CSB1'], C[ref]), lr2=p0.paired_boot(L['CSB1'], L[ref]),
                                              mean_pp=(entry['mean']['CSB1'] - entry['mean'][ref])*100)
        best = max(SINGLES, key=lambda k: entry['mean'][k])
        entry['best_single_by_mean'] = best
        entry['router_free_system_minus_best_single_pp'] = (entry['mean']['CSB1'] - entry['mean'][best])*100
        entry['clean_csb1_minus_bclean_pp'] = (entry['clean']['CSB1']['hmean'] - entry['clean']['B-Clean .40/.60']['hmean'])*100
        entry['tiny_lr2_csb1_minus_b1_040_060_pp'] = (entry['lr2']['CSB1']['tiny_recall'] - entry['lr2']['B1 .40/.60']['tiny_recall'])*100
        res['splits'][name] = entry
    A = res['splits']['A']
    res['group_gate_A_descriptive'] = dict(
        mean_ge_best_single_plus_0p5=A['router_free_system_minus_best_single_pp'] >= .5,
        clean_ge_bclean_minus_0p3=A['clean_csb1_minus_bclean_pp'] >= -.3,
        tiny_ge_b1_minus_3=A['tiny_lr2_csb1_minus_b1_040_060_pp'] >= -3,
        note='Router-free system (CS-B1 on every image). Descriptive: CS-B1 was chosen after seeing A, and A/B were '
             'used many times; this cannot confirm an improvement.')
    save_json(Path(a.output)/'csb1_clean_result.json', res)
    print(json.dumps({k: res['splits'][k]['mean'] for k in res['splits']}, indent=1))
    print(json.dumps(res['group_gate_A_descriptive'], indent=1))


def main():
    sys.stdout.reconfigure(errors='backslashreplace')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bundle', required=True, type=Path)
    ap.add_argument('--clean-dump', required=True, type=Path)
    ap.add_argument('--lr2-dump', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    evaluate(a)


if __name__ == '__main__':
    main()
