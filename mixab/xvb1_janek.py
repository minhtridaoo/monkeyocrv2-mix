"""XV-B1 pilot (CPU, T4 dump, split A only). Spec locked before running: docs/research/janek/xvb1_spec_20261123_janek.md

Same-family ablation of DualAgree v1 (B1+agree branch), NOT an independent method.
Rule (predictions + box geometry only, never GT):
  parents  = B1 epsilon_005 boxes with raw >= .5   (= baseline output epsilon_005/.50)
  evidence = every B-Clean mask_040 candidate in the dump (raw >= .20)
  remove a parent iff short side (min-area rect, projected to native LR2 px) >= 8 AND no evidence box has IoU >= .3
  with it (many-to-one allowed). Everything else is kept. No parameter is swept.
Split B and Clean are NOT computed here.
"""
import argparse
import csv
import gzip
import json
import sys
import time
from pathlib import Path

from mixab import require, save_json 
from mixab import dualagree_janek as da 
from mixab import phase0_janek as p0 

CUT = .5
MIN_SIDE_LR2 = 8.0
IOU_CUT = .3
GEOM_PARENT, GEOM_EVID = 'epsilon_005', 'mask_040'
SET_PARENT, SET_TUAN = p0.SET_B1, p0.SET_TUAN
GATE_PP, TINY_PP = .3, -3.


# ---------------------------------------------------------------- pure rule (CPU-testable)

def removable(parent_coords, evidence_coords, clean_h, min_side=MIN_SIDE_LR2, iou_cut=IOU_CUT):
    """Indices (into parent_coords) to remove. Polygons that cannot be built are never removed."""
    import numpy as np
    if not parent_coords:
        return set()
    iou = da.iou_matrix(parent_coords, evidence_coords) if evidence_coords else np.zeros((len(parent_coords), 0))
    out = set()
    for i, coords in enumerate(parent_coords):
        if da.polygon_of(coords) is None:
            continue
        if p0.short_edge_lr2(coords, clean_h) >= min_side and not (iou[i] >= iou_cut).any():
            out.add(i)
    return out


def label(delta_pp, ci_low, tiny_delta_pp, fp_rule, fp_ref):
    """Locked outcome label (spec section 6) and the three flags."""
    flags = dict(c1_gain_ge_0p3_and_ci_gt0=bool(delta_pp >= GATE_PP and ci_low > 0),
                 c2_tiny_ge_minus_3=bool(tiny_delta_pp >= TINY_PP), c3_fp_lower=bool(fp_rule < fp_ref))
    if all(flags.values()):
        name = 'PASS_SCREEN'
    elif delta_pp <= 0:
        name = 'NEGATIVE'
    else:
        name = 'POSITIVE_BELOW_GATE'
    return name, flags


# ---------------------------------------------------------------- run

def run(a):
    import numpy as np
    from mixab import reselect_split_janek as rs 
    rows, policy, lr2, geoms, scores, dl, dc, ident = p0.load_all(a.dump, a.bundle)
    from rev4.decision_core import CandidateMatcher
    out_dir = Path(a.output)
    require(not (out_dir/'xvb1_result.json').exists(), f'Refusing to overwrite {out_dir}/xvb1_result.json')
    gi = {g: i for i, g in enumerate(geoms)}
    sids = sorted((s for s in dl if rs.split_of(lr2[s]) == 'A'), key=lambda s: lr2[s])
    require(len(sids) == 1130, f'Split A must have 1,130 sources, got {len(sids)}')
    out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(Path(a.cache)/'t4_counts_B1.npz') as z:
        srcs, c72 = list(z['source_ids']), z['counts']
    idx = {s: i for i, s in enumerate(srcs)}
    k_par = gi[GEOM_PARENT]*len(scores) + scores.index(.5)
    k_tuan = gi[GEOM_EVID]*len(scores) + scores.index(.6)
    a_rows = np.array([idx[lr2[s]] for s in sids])
    ref_par, tuan = c72[a_rows, k_par], c72[a_rows, k_tuan]
    base_rows, rule_rows, per = [], [], []
    tic = time.perf_counter()
    for n, sid in enumerate(sids):
        source = lr2[sid]
        row = rows[source]
        e = dl[sid][gi[GEOM_PARENT]]
        ev = dc[sid][gi[GEOM_EVID]]
        m = CandidateMatcher(dict(polygons=e['polygons'], scores=e['packed']), e['raw'], row, policy)
        groups = [{i for i, h in enumerate(m.heights) if lo <= h < hi} for lo, hi in ((0, 4), (4, 8), (8, 16), (16, float('inf')))]
        ids = [i for i, r in enumerate(e['raw']) if r >= CUT]
        rem_pos = removable([e['polygons'][i] for i in ids], ev['polygons'], row['clean_size'][1])
        removed = {ids[p] for p in rem_pos}
        kept_all, kept_rule = set(ids), set(ids) - removed
        pb, ib = p0.match_pairs(m, kept_all)
        pr, ir = p0.match_pairs(m, kept_rule)
        b_row = p0.row_from_pairs(m, groups, pb, ib, kept_all)
        r_row = p0.row_from_pairs(m, groups, pr, ir, kept_rule)
        matched_boxes = {p for _, p in pb}
        tp_removed = b_row[0] - r_row[0]
        fp_removed = (b_row[2] - r_row[2]) - tp_removed
        per.append(dict(source=source, boxes=len(ids), removed=len(removed), tp_base=b_row[0], tp_rule=r_row[0],
                        pred_base=b_row[2], pred_rule=r_row[2], gt=b_row[1], tp_removed=tp_removed, fp_removed=fp_removed,
                        removed_boxes_matched_in_base=len(removed & matched_boxes),
                        gt_lost=len({g for g, _ in pb} - {g for g, _ in pr}),
                        gt_gained=len({g for g, _ in pr} - {g for g, _ in pb})))
        base_rows.append(b_row); rule_rows.append(r_row)
        if n % 200 == 0:
            print(f'{n + 1}/{len(sids)} ({time.perf_counter() - tic:.0f} s)', flush=True)
    base_rows, rule_rows = np.array(base_rows), np.array(rule_rows)
    check = (base_rows[:, :3] == ref_par[:, :3]).all()
    require(bool(check), 'Baseline epsilon_005/.50 differs from the 72-setting recompute of the same dump')
    tb, tr, tt = base_rows.sum(0), rule_rows.sum(0), tuan.sum(0)
    t_rule, t_tuan, t_base = p0.totals(tr), p0.totals(tt), p0.totals(tb)
    boot_tuan, boot_par = p0.paired_boot(rule_rows, tuan), p0.paired_boot(rule_rows, base_rows)
    tiny_pp = (t_rule['tiny_recall'] - t_tuan['tiny_recall'])*100
    name, flags = label(boot_tuan['delta_pp'], boot_tuan['ci95_pp'][0], tiny_pp, t_rule['fp'], t_tuan['fp'])

    def bins(t):
        return {k: (int(t[2*i + 4])/int(t[2*i + 3]) if int(t[2*i + 3]) else None)
                for i, k in enumerate(('lt4', '4_8', '8_16', 'ge16'))}
    d_tp = rule_rows[:, 0] - tuan[:, 0]
    d_pred = rule_rows[:, 2] - tuan[:, 2]
    net = p0.net_contributions(d_tp, d_pred, p0.hmean(tt))
    pos = sorted((p['removed'] for p in per), reverse=True)
    res = dict(label='XV-B1 pilot: same-family ablation of DualAgree v1 (B1+agree). Split A only (already used many times); '
                     'exploratory, not independent. T4 dump -> PENDING_REFERENCE_CHECK. B and Clean NOT evaluated.',
               spec='docs/research/janek/xvb1_spec_20261123_janek.md', dump_identity_sha256=ident, sources=len(sids),
               rule=dict(parent=f'{GEOM_PARENT}/raw>={CUT}', evidence=f'{GEOM_EVID}/raw>=0.20 (any)', min_side_lr2_px=MIN_SIDE_LR2,
                         iou=IOU_CUT, matching='many-to-one, no 1-1'),
               baseline_check='epsilon_005/.50 counts equal the 72-setting recompute on all 1,130 sources',
               xvb1=t_rule, b1_040_060=t_tuan, b1_eps005_050=t_base,
               delta_vs_040_060=boot_tuan, delta_vs_eps005_050=boot_par,
               tiny_delta_pp_vs_040_060=tiny_pp, tiny_delta_pp_vs_eps005_050=(t_rule['tiny_recall'] - t_base['tiny_recall'])*100,
               recall=dict(xvb1=t_rule['tp']/t_rule['gt'], b1_040_060=t_tuan['tp']/t_tuan['gt'], b1_eps005_050=t_base['tp']/t_base['gt']),
               precision=dict(xvb1=t_rule['tp']/t_rule['pred'], b1_040_060=t_tuan['tp']/t_tuan['pred'], b1_eps005_050=t_base['tp']/t_base['pred']),
               recall_by_bin=dict(xvb1=bins(tr), b1_040_060=bins(tt), b1_eps005_050=bins(tb)),
               removal=dict(boxes_removed=sum(p['removed'] for p in per), sources_affected=sum(p['removed'] > 0 for p in per),
                            tp_removed=int(tb[0] - tr[0]), fp_removed=int((tb[2] - tr[2]) - (tb[0] - tr[0])),
                            removed_boxes_matched_in_baseline=sum(p['removed_boxes_matched_in_base'] for p in per),
                            gt_lost=sum(p['gt_lost'] for p in per), gt_gained=sum(p['gt_gained'] for p in per),
                            note='tp_removed / fp_removed are exact count differences vs epsilon_005/.50 (the parent). '
                                 'removed_boxes_matched_in_baseline uses one maximum matching (not unique).'),
               per_source_vs_040_060=dict(sources_up=int((np.array(net) > 0).sum()), sources_down=int((np.array(net) < 0).sum()),
                                          top10_share_of_net_gain=p0.top_share(net, 10)),
               flags=flags, outcome=name,
               group_gate='not evaluated (needs Clean)')
    save_json(out_dir/'xvb1_result.json', res)
    with (out_dir/'per_source_A.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        cols = list(per[0]) + ['tp_040_060', 'pred_040_060', 'net_vs_040_060']
        w.writerow(cols)
        for p, t_, n_ in zip(per, tuan, net):
            w.writerow([p[c] for c in list(per[0])] + [int(t_[0]), int(t_[2]), round(float(n_), 6)])
    print(json.dumps({k: res[k] for k in ('xvb1', 'b1_040_060', 'delta_vs_040_060', 'delta_vs_eps005_050',
                                          'tiny_delta_pp_vs_040_060', 'removal', 'per_source_vs_040_060', 'flags', 'outcome')},
                     indent=1, default=str))


def main():
    sys.stdout.reconfigure(errors='backslashreplace')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for arg in ('--dump', '--bundle', '--cache', '--output'):
        ap.add_argument(arg, required=True, type=Path)
    run(ap.parse_args())


if __name__ == '__main__':
    main()
