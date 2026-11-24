"""Phase 0 for CS-B1@0.7 (CPU only, from the existing T4 dump). Plan: docs/phase0_csb1_plan_20261122.md.

Commands
  refcheck  Section 3: T4 dump vs A2r A100 counts, per field / setting / split; totals at the CS-B1 settings;
            sensitivity of Delta(CS-B1 - B1) when sources that drift at those settings are dropped.
  stats     Section 4 (Q1-Q5): per-object records for B1 alone, CS-B1 and DualAgree v1 on every LR2 image.
            Development numbers are split A. Split B is reported descriptively only, never used to pick anything.
  cards     Section 5 (80 cards, user choice 22/11): 70 stratified cards from A + 10 calibration cards from B.

Reused, unchanged, from Tuan (MONKEY_V2): CandidateMatcher (matching, ROI, ignore), maximum_matching,
rev4.decision_review.fp_proxy (FP proxies), and the box-based order of rev4.b1_diagnostics.reason (GT diagnosis).
The P-map part of Tuan's GT diagnosis (invalid_kernel / weak_response) cannot be computed: the dump has no P-map.
Those GTs end up in 'no_box_in_dump' and must NOT be called weak response.

CS-B1@0.7 = mixab.dualagree.LOCKED: B1 epsilon_005 raw >= .5, plus B-Clean mask_040 raw >= .7 boxes that do not
overlap a kept B1 box at IoU >= .5 (greedy one-to-one). Nothing here changes it.
"""
import argparse
import gzip
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from mixab import read_json, require, save_json
from mixab import dualagree as da
from mixab import merge_audit as ma

FIELDS = ('tp', 'gt', 'pred', 'gt_lt4', 'tp_lt4', 'gt_4_8', 'tp_4_8', 'gt_8_16', 'tp_8_16', 'gt_ge16', 'tp_ge16')
CSB1 = (da.LOCKED['t_l'], da.LOCKED['t_c'], da.LOCKED['t_a'], da.LOCKED['theta'])      # (.5, .7, inf, .5)
BASE = (.5, da.INF, da.INF, .3)                                                         # B1 alone, epsilon_005/.50
DUALAGREE_V1 = (.7, .7, .5, .3)                     # locked pick of v1 (runs/dualagree_analysis_v1_T4/decision.json)
SET_B1, SET_BC = 'epsilon_005/score_0.50', 'mask_040/score_0.70'
SET_TUAN = 'mask_040/score_0.60'                    # Tuan's .40/.60 operating point (B1), for orientation only
CARD_SEED = 20261123
BOOT_SEED, BOOT_N = 431, 2000
DIAG_ORDER = ('ambiguous_matching', 'score_filtering', 'geometry', 'ambiguous_score_and_geometry',
              'only_box_removed_by_roi_ignore', 'no_box_in_dump')


def hmean(t):
    tp, gt, pred = int(t[0]), int(t[1]), int(t[2])
    return float(Fraction(2*tp, gt + pred)) if gt + pred else 0.


def totals(t):
    tp, gt, pred = int(t[0]), int(t[1]), int(t[2])
    return dict(tp=tp, fp=pred - tp, fn=gt - tp, gt=gt, pred=pred, hmean=hmean(t),
                tiny_recall=(int(t[4])/int(t[3])) if int(t[3]) else None)


def paired_boot(new, old, seed=BOOT_SEED, n=BOOT_N):
    """Delta Hmean (pp) of summed counts, with a paired bootstrap over sources (rows)."""
    import numpy as np
    new, old = np.asarray(new), np.asarray(old)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        j = rng.integers(0, len(new), len(new))
        d.append((hmean(new[j].sum(0)) - hmean(old[j].sum(0)))*100)
    return dict(delta_pp=(hmean(new.sum(0)) - hmean(old.sum(0)))*100, sources=int(len(new)),
                ci95_pp=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])


def net_contributions(d_tp, d_pred, h_base):
    """Exact additive split of Delta Hmean: H_new - H_base = sum_i (2 dTP_i - H_base dPred_i) / (GT + Pred_new)."""
    return [2*t - h_base*p for t, p in zip(d_tp, d_pred)]


def top_share(net, k=10):
    """Share of the total net gain coming from the k largest per-source contributions."""
    total = sum(net)
    top = sorted(net, reverse=True)[:k]
    return None if total <= 0 else sum(top)/total


# ---------------------------------------------------------------- geometry helpers (CPU-testable)

def short_edge_lr2(coords, clean_h):
    """Short side of the minimum-area rotated rectangle of a polygon (clean px), projected to native LR2 px."""
    import cv2
    import numpy as np
    pts = np.asarray(coords, dtype=np.float32).reshape(-1, 2)
    (_, _), (w, h), _ = cv2.minAreaRect(pts)
    return float(min(w, h))*(max(1, clean_h//2)/clean_h)


def size_bin(px):
    return '<4' if px < 4 else '4-8' if px < 8 else '8-16' if px < 16 else '>=16'


def score_bucket(raw, cuts=(.6, .7, .8, .9), lo=.5):
    """'0.5-0.6', ..., '>=0.9' for B1 boxes (kept from .5); use cuts=(.8, .9), lo=.7 for added B-Clean boxes."""
    if raw < lo:
        return f'<{lo:.1f}'
    for c in cuts:
        if raw < c:
            return f'{lo:.1f}-{c:.1f}'
        lo = c
    return f'>={cuts[-1]:.1f}'


def gt_diagnosis(kept_ious, cand_ious, removed_overlap, iou_cut):
    """Box-based order of Tuan's rev4.b1_diagnostics.reason (P-map steps unavailable -> folded into the tail)."""
    if any(i >= iou_cut for i in kept_ious):
        return 'ambiguous_matching'
    if any(i >= iou_cut for i in cand_ious):
        return 'score_filtering'
    if kept_ious:
        return 'geometry'
    if cand_ious:
        return 'ambiguous_score_and_geometry'
    if removed_overlap:
        return 'only_box_removed_by_roi_ignore'
    return 'no_box_in_dump'


def match_pairs(matcher, kept_set):
    """CandidateMatcher.score() on an arbitrary kept subset of the pool -> (pairs (gt_i, pool_p), invalid kept).
    Same maximum matching as Tuan; which prediction pairs with which GT is one maximum matching (counts exact)."""
    from lowres.evaluate_shared import maximum_matching
    kept = [p for p in matcher.originals if p in kept_set]
    to_local = {p: i for i, p in enumerate(kept)}
    adjacency = [[to_local[p] for p in edge if p in to_local] for edge in matcher.edges]
    pairs = maximum_matching(adjacency, len(kept))
    return [(g, kept[p]) for g, p in pairs], [p for p in matcher.invalid if p in kept_set]


def row_from_pairs(matcher, groups, pairs, invalid, kept_set):
    matched = {g for g, _ in pairs}
    kept_valid = sum(p in kept_set for p in matcher.originals)
    row = [len(pairs), len(matcher.ann_ids), kept_valid + len(invalid)]
    for group in groups:
        row += [len(group), len(group & matched)]
    return row


def allocate(populations, total, minimum=0):
    """Quotas per stratum: min(minimum, pop) first, rest by largest remainder proportional to the population
    still available; a stratum never gets more than its population. Deterministic (ties by key)."""
    keys = sorted(populations)
    quota = {k: min(minimum, populations[k]) for k in keys}
    left = total - sum(quota.values())
    while left > 0:
        room = {k: populations[k] - quota[k] for k in keys if populations[k] > quota[k]}
        if not room:
            break
        mass = sum(room.values())
        share = {k: left*room[k]/mass for k in room}
        add = {k: min(int(share[k]), room[k]) for k in room}
        if sum(add.values()) == 0:
            order = sorted(room, key=lambda k: (-(share[k] - int(share[k])), k))
            add = {k: 0 for k in room}
            for k in order[:left]:
                add[k] = 1
        for k, v in add.items():
            quota[k] += v
        left = total - sum(quota.values())
    return quota


def equal_allocate(populations, total):
    """Equal quota per stratum, capped at the population; the shortfall is spread over the others."""
    keys = sorted(k for k in populations if populations[k] > 0)
    quota = {k: 0 for k in populations}
    left = total
    while left > 0:
        room = [k for k in keys if populations[k] > quota[k]]
        if not room:
            break
        each = max(1, left//len(room))
        for k in room:
            if left == 0:
                break
            give = min(each, populations[k] - quota[k], left)
            quota[k] += give
            left -= give
    return quota


def rank(case_id, seed=CARD_SEED):
    return hashlib.sha256(f'{seed}:{case_id}'.encode('utf-8')).hexdigest()


def sample_strata(cases, key, quota, seed=CARD_SEED):
    """cases: list of dicts with 'case_id'. Takes the quota[s] smallest SHA ranks in each stratum s."""
    by = defaultdict(list)
    for c in cases:
        by[key(c)].append(c)
    out = []
    for s, q in sorted(quota.items()):
        pool = sorted(by.get(s, []), key=lambda c: rank(c['case_id'], seed))
        for c in pool[:q]:
            out.append(dict(c, stratum=s, stratum_population=len(by.get(s, [])), stratum_quota=q,
                            inclusion_probability=q/len(by[s]), population_weight=len(by[s])/q))
    return out


# ---------------------------------------------------------------- data loading

def load_all(dump, bundle):
    rows, policy, lr2, geoms, scores = da._context(bundle)
    ident = read_json(Path(dump)/'dump_identity.json')['identity_sha256']
    dl = da.load_model_dump(Path(dump)/'B1', len(geoms), ident)
    dc = da.load_model_dump(Path(dump)/'B-Clean', len(geoms), ident)
    require(set(dl) == set(dc) and len(dl) == 2262, 'Dump must cover the same 2,262 LR2 tasks for both models')
    return rows, policy, lr2, geoms, scores, dl, dc, ident


# ---------------------------------------------------------------- refcheck (plan section 3)

def t4_counts(entries_by_sid, lr2, rows, policy, scores, source_order):
    """[source, 72, 11] LR2 counts recomputed from the dump with Tuan's CandidateMatcher."""
    import numpy as np
    from rev4.decision_core import CandidateMatcher
    index = {s: i for i, s in enumerate(source_order)}
    n_geom = len(next(iter(entries_by_sid.values())))
    out = np.zeros((len(source_order), n_geom*len(scores), len(FIELDS)), dtype=np.int64)
    for sid, entries in entries_by_sid.items():
        source = lr2[sid]
        for gi, e in enumerate(entries):
            m = CandidateMatcher(dict(polygons=e['polygons'], scores=e['packed']), e['raw'], rows[source], policy)
            for si, s in enumerate(scores):
                out[index[source], gi*len(scores) + si] = ma.count_row(m.score(s))
    return out


def refcheck(a):
    import numpy as np
    from mixab import reselect_split as rs
    rows, policy, lr2, geoms, scores, dl, dc, ident = load_all(a.dump, a.bundle)
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    out = dict(label='Phase 0 step 1: T4 dump vs A2r (A100-SXM4-40GB). Same commit, checkpoints, config, '
                     'postprocessor SHA, torch/CUDA/cuDNN; only the GPU differs (dump_identity vs audit_identity).',
               dump_identity_sha256=ident, models={})
    t4, ref, keys = {}, {}, None
    for name, entries in (('B1', dl), ('B-Clean', dc)):
        r = rs.load_counts(Path(a.npz_run)/f'{name}_source_counts.npz')
        settings = [rs.setting_key(s) for s in r['settings']]
        keys = keys or settings
        require(settings == keys, 'Setting order differs between models')
        path = cache/f't4_counts_{name}.npz'
        if path.is_file():
            with np.load(path) as z:
                require(str(z['identity']) == ident and list(z['source_ids']) == r['source_ids'], f'stale {path}')
                c = z['counts']
        else:
            tic = time.perf_counter()
            c = t4_counts(entries, lr2, rows, policy, scores, r['source_ids'])
            np.savez_compressed(path, counts=c, source_ids=np.asarray(r['source_ids']), identity=np.asarray(ident))
            print(f'{name}: recomputed T4 counts in {time.perf_counter() - tic:.0f} s', flush=True)
        t4[name], ref[name] = c, r
    sources = ref['B1']['source_ids']
    require(ref['B-Clean']['source_ids'] == sources, 'A2r source order differs between models')
    split = np.array([rs.split_of(s) for s in sources])
    for name in ('B1', 'B-Clean'):
        a100 = ref[name]['counts'][:, 1]                        # LR2
        diff = t4[name] - a100
        cell = (diff != 0).any(axis=2)                          # [source, setting]
        per_field = {f: int((diff[..., k] != 0).sum()) for k, f in enumerate(FIELDS)}
        mags = {f: dict(sorted(Counter(int(v) for v in diff[..., k][diff[..., k] != 0]).items()))
                for k, f in enumerate(FIELDS) if per_field[f]}
        per_setting = sorted(((int(cell[:, j].sum()), keys[j]) for j in range(len(keys))), reverse=True)
        out['models'][name] = dict(
            cells=int(cell.size), mismatched_cells=int(cell.sum()), mismatched_fraction=float(cell.mean()),
            by_split={s: dict(cells=int(cell[split == s].size), mismatched=int(cell[split == s].sum()))
                      for s in ('A', 'B')},
            cells_with_field_diff=per_field, field_diff_values=mags,
            sources_with_any_mismatch=int(cell.any(axis=1).sum()),
            settings_most_mismatched=[dict(setting=k, cells=v) for v, k in per_setting[:8]],
            settings_least_mismatched=[dict(setting=k, cells=v) for v, k in per_setting[-4:]],
            gt_field_ever_differs=bool(per_field['gt'] or any(per_field[f] for f in FIELDS if f.startswith('gt_'))))
    # totals at the settings CS-B1 uses (+ Tuan's .40/.60 for orientation)
    tot = {}
    for name, setting in (('B1', SET_B1), ('B-Clean', SET_BC), ('B1', SET_TUAN)):
        j = keys.index(setting)
        for s in ('A', 'B'):
            m = split == s
            t_t4, t_ref = t4[name][m, j].sum(0), ref[name]['counts'][m, 1, j].sum(0)
            tot[f'{name} {setting} {s}'] = dict(T4=totals(t_t4), A100=totals(t_ref),
                                                delta_hmean_pp=(hmean(t_t4) - hmean(t_ref))*100,
                                                delta_tp=int(t_t4[0] - t_ref[0]), delta_pred=int(t_t4[2] - t_ref[2]),
                                                sources_drifting=int((t4[name][m, j] != ref[name]['counts'][m, 1, j]).any(1).sum()))
    out['totals_at_csb1_settings'] = tot
    # sensitivity of Delta CS-B1 (computable only on T4: A100 has no boxes)
    gl, gc = geoms.index(da.GEOM_L), geoms.index(da.GEOM_C)
    sid_of = {lr2[s]: s for s in dl}
    fused, base = [], []
    for source in sources:
        sid = sid_of[source]
        pool = da.PoolScorer(dl[sid][gl], dc[sid][gc], rows[source], policy)
        fused.append(pool.fused(*CSB1)); base.append(pool.fused(*BASE))
    fused, base = np.array(fused), np.array(base)
    require((base == t4['B1'][:, keys.index(SET_B1)]).all(), 'B1-alone pool counts differ from CandidateMatcher counts')
    jl, jc = keys.index(SET_B1), keys.index(SET_BC)
    drift_used = ((t4['B1'][:, jl] != ref['B1']['counts'][:, 1, jl]).any(1) |
                  (t4['B-Clean'][:, jc] != ref['B-Clean']['counts'][:, 1, jc]).any(1))
    geo_l = [k for k, key in enumerate(keys) if key.startswith(da.GEOM_L + '/')]
    geo_c = [k for k, key in enumerate(keys) if key.startswith(da.GEOM_C + '/')]
    drift_geom = ((t4['B1'][:, geo_l] != ref['B1']['counts'][:, 1][:, geo_l]).any((1, 2)) |
                  (t4['B-Clean'][:, geo_c] != ref['B-Clean']['counts'][:, 1][:, geo_c]).any((1, 2)))
    sens = {}
    for s in ('A', 'B'):
        m = split == s
        sens[s] = dict(all_sources=paired_boot(fused[m], base[m]),
                       drop_drift_at_the_two_settings=paired_boot(fused[m & ~drift_used], base[m & ~drift_used]),
                       drop_drift_anywhere_in_the_two_geometries=paired_boot(fused[m & ~drift_geom], base[m & ~drift_geom]),
                       sources_dropped=dict(two_settings=int((m & drift_used).sum()), two_geometries=int((m & drift_geom).sum())))
        full = sens[s]['all_sources']
        for k in ('drop_drift_at_the_two_settings', 'drop_drift_anywhere_in_the_two_geometries'):
            r = sens[s][k]
            r['positive'] = r['delta_pp'] > 0
            r['inside_original_ci'] = full['ci95_pp'][0] <= r['delta_pp'] <= full['ci95_pp'][1]
    out['sensitivity_delta_csb1_minus_b1'] = sens
    # hypothesis (descriptive): drifting sources have a box whose raw score sits close to the cutoff
    near = {}
    for name, entries, setting, g, cut in (('B1', dl, SET_B1, gl, .5), ('B-Clean', dc, SET_BC, gc, .7)):
        j = keys.index(setting)
        drift = (t4[name][:, j] != ref[name]['counts'][:, 1, j]).any(1)
        gap = np.array([min((abs(r - cut) for r in entries[sid_of[src]][g]['raw']), default=1.) for src in sources])
        near[f'{name} {setting}'] = {
            lab: dict(sources=int(sel.sum()), median_min_gap=float(np.median(gap[sel])) if sel.any() else None,
                      share_gap_lt_0p005=float((gap[sel] < .005).mean()) if sel.any() else None)
            for lab, sel in (('drifting', drift), ('not_drifting', ~drift))}
    out['score_gap_to_cutoff_hypothesis'] = near
    a_s = sens['A']
    out['decision_inputs'] = dict(
        condition_A1_positive_and_inside_ci=bool(a_s['drop_drift_at_the_two_settings']['positive'] and
                                                 a_s['drop_drift_at_the_two_settings']['inside_original_ci']),
        sign_flip_when_dropping=bool(not a_s['drop_drift_at_the_two_settings']['positive'] or
                                     not a_s['drop_drift_anywhere_in_the_two_geometries']['positive']))
    save_json(Path(a.output)/'reference_check.json', out)
    print(json.dumps(dict(models={k: dict(mismatched=v['mismatched_cells'], frac=v['mismatched_fraction'])
                                  for k, v in out['models'].items()}, sens=sens), indent=1))


# ---------------------------------------------------------------- stats (plan section 4)

def image_records(sid, source, row, policy, e_l, e_c, geoms_l_all):
    """All per-object records for one LR2 image. geoms_l_all: B1 entries of the 9 geometries (for Q5)."""
    from shapely.geometry import shape
    from shapely.strtree import STRtree
    from rev4.decision_review import fp_proxy
    pool = da.PoolScorer(e_l, e_c, row, policy)
    m, n_l = pool.matcher, pool.n_l
    raw = list(e_l['raw']) + list(e_c['raw'])
    polys = list(e_l['polygons']) + list(e_c['polygons'])
    clean_h = row['clean_size'][1]
    iou_cut = policy['match_iou']
    sets = {}
    for name, params in (('B1', BASE), ('CSB1', CSB1), ('DA1', DUALAGREE_V1)):
        kl, kc = da.fuse_keep(pool.raw_l, pool.raw_c, pool.overlaps, *params)
        kept = set(kl) | {n_l + j for j in kc}
        pairs, invalid = match_pairs(m, kept)
        sets[name] = dict(kept=kept, pairs=pairs, invalid=invalid,
                          row=row_from_pairs(m, pool.groups, pairs, invalid, kept),
                          matched_gt={g for g, _ in pairs}, matched_pred={p for _, p in pairs})
    require(sets['B1']['row'] == pool.fused(*BASE) and sets['CSB1']['row'] == pool.fused(*CSB1),
            'Per-object matching differs from PoolScorer counts')
    entries = row['gt_clean']['valid']
    gt_shapes = [shape(e['geometry']) for e in entries]
    gt_index = ([str(e['ann_id']) for e in entries], gt_shapes, None, STRtree(gt_shapes)) if gt_shapes else None
    out = dict(sid=sid, source=source, rows={k: v['row'] for k, v in sets.items()}, fp=[], added=[], gt=[])
    # predictions: FP of B1 alone and of CS-B1 (proxy), and every added B-Clean box
    for name in ('B1', 'CSB1'):
        s = sets[name]
        matched_ann = {m.ann_ids[g] for g in s['matched_gt']}
        for p in sorted(s['kept']):
            if p in s['matched_pred'] or (p not in m.geometries and p not in s['invalid']):
                continue                                    # TP, or dropped by ROI/ignore (not a prediction)
            geometry = m.geometries.get(p)
            proxy = (fp_proxy(geometry, gt_index, matched_ann, policy)['proxy'] if gt_index and geometry is not None
                     else 'invalid_output' if geometry is None else 'no_valid_gt_overlap_background_or_missing_annotation')
            out['fp'].append(dict(arm=name, p=p, origin='B1' if p < n_l else 'BC_added', raw=raw[p], proxy=proxy,
                                  short_edge_lr2=short_edge_lr2(polys[p], clean_h),
                                  bounds=list(geometry.bounds) if geometry is not None else None))
    cs = sets['CSB1']
    gt_of_pred = {p: g for g, p in cs['pairs']}
    for p in sorted(x for x in cs['kept'] if x >= n_l):
        if p not in m.geometries and p not in cs['invalid']:
            status = 'removed_by_roi_ignore'
        elif p in cs['matched_pred']:
            status = 'TP'
        else:
            status = 'FP'
        g = gt_of_pred.get(p)
        out['added'].append(dict(p=p, j=p - n_l, raw=raw[p], status=status, short_edge_lr2=short_edge_lr2(polys[p], clean_h),
                                 gt=None if g is None else m.ann_ids[g], gt_height_lr2=None if g is None else float(m.heights[g]),
                                 gt_rescued=None if g is None else g not in sets['B1']['matched_gt']))
    # GTs
    pool_geoms = {p: gm for p, gm in m.geometries.items()}
    tree_keys = list(pool_geoms)
    tree = STRtree([pool_geoms[p] for p in tree_keys]) if tree_keys else None
    raw_polys = [da.polygon_of(c) for c in polys]
    for g, (ann, gshape) in enumerate(zip(m.ann_ids, gt_shapes)):
        ious = {}
        if tree is not None:
            for k in tree.query(gshape, predicate='intersects'):
                p = tree_keys[int(k)]
                inter = pool_geoms[p].intersection(gshape).area
                if inter > 0:
                    ious[p] = inter/(pool_geoms[p].area + gshape.area - inter)
        rec = dict(ann=ann, height_lr2=float(m.heights[g]), tp={k: g in v['matched_gt'] for k, v in sets.items()})
        for name in ('B1', 'CSB1'):
            if rec['tp'][name]:
                continue
            kept = sets[name]['kept']
            k_ious = [v for p, v in ious.items() if p in kept]
            c_ious = [v for p, v in ious.items() if p not in kept]
            removed = any(raw_polys[p] is not None and p in kept and p not in pool_geoms and
                          raw_polys[p].intersects(gshape) for p in range(len(polys)))
            rec[f'diag_{name}'] = gt_diagnosis(k_ious, c_ious, removed, iou_cut)
            if name == 'CSB1':
                rec['score_filtering_by'] = ('B1' if any(v >= iou_cut for p, v in ious.items() if p < n_l and p not in kept)
                                             else 'BC' if any(v >= iou_cut for p, v in ious.items() if p >= n_l and p not in kept)
                                             else None)
                rec['bc_ge_0p7_iou_ge_0p5_not_added'] = any(p >= n_l and p not in kept and raw[p] >= .7 and v >= iou_cut
                                                          for p, v in ious.items())
        # Q5 for GTs rescued by CS-B1: what did B1 itself have near this GT?
        if rec['tp']['CSB1'] and not rec['tp']['B1']:
            b1_ious = {p: v for p, v in ious.items() if p < n_l}
            rec['q5'] = dict(
                b1_box_iou_ge_0p3=any(v >= .3 for v in b1_ious.values()),
                b1_box_iou_ge_0p5_score_lt_0p5=any(v >= iou_cut and raw[p] < .5 for p, v in b1_ious.items()),
                b1_box_iou_0p3_0p5=any(.3 <= v < iou_cut for v in b1_ious.values()),
                best_b1_iou=max(b1_ious.values(), default=0.),
                other_b1_geometries=q5_other_geometries(gshape, row, policy, geoms_l_all, iou_cut))
        # Q4: tiny GT found by B1 alone but lost by DualAgree v1
        if rec['tp']['B1'] and not rec['tp']['DA1'] and rec['height_lr2'] < 4:
            covering = [p for p in m.edges[g] if p < n_l and raw[p] >= .5]
            in_da = [p for p in covering if p in sets['DA1']['kept']]
            rec['q4'] = dict(reason='b1_box_kept_but_matched_elsewhere' if in_da else
                             ('b1_box_raw_0p5_0p7_cut' if all(raw[p] < .7 for p in covering) else 'other'),
                             covering_raw=[raw[p] for p in covering])
        out['gt'].append(rec)
    return out


def q5_other_geometries(gshape, row, policy, geoms_l_all, iou_cut):
    """For each B1 geometry of the dump: best IoU with this GT among its (ROI/ignore-filtered) boxes, and the
    highest raw score among boxes with IoU >= match_iou (None if none)."""
    from rev4.paired_eda_core import filtered_geometry
    res = {}
    for gname, e in geoms_l_all.items():
        shapes, _ = filtered_geometry(dict(polygons=e['polygons'], scores=e['packed']), row, policy)
        best, best_raw = 0., None
        for p, gm in shapes.items():
            if not gm.intersects(gshape):
                continue
            inter = gm.intersection(gshape).area
            v = inter/(gm.area + gshape.area - inter) if inter else 0.
            best = max(best, v)
            if v >= iou_cut:
                best_raw = max(best_raw or 0., e['raw'][p])
        res[gname] = dict(best_iou=best, best_raw_at_match_iou=best_raw)
    return res


def stats(a):
    import numpy as np
    from mixab import reselect_split as rs
    rows, policy, lr2, geoms, scores, dl, dc, ident = load_all(a.dump, a.bundle)
    gl, gc = geoms.index(da.GEOM_L), geoms.index(da.GEOM_C)
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    rec_path = cache/'phase0_records.jsonl.gz'
    sids = sorted(dl, key=lambda s: lr2[s])
    if not rec_path.is_file():
        tic = time.perf_counter()
        tmp = rec_path.with_name(rec_path.name + '.part')
        with gzip.open(tmp, 'wt', encoding='utf-8') as f:
            for k, sid in enumerate(sids):
                source = lr2[sid]
                rec = image_records(sid, source, rows[source], policy, dl[sid][gl], dc[sid][gc],
                                    {g: dl[sid][i] for i, g in enumerate(geoms)})
                rec['split'] = rs.split_of(source)
                f.write(json.dumps(rec) + '\n')
                if k % 250 == 0:
                    print(f'records {k + 1}/{len(sids)} ({time.perf_counter() - tic:.0f} s)', flush=True)
        tmp.replace(rec_path)
    recs = [json.loads(line) for line in gzip.open(rec_path, 'rt', encoding='utf-8')]
    out = dict(label='Phase 0 automatic statistics (Q1-Q5). Development = split A; B descriptive only (already seen). '
                     'T4 dump -> PENDING_REFERENCE_CHECK.', dump_identity_sha256=ident,
               csb1=dict(t_l=.5, t_c=.7, theta=.5, geometry=dict(B1=da.GEOM_L, BClean=da.GEOM_C)),
               dualagree_v1=dict(t_l=.7, t_c=.7, t_a=.5, theta=.3),
               notes=['GT diagnosis = box-based order of Tuan rev4.b1_diagnostics.reason at the CS-B1 operating point; '
                      'P-map steps (invalid_kernel, weak_response) are NOT computable from the dump and are folded into '
                      'no_box_in_dump. Tuan .40/.60 gt_diagnosis is context only (different operating point).',
                      'FP proxy = Tuan rev4.decision_review.fp_proxy, unchanged. Proxies are signals, not causes.',
                      'Per-box TP attribution uses one maximum matching; counts are exact, the pairing is not unique.'])
    for split in ('A', 'B'):
        R = [r for r in recs if r['split'] == split]
        tot = {k: np.sum([r['rows'][k] for r in R], axis=0) for k in ('B1', 'CSB1', 'DA1')}
        res = dict(sources=len(R), totals={k: totals(v) for k, v in tot.items()},
                   delta_csb1_minus_b1_pp=(hmean(tot['CSB1']) - hmean(tot['B1']))*100)
        # Q1: FP proxies
        q1 = {}
        for arm in ('B1', 'CSB1'):
            fps = [f for r in R for f in r['fp'] if f['arm'] == arm]
            q1[arm] = dict(fp=len(fps), by_proxy=dict(Counter(f['proxy'] for f in fps).most_common()),
                           by_proxy_and_size={f'{p} | {s}': n for (p, s), n in sorted(Counter(
                               (f['proxy'], size_bin(f['short_edge_lr2'])) for f in fps).items())},
                           by_score=dict(sorted(Counter(score_bucket(f['raw']) for f in fps).items())),
                           by_origin=dict(Counter(f['origin'] for f in fps)))
        res['Q1_fp_proxies'] = q1
        # Q2: tiny GT missed by CS-B1
        tiny = [g for r in R for g in r['gt'] if g['height_lr2'] < 4]
        miss = [g for g in tiny if not g['tp']['CSB1']]
        q2 = Counter(g['diag_CSB1'] for g in miss)
        res['Q2_tiny_missed_by_csb1'] = dict(
            tiny_gt=len(tiny), missed=len(miss),
            by_diagnosis={k: q2.get(k, 0) for k in DIAG_ORDER},
            score_filtering_by=dict(Counter(g['score_filtering_by'] for g in miss if g['diag_CSB1'] == 'score_filtering')),
            plan_groups=dict(a_no_box=q2.get('no_box_in_dump', 0) + q2.get('only_box_removed_by_roi_ignore', 0),
                             b_score=q2.get('score_filtering', 0), c_geometry=q2.get('geometry', 0) + q2.get('ambiguous_score_and_geometry', 0),
                             d_matching=q2.get('ambiguous_matching', 0),
                             e_bclean_ge_0p7_not_added=sum(g['bc_ge_0p7_iou_ge_0p5_not_added'] for g in miss)),
            b1_alone_by_diagnosis=dict(Counter(g['diag_B1'] for g in tiny if not g['tp']['B1'])))
        # Q3: added B-Clean boxes
        added = [dict(x, source=r['source']) for r in R for x in r['added']]
        h_base = hmean(tot['B1'])
        per_src = [(r['source'], r['rows']['CSB1'][0] - r['rows']['B1'][0], r['rows']['CSB1'][2] - r['rows']['B1'][2]) for r in R]
        net = net_contributions([t for _, t, _ in per_src], [p for _, _, p in per_src], h_base)
        denom = int(tot['CSB1'][1] + tot['CSB1'][2])
        order = sorted(range(len(R)), key=lambda i: -net[i])
        keep = np.ones(len(R), bool)
        keep[order[:10]] = False
        rescued = [g for r in R for g in r['gt'] if g['tp']['CSB1'] and not g['tp']['B1']]
        res['Q3_added_boxes'] = dict(
            added=len(added), by_status=dict(Counter(x['status'] for x in added)),
            tp_by_size_score={f'{s} | {b}': n for (s, b), n in sorted(Counter((size_bin(x['short_edge_lr2']), score_bucket(x['raw'], (.8, .9), .7))
                                                                               for x in added if x['status'] == 'TP').items())},
            fp_by_size_score={f'{s} | {b}': n for (s, b), n in sorted(Counter((size_bin(x['short_edge_lr2']), score_bucket(x['raw'], (.8, .9), .7))
                                                                               for x in added if x['status'] == 'FP').items())},
            fp_proxy=dict(Counter(f['proxy'] for r in R for f in r['fp'] if f['arm'] == 'CSB1' and f['origin'] == 'BC_added')),
            delta_tp=int(tot['CSB1'][0] - tot['B1'][0]), delta_fp=int((tot['CSB1'][2] - tot['CSB1'][0]) - (tot['B1'][2] - tot['B1'][0])),
            rescued_gt=len(rescued), rescued_by_gt_height=dict(sorted(Counter(size_bin(g['height_lr2']) for g in rescued).items())),
            lost_gt=sum(1 for r in R for g in r['gt'] if g['tp']['B1'] and not g['tp']['CSB1']),
            sources_net_positive=int(sum(v > 0 for v in net)), sources_net_negative=int(sum(v < 0 for v in net)),
            delta_pp_from_net=sum(net)/denom*100 if denom else None,
            top10_share_of_net_gain=top_share(net, 10),
            top10_sources=[R[i]['source'] for i in order[:10]],
            delta_pp_without_top10=(hmean(np.sum([R[i]['rows']['CSB1'] for i in range(len(R)) if keep[i]], 0)) -
                                    hmean(np.sum([R[i]['rows']['B1'] for i in range(len(R)) if keep[i]], 0)))*100)
        # Q4: DualAgree v1 tiny loss mechanism
        q4 = [g for g in tiny if 'q4' in g]
        res['Q4_dualagree_v1_tiny_loss'] = dict(
            tiny_tp_b1_lost_by_da1=len(q4), by_reason=dict(Counter(g['q4']['reason'] for g in q4)),
            of_these_tp_in_csb1=sum(g['tp']['CSB1'] for g in q4),
            tiny_recall=dict(B1=totals(tot['B1'])['tiny_recall'], CSB1=totals(tot['CSB1'])['tiny_recall'],
                             DA1=totals(tot['DA1'])['tiny_recall']))
        # Q5: did B1 already have a box near the rescued GTs?
        q5 = [g['q5'] for g in rescued]
        n5 = len(q5) or 1
        geo_hit = Counter()
        geo_hit_cut = Counter()
        for x in q5:
            for gname, v in x['other_b1_geometries'].items():
                if v['best_raw_at_match_iou'] is not None:
                    geo_hit[gname] += 1
                    if v['best_raw_at_match_iou'] >= .5:
                        geo_hit_cut[gname] += 1
        any_geo_cut = sum(any(v['best_raw_at_match_iou'] is not None and v['best_raw_at_match_iou'] >= .5
                              for v in x['other_b1_geometries'].values()) for x in q5)
        res['Q5_b1_already_had_box'] = dict(
            rescued_gt=len(q5),
            b1_box_iou_ge_0p3=sum(x['b1_box_iou_ge_0p3'] for x in q5), share_b1_box_iou_ge_0p3=sum(x['b1_box_iou_ge_0p3'] for x in q5)/n5,
            b1_box_iou_ge_0p5_but_score_lt_0p5=sum(x['b1_box_iou_ge_0p5_score_lt_0p5'] for x in q5),
            b1_box_iou_0p3_0p5=sum(x['b1_box_iou_0p3_0p5'] for x in q5),
            no_b1_box_iou_ge_0p3=sum(not x['b1_box_iou_ge_0p3'] for x in q5),
            other_geometries_box_at_match_iou_any_score=dict(geo_hit),
            other_geometries_box_at_match_iou_score_ge_0p5=dict(geo_hit_cut),
            rescued_reachable_by_some_b1_geometry_at_score_ge_0p5=any_geo_cut)
        out[split] = res
    # descriptive sensitivity around the reference, from the v1 sweep (no re-selection)
    z = np.load(Path(a.sweep), allow_pickle=False)
    combos = z['combos']
    sens = {}
    base_k = [k for k, c in enumerate(combos) if c[0] == .5 and np.isinf(c[1]) and np.isinf(c[2])]
    require(len(base_k) == 1, 'B1-alone row not unique in sweep_totals')
    for tc in (.6, .7):
        for th in (.3, .5, .7):
            k = [i for i, c in enumerate(combos) if c[0] == .5 and abs(c[1] - tc) < 1e-9 and np.isinf(c[2]) and abs(c[3] - th) < 1e-9]
            require(len(k) == 1, f'combo t_c={tc} theta={th} missing in sweep_totals')
            sens[f't_c={tc} theta={th}'] = {s: (hmean(z[s][k[0]]) - hmean(z[s][base_k[0]]))*100 for s in ('A', 'B')}
    out['sensitivity_descriptive_pp'] = sens
    ref = sens['t_c=0.7 theta=0.5']['A']
    require(abs(ref - out['A']['delta_csb1_minus_b1_pp']) < 1e-9, 'sweep_totals and per-object totals disagree on A')
    # Tuan's .40/.60 diagnostics as context (different operating point)
    tuan = Path(a.tuan_diag) if a.tuan_diag else None
    if tuan and tuan.is_file():
        out['tuan_040_060_context'] = dict(path=str(tuan), note='Operating point .40/.60 (mask .40, score .60, eps .01): '
                                           'context only, not comparable cell by cell.', summary=read_json(tuan))
    A = out['A']
    q5a, q2a, q3a = A['Q5_b1_already_had_box'], A['Q2_tiny_missed_by_csb1'], A['Q3_added_boxes']
    out['decision_inputs_auto'] = dict(
        A2_tp_rescued_gt_fp_added=q3a['delta_tp'] > q3a['delta_fp'],
        A2_top10_share_lt_0p5=q3a['top10_share_of_net_gain'] is not None and q3a['top10_share_of_net_gain'] < .5,
        A3_sign_positive_tc06_theta03_07=all(sens[f't_c={tc} theta={th}']['A'] > 0 for tc, th in ((.6, .5), (.7, .3), (.7, .7), (.6, .3), (.6, .7))),
        A4_q5_share_lt_0p5=q5a['share_b1_box_iou_ge_0p3'] < .5,
        B_q5_share_ge_0p5=q5a['share_b1_box_iou_ge_0p3'] >= .5,
        B_q2_b_plus_c_share_ge_0p3=(q2a['plan_groups']['b_score'] + q2a['plan_groups']['c_geometry'])/max(q2a['missed'], 1) >= .3,
        C_q2_a_share_gt_0p6=q2a['plan_groups']['a_no_box']/max(q2a['missed'], 1) > .6,
        D_fp_added_ge_tp_rescued=q3a['delta_fp'] >= q3a['delta_tp'],
        D_top10_share_ge_0p5=q3a['top10_share_of_net_gain'] is None or q3a['top10_share_of_net_gain'] >= .5,
        note='Only the automatic parts. Step 1 (reference_check.json) and the card review (khong_chac > 30%) are '
             'evaluated separately; the final A/B/C/D call is made after both, without changing any threshold.')
    save_json(Path(a.output)/'auto_stats.json', out)
    print(json.dumps(dict(A=dict(delta=A['delta_csb1_minus_b1_pp'], Q2=q2a['by_diagnosis'], Q3=dict(
        status=q3a['by_status'], top10=q3a['top10_share_of_net_gain']), Q5=q5a['share_b1_box_iou_ge_0p3']),
        auto=out['decision_inputs_auto']), indent=1, default=str))


# ---------------------------------------------------------------- cards (plan section 5, 80-card variant)

QUOTA_MAIN = dict(F=16, S=24, T=30)          # 70 cards on A (user 22/11: 80 = 70 + 10 calibration; F kept)
QUOTA_CAL = dict(F=3, S=3, T=4)              # calibration on B, not counted
LABELS = dict(
    F=['nen', 'thieu_nhan', 'dinh_vi', 'gop', 'tach', 'trung', 'ignore', 'khong_chac'],
    S_FP=['nen', 'thieu_nhan', 'dinh_vi', 'gop', 'tach', 'trung', 'ignore', 'khong_chac'],
    S_TP=['dung', 'dung_nhung_lech', 'nghi_gt_sai', 'khong_chac'],
    T=['mat_thong_tin', 'thay_nhung_khong_box', 'co_box_lech', 'bi_gop', 'nghi_gt_sai', 'khong_chac'])


def card_populations(recs, split):
    """Candidate cases per group for one split (pure function of the records)."""
    F, S, T = [], [], []
    for r in recs:
        if r['split'] != split:
            continue
        for f in r['fp']:
            if f['arm'] == 'CSB1' and f['origin'] == 'B1' and f['bounds'] is not None:
                F.append(dict(case_id=f'F:{r["source"]}:{f["p"]}', group='F', source=r['source'], sid=r['sid'], p=f['p'],
                              raw=f['raw'], proxy=f['proxy'], short_edge_lr2=f['short_edge_lr2'],
                              size='<8' if f['short_edge_lr2'] < 8 else '>=8'))
        for x in r['added']:
            if x['status'] in ('TP', 'FP'):
                S.append(dict(case_id=f'S:{r["source"]}:{x["p"]}', group='S', source=r['source'], sid=r['sid'], p=x['p'],
                              raw=x['raw'], status=x['status'], short_edge_lr2=x['short_edge_lr2'],
                              size='<8' if x['short_edge_lr2'] < 8 else '>=8', score='0.7-0.8' if x['raw'] < .8 else '>=0.8',
                              gt=x['gt'], gt_height_lr2=x['gt_height_lr2']))
        for g in r['gt']:
            if g['height_lr2'] < 4 and not g['tp']['CSB1']:
                T.append(dict(case_id=f'T:{r["source"]}:{g["ann"]}', group='T', source=r['source'], sid=r['sid'],
                              ann=g['ann'], height_lr2=g['height_lr2'], diagnosis=g['diag_CSB1']))
    return F, S, T


def select_cards(recs):
    F, S, T = card_populations(recs, 'A')
    kf = lambda c: f'{c["proxy"]} | {c["size"]}'                       # noqa: E731
    ks = lambda c: f'{c["status"]} | {c["size"]} | {c["score"]}'      # noqa: E731
    kt = lambda c: c['diagnosis']                                      # noqa: E731
    pops = lambda cases, key: Counter(key(c) for c in cases)           # noqa: E731
    sel = []
    sel += sample_strata(F, kf, equal_allocate(pops(F, kf), QUOTA_MAIN['F']))
    s_tp = [c for c in S if c['status'] == 'TP']
    s_fp = [c for c in S if c['status'] == 'FP']
    sel += sample_strata(s_tp, ks, equal_allocate(pops(s_tp, ks), QUOTA_MAIN['S']//2))
    sel += sample_strata(s_fp, ks, equal_allocate(pops(s_fp, ks), QUOTA_MAIN['S'] - QUOTA_MAIN['S']//2))
    sel += sample_strata(T, kt, allocate(pops(T, kt), QUOTA_MAIN['T'], minimum=4))
    # calibration (split B): one per distinct stratum in rank order, so the reviewer sees several label situations
    Fb, Sb, Tb = card_populations(recs, 'B')
    cal = []
    for group, cases, key, k in (('F', Fb, kf, QUOTA_CAL['F']), ('S', Sb, ks, QUOTA_CAL['S']), ('T', Tb, kt, QUOTA_CAL['T'])):
        seen, chosen = set(), []
        for c in sorted(cases, key=lambda c: rank(c['case_id'], CARD_SEED + 1)):
            if key(c) not in seen:
                seen.add(key(c)); chosen.append(c)
            if len(chosen) == k:
                break
        if group == 'S' and not any(c['status'] == 'TP' for c in chosen):
            tp = next(c for c in sorted(cases, key=lambda c: rank(c['case_id'], CARD_SEED + 1)) if c['status'] == 'TP')
            chosen[-1] = tp
        cal += [dict(c, stratum=key(c), calibration=True, inclusion_probability=None, population_weight=None) for c in chosen]
    populations = dict(F=dict(pops(F, kf)), S_TP=dict(pops(s_tp, ks)), S_FP=dict(pops(s_fp, ks)), T=dict(pops(T, kt)))
    return sel, cal, populations


def label_set(case):
    return 'S_' + case['status'] if case['group'] == 'S' else case['group']


def cards(a):
    import csv
    import random
    rows, policy, lr2, geoms, scores, dl, dc, ident = load_all(a.dump, a.bundle)
    recs = [json.loads(line) for line in gzip.open(Path(a.cache)/'phase0_records.jsonl.gz', 'rt', encoding='utf-8')]
    sel, cal, populations = select_cards(recs)
    out = Path(a.output)
    require(not (out/'review_sheet.csv').exists(), f'Refusing to overwrite {out}/review_sheet.csv (selection is locked)')
    (out/'cards').mkdir(parents=True, exist_ok=True)
    (out/'calibration').mkdir(parents=True, exist_ok=True)
    gl, gc = geoms.index(da.GEOM_L), geoms.index(da.GEOM_C)
    decode = []
    numbered = [(f'K{i + 1:02d}', c) for i, c in enumerate(cal)]
    main = sorted(sel, key=lambda c: rank(c['case_id'], CARD_SEED + 2))           # shuffled reading order
    numbered += [(f'C{i + 1:02d}', c) for i, c in enumerate(main)]
    import zipfile
    zf = zipfile.ZipFile(a.clean_zip)
    for card_id, c in numbered:
        rng = random.Random(f'{CARD_SEED}:{card_id}')
        b1_first = rng.random() < .5
        row = rows[c['source']]
        info = render_card(zf, row, policy, dl[c['sid']][gl], dc[c['sid']][gc], c, card_id, b1_first,
                           out/('calibration' if c.get('calibration') else 'cards')/f'{card_id}.png', a.font)
        decode.append(dict(card_id=card_id, case_id=c['case_id'], group=c['group'], label_set=label_set(c),
                           set1='B1' if b1_first else 'B-Clean', set2='B-Clean' if b1_first else 'B1', **info))
    zf.close()
    # Blind sheet: no stratum / proxy / diagnosis / case_id (they reveal the automatic label). The weights and strata
    # live in selection.json and are joined back by card_id when the review is aggregated.
    fields = ['card_id', 'group', 'label_set', 'human_label', 'confidence', 'reviewer', 'notes']
    for name, items in (('calibration_sheet.csv', [x for x in numbered if x[1].get('calibration')]),
                        ('review_sheet.csv', [x for x in numbered if not x[1].get('calibration')])):
        with (out/name).open('w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fields, extrasaction='ignore')
            w.writeheader()
            for card_id, c in items:
                w.writerow(dict(c, card_id=card_id, label_set=label_set(c), human_label='', confidence='', reviewer='', notes=''))
    save_json(out/'card_decode.json', dict(note='Giải mã (KHÔNG mở khi đang chấm). set1/set2 = model của "bộ 1"/"bộ 2".',
                                           cards=decode))
    save_json(out/'selection.json', dict(seed=CARD_SEED, rule='smallest sha256(seed:case_id) per stratum; F and S equal '
                                         'quota per stratum, T minimum 4 per diagnosis then proportional (largest remainder)',
                                         quotas=dict(main=QUOTA_MAIN, calibration=QUOTA_CAL), populations=populations,
                                         labels=LABELS, main=[dict(card_id=k, **{x: c.get(x) for x in (
                                             'case_id', 'group', 'stratum', 'stratum_population', 'stratum_quota',
                                             'inclusion_probability', 'population_weight')}) for k, c in numbered if not c.get('calibration')],
                                         calibration=[dict(card_id=k, case_id=c['case_id'], stratum=c['stratum'])
                                                      for k, c in numbered if c.get('calibration')],
                                         dump_identity_sha256=ident, locked_before_review=True))
    print(f'{len(sel)} main cards (A) + {len(cal)} calibration cards (B) -> {out}')


def render_card(zf, row, policy, e_l, e_c, case, card_id, b1_first, path, font_path):
    """Two panels (LR2 nearest x k | Clean) of the same window + a neutral caption. Returns geometry info."""
    import io
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from shapely.geometry import shape
    from rev4.training_data import lr2_dimensions
    W, H = row['clean_size']
    fy = max(1, H//2)/H
    pool_polys = list(e_l['polygons']) + list(e_c['polygons'])
    n_l = len(e_l['polygons'])
    if case['group'] == 'T':
        gt = next(e for e in row['gt_clean']['valid'] if str(e['ann_id']) == case['ann'])
        target = np.asarray(shape(gt['geometry']).exterior.coords)[:-1].reshape(-1).tolist()
    else:
        target = pool_polys[case['p']]
    pts = np.asarray(target, dtype=float).reshape(-1, 2)
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    side = max(x1 - x0, y1 - y0)
    margin = max(24., 1.2*side)
    bx0, by0 = max(0., x0 - margin), max(0., y0 - margin*.6)
    bx1, by1 = min(W, x1 + margin), min(H, y1 + margin*.6)
    with Image.open(io.BytesIO(zf.read(row['clean_relpath']))) as im:
        clean = im.convert('RGB')
    # Display only: same recipe as Tuan's native_lr2 (bicubic to lr2_dimensions), but local Pillow is not 11.3.0,
    # so these pixels are NOT the locked LR2 bytes and are never scored.
    lr2 = clean.resize(lr2_dimensions(clean.size), Image.Resampling.BICUBIC)
    lw, lh = lr2.size
    sx, sy = lw/W, lh/H
    l_box = (int(np.floor(bx0*sx)), int(np.floor(by0*sy)), int(np.ceil(bx1*sx)), int(np.ceil(by1*sy)))
    crop_l = lr2.crop(l_box)
    k = max(1, min(8, int(560/max(crop_l.width, 1)), int(300/max(crop_l.height, 1))))
    panel_l = crop_l.resize((crop_l.width*k, crop_l.height*k), Image.NEAREST)
    c_box = (l_box[0]/sx, l_box[1]/sy, l_box[2]/sx, l_box[3]/sy)
    crop_c = clean.crop(tuple(int(round(v)) for v in c_box))
    panel_c = crop_c.resize(panel_l.size, Image.LANCZOS)
    scale = (panel_l.width/(c_box[2] - c_box[0]), panel_l.height/(c_box[3] - c_box[1]))

    def to_px(coords):
        p = np.asarray(coords, dtype=float).reshape(-1, 2)
        return [((x - c_box[0])*scale[0], (y - c_box[1])*scale[1]) for x, y in p]

    def dashed(draw, coords, color, width=2, dash=6):
        pts_ = to_px(coords) + [to_px(coords)[0]]
        for (xa, ya), (xb, yb) in zip(pts_, pts_[1:]):
            n = max(1, int(np.hypot(xb - xa, yb - ya)//dash))
            for i in range(0, n, 2):
                t0, t1 = i/n, min(1, (i + 1)/n)
                draw.line([(xa + (xb - xa)*t0, ya + (yb - ya)*t0), (xa + (xb - xa)*t1, ya + (yb - ya)*t1)], fill=color, width=width)

    col = {'set1': (30, 110, 255), 'set2': (255, 140, 0)}
    sets = {('set1' if b1_first else 'set2'): [(p, e_l['raw'][p]) for p in range(n_l) if e_l['raw'][p] >= .5],
            ('set2' if b1_first else 'set1'): [(n_l + j, e_c['raw'][j]) for j in range(len(e_c['raw'])) if e_c['raw'][j] >= .7]}
    window = (c_box[0] - 1, c_box[1] - 1, c_box[2] + 1, c_box[3] + 1)

    def visible(coords):
        p = np.asarray(coords, dtype=float).reshape(-1, 2)
        return not (p[:, 0].max() < window[0] or p[:, 0].min() > window[2] or p[:, 1].max() < window[1] or p[:, 1].min() > window[3])
    panels = []
    for base in (panel_l, panel_c):
        img = base.copy()
        d = ImageDraw.Draw(img)
        for e in row['gt_clean']['valid']:
            g = np.asarray(shape(e['geometry']).exterior.coords)[:-1].reshape(-1)
            if visible(g):
                d.polygon(to_px(g), outline=(0, 200, 0), width=2)
        for e in row['gt_clean']['ignored']:
            g = np.asarray(shape(e['geometry']).exterior.coords)[:-1].reshape(-1)
            if visible(g):
                dashed(d, g, (150, 150, 150))
        for name in ('set1', 'set2'):
            for p, _ in sets[name]:
                if visible(pool_polys[p]):
                    d.polygon(to_px(pool_polys[p]), outline=col[name], width=2)
        dashed(d, target, (230, 0, 230), width=4, dash=8)
        panels.append(img)
    font = ImageFont.truetype(font_path, 17)
    small = ImageFont.truetype(font_path, 15)
    gap, pad = 12, 10
    target_raw = None if case['group'] == 'T' else (e_l['raw'][case['p']] if case['p'] < n_l else e_c['raw'][case['p'] - n_l])
    target_set = None if case['group'] == 'T' else ('set1' if (case['p'] < n_l) == b1_first else 'set2')
    question = dict(F='Khung tím đứt nét = một box CÓ trong đầu ra cuối và evaluator tính là KHÔNG khớp GT (FP).',
                    S_FP='Khung tím đứt nét = một box CÓ trong đầu ra cuối và evaluator tính là KHÔNG khớp GT (FP).',
                    S_TP='Khung tím đứt nét = một box CÓ trong đầu ra cuối và evaluator tính là KHỚP một GT (TP).',
                    T='Khung tím đứt nét = một GT chữ nhỏ (< 4 px trên LR2) KHÔNG được box nào trong đầu ra cuối khớp.')[label_set(case)]
    lines = [(f'{card_id}   ·   chọn một nhãn: {", ".join(LABELS[label_set(case)])}', font), (question, font)]
    if case['group'] == 'T':
        lines.append((f'Chiều cao GT: {case["height_lr2"]:.1f} px LR2.', small))
    else:
        lines.append((f'Box thuộc "{"bộ 1" if target_set == "set1" else "bộ 2"}", score {target_raw:.3f}, '
                      f'cạnh ngắn {case["short_edge_lr2"]:.1f} px LR2.', small))
    lines.append(('Xanh lá = GT hợp lệ · xám đứt = GT ignore · xanh dương = bộ 1 · cam = bộ 2 '
                  '(chỉ vẽ box đủ ngưỡng của bộ đó).', small))
    width = max(panels[0].width*2 + gap + 2*pad, 980)
    wrapped = []
    for text, f in lines:
        words, cur = text.split(' '), ''
        for w_ in words:
            trial = (cur + ' ' + w_).strip()
            if f.getlength(trial) > width - 2*pad and cur:
                wrapped.append((cur, f)); cur = w_
            else:
                cur = trial
        wrapped.append((cur, f))
    top = pad + 22
    canvas = Image.new('RGB', (width, top + panels[0].height + 10 + 24*len(wrapped) + pad), 'white')
    d = ImageDraw.Draw(canvas)
    x_l = (width - (panels[0].width*2 + gap))//2
    d.text((x_l, pad), f'LR2 (phóng ×{k}, nearest)', fill='black', font=small)
    d.text((x_l + panels[0].width + gap, pad), 'Clean (cùng vùng, chỉ để tham chiếu)', fill='black', font=small)
    canvas.paste(panels[0], (x_l, top))
    canvas.paste(panels[1], (x_l + panels[0].width + gap, top))
    y = top + panels[0].height + 10
    for i, (text, f) in enumerate(wrapped):
        d.text((pad, y + i*24), text, fill='black', font=f)
    canvas.save(path)
    return dict(source=case['source'], target_bounds_clean=[float(x0), float(y0), float(x1), float(y1)],
                lr2_zoom=k, target_raw=target_raw, target_set=target_set)


def main():
    sys.stdout.reconfigure(errors='backslashreplace')
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('refcheck', 'stats', 'cards'):
        s = sub.add_parser(name)
        s.add_argument('--dump', required=True, type=Path)
        s.add_argument('--bundle', required=True, type=Path)
        s.add_argument('--cache', required=True, type=Path, help='Scratch dir for recomputed counts / records')
        s.add_argument('--output', required=True, type=Path)
    sub.choices['refcheck'].add_argument('--npz-run', required=True, type=Path, help='A2r run dir (A100 counts)')
    sub.choices['stats'].add_argument('--sweep', required=True, type=Path, help='dualagree_analysis_v1_T4/sweep_totals.npz')
    sub.choices['stats'].add_argument('--tuan-diag', type=Path, help='Tuan fixed_diagnostics/review/diagnosis_summary.json')
    sub.choices['cards'].add_argument('--clean-zip', required=True, type=Path, help='cache/textvqa_val2262_clean.zip')
    sub.choices['cards'].add_argument('--font', default='C:/Windows/Fonts/arial.ttf')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    dict(refcheck=refcheck, stats=stats, cards=cards)[a.command](a)


if __name__ == '__main__':
    main()
