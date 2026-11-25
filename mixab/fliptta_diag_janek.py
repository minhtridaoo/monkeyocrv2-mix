"""Flip-TTA diagnosis, CPU only, from the EXISTING pilot records (docs/handoff/artifacts/janek/fliptta_v1_T4 + runtime/kaggle_downloads/fliptta_pilot_1).
Post-hoc analysis (after the pilot) to decide whether a new variant is warranted. Chooses no rule and no threshold.

Records hold, per task and arm, the mask_040 candidates with raw >= .20 (polygons in Clean pixels, raw + packed scores).
They do NOT hold P-maps, flip-branch-only predictions or per-stage timings.

Counts: Tuan's CandidateMatcher (checked against the unchanged official evaluator for every task and arm).
GT events use COVERAGE (an IoU >= match_iou edge to a kept box), which does not depend on which maximum matching is chosen:
  lost   = covered by a kept F box, not by any kept T box;   gained = the reverse.
  matching-only = matched status differs but coverage is the same in both arms.
Box correspondence F <-> T: greedy one-to-one on kept boxes by IoU (desc), IoU >= .5 ("same box").
"""
import argparse
import csv
import gzip
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from mixab import require, save_json 
from mixab import dualagree_janek as da 
from mixab import fliptta_janek as ft 
from mixab import phase0_janek as p0 

CUT = ft.CUT
PAIR_IOU = .5
SCORE_BINS = ((.6, .65), (.65, .7), (.7, .8), (.8, 1.01))
ASPECT_BINS = ((0, 3), (3, 6), (6, 12), (12, float('inf')))
CUTOFFS = [round(.50 + .02*k, 2) for k in range(19)]          # .50 ... .86, descriptive PR curve only
FIG_SEED = '20261123:fliptta-diag:'


def sbin(x):
    for lo, hi in SCORE_BINS:
        if lo <= x < hi:
            return f'{lo:.2f}-{min(hi, 1):.2f}'
    return '<0.60'


def aspect(coords):
    import cv2
    import numpy as np
    (_, _), (w, h), _ = cv2.minAreaRect(np.asarray(coords, dtype=np.float32).reshape(-1, 2))
    s, l_ = min(w, h), max(w, h)
    return float('inf') if s <= 0 else l_/s


def abin(a):
    for lo, hi in ASPECT_BINS:
        if lo <= a < hi:
            return f'{lo}-{hi}' if hi != float('inf') else f'>={lo}'
    return '?'


def greedy_pairs(iou, rows, cols, thr=PAIR_IOU):
    cand = sorted(((-iou[i, j], i, j) for i in rows for j in cols if iou[i, j] >= thr))
    ur, uc, out = set(), set(), []
    for neg, i, j in cand:
        if i not in ur and j not in uc:
            ur.add(i); uc.add(j); out.append((i, j, -neg))
    return out


def analyse_task(rec, row, policy):
    """-> dict with counts (F, T), GT events, box events. Pool = all F candidates then all T candidates."""
    import numpy as np
    from rev4.decision_core import CandidateMatcher
    from shapely.geometry import shape
    F, T = rec['F'], rec['T']
    nF, nT = len(F['raw']), len(T['raw'])
    polys = F['polygons'] + T['polygons']
    raw = F['raw'] + T['raw']
    m = CandidateMatcher(dict(polygons=polys, scores=F['packed'] + T['packed']), raw, row, policy)
    groups = [{i for i, h in enumerate(m.heights) if lo <= h < hi} for lo, hi in ((0, 4), (4, 8), (8, 16), (16, float('inf')))]
    kF = {i for i in range(nF) if raw[i] >= CUT}
    kT = {nF + j for j in range(nT) if raw[nF + j] >= CUT}
    pF, iF = p0.match_pairs(m, kF)
    pT, iT = p0.match_pairs(m, kT)
    rowF = p0.row_from_pairs(m, groups, pF, iF, kF)
    rowT = p0.row_from_pairs(m, groups, pT, iT, kT)
    shapes = [da.polygon_of(c) for c in polys]
    gts = [shape(e['geometry']) for e in row['gt_clean']['valid']]

    def iou(a, b):
        if a is None or b is None or not a.intersects(b):
            return 0.
        inter = a.intersection(b).area
        return inter/(a.area + b.area - inter) if inter > 0 else 0.

    # IoU of every pool box with every GT (only intersecting pairs are non-zero)
    gt_iou = np.zeros((len(gts), len(polys)))
    gt_frac = np.zeros((len(gts), len(polys)))        # area(GT & box)/area(GT)
    for g, gs in enumerate(gts):
        for p, ps in enumerate(shapes):
            if ps is not None and gs.intersects(ps):
                inter = gs.intersection(ps).area
                if inter > 0:
                    gt_iou[g, p] = inter/(gs.area + ps.area - inter)
                    gt_frac[g, p] = inter/gs.area
    valid = set(m.geometries)
    edges = [set(e) for e in m.edges]
    mF, mT = {g for g, _ in pF}, {g for g, _ in pT}
    F_ids, T_ids = list(range(nF)), list(range(nF, nF + nT))
    gt_events = []
    for g in range(len(gts)):
        covF, covT = bool(edges[g] & kF), bool(edges[g] & kT)
        if covF == covT:
            if (g in mF) != (g in mT):
                gt_events.append(dict(kind='matching_only', g=g, lost=g in mF))
            continue
        lost = covF
        own, other, kept_other = (kF, T_ids, kT) if lost else (kT, F_ids, kF)
        own_score = max(raw[p] for p in edges[g] & own)
        cand = [p for p in other if p in edges[g]]          # other-arm candidates (any score) with IoU >= .5, valid
        kept_overlap = [p for p in kept_other if gt_iou[g, p] > 0]
        if cand:
            cause, other_score = 'score_crossed_cut', max(raw[p] for p in cand)
        elif kept_overlap:
            best = max(kept_overlap, key=lambda p: gt_iou[g, p])
            n_gt_in_best = int((gt_frac[:, best] >= .5).sum())
            n_kept_covering = sum(gt_frac[g, p] >= .2 for p in kept_overlap)
            cause = 'merged' if n_gt_in_best >= 2 else 'split' if n_kept_covering >= 2 else 'shifted'
            other_score = raw[best]
        else:
            weak = [p for p in other if gt_iou[g, p] > 0]
            cause = 'vanished_weak' if weak else 'vanished_none'
            other_score = max((raw[p] for p in weak), default=None)
        gt_events.append(dict(kind='lost' if lost else 'gained', g=g, cause=cause, height=float(m.heights[g]),
                              own_score=own_score, other_score=other_score,
                              best_iou_other_kept=max((gt_iou[g, p] for p in kept_other), default=0.)))
    # box events
    import numpy as _np
    iouFT = _np.zeros((nF, nT))
    for i in kF:
        for j in kT:
            iouFT[i, j - nF] = iou(shapes[i], shapes[j])
    pairs = greedy_pairs(iouFT, sorted(kF), [j - nF for j in sorted(kT)])
    pF_box, pT_box = {p for _, p in pF}, {p for _, p in pT}
    has_edge = set().union(*edges) if edges else set()

    def cls(p, matched_boxes):
        if p not in valid:
            return 'invalid'
        if p in matched_boxes:
            return 'tp'
        return 'dup_or_competing' if p in has_edge else 'no_gt'
    paired_F = {i for i, _, _ in pairs}
    paired_T = {nF + j for _, j, _ in pairs}
    box_events = []
    for i, j, v in pairs:
        box_events.append(dict(kind='pair', F=cls(i, pF_box), T=cls(nF + j, pT_box), iou=v, d_raw=raw[nF + j] - raw[i],
                               f_raw=raw[i], aspect=aspect(polys[i])))
    for side, kept, paired, other_all, other_kept, mb in (('F_only', kF, paired_F, T_ids, kT, pF_box),
                                                         ('T_only', kT, paired_T, F_ids, kF, pT_box)):
        for p in sorted(kept - paired):
            c = cls(p, mb)
            below = [q for q in other_all if raw[q] < CUT and iou(shapes[p], shapes[q]) >= PAIR_IOU]
            partial = [q for q in other_kept if iou(shapes[p], shapes[q]) > 0]
            cause = 'score_crossed_cut' if below else 'geometry_change' if partial else 'no_counterpart'
            box_events.append(dict(kind=side, cls=c, cause=cause, raw=raw[p], aspect=aspect(polys[p]),
                                   other_raw=max((raw[q] for q in below), default=None)))
    return dict(rowF=rowF, rowT=rowT, gt_events=gt_events, box_events=box_events, m=m, kF=kF, kT=kT, nF=nF, raw=raw)


def curve(m, nF, raw, nT, cut):
    groups = [{i for i, h in enumerate(m.heights) if lo <= h < hi} for lo, hi in ((0, 4), (4, 8), (8, 16), (16, float('inf')))]
    kF = {i for i in range(nF) if raw[i] >= cut}
    kT = {nF + j for j in range(nT) if raw[nF + j] >= cut}
    a, ia = p0.match_pairs(m, kF)
    b, ib = p0.match_pairs(m, kT)
    return p0.row_from_pairs(m, groups, a, ia, kF), p0.row_from_pairs(m, groups, b, ib, kT)


def run(a):
    import numpy as np
    rows, policy, lr2, geoms, scores = da._context(a.bundle)     # puts rev4 on sys.path
    from rev4.decision_core import CandidateMatcher
    recs = ft.load_records(a.infer_dir)
    require(len(recs) == 600, f'Expected 600 records, got {len(recs)}')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    ref = {(r['source'], r['condition']): r for r in csv.DictReader(open(a.per_source, encoding='utf-8'))}
    per = {}
    off_checked = 0
    tic = time.perf_counter()
    curves = defaultdict(lambda: np.zeros((len(CUTOFFS), 2, 11), dtype=np.int64))
    curve_rows = defaultdict(dict)
    for k, rec in enumerate(recs):
        row = rows[rec['source']]
        res = analyse_task(rec, row, policy)
        r0 = ref[(rec['source'], rec['condition'])]
        require([res['rowF'][0], res['rowF'][1], res['rowF'][2], res['rowT'][0], res['rowT'][2]] ==
                [int(r0['tp_F']), int(r0['gt']), int(r0['pred_F']), int(r0['tp_T']), int(r0['pred_T'])],
                f'Pool counts differ from per_source.csv for {rec["task"]}')
        if a.official:
            for arm in ('F', 'T'):
                e = rec[arm]
                CandidateMatcher(dict(polygons=e['polygons'], scores=e['packed']), e['raw'], row, policy).assert_official(CUT)
                off_checked += 1
        nT = len(rec['T']['raw'])
        cr = []
        for ci, c in enumerate(CUTOFFS):
            f_, t_ = curve(res['m'], res['nF'], res['raw'], nT, c)
            curves[rec['condition']][ci, 0] += f_
            curves[rec['condition']][ci, 1] += t_
            cr.append((f_, t_))
        curve_rows[rec['condition']][rec['source']] = cr
        per[(rec['source'], rec['condition'])] = dict(rowF=res['rowF'], rowT=res['rowT'], gt_events=res['gt_events'],
                                                      box_events=res['box_events'])
        if k % 100 == 0:
            print(f'{k + 1}/600 ({time.perf_counter() - tic:.0f} s)', flush=True)
    summary = dict(label='Flip-TTA diagnosis on the existing 300-source pilot records (post-hoc, descriptive; no rule chosen). '
                         'T4 -> PENDING_REFERENCE_CHECK.',
                   official_evaluator_checked=off_checked, pool_counts_equal_per_source_csv=True, conditions={})
    for cond in ('lr2', 'clean'):
        items = [(s, v) for (s, c_), v in per.items() if c_ == cond]
        tf = np.sum([v['rowF'] for _, v in items], axis=0)
        tt = np.sum([v['rowT'] for _, v in items], axis=0)
        ge = [dict(e, source=s) for s, v in items for e in v['gt_events']]
        be = [dict(e, source=s) for s, v in items for e in v['box_events']]
        lost = [e for e in ge if e['kind'] == 'lost']
        gained = [e for e in ge if e['kind'] == 'gained']
        mo = [e for e in ge if e['kind'] == 'matching_only']
        fp_like = ('no_gt', 'dup_or_competing', 'invalid')
        fp_removed = [e for e in be if e['kind'] == 'F_only' and e['cls'] in fp_like]
        fp_added = [e for e in be if e['kind'] == 'T_only' and e['cls'] in fp_like]
        tp_box_only_F = [e for e in be if e['kind'] == 'F_only' and e['cls'] == 'tp']
        tp_box_only_T = [e for e in be if e['kind'] == 'T_only' and e['cls'] == 'tp']
        pairs = [e for e in be if e['kind'] == 'pair']
        d_raw = np.array([e['d_raw'] for e in pairs])
        iou_pairs = np.array([e['iou'] for e in pairs])

        def hb(h):
            return p0.size_bin(h)

        def tab(evts, key):
            return dict(sorted(Counter(key(e) for e in evts).items()))
        # per-source concentration of GT events and net
        src_lost = Counter(e['source'] for e in lost)
        src_gain = Counter(e['source'] for e in gained)
        cons = dict(sources_with_lost=len(src_lost), sources_with_gained=len(src_gain),
                    top10_share_of_lost=sum(n for _, n in src_lost.most_common(10))/max(len(lost), 1),
                    top10_share_of_gained=sum(n for _, n in src_gain.most_common(10))/max(len(gained), 1))
        # PR curves (descriptive) and T(.60) vs F at the cutoff with the closest recall
        cv = curves[cond]
        cur = []
        for ci, c in enumerate(CUTOFFS):
            fF, fT = p0.totals(cv[ci, 0]), p0.totals(cv[ci, 1])
            cur.append(dict(cut=c, F=dict(recall=fF['tp']/fF['gt'], fp=fF['fp'], hmean=fF['hmean']),
                            T=dict(recall=fT['tp']/fT['gt'], fp=fT['fp'], hmean=fT['hmean'])))
        t_at = p0.totals(tt)
        rec_T = t_at['tp']/t_at['gt']
        # F's FP at T's recall by linear interpolation along F's curve (recall decreases with the cutoff)
        fr = [(x['F']['recall'], x['F']['fp']) for x in cur]
        interp = None
        for (r1, f1), (r2, f2) in zip(fr, fr[1:]):
            if r2 <= rec_T <= r1 and r1 != r2:
                interp = f1 + (f2 - f1)*(r1 - rec_T)/(r1 - r2)
        ci_star = min(range(len(CUTOFFS)), key=lambda i: abs(cur[i]['F']['recall'] - rec_T))
        srcs = sorted(curve_rows[cond])
        Fstar = np.array([curve_rows[cond][s][ci_star][0] for s in srcs])
        T60 = np.array([curve_rows[cond][s][CUTOFFS.index(.6)][1] for s in srcs])
        summary['conditions'][cond] = dict(
            F=p0.totals(tf), T=p0.totals(tt),
            gt=dict(lost=len(lost), gained=len(gained), net=len(gained) - len(lost), matching_only=len(mo),
                    lost_by_cause=tab(lost, lambda e: e['cause']), gained_by_cause=tab(gained, lambda e: e['cause']),
                    lost_by_height=tab(lost, lambda e: hb(e['height'])), gained_by_height=tab(gained, lambda e: hb(e['height'])),
                    lost_by_F_score=tab(lost, lambda e: sbin(e['own_score'])),
                    lost_score_crossed_T_score=dict(median=float(np.median([e['other_score'] for e in lost if e['cause'] == 'score_crossed_cut'])) if any(e['cause'] == 'score_crossed_cut' for e in lost) else None),
                    concentration=cons),
            boxes=dict(pairs=len(pairs), pair_iou_median=float(np.median(iou_pairs)) if len(pairs) else None,
                       pair_iou_lt_0p8=int((iou_pairs < .8).sum()),
                       d_raw_T_minus_F=dict(median=float(np.median(d_raw)), p10=float(np.percentile(d_raw, 10)),
                                            p90=float(np.percentile(d_raw, 90)), share_negative=float((d_raw < 0).mean())) if len(pairs) else None,
                       d_raw_by_F_class=tab(pairs, lambda e: e['F']),
                       d_raw_median_tp_pairs=float(np.median([e['d_raw'] for e in pairs if e['F'] == 'tp'])),
                       d_raw_median_fp_pairs=float(np.median([e['d_raw'] for e in pairs if e['F'] != 'tp'])),
                       pair_class_change=tab(pairs, lambda e: f"{e['F']}->{e['T']}"),
                       fp_removed=len(fp_removed), fp_added=len(fp_added),
                       fp_removed_by_cause=tab(fp_removed, lambda e: e['cause']), fp_added_by_cause=tab(fp_added, lambda e: e['cause']),
                       fp_removed_by_class=tab(fp_removed, lambda e: e['cls']), fp_added_by_class=tab(fp_added, lambda e: e['cls']),
                       fp_removed_by_F_score=tab(fp_removed, lambda e: sbin(e['raw'])),
                       fp_removed_by_aspect=tab(fp_removed, lambda e: abin(e['aspect'])),
                       tp_box_only_F=len(tp_box_only_F), tp_box_only_T=len(tp_box_only_T),
                       tp_box_only_F_by_cause=tab(tp_box_only_F, lambda e: e['cause']),
                       tp_box_only_F_by_score=tab(tp_box_only_F, lambda e: sbin(e['raw']))),
            pr_curve=cur,
            T60_vs_F_curve=dict(T_recall=rec_T, T_fp=t_at['fp'], F_fp_interpolated_at_T_recall=interp,
                                nearest_F_cut=CUTOFFS[ci_star], nearest_F=cur[ci_star]['F'],
                                T60_minus_F_nearest=p0.paired_boot(T60, Fstar),
                                note='Descriptive control: is T better than simply raising F\'s score cutoff? The cutoff is '
                                     'chosen post hoc by recall proximity; not a configuration choice.'))
        with gzip.open(out/f'events_{cond}.json.gz', 'wt', encoding='utf-8') as fh:
            json.dump(dict(gt=ge, box=be), fh)
    # same-source Clean vs LR2 (net contributions from the pilot CSV)
    net = defaultdict(dict)
    for (s, c_), r0 in ref.items():
        net[s][c_] = float(r0['net_contribution'])
    both = [(v['lr2'], v['clean']) for v in net.values() if 'lr2' in v and 'clean' in v]
    x, y = np.array([b[0] for b in both]), np.array([b[1] for b in both])
    rk = lambda v: np.argsort(np.argsort(v))  # noqa: E731
    summary['same_source_lr2_vs_clean'] = dict(sources=len(both), spearman=float(np.corrcoef(rk(x), rk(y))[0, 1]),
                                               both_up=int(((x > 0) & (y > 0)).sum()), both_down=int(((x < 0) & (y < 0)).sum()),
                                               opposite=int(((x > 0) & (y < 0)).sum() + ((x < 0) & (y > 0)).sum()))
    save_json(out/'fliptta_diag.json', summary)
    print(json.dumps({c: {k: v for k, v in summary['conditions'][c].items() if k != 'pr_curve'} for c in summary['conditions']},
                     indent=1, default=str))
    print(json.dumps(summary['same_source_lr2_vs_clean']))


# ---------------------------------------------------------------- figures (hash-selected cases)

def figures(a):
    import io  # noqa: F401
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from shapely.geometry import shape
    rows, policy, lr2, geoms, scores = da._context(a.bundle)
    from rev4.training_data import lr2_dimensions
    recs = {(r['source'], r['condition']): r for r in ft.load_records(a.infer_dir)}
    out = Path(a.output)/'figures'
    out.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(a.font, 15)
    cases = []
    for cond in ('lr2', 'clean'):
        ev = json.load(gzip.open(Path(a.output)/f'events_{cond}.json.gz', 'rt', encoding='utf-8'))
        for e in ev['gt']:
            if e['kind'] in ('lost', 'gained'):
                key = f"gt_{e['kind']}_{'score' if e['cause'] == 'score_crossed_cut' else 'other'}"
                cases.append(dict(e, cond=cond, cat=key, cid=f"{cond}:{e['source']}:g{e['g']}"))
    rank = lambda c: hashlib.sha256((FIG_SEED + c['cid']).encode()).hexdigest()  # noqa: E731
    chosen = []
    for cat in ('gt_lost_score', 'gt_lost_other', 'gt_gained_score', 'gt_gained_other'):
        pool = sorted((c for c in cases if c['cat'] == cat), key=rank)
        chosen += pool[:a.per_category]
    made = []
    for n, c in enumerate(chosen, 1):
        row = rows[c['source']]
        rec = recs[(c['source'], c['cond'])]
        g = shape(row['gt_clean']['valid'][c['g']]['geometry'])
        W, H = row['clean_size']
        x0, y0, x1, y1 = g.bounds
        mg = max(24., .6*max(x1 - x0, y1 - y0))
        box = (max(0, x0 - mg), max(0, y0 - mg), min(W, x1 + mg), min(H, y1 + mg))
        with Image.open(Path(a.clean_root)/row['clean_relpath']) as im:
            clean = im.convert('RGB')
        img = clean.resize(lr2_dimensions(clean.size), Image.Resampling.BICUBIC) if c['cond'] == 'lr2' else clean   # display only
        sx, sy = img.width/W, img.height/H
        lb = (int(box[0]*sx), int(box[1]*sy), int(np.ceil(box[2]*sx)), int(np.ceil(box[3]*sy)))
        crop = img.crop(lb)
        k = max(1, min(8, 700//max(crop.width, 1), 300//max(crop.height, 1)))
        crop = crop.resize((crop.width*k, crop.height*k), Image.NEAREST)
        d = ImageDraw.Draw(crop)

        def px(coords):
            q = np.asarray(coords, float).reshape(-1, 2)
            return [((x*sx - lb[0])*k, (y*sy - lb[1])*k) for x, y in q]
        for e in row['gt_clean']['valid']:
            d.polygon(px(np.asarray(shape(e['geometry']).exterior.coords)[:-1].reshape(-1)), outline=(0, 200, 0), width=1)
        d.polygon(px(np.asarray(g.exterior.coords)[:-1].reshape(-1)), outline=(0, 230, 0), width=3)
        for arm, col, wid in (('F', (30, 110, 255), 3), ('T', (255, 140, 0), 2)):
            e = rec[arm]
            for poly, r in zip(e['polygons'], e['raw']):
                q = da.polygon_of(poly)
                if q is None or not q.intersects(shape(row['gt_clean']['valid'][c['g']]['geometry']).buffer(mg)):
                    continue
                d.polygon(px(poly), outline=col if r >= CUT else tuple(v//2 + 100 for v in col), width=wid if r >= CUT else 1)
        canvas = Image.new('RGB', (max(crop.width, 760), crop.height + 66), 'white')
        canvas.paste(crop, (0, 0))
        cd = ImageDraw.Draw(canvas)
        cd.text((6, crop.height + 4), f"#{n} {c['cat']} · {c['cond']} · nguồn {c['source']} · GT {c['g']} · cao {c['height']:.1f} px LR2 · "
                                     f"nguyên nhân {c['cause']}", fill='black', font=font)
        os_ = '-' if c['other_score'] is None else f"{c['other_score']:.3f}"
        cd.text((6, crop.height + 24), f"score bên có GT {c['own_score']:.3f} · score bên kia {os_} · cắt {CUT} · "
                                      f"IoU tốt nhất bên kia {c['best_iou_other_kept']:.2f}", fill='black', font=font)
        cd.text((6, crop.height + 44), 'xanh lá = GT (đậm: GT đang xét) · xanh dương = F · cam = T · nét mảnh nhạt = dưới ngưỡng 0,60 · '
                                      f'phóng ×{k}' + (' · LR2 tạo lại để hiển thị (bicubic)' if c['cond'] == 'lr2' else ''), fill='black', font=font)
        path = out/f"{n:02d}_{c['cat']}_{c['cond']}.png"
        canvas.save(path)
        made.append(dict(path=str(path), case=c['cid'], cat=c['cat']))
    save_json(out/'figures_index.json', dict(seed=FIG_SEED, per_category=a.per_category,
                                             rule='per category, smallest sha256(seed + case id); no outcome-based choice', figures=made))
    print('\n'.join(m['path'] for m in made))


def main():
    sys.stdout.reconfigure(errors='backslashreplace')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    f = sub.add_parser('figures')
    for s in (r, f):
        s.add_argument('--bundle', required=True, type=Path)
        s.add_argument('--infer-dir', required=True, type=Path)
        s.add_argument('--output', required=True, type=Path)
    r.add_argument('--per-source', required=True, type=Path)
    r.add_argument('--official', action='store_true')
    f.add_argument('--clean-root', required=True, type=Path)
    f.add_argument('--per-category', type=int, default=3)
    f.add_argument('--font', default='C:/Windows/Fonts/arial.ttf')
    a = ap.parse_args()
    run(a) if a.command == 'run' else figures(a)


if __name__ == '__main__':
    main()
