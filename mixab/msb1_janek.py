"""MSB1 (Mask-Split B1) pilot, CPU only, from the existing T4 dump. Spec (locked before running): docs/research/janek/msb1_spec_20261122_janek.md

Rule (predictions + box geometry only, never GT):
  parents  = B1 epsilon_005 boxes with raw >= .5   (= the B1 baseline output)
  children = B1 mask_040 boxes with raw >= .5
  child c is assigned to the parent p with the largest frac = area(c & p)/area(c) if frac >= .8 (ties: lower p)
  a parent with >= 2 assigned children is replaced by them; other parents stay; nothing else is added.
Ceiling: exact maximum of global Hmean over "split / do not split" per candidate parent (same assignment), by
Dinkelbach across sources and full 2^k enumeration with real maximum matching inside a source (k <= K_MAX);
sources with more candidates use a safe bound (TP <= matching on the union, Pred >= smallest additive Pred).
Diagnostic (mechanism only): the same rule with parents = B1 `base` (mask .3, eps .01) isolates the mask threshold.

Idea formed AFTER seeing Phase 0 on split A -> exploratory; split B descriptive only.
"""
import argparse
import gzip
import hashlib
import json
import sys
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

from mixab import require, save_json 
from mixab import dualagree_janek as da 
from mixab import phase0_janek as p0 

CUT = .5            # score cutoff for parents AND children (= baseline cutoff, locked)
FRAC = .8           # containment threshold (locked)
K_MAX = 12          # full enumeration up to 2^12 subsets per source
SEED = 20261123


# ---------------------------------------------------------------- pure rule (CPU-testable)

def assign_children(parents, children, frac=FRAC):
    """parents/children: lists of shapely polygons (None allowed) -> {parent index: [child indices]}."""
    out = {}
    for j, c in enumerate(children):
        if c is None:
            continue
        best = None
        for i, p in enumerate(parents):
            if p is None or not p.intersects(c):
                continue
            f = p.intersection(c).area/c.area
            if f >= frac and (best is None or f > best[0]):
                best = (f, i)
        if best is not None:
            out.setdefault(best[1], []).append(j)
    return out


def split_candidates(assignment):
    return {p: cs for p, cs in assignment.items() if len(cs) >= 2}


def kept_set(parent_ids, candidates, chosen, n_parent_pool):
    """Pool indices kept when the parents in `chosen` are split (children indexed after the parent pool)."""
    kept = {i for i in parent_ids if i not in chosen}
    for p in chosen:
        kept |= {n_parent_pool + j for j in candidates[p]}
    return kept


def dinkelbach(options, gt_total, start=0.):
    """options: per source a list of (tp, pred). Maximise 2*sum(tp)/(gt_total + sum(pred)) exactly.
    Returns (hmean, chosen option index per source)."""
    lam = start
    for _ in range(200):
        pick = [max(range(len(o)), key=lambda k: (2*o[k][0] - lam*o[k][1], -k)) for o in options]
        tp = sum(o[k][0] for o, k in zip(options, pick))
        pred = sum(o[k][1] for o, k in zip(options, pick))
        h = 2*tp/(gt_total + pred) if gt_total + pred else 0.
        if h <= lam + 1e-15:
            return h, pick
        lam = h
    raise RuntimeError('Dinkelbach did not converge')


# ---------------------------------------------------------------- per-source evaluation

class Source:
    def __init__(self, e_parent, e_child, row, policy):
        from rev4.decision_core import CandidateMatcher
        self.n_p = len(e_parent['raw'])
        pred = dict(polygons=e_parent['polygons'] + e_child['polygons'], scores=e_parent['packed'] + e_child['packed'])
        self.m = CandidateMatcher(pred, e_parent['raw'] + e_child['raw'], row, policy)
        self.groups = [{i for i, h in enumerate(self.m.heights) if lo <= h < hi}
                       for lo, hi in ((0, 4), (4, 8), (8, 16), (16, float('inf')))]
        self.parent_ids = [i for i, r in enumerate(e_parent['raw']) if r >= CUT]
        child_ids = [j for j, r in enumerate(e_child['raw']) if r >= CUT]
        pp = [da.polygon_of(e_parent['polygons'][i]) for i in self.parent_ids]
        cp = [da.polygon_of(e_child['polygons'][j]) for j in child_ids]
        a = assign_children(pp, cp)
        self.candidates = {self.parent_ids[i]: [child_ids[j] for j in cs] for i, cs in split_candidates(a).items()}
        self.e_parent, self.e_child, self.row = e_parent, e_child, row

    def evaluate(self, chosen):
        kept = kept_set(self.parent_ids, self.candidates, set(chosen), self.n_p)
        pairs, invalid = p0.match_pairs(self.m, kept)
        return p0.row_from_pairs(self.m, self.groups, pairs, invalid, kept), {g for g, _ in pairs}, kept

    def counted(self, idx):
        return int(idx in self.m.geometries or idx in self.m.invalid)

    def oracle_options(self):
        """[(tp, pred)] over all subsets (exact) or one safe bound option; flag exact."""
        cands = sorted(self.candidates)
        if len(cands) <= K_MAX:
            opts = []
            for r in range(len(cands) + 1):
                for sub in combinations(cands, r):
                    row, _, _ = self.evaluate(sub)
                    opts.append((row[0], row[2]))
            return opts, True
        base_row, _, _ = self.evaluate(())
        union = set(self.parent_ids) | {self.n_p + j for p in cands for j in self.candidates[p]}
        pairs, _ = p0.match_pairs(self.m, union)
        d = [sum(self.counted(self.n_p + j) for j in self.candidates[p]) - self.counted(p) for p in cands]
        return [(len(pairs), base_row[2] + sum(min(0, x) for x in d))], False


# ---------------------------------------------------------------- run

def run(a):
    import numpy as np
    from mixab import reselect_split_janek as rs 
    rows, policy, lr2, geoms, scores, dl, dc, ident = p0.load_all(a.dump, a.bundle)
    gi = {g: i for i, g in enumerate(geoms)}
    out_dir = Path(a.output)
    require(not (out_dir/'msb1_result.json').exists(), f'Refusing to overwrite {out_dir}/msb1_result.json')
    out_dir.mkdir(parents=True, exist_ok=True)
    sids = sorted(dl, key=lambda s: lr2[s])
    split = np.array([rs.split_of(lr2[s]) for s in sids])
    res = dict(label='MSB1 pilot (CPU, T4 dump -> PENDING_REFERENCE_CHECK). Idea formed after Phase 0 on A: exploratory. '
                     'B descriptive only.', spec='docs/research/janek/msb1_spec_20261122_janek.md', dump_identity_sha256=ident,
               rule=dict(parent='epsilon_005', child='mask_040', cut=CUT, frac=FRAC, min_children=2))
    detail = {}
    for variant, gpar in (('MSB1', 'epsilon_005'), ('diag_base_parent', 'base')):
        tic = time.perf_counter()
        base_rows, rule_rows, per = [], [], []
        oracle_opts, exact_flags = [], []
        for k, sid in enumerate(sids):
            s = Source(dl[sid][gi[gpar]], dl[sid][gi['mask_040']], rows[lr2[sid]], policy)
            b_row, b_tp, _ = s.evaluate(())
            r_row, r_tp, _ = s.evaluate(tuple(s.candidates))
            base_rows.append(b_row); rule_rows.append(r_row)
            rec = dict(source=lr2[sid], sid=sid, split=split[k], candidates=len(s.candidates),
                       children=sum(len(v) for v in s.candidates.values()),
                       rescued=[s.m.ann_ids[g] for g in sorted(r_tp - b_tp)], lost=[s.m.ann_ids[g] for g in sorted(b_tp - r_tp)],
                       rescued_tiny=sum(1 for g in r_tp - b_tp if s.m.heights[g] < 4),
                       lost_tiny=sum(1 for g in b_tp - r_tp if s.m.heights[g] < 4))
            if variant == 'MSB1':
                # single-split effect of each candidate parent (illustration only; interactions ignored)
                singles = []
                for p in sorted(s.candidates):
                    row1, tp1, _ = s.evaluate((p,))
                    singles.append(dict(parent=p, children=s.candidates[p], d_tp=row1[0] - b_row[0], d_pred=row1[2] - b_row[2],
                                        raw=s.e_parent['raw'][p]))
                rec['singles'] = singles
                if split[k] == 'A':
                    o, ex = s.oracle_options()
                    oracle_opts.append(o); exact_flags.append(ex)
            per.append(rec)
            if k % 500 == 0:
                print(f'{variant}: {k + 1}/{len(sids)} ({time.perf_counter() - tic:.0f} s)', flush=True)
        base_rows, rule_rows = np.array(base_rows), np.array(rule_rows)
        v = dict(parent_geometry=gpar)
        for sname in ('A', 'B'):
            msk = split == sname
            tb, tr = base_rows[msk].sum(0), rule_rows[msk].sum(0)
            pr = [x for x, mm in zip(per, msk) if mm]
            v[sname] = dict(base=p0.totals(tb), rule=p0.totals(tr), boot=p0.paired_boot(rule_rows[msk], base_rows[msk]),
                            delta_tiny_pp=(p0.totals(tr)['tiny_recall'] - p0.totals(tb)['tiny_recall'])*100,
                            parents_split=sum(x['candidates'] for x in pr), children_in=sum(x['children'] for x in pr),
                            sources_with_split=sum(x['candidates'] > 0 for x in pr),
                            gt_rescued=sum(len(x['rescued']) for x in pr), gt_lost=sum(len(x['lost']) for x in pr),
                            tiny_rescued=sum(x['rescued_tiny'] for x in pr), tiny_lost=sum(x['lost_tiny'] for x in pr))
        if variant == 'MSB1':
            msk = split == 'A'
            gt_a = int(base_rows[msk, 1].sum())
            h_or, pick = dinkelbach(oracle_opts, gt_a, start=p0.hmean(base_rows[msk].sum(0)))
            v['A']['oracle'] = dict(hmean_upper_bound=h_or, delta_pp_upper_bound=(h_or - p0.hmean(base_rows[msk].sum(0)))*100,
                                    sources_exact=int(sum(exact_flags)), sources_safe_bound=int(len(exact_flags) - sum(exact_flags)),
                                    gate_reachable=(h_or - p0.hmean(base_rows[msk].sum(0)))*100 >= .3,
                                    note='Exact max over split/no-split per candidate parent with the locked assignment; '
                                         'sources with > K_MAX candidates use a safe (looser) bound. Does not bound other '
                                         'containment/score thresholds or geometries.')
            # FP at same recall vs the 72 B1 settings recomputed from the SAME T4 dump
            with np.load(Path(a.cache)/'t4_counts_B1.npz') as z:
                srcs, c72 = list(z['source_ids']), z['counts']
            idx = {s_: i for i, s_ in enumerate(srcs)}
            a_rows = [idx[lr2[s_]] for s_, mm in zip(sids, msk) if mm]
            t72 = c72[a_rows].sum(0)
            rr = v['A']['rule']
            recall = rr['tp']/rr['gt']
            ok = [int(t[2] - t[0]) for t in t72 if t[0]/t[1] >= recall]
            v['A']['b1_min_fp_at_rule_recall'] = min(ok) if ok else None
            base_check = t72[geoms.index('epsilon_005')*len(scores) + scores.index(.5)]
            require(list(base_check[:3]) == list(base_rows[msk].sum(0)[:3]), 'Baseline counts differ from the 72-setting recompute')
            v['A']['gates'] = dict(lr2_gain_ge_0p3=v['A']['boot']['delta_pp'] >= .3,
                                   tiny_ge_minus_3=v['A']['delta_tiny_pp'] >= -3,
                                   fp_lower_at_same_recall=(not ok) or rr['fp'] < min(ok))
            v['A']['pass'] = all(v['A']['gates'].values())
            # single-split outcome summary on A (illustration)
            hb = p0.hmean(base_rows[msk].sum(0))
            sing = [dict(sg, source=x['source'], sid=x['sid']) for x in per if x['split'] == 'A' for sg in x['singles']]
            v['A']['single_split_effects'] = dict(
                parents=len(sing), helps=sum(2*s_['d_tp'] - hb*s_['d_pred'] > 0 for s_ in sing),
                hurts=sum(2*s_['d_tp'] - hb*s_['d_pred'] < 0 for s_ in sing),
                d_tp_hist=dict(sorted(Counter(s_['d_tp'] for s_ in sing).items())),
                children_hist=dict(sorted(Counter(len(s_['children']) for s_ in sing).items())))
            detail['singles_A'] = sing
        res[variant] = v
        detail[variant] = per
        print(f'{variant} done ({time.perf_counter() - tic:.0f} s)', flush=True)
    save_json(out_dir/'msb1_result.json', res)
    with gzip.open(Path(a.cache)/'msb1_detail.json.gz', 'wt', encoding='utf-8') as f:
        json.dump(detail, f, default=str)
    print(json.dumps({k: (v if not isinstance(v, dict) else {s: (v[s] if s not in ('A', 'B') else {
        kk: v[s][kk] for kk in ('base', 'rule', 'boot', 'delta_tiny_pp', 'parents_split', 'gt_rescued', 'gt_lost', 'oracle', 'gates', 'pass')
        if kk in v[s]}) for s in v}) for k, v in res.items()}, indent=1, default=str))


# ---------------------------------------------------------------- example panels

def examples(a):
    import io
    import zipfile
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from shapely.geometry import shape
    rows, policy, lr2, geoms, scores, dl, dc, ident = p0.load_all(a.dump, a.bundle)    # puts rev4 on sys.path
    from rev4.training_data import lr2_dimensions
    gi = {g: i for i, g in enumerate(geoms)}
    detail = json.load(gzip.open(Path(a.cache)/'msb1_detail.json.gz', 'rt', encoding='utf-8'))
    sing = detail['singles_A']
    rk = lambda s_: hashlib.sha256(f'{SEED}:{s_["source"]}:{s_["parent"]}'.encode()).hexdigest()   # noqa: E731
    good = sorted([s_ for s_ in sing if s_['d_tp'] > 0 and s_['d_tp'] >= s_['d_pred'] - 1], key=rk)[:4]
    bad = sorted([s_ for s_ in sing if s_['d_tp'] <= 0], key=rk)[:4]
    out = Path(a.output)/'examples'
    out.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(a.font, 16)
    zf = zipfile.ZipFile(a.clean_zip)
    made = []
    for tag, items in (('tach_dung', good), ('tach_sai', bad)):
        for n, s_ in enumerate(items, 1):
            row = rows[s_['source']]
            ep, ec = dl[s_['sid']][gi['epsilon_005']], dl[s_['sid']][gi['mask_040']]
            parent = ep['polygons'][s_['parent']]
            kids = [ec['polygons'][j] for j in s_['children']]
            pts = np.asarray(parent).reshape(-1, 2)
            W, H = row['clean_size']
            x0, y0 = pts.min(0); x1, y1 = pts.max(0)
            mg = max(16., .3*max(x1 - x0, y1 - y0))
            box = (max(0, x0 - mg), max(0, y0 - mg), min(W, x1 + mg), min(H, y1 + mg))
            with Image.open(io.BytesIO(zf.read(row['clean_relpath']))) as im:
                clean = im.convert('RGB')
            lr = clean.resize(lr2_dimensions(clean.size), Image.Resampling.BICUBIC)     # display only
            sx, sy = lr.width/W, lr.height/H
            lb = (int(box[0]*sx), int(box[1]*sy), int(np.ceil(box[2]*sx)), int(np.ceil(box[3]*sy)))
            crop = lr.crop(lb)
            k = max(1, min(8, 900//max(crop.width, 1), 260//max(crop.height, 1)))
            img = crop.resize((crop.width*k, crop.height*k), Image.NEAREST)
            d = ImageDraw.Draw(img)

            def px(c):
                q = np.asarray(c, float).reshape(-1, 2)
                return [((x*sx - lb[0])*k, (y*sy - lb[1])*k) for x, y in q]
            for g in row['gt_clean']['valid']:
                gp = np.asarray(shape(g['geometry']).exterior.coords)[:-1].reshape(-1)
                d.polygon(px(gp), outline=(0, 200, 0), width=2)
            d.polygon(px(parent), outline=(30, 110, 255), width=4)
            for c in kids:
                d.polygon(px(c), outline=(255, 140, 0), width=2)
            canvas = Image.new('RGB', (max(img.width, 900), img.height + 70), 'white')
            canvas.paste(img, (0, 0))
            cd = ImageDraw.Draw(canvas)
            cd.text((6, img.height + 6), f'{tag} #{n} · nguồn {s_["source"]} · tách 1 box cha (xanh dương, score {s_["raw"]:.2f}) '
                                         f'thành {len(kids)} box con (cam)', fill='black', font=font)
            cd.text((6, img.height + 30), f'Chỉ tách box này: ΔTP {s_["d_tp"]:+d}, ΔPred {s_["d_pred"]:+d} · xanh lá = GT · '
                                          f'LR2 phóng ×{k} (tạo lại để hiển thị)', fill='black', font=font)
            path = out/f'{tag}_{n}.png'
            canvas.save(path)
            made.append(str(path))
    # descriptive overview of the A source with the most split parents (not part of the locked example selection)
    worst = max((x for x in detail['MSB1'] if x['split'] == 'A'), key=lambda x: x['candidates'])
    row = rows[worst['source']]
    ep, ec = dl[worst['sid']][gi['epsilon_005']], dl[worst['sid']][gi['mask_040']]
    with Image.open(io.BytesIO(zf.read(row['clean_relpath']))) as im:
        clean = im.convert('RGB')
    W, H = row['clean_size']
    sc = 1000/W
    img = clean.resize(lr2_dimensions(clean.size), Image.Resampling.BICUBIC).resize((1000, int(H*sc)), Image.NEAREST)
    d = ImageDraw.Draw(img)

    def px2(c):
        return [(x*sc, y*sc) for x, y in np.asarray(c, float).reshape(-1, 2)]
    for g in row['gt_clean']['valid']:
        d.polygon(px2(np.asarray(shape(g['geometry']).exterior.coords)[:-1].reshape(-1)), outline=(0, 200, 0), width=1)
    for sg in (s_ for s_ in sing if s_['source'] == worst['source']):
        d.polygon(px2(ep['polygons'][sg['parent']]), outline=(30, 110, 255), width=2)
        for j in sg['children']:
            d.polygon(px2(ec['polygons'][j]), outline=(255, 140, 0), width=1)
    canvas = Image.new('RGB', (img.width, img.height + 40), 'white')
    canvas.paste(img, (0, 0))
    ImageDraw.Draw(canvas).text((6, img.height + 8), f'Toàn cảnh nguồn {worst["source"]} (A): {worst["candidates"]} box cha bị '
                                f'tách (xanh dương) -> {worst["children"]} box con (cam); GT xanh lá. Mô tả, ngoài mẫu khóa.',
                                fill='black', font=font)
    path = out/'toan_canh_nguon_tach_nhieu_nhat.png'
    canvas.save(path)
    made.append(str(path))
    print('\n'.join(made))


def main():
    sys.stdout.reconfigure(errors='backslashreplace')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    for name in ('run', 'examples'):
        s = sub.add_parser(name)
        s.add_argument('--dump', required=True, type=Path)
        s.add_argument('--bundle', required=True, type=Path)
        s.add_argument('--cache', required=True, type=Path)
        s.add_argument('--output', required=True, type=Path)
    sub.choices['examples'].add_argument('--clean-zip', required=True, type=Path)
    sub.choices['examples'].add_argument('--font', default='C:/Windows/Fonts/arial.ttf')
    a = ap.parse_args()
    run(a) if a.command == 'run' else examples(a)


if __name__ == '__main__':
    main()
