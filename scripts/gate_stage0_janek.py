"""Giai đoạn 0 của PLAN_DAT_CONG_NHOM_20261125 (CPU, artifact có sẵn, chỉ split A; KHÔNG fit, KHÔNG chọn lại, không B/TEST).

Inputs
  * dev A box-quality run (Kaggle T4): B1 mask_040 pool raw >= .20 + 13 features per task -> F (B1 .40/.60) and FULL
  * T4 dumps (LR2 Colab, Clean Kaggle): B1 and B-Clean, 9 geometries, raw >= .20 -> B-Clean .40/.60, other B1 geometries
Outputs (docs/handoff/artifacts/janek/gate_stage0_20261125/)
  stage0_janek.json        GT-miss taxonomy for F/FULL, FP taxonomy, B-Clean vs B1/FULL, pool oracles, gap table
  review/                  crops of FPs removed / kept by FULL with no GT overlap (for a visual missing-label check)
Every count uses the official evaluator (lowres.evaluate_shared.evaluate_image) or Tuan's CandidateMatcher edges
(IoU >= match_iou after the ROI/ignore filter). Oracles use GT: they are DIAGNOSTIC UPPER BOUNDS, not methods.
Run from the workspace root:  python scripts/gate_stage0_janek.py
"""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding='utf-8')
from mixab import boxquality_devA_janek as dv  # noqa: E402
from mixab import boxquality_janek as bq  # noqa: E402
from mixab import dualagree_janek as da  # noqa: E402
from mixab import read_json, save_json  # noqa: E402
from mixab.monkey_janek import monkey_root, use_monkey  # noqa: E402

RUN = ROOT/'runtime/kaggle_downloads/boxquality_devA_T4_20261125-084951/out'
ART = ROOT/'docs/handoff/artifacts/janek/boxquality_v1_T4'
DUMPS = dict(lr2=ROOT/'runtime/phase0_inputs/dualagree_dump_v1_T4',
             clean=ROOT/'runtime/kaggle_downloads/csb1_clean_T4_20261123-110923/out/clean_dump')
CLEAN_IMG = ROOT/'runtime/kaggle_upload/cs431-csb1-clean/clean'
OUT = ROOT/'docs/handoff/artifacts/janek/gate_stage0_20261125'
GEO_IOU, GEO_COVER = .1, .5            # "wrong geometry": max IoU in [.1, .5) or a box covering >= 50 % of the GT
BINS = dv.BINS
REVIEW_SEED = '20261125:stage0:review:'
GATE = dict(mean=79.0019, clean=80.2202)  # from PLAN §1 (recomputed below as a check)


def hb(h):
    for lo, hi in BINS:
        if lo <= h < hi:
            return f'{lo}-{hi}'


def H(tp, gt, pred):
    return 2*tp/(gt + pred) if gt + pred else 0.


def main():
    use_monkey()
    import numpy as np
    from shapely.geometry import shape
    from shapely.strtree import STRtree
    from lowres.evaluate_shared import evaluate_image
    from rev4.decision_core import CandidateMatcher
    from rev4.decision_plan import GEOMETRIES
    from mixab import phase0_janek as p0
    geoms = [g[0] for g in GEOMETRIES]
    bundle = monkey_root()/'docs/handoff/artifacts/pr_quad_audit_v1'
    rows, A, _ = dv._context(bundle)
    policy = read_json(bundle/'provenance/textvqa_evaluation_protocol.json')
    model = dv.load_model(ART/'model_janek.npz', ART/'boxquality_fit_janek.json')
    recs = {(r['source'], r['condition']): r for r in bq.load_records(RUN/'extract')}
    dumps = {}
    for c, d in DUMPS.items():
        ident = read_json(d/'dump_identity.json')['identity_sha256']
        dumps[c] = {k: da.load_model_dump(d/k, len(geoms), ident) for k in ('B1', 'B-Clean')}
    gi = geoms.index('mask_040')

    res = dict(label='Stage 0 (CPU) on development A: error taxonomy, B-Clean vs B1/FULL, pool oracles. Oracles use GT and are '
                     'diagnostic upper bounds, NOT methods or achievable gains. No fitting, no selection, no B/TEST.',
               plan='docs/research/janek/PLAN_DAT_CONG_NHOM_20261125_janek.md', sources=len(A), conditions={})
    review_pool = []
    for c in dv.CONDS:
        tot = defaultdict(lambda: np.zeros(3, dtype=np.int64))       # arm -> tp, gt, pred
        miss = {arm: Counter() for arm in ('F', 'FULL')}
        miss_bin = {arm: Counter() for arm in ('F', 'FULL')}
        bc_only = Counter()                                         # GT found by B-Clean, missed by FULL -> FULL category
        b1_only = Counter()                                         # GT found by FULL, missed by B-Clean
        fp = {arm: Counter() for arm in ('F', 'FULL', 'BC')}
        removed = Counter()                                         # F FPs that FULL drops, by type
        oracle = defaultdict(lambda: np.zeros(3, dtype=np.int64))
        for s in A:
            row, r = rows[s], recs[(s, c)]
            p_full = dv.full_scores(model, r['features'])
            K = dv.kept_indices(r['raw'], p_full, model)
            m = CandidateMatcher(dict(polygons=r['polygons'], scores=r['packed']), r['raw'], row, policy)
            gts = [shape(e['geometry']) for e in row['gt_clean']['valid']]
            ann = [str(e['ann_id']) for e in row['gt_clean']['valid']]
            heights = m.heights.tolist()
            idx = list(m.geometries)
            tree = STRtree([m.geometries[i] for i in idx]) if idx else None

            def best(g):
                """(max IoU, max coverage of g, pool index of max IoU) over the B1 pool after the ROI/ignore filter."""
                bi = bc = 0.
                arg = None
                if tree is None:
                    return 0., 0., None
                for loc in tree.query(g, predicate='intersects'):
                    p = m.geometries[idx[int(loc)]]
                    inter = g.intersection(p).area
                    iou = inter/(g.area + p.area - inter)
                    if iou > bi:
                        bi, arg = iou, idx[int(loc)]
                    bc = max(bc, inter/g.area)
                return bi, bc, arg

            def official(pool_idx, polys, packed):
                pred = dict(polygons=[polys[i] for i in pool_idx], scores=[packed[i] for i in pool_idx])
                o = evaluate_image(row['gt_clean'], row['roi_clean'], pred, policy)
                return o, [pool_idx[j] for j in o['fp_prediction_indices']]
            arms = {}
            for arm in ('F', 'FULL'):
                o, fps = official(K[arm], r['polygons'], r['packed'])
                arms[arm] = (set(o['matched_ann_ids']), fps, set(K[arm]))
                tot[arm] += o['counts'][:3]
            e = dumps[c]['B-Clean'][f'{c}::{s}'][gi]
            kb = [i for i, v in enumerate(e['raw']) if v >= dv.CUT_F]
            ob, fps_b = official(kb, e['polygons'], e['packed'])
            tot['BC'] += ob['counts'][:3]
            bc_matched = set(ob['matched_ann_ids'])
            bc_polys = [shape({'type': 'Polygon', 'coordinates': [np.asarray(e['polygons'][i], float).reshape(-1, 2).tolist()]})
                        for i in kb]
            # ---- GT-miss taxonomy for F and FULL
            cat_of = {}
            for gidx, g in enumerate(gts):
                edge = set(m.edges[gidx])
                for arm in ('F', 'FULL'):
                    matched, _, kept = arms[arm]
                    if ann[gidx] in matched:
                        continue
                    if edge:
                        cat = 'candidate_kept_but_taken' if edge <= kept else 'candidate_score_filtered'
                    else:
                        bi, bc, _ = best(g)
                        cat = ('geometry_merge' if bc >= GEO_COVER else 'geometry_partial') if (bi >= GEO_IOU or bc >= GEO_COVER) else 'no_candidate'
                    miss[arm][cat] += 1
                    miss_bin[arm][(cat, hb(heights[gidx]))] += 1
                    if arm == 'FULL':
                        cat_of[gidx] = cat
                if ann[gidx] in bc_matched and ann[gidx] not in arms['FULL'][0]:
                    bc_only[cat_of[gidx]] += 1
                    bc_only[('bin', hb(heights[gidx]))] += 1
                if ann[gidx] in arms['FULL'][0] and ann[gidx] not in bc_matched:
                    b1_only[hb(heights[gidx])] += 1
            # ---- FP taxonomy (official FP indices)
            def fp_type(poly):
                bi = bc = 0.
                for g in gts:
                    if poly.intersects(g):
                        inter = g.intersection(poly).area
                        bi = max(bi, inter/(g.area + poly.area - inter)); bc = max(bc, inter/poly.area)
                if bi >= .5:
                    return 'duplicate_or_taken'
                if bi >= GEO_IOU or bc >= .3:
                    return 'geometry'
                return 'no_gt_overlap'
            for arm in ('F', 'FULL'):
                for p in arms[arm][1]:
                    t = fp_type(m.geometries[p]) if p in m.geometries else 'invalid'
                    fp[arm][t] += 1
            for j in fps_b:
                fp['BC'][fp_type(bc_polys[kb.index(j)])] += 1
            full_kept = arms['FULL'][2]
            for p in arms['F'][1]:
                if p in full_kept:
                    continue
                t = fp_type(m.geometries[p]) if p in m.geometries else 'invalid'
                bc_sees = p in m.geometries and any(m.geometries[p].intersects(q) and
                                                   m.geometries[p].intersection(q).area/(m.geometries[p].area + q.area - m.geometries[p].intersection(q).area) >= .5
                                                   for q in bc_polys)
                removed[(t, 'bclean_also_detects' if bc_sees else 'bclean_not')] += 1
                if t == 'no_gt_overlap' and c == 'clean':
                    review_pool.append(dict(kind='removed', source=s, cand=p, raw=r['raw'][p], p_full=p_full[p], bclean=bc_sees))
            if c == 'clean':
                for p in arms['FULL'][1]:
                    if p in m.geometries and fp_type(m.geometries[p]) == 'no_gt_overlap':
                        review_pool.append(dict(kind='kept', source=s, cand=p, raw=r['raw'][p], p_full=p_full[p],
                                                bclean=any(m.geometries[p].intersects(q) for q in bc_polys)))
            # ---- oracles (GT-based upper bounds) on the B1 mask_040 pool
            all_valid = set(m.geometries)
            M_pool = len(p0.match_pairs(m, all_valid)[0])
            n_invalid = len(m.invalid)
            tpF, gtF, prF = (int(x) for x in official(K['FULL'], r['polygons'], r['packed'])[0]['counts'][:3])
            oracle['pool_perfect'] += [M_pool, gtF, M_pool]                      # pick exactly one right box per GT
            oracle['full_no_fp'] += [tpF, gtF, tpF]                              # FULL with every FP removed
            oracle['full_plus_rescue'] += [M_pool, gtF, prF + max(0, M_pool - tpF)]  # FULL + optimistic rescue, FPs kept
            _ = n_invalid
        cond = dict(arms={k: dict(tp=int(v[0]), gt=int(v[1]), pred=int(v[2]), fp=int(v[2] - v[0]), hmean=H(*v)) for k, v in tot.items()},
                    gt_miss={arm: dict(miss[arm]) for arm in miss},
                    gt_miss_by_bin={arm: {f'{k[0]}|{k[1]}': v for k, v in sorted(miss_bin[arm].items())} for arm in miss_bin},
                    bclean_found_full_missed={(k if isinstance(k, str) else f'{k[0]}|{k[1]}'): v for k, v in bc_only.items()},
                    full_found_bclean_missed_by_bin=dict(b1_only),
                    fp_types={arm: dict(fp[arm]) for arm in fp},
                    fp_removed_by_full={f'{k[0]}|{k[1]}': v for k, v in removed.items()},
                    oracles={k: dict(tp=int(v[0]), gt=int(v[1]), pred=int(v[2]), hmean=H(*v)) for k, v in oracle.items()})
        res['conditions'][c] = cond
        print(c, json.dumps({k: round(v['hmean']*100, 3) for k, v in cond['arms'].items()}),
              json.dumps({k: round(v['hmean']*100, 3) for k, v in cond['oracles'].items()}), flush=True)
    C, L = res['conditions']['clean'], res['conditions']['lr2']
    bc_mean = (C['arms']['BC']['hmean'] + L['arms']['BC']['hmean'])/2*100
    res['gate_targets_recomputed'] = dict(clean_min=C['arms']['BC']['hmean']*100 - .3, mean_min=bc_mean + .5,
                                          note='B-Clean .40/.60 is the best single model on A (devA_result); targets for a single B1-based system')
    res['oracle_vs_gate'] = {k: dict(clean=C['oracles'][k]['hmean']*100, lr2=L['oracles'][k]['hmean']*100,
                                     mean=(C['oracles'][k]['hmean'] + L['oracles'][k]['hmean'])/2*100) for k in C['oracles']}
    # ---- review crops (hash-chosen): 24 removed + 12 kept no-GT FPs on Clean
    OUT.mkdir(parents=True, exist_ok=True)
    rk = lambda x: hashlib.sha256(f'{REVIEW_SEED}{x["kind"]}:{x["source"]}:{x["cand"]}'.encode()).hexdigest()  # noqa: E731
    pick = sorted([x for x in review_pool if x['kind'] == 'removed'], key=rk)[:24] + sorted([x for x in review_pool if x['kind'] == 'kept'], key=rk)[:12]
    # teacher-background set: FULL no-GT FPs where B-Clean .40/.60 has NO intersecting box (what anchoring would suppress)
    tb = [dict(x, kind='kept_bclean_none') for x in review_pool if x['kind'] == 'kept' and not x['bclean']]
    pick += sorted(tb, key=rk)[:24]
    res['teacher_background_set_clean'] = dict(size=len(tb), of_full_no_gt_fp=sum(x['kind'] == 'kept' for x in review_pool))
    res['review'] = dict(pool=dict(Counter(x['kind'] for x in review_pool)), picked=len(pick), rule=f'hash rank, seed {REVIEW_SEED}')
    render(pick, rows, recs, OUT/'review')
    save_json(OUT/'stage0_janek.json', res)
    print(json.dumps(res['oracle_vs_gate'], indent=1), json.dumps(res['gate_targets_recomputed'], indent=1))


def render(pick, rows, recs, folder):
    import numpy as np
    from PIL import Image, ImageDraw
    from shapely.geometry import shape
    folder.mkdir(parents=True, exist_ok=True)
    tiles, sheet = [], []
    for n, x in enumerate(pick, 1):
        row = rows[x['source']]
        with Image.open(CLEAN_IMG/row['clean_relpath']) as im:
            img = im.convert('RGB')
        pts = np.asarray(recs[(x['source'], 'clean')]['polygons'][x['cand']], float).reshape(-1, 2)
        x0, y0 = pts.min(0); x1, y1 = pts.max(0)
        mg = max(30., .8*max(x1 - x0, y1 - y0))
        box = (int(max(0, x0 - mg)), int(max(0, y0 - mg)), int(min(img.width, x1 + mg)), int(min(img.height, y1 + mg)))
        crop = img.crop(box)
        d = ImageDraw.Draw(crop)
        off = np.array(box[:2], float)
        for e in row['gt_clean']['ignored']:
            g = shape(e['geometry'])
            for part in getattr(g, 'geoms', [g]):
                d.polygon([tuple(p) for p in np.asarray(part.exterior.coords) - off], outline=(160, 160, 160), width=1)
        for e in row['gt_clean']['valid']:
            g = shape(e['geometry'])
            for part in getattr(g, 'geoms', [g]):
                d.polygon([tuple(p) for p in np.asarray(part.exterior.coords) - off], outline=(0, 200, 0), width=1)
        d.polygon([tuple(p) for p in pts - off], outline=(255, 80, 0), width=3)
        scale = 300/max(crop.size)
        crop = crop.resize((max(1, int(crop.width*scale)), max(1, int(crop.height*scale))))
        tile = Image.new('RGB', (310, 330), 'white')
        tile.paste(crop, (5, 25))
        ImageDraw.Draw(tile).text((5, 5), f"{n:02d} {x['kind']} raw {x['raw']:.2f} p {x['p_full']:.2f} BC {'y' if x['bclean'] else 'n'}", fill=(0, 0, 0))
        tiles.append(tile)
        sheet.append(dict(n=n, **{k: x[k] for k in ('kind', 'source', 'cand', 'raw', 'p_full', 'bclean')}, label=''))
    for k in range(0, len(tiles), 12):
        page = Image.new('RGB', (310*4, 330*3), 'white')
        for j, t in enumerate(tiles[k:k + 12]):
            page.paste(t, ((j % 4)*310, (j//4)*330))
        page.save(folder/f'sheet_{k//12 + 1}.png')
    (folder/'review_sheet.json').write_text(json.dumps(sheet, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
