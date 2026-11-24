"""ZB1 (Zoom-B1) screening pilot: one B1 checkpoint, extra forwards on enlarged image regions. Spec (locked):
docs/zoom_pilot_spec_20261122.md. Runs on Kaggle (user); local = code + light tests only.

Why crops enlarge text: Tuan's test pipeline resizes every input to short side 1024 (Resize keep_ratio,
scale=(4068, 1024)), so whole-image upscaling is a no-op, while a crop with short side s is magnified 1024/s.

Commands
  prepare   materialize native LR2 for the sample (Tuan's native_lr2, SHA-checked)           CPU
  infer     F (full) + 2x2 + 3x3 tiles + S2 (whole image at short side 2048) per task, raw boxes GPU
            --limit-sources K --identity-check : smoke test (1x1 crop must reproduce F exactly)
  evaluate  build arms F/G2/G3/S2/R3 with the locked merge rule, score with CandidateMatcher  CPU
  pagecheck short check of the dense newspaper page, GT and evaluator                         CPU
"""
import argparse
import copy
import gzip
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

from mixab import read_json, require, save_json, sha256
from mixab import dualagree as da
from mixab import merge_audit as ma

SEED_KEY = '20261122:zoom:'
N_SOURCES = 300
GRIDS = (2, 3)
PAD = .15            # tile padding, fraction of the core cell size per axis
EDGE = 1.0           # drop a tile box with a vertex closer than this (task px) to an interior crop edge
SIZE_CAP = 8.        # zoom boxes considered only if short edge (LR2 px) < SIZE_CAP
IOU = .5             # add a zoom box only if max IoU with kept boxes < IOU
CUT = .5             # postprocessor min_text_score (= baseline)
GEOM = 'epsilon_005'
S2_SCALE = (8136, 2048)
R3_K = 3
DENSE_SOURCE = '6e340fcbb1e0a877'


# ---------------------------------------------------------------- pure, CPU-testable

def select_sources(source_ids, n=N_SOURCES):
    """n split-A sources with the smallest sha256(SEED_KEY + id) (no outcome information)."""
    from mixab import reselect_split as rs
    a = [s for s in source_ids if rs.split_of(s) == 'A']
    return sorted(sorted(a, key=lambda s: hashlib.sha256((SEED_KEY + s).encode('utf-8')).hexdigest())[:n])


def grid_tiles(W, H, n, pad=PAD):
    """n x n tiles: core cells partition the image; crops = cores grown by ceil(pad*cell) per axis, clipped."""
    import math
    xs = [round(W*i/n) for i in range(n + 1)]
    ys = [round(H*i/n) for i in range(n + 1)]
    tiles = []
    for r in range(n):
        for c in range(n):
            x0, x1, y0, y1 = xs[c], xs[c + 1], ys[r], ys[r + 1]
            px, py = math.ceil(pad*(x1 - x0)), math.ceil(pad*(y1 - y0))
            tiles.append(dict(index=r*n + c, core=[x0, y0, x1, y1],
                              crop=[max(0, x0 - px), max(0, y0 - py), min(W, x1 + px), min(H, y1 + py)]))
    return tiles


def keep_tile_box(poly_task, tile, W, H, edge=EDGE):
    """Center in the core cell (half-open; last cell closed at the image border) and no vertex within `edge`
    of a crop edge that lies inside the image."""
    import numpy as np
    p = np.asarray(poly_task, float).reshape(-1, 2)
    cx, cy = p.mean(0)
    x0, y0, x1, y1 = tile['core']
    inx = x0 <= cx < x1 or (x1 == W and cx == W)
    iny = y0 <= cy < y1 or (y1 == H and cy == H)
    if not (inx and iny):
        return False
    a0, b0, a1, b1 = tile['crop']
    if a0 > 0 and (p[:, 0] < a0 + edge).any():
        return False
    if b0 > 0 and (p[:, 1] < b0 + edge).any():
        return False
    if a1 < W and (p[:, 0] > a1 - edge).any():
        return False
    if b1 < H and (p[:, 1] > b1 - edge).any():
        return False
    return True


def crop_to_task(poly_crop, crop):
    out = list(map(float, poly_crop))
    for k in range(0, len(out), 2):
        out[k] += crop[0]; out[k + 1] += crop[1]
    return out


def task_to_clean(poly_task, restore):
    return [float(v)*float(restore[k % 2]) for k, v in enumerate(poly_task)]


def merge(full, zoom, clean_h, size_cap=SIZE_CAP, iou=IOU):
    """full: [(poly_clean, score)]; zoom: [(poly_clean, score, order_key)].
    Keep all full boxes; scan zoom boxes by (-score, order_key); add if short edge (LR2 px) < size_cap and
    max IoU with every kept box < iou. Returns (kept [(poly, score)], added indices into zoom)."""
    from mixab import phase0 as p0
    kept = [(p, s) for p, s in full]
    shapes = [da.polygon_of(p) for p, _ in kept]
    added = []
    for k in sorted(range(len(zoom)), key=lambda k: (-zoom[k][1], zoom[k][2])):
        poly, score, _ = zoom[k]
        if p0.short_edge_lr2(poly, clean_h) >= size_cap:
            continue
        g = da.polygon_of(poly)
        if g is None:
            continue
        best = 0.
        for h in shapes:
            if h is not None and h.intersects(g):
                inter = h.intersection(g).area
                best = max(best, inter/(h.area + g.area - inter))
        if best < iou:
            kept.append((poly, score)); shapes.append(g); added.append(k)
    return kept, added


def r3_select(full_task_polys, clean_h, restore, tiles, k=R3_K):
    """Rank the 3x3 tiles by the number of F boxes with short edge (LR2 px) < SIZE_CAP whose center lies in the
    core cell; ties by tile index; return the indices of the top k. Predictions only, no GT."""
    import numpy as np
    from mixab import phase0 as p0
    counts = []
    for t in tiles:
        x0, y0, x1, y1 = t['core']
        n = 0
        for p in full_task_polys:
            if p0.short_edge_lr2(task_to_clean(p, restore), clean_h) >= SIZE_CAP:
                continue
            cx, cy = np.asarray(p, float).reshape(-1, 2).mean(0)
            if x0 <= cx < x1 and y0 <= cy < y1:
                n += 1
        counts.append((-n, t['index']))
    return [i for _, i in sorted(counts)[:k]]


# ---------------------------------------------------------------- tasks

def sample_tasks(bundle, clean_root, prepared_root, limit=0):
    rows = ma.load_rows(bundle)
    sources = select_sources(rows)
    if limit:
        sources = sources[:limit]
    roots = dict(TextVQA_clean_root=clean_root, prepared_root=prepared_root)
    tasks = sorted(ma.resolve_tasks(ma.load_task_metadata(bundle), roots, sources),
                   key=lambda t: (t['source_image_id'], t['condition']))
    require(len(tasks) == 2*len(sources), 'Each sampled source needs one Clean and one LR2 task')
    return rows, sources, tasks


def prepare(a):
    from mixab.monkey import use_monkey
    use_monkey()
    from PIL import Image
    from rev4.training_data import native_lr2
    rows, sources, tasks = sample_tasks(a.bundle, a.clean_root, a.prepared_root)
    made = 0
    for t in tasks:
        if t['condition'] == 'clean':
            require(sha256(t['path']) == t['image_sha256'], f'Clean bytes differ: {t["path"]}')
            continue
        out = Path(t['path'])
        if not out.is_file():
            row = rows[t['source_image_id']]
            with Image.open(Path(a.clean_root)/row['clean_relpath']) as im:
                lr2 = native_lr2(im)
            lr2.info.clear()
            out.parent.mkdir(parents=True, exist_ok=True)
            lr2.save(out, format='PNG')
            made += 1
        require(sha256(out) == t['image_sha256'], f'LR2 bytes differ from the locked task: {out}')
    save_json(Path(a.output)/'sample.json', dict(seed_key=SEED_KEY, n=len(sources), sources=sources))
    print(json.dumps(dict(sources=len(sources), tasks=len(tasks), lr2_made=made)))


# ---------------------------------------------------------------- infer (GPU)

def infer(a):
    import cv2
    import mmcv
    import numpy as np
    import torch
    from mixab.monkey import use_monkey
    monkey = use_monkey()
    from rev4.decision_plan import GEOMETRIES
    from rev4.stage_a import detector_for

    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Need exactly one visible GPU')
    rows, sources, tasks = sample_tasks(a.bundle, a.clean_root, a.prepared_root, a.limit_sources)
    manifest = read_json(Path(a.weights_dir)/'merge_manifest.json')
    b1 = next(m for m in manifest['models'] if m['alpha'] == '1')
    ckpt = Path(a.weights_dir)/Path(b1['checkpoint']).name
    require(sha256(ckpt) == b1['checkpoint_sha256'], f'{ckpt} differs from merge manifest')
    det, native, _, _ = detector_for(argparse.Namespace(model_config=Path(a.model_config), checkpoint=ckpt))
    _, mask, ratio, width, eps = {g[0]: g for g in GEOMETRIES}[GEOM]
    pp = copy.copy(native)
    pp.mask_thr, pp.unclip_ratio, pp.min_text_width, pp.epsilon_ratio, pp.min_text_score = mask, ratio, width, eps, CUT
    resize = [t for t in det.pipeline.transforms if type(t).__name__ == 'Resize']
    require(len(resize) == 1 and tuple(resize[0].scale) == (4068, 1024), 'Unexpected Resize in the inference pipeline')
    resize = resize[0]
    tmp = Path(a.tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    def predict(path, scale=None):
        old = resize.scale
        if scale is not None:
            resize.scale = scale
        try:
            torch.cuda.synchronize(); t0 = time.perf_counter()
            _, inputs = next(iter(det.preprocess([str(path)], batch_size=1)))
            with torch.inference_mode():
                processed = det.model.data_preprocessor(inputs, training=False)
                prob = det.model(**processed, mode='tensor')
            sample = processed['data_samples'][0]
            res = pp(prob, [copy.deepcopy(sample)])[0].pred_instances
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0)*1000
        finally:
            resize.scale = old
        polys = [np.asarray(p, dtype=np.float64).reshape(-1).tolist() for p in res.polygons]
        return dict(polys=polys, scores=res.scores.detach().cpu().tolist(), ms=ms,
                    img_shape=[int(v) for v in sample.metainfo['img_shape'][:2]],
                    scale_factor=[float(v) for v in sample.metainfo['scale_factor']])

    done = set()
    for p in sorted(out.glob('shard_*.jsonl.gz')):
        for line in gzip.open(p, 'rt', encoding='utf-8'):
            done.add(json.loads(line)['task'])
    todo = [t for t in tasks if t['sample_id'] not in done]
    print(f'{len(tasks)} tasks, {len(done)} done, {len(todo)} to run', flush=True)
    torch.cuda.reset_peak_memory_stats()
    buf, shard_id = [], len(list(out.glob('shard_*.jsonl.gz')))
    ident_fail = []
    tic = time.perf_counter()
    for number, t in enumerate(todo, 1):
        require(sha256(t['path']) == t['image_sha256'], f'Image SHA mismatch {t["path"]}')
        row = rows[t['source_image_id']]
        W, H = t['size']
        restore = [row['clean_size'][0]/W, row['clean_size'][1]/H]
        img = mmcv.imread(t['path'], flag='color_ignore_orientation', backend='cv2')
        require(img.shape[1] == W and img.shape[0] == H, 'Decoded size differs from task size')
        rec = dict(task=t['sample_id'], source=t['source_image_id'], condition=t['condition'], size=[W, H],
                   clean_size=row['clean_size'], restore=restore, full=predict(t['path']), tiles={})
        torch.cuda.reset_peak_memory_stats()
        for n in GRIDS:
            items = []
            for tile in grid_tiles(W, H, n):
                x0, y0, x1, y1 = tile['crop']
                cpath = tmp/f'crop_{n}_{tile["index"]}.png'
                require(cv2.imwrite(str(cpath), img[y0:y1, x0:x1]), 'cv2.imwrite failed')
                r = predict(cpath)
                r['polys'] = [crop_to_task(p, tile['crop']) for p in r['polys']]
                items.append(dict(tile, **r))
            rec['tiles'][str(n)] = items
        rec['peak_mib_tiles'] = torch.cuda.max_memory_allocated()/2**20
        torch.cuda.reset_peak_memory_stats()
        try:
            rec['s2'] = predict(t['path'], scale=S2_SCALE)
            rec['s2']['status'] = 'ok'
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            rec['s2'] = dict(status='OOM')
        rec['peak_mib_s2'] = torch.cuda.max_memory_allocated()/2**20
        if a.identity_check:
            cpath = tmp/'crop_1x1.png'
            cv2.imwrite(str(cpath), img)
            r = predict(cpath)
            same = (r['scores'] == rec['full']['scores'] and len(r['polys']) == len(rec['full']['polys']) and
                    all(max(abs(u - v) for u, v in zip(x, y)) < 1e-6 for x, y in zip(r['polys'], rec['full']['polys'])))
            rec['identity_1x1'] = bool(same)
            if not same:
                ident_fail.append(t['sample_id'])
        buf.append(rec)
        if len(buf) == a.shard or number == len(todo):
            path = out/f'shard_{shard_id:04d}.jsonl.gz'
            with gzip.open(path.with_suffix('.tmp'), 'wt', encoding='utf-8') as f:
                for r_ in buf:
                    f.write(json.dumps(r_) + '\n')
            path.with_suffix('.tmp').replace(path)
            shard_id += 1
            buf = []
            el = time.perf_counter() - tic
            print(f'{number}/{len(todo)} tasks · {el/number:.1f} s/task · ETA {(len(todo) - number)*el/number/60:.0f} min', flush=True)
    summary = dict(monkey_v2=monkey, gpu=torch.cuda.get_device_name(0), tasks=len(tasks), ran=len(todo),
                   seconds_per_task=(time.perf_counter() - tic)/max(len(todo), 1),
                   projected_minutes_full_pilot=(time.perf_counter() - tic)/max(len(todo), 1)*2*N_SOURCES/60,
                   identity_check=None if not a.identity_check else dict(checked=len(todo), failed=ident_fail))
    save_json(out/'infer_summary.json', summary)
    print(json.dumps(summary, indent=1))
    if a.identity_check:
        require(not ident_fail, f'1x1 crop does not reproduce the full-image prediction: {ident_fail[:5]}')


# ---------------------------------------------------------------- evaluate (CPU)

def load_records(folder):
    recs = []
    for p in sorted(Path(folder).glob('shard_*.jsonl.gz')):
        recs += [json.loads(line) for line in gzip.open(p, 'rt', encoding='utf-8')]
    return recs


def arm_candidates(rec, arm):
    """Zoom candidates [(poly_clean, score, order_key)] of one arm (before merge), or None if unavailable."""
    W, H = rec['size']
    rs_ = rec['restore']
    out = []
    if arm in ('G2', 'G3', 'R3'):
        n = '2' if arm == 'G2' else '3'
        tiles = rec['tiles'][n]
        allowed = None
        if arm == 'R3':
            allowed = set(r3_select(rec['full']['polys'], rec['clean_size'][1], rs_, tiles))
        for t in tiles:
            if allowed is not None and t['index'] not in allowed:
                continue
            for k, (p, s) in enumerate(zip(t['polys'], t['scores'])):
                if keep_tile_box(p, t, W, H):
                    out.append((task_to_clean(p, rs_), s, (t['index'], k)))
        return out
    if arm == 'S2':
        if rec['s2'].get('status') != 'ok':
            return None
        return [(task_to_clean(p, rs_), s, (0, k)) for k, (p, s) in enumerate(zip(rec['s2']['polys'], rec['s2']['scores']))]
    raise ValueError(arm)


def forwards(arm):
    return dict(F=1, G2=5, G3=10, S2=2, R3=1 + R3_K)[arm]


def arm_ms(rec, arm):
    f = rec['full']['ms']
    if arm == 'F':
        return f
    if arm in ('G2', 'G3'):
        return f + sum(t['ms'] for t in rec['tiles'][arm[1]])
    if arm == 'R3':
        sel = set(r3_select(rec['full']['polys'], rec['clean_size'][1], rec['restore'], rec['tiles']['3']))
        return f + sum(t['ms'] for t in rec['tiles']['3'] if t['index'] in sel)
    return f + rec['s2']['ms'] if rec['s2'].get('status') == 'ok' else None


def score_task(rec, row, policy, arm, size_cap=SIZE_CAP):
    """-> (count row [11], extra) for one task and arm. Pool matcher on F + all zoom candidates of the arm."""
    from rev4.decision_core import CandidateMatcher
    from rev4.decision_review import fp_proxy
    from shapely.geometry import shape
    from shapely.strtree import STRtree
    from mixab import phase0 as p0
    full = [(task_to_clean(p, rec['restore']), s) for p, s in zip(rec['full']['polys'], rec['full']['scores'])]
    zoom = [] if arm == 'F' else arm_candidates(rec, arm)
    if zoom is None:
        return None, None
    kept, added = merge(full, zoom, rec['clean_size'][1], size_cap=size_cap)
    pool = [p for p, _ in full] + [z[0] for z in zoom]
    scores = [s for _, s in full] + [z[1] for z in zoom]
    m = CandidateMatcher(dict(polygons=pool, scores=scores), scores, row, policy)
    groups = [{i for i, h in enumerate(m.heights) if lo <= h < hi} for lo, hi in ((0, 4), (4, 8), (8, 16), (16, float('inf')))]
    kept_idx = set(range(len(full))) | {len(full) + k for k in added}
    pairs, invalid = p0.match_pairs(m, kept_idx)
    counts = p0.row_from_pairs(m, groups, pairs, invalid, kept_idx)
    matched = {g for g, _ in pairs}
    extra = dict(added=len(added))
    if arm != 'F':
        # diagnostics with GT (reported separately): FN of F reachable by any zoom box; proxies of added FPs
        f_pairs, _ = p0.match_pairs(m, set(range(len(full))))
        f_matched = {g for g, _ in f_pairs}
        zoom_ids = set(range(len(full), len(pool)))
        reach = Counter()
        for g, edge in enumerate(m.edges):
            if g not in f_matched and zoom_ids & set(edge):
                reach[p0.size_bin(float(m.heights[g]))] += 1
        entries = row['gt_clean']['valid']
        gshapes = [shape(e['geometry']) for e in entries]
        gindex = ([str(e['ann_id']) for e in entries], gshapes, None, STRtree(gshapes)) if gshapes else None
        matched_ann = {m.ann_ids[g] for g in matched}
        prox = Counter()
        matched_pred = {p for _, p in pairs}
        for k in added:
            p = len(full) + k
            if p in matched_pred or p not in m.geometries:
                continue
            prox[fp_proxy(m.geometries[p], gindex, matched_ann, policy)['proxy'] if gindex else 'no_gt_in_image'] += 1
        extra.update(fn_reach=dict(reach), added_fp_proxy=dict(prox))
    return counts, extra


def evaluate(a):
    import numpy as np
    from mixab import phase0 as p0
    from mixab import reselect_split as rs
    rows, policy, lr2, geoms, scores = da._context(a.bundle)
    recs = load_records(a.infer_dir)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    sources = sorted({r['source'] for r in recs})
    require(all(rs.split_of(s) == 'A' for s in sources), 'Pilot must only contain split-A sources')
    have = Counter(r['source'] for r in recs)
    complete = [s for s in sources if have[s] == 2]
    arms = ['F', 'G2', 'G3', 'S2', 'R3']
    by = {}
    tic = time.perf_counter()
    for i, r in enumerate(recs):
        if r['source'] not in complete:
            continue
        row = rows[r['source']]
        for arm in arms:
            c, e = score_task(r, row, policy, arm)
            by[(r['source'], r['condition'], arm)] = (c, e, arm_ms(r, arm))
            if arm in ('G2', 'G3', 'S2'):
                c2, _ = score_task(r, row, policy, arm, size_cap=float('inf'))
                by[(r['source'], r['condition'], arm + '_nocap')] = (c2, None, None)
        if i % 100 == 0:
            print(f'evaluate {i + 1}/{len(recs)} ({time.perf_counter() - tic:.0f} s)', flush=True)
    res = dict(label='ZB1 screening pilot on a fixed 300-source sample of split A (not a confirmation). T4 Kaggle.',
               spec='docs/zoom_pilot_spec_20261122.md', sources=len(complete), arms={})
    for arm in arms + ['G2_nocap', 'G3_nocap', 'S2_nocap']:
        entry = dict(forwards=forwards(arm.split('_')[0]))
        for cond in ('clean', 'lr2'):
            ok = [s for s in complete if by[(s, cond, arm)][0] is not None]
            if not ok:
                entry[cond] = dict(status='unavailable (OOM)')
                continue
            new = np.array([by[(s, cond, arm)][0] for s in ok])
            base = np.array([by[(s, cond, 'F')][0] for s in ok])
            t = new.sum(0)
            d = dict(sources=len(ok), totals=p0.totals(t),
                     recall_by_bin={b: (int(t[4 + 2*k])/int(t[3 + 2*k]) if t[3 + 2*k] else None)
                                    for k, b in enumerate(('<4', '4-8', '8-16', '>=16'))})
            if arm != 'F':
                d['vs_F'] = p0.paired_boot(new, base)
                d['delta_tiny_pp'] = (p0.totals(t)['tiny_recall'] - p0.totals(base.sum(0))['tiny_recall'])*100
                ex = [by[(s, cond, arm)][1] for s in ok]
                if ex[0] is not None:
                    d['added_boxes'] = sum(x['added'] for x in ex)
                    d['diag_fn_of_F_reachable_by_zoom'] = dict(sum((Counter(x['fn_reach']) for x in ex), Counter()))
                    d['diag_added_fp_proxy'] = dict(sum((Counter(x['added_fp_proxy']) for x in ex), Counter()))
            ms = [by[(s, cond, arm)][2] for s in ok]
            if ms[0] is not None:
                d['ms_per_image'] = dict(median=float(np.median(ms)), mean=float(np.mean(ms)))
            entry[cond] = d
        if 'totals' in entry.get('clean', {}) and 'totals' in entry.get('lr2', {}):
            entry['system_mean_hmean'] = (entry['clean']['totals']['hmean'] + entry['lr2']['totals']['hmean'])/2
        res['arms'][arm] = entry
    g2 = res['arms']['G2']
    res['screening_gate_G2'] = dict(
        lr2_delta_ge_0p3_and_ci_low_gt_0=g2['lr2']['vs_F']['delta_pp'] >= .3 and g2['lr2']['vs_F']['ci95_pp'][0] > 0,
        clean_delta_ge_minus_0p3=g2['clean']['vs_F']['delta_pp'] >= -.3,
        lr2_tiny_ge_minus_3=g2['lr2']['delta_tiny_pp'] >= -3)
    res['screening_pass_G2'] = all(res['screening_gate_G2'].values())
    # consistency: F on LR2 vs the Colab T4 dump; F on Clean vs A2r (A100)
    cons = {}
    if a.dump and Path(a.dump).is_dir():
        ident = read_json(Path(a.dump)/'dump_identity.json')['identity_sha256']
        dl = da.load_model_dump(Path(a.dump)/'B1', len(geoms), ident)
        from rev4.decision_core import CandidateMatcher
        gi = geoms.index(GEOM)
        mism = 0
        for s in complete:
            e = dl['lr2::' + s][gi]
            want = ma.count_row(CandidateMatcher(dict(polygons=e['polygons'], scores=e['packed']), e['raw'], rows[s], policy).score(CUT))
            mism += want != list(by[(s, 'lr2', 'F')][0])
        cons['F_lr2_vs_colab_t4_dump'] = dict(sources=len(complete), mismatched=mism)
    if a.a2r and Path(a.a2r).is_file():
        r = rs.load_counts(a.a2r)
        k = [rs.setting_key(x) for x in r['settings']].index(f'{GEOM}/score_{CUT:.2f}')
        idx = {s: i for i, s in enumerate(r['source_ids'])}
        mism = sum(list(r['counts'][idx[s], 0, k]) != list(by[(s, 'clean', 'F')][0]) for s in complete)
        cons['F_clean_vs_a2r_a100'] = dict(sources=len(complete), mismatched=int(mism))
    res['consistency'] = cons
    save_json(out/'zoom_result.json', res)
    print(json.dumps(dict(gate=res['screening_gate_G2'], passed=res['screening_pass_G2'],
                          table={arm: {c: (e[c]['vs_F'] if 'vs_F' in e.get(c, {}) else e.get(c, {}).get('totals'))
                                       for c in ('clean', 'lr2')} for arm, e in res['arms'].items()},
                          consistency=cons), indent=1, default=str))


# ---------------------------------------------------------------- pagecheck (CPU)

def pagecheck(a):
    import numpy as np
    from shapely.geometry import shape
    rows, policy, lr2, geoms, scores = da._context(a.bundle)   # use_monkey() puts rev4/lowres on sys.path
    from rev4.decision_core import CandidateMatcher
    from lowres.evaluate_shared import evaluate_image
    from mixab import reselect_split as rs
    ident = read_json(Path(a.dump)/'dump_identity.json')['identity_sha256']
    dl = da.load_model_dump(Path(a.dump)/'B1', len(geoms), ident)
    gi = geoms.index(GEOM)
    per = []
    for sid, entries in dl.items():
        s = lr2[sid]
        if rs.split_of(s) != 'A':
            continue
        e = entries[gi]
        c = ma.count_row(CandidateMatcher(dict(polygons=e['polygons'], scores=e['packed']), e['raw'], rows[s], policy).score(CUT))
        per.append((c[2] - c[0], s, c[1], c[2]))
    per.sort(reverse=True)
    fp_total = sum(x[0] for x in per)
    row = rows[DENSE_SOURCE]
    e = dl['lr2::' + DENSE_SOURCE][gi]
    m = CandidateMatcher(dict(polygons=e['polygons'], scores=e['packed']), e['raw'], row, policy)
    fast = m.assert_official(CUT)          # raises if CandidateMatcher differs from the unchanged evaluator
    roi = shape(row['roi_clean'])
    W, H = row['clean_size']
    out = dict(
        A_b1_fp_total=fp_total, A_fp_share_top1=per[0][0]/fp_total, A_fp_share_top10=sum(x[0] for x in per[:10])/fp_total,
        A_top10=[dict(source=s, fp=f, gt=g, pred=p) for f, s, g, p in per[:10]],
        dense_source=dict(source=DENSE_SOURCE, clean_size=[W, H], valid_gt=len(row['gt_clean']['valid']),
                          ignored_gt=len(row['gt_clean']['ignored']), roi_area_fraction=roi.area/(W*H),
                          b1_lr2_counts=fast['counts'], official_evaluator_agrees=True,
                          note='Context only: no result or protocol is changed because of this check.'))
    if a.clean_root:
        from PIL import Image, ImageDraw
        with Image.open(Path(a.clean_root)/row['clean_relpath']) as im:
            img = im.convert('RGB')
        d = ImageDraw.Draw(img)
        for g in row['gt_clean']['valid']:
            d.polygon([tuple(p) for p in np.asarray(shape(g['geometry']).exterior.coords)], outline=(0, 200, 0), width=3)
        for g in row['gt_clean']['ignored']:
            d.polygon([tuple(p) for p in np.asarray(shape(g['geometry']).exterior.coords)], outline=(150, 150, 150), width=3)
        img.save(Path(a.output)/'pagecheck_dense_source.png')
    save_json(Path(a.output)/'pagecheck.json', out)
    print(json.dumps(out, indent=1))


def main():
    sys.stdout.reconfigure(errors='backslashreplace')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    i = sub.add_parser('infer')
    for s in (p, i):
        s.add_argument('--bundle', required=True, type=Path)
        s.add_argument('--clean-root', required=True, type=Path)
        s.add_argument('--prepared-root', required=True, type=Path)
        s.add_argument('--output', required=True, type=Path)
    for arg in ('--model-config', '--weights-dir', '--tmp'):
        i.add_argument(arg, required=True, type=Path)
    i.add_argument('--limit-sources', type=int, default=0)
    i.add_argument('--identity-check', action='store_true')
    i.add_argument('--shard', type=int, default=20)
    e = sub.add_parser('evaluate')
    e.add_argument('--bundle', required=True, type=Path)
    e.add_argument('--infer-dir', required=True, type=Path)
    e.add_argument('--output', required=True, type=Path)
    e.add_argument('--dump', type=Path)
    e.add_argument('--a2r', type=Path, help='B1_source_counts.npz (A2r, A100)')
    c = sub.add_parser('pagecheck')
    c.add_argument('--bundle', required=True, type=Path)
    c.add_argument('--dump', required=True, type=Path)
    c.add_argument('--output', required=True, type=Path)
    c.add_argument('--clean-root', type=Path)
    a = ap.parse_args()
    if a.command in ('evaluate', 'pagecheck'):
        a.output.mkdir(parents=True, exist_ok=True)
    dict(prepare=prepare, infer=infer, evaluate=evaluate, pagecheck=pagecheck)[a.command](a)


if __name__ == '__main__':
    main()
