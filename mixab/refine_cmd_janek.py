"""Commands for learned box refinement (design locked in mixab/refine_janek.py; spec docs/research/janek/refine_spec_20261125_janek.md).

  extract   GPU  B1 forward ONCE per task -> Tuan's capture_candidates (consistency) + box-quality v1 instrumented
                 post-processing (bq.capture_with_features, unchanged) + P-map crops for every pool candidate (raw >= .20).
                 Every 25th task (and all with --consistency) a second bq.run_task must reproduce the record exactly.
  train     CPU  TRAIN fit fold only: v1 kept boxes, one-to-one eligible pairs, clamped targets, MLP, checkpoints.
  select    CPU  TRAIN-calibration only: offset-0 and every checkpoint through the official evaluator, locked rule.
  evaluate  CPU  split A, once, chosen checkpoint: F / v1 / v1+R; counts, CI, gates, rescued/lost GT, movement by class.
  latency   GPU  F vs v1 vs v1+R end-to-end, rotated order, peak memory.
"""
import argparse
import copy
import gzip
import json
import time
from pathlib import Path

import numpy as np

from mixab import read_json, require, save_json
from mixab import refine_geom_janek as rg
from mixab import refine_janek as rf

SPEC = 'docs/research/janek/refine_spec_20261125_janek.md'
CKPT_EPOCHS = tuple(range(rf.TRAIN['ckpt_every'], rf.TRAIN['epochs'] + 1, rf.TRAIN['ckpt_every']))
GATE = dict(mean=79.0019, clean=80.2202, tiny_pp=-3.)      # dev A, T4 dumps: B-Clean .40/.60 mean + .5 ; Clean - .3
CONDS = ('clean', 'lr2')


def _bq():
    from mixab import boxquality_janek as bq
    return bq


def _dv():
    from mixab import boxquality_devA_janek as dv
    return dv


# ---------------------------------------------------------------- coordinates and crops

def net_quad(poly, restore, scale_factor):
    """Final polygon (clean-image coords) -> network-space quad, SAME vertex order (undo restore, redo resize)."""
    return np.asarray(poly, float).reshape(-1, 2)/np.asarray(restore, float)*np.asarray(scale_factor[:2], float)


def crops_for(pmap, polys, restore, scale_factor, raw):
    C = np.zeros((len(polys), rf.GRID_T, rf.GRID_S), np.float16)
    E = np.zeros((len(polys), rf.N_EXTRA), np.float32)
    for k, p in enumerate(polys):
        q = net_quad(p, restore, scale_factor)
        if q.shape == (4, 2):
            C[k] = rf.sample_crop(pmap, q).astype(np.float16)
            E[k] = rf.extras(q, raw[k])
    return C, E


def features_of(C, E):
    return np.concatenate([np.asarray(C, np.float32).reshape(len(C), -1), np.asarray(E, np.float32)], 1)


def frame_check(rec, quads_net):
    """Network quads recomputed from the final polygons must equal the post-processor's own network quads
    (cv2.boxPoints) as point sets, and lie inside the valid (unpadded) input. Returns max deviations in px."""
    h, w = rec['img_shape']
    dev = outside = 0.
    for p, qn in zip(rec['polygons'], quads_net):
        q = net_quad(p, rec['restore'], rec['scale_factor'])
        d = np.abs(q[:, None, :] - np.asarray(qn, float)[None, :, :]).max(2).min(1).max()
        dev = max(dev, float(d))
        outside = max(outside, float(max(0., -q[:, 0].min(), -q[:, 1].min(), q[:, 0].max() - (w + 1), q[:, 1].max() - (h + 1))))
    return dict(max_net_quad_dev_px=dev, max_outside_px=outside)


def _net_quads(native, prob, sample):
    """Network-space quads of the accepted candidates, observed with the same hook pattern as capture_with_features."""
    import cv2
    bq = _bq()
    name, mask, ratio, width, epsilon = bq.GEOM
    processor = copy.copy(native)
    processor.mask_thr, processor.unclip_ratio = mask, ratio
    processor.min_text_width, processor.epsilon_ratio = width, epsilon
    processor.min_text_score = 0.
    unclip_method, score_method = processor._unclip, processor._get_bbox_score
    ev = []

    def obs_score(score_map, polygon):
        s = float(score_method(score_map, polygon))
        ev.append(dict(raw=s, status='scored'))
        return s

    def obs_unclip(polygon):
        res = unclip_method(polygon)
        if len(res):
            rect = cv2.minAreaRect(res.reshape(-1, 2))
            if min(rect[1]) >= width:
                ev[-1].update(status='accepted', quad=cv2.boxPoints(rect))
        return res
    processor._get_bbox_score, processor._unclip = obs_score, obs_unclip
    processor(prob, [copy.deepcopy(sample)])
    return [e for e in ev if e['status'] == 'accepted']


def run_task_crops(det, native, task, check):
    """Same record as bq.run_task (one forward) + P-map crops."""
    import torch
    from rev4.decision_core import capture_candidates
    from rev4.decision_plan import GEOMETRIES
    bq = _bq()
    W, H = task['size']
    clean_w, clean_h = (task['size'] if task['condition'] == 'clean' else task['clean_size'])
    restore = np.asarray([clean_w, clean_h])/np.asarray([W, H])
    torch.cuda.synchronize(); t0 = time.perf_counter()
    processed, prob = bq._forward(det, task['path'])
    torch.cuda.synchronize(); t1 = time.perf_counter()
    sample = processed['data_samples'][0]
    pred2, raw2, feats = bq.capture_with_features(native, prob, sample, restore)
    t2 = time.perf_counter()
    keep = [k for k, r in enumerate(raw2) if r >= bq.FLOOR]
    polys = [pred2['polygons'][k] for k in keep]
    pmap = prob[0].detach().float().cpu().numpy()
    sf = [float(v) for v in sample.metainfo['scale_factor']]
    C, E = crops_for(pmap, polys, restore, sf, [raw2[k] for k in keep])
    t3 = time.perf_counter()
    same, fc = None, None
    if check:
        pred, raw, _ = capture_candidates(native, prob, sample, restore, {g[0]: g for g in GEOMETRIES}[bq.GEOM[0]])
        same = (raw == raw2 and len(pred['polygons']) == len(pred2['polygons']) and
                all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(pred['polygons'], pred2['polygons'])))
        require(same, f'Instrumented post-processing differs from capture_candidates on {task["source"]}/{task["condition"]}')
    rec = dict(source=task['source'], condition=task['condition'], role=task.get('role'), size=[W, H],
               polygons=polys, raw=[raw2[k] for k in keep], packed=[pred2['scores'][k] for k in keep],
               features=[feats[k] for k in keep], scale_factor=sf, restore=restore.tolist(), pmap_shape=list(pmap.shape),
               ms=dict(forward=(t1 - t0)*1000, postprocess_features=(t2 - t1)*1000, crops=(t3 - t2)*1000),
               consistency=same, img_shape=[int(v) for v in sample.metainfo['img_shape'][:2]],
               image_sha256=bq.sha256_file(task['path']))
    if check:
        acc = _net_quads(native, prob, sample)
        require(len(acc) == len(raw2), 'Network-quad observation count differs')
        fc = frame_check(rec, [acc[k]['quad'] for k in keep])
    return rec, C, E, fc


# ---------------------------------------------------------------- shards

def _save_shard(out, sid, recs, Cs, Es):
    start = 0
    for r, C in zip(recs, Cs):
        r['crop_rows'] = [start, start + len(C)]
        start += len(C)
    tmp = out/f'shard_{sid:04d}.tmp.npz'
    np.savez_compressed(tmp, crops=np.concatenate(Cs) if Cs else np.zeros((0, rf.GRID_T, rf.GRID_S), np.float16),
                        extras=np.concatenate(Es) if Es else np.zeros((0, rf.N_EXTRA), np.float32))
    path = out/f'shard_{sid:04d}.jsonl.gz'
    with gzip.open(path.with_suffix('.tmp'), 'wt', encoding='utf-8') as f:
        for r in recs:
            f.write(json.dumps(r) + '\n')
    tmp.replace(out/f'shard_{sid:04d}.npz')
    path.with_suffix('.tmp').replace(path)


def load_shards(folder):
    """[(record, crops (n,T,S) f16, extras (n,4))] in shard order."""
    out = []
    for p in sorted(Path(folder).glob('shard_*.jsonl.gz')):
        z = np.load(p.with_name(p.name.replace('.jsonl.gz', '.npz')))
        C, E = z['crops'], z['extras']
        for line in gzip.open(p, 'rt', encoding='utf-8'):
            r = json.loads(line)
            a, b = r['crop_rows']
            require(b - a == len(r['raw']), f'crop rows mismatch {r["source"]}')
            out.append((r, C[a:b], E[a:b]))
    return out


def _tasks(a):
    bq, dv = _bq(), _dv()
    if a.split == 'train':
        rows = bq.load_rows(a.rows)
        tasks = bq.tasks_for(rows, set(a.roles.split(',')), a.clean_root, a.prepared_root)
        for t in tasks:
            t['clean_size'] = rows[t['source']]['clean_size']
            t['image_sha256'] = t.get('sha256')
        if a.limit:
            keep = set(sorted({t['source'] for t in tasks})[:a.limit])
            tasks = [t for t in tasks if t['source'] in keep]
        return tasks
    rows, A, _ = dv._context(a.bundle)
    src = dv.smoke_sources(A)[:a.limit] if a.limit else A
    return dv.dev_tasks(a.bundle, a.clean_root, a.prepared_root, src)


def extract(a):
    import torch
    bq = _bq()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Need exactly one visible GPU')
    tasks = _tasks(a)
    monkey, det, native = bq._detector(a)
    done = {(r['source'], r['condition']) for r in bq.load_records(out)}
    todo = [t for t in tasks if (t['source'], t['condition']) not in done]
    print(f'{len(tasks)} tasks, {len(done)} done, {len(todo)} to run', flush=True)
    torch.cuda.reset_peak_memory_stats()
    sid = len(list(out.glob('shard_*.jsonl.gz')))
    prev = read_json(out/'extract_summary.json')['checks'] if (out/'extract_summary.json').is_file() else []
    buf, Cs, Es, checks, tic = [], [], [], [], time.perf_counter()
    for n, t in enumerate(todo, 1):
        if t.get('image_sha256'):
            require(bq.sha256_file(t['path']) == t['image_sha256'], f'Image SHA mismatch {t["path"]}')
        chk = a.consistency or n % 25 == 1
        r, C, E, fc = run_task_crops(det, native, t, chk)
        if chk:
            ref = bq.run_task(det, native, t, True)                 # second forward, untouched v1 path
            same = (ref['raw'] == r['raw'] and ref['features'] == r['features'] and ref['packed'] == r['packed'] and
                    all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(ref['polygons'], r['polygons'])))
            require(same, f'Crop extraction changed the B1/v1 record on {t["source"]}/{t["condition"]}')
            require(fc['max_net_quad_dev_px'] < 1e-2, f'Network quad mapping differs ({fc}) on {t["source"]}/{t["condition"]}')
            checks.append(dict(source=t['source'], condition=t['condition'], identical_to_bq_run_task=same, **fc))
        if a.split == 'devA':
            r['sample_id'] = t['sample_id']
        buf.append(r); Cs.append(C); Es.append(E)
        if len(buf) == a.shard or n == len(todo):
            _save_shard(out, sid, buf, Cs, Es)
            sid += 1
            buf, Cs, Es = [], [], []
            el = time.perf_counter() - tic
            print(f'{n}/{len(todo)} · {el/n:.2f} s/task · ETA {(len(todo) - n)*el/n/60:.0f} min', flush=True)
    recs = bq.load_records(out)
    checks = prev + checks
    ms = {k: [r['ms'][k] for r in recs] for k in ('forward', 'postprocess_features', 'crops')}
    summary = dict(monkey_v2=monkey, gpu=torch.cuda.get_device_name(0), split=a.split, tasks=len(tasks), records=len(recs),
                   ran=len(todo), seconds_per_task_wall=(time.perf_counter() - tic)/max(len(todo), 1),
                   peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20, checks=checks, n_checks=len(checks),
                   checks_all_identical=all(c['identical_to_bq_run_task'] for c in checks),
                   consistency_checked=sum(r.get('consistency') is not None for r in recs),
                   consistency_all_true=any(r.get('consistency') is not None for r in recs) and all(r['consistency'] is True for r in recs if r.get('consistency') is not None),
                   max_net_quad_dev_px=max([c['max_net_quad_dev_px'] for c in checks] or [0.]),
                   max_outside_px=max([c['max_outside_px'] for c in checks] or [0.]),
                   candidates=sum(len(r['raw']) for r in recs),
                   ms={k: dict(median=float(np.median(v)), p95=float(np.percentile(v, 95))) for k, v in ms.items()})
    save_json(out/'extract_summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'checks'}, indent=1))


# ---------------------------------------------------------------- shared CPU helpers

def _policy(bundle):
    return read_json(Path(bundle)/'provenance/textvqa_evaluation_protocol.json')


def _v1_kept(model, r):
    dv = _dv()
    return dv.kept_indices(r['raw'], dv.full_scores(model, r['features']), model)['FULL']


def pairs_for(row, polys, scores, policy):
    """One-to-one eligible pairs {kept position: gt index}, classes, GT shapes, IoU table (after the evaluator filter)."""
    from shapely.geometry import shape
    from rev4.paired_eda_core import filtered_geometry
    shapes, _ = filtered_geometry(dict(polygons=polys, scores=scores), row, policy)
    gts = [shape(e['geometry']) for e in row['gt_clean']['valid']]

    def iou(x, y):
        inter = x.intersection(y).area
        return inter/(x.area + y.area - inter) if inter else 0.
    ious = {j: [iou(g, p) if g.intersects(p) else 0. for g in gts] for j, p in shapes.items()}
    covs = {j: [(g.intersection(p).area/g.area) if g.intersects(p) else 0. for g in gts] for j, p in shapes.items()}
    pairs, cls = rf.eligible(ious, covs)
    return pairs, cls, gts, ious


def refine_polys(polys, offsets, counter=None):
    out = []
    for p, d in zip(polys, offsets):
        new, st = rg.apply_offsets(rg.quad(p), d)
        if new is None:
            new = list(p)
        if counter is not None:
            counter[st] = counter.get(st, 0) + 1
        out.append(new)
    return out


def _save_ckpt(path, model, norm, extra=None):
    import torch
    torch.save(dict(state=model.state_dict(), mean=norm['mean'], std=norm['std'], grid=[rf.GRID_T, rf.GRID_S, rf.MARGIN],
                    in_dim=rf.IN_DIM, hidden=rf.HIDDEN, **(extra or {})), path)


def load_ckpt(path):
    import torch
    z = torch.load(path, map_location='cpu', weights_only=False)
    require(list(z['grid']) == [rf.GRID_T, rf.GRID_S, rf.MARGIN] and z['in_dim'] == rf.IN_DIM, f'{path}: design differs')
    m = rf.make_model()
    m.load_state_dict(z['state'])
    m.eval()
    return m, dict(mean=z['mean'], std=z['std'])


# ---------------------------------------------------------------- train (TRAIN fit fold)

def training_set(items, rows, policy, model, fold='fit'):
    X, Y, is_tp, cls_tot = [], [], [], {}
    for r, C, E in items:
        if rows[r['source']]['role'] != fold:
            continue
        K = _v1_kept(model, r)
        polys = [r['polygons'][i] for i in K]
        pairs, cls, gts, ious = pairs_for(rows[r['source']], polys, [r['packed'][i] for i in K], policy)
        for k, v in cls.items():
            cls_tot[k] = cls_tot.get(k, 0) + v
        for j, gi in pairs.items():
            q = rg.quad(polys[j])
            d = None if q is None else rg.target_offsets(q, np.asarray(gts[gi].exterior.coords))
            if d is None or not np.isfinite(d).all():
                cls_tot['target_fallback'] = cls_tot.get('target_fallback', 0) + 1
                continue
            X.append(features_of(C[K[j]:K[j] + 1], E[K[j]:K[j] + 1])[0])
            Y.append(np.clip(d, -rg.BOUND, rg.BOUND))
            is_tp.append(ious[j][gi] >= .5)
    return np.asarray(X, np.float32).reshape(-1, rf.IN_DIM), np.asarray(Y, np.float32).reshape(-1, 4), np.asarray(is_tp), cls_tot


def train(a):
    from mixab.monkey_janek import use_monkey
    use_monkey()
    bq, dv = _bq(), _dv()
    policy, rows, model = _policy(a.bundle), bq.load_rows(a.rows), dv.load_model(a.model, a.fit)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    X, Y, is_tp, cls_tot = training_set(load_shards(a.extract_dir), rows, policy, model)
    require(len(X) > 0 and np.isfinite(X).all() and np.isfinite(Y).all(), 'Empty or non-finite training set')
    print(f'train pairs {len(X)} (TP {int(is_tp.sum())}, non-TP {int((~is_tp).sum())}) · classes {cls_tot}', flush=True)
    tic = time.perf_counter()
    _, _, hist = rf.fit(X, Y, np.ones(len(X)), ckpt_cb=lambda ep, m, n: _save_ckpt(out/f'ckpt_e{ep:02d}.pt', m, n, dict(epoch=ep)))
    save_json(out/'train.json', dict(spec=SPEC, pairs=len(X), tp_pairs=int(is_tp.sum()), classes=cls_tot, history=hist,
                                     seconds=time.perf_counter() - tic, train=rf.TRAIN,
                                     design=dict(grid=[rf.GRID_T, rf.GRID_S], margin=rf.MARGIN, bound=rg.BOUND, hidden=rf.HIDDEN, beta=rf.BETA),
                                     target_clamped_frac=float((np.abs(Y) >= rg.BOUND - 1e-9).any(1).mean())))
    print(f'trained in {time.perf_counter() - tic:.0f} s; final loss {hist[-1]["loss"]:.5f}')


# ---------------------------------------------------------------- arm evaluation (select / evaluate)

def eval_tasks(items, rows, policy, model, ckpts, with_bins=False):
    """ckpts: {name: (mlp, norm)}; arms 'F' (B1 .40/.60) and 'v1' (= offset 0) are always added."""
    from lowres.evaluate_shared import evaluate_image
    from rev4.decision_core import CandidateMatcher
    dv = _dv()
    per = {}
    diag = {n: dict(gained=0, lost=0, status={}, moved={}) for n in ckpts}
    for r, C, E in items:
        s, c = r['source'], r['condition']
        row = rows[s]
        K = _v1_kept(model, r)
        polys = [r['polygons'][i] for i in K]
        scores = [r['packed'][i] for i in K]
        base = evaluate_image(row['gt_clean'], row['roi_clean'], dict(polygons=polys, scores=scores), policy)
        Fk = [i for i, v in enumerate(r['raw']) if v >= dv.CUT_F]
        fo = evaluate_image(row['gt_clean'], row['roi_clean'], dict(polygons=[r['polygons'][i] for i in Fk],
                                                                    scores=[r['packed'][i] for i in Fk]), policy)
        if with_bins:
            m = CandidateMatcher(dict(polygons=r['polygons'], scores=r['packed']), r['raw'], row, policy)
            heights = {str(e['ann_id']): h for e, h in zip(row['gt_clean']['valid'], m.heights.tolist())}
            mk = lambda o: dv.bin_row(*o['counts'][:3], [str(x) for x in o['matched_ann_ids']], heights)  # noqa: E731
        else:
            mk = lambda o: [int(x) for x in o['counts'][:3]]  # noqa: E731
        rowc = per.setdefault(s, {})
        rowc[('F', c)], rowc[('v1', c)] = mk(fo), mk(base)
        if not ckpts:
            continue
        pairs, _, _, ious = pairs_for(row, polys, scores, policy)
        cls_box = ['filtered' if j not in ious else 'no_gt_overlap' if max(ious[j] or [0.]) <= 0 else
                   ('pair_tp' if ious[j][pairs[j]] >= .5 else 'pair_nontp') if j in pairs else 'other_overlap'
                   for j in range(len(polys))]
        X = features_of(C[K], E[K]) if K else np.zeros((0, rf.IN_DIM), np.float32)
        m0 = set(map(str, base['matched_ann_ids']))
        for name, ck in ckpts.items():
            off = rf.predict_offsets(ck[0], X, ck[1]) if len(X) else np.zeros((0, 4))
            newp = refine_polys(polys, off, diag[name]['status'])
            o = evaluate_image(row['gt_clean'], row['roi_clean'], dict(polygons=newp, scores=scores), policy)
            rowc[(name, c)] = mk(o)
            m1 = set(map(str, o['matched_ann_ids']))
            diag[name]['gained'] += len(m1 - m0); diag[name]['lost'] += len(m0 - m1)
            mag = np.abs(np.clip(off, -rg.BOUND, rg.BOUND)).max(1) if len(off) else np.zeros(0)
            for k, cl in enumerate(cls_box):
                d = diag[name]['moved'].setdefault(cl, dict(n=0, over_0_05=0, over_0_10=0, over_0_25=0, sum_abs=0.))
                d['n'] += 1; d['sum_abs'] += float(mag[k])
                d['over_0_05'] += int(mag[k] > .05); d['over_0_10'] += int(mag[k] > .10); d['over_0_25'] += int(mag[k] > .25)
    return per, diag


def _hm(t):
    return 2*t[0]/(t[1] + t[2])*100 if t[1] + t[2] else 0.


def summarize(per, name):
    tot = {c: np.sum([v[(name, c)][:3] for v in per.values()], 0) for c in CONDS}
    return {c: dict(tp=int(t[0]), gt=int(t[1]), pred=int(t[2]), fp=int(t[2] - t[0]), fn=int(t[1] - t[0]), hmean=_hm(t)) for c, t in tot.items()}


def mean_h(per, name):
    s = summarize(per, name)
    return (s['clean']['hmean'] + s['lr2']['hmean'])/2


def paired_ci(per, a_name, b_name, cond=None, n=2000, seed=431):
    src = sorted(per)
    conds = CONDS if cond is None else (cond,)
    A_ = np.array([[per[s][(a_name, c)][:3] for c in conds] for s in src], float)
    B_ = np.array([[per[s][(b_name, c)][:3] for c in conds] for s in src], float)

    def stat(ix):
        x, y = A_[ix].sum(0), B_[ix].sum(0)
        return float(np.mean([_hm(x[k]) - _hm(y[k]) for k in range(len(conds))]))
    rng = np.random.default_rng(seed)
    bs = [stat(rng.integers(0, len(src), len(src))) for _ in range(n)]
    return dict(delta_pp=stat(np.arange(len(src))), ci95_pp=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])


def choose(table):
    """Locked rule: highest mean Hmean among checkpoints in epoch order (ties -> earlier); must beat offset 0 (v1)."""
    best = None
    for ep in CKPT_EPOCHS:
        n = f'e{ep:02d}'
        if best is None or table[n]['mean'] > table[best]['mean']:
            best = n
    return best if table[best]['mean'] > table['v1']['mean'] else None


# ---------------------------------------------------------------- select (TRAIN-calibration)

def select(a):
    from mixab.monkey_janek import use_monkey
    use_monkey()
    bq, dv = _bq(), _dv()
    policy, rows, model = _policy(a.bundle), bq.load_rows(a.rows), dv.load_model(a.model, a.fit)
    items = [x for x in load_shards(a.extract_dir) if rows[x[0]['source']]['role'] == 'calibration']
    ckpts = {f'e{ep:02d}': load_ckpt(Path(a.train_dir)/f'ckpt_e{ep:02d}.pt') for ep in CKPT_EPOCHS}
    per, diag = eval_tasks(items, rows, policy, model, ckpts)
    names = ['F', 'v1'] + list(ckpts)
    table = {n: dict(mean=mean_h(per, n), **{c: v['hmean'] for c, v in summarize(per, n).items()}) for n in names}
    chosen = choose(table)
    res = dict(spec=SPEC, sources=len(per), rule=choose.__doc__, table=table, chosen=chosen,
               chosen_minus_v1=paired_ci(per, chosen, 'v1') if chosen else None, diag=diag,
               counts={n: summarize(per, n) for n in names},
               outcome='SELECTED' if chosen else 'NO_SIGNAL_ON_CALIBRATION')
    save_json(Path(a.output), res)
    for n, v in table.items():
        print(f"{n:4} mean {v['mean']:.3f} · Clean {v['clean']:.3f} · LR2 {v['lr2']:.3f}")
    print('chosen:', chosen, res['chosen_minus_v1'], '·', res['outcome'])


# ---------------------------------------------------------------- evaluate (A, once)

def evaluate(a):
    from mixab.monkey_janek import use_monkey
    use_monkey()
    dv = _dv()
    sel = read_json(a.selection)
    require(sel['chosen'], 'Calibration selected nothing (NO_SIGNAL_ON_CALIBRATION): A is not evaluated (spec)')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    require(not (out/'refine_devA.json').exists(), 'Refusing to overwrite the single A evaluation')
    policy = _policy(a.bundle)
    rows, A, _ = dv._context(a.bundle)
    model = dv.load_model(a.model, a.fit)
    items = load_shards(a.extract_dir)
    require({x[0]['source'] for x in items} <= set(A), 'Non-A record in the A extraction')
    ck = load_ckpt(Path(a.train_dir)/f'ckpt_{sel["chosen"]}.pt')
    per, diag = eval_tasks(items, rows, policy, model, {'v1R': ck}, with_bins=True)
    arms = {}
    for n in ('F', 'v1', 'v1R'):
        arms[n] = {c: dv.totals([per[s][(n, c)] for s in per]) for c in CONDS}
        arms[n]['mean_hmean'] = (arms[n]['clean']['hmean'] + arms[n]['lr2']['hmean'])/2
    diffs = {f'{x}-{y}': dict(mean=paired_ci(per, x, y), clean=paired_ci(per, x, y, 'clean'), lr2=paired_ci(per, x, y, 'lr2'))
             for x, y in (('v1R', 'v1'), ('v1R', 'F'), ('v1', 'F'))}
    ref = read_json(a.devA_ref)
    ref_check = {n: {c: bool([arms[n][c]['tp'], arms[n][c]['pred']] == [ref['arms'][dn][c]['tp'], ref['arms'][dn][c]['pred']])
                     for c in CONDS} for n, dn in (('F', 'F'), ('v1', 'FULL'))} if len(per) == len(A) else 'subset'
    m, cl = float(arms['v1R']['mean_hmean']*100), float(arms['v1R']['clean']['hmean']*100)
    tiny = float((arms['v1R']['lr2']['tiny_recall'] - arms['F']['lr2']['tiny_recall'])*100)
    gates = dict(g1_mean=bool(m >= GATE['mean']), g2_clean=bool(cl >= GATE['clean']), g3_tiny_lr2=bool(tiny >= GATE['tiny_pp']))
    d = diffs['v1R-v1']['mean']
    label = ('GROUP_GATE_PASS' if all(gates.values()) and d['ci95_pp'][0] > 0 else
             'NO_BENEFIT' if d['delta_pp'] <= 0 else 'INCONCLUSIVE' if d['ci95_pp'][0] <= 0 else 'POSITIVE_BELOW_GATE')
    res = dict(spec=SPEC, split='A', sources=len(per), chosen=sel['chosen'], arms=arms, differences=diffs,
               gates=gates, gate_values=dict(mean=m, clean=cl, tiny_lr2_pp_vs_F=tiny, targets=GATE),
               reproduces_devA_result=ref_check, diag=diag['v1R'], outcome=label,
               per_source={s: {f'{k[0]}|{k[1]}': v for k, v in d_.items()} for s, d_ in per.items()})
    save_json(out/'refine_devA.json', res)
    for n in arms:
        print(n, {c: round(arms[n][c]['hmean']*100, 3) for c in CONDS}, round(arms[n]['mean_hmean']*100, 3))
    print(json.dumps(diffs, indent=1), gates, ref_check, '\nNHÃN:', label)


# ---------------------------------------------------------------- latency (GPU)

def latency(a):
    import resource
    import torch
    from mixab.monkey_janek import use_monkey
    use_monkey()
    from lowres.inference import unpack
    from mixab import csb1_latency_janek as cl
    bq, dv = _bq(), _dv()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    model = dv.load_model(a.model, a.fit)
    mlp, norm = load_ckpt(Path(a.train_dir)/f'ckpt_{read_json(a.selection)["chosen"]}.pt')
    rows, A, _ = dv._context(a.bundle)
    warm, timed = dv.latency_sources(A)
    tasks = dv.dev_tasks(a.bundle, a.clean_root, a.prepared_root, warm + timed)
    for t in tasks:
        Path(t['path']).read_bytes()
    by_key = {(t['source'], t['condition']): t for t in tasks}
    monkey, det, native = bq._detector(a)
    pp_f = copy.copy(native)
    _, mask, ratio, width, eps = bq.GEOM
    pp_f.mask_thr, pp_f.unclip_ratio, pp_f.min_text_width, pp_f.epsilon_ratio, pp_f.min_text_score = mask, ratio, width, eps, dv.CUT_F
    sync = torch.cuda.synchronize

    def fwd(path):
        _, inputs = next(iter(det.preprocess([str(path)], batch_size=1)))
        with torch.inference_mode():
            processed = det.model.data_preprocessor(inputs, training=False)
            prob = det.model(**processed, mode='tensor')
        return processed['data_samples'][0], prob

    def restore_of(t):
        return np.asarray(t['clean_size'])/np.asarray(t['size'])

    def arm_f(t):
        sample, prob = fwd(t['path'])
        return unpack(pp_f(prob, [copy.deepcopy(sample)])[0], restore_of(t))['polygons'], 0.

    def v1_core(t):
        sample, prob = fwd(t['path'])
        pred, raw, feats = bq.capture_with_features(native, prob, sample, restore_of(t))
        keep = [k for k, r in enumerate(raw) if r >= bq.FLOOR]
        p = dv.full_scores(model, [feats[k] for k in keep])
        return sample, prob, pred, raw, [keep[j] for j, v in enumerate(p) if v >= model['cut_full']]

    def arm_v1(t):
        _, _, pred, _, K = v1_core(t)
        return [pred['polygons'][k] for k in K], 0.

    def arm_v1r(t):
        sample, prob, pred, raw, K = v1_core(t)
        s0 = time.perf_counter()
        polys = [pred['polygons'][k] for k in K]
        pmap = prob[0].detach().float().cpu().numpy()
        C, E = crops_for(pmap, polys, restore_of(t), [float(v) for v in sample.metainfo['scale_factor']], [raw[k] for k in K])
        off = rf.predict_offsets(mlp, features_of(C, E), norm) if K else np.zeros((0, 4))
        return refine_polys(polys, off), time.perf_counter() - s0

    arms = dict(F=arm_f, v1=arm_v1, v1R=arm_v1r)
    memory = {}
    for name, fn in arms.items():
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        for s_ in warm:
            for c in CONDS:
                fn(by_key[(s_, c)])
        sync()
        memory[name] = dict(peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20, peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    timed_tasks = [by_key[(s_, c)] for s_ in timed for c in CONDS]
    orders = [('F', 'v1', 'v1R'), ('v1R', 'F', 'v1'), ('v1', 'v1R', 'F')]
    samples = []
    for pi in range(a.passes):
        for i, t in enumerate(timed_tasks):
            for name in orders[(i + pi) % 3]:
                sync(); t0 = time.perf_counter()
                _, ref_s = arms[name](t)
                sync()
                samples.append(dict(pass_index=pi, source=t['source'], condition=t['condition'], arm=name,
                                    total_ms=(time.perf_counter() - t0)*1000, refine_ms=ref_s*1000))
        print(f'pass {pi + 1}/{a.passes}', flush=True)
    key = lambda x: (x['pass_index'], x['source'], x['condition'])  # noqa: E731
    by = {n: sorted([x for x in samples if x['arm'] == n], key=key) for n in arms}
    res = dict(spec=SPEC, monkey_v2=monkey, environment=cl.environment(torch), passes=a.passes, timed_images=len(timed_tasks),
               ms={n: dict(total=dv._quant([x['total_ms'] for x in by[n]]), refine=dv._quant([x['refine_ms'] for x in by[n]])) for n in arms},
               v1R_over_v1=dv._paired([x['total_ms'] for x in by['v1']], [x['total_ms'] for x in by['v1R']]),
               v1R_over_F=dv._paired([x['total_ms'] for x in by['F']], [x['total_ms'] for x in by['v1R']]),
               memory=memory, process_max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
               note='Timed: decode -> B1 -> post-process (-> features -> scorer) (-> GPU->CPU P-map, crops, MLP on CPU, refine).')
    with (out/'samples.jsonl').open('w', encoding='utf-8') as f:
        for x in samples:
            f.write(json.dumps(x) + '\n')
    save_json(out/'latency.json', res)
    print(json.dumps({k: res[k] for k in ('ms', 'v1R_over_v1', 'v1R_over_F', 'memory')}, indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('extract')
    e.add_argument('--split', choices=('train', 'devA'), required=True)
    for k in ('--rows', '--roles', '--bundle', '--clean-root', '--prepared-root', '--model-config', '--weights-dir', '--output'):
        e.add_argument(k)
    e.add_argument('--limit', type=int, default=0)
    e.add_argument('--shard', type=int, default=50)
    e.add_argument('--consistency', action='store_true')
    for name in ('train', 'select', 'evaluate'):
        s = sub.add_parser(name)
        for k in ('--bundle', '--rows', '--model', '--fit', '--extract-dir', '--train-dir', '--output', '--selection', '--devA-ref'):
            s.add_argument(k)
    la = sub.add_parser('latency')
    for k in ('--bundle', '--clean-root', '--prepared-root', '--model-config', '--weights-dir', '--model', '--fit', '--train-dir',
              '--selection', '--output'):
        la.add_argument(k)
    la.add_argument('--passes', type=int, default=2)
    a = p.parse_args(argv)
    dict(extract=extract, train=train, select=select, evaluate=evaluate, latency=latency)[a.cmd](a)


if __name__ == '__main__':
    main()
