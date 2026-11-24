"""Design 2 (DualAgree) v1: run B1 AND B-Clean on native-LR2 images, fuse their boxes asymmetrically.

Two parts:
  dump     GPU. For B-Clean and B1 (endpoint copies from mixab.weight_merge, SHA-checked), every LR2 task,
           every geometry of Tuan's 9-geometry grid: store the polygons and raw/packed scores that Tuan's
           own `capture_candidates` returns (raw >= FLOOR = lowest grid score). Batch 1, FP32, same
           `detector_for`/`forward` as merge_audit. Resumable shards; never overwrites a finished dump.
  analyze  CPU. (1) Fidelity: recompute each model's LR2 counts at all 72 settings from the dump with
           Tuan's CandidateMatcher and compare with the A2r NPZ (exact on the same GPU model).
           (2) Fuse and score with the SAME CandidateMatcher (built once per image on the union pool).
           (3) Choose fusion parameters on split A only, lock, report B as exploratory.

Fusion rule v1 (geometry per expert fixed to the one locked on A in p0_fixed: B1 epsilon_005, B-Clean mask_040):
  pairs   = greedy one-to-one B1/B-Clean candidate pairs by IoU (desc) >= theta, among candidates with
            B1 raw >= min(t_l, t_a) and B-Clean raw >= min(t_c, t_a)
  pair    -> emit B1 polygon if (raw_l >= t_a and raw_c >= t_a) or raw_l >= t_l;
             else emit B-Clean polygon if raw_c >= t_c; else nothing   (at most one box per pair)
  B1 only -> emit if raw_l >= t_l;   B-Clean only -> emit if raw_c >= t_c
  t = inf disables a branch, so (t_c = t_a = inf) is exactly "B1 alone at cutoff t_l".
Difference from the design doc: agreed pairs keep the B1 polygon (no WBF coordinate averaging) in v1.
Clean images are not touched: the system uses B-Clean on Clean (A2r counts).
"""
import argparse
import hashlib
import json
import sys
import time
from fractions import Fraction
from pathlib import Path, PureWindowsPath

from mixab import read_json, require, save_json, sha256
from mixab import merge_audit as ma

STAGE = 'DualAgree dump v1'
EXPERTS = (('B-Clean', '0'), ('B1', '1'))
FLOOR = 0.20                       # lowest score of Tuan's grid (rev4.decision_plan.SCORES)
INF = float('inf')
GEOM_L, GEOM_C = 'epsilon_005', 'mask_040'     # locked on A (results/diag_v2_cpu/p0_fixed.json, oracle_cond)
THETAS = (.3, .5, .7)


# ---------------------------------------------------------------- dump shards (CPU-testable)

def write_shard(path, identity_sha, sample_ids, records):
    """records: list of (sample_id, geometry_index, polygons, raw_scores, packed_scores)."""
    import numpy as np

    index = {s: i for i, s in enumerate(sample_ids)}
    task, geom, raw, packed, start, length, coords = [], [], [], [], [], [], []
    for sid, gi, polygons, raws, packs in records:
        require(len(polygons) == len(raws) == len(packs), 'polygon/score lengths differ')
        for poly, r, p in zip(polygons, raws, packs):
            task.append(index[sid]); geom.append(gi); raw.append(float(r)); packed.append(float(p))
            start.append(len(coords)); length.append(len(poly)); coords.extend(float(v) for v in poly)
    temp = Path(path).with_name(Path(path).name + '.tmp.npz')
    np.savez_compressed(temp, identity_sha256=np.asarray(identity_sha), sample_ids=np.asarray(sample_ids),
                        task=np.asarray(task, dtype=np.int32), geom=np.asarray(geom, dtype=np.int16),
                        raw=np.asarray(raw, dtype=np.float64), packed=np.asarray(packed, dtype=np.float64),
                        start=np.asarray(start, dtype=np.int64), length=np.asarray(length, dtype=np.int32),
                        coords=np.asarray(coords, dtype=np.float64))
    temp.replace(path)


def shard_identity(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as d:
        return str(d['identity_sha256'])


def read_shard(path, n_geom):
    """-> {sample_id: [dict(polygons, raw, packed) for each geometry]} (raw as Python floats)."""
    import numpy as np
    with np.load(path, allow_pickle=False) as d:
        out = {str(s): [dict(polygons=[], raw=[], packed=[]) for _ in range(n_geom)] for s in d['sample_ids']}
        ids = [str(s) for s in d['sample_ids']]
        coords = d['coords']
        for t, g, r, p, s, n in zip(d['task'], d['geom'], d['raw'], d['packed'], d['start'], d['length']):
            e = out[ids[int(t)]][int(g)]
            e['polygons'].append(coords[int(s):int(s) + int(n)].tolist())
            e['raw'].append(float(r)); e['packed'].append(float(p))
    return out


def load_model_dump(folder, n_geom, identity_sha=None):
    out = {}
    for path in sorted(Path(folder).glob('shard_*.npz')):
        if identity_sha is not None:
            require(shard_identity(path) == identity_sha, f'{path}: shard belongs to another dump identity')
        out.update(read_shard(path, n_geom))
    return out


# ---------------------------------------------------------------- fusion (CPU-testable)

def polygon_of(coords):
    import numpy as np
    from shapely.geometry import Polygon
    pts = np.asarray(coords, dtype=float)
    if pts.size < 6 or pts.size % 2 or not np.isfinite(pts).all():
        return None
    poly = Polygon(pts.reshape(-1, 2))
    if not poly.is_valid:
        poly = poly.buffer(0)
    return None if poly.is_empty or poly.area <= 0 else poly


def iou_matrix(polys_l, polys_c):
    import numpy as np
    a = [polygon_of(p) for p in polys_l]
    b = [polygon_of(p) for p in polys_c]
    out = np.zeros((len(a), len(b)))
    for i, pa in enumerate(a):
        if pa is None:
            continue
        ax0, ay0, ax1, ay1 = pa.bounds
        for j, pb in enumerate(b):
            if pb is None:
                continue
            bx0, by0, bx1, by1 = pb.bounds
            if bx0 > ax1 or bx1 < ax0 or by0 > ay1 or by1 < ay0:
                continue
            inter = pa.intersection(pb).area
            if inter > 0:
                out[i, j] = inter/(pa.area + pb.area - inter)
    return out


def sorted_overlaps(iou, theta_min=THETAS[0]):
    """Candidate pairs with IoU >= theta_min, sorted by IoU desc (ties: lower i, then lower j)."""
    return sorted((-float(iou[i, j]), i, j) for i in range(iou.shape[0]) for j in range(iou.shape[1])
                  if iou[i, j] >= theta_min)


def greedy_pairs(overlaps, ok_l, ok_c, theta):
    """One-to-one pairs from sorted_overlaps with IoU >= theta among allowed rows/cols."""
    used_l, used_c, pairs = set(), set(), []
    for neg, i, j in overlaps:
        if -neg < theta:
            break
        if ok_l[i] and ok_c[j] and i not in used_l and j not in used_c:
            used_l.add(i); used_c.add(j); pairs.append((i, j))
    return pairs


def fuse_keep(raw_l, raw_c, overlaps, t_l, t_c, t_a, theta):
    """Rule v1 (module docstring) -> (kept B1 indices, kept B-Clean indices)."""
    fl, fc = min(t_l, t_a), min(t_c, t_a)
    pairs = greedy_pairs(overlaps, [r >= fl for r in raw_l], [r >= fc for r in raw_c], theta) if fl < INF and fc < INF else []
    keep_l, keep_c = set(), set()
    paired_l, paired_c = set(), set()
    for i, j in pairs:
        paired_l.add(i); paired_c.add(j)
        if (raw_l[i] >= t_a and raw_c[j] >= t_a) or raw_l[i] >= t_l:
            keep_l.add(i)
        elif raw_c[j] >= t_c:
            keep_c.add(j)
    keep_l |= {i for i, r in enumerate(raw_l) if i not in paired_l and r >= t_l}
    keep_c |= {j for j, r in enumerate(raw_c) if j not in paired_c and r >= t_c}
    return keep_l, keep_c


class PoolScorer:
    """Tuan's CandidateMatcher built ONCE on the union pool (B1 candidates, then B-Clean candidates);
    score(kept) reproduces CandidateMatcher.score for an arbitrary kept subset."""

    def __init__(self, entry_l, entry_c, row, policy):
        from rev4.decision_core import CandidateMatcher
        self.n_l = len(entry_l['raw'])
        prediction = dict(polygons=entry_l['polygons'] + entry_c['polygons'],
                          scores=entry_l['packed'] + entry_c['packed'])
        self.matcher = CandidateMatcher(prediction, entry_l['raw'] + entry_c['raw'], row, policy)
        m = self.matcher
        self.groups = [{i for i, h in enumerate(m.heights) if lo <= h < hi}
                       for lo, hi in ((0, 4), (4, 8), (8, 16), (16, float('inf')))]
        self.iou = iou_matrix(entry_l['polygons'], entry_c['polygons'])
        self.overlaps = sorted_overlaps(self.iou)
        self.raw_l, self.raw_c = entry_l['raw'], entry_c['raw']

    def score(self, keep_l, keep_c):
        from lowres.evaluate_shared import maximum_matching
        m = self.matcher
        kept_set = set(keep_l) | {self.n_l + j for j in keep_c}
        kept = [p for p in m.originals if p in kept_set]
        to_local = {p: i for i, p in enumerate(kept)}
        adjacency = [[to_local[p] for p in edge if p in to_local] for edge in m.edges]
        pairs = maximum_matching(adjacency, len(kept))
        gt_indices = {g for g, _ in pairs}
        invalid = sum(p in kept_set for p in m.invalid)
        row = [len(pairs), len(m.ann_ids), len(kept) + invalid]
        for group in self.groups:
            row += [len(group), len(group & gt_indices)]
        return row

    def fused(self, t_l, t_c, t_a, theta):
        return self.score(*fuse_keep(self.raw_l, self.raw_c, self.overlaps, t_l, t_c, t_a, theta))


def grid(scores):
    s = [float(x) for x in scores]
    return [(t_l, t_c, t_a, th) for t_l in s for t_c in s + [INF] for t_a in s + [INF] for th in THETAS
            if not (t_c == INF and t_a == INF and th != THETAS[0])]   # B1-alone rows once


def family(t_l, t_c, t_a):
    if t_c == INF and t_a == INF:
        return 'B1_alone'
    if t_c == INF:
        return 'B1+agree'
    if t_a == INF:
        return 'B1+BConly'
    return 'B1+agree+BConly'


# ---------------------------------------------------------------- GPU worker

def dump(a):
    import numpy as np  # noqa: F401
    import torch

    from mixab.monkey import use_monkey
    monkey = use_monkey()
    from rev4.decision_audit import installed_postprocessor
    from rev4.decision_core import capture_candidates
    from rev4.decision_plan import GEOMETRIES, INSTALLED_PP_SHA256
    from rev4.stage_a import detector_for, forward

    output = Path(a.output)
    require(not (output/'dump_done.json').exists(), f'Refusing to overwrite a finished dump in {output}')
    output.mkdir(parents=True, exist_ok=True)
    pp_path = installed_postprocessor()
    bundle = Path(a.bundle)
    rows = ma.load_rows(bundle)
    metadata = ma.load_task_metadata(bundle)
    sources = ma.probe_sources(sorted(rows), a.probe) if a.probe else sorted(rows)
    roots = dict(TextVQA_clean_root=a.clean_root, prepared_root=a.prepared_root)
    cond = getattr(a, 'condition', 'lr2')
    tasks = [t for t in ma.resolve_tasks(metadata, roots, sources) if t['condition'] == cond]
    require(len(tasks) == len(sources), f'Each source needs exactly one {cond} task')
    manifest_path = Path(a.weights_dir)/'merge_manifest.json'
    manifest = read_json(manifest_path)
    by_alpha = {m['alpha']: m for m in manifest['models']}
    identity = dict(stage=STAGE, monkey_v2=monkey, installed_db_postprocessor_sha256=INSTALLED_PP_SHA256,
                    installed_db_postprocessor_path=str(pp_path), gt_manifest_sha256=ma.GT_MANIFEST_SHA256,
                    bundle_manifest_sha256=sha256(bundle/'bundle_manifest.json'),
                    merge_manifest_sha256=sha256(manifest_path), probe_sources=a.probe or None,
                    sources_sha256=hashlib.sha256('\n'.join(sources).encode('utf-8')).hexdigest(),
                    geometries=[list(g) for g in GEOMETRIES], floor=FLOOR, chunk=a.chunk, condition=cond,
                    model_config_sha256=sha256(a.model_config), gpu=torch.cuda.get_device_name(0),
                    torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                    no_test=True, training=False)
    identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True).encode('utf-8')).hexdigest()
    old = output/'dump_identity.json'
    if old.is_file():
        require(read_json(old)['identity_sha256'] == identity_sha, f'{output} holds a dump with another identity')
    save_json(old, dict(identity, identity_sha256=identity_sha))
    profile = {}
    for name, alpha in EXPERTS:
        model = by_alpha[alpha]
        checkpoint = Path(a.weights_dir)/PureWindowsPath(model['checkpoint']).name
        require(sha256(checkpoint) == model['checkpoint_sha256'], f'{checkpoint} differs from merge manifest')
        folder = output/name
        folder.mkdir(exist_ok=True)
        detector = native = None
        seconds, started = [], time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        shards = [tasks[k:k + a.chunk] for k in range(0, len(tasks), a.chunk)]
        for si, shard in enumerate(shards):
            path = folder/f'shard_{si:04d}.npz'
            if path.is_file():
                require(shard_identity(path) == identity_sha, f'{path}: belongs to another dump identity')
                continue
            if detector is None:
                detector, native, _, _ = detector_for(argparse.Namespace(model_config=Path(a.model_config),
                                                                         checkpoint=checkpoint))
            records = []
            for task in shard:
                tic = time.perf_counter()
                row = rows[task['source_image_id']]
                _, probability, sample, _, restore, _, _ = forward(detector, native, task, row)
                for gi, geometry in enumerate(GEOMETRIES):
                    prediction, raw, _ = capture_candidates(native, probability, sample, restore, geometry)
                    keep = [k for k, r in enumerate(raw) if r >= FLOOR]
                    records.append((task['sample_id'], gi, [prediction['polygons'][k] for k in keep],
                                    [raw[k] for k in keep], [prediction['scores'][k] for k in keep]))
                seconds.append(time.perf_counter() - tic)
            write_shard(path, identity_sha, [t['sample_id'] for t in shard], records)
            print(f'{name}: shard {si + 1}/{len(shards)}, '
                  f'{sum(seconds)/max(len(seconds), 1):.2f} s/task', flush=True)
        profile[name] = dict(tasks_this_session=len(seconds),
                             mean_seconds_per_task=sum(seconds)/max(len(seconds), 1),
                             wall_seconds=time.perf_counter() - started,
                             peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20)
        del detector
        torch.cuda.empty_cache()
    save_json(output/'dump_done.json', dict(identity_sha256=identity_sha, profile=profile, shards_per_model=len(shards)))
    print(json.dumps(profile, indent=1))


# ---------------------------------------------------------------- CPU analysis

def _context(bundle):
    from mixab.monkey import use_monkey
    use_monkey()
    from rev4.decision_plan import GEOMETRIES, SCORES
    rows = ma.load_rows(bundle)
    policy = read_json(Path(bundle)/'provenance/textvqa_evaluation_protocol.json')
    lr2 = {m['sample_id']: m['source_id'] for m in ma.load_task_metadata(bundle) if m['condition'] == 'lr2'}
    return rows, policy, lr2, [g[0] for g in GEOMETRIES], [float(s) for s in SCORES]


def fidelity(dump_dir, bundle, npz_run, identity_sha=None):
    """Recompute counts at all 72 settings from the dump (LR2 or Clean, read from the sample id); compare with the
    A2r NPZ (endpoint copies) at the same condition."""
    rows, policy, lr2, geoms, scores = _context(bundle)   # use_monkey() puts rev4 on sys.path
    meta = {m['sample_id']: (m['source_id'], m['condition']) for m in ma.load_task_metadata(bundle)}
    from rev4.decision_core import CandidateMatcher
    from mixab import reselect_split as rs
    out = {}
    for name, _ in EXPERTS:
        dump_ = load_model_dump(Path(dump_dir)/name, len(geoms), identity_sha)
        ref = rs.load_counts(Path(npz_run)/f'{name}_source_counts.npz')
        index = {s: i for i, s in enumerate(ref['source_ids'])}
        checked = mismatched = 0
        examples = []
        for sid, entries in dump_.items():
            source, condition = meta[sid]
            ci = 0 if condition == 'clean' else 1
            row = rows[source]
            for gi, e in enumerate(entries):
                matcher = CandidateMatcher(dict(polygons=e['polygons'], scores=e['packed']), e['raw'], row, policy)
                for si, s in enumerate(scores):
                    got = ma.count_row(matcher.score(s))
                    want = [int(v) for v in ref['counts'][index[source], ci, gi*len(scores) + si]]
                    checked += 1
                    if got != want:
                        mismatched += 1
                        if len(examples) < 10:
                            examples.append(dict(source=source, setting=f'{geoms[gi]}/{s:.2f}', got=got, npz=want))
        out[name] = dict(images=len(dump_), checked=checked, mismatched=mismatched,
                         status='EXACT' if checked and not mismatched else 'DRIFT', examples=examples)
    return out


def _h(t):
    return Fraction(2*int(t[0]), int(t[1] + t[2])) if (t[1] + t[2]) else Fraction(0)


def analyze(a):
    import numpy as np

    from mixab import reselect_split as rs
    rows, policy, lr2, geoms, scores = _context(a.bundle)
    out_dir = Path(a.output)
    require(not (out_dir/'decision.json').exists(), f'Refusing to overwrite {out_dir}/decision.json')
    out_dir.mkdir(parents=True, exist_ok=True)
    ident = read_json(Path(a.dump)/'dump_identity.json')['identity_sha256']
    fid = fidelity(a.dump, a.bundle, a.npz_run, ident)
    save_json(out_dir/'fidelity.json', fid)
    gl, gc = geoms.index(GEOM_L), geoms.index(GEOM_C)
    dl = load_model_dump(Path(a.dump)/'B1', len(geoms), ident)
    dc = load_model_dump(Path(a.dump)/'B-Clean', len(geoms), ident)
    require(set(dl) == set(dc), 'B1 and B-Clean dumps cover different LR2 tasks')
    sids = sorted(dl, key=lambda s: lr2[s])
    sources = [lr2[s] for s in sids]
    split = np.array([rs.split_of(s) for s in sources])
    pools = [PoolScorer(dl[s][gl], dc[s][gc], rows[lr2[s]], policy) for s in sids]
    combos = grid(scores)
    per = np.zeros((len(combos), len(pools), ma.FIELDS), dtype=np.int64)
    tic = time.perf_counter()
    for k, (t_l, t_c, t_a, th) in enumerate(combos):
        for i, pool in enumerate(pools):
            per[k, i] = pool.fused(t_l, t_c, t_a, th)
        if k % 100 == 0:
            print(f'combo {k + 1}/{len(combos)} ({time.perf_counter() - tic:.0f} s)', flush=True)
    A, B = split == 'A', split == 'B'
    tot = {s: per[:, m].sum(axis=1) for s, m in (('A', A), ('B', B))}
    key = lambda c: dict(t_l=c[0], t_c=None if c[1] == INF else c[1], t_a=None if c[2] == INF else c[2], theta=c[3])  # noqa: E731

    def stats(t):
        tp, gt, pred, tg, tt = int(t[0]), int(t[1]), int(t[2]), int(t[3]), int(t[4])
        return dict(tp=tp, gt=gt, pred=pred, fp=pred - tp, recall=tp/gt if gt else 0., precision=tp/pred if pred else 0.,
                    hmean=float(_h(t)), tiny_recall=tt/tg if tg else None)

    fams = {}
    for k, c in enumerate(combos):
        f = family(*c[:3])
        if f not in fams or _h(tot['A'][k]) > _h(tot['A'][fams[f]]):
            fams[f] = k
    k_base, k_best = fams['B1_alone'], max(fams.values(), key=lambda k: _h(tot['A'][k]))
    # FP at same recall: lowest FP among ALL 72 B1 LR2 settings (A2r counts) whose recall >= the fused recall
    b1_ref = rs.load_counts(Path(a.npz_run)/'B1_source_counts.npz')
    b1_idx = {s: i for i, s in enumerate(b1_ref['source_ids'])}
    b1_rows = np.array([b1_idx[s] for s in sources])

    def fp_at_recall(mask, recall):
        t = b1_ref['counts'][b1_rows[mask], 1].sum(axis=0)          # [72, 11]
        ok = [int(r[2] - r[0]) for r in t if r[1] and r[0]/r[1] >= recall]
        return min(ok) if ok else None
    # system mean: Clean = B-Clean at the A-locked oracle setting, from A2r counts
    p0 = read_json(Path(a.p0_fixed))
    clean_ref = rs.load_counts(Path(a.npz_run)/'B-Clean_source_counts.npz')
    ckey = p0['oracle_cond']['settings']['bclean']
    ck = [rs.setting_key(s) for s in clean_ref['settings']].index(ckey)
    cidx = {s: i for i, s in enumerate(clean_ref['source_ids'])}
    crow = np.array([cidx[s] for s in sources])
    result = dict(label='DualAgree v1. Parameters chosen on A only; B is EXPLORATORY (already seen).',
                  geometry=dict(B1=GEOM_L, BClean=GEOM_C), clean_setting=ckey, grid_size=len(combos),
                  fidelity={n: f['status'] for n, f in fid.items()},
                  verification='VERIFIED_SAME_GPU' if all(f['status'] == 'EXACT' for f in fid.values()) else 'PENDING_REFERENCE_CHECK',
                  families={f: dict(params=key(combos[k]), A=stats(tot['A'][k]), B=stats(tot['B'][k])) for f, k in fams.items()})
    for s, m in (('A', A), ('B', B)):
        clean_t = clean_ref['counts'][crow[m], 0, ck].sum(axis=0)
        best_single = p0['margin'][s]['best_single'] if s == 'A' else p0['singles'][p0['margin']['A']['best_single_model']]['B']['mean']
        fused, base = stats(tot[s][k_best]), stats(tot[s][k_base])
        result[s] = dict(best=fused, b1_alone=base, delta_lr2_pp=(fused['hmean'] - base['hmean'])*100,
                         delta_tiny_pp=None if fused['tiny_recall'] is None else (fused['tiny_recall'] - base['tiny_recall'])*100,
                         b1_min_fp_at_fused_recall=fp_at_recall(m, fused['recall']),
                         clean_hmean=float(_h(clean_t)),
                         system_mean=(float(_h(clean_t)) + fused['hmean'])/2,
                         best_single_mean_ref=best_single,
                         system_minus_best_single_pp=((float(_h(clean_t)) + fused['hmean'])/2 - best_single)*100)
    result['locked_params'] = key(combos[k_best])
    result['b1_alone_params'] = key(combos[k_base])
    rA = result['A']
    fp_ref = rA['b1_min_fp_at_fused_recall']
    result['kill_rules_A'] = dict(
        lr2_gain_ge_0p3=rA['delta_lr2_pp'] >= .3,
        fp_lower_at_same_recall=fp_ref is None or rA['best']['fp'] < fp_ref,   # None: no B1 setting reaches this recall
        tiny_ge_b1_minus_3=rA['delta_tiny_pp'] is not None and rA['delta_tiny_pp'] >= -3)
    result['continue'] = all(result['kill_rules_A'].values())
    result['group_gate_A_mean_plus_0p5'] = rA['system_minus_best_single_pp'] >= .5
    # paired bootstrap on B over sources: fused vs B1-alone (LR2)
    rng = np.random.default_rng(431)
    idx = np.where(B)[0]
    fb, bb = per[k_best, idx], per[k_base, idx]
    diffs = []
    for _ in range(2000):
        j = rng.integers(0, len(idx), len(idx))
        diffs.append(float(_h(fb[j].sum(0)) - _h(bb[j].sum(0)))*100)
    result['boot_B_lr2_fused_minus_b1'] = dict(mean_pp=float(np.mean(diffs)),
                                                ci95_pp=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))])
    np.savez_compressed(out_dir/'sweep_totals.npz', combos=np.asarray([[c[0], c[1], c[2], c[3]] for c in combos]),
                        A=tot['A'], B=tot['B'])
    save_json(out_dir/'decision.json', result)
    print(json.dumps(result, indent=1))


# ---------------------------------------------------------------- latency of the locked variant (GPU)

# Exploratory variant chosen on 22/11 AFTER the T4 sweep (results/dualagree_v1_T4_followup.json):
# B1 (epsilon_005, raw >= .5) + B-Clean-only boxes (mask_040, raw >= .7), duplicates removed at IoU >= .5.
LOCKED = dict(t_l=.5, t_c=.7, t_a=INF, theta=.5)


def fuse_locked(entry_l, entry_c, p=LOCKED):
    """Deployable fusion on one image: returns the output polygons (B1 kept first, then B-Clean)."""
    ov = sorted_overlaps(iou_matrix(entry_l['polygons'], entry_c['polygons']))
    keep_l, keep_c = fuse_keep(entry_l['raw'], entry_c['raw'], ov, p['t_l'], p['t_c'], p['t_a'], p['theta'])
    return [entry_l['polygons'][i] for i in sorted(keep_l)] + [entry_c['polygons'][j] for j in sorted(keep_c)]


def summarize_ms(values):
    import numpy as np
    v = np.asarray(values, dtype=float)
    return dict(n=int(v.size), median=float(np.median(v)), p90=float(np.percentile(v, 90)), mean=float(v.mean()))


def latency(a):
    import torch

    from mixab.monkey import use_monkey
    monkey = use_monkey()
    from rev4.decision_core import capture_candidates
    from rev4.decision_plan import GEOMETRIES
    from rev4.stage_a import detector_for, forward

    output = Path(a.output)
    require(not (output/'latency.json').exists(), f'Refusing to overwrite {output}/latency.json')
    output.mkdir(parents=True, exist_ok=True)
    bundle = Path(a.bundle)
    rows = ma.load_rows(bundle)
    sources = ma.probe_sources(sorted(rows), a.n + a.warmup, seed=20261122)
    roots = dict(TextVQA_clean_root=a.clean_root, prepared_root=a.prepared_root)
    tasks = [t for t in ma.resolve_tasks(ma.load_task_metadata(bundle), roots, sources) if t['condition'] == 'lr2']
    geom = {g[0]: g for g in GEOMETRIES}
    manifest = read_json(Path(a.weights_dir)/'merge_manifest.json')
    by_alpha = {m['alpha']: m for m in manifest['models']}
    models = {}
    torch.cuda.reset_peak_memory_stats()
    for name, alpha in EXPERTS:                       # both checkpoints resident, as in deployment
        ck = Path(a.weights_dir)/PureWindowsPath(by_alpha[alpha]['checkpoint']).name
        require(sha256(ck) == by_alpha[alpha]['checkpoint_sha256'], f'{ck} differs from merge manifest')
        det, nat, _, _ = detector_for(argparse.Namespace(model_config=Path(a.model_config), checkpoint=ck))
        models[name] = (det, nat)

    def run(name, task, row, gname, cut):
        det, nat = models[name]
        _, prob, sample, _, restore, _, _ = forward(det, nat, task, row)
        pred, raw, _ = capture_candidates(nat, prob, sample, restore, geom[gname])
        keep = [k for k, r in enumerate(raw) if r >= cut]
        torch.cuda.synchronize()
        return dict(polygons=[pred['polygons'][k] for k in keep], raw=[raw[k] for k in keep],
                    packed=[pred['scores'][k] for k in keep])
    rec = dict(b1=[], bclean=[], fuse=[], system=[])
    outputs = {}
    for number, task in enumerate(tasks):
        row = rows[task['source_image_id']]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        e_l = run('B1', task, row, GEOM_L, LOCKED['t_l'])
        t1 = time.perf_counter()
        e_c = run('B-Clean', task, row, GEOM_C, LOCKED['t_c'])
        t2 = time.perf_counter()
        fused = fuse_locked(e_l, e_c)
        t3 = time.perf_counter()
        if number < a.warmup:
            continue
        rec['b1'].append((t1 - t0)*1000); rec['bclean'].append((t2 - t1)*1000)
        rec['fuse'].append((t3 - t2)*1000); rec['system'].append((t3 - t0)*1000)
        outputs[task['sample_id']] = dict(b1=len(e_l['raw']), fused=len(fused))
    res = dict(stage='DualAgree locked-variant latency v1', monkey_v2=monkey, gpu=torch.cuda.get_device_name(0),
               torch=torch.__version__, locked={k: (None if v == INF else v) for k, v in LOCKED.items()},
               geometry=dict(B1=GEOM_L, BClean=GEOM_C), timed_tasks=len(rec['system']), warmup=a.warmup,
               note=('Sequential on one GPU, batch 1 FP32, via Tuan forward() (includes image SHA-256 check and one '
                     'native decode) + capture_candidates for the deployed geometry. Same path for both arms.'),
               ms={k: summarize_ms(v) for k, v in rec.items()},
               peak_gpu_mib_both_loaded=torch.cuda.max_memory_allocated()/2**20)
    res['system_over_b1_median'] = res['ms']['system']['median']/res['ms']['b1']['median']
    if a.dump:                                       # same GPU type -> fused box counts must equal the dump's
        ident = read_json(Path(a.dump)/'dump_identity.json')['identity_sha256']
        _, _, lr2, geoms, _ = _context(bundle)
        dl = load_model_dump(Path(a.dump)/'B1', len(geoms), ident)
        dc = load_model_dump(Path(a.dump)/'B-Clean', len(geoms), ident)
        gl, gc = geoms.index(GEOM_L), geoms.index(GEOM_C)
        mism = []
        for sid, o in outputs.items():
            want = len(fuse_locked(dl[sid][gl], dc[sid][gc]))
            if want != o['fused']:
                mism.append(dict(sample=sid, live=o['fused'], dump=want))
        res['check_vs_dump'] = dict(checked=len(outputs), mismatched=len(mism), examples=mism[:10])
    save_json(output/'latency.json', res)
    print(json.dumps(res, indent=1))


def main():
    sys.stdout.reconfigure(errors='backslashreplace')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    d = sub.add_parser('dump')
    d.add_argument('--bundle', required=True, type=Path)
    d.add_argument('--clean-root', required=True, type=Path)
    d.add_argument('--prepared-root', required=True, type=Path)
    d.add_argument('--model-config', required=True, type=Path)
    d.add_argument('--weights-dir', required=True, type=Path)
    d.add_argument('--output', required=True, type=Path)
    d.add_argument('--probe', type=int, default=0, help='Fixed subset of N sources (0 = all 2,262)')
    d.add_argument('--chunk', type=int, default=64, help='Tasks per shard (part of the dump identity)')
    d.add_argument('--condition', choices=('lr2', 'clean'), default='lr2', help='Which task condition to dump')
    f = sub.add_parser('fidelity')
    f.add_argument('--dump', required=True, type=Path)
    f.add_argument('--bundle', required=True, type=Path)
    f.add_argument('--npz-run', required=True, type=Path, help='A2r run dir with <model>_source_counts.npz')
    n = sub.add_parser('analyze')
    n.add_argument('--dump', required=True, type=Path)
    n.add_argument('--bundle', required=True, type=Path)
    n.add_argument('--npz-run', required=True, type=Path)
    n.add_argument('--p0-fixed', required=True, type=Path, help='results/diag_v2_cpu/p0_fixed.json')
    n.add_argument('--output', required=True, type=Path)
    lt = sub.add_parser('latency')
    for arg in ('--bundle', '--clean-root', '--prepared-root', '--model-config', '--weights-dir', '--output'):
        lt.add_argument(arg, required=True, type=Path)
    lt.add_argument('--n', type=int, default=200, help='Timed LR2 images (fixed subset)')
    lt.add_argument('--warmup', type=int, default=10)
    lt.add_argument('--dump', type=Path, help='Same-GPU dump: fused box counts must match it')
    a = parser.parse_args()
    if a.command == 'latency':
        latency(a)
    elif a.command == 'dump':
        dump(a)
    elif a.command == 'fidelity':
        ident = read_json(a.dump/'dump_identity.json')['identity_sha256']
        print(json.dumps(fidelity(a.dump, a.bundle, a.npz_run, ident), indent=1))
    else:
        analyze(a)


if __name__ == '__main__':
    main()
