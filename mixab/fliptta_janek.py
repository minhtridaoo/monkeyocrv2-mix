"""Flip-TTA on the P-map (B1, one checkpoint) screening pilot. Spec (locked): docs/research/janek/fliptta_spec_20261123_janek.md
Runs on Kaggle (user); local = code + light tests only. NOT yet run on any GPU.

Arm F: one forward -> P0 -> Tuan's post-processing (geometry mask_040, cutoff .60 = B1 `.40/.60`).
Arm T: P0, plus a forward on the horizontally flipped *content region* of the same preprocessed tensor (padding untouched,
       no resampling), flipped back -> P1b; P = (P0 + P1b)/2 inside the content region (padding keeps P0); same post-processing.

Commands
  prepare   materialize native LR2 for the sample (Tuan's native_lr2, SHA-checked)                      CPU
  infer     per task: F and T (alternating order, timed end to end), alignment checks, candidates        GPU
  evaluate  score F and T with CandidateMatcher, paired CIs, per-source CSV, latency, consistency        CPU
"""
import argparse
import csv
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path, PureWindowsPath

from mixab import read_json, require, save_json, sha256 
from mixab import dualagree_janek as da 
from mixab import merge_audit_janek as ma 

SEED_KEY = '20261123:fliptta:'
N_SOURCES = 300
GEOM = 'mask_040'
CUT = .6                 # B1 `.40/.60`
FLOOR = da.FLOOR         # candidates stored from raw >= .20, only CUT is evaluated
BIN_THR = .4             # binarisation for the alignment diagnostic (= mask_thr of the geometry)
WARMUP = 10              # first tasks excluded from latency statistics only
GATE_PP, CLEAN_PP, TINY_PP = .3, -.3, -3.


# ---------------------------------------------------------------- pure, CPU-testable (torch only)

def select_sources(source_ids, n=N_SOURCES):
    """n split-A sources with the smallest sha256(SEED_KEY + id) (no outcome information)."""
    from mixab import reselect_split_janek as rs 
    a = [s for s in source_ids if rs.split_of(s) == 'A']
    return sorted(sorted(a, key=lambda s: hashlib.sha256((SEED_KEY + s).encode('utf-8')).hexdigest())[:n])


def flip_inputs(inputs, hw):
    """Flip only the content region [:h, :w] of a padded input tensor [N, C, H, W]; padding is left as is."""
    h, w = hw
    out = inputs.clone()
    out[..., :h, :w] = inputs[..., :h, :w].flip(-1)
    return out


def unflip_pmap(prob, hw):
    """Flip the content region [:h, :w] of a P-map [N, H, W] back to the original orientation."""
    h, w = hw
    out = prob.clone()
    out[..., :h, :w] = prob[..., :h, :w].flip(-1)
    return out


def average_pmaps(p0, p1b, hw):
    """Arithmetic mean inside the content region; padding keeps P0."""
    h, w = hw
    out = p0.clone()
    out[..., :h, :w] = (p0[..., :h, :w] + p1b[..., :h, :w])/2
    return out


def padding_is_constant(inputs, hw):
    """True iff every element outside the content region [:h, :w] equals one single value."""
    import torch
    h, w = hw
    mask = torch.ones(inputs.shape[-2:], dtype=torch.bool, device=inputs.device)
    mask[:h, :w] = False
    if not bool(mask.any()):
        return True
    pad = inputs[..., mask]
    return bool((pad == pad.reshape(-1)[0]).all())


def mask_iou(a, b, hw, thr=BIN_THR):
    """IoU of the binarised content regions of two P-maps; None if both are empty."""
    h, w = hw
    x, y = a[..., :h, :w] >= thr, b[..., :h, :w] >= thr
    union = int((x | y).sum())
    return None if union == 0 else float((x & y).sum())/union


def alignment_checks(inputs, p0, p1, p1b, hw):
    """The four hard checks (spec section 3) for one image + the diagnostics. Metainfo equality is checked by the caller."""
    import torch
    ok_const = padding_is_constant(inputs, hw)
    ok_twice = bool(torch.equal(flip_inputs(flip_inputs(inputs, hw), hw), inputs))
    ok_shape = tuple(p1b.shape) == tuple(p0.shape) and bool(torch.isfinite(p1b).all())
    h, w = hw
    return dict(pad_constant=ok_const, flip_twice_identity=ok_twice, pmap_shape_finite=ok_shape,
                iou_aligned=mask_iou(p0, p1b, hw), iou_control=mask_iou(p0, p1, hw),
                mean_abs_diff=float((p0[..., :h, :w] - p1b[..., :h, :w]).abs().mean()))


def alignment_gate(checks):
    """Hard gate over all images: (a)-(d) everywhere and median aligned IoU > median control IoU."""
    import numpy as np
    hard = all(c['pad_constant'] and c['flip_twice_identity'] and c['pmap_shape_finite'] and c['same_metainfo'] for c in checks)
    al = [c['iou_aligned'] for c in checks if c['iou_aligned'] is not None and c['iou_control'] is not None]
    ct = [c['iou_control'] for c in checks if c['iou_aligned'] is not None and c['iou_control'] is not None]
    med_a = float(np.median(al)) if al else None
    med_c = float(np.median(ct)) if ct else None
    return dict(images=len(checks), hard_checks_all_true=bool(hard), median_iou_aligned=med_a, median_iou_control=med_c,
                aligned_gt_control=bool(med_a is not None and med_c is not None and med_a > med_c),
                passed=bool(hard and med_a is not None and med_c is not None and med_a > med_c))


def label(alignment_ok, complete, lr2_delta, lr2_ci_low, clean_delta, tiny_delta):
    """Locked outcome label (spec section 6) and the c1-c3 flags."""
    flags = dict(c1_lr2_gain_ge_0p3_and_ci_gt0=bool(lr2_delta >= GATE_PP and lr2_ci_low > 0),
                 c2_clean_ge_minus_0p3=bool(clean_delta >= CLEAN_PP), c3_lr2_tiny_ge_minus_3=bool(tiny_delta >= TINY_PP))
    if not alignment_ok or not complete:
        name = 'INCONCLUSIVE_ALIGNMENT'
    elif lr2_delta <= 0:
        name = 'NEGATIVE'
    elif lr2_ci_low <= 0:
        name = 'INCONCLUSIVE'
    elif all(flags.values()):
        name = 'PASS_SCREEN'
    else:
        name = 'POSITIVE_BELOW_GATE'
    return name, flags


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
    from mixab.monkey_janek import use_monkey
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
            with Image.open(Path(a.clean_root)/rows[t['source_image_id']]['clean_relpath']) as im:
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
    import numpy as np
    import torch
    from mixab.monkey_janek import use_monkey
    monkey = use_monkey()
    from rev4.decision_core import capture_candidates
    from rev4.decision_plan import GEOMETRIES
    from rev4.stage_a import detector_for

    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Need exactly one visible GPU')
    rows, sources, tasks = sample_tasks(a.bundle, a.clean_root, a.prepared_root, a.limit_sources)
    manifest = read_json(Path(a.weights_dir)/'merge_manifest.json')
    b1 = next(m for m in manifest['models'] if m['alpha'] == '1')
    ckpt = Path(a.weights_dir)/PureWindowsPath(b1['checkpoint']).name
    require(sha256(ckpt) == b1['checkpoint_sha256'], f'{ckpt} differs from merge manifest')
    det, native, _, _ = detector_for(argparse.Namespace(model_config=Path(a.model_config), checkpoint=ckpt))
    geometry = {g[0]: g for g in GEOMETRIES}[GEOM]

    def preprocess(path):
        _, inputs = next(iter(det.preprocess([str(path)], batch_size=1)))
        with torch.inference_mode():
            processed = det.model.data_preprocessor(inputs, training=False)
        return processed

    def decode(prob, sample, restore):
        pred, raw, _ = capture_candidates(native, prob, sample, restore, geometry)
        keep = [k for k, r in enumerate(raw) if r >= FLOOR]
        return dict(polygons=[pred['polygons'][k] for k in keep], raw=[raw[k] for k in keep],
                    packed=[pred['scores'][k] for k in keep])

    def run_f(path, restore):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        processed = preprocess(path)
        with torch.inference_mode():
            p0 = det.model(**processed, mode='tensor')
        res = decode(p0, processed['data_samples'][0], restore)
        torch.cuda.synchronize()
        res['ms'] = (time.perf_counter() - t0)*1000
        return res

    def run_t(path, restore):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        processed = preprocess(path)
        sample = processed['data_samples'][0]
        h, w = (int(v) for v in sample.metainfo['img_shape'][:2])
        with torch.inference_mode():
            p0 = det.model(**processed, mode='tensor')
            flipped = dict(processed)
            flipped['inputs'] = flip_inputs(processed['inputs'], (h, w))
            p1 = det.model(**flipped, mode='tensor')
            p1b = unflip_pmap(p1, (h, w))
            p = average_pmaps(p0, p1b, (h, w))
        res = decode(p, sample, restore)
        torch.cuda.synchronize()
        res['ms'] = (time.perf_counter() - t0)*1000
        return res, (processed, p0, p1, p1b, (h, w))

    done = set()
    for p in sorted(out.glob('shard_*.jsonl.gz')):
        for line in gzip.open(p, 'rt', encoding='utf-8'):
            done.add(json.loads(line)['task'])
    todo = [t for t in tasks if t['sample_id'] not in done]
    print(f'{len(tasks)} tasks, {len(done)} done, {len(todo)} to run', flush=True)
    torch.cuda.reset_peak_memory_stats()
    buf, shard_id = [], len(list(out.glob('shard_*.jsonl.gz')))
    tic = time.perf_counter()
    bad = []
    for number, t in enumerate(todo, 1):
        require(sha256(t['path']) == t['image_sha256'], f'Image SHA mismatch {t["path"]}')
        row = rows[t['source_image_id']]
        W, H = t['size']
        restore = np.asarray(row['clean_size'])/np.asarray(t['size'])
        torch.cuda.reset_peak_memory_stats()
        if number % 2:           # alternate the order so neither arm always runs second
            f = run_f(t['path'], restore); tt, tens = run_t(t['path'], restore)
        else:
            tt, tens = run_t(t['path'], restore); f = run_f(t['path'], restore)
        peak = torch.cuda.max_memory_allocated()/2**20
        processed, p0, p1, p1b, hw = tens
        meta2 = preprocess(t['path'])['data_samples'][0].metainfo      # fresh preprocess of the same file
        meta1 = processed['data_samples'][0].metainfo
        chk = alignment_checks(processed['inputs'], p0, p1, p1b, hw)
        chk['same_metainfo'] = bool(tuple(meta1['img_shape'][:2]) == tuple(meta2['img_shape'][:2]) and
                                    tuple(np.asarray(meta1['scale_factor']).tolist()) == tuple(np.asarray(meta2['scale_factor']).tolist()))
        if not (chk['pad_constant'] and chk['flip_twice_identity'] and chk['pmap_shape_finite'] and chk['same_metainfo']):
            bad.append(t['sample_id'])
        buf.append(dict(task=t['sample_id'], source=t['source_image_id'], condition=t['condition'], size=[W, H],
                        clean_size=row['clean_size'], order='F,T' if number % 2 else 'T,F', img_shape=list(hw),
                        F=f, T=tt, check=chk, peak_mib=peak, number=len(done) + number))
        if len(buf) == a.shard or number == len(todo):
            path = out/f'shard_{shard_id:04d}.jsonl.gz'
            with gzip.open(path.with_suffix('.tmp'), 'wt', encoding='utf-8') as fh:
                for r_ in buf:
                    fh.write(json.dumps(r_) + '\n')
            path.with_suffix('.tmp').replace(path)
            shard_id += 1
            buf = []
            el = time.perf_counter() - tic
            print(f'{number}/{len(todo)} tasks · {el/number:.1f} s/task · ETA {(len(todo) - number)*el/number/60:.0f} min', flush=True)
    recs = load_records(out)
    gate = alignment_gate([r['check'] for r in recs])
    summary = dict(monkey_v2=monkey, gpu=torch.cuda.get_device_name(0), tasks=len(tasks), ran=len(todo),
                   seconds_per_task=(time.perf_counter() - tic)/max(len(todo), 1),
                   projected_minutes_full_pilot=(time.perf_counter() - tic)/max(len(todo), 1)*2*N_SOURCES/60,
                   alignment_gate=gate, hard_check_failures=bad[:10])
    save_json(out/'infer_summary.json', summary)
    print(json.dumps(summary, indent=1))
    require(not bad, f'Hard alignment checks failed on {len(bad)} images, e.g. {bad[:3]}')


# ---------------------------------------------------------------- evaluate (CPU)

def load_records(folder):
    recs = []
    for p in sorted(Path(folder).glob('shard_*.jsonl.gz')):
        recs += [json.loads(line) for line in gzip.open(p, 'rt', encoding='utf-8')]
    return recs


def count_row(entry, row, policy):
    from rev4.decision_core import CandidateMatcher
    m = CandidateMatcher(dict(polygons=entry['polygons'], scores=entry['packed']), entry['raw'], row, policy)
    return ma.count_row(m.score(CUT))


def evaluate(a):
    import numpy as np
    from mixab import phase0_janek as p0 
    from mixab import reselect_split_janek as rs 
    rows, policy, lr2, geoms, scores = da._context(a.bundle)
    recs = load_records(a.infer_dir)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    require(not (out/'fliptta_result.json').exists(), f'Refusing to overwrite {out}/fliptta_result.json')
    sources = sorted({r['source'] for r in recs})
    require(all(rs.split_of(s) == 'A' for s in sources), 'Pilot must only contain split-A sources')
    complete = [s for s in sources if sum(r['source'] == s for r in recs) == 2]
    expected = sample_sources_from_bundle(a.bundle)
    is_complete = complete == expected and len(recs) == 2*len(expected)
    gate = alignment_gate([r['check'] for r in recs])
    by = {}
    tic = time.perf_counter()
    for i, r in enumerate(recs):
        if r['source'] in complete:
            by[(r['source'], r['condition'])] = dict(F=count_row(r['F'], rows[r['source']], policy),
                                                    T=count_row(r['T'], rows[r['source']], policy), rec=r)
        if i % 100 == 0:
            print(f'evaluate {i + 1}/{len(recs)} ({time.perf_counter() - tic:.0f} s)', flush=True)
    res = dict(label='Flip-TTA on P-map: screening pilot on a fixed 300-source sample of split A (not a confirmation); '
                     'T4 Kaggle -> PENDING_REFERENCE_CHECK.', spec='docs/research/janek/fliptta_spec_20261123_janek.md',
               rule=dict(model='B1 alpha_1p00 FP32 batch 1', postprocess=f'{GEOM}/score_{CUT:.2f} (= B1 .40/.60)',
                         tta='mean of P0 and unflipped P(flip content), padding keeps P0'),
               sources=len(complete), complete=bool(is_complete), alignment_gate=gate, conditions={})
    per_rows = []
    for cond in ('lr2', 'clean'):
        ok = [s for s in complete if (s, cond) in by]
        f = np.array([by[(s, cond)]['F'] for s in ok])
        t = np.array([by[(s, cond)]['T'] for s in ok])
        tf, tt = f.sum(0), t.sum(0)

        def bins(x):
            return {b: (int(x[4 + 2*k])/int(x[3 + 2*k]) if x[3 + 2*k] else None) for k, b in enumerate(('<4', '4-8', '8-16', '>=16'))}
        d_tp, d_pred = t[:, 0] - f[:, 0], t[:, 2] - f[:, 2]
        net = p0.net_contributions(d_tp, d_pred, p0.hmean(tf))
        order = sorted(range(len(ok)), key=lambda k: net[k])
        ms = [by[(s, cond)]['rec'] for s in ok]
        timed = [r for r in ms if r['number'] > WARMUP]
        res['conditions'][cond] = dict(
            sources=len(ok), F=p0.totals(tf), T=p0.totals(tt), delta=p0.paired_boot(t, f),
            delta_tiny_pp=(p0.totals(tt)['tiny_recall'] - p0.totals(tf)['tiny_recall'])*100,
            recall=dict(F=int(tf[0])/int(tf[1]), T=int(tt[0])/int(tt[1])),
            precision=dict(F=int(tf[0])/int(tf[2]), T=int(tt[0])/int(tt[2])),
            recall_by_bin=dict(F=bins(tf), T=bins(tt)),
            per_source=dict(up=int((np.array(net) > 0).sum()), down=int((np.array(net) < 0).sum()),
                            same=int((np.array(net) == 0).sum()),
                            worst=[(ok[k], float(net[k])) for k in order[:3]], best=[(ok[k], float(net[k])) for k in order[-3:]]),
            latency_ms=dict(F_median=float(np.median([r['F']['ms'] for r in timed])) if timed else None,
                            T_median=float(np.median([r['T']['ms'] for r in timed])) if timed else None,
                            ratio_median_of_per_image=float(np.median([r['T']['ms']/r['F']['ms'] for r in timed])) if timed else None,
                            timed_images=len(timed), warmup_excluded=WARMUP,
                            note='End to end per arm: preprocess + forward(s) + flip/average + instrumented post-processing; '
                                 'T includes its own P0 forward; alternating order; T4, batch 1, FP32.'),
            peak_mib_max=float(max(r['peak_mib'] for r in ms)))
        for s, x, y, n_ in zip(ok, f, t, net):
            per_rows.append([s, cond, int(x[0]), int(x[1]), int(x[2]), int(y[0]), int(y[2]), int(y[0] - x[0]), int(y[2] - x[2]), round(float(n_), 6)])
    c = res['conditions']
    name, flags = label(gate['passed'], is_complete, c['lr2']['delta']['delta_pp'], c['lr2']['delta']['ci95_pp'][0],
                        c['clean']['delta']['delta_pp'], c['lr2']['delta_tiny_pp'])
    res.update(flags=flags, outcome=name, group_gate='not evaluated at this pilot')
    for cond in ('lr2', 'clean'):
        c[cond]['system_note'] = 'Sample of 300 sources; not comparable with full-set numbers.'
    c['system_mean_hmean_F'] = (c['lr2']['F']['hmean'] + c['clean']['F']['hmean'])/2
    c['system_mean_hmean_T'] = (c['lr2']['T']['hmean'] + c['clean']['T']['hmean'])/2
    cons = {}
    if a.dump and Path(a.dump).is_dir():
        ident = read_json(Path(a.dump)/'dump_identity.json')['identity_sha256']
        dl = da.load_model_dump(Path(a.dump)/'B1', len(geoms), ident)
        gi = geoms.index(GEOM)
        mism = 0
        for s in complete:
            e = dl['lr2::' + s][gi]
            want = count_row(e, rows[s], policy)
            mism += want != list(by[(s, 'lr2')]['F'])
        cons['F_lr2_vs_colab_t4_dump'] = dict(sources=len(complete), mismatched=int(mism))
    if a.a2r and Path(a.a2r).is_file():
        r = rs.load_counts(a.a2r)
        k = [rs.setting_key(x) for x in r['settings']].index(f'{GEOM}/score_{CUT:.2f}')
        idx = {s: i for i, s in enumerate(r['source_ids'])}
        mism = sum(list(r['counts'][idx[s], 0, k]) != list(by[(s, 'clean')]['F']) for s in complete)
        cons['F_clean_vs_a2r_a100'] = dict(sources=len(complete), mismatched=int(mism))
    res['consistency'] = cons
    zoom_path = Path(a.zoom_sample) if a.zoom_sample else None
    if zoom_path and zoom_path.is_file():
        zs = set(read_json(zoom_path)['sources'])
        res['overlap_with_zoom_sample'] = len(zs & set(complete))
    save_json(out/'fliptta_result.json', res)
    with (out/'per_source.csv').open('w', encoding='utf-8', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['source', 'condition', 'tp_F', 'gt', 'pred_F', 'tp_T', 'pred_T', 'd_tp', 'd_pred', 'net_contribution'])
        w.writerows(per_rows)
    print(json.dumps(dict(outcome=name, flags=flags, alignment=gate,
                          lr2=dict(F=c['lr2']['F'], T=c['lr2']['T'], delta=c['lr2']['delta'], tiny=c['lr2']['delta_tiny_pp'],
                                   latency=c['lr2']['latency_ms']),
                          clean=dict(F=c['clean']['F'], T=c['clean']['T'], delta=c['clean']['delta']), consistency=cons),
                     indent=1, default=str))


def sample_sources_from_bundle(bundle):
    return select_sources(ma.load_rows(bundle))


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
    for arg in ('--model-config', '--weights-dir'):
        i.add_argument(arg, required=True, type=Path)
    i.add_argument('--limit-sources', type=int, default=0)
    i.add_argument('--shard', type=int, default=20)
    e = sub.add_parser('evaluate')
    e.add_argument('--bundle', required=True, type=Path)
    e.add_argument('--infer-dir', required=True, type=Path)
    e.add_argument('--output', required=True, type=Path)
    e.add_argument('--dump', type=Path)
    e.add_argument('--a2r', type=Path, help='B1_source_counts.npz (A2r, A100)')
    e.add_argument('--zoom-sample', type=Path, help='sample.json of the zoom pilot (overlap, descriptive)')
    a = ap.parse_args()
    dict(prepare=prepare, infer=infer, evaluate=evaluate)[a.command](a)


if __name__ == '__main__':
    main()
