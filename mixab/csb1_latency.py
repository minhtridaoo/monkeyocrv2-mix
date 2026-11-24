"""Fair latency of B1 alone vs CS-B1@0.7 on ONE GPU (written for Kaggle T4; also runs on Colab).

Why a new command (instead of `dualagree latency`): the old one timed through Tuan's forward(), which also runs a
SHA-256 of the image file and a full native postprocess that deployment does not need, and it measured B1 only as a
stage inside the CS-B1 run. Here both arms are timed as deployed, on the same images, interleaved:

  B1     decode+resize (MMOCR pipeline) -> B1 forward -> DB postprocess (epsilon_005, min_text_score .5)
  CS-B1  decode+resize ONCE (same config file for both models) -> B1 forward -> B-Clean forward ->
         DB postprocess B1 (epsilon_005, .5) and B-Clean (mask_040, .7) -> box fusion (IoU >= .5, greedy, CPU)

Protocol
  * One visible GPU (CUDA_VISIBLE_DEVICES=0 set by the notebook; Tuan's detector_for requires exactly one device).
  * batch 1, FP32, the historical config (detector_for checks it). cudnn.benchmark stays at its default (False).
  * Memory: B1 is loaded FIRST and run alone on the warm-up images -> peak memory of the B1 deployment. Then B-Clean
    is loaded and CS-B1 runs on the warm-up images -> peak memory of the CS-B1 deployment (both models resident).
  * Timing: both models resident; every timed image is run by both arms, order alternating (B1 first on even
    images, CS-B1 first on odd ones, reversed on the second pass) so drift and caching hit both arms equally.
    torch.cuda.synchronize() before each start and after each arm. Image files are read once before timing
    (OS page cache warm for both arms). The SHA-256 check of every image happens before timing, never inside.
  * Correctness: the live box counts are compared with the T4 dump (Colab) for the same images. A mismatch is
    REPORTED, not hidden: Kaggle and Colab may differ in driver/library builds.
Numbers are only comparable INSIDE one run. Do not compare them with latencies measured on another machine.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from mixab import read_json, require, save_json, sha256
from mixab import dualagree as da
from mixab import merge_audit as ma

SEED = 20261122                      # same fixed subset rule as the old `dualagree latency` (probe_sources)
ARMS = ('B1', 'CSB1')
STAGES = ('pre', 'fwd_b1', 'pp_b1', 'fwd_bc', 'pp_bc', 'fuse')


# ---------------------------------------------------------------- CPU-testable helpers

def select_sources(source_ids, n, warmup, seed=SEED):
    """(warmup_sources, timed_sources): fixed subset of n + warmup, the warm-up part chosen by a second rank."""
    chosen = ma.probe_sources(sorted(source_ids), n + warmup, seed=seed)
    by = sorted(chosen, key=lambda s: hashlib.sha256(f'{seed}:warmup:{s}'.encode('utf-8')).hexdigest())
    return sorted(by[:warmup]), sorted(by[warmup:])


def arm_order(index, pass_index):
    """Alternate which arm runs first; the second pass mirrors the first."""
    first_b1 = (index % 2 == 0) != (pass_index % 2 == 1)
    return ('B1', 'CSB1') if first_b1 else ('CSB1', 'B1')


def fuse_thresholded(polys_l, polys_c, theta=da.LOCKED['theta']):
    """CS-B1 fusion on boxes already filtered by the postprocessor (B1 >= .5, B-Clean >= .7): keep every B1 box,
    add each B-Clean box not paired one-to-one with a B1 box at IoU >= theta. Same result as dualagree.fuse_locked."""
    ov = da.sorted_overlaps(da.iou_matrix(polys_l, polys_c))
    ones_l, ones_c = [1.]*len(polys_l), [1.]*len(polys_c)
    keep_l, keep_c = da.fuse_keep(ones_l, ones_c, ov, .5, .7, da.INF, theta)
    return [polys_l[i] for i in sorted(keep_l)] + [polys_c[j] for j in sorted(keep_c)]


def quantiles(values):
    import numpy as np
    v = np.asarray(values, dtype=float)
    return dict(n=int(v.size), median=float(np.median(v)), mean=float(v.mean()), std=float(v.std(ddof=1)) if v.size > 1 else 0.,
                p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)), min=float(v.min()), max=float(v.max()))


def paired_summary(b1_ms, cs_ms, seed=431, n_boot=2000):
    """Paired comparison over images: ratio of medians, median per-image ratio, mean difference, bootstrap CIs."""
    import numpy as np
    b, c = np.asarray(b1_ms, float), np.asarray(cs_ms, float)
    require(b.shape == c.shape and b.size > 1, 'Need paired timings')
    rng = np.random.default_rng(seed)
    rm, md = [], []
    for _ in range(n_boot):
        j = rng.integers(0, b.size, b.size)
        rm.append(np.median(c[j])/np.median(b[j])); md.append(float((c[j] - b[j]).mean()))
    return dict(pairs=int(b.size), ratio_of_medians=float(np.median(c)/np.median(b)),
                ratio_of_medians_ci95=[float(np.percentile(rm, 2.5)), float(np.percentile(rm, 97.5))],
                median_of_per_image_ratio=float(np.median(c/b)),
                mean_extra_ms=float((c - b).mean()), mean_extra_ms_ci95=[float(np.percentile(md, 2.5)), float(np.percentile(md, 97.5))])


def dump_expectation(dl_entry, dc_entry):
    """Box counts the dump predicts for one image: (B1 boxes >= .5, CS-B1 boxes)."""
    return sum(r >= .5 for r in dl_entry['raw']), len(da.fuse_locked(dl_entry, dc_entry))


# ---------------------------------------------------------------- prepare (CPU, on the GPU machine)

def prepare(a):
    """Write the subset list and materialize native LR2 for exactly these sources (Tuan's native_lr2, SHA-checked)."""
    from mixab.monkey import use_monkey
    use_monkey()
    from PIL import Image
    from rev4.training_data import native_lr2
    rows = ma.load_rows(a.bundle)
    warm, timed = select_sources(rows, a.n, a.warmup)
    roots = dict(TextVQA_clean_root=a.clean_root, prepared_root=a.prepared_root)
    tasks = {t['source_image_id']: t for t in ma.resolve_tasks(ma.load_task_metadata(a.bundle), roots, warm + timed)
             if t['condition'] == 'lr2'}
    require(len(tasks) == len(warm) + len(timed), 'Each selected source needs one LR2 task')
    made = 0
    for source, task in sorted(tasks.items()):
        out = Path(task['path'])
        if not out.is_file():
            clean = Path(a.clean_root)/rows[source]['clean_relpath']
            require(clean.is_file(), f'Missing Clean image {clean}')
            require(sha256(clean) == rows[source]['clean_sha256'], f'Clean bytes differ: {clean}')
            with Image.open(clean) as image:
                lr2 = native_lr2(image)
            lr2.info.clear()
            out.parent.mkdir(parents=True, exist_ok=True)
            lr2.save(out, format='PNG')
            made += 1
        require(sha256(out) == task['image_sha256'], f'LR2 bytes differ from the locked task: {out}')
    subset = dict(seed=SEED, n=a.n, warmup=a.warmup, warmup_sources=warm, timed_sources=timed)
    if a.subset_out:
        save_json(a.subset_out, subset)
    print(json.dumps(dict(lr2_made=made, lr2_verified=len(tasks), warmup=len(warm), timed=len(timed))))


def needed_clean(bundle, n, warmup):
    """[(clean_relpath, sha256)] of the Clean images the latency run needs (used to build the upload)."""
    rows = ma.load_rows(bundle)
    warm, timed = select_sources(rows, n, warmup)
    return [(rows[s]['clean_relpath'], rows[s]['clean_sha256']) for s in sorted(warm + timed)]


# ---------------------------------------------------------------- run (GPU)

def environment(torch):
    def out(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception as e:                               # noqa: BLE001
            return f'unavailable: {e}'
    cpu = ''
    if Path('/proc/cpuinfo').is_file():
        cpu = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                    if line.startswith('model name')), '')
    import importlib.metadata as md
    versions = {}
    for k in ('torch', 'torchvision', 'mmengine', 'mmcv', 'mmdet', 'mmocr', 'numpy', 'opencv-python', 'Pillow', 'shapely'):
        try:
            versions[k] = md.version(k)
        except md.PackageNotFoundError:
            versions[k] = None
    return dict(gpu=torch.cuda.get_device_name(0), visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                device_count_visible=torch.cuda.device_count(),
                nvidia_smi_all_gpus=out(['nvidia-smi', '--query-gpu=index,name,driver_version,memory.total,clocks.max.sm',
                                         '--format=csv,noheader']),
                torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                cudnn_benchmark=torch.backends.cudnn.benchmark, allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                allow_tf32_cudnn=torch.backends.cudnn.allow_tf32, python=platform.python_version(), cpu=cpu,
                cpu_count=os.cpu_count(), versions=versions)


def gpu_state():
    try:
        return subprocess.run(['nvidia-smi', '--id=' + os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0],
                               '--query-gpu=temperature.gpu,clocks.sm,power.draw,memory.used', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:                                   # noqa: BLE001
        return f'unavailable: {e}'


def run(a):
    import numpy as np
    import torch

    from mixab.monkey import use_monkey
    monkey = use_monkey()
    from lowres.inference import unpack
    from rev4.decision_plan import GEOMETRIES, INSTALLED_PP_SHA256
    from rev4.stage_a import detector_for

    out_dir = Path(a.output)
    require(not (out_dir/'latency.json').exists(), f'Refusing to overwrite {out_dir}/latency.json')
    out_dir.mkdir(parents=True, exist_ok=True)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            'Need exactly one visible GPU (set CUDA_VISIBLE_DEVICES=0 on a 2xT4 machine)')
    rows = ma.load_rows(a.bundle)
    warm, timed = select_sources(rows, a.n, a.warmup)
    roots = dict(TextVQA_clean_root=a.clean_root, prepared_root=a.prepared_root)
    tasks = {t['source_image_id']: t for t in ma.resolve_tasks(ma.load_task_metadata(a.bundle), roots, warm + timed)
             if t['condition'] == 'lr2'}
    for t in tasks.values():                                     # SHA check outside timing
        require(sha256(t['path']) == t['image_sha256'], f'LR2 bytes differ: {t["path"]}')
        Path(t['path']).read_bytes()                             # warm the OS page cache for both arms
    manifest_path = Path(a.weights_dir)/'merge_manifest.json'
    manifest = read_json(manifest_path)
    by_alpha = {m['alpha']: m for m in manifest['models']}
    ckpt = {}
    for name, alpha in da.EXPERTS:
        path = Path(a.weights_dir)/Path(by_alpha[alpha]['checkpoint']).name
        require(sha256(path) == by_alpha[alpha]['checkpoint_sha256'], f'{path} differs from merge manifest')
        ckpt[name] = path
    geom = {g[0]: g for g in GEOMETRIES}

    def processor(native, gname, cut):
        _, mask, ratio, width, eps = geom[gname]
        p = copy.copy(native)
        p.mask_thr, p.unclip_ratio, p.min_text_width, p.epsilon_ratio, p.min_text_score = mask, ratio, width, eps, cut
        return p

    def mib(x):
        return x/2**20
    memory = {}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    det_b1, nat_b1, _, _ = detector_for(argparse.Namespace(model_config=Path(a.model_config), checkpoint=ckpt['B1']))
    pp_b1 = processor(nat_b1, da.GEOM_L, da.LOCKED['t_l'])
    memory['after_load_B1'] = dict(allocated_mib=mib(torch.cuda.memory_allocated()), reserved_mib=mib(torch.cuda.memory_reserved()),
                                   nvidia_smi=gpu_state())
    sync = torch.cuda.synchronize

    def decode(path):
        _, inputs = next(iter(det_b1.preprocess([path], batch_size=1)))
        return inputs

    def fwd(det, inputs):
        with torch.inference_mode():
            processed = det.model.data_preprocessor(inputs, training=False)
            prob = det.model(**processed, mode='tensor')
        return processed['data_samples'][0], prob

    def post(proc, prob, sample, restore):
        return unpack(proc(prob, [copy.deepcopy(sample)])[0], restore)['polygons']

    def arm_b1(task):
        restore = np.asarray(rows[task['source_image_id']]['clean_size'])/np.asarray(task['size'])
        t = [time.perf_counter()]
        inputs = decode(task['path']); t.append(time.perf_counter())
        sample, prob = fwd(det_b1, inputs); sync(); t.append(time.perf_counter())
        polys = post(pp_b1, prob, sample, restore); t.append(time.perf_counter())
        return polys, dict(pre=t[1] - t[0], fwd_b1=t[2] - t[1], pp_b1=t[3] - t[2])

    det_bc = pp_bc = None

    def arm_cs(task):
        restore = np.asarray(rows[task['source_image_id']]['clean_size'])/np.asarray(task['size'])
        t = [time.perf_counter()]
        inputs = decode(task['path']); t.append(time.perf_counter())
        s_l, p_l = fwd(det_b1, inputs); sync(); t.append(time.perf_counter())
        polys_l = post(pp_b1, p_l, s_l, restore); t.append(time.perf_counter())
        s_c, p_c = fwd(det_bc, inputs); sync(); t.append(time.perf_counter())
        polys_c = post(pp_bc, p_c, s_c, restore); t.append(time.perf_counter())
        fused = fuse_thresholded(polys_l, polys_c); t.append(time.perf_counter())
        return (polys_l, polys_c, fused), dict(pre=t[1] - t[0], fwd_b1=t[2] - t[1], pp_b1=t[3] - t[2],
                                               fwd_bc=t[4] - t[3], pp_bc=t[5] - t[4], fuse=t[6] - t[5])

    # memory phase 1: B1 deployment alone
    for s in warm:
        arm_b1(tasks[s])
    sync()
    memory['B1_deployment_peak'] = dict(allocated_mib=mib(torch.cuda.max_memory_allocated()),
                                        reserved_mib=mib(torch.cuda.max_memory_reserved()), nvidia_smi=gpu_state())
    # memory phase 2: load B-Clean, CS-B1 deployment (both resident)
    torch.cuda.reset_peak_memory_stats()
    det_bc, nat_bc, _, _ = detector_for(argparse.Namespace(model_config=Path(a.model_config), checkpoint=ckpt['B-Clean']))
    pp_bc = processor(nat_bc, da.GEOM_C, da.LOCKED['t_c'])
    memory['after_load_both'] = dict(allocated_mib=mib(torch.cuda.memory_allocated()), reserved_mib=mib(torch.cuda.memory_reserved()))
    shared_ok = []
    for s in warm:
        (pl, pc, fused), _ = arm_cs(tasks[s])
        # sanity (untimed): sharing the decoded input gives the same B-Clean boxes as B-Clean's own preprocess
        _, inputs_c = next(iter(det_bc.preprocess([tasks[s]['path']], batch_size=1)))
        sc, prc = fwd(det_bc, inputs_c)
        restore = np.asarray(rows[s]['clean_size'])/np.asarray(tasks[s]['size'])
        shared_ok.append(post(pp_bc, prc, sc, restore) == pc)
    sync()
    memory['CSB1_deployment_peak'] = dict(allocated_mib=mib(torch.cuda.max_memory_allocated()),
                                          reserved_mib=mib(torch.cuda.max_memory_reserved()), nvidia_smi=gpu_state())
    require(all(shared_ok), 'Shared decoded input changes B-Clean output: the shared-preprocess assumption is wrong')
    # timing phase
    samples, live = [], {}
    state_before = gpu_state()
    tic = time.perf_counter()
    for pass_index in range(a.passes):
        for i, s in enumerate(timed):
            task = tasks[s]
            for arm in arm_order(i, pass_index):
                sync()
                t0 = time.perf_counter()
                res, stages = arm_b1(task) if arm == 'B1' else arm_cs(task)
                sync()
                total = time.perf_counter() - t0
                samples.append(dict(pass_index=pass_index, source=s, arm=arm, total_ms=total*1000,
                                    **{k: v*1000 for k, v in stages.items()}))
                if pass_index == 0:
                    if arm == 'B1':
                        live.setdefault(s, {})['b1'] = res
                    else:
                        live.setdefault(s, {})['cs'] = res
        print(f'pass {pass_index + 1}/{a.passes} done ({time.perf_counter() - tic:.0f} s)', flush=True)
    state_after = gpu_state()
    with (out_dir/'samples.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, ['pass_index', 'source', 'arm', 'total_ms', *STAGES], extrasaction='ignore')
        w.writeheader(); w.writerows(samples)
    by_arm = {arm: [x for x in samples if x['arm'] == arm] for arm in ARMS}
    key = lambda x: (x['pass_index'], x['source'])                # noqa: E731
    b1_sorted, cs_sorted = sorted(by_arm['B1'], key=key), sorted(by_arm['CSB1'], key=key)
    require([key(x) for x in b1_sorted] == [key(x) for x in cs_sorted], 'Unpaired samples')
    res = dict(stage='CS-B1@0.7 vs B1 latency (Kaggle-ready) v1', monkey_v2=monkey,
               installed_db_postprocessor_sha256_expected=INSTALLED_PP_SHA256,
               model_config_sha256=sha256(a.model_config), merge_manifest_sha256=sha256(manifest_path),
               checkpoints={k: sha256(v) for k, v in ckpt.items()}, environment=environment(torch),
               protocol=dict(batch=1, precision='fp32', passes=a.passes, timed_images=len(timed), warmup_images=len(warm),
                             subset_seed=SEED, order='alternating per image, mirrored on pass 2', shared_decode_in_csb1=True,
                             sync='torch.cuda.synchronize before start and after each arm/stage',
                             excluded_from_timing=['image SHA-256', 'file read (page cache warmed)', 'model loading']),
               csb1=dict(B1=dict(geometry=da.GEOM_L, min_text_score=.5), BClean=dict(geometry=da.GEOM_C, min_text_score=.7),
                         fusion=dict(iou=da.LOCKED['theta'], rule='keep all B1; add unpaired B-Clean (greedy one-to-one)')),
               ms={arm: dict(total=quantiles([x['total_ms'] for x in by_arm[arm]]),
                             stages={st: quantiles([x[st] for x in by_arm[arm]]) for st in STAGES if st in by_arm[arm][0]})
                   for arm in ARMS},
               paired=paired_summary([x['total_ms'] for x in b1_sorted], [x['total_ms'] for x in cs_sorted]),
               memory=memory, gpu_state=dict(before_timing=state_before, after_timing=state_after),
               shared_decode_check=dict(images=len(shared_ok), identical=sum(shared_ok)),
               note='Comparable only within this run (same machine, same session). Not comparable with other GPUs/hosts.')
    # live boxes vs the T4 dump (Colab) for the same images
    if a.dump:
        ident = read_json(Path(a.dump)/'dump_identity.json')['identity_sha256']
        n_geom = len(GEOMETRIES)
        gl, gc = [g[0] for g in GEOMETRIES].index(da.GEOM_L), [g[0] for g in GEOMETRIES].index(da.GEOM_C)
        dl = da.load_model_dump(Path(a.dump)/'B1', n_geom, ident)
        dc = da.load_model_dump(Path(a.dump)/'B-Clean', n_geom, ident)
        mism, exact_polys = [], 0
        for s, v in live.items():
            sid = tasks[s]['sample_id']
            want_b1, want_cs = dump_expectation(dl[sid][gl], dc[sid][gc])
            got_b1, got_cs = len(v['b1']), len(v['cs'][2])
            if (got_b1, got_cs) != (want_b1, want_cs):
                mism.append(dict(source=s, live=[got_b1, got_cs], dump=[want_b1, want_cs]))
            else:
                ref = [p for p, r in zip(dl[sid][gl]['polygons'], dl[sid][gl]['raw']) if r >= .5]
                exact_polys += all(len(x) == len(y) and max(abs(p - q) for p, q in zip(x, y)) < 1e-6
                                   for x, y in zip(v['b1'], ref))
        res['check_vs_t4_dump'] = dict(images=len(live), count_mismatch=len(mism), b1_polygons_identical=exact_polys,
                                       examples=mism[:10], dump_gpu=read_json(Path(a.dump)/'dump_identity.json').get('gpu'))
    save_json(out_dir/'latency.json', res)
    print(json.dumps(dict(ms={k: v['total'] for k, v in res['ms'].items()}, paired=res['paired'], memory=memory,
                          check=res.get('check_vs_t4_dump')), indent=1))


def main():
    sys.stdout.reconfigure(errors='backslashreplace')
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'run'):
        s = sub.add_parser(name)
        s.add_argument('--bundle', required=True, type=Path)
        s.add_argument('--clean-root', required=True, type=Path)
        s.add_argument('--prepared-root', required=True, type=Path)
        s.add_argument('--n', type=int, default=200)
        s.add_argument('--warmup', type=int, default=20)
    sub.choices['prepare'].add_argument('--subset-out', type=Path)
    r = sub.choices['run']
    for arg in ('--model-config', '--weights-dir', '--output'):
        r.add_argument(arg, required=True, type=Path)
    r.add_argument('--passes', type=int, default=2)
    r.add_argument('--dump', type=Path, help='T4 dump (Colab) for the box-count cross-check')
    a = p.parse_args()
    prepare(a) if a.command == 'prepare' else run(a)


if __name__ == '__main__':
    main()
