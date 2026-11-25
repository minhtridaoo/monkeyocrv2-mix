"""Mix A-core step A2: forward each merged checkpoint once and score Tuan's 9 x 8 quad grid.

Reuses MONKEY_V2 code unchanged (imported via mixab.monkey_janek at the pinned checkout):
`rev4.stage_a.detector_for/forward`, `rev4.decision_core.capture_candidates/CandidateMatcher`,
`rev4.decision_plan.GEOMETRIES/SCORES/INSTALLED_PP_SHA256`, `rev4.training_data.native_lr2`.
Inputs come from Tuan's published bundle `docs/handoff/artifacts/pr_quad_audit_v1/`.

Hardware: the member has only T4 or A100 GPUs (Tuan's reference NPZ came from H200). Batch 1,
FP32 as the original pipeline; progress is checkpointed in chunks so an interrupted session resumes.
Cross-GPU numerics may change counts: run `--probe` on the target GPU first and read gates.json.
Never loosen a gate in code; report mismatches (plan section 4.3, hardware note).

Reference policy (plan section 4.3, decision of 20/11 = option a): with
`--reference-policy same_gpu_endpoints --same-gpu-probe <probe run dir>` the full run must have the
same identity as that probe (code, config, manifests, GPU model, torch/CUDA/cuDNN; only the source
subset differs). reselect_split then requires the endpoint counts to equal the probe counts exactly;
gate 3a/3b against H200 are kept as cross-GPU drift reports.

Outputs (new directory, never overwritten):
  <name>_source_counts.npz   counts [source, cond, setting, 11] = tp, gt, pred, (gt_bi, tp_bi) x 4 bins
  gates.json                 gate 3a (EDA .30/.40, endpoints), P-map endpoint check, status per model
  audit_identity.json        MONKEY_V2 commit/dirty, bundle/manifest hashes, GPU, torch, flags
  reselect_spec.json         ready for `python -m mixab.reselect_split_janek --spec ...`
  profile.json               seconds per task and peak GPU memory per model (this GPU only)

Subcommands:
  prepare-lr2  materialize native LR2 PNGs from Clean with Tuan's native_lr2 and check SHA-256
  run          the audit (use --probe N for a small fixed subset first)
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import sys
import time
from pathlib import Path, PureWindowsPath

from mixab import read_json, require, save_json, sha256 

STAGE = 'Mix A-core merge audit v1'
GT_MANIFEST_SHA256 = '052857fba8b6cd7c78e48a3dcb53475730d03f2ca9d885b66a4a562eea38872c'
PROBE_SEED = 20261119
PMAP_IMAGES = 20                       # 10 Clean + 10 LR2, plan section 4.3
PMAP_TOLERANCE = 1e-6
FIELDS = 11
CONDITIONS = ('clean', 'lr2')
ROOT_KEYS = ('TextVQA_clean_root', 'prepared_root')
POLICIES = ('h200_reference', 'same_gpu_endpoints')
SUBSET_KEYS = ('probe_sources', 'sources_sha256', 'identity_sha256')   # may differ between probe and full run


def identity_mismatch(full, probe):
    """Keys (other than the source subset) whose values differ between a full run and its probe."""
    keys = (set(full) | set(probe)) - set(SUBSET_KEYS)
    return sorted(k for k in keys if full.get(k) != probe.get(k))


def same_gpu_reference(probe_dir, identity):
    """Validate a finished probe run as the same-GPU reference; return the reselect_spec block."""
    probe_dir = Path(probe_dir)
    require((probe_dir/'gates.json').is_file(), f'{probe_dir}: probe run is not finished (no gates.json)')
    probe_identity = read_json(probe_dir/'audit_identity.json')
    require(probe_identity.get('probe_sources'), f'{probe_dir}: not a probe run')
    diff = identity_mismatch(identity, probe_identity)
    require(not diff, f'{probe_dir}: probe identity differs from this run in {diff}')
    npz = {name: str(probe_dir/f'{name}_source_counts.npz') for name in ('B-Clean', 'B1')}
    for path in npz.values():
        require(Path(path).is_file(), f'Missing probe counts {path}')
    return dict(run_gpu=identity['gpu'], reference_gpu=probe_identity['gpu'], probe_dir=str(probe_dir),
                probe_identity_sha256=probe_identity['identity_sha256'], npz=npz)


# ---------------------------------------------------------------- pure helpers (CPU-testable)

def model_name(alpha):
    return {'0': 'B-Clean', '1': 'B1'}.get(str(alpha), f'merge_a{str(alpha).replace("/", "_")}')


def load_rows(bundle):
    """GT/ROI manifest (exact locked bytes) keyed by source id."""
    raw = gzip.decompress((Path(bundle)/'data/validation_remaining.jsonl.gz').read_bytes())
    require(hashlib.sha256(raw).hexdigest() == GT_MANIFEST_SHA256, 'GT manifest bytes differ from the locked SHA-256')
    rows = {}
    for line in io.TextIOWrapper(io.BytesIO(raw), encoding='utf-8'):
        if line.strip():
            row = json.loads(line)
            rows[row['source_image_id']] = row
    return rows


def load_task_metadata(bundle):
    with (Path(bundle)/'data/tasks_metadata.csv').open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def resolve_tasks(metadata, roots, sources=None):
    """Build MONKEY_V2 task dicts (stage_a.make_task layout) from metadata + the runner's roots."""
    require(set(roots) >= {m['path_base'] for m in metadata}, f'Need roots for {sorted({m["path_base"] for m in metadata})}')
    keep = None if sources is None else set(sources)
    tasks = []
    for m in metadata:
        if keep is not None and m['source_id'] not in keep:
            continue
        tasks.append(dict(sample_id=m['sample_id'], source_image_id=m['source_id'], condition=m['condition'],
                          path=str(Path(roots[m['path_base']])/m['relative_path']),
                          size=[int(m['width']), int(m['height'])], image_sha256=m['image_sha256']))
    return tasks


def probe_sources(source_ids, count, seed=PROBE_SEED):
    """Fixed, order-independent subset: smallest SHA-256(seed:source) ranks."""
    ranked = sorted(source_ids, key=lambda s: hashlib.sha256(f'{seed}:{s}'.encode('utf-8')).hexdigest())
    return sorted(ranked[:count])


def pmap_tasks(tasks, seed=PROBE_SEED, per_condition=PMAP_IMAGES//2):
    out = []
    for condition in CONDITIONS:
        pool = sorted((t for t in tasks if t['condition'] == condition),
                      key=lambda t: hashlib.sha256(f'{seed}:pmap:{t["sample_id"]}'.encode('utf-8')).hexdigest())
        out += pool[:per_condition]
    return out


def build_settings(geometries, scores):
    """Same dicts and order as rev4/decision_audit.py lines 149-153."""
    return [dict(geometry=g[0], mask_thr=g[1], unclip_ratio=g[2], min_text_width=g[3], epsilon_ratio=g[4],
                 min_text_score=s) for g in geometries for s in scores]


def count_row(scored):
    """CandidateMatcher.score() output -> the 11-column row."""
    tp, gt, pred = scored['counts']
    bins = scored['bins']
    require(len(bins) == 4, 'CandidateMatcher must return four height bins')
    return [int(tp), int(gt), int(pred)] + [int(v) for pair in bins for v in pair]


def load_eda_counts(bundle):
    """(model, source_id, condition) -> (tp, gt, pred) at common .30/.40 from the locked EDA caches."""
    out = {}
    with gzip.open(Path(bundle)/'provenance/eda_common_counts.csv.gz', 'rt', encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            require(float(r['mask_thr']) == .3 and float(r['min_text_score']) == .4, 'EDA rows must be .30/.40')
            out[(r['model'], r['source_id'], r['condition'])] = (int(r['tp']), int(r['gt']), int(r['pred']))
    return out


def gate_3a(name, counts, source_ids, common_index, eda):
    """Compare model counts at the common setting with EDA; endpoints only (EDA has B-Clean/B1)."""
    checked = mismatched = missing = 0
    examples = []
    for i, source in enumerate(source_ids):
        for c, condition in enumerate(CONDITIONS):
            ref = eda.get((name, source, condition))
            checked += 1
            if ref is None:
                missing += 1
                continue
            got = tuple(int(x) for x in counts[i, c, common_index, :3])
            if got != ref:
                mismatched += 1
                if len(examples) < 10:
                    examples.append(dict(source_id=source, condition=condition, got=list(got), eda=list(ref)))
    status = 'PASS' if checked and not mismatched and not missing else 'FAIL'
    return dict(status=status, checked=checked, mismatched=mismatched, missing=missing, examples=examples)


class Progress:
    """Chunk checkpoint of one model's count matrix, bound to a run identity (resumable)."""

    def __init__(self, folder, name, identity_sha, shape):
        import numpy as np

        self.path = Path(folder)/f'.{name}.partial.npz'
        self.meta = Path(folder)/f'.{name}.partial.json'
        self.identity_sha = identity_sha
        self.matrix = np.zeros(shape, dtype=np.int64)
        self.done = set()
        if self.meta.is_file():
            meta = read_json(self.meta)
            require(meta['identity_sha256'] == identity_sha, f'Partial {name} belongs to another run identity')
            with np.load(self.path, allow_pickle=False) as data:
                require(data['counts'].shape == shape, 'Partial matrix shape changed')
                self.matrix[...] = data['counts']
            self.done = set(meta['done'])

    def save(self):
        import numpy as np

        temp = self.path.with_name(self.path.name + '.tmp.npz')
        np.savez_compressed(temp, counts=self.matrix)
        temp.replace(self.path)
        save_json(self.meta, dict(identity_sha256=self.identity_sha, done=sorted(self.done)))

    def clear(self):
        for p in (self.path, self.meta):
            if p.exists():
                p.unlink()


def write_counts(path, source_ids, settings, counts):
    import numpy as np

    temp = Path(path).with_name(Path(path).name + '.tmp.npz')
    np.savez_compressed(temp, source_ids=np.asarray(source_ids),
                        settings=np.asarray([json.dumps(s, sort_keys=True) for s in settings]), counts=counts)
    temp.replace(path)


def overall_status(gates):
    values = [gates.get(k, {}).get('status', 'NOT_RUN') for k in ('gate_3a', 'pmap_endpoint')]
    return 'FAIL' if 'FAIL' in values else ('PASS' if all(v == 'PASS' for v in values) else 'NOT_RUN')


# ---------------------------------------------------------------- GPU worker (needs MMOCR + CUDA)

def prepare_lr2(a):
    from mixab.monkey_janek import use_monkey
    use_monkey()
    from PIL import Image
    from rev4.training_data import native_lr2

    rows = load_rows(a.bundle)
    roots = dict(TextVQA_clean_root=a.clean_root, prepared_root=a.prepared_root)
    made = verified = 0
    for task in resolve_tasks(load_task_metadata(a.bundle), roots):
        if task['condition'] != 'lr2':
            continue
        out = Path(task['path'])
        if not out.is_file():
            clean = Path(a.clean_root)/rows[task['source_image_id']]['clean_relpath']
            require(sha256(clean) == rows[task['source_image_id']]['clean_sha256'], f'Clean bytes differ: {clean}')
            with Image.open(clean) as image:
                lr2 = native_lr2(image)          # Pillow 11.3.0 bicubic, max(1, size//2), as Tuan's pipeline
            lr2.info.clear()                     # same as rev4/training_data.validation_tasks
            out.parent.mkdir(parents=True, exist_ok=True)
            lr2.save(out, format='PNG')
            made += 1
        require(sha256(out) == task['image_sha256'], f'LR2 bytes differ from the locked task: {out}')
        verified += 1
    print(f'LR2 ready: {verified} verified ({made} newly written)')


def run(a):
    import numpy as np
    import torch

    from mixab.monkey_janek import use_monkey
    monkey = use_monkey()
    from rev4.decision_core import CandidateMatcher, capture_candidates
    from rev4.decision_plan import GEOMETRIES, INSTALLED_PP_SHA256, SCORES
    from rev4.decision_audit import installed_postprocessor
    from rev4.stage_a import detector_for, forward

    output = Path(a.output)
    require(not (output/'gates.json').exists(), f'Refusing to overwrite a finished run in {output}')
    output.mkdir(parents=True, exist_ok=True)
    pp_path = installed_postprocessor()          # requires the pinned DBPostprocessor SHA-256
    bundle = Path(a.bundle)
    rows = load_rows(bundle)
    policy = read_json(bundle/'provenance/textvqa_evaluation_protocol.json')
    metadata = load_task_metadata(bundle)
    all_sources = sorted(rows)
    sources = probe_sources(all_sources, a.probe) if a.probe else all_sources
    roots = dict(TextVQA_clean_root=a.clean_root, prepared_root=a.prepared_root)
    tasks = resolve_tasks(metadata, roots, sources)
    require(len(tasks) == 2*len(sources), 'Each source needs exactly one Clean and one LR2 task')
    settings = build_settings(GEOMETRIES, SCORES)
    common = next(i for i, s in enumerate(settings) if s['geometry'] == 'base' and s['min_text_score'] == .4)
    manifest_path = Path(a.weights_dir)/'merge_manifest.json'
    manifest = read_json(manifest_path)
    exports = {'0': Path(a.bclean_export), '1': Path(a.b1_export)}
    require(sha256(exports['0']) == manifest['endpoints']['bclean']['sha256'], 'B-Clean export differs from merge manifest')
    require(sha256(exports['1']) == manifest['endpoints']['b1']['sha256'], 'B1 export differs from merge manifest')
    identity = dict(stage=STAGE, monkey_v2=monkey, installed_db_postprocessor_sha256=INSTALLED_PP_SHA256,
                    installed_db_postprocessor_path=str(pp_path), gt_manifest_sha256=GT_MANIFEST_SHA256,
                    bundle_manifest_sha256=sha256(bundle/'bundle_manifest.json'),
                    merge_manifest_sha256=sha256(manifest_path), probe_sources=a.probe or None,
                    sources_sha256=hashlib.sha256('\n'.join(sources).encode('utf-8')).hexdigest(),
                    settings=settings, model_config_sha256=sha256(a.model_config),
                    gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
                    cudnn=torch.backends.cudnn.version(), no_test=True, training=False)
    identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True).encode('utf-8')).hexdigest()
    require(a.reference_policy in POLICIES, f'Unknown reference policy {a.reference_policy!r}')
    same_gpu = None
    if a.reference_policy == 'same_gpu_endpoints':
        require(not a.probe and a.same_gpu_probe, 'same_gpu_endpoints is for the full run and needs --same-gpu-probe')
        same_gpu = same_gpu_reference(a.same_gpu_probe, identity)      # fail before any GPU work
        print(f'Same-GPU reference: {same_gpu["probe_dir"]} ({same_gpu["reference_gpu"]})', flush=True)
    else:
        require(not a.same_gpu_probe, '--same-gpu-probe only applies to --reference-policy same_gpu_endpoints')
    save_json(output/'audit_identity.json', dict(identity, identity_sha256=identity_sha))
    source_index = {s: i for i, s in enumerate(sources)}
    eda = load_eda_counts(bundle)
    gates, profile, spec_models = {}, {}, []
    for model in manifest['models']:
        alpha, name = model['alpha'], model_name(model['alpha'])
        checkpoint = Path(a.weights_dir)/PureWindowsPath(model['checkpoint']).name
        require(sha256(checkpoint) == model['checkpoint_sha256'], f'{checkpoint} differs from merge manifest')
        final = output/f'{name}_source_counts.npz'
        record_path = output/f'{name}.model.json'     # written only after counts AND gates are done
        spec_models.append(dict(name=name, alpha=alpha, npz=str(final)))
        if final.is_file() and record_path.is_file():
            record = read_json(record_path)
            require(record['identity_sha256'] == identity_sha, f'{name}: finished record belongs to another run')
            profile[name] = record['profile']
            if 'gates' in record:
                gates[name] = record['gates']
            print(f'{name}: already complete, reusing saved counts/gates', flush=True)
            continue
        progress = Progress(output, name, identity_sha, (len(sources), 2, len(settings), FIELDS))
        options = argparse.Namespace(model_config=Path(a.model_config), checkpoint=checkpoint)
        detector, native, _, _ = detector_for(options)
        torch.cuda.reset_peak_memory_stats()
        seconds, started = [], time.perf_counter()
        for number, task in enumerate(tasks, 1):
            if task['sample_id'] in progress.done:
                continue
            tic = time.perf_counter()
            row = rows[task['source_image_id']]
            _, probability, sample, _, restore, _, _ = forward(detector, native, task, row)
            c = CONDITIONS.index(task['condition'])
            i = source_index[task['source_image_id']]
            for gi, geometry in enumerate(GEOMETRIES):
                prediction, raw_scores, _ = capture_candidates(native, probability, sample, restore, geometry)
                matcher = CandidateMatcher(prediction, raw_scores, row, policy)
                for si, score in enumerate(SCORES):
                    progress.matrix[i, c, gi*len(SCORES) + si] = count_row(matcher.score(score))
                if gi == 0 and number == 1:
                    matcher.assert_official(.40)      # evaluator equivalence, as decision_audit does
            progress.done.add(task['sample_id'])
            seconds.append(time.perf_counter() - tic)
            if len(progress.done) % a.chunk == 0:
                progress.save()
                print(f'{name}: {len(progress.done):,}/{len(tasks):,} tasks, '
                      f'{sum(seconds)/len(seconds):.2f} s/task', flush=True)
        progress.save()
        profile[name] = dict(tasks=len(seconds), mean_seconds_per_task=sum(seconds)/max(len(seconds), 1),
                             wall_seconds=time.perf_counter() - started,
                             peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20)
        if alpha in ('0', '1'):
            ref_name = 'B-Clean' if alpha == '0' else 'B1'
            gates[name] = dict(gate_3a=gate_3a(ref_name, progress.matrix, sources, common, eda),
                               pmap_endpoint=pmap_check(detector, native, exports[alpha], a.model_config,
                                                        pmap_tasks(tasks), rows, forward, detector_for))
            gates[name]['status'] = overall_status(gates[name])
            print(f'{name}: gate 3a {gates[name]["gate_3a"]["status"]} '
                  f'({gates[name]["gate_3a"]["mismatched"]} mismatches), '
                  f'P-map {gates[name]["pmap_endpoint"]["status"]}', flush=True)
        write_counts(final, sources, settings, progress.matrix)
        record = dict(identity_sha256=identity_sha, profile=profile[name])
        if name in gates:
            record['gates'] = gates[name]
        save_json(record_path, record)
        progress.clear()
        del detector
        torch.cuda.empty_cache()
        save_json(output/'profile.json', profile)
    endpoint_status = [gates.get(n, {}).get('status', 'NOT_RUN') for n in ('B-Clean', 'B1')]
    checks = dict(gate_3a=_combined(gates, 'gate_3a'), pmap_endpoint=_combined(gates, 'pmap_endpoint'))
    save_json(output/'gates.json', dict(models=gates, endpoint_status=endpoint_status, probe=a.probe or None,
                                        gpu=identity['gpu']))
    counts_dir = bundle/'counts'
    references = {} if a.probe else {'B-Clean': str(counts_dir/'B-Clean_source_counts.npz'),
                                     'B1': str(counts_dir/'B1_source_counts.npz')}
    spec = dict(models=spec_models, references=references, checks=checks, reference_policy=a.reference_policy,
                _note='probe runs carry no gate 3b reference: subset only')
    if same_gpu:
        spec['same_gpu'] = same_gpu
        spec['_note'] = 'same_gpu_endpoints: H200 gate 3a/3b are reported as cross-GPU drift only (plan 4.3)'
    save_json(output/'reselect_spec.json', spec)
    print(json.dumps(dict(gates=endpoint_status, checks=checks), indent=1))


def _combined(gates, key):
    values = [gates.get(n, {}).get(key, {}).get('status', 'NOT_RUN') for n in ('B-Clean', 'B1')]
    return 'FAIL' if 'FAIL' in values else ('PASS' if all(v == 'PASS' for v in values) else 'NOT_RUN')


def pmap_check(detector, native, export, model_config, tasks, rows, forward, detector_for):
    """Endpoint copy vs the original export through the original inference path, same GPU."""
    import torch

    original, native_original, _, _ = detector_for(argparse.Namespace(model_config=Path(model_config),
                                                                      checkpoint=Path(export)))
    worst = 0.
    for task in tasks:
        row = rows[task['source_image_id']]
        _, p_copy, _, _, _, _, _ = forward(detector, native, task, row)
        _, p_orig, _, _, _, _, _ = forward(original, native_original, task, row)
        require(p_copy.shape == p_orig.shape, 'P-map shapes differ')
        worst = max(worst, float((p_copy - p_orig).abs().max()))
    del original
    torch.cuda.empty_cache()
    return dict(status='PASS' if worst <= PMAP_TOLERANCE else 'FAIL', images=len(tasks),
                max_abs_diff=worst, tolerance=PMAP_TOLERANCE)


def main():
    sys.stdout.reconfigure(errors='backslashreplace')   # never crash on non-ASCII paths (Windows cp1252)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare-lr2', 'run'):
        p = sub.add_parser(name)
        p.add_argument('--bundle', required=True, type=Path, help='MONKEY_V2/docs/handoff/artifacts/pr_quad_audit_v1')
        p.add_argument('--clean-root', required=True, type=Path, help='Root holding train_images/ (TextVQA)')
        p.add_argument('--prepared-root', required=True, type=Path, help='Root that holds/receives validation_lr2/')
    run_p = sub.choices['run']
    run_p.add_argument('--model-config', required=True, type=Path, help='MONKEY_V2/docs/handoff/inference_config.py (encoder paths set)')
    run_p.add_argument('--weights-dir', required=True, type=Path, help='Output of mixab.weight_merge_janek')
    run_p.add_argument('--bclean-export', required=True, type=Path)
    run_p.add_argument('--b1-export', required=True, type=Path)
    run_p.add_argument('--output', required=True, type=Path)
    run_p.add_argument('--probe', type=int, default=0, help='Fixed subset of N sources (0 = all 2,262)')
    run_p.add_argument('--chunk', type=int, default=64, help='Save progress every N tasks')
    run_p.add_argument('--reference-policy', choices=POLICIES, default='h200_reference',
                       help='same_gpu_endpoints = plan 4.3 option (a), needs --same-gpu-probe')
    run_p.add_argument('--same-gpu-probe', type=Path, help='Finished probe run on the same GPU model')
    a = parser.parse_args()
    prepare_lr2(a) if a.command == 'prepare-lr2' else run(a)


if __name__ == '__main__':
    main()
