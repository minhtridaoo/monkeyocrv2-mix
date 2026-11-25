"""E-BN pilot, Stage 0 (plan section 4.4, docs/research/janek/design_bn_recalibration_pilot_janek.md). EXPLORATORY.

Question: after linear weight interpolation (Mix A-core v1 checkpoints), how far are the stored
(interpolated) BatchNorm statistics from the statistics the merged network actually produces on
fixed TRAIN images? Endpoints (alpha 0, 1) are measured the same way as controls.

Stage 0 only MEASURES: the model stays in eval() and nothing is written back to any BN buffer.
Only det_head.binarize.1 and det_head.binarize.4 are measured: in MMOCR v1.0.1 the detector forward
(mode 'tensor' -> DBHead mode 'predict') runs only the binarize branch; backbone and neck have no BN.
Stage 1 (recalibrate + evaluate counts) is NOT implemented here: it waits for agreed criteria.

TRAIN input (never re-split TextVQA ourselves):
  * full manifest  manifest__train__textvqa_clean_degraded_same_textocr_word_gt.jsonl  -> SHA-256 must
    equal TRAIN_MANIFEST_SHA256 (Tuan, 20/11), otherwise stop;
  * or an ID list CSV with columns source_image_id, clean_relpath, clean_sha256 (optional clean_size):
    accepted, but recorded as manifest_sha256_status = NOT_VERIFIABLE_ID_LIST.
Both must hold exactly 15,222 unique sources, disjoint (ID and clean SHA) from the 2,262 published
remaining-validation sources. The 1,000 S1/S2 validation sources are not published: not checked.

Subcommands:
  check-train     validate the TRAIN input, choose the fixed subset (no images needed)
  prepare-images  verify Clean bytes of the subset, write native LR2 PNGs with Tuan's native_lr2
  stage0          measure BN input statistics for every checkpoint in merge_manifest.json (GPU, resumable)
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import sys
import time
from pathlib import Path, PurePosixPath

from mixab import read_json, require, save_json, sha256 

STAGE = 'E-BN stage 0 (exploratory) v1'
TRAIN_MANIFEST_NAME = 'manifest__train__textvqa_clean_degraded_same_textocr_word_gt.jsonl'
TRAIN_MANIFEST_SHA256 = 'ca1b91636cbe025823041ce336ffb59affa826a073f666dd7ba386bcab6c983d'
TRAIN_SOURCES = 15_222
SUBSET_SOURCES = 1_000
SUBSET_SEED = 20261120
BN_LAYERS = ('det_head.binarize.1', 'det_head.binarize.4')
CONDITIONS = ('clean', 'lr2')
BN_EPS = 1e-5                       # torch.nn.BatchNorm2d default; read from the module when available
PROPOSED_SIGNAL_RATIO = 2.0         # PROPOSAL (not approved by Tuan): see stage0_gate
ID_COLUMNS = ('source_image_id', 'clean_relpath', 'clean_sha256')
HEX64 = re.compile(r'^[0-9a-f]{64}$')


def say(text):
    """Print without crashing on a non-UTF-8 console (Windows cp1252); Colab is UTF-8."""
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'backslashreplace').decode('ascii'), flush=True)


# ---------------------------------------------------------------- TRAIN input (CPU)

def _rows_from_jsonl(raw):
    rows = []
    for line in io.TextIOWrapper(io.BytesIO(raw), encoding='utf-8'):
        if line.strip():
            row = json.loads(line)
            rows.append({k: row.get(k) for k in ID_COLUMNS + ('clean_size',)})
    return rows


def read_train_input(path):
    """Return dict(kind, input_sha256, manifest_sha256_status, rows). Stops on a wrong full manifest."""
    path = Path(path)
    require(path.is_file(), f'Không thấy file TRAIN: {path}')
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    name = path.name.lower()
    if name.endswith('.jsonl') or name.endswith('.jsonl.gz'):
        if name.endswith('.gz'):
            raw = gzip.decompress(raw)
            digest_plain = hashlib.sha256(raw).hexdigest()
        else:
            digest_plain = digest
        require(digest_plain == TRAIN_MANIFEST_SHA256,
                f'Manifest TRAIN có SHA-256 {digest_plain}, khác mã Tuấn xác nhận {TRAIN_MANIFEST_SHA256}. '
                'Dừng: không dùng manifest khác bản gốc.')
        return dict(kind='full_manifest', input_sha256=digest, manifest_sha256=digest_plain,
                    manifest_sha256_status='VERIFIED_MATCH', rows=_rows_from_jsonl(raw))
    if name.endswith('.csv'):
        text = raw.decode('utf-8-sig')
        reader = csv.DictReader(io.StringIO(text))
        missing = [c for c in ID_COLUMNS if c not in (reader.fieldnames or [])]
        require(not missing, f'File danh sách TRAIN thiếu cột {missing}; cần {list(ID_COLUMNS)}')
        rows = []
        for r in reader:
            size = r.get('clean_size')
            rows.append(dict(source_image_id=r['source_image_id'].strip(), clean_relpath=r['clean_relpath'].strip(),
                             clean_sha256=r['clean_sha256'].strip().lower(),
                             clean_size=json.loads(size) if size else None))
        return dict(kind='id_list', input_sha256=digest, manifest_sha256=None,
                    manifest_sha256_status='NOT_VERIFIABLE_ID_LIST', rows=rows)
    raise ValueError(f'Không nhận dạng được {path.name}: cần {TRAIN_MANIFEST_NAME} (hoặc .jsonl.gz) hoặc file .csv')


def safe_relpath(relpath):
    parts = PurePosixPath(relpath).parts
    return bool(parts) and parts[0] == 'train_images' and '..' not in parts and not relpath.startswith('/')


def validate_train_rows(rows, validation_rows):
    """Exact checks; returns a report. validation_rows: {source_id: row} of remaining-validation."""
    ids = [r['source_image_id'] for r in rows]
    problems = []
    if len(ids) != TRAIN_SOURCES:
        problems.append(f'{len(ids)} nguồn, kỳ vọng đúng {TRAIN_SOURCES}')
    if len(set(ids)) != len(ids):
        problems.append(f'{len(ids) - len(set(ids))} source_image_id bị trùng')
    bad_sha = [r['source_image_id'] for r in rows if not HEX64.match(str(r['clean_sha256'] or ''))]
    if bad_sha:
        problems.append(f'{len(bad_sha)} dòng clean_sha256 không hợp lệ, ví dụ {bad_sha[:3]}')
    bad_path = [r['source_image_id'] for r in rows if not safe_relpath(str(r['clean_relpath'] or ''))]
    if bad_path:
        problems.append(f'{len(bad_path)} clean_relpath không nằm trong train_images/, ví dụ {bad_path[:3]}')
    val_ids = set(validation_rows)
    val_sha = {r['clean_sha256'] for r in validation_rows.values()}
    overlap_id = sorted(set(ids) & val_ids)
    overlap_sha = sorted({r['clean_sha256'] for r in rows} & val_sha)
    if overlap_id or overlap_sha:
        problems.append(f'Trùng với remaining-validation: {len(overlap_id)} ID, {len(overlap_sha)} SHA')
    return dict(status='PASS' if not problems else 'FAIL', problems=problems, sources=len(ids),
                validation_overlap=dict(checked_against=len(val_ids), id=len(overlap_id), clean_sha=len(overlap_sha),
                                        not_checked='1.000 nguồn S1/S2 của validation đóng băng (chưa công bố)'))


def select_subset(rows, n=SUBSET_SOURCES, seed=SUBSET_SEED):
    """Fixed subset: rows sorted by source_image_id, numpy default_rng(seed).choice without replacement."""
    import numpy as np

    ordered = sorted(rows, key=lambda r: r['source_image_id'])
    require(len(ordered) >= n, f'Cần ít nhất {n} nguồn')
    picks = np.sort(np.random.default_rng(seed).choice(len(ordered), size=n, replace=False))
    return [ordered[int(i)] for i in picks]


def write_csv(path, rows, fields):
    path = Path(path)
    temp = path.with_name(path.name + '.part')
    with temp.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (json.dumps(r[k]) if isinstance(r.get(k), list) else r.get(k)) for k in fields})
    temp.replace(path)


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def load_validation_rows(bundle):
    from mixab.merge_audit_janek import load_rows
    return load_rows(bundle)


def check_train(a):
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    info = read_train_input(a.train_input)
    report = validate_train_rows(info['rows'], load_validation_rows(a.bundle))
    subset = select_subset(info['rows']) if report['status'] == 'PASS' else []
    record = dict(stage=STAGE, train_input=str(a.train_input), kind=info['kind'], input_sha256=info['input_sha256'],
                  manifest_sha256=info['manifest_sha256'], manifest_sha256_status=info['manifest_sha256_status'],
                  expected_manifest_sha256=TRAIN_MANIFEST_SHA256, check=report,
                  subset=dict(sources=len(subset), seed=SUBSET_SEED, rule='sorted by source_image_id; '
                              'numpy default_rng(seed).choice(n, replace=False)'))
    subset_path = out/'subset.csv'
    if subset:
        fields = ['source_image_id', 'clean_relpath', 'clean_sha256', 'clean_size']
        if subset_path.is_file():
            old = read_csv(subset_path)
            require([r['source_image_id'] for r in old] == [r['source_image_id'] for r in subset] and
                    [r['clean_sha256'] for r in old] == [r['clean_sha256'] for r in subset],
                    f'{subset_path} đã có và khác subset mới (đầu vào TRAIN đã đổi?). Không ghi đè.')
        else:
            write_csv(subset_path, subset, fields)
        record['subset']['sha256'] = sha256(subset_path)
    save_json(out/'train_check.json', record)
    say(json.dumps({k: v for k, v in record.items() if k != 'check'} | dict(check=report['status']), indent=1,
                   ensure_ascii=False))
    require(report['status'] == 'PASS', 'Danh sách TRAIN không hợp lệ: ' + '; '.join(report['problems']))


# ---------------------------------------------------------------- images (CPU, needs Pillow 11.3.0 + MONKEY_V2)

def prepare_images(a):
    from mixab.monkey_janek import use_monkey
    use_monkey()
    from PIL import Image
    from rev4.training_data import native_lr2

    rows = read_csv(a.subset)
    record_path = Path(a.record)
    previous = {r['source_image_id']: r for r in read_csv(record_path)} if record_path.is_file() else {}
    out_rows = []
    for r in rows:
        clean = Path(a.clean_root)/r['clean_relpath']
        require(clean.is_file(), f'Thiếu ảnh Clean {clean}')
        require(sha256(clean) == r['clean_sha256'], f'Ảnh Clean sai SHA-256: {clean}')
        lr2 = Path(a.lr2_root)/f'{r["source_image_id"]}.png'
        with Image.open(clean) as image:
            size = list(image.size)
            if r.get('clean_size'):
                require(json.loads(r['clean_size']) == size, f'Kích thước Clean khác manifest: {clean}')
            if not lr2.is_file():
                out = native_lr2(image)             # Pillow 11.3.0 bicubic, max(1, side//2), EXIF must be 1
                out.info.clear()                    # as rev4/training_data.validation_tasks
                lr2.parent.mkdir(parents=True, exist_ok=True)
                temp = lr2.with_name(lr2.name + '.part')
                out.save(temp, format='PNG')
                temp.replace(lr2)
        lr2_sha = sha256(lr2)
        old = previous.get(r['source_image_id'])
        if old:   # LR2 regenerated on a new VM must be byte-identical to the first run
            require(old['lr2_sha256'] == lr2_sha, f'LR2 tạo lại khác lần trước: {r["source_image_id"]}')
        out_rows.append(dict(source_image_id=r['source_image_id'], clean_relpath=r['clean_relpath'],
                             clean_sha256=r['clean_sha256'], lr2_name=lr2.name, lr2_sha256=lr2_sha))
    if previous:
        require(set(previous) == {r['source_image_id'] for r in out_rows}, f'{record_path} thuộc subset khác')
    else:
        record_path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(record_path, out_rows, list(out_rows[0]))
    say(f'Ảnh TRAIN subset: {len(out_rows)} Clean SHA khớp, {len(out_rows)} LR2 '
        f'({"khớp lần trước" if previous else "đã ghi SHA lần đầu"}).')


# ---------------------------------------------------------------- statistics (pure, CPU-testable)

class Accumulator:
    """Per (layer, condition): sums over images of per-image channel mean and unbiased variance
    (what torch BatchNorm with momentum=None would average with batch size 1), plus pixel-pooled
    sum / sum of squares / count."""

    def __init__(self, channels):
        import numpy as np
        self.data = {f'{layer}|{cond}|{key}': np.zeros(ch if key != 'n' else 1, dtype=np.float64)
                     for layer, ch in channels.items() for cond in CONDITIONS
                     for key in ('n', 'sum_mean', 'sum_var', 'pix_sum', 'pix_sumsq', 'pix_count')}

    def update(self, layer, condition, x):
        """x: tensor [1, C, H, W]."""
        x = x.detach().double()
        require(x.ndim == 4 and x.shape[0] == 1, 'Batch 1 expected')
        flat = x[0].reshape(x.shape[1], -1)
        k = f'{layer}|{condition}|'
        self.data[k + 'n'] += 1
        self.data[k + 'sum_mean'] += flat.mean(dim=1).cpu().numpy()
        self.data[k + 'sum_var'] += flat.var(dim=1, unbiased=True).cpu().numpy()
        self.data[k + 'pix_sum'] += flat.sum(dim=1).cpu().numpy()
        self.data[k + 'pix_sumsq'] += (flat*flat).sum(dim=1).cpu().numpy()
        self.data[k + 'pix_count'] += flat.shape[1]

    def stats(self, layer, conditions=CONDITIONS):
        """Image-averaged mean/var over the given conditions (equal weight per image)."""
        n = sum(float(self.data[f'{layer}|{c}|n'][0]) for c in conditions)
        require(n > 0, f'No images for {layer} {conditions}')
        mean = sum(self.data[f'{layer}|{c}|sum_mean'] for c in conditions)/n
        var = sum(self.data[f'{layer}|{c}|sum_var'] for c in conditions)/n
        return mean, var, int(n)

    def save(self, path):
        import numpy as np
        temp = Path(str(path) + '.part.npz')
        np.savez(temp, **self.data)
        temp.replace(path)

    def load(self, path):
        import numpy as np
        with np.load(path) as f:
            require(set(f.files) == set(self.data), 'Accumulator file has other layers/conditions')
            self.data = {k: f[k].astype(np.float64) for k in f.files}


def shift_metrics(mean, var, stored_mean, stored_var, eps=BN_EPS):
    """Standardized mean shift |mu - mu_s| / sqrt(var_s + eps) and |log variance ratio|, per channel."""
    import numpy as np
    z = np.abs(mean - stored_mean)/np.sqrt(stored_var + eps)
    lv = np.abs(np.log((var + eps)/(stored_var + eps)))
    return dict(z_mean=float(z.mean()), z_max=float(z.max()), logvar_mean=float(lv.mean()), logvar_max=float(lv.max()))


def nonlinearity(measured, alpha):
    """How far measured stats of alpha are from linear interpolation of the endpoints' measured stats.
    measured: {alpha (Fraction): (mean, var)}; standardized by the interpolated variance."""
    import numpy as np
    (m0, v0), (m1, v1), (ma, va) = measured[0], measured[1], measured[alpha]
    a = float(alpha)
    lm, lv = (1 - a)*m0 + a*m1, (1 - a)*v0 + a*v1
    z = np.abs(ma - lm)/np.sqrt(lv + BN_EPS)
    return dict(z_mean=float(z.mean()), z_max=float(z.max()),
                logvar_mean=float(np.abs(np.log((va + BN_EPS)/(lv + BN_EPS))).mean()))


def stage0_gate(per_model, layer='det_head.binarize.1', ratio=PROPOSED_SIGNAL_RATIO):
    """PROPOSED gate (not approved): signal if any interior alpha's z_mean (stored vs measured, Clean+LR2)
    exceeds ratio x the larger endpoint z_mean on `layer`. per_model: {alpha_str: {layer: {'all': metrics}}}."""
    from fractions import Fraction
    ends = [per_model[a][layer]['all']['z_mean'] for a in per_model if Fraction(a) in (0, 1)]
    require(len(ends) == 2, 'Both endpoints are required as controls')
    base = max(ends)
    interior = {a: per_model[a][layer]['all']['z_mean'] for a in per_model if Fraction(a) not in (0, 1)}
    hits = sorted(a for a, z in interior.items() if z > ratio*base)
    return dict(status='PROPOSED_NOT_APPROVED', rule=f'interior z_mean({layer}, Clean+LR2) > {ratio} x max(endpoint z_mean)',
                endpoint_baseline=base, interior=interior, signal=bool(hits), alphas_with_signal=hits)


# ---------------------------------------------------------------- GPU worker

def state_digest(model):
    """SHA-256 over every parameter and buffer (to prove stage 0 changed nothing)."""
    digest = hashlib.sha256()
    for name, tensor in sorted(list(model.state_dict().items())):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def attach_hooks(model, accumulator, condition_ref, layers=BN_LAYERS):
    modules = dict(model.named_modules())
    missing = [n for n in layers if n not in modules]
    require(not missing, f'Model has no BN layer {missing}')
    handles = []
    for name in layers:
        def hook(module, args, name=name):
            accumulator.update(name, condition_ref[0], args[0])
        handles.append(modules[name].register_forward_pre_hook(hook))
    return handles


def stage0(a):
    import numpy as np
    import torch

    from mixab.monkey_janek import use_monkey
    monkey = use_monkey()
    from rev4.stage_a import detector_for

    output = Path(a.output)
    require(not (output/'stage0_summary.json').exists(), f'Đã có kết quả Giai đoạn 0 trong {output}; không ghi đè')
    output.mkdir(parents=True, exist_ok=True)
    images = read_csv(a.subset_images)
    tasks = [dict(id=r['source_image_id'], condition='clean', path=Path(a.clean_root)/r['clean_relpath'],
                  sha=r['clean_sha256']) for r in images]
    tasks += [dict(id=r['source_image_id'], condition='lr2', path=Path(a.lr2_root)/r['lr2_name'], sha=r['lr2_sha256'])
              for r in images]
    manifest_path = Path(a.weights_dir)/'merge_manifest.json'
    manifest = read_json(manifest_path)
    identity = dict(stage=STAGE, monkey_v2=monkey, layers=list(BN_LAYERS), subset_images_sha256=sha256(a.subset_images),
                    train_check=read_json(a.train_check) if a.train_check else None,
                    merge_manifest_sha256=sha256(manifest_path), model_config_sha256=sha256(a.model_config),
                    gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
                    cudnn=torch.backends.cudnn.version(), mode='eval, forward pre-hooks only, no buffer writes')
    identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
    id_path = output/'stage0_identity.json'
    if id_path.is_file():
        require(read_json(id_path)['identity_sha256'] == identity_sha,
                'Thư mục này thuộc lần chạy khác (code/GPU/subset/weights đổi). Dùng thư mục mới.')
    else:
        save_json(id_path, dict(identity, identity_sha256=identity_sha))

    per_model, profile = {}, {}
    for model in manifest['models']:
        alpha = str(model['alpha'])
        tag = alpha.replace('/', '_')
        record = output/f'stage0_alpha_{tag}.json'
        if record.is_file():
            saved = read_json(record)
            require(saved['identity_sha256'] == identity_sha, f'{record} thuộc lần chạy khác')
            per_model[alpha], profile[alpha] = saved['metrics'], saved['profile']
            say(f'alpha {alpha}: đã xong trước đó, dùng lại')
            continue
        checkpoint = Path(a.weights_dir)/Path(model['checkpoint']).name
        require(sha256(checkpoint) == model['checkpoint_sha256'], f'{checkpoint} khác merge manifest')
        detector, _, _, _ = detector_for(argparse.Namespace(model_config=Path(a.model_config), checkpoint=checkpoint))
        net = detector.model
        require(not net.training, 'Model phải ở eval()')
        before = state_digest(net)
        modules = dict(net.named_modules())
        channels = {n: modules[n].num_features for n in BN_LAYERS}
        eps = {n: float(modules[n].eps) for n in BN_LAYERS}
        stored = {n: (modules[n].running_mean.detach().double().cpu().numpy(),
                      modules[n].running_var.detach().double().cpu().numpy()) for n in BN_LAYERS}
        acc = Accumulator(channels)
        acc_path, done_path = output/f'.alpha_{tag}.acc.npz', output/f'.alpha_{tag}.done.json'
        done = set()
        if done_path.is_file():
            meta = read_json(done_path)
            require(meta['identity_sha256'] == identity_sha, 'Tiến độ dở dang thuộc lần chạy khác')
            acc.load(acc_path)
            done = set(meta['done'])
        condition = ['clean']
        handles = attach_hooks(net, acc, condition)
        seconds, started = [], time.perf_counter()
        try:
            for number, task in enumerate(tasks, 1):
                key = f'{task["condition"]}::{task["id"]}'
                if key in done:
                    continue
                require(sha256(task['path']) == task['sha'], f'Ảnh sai SHA-256: {task["path"]}')
                tic = time.perf_counter()
                condition[0] = task['condition']
                _, inputs = next(iter(detector.preprocess([str(task['path'])], batch_size=1)))
                with torch.inference_mode():
                    processed = net.data_preprocessor(inputs, training=False)
                    net(**processed, mode='tensor')             # same call as rev4.stage_a.forward
                done.add(key)
                seconds.append(time.perf_counter() - tic)
                if len(done) % a.chunk == 0:
                    acc.save(acc_path)
                    save_json(done_path, dict(identity_sha256=identity_sha, done=sorted(done)))
                    say(f'alpha {alpha}: {len(done):,}/{len(tasks):,} ảnh, {np.mean(seconds):.3f} s/ảnh')
        finally:
            for h in handles:
                h.remove()
        require(len(done) == len(tasks), 'Chưa đủ ảnh')
        require(state_digest(net) == before, 'Trọng số hoặc buffer đã bị thay đổi trong Giai đoạn 0')
        metrics = {}
        for n in BN_LAYERS:
            metrics[n] = {}
            for label, conds in (('all', CONDITIONS), ('clean', ('clean',)), ('lr2', ('lr2',))):
                mean, var, count = acc.stats(n, conds)
                metrics[n][label] = dict(images=count, **shift_metrics(mean, var, *stored[n], eps=eps[n]))
            mean, var, _ = acc.stats(n)
            metrics[n]['measured_mean'] = mean.tolist()
            metrics[n]['measured_var'] = var.tolist()
            metrics[n]['stored_mean'] = stored[n][0].tolist()
            metrics[n]['stored_var'] = stored[n][1].tolist()
        profile[alpha] = dict(images=len(seconds), mean_seconds_per_image=float(np.mean(seconds)) if seconds else None,
                              wall_seconds=time.perf_counter() - started)
        per_model[alpha] = metrics
        save_json(record, dict(identity_sha256=identity_sha, metrics=metrics, profile=profile[alpha],
                               state_digest_unchanged=before))
        acc_path.unlink(missing_ok=True)
        done_path.unlink(missing_ok=True)
        del detector, net
        torch.cuda.empty_cache()

    save_json(output/'stage0_summary.json', summarize(per_model, profile, identity_sha))
    print(json.dumps(read_json(output/'stage0_summary.json')['gate'], indent=1))


def summarize(per_model, profile, identity_sha):
    import numpy as np
    from fractions import Fraction

    table = {a: {n: {k: per_model[a][n][k] for k in ('all', 'clean', 'lr2')} for n in BN_LAYERS} for a in per_model}
    measured = {n: {Fraction(a): (np.asarray(per_model[a][n]['measured_mean']), np.asarray(per_model[a][n]['measured_var']))
                    for a in per_model} for n in BN_LAYERS}
    nonlin = {n: {a: nonlinearity(measured[n], Fraction(a)) for a in per_model if Fraction(a) not in (0, 1)}
              for n in BN_LAYERS}
    return dict(stage=STAGE, label='EXPLORATORY; Stage 1 needs agreed criteria; no TEST data used',
                identity_sha256=identity_sha, stored_vs_measured=table, nonlinearity_vs_endpoint_interp=nonlin,
                gate=stage0_gate(per_model), profile=profile,
                note=('binarize.4 input depends on binarize.1 normalisation with its STORED stats (eval mode); '
                      'only binarize.1 is a clean measurement. Gate uses binarize.1.'))


def main():
    sys.stdout.reconfigure(errors='backslashreplace')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('check-train')
    p.add_argument('--train-input', required=True, type=Path)
    p.add_argument('--bundle', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p = sub.add_parser('prepare-images')
    p.add_argument('--subset', required=True, type=Path)
    p.add_argument('--clean-root', required=True, type=Path)
    p.add_argument('--lr2-root', required=True, type=Path)
    p.add_argument('--record', required=True, type=Path)
    p = sub.add_parser('stage0')
    p.add_argument('--subset-images', required=True, type=Path)
    p.add_argument('--train-check', type=Path)
    p.add_argument('--clean-root', required=True, type=Path)
    p.add_argument('--lr2-root', required=True, type=Path)
    p.add_argument('--model-config', required=True, type=Path)
    p.add_argument('--weights-dir', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--chunk', type=int, default=100)
    a = parser.parse_args()
    dict(**{'check-train': check_train, 'prepare-images': prepare_images, 'stage0': stage0})[a.command](a)


if __name__ == '__main__':
    main()
