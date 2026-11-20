"""Standard-library helpers for notebooks/colab_E1_A2_probe.ipynb (testable without GPU/MMOCR).

Rules enforced here: never overwrite a result (timestamped names, move-aside instead of delete),
verify every input by SHA-256, and stream subprocess output to a persistent log.
"""
import csv
import datetime as _dt
import hashlib
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# Tuan's inference config uses these two relative encoder paths (docs/handoff/inference_config.py).
CONFIG_MODEL_DIR = "'pretrained/monkeyocrv2_as'"
CONFIG_WEIGHTS = "'pretrained/monkeyocrv2_as/model.safetensors'"


class StepError(RuntimeError):
    """Error with a short Vietnamese explanation for the notebook user."""


def fail(message):
    raise StepError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def stamp():
    return _dt.datetime.now().strftime('%Y%m%d_%H%M%S')


def new_path(folder, stem, suffix):
    """A path that does not exist yet: <stem>_<timestamp><suffix>, with a counter if needed."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    base = f'{stem}_{stamp()}'
    candidate, n = folder/f'{base}{suffix}', 1
    while candidate.exists():
        candidate, n = folder/f'{base}_{n}{suffix}', n + 1
    return candidate


def move_aside(path):
    """Rename an incomplete artifact instead of deleting it; returns the new path or None."""
    path = Path(path)
    if not path.exists():
        return None
    target = path.with_name(f'{path.name}_incomplete_{stamp()}')
    path.rename(target)
    return target


def gpu_label(name):
    """'Tesla T4' -> 'T4'; 'NVIDIA A100-SXM4-40GB' -> 'A100-SXM4-40GB'; others sanitized."""
    if 'T4' in name.split():
        return 'T4'
    match = re.search(r'A100[\w-]*', name)
    if match:
        return match.group(0)
    return re.sub(r'[^A-Za-z0-9-]+', '_', name).strip('_') or 'unknown_gpu'


def gpu_supported(name):
    return gpu_label(name) == 'T4' or gpu_label(name).startswith('A100')


def run(cmd, log_path, cwd=None, env=None, title=None):
    """Run a command, stream output to the cell and append it to log_path; raise on failure."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    shown = title or ' '.join(str(c) for c in cmd)
    full_env = dict(os.environ, **(env or {}))
    with log_path.open('a', encoding='utf-8') as log:
        log.write(f'\n===== {stamp()} $ {shown}\n')
        tail = []
        with subprocess.Popen([str(c) for c in cmd], cwd=cwd, env=full_env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace') as process:
            for line in process.stdout:
                sys.stdout.write(line)
                log.write(line)
                tail = (tail + [line])[-30:]
            code = process.wait()
        log.write(f'===== exit {code}\n')
    if code != 0:
        fail(f'Lệnh thất bại (exit {code}): {shown}\nXem log: {log_path}\n--- 30 dòng cuối ---\n' + ''.join(tail))
    return code


def output_of(cmd, cwd=None, env=None):
    """stdout of a command or None if it fails (used for version probes)."""
    try:
        done = subprocess.run([str(c) for c in cmd], cwd=cwd, env=dict(os.environ, **(env or {})),
                              capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def package_version(name, python=sys.executable):
    return output_of([python, '-c', f'import importlib.metadata as m; print(m.version({name!r}))'])


def find_cuda_toolkit(candidates, version='12.8'):
    """Check each actual compiler, rather than trusting PATH or a folder name."""
    for home in candidates:
        if not home:
            continue
        home = Path(home)
        compiler = home/'bin/nvcc'
        if not compiler.is_file():
            continue
        output = output_of([compiler, '--version']) or ''
        match = re.search(r'release (\d+\.\d+)(?:,|\s)', output)
        if match and match.group(1) == version and (home/'include/cuda.h').is_file():
            return str(home.resolve())
    return None


def patch_inference_config(text, encoder_dir):
    """Point the two relative encoder paths at an absolute encoder directory. Nothing else changes."""
    encoder_dir = str(encoder_dir).replace('\\', '/')
    if text.count(CONFIG_WEIGHTS) != 1 or text.count(CONFIG_MODEL_DIR) != 1:
        fail('inference_config.py không còn đúng dạng đã kiểm (2 đường dẫn encoder tương đối). '
             'Có thể Tuấn đã đổi config: dừng lại và đối chiếu.')
    text = text.replace(CONFIG_WEIGHTS, repr(f'{encoder_dir}/model.safetensors'))
    return text.replace(CONFIG_MODEL_DIR, repr(encoder_dir))


def needed_clean_images(bundle):
    """[(clean_relpath, clean_sha256)] for the 2,262 remaining-validation sources."""
    with (Path(bundle)/'data/source_cohort.csv').open(encoding='utf-8', newline='') as f:
        rows = [(r['clean_relpath'], r['clean_sha256']) for r in csv.DictReader(f)]
    if len(rows) != 2262 or len({r for r, _ in rows}) != 2262:
        fail(f'source_cohort.csv có {len(rows)} dòng, kỳ vọng đúng 2.262 nguồn khác nhau')
    return rows


def verify_images(root, items):
    """Return dict(ok, missing, bad) counts plus up to 5 examples of each problem."""
    root = Path(root)
    missing, bad = [], []
    for relpath, digest in items:
        path = root/relpath
        if not path.is_file():
            missing.append(relpath)
        elif sha256(path) != digest:
            bad.append(relpath)
    return dict(ok=len(items) - len(missing) - len(bad), missing=len(missing), bad=len(bad),
                examples_missing=missing[:5], examples_bad=bad[:5])


def extract_members(zip_path, relpaths, dest):
    """Extract only the listed members (exact names) from a zip; fail if any is absent."""
    dest = Path(dest)
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        absent = [r for r in relpaths if r not in names]
        if absent:
            fail(f'{len(absent)} ảnh không có trong {Path(zip_path).name}, ví dụ {absent[:3]}')
        for relpath in relpaths:
            target = dest/relpath
            if target.is_file():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(relpath) as src, open(f'{target}.part', 'wb') as out:
                shutil.copyfileobj(src, out)
            os.replace(f'{target}.part', target)
    return len(relpaths)


def build_cache_zip(root, relpaths, out_zip):
    """Store-only zip of the verified subset; written to a temp name and renamed when complete."""
    out_zip = Path(out_zip)
    if out_zip.exists():
        fail(f'{out_zip} đã tồn tại; không ghi đè')
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    temp = out_zip.with_name(out_zip.name + '.part')
    with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_STORED) as zf:
        for relpath in relpaths:
            zf.write(Path(root)/relpath, relpath)
    os.replace(temp, out_zip)
    return out_zip


def failed_checks(report):
    return [r for r in report.get('rows', []) if r.get('status') == 'FAIL']


def summarize_probe(gates, profile, full_sources=2262, models=5):
    """Plain dict for display: gate status per endpoint and a rough full-run time estimate."""
    rows = []
    for name in ('B-Clean', 'B1'):
        g = gates.get('models', {}).get(name, {})
        rows.append(dict(model=name, gate_3a=g.get('gate_3a', {}).get('status', 'NOT_RUN'),
                         mismatched_3a=g.get('gate_3a', {}).get('mismatched'),
                         checked_3a=g.get('gate_3a', {}).get('checked'),
                         pmap=g.get('pmap_endpoint', {}).get('status', 'NOT_RUN'),
                         pmap_max_abs_diff=g.get('pmap_endpoint', {}).get('max_abs_diff')))
    seconds = [p['mean_seconds_per_task'] for p in profile.values() if p.get('tasks')]
    per_task = sum(seconds)/len(seconds) if seconds else None
    estimate_h = None if per_task is None else per_task*2*full_sources*models/3600
    peak = max((p.get('peak_gpu_mib', 0) for p in profile.values()), default=None)
    ready = all(r['gate_3a'] == 'PASS' and r['pmap'] == 'PASS' for r in rows)
    # Plan 4.3 option (a): endpoints are re-run on the same GPU, so only the P-map check gates the full run.
    ready_same_gpu = all(r['pmap'] == 'PASS' for r in rows)
    return dict(endpoints=rows, mean_seconds_per_task=per_task, full_run_estimate_hours=estimate_h,
                peak_gpu_mib=peak, ready_for_full_run=ready, ready_for_full_run_same_gpu=ready_same_gpu,
                gpu=gates.get('gpu'))
