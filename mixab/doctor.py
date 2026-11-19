"""Pre-flight check of a T4/A100 machine before spending GPU time on merge_audit / Mix B.

Every check reports PASS, WARN or FAIL; nothing is installed or downloaded. Exit code 1 if any
FAIL. Hard requirements (FAIL): CUDA usable on 1 device, Pillow 11.3.0, the pinned MMOCR
DBPostprocessor SHA-256, MonkeyOCRv2ViTAEBackbone registered, model config + encoder files
present, bundle GT manifest intact, image roots readable with matching SHA-256 on a sample.
Version differences against Tuan's verified environment (docs/handoff/checkpoints.md) are WARN:
they may still work but can change numerics, which matters for the exact reproduction gates.

Usage:
  python -m mixab.doctor --bundle B --clean-root C --prepared-root P --model-config CFG \
      [--weights-dir runtime/merge_weights_v1] [--output report.json]
"""
import argparse
import importlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

from mixab import save_json, sha256

# docs/handoff/checkpoints.md (MONKEY_V2 dc19b85): the verified environment.
EXPECTED = {'torch': '2.9.0', 'mmcv': '2.0.1', 'mmengine': '0.10.7', 'mmocr': '1.0.1', 'cv2': '4.10.0'}
PILLOW = '11.3.0'
SAMPLE_IMAGES = 20
MIN_FREE_GIB = 5


class Report:
    def __init__(self):
        self.rows = []

    def add(self, name, status, detail):
        self.rows.append(dict(check=name, status=status, detail=str(detail)))
        return status == 'PASS'

    def ok(self, name, condition, detail, fail='FAIL'):
        return self.add(name, 'PASS' if condition else fail, detail)

    @property
    def failed(self):
        return any(r['status'] == 'FAIL' for r in self.rows)

    def print(self):
        width = max(len(r['check']) for r in self.rows)
        for r in self.rows:
            print(f'{r["status"]:4}  {r["check"]:<{width}}  {r["detail"]}')
        counts = {s: sum(r['status'] == s for r in self.rows) for s in ('PASS', 'WARN', 'FAIL')}
        print(f'\n{counts["PASS"]} PASS, {counts["WARN"]} WARN, {counts["FAIL"]} FAIL')


def version_of(module):
    try:
        mod = importlib.import_module(module)
        return getattr(mod, '__version__', 'unknown')
    except Exception as exc:  # noqa: BLE001 - report any import failure
        return None if isinstance(exc, ImportError) else f'import error: {type(exc).__name__}: {exc}'


def check_python_stack(r):
    r.add('python', 'PASS', sys.version.split()[0])
    for module, expected in EXPECTED.items():
        found = version_of(module)
        if found is None:
            r.add(f'version:{module}', 'FAIL', f'not installed (verified env: {expected})')
        elif str(found).startswith('import error'):
            r.add(f'version:{module}', 'FAIL', found)
        else:
            r.ok(f'version:{module}', str(found).startswith(expected),
                 f'{found} (verified env: {expected})', fail='WARN')
    found = version_of('PIL')
    r.ok('version:Pillow', found == PILLOW, f'{found} (native LR2 requires exactly {PILLOW})')


def check_cuda(r):
    try:
        import torch
    except ImportError:
        return r.add('cuda', 'FAIL', 'torch not importable')
    if not torch.cuda.is_available():
        return r.add('cuda', 'FAIL', 'torch.cuda.is_available() is False')
    count = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    total = torch.cuda.get_device_properties(0).total_memory/2**30
    r.ok('cuda:devices', count == 1, f'{count} visible (stage_a.detector_for requires exactly 1; '
                                     f'set CUDA_VISIBLE_DEVICES)')
    arch = f'sm_{cap[0]}{cap[1]}'
    r.ok('cuda:arch', arch in torch.cuda.get_arch_list(), f'{name}, {arch}, {total:.1f} GiB; '
                                                          f'wheel archs {torch.cuda.get_arch_list()}')
    try:
        x = torch.randn(256, 256, device='cuda')
        conv = torch.nn.functional.conv2d(x[None, None], torch.ones(1, 1, 3, 3, device='cuda'))
        torch.cuda.synchronize()
        r.ok('cuda:kernel', bool(torch.isfinite(conv).all()), 'matmul/conv kernel ran')
    except Exception as exc:  # noqa: BLE001
        r.add('cuda:kernel', 'FAIL', f'{type(exc).__name__}: {exc}')
    tf32 = 'TF32 available' if cap[0] >= 8 else 'no TF32 (pre-Ampere): numerics differ from H200 reference'
    r.add('cuda:numerics', 'PASS' if cap[0] >= 8 else 'WARN', tf32)


def check_mmocr(r):
    try:
        from mixab.monkey import use_monkey
        use_monkey()
        from rev4.decision_plan import INSTALLED_PP_SHA256
        spec = importlib.util.find_spec('mmocr.models.textdet.postprocessors.db_postprocessor')
        if spec is None or not spec.origin:
            return r.add('mmocr:db_postprocessor', 'FAIL', 'MMOCR DBPostprocessor not found')
        digest = sha256(spec.origin)
        r.ok('mmocr:db_postprocessor', digest == INSTALLED_PP_SHA256,
             f'{digest[:12]}... (pinned {INSTALLED_PP_SHA256[:12]}...; reference bytes in '
             'bundle/provenance/installed_db_postprocessor.py)')
    except Exception as exc:  # noqa: BLE001
        return r.add('mmocr:db_postprocessor', 'FAIL', f'{type(exc).__name__}: {exc}')
    try:
        from mmocr.registry import MODELS
        from mmocr.utils import register_all_modules
        register_all_modules(init_default_scope=True)
        found = MODELS.get('MonkeyOCRv2ViTAEBackbone') is not None
        r.ok('mmocr:backbone_registered', found, 'MonkeyOCRv2ViTAEBackbone in MMOCR registry'
             if found else 'not registered: install the MonkeyOCRv2 detection code (config has no custom_imports)')
    except Exception as exc:  # noqa: BLE001
        r.add('mmocr:backbone_registered', 'FAIL', f'{type(exc).__name__}: {exc}')


def encoder_paths(config_text):
    """model_dir / weights strings of the backbone, read without executing the config."""
    import re

    out = {}
    for key in ('model_dir', 'weights'):
        m = re.search(rf"{key}=\s*'([^']+)'", config_text)
        out[key] = m.group(1) if m else None
    return out


def check_config(r, config):
    config = Path(config)
    if not r.ok('config:file', config.is_file(), config):
        return
    text = config.read_text(encoding='utf-8')
    r.ok('config:fp32', "textdet_parameter_precision = 'fp32'" in text and 'use_bf16=False' in text,
         'FP32 config, use_bf16=False (required by stage_a.detector_for)')
    paths = encoder_paths(text)
    for key, value in paths.items():
        if value is None:
            r.add(f'config:{key}', 'FAIL', 'not found in config')
            continue
        path = Path(value)
        resolved = path if path.is_absolute() else Path.cwd()/path
        r.ok(f'config:{key}', resolved.exists(),
             f'{value} -> {resolved} (relative paths resolve against the CWD at run time)')


def check_bundle_and_images(r, bundle, clean_root, prepared_root):
    from mixab import merge_audit as ma

    try:
        rows = ma.load_rows(bundle)
        r.add('bundle:gt_manifest', 'PASS', f'{len(rows)} sources, locked SHA-256 matches')
    except Exception as exc:  # noqa: BLE001
        return r.add('bundle:gt_manifest', 'FAIL', f'{type(exc).__name__}: {exc}')
    roots = dict(TextVQA_clean_root=clean_root, prepared_root=prepared_root)
    tasks = ma.resolve_tasks(ma.load_task_metadata(bundle), roots)
    for condition in ma.CONDITIONS:
        pool = [t for t in ma.pmap_tasks(tasks, per_condition=SAMPLE_IMAGES//2) if t['condition'] == condition]
        present = [t for t in pool if Path(t['path']).is_file()]
        matching = [t for t in present if sha256(t['path']) == t['image_sha256']]
        total = sum(Path(t['path']).is_file() for t in tasks if t['condition'] == condition)
        status = 'PASS' if len(matching) == len(pool) else ('WARN' if condition == 'lr2' and not present else 'FAIL')
        hint = ' (run: python -m mixab.merge_audit prepare-lr2 ...)' if condition == 'lr2' and status != 'PASS' else ''
        r.add(f'images:{condition}', status, f'sample {len(matching)}/{len(pool)} SHA ok; '
                                             f'{total}/{len(tasks)//2} files present{hint}')


def check_weights(r, weights_dir):
    from mixab import read_json

    manifest = Path(weights_dir)/'merge_manifest.json'
    if not manifest.is_file():
        return r.add('weights:merge_manifest', 'WARN', f'{manifest} missing (run mixab.weight_merge first)')
    models = read_json(manifest)['models']
    bad = [m['alpha'] for m in models
           if not (Path(weights_dir)/Path(m['checkpoint']).name).is_file()
           or sha256(Path(weights_dir)/Path(m['checkpoint']).name) != m['checkpoint_sha256']]
    r.ok('weights:merged', not bad, f'{len(models)} checkpoints' + (f'; bad alphas {bad}' if bad else ', SHA ok'))


def check_disk(r, output):
    folder = Path(output).resolve().parent
    folder.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(folder).free/2**30
    r.ok('disk:free', free >= MIN_FREE_GIB, f'{free:.1f} GiB free at {folder} (need >= {MIN_FREE_GIB})', fail='WARN')


def run(a):
    r = Report()
    check_python_stack(r)
    check_cuda(r)
    check_mmocr(r)
    check_config(r, a.model_config)
    check_bundle_and_images(r, a.bundle, a.clean_root, a.prepared_root)
    if a.weights_dir:
        check_weights(r, a.weights_dir)
    check_disk(r, a.output)
    r.print()
    save_json(a.output, dict(rows=r.rows, failed=r.failed))
    return r


def main():
    sys.stdout.reconfigure(errors='backslashreplace')   # never crash on non-ASCII paths (Windows cp1252)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--clean-root', required=True, type=Path)
    parser.add_argument('--prepared-root', required=True, type=Path)
    parser.add_argument('--model-config', required=True, type=Path)
    parser.add_argument('--weights-dir', type=Path)
    parser.add_argument('--output', type=Path, default=Path('runtime/doctor_report.json'))
    a = parser.parse_args()
    sys.exit(1 if run(a).failed else 0)


if __name__ == '__main__':
    main()
