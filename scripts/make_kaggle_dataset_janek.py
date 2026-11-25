"""Build the ONE folder (and zip) the user uploads as a Kaggle Dataset for the CS-B1 latency notebook.

runtime/kaggle_upload/cs431-csb1-latency/
  KAGGLE_MANIFEST.json   SHA-256 + size + purpose of every file (the notebook checks all of them)
  code/                  mixab/, tests/, configs/, docs/handoff/artifacts/janek/diag_v2_cpu/p0_fixed_janek.json + CODE_MANIFEST.json
  weights/               merge_manifest.json + alpha_0p00.pth (B-Clean) + alpha_1p00.pth (B1): the SAME endpoint
                         copies the T4 dump used (SHA in merge_manifest)
  encoder/               MonkeyOCRv2-AS encoder snapshot used in all earlier runs (revision in _revision.txt)
  clean/train_images/    only the Clean images of the fixed latency subset (n + warm-up), exact bytes from the cache zip
  dump/                  the T4 dump (Colab) for the box-count cross-check
  wheels/                MMCV 2.0.1 wheel built earlier (cp313; used only if Kaggle runs Python 3.13)
runtime/kaggle_upload/cs431-csb1-latency.zip   the same content, one file to upload

Run from the workspace root:  python scripts/make_kaggle_dataset_janek.py [--drive "H:/My Drive/CS431_MixA"]
"""
import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mixab import csb1_latency_janek as lat # noqa: E402
from mixab.monkey_janek import monkey_root  # noqa: E402

NAME = 'cs431-csb1-latency'
CODE_INCLUDE = ['mixab', 'tests', 'configs', 'docs/handoff/artifacts/janek/diag_v2_cpu/p0_fixed_janek.json']


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def code_files():
    for item in CODE_INCLUDE:
        path = ROOT/item
        for p in ([path] if path.is_file() else sorted(path.rglob('*'))):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.tmp'):
                yield p


def main():
    sys.stdout.reconfigure(errors='backslashreplace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--drive', default='H:/My Drive/CS431_MixA', type=Path)
    ap.add_argument('--dump', default=ROOT/'runtime/phase0_inputs/dualagree_dump_v1_T4', type=Path)
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--warmup', type=int, default=20)
    a = ap.parse_args()
    out = ROOT/'runtime/kaggle_upload'/NAME
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    purpose = {}
    # code
    manifest = {}
    for p in code_files():
        rel = p.relative_to(ROOT).as_posix()
        (out/'code'/rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out/'code'/rel)
        manifest[rel] = sha(p)
    (out/'code/CODE_MANIFEST.json').write_text(json.dumps(dict(files=manifest), indent=1, sort_keys=True), encoding='utf-8')
    purpose['code/'] = 'mixab code (csb1_latency, dualagree, merge_audit, colab_helpers) + tests'
    # weights: only the two endpoint copies used by the dump
    wsrc = a.drive/'weights/merge_weights_v1'
    m = json.loads((wsrc/'merge_manifest.json').read_text(encoding='utf-8'))
    (out/'weights').mkdir()
    shutil.copy2(wsrc/'merge_manifest.json', out/'weights/merge_manifest.json')
    for model in m['models']:
        if model['alpha'] in ('0', '1'):
            name = Path(model['checkpoint']).name
            shutil.copy2(wsrc/name, out/'weights'/name)
            assert sha(out/'weights'/name) == model['checkpoint_sha256'], name
    purpose['weights/'] = 'alpha_0p00 = B-Clean, alpha_1p00 = B1 (merge endpoint copies, SHA in merge_manifest.json)'
    # encoder
    esrc = a.drive/'cache/encoder_monkeyocrv2_as'
    for p in sorted(esrc.iterdir()):
        if p.is_file() and p.name != 'desktop.ini':
            (out/'encoder').mkdir(exist_ok=True)
            shutil.copy2(p, out/'encoder'/p.name)
    purpose['encoder/'] = 'MonkeyOCRv2-AS encoder (same snapshot as the A2r/dump runs)'
    # Clean images of the latency subset
    bundle = monkey_root()/'docs/handoff/artifacts/pr_quad_audit_v1'
    items = lat.needed_clean(bundle, a.n, a.warmup)
    with zipfile.ZipFile(a.drive/'cache/textvqa_val2262_clean.zip') as zf:
        for rel, digest in items:
            target = out/'clean'/rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(rel))
            assert sha(target) == digest, rel
    purpose['clean/'] = f'{len(items)} Clean images = fixed latency subset (seed {lat.SEED}, n={a.n}, warm-up={a.warmup})'
    # dump
    shutil.copytree(a.dump, out/'dump', ignore=shutil.ignore_patterns('desktop.ini'))
    purpose['dump/'] = 'T4 (Colab) dump for the live-vs-dump box-count check'
    # wheels (optional)
    for whl in sorted((a.drive/'cache/wheels').rglob('mmcv-*.whl')):
        (out/'wheels').mkdir(exist_ok=True)
        shutil.copy2(whl, out/'wheels'/whl.name)
    purpose['wheels/'] = 'MMCV 2.0.1 wheel(s) built earlier; used only if the Python tag matches'
    files = {p.relative_to(out).as_posix(): dict(sha256=sha(p), bytes=p.stat().st_size)
             for p in sorted(out.rglob('*')) if p.is_file()}
    (out/'KAGGLE_MANIFEST.json').write_text(json.dumps(dict(name=NAME, purpose=purpose, n=a.n, warmup=a.warmup,
                                                            files=files), indent=1), encoding='utf-8')
    zpath = out.with_suffix('.zip')
    with zipfile.ZipFile(zpath, 'w', compression=zipfile.ZIP_STORED) as zf:
        for p in sorted(out.rglob('*')):
            if p.is_file():
                zf.write(p, f'{NAME}/{p.relative_to(out).as_posix()}')
    total = sum(f['bytes'] for f in files.values())
    print(f'{out}\n{len(files)} files, {total/2**20:.0f} MiB\nzip: {zpath} ({zpath.stat().st_size/2**20:.0f} MiB) '
          f'SHA-256 {sha(zpath)}')


if __name__ == '__main__':
    main()
