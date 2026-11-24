"""Kaggle Dataset for the CS-B1 Clean check (docs/csb1_clean_check_spec_20261123.md). File copying only.

runtime/kaggle_upload/cs431-csb1-clean/ (+ .zip): code/, weights/ (alpha_0p00 + alpha_1p00 + merge_manifest),
encoder/, dump/ (existing LR2 T4 dump),
a2r/ (B1 + B-Clean A100 counts, for the fidelity probe), wheels/, KAGGLE_MANIFEST.json (kind csb1_clean).
clean/train_images/: the 2,262 Clean images copied byte-exact from the Drive cache zip (SHA-checked) when the zip is
available; otherwise they are omitted and the notebook downloads the official TextVQA zip instead.
Inputs default to local copies (Drive not needed): runtime/merge_weights_v1 and the latency dataset folder.
Run from the workspace root:  python scripts/make_kaggle_csb1_clean_dataset.py
"""
import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_kaggle_dataset import code_files, sha  # noqa: E402
from mixab import merge_audit as ma  # noqa: E402
from mixab.monkey import monkey_root  # noqa: E402

NAME = 'cs431-csb1-clean'


def main():
    sys.stdout.reconfigure(errors='backslashreplace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights-dir', default=ROOT/'runtime/merge_weights_v1', type=Path)
    ap.add_argument('--encoder', default=ROOT/'runtime/kaggle_upload/cs431-csb1-latency/encoder', type=Path)
    ap.add_argument('--wheels', default=ROOT/'runtime/kaggle_upload/cs431-csb1-latency/wheels', type=Path)
    ap.add_argument('--clean-zip', default=Path('H:/My Drive/CS431_MixA/cache/textvqa_val2262_clean.zip'), type=Path)
    a = ap.parse_args()
    out = ROOT/'runtime/kaggle_upload'/NAME
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    manifest = {}
    for p in code_files():
        rel = p.relative_to(ROOT).as_posix()
        (out/'code'/rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out/'code'/rel)
        manifest[rel] = sha(p)
    (out/'code/CODE_MANIFEST.json').write_text(json.dumps(dict(files=manifest), indent=1, sort_keys=True), encoding='utf-8')
    wsrc = a.weights_dir
    m = json.loads((wsrc/'merge_manifest.json').read_text(encoding='utf-8'))
    (out/'weights').mkdir()
    shutil.copy2(wsrc/'merge_manifest.json', out/'weights/merge_manifest.json')
    for x in m['models']:
        if x['alpha'] in ('0', '1'):
            name = PureWindowsPath(x['checkpoint']).name
            shutil.copy2(wsrc/name, out/'weights'/name)
            assert sha(out/'weights'/name) == x['checkpoint_sha256'], name
    (out/'encoder').mkdir()
    for p in sorted(Path(a.encoder).iterdir()):
        if p.is_file() and p.name != 'desktop.ini':
            shutil.copy2(p, out/'encoder'/p.name)
    rows = ma.load_rows(monkey_root()/'docs/handoff/artifacts/pr_quad_audit_v1')
    with_images = Path(a.clean_zip).is_file()
    if with_images:
        with zipfile.ZipFile(a.clean_zip) as zf:
            for s in sorted(rows):
                rel = rows[s]['clean_relpath']
                (out/'clean'/rel).parent.mkdir(parents=True, exist_ok=True)
                (out/'clean'/rel).write_bytes(zf.read(rel))
                assert sha(out/'clean'/rel) == rows[s]['clean_sha256'], rel
    else:
        print(f'WARNING: {a.clean_zip} not found -> no Clean images in the dataset (notebook will download them)')
    shutil.copytree(ROOT/'runtime/phase0_inputs/dualagree_dump_v1_T4', out/'dump', ignore=shutil.ignore_patterns('desktop.ini'))
    (out/'a2r').mkdir()
    for name in ('B1', 'B-Clean'):
        shutil.copy2(ROOT/f'runtime/phase0_inputs/a2r/{name}_source_counts.npz', out/'a2r'/f'{name}_source_counts.npz')
    for whl in sorted(Path(a.wheels).rglob('mmcv-*.whl')):
        (out/'wheels').mkdir(exist_ok=True)
        shutil.copy2(whl, out/'wheels'/whl.name)
    files = {p.relative_to(out).as_posix(): dict(sha256=sha(p), bytes=p.stat().st_size)
             for p in sorted(out.rglob('*')) if p.is_file()}
    (out/'KAGGLE_MANIFEST.json').write_text(json.dumps(dict(name=NAME, kind='csb1_clean', sources=len(rows), clean_images_included=with_images, files=files), indent=1),
                                            encoding='utf-8')
    zpath = out.with_suffix('.zip')
    with zipfile.ZipFile(zpath, 'w', compression=zipfile.ZIP_STORED) as zf:
        for p in sorted(out.rglob('*')):
            if p.is_file():
                zf.write(p, f'{NAME}/{p.relative_to(out).as_posix()}')
    print(f'{out}\n{len(files)} files, {sum(f["bytes"] for f in files.values())/2**20:.0f} MiB\n'
          f'zip: {zpath} ({zpath.stat().st_size/2**20:.0f} MiB) SHA-256 {sha(zpath)}')


if __name__ == '__main__':
    main()
