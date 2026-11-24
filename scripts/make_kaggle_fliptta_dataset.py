"""Build the Kaggle Dataset for the Flip-TTA pilot (docs/fliptta_spec_20261123.md). File copying only.

runtime/kaggle_upload/cs431-fliptta-pilot/ (+ .zip):
  KAGGLE_MANIFEST.json   SHA-256/size of every file + the locked sample list
  code/                  mixab + tests (same selection as the other datasets)
  weights/               merge_manifest.json + alpha_1p00.pth (B1 endpoint copy used by the T4 dump)
  encoder/               MonkeyOCRv2-AS encoder snapshot
  clean/train_images/    Clean images of the 300 sampled split-A sources
  dump/                  T4 (Colab) dump: consistency of F on LR2
  a2r/B1_source_counts.npz   A100 counts: consistency of F on Clean
  ref/zoom_sample.json   sources of the zoom pilot (overlap, descriptive only)
  wheels/                MMCV 2.0.1 wheel(s) (used only if the Python tag matches)
Run from the workspace root:  python scripts/make_kaggle_fliptta_dataset.py
"""
import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_kaggle_dataset import code_files, sha  # noqa: E402
from mixab import fliptta, zoom  # noqa: E402
from mixab import merge_audit as ma  # noqa: E402
from mixab.monkey import monkey_root  # noqa: E402

NAME = 'cs431-fliptta-pilot'


def main():
    sys.stdout.reconfigure(errors='backslashreplace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--drive', default='H:/My Drive/CS431_MixA', type=Path)
    ap.add_argument('--dump', default=ROOT/'runtime/phase0_inputs/dualagree_dump_v1_T4', type=Path)
    ap.add_argument('--a2r', default=ROOT/'runtime/phase0_inputs/a2r/B1_source_counts.npz', type=Path)
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
    wsrc = a.drive/'weights/merge_weights_v1'
    m = json.loads((wsrc/'merge_manifest.json').read_text(encoding='utf-8'))
    (out/'weights').mkdir()
    shutil.copy2(wsrc/'merge_manifest.json', out/'weights/merge_manifest.json')
    b1 = next(x for x in m['models'] if x['alpha'] == '1')
    name = Path(b1['checkpoint'].replace('\\', '/')).name
    shutil.copy2(wsrc/name, out/'weights'/name)
    assert sha(out/'weights'/name) == b1['checkpoint_sha256']
    (out/'encoder').mkdir()
    for p in sorted((a.drive/'cache/encoder_monkeyocrv2_as').iterdir()):
        if p.is_file() and p.name != 'desktop.ini':
            shutil.copy2(p, out/'encoder'/p.name)
    rows = ma.load_rows(monkey_root()/'docs/handoff/artifacts/pr_quad_audit_v1')
    sources = fliptta.select_sources(rows)
    with zipfile.ZipFile(a.drive/'cache/textvqa_val2262_clean.zip') as zf:
        for s in sources:
            rel = rows[s]['clean_relpath']
            (out/'clean'/rel).parent.mkdir(parents=True, exist_ok=True)
            (out/'clean'/rel).write_bytes(zf.read(rel))
            assert sha(out/'clean'/rel) == rows[s]['clean_sha256'], rel
    shutil.copytree(a.dump, out/'dump', ignore=shutil.ignore_patterns('desktop.ini'))
    (out/'a2r').mkdir()
    shutil.copy2(a.a2r, out/'a2r/B1_source_counts.npz')
    (out/'ref').mkdir()
    (out/'ref/zoom_sample.json').write_text(json.dumps(dict(seed_key=zoom.SEED_KEY, sources=zoom.select_sources(rows))), encoding='utf-8')
    for whl in sorted((a.drive/'cache/wheels').rglob('mmcv-*.whl')):
        (out/'wheels').mkdir(exist_ok=True)
        shutil.copy2(whl, out/'wheels'/whl.name)
    files = {p.relative_to(out).as_posix(): dict(sha256=sha(p), bytes=p.stat().st_size)
             for p in sorted(out.rglob('*')) if p.is_file()}
    (out/'KAGGLE_MANIFEST.json').write_text(json.dumps(dict(
        name=NAME, kind='fliptta_pilot', seed_key=fliptta.SEED_KEY, n_sources=len(sources), sources=sources,
        overlap_with_zoom_sample=len(set(sources) & set(zoom.select_sources(rows))), files=files), indent=1), encoding='utf-8')
    zpath = out.with_suffix('.zip')
    with zipfile.ZipFile(zpath, 'w', compression=zipfile.ZIP_STORED) as zf:
        for p in sorted(out.rglob('*')):
            if p.is_file():
                zf.write(p, f'{NAME}/{p.relative_to(out).as_posix()}')
    print(f'{out}\n{len(files)} files, {sum(f["bytes"] for f in files.values())/2**20:.0f} MiB\n'
          f'zip: {zpath} ({zpath.stat().st_size/2**20:.0f} MiB) SHA-256 {sha(zpath)}')


if __name__ == '__main__':
    main()
