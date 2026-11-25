"""Build the Kaggle Dataset for the box-quality scorer (docs/research/janek/boxquality_spec_20261124_janek.md). File copying only.

runtime/kaggle_upload/cs431-boxquality/ (+ .zip):
  KAGGLE_MANIFEST.json   SHA-256/size of every file
  code/                  mixab (incl. vendor/shared_manifest.py) + tests + CODE_MANIFEST.json
  weights/               merge_manifest.json + alpha_1p00.pth (B1 endpoint, SHA in manifest)
  encoder/               MonkeyOCRv2-AS encoder snapshot
  wheels/                MMCV 2.0.1 wheel(s)
  inputs/                train_rows.jsonl.gz, selection.csv, p0.json, train_ids.csv (not in the pinned MONKEY_V2 commit),
                         devcheck_sources.json
  dev_clean/             Clean images of the 5 devcheck sources (pipeline check only; no development labels are used)
  dump/                  T4 Colab dump, B1 only (+ identity): devcheck reference
Images of TRAIN are NOT shipped: the notebook fetches them from the official TextVQA zip (per member, CRC + SHA-256).
Weights/encoder/wheels/dump/dev images are copied from the hash-verified Flip-TTA staging folder.
Run from the workspace root:  python scripts/make_kaggle_boxquality_dataset_janek.py
"""
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_kaggle_dataset_janek import code_files, sha  # noqa: E402
from mixab import merge_audit_janek as ma # noqa: E402
from mixab.monkey_janek import monkey_root  # noqa: E402

NAME = 'cs431-boxquality'
STAGE = ROOT/'runtime/kaggle_upload/cs431-fliptta-pilot'
P0 = ROOT/'docs/handoff/artifacts/janek/boxquality_p0_20261124'


def main():
    sys.stdout.reconfigure(errors='backslashreplace')
    stage_manifest = json.loads((STAGE/'KAGGLE_MANIFEST.json').read_text(encoding='utf-8'))['files']
    out = ROOT/'runtime/kaggle_upload'/NAME
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    def from_stage(rel, dest_rel=None):
        src = STAGE/rel
        assert sha(src) == stage_manifest[rel]['sha256'], f'staging file changed: {rel}'
        dst = out/(dest_rel or rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    manifest = {}
    for p in code_files():
        rel = p.relative_to(ROOT).as_posix()
        (out/'code'/rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out/'code'/rel)
        manifest[rel] = sha(p)
    assert 'mixab/vendor/shared_manifest_janek.py' in manifest and 'mixab/boxquality_janek.py' in manifest
    (out/'code/CODE_MANIFEST.json').write_text(json.dumps(dict(files=manifest), indent=1, sort_keys=True), encoding='utf-8')
    for rel in stage_manifest:
        if rel.startswith(('weights/', 'encoder/', 'wheels/')):
            from_stage(rel)
        elif rel.startswith('dump/') and not rel.startswith('dump/B-Clean/'):
            from_stage(rel)
    rows = ma.load_rows(monkey_root()/'docs/handoff/artifacts/pr_quad_audit_v1')
    dev = json.loads((P0/'devcheck_sources_janek.json').read_text(encoding='utf-8'))['sources']
    for s in dev:
        rel = rows[s]['clean_relpath']
        from_stage(f'clean/{rel}', f'dev_clean/{rel}')
    (out/'inputs').mkdir()
    for name in ('train_rows.jsonl.gz', 'selection.csv', 'p0.json', 'devcheck_sources.json'):
        published = Path(name).stem + '_janek' + Path(name).suffix
        shutil.copy2(P0/published, out/'inputs'/name)
    shutil.copy2(monkey_root()/'docs/handoff/data/train_ids.csv', out/'inputs/train_ids.csv')
    p0 = json.loads((P0/'p0_janek.json').read_text(encoding='utf-8'))
    assert sha(out/'inputs/train_rows.jsonl.gz') == p0['train_rows_sha256']
    files = {p.relative_to(out).as_posix(): dict(sha256=sha(p), bytes=p.stat().st_size)
             for p in sorted(out.rglob('*')) if p.is_file()}
    (out/'KAGGLE_MANIFEST.json').write_text(json.dumps(dict(
        name=NAME, kind='boxquality', spec='docs/research/janek/boxquality_spec_20261124_janek.md', selection_sha256=p0['selection']['selection_sha256'],
        train_rows_sha256=p0['train_rows_sha256'], devcheck_sources=dev, files=files), indent=1), encoding='utf-8')
    zpath = out.with_suffix('.zip')
    with zipfile.ZipFile(zpath, 'w', compression=zipfile.ZIP_STORED) as zf:
        for p in sorted(out.rglob('*')):
            if p.is_file():
                zf.write(p, f'{NAME}/{p.relative_to(out).as_posix()}')
    print(f'{out}\n{len(files)} files, {sum(f["bytes"] for f in files.values())/2**20:.0f} MiB\n'
          f'zip: {zpath} ({zpath.stat().st_size/2**20:.0f} MiB) SHA-256 {sha(zpath)}\n'
          f'code manifest SHA-256 {sha(out/"code/CODE_MANIFEST.json")}')


if __name__ == '__main__':
    main()
