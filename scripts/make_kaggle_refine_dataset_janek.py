"""Small Kaggle Dataset for learned box refinement (docs/research/janek/refine_spec_20261125_janek.md).
It is used TOGETHER with the existing Kaggle dataset cs431-boxquality-devA (B1, encoder, wheels, frozen scorer,
1,130 A Clean images), which is attached unchanged and checked by its own manifest.

runtime/kaggle_upload/cs431-refine/ (+ .zip):
  KAGGLE_MANIFEST.json      kind refine; SHA-256/size of every file; SHA of the devA dataset manifest it expects
  code/                     mixab + tests + configs + CODE_MANIFEST.json
  inputs/                   TRAIN labels (train_rows as .gzip.bin, Kaggle unpacks .gz), selection.csv, train_ids.csv
  ref/devA_result_janek.json  box-quality dev A result: F / v1 counts must be reproduced on A
Run from the workspace root:  python scripts/make_kaggle_refine_dataset_janek.py
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

NAME = 'cs431-refine'
STAGE_BQ = ROOT/'runtime/kaggle_upload/cs431-boxquality'
DEVA = ROOT/'runtime/kaggle_upload/cs431-boxquality-devA'
REF = ROOT/'docs/handoff/artifacts/janek/boxquality_devA_20261125/devA_result_janek.json'


def main():
    sys.stdout.reconfigure(errors='backslashreplace')
    out = ROOT/'runtime/kaggle_upload'/NAME
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    code = {}
    for p in code_files():
        rel = p.relative_to(ROOT).as_posix()
        (out/'code'/rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out/'code'/rel)
        code[rel] = sha(p)
    for need in ('mixab/refine_janek.py', 'mixab/refine_cmd_janek.py', 'mixab/refine_geom_janek.py', 'mixab/boxquality_devA_janek.py'):
        assert need in code, need
    (out/'code/CODE_MANIFEST.json').write_text(json.dumps(dict(files=code), indent=1, sort_keys=True), encoding='utf-8')
    bq_files = json.loads((STAGE_BQ/'KAGGLE_MANIFEST.json').read_text(encoding='utf-8'))
    (out/'inputs').mkdir()
    for rel, dst in (('inputs/selection.csv', 'selection.csv'), ('inputs/train_ids.csv', 'train_ids.csv'),
                     ('inputs/train_rows.jsonl.gz', 'train_rows.gzip.bin')):
        assert sha(STAGE_BQ/rel) == bq_files['files'][rel]['sha256'], rel
        shutil.copy2(STAGE_BQ/rel, out/'inputs'/dst)
    assert sha(out/'inputs/train_rows.gzip.bin') == bq_files['train_rows_sha256']
    (out/'ref').mkdir()
    shutil.copy2(REF, out/'ref/devA_result_janek.json')
    files = {p.relative_to(out).as_posix(): dict(sha256=sha(p), bytes=p.stat().st_size)
             for p in sorted(out.rglob('*')) if p.is_file()}
    assert not any(k.endswith('.gz') for k in files), 'Kaggle unpacks .gz files; do not ship them'
    deva_manifest = sha(DEVA/'KAGGLE_MANIFEST.json')
    (out/'KAGGLE_MANIFEST.json').write_text(json.dumps(dict(
        name=NAME, kind='refine', spec='docs/research/janek/refine_spec_20261125_janek.md',
        train_rows_sha256=bq_files['train_rows_sha256'], selection_sha256=bq_files['selection_sha256'],
        devA_dataset_manifest_sha256=deva_manifest, files=files), indent=1), encoding='utf-8')
    zpath = out.with_suffix('.zip')
    with zipfile.ZipFile(zpath, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out.rglob('*')):
            if p.is_file():
                zf.write(p, f'{NAME}/{p.relative_to(out).as_posix()}')
    print(f'{out}\n{len(files)} files, {sum(f["bytes"] for f in files.values())/2**20:.1f} MiB\n'
          f'zip: {zpath} ({zpath.stat().st_size/2**20:.1f} MiB) SHA-256 {sha(zpath)}\n'
          f'code manifest SHA-256 {sha(out/"code/CODE_MANIFEST.json")} · expects devA manifest {deva_manifest[:12]}…')


if __name__ == '__main__':
    main()
