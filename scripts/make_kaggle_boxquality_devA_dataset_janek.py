"""Kaggle Dataset for the supplementary box-quality evaluation on development A
(docs/research/janek/boxquality_devA_spec_20261125_janek.md). File copying with hash checks only.

runtime/kaggle_upload/cs431-boxquality-devA/ (+ .zip):
  KAGGLE_MANIFEST.json   kind boxquality_devA; SHA-256/size of every file
  code/                  mixab + tests + configs + CODE_MANIFEST.json
  weights/               merge_manifest.json + alpha_1p00.pth (B1)          <- cs431-boxquality staging (hash-checked)
  encoder/, wheels/      MonkeyOCRv2-AS encoder snapshot, MMCV 2.0.1 wheel   <- same staging
  model/                 model_janek.npz + boxquality_fit_janek.json (frozen scorer, SHA pinned in the module)
  clean/train_images/    Clean images of the 1,130 A sources                 <- cs431-csb1-clean staging (+ bundle SHA)
  dump_lr2/, dump_clean/ existing T4 dumps (B1 + B-Clean): references for the group gates and the live-vs-dump check
No .gz files (Kaggle unpacks them on upload). LR2 is rebuilt on Kaggle with Tuan's native_lr2 and checked against the
locked task SHA-256. Run from the workspace root:  python scripts/make_kaggle_boxquality_devA_dataset_janek.py
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
from mixab import boxquality_devA_janek as dv  # noqa: E402
from mixab import merge_audit_janek as ma  # noqa: E402
from mixab.monkey_janek import monkey_root  # noqa: E402

NAME = 'cs431-boxquality-devA'
STAGE_BQ = ROOT/'runtime/kaggle_upload/cs431-boxquality'
STAGE_CLEAN = ROOT/'runtime/kaggle_upload/cs431-csb1-clean'
ART = ROOT/'docs/handoff/artifacts/janek/boxquality_v1_T4'
DUMPS = dict(dump_lr2=ROOT/'runtime/phase0_inputs/dualagree_dump_v1_T4',
             dump_clean=ROOT/'runtime/kaggle_downloads/csb1_clean_T4_20261123-110923/out/clean_dump')


def main():
    sys.stdout.reconfigure(errors='backslashreplace')
    out = ROOT/'runtime/kaggle_upload'/NAME
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    def staged(stage, rel, dest_rel=None):
        files = json.loads((stage/'KAGGLE_MANIFEST.json').read_text(encoding='utf-8'))['files']
        assert sha(stage/rel) == files[rel]['sha256'], f'staging file changed: {stage.name}/{rel}'
        dst = out/(dest_rel or rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stage/rel, dst)

    code = {}
    for p in code_files():
        rel = p.relative_to(ROOT).as_posix()
        (out/'code'/rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out/'code'/rel)
        code[rel] = sha(p)
    assert 'mixab/boxquality_devA_janek.py' in code and 'mixab/boxquality_janek.py' in code
    (out/'code/CODE_MANIFEST.json').write_text(json.dumps(dict(files=code), indent=1, sort_keys=True), encoding='utf-8')
    bq_files = json.loads((STAGE_BQ/'KAGGLE_MANIFEST.json').read_text(encoding='utf-8'))['files']
    for rel in bq_files:
        if rel.startswith(('weights/', 'encoder/', 'wheels/')):
            staged(STAGE_BQ, rel)
    (out/'model').mkdir()
    shutil.copy2(ART/'model_janek.npz', out/'model/model_janek.npz')
    shutil.copy2(ART/'boxquality_fit_janek.json', out/'model/boxquality_fit_janek.json')
    assert sha(out/'model/model_janek.npz') == dv.MODEL_SHA256 and sha(out/'model/boxquality_fit_janek.json') == dv.FIT_SHA256
    rows = ma.load_rows(monkey_root()/'docs/handoff/artifacts/pr_quad_audit_v1')
    A = dv.a_sources(rows)
    for s in A:
        rel = rows[s]['clean_relpath']
        staged(STAGE_CLEAN, f'clean/{rel}')
        assert sha(out/'clean'/rel) == rows[s]['clean_sha256'], rel
    for dest, src in DUMPS.items():
        shutil.copytree(src, out/dest, ignore=shutil.ignore_patterns('desktop.ini'))
    files = {p.relative_to(out).as_posix(): dict(sha256=sha(p), bytes=p.stat().st_size)
             for p in sorted(out.rglob('*')) if p.is_file()}
    assert not any(k.endswith('.gz') for k in files), 'Kaggle unpacks .gz files; do not ship them'
    (out/'KAGGLE_MANIFEST.json').write_text(json.dumps(dict(
        name=NAME, kind='boxquality_devA', spec=dv.SPEC, split=dv.SPLIT, sources=len(A), model_sha256=dv.MODEL_SHA256,
        fit_sha256=dv.FIT_SHA256, files=files), indent=1), encoding='utf-8')
    zpath = out.with_suffix('.zip')
    with zipfile.ZipFile(zpath, 'w', compression=zipfile.ZIP_STORED) as zf:
        for p in sorted(out.rglob('*')):
            if p.is_file():
                zf.write(p, f'{NAME}/{p.relative_to(out).as_posix()}')
    print(f'{out}\n{len(files)} files, {sum(f["bytes"] for f in files.values())/2**20:.0f} MiB, A sources {len(A)}\n'
          f'zip: {zpath} ({zpath.stat().st_size/2**20:.0f} MiB) SHA-256 {sha(zpath)}\n'
          f'code manifest SHA-256 {sha(out/"code/CODE_MANIFEST.json")}')


if __name__ == '__main__':
    main()
