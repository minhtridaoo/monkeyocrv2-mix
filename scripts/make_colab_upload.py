"""Build runtime/colab_upload/mixab_code.zip: the only file the user uploads to Google Drive.

Contains mixab/, tests/, configs/, scripts/ and the runbook, plus CODE_MANIFEST.json with the
SHA-256 of every file so the notebook can show exactly which code version ran.
Run from the workspace root:  python scripts/make_colab_upload.py
"""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ['mixab', 'tests', 'configs', 'scripts', 'docs/RUNBOOK_mix_a_T4_A100.md',
           'results/diag_v2_cpu/p0_fixed.json']
OUT = ROOT/'runtime/colab_upload/mixab_code.zip'


def files():
    for item in INCLUDE:
        path = ROOT/item
        candidates = [path] if path.is_file() else sorted(p for p in path.rglob('*') if p.is_file())
        for p in candidates:
            if '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.tmp'):
                yield p


def main():
    import sys
    sys.stdout.reconfigure(errors='backslashreplace')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    manifest = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files()}
    temp = OUT.with_name(OUT.name + '.part')
    with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(manifest):
            zf.write(ROOT/name, name)
        zf.writestr('CODE_MANIFEST.json', json.dumps(dict(files=manifest), indent=1, sort_keys=True))
    temp.replace(OUT)
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print(f'{OUT} ({OUT.stat().st_size/1024:.0f} KiB, {len(manifest)} files)\nSHA-256 {digest}')


if __name__ == '__main__':
    main()
