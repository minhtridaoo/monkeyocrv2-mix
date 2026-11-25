"""Validate preserved artifacts and export a conflict-checked Janek overlay."""
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mixab.monkey_janek import monkey_root


def main():
    records = json.loads((ROOT/'file_ownership_janek.json').read_text(encoding='utf-8'))
    preserved = 0
    for row in records:
        p = ROOT/row['new']
        if (row['old'].startswith('results/') and p.suffix != '.md') or '/vendor/' in row['old']:
            assert hashlib.sha256(p.read_bytes()).hexdigest() == row['sha256_before'], row['new']
            preserved += 1
        row['sha256_after'] = hashlib.sha256(p.read_bytes()).hexdigest()
    (ROOT/'file_ownership_janek.json').write_text(json.dumps(records, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    names = subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'], cwd=ROOT).decode().split('\0')
    names = sorted({n for n in names if n and (ROOT/n).is_file() and n not in ('.gitignore','tests/__init__.py')})
    ref = monkey_root()
    assert ref != ROOT, 'Export from the standalone Janek checkout before merging.'
    other = set(subprocess.check_output(['git','ls-tree','-r','--name-only','HEAD'], cwd=ref).decode().splitlines())
    collisions = sorted(set(names) & other)
    assert not collisions, f'Conflicting files: {collisions}'
    out = ROOT/'runtime/restructure_janek/merge_ready_janek.zip'
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for n in names:z.write(ROOT/n,n)
    print(f'{len(names)} files; {preserved} artifact/vendor hashes preserved; 0 collisions.')
    print('Output: runtime/restructure_janek/merge_ready_janek.zip')


if __name__ == '__main__':
    main()
