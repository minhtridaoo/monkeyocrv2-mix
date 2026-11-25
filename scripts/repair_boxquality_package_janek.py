"""Build a matched notebook/dataset pair from the verified legacy archive, without altering it."""
import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def digest(b): return hashlib.sha256(b).hexdigest()
def main():
 src=ROOT/'runtime/kaggle_upload/cs431-boxquality.zip'
 dst=src.with_name('cs431-boxquality-janek-fixed.zip')
 with zipfile.ZipFile(src) as z:
  prefix='cs431-boxquality/'
  m=json.loads(z.read(prefix+'KAGGLE_MANIFEST.json'))
  payload={}
  for rel,info in m['files'].items():
   raw=z.read(prefix+rel)
   assert digest(raw)==info['sha256'],rel
   if not rel.startswith('code/'):
    payload[rel.replace('inputs/train_rows.jsonl.gz','inputs/train_rows.gzip.bin')]=raw
 files={}
 for folder in ('mixab','tests','configs'):
  for f in sorted((ROOT/folder).rglob('*')):
   if f.is_file() and '__pycache__' not in f.parts and f.suffix not in ('.pyc','.tmp'):
    rel=f.relative_to(ROOT).as_posix();raw=f.read_bytes();payload['code/'+rel]=raw;files[rel]=digest(raw)
 extra=ROOT/'docs/handoff/artifacts/janek/diag_v2_cpu/p0_fixed_janek.json'
 if extra.is_file():
  rel=extra.relative_to(ROOT).as_posix();raw=extra.read_bytes();payload['code/'+rel]=raw;files[rel]=digest(raw)
 payload['code/CODE_MANIFEST.json']=json.dumps(dict(files=files),sort_keys=True,indent=1).encode()
 assert 'code/mixab/boxquality_janek.py' in payload
 m['name']='cs431-boxquality-janek-fixed'
 m['packaging_note']='Original gzip bytes stored as .bin; notebook restores original name after SHA verification. No label/weight changes.'
 m['files']={k:dict(sha256=digest(v),bytes=len(v)) for k,v in payload.items()}
 payload['KAGGLE_MANIFEST.json']=json.dumps(m,indent=1).encode()
 with zipfile.ZipFile(dst,'w',compression=zipfile.ZIP_STORED) as z:
  for rel,raw in payload.items():z.writestr(prefix+rel,raw)
 nb=json.loads((ROOT/'notebooks/kaggle_boxquality_janek.ipynb').read_text(encoding='utf-8'))
 for c in nb['cells']:
  s=''.join(c['source'])
  if c['cell_type']=='markdown':s=s.replace('cs431-boxquality.zip','cs431-boxquality-janek-fixed.zip')
  if "bad = [rel" in s:
   start=s.index('bad = [rel');end=s.index("print(f'Dataset:",start)
   s=s[:start]+'''bad = []
for rel, info in manifest['files'].items():
    path = DS/rel
    if not path.is_file():
        bad.append(dict(file=rel, reason='MISSING', expected_sha256=info['sha256']))
    else:
        actual = sha256(path)
        if actual != info['sha256']:
            bad.append(dict(file=rel, reason='SHA_MISMATCH', expected_sha256=info['sha256'], actual_sha256=actual, bytes=path.stat().st_size))
if bad:
    print(json.dumps(bad, indent=2))
    fail('Dataset thiếu hoặc thay đổi file; xem chi tiết phía trên.')
if not (DS/'code/mixab/boxquality_janek.py').is_file() or not (DS/'inputs/train_rows.gzip.bin').is_file():
    fail('Hãy gắn dataset cs431-boxquality-janek-fixed, bỏ dataset cũ.')
'''+s[end:]
  s=s.replace("INP = DS/'inputs'",'''INP = T/'boxquality_inputs_verified'
shutil.copytree(DS/'inputs', INP, dirs_exist_ok=True)
shutil.copyfile(INP/'train_rows.gzip.bin', INP/'train_rows.jsonl.gz')
if sha256(INP/'train_rows.jsonl.gz') != manifest['train_rows_sha256']:
    fail('Nhãn TRAIN sau khôi phục tên không khớp SHA gốc.')''')
  c['source']=s.splitlines(keepends=True)
  if c['cell_type']=='code':
   compile(s,'<notebook>','exec');c['outputs']=[];c['execution_count']=None
 out=ROOT/'notebooks/kaggle_boxquality_janek_fixed.ipynb'
 out.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
 with zipfile.ZipFile(dst) as z:
  for rel,info in m['files'].items():assert digest(z.read(prefix+rel))==info['sha256'],rel
 print('PASS: source and new archive hashes; notebook syntax; original label bytes preserved.')
 print(dst.name,out.name,len(m['files']))
if __name__=='__main__':main()
