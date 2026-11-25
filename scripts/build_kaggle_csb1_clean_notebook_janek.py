"""Generate notebooks/kaggle_csb1_clean_janek.ipynb: Clean dump + router-free evaluation of CS-B1@0.7.
Spec: docs/research/janek/csb1_clean_check_spec_20261123_janek.md. Ô 1 is the zoom notebook's Ô 1 with the dataset kind/paths changed;
Ô 2 (environment) is reused verbatim. Re-run: python scripts/build_kaggle_csb1_clean_notebook_janek.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_kaggle_zoom_notebook_janek as zoomnb  # noqa: E402

OUT = ROOT/'notebooks/kaggle_csb1_clean_janek.ipynb'
CELLS = []


def md(text):
    CELLS.append(dict(cell_type='markdown', metadata={}, source=text.strip('\n')))


def code(text):
    CELLS.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=text.strip('\n')))


md(r'''
# CS431 · Bổ sung Clean cho CS-B1@0.7 (Kaggle T4)

**Notebook làm gì:** chạy B1 và B-Clean trên **2.262 ảnh Clean**, lưu box + score giống hệt dump LR2 trước đây. Sau đó đánh giá **CS-B1 áp cho mọi ảnh (không định tuyến)** so với B1 `.40/.60`, B1 `epsilon_005/.50` và B-Clean `.40/.60`, trên cả Clean và LR2.

- **Không phải phương pháp mới**, không quét ngưỡng. Spec khóa trước: `docs/research/janek/csb1_clean_check_spec_20261123_janek.md`.
- Dữ liệu là development, nên **không xác nhận** được cải tiến.

### Trước khi chạy
1. Kaggle → **New Dataset**, tải `cs431-csb1-clean.zip` lên (Private).
2. Import notebook này. Bật **GPU T4 x2** (hoặc T4), **Internet On**, **Add Input** chọn dataset vừa tạo.
3. Chạy **Ô 1 → Ô 4**, xem thời gian dự kiến ở Ô 4, rồi chạy **Ô 5 → Ô 7**.

**Thời gian (chưa đo):** cài môi trường 5–40 phút; dump khoảng 55 phút; đánh giá khoảng 10 phút.
''')

c1 = zoomnb.CELLS[2]['source']
c1 = c1.replace("get('kind') == 'zoom_pilot'", "get('kind') == 'csb1_clean'").replace("'zoom' in z.name.lower()", "'csb1-clean' in z.name.lower()")
c1 = c1.replace("Không thấy dataset cs431-zoom-pilot", "Không thấy dataset cs431-csb1-clean")
c1 = c1.replace("print(f'Dataset: {DS} ({len(manifest[\"files\"])} file, SHA khớp) · mẫu {manifest[\"n_sources\"]} nguồn A')",
                "print(f'Dataset: {DS} ({len(manifest[\"files\"])} file, SHA khớp) · {manifest[\"sources\"]} nguồn')")
c1 = c1.replace("WEIGHTS, CLEAN, DUMP, A2R = DS/'weights', DS/'clean', DS/'dump', DS/'a2r/B1_source_counts.npz'",
                "WEIGHTS, CLEAN, DUMP, A2R = DS/'weights', DS/'clean', DS/'dump', DS/'a2r'")
assert "csb1_clean" in c1 and "DS/'a2r'" in c1 and 'manifest["sources"]' in c1
CELLS.append(dict(zoomnb.CELLS[1]))
CELLS.append(dict(zoomnb.CELLS[2], source=c1))
CELLS.extend(zoomnb.CELLS[3:5])             # Ô 2 environment (verbatim)

md(r'''
## Ô 3 · Repo của Tuấn, config, 2 checkpoint, 2.262 ảnh Clean

Clone `MONKEY_V2` @ `dc19b85`, vá config (chỉ đường dẫn encoder), kiểm SHA 2 checkpoint. Dùng 2.262 ảnh Clean có sẵn trong dataset (kiểm SHA từng ảnh). Chỉ khi dataset thiếu ảnh mới tải zip TextVQA chính thức (7,07 GB, chưa đo thời gian). Sau đó chạy test CPU.
''')

code(r'''
# ===================== Ô 3 · DỮ LIỆU =====================
from pathlib import PureWindowsPath
# Dataset remains read-only and hash-verified. Patch only the working code copy
# so existing Windows-generated manifests work on Linux; keep manifest bytes intact.
patch_receipt = {}
for rel in ('mixab/dualagree_janek.py', 'mixab/merge_audit_janek.py'):
    target = CODE/rel
    original = (DS/'code'/rel).read_text(encoding='utf-8')
    updated = original.replace('from pathlib import Path\n', 'from pathlib import Path, PureWindowsPath\n')
    for expr in ("model['checkpoint']", "by_alpha[alpha]['checkpoint']"):
        updated = updated.replace(f'Path({expr}).name', f'PureWindowsPath({expr}).name')
    compile(updated, str(target), 'exec')
    target.write_text(updated, encoding='utf-8')
    patch_receipt[rel] = dict(dataset_sha256=sha256(DS/'code'/rel), executed_sha256=sha256(target))
(OUT/'checkpoint_path_compatibility.json').write_text(json.dumps(patch_receipt, indent=2), encoding='utf-8')
LOG = LOGS/f'data_{GPU}.log' 
if not (MV2/'.git').is_dir():
    ch.run(['git', 'clone', PIN['MONKEY_V2_URL'], MV2], LOG, env=dict(GIT_LFS_SKIP_SMUDGE='1'))
ch.run(['git', '-C', MV2, 'checkout', '--detach', PIN['MONKEY_V2_COMMIT']], LOG, env=dict(GIT_LFS_SKIP_SMUDGE='1'))
if ch.output_of(['git', '-C', MV2, 'status', '--porcelain', '--untracked-files=no']):
    fail('Checkout MONKEY_V2 bị sửa.')
if sha256(BUNDLE/'bundle_manifest.json') != PIN['BUNDLE_MANIFEST_SHA256']:
    fail('bundle_manifest.json khác bản đã khóa.')
CONFIG_PATH.write_text(ch.patch_inference_config((MV2/'docs/handoff/inference_config.py').read_text(encoding='utf-8'), ENCODER), encoding='utf-8')
m = json.loads((WEIGHTS/'merge_manifest.json').read_text())
missing = [PureWindowsPath(x['checkpoint']).name for x in m['models']
           if x['alpha'] in ('0', '1') and not (WEIGHTS/PureWindowsPath(x['checkpoint']).name).is_file()]
if missing:
    fail(f'Dataset thiếu checkpoint: {missing}')
for x in m['models']:
    if x['alpha'] in ('0', '1') and sha256(WEIGHTS/PureWindowsPath(x['checkpoint']).name) != x['checkpoint_sha256']:
        fail(f"{PureWindowsPath(x['checkpoint']).name} sai SHA.")
# Ảnh Clean: tải zip TextVQA chính thức (7,07 GB) vào /tmp, chỉ trích 2.262 ảnh, kiểm SHA từng ảnh với source_cohort.csv
items = ch.needed_clean_images(BUNDLE)
CLEAN = DS/'clean'                                   # ưu tiên ảnh đã có trong dataset (đúng byte từ cache Drive)
check = ch.verify_images(CLEAN, items)
if check['ok'] != len(items):
    print('Dataset không có đủ ảnh Clean đúng SHA:', check, '-> tải zip TextVQA chính thức.')
    CLEAN = T/'textvqa'
    check = ch.verify_images(CLEAN, items)
if check['ok'] != len(items):
    big = T/'downloads/train_val_images.zip'
    big.parent.mkdir(parents=True, exist_ok=True)
    URL, BYTES = 'https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip', 7072297970
    if not (big.is_file() and big.stat().st_size == BYTES):
        ch.run(['wget', '-c', '-q', '-O', big, URL], LOG, title='download TextVQA zip (7 GB)')
    if big.stat().st_size != BYTES:
        fail(f'Zip TextVQA có {big.stat().st_size} byte, kỳ vọng {BYTES}; chạy lại ô để tải tiếp.')
    ch.extract_members(big, [r for r, _ in items], CLEAN)
    check = ch.verify_images(CLEAN, items)
    if check['ok'] == len(items):
        big.unlink()
print('Ảnh Clean:', check)
if check['ok'] != 2262:
    fail('Ảnh Clean chưa đủ hoặc sai SHA.')
ch.run([PY, '-m', 'unittest', 'discover', '-s', 'tests', '-t', '.'], LOG, cwd=CODE, env=ENV, title='mixab unit tests')
print('\nÔ 3 xong.')
''')

md(r'''
## Ô 4 · Probe 40 nguồn Clean: thời gian + độ lệch so với A2r (A100)

So counts tính lại từ dump với counts Clean của A2r ở cả 72 setting. T4 khác A100 nên dự kiến có lệch ±1 count; ô chỉ báo tỉ lệ, không chặn. Ô cũng in thời gian dự kiến cho dump đầy đủ.
''')

code(r'''
# ===================== Ô 4 · PROBE =====================
PROBE = OUT/'clean_probe'
if not (PROBE/'dump_done.json').is_file():
    ch.run([PY, '-m', 'mixab.dualagree_janek', 'dump', '--condition', 'clean', '--bundle', BUNDLE, '--clean-root', CLEAN,
            '--prepared-root', T/'prepared', '--model-config', CONFIG_PATH, '--weights-dir', WEIGHTS, '--probe', '40',
            '--chunk', '16', '--output', PROBE], LOGS/'clean_probe.log', cwd=CODE, env=ENV, title='clean probe dump')
done = json.loads((PROBE/'dump_done.json').read_text())
for name, p in done['profile'].items():
    print(f"  {name:8} {p['mean_seconds_per_task']:.2f} s/ảnh → dump đủ 2.262 ảnh ~{p['mean_seconds_per_task']*2262/60:.0f} phút")
proc = subprocess.run([PY, '-m', 'mixab.dualagree_janek', 'fidelity', '--dump', str(PROBE), '--bundle', str(BUNDLE), '--npz-run', str(A2R)],
                      cwd=CODE, env=dict(os.environ, **ENV), capture_output=True, text=True)
if proc.returncode != 0 or '{' not in proc.stdout:
    print(proc.stdout[-3000:]); print(proc.stderr[-4000:]); fail('Kiểm trung thực lỗi.')
fid = json.loads(proc.stdout[proc.stdout.index('{'):])
(PROBE/'fidelity.json').write_text(json.dumps(fid, indent=1))
for name, f in fid.items():
    print(f"  {name:8} so với A2r Clean (A100): {f['status']} ({f['mismatched']}/{f['checked']} ô lệch)")
''')

md(r'''
## Ô 5 · Dump đủ 2.262 ảnh Clean × 2 model (GPU)

Lưu theo shard 64 ảnh trong `cs431_out/clean_dump/`. Bị ngắt trong cùng phiên thì chạy lại ô này, shard xong sẽ bỏ qua.
''')

code(r'''
# ===================== Ô 5 · DUMP CLEAN =====================
CDUMP = OUT/'clean_dump'
if not (CDUMP/'dump_done.json').is_file():
    ch.run([PY, '-m', 'mixab.dualagree_janek', 'dump', '--condition', 'clean', '--bundle', BUNDLE, '--clean-root', CLEAN,
            '--prepared-root', T/'prepared', '--model-config', CONFIG_PATH, '--weights-dir', WEIGHTS, '--chunk', '64',
            '--output', CDUMP], LOGS/f'clean_dump_{time.strftime("%Y%m%d-%H%M%S")}.log', cwd=CODE, env=ENV, title='clean full dump')
print(json.loads((CDUMP/'dump_done.json').read_text())['profile'])
''')

md(r'''
## Ô 6 · Đánh giá CS-B1 trên Clean + hệ thống không định tuyến (CPU)
''')

code(r'''
# ===================== Ô 6 · ĐÁNH GIÁ =====================
EV = OUT/'csb1_clean_eval'
ch.run([PY, '-m', 'mixab.csb1_clean_janek', '--bundle', BUNDLE, '--clean-dump', OUT/'clean_dump', '--lr2-dump', DUMP, '--output', EV],
       LOGS/'csb1_clean_eval.log', cwd=CODE, env=ENV, title='csb1 clean evaluate')
r = json.loads((EV/'csb1_clean_result.json').read_text())
for sp in ('ALL', 'A', 'B'):
    e = r['splits'][sp]
    print(f"== {sp} ({e['sources']} nguồn)")
    for arm in e['mean']:
        print(f"  {arm:16} Clean {e['clean'][arm]['hmean']*100:.2f}  LR2 {e['lr2'][arm]['hmean']*100:.2f}  mean {e['mean'][arm]*100:.2f}")
    d = e['CSB1_minus_B1 .40/.60']
    print(f"  CS-B1 − B1 .40/.60: Clean {d['clean']['delta_pp']:+.2f} {d['clean']['ci95_pp']}, LR2 {d['lr2']['delta_pp']:+.2f} {d['lr2']['ci95_pp']}")
    print(f"  hệ thống không định tuyến − model đơn tốt nhất ({e['best_single_by_mean']}): {e['router_free_system_minus_best_single_pp']:+.2f} pp")
print('Cổng nhóm trên A (mô tả):', r['group_gate_A_descriptive'])
''')

md(r'''
## Ô 7 · Đóng gói

Tạo `/kaggle/working/cs431_csb1_clean_<GPU>_<thời gian>.zip`. Tải ở khung **Output** và gửi lại Claude (hoặc báo Claude tải bằng Kaggle API).
''')

code(r'''
# ===================== Ô 7 · ĐÓNG GÓI =====================
pack = W/f'cs431_csb1_clean_{GPU}_{time.strftime("%Y%m%d-%H%M%S")}.zip'
with zipfile.ZipFile(pack, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.suffix != '.whl':
            zf.write(p, p.relative_to(OUT).as_posix())
    zf.writestr('code_manifest_sha256.txt', code_sha + '\n')
print('Đã đóng gói:', pack, f'({pack.stat().st_size/2**20:.1f} MiB)')
''')


def main():
    sys.stdout.reconfigure(errors='backslashreplace')
    notebook = dict(nbformat=4, nbformat_minor=5,
                    metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python'),
                                  language_info=dict(name='python'),
                                  kaggle=dict(accelerator='nvidiaTeslaT4', isInternetEnabled=True, isGpuEnabled=True)),
                    cells=[dict(c, id=f'cell-{i:02d}',
                                source=[line + '\n' for line in c['source'].split('\n')][:-1] +
                                [c['source'].split('\n')[-1]]) for i, c in enumerate(CELLS)])
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(f'{OUT}: {len(CELLS)} cells')


if __name__ == '__main__':
    main()
