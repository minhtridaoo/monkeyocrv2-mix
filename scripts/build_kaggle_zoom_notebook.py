"""Generate notebooks/kaggle_zoom_pilot.ipynb: ZB1 zoom screening pilot (docs/zoom_pilot_spec_20261122.md).

The environment cell (Ô 2) is reused verbatim from scripts/build_kaggle_notebook.py (same pins as the T4 dump).
Re-run: python scripts/build_kaggle_zoom_notebook.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_kaggle_notebook as base  # noqa: E402

OUT = ROOT/'notebooks/kaggle_zoom_pilot.ipynb'
CELLS = []


def md(text):
    CELLS.append(dict(cell_type='markdown', metadata={}, source=text.strip('\n')))


def code(text):
    CELLS.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=text.strip('\n')))


md(r'''
# CS431 · Pilot sàng lọc ZB1 (zoom cục bộ bằng B1) trên Kaggle T4

**Kiểm tra giả thuyết:** detector làm việc ở cạnh ngắn 1024, nên chữ nhỏ quá nhỏ trên đầu vào của nó. Nếu cắt vùng ảnh và để pipeline phóng vùng đó lên (cạnh ngắn của vùng → 1024), B1 có thấy thêm chữ nhỏ mà không thêm quá nhiều FP không?

**Phạm vi:** một checkpoint B1, không train, không SR. Mẫu cố định gồm 300 nguồn của split A, chạy cả Clean và LR2 (600 ảnh). Spec khóa trước: `docs/zoom_pilot_spec_20261122.md`.

**Các arm:**
- **F:** ảnh đầy đủ (baseline).
- **G2:** F + lưới 2×2. **Arm chính, quyết định cổng.**
- **G3:** F + lưới 3×3 (phụ).
- **S2:** F + cả ảnh ở cạnh ngắn 2048 (phụ).
- **R3:** F + 3 ô G3 chọn bằng dự đoán (phụ, mô phỏng từ G3).

### Trước khi chạy
1. Kaggle → **Create → New Dataset**, tải `cs431-zoom-pilot.zip` lên, để **Private**.
2. **File → Import Notebook**, chọn file này.
3. Ở cột phải: **Accelerator** `GPU T4 x2` (hoặc `GPU T4`), **Internet On**, **Add Input** → chọn dataset vừa tạo.
4. Chạy **Ô 1 → Ô 5**, đọc kết quả chạy thử ở Ô 5, rồi mới chạy **Ô 6 → Ô 8**.

**Thời gian: chưa đo, chỉ ước lượng.**
- Ô 2: 5–40 phút, tùy có phải build MMCV không.
- Ô 6: khoảng 1–2 giờ cho 600 ảnh, mỗi ảnh khoảng 15 lượt forward cộng một lượt S2 lớn.
- Ô 5 in ra con số dự kiến **đo thật** trên máy của bạn.

**Dừng nếu:**
- Ô 5 báo kiểm tọa độ 1×1 sai;
- hoặc thời gian dự kiến vượt 4 giờ (báo lại Claude, không tự giảm số ảnh).

Kết quả nằm trong `/kaggle/working/cs431_out/`. Ô 8 tạo một file zip để tải về.

**Trạng thái:** notebook đã tạo và kiểm cú pháp ở máy local; **chưa chạy trên Kaggle**.
''')

md(r'''
## Ô 1 · Cấu hình, tìm dataset, kiểm GPU

Kiểm SHA của mọi file trong dataset. Ghim **GPU 0** (`CUDA_VISIBLE_DEVICES=0`), vì code của Tuấn yêu cầu đúng một GPU.
''')

code(r'''
# ===================== Ô 1 · CẤU HÌNH =====================
import hashlib, json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path

PIN = dict(
    MONKEY_V2_URL='https://github.com/ncat218/MONKEY_V2.git',
    MONKEY_V2_COMMIT='dc19b85c8f73d56dd0510165069597363fe25380',
    MONKEYOCRV2_URL='https://github.com/Yuliang-Liu/MonkeyOCRv2.git',
    MONKEYOCRV2_COMMIT='da96afede92f198c19aca74e1e5f5aa7f4c26af2',
    TORCH='2.9.0', TORCHVISION='0.24.0', TORCHAUDIO='2.9.0', TORCH_INDEX='https://download.pytorch.org/whl/cu128',
    MMENGINE='0.10.7', MMCV='2.0.1', MMOCR='1.0.1', PILLOW='11.3.0', OPENCV='4.10.0.84',
    DB_PP_SHA256='9bfff8aaaca103dd08d7565f2b7a5410ab20a6427fe1f0baad5a5d04d207c8b8',
    BUNDLE_MANIFEST_SHA256='e06ef54e720b43bcb658fd734291fb1b6e4a08db13a8bd4048f16cfd3a2dc573',
)
W = Path('/kaggle/working'); OUT = W/'cs431_out'; LOGS = OUT/'logs'
T = Path('/tmp/cs431')
for p in (OUT, LOGS, T):
    p.mkdir(parents=True, exist_ok=True)

def fail(msg):
    raise SystemExit('LỖI: ' + msg)

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

found = [p for p in Path('/kaggle/input').glob('**/KAGGLE_MANIFEST.json') if json.loads(p.read_text()).get('kind') == 'zoom_pilot']
if not found:
    zips = [z for z in Path('/kaggle/input').glob('**/*.zip') if 'zoom' in z.name.lower()]
    if not zips:
        fail('Không thấy dataset cs431-zoom-pilot. Bấm "Add Input" và chọn dataset đó.')
    with zipfile.ZipFile(zips[0]) as zf:
        zf.extractall(T/'dataset')
    found = [p for p in (T/'dataset').glob('**/KAGGLE_MANIFEST.json') if json.loads(p.read_text()).get('kind') == 'zoom_pilot']
if len(found) != 1:
    fail(f'Thấy {len(found)} dataset zoom; chỉ gắn một.')
DS = found[0].parent
manifest = json.loads(found[0].read_text())
bad = [rel for rel, info in manifest['files'].items()
       if rel != 'KAGGLE_MANIFEST.json' and (not (DS/rel).is_file() or sha256(DS/rel) != info['sha256'])]
if bad:
    fail(f'{len(bad)} file thiếu hoặc sai SHA, ví dụ {bad[:3]}.')
print(f'Dataset: {DS} ({len(manifest["files"])} file, SHA khớp) · mẫu {manifest["n_sources"]} nguồn A')

CODE = T/'code'
if CODE.exists():
    shutil.rmtree(CODE)
shutil.copytree(DS/'code', CODE)
code_sha = hashlib.sha256((DS/'code/CODE_MANIFEST.json').read_bytes()).hexdigest()
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
import importlib
import mixab.colab_helpers as ch
importlib.reload(ch)

gpus = (ch.output_of(['nvidia-smi', '--query-gpu=index,name,driver_version,memory.total', '--format=csv,noheader']) or '').splitlines()
if not gpus:
    fail('Không thấy GPU. Settings -> Accelerator -> GPU T4 x2 (hoặc GPU T4).')
print('GPU của máy:'); [print('  ', g) for g in gpus]
GPU = ch.gpu_label(gpus[0].split(',')[1].strip())
CONFIG_NOTE = f'{len(gpus)} GPU trên máy; chỉ dùng GPU 0 qua CUDA_VISIBLE_DEVICES=0'
print(CONFIG_NOTE)

PY = sys.executable
MV2, MOCR = T/'MONKEY_V2', T/'MonkeyOCRv2'
BUNDLE = MV2/'docs/handoff/artifacts/pr_quad_audit_v1'
WEIGHTS, CLEAN, DUMP, A2R = DS/'weights', DS/'clean', DS/'dump', DS/'a2r/B1_source_counts.npz'
ENCODER = T/'encoder'
if not (ENCODER/'model.safetensors').is_file():
    shutil.copytree(DS/'encoder', ENCODER, dirs_exist_ok=True)
PREPARED, CONFIG_PATH = T/'prepared', T/'inference_config_kaggle.py'
ENV = dict(MONKEY_V2_ROOT=str(MV2), PYTHONPATH=str(CODE), CUDA_VISIBLE_DEVICES='0')
print(f'Python {sys.version.split()[0]} · code manifest {code_sha[:12]}… · kết quả: {OUT}')
''')

CELLS.extend(base.CELLS[3:5])          # Ô 2 (environment) verbatim from the latency notebook

md(r'''
## Ô 3 · Repo của Tuấn, config, checkpoint B1, ảnh LR2 của mẫu

- Clone `MONKEY_V2` @ `dc19b85`, không cần LFS. Kiểm SHA bundle.
- Vá config: chỉ đổi 2 đường dẫn encoder.
- Kiểm SHA của B1.
- Tạo 300 ảnh LR2 bằng `native_lr2` của Tuấn, kiểm SHA từng ảnh.
- Chạy test CPU.
''')

code(r'''
# ===================== Ô 3 · DỮ LIỆU =====================
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
b1 = next(x for x in m['models'] if x['alpha'] == '1')
if sha256(WEIGHTS/Path(b1['checkpoint']).name) != b1['checkpoint_sha256']:
    fail('alpha_1p00.pth (B1) sai SHA.')
ch.run([PY, '-m', 'mixab.zoom', 'prepare', '--bundle', BUNDLE, '--clean-root', CLEAN, '--prepared-root', PREPARED,
        '--output', OUT], LOG, cwd=CODE, env=ENV, title='zoom prepare')
ch.run([PY, '-m', 'unittest', 'discover', '-s', 'tests', '-t', '.'], LOG, cwd=CODE, env=ENV, title='mixab unit tests')
print('\nÔ 3 xong.')
''')

md(r'''
## Ô 4 · Kiểm ngắn ảnh trang báo, GT, evaluator (CPU, khoảng vài phút)

Chỉ ghi nhận bối cảnh, **không** đổi kết quả hay protocol:
- nguồn `6e340fcbb1e0a877`: số GT hợp lệ và GT ignore, ROI;
- so counts của `CandidateMatcher` với evaluator chính thức;
- phần FP của B1 trên A thuộc 1 và 10 nguồn nhiều FP nhất;
- hình chồng GT lên ảnh.
''')

code(r'''
# ===================== Ô 4 · KIỂM TRANG BÁO =====================
PC = OUT/'pagecheck'
if not (PC/'pagecheck.json').is_file():
    ch.run([PY, '-m', 'mixab.zoom', 'pagecheck', '--bundle', BUNDLE, '--dump', DUMP, '--clean-root', CLEAN, '--output', PC],
           LOGS/'pagecheck.log', cwd=CODE, env=ENV, title='pagecheck')
r = json.loads((PC/'pagecheck.json').read_text())
d = r['dense_source']
print(f"Nguồn {d['source']}: GT hợp lệ {d['valid_gt']}, GT ignore {d['ignored_gt']}, ROI {d['roi_area_fraction']:.1%}, "
      f"B1 LR2 counts {d['b1_lr2_counts']} (evaluator chính thức khớp: {d['official_evaluator_agrees']})")
print(f"Trên A: nguồn nhiều FP nhất chiếm {r['A_fp_share_top1']:.1%} FP của B1; 10 nguồn đầu chiếm {r['A_fp_share_top10']:.1%}")
from IPython.display import Image as _Img, display
display(_Img(filename=str(PC/'pagecheck_dense_source.png'), width=900))
''')

md(r'''
## Ô 5 · Chạy thử 3 nguồn (6 ảnh): kiểm tọa độ, bộ nhớ, thời gian

- **Kiểm tọa độ:** crop 1×1 (cả ảnh, ghi PNG rồi chạy lại) phải cho **đúng** box của ảnh đầy đủ. Sai thì ô dừng.
- In thời gian mỗi ảnh, **thời gian dự kiến cho toàn pilot** (đo thật), bộ nhớ đỉnh, và trạng thái S2 (OOM hay chạy được).
- Dự kiến vượt 4 giờ thì **dừng và báo Claude**, không tự đổi cấu hình.
''')

code(r'''
# ===================== Ô 5 · CHẠY THỬ =====================
SMOKE = OUT/'zoom_smoke'
if not (SMOKE/'infer_summary.json').is_file():
    ch.run([PY, '-m', 'mixab.zoom', 'infer', '--bundle', BUNDLE, '--clean-root', CLEAN, '--prepared-root', PREPARED,
            '--model-config', CONFIG_PATH, '--weights-dir', WEIGHTS, '--tmp', T/'crops_smoke', '--limit-sources', '3',
            '--identity-check', '--shard', '6', '--output', SMOKE], LOGS/'zoom_smoke.log', cwd=CODE, env=ENV, title='zoom smoke')
s = json.loads((SMOKE/'infer_summary.json').read_text())
import gzip
recs = [json.loads(l) for p in sorted(SMOKE.glob('shard_*.jsonl.gz')) for l in gzip.open(p, 'rt')]
print(f"Kiểm tọa độ 1×1: {s['identity_check']}")
print(f"{s['seconds_per_task']:.1f} s/ảnh → dự kiến toàn pilot (600 ảnh): {s['projected_minutes_full_pilot']:.0f} phút")
for r in recs:
    f = r['full']
    print(f"  {r['condition']:5} {r['size']} → tensor {f['img_shape']} · F {f['ms']:.0f} ms, {len(f['polys'])} box · "
          f"ô 2×2 {sum(t['ms'] for t in r['tiles']['2']):.0f} ms · ô 3×3 {sum(t['ms'] for t in r['tiles']['3']):.0f} ms · "
          f"S2 {r['s2'].get('status')} {r['s2'].get('img_shape', '')} · đỉnh {max(r['peak_mib_tiles'], r['peak_mib_s2']):.0f} MiB")
if s['projected_minutes_full_pilot'] > 240:
    print('\nDỪNG: dự kiến vượt 4 giờ. Báo Claude trước khi chạy Ô 6.')
else:
    print('\nĐược chạy Ô 6.')
''')

md(r'''
## Ô 6 · Chạy pilot đủ 600 ảnh (GPU)

- Lưu sau mỗi 20 ảnh vào `cs431_out/zoom_infer/`. Bị ngắt **trong cùng phiên** thì chạy lại ô này, ảnh đã xong sẽ bỏ qua.
- Nếu phiên Kaggle kết thúc, thư mục `/kaggle/working` mất. Trước khi dừng, chạy Ô 8 để tải phần đã xong về.
''')

code(r'''
# ===================== Ô 6 · PILOT =====================
INF = OUT/'zoom_infer'
ch.run([PY, '-m', 'mixab.zoom', 'infer', '--bundle', BUNDLE, '--clean-root', CLEAN, '--prepared-root', PREPARED,
        '--model-config', CONFIG_PATH, '--weights-dir', WEIGHTS, '--tmp', T/'crops', '--output', INF],
       LOGS/f'zoom_infer_{time.strftime("%Y%m%d-%H%M%S")}.log', cwd=CODE, env=ENV, title='zoom pilot infer')
print(json.loads((INF/'infer_summary.json').read_text()))
''')

md(r'''
## Ô 7 · Đánh giá (CPU, chưa đo, ước lượng 10–20 phút)

- Dựng các arm F, G2, G3, S2, R3 bằng **đúng luật hợp nhất đã khóa**, chấm bằng `CandidateMatcher` của Tuấn.
- Báo LR2 và Clean: Hmean, TP/FP/FN, recall theo bin; Δ so với F kèm CI theo nguồn; mean hệ thống; số lượt forward; thời gian.
- **Cổng sàng lọc chỉ áp cho G2** (spec mục 7).
- Chẩn đoán dùng GT được báo riêng.
''')

code(r'''
# ===================== Ô 7 · ĐÁNH GIÁ =====================
EV = OUT/'zoom_eval'
ch.run([PY, '-m', 'mixab.zoom', 'evaluate', '--bundle', BUNDLE, '--infer-dir', OUT/'zoom_infer', '--dump', DUMP,
        '--a2r', A2R, '--output', EV], LOGS/'zoom_eval.log', cwd=CODE, env=ENV, title='zoom evaluate')
r = json.loads((EV/'zoom_result.json').read_text())
print(f"Nguồn: {r['sources']} · nhất quán: {r['consistency']}")
for arm, e in r['arms'].items():
    for c in ('lr2', 'clean'):
        x = e.get(c, {})
        if 'totals' not in x:
            print(f'  {arm:9} {c:5} {x.get("status")}'); continue
        t = x['totals']
        dv = x.get('vs_F')
        print(f"  {arm:9} {c:5} H {t['hmean']*100:.2f}  TP {t['tp']} FP {t['fp']} FN {t['fn']}  tiny {t['tiny_recall']*100:.1f}"
              + (f"  Δ {dv['delta_pp']:+.2f} [{dv['ci95_pp'][0]:+.2f}; {dv['ci95_pp'][1]:+.2f}]" if dv else '')
              + (f"  {x['ms_per_image']['median']:.0f} ms" if 'ms_per_image' in x else '') + f"  ({e['forwards']} forward)")
print('\nCổng sàng lọc G2:', r['screening_gate_G2'], '→', 'ĐẠT' if r['screening_pass_G2'] else 'KHÔNG ĐẠT')
''')

md(r'''
## Ô 8 · Đóng gói kết quả

Tạo `/kaggle/working/cs431_zoom_<GPU>_<thời gian>.zip` gồm toàn bộ `cs431_out` (trừ wheel). Tải về ở khung **Output** và gửi lại Claude.
''')

code(r'''
# ===================== Ô 8 · ĐÓNG GÓI =====================
pack = W/f'cs431_zoom_{GPU}_{time.strftime("%Y%m%d-%H%M%S")}.zip'
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
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(f'{OUT}: {len(CELLS)} cells')


if __name__ == '__main__':
    main()
