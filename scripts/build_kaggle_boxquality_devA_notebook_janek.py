"""Build notebooks/kaggle_boxquality_devA_janek.ipynb (spec docs/research/janek/boxquality_devA_spec_20261125_janek.md).
Ô 2 (environment install) is copied verbatim from the box-quality notebook that already ran on Kaggle T4.
Run from the workspace root:  python scripts/build_kaggle_boxquality_devA_notebook_janek.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/'notebooks/kaggle_boxquality_janek_fixed.ipynb'
OUT = ROOT/'notebooks/kaggle_boxquality_devA_janek.ipynb'

INTRO = """# CS431 · Box-quality scorer đã khóa · đánh giá bổ sung trên development A · Kaggle T4

**Đây là quyết định của user sau khi xem pilot, không phải vì pilot đã vượt cổng.** Pilot TRAIN-calibration giữ nhãn **`POSITIVE_BELOW_GATE`**. Spec bổ sung (viết trước khi chạy): `docs/research/janek/boxquality_devA_spec_20261125_janek.md`.

- **Không fit, không chọn lại:** model, normalizer, 13 đặc trưng, pool ứng viên và hai ngưỡng (`cut_full`, `cut_raw`) lấy nguyên từ pilot, có kiểm SHA.
- **Dữ liệu:** split A (1.130 nguồn), Clean + native LR2, nhãn dev đã khóa, evaluator chính thức. **Không** mở B, **không** mở TEST.
- **So sánh:** B1 `.40/.60` (F), RAW đã khóa và FULL trên cùng ảnh, cùng evaluator. Báo đủ Hmean, TP/FP/FN, recall theo kích thước, CI ghép cặp theo nguồn, cổng nhóm, latency và bộ nhớ.

### Trước khi chạy
1. Kaggle → **New Dataset**, tải `cs431-boxquality-devA.zip` lên (Private).
2. Import notebook này. Bật **GPU T4 x2** (hoặc T4), **Internet On**, **Add Input** chọn dataset vừa tạo.
3. Chạy **Ô 1 → Ô 4**, đọc kết luận ở Ô 4, rồi mới chạy **Ô 5 → Ô 8**.

**Thời gian (ước lượng, chưa đo):**
- Ô 2: 5–40 phút.
- Ô 5: khoảng 20 phút, ngoại suy từ pilot (0,51 s/task × 2.260 task); Ô 4 in số đo thật.
- Ô 6: khoảng 6–10 phút.
- Ô 7: vài phút CPU.

**Dừng sau Ô 4 nếu ô in `DỪNG`.** Gửi Claude output Ô 4, không tự đổi cấu hình.

**Trạng thái:** đã kiểm cú pháp ở local. `evaluate` đã chạy thử trên bản ghi giả dựng từ dump T4 (đặc trưng giả, số vô nghĩa). **Chưa chạy GPU.**"""

C1_MD = """## Ô 1 · Cấu hình, tìm dataset, kiểm GPU

Kiểm SHA của mọi file trong dataset. Ghim **GPU 0** (`CUDA_VISIBLE_DEVICES=0`), vì code của Tuấn yêu cầu đúng một GPU."""

C1 = r"""# ===================== Ô 1 · CẤU HÌNH =====================
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

KIND = 'boxquality_devA'
found = [p for p in Path('/kaggle/input').glob('**/KAGGLE_MANIFEST.json') if json.loads(p.read_text()).get('kind') == KIND]
if not found:
    zips = [z for z in Path('/kaggle/input').glob('**/*.zip') if 'devA' in z.name or 'deva' in z.name.lower()]
    if not zips:
        fail('Không thấy dataset cs431-boxquality-devA. Bấm "Add Input" và chọn dataset đó.')
    with zipfile.ZipFile(zips[0]) as zf:
        zf.extractall(T/'dataset')
    found = [p for p in (T/'dataset').glob('**/KAGGLE_MANIFEST.json') if json.loads(p.read_text()).get('kind') == KIND]
if len(found) != 1:
    fail(f'Thấy {len(found)} dataset {KIND}; chỉ gắn một.')
DS = found[0].parent
manifest = json.loads(found[0].read_text())
bad = []
for rel, info in manifest['files'].items():
    path = DS/rel
    if not path.is_file():
        bad.append(dict(file=rel, reason='MISSING', expected_sha256=info['sha256']))
    elif sha256(path) != info['sha256']:
        bad.append(dict(file=rel, reason='SHA_MISMATCH', expected_sha256=info['sha256'], bytes=path.stat().st_size))
if bad:
    print(json.dumps(bad[:20], indent=2), f'... ({len(bad)} file lỗi)' if len(bad) > 20 else '')
    fail('Dataset thiếu hoặc thay đổi file; xem chi tiết phía trên.')
print(f'Dataset: {DS} ({len(manifest["files"])} file, SHA khớp) · split {manifest["split"]} · {manifest["sources"]} nguồn')

CODE = T/'code'
if CODE.exists():
    shutil.rmtree(CODE)
shutil.copytree(DS/'code', CODE)
code_sha = hashlib.sha256((DS/'code/CODE_MANIFEST.json').read_bytes()).hexdigest()
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
import importlib
import mixab.colab_helpers_janek as ch
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
WEIGHTS, CLEAN = DS/'weights', DS/'clean'
DUMP_LR2, DUMP_CLEAN = DS/'dump_lr2', DS/'dump_clean'
MODEL, FIT = DS/'model/model_janek.npz', DS/'model/boxquality_fit_janek.json'
ENCODER = T/'encoder'
if not (ENCODER/'model.safetensors').is_file():
    shutil.copytree(DS/'encoder', ENCODER, dirs_exist_ok=True)
PREPARED, CONFIG_PATH = T/'prepared', T/'inference_config_kaggle.py'
ENV = dict(MONKEY_V2_ROOT=str(MV2), PYTHONPATH=os.pathsep.join((str(CODE), str(MV2))), CUDA_VISIBLE_DEVICES='0')
print(f'Python {sys.version.split()[0]} · code manifest {code_sha[:12]}… · kết quả: {OUT}')"""

C3_MD = """## Ô 3 · Repo của Tuấn, config, B1, scorer đóng băng, ảnh A

- Clone `MONKEY_V2` @ `dc19b85`, kiểm `bundle_manifest.json`, vá config (chỉ đường dẫn encoder), kiểm SHA của B1.
- Kiểm **scorer đóng băng**: SHA `model_janek.npz` = `81867a49…`, SHA `boxquality_fit_janek.json` = `76386825…`, hai ngưỡng trùng file fit. Sai thì dừng.
- Kiểm SHA 1.130 ảnh Clean của A. Dựng 1.130 ảnh LR2 bằng `native_lr2` của Tuấn; mỗi ảnh phải trùng SHA của task đã khóa.
- Chạy test CPU."""

C3 = r"""# ===================== Ô 3 · DỮ LIỆU =====================
from pathlib import PureWindowsPath
import mixab.boxquality_devA_janek as dv
importlib.reload(dv)
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
if sha256(WEIGHTS/PureWindowsPath(b1['checkpoint']).name) != b1['checkpoint_sha256']:
    fail('alpha_1p00.pth (B1) sai SHA.')
model = dv.load_model(MODEL, FIT)      # raises if the scorer or its cutoffs differ from the pilot
print(f"Scorer đóng băng: cut_full {model['cut_full']!r} · cut_raw {model['cut_raw']!r} · {len(model['features'])} đặc trưng")
ch.run([PY, '-m', 'mixab.boxquality_devA_janek', 'prepare', '--bundle', BUNDLE, '--clean-root', CLEAN, '--prepared-root', PREPARED,
        '--subset', 'all'], LOG, cwd=CODE, env=ENV, title='A: check Clean SHA, build + check LR2')
ch.run([PY, '-m', 'unittest', 'discover', '-s', 'tests', '-t', '.'], LOG, cwd=CODE, env=ENV, title='mixab unit tests')
print('\nÔ 3 xong.')"""

C4_MD = """## Ô 4 · Smoke 10 nguồn A (20 task): kiểm hợp lệ + thời gian thật

1. **Trích:** đường đặc trưng phải trùng tuyệt đối `capture_candidates` của Tuấn ở **mọi** task (lệch thì lệnh dừng với lỗi).
2. **Đánh giá thử 10 nguồn:**
   - evaluator chính thức phải trùng đường nhanh ở mọi task × cánh;
   - in số task có ứng viên trùng tuyệt đối dump T4 (chỉ báo).
3. In thời gian dự kiến cho đủ A.

**Dừng** nếu một kiểm sai, hoặc dự kiến > 3 giờ."""

C4 = r"""# ===================== Ô 4 · SMOKE =====================
SM = OUT/'smoke_extract'
if not (SM/'extract_summary.json').is_file():
    ch.run([PY, '-m', 'mixab.boxquality_devA_janek', 'extract', '--bundle', BUNDLE, '--clean-root', CLEAN, '--prepared-root', PREPARED,
            '--subset', 'smoke', '--model-config', CONFIG_PATH, '--weights-dir', WEIGHTS, '--shard', '20', '--output', SM],
           LOGS/'smoke_extract.log', cwd=CODE, env=ENV, title='smoke extract (A)')
SE = OUT/'smoke_eval'
if not (SE/'devA_result.json').is_file():
    ch.run([PY, '-m', 'mixab.boxquality_devA_janek', 'evaluate', '--bundle', BUNDLE, '--extract-dir', SM, '--model', MODEL, '--fit', FIT,
            '--dump-lr2', DUMP_LR2, '--dump-clean', DUMP_CLEAN, '--subset', 'smoke', '--output', SE],
           LOGS/'smoke_eval.log', cwd=CODE, env=ENV, title='smoke evaluate (checks only)')
s = json.loads((SM/'extract_summary.json').read_text())
e = json.loads((SE/'devA_result.json').read_text())
print(f"Trích: {s['records']} task, {s['candidates']} ứng viên · nhất quán với capture_candidates: {s['consistency_checked']} task kiểm, tất cả đúng = {s['consistency_all_true']}")
print(f"ms: {s['ms']} · VRAM đỉnh {s['peak_gpu_mib']:.0f} MiB")
print(f"Evaluator chính thức = đường nhanh: {e['checks']['fast_equals_official']} · task {e['checks']['tasks']}")
print('Ứng viên trùng dump T4 (chỉ báo):', {c: f"{v['identical_candidates']}/{v['tasks']}" for c, v in e['live_vs_t4_dump'].items()})
proj = s['projected_minutes_for_all_A']
print(f"\nDự kiến trích đủ A (2.260 task): {proj:.0f} phút (đo thật trên smoke)")
ok = s['consistency_all_true'] and e['checks']['fast_equals_official'] and e['checks']['tasks'] == 20 and s['records'] == 20
if not ok:
    print('DỪNG: có kiểm hợp lệ sai. Gửi Claude output ô này.')
elif proj > 180:
    print('DỪNG: dự kiến vượt 3 giờ. Báo Claude.')
else:
    print('Được chạy Ô 5. (Số Hmean của 10 nguồn smoke không có ý nghĩa; không đọc.)')"""

C5_MD = """## Ô 5 · Trích đủ split A (2.260 task, GPU)

- Lưu sau mỗi 50 task vào `cs431_out/extract/`. Bị ngắt **trong cùng phiên** thì chạy lại ô này, phần đã xong sẽ bỏ qua.
- Kiểm nhất quán với `capture_candidates` ở **mọi** task.
- Nếu phiên sắp hết giờ: chạy Ô 8 để tải phần đã xong."""

C5 = r"""# ===================== Ô 5 · TRÍCH A =====================
EX = OUT/'extract'
ch.run([PY, '-m', 'mixab.boxquality_devA_janek', 'extract', '--bundle', BUNDLE, '--clean-root', CLEAN, '--prepared-root', PREPARED,
        '--subset', 'all', '--model-config', CONFIG_PATH, '--weights-dir', WEIGHTS, '--shard', '50', '--output', EX],
       LOGS/f'extract_{time.strftime("%Y%m%d-%H%M%S")}.log', cwd=CODE, env=ENV, title='A extract')
s = json.loads((EX/'extract_summary.json').read_text())
print({k_: s[k_] for k_ in ('tasks', 'records', 'candidates', 'consistency_checked', 'consistency_all_true', 'ms', 'peak_gpu_mib')})
if not (s['records'] == 2260 and s['consistency_all_true']):
    print('DỪNG: chưa đủ 2.260 task hoặc có task lệch. Chạy lại ô (resume) hoặc gửi Claude.')"""

C6_MD = """## Ô 6 · Latency công bằng (GPU)

- 100 nguồn A (theo hash) × Clean/LR2 = 200 ảnh đo, cộng 10 ảnh warm-up khác. 2 lượt, xen kẽ thứ tự cánh.
- **F:** decode → B1 → post-process `mask_040`, `min_text_score = .60` (như triển khai).
- **FULL:** decode → B1 → post-process có 13 đặc trưng → scorer → ngưỡng. Đo nguyên bản cài đặt, **chưa tối ưu**.
- Báo median, p95, mean, từng giai đoạn, theo điều kiện; tỉ số FULL/F kèm CI; VRAM đỉnh mỗi cánh; RSS tiến trình.
- Kiểm: box của cả hai cánh trong lượt đo phải trùng box của lượt trích ở Ô 5.
- **Không chạy việc khác** trên máy trong lúc đo."""

C6 = r"""# ===================== Ô 6 · LATENCY =====================
LT = OUT/'latency'
if not (LT/'latency.json').is_file():
    ch.run([PY, '-m', 'mixab.boxquality_devA_janek', 'latency', '--bundle', BUNDLE, '--clean-root', CLEAN, '--prepared-root', PREPARED,
            '--model-config', CONFIG_PATH, '--weights-dir', WEIGHTS, '--model', MODEL, '--fit', FIT, '--extract-dir', OUT/'extract',
            '--passes', '2', '--output', LT], LOGS/'latency.log', cwd=CODE, env=ENV, title='latency F vs FULL')
L = json.loads((LT/'latency.json').read_text())
for arm in ('F', 'FULL'):
    t = L['ms'][arm]['total']
    print(f"{arm:5} median {t['median']:.1f} ms · p95 {t['p95']:.1f} ms · mean {t['mean']:.1f} ms · VRAM đỉnh {L['memory'][arm]['peak_allocated_mib']:.0f} MiB")
p = L['paired']
print(f"FULL/F median {p['ratio_of_medians']:.3f} [{p['ratio_of_medians_ci95'][0]:.3f}; {p['ratio_of_medians_ci95'][1]:.3f}] · p95 {p['ratio_of_p95']:.3f} · thêm {p['mean_extra_ms']:.1f} ms/ảnh")
print('Kiểm box:', L['output_check'], '· RSS tiến trình', round(L['process_max_rss_mib']), 'MiB')"""

C7_MD = """## Ô 7 · Đánh giá A (CPU) và nhãn khóa

- Evaluator chính thức cho F, RAW, FULL ở mọi task; đường nhanh phải trùng.
- Mốc cổng nhóm lấy từ dump T4 của cùng ảnh.
- CI bootstrap ghép cặp theo nguồn (2.000 lần, seed 431).
- Nhãn theo spec §5: `GROUP_GATE_PASS` / `SMALL_USEFUL` / `POSITIVE_NOT_USEFUL` / `INCONCLUSIVE` / `NO_BENEFIT` / `INVALID_RUN`. **Không chỉnh gì sau khi xem.**"""

C7 = r"""# ===================== Ô 7 · ĐÁNH GIÁ =====================
EV = OUT/'eval'
if not (EV/'devA_result.json').is_file():
    ch.run([PY, '-m', 'mixab.boxquality_devA_janek', 'evaluate', '--bundle', BUNDLE, '--extract-dir', OUT/'extract', '--model', MODEL,
            '--fit', FIT, '--dump-lr2', DUMP_LR2, '--dump-clean', DUMP_CLEAN, '--latency', OUT/'latency/latency.json',
            '--subset', 'all', '--output', EV], LOGS/'evaluate.log', cwd=CODE, env=ENV, title='evaluate A')
r = json.loads((EV/'devA_result.json').read_text())
print('Kiểm:', {k_: v for k_, v in r['checks'].items() if k_ != 'fast_mismatch'})
for arm in ('F', 'RAW', 'FULL'):
    a_ = r['arms'][arm]
    print(f"{arm:5} Clean H {a_['clean']['hmean']*100:.2f} (TP {a_['clean']['tp']} FP {a_['clean']['fp']} FN {a_['clean']['fn']}, tiny {a_['clean']['tiny_recall']*100:.1f}%) · "
          f"LR2 H {a_['lr2']['hmean']*100:.2f} (TP {a_['lr2']['tp']} FP {a_['lr2']['fp']} FN {a_['lr2']['fn']}, tiny {a_['lr2']['tiny_recall']*100:.1f}%) · mean {a_['mean_hmean']*100:.2f}")
for k_, d_ in r['differences'].items():
    print(f"{k_:14} mean {d_['mean']['delta_pp']:+.2f} [{d_['mean']['ci95_pp'][0]:+.2f}; {d_['mean']['ci95_pp'][1]:+.2f}] · "
          f"Clean {d_['clean']['delta_pp']:+.2f} [{d_['clean']['ci95_pp'][0]:+.2f}; {d_['clean']['ci95_pp'][1]:+.2f}] · "
          f"LR2 {d_['lr2']['delta_pp']:+.2f} [{d_['lr2']['ci95_pp'][0]:+.2f}; {d_['lr2']['ci95_pp'][1]:+.2f}]")
print('Cổng nhóm:', r['group_gates'], '· model đơn tốt nhất:', r['group_gate_detail']['best_single'])
print('Ứng viên trùng dump T4 (chỉ báo):', {c: f"{v['identical_candidates']}/{v['tasks']}" for c, v in r['live_vs_t4_dump'].items()})
print(f"\nNHÃN: {r['outcome']} {r['small_useful_failed'] or ''} · tổng quát hóa (FULL − RAW): {r['generalization_full_minus_raw']['reading']}")
print('(Mô tả trên A đã xem nhiều lần; không phải xác nhận. Không chỉnh gì sau khi xem.)')"""

C8_MD = """## Ô 8 · Đóng gói

Tạo `/kaggle/working/cs431_boxquality_devA_<GPU>_<thời gian>.zip`, gồm toàn bộ `cs431_out` trừ wheel. Tải ở khung **Output** và gửi Claude (hoặc báo Claude tải bằng Kaggle API)."""

C8 = r"""# ===================== Ô 8 · ĐÓNG GÓI =====================
pack = W/f'cs431_boxquality_devA_{GPU}_{time.strftime("%Y%m%d-%H%M%S")}.zip'
with zipfile.ZipFile(pack, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.suffix != '.whl':
            zf.write(p, p.relative_to(OUT).as_posix())
    zf.writestr('code_manifest_sha256.txt', code_sha + '\n')
print('Đã đóng gói:', pack, f'({pack.stat().st_size/2**20:.1f} MiB)')"""


def cell(kind, text):
    c = dict(cell_type=kind, metadata={}, source=text.splitlines(keepends=True))
    if kind == 'code':
        compile(text, '<notebook>', 'exec')
        c.update(outputs=[], execution_count=None)
    return c


def main():
    src = json.loads(SRC.read_text(encoding='utf-8'))
    install_md, install = src['cells'][3], src['cells'][4]
    assert ''.join(install['source']).startswith('# ===================== Ô 2 · CÀI MÔI TRƯỜNG'), 'unexpected source notebook'
    cells = [cell('markdown', INTRO), cell('markdown', C1_MD), cell('code', C1),
             dict(install_md), dict(install, outputs=[], execution_count=None),
             cell('markdown', C3_MD), cell('code', C3), cell('markdown', C4_MD), cell('code', C4),
             cell('markdown', C5_MD), cell('code', C5), cell('markdown', C6_MD), cell('code', C6),
             cell('markdown', C7_MD), cell('code', C7), cell('markdown', C8_MD), cell('code', C8)]
    nb = dict(src, cells=cells)
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(OUT, len(cells), 'cells')


if __name__ == '__main__':
    main()
