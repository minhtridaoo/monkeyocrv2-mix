"""Build notebooks/kaggle_refine_janek.ipynb (spec docs/research/janek/refine_spec_20261125_janek.md).
Ô 2 (environment install) is copied verbatim from the box-quality dev A notebook that already ran on Kaggle T4.
Run from the workspace root:  python scripts/build_kaggle_refine_notebook_janek.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/'notebooks/kaggle_boxquality_devA_janek.ipynb'
OUT = ROOT/'notebooks/kaggle_refine_janek.ipynb'

INTRO = """# CS431 · Learned box refinement trên B1 + box-quality v1 · Kaggle T4

Spec khóa trước: `docs/research/janek/refine_spec_20261125_janek.md`.

- **Pipeline:** ảnh → B1 frozen (một forward) → P-map + box → box-quality v1 **giữ nguyên** → MLP nhỏ sửa 4 cạnh của từng box (cùng góc, cùng score, cùng tập box) → evaluator chính thức.
- **Học bằng GT TRAIN** (1.600 nguồn fit; chọn checkpoint trên 400 nguồn calibration). Không B-Clean teacher. Không train detector.
- **A chỉ đánh giá một lần** với checkpoint đã chọn. **Không** mở B/TEST.
- Oracle CPU (dùng GT, chỉ chẩn đoán): sửa cạnh có giới hạn trên A lên Clean 81,28 / mean 80,24 (v1: 79,39 / 78,18). Model phải bắt được khoảng 40–45% lợi ích đó mới qua cổng; **không có gì bảo đảm**.

### Trước khi chạy
1. Kaggle → **New Dataset**, tải `cs431-refine.zip` lên (Private, khoảng 7 MiB).
2. Import notebook này. Bật **GPU T4 x2** (hoặc T4), **Internet On**.
3. **Add Input:** `cs431-refine` **và** dataset `cs431-boxquality-devA` đã có (giữ nguyên, không tải lại).
4. Chạy **Ô 1 → Ô 4**, đọc kết luận Ô 4, rồi mới chạy **Ô 5 → Ô 10**.

**Dừng sau Ô 4 nếu ô in `DỪNG`.** Gửi Claude output Ô 4, không tự đổi cấu hình. Thời gian thật in ở Ô 4 (chưa profile trước)."""

C1_MD = """## Ô 1 · Cấu hình, tìm hai dataset, kiểm GPU

Kiểm SHA của mọi file trong **cả hai** dataset. Dataset devA phải đúng bản đã chạy box-quality (SHA manifest). Ghim **GPU 0**."""

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

def find(kind, hint):
    found = [p for p in Path('/kaggle/input').glob('**/KAGGLE_MANIFEST.json') if json.loads(p.read_text()).get('kind') == kind]
    if not found:
        for z in [z for z in Path('/kaggle/input').glob('**/*.zip') if hint in z.name.lower()]:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(T/f'ds_{kind}')
        found = [p for p in T.glob(f'ds_{kind}/**/KAGGLE_MANIFEST.json') if json.loads(p.read_text()).get('kind') == kind]
    if len(found) != 1:
        fail(f'Thấy {len(found)} dataset kind={kind}; cần đúng một. Bấm "Add Input".')
    m = json.loads(found[0].read_text())
    bad = [rel for rel, info in m['files'].items() if not (found[0].parent/rel).is_file() or sha256(found[0].parent/rel) != info['sha256']]
    if bad:
        fail(f'Dataset {kind}: {len(bad)} file thiếu/sai SHA, ví dụ {bad[:5]}')
    print(f'Dataset {kind}: {found[0].parent} ({len(m["files"])} file, SHA khớp)')
    return found[0].parent, m, sha256(found[0])

RD, RMAN, _ = find('refine', 'refine')
DS, DMAN, DMAN_SHA = find('boxquality_devA', 'deva')
if DMAN_SHA != RMAN['devA_dataset_manifest_sha256']:
    fail('Dataset cs431-boxquality-devA khác bản đã chạy box-quality (SHA manifest).')

CODE = T/'code'
if CODE.exists():
    shutil.rmtree(CODE)
shutil.copytree(RD/'code', CODE)
code_sha = hashlib.sha256((RD/'code/CODE_MANIFEST.json').read_bytes()).hexdigest()
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
WEIGHTS, CLEAN_A = DS/'weights', DS/'clean'
MODEL, FIT = DS/'model/model_janek.npz', DS/'model/boxquality_fit_janek.json'
ENCODER = T/'encoder'
if not (ENCODER/'model.safetensors').is_file():
    shutil.copytree(DS/'encoder', ENCODER, dirs_exist_ok=True)
PREP_A, PREP_T, TRAIN_CLEAN, CONFIG_PATH = T/'prepared_A', T/'prepared_T', T/'train_clean', T/'inference_config_kaggle.py'
INP = T/'inputs'
ENV = dict(MONKEY_V2_ROOT=str(MV2), PYTHONPATH=os.pathsep.join((str(CODE), str(MV2))), CUDA_VISIBLE_DEVICES='0')
REF = RD/'ref/devA_result_janek.json'
print(f'Python {sys.version.split()[0]} · code manifest {code_sha[:12]}… · kết quả: {OUT}')"""

C3_MD = """## Ô 3 · Repo của Tuấn, config, B1, scorer đóng băng, ảnh A, nhãn TRAIN, ảnh smoke TRAIN

- Clone `MONKEY_V2` @ `dc19b85`, kiểm `bundle_manifest.json`, vá config (chỉ đường dẫn encoder), kiểm SHA B1.
- Scorer v1 đóng băng (kiểm SHA trong `load_model`).
- A: kiểm SHA 1.130 ảnh Clean, dựng 1.130 LR2 và kiểm SHA task đã khóa.
- TRAIN: khôi phục `train_rows` (kiểm SHA), tải 10 ảnh **smoke** TRAIN (CRC + SHA), dựng LR2. Chạy test CPU."""

C3 = r"""# ===================== Ô 3 · DỮ LIỆU =====================
import csv
from pathlib import PureWindowsPath
import mixab.boxquality_devA_janek as dv, mixab.boxquality_janek as bq
importlib.reload(dv); importlib.reload(bq)
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
model = dv.load_model(MODEL, FIT)
print(f"Scorer v1 đóng băng: cut_full {model['cut_full']!r} · {len(model['features'])} đặc trưng")
ch.run([PY, '-m', 'mixab.boxquality_devA_janek', 'prepare', '--bundle', BUNDLE, '--clean-root', CLEAN_A, '--prepared-root', PREP_A,
        '--subset', 'all'], LOG, cwd=CODE, env=ENV, title='A: check Clean SHA, build + check LR2')
INP.mkdir(parents=True, exist_ok=True)
shutil.copyfile(RD/'inputs/train_rows.gzip.bin', INP/'train_rows.jsonl.gz')
for f_ in ('selection.csv', 'train_ids.csv'):
    shutil.copyfile(RD/'inputs'/f_, INP/f_)
if sha256(INP/'train_rows.jsonl.gz') != RMAN['train_rows_sha256'] or sha256(INP/'selection.csv') != RMAN['files']['inputs/selection.csv']['sha256']:
    fail('Nhãn TRAIN hoặc selection sai SHA.')
ROWS = INP/'train_rows.jsonl.gz'
SEL = list(csv.DictReader(open(INP/'selection.csv', encoding='utf-8')))
r_ = bq.fetch_members([(x['clean_relpath'], x['clean_sha256']) for x in SEL if x['role'] == 'smoke'], TRAIN_CLEAN)
print('Ảnh smoke TRAIN:', r_)
ch.run([PY, '-m', 'mixab.boxquality_janek', 'prepare', '--rows', ROWS, '--clean-root', TRAIN_CLEAN, '--prepared-root', PREP_T,
        '--roles', 'smoke'], LOG, cwd=CODE, env=ENV, title='prepare smoke LR2')
ch.run([PY, '-m', 'unittest', 'discover', '-s', 'tests', '-t', '.'], LOG, cwd=CODE, env=ENV, title='mixab unit tests')
print('\nÔ 3 xong.')"""

C4_MD = """## Ô 4 · Smoke: 10 nguồn smoke TRAIN (20 task) + 5 nguồn A (10 task), kiểm ở MỌI task

1. **Crop không đổi output:** mỗi task chạy thêm `bq.run_task` (forward thứ hai) và phải cho record v1 **trùng tuyệt đối**; post-process phải trùng `capture_candidates` của Tuấn.
2. **Tọa độ:** quad mạng tính lại từ polygon cuối (qua restore + scale_factor) phải trùng quad của post-processor (< 0,01 px), và nằm trong ảnh (lệch ≤ 2 px).
3. In thời gian thật mỗi giai đoạn và dự kiến cho 4.000 task TRAIN + 2.260 task A.

**Dừng** nếu một kiểm sai, hoặc dự kiến > 4 giờ. Số Hmean không in ở ô này."""

C4 = r"""# ===================== Ô 4 · SMOKE =====================
EXT = lambda split, out, extra: [PY, '-m', 'mixab.refine_cmd_janek', 'extract', '--split', split, '--model-config', CONFIG_PATH,
                                 '--weights-dir', WEIGHTS, '--output', out, *extra]
ST = OUT/'smoke_train'
if not (ST/'extract_summary.json').is_file():
    ch.run(EXT('train', ST, ['--rows', ROWS, '--roles', 'smoke', '--clean-root', TRAIN_CLEAN, '--prepared-root', PREP_T,
                            '--consistency', '--shard', '20']), LOGS/'smoke_train.log', cwd=CODE, env=ENV, title='smoke TRAIN extract')
SA = OUT/'smoke_A'
if not (SA/'extract_summary.json').is_file():
    ch.run(EXT('devA', SA, ['--bundle', BUNDLE, '--clean-root', CLEAN_A, '--prepared-root', PREP_A, '--limit', '5',
                           '--consistency', '--shard', '10']), LOGS/'smoke_A.log', cwd=CODE, env=ENV, title='smoke A extract')
ok, per_task = True, []
for name, d in (('TRAIN smoke', ST), ('A smoke', SA)):
    s = json.loads((d/'extract_summary.json').read_text())
    # Read per-task evidence, including summaries created by the older dataset.
    records = bq.load_records(d)
    s['consistency_all_true'] = (len(records) == s['records'] and len(records) > 0
                                 and all(r.get('consistency') is True for r in records))
    print(f"{name}: {s['records']} task · {s['candidates']} ứng viên · kiểm {s['n_checks']} task, trùng v1 = {s['checks_all_identical']}, "
          f"post-process = capture_candidates: {s['consistency_all_true']} · lệch quad mạng {s['max_net_quad_dev_px']:.2e} px · "
          f"ngoài ảnh {s['max_outside_px']:.2f} px · VRAM {s['peak_gpu_mib']:.0f} MiB")
    print('   ms median:', {k_: round(v['median'], 1) for k_, v in s['ms'].items()})
    ok &= s['checks_all_identical'] and s['consistency_all_true'] and s['n_checks'] == s['records'] and s['max_outside_px'] <= 2
    per_task.append(sum(v['median'] for v in s['ms'].values())/1000 + .05)
sec = max(per_task)
proj = sec*(4000*(1 + 1/25) + 2260*(1 + 1/25))/60
print(f'\nDự kiến trích TRAIN 4.000 + A 2.260 task: khoảng {proj:.0f} phút (từ median đo thật + 50 ms đọc ảnh/task; chưa gồm tải 2.000 ảnh TRAIN)')
if not ok:
    print('DỪNG: có kiểm hợp lệ sai. Gửi Claude output ô này.')
elif proj > 240:
    print('DỪNG: dự kiến vượt 4 giờ. Báo Claude.')
else:
    print('Được chạy Ô 5.')"""

C5_MD = """## Ô 5 · Tải 2.000 ảnh TRAIN, dựng LR2, trích 4.000 task TRAIN (GPU)

- Ảnh tải từng file (CRC + SHA-256); ảnh đã có thì bỏ qua.
- Lưu sau mỗi 50 task (`cs431_out/extract_train/`, gồm `.jsonl.gz` + `.npz` crop). Bị ngắt **trong cùng phiên** thì chạy lại ô này.
- Cứ 25 task kiểm lại một lần như Ô 4."""

C5 = r"""# ===================== Ô 5 · TRÍCH TRAIN =====================
tic = time.time()
r_ = bq.fetch_members([(x['clean_relpath'], x['clean_sha256']) for x in SEL if x['role'] in ('fit', 'calibration')], TRAIN_CLEAN)
print(f"Ảnh fit + calibration: {r_} ({(time.time() - tic)/60:.1f} phút)")
ch.run([PY, '-m', 'mixab.boxquality_janek', 'prepare', '--rows', ROWS, '--clean-root', TRAIN_CLEAN, '--prepared-root', PREP_T,
        '--roles', 'fit,calibration'], LOGS/'prepare_train.log', cwd=CODE, env=ENV, title='prepare TRAIN LR2')
ET = OUT/'extract_train'
ch.run(EXT('train', ET, ['--rows', ROWS, '--roles', 'fit,calibration', '--clean-root', TRAIN_CLEAN, '--prepared-root', PREP_T]),
       LOGS/f'extract_train_{time.strftime("%Y%m%d-%H%M%S")}.log', cwd=CODE, env=ENV, title='TRAIN extract')
s = json.loads((ET/'extract_summary.json').read_text())
print({k_: s[k_] for k_ in ('tasks', 'records', 'candidates', 'n_checks', 'checks_all_identical', 'max_net_quad_dev_px', 'ms', 'peak_gpu_mib')})
if not (s['records'] == 4000 and s['checks_all_identical']):
    print('DỪNG: chưa đủ 4.000 task hoặc có kiểm sai. Chạy lại ô (resume) hoặc gửi Claude.')"""

C6_MD = """## Ô 6 · Trích split A (2.260 task, GPU)

Chỉ trích (P-map crop + record v1). **Chưa** đánh giá; A được dùng đúng một lần ở Ô 8 sau khi đã chọn checkpoint trên TRAIN."""

C6 = r"""# ===================== Ô 6 · TRÍCH A =====================
EA = OUT/'extract_A'
ch.run(EXT('devA', EA, ['--bundle', BUNDLE, '--clean-root', CLEAN_A, '--prepared-root', PREP_A]),
       LOGS/f'extract_A_{time.strftime("%Y%m%d-%H%M%S")}.log', cwd=CODE, env=ENV, title='A extract')
s = json.loads((EA/'extract_summary.json').read_text())
print({k_: s[k_] for k_ in ('tasks', 'records', 'candidates', 'n_checks', 'checks_all_identical', 'max_net_quad_dev_px', 'ms', 'peak_gpu_mib')})
if not (s['records'] == 2260 and s['checks_all_identical']):
    print('DỪNG: chưa đủ 2.260 task hoặc có kiểm sai. Chạy lại ô (resume) hoặc gửi Claude.')"""

C7_MD = """## Ô 7 · Train bộ sửa (CPU) + chọn checkpoint trên TRAIN-calibration

- Train trên fold **fit** (cặp box v1 ↔ GT ghép một-một; detector frozen). Công thức khóa trong `mixab/refine_janek.py`.
- Chọn trong e05…e30 theo mean Hmean chính thức trên 400 nguồn **calibration**; phải hơn v1 (offset 0). Không thì `NO_SIGNAL_ON_CALIBRATION` và **không** chạy Ô 8."""

C7 = r"""# ===================== Ô 7 · TRAIN + CHỌN =====================
TR = OUT/'train'
COMMON = ['--bundle', BUNDLE, '--model', MODEL, '--fit', FIT]
if not (TR/'train.json').is_file():
    ch.run([PY, '-m', 'mixab.refine_cmd_janek', 'train', *COMMON, '--rows', ROWS, '--extract-dir', ET, '--output', TR],
           LOGS/'train.log', cwd=CODE, env=ENV, title='train refiner (TRAIN fit)')
t_ = json.loads((TR/'train.json').read_text())
print(f"cặp train {t_['pairs']} (TP {t_['tp_pairs']}) · {t_['seconds']:.0f} s · loss", [round(h['loss'], 5) for h in t_['history'][::5]])
SELJ = OUT/'selection.json'
if not SELJ.is_file():
    ch.run([PY, '-m', 'mixab.refine_cmd_janek', 'select', *COMMON, '--rows', ROWS, '--extract-dir', ET, '--train-dir', TR,
            '--output', SELJ], LOGS/'select.log', cwd=CODE, env=ENV, title='select on TRAIN-calibration')
S_ = json.loads(SELJ.read_text())
for n_, v_ in S_['table'].items():
    print(f"  {n_:4} mean {v_['mean']:.3f} · Clean {v_['clean']:.3f} · LR2 {v_['lr2']:.3f}")
print('Chọn:', S_['chosen'], S_['chosen_minus_v1'], '·', S_['outcome'])
if S_['chosen']:
    d_ = S_['diag'][S_['chosen']]
    print('  GT cứu/mất:', d_['gained'], '/', d_['lost'], '· fallback:', d_['status'])
    print('  độ dịch theo loại box (tỉ lệ > 0,10 cạnh):', {k_: round(v_['over_0_10']/max(v_['n'], 1), 3) for k_, v_ in d_['moved'].items()})
else:
    print('DỪNG: không có tín hiệu trên calibration. Không chạy Ô 8. Chạy Ô 10 để đóng gói, gửi Claude.')"""

C8_MD = """## Ô 8 · Một lượt đánh giá A đóng băng (CPU)

F (B1 `.40/.60`), v1, v1 + refinement. Kiểm F và v1 trùng `devA_result`. Nhãn khóa theo spec §5. **Không chỉnh gì sau khi xem.**"""

C8 = r"""# ===================== Ô 8 · ĐÁNH GIÁ A =====================
EV = OUT/'eval_A'
if not (EV/'refine_devA.json').is_file():
    ch.run([PY, '-m', 'mixab.refine_cmd_janek', 'evaluate', *COMMON, '--extract-dir', EA, '--train-dir', TR, '--selection', SELJ,
            '--devA-ref', REF, '--output', EV], LOGS/'evaluate.log', cwd=CODE, env=ENV, title='evaluate A (once)')
R_ = json.loads((EV/'refine_devA.json').read_text())
print('Trùng devA_result (F, v1):', R_['reproduces_devA_result'])
for n_ in ('F', 'v1', 'v1R'):
    a_ = R_['arms'][n_]
    print(f"{n_:4} Clean {a_['clean']['hmean']*100:.2f} (TP {a_['clean']['tp']} FP {a_['clean']['fp']} FN {a_['clean']['fn']}, tiny {a_['clean']['tiny_recall']*100:.1f}) · "
          f"LR2 {a_['lr2']['hmean']*100:.2f} (TP {a_['lr2']['tp']} FP {a_['lr2']['fp']} FN {a_['lr2']['fn']}, tiny {a_['lr2']['tiny_recall']*100:.1f}) · mean {a_['mean_hmean']*100:.2f}")
for k_, d_ in R_['differences'].items():
    print(f"{k_:7} mean {d_['mean']['delta_pp']:+.2f} [{d_['mean']['ci95_pp'][0]:+.2f}; {d_['mean']['ci95_pp'][1]:+.2f}] · Clean {d_['clean']['delta_pp']:+.2f} · LR2 {d_['lr2']['delta_pp']:+.2f}")
print('GT cứu/mất:', R_['diag']['gained'], '/', R_['diag']['lost'], '· fallback:', R_['diag']['status'])
print('Cổng:', R_['gates'], R_['gate_values'])
print('\nNHÃN:', R_['outcome'])"""

C9_MD = """## Ô 9 · Latency + bộ nhớ (GPU)

100 nguồn A × Clean/LR2, 2 lượt, thứ tự 3 cánh xoay vòng. Phần sửa gồm: chuyển P-map về CPU, crop, MLP (CPU), dựng quad. **Không chạy việc khác** trong lúc đo."""

C9 = r"""# ===================== Ô 9 · LATENCY =====================
LT = OUT/'latency'
if S_['chosen'] and not (LT/'latency.json').is_file():
    ch.run([PY, '-m', 'mixab.refine_cmd_janek', 'latency', *COMMON, '--clean-root', CLEAN_A, '--prepared-root', PREP_A,
            '--model-config', CONFIG_PATH, '--weights-dir', WEIGHTS, '--train-dir', TR, '--selection', SELJ, '--output', LT],
           LOGS/'latency.log', cwd=CODE, env=ENV, title='latency F / v1 / v1R')
if (LT/'latency.json').is_file():
    L_ = json.loads((LT/'latency.json').read_text())
    for n_, v_ in L_['ms'].items():
        print(f"{n_:4} median {v_['total']['median']:.1f} ms · p95 {v_['total']['p95']:.1f} · phần sửa median {v_['refine']['median']:.1f} ms · VRAM {L_['memory'][n_]['peak_allocated_mib']:.0f} MiB")
    print('v1R/v1 median', round(L_['v1R_over_v1']['ratio_of_medians'], 3), L_['v1R_over_v1']['ratio_of_medians_ci95'],
          '· v1R/F', round(L_['v1R_over_F']['ratio_of_medians'], 3))"""

C10_MD = """## Ô 10 · Đóng gói

Tạo `/kaggle/working/cs431_refine_<GPU>_<thời gian>.zip` gồm toàn bộ `cs431_out` (bản ghi, crop, checkpoint, kết quả, log). Tải ở khung **Output** và gửi Claude (hoặc báo Claude tải bằng Kaggle API)."""

C10 = r"""# ===================== Ô 10 · ĐÓNG GÓI =====================
pack = W/f'cs431_refine_{GPU}_{time.strftime("%Y%m%d-%H%M%S")}.zip'
with zipfile.ZipFile(pack, 'w', compression=zipfile.ZIP_STORED) as zf:
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
             dict(install_md), dict(install, outputs=[], execution_count=None)]
    for md, code in ((C3_MD, C3), (C4_MD, C4), (C5_MD, C5), (C6_MD, C6), (C7_MD, C7), (C8_MD, C8), (C9_MD, C9), (C10_MD, C10)):
        cells += [cell('markdown', md), cell('code', code)]
    nb = dict(src, cells=cells)
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(OUT, len(cells), 'cells')


if __name__ == '__main__':
    main()
