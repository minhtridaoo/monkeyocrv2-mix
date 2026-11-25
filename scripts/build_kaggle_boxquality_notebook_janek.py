"""Generate notebooks/kaggle_boxquality_janek.ipynb: box-quality scorer (docs/research/janek/boxquality_spec_20261124_janek.md).
Ô 1 is the zoom notebook's Ô 1 with the dataset kind changed; Ô 2 (environment) is reused verbatim (same pins as all
earlier Kaggle runs). Re-run: python scripts/build_kaggle_boxquality_notebook_janek.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_kaggle_zoom_notebook_janek as zoomnb  # noqa: E402

OUT = ROOT/'notebooks/kaggle_boxquality_janek.ipynb'
CELLS = []


def md(text):
    CELLS.append(dict(cell_type='markdown', metadata={}, source=text.strip('\n')))


def code(text):
    CELLS.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=text.strip('\n')))


md(r'''
# CS431 · Learned box-quality scorer cho B1 (một forward) · Kaggle T4

**Câu hỏi:** đặc trưng lấy từ P-map của **cùng một forward B1** (phân bố xác suất trong box, vòng quanh box, hình dạng) có giúp xếp hạng box tốt hơn raw score không? Phải hơn cả **ngưỡng raw được chọn lại trên cùng tập**, tức không chỉ là calibration.

- **Dữ liệu:** chỉ TRAIN. Gồm 1.600 nguồn fit, 400 nguồn calibration, 10 nguồn smoke; mỗi nguồn có Clean + LR2. **Không** dùng A/B để fit, **không** mở TEST.
- **Model:** logistic L2 cố định, không quét. Đối chứng là RAW (ngưỡng raw chọn trên calibration) và SCORE_ONLY (kiểm tính hợp lệ).
- **Spec khóa trước:** `docs/research/janek/boxquality_spec_20261124_janek.md`. Calibration chỉ là sàng lọc, **không** phải xác nhận; chưa được gọi là cải tiến.

### Trước khi chạy
1. Kaggle → **New Dataset**, tải `cs431-boxquality.zip` lên (Private).
2. Import notebook này. Bật **GPU T4 x2** (hoặc T4), **Internet On**, **Add Input** chọn dataset vừa tạo.
3. Chạy **Ô 1 → Ô 4**, đọc kết luận ở Ô 4, rồi mới chạy **Ô 5 → Ô 7**.

**Thời gian: chưa đo.** Ô 2: 5–40 phút. Ô 4 in số **đo thật** cho 4.000 task. Ảnh TRAIN (549 MiB) được tải từng file từ zip TextVQA chính thức; không tải cả 7 GB.

**Dừng sau Ô 4 nếu:** một kiểm nhất quán sai, hoặc dự kiến > 3 giờ. Gửi Claude output Ô 4, không tự đổi cấu hình.

**Trạng thái:** notebook đã tạo và kiểm cú pháp ở local; các bước CPU đã chạy thử trên dữ liệu giả; **chưa chạy GPU**.
''')

CELLS.append(dict(zoomnb.CELLS[1]))
c1 = zoomnb.CELLS[2]['source']
c1 = c1.replace("get('kind') == 'zoom_pilot'", "get('kind') == 'boxquality'").replace("'zoom' in z.name.lower()", "'boxquality' in z.name.lower()")
c1 = c1.replace("Không thấy dataset cs431-zoom-pilot", "Không thấy dataset cs431-boxquality").replace("Thấy {len(found)} dataset zoom", "Thấy {len(found)} dataset boxquality")
c1 = c1.replace("print(f'Dataset: {DS} ({len(manifest[\"files\"])} file, SHA khớp) · mẫu {manifest[\"n_sources\"]} nguồn A')",
                "print(f'Dataset: {DS} ({len(manifest[\"files\"])} file, SHA khớp) · selection {manifest[\"selection_sha256\"][:12]}…')")
assert "'boxquality'" in c1 and 'selection_sha256' in c1 and 'zoom_pilot' not in c1
# Each subprocess needs both the packaged code and the separate MONKEY_V2 checkout.
c1 = c1.replace('PYTHONPATH=str(CODE)', 'PYTHONPATH=os.pathsep.join((str(CODE), str(MV2)))')
CELLS.append(dict(zoomnb.CELLS[2], source=c1))
CELLS.extend(zoomnb.CELLS[3:5])             # Ô 2 environment (verbatim)

md(r'''
## Ô 3 · Repo của Tuấn, config, B1, kiểm lại nhãn TRAIN từ nguồn gốc, ảnh smoke

- Clone `MONKEY_V2` @ `dc19b85`, vá config (chỉ đường dẫn encoder), kiểm SHA của B1.
- Tải TextOCR train JSON chính thức (280 MB), kiểm SHA `5bac8303…`. **Dựng lại nhãn TRAIN** và so với `inputs/train_rows.jsonl.gz` (phải trùng từng dòng).
- Tải 10 ảnh smoke từ zip TextVQA chính thức (CRC + SHA với `train_ids.csv`), tạo LR2 bằng `native_lr2` của Tuấn, chạy test CPU.
''')

code(r'''
# ===================== Ô 3 · DỮ LIỆU =====================
import csv, gzip
from pathlib import PureWindowsPath
import mixab.boxquality_janek as bq
importlib.reload(bq)
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
INP = DS/'inputs'
if sha256(INP/'train_ids.csv') != bq.TRAIN_IDS_SHA256:
    fail('train_ids.csv sai SHA.')
TXT = T/'textocr/TextOCR_0.1_train.json'
TXT.parent.mkdir(parents=True, exist_ok=True)
if not (TXT.is_file() and TXT.stat().st_size == bq.TEXTOCR_BYTES and sha256(TXT) == bq.TEXTOCR_TRAIN_SHA256):
    ch.run(['wget', '-c', '-q', '-O', TXT, bq.TEXTOCR_URL], LOG, title='download TextOCR train JSON (280 MB)')
if sha256(TXT) != bq.TEXTOCR_TRAIN_SHA256:
    fail('TextOCR train JSON sai SHA; chạy lại ô để tải tiếp.')
RC = T/'rows_check'
ch.run([PY, '-m', 'mixab.boxquality_janek', 'rows', '--textocr', TXT, '--train-ids', INP/'train_ids.csv', '--output', RC], LOG,
       cwd=CODE, env=ENV, title='rebuild TRAIN label rows')
a_ = [json.loads(l) for l in gzip.open(RC/'train_rows.jsonl.gz', 'rt', encoding='utf-8')]
b_ = [json.loads(l) for l in gzip.open(INP/'train_rows.jsonl.gz', 'rt', encoding='utf-8')]
if a_ != b_:
    fail('Nhãn TRAIN dựng lại khác dataset.')
print(f'Nhãn TRAIN dựng lại trùng dataset: {len(a_)} nguồn', json.loads((RC/'rows_receipt.json').read_text())['roles'])
TRAIN_CLEAN = T/'train_clean'
SEL = list(csv.DictReader(open(INP/'selection.csv', encoding='utf-8')))
tic = time.time()
r = bq.fetch_members([(x['clean_relpath'], x['clean_sha256']) for x in SEL if x['role'] == 'smoke'], TRAIN_CLEAN)
print(f"Ảnh smoke: {r} ({time.time() - tic:.1f} s)")
ch.run([PY, '-m', 'mixab.boxquality_janek', 'prepare', '--rows', INP/'train_rows.jsonl.gz', '--clean-root', TRAIN_CLEAN,
        '--prepared-root', PREPARED, '--roles', 'smoke'], LOG, cwd=CODE, env=ENV, title='prepare smoke LR2')
ch.run([PY, '-m', 'unittest', 'discover', '-s', 'tests', '-t', '.'], LOG, cwd=CODE, env=ENV, title='mixab unit tests')
print('\nÔ 3 xong.')
''')

md(r'''
## Ô 4 · Smoke 10 nguồn TRAIN (20 task) + kiểm nhất quán + chi phí thật

1. **Trích đặc trưng:** đường đặc trưng phải cho polygon và raw **trùng tuyệt đối** `capture_candidates` của Tuấn ở cả 20 task.
2. **`smokecheck` (CPU):**
   - evaluator chính thức khớp `CandidateMatcher` ở B1 `.40/.60`;
   - raw feature bằng raw score;
   - đo thời gian gán nhãn.
3. **`devcheck`:** chạy cùng đường trên 5 ảnh dev LR2 và so với dump T4 cũ. Phải trùng tuyệt đối. Bước này chỉ kiểm pipeline, **không** đọc nhãn dev.
4. In chi phí: forward, post-process thường, post-process + đặc trưng (median / p95), VRAM đỉnh, dự kiến cho 4.000 task.
''')

code(r'''
# ===================== Ô 4 · SMOKE =====================
SM = OUT/'smoke_extract'
if not (SM/'extract_summary.json').is_file():
    ch.run([PY, '-m', 'mixab.boxquality_janek', 'extract', '--rows', INP/'train_rows.jsonl.gz', '--clean-root', TRAIN_CLEAN,
            '--prepared-root', PREPARED, '--roles', 'smoke', '--model-config', CONFIG_PATH, '--weights-dir', WEIGHTS,
            '--consistency', '--shard', '20', '--output', SM], LOGS/'smoke_extract.log', cwd=CODE, env=ENV, title='smoke extract')
ch.run([PY, '-m', 'mixab.boxquality_janek', 'smokecheck', '--bundle', BUNDLE, '--rows', INP/'train_rows.jsonl.gz', '--extract-dir', SM,
        '--output', OUT/'smokecheck'], LOGS/'smokecheck.log', cwd=CODE, env=ENV, title='smokecheck')
if not (OUT/'devcheck/devcheck.json').is_file():
    ch.run([PY, '-m', 'mixab.boxquality_janek', 'devcheck', '--bundle', BUNDLE, '--sources', INP/'devcheck_sources.json',
            '--clean-root', DS/'dev_clean', '--prepared-root', T/'dev_prepared', '--model-config', CONFIG_PATH,
            '--weights-dir', WEIGHTS, '--dump', DUMP, '--output', OUT/'devcheck'], LOGS/'devcheck.log', cwd=CODE, env=ENV, title='devcheck')
s = json.loads((SM/'extract_summary.json').read_text())
k = json.loads((OUT/'smokecheck/smokecheck.json').read_text())
d = json.loads((OUT/'devcheck/devcheck.json').read_text())
print(f"Trích: {s['tasks']} task, {s['candidates']} ứng viên · nhất quán với capture_candidates: {s['consistency_checked']} task kiểm, tất cả đúng = {s['consistency_all_true']}")
print(f"ms: {s['ms']} · tỉ số post-process có đặc trưng/thường (median) {s['feature_overhead_ratio_median']:.2f} · VRAM đỉnh {s['peak_gpu_mib']:.0f} MiB")
print(f"smokecheck: evaluator chính thức khớp = {k['official_evaluator_agrees']}, raw feature đúng = {k['raw_feature_ok']}, gán nhãn {k['label_seconds_per_task_median']*1000:.1f} ms/task")
print(f"devcheck: {d['identical']}/{d['sources']} ảnh dev LR2 trùng tuyệt đối dump T4 ·", [(x['source'][:8], x['dump_candidates'], x['live_candidates'], x['max_abs_raw_diff']) for x in d['detail']])
proj = s['projected_minutes_for_4000_tasks']
print(f"\nDự kiến trích 4.000 task: {proj:.0f} phút (đo thật trên smoke, gồm kiểm SHA + post-process 2 lần ở task kiểm)")
ok = s['consistency_all_true'] and k['official_evaluator_agrees'] and k['raw_feature_ok'] and d['identical'] == d['sources']
if not ok:
    print('DỪNG: có kiểm nhất quán sai. Gửi Claude output ô này.')
elif proj > 180:
    print('DỪNG: dự kiến vượt 3 giờ. Báo Claude.')
else:
    print('Được chạy Ô 5.')
''')

md(r'''
## Ô 5 · Tải 2.000 ảnh TRAIN, dựng LR2, trích 4.000 task (GPU)

- Ảnh được tải từng file (CRC + SHA-256). Ảnh đã có và đúng SHA sẽ bỏ qua.
- Trích lưu sau mỗi 50 task vào `cs431_out/extract/`. Bị ngắt **trong cùng phiên** thì chạy lại ô này, phần xong sẽ bỏ qua.
- Cứ 50 task lại kiểm nhất quán với `capture_candidates` một lần.
- Nếu phiên sắp hết giờ: chạy Ô 7 để tải phần đã xong.
''')

code(r'''
# ===================== Ô 5 · TRÍCH TRAIN =====================
tic = time.time()
r = bq.fetch_members([(x['clean_relpath'], x['clean_sha256']) for x in SEL if x['role'] in ('fit', 'calibration')], TRAIN_CLEAN)
print(f"Ảnh fit + calibration: {r} ({(time.time() - tic)/60:.1f} phút)")
ch.run([PY, '-m', 'mixab.boxquality_janek', 'prepare', '--rows', INP/'train_rows.jsonl.gz', '--clean-root', TRAIN_CLEAN,
        '--prepared-root', PREPARED, '--roles', 'fit,calibration'], LOGS/'prepare_train.log', cwd=CODE, env=ENV, title='prepare TRAIN LR2')
EX = OUT/'extract'
ch.run([PY, '-m', 'mixab.boxquality_janek', 'extract', '--rows', INP/'train_rows.jsonl.gz', '--clean-root', TRAIN_CLEAN,
        '--prepared-root', PREPARED, '--roles', 'fit,calibration', '--model-config', CONFIG_PATH, '--weights-dir', WEIGHTS,
        '--shard', '50', '--output', EX], LOGS/f'extract_{time.strftime("%Y%m%d-%H%M%S")}.log', cwd=CODE, env=ENV, title='TRAIN extract')
s = json.loads((EX/'extract_summary.json').read_text())
print({k_: s[k_] for k_ in ('tasks', 'candidates', 'consistency_checked', 'consistency_all_true', 'ms', 'peak_gpu_mib')})
''')

md(r'''
## Ô 6 · Fit + chọn ngưỡng trên TRAIN-calibration (CPU)

- Gán nhãn theo evaluator. Fit FULL và SCORE_ONLY trên 1.600 nguồn fit.
- Chọn ngưỡng cho RAW, SCORE_ONLY và FULL trên **cùng** 400 nguồn calibration (luật khóa ở spec mục 6).
- Báo FULL − RAW, FULL − F, RAW − F kèm CI ghép cặp theo nguồn, và nhãn khóa.
- Xuất model, ngưỡng, đường cong, và bảng review 32 ảnh (24 FP điểm cao không chồng GT + 8 TP) để xem chữ thiếu nhãn.
''')

code(r'''
# ===================== Ô 6 · FIT =====================
FT = OUT/'fit'
if not (FT/'boxquality_fit.json').is_file():
    ch.run([PY, '-m', 'mixab.boxquality_janek', 'fit', '--bundle', BUNDLE, '--rows', INP/'train_rows.jsonl.gz', '--extract-dir', OUT/'extract',
            '--output', FT, '--clean-root', TRAIN_CLEAN], LOGS/'fit.log', cwd=CODE, env=ENV, title='fit + calibrate')
r = json.loads((FT/'boxquality_fit.json').read_text())
print('Ứng viên:', r['candidates'])
print('F .40/.60 (calibration):', {c: round(r['F_040_060_calibration'][c]['hmean']*100, 2) for c in ('clean', 'lr2')})
for name, v in r['chosen'].items():
    print(f"  {name:10} cut {v['cut']:.4f} · Clean {v['clean']['hmean']*100:.2f} · LR2 {v['lr2']['hmean']*100:.2f} · mean {v['mean_hmean']*100:.2f} · tiny LR2 {v['lr2']['tiny_recall']*100:.1f}")
for k_ in ('full_minus_raw', 'score_only_minus_raw', 'full_minus_F', 'raw_minus_F'):
    print(f"  {k_:22} {r[k_]['delta_pp']:+.2f} pp [{r[k_]['ci95_pp'][0]:+.2f}; {r[k_]['ci95_pp'][1]:+.2f}]")
print('\nNhãn:', r['outcome'], '(chỉ là sàng lọc trên TRAIN-calibration; không phải xác nhận, không phải cải tiến)')
''')

md(r'''
## Ô 7 · Đóng gói

Tạo `/kaggle/working/cs431_boxquality_<GPU>_<thời gian>.zip`, gồm toàn bộ `cs431_out` trừ wheel. Ảnh TRAIN nằm ở `/tmp` nên không có trong zip. Tải ở khung **Output** và gửi Claude (hoặc báo Claude tải bằng Kaggle API).
''')

code(r'''
# ===================== Ô 7 · ĐÓNG GÓI =====================
pack = W/f'cs431_boxquality_{GPU}_{time.strftime("%Y%m%d-%H%M%S")}.zip'
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
    for i, c in enumerate(CELLS):
        if c['cell_type'] == 'code':
            compile(c['source'], f'cell-{i}', 'exec')
    print(f'{OUT}: {len(CELLS)} cells')


if __name__ == '__main__':
    main()
