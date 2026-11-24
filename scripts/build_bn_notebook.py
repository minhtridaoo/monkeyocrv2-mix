"""Generate notebooks/colab_bn_pilot_stage0.ipynb (E-BN pilot, Stage 0 only).

Cells 1-4 (environment, data/weights, doctor) are copied verbatim from scripts/build_colab_notebook.py,
which ran successfully on Colab A100 on 20/11; only the closing hint of cell 4 is redirected.
The Mix A-core v1 notebook and its results are not modified.
Run from the workspace root:  python scripts/build_bn_notebook.py
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'notebooks/colab_bn_pilot_stage0.ipynb'


def base_cells():
    spec = importlib.util.spec_from_file_location('build_colab_notebook', ROOT/'scripts/build_colab_notebook.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)               # only builds the CELLS list; main() is not called
    cells = module.CELLS
    start = next(i for i, c in enumerate(cells) if c['source'].lstrip().startswith('## Ô 1 '))
    end = next(i for i, c in enumerate(cells) if c['source'].lstrip().startswith('## Ô 5 '))
    copied = [dict(c) for c in cells[start:end]]
    assert [c['cell_type'] for c in copied] == ['markdown', 'code']*4, 'Ô 1-4 layout changed'
    last = copied[-1]
    assert last['source'].count('Được phép chạy Ô 5.') == 1
    last['source'] = last['source'].replace('Được phép chạy Ô 5.', 'Được phép chạy các ô BN (B1 → B4).')
    return copied


CELLS = []


def md(text):
    CELLS.append(dict(cell_type='markdown', metadata={}, source=text.strip('\n')))


def code(text):
    CELLS.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=text.strip('\n')))


md(r'''
# CS431 · E-BN pilot, Giai đoạn 0: đo độ lệch thống kê BatchNorm sau khi trộn (KHÁM PHÁ)

**Kết luận Mix A-core v1 giữ nguyên:** α\* = B-Clean, Mix A chưa thắng theo quy tắc đã khóa. Notebook này **không** sửa notebook v1 hay kết quả v1. Thiết kế: `docs/design_bn_recalibration_pilot.md`, kế hoạch §4.4.

**Giai đoạn 0 làm gì:**
- Model ở `eval()`, chỉ gắn hook đọc đầu vào của 2 lớp BN đang có tác dụng lúc inference: `det_head.binarize.1` và `.4`.
- **Không ghi gì** vào trọng số hay buffer; code kiểm SHA của toàn bộ `state_dict` trước và sau.
- Đo trên 1.000 nguồn **TRAIN** cố định × (Clean, LR2) = 2.000 ảnh, cho cả 5 checkpoint của v1. Hai endpoint là đối chứng.
- Không dùng ảnh A/B (remaining-validation) hay TEST để đo.

**Giai đoạn 1** (tính lại BN rồi chấm counts) **không có trong notebook này**. Nó chỉ mở khi nhóm đã thống nhất tiêu chí và Giai đoạn 0 có tín hiệu.

**Trạng thái:** code CPU có test ở máy local. **Phần GPU chưa chạy lần nào.** Mọi ngưỡng dừng hay đi tiếp ở đây là **đề xuất, Tuấn chưa duyệt**.

### Bạn cần chuẩn bị
1. Trên máy local, trong `cải tiến/mới`, chạy `python scripts/make_colab_upload.py`. Tải `runtime/colab_upload/mixab_code.zip` lên `MyDrive/CS431_MixA/upload/`, ghi đè bản cũ.
2. **File TRAIN từ Tuấn**, đặt vào `MyDrive/CS431_MixA/bn_pilot/input/`, **chỉ một trong ba dạng**:
   - `manifest__train__textvqa_clean_degraded_same_textocr_word_gt.jsonl` (hoặc `.jsonl.gz`): bản đầy đủ. SHA-256 phải đúng `ca1b9163…c983d`, sai là dừng.
   - `train_ids.csv` với các cột `source_image_id,clean_relpath,clean_sha256` (thêm `clean_size` nếu có). Được chấp nhận, nhưng ghi rõ **chưa xác minh được hash manifest gốc**.
3. Runtime → GPU **A100**. Chạy Ô 1 → Ô 4, rồi Ô B1 → Ô B4.

Thiếu file TRAIN thì Ô B1 dừng và báo rõ, **trước** mọi bước xử lý ảnh hoặc dùng GPU. Ô 1–4 vẫn chạy được để dựng sẵn môi trường.

### Dữ liệu tạm và dữ liệu giữ lại
- Tạm (`/content/cs431_tmp/`): ảnh TRAIN đã giải nén, ảnh LR2 TRAIN, code.
- Giữ lại trên Drive:
  - `bn_pilot/input/` (file TRAIN của bạn);
  - `bn_pilot/train_check/` (kết quả kiểm, subset, SHA ảnh);
  - `cache/textvqa_train_bn1000_*.zip` (1.000 ảnh Clean TRAIN);
  - `runs/bn_stage0_v1_<GPU>/` (kết quả, resume được);
  - `logs/`, `export/`.
''')

for cell in base_cells():
    CELLS.append(cell)

md(r'''
## Ô B1 · Kiểm file TRAIN và chọn subset cố định (chỉ CPU, không cần ảnh)

- Đọc file trong `bn_pilot/input/`. Manifest đầy đủ phải đúng SHA Tuấn xác nhận; danh sách ID thì ghi `NOT_VERIFIABLE_ID_LIST`.
- Kiểm: đúng **15.222** nguồn, không trùng ID, SHA hợp lệ, đường dẫn nằm trong `train_images/`.
- Kiểm **không trùng** ID hoặc SHA ảnh Clean với 2.262 nguồn remaining-validation đã công bố. 1.000 nguồn S1/S2 của validation chưa công bố nên không kiểm được; kết quả ghi rõ điều này.
- Chọn 1.000 nguồn cố định: sắp theo ID, `numpy.default_rng(20261120)`. Lần chạy sau phải ra đúng subset cũ, nếu khác thì dừng, không ghi đè.

Không tự chia lại TextVQA. Không có file của Tuấn thì dừng ở đây.
''')

code(r'''
# ===================== Ô B1 · FILE TRAIN + SUBSET =====================
BN = D/'bn_pilot'
BN_INPUT, BN_CHECK = BN/'input', BN/'train_check'
for p in (BN_INPUT, BN_CHECK):
    p.mkdir(parents=True, exist_ok=True)
TRAIN_NAMES = ['manifest__train__textvqa_clean_degraded_same_textocr_word_gt.jsonl',
               'manifest__train__textvqa_clean_degraded_same_textocr_word_gt.jsonl.gz', 'train_ids.csv']
found = [BN_INPUT/n for n in TRAIN_NAMES if (BN_INPUT/n).is_file()]
if not found:
    ch.fail('Chưa có file TRAIN. Dừng trước mọi bước xử lý ảnh/GPU.\n'
            f'Đặt MỘT trong các file sau vào {BN_INPUT}:\n  - ' + '\n  - '.join(TRAIN_NAMES) +
            '\n(manifest đầy đủ của Tuấn, SHA-256 ca1b91636cbe…c983d; hoặc train_ids.csv có cột '
            'source_image_id,clean_relpath,clean_sha256) rồi chạy lại Ô B1.')
if len(found) > 1:
    ch.fail(f'Có nhiều file TRAIN trong {BN_INPUT}: {[p.name for p in found]}. Chỉ giữ một file.')
LOG = LOGS/f'bn_check_{GPU}.log'
ch.run([PY, '-m', 'mixab.bn_recal', 'check-train', '--train-input', found[0], '--bundle', BUNDLE, '--output', BN_CHECK],
       LOG, cwd=CODE, env=ENV, title='bn_recal check-train')
check = json.loads((BN_CHECK/'train_check.json').read_text())
print(f"\nDạng file: {check['kind']}   trạng thái hash manifest: {check['manifest_sha256_status']}")
if check['manifest_sha256_status'] != 'VERIFIED_MATCH':
    print('LƯU Ý: đây là danh sách ID/SHA ảnh, CHƯA xác minh được hash manifest gốc của Tuấn. Ghi điều này vào báo cáo.')
print('Subset:', check['subset']['sources'], 'nguồn, SHA', check['subset'].get('sha256', '')[:16], '...')
''')

md(r'''
## Ô B2 · Ảnh TRAIN Clean của subset + tạo LR2 bằng hàm của Tuấn

- **Ảnh Clean:** lấy từ `cache/textvqa_train_bn1000_<sha>.zip` trên Drive nếu đã có. Nếu chưa, tải lại zip TextVQA 7,07 GB vào thư mục tạm, chỉ trích 1.000 ảnh của subset, kiểm SHA từng ảnh với file TRAIN, rồi lưu bản cache lên Drive.
- **LR2:** tạo bằng `rev4.training_data.native_lr2` (Pillow 11.3.0, bicubic, mỗi chiều còn một nửa, không phóng lại), lưu PNG như cách Tuấn làm với validation.
  - Không có SHA LR2 khóa sẵn cho TRAIN, nên lần đầu ghi SHA vào `train_check/subset_images.csv`.
  - Các lần sau (VM mới), LR2 tạo lại phải **trùng từng byte**, nếu không thì dừng.
''')

code(r'''
# ===================== Ô B2 · ẢNH TRAIN + LR2 =====================
import csv
subset_csv = BN_CHECK/'subset.csv'
if not subset_csv.is_file():
    ch.fail('Chưa có subset.csv. Chạy Ô B1 trước.')
with subset_csv.open(encoding='utf-8', newline='') as f:
    subset = list(csv.DictReader(f))
items = [(r['clean_relpath'], r['clean_sha256']) for r in subset]
names = [r for r, _ in items]
SUBSET_TAG = ch.sha256(subset_csv)[:12]
TRAIN_CACHE = CACHE/f'textvqa_train_bn1000_{SUBSET_TAG}.zip'
TRAIN_IMAGES, TRAIN_LR2 = T/'textvqa_train_bn', T/'train_lr2_bn'
LOG = LOGS/f'bn_data_{GPU}.log'
if TRAIN_CACHE.is_file():
    ch.extract_members(TRAIN_CACHE, names, TRAIN_IMAGES)
else:
    big = DOWNLOADS/'train_val_images.zip'
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    if not (big.is_file() and big.stat().st_size == PIN['TEXTVQA_ZIP_BYTES']):
        ch.run(['wget', '-c', '-q', '--show-progress', '-O', big, PIN['TEXTVQA_ZIP_URL']], LOG, title='download TextVQA zip (7 GB)')
    if big.stat().st_size != PIN['TEXTVQA_ZIP_BYTES']:
        ch.fail(f'Zip TextVQA có {big.stat().st_size} byte, kỳ vọng {PIN["TEXTVQA_ZIP_BYTES"]}; chạy lại để tải tiếp.')
    ch.extract_members(big, names, TRAIN_IMAGES)
check_images = ch.verify_images(TRAIN_IMAGES, items)
print('Ảnh Clean TRAIN:', check_images)
if check_images['ok'] != len(items):
    ch.fail(f'Ảnh Clean TRAIN chưa khớp SHA với file TRAIN: {check_images}')
if not TRAIN_CACHE.is_file():
    ch.build_cache_zip(TRAIN_IMAGES, names, TRAIN_CACHE)
    print('Đã lưu cache 1.000 ảnh TRAIN lên Drive:', TRAIN_CACHE)
    big.unlink()
ch.run([PY, '-m', 'mixab.bn_recal', 'prepare-images', '--subset', subset_csv, '--clean-root', TRAIN_IMAGES,
        '--lr2-root', TRAIN_LR2, '--record', BN_CHECK/'subset_images.csv'], LOG, cwd=CODE, env=ENV, title='bn_recal prepare-images')
print('\nÔ B2 xong: 1.000 Clean (SHA khớp) + 1.000 LR2 (native_lr2).')
''')

md(r'''
## Ô B3 · Giai đoạn 0 trên GPU (resume được)

- Chạy lần lượt 5 checkpoint (α = 0, ¼, ½, ¾, 1) của v1, mỗi checkpoint 2.000 ảnh TRAIN, batch 1, FP32. Đường forward giống `rev4.stage_a.forward` (`mode='tensor'`).
- Ô dừng nếu trọng số hoặc buffer bị thay đổi (so SHA toàn bộ `state_dict` trước và sau).
- **Thời gian chưa đo.** Ô in số giây mỗi ảnh sau mỗi 100 ảnh. Ước lượng thô từ A2r: dưới 1 giờ trên A100, vì không chấm 72 setting.
- **Bị ngắt:** tiến độ lưu mỗi 100 ảnh trong `runs/bn_stage0_v1_<GPU>/`. Chạy lại Ô 1 → Ô 4, Ô B1, Ô B2, Ô B3; nó chạy tiếp từ chỗ dừng. Model nào xong rồi sẽ không chạy lại. Đổi code, GPU, subset hay weights thì identity đổi và ô từ chối trộn kết quả.
''')

code(r'''
# ===================== Ô B3 · GIAI ĐOẠN 0 =====================
pass_flag = T/'doctor_pass.json'
if not pass_flag.is_file():
    ch.fail('Chưa có doctor 0 FAIL trong phiên này. Chạy Ô 4 trước.')
flag = json.loads(pass_flag.read_text())
if flag['gpu'] != GPU or flag['code_zip_sha256'] != code_sha:
    ch.fail('Doctor đã chạy với GPU hoặc code khác. Chạy lại Ô 4.')
if not (BN_CHECK/'subset_images.csv').is_file():
    ch.fail('Chưa có subset_images.csv. Chạy Ô B2 trước.')
BN_OUT = RUNS/f'bn_stage0_v1_{GPU}'
if (BN_OUT/'stage0_summary.json').is_file():
    print(f'Giai đoạn 0 trên {GPU} đã xong trước đó ({BN_OUT}). Không chạy lại; sang Ô B4.')
else:
    log = ch.new_path(LOGS, f'bn_stage0_{GPU}', '.log')
    ch.run([PY, '-m', 'mixab.bn_recal', 'stage0', '--subset-images', BN_CHECK/'subset_images.csv',
            '--train-check', BN_CHECK/'train_check.json', '--clean-root', TRAIN_IMAGES, '--lr2-root', TRAIN_LR2,
            '--model-config', CONFIG_PATH, '--weights-dir', WEIGHTS, '--output', BN_OUT, '--chunk', '100'],
           log, cwd=CODE, env=ENV, title=f'bn_recal stage0 on {GPU}')
    print(f'\nGiai đoạn 0 xong. Kết quả: {BN_OUT}\nLog: {log}')
''')

md(r'''
## Ô B4 · Tổng hợp và đóng gói gửi lại

Các số được in:
- **Lệch giữa thống kê đã lưu và thống kê đo được**, theo từng lớp và từng α:
  - `z_mean`: |Δμ|/σ trung bình theo kênh;
  - `logvar_mean`: |log tỉ lệ phương sai| trung bình;
  - mỗi số tách theo Clean, LR2 và tổng.
- **Phi tuyến:** thống kê đo được của α lệch bao nhiêu so với nội suy tuyến tính thống kê đo được của hai endpoint.
- **Cổng đề xuất (CHƯA DUYỆT):** có tín hiệu nếu `z_mean` của `binarize.1` ở một α giữa lớn hơn 2 lần mức lớn nhất của hai endpoint. Đây chỉ là gợi ý để bàn với Tuấn, **không** tự mở Giai đoạn 1.
- `binarize.4` đo sau `binarize.1` với thống kê đã lưu, nên chỉ để tham khảo.

Cuối ô tạo `export/cs431_bn_stage0_<GPU>_<thời gian>.zip` để gửi lại.
''')

code(r'''
# ===================== Ô B4 · TỔNG HỢP + ĐÓNG GÓI =====================
BN_OUT = RUNS/f'bn_stage0_v1_{GPU}'
if not (BN_OUT/'stage0_summary.json').is_file():
    ch.fail('Chưa có stage0_summary.json. Chạy Ô B3 trước.')
s = json.loads((BN_OUT/'stage0_summary.json').read_text())
for layer in ('det_head.binarize.1', 'det_head.binarize.4'):
    print(f'\n{layer}  (đã lưu so với đo trên TRAIN)')
    for alpha, row in s['stored_vs_measured'].items():
        r = row[layer]
        print(f"  α={alpha:4}  z_mean all/clean/lr2 = {r['all']['z_mean']:.3f}/{r['clean']['z_mean']:.3f}/{r['lr2']['z_mean']:.3f}"
              f"   logvar_mean all = {r['all']['logvar_mean']:.3f}")
    for alpha, r in s['nonlinearity_vs_endpoint_interp'][layer].items():
        print(f"  phi tuyến α={alpha:4}: z_mean = {r['z_mean']:.3f}, logvar_mean = {r['logvar_mean']:.3f}")
g = s['gate']
print(f"\nCổng ĐỀ XUẤT (chưa duyệt): {g['rule']}\n  mức nền endpoint = {g['endpoint_baseline']:.3f}; "
      f"α giữa = { {k: round(v, 3) for k, v in g['interior'].items()} }\n  tín hiệu: {g['signal']} {g['alphas_with_signal']}")
print('Giai đoạn 1 KHÔNG tự chạy: cần Tuấn duyệt tiêu chí trước.')
for alpha, p in s['profile'].items():
    if p.get('mean_seconds_per_image'):
        print(f"  α={alpha:4}: {p['images']} ảnh, {p['mean_seconds_per_image']:.3f} s/ảnh")

pack = ch.new_path(EXPORT, f'cs431_bn_stage0_{GPU}', '.zip')
with zipfile.ZipFile(pack, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for p in sorted(BN_OUT.iterdir()):
        if p.is_file() and not p.name.startswith('.'):
            zf.write(p, f'stage0/{p.name}')
    for name in ('train_check.json', 'subset.csv', 'subset_images.csv'):
        if (BN_CHECK/name).is_file():
            zf.write(BN_CHECK/name, f'train_check/{name}')
    for folder, pattern in ((REPORTS, f'*_{GPU}_*.json'), (LOGS, f'*bn*{GPU}*.log'), (LOGS, f'doctor_{GPU}.log')):
        for p in sorted(folder.glob(pattern)):
            zf.write(p, f'{folder.name}/{p.name}')
    zf.writestr('code_zip_sha256.txt', code_sha + '\n')
print(f'\nĐã đóng gói: {pack} ({pack.stat().st_size/1024:.0f} KiB). Có thể tải trực tiếp:')
try:
    from google.colab import files
    files.download(str(pack))
except Exception as exc:
    print('Không tự tải được, hãy tải từ Drive:', exc)
''')

md(r'''
## Giai đoạn 1: chưa mở

Giai đoạn 1 tính lại thống kê 2 lớp BN `binarize` trên cùng subset TRAIN rồi chấm counts 72 setting, có đối chứng trước/sau và endpoint. Nó **chỉ được thêm vào notebook** khi đủ cả ba điều kiện:
1. Tuấn duyệt cổng Giai đoạn 0, ngưỡng "đáng xác nhận" (§6–§7 của thiết kế), và cách xác nhận (§8).
2. Giai đoạn 0 có tín hiệu theo cổng đã duyệt.
3. Kết quả Giai đoạn 0 đã được đọc và ghi vào WORKLOG.

Không mở TEST. Kết quả trên A/B chỉ là khám phá.
''')


def main():
    sys.stdout.reconfigure(errors='backslashreplace')
    def lines(text):
        parts = text.split('\n')
        return [p + '\n' for p in parts[:-1]] + [parts[-1]]
    notebook = dict(nbformat=4, nbformat_minor=5,
                    metadata=dict(accelerator='GPU', colab=dict(provenance=[], gpuType='A100'),
                                  kernelspec=dict(name='python3', display_name='Python 3'),
                                  language_info=dict(name='python')),
                    cells=[dict(c, id=f'cell-{i:02d}', source=lines(c['source'])) for i, c in enumerate(CELLS)])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(f'{OUT}: {len(CELLS)} cells')


if __name__ == '__main__':
    main()
