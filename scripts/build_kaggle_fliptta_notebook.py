"""Generate notebooks/kaggle_fliptta_pilot.ipynb: Flip-TTA on the P-map (B1) screening pilot (docs/fliptta_spec_20261123.md).
Ô 1 and Ô 3 are derived from the zoom notebook (dataset kind, module and checkpoint-name handling changed); Ô 2 (environment) is
reused verbatim. Re-run: python scripts/build_kaggle_fliptta_notebook.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_kaggle_zoom_notebook as zoomnb  # noqa: E402

OUT = ROOT/'notebooks/kaggle_fliptta_pilot.ipynb'
CELLS = []


def md(text):
    CELLS.append(dict(cell_type='markdown', metadata={}, source=text.strip('\n')))


def code(text):
    CELLS.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=text.strip('\n')))


md(r'''
# CS431 · Pilot sàng lọc Flip-TTA trên P-map (B1, 1 checkpoint) trên Kaggle T4

**Kiểm tra giả thuyết:** trung bình bản đồ xác suất của ảnh gốc và ảnh lật ngang (đã lật ngược về đúng tọa độ), trước hậu xử lý DB, có giảm FP / tăng Hmean của B1 `.40/.60` không?

**Phạm vi:** một checkpoint B1, **không train**, hậu xử lý giữ nguyên (`mask_040`, score 0,60), **không quét ngưỡng**. Mẫu cố định gồm 300 nguồn của split A, cả Clean và LR2 (600 ảnh). Spec khóa trước: `docs/fliptta_spec_20261123.md`.
Đây là pilot development, **không phải xác nhận độc lập**; TTA chưa mặc định được tính là đóng góp phương pháp mới.

**Hai cánh, cùng một lượt chạy, cùng GPU:**
- **F:** một forward trên ảnh gốc (baseline B1 `.40/.60`).
- **T:** P-map = trung bình của P0 và P1b (P-map của ảnh lật, lật ngược). Chỉ lật vùng nội dung, vùng pad giữ nguyên, không lấy mẫu lại.

**Kiểm căn chỉnh (cứng) chạy cho từng ảnh:** pad hằng, lật hai lần = nguyên bản, cùng metainfo, P-map hữu hạn cùng shape, và trung vị IoU đã căn phải lớn hơn đối chứng chưa lật ngược. Trượt thì **không diễn giải chất lượng**.

### Trước khi chạy
1. Kaggle → **Create → New Dataset**, tải `cs431-fliptta-pilot.zip` lên, để **Private**.
2. **File → Import Notebook**, chọn file này.
3. Cột phải: **Accelerator** `GPU T4 x2` (hoặc `GPU T4`), **Internet On**, **Add Input** → dataset vừa tạo.
4. Chạy **Ô 1 → Ô 4**, đọc kết quả chạy thử ở Ô 4, rồi mới chạy **Ô 5 → Ô 7**.

**Thời gian: chưa đo.** Ô 2: 5–40 phút. Ô 5: mỗi ảnh chạy F + T (khoảng 3 lượt forward) + kiểm căn chỉnh; Ô 4 in ước lượng **đo thật**.

**Dừng nếu:** Ô 4 báo kiểm căn chỉnh cứng sai, hoặc thời gian dự kiến vượt 3 giờ (báo lại Claude, không tự đổi cấu hình hay giảm số ảnh).

**Trạng thái:** notebook đã tạo và kiểm cú pháp ở máy local; **chưa chạy trên Kaggle**.
''')

CELLS.append(dict(zoomnb.CELLS[1]))
c1 = zoomnb.CELLS[2]['source']
c1 = c1.replace("get('kind') == 'zoom_pilot'", "get('kind') == 'fliptta_pilot'").replace("'zoom' in z.name.lower()", "'fliptta' in z.name.lower()")
c1 = c1.replace("Không thấy dataset cs431-zoom-pilot", "Không thấy dataset cs431-fliptta-pilot").replace("Thấy {len(found)} dataset zoom", "Thấy {len(found)} dataset fliptta")
assert 'fliptta_pilot' in c1 and 'cs431-fliptta-pilot' in c1
CELLS.append(dict(zoomnb.CELLS[2], source=c1))
CELLS.extend(zoomnb.CELLS[3:5])             # Ô 2 environment (verbatim)

md(r'''
## Ô 3 · Repo của Tuấn, config, checkpoint B1, ảnh LR2 của mẫu

Clone `MONKEY_V2` @ `dc19b85`, vá config (chỉ đường dẫn encoder), kiểm SHA B1, tạo 300 ảnh LR2 bằng `native_lr2` của Tuấn (kiểm SHA từng ảnh), chạy test CPU.
''')

c3 = zoomnb.CELLS[6]['source']
c3 = c3.replace("if sha256(WEIGHTS/Path(b1['checkpoint']).name) != b1['checkpoint_sha256']:",
                "from pathlib import PureWindowsPath\nif sha256(WEIGHTS/PureWindowsPath(b1['checkpoint']).name) != b1['checkpoint_sha256']:")
c3 = c3.replace("'-m', 'mixab.zoom', 'prepare'", "'-m', 'mixab.fliptta', 'prepare'").replace("title='zoom prepare'", "title='fliptta prepare'")
assert 'PureWindowsPath' in c3 and 'mixab.fliptta' in c3 and 'mixab.zoom' not in c3
code(c3)

md(r'''
## Ô 4 · Chạy thử 3 nguồn (6 ảnh): kiểm căn chỉnh, bộ nhớ, thời gian

- Chạy F và T trên 3 nguồn, in **kiểm căn chỉnh cứng**. Nếu pad/lật/metainfo sai thì ô dừng ngay.
- In thời gian từng cánh (ms), IoU đã căn so với đối chứng, thời gian **dự kiến toàn pilot** (đo thật), bộ nhớ đỉnh.
- Dự kiến vượt 3 giờ thì **dừng và báo Claude**.
''')

code(r'''
# ===================== Ô 4 · CHẠY THỬ =====================
SMOKE = OUT/'fliptta_smoke'
if not (SMOKE/'infer_summary.json').is_file():
    ch.run([PY, '-m', 'mixab.fliptta', 'infer', '--bundle', BUNDLE, '--clean-root', CLEAN, '--prepared-root', PREPARED,
            '--model-config', CONFIG_PATH, '--weights-dir', WEIGHTS, '--limit-sources', '3', '--shard', '6',
            '--output', SMOKE], LOGS/'fliptta_smoke.log', cwd=CODE, env=ENV, title='fliptta smoke')
s = json.loads((SMOKE/'infer_summary.json').read_text())
import gzip
recs = [json.loads(l) for p in sorted(SMOKE.glob('shard_*.jsonl.gz')) for l in gzip.open(p, 'rt')]
print(f"Kiểm căn chỉnh: {s['alignment_gate']}")
print(f"{s['seconds_per_task']:.1f} s/ảnh → dự kiến toàn pilot (600 ảnh): {s['projected_minutes_full_pilot']:.0f} phút")
for r in recs:
    c = r['check']
    print(f"  {r['condition']:5} {r['size']} → tensor {r['img_shape']} · F {r['F']['ms']:.0f} ms ({len(r['F']['raw'])} ứng viên), "
          f"T {r['T']['ms']:.0f} ms ({len(r['T']['raw'])}) · thứ tự {r['order']} · IoU căn {c['iou_aligned']} / đối chứng {c['iou_control']} "
          f"· pad hằng {c['pad_constant']} · lật 2 lần {c['flip_twice_identity']} · cùng metainfo {c['same_metainfo']} · đỉnh {r['peak_mib']:.0f} MiB")
if not s['alignment_gate']['passed']:
    print('\nDỪNG: kiểm căn chỉnh không đạt. Báo Claude trước khi chạy Ô 5.')
elif s['projected_minutes_full_pilot'] > 180:
    print('\nDỪNG: dự kiến vượt 3 giờ. Báo Claude trước khi chạy Ô 5.')
else:
    print('\nĐược chạy Ô 5.')
''')

md(r'''
## Ô 5 · Chạy pilot đủ 600 ảnh (GPU)

- Lưu sau mỗi 20 ảnh vào `cs431_out/fliptta_infer/`. Bị ngắt **trong cùng phiên** thì chạy lại ô này, ảnh đã xong sẽ bỏ qua.
- Nếu phiên Kaggle kết thúc, `/kaggle/working` mất. Trước khi dừng, chạy Ô 7 để tải phần đã xong.
- Kiểm cứng căn chỉnh chạy cho từng ảnh; sai thì ô dừng và báo ảnh nào.
''')

code(r'''
# ===================== Ô 5 · PILOT =====================
INF = OUT/'fliptta_infer'
ch.run([PY, '-m', 'mixab.fliptta', 'infer', '--bundle', BUNDLE, '--clean-root', CLEAN, '--prepared-root', PREPARED,
        '--model-config', CONFIG_PATH, '--weights-dir', WEIGHTS, '--output', INF],
       LOGS/f'fliptta_infer_{time.strftime("%Y%m%d-%H%M%S")}.log', cwd=CODE, env=ENV, title='fliptta pilot infer')
print(json.loads((INF/'infer_summary.json').read_text()))
''')

md(r'''
## Ô 6 · Đánh giá (CPU, chưa đo)

Chấm F và T bằng `CandidateMatcher` của Tuấn. Báo Clean và LR2: Hmean, TP/FP/FN, tiny, recall theo bin; Δ so với F kèm CI theo nguồn; chênh lệch theo từng nguồn (`per_source.csv`); latency thực đo; nhất quán của F với dump T4 và A2r. Nhãn kết quả tính theo luật khóa (spec mục 6).
''')

code(r'''
# ===================== Ô 6 · ĐÁNH GIÁ =====================
EV = OUT/'fliptta_eval'
ch.run([PY, '-m', 'mixab.fliptta', 'evaluate', '--bundle', BUNDLE, '--infer-dir', OUT/'fliptta_infer', '--dump', DUMP,
        '--a2r', A2R, '--zoom-sample', DS/'ref/zoom_sample.json', '--output', EV], LOGS/'fliptta_eval.log', cwd=CODE, env=ENV,
       title='fliptta evaluate')
r = json.loads((EV/'fliptta_result.json').read_text())
print(f"Nguồn: {r['sources']} (đủ: {r['complete']}) · căn chỉnh: {r['alignment_gate']}")
print(f"Nhất quán F: {r['consistency']} · chồng lấn mẫu zoom (mô tả): {r.get('overlap_with_zoom_sample')}")
for cond in ('lr2', 'clean'):
    c = r['conditions'][cond]
    for arm in ('F', 'T'):
        t = c[arm]
        print(f"  {cond:5} {arm}  H {t['hmean']*100:.2f}  TP {t['tp']} FP {t['fp']} FN {t['fn']}  recall {t['tp']/t['gt']*100:.2f}  tiny {t['tiny_recall']*100:.1f}")
    d = c['delta']
    print(f"  {cond:5} T − F: {d['delta_pp']:+.2f} pp, CI {d['ci95_pp'][0]:+.2f}..{d['ci95_pp'][1]:+.2f} · tiny {c['delta_tiny_pp']:+.2f} pp · "
          f"nguồn lên/xuống/bằng {c['per_source']['up']}/{c['per_source']['down']}/{c['per_source']['same']} · latency {c['latency_ms']}")
print('\nCờ:', r['flags'], '\nNhãn:', r['outcome'])
''')

md(r'''
## Ô 7 · Đóng gói kết quả

Tạo `/kaggle/working/cs431_fliptta_<GPU>_<thời gian>.zip` gồm toàn bộ `cs431_out` (trừ wheel). Tải ở khung **Output** và gửi lại Claude (hoặc báo Claude tải bằng Kaggle API).
''')

code(r'''
# ===================== Ô 7 · ĐÓNG GÓI =====================
pack = W/f'cs431_fliptta_{GPU}_{time.strftime("%Y%m%d-%H%M%S")}.zip'
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
