"""Generate notebooks/colab_dualagree_v1.ipynb: DualAgree v1 dump (GPU) + analysis (CPU).

Ô 1–4 (config, environment, data/weights, doctor) are reused verbatim from build_colab_notebook.py
so the environment is identical to the A2r run. Re-run: python scripts/build_dualagree_notebook.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_colab_notebook as base  # noqa: E402

OUT = ROOT/'notebooks/colab_dualagree_v1.ipynb'
CELLS = []


def md(text):
    CELLS.append(dict(cell_type='markdown', metadata={}, source=text.strip('\n')))


def code(text):
    CELLS.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=text.strip('\n')))


md(r'''
# CS431 · Thiết kế 2 (DualAgree v1): chạy B1 **và** B-Clean trên ảnh LR2, gộp box bất đối xứng

**Notebook làm gì:**
1. **Ô 1–4:** dựng môi trường, dữ liệu, weights và doctor. Giống hệt notebook A2r (`colab_E1_A2_probe.ipynb`), nên môi trường trùng với lần chạy A2r.
2. **Ô 5 · Probe 40 nguồn:** lưu box + score của 2 model trên ảnh LR2, rồi kiểm **độ trung thực**. Counts tính lại từ dump phải trùng tuyệt đối với counts A2r (72 setting).
3. **Ô 6 · Dump đầy đủ:** 2 model × 2.262 ảnh LR2. Ước lượng khoảng 32 phút trên A100 (ngoại suy từ A2r 0,41–0,43 s/task); T4 **chưa đo**.
4. **Ô 7 · Phân tích (CPU):** quét luật gộp trên A, khóa, báo cáo B (khám phá), rồi đóng gói zip.

**Không train gì, không dùng TEST, không sửa repo của Tuấn.** Ảnh Clean không chạy lại: hệ thống dùng B-Clean cho Clean (counts A2r).

**Nên dùng A100-SXM4-40GB**, cùng loại GPU với A2r, để cổng trung thực so được tuyệt đối. Nếu chạy trên T4, notebook vẫn chạy nhưng kết quả ghi `PENDING_REFERENCE_CHECK` và phải báo tỉ lệ lệch.

### Chuẩn bị (một lần)
1. Trên máy local, trong thư mục `cải tiến/mới`, chạy `python scripts/make_colab_upload.py`.
2. Tải `runtime/colab_upload/mixab_code.zip` lên `MyDrive/CS431_MixA/upload/`, **thay file cũ** (code mới có `mixab/dualagree.py`).
3. Runtime → GPU (ưu tiên A100). Chạy Ô 1 → Ô 7.

**Bị ngắt giữa chừng:** dump lưu theo shard 64 ảnh trên Drive. Chạy lại Ô 1 → Ô 4 rồi ô đang dừng; shard xong rồi sẽ không chạy lại.
''')

CELLS.extend(base.CELLS[1:9])          # Ô 1–4 unchanged

md(r'''
## Ô 5 · Probe 40 nguồn + cổng trung thực

- Chạy `mixab.dualagree dump --probe 40` vào `runs/dualagree_probe_v1_<GPU>/`.
- Sau đó tính lại counts LR2 của **cả 72 setting** từ dump bằng `CandidateMatcher` của Tuấn, và so với counts A2r.
  - **EXACT:** dump trung thực, sang Ô 6.
  - **DRIFT:** nếu GPU khác A2r thì đây là độ lệch giữa GPU (ghi lại). Nếu **cùng GPU mà DRIFT thì dừng** và gửi log.
- Ô cũng in thời gian mỗi task trên GPU này để ước lượng Ô 6.
''')

code(r'''
# ===================== Ô 5 · PROBE DUMP + TRUNG THỰC =====================
pass_flag = T/'doctor_pass.json'
if not pass_flag.is_file():
    ch.fail('Chưa có doctor 0 FAIL trong phiên này. Chạy Ô 4 trước.')
flag = json.loads(pass_flag.read_text())
if flag['gpu'] != GPU or flag['code_zip_sha256'] != code_sha:
    ch.fail('Doctor đã chạy với GPU hoặc code khác. Chạy lại Ô 4.')
A2R_GPU = 'A100-SXM4-40GB'
A2R = RUNS/f'merge_audit_v1_{A2R_GPU}'
if not (A2R/'B1_source_counts.npz').is_file():
    ch.fail(f'Không thấy counts A2r ở {A2R}.')
DA_PROBE = RUNS/f'dualagree_probe_v1_{GPU}'
if not (DA_PROBE/'dump_done.json').is_file():
    log = ch.new_path(LOGS, f'da_probe_{GPU}', '.log')
    ch.run([PY, '-m', 'mixab.dualagree', 'dump', '--bundle', BUNDLE, '--clean-root', IMAGES, '--prepared-root', PREPARED,
            '--model-config', CONFIG_PATH, '--weights-dir', WEIGHTS, '--probe', '40', '--chunk', '16', '--output', DA_PROBE],
           log, cwd=CODE, env=ENV, title=f'dualagree probe dump on {GPU}')
done = json.loads((DA_PROBE/'dump_done.json').read_text())
for name, p in done['profile'].items():
    if p['tasks_this_session']:
        print(f"  {name:8} {p['mean_seconds_per_task']:.2f} s/task -> dump đầy đủ ~{p['mean_seconds_per_task']*2262/60:.0f} phút/model")
fid_path = DA_PROBE/'fidelity.json'
if not fid_path.is_file():
    proc = subprocess.run([str(c) for c in (PY, '-m', 'mixab.dualagree', 'fidelity', '--dump', DA_PROBE,
                                            '--bundle', BUNDLE, '--npz-run', A2R)],
                          cwd=CODE, env=dict(os.environ, **ENV), capture_output=True, text=True)
    raw = proc.stdout.strip()
    if proc.returncode != 0 or '{' not in raw:
        print(proc.stdout[-3000:]); print(proc.stderr[-5000:])
        ch.fail('Kiểm trung thực lỗi; traceback ở trên.')
    fid_path.write_text(raw[raw.index('{'):])
fid = json.loads(fid_path.read_text())
for name, f in fid.items():
    print(f"  {name:8} trung thực: {f['status']} ({f['mismatched']}/{f['checked']} ô lệch)")
same_gpu = GPU == A2R_GPU
if same_gpu and any(f['status'] != 'EXACT' for f in fid.values()):
    ch.fail('Cùng GPU với A2r mà counts lệch: dump không trung thực. Dừng, gửi log + fidelity.json.')
print('\nĐược chạy Ô 6.' + ('' if same_gpu else f' (GPU {GPU} khác A2r: kết quả sẽ ghi PENDING_REFERENCE_CHECK)'))
''')

md(r'''
## Ô 6 · Dump đầy đủ 2.262 ảnh LR2 × 2 model

Kết quả ở `runs/dualagree_dump_v1_<GPU>/` trên Drive (mỗi model một thư mục shard). Dump đã xong thì ô chỉ báo "đã xong".
''')

code(r'''
# ===================== Ô 6 · DUMP ĐẦY ĐỦ =====================
if not (DA_PROBE/'fidelity.json').is_file():
    ch.fail('Chưa có kết quả trung thực của probe. Chạy Ô 5 trước.')
DA_FULL = RUNS/f'dualagree_dump_v1_{GPU}'
if (DA_FULL/'dump_done.json').is_file():
    print(f'Dump trên {GPU} đã xong trước đó ({DA_FULL}). Sang Ô 7.')
else:
    log = ch.new_path(LOGS, f'da_full_{GPU}', '.log')
    ch.run([PY, '-m', 'mixab.dualagree', 'dump', '--bundle', BUNDLE, '--clean-root', IMAGES, '--prepared-root', PREPARED,
            '--model-config', CONFIG_PATH, '--weights-dir', WEIGHTS, '--chunk', '64', '--output', DA_FULL],
           log, cwd=CODE, env=ENV, title=f'dualagree full dump on {GPU}')
    print(f'\nDump xong: {DA_FULL}\nLog: {log}')
''')

md(r'''
## Ô 7 · Phân tích (CPU, ước lượng 10–20 phút) + đóng gói

- Cổng trung thực trên toàn bộ dump (72 setting × 2 model).
- Quét 1.928 tổ hợp luật gộp (t_l, t_c, t_a, θ) **chỉ trên A**, khóa tổ hợp tốt nhất, rồi báo B (khám phá) kèm bootstrap theo nguồn.
- Luật dừng (trên A, theo tài liệu thiết kế): LR2 ≥ B1 + 0,3 pp; FP ở cùng recall thấp hơn B1; tiny ≥ B1 − 3 pp. Cổng nhóm: mean hệ thống ≥ model đơn tốt nhất + 0,5 pp.
- Không ghi đè: đã có `decision.json` thì chỉ đọc lại.
''')

code(r'''
# ===================== Ô 7 · PHÂN TÍCH + ĐÓNG GÓI =====================
DA_FULL = RUNS/f'dualagree_dump_v1_{GPU}'
if not (DA_FULL/'dump_done.json').is_file():
    ch.fail('Dump chưa xong. Chạy Ô 6.')
DA_OUT = RUNS/f'dualagree_analysis_v1_{GPU}'
if not (DA_OUT/'decision.json').is_file():
    log = ch.new_path(LOGS, f'da_analyze_{GPU}', '.log')
    ch.run([PY, '-m', 'mixab.dualagree', 'analyze', '--dump', DA_FULL, '--bundle', BUNDLE, '--npz-run', A2R,
            '--p0-fixed', CODE/'results/diag_v2_cpu/p0_fixed.json', '--output', DA_OUT],
           log, cwd=CODE, env=ENV, title='dualagree analyze (CPU)')
d = json.loads((DA_OUT/'decision.json').read_text())
print('verification:', d['verification'], d['fidelity'])
print('tham số khóa trên A:', d['locked_params'])
for s in ('A', 'B'):
    r = d[s]
    print(f"{s}: LR2 gộp {r['best']['hmean']:.4%} vs B1 {r['b1_alone']['hmean']:.4%} ({r['delta_lr2_pp']:+.2f} pp), "
          f"FP {r['best']['fp']} (B1 FP nhỏ nhất ở cùng recall: {r['b1_min_fp_at_fused_recall']}), "
          f"tiny {r['delta_tiny_pp']:+.2f} pp, mean hệ thống - model đơn tốt nhất {r['system_minus_best_single_pp']:+.2f} pp")
print('luật dừng trên A:', d['kill_rules_A'], '-> continue:', d['continue'])
print('cổng nhóm +0,5 pp trên A:', d['group_gate_A_mean_plus_0p5'])
print('bootstrap B (LR2 gộp - B1):', d['boot_B_lr2_fused_minus_b1'])
pack = ch.new_path(EXPORT, f'cs431_dualagree_{GPU}', '.zip')
with zipfile.ZipFile(pack, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for folder, arc in ((DA_OUT, 'analysis'), (DA_PROBE, 'probe')):
        for p in sorted(folder.glob('*.json')):
            zf.write(p, f'{arc}/{p.name}')
    for p in sorted(DA_FULL.glob('*.json')):
        zf.write(p, f'dump/{p.name}')
    for p in sorted(LOGS.glob(f'da_*{GPU}*.log')):
        zf.write(p, f'logs/{p.name}')
    zf.writestr('code_zip_sha256.txt', code_sha + '\n')
print(f'\nĐã đóng gói: {pack}')
''')


md(r'''
## Ô 8 · Đo latency của biến thể đã chốt (B1 + box chỉ B-Clean thấy)

**Chạy sau Ô 1 → Ô 4** (không cần chạy lại Ô 5–7).
- Nạp **cả hai** checkpoint cùng lúc như khi triển khai. Đo 200 ảnh LR2 cố định (bỏ 10 ảnh khởi động), batch 1, FP32.
- Ghi lại từng phần: B1 một mình, B-Clean, bước gộp (CPU), và tổng; kèm VRAM đỉnh khi giữ 2 model.
- Kiểm chéo: số box sau khi gộp ở mỗi ảnh phải **trùng** với kết quả tính từ dump T4 (cùng loại GPU).
- Khoảng 5–10 phút trên T4 (ước lượng). Kết quả ở `runs/dualagree_latency_v1_<GPU>/latency.json`; đã có thì chỉ đọc lại.
''')

code(r'''
# ===================== Ô 8 · LATENCY BIẾN THỂ ĐÃ CHỐT =====================
pass_flag = T/'doctor_pass.json'
if not pass_flag.is_file():
    ch.fail('Chưa có doctor 0 FAIL trong phiên này. Chạy Ô 4 trước.')
LAT = RUNS/f'dualagree_latency_v1_{GPU}'
DA_FULL = RUNS/f'dualagree_dump_v1_{GPU}'
if not (LAT/'latency.json').is_file():
    log = ch.new_path(LOGS, f'da_latency_{GPU}', '.log')
    cmd = [PY, '-m', 'mixab.dualagree', 'latency', '--bundle', BUNDLE, '--clean-root', IMAGES, '--prepared-root', PREPARED,
           '--model-config', CONFIG_PATH, '--weights-dir', WEIGHTS, '--n', '200', '--warmup', '10', '--output', LAT]
    if (DA_FULL/'dump_done.json').is_file():
        cmd += ['--dump', DA_FULL]
    ch.run(cmd, log, cwd=CODE, env=ENV, title=f'dualagree latency on {GPU}')
r = json.loads((LAT/'latency.json').read_text())
for k in ('b1', 'bclean', 'fuse', 'system'):
    m = r['ms'][k]
    print(f"  {k:7} median {m['median']:.1f} ms   p90 {m['p90']:.1f} ms")
print(f"Hệ thống / B1 một mình (median): {r['system_over_b1_median']:.2f}x   VRAM đỉnh khi giữ 2 model: {r['peak_gpu_mib_both_loaded']:.0f} MiB")
print('Kiểm chéo với dump:', r.get('check_vs_dump', 'không có dump cùng GPU'))
''')


def main():
    sys.stdout.reconfigure(errors='backslashreplace')
    notebook = dict(nbformat=4, nbformat_minor=5,
                    metadata=dict(accelerator='GPU', colab=dict(provenance=[], gpuType='A100'),
                                  kernelspec=dict(name='python3', display_name='Python 3'),
                                  language_info=dict(name='python')),
                    cells=[dict(c, id=f'cell-{i:02d}',
                                source=[line + '\n' for line in c['source'].split('\n')][:-1] +
                                [c['source'].split('\n')[-1]]) for i, c in enumerate(CELLS)])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(f'{OUT}: {len(CELLS)} cells')


if __name__ == '__main__':
    main()
