# HANDOFF — trạng thái hiện tại (đọc đầu mỗi session)

> Hiện trạng ghi ở đây; lịch sử chỉ append vào [WORKLOG.md](WORKLOG.md). Phạm vi và protocol theo [kế hoạch chính](Ke_hoach_phoi_hop_MixA_MixB_20261119.md); đọc §0 trước khi làm. Không lấy số toàn cohort làm baseline trên B.

## TRẠNG THÁI HIỆN TẠI

- **Cập nhật:** 20/11/2026 (Claude). **Notebook Colab** `notebooks/colab_E1_A2_probe.ipynb` cho E1 (cài môi trường + doctor) và probe A2 đã sẵn sàng. Kèm `mixab/colab_helpers.py`, `scripts/build_colab_notebook.py` (sinh notebook) và `scripts/make_colab_upload.py` (tạo `runtime/colab_upload/mixab_code.zip`, file duy nhất user cần upload lên Drive). **Notebook chưa chạy trên Colab**, nên E1 và A2 vẫn chưa xong. Tin nhắn Tuấn vẫn **chưa gửi**.
- **Ràng buộc GPU:** user chỉ có **T4 hoặc A100**, không có H200 như Tuấn.
  - Mọi job phải chạy được với batch 1 FP32, resume được.
  - Thời gian phải profile trên chính GPU đó.
  - Cổng 3a/3b so với NPZ chạy trên H200 có thể FAIL chỉ vì khác GPU: chạy probe trước, không tự nới cổng (kế hoạch §4.3).
- **Nơi đặt code:** `mixab/`, `tests/`, `configs/`, `results/`, `docs/` trong thư mục này. Trọng số và output nặng nằm ở `runtime/` (đã `.gitignore`). MONKEY_V2 chỉ để tham chiếu. Không tự init, push hay submit.
- **Repo Tuấn:** `main = origin/main = dc19b85` (fetch lúc rạng sáng 20/11, không có commit mới). Mình đã tự chạy `verify_bundle.py --counts --weights`: **PASS**. Hai file export khớp SHA-256.
- **Test local:** `python -m unittest discover -s tests -t . -v`: **50/50 PASS** (gồm 7 test `colab_helpers`). Chạy lại được từ code giải nén khỏi `mixab_code.zip` (mô phỏng Ô 3.2 của notebook), test chạy trên bundle thật có chạy (không skip).

| Phần mình | Hiện trạng | Còn thiếu |
|---|---|---|
| A1 `mixab/weight_merge.py` | **Đã chạy thật** trên B-Clean×B1 export: 5 α vào `runtime/merge_weights_v1/` (423 MB), endpoint giống hệt bản gốc, đọc lại giống hệt, `num_batches_tracked` của hai endpoint bằng nhau | Trên máy GPU phải tạo lại (10 giây) hoặc copy kèm `merge_manifest.json` |
| A3 `mixab/reselect_split.py` | **Đã chạy trên NPZ thật** (chỉ baseline) → `results/baseline_ab_v1/` + `NOTES.md`; khớp từng số với bảng §0 khi chọn toàn cohort | Chạy lại với 5 α khi có counts từ A2 |
| A2 `mixab/merge_audit.py` | **Đã viết**: `prepare-lr2`, `run --probe N`, resume theo chunk, cổng 3a/P-map, profile, tự sinh `reselect_spec.json`; phần CPU có test, gồm test trên bundle thật | **Phần GPU chưa chạy lần nào** (local không có MMOCR). Cần user dựng môi trường và chạy probe theo `docs/RUNBOOK_mix_a_T4_A100.md` |
| A4 | Chưa có | Sau A2 + A3 |
| Mix B `mixab/fsr.py` | Torch thuần, có test CPU | Thân R, padding, scheduler, tiêu chí chưa chốt; chưa tích hợp MMOCR |

**Phát hiện mới từ baseline A→B (chi tiết ở `results/baseline_ab_v1/NOTES.md`):**
- Trên A, B1 chọn setting **45** (`epsilon_005`, score .50), không phải 70. Top-4 setting cách nhau ≤ 0,23 pp, tức điểm tối ưu phẳng.
- **Chính B1 trượt mốc tiny 42% trên B** (37,55% ở setting 45, 39,38% ở setting 70), trong khi trên A là 52,19%. Nguyên nhân: GT <4 px tập trung vào ít ảnh, **5 nguồn chiếm 53% tiny GT của A**.
- Vì vậy mốc tuyệt đối 42% không ổn định. **Cần Tuấn quyết** có đổi sang mốc tương đối so với B1 trên cùng split hay không. Code chưa đổi.

## BƯỚC TIẾP THEO (theo thứ tự)

1. **User:** chạy `python scripts/make_colab_upload.py`, upload `runtime/colab_upload/mixab_code.zip` lên `MyDrive/CS431_MixA/upload/`, mở `notebooks/colab_E1_A2_probe.ipynb` trên Colab (GPU T4 hoặc A100), rồi chạy Ô 1 → Ô 6. Gửi lại file zip trong `MyDrive/CS431_MixA/export/`.
2. **User gửi Tuấn** `docs/msg_tuan_20261120.md` (đọc lại và sửa trước). Bốn điểm cần quyết: mốc tiny 42%, xử lý cổng khi khác GPU, xác nhận điểm tối ưu phẳng/R1, thân R và padding cho Mix B. Ghi câu trả lời vào kế hoạch chính.
3. Nếu probe PASS: chạy đầy đủ (runbook §4) → `reselect_split` (runbook §5) → viết báo cáo Mix A (R1 trong dashboard).
4. **Song song khi chờ GPU** (Mix B): chỉ sau khi Tuấn chốt thân R/padding mới tích hợp `mixab/fsr.py` vào FPNC/DBHead và viết probe Mix B. Train vẫn theo §5.1.
5. Model Stock hoãn cho tới khi có báo cáo Mix A-core.

## ĐIỂM CHỜ XÁC NHẬN

| Điểm | Trạng thái |
|---|---|
| Hướng Mix A trước, Mix B có điều kiện | Tuấn ủng hộ theo tin nhắn user chuyển lại; không suy ra là đã duyệt toàn bộ config |
| Output version, ngưỡng +0,5 pp, protocol R1 A→B | Cần khóa. README của bundle ghi gói này chưa chạy R1 A→B |
| **Mốc tiny ≥ 42%** | **Có bằng chứng là không ổn định** (`results/baseline_ab_v1/NOTES.md`); cần quyết lại |
| **Cổng 3a/3b khi chạy trên T4/A100 thay vì H200** | Chưa có dữ liệu; chờ probe; cách xử lý đề xuất ở §4.3, chưa thống nhất |
| Human review Bước 4 | PENDING, 0/160 thẻ có nhãn; review hiện ở `.30/.40` |
| Mix B: thân R, padding, scheduler, tiêu chí, phân công controls | Chưa thống nhất; giữ init B1 |

## SỰ THẬT ĐÃ KIỂM VÀ ĐƯỜNG DẪN

- **Cách cài môi trường (kiểm 20/11):** hướng dẫn chính thức ở `MonkeyOCRv2/detection/mmocr/README.md` + `install.sh` @ `da96afed…`.
  - Python 3.11, torch 2.9.0 cu128, mmengine 0.10.7, **MMCV 2.0.1 build từ source** (đổi c++14 thành c++17).
  - `install.sh` clone MMOCR v1.0.1 rồi chép add-on (`MonkeyOCRv2ViTAEBackbone`; patch chỉ đụng `pse_postprocessor`), cài mmdet 3.1.0 + transformers 4.57.1 + safetensors 0.7.0, và vá imgaug (NumPy 2) cùng mmengine (`torch.load`).
  - Trên HPC, DBPostprocessor của Tuấn nằm trong `upstream_MonkeyOCRv2/detection/mmocr/mmocr/...` (`sweep_run_identity.json`).
  - **File `db_postprocessor.py` của MMOCR v1.0.1 gốc trên GitHub có SHA `9bfff8aa…`, trùng bản Tuấn ghim** (tải bằng curl rồi hash).
- **Nguồn tải (kiểm 20/11):**
  - TextVQA `https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip`: 200 OK, 7.072.297.970 byte. Đọc mục lục từ xa bằng HTTP Range: tên file dạng `train_images/<id>.jpg`, khớp `clean_relpath`.
  - Hugging Face `zenosai/MonkeyOCRv2-AS`: có `model.safetensors`, `config.json` và code.
  - Hai repo GitHub đều public. Weights nằm trong Git LFS (`docs/handoff/weights/*.pth`).

- **Bundle:** `../../MONKEY_V2/docs/handoff/artifacts/pr_quad_audit_v1/`.
  - NPZ `(2262, 2, 72, 5)`: cond 0 = Clean, 1 = LR2; các cột `tp, gt, pred, tiny_gt, tiny_tp`.
  - `eda_common_counts.csv.gz`: `model, source_id, condition, sample_id, tp, gt, pred, covered, mask_thr, min_text_score`. Cổng 3a của mình cho PASS trên counts của Tuấn (4.524/4.524 mỗi endpoint).
  - `tasks_metadata.csv`: 4.524 task. `path_base` thuộc `TextVQA_clean_root` hoặc `prepared_root`; LR2 nằm ở `validation_lr2/<id>.png`.
  - `validation_remaining.jsonl.gz`: SHA-256 sau giải nén là `052857fb…` (mình đã kiểm trong `load_rows`).
- **SHA-256 export:** B-Clean `94f0e4ec…6aa`, B1 `9c3814e4…147`. Không dùng lẫn với hash checkpoint runner gốc.
- **Jobs của Tuấn:** 54308 FAILED sau sweep (01:06:37 trên H200, 3 model); recovery 63985 COMPLETED (00:04:07). Không dùng các con số này để ước lượng T4/A100.
- **Baseline toàn cohort** (bảng §0, mình đã tính lại khớp): B0 idx 44, B-Clean idx 70, B1 idx 70; B1 LR2 76,90%, FP 11.997, tiny 45,79%.
- **Các hàm MONKEY_V2 được gọi lại:**
  - `rev4.stage_a.detector_for/forward`: chỉ cần `model_config` và `checkpoint`; yêu cầu đúng 1 CUDA device và FP32.
  - `rev4.decision_core.capture_candidates/CandidateMatcher`: `score()` trả 4 bin.
  - `rev4.decision_plan.GEOMETRIES/SCORES`.
  - `rev4.training_data.native_lr2`: Pillow 11.3.0.
- Môi trường local: Python 3.14.4, torch CPU. Lệnh `--help` của các CLI đã sửa để chạy được trên console cp1252.

## CẤU TRÚC WORKSPACE

| Đường dẫn | Vai trò |
|---|---|
| `mixab/` | `weight_merge` (A1), `merge_audit` (A2), `reselect_split` (A3), `fsr` (Mix B), `monkey` (đường dẫn + commit MONKEY_V2) |
| `tests/` | 39 test; có test chạy trên bundle thật (tự skip nếu không có bundle) |
| `configs/` | Spec chạy, ví dụ `baseline_ab_v1.json` |
| `results/` | Kết quả nhỏ có ghi chú, sẽ đưa vào repo (ví dụ `baseline_ab_v1/`) |
| `docs/RUNBOOK_mix_a_T4_A100.md` | Hướng dẫn chạy trên GPU của user |
| `runtime/` | Trọng số và output nặng, **không** đưa vào repo |
| `HANDOFF.md` · `WORKLOG.md` · kế hoạch chính · `dashboard/` | Điều phối |
| `_archive/` | Lịch sử, không phải yêu cầu triển khai |
| `Đồ án/CLAUDE.md` + `Đồ án/.claude/` | Quy tắc và hook của session |
