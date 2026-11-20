# HANDOFF — trạng thái hiện tại (đọc đầu mỗi session)

> Hiện trạng ghi ở đây; lịch sử chỉ append vào [WORKLOG.md](WORKLOG.md). Phạm vi và protocol theo [kế hoạch chính](Ke_hoach_phoi_hop_MixA_MixB_20261119.md); đọc §0 trước khi làm. Không lấy số toàn cohort làm baseline trên B.

## TRẠNG THÁI HIỆN TẠI

- **Mới nhất (20/11, Claude):** user chuyển hướng: giữ kết luận âm v1; khám phá cách chọn trên CPU, rồi mới xét pilot BN. Luật được ghi ở kế hoạch §4.4 trước khi chạy.
  - **E-sel xong** (`results/explore_sel_v1/NOTES.md`; code `mixab/explore_selection.py`, 5 test). Nhãn **khám phá**.
    - Chọn có ràng buộc tiny trên A loại B-Clean và α 1/4 (0 setting khả thi); α\* = α 3/4.
    - Trên B, α 3/4 vẫn **trượt**: +0,23 pp so với B1 (+0,46 so với B-Clean), tiny −3,07 pp.
    - Ở cùng recall LR2, α 3/4 có **nhiều FP hơn** B1: +7,2% trên A, +6,0% trên B.
    - Kết luận: đổi cách chọn không tạo ra ứng viên đạt mục tiêu.
  - **E-BN mới là thiết kế** (`docs/design_bn_recalibration_pilot.md`), chưa có code GPU, chưa chạy.
    - Sự thật đã kiểm: chỉ 4 lớp BN, đều trong DBHead; inference chỉ dùng 2 lớp `binarize` (MMOCR v1.0.1 mode `predict`); hai endpoint cách nhau 1,9% về trọng số.
    - **Bị chặn:** manifest TRAIN không có trong repo Tuấn.
    - Kỳ vọng thấp, nên có Giai đoạn 0 chẩn đoán rẻ để dừng sớm.
- **Kết luận Mix A-core v1 giữ nguyên:** α\* = B-Clean, chưa thắng theo quy tắc đã khóa. `results/mix_a_v1/` có trạng thái `VERIFIED` (same-GPU A100).
- **Tuấn đã chốt** (kế hoạch §4.2 và §4.3): phương án (a) cùng A100; R1 và tie-break; +0,5 pp so với cả hai baseline; tiny ≥ B1 − 3 pp trên cùng split B; báo CI paired; B không phải holdout.
- **Test:** 60/60 PASS.
- **Nháp tin nhắn Tuấn** `docs/msg_tuan_20261120_mixA_result.md` (kết quả v1) **chưa gửi**. Cần thêm: E-sel, xin manifest TRAIN, xin duyệt thiết kế E-BN.
- **Ràng buộc:** chỉ T4/A100; MONKEY_V2 chỉ để tham chiếu (bản hiện tại `aa1e17f`, run Mix A v1 ghim `dc19b85`); không submit GPU nếu user chưa bảo.

| Phần mình | Hiện trạng | Còn thiếu |
|---|---|---|
| Mix A-core v1 (A1–A4) | Xong, `VERIFIED`; kết quả âm | Hình đường α cho báo cáo R1 |
| E-sel (khám phá) | Xong; không có ứng viên đạt | — |
| E-BN (pilot BN) | Thiết kế xong | Manifest TRAIN; nhóm duyệt cổng/ngưỡng/vòng xác nhận; rồi mới code + Colab |
| A5 Model Stock | Gated | Đề xuất bỏ; chờ Tuấn |
| Mix B `mixab/fsr.py` | Torch thuần, có test CPU | Thân R/padding chờ Tuấn; train theo §5.1 |

## BƯỚC TIẾP THEO (theo thứ tự)

1. **User gửi Tuấn**, có thể ghép một tin, gồm 4 ý:
   - (a) kết quả v1 (`docs/msg_tuan_20261120_mixA_result.md`);
   - (b) tóm tắt E-sel;
   - (c) xin manifest TRAIN (`manifest__train__textvqa_clean_degraded_same_textocr_word_gt.jsonl`, SHA `ca1b9163…`) hoặc danh sách ID/SHA;
   - (d) xin duyệt §6–§8 của `docs/design_bn_recalibration_pilot.md`: cổng Giai đoạn 0, ngưỡng "đáng xác nhận", và cách xác nhận (Bước 8 hay tập mới).
2. **Claude, không cần GPU:** vẽ đường α và hoàn thiện báo cáo Mix A (R1), gồm v1 và phụ lục E-sel.
3. **Khi có manifest TRAIN và được duyệt:** viết `mixab/bn_recal.py` và test CPU, thêm ô Colab, probe 40 nguồn. Sau đó chạy Giai đoạn 0 (khoảng 15 phút). Chỉ chạy Giai đoạn 1 nếu qua cổng.
4. **Nếu E-BN dừng ở Giai đoạn 0 hoặc không đạt:** đóng Mix A bằng báo cáo âm; chuyển sang Mix B khi Tuấn chốt thân R.
5. **R2 bàn giao** trước 24/11; chỉ push khi user báo.

## ĐIỂM CHỜ XÁC NHẬN

| Điểm | Trạng thái |
|---|---|
| Hướng Mix A trước, Mix B có điều kiện | Tuấn ủng hộ theo tin nhắn user chuyển lại; không suy ra là đã duyệt toàn bộ config |
| Output version, ngưỡng +0,5 pp, protocol R1 A→B | Cần khóa. README của bundle ghi gói này chưa chạy R1 A→B |
| **Mốc tiny** | **Đã chốt 20/11:** α\* ≥ B1 − 3 pp trên cùng split B (thay 42%) |
| **Cổng 3a/3b khi chạy trên A100 thay vì H200** | **Đã chốt (a)** ngày 20/11 (user chuyển lời Tuấn). Cách triển khai (P-map + `gate_same_gpu`; H200 → `cross_gpu_drift`) là đề xuất của mình, ghi ở §4.3; Tuấn chưa duyệt chi tiết |
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
