# Hướng dẫn chạy box-quality scorer trên Kaggle T4

**Trạng thái (24/11/2026):**
- Đã chuẩn bị ở local: spec, code, test (137/137 PASS), notebook, dataset.
- Phần CPU đã chạy thử trên dữ liệu giả: `fit`, `smokecheck`, dựng lại nhãn từ chính zip.
- **Chưa chạy GPU.**

Spec khóa trước: `docs/research/janek/boxquality_spec_20261124_janek.md`. Kết quả, nếu có, chỉ là **sàng lọc trên TRAIN-calibration**, không phải cải tiến đã xác nhận.

## 1. File (trong `cải tiến/mới/`)
| File | Dùng để |
|---|---|
| `runtime/kaggle_upload/cs431-boxquality.zip` (238 MiB, SHA-256 `42be0acca2db2c375fcd3230e666e1cbcfac6a31f96ef1ef8d9757910d5acaf2`; code manifest `a5077b9d…`) | Tải lên làm **Kaggle Dataset** (Private) |
| `notebooks/kaggle_boxquality_janek.ipynb` | **Import** thành Kaggle Notebook |

Trong zip có:
- code `mixab` (gồm `vendor/shared_manifest_janek.py`);
- checkpoint B1 và encoder;
- wheel MMCV;
- `inputs/`: nhãn TRAIN, danh sách chọn, `train_ids.csv`, kết quả P0;
- 5 ảnh dev và dump T4 của B1, chỉ để kiểm pipeline.

**Ảnh TRAIN không có trong zip.** Notebook tải đúng 2.010 ảnh (549 MiB) từ zip TextVQA chính thức theo từng file, kiểm CRC + SHA-256. TextOCR train JSON (280 MB) cũng được tải và kiểm SHA.

## 2. Bật tùy chọn
- **Accelerator:** `GPU T4 x2` hoặc `GPU T4` (notebook chỉ dùng GPU 0).
- **Internet:** **On** (bắt buộc, để tải ảnh và TextOCR).
- **Add Input:** `cs431-boxquality`.

## 3. Chạy từng ô (thời gian **chưa đo**)
| Ô | Việc | Ghi chú |
|---|---|---|
| 1 | Tìm dataset, kiểm SHA, kiểm GPU | < 1 phút |
| 2 | Cài môi trường (giống các notebook trước) | 5–40 phút |
| 3 | Clone repo, kiểm B1, tải TextOCR, **dựng lại nhãn TRAIN và so với dataset**, tải 10 ảnh smoke, tạo LR2, test CPU | vài phút |
| 4 | **Smoke 20 task** + `smokecheck` + `devcheck` (5 ảnh dev LR2 so với dump T4) + chi phí thật | vài phút |
| 5 | Tải 2.000 ảnh, dựng LR2, trích 4.000 task (lưu sau mỗi 50 task) | Ô 4 in dự kiến đo thật |
| 6 | Fit + chọn ngưỡng trên calibration (CPU), nhãn kết quả, bảng review | chưa đo |
| 7 | Đóng gói zip | vài giây |

**Dừng sau Ô 4** nếu ô in `DỪNG`: có kiểm nhất quán sai, hoặc dự kiến > 3 giờ. Gửi Claude output Ô 4, không tự đổi cấu hình.

## 4. Gửi lại
File `/kaggle/working/cs431_boxquality_T4_<thời gian>.zip` (khung **Output**), hoặc báo Claude tải bằng Kaggle API. Có thể điền cột `reviewer_label` trong `fit/review/review_sheet_janek.csv` (`text_missing_label` | `background` | `unsure`). Phần này không bắt buộc và không đổi nhãn.
