# Hướng dẫn chạy learned box refinement (Kaggle T4)

**Trạng thái (25/11/2026):** spec, code, test, notebook, dataset đã chuẩn bị; dry-run CPU đã chạy (không P-map, số vô nghĩa). **Chưa chạy GPU.**
Spec: `docs/research/janek/refine_spec_20261125_janek.md`.

## 1. File (trong `cải tiến/mới/`)
| File | Dùng để |
|---|---|
| `runtime/kaggle_upload/cs431-refine.zip` | Tải lên làm **Kaggle Dataset mới** (Private). Nhỏ: chỉ code, nhãn TRAIN, kết quả devA để đối chiếu. SHA ghi ở HANDOFF. |
| Dataset **`cs431-boxquality-devA`** đã có trên Kaggle | **Gắn lại, không tải lại.** Notebook kiểm SHA manifest phải trùng bản đã chạy box-quality. |
| `notebooks/kaggle_refine_janek.ipynb` | **Import** thành Kaggle Notebook |

## 2. Bật tùy chọn
- **Accelerator:** `GPU T4 x2` hoặc `GPU T4` (chỉ dùng GPU 0).
- **Internet:** **On** (cài môi trường, clone repo, tải 2.010 ảnh TRAIN từ zip TextVQA bằng HTTP Range như pilot box-quality).
- **Add Input:** `cs431-refine` **và** `cs431-boxquality-devA`.

## 3. Chạy từng ô
Thời gian là **ước lượng**, chưa profile. Ô 4 in số đo thật.

| Ô | Việc | Ghi chú |
|---|---|---|
| 1 | Tìm 2 dataset, kiểm SHA từng file, kiểm GPU | < 2 phút |
| 2 | Cài môi trường (y hệt notebook box-quality) | 5–40 phút |
| 3 | Clone repo, kiểm B1 và scorer v1, kiểm 1.130 ảnh A và dựng LR2, nhãn TRAIN, 10 ảnh smoke TRAIN, test CPU | vài phút |
| 4 | **Smoke:** 20 task TRAIN + 10 task A, kiểm ở mọi task (crop không đổi output v1, tọa độ quad) | vài phút |
| 5 | Tải 2.000 ảnh TRAIN, dựng LR2, trích 4.000 task (resume được) | pilot box-quality: khoảng 0,5 s/task; crop thêm chi phí, xem Ô 4 |
| 6 | Trích 2.260 task A (resume được) | như trên |
| 7 | Train MLP (CPU) trên fold fit, chọn checkpoint trên calibration | vài phút đến khoảng 30 phút (CPU Kaggle, chưa đo) |
| 8 | **Một lượt** đánh giá A + nhãn khóa | khoảng 10–20 phút CPU |
| 9 | Latency F / v1 / v1R, 200 ảnh × 2 lượt | khoảng 10–15 phút |
| 10 | Đóng gói zip | vài giây đến 1 phút |

- **Dừng sau Ô 4** nếu ô in `DỪNG`. Gửi Claude output Ô 4.
- **Dừng sau Ô 7** nếu in `NO_SIGNAL_ON_CALIBRATION`: không chạy Ô 8 và Ô 9, chạy Ô 10.
- Phiên sắp hết giờ thì chạy Ô 10 để lấy phần đã xong. Các ô 5–8 bỏ qua phần đã có.
- Ô 9: không chạy việc khác trong lúc đo.

## 4. Gửi lại
Gửi `/kaggle/working/cs431_refine_<GPU>_<thời gian>.zip` (khung **Output**), hoặc báo Claude tải bằng Kaggle API. Claude sẽ:
- kiểm manifest, môi trường và các kiểm hợp lệ;
- tính lại `select` và `evaluate` ở local từ bản ghi và checkpoint đã tải;
- báo nhãn theo spec §5, xem ca sửa tốt / sửa hỏng / box không nhãn;
- không chỉnh gì sau khi xem.
