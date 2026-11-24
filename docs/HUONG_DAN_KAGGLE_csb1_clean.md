# Hướng dẫn chạy bổ sung Clean cho CS-B1@0.7 (Kaggle T4) · tùy chọn

**Lượt này giải quyết:** CS-B1 áp cho mọi ảnh (không định tuyến) có làm giảm Clean không, và mean hệ thống khi không dùng shortcut kích thước ảnh.

**Lượt này không giải quyết:**
- ý nghĩa thống kê trên LR2 so với B1 đã calibration (CI hiện chứa 0);
- cổng nhóm;
- thiên lệch do đã xem A/B nhiều lần.

Spec khóa trước: `docs/csb1_clean_check_spec_20261123.md`.

**Trạng thái:** notebook và dataset đã tạo, test CPU 108/108 PASS; **chưa chạy trên Kaggle**.

## File
| File (trong `cải tiến/mới/`) | Dùng để |
|---|---|
| `runtime/kaggle_upload/cs431-csb1-clean.zip` (939 MiB, SHA-256 `337f4b06991f64ebb09875099f3c8c099a908c08067e4b2b68323955b1450eb0`) | Kaggle Dataset (Private) |
| `notebooks/kaggle_csb1_clean.ipynb` | Import thành Notebook |

**Trong zip:**
- code;
- 2 checkpoint (`alpha_0p00` = B-Clean, `alpha_1p00` = B1) + `merge_manifest.json`;
- encoder;
- dump LR2 T4 có sẵn;
- counts A2r (A100) của 2 model;
- wheel MMCV.

**Ảnh Clean có sẵn trong zip** (2.262 ảnh, chép đúng byte từ cache Drive `textvqa_val2262_clean.zip`, kiểm SHA khi đóng gói). Ô 3 kiểm SHA lại từng ảnh; chỉ khi thiếu hoặc sai mới tự tải zip TextVQA chính thức (7,07 GB) làm dự phòng.

## Bật tùy chọn
- **GPU T4 x2** (hoặc T4); chỉ dùng GPU 0.
- **Internet On.**
- **Add Input:** dataset `cs431-csb1-clean`.

## Chạy (thời gian chưa đo)
| Ô | Việc | Ước lượng |
|---|---|---|
| 1–2 | Kiểm dataset, cài môi trường | 5–40 phút |
| 3 | Clone repo, kiểm SHA checkpoint, kiểm SHA 2.262 ảnh Clean có sẵn, chạy test | 3–8 phút (thêm 10–20 phút nếu phải tải dự phòng) |
| 4 | Probe 40 ảnh Clean: thời gian thật, độ lệch so với A2r | vài phút |
| 5 | Dump 2.262 ảnh Clean × 2 model | khoảng 55 phút |
| 6 | Đánh giá CS-B1 trên Clean + hệ thống không định tuyến (CPU) | khoảng 10 phút |
| 7 | Đóng gói zip | vài giây |

**Dừng sau Ô 4** nếu thời gian dự kiến của Ô 5 vượt 3 giờ. Chạy xong thì báo Claude tải kết quả bằng Kaggle API, hoặc gửi file zip.
