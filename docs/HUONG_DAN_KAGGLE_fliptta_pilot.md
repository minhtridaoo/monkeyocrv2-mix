# Hướng dẫn chạy pilot Flip-TTA (P-map, B1) trên Kaggle T4

**Trạng thái (23/11/2026):** code, notebook, dataset đã tạo ở máy local; test CPU 124/124 PASS; đường `evaluate` đã chạy thử trên bản ghi giả (đã xóa, số không có ý nghĩa). **Chưa chạy trên GPU/Kaggle.** Spec khóa trước: `docs/fliptta_spec_20261123.md`. Bạn đã báo sẽ chạy **Clean CS-B1 trước**, rồi mới chạy pilot này.

## 1. File cần dùng (trong `cải tiến/mới/`)
| File | Dùng để |
|---|---|
| `runtime/kaggle_upload/cs431-fliptta-pilot.zip` (337 MiB, SHA-256 `81e6b94d1c9d808d403f314aa1b648310e2aad6c6a98625b06b0f2f6846c6b93`; code manifest `bc0835c6…`) | Tải lên làm **Kaggle Dataset** (Private) |
| `notebooks/kaggle_fliptta_pilot.ipynb` | **Import** thành Kaggle Notebook |

Trong zip: code `mixab`; **chỉ checkpoint B1**; encoder; 300 ảnh Clean của mẫu A (chồng lấn 88 nguồn với mẫu zoom, chỉ mô tả); dump T4 và counts A2r của B1 (kiểm nhất quán); wheel MMCV. Repo Tuấn được notebook tự clone.

## 2. Bật tùy chọn
**Accelerator** `GPU T4 x2` (hoặc `GPU T4`), notebook chỉ dùng GPU 0 · **Internet On** · **Add Input** → `cs431-fliptta-pilot`.

## 3. Chạy từng ô (thời gian **chưa đo**)
| Ô | Việc | Ước lượng |
|---|---|---|
| 1 | Tìm dataset, kiểm SHA, kiểm GPU | < 1 phút |
| 2 | Cài môi trường (giống notebook trước) | 5–40 phút |
| 3 | Clone repo, vá config, kiểm SHA B1, tạo 300 ảnh LR2, test CPU | 2–5 phút |
| 4 | **Chạy thử 3 nguồn:** kiểm căn chỉnh cứng, thời gian F/T, bộ nhớ, dự kiến toàn pilot | vài phút |
| 5 | Pilot 600 ảnh (lưu sau mỗi 20 ảnh) | chưa đo; Ô 4 in số đo thật |
| 6 | Đánh giá (CPU) | vài phút |
| 7 | Đóng gói zip | vài giây |

**Dừng sau Ô 4** nếu kiểm căn chỉnh báo không đạt, hoặc dự kiến > 3 giờ. Gửi Claude ảnh chụp output Ô 4. Không tự đổi cấu hình.

## 4. Gửi lại
File `/kaggle/working/cs431_fliptta_T4_<thời gian>.zip` (khung **Output**), hoặc báo Claude tải bằng Kaggle API. Claude tiếp nhận theo spec (kiểm code manifest, kiểm căn chỉnh, báo nhãn theo luật khóa).
