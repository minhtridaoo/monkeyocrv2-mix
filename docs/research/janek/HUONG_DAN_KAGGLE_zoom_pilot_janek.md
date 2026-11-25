# Hướng dẫn chạy pilot ZB1 (zoom cục bộ bằng B1) trên Kaggle T4

**Trạng thái:** code, notebook và dataset đã tạo ở máy local, test CPU 107/107 PASS; **chưa chạy trên Kaggle**. Spec khóa trước: `docs/research/janek/zoom_pilot_spec_20261122_janek.md`.

## 1. File cần dùng
| File (trong `cải tiến/mới/`) | Dùng để |
|---|---|
| `runtime/kaggle_upload/cs431-zoom-pilot.zip` (336 MiB, SHA-256 `904cf105558f01d556d8c314c292e86a3c2062d4ea22d1c43baf384f7c496c1e`) | Tải lên làm **Kaggle Dataset** (Private) |
| `notebooks/kaggle_zoom_pilot_janek.ipynb` | **Import** thành Kaggle Notebook |

**Trong zip:**
- code `mixab`;
- **chỉ checkpoint B1** (`alpha_1p00.pth`, kèm `merge_manifest.json`);
- encoder;
- 300 ảnh Clean của mẫu A;
- dump T4 và counts A2r của B1, để kiểm nhất quán;
- wheel MMCV.

Repo của Tuấn được notebook tự clone. Dataset này khác dataset latency, nên tải lên riêng.

## 2. Bật tùy chọn
- **Accelerator:** `GPU T4 x2` (hoặc `GPU T4`). Notebook chỉ dùng GPU 0.
- **Internet:** On.
- **Add Input:** chọn dataset `cs431-zoom-pilot`.

## 3. Chạy từng ô (thời gian **chưa đo**, chỉ ước lượng)
| Ô | Việc | Ước lượng |
|---|---|---|
| 1 | Tìm dataset, kiểm SHA, kiểm GPU | < 1 phút |
| 2 | Cài môi trường (giống notebook latency) | 5–40 phút |
| 3 | Clone repo, vá config, kiểm SHA B1, tạo 300 ảnh LR2, chạy test | 2–5 phút |
| 4 | Kiểm ngắn ảnh trang báo, GT, evaluator (CPU) | vài phút |
| 5 | **Chạy thử 3 nguồn:** kiểm tọa độ 1×1, bộ nhớ, S2 có OOM không, **in thời gian dự kiến đo thật** | vài phút |
| 6 | Pilot 600 ảnh (GPU, lưu sau mỗi 20 ảnh) | khoảng 1–2 giờ (ước lượng) |
| 7 | Đánh giá (CPU) | 10–20 phút |
| 8 | Đóng gói zip | vài giây |

**Dừng sau Ô 5** nếu kiểm tọa độ sai, hoặc thời gian dự kiến vượt 4 giờ. Khi đó gửi Claude ảnh chụp output của Ô 5.

## 4. Gửi lại
- File `/kaggle/working/cs431_zoom_T4_<thời gian>.zip`, tải ở khung **Output**.
- Nếu Ô 2 đã build MMCV: tải thêm wheel trong `cs431_out/wheels/`.
- Nếu phiên sắp hết giờ trong lúc Ô 6 chạy: chạy Ô 8 để lấy phần đã xong.
