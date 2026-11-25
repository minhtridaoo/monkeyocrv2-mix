# Hướng dẫn chạy đánh giá box-quality trên development A (Kaggle T4)

**Trạng thái (25/11/2026):**
- Đã có spec bổ sung, code, test (149/149 PASS), notebook và dataset.
- Kiểm ROI/ignore đã chạy (CPU): **không đáng kể**, 0,77% < 2%.
- `evaluate` đã chạy thử trên bản ghi giả dựng từ dump T4 (đặc trưng giả, số FULL vô nghĩa). Trên đủ 2.260 task × 3 cánh, evaluator chính thức trùng đường nhanh; F trùng dump.
- **Chưa chạy GPU.**

Spec: `docs/research/janek/boxquality_devA_spec_20261125_janek.md`. Đây là quyết định sau khi xem pilot. Pilot vẫn là `POSITIVE_BELOW_GATE`.

## 1. File (trong `cải tiến/mới/`)
| File | Dùng để |
|---|---|
| `runtime/kaggle_upload/cs431-boxquality-devA.zip` (615 MiB, SHA-256 `05907bbb7c5d077bd52195c205d2821633273c386dd2732f98fdb9682ef40ab1`; code manifest `5693b621…`) | Tải lên làm **Kaggle Dataset** (Private) |
| `notebooks/kaggle_boxquality_devA_janek.ipynb` | **Import** thành Kaggle Notebook |

Trong zip có:
- code `mixab`;
- B1, encoder, wheel MMCV (chép từ gói box-quality đã chạy, có kiểm SHA);
- scorer đóng băng `model/model_janek.npz` + `model/boxquality_fit_janek.json`;
- 1.130 ảnh Clean của A;
- dump T4 LR2 và Clean (B1 + B-Clean), dùng làm mốc cổng nhóm.

LR2 được dựng trên Kaggle và kiểm SHA với task đã khóa. **Không cần Internet để tải ảnh**, nhưng vẫn cần Internet để cài môi trường và clone repo.

## 2. Bật tùy chọn
- **Accelerator:** `GPU T4 x2` hoặc `GPU T4` (chỉ dùng GPU 0).
- **Internet:** **On**.
- **Add Input:** `cs431-boxquality-devA`. Chỉ gắn dataset này; không gắn kèm dataset box-quality cũ.

## 3. Chạy từng ô (thời gian **ước lượng, chưa đo**)
| Ô | Việc | Ghi chú |
|---|---|---|
| 1 | Tìm dataset, kiểm SHA từng file, kiểm GPU | < 1 phút |
| 2 | Cài môi trường (giống notebook box-quality) | 5–40 phút |
| 3 | Clone repo, kiểm B1 và scorer đóng băng, kiểm 1.130 ảnh Clean, dựng 1.130 LR2 (kiểm SHA), test CPU | vài phút |
| 4 | **Smoke 20 task:** kiểm nhất quán, evaluator = đường nhanh, so dump, thời gian thật | vài phút |
| 5 | Trích đủ 2.260 task A (lưu sau mỗi 50 task) | khoảng 20 phút, theo tốc độ pilot |
| 6 | Latency F với FULL, 200 ảnh × 2 lượt, kèm bộ nhớ | khoảng 6–10 phút |
| 7 | Đánh giá CPU, cổng nhóm, **nhãn khóa** | vài phút |
| 8 | Đóng gói zip | vài giây |

- **Dừng sau Ô 4** nếu ô in `DỪNG`. Gửi Claude output Ô 4, không tự đổi cấu hình.
- Ô 6: không chạy việc khác trong lúc đo.
- Số Hmean của 10 nguồn smoke ở Ô 4 không có ý nghĩa; không đọc.

## 4. Gửi lại
Gửi file `/kaggle/working/cs431_boxquality_devA_T4_<thời gian>.zip` (khung **Output**), hoặc báo Claude tải bằng Kaggle API. Claude sẽ:
- kiểm manifest, môi trường và mọi kiểm hợp lệ;
- tính lại `evaluate` ở local từ bản ghi đã tải;
- báo nhãn theo spec §5;
- không chỉnh gì sau khi xem.
