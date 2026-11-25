# Kết quả box-quality scorer (Kaggle T4, TRAIN-calibration) · 25/11/2026

Spec khóa trước: `docs/research/janek/boxquality_spec_20261124_janek.md`. Bản chính: `docs/handoff/artifacts/janek/boxquality_v1_T4/`. Output gốc: `runtime/kaggle_downloads/boxquality_T4_20261124-224054/` (kernel `ojanek/cs431-csb1-boxquality-fixed`).

**Đây là sàng lọc trên TRAIN-calibration (B1 đã train trên TRAIN), không phải kết quả development, không phải xác nhận, không phải cải tiến.** Số T4 → `PENDING_REFERENCE_CHECK`.

## 1. Tiếp nhận
- **Gói chạy là bản tái cấu trúc `_janek`** (`cs431-boxquality-janek-fixed.zip`, code manifest `8d3caafe…`), **không phải** gói 24/11 (`a5077b9d…`).
  - Đã diff mọi module sau khi bỏ hậu tố `_janek`. Chỉ khác ở đường dẫn trong docstring, khoảng trắng, alias import, và `monkey_root()` thêm một ứng viên (không dùng trên Kaggle vì `MONKEY_V2_ROOT` đã đặt). **Logic `boxquality` không đổi.**
  - Notebook của kernel chỉ thêm phần chẩn đoán SHA, khôi phục `train_rows.gzip.bin` → `.jsonl.gz` có kiểm SHA (`dbe1668d…`, đúng nhãn gốc) và `PYTHONPATH`. Lệnh smoke/extract/fit và tham số giữ nguyên.
- **Môi trường:** Pillow 11.3.0, DBPostprocessor `9bfff8aa…`, 1 GPU T4, torch 2.9.0+cu128, MONKEY_V2 `dc19b85` sạch.
- **Kiểm nhất quán (đều đạt):**
  - đường đặc trưng trùng tuyệt đối `capture_candidates`: 20/20 task smoke, 80/80 task kiểm trong lượt trích;
  - evaluator chính thức khớp (smoke);
  - **devcheck 5/5** ảnh dev LR2 trùng tuyệt đối dump T4 (raw lệch 0);
  - trích đủ **4.000 task**, 177.020 ứng viên.
- **Tính lại ở local** (`mixab.boxquality_janek fit` trên bản ghi đã tải): mọi count, Δ, CI và nhãn **trùng tuyệt đối**. `beta_full` lệch ≤ 5·10⁻¹⁴ và ngưỡng FULL lệch 9·10⁻¹⁶ (dấu phẩy động, khác BLAS).

## 2. Kết quả (400 nguồn calibration, Clean + LR2)
Ứng viên dùng để học: 142.436 (fit 117.103, calibration 25.333); tỉ lệ dương có trọng số trên fit 68%.

| Arm | Ngưỡng | Clean H | Clean TP / FP | LR2 H | LR2 TP / FP | mean | tiny LR2 | tiny Clean |
|---|---|---|---|---|---|---|---|---|
| F (B1 `.40/.60`) | raw 0,60 | 76,98 | 8.200 / 1.937 | 72,92 | 7.279 / 1.518 | 74,95 | 29,07% | 48,14% |
| RAW (ngưỡng chọn trên calibration) | raw 0,598 | 76,99 | 8.215 / 1.959 | 72,95 | 7.292 / 1.534 | 74,97 | 29,07% | 48,35% |
| SCORE_ONLY | (tương đương 0,598) | 76,99 | 8.215 / 1.959 | 72,95 | 7.292 / 1.534 | 74,97 | 29,07% | 48,35% |
| **FULL** | 0,332 | **77,13** | 8.167 / 1.844 | **73,09** | 7.273 / 1.462 | **75,11** | 28,76% | 46,80% |

| So sánh (mean Hmean Clean/LR2) | Δ | CI95 ghép cặp theo nguồn (2.000, seed 431) |
|---|---|---|
| **FULL − RAW** | **+0,14 pp** | **[+0,0004; +0,29]** |
| FULL − F | +0,16 pp | [+0,02; +0,31] |
| RAW − F | +0,02 pp | [−0,02; +0,06] |
| SCORE_ONLY − RAW | 0,00 | [0; 0] → đối chứng hợp lệ |

**Nhãn khóa: `POSITIVE_BELOW_GATE`.** Δ dương, cận dưới CI chỉ vừa trên 0, nhưng dưới cổng sàng lọc +0,30 pp.

## 3. Đọc đúng
- **Không phải calibration:** RAW tự chọn ngưỡng 0,598, gần như trùng 0,60 của Tuấn (RAW − F = +0,02). Phần +0,14 của FULL đến từ thông tin ngoài raw score.
- **Cơ chế (mô tả):** FULL bớt FP (Clean −115, LR2 −72 so với RAW) và mất ít TP hơn (−48 / −19). Hai điều kiện tăng như nhau (+0,14 / +0,14). Tiny giảm: LR2 −0,31 pp, Clean −1,55 pp.
- **Hệ số không diễn giải riêng được:** các đặc trưng xác suất tương quan mạnh với nhau (ví dụ `pmax` −2,10, `quad_mean` −1,57, `std` +1,80, `ring_mean` +1,56).
- **Giới hạn của chính phép đo:**
  - calibration là TRAIN mà B1 đã học, nên phân bố ứng viên lạc quan hơn dev;
  - ngưỡng của cả RAW và FULL được chọn và đánh giá trên **cùng** 400 nguồn (thiên lệch lạc quan như nhau cho hai cánh);
  - nhãn TRAIN dùng ROI = toàn ảnh.
- **Thiếu nhãn:** quy tắc review chọn được 3 ca "FP điểm cao, không chồng GT", đều cùng một bìa sách. Cả 3 là chữ thật ("ASIMOV", "DAVID") nhưng **không có nhãn**. Không đổi nhãn; chỉ ghi nhận rủi ro có thật.
- **Chi phí** (từ thời gian lượt trích, chưa phải benchmark latency công bằng):
  - post-process có đặc trưng: median 26,9 ms so với 16,2 ms thường (+10,8 ms); p95 194 so với 71 ms;
  - forward (gồm tiền xử lý) median 403 ms;
  - nên đầu-cuối khoảng **+2,6% ở median**, đuôi p95 nặng hơn ở ảnh nhiều chữ;
  - VRAM đỉnh 1.731 MiB; một checkpoint, một forward.

## 4. Kết luận theo spec
- Spec §7: chỉ `PASS_CALIBRATION_SCREEN` mới cho phép đề xuất P3 (một lượt dev đã khóa). Kết quả là `POSITIVE_BELOW_GATE` → **dừng theo luật**.
- Không thêm đặc trưng, model hay tham số; không chạy dev; không mở TEST.
- **Được tuyên bố:** "trên TRAIN-calibration, scorer học được từ P-map của cùng forward hơn ngưỡng raw đã chọn lại +0,14 pp mean Hmean (CI [0,0004; 0,29]), dưới cổng +0,30; chi phí thêm khoảng 3% median."
- **Không được tuyên bố:** cải tiến trên dev/TEST, hay vượt cổng nhóm.
- Muốn đi tiếp thì cần quyết định mới của user (và Tuấn), ghi rõ là ngoài luật đã khóa. Ví dụ: một lượt dev duy nhất để **mô tả**, với model và ngưỡng đã khóa từ lượt này, không chỉnh gì.
