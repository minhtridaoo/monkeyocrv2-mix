# Spec khóa trước · Learned box refinement trên B1 + box-quality v1 · 25/11/2026

> **Trạng thái:** oracle (CPU, A) và thống kê TRAIN đã chạy; code, test, notebook, dataset đã chuẩn bị. **Chưa chạy GPU, chưa train, chưa có số của model.**
> Đây là adaptation của bounding-box regression (dịch cạnh chuẩn hóa theo kích thước box). Không phải Cascade R-CNN, không phải kiến trúc mới.
> Box-quality v1 `SMALL_USEFUL` và mọi kết quả/cổng cũ giữ nguyên. Không mở B/TEST.

## 1. Pipeline
```
ảnh → B1 (frozen, FP32, batch 1, một forward) → P-map + ứng viên mask_040 (raw ≥ 0,20)
    → box-quality v1 giữ nguyên (scorer, ngưỡng cut_full 0,3316, kiểm SHA) → tập box v1 + score cũ
    → mỗi box: crop P-map trong hệ tọa độ box → MLP nhỏ → 4 độ dịch cạnh → quad mới (cùng góc, cùng score)
    → evaluator chính thức
```
- Một checkpoint detector, một forward. Không train backbone/FPNC/DBHead.
- Không tách box, sinh box, xóa box, NMS, chấm lại score hay quét ngưỡng.
- Không dùng B-Clean teacher. Học bằng GT TRAIN.

## 2. Kiểm dư địa (CPU, đã chạy trước mọi thiết kế model)
Script `scripts/refine_oracle_janek.py` → `docs/handoff/artifacts/janek/refine_oracle_20261125/refine_oracle_janek.json`. Oracle **dùng GT, chỉ chẩn đoán**, không phải phương pháp.

**Luật oracle (khóa trước khi tính):**
- Giữ nguyên tập box v1 và score; không xóa FP, không sinh, không tách.
- Box được sửa khi có ghép một-một hợp lệ: không phủ ≥ 50% của ≥ 2 GT (không phải box gộp); IoU tốt nhất ≥ 0,1 và ≥ 2× IoU thứ hai; là box có IoU tốt nhất với GT đó (không phải duplicate).
- Đích = khoảng GT chiếu lên hệ tọa độ box (giữ góc), clamp |dịch| ≤ 0,5 cạnh (chính). Matching tính lại toàn ảnh bằng evaluator chính thức.

**Phân loại box v1 trên A (Clean / LR2):**

| Loại box | Clean | LR2 |
|---|---|---|
| ghép một-một, đang TP (IoU ≥ 0,5) | 29.136 | 27.042 |
| ghép một-một, chưa TP (IoU 0,1–0,5): **đối tượng chính** | 1.054 | 1.082 |
| gộp nhiều GT | 1.093 | 1.017 |
| duplicate (không phải box tốt nhất của GT) | 648 | 779 |
| chồng yếu / mơ hồ | 396 | 270 |
| không chồng GT (phần lớn là chữ thiếu nhãn, theo Giai đoạn 0) | 5.008 | 3.962 |

**Kết quả oracle (A):**

| Cánh | Clean | LR2 | mean | GT cứu / mất (Clean · LR2) |
|---|---|---|---|---|
| v1 (FULL) | 79,39 | 76,97 | 78,18 | — |
| **sửa cạnh, bound 0,5 (chính)** | **81,28** | **79,21** | **80,24** | 950 / 248 · 967 / 167 |
| bound 0,25 (độ nhạy) | 81,46 | 79,20 | 80,33 | 846 / 76 · 857 / 59 |
| bound 1,0 (độ nhạy) | 81,19 | 79,26 | 80,23 | 957 / 287 · 1.004 / 185 |
| dịch đều mọi cạnh theo box, tối ưu IoU (kiểu unclip riêng từng box) | 81,16 | 78,83 | 79,99 | 660 / 5 · 661 / 0 |
| Cổng nhóm | ≥ 80,22 | — | ≥ 79,00 | |

- Oracle chính − v1: mean **+2,07 pp [+1,69; +2,48]** (bootstrap theo nguồn). Clean dư so với cổng +1,06 pp; mean +1,24 pp.
- **Để đạt cổng, model cần bắt được khoảng 44% lợi ích oracle trên Clean (0,83/1,89) và 40% trên mean (0,82/2,07)**, mà không làm hỏng box đúng. Oracle cao không chứng minh model học được.
- **Rủi ro làm hỏng:** 1.698 (Clean) / 1.467 (LR2) box TP có IoU chỉ 0,5–0,6. Ngay oracle chính cũng làm mất 248 / 167 GT, vì đích "khoảng GT trong hệ tọa độ box" không tối ưu IoU khi GT xoay khác box. Đây là giới hạn của chính đích hồi quy.
- Số Pred đổi vì ROI/ignore khi sửa: −12 (Clean), −5 (LR2) ở oracle chính.

**TRAIN (CPU, `train_stats_janek.json`, cùng luật):** oracle trên 400 nguồn calibration: Clean 77,13 → 78,96 (+1,84), LR2 73,09 → 75,17 (+2,08), cùng cỡ với A. Fit fold có 75.162 cặp TP và 3.425 cặp chưa TP.
- Cặp chưa TP cần dịch lớn: trung vị |dịch| khoảng 0,25 cạnh; 76% có ít nhất một cạnh > 0,5; 26–30% > 1,0.
- Cặp TP: trung vị khoảng 0,05; chỉ 2% vượt 0,5.

**Đối chiếu hướng đã thử:**
- **unclip / min_width (22/11):** đổi tham số chung cho mọi box. min_width 3/7 không thêm GT nào có thể đạt; unclip 1,2/1,8 chỉ thêm khoảng 127 GT. Bộ sửa này khác ở chỗ học độ dịch **riêng cho từng cạnh của từng box** từ P-map. Ngay cả "dịch đều riêng từng box" với GT (oracle) cũng kém sửa cạnh 0,25 pp mean, chủ yếu ở LR2 (78,83 so với 79,21).
- **MSB1 (22/11, âm):** thay box gộp bằng ≥ 2 box của mask .40, tức **thêm box**. 60% lần tách không đổi TP mà chỉ thêm box. Bộ sửa này **không đổi số box** và không đụng box gộp: nó bỏ ngỏ nhóm gộp (1.093 / 1.017 box), chỉ nhắm box một-một lệch.

## 3. Thiết kế (khóa; căn cứ hình học và TRAIN, không từ A)
Code: `mixab/refine_geom_janek.py`, `mixab/refine_janek.py`, `mixab/refine_cmd_janek.py`.

| Mục | Giá trị khóa | Căn cứ |
|---|---|---|
| Hệ tọa độ | ánh xạ song tuyến từ quad (p0..p3) → hình vuông đơn vị; box mới = ảnh của [d_l, 1+d_r] × [d_t, 1+d_b] | giữ góc; offset 0 trả đúng quad gốc (test); không tự cắt, không đảo chiều |
| Giới hạn dịch | \|d\| ≤ 0,5 cạnh; cạnh mới ≥ 10% cạnh cũ, nếu không thì fallback quad gốc (đếm) | oracle chính; 0,25 tốt hơn trên Clean nhưng không được chọn theo A |
| Crop | P-map ở lưới mạng, 12 × 40 mẫu, s, t ∈ [−0,5; 1,5] (ngữ cảnh = giới hạn dịch), song tuyến, ngoài map = 0 | phủ đúng vùng box có thể dịch tới |
| Đầu vào phụ | log cạnh ngắn, log cạnh dài (px mạng), log aspect, raw score | |
| Model | MLP 484 → 256 → 256 → 4, ReLU; lớp cuối khởi tạo 0 | ban đầu trả lại box gốc |
| Loss | Smooth-L1 (β 0,05) trên đích clamp ±0,5; trọng số đều mỗi cặp; **không bỏ cặp nào** | TRAIN: luật "bỏ cặp \|đích\| > 1" (bản nháp) sẽ bỏ 26–30% cặp chưa TP, nên đổi trước khi chạy GPU |
| Optimizer / lịch | AdamW lr 1e-3, wd 1e-4, batch 512, 30 epoch, cosine, seed 431; checkpoint ở epoch 5, 10, …, 30 | |

**Nhãn train (chỉ fold fit TRAIN, ROI = toàn ảnh như box-quality):**
- Box v1 trên TRAIN (cùng scorer đóng băng), lọc ROI/ignore như evaluator.
- Chỉ cặp ghép một-một theo luật §2, gồm cả box đang TP (học giữ ổn định) và box chưa TP (học sửa).
- Không dùng box không chồng GT làm "nền", không dạy xóa. Không lấy box gộp làm đích.
- **Selection bias:** model chỉ thấy box có GT, nhưng khi chạy gặp cả box không nhãn (5.008 trên A Clean). Báo độ dịch của model theo từng loại box (`moved`) trên calibration và A.

## 4. Chọn checkpoint (chỉ TRAIN-calibration, 400 nguồn × Clean/LR2)
- Ứng viên: e05, e10, …, e30. Mục tiêu: mean Hmean chính thức (Clean, LR2) từ tổng counts. Hòa → epoch sớm hơn.
- Phải hơn offset 0 (= v1) trên calibration. Nếu không: nhãn `NO_SIGNAL_ON_CALIBRATION`, **không đánh giá A**.
- Báo kèm GT cứu/mất và độ dịch theo loại box.

## 5. Một lượt đánh giá A đóng băng
- Cánh: **F** = B1 `.40/.60`; **v1** = box-quality v1; **v1R** = v1 + refinement (cùng tập box, cùng score, chỉ đổi tọa độ).
- Kiểm hợp lệ: F và v1 phải trùng counts `devA_result_janek.json`.
- Báo: Clean/LR2/mean, TP/FP/FN, recall theo chiều cao (tiny < 4 px), GT cứu/mất, CI ghép cặp theo nguồn (2.000, seed 431), fallback, độ dịch theo loại box.
- **Cổng nhóm** (mốc dump T4 cùng A): g1 mean ≥ 79,0019; g2 Clean ≥ 80,2202; g3 tiny LR2 ≥ F − 3 pp.
- **Nhãn khóa** (theo v1R − v1, mean):
  - `GROUP_GATE_PASS`: đủ 3 cổng và cận dưới CI > 0;
  - `NO_BENEFIT`: Δ ≤ 0;
  - `INCONCLUSIVE`: cận dưới CI ≤ 0;
  - `POSITIVE_BELOW_GATE`: còn lại.
- Latency: 100 nguồn A × 2 điều kiện × 2 lượt, thứ tự xoay vòng 3 cánh; median/p95, phần sửa (chuyển P-map về CPU, crop, MLP), VRAM đỉnh.
- Không chỉnh tiếp theo điểm A trong lượt này.

## 6. Kiểm bằng code (CPU, `tests/test_refine_geom_janek.py`, `tests/test_refine_janek.py`)
- Offset 0 tái tạo chính xác quad (cả quad không phải hình bình hành).
- Đích → box: khôi phục đúng box xoay.
- 500 offset ngẫu nhiên: không tự cắt, không đảo chiều, diện tích dương.
- Fallback: không hữu hạn, sụp cạnh, không phải quad.
- Crop của map tuyến tính khớp giải tích; ngoài map = 0.
- Model khởi tạo trả 0; fit có loss hữu hạn, giảm, lưu checkpoint đúng epoch.
- **Dry-run CPU** (`scripts/refine_dryrun_janek.py`, không P-map, số vô nghĩa): train + select chạy hết trên TRAIN; trên A, checkpoint chưa train cho v1R = v1 và F/v1 trùng `devA_result`.
- **Trên GPU (Ô smoke):** cứ 25 task, một lượt `bq.run_task` thứ hai phải cho record trùng tuyệt đối (crop không đổi output B1/v1); quad mạng tính lại từ polygon cuối phải trùng quad của post-processor (< 0,01 px) và nằm trong vùng ảnh.
