# Spec khóa trước · Pilot Flip-TTA trên P-map (B1, 1 checkpoint) · 23/11/2026

> **Trạng thái:** spec + code + notebook đã chuẩn bị ở local; **chưa chạy GPU**. User sẽ chạy trên Kaggle (sau lượt Clean của CS-B1).
> Pilot khám phá trên **mẫu 300 nguồn của split A** (A đã dùng nhiều lần), **không phải xác nhận độc lập**.
> TTA là kỹ thuật phổ biến; **chưa tự động tính là đóng góp phương pháp mới**. Không quét ngưỡng, không chỉnh luật sau khi xem số.

## 1. Kiểm khả thi (đã làm, chỉ đọc code)

| Câu hỏi | Kết quả | Nguồn |
|---|---|---|
| Lấy được P-map trước hậu xử lý? | **Có.** `forward()` gọi `detector.model(**processed, mode='tensor')` ra `probability` FP32 dạng `[1,H,W]` trên lưới đã pad; hậu xử lý chạy riêng từ tensor đó (`pp(probability, [sample])`, `capture_candidates(native, probability, sample, restore, geometry)`). Nên có thể thay P-map bằng bản trung bình rồi chạy đúng hậu xử lý cũ | `MONKEY_V2/rev4/stage_a.py:212-235`, `rev4/decision_core.py:20-31` |
| B1 có train với lật ngang không? | **Có (theo cấu hình).** Pipeline train của B1 là bản sao `cfg.train_pipeline` chỉ đổi bước nạp ảnh; `train_pipeline` có `ImgAugWrapper` với `Fliplr 0.5` (+ Affine ±10°, Resize 0,5–3,0). Bảng so sánh của Tuấn ghi "flip/rotate/scale" cho cả B-Clean và B1 | `rev4/b1_training.py:31-35`; `docs/handoff/inference_config.py:296`; `docs/bclean-and-paired-eda.md:25`. **Chưa kiểm** recipe thật trên server, chỉ cấu hình trong repo |
| Đã thử chưa? | **Chưa có TTA/flip-ở-suy-luận** trong repo của Tuấn (các chỗ "flip" khác đều là căn chỉnh augment lúc train) và trong workspace của Trí | grep `MONKEY_V2`, `cải tiến/mới` ngày 23/11 |
| Padding | `data_preprocessor.pad_size_divisor = 32`; padding ở **phải/dưới**; `img_shape` trong `metainfo` là kích thước vùng nội dung sau Resize | `inference_config.py:127-131`; `stage_a.py:228-232` |

**Rủi ro cần ghi nhận (chưa phải lý do loại):**
- Lật làm chữ bị soi gương; B1 chỉ được train lật với xác suất 0,5, chưa biết chất lượng trên chữ lật ở LR2 nhỏ.
- Trung bình P-map làm mềm đỉnh xác suất, có thể làm chữ nhỏ tụt dưới ngưỡng `.60` hoặc lệch biên với `mask_thr .40`.
- Hai lượt forward cho một checkpoint: chi phí thực tế sẽ đo (không nhận 2× theo suy đoán).
- Cùng sắp đặt với phép trung bình hai quan sát: có thể giảm nhiễu nhưng cũng có thể giảm recall.

## 2. Cách làm (khóa)
- **Model:** B1 endpoint (`alpha_1p00.pth`, SHA trong merge manifest), FP32, batch 1, một GPU T4 (GPU 0).
- **Hậu xử lý (cả hai cánh, không đổi):** B1 `.40/.60` = hình học `mask_040` (mask .40, unclip 1,5, min_width 5, eps .01), score cutoff **0,60**, cùng `capture_candidates` + `CandidateMatcher` của Tuấn.
- **Cánh F (baseline):** một forward trên ảnh gốc, P-map `P0`.
- **Cánh T (TTA):** 
  1. `P0` như trên;
  2. lật ngang **chỉ vùng nội dung** `[:h, :w]` của tensor đầu vào đã tiền xử lý (`h,w = img_shape`); vùng pad giữ nguyên → không lấy mẫu lại, không dịch nửa điểm ảnh;
  3. forward → `P1`; lật ngược vùng nội dung → `P1b` (về đúng tọa độ);
  4. `P = (P0 + P1b)/2` trong vùng nội dung; vùng pad giữ `P0`;
  5. hậu xử lý như F trên `P`.
  Trung bình số học, không trọng số, không max; không tham số nào để chỉnh.

## 3. Kiểm căn chỉnh (BẮT BUỘC trước khi tin số chất lượng; cứng, chạy trong chính pilot)
Với **mỗi** ảnh (600):
- **(a) Pad hằng:** vùng ngoài nội dung của tensor đầu vào chỉ có một giá trị duy nhất (nên lật nội dung không đổi pad).
- **(b) Lật hai lần = nguyên bản:** `flip(flip(x)) == x` chính xác trên tensor đầu vào.
- **(c) Cùng metainfo:** `img_shape`, `scale_factor` giống nhau giữa hai lượt (cùng tệp ảnh, cùng Resize).
- **(d) Kích thước P-map:** `P1b` cùng shape với `P0`, hữu hạn.
Chẩn đoán (báo, không chỉnh): IoU nhị phân (ngưỡng 0,4, vùng nội dung) giữa `P0` và `P1b` (đã căn) so với giữa `P0` và `P1` (chưa lật ngược, đối chứng); sai khác trung bình tuyệt đối.
**Cổng căn chỉnh (cứng):** (a)–(d) đúng ở mọi ảnh **và** trung vị IoU đã căn **>** trung vị IoU đối chứng. Trượt → nhãn `INCONCLUSIVE_ALIGNMENT`, **không** diễn giải chất lượng.
Ô chạy thử 3 nguồn (6 ảnh) chạy cùng kiểm này và dừng nếu (a)–(d) sai.

## 4. Mẫu và dữ liệu
- **300 nguồn của split A** chọn bằng hash: nhỏ nhất `sha256('20261123:fliptta:' + source_id)` (không dùng thông tin kết quả). Danh sách được ghi vào `KAGGLE_MANIFEST.json`. Cả **Clean và LR2** (600 ảnh; LR2 tạo bằng `native_lr2` của Tuấn, kiểm SHA).
- Mẫu độc lập với danh sách zoom (seed khác); mức chồng lấn sẽ báo khi đánh giá (chỉ mô tả).
- **Không mở B, không dùng benchmark cuối, không GPU ngoài lượt Kaggle của user.**

## 5. Báo cáo bắt buộc (mỗi điều kiện Clean/LR2, F so với T)
Hmean (từ tổng counts), TP/FP/FN, recall, precision, tiny recall (`<4 px`) và recall theo bin; Δ Hmean kèm CI95 ghép cặp theo nguồn (bootstrap 2.000, seed 431); chênh lệch **theo từng nguồn** (CSV: ΔTP, ΔPred, đóng góp ròng) và số nguồn lên/xuống; **latency thực đo** (trung vị ms/ảnh của F và T, tỉ số, bộ nhớ đỉnh); kiểm nhất quán F với dump T4 (LR2) và A2r A100 (Clean) ở `mask_040/.60` (chỉ báo tỉ lệ lệch, không nới cổng). Kết quả mô tả chia theo ảnh trang báo `6e340fcb…` nếu có trong mẫu (chỉ mô tả).

## 6. Tiêu chí khóa (so với F cùng lượt chạy)
- **c1:** LR2 ΔHmean ≥ **+0,30 pp** và cận dưới CI95 > 0.
- **c2:** Clean ΔHmean ≥ **−0,30 pp**.
- **c3:** LR2 tiny recall Δ ≥ **−3 pp**.
- Latency: chỉ báo, không có cổng (kỳ vọng khoảng 2×, chưa đo).
- Cổng nhóm (+0,5 pp mean hệ thống) **không** đánh giá ở pilot này.

**Nhãn kết quả** (tính theo thứ tự này, không chỉnh sau khi xem số):
1. `INCONCLUSIVE_ALIGNMENT`: trượt cổng căn chỉnh hoặc thiếu ảnh.
2. `NEGATIVE`: LR2 ΔHmean ≤ 0.
3. `INCONCLUSIVE`: ΔHmean > 0 nhưng cận dưới CI95 ≤ 0 (chưa phân biệt được với 0 ở n = 300; **chưa đủ bằng chứng**, không phải âm cũng không phải dương).
4. `PASS_SCREEN`: đủ c1–c3. Chỉ nghĩa là đáng làm bước sau; **không** nghĩa là đạt cổng nhóm hay xác nhận.
5. `POSITIVE_BELOW_GATE`: các trường hợp còn lại (Δ > 0, CI > 0 nhưng dưới +0,30, hoặc trượt c2/c3). Báo rõ cờ nào trượt và recall/tiny thay đổi bao nhiêu.

**Dừng:** không tự mở rộng mẫu, không đổi ngưỡng/cách gộp (max, trọng số), không thêm cấu hình sau khi xem số. Nếu `PASS_SCREEN`, chỉ **đề xuất** (chưa chạy) bước tiếp trên toàn A cùng luật khóa, và đăng ký trước cho benchmark cuối cùng Tuấn.
