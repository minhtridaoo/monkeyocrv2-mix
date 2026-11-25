# Spec khóa trước · ZB1 (Zoom-B1) · pilot sàng lọc trên Kaggle T4 (22/11/2026)

> Khóa **trước khi** chạy. Sửa sau khi chạy thì ghi ở cuối file, kèm lý do.
> **Phạm vi:** một checkpoint B1, không train, không SR (chưa dựng EDSR), không dùng GT khi suy luận. Pilot chỉ để **sàng lọc**, không phải xác nhận cải tiến cuối.
> **Nơi chạy:** user tự chạy notebook `notebooks/kaggle_zoom_pilot_janek.ipynb` trên Kaggle. Local chỉ chuẩn bị code và test nhẹ.

## 0. Đối chiếu repo và cơ chế resize (đã kiểm, nguồn có ghi)
- **Resize hiện tại:** `test_pipeline` ở `MONKEY_V2/docs/handoff/inference_config.py:199-204` dùng `Resize(keep_ratio=True, scale=(4068, 1024))`.
  - Hệ số phóng là `min(1024/cạnh_ngắn, 4068/cạnh_dài)`, tức cạnh ngắn về 1024.
  - Ảnh LR2 512×384 và ảnh Clean 1024×768 đều thành tensor 1365×1024. **Một chữ cao h px LR2 chiếm khoảng 2,67·h px trên đầu vào detector ở cả hai điều kiện.**
- **Phóng cả ảnh là phép tương đương.** Bicubic ×2 toàn ảnh của Tuấn cho 75,24 so với native 75,29 (`MONKEY_V2/handoff.md:74`). Ảnh to gấp đôi vẫn bị Resize đưa về cạnh ngắn 1024.
- **Một crop có cạnh ngắn s thì được phóng 1024/s**, nên chữ thật sự to hơn trên đầu vào detector, trừ khi cạnh dài chạm trần 4068 (ghi lại hệ số thực của từng crop).
- **Repo chưa có zoom cục bộ hay tile inference cho detector.** `grep` các từ tile/zoom/sliding/tta trong `rev4/`, `lowres/`: chỉ thấy tile trong `rev4/edsr.py` (chia tile để chạy EDSR, không phải detector) và crop để vẽ thẻ review. Dùng lại được: `detector_for` (nạp model và kiểm config FP32), DBPostprocessor gốc, `CandidateMatcher`, `fp_proxy`.
- **Lý do đáng thử** (counts A2r A100, split A, B1 `epsilon_005/0.50`), recall theo bin chiều cao (LR2 px):

  | Bin | < 4 | 4–8 | 8–16 | ≥ 16 |
  |---|---|---|---|---|
  | Clean | 61,2% | 80,6% | 84,3% | 75,3% |
  | LR2 | 48,8% | 72,6% | 80,4% | 73,5% |

  Ở Clean, chữ tiny có đủ chi tiết pixel mà recall vẫn chỉ 61%, dù kích thước trên tensor bằng LR2. Gợi ý (suy luận): **kích thước chữ trên đầu vào detector là một nút thắt**, độc lập với việc mất thông tin. Phần chênh 12,4 pp giữa Clean và LR2 là mất thông tin; zoom bằng bicubic không bù được phần đó.
- **Rủi ro đã biết:**
  - MSB1 cho thấy thêm box trên chữ không có nhãn sinh nhiều FP. Một ảnh chữ dày (10 GT, 1.796 box B1) chiếm khoảng 27% FP của B1 trên A.
  - Zoom sẽ tìm thêm chữ nhỏ không có nhãn.
  - Chi phí là nhiều lượt suy luận.

## 1. Tên
**ZB1** (Zoom-B1). Arm chính: `ZB1-G2` (lưới 2×2).

**Khác MSB1:** MSB1 chỉ hậu xử lý lại **cùng một P-map**. ZB1 chạy **thêm lượt forward** trên vùng ảnh được phóng, nên tạo P-map mới có độ phân giải cao hơn cho chữ nhỏ.

## 2. Mẫu (quy tắc khóa, không dùng kết quả)
- 300 nguồn của **split A**: 300 hạng nhỏ nhất theo `sha256("20261122:zoom:" + source_id)`.
- Mỗi nguồn chạy **cả Clean và LR2**, tổng 600 task. Không dùng B. Không chọn ảnh theo lỗi hay theo khả năng được cứu.

## 3. Suy luận (một checkpoint B1 = `alpha_1p00.pth`, batch 1, FP32)
- **Hậu xử lý** mọi lượt: DBPostprocessor gốc, geometry `epsilon_005` (mask 0,30, unclip 1,5, min_width 5, epsilon 0,005), `min_text_score` 0,50. Giống hệt baseline.
- **Đọc ảnh:** `mmcv.imread(..., flag='color_ignore_orientation')`, giống `LoadImageFromFile` của pipeline. Crop ghi ra PNG (không mất dữ liệu) rồi đưa qua đúng pipeline gốc.
- **Các arm:**

| Arm | Mô tả | Lượt forward / ảnh | Hệ số phóng so với ảnh đầy đủ (ước lượng) |
|---|---|---|---|
| **F** (baseline) | ảnh đầy đủ | 1 | 1 |
| **G2** (chính) | F + lưới 2×2 | 5 | khoảng 1,54× |
| G3 (phụ, liều lượng) | F + lưới 3×3 | 10 | khoảng 2,31× |
| S2 (phụ, đối chứng phóng toàn ảnh) | F + cả ảnh với `Resize scale=(8136, 2048)` | 2 (lượt thứ hai khoảng 4× pixel) | 2× |
| R3 (phụ, chọn vùng) | F + 3 trong 9 ô của G3, chọn bằng dự đoán (mục 5); mô phỏng từ chính output G3, không chạy thêm GPU | 4 | khoảng 2,31× |

- S2 bị OOM trên T4 → ghi "không chạy được trên T4", không thay bằng cấu hình khác.
- **Nhiễu đã biết, không tách riêng được:** `min_width` 5 và các ngưỡng hình học tính bằng px của tensor. Ở lượt zoom (phóng k lần), box nhỏ hơn k lần trên ảnh gốc cũng qua được min_width. Đây là một phần của hiệu ứng "chữ to hơn trên đầu vào detector", không phải yếu tố độc lập được kiểm riêng.

## 4. Chia ô, padding, biên, quy đổi tọa độ (khóa)
- Ảnh task W×H (pixel của ảnh Clean hoặc LR2). Lưới n×n: cạnh ô lõi `x_i = round(W·i/n)`, `y_i = round(H·i/n)`.
- **Crop** = ô lõi nở thêm `ceil(0,15·w_ô)` theo x và `ceil(0,15·h_ô)` theo y, cắt trong biên ảnh. Không phản chiếu, không thêm viền.
- **Giữ box của ô** khi đủ cả hai điều kiện:
  - tâm box (trung bình 4 đỉnh, tọa độ task) nằm trong ô lõi (nửa mở `[x0, x1)`; ô cuối đóng ở biên ảnh);
  - không đỉnh nào cách **cạnh crop nằm trong ảnh** dưới 1 px (bỏ chữ bị cắt).

  Mỗi vị trí chỉ thuộc một ô.
- **Quy đổi:** polygon từ postprocessor ở tọa độ crop. Cộng offset `(cx0, cy0)` rồi nhân `restore = clean_size/task_size` để về tọa độ Clean, giống `forward()` của Tuấn.
- **Kiểm tọa độ bắt buộc ở ô chạy thử:** lưới 1×1 (crop = cả ảnh qua PNG) phải tái tạo **đúng** polygon của F. Không khớp thì dừng.

## 5. Hợp nhất box (khóa, một luật cho mọi arm)
1. Giữ toàn bộ box của F.
2. Box zoom (từ ô hoặc từ S2) được **xét thêm** nếu cạnh ngắn hình chữ nhật xoay nhỏ nhất, quy về px LR2 (như bin của evaluator), **< 8 px**. Giả thuyết nhắm chữ nhỏ; chữ lớn để ảnh đầy đủ lo vì ô dễ cắt chữ lớn.
3. Duyệt box zoom theo score giảm dần (hòa thì theo thứ tự ô và chỉ số). Thêm box nếu IoU lớn nhất với mọi box đã giữ **< 0,5**.

Không có tham số nào được quét. 0,15 · 1 px · 8 px · 0,5 · 0,50 khóa ở đây.

- **R3 (chọn vùng không dùng GT):** xếp 9 ô G3 theo số box F (score ≥ 0,5, cạnh ngắn LR2 < 8 px) có tâm trong ô lõi, giảm dần; hòa thì theo chỉ số ô. Lấy 3 ô đầu.
  - Rủi ro: R3 chỉ nhìn lại chỗ lượt đầu đã thấy chữ nhỏ, nên **có thể bỏ qua chữ mà lượt đầu không thấy**. So R3 với G3 để đo phần bị bỏ qua.

## 6. Đánh giá (CPU trên Kaggle)
- **Evaluator:** `CandidateMatcher` của Tuấn trên đúng tập box đầu ra (cutoff 0), có ROI và ignore. Kiểm `assert_official` trên vài ảnh.
- **Báo theo từng arm, riêng Clean và LR2:**
  - Hmean, TP/FP/FN, recall theo 4 bin (tiny < 4 px);
  - Δ so với F, kèm CI95 bootstrap ghép cặp theo nguồn (2.000 lần, seed 431);
  - mean hệ thống `(Clean + LR2)/2`;
  - số lượt forward, thời gian mỗi lượt và mỗi arm (đo tuần tự, chưa phải benchmark latency công bằng), bộ nhớ đỉnh.
- **Kiểm nhất quán:** counts của F trên LR2 so với dump T4 (Colab) cho cùng nguồn; counts của F trên Clean so với counts A2r (A100). Chỉ báo tỉ lệ lệch.
- **Chẩn đoán dùng GT, báo riêng:**
  - tỉ lệ GT là FN của F mà có box zoom (mọi kích thước, score ≥ 0,5) khớp IoU ≥ 0,5, theo bin, cho từng arm;
  - proxy FP (`fp_proxy` của Tuấn) của box zoom được thêm;
  - biến thể không giới hạn kích thước (bỏ bước 2 của mục 5), chỉ mô tả.

## 7. Cổng sàng lọc (chỉ cho ZB1-G2, đánh giá trên mẫu 300 nguồn của A)
1. LR2: ΔHmean ≥ **+0,30 pp** **và** cận dưới CI95 > 0.
2. Clean: ΔHmean ≥ **−0,30 pp**.
3. Tiny recall LR2 ≥ F − 3 pp.

- **Đạt cả 3:** đề xuất (chưa chạy) một lượt đủ 1.130 nguồn A với đúng G2 đã khóa, cộng đo latency công bằng. Sau đó mới xét **cổng cuối của nhóm** (giữ nguyên: mean hệ thống ≥ model đơn tốt nhất + 0,5 pp, Clean ≥ B-Clean − 0,3 pp, tiny ≥ B1 − 3 pp, kèm báo đủ chi phí).
- **G2 trượt:** dừng nhánh zoom cục bộ. G3, S2, R3 chỉ báo mô tả. Không đổi arm chính sau khi xem số.
  - Nếu G3 hoặc S2 cho thấy hiệu ứng liều rõ (Δ tăng theo hệ số phóng), chỉ được **đề xuất** một spec mới khóa trước. Đề xuất đó phải ghi rõ đã chọn sau khi xem mẫu pilot, và mẫu pilot nằm trong A.
- **Dừng sớm:** kiểm tọa độ 1×1 sai; hoặc ô chạy thử báo thời gian dự kiến của toàn pilot vượt 4 giờ (khi đó báo user, không tự giảm N).

## 8. Bước kiểm ngắn: ảnh trang báo, GT, evaluator (CPU, cùng notebook)
- Mục đích: ghi nhận bối cảnh, **không** đổi kết quả hay protocol đã khóa.
- Với nguồn `6e340fcbb1e0a877`:
  - số GT hợp lệ và GT ignore, tỉ lệ ROI, kích thước ảnh;
  - số box B1 (dump), FP;
  - so counts `CandidateMatcher` với `evaluate_image` chính thức;
  - hình chồng GT hợp lệ/ignore lên ảnh Clean.
- Trên A: phần FP của B1 thuộc 1 và 10 nguồn nhiều FP nhất. Chỉ mô tả.

---
## Nhật ký sửa spec sau khi chạy
- (Trước khi chạy, khi đóng gói dataset) Nguồn chữ dày `6e340fcbb1e0a877` **nằm trong mẫu 300 nguồn** theo đúng quy tắc hash. Giữ nguyên, không loại. Nguồn này có thể kéo FP của các arm zoom lên mạnh; khi báo cáo sẽ ghi rõ, kèm mô tả có/không có nguồn này (chỉ mô tả, cổng vẫn tính trên đủ 300 nguồn).

- (23/11, sau khi chạy) Không sửa luật, arm hay cổng. Kết quả: `docs/research/janek/zoom_pilot_result_20261123_janek.md` — G2 trượt cổng sàng lọc, dừng nhánh. Phần tính "bỏ riêng ảnh trang báo" là mô tả sau khi xem số, không dùng cho cổng.
