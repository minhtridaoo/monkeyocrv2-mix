# ZB1 (zoom cục bộ bằng B1) · kết quả pilot sàng lọc trên Kaggle T4 (23/11/2026)

> **Kết luận theo spec khóa trước: KHÔNG ĐẠT cổng sàng lọc → dừng nhánh zoom cục bộ.** Không đổi arm chính, không chỉnh ngưỡng sau khi xem số.
> Spec: `docs/zoom_pilot_spec_20261122.md`. User chạy notebook `kaggle_zoom_pilot.ipynb` trên Kaggle (kernel `ojanek/notebook5b3e8ce4a1`, 22/11). Claude tải output về bằng Kaggle API (chỉ đọc).
> Số liệu: `results/zoom_pilot_v1_T4/`. Máy: Kaggle 2×T4, chỉ dùng GPU 0. torch 2.9.0+cu128, mmcv 2.0.1, mmocr 1.0.1, Pillow 11.3.0. DBPostprocessor SHA đúng bản ghim. MONKEY_V2 `dc19b85` sạch.
> Code chạy trên Kaggle đúng bản đã đóng gói: manifest SHA `581e5660…` khớp, `zoom.py` local trùng dataset.

## 1. Kiểm tính đúng của lần chạy
- **Kiểm tọa độ 1×1:** 6/6 ảnh tái tạo đúng box của ảnh đầy đủ (`smoke_summary.json`).
- **Kiểm nhất quán:**
  - F trên LR2 so với dump T4 (Colab): **0/300** nguồn lệch.
  - F trên Clean so với counts A2r (A100): 3/300 nguồn lệch (do khác GPU).
- 600/600 task xong, 10,75 s/task, khoảng 107 phút.
- S2 (cạnh ngắn 2048) **OOM ở 6/600 task**; đỉnh bộ nhớ 10,5 GB. Lượt ô zoom có đỉnh 1,9 GB.

## 2. Kết quả (mẫu 300 nguồn A, chỉ sàng lọc)

| Arm | Forward | LR2 Hmean | Δ LR2 (CI95) | Clean Hmean | Δ Clean (CI95) | Tiny LR2 | ms/ảnh (median, tuần tự) |
|---|---|---|---|---|---|---|---|
| F (B1) | 1 | 70,47 | — | 72,39 | — | 66,0% | 460 |
| **G2 (chính)** | 5 | 67,47 | **−3,01 [−6,39; +0,66]** | 68,74 | **−3,65 [−6,99; +0,47]** | 76,8% (+10,8) | 2.227 |
| G3 | 10 | 67,79 | −2,69 [−5,28; +0,33] | 68,14 | −4,25 [−7,76; +0,19] | 73,9% (+8,0) | 4.409 |
| S2 (297 nguồn) | 2 | 68,39 | −2,04 [−4,30; +0,72] | 68,39 | −3,97 [−7,16; +0,39] | 77,9% (+11,9) | 4.111 |
| R3 | 4 | 68,17 | −2,31 [−4,46; +0,30] | 69,71 | −2,68 [−5,03; +0,23] | 71,8% (+5,8) | 1.830 |

- **G2 trên LR2:** TP 7.364 → 7.687 (+323), FP 3.593 → 5.159 (+1.566).
- **G2 trên Clean:** TP +446, FP +2.007.

**Cổng sàng lọc G2:**

| Cổng | Kết quả |
|---|---|
| LR2 ≥ +0,3 pp và cận dưới CI > 0 | **không đạt** |
| Clean ≥ −0,3 pp | **không đạt** |
| Tiny ≥ −3 pp | đạt (+10,8) |

→ **KHÔNG ĐẠT.**

- **Không có hiệu ứng liều dương:** mọi arm đều âm. G3 (phóng khoảng 2,3×) không tốt hơn G2 (khoảng 1,5×). Theo spec mục 7, không có căn cứ đề xuất spec mới cho nhánh này.
- Biến thể không giới hạn kích thước (chỉ mô tả) còn âm hơn.

## 3. Cơ chế (mô tả)
- **Zoom thấy thêm chữ nhỏ thật.**
  - Tiny recall LR2 tăng từ 66,0 lên 76,8%, Clean từ 79,1 lên 90,9%.
  - Có 501 GT là FN của F mà box G2 khớp được (120 trong số đó là tiny).
  - Giả thuyết "kích thước chữ trên đầu vào detector là nút thắt của recall" **được ủng hộ về recall**.
- **Nhưng phần lớn box thêm vào bị evaluator tính là FP.**
  - G2 trên LR2: 1.312/1.553 FP của box thêm (84%) có proxy "không chồng GT hợp lệ" (nền **hoặc chữ thiếu nhãn**).
  - Proxy không phân biệt được hai trường hợp này; cần chấm tay.
- **Ảnh trang báo dày chữ** (`6e340fcbb1e0a877`, nằm trong mẫu theo quy tắc hash):
  - 10 GT; B1 đã có 1.796 box trên LR2; G2 thêm 949 box (LR2) và 919 box (Clean).
  - Riêng ảnh này làm Hmean của F trên mẫu chỉ còn 70,47 (bỏ ảnh này thì 77,08).
- **Mô tả khi bỏ riêng ảnh này** (`descriptive_without_dense_page.json`, không đổi kết luận):

  | Arm | Δ LR2 | Δ Clean |
  |---|---|---|
  | G2 | −0,40 | −1,59 |
  | G3 | −0,87 | −2,43 |
  | S2 | −0,73 | −2,10 |
  | R3 | −0,90 | −1,60 |

  → **Kết quả âm không chỉ do ảnh đó.**

## 4. Kiểm ngắn ảnh trang báo, GT, evaluator (`pagecheck.json`, `pagecheck_dense_source.png`)
- Nguồn `6e340fcbb1e0a877`: Clean 1024×652, **10 GT hợp lệ, 0 GT ignore**, ROI phủ 100% ảnh.
  - Hình chồng cho thấy GT chỉ nằm ở tiêu đề và một dòng. Toàn bộ chữ thân bài (đọc được trên Clean) **không có nhãn và không bị đánh dấu ignore**.
- `CandidateMatcher` khớp evaluator chính thức trên ảnh này (`assert_official` PASS; B1 LR2 counts [6, 10, 1796]).
- **Trên toàn A:** ảnh này chiếm **26,8%** FP của B1 (`epsilon_005/0.50`); 10 ảnh nhiều FP nhất chiếm **48,3%**.
- **Ghi nhận:** điểm Hmean trên A nhạy mạnh với vài ảnh có nhãn thiếu.
  - Không sửa GT, không loại ảnh, không đổi protocol.
  - Đây là điểm nên báo Tuấn (chưa gửi).

## 5. Hệ quả
- Dừng nhánh zoom cục bộ theo luật dừng.
- Ghi vào báo cáo như **kết quả âm có cơ chế rõ**: zoom tăng recall chữ nhỏ khoảng 11 pp, nhưng FP trên vùng chữ không có nhãn lớn hơn phần TP cứu được. Chi phí 5 lần forward (khoảng 4,8× thời gian, đo tuần tự).
- Giới hạn: mẫu 300 nguồn, chỉ sàng lọc; thời gian đo tuần tự, không phải benchmark latency công bằng.
