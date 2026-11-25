# Tin nhắn dự thảo gửi Tuấn (23/11, bản sửa sau khi chốt baseline) — CHƯA GỬI, user duyệt trước

Tuấn ơi, mình gửi kết quả đã kiểm lại của CS-B1 và mấy pilot hậu xử lý phía mình. Nói trước là **chưa có cái nào đạt cổng nhóm**.

**Tập ảnh: đã chạy đủ 2.262 nguồn LR2**
- Dump B1 và B-Clean trên **toàn bộ 2.262 ảnh LR2** của remaining-validation trong bundle `pr_quad_audit_v1`, không bỏ ảnh nào, không lỗi.
- Mình kiểm ID: A 1.130 + B 1.132 = `source_cohort.csv` = GT manifest (`052857fb…`). Chưa chạy Clean.
- Cấu hình: bản sao endpoint B1/B-Clean (`merge_manifest` `8582c9f3…`), MONKEY_V2 `dc19b85`, DBPostprocessor `9bfff8aa…`, config FP32. Chạy trên T4, batch 1. Chấm bằng `CandidateMatcher` của ông. Hmean tính từ tổng counts của đủ 2.262 nguồn.

**Baseline mình so** (chốt từ `decision_audit_report.md` và `counts/B1_source_counts.npz` của ông)
- B1 `.40/.60` (mask .40, score .60, eps .01) là điểm ông chọn lại trên lưới mở rộng: LR2 76,90%, FP 11.997. Mức +1,61 pp là so với `.30/.40` (75,29%). Trên đủ tập, `.40/.60` vừa thắng theo luật mean Clean + LR2 của ông, vừa là setting LR2 tốt nhất trong 72 setting.
- FP lệch nhẹ theo GPU: counts H200 của ông 11.997, A2r A100 của mình 11.999, dump T4 11.988. Có 32 nguồn lệch ±1 giữa H200 và A100; Hmean trùng đến 0,01 pp.

**Kết quả CS-B1@0.7, LR2, đủ 2.262 nguồn, CI ghép cặp theo nguồn, cả hai arm cùng một dump T4**
- CS-B1 = giữ toàn bộ box B1 `epsilon_005` (score ≥ .5), thêm box B-Clean `mask_040` (score ≥ .7) không trùng box B1 nào (IoU ≥ .5). Mình chọn biến thể này sau khi xem split A.
- So với B1 `epsilon_005/.50` (cấu hình nền của CS-B1; trên đủ tập là 76,86%): +0,23 pp [0,15; 0,32].
- So với B1 `.40/.60` của ông: 77,09 so với 76,91 → +0,18 pp [−0,21; +0,58], **không phân biệt được**. Trên A +0,41, trên B −0,04. Tiny −1,96 pp.
- Luật dừng khóa trước (so với B1 tốt nhất trong 72 setting trên A): +0,177 < 0,3 → **trượt**.
- Mình chưa so được với các checkpoint khác của ông (fusion ASF/nearest, adapter, EDSR): repo chưa có per-source counts của chúng, và chúng là trọng số khác.
- Latency trên Kaggle T4: 447 → 889 ms/ảnh (**1,99×**), bộ nhớ 894 → 991 MiB. Box khớp dump Colab T4 ở 200/200 ảnh.

**Pilot khác (đều âm hoặc trượt cổng)**
- MSB1 (tách box gộp bằng mask 0,4, 1 model): A +0,22 [−0,40; +0,81].
- Zoom cục bộ bằng B1 (300 nguồn A, cả Clean và LR2): lưới 2×2 cho LR2 −3,01, Clean −3,65 pp. Tiny +10,8 pp nhưng FP tăng mạnh.

**Một điểm về dữ liệu, chỉ ghi nhận** (mình không sửa gì)
- Nguồn `6e340fcbb1e0a877` (trang báo) có 10 GT, 0 ignore, ROI 100%, nhưng toàn bộ chữ thân bài không có nhãn.
- Riêng ảnh này chiếm 26,8% FP của B1 `epsilon_005/.50` trên A; 10 ảnh nhiều FP nhất chiếm 48,3%. Evaluator chính thức khớp `CandidateMatcher` trên ảnh này.
- Ông thấy có nên thêm một phân tích độ nhạy mô tả vào báo cáo không?

**Mình muốn hỏi ông**
1. Ông có đang định chạy EDSR ở `.40/.60` và đo Clean không? Mình không muốn chạy trùng. Nếu ông cần mình chạy trên Kaggle T4 (có đối chứng bicubic, cùng hậu xử lý) thì báo mình.
2. Theo ông, thầy tính "cải tiến" so với B0 hay so với B1 ở `.40/.60`?
3. Nhóm có chấp nhận hệ 2 checkpoint với latency khoảng 2× không? Nếu không thì CS-B1 chỉ để làm kết quả mô tả.
4. Ông có per-source counts của các arm fusion không? Có thì mình so ghép cặp được với CS-B1.

File: `docs/research/janek/ROADMAP_toan_bo_20261123_janek.md`, `docs/handoff/artifacts/janek/csb1_full_v1_T4/csb1_full_janek.json`, `docs/handoff/artifacts/janek/csb1_full_v1_T4/baseline_check_janek.json`, `docs/handoff/artifacts/janek/csb1_latency_v1_T4/latency_janek.json`, `docs/research/janek/zoom_pilot_result_20261123_janek.md`.
