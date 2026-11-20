# Nháp tin nhắn gửi Tuấn: kết quả Mix A-core v1 (user đọc lại và sửa trước khi gửi)

> Nguồn số liệu: `results/mix_a_v1/NOTES.md`, `decision.json`, `evaluation_B.csv`, `bootstrap_B.json`. Tất cả chạy trên Colab A100-SXM4-40GB, MONKEY_V2 `dc19b85`.

---

Tuấn ơi, Mix A-core v1 chạy xong rồi. Mình báo kết quả theo đúng quy tắc đã khóa.

**Kết luận: Mix A chưa thắng theo quy tắc đã khóa.**
- Trên A, R1 chọn α\* = 0, tức B-Clean (endpoint). Tổng Hmean Clean+LR2 trên A: B-Clean 157,02; α 3/4 156,60; α 1/2 156,52; α 1/4 156,35; B1 156,33.
- Vì α\* là endpoint nên không đi tiếp. Tiny của B-Clean trên B là 16,1%, cũng dưới mốc B1 − 3 pp = 34,6%.

**Kiểm chứng:**
- 5 model, gồm hai endpoint, chạy chung một job trên A100, cùng preprocessing, FP32, batch 1.
- P-map endpoint khớp tuyệt đối (sai khác 0,0).
- Counts của endpoint trùng 100% với probe A100 chạy ở phiên trước (0/5.760 ô lệch).
- Merge: endpoint giữ nguyên `state_dict` gốc; BN statistics nội suy tuyến tính, không ước lượng lại; `num_batches_tracked` lấy từ B1.
- So với H200 (chỉ báo cáo): khoảng 3% ô lệch ±1–7 polygon, `gt` không lệch ô nào, Hmean toàn cohort lệch ≤ 0,023 pp.
- Mốc tiny B1 − 3 pp mình đưa vào code trước khi mở file kết quả. Ô Colab lỡ chạy với code cũ (mốc 42%) thì chỉ giữ để đối chiếu, không ảnh hưởng α\*.

**Trên B, sau khi khóa (chỉ mô tả, không dùng để chọn lại):**

| Model | Clean H | LR2 H | Mean | LR2 FP | Tiny <4 px |
|---|---|---|---|---|---|
| B-Clean (α\*) | 81,16 | 74,59 | 77,87 | 3.211 | 16,1% |
| α 1/4 | 80,87 | 75,81 | 78,34 | 3.694 | 21,6% |
| α 1/2 | 80,27 | 76,28 | 78,28 | 4.451 | 32,5% |
| α 3/4 | 79,99 | 76,69 | 78,34 | 4.861 | 34,6% |
| B1 | 79,52 | 76,69 | 78,11 | 5.456 | 37,6% |

- Thứ tự endpoint đảo giữa A và B: B-Clean dẫn trên A khoảng 0,34 pp, B1 dẫn trên B 0,24 pp.
- Các α ở giữa cao hơn endpoint tốt nhất 0,17–0,23 pp, không α nào đạt +0,5 pp, và mọi CI paired 95% với từng baseline đều chứa 0.
- α 3/4 so với B1: Clean +0,47 pp (CI 0,20–0,88), LR2 bằng nhau, LR2 FP giảm 11%, nhưng tiny thấp hơn 3,07 pp.
- B không phải holdout mới, và đây chỉ là một seed.

**Mình đề xuất bước tiếp, Tuấn xem giúp:**
1. Đóng Mix A-core v1 bằng báo cáo kết quả âm, kèm đường α (Clean, LR2, FP, tiny theo α), rồi bàn giao.
2. **Không** đổi quy tắc chọn rồi chấm lại trên cùng B. Nếu muốn xem quy tắc "chọn có ràng buộc tiny trên A", mình chỉ ghi nó là phân tích độ nhạy, nói rõ là khám phá. Lý do: hàm chọn hiện tại không thưởng tiny, nên B-Clean thắng trên A ở setting có tiny thấp.
3. Model Stock: đường α khá phẳng (chênh lệch ≤0,5 pp, CI chứa 0), nên mình nghĩ kỳ vọng thấp. Mình đề xuất bỏ, để dồn thời gian. Tuấn đồng ý không?
4. Mix B: mình chờ Tuấn chốt thân R/padding để tích hợp. (Mức "giảm 11% FP" của α 3/4 không còn đúng khi so ở cùng recall, xem mục 5.)

---

**Phần bổ sung (sau góp ý của Tuấn về cách chọn và BN):**

5. **Chọn có ràng buộc tiny trên A (khám phá, không thay kết luận v1).** Luật: trên A, loại cấu hình có tiny thấp hơn B1 quá 3 pp, rồi tối đa mean Hmean.
   - B-Clean và α 1/4 bị loại hết. α\* = α 3/4.
   - Trên B, α 3/4 vẫn trượt: +0,23 pp so với B1, tiny −3,07 pp.
   - Ở cùng recall LR2, α 3/4 có nhiều FP hơn B1 (+6–7%). Như vậy "giảm 11% FP" trước đó chỉ là do recall thấp hơn, đúng như Tuấn lưu ý.
   - Kết luận: đổi cách chọn không tạo ra ứng viên đạt mục tiêu. Chi tiết: `results/explore_sel_v1/NOTES.md`.
6. **Pilot tính lại BN: mình mới thiết kế, chưa chạy** (`docs/design_bn_recalibration_pilot.md`).
   - Model chỉ có 4 BN, đều ở DBHead, và lúc inference chỉ 2 BN của nhánh `binarize` có tác dụng. Hai endpoint cách nhau 1,9% về trọng số, nên mình đoán hiệu ứng nhỏ.
   - Mình đề xuất làm bước chẩn đoán rẻ trước (khoảng 15 phút GPU), qua cổng mới đánh giá counts. Endpoint cũng được tính lại để làm đối chứng.
   - Tuấn giúp mình 3 việc:
     - (a) Chia sẻ manifest TRAIN `manifest__train__textvqa_clean_degraded_same_textocr_word_gt.jsonl` (SHA `ca1b9163…`), hoặc danh sách `source_image_id`/`clean_relpath`/`clean_sha256`. Mình chỉ dùng ảnh TRAIN để cập nhật BN.
     - (b) Duyệt cổng dừng sau chẩn đoán và ngưỡng "đáng xác nhận": α ở giữa tăng ≥ +0,3 pp, CI không chứa 0, hơn mức tăng của endpoint, tiny không thấp hơn B1 quá 3 pp.
     - (c) Chọn vòng xác nhận nếu có tín hiệu: đưa cấu hình đã khóa vào Bước 8 của Tuấn, hay nhóm còn tập validation nào chưa dùng?
