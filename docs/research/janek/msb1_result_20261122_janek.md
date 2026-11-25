# MSB1 (Mask-Split B1) · kết quả pilot CPU (22/11/2026)

> **Kết luận theo luật khóa: KHÔNG ĐẠT** cổng LR2 trên A. Ghi là **kết quả âm**; không chỉnh 0,80 hay 0,50, không thêm luật.
> Spec khóa trước khi chạy: `docs/research/janek/msb1_spec_20261122_janek.md`. Code: `mixab/msb1_janek.py`, test `tests/test_msb1_janek.py`; toàn bộ test 100/100 PASS. Số liệu: `docs/handoff/artifacts/janek/msb1_v1_T4/msb1_result_janek.json`. Ca minh họa: `docs/handoff/artifacts/janek/msb1_v1_T4/examples/`.
> Dump T4 → mọi số là PENDING_REFERENCE_CHECK. Ý tưởng hình thành **sau khi xem Phase 0 trên A**, nên mọi tín hiệu dương chỉ là khám phá.

## 1. Luật (nhắc lại)
- Giữ toàn bộ box B1 `epsilon_005` có score ≥ 0,5 (box cha).
- Box cha nào chứa ≥ 2 box B1 `mask_040` có score ≥ 0,5, mỗi box nằm ≥ 80% diện tích trong box cha (box con), thì được thay bằng các box con đó.
- Một model, không dùng GT, không dùng B-Clean.

## 2. Kiểm trần (oracle)
- **Cách tính:** maximum chính xác của Hmean toàn tập trên A, khi chọn tách hay không tách từng box cha ứng viên (giữ nguyên phép gán con).
  - Dùng Dinkelbach qua các nguồn. Trong mỗi nguồn liệt kê đủ 2^k tổ hợp, chấm bằng maximum matching thật.
  - 1.117 nguồn được tính chính xác. 13 nguồn có trên 12 ứng viên dùng cận an toàn (lỏng hơn nhưng vẫn hợp lệ).
- **Cận trên:** 78,70%, tức **+1,67 pp** so với B1. Cao hơn mốc +0,3 pp, nên **không loại được hướng này**. Theo spec, chạy luật khóa.
- Oracle và luật được tính trong cùng một lần chạy, nhưng quyết định vẫn đi đúng thứ tự của spec.

## 3. Luật khóa trên A (phát triển)

| | TP | FP | FN | LR2 Hmean | tiny recall |
|---|---|---|---|---|---|
| B1 `epsilon_005/0.50` (cùng dump) | 27.628 | 6.680 | 9.795 | 77,032 | 48,78% |
| **MSB1** | 28.269 | 7.490 | 9.154 | **77,257** | **54,08%** |

- **ΔHmean = +0,224 pp, CI95 theo nguồn [−0,40; +0,81]**, chứa 0.
- Tiny +5,3 pp (147 GT tiny được cứu, 10 bị mất).
- 1.592 box cha bị tách thành 3.434 box con, ở 422/1.130 nguồn.
- So với B1: **757 GT được cứu, 116 GT bị mất**; +641 TP, +810 FP.

**Cổng (khóa):**

| Cổng | Kết quả |
|---|---|
| LR2 ≥ B1 + 0,30 pp | **không đạt** (+0,224) |
| tiny ≥ B1 − 3 pp | đạt (+5,3) |
| FP ở cùng recall | đạt **theo định nghĩa**: không setting nào trong 72 setting B1 đạt recall 75,5% của MSB1, nên không có gì để so. Đây không phải bằng chứng FP tốt hơn |

→ **Trượt cổng.**

## 4. Vì sao trượt: cơ chế (mô tả)
- **Hiệu ứng khi tách riêng từng box cha** (bỏ qua tương tác matching, chỉ để minh họa): trong 1.592 box cha, 961 (60%) khi tách **không đổi TP** mà chỉ thêm dự đoán; 111 mất 1 TP; 520 thêm từ 1 TP trở lên. Tính theo lợi ích ròng: 517 box có lợi, 798 box có hại.
- **Một ảnh chi phối CI:** nguồn `6e340fcbb1e0a877` là trang báo dày chữ.
  - Chỉ **10 GT** (tiêu đề), trong khi **B1 đã sinh 1.796 box**, khoảng 27% FP của B1 trên A.
  - MSB1 tách 242 box cha trên vùng chữ thân bài không có nhãn: +292 dự đoán, +2 TP.
  - Xem `examples/toan_canh_nguon_tach_nhieu_nhat_janek.png`.
  - Bỏ riêng ảnh này thì Δ thành +0,51 pp. **Chỉ là mô tả, không dùng để đổi kết luận.** Loại ảnh sau khi xem số là vi phạm protocol.
- **Ca minh họa** (`examples/tach_dung_*.png`, `tach_sai_*.png`, chọn theo hạng SHA, seed 20261123):
  - "Tách đúng": một box B1 phủ 2–3 chữ có GT, box con khớp từng chữ.
  - "Tách sai": box cha nằm trên chữ không có GT, hoặc đã khớp; tách chỉ thêm FP.

## 5. Chẩn đoán phụ: cô lập mask threshold
- Cùng luật, nhưng cha là `base` (mask 0,3, epsilon 0,01), chỉ khác con ở mask.
- A: +0,275 pp so với `base/0.50` (76,802 → 77,077), CI [−0,29; +0,80], tiny +5,0 pp. B (mô tả): +0,71 pp.
- Biên độ tương tự luật chính. Suy luận: tác động chủ yếu đến từ **mask threshold**, chênh lệch epsilon không phải yếu tố chính. Đây là so sánh mô tả, mỗi bên so với baseline của chính nó.

## 6. Split B (khám phá, không dùng chọn)
- MSB1 so với B1: +0,69 pp, CI [+0,28; +1,13], tiny +4,2 pp.
- Dương hơn A, nhưng **không thay kết luận**: cổng chỉ đánh giá trên A.

## 7. Hệ quả
- **Không đề xuất lượt GPU kiểm Clean**, vì cổng LR2 không đạt.
- Kết quả đáng ghi vào báo cáo dưới dạng **âm, có tín hiệu**:
  - tách box gộp cứu được nhiều chữ, kể cả tiny (+5,3 pp);
  - nhưng FP thêm trên vùng chữ không có nhãn làm Hmean gần như không đổi và không ổn định.
- **Điểm đáng báo Tuấn** (bối cảnh đánh giá, không dùng để cứu MSB1): một ảnh có 10 GT nhưng 1.796 dự đoán chiếm khoảng 27% FP của B1 trên A. Kết quả Hmean trên A nhạy với ảnh chữ dày thiếu nhãn. Cần kiểm xem điều này có ảnh hưởng tới các so sánh khác không (chưa kiểm).
- Xác nhận thật sự cần dữ liệu mới: TEST không được dùng. Mọi luật mới sửa theo A rồi thử lại trên B đều không còn là xác nhận.

## 8. Ghi chú sau khi chạy (không đổi luật)
- Tiêu chí chọn ca "tách đúng" trong code là ΔTP > 0 và ΔTP ≥ ΔPred − 1, chặt hơn chữ "tăng TP" của spec mục 9 một chút. Chỉ ảnh hưởng ca minh họa, không ảnh hưởng số liệu.
