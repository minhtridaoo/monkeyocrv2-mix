# E-sel: chọn có ràng buộc tiny trên A (KHÁM PHÁ, không thay kết luận v1)

**Nhãn:** B đã được xem ở v1, nên mọi số trên B ở đây **chỉ là khám phá**. Kết luận Mix A-core v1 giữ nguyên: α\* = B-Clean, Mix A chưa thắng theo quy tắc đã khóa.

Luật được ghi ở kế hoạch §4.4 trước khi chạy. Code: `mixab/explore_selection_janek.py`, có test trong `tests/test_explore_selection_janek.py`. Đầu vào là counts A2r trên A100 (`docs/handoff/artifacts/janek/mix_a_v1/reselect_spec_local_janek.json`). Đầu ra: `explore_selection_janek.json`.

## Kết quả chọn trên A

- `t_ref`: tiny LR2 trên A của B1 tại setting R1 (`epsilon_005/score_0.50`) = **48,82%**.
- Mốc khả thi = t_ref − 3 pp = **45,82%**.

| Model | Số setting khả thi / 72 | Setting chọn | Mean Hmean trên A | Tiny trên A |
|---|---|---|---|---|
| B-Clean | **0** | bị loại | — | — |
| α 1/4 | **0** | bị loại | — | — |
| α 1/2 | 7 | `mask_040/score_0.60` | 78,145 | 46,88% |
| **α 3/4** | 47 | `epsilon_005/score_0.50` | **78,300** | 46,57% |
| B1 | 49 | `epsilon_005/score_0.50` | 78,166 | 48,82% |

α\* (khám phá) = **α 3/4**, ở đúng setting mà nó cũng chọn theo R1.

## Trên B (khám phá)

So với hai baseline: B-Clean ở setting R1 (77,87) và B1 ở setting có ràng buộc (78,11).

- α 3/4: mean **78,34**, hơn B-Clean +0,46 pp và hơn B1 +0,23 pp → **không đạt +0,5 pp**. CI paired mean so với B-Clean [−0,53; 1,48], so với B1 [−0,12; 0,71]; cả hai chứa 0.
- Tiny trên B: 34,56% so với B1 37,63% (−3,07 pp) → **không đạt** mốc B1 − 3 pp.
- **Không có ứng viên nào đạt cả hai mục tiêu**, kể cả khi vòng chọn đã xét tiny.

Các số còn lại trên B chỉ mô tả và **không được dùng để chọn**. Ví dụ α 1/2 ở `mask_040/score_0.60` có mean trên B 78,74, nhưng nó không được chọn trên A, và tiny trên B 33,01% cũng dưới mốc. Chọn nó vì đẹp trên B là cherry-pick.

## FP ở recall tương đương (mô tả)

Ngưỡng: recall LR2 trên A ≥ recall của B1 tại setting R1 trên A (73,83%). Với mỗi model, lấy setting có ít FP LR2 nhất trên A trong các setting đạt ngưỡng.

| Model | Setting | Recall LR2 A / B | FP LR2 A / B | Tiny B |
|---|---|---|---|---|
| B1 | `epsilon_005/score_0.50` | 73,83% / 70,52% | 6.685 / 5.456 | 37,63% |
| α 3/4 | `epsilon_005/score_0.45` | 73,99% / 70,61% | 7.163 / 5.781 | 36,47% |
| B-Clean, α 1/4, α 1/2 | không setting nào đạt recall của B1 | — | — | — |

Ở cùng mức recall, α 3/4 có **nhiều FP hơn** B1: +7,2% trên A, +6,0% trên B. Như vậy "giảm 11% FP" ở bảng v1 đến từ việc recall thấp hơn, không phải từ model tốt hơn. Đúng như Tuấn lưu ý.

## Kết luận khám phá

- **Đổi cách chọn không tạo ra ứng viên đạt mục tiêu.** Model trộn tốt nhất (α 3/4) vẫn thiếu cả mức tăng tổng thể (+0,23 pp so với B1) lẫn tiny (−3,07 pp), và không có lợi thế FP ở cùng recall.
- **Điểm sáng duy nhất:** Clean của α 3/4 tăng +0,47 pp so với B1, CI [0,20; 0,88] không chứa 0, mà LR2 giữ nguyên. Đây là quan sát trên dữ liệu đã xem.
- Vì vậy, nếu pilot BN được làm thì phải coi là **thử nghiệm có kỳ vọng thấp**. Xem `docs/research/janek/design_bn_recalibration_pilot_janek.md`.
