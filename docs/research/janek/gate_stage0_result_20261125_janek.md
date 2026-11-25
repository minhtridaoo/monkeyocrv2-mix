# Giai đoạn 0 · Cái gì đang giới hạn kết quả trên A? (CPU, artifact có sẵn) · 25/11/2026

- **Theo:** `docs/research/janek/PLAN_DAT_CONG_NHOM_20261125_janek.md` §3 và §8.
- **Script:** `scripts/gate_stage0_janek.py`.
- **Bản chính:** `docs/handoff/artifacts/janek/gate_stage0_20261125/` (`stage0_janek.json`, `review/`).

**Phạm vi đã giữ:**
- Chỉ split A. Không fit, không chọn lại, không mở B/TEST.
- Box-quality v1 và kết quả `SMALL_USEFUL` giữ nguyên.

**Nguồn dữ liệu:**
- B1 và FULL: bản ghi dev A (Kaggle T4).
- B-Clean `.40/.60`: dump T4 của cùng ảnh.
- Mọi count tính bằng evaluator chính thức hoặc cạnh IoU ≥ 0,5 của `CandidateMatcher`, sau lọc ROI/ignore.
- Số của ba cánh trùng `devA_result` (Clean F 79,126 / FULL 79,387 / B-Clean 80,520).

## 1. Khoảng thiếu (tính lại từ JSON)
- Clean cần ≥ **80,220**; FULL đang 79,387 → thiếu **0,83 pp**.
- mean cần ≥ **79,002**; FULL đang 78,178 → thiếu **0,82 pp**.
- tiny LR2 đã đạt.
- Giữ nguyên TP, muốn đạt cổng Clean thì phải bớt ≥ **777 FP Clean** mà không mất TP nào.

## 2. GT bị bỏ sót: phân loại theo pool B1 (`mask_040`, raw ≥ 0,20)

| Loại (FULL bỏ sót) | Clean | LR2 |
|---|---|---|
| Có ứng viên IoU ≥ 0,5 nhưng bị score lọc | 464 (6%) | 568 (6%) |
| Có ứng viên được giữ nhưng đã ghép với GT khác | 11 | 6 |
| Sai hình học: một box phủ ≥ 50% GT (gộp nhiều chữ) | 2.306 (30%) | 2.252 (23%) |
| Sai hình học: chồng một phần, IoU 0,1–0,5 | 1.677 (22%) | 1.955 (20%) |
| Không có ứng viên (IoU < 0,1, phủ < 50%) | 3.291 (42%) | 5.097 (52%) |
| **Tổng** | 7.749 | 9.878 |

- F (B1 `.40/.60`) có phân bố gần như giống hệt; chỉ khác 411 → 464 ở nhóm bị score lọc.
- **Scorer chỉ chạm được khoảng 6% số GT bỏ sót.** 94% còn lại cần box khác: tách hoặc gộp đúng, hoặc tạo được box mới.

## 3. B-Clean so với FULL trên Clean
| | TP | FP | FN |
|---|---|---|---|
| FULL | 29.674 | 7.661 | 7.749 |
| B-Clean `.40/.60` | 29.933 | 6.993 | 7.490 |

**Về GT:**
- B-Clean tìm được 864 GT mà FULL bỏ sót. Phân theo loại của FULL: bị score lọc 227, gộp 274, chồng một phần 304, không ứng viên 58.
- Ngược lại, FULL tìm được 605 GT mà B-Clean bỏ sót. TP ròng chỉ +259.

**Về FP:**
- FP "không chồng GT": B-Clean 4.357, FULL 5.173 (−816).
- FP hình học: B-Clean 2.636, FULL 2.488 (+148).

**Tách khoảng cách 1,13 pp** (đếm chính xác, cùng GT):

| Kịch bản | Clean Hmean |
|---|---|
| FULL lấy thêm phần TP ròng của B-Clean, FP giữ nguyên | 79,80 (+0,42) |
| FULL bớt FP đúng bằng mức của B-Clean, TP giữ nguyên | 80,10 (+0,72) |
| FULL bớt toàn bộ 754 FP "không chồng GT" mà B-Clean không có box nào giao | 80,20 |

Cả ba kịch bản đều **dưới 80,22**. Kịch bản cuối là cận lạc quan cho việc "dạy B1 im lặng ở chỗ B-Clean im lặng".

## 4. FP "không chồng GT" có phải chữ thiếu nhãn không? (xem bằng mắt)
- **Cách xem:** 60 crop chọn theo hash (seed `20261125:stage0:review:`), nhãn của Claude. Chỉ một người xem; không mù với nhóm và score; box cam che một phần chữ.
- File: `review/sheet_1..5.png`, `review/review_sheet.json`.

| Nhóm (Clean) | Cỡ nhóm | Mẫu | Chữ thật thiếu nhãn | Chữ mờ / khó đọc | Nền | Không chắc |
|---|---|---|---|---|---|---|
| FULL đã bớt so với F | 261 | 24 | 13 | 2 | 5 | 4 |
| FULL vẫn giữ (mọi FP không chồng GT) | 5.173 | 12 | 9 | 3 | 0 | 0 |
| FULL giữ **và B-Clean không có box** (vùng teacher sẽ dạy B1 nén xuống) | 754 | 24 | 17 | 3 | 3 | 1 |

Tỉ lệ "là chữ" (thiếu nhãn hoặc mờ), CI Wilson 95%:
- nhóm FULL giữ: 12/12 [0,76; 1,00];
- nhóm teacher-background: 20/24 [0,64; 0,93];
- nhóm FULL đã bớt: 15/24.

Các nguồn hay gặp: trang sách hoặc báo không gắn nhãn thân bài, nhãn hàng chữ nhỏ, chữ mờ. Nền gồm mã vạch, icon, logo, thân xe.

**Hệ quả:**
- Phần lớn FP còn lại của B1/FULL là **chữ thật không có nhãn**, cả ở chỗ B-Clean im lặng.
- Lợi thế FP của B-Clean trên Clean, chiếm khoảng 63% khoảng cách, phần lớn là **ít bắn vào chữ không nhãn hơn**. Đó không phải lợi thế phát hiện chữ.
- Một phần lợi của FULL (13/24 trong mẫu đã bớt) cũng thuộc loại này.

## 5. Giới hạn lạc quan của pool (oracle dùng GT; **không phải phương pháp hay mức tăng khả thi**)
| Oracle | Clean | LR2 | mean |
|---|---|---|---|
| Chọn đúng một box cho mỗi GT có trong pool (precision = 1) | 89,21 | 85,79 | 87,50 |
| FULL bỏ mọi FP, TP giữ nguyên | 88,45 | 84,80 | 86,62 |
| FULL + cứu lạc quan mọi GT có ứng viên bị lọc, FP giữ nguyên | **80,13** | 77,94 | 79,03 |

**Đọc:**
- Hai oracle đầu cao chỉ vì dùng GT để xóa FP, mà FP phần lớn là chữ không nhãn. Chúng không nói gì về khả năng học.
- Oracle cứu ứng viên (phần duy nhất scorer làm được mà không đụng FP) vẫn **dưới cổng Clean** dù lạc quan.

## 6. Trả lời: head, score hay candidate đang giới hạn?
- **Không phải score.** Scorer chỉ chạm 6% số GT bỏ sót; cứu toàn bộ vẫn < 80,22. Muốn qua cổng bằng scorer thì phải bớt ≥ 777 FP Clean mà không mất TP, trong khi FP còn lại chủ yếu là chữ thiếu nhãn. **Không nên đầu tư thêm vào scorer** để đạt cổng.
- **Giới hạn thật là candidate/geometry:** 51% số GT Clean bỏ sót là box gộp hoặc lệch, 42% không có ứng viên. Đây là việc của head/P-map và hậu xử lý hình học. MSB1 (tách box gộp bằng mask .40) đã thử và ra âm; zoom cũng âm.
- **Teacher audit không ủng hộ Clean-output anchoring từ B-Clean** theo điều kiện của kế hoạch §3–4:
  - phần lợi hợp lệ phía TP của B-Clean chỉ tương đương +0,42 pp nếu chuyển trọn;
  - phần lớn còn lại là B-Clean im lặng ở chữ không nhãn (khoảng 83% trong mẫu teacher-background);
  - anchoring sẽ dạy B1 nén chữ thật thiếu nhãn, đúng trường hợp kế hoạch dặn "không thiết kế loss để khai thác lỗi nhãn";
  - ngay cả nén hết 754 box đó cũng chỉ lên 80,20.
- **Đối chiếu hướng cũ:**
  - PRC (MSE xác suất, teacher B0 Clean → student LR2) và feature-KD (feature Clean → LR2) đều âm, nhưng chúng nhắm LR2.
  - Anchoring Clean→Clean là hướng khác. Bằng chứng ở đây cho thấy tín hiệu của teacher trên Clean chủ yếu là "ít bắn vào chữ không nhãn". Đây là lý do khác với PRC/KD, nhưng cũng đủ để không mở arm P.

## 7. Quyết định theo kế hoạch §3
"Nếu phần lớn cải thiện kỳ vọng chỉ là xóa chữ thiếu nhãn, không thiết kế loss để khai thác lỗi nhãn. Báo hạn chế đo lường và xem lại mục tiêu với nhóm."
→ **Dừng ở bằng chứng.** Không viết spec C/P, không chuẩn bị notebook train.

**Điều nên báo Tuấn:**
- Cổng g2 (Clean ≥ B-Clean − 0,3) so với một mốc mà phần lợi đến nhiều từ việc ít bắn vào chữ không nhãn.
- Cùng hiện tượng với ảnh trang báo (`6e340fcbb1e0a877`), nhưng ở quy mô toàn A.
- Cần nhóm quyết định cách đối xử với chữ không nhãn trong đánh giá (ví dụ: một phép đo phụ trên ảnh có nhãn đầy đủ). Không tự sửa GT.

**Nếu nhóm vẫn muốn một hướng train:** bằng chứng chỉ ra **hình học (gộp/tách chữ)**, không chỉ ra anchoring.
- Chưa có giả thuyết khóa cho hướng này. Cần spec riêng và đối chiếu với MSB1 âm.
- Mình không tự mở module mới.

**Giới hạn của Giai đoạn 0:**
- A là development đã xem. Nhãn crop chỉ do một người (Claude) xem, không mù.
- Phân loại hình học dùng ngưỡng mô tả (IoU 0,1, phủ 50%).
- Audit teacher làm trên A, chưa làm trên TRAIN.
- T4 → `PENDING_REFERENCE_CHECK`.
