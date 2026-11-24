# Flip-TTA: chẩn đoán nguyên nhân và kế hoạch tiếp theo (23/11/2026)

> **Phạm vi:** phân tích hậu kiểm (sau khi đã xem pilot) bằng CPU trên artifact có sẵn. **Không chạy GPU, không quét tham số, không triển khai biến thể nào.** Pilot cũ (`docs/fliptta_spec_20261123.md`, `docs/fliptta_result_20261123.md`) giữ nguyên spec, nhãn `INCONCLUSIVE` và cổng.
> Mẫu 300 nguồn A đã dùng để chẩn đoán, nên **không thể** dùng nó để xác nhận bất kỳ biến thể mới nào. Số T4 → `PENDING_REFERENCE_CHECK`.
> AGENTS.md: hai file tìm thấy (`Chọn monkey hay paddle/CS431/AGENTS.md`, `.../monkey/AGENTS.md`) thuộc repo Paddle/Monkey H200 khác, không áp cho `Monkey v2/cải tiến/mới`. Quy tắc áp dụng: `CLAUDE.md` gốc.

## 0. Kết luận ngắn
- Lợi ích đo được của Flip-TTA **phần lớn tương đương việc nâng ngưỡng score của F** thêm khoảng 0,01–0,02. Trên **Clean, hoàn toàn tương đương** (Δ +0,004 pp ở cùng FP). Trên **LR2 còn một phần nhỏ vượt ngưỡng** (+0,19 đến +0,27 pp, CI chứa 0).
- Phía sau mức net TP −22 / −38 là **churn lớn**: 320 / 336 GT đổi trạng thái. Khoảng 2/3 do box dao động quanh ngưỡng IoU 0,5 và gần như cân hai chiều. Phần lệch có hệ thống là **TP mất vì score tụt dưới 0,60**.
- Ngay cả khi cứu được miễn phí mọi TP mất vì score (cận trên lạc quan), T chỉ tăng thêm tối đa +0,40 (LR2) và +0,31 (Clean) so với T0. Muốn vậy lại cần luật chọn vùng, và luật đó sẽ trả lại một phần FP.
- Chi phí đo được là **1,95×**.
- **Khuyến nghị: dừng hướng Flip-TTA** ở mức "chưa đủ bằng chứng; phần lớn tương đương hiệu chỉnh ngưỡng". Không chạy GPU thêm. Nếu user vẫn muốn tiếp, bước kế tiếp hợp lệ duy nhất là **một lượt chẩn đoán P-map** (mục 6), chưa phải biến thể.

## 1. Đối chiếu trạng thái với file gốc
| Số | Giá trị | Nguồn |
|---|---|---|
| LR2 F → T, Δ | 78,37 → 78,64; +0,27 [−0,16; +0,66] | `results/fliptta_v1_T4/fliptta_result.json` |
| LR2 TP/FP | 6.736/1.181 → 6.714/1.089 (TP −22, FP −92); tiny 32,94 → 31,75 (−1,18) | như trên |
| Clean F → T, Δ | 80,52 → 80,73; +0,21 [−0,22; +0,68]; TP −36, FP −99, tiny −0,95 | như trên |
| Latency | LR2 476 → 903 ms (1,95×), Clean 479 → 907 ms (1,94×) | như trên, `latency_ms` |
| Cờ | c1 trượt; c2, c3 đạt; nhãn `INCONCLUSIVE` | như trên |

Kiểm thêm trong lượt chẩn đoán: counts tính trên pool F ∪ T **bằng** `per_source.csv` ở cả 600 task; **evaluator chính thức** (`CandidateMatcher.assert_official`) khớp ở **1.200/1.200** cặp task × cánh.

**Artifact có:** polygon + raw/packed score của ứng viên `mask_040` (raw ≥ 0,20) cho F và T; kiểm căn chỉnh và IoU nhị phân cho từng ảnh; ms đầu-cuối cho mỗi cánh.
**Artifact KHÔNG có:** P-map (P0, P1b); dự đoán riêng của nhánh lật (decode từ P1b một mình); score của từng box dưới P0 và P1b; thời gian từng công đoạn; phép chạy batch 2.

## 2. Phân tích (CPU, `mixab/fliptta_diag.py`, `scripts/fliptta_matched_control.py` → `results/fliptta_diag_v1/`)
**Quy tắc ghép:**
- GT "mất/được thêm" dựa trên **độ phủ**: có cạnh IoU ≥ 0,5 tới một box được giữ (score ≥ 0,60). Cách này không phụ thuộc maximum matching nào được chọn.
- Box F ↔ T ghép một-một tham lam theo IoU giảm dần, ngưỡng IoU ≥ 0,5.
- Nguyên nhân của một GT mất (với GT được thêm thì đối xứng):
  - `score_crossed_cut`: bên kia có ứng viên IoU ≥ 0,5 nhưng score dưới 0,60;
  - `merged` / `split` / `shifted`: bên kia có box được giữ chồng GT nhưng IoU dưới 0,5 (gộp ≥ 2 GT, tách, hoặc dịch biên);
  - `vanished`: không có ứng viên chồng.

### 2.1 GT: mất và được thêm (không coi ΔTP là "TP bị xóa")
| | LR2 | Clean |
|---|---|---|
| Mất / được thêm / net | 171 / 149 / −22 | 187 / 149 / −38 |
| Mất do score tụt dưới 0,60 | **57** | **46** |
| Được thêm do score vượt 0,60 | 29 | 25 |
| Mất do hình học (gộp / dịch / tách) | 98 (47 / 38 / 13) | 124 (54 / 64 / 6) |
| Được thêm do hình học | 110 (50 / 46 / 14) | 112 (49 / 43 / 20) |
| Biến mất (không ứng viên mạnh) mất / thêm | 16 / 10 | 17 / 12 |
| Sự kiện hình học có IoU bên kia trong [0,4; 0,5) | mất 61/98, thêm 74/110 (trung vị 0,44–0,45) | mất 86/124, thêm 72/112 |
| Chênh score F − T ở ca mất do score | trung vị **0,081**; 45/57 ca > 0,05 | trung vị 0,084; 31/46 ca > 0,05 |
| Mất theo chiều cao GT `<4 / 4–8 / 8–16 / ≥16` | 10 / 59 / 49 / 53 | 10 / 61 / 53 / 63 |
| Độ tập trung: số nguồn có GT mất; top 10 chiếm | 88 nguồn; 33% | 98 nguồn; 27% |

**Hạch toán net TP (LR2):** score −28, hình học +12, biến mất −6, tổng −22.

### 2.2 Box và FP
| | LR2 | Clean |
|---|---|---|
| Cặp box F ↔ T (IoU ≥ 0,5); IoU trung vị | 8.418; 0,926 | 9.417; 0,923 |
| Δ raw (T − F) của cặp: trung vị [p10; p90] | −0,0036 [−0,030; +0,020] | −0,0038 [−0,032; +0,020] |
| Δ raw trung vị: cặp TP so với cặp FP | −0,0033 so với −0,0044 | −0,0032 so với −0,0059 |
| Box giống-FP của F không còn ở T | 288 (score 180 · hình học 83 · không còn đối ứng 25) | 414 (235 · 127 · 52) |
| … theo score F `.60–.65 / .65–.70 / .70–.80 / ≥.80` | **183** / 56 / 45 / 4 | **241** / 98 / 64 / 11 |
| Box giống-FP mới ở T | 141 (score 58 · hình học 79) | 194 (66 · 117) |
| Cặp đổi lớp TP → FP / FP → TP | 60 / 72 | 86 / 68 |

Ghi chú: "giống-FP" là box không ghép GT (`no_gt`, trùng, hoặc bị evaluator loại khỏi ROI/ignore, vẫn tính vào Pred). Proxy FP của Tuấn (`fp_proxy`) chưa áp riêng cho từng box ở đây.

### 2.3 Đối chứng quyết định: T có hơn việc chỉ nâng ngưỡng F không? (`matched_control.json`; mô tả, hậu kiểm, ngưỡng chọn theo độ gần recall/FP)
| | F ở cùng recall với T | F ở cùng FP với T |
|---|---|---|
| LR2 | F cắt .608: TP 6.714, FP **1.147** so với T 1.089 → T − F **+0,27 [−0,15; +0,66]** | F cắt .620: FP 1.085, TP **6.685** so với T 6.714 → **+0,19 [−0,23; +0,58]** |
| Clean | F cắt .616: FP 1.264 so với 1.252 → **+0,06 [−0,35; +0,52]** | F cắt .618: TP 7.122 so với 7.124 → **+0,004 [−0,40; +0,45]** |

Trên mẫu này, F tốt nhất theo lưới mịn là .622 (LR2, 78,50) và .634 (Clean, 80,81). F ở Clean .634 **cao hơn T** (80,73). Không dùng các ngưỡng này làm cấu hình: `.40/.60` do Tuấn chọn trên đủ 2.262 nguồn.

### 2.4 Cùng nguồn Clean và LR2
Đóng góp ròng theo nguồn giữa Clean và LR2 chỉ tương quan yếu (Spearman 0,21). Cùng tăng 45 nguồn, cùng giảm 32, ngược chiều 48. Hiệu ứng theo từng ảnh không ổn định giữa hai điều kiện.

### 2.5 Hình (chọn theo hash, không chọn theo kết quả)
- 12 hình ở `results/fliptta_diag_v1/figures/`: 3 ca mỗi loại {mất do score, mất do hình học, thêm do score, thêm do hình học}. Mỗi loại lấy các ca có `sha256('20261123:fliptta-diag:' + id)` nhỏ nhất (`figures_index.json`).
- Đã xem 4 hình:
  - #1: từ đơn lẻ, score F 0,685 nhưng ứng viên T chỉ 0,502;
  - #4, #10: box hai cánh cùng thấy chữ, IoU bên kia 0,41 / 0,49, tức dao động quanh 0,5;
  - #6 (Clean): T gộp hai từ thành một box.
- LR2 trong hình là bicubic tạo lại để hiển thị (Pillow local không phải 11.3.0).

## 3. Cơ chế: có bằng chứng và còn là giả thuyết
**Có bằng chứng (từ box/score, chưa từ P-map):**
1. Giảm FP chủ yếu do box F điểm thấp (0,60–0,65) tụt dưới ngưỡng, **giống hiệu ứng nâng ngưỡng**. Đối chứng 2.3 xác nhận: Clean không có lợi nào ngoài nâng ngưỡng; LR2 có phần nhỏ, không phân biệt được với 0.
2. Churn lớn và gần cân: khoảng 60–70% sự kiện hình học có IoU bên kia sát 0,5. Đây là **dao động định vị quanh ngưỡng ghép**, không phải mất chữ.
3. Phần lệch có hệ thống làm mất TP: **ca mất do score** (57 so với 29 được thêm ở LR2) có chênh score lớn (trung vị 0,08, gấp khoảng 20 lần dịch score điển hình). Phù hợp với "nhánh lật yếu hẳn ở vài chữ".
4. Hiệu ứng không tập trung (top 10 nguồn chỉ khoảng 30% sự kiện). Không do một ảnh chi phối.

**Còn là giả thuyết (thiếu P-map / nhánh lật riêng):**
- Ca mất do score là do P1b thấp, tức model kém trên chữ soi gương, chứ không do tương tác khi gộp. Cần score từng box dưới P0 và P1b riêng.
- Bất đồng nằm ở vùng biên ảnh, gần padding, hay do Resize. Kiểm căn chỉnh cứng đã loại **lỗi căn chỉnh** (lật hai lần, pad hằng, IoU căn 0,85 so với 0,03), nhưng chưa đo bất đồng theo vị trí.
- Gộp/tách ở T là do P-map trung bình nối liền khoảng trống giữa từ.

## 4. Các hướng chất lượng (đánh giá, không triển khai)
| Hướng | Bằng chứng ủng hộ | Bằng chứng chống / rủi ro | Đánh giá |
|---|---|---|---|
| **A. Gộp thiên về ảnh gốc** (`P = w·P0 + (1−w)·P1b`, w > 0,5) | Ca mất do score (mục 3.3) | Lợi ích hiện tại (giảm FP) cũng đến từ việc kéo score xuống. Tăng w kéo cả hai về baseline. Trên Clean, T đã nằm đúng trên đường PR của F, nên A gần như chắc chắn ≤ nâng ngưỡng. Cần chọn w, tức quét | **Không chọn** |
| **B. Đồng thuận chỉ ở vùng bất định** (giữ P0 nơi P0 ≥ τ, trung bình ở chỗ khác) | Nhắm đúng ca mất do score có score F cao (26/57 ca LR2 có score F ≥ 0,65) | FP bị loại cũng có score F 0,60–0,80 (105/288 ≥ 0,65); bảo vệ vùng P0 cao sẽ trả lại một phần. τ chưa có căn cứ trước dữ liệu (mọi τ chọn bây giờ là hậu kiểm). Cận trên lạc quan chỉ +0,40 / +0,31 so với T0 | **Ứng viên duy nhất nếu tiếp tục**, nhưng **chưa đủ cơ sở chọn τ**. Cần P-map trước |
| **C. Giảm bất đồng ở biên** | Không có: căn chỉnh đã đạt; chưa đo bất đồng theo vị trí | Thêm thao tác hình học cho vấn đề chưa chứng minh | **Không chọn** |

**Đối chiếu hướng cũ:**
- B là cùng họ "lọc/chấm lại bằng tín hiệu thứ hai" với DualAgree v1 / XV-B1 (hai checkpoint) và MSB1 / ngưỡng theo kích thước (hậu xử lý một model). Khác ở chỗ nguồn tín hiệu là ảnh lật của **cùng** checkpoint.
- Tất cả là **TTA / hậu xử lý thông thường**, không phải đóng góp phương pháp độc lập.
- Tài liệu: Shanmugam, Blalock, Balakrishnan, Guttag, *Better Aggregation in Test-Time Augmentation*, ICCV 2021 (arXiv:2011.11156; đã đọc abstract). Bài báo nêu cách gộp truyền thống là trung bình đơn giản, và TTA có thể biến nhiều dự đoán đúng thành sai dù độ chính xác tổng tăng. Họ đề xuất học trọng số gộp từ dữ liệu có nhãn. Đó là **phân loại ảnh**, không phải detection; chỉ trích ở mức abstract.

## 5. Latency: những gì đo được và cần đo
- **Đã đo:** đầu-cuối mỗi cánh (median 476 / 903 ms LR2). **Chưa có** p95, chưa tách công đoạn, chưa đo throughput.
- **Nếu chạy chẩn đoán**, đo cho cả F và T, cùng T4, FP32, batch 1, cùng ảnh, 10 ảnh khởi động, `torch.cuda.synchronize()` ở mỗi mốc:
  - các công đoạn: tiền xử lý (đọc + Resize + `data_preprocessor`), H2D, forward gốc, forward lật, lật về + trung bình, hậu xử lý;
  - báo median, p95, bộ nhớ đỉnh.
- **Tối ưu giữ nguyên phép tính:** batch 2 (ảnh gốc + ảnh lật trong một forward, cùng shape nên không pad thêm). Chỉ được coi là tương đương nếu box và score trùng với chạy tuần tự. Nếu sai khác số học làm đổi box thì đó là thay đổi thuật toán. Không hứa giảm một nửa: forward chỉ là một phần của 903 ms (chưa đo phần đó).
- **Không đề xuất** router "chạy nhánh lật có chọn lọc" hay nhánh lật ở độ phân giải thấp cho phép thử đầu: chưa có biến thể chất lượng nào đáng tối ưu.

## 6. Nếu user vẫn muốn tiếp: phép chạy chẩn đoán tối thiểu (chưa chuẩn bị notebook)
- **Mục đích:** kiểm giả thuyết 3.1 trước khi viết bất kỳ spec biến thể nào.
- **Cách làm:** cùng 300 nguồn, Clean + LR2, B1, FP32. Với mỗi box của F và T, lưu raw score tính trên **P0, P1b và trung bình** riêng (dùng cùng `_get_bbox_score`). Lưu thêm decode của P1b một mình, thời gian từng công đoạn, phép batch 2, và P-map float16 cắt quanh khoảng 40 ca chọn theo hash (từ `events_*.json.gz`).
- **Chi phí ước lượng:** khoảng 20–30 phút T4 (pilot cũ 1,46 s/ảnh; thêm decode thứ ba và lưu crop), dataset tái dùng `cs431-fliptta-pilot`.
- **Câu hỏi phân biệt đúng/sai:**
  - (a) Ở ca mất do score, P1b thấp hơn P0 rõ rệt (ví dụ trung vị > 0,05)? Nếu không, giả thuyết "nhánh lật kéo xuống" sai.
  - (b) Ở FP bị loại, P1b cũng thấp? Nếu FP bị loại và TP bị mất có phân bố (P0, P1b) **chồng nhau**, thì không có τ nào của B tách được chúng → **dừng B**.
- Chỉ khi (a) đúng **và** (b) tách được thì mới viết spec T1 = B với τ cố định. τ sẽ suy ra từ phân bố của chính mẫu này, nên phải ghi rõ là hậu kiểm. Đánh giá T1 cần **nguồn chưa dùng trong bước này**: 830 nguồn A ngoài mẫu 300 (vẫn là development, không phải TEST sạch), so F / T0 / T1, cả Clean và LR2, CI ghép cặp theo nguồn, cổng nhóm giữ nguyên.

## 7. Điều kiện dừng và giới hạn
- **Dừng ngay (khuyến nghị):**
  - lợi ích của T0 trên Clean tương đương nâng ngưỡng, trên LR2 chỉ phần nhỏ;
  - cận trên lạc quan của mọi sửa đổi kiểu B (+0,40 / +0,31 so với T0) đã phải giả định cứu miễn phí;
  - chi phí 1,95×.
- **Giới hạn:**
  - mọi số trên đều là development (A đã dùng nhiều lần), và mẫu 300 nguồn đã dùng để chẩn đoán;
  - không còn tập chưa dùng để xác nhận độc lập, ngoài benchmark cuối do Tuấn chủ trì;
  - đối chứng "nâng ngưỡng" là hậu kiểm, chỉ mô tả.
- **Trước khi cân nhắc benchmark cuối,** cần một biến thể có Δ ≥ +0,30 pp với CI > 0 trên 830 nguồn A chưa dùng, **so với cả F và F đã nâng ngưỡng**, Clean ≥ −0,30, tiny ≥ −3. Hiện không có ứng viên nào như vậy.
