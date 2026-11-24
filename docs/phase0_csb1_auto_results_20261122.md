# Phase 0 · CS-B1@0.7 · kết quả phần CPU tự động (22/11/2026)

> **Trạng thái:** đã chạy trên CPU local từ dump T4 có sẵn (`dump_identity` `beefe28e…`). Mọi số đều là **PENDING_REFERENCE_CHECK** (dump T4, chưa có A100). Phát triển chỉ dựa trên **A**. B chỉ để mô tả, không dùng chọn gì.
> **Chưa có:** latency (chờ notebook Kaggle), chấm tay (chờ 10 thẻ hiệu chuẩn). Vì vậy **chưa có kết luận A/B/C/D cuối cùng**.
> Kế hoạch gốc: `docs/phase0_csb1_plan_20261122.md`. Cấu hình `CS-B1@0.7`, cổng và bảng quyết định **không đổi**.

**Code và lệnh:**
- `mixab/phase0.py`: `refcheck`, `stats`, `cards`; test `tests/test_phase0.py`. Toàn bộ test 96/96 PASS.
- Dùng lại nguyên của Tuấn:
  - `CandidateMatcher` và `maximum_matching`;
  - `rev4.decision_review.fp_proxy` cho proxy FP;
  - thứ tự phân loại dựa trên box của `rev4.b1_diagnostics.reason` cho GT bỏ sót.
- **Khác với chẩn đoán gốc của Tuấn:** dump không có P-map, nên không tính được hai bước `invalid_kernel` và `weak_response`. Các GT này rơi vào nhóm `no_box_in_dump`, và **không được gọi là "phản hồi yếu"**.

Đầu ra:
- `results/phase0/reference_check.json`
- `results/phase0/auto_stats.json`
- `results/phase0/review/` (thẻ và phiếu)
- Bản ghi từng đối tượng ở `runtime/phase0_cache/phase0_records.jsonl.gz` (không đưa vào Git).

---

## 1. Kiểm tra tham chiếu T4 so với A100 (mục 3 kế hoạch)

**Nguồn:** `results/phase0/reference_check.json`.

| | B-Clean | B1 |
|---|---|---|
| Ô lệch / tổng | 5.953 / 162.864 (3,66%) | 6.896 / 162.864 (4,23%) |
| Ô lệch ở A / B | 2.849 / 3.104 | 3.207 / 3.689 |
| Trường `gt` (mọi bin) có lệch không | **không bao giờ** | **không bao giờ** |
| Ô lệch trường `tp` / `pred` | 1.975 / 5.153 | 2.167 / 5.967 |
| Độ lớn lệch `tp` | ±1: 1.672 ô · ±2: 266 · ±3: 37 | ±1: 1.899 · ±2: 249 · ±3: 19 |
| Setting lệch nhiều nhất | `width_3/0.20` (304 ô) | `width_3/0.20` (326 ô) |

**Tổng tại các setting CS-B1 dùng** (T4 so với A100):

| Setting · split | Hmean T4 | Hmean A100 | Δ (pp) | ΔTP | ΔPred | Số nguồn lệch |
|---|---|---|---|---|---|---|
| B1 `epsilon_005/0.50` · A | 77,0322 | 77,0320 | +0,0002 | −3 | −8 | 22 |
| B1 `epsilon_005/0.50` · B | 76,6866 | 76,6938 | −0,0072 | −5 | −6 | 19 |
| B-Clean `mask_040/0.70` · A | 73,7690 | 73,7779 | −0,0089 | −5 | −6 | 16 |
| B-Clean `mask_040/0.70` · B | 71,4064 | 71,4058 | +0,0006 | +2 | +5 | 19 |

**Độ nhạy của ΔCS-B1 khi loại nguồn lệch.** Không tính được CS-B1 trên A100 vì A100 không lưu box. ΔCS-B1 so với B1, kèm CI95 bootstrap theo nguồn:

| Tập nguồn | A | B (mô tả) |
|---|---|---|
| Tất cả | +0,177 [0,065; 0,304] (1.130 nguồn) | +0,283 [0,168; 0,408] |
| Bỏ 35 nguồn lệch ở 2 setting đang dùng (**cách của kế hoạch**) | **+0,127 [0,052; 0,204]**, vẫn dương, nằm trong CI gốc | +0,237 [0,125; 0,366] |
| Bỏ 240 nguồn lệch ở **bất kỳ** setting nào của 2 geometry (chặt hơn, mô tả) | +0,079 [−0,004; 0,168] | +0,161 [0,056; 0,269] |

**Giả thuyết "lệch do box có score sát ngưỡng"** (mô tả):
- Ở nguồn lệch, box có score gần ngưỡng nhất cách ngưỡng median 0,0009 (B1) và 0,0009 (B-Clean). Ở nguồn không lệch là 0,028 và 0,020.
- Khoảng cách < 0,005: 78% nguồn lệch so với 20% nguồn không lệch (B1); 77% so với 23% (B-Clean).
- Phù hợp với giả thuyết, nhưng đây chỉ là tương quan.

**Đọc:**
- Lệch T4/A100 là ±1–3 count, không bao giờ ở GT, và chủ yếu ở nguồn có box sát ngưỡng.
- Điều kiện A(1) của bảng quyết định **đạt**: sau khi loại nguồn lệch, Δ vẫn dương và nằm trong CI gốc.
- Với cách loại chặt hơn, CI chạm 0. Cách này vừa bỏ 21% nguồn, vừa có thể thiên lệch vì nguồn nhiều box dễ lệch hơn.

---

## 2. Thống kê tự động Q1–Q5 (mục 4 kế hoạch)

**Nguồn:** `results/phase0/auto_stats.json`. **Tổng trên A:**

| | TP | FP | FN | Hmean LR2 | tiny recall |
|---|---|---|---|---|---|
| B1 (`epsilon_005/0.50`) | 27.628 | 6.680 | 9.795 | 77,032 | 48,78% |
| CS-B1@0.7 | 27.876 | 6.910 | 9.547 | 77,209 (+0,177) | 49,48% |
| DualAgree v1 (khóa) | 26.369 | 4.070 | 11.054 | 77,714 | 28,96% |

### Q1 · FP thuộc loại nào (proxy của Tuấn, A)
- **B1:** 6.680 FP.
  - không chồng GT hợp lệ (nền hoặc thiếu nhãn): 3.898 (58%);
  - chồng GT, sai định vị hoặc chỉ phủ một phần: 1.944 (29%);
  - nghi gộp: 837 (13%);
  - trùng: 1.
- **CS-B1:** 6.910 FP, trong đó 220 FP đến từ box B-Clean được thêm. Proxy của 220 FP này: chồng GT / định vị **174 (79%)** · không chồng GT 23 · nghi gộp 20 · trùng 3.
- **Đọc:** FP do box bổ sung **chủ yếu là box nằm trên chữ nhưng lệch hoặc chỉ phủ một phần**, ít khi là nền. Đây vẫn chỉ là proxy, cần chấm tay để xác nhận.

### Q2 · GT tiny (< 4 px) bị CS-B1 bỏ sót (A: 1.305 / 2.583 GT tiny)

| Chẩn đoán (thứ tự của Tuấn, chỉ phần box) | Số GT | Nhóm kế hoạch |
|---|---:|---|
| `no_box_in_dump` (không có box nào trong dump chạm GT; có thể do P-map yếu **hoặc** box bị loại vì min width trước khi lưu, không phân biệt được) | 684 | (a) |
| `only_box_removed_by_roi_ignore` | 41 | (a) |
| `geometry` (có box trong đầu ra chạm GT nhưng IoU < 0,5) | 395 | (c) |
| `ambiguous_score_and_geometry` (chỉ có box dưới ngưỡng chạm GT, IoU < 0,5) | 77 | (c) |
| `score_filtering` (có box IoU ≥ 0,5 nhưng dưới ngưỡng: 54 của B1, 54 của B-Clean) | 108 | (b) |
| `ambiguous_matching` | 0 | (d) |
| (e) có box B-Clean ≥ 0,7, IoU ≥ 0,5 với GT nhưng bị khử trùng | 3 | (e) |

Theo nhóm: (a) 725 = 55,6% · (b)+(c) 580 = **44,4%**. Nếu chỉ tính box của B1 ở nhóm (b): (54 + 472)/1.305 = 40,3%.

### Q3 · Box B-Clean bổ sung (A)
- **537 box được thêm.**
  - 59 box bị ROI/ignore loại, không tính là dự đoán.
  - **258 TP** · **220 FP**.
  - Trên toàn tập: **+248 TP, +230 FP**. Không GT nào bị mất so với B1.
- **GT được cứu theo chiều cao:** < 4 px: 18 · 4–8: 82 · 8–16: 79 · ≥ 16: 69. Lợi ích **chủ yếu ở chữ 4–16 px**, rất ít ở chữ tiny.
- **Độ tập trung**, phân rã cộng chính xác ΔHmean = Σᵢ(2ΔTPᵢ − H_B1·ΔPredᵢ)/(GT + Pred_CS):
  - 120 nguồn có đóng góp dương, 100 nguồn có đóng góp âm;
  - **10 nguồn đóng góp nhiều nhất chiếm 66,6% lợi ích ròng**;
  - bỏ 10 nguồn đó thì Δ còn +0,066 pp.
  - B (mô tả): 54,5%; bỏ top 10 còn +0,143 pp.

### Q4 · Vì sao DualAgree v1 mất tiny còn CS-B1 giữ được (A)
- 527 GT tiny là TP của B1 nhưng là FN của DualAgree v1.
- **Cả 527 đều cùng một lý do:** box B1 phủ GT có score trong [0,5; 0,7), đứng một mình nên bị cắt ở t_l = 0,7.
- CS-B1 giữ đúng cả 527 GT này. Cơ chế đã được xác nhận định lượng.

### Q5 · Phần lợi ích B1 có tự lấy lại được không (A, 248 GT được cứu)
- **Theo quy tắc khóa** (box B1 `epsilon_005` chạm GT với IoU ≥ 0,3): 110/248 = **44,4%**, dưới mốc 50%.
  - Trong đó: IoU từ 0,3 đến dưới 0,5: 109 GT; IoU ≥ 0,5 nhưng score < 0,5: 1 GT.
  - 138 GT không có box B1 `epsilon_005` nào chạm IoU ≥ 0,3.
- **Mô tả thêm** (các geometry khác của B1 trong dump):
  - 171/248 (69%) GT được cứu có box B1 khớp (IoU ≥ 0,5, score ≥ 0,5) ở **một** geometry khác, gần như toàn bộ là **`mask_040`** (167). B-Clean cũng dùng `mask_040`.
  - Nhưng B1 `mask_040` ở mọi ngưỡng score đều **thấp hơn** B1 `epsilon_005/0.50` trên A: tốt nhất là `mask_040/0.60` với 76,79 so với 77,03 (counts A2r A100).
  - **Đọc (suy luận):** phần lớn GT được cứu là chữ mà mask cao hơn tách được box. Dùng mask cao hơn cho toàn ảnh lại làm mất chữ ở chỗ khác. Kiểm cơ chế này cần thêm phân tích, chưa làm.

### Độ nhạy quanh cấu hình tham chiếu (mô tả, lấy từ `sweep_totals.npz`, không chọn lại)

| | θ = 0,3 | θ = 0,5 | θ = 0,7 |
|---|---|---|---|
| t_c = 0,6 · A / B | +0,147 / +0,361 | +0,084 / +0,298 | **−1,109 / −0,913** |
| t_c = 0,7 · A / B | +0,146 / +0,253 | **+0,177 / +0,283** (tham chiếu) | **−0,325 / −0,242** |

Ở θ = 0,7, các cặp box có IoU từ 0,5 đến dưới 0,7 không còn được khử trùng nên box trùng bị cộng thành FP (suy luận từ luật, chưa đếm riêng).

### Đối chiếu với baseline .40/.60 của Tuấn (mô tả)
B1 `mask_040/0.60` (A100): A 76,79 · B 77,01. CS-B1: A 77,21 · B 76,97. **Trên B, B1 ở .40/.60 nhỉnh hơn CS-B1 0,04 pp.** Chỉ để người đọc đối chiếu, không dùng để chọn.

---

## 3. Đối chiếu bảng quyết định (chỉ phần tự động, ngưỡng giữ nguyên)

| Kết luận | Điều kiện | Trạng thái |
|---|---|---|
| **A · Tiếp tục** | (1) Δ sau khi loại nguồn lệch > 0 và trong CI | **đạt** (+0,127) |
| | (2) TP cứu > FP thêm **và** top 10 < 50% | TP > FP đạt (248 > 230); **top 10 = 66,6%: không đạt** |
| | (3) Dấu Δ dương ở t_c 0,6 và θ 0,3/0,7 | **không đạt** (θ 0,7 âm) |
| | (4) Q5 < 50% | đạt (44,4%) |
| **B · Hậu xử lý riêng B1** | Q5 ≥ 50% **hoặc** Q2 (b)+(c) ≥ 30% | **điều kiện Q2 đạt** (44,4%) |
| **C · Cần thêm dữ liệu** | Bước 1 không kết luận được, hoặc (a) > 60%, hoặc `khong_chac` > 30% | (a) 55,6%: không; Bước 1 kết luận được; `khong_chac`: **chờ chấm** |
| **D · Dừng bổ sung box** | Δ đổi dấu ở bước 3 **hoặc ở độ nhạy**, hoặc FP ≥ TP, hoặc ≥ 50% lợi ích từ ≤ 10 nguồn | **đạt ở 2 điều kiện:** độ nhạy θ 0,7 đổi dấu; top 10 chiếm 66,6% |

**Ghi chú cho nhánh B (sự thật từ dữ liệu sẵn có):**
- Phép thử của nhánh B theo kế hoạch là chọn lại trong 72 setting của B1 trên A, với cùng luật.
- Trên A, setting tốt nhất của B1 trong 72 setting **chính là** `epsilon_005/0.50` (77,032; tiếp theo là `width_7/0.50` 76,825; nguồn: counts A2r).
- Vì vậy phép thử đó **không thể** cho +0,3 pp so với B1 trên A.

**Kết luận cuối chưa đưa ra:**
- Theo luật khóa, phần tự động đã **loại A** và **kích hoạt D** (2 điều kiện).
- Nhánh B cũng được kích hoạt, nhưng phép thử của nó đã có đáp án âm từ dữ liệu A2r.
- Bảng không quy định thứ tự ưu tiên khi B và D cùng đạt. **User quyết định** sau khi có kết quả chấm thẻ.
- Không có ngưỡng nào bị đổi.
