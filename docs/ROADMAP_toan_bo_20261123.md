# Roadmap toàn bộ đồ án CS431 · MonkeyOCRv2 Text Detection dưới native LR2 (bản sửa 23/11/2026, lần 2)

> Cập nhật 24/11 Codex: hướng mới DỰ KIẾN ở `docs/KE_HOACH_BOX_QUALITY_20261124.md` (với file trong docs, xem cùng thư mục). Chỉ audit và chuẩn bị membership TRAIN; chưa implementation/GPU/kết quả. Không đổi protocol hoặc kết quả cũ. Đọc trạng thái mới nhất trong HANDOFF.

> **Mục tiêu đã đính chính (user, 23/11):** Tuấn đã có 2 hướng có kết quả cải thiện. Trí cần **ít nhất một hướng của mình** có cải thiện có bằng chứng.
> Không hạ cổng, không đổi cách tính, không chọn thêm cấu hình sau khi xem số.

> **Ba mức khác nhau:**
> 1. **có mức tăng** (Δ > 0 với CI không chứa 0);
> 2. **đạt cổng nhóm** (cổng đã khóa);
> 3. **xác nhận trên TEST độc lập**.

> **Nguồn và mức xác minh:**
> - **[repo]**: `MONKEY_V2` @ `b9eaf37`.
> - **[Tuấn cung cấp]**: bản tóm tắt user chuyển 23/11, đọc từ HPC `/datastore/dungmt/textvqa/Tuan`.
> - **[Trí]**: file trong `cải tiến/mới/`.
> - Mọi số trên 2.262 nguồn remaining-validation là **development** (đã dùng lặp lại). Số của Trí đo trên T4 là PENDING_REFERENCE_CHECK.

**Đã sửa so với bản trước** (theo rà soát 23/11):
- tách lợi ích fine-tune so với B0 khỏi so sánh B1 với B-Clean;
- calibration **có** vượt B1 đã fine-tune ở ngưỡng cũ;
- bỏ các câu "không bền", "nguyên nhân là mất thông tin pixel", "thiếu nhãn chặn mọi phương án";
- phân loại kết quả thành 4 mức;
- ghi nguồn cho fusion và Giai đoạn 1;
- **bỏ đề xuất đưa CS-B1 vào TEST**;
- bỏ đề xuất Trí tự chạy EDSR để có kết quả dương.

---

## 1. Bài toán và pipeline
Text detection (không có recognition hay VQA) trên TextVQA, nhãn word-level TextOCR.

```
Clean ──► native LR2 (Pillow 11.3.0 bicubic, W/2×H/2) ──► Resize cạnh ngắn 1024 (scale (4068,1024)) [repo inference_config.py:199-204]
      ──► MonkeyOCRv2-AS ──► FPNC ──► DBHead (P-map) ──► DBPostprocessor (quad) ──► evaluator TextOCR (max matching IoU ≥ .5, ROI, ignore)
```

- Clean và LR2 vào detector cùng kích thước tensor. Phóng cả ảnh là phép tương đương (bicubic ×2: 75,24 so với 75,29 [repo `handoff.md:74`]).
- **Tập dữ liệu:**
  - TRAIN 15.222 nguồn;
  - 2.262 nguồn remaining-validation (development; Trí chia A 1.130 / B 1.132 theo hash, **A ∪ B = đúng 2.262 nguồn của cohort**, đã kiểm ID);
  - TEST nội bộ (đã bị xem ở Phase 4–5 [Tuấn cung cấp]);
  - ICDAR2015.

| Model ([repo `handoff.md:54-56`], `.40/.60`, 2.262 nguồn) | Clean H | LR2 H | LR2 FP | tiny LR2 |
|---|---|---|---|---|
| B0 (gốc) | 74,09 | 62,38 | 4.069 | 1,01% |
| B-Clean (fine-tune 100% Clean) | 80,86 | 75,51 | 6.578 | 15,96% |
| B1 (fine-tune 50/50 Clean/LR2) | 79,43 | 76,90 | 11.997 | 45,79% |

**Chốt baseline từ file gốc** [repo `pr_quad_audit_v1/results/decision_audit_report.md`, `counts/B1_source_counts.npz`; Trí `results/csb1_full_v1_T4/baseline_check.json`]:
- **B1 `.40/.60`** (mask .40, score .60, eps .01) là điểm Tuấn **chọn lại trên lưới mở rộng**: luật tối đa mean Clean + LR2 trên đủ 2.262 nguồn. Đây là điểm **sau** calibration lần 2. Trên đủ tập, nó cũng là setting LR2 tốt nhất trong 72 setting (H200: 76,901).
- Mức **+1,61 pp** là so với **`.30/.40`**, mà báo cáo gọi là "original calibrated" (handoff gọi là B1-cal): 75,291 → 76,901.
- **FP 11.997 so với 11.999:** khác GPU. Counts H200 gốc của Tuấn: TP 56.360, Pred 68.357, FP 11.997. A2r A100: 56.361 / 68.360 / 11.999. Dump T4: 56.361 / 68.349 / 11.988. 32 nguồn lệch ±1 giữa H200 và A100; Hmean trùng 76,90.

---

## 2. Kết quả của Tuấn, xếp theo mức bằng chứng

| So sánh | Kết quả | Mức |
|---|---|---|
| B-Clean so với B0 (thích nghi miền) | LR2 +13,13, Clean +6,77 pp [repo] | **có mức tăng lớn** (không cần CI để thấy) |
| B1 so với B-Clean (trộn độ phân giải) | LR2 **+1,39 CI [−1,26; +3,72]**; Clean **−1,43 CI [−2,11; −0,90]** [repo `handoff.md:58`] | LR2 **chưa đủ bằng chứng**; Clean **âm có ý nghĩa** |
| **Calibration** B1 `.30/.40` → `.40/.60` | LR2 **+1,61 CI [1,01; 2,30]**, FP 16.487 → 11.997; Clean +1,88 [repo `handoff.md:66`] | **có mức tăng**, vượt chính B1 đã fine-tune ở ngưỡng cũ (hậu xử lý, không phải kiến trúc) |
| EDSR ×2 trước B1 (ngưỡng cũ `.30/.40`) | LR2 +0,60 CI [0,24; 0,96]; FP +952; 137 → 205 ms [repo `handoff.md:74`] | **có mức tăng** ở ngưỡng cũ; **chưa đủ bằng chứng** (chưa có Clean, chưa có `.40/.60`) |
| PRC | 75,27 so với 75,29 | âm |
| Feature KD | Δ −0,0005 pp | âm |
| Latent adapter A0/A1/A2 (10k) | 76,87 / 76,86 / 76,86 so với 76,90 | âm |
| CRAFT độc lập | 47,00 LR2 | âm (screen) |
| Fusion FPNC (job 111683) | LR2 nearest 76,94 / ASF 76,95 / DySample 76,94 / bilinear 76,69; ASF − nearest +0,014 CI [−0,08; +0,11] | âm [**Tuấn cung cấp**; repo chỉ có biên nhận khởi động, chưa có báo cáo số] |
| Giai đoạn 1 (FULL_R1 tổng hợp; ASF, BF16/FP32, A′) | A′ degraded +15,04 so với E0, ICDAR2015 −12,58 | [**Tuấn cung cấp**; run folder không còn trên server] |

**"Tuấn đã có 2 hướng cải thiện"** là thông tin user cung cấp. Theo bảng trên, hai mức tăng có CI rõ là **thích nghi miền (fine-tune)** và **calibration**. Thầy có chấp nhận hai hướng đó là "phương án cải tiến" hay không thì **chưa biết**; không được mặc định chỉ từ số đo.

**Q-LRG:** theo repo (`docs/research/s1b-craft-monkey-feasibility-20261119.md`), đây là một dự án khác (VQA, chèn feature trước language model), ngoài phạm vi Text Detection. Trí không có quyền SSH HPC nên chưa đọc.

---

## 3. Kết quả của Trí, xếp theo mức bằng chứng

**Định nghĩa 4 mức:**
- **Âm:** Δ ≤ 0, hoặc trượt ràng buộc cứng (vd. tiny).
- **Dương quá nhỏ:** CI > 0 nhưng dưới cổng.
- **Chưa đạt cổng:** Δ dương nhưng CI chứa 0, hoặc dưới mốc.
- **Chưa đủ bằng chứng:** thiếu phép đo cần thiết.

| # | Phương án | Kết quả | Mức | Nguồn |
|---|---|---|---|---|
| 1 | Mix A-core (nội suy trọng số) | α* trên A = B-Clean; không α nào +0,5 | âm | `docs/report_mix_a_v1.md` |
| 2 | E-sel / E-BN | không đạt / không lệch BN | âm | `results/explore_sel_v1/`, `results/bn_stage0_v1/` |
| 3 | QRoute (router Clean/LR2) | trần trên A +0,27 pp (< 0,5); dựa vào shortcut kích thước ảnh | chưa đạt cổng | `results/diag_v2_cpu/` |
| 4 | DualAgree v1 (khóa trước) | A +0,68 nhưng tiny −19,8 pp | âm (trượt tiny) | `runs/dualagree_analysis_v1_T4/` |
| 5 | Ngưỡng theo kích thước | A +0,62, tiny −7,1 | âm (trượt tiny) | `results/sizethr_v1_T4/` |
| 6 | **CS-B1@0.7** | xem §4 | **dương quá nhỏ** so với B1 `epsilon_005/.50`; **chưa đạt cổng** so với B1 `.40/.60`; Clean (không định tuyến): +0,39 [+0,01; +0,81] so với B1 `.40/.60` nhưng −1,02 [−1,76; −0,49] so với B-Clean → cần router để giữ Clean | `results/csb1_full_v1_T4/`, `results/csb1_clean_v1_T4/` |
| 7 | MSB1 (tách box gộp, mask 0,4) | A +0,22 CI [−0,40; +0,81]; trượt +0,3 | chưa đạt cổng | `docs/msb1_result_20261122.md` |
| 8 | ZB1 zoom cục bộ (pilot 300 nguồn) | G2 LR2 −3,01, Clean −3,65; tiny +10,8 | âm → dừng | `docs/zoom_pilot_result_20261123.md` |
| 9 | XV-B1 (ablation cùng họ DualAgree v1; lọc box B1 ≥ 8 px không có B-Clean xác nhận) | A: +0,153 [−0,46; +0,80] so với `.40/.60`, nhưng −0,079 [−0,14; −0,02] so với cha `epsilon_005/.50`; tiny −3,33; TP bị xóa 128 / FP bị xóa 131; Clean chưa đánh giá | `POSITIVE_BELOW_GATE` theo nhãn khóa; luật tự nó âm → dừng | `docs/xvb1_result_20261123.md` |
| 10 | Flip-TTA trên P-map (B1, 1 checkpoint) | pilot 300 nguồn A: LR2 +0,27 [−0,16; +0,66], Clean +0,21 [−0,22; +0,68]; FP −8%, tiny LR2 −1,18; latency 1,95× | `INCONCLUSIVE` (chưa đủ bằng chứng; trượt c1). Chẩn đoán: so với F nâng ngưỡng, Clean ≈ 0, LR2 +0,19 đến +0,27 (CI chứa 0) → khuyến nghị dừng | `docs/fliptta_result_20261123.md`, `docs/fliptta_diagnosis_plan_20261123.md` |

**Quan sát (không phải kết luận nhân quả):**
- **Ở Clean**, cùng kích thước tensor với LR2 mà tiny recall chỉ 61,2% (LR2: 48,8%). Gợi ý kích thước chữ trên đầu vào là một giới hạn. Khoảng chênh Clean − LR2 **phù hợp** với giả thuyết mất chi tiết pixel, nhưng **chưa chứng minh** đó là nguyên nhân duy nhất.
- **Trang báo thiếu nhãn:** nguồn `6e340fcbb1e0a877` (10 GT, 0 ignore) chiếm 26,8% FP của B1 trên A; 10 nguồn chiếm 48,3%.
  - Thiếu nhãn **có thể** làm evaluator phạt cả phát hiện chữ hợp lệ, và đã góp phần làm CS-B1, MSB1, ZB1 nhiều FP.
  - **Chưa chứng minh** nó chặn mọi phương án tăng recall.
  - Không sửa GT.

---

## 4. CS-B1@0.7: đã đánh giá đến đâu (rà soát 23/11)

**Cấu hình** (`mixab/dualagree.py` `LOCKED`, không đổi):
- Đầu ra B1 `epsilon_005` (mask .30, unclip 1,5, min_width 5, eps .005) với score ≥ 0,5;
- cộng box B-Clean `mask_040` (mask .40, eps .01) với score ≥ 0,7, ghép một-một, khử trùng ở IoU ≥ 0,5.
- Checkpoint: bản sao endpoint `alpha_1p00` / `alpha_0p00` (SHA trong merge manifest `8582c9f3…`).
- GT manifest `052857fb…`; bundle `e06ef54e…`; DBPostprocessor `9bfff8aa…`; MONKEY_V2 `dc19b85`; torch 2.9.0+cu128, T4.
- Evaluator: `CandidateMatcher` của Tuấn. Hmean tính từ **tổng counts**, không lấy trung bình điểm A/B.

**Tập ảnh đã đo:**
- **LR2: đã chạy đủ 2.262 nguồn**, A 1.130 + B 1.132 (dump T4 Colab trên toàn bộ 2.262 ảnh LR2, không ảnh lỗi, không bỏ ảnh).
- **Clean: đã đo 23/11** (Kaggle T4, đủ 2.262 nguồn, không định tuyến) → `docs/csb1_clean_result_20261123.md`.
- Những tập khác **không** dùng để tính Δ chất lượng: 200 ảnh của benchmark latency, 300 nguồn của pilot zoom, 10 thẻ K01–K10.

**Kết quả LR2 trên đủ 2.262 nguồn** [`results/csb1_full_v1_T4/csb1_full.json`]:

| So với | Δ LR2 Hmean | CI95 (ghép cặp theo nguồn) | TP / FP / FN (CS-B1) | tiny |
|---|---|---|---|---|
| B1 `epsilon_005/.50` (nền của CS-B1) | **+0,23 pp** (77,09 so với 76,86) | **[+0,15; +0,32]** | 56.968 / 12.613 / 21.253 | +0,60 pp |
| B1 `.40/.60` (điểm Tuấn chọn trên lưới mở rộng, tốt nhất của B1 trong 72 setting trên đủ tập) | +0,18 pp (77,09 so với 76,91, cùng dump T4) | **[−0,21; +0,58]** | — | −1,96 pp |
| Theo split, so với `.40/.60` | A +0,41 [−0,16; +1,06]; B −0,04 [−0,51; +0,47] | — | — | — |

**Đọc:**
- **Có mức tăng nhỏ** so với chính B1 ở cấu hình hậu xử lý CS-B1 dựa vào.
- **Không phân biệt được** so với B1 `.40/.60`, cấu hình B1 tốt nhất của Tuấn trên lưới 72.
- **Chưa so** với các checkpoint khác của Tuấn (fusion ASF/nearest 76,95/76,94 [Tuấn cung cấp], adapter, EDSR): không có per-source counts, và đó là trọng số khác.
- Lưu ý: `.40/.60` được chọn trên chính 2.262 nguồn, còn `epsilon_005/.50` được Trí chọn trên A. Cả hai đều là điểm chọn sau khi xem dữ liệu.
- Luật dừng đã khóa (so với B1 tốt nhất trong 72 setting trên A, tức `epsilon_005/.50`): A +0,177 < +0,30 → **trượt**. Mức +0,41 so với `.40/.60` trên A **không được dùng** thay, vì như vậy là đổi baseline sau khi xem số.
- **Cổng nhóm (mô tả):** mean hệ thống trên A (Clean dùng B-Clean qua router + LR2 dùng CS-B1) chỉ +0,36 pp so với model đơn tốt nhất (< 0,5). Trên đủ tập là +0,79 pp, nhưng khoảng 0,7 pp trong đó đến từ router (QRoute), không phải từ CS-B1.
- **Chi phí:** latency **1,99×** B1 [1,98; 2,02] trên Kaggle T4 (447 → 889 ms/ảnh); bộ nhớ đỉnh 894 → 991 MiB; cần 2 checkpoint.
- **Phase 0:**
  - 10 nguồn chiếm 66,6% lợi ích ròng trên A, cho thấy lợi ích **tập trung**, chưa tự chứng minh không bền;
  - độ nhạy θ = 0,7 cho Δ âm (do khử trùng yếu hơn, suy luận);
  - **K01–K10 chưa chấm** (`results/phase0/review/calibration_sheet.csv` trống); 70 thẻ chính chưa gửi.

**Còn thiếu:**
1. ~~Clean khi áp CS-B1 cho mọi ảnh~~ **đã đo 23/11**: Clean 79,82 (B1 `.40/.60` 79,44; B-Clean 80,84); mean không định tuyến +0,28 (ALL) / −0,18 (A) so với model đơn tốt nhất; cổng nhóm A không đạt.
2. Chấm K01–K10 rồi 70 thẻ.

**Không còn cách hợp lệ nào** để CS-B1 đạt luật dừng +0,3 hay cổng nhóm +0,5 trên dữ liệu development hiện có. Đo thêm không xóa được thiên lệch do đã xem A/B. **Không đề xuất TEST cho CS-B1**, vì nó đã trượt ở development. Dùng TEST lúc này là để cứu phương án đã bị loại.

---

## 5. Bước tiếp theo theo bằng chứng
1. **(Tùy chọn, user chạy)** Bổ sung Clean cho CS-B1: `notebooks/kaggle_csb1_clean.ipynb` + dataset `runtime/kaggle_upload/cs431-csb1-clean.zip`.
   - **Giải quyết:** CS-B1 có làm hại Clean khi không định tuyến hay không, và mean hệ thống không dựa vào shortcut.
   - **Không giải quyết:** ý nghĩa thống kê trên LR2 so với B1 đã calibration, và cổng.
   - Ước lượng khoảng 1,5–2 giờ Kaggle (chưa đo).
2. **Chấm K01–K10** (user) để hoàn tất Phase 0 và mô tả loại lỗi của box bổ sung.
3. **EDSR:** đây là hướng của Tuấn. Hoàn thiện nó (đo Clean, ngưỡng `.40/.60`, đối chứng bicubic cùng kích thước đầu vào detector và cùng hậu xử lý) phục vụ **kết quả chung của nhóm**, **không** phải đóng góp riêng của Trí. Chỉ làm nếu Tuấn đồng ý và không đang tự chạy. Không gọi phần đo bổ sung là phương pháp mới.
4. **Giữ dừng:** zoom, MSB1, module mới, train mới, quét A/B.
5. **Hỏi thầy / Tuấn:** tiêu chí "cải tiến" (so với B0 hay với B1 đã calibration); có chấp nhận hệ 2 checkpoint với chi phí khoảng 2× không.

**Nói thẳng:** hiện **không có** hướng nào của Trí đạt cổng. Không có lần chạy thêm nào trên dữ liệu hiện có có thể biến CS-B1 thành "đạt cổng" một cách hợp lệ. Muốn có một hướng đạt cổng phải có ý tưởng mới được khóa trước, và tập đánh giá chưa dùng để chọn. Điều đó không hứa trước được kết quả.

---

## 6. Trạng thái bàn giao (23/11)
- **Đang chờ user chạy:** bổ sung Clean cho CS-B1 (`notebooks/kaggle_csb1_clean.ipynb`, dataset 939 MiB `337f4b06…`). Chưa có kết quả.
- **Hướng thứ hai của Trí:** đề xuất ở `docs/SESSION_START_20261123.md` §4 (ưu tiên XV-B1), chưa triển khai, chờ user chọn.
- **Dữ liệu:** không còn tập chưa dùng để chọn phương án (validation = S1 + S2 + remaining đều đã xem; TEST và ICDAR2015 để dành cho benchmark cuối). A/B **không phải** TEST độc lập.
