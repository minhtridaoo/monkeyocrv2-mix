# MONKEY_V2 · Phần của Trí: bài toán, phương pháp đã thử và kết quả hiện tại

Cập nhật 24/11/2026. Trạng thái: **DRAFT, cần Tuấn và nhóm kiểm tra.**
Viết theo khung của tài liệu "MONKEY_V2 · Bài toán, câu hỏi nghiên cứu, phương pháp và kết quả hiện tại" (Tuấn, 24/11). Tài liệu này chỉ nói phần Trí làm. Bài toán chung, EDA của B0/B-Clean/B1 và các phương pháp của Tuấn (PRC, KD, adapter, ASF/DySample, EDSR, calibration) xem ở tài liệu của Tuấn, không chép lại.

**Sơ đồ trực quan (box-to-box):** [pipeline_Tri_20261124_janek.html](pipeline_Tri_20261124_janek.html). Mở bằng trình duyệt.

Phạm vi: nhánh **text detection** (AS → FPNC → DBNet), ảnh TextVQA, nhãn TextOCR, Clean và native LR2. Không có recognition hay VQA.

Mạch trình bày:

1. Phần của Trí nằm ở đâu trong pipeline nhóm (§1)
2. Phát biểu bài toán phần Trí (§2)
3. Nêu vấn đề bằng chẩn đoán CPU (§3)
4. Câu hỏi nghiên cứu (§4)
5. Phương pháp đã thử và lý do chọn (§5)
6. Kết quả hiện tại, đã có cải tiến hay chưa (§6)
7. Đối chiếu với phần Tuấn (§7)
8. Giới hạn và việc cần kiểm (§8)

Quy ước nhãn:

| Nhãn | Ý nghĩa |
|---|---|
| **[TRÍ]** | File trong workspace của Trí (`cải tiến/mới/`), có ghi đường dẫn |
| **[REPO]** | File trong repo `ncat218/MONKEY_V2` của Tuấn |
| **[TUẤN]** | Số Tuấn cung cấp qua user, chưa có trong repo |
| **[TÍNH]** | Tự tính từ số có nguồn |
| **[GIẢ ĐỊNH]** | Suy luận chưa kiểm bằng dữ liệu |
| **[KẾ HOẠCH]** | Đã thiết kế hoặc đã code, chưa chạy GPU |

Mọi số của Trí là **development** trên 2.262 nguồn remaining-validation, một seed, chưa phải TEST. Số đo trên T4 là `PENDING_REFERENCE_CHECK`: T4 lệch A2r (A100) khoảng 3–4% ô, ±1–3 count, không lệch GT; ở các setting dùng cho CS-B1, Hmean lệch ≤ 0,009 pp. [TRÍ `docs/handoff/artifacts/janek/phase0/reference_check_janek.json`]

---

## 1. Phần của Trí nằm ở đâu trong pipeline nhóm

### 1.1 Điểm xuất phát: hai checkpoint có đánh đổi

Tuấn fine-tune từ cùng B0 hai checkpoint, khác nhau duy nhất ở dữ liệu LR2. [REPO `handoff.md` mục 3–4]

| Model (`.40/.60`, 2.262 nguồn) | Clean H | LR2 H | LR2 FP | Recall LR2 < 4 px |
|---|---:|---:|---:|---:|
| B-Clean (100% Clean) | 80,86% | 75,51% | 6.578 | 15,96% |
| B1 (50% Clean / 50% LR2) | 79,43% | 76,90% | 11.997 | 45,79% |

- B-Clean mạnh hơn ở Clean và ít FP hơn nhiều.
- B1 mạnh hơn ở chữ nhỏ trên LR2.
- Trên LR2, B1 cứu thêm 6.150 GT nhưng làm mất 1.453 GT mà B-Clean bắt được (EDA của Tuấn ở `.30/.40`). [REPO `paired_adaptation_eda_report.md`, trích ở TRÍ `research_v2_design_20261121_janek.md` E9]

### 1.2 Vị trí can thiệp

Tuấn can thiệp **bên trong mô hình** (loss, đặc trưng, FPNC, ảnh vào). Trí can thiệp **bên ngoài mô hình, không train detector**:

| Vị trí | Phương pháp của Trí |
|---|---|
| Không gian trọng số (trước khi chạy) | Mix A-core, E-BN |
| Chọn chuyên gia theo ảnh | QRoute |
| Gộp đầu ra của hai checkpoint | DualAgree v1, CS-B1, XV-B1 |
| Hậu xử lý của một checkpoint B1 | Ngưỡng theo kích thước, MSB1, box-quality |
| Cách đưa ảnh vào B1 (test-time) | Zoom cục bộ ZB1, Flip-TTA |

Sơ đồ 1 trong file HTML vẽ từng vị trí trên pipeline chung.

---

## 2. Phát biểu bài toán phần Trí

| Thành phần | Nội dung | Nguồn |
|---|---|---|
| Đầu vào | B-Clean và B1 đã train xong (SHA export `94f0e4ec…` và `9c3814e4…`), ảnh Clean và native LR2 | [TRÍ HANDOFF, "Sự thật đã kiểm"] |
| Ràng buộc | Không train lại detector; chạy được trên T4 (Kaggle) hoặc A100 (Colab); FP32, batch 1, cùng code `rev4.stage_a` của Tuấn | [TRÍ CLAUDE.md, HANDOFF] |
| Câu hỏi | Có tận dụng được đánh đổi B-Clean ↔ B1 để vượt **cả hai** không? | |
| Metric | Micro Hmean bằng `CandidateMatcher` của Tuấn (IoU ≥ 0,5, ROI, ignore); recall theo bin chiều cao; FP; latency | [REPO `rev4/decision_core.py`] |
| Chia dữ liệu | 2.262 nguồn chia theo SHA-256 của source ID: A 1.130, B 1.132. Clean và LR2 của cùng nguồn luôn cùng split. Chọn và khóa trên A; B chỉ để kiểm hoặc mô tả | [TRÍ `Ke_hoach_phoi_hop_MixA_MixB_20261119_janek.md` §4.2] |
| CI | Bootstrap ghép cặp theo nguồn, 2.000 lượt, seed 20261107 | như trên |
| Luật dừng của Trí (khóa trước, trên A) | LR2 ≥ +0,30 pp so với B1 tốt nhất trong 72 setting; tiny ≥ B1 − 3 pp; FP không cao hơn ở cùng recall | [TRÍ HANDOFF 22/11] |
| Cổng nhóm (mô tả) | Mean Hmean ≥ +0,5 pp so với cả hai baseline; Clean ≥ B-Clean − 0,3 pp; tiny ≥ B1 − 3 pp | [TRÍ kế hoạch §4.2; mốc tiny chốt 20/11] |

**Đã duyệt:** Tuấn chọn phương án (a) cho cổng GPU (endpoint tham chiếu chạy trên cùng GPU, 20/11) và đồng ý dùng hệ 2 checkpoint (22/11). Cả hai do user chuyển lời. [TRÍ HANDOFF]

---

## 3. Nêu vấn đề: chẩn đoán CPU trước khi chọn phương pháp

Tất cả chạy trên counts NPZ A2r `(2262, 2, 72, 5)` của Tuấn, không cần GPU. [TRÍ `docs/handoff/artifacts/janek/diag_v2_cpu/`]

### 3.1 Trộn trọng số chỉ đi dọc đường nối hai endpoint

Trên B, mean Hmean của α ¼, ½, ¾ là 78,34 / 78,28 / 78,34. B-Clean đạt 77,87 và B1 đạt 78,11. Mọi CI so với endpoint đều chứa 0 (§6). Clean giảm gần tuyến tính theo α; LR2 lồi nhẹ nhưng không vượt B1. [TRÍ `report_mix_a_v1_janek.md` §4]

### 3.2 Biết điều kiện ảnh (Clean hay LR2) gần như không giúp trên A

- Model đơn tốt nhất trên A: B-Clean, mean 78,51. Oracle điều kiện (B-Clean cho Clean, B1 cho LR2): 78,78. Chênh chỉ **+0,27 pp**, dưới cổng +0,5. [TRÍ `p0_fixed_janek.json`]
- Router theo kích thước ảnh đạt đúng mức oracle, vì cạnh dài Clean 768–1.024 px còn LR2 384–512 px. Đây là **shortcut** của protocol native LR2, không phải router học được. [TRÍ `diagnosis_janek.json`]

### 3.3 Bổ sung thật nằm *bên trong* ảnh LR2

Oracle chọn chuyên gia theo **từng ảnh** (dùng GT, chỉ là cận trên):

| Cặp so sánh trên A | Lợi so với model tốt hơn trong cặp (Clean / LR2, pp) |
|---|---|
| α ½ vs α ¾ (đối chứng) | +0,37 / +0,54 |
| α ¾ vs B1 (đối chứng) | +0,36 / +0,64 |
| B1 hai ngưỡng score (đối chứng) | +1,05 / +0,85 |
| **B-Clean vs B1 (cặp thật)** | **+1,19 / +3,70** |

[TRÍ `p0b_per_image_oracle_janek.json`, `scripts/p0b_null_pairs_janek.py`]

Trên Clean, lợi ngang mức "chọn ngưỡng theo ảnh", nên không thấy bổ sung thật. Trên LR2, cặp thật cao gấp 3–6 lần đối chứng. Oracle điều kiện luôn gửi LR2 sang B1, nên phần bổ sung này nằm trong ảnh LR2, ở mức **từng box**.

### 3.4 Một ảnh thiếu nhãn chi phối FP

Nguồn `6e340fcbb1e0a877` (trang báo) có 10 GT, 0 ignore, ROI 100%; chữ thân bài không có nhãn. Ảnh này chiếm 26,8% FP của B1 trên A; 10 ảnh chiếm 48,3%. Evaluator chính thức khớp. Không sửa GT. [TRÍ `zoom_pilot_result_20261123_janek.md`, kiểm trang báo]

### 3.5 Chốt vấn đề

- **T1.** Trộn trọng số không vượt endpoint (§3.1).
- **T2.** Router theo điều kiện có trần +0,27 pp trên A và dựa vào shortcut kích thước (§3.2).
- **T3.** Phần bổ sung đáng khai thác là ở mức box trong ảnh LR2: B-Clean thấy một số chữ mà B1 bỏ, B1 thấy chữ nhỏ mà B-Clean bỏ (§3.3).
- **T4.** Mọi phương pháp thêm box sẽ chịu phạt từ các vùng chữ thiếu nhãn (§3.4).

---

## 4. Câu hỏi nghiên cứu (phần Trí)

| RQ | Câu hỏi | Từ | Trả lời bằng | Trạng thái |
|---|---|---|---|---|
| RQ-T1 | Nội suy trọng số B-Clean ↔ B1 có cho model ở giữa vượt cả hai không? | §1.1 | Mix A, E-sel, E-BN | Đã trả lời: không |
| RQ-T2 | Chọn chuyên gia theo điều kiện ảnh có đủ không? | T2 | QRoute P0/P0b | Đã trả lời: trần quá thấp |
| RQ-T3 | Gộp box của hai chuyên gia trong ảnh LR2 có vượt B1 mà giữ chữ nhỏ không? | T3 | DualAgree v1, CS-B1, XV-B1 | Có mức tăng nhỏ, chưa đạt cổng |
| RQ-T4 | Chỉ với B1 (một checkpoint), hậu xử lý hoặc test-time có làm được tương tự không? | T3, T4 | Ngưỡng theo kích thước, MSB1, ZB1, Flip-TTA, box-quality | Chưa có hướng đạt; box-quality dương nhỏ trên TRAIN-calibration, dưới cổng |

---

## 5. Phương pháp đã thử và lý do chọn

**Nguyên tắc chung**

- Mỗi phương pháp có spec khóa trước khi chạy. Luật chọn và ngưỡng không đổi sau khi xem số.
- Chọn trên A, mô tả trên B. Kết quả âm được giữ.
- Mọi số Hmean tính từ tổng counts, không lấy trung bình theo ảnh.

| # | Phương pháp | Nhắm | Đối chứng | Trạng thái |
|---|---|---|---|---|
| 5.1 | Mix A-core | RQ-T1 | B-Clean, B1 | Đã chạy (A100), âm |
| 5.2 | E-sel, E-BN | RQ-T1 | như trên | Đã chạy (CPU), âm |
| 5.3 | QRoute | RQ-T2 | Oracle điều kiện | Đã chạy (CPU), dừng |
| 5.4 | DualAgree v1 | RQ-T3 | B1 tốt nhất 72 setting | Đã chạy (T4), âm |
| 5.5 | Ngưỡng theo kích thước (B1) | RQ-T4 | như trên | Đã chạy (CPU trên dump T4), âm |
| 5.6 | **CS-B1@0.7** | RQ-T3 | B1 `epsilon_005/.50` và `.40/.60` | Đã chạy đủ 2.262 nguồn Clean + LR2 (T4) |
| 5.7 | MSB1 | RQ-T4 | B1 | Đã chạy (CPU), chưa đạt |
| 5.8 | ZB1 zoom cục bộ | RQ-T4 | B1 toàn ảnh | Pilot 300 nguồn (Kaggle T4), âm |
| 5.9 | XV-B1 | RQ-T3 | cha `epsilon_005/.50` | Đã chạy (CPU), dừng |
| 5.10 | Flip-TTA trên P-map | RQ-T4 | B1 cùng lượt | Pilot 300 nguồn (Kaggle T4), `INCONCLUSIVE` |
| 5.11 | Box-quality scorer | RQ-T4 | raw score, score-only | [TRÍ] Đã chạy (Kaggle T4); `POSITIVE_BELOW_GATE` trên TRAIN-calibration |

### 5.1 Mix A-core: nội suy trọng số · RQ-T1

**Vì sao.** Hai checkpoint cùng gốc B0 và đánh đổi ngược chiều nhau. Ý tưởng mượn từ WiSE-FT: điểm giữa có thể giữ phần mạnh của cả hai. Không có bảo đảm lý thuyết vì hai endpoint không phải "zero-shot vs fine-tune". [TRÍ `report_mix_a_v1_janek.md` §2]

**Làm gì.** W_α = (1 − α)·W_B-Clean + α·W_B1, α ∈ {0, ¼, ½, ¾, 1}; trộn float64; BN nội suy. 5 model × 2.262 nguồn × 2 điều kiện × 72 setting trong một job A100. Tái lập cùng GPU: P-map lệch 0,0, counts 0/5.760 ô lệch → `VERIFIED` (chỉ xác nhận tái lập). [TRÍ `mixab/weight_merge_janek.py`, `docs/handoff/artifacts/janek/mix_a_v1/`]

### 5.2 E-sel và E-BN · RQ-T1

**E-sel.** Đổi luật chọn: loại cấu hình có tiny thấp hơn B1 quá 3 pp, rồi chọn mean cao nhất. **E-BN.** Kiểm thống kê BN sau khi trộn có lệch không. Cả hai là khám phá sau khi v1 âm. [TRÍ `explore_sel_v1/`, `bn_stage0_v1/`]

### 5.3 QRoute: chọn chuyên gia theo điều kiện · RQ-T2

**Vì sao.** Trên B, định tuyến B-Clean cho Clean và B1 cho LR2 đạt mean 78,93, hơn B-Clean +1,06 và hơn B1 +0,82 pp (khám phá). [TRÍ `partial_from_csv_janek.json`]

**Làm gì.** Tính trần oracle điều kiện trên A, sửa lỗi rò nhãn (khóa một setting mỗi chuyên gia), và tính oracle theo từng ảnh kèm đối chứng (§3.2–3.3). Dừng vì trần trên A chỉ +0,27 pp.

### 5.4 DualAgree v1: chỉ giữ box hai chuyên gia đồng thuận · RQ-T3

**Vì sao.** P0b cho thấy bổ sung ở mức box (§3.3). Box cả hai cùng thấy đáng tin hơn; box chỉ một bên thấy cần ngưỡng cao hơn.

**Làm gì.** Một lần GPU lưu polygon và raw score của B-Clean và B1 trên 2.262 ảnh LR2, 9 geometry, raw ≥ 0,20 (dump T4, shard 64, chạy lại được). Ghép cặp IoU ≥ θ; cặp đồng thuận giữ polygon B1; box đơn lẻ qua ngưỡng t_l, t_c, t_a. Quét 1.928 tổ hợp trên A, khóa một tổ hợp. [TRÍ `mixab/dualagree_janek.py`]

### 5.5 Ngưỡng score theo kích thước box (B1) · RQ-T4

**Vì sao.** Box nhỏ có score thấp hơn do P-map yếu ở biên. Một ngưỡng chung có thể cắt chữ nhỏ hoặc giữ FP lớn.

**Làm gì.** Trên dump B1, chọn ngưỡng score riêng cho từng bin kích thước trên A. [TRÍ `mixab/sizethr_janek.py`]

### 5.6 CS-B1@0.7: B1 cộng box chỉ B-Clean thấy · RQ-T3

**Vì sao.** DualAgree v1 làm mất 527 GT tiny, tất cả do box B1 score 0,5–0,7 bị cắt. Đảo chiều: giữ nguyên toàn bộ B1 và chỉ **thêm** box B-Clean chắc chắn, nhắm vào 1.453 GT B1 làm mất. [TRÍ `phase0_csb1_auto_results_20261122_janek.md` Q4]

**Làm gì.**

- B1 `epsilon_005` (mask 0,30, unclip 1,5, min_width 5), score ≥ 0,5;
- cộng box B-Clean `mask_040`, score ≥ 0,7, không trùng box B1 nào ở IoU ≥ 0,5;
- đánh giá đủ 2.262 nguồn LR2 (dump Colab T4) và 2.262 nguồn Clean (Kaggle T4, không định tuyến);
- đo latency 200 ảnh × 2 lượt, xen kẽ, chỉ GPU 0. [TRÍ `csb1_full_v1_T4/`, `csb1_clean_v1_T4/`, `csb1_latency_v1_T4/`]

Cấu hình được tìm **sau** khi xem kết quả DualAgree v1, nên là post-hoc.

### 5.7 MSB1: tách box gộp bằng mask 0,4 · RQ-T4

**Vì sao.** 85% GT được CS-B1 cứu là chữ bị **gộp** trong một box B1 ở mask 0,3, và `mask_040` tách được. Nếu chỉ cần B1 với mask cao hơn thì bỏ được checkpoint thứ hai. [TRÍ HANDOFF 22/11 khuya]

**Làm gì.** Thay box B1 `epsilon_005` (≥ 0,5) bằng ≥ 2 box B1 `mask_040` (≥ 0,5) nằm ≥ 80% bên trong nó. Một model, không dùng GT. Trước đó tính oracle chính xác (+1,67 pp) để chắc hướng này không bị loại từ đầu. [TRÍ `msb1_spec_20261122_janek.md`]

### 5.8 ZB1: zoom cục bộ bằng B1 · RQ-T4

**Vì sao.** Detector đưa mọi ảnh về cạnh ngắn 1.024, nên phóng cả ảnh không làm chữ to hơn trong mạng. Chỉ cắt vùng rồi đưa vào detector mới làm được. Trên Clean (cùng kích thước tensor), tiny recall của B1 vẫn chỉ 61,2%, gợi ý kích thước chữ trong mạng là nút thắt riêng. [TRÍ `zoom_pilot_spec_20261122_janek.md`; REPO `inference_config.py:199-204`]

**Làm gì.** 300 nguồn A theo hash, Clean + LR2. Arm chính G2: lưới 2×2 ô (5 forward). Hợp nhất: giữ toàn bộ box toàn ảnh, thêm box zoom có cạnh ngắn LR2 < 8 px và IoU < 0,5. Kiểm tọa độ 6/6. [TRÍ `mixab/zoom_janek.py`]

### 5.9 XV-B1: lọc box B1 ≥ 8 px không được B-Clean xác nhận · RQ-T3

**Vì sao.** Ablation cùng họ DualAgree v1: v1 giảm FP trên A từ 6.680 xuống 3.920 nhưng tiny −20 pp. Miễn trừ box < 8 px để giữ chữ nhỏ.

**Làm gì.** Bằng chứng B-Clean lỏng (mọi ứng viên `mask_040`, IoU ≥ 0,3); không phụ thuộc score B1; chỉ split A; luật khóa chạy một lần. [TRÍ `xvb1_spec_20261123_janek.md`]

### 5.10 Flip-TTA trên P-map · RQ-T4

**Vì sao.** Một checkpoint, không cần B-Clean. B1 được train với lật ngang 0,5, nên trung bình P-map ảnh gốc và ảnh lật có thể giảm nhiễu. [TRÍ `fliptta_spec_20261123_janek.md`; REPO `b1_training.py:31-35`]

**Làm gì.** Lật chỉ vùng nội dung (pad giữ nguyên), trung bình số học P-map rồi đưa vào DBPostprocessor. 300 nguồn A, Clean + LR2. Kiểm căn chỉnh cứng 600/600 ảnh. Sau đó chẩn đoán CPU: so với chính B1 nâng ngưỡng ở cùng recall/FP. [TRÍ `mixab/fliptta_janek.py`, `fliptta_diag_v1/`]

### 5.11 Box-quality scorer · RQ-T4 · [TRÍ]

**Vì sao.** Score của DBNet là trung bình P-map trong box, không biết box có ôm đúng chữ không. Một logistic nhỏ trên đặc trưng P-map (một forward B1) có thể xếp lại box tốt hơn raw score.

**Làm gì.** Fit trên TRAIN (15.222 nguồn, không trùng dev), nhãn mô phỏng evaluator; đối chứng raw score và score-only trên cùng tập chọn ngưỡng. P0 đã kiểm: GT dev tái tạo 2.262/2.262; TRAIN không trùng dev; 2.010/2.010 ảnh có trong zip TextVQA chính thức. 

**Kết quả (25/11, Kaggle T4, 400 nguồn TRAIN-calibration):**
- FULL − RAW (ngưỡng raw chọn lại trên cùng tập) = **+0,14 pp** mean Hmean, CI [+0,0004; +0,29]. RAW tự chọn 0,598, gần như 0,60 (RAW − F +0,02), nên phần lợi không đến từ calibration.
- Clean và LR2 cùng +0,14. Tiny LR2 −0,31 pp, Clean −1,55 pp. Chi phí thêm khoảng 3% median.
- Nhãn khóa **`POSITIVE_BELOW_GATE`** (dưới +0,30), nên **dừng theo spec**. Đây là TRAIN mà B1 đã học, không phải development.
[TRÍ `boxquality_result_20261125_janek.md`, `boxquality_v1_T4/`]

---

## 6. Kết quả hiện tại

### 6.1 Bảng tổng hợp

Δ tính bằng pp; CI 95% ghép cặp theo nguồn.

| Phương pháp | Tập | So với | Δ (CI 95%) | Ràng buộc khác | Kết luận | Nguồn |
|---|---|---|---|---|---|---|
| Mix A-core, α\* | A → B | Luật khóa chọn α\* = B-Clean | Không có α ở giữa được chọn | | Âm | `report_mix_a_v1_janek.md` |
| Mix A α ¼ / ½ / ¾ (mô tả) | B, mean H | B-Clean | +0,47 [−0,04; 0,90] / +0,40 [−0,42; 1,26] / +0,46 [−0,53; 1,48] | Không α nào ≥ +0,5 so với cả hai | Âm | như trên |
| E-sel | B | B1 | α ¾: +0,23 | tiny −3,07; FP cùng recall nhiều hơn B1 6–7% | Âm | `explore_sel_v1/` |
| E-BN | | | Không thấy lệch BN đáng kể | | Âm | `bn_stage0_v1/` |
| QRoute | A, mean H | Model đơn tốt nhất | Trần oracle +0,27 | Dựa vào shortcut kích thước | Dừng | `p0_fixed_janek.json` |
| DualAgree v1 | A / B, LR2 | B1 tốt nhất 72 setting | +0,68 / −0,41 | **tiny −19,8** | Âm (trượt tiny) | HANDOFF 22/11 |
| Ngưỡng theo kích thước | A / B, LR2 | như trên | +0,62 / −0,02 | **tiny −7,1** | Âm (trượt tiny) | `sizethr_v1_T4/` |
| **CS-B1@0.7** | 2.262, LR2 | B1 `epsilon_005/.50` | **+0,23 [0,15; 0,32]** | tiny +0,60 | Dương quá nhỏ | `csb1_full_janek.json` |
| **CS-B1@0.7** | 2.262, LR2 | B1 `.40/.60` | +0,18 [−0,21; 0,58] | tiny −1,96 | Chưa đạt cổng | như trên |
| CS-B1@0.7 | A, LR2 (luật dừng) | B1 `epsilon_005/.50` | +0,177 | < +0,30 | Trượt luật dừng | như trên |
| CS-B1@0.7 | 2.262, Clean | B1 `.40/.60` | +0,39 [0,01; 0,81] | | | `csb1_clean_v1_T4/` |
| CS-B1@0.7 | 2.262, Clean | B-Clean | **−1,02 [−1,76; −0,49]** | Clean < B-Clean − 0,3 | Cần router để giữ Clean | như trên |
| MSB1 | A, LR2 | B1 | +0,224 [−0,40; 0,81] | tiny +5,3; +641 TP / +810 FP | Chưa đạt cổng | `msb1_result_20261122_janek.md` |
| ZB1 zoom G2 | 300 A, LR2 / Clean | B1 toàn ảnh | −3,01 [−6,39; 0,66] / −3,65 [−6,99; 0,47] | tiny +10,8; TP +323 / FP +1.566 | Âm, dừng | `zoom_pilot_result_20261123_janek.md` |
| XV-B1 | A, LR2 | B1 `.40/.60` | +0,153 [−0,46; 0,80] | tiny −3,33 | | `xvb1_result_20261123_janek.md` |
| XV-B1 | A, LR2 | cha `epsilon_005/.50` | **−0,079 [−0,14; −0,02]** | xóa 128 TP và 131 FP | Luật tự nó âm, dừng | như trên |
| Flip-TTA | 300 A, LR2 / Clean | B1 cùng lượt | +0,27 [−0,16; 0,66] / +0,21 [−0,22; 0,68] | FP −8%, tiny LR2 −1,18, latency 1,95× | `INCONCLUSIVE` | `fliptta_result_20261123_janek.md` |
| Flip-TTA (chẩn đoán) | 300 A | B1 nâng ngưỡng cùng recall/FP | Clean +0,06; LR2 +0,19 đến +0,27 (CI chứa 0) | | Tương đương nâng ngưỡng, khuyến nghị dừng | `fliptta_diag_v1/` |
| Box-quality | 400 nguồn TRAIN-calibration, Clean + LR2 (mean) | RAW chọn lại ngưỡng trên cùng tập | +0,14 [+0,0004; +0,29] | FP −115 / −72, TP −48 / −19; tiny LR2 −0,31 | `POSITIVE_BELOW_GATE` (không phải dev) | `boxquality_result_20261125_janek.md` |
| **Box-quality (đóng băng)** | **A, 1.130 nguồn, Clean + LR2 (mean)** | **B1 `.40/.60`** | **+0,22 [+0,13; +0,30]** (Clean +0,26 [+0,16; +0,38], LR2 +0,17 [+0,07; +0,27]) | tiny −0,93 / −0,43; latency 1,065×; 1 checkpoint, 1 forward; g1/g2 không đạt | **`SMALL_USEFUL`** | `boxquality_devA_result_20261125_janek.md` |
| Learned box refinement | 400 nguồn TRAIN-calibration (mean) | v1 | −0,82 đến −1,25 (6 checkpoint) | A không đánh giá | `NO_SIGNAL_ON_CALIBRATION`, dừng | `refine_result_20261125_janek.md` |

### 6.2 Chi phí của CS-B1

Latency trên Kaggle T4 (chỉ GPU 0): B1 447 ms/ảnh, CS-B1 889 ms, tỉ số **1,99× [1,98; 2,02]**. Bộ nhớ allocated đỉnh 894 → 991 MiB. Cần 2 checkpoint. Box so với dump Colab T4: 200/200 ảnh khớp. [TRÍ `csb1_latency_v1_T4/latency_janek.json`]

### 6.3 Diễn giải

Các ý dưới đây là [GIẢ ĐỊNH] dựa trên số ở §6.1:

- **Thêm box thì tăng recall nhưng bị phạt FP.** ZB1, MSB1 và CS-B1 đều thêm TP, nhưng FP tăng nhiều hơn hoặc ngang. Ở ZB1, 84% FP của box thêm không chồng GT hợp lệ (nền hoặc chữ thiếu nhãn). Phần nào trong đó là chữ thật chưa biết vì review bằng mắt chưa làm.
- **Lọc box thì giảm FP nhưng cắt chữ nhỏ.** DualAgree v1 và ngưỡng theo kích thước đều trượt tiny. XV-B1 xóa TP gần bằng số FP (128 so với 131).
- **Lợi ích của CS-B1 tập trung.** 10 nguồn chiếm 66,6% lợi ích ròng trên A; bỏ 10 nguồn đó Δ còn +0,066 pp. Đổi θ ghép cặp sang 0,7 thì Δ đổi dấu (−0,32 pp). [TRÍ `phase0/auto_stats_janek.json`]
- **Phần lớn trần P0b +3,70 pp chưa khai thác được.** Luật đơn giản trên box và score chỉ lấy được khoảng 0,2 pp.

### 6.4 Đã có cải tiến chưa?

- **Theo luật dừng của Trí và cổng nhóm: chưa có phương pháp nào đạt.** Chưa phương pháp nào được đưa ra TEST.
- **Có CI dương trên development:** CS-B1 so với nền `epsilon_005/.50` (+0,23 pp LR2) và CS-B1 so với B1 `.40/.60` trên Clean (+0,39 pp). Đây là mức "dương quá nhỏ", tốn 2 checkpoint và 1,99× latency, và CS-B1 vẫn thấp hơn B-Clean ở Clean.
- **Âm hoặc dừng:** Mix A, E-sel, E-BN, QRoute, DualAgree v1, ngưỡng theo kích thước, ZB1, XV-B1.
- **Chưa đủ bằng chứng:** MSB1 (CI chứa 0), Flip-TTA (`INCONCLUSIVE`, tương đương nâng ngưỡng).
- **Phương pháp tốt nhất (chốt 25/11): box-quality scorer v1.** Fit và chọn ngưỡng trên TRAIN, đóng băng, rồi đánh giá một lần trên A: +0,22 pp mean [+0,13; +0,30] so với B1 `.40/.60`, cả Clean và LR2 đều có CI > 0. Chi phí 1,065× latency, không thêm VRAM, một checkpoint. Nhãn `SMALL_USEFUL`: tăng nhỏ nhưng hữu ích, **chưa đạt cổng nhóm** (g1 −0,32, g2 −1,13). Chi tiết: `CHOT_PHUONG_PHAP_Tri_20261125_janek.md`.
- **Âm có kiểm soát (25/11):** learned box refinement. Oracle cho dư địa, nhưng model kém v1 trên TRAIN-calibration, nên A không được đánh giá.
- **Không đề xuất TEST cho CS-B1**, vì nó đã trượt ở development. [TRÍ `ROADMAP_toan_bo_20261123_janek.md` §4]

---

## 7. Đối chiếu với phần Tuấn

| | Tuấn | Trí |
|---|---|---|
| Can thiệp | Bên trong model: loss (PRC, KD), đặc trưng (adapter, ASF, DySample), ảnh vào (EDSR), ngưỡng (calibration) | Bên ngoài model: trộn trọng số, định tuyến, gộp đầu ra, hậu xử lý, test-time |
| Train | Có (1k–10k updates, H200) | Không train detector |
| GPU | H200 | T4 (Kaggle), A100 (Colab) |
| Kết quả dương có CI | Thích nghi miền (B0 → B-Clean +13,13 pp LR2), calibration (+1,61 [1,01; 2,30]), EDSR ở ngưỡng cũ (+0,60 [0,24; 0,96]) [REPO, TUẤN] | CS-B1 so với nền của nó (+0,23 [0,15; 0,32]) |
| Không có lợi | PRC (chưa hợp lệ), KD, adapter, ASF, DySample, CRAFT | Xem §6.4 |

- **Không trùng việc.** Hai bên can thiệp ở hai vị trí khác nhau của pipeline (Sơ đồ 1 trong file HTML).
- **Có thể ghép.** CS-B1 và Flip-TTA không phụ thuộc kiến trúc; chạy được với bất kỳ checkpoint nào của Tuấn. Chưa thử với checkpoint fusion vì chưa có per-source counts. [GIẢ ĐỊNH về tính ghép được]
- **Cùng quan sát.** Cả hai bên đều thấy FP dồn vào vùng chữ thiếu nhãn (P6 của Tuấn; §3.4 ở đây), và review bằng mắt đều còn PENDING (127 thẻ của Tuấn; K01–K10 và 70 thẻ của Trí).
- **Câu hỏi chung chưa có lời đáp:** "cải tiến" được tính so với MonkeyOCRv2 gốc (B0) hay so với B1 đã calibration. Cần hỏi thầy. [TRÍ ROADMAP §5]

---

## 8. Giới hạn và việc cần kiểm

1. **Lệch GPU.** Mọi số T4 là `PENDING_REFERENCE_CHECK`. Lệch đã đo không đổi kết luận của CS-B1 (§0), nhưng chưa kiểm cho từng phương pháp.
2. **Post-hoc.** CS-B1 và XV-B1 được nghĩ ra sau khi xem kết quả DualAgree v1 trên A. B đã bị xem từ Mix A. Không còn tập development chưa dùng.
3. **Review bằng mắt.** K01–K10 chưa chấm; 70 thẻ chính C01–C70 đã tạo, chưa gửi. Nhãn FP hiện là proxy.
4. **Một seed, một cách chia A/B.**
5. **Box-quality** đã chạy trên TRAIN-calibration (gói `_janek`; tương đương logic với gói đã khóa). Chưa có số trên development. Ngưỡng được chọn và đánh giá trên cùng 400 nguồn.
6. **Kết quả fusion của Tuấn** (job 111683) là số Tuấn cung cấp; repo hiện chỉ có biên nhận. [REPO `b9eaf37`]

**Bước tiếp đề xuất (cần user và Tuấn quyết):**

- Box-quality: theo spec thì dừng (`POSITIVE_BELOW_GATE`). Một lượt dev mô tả với model và ngưỡng đã khóa chỉ làm nếu user và Tuấn đồng ý, và phải ghi rõ là ngoài luật.
- Chấm K01–K10 để khóa cách chấm, rồi gửi 70 thẻ.
- Hỏi thầy tiêu chí "cải tiến".
- Không mở thêm biến thể của các hướng đã dừng.
