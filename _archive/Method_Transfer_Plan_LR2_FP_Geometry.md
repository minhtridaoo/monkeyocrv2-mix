# Kế hoạch chuyển giao phương pháp liên miền cho B1 trên native LR2
## Giữ Recall chữ <4 px, giảm FP, sửa lệch biên, bảo toàn Clean

**Ngày:** 19/11/2026 · **Vai trò:** Senior reviewer CV · **Đối tượng:** Thành viên mới triển khai phần cải tiến
**Căn cứ đã đọc:** `MonkeyOCRv2_TextDetection_Progress_Results_Research_Roadmap.md`; repo `Monkey v2/MONKEY_V2` (`handoff.md`, `docs/bclean-and-paired-eda.md`, `docs/handoff/inference_config.py`, `rev4/calibration.py`, `rev4/training_data.py`, `rev4/stage_a.py`, `lowres/inference.py`); `academic-pipeline/references/ai_research_failure_modes.md`.
**Chưa đọc được:** `D:\MINHTRIHOCCODE\LABB\workspace\h_ng_d_n_th_c_chi_n_quy_tr_nh_nghi_n_c_u_ai_ocr.md` không tồn tại tại đường dẫn đó (và không tìm thấy trong `LABB/` theo tên đó; đính chính 22/11: có file nội dung phù hợp là `LABB/workspace/Step by step research.md`, chưa xác nhận là file thay thế). Tài liệu này không dựa vào file đó.

> **Quy ước:** Mọi số liệu trong tài liệu này lấy từ báo cáo của Tuấn hoặc được tính trực tiếp từ những số đó (ghi rõ công thức). Chưa có thí nghiệm nào trong tài liệu này được chạy. Mọi tác dụng của phương pháp chỉ là **giả thuyết**.

---

## 0. Kết luận của reviewer (đọc trước)

1. **Lưới calibration chưa từng thử các ngưỡng chặt hơn.** `rev4/training_data.py:25` định nghĩa `mask_thr ∈ {.30,.20,.15,.10,.05}` và `min_text_score ∈ {.30,.20,.40}`. Cặp đã chọn `0.30/0.40` nằm **ở góc trên cùng của lưới**: đó là mask lớn nhất và score lớn nhất được phép. Khi điểm tối ưu nằm trên biên lưới, điểm tối ưu thật có thể nằm ngoài lưới. Mọi cặp ngưỡng có thể cắt FP (mask > 0,30, score > 0,40) **đều chưa được đo**. Vì vậy chưa thể kết luận +6.030 FP là vấn đề của biểu diễn đặc trưng. Đây là lý do mạnh nhất để chạy Giai đoạn 1 trước khi train bất cứ thứ gì.
2. **"Native LR2" có thể đang bị phóng to bên trong pipeline.** Test pipeline trong `docs/handoff/inference_config.py:32` dùng `Resize(scale=(4068,1024), keep_ratio=True)`. `rev4/stage_a.py:forward` gọi `detector.preprocess` (TextDetInferencer) mà không override bước này. Nếu job TextVQA dùng cùng config, ảnh LR2 (ví dụ 512×384) sẽ được **nội suy lên cạnh ngắn ≈1024** trước backbone. Khi đó Clean và LR2 có kích thước tensor gần như nhau, và LR2 chỉ khác Clean ở chỗ **mất tần số cao (bị mờ)**. Một chữ "<4 px trên LR2" sẽ cao khoảng ~10 px trên tensor. **Phải đối chiếu `input_shapes.csv` (job 48834, cột `resized_hw`/`padded_input`) trước khi lập luận theo hướng tiny-object/stride.** Hệ quả:
   - Các phương pháp chống mất thông tin do stride (SPD-Conv) **kém phù hợp** hơn mức tưởng tượng.
   - Các phương pháp khôi phục **tần số cao và căn chỉnh biên** (FreqFusion, FaPN, boundary loss) **phù hợp hơn**.
   - Nhánh đối chứng "LR2 + bicubic upscale" trong roadmap có thể gần như trùng với điều đang xảy ra (khác ở chỗ bicubic so với bilinear của mmcv).
3. **Ngưỡng hòa vốn của mọi bộ lọc FP (tính từ số của B1).** Từ P = 77,62%, R = 73,10%, FP = 16.487 trên LR2:
   - TP ≈ 57.182, FN ≈ 21.042 (khớp với 21.043 FN trong EDA), D = 2TP + FP + FN ≈ 151.893.
   - Loại một tập dự đoán gồm *f* FP và *t* TP sẽ **tăng Hmean khi và chỉ khi f / t > 2/H − 1 = 1,656**.
   - Nói cách khác: mỗi TP bị mất phải đổi được ≥1,66 FP.
   - Nếu loại sạch 6.030 FP mà không mất TP nào, Hmean LR2 sẽ đạt 114.364 / 145.863 = **78,40%**. Đây là trần lý thuyết của hướng "dập FP".
   - Dùng công thức này làm tiêu chí Go/No-Go có thể kiểm toán.
4. **Tiêu chí thành công đã khóa sẵn trong `handoff.md` §3.3:** Hmean LR2 của phương pháp mới phải hơn B1-cal **≥1,00 pp**, cận dưới CI 95% > 0, và Clean giảm **≤1,00 pp**. Tài liệu này dùng đúng tiêu chí đó, không tự đặt tiêu chí mới. Ràng buộc tiny-text do tôi đề xuất là Recall <4 px ≥ 42,0% (không mất quá 3 pp so với 45,02%). Ràng buộc này cần nhóm duyệt.
5. **Về cách đánh số lỗi:** trong `ai_research_failure_modes.md`, *Methodology fabrication* là **Mode 6**, còn **Mode 4** là *Shortcut reliance*. Tài liệu này kiểm soát cả hai (xem §1). Ràng buộc "trích dẫn gốc + chứng minh tương đồng" thuộc về Mode 2 (citation) và Mode 6.

---

## 1. Stage 2.5 Integrity Gate cho chính kế hoạch này

| Mode | Trạng thái | Bằng chứng / hành động |
|---|---|---|
| 1. Implementation bug | INSUFFICIENT EVIDENCE | Bộ hậu xử lý mới **bắt buộc** tái lập đúng số đếm từng ảnh của DBPostprocessor tại `0.30/0.40` (assert từng ảnh) trước khi sweep. Spec Codex §7 có gate này. |
| 2. Hallucinated citation | INSUFFICIENT EVIDENCE | Trích dẫn ở §8 lấy từ trí nhớ và **chưa được đối chiếu Semantic Scholar/DOI trong phiên này**. Phải chạy `/ars-citation-check` trước khi đưa vào báo cáo đồ án. |
| 3. Hallucinated result | CLEAR (cho tài liệu này) | Không công bố kết quả mới. Mọi con số là số của Tuấn hoặc phép tính có công thức. |
| 4. Shortcut reliance | FLAG | Rủi ro thật: có thể dập FP bằng cách học **bỏ qua chữ không được gán nhãn** (TextOCR bỏ sót chữ nhỏ). Khi đó Precision tăng nhờ khớp nhiễu nhãn, chứ không nhờ phân biệt chữ/nền tốt hơn. Bắt buộc có nhãn `unlabeled-text` trong FP taxonomy (§2.3). |
| 5. Bug reframed as insight | FLAG | Phát hiện về `Resize` ở §0.2 phải được xác nhận bằng `input_shapes.csv` trước khi viết thành "insight". |
| 6. Methodology fabrication | CLEAR có điều kiện | Mọi module đều có paper gốc (§3, §8). Tên module mới chỉ là nhãn mô tả, ví dụ "FreqFusion-lite = FreqFusion bỏ offset generator", không phải phát minh mới. |
| 7. Frame-lock | FLAG | Câu hỏi "cần adapter nào" có thể là frame sai nếu Giai đoạn 1 cho thấy hậu xử lý hoặc nội suy trọng số đã đạt tiêu chí. Kế hoạch có nhánh "dừng, không train". |

---

## 2. Giai đoạn 1: Chẩn đoán nhanh trong 24h (Task 1 + Task 2 của Tuấn, không train)

### 2.1. Câu hỏi chẩn đoán và cách trả lời

| ID | Câu hỏi | Thí nghiệm | Chi phí |
|---|---|---|---|
| D0 | Tensor đầu vào LR2 có bị upscale không? | Đọc `input_shapes.csv` (48834). So `native_size` với `resized_hw` cho Clean và LR2. | 10 phút, CPU |
| D1 | Bao nhiêu trong +6.030 FP có thể cắt bằng ngưỡng? | Cache P-map B-Clean và B1 (2.262 × {Clean, LR2}), sau đó sweep **lưới mở rộng** trên CPU: `mask_thr ∈ {.20,.25,.30,.35,.40,.45,.50}` × `min_text_score ∈ {.30,.40,.50,.55,.60,.65,.70,.80}`. | 1 job GPU ~1h + CPU vài giờ |
| D2 | Bộ lọc nào của quad đang giết chữ <4 px? | Trace từng candidate qua: contour → `min_text_score` → `unclip` → `min_text_width=5` → `max_candidates`. Ghi lý do bị loại. Sweep `min_text_width ∈ {2,3,5}`, `unclip_ratio ∈ {1.3,1.5,1.7,2.0}`. | CPU |
| D3 | Ngưỡng theo kích thước có tách được FP khỏi tiny TP không? | Ngưỡng score phụ thuộc chiều cao candidate *h*: τ(h) = τ_small khi h < h₀, τ_large khi h ≥ h₀. Sweep h₀ ∈ {8,12,16} px trên lưới tensor. | CPU |
| D4 | Hysteresis (2 ngưỡng) có giữ được tiny text không? | Seed = P > τ_hi, vùng = thành phần liên thông của P > τ_lo có chứa seed. τ_lo ∈ {.20,.25,.30}, τ_hi ∈ {.45,.55,.65}. | CPU |
| D5 | Nội suy B-Clean↔B1 có thắng cả hai không? | θ(α) = (1−α)θ_BC + αθ_B1 với α ∈ {.25,.50,.75}, thêm ensemble P-map trong không gian logit. Đo ở lưới mở rộng D1. | GPU ~1,5h |
| D6 | FP thuộc loại nào? | Taxonomy tự động (§2.3), rồi review tay 200 FP lấy ngẫu nhiên có phân tầng. | CPU + 2–3h người |

### 2.2. Chống overfit development

- Chia 2.262 nguồn thành **A/B (1.131/1.131)** theo `sha256(source_id) mod 2`.
- Chọn mọi cấu hình trên A, báo cáo trên B. Không chọn lại trên B.
- Bootstrap CI ghép cặp theo nguồn trên B: 2.000 lượt, seed `20261107`.
- Mọi cấu hình mới đều phải so với **B1-cal và B-Clean đã áp dụng cùng quy tắc chọn**. Không so B1 đã sweep với B-Clean ở ngưỡng cũ.

### 2.3. FP taxonomy tự động (mỗi FP thuộc đúng một nhãn, xét theo thứ tự ưu tiên)

1. `dup`: IoU > 0,5 với một prediction khác có score cao hơn.
2. `merge`: phủ ≥2 GT, mỗi GT có IoA(GT ⊂ pred) > 0,5.
3. `split`: là một trong ≥2 prediction cùng phủ một GT, mỗi cái có IoA(pred ⊂ GT) > 0,5.
4. `loc`: max IoU với GT trong khoảng [0,1; 0,5). Đây là lỗi geometry.
5. `ignore-near`: chồng lên vùng ignore/`###` nhưng dưới ngưỡng loại của evaluator.
6. `bg`: max IoU với GT < 0,1. Đây là false alarm trên nền.
7. Thuộc tính phụ (không loại trừ nhau):
   - `b1_only`: không có prediction B-Clean nào có IoU ≥ 0,5.
   - `tiny`: h < 8 px trên lưới LR2.
   - `score_bin`.

**Review tay (Mode 4):** lấy 200 FP thuộc nhóm `b1_only ∧ bg`, phân tầng theo score_bin × size. Gắn nhãn {nền thật, **chữ không được gán nhãn**, chữ khó đọc, không chắc}. Ghi mẫu số rõ ràng. Nếu nhóm "chữ không gán nhãn" ≥ 30%, phải báo cáo rằng một phần FP là **nhiễu GT** và không thiết kế loss để dập nhóm này.

### 2.4. Bảng quyết định sau Giai đoạn 1 (trên split B)

| Kết quả | Quyết định |
|---|---|
| **G-PP:** có cấu hình zero-training đạt LR2 ≥ B1-cal + 1,00 pp, CI > 0, Clean ≥ B1-cal − 1,00 pp, Recall <4 px ≥ 42% | **Dừng nhánh train.** Báo cáo Công thức 1 là đóng góp. Mọi phương pháp học sau này phải so với B1 + hậu xử lý mới. |
| Có cấu hình cắt ≥50% của +6.030 FP với f/t > 1,66 nhưng chưa đủ +1 pp | Lấy hậu xử lý mới làm **baseline mới**, sau đó làm Công thức 2 hoặc 3 trên nền đó. |
| `loc + split + merge` ≥ 40% FP **và** regressions vẫn nhiều sau sweep | Công thức 2 nhánh **boundary/alignment** (FaPN/FreqFusion-lite hoặc boundary loss). |
| `bg` chiếm đa số, score cao, không cắt được bằng ngưỡng mà không mất tiny TP | Công thức 3 (loss/recipe) hoặc ensemble/veto bằng B-Clean. |
| D5: α trung gian vượt cả hai đầu mút trên cả Clean và LR2 | Đây là bằng chứng có trade-off tuyến tính. Ưu tiên WiSE-FT, và cân nhắc 75/25 (R0) làm đối chứng recipe. |

---

## 3. Bảng ánh xạ phương pháp liên miền (Deliverable 1)

Cột "Tương đồng" giải thích **vì sao bài toán gốc có cùng cấu trúc toán học** với bài toán của nhóm. Cột "Fit" là đánh giá của reviewer sau khi đã tính đến phát hiện §0.2.

| # | Paper & bài toán gốc | Cơ chế cốt lõi | Tương đồng với Scene Text LR2 | Nút thắt nhắm tới | Độ khó code / thời gian | Fit |
|---|---|---|---|---|---|---|
| 1 | **Canny (1986), IEEE TPAMI**: phát hiện biên có nhiễu | Hysteresis: seed ở ngưỡng cao, mở rộng theo liên thông ở ngưỡng thấp | P-map của chữ nhỏ có **đỉnh** đủ cao nhưng **phần lan rộng** yếu. Một ngưỡng duy nhất buộc phải chọn giữa mất chữ nhỏ và giữ nhiễu nền. Hai ngưỡng tách "có tồn tại không" khỏi "biên ở đâu". | FP + tiny Recall | Thấp; ~80 dòng; CPU | Cao |
| 2 | **Baek et al. (2019), CRAFT, CVPR**: text detection mức ký tự | Hai ngưỡng `text_threshold` (seed) và `low_text` (phạm vi vùng) trong hậu xử lý chính thức | Cùng miền scene text, cùng dạng heatmap → connected components. Đây là bằng chứng hysteresis đã dùng được cho text. | FP + Geometry | Thấp | Cao |
| 3 | **Wang et al. (2019), PSENet, CVPR**: chữ hình dạng tùy ý | Bắt đầu từ kernel nhỏ nhất rồi mở rộng dần theo BFS | DB cũng dự đoán **shrink kernel** (r = 0,4) rồi unclip. Mở rộng từ seed giúp giữ đúng tách biệt giữa các instance và giảm `merge`. | Geometry (merge/split) | Trung bình | Trung bình |
| 4 | **Küppers et al. (2020), Multivariate Confidence Calibration for Object Detection, CVPRW** | Hiệu chỉnh confidence **có điều kiện theo vị trí và kích thước hộp** | Lưu ý: một phép calibration đơn điệu toàn cục (temperature, power-law; **Guo et al., 2017, ICML**) **không đổi thứ hạng** score, nên với một ngưỡng chung **không đổi PR curve**. Chỉ calibration **có điều kiện theo kích thước** mới tách được FP lớn khỏi TP nhỏ. Đây là lý do D3 dùng τ(h). | FP với tiny Recall giữ nguyên | Thấp | Cao |
| 5 | **Huang et al. (2019), Mask Scoring R-CNN, CVPR** · **Jiang et al. (2018), IoU-Net, ECCV** | Score cuối = classification score × chất lượng định vị dự đoán | Score DB = mean P trong polygon, **không phản ánh chất lượng hình học**. Polygon lệch biên vẫn có score cao (đúng với quan sát EDA "B1 score cao nhưng polygon lệch"). Bản không học: nhân score với độ đặc của contour, c = area(contour)/area(quad). | Geometry FP (`loc`) | Thấp (bản heuristic); trung bình (bản học) | Trung bình, cần D6 xác nhận |
| 6 | **Wortsman et al. (2022), WiSE-FT, CVPR** · **Model soups, ICML 2022** · **Frankle et al. (2020), Linear Mode Connectivity, ICML** | Nội suy tuyến tính trọng số giữa hai model fine-tune từ cùng init để lấy trade-off robustness/accuracy | B-Clean và B1 có **cùng init B0, cùng sampler index stream, cùng optimizer, LR 2e-5, 10k updates**. Đây là điều kiện lý tưởng để có linear mode connectivity. B-Clean mạnh ở Clean và Precision, B1 mạnh ở tiny Recall, nên θ(α) là một trục trade-off **không cần train**. | Clean drop + FP | Rất thấp; ~30 dòng | **Cao nhất** |
| 7 | **Sunkara & Luo (2022), SPD-Conv, ECML PKDD**: ảnh độ phân giải thấp, vật thể nhỏ | Thay stride/pooling bằng space-to-depth + conv không stride | Chỉ tương đồng nếu tensor thật sự nhỏ. Theo §0.2, LR2 có thể đã bị upscale, nên mất mát nằm ở **pixel** chứ không ở **stride**. Backbone là ViTAE (patch embed), nên thay stride sẽ phá trọng số pretrained. | Tiny Recall | Cao; phá pretrained | **Thấp**, không khuyến nghị |
| 8 | **Chen et al. (2024), FreqFusion, IEEE TPAMI**: dense prediction, lỗi khi hợp nhất feature | Low-pass thích nghi cho feature sâu được upsample + high-pass thích nghi cho feature nông (+ offset) | FPNC upsample P3–P5 lên 1/4 rồi concat. Upsample thô làm **lệch biên và mờ biên** đúng loại lỗi `loc`. LR2 (bị mờ) càng cần high-pass ở P2. Tương đồng: cả hai đều là lỗi dense prediction ở biên đối tượng mảnh. | Geometry + tiny Recall | Trung bình; ~150 dòng; 1 lần train ~2,5h | Cao |
| 9 | **Huang et al. (2021), FaPN, ICCV** | Căn chỉnh feature upsample bằng deformable offset theo feature nông | Cùng vấn đề misalignment trong FPN. Đây là phương án thay thế cho #8. | Geometry | Trung bình–cao (cần DCN) | Trung bình |
| 10 | **Kervadec et al. (2019), Boundary loss, MIDL**: segmentation y tế, mất cân bằng mạnh, biên mờ | Loss = ∫ φ_G · s, với φ_G là distance map có dấu tới biên GT | Chữ nhỏ trên LR2 giống tổn thương nhỏ trong ảnh y tế: vùng dương rất ít pixel, biên mờ, Dice/BCE gần như không phạt lệch biên. Boundary loss phạt theo **khoảng cách**. | Geometry | Thấp–trung bình; 1 lần train | Trung bình–cao |
| 11 | **Deng et al. (2018), PixelLink, AAAI**: scene text segmentation | Instance-balanced cross-entropy: mỗi instance có tổng trọng số như nhau | BCE + OHEM của DB bị pixel của chữ lớn chi phối. Chữ <4 px chiếm rất ít pixel dương. Cân bằng theo instance **giữ tín hiệu học cho tiny text** trong khi loss âm có thể được tăng để dập FP. | Giữ tiny Recall khi tăng phạt FP | Thấp; 1 lần train | Cao (đi cùng #12) |
| 12 | **Lin et al. (2017), Focal loss, ICCV** · **Shrivastava et al. (2016), OHEM, CVPR** · **Mukhoti et al. (2020), NeurIPS** | Giảm trọng số mẫu dễ, tập trung vào mẫu âm khó; focal cũng cải thiện calibration | DB **đã dùng OHEM tỉ lệ 3:1**, nên mọi đề xuất "focal" phải so với OHEM chứ không so với BCE thường. Biến thể hợp lý: focal modulation trên phần âm hoặc OHEM 5:1. | FP | Thấp | Trung bình |
| 13 | **Li & Hoiem (2016), Learning without Forgetting, ECCV** · **Hinton et al. (2015), KD** | Giữ đầu ra trên miền cũ bằng distillation từ model trước | Clean drop của B1 là dạng "quên" do thêm miền LR2. Khác pilot distillation đã thất bại (pilot đó khớp **feature FPNC**), ở đây chỉ distill **P-map trên vùng nền** (mật độ cao). PRC thất bại vì mask chỉ phủ 3,04% microbatch, còn mask nền phủ phần lớn ảnh. | Clean drop + FP | Trung bình; 1 lần train | Trung bình (rủi ro giống pilot cũ) |
| 14 | **Chen et al. (2020), Stitcher, arXiv**: phản hồi theo tỉ lệ loss của vật thể nhỏ | Chỉ đưa thêm dữ liệu vật thể nhỏ khi loss của chúng thấp | Thay vì 50/50 cố định, chỉ chọn view LR2 cho nguồn có nhiều chữ nhỏ. Lợi ích tiny-text tập trung ở <8 px, nên LR2 cho ảnh toàn chữ lớn có thể chỉ thêm FP. | Clean drop + FP | Thấp (sampler) | Trung bình; cần R0 75/25 làm đối chứng |

**Đã cân nhắc và loại:** NWD (Wang et al., 2021) cho tiny object. NWD thay IoU trong label assignment hoặc NMS, nhưng DBNet không có anchor/NMS và evaluator IoU ≥ 0,5 đã khóa, nên NWD không có chỗ để cắm vào.

---

## 4. Ba "Công thức Lẩu"

### Công thức 1: Zero-training. "WiSE-DB + Hysteresis + τ(h)" *(ưu tiên số 1)*

Gia vị gồm #6 (WiSE-FT), #1–#2 (hysteresis Canny/CRAFT), #4 (calibration theo kích thước) và tùy chọn #5 (score theo độ đặc).
**Logic kết hợp:**
- Nội suy trọng số chọn **điểm trên trục Clean ↔ tiny-text** ở mức biểu diễn.
- Hysteresis tách **tồn tại** khỏi **phạm vi**.
- τ(h) tách **FP lớn** khỏi **TP nhỏ**.

Ba thành phần tác động lên ba bậc tự do khác nhau, nên đây là phép kết hợp có cơ sở chứ không phải ghép module tùy tiện. Ablation phải bật từng thành phần một (§5.4).

### Công thức 2: Plugin nhẹ ở neck/head. "FreqFusion-lite trong FPNC + Boundary loss"

Gia vị gồm #8 (FreqFusion, bỏ offset generator), tùy chọn thêm #10 (boundary loss cho P-map). Chỉ làm nếu D6 cho thấy `loc/split/merge` ≥ 40%, hoặc regressions geometry vẫn còn sau Công thức 1.

### Công thức 3: Recipe/loss. "IB-BCE + focal-negative + background-KD từ B-Clean"

Gia vị gồm #11 (instance-balanced), #12 (focal trên âm), #13 (distill P-map nền của B-Clean trên view LR2), và đối chứng #14/R0 (75/25).

**Hàm loss** (init từ B1 hoặc B0; ghi rõ phương án nào được dùng):

L = L_s^IB + α·L_b + β·L_t + λ_bg·L_bg

- **L_s^IB** (instance-balanced BCE trên P):
  - Pixel dương thuộc instance i (kernel S_i) có trọng số w_i = (Σ_j |S_j|) / (N·|S_i|).
  - Pixel âm giữ OHEM 3:1, hoặc thay bằng focal: −(p)^γ·log(1−p), với γ ∈ {1, 2}.
- **L_b, L_t:** giữ nguyên như DBNet (Dice trên approximate binary map, L1 trên threshold map). Hệ số α, β giữ theo `DBModuleLoss` hiện tại, cần đọc config để xác nhận.
- **L_bg** (background KD):
  - Tập pixel Ω = {x : G(x)=0, không ignore, dist(x, GT) > 3 px, P^T_Clean(x) < 0,05}, với P^T là P-map của B-Clean đóng băng trên view Clean cùng nguồn, ánh xạ sang lưới LR2.
  - L_bg = mean_{x∈Ω} −log(1 − P^S_LR2(x)).
  - Chỉ áp dụng cho microbatch có view LR2.
  - Ghi log tỉ lệ microbatch có |Ω| > 0 để tránh lặp lại lỗi phủ thấp như PRC (3,04%).
- **Cảnh báo Mode 4:** nếu D6 cho thấy nhiều FP là chữ không gán nhãn, L_bg sẽ dạy model bỏ qua chữ thật. Trong trường hợp đó, chỉ áp dụng L_bg cho Ω ở vùng mà **cả B-Clean lẫn B0** đều có P < 0,05.

---

## 5. Phương án tối ưu: chi tiết toán học và luồng tensor (Deliverable 2)

**Lý do chọn Công thức 1 làm phương án tối ưu:**
- Chi phí GPU thấp nhất (không train).
- Nhắm trực tiếp vào phát hiện §0.1 (lưới ngưỡng bị chặn).
- Dùng hai checkpoint đã có.
- Nếu không đạt, kết quả vẫn trở thành baseline bắt buộc cho Công thức 2/3.

Phần tensor flow của Công thức 2 ở §5.5, dùng khi cần train.

### 5.1. Luồng tensor

Ký hiệu: ảnh LR2 có kích thước h×w = (H//2)×(W//2). Các stride 4/8/16/32 **giả định theo quy ước `res2–res5`** và cần xác nhận bằng cột backbone/FPNC trong `input_shapes.csv`.

```text
I_LR2           uint8  h × w × 3
  │ Resize keep_ratio, scale (4068,1024)   ← cần xác minh D0
  │   s = min(4068/max(h,w), 1024/min(h,w)); H'=round(s·h), W'=round(s·w)
  │ Pad tới bội số 32                       → Hp × Wp
x               fp32   1 × 3 × Hp × Wp
  │ Backbone MonkeyOCRv2-AS với trọng số θ(α) = (1−α)·θ_BC + α·θ_B1   (mọi tensor float trong state_dict, kể cả BN running_mean/var)
C2  1 × 64  × Hp/4  × Wp/4
C3  1 × 128 × Hp/8  × Wp/8
C4  1 × 256 × Hp/16 × Wp/16
C5  1 × 512 × Hp/32 × Wp/32
  │ FPNC: lateral 1×1 → 256; top-down cộng; smooth 3×3 → 64 mỗi mức;
  │       upsample về Hp/4; concat 4 mức
F   1 × 256 × Hp/4 × Wp/4
  │ DBHead (binarize): Conv3×3 256→64, BN, ReLU, ConvT 2×2 s2 64→64, BN, ReLU, ConvT 2×2 s2 64→1, Sigmoid
P   1 × Hp × Wp   ∈ [0,1]                    (khớp require trong rev4/stage_a.py: P-map ở lưới padded input)
  │ Cắt về H'×W' (vùng hợp lệ) → hậu xử lý WiSE-DB (5.2) → quads trên lưới H'×W'
  │ Chia cho scale_factor → tọa độ LR2 → nhân (W/w, H/h) → tọa độ Clean để chấm (như evaluator hiện tại)
```

**Biến thể ensemble đầu ra** (đối chứng cho nội suy trọng số, tốn 2 lần forward):
P_ens = σ((1−α)·logit(P_BC) + α·logit(P_B1)). Đây là log-linear pooling, tương đương tích có trọng số của hai "expert".

### 5.2. Hậu xử lý WiSE-DB

Cho P ∈ [0,1]^{H'×W'} và các tham số Φ = (τ_lo, τ_hi, τ_s,small, τ_s,large, h₀, r_unclip, w_min, γ_c):

1. **Seed:** S = {x : P(x) > τ_hi}.
2. **Vùng:** M = 𝟙[P > τ_lo]. Lấy các thành phần liên thông 8-neighbour {R_k} của M. Giữ R_k khi R_k ∩ S ≠ ∅.
   Khi τ_hi = τ_lo = mask_thr, bước này trở về đúng DBPostprocessor gốc. **Đây là điều kiện tái lập bắt buộc.**
3. **Quad:** b_k = minAreaRect(contour(R_k)), giống đường `quad` hiện tại (với `epsilon_ratio = 0,01`, `max_candidates = 3000` giữ nguyên).
4. **Score:** s_k = mean_{x∈b_k} P(x), như DB gốc.
   - Tùy chọn (#5): s'_k = s_k · c_k^{γ_c}, với c_k = |R_k| / area(b_k) ∈ (0,1].
   - Với γ_c = 0, s'_k trở về s_k.
5. **Ngưỡng theo kích thước (#4):** h_k = cạnh ngắn của b_k trên lưới H'×W'. Giữ b_k khi s'_k ≥ τ_s(h_k), với τ_s(h) = τ_s,small nếu h < h₀, ngược lại τ_s,large.
6. **Unclip:** offset D_k = area(b_k) · r_unclip / perimeter(b_k). Dilate rồi lấy lại minAreaRect. Bỏ nếu cạnh ngắn < w_min.

**Mặc định tái lập B1-cal:** α = 1, τ_lo = τ_hi = 0,30, τ_s,small = τ_s,large = 0,40, r = 1,5, w_min = 5, γ_c = 0.

### 5.3. Mục tiêu chọn tham số trên split A

max_{α,Φ}  ½·(H_Clean + H_LR2)    (giữ nguyên tiêu chí chọn của protocol)

với các ràng buộc:
- R_<4px,LR2 ≥ 0,42
- H_Clean ≥ H_Clean(B1-cal) − 1,00

**Lưới đề xuất:**
- α ∈ {0, .25, .5, .75, 1}
- τ_lo ∈ {.20, .25, .30}
- τ_hi ∈ {.30, .45, .55, .65}
- τ_s,small ∈ {.30, .40, .50}
- τ_s,large ∈ {.40, .50, .60, .70}
- h₀ ∈ {8, 12, 16}
- r ∈ {1.3, 1.5, 1.7}
- w_min ∈ {3, 5}
- γ_c ∈ {0, 1}

Lưới này quá lớn cho tìm kiếm vét cạn. Dùng **tìm kiếm theo tọa độ** theo thứ tự (mask/score toàn cục → τ_hi → τ(h) → r, w_min → γ_c), mỗi bước giữ tham số tốt nhất. Khai báo trước số cấu hình đã thử để báo cáo minh bạch (multiple-comparison).

### 5.4. Ablation bắt buộc (báo cáo trên split B)

| Hàng | α | Hysteresis | τ(h) | γ_c | Câu hỏi |
|---|---|---|---|---|---|
| A0 | 1 | không | không | 0 | B1-cal gốc (tái lập) |
| A1 | 1 | không | không | 0 | Chỉ mở rộng lưới ngưỡng |
| A2 | 1 | có | không | 0 | Giá trị của hysteresis |
| A3 | 1 | có | có | 0 | Giá trị của τ(h) |
| A4 | α* | không | không | 0 | Giá trị của nội suy trọng số |
| A5 | α* | có | có | γ* | Toàn bộ |
| A6 | 0 (B-Clean) | có | có | γ* | Cùng hậu xử lý áp lên B-Clean (đối chứng công bằng) |

### 5.5. Tensor flow của Công thức 2 (FreqFusion-lite, dùng khi Giai đoạn 1 chỉ sang hướng geometry)

Thay bước "upsample về Hp/4" trong FPNC cho P3, P4, P5. P2 được làm sắc bằng high-pass.

```text
Pk (k=3..5)  1 × 64 × Hp/2^k × Wp/2^k   (sau smooth 3×3)
U_k = Upsample_bilinear(Pk)            1 × 64 × Hp/4 × Wp/4
Z_k = concat(U_k, P2)                  1 × 128 × Hp/4 × Wp/4
K_lp,k = softmax_c( Conv1×1(128→25)(Z_k) )   1 × 25 × Hp/4 × Wp/4   (kernel 5×5 cho từng pixel)
Ũ_k = U_k + γ_k · ( Σ_25 K_lp,k ⊙ unfold5×5(U_k) − U_k )     γ_k khởi tạo = 0
K_hp = softmax_c( Conv1×1(64→9)(P2) )        1 × 9 × Hp/4 × Wp/4
P̃2 = P2 + γ_2 · ( P2 − Σ_9 K_hp ⊙ unfold3×3(P2) )         (high-pass = identity − low-pass), γ_2 = 0
F  = concat(P̃2, Ũ3, Ũ4, Ũ5)           1 × 256 × Hp/4 × Wp/4   → DBHead như cũ
```

- **Tham số thêm:** 3×(128·25 + 25) + (64·9 + 9) ≈ 10,4k (≈0,01% của model).
- **Khởi tạo:** γ = 0 nên tại bước 0 model **đúng bằng B1**. Có thể dùng lại control "B1 train tiếp" (77,33/75,24) nếu nó có cùng ngân sách và cùng thành phần trainable. Nếu khác, phải chạy lại control.
- **Tùy chọn boundary loss (#10):** L_bd = mean_x φ_G(x)·P(x), với φ_G là signed distance (âm bên trong) tới biên polygon GT shrink. Tăng λ_bd tuyến tính từ 0 lên 0,01 trong 2k update đầu.
- **Đối chứng tối thiểu:** cùng init B1, cùng 15.222 ID, cùng số update, cùng optimizer/scheduler/augmentation/seed 42, cùng trạng thái trainable (backbone frozen hoặc không, giống nhau ở cả hai), chỉ khác γ được học hay không.

---

## 6. Kế hoạch tác chiến theo ngày (Deliverable 4)

| Ngày | Việc | Checkpoint | GPU? | Go/No-Go cuối ngày |
|---|---|---|---|---|
| **Thứ Năm 19/11 (D0, tối)** | D0: đọc `input_shapes.csv`, `fp_analysis.csv`, `model_metrics.csv`. Xác nhận resize. Đếm GT <4 px. Giao spec §7 cho Codex. | có sẵn | Không | Nếu `resized_hw` ≠ `native_size` cho LR2: ghi phát hiện, hạ ưu tiên SPD-Conv. |
| **Thứ Sáu 20/11 (D1)** | Job cache P-map B-Clean/B1 (Clean + LR2), D5 (α = .25/.5/.75 cache P-map), tái lập counts tại 0.30/0.40. | B-Clean, B1 | **Có, ~2–3h** (1 job) | **Gate tái lập:** counts từng ảnh khớp 100%. Nếu FAIL thì dừng và sửa, không sweep. |
| **Thứ Bảy 21/11 (D2)** | Sweep CPU D1–D4 trên split A. FP taxonomy D6 tự động. Bắt đầu review tay 200 FP. | cache | Không | Lập bảng §2.4. Nếu **G-PP đạt trên A**: chấm B một lần, chuyển sang viết. |
| **Chủ Nhật 22/11 (D3)** | Chốt cấu hình Công thức 1 trên A, chấm B, CI. Hoàn tất review tay. Quyết định có train hay không. | cache | Không | **Go train** chỉ khi đúng một hàng của §2.4 trỏ tới Công thức 2 hoặc 3. Nếu không, dừng train. |
| **Thứ Hai 23/11 (D4)** | Code Công thức 2 hoặc 3 + unit test (γ = 0 cho đầu ra trùng B1 đến 1e-6). Pilot 500 update để kiểm tra loss/mask coverage. | B1 | Có, ~20 phút | Nếu L_bg coverage < 20% microbatch, hoặc loss NaN, hoặc đầu ra lúc γ = 0 lệch B1: sửa trước khi train full. |
| **Thứ Ba 24/11 (D5)** | Train method + matched control song song (mỗi job ~2,5h, 10k update). | B1 hoặc B0 | **Có, 2 job** | - |
| **Thứ Tư 25/11 (D6)** | Chấm cả hai trên split B với hậu xử lý Công thức 1 đã chốt. CI ghép cặp method − control. | mới | Có, ~1h | **Go seed 2** khi Δ LR2 ≥ +1,0 pp so với B1 + Công thức 1, CI > 0, Clean ≥ −1,0. Nếu không đạt: báo kết quả âm. |
| **Thứ Năm 26/11 (D7)** | Seed 2 (nếu Go) hoặc viết báo cáo kết quả âm và cập nhật roadmap. | - | Tùy | Khóa cấu hình. **Chỉ sau đó** mới đề xuất chấm TEST theo `handoff.md` §3. |

**Ngân sách GPU tối đa:** khoảng 3h (D1) + 0,3h (D4) + 5h (D5) + 1h (D6) + 2,5h (seed 2) ≈ **12h**. Nếu G-PP đạt ở D2/D3 thì chỉ tốn ≈ **3h**.

---

## 7. Subtask Spec cho Codex CLI: chẩn đoán PR/Threshold & FP Audit (Deliverable 3)

```markdown
# SUBTASK: pp_audit — Extended threshold/post-processing sweep + FP taxonomy for B-Clean vs B1 (LR2 + Clean)

## Context
Repo: MONKEY_V2 (local `D:/MINHTRIHOCCODE/HOCKI5/CS431/Đồ án/Monkey v2/MONKEY_V2`; HPC `/datastore/dungmt/textvqa/Tuan/MONKEY_V2`).
Model: MonkeyOCRv2-AS + FPNC + DBHead (MMOCR), FP32, DBPostprocessor `quad`.
Frozen facts you must respect:
- Validation cohort = 2,262 remaining-validation sources, conditions {clean, lr2}. Load via `rev4.calibration.validation_cohort`.
- NEVER read TEST data. NEVER rewrite historical artifacts/reports/checkpoints. Write only to a new output dir.
- Native postprocessor (asserted in `rev4/stage_a.detector_for`): mask_thr .30, min_text_score .30, min_text_width 5,
  unclip_ratio 1.5, text_repr_type 'quad', epsilon_ratio .01, max_candidates 3000. Common operating point = 0.30/0.40.
- Matching/evaluator: `lowres/evaluate_shared.evaluate_image` (one-to-one max matching, IoU >= 0.50, ROI/ignore rules from
  `docs/handoff/textvqa_evaluation_protocol.json`). Matching trace helpers: `rev4/paired_eda_core.py`
  (`filtered_geometry`, `trace_matching`, `fp_transitions`).
- Model build/forward: `rev4/stage_a.detector_for`, `rev4/stage_a.forward` (returns P-map at padded input grid + metainfo).
- Checkpoints: B-Clean (job 48833 output, see `rev4/bclean.py` for path/identity) and B1 (`docs/handoff/weights/b1-inference.pth`,
  SHA in `docs/handoff/checkpoints.md`). Verify SHA256 before use.
- Bootstrap: paired by source, 2,000 replicates, seed 20261107 (`lowres/evaluate_shared.bootstrap_paired` or the EDA's version).
- Height bins on native-LR2 AABB grid: <4, 4–<8, 8–<16, >=16 px (reuse the definition in `rev4/paired_eda_core.gt_attributes`).

## Deliverables (new files only)
1. `rev4/pp_audit.py` — worker (GPU part + CPU sweep part, separable via `--stage cache|sweep|taxonomy|report`).
2. `rev4/pp_audit_post.py` — pure-numpy/OpenCV re-implementation of DB quad post-processing with extensions (below).
3. `submit_pp_audit.py` — launcher mirroring `submit_paired_eda.py` (plan-only mode, login-node lightweight, SHA ticket).
4. `tests/test_pp_audit.py` — synthetic tests (see Tests).
5. Outputs under `<out>/pp_audit/`: `pmap_cache/`, `reproduction_check.json`, `threshold_frontier.csv`, `pr_curves.csv`,
   `quad_filter_audit.csv`, `fp_taxonomy.csv`, `fp_review_sample.csv`, `split_manifest.csv`, `calibration_diagnostics.md`.

## Stage `cache` (GPU, one job)
- For model in {bclean, b1, interp_a25, interp_a50, interp_a75} and condition in {clean, lr2}, for every source:
  run `forward`; save P-map cropped to valid `img_shape` as float16 `.npz` (key `p`) + metainfo JSON (scale_factor, img_shape,
  ori_shape, padded shape, restore factors). Reuse the atomic `.npz.tmp → replace` pattern from `lowres/inference.py`.
- Interpolated models: θ(α) = (1−α)·θ_bclean + α·θ_b1 over every floating-point tensor in both state_dicts (including BN
  running stats); integer buffers (e.g. num_batches_tracked) copied from b1. Assert identical key sets and shapes.
- Record per-sample latency and peak VRAM, but label as non-benchmark timing.

## REPRODUCTION GATE (must pass before any sweep; on FAIL write reproduction_check.json and exit non-zero)
- For bclean and b1, at (mask .30, score .40) and native (.30, .30), run `pp_audit_post` on the float16 cache and require
  per-image (TP, GT, #pred) EXACTLY equal to the historical per-image counts (b1: calibration grid cache used by
  `rev4/b1_diagnostics.py`; bclean: grid cache from `rev4/bclean.py`). Also require equality with calling the MMOCR
  DBPostprocessor on the same cached map. If float16 breaks exactness, re-cache that model as float32 and re-check.

## Stage `sweep` (CPU, multiprocessing)
Split: source in A if int(sha256(source_id).hexdigest(),16) % 2 == 0 else B. Save `split_manifest.csv`.
`pp_audit_post.postprocess(P, params)` must implement, with defaults reproducing DBPostprocessor exactly:
  1. hysteresis: seeds P>tau_hi; regions = 8-connected components of P>tau_lo containing ≥1 seed (tau_hi==tau_lo → original)
  2. quad via minAreaRect on each region contour (same contour extraction/order/max_candidates as MMOCR)
  3. score s = mean P inside quad (same polygon-mask rasterization as MMOCR `_get_bbox_score`/box score)
  4. optional compactness rescoring s' = s * (|region|/area(quad))**gamma_c
  5. size-conditioned cutoff: keep if s' >= (tau_s_small if short_side(quad) < h0 else tau_s_large); short side on network grid
  6. unclip (pyclipper, ratio r) + min_text_width w_min, exactly as MMOCR
  Every candidate gets a `drop_reason` ∈ {kept, score, width, unclip_fail, max_candidates} → `quad_filter_audit.csv`
  (per candidate: model, condition, source, area, short_side, s, s', drop_reason, matched_gt_id_if_kept, best_gt_iou).
Grids (coordinate-wise, evaluate on split A, log every evaluated config with an incrementing config_id):
  step1 global: mask_thr {.20,.25,.30,.35,.40,.45,.50} × score {.30,.40,.50,.55,.60,.65,.70,.80}
  step2 hysteresis: tau_lo {.20,.25,.30} × tau_hi {.30,.45,.55,.65} (tau_hi ≥ tau_lo)
  step3 size cutoff: h0 {8,12,16} × tau_s_small {.30,.40,.50} × tau_s_large {.40,.50,.60,.70}
  step4 geometry: unclip {1.3,1.5,1.7,2.0} × w_min {2,3,5}
  step5 gamma_c {0,1}
  Run all steps for each model in {bclean, b1, interp_a25, interp_a50, interp_a75}.
Per config × model × condition write: TP, FP, FN, P, R, Hmean (micro: 2TP/(2TP+FP+FN)), recall per height bin,
  FP count, FP among `b1_only` set, and mean Clean/LR2 Hmean → `threshold_frontier.csv`; PR points → `pr_curves.csv`.
Selection rule (split A): maximize mean(Hmean_clean, Hmean_lr2) subject to recall_<4px_lr2 ≥ 0.42 and
  Hmean_clean ≥ Hmean_clean(B1 @0.30/0.40) − 1.00 pp; ties → closer to native params. Then evaluate the chosen config
  ONCE on split B and compute paired bootstrap CI vs B1@0.30/0.40 and vs B-Clean@(its own split-A-selected config).
Also report break-even check: for each config vs B1@0.30/0.40 on LR2, removed FP f and lost TP t, ratio f/t, and
  threshold 2/H−1 (H = B1 LR2 Hmean).

## Stage `taxonomy` (CPU), at B1 @0.30/0.40 and at the split-A-selected config, LR2 and Clean
Each FP gets exactly one primary label by priority: dup (IoU>0.5 with higher-score pred) > merge (covers ≥2 GT with
IoA(GT⊂pred)>0.5) > split (one of ≥2 preds each IoA(pred⊂GT)>0.5 of the same GT) > loc (max GT IoU in [0.1,0.5)) >
ignore_near (any overlap with ignore/### region, below evaluator exclusion) > bg (max GT IoU < 0.1).
Attributes: b1_only (no B-Clean pred with IoU ≥ 0.5 at B-Clean's same operating point), tiny (short side < 8 px on LR2
grid), score_bin [.4,.5),[.5,.6),[.6,.7),[.7,1]. Write `fp_taxonomy.csv` and counts with explicit denominators.
Review sample: 200 FPs from (b1_only ∧ bg), stratified by score_bin × tiny, fixed seed 20261119; export crops
(original + P-map + GT overlay) and `fp_review_sample.csv` with empty columns: human_label ∈ {background, unlabeled_text,
hard_text, unsure}, reviewer, note. Do NOT reuse the 192 illustrative EDA examples.

## Stage `report`
`calibration_diagnostics.md`: reproduction gate result; best config per step on A; split-B result with CI; how many of
the +6,030 B1−B-Clean LR2 FPs are removed and at what TP cost; tiny-recall vs FP curve; taxonomy table; explicit
statement "development data, post-selection descriptive CI, single training seed".

## Tests (`tests/test_pp_audit.py`, run inside worker before research workload, like existing jobs)
- Synthetic P-maps: default params reproduce MMOCR DBPostprocessor output (boxes, scores) exactly.
- tau_hi==tau_lo equals original; hysteresis keeps a low-plateau blob only when it contains a seed.
- size cutoff applies per-candidate; gamma_c=0 is identity.
- θ(α) interpolation: α=0 and α=1 reproduce bclean / b1 state_dicts bit-exactly.
- split assignment deterministic; taxonomy labels mutually exclusive and exhaustive on a hand-built case.

## Constraints
- No TEST, no changes to frozen evaluator, no editing historical files. Fail loudly (`rev4.require`) on any hash/shape mismatch.
- Keep login-node work lightweight; one GPU job for `cache`; CPU stages may run on the same allocation.
- Report numbers only from files produced by this run; never hand-copy metrics into the report.
```

---

## 8. Danh mục trích dẫn (trạng thái: **chưa đối chiếu DB trong phiên này**, phải chạy `/ars-citation-check`)

1. Liao, M., Wan, Z., Yao, C., Chen, K., Bai, X. *Real-time Scene Text Detection with Differentiable Binarization.* AAAI 2020.
2. Canny, J. *A Computational Approach to Edge Detection.* IEEE TPAMI, 8(6), 1986.
3. Baek, Y., Lee, B., Han, D., Yun, S., Lee, H. *Character Region Awareness for Text Detection (CRAFT).* CVPR 2019.
4. Wang, W., Xie, E., Li, X., Hou, W., Lu, T., Yu, G., Shao, S. *Shape Robust Text Detection with Progressive Scale Expansion Network (PSENet).* CVPR 2019.
5. Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. *On Calibration of Modern Neural Networks.* ICML 2017.
6. Küppers, F., Kronenberger, J., Shantia, A., Haselhoff, A. *Multivariate Confidence Calibration for Object Detection.* CVPR Workshops 2020.
7. Huang, Z., Huang, L., Gong, Y., Huang, C., Wang, X. *Mask Scoring R-CNN.* CVPR 2019.
8. Jiang, B., Luo, R., Mao, J., Xiao, T., Jiang, Y. *Acquisition of Localization Confidence for Accurate Object Detection (IoU-Net).* ECCV 2018.
9. Wortsman, M., Ilharco, G., Kim, J. W., et al. *Robust Fine-tuning of Zero-shot Models (WiSE-FT).* CVPR 2022.
10. Wortsman, M., Ilharco, G., Gadre, S. Y., et al. *Model Soups.* ICML 2022.
11. Frankle, J., Dziugaite, G. K., Roy, D., Carbin, M. *Linear Mode Connectivity and the Lottery Ticket Hypothesis.* ICML 2020.
12. Sunkara, R., Luo, T. *No More Strided Convolutions or Pooling: A New CNN Building Block for Low-Resolution Images and Small Objects (SPD-Conv).* ECML PKDD 2022.
13. Chen, L., Fu, Y., Gu, L., Yan, C., Harada, T., Huang, G. *Frequency-aware Feature Fusion for Dense Image Prediction (FreqFusion).* IEEE TPAMI 2024.
14. Huang, S., Lu, Z., Cheng, R., He, C. *FaPN: Feature-aligned Pyramid Network for Dense Image Prediction.* ICCV 2021.
15. Kervadec, H., Bouchtiba, J., Desrosiers, C., Granger, E., Dolz, J., Ben Ayed, I. *Boundary Loss for Highly Unbalanced Segmentation.* MIDL 2019 (bản mở rộng MedIA 2021).
16. Deng, D., Liu, H., Li, X., Cai, D. *PixelLink: Detecting Scene Text via Instance Segmentation.* AAAI 2018.
17. Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollár, P. *Focal Loss for Dense Object Detection.* ICCV 2017.
18. Shrivastava, A., Gupta, A., Girshick, R. *Training Region-based Object Detectors with Online Hard Example Mining.* CVPR 2016.
19. Mukhoti, J., Kulharia, V., Sanyal, A., Golodetz, S., Torr, P., Dokania, P. *Calibrating Deep Neural Networks using Focal Loss.* NeurIPS 2020.
20. Li, Z., Hoiem, D. *Learning without Forgetting.* ECCV 2016.
21. Hinton, G., Vinyals, O., Dean, J. *Distilling the Knowledge in a Neural Network.* NIPS Deep Learning Workshop 2015 (arXiv:1503.02531).
22. Chen, Y., Zhang, P., Li, Z., et al. *Stitcher: Feedback-driven Data Provider for Object Detection.* arXiv:2004.12432, 2020.
23. Wang, J., Xu, C., Yang, W., Yu, L. *A Normalized Gaussian Wasserstein Distance for Tiny Object Detection.* arXiv:2110.13389, 2021 (đã loại, xem §3).

---

## 9. Cập nhật 19/11/2026: thay hoặc bổ sung bằng paper 2024–2026

**Mức độ xác minh:** đã kiểm tra mỗi paper có tồn tại và đã đọc **abstract / trang hội nghị / README repo**. **Chưa đọc full-text.** Mọi chi tiết cài đặt phải lấy từ paper/code gốc trước khi code.

### 9.1. Nguyên tắc chọn: "mới hơn" không tự động "tốt hơn"

1. **Sự khớp cơ chế quan trọng hơn năm xuất bản.** Một phương pháp SOTA trên COCO/AI-TOD được tối ưu cho detector anchor/DETR. Nó chỉ có lợi cho DBNet khi giải đúng nút thắt của B1 (FP, lệch biên, Clean drop).
2. **Paper mới thường chứa paper cũ làm baseline.** Model Stock (2024) và DaWin (2025) đều so với WiSE-FT (2022). Reviewer sẽ yêu cầu chạy **cả bản cũ lẫn bản mới**: nếu bản mới không hơn WiSE-FT trên dữ liệu của nhóm, đó vẫn là một kết quả.
3. **Kỹ thuật cổ điển không có bản thay thế mới tốt hơn thì vẫn giữ.** Hysteresis (Canny/CRAFT) là ví dụ. Không tìm thấy công trình 2024–2026 nào thay thế nó cho heatmap dạng DB.
4. **Rủi ro mới:** paper mới ít được tái lập độc lập, và code thường gắn chặt với framework (mmdet/mmrotate). Chi phí chuyển sang MMOCR cao hơn.

### 9.2. Bảng ánh xạ bổ sung (2024–2026)

| # | Paper (đã xác minh tồn tại) | Bài toán gốc → cơ chế | Tương đồng với B1/LR2 | Nút thắt | Công thức | Độ khó | Fit |
|---|---|---|---|---|---|---|---|
| N1 | **Jang, Yun, Han. Model Stock. ECCV 2024** ([arXiv 2403.19522](https://arxiv.org/abs/2403.19522)) | Robust fine-tuning CLIP. Từ **pretrained anchor + 2 model fine-tune**, chọn điểm gần tâm hình học của vùng fine-tune, theo từng layer. Báo cáo tốt hơn đường WiSE-FT. | Nhóm có đúng cấu hình đó: anchor **B0**, hai model fine-tune **B-Clean** và **B1** cùng init. Bản hiện tại của Công thức 1 chỉ nội suy B-Clean↔B1 mà bỏ qua anchor. | Clean drop + FP | 1 | Rất thấp, không train | **Rất cao** |
| N2 | **Oh, Li, Song, Yun, Han. DaWin. ICLR 2025** ([arXiv 2410.03782](https://arxiv.org/abs/2410.03782)) | Nội suy trọng số **động theo từng mẫu**, không train. Hệ số lấy từ entropy dự đoán của mỗi model trên mẫu đó. | α nên khác nhau theo ảnh: ảnh nhiều chữ nhỏ cần B1, ảnh chữ lớn hoặc sạch cần B-Clean. Bản dense do nhóm chuyển thể: α_img từ entropy nhị phân trung bình của P-map mỗi model trên vùng ứng viên. | FP + Clean drop | 1 | Thấp; 2 forward/ảnh | Cao |
| N3 | **Kuzucu, Oksuz, Sadeghi, Dokania. On Calibration of Object Detectors: Pitfalls, Evaluation and Baselines. ECCV 2024 (oral)** ([ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03148.pdf), [code](https://github.com/fiveai/detection_calibration)) | Hiệu chỉnh confidence theo **chất lượng định vị** (LaECE/LaACE). Baseline post-hoc: isotonic/linear regression. | Score DB = mean P không phản ánh IoU. Polygon lệch vẫn có score cao. Fit isotonic s → E[IoU \| s, h] trên split A, rồi lọc theo IoU kỳ vọng. Đây là **bản có căn cứ 2024** thay cho heuristic γ_c. | Geometry FP + chọn ngưỡng | 1 | Thấp, CPU | Cao |
| N4 | **Sun et al. SET: Spectral Enhancement for Tiny Object Detection. CVPR 2025** ([CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Sun_SET_Spectral_Enhancement_for_Tiny_Object_Detection_CVPR_2025_paper.html), [code](https://github.com/HuixinSun/SET)) | Tiny object (AI-TOD). **HBS** làm mượt nhiễu tần số cao ở **nền**. **API** chèn nhiễu đối kháng để tăng độ nổi của vùng đối tượng. **Chỉ dùng khi train, không tốn chi phí lúc suy luận.** Dùng bin very-tiny 2–8 px. | Gần nhất với nút thắt: B1 tăng nhạy với tín hiệu tần số cao (chữ nhỏ) nhưng kéo theo FP trên nền có texture. HBS nhắm đúng vào việc dập **tần số cao ở nền**. Bin 2–8 px tương ứng nhóm <4 và 4–8 px của nhóm. | **FP** + tiny Recall | 2 (thay FreqFusion-lite) | Trung bình; code mmdet cần port sang MMOCR FPNC | **Cao** |
| N5 | **Chen, Gu, Fu et al. FADC: Frequency-Adaptive Dilated Convolution. CVPR 2024 (Highlight)** ([CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Frequency-Adaptive_Dilated_Convolution_for_Semantic_Segmentation_CVPR_2024_paper.html)) | Segmentation. Dilation và kernel thay đổi theo tần số cục bộ. **FreqSelect** dập tần số cao ở nền. | Cùng nhóm tác giả FreqFusion. FreqSelect có thể dùng làm module nhẹ cho smooth-conv của FPNC. | FP + biên | 2 (dự phòng) | Trung bình | Trung bình |
| N6 | **LoftUp (2025)** · **JAFAR, NeurIPS 2025** ([arXiv 2506.11136](https://arxiv.org/abs/2506.11136)) · **AnyUp (arXiv 2510.12764)** · ViT-Up (arXiv 2606.14024, preprint 2026) | Upsample feature của ViT foundation model bằng attention hướng dẫn bởi ảnh độ phân giải cao, để biên khớp hơn. | Backbone là ViTAE nên nghe có vẻ hợp. Nhưng FPNC đã có C2 ở stride 4, và các upsampler này được train cho DINO/CLIP nên phải train lại. Lợi ích chưa rõ so với FreqFusion. | Geometry | 2 | Cao | **Thấp–trung bình**, chỉ là related work |
| N7 | **Chen, Dang, Yang, Chau, Wang. TextDS. ECCV 2026** ([arXiv 2606.28077](https://arxiv.org/abs/2606.28077)) | **Scene text detection dưới distribution shift + điều kiện chụp xấu.** Step-wise LoRA + Common Subspace Fusion, 4,9M tham số trainable. | **Cùng miền và cùng bài toán.** Gợi ý recipe thay cho full fine-tune: LoRA trên backbone, giữ B-Clean làm nền, để giảm "quên" Clean. Bắt buộc phải nằm trong Related Work. | Clean drop | 3 (biến thể PEFT) | Trung bình–cao | Cao cho related work, trung bình cho triển khai gấp |
| N8 | **Cao, Lyu, Zeng, Mu, Zhou. The Devil is in Fine-tuning and Long-tailed Problems. IJCAI 2025** ([arXiv 2505.15649](https://arxiv.org/abs/2505.15649)) | Benchmark STD. Fine-tune theo từng dataset làm tăng điểm miền đó nhưng giảm miền khác. Đề xuất Joint-Dataset Learning. | Ủng hộ cách diễn giải "B1 đánh đổi Clean" là hiện tượng đã biết trong STD. | Lập luận / Related Work | - | - | Cao (trích dẫn) |
| N9 | **S2Teacher (arXiv 2504.11111, 2025)**: Focal Ignore Loss cho sparsely-annotated detection | Vật thể không gán nhãn bị coi là nền, gây tín hiệu âm sai. Reweight loss để giảm phạt các vùng nghi là vật thể thiếu nhãn. | TextOCR có thể bỏ sót chữ nhỏ. Nếu review FP (§2.3) cho thấy nhiều "chữ không gán nhãn", Công thức 3 phải **giảm** phạt ở vùng đó thay vì tăng. | FP (đo đúng) | 3 | Thấp–trung bình | Có điều kiện (phụ thuộc D6) |
| N10 | **Sun et al. TOLF: Noise-Robust Tiny Object Localization with Flows. arXiv 2601.00617 (preprint 2026)** | Tiny object rất nhạy với **nhiễu annotation khi định vị**. Normalizing flow + **giảm gradient của mẫu có độ bất định cao**. | 1.209/1.453 regression là lỗi geometry. GT polygon của chữ <4 px có sai số tương đối lớn. Ý tưởng chuyển giao: giảm trọng số loss biên cho instance nhỏ có độ bất định cao. | Geometry | 3 | Trung bình (bản gốc cho box regression) | Trung bình; preprint |
| N11 | **Kirchhoff et al. Skeleton Recall Loss. ECCV 2024** ([arXiv 2404.03010](https://arxiv.org/abs/2404.03010)) | Segmentation cấu trúc mảnh dạng ống. Soft recall trên skeleton GT, chi phí thấp. | Chữ 2–4 px trên LR2 gần như là một đường mảnh. Recall trên trục giữa kernel có thể giữ tiny Recall khi tăng phạt FP. Fit chỉ một phần. | Giữ tiny Recall | 3 | Thấp | Trung bình–thấp |

### 9.3. Ba công thức sau khi cập nhật

- **Công thức 1 (zero-training), giữ hysteresis + τ(h), thay phần trộn trọng số và scoring:**
  1. **Trộn trọng số:** WiSE-FT (2022, baseline) → **Model Stock (ECCV 2024)** với anchor B0 → **DaWin (ICLR 2025)** cho α theo ảnh. Ba hàng ablation: A4a WiSE, A4b Model Stock, A4c DaWin.
  2. **Scoring:** heuristic γ_c → **localisation-aware isotonic calibration (Kuzucu et al., ECCV 2024)**: ŝ = g(s, h) ≈ E[IoU_best | s, h], fit trên split A, lọc theo ŝ ≥ τ_IoU.
- **Công thức 2 (plugin):** ưu tiên **SET-HBS (CVPR 2025)** áp lên đầu ra FPNC (chỉ hoạt động khi train, **không tốn chi phí suy luận**). FreqFusion-lite (TPAMI 2024) là phương án thứ hai cho lệch biên. FADC-FreqSelect là dự phòng.
- **Công thức 3 (recipe):** IB-BCE + focal-negative như cũ. Thêm **nhánh rẽ theo D6**: nếu FP chủ yếu là chữ không gán nhãn, dùng Focal-Ignore (S2Teacher 2025) thay cho background-KD. Biến thể LoRA kiểu TextDS (ECCV 2026) trên nền B-Clean chỉ chạy nếu còn GPU.

### 9.4. Bổ sung cho Codex spec (§7)

- Stage `cache`: thêm `model_stock` (anchor B0; hệ số theo từng layer theo paper Model Stock, **đọc code gốc trước khi cài**) và `dawin_pmap` (ensemble P-map, α theo ảnh từ entropy; ghi rõ là bản chuyển thể dense của nhóm).
- Stage `sweep`: thêm `--rescore isotonic_iou`. Fit `sklearn.isotonic.IsotonicRegression` trên candidate của split A (đặc trưng s theo từng bin h; nhãn = best GT IoU), áp lên split B.
