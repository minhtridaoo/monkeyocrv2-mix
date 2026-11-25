# Sau Mix A v1: chẩn đoán, research gap và thiết kế v2 (CHỈ THIẾT KẾ, chưa code notebook, chưa chạy GPU) · 21/11/2026

**Phạm vi:** Text Detection, Clean và native LR2, chỉ dữ liệu development (remaining-validation A/B và TRAIN). **Không dùng TEST.** Mix A v1 cùng các file của nó giữ nguyên làm mốc kết quả âm (`docs/handoff/artifacts/janek/mix_a_v1/`, `docs/research/janek/report_mix_a_v1_janek.md`). Báo cáo cũ không sửa.

**Nhãn số liệu:** mọi số trên split B đều là **khám phá** vì B đã được xem từ v1. Số liệu chỉ được dùng để xếp hạng hướng đi, không dùng để tuyên bố cải tiến.

**Cập nhật 22/11 (sau review 6 điểm và P0 sửa lỗi; user cho phép sửa):** chỗ đã sửa được đánh dấu *[22/11]*. Số liệu cũ giữ nguyên; thêm E13–E16, mềm hóa các câu quá mạnh, sửa logic về kích thước, chuyển balanced accuracy từ cổng thành chẩn đoán, ghi biên trên A. Các ngưỡng +0,5 pp / −3 pp / −0,3 pp **không đổi**. Bản gốc 21/11 còn trong lịch sử WORKLOG và HANDOFF.

**Về các skill được yêu cầu:**
- Đã áp dụng checklist Stage 2.5 bằng cách đọc trực tiếp `academic-pipeline/references/ai_research_failure_modes.md`.
- `research-ideation`, `model-architecture`, `mechanistic-interpretability` **là tên nhóm, không phải tên skill** (theo `LABB/workspace/Chỉ dẫn master prompt gửi gemini.md`, Nhóm C). Bản 21/11 ghi "không có trong phiên" là sai tên tìm; skill con có sẵn (ánh xạ và mức đã đọc/áp dụng ở `docs/research/janek/audit_v2_vs_framework_20261122_janek.md` §3–4) nhưng **chưa được kích hoạt** khi viết tài liệu này.
- `academic-pipeline` và `/ars-plan` là quy trình viết paper, không hợp với bước thiết kế thí nghiệm này, nên không chạy cả pipeline.
- Khung hướng dẫn: đường dẫn `LABB/workspace/h_ng_d_n_th_c_chi_n_quy_tr_nh_nghi_n_c_u_ai_ocr.md` không tồn tại. `LABB/workspace/Step by step research.md` có nội dung phù hợp nhưng **chưa xác nhận là file thay thế**. Bản thiết kế này được viết trước khi đối chiếu; kết quả đối chiếu (22/11) ở `docs/research/janek/audit_v2_vs_framework_20261122_janek.md`.

---

## 1. Chẩn đoán vì sao Mix A v1 không vượt baseline

### 1.1 Bằng chứng

| Mã | Quan sát | Nguồn |
|---|---|---|
| E1 | LR2 Hmean theo α (B, setting R1): 74,59 → 75,81 → 76,28 → 76,69 → 76,69. Cao hơn đường thẳng nối hai endpoint **+0,52 đến +0,69 pp**, và **không vượt B1**. | `docs/handoff/artifacts/janek/diag_v2_cpu/partial_from_csv_janek.json` |
| E2 | Clean Hmean theo α: 81,16 → 80,87 → 80,27 → 79,99 → 79,52. Lệch khỏi đường thẳng ≤ 0,13 pp, tức **giảm gần tuyến tính**. | như trên |
| E3 | Recall LR2 <4 px theo α: 16,1 → 21,6 → 32,5 → 34,6 → 37,6%. Không α nào ở giữa vượt B1. | như trên |
| E4 | FP LR2 tăng đơn điệu theo α (3.211 → 5.456), và recall LR2 cũng tăng theo (64,2 → 70,5%). **Ở cùng recall**, α ¾ có nhiều FP hơn B1 6–7%. | `docs/handoff/artifacts/janek/explore_sel_v1/NOTES_janek.md` |
| E5 | Cổng chọn: B-Clean thắng trên A với khoảng cách 0,21 pp mean, nhưng thua B1 0,24 pp trên B. Mọi CI của α ở giữa so với từng baseline đều chứa 0. | `docs/handoff/artifacts/janek/mix_a_v1/NOTES_janek.md` |
| E6 | BN: thống kê của α ở giữa ≈ nội suy endpoint (≤ 0,005 SD). Trong cách đo đó, không thấy sai lệch riêng do trộn. | `docs/handoff/artifacts/janek/bn_stage0_v1/NOTES_janek.md` |
| E7 | Hai endpoint rất gần nhau về trọng số (‖ΔW‖/‖W‖ = 1,9%), cùng khởi tạo B0, cùng TRAIN, cùng 10k update. Chỉ khác tỉ lệ LR2 (0% so với 50%). | `docs/research/janek/design_bn_recalibration_pilot_janek.md` §2; `MONKEY_V2/docs/handoff/docs/handoff/artifacts/janek/bclean_report.md` |
| E8 | Định tuyến theo điều kiện (B-Clean cho ảnh Clean, B1 cho ảnh LR2, mỗi model giữ setting R1): mean B **78,93** (Clean 81,16, LR2 76,69, tiny 37,6%). Cao hơn B-Clean +1,06 pp, cao hơn B1 +0,82 pp, và **cao hơn α tốt nhất (78,34) +0,58 pp** *[22/11: số trên B, khám phá; trên A cùng phép tính chỉ +0,27 pp, xem E14]*. | `partial_from_csv_janek.json` |
| E9 | Theo EDA của Tuấn (common `.30/.40`), từ B-Clean sang B1 trên LR2: B1 cứu thêm **6.150** GT nhưng làm mất **1.453** GT mà B-Clean bắt được. Trên Clean: cứu 1.536, mất 1.651. | `MONKEY_V2/docs/handoff/docs/handoff/artifacts/janek/paired_adaptation_eda_report.md` |
| E10 | Lỗi còn lại của B1 trên LR2 (ảnh Clean bắt được nhưng LR2 bỏ sót, 6.454 GT): **geometry 3.280, weak response 2.791** (c0 = .30). | `b1_cal_diagnostics.md` |
| E11 | EDSR ×2 trước B1: LR2 +0,60 pp (CI [0,24; 0,96]), tiny 45,0 → 48,9%, nhưng **thêm 952 FP**, latency 137 → 205 ms. Bicubic ×2 không giúp (−0,05 pp). Chưa đo EDSR trên Clean. | `restoration_report.md`, `MONKEY_V2/handoff.md` |
| E12 | PRC −0,02 pp. Latent pilot 1k update: chênh lệch adapter/đối chứng "gần 0", chưa chứng minh hiệu quả. Bản 10k chưa có kết quả. | `prc_report.md`, `docs/latent-10k-extension.md` |
| E13 *[22/11]* | Cạnh dài trên A+B: Clean 768–1024, LR2 384–512 (LR2 suy ra bằng chia đôi, **chưa đọc file LR2**), không chồng lấn. Router kích thước T = 768 cho mean A 78,78 và B 78,93, bằng oracle điều kiện. Chỉ đúng cho cohort và quy trình tạo LR2 này. | `docs/handoff/artifacts/janek/diag_v2_cpu/p0_fixed_janek.json`, `diagnosis_janek.json` |
| E14 *[22/11]* | **Biên trên A:** model đơn tốt nhất B-Clean 78,51; oracle điều kiện 78,78 = **+0,27 pp**, dưới mục tiêu +0,5 pp. Trên B: +0,58 pp so với α¼ (khám phá). Bootstrap theo nguồn trên B, oracle − B-Clean: +1,07 pp, CI95 [0,21; 1,89]. Setting của model đơn và của oracle đều chọn trên A, nên số trên A là in-sample. | `p0_fixed_janek.json` |
| E15 *[22/11]* | Đối chứng hậu xử lý theo điều kiện (một detector): trên A không thêm gì so với B-Clean (78,51); trên B tốt nhất là α¾ 78,49, oracle chỉ hơn +0,43 pp. | `p0_fixed_janek.json` |
| E16 *[22/11]* | Frontier trên A: ở LR2, 13–17/72 setting của mỗi α (¼: 13, ½: 17, ¾: 14) không bị endpoint nào áp đảo về (recall, precision); Clean 1–5/72. Theo bin trên B: LR2 bin < 4 px tại α½ cao hơn đường thẳng +5,65 pp, Clean −1,63 pp (bin nhỏ, chưa có CI). Biên độ vượt frontier chưa đo. | `diagnosis_janek.json` |

**Lưu ý đo lường cho E1–E3:** mỗi model dùng setting tự chọn trên A, nên đường cong trộn lẫn hiệu ứng của model và của setting.

### 1.2 Kết luận chẩn đoán (suy luận có bằng chứng, chưa phải định lý)

1. **Nguyên nhân chính là bản chất của đánh đổi, không phải lỗi cài đặt.**
   - B-Clean và B1 là hai **chuyên gia theo điều kiện**: B-Clean tốt nhất trên Clean, B1 tốt nhất trên LR2 (E1, E2, E8).
   - Trộn trọng số cho ra một model *duy nhất* nằm trên đường nối hai chuyên gia: Clean giảm gần tuyến tính, LR2 chỉ lồi nhẹ (+0,5–0,7 pp so với đường thẳng) và **không vượt được endpoint mạnh hơn trên LR2** (theo Hmean). *[22/11: theo E16, 'đi dọc đường nối' chỉ là xấp xỉ theo Hmean; về frontier (recall, precision) trộn có mở rộng nhẹ ở LR2, biên độ chưa đo.]*
   - Trung bình hai điều kiện vì vậy chỉ cao hơn endpoint tốt nhất ≤ 0,23 pp. Ngưỡng +0,5 pp so với *cả hai* baseline gần như không thể đạt chỉ bằng độ lồi cỡ này.
2. **Giả định của WiSE-FT không khớp bối cảnh này.**
   - WiSE-FT trộn một model zero-shot có độ bền rộng với một model fine-tune chuyên biệt, và lợi ích chính là độ bền dưới distribution shift ([Wortsman et al., CVPR 2022](https://arxiv.org/abs/2109.01903); đã đọc abstract).
   - Ở đây cả hai endpoint đều là fine-tune từ cùng B0 trên cùng TRAIN (E7). B-Clean không phải "model bền hơn" mà là chuyên gia Clean.
   - E7 (‖ΔW‖/‖W‖ = 1,9%, cùng khởi tạo) cho thấy hai endpoint gần nhau về trọng số; nó **không chứng minh** rằng không có độ bền bổ sung để phép trộn khai thác. "Lệch giả định" là **giả thuyết**, chưa kiểm. *[22/11: mềm hóa; trước đó ghi như sự thật.]*
3. **Chưa thấy bằng chứng cho thấy BN hay hậu xử lý là nguyên nhân chính** (chưa loại trừ hẳn). *[22/11: bản 21/11 ghi "không phải", quá mạnh.]*
   - E6 không thấy "lệch BN do trộn" trong cách đo đã dùng; chưa phải bằng chứng loại trừ.
   - E-sel cho thấy đổi cách chọn hậu xử lý không tạo ra ứng viên đạt (E4).
   - FP thấp hơn chỉ đi kèm recall thấp hơn; cùng recall thì không có lợi về FP.
4. **Có xung đột mục tiêu tối ưu, nhưng không quyết định kết quả.** Hàm chọn của R1 (tổng Hmean) không thưởng tiny, nên B-Clean thắng trên A ở setting có tiny 16% (E5). Sửa điểm này (E-sel) vẫn không ra ứng viên đạt, nên đây không phải nguyên nhân gốc.
5. **Cơ hội có thể nằm ở cấp hệ thống:** dùng mỗi chuyên gia đúng nơi nó mạnh (E8, E9). *[22/11]* Độ lớn hẹp: trần oracle điều kiện trên A chỉ +0,27 pp (E14), dưới +0,5 pp; trên B là +0,58 pp (khám phá). Đổi lại phải giải được hai bài toán:
   - (a) biết ảnh thuộc chế độ nào mà **không** dựa vào nhãn điều kiện;
   - (b) trong cùng một ảnh LR2, kết hợp phần bổ sung của hai chuyên gia (1.453 GT chỉ B-Clean bắt được) mà không cộng dồn FP của cả hai.

### 1.3 Cập nhật của Tuấn liên quan

- **Danh sách TRAIN 15.222 nguồn đã công bố** (`e4a2a89`). Dùng được để huấn luyện hoặc hiệu chỉnh router mà **không động tới A/B**.
- **Latent A0/A1/A2:** bản 1k gần 0; bản 10k đang chờ (E12).
  - Hướng can thiệp sau FPNC hiện chưa có tín hiệu.
  - Lưu ý: latent khởi tạo từ B1 và train 50/50, nên nó không giải quyết việc Clean tụt so với B-Clean (E2). Đó là đánh đổi mà đề xuất dưới đây nhắm tới.
- **CRAFT screen:** chưa thấy kết quả trong repo.

### 1.4 Research gap (giới hạn kiểm chứng: đọc abstract hoặc kết quả tìm kiếm, chưa đọc toàn văn)

- Có định tuyến expert theo độ phân giải hoặc mức suy giảm ở các bài toán lân cận: nhận diện khuôn mặt độ phân giải thấp (FaceMoE, [arXiv 2606.32040](https://arxiv.org/html/2606.32040)), phát hiện vật thể trong thời tiết xấu (C2FXNet, [arXiv 2609.25693](https://arxiv.org/pdf/2609.25693)), siêu phân giải có định tuyến theo mức suy giảm ([arXiv 2511.16024](https://arxiv.org/html/2511.16024)). Mình **chưa tìm thấy** công trình định tuyến chuyên gia theo độ phân giải hoặc chất lượng cho **phát hiện chữ**. "Chưa tìm thấy" không có nghĩa là chắc chắn không có.
- Trộn trọng số bị nhiễu khi các task vector xung đột ([TIES-Merging, NeurIPS 2023](https://arxiv.org/abs/2306.01708)). Trộn tốt hơn chưa chắc vượt được việc giữ riêng hai chuyên gia khi chúng chuyên biệt theo hai miền tách biệt.
- Ensemble box (WBF, [Solovyev et al., IVC 2021](https://arxiv.org/abs/1910.13302)) là kỹ thuật đã có. Điểm mới, nếu có, chỉ nằm ở **cách kết hợp bất đối xứng theo điều kiện LR** và ở **đối chứng shortcut**, không nằm ở WBF.
- **Gap đề xuất, cần kiểm chứng:** với các chuyên gia tách theo điều kiện chất lượng ảnh, liệu *định tuyến theo chất lượng không dùng kích thước* cộng với *kết hợp bất đối xứng trong ảnh LR* có thắng trộn trọng số và thắng từng chuyên gia trên cả Clean, LR2 và tiny, với chi phí suy luận gần một model hay không.

---

## 2. Ba thiết kế

### Thiết kế 1 · QRoute: định tuyến chuyên gia theo chất lượng ảnh, không dùng kích thước (Trục 1)

**(1) Vấn đề và bằng chứng.**
- Một model đơn không giữ được đồng thời Clean của B-Clean và LR2/tiny của B1 (E1–E3).
- Định tuyến theo điều kiện đạt mean 78,93, hơn cả hai baseline (+1,06 và +0,82 pp) và hơn α tốt nhất +0,58 pp (E8). Tiny giữ đúng mức B1 (37,6%), Clean giữ đúng mức B-Clean. *[22/11]* Trên A cùng phép tính chỉ +0,27 pp (E14): router theo điều kiện, dù hoàn hảo, **không thể đạt cổng +0,5 pp trên A**.
- Đây mới là **cận tham chiếu** vì nó giả định biết điều kiện của ảnh.

**(2) Đóng góp của từng thành phần.**
- **E_c = B-Clean** và **E_l = B1**, giữ nguyên; mỗi chuyên gia có hậu xử lý riêng chọn trên A cho đúng phần ảnh được giao.
- **Router q(x)** quyết định dùng chuyên gia nào:
  - Đặc trưng tính trên ảnh **đã resize về cạnh dài cố định** (ví dụ 1024) để xóa tín hiệu kích thước.
  - Đặc trưng gồm: tỉ lệ năng lượng cao tần ‖x − G_σ∗x‖²/‖x‖² ở 2–3 mức σ, phương sai Laplacian, và năng lượng phổ ngoài dải.
  - Gom thành vector k ≈ 6; dùng logistic regression, hoặc ngưỡng 1 chiều nếu đủ tốt.
- **Luồng xử lý:** ảnh x (H×W×3) → q(x) ∈ [0, 1] → nếu q > τ thì chạy E_l, ngược lại E_c (một lần forward) → P-map (H′×W′) → DBPostprocessor (mask, score) của chuyên gia đó → box.
- **Chi phí suy luận:** một forward detector cộng router CPU (ước lượng vài ms, chưa đo). Bộ nhớ chứa hai checkpoint.

**(3) Kế thừa và điều chỉnh.**
- **Kế thừa:** hai checkpoint, lưới hậu xử lý, evaluator, R1 và tiêu chí của Tuấn.
- **Điều chỉnh:** router size-free và hậu xử lý theo từng chuyên gia.
- **Điểm mới dự kiến** nằm ở giao thức hơn là kiến trúc: chứng minh có đối chứng rằng *định tuyến chuyên gia thắng trộn trọng số* cho đánh đổi Clean/LR trong phát hiện chữ, với router không khai thác shortcut kích thước.
- **Nói thẳng:** đây là phương pháp cấp hệ thống, đơn giản; giá trị khoa học phụ thuộc vào phần đối chứng.

**(4) Ablation, chi phí, điều kiện dừng.**
- **Ablation:**
  - router oracle (theo nhãn), router kích thước (đối chứng shortcut), router size-free;
  - hậu xử lý riêng từng chuyên gia so với dùng chung;
  - so với B1, B-Clean và α ¾;
  - mức phạt khi định tuyến sai, tính từ counts: ảnh Clean bị gửi sang B1 và ngược lại.
- **Chi phí:** pilot hoàn toàn **CPU**.
  - Routing tính trực tiếp từ counts A2r đã có.
  - Đặc trưng router trên 4.524 ảnh A/B và 2.000 ảnh TRAIN (cache có sẵn trên Drive): ước lượng 10–20 phút CPU Colab (chưa đo).
  - **Không cần GPU T4/A100.** Chỉ khi pilot pass mới cần GPU để đo latency thật (khoảng 15 phút).
- **Kill (đề xuất, cần Tuấn duyệt):** dừng nếu
  - *[22/11: bỏ điều kiện "balanced accuracy < 95%"; chỉ báo như chẩn đoán, vì độ chính xác định tuyến chỉ quan trọng qua tác động lên Hmean]*
  - routed mean Hmean trên A < max(B-Clean, B1, α tốt nhất) + 0,5 pp; **hoặc**
  - tiny LR2 < B1 − 3 pp; **hoặc**
  - Clean thấp hơn B-Clean quá 0,3 pp.

### Thiết kế 2 · DualAgree: kết hợp bất đối xứng hai chuyên gia trên ảnh LR2 (Trục 1, thêm quy tắc kết hợp)

**(1) Vấn đề và bằng chứng.**
- Ngay trên LR2, B1 làm mất 1.453 GT mà B-Clean bắt được, và có nhiều FP hơn (E4, E9).
- Thiết kế 1 chỉ chọn một chuyên gia cho mỗi ảnh nên không lấy được phần bù này.

**(2) Đóng góp.** Trên ảnh được định tuyến là LR, chạy **cả hai** chuyên gia và kết hợp ở mức box:
- Box hai bên trùng nhau (IoU ≥ 0,5): gộp bằng WBF.
- Box chỉ B1 có: giữ nếu score ≥ τ_l, ngưỡng cao hơn để cắt FP.
- Box chỉ B-Clean có: giữ nếu score ≥ τ_c, để lấy lại phần GT bị mất.

Ba tham số (τ_l, τ_c, IoU) chọn trên A. Phương án thay thế ở mức P-map là P = max(P_l, P_c) hoặc w·P_l + (1 − w)·P_c, rồi qua cùng DBPostprocessor. Ảnh Clean vẫn đi qua E_c (kế thừa Thiết kế 1).

**(3) Kế thừa và điều chỉnh.**
- **Kế thừa:** WBF (đã có), hai checkpoint, Thiết kế 1.
- **Điều chỉnh:** quy tắc bất đối xứng theo vai trò của từng chuyên gia.
- **Điểm mới hạn chế:** biến thể ensemble có động lực từ số liệu E9.

**(4) Ablation, chi phí, kill.**
- **Ablation:** chỉ phần trùng; trùng + B1-only; trùng + B1-only + B-Clean-only; kết hợp ở P-map so với ở box; so với B1 ở cùng recall (FP ở recall tương đương).
- **Chi phí:**
  - Cần một lần chạy GPU lưu box và score (không lưu P-map) của hai chuyên gia trên 2.262 ảnh LR2: khoảng 2 × 2.262 × 0,4 s ≈ 30 phút A100, hoặc khoảng 1 giờ T4 (ngoại suy từ 0,41–0,47 s/task, chưa đo cho tác vụ này).
  - Phần quét quy tắc chạy trên CPU.
  - Suy luận trên ảnh LR tốn gấp 2 lần.
- **Kill:** dừng nếu trên A, LR2 Hmean ≤ B1 + 0,3 pp, **hoặc** FP ở cùng recall không giảm, **hoặc** tiny < B1 − 3 pp.

### Thiết kế 3 · SRVerify: góc nhìn siêu phân giải cộng kiểm chứng nhất quán đa tỉ lệ (Trục 2)

**(1) Vấn đề và bằng chứng.** EDSR ×2 tăng recall và tiny nhưng thêm FP (E11). Một nửa lỗi còn lại của B1 là geometry hoặc weak response (E10).

**(2) Đóng góp.** Trên ảnh LR, chạy B1 cả trên ảnh native và trên ảnh EDSR ×2:
- Box có ở cả hai góc nhìn: giữ.
- Box chỉ có ở góc nhìn SR: giữ nếu P-map native trong box đạt mean ≥ γ, tức có "phản hồi yếu nhưng có thật".

Ảnh Clean bỏ qua SR nhờ router của Thiết kế 1.

**(3) Kế thừa và điều chỉnh.**
- **Kế thừa:** EDSR pretrained (torchSR v1.0.3, Tuấn đã dùng), B1.
- **Điều chỉnh:** kiểm chứng chéo tỉ lệ để cắt FP do SR sinh ra. Điểm mới ở mức trung bình.

**(4) Ablation, chi phí, kill.**
- **Ablation:** native; SR; hợp hai góc nhìn; giao hai góc nhìn; giao + kiểm chứng γ.
- **Chi phí:** dựng lại EDSR trên Colab (rủi ro kỹ thuật); 2 lần forward cộng SR trên 2.262 ảnh LR2, khoảng 1–1,5 giờ A100 (chưa đo). Latency theo đo của Tuấn: khoảng 205 so với 137 ms.
- **Kill:** dừng nếu trên A, kết quả không vượt EDSR-only ở cùng recall với FP thấp hơn, **hoặc** LR2 ≤ B1 + 0,3 pp.

### Bảng so sánh

| Tiêu chí | 1 · QRoute | 2 · DualAgree | 3 · SRVerify |
|---|---|---|---|
| Nhắm vào | Đánh đổi Clean ↔ LR2 | FP và GT bị mất trong ảnh LR2 | Weak response và tiny trong LR2 |
| Bằng chứng hiện có | Mạnh: cận theo điều kiện +0,58 đến +1,06 pp (E8) | Trung bình: phần bù 1.453 GT (E9); chưa biết mức trùng của FP | Trung bình: EDSR +0,60 pp nhưng +952 FP (E11) |
| GPU cho pilot | **Không** (CPU, counts có sẵn) | Khoảng 30 phút A100 / khoảng 1 giờ T4 | Khoảng 1–1,5 giờ A100, cộng dựng EDSR |
| Chi phí suy luận | Khoảng 1× (cộng router) | 2× trên ảnh LR | Khoảng 2× + SR |
| Rủi ro chính | Shortcut kích thước (Mode 4); "đơn giản quá" | Ngưỡng quá khớp A; FP cộng dồn | Rủi ro kỹ thuật EDSR; Clean chưa đo |
| Phụ thuộc | Không | *[22/11]* Pilot độc lập trên ảnh LR đã biết điều kiện; hệ thống đầy đủ cần biết ảnh nào là LR | như Thiết kế 2 |

**Chọn Thiết kế 1** để pilot trước (quyết định 21/11). Lý do khi đó: bằng chứng mạnh nhất, chi phí gần 0, kill sớm rõ ràng. *[22/11]* Sau P0 sửa lỗi, "bằng chứng mạnh nhất" chỉ còn đúng trên B (khám phá); trên A trần của định tuyến theo điều kiện là +0,27 pp (E14). Có tiếp tục Thiết kế 1 hay không do user quyết định (HANDOFF).

*[22/11]* Thiết kế 2 và 3 **không phụ thuộc** vào việc router của Thiết kế 1 sống hay chết khi pilot: cả hai chạy trên ảnh LR đã biết điều kiện (Thiết kế 2 chạy hai chuyên gia trên LR; Thiết kế 3 chạy B1 trên native và SR). Router chỉ cần khi triển khai hệ thống đầy đủ, để biết ảnh nào đi nhánh LR; nếu không có router thì phải chạy nhánh LR cho mọi ảnh, kể cả Clean (chi phí gấp 2). Câu "mất cơ sở định tuyến" ở bản 21/11 quá mạnh.

---

## 3. Pilot cho Thiết kế 1 (CPU, chỉ dữ liệu development)

**Dữ liệu:**
- **TRAIN:** 1.000 nguồn của pilot BN, đã kiểm SHA, Clean và LR2 đã có cache trên Drive. Dùng để **fit router**: Clean = 0, LR2 = 1, sau khi resize về cạnh dài cố định.
- **A (1.130 nguồn):** chọn ngưỡng τ và hậu xử lý theo từng chuyên gia.
- **B (1.132 nguồn):** báo cáo khám phá, vì B đã bị xem.
- **TEST:** không dùng.

**Các bước tối thiểu:**
- **P0** *[22/11: đã chạy]*: `scripts/diagnose_tradeoff_v2_janek.py` (kết quả `diagnosis_janek.json`). Q4 cũ **rò nhãn**: chọn setting theo (expert × điều kiện thật) (`diagnose_tradeoff_v2_janek.py:122-138`, đã ghi KNOWN FLAW). Bản sửa `scripts/p0_fixed_router_eval_janek.py` khóa **một setting mỗi expert** trên A (`p0_fixed_janek.json`); số đổi 0,001 pp. Kết quả ở E13–E16.
  - Q1: frontier precision/recall của α so với endpoint trên A.
  - Q4: router theo **kích thước**. Đây là đối chứng shortcut, không phải phương pháp.
  - Lấy routed mean trên A/B cho oracle và router kích thước.
- **P1:** tính đặc trưng router size-free cho 2.000 ảnh TRAIN và 4.524 ảnh A/B (CPU). Có hai rủi ro cần kiểm:
  - LR2 phải được tạo bằng `native_lr2` thật (Pillow 11.3.0).
  - Ảnh JPEG gốc so với LR2 PNG có thể để lại dấu vết nén. Đó là một shortcut khác (Mode 4), nên phải thêm một đối chứng: nén lại cả hai về cùng chuẩn JPEG trước khi tính đặc trưng.
- **P2:** fit router trên TRAIN, chọn τ trên A. Báo balanced accuracy trên A/B và đường ROC.
- **P3:** routed counts trên A/B từ counts A2r (định tuyến theo từng ảnh), với hậu xử lý theo từng chuyên gia chọn trên A. Báo Clean, LR2, mean, FP ở cùng recall, tiny, và CI paired so với B-Clean, B1, α ¾.
- **P4:** phân tích lỗi định tuyến: những ảnh nào bị gửi sai (kích thước gốc nhỏ, ảnh vốn mờ), và mỗi lần gửi sai tốn bao nhiêu Hmean.

**Pass/Fail (đề xuất, cần Tuấn duyệt trước khi chạy P1):**
- **PASS** khi đủ cả **ba** điều kiện trên A *[22/11: bỏ điều kiện balanced accuracy khỏi cổng, chuyển thành chẩn đoán]*:
  1. routed mean ≥ max(B-Clean, B1, α ¾) + 0,5 pp;
  2. tiny LR2 ≥ B1 − 3 pp;
  3. Clean ≥ B-Clean − 0,3 pp.

  Ba điều kiện phải đúng với router dùng đặc trưng đã qua **cả hai đối chứng shortcut** (resize cạnh dài cố định và nén chuẩn hóa), tính **với lỗi định tuyến thật**. Balanced accuracy và ROC (kích thước, nén) chỉ **báo cáo như chẩn đoán**, không phải cổng: độ chính xác định tuyến chỉ quan trọng qua tác động lên Hmean.

  *[22/11] Lưu ý:* oracle điều kiện trên A chỉ +0,27 pp (E14) nên điều kiện 1 **không thể đạt** bằng router theo điều kiện; chỉ đạt được nếu định tuyến theo từng ảnh vượt oracle điều kiện (chưa tính). **Không hạ ngưỡng sau khi thấy A.** Muốn đổi cổng phải ghi rõ là đổi sau khi thấy số và hỏi Tuấn.

  Trên B (khám phá), kết quả phải cùng chiều. Ngưỡng **không chỉnh sau khi xem kết quả**.
- **FAIL** ở bất kỳ điều kiện nào thì dừng Thiết kế 1, ghi kết quả âm, và không mở Thiết kế 2/3 dựa trên router này.
- **Kể cả khi PASS:** đó chỉ là "đáng xác nhận". Vòng xác nhận cần dữ liệu mới do nhóm chọn, chẳng hạn Bước 8 của Tuấn hoặc một tập chưa dùng. Chia lại A/B không tạo ra tập kiểm chứng mới.

**Câu hỏi cần chốt trước khi lập trình:**
1. Bài toán của nhóm có cho phép **hệ thống 2 checkpoint** (bộ nhớ gấp 2, mỗi ảnh vẫn một forward) không, hay bắt buộc **một model duy nhất**? Nếu bắt buộc một model, Thiết kế 1 không hợp lệ (đây là Mode 7, khóa khung sớm, cần hỏi lại).
2. Trong đánh giá cuối, ảnh "LR" có luôn là native LR2 (nửa kích thước) không? *[22/11: sửa logic]* Trên cohort A+B hiện tại, cạnh dài Clean 768–1024 và LR2 384–512 (LR2 suy ra bằng chia đôi, chưa đọc file) nên router kích thước tách được (E13). Điều này chỉ đúng cho quy trình tạo LR2 này; không thể suy từ "nửa kích thước" ra rằng kích thước luôn tách được. Nếu đánh giá cuối dùng LR theo cách khác (kích thước hoặc nén khác), router kích thước sẽ mất hiệu lực và một router không dùng kích thước mới là phần có giá trị. Vì vậy router size-free và đối chứng nén vẫn là bắt buộc làm chẩn đoán. Tuấn có đồng ý dùng chúng làm cổng không?
3. Ngưỡng ở mục Pass/Fail (95%, +0,5 pp, −3 pp, −0,3 pp) có hợp với Tuấn không?
4. Có nên chờ kết quả latent 10k trước, hay pilot Thiết kế 1 song song? Pilot này chỉ dùng CPU, không tranh GPU.

---

## 4. Integrity Gate (Stage 2.5, 7 mode) cho tài liệu này

| Mode | Kết quả | Bằng chứng / việc cần làm |
|---|---|---|
| 1 · Bug lọt qua tự kiểm | **CLEAR** cho E1–E8: tính từ `evaluation_B_janek.csv` đã lưu, script và JSON kèm theo. *[22/11]* **CLEAR (giới hạn)** cho E13–E16: P0 đã chạy; phát hiện và sửa rò nhãn ở Q4 cũ (`diagnose_tradeoff_v2_janek.py:122-138`; số đổi 0,001 pp); số trên A là in-sample; chưa tính oracle theo từng ảnh (trần thật của router); kích thước LR2 suy ra, chưa đọc file; 70/70 test chỉ xác nhận phần mềm. | `partial_from_csv_janek.json`, `p0_fixed_janek.json`, `diagnosis_janek.json` |
| 2 · Trích dẫn ảo | **CLEAR (giới hạn)**: WiSE-FT, TIES-Merging và WBF đã kiểm qua arXiv/kết quả tìm kiếm. Ba paper MoE chỉ thấy ở mức kết quả tìm kiếm, đã ghi rõ. | §1.4 |
| 3 · Kết quả thực nghiệm ảo | **CLEAR**: không có số nào của Thiết kế 1–3 được báo như đã chạy. Mọi chi phí đều ghi "ước lượng, chưa đo". | §2 |
| 4 · Shortcut | *[22/11]* **CONFIRMED** cho router kích thước: trên cohort và quy trình tạo LR2 này, kích thước tách hoàn toàn hai điều kiện (E13). **SUSPECTED** cho khác biệt nén JPEG/PNG: chưa chạy (P1). Hai đối chứng nằm trong pilot (P0 đã làm phần kích thước, P1 còn phần nén). | §3 |
| 5 · Bug bị biến thành insight | **CLEAR**: "định tuyến thắng trộn" không phải phát hiện bất ngờ, vì chuyên gia mạnh trên miền của nó là điều có thể dự đoán. Tài liệu không gọi đây là phát hiện mới. | §1.2 |
| 6 · Bịa phương pháp | **CLEAR**: thiết kế viết ở thì đề xuất; mọi tham số pilot là đề xuất chờ duyệt. | §3 |
| 7 · Khóa khung sớm | **INSUFFICIENT EVIDENCE**: chưa biết nhóm có ràng buộc "một model" hay không. Nếu có, Thiết kế 1/2 sai khung. Đây là câu hỏi 1 ở §3. | §3 |
