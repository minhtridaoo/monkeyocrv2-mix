# Phase 0 cho ứng viên CS-B1 · kế hoạch (22/11/2026)

> **Trạng thái tài liệu:** chỉ là kế hoạch. Chưa chạy phân tích Phase 0 nào, chưa tạo thẻ, chưa chạy GPU mới. Mọi con số trong mục 1–2 đều đã có từ trước, có ghi nguồn. Mọi việc ở mục 3–7 là **đề xuất**.

---

## 1. Tên và định nghĩa phương pháp

**Tên dùng thống nhất:** **CS-B1 — Clean-Supplemented B1**

**Diễn giải:** *B1 được bổ sung box từ chuyên gia Clean.* B1 vẫn là detector chính. B-Clean chỉ thêm những box mà B1 không có.

**Cấu hình tham chiếu hiện tại:** `CS-B1@0.7`. Đây là cấu hình khám phá được chốt ngày 22/11 (`mixab/dualagree.py`, hằng `LOCKED`).

Trên mỗi ảnh LR2:
1. Chạy B1 (geometry `epsilon_005`: mask .30, unclip 1.5, min width 5, epsilon .005). Giữ box có raw score ≥ 0,5.
2. Chạy B-Clean (geometry `mask_040`: mask .40, unclip 1.5, min width 5, epsilon .010). Lấy box có raw score ≥ 0,7.
3. Ghép từng cặp (một-một) box B1 đã giữ và box B-Clean đã lấy, theo IoU giảm dần. Cặp nào có IoU ≥ 0,5 coi là cùng một chữ: giữ box B1 và bỏ box B-Clean. Box B1 có score < 0,5 không vào đầu ra và cũng không tham gia ghép. Vì vậy một box B-Clean chỉ trùng với box B1 dưới ngưỡng vẫn được thêm vào.
4. Box B-Clean không ghép được với box B1 nào sẽ được **thêm vào** đầu ra.
5. Ảnh Clean không đi qua CS-B1. Hệ thống dùng B-Clean cho Clean (xem mục 2.4 về cách biết ảnh nào là LR2).

**Tên này không ngụ ý:**
- không train thêm;
- không có kiến trúc mới;
- không có bước "verification" hay kiểm chứng P-map;
- không gộp tọa độ kiểu WBF.

**Khác DualAgree v1 (đã không đạt):** DualAgree v1 có thêm nhánh đồng thuận và chọn trên A ngưỡng riêng của B1 là **0,7**. Hệ quả là các box B1 đứng một mình có score 0,5–0,7 bị cắt, trong đó có nhiều chữ tiny (tiny −19,8 pp trên A). CS-B1 **không đụng tới đầu ra B1**: B1 giữ nguyên ngưỡng 0,5, chỉ có thêm box.

**Mức bằng chứng:** biến thể **post-hoc**. Nó là tổ hợp tốt nhất trên A trong số các tổ hợp qua luật tiny, chọn sau khi đã thấy kết quả quét. Tín hiệu dương nhưng nhỏ, **chưa phải cải tiến đã xác nhận**.

| Split | ΔLR2 Hmean so với B1 (`epsilon_005/0.50`) | CI95 bootstrap theo nguồn | TP | FP | Tiny <4 px |
|---|---|---|---|---|---|
| A (1.130 nguồn) | +0,177 pp | [+0,065; +0,304] | 27.628 → 27.876 | 6.680 → 6.910 | 48,8 → 49,5% |
| B (1.132 nguồn, khám phá) | +0,283 pp | [+0,168; +0,407] | 28.764 → 29.092 | 5.455 → 5.703 | 37,6 → 38,1% |

Nguồn số liệu: `results/dualagree_v1_T4_followup.json`. Nhãn: **PENDING_REFERENCE_CHECK** (dump chạy trên T4). **Chưa đạt** luật dừng +0,3 pp trên A, và chưa đạt cổng nhóm +0,5 pp mean (mean hệ thống trên A là +0,36 pp, phần lớn đến từ định tuyến).

---

## 2. Trạng thái mới nhất (đã kiểm ngày 22/11)

### 2.1 Cập nhật của Tuấn (`MONKEY_V2` fast-forward tới `677b811`, 22/11 16:50)
- **Latent 10k: âm.** A0/A1/A2 LR2 76,86–76,87% so với B1 76,90%; A2 − B1 = −0,037 pp, CI [−0,113; 0,050] (`handoff.md` §5).
- **Fusion FPNC (ASF/DySample), job 106090:** đang chạy, chưa có metric cuối (`handoff.md` §6).
- **Chẩn đoán ở `.40/.60`, dùng lại được cho Phase 0** (`docs/handoff/artifacts/research_20261122/fixed_diagnostics/`):
  - Proxy FP cho từng dự đoán (`review/fp_taxonomy_proxies.csv.gz`, 45.948 dòng). B1 trên LR2 có 11.997 FP, chia thành: không chồng GT hợp lệ 5.811 · chồng GT, sai định vị hoặc chỉ phủ một phần 4.903 · nghi gộp 1.282 · trùng/gán 1 (`review/diagnosis_summary.json`).
  - Chẩn đoán từng GT, kèm độ phủ kernel P-map > 0,40 (`review/gt_diagnosis.csv.gz`). Riêng GT tiny <4 px trên LR2 mà B1 bỏ sót ổn định: weak_response 1.794 · geometry 581 · score_and_geometry 136 · no_candidate 129 · score_filtering 111.
  - 127 thẻ cặp ở `.40/.60`, **human review vẫn PENDING**.
- **Lưu ý quan trọng:** các chẩn đoán này ở operating point `.40/.60` của Tuấn (mask .40, score .60, epsilon .01), **khác** điểm vận hành của CS-B1 (B1 `epsilon_005/0.50`). Chỉ dùng làm bối cảnh hoặc đối chiếu, không thay cho phân tích tại điểm của CS-B1.
- Handoff của Tuấn ghi rằng nhánh hai checkpoint thuộc workflow cộng tác viên, phải báo đủ chi phí, và chưa được gộp vào kết luận main flow.

### 2.2 Phía mình
- Dump T4 xong: 36/36 shard mỗi model, 2.262 ảnh LR2 (`H:/My Drive/CS431_MixA/runs/dualagree_dump_v1_T4/`).
- Phân tích DualAgree v1 đã chạy trên Colab (`runs/dualagree_analysis_v1_T4/decision.json`).
- **Ô 8 (latency) chưa chạy:** trên Drive chưa có `runs/dualagree_latency_v1_T4/`. Zip đang ở `upload/` vẫn là bản 15:53, chưa có lệnh `latency`. Zip mới nằm local, SHA `343342a0…`.
- T4 tái lập tuyệt đối giữa hai lần chạy: 720/720 mục giống hệt (`results/t4_repeatability.json`).
- Tuấn đồng ý dùng hai checkpoint (lời user chuyển, chưa có văn bản gốc).

### 2.3 Bảng dữ liệu

| Hạng mục | Có / không | Chi tiết, nguồn |
|---|---|---|
| Polygon dự đoán | **Có** | Quad 8 tọa độ, ở hệ tọa độ ảnh Clean (đã nhân `restore`), B1 và B-Clean, đủ 9 geometry |
| Score | **Có** | Raw score (đúng số mà postprocessor dùng để so ngưỡng) và packed score (FP32) |
| Box điểm thấp | **Có, có giới hạn** | Lưu mọi box có raw ≥ 0,20. Thực tế score thấp nhất mỗi geometry ≈ ngưỡng mask (0,29–0,40), nên sàn chỉ có tác dụng ở `mask_020` |
| Box bị postprocessor loại trước khi lưu | **Không** | Box bị loại do min width (5, hoặc 3 ở `width_3`), unclip rỗng, hoặc không qua bước xấp xỉ contour đều **không** được lưu |
| P-map | **Không** | Dump không lưu P-map. Không suy diễn về P-map từ dump |
| Source ID, sample ID | **Có** | `sample_ids` trong shard; ánh xạ sang source qua `tasks_metadata.csv` của bundle |
| Kích thước ảnh | **Có** | `clean_size` trong GT manifest đã khóa; LR2 = floor(W/2), floor(H/2) |
| GT hợp lệ / ignore / ROI | **Có** | `gt_clean.valid`, `gt_clean.ignored`, `roi_clean` trong `validation_remaining.jsonl.gz` (SHA đã khóa) |
| Điều kiện Clean | **Không** | Dump chỉ có LR2 |
| Ảnh để làm thẻ | **Có, cần lưu ý** | Ảnh Clean nằm trong `cache/textvqa_val2262_clean.zip` trên Drive. LR2 tạo lại được bằng `native_lr2`, nhưng máy local có Pillow 12.2.0 thay vì 11.3.0, nên ảnh LR2 local **chỉ dùng để hiển thị**, không dùng để chấm điểm |
| Latency | **Không** | Ô 8 đã viết sẵn, chưa chạy |
| Chẩn đoán của Tuấn ở `.40/.60` | **Có** | Xem mục 2.1; khác điểm vận hành |

### 2.4 Việc nào làm trên CPU, việc nào thật sự cần GPU

| Việc | CPU từ dữ liệu hiện có | Cần GPU bổ sung |
|---|---|---|
| Kiểm tra tham chiếu T4/A100, thống kê lỗi, phân tích box bổ sung, tiny bỏ sót, so DualAgree v1 | **Có**, toàn bộ | — |
| Bộ thẻ và phiếu chấm | **Có** | — |
| Latency của CS-B1 | — | **Có:** Ô 8, khoảng 10 phút chạy + 30–40 phút dựng môi trường trên T4 (ước lượng) |
| Biết tiny bỏ sót có phản hồi yếu trên P-map hay không, tại điểm của CS-B1 | — | **Có:** cần dump P-map (chưa đề xuất trong Phase 0). Tạm dùng `gt_diagnosis` của Tuấn ở `.40/.60` làm bối cảnh |
| Áp CS-B1 cho mọi ảnh, kể cả Clean, không định tuyến | — | **Có:** dump Clean cho cả hai model, khoảng 50 phút T4 (ước lượng). Không cần nếu hệ thống định tuyến theo kích thước như hiện tại |
| Đối chiếu A100 | — | Không có A100. Dùng cách ở mục 3 |

**Cách biết ảnh nào là LR2 khi triển khai:** hiện dùng router kích thước (cạnh dài ≤ 768 → LR2). Trong protocol này router đó tách đúng hoàn toàn (E13), nhưng đây là shortcut của quy trình tạo LR2. Báo cáo phải ghi rõ giới hạn này.

---

## 3. Bước đầu tiên: kiểm tra tham chiếu T4/A100

### 3.1 "Lệch 3,7–4,2% ô" là lệch gì
- **Một ô** là vector counts 11 trường `[tp, gt, pred, (gt_bin, tp_bin) × 4 bin chiều cao]` của **một nguồn × điều kiện LR2 × một trong 72 setting**. Tổng cộng 2.262 × 72 = 162.864 ô mỗi model.
- **Ô lệch** là ô có ít nhất một trường khác với counts A2r trên A100. Số ô lệch: B-Clean 5.953 (3,66%), B1 6.896 (4,23%) (`runs/dualagree_analysis_v1_T4/fidelity.json`).
- **Đã biết** (từ 10 ví dụ đầu mỗi model): lệch là ±1 TP hoặc ±1 pred; `gt` không bao giờ lệch, vì GT cố định.
- **Chưa biết:**
  - phân bố lệch theo từng trường (TP, pred, tiny);
  - lệch theo setting và theo split;
  - lệch có tập trung ở box có score sát ngưỡng không (giả thuyết, chưa kiểm).
- **Đã biết ở mức tổng:** LR2 Hmean của B1 tại `epsilon_005/0.50`:
  - A: T4 77,0322% so với A100 77,0320%;
  - B: T4 76,6866% so với A100 76,6938%.
  - Chênh tương ứng +0,0002 và −0,0072 pp. Nguồn: `decision.json` (`b1_alone`) và `results/diag_v2_cpu/p0_fixed.json` (`oracle_cond.*.lr2`).
- **Điều kiện so sánh:** cùng ảnh, cùng checkpoint (SHA qua merge manifest), cùng config, cùng DBPostprocessor SHA, cùng commit MONKEY_V2, cùng torch/CUDA/cuDNN. Chỉ khác GPU (Tesla T4 so với A100-SXM4-40GB). Đã đối chiếu `dump_identity.json` với `audit_identity.json`.
- T4 tự tái lập tuyệt đối (720/720 mục). Vậy lệch là do **khác GPU**, không phải do dump ngẫu nhiên.

### 3.2 Đề xuất kiểm tra (CPU, khoảng 30 phút máy)
1. **Bảng lệch theo trường:** với cả 162.864 ô mỗi model, đếm số ô lệch theo từng trường (tp, pred, từng bin tiny), theo setting, theo split A/B. Thêm phân bố độ lớn lệch (±1, ±2, …).
2. **Tác động lên các số dùng cho CS-B1:**
   - Tính lại tổng TP, FP, FN, Hmean và tiny của B1 `epsilon_005/0.50` và B-Clean `mask_040/0.70` trên A và B, ở cả T4 và A100.
   - Báo chênh lệch theo pp và theo số đếm.
3. **Độ nhạy của ΔCS-B1:** không thể tính CS-B1 trên A100, vì A100 không có box. Thay vào đó:
   - tính lại ΔLR2 của CS-B1 sau khi **loại các nguồn** có ô lệch ở một trong hai setting đang dùng;
   - so ΔLR2 trước và sau khi loại. Nếu dấu và độ lớn giữ nguyên trong CI, phần lệch không giải thích được tín hiệu.
4. **Phân tách kết luận:**
   - **Chẩn đoán** (loại lỗi, nhóm kích thước, so DualAgree v1) **được tiếp tục** trên T4, vì đó là so sánh trong cùng một dump.
   - **Kết luận "CS-B1 cải tiến"** giữ nhãn PENDING_REFERENCE_CHECK cho tới khi bước 3 xong. Kể cả khi đạt, vẫn ghi "đo trên T4, tái lập trên T4".

**Không** coi lệch nhỏ là vô hại trước khi có bảng ở bước 1–3. **Không** coi toàn bộ kết quả vô hiệu: bằng chứng hiện có chỉ cho thấy lệch ±1 count và lệch tổng rất nhỏ ở một setting.

---

## 4. Câu hỏi Phase 0 phải trả lời, và mức kết luận dữ liệu cho phép

Tất cả tính trên **A** để phát triển. **B chỉ báo mô tả**, không dùng để chọn hay sửa luật. Matching, ROI và ignore dùng nguyên `CandidateMatcher` và protocol đã khóa. Phân loại FP dùng lại đúng hàm `fp_proxy` của Tuấn (`rev4/decision_review.py:55`) để hai bên so được với nhau. IoU chỉ là dấu hiệu hỗ trợ, không chứng minh loại lỗi.

**Định nghĩa kích thước:**
- **Tiny theo GT:** chiều cao GT quy về LR2 native < 4 px. Đây đúng là bin `<4` của evaluator: `height × (H//2)/H`.
- **Kích thước box dự đoán:** cạnh ngắn của hình chữ nhật xoay nhỏ nhất bao box, quy về LR2 native. Đây là đại lượng **khác** chiều cao GT. Không trộn hai đại lượng trong cùng một bảng.

| # | Câu hỏi | Thống kê tự động | Mức kết luận cho phép |
|---|---|---|---|
| Q1 | FP của B1 (và của CS-B1) chủ yếu là loại nào? | `fp_proxy` cho mọi FP của B1 `epsilon_005/0.50` và của CS-B1 trên A, chia theo kích thước box dự đoán và bucket score. Đối chiếu với bảng proxy ở `.40/.60` của Tuấn | Proxy chỉ là **dấu hiệu**. Phân biệt "nền thật" với "chữ thật thiếu nhãn" chỉ làm được qua chấm tay, và chỉ ghi nhận, không sửa GT |
| Q2 | Chữ tiny bị bỏ sót nằm ở đâu? | Với mỗi GT tiny là FN của CS-B1 trên A, xếp vào **một** nhóm theo thứ tự: (a) **không có box B1 nào** trong dump chạm GT; (b) có box B1 IoU ≥ 0,5 nhưng raw < 0,5 → **bị lọc theo score**; (c) có box B1 chạm GT nhưng IoU < 0,5 → **lệch hình học**; (d) có box IoU ≥ 0,5 và đủ score nhưng đã ghép với GT khác → **gán/matching**; (e) như (a) nhưng có box B-Clean ≥ 0,7 IoU ≥ 0,5 mà **luật ghép không thêm** | Nhóm (a) **không** được gọi là "phản hồi P-map yếu": dump không có P-map, và box có thể đã bị loại vì min width trước khi lưu. Chỉ dùng `gt_diagnosis` của Tuấn (`.40/.60`) làm bối cảnh, ghi rõ là khác điểm vận hành |
| Q3 | Box B-Clean bổ sung cứu được gì, tốn gì? | Với mọi box được thêm trên A: TP (cứu GT nào, bin chiều cao GT) hay FP (`fp_proxy`), theo kích thước box và bucket score (0,70–0,80, 0,80–0,90, ≥0,90). Đo độ tập trung: số nguồn đóng góp, và phần trăm lợi ích ròng đến từ 10 nguồn đóng góp nhiều nhất | Định lượng đầy đủ, vì dữ liệu đủ |
| Q4 | Vì sao DualAgree v1 mất tiny còn CS-B1 giữ được? | Với GT tiny là TP của B1 đơn trên A nhưng FN của DualAgree v1 (cấu hình khóa: t_l .7, t_c .7, t_a .5, θ .3): xếp theo lý do (box B1 đứng một mình có raw trong [0,5; 0,7); nằm trong cặp nhưng không đạt t_a; …). Đối chiếu tập đó với CS-B1 | Đây là phép kiểm cơ chế, định lượng đầy đủ |
| Q5 | Tiếp tục bổ sung box hay chuyển sang sửa hậu xử lý riêng B1? | Trong các GT được CS-B1 cứu trên A: tỉ lệ GT mà B1 **đã có box chạm GT** (IoU ≥ 0,3) nhưng bị lọc vì score hoặc lệch hình học. Ở các geometry khác của B1 (9 geometry đã có trong dump), có box nào cứu được chính GT đó không | Nếu phần lớn lợi ích đến từ box B1 đã có sẵn thì một model có thể tự lấy lại được, nên thử hậu xử lý B1. Nếu B1 không có box nào thì cần model thứ hai |

**Mô tả phụ (không dùng để đổi cấu hình):**
- Độ nhạy quanh tham chiếu trên A và B: t_c ∈ {0,60; 0,70}, θ ∈ {0,3; 0,5; 0,7}, lấy từ `sweep_totals.npz` đã có.
- So CS-B1 với B1 ở `.40/.60` (baseline chính của Tuấn, có trong 72 setting của dump) để người đọc của nhóm dễ đối chiếu.

---

## 5. Chấm tay: khoảng 120 thẻ

### 5.1 Chia mẫu (chỉ trên A)

| Nhóm | Số thẻ | Tầng (strata) | Cách chọn |
|---|---:|---|---|
| **F · FP của B1 giữ trong CS-B1** | 40 | 4 proxy × 2 kích thước box (< 8 px / ≥ 8 px), 5 thẻ mỗi tầng | Mỗi tầng xếp theo `sha256(seed:case_id)`, lấy 5 hạng nhỏ nhất. Tầng có < 5 ca thì lấy hết, phần dư chia đều cho các tầng khác |
| **S · Box bổ sung từ B-Clean** | 40 | 20 TP (cứu GT) + 20 FP; mỗi bên 2 kích thước × 2 bucket score | Như trên |
| **T · GT tiny bị CS-B1 bỏ sót** | 40 | Các nhóm (a)–(e) ở Q2, tối thiểu 6 thẻ mỗi nhóm, phần còn lại chia theo tỉ lệ quần thể | Như trên |

- Seed cố định (đề xuất `20261123`). Danh sách thẻ, quần thể mỗi tầng và xác suất chọn được lưu **trước** khi chấm.
- Không chọn tay, không "chọn ví dụ đẹp".
- Thêm **10 thẻ hiệu chuẩn** (5 FP, 5 tiny bỏ sót), lấy theo cùng quy tắc từ **B**. Dùng để tập chấm, không đưa vào thống kê.

### 5.2 Nội dung một thẻ
- Crop quanh box hoặc GT, có viền ngữ cảnh. Hai ô cạnh nhau: ảnh **LR2** (phóng ×4, nội suy nearest) và ảnh **Clean** cùng vùng để tham chiếu.
- GT hợp lệ viền xanh lá, GT ignore viền xám đứt nét, ROI nếu cắt vào crop.
- Dự đoán của hai model với nhãn trung tính **"bộ 1"** và **"bộ 2"**: màu cố định cho từng bộ, **thứ tự trộn ngẫu nhiên** theo seed, có ghi score.
- Trạng thái cuối, ghi trung tính: "có trong đầu ra cuối / không".
- Không ghi tên CS-B1, DualAgree, B1 hay B-Clean trên thẻ. Bảng giải mã nằm ở file riêng.
- Mã thẻ, nguồn, chiều cao GT (px LR2), cạnh ngắn box (px LR2).

### 5.3 Phiếu chấm (CSV, cột tương thích `human_review.csv` của Tuấn)
`case_id, group, stratum, inclusion_probability, population_weight, human_label, confidence, reviewer, notes`

**Nhãn (chọn một):**
- **Nhóm F và S-FP:**
  - `nen` — nền hoặc hoa văn giống chữ, không có chữ;
  - `thieu_nhan` — có chữ thật đọc được nhưng không có GT (**nghi vấn**, chỉ ghi nhận);
  - `dinh_vi` — box phủ đúng chữ nhưng lệch hoặc to/nhỏ quá;
  - `gop` — một box phủ từ 2 chữ trở lên;
  - `tach` — box chỉ phủ một phần chữ;
  - `trung` — trùng với box khác đã đúng;
  - `ignore` — dính vùng ignore;
  - `khong_chac`.
- **Nhóm S-TP:**
  - `dung` — box khớp chữ;
  - `dung_nhung_lech` — được tính TP nhưng mắt thấy lệch;
  - `nghi_gt_sai`;
  - `khong_chac`.
- **Nhóm T:**
  - `mat_thong_tin` — chữ không còn nhìn thấy trên LR2;
  - `thay_nhung_khong_box` — nhìn thấy chữ nhưng không có box nào gần;
  - `co_box_lech` — có box gần nhưng lệch;
  - `bi_gop` — chữ bị gộp vào box lớn;
  - `nghi_gt_sai`;
  - `khong_chac`.
- Cột `confidence`: `chac` hoặc `tam`.
- Hướng dẫn 1 trang kèm 1 ví dụ cho mỗi nhãn, lấy từ thẻ hiệu chuẩn.

### 5.4 Quy tắc tổng hợp
- Báo **số đếm theo từng tầng**.
- Ước lượng tỉ lệ toàn tập chỉ bằng trọng số `population_weight` (Horvitz–Thompson), kèm CI bootstrap **theo nguồn**: các thẻ cùng nguồn đi cùng nhau.
- Với 40 thẻ mỗi nhóm, CI sẽ rộng, cần nói rõ.
- **Không** dùng tỉ lệ thô của 120 thẻ để tuyên bố tỉ lệ lỗi toàn tập.
- Nhãn `thieu_nhan` và `nghi_gt_sai` **không** được dùng để sửa GT hay tính lại điểm.
- Tỉ lệ `khong_chac` > 30% ở một nhóm thì nhóm đó được coi là không kết luận được.

---

## 6. Lịch 3 ngày

**Lưu ý:** `CLAUDE.md` ghi hạn là **24/11/2026**, còn user nói còn khoảng 3 ngày. Lịch dưới đánh số N1–N3. Nếu hạn thật là 24/11 thì gộp N2 và N3, và cắt chấm tay xuống 80 thẻ (bỏ 40 thẻ nhóm F, chỉ dùng proxy tự động cho F).

| Ngày | Bước | Đầu vào | Thao tác | Đầu ra | Người làm | Điều kiện xong | Thời gian user / máy |
|---|---|---|---|---|---|---|---|
| N1 | 1. Kiểm tra tham chiếu | `fidelity.json`, NPZ A2r, dump T4 | Bảng lệch theo trường; tác động lên 2 setting; độ nhạy ΔCS-B1 khi loại nguồn lệch | `results/phase0/reference_check.json` + bảng | Claude | Đủ ba bảng ở mục 3.2 | 0 / khoảng 30 phút |
| N1 | 2. Thống kê tự động Q1–Q5 | Dump, bundle, `sweep_totals.npz`, `fp_proxy` của Tuấn | Viết script CPU + test, chạy trên A, mô tả B | `results/phase0/auto_stats.json` + bảng | Claude | Có số cho Q1–Q5, test PASS | 0 / khoảng 30–60 phút |
| N1 | 3. Chạy latency | Zip mới (SHA `343342a0…`) | Tải zip lên Drive, chạy Ô 1–4 rồi Ô 8 trên T4 | `runs/dualagree_latency_v1_T4/latency.json` | **User** bấm chạy; Claude đọc | Có file, kiểm chéo khớp dump | **10 phút** / 40–60 phút |
| N1 | 4. Tạo thẻ và phiếu | Danh sách mẫu ở mục 5.1, ảnh từ cache Drive | Sinh 130 thẻ PNG + CSV phiếu + file giải mã + hướng dẫn | `results/phase0/cards/`, `review_sheet.csv`, `guide.md` | Claude | Đủ thẻ, trọng số lưu trước | 0 / khoảng 15 phút |
| N1 | 5. Đọc và hiệu chuẩn | Bảng 1–2, 10 thẻ hiệu chuẩn | Đọc tóm tắt; chấm 10 thẻ hiệu chuẩn; hỏi chỗ chưa rõ | Ghi chú | **User** | Hiểu nhãn | **50 phút** |
| N1 | 6. Chấm đợt 1 | Thẻ nhóm S (40) | Chấm | `review_sheet.csv` đợt 1 | **User** | 40 thẻ có nhãn | **60 phút** |
| N2 | 7. Chấm đợt 2 | Thẻ nhóm T (40) và F (40) | Chấm | `review_sheet.csv` đủ | **User** | 120 thẻ có nhãn | **100 phút** |
| N2 | 8. Tổng hợp | Phiếu + trọng số | Ước lượng có trọng số, CI theo nguồn | `results/phase0/review_summary.json` + bảng | Claude | Có bảng | 0 / 5 phút |
| N2 | 9. Đối chiếu nhanh | Bảng 8 | User xem các ca `khong_chac` | Ghi chú | **User** | — | **20 phút** |
| N3 | 10. Áp bảng quyết định | Tất cả bảng | So với mục 7, **không đổi ngưỡng** | `docs/phase0_csb1_conclusion.md` | Claude soạn, **user duyệt** | Một nhánh được chọn kèm lý do | **40 phút** |
| N3 | 11. Chuẩn bị nội dung báo cáo | Kết luận Phase 0 | Bảng kết quả, đối chứng, giới hạn | Bản nháp phần báo cáo | Claude soạn, user đọc | User đồng ý | **80 phút** |

**Sản phẩm tối thiểu:**
1. Bảng kiểm tra tham chiếu.
2. Thống kê lỗi (Q1, Q2).
3. Phân tích box bổ sung (Q3, Q5).
4. Đối chiếu DualAgree v1 (Q4).
5. Bộ thẻ, phiếu và hướng dẫn.
6. Bảng tổng hợp chấm tay.
7. Bản kết luận ngắn.

---

## 7. Bảng quyết định (khóa trước khi xem kết quả Phase 0)

Mọi điều kiện tính trên **A**. B chỉ để xem có cùng chiều hay không. Các cổng cũ (+0,3 pp luật dừng, +0,5 pp cổng nhóm) **không đổi**, và CS-B1 hiện **chưa đạt** cả hai. Phase 0 không làm cho nó đạt; Phase 0 quyết định cách trình bày và hướng tiếp theo.

| Kết luận | Điều kiện (đủ tất cả) |
|---|---|
| **A · Tiếp tục CS-B1** làm ứng viên chính, báo là "tín hiệu dương nhỏ, khám phá" | (1) Bước 3: ΔCS-B1 sau khi loại nguồn lệch vẫn > 0 và nằm trong CI gốc. (2) Q3: TP cứu được > FP thêm, và 10 nguồn đóng góp nhiều nhất chiếm < 50% lợi ích ròng. (3) Mô tả độ nhạy: dấu của Δ giữ dương ở t_c 0,6 và θ 0,3/0,7 trên A. (4) Q5: < 50% GT được cứu là loại B1 đã có box chạm GT |
| **B · Thử hậu xử lý riêng B1** (1 model) | Q5: ≥ 50% GT được CS-B1 cứu là loại B1 đã có box chạm GT (IoU ≥ 0,3) nhưng bị lọc score hoặc lệch hình học. **Hoặc** Q2: nhóm (b)+(c) chiếm ≥ 30% GT tiny bị bỏ sót. Phép thử dùng đúng 72 setting và 9 geometry đã có trong dump, chọn trên A với **cùng luật** (+0,3 pp, tiny ≥ −3 pp), không thêm luật mới |
| **C · Cần thêm dữ liệu** | Bước 3 không kết luận được (Δ đổi dấu khi loại nguồn lệch, hoặc số nguồn còn lại quá ít). **Hoặc** nhóm (a) ở Q2 chiếm > 60% GT tiny bỏ sót, tức không có box nào nên chỉ P-map mới phân biệt được (đề xuất dump P-map, không mở trong 3 ngày). **Hoặc** `khong_chac` > 30% ở nhóm S hay T |
| **D · Dừng nhánh bổ sung box** | ΔCS-B1 đổi dấu ở bước 3 hoặc ở độ nhạy. **Hoặc** FP thêm ≥ TP cứu. **Hoặc** ≥ 50% lợi ích ròng đến từ ≤ 10 nguồn. Khi dừng: báo cáo CS-B1 là kết quả khám phá không bền, kèm phân tích lỗi Phase 0 |

**Ghi chú về nhãn `thieu_nhan`:** nếu nhiều FP của box bổ sung là chữ thật thiếu nhãn, đó là thông tin cho báo cáo (điểm có thể bị đánh giá thấp). Nó **không** được dùng để đổi kết luận A/B/C/D hay để tính lại Hmean.

**Không làm trong Phase 0:**
- mở phương pháp mới;
- chỉnh ngưỡng CS-B1 sau khi xem Phase 0;
- dùng B để chọn;
- train;
- đọc TEST.

---

## 8. Phương pháp hiện tại và Phase 0, nói đơn giản

- **CS-B1 làm gì:** với ảnh độ phân giải thấp, chạy B1 như bình thường, rồi chạy thêm B-Clean. Chữ nào B-Clean thấy rất chắc (score ≥ 0,7) mà B1 không có box thì thêm vào. B1 không bị bớt box nào.
- **Phase 0 giúp biết:**
  1. Lệch giữa T4 và A100 có làm sai tín hiệu nhỏ +0,18 đến +0,28 pp hay không.
  2. Box thêm vào cứu được những chữ nào, tốn bao nhiêu FP, và có phụ thuộc vào vài ảnh không.
  3. Chữ tiny bị mất ở đâu.
  4. Liệu chỉ cần sửa hậu xử lý của B1 (một model) là đủ, hay thật sự cần model thứ hai.
- **Bạn cần làm tiếp:**
  1. Duyệt kế hoạch này, nhất là bảng quyết định ở mục 7 và số thẻ (120 hay 80 nếu hạn là 24/11).
  2. Tải zip mới lên Drive, chạy Ô 1–4 rồi Ô 8 (latency) trên T4.
  3. Khi Claude gửi thẻ, chấm 10 thẻ hiệu chuẩn trước, rồi chấm theo lịch.
