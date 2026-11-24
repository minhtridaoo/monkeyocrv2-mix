# HANDOFF — trạng thái hiện tại (đọc đầu mỗi session)

> Hiện trạng ghi ở đây; lịch sử chỉ append vào [WORKLOG.md](WORKLOG.md). Phạm vi và protocol theo [kế hoạch chính](Ke_hoach_phoi_hop_MixA_MixB_20261119.md); đọc §0 trước khi làm. Không lấy số toàn cohort làm baseline trên B.

## TRẠNG THÁI HIỆN TẠI

- **24/11 (Codex): user yêu cầu tìm hướng cải tiến tiếp.** Đã audit code + dữ liệu và viết `docs/KE_HOACH_BOX_QUALITY_20261124.md`: đề xuất learned box-quality scoring, một forward B1, fit trên TRAIN; CHƯA triển khai/chưa GPU/chưa có cải thiện. Đã kiểm 15.222 TRAIN không trùng 2.262 dev, chuẩn bị danh sách dự kiến 1.600 fit/400 calibration ở `results/boxquality_preparation_20261124/`. Ảnh/GT TRAIN chưa xác minh, Drive H không có lúc kiểm. Bước tiếp: P0 data/label audit → notebook smoke; không mở TEST, không sửa kết luận cũ. Các mục chờ Clean/Flip pilot bên dưới là lịch sử đã được kết quả 23/11 thay thế.

- **MỚI NHẤT · 23/11 khuya (Claude): chẩn đoán Flip-TTA (CPU, hậu kiểm) → khuyến nghị DỪNG** (`docs/fliptta_diagnosis_plan_20261123.md`, `results/fliptta_diag_v1/`, `mixab/fliptta_diag.py`, `scripts/fliptta_matched_control.py`).
  - **Đã kiểm:** counts pool = `per_source.csv` (600 task); evaluator chính thức khớp 1.200/1.200. Spec, nhãn `INCONCLUSIVE` và cổng của pilot cũ **không đổi**.
  - **Bằng chứng:**
    - (1) So với F nâng ngưỡng ở cùng recall/FP (hậu kiểm, mô tả): Clean +0,06 / +0,004 pp, tức **tương đương nâng ngưỡng**; LR2 +0,27 [−0,15; 0,66] / +0,19 [−0,23; 0,58].
    - (2) Churn: GT mất/thêm LR2 171/149, Clean 187/149. Khoảng 60–70% sự kiện hình học có IoU sát 0,5.
    - (3) Lệch có hệ thống: GT mất do score tụt (LR2 57 so với 29 được thêm; chênh score trung vị 0,08).
    - (4) FP bị loại chủ yếu là box F điểm 0,60–0,65.
    - (5) Không tập trung (top 10 nguồn ≈ 30%).
    - Cận trên lạc quan nếu cứu miễn phí mọi TP mất do score: +0,40 (LR2) / +0,31 (Clean) so với T0.
  - **Giả thuyết (thiếu P-map / nhánh lật riêng):** nhánh lật kéo score xuống ở chữ soi gương; gộp/tách do P-map trung bình.
  - **Hướng:** A và C không chọn. B (đồng thuận vùng bất định) là ứng viên duy nhất nhưng **chưa có cơ sở chọn τ**. Chỉ làm tiếp nếu user duyệt một lượt chẩn đoán P-map (~20–30 phút T4, **chưa chuẩn bị notebook**), xem mục 6 của tài liệu.
  - 12 hình chọn theo hash ở `results/fliptta_diag_v1/figures/`. Test 127/127 PASS.
- **MỚI NHẤT · 23/11 tối (Claude): đã tiếp nhận pilot Flip-TTA** (`docs/fliptta_result_20261123.md`, bản chính `results/fliptta_v1_T4/`; output `runtime/kaggle_downloads/fliptta_pilot_1/`).
  - Kiểm đạt: manifest `bc0835c6…`, env đúng, mẫu = 300 nguồn đã khóa (không có ảnh trang báo), 600/600 ảnh; **cổng căn chỉnh đạt** (kiểm cứng 600/600; IoU đã căn 0,847 so với đối chứng 0,030); F LR2 khớp dump T4 300/300, Clean lệch A2r 5/300 (GPU). Tính lại local **trùng tuyệt đối**.
  - LR2: 78,37 → 78,64, **+0,27 [−0,16; +0,66]**; Clean: 80,52 → 80,73, **+0,21 [−0,22; +0,68]**. FP −92/−99 (≈ −8%), TP −22/−36; tiny LR2 −1,18 pp. Latency **1,95×** (476 → 903 ms), 1 checkpoint.
  - Nhãn khóa **`INCONCLUSIVE`** (trượt c1; c2, c3 đạt). Không tự mở rộng mẫu, không đổi cách gộp/ngưỡng. Bước tiếp cần user (và Tuấn) quyết định + spec mới khóa trước.
- **MỚI NHẤT · 23/11 tối (Claude): đã tiếp nhận Clean CS-B1** (`docs/csb1_clean_result_20261123.md`, bản chính `results/csb1_clean_v1_T4/`; output gốc `runtime/kaggle_downloads/csb1_clean_T4_20261123-110923/`).
  - Kiểm đạt: code manifest `f7f5f699…`; DBPostprocessor `9bfff8aa…`, Pillow 11.3.0, T4 GPU 0; 36 shard/model, `condition = clean`, cùng 2.262 nguồn với dump LR2; `sources = 2262`. **Tính lại ở local trùng tuyệt đối** JSON của Kaggle. Probe 40 nguồn lệch A2r A100 ≈ 2,9% ô (chỉ báo).
  - ALL (2.262): Clean CS-B1 **79,82** so với B1 `.40/.60` 79,44 (**+0,39 [+0,01; +0,81]**), B1 `eps005/.50` 79,41, B-Clean **80,84** (CS-B1 **−1,02 [−1,76; −0,49]**). LR2 không đổi (+0,18 [−0,21; +0,58] so với `.40/.60`).
  - Mean không định tuyến: +0,28 pp so với model đơn tốt nhất (ALL), **−0,18 trên A**, +0,21 trên B.
  - Cổng nhóm A (mô tả): mean +0,5 **không đạt**; Clean ≥ B-Clean − 0,3 **không đạt** (−1,09); tiny đạt (−2,63). Theo spec: muốn giữ Clean thì CS-B1 cần router. Không chỉnh cấu hình, không đề xuất TEST.
  - Còn chờ: Flip-TTA (user chạy `kaggle_fliptta_pilot.ipynb`), K01–K10, duyệt tin cho Tuấn (cần cập nhật số Clean trước khi gửi).
- **MỚI NHẤT · 23/11 (Claude, session tiếp theo): XV-B1 đã chạy CPU (dừng); Flip-TTA đã chuẩn bị (chưa chạy GPU).**
  - **XV-B1** (`docs/xvb1_spec_20261123.md`, `docs/xvb1_result_20261123.md`, `mixab/xvb1.py`, `results/xvb1_v1_T4/`): **ablation cùng họ DualAgree v1** (nhánh `B1+agree`: v1 trên A cho FP 6.680 → 3.920 nhưng tiny −20 pp), không phải phương pháp độc lập. Khác v1: miễn trừ box < 8 px LR2, không phụ thuộc score B1, bằng chứng B-Clean lỏng (mọi ứng viên `mask_040`, IoU ≥ 0,3). Idea chỉnh sau khi biết v1 trượt tiny.
    - Chỉ split A (không B, không GPU, không ghép CS-B1; Clean **chưa đánh giá**). Luật khóa chạy một lần.
    - So với B1 `.40/.60`: **+0,153 pp CI [−0,46; +0,80]**; tiny −3,33 pp (trượt c2), FP thấp hơn (đạt c3). So với cha `epsilon_005/.50`: **−0,079 pp CI [−0,14; −0,02]**. Xóa 356 box ở 217 nguồn: TP bị xóa 128, FP bị xóa 131 (precision cận biên ≈ 49%).
    - Nhãn khóa: **`POSITIVE_BELOW_GATE`**, nhưng mức dương đến từ cha, luật tự nó **âm có ý nghĩa** → dừng, không sửa luật. Giả thuyết "B-Clean không thấy ≠ B1 sai" có biểu hiện (128 TP mất); chưa tách nguyên nhân.
  - **Flip-TTA trên P-map** (chưa chạy): spec `docs/fliptta_spec_20261123.md`, `mixab/fliptta.py`, `tests/test_fliptta.py`, `notebooks/kaggle_fliptta_pilot.ipynb`, dataset `runtime/kaggle_upload/cs431-fliptta-pilot.zip` (337 MiB, SHA `81e6b94d…`, code manifest `bc0835c6…`), hướng dẫn `docs/HUONG_DAN_KAGGLE_fliptta_pilot.md`.
    - Khả thi: lấy được P-map (`stage_a.py:212-235`); B1 train với `Fliplr 0.5` theo cấu hình (`b1_training.py:31-35`, `inference_config.py:296`, chưa kiểm recipe thật); repo chưa có TTA suy luận.
    - Thiết kế: B1 `.40/.60`, F so với T cùng lượt; lật chỉ vùng nội dung (pad giữ nguyên, không lấy mẫu lại); trung bình số học; 300 nguồn A (seed `20261123:fliptta:`, chồng 88 nguồn với mẫu zoom), Clean + LR2; kiểm căn chỉnh cứng từng ảnh; latency đo thật; nhãn khóa `INCONCLUSIVE_ALIGNMENT/NEGATIVE/INCONCLUSIVE/PASS_SCREEN/POSITIVE_BELOW_GATE`.
    - Kiểm local: test 124/124 PASS; `evaluate` chạy thử trên bản ghi giả (đã xóa, số không có ý nghĩa). **Chưa chạy `infer` trên GPU** (không có mmocr local) → lỗi riêng của đường GPU (ví dụ tên tham số `det.model`) chỉ lộ ở Ô 4 chạy thử 3 nguồn.
    - User đã báo: chạy Clean CS-B1 trước, rồi chạy pilot này. **Chưa được ghi là đã chạy hoặc có kết quả.**
- **MỚI NHẤT · 23/11 (Claude): bàn giao sang session mới → đọc `docs/SESSION_START_20261123.md` trước.**
  - **ĐÃ XONG:** CS-B1 đã đánh giá **đủ 2.262 nguồn LR2** (development): +0,23 pp [0,15; 0,32] so với nền `epsilon_005/.50`; +0,18 pp [−0,21; +0,58] so với B1 `.40/.60` (sau calibration lần 2, cấu hình B1 tốt nhất của Tuấn), CI chứa 0; latency 1,99×; **chưa đạt cổng nhóm**, luật dừng A trượt. Baseline đã chốt từ file gốc (`results/csb1_full_v1_T4/baseline_check.json`).
  - **ĐANG CHỜ USER CHẠY:** bổ sung Clean `notebooks/kaggle_csb1_clean.ipynb` + `runtime/kaggle_upload/cs431-csb1-clean.zip` (939 MiB, SHA `337f4b06…`; code manifest `f7f5f699…`), hướng dẫn `docs/HUONG_DAN_KAGGLE_csb1_clean.md`. **Chưa có kết quả.** Mục đích: hoàn thiện đánh giá và đo đánh đổi; không mặc định sẽ đạt cổng. Cách tiếp nhận output: SESSION_START §2. Cũng đang chờ: K01–K10, duyệt tin cho Tuấn.
  - **KẾ HOẠCH (chưa làm, chờ user chọn):** hướng thứ hai. Ưu tiên XV-B1 (lọc FP của B1 bằng sự vắng mặt của B-Clean ở box ≥ 8 px); các phương án khác: Flip-TTA trên P-map, điểm ổn định theo mask. Không còn tập chưa dùng để chọn phương án → pilot chỉ là khám phá trên A.
- **MỚI NHẤT · 23/11 (Claude): chốt baseline từ file gốc** (`results/csb1_full_v1_T4/baseline_check.json`):
  - B1 `.40/.60` 76,90 là điểm Tuấn chọn lại trên lưới mở rộng (luật mean Clean + LR2), **sau** calibration lần 2. Mức +1,61 pp là so với `.30/.40` ("original calibrated", 75,29). `.40/.60` cũng là setting LR2 tốt nhất trong 72 setting trên đủ tập.
  - FP 11.997 (H200 của Tuấn) / 11.999 (A100) / 11.988 (T4) là do khác GPU; 32 nguồn lệch ±1.
  - CS-B1 đang hơn **B1 `epsilon_005/.50`** (+0,23 [0,15; 0,32]); so với **B1 `.40/.60`** thì +0,18 [−0,21; +0,58]. Chưa so với fusion/adapter/EDSR (không có per-source counts).
  - Tin nhắn cho Tuấn sửa lại, ghi rõ đã chạy đủ 2.262 nguồn LR2. Chưa chạy thêm; user chốt phép so sánh trước khi quyết định chạy bổ sung Clean.
- **MỚI NHẤT · 23/11 (Claude): rà soát CS-B1 + sửa roadmap theo đính chính của user.** Mục tiêu: Trí cần ít nhất một hướng có cải thiện có bằng chứng; phân biệt rõ "có mức tăng" / "đạt cổng nhóm" / "xác nhận TEST".
  - **Xác minh:** A ∪ B = đúng 2.262 nguồn của cohort (ID khớp GT manifest và counts A2r). B1 `.40/.60` tính lại từ A2r = 76,90%, FP 11.999 (Tuấn ghi 11.997).
  - **CS-B1@0.7 trên đủ 2.262 nguồn LR2** (`scripts/csb1_full_verify.py` → `results/csb1_full_v1_T4/csb1_full.json`, tính nhẹ từ counts đã có):
    - so với B1 `epsilon_005/.50`: **+0,23 pp [0,15; 0,32]**;
    - so với B1 `.40/.60` (đã calibration): **+0,18 pp [−0,21; +0,58]** (A +0,41, B −0,04); tiny −1,96 pp.
    - Luật dừng khóa trước trượt (A +0,177 < 0,3). Không đổi baseline sau khi xem số.
    - Mean hệ thống có router: A +0,36 (< 0,5); trên đủ tập +0,79, nhưng khoảng 0,7 trong đó đến từ router. Latency 1,99×.
    - **Clean chưa đo.** K01–K10 chưa chấm.
  - **Mức có thể tuyên bố:** "dương quá nhỏ" so với nền của nó; "chưa đạt cổng" so với B1 đã calibration. Không đủ điều kiện đưa vào TEST.
  - **Bổ sung tùy chọn:** `notebooks/kaggle_csb1_clean.ipynb` + `runtime/kaggle_upload/cs431-csb1-clean.zip` (939 MiB, SHA `337f4b06…`; có sẵn 2.262 ảnh Clean từ cache Drive, tải ảnh chính thức chỉ là dự phòng). Đo Clean khi không định tuyến; không giải quyết cổng. `dualagree dump` có thêm `--condition` (mặc định `lr2`, identity cũ không đổi).
  - Roadmap sửa (`docs/ROADMAP_toan_bo_20261123.md`): 4 mức kết quả; calibration **có** vượt B1 đã fine-tune ở ngưỡng cũ; bỏ đề xuất TEST cho CS-B1; EDSR là hướng của Tuấn, chỉ làm nếu Tuấn đồng ý. Tin nhắn dự thảo cho Tuấn: `docs/msg_tuan_20261123_csb1.md` (chưa gửi).
  - Drive H: đã mount lại; dataset dựng lại có ảnh Clean. AGENTS.md tìm thấy thuộc workspace khác (Paddle); Q-LRG ngoài phạm vi, không có quyền HPC. Test 108/108 PASS.
- **MỚI NHẤT · 23/11 (Claude): đồng bộ repo Tuấn + roadmap toàn bộ** → `docs/ROADMAP_toan_bo_20261123.md`.
  - MONKEY_V2 fast-forward `677b811 → b9eaf37`: 3 commit về recovery fusion 111683 (code `rev4/fusion_recovery.py`, biên nhận preflight/probe/import PASS). **Repo chưa có số kết quả fusion.** Số fusion chỉ có trong bản tóm tắt HPC user dán: nearest 76,94 / ASF 76,95 / DySample 76,94 / bilinear 76,69 LR2, ASF − nearest +0,014 pp → âm.
  - Kết luận roadmap: **chưa có phương án nào của Trí đạt cổng đã khóa.** Dương nhất là CS-B1 (+0,18 A / +0,28 B, CI > 0, post-hoc, 1,99× latency). Phía Tuấn: cải thiện có ý nghĩa so với B0 là fine-tune thích nghi (LR2 +13 đến +14,5 pp) và calibration (+1,61 pp CI [1,01; 2,30]); EDSR +0,60 pp chưa đủ (thiếu Clean và ngưỡng .60). Không module kiến trúc nào vượt đối chứng.
  - **Cần user và Tuấn hỏi thầy:** "cải tiến" được tính so với MonkeyOCRv2 gốc hay so với baseline đã fine-tune.
  - Đề xuất tiếp theo: (1) hoàn thiện EDSR ×2 ở `.40/.60`, đo cả Clean, áp cho mọi ảnh, có đối chứng bicubic (Kaggle T4, phối hợp với Tuấn trước); (2) khóa CS-B1 làm ứng viên đăng ký trước cho benchmark TEST cuối; (3) báo Tuấn về độ nhạy của Hmean với ảnh thiếu nhãn.
- **MỚI NHẤT · 23/11 (Claude): đã đọc kết quả Kaggle của 2 notebook user chạy ngày 22/11 (tải bằng Kaggle API, chỉ đọc; output ở `runtime/kaggle_downloads/`, bản chính ở `results/`).**
  - **Pilot ZB1 zoom: KHÔNG ĐẠT cổng sàng lọc → dừng nhánh zoom** (`docs/zoom_pilot_result_20261123.md`).
    - Kiểm tọa độ 6/6; F trên LR2 khớp dump Colab 300/300.
    - G2: LR2 −3,01 pp [−6,39; +0,66], Clean −3,65 pp; tiny LR2 +10,8 pp; FP LR2 +1.566 so với TP +323. Mọi arm âm, không có hiệu ứng liều.
    - Bỏ riêng ảnh trang báo (chỉ mô tả): G2 vẫn −0,40 (LR2) / −1,59 (Clean).
    - 84% FP của box thêm có proxy "không chồng GT" (nền hoặc thiếu nhãn).
  - **Kiểm trang báo:** `6e340fcbb1e0a877` có 10 GT, 0 ignore, ROI 100%; chữ thân bài không có nhãn; evaluator chính thức khớp. Ảnh này chiếm 26,8% FP của B1 trên A; top 10 ảnh chiếm 48,3%. Chỉ ghi nhận, nên báo Tuấn (chưa gửi).
  - **Latency CS-B1@0.7 trên Kaggle** (kernel `ojanek/notebookef1d03734c`, 2×T4, chỉ GPU 0; `results/csb1_latency_v1_T4/latency.json`):
    - B1 median 447 ms/ảnh, CS-B1 889 ms; **tỉ số 1,99× [1,98; 2,02]**, thêm 470 ms/ảnh. 200 ảnh × 2 lượt, xen kẽ thứ tự.
    - Bộ nhớ allocated đỉnh: B1 894 MiB, CS-B1 991 MiB.
    - Dùng chung ảnh đã giải mã: 20/20 ảnh cho box giống hệt. Box so với dump Colab T4: 200/200 ảnh khớp, polygon B1 trùng tuyệt đối.
  - Phase 0 CS-B1: phần latency đã xong. Vẫn chờ user chấm K01–K10.
- **MỚI NHẤT · 22/11 khuya (Claude): chuẩn bị pilot ZB1 (zoom cục bộ bằng B1). Chưa chạy; user tự chạy trên Kaggle. Từ nay local chỉ làm việc nhẹ.**
  - Kiểm repo:
    - `Resize(keep_ratio, scale=(4068, 1024))` (`inference_config.py:199-204`) đưa cạnh ngắn về 1024, nên Clean và LR2 vào detector cùng kích thước tensor. Bicubic ×2 toàn ảnh là phép tương đương (75,24 so với 75,29, `handoff.md:74`). Crop có cạnh ngắn s thì được phóng 1024/s.
    - Repo chưa có tile hay zoom cho detector.
  - Bằng chứng ủng hộ (A2r, A): tiny recall của B1 trên Clean chỉ 61,2% so với LR2 48,8%, dù cùng kích thước tensor → suy luận: kích thước chữ trên đầu vào detector là một nút thắt riêng.
  - Spec khóa trước: `docs/zoom_pilot_spec_20261122.md`.
    - Mẫu 300 nguồn A theo hash (gồm cả nguồn chữ dày `6e340fcbb1e0a877`, giữ nguyên), cả Clean và LR2.
    - Arm: F, **G2 (chính, 5 forward)**, G3 (10 forward), S2 (cạnh ngắn 2048), R3 (chọn 3/9 ô bằng dự đoán).
    - Hợp nhất: giữ F, thêm box zoom có cạnh ngắn LR2 < 8 px và IoU < 0,5.
    - Cổng sàng lọc G2: LR2 ≥ +0,3 pp với cận dưới CI > 0; Clean ≥ −0,3 pp; tiny ≥ −3 pp. Cổng cuối của nhóm không đổi.
  - Code `mixab/zoom.py` (`prepare`, `infer`, `evaluate`, `pagecheck`) và `tests/test_zoom.py`. Test 107/107 PASS.
  - Notebook `notebooks/kaggle_zoom_pilot.ipynb` (8 ô; Ô 5 chạy thử 3 nguồn, kiểm tọa độ 1×1); dataset `runtime/kaggle_upload/cs431-zoom-pilot.zip` (336 MiB, SHA `92db0747…`); hướng dẫn `docs/HUONG_DAN_KAGGLE_zoom_pilot.md`.
  - Ô 4 gom bước kiểm ngắn ảnh trang báo, GT và evaluator. MSB1 vẫn là kết quả âm, không đổi.
- **MỚI NHẤT · 22/11 khuya (Claude): pilot CPU MSB1 (Mask-Split B1) — KẾT QUẢ ÂM theo luật khóa.** Spec khóa trước: `docs/msb1_spec_20261122.md`; kết quả: `docs/msb1_result_20261122.md`, `results/msb1_v1_T4/`.
  - Luật: thay box B1 `epsilon_005` (≥ .5) bằng ≥ 2 box B1 `mask_040` (≥ .5) nằm ≥ 80% bên trong; 1 model, không GT.
  - Oracle chính xác (Dinkelbach + liệt kê 2^k, 13 nguồn dùng cận an toàn): +1,67 pp → không loại được hướng → chạy luật khóa.
  - A: **+0,224 pp, CI [−0,40; +0,81]** → trượt cổng +0,3. Tiny +5,3 pp; +641 TP / +810 FP; 757 GT cứu, 116 mất; cổng FP ở cùng recall đạt theo định nghĩa (không setting B1 nào đạt recall đó). B (mô tả) +0,69 [0,28; 1,13].
  - Cơ chế (mô tả): 60% lần tách không đổi TP mà chỉ thêm box. Một ảnh chữ dày (`6e340fcbb1e0a877`, 10 GT, B1 sinh 1.796 box ≈ 27% FP của B1 trên A) kéo Δ xuống; bỏ riêng ảnh đó thì +0,51 pp, chỉ mô tả, không đổi kết luận.
  - Chẩn đoán cô lập mask (cha `base`): +0,275 pp so với `base` → tác động chủ yếu đến từ mask threshold.
  - Không đề xuất GPU kiểm Clean (trượt cổng LR2). Test 100/100 PASS.
- **22/11 khuya (Claude, chỉ tư vấn):** phản biện đề xuất Hướng 2A unclip/min_width của Codex (WORKLOG cùng ngày). Bằng chứng CPU chỉ đọc: 85% GT được CS-B1 cứu là chữ bị **gộp** trong box B1 ở mask 0,3, và `mask_040` tách được; min_width 3/7 không thêm GT nào có thể đạt; unclip 1,2/1,8 chỉ thêm khoảng 127 GT (≈1,3% FN). **Đính chính:** dump đủ để chọn unclip/min_width theo từng box quanh geometry `base` (raw score khớp 1-1). Khuyến nghị: không ưu tiên 2A; hoàn tất Phase 0 + báo cáo; tùy chọn là kiểm trần CPU cho "tách box gộp bằng mask 0,4", cần user duyệt. Kế hoạch đã khóa không đổi.
- **MỚI NHẤT · 22/11 tối (Claude): Phase 0 CS-B1@0.7 đã chạy phần CPU; notebook Kaggle và thẻ hiệu chuẩn đã sẵn sàng.** User duyệt Phase 0, chọn **80 thẻ (10 hiệu chuẩn + 70 chính)**, chạy T4 trên **Kaggle**. Chi tiết và bảng số: `docs/phase0_csb1_auto_results_20261122.md`. Mọi số: PENDING_REFERENCE_CHECK (dump T4).
  - **Kiểm T4/A100** (`results/phase0/reference_check.json`):
    - Lệch ±1–3 count, không bao giờ ở GT; ở 2 setting CS-B1 dùng, Hmean lệch ≤ 0,009 pp.
    - Bỏ 35 nguồn lệch trên A: ΔCS-B1 = +0,127 [0,052; 0,204], vẫn dương, nằm trong CI gốc → điều kiện A(1) đạt.
    - Cách loại chặt hơn (bỏ 240 nguồn): +0,079 [−0,004; 0,168].
  - **Thống kê tự động trên A** (`results/phase0/auto_stats.json`):
    - Q3: +248 TP / +230 FP. **10 nguồn chiếm 66,6% lợi ích ròng**; bỏ 10 nguồn đó Δ còn +0,066 pp.
    - Độ nhạy: θ 0,7 cho Δ âm (−0,32 pp).
    - Q2: tiny bỏ sót (b)+(c) chiếm 44,4%.
    - Q4: cả 527 GT tiny DualAgree v1 làm mất đều do box B1 có score 0,5–0,7 bị cắt; CS-B1 giữ cả 527.
    - Q5: 44,4% theo quy tắc khóa; mô tả thêm: 69% GT được cứu thì B1 `mask_040` cũng tìm được.
  - **Bảng quyết định (chỉ phần tự động, ngưỡng không đổi):**
    - A **không đạt**: top 10 ≥ 50%, θ 0,7 đổi dấu.
    - **D đạt 2 điều kiện.**
    - B đạt (Q2), nhưng phép thử của B (chọn lại trong 72 setting B1 trên A) đã có đáp án từ counts A2r: `epsilon_005/0.50` đã là tốt nhất.
    - **Chưa chốt kết luận:** chờ chấm thẻ (`khong_chac`) và user quyết định.
  - **Thẻ:** `results/phase0/review/`. Đã khóa lựa chọn: seed 20261123, `selection.json`.
    - 10 thẻ hiệu chuẩn K01–K10 (split B) kèm `HUONG_DAN_CHAM.md` và `calibration_sheet.csv` (phiếu mù).
    - 70 thẻ chính C01–C70 (A: F 16, S 24, T 30) **đã tạo, chưa gửi**.
  - **Kaggle:**
    - `notebooks/kaggle_csb1_latency.ipynb` (6 ô), builder `scripts/build_kaggle_notebook.py`.
    - Module mới `mixab/csb1_latency.py`: đo xen kẽ, có warm-up và sync, đo riêng bộ nhớ B1 và CS-B1, chỉ dùng GPU 0.
    - Dataset `runtime/kaggle_upload/cs431-csb1-latency.zip` (398 MiB, SHA `b6ceffa7…`).
    - Hướng dẫn: `docs/HUONG_DAN_KAGGLE_csb1_latency.md`.
    - **Chưa chạy trên Kaggle.**
  - Test 96/96 PASS (thêm 12 test trong `tests/test_phase0.py`). MONKEY_V2 không có commit mới (`677b811`).
- **Cập nhật tiến độ 22/11 chiều (phiên mới, Claude):** `git fetch` MONKEY_V2: **không có commit mới**, vẫn `677b811` (chỉ có `scripts/` chưa theo dõi của Tuấn, bình thường). Không có file nào trong workspace mới hơn WORKLOG; không có `runs/` local → Ô 8 latency vẫn **chưa chạy**, Phase 0 CS-B1 vẫn **chưa bắt đầu** (chờ user duyệt). Không có kết quả fusion 106090 trong repo. Dashboard đã thêm bước CSB1 (trước đó chỉ có trong log).
- **Đồng bộ Git 22/11 (Codex):** fetch + fast-forward `ae947e9 → 96612b4`. Đã đọc `docs/fusion_research_and_generalization.md` và `rev4/fusion_plan.py`.
  - **Đính chính:** latent adapter cả 1k và 10k đã có kết luận **chưa hơn B1/control** theo tài liệu mới của Tuấn và ảnh user cung cấp. Không còn chờ kết quả 10k. Chưa có raw report 10k local để kiểm lại từng số.
  - **CRAFT đã sàng lọc:** tài liệu mới báo LR2 Hmean 47,00%, 0 TP bin native LR2 <4 px; có phần bổ trợ với B1. Teacher Clean TRAIN: 32 thẻ/200 nguồn chưa review ký tự; chưa train character/affinity head.
  - **Hướng Tuấn mới: fusion/upsampling FPNC**, code có bốn arm nearest/ASF/bilinear/DySample. Tất cả train FPNC+DBHead, chỉ freeze backbone. Một lịch 10k, mốc 1k là gate kỹ thuật trong cùng run. Stress 500 nguồn LR4/LR8/Blur/JPEG với ngưỡng khóa. **Đã code/commit; chưa xác nhận submit/completion/result của fusion mới.**
  - Phần Trí: Mix A âm giữ nguyên; v2 DualAgree ở output khác vị trí can thiệp FPNC của Tuấn. Mix B không tự mở sau latent âm, cần giả thuyết/đối chứng mới và phối hợp tránh trùng.


- **Mới nhất (22/11 tối, Claude): ứng viên được đặt tên CS-B1 (Clean-Supplemented B1), cấu hình `CS-B1@0.7`; kế hoạch Phase 0 ở `docs/phase0_csb1_plan_20261122.md` (chỉ kế hoạch, chưa chạy).**
  - CS-B1 = giữ nguyên đầu ra B1 (`epsilon_005`, ≥ .5) + thêm box chỉ B-Clean thấy (`mask_040`, ≥ .7, khử trùng IoU ≥ .5). Post-hoc; LR2 +0,18 pp A / +0,28 pp B; PENDING_REFERENCE_CHECK; chưa đạt +0,3 / +0,5.
  - Tuấn `677b811`: latent 10k âm; fusion 106090 đang chạy; có chẩn đoán `.40/.60` (proxy FP, GT diagnosis, 127 thẻ PENDING) dùng lại được làm bối cảnh.
  - Ô 8 latency **chưa chạy**; zip mới (SHA `343342a0…`) chưa tải lên Drive.
- **22/11 (trước đó), Claude:** Tuấn đồng ý 2 checkpoint (user chuyển lời). User chỉ có T4. **T4 tái lập tuyệt đối** giữa 2 lần chạy (720/720 mục giống hệt, `results/t4_repeatability.json`). Đã viết lệnh `latency` + Ô 8 notebook (chưa chạy); zip mới SHA `343342a0…`. Báo cáo: chờ user xem lại phương pháp.
- **22/11 (trước đó): có kết quả dump T4.** Mọi số là PENDING_REFERENCE_CHECK (T4 lệch A2r 3,7–4,2% ô, ±1).
  - **DualAgree v1 (lựa chọn khóa trước): KHÔNG đạt** — A LR2 +0,68 pp nhưng tiny −19,8 pp; B LR2 −0,41 pp.
  - **Ngưỡng theo kích thước B1: KHÔNG đạt** — A +0,62 nhưng tiny −7,1; B −0,02 pp.
  - **Khám phá (post-hoc):** "B1 + box chỉ B-Clean thấy (score ≥ .7, IoU ≥ .5)": LR2 +0,18 pp trên A (CI [0,07; 0,30]), +0,28 pp trên B (CI [0,17; 0,41]), tiny giữ, FP +230/+248. Nhỏ nhưng nhất quán; chưa đạt +0,3 pp trên A; cần 2 checkpoint. Chi tiết WORKLOG 22/11.
- **22/11 (trước đó): user chạy dump trên T4; thêm phân tích ngưỡng theo kích thước cho B1** (`mixab/sizethr.py`, 82/82 test PASS).
  - Probe T4 xong: fidelity DRIFT 54–55/2.880 ô so với A2r A100 (lệch GPU) → PENDING_REFERENCE_CHECK. Dump đầy đủ đang chạy (`runs/dualagree_dump_v1_T4/`), ước lượng ~23 phút/model.
  - Dump đủ cho ngưỡng kích thước trên LR2; **không có Clean** (cần dump Clean của B1 nếu muốn tuyên bố 1 model), không thử được min width < 3 hay unclip theo box.
  - Luật đạt khóa trước (A): +0,30 pp LR2 so với tốt nhất 72 setting, tiny ≥ −3 pp, FP không cao hơn ở cùng recall.
- **22/11 (trước đó): DualAgree v1 đã code + test CPU, CHƯA chạy GPU** (user bảo "triển khai thử").
  - `mixab/dualagree.py`: `dump` (GPU: lưu polygon + raw/packed score của B-Clean và B1 trên 2.262 ảnh LR2, 9 geometry, raw ≥ 0,20; shard 64, resume được), `fidelity` (tính lại counts 72 setting từ dump bằng `CandidateMatcher` của Tuấn, so với NPZ A2r), `analyze` (quét 1.928 tổ hợp t_l, t_c, t_a, θ trên A; khóa; báo B khám phá + bootstrap; luật dừng; cổng nhóm).
  - Luật v1: geometry B1 `epsilon_005`, B-Clean `mask_040` (khóa trên A ở p0_fixed); cặp (IoU ≥ θ) chỉ ra một box; cặp đồng thuận giữ polygon B1. **Khác tài liệu thiết kế: chưa có WBF trung bình tọa độ.**
  - `notebooks/colab_dualagree_v1.ipynb` (build bằng `scripts/build_dualagree_notebook.py`; Ô 1–4 lấy nguyên từ notebook A2r; Ô 5 probe 40 + cổng trung thực; Ô 6 dump đầy đủ; Ô 7 phân tích + zip). nbformat hợp lệ; **chưa chạy trên Colab**.
  - `runtime/colab_upload/mixab_code.zip` đã tạo lại (SHA-256 `ce1e2092…`), thêm `results/diag_v2_cpu/p0_fixed.json`.
  - Test 78/78 PASS (thêm 8 test: shard, ghép cặp, luật gộp, `PoolScorer` trùng tuyệt đối `CandidateMatcher` trên một ảnh thật của bundle).
  - Chạy thử `analyze` đầu-cuối trên **dump giả** (80 nguồn, box GT xê dịch + FP ngẫu nhiên): chạy thông khoảng 15–19 s; cổng trung thực báo DRIFT đúng như kỳ vọng. Số của dump giả **không có ý nghĩa**.
  - Ước lượng dump: khoảng 32 phút A100 (ngoại suy A2r 0,41–0,43 s/task), T4 chưa đo. Nên dùng A100-SXM4-40GB (cùng GPU A2r) để cổng trung thực so tuyệt đối.
- **22/11 (trước đó), Claude: trần router theo từng ảnh (P0b, CPU)** (`scripts/p0b_per_image_oracle.py` → `results/diag_v2_cpu/p0b_per_image_oracle.json`; `scripts/p0b_null_pairs.py`). Dùng GT nên là **cận trên**, không phải phương pháp.
  - Oracle từng ảnh (B-Clean `mask_040/score_0.60` | B1 `epsilon_005/score_0.50`, chọn tối ưu chính xác bằng Dinkelbach): A mean 81,22 (+2,72 pp so với model đơn tốt nhất, +2,44 so với oracle điều kiện); B 80,78.
  - **Đối chứng "may mắn khi chọn sau"** trên A (lợi so với model tốt hơn trong cặp, Clean / LR2): α½ vs α¾ +0,37 / +0,54; α¾ vs B1 +0,36 / +0,64; cùng B1 hai ngưỡng score +1,05 / +0,85. Cặp thật B-Clean vs B1: **+1,19 / +3,70**.
  - Đọc: trên Clean, lợi từng ảnh ngang mức chọn ngưỡng theo ảnh → không thấy bổ sung thật. Trên **LR2 có bổ sung thật** (3,70 so với ≤ 1,05), khớp E9 (B-Clean cứu 1.453 GT trên LR2). Nhưng oracle điều kiện luôn gửi LR2 sang B1, nên phần bổ sung này nằm **trong ảnh LR2**, gần với Thiết kế 2 (DualAgree) hơn QRoute.
  - Muốn khai thác cần đặc trưng từ đầu ra model (score, box) của cả hai expert → counts A2r không đủ, cần một lần chạy GPU lưu box + score (chưa làm, cần user cho phép).
- **22/11 (trước đó), Claude: kiểm review 6 điểm về thiết kế v2 + P0 sửa lỗi** (`scripts/p0_fixed_router_eval.py` → `results/diag_v2_cpu/p0_fixed.json`; CPU).
  - **Lỗi rò nhãn ở Q4 cũ: đúng.** `diagnose_tradeoff_v2.py:122-138` chọn setting theo (expert × điều kiện thật). Bản sửa khóa **một setting mỗi expert** trên A, chọn chung (T, k_bclean, k_b1). Trên dữ liệu này tác động không đáng kể (B: 78,9264 so với 78,9251) vì router kích thước gần như không sai; với router size-free có lỗi thì tác động sẽ lớn hơn.
  - **Biên trên A (mới, quan trọng):** model đơn tốt nhất trên A là B-Clean 78,51; oracle điều kiện trên A 78,78 → chỉ **+0,27 pp**, **dưới** mục tiêu +0,5 pp. Tức là QRoute ở mức điều kiện **không thể đạt cổng +0,5 pp trên A** kể cả với router hoàn hảo. Trên B là +0,58 pp (khám phá).
  - **Đối chứng hậu xử lý theo điều kiện (một detector):** trên A không thêm gì so với B-Clean; trên B tốt nhất là α ¾ 78,49, oracle chỉ hơn +0,43 pp.
  - **Bootstrap theo nguồn trên B** (giữ cặp Clean/LR2): oracle − B-Clean = +1,07 pp, CI95 [0,21; 1,89].
  - **Tách kích thước:** trên toàn bộ A+B, cạnh dài Clean 768–1024, LR2 384–512 (LR2 suy ra bằng chia đôi, chưa đọc file LR2) → không chồng lấn. Chỉ đúng cho cohort và quy trình tạo LR2 này; lập luận "nửa kích thước ⇒ tách hoàn hảo" trong tài liệu v2 §3 câu 2 là sai về logic.
  - Review đúng ở cả 6 điểm (chi tiết trong WORKLOG 22/11). **Đã sửa tài liệu v2 (user cho phép, 22/11):** 17 chỗ đánh dấu *[22/11]* (thêm E13–E16; mềm hóa "không phải BN/hậu xử lý" và "lệch giả định"; sửa logic kích thước ở §3 câu 2; balanced accuracy chuyển từ cổng thành chẩn đoán; ghi biên trên A +0,27 pp; Thiết kế 2/3 không phụ thuộc router khi pilot; Mode 1 → CLEAR (giới hạn), Mode 4 → CONFIRMED cho router kích thước). Ngưỡng +0,5/−3/−0,3 pp **không đổi**.
- **22/11 (trước đó), Claude: P0 đã chạy** (Drive H: mount lại; CPU, ~2 s; `results/diag_v2_cpu/diagnosis.json`). Mọi số trên B là **khám phá**.
  - **Q4 (router kích thước) = đúng bằng oracle điều kiện:** mean B 78,93 (Clean 81,16, LR2 76,69). Lý do: cạnh dài Clean ≥ 768, LR2 ≤ 512, nên ngưỡng T = 768 tách gần hoàn hảo hai điều kiện (0,27% ảnh Clean, tức 3 ảnh 768×768 trên B, bị gửi sang B1; đã kiểm counts của 3 ảnh này giống hệt nhau ở hai model). **Xác nhận shortcut Mode 4**: trong protocol native LR2 hiện tại, kích thước ảnh làm lộ điều kiện. Router size-free và đối chứng nén là bắt buộc; câu hỏi 2 ở §3 tài liệu v2 trở thành câu quyết định.
  - **Q1 (frontier trên A):** ở LR2, 13–17/72 setting của mỗi α (¼: 13, ½: 17, ¾: 14) **không bị** setting nào của hai endpoint áp đảo về (recall, precision); ở Clean chỉ 1–5/72. Nghĩa là trộn có mở rộng nhẹ frontier LR2, **chưa đo biên độ**. Điều này làm mềm câu "chỉ đi dọc đường nối" trong chẩn đoán; chưa sửa tài liệu v2.
  - **Q2 (theo bin, B):** LR2 bin < 4 px tại α ½ cao hơn đường thẳng +5,65 pp; Clean bin < 4 px tại α ½ thấp hơn −1,63 pp. Bin nhỏ, có thể nhiễu (chưa tính CI).
- **21/11, Claude:** user yêu cầu chẩn đoán gốc rễ, research gap và 2–3 thiết kế mới, **chưa code notebook, chưa chạy GPU**. Mix A v1 giữ nguyên làm mốc âm; báo cáo cũ không sửa.
  - **Tài liệu:** `docs/research_v2_design_20261121.md`, gồm chẩn đoán E1–E12, gap, 3 thiết kế, pilot cho thiết kế 1 và bảng Integrity Gate 7 mode.
  - **Chẩn đoán chính:** B-Clean và B1 là hai chuyên gia theo điều kiện; trộn trọng số chỉ đi dọc đường nối hai bên (Clean giảm gần tuyến tính, LR2 lồi nhẹ +0,5–0,7 pp nhưng không vượt B1). Không phải lỗi BN, cũng không phải lỗi chọn hậu xử lý. Giả định của WiSE-FT không khớp vì hai endpoint cùng gốc B0.
  - **Bằng chứng mới, khám phá trên B** (`results/diag_v2_cpu/partial_from_csv.json`): định tuyến theo điều kiện (B-Clean cho Clean, B1 cho LR2) đạt mean 78,93, hơn B-Clean +1,06, hơn B1 +0,82, hơn α tốt nhất +0,58 pp. Đây là cận tham chiếu vì giả định biết điều kiện.
  - **Ba thiết kế:** (1) QRoute: router size-free chọn chuyên gia; (2) DualAgree: kết hợp bất đối xứng hai chuyên gia trên LR2; (3) SRVerify: góc nhìn EDSR cộng kiểm chứng đa tỉ lệ. **Chọn (1) để pilot, chỉ dùng CPU.**
  - **Đính chính 22/11 (bản sửa lần 2, chi tiết ở `docs/audit_v2_vs_framework_20261122.md`):**
    - **Khung:** file mà master prompt trỏ tới (`LABB\workspace\h_ng_d_n_th_c_chi_n_quy_tr_nh_nghi_n_c_u_ai_ocr.md`) **vẫn không có trên máy**. `LABB\workspace\Step by step research.md` có nội dung phù hợp (tiêu đề "Hướng Dẫn Thực Chiến Quy Trình Nghiên Cứu Khoa Học AI/OCR"), nhưng **chưa xác nhận là file thay thế**; bản 22/11 trước ghi như đã xác nhận là quá mức. Cần user xác nhận. Trong lúc chờ gọi là "khung ứng viên".
    - **Đối chiếu v2 với khung ứng viên:** có Data/Baseline insight, giả thuyết, ablation (kế hoạch); **thiếu Bước 5.5** (sanity 200–300 ảnh); Bước 4 chỉ dựa counts (human review 0/160); gap chưa tách core/sub-gap. Ba đề xuất chờ duyệt ở mục 2 của file audit.
    - **Skill:** `research-ideation`, `model-architecture`, `mechanistic-interpretability` là **tên nhóm**; ánh xạ lấy từ `~/.claude/skills/0-autoresearch-skill/references/skill-routing.md`. `brainstorming-research-ideas` **đã đọc đủ và áp dụng** (lọc Converge lên 3 thiết kế, không đổi lựa chọn). `creative-thinking-for-research`, `nnsight`, `pyvene` mới đọc phần đầu, chưa áp dụng; `transformer-lens`, `saelens`, `mamba`, `rwkv`, `litgpt`, `nanogpt` **chỉ xác nhận có cài**; nhóm `model-architecture` không có skill về detector.
    - `/audit-analysis`, `/review-paper-code` (Nhóm B) nằm ở `LABB/.claude/skills/`, không có trong project này.
- **Kết luận cũ giữ nguyên:** Mix A v1 chưa vượt baseline (`docs/report_mix_a_v1.md`); E-sel không có ứng viên; E-BN Giai đoạn 0 không có tín hiệu, không mở Giai đoạn 1.
- **Repo Tuấn hiện tại:** `96612b4`, đã fetch/fast-forward 22/11; xem cập nhật mới nhất phía trên.
- **Test:** 70/70 PASS (chỉ xác nhận phần mềm). Git workspace: nhiều file chưa commit.

| Phần mình | Hiện trạng | Còn thiếu |
|---|---|---|
| Mix A v1 + E-sel + E-BN-0 | Đóng; báo cáo âm | User gửi Tuấn |
| Thiết kế v2 | `docs/research_v2_design_20261121.md` (chỉ thiết kế; đã vá theo review + P0 ngày 22/11) | User/Tuấn trả lời 4 câu hỏi ở §3; duyệt ngưỡng Pass/Fail |
| Pilot QRoute (CPU) | P0 sửa lỗi xong: oracle trên A chỉ +0,27 pp (< +0,5) | Quyết định: dừng QRoute mức điều kiện, hay đổi mục tiêu (router theo từng ảnh / bàn lại cổng với Tuấn) |
| Mix B | Có code CPU, chưa train; latent 1k/10k đã âm | Cần chốt lại căn cứ và đối chứng trước khi mở |

## BƯỚC TIẾP THEO (theo thứ tự)

- **Ưu tiên 24/11:** đọc `docs/KE_HOACH_BOX_QUALITY_20261124.md`, xác minh TRAIN/GT và chuẩn bị phép smoke. Đây là hướng mới được đề xuất, không phải thí nghiệm đã khóa hoặc có kết quả. Các bước 23/11 dưới đây là lịch sử.

- **Tiếp (23/11 khuya, sau chẩn đoán):** user chọn (a) dừng Flip-TTA (khuyến nghị) hoặc (b) duyệt lượt chẩn đoán P-map tối thiểu (`docs/fliptta_diagnosis_plan_20261123.md` §6; Claude chuẩn bị notebook sau khi duyệt). Không chạy biến thể nào trước chẩn đoán. Song song: cập nhật tin cho Tuấn (Clean CS-B1 + Flip-TTA), K01–K10. Prompt tiếp tục: `docs/PROMPT_TIEP_TUC_20261123.md`.
- **Tiếp (23/11, sau Flip-TTA):** Flip-TTA = `INCONCLUSIVE`. User quyết định: dừng ở mức "chưa đủ bằng chứng", hay đăng ký trước một lượt lớn hơn (spec mới, khóa trước, hỏi Tuấn), hay đưa vào đăng ký benchmark cuối. Cập nhật tin cho Tuấn (Clean CS-B1 + Flip-TTA) trước khi gửi.
- **Tiếp (23/11 tối):** Clean CS-B1 đã tiếp nhận. (1) cập nhật số Clean vào `docs/msg_tuan_20261123_csb1.md` trước khi user gửi; (2) user chạy `kaggle_fliptta_pilot.ipynb`: dừng sau Ô 4 nếu kiểm căn chỉnh sai hoặc dự kiến > 3 giờ; (3) Claude tiếp nhận, báo nhãn theo luật khóa, không chỉnh. XV-B1 đã đóng (không mở lại). Không có B/benchmark cuối để chọn lại luật.
- **Session mới (từ 23/11):** đọc `docs/SESSION_START_20261123.md`. (1) Nếu user đã chạy notebook Clean thì tiếp nhận theo §2; (2) trình bày §4 để user chọn hướng thứ hai, chưa triển khai khi user chưa duyệt spec.
- **23/11 (sau rà soát):** (1) user duyệt và gửi tin cho Tuấn (`docs/msg_tuan_20261123_csb1.md`); (2) chấm K01–K10; (3) tùy chọn chạy `kaggle_csb1_clean.ipynb`; (4) không mở hướng mới khi chưa có ý tưởng khóa trước và tập đánh giá chưa dùng.
- **23/11 (roadmap):** user/Tuấn hỏi thầy tiêu chí "cải tiến"; user quyết định có làm EDSR ×2 ở `.40/.60` + Clean trên Kaggle không (phải hỏi Tuấn để tránh trùng); chấm K01–K10 để khóa CS-B1.
- **23/11:** (1) user chấm K01–K10 để hoàn tất Phase 0 CS-B1 (latency đã có: 1,99×); (2) quyết định có báo Tuấn về độ nhạy của Hmean A với ảnh thiếu nhãn (top 10 ảnh = 48,3% FP của B1) hay không; (3) nhánh zoom đã dừng theo luật, chỉ mở lại nếu có spec mới được duyệt.
- **Ưu tiên mới (22/11 khuya):** user chạy `kaggle_zoom_pilot.ipynb`. Sau Ô 5, đọc kiểm tọa độ và thời gian dự kiến; quá 4 giờ thì dừng và báo. Sau đó chạy Ô 6–8, gửi zip; Claude đọc `zoom_result.json` và áp cổng G2 đã khóa. Song song: Kaggle latency CS-B1 và chấm K01–K10 vẫn đang chờ.
- **22/11 tối, ưu tiên:**
  1. **User:** tải `cs431-csb1-latency.zip` lên Kaggle Dataset, import `kaggle_csb1_latency.ipynb`, bật GPU T4 + Internet, chạy Ô 1–6, gửi lại zip kết quả. Hướng dẫn ở `docs/HUONG_DAN_KAGGLE_csb1_latency.md`.
  2. **User:** chấm K01–K10 theo `results/phase0/review/HUONG_DAN_CHAM.md`, gửi lại `calibration_sheet.csv`.
  3. **Claude:** đối chiếu nhãn hiệu chuẩn. Thống nhất cách chấm xong mới gửi C01–C70 (đã tạo sẵn, không chọn lại).
  4. **Claude:** đọc `latency.json`, viết phần chi phí. Tổng hợp nhãn bằng trọng số `population_weight`, CI theo nguồn.
  5. **Claude:** áp bảng quyết định mục 7 (không đổi ngưỡng), soạn `docs/phase0_csb1_conclusion.md`. **User duyệt** lựa chọn giữa B và D (cả hai đều đạt điều kiện tự động).
- **Còn 2 ngày tới hạn 24/11 (R1 đã xong, R2 bàn giao chưa làm).** Việc chặn duy nhất của mình là quyết định của user ở bước 0 bên dưới (duyệt Phase 0 / 120 hay 80 thẻ; chạy Ô 8 trên T4). Nếu user không duyệt kịp: R2 bàn giao bằng báo cáo Mix A v1 + DualAgree/CS-B1 ghi rõ là khám phá PENDING_REFERENCE_CHECK. Tiếp tục fetch repo Tuấn mỗi session để bắt kết quả fusion.
- **Cập nhật 22/11:** không chờ latent 10k nữa. Theo dõi artifact fusion mới của Tuấn khi được chia sẻ; phối hợp hướng output DualAgree của Trí với hướng FPNC của Tuấn. Không tự submit hoặc coi fusion đã có kết quả.

0. **User duyệt kế hoạch Phase 0** (`docs/phase0_csb1_plan_20261122.md`: bảng quyết định mục 7, 120 hay 80 thẻ), tải zip mới, chạy Ô 1–4 + Ô 8 trên T4. Sau khi duyệt: Claude làm bước 1–2 và 4 của lịch N1 (CPU).
0a. (cũ) **User chạy DualAgree:** chạy `python scripts/make_colab_upload.py`, tải `mixab_code.zip` lên `MyDrive/CS431_MixA/upload/` (thay file cũ), mở `notebooks/colab_dualagree_v1.ipynb` trên Colab A100, chạy Ô 1 → Ô 7. Drive H: đang mount nên Claude có thể đọc kết quả trực tiếp. Tuấn vẫn chưa trả lời câu 2 checkpoint.
0b. (cũ) **User quyết định hướng sau P0b:** (a) chuyển trọng tâm sang bổ sung trong ảnh LR2 (Thiết kế 2), bước đầu là một lần GPU lưu box + score của B-Clean và B1 trên A/B LR2 (user chạy, T4/A100); hoặc (b) dừng QRoute, ghi kết quả âm. Câu hỏi 1 cho Tuấn (cho phép 2 checkpoint) vẫn chặn cả hai.
1. **User quyết định hướng QRoute** sau kết quả P0 sửa lỗi: trên A, oracle điều kiện chỉ +0,27 pp, không đạt +0,5 pp. Không được nới cổng sau khi xem số; nếu muốn đổi cổng phải ghi rõ là đổi sau khi thấy A và hỏi Tuấn.
2. ~~Sửa tài liệu v2 theo review 6 điểm~~ **Đã xong 22/11** (xem mục "Mới nhất"). Còn lại từ đối chiếu khung (chưa làm, chờ duyệt, `docs/audit_v2_vs_framework_20261122.md` §2): thêm Bước 5.5 (sanity 200–300 ảnh) vào P1; tách §1.4 thành core gap + sub-gap; đối chiếu E9/E10 với overlay khi có nhãn Bước 4.
3. **User/Tuấn trả lời 4 câu hỏi** ở §3 của tài liệu thiết kế v2:
   - cho phép hệ thống 2 checkpoint hay bắt buộc một model;
   - ảnh LR khi đánh giá có luôn là native LR2 không;
   - có duyệt ngưỡng Pass/Fail không;
   - chạy song song với latent 10k hay chờ.
   - Câu 2 giờ là câu quyết định: trên cohort và quy trình tạo LR2 hiện tại, kích thước ảnh tách Clean/LR2 (chỉ đúng cho quy trình này, không phải kết luận chung).
4. ~~Cập nhật §1/§4 tài liệu v2 theo P0~~ **Đã xong 22/11**, gộp trong lần sửa theo review 6 điểm.
5. **Nếu được duyệt:** viết code P1–P4 (đặc trưng router size-free, đối chứng nén JPEG, routed counts, CI) và test CPU. Chưa cần GPU.
4. Gửi Tuấn báo cáo Mix A v1 (`docs/msg_tuan_20261121_close_mixA.md`) nếu chưa gửi.

## ĐIỂM CHỜ XÁC NHẬN

| Điểm | Trạng thái |
|---|---|
| Hướng Mix A trước, Mix B có điều kiện | Tuấn ủng hộ theo tin nhắn user chuyển lại; không suy ra là đã duyệt toàn bộ config |
| Output version, ngưỡng +0,5 pp, protocol R1 A→B | Cần khóa. README của bundle ghi gói này chưa chạy R1 A→B |
| **Mốc tiny** | **Đã chốt 20/11:** α\* ≥ B1 − 3 pp trên cùng split B (thay 42%) |
| **Cổng 3a/3b khi chạy trên A100 thay vì H200** | **Đã chốt (a)** ngày 20/11 (user chuyển lời Tuấn). Cách triển khai (P-map + `gate_same_gpu`; H200 → `cross_gpu_drift`) là đề xuất của mình, ghi ở §4.3; Tuấn chưa duyệt chi tiết |
| Human review Bước 4 | PENDING, 0/160 thẻ có nhãn; review hiện ở `.30/.40` |
| Mix B: thân R, padding, scheduler, tiêu chí, phân công controls | Chưa thống nhất; giữ init B1 |

## SỰ THẬT ĐÃ KIỂM VÀ ĐƯỜNG DẪN

- **Cách cài môi trường (kiểm 20/11):** hướng dẫn chính thức ở `MonkeyOCRv2/detection/mmocr/README.md` + `install.sh` @ `da96afed…`.
  - Python 3.11, torch 2.9.0 cu128, mmengine 0.10.7, **MMCV 2.0.1 build từ source** (đổi c++14 thành c++17).
  - `install.sh` clone MMOCR v1.0.1 rồi chép add-on (`MonkeyOCRv2ViTAEBackbone`; patch chỉ đụng `pse_postprocessor`), cài mmdet 3.1.0 + transformers 4.57.1 + safetensors 0.7.0, và vá imgaug (NumPy 2) cùng mmengine (`torch.load`).
  - Trên HPC, DBPostprocessor của Tuấn nằm trong `upstream_MonkeyOCRv2/detection/mmocr/mmocr/...` (`sweep_run_identity.json`).
  - **File `db_postprocessor.py` của MMOCR v1.0.1 gốc trên GitHub có SHA `9bfff8aa…`, trùng bản Tuấn ghim** (tải bằng curl rồi hash).
- **Nguồn tải (kiểm 20/11):**
  - TextVQA `https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip`: 200 OK, 7.072.297.970 byte. Đọc mục lục từ xa bằng HTTP Range: tên file dạng `train_images/<id>.jpg`, khớp `clean_relpath`.
  - Hugging Face `zenosai/MonkeyOCRv2-AS`: có `model.safetensors`, `config.json` và code.
  - Hai repo GitHub đều public. Weights nằm trong Git LFS (`docs/handoff/weights/*.pth`).

- **Bundle:** `../../MONKEY_V2/docs/handoff/artifacts/pr_quad_audit_v1/`.
  - NPZ `(2262, 2, 72, 5)`: cond 0 = Clean, 1 = LR2; các cột `tp, gt, pred, tiny_gt, tiny_tp`.
  - `eda_common_counts.csv.gz`: `model, source_id, condition, sample_id, tp, gt, pred, covered, mask_thr, min_text_score`. Cổng 3a của mình cho PASS trên counts của Tuấn (4.524/4.524 mỗi endpoint).
  - `tasks_metadata.csv`: 4.524 task. `path_base` thuộc `TextVQA_clean_root` hoặc `prepared_root`; LR2 nằm ở `validation_lr2/<id>.png`.
  - `validation_remaining.jsonl.gz`: SHA-256 sau giải nén là `052857fb…` (mình đã kiểm trong `load_rows`).
- **SHA-256 export:** B-Clean `94f0e4ec…6aa`, B1 `9c3814e4…147`. Không dùng lẫn với hash checkpoint runner gốc.
- **Jobs của Tuấn:** 54308 FAILED sau sweep (01:06:37 trên H200, 3 model); recovery 63985 COMPLETED (00:04:07). Không dùng các con số này để ước lượng T4/A100.
- **Baseline toàn cohort** (bảng §0, mình đã tính lại khớp): B0 idx 44, B-Clean idx 70, B1 idx 70; B1 LR2 76,90%, FP 11.997, tiny 45,79%.
- **Các hàm MONKEY_V2 được gọi lại:**
  - `rev4.stage_a.detector_for/forward`: chỉ cần `model_config` và `checkpoint`; yêu cầu đúng 1 CUDA device và FP32.
  - `rev4.decision_core.capture_candidates/CandidateMatcher`: `score()` trả 4 bin.
  - `rev4.decision_plan.GEOMETRIES/SCORES`.
  - `rev4.training_data.native_lr2`: Pillow 11.3.0.
- Môi trường local: Python 3.14.4, torch CPU. Lệnh `--help` của các CLI đã sửa để chạy được trên console cp1252.

## CẤU TRÚC WORKSPACE

| Đường dẫn | Vai trò |
|---|---|
| `mixab/` | `weight_merge` (A1), `merge_audit` (A2), `reselect_split` (A3), `fsr` (Mix B), `monkey` (đường dẫn + commit MONKEY_V2) |
| `tests/` | 39 test; có test chạy trên bundle thật (tự skip nếu không có bundle) |
| `configs/` | Spec chạy, ví dụ `baseline_ab_v1.json` |
| `results/` | Kết quả nhỏ có ghi chú, sẽ đưa vào repo (ví dụ `baseline_ab_v1/`) |
| `docs/RUNBOOK_mix_a_T4_A100.md` | Hướng dẫn chạy trên GPU của user |
| `runtime/` | Trọng số và output nặng, **không** đưa vào repo |
| `HANDOFF.md` · `WORKLOG.md` · kế hoạch chính · `dashboard/` | Điều phối |
| `_archive/` | Lịch sử, không phải yêu cầu triển khai |
| `Đồ án/CLAUDE.md` + `Đồ án/.claude/` | Quy tắc và hook của session |
