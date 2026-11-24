# Thiết kế pilot E-BN: tính lại thống kê BatchNorm sau khi trộn (CHƯA CHẠY, chờ nhóm duyệt)

Thuộc kế hoạch §4.4. Đây là **thiết kế**: chưa viết code GPU, chưa sửa notebook, chưa chạy gì trên GPU.

## 0. Trạng thái triển khai (cập nhật 20/11)

- **Giai đoạn 0 đã có code và notebook, chưa chạy GPU:**
  - `mixab/bn_recal.py` có 3 lệnh: `check-train`, `prepare-images`, `stage0`.
  - `tests/test_bn_recal.py` có 10 test CPU, gồm mô phỏng Colab bị ngắt rồi chạy tiếp ra đúng kết quả.
  - Notebook riêng `notebooks/colab_bn_pilot_stage0.ipynb`, sinh bằng `scripts/build_bn_notebook.py`. Ô 1–4 chép nguyên từ notebook v1; notebook và kết quả v1 không đổi.
- **Khác so với bản thiết kế ban đầu:** Giai đoạn 0 **chỉ đo**. Model ở `eval()`, hook đọc đầu vào BN, không đặt `train()`, không ghi buffer, và kiểm SHA `state_dict` trước/sau. Cách tính lại BN ở §4 chỉ dùng cho Giai đoạn 1.
- **Giai đoạn 1 chưa có code** và chỉ được thêm khi đủ điều kiện ở §6–§8.
- **Đầu vào TRAIN.** Tuấn xác nhận ngày 20/11: SHA manifest `ca1b9163…c983d`, 15.222 nguồn FULL_R1, nhãn TextOCR, LR2 dùng `native_lr2`. Manifest đầy đủ chưa có bản tải. Dựng lại bằng `rev4/training_data.recover_train` hoặc `rev4/train_recovery.py` cần các file **không có** trong repo:
  1. metadata job 8323 `/datastore/dungmt/textvqa/Danh/results/nckh_restart_v1/phase_a_full_textvqa_seed42/phase_a_unified_manifest.csv` (SHA `a2f03bb1…`);
  2. repo cũ chứa `src/monkey_textdet` ở commit `816c6d6aa0cd15de50a13c3ddfa621aed872b623`;
  3. manifest validation đóng băng 3.262 nguồn (repo chỉ có 2.262 nguồn remaining).

  Annotation TextVQA train (SHA `95f5c407…`) và TextOCR train (SHA `5bac8303…`) là file công khai, nhưng không đủ nếu thiếu 1–3. **Vì vậy không tự dựng lại TRAIN.** Notebook nhận manifest đầy đủ (kiểm đúng SHA), hoặc danh sách ID/đường dẫn/SHA ảnh (ghi `NOT_VERIFIABLE_ID_LIST`).
- **Giới hạn kiểm trùng:** chỉ kiểm được không trùng với 2.262 nguồn remaining-validation. 1.000 nguồn S1/S2 chưa công bố.

- **Kết quả Giai đoạn 0 (20/11, T4):**
  - Không có tín hiệu: phi tuyến ≤ 0,005 SD trung bình; α ở giữa có z 0,157–0,159, trong khi mức endpoint là 0,161.
  - Phát hiện thêm: phương sai đo được bằng khoảng 0,45 lần phương sai đã lưu ở mọi model. Cập nhật BN theo §4 như đang viết (batch 1, `momentum=None`) có thể làm thay đổi đáng kể cả endpoint; chưa biết tốt hay xấu vì chưa chạy. Muốn làm thì phải thiết kế lại §4 (phương sai gộp hoặc batch giống lúc train) như một câu hỏi riêng.
  - **Chốt 21/11 (user): không mở Giai đoạn 1.** Chi tiết: `results/bn_stage0_v1/NOTES.md`.

## 1. Giả thuyết và căn cứ

- **Giả thuyết:** bản merge v1 nội suy thống kê BN theo đường thẳng (`merge_manifest.json`: `bn_statistics: interpolated, not re-estimated`). Sau khi trọng số thay đổi, phân bố activation thật có thể lệch khỏi thống kê nội suy. Tính lại thống kê có thể làm P-map của model trộn tốt hơn.
- **Căn cứ:** tài liệu PyTorch cho `AveragedModel` ghi rằng với model trung bình trọng số có BatchNorm, có thể cần cập nhật thống kê activation; `update_bn()` làm việc này bằng cách cho dữ liệu đi qua model. Cách nào tốt hơn cần thử trên bài toán cụ thể ([PyTorch docs](https://docs.pytorch.org/docs/2.15/generated/torch.optim.swa_utils.AveragedModel.html), đã đọc 20/11).
- Đây là **giả thuyết cần kiểm**, chưa phải nguyên nhân đã xác nhận trong dự án.

## 2. Sự thật đã kiểm, giới hạn phạm vi

- **Chỉ có 4 lớp BN trong cả model**, đều ở DBHead: `det_head.binarize.{1,4}` và `det_head.threshold.{1,4}`, mỗi lớp 64 kênh. Backbone ViTAE và FPNC **không có BN**. Kiểm bằng cách đọc key `running_mean` trong `alpha_0p00.pth` / `alpha_1p00.pth`.
- **Lúc inference chỉ nhánh `binarize` chạy.** Theo MMOCR v1.0.1:
  - `SingleStageTextDetector._forward` gọi `det_head(x, data_samples)` với mode mặc định `'predict'`.
  - Ở mode đó `DBHead.forward` chỉ tính `binarize` rồi trả `prob_map`; nhánh `threshold` chỉ chạy ở mode `loss`/`both`.
  - Config của Tuấn dùng `type='DBHead'`.
  - Vậy **chỉ 2 lớp BN, tổng 128 kênh, ảnh hưởng tới kết quả**.
- **Hai endpoint rất gần nhau:** ‖W_B1 − W_BClean‖ / ‖W_BClean‖ = 1,9%. Thống kê BN của hai endpoint khác nhau vừa phải:
  - `binarize.1`: lệch trung bình 0,13 SD (tối đa 0,64 SD), tỉ lệ phương sai 0,76–1,04.
  - `binarize.4`: lệch trung bình 0,01 SD, tỉ lệ phương sai 0,91–1,10.
- **Suy luận, chưa kiểm:** vì hai endpoint gần nhau, thống kê thật của model trộn có thể gần thống kê nội suy, nên hiệu ứng có thể nhỏ. Lưới 72 setting (score và mask threshold) cũng có thể hấp thụ một phần lệch hiệu chuẩn P-map. Kết hợp với kết quả E-sel (`results/explore_sel_v1/NOTES.md`), **kỳ vọng thấp**. Vì vậy pilot có bước chẩn đoán rẻ ở đầu (Giai đoạn 0) để dừng sớm.

## 3. Dữ liệu cập nhật BN: chỉ TRAIN

- **Chỉ dùng ảnh TRAIN** (15.222 nguồn của B1/B-Clean). Không dùng A/B (remaining-validation), không dùng TEST.
- **Đang bị chặn:** danh sách TRAIN **không có trong repo của Tuấn**.
  - `rev4/ssim_calibration.py:16–17` định nghĩa `manifest__train__textvqa_clean_degraded_same_textocr_word_gt.jsonl`, SHA-256 `ca1b9163…`.
  - Khôi phục file này cần repo cũ và annotation TextOCR trên HPC (`rev4/README.md:56–96`).
  - **Cần Tuấn chia sẻ** file manifest TRAIN, hoặc tối thiểu danh sách `source_image_id`, `clean_relpath`, `clean_sha256` của 15.222 nguồn.
- **Không tự chọn ảnh TextVQA khác:** có thể dính TEST hoặc validation.
- **Tập cố định:**
  - 1.000 nguồn TRAIN, chọn bằng seed 20261120 trên danh sách đã sắp xếp. Ảnh lấy từ zip TextVQA gốc, kiểm SHA từng ảnh.
  - Mỗi nguồn dùng cả Clean và native LR2 (`native_lr2` của Tuấn), tức 2.000 ảnh tỉ lệ 50/50, giống tỉ lệ huấn luyện của B1.
  - Cùng một tập cho mọi model.
  - Kiểm không trùng ID và không trùng SHA Clean với 3.262 nguồn validation, theo `rev4/training_data.check_split`.
- **Tiền xử lý:** đúng `test_pipeline` của config (resize giữ tỉ lệ, không augmentation), batch 1, FP32. Lý do: thống kê phải khớp phân bố đầu vào lúc inference.

## 4. Cách tính lại (giống `update_bn` nhưng hẹp hơn)

`torch.optim.swa_utils.update_bn` (đọc mã nguồn bản torch local) làm như sau: reset running stats của mọi `_BatchNorm`, đặt `momentum=None` để lấy trung bình tích lũy, gọi `model.train()` cho **toàn model**, rồi cho dữ liệu đi qua.

Pilot dùng cùng nguyên tắc với hai khác biệt, vì `model.train()` toàn bộ có thể bật dropout/drop-path trong ViTAE:
1. Model ở `eval()`. **Chỉ** `det_head.binarize.1` và `det_head.binarize.4` được đặt `train()`, `momentum=None` và `reset_running_stats()`.
   - Nhánh `threshold` giữ nguyên, vì không được cập nhật ở mode `predict` và không ảnh hưởng inference.
   - Trọng số, kể cả γ/β của BN, **không đổi**.
2. `torch.no_grad()`, forward mode `tensor` giống `stage_a.forward`, một ảnh mỗi batch. Thống kê là trung bình theo ảnh của mean/var từng batch, đúng cách `momentum=None` hoạt động.

## 5. Ma trận đối chứng

| Model | BN nội suy (v1) | BN tính lại trên TRAIN |
|---|---|---|
| B-Clean (α 0) | đối chứng (= v1) | **đối chứng endpoint** |
| α 1/4, 1/2, 3/4 | v1 | **xử lý** |
| B1 (α 1) | đối chứng (= v1) | **đối chứng endpoint** |

Hiệu ứng cần đọc là **chênh lệch trong chênh lệch**. Nếu tính lại BN giúp α ở giữa nhiều hơn hẳn mức nó giúp endpoint, đó mới là bằng chứng về "lệch BN do trộn". Nếu endpoint cũng tăng tương tự, đó là hiệu ứng hiệu chuẩn chung, không riêng gì merge.

## 6. Các giai đoạn và cổng dừng

**Giai đoạn 0: chẩn đoán, chỉ GPU forward, khoảng 15 phút trên A100 (ước lượng, chưa đo).**
- Với α 1/2, đo thống kê thật của 2 lớp BN `binarize` trên 2.000 ảnh TRAIN. So với thống kê nội suy: độ lệch trung bình chuẩn hóa |Δμ|/σ và tỉ lệ phương sai từng kênh.
- Đo tương tự cho hai endpoint (thống kê thật so với buffer đã lưu) để biết mức nền.
- **Đề xuất cổng, cần nhóm duyệt:** chỉ sang Giai đoạn 1 nếu lệch của α ở giữa vượt rõ mức nền của endpoint, ví dụ |Δμ|/σ trung bình lớn hơn 2 lần mức nền. Nếu không, dừng và ghi kết quả âm.

**Giai đoạn 1: đánh giá counts (chỉ khi qua Giai đoạn 0).**
- Mỗi ảnh đánh giá chỉ chạy backbone và neck **một lần**, rồi chạy DBHead hai lần: BN v1 và BN tính lại. Cách này giảm gần một nửa thời gian.
- **Cổng tái lập:** đường "BN v1" của code mới phải cho P-map trùng `stage_a.forward` (≤1e-6, 20 ảnh cố định như §4.3), và counts trùng A2r trên 40 nguồn probe.
- Lưới 72 setting như cũ, 5 α × 2 phiên bản BN, 2.262 nguồn × (Clean, LR2).
- Thời gian **chưa đo**. Ước lượng thô từ A2r (0,41–0,43 s/task): từ khoảng 2,7 giờ (nếu forward chiếm phần lớn) tới khoảng 5,4 giờ (nếu chấm 72 setting chiếm phần lớn). Sẽ probe 40 nguồn trước để đo.

## 7. Đọc kết quả, khóa trước khi chạy

- **Chỉ số chính, chỉ mô tả hiệu ứng:**
  - Δ mean Hmean (BN tính lại − BN v1) cho từng α, tại **setting đã khóa ở v1** của α đó.
  - Paired bootstrap theo source trên 2.262 nguồn (Clean/LR2 cùng source).
  - Báo kèm Δ của endpoint (chênh lệch trong chênh lệch), tiny <4 px, và FP ở recall tương đương (cách của E-sel).
- **Phụ:** chạy lại R1 và E-sel trên bộ counts mới, gắn nhãn khám phá.
- **Đề xuất ngưỡng "đáng xác nhận", cần nhóm duyệt trước khi chạy:**
  - Tồn tại α ở giữa có Δ mean ≥ +0,3 pp, CI không chứa 0.
  - Δ đó lớn hơn Δ của cả hai endpoint.
  - Tiny của α đó không thấp hơn B1 quá 3 pp.
- **Đạt ngưỡng chỉ có nghĩa là đáng xác nhận, không phải đã cải tiến.**

## 8. Vòng xác nhận (cần nhóm thống nhất)

A/B đã xem, nên chia lại dữ liệu này không tạo ra tập kiểm chứng mới. Nếu Giai đoạn 1 cho tín hiệu, nhóm cần chọn **trước** một trong các cách:
- (a) Đưa cấu hình đã khóa (α, cách tính BN, setting) vào đánh giá cuối Bước 8 của Tuấn (seed và TEST theo protocol của Tuấn). Không tune thêm.
- (b) Một tập validation khác chưa dùng, nếu nhóm có. **Chưa biết có hay không; cần hỏi Tuấn.**

## 9. Việc cần trước khi code hoặc chạy GPU

1. Tuấn chia sẻ manifest TRAIN, hoặc danh sách ID/SHA.
2. Nhóm duyệt: cổng Giai đoạn 0, ngưỡng ở §7, cách xác nhận ở §8.
3. Sau đó mới viết `mixab/bn_recal.py`, kèm test CPU trên module giả có BN để kiểm chỉ 2 lớp đổi, γ/β không đổi, threshold không đổi và momentum được khôi phục. Kế đó thêm ô Colab và probe 40 nguồn.

Ước lượng công: code + test khoảng 1,5 giờ; GPU khoảng 15 phút (Giai đoạn 0) và 3–6 giờ (Giai đoạn 1, chưa đo).
