# Spec khóa trước · Bổ sung Clean cho CS-B1@0.7 (23/11/2026)

> Đây chỉ là **đo bổ sung** cho một phương án đã có, **không phải phương pháp mới**. Cấu hình CS-B1@0.7 giữ nguyên (`mixab/dualagree_janek.py` `LOCKED`). Không quét ngưỡng, không đổi geometry.
> Dữ liệu là development (2.262 nguồn remaining-validation, đã xem nhiều lần). **Không thể xác nhận cải tiến** bằng lượt này.

## 1. Vì sao cần
- Dump hiện có chỉ có **LR2**. Mọi số CS-B1 trên Clean trước đây đều dùng B-Clean qua router kích thước ảnh. Router này là shortcut của quy trình tạo LR2 (QRoute, E13).
- Lượt này đo **CS-B1 áp cho mọi ảnh, không định tuyến**: Clean Hmean, TP/FP/FN, tiny. Từ đó có mean hệ thống không dùng shortcut.
- **Lượt này không giải quyết** việc trên LR2, CS-B1 so với B1 đã calibration (`.40/.60`) có CI chứa 0 (+0,18 [−0,21; +0,58] trên 2.262 nguồn). Đo thêm trên cùng dữ liệu không làm mất thiên lệch đã xem A/B.

## 2. Cố định trước
- **Ảnh:** đủ 2.262 nguồn Clean của bundle (`source_cohort.csv`). Ảnh nằm sẵn trong dataset (chép đúng byte từ cache Drive, kiểm SHA từng ảnh). Notebook kiểm SHA lại; nếu thiếu thì tải zip TextVQA chính thức (7.072.297.970 byte) làm dự phòng. Thiếu hoặc sai một ảnh thì dừng.
- **Checkpoint:** `alpha_0p00.pth` (B-Clean) và `alpha_1p00.pth` (B1), SHA theo `merge_manifest.json`. Đây là cùng bản dump LR2 đã dùng.
- **Pipeline:** `dualagree dump --condition clean` (9 geometry, raw ≥ 0,20, batch 1, FP32). MONKEY_V2 @ `dc19b85`; DBPostprocessor SHA `9bfff8aa…`; config FP32, chỉ vá đường dẫn encoder.
- **Kiểm trung thực:** probe 40 nguồn, so counts tính lại với counts Clean của A2r (A100). Chỉ báo tỉ lệ lệch do khác GPU; không chặn.
- **Đánh giá:** `mixab/csb1_clean_janek.py`. `CandidateMatcher` của Tuấn trên cùng dump cho mọi arm.
  - So CS-B1 với B1 `.40/.60`, B1 `epsilon_005/.50` và B-Clean `.40/.60`, trên ALL, A, B.
  - Báo Δ Clean và Δ LR2 kèm CI bootstrap ghép cặp theo nguồn (2.000 lần, seed 431), và mean hệ thống.
- **Ảnh lỗi / OOM:** dump dừng khi lỗi. Ô đánh giá yêu cầu đủ 2.262 nguồn; thiếu thì báo, **không âm thầm loại**.

## 3. Đọc kết quả (mô tả, không chọn lại)
- **Cổng nhóm, tính trên A, mô tả** (không còn là xác nhận):
  - mean hệ thống ≥ model đơn tốt nhất + 0,5 pp;
  - Clean ≥ B-Clean − 0,3 pp;
  - tiny ≥ B1 − 3 pp.
- Nếu Clean của CS-B1 thấp hơn B-Clean quá 0,3 pp: ghi rằng muốn giữ Clean thì CS-B1 cần router (shortcut).
- **Không** đề xuất TEST cho CS-B1 dựa trên lượt này. CS-B1 đã trượt luật dừng +0,3 pp trên A khi so với baseline đã khóa (B1 tốt nhất trong 72 setting = `epsilon_005/0.50`; Δ +0,177 pp).

## 4. Chi phí (ước lượng, chưa đo)
- Dump LR2 trước đây mất 26–27 phút/model trên Colab T4.
- Ảnh Clean cũng được resize về cạnh ngắn 1024, nên ước lượng khoảng 55 phút cho 2 model, cộng 5–40 phút cài môi trường, cộng khoảng 10 phút đánh giá CPU.
- Ô probe in thời gian đo thật.
