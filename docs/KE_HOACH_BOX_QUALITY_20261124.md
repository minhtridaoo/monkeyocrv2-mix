# Hướng tiếp theo: học chấm chất lượng box với một forward B1

Ngày 24/11/2026 · Codex · **ĐỀ XUẤT VÀ CHUẨN BỊ DỮ LIỆU, chưa train/chưa chạy GPU/chưa có mức tăng.**

## 1. Quyết định và giới hạn

User muốn tiếp tục tìm một phương án có cải thiện có bằng chứng. Đề xuất ưu tiên một nghiên cứu khả thi về **learned box-quality scoring** thay vì thêm một luật veto hoặc TTA. Không mở lại kết quả cũ, không nới cổng. Đây là adaptation của kỹ thuật đã có, chưa phải ý tưởng mới cấp nghiên cứu và chưa chắc được thầy tính là đóng góp độc lập.

Không có cách hứa kết quả dương. Có thể tiếp tục phát triển hợp lệ trên development đã xem nếu công bố đúng quá trình; không phải cứ đã xem validation là cấm nghiên cứu tiếp. Điều không hợp lệ là gọi nó TEST độc lập, thay luật cũ sau khi xem số, hoặc giấu các thử nghiệm thất bại.

Hôm nay đã là hạn 24/11 được ghi trong workspace. Không cam kết train và xác nhận một phương pháp mới trong ngày. Nếu hạn không thể lùi, báo cáo hiện có cần chốt trung thực; nghiên cứu này là bổ sung, không phải kết quả đã đạt.

## 2. Căn cứ đã kiểm trực tiếp

- `../../MONKEY_V2/rev4/b1_training.py`: B1 có sampler cân bằng Clean/LR2; `training_data.py` RECIPE đặt clean_replay=0.5. Không đề xuất lại replay 50/50 như đóng góp mới.
- `../../MONKEY_V2/docs/handoff/artifacts/pr_quad_audit_v1/provenance/installed_db_postprocessor.py:91-156`: score lấy từ mean P-map trong polygon xấp xỉ trước unclip, sau đó lọc score, unclip, minAreaRect/min width. Nó không phải dự đoán IoU được huấn luyện riêng.
- `docs/fliptta_diagnosis_plan_20261123.md`: giảm FP qua score có thể đánh đổi TP; chưa có P-map lưu để xác định tín hiệu phân biệt. Đây là động cơ kiểm chứng, không chứng minh scorer mới sẽ hoạt động.
- `../../MONKEY_V2/docs/handoff/data/train_ids.csv` và receipt: 15.222 ID TRAIN, chỉ membership/hash/relative path, KHÔNG có GT hoặc ảnh.
- Đã kiểm 15.222 ID unique, không trùng 2.262 ID A/B (`results/baseline_ab_v1/split_manifest.csv`). Receipt mới ở `results/boxquality_preparation_20261124/receipt.json`.
- Có danh sách DỰ KIẾN 2.000 nguồn TRAIN theo hash cố định: 1.600 fit, 400 calibration. Hai phiên bản Clean/LR2 của một nguồn luôn cùng fold. Đây vẫn là dữ liệu B1 từng train; 400 nguồn không phải holdout độc lập của detector.
- Git fetch repo Tuấn: không có commit mới ở main..origin/main; có scripts/ untracked, không sửa. Không sửa repository Tuấn.
- Drive H không xuất hiện trong Get-PSDrive lúc kiểm. Chưa xác minh đủ ảnh TRAIN/GT ở cache khác; không tuyên bố dataset đã sẵn sàng.

## 3. Giả thuyết và pipeline

Giả thuyết: **phân bố xác suất trong một vùng chữ và hình dạng contour bổ sung thông tin về chất lượng box ngoài raw mean score**, cho phép giữ TP tốt hơn một ngưỡng chung ở cùng FP. Nếu không vượt đối chứng chỉ dùng raw score, dừng hướng.

Ảnh → B1 (một forward, frozen) → P-map → contour mask .40 và geometry cũ → vector đặc trưng → scorer nhỏ → ngưỡng chọn trên TRAIN-calibration → box → evaluator không đổi.

Vòng đầu giữ proposal generation/geometry cố định. Không sửa box, không NMS mới, không gộp hai checkpoint, không feature backbone mới. Có thể dump ứng viên raw≥.20 như harness hiện có; đây là pool dùng chung cho mọi đối chứng. Cần đo chi phí xuất thêm ứng viên, không mặc định miễn phí.

Khác calibration Tuấn: calibration thay ngưỡng trên score cũ; hướng này HỌC thứ tự chất lượng từ thông tin khác, dùng TRAIN. Khác XV-B1: không lấy sự vắng mặt của model thứ hai làm phủ quyết. Tuy nhiên nó vẫn thuộc nhóm hậu xử lý học được, không phải kiến trúc detector mới.

## 4. Đặc trưng và model tối thiểu

Đề xuất vòng đầu, cần khóa công thức sau khi audit implementation, trước khi chấm development:

1. Raw mean score hiện có (đối chứng).
2. Std và quantile 10/50/90 của xác suất trong polygon trước unclip.
3. Mean xác suất vùng nằm giữa polygon trước unclip và quad cuối (dùng cùng P-map, mask đúng tọa độ).
4. Tỷ lệ diện tích contour/quad và tỷ lệ contour/convex hull; định nghĩa rõ contour nào, xử lý rỗng/degenerate.
5. Aspect ratio và short side chuẩn hóa theo kích thước tensor detector; không đưa nhãn condition, ID ảnh, kích thước file hay GT vào model.

Không tự coi tỉ lệ kích thước là tín hiệu tốt: đã có hướng size-threshold thất bại. Phải so scorer chỉ dùng raw score với scorer đầy đủ để chứng minh thông tin bổ sung.

Model đầu: standardized logistic regression L2 (regularization cố định, không quét), nhắm xác suất box có overlap đạt điều kiện định vị. Đây là proxy dùng để học; đánh giá cuối vẫn phải matching chính thức. Không gọi proxy label là TP chính thức.

Không chọn đồng thời MLP, tree và logistic rồi lấy cái thắng. Nếu logistic không có tín hiệu, báo cơ chế/độ tách trước khi đề xuất thêm capacity.

## 5. GT, ignore và thiếu nhãn — điều kiện mở thí nghiệm

- Lấy GT TRAIN từ manifest/annotation gốc khớp pipeline Tuấn, không lấy A/B để fit. Giữ polygon, ROI, ignore và coordinate transform đúng.
- Dự kiến nhãn localization: positive max IoU≥.5 tới GT hợp lệ; negative có overlap GT hợp lệ nhưng max IoU<.5. Box không chồng GT không tự động được gán là nền vì rủi ro chữ thiếu nhãn.
- Trước khi chốt chế độ background negative: lấy một mẫu TRAIN cố định, phân tầng theo score/size, xem ảnh để biết lượng chữ thiếu nhãn. Nếu không xác định được negative sạch, bỏ nhóm ambiguous khỏi loss và ghi rõ model học localization, không khẳng định có thể lọc background.
- Hỏi/ghi lại nhãn không chắc, không tự biến tất cả "không GT" thành negative; không sửa evaluator hay GT validation. Mẫu review phải giữ đại diện chữ nhỏ, không chỉ nền dễ.
- Duplicates có IoU cao không được gọi là true positive độc lập. Scorer không giải quyết duplicate assignment nếu không có cơ chế riêng; không mở thêm cơ chế trong vòng đầu.
- Mỗi nguồn có trọng số bằng nhau; cân bằng Clean/LR2 trong nguồn để ảnh báo nhiều box không thống trị loss. Nếu sampling box, lưu xác suất/weight và quy tắc trước.

Nếu nhãn hoặc đặc trưng không đủ tin cậy, kết luận BLOCKED_DATA hoặc NO_SIGNAL, không cố chạy cả dev để kiếm mức tăng.

## 6. Các bước thực hiện và output

### P0 — đã làm một phần, CPU
- Đã kiểm source membership và tạo `proposed_train_sources.csv` cùng receipt; chưa khóa trial, chưa xác minh image bytes hoặc GT.
- Việc tiếp: tìm ảnh/GT trong tài nguyên đã có, kiểm SHA, leakage theo source và hash ảnh, kiểm ignore/ROI; không đọc TEST.
- Kiểm khả năng tái dùng `_get_bbox_score`/postprocessor ở runtime. Không chỉnh bản cài thuộc repo Tuấn; thêm adapter ở workspace của Trí.
- Chuẩn bị notebook có các ô: doctor → kiểm input → smoke 10 nguồn → extract TRAIN → fit/calibrate → xuất artifact. Development là bước sau, không tự chạy trong lần fit.

### P1 — smoke GPU 10 nguồn, user chạy
- Kiểm B1 F .40/.60 khớp baseline trong cùng run, coordinate mapping, finite P-map/features, replay từ dump cho counts trùng.
- Thu profile extract và VRAM, ước lượng tổng từ đo thật. Tách thời gian chuẩn bị/tải ảnh. Không hứa 1–2 giờ trước khi đo.
- Nếu không đủ tài nguyên/ảnh/GT, dừng và sửa lỗi kỹ thuật; không thay phương pháp theo điểm dev.

### P2 — TRAIN feasibility, 2.000 nguồn đề xuất
- 1.600 fit/400 calibration theo CSV mới. B1 đã thấy các nguồn khi train; sẽ có optimism của proposal distribution, phải ghi rõ.
- Dùng cùng candidates cho raw-score control, score-only learned control và full-feature scorer.
- Chọn một cutoff trên calibration bằng mean Hmean Clean/LR2, ràng buộc tiny; quy tắc tie-break cố định. Không so unfair với baseline chưa được chọn cutoff trên cùng calibration.
- Báo precision–recall, TP/FP/FN và mức cải thiện so với best raw-score threshold trên calibration. Calibration không phải xác nhận.
- Nếu full scorer không cho lợi ích hữu ích ngoài raw-score, dừng trước khi chấm A.
- Lưu model/normalizer/features/schema/manifest/hash/threshold. Không dùng filename hoặc condition để chọn cutoff khi inference.

### P3 — development một lần sau khóa
- Đánh giá F chuẩn .40/.60, raw-score baseline chọn trên TRAIN-calibration, và scorer đã khóa trên đủ A, cả Clean/LR2; không chỉ chọn lại 300 ảnh thuận lợi.
- Counts theo evaluator chính thức; bootstrap ghép cặp theo nguồn và giữ Clean/LR2 cùng nguồn. Báo cả Hmean từng condition, mean hệ thống, tiny, per-source, worst cases.
- Đối chiếu đầy đủ cổng nhóm từ protocol hiện tại: pilot LR2 +.30 pp với CI lower>0 chỉ là screening; cổng hệ thống mean +.50 so model đơn tốt nhất và Clean≥B-Clean−.30, tiny≥B1−3 cần tính trên cùng split. Không lấy số ALL để trừ số A.
- Clean gate khó: toàn cohort B1~79.44 và B-Clean~80.84 → để cách B-Clean≤.30 cần ~80.54, tức B1 cần tăng ~1.10 pp. Đây là minh họa ALL, không thay số A. Vì vậy +.30 LR2 không đủ để gọi thành công hệ thống.
- Nếu đạt development, mới thống nhất với Tuấn việc đánh giá B/benchmark cuối; B là development đã xem. Không mở TEST để chọn biến thể hoặc cutoff.

### P4 — latency cùng GPU
- Đo F và scorer với cùng decode/preprocess/precision/batch/warmup; include transfer P-map, CPU feature extraction, mọi candidates. Alternate order, median/p95/peak memory.
- Một forward chỉ giảm rủi ro chi phí 2×, không bảo đảm overhead thấp: ảnh dày chữ có thể rất chậm trên CPU.
- Mục tiêu thiết kế đề xuất ≤1.10× median F; đây chưa phải cổng nhóm đã chấp nhận, cần khóa trước run. Nếu quality có lợi mà overhead cao, chỉ tối ưu thực thi giữ phép tính rồi đo lại; không thay luật để che trade-off.

## 7. Căn cứ học thuật và điều chưa được chứng minh

- Huang et al., Mask Scoring R-CNN, CVPR 2019: abstract nói classification score chưa tương quan tốt với mask IoU và học mask quality. https://openaccess.thecvf.com/content_CVPR_2019/html/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.html
- Jiang et al., Acquisition of Localization Confidence for Accurate Object Detection, ECCV 2018: hướng học độ tin cậy định vị. https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Borui_Jiang_Acquisition_of_Localization_ECCV_2018_paper.php

Chỉ dùng ý tưởng ở mức abstract, KHÔNG khẳng định tái hiện kiến trúc hoặc kết quả của paper. Hai paper không chứng minh logistic/P-map statistics có hiệu quả cho MonkeyOCRv2. Bằng chứng đó phải do thí nghiệm này tạo ra.

## 8. Bàn giao Claude

Đọc tài liệu này cùng kế hoạch chính và giữ lịch sử. User đã yêu cầu tìm hướng mới; kế hoạch này thay vì lặp thêm TTA. Bước kế là P0: xác minh ảnh và GT TRAIN, kiểm tính khả thi của feature/label, rồi chuẩn bị notebook smoke và spec chính xác. User là người chạy GPU. Chưa train detector, chưa gọi hướng này là đạt cải tiến. Không tự rẽ sang loss/module khác nếu scorer thất bại.

Không còn bằng chứng rằng chỉ luật hậu xử lý sẽ đủ. Nếu hướng này không có tín hiệu, bước tiếp phải được cân nhắc lại theo ngân sách và nguồn dữ liệu; không hứa thêm một phép thử là có kết quả tốt.
