# Spec khóa trước · Learned box-quality scorer (B1, một forward) · 24/11/2026

> Dựa trên `docs/research/janek/KE_HOACH_BOX_QUALITY_20261124_janek.md` (Codex), đã sửa các điểm sai hoặc không khả thi ở mục 2. **Khóa trước mọi lượt GPU.**
> **Trạng thái:**
> - P0 đã kiểm (CPU, mục 1);
> - code, test, notebook, dataset đã chuẩn bị;
> - **chưa chạy GPU, chưa fit, chưa có số chất lượng**.
>
> Đây là hậu xử lý học được (adaptation của "học độ tin cậy định vị", Huang et al. CVPR 2019 và Jiang et al. ECCV 2018, chỉ ở mức abstract). Không phải kiến trúc mới, **chưa được gọi là cải tiến**.
> Fit và calibration đều trên **TRAIN** (B1 đã train trên đó). **Không** dùng A/B, **không** mở TEST.

## 1. P0: đã xác minh (CPU) · `docs/handoff/artifacts/janek/boxquality_p0_20261124/`
| Kiểm | Kết quả | Bằng chứng |
|---|---|---|
| TextOCR train JSON | SHA `5bac8303…` = hằng số của Tuấn (`rev4/train_recovery.py:20`); 21.778 ảnh, 1.052.354 annotation | `p0_janek.json` |
| Code nhãn (`gt_for_source`, vendored nguyên văn, SHA `6b2ecde7…`) | Với ROI đã khóa, tái tạo **2.262/2.262** `gt_clean` dev và **2.262/2.262** `removed_gt_ids` **chính xác** | `p0_janek.json`; `mixab/vendor/README_janek.md` |
| Commit legacy `816c6d6` | Không còn trong lịch sử local → chưa chứng minh trùng byte; chỉ chứng minh **tương đương chức năng** (dòng trên) | `git cat-file` thất bại |
| TRAIN membership | 15.222 ID unique; `train_ids.csv` SHA `8ac5cbd9…`; trùng dev **0** theo ID và **0** theo SHA ảnh; 15.222/15.222 có trong TextOCR; 0 nguồn thiếu GT valid | `p0_janek.json` |
| Ảnh TRAIN | Zip chính thức `train_val_images.zip` (7.072.297.970 byte) có 25.120 member; **2.010/2.010** ảnh được chọn có mặt (549 MiB). 10 ảnh smoke tải bằng HTTP Range: CRC + SHA-256 khớp `train_ids.csv`, kích thước khớp TextOCR | `zip_listing_janek.json`, `smoke_fetch_local_janek.json` |
| ROI | Trên dev, 881/2.262 nguồn có `roi_clean` < toàn ảnh (≥ 0,881). ROI lấy từ perspective của ảnh suy giảm lịch sử (CSV job 8323, chỉ có trên HPC) → **không dựng được ROI lịch sử cho TRAIN** | `shared_manifest.source_geometry`; `p0_janek.json` |
| Nhiễu nhãn do ROI = toàn ảnh (đo trên dev, không đổi nhãn dev) | 168/2.262 nguồn đổi trạng thái GT: 431 ignored → valid, 102 removed → valid, 111 removed → ignored; 424 GT valid đổi hình dạng. Tổng GT valid dev 78.221 | `p0_janek.json` |
| Ignore | Tỉ lệ GT ignore (chủ yếu "###" không đọc được): fit 32%, calibration 38% | `p0_janek.json` |
| Thiếu nhãn | Xem 3 ảnh smoke có GT vẽ chồng: TextOCR dày; ảnh đường phố có biển xanh trên tòa nhà không có nhãn (có thể thiếu nhãn). **Chưa định lượng** | `smoke_gt_overlays/` |
| Đặc trưng từ một forward | Hook `_get_bbox_score`/`_unclip` (y như `capture_candidates`) nhận đúng `score_map` và polygon trước unclip. Mean trong mask **trùng chính xác** score gốc (test so với bản chép nguyên văn của `_get_bbox_score`) | `tests/test_boxquality_janek.py` |
| Evaluator | `filtered_geometry` **bỏ hẳn** box có < 50% trong ROI hoặc > 50% chồng ignore (không tính Pred); box hỏng hình học tính là FP | `rev4/paired_eda_core.py:14-37` |

## 2. Sửa so với kế hoạch Codex (có bằng chứng)
1. **Danh sách 1.600/400 không tái tạo được.** Không có script/quy tắc ghi lại; các quy tắc hash phổ biến chỉ trùng ~250–280/2.000 (mức ngẫu nhiên). **Thay bằng quy tắc khóa** (mục 3). Trùng với danh sách Codex: 236/2.000.
2. **ROI TRAIN = toàn ảnh**, vì metadata lịch sử chỉ có trên HPC. Ảnh Clean/LR2 native không bị warp, nên toàn ảnh là trường nhìn thật. Độ lệch đã đo ở mục 1. **Dev giữ nguyên ROI đã khóa.**
3. **Nhãn mô phỏng đúng evaluator** thay vì loại box "không chồng GT". Lý do: evaluator tính các box đó là FP. Trong chẩn đoán Flip-TTA, box `no_gt` là lớp FP lớn nhất, nên loại chúng ra sẽ làm scorer "mù" với loại FP chính. Rủi ro thiếu nhãn được **báo** qua bảng review chọn theo hash, **không** đổi nhãn.
4. **Score-only logistic đơn điệu theo raw** (một đặc trưng, hệ số dương), nên tập box giữ lại ở mọi ngưỡng trùng với RAW. Nó là **kiểm tính hợp lệ của đối chứng** (phải trùng RAW trong 0,05 pp; lệch thì nhãn `INVALID_CONTROL`), không phải đối chứng độc lập thứ hai. Đối chứng thật là **RAW chọn ngưỡng trên cùng tập calibration**.
5. **`train_ids.csv` không có ở commit ghim `dc19b85`** (thêm ở `e4a2a89`), nên dataset mang theo file này, kiểm SHA.
6. **Ảnh:** không cần tải cả zip 7 GB. Tải từng member bằng HTTP Range, kiểm CRC-32 + SHA-256. Lỗi mạng thì dừng và báo; không tự đổi nguồn.
7. **Đặc trưng #2/#3 của Codex** định nghĩa chính xác trên **cùng mask** với `_get_bbox_score` (pre-unclip, cắt theo bbox), và vòng quanh = quad cuối trừ polygon trước unclip, trên lưới mạng.

## 3. Dữ liệu (khóa)
- Eligible = 15.222 nguồn TRAIN (đều có ≥ 1 GT valid).
- Thứ tự `sha256("20261124:boxquality-v1:" + id)`:
  - 10 đầu = **smoke** (không dùng để fit/calibration);
  - 2.000 kế tiếp = được chọn. Trong đó 400 có `sha256("20261124:boxquality-v1:cal:" + id)` nhỏ nhất = **calibration**, 1.600 còn lại = **fit**.
- `selection_janek.csv` SHA được ghi trong `p0_janek.json` (`selection_sha256 10772b42…`); `train_rows.jsonl_janek.gz` SHA `dbe1668d…`.
- Mỗi nguồn có Clean + LR2 (`native_lr2`, Pillow 11.3.0), luôn cùng fold.
- Calibration **không độc lập** với detector (B1 đã train trên TRAIN). Phân bố ứng viên trên TRAIN có thể lạc quan hơn dev.

## 4. Pipeline (khóa)
```
ảnh → B1 (frozen, FP32, batch 1, một forward) → P-map → DB post-processor của Tuấn, geometry mask_040 (mask .40, unclip 1,5, min width 5, eps .01), min_text_score = 0
    → ứng viên được chấp nhận, raw ≥ 0,20 (pool chung cho mọi đối chứng) + 13 đặc trưng
    → scorer → giữ box có score ≥ ngưỡng (chọn trên TRAIN-calibration) → evaluator không đổi
```
**Đặc trưng** (`FEATURES`, tính trên lưới mạng):
- raw, std, q10, q50, q90, max: lấy trong mask của `_get_bbox_score`;
- `ring_mean`: quad cuối trừ polygon trước unclip;
- `quad_mean`;
- `fill_quad` = diện tích polygon / diện tích quad;
- `fill_hull` = diện tích polygon / diện tích convex hull;
- `short_norm` = cạnh ngắn quad / cạnh ngắn tensor;
- `log_aspect`;
- `n_vertices`.

**Không** dùng condition, ID, kích thước file hay GT.

## 5. Nhãn và trọng số (khóa)
- `CandidateMatcher` của Tuấn trên row TRAIN (ROI = toàn ảnh).
- Nhãn từng ứng viên:
  - **1** nếu có cạnh IoU ≥ 0,5 tới GT valid;
  - **0** nếu được tính vào Pred mà không có cạnh đó (gồm cả hình học hỏng);
  - **bỏ** nếu evaluator lọc (ROI/ignore).
- Duplicate đều nhận 1 (proxy; matching cuối vẫn là chính thức).
- Trọng số: mỗi task (nguồn × điều kiện) tổng 0,5, nên mỗi nguồn tổng 1.

## 6. Model và đối chứng (khóa, không quét)
- **FULL:** logistic L2 có trọng số trên 13 đặc trưng chuẩn hóa (mean/std có trọng số của fold fit). Ridge cố định λ = 1 trên `||β||²/(2n)`, intercept không phạt, Newton/IRLS, tất định.
- **SCORE_ONLY:** cùng model nhưng chỉ dùng raw (kiểm tính hợp lệ, xem mục 2.4).
- **RAW:** ngưỡng trực tiếp trên raw.
- **F:** B1 `.40/.60` (raw ≥ 0,60), chỉ báo.
- **Chọn ngưỡng (cho từng RAW, SCORE_ONLY, FULL), chỉ trên 400 nguồn calibration:**
  - lưới = 101 quantile (0–100%) của score trên ứng viên calibration;
  - mục tiêu = trung bình Hmean Clean và Hmean LR2, mỗi Hmean từ **tổng counts**;
  - ràng buộc tiny LR2 ≥ tiny của F trên cùng nguồn − 3 pp;
  - hòa thì lấy ngưỡng cao hơn.

## 7. Kết quả và nhãn (khóa; calibration là sàng lọc, KHÔNG phải xác nhận)
- Báo cho RAW, SCORE_ONLY, FULL và F: TP/FP/GT/Hmean mỗi điều kiện, tiny, mean.
- Báo các hiệu FULL − RAW, FULL − F, RAW − F, SCORE_ONLY − RAW, kèm bootstrap ghép cặp theo nguồn (2.000, seed 431; giữ cặp Clean/LR2).
- **Nhãn** (dựa trên FULL − RAW):
  - `INVALID_CONTROL`: SCORE_ONLY lệch RAW ≥ 0,05 pp.
  - `NO_SIGNAL`: Δ ≤ 0.
  - `INCONCLUSIVE`: cận dưới CI ≤ 0.
  - `POSITIVE_BELOW_GATE`: Δ < +0,30.
  - `PASS_CALIBRATION_SCREEN`: Δ ≥ +0,30 và CI > 0.
- **Chỉ khi `PASS_CALIBRATION_SCREEN`** mới được đề xuất (không tự chạy) P3:
  - một lượt dev duy nhất với model + ngưỡng đã khóa (FULL và RAW cùng ngưỡng TRAIN, so với F), cả A, Clean + LR2;
  - cổng nhóm cũ giữ nguyên;
  - cần dump đặc trưng trên dev (GPU riêng).
- Ngoài trường hợp đó: **dừng**, không thêm đặc trưng, model hay tham số.

## 8. Notebook và điểm dừng
`notebooks/kaggle_boxquality_janek.ipynb`, dataset `runtime/kaggle_upload/cs431-boxquality.zip`, hướng dẫn `docs/research/janek/HUONG_DAN_KAGGLE_boxquality_janek.md`.
- **Ô 3** (kiểm input): tải lại TextOCR (SHA), dựng lại `train_rows` và so SHA với dataset; tải 10 ảnh smoke (CRC + SHA).
- **Ô 4** (smoke, GPU): 20 task.
  - Đường đặc trưng phải trùng tuyệt đối `capture_candidates` (cả 20 task);
  - `smokecheck`: evaluator chính thức khớp, raw feature = raw;
  - `devcheck`: 5 nguồn dev LR2 phải **trùng tuyệt đối** dump T4 (chỉ kiểm pipeline, không đọc nhãn dev);
  - in chi phí thật (forward, post-process thường, post-process + đặc trưng, VRAM) và dự kiến cho 4.000 task.
  - **Dừng** nếu một kiểm sai, hoặc dự kiến > 3 giờ.
- **Ô 5:** tải 2.000 ảnh (Range), dựng LR2, trích 4.000 task (resumable).
- **Ô 6:** fit + calibration (CPU), kết quả + bảng review.
- **Ô 7:** đóng gói.
- **Latency:** chi phí thêm của đường đặc trưng so với post-process thường được đo ở Ô 4/5. Mục tiêu thiết kế ≤ 1,10× median F đầu-cuối **chưa phải cổng nhóm**. Đo latency công bằng trên dev là việc của P3/P4.
