# Bắt đầu session mới · CS431 phần của Trí (cập nhật 23/11/2026)

> Cập nhật 24/11 Codex: hướng mới DỰ KIẾN ở `docs/research/janek/KE_HOACH_BOX_QUALITY_20261124_janek.md` (với file trong docs, xem cùng thư mục). Chỉ audit và chuẩn bị membership TRAIN; chưa implementation/GPU/kết quả. Không đổi protocol hoặc kết quả cũ. Đọc trạng thái mới nhất trong HANDOFF.

> Đọc file này trước, rồi `CLAUDE.md` (gốc workspace), `HANDOFF_janek.md`, `docs/research/janek/ROADMAP_toan_bo_20261123_janek.md`. Trả lời bằng tiếng Việt.
> Quy tắc cứng:
> - không viết điều chưa kiểm như sự thật;
> - không push, không gửi tin;
> - **local chỉ làm việc nhẹ**: đọc, sửa code, chuẩn bị notebook, test CPU. Phần nặng chạy trên Kaggle, **user tự chạy**;
> - không sửa GT/TEST/kết quả đã khóa; không hạ cổng, không chọn cấu hình sau khi xem số.

## 0. Ba trạng thái

| Trạng thái | Việc |
|---|---|
| **ĐÃ XONG** | CS-B1 trên đủ 2.262 nguồn LR2; latency Kaggle; chốt baseline từ file gốc; MSB1 (âm); zoom ZB1 (âm, dừng); notebook và dataset bổ sung Clean |
| **ĐANG CHỜ USER CHẠY** | `notebooks/kaggle_csb1_clean_janek.ipynb` (bổ sung Clean cho CS-B1); chấm 10 thẻ K01–K10; duyệt và gửi tin cho Tuấn (`docs/research/janek/msg_tuan_20261123_csb1_janek.md`) |
| **KẾ HOẠCH, CHƯA THỰC HIỆN** | hướng cải tiến thứ hai (mục 4), chờ user chọn |

## 1. CS-B1@0.7: kết quả đã xác minh (development, T4 → PENDING_REFERENCE_CHECK)
- **Cấu hình khóa** (`mixab/dualagree_janek.py` `LOCKED`):
  - giữ toàn bộ box B1 `epsilon_005` (mask .30, unclip 1,5, min_width 5, eps .005) có score ≥ .5;
  - thêm box B-Clean `mask_040` (mask .40, eps .01) có score ≥ .7 nếu không ghép được với box B1 nào ở IoU ≥ .5 (greedy một-một).
  - Biến thể này chọn **sau khi xem split A**.
- **Đã chạy đủ 2.262 nguồn LR2** (A 1.130 + B 1.132 = `source_cohort.csv` = GT manifest `052857fb…`). Dump T4 Colab, không ảnh lỗi. Hmean tính từ tổng counts [`docs/handoff/artifacts/janek/csb1_full_v1_T4/csb1_full_janek.json`]:

  | So với | Δ LR2 | CI95 (ghép cặp theo nguồn) |
  |---|---|---|
  | B1 `epsilon_005/.50` (cấu hình nền) | **+0,23 pp** (77,09 so với 76,86) | [0,15; 0,32] |
  | B1 `.40/.60` (B1 sau calibration lần 2; cấu hình B1 tốt nhất của Tuấn trong 72 setting) | +0,18 pp (77,09 so với 76,91) | **[−0,21; +0,58]**, chứa 0 |

  Tiny so với `.40/.60`: −1,96 pp. Theo split, so với `.40/.60`: A +0,41, B −0,04.
- **Latency** (Kaggle T4, chỉ GPU 0): 447 → 889 ms/ảnh, **1,99×** [1,98; 2,02]; bộ nhớ 894 → 991 MiB [`docs/handoff/artifacts/janek/csb1_latency_v1_T4/latency_janek.json`].
- **Cổng:**
  - luật dừng khóa trước trên A: +0,177 < +0,3 → **trượt**;
  - mean hệ thống trên A có router: +0,36 < +0,5 → **chưa đạt cổng nhóm**.
- **Được tuyên bố:** "dương nhỏ so với cấu hình nền; không phân biệt được với B1 `.40/.60`; chi phí khoảng 2×; chọn sau khi xem A."
- **Không** đủ điều kiện đưa vào TEST.

**Baseline đã chốt từ file gốc** [repo `pr_quad_audit_v1/docs/handoff/artifacts/janek/decision_audit_report.md`, `counts/B1_source_counts.npz`; Trí `docs/handoff/artifacts/janek/csb1_full_v1_T4/baseline_check_janek.json`]:
- `.40/.60` là điểm Tuấn chọn trên lưới mở rộng theo luật mean Clean + LR2 trên đủ 2.262 nguồn: **sau** calibration lần 2.
- +1,61 pp là so với `.30/.40` ("original calibrated", 75,29) → 76,90.
- FP: H200 (Tuấn) 11.997 · A100 11.999 · T4 11.988, do khác GPU; 32 nguồn lệch ±1.

## 2. Bổ sung Clean

> **ĐÃ TIẾP NHẬN 23/11:** kết quả ở `docs/research/janek/csb1_clean_result_20261123_janek.md`, bản chính `docs/handoff/artifacts/janek/csb1_clean_v1_T4/`. Phần dưới giữ để tra cứu quy trình.

**Mục đích:** hoàn thiện đánh giá, đo đánh đổi khi áp CS-B1 cho mọi ảnh (không định tuyến). **Không mặc định sẽ đạt cổng.**

**File:**
- Notebook: `notebooks/kaggle_csb1_clean_janek.ipynb` (builder `scripts/build_kaggle_csb1_clean_notebook_janek.py`).
- Dataset: `runtime/kaggle_upload/cs431-csb1-clean.zip`, **939 MiB**, SHA-256 `337f4b06991f64ebb09875099f3c8c099a908c08067e4b2b68323955b1450eb0`. Có sẵn 2.262 ảnh Clean; `code/CODE_MANIFEST.json` SHA `f7f5f6996b288c9501e7837c36b52b8a421282cb3a4a24a6fed67ca1749164a9`.
- Hướng dẫn: `docs/research/janek/HUONG_DAN_KAGGLE_csb1_clean_janek.md`.
- Spec khóa trước: `docs/research/janek/csb1_clean_check_spec_20261123_janek.md`.
- Code: `mixab/dualagree_janek.py` (`dump --condition clean`), `mixab/csb1_clean_janek.py`.

**Tiếp nhận output** (session mới làm, chỉ đọc):
1. Lấy output về `runtime/kaggle_downloads/<slug>/`, chạy với `PYTHONIOENCODING=utf-8`:
   - liệt kê notebook: `python -c "import sys; from kaggle.cli import main; sys.argv=['kaggle','kernels','list','--mine','--sort-by','dateRun']; main()"`;
   - tải output: `... sys.argv=['kaggle','kernels','output','ojanek/<slug>','-p','<thư mục>','-o'] ...`.
   - Token Kaggle có sẵn trong biến `KAGGLE_API_TOKEN`; thư viện `kaggle` 2.2.4 cài sẵn.
2. **Kiểm** (dừng và báo nếu sai):
   - `code_manifest_sha256_janek.txt` trong zip output phải là `f7f5f699…`;
   - `env_T4_janek.json` có DBPostprocessor `9bfff8aa…` và Pillow 11.3.0;
   - `clean_dump/dump_done_janek.json` có 36 shard/model, identity `condition = clean`;
   - `clean_probe/fidelity.json`: chỉ báo tỉ lệ lệch so với A2r (do khác GPU);
   - `csb1_clean_eval/csb1_clean_result_janek.json` có `sources = 2262` cho cả hai điều kiện.
   - Có ảnh lỗi hay thiếu thì **báo**, không âm thầm loại.
3. **Báo cáo** (không chỉnh cấu hình, không quét ngưỡng):
   - Clean và LR2 Hmean, TP/FP/FN, tiny cho CS-B1, B1 `.40/.60`, B1 `eps005/.50`, B-Clean `.40/.60` trên ALL, A, B;
   - Δ kèm CI;
   - mean hệ thống **không định tuyến** so với model đơn tốt nhất;
   - cổng nhóm trên A, chỉ mô tả.
   - Chép bản chính vào `docs/handoff/artifacts/janek/csb1_clean_v1_T4/`, rồi cập nhật HANDOFF, WORKLOG, roadmap, dashboard.

## 3. Đã thử gì (tránh trùng); chi tiết ở roadmap §2–§3

| Người | Hướng | Kết quả | Lý do dừng / còn mở |
|---|---|---|---|
| Tuấn | Fine-tune B-Clean, B1 | B-Clean so với B0: LR2 +13,1; B1 so với B-Clean: LR2 +1,39 [−1,26; 3,72], Clean −1,43 có ý nghĩa | xong |
| Tuấn | Calibration `.30/.40` → `.40/.60` | B1 LR2 +1,61 [1,01; 2,30] | xong (hậu xử lý) |
| Tuấn | EDSR ×2 | +0,60 [0,24; 0,96] ở ngưỡng cũ | **mở:** Clean, `.40/.60`, transfer. Là việc của Tuấn |
| Tuấn | PRC, KD, adapter A0–A2, CRAFT, fusion ASF/DySample | âm | đóng |
| Trí | Mix A, E-sel, E-BN, QRoute, DualAgree v1, ngưỡng theo kích thước | âm hoặc chưa đạt cổng | đóng |
| Trí | CS-B1@0.7 | §1 | **mở:** Clean (đang chờ), K01–K10 |
| Trí | MSB1 (tách box gộp, mask .4) | A +0,22 [−0,40; 0,81] | đóng (trượt) |
| Trí | ZB1 zoom cục bộ | G2 LR2 −3,01, Clean −3,65 | đóng (**không mở lại**) |

## 4. Hướng cải tiến thứ hai

> **CẬP NHẬT 23/11 khuya:** Flip-TTA đã chạy (`INCONCLUSIVE`, `docs/research/janek/fliptta_result_20261123_janek.md`) và đã chẩn đoán (`docs/research/janek/fliptta_diagnosis_plan_20261123_janek.md`): lợi ích phần lớn tương đương nâng ngưỡng score; khuyến nghị dừng, hoặc chỉ chạy lượt chẩn đoán P-map nếu user duyệt.

> **CẬP NHẬT (23/11, session tiếp):** hướng 1 (XV-B1) **đã chạy CPU và dừng**: ablation cùng họ DualAgree v1, nhãn `POSITIVE_BELOW_GATE`, luật tự nó âm (−0,079 pp CI [−0,14; −0,02] so với cha); xem `docs/research/janek/xvb1_result_20261123_janek.md`. Hướng 2 (Flip-TTA) **đã chuẩn bị, chưa chạy GPU**: `docs/research/janek/fliptta_spec_20261123_janek.md`, `docs/research/janek/HUONG_DAN_KAGGLE_fliptta_pilot_janek.md`. Hướng 3 chưa làm (họ hậu xử lý gần cạn). Bảng dưới là đề xuất gốc, giữ để tra cứu.

**Điểm yếu còn lại** (có số):
1. **FP của B1 cao:** LR2 11.997 so với 6.578 của B-Clean; precision 82,45% so với 88,66% [repo `handoff.md` §4].
2. Tiny recall LR2 45,79% (Clean, cùng tensor: khoảng 61%).
3. B1 mất Clean −1,43 pp so với B-Clean.
4. Lợi ích của CS-B1 tập trung (10 nguồn chiếm 66,6%) và không phân biệt được với `.40/.60`.

**Dữ liệu cho pilot:** **không còn tập nào chưa dùng để chọn phương án.**
- Validation 3.262 nguồn = S1 500 + S2 500 + remaining 2.262 [repo `CS431_Pilot_Protocol_v2.1_FINAL.md:14,43`]; cả ba đã xem.
- TEST nội bộ để dành cho benchmark cuối, và đã bị xem ở Phase 4–5 [Tuấn cung cấp].
- ICDAR2015 để dành cho benchmark cuối.
- Mọi pilot mới chỉ là **khám phá trên A**. Xác nhận thật chỉ có thể là **một lần** trong benchmark cuối do Tuấn chủ trì, với phương pháp khóa trước.

| # | Hướng | Cơ chế | Khác hướng cũ | Chi phí | Rủi ro chính |
|---|---|---|---|---|---|
| **1 (ưu tiên)** | **XV-B1: lọc FP của B1 bằng bằng chứng chéo từ B-Clean** | Box B1 **đủ lớn** (cạnh ngắn ≥ 8 px LR2) mà B-Clean không có ứng viên nào chồng lên (kể cả ở score thấp) thì bỏ. Box nhỏ giữ nguyên. Lý do: ở box ≥ 8 px, recall của B-Clean gần bằng B1 (A, `epsilon_005/.50`: 8–16 px 78,3% so với 80,4%; ≥ 16 px 73,0% so với 73,5%), nên "B-Clean không thấy gì" là dấu hiệu FP đáng tin hơn | Nhắm **precision**, ngược với CS-B1 (thêm recall). DualAgree v1 cắt box B1 theo score và làm mất tiny; XV-B1 cắt theo **sự vắng mặt của B-Clean** và miễn trừ box nhỏ. Dùng lại đúng 2 forward của CS-B1, nên ghép với CS-B1 **không tốn thêm latency** | Pilot CPU trên dump có sẵn (LR2), cộng dump Clean sắp có | FP trên chữ **thiếu nhãn** thì B-Clean cũng "thấy", nên không lọc được. Cùng họ 2 checkpoint với CS-B1 (thầy có thể coi là một hướng). Ý tưởng hình thành sau Phase 0 |
| 2 | **Flip-TTA trên P-map (1 checkpoint)** | Chạy B1 trên ảnh và ảnh lật ngang, lật P-map về, lấy trung bình **trước** hậu xử lý DB | Khác zoom: không crop, không đổi tỉ lệ, gộp ở mức P-map chứ không cộng box. Một checkpoint | 2 forward (khoảng 2× latency); cần Kaggle GPU | Cơ sở yếu: lợi ích TTA chỉ là hiểu biết chung, **chưa có bằng chứng trong repo**. P-map mềm hơn có thể lệch ngưỡng `.40/.60`. Chưa kiểm B1 có train với lật ảnh không (config có `Fliplr 0.5`) |
| 3 | Điểm ổn định theo mask (1 checkpoint, 0 forward thêm) | Box xuất hiện ở cả mask .3 và .4 thì tin hơn; dùng làm tín hiệu chấm lại | Cùng họ hậu xử lý với MSB1 và ngưỡng theo kích thước, **nguy cơ trùng cao** | CPU | Họ hậu xử lý đã gần cạn |

**Pilot tối thiểu cho hướng 1 (XV-B1)** (chưa chạy; phải viết spec khóa trước rồi user duyệt):
- **Bước 0 (CPU, dump LR2 có sẵn, split A):**
  - **Luật khóa:** xuất phát từ đầu ra B1 `epsilon_005` ≥ .5. Bỏ box B1 nếu (a) cạnh ngắn ≥ 8 px LR2 và (b) không có box B-Clean `mask_040` nào trong dump (mọi score lưu được, ≥ mức mask) chồng lên với IoU ≥ 0,3.
  - Không tham số nào khác. Không quét 8 px hay 0,3.
- **Baseline chính:** **B1 `.40/.60`** (cấu hình mạnh nhất của Tuấn), tránh lặp lại vấn đề của CS-B1. Báo thêm `epsilon_005/.50` và XV-B1 + CS-B1 (mô tả).
- **Tiêu chí đi tiếp** (A, khóa trước):
  - ΔLR2 so với B1 `.40/.60` ≥ **+0,30 pp** và cận dưới CI > 0;
  - tiny ≥ −3 pp;
  - FP thấp hơn B1 `.40/.60`.
  - Nếu dump Clean đã có: Clean ≥ B-Clean − 0,3 pp khi áp cho mọi ảnh.
- **Dừng:** trượt bất kỳ điều kiện nào → ghi âm, không chỉnh luật.
- **Đạt:** chỉ đề xuất (chưa chạy) bước P-map support trên Kaggle, và đăng ký trước cho benchmark cuối với Tuấn.
- **Chi phí:** bước 0 vài phút CPU. Theo quy tắc mới, chạy trên Kaggle CPU hoặc xin user cho phép chạy local vì đây là việc nhẹ.

## 5. Việc của session mới
1. Nếu user đã chạy notebook Clean: tiếp nhận theo §2.
2. Trình bày lại §4 để user chọn hướng. **Không** triển khai trước khi user chọn và duyệt spec.
3. Không mở lại zoom. Không coi việc hoàn thiện EDSR của Tuấn là phương pháp riêng của Trí.
