# Kết quả pilot Flip-TTA trên P-map (B1, Kaggle T4, 23/11/2026)

Spec khóa trước: `docs/research/janek/fliptta_spec_20261123_janek.md`. Bản chính: `docs/handoff/artifacts/janek/fliptta_v1_T4/`. Output gốc: `runtime/kaggle_downloads/fliptta_pilot_1/` (kernel `ojanek/cs431-fliptta-pilot-1`, zip 13:04 UTC).
**Pilot khám phá trên mẫu cố định 300 nguồn của split A** (Clean + LR2 = 600 ảnh). Không phải xác nhận độc lập; TTA chưa tính là phương pháp mới. T4 → `PENDING_REFERENCE_CHECK`. Không chỉnh cấu hình, không quét ngưỡng.

## Kiểm tiếp nhận (đều đạt)
- Code manifest `bc0835c6…` ✓; Pillow 11.3.0, DBPostprocessor `9bfff8aa…`, 1 GPU T4 nhìn thấy, MONKEY_V2 `dc19b85` sạch.
- Mẫu = đúng 300 nguồn trong `KAGGLE_MANIFEST.json`; 600/600 ảnh; không ảnh lỗi. Ảnh trang báo `6e340fcb…` **không** có trong mẫu. Chồng 88 nguồn với mẫu zoom (mô tả).
- **Cổng căn chỉnh đạt:** kiểm cứng (pad hằng, lật 2 lần = nguyên bản, cùng metainfo, P-map hữu hạn) đúng ở **600/600** ảnh; trung vị IoU đã căn **0,847** so với đối chứng chưa lật ngược **0,030**. (Chạy thử 6 ảnh: 0,880 so với 0,000.)
- Nhất quán của F: LR2 khớp dump T4 Colab **300/300**; Clean lệch A2r A100 ở 5/300 nguồn (khác GPU, chỉ báo).
- **Tính lại ở local** từ bản ghi đã tải: `fliptta_result_janek.json` và `per_source_janek.csv` **trùng tuyệt đối**.
- File `Downloads/cs431-fliptta-pilot-1.log` là một lượt "Save Version" khác, dừng ở Ô 1 (chưa gắn dataset); không tạo kết quả, không ảnh hưởng.

## Kết quả (F = B1 `.40/.60`; T = Flip-TTA; cùng lượt chạy, cùng ảnh)

| | Hmean F → T | Δ (CI95 ghép cặp theo nguồn) | TP | FP | recall | precision | tiny (<4 px) |
|---|---|---|---|---|---|---|---|
| **LR2** | 78,37 → 78,64 | **+0,27 [−0,16; +0,66]** | 6.736 → 6.714 (−22) | 1.181 → 1.089 (−92) | 72,64 → 72,40 | 85,08 → 86,04 | 32,94 → 31,75 (−1,18 pp) |
| **Clean** | 80,52 → 80,73 | **+0,21 [−0,22; +0,68]** | 7.160 → 7.124 (−36) | 1.351 → 1.252 (−99) | 77,21 → 76,83 | 84,13 → 85,05 | 44,79 → 43,84 (−0,95 pp) |

- Mean Clean + LR2 trên mẫu: 79,45 → 79,68.
- Theo nguồn (LR2): 87 lên, 67 xuống, 146 không đổi. Clean: 93 / 81 / 126.
- Recall theo bin (LR2, F → T): 4–8 px 72,4 → 72,2; 8–16 px 80,4 → 80,0; ≥ 16 px 69,5 → 69,6.
- **Latency thực đo** (T4, batch 1, FP32, đầu-cuối, thứ tự xen kẽ, bỏ 10 ảnh khởi động): LR2 trung vị F 476 ms → T 903 ms, tỉ số trung vị **1,95×**; Clean 479 → 907 ms (1,94×). Bộ nhớ đỉnh ≤ 1.565 MiB. Toàn pilot 1,46 s/ảnh (gồm cả kiểm căn chỉnh).

## Cờ và nhãn khóa
- c1 (LR2 ≥ +0,30 pp và cận dưới CI > 0): **trượt** (+0,27; CI chứa 0).
- c2 (Clean ≥ −0,30 pp): đạt (+0,21).
- c3 (tiny LR2 ≥ −3 pp): đạt (−1,18).
- **Nhãn: `INCONCLUSIVE`**: Δ dương nhưng CI chứa 0 ở n = 300. **Chưa đủ bằng chứng**, không phải âm cũng không phải dương.

## Đọc đúng
- Cơ chế nhất quán ở cả hai điều kiện: trung bình P-map **cắt FP** (khoảng −8% FP) và **mất một ít TP** (tiny −1 pp). Precision +0,9 đến +1 pp, recall −0,2 đến −0,4 pp. Kiểu đánh đổi này phù hợp với việc "làm mềm đỉnh xác suất" đã ghi rủi ro trước (suy luận, chưa tách).
- Δ dương ở cả Clean và LR2 với một checkpoint, nhưng CI ở cả hai đều chứa 0. Mức điểm +0,27 cũng dưới mốc +0,30.
- Chi phí khoảng 1,95× B1 đơn: hai forward. Gần bằng CS-B1 (1,99×) nhưng chỉ cần một checkpoint.
- **Theo spec:** không tự mở rộng mẫu, không đổi cách gộp (max, trọng số) hay ngưỡng. Bước tiếp nào cũng cần **user (và Tuấn) quyết định trước** và một spec mới khóa trước. Mọi lượt thêm trên A vẫn là development data.
