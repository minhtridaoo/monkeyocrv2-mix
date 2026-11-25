# Kết quả bổ sung Clean cho CS-B1@0.7 (Kaggle T4, 23/11/2026)

Spec khóa trước: `docs/research/janek/csb1_clean_check_spec_20261123_janek.md`. Bản chính: `docs/handoff/artifacts/janek/csb1_clean_v1_T4/` (`csb1_clean_result_janek.json`, `intake_receipt_janek.json`, `probe40_fidelity_vs_a2r_janek.json`, `dump_identity_janek.json`, `env_T4_janek.json`). Output gốc: `runtime/kaggle_downloads/csb1_clean_T4_20261123-110923/`.
**Development data đã xem nhiều lần; CS-B1 chọn sau khi xem A. T4 → `PENDING_REFERENCE_CHECK`. Không chỉnh cấu hình, không quét ngưỡng.**

## Kiểm tiếp nhận (đều đạt)
- `code_manifest_sha256_janek.txt` = `f7f5f699…` ✓. Ô 3 chỉ vá `Path(...).name` → `PureWindowsPath(...).name` trong bản chạy của `dualagree_janek.py`/`merge_audit_janek.py` (biên nhận `checkpoint_path_compatibility_janek.json`).
- `env_T4_janek.json`: DBPostprocessor `9bfff8aa…` ✓, Pillow 11.3.0 ✓, Tesla T4, chỉ GPU 0, torch 2.9.0+cu128, MONKEY_V2 `dc19b85` (sạch).
- `clean_dump/dump_done_janek.json`: 36 shard/model, 2.262 ảnh/model, identity `condition = clean` ✓; danh sách nguồn (`sources_sha256`) trùng dump LR2. `forward()` kiểm SHA từng ảnh; không ảnh lỗi/thiếu.
- `csb1_clean_result_janek.json`: `sources = 2262` ✓.
- **Tính lại ở local** (`python -m mixab.csb1_clean_janek` trên dump đã tải + dump LR2 local) cho JSON **trùng tuyệt đối** với file của Kaggle.
- Test trong notebook: 108 OK (skip 4). Dòng `"check": "FAIL"` trong `data_T4.log` là output mong đợi của test rò rỉ E-BN.
- **Độ lệch GPU (chỉ báo):** probe 40 nguồn, so với A2r A100 Clean: B1 83/2.880 ô, B-Clean 84/2.880 ô lệch (≈ 2,9%). Trên đủ tập ở `.40/.60` Clean: B1 79,436 (T4) so với 79,434 (A100); B-Clean 80,844 so với 80,852.

## Bảng chính: ALL (2.262 nguồn), Hmean %

| Arm | Clean H | Clean TP / FP / FN | Clean tiny | LR2 H | LR2 tiny | mean |
|---|---|---|---|---|---|---|
| **CS-B1@0.7** | **79,82** | 62.272 / 15.536 / 15.949 | 61,13% | **77,09** | 43,78% | **78,45** |
| B1 `.40/.60` | 79,44 | 61.252 / 14.745 / 16.969 | 61,19% | 76,91 | 45,73% | 78,17 |
| B1 `epsilon_005/.50` | 79,41 | 61.087 / 14.541 / 17.134 | 58,76% | 76,86 | 43,18% | 78,13 |
| B-Clean `.40/.60` | **80,84** | 61.644 / 12.636 / 16.577 | 61,17% | 75,50 | 15,99% | 78,17 |

| CS-B1 trừ | Δ Clean (CI95 ghép cặp theo nguồn) | Δ LR2 (CI95) |
|---|---|---|
| B1 `.40/.60` | **+0,39 [+0,01; +0,81]** | +0,18 [−0,21; +0,58] |
| B1 `epsilon_005/.50` | +0,41 [+0,19; +0,61] | +0,23 [+0,15; +0,32] |
| B-Clean `.40/.60` | **−1,02 [−1,76; −0,49]** | +1,59 [−0,84; +3,69] |

Hệ thống **không định tuyến** (CS-B1 cho mọi ảnh) trừ model đơn tốt nhất theo mean (B-Clean `.40/.60`, 78,17): **+0,28 pp**.

## Theo split (mô tả)
| | A (1.130) | B (1.132) |
|---|---|---|
| Clean: CS-B1 − B1 `.40/.60` | +0,30 [−0,06; +0,70] | +0,46 [−0,13; +1,28] |
| Clean: CS-B1 − B-Clean | −1,09 [−1,62; −0,67] | −0,96 [−2,27; −0,00] |
| LR2: CS-B1 − B1 `.40/.60` | +0,41 [−0,16; +1,06] | −0,04 [−0,51; +0,47] |
| mean CS-B1 − model đơn tốt nhất | **−0,18** (B-Clean tốt nhất) | +0,21 (B1 `.40/.60` tốt nhất) |
| tiny LR2: CS-B1 − B1 `.40/.60` | −2,63 pp | −1,28 pp |

## Cổng nhóm trên A (mô tả, khóa trước)
- mean hệ thống ≥ model đơn tốt nhất + 0,5 pp: **không đạt** (−0,18).
- Clean ≥ B-Clean − 0,3 pp: **không đạt** (−1,09).
- tiny ≥ B1 − 3 pp: đạt (−2,63).

## Đọc đúng
- Áp CS-B1 cho mọi ảnh **làm Clean thấp hơn B-Clean khoảng 1 pp** (CI không chứa 0). Theo spec §3: **muốn giữ Clean thì CS-B1 cần router** (shortcut kích thước ảnh của QRoute).
- So với B1 cùng checkpoint, CS-B1 **có mức tăng nhỏ trên Clean**: +0,39 pp so với `.40/.60` (cận dưới CI chỉ +0,01; riêng A thì CI chứa 0). Đây là phép đo mới trên Clean, nhưng vẫn là development data và CS-B1 được chọn sau khi xem A, nên chỉ mô tả, **không xác nhận**.
- Mean không định tuyến +0,28 pp trên ALL so với model đơn tốt nhất, nhưng trên A là −0,18. Dưới cổng +0,5 pp ở cả hai.
- Clean tiny của CS-B1 gần bằng B1 `.40/.60` và B-Clean (khoảng 61%). Phần tăng Clean chủ yếu ở TP: +1.020 TP, +791 FP so với B1 `.40/.60`.
- **Không đổi kết luận cũ:** CS-B1 đã trượt luật dừng trên A và chưa đạt cổng nhóm. Không đề xuất TEST.
