# Kết quả pilot XV-B1 (CPU, split A, 23/11/2026)

Spec khóa trước: `docs/research/janek/xvb1_spec_20261123_janek.md`. Code `mixab/xvb1_janek.py`, test `tests/test_xvb1_janek.py` (6 test PASS). Kết quả: `docs/handoff/artifacts/janek/xvb1_v1_T4/xvb1_result_janek.json`, `per_source_A_janek.csv` (1.130 dòng).
**Ablation cùng họ với DualAgree v1 (B1+agree), không phải phương pháp độc lập.** Chỉ A (đã dùng nhiều lần), dump T4 → `PENDING_REFERENCE_CHECK`. **Không tính B. Clean: chưa đánh giá** (chưa có dump Clean). Luật chạy đúng một lần, không chỉnh.
Kiểm cứng: đếm baseline `epsilon_005/.50` tái tạo khớp 72-setting tính lại từ cùng dump trên cả 1.130 nguồn.

## Số trên A

| | Hmean | TP | FP | FN | recall | precision | tiny (<4 px) |
|---|---|---|---|---|---|---|---|
| **XV-B1** | 76,95 | 27.500 | 6.549 | 9.923 | 73,48% | 80,77% | 48,78% |
| B1 `.40/.60` (baseline chính) | 76,80 | 27.577 | 6.815 | 9.846 | 73,69% | 80,18% | 52,11% |
| B1 `epsilon_005/.50` (cha của XV-B1) | 77,03 | 27.628 | 6.680 | 9.795 | 73,83% | 80,53% | 48,78% |

Recall theo bin (XV-B1 / `.40/.60` / cha): 4–8 px 72,5 / 73,2 / 72,6; 8–16 px 80,0 / 79,9 / 80,4; ≥16 px 72,6 / 71,5 / 73,4.

| Phép so | ΔHmean | CI95 (ghép cặp theo nguồn, 2.000 lần, seed 431) |
|---|---|---|
| XV-B1 so với **`.40/.60`** (phép so khóa) | **+0,153 pp** | **[−0,459; +0,799]** |
| XV-B1 so với **cha `epsilon_005/.50`** (tách riêng tác dụng của luật) | **−0,079 pp** | **[−0,140; −0,019]** |

## Luật xóa đã làm gì (A)
- Xóa **356 box** ở **217 nguồn**. Trong đó 259 box có tính vào Pred (**TP bị xóa 128, FP bị xóa 131**, chênh lệch đếm chính xác so với cha); 97 box còn lại không đổi Pred (suy luận: nằm trong vùng bỏ qua của evaluator, **chưa kiểm**).
- TP bị xóa = 128 GT mất (không GT nào được thêm); xảy ra ở 105 nguồn. FP bị xóa ở 88 nguồn.
- Nghĩa là: trong số box bị xóa có số TP gần bằng số FP (precision cận biên ≈ 49%), cao hơn mức làm tăng Hmean (≈ nửa Hmean, khoảng 0,38). Xóa không có lợi ròng.
- Tiny: không đổi so với cha (miễn trừ đúng như thiết kế, 0,00 pp).

## Cờ và nhãn khóa (so với `.40/.60`)
- c1 (≥ +0,30 pp và cận dưới CI > 0): **trượt** (+0,153; CI chứa 0).
- c2 (tiny ≥ −3 pp): **trượt** (−3,33 pp).
- c3 (FP thấp hơn): đạt (6.549 so với 6.815).
- **Nhãn: `POSITIVE_BELOW_GATE`** (Δ > 0 nhưng trượt c1 và c2).

## Đọc đúng
- Mức dương +0,153 so với `.40/.60` **không đến từ luật XV-B1**. Nó đến từ chính cha `epsilon_005/.50` (so với `.40/.60`: dương ở A) cộng với tác dụng **âm** của luật (−0,079 pp, CI không chứa 0).
- Trượt tiny (−3,33) cũng kế thừa từ cha: cha có tiny 48,78% so với 52,11% của `.40/.60`; XV-B1 giữ nguyên tiny của cha.
- Chênh lệch theo nguồn so với `.40/.60`: 346 nguồn lên, 320 xuống; nguồn lớn nhất (`6e340fcb…`, ảnh trang báo, +190 đóng góp ròng) có 0 box bị xóa, tức do cha chứ không do luật. Tỉ lệ "top-10 nguồn" của tổng lợi ròng là 308% (tổng ròng nhỏ, bị các nguồn âm bù), nên lợi ích không phân bố đều.
- Giả thuyết rủi ro đã ghi trước ("B-Clean không phát hiện ≠ B1 sai") **có biểu hiện trong số liệu**: 128 TP bị mất. Chưa tách được vì sao (chữ B1 học thêm, hay box gộp có IoU < 0,3 với từng box B-Clean).
- Chi phí vẫn là 2 checkpoint (khoảng 2× B1 đơn); luật này không bù lại được.

## Kết luận
Luật khóa trượt c1 và c2 → **dừng**. Không sửa 8 px, 0,3 hay 0,5. Không ghép CS-B1. Không chạy B, không chạy GPU. Cùng họ với DualAgree v1: cắt FP bằng bằng chứng chéo cho thấy tín hiệu "B-Clean vắng mặt" không phân biệt FP với TP đủ tốt, kể cả khi miễn trừ box nhỏ.
