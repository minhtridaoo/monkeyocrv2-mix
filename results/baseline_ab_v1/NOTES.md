# Baseline A→B v1: ghi chú kết quả (19/11)

**Đầu vào:** NPZ của Tuấn `MONKEY_V2/docs/handoff/artifacts/pr_quad_audit_v1/counts/*.npz` @ `dc19b85`. Tôi đã tự chạy `verify_bundle.py --counts --weights`: PASS.
**Lệnh:** `python -m mixab.reselect_split --spec configs/baseline_ab_v1.json --output results/baseline_ab_v1`
**Trạng thái:** `baseline_only`, `PENDING_REFERENCE_CHECK`. Đây **không phải kết quả Mix A**; chưa có α ở giữa nào.

## Đối chiếu độc lập với bảng §0

Chọn trên toàn bộ 2.262 nguồn bằng `mixab.reselect_split.choose_setting` cho ra index 44/70/70. Clean và LR2 Hmean, LR2 FP và tiny recall **trùng đúng từng số** với bảng §0 kế hoạch chính (do Codex tính độc lập).

## Kết quả chia A/B (A = 1.130 nguồn, B = 1.132 nguồn)

| Model | Setting chọn trên A | Mean H trên B | Clean H | LR2 H | LR2 FP | LR2 recall <4 px |
|---|---|---:|---:|---:|---:|---:|
| B-Clean | 70 `mask_040` / score .60 | 77,88% | 81,16% | 74,59% | 3.207 | 16,08% |
| B1 | 45 `epsilon_005` / score .50 | 78,11% | 79,52% | 76,70% | 5.454 | 37,55% |
| B0 | 45 `epsilon_005` / score .50 | — | 74,33% | 60,25% | 1.747 | 0,46% |

B1 trên B ở setting 70 (để so): mean 78,36%, LR2 77,00%, FP 5.178, tiny 39,38%.

## Hai quan sát cần báo Tuấn (mô tả, không phải kết luận)

1. **Điểm tối ưu của B1 phẳng.** Trên A, top-4 setting chênh nhau ≤ 0,23 pp mean H (45: 78,169; 37: 77,973; 70: 77,959; 5: 77,942). Chọn trên một nửa dữ liệu có thể ra setting khác với chọn trên toàn cohort. Đây là điều R1 chấp nhận, nhưng phải nói rõ trong báo cáo.
2. **Mốc tuyệt đối "LR2 recall <4 px ≥ 42%" không ổn định.**
   - B1 ở setting 70: tiny recall 52,19% trên A nhưng 39,38% trên B. Hai nửa có số tiny GT gần bằng nhau (2.583 so với 2.575).
   - Tiny GT tập trung vào ít ảnh: 233/237 nguồn có tiny GT; **5 nguồn chiếm 53,3% tiny GT ở A** và 34,7% ở B.
   - Vì vậy chính baseline B1 cũng trượt mốc 42% trên B.
   - **Đề xuất (chưa áp dụng, cần Tuấn đồng ý):** thay mốc tuyệt đối bằng mốc tương đối trên cùng split, ví dụ tiny(α\*) − tiny(B1) ≥ −3 pp trên B, kèm CI bootstrap theo nguồn. Code hiện vẫn dùng mốc 42% như kế hoạch §4.2.
