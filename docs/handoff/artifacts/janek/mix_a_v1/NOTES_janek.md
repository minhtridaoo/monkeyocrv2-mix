# Mix A-core v1: A2r (A100) + chọn lại A→B theo R1

**Kết luận theo quy tắc đã khóa:** α\* chọn trên A là **α = 0 (B-Clean, một endpoint)**. Vì vậy **Mix A chưa thắng theo quy tắc đã khóa**, `continue_mix_a = false`. Trạng thái kiểm chứng là `VERIFIED` theo chính sách `same_gpu_endpoints`. Trạng thái này xác nhận tái lập, không chứng minh phương pháp tốt hơn.

Nguồn: `decision_janek.json`, `selection_janek.json`, `evaluation_B_janek.csv`, `bootstrap_B_janek.json`. Bản gốc nằm trên Drive `MyDrive/CS431_MixA/runs/{merge_audit_v1,reselect_v1}_A100-SXM4-40GB/`.

## Thứ tự quyết định và chạy

1. **Ngày 20/11:** user báo "Tuấn chọn a", ghi vào kế hoạch §4.3 trước khi chạy A2r.
2. **A2r trên Colab A100-SXM4-40GB:** chạy 5 model × 2.262 nguồn × (Clean, LR2) × 72 setting.
   - Identity trùng probe; chỉ khác `probe_sources` và `sources_sha256`.
   - MONKEY_V2 `dc19b85`, sạch. torch 2.9.0+cu128, cuDNN 91002.
3. **Ô 8 trên Colab chạy với code cũ** (mốc tiny tuyệt đối 42%), lưu ở `decision_colab_legacy42_janek.json`.
4. **Sau đó user chuyển ý kiến Tuấn:** đồng ý R1 và +0,5 pp; đổi mốc tiny thành tương đối (B1 − 3 pp). Quyết định được ghi vào kế hoạch §4.2 **trước khi** mở `decision_janek.json`. Code đổi theo, có test.
5. **Chạy lại `reselect_split` ở local** trên đúng các NPZ đó, bằng `reselect_spec_local_janek.json` (chỉ đổi đường dẫn). Kết quả chính thức là `decision_janek.json`.
   - α\*, các giá trị trên A/B và trạng thái kiểm chứng trùng bản Colab.
   - Chỉ phần tiêu chí tiny khác.

## Kiểm chứng

| Cổng | Kết quả |
|---|---|
| P-map endpoint (20 ảnh, ≤1e-6) | PASS ở cả hai endpoint, max diff 0,0 |
| `gate_same_gpu` (11 cột, chính xác, 40 nguồn probe) | PASS: B-Clean 0/5.760 ô lệch, B1 0/5.760 |
| Endpoint và chính sách BN (`merge_manifest.json`) | Endpoint giữ nguyên `state_dict` gốc (key, shape, dtype, giá trị bằng nhau). BN: thống kê nội suy tuyến tính, **không ước lượng lại**. Tensor không phải float (`num_batches_tracked`) lấy từ B1 |
| Độ lệch so với H200 (chỉ báo cáo) | 3a: B-Clean 72/4.524 ô, B1 66/4.524. 3b trên 72 setting: B-Clean 10.260/325.728 ô (3,1%), B1 11.098/325.728 (3,4%), `gt` không lệch ô nào. \|ΔHmean\| toàn cohort lớn nhất 0,014 pp (B-Clean) và 0,023 pp (B1) (`compare_vs_H200_full_janek.json`) |

Trên toàn cohort, sai khác giữa A100 và H200 triệt tiêu gần hết (≤0,023 pp). Nhưng trên từng ảnh vẫn có lệch ±1–7 polygon, nên số A100 không được so trực tiếp từng ô với H200.

## Chọn trên A (R1)

Tổng micro-Hmean Clean + LR2 trên A tại setting tự chọn; mean = tổng / 2.

| Model | α | Setting chọn trên A | Tổng trên A |
|---|---|---|---|
| **B-Clean** | 0 | `mask_040/score_0.60` | **157,016** |
| merge | 3/4 | `epsilon_005/score_0.50` | 156,600 |
| merge | 1/2 | `epsilon_005/score_0.50` | 156,518 |
| merge | 1/4 | `mask_040/score_0.60` | 156,353 |
| B1 | 1 | `epsilon_005/score_0.50` | 156,332 |

B-Clean dẫn α tốt thứ hai 0,42 điểm tổng, tức khoảng 0,21 pp mean Hmean.

## Đánh giá trên B (đã khóa; chỉ α\* được xét)

| Model | Clean H | LR2 H | Mean H | LR2 FP | Clean FP | LR2 recall <4 px |
|---|---|---|---|---|---|---|
| **B-Clean (α\*)** | 81,16 | 74,59 | **77,87** | 3.211 | 5.636 | 16,08% |
| α 1/4 | 80,87 | 75,81 | 78,34 | 3.694 | 6.042 | 21,55% |
| α 1/2 | 80,27 | 76,28 | 78,28 | 4.451 | 6.271 | 32,50% |
| α 3/4 | 79,99 | 76,69 | 78,34 | 4.861 | 6.491 | 34,56% |
| B1 | 79,52 | 76,69 | 78,11 | 5.456 | 6.728 | 37,63% |

Tiêu chí của α\*:
- `passes_margin = false`: α\* là endpoint, và mean của nó (77,87) còn thấp hơn B1 (78,11).
- `passes_tiny = false`: 16,08% so với mốc 37,63 − 3 = 34,63%.
- Mốc cũ 42% (chỉ để đối chiếu): cũng FAIL, và chính B1 cũng dưới 42%.

## Mô tả thêm, không dùng để đổi kết luận

Đây là quan sát trên B **sau khi đã khóa**. Không được dùng để chọn lại α:
- **Thứ tự endpoint đảo giữa A và B.** Trên A, B-Clean hơn B1 khoảng 0,34 pp mean. Trên B, B1 hơn B-Clean 0,24 pp. Mọi khác biệt đều dưới 0,5 pp, nhỏ hơn độ phân giải của tập development này.
- **Các α ở giữa trên B** cao hơn cả hai endpoint 0,17–0,23 pp mean, nhưng **không α nào đạt +0,5 pp**. CI paired 95% (bootstrap 2.000 lượt theo source, Clean/LR2 cùng source) đều chứa 0:
  - α 1/4: +0,47 [−0,04; 0,90] so với B-Clean, +0,23 [−0,78; 1,42] so với B1.
  - α 1/2: +0,40 [−0,42; 1,26] so với B-Clean, +0,17 [−0,40; 0,98] so với B1.
  - α 3/4: +0,46 [−0,53; 1,48] so với B-Clean, +0,23 [−0,12; 0,71] so với B1.
- **Đánh đổi thấy rõ ở α 3/4 so với B1:**
  - Clean +0,47 pp, CI [0,20; 0,88], không chứa 0.
  - LR2 −0,01 pp.
  - LR2 FP giảm 11% (4.861 so với 5.456), Clean FP giảm 3,5%.
  - Tiny <4 px thấp hơn 3,07 pp, ngay dưới mốc −3 pp.

  Đây là gợi ý cho hướng tiếp theo, **không phải kết quả đạt chuẩn**.
- **Hàm chọn trên A (tổng Hmean) không thưởng cho tiny recall.** B-Clean thắng trên A nhờ Clean cao ở `mask_040/score_0.60`, setting có tiny chỉ 16%. Đây là mâu thuẫn giữa mục tiêu chọn và tiêu chí đi tiếp, và nên bàn lại. Lưu ý: đổi quy tắc sau khi đã thấy B thì kết quả trên B này không còn là kiểm chứng độc lập.

## Giới hạn (theo yêu cầu của Tuấn)

- A/B thuộc remaining-validation đã ảnh hưởng tới các quyết định trước, nên B không phải holdout mới.
- +0,5 pp là ngưỡng thực dụng. CI là mô tả sau chọn, với một seed huấn luyện.
- Số A100 không so trực tiếp với bảng H200. Bảng baseline toàn cohort của Tuấn (B-Clean 80,86/75,51, B1 79,43/76,90) không phải số split B.

## Chi phí GPU

5 model trên A100-40GB: 0,41–0,43 s/task, tổng wall khoảng 2,7 giờ, gồm P-map. Probe trước đó đo 0,335 s/task. Peak VRAM 1,9 GiB.
