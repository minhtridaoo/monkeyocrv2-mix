# Phần của tôi trong pipeline của Tuấn: v3 (sửa theo review lần 2, 19/11)

**Trạng thái:** chưa có code, chưa có job. Ưu tiên theo thứ tự: **giao thức A→B → đường cong α (Mix A-core) → báo cáo**. Model Stock là nhánh phụ. Mix B có thể dời sau 24/11.

## Thay đổi so với v2 (đã kiểm tra lại trong code `MONKEY_V2` @ `a831c4b`)

| # | Điểm review | Đã kiểm | Xử lý trong v3 |
|---|---|---|---|
| 0 | Phụ lục H.4 của Model Stock | Đã tự đọc bản HTML v2 | Thay "theo review" bằng trích dẫn trực tiếp (§2.3) |
| 1 | Tie-break trong code là `.30/.30`, không phải `.30/.40` | `rev4/decision_core.py:110–118`: `native = geometry=='base' and min_text_score==.3`. `pr-quad-audit-v1.md` lại ghi "prefer base 0.30/0.40". Tài liệu và code của Tuấn **lệch nhau**. | Khóa **quy tắc R1** (§2.2): giữ `.30/.30` như code và như `rev4/calibration.py` (`NATIVE=(.30,.30)`). Thêm tie-break cho α. Báo Tuấn sửa câu trong doc. |
| 2 | "Bất kỳ α nào đạt trên B" là chọn bằng B | Đúng, v2 §2.5 mâu thuẫn với §2.2 | Chỉ kiểm tiêu chí cho **α\* đã khóa trên A**. Các α khác chỉ để mô tả. |
| 3 | NPZ chỉ có `(tp, gt, pred, tiny_gt, tiny_tp)`; NPZ chưa có ở máy local | `decision_audit.py:182–184` chỉ lưu `bins[0]`. Không tìm thấy `*_source_counts.npz` trong checkout local. | Job merge của tôi **chạy lại cả hai endpoint**, lưu **đủ 4 bin**, và đối chiếu 5 cột chung với NPZ của Tuấn. Chọn lại baseline bằng CPU chỉ thực hiện **sau khi** job audit của Tuấn hoàn tất và artifact qua kiểm hash. |
| 4 | Endpoint/BN mâu thuẫn; BN của Model Stock chưa xác định | Đúng | α ∈ {0, 1} sao chép nguyên `state_dict`. Tách 3 loại kiểm tra. Khóa cách gán hệ số cho BN (§2.4). |
| 5 | Biên số học của Model Stock | Đúng (`0 < 0` khi ‖w₀‖ = 0) | Ngưỡng tuyệt đối + tương đối, tách 3 trường hợp. c ≤ 0 là **chính sách sàng lọc của nhóm**, không phải định nghĩa tính hợp lệ của Model Stock. |
| 6 | Mix B: khởi tạo gate, trạng thái init chung, recipe, diễn giải taxonomy | Đúng | Gate zero-init (A = 0,5). Một file init dùng chung cho mọi nhánh. Ghi lại recipe. Hạ mức diễn giải (§3). |

---

## 1. Vị trí trong pipeline của Tuấn

```text
Bước 3 decision_audit v1 (Tuấn) ──► artifact *_source_counts.npz + run_identity.json
   └─► [A] Giao thức A→B, quy tắc R1 (CPU, sau khi artifact hợp lệ)
          └─► [Mix A-core] merge_audit v1: α ∈ {0, .25, .5, .75, 1} (1 job GPU, lưu đủ 4 bin)
                 └─► [Mix A-opt] Model Stock: chỉ khi stats đạt §2.3
Bước 4 taxonomy (Tuấn) ──► quy tắc phân bổ budget cho Mix B
Bước 6 (Tuấn) ──► [Mix B] FSR + gate-only, cùng recipe và cùng init với Head-control/Residual
```

**Cần chốt với Tuấn:**
1. Output version `rev4_merge_audit/v1`.
2. Quy tắc R1 (gồm cả tie-break cho α).
3. Câu "prefer base 0.30/0.40" trong `pr-quad-audit-v1.md` cần sửa, vì code dùng `.30/.30`.
4. +0,5 pp và tiny ≥ 42% là **ngưỡng thực dụng nhóm đặt trước**, chỉ dùng để quyết định đi tiếp.
5. Nhánh FSR + gate-only trong Bước 6.

---

## 2. Mix A-core: Linear weight interpolation

### 2.1. Định nghĩa

- Với 0 < α < 1: w(α) = (1 − α)·w_BC + α·w_B1, áp **cùng một scalar α cho mọi tensor floating-point** trong `state_dict` (weight, bias, `running_mean`, `running_var`). Tensor integer như `num_batches_tracked` lấy từ B1. Cả hai endpoint có `num_batches_tracked` được ghi vào log để minh bạch.
- Với α = 0 và α = 1: **sao chép nguyên** `state_dict` của B-Clean / B1. Không đi qua công thức.
- Tên gọi: *linear weight interpolation between two fine-tuned checkpoints, inspired by WiSE-FT (Wortsman et al., CVPR 2022)*.
- Với α ∈ (0, 1), `running_var` nội suy là tổ hợp lồi của hai giá trị dương nên vẫn dương. Vẫn kiểm tra giá trị hữu hạn và > 0.

**Ba kiểm tra endpoint, tách riêng:**
1. **state_dict:** cùng tập key, shape, dtype, và `torch.equal` cho mọi tensor so với checkpoint gốc. **Không** yêu cầu SHA-256 file giống nhau, vì metadata/serialization có thể khác dù tensor y hệt. Ghi SHA-256 của từng file để truy xuất nguồn gốc. Chỉ yêu cầu hash trùng khi sao chép nguyên file theo byte.
2. **Đầu ra:** P-map trên 20 ảnh cố định (10 Clean, 10 LR2, seed 20261119), cùng eval mode, FP32 và preprocessing; max |ΔP| ≤ 1e-6. Đặt dung sai vì cuDNN có thể không tất định; nếu thực tế bằng 0 thì ghi là 0.
3. **Evaluator, gồm hai tầng:**
   - **3a. Tham chiếu độc lập, luôn chạy:** checkpoint endpoint đi qua đường inference gốc (`rev4/stage_a.detector_for/forward`) và evaluator chính thức. Base `.30/.40` phải trùng số đếm EDA lịch sử theo từng ảnh, giống cổng ở `decision_audit.py:186–190`, khi artifact EDA có sẵn. FAIL thì dừng.
   - **3b. Đối chiếu với audit của Tuấn:** các cột `(tp, gt, pred, tiny_gt, tiny_tp)` phải trùng NPZ của Tuấn với mọi source × condition × 72 setting. **Ghép theo key `(source_id, condition, setting_key)`**, không giả định hai mảng cùng thứ tự. So endpoint của job merge với chính nó **không** được tính là kiểm chứng.

### 2.2. Giao thức chọn A→B, quy tắc R1 (áp dụng cho mọi model)

1. **Split:** A nếu `int(sha256(source_id).hexdigest(),16) % 2 == 0`, ngược lại là B. Lưu `split_manifest_janek.csv` kèm hash.
2. **Chọn setting cho từng model trên A** (gồm B1, B-Clean và mỗi α), theo thứ tự ưu tiên:
   1. Tổng exact `Fraction` của micro-Hmean Clean + LR2 lớn hơn.
   2. Setting `base` với score `.30` (đúng như `choose_setting()` hiện tại).
   3. Nếu vẫn hòa: thứ tự xuất hiện trong `decision_plan.grid()`.
3. **Chọn α\* trên A** trong {0, .25, .5, .75, 1}, mỗi α ở setting của nó đã chọn ở bước 2, theo thứ tự ưu tiên:
   1. Giá trị exact lớn hơn.
   2. Endpoint (α ∈ {0, 1}) được ưu tiên hơn α nội suy, nghĩa là khi hòa thì không merge.
   3. α lớn hơn.
4. **Khóa** α\* và setting, rồi đánh giá **mọi** model trên B.
5. **CI:** bootstrap ghép cặp theo source trên B; Clean và LR2 của cùng một source đi cùng nhau; 2.000 lượt, seed 20261107, micro-metrics tính lại mỗi lượt.
6. **Ghi rõ:** quy tắc R1 trùng với code `decision_core` về tie-break native, nhưng **khác** câu "0.30/0.40" trong `pr-quad-audit-v1.md`. Remaining-validation đã ảnh hưởng đến các lựa chọn trước (workflow Tuấn §8.4), nên chia A/B không tạo holdout mới. CI là descriptive, 1 seed.

### 2.3. Tiêu chí đi tiếp (chỉ cho α\*)

**Đi tiếp** khi α\* ∈ (0, 1) và trên B thỏa cả hai:
- mean(Clean, LR2) Hmean ≥ max(B1, B-Clean) + 0,5 pp, với B1 và B-Clean ở setting chọn trên A;
- Recall LR2 <4 px ≥ 42%.

Cả hai ngưỡng do nhóm đặt trước, **không** suy ra từ paper.

**Trường hợp khác:**
- α\* là endpoint: kết luận "trên A, merge không thắng". Dừng Mix A-core.
- α\* không đạt trên B nhưng một α khác trông đẹp trên B: chỉ là **quan sát khám phá**, không thay người thắng.

### 2.4. Bảng báo cáo vòng 1

- **Từ job merge (đủ 4 bin)**, cho mọi α kể cả endpoint, ở cả base `.30/.40` lẫn setting đã chọn trên A:
  - Clean/LR2 P, R, Hmean, FP;
  - Recall theo **các nhóm GT phân bằng chiều cao bounding box quy về native LR2** (`<4`, `4–8`, `8–16`, `≥16` px; `decision_core.CandidateMatcher`, hệ số `(H//2)/H`). Hàng Clean và hàng LR2 dùng **cùng nhóm GT**, nên bin của hàng Clean không tính theo pixel ảnh Clean.
- **Từ NPZ của Tuấn** (chỉ dùng cho bước chọn và đối chiếu): P/R/FP/Hmean và recall `<4`. Không hứa các bin còn lại từ nguồn này.
- **Đường trade-off:** Clean Hmean, LR2 Hmean, LR2 FP, tiny recall theo α. Đây là kết quả chính dù đi tiếp hay không.

**Yêu cầu code cho `merge_audit`:** lưu `bins` đầy đủ từ `CandidateMatcher.score()` dưới dạng `(tp, gt, pred, gt_b0, tp_b0, …, gt_b3, tp_b3)`. Khi đối chiếu với NPZ của Tuấn, chỉ so 5 cột chung.

---

## 2.5. Model Stock (nhánh phụ)

**Nguồn:**
- Jang, Yun, Han, ECCV 2024, Eq. 2: t = 2c/(1 + c), w_H = t·(w₁ + w₂)/2 + (1 − t)·w₀.
- Theo reviewer: góc được tính giữa w₁ − w₀ và w₂ − w₀, đã đối chiếu với notebook chính thức.
- **Phụ lục H.4 (đã đọc trực tiếp bản HTML v2):**
  - Hai model fine-tune khác learning rate hoặc data augmentation, giữ batch size và số epoch.
  - 5 lần chạy, CLIP ViT-B/32.
  - Kết quả: 79,75 ± 0,45 (ImageNet) và 50,40 ± 0,84 (trung bình distribution shift), so với 79,89 / 50,99 của bản gốc. Tác giả đánh giá là tương đương.
  - Điều này **chưa** xác nhận trường hợp khác **dữ liệu** (Clean-only so với mixed) của nhóm.

**Xử lý số cho từng tensor floating-point k** (δ₁ = w₁ − w₀, δ₂ = w₂ − w₀, ε(k) = 1e-12 + 1e-8·‖w₀ᵏ‖):

| Trường hợp | Xử lý |
|---|---|
| ‖δ₁‖ ≤ ε và ‖δ₂‖ ≤ ε | Giữ w₀ᵏ. Ghi "unchanged". **Loại khỏi** điều kiện c > 0. |
| Đúng một trong hai ≤ ε | Góc không xác định. Ghi "undefined". Nếu có ≥ 1 tensor rơi vào đây thì **không chạy** Model Stock trong vòng 1. |
| Cả hai > ε | c = ⟨δ₁, δ₂⟩ / (‖δ₁‖‖δ₂‖), tính bằng float64 rồi kẹp vào [−1, 1] chỉ để sửa sai số làm tròn. |

**Hệ số cho BN:**
- `weight` và `bias` của BN dùng t của chính tensor đó (layer-wise như paper).
- `running_mean` và `running_var` dùng **t của `weight` của cùng module BN**. Đây là lựa chọn của nhóm, khai báo trước, vì góc của thống kê BN không phải thứ paper định nghĩa.
- `num_batches_tracked` lấy từ B1.

**Sàng lọc (chính sách bảo thủ của nhóm):**
- c = 0 cho t = 0 là hợp lệ về toán. Rủi ro bùng nổ nằm ở c → −1.
- Nhóm chọn chỉ chạy khi **mọi tensor đã xác định có c > 0**, khi đó t ∈ (0, 1] và `running_var` vẫn là tổ hợp lồi.
- Nếu có c ≤ 0: báo cáo tỉ lệ tensor và tỉ lệ tham số, không chạy dưới tên Model Stock. Biến thể có clamp chỉ chạy khi đã được đặt tên và khai báo trước như một phương pháp khác.

**Đầu ra trước GPU:** `model_stock_stats.csv` với cột `tensor, numel, norm_w0, norm_d1, norm_d2, case, cos, t`.

---

## 3. Mix B: FSR (pilot có điều kiện)

### 3.1. Quy tắc phân bổ budget (do nhóm đặt, không chứng minh cơ chế)

Mở Mix B khi đủ 3 điều kiện:
1. **Taxonomy Bước 4:** dùng cùng cách lấy mẫu và trọng số quần thể cho hai tập **đối xứng**, B1-newer-only (9.697) và B-Clean-older-only (3.667), theo đối chiếu IoU 0,5 của Tuấn.
   - Tỉ lệ FP *background* có trọng số trong B1-newer-only ≥ 40%;
   - và tỉ lệ đó cao hơn tỉ lệ tương ứng trong B-Clean-older-only. Báo cáo cả hai mẫu số.
2. Hậu xử lý (Bước 3) và Mix A-core không dập được nhóm FP đó mà không mất tiny recall.
3. Residual control của Bước 6 đã có code + probe PASS.

**Giới hạn diễn giải:** tỉ lệ *background* cao **không** chứng minh nhiễu cao tần là nguyên nhân.

### 3.2. Các nhánh và khởi tạo (cùng file init)

Ký hiệu: L = AvgPool₃ₓ₃(F), H = F − L, [A_l, A_h] = σ(G(F)) với G = Conv1×1(256→2).

| Nhánh | Z (đầu vào thân R) | Tại init |
|---|---|---|
| Residual | F | Z = F |
| Gate-only | ½(A_l + A_h)·F | G zero weight+bias nên A = 0,5, Z = 0,5F |
| FSR | A_l·L + A_h·H | A = 0,5, nên Z = 0,5(L + H) = 0,5F |

- **Init chung:** tạo **một** file `init_state.pt` chứa trạng thái khởi tạo của thân R (sinh một lần, seed cố định), DBHead (từ B1) và G (zero). Cả 3 nhánh **nạp cùng file này**, không dựa vào "cùng seed".
  - Lớp cuối của R zero cả weight lẫn bias, nên F' = F tại init ở mọi nhánh.
  - Ở Residual, thân R nhận F thay vì 0,5F. Khác biệt này được ghi lại; vì lớp cuối bằng 0 nên đầu ra tại init vẫn trùng.
- **Recipe** (theo workflow Tuấn §6, phải chốt bằng config ghi trước khi chạy):
  - Init B1; frozen backbone + FPNC (gồm cả buffer BN, eval mode); trainable DBHead + module mới.
  - BN trong adapter ở train mode, như nhau ở mọi nhánh.
  - 15.222 TRAIN ID; 50/50 Clean/native LR2; cùng source order/sampler stream và augmentation; seed 42.
  - Microbatch 2 × accumulation 2; AdamW LR 2e-5, weight decay 1e-4, clip 5.
  - **1.000 optimizer updates**, không sweep checkpoint.
  - Scheduler cho 1.000 update (warmup/poly) **chưa chốt**, phải ghi trước khi chạy.
- **Probe:**
  - Đầu ra trùng B1 (eval, FP32, cùng preprocessing, ≤ 1e-6).
  - Ở bước 1 chỉ yêu cầu gradient ở lớp cuối của R khác 0.
  - Sau ≤ 20 update, gradient của thân R và G khác 0.
  - Hash backbone/FPNC không đổi.
  - Đo latency/VRAM của cả 3 nhánh trong cùng job.

### 3.3. Cách viết kết luận

- **FSR > Gate-only:** "Phép tách và tái phối hợp low/high đem lại lợi ích **trong đối chứng này**." Không khẳng định cơ chế là loại nhiễu cao tần nền, vì F' = F + R(Z) vẫn giữ đường tắt F.
- **Chỉ có FSR và Residual:** chỉ được kết luận về "toàn bộ module FSR".
- **Tên gọi:** "FreqSelect-inspired (FADC, CVPR 2024), avg-pool two-band decomposition". SET (CVPR 2025) chỉ là động cơ.

---

### 3.4. Việc treo, xử lý khi mở nhánh phụ (không chặn Mix A-core)

- **Model Stock:**
  - Tập tính cosine và sàng lọc = **các parameter** (`named_parameters`). BN running statistics **bị loại** khỏi tập này. Đoạn "mọi tensor floating-point" ở §2.5 chỉ áp cho Mix A-core.
  - Nếu `BN.weight` của một module rơi vào trường hợp "unchanged" mà buffer của module đó có thay đổi, vòng đầu đánh dấu là **chưa hỗ trợ** và **bỏ nhánh Model Stock**. Nội suy tuyến tính không bị ảnh hưởng.
- **Mix B:**
  - Điều kiện "gradient của G khác 0" chỉ áp cho FSR và gate-only.
  - Loader của Residual phải liệt kê **chính xác** các key được phép thiếu hoặc thừa (các key của G). Không dùng `strict=False` rồi bỏ qua mọi lỗi.
  - Trước khi train: khóa scheduler, quy trình đánh giá (chọn setting trên A theo R1, đánh giá trên B, cùng CI) và **tiêu chí đi tiếp FSR so với gate-only** theo cùng mẫu ở §2.3.

## 4. A4 (isotonic): hoãn

Giữ như v2 §4:
- Isotonic không đảo thứ hạng trong cùng bin, nên tương đương ngưỡng theo size-bin.
- Nếu làm thì cần nhãn IoU theo match một-một của evaluator (Kuzucu et al., ECCV 2024, §4) và đối chứng ngưỡng theo size-bin.

---

## 5. Lịch 19/11 → 24/11 (2 giờ/ngày)

| Ngày | Việc | GPU nền | Cổng |
|---|---|---|---|
| T5 19/11 | Gửi Tuấn 5 điểm chốt (§1). Hỏi job `decision_audit` đã chạy chưa, kèm log thời gian và đường dẫn NPZ. | - | Đã chốt R1 |
| T6 20/11 | `weight_merge_janek.py` (§2.1, §2.5 stats) + test: 3 kiểm tra endpoint trên CPU (state_dict), các trường hợp biên của Model Stock (‖w₀‖ = 0, một δ = 0, c = −1, c = 0). `reselect_split_janek.py` (R1, CPU). | - | Test PASS |
| T7 21/11 | `merge_audit_janek.py` (lưu đủ 4 bin, cổng NPZ) + submit α ∈ {0, .25, .5, .75, 1}. Chạy `model_stock_stats.csv` trên CPU. | Job merge | Cổng endpoint |
| CN 22/11 | **Đệm.** Nếu job xong: chọn trên A, khóa, đánh giá B, CI, vẽ đường trade-off. | (Model Stock nếu stats đạt) | Quyết định đi tiếp |
| T2 23/11 | Viết báo cáo Mix A (kể cả kết quả âm). Chỉ khi §3.1 đạt: soạn config/probe cho Mix B. | - | - |
| T3 24/11 | Hoàn thiện, gửi Tuấn. Mix B train chỉ sau khi probe PASS, có thể dời. | Tùy | Tóm tắt cho nhóm |

**Trạng thái kiểm chứng (thay cho "phụ thuộc chặn"):**
- **Thiếu NPZ không chặn tính toán.** Job merge vẫn chạy và lưu kết quả, với kiểm tra 1, 2, 3a.
- Kết quả mang trạng thái **`PENDING_REFERENCE_CHECK`**. Chỉ chuyển sang `VERIFIED` khi kiểm tra 3b PASS.
- Nếu 3b FAIL: chuyển sang `INVALIDATED`, điều tra nguyên nhân, và **vô hiệu hóa** mọi kết luận phụ thuộc vào kết quả đó (bảng A→B, α\*, CI) cho đến khi sửa xong.

## 6. Trích dẫn

- Wortsman et al. *WiSE-FT.* CVPR 2022: nguồn cảm hứng.
- Jang, Yun, Han. *Model Stock.* ECCV 2024, arXiv:2403.19522v2 (Eq. 2–3; phụ lục H.4 đã đọc trực tiếp: <https://arxiv.org/html/2403.19522v2#A8.SS4>).
- Kuzucu et al. *On Calibration of Object Detectors.* ECCV 2024 (§4, cho A4 khi làm sau).
- Chen, Gu, Fu et al. *FADC.* CVPR 2024. Sun et al. *SET.* CVPR 2025 (động cơ).
