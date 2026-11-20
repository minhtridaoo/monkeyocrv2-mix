# Probe A2 trên A100 (40 nguồn), chạy trên Colab đêm 19–20/11/2026

User chạy notebook `notebooks/colab_E1_A2_probe.ipynb`. Bản gốc của kết quả nằm trên Drive `MyDrive/CS431_MixA/` (`runs/merge_audit_probe_v1_A100-SXM4-40GB/`, `reports/`, `logs/`, `export/cs431_probe_A100-SXM4-40GB_20261119_193657.zip`). Thư mục này giữ bản sao các file JSON nhỏ. Mốc giờ trong tên file là giờ UTC của VM.

## Môi trường (E1)

Nguồn: `env_*.json`, `doctor_*.json`.

- GPU NVIDIA A100-SXM4-40GB (sm_80), Python 3.13.16, torch 2.9.0+cu128, mmcv 2.0.1 (build từ source), mmengine 0.10.7, mmdet 3.1.0, mmocr 1.0.1, Pillow 11.3.0, opencv 4.10.0.84, numpy 2.4.6.
- Doctor: **22/22 PASS, 0 FAIL, 0 WARN**. DBPostprocessor có SHA `9bfff8aa…`, trùng bản ghim. Backbone đã đăng ký. Config FP32. Đủ 2.262/2.262 ảnh Clean và LR2. Cả 5 checkpoint khớp SHA.
- MONKEY_V2 ở `dc19b85`, sạch (`audit_identity.json`).
- Cờ TF32 giống Tuấn: `rev4/stage_a.py:188–189` đặt matmul TF32 = False, cuDNN TF32 = True.
- Khác biệt so với job của Tuấn: GPU (A100 thay vì H200) và Python 3.13. Python bên Tuấn dùng **chưa kiểm**.

## Cổng (`gates.json`)

| Endpoint | 3a (EDA `.30/.40`, 40 nguồn × 2 điều kiện) | P-map (20 ảnh, ngưỡng 1e-6) |
|---|---|---|
| B-Clean | **FAIL**: 1/80 ô lệch. Nguồn `9dfc8222c88569f0`, LR2: tp 144 so với 145 của EDA; gt và pred trùng | PASS, max diff 0,0 |
| B1 | PASS, 0/80 | PASS, max diff 0,0 |

`ready_for_full_run = false`, vì cổng 3a là cổng khớp chính xác và không được nới.

## So với NPZ H200 trên đủ 72 setting

Chỉ để mô tả, không đổi trạng thái cổng. Lệnh: `python scripts/compare_probe_reference.py <run_dir>` → `compare_vs_H200_probe40.json`.

| Endpoint | Ô lệch / 5.760 | Nguồn có lệch | \|diff\| lớn nhất | \|ΔHmean\| lớn nhất trên 40 nguồn |
|---|---|---|---|---|
| B-Clean | 104 (1,8%) | 13/40 | tp 1, pred 2, tiny_tp 1; gt 0 | 0,15 pp |
| B1 | 92 (1,6%) | 15/40 | tp 1, pred 3; gt 0 | 0,26 pp |

Diễn giải. Mục này là **suy luận**, chưa chứng minh:
- `gt` không bao giờ lệch, nên phần chuẩn bị dữ liệu và GT trùng khớp.
- Lệch chỉ ở mức ±1–3 polygon trên một số ảnh. Một polygon lệch thường xuất hiện ở nhiều score liền nhau, nên số ô lệch phóng đại số polygon thật sự khác.
- P-map trên cùng GPU khớp tuyệt đối, nên phần trộn trọng số và load checkpoint không phải nguyên nhân.
- Nguyên nhân hợp lý nhất là khác phần cứng: thuật toán cuDNN và TF32 trên A100 khác H200, đẩy một số ô xác suất qua ngưỡng. Chưa có thí nghiệm tách riêng để xác nhận.
- Tầm quan trọng: 0,15–0,26 pp lớn ngang khoảng cách top-4 setting (0,23 pp) và bằng một nửa ngưỡng +0,5 pp. Vì vậy **không thể trộn số A100 với endpoint H200** trong cùng một so sánh.

## Thời gian và VRAM (`profile.json`)

- Trung bình 0,335 s/task, gần như như nhau giữa 5 model. Peak VRAM 1.153 MiB.
- Ước lượng chạy đủ 5 model × 2 điều kiện × 2.262 nguồn trên A100: khoảng **2,1 giờ** cho phần inference. Chưa tính thời gian cài đặt và chuẩn bị dữ liệu (chưa đo).
- T4 **chưa profile**.

## Hệ quả theo kế hoạch §4.3

Trường hợp "có lệch" đã xảy ra. Không nới cổng. Cần Tuấn chọn:
- (a) So các α với endpoint chạy lại trên cùng A100. `merge_audit` vốn đã chạy lại cả hai endpoint trong cùng job. Khi đó cổng 3a/3b so với H200 chỉ để báo cáo độ lệch, cần Tuấn đồng ý ghi lại quy tắc.
- (b) Tuấn chạy `merge_audit` trên H200.

Chưa chạy A2r đầy đủ.
