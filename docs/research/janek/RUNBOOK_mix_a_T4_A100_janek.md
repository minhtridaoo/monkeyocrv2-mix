# Runbook Mix A-core trên T4 / A100 (người chạy: user)

> GPU của mình chỉ có **T4 (16 GB) hoặc A100**. NPZ tham chiếu của Tuấn được tạo trên **H200**. Code GPU của `mixab/merge_audit_janek.py` **chưa chạy thử lần nào** (local không có MMOCR/CUDA); các phần CPU đã có test. Luôn chạy **probe trước**.

## Đường tắt: một lệnh

Sửa 4 đường dẫn ở đầu `scripts/run_probe_janek.sh` (hoặc `export M=... CLEAN_ROOT=... PREPARED_ROOT=... CONFIG=...`), rồi chạy `bash scripts/run_probe_janek.sh`. Script chạy lần lượt: test CPU → merge weights (nếu chưa có) → **doctor** → `prepare-lr2` → probe 40 nguồn, và dừng ngay ở bước lỗi đầu tiên.

**Doctor** (`python -m mixab.doctor_janek ...`) kiểm mọi thứ trước khi tốn GPU:
- Phiên bản package so với môi trường Tuấn đã kiểm.
- CUDA: đúng 1 device, arch có trong wheel, chạy thử được một kernel.
- T4 không có TF32 thì báo WARN.
- SHA của DBPostprocessor và việc `MonkeyOCRv2ViTAEBackbone` đã được đăng ký chưa.
- Đường dẫn encoder trong config.
- GT của bundle; SHA của mẫu ảnh Clean/LR2; 5 checkpoint merge; dung lượng đĩa.

Đọc dòng nào ghi FAIL rồi sửa đúng chỗ đó. Ở máy local (CPU), doctor báo đúng 9 FAIL môi trường và PASS cho bundle/weights (kiểm 20/11).

## 0. Môi trường (theo `MONKEY_V2/docs/handoff/checkpoints.md`, chưa tự kiểm cài trên T4/A100)

- Torch 2.9.0+cu128, MMCV 2.0.1, MMEngine 0.10.7, MMOCR 1.0.1, OpenCV 4.10.0.84, **Pillow 11.3.0** (bắt buộc cho native LR2).
- Code detection của MonkeyOCRv2 (đăng ký `MonkeyOCRv2ViTAEBackbone`) và encoder `zenosai/MonkeyOCRv2-AS`. Sửa `model_dir` và `weights` trong bản sao của `docs/handoff/inference_config.py` cho trỏ đúng tới encoder.
- `merge_audit` tự dừng nếu SHA-256 của DBPostprocessor đã cài khác bản Tuấn (`9bfff8aa…`). Bản đúng bytes nằm ở `bundle/provenance/installed_db_postprocessor.py`.
- Kiểm GPU trước: `python -c "import torch;print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"`.

Đặt biến cho gọn:

```bash
M=/path/to/MONKEY_V2                      # checkout của Tuấn, đúng commit dc19b85 hoặc mới hơn
W=/path/to/cai_tien_moi                   # repo của mình (thư mục chứa mixab/)
B=$M/docs/handoff/artifacts/pr_quad_audit_v1
export MONKEY_V2_ROOT=$M
cd $W
```

## 1. Trọng số merge (CPU, khoảng 10 giây, ~420 MB)

```bash
python -m mixab.weight_merge_janek --bclean $M/docs/handoff/weights/bclean-inference.pth \
  --b1 $M/docs/handoff/weights/b1-inference.pth --output runtime/merge_weights_v1
```

Hai file export là state_dict phẳng, nên **không cần** `--allow-pickle`. Bước này đã chạy thành công trên máy local ngày 19/11 (endpoint giống hệt bản gốc, đọc lại giống hệt).

## 2. Ảnh native LR2

```bash
python -m mixab.merge_audit_janek prepare-lr2 --bundle $B --clean-root /path/to/textvqa --prepared-root /path/to/prepared
```

Lệnh này tạo `validation_lr2/*.png` bằng `rev4.training_data.native_lr2` và kiểm SHA-256 với `tasks_metadata.csv`. Nếu SHA lệch thì dừng; thường là do sai phiên bản Pillow.

## 3. Probe (bắt buộc trước khi chạy đầy đủ)

```bash
python -m mixab.merge_audit_janek run --bundle $B --clean-root ... --prepared-root ... \
  --model-config my_inference_config.py --weights-dir runtime/merge_weights_v1 \
  --bclean-export $M/docs/handoff/weights/bclean-inference.pth --b1-export $M/docs/handoff/weights/b1-inference.pth \
  --probe 40 --output runtime/merge_audit_probe_v1
```

Đọc kết quả:
- `gates_janek.json`:
  - **Cổng 3a** = số đếm `.30/.40` của endpoint so với EDA H200, trên 40 nguồn × 2 điều kiện.
  - **P-map** = bản sao endpoint so với file export gốc trên cùng GPU; phải ≤ 1e-6.
- `profile_janek.json`: s/task và VRAM đỉnh **trên GPU này**. Ước lượng chạy đầy đủ ≈ s/task × 4.524 task × 5 model (cộng P-map cho 2 endpoint).

Quyết định:
- **3a PASS + P-map PASS:** chạy đầy đủ (bước 4).
- **3a FAIL:** khả năng cao do khác GPU, **không tự nới cổng**. Ghi số mismatch và các ví dụ vào WORKLOG, báo Tuấn, rồi chọn theo §4.3 của kế hoạch chính: (a) so cùng GPU, hoặc (b) nhờ chạy trên H200.
- **P-map FAIL:** lỗi khi lưu hoặc nạp trọng số. Dừng và điều tra.

## 4. Chạy đầy đủ (resume được)

Dùng lệnh như bước 3 nhưng bỏ `--probe`, đổi `--output runtime/merge_audit_v1`, và thêm `--reference-policy same_gpu_endpoints --same-gpu-probe <thư mục probe cùng GPU>`. Đây là phương án (a) ở kế hoạch §4.3, Tuấn chọn ngày 20/11. Identity của job phải trùng probe (chỉ khác tập nguồn), nếu không chương trình dừng trước khi dùng GPU. Trên Colab, dùng Ô 7 của notebook. Nếu phiên bị ngắt, chạy lại **đúng lệnh cũ**: tiến độ lưu mỗi 64 task (`--chunk`) trong file `.<model>.partial.*`; model nào xong rồi thì được dùng lại. Đổi bất kỳ input nào thì identity đổi, và chương trình từ chối dùng file partial cũ.

## 5. Chọn A→B (CPU)

```bash
python -m mixab.reselect_split_janek --spec runtime/merge_audit_v1/reselect_spec.json --output docs/handoff/artifacts/janek/mix_a_v1
```

Với `same_gpu_endpoints`, trạng thái tính từ P-map và `gate_same_gpu`: counts 11 cột của hai endpoint phải trùng tuyệt đối với probe cùng GPU. Cổng 3a/3b so với NPZ H200 của Tuấn vẫn được tính nhưng chỉ ghi vào `cross_gpu_drift`. Với `h200_reference` (mặc định cũ), trạng thái chỉ thành `VERIFIED` khi cả 3 cổng PASS. Trên Colab, dùng Ô 8. Ghi `decision_janek.json` và `NOTES_janek.md` vào `docs/handoff/artifacts/janek/`, kèm GPU đã dùng.

## Lưu ý về mốc tiny 42%

Ở baseline A→B (`docs/handoff/artifacts/janek/baseline_ab_v1/NOTES_janek.md`), chính B1 đã trượt mốc 42% trên B, vì GT <4 px tập trung vào ít ảnh. Mốc này **cần Tuấn quyết lại** trước khi đọc kết quả Mix A; code vẫn giữ 42% như kế hoạch.
