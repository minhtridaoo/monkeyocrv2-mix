# Hướng dẫn chạy đo latency CS-B1@0.7 trên Kaggle T4

**Mục đích:** đo công bằng **B1 một mình** so với **CS-B1@0.7** trên cùng máy, cùng phiên và cùng 200 ảnh LR2. Kết quả gồm thời gian và bộ nhớ GPU. Không train, không chạy lại toàn bộ dự đoán.

**Trạng thái:** notebook và dataset đã tạo ở máy local, đã kiểm cú pháp và test CPU (96/96). **Chưa chạy thử trên Kaggle.**

## 1. Hai file bạn cần

| File (trong `cải tiến/mới/`) | Dùng để |
|---|---|
| `runtime/kaggle_upload/cs431-csb1-latency.zip` (398 MiB, SHA-256 `b6ceffa7c4f8d6507ef291625b8bce786b28eff5bb7f351f435ed1b32e8b3bf9`) | Tải lên làm **Kaggle Dataset** |
| `notebooks/kaggle_csb1_latency_janek.ipynb` | **Import** thành Kaggle Notebook |

**Trong zip có gì** (danh sách và SHA từng file ở `KAGGLE_MANIFEST.json`; notebook tự kiểm):
- `code/`: code `mixab` cùng test.
- `weights/`: `alpha_0p00.pth` là B-Clean, `alpha_1p00.pth` là B1, kèm `merge_manifest.json`. Đây đúng là 2 checkpoint dump T4 đã dùng.
- `encoder/`: encoder MonkeyOCRv2-AS, revision `51778e77…`, cùng bản đã dùng trước đây.
- `clean/train_images/`: 220 ảnh Clean của tập đo cố định (seed 20261122: 20 ảnh khởi động + 200 ảnh đo).
- `dump/`: dump T4 trên Colab, dùng để kiểm chéo số box.
- `wheels/`: wheel MMCV 2.0.1 đã build cho Python 3.13. Chỉ dùng được nếu Kaggle chạy Python 3.13; nếu không, notebook tự build.

Repo của Tuấn (`MONKEY_V2` @ `dc19b85`) và MonkeyOCRv2 được notebook clone từ GitHub, không cần tải lên.

## 2. Tạo Dataset (một lần)
1. Vào kaggle.com → **Create → New Dataset**.
2. Kéo thả file `cs431-csb1-latency.zip`. Đặt tiêu đề (ví dụ `cs431-csb1-latency`) và chọn **Private**.
3. Bấm **Create**. Kaggle có thể tự giải nén zip; notebook xử lý được cả hai trường hợp.

## 3. Tạo Notebook và bật tùy chọn
1. **Create → New Notebook → File → Import Notebook**, chọn `kaggle_csb1_latency_janek.ipynb`.
2. Ở cột phải, mục **Session options**:
   - **Accelerator:** chọn `GPU T4 x2`, hoặc `GPU T4` nếu có. Notebook chỉ dùng **GPU 0** (`CUDA_VISIBLE_DEVICES=0`) và ghi rõ cấu hình máy vào kết quả.
   - **Internet:** **On**. Tài khoản Kaggle phải xác minh số điện thoại mới bật được.
   - **Persistence:** để mặc định.
3. **Add Input** (cột phải) → **Your Datasets** → chọn dataset vừa tạo.

## 4. Chạy từng ô

| Ô | Làm gì | Thời gian ước lượng (chưa đo trên Kaggle) |
|---|---|---|
| Ô 1 | Tìm dataset, kiểm SHA mọi file, kiểm GPU, ghim GPU 0 | < 1 phút |
| Ô 2 | Cài torch 2.9.0 cu128, mmengine, **MMCV 2.0.1**, MMOCR + add-on, ghim Pillow 11.3.0 / OpenCV 4.10 | 5–10 phút nếu có wheel; **20–40 phút** nếu phải build MMCV |
| Ô 3 | Clone repo Tuấn @ `dc19b85`, vá config (chỉ đường dẫn encoder), kiểm SHA checkpoint, tạo 220 ảnh LR2 (kiểm SHA từng ảnh), chạy test | 2–5 phút |
| Ô 4 | Kiểm GPU 0 đang rảnh | vài giây |
| Ô 5 | **Đo latency** (200 ảnh × 2 lượt × 2 cách) | 10–15 phút |
| Ô 6 | Đóng gói kết quả thành zip | vài giây |

- Có thể bấm **Run All**.
- Nếu Kaggle yêu cầu restart sau khi cài gói: restart, rồi chạy lại **Ô 1** và **Ô 2** (phần đã cài sẽ bỏ qua) và chạy tiếp.

## 5. Gửi lại cho Claude
- File `/kaggle/working/cs431_csb1_latency_T4_<thời gian>.zip`: tải ở khung **Output** bên phải.
- Nếu Ô 2 đã build MMCV: tải thêm `/kaggle/working/cs431_out/wheels/mmcv-….whl`. Thêm file này vào dataset để lần sau khỏi build.
- Nếu một ô lỗi: gửi ảnh chụp thông báo lỗi và file log trong `/kaggle/working/cs431_out/logs/`.

## 6. Notebook đo như thế nào
- **Hai cách chạy:**
  - **B1:** giải mã ảnh → B1 → hậu xử lý DB (`epsilon_005`, score ≥ 0,5).
  - **CS-B1@0.7:** giải mã **một lần** (hai model dùng chung một file config) → B1 → B-Clean → hậu xử lý cả hai (B1 ≥ 0,5; B-Clean `mask_040` ≥ 0,7) → gộp box (IoU ≥ 0,5, CPU).
- **Khởi động và bộ nhớ:**
  - Nạp B1 trước, chạy 20 ảnh khởi động → **bộ nhớ đỉnh của B1 một mình**.
  - Nạp tiếp B-Clean, chạy CS-B1 trên 20 ảnh đó → **bộ nhớ đỉnh của CS-B1** (giữ cả 2 model).
- **Kiểm giả định dùng chung ảnh đã giải mã:** box của B-Clean phải **giống hệt** khi B-Clean tự giải mã ảnh. Nếu khác, notebook dừng.
- **Đo:**
  - 200 ảnh × 2 lượt. Mỗi ảnh chạy cả hai cách, **xen kẽ thứ tự** (lượt 2 đảo lại).
  - `torch.cuda.synchronize()` trước khi bắt đầu và sau mỗi cách chạy.
  - Không tính vào thời gian: kiểm SHA ảnh, đọc file (đã nạp sẵn vào cache) và nạp model.
- **Báo cáo:**
  - median, p10, p90, mean của từng cách chạy, kèm thời gian từng bước;
  - **tỉ số median CS-B1/B1** với CI95 bootstrap theo ảnh;
  - thời gian thêm trung bình mỗi ảnh;
  - bộ nhớ đỉnh (allocated, reserved);
  - trạng thái GPU (nhiệt độ, xung nhịp) trước và sau khi đo.
- **Kiểm chéo với dump T4 (Colab):** so số box B1 và CS-B1 từng ảnh. Chỉ báo cáo, không chặn: Kaggle và Colab có thể khác driver.
- **Không so** số này với latency đo ở máy hay GPU khác để kết luận "chậm hơn bao nhiêu". Chỉ tỉ số **trong cùng lần chạy** mới có ý nghĩa.
