# Kết quả learned box refinement (Kaggle T4) · 25/11/2026 → **`NO_SIGNAL_ON_CALIBRATION`, dừng theo spec**

- **Spec:** `docs/research/janek/refine_spec_20261125_janek.md`.
- **Nguồn:** output notebook user tải về, `C:/Users/ADMIN/Downloads/cs431-refine-p1.ipynb` (tài khoản Kaggle `hqnhuu`). Zip kết quả `cs431_refine_T4_20261125-130632.zip` (273 MiB) **chưa tải về local**, nên số dưới đây lấy từ output in trong notebook, **chưa tính lại ở local**.
- **A không được đánh giá.** Ô 8 từ chối chạy đúng như spec. B và TEST không mở.

## 1. Kiểm hợp lệ
| Kiểm | Kết quả |
|---|---|
| Dataset | `cs431-refine` 57 file và `cs431-boxquality-devA` 1.339 file, đều khớp SHA; code manifest `032af37e…` |
| Môi trường | Tesla T4 (chỉ GPU 0), torch 2.9.0+cu128, mmocr 1.0.1, Pillow 11.3.0, DBPostprocessor `9bfff8aa…`, MONKEY_V2 `dc19b85` sạch. Test 160 OK (9 skip) |
| Crop không đổi output B1/v1 | Forward thứ hai (`bq.run_task`) cho record trùng tuyệt đối: smoke 30/30, TRAIN 160/160, A 91/91 lần kiểm |
| Post-process = `capture_candidates` | đúng ở mọi task được kiểm |
| Quad mạng tính lại từ polygon cuối | lệch tối đa 4,5·10⁻¹³ px |
| Trích | TRAIN 4.000/4.000 task (177.020 ứng viên, đúng bằng pilot box-quality); A 2.260/2.260 (103.588) |

**Về chữ `DỪNG` ở Ô 4:** chữ này do kiểm "quad nằm trong ảnh ≤ 2 px" mà mình tự đặt (đo được 2,22 / 5,50 px; ở lượt trích đầy đủ 25 / 48 px). Đây không phải kiểm tính đúng.
- Suy luận, chưa kiểm từng box: box DB sau unclip ở sát mép ảnh có thể vượt ra ngoài. Crop lấy mẫu = 0 ở ngoài map.
- Các kiểm tính đúng (trùng v1, mapping quad) đều đạt.
- User đã chạy tiếp qua chữ `DỪNG`. Ngưỡng 2 px là do mình đặt sai, không phải lỗi của pipeline.

## 2. Train (fold fit TRAIN)
- 78.587 cặp (75.162 TP, 3.425 chưa TP). Train 25 giây.
- Loss theo epoch 1, 6, …, 26: 0,188 → 0,167 → 0,155 → 0,141 → 0,130 → 0,123.
- Dry-run không có P-map dừng ở khoảng 0,223. P-map giúp khớp đích rõ rệt.

## 3. Chọn checkpoint (400 nguồn TRAIN-calibration, mean Hmean chính thức)
| | mean | Clean | LR2 |
|---|---|---|---|
| F (B1 `.40/.60`) | 74,951 | 76,981 | 72,921 |
| **v1 (offset 0)** | **75,108** | **77,127** | **73,088** |
| e05 | 74,290 | 76,168 | 72,412 |
| e10 | 74,150 | 76,031 | 72,268 |
| e15 | 73,989 | 75,936 | 72,041 |
| e20 | 73,929 | 75,776 | 72,081 |
| e25 | 73,960 | 75,847 | 72,074 |
| e30 | 73,859 | 75,754 | 71,963 |

- **Mọi checkpoint đều kém v1:** từ −0,82 pp (e05) đến −1,25 pp (e30). Train càng lâu càng kém, trong khi loss vẫn giảm.
- Theo luật khóa: `NO_SIGNAL_ON_CALIBRATION` → **không đánh giá A**, không chỉnh, không thử biến thể khác.
- So sánh: oracle cùng luật trên cùng tập calibration là +1,84 Clean / +2,08 LR2. Model đi theo chiều ngược lại.

## 4. Cơ chế
**Chưa kiểm**, cần `selection.json` trong zip: GT cứu/mất và độ dịch theo loại box.

**Giả thuyết (suy luận, chưa kiểm):**
- Loss giảm, nhưng Hmean xấu đi theo epoch. Đây là dấu hiệu lỗi nhỏ trên số đông box TP gây hại nhiều hơn lợi từ số ít box được sửa.
- Có 1.698 / 1.467 box TP (A) với IoU 0,5–0,6 (oracle §2 của spec). Trên TRAIN, 96% cặp là TP với độ dịch trung vị khoảng 0,05 cạnh. Một sai số cỡ đó đủ đẩy box sát ngưỡng xuống dưới 0,5.
- Smooth-L1 trung bình không phạt riêng việc vượt ngưỡng IoU.
- Hiện tượng cùng chiều ở dry-run không có P-map: thua v1 0,25–0,5 pp.

Zip sẽ cho biết số GT mất (TP bị sửa hỏng) so với số GT cứu được, và độ dịch trên box không nhãn.

## 5. Kết luận
- Trong lượt này, learned box refinement **không cho tín hiệu** trên TRAIN-calibration, dù oracle cùng luật cho dư địa lớn.
- Box-quality v1 `SMALL_USEFUL` và mọi cổng/kết quả cũ không đổi.
- **Không được tuyên bố:** bất kỳ số nào trên A cho hướng này (A chưa được đánh giá).
- Chi phí đo được: crop median 7,6 ms/task, so với forward 394 ms. Latency end-to-end chưa đo (Ô 9 không chạy).
