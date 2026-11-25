# E-BN Giai đoạn 0: đo độ lệch thống kê BN (KHÁM PHÁ), Colab T4, 20/11/2026

**Chốt (21/11, user):** Giai đoạn 0 không tìm thấy tín hiệu BN bất thường riêng do trộn trọng số. **Không mở Giai đoạn 1**, vì bằng chứng hiện tại chưa đủ để ưu tiên hướng này. Kết luận Mix A-core v1 giữ nguyên: chưa vượt baseline theo tiêu chí đã khóa.

**Kết quả: trong cách đo này, chưa thấy sai lệch BN đáng kể do phép trộn.**
- Thống kê đo được của các α ở giữa gần như đúng bằng nội suy tuyến tính thống kê đo được của hai endpoint: lệch trung bình ≤ 0,005 SD, lớn nhất ≤ 0,014 SD.
- Mức lệch giữa thống kê đã lưu và thống kê đo được của α ở giữa **không lớn hơn** của endpoint.
- Cổng đề xuất (chưa duyệt) cho **không có tín hiệu**.

Kết luận Mix A-core v1 không đổi. Đây là **chẩn đoán BN trên T4**, không phải kết quả chất lượng phát hiện chữ, nên không đem trừ trực tiếp với điểm A100 của v1.

Nguồn: thư mục này (bản sao JSON), bản gốc `MyDrive/CS431_MixA/runs/bn_stage0_v1_T4/` và `export/cs431_bn_stage0_T4_20261120_085809.zip`. Code: `mixab/bn_recal_janek.py`. Đã chạy đúng code và dữ liệu đã kiểm: bản `mixab_code.zip` đã upload trùng từng file với code local (26/26 file khớp `CODE_MANIFEST.json`). Test phần mềm PASS chỉ xác nhận các kiểm tra phần mềm đạt, không xác nhận phương pháp nghiên cứu có hiệu quả.

## Đầu vào và điều kiện chạy

- **TRAIN:** `train_ids.csv` có SHA-256 `8ac5cbd9…5f60`, 15.222 dòng.
  - Trùng **đúng** file Tuấn công bố ở MONKEY_V2 commit `e4a2a89` (`docs/handoff/data/train_ids.csv`, kèm `train_ids_receipt_janek.json`). Receipt ghi file được xuất từ manifest gốc SHA `ca1b9163…c983d` và giữ nguyên thứ tự dòng.
  - Notebook ghi `NOT_VERIFIABLE_ID_LIST` vì mình không có manifest gốc để tự hash. Việc khớp manifest dựa trên receipt của Tuấn.
- **Kiểm danh sách:** PASS. Đúng 15.222 nguồn, không trùng ID hay SHA với 2.262 nguồn remaining-validation. 1.000 nguồn S1/S2 không kiểm được.
- **Subset:** 1.000 nguồn (seed 20261120), subset SHA `0463e5d0…`.
  - 1.000 ảnh Clean khớp SHA từng ảnh.
  - 1.000 LR2 tạo bằng `native_lr2` thật (Pillow 11.3.0 trên Colab), SHA ghi ở `subset_images.csv` (lần đầu, chưa có lần tạo lại để so).
- **Môi trường:**
  - GPU **Tesla T4**, không phải A100 như hướng dẫn. T4 không có TF32; doctor 0 FAIL, 1 WARN đúng mục này.
  - torch 2.9.0+cu128, cuDNN 91002. MONKEY_V2 `dc19b85` sạch. Cùng 5 checkpoint v1 (merge manifest `8582c9f3…`).
- **Cách đo:** model ở `eval()`, hook đọc đầu vào `det_head.binarize.1` và `.4`. SHA `state_dict` trước và sau bằng nhau ở cả 5 model (code dừng nếu khác).
- **Thời gian:** 0,47 s/ảnh trên T4, khoảng 15,7 phút mỗi model, tổng khoảng 79 phút.

## Số đo

`z` = |μ_đo − μ_lưu| / σ_lưu, trung bình theo kênh; `lv` = |log(var_đo / var_lưu)|, trung bình. Dữ liệu là 2.000 ảnh (1.000 Clean + 1.000 LR2).

| Lớp | α 0 | α ¼ | α ½ | α ¾ | α 1 |
|---|---|---|---|---|---|
| `binarize.1` z | 0,161 | 0,159 | 0,158 | 0,157 | 0,158 |
| `binarize.1` lv | 0,778 | 0,769 | 0,763 | 0,759 | 0,759 |
| `binarize.4` z | 0,097 | 0,093 | 0,091 | 0,089 | 0,089 |

Clean và LR2 cho mức gần như nhau ở mọi model; chênh lệch ≤ 0,009 về z.

**Phi tuyến:** độ lệch của thống kê đo được của α so với nội suy tuyến tính từ thống kê đo được của hai endpoint.

| Lớp | α ¼ | α ½ | α ¾ |
|---|---|---|---|
| `binarize.1` z trung bình / lớn nhất | 0,004 / 0,011 | 0,005 / 0,014 | 0,004 / 0,010 |
| `binarize.4` z trung bình | 0,002 | 0,003 | 0,002 |

**Cổng đề xuất (chưa duyệt):** z của α ở giữa trên `binarize.1` phải lớn hơn 2 lần mức lớn nhất của endpoint (0,161). Các α ở giữa đạt 0,157–0,159, nên **không có tín hiệu**. Thực tế chúng còn thấp hơn mức của endpoint.

## Diễn giải

- **Đã đo:** với cách đo này (trong từng ảnh, 2 lớp BN `binarize`, 2.000 ảnh TRAIN), thống kê của α ở giữa gần như tuyến tính theo α, và lệch so với thống kê đã lưu không lớn hơn endpoint. Vậy **chưa thấy sai lệch đáng kể do phép trộn**. Điều này **không** chứng minh thống kê BN hiện tại là tối ưu cho phát hiện chữ, và không loại trừ hiệu ứng mà cách đo này không thấy được.
- **Lệch nền chung của mọi model, kể cả B-Clean và B1 gốc:** phương sai đo được bằng khoảng **0,45 lần** phương sai đã lưu, ở 100% số kênh của cả hai lớp.
  - **Suy luận, chưa kiểm:** nguyên nhân có thể nằm ở cách đo. Code lấy phương sai trong từng ảnh rồi lấy trung bình (giống `momentum=None` với batch 1), còn thống kê lúc train được tính trên micro-batch 2 ảnh (`rev4/training_data.py:20`, `micro_batch_size=2`) qua pipeline train có augmentation, nên có cả phần chênh lệch giữa các ảnh.
  - Accumulator có lưu tổng pixel gộp, nhưng file tạm đã xóa sau khi mỗi model xong, nên chưa tính được phương sai gộp để kiểm giả thuyết này.
  - Lệch nền này không ảnh hưởng hai phép so sánh chính (α ở giữa so với endpoint, và phi tuyến), vì nó giống nhau ở mọi model.
- **Hệ quả cho Giai đoạn 1 nếu có ai muốn mở:** cập nhật BN theo thống kê đo hiện tại (batch 1, trung bình trong ảnh) **có thể làm thay đổi đáng kể cả endpoint**, vì `running_var` sẽ đổi khoảng 0,45 lần. **Chưa biết thay đổi đó tốt hay xấu**, vì chưa chạy. Đó sẽ là một can thiệp hiệu chuẩn chung chứ không riêng Mix A, nên nếu làm thì phải thiết kế lại cách tính (phương sai gộp hoặc batch giống lúc train) như một câu hỏi riêng.
- **Giới hạn của phương pháp đo:** phương sai trong từng ảnh có thể khác cách BN tính lúc train (micro-batch 2 ảnh, pipeline có augmentation). Chênh lệch 0,45 lần vì vậy **không tự chứng minh pipeline của Tuấn có lỗi**; nhiều khả năng nó phản ánh khác biệt cách đo (suy luận, chưa kiểm).
- **T4 thay vì A100:** cả 5 model được đo cùng điều kiện trên T4, nên phép so sánh α ở giữa với endpoint vẫn có giá trị. Đây là chẩn đoán BN, không phải điểm chất lượng để so trực tiếp với kết quả A100.

## Quyết định

- **Dừng E-BN ở Giai đoạn 0. Không mở Giai đoạn 1 cho Mix A** (user chốt ngày 21/11).
- Đóng Mix A-core bằng báo cáo kết quả âm, gồm v1, E-sel và E-BN Giai đoạn 0.
- Câu hỏi về lệch nền khoảng 0,45 lần (nếu Tuấn quan tâm) là việc riêng của pipeline, không thuộc Mix A.
