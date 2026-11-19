# Hướng dẫn dashboard tiến độ

**Pipeline và phạm vi công việc:** đọc [kế hoạch chính](../Ke_hoach_phoi_hop_MixA_MixB_20261119.md), đặc biệt mục 2 có khung Mix A/Mix B. File này chỉ hướng dẫn dùng dashboard, không duy trì thêm một bản pipeline hoặc lịch riêng.

## Cách dùng

- Mở [index.html](index.html) để xem/tick tiến độ. Dữ liệu mặc định nằm trong [status.js](status.js).
- Thay đổi trong trình duyệt được giữ ở localStorage của trình duyệt đó; chúng không tự ghi vào file `status.js` trên đĩa.
- Muốn chia sẻ các tick: dùng nút **Tải status.js**, kiểm tra file tải về rồi thay file dữ liệu trong cùng thư mục dashboard.
- Trạng thái công việc nên có bằng chứng tương ứng: commit, kết quả kiểm thử, job ID, log hoặc artifact. Không đánh đồng “đã code”, “đã submit”, “job xong” và “đã kiểm chứng”.

## Những chỗ phải đối chiếu với kế hoạch chính

- Câu “Tuấn đã duyệt” trong dữ liệu mặc định cần đối chiếu xác nhận thực tế; hoàn tất review nội bộ không tự chứng minh Tuấn đã đồng ý.
- Lịch là dự kiến. Ngày đệm 22/11 và ưu tiên Mix A/báo cáo nằm trong kế hoạch chính; lịch dashboard không tự trở thành cam kết train Mix B trước 24/11.
- Có thể chuẩn bị Mix B khi chờ Mix A. Train đầy đủ vẫn cần taxonomy, kết quả lỗi còn lại, residual control sẵn sàng/probe PASS, recipe và tiêu chí đã khóa.
- “Ba nhánh” Residual/gate-only/FSR chỉ là nhóm so sánh của Mix B. Ma trận toàn nhóm có năm nhánh nếu giữ cả Head-control và Residual+KD của Tuấn.
- Các task ID A1–A6/B1–B4 của dashboard là mã quản lý việc, không phải tên checkpoint B1 hay hàng ablation. Đọc theo tiêu đề task.
- `VERIFIED` chỉ dùng khi các kiểm tra bắt buộc trong kế hoạch đều PASS; trạng thái dashboard không thay được các kiểm tra đó.

**Cập nhật 19/11 tối:** `status.js` đã được đồng bộ theo kế hoạch chính.
- Task ID A1–A4 nay khớp §2.3: A1 weight_merge, A2 merge_audit, A3 reselect_split, A4 đánh giá.
- Bỏ câu "Tuấn đã duyệt"; thay bằng mô tả đúng mức xác nhận.
- Mix B không còn xếp lịch train trước 24/11.
- Thêm trạng thái job 54308 lấy từ `pr-quad-audit-v1.md` @ `37384cf`.

**Cập nhật tiếp sau khi fetch `dc19b85`:** T3 đã hoàn tất (recovery 63985); gói counts/weights kiểm local PASS. T4 vẫn chưa có nhãn (0/160). NPZ/B-Clean đã có, A3 chuyển sang kiểm tích hợp dữ liệu thật; A2 chưa viết. Chi tiết hiện hành ở [HANDOFF](../HANDOFF.md) và §0 kế hoạch. Nếu trình duyệt vẫn hiện tick cũ, kiểm tra localStorage trước; cập nhật file không tự xóa lựa chọn đã lưu trong trình duyệt.

Dữ liệu và các tick hiện có được giữ nguyên trong lần sắp xếp tài liệu này. Sơ đồ tĩnh trước đây được giữ tại [bản lưu trữ](../_archive/dashboard_PIPELINE_20261119.md); nội dung đó không còn dùng làm hướng dẫn triển khai.
