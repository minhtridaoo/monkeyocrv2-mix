# Bắt đầu ở đây — workflow Mix A / Mix B

**Sắp xếp tài liệu ngày 19/11/2026.** Thư mục này dùng một kế hoạch chính; các bản khảo sát/review đã thay thế được giữ trong `_archive/` để tra lịch sử.

**Tiến độ mới nhất đã kiểm:** repo Tuấn `dc19b85`, recovery 63985 xong; gói NPZ/GT/EDA và weights B-Clean/B1 đã tải, `verify_bundle.py --counts --weights` PASS. 29/29 test local của mình PASS; review tay 0/160. Claude đọc [HANDOFF.md](HANDOFF.md) rồi §0 kế hoạch để tiếp tục A3 trên NPZ thật và viết A2; không còn chờ B-Clean/NPZ. Chưa có kết quả inference/train Mix A/B.

## Mở file nào?

| Mục đích | File cần dùng | Vai trò |
|---|---|---|
| Làm việc hằng ngày, code theo kế hoạch, trao đổi hoặc gửi Tuấn | [Kế hoạch phối hợp Mix A/Mix B](Ke_hoach_phoi_hop_MixA_MixB_20261119.md) | **Tài liệu chính duy nhất về phạm vi, pipeline, protocol pilot và việc cần thống nhất** |
| Tra kết quả B0/B-Clean/B1, EDA và các pilot trước | [Báo cáo kết quả ngày 18/11](MonkeyOCRv2_TextDetection_Progress_Results_Research_Roadmap.md) | Bối cảnh/kết quả lịch sử; phần roadmap trong đó không phải kế hoạch triển khai mới nhất |
| Tick tiến độ trong trình duyệt | [Dashboard](dashboard/index.html) · [Hướng dẫn](dashboard/PIPELINE.md) | Công cụ theo dõi; không thay tài liệu chính hay artifact kiểm chứng |

**Gửi Tuấn:** chỉ gửi `Ke_hoach_phoi_hop_MixA_MixB_20261119.md`. Gửi thêm báo cáo kết quả nếu cần đối chiếu số liệu; không cần gửi các bản cũ.

## Cách cập nhật để không bị lệch phiên bản

1. Thay đổi phương pháp, phân công, điều kiện mở nhánh hoặc protocol: sửa kế hoạch chính. Ghi rõ đó là đề xuất hay đã được hai bên thống nhất.
2. Thay đổi tiến độ: cập nhật dashboard kèm đường dẫn commit/job/artifact nếu có. Dấu tick không tự xác nhận job PASS hay Tuấn đã đồng ý.
3. Có kết quả mới: ghi kèm run identity, trạng thái kiểm chứng và nguồn; không ghi đè số liệu lịch sử ngày 18/11.
4. Không duy trì thêm một bản kế hoạch cá nhân hoặc sơ đồ tĩnh độc lập. Pipeline và phần việc A/B được cập nhật ngay trong mục 2 của kế hoạch chính.

## Đã đưa ra khỏi workflow hiện tại

| File lưu trữ | Lý do |
|---|---|
| [_archive/Phan_cua_toi_2_Mix_trong_pipeline_Tuan.md](_archive/Phan_cua_toi_2_Mix_trong_pipeline_Tuan.md) | Kế hoạch cá nhân v3 và lịch sử review đã được tích hợp vào bản phối hợp. Một số chỉ dẫn còn nằm ở phần hiệu chỉnh cuối file, dễ bị đọc thiếu |
| [_archive/Method_Transfer_Plan_LR2_FP_Geometry.md](_archive/Method_Transfer_Plan_LR2_FP_Geometry.md) | Khảo sát phương án cũ, còn hysteresis/DaWin/loss và spec chưa thuộc phạm vi hiện tại. Giữ để tra ý tưởng, không dùng làm yêu cầu triển khai |
| [_archive/dashboard_PIPELINE_20261119.md](_archive/dashboard_PIPELINE_20261119.md) | Sơ đồ tĩnh cũ trùng kế hoạch chính nhưng khác lịch và phụ thuộc. Đã thay file ở dashboard bằng hướng dẫn ngắn |

Ba bản lưu trữ được di chuyển nguyên nội dung và kiểm tra SHA-256 trước/sau. Không có nội dung nghiên cứu bị xóa vĩnh viễn. Các đường dẫn/trạng thái bên trong chúng là thông tin lịch sử, không phải xác nhận hiện tại.

**Code:** package [mixab/](mixab/) (A1 `weight_merge`, A3 `reselect_split`, Mix B `fsr`) cùng [tests/](tests/). Chạy test: `python -m unittest discover -s tests -t . -v`. Thư mục này sẽ thành repo riêng; MONKEY_V2 của Tuấn chỉ dùng để tham chiếu pipeline.

**Phiên làm việc:** đọc [HANDOFF.md](HANDOFF.md) (trạng thái hiện tại), ghi nhật ký vào [WORKLOG.md](WORKLOG.md). Quy tắc chung nằm ở `Đồ án/CLAUDE.md`.

**Lưu ý dashboard (đã đồng bộ tối 19/11, xem dashboard/PIPELINE.md):** bản trước có câu “Tuấn đã duyệt”, lịch train Mix B sớm và cách ghi ba nhánh của một phần pilot. Đây là ghi chép trong dashboard, chưa đủ bằng chứng để tự cập nhật kế hoạch thành “đã thống nhất”. Khi có khác biệt, kiểm tra log/artifact/xác nhận thực tế, rồi đồng bộ với kế hoạch chính; xem hướng dẫn dashboard trước khi dùng lịch làm cam kết.
