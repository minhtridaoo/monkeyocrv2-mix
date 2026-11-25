> **Cấu trúc mới 24/11:** xem [hướng dẫn nhập repo chung](MERGE_GUIDE_janek.md). Tên file/import đã có hậu tố `_janek`; trạng thái nghiên cứu mới nhất ở [HANDOFF](HANDOFF_janek.md).

# Bắt đầu ở đây — workflow Mix A / Mix B

**Sắp xếp tài liệu ngày 19/11/2026.** Thư mục này dùng một kế hoạch chính; các bản khảo sát/review đã thay thế được giữ trong `_archive/` để tra lịch sử.

**Tiến độ mới nhất đã kiểm (20/11/2026):** [Mix A-core v1](docs/handoff/artifacts/janek/mix_a_v1/NOTES_janek.md) đã chạy trên A100 và đạt trạng thái tái lập `VERIFIED` theo chính sách cùng GPU. Quy tắc đã khóa chọn α = 0 (B-Clean), nên Mix A chưa thắng. [Phân tích chọn có ràng buộc tiny](docs/handoff/artifacts/janek/explore_sel_v1/NOTES_janek.md) chỉ là khám phá và cũng không tìm được ứng viên đạt mục tiêu. [Pilot BN](docs/research/janek/design_bn_recalibration_pilot_janek.md) mới ở mức thiết kế, chưa chạy. 60/60 test local PASS. Repo Tuấn hiện ở `aa1e17f`; kết quả Mix A v1 vẫn ghim vào `dc19b85`. Xem [HANDOFF_janek.md](HANDOFF_janek.md) để biết việc tiếp theo.

## Mở file nào?

| Mục đích | File cần dùng | Vai trò |
|---|---|---|
| Làm việc hằng ngày, code theo kế hoạch, trao đổi hoặc gửi Tuấn | [Kế hoạch phối hợp Mix A/Mix B](docs/research/janek/Ke_hoach_phoi_hop_MixA_MixB_20261119_janek.md) | **Tài liệu chính duy nhất về phạm vi, pipeline, protocol pilot và việc cần thống nhất** |
| Tra kết quả B0/B-Clean/B1, EDA và các pilot trước | [Báo cáo kết quả ngày 18/11](docs/research/janek/MonkeyOCRv2_TextDetection_Progress_Results_Research_Roadmap_janek.md) | Bối cảnh/kết quả lịch sử; phần roadmap trong đó không phải kế hoạch triển khai mới nhất |
| Tick tiến độ trong trình duyệt | [Dashboard](tools/dashboard_janek/index_janek.html) · [Hướng dẫn](tools/dashboard_janek/PIPELINE_janek.md) | Công cụ theo dõi; không thay tài liệu chính hay artifact kiểm chứng |

**Gửi Tuấn:** [nháp báo kết quả Mix A v1](docs/research/janek/msg_tuan_20261120_mixA_result_janek.md) cần user đọc lại trước khi gửi; gửi kèm [kế hoạch chính](docs/research/janek/Ke_hoach_phoi_hop_MixA_MixB_20261119_janek.md) khi cần đối chiếu quy tắc. Nháp chưa được gửi.

## Cách cập nhật để không bị lệch phiên bản

1. Thay đổi phương pháp, phân công, điều kiện mở nhánh hoặc protocol: sửa kế hoạch chính. Ghi rõ đó là đề xuất hay đã được hai bên thống nhất.
2. Thay đổi tiến độ: cập nhật dashboard kèm đường dẫn commit/job/artifact nếu có. Dấu tick không tự xác nhận job PASS hay Tuấn đã đồng ý.
3. Có kết quả mới: ghi kèm run identity, trạng thái kiểm chứng và nguồn; không ghi đè số liệu lịch sử ngày 18/11.
4. Không duy trì thêm một bản kế hoạch cá nhân hoặc sơ đồ tĩnh độc lập. Pipeline và phần việc A/B được cập nhật ngay trong mục 2 của kế hoạch chính.

## Đã đưa ra khỏi workflow hiện tại

| File lưu trữ | Lý do |
|---|---|
| [docs/archive/janek/Phan_cua_toi_2_Mix_trong_pipeline_Tuan_janek.md](docs/archive/janek/Phan_cua_toi_2_Mix_trong_pipeline_Tuan_janek.md) | Kế hoạch cá nhân v3 và lịch sử review đã được tích hợp vào bản phối hợp. Một số chỉ dẫn còn nằm ở phần hiệu chỉnh cuối file, dễ bị đọc thiếu |
| [docs/archive/janek/Method_Transfer_Plan_LR2_FP_Geometry_janek.md](docs/archive/janek/Method_Transfer_Plan_LR2_FP_Geometry_janek.md) | Khảo sát phương án cũ, còn hysteresis/DaWin/loss và spec chưa thuộc phạm vi hiện tại. Giữ để tra ý tưởng, không dùng làm yêu cầu triển khai |
| [docs/archive/janek/dashboard_PIPELINE_20261119_janek.md](docs/archive/janek/dashboard_PIPELINE_20261119_janek.md) | Sơ đồ tĩnh cũ trùng kế hoạch chính nhưng khác lịch và phụ thuộc. Đã thay file ở dashboard bằng hướng dẫn ngắn |

Ba bản lưu trữ được di chuyển nguyên nội dung và kiểm tra SHA-256 trước/sau. Không có nội dung nghiên cứu bị xóa vĩnh viễn. Các đường dẫn/trạng thái bên trong chúng là thông tin lịch sử, không phải xác nhận hiện tại.

**Code:** package [mixab/](mixab) (A1 `weight_merge`, A2 `merge_audit`, A3 `reselect_split`, E-sel `explore_selection`, Mix B `fsr`) cùng [tests/](tests). Chạy test: `python -m unittest discover -s tests -t . -v`. Đây là repo riêng; MONKEY_V2 của Tuấn chỉ dùng để tham chiếu pipeline. File trọng số và output lớn ở `runtime/` hoặc Drive, không có trong Git.

**Phiên làm việc:** đọc [HANDOFF_janek.md](HANDOFF_janek.md) (trạng thái hiện tại), ghi nhật ký vào [docs/research/janek/WORKLOG_janek.md](docs/research/janek/WORKLOG_janek.md). Quy tắc chung nằm ở `Đồ án/CLAUDE.md`.

**Lưu ý dashboard (đã đồng bộ tối 19/11, xem tools/dashboard_janek/PIPELINE_janek.md):** bản trước có câu “Tuấn đã duyệt”, lịch train Mix B sớm và cách ghi ba nhánh của một phần pilot. Đây là ghi chép trong dashboard, chưa đủ bằng chứng để tự cập nhật kế hoạch thành “đã thống nhất”. Khi có khác biệt, kiểm tra log/artifact/xác nhận thực tế, rồi đồng bộ với kế hoạch chính; xem hướng dẫn dashboard trước khi dùng lịch làm cam kết.
