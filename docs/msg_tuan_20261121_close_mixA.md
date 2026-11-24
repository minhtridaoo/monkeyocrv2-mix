# Nháp tin nhắn gửi Tuấn, 21/11: đóng Mix A v1 (user đọc lại và sửa trước khi gửi)

---

Tuấn ơi, mình chốt Mix A như sau. Báo cáo đầy đủ ở `docs/report_mix_a_v1.md`.

- **Mix A-core v1:** theo R1 đã khóa, α\* chọn trên A là B-Clean (endpoint), nên Mix A chưa vượt baseline theo tiêu chí đã khóa.
  - Trên B, các α ở giữa chỉ hơn endpoint tốt nhất 0,17–0,23 pp, và mọi CI đều chứa 0.
  - Kiểm chứng `VERIFIED` cùng A100: P-map 0,0, counts endpoint trùng 100% với probe.
- **Đổi cách chọn (ràng buộc tiny, khám phá):** α ¾ được chọn nhưng vẫn trượt +0,5 pp và tiny (−3,07 pp). Ở cùng recall nó còn nhiều FP hơn B1 6–7%.
- **Chẩn đoán BN, Giai đoạn 0 (T4, danh sách TRAIN của Tuấn):** trong cách đo này, chưa thấy sai lệch BN đáng kể riêng do phép trộn. Mình **không mở Giai đoạn 1**, vì bằng chứng chưa đủ để ưu tiên.
  - Có một lệch nền: phương sai đo được bằng khoảng 0,45 lần phương sai đã lưu, ở mọi model.
  - Nhiều khả năng đó là khác biệt cách đo (trong từng ảnh so với micro-batch lúc train). Nó không tự chứng minh pipeline có lỗi, và mình không bàn tiếp phần này.
- **Đóng Mix A v1** là hoàn tất một thí nghiệm có kết quả âm, không phải bỏ mục tiêu cải tiến.

**Bước tiếp mình đề xuất:** chờ kết quả latent 10.000 update của Tuấn.
- Nếu A1/A2 có tín hiệu so với A0 và B1, mình đề xuất thêm FSR và gate-only (Mix B) thành hai nhánh trên **cùng recipe**, với A0/A1 làm đối chứng, sau khi thống nhất thân R và tiêu chí.
- Nếu không có tín hiệu, mình nghĩ Mix B cũng nên tạm dừng.

Tuấn thấy sao?
