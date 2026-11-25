# Kế hoạch tiến tới cổng nhóm sau box-quality dev A

25/11/2026 · Codex · **Kế hoạch, chưa thực hiện thí nghiệm mới.**

## 1. Đích thực tế, tính lại từ JSON gốc

Nguồn: `docs/handoff/artifacts/janek/boxquality_devA_20261125/devA_result_janek.json`. Không trộn mốc ALL/B với A.

| Đại lượng A | Hiện tại FULL | Yêu cầu nhóm | Khoảng thiếu |
|---|---:|---:|---:|
| mean Clean/LR2 | 78,1775% | ≥79,0019% | +0,8244 pp |
| Clean Hmean | 79,3868% | ≥80,2202% | +0,8334 pp |
| tiny LR2 recall | ~51,68% | ≥49,1099% | hiện đạt |

Nếu Clean chỉ đạt tối thiểu 80,2202%, LR2 phải đạt ≥77,7836% để mean đạt. Nếu Clean bằng B-Clean 80,5202%, LR2 vẫn cần ≥77,4836%. Đây là các cặp minh họa trên đường biên, không phải hai cổng LR2 độc lập mới.

Giữ nguyên kết quả FULL cũ SMALL_USEFUL, ngưỡng/model và protocol lịch sử. Không hứa đổi một module sẽ đạt. Không diễn giải chưa đạt cổng thành không được nghiên cứu nữa: biến thể mới phải có spec mới, ghi đúng là hậu kiểm, TRAIN và development tách vai trò, TEST do nhóm quản lý.

## 2. Chưa có bằng chứng bắt buộc đổi backbone

Box-quality chỉ chấm lại một pool box cố định. Nó có thể giữ lại ứng viên raw thấp, nhưng không thể tạo chữ mà pool không có, cũng không sửa polygon trong phiên bản hiện tại.

Vì vậy câu "hậu xử lý không còn đường đạt cổng" trong báo cáo là quá mạnh nếu chưa tính khả năng của pool và cấu trúc lỗi. Có thể không đáng đầu tư thêm, nhưng chưa chứng minh bất khả thi.

Mặc định: không đổi backbone, không thêm neck/fusion mới, không thêm teacher khi inference. Repo đã có latent/fusion/PRC/feature-KD âm. Chỉ chuyển sang train khi xác định rõ cái gì scorer không làm được.

## 3. Giai đoạn 0: kiểm tra khả năng cải thiện, dùng artifact hiện có

Thực hiện CPU trước, không mở B/TEST, không fit thêm trên A:

1. Đối chiếu B-Clean, B1 .40/.60 và FULL trên đúng A/Clean: lỗi TP mất, FP thừa, GT được B-Clean nhận nhưng B1 không nhận. Dùng matching chính thức, không nhầm net TP với số GT mất.
2. Với GT B1/FULL bỏ sót: trong pool B1 raw≥.20 còn box IoU≥.5 không? Tách có ứng viên nhưng bị score lọc / có overlap nhưng sai geometry / không có ứng viên.
3. Tính một giới hạn lạc quan của pool: maximum matching tới GT valid sau ignore/ROI. Nếu có thể chọn đúng một box cho mỗi GT khớp, precision=1 và H=2M/(GT+M). Đây chỉ là oracle dùng GT để chẩn đoán, KHÔNG phải model hoặc mức tăng khả thi. Giới hạn thấp hơn cổng thì loại nhánh scorer; cao hơn không chứng minh học được.
4. Tính các kịch bản "cứu nhóm score-filtered" hoặc "loại nhóm FP" riêng để định lượng phần nào có đủ dư địa. Không cộng lợi ích độc lập nếu matching tương tác; tính lại toàn bộ tập box.
5. Xem bằng mắt top 10 nguồn tạo lợi ích và một mẫu phân tầng theo nguồn/size/score: FP bị bớt là nền, lệch biên, trùng, hay chữ thiếu nhãn? Không sửa GT hoặc loại các ảnh này khỏi điểm chính.

**Quyết định sau chẩn đoán:**
- Nếu vấn đề chủ yếu là thứ tự score và có tín hiệu TRAIN tách được: xem xét đúng một thay đổi của scorer; chưa cần train detector. Không quét feature/model trên A.
- Nếu cần tạo/sửa box trước scorer, chuyển Giai đoạn 1 dưới đây.
- Nếu phần lớn cải thiện kỳ vọng chỉ là xóa chữ thiếu nhãn, không thiết kế loss để khai thác lỗi nhãn. Báo hạn chế đo lường và xem lại mục tiêu với nhóm.

## 4. Hướng train ưu tiên có điều kiện: khôi phục Clean trong một detector

**Giả thuyết cần kiểm:** có thể cải thiện đầu ra Clean của B1 bằng hướng dẫn của B-Clean trên Clean, trong khi tiếp tục giám sát LR2 bằng GT để giữ tiny. Không khẳng định đây là catastrophic forgetting: hiện chỉ quan sát đánh đổi giữa hai checkpoint.

### Cấu trúc

Inference giữ nguyên: ảnh → backbone AS → FPNC → DBHead → hậu xử lý. Một checkpoint, một forward. Box-quality v1 lưu riêng như kết quả đã có.

Training dự kiến:
- Student khởi tạo B1; vòng đầu freeze backbone và FPNC, chỉ train DBHead.
- B-Clean frozen/eval/no_grad chỉ là teacher trên ảnh Clean, cùng augmentation và lưới với student Clean.
- Clean và LR2 vẫn 50/50 theo TRAIN; B1 đã có replay này nên không gọi 50/50 là đóng góp.
- Loss = DB loss có giám sát Clean/LR2 + λ·Clean-output anchoring. Teacher chỉ xuất hiện khi train, không serialize chung vào model inference.
- Anchoring là giả thuyết, chưa khóa hàm/λ. Spec phải nêu rõ công thức, pixel mask, normalization và các vùng bị bỏ trước khi chạy. Không dùng teacher score cao làm chân lý, không phạt GT foreground chỉ vì teacher không thấy; phải ưu tiên GT, loại ignore/pad và vùng bất định.
- Việc giảm FP cần cả phần nền đáng tin. Không có mask nền đáng tin hoặc teacher liên tục mâu thuẫn GT thì chưa đủ điều kiện mở arm anchoring. Không dùng mọi vùng thiếu GT như negative đặc biệt do đã biết thiếu nhãn.

### Đối chiếu các hướng cũ — bắt buộc

Đã đọc code:
- `MONKEY_V2/rev4/prc_training.py`: PRC = masked probability MSE, frozen **B0**, dùng Clean teacher để dạy **LR2**; `prc_plugins.py` loss gắn vào sample LR2.
- `feature_distillation_plugins.py`: teacher Clean có feature 2×, student feature LR2, masked feature loss.

Đề xuất này khác ở **nhánh Clean→Clean, teacher B-Clean và mục tiêu khôi phục Clean**, không phải Clean→LR2 hay feature restoration. Nhưng vẫn cùng họ distillation/regularization. Chỉ khác tên teacher chưa đủ: cần audit TRAIN cho thấy B-Clean thực sự tốt hơn trên vùng cần sửa và teacher signal tương thích GT. Không tuyên bố kiến trúc mới.

### Các arm tối thiểu

| Arm | Thay đổi | Vai trò |
|---|---|---|
| B1 frozen | không train | mốc hiện tại |
| C | B1 train tiếp DBHead, DB loss, Clean/LR2 50/50 | đối chứng lợi ích do train thêm |
| P | giống C, thêm Clean anchoring | kiểm đóng góp phương pháp |

C/P cùng init, TRAIN IDs, augmentation RNG, optimizer, batch hiệu dụng, updates, lịch checkpoint, cách chọn checkpoint. Nếu P không hơn C thì không ghi lợi ích là do anchoring.

Nếu head-only không đủ, không tự động mở neck/backbone để chạy tiếp. Chỉ cân nhắc thêm FPNC khi probe cho thấy head không biểu diễn được hướng sửa, và khi đã đối chiếu với train-neck control của fusion cũ. Backbone là bước cuối, chưa có căn cứ đổi lúc này.

## 5. Dữ liệu, probe và ngân sách

- Dùng TRAIN đã khóa; hiện có ảnh/GT của box-quality nhưng ROI TRAIN là toàn ảnh. Cần ghi đúng khác biệt với recipe lịch sử, dùng cùng nhãn cho C/P và kiểm cách shrink kernel/ignore trong DB loss. Không tự coi audit scorer đã chứng minh tương đương training detector.
- Tách TRAIN-fit và TRAIN-selection theo nguồn; Clean/LR2 cùng nguồn. B1/B-Clean từng train trên cả hai, selection chỉ phục vụ chọn cấu hình mới, không phải xác nhận độc lập.
- Probe vài batch trước: student step 0 trùng B1, teacher frozen, gradient chỉ vào DBHead, augmentation/lưới đúng, ignore/pad không vào loss, tiny kernels còn có pixel hữu hiệu, checkpoint resume đúng.
- Đo thật VRAM, sec/update và sec/evaluation trên T4/A100 đang dùng. Teacher tăng chi phí TRAIN dù không tăng inference. Chưa có ETA hợp lệ; không hứa train xong trong 1–2 giờ.
- Sau profile mới khóa tổng updates/thời gian cho C/P. Không chạy thêm vì đường val đang "có vẻ sắp tăng". Khóa λ theo căn cứ TRAIN hoặc phép cân bằng loss/gradient mô tả trước; không quét λ trên A.
- Không thay đồng thời resolution, loss, augmentation, sampler và backbone. Chỉ thêm Clean anchoring trong phép so đầu.

## 6. Scorer sẽ ghép ở đâu?

Đánh giá detector mới **không scorer trước** để biết có sửa được nền hay không. Sau khi detector được chọn và khóa trên TRAIN-selection, có thể chuẩn bị một scorer mới theo đúng recipe cũ trên TRAIN, vì P-map/score distribution của checkpoint mới đã thay đổi.

Không mặc định đem logistic v1 gắn sang checkpoint mới vẫn đúng; cũng không tune lại scorer trên A. Nếu ngân sách cho phép phép ghép, phải có detector mới ± scorer, giữ v1 nguyên vẹn làm đối chứng. Không cộng số +0,22 pp cũ vào điểm detector mới để dự báo.

## 7. Đánh giá và quyết định

- Ưu tiên TRAIN-selection để chọn C/P/checkpoint; sau khóa mới đánh giá A một lần cả Clean/LR2 với baseline cùng GPU.
- Báo P−C, P−B1, P−B-Clean; nếu ghép scorer, thêm P+scorer−P. Không chỉ so với baseline yếu nhất.
- Báo đúng cổng mean ≥ best single +.50, Clean ≥ B-Clean−.30, tiny LR2 ≥ B1−3; CI ghép cặp nguồn cho các so sánh, phân bố và độ tập trung lợi ích. Không cho mean bù cho một cổng Clean/tiny trượt.
- Chứng minh inference chỉ một checkpoint/forward, đo median/p95/VRAM cùng protocol. Scorer overhead được đo riêng.
- A/B đều là development đã dùng; CI không loại bỏ selection bias của toàn chương trình nghiên cứu. Benchmark cuối chỉ sau quyết định nhóm, không dùng để sửa phương án.
- P tăng Clean nhưng mất LR2/tiny: thất bại trade-off. P=C: chưa có đóng góp của anchoring. Loss giảm mà Hmean không tăng: không coi là thành công.

## 8. Việc giao Claude ngay

1. Làm Giai đoạn 0 bằng dữ liệu có sẵn; lưu bảng khoảng thiếu và phân loại lỗi, kèm oracle được ghi nhãn rõ.
2. Trả lời head/score/candidate nào đang giới hạn kết quả; đối chiếu với các phương án cũ.
3. Nếu cần train và teacher audit ủng hộ: viết spec C/P, chuẩn bị notebook probe và đo budget, chưa chạy GPU dài. User chạy GPU.
4. Nếu chẩn đoán không ủng hộ hướng train này, dừng tại bằng chứng; không tự thay bằng module đang phổ biến.

Mục tiêu thực tế là tìm một phương án qua cổng bằng bằng chứng, không thể đảm bảo trước. Box-quality SMALL_USEFUL vẫn được giữ làm kết quả riêng dù nhánh train thất bại.

Tham khảo khái niệm giữ năng lực bằng ràng buộc đầu ra khi huấn luyện: Learning without Forgetting, https://arxiv.org/abs/1606.09282 (đã đọc abstract; không khẳng định recipe trên tái hiện paper hoặc hiệu quả cho TextOCR).
