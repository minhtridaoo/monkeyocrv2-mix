# Đối chiếu thiết kế v2 với khung thực chiến và ánh xạ skill · 22/11/2026

**Phạm vi:** chỉ đọc và đối chiếu tài liệu. Không đổi số liệu, không chạy GPU, không sửa thiết kế v2. Mọi "áp dụng" dưới đây là áp dụng lên **nội dung đã có** của `docs/research/janek/research_v2_design_20261121_janek.md`, không tạo bằng chứng thực nghiệm mới.

**Nhãn bằng chứng:** `[đã kiểm]` có nguồn mở được · `[suy luận]` từ dấu hiệu gián tiếp · `[chưa kiểm]`.

---

## 1. File khung: "nội dung phù hợp" khác với "đã xác nhận là file thay thế"

| Điểm | Quan sát | Nhãn |
|---|---|---|
| Master prompt gọi gì | `Chỉ dẫn master prompt gửi gemini.md` dòng 20: "File khung tư duy thực chiến của nhóm" tại `LABB\workspace\h_ng_d_n_th_c_chi_n_quy_tr_nh_nghi_n_c_u_ai_ocr.md`. Master prompt **không mô tả nội dung** của khung, nên không có tiêu chí để so "đúng nội dung". | đã kiểm |
| File đó có trên máy không | `find` theo `*h_ng_d_n*`, `*th_c_chi_n*` trong `LABB/`: không ra file nào. `LABB/workspace/` chỉ có `Chỉ dẫn master prompt gửi gemini.md`, `Step by step research.md`, `soạn nội dung slide.md` và vài thư mục. | đã kiểm |
| Ứng viên | `Step by step research.md` (5.306 byte, sửa 2/11), tiêu đề "Hướng Dẫn Thực Chiến Quy Trình Nghiên Cứu Khoa Học AI/OCR (Phiên Bản Nâng Cấp)". `grep "Hướng Dẫn Thực Chiến"` trên `LABB/*.md` chỉ ra đúng file này. | đã kiểm |
| Dấu hiệu cùng một tài liệu | Phần chữ ASCII còn lại trong tên `h_ng_d_n_th_c_chi_n_quy_tr_nh_nghi_n_c_u_ai_ocr` khớp các từ trong tiêu đề (hướng dẫn, thực chiến, quy trình, nghiên cứu, AI/OCR), **trừ** cụm "Khoa Học". Master prompt ghi 5/11, ứng viên sửa 2/11 (ứng viên có trước). | suy luận |
| Dấu hiệu chưa chắc | Tên file khác hẳn; không có bản sao, hash, hay ghi chú nào nối hai file; tiêu đề có "Phiên Bản Nâng Cấp" (có thể còn một bản gốc). | suy luận |
| **Kết luận** | `Step by step research.md` **có nội dung phù hợp** (khung quy trình nghiên cứu AI/OCR). **Chưa xác nhận** là file mà master prompt trỏ tới. Cần user xác nhận. Trong lúc chờ, gọi nó là "khung ứng viên". | |

**Hai đường dẫn khác trong master prompt:**
- `ai_research_failure_modes.md`: có đúng chỗ `LABB/academic-research-skills/academic-pipeline/references/` `[đã kiểm]`.
- `BAT_DAU_SESSION_MOI.md` (ghi ở `workspace/`): thực tế chỉ tìm thấy tại `workspace/lab cô quý/đề tài 2/VAD_IEMOCAP/`, chưa đọc `[đã kiểm vị trí]`.

**Chưa kiểm:** master prompt viết cho "Đề tài 2 – Lab cô Quý", với ví dụ phân loại (Macro-F1). Chưa kiểm đồ án CS431 (text detection, Hmean) có dùng chung quy ước đó không. Khung ứng viên thì nêu ví dụ OCR/TextVQA nên hợp bối cảnh hơn.

---

## 2. Đối chiếu thiết kế v2 với khung ứng viên (8 bước + 4 tầng insight)

| Khung ứng viên | Thiết kế v2 / workspace hiện có | Mức |
|---|---|---|
| §2A: đúng 1 core gap, 1–2 sub-gap; gap sinh ra từ tổng hợp hàng chục tài liệu | §1.4 nêu một gap (`research_v2_design…md:68`) nhưng ghép hai ý (định tuyến size-free **và** kết hợp bất đối xứng). Chỉ ~7 nguồn, phần lớn mới đọc abstract hoặc kết quả tìm kiếm (§1.4 tiêu đề). | **Một phần**: gap chưa tách core/sub-gap, nền tài liệu mỏng |
| Bước 3: đánh giá công bằng, Data Audit + shared manifest | Bundle `pr_quad_audit_v1`, `tasks_metadata.csv`, evaluator và split A/B của Tuấn (HANDOFF "SỰ THẬT ĐÃ KIỂM"). Số trên B gắn nhãn khám phá. | **Có** (do Tuấn; v2 kế thừa) |
| Data Insight | P0/Q4: cạnh dài Clean ≥ 768, LR2 ≤ 512, router kích thước = oracle (HANDOFF 22/11, `diagnosis_janek.json`). Đây đúng loại data insight, và nó lộ shortcut. | **Có** |
| Baseline Behavior Insight | E1–E3, E8 (§1.1). | **Có** (số B là khám phá) |
| Bước 4: soi Error Cases / Overlay Inspection → Insight 1 | E9, E10 lấy từ báo cáo của Tuấn. Human review Bước 4: **PENDING, 0/160 thẻ có nhãn** (HANDOFF "ĐIỂM CHỜ XÁC NHẬN"). Chẩn đoán v2 dựa trên đếm (counts), không dựa trên nhìn overlay. | **Một phần**: thiếu nhãn người |
| Bước 5: giả thuyết | Thiết kế 1–3, chọn QRoute (§2). | **Có** |
| **Bước 5.5: sanity check 200–300 ảnh lỗi trước khi chạy lớn** | Pilot P1 chạy thẳng 2.000 ảnh TRAIN + 4.524 ảnh A/B (§3). Không có vòng mini-dev 200–300 ảnh. | **Thiếu** |
| Bước 6: ablation + tốc độ | Ablation nêu cho từng thiết kế (§2 mục 4). Latency "chưa đo", chỉ đo khi pilot pass. | **Có (kế hoạch)**, chưa chạy |
| Bước 7: Insight 2 | Chưa có kết quả pilot. | n/a |
| Bước 8: viết cuốn chiếu | Đồ án có báo cáo Mix A v1 (`docs/research/janek/report_mix_a_v1_janek.md`); chưa có phần viết cho v2. | n/a |
| Tầng Component/Ablation, Failure Insight | Chờ P3–P4. | n/a |

**Đề xuất (chưa làm, cần user duyệt vì CLAUDE.md cấm tự thêm bước ngoài kế hoạch):**
1. Thêm Bước 5.5 vào P1: chạy thử đặc trưng router trên khoảng 200–300 ảnh, gồm các ảnh bị gửi sai (3 ảnh Clean 768×768) và ảnh LR2 vốn mờ, trước khi chạy 6.524 ảnh.
2. Tách §1.4 thành 1 core gap + sub-gap.
3. Khi Tuấn cho nhãn Bước 4, đối chiếu lại E9/E10 với overlay.

---

## 3. Ánh xạ tên nhóm → skill có trong phiên

**Nguồn ánh xạ:** `~/.claude/skills/0-autoresearch-skill/references/skill-routing.md` (dòng 20–28, 48–55, 173–178) liệt kê nhóm và skill `[đã kiểm]`. Tên thư mục cài đặt khác tên trong file đó (ví dụ `transformer-lens` so với `transformerlens`; cài đặt là cấu trúc phẳng).

| Nhóm trong master prompt | Skill trong nhóm (theo routing file) | Đã cài (`ls ~/.claude/skills`) | Hợp phát hiện chữ (detection)? |
|---|---|---|---|
| `research-ideation` (`21-research-ideation/`) | routing file chỉ ghi nhóm, không ghi tên skill con. Hai skill đã cài mang tag "Research Ideation", tác giả Orchestra Research: `brainstorming-research-ideas`, `creative-thinking-for-research` | có cả hai `[suy luận: thuộc nhóm này]` | **Có**: độc lập loại mô hình, dùng để lọc/xếp hạng ý tưởng |
| `model-architecture` (`01-model-architecture/`) | Megatron-Core, LitGPT, Mamba, RWKV, NanoGPT | `litgpt`, `mamba`, `rwkv`, `nanogpt` có; **Megatron-Core không có** | **Không**: đều là kiến trúc/huấn luyện LLM. Chưa thấy skill nào về detector (DBNet, ViTAE) trong nhóm |
| `mechanistic-interpretability` (`04-…`) | TransformerLens, SAELens, NNsight, Pyvene | cả bốn có | **Có điều kiện, chưa dùng**: `nnsight` và `pyvene` ghi "any PyTorch architecture"; `transformer-lens`, `saelens` hướng tới transformer/LLM. Ví dụ trong hai SKILL.md đều dùng GPT-2/HF; **chưa kiểm** chúng chạy được trên DBNet/MMOCR. Local không có mmocr (CLAUDE.md) |

**Nhóm B của master prompt** (`/audit-analysis`, `/review-paper-code`): có trên đĩa tại `LABB/.claude/skills/` `[đã kiểm]`, **không** nằm trong project `Đồ án/.claude` và không có trong danh sách skill của phiên này. Chưa kích hoạt được từ đây.

**Chưa đối chiếu:** master prompt ghi "66 skills" và "59 skills Orchestra"; `~/.claude/skills` có 65 mục. Chưa đếm lại theo nhóm.

---

## 4. Skill đã đọc và đã áp dụng

| Skill | Đã đọc | Trạng thái |
|---|---|---|
| `brainstorming-research-ideas` | toàn bộ `SKILL.md` (385 dòng) | **Đã áp dụng** (mục 5): lọc Converge (F10, F1, F7, F8, feasibility) cho 3 thiết kế; F6 cho biên phân phối |
| `creative-thinking-for-research` | dòng 1–110 trên ~22 KB (Framework 1–3) | **Chỉ đã đọc phần đầu, chưa áp dụng**. Skill này tự ghi mục đích là sinh ý tưởng mới; lúc này kế hoạch đã chọn hướng, và CLAUDE.md không cho thêm phương pháp ngoài kế hoạch |
| `nnsight` | dòng 1–90 | **Chỉ xác nhận có cài**, đã đọc phần đầu. Không dùng: pilot v2 chạy trên counts, không cần xem nội bộ model |
| `pyvene` | dòng 1–90 | như `nnsight` |
| `transformer-lens`, `saelens` | chưa mở SKILL.md | **Chỉ xác nhận có cài** |
| `mamba`, `rwkv`, `litgpt`, `nanogpt` | chưa mở SKILL.md | **Chỉ xác nhận có cài**; không hợp detection (mục 3) |
| `0-autoresearch-skill` | chỉ `references/skill-routing.md` và các dòng khớp tên nhóm trong `SKILL.md` | dùng làm nguồn ánh xạ; **chưa chạy workflow** của nó |

---

## 5. Áp dụng `brainstorming-research-ideas` lên 3 thiết kế (chỉ từ nội dung đã có)

**Bộ lọc Converge của skill** (`SKILL.md` "Phase 2"). Đây là đánh giá của mình dựa trên chữ trong `research_v2_design_20261121_janek.md`, **không phải bằng chứng mới**.

| Bộ lọc | 1 · QRoute | 2 · DualAgree | 3 · SRVerify |
|---|---|---|---|
| **Explain-It (F10)**: viết được 2 câu | Được. Câu 1: một model trộn mất ≥ 0,58 pp so với dùng mỗi chuyên gia đúng miền (E8, cận oracle, số B khám phá). Câu 2: chọn chuyên gia bằng router size-free vì hai chuyên gia tách miền. Phần "vì sao hoạt động" **chưa kiểm** (P1–P3) | Được nhưng phụ thuộc E9 (1.453 GT); mức trùng FP chưa biết | Được; nền là E11 (+952 FP) |
| **Problem-First (F1)** | Có (E1–E3) | Có (E4, E9) | Có (E10, E11) |
| **Simplicity (F7)** | Đơn giản nhất; router 1 chiều là phương án tối thiểu | 3 tham số (τ_l, τ_c, IoU) chọn trên A: cần canh chừng quá khớp, như bảng so sánh đã ghi | Nhiều thành phần nhất (EDSR + γ); phải thắng "EDSR-only cùng recall" |
| **Stakeholder (F8)** | Vận hành: bộ nhớ gấp 2 checkpoint (câu hỏi 1 §3) | Vận hành: suy luận LR gấp 2 | Vận hành: gấp 2 + SR, latency 205 so với 137 ms |
| **Feasibility** | CPU, pilot không GPU | ~30 phút A100 / ~1 giờ T4 (ngoại suy, chưa đo) | 1–1,5 giờ A100 + dựng EDSR (chưa đo) |

**Kết quả:** bộ lọc không làm đổi lựa chọn đã ghi (Thiết kế 1 trước). Nó chỉ xác nhận lập luận hiện có; không thêm ý tưởng mới.

**F6 (Failure/Boundary probing, trục phân phối):** P0 đã làm một phép thử biên không chủ ý: router kích thước đạt bằng oracle (mean B 78,93). Hệ quả đúng với skill: kết luận "định tuyến thắng trộn" hiện chỉ chắc trong protocol mà LR luôn là native LR2 nửa kích thước. Nếu đánh giá cuối có LR theo cách khác (câu hỏi 2 §3), router size-free mới là phần có giá trị. Điều này đã ghi ở HANDOFF 22/11; không phát hiện mới.

---

## 6. Việc còn mở

- User xác nhận: `Step by step research.md` có phải khung của master prompt không.
- User duyệt (hoặc từ chối) 3 đề xuất ở cuối mục 2.
- Nếu muốn dùng `creative-thinking-for-research`, `nnsight`, `pyvene`: đọc đủ SKILL.md trước; hiện chưa có lý do kỹ thuật để dùng.
