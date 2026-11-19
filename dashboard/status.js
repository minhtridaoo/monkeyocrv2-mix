// Nguồn trạng thái duy nhất của dashboard. Sửa file này (hoặc bấm "Tải status.js" trên dashboard) để cập nhật.
// status: done | doing | todo | waiting (chờ phụ thuộc) | gated (chỉ mở khi đạt điều kiện) | nogo | skipped
window.PIPELINE_STATUS = {
  updated: "2026-11-20 14:00",
  deadline: "2026-11-24",
  start: "2026-11-19",
  hoursBudget: 12,
  youAreHere: "A2",
  phases: [
    { id: "P0", title: "Nền tảng (đã có)" },
    { id: "P1", title: "Chẩn đoán · Bước 3–4" },
    { id: "P2", title: "Mix A-core · trộn trọng số" },
    { id: "P3", title: "Mix B · FSR pilot" },
    { id: "P4", title: "Quyết định & báo cáo" }
  ],
  steps: [
    // Task ID theo kế hoạch chính §2.3 (A1–A4): là mã việc, KHÔNG phải tên checkpoint hay hàng ablation.
    // ---- P0
    { id: "T0", phase: "P0", owner: "tuan", track: "tuan", status: "done", title: "B0 · B-Clean · B1 checkpoints",
      detail: "Job 48833 (B-Clean), B1 12187/12396. LR2 Hmean 61,95 / 74,36 / 75,29 (báo cáo 18/11).", deps: [] },
    { id: "T1", phase: "P0", owner: "tuan", track: "tuan", status: "done", title: "EDA ghép cặp 3 model",
      detail: "Job 48832, 48834: transitions, FP correspondence, 192 ảnh/P-map.", deps: ["T0"] },
    { id: "T2", phase: "P0", owner: "tuan", track: "tuan", status: "done", title: "Pilot PRC · feature-KD · EDSR/bicubic",
      detail: "Không vượt B1. EDSR−bicubic +0,66 pp LR2 (chưa có Clean).", deps: ["T0"] },
    { id: "S0", phase: "P0", owner: "me", track: "shared", status: "done", title: "Kế hoạch chính + harness workspace",
      detail: "Ke_hoach_phoi_hop_MixA_MixB_20261119.md là nguồn sự thật. Tuấn đồng ý triển khai Mix A-core sau khi sửa 3 điểm (tin nhắn do user chuyển lại, chưa có văn bản gốc). Harness: CLAUDE.md, HANDOFF.md, WORKLOG.md, hook SessionStart.", deps: [] },

    // ---- P1
    { id: "T3", phase: "P1", owner: "tuan", track: "tuan", status: "done", title: "Bước 3 · PR/quad audit + recovery hoàn tất",
      detail: "dc19b85: 54308 xong sweep rồi lỗi review; recovery 63985 COMPLETED exit 0 (4:07). Đã tải artifacts/LFS; verify_bundle.py --counts --weights PASS. Baseline toàn cohort: B-Clean H 80,86/75,51; B1 79,43/76,90. Chưa phải kết quả Mix A.", deps: ["T1"],
      output: "docs/handoff/artifacts/pr_quad_audit_v1/", gpu: true },
    { id: "T4", phase: "P1", owner: "tuan", track: "tuan", status: "todo", title: "Bước 4 · FP taxonomy + review tay",
      detail: "160 thẻ đã có, 0 nhãn tay. Review/quần thể 9.697/3.667 thuộc .30/.40. Cần review residual FP ở setting phù hợp (.40/.60 là optimum toàn cohort mới); không dùng nguyên quần thể cũ, tăng mẫu nếu gần mốc 40%.", deps: ["T3"] },
    { id: "S1", phase: "P1", owner: "me", track: "shared", status: "doing", title: "Chốt §1 và môi trường HPC",
      detail: "Nháp tin nhắn đã soạn: docs/msg_tuan_20261120.md (CHƯA gửi). 4 điểm cần Tuấn quyết: mốc tiny 42%, cổng khi khác GPU, điểm tối ưu phẳng/R1, thân R + padding Mix B.", deps: ["S0"], est: 0.5, day: "2026-11-19" },

    // ---- P2: Mix A-core
    { id: "A1", phase: "P2", owner: "me", track: "A", status: "done", title: "A1 · weight_merge.py + test",
      detail: "mixab/weight_merge.py đã code, test local PASS gồm smoke test B1. B-Clean thật nay đã tải và hash PASS; chưa chạy merge B-Clean với B1 thật, chưa inference.", deps: ["S0"], est: 1.5, day: "2026-11-19" },
    { id: "A3", phase: "P2", owner: "me", track: "A", status: "done", title: "A3 · reselect_split.py (A→B, R1) · CPU",
      detail: "mixab/reselect_split.py: đã chạy trên NPZ thật (chỉ baseline) -> results/baseline_ab_v1/ + NOTES.md; chọn toàn cohort khớp từng số bảng §0. Phát hiện: B1 trên A chọn setting 45; B1 trượt mốc tiny 42% trên B (37,55%). Sẽ chạy lại với 5 α khi có counts A2.", deps: ["A1"], est: 1.5, day: "2026-11-20" },
    { id: "A2", phase: "P2", owner: "me", track: "A", status: "doing", title: "A2 · merge_audit.py (probe rồi chạy)",
      detail: "mixab/merge_audit.py đã viết (prepare-lr2, --probe, resume theo chunk, cổng 3a/P-map, profile, reselect_spec). Phần CPU có test, gồm bundle thật; PHẦN GPU CHƯA CHẠY. Hướng dẫn: docs/RUNBOOK_mix_a_T4_A100.md.", deps: ["A1", "S1"], est: 2, day: "2026-11-21", gpu: true },
    { id: "E1", phase: "P2", owner: "me", track: "A", status: "todo", title: "Dựng môi trường T4/A100 + doctor 0 FAIL",
      detail: "Notebook Colab đã sẵn sàng: notebooks/colab_E1_A2_probe.ipynb (upload runtime/colab_upload/mixab_code.zip lên MyDrive/CS431_MixA/upload/). CHƯA chạy trên Colab; E1 chỉ xong khi Ô 2 + doctor (Ô 4) báo 0 FAIL.", deps: ["A2"], est: 1.5, day: "2026-11-21" },
    { id: "A2r", phase: "P2", owner: "me", track: "A", status: "todo", title: "A2 · job GPU 5 α × Clean/LR2 × 72 setting",
      detail: "User chạy trên T4/A100 (không có H200): probe 40 nguồn trước, đọc gates.json/profile.json, rồi mới chạy đầy đủ. Thời gian lấy từ profile trên GPU đó, không suy từ job H200 của Tuấn.", deps: ["A2", "E1"], gpuOnly: true },
    { id: "A4", phase: "P2", owner: "me", track: "A", status: "todo", title: "A4 · đánh giá B, đường α, CI",
      detail: "Chỉ α* đã khóa trên A được xét đi tiếp (+0,5 pp so với cả hai endpoint, tiny <4 px ≥ 42%; đây là ngưỡng thực dụng). Giữ PENDING_REFERENCE_CHECK đến khi 3b PASS.", deps: ["A2r", "A3"], est: 1.5, day: "2026-11-22" },
    { id: "A5", phase: "P2", owner: "me", track: "A", status: "gated", title: "Model Stock (nhánh phụ)",
      detail: "Hoãn tới khi Mix A-core có báo cáo. Sau đó mới xét stats CPU, B0/hash và budget.", deps: ["R1"], est: 0.5, gate: "báo cáo Mix A xong · stats c>0" },

    // ---- P3: Mix B (chuẩn bị song song, train có điều kiện, có thể dời sau 24/11)
    { id: "B1", phase: "P3", owner: "me", track: "B", status: "doing", title: "Chuẩn bị FSR + gate-only (code)",
      detail: "mixab/fsr.py torch thuần đã có + test CPU PASS. G zero-init, init chung, loader có key policy. Thân R/padding vẫn chờ thống nhất; chưa tích hợp MMOCR hay probe GPU.", deps: ["S1"], est: 1.5, day: "2026-11-23" },
    { id: "B2", phase: "P3", owner: "me", track: "B", status: "gated", title: "Probe GPU",
      detail: "Đầu ra trùng B1 ≤1e-6; gradient lớp cuối R ở backward đầu; thân R có gradient sau ≤20 update; G chỉ kiểm ở FSR/gate-only; đo latency/VRAM.", deps: ["B1", "T6"], gpu: true, gate: "residual control của Tuấn sẵn sàng" },
    { id: "B3", phase: "P3", owner: "me", track: "B", status: "gated", title: "Train FSR + gate-only (1.000 update)",
      detail: "Chỉ khi đạt §5.1: taxonomy cho background ≥40%, hậu xử lý/Mix A chưa giải quyết được, residual probe PASS, recipe + tiêu chí đã khóa. Có thể dời sau 24/11.", deps: ["B2", "T4", "A4"], gpu: true, gate: "§5.1 kế hoạch chính" },
    { id: "B4", phase: "P3", owner: "me", track: "B", status: "gated", title: "Đánh giá FSR − gate-only",
      detail: "Cùng R1 và CI; tiêu chí định lượng phải khóa trước khi train.", deps: ["B3"], gate: "B3 xong" },

    // ---- P4
    { id: "T6", phase: "P4", owner: "tuan", track: "tuan", status: "todo", title: "Bước 6 · Head-control / Residual / Residual+KD",
      detail: "Chỉ khi Bước 5 chọn feature pilot. Cần chốt ai code và chạy từng control.", deps: ["T4"] },
    { id: "R1", phase: "P4", owner: "me", track: "shared", status: "todo", title: "Báo cáo Mix A (kể cả kết quả âm)",
      detail: "Đường trade-off, bảng A→B, CI, trạng thái kiểm chứng, hạn chế (1 seed, dev đã dùng để chọn, A/B không phải holdout).", deps: ["A4"], est: 1, day: "2026-11-23" },
    { id: "R2", phase: "P4", owner: "me", track: "shared", status: "todo", title: "Bàn giao cho Tuấn (24/11)",
      detail: "Code/config/test trên branch, lệnh HPC, manifest/hash, bảng/đồ thị/CI, đề xuất bước tiếp.", deps: ["R1"], est: 1, day: "2026-11-24" },
    { id: "T8", phase: "P4", owner: "tuan", track: "tuan", status: "todo", title: "Bước 8 · seed + đánh giá cuối",
      detail: "Sau khi khóa phương pháp. Ngoài quỹ thời gian đến 24/11.", deps: ["R2"] }
  ],
  // Lịch theo kế hoạch chính §6 (dự kiến, không phải cam kết)
  schedule: [
    { day: "2026-11-19", lane: "me",  track: "shared", label: "S1 chốt §1 với Tuấn", h: 0.5 },
    { day: "2026-11-19", lane: "me",  track: "A", label: "A1 weight_merge + test", h: 1.5 },
    { day: "2026-11-20", lane: "me",  track: "A", label: "A3 reselect_split + test", h: 1.5 },
    { day: "2026-11-20", lane: "me",  track: "shared", label: "S1 nhận đường dẫn HPC", h: 0.5 },
    { day: "2026-11-21", lane: "me",  track: "A", label: "A2 merge_audit + probe/submit", h: 2 },
    { day: "2026-11-21", lane: "gpu", track: "A", label: "A2r job merge (nền)", span: 2 },
    { day: "2026-11-22", lane: "me",  track: "A", label: "A4 đánh giá + CI (đệm queue)", h: 2 },
    { day: "2026-11-23", lane: "me",  track: "shared", label: "R1 báo cáo Mix A", h: 1 },
    { day: "2026-11-23", lane: "me",  track: "B", label: "B1 chuẩn bị Mix B (nếu đủ điều kiện)", h: 1 },
    { day: "2026-11-24", lane: "me",  track: "shared", label: "R2 bàn giao + đệm", h: 2 },
    { day: "2026-11-19", lane: "tuan", track: "tuan", label: "T3 recovery review prep", span: 2 },
    { day: "2026-11-21", lane: "tuan", track: "tuan", label: "T4 taxonomy + review", span: 3 }
  ],
  log: [
    { at: "2026-11-20", text: "Soạn notebook Colab E1 + probe A2 (13 cell, nbformat hợp lệ); 50/50 test PASS. Chưa chạy trên Colab: E1/A2 chưa xong." },
    { at: "2026-11-20", text: "Thêm mixab/doctor.py, scripts/run_probe.sh, nháp tin nhắn Tuấn. 43/43 test PASS. Chờ user dựng môi trường GPU." },
    { at: "2026-11-20", text: "GPU chỉ có T4/A100 (ghi vào CLAUDE.md, kế hoạch §4.3). A1 merge thật OK; A3 baseline A→B thật xong; A2 merge_audit đã viết (GPU chưa chạy). 39/39 test PASS." },
    { at: "2026-11-19", text: "Kế hoạch v3 xong sau 3 vòng review; kế hoạch chính đã được sắp xếp lại (Ke_hoach_phoi_hop...)." },
    { at: "2026-11-19", text: "Dựng harness workspace (CLAUDE.md, HANDOFF, WORKLOG, hook SessionStart). A1 weight_merge.py: test local 10/10 PASS, chưa commit." },
    { at: "2026-11-19", text: "Chuyển code sang package riêng mixab/ (MONKEY_V2 chỉ để tham chiếu). A3 reselect_split + Mix B fsr.py: tổng 29/29 test PASS. Chặn: chờ Tuấn gửi đường dẫn B-Clean + NPZ 54308." },
    { at: "2026-11-19", text: "Codex fetch + ff dc19b85, tải LFS B-Clean/B1/review ZIP. Verifier --counts --weights PASS; tính lại optimum duy nhất 44/70/70. 0/160 nhãn review. Chạy lại 29/29 tests PASS. Bỏ blocker thiếu NPZ/B-Clean; tiếp tục A3 trên NPZ thật và viết A2. Chưa chạy Mix A/B hay submit HPC." }
  ]
};
