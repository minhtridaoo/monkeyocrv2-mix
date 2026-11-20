// Nguồn trạng thái duy nhất của dashboard. Sửa file này (hoặc bấm "Tải status.js" trên dashboard) để cập nhật.
// status: done | doing | todo | waiting (chờ phụ thuộc) | gated (chỉ mở khi đạt điều kiện) | nogo | skipped
window.PIPELINE_STATUS = {
  updated: "2026-11-20 12:00",
  deadline: "2026-11-24",
  start: "2026-11-19",
  hoursBudget: 12,
  youAreHere: "AY",
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
    { id: "S1", phase: "P1", owner: "me", track: "shared", status: "doing", title: "Các quyết định còn chờ của nhóm",
      detail: "Đã chốt 20/11 (user chuyển lời Tuấn): phương án (a) cùng A100; R1 + tie-break; +0,5 pp so với cả hai baseline; tiny ≥ B1 − 3 pp trên cùng split B. Còn mở: hướng sau khi Mix A chưa thắng; Mix B/thân R bàn riêng, không chặn Mix A.", deps: ["S0"], est: 0.5, day: "2026-11-19" },

    // ---- P2: Mix A-core
    { id: "A1", phase: "P2", owner: "me", track: "A", status: "done", title: "A1 · weight_merge.py + test",
      detail: "Đã tạo thật 5 checkpoint trộn B-Clean/B1; có ở runtime/ và Drive weights/merge_weights_v1/. Doctor A100 kiểm SHA PASS; hai endpoint đã chạy probe.", deps: ["S0"], est: 1.5, day: "2026-11-19" },
    { id: "A3", phase: "P2", owner: "me", track: "A", status: "done", title: "Chọn lại A→B với 5 α · xong",
      detail: "reselect_split chạy trên counts A2r với quy tắc Tuấn chốt 20/11 (R1, +0,5 pp, tiny ≥ B1 − 3 pp) -> results/mix_a_v1/.", deps: ["A1"], est: 1.5, day: "2026-11-20" },
    { id: "A2", phase: "P2", owner: "me", track: "A", status: "done", title: "Probe 40 nguồn A100 · đã chạy; có lệch H200",
      detail: "Probe 40 nguồn chạy xong trên A100: P-map PASS (0,0); 3a B1 PASS, B-Clean FAIL 1/80 (tp lệch 1). So với H200 trên 72 setting: 1,6–1,8% ô lệch, ΔHmean ≤ 0,26 pp. 0,335 s/task, 1,15 GiB. Chi tiết: results/probe_A100_v1/NOTES.md.", deps: ["A1", "E1"], est: 2, day: "2026-11-21", gpu: true },
    { id: "E1", phase: "P2", owner: "me", track: "A", status: "done", title: "Môi trường A100 · doctor 0 FAIL",
      detail: "Xong trên Colab A100-SXM4-40GB: doctor 22/22 PASS, 0 FAIL (results/probe_A100_v1/). Python 3.13, torch 2.9.0+cu128, mmcv 2.0.1 build từ source. T4 chưa thử.", deps: ["A1"], est: 1.5, day: "2026-11-21" },
    { id: "A2r", phase: "P2", owner: "me", track: "A", status: "done", title: "A2r · đủ 5 model trên A100 · VERIFIED",
      detail: "Xong trên Colab A100-SXM4-40GB (20/11): 5 model × 2.262 nguồn, khoảng 2,7 giờ wall. P-map PASS, gate_same_gpu PASS (0/5.760 ô lệch), verification VERIFIED (same_gpu_endpoints). Độ lệch so với H200 chỉ báo cáo: 3,1–3,4% ô, |ΔHmean| toàn cohort ≤ 0,023 pp.", deps: ["A2", "E1"], gpuOnly: true },
    { id: "A4", phase: "P2", owner: "me", track: "A", status: "done", title: "Đánh giá B · α* = B-Clean · Mix A chưa thắng",
      detail: "α* trên A = B-Clean (endpoint): Mix A chưa thắng theo quy tắc đã khóa. Trên B (chỉ mô tả): các α giữa cao hơn endpoint tốt nhất 0,17–0,23 pp, không α nào +0,5 pp, CI chứa 0; α 3/4 so với B1: Clean +0,47 pp, LR2 FP −11%, tiny −3,07 pp. Chi tiết results/mix_a_v1/NOTES.md.", deps: ["A2r", "A3"], est: 1.5, day: "2026-11-22" },
    { id: "A5", phase: "P2", owner: "me", track: "A", status: "gated", title: "Model Stock (nhánh phụ)",
      detail: "Hoãn tới khi Mix A-core có báo cáo. Sau đó mới xét stats CPU, B0/hash và budget.", deps: ["R1"], est: 0.5, gate: "báo cáo Mix A xong · stats c>0" },
    { id: "AX", phase: "P2", owner: "me", track: "A", status: "done", title: "E-sel · chọn có ràng buộc tiny (khám phá)",
      detail: "Khám phá, không thay kết luận v1. α* = 3/4 nhưng vẫn trượt +0,5 pp và tiny (−3,07 pp); ở cùng recall, α 3/4 nhiều FP hơn B1 (+6–7%). results/explore_sel_v1/NOTES.md.", deps: ["A4"], est: 1, day: "2026-11-20" },
    { id: "AY", phase: "P2", owner: "me", track: "A", status: "waiting", title: "E-BN · pilot tính lại BN (thiết kế xong)",
      detail: "docs/design_bn_recalibration_pilot.md. Chỉ 2 lớp BN binarize ảnh hưởng inference. Chờ: manifest TRAIN từ Tuấn + nhóm duyệt cổng/ngưỡng/vòng xác nhận. Chưa code GPU, chưa chạy.", deps: ["AX"], est: 1.5, gpu: true },

    // ---- P3: Mix B (chuẩn bị song song, train có điều kiện, có thể dời sau 24/11)
    { id: "B1", phase: "P3", owner: "me", track: "B", status: "doing", title: "Chuẩn bị FSR + gate-only (code)",
      detail: "mixab/fsr.py torch thuần đã có + test CPU PASS. G zero-init, init chung, loader có key policy. R là mạng residual đề xuất sau FPNC, không phải phần Tuấn đã nhận làm. Nhóm chưa chốt hướng Mix B, kiến trúc/padding và người triển khai; chưa tích hợp MMOCR hay probe GPU.", deps: ["S1"], est: 1.5, day: "2026-11-23" },
    { id: "B2", phase: "P3", owner: "me", track: "B", status: "gated", title: "Probe GPU",
      detail: "Đầu ra trùng B1 ≤1e-6; gradient lớp cuối R ở backward đầu; thân R có gradient sau ≤20 update; G chỉ kiểm ở FSR/gate-only; đo latency/VRAM.", deps: ["B1", "T6"], gpu: true, gate: "nhóm chốt thiết kế; residual control sẵn sàng" },
    { id: "B3", phase: "P3", owner: "me", track: "B", status: "gated", title: "Train FSR + gate-only (1.000 update)",
      detail: "Chỉ khi đạt §5.1: taxonomy cho background ≥40%, hậu xử lý/Mix A chưa giải quyết được, residual probe PASS, recipe + tiêu chí đã khóa. Có thể dời sau 24/11.", deps: ["B2", "T4", "A4"], gpu: true, gate: "§5.1 kế hoạch chính" },
    { id: "B4", phase: "P3", owner: "me", track: "B", status: "gated", title: "Đánh giá FSR − gate-only",
      detail: "Cùng R1 và CI; tiêu chí định lượng phải khóa trước khi train.", deps: ["B3"], gate: "B3 xong" },

    // ---- P4
    { id: "T6", phase: "P4", owner: "me", track: "shared", status: "waiting", title: "Thống nhất có làm residual R và phân công",
      detail: "Điểm phối hợp của nhóm, không phải việc đã giao Tuấn. R và các nhánh Head-control/Residual/Residual+KD mới là đề xuất có điều kiện trong workflow. Cần xác nhận có chọn hướng feature, kiến trúc và người code/chạy trước khi tích hợp Mix B.", deps: ["T4"] },
    { id: "R1", phase: "P4", owner: "me", track: "shared", status: "doing", title: "Báo cáo Mix A (kể cả kết quả âm)",
      detail: "NOTES.md đã có bảng, CI, kiểm chứng và giới hạn. Còn: hình đường α, bản báo cáo gửi Tuấn (docs/msg_tuan_20261120_mixA_result.md, nháp, chưa gửi).", deps: ["A4"], est: 1, day: "2026-11-23" },
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
    { day: "2026-11-20", lane: "me",  track: "shared", label: "Đã thống nhất phương án chạy A100", h: 0.5 },
    { day: "2026-11-21", lane: "me",  track: "A", label: "Chuẩn bị phiên Colab A100 cho Ô 7–8", h: 2 },
    { day: "2026-11-21", lane: "gpu", track: "A", label: "Bạn chạy A2r trên A100 (lịch dự kiến)", span: 2 },
    { day: "2026-11-22", lane: "me",  track: "A", label: "A4 đánh giá + CI (đệm queue)", h: 2 },
    { day: "2026-11-23", lane: "me",  track: "shared", label: "R1 báo cáo Mix A", h: 1 },
    { day: "2026-11-23", lane: "me",  track: "B", label: "B1 chuẩn bị Mix B (nếu đủ điều kiện)", h: 1 },
    { day: "2026-11-24", lane: "me",  track: "shared", label: "R2 bàn giao + đệm", h: 2 },
    { day: "2026-11-19", lane: "tuan", track: "tuan", label: "T3 recovery review prep", span: 2 },
    { day: "2026-11-21", lane: "tuan", track: "tuan", label: "T4 taxonomy + review", span: 3 }
  ],
  log: [
    { at: "2026-11-20", text: "Rà soát để đồng bộ Git: Mix A-core v1 đã VERIFIED theo cùng GPU nhưng không thắng; E-sel chỉ là khám phá. 60/60 test PASS. Repo tham chiếu Tuấn đã fast-forward tới aa1e17f; run v1 vẫn ghim dc19b85. Output lớn không nằm trong Git." },
    { at: "2026-11-20", text: "E-sel (khám phá): chọn có ràng buộc tiny không cho ứng viên đạt; FP ở cùng recall không tốt hơn B1. Thiết kế pilot BN xong, chờ manifest TRAIN + duyệt. 60/60 test PASS." },
    { at: "2026-11-20", text: "A2r xong trên A100 (VERIFIED, same-GPU). Tuấn chốt R1, +0,5 pp, tiny ≥ B1 − 3 pp. α* = B-Clean: Mix A chưa thắng theo quy tắc đã khóa. 55/55 test PASS." },
    { at: "2026-11-20", text: "Codex đồng bộ dashboard theo user và HANDOFF/notebook: E1 + probe đã xong; A2r chờ user chạy Ô 7–8, không chờ Tuấn chọn a/b. Sửa A1 còn mô tả cũ, phụ thuộc E1 trước probe, tiêu chí A4 cùng A100. Mix B/R chưa được nhận triển khai bởi Tuấn. Không chạy job GPU trong lần cập nhật này." },
    { at: "2026-11-20", text: "Tuấn chọn (a). Thêm chính sách cổng cùng GPU, Ô 7 (A2r) và Ô 8 (reselect) vào notebook; 54/54 test PASS. Chưa chạy trên Colab." },
    { at: "2026-11-20", text: "Colab A100: E1 xong (doctor 0 FAIL). Probe A2 xong: P-map PASS, 3a B-Clean FAIL 1/80, ΔHmean so với H200 ≤ 0,26 pp. Chờ Tuấn chọn (a)/(b) trước A2r." },
    { at: "2026-11-20", text: "Soạn notebook Colab E1 + probe A2 (13 cell, nbformat hợp lệ); 50/50 test PASS. Chưa chạy trên Colab: E1/A2 chưa xong." },
    { at: "2026-11-20", text: "Thêm mixab/doctor.py, scripts/run_probe.sh, nháp tin nhắn Tuấn. 43/43 test PASS. Chờ user dựng môi trường GPU." },
    { at: "2026-11-20", text: "GPU chỉ có T4/A100 (ghi vào CLAUDE.md, kế hoạch §4.3). A1 merge thật OK; A3 baseline A→B thật xong; A2 merge_audit đã viết (GPU chưa chạy). 39/39 test PASS." },
    { at: "2026-11-19", text: "Kế hoạch v3 xong sau 3 vòng review; kế hoạch chính đã được sắp xếp lại (Ke_hoach_phoi_hop...)." },
    { at: "2026-11-19", text: "Dựng harness workspace (CLAUDE.md, HANDOFF, WORKLOG, hook SessionStart). A1 weight_merge.py: test local 10/10 PASS, chưa commit." },
    { at: "2026-11-19", text: "Chuyển code sang package riêng mixab/ (MONKEY_V2 chỉ để tham chiếu). A3 reselect_split + Mix B fsr.py: tổng 29/29 test PASS. Chặn: chờ Tuấn gửi đường dẫn B-Clean + NPZ 54308." },
    { at: "2026-11-19", text: "Codex fetch + ff dc19b85, tải LFS B-Clean/B1/review ZIP. Verifier --counts --weights PASS; tính lại optimum duy nhất 44/70/70. 0/160 nhãn review. Chạy lại 29/29 tests PASS. Bỏ blocker thiếu NPZ/B-Clean; tiếp tục A3 trên NPZ thật và viết A2. Chưa chạy Mix A/B hay submit HPC." }
  ]
};
