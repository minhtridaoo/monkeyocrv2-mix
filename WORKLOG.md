# WORKLOG — nhật ký làm việc (chỉ append, mới nhất ở cuối)

Mỗi mục ghi theo mẫu dưới đây. Cột "Bằng chứng" phải trỏ được tới file, commit, lệnh test, job ID hoặc tin nhắn. Không có bằng chứng thì ghi "chưa kiểm".

```
## YYYY-MM-DD · <tiêu đề ngắn>
- Làm: ...
- Bằng chứng: ...
- Chưa kiểm / rủi ro: ...
- Tiếp theo: ...
```

---

## 2026-11-19 · Khảo sát, lập kế hoạch Mix A/B (trước khi có harness)

- **Làm:**
  - Đọc báo cáo 18/11, repo MONKEY_V2 và workflow của Tuấn (`docs/research/next-workflow-20261119.md`, `pr-quad-audit-v1.md`).
  - Khảo sát paper và chốt 2 mix.
  - Qua 3 vòng review với bạn nhóm/Tuấn: v1 → v2 → v3, sau đó sửa thêm 3 điểm (hash file, `PENDING_REFERENCE_CHECK`, nhãn bin).
  - Có người sắp xếp lại thư mục: kế hoạch chính là `Ke_hoach_phoi_hop_MixA_MixB_20261119.md`, các bản cũ chuyển vào `_archive/`.
  - Dựng dashboard `dashboard/index.html` + `status.js`, đã kiểm trong trình duyệt và sửa lỗi lệch ngày do UTC+7.
- **Bằng chứng:**
  - File trong thư mục này.
  - Các dòng code đã đối chiếu: `decision_core.py:110–118` (tie-break `.30/.30`), `decision_audit.py:182–186`.
  - Model Stock Eq. 2–3 và phụ lục H.4 (arXiv 2403.19522v2, bản HTML).
- **Chưa kiểm / rủi ro:**
  - Chưa có NPZ hay artifact HPC nào ở local.
  - Trích dẫn từ trí nhớ trong `_archive/Method_Transfer_Plan…` chưa qua citation check.
  - Bài học: phải tách rõ "đã đọc/đã chạy" với "suy đoán". Ví dụ điển hình: những nội dung tự biên ra mà không có nguồn.
- **Tiếp theo:** dựng harness rồi code A1.

## 2026-11-19 · Dựng harness workspace + bắt đầu A1

- **Làm:**
  - Tạo `HANDOFF.md`, `WORKLOG.md` (file này) và `Đồ án/CLAUDE.md`.
  - Tạo hook SessionStart (`Đồ án/.claude/settings.json` → `.claude/harness/session_start.py`).
  - Đồng bộ dashboard theo kế hoạch chính.
  - Code A1 trên branch `codex/mix-a-core`.
- **Bằng chứng:**
  - **Hook:**
    - Pipe-test `echo '{}' | python -I .claude/harness/session_start.py`: exit 0, ra JSON hợp lệ (đã parse lại).
    - Chạy lại qua `$CLAUDE_PROJECT_DIR` từ cwd `/`: exit 0.
    - Hai lỗi được phát hiện và sửa: (1) stdout cp1252 trên Windows, sửa bằng `ensure_ascii=True`; (2) regex không khớp tiêu đề có hậu tố.
    - Hook chỉ có hiệu lực ở **session mới** mở tại `Đồ án/`.
  - **Repo:**
    - `git fetch` thấy commit mới của Tuấn `37384cf`: job 54308 chạy xong 3 sweep, tái lập base counts, lỗi ở bước chuẩn bị review, có job recovery.
    - Branch `codex/mix-a-core` tạo từ `origin/main` @ `37384cf`, đã `--unset-upstream`.
  - **A1:**
    - `python -m unittest discover -s tests -p test_weight_merge.py -v`: **10/10 OK**. `RealB1SmokeTest` có chạy, không bị skip; SHA-256 B1 khớp `docs/handoff/checkpoints.md`.
    - CLI trên B1 thật (`--bclean B1 --b1 B1 --alphas 0 0.5 1`): ghi 3 file, reload bằng nhau, chạy lần 2 thì báo `Refusing to overwrite`.
    - Thông tin B1: 377 tensor, 373 floating, 4 BN (đều ở det_head), 4 `num_batches_tracked`.
- **Chưa kiểm / rủi ro:**
  - Chưa merge B-Clean thật, vì local không có file.
  - Chưa kiểm rằng mmengine `load_checkpoint` trên HPC đọc được state_dict phẳng do script lưu ra. Theo hiểu biết thì đọc được, và B1 export có cùng định dạng, nhưng chưa thử.
  - Chưa commit.
  - Lệnh test trong HANDOFF bản đầu ghi sai (`unittest tests.x`); đã sửa sang `discover`.
- **Tiếp theo:** user duyệt diff, commit local; hỏi Tuấn đường dẫn B-Clean và NPZ; viết A3 `reselect_split.py` + test tổng hợp.

## 2026-11-19 (khuya) · Đổi chỗ đặt code + A3 + chuẩn bị Mix B

- **User dặn:**
  - MONKEY_V2 chỉ để tham chiếu pipeline và phải kiểm thường xuyên.
  - Việc của mình nằm ở `cải tiến/mới`, sau này lên repo riêng (user sẽ báo khi nào up).
  - Thư mục của Tuấn biến mất là bình thường.
  - Mix B được làm song song khi Mix A chờ.
- **Làm:**
  - **Chuyển A1** từ `MONKEY_V2/rev4/` sang package `mixab/`, tự chứa: `require`, `sha256`, `canonical`, `save_json` cùng định dạng byte với `lowres.common`. Test đi theo sang `tests/`.
  - **Trả repo Tuấn về sạch:** xóa 2 file của mình, `checkout main`, `merge --ff-only origin/main`, xóa branch `codex/mix-a-core`. Branch có 0 commit nên không mất gì.
  - Thêm `mixab/monkey.py`: tìm MONKEY_V2 (biến `MONKEY_V2_ROOT` hoặc thư mục bên cạnh) và ghi commit/dirty.
  - Thêm `.gitignore` để không đưa `*.pth`, `*.npz` và output vào repo.
  - **A3 `mixab/reselect_split.py`:**
    - Chia A/B theo SHA-256 của UTF-8 ID.
    - R1: chọn trên A bằng `Fraction` chính xác, tie-break native `.30`, rồi theo thứ tự lưới.
    - α\*: ưu tiên endpoint, rồi α lớn hơn; khóa α\*, đánh giá B.
    - Tiêu chí đi tiếp tính bằng `Fraction` (+0,5 pp, tiny ≥ 42%).
    - Bootstrap ghép cặp trên B.
    - Cổng 3b ghép theo `(source_id, condition, setting_key)`.
    - Trạng thái `VERIFIED` / `PENDING_REFERENCE_CHECK` / `INVALIDATED`.
    - Đọc được cả 5 cột (NPZ Tuấn) lẫn 11 cột (merge).
  - **Mix B chuẩn bị `mixab/fsr.py`:** residual / gate-only / FSR. G zero-init, lớp cuối R zero weight+bias, init dùng chung, residual chỉ được bỏ qua `gate.*`.
  - Cập nhật kế hoạch chính (§1, §4.1, §6, ghi chú làm song song ở §5), `CLAUDE.md` và memory.
- **Bằng chứng:**
  - `python -m unittest discover -s tests -t . -v`: **29/29 OK**, không skip. Gồm 10 test A1 (smoke test B1 thật có chạy), 12 test A3, 7 test FSR.
  - Có một test A3 ban đầu đỗ nhầm: setting native nằm đầu lưới nên không phân biệt được với thứ tự lưới. Đã sửa, và kiểm bằng mutation (bỏ quy tắc native thì code chọn `unclip_1p2`, test FAIL).
  - Một test FSR FAIL lần đầu do so thứ tự key, đã sửa test; logic không đổi.
  - `git log origin/main..` = 0 commit trước khi xóa branch. `git status` repo Tuấn: `main...origin/main`, chỉ `scripts/` untracked.
- **Chưa kiểm / rủi ro:**
  - A3 chưa chạy trên NPZ thật. Giả định layout lấy từ việc đọc code, chưa mở file thật.
  - Thân R của FSR (Conv3×3 → BN → ReLU → Conv1×1) và padding AvgPool (`count_include_pad=False`) là đề xuất của mình, chưa thống nhất.
  - A2 `merge_audit.py` chưa viết.
- **Tiếp theo:** hỏi Tuấn (B-Clean path/SHA, NPZ 54308, thân R); viết A2 sau khi đọc `rev4/decision_audit.py:150–200`.

## 2026-11-19 · Codex cập nhật tiến độ từ repo Tuấn trước bàn giao Claude

- **Làm:** fetch origin và fast-forward MONKEY_V2 `37384cf → dc19b85` (Publish updated baselines and verified Mix A reference artifacts). Không sửa code repo Tuấn, không đụng `scripts/` untracked. Tải LFS có chọn lọc B-Clean/B1 và ZIP review. Cập nhật §0 và trạng thái kế hoạch chính, HANDOFF, README và dashboard; giữ pipeline/recipe R1 và lịch sử số cũ.
- **Bằng chứng:**
  - Gói: `MONKEY_V2/docs/handoff/artifacts/pr_quad_audit_v1/`.
  - `python docs/handoff/artifacts/pr_quad_audit_v1/verify_bundle.py --counts --weights`: **PASS: completed-job report, bundle hashes, source counts and metrics, LFS weights. Human review remains PENDING.**
  - `provenance/job_54308_sacct.txt`: FAILED, 01:06:37, exit 1:0. `job_63985_sacct.txt`: COMPLETED, 00:04:07, exit 0:0.
  - Tự tính lại tổng counts và exact Fraction Clean+LR2 Hmean từ NPZ: optimum duy nhất B0 index 44; B-Clean/B1 index 70. B-Clean Clean/LR2 H=80.85723091854601/75.50704318400363%, FP LR2=6578, tiny recall=15.95579682047305%. B1 H=79.43148366682661/76.90103562608304%, FP LR2=11997, tiny recall=45.792943001163245%. Đây là toàn cohort, chưa phải A→B.
  - `review/human_review.csv`: 160 dòng, **0** human_label không rỗng. Code `prepare_review` dùng common và render `.30/.40`; không đại diện mặc định cho setting mới.
  - `python -m unittest discover -s tests -t . -v` trong thư mục này: **29/29 OK, không skip**, gồm RealB1SmokeTest; không viết hay thay code thuật toán.
  - B-Clean tải về là plain state_dict, hash `94f0e4ecbb6881b4430c841845994c6568c145af7bb54fc41514eb0c78f3a6aa`; B1 hash `9c3814e494a73a0a6089ba5a6585f079ea8f5a6defdc9347213bb98a64337147`, đã qua verifier. Không cần `--allow-pickle` cho hai bản xuất.
- **Chưa kiểm / rủi ro:**
  - Chưa chạy merge B-Clean+B1 thật, chưa chạy module reselect trên NPZ thật, chưa viết merge_audit; chưa inference/probe/train GPU và không submit HPC.
  - Gói baseline PASS không làm thí nghiệm Mix A trở thành VERIFIED. Cần endpoint/P-map/evaluator và các cổng riêng của thí nghiệm.
  - Còn map root ảnh/encoder/môi trường HPC; chưa hoàn tất review, thân R/padding/scheduler và tiêu chí B chưa khóa.
  - Baseline toàn cohort đã chọn không được dùng thay baseline chọn trên A, đo trên B. Taxonomy ở operating point mới cần quần thể/review phù hợp; một mẫu/tầng không bảo đảm chính xác gần 40%.
- **Tiếp theo:** Claude dùng artifacts đã có để kiểm A3/baseline A→B; kiểm A1 với hai weights thật; viết A2 và chuẩn bị lệnh/probe HPC cho user. Model Stock hoãn đến sau báo cáo Mix A. Không hỏi lại NPZ/B-Clean như blocker.

## 2026-11-19 · Giải thích baseline và split A/B trên pipeline

- **Làm:** thêm bảng phân biệt B1 / split A / split B / Mix B và sơ đồ checkpoint → counts → chọn A → đánh giá B vào mục 2 kế hoạch chính. Không đổi protocol hay trạng thái job.
- **Bằng chứng:** đếm SHA-256 parity từ source_ids của B1_source_counts.npz: 2.262 nguồn = 1.130 A + 1.132 B; mỗi nguồn giữ cặp Clean/LR2 cùng split.
- **Chưa kiểm/rủi ro:** chỉ đếm split, chưa chạy module reselect hay inference Mix A. B là development đã từng được xem, không phải final TEST.
- **Tiếp theo:** Claude làm theo HANDOFF; giải thích được vì sao phải chọn lại baseline trên A và chấm trên B trước khi đọc số so sánh.

## 2026-11-20 (rạng sáng) · Ràng buộc GPU T4/A100, baseline A→B thật, A1 thật, viết A2

- **User dặn:**
  - Chỉ có T4 hoặc A100, không có H200.
  - Cập nhật tiến độ hợp lý và quản lý workspace gọn gàng.
- **Làm:**
  - Ghi ràng buộc GPU vào `CLAUDE.md`, kế hoạch §4.3 (có cách xử lý đề xuất khi cổng lệch do khác GPU, chưa thống nhất với Tuấn), §6 và memory.
  - Tự kiểm lại bàn giao của Codex: `git fetch` (vẫn `dc19b85`), `verify_bundle.py --counts --weights` PASS, `sha256sum` hai export khớp.
  - **A3 trên NPZ thật** (`configs/baseline_ab_v1.json` → `results/baseline_ab_v1/`).
    - Thêm chế độ `baseline_only`: khi chỉ có endpoint thì không áp tiêu chí đi tiếp, tránh câu "merge did not win" gây hiểu nhầm.
    - Ghi chú phân tích ở `results/baseline_ab_v1/NOTES.md`.
  - **A1 thật:** merge B-Clean×B1 export ra 5 α trong `runtime/merge_weights_v1/` (8 giây, 423 MB).
  - **A2 `mixab/merge_audit.py`:** gọi lại `detector_for/forward/capture_candidates/CandidateMatcher/native_lr2/GEOMETRIES/SCORES` của MONKEY_V2.
    - `prepare-lr2` có kiểm SHA.
    - `--probe N` lấy tập con cố định.
    - Resume theo chunk; bản ghi theo từng model để không mất cổng khi resume.
    - Cổng 3a dùng `eda_common_counts.csv.gz`; kiểm P-map endpoint (bản sao so với export, cùng GPU).
    - Ghi profile s/task và VRAM; tự sinh `reselect_spec.json`.
  - Viết `docs/RUNBOOK_mix_a_T4_A100.md`.
  - Docstring các CLI chuyển sang ASCII vì `--help` crash trên cp1252.
- **Bằng chứng:**
  - Chọn toàn cohort bằng `choose_setting` cho index 44/70/70 và **trùng từng số** với bảng §0.
  - Baseline A→B: A 1.130 / B 1.132. B-Clean chọn 70, B1 chọn 45, B0 chọn 45.
    - Trên B: B1 mean 78,11%, LR2 76,70%, FP 5.454, tiny 37,55%.
    - B1 ở setting 70 trên B: tiny 39,38%. Cùng setting 70, trên A tiny là 52,19%; tiny GT A/B = 2.583/2.575; top-5 nguồn chiếm 53,3% tiny GT của A.
  - Manifest merge: endpoint `equal: True`, đọc lại `equal: True` cho cả 5; `non_floating_differ` = 0.
  - `python -m unittest discover -s tests -t .`: **39/39 OK**. Test trên bundle thật có chạy:
    - Cổng 3a PASS 4.524/4.524 trên counts của Tuấn; sửa 1 số đếm thì FAIL, đúng 1 mismatch.
    - `build_settings(GEOMETRIES, SCORES)` trùng đúng 72 setting của NPZ.
  - `py_compile` OK; `--help` của 3 CLI OK.
- **Chưa kiểm / rủi ro:**
  - **Phần GPU của `merge_audit` chưa chạy lần nào** (không có MMOCR/CUDA ở local); có thể còn lỗi chỉ lộ ra trên máy thật.
  - Chưa biết số đếm trên T4/A100 có trùng NPZ H200 không.
  - Mốc tiny 42% không ổn định và cần Tuấn quyết.
  - Môi trường (torch 2.9 cu128 + mmcv 2.0.1 trên T4) chưa kiểm.
  - Mix B vẫn chờ chốt thân R.
- **Tiếp theo:** user chạy runbook bước 1–3 (probe 40) rồi gửi `gates.json` + `profile.json`; báo Tuấn về mốc tiny và cách xử lý khi khác GPU.

## 2026-11-20 · Doctor môi trường + script probe + nháp tin nhắn Tuấn

- **Kiểm đầu phiên:**
  - `git fetch`: MONKEY_V2 vẫn `dc19b85`, không có commit mới.
  - HANDOFF/WORKLOG/kế hoạch không bị ai sửa kể từ phiên trước.
- **Làm:**
  - `mixab/doctor.py` kiểm, mỗi mục báo PASS/WARN/FAIL:
    - Phiên bản package so với `checkpoints.md`.
    - CUDA: đúng 1 device, arch có trong wheel, chạy thử kernel; báo thiếu TF32 nếu GPU < Ampere (T4).
    - Pillow 11.3.0.
    - SHA của DBPostprocessor; `MonkeyOCRv2ViTAEBackbone` đã đăng ký chưa (config không có `custom_imports`).
    - Đường dẫn encoder trong config, FP32.
    - GT của bundle; SHA của mẫu ảnh Clean/LR2; weights merge; dung lượng đĩa.
  - `scripts/run_probe.sh`: test → merge → doctor → `prepare-lr2` → probe 40. Doctor chạy trước `prepare-lr2` để bắt lỗi môi trường sớm.
  - Runbook có thêm mục "Đường tắt".
  - `docs/msg_tuan_20261120.md`: 4 điểm cần Tuấn quyết, có số liệu.
  - `sys.stdout.reconfigure(errors='backslashreplace')` trong doctor/merge_audit, vì doctor crash khi in đường dẫn tiếng Việt trên cp1252.
- **Bằng chứng:**
  - Chạy doctor ở máy local: 6 PASS, 3 WARN, 9 FAIL, exit 1.
    - Các FAIL đúng là thiếu thật: mmcv/mmengine/mmocr, CUDA, Pillow 12.2.0 ≠ 11.3.0, encoder `pretrained/...`, ảnh Clean.
    - Các PASS đúng là đã có: bundle GT 2.262 nguồn, 5 checkpoint merge SHA ok, đĩa 216,8 GiB.
  - `python -m unittest discover -s tests -t .`: **43/43 OK** (thêm 4 test doctor).
  - `bash -n scripts/run_probe.sh` OK.
  - Số trong tin nhắn đối chiếu với `results/baseline_ab_v1/NOTES.md` (52,19/39,38; 2.583/2.575; 53,3%; 37,55%; 0,23 pp).
- **Chưa kiểm / rủi ro:**
  - Doctor và script chưa chạy trên máy GPU thật.
  - Các nhánh kiểm CUDA/MMOCR chỉ mới được thử ở trạng thái "thiếu".
  - Tin nhắn **chưa gửi**.
- **Tiếp theo:** user dựng môi trường, chạy doctor cho tới 0 FAIL, rồi chạy `run_probe.sh`; gửi tin nhắn cho Tuấn.

## 2026-11-20 · Notebook Colab cho E1 + probe A2 (chưa chạy)

- **User yêu cầu:** notebook Colab hoàn chỉnh để user tự chạy E1 (môi trường T4/A100 + doctor 0 FAIL) và probe A2. Không chạy A2r đầy đủ, không làm Mix B, không gửi tin nhắn cho Tuấn.
- **Kiểm trước khi viết (bằng chứng):**
  - `git fetch`: MONKEY_V2 vẫn `dc19b85`.
  - `sweep_run_identity.json`: DBPostprocessor của Tuấn nằm trong `upstream_MonkeyOCRv2/detection/mmocr/mmocr/`.
  - Đọc `MonkeyOCRv2/detection/mmocr/README.md` + `install.sh` @ `da96afed…`: công thức cài; MMCV 2.0.1 phải build từ source; patch không đụng DBPostprocessor.
  - Hash file `db_postprocessor.py` của MMOCR v1.0.1 gốc: `9bfff8aa…`, trùng bản ghim.
  - `curl -I` TextVQA zip: 7.072.297.970 byte. Đọc mục lục từ xa bằng HTTP Range: `train_images/<id>.jpg`.
  - API Hugging Face của encoder có file `model.safetensors`; hai repo GitHub đều 200.
  - Config Tuấn không có `custom_imports`, nên backbone phải được đăng ký qua add-on.
- **Làm:**
  - `mixab/colab_helpers.py` (chỉ dùng thư viện chuẩn):
    - Đặt tên có timestamp để không ghi đè; `move_aside` thay vì xóa.
    - `run` có stream log và báo lỗi tiếng Việt kèm 30 dòng cuối.
    - Vá config chỉ đổi 2 đường dẫn encoder (có assert).
    - Chọn 2.262 ảnh từ `source_cohort.csv`, trích đúng các ảnh này từ zip, kiểm SHA, tạo zip cache theo kiểu atomic.
    - `summarize_probe` kèm ước lượng thời gian chạy đầy đủ.
  - `tests/test_colab_helpers.py`: 7 test.
  - `scripts/build_colab_notebook.py` sinh `notebooks/colab_E1_A2_probe.ipynb` (13 cell, 6 cell code, id cố định).
  - `scripts/make_colab_upload.py` sinh `runtime/colab_upload/mixab_code.zip` (21 file, có `CODE_MANIFEST.json` với SHA từng file).
  - **Ô 1:** cấu hình, commit ghim, mount Drive, giải nén code, GPU.
  - **Ô 2:** cài torch, mmengine, MMCV (wheel cache trên Drive, arch 7.5;8.0), `install.sh` (đặt PATH trỏ đúng interpreter), ghim Pillow/OpenCV; kiểm cuối trong tiến trình mới, report env.
  - **Ô 3:** MONKEY_V2 + LFS + `verify_bundle`, test CPU, encoder + config, ảnh Clean (cache trên Drive), 5 checkpoint (dùng lại nếu SHA khớp), LR2.
  - **Ô 4:** doctor; chỉ ghi "giấy thông hành" khi 0 FAIL; không có cách bỏ qua.
  - **Ô 5:** probe 40, chunk 16, resume được, thư mục tách theo GPU; bỏ qua nếu đã có `gates.json`.
  - **Ô 6:** tổng hợp và đóng zip export có timestamp.
- **Bằng chứng:**
  - `nbformat.validate` PASS khi coi warning là lỗi (lần đầu báo thiếu `id` cell; đã sửa).
  - 6 cell code compile, không dùng shell magic.
  - `python -m unittest discover -s tests -t .`: **50/50 OK**.
  - Giải nén `mixab_code.zip` vào thư mục tạm rồi chạy test với `MONKEY_V2_ROOT`: **50/50 OK**.
- **Sửa trong lúc làm:**
  - `run()` để rò pipe (`ResourceWarning`).
  - OpenCV dùng `--force-reinstall` sau khi gỡ các biến thể khác.
  - `install.sh` cần PATH trỏ đúng interpreter.
  - Hai script in lỗi trên console cp1252.
- **Chưa kiểm / rủi ro (GPU, Colab):**
  - Chưa cell nào chạy trên Colab.
  - Build MMCV 2.0.1 với torch 2.9 trên Python của Colab chưa thử; README dùng Python 3.11. Thời gian build chưa đo.
  - nvcc của Colab có phải 12.x không thì chưa biết.
  - `install.sh` có thể kéo thêm package khác.
  - Số đếm trên T4/A100 so với H200 chưa biết.
- **Trạng thái:** E1 và A2 **chưa xong**; chỉ có notebook đã sẵn sàng.
- **Tiếp theo:** user chạy notebook rồi gửi zip trong `export/`.
