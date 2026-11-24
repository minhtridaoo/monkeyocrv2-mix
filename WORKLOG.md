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


## 20/11/2026 — Sửa Ô 2 Colab: nvcc 13.0 / torch cu128

- Làm: helper kiểm compiler và cuda.h thực, ưu tiên toolkit 12.8; nếu thiếu tự cài cuda-toolkit-12-8 từ repo NVIDIA trên Ubuntu 22.04/24.04 x86_64; ghim CUDA_HOME/PATH khi build. Công cụ build được cài độc lập với mmengine. Sinh lại notebook và zip upload.
- Bằng chứng: user traceback; 51/51 test local PASS gồm tình huống PATH 13.0, tên folder sai phiên bản, thiếu header; kiểm syntax 6 code cells.
- Chưa kiểm/rủi ro: apt/build MMCV/GPU chưa chạy ở máy này; Python 3.13 còn chưa kiểm tương thích, E1 chưa PASS.
- Tiếp theo: user thay notebook và zip Drive, chạy Ô 1 rồi Ô 2; chỉ tiếp tục khi kiểm import thành công, doctor còn ở Ô 4.


## 20/11 — MMCV metadata trên Python 3.13
- Làm: vá setup.py get_version dùng namespace explicit, pip wheel verbose; sinh lại notebook/ZIP.
- Bằng chứng: traceback user KeyError __version__; Python PEP 667; kiểm patch trên upstream v2.0.1, đọc version, chạy lại patch và syntax các ô PASS.
- Chưa kiểm: build/import GPU; chưa xác nhận toàn stack hỗ trợ Python 3.13.
- Tiếp: user vá setup.py ở VM hiện tại rồi chạy lại Ô 2; không cần cài lại CUDA.


## 20/11 — MMCV build xong, import bi chan boi torchaudio ABI
- User log: MMCV 2.0.1 cp313 wheel built; import fails in torchaudio _torchaudio.abi3.so, undefined symbol torch_library_impl with torch 2.9.0+cu128.
- Fix: pin torchaudio 2.9.0+cu128 before MMCV check, preserve torch via --no-deps; log subprocess stderr and use return code plus marker. Regenerate notebook/ZIP.
- Validation: notebook syntax and isolated import checker (extra stdout, failed import, log) PASS. GPU install/import pending user; E1 not complete.


## 20/11 — MMOCR editable metadata
- Log user: passed MMCV import; mmdet 3.1.0 installed; MMOCR editable build requirements fails with hidden traceback.
- Upstream MMOCR v1.0.1 setup.py independently confirmed to use implicit exec/locals incompatible with Python 3.13. Added explicit namespace patch before install.sh, including short_version branch; verbose pip enabled.
- Check: upstream version function both branches and notebook syntax PASS. Colab install still pending; root traceback in user log not available.
- Next: patch existing VM setup.py, rerun cell 2 with verbose logs, no CUDA/MMCV rebuild needed.


## 20/11 — Cell 3 missing LFS review archive
- User: verify_bundle fails hash/size review/review_cards.zip. Generator only pulled two weights while upstream verifier checks all manifest files; .gitattributes confirms review ZIP uses LFS.
- Fixed include list to fetch entire pinned audit bundle plus weights, explicit empty exclude. Verifier unchanged. Notebook/ZIP regenerated; syntax/include checks PASS.
- Pending: user reruns LFS pull and original verifier on Colab. No experiment result claimed.


## 2026-11-20 · Đọc kết quả Colab A100: E1 xong, probe A2 chạy xong, cổng 3a FAIL 1/80
- **Làm:**
  - Đọc `H:\My Drive\CS431_MixA`: reports, gates, profile, identity, log.
  - Viết `scripts/compare_probe_reference.py` để so probe với NPZ H200 trên đủ 72 setting; chỉ mô tả, không đổi trạng thái cổng.
  - Chép JSON nhỏ và `NOTES.md` vào `results/probe_A100_v1/`.
  - Cập nhật kế hoạch §4.3 (kết quả probe, chờ Tuấn chọn) và điểm 2 của `docs/msg_tuan_20261120.md`.
- **Bằng chứng:**
  - `doctor_A100-SXM4-40GB_20261119_193007.json`: 22 PASS, `failed: false`.
  - `gates.json`: B-Clean 3a FAIL, 1/80 ô lệch (`9dfc8222c88569f0`, LR2, tp 144 so với 145); B1 3a PASS; P-map của cả hai max diff 0,0.
  - `compare_vs_H200_probe40.json`: B-Clean 104/5.760 ô lệch, B1 92/5.760; `gt` không lệch; |ΔHmean| lớn nhất 0,153 và 0,262 pp.
  - `profile.json`: 0,335 s/task, peak 1.153 MiB.
  - `rev4/stage_a.py:188–189`: cờ TF32 giống Tuấn.
  - MONKEY_V2 fetch: không có commit mới (`dc19b85`).
  - Test 51/51 PASS.
- **Chưa kiểm/rủi ro:**
  - Nguyên nhân lệch được suy luận là khác GPU (cuDNN/TF32), chưa có thí nghiệm tách riêng. Chưa kiểm Python bên Tuấn.
  - Chưa thử T4.
  - Chưa kiểm wheel MMCV cache có tái dùng được ở VM mới không.
- **Tiếp theo:** user gửi tin nhắn cho Tuấn để chọn (a) hay (b). Nếu (a), thêm Ô 7 chạy A2r đầy đủ trên A100 (khoảng 2,1 giờ). Chưa chạy A2r.


## 2026-11-20 · Tuấn chọn (a): chính sách cổng cùng GPU, notebook có Ô 7 (A2r) và Ô 8 (reselect)
- **Làm:**
  - Ghi quyết định vào kế hoạch §4.3, dẫn lời user: "Tuấn chọn a".
  - `merge_audit`: thêm `--reference-policy same_gpu_endpoints` và `--same-gpu-probe`; hàm `identity_mismatch`/`same_gpu_reference` kiểm identity trùng probe trước khi dùng GPU; spec ghi chính sách.
  - `reselect_split`: thêm `REQUIRED_CHECKS` theo chính sách; `gate_same_gpu` so đủ 11 cột chính xác; `cross_gpu_drift` chỉ để báo cáo.
  - `colab_helpers.summarize_probe`: thêm `ready_for_full_run_same_gpu`.
  - Notebook có thêm Ô 7 và Ô 8; Ô 6 sửa lời theo (a). Cập nhật runbook §4–5.
- **Bằng chứng:**
  - Test 54/54 PASS (thêm 3 test: chính sách cùng GPU, identity mismatch, khối tham chiếu). Chạy từ zip giải nén: 54 test, 2 skip vì không có `results/`.
  - nbformat validate (cảnh báo coi là lỗi) OK; 8 code cell compile.
  - Thử đầu-cuối với file probe A100 thật: `same_gpu_checks` 5760/5760 khớp, `verification = VERIFIED`, `cross_gpu_drift` 3a/3b FAIL như kỳ vọng.
  - Zip code SHA `e4d54077…`.
- **Chưa kiểm/rủi ro:**
  - Ô 7–8 chưa chạy trên Colab.
  - Identity của phiên Colab mới có thể khác probe (cuDNN/torch, tên GPU 40GB/80GB). Khi đó Ô 7 dừng sớm, hoặc Ô 5 chạy probe mới cho GPU đó.
  - Identity không hash code `mixab`; cổng `gate_same_gpu` chính xác bù cho điều này.
  - Mốc tiny 42% vẫn chờ Tuấn.
- **Tiếp theo:** user upload zip mới, chạy Ô 1 → 4, 5, 7, 8 trên A100, rồi gửi `export/cs431_A2r_*.zip`.

## 2026-11-20 · Codex đồng bộ dashboard sau quyết định A100
- **Làm:** cập nhật dashboard: E1/probe đã xong; A2r chờ user chạy Ô 7–8 trên A100; sửa mô tả A1 cũ, phụ thuộc E1 trước probe, A4 theo baseline cùng A100. Làm rõ R/Mix B là đề xuất chưa phân công, không coi là Tuấn đã nhận làm. Đồng bộ ghi chú HANDOFF.
- **Bằng chứng:** user yêu cầu cập nhật; HANDOFF hiện hành, notebook `notebooks/colab_E1_A2_probe.ipynb` có Ô 7–8; kết quả probe ở `results/probe_A100_v1/`.
- **Chưa kiểm/rủi ro:** không chạy GPU, không xác nhận Ô 7–8 đã chạy. Tick cục bộ trong trình duyệt có thể đè trạng thái file; giữ nguyên dữ liệu trình duyệt.
- **Tiếp theo:** user upload code mới và chạy notebook trên A100; gửi gói export để đánh giá.


## 2026-11-20 · A2r xong trên A100; Tuấn chốt tiêu chí; Mix A-core v1 chưa thắng theo quy tắc đã khóa
- **Làm:**
  - Ghi ý kiến Tuấn vào kế hoạch §4.2 trước khi mở `decision.json`: R1, +0,5 pp so với cả hai baseline, tiny ≥ B1 − 3 pp, CI paired, B không phải holdout.
  - Sửa `reselect_split` theo mốc tiny tương đối và thêm test biên (−3 pp đạt, −3,2 pp trượt).
  - Kiểm gates, identity và log của A2r.
  - Chạy lại reselect ở local trên NPZ Drive → `results/mix_a_v1/`.
  - Đo độ lệch so với H200 trên toàn cohort.
  - Viết `NOTES.md` và nháp `docs/msg_tuan_20261120_mixA_result.md`.
- **Bằng chứng:**
  - `gates.json`: P-map 0,0 ở cả hai endpoint; 3a so với H200: 72/4.524 và 66/4.524 ô lệch (chỉ báo cáo).
  - `decision.json`: `VERIFIED` (`same_gpu_endpoints`), `gate_same_gpu` 0/5.760 ô lệch; α\* = 0; `passes_margin` và `passes_tiny` đều false.
  - Tiny B1 trên B 37,63%; α\* 16,08%.
  - `selection.json`: tổng trên A B-Clean 157,016, α 3/4 156,600.
  - `compare_vs_H200_full.json`: |ΔHmean| ≤ 0,023 pp.
  - `merge_manifest.json`: endpoint bằng nhau; BN nội suy, không ước lượng lại.
  - Identity A2r chỉ khác probe ở tập nguồn.
  - Test 55/55 PASS.
- **Chưa kiểm/rủi ro:**
  - Ô 8 Colab chạy trước khi đổi mốc tiny. α\* không phụ thuộc mốc này, và bản chính thức đã chạy lại với mốc mới.
  - Quan sát α 3/4 là sau khi khóa: không dùng để đổi kết luận.
  - Một seed; B không phải holdout.
- **Tiếp theo:** user gửi Tuấn; vẽ đường α và hoàn thiện báo cáo R1; chờ Tuấn về Model Stock và thân R của Mix B.


## 2026-11-20 · E-sel (khám phá chọn có ràng buộc tiny) + thiết kế pilot BN
- **User yêu cầu:** giữ kết luận âm v1; phân tích trên CPU việc chọn có ràng buộc tiny rồi tối ưu mean Hmean, ghi rõ là khám phá; kiểm khả thi pilot tính lại BN bằng TRAIN có đối chứng; chưa chạy GPU.
- **Làm:**
  - Ghi luật E-sel và E-BN vào kế hoạch §4.4 trước khi chạy.
  - Viết `mixab/explore_selection.py` và 5 test; chạy trên counts A2r → `results/explore_sel_v1/`.
  - Khảo sát BN: key checkpoint, mã MMOCR v1.0.1 (`db_head.py`, `single_stage_text_detector.py` qua raw GitHub), `update_bn` trong torch local, tài liệu PyTorch `AveragedModel` 2.15.
  - Viết `docs/design_bn_recalibration_pilot.md`.
- **Bằng chứng:**
  - `explore_selection.json`: t_ref 48,82%; B-Clean và α 1/4 có 0 setting khả thi; α\* = 3/4; trên B mean 78,34 so với 77,87 và 78,11, tiny 34,56 so với 37,63.
  - FP ở cùng recall: α 3/4 7.163 / 5.781 so với B1 6.685 / 5.456 (A / B).
  - Checkpoint có 4 lớp BN, đều ở DBHead; MMOCR `DBHead.forward` mode `predict` chỉ chạy `binarize`.
  - ‖ΔW‖/‖W‖ = 1,9%.
  - `rev4/ssim_calibration.py:16–17`: tên và SHA manifest TRAIN; file không có trong repo.
  - Test 60/60 PASS.
- **Chưa kiểm/rủi ro:**
  - Kỳ vọng hiệu ứng BN nhỏ: đây là suy luận.
  - Thời gian GPU cho E-BN chưa đo.
  - Chưa biết nhóm có tập validation chưa dùng hay không.
- **Tiếp theo:** user gửi Tuấn (kết quả v1, E-sel, xin manifest TRAIN, xin duyệt thiết kế); Claude vẽ đường α và viết báo cáo R1.

## 2026-11-20 · Chuẩn bị đồng bộ kết quả Mix A-core v1 lên repo riêng
- **Làm:** rà soát các thay đổi từ commit đầu tiên; bổ sung liên kết kết quả và trạng thái mới vào README; cập nhật mốc tham chiếu MONKEY_V2 sau khi fetch và fast-forward.
- **Bằng chứng:** `results/mix_a_v1/NOTES.md` ghi α\* = B-Clean, `VERIFIED` theo chính sách cùng GPU, chưa thắng; `results/explore_sel_v1/NOTES.md` ghi rõ kết quả khám phá. `python -m unittest discover -s tests -t . -q`: 60/60 PASS. MONKEY_V2 `aa1e17f` thêm chẩn đoán residual và hai pilot độc lập, không đổi mã `rev4/stage_a.py` được dùng bởi run Mix A v1 tại `dc19b85`.
- **Chưa kiểm/rủi ro:** NPZ đếm đầy đủ, checkpoint và output lớn ở Drive/`runtime/`, không nằm trong Git; Git chỉ chứa tóm tắt, identity và các JSON/CSV nhỏ. B đã được xem trước, không phải holdout mới.
- **Tiếp theo:** đồng bộ repo riêng lên GitHub; gửi đường dẫn commit cho user; báo cáo R1 vẫn cần hình đường α.


## 2026-11-20 · Chuẩn bị Giai đoạn 0 pilot BN (code + notebook riêng, chưa chạy GPU)
- **User yêu cầu:** chuẩn bị pilot BN để tự chạy trên Colab A100. Kiểm code dựng lại TRAIN; notebook nhận manifest hoặc danh sách ID qua Drive; tự lấy ảnh TextVQA và tạo LR2; Giai đoạn 0 có đối chứng endpoint, resume được; Giai đoạn 1 chỉ khi đã thống nhất; giữ kết luận âm v1.
- **Tuấn xác nhận** (user chuyển lời): ảnh TextVQA, nhãn TextOCR; LR2 = `native_lr2`; TRAIN 15.222 nguồn FULL_R1; SHA manifest `ca1b9163…c983d`; manifest đầy đủ chưa có bản tải.
- **Làm:**
  - Đọc `rev4/training_data.py` (`recover_train`, `check_split`, `verify_clean`, `native_lr2`), `rev4/train_recovery.py`, `recover_b1_train.py`. Dựng lại TRAIN cần metadata job 8323 trên `/datastore`, repo cũ `monkey_textdet` @ `816c6d6a…` và manifest validation 3.262 nguồn: không có, nên không dựng lại.
  - Viết `mixab/bn_recal.py` và 10 test.
  - Viết `scripts/build_bn_notebook.py` → `notebooks/colab_bn_pilot_stage0.ipynb`.
  - Thêm mục trạng thái vào `docs/design_bn_recalibration_pilot.md` §0.
- **Bằng chứng:**
  - Test 70/70 PASS, gồm mô phỏng stage0 bị ngắt rồi chạy tiếp với kết quả giống hệt, và kiểm `state_dict` không đổi.
  - Chạy thử `check-train` bằng danh sách giả 15.222 nguồn trên bundle thật: PASS, subset 1.000.
  - nbformat hợp lệ; 8 code cell compile.
  - SHA notebook v1 không đổi. Zip code `36bd0a8d…`.
- **Chưa kiểm/rủi ro:**
  - Toàn bộ phần GPU (stage0 với MMOCR thật, thời gian mỗi ảnh) chưa chạy.
  - `native_lr2` thật chỉ chạy trên Colab (local Pillow 12.2; test dùng hàm thay thế).
  - Không kiểm được trùng với 1.000 nguồn S1/S2.
  - Cổng 2× endpoint là đề xuất chưa duyệt.
- **Tiếp theo:** user xin file TRAIN từ Tuấn, upload zip code, chạy notebook BN trên A100, gửi export.


## 2026-11-20 · Đọc kết quả E-BN Giai đoạn 0 (Colab T4): không có tín hiệu lệch BN do trộn
- **Làm:**
  - Giải nén `export/cs431_bn_stage0_T4_20261120_085809.zip` vào `runtime/bn_stage0_T4_export/`.
  - Đọc `train_check`, identity, summary, 5 file theo α, doctor, env và log.
  - Đối chiếu file TRAIN với repo Tuấn: fetch và fast-forward MONKEY_V2 lên `0c6c5b0`.
  - So code zip đã upload với code local.
  - Chép JSON vào `results/bn_stage0_v1/`, viết `NOTES.md`.
  - Ghi kết quả vào kế hoạch §4.4 và §0 của tài liệu thiết kế.
- **Bằng chứng:**
  - `train_ids.csv` có SHA `8ac5cbd9…`, trùng `docs/handoff/data/train_ids.csv` @ `e4a2a89` (receipt PASS, nguồn là manifest `ca1b9163…`).
  - `train_check` PASS (15.222 nguồn, 0 trùng với 2.262 nguồn remaining). Subset `0463e5d0…`; 1.000 Clean khớp SHA; 1.000 LR2 ghi SHA lần đầu.
  - `stage0_summary.json`: gate `signal=false`; z `binarize.1` α ở giữa 0,157–0,159 so với endpoint tối đa 0,161; phi tuyến ≤ 0,0053 trung bình, ≤ 0,0139 lớn nhất.
  - Tỉ lệ phương sai đo/lưu, trung vị: khoảng 0,45 ở α 0, ½, 1 (100% kênh < 1).
  - Digest `state_dict` không đổi. Profile 0,47 s/ảnh.
  - Code zip 26/26 file trùng local.
  - GPU: Tesla T4; doctor 0 FAIL, 1 WARN (không TF32).
- **Chưa kiểm/rủi ro:**
  - Chạy T4 chứ không phải A100.
  - Nguyên nhân lệch nền 0,45 lần là suy luận: không tính được phương sai gộp vì file accumulator tạm đã xóa sau khi xong.
  - Không có lần tạo LR2 thứ hai để kiểm byte-identical.
  - Cổng là đề xuất chưa duyệt.
- **Tiếp theo:** gửi Tuấn đề xuất dừng E-BN; viết báo cáo Mix A (R1).


## 2026-11-21 · Chốt dừng pilot BN, sửa câu chữ theo mức bằng chứng, viết báo cáo Mix A v1
- **User chốt:** Giai đoạn 0 không có tín hiệu BN bất thường riêng do trộn; không mở Giai đoạn 1; giữ kết luận v1; đóng v1 là kết quả âm, không bỏ mục tiêu cải tiến; tiếp theo hoàn thiện báo cáo rồi xem hướng latent của Tuấn. User cũng yêu cầu sửa 3 câu nói quá chắc.
- **Làm:**
  - Sửa câu chữ ở `results/bn_stage0_v1/NOTES.md`, kế hoạch §4.4, §0 của tài liệu thiết kế:
    - "thống kê nội suy đã đúng" → "chưa thấy sai lệch đáng kể do phép trộn trong cách đo này";
    - "endpoint bị kéo lệch" → "có thể đổi đáng kể cả endpoint, chưa biết tốt hay xấu";
    - thêm giới hạn của phương pháp đo, và ghi lệch 0,45 lần không chứng minh pipeline có lỗi;
    - ghi T4 chỉ là chẩn đoán, không so điểm với A100;
    - ghi test PASS chỉ xác nhận phần mềm.
  - Viết `scripts/plot_alpha_curve.py` → `results/mix_a_v1/alpha_curve_B.png`; đã xem ảnh, sửa nhãn đè lên đường.
  - Viết `docs/report_mix_a_v1.md` và nháp `docs/msg_tuan_20261121_close_mixA.md`.
  - Fetch và fast-forward MONKEY_V2 lên `ae947e9`; đọc `docs/latent-and-craft-trials.md`, `latent-recovery-v3.md`, `latent-10k-extension.md`.
- **Bằng chứng:**
  - Số liệu trong báo cáo lấy từ `results/mix_a_v1/{selection.json, evaluation_B.csv, bootstrap_B.json, decision.json}`, `results/explore_sel_v1/explore_selection.json` và `results/bn_stage0_v1/stage0_summary.json`.
  - Câu về pilot latent trích `docs/latent-10k-extension.md` ("near-zero adapter/control differences and did not establish efficacy").
  - Test 70/70 PASS.
- **Chưa kiểm/rủi ro:**
  - Chưa thấy file kết quả latent (nằm ở `.runtime` trên HPC).
  - Đề xuất ghép FSR vào recipe latent chưa bàn với Tuấn.
- **Tiếp theo:** user gửi báo cáo và tin nhắn; chờ kết quả latent 10k.


## 2026-11-21 · Chẩn đoán gốc rễ Mix A v1 + research gap + 3 thiết kế v2 (chỉ thiết kế)
- **User yêu cầu:** đóng băng v1; chẩn đoán vì sao v1 thất bại; khảo sát theo 2 trục; 2–3 thiết kế có ablation, chi phí và điều kiện dừng; chọn 1 và thiết kế pilot; chưa code notebook, chưa chạy GPU; tuân thủ Integrity Gate.
- **Làm:**
  - Đọc `ai_research_failure_modes.md` (7 mode, Stage 2.5). Khung hướng dẫn OCR yêu cầu thì không tồn tại **[đính chính 22/11: đường dẫn đó vẫn không có; có file nội dung phù hợp tên khác nhưng chưa xác nhận là thay thế; xem hai mục 2026-11-22]**.
  - Fetch MONKEY_V2 (`ae947e9`). Đọc báo cáo của Tuấn: restoration, paired EDA, PRC, B1, B-Clean, G4, S2-A/B, calibration, B1-cal diagnostics, fixed diagnostics, KD pilot, latent docs.
  - Tính độ tuyến tính theo α và định tuyến theo điều kiện từ `results/mix_a_v1/evaluation_B.csv` → `results/diag_v2_cpu/partial_from_csv.json`.
  - Viết `scripts/diagnose_tradeoff_v2.py`; chưa chạy vì Drive không mount.
  - Kiểm trích dẫn (WiSE-FT, TIES-Merging, WBF) qua tìm kiếm; ba paper MoE chỉ ở mức kết quả tìm kiếm.
  - Viết `docs/research_v2_design_20261121.md`.
- **Bằng chứng:**
  - LR2 Hmean theo α cao hơn đường thẳng +0,52 đến +0,69 pp, không vượt B1; Clean lệch đường thẳng ≤ 0,13 pp.
  - Định tuyến theo điều kiện: mean B 78,93 (Clean 81,16, LR2 76,69, tiny 37,6%); +0,58 pp so với α tốt nhất 78,34.
  - EDA của Tuấn: B-Clean → B1 trên LR2, cứu 6.150 / mất 1.453.
  - EDSR: +0,60 pp, +952 FP.
- **Chưa kiểm/rủi ro:**
  - Q1 (frontier) và Q4 (router kích thước) chưa chạy.
  - Mode 4: router kích thước khai thác cách tạo LR2; khác biệt JPEG/PNG có thể là shortcut.
  - Mode 7: chưa biết nhóm có ràng buộc một model hay không.
  - Mọi số trên B là khám phá.
- **Sửa một con số đã báo trước đó:** trong tin nhắn trước tôi viết định tuyến "cao hơn model đơn tốt nhất 0,82 pp". Đúng ra là +0,82 so với **B1**; so với model đơn tốt nhất trên B (α ¼ hoặc ¾, 78,34) là **+0,58 pp**.
- **Tiếp theo:** user/Tuấn trả lời 4 câu hỏi; mount Drive để chạy P0; nếu được duyệt thì code P1–P4 (CPU).

## 2026-11-22 · Cập nhật tiến độ + chạy P0 (CPU)
- **Làm:**
  - Fetch MONKEY_V2: không có commit mới, vẫn `ae947e9` (latent 10k chưa có kết quả).
  - Drive H: đã mount lại → chạy `python scripts/diagnose_tradeoff_v2.py` (CPU, ~2 s, exit 0) → `results/diag_v2_cpu/diagnosis.json`.
  - Kiểm riêng 3 ảnh Clean 768×768 trên B mà router kích thước gửi sang B1: counts B-Clean (`mask_040/score_0.60`) và B1 (`unclip_1p8/score_0.50`) giống hệt nhau, nên Q4 trùng số với Q3.
  - Cập nhật HANDOFF, dashboard.
- **Bằng chứng** (`diagnosis.json`; số trên B là khám phá):
  - Q4: T = 768; cạnh dài Clean (quantile) 768–1024, LR2 384–512; tỉ lệ gửi sang B1: Clean 0,27%, LR2 100%. Mean B 78,93 = Q3 oracle (Clean 81,16 / LR2 76,69).
  - Q1 (A, LR2): setting không bị endpoint áp đảo: α¼ 13/72, α½ 17/72, α¾ 14/72; Clean 5/4/1.
  - Q2 (B): LR2 bin < 4 px, α½ +5,65 pp trên đường thẳng; Clean bin < 4 px, α½ −1,63 pp.
  - Test 70/70 PASS.
- **Chưa kiểm/rủi ro:**
  - Q1 chỉ đếm điểm không bị áp đảo, chưa đo biên độ vượt frontier.
  - Q2 bin nhỏ, chưa có CI.
  - Mode 4 xác nhận cho router kích thước: trong protocol native LR2, kích thước lộ điều kiện. Router size-free vẫn có thể học shortcut nén (chưa kiểm, P1).
  - Tài liệu v2 (§1 chẩn đoán, §4 Integrity Gate) chưa sửa theo P0.
- **Tiếp theo:** user/Tuấn trả lời 4 câu hỏi §3 (câu 2 giờ là câu quyết định); sửa tài liệu v2 nếu user đồng ý; nếu được duyệt thì code P1–P4.


## 2026-11-22 · Đính chính: khung thực chiến và tên skill
- **Làm:**
  - User chỉ ra file `LABB/workspace/Chỉ dẫn master prompt gửi gemini.md`. Đọc: `research-ideation`, `model-architecture`, `mechanistic-interpretability` là tên nhóm (Nhóm C, Orchestra-Research), không phải tên skill.
  - Kiểm `~/.claude/skills/` (65 mục) và `~/.codex/skills`: skill con có sẵn (`brainstorming-research-ideas`, `creative-thinking-for-research`, `mamba`, `rwkv`, `transformer-lens`, `nnsight`, `pyvene`, `saelens`...).
  - Khung thực chiến nằm ở `LABB/workspace/Step by step research.md` (đọc xong); đường dẫn `h_ng_d_n_...md` ghi trong file Gemini không có trên máy. `ai_research_failure_modes.md` có đúng chỗ.
  - Sửa câu sai ở HANDOFF, WORKLOG (mục 21/11), `docs/research_v2_design_20261121.md`, `_archive/Method_Transfer_Plan_LR2_FP_Geometry.md`, `dashboard/status.js`.
- **Bằng chứng:** `ls` hai thư mục skill và `LABB/workspace`; nội dung hai file đã đọc.
- **Chưa kiểm/rủi ro:**
  - Thiết kế v2 chưa đối chiếu với `Step by step research.md` (Bước 5.5 sanity check, tầng insight). Skill ideation/interpretability chưa kích hoạt; chưa biết có hợp với bài detection không.
  - Không đổi số liệu hay kết luận khoa học nào.
- **Tiếp theo:** đối chiếu thiết kế v2 với khung (nếu user muốn); các việc cũ ở HANDOFF giữ nguyên.


## 2026-11-22 · Đính chính lần 2: đối chiếu khung, ánh xạ skill (chỉ tài liệu)
- **Làm:**
  - Đọc đủ `LABB/workspace/Step by step research.md` và `Chỉ dẫn master prompt gửi gemini.md`; đọc lại HANDOFF/WORKLOG/dashboard hiện tại.
  - Sửa chỗ quá mức của mục "Đính chính" ngay phía trên (mục cùng ngày giữ nguyên làm lịch sử): `Step by step research.md` chỉ là **nội dung phù hợp**, **chưa xác nhận** là file master prompt trỏ tới (`find` không ra file `h_ng_d_n_…`; không có bản sao/hash nối hai file; chỉ có dấu hiệu gián tiếp về tên và tiêu đề). Cần user xác nhận.
  - Ánh xạ nhóm → skill từ `0-autoresearch-skill/references/skill-routing.md`; đọc đủ `brainstorming-research-ideas` và áp dụng lọc Converge lên 3 thiết kế (không đổi lựa chọn QRoute).
  - Viết `docs/audit_v2_vs_framework_20261122.md`; sửa HANDOFF, `docs/research_v2_design_20261121.md`, `_archive/Method_Transfer_Plan…md`, `dashboard/status.js`.
- **Bằng chứng:** `find`/`grep`/`ls` trong `LABB/` và `~/.claude/skills`; nội dung các SKILL.md đã mở (nêu dòng đã đọc trong file audit §4).
- **Chưa kiểm/rủi ro:**
  - `creative-thinking-for-research`, `nnsight`, `pyvene` chỉ đọc phần đầu, chưa áp dụng; `transformer-lens`, `saelens`, `mamba`, `rwkv`, `litgpt`, `nanogpt` chỉ xác nhận có cài (nhóm `model-architecture` không có skill detector).
  - Chưa kiểm khả năng chạy `nnsight`/`pyvene` trên DBNet/MMOCR.
  - Chưa kiểm master prompt (Đề tài 2, Lab cô Quý) có áp cho đồ án CS431 hay không; chưa đếm lại "66/59 skills".
  - `/audit-analysis`, `/review-paper-code` nằm ở `LABB/.claude/skills/`, không có trong project này.
  - Đối chiếu v2 với khung ứng viên: thiếu Bước 5.5; Bước 4 chỉ counts (human review 0/160); gap chưa tách core/sub-gap. Ba đề xuất chưa làm vì cần duyệt.
  - Không đổi số liệu hay kết luận khoa học; không chạy test/GPU.
- **Tiếp theo:** user xác nhận file khung; duyệt hay bỏ 3 đề xuất ở `docs/audit_v2_vs_framework_20261122.md` §2; các việc cũ ở HANDOFF giữ nguyên.

## 2026-11-22 · Kiểm review 6 điểm về thiết kế v2 + P0 sửa lỗi rò nhãn
- **Làm:**
  - User dán review (nguồn bên ngoài) với 6 điểm về `docs/research_v2_design_20261121.md` và script P0. Đối chiếu từng điểm với tài liệu và code.
  - Viết `scripts/p0_fixed_router_eval.py`: một setting mỗi expert khóa trên A; chọn chung (T, k_bclean, k_b1); đối chứng một detector + setting theo điều kiện; bootstrap theo nguồn (2.000 lần, seed 431). Output `results/diag_v2_cpu/p0_fixed.json`.
  - Thêm ghi chú KNOWN FLAW vào Q4 của `diagnose_tradeoff_v2.py`. Không sửa tài liệu v2.
- **Bằng chứng** (`p0_fixed.json`):
  - Model đơn ở setting R1 (mean A / B): B-Clean 78,51 / 77,87; α¼ 78,18 / 78,34; α½ 78,26 / 78,28; α¾ 78,30 / 78,34; B1 78,17 / 78,11.
  - Oracle điều kiện (B-Clean `mask_040/score_0.60`, B1 `epsilon_005/score_0.50`): A 78,78 (+0,27 pp so với B-Clean), B 78,93 (+0,58 pp so với α¼).
  - Router kích thước sửa lỗi: T = 768, cùng hai setting; A 78,78, B 78,93 (lệch so với bản lỗi 0,001 pp).
  - Đối chứng hậu xử lý theo điều kiện: tốt nhất A = B-Clean 78,51 (không đổi), B = α¾ 78,49.
  - Bootstrap B, oracle − B-Clean: +1,07 pp, CI95 [0,21; 1,89].
  - Cạnh dài: Clean 768–1024, LR2 384–512 trên A+B (LR2 = chia đôi, giả định).
  - Test 70/70 PASS.
- **Kết quả đối chiếu review:** đúng cả 6 điểm.
  1. Câu "Không phải do BN, cũng không phải do hậu xử lý" (§1, dòng 46) quá mạnh; E7 (1,9%) không chứng minh không có độ bền bổ sung.
  2. Rò nhãn có thật (dòng 122–138). Review ghi "P0 chưa chạy" là cũ: P0 đã chạy sáng 22/11 với lỗi này; tác động lên số hiện tại không đáng kể.
  3. Logic ở §3 câu 2 sai; dữ liệu cụ thể thì tách được, nhưng kết luận chỉ đúng cho quy trình LR2 này.
  4. Balanced accuracy nên là chẩn đoán, không phải cổng.
  5. Số trên B khớp (78,3404 / 78,9251 / 78,8404). Trên A còn hẹp hơn: +0,27 pp.
  6. Dòng 168 ("cả hai thiết kế kia cũng mất cơ sở định tuyến") quá mạnh.
- **Chưa kiểm/rủi ro:**
  - Chưa tính oracle theo từng ảnh (chọn expert tốt hơn cho mỗi ảnh); đây là trần thật của router, không phải oracle điều kiện.
  - Setting R1 của model đơn và setting oracle đều chọn trên A, nên số trên A là in-sample.
  - Kích thước LR2 suy ra, chưa đọc file.
- **Tiếp theo:** user quyết định hướng QRoute (biên trên A dưới cổng); sửa tài liệu v2 nếu user đồng ý.


## 2026-11-22 · Sửa tài liệu v2 theo review 6 điểm và P0 sửa lỗi (user cho phép)
- **Làm:**
  - Vá `docs/research_v2_design_20261121.md` bằng script (mỗi chuỗi cũ phải khớp đúng 1 lần; 17 chỗ; script tạm đã xóa). Chỗ sửa đều đánh dấu *[22/11]*, bản gốc 21/11 vẫn đọc được qua ngữ cảnh đánh dấu.
  - Nội dung: thêm E13–E16 (kích thước, biên trên A +0,27 pp, đối chứng hậu xử lý, frontier/bin); mềm hóa "không phải BN/hậu xử lý" (điểm 1) và "lệch giả định" (E7 1,9% không chứng minh); ghi rò nhãn ở Q4 cũ và bản sửa (điểm 2); sửa logic kích thước ở §3 câu 2 (điểm 3); balanced accuracy chuyển từ cổng PASS/kill thành chẩn đoán, cổng còn ba điều kiện tính với lỗi định tuyến thật và đặc trưng đã qua hai đối chứng (điểm 4); ghi biên trên A và bỏ "bằng chứng mạnh nhất" khỏi lý do chọn (điểm 5); Thiết kế 2/3 không phụ thuộc router khi pilot (điểm 6); Integrity Gate Mode 1 → CLEAR (giới hạn), Mode 4 → CONFIRMED cho router kích thước, nén vẫn SUSPECTED.
  - Cập nhật HANDOFF (bỏ "chưa sửa", đánh dấu xong các bước 2 và 4 cũ, sửa số thứ tự trùng) và dashboard.
- **Bằng chứng:** script báo "OK, đã vá 17 chỗ"; `grep` xác nhận các câu cũ ("Không phải do BN", "mất cơ sở định tuyến", "loại trừ") không còn ở dạng gốc; số liệu E13–E16 chép từ mục WORKLOG "Kiểm review 6 điểm" và `p0_fixed.json`/`diagnosis.json` (không tính lại trong lần này).
- **Chưa kiểm/rủi ro:**
  - Bỏ balanced accuracy khỏi cổng là **đổi thiết kế cổng**, làm sau khi đã thấy số P0 trên A; ngưỡng còn lại không đổi nhưng Tuấn chưa duyệt bản này. Điều kiện +0,5 pp trên A **không đạt được** bằng router theo điều kiện (oracle +0,27 pp).
  - Chưa tính oracle theo từng ảnh; số trên A là in-sample; kích thước LR2 suy ra, chưa đọc file.
  - Chưa đọc lại toàn văn tài liệu sau khi vá để soát giọng văn; chưa làm ba đề xuất của file audit (Bước 5.5, tách gap, đối chiếu nhãn Bước 4).
  - Không chạy test (không đổi code); 70/70 test của lần trước không được chạy lại ở đây.
- **Tiếp theo:** user quyết định hướng QRoute; Tuấn duyệt cổng; trả lời 4 câu hỏi §3.

## 2026-11-22 · Giải thích hướng v2 cho user + trần router theo từng ảnh (P0b, CPU)
- **Làm:** user hỏi hướng mới cụ thể làm gì, nằm đâu trong pipeline, có đáng không, đã triển khai đến đâu. Đọc lại tài liệu v2 (bản vá 22/11). Viết `scripts/p0b_per_image_oracle.py` (oracle từng ảnh, Dinkelbach, setting khóa trên A) và `scripts/p0b_null_pairs.py` (đối chứng cặp gần giống nhau).
- **Bằng chứng:** `results/diag_v2_cpu/p0b_per_image_oracle.json`; output `p0b_null_pairs.py` (A, Clean/LR2 pp): α½–α¾ 0,37/0,54; α¾–B1 0,36/0,64; B1 hai ngưỡng 1,05/0,85; B-Clean–B1 1,19/3,70. Oracle từng ảnh A 81,22, B 80,78. Test 70/70 PASS.
- **Chưa kiểm/rủi ro:** oracle từng ảnh dùng GT, là trần lạc quan; chưa biết đặc trưng nào dự đoán được expert tốt hơn trên ảnh LR2; counts A2r không có box/score; Tuấn chưa trả lời câu cho phép 2 checkpoint.
- **Tiếp theo:** user chọn (a) chuẩn bị lần GPU lưu box + score cho LR2 (Thiết kế 2) hoặc (b) dừng QRoute.


## 22/11 — Đồng bộ Tuấn 96612b4 và pipeline
- Làm: fetch + fast-forward reference ae947e9→96612b4, giữ scripts/ untracked; đọc commit/doc/fusion_plan; cập nhật kế hoạch chung, HANDOFF, dashboard.
- Bằng chứng: repo mới nói latent 1k/10k chưa hơn B1/control; CRAFT LR2 47,00%, tiny<4px 0 TP; fusion bốn arm và stress đã code.
- Chưa kiểm/rủi ro: chưa có raw report latent10k/CRAFT để tính lại; không xác nhận submit/completion fusion. Không đổi protocol, không chạy GPU/push.
- Tiếp: phối hợp v2 output của Trí với fusion FPNC của Tuấn; không tiếp tục ghi chờ latent10k.

## 2026-11-22 · DualAgree v1: code + test CPU + notebook (chưa chạy GPU)
- **User yêu cầu:** "thử triển khai xem có cải tiến không, bắt đầu đi".
- **Làm:**
  - Fetch MONKEY_V2: không có commit mới.
  - Đọc `rev4/decision_core.py` (`capture_candidates`, `CandidateMatcher`), `rev4/stage_a.py` (`detector_for`, `forward`), `lowres/inference.py` (`unpack`), `rev4/paired_eda_core.py` (`filtered_geometry`). `rev4/fusion_*` của Tuấn là fusion FPN/upsampling, **không trùng** với gộp box.
  - Cài `shapely` 2.2.0 (user site) để chạy matcher của Tuấn trên CPU local.
  - Viết `mixab/dualagree.py`, `tests/test_dualagree.py` (8 test), `scripts/build_dualagree_notebook.py` → `notebooks/colab_dualagree_v1.ipynb`; thêm `p0_fixed.json` vào `make_colab_upload.py`.
  - Chạy thử analyze trên dump giả (`/tmp/da_dry`, không lưu vào workspace) để kiểm đường ống. Sửa luật "FP ở cùng recall": so với cả 72 setting B1 từ NPZ A2r; nếu B1 không setting nào đạt recall đó thì coi là không bị áp đảo.
- **Bằng chứng:** 78/78 test PASS; nbformat validate OK, 7 code cell parse OK; `make_colab_upload.py` → 35 file, SHA-256 `ce1e20927acecf214172e579de9dd9c18a3bb38c171826f94ac1aac95cd74337`; dry-run analyze exit 0, fidelity DRIFT trên dump giả (đúng kỳ vọng).
- **Chưa kiểm/rủi ro:**
  - Đoạn `dump` (GPU) chưa chạy lần nào; cổng trung thực ở Ô 5 là chốt kiểm.
  - Luật v1 không có WBF; θ, t_a, t_c, t_l quét 1.928 tổ hợp trên A → nguy cơ khớp quá A, B chỉ là khám phá.
  - Tuấn chưa duyệt hệ thống 2 checkpoint; suy luận ảnh LR tốn 2×.
  - Thời gian dump là ngoại suy, chưa đo; T4 chưa đo.
- **Tiếp theo:** user tải zip mới, chạy notebook trên A100; Claude đọc `runs/dualagree_analysis_v1_<GPU>/decision.json` từ Drive và báo cáo.

## 2026-11-22 · Đọc đề xuất S-ACV/SA-DB/ROI-Focus (file user gửi) + trả lời "chạy notebook xong có kiểm được không"
- **Làm:** đọc `C:/Users/ADMIN/Downloads/xu_t_nghi_n_c_u_c_i_ti_n_monkeyocrv2_cs431.md` (348 dòng, đề xuất do Gemini viết theo lời user). Đối chiếu số liệu với repo Tuấn.
- **Bằng chứng:**
  - Khớp: B-Clean/B1 ở `.40/.60` toàn cohort (P 88,66/82,45%, FP 6.578/11.997, tiny 15,96/45,79%) ở `MONKEY_V2/docs/handoff/artifacts/pr_quad_audit_v1/results/decision_audit_report.md:12-13`; `Resize scale=(4068, 1024)` ở `docs/handoff/inference_config.py:32-35`; số P0/P0b khớp với file của mình.
  - Không kiểm được: "latent 10k âm theo cập nhật Tuấn" (repo chưa có kết quả 10k); câu "bổ trợ nằm ở phân bố kích thước chữ" là suy luận, P0b chưa tách theo kích thước; các trích dẫn văn liệu chưa mở bài gốc.
  - MONKEY_V2 local đang ở `96612b4` ("single-job pilot/full fusion study and fixed-threshold stress transfer" = nhánh FPNC ASF/DySample, chỉ code); fetch không có gì mới hơn.
- **Tiếp theo:** user chạy notebook DualAgree; có thể thêm vào phần phân tích CPU: (a) ngưỡng theo kích thước cho B1 đơn (1 model), (b) luật gộp phân tầng kích thước — không cần đổi notebook.

## 2026-11-22 · Kiểm dump đủ cho ngưỡng theo kích thước (B1) + viết phân tích `mixab/sizethr.py`
- **Bối cảnh:** user đang chạy notebook DualAgree trên **T4** (không phải A100). Trên Drive: `runs/dualagree_probe_v1_T4/` xong; `runs/dualagree_dump_v1_T4/` đang chạy. Zip code trên Drive (`upload/mixab_code (1).zip`) có SHA trùng zip local.
- **Bằng chứng kiểm dump (đọc shard probe thật, 40 ảnh LR2):**
  - Polygon 8 tọa độ (quad) ở hệ tọa độ Clean, raw + packed score, đủ 9 geometry.
  - Score nhỏ nhất mỗi geometry ≈ ngưỡng mask (0,29–0,40; mask_020: 0,200) → sàn 0,20 thực tế **không cắt mất box nào** đã được postprocessor chấp nhận.
  - Cạnh ngắn box B1 (epsilon_005, quy về px native LR2): quantile 0/10/25/50/75/90% = 1,3/2,0/4,5/6,8/10,5/15,9.
  - Probe fidelity T4 so với A2r A100: B-Clean 54/2.880, B1 55/2.880 ô lệch (DRIFT, lệch giữa GPU) → kết quả sẽ là PENDING_REFERENCE_CHECK; so sánh trong cùng dump vẫn công bằng.
  - Thời gian T4 đo ở probe: 0,61 / 0,59 s/task (B-Clean / B1) → khoảng 23 phút/model cho 2.262 ảnh.
- **Đủ / thiếu:**
  - Đủ: ngưỡng score theo kích thước cho B1 trên LR2, mọi ngưỡng ≥ ngưỡng mask, 9 geometry; đánh giá bằng CandidateMatcher.
  - Thiếu: (1) ảnh **Clean** không được dump → không đo được tác động của luật lên Clean, nên chưa thể tuyên bố "1 model cải tiến cả hệ thống"; (2) box bị loại vì min width (5, hoặc 3 ở width_3) không được lưu → không thử min width 2; (3) không có tương ứng từng contour giữa các mức unclip → không thử unclip theo kích thước.
- **Làm:** viết `mixab/sizethr.py` (luật, lưới 1.024 tổ hợp, baseline 72 setting từ cùng dump, luật đạt khóa trước: +0,30 pp LR2 trên A, tiny ≥ base − 3 pp, FP không cao hơn ở cùng recall; bootstrap B) và `tests/test_sizethr.py` (4 test). 82/82 test PASS. Chạy thử trên dump probe thật (40 ảnh, 18 A): chạy thông; số 40 ảnh không có ý nghĩa.
- **Tiếp theo:** khi dump đầy đủ xong, chạy `python -m mixab.sizethr --dump "H:/My Drive/CS431_MixA/runs/dualagree_dump_v1_T4" --bundle ../../MONKEY_V2/docs/handoff/artifacts/pr_quad_audit_v1 --output results/sizethr_v1_T4` và `dualagree analyze`. Nếu ngưỡng kích thước đạt trên LR2, mới cần dump Clean của B1 (thêm option `--condition clean`, khoảng 23 phút T4) để kiểm Clean.

## 2026-11-22 · Kết quả dump T4: DualAgree v1 và ngưỡng theo kích thước (đều KHÔNG đạt luật khóa trước); biến thể "B1 + box chỉ B-Clean" dương nhỏ, nhất quán (khám phá)
- **Dump:** `runs/dualagree_dump_v1_T4/` xong, 36/36 shard mỗi model, 2.262 ảnh LR2; T4 0,68/0,72 s/task (B-Clean/B1), khoảng 26–27 phút/model.
- **Trung thực (T4 so với A2r A100):** B-Clean 5.953/162.864 ô lệch (3,66%), B1 6.896/162.864 (4,23%); ví dụ đều lệch ±1 TP hoặc ±1 box → PENDING_REFERENCE_CHECK. So sánh trong cùng dump vẫn công bằng.
- **DualAgree v1, lựa chọn khóa trước** (`runs/dualagree_analysis_v1_T4/decision.json`, Ô 7 Colab): t_l .7, t_c .7, t_a .5, θ .3. A: LR2 +0,68 pp so với B1 (77,71 vs 77,03), FP 4.070 vs 6.680, **tiny −19,8 pp** → trượt luật tiny; B: LR2 −0,41 pp (CI [−1,56; 0,84]), tiny −15,5 pp. `continue = false`. (Cổng nhóm mean +0,62 pp trên A nhưng vô nghĩa khi trượt tiny.)
- **Ngưỡng theo kích thước, B1 đơn** (`results/sizethr_v1_T4/size_decision.json`, chạy local): khóa epsilon_005, cut 3 px, t_small .7, t_large .5. A +0,62 pp nhưng tiny −7,1 pp (trượt); B −0,02 pp (CI [−0,17; 0,13]). Khám phá: nếu buộc tiny ≥ base − 3 pp thì chỉ +0,06 (A) / +0,06 (B). → **Không có cải tiến.**
- **Khám phá sau khi xem kết quả** (`scripts/dualagree_followup_bconly.py` → `results/dualagree_v1_T4_followup.json`): combo tốt nhất trên A trong số thỏa luật tiny = "B1 (t_l .5) + box chỉ B-Clean thấy (t_c .7), khử trùng IoU ≥ .5", không có nhánh đồng thuận.
  - A: LR2 +0,177 pp, CI95 [0,065; 0,304]; TP 27.876 vs 27.628; FP 6.910 vs 6.680; tiny 49,5 vs 48,8%.
  - B: LR2 +0,283 pp, CI95 [0,168; 0,407]; TP 29.092 vs 28.764; FP 5.703 vs 5.455; tiny 38,1 vs 37,6%.
  - Mean hệ thống (B-Clean cho Clean + biến thể cho LR2) so với model đơn tốt nhất: A +0,36 pp (< +0,5), B +1,19 pp; phần lớn là từ định tuyến (B1 đơn đã +0,27 trên A).
- **Chưa kiểm/rủi ro:** chọn biến thể sau khi xem kết quả (post-hoc) dù vẫn trong lưới và luật đã khóa; chưa đạt mức +0,3 pp trên A theo luật dừng; T4 lệch GPU; cần 2 checkpoint và 2× suy luận trên LR2; Tuấn chưa duyệt 2 checkpoint; B đã bị xem từ v1.
- **Tiếp theo:** user quyết định có dùng biến thể này làm kết quả (ghi rõ khám phá) hay không; nếu dùng, nên chạy lại analyze cố định đúng biến thể trên A100 (cùng GPU A2r) để bỏ PENDING_REFERENCE_CHECK.

## 2026-11-22 · Tuấn đồng ý 2 checkpoint (user chuyển lời); chỉ có T4 → kiểm lặp lại trên T4; viết đo latency
- **User báo:** (1) Tuấn OK hệ thống 2 checkpoint (lời user chuyển, chưa có văn bản gốc); (2) user chỉ có T4, không chạy lại A100 được; (3) đồng ý đo latency; (4) báo cáo để sau, user muốn xem lại chi tiết phương pháp khi chạy xong.
- **Làm:**
  - `scripts/t4_repeatability.py` → `results/t4_repeatability.json`: so dump probe (40 nguồn, lần chạy T4 thứ nhất) với dump đầy đủ (lần thứ hai) trên các nguồn chung: B-Clean 360/360, B1 360/360 mục (polygon + raw score, 9 geometry) **giống hệt**. → T4 tái lập tuyệt đối giữa các lần chạy; lệch chỉ là T4 so với A100 (±1 count, 3,7–4,2% ô).
  - `mixab/dualagree.py`: thêm `LOCKED` (biến thể khám phá đã chốt: B1 epsilon_005 ≥ .5 + B-Clean mask_040 ≥ .7, khử trùng IoU ≥ .5), `fuse_locked`, `summarize_ms`, lệnh `latency` (nạp cả 2 model, 200 ảnh LR2 cố định seed 20261122, 10 ảnh khởi động, đo B1/B-Clean/gộp/tổng, VRAM đỉnh, kiểm chéo số box sau gộp với dump cùng GPU).
  - Notebook thêm Ô 8 (chạy sau Ô 1–4). Zip mới: 39 file, SHA-256 `343342a00d82c75d6e55198e53a8ab5199799310d59396650e6fcb545d3253c6`.
  - Test 84/84 PASS (thêm 2 test: `fuse_locked`, `summarize_ms`). Lệnh `latency` (GPU) chưa chạy.
- **Chưa kiểm/rủi ro:** latency đo qua `forward()` của Tuấn (gồm kiểm SHA ảnh và một lần decode native) — cùng đường cho cả hai nhánh, nhưng không phải latency triển khai tối ưu.
- **Tiếp theo:** user tải zip mới, chạy Ô 1–4 rồi Ô 8 trên T4; Claude đọc `runs/dualagree_latency_v1_T4/latency.json`. Sau đó user xem lại chi tiết phương pháp; báo cáo viết sau.

## 2026-11-22 · Đặt tên CS-B1 + viết kế hoạch Phase 0 (chỉ tài liệu, chưa chạy phân tích mới)
- **Đã làm (kiểm tra, đọc):**
  - Fetch + ff MONKEY_V2 `96612b4 → 677b811` ("Publish research evidence and B0/latent weights; refresh current handoff"). Đọc `handoff.md`, `docs/research/current-workflow.md`, `fixed_diagnostics_report.md`, `diagnosis_summary.json`, `rev4/decision_review.py` (`fp_proxy`).
    - Latent 10k âm (A2 − B1 LR2 −0,037 pp, CI [−0,113; 0,050]).
    - Fusion job 106090 đang chạy.
    - Chẩn đoán `.40/.60` có proxy FP, GT diagnosis (kernel coverage P-map), 127 thẻ PENDING.
    - Tuấn ghi nhánh 2 checkpoint thuộc workflow cộng tác viên, phải báo đủ chi phí.
  - Trạng thái Drive: chưa có `runs/dualagree_latency_v1_T4/` → Ô 8 chưa chạy; zip trên Drive vẫn bản 15:53.
  - Đối chiếu `dump_identity.json` (T4) với `audit_identity.json` (A2r A100): trùng commit, postprocessor SHA, GT/bundle/merge manifest, config, torch/CUDA/cuDNN; chỉ khác GPU.
  - Mức tổng: LR2 Hmean B1 `epsilon_005/0.50` T4 so với A100: A 77,0322 vs 77,0320; B 76,6866 vs 76,6938.
  - Bối cảnh `gt_diagnosis` của Tuấn (`.40/.60`, B1, LR2, GT < 4 px, bỏ sót ổn định): weak_response 1.794, geometry 581, score_and_geometry 136, no_candidate 129, score_filtering 111.
- **Đã viết:** `docs/phase0_csb1_plan_20261122.md`.
  - Tên thống nhất **CS-B1 (Clean-Supplemented B1)**, cấu hình tham chiếu `CS-B1@0.7`.
  - Bảng dữ liệu có/thiếu, CPU/GPU.
  - Kiểm tra tham chiếu T4/A100 là bước 1.
  - Q1–Q5, chấm tay 120 thẻ (F/S/T × 40 + 10 hiệu chuẩn), lịch N1–N3, bảng quyết định A/B/C/D khóa trước.
  - Sửa chú thích trong `tests/test_dualagree.py`: box B1 dưới ngưỡng không tham gia ghép; logic không đổi, test PASS.
- **Đề xuất, chưa làm:** mọi bước Phase 0 trong kế hoạch; chạy Ô 8.
- **Rủi ro:** `CLAUDE.md` ghi hạn 24/11 trong khi user nói còn khoảng 3 ngày; kế hoạch có phương án rút gọn (80 thẻ).

## 2026-11-22 (chiều) · Cập nhật tiến độ đầu phiên mới
- **Làm:** `git fetch` MONKEY_V2: HEAD = origin/main = `677b811`, không commit mới (`?? scripts/` của Tuấn, bình thường). Đối chiếu workspace: không file nào mới hơn WORKLOG, không có `runs/` local. Cập nhật HANDOFF, `dashboard/status.js` (thêm bước CSB1, thêm log).
- **Bằng chứng:** `git log HEAD..origin/main` rỗng; `find -newer WORKLOG.md` rỗng.
- **Chưa kiểm/rủi ro:** không thấy được Drive/Colab từ phiên này nên không biết user đã chạy Ô 8 chưa; trạng thái job fusion 106090 chưa biết. Hạn 24/11 còn 2 ngày.
- **Tiếp theo:** user duyệt Phase 0 + chạy Ô 8 (T4); nếu không kịp thì chuẩn bị R2 với kết quả hiện có, ghi rõ PENDING_REFERENCE_CHECK.

## 2026-11-22 (tối) · Phase 0 CS-B1@0.7: phần CPU đã chạy, thẻ hiệu chuẩn và notebook Kaggle sẵn sàng
- **User yêu cầu:** duyệt Phase 0, chọn 80 thẻ (gồm 10 thẻ hiệu chuẩn chấm trước), chạy T4 trên Kaggle thay Colab. Làm phần CPU, chuẩn bị notebook Kaggle đo latency công bằng, gửi 10 thẻ hiệu chuẩn trước. Giữ nguyên CS-B1@0.7, cổng và bảng quyết định; không dùng B để chọn.
- **Làm:**
  - `mixab/phase0.py` (`refcheck`, `stats`, `cards`) và `tests/test_phase0.py` (12 test). Dùng lại nguyên `CandidateMatcher`, `maximum_matching`, `fp_proxy` và thứ tự chẩn đoán dựa trên box của Tuấn. Dump không có P-map nên không tính được `weak_response`/`invalid_kernel`; nhóm đó gộp vào `no_box_in_dump`.
  - Chép dump T4 và counts A2r từ Drive về `runtime/phase0_inputs/` (52 MB, không vào Git).
  - Kết quả `refcheck` → `results/phase0/reference_check.json`:
    - B-Clean 5.953 ô lệch (3,66%), B1 6.896 (4,23%); trường `gt` không bao giờ lệch; lệch `tp` chủ yếu ±1.
    - Ở 2 setting CS-B1 dùng, Hmean lệch ≤ 0,009 pp.
    - ΔCS-B1 trên A sau khi bỏ 35 nguồn lệch: +0,127 [0,052; 0,204]; bỏ 240 nguồn (cách chặt hơn): +0,079 [−0,004; 0,168].
    - Nguồn lệch có box sát ngưỡng (median khoảng cách 0,0009 so với 0,02–0,03).
  - Kết quả `stats` → `results/phase0/auto_stats.json`, trên A:
    - Q1: FP của box bổ sung 79% là "chồng GT, định vị hoặc phủ một phần".
    - Q2: 1.305 GT tiny bỏ sót; (a) 55,6%, (b)+(c) 44,4%.
    - Q3: 537 box thêm → +248 TP / +230 FP, 0 GT bị mất; top 10 nguồn chiếm 66,6% lợi ích ròng; bỏ top 10 thì Δ còn +0,066 pp.
    - Q4: cả 527 GT tiny DualAgree v1 làm mất đều do box B1 score 0,5–0,7 bị cắt.
    - Q5: 44,4% theo quy tắc khóa; mô tả thêm: 69% GT được cứu có box B1 `mask_040` khớp.
    - Độ nhạy: θ 0,7 → −0,32 pp (A).
  - Kiểm thêm từ counts A2r: trên A, B1 `epsilon_005/0.50` đã là tốt nhất trong 72 setting (77,032). Trên B, B1 `mask_040/0.60` (77,007) nhỉnh hơn CS-B1 (76,970); chỉ mô tả.
  - Viết `docs/phase0_csb1_auto_results_20261122.md`.
  - Thẻ `results/phase0/review/`: seed 20261123, lựa chọn khóa ở `selection.json`.
    - 70 thẻ chính trên A: F 16 (proxy × kích thước, chia đều), S 24 (TP 12 + FP 12, kích thước × score), T 30 (tối thiểu 4 mỗi chẩn đoán, phần còn lại theo tỉ lệ).
    - 10 thẻ hiệu chuẩn trên B: F 3, S 3, T 4.
    - Phiếu chấm **mù** (không có stratum/proxy); giải mã ở `card_decode.json`. Kèm `HUONG_DAN_CHAM.md`.
    - **Khác kế hoạch §5/§6:** phương án 80 thẻ của kế hoạch bỏ nhóm F, nhưng user yêu cầu giữ FP nên F vẫn có 16 thẻ.
  - `mixab/csb1_latency.py` (`prepare`, `run`):
    - Đo B1 và CS-B1 xen kẽ trên cùng ảnh, warm-up 20, 200 ảnh × 2 lượt, sync trước và sau mỗi lần đo.
    - Đo riêng bộ nhớ đỉnh B1 (chỉ nạp B1) và CS-B1 (nạp cả 2 model).
    - Giải mã một lần, có kiểm tương đương: B-Clean tự giải mã cho cùng box.
    - Hậu xử lý triển khai thật: `min_text_score` đặt trong DBPostprocessor.
    - So số box với dump T4; chỉ dùng 1 GPU.
  - `notebooks/kaggle_csb1_latency.ipynb` (builder `scripts/build_kaggle_notebook.py`, nbformat hợp lệ).
  - `scripts/make_kaggle_dataset.py` → `runtime/kaggle_upload/cs431-csb1-latency.zip` (398 MiB, SHA `b6ceffa7c4f8d6507ef291625b8bce786b28eff5bb7f351f435ed1b32e8b3bf9`).
  - Viết `docs/HUONG_DAN_KAGGLE_csb1_latency.md`.
- **Bằng chứng:** test 96/96 PASS; log `runtime/scratch/stats.log`, `runtime/scratch/refcheck.log`; code trong dataset trùng `mixab/` (diff rỗng).
- **Chưa kiểm/rủi ro:**
  - Notebook Kaggle **chưa chạy**. Python của Kaggle có thể khác 3.13: khi đó phải build MMCV (20–40 phút, ước lượng). Driver Kaggle phải chạy được torch cu128.
  - Ảnh LR2 trên thẻ được tạo lại bằng Pillow 12.2, chỉ để hiển thị.
  - Ghép cặp TP từng box dùng một maximum matching: counts chính xác, nhưng cách ghép cặp không duy nhất.
  - Bảng quyết định không quy định thứ tự khi B và D cùng đạt.
- **Tiếp theo:** user chạy Kaggle và chấm K01–K10; Claude đối chiếu rồi gửi C01–C70; tổng hợp; soạn kết luận Phase 0 cho user duyệt.

## 2026-11-22 (khuya) · Tư vấn: phản biện đề xuất "Hướng 2A unclip/min_width theo kích thước" của Codex (chỉ đối chiếu, không đổi kế hoạch)
- **User hỏi:** nếu dừng CS-B1, có nên ưu tiên phần chưa thử của Hướng 2A (unclip và min_width theo kích thước, chỉ B1, không train) không. Yêu cầu chỉ tư vấn: không sửa code dự án, không tạo notebook, không chạy GPU.
- **Làm (phân tích CPU chỉ đọc, script nháp `runtime/scratch/advice_reach.py`, `advice_check2.py`, kết quả `runtime/scratch/advice_reach.json`; không vào code dự án):**
  - Trần đạt được từng GT: với 9.795 GT là FN của B1 `epsilon_005/0.50` trên A, đếm GT có box của từng geometry B1 khớp (IoU ≥ 0,5, score ≥ 0,5): `mask_040` 1.126 (trong đó 178 tiny) · `unclip_1p2` 208 · `unclip_1p8` 180 · `base`/`width_3`/`width_7` đều 81 (min_width 3/7 **không thêm GT nào** so với `base` ở score 0,5) · có ở ít nhất một geometry: 1.417, chỉ ở `mask_040`: 926.
  - Cơ chế của 248 GT được CS-B1 cứu: 212 (85%) nằm trong một box B1 `epsilon_005` phủ ≥ 2 GT (93 có IoU ≥ 0,3 + 119 có IoU < 0,3 nhưng bị box lớn nuốt); 18 không có box nào phủ; 18 khác → cơ chế chính là **gộp chữ ở mask 0,3**, `mask_040` tách được. Không phải unclip hay min_width.
  - Điểm toàn cục trên A (counts A2r): `epsilon_005/0.50` 77,032 · `base` 76,804 · `width_7` 76,825 · `width_3` 76,803 · `unclip_1p2` 76,349 · `unclip_1p8` 75,850 · `mask_040/0.60` 76,791.
  - **Đính chính ghi chú cũ** (WORKLOG mục sizethr; HANDOFF "không thử được unclip theo box"): giữa `base` và `unclip_1p2`/`unclip_1p8`/`width_3`/`width_7`, raw score khớp 1-1 và duy nhất (vd. 49.142/49.153 box `unclip_1p2`). Vậy **chọn unclip ∈ {1,2; 1,5; 1,8} và min_width ∈ {3, 5, 7} theo từng box làm được trên CPU**, nhưng chỉ quanh geometry `base` (epsilon 0,01), không quanh `epsilon_005`. Min_width < 3 và unclip liên tục vẫn cần GPU.
- **Kết luận tư vấn (chưa thành quyết định):** không nên ưu tiên unclip/min_width. Nên hoàn tất Phase 0 và báo cáo. Nếu còn thời gian, chỉ cân nhắc một kiểm trần CPU cho "tách box gộp cục bộ bằng mask 0,4" (một model), phải được duyệt trước.
- **Chưa kiểm/rủi ro:** trần đạt được từng GT không phải lợi ích ròng; mức +0,35 pp là cận trên thô (2·127/71.739), chưa phải oracle chính xác; số đếm "gộp" dùng ngưỡng phủ 50%.
- **Tiếp theo:** chờ user quyết định; Phase 0 vẫn chờ Kaggle và chấm K01–K10.

## 2026-11-22 (khuya) · Pilot CPU MSB1 (Mask-Split B1): kết quả âm theo luật khóa
- **User yêu cầu:** thử "tách box gộp bằng box B1 mask 0,4" trên CPU, song song với hoàn tất Phase 0. Viết spec trước khi chạy; luật không dùng GT; kiểm khác biệt epsilon; oracle phải là cận hợp lệ; không chỉnh ngưỡng hay cổng sau khi xem số; B chỉ mô tả.
- **Làm:**
  - `docs/msb1_spec_20261122.md` (khóa trước khi chạy).
  - `mixab/msb1.py` (`run`, `examples`) và `tests/test_msb1.py` (4 test: gán con, tách, kept set, Dinkelbach trùng brute force).
  - Đọc `installed_db_postprocessor.py`: epsilon ảnh hưởng polygon xấp xỉ → score, mức nở khi unclip, quad, và việc bỏ contour < 4 đỉnh. Vì vậy `epsilon_005` → `mask_040` là thay **hai** cấu hình hậu xử lý. Thêm chẩn đoán với cha là `base`, chỉ khác mask.
  - Chạy `python -m mixab.msb1 run` → `results/msb1_v1_T4/msb1_result.json`, khoảng 3 phút. Chạy `examples` → 8 ca theo hạng SHA + 1 hình toàn cảnh.
  - `docs/msb1_result_20261122.md`.
- **Bằng chứng (A, dump T4):**
  - Oracle +1,67 pp (1.117 nguồn chính xác, 13 dùng cận an toàn).
  - MSB1 77,257 so với B1 77,032 → **+0,224 pp, CI [−0,40; +0,81]**.
  - Tiny 48,78 → 54,08%; TP +641, FP +810; tách 1.592 cha thành 3.434 con ở 422 nguồn; GT cứu 757, mất 116.
  - Cổng: LR2 không đạt; tiny đạt; FP ở cùng recall đạt theo định nghĩa (không setting B1 nào đạt recall 75,5%).
  - Chẩn đoán cha `base`: +0,275 pp. B: +0,69 [0,28; 1,13] (mô tả).
  - Nguồn `6e340fcbb1e0a877`: 10 GT, B1 1.796 box; MSB1 thêm +292 box và +2 TP. Bỏ riêng nguồn này thì +0,51 pp (mô tả, `runtime/scratch/msb1_drop1.py`).
  - Test 100/100 PASS.
- **Chưa kiểm/rủi ro:**
  - Hiệu ứng tách riêng từng box bỏ qua tương tác.
  - Mức độ ảnh chữ dày thiếu nhãn ảnh hưởng tới các so sánh khác (CS-B1, 72 setting) chưa kiểm.
  - Ý tưởng hình thành sau khi xem A.
- **Tiếp theo:** ghi MSB1 là kết quả âm có tín hiệu vào báo cáo; không chạy GPU Clean. Phase 0 CS-B1 vẫn chờ Kaggle và chấm K01–K10. Có thể báo Tuấn về độ nhạy với ảnh chữ dày thiếu nhãn (chưa gửi).

## 2026-11-22 (khuya) · Chuẩn bị pilot ZB1 (zoom cục bộ bằng B1) cho Kaggle — chưa chạy
- **User yêu cầu:** đánh giá và lập pilot Hướng 3 (zoom cục bộ, một checkpoint B1, không train). Từ nay local chỉ đọc, sửa code, chuẩn bị notebook, kiểm nhẹ; phần nặng chạy trên Kaggle, do user chạy. Gom bước kiểm ảnh trang báo vào một bước ngắn trên Kaggle. Giữ nguyên kết luận MSB1.
- **Làm (chỉ đọc và kiểm nhẹ):**
  - Đọc `inference_config.py:199-204`: Resize đưa cạnh ngắn về 1024 → Clean và LR2 cùng kích thước tensor. Bicubic ×2 toàn ảnh của Tuấn 75,24 so với 75,29 (`handoff.md:74`), xác nhận đó là phép tương đương.
  - `grep` repo: không có tile/zoom cho detector; `rev4/edsr.py` chỉ chia tile để chạy EDSR.
  - Recall theo bin từ counts A2r (A, B1 `epsilon_005/0.50`): Clean 61,2/80,6/84,3/75,3%; LR2 48,8/72,6/80,4/73,5% (<4/4–8/8–16/≥16).
  - Viết `docs/zoom_pilot_spec_20261122.md` (khóa trước), `mixab/zoom.py`, `tests/test_zoom.py` (7 test), `scripts/build_kaggle_zoom_notebook.py` → `notebooks/kaggle_zoom_pilot.ipynb` (nbformat hợp lệ, các ô parse được), `scripts/make_kaggle_zoom_dataset.py` → `runtime/kaggle_upload/cs431-zoom-pilot.zip` (336 MiB, SHA `92db0747462da984a3d1c850ce94c51a32583c68d2f625f910f7ed83e455416d`; chỉ chép file), `docs/HUONG_DAN_KAGGLE_zoom_pilot.md`.
- **Bằng chứng:** test 107/107 PASS; code trong dataset trùng `mixab/`; mẫu có 300 nguồn A, trong đó có nguồn chữ dày `6e340fcbb1e0a877` (đúng theo quy tắc hash, giữ nguyên).
- **Chưa kiểm/rủi ro:**
  - Notebook chưa chạy. Việc đổi `Resize.scale` để chạy S2 và đường crop qua PNG mới chỉ được kiểm bằng logic. Ô 5 sẽ kiểm tọa độ 1×1 và OOM.
  - Thời gian chưa đo.
  - `min_width` tính bằng px tensor, nên zoom đồng thời nới min_width tương đối; đã ghi trong spec.
  - Nguy cơ FP trên chữ không có nhãn, như đã thấy ở MSB1.
- **Tiếp theo:** user chạy notebook trên Kaggle; Claude đọc kết quả và áp cổng G2 đã khóa.

## 2026-11-23 · Đọc kết quả Kaggle: pilot ZB1 zoom (âm, dừng nhánh) + latency CS-B1 (1,99×)
- **Làm:** dùng Kaggle API (`python -m kaggle` 2.2.4, token có sẵn trên máy) liệt kê notebook của user và tải output của `ojanek/notebook5b3e8ce4a1` (zoom) và `ojanek/notebookef1d03734c` (latency) vào `runtime/kaggle_downloads/` (thư mục riêng; chỉ đọc JSON). Chép bản chính vào `results/zoom_pilot_v1_T4/` và `results/csb1_latency_v1_T4/`. Tính nhẹ phần mô tả "bỏ ảnh trang báo" (chấm lại 2 task, `runtime/scratch/zoom_dense_desc.py`). Viết `docs/zoom_pilot_result_20261123.md`.
- **Bằng chứng:**
  - Code chạy đúng bản đóng gói: manifest SHA `581e5660…`.
  - Zoom: kiểm tọa độ 6/6; 600/600 task; S2 OOM 6/600.
    - G2: LR2 70,47 → 67,47 (−3,01 [−6,39; +0,66]), Clean 72,39 → 68,74 (−3,65); tiny LR2 +10,8 pp.
    - G3 −2,69 / −4,25; S2 −2,04 / −3,97; R3 −2,31 / −2,68.
    - Pagecheck: 10 GT, 0 ignore, ROI 100%, `assert_official` PASS; top 1 = 26,8%, top 10 = 48,3% FP của B1 trên A.
  - Latency: B1 447 ms, CS-B1 889 ms, tỉ số 1,99× [1,98; 2,02]; bộ nhớ 894/991 MiB; box 200/200 khớp dump Colab.
- **Chưa kiểm/rủi ro:** pilot là mẫu 300 nguồn, chỉ sàng lọc; thời gian zoom đo tuần tự. Không phân biệt được FP do nền hay do chữ thiếu nhãn khi chưa chấm tay.
- **Tiếp theo:** user chấm K01–K10; quyết định có báo Tuấn về ảnh thiếu nhãn không. Nhánh zoom dừng.

## 2026-11-23 · Đồng bộ repo Tuấn + viết roadmap toàn bộ
- **Làm:** `git fetch` + fast-forward MONKEY_V2 `677b811 → b9eaf37` (727dcc2 recovery fusion dùng node-local scratch; 930001b ghi job 111683 + audit trước khi submit; b9eaf37 biên nhận GPU probe / checkpoint import). Đọc `handoff.md`, `docs/handoff/artifacts/fusion_recovery_111683/README.md`, `docs/lr2-restoration-proposal.md`, `rev4/edsr.py`. Đối chiếu với bản tóm tắt HPC user dán. Viết `docs/ROADMAP_toan_bo_20261123.md`.
- **Bằng chứng:**
  - Repo: README recovery ghi "This is not a final performance report", kết quả nằm ở `.runtime` trên HPC → số fusion chỉ có từ bản tóm tắt HPC.
  - B0/B-Clean/B1 và các thử PRC/KD/EDSR/CRAFT/adapter lấy từ `handoff.md` §4–5.
  - EDSR dùng `torchSR v1.0.3 edsr64_x2.pt` (`rev4/restoration.py:18`).
- **Chưa kiểm/rủi ro:** số Giai đoạn 1 và fusion chưa có trong repo; chưa đọc Q-LRG; tiêu chí "cải tiến" của thầy chưa rõ.
- **Tiếp theo:** chờ user quyết định đề xuất 1–3 trong roadmap §6.

## 2026-11-23 · Rà soát CS-B1 trên đủ 2.262 nguồn + sửa roadmap + chuẩn bị bổ sung Clean
- **User yêu cầu:** sửa các kết luận quá mạnh trong roadmap; xác minh tập nguồn, cấu hình và baseline của CS-B1; tổng hợp đủ tập bằng tổng counts; nếu thiếu dữ liệu thì chuẩn bị lượt chạy bổ sung tối thiểu; không chạy EDSR chỉ để có kết quả dương; không đề xuất TEST để cứu phương án; soạn tin cho Tuấn; không gửi, không push.
- **Làm:**
  - Đọc AGENTS.md: chỉ có ở `Chọn monkey hay paddle/CS431` (workspace Paddle), không áp dụng trực tiếp.
  - Đối chiếu số của Tuấn với `handoff.md:54-74` (khớp).
  - `scripts/csb1_full_verify.py` → `results/csb1_full_v1_T4/csb1_full.json` (CPU nhẹ, dùng records và counts đã có).
  - `runtime/scratch/system_mean.py` → `results/csb1_full_v1_T4/system_mean_descriptive.json`.
  - Thêm `--condition` cho `dualagree dump`/`fidelity`; viết `mixab/csb1_clean.py` + `tests/test_csb1_clean.py`.
  - Viết spec `docs/csb1_clean_check_spec_20261123.md`, notebook `notebooks/kaggle_csb1_clean.ipynb` (builder `scripts/build_kaggle_csb1_clean_notebook.py`), dataset `runtime/kaggle_upload/cs431-csb1-clean.zip` (339 MiB, SHA `b7b79b456f0ebfd1ea02f9e4f06080e0d165e0837d105055095d726bf021ee48`), hướng dẫn `docs/HUONG_DAN_KAGGLE_csb1_clean.md`.
  - Viết lại `docs/ROADMAP_toan_bo_20261123.md`; viết tin dự thảo `docs/msg_tuan_20261123_csb1.md`.
- **Bằng chứng:**
  - Cohort 2.262 = records = A2r (A 1.130 / B 1.132).
  - B1 `epsilon_005/.50` từ records trùng counts tính lại của lưới 72.
  - B1 `.40/.60` A100 = 76,9008%, FP 11.999.
  - CS-B1 ALL: 77,087 so với 76,856 (`epsilon_005`) → +0,231 [0,151; 0,322]; so với 76,907 (`.40/.60`, T4) → +0,180 [−0,211; +0,582]. A +0,409 [−0,162; +1,055]; B −0,039 [−0,513; +0,468].
  - `calibration_sheet.csv` trống.
  - Test 108/108 PASS.
- **Chưa kiểm/rủi ro:**
  - Clean chưa đo; notebook bổ sung chưa chạy (tải 7 GB trên Kaggle).
  - Các số `.40/.60` của CS-B1 là T4 so với T4 (cùng dump), còn số Tuấn công bố là H200.
  - Drive H: không mount.
- **Tiếp theo:** user duyệt tin cho Tuấn, chấm K01–K10, tùy chọn chạy notebook Clean.

## 2026-11-23 · Dựng lại dataset bổ sung Clean khi Drive H: đã mount
- **Làm:** Drive H: đọc được lại. Đưa 2.262 ảnh Clean từ `cache/textvqa_val2262_clean.zip` vào dataset (chép đúng byte, kiểm SHA từng ảnh khi đóng gói), để Kaggle không phải tải 7 GB. Notebook ưu tiên `DS/clean`, chỉ tải zip TextVQA chính thức khi thiếu. Dựng lại `notebooks/kaggle_csb1_clean.ipynb` và dataset.
- **Bằng chứng:** `runtime/kaggle_upload/cs431-csb1-clean.zip` 939 MiB, SHA `337f4b06991f64ebb09875099f3c8c099a908c08067e4b2b68323955b1450eb0` (thay bản 339 MiB `b7b79b45…`); 2.262 ảnh; `clean_images_included = true`; code trong dataset trùng `mixab/`; nbformat hợp lệ; test 108/108 PASS.
- **Chưa kiểm/rủi ro:** notebook chưa chạy trên Kaggle.
- **Tiếp theo:** như trước (duyệt tin cho Tuấn, chấm K01–K10, tùy chọn chạy notebook Clean).

## 2026-11-23 · Chốt baseline của phép so sánh CS-B1 (chưa chạy thêm)
- **Làm:** đọc `pr_quad_audit_v1/results/decision_audit_report.md` và `baseline_settings.json`; so counts H200 gốc (`counts/B1_source_counts.npz`), A2r A100 và dump T4 tại `.40/.60`, `epsilon_005/.50`, `.30/.40` (`runtime/scratch/baseline_check.py` → `results/csb1_full_v1_T4/baseline_check.json`, đọc nhẹ). Sửa roadmap và tin nhắn cho Tuấn.
- **Bằng chứng:**
  - Báo cáo của Tuấn: luật chọn = tối đa mean không trọng số Clean + LR2 trên lưới cho từng model; B1 chọn `mask_040/score_0.60`; Δ so với "original calibrated `.30/.40`" là LR2 +1,610 [1,006; 2,301].
  - H200: `.40/.60` TP 56.360, Pred 68.357, FP 11.997, H 76,901; `.30/.40` H 75,291, FP 16.487.
  - A100: FP 11.999; T4: FP 11.988; 32 nguồn lệch ±1 giữa H200 và A100.
  - Luật của Tuấn trên H200 đủ tập chọn `.40/.60` (mean 78,166); LR2 tốt nhất cũng là `.40/.60`.
  - `epsilon_005/.50` trên đủ tập: 76,859 (H200).
- **Chưa kiểm/rủi ro:** chưa có per-source counts của fusion/adapter/EDSR nên chưa so ghép cặp được với CS-B1.
- **Tiếp theo:** user chốt phép so sánh và duyệt tin, rồi mới quyết định chạy bổ sung Clean.

## 2026-11-23 · Bàn giao sang session mới + đề xuất hướng thứ hai (chưa triển khai)
- **User:** sẽ tự chạy notebook bổ sung Clean; yêu cầu cập nhật tiến độ, chuẩn bị bàn giao, tóm tắt các hướng đã thử, đề xuất 2–3 hướng mới (chọn một ưu tiên, kèm pilot và tiêu chí khóa trước), kiểm còn dữ liệu chưa dùng không. Chưa triển khai; không mở lại zoom; EDSR không phải phương pháp riêng của Trí.
- **Làm (chỉ đọc):**
  - Kiểm protocol: validation 3.262 = S1 500 + S2 500 + remaining 2.262 (`CS431_Pilot_Protocol_v2.1_FINAL.md:14,43`) → không còn tập chưa dùng.
  - Đọc train pipeline trong `inference_config.py:281-312` (`Fliplr 0.5`, `Affine ±10°`, `Resize 0.5–3.0`, `RandomCrop`, 640×640); chưa kiểm recipe thật của B1.
  - Lấy SHA `code/CODE_MANIFEST.json` của gói Clean (`f7f5f699…`).
  - Viết `docs/SESSION_START_20261123.md`; cập nhật roadmap §5, HANDOFF, dashboard.
- **Bằng chứng cho đề xuất XV-B1:** recall theo bin trên A ở `epsilon_005/.50` (counts A2r): B1 / B-Clean LR2 8–16 px 80,4 / 78,3%; ≥ 16 px 73,5 / 73,0%; FP LR2 `.40/.60` B1 11.997 so với B-Clean 6.578 (`handoff.md` §4).
- **Chưa kiểm/rủi ro:** XV-B1 chưa có số nào; FP trên chữ thiếu nhãn có thể không lọc được; cùng họ 2 checkpoint với CS-B1.
- **Tiếp theo:** session mới tiếp nhận output Clean (nếu có) và để user chọn hướng.

## 2026-11-23 (session tiếp) · Kiểm tiếp nhận Clean + trình bày lại hướng thứ hai
- **Làm (chỉ đọc):** đọc SESSION_START, CLAUDE.md, HANDOFF, ROADMAP; `kaggle kernels list --mine`; `git fetch` MONKEY_V2.
- **Bằng chứng:** kernel mới nhất là `ojanek/notebookef1d03734c` (22/11 18:59, latency, đã tiếp nhận); không có lượt chạy nào của `kaggle_csb1_clean` → Việc 1 chưa có output để tiếp nhận. `git log main..origin/main` rỗng → repo Tuấn không có commit mới.
- **Chưa kiểm/rủi ro:** kết quả Clean của CS-B1 vẫn trống; K01–K10 và duyệt tin Tuấn vẫn chờ user.
- **Tiếp theo:** user chọn hướng ở SESSION_START §4 (hoặc báo đã chạy notebook Clean); chưa triển khai gì.

## 2026-11-23 (session tiếp, phần 2) · XV-B1 pilot CPU (dừng) + chuẩn bị Flip-TTA
- **User:** (1) đính chính 3 điểm: "B-Clean không thấy ≠ B1 sai", "không thêm latency chỉ đúng khi đã chạy cả hai model", "trượt cổng ≠ âm"; yêu cầu đối chiếu XV-B1 với DualAgree v1/QRoute/CS-B1, nếu cùng họ thì ghi là ablation; chạy CPU trên A, không B, không ghép CS-B1. (2) Muốn xem Flip-TTA: kiểm P-map, lịch sử flip, đã thử chưa; chuẩn bị notebook, chưa chạy GPU.
- **Làm:**
  - Đối chiếu: họ `B1+agree` của DualAgree v1 trên A (`runtime/phase0_inputs/decision.json`): Hmean .7755 so với .7703, FP 6.680 → 3.920, tiny 48,8 → 28,4%. XV-B1 = cùng cơ chế + miễn trừ < 8 px + bằng chứng lỏng + không phụ thuộc score → **ablation cùng họ**, ghi rõ trong spec.
  - XV-B1: spec `docs/xvb1_spec_20261123.md`, `mixab/xvb1.py`, 6 test; chạy một lần trên A (1.130 nguồn, ~45 s CPU) → `results/xvb1_v1_T4/` + `docs/xvb1_result_20261123.md`. Baseline epsilon_005/.50 tái tạo khớp 72-setting.
  - Flip-TTA: spec, `mixab/fliptta.py`, 10 test, builder notebook + dataset (337 MiB), hướng dẫn. Dry-run `evaluate` trên bản ghi giả (xóa sau đó).
- **Bằng chứng:** XV-B1 A: Hmean 76,95 so với 76,80 (`.40/.60`), +0,153 pp CI [−0,459; 0,799]; so với cha −0,079 CI [−0,140; −0,019]; tiny 48,78 so với 52,11 (−3,33); FP 6.549 so với 6.815; TP bị xóa 128, FP bị xóa 131; 356 box/217 nguồn. Test 124/124 PASS. Dataset Flip-TTA SHA zip `81e6b94d…`.
- **Chưa kiểm/rủi ro:** 97/356 box bị xóa không đổi Pred (suy luận: vùng bỏ qua của evaluator, chưa kiểm); nguyên nhân 128 TP mất (chữ B1 học thêm hay box gộp IoU < 0,3) chưa tách. Flip-TTA: đường `infer` GPU chưa chạy (không có mmocr local); B1 train với flip chỉ theo cấu hình repo; mọi số T4 là PENDING_REFERENCE_CHECK. Chi phí 2 forward chưa đo.
- **Tiếp theo:** user chạy Clean CS-B1 rồi Flip-TTA (dừng sau Ô 4 nếu căn chỉnh sai/dự kiến > 3 giờ); Claude tiếp nhận, báo nhãn khóa, không chỉnh.

## 2026-11-23 (tối) · Tiếp nhận kết quả Clean CS-B1 (Kaggle T4)
- **User:** đã chạy xong notebook Clean, nhờ tiếp nhận.
- **Làm:** Kaggle API chưa liệt kê lượt chạy mới (kernel mới nhất vẫn 22/11); tìm thấy zip trong Downloads (`cs431_csb1_clean_T4_20261123-110923.zip`, 23/11 19:09, SHA `c529f1e5…`) + log `cs431-csb1-cleann.log` (lượt commit 16:33, chỉ ~420 s, không phải lượt tạo zip). Giải nén vào `runtime/kaggle_downloads/csb1_clean_T4_20261123-110923/`; kiểm theo SESSION_START §2; tính lại `mixab.csb1_clean` ở local; chép bản chính `results/csb1_clean_v1_T4/` + `intake_receipt.json`; viết `docs/csb1_clean_result_20261123.md`.
- **Bằng chứng:** manifest `f7f5f699…` khớp; env đạt; 36 shard × 2 model, 2.262 ảnh, `sources_sha256` trùng dump LR2; JSON tính lại IDENTICAL. Clean ALL: CS-B1 79,82 / B1 .40/.60 79,44 / B-Clean 80,84; Δ so với B1 .40/.60 +0,39 [0,01; 0,81], so với B-Clean −1,02 [−1,76; −0,49]; mean không định tuyến +0,28 (ALL), −0,18 (A).
- **Chưa kiểm/rủi ro:** T4 → PENDING_REFERENCE_CHECK (probe lệch A2r ≈ 2,9% ô); dataset trong log nằm dưới tài khoản `hqnhuu` nhưng SHA 2.385 file khớp. Kết quả là development, CS-B1 chọn sau khi xem A.
- **Tiếp theo:** cập nhật tin nhắn cho Tuấn với số Clean (chưa gửi); user chạy Flip-TTA; K01–K10.

## 2026-11-23 (tối, phần 2) · Tiếp nhận pilot Flip-TTA (Kaggle T4)
- **User:** đã chạy xong Flip-TTA, nhờ xem.
- **Làm:** `kaggle kernels output ojanek/cs431-fliptta-pilot-1` → `runtime/kaggle_downloads/fliptta_pilot_1/` (zip `cs431_fliptta_T4_20261123-130454.zip`, SHA `d24f11e3…`). Log trong Downloads là lượt Save Version dừng ở Ô 1 (chưa gắn dataset), không liên quan. Kiểm theo spec; tính lại `mixab.fliptta evaluate` ở local; chép `results/fliptta_v1_T4/` + `intake_receipt.json`; viết `docs/fliptta_result_20261123.md`.
- **Bằng chứng:** manifest `bc0835c6…`; cổng căn chỉnh đạt (600/600 kiểm cứng; IoU 0,847 so với 0,030); F LR2 khớp dump 300/300; JSON + CSV tính lại IDENTICAL. LR2 +0,27 [−0,16; 0,66], Clean +0,21 [−0,22; 0,68], FP −92/−99, TP −22/−36, tiny LR2 −1,18; latency 1,95×. Nhãn `INCONCLUSIVE`.
- **Chưa kiểm/rủi ro:** T4 → PENDING_REFERENCE_CHECK; n = 300 nên CI rộng; cơ chế "làm mềm đỉnh" chỉ là suy luận; không có ảnh trang báo trong mẫu nên không biết tác động trên ảnh chữ dày.
- **Tiếp theo:** user quyết định bước sau cho Flip-TTA (không tự mở rộng); cập nhật tin cho Tuấn; K01–K10.

## 2026-11-23 (khuya) · Chẩn đoán Flip-TTA trên artifact có sẵn (CPU)
- **User:** muốn cải tiến Flip-TTA có cơ sở; phân tích nguyên nhân trước, tối đa một biến thể, kế hoạch latency, spec trước khi chạy; được đề xuất dừng.
- **Làm:** đọc tài liệu; AGENTS.md tìm thấy thuộc repo Paddle/Monkey H200 khác (không áp). Viết `mixab/fliptta_diag.py` (+3 test), `scripts/fliptta_matched_control.py`; chạy trên 600 bản ghi pilot → `results/fliptta_diag_v1/` (`fliptta_diag.json`, `matched_control.json`, `event_margins.json`, `events_*.json.gz`, 12 hình). Tra abstract Shanmugam et al. ICCV 2021 (arXiv:2011.11156). Viết `docs/fliptta_diagnosis_plan_20261123.md`.
- **Bằng chứng:** evaluator chính thức khớp 1.200/1.200; pool counts = per_source.csv. T so với F nâng ngưỡng: Clean +0,004 pp (cùng FP), LR2 +0,19 [−0,23; 0,58] (cùng FP) / +0,27 [−0,15; 0,66] (cùng recall). GT mất/thêm LR2 171/149 (score 57/29, hình học 98/110); FP bị loại 288, mới 141; Δ raw cặp trung vị −0,0036. Cận trên +0,40/+0,31 so với T0. Test 127/127 PASS.
- **Chưa kiểm/rủi ro:** không có P-map, nhánh lật riêng, thời gian công đoạn, batch 2 → cơ chế nhánh lật chỉ là giả thuyết. Đối chứng nâng ngưỡng là hậu kiểm. Mẫu 300 đã dùng để chẩn đoán. LR2 trong hình là bicubic tạo lại (Pillow local ≠ 11.3.0).
- **Tiếp theo:** user chọn dừng (khuyến nghị) hoặc duyệt lượt chẩn đoán P-map.


## 2026-11-24 — Codex: kế hoạch learned box-quality scoring
- Làm: đọc code B1/postprocessor, roadmap và diagnosis; fetch repo Tuấn (không commit mới); chuẩn bị kế hoạch và source membership, cập nhật handoff/dashboard. Không sửa repo Tuấn hoặc chạy GPU.
- Bằng chứng: train_ids SHA 8ac5cbd9… khớp receipt; 15.222 unique TRAIN disjoint 2.262 dev. `results/boxquality_preparation_20261124/receipt.json` và `docs/KE_HOACH_BOX_QUALITY_20261124.md`. Paper chỉ dùng mức abstract.
- Chưa kiểm/rủi ro: TRAIN pixels/GT chưa có bằng chứng đủ; H không mount; không có improvement hay timing. Scorer là giả thuyết, vẫn có rủi ro thiếu nhãn và overfit; deadline hiện tại 24/11.
- Tiếp: xác minh dữ liệu, label quality, feature feasibility rồi chuẩn bị notebook smoke; user chạy GPU. Không đổi cổng, không tự mở TEST.
- Kiểm tra cuối: suite hiện có 127/127 PASS (17,760 s); chỉ regression tests, không phải bằng chứng phương pháp mới có hiệu quả. Đã đọc lại tài liệu kế hoạch và trạng thái mới.
