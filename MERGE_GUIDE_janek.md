# Ghép phần Janek vào MONKEY_V2

Ngày 24/11/2026. Đây là sắp xếp thư mục và tên file; không phải kết quả thí nghiệm mới.

## Cấu trúc chung

| Đường dẫn | Vai trò |
|---|---|
| `mixab/*_janek.py` | Phần code của Janek, gọi pipeline chung `rev4` / `lowres` của Tuấn |
| `scripts/*_janek.py` | Chuẩn bị dữ liệu, phân tích, tạo notebook và gói upload |
| `tests/test_*_janek.py` | Kiểm thử CPU của Janek |
| `notebooks/*_janek.ipynb` | Notebook Colab/Kaggle |
| `configs/*_janek.json` | Cấu hình của Janek |
| `docs/research/janek/` | Kế hoạch, spec, báo cáo, hướng dẫn và WORKLOG |
| `docs/handoff/artifacts/janek/` | Bản công bố kết quả, số đếm, identity và hình |
| `docs/archive/janek/` | Tài liệu cũ |
| `tools/dashboard_janek/index_janek.html` | Dashboard |
| `HANDOFF_janek.md` | Điểm bắt đầu phiên làm việc |
| `file_ownership_janek.json` | Bản đồ từng file cũ → mới, hash trước/sau và nguồn gốc |

Giữ namespace `mixab` để phần Janek không đè các module `rev4` của Tuấn. Khi nằm chung repo, `mixab.monkey_janek.monkey_root()` tự tìm `rev4` và `lowres` ở root. Khi còn tách repo, vẫn hỗ trợ `MONKEY_V2_ROOT` và thư mục MONKEY_V2 bên cạnh như cũ.

## Quy tắc tên và nguồn gốc

- File đóng góp có hậu tố `_janek` trước phần mở rộng. Với `.json.gz`, tên là `events_clean.json_janek.gz`; vẫn là gzip chứa JSON.
- `__init__.py` và `.gitignore` giữ tên bắt buộc. Khi nhập chung, giữ `tests/__init__.py` của Tuấn và ghép các quy tắc ignore từ `gitignore_janek.txt`, không ghi đè `.gitignore` của Tuấn.
- `mixab/vendor/shared_manifest_janek.py` là mã upstream được giữ nguyên byte. Hậu tố chỉ xác định phần đóng gói của Janek, không tuyên bố Janek là tác giả mã upstream. Giữ nguyên nguồn gốc/hash trong inventory và tài liệu box-quality.
- JSON/CSV/ảnh/log/gzip kết quả lịch sử giữ nguyên byte. Các đường dẫn nằm bên trong identity/spec lịch sử vẫn mô tả lần chạy cũ; dùng inventory để tra vị trí công bố hiện tại.
- `runtime/`, checkpoint và các gói dataset đã tạo trước đây giữ tên cũ, không nhập vào Git. Tên đầu ra runtime là giao diện của thí nghiệm; hậu tố áp dụng cho bản file công bố trong repo.

## Nhập chung

1. Chạy `python tools/export_merge_janek.py` tại root repo Janek. Công cụ kiểm hash kết quả và xung đột với checkout MONKEY_V2, rồi tạo `runtime/restructure_janek/merge_ready_janek.zip`.
2. Trên nhánh tích hợp của repo chung, giải nén gói vào root. Gói không chứa `.git`, `.gitignore`, `tests/__init__.py`, runtime hay checkpoint. Ghép các quy tắc trong `gitignore_janek.txt` vào `.gitignore` chung.
3. Kiểm thử từ root: `python -m unittest discover -s tests -t . -p 'test_*_janek.py' -v`. Xem cả số test bị skip nếu thiếu bundle tham chiếu.
4. Thêm liên kết tới `README_janek.md` và `HANDOFF_janek.md` từ README/handoff chung. Kiểm diff trước khi merge; không ghi đè tài liệu của Tuấn.

Ví dụ CLI mới: `python -m mixab.boxquality_janek --help`. Dashboard: mở `tools/dashboard_janek/index_janek.html` hoặc phục vụ thư mục này bằng HTTP.

**Notebook mới cần gói code/dataset được dựng lại bằng script `_janek.py`.** Không dùng notebook đã đổi import với ZIP cũ trong Drive/Kaggle. Các ZIP cũ còn giá trị tái lập lần chạy cũ. Sau khi thay đổi cấu trúc, mới kiểm CPU và tạo notebook/gói code; chưa xác nhận lượt chạy GPU với cấu trúc mới.

Bản sao trước sắp xếp gồm cả thay đổi chưa commit: `runtime/restructure_janek/before.zip`. Công việc này chưa merge vào repo Tuấn và chưa push.
