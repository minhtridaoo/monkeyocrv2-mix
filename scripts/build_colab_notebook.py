"""Generate notebooks/colab_E1_A2_probe.ipynb (Colab notebook for E1 + A2 probe).

Edit the cell texts here and re-run: python scripts/build_colab_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'notebooks/colab_E1_A2_probe.ipynb'

CELLS = []


def md(text):
    CELLS.append(dict(cell_type='markdown', metadata={}, source=text.strip('\n')))


def code(text):
    CELLS.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=text.strip('\n')))


md(r'''
# CS431 · Mix A — E1 (môi trường T4/A100 + doctor) và A2 (probe `merge_audit`)

Notebook này chỉ làm **hai việc**:
1. **E1:** dựng môi trường trên Colab T4/A100, rồi chạy doctor tới khi **0 FAIL**.
2. **A2:** chạy **probe 40 nguồn** của `merge_audit`.

Notebook **không** chạy toàn bộ A2r, **không** train Mix B và **không** gửi gì cho ai.

**Trạng thái kiểm chứng của chính notebook:**
- Các hàm xử lý dữ liệu (`mixab/colab_helpers.py`) và code CPU đã có test ở máy local.
- **Mọi bước GPU/cài đặt chưa từng chạy trên Colab:** build MMCV, `install.sh`, doctor trên GPU, probe.
- Nếu một ô lỗi, đọc thông báo tiếng Việt và file log được in ra.

### Chuẩn bị (một lần)
1. Trên máy local, chạy `python scripts/make_colab_upload.py` trong thư mục `cải tiến/mới`. Lệnh này tạo `runtime/colab_upload/mixab_code.zip`.
2. Trên Google Drive, tạo thư mục **`MyDrive/CS431_MixA/upload/`**, rồi tải `mixab_code.zip` lên đó. **Đây là file duy nhất bạn phải tải lên bằng tay.**
3. Trên Colab: `Runtime → Change runtime type → GPU T4 hoặc A100`.
4. Chạy lần lượt Ô 1 → Ô 6. Mọi thứ khác notebook tự tải từ nguồn chính thức:
   - Ảnh TextVQA: `dl.fbaipublicfiles.com`, ~7,07 GB, chỉ giữ 2.262 ảnh cần dùng.
   - Encoder: Hugging Face `zenosai/MonkeyOCRv2-AS`.
   - Repo `ncat218/MONKEY_V2` và `Yuliang-Liu/MonkeyOCRv2` trên GitHub.

### Dữ liệu tạm và dữ liệu giữ lại
- **`/content/cs431_tmp/` = TẠM.** Mất khi Colab ngắt hoặc reset VM. Chứa code đã giải nén, các repo, mã nguồn MMCV, ảnh đã giải nén, ảnh LR2, config đã sửa. Tất cả tạo lại được.
- **`MyDrive/CS431_MixA/` = GIỮ LẠI:**
  - `upload/` (code bạn tải lên)
  - `cache/` (wheel MMCV đã build, encoder, zip 2.262 ảnh)
  - `weights/` (5 checkpoint merge)
  - `runs/` (kết quả probe, resume được)
  - `reports/` (env/doctor)
  - `logs/`
  - `export/` (gói zip để gửi lại)
- Cần khoảng **1,5 GB trống trên Drive**, và khoảng 20 GB trên đĩa Colab cho lần tải ảnh đầu tiên.

### Chạy lại an toàn
- Mỗi ô **bỏ qua bước nặng đã xong** (kiểm bằng phiên bản hoặc SHA, không dựa vào dấu "đã chạy").
- Không ô nào ghi đè kết quả: report, log và export đều có timestamp; thư mục dở dang được đổi tên chứ không bị xóa.
- **Sau khi restart session hoặc khi Colab cấp VM mới: luôn chạy lại Ô 1 trước**, rồi chạy tiếp từ ô bạn đang dừng.
''')

md(r'''
## Ô 1 · Cấu hình và mount Drive (chạy lại ô này sau mỗi lần restart)

Chỉ cần sửa `DRIVE_ROOT` nếu bạn đặt thư mục ở chỗ khác. Các commit được **ghim cố định** để kết quả so được với của Tuấn:
- MONKEY_V2 `dc19b85` là commit có bundle `pr_quad_audit_v1`.
- MonkeyOCRv2 `da96afe` là bản đã đối chiếu ở máy local.

Đổi commit thì đổi luôn identity của lần chạy, nên phải ghi lý do vào WORKLOG.
''')

code(r'''
# ===================== Ô 1 · CẤU HÌNH =====================
import json, os, subprocess, sys, zipfile
from pathlib import Path

DRIVE_ROOT = '/content/drive/MyDrive/CS431_MixA'      # <-- thư mục của bạn trên Google Drive

PIN = dict(
    MONKEY_V2_URL='https://github.com/ncat218/MONKEY_V2.git',
    MONKEY_V2_COMMIT='dc19b85c8f73d56dd0510165069597363fe25380',
    MONKEYOCRV2_URL='https://github.com/Yuliang-Liu/MonkeyOCRv2.git',
    MONKEYOCRV2_COMMIT='da96afede92f198c19aca74e1e5f5aa7f4c26af2',
    TEXTVQA_ZIP_URL='https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip',
    TEXTVQA_ZIP_BYTES=7072297970,                          # đo bằng HTTP HEAD ngày 20/11/2026
    ENCODER_REPO='zenosai/MonkeyOCRv2-AS',
    BCLEAN_SHA256='94f0e4ecbb6881b4430c841845994c6568c145af7bb54fc41514eb0c78f3a6aa',
    B1_SHA256='9c3814e494a73a0a6089ba5a6585f079ea8f5a6defdc9347213bb98a64337147',
    # Môi trường Tuấn đã kiểm (MONKEY_V2 docs/handoff/checkpoints.md + MonkeyOCRv2 detection/mmocr/README.md)
    TORCH='2.9.0', TORCHVISION='0.24.0', TORCH_INDEX='https://download.pytorch.org/whl/cu128',
    MMENGINE='0.10.7', MMCV='2.0.1', MMOCR='1.0.1', PILLOW='11.3.0', OPENCV='4.10.0.84',
)
PROBE_SOURCES = 40       # probe cố định 40 nguồn x 2 điều kiện; KHÔNG đổi giữa các lần resume
PROBE_CHUNK = 16         # lưu tiến độ sau mỗi 16 task

# ---- mount Drive (chỉ chạy trên Colab) ----
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    raise SystemExit('Notebook này chạy trên Google Colab (cần google.colab để mount Drive).')

D = Path(DRIVE_ROOT)
UPLOAD, CACHE, WEIGHTS = D/'upload', D/'cache', D/'weights/merge_weights_v1'
RUNS, REPORTS, LOGS, EXPORT = D/'runs', D/'reports', D/'logs', D/'export'
T = Path('/content/cs431_tmp')                       # TẠM: mất khi VM reset
CODE, MV2, MOCR = T/'code', T/'MONKEY_V2', T/'MonkeyOCRv2'
IMAGES, PREPARED, DOWNLOADS = T/'textvqa', T/'prepared', T/'downloads'
CONFIG_PATH = T/'inference_config_colab.py'
for p in (UPLOAD, CACHE, RUNS, REPORTS, LOGS, EXPORT, T):
    p.mkdir(parents=True, exist_ok=True)

# ---- code của mình (mixab) từ Drive -> /content (giải nén lại khi zip đổi) ----
CODE_ZIP = UPLOAD/'mixab_code.zip'
if not CODE_ZIP.is_file():
    raise SystemExit(f'Chưa thấy {CODE_ZIP}. Trên máy local chạy `python scripts/make_colab_upload.py` '
                     'rồi tải runtime/colab_upload/mixab_code.zip lên thư mục upload/ trên Drive.')
import hashlib
code_sha = hashlib.sha256(CODE_ZIP.read_bytes()).hexdigest()
marker = CODE/'.code_zip_sha256'
if not (marker.is_file() and marker.read_text() == code_sha):
    if CODE.exists():
        CODE.rename(T/f'code_old_{code_sha[:8]}_{int(__import__("time").time())}')
    with zipfile.ZipFile(CODE_ZIP) as zf:
        zf.extractall(CODE)
    marker.write_text(code_sha)
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
import importlib
import mixab.colab_helpers as ch
importlib.reload(ch)

# ---- GPU ----
gpu_name = (ch.output_of(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader']) or '').splitlines()
if not gpu_name:
    raise SystemExit('Không thấy GPU. Runtime -> Change runtime type -> chọn T4 hoặc A100, rồi chạy lại Ô 1.')
GPU = ch.gpu_label(gpu_name[0])
if not ch.gpu_supported(gpu_name[0]):
    print(f'CẢNH BÁO: GPU {gpu_name[0]} không phải T4/A100 (ngoài phạm vi đã thống nhất). Nên đổi runtime.')
PY = sys.executable
ENV = dict(MONKEY_V2_ROOT=str(MV2), PYTHONPATH=str(CODE))
BUNDLE = MV2/'docs/handoff/artifacts/pr_quad_audit_v1'
print(f'GPU: {gpu_name[0]} -> nhãn {GPU}\nPython: {sys.version.split()[0]}\nCode zip SHA-256: {code_sha[:16]}...\n'
      f'Drive (giữ lại): {D}\nTạm: {T}')
''')

md(r'''
## Ô 2 · Kiểm GPU và cài môi trường (E1)

Thứ tự cài theo hướng dẫn chính thức của MonkeyOCRv2 (`detection/mmocr/README.md`):
1. torch 2.9.0 cu128
2. mmengine 0.10.7
3. **MMCV 2.0.1 build từ source** (không có wheel cho torch 2.9)
4. `install.sh`: clone MMOCR v1.0.1, chép add-on có `MonkeyOCRv2ViTAEBackbone`, cài mmdet 3.1.0, transformers 4.57.1, và vá imgaug/mmengine
5. ghim lại Pillow 11.3.0 và OpenCV 4.10.0.84

- **Build MMCV lâu**, có thể vài chục phút với 2 CPU của T4 (chưa đo). Wheel được lưu vào `cache/wheels/` trên Drive với `TORCH_CUDA_ARCH_LIST=7.5;8.0`, nên lần sau dùng chung cho cả T4 và A100, miễn là cùng phiên bản Python.
- Mọi lệnh đều được ghi log vào `logs/install_<GPU>.log`. Ô có thể chạy lại: phần nào đúng phiên bản rồi thì bỏ qua.
- **Restart:** các bước nặng đều chạy trong tiến trình con, nên thường **không cần restart**. Nếu Colab hiện nút *Restart session* sau khi pip đổi gói, bấm restart, **chạy lại Ô 1**, rồi chạy tiếp **Ô 2** (bỏ qua rất nhanh) và **Ô 3**.
''')

code(r'''
# ===================== Ô 2 · CÀI MÔI TRƯỜNG =====================
import platform, re, shutil
LOG = LOGS/f'install_{GPU}.log'
pyver = sys.version_info
if not ((3, 10) <= pyver[:2] <= (3, 12)):
    print(f'CẢNH BÁO: Python {platform.python_version()} chưa được kiểm với MMCV 2.0.1 (README dùng 3.11).')

def ver(name):
    return ch.package_version(name)

def pip(*args):
    ch.run([PY, '-m', 'pip', 'install', '-q', *args], LOG, title='pip install ' + ' '.join(args))

# 1) torch
if ver('torch') != f"{PIN['TORCH']}+cu128" or ver('torchvision') != f"{PIN['TORCHVISION']}+cu128":
    pip(f"torch=={PIN['TORCH']}", f"torchvision=={PIN['TORCHVISION']}", '--index-url', PIN['TORCH_INDEX'])
cuda_ok = ch.output_of([PY, '-c', 'import torch;print(torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_arch_list())'])
print('torch CUDA:', cuda_ok)
if not cuda_ok or 'True' not in cuda_ok:
    ch.fail('torch không dùng được CUDA. Kiểm runtime GPU rồi chạy lại Ô 1 và Ô 2.')

# 2) mmengine + công cụ build
if ver('mmengine') != PIN['MMENGINE']:
    pip(f"mmengine=={PIN['MMENGINE']}", 'setuptools<81', 'wheel', 'ninja')

# 3) MMCV 2.0.1 với ops (dùng wheel cache trên Drive nếu có)
def mmcv_ok():
    return ver('mmcv') == PIN['MMCV'] and ch.output_of([PY, '-c', 'from mmcv.ops import nms; print("ok")']) == 'ok'

wheel_dir = CACHE/f"wheels/mmcv{PIN['MMCV']}_py{pyver[0]}{pyver[1]}_torch{PIN['TORCH']}_cu128_sm75-80"
if not mmcv_ok():
    cached = sorted(wheel_dir.glob('mmcv-*.whl'))
    if cached:
        print('Cài MMCV từ wheel cache:', cached[-1].name)
        pip('--no-deps', str(cached[-1]))
    else:
        nvcc = shutil.which('nvcc') or '/usr/local/cuda/bin/nvcc'
        nvcc_ver = ch.output_of([nvcc, '--version']) or ''
        m = re.search(r'release (\d+)\.(\d+)', nvcc_ver)
        if not m or m.group(1) != '12':
            ch.fail(f'Cần CUDA toolkit 12.x (nvcc) để build MMCV cho torch cu128; tìm thấy: {nvcc_ver[-80:]!r}')
        cuda_home = '/usr/local/cuda-12.8' if Path('/usr/local/cuda-12.8').is_dir() else str(Path(nvcc).resolve().parents[1])
        src = T/'mmcv_src'
        if not (src/'setup.py').is_file():
            ch.run(['git', 'clone', '-b', f"v{PIN['MMCV']}", '--depth', '1', 'https://github.com/open-mmlab/mmcv.git', src], LOG)
        setup = (src/'setup.py').read_text()
        if '-std=c++14' in setup:                         # bước sed của README chính thức
            (src/'setup.py').write_text(setup.replace('-std=c++14', '-std=c++17'))
        print(f'Build MMCV (CUDA_HOME={cuda_home}, nvcc {m.group(1)}.{m.group(2)}, {os.cpu_count()} CPU). Có thể mất lâu...')
        build_env = dict(CUDA_HOME=cuda_home, MMCV_WITH_OPS='1', MAX_JOBS=str(os.cpu_count() or 2),
                         TORCH_CUDA_ARCH_LIST='7.5;8.0')
        wheel_tmp = T/'wheelhouse'
        ch.run([PY, '-m', 'pip', 'wheel', '--no-build-isolation', '--no-deps', '-w', wheel_tmp, src], LOG,
               env=build_env, title='build MMCV wheel')
        built = sorted(wheel_tmp.glob('mmcv-*.whl'))
        if not built:
            ch.fail('Build MMCV không tạo ra wheel; xem log.')
        wheel_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(built[-1], wheel_dir/built[-1].name)
        pip('--no-deps', str(built[-1]))
    if not mmcv_ok():
        ch.fail('MMCV vẫn chưa import được mmcv.ops.nms sau khi cài; xem log.')

# 4) MonkeyOCRv2 add-on + MMOCR v1.0.1 (install.sh chính thức)
if not (MOCR/'.git').is_dir():
    ch.run(['git', 'clone', PIN['MONKEYOCRV2_URL'], MOCR], LOG)
ch.run(['git', '-C', MOCR, 'checkout', '--detach', PIN['MONKEYOCRV2_COMMIT']], LOG)
addon = MOCR/'detection/mmocr'
mmocr_root = addon/'mmocr'
backbone_file = mmocr_root/'mmocr/models/common/backbones/monkeyocr_v2_vitae.py'
mmocr_loc = ch.output_of([PY, '-c', 'import mmocr,os;print(os.path.dirname(mmocr.__file__))'])
if not (ver('mmocr') == PIN['MMOCR'] and backbone_file.is_file() and mmocr_loc and str(mmocr_root) in mmocr_loc):
    # install.sh gọi `python`/`pip` theo PATH: đặt interpreter của kernel lên đầu PATH
    ch.run(['bash', 'install.sh'], LOG, cwd=addon, title='MonkeyOCRv2 detection/mmocr/install.sh',
           env=dict(PATH=f"{Path(PY).parent}:{os.environ.get('PATH', '')}"))

# 5) ghim lại Pillow / OpenCV (requirements của MMOCR có thể đã kéo bản khác)
if ver('Pillow') != PIN['PILLOW']:
    pip(f"Pillow=={PIN['PILLOW']}")
if ver('opencv-python') != PIN['OPENCV'] or any(ver(n) for n in ('opencv-python-headless', 'opencv-contrib-python')):
    ch.run([PY, '-m', 'pip', 'uninstall', '-y', 'opencv-python-headless', 'opencv-contrib-python',
            'opencv-contrib-python-headless'], LOG)
    # gỡ một biến thể có thể làm hỏng thư mục cv2 dùng chung -> cài lại opencv-python sạch
    pip('--force-reinstall', '--no-deps', f"opencv-python=={PIN['OPENCV']}")

# 6) kiểm cuối trong tiến trình MỚI (không dùng module cũ của kernel)
VERIFY = """
import hashlib, importlib.metadata as m, importlib.util, json, torch
from mmocr.utils import register_all_modules
from mmocr.registry import MODELS
from mmcv.ops import nms
register_all_modules(init_default_scope=True)
spec = importlib.util.find_spec('mmocr.models.textdet.postprocessors.db_postprocessor')
out = dict(versions={k: m.version(k) for k in ('torch','torchvision','mmengine','mmcv','mmdet','mmocr','Pillow','opencv-python','numpy','transformers')},
           cuda=torch.cuda.is_available(), device=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
           backbone_registered=MODELS.get('MonkeyOCRv2ViTAEBackbone') is not None,
           db_postprocessor_sha256=hashlib.sha256(open(spec.origin,'rb').read()).hexdigest(), db_postprocessor_path=spec.origin)
print(json.dumps(out))
"""
raw = ch.output_of([PY, '-c', VERIFY])
if not raw:
    ch.fail('Kiểm cuối môi trường thất bại khi import (torch/mmcv/mmocr). Chạy lại Ô 2 và xem log: ' + str(LOG))
env_report = json.loads(raw.splitlines()[-1])
env_path = ch.new_path(REPORTS, f'env_{GPU}', '.json')
env_path.write_text(json.dumps(dict(env_report, pins=PIN), indent=1))
print(json.dumps(env_report, indent=1))
problems = []
if env_report['db_postprocessor_sha256'] != '9bfff8aaaca103dd08d7565f2b7a5410ab20a6427fe1f0baad5a5d04d207c8b8':
    problems.append('DBPostprocessor khác bản Tuấn đã ghim')
if not env_report['backbone_registered']:
    problems.append('MonkeyOCRv2ViTAEBackbone chưa được đăng ký')
if env_report['versions']['Pillow'] != PIN['PILLOW']:
    problems.append('Pillow không phải 11.3.0')
if problems:
    ch.fail('Môi trường chưa đạt: ' + '; '.join(problems))
print(f'\nE1 (phần cài đặt) xong. Report: {env_path}\nĐây CHƯA phải doctor; chạy tiếp Ô 3 rồi Ô 4.')
''')

md(r'''
## Ô 3 · Dữ liệu và weights

Thứ tự phụ thuộc, mỗi bước bỏ qua nếu đã đúng SHA:
1. **MONKEY_V2** ở commit ghim. Chỉ tải 2 file weights qua Git LFS, rồi chạy `verify_bundle.py --counts --weights` (phải PASS).
2. **Test CPU** của `mixab` trên đúng bundle và weights này.
3. **Encoder** MonkeyOCRv2-AS: tải vào `cache/` trên Drive và ghi lại revision. Sau đó tạo bản config chỉ đổi **2 đường dẫn encoder**, mọi thứ khác giữ nguyên.
4. **Ảnh Clean:** lấy từ `cache/textvqa_val2262_clean.zip` trên Drive nếu đã có. Nếu chưa, tải zip TextVQA 7,07 GB vào thư mục tạm, chỉ trích 2.262 ảnh, kiểm SHA từng ảnh với `source_cohort.csv`, rồi lưu bản cache lên Drive. Lần sau không phải tải lại.
5. **5 checkpoint merge** trên Drive: dùng lại nếu SHA khớp manifest. Nếu có thư mục dở dang thì đổi tên chứ không xóa, rồi tạo lại.
6. **Ảnh native LR2** trong thư mục tạm: `prepare-lr2` dùng đúng hàm `native_lr2` của Tuấn và kiểm SHA từng ảnh.
''')

code(r'''
# ===================== Ô 3 · DỮ LIỆU + WEIGHTS =====================
LOG = LOGS/f'data_{GPU}.log'
# 3.1 MONKEY_V2 (tham chiếu, không sửa)
if not (MV2/'.git').is_dir():
    ch.run(['git', 'clone', PIN['MONKEY_V2_URL'], MV2], LOG, env=dict(GIT_LFS_SKIP_SMUDGE='1'))
ch.run(['git', '-C', MV2, 'checkout', '--detach', PIN['MONKEY_V2_COMMIT']], LOG, env=dict(GIT_LFS_SKIP_SMUDGE='1'))
if ch.output_of(['git', 'lfs', 'version']) is None:
    ch.run(['bash', '-c', 'apt-get -qq update && apt-get -qq install -y git-lfs'], LOG, title='install git-lfs')
ch.run(['git', '-C', MV2, 'lfs', 'install', '--local'], LOG)
weights_rel = ['docs/handoff/weights/bclean-inference.pth', 'docs/handoff/weights/b1-inference.pth']
ch.run(['git', '-C', MV2, 'lfs', 'pull', '--include=' + ','.join(weights_rel)], LOG)
for rel, want in zip(weights_rel, (PIN['BCLEAN_SHA256'], PIN['B1_SHA256'])):
    if ch.sha256(MV2/rel) != want:
        ch.fail(f'{rel} sai SHA-256 (Git LFS chưa tải đủ?).')
dirty = ch.output_of(['git', '-C', MV2, 'status', '--porcelain', '--untracked-files=no'])
if dirty:
    ch.fail(f'Checkout MONKEY_V2 bị sửa (dirty), không dùng làm tham chiếu được:\n{dirty}')
ch.run([PY, BUNDLE/'verify_bundle.py', '--counts', '--weights'], LOG, cwd=MV2, title='verify_bundle.py --counts --weights')

# 3.2 test CPU của mixab trên đúng bundle/weights này
ch.run([PY, '-m', 'unittest', 'discover', '-s', 'tests', '-t', '.'], LOG, cwd=CODE, env=ENV, title='mixab unit tests')

# 3.3 encoder + config
ENCODER = CACHE/'encoder_monkeyocrv2_as'
if not ((ENCODER/'model.safetensors').is_file() and (ENCODER/'config.json').is_file()):
    ch.run([PY, '-c', f"from huggingface_hub import snapshot_download; snapshot_download({PIN['ENCODER_REPO']!r}, local_dir={str(ENCODER)!r})"],
           LOG, title='download encoder')
    rev = ch.output_of([PY, '-c', f"from huggingface_hub import HfApi; print(HfApi().model_info({PIN['ENCODER_REPO']!r}).sha)"])
    (ENCODER/'_revision.txt').write_text(f'{rev}\n')
print('Encoder revision:', (ENCODER/'_revision.txt').read_text().strip() if (ENCODER/'_revision.txt').is_file() else 'chưa ghi')
patched = ch.patch_inference_config((MV2/'docs/handoff/inference_config.py').read_text(encoding='utf-8'), ENCODER)
if not CONFIG_PATH.is_file() or CONFIG_PATH.read_text(encoding='utf-8') != patched:
    CONFIG_PATH.write_text(patched, encoding='utf-8')

# 3.4 ảnh Clean (2.262 ảnh)
items = ch.needed_clean_images(BUNDLE)
names = [r for r, _ in items]
CACHE_ZIP = CACHE/'textvqa_val2262_clean.zip'
if CACHE_ZIP.is_file():
    ch.extract_members(CACHE_ZIP, names, IMAGES)
else:
    big = DOWNLOADS/'train_val_images.zip'
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    if not (big.is_file() and big.stat().st_size == PIN['TEXTVQA_ZIP_BYTES']):
        ch.run(['wget', '-c', '-q', '--show-progress', '-O', big, PIN['TEXTVQA_ZIP_URL']], LOG, title='download TextVQA zip (7 GB)')
    if big.stat().st_size != PIN['TEXTVQA_ZIP_BYTES']:
        ch.fail(f'Zip TextVQA có {big.stat().st_size} byte, kỳ vọng {PIN["TEXTVQA_ZIP_BYTES"]}; chạy lại để tải tiếp.')
    ch.extract_members(big, names, IMAGES)
check = ch.verify_images(IMAGES, items)
print('Ảnh Clean:', check)
if check['ok'] != len(items):
    ch.fail(f'Ảnh Clean chưa khớp SHA: {check}')
if not CACHE_ZIP.is_file():
    ch.build_cache_zip(IMAGES, names, CACHE_ZIP)
    print('Đã lưu cache ảnh lên Drive:', CACHE_ZIP)
    big.unlink()                                              # file tải 7 GB chỉ nằm ở thư mục tạm
print('Ảnh Clean OK: 2.262/2.262 SHA khớp')

# 3.5 5 checkpoint merge (Drive)
manifest = WEIGHTS/'merge_manifest.json'
def weights_ok():
    if not manifest.is_file():
        return False
    m = json.loads(manifest.read_text())
    good = all((WEIGHTS/Path(r['checkpoint']).name).is_file() and ch.sha256(WEIGHTS/Path(r['checkpoint']).name) == r['checkpoint_sha256']
               for r in m['models'])
    return good and m['endpoints']['bclean']['sha256'] == PIN['BCLEAN_SHA256'] and m['endpoints']['b1']['sha256'] == PIN['B1_SHA256']
if weights_ok():
    print('Dùng lại 5 checkpoint merge đã có trên Drive (SHA khớp).')
else:
    moved = ch.move_aside(WEIGHTS)
    if moved:
        print('Thư mục weights cũ chưa đầy đủ đã được đổi tên thành:', moved)
    ch.run([PY, '-m', 'mixab.weight_merge', '--bclean', MV2/weights_rel[0], '--b1', MV2/weights_rel[1], '--output', WEIGHTS],
           LOG, cwd=CODE, env=ENV, title='mixab.weight_merge')
    if not weights_ok():
        ch.fail('Checkpoint merge vừa tạo không qua kiểm SHA.')

# 3.6 native LR2 (tạm, tái tạo được, kiểm SHA từng ảnh)
ch.run([PY, '-m', 'mixab.merge_audit', 'prepare-lr2', '--bundle', BUNDLE, '--clean-root', IMAGES, '--prepared-root', PREPARED],
       LOG, cwd=CODE, env=ENV, title='prepare-lr2')
print('\nÔ 3 xong: bundle PASS, test PASS, encoder + config, 2.262 ảnh Clean, 5 checkpoint, LR2.')
''')

md(r'''
## Ô 4 · Doctor (cổng vào của probe)

Doctor ghi report mới với timestamp vào `reports/doctor_<GPU>_*.json`, không bao giờ ghi đè report cũ.
- **Chỉ khi 0 FAIL** thì ô này mới ghi "giấy thông hành" cho phiên hiện tại (`/content/cs431_tmp/doctor_pass.json`).
- Không có tùy chọn bỏ qua hay nới kiểm tra. Còn FAIL thì sửa đúng mục đó rồi chạy lại.
- WARN (ví dụ T4 không có TF32, phiên bản OpenCV hay torch lệch nhỏ) **không chặn**, nhưng được ghi lại để đối chiếu nếu cổng 3a lệch.
''')

code(r'''
# ===================== Ô 4 · DOCTOR =====================
doctor_path = ch.new_path(REPORTS, f'doctor_{GPU}', '.json')
LOG = LOGS/f'doctor_{GPU}.log'
cmd = [PY, '-m', 'mixab.doctor', '--bundle', BUNDLE, '--clean-root', IMAGES, '--prepared-root', PREPARED,
       '--model-config', CONFIG_PATH, '--weights-dir', WEIGHTS, '--output', doctor_path]
try:
    ch.run(cmd, LOG, cwd=CODE, env=ENV, title='mixab.doctor')
except ch.StepError:
    pass                                   # doctor exit 1 khi có FAIL; đọc report bên dưới
if not doctor_path.is_file():
    ch.fail(f'Doctor không tạo được report; xem {LOG}')
report = json.loads(doctor_path.read_text())
fails = ch.failed_checks(report)
pass_flag = T/'doctor_pass.json'
if pass_flag.exists():
    pass_flag.unlink()                     # giấy thông hành chỉ hợp lệ cho lần doctor mới nhất
if fails:
    ch.fail('Doctor còn FAIL, chưa được chạy probe:\n' + '\n'.join(f"- {r['check']}: {r['detail']}" for r in fails)
            + f'\nReport: {doctor_path}')
pass_flag.write_text(json.dumps(dict(report=str(doctor_path), gpu=GPU, code_zip_sha256=code_sha)))
print(f'\nDoctor: 0 FAIL. Report: {doctor_path}\nĐược phép chạy Ô 5.')
''')

md(r'''
## Ô 5 · Probe 40 nguồn (A2)

`merge_audit run --probe 40` chạy lần lượt 5 checkpoint (α = 0, ¼, ½, ¾, 1) × 40 nguồn × (Clean, LR2):
- Dùng đúng `detector_for`/`forward` của Tuấn: **batch 1 (từng ảnh), FP32**; doctor đã kiểm config là FP32.
- Mỗi ảnh được chấm trên lưới 9 geometry × 8 score.
- Riêng 2 endpoint chạy thêm cổng 3a (so với EDA của H200) và cổng P-map (bản sao so với export gốc, trên cùng GPU).

**Kết quả lưu trên Drive** ở `runs/merge_audit_probe_v1_<GPU>/`:
- Tiến độ được lưu sau mỗi 16 task, nên nếu Colab ngắt, chỉ cần chạy lại Ô 1 → Ô 3 → Ô 4 → Ô 5 và nó **chạy tiếp từ chỗ dừng**.
- Probe đã xong (có `gates.json`) thì ô này chỉ báo "đã xong", không chạy lại.
- Thư mục tách theo GPU, vì kết quả T4 và A100 không được trộn với nhau.

Thời gian và VRAM đỉnh trên GPU thật được ghi vào `profile.json`.
''')

code(r'''
# ===================== Ô 5 · PROBE =====================
pass_flag = T/'doctor_pass.json'
if not pass_flag.is_file():
    ch.fail('Chưa có doctor 0 FAIL trong phiên này. Chạy Ô 4 trước.')
flag = json.loads(pass_flag.read_text())
if flag['gpu'] != GPU or flag['code_zip_sha256'] != code_sha:
    ch.fail('Doctor đã chạy với GPU hoặc code khác. Chạy lại Ô 4.')
OUT = RUNS/f'merge_audit_probe_v1_{GPU}'
if (OUT/'gates.json').is_file():
    print(f'Probe trên {GPU} đã hoàn tất trước đó ({OUT}). Không chạy lại; sang Ô 6.')
else:
    log = ch.new_path(LOGS, f'probe_{GPU}', '.log')
    ch.run([PY, '-m', 'mixab.merge_audit', 'run', '--bundle', BUNDLE, '--clean-root', IMAGES, '--prepared-root', PREPARED,
            '--model-config', CONFIG_PATH, '--weights-dir', WEIGHTS,
            '--bclean-export', MV2/'docs/handoff/weights/bclean-inference.pth',
            '--b1-export', MV2/'docs/handoff/weights/b1-inference.pth',
            '--probe', str(PROBE_SOURCES), '--chunk', str(PROBE_CHUNK), '--output', OUT],
           log, cwd=CODE, env=ENV, title=f'merge_audit probe {PROBE_SOURCES} on {GPU}')
    print(f'\nProbe xong. Kết quả: {OUT}\nLog: {log}')
''')

md(r'''
## Ô 6 · Tổng hợp và đóng gói gửi lại

Ô này đọc `gates.json` và `profile.json`, in trạng thái cổng kiểm, thời gian và VRAM, rồi tạo **một file zip mới có timestamp** trong `export/` để bạn tải về gửi lại. Zip gồm:
- `gates.json`, `profile.json`, `audit_identity.json`, `reselect_spec.json` và các NPZ probe;
- toàn bộ report env/doctor và log của GPU này;
- `probe_summary.json`.

Cách đọc kết quả:
- **Cổng 3a PASS và P-map PASS ở cả hai endpoint:** có thể đề xuất chạy đầy đủ (A2r). Đó là quyết định của lượt sau, không phải của notebook này.
- **3a FAIL:** nhiều khả năng do khác GPU so với H200. **Không tự nới cổng.** Gửi zip lại để báo Tuấn theo kế hoạch §4.3.
- **P-map FAIL:** lỗi nạp hoặc lưu trọng số. Dừng và gửi log.

E1 và A2 chỉ được đánh dấu xong trên dashboard **sau khi** đã đọc các file này.
''')

code(r'''
# ===================== Ô 6 · TỔNG HỢP + ĐÓNG GÓI =====================
OUT = RUNS/f'merge_audit_probe_v1_{GPU}'
if not (OUT/'gates.json').is_file():
    ch.fail('Chưa có gates.json: probe chưa xong. Chạy Ô 5.')
gates = json.loads((OUT/'gates.json').read_text())
profile = json.loads((OUT/'profile.json').read_text()) if (OUT/'profile.json').is_file() else {}
summary = ch.summarize_probe(gates, profile)
print(f"GPU: {summary['gpu']}")
for r in summary['endpoints']:
    print(f"  {r['model']:8} cổng 3a: {r['gate_3a']:7} (lệch {r['mismatched_3a']}/{r['checked_3a']})   "
          f"P-map: {r['pmap']} (max |dP| = {r['pmap_max_abs_diff']})")
for name, p in profile.items():
    print(f"  {name:12} {p['tasks']} task, {p['mean_seconds_per_task']:.2f} s/task, VRAM đỉnh {p['peak_gpu_mib']:.0f} MiB")
if summary['full_run_estimate_hours'] is not None:
    print(f"Ước lượng chạy đầy đủ 5 model x 4.524 task trên GPU này: ~{summary['full_run_estimate_hours']:.1f} giờ "
          '(chưa tính thời gian kiểm P-map; chỉ là ngoại suy từ probe)')
print('Sẵn sàng đề xuất chạy đầy đủ:' , 'CÓ' if summary['ready_for_full_run'] else 'KHÔNG — xem cổng FAIL ở trên')

pack = ch.new_path(EXPORT, f'cs431_probe_{GPU}', '.zip')
(OUT/'probe_summary.json').write_text(json.dumps(summary, indent=1)) if not (OUT/'probe_summary.json').exists() else None
with zipfile.ZipFile(pack, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for p in sorted(OUT.iterdir()):
        if p.is_file() and not p.name.startswith('.'):
            zf.write(p, f'run/{p.name}')
    for folder, pattern in ((REPORTS, f'*_{GPU}_*.json'), (LOGS, f'*{GPU}*.log')):
        for p in sorted(folder.glob(pattern)):
            zf.write(p, f'{folder.name}/{p.name}')
    zf.writestr('code_zip_sha256.txt', code_sha + '\n')
print(f'\nĐã đóng gói: {pack} ({pack.stat().st_size/1024:.0f} KiB). File nằm trên Drive; có thể tải trực tiếp:')
try:
    from google.colab import files
    files.download(str(pack))
except Exception as exc:
    print('Không tự tải được, hãy tải từ Drive:', exc)
''')


def main():
    import sys
    sys.stdout.reconfigure(errors='backslashreplace')
    notebook = dict(nbformat=4, nbformat_minor=5,
                    metadata=dict(accelerator='GPU', colab=dict(provenance=[], gpuType='T4'),
                                  kernelspec=dict(name='python3', display_name='Python 3'),
                                  language_info=dict(name='python')),
                    cells=[dict(c, id=f'cell-{i:02d}',
                                source=[line + '\n' for line in c['source'].split('\n')][:-1] +
                                [c['source'].split('\n')[-1]]) for i, c in enumerate(CELLS)])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(f'{OUT}: {len(CELLS)} cells')


if __name__ == '__main__':
    main()
