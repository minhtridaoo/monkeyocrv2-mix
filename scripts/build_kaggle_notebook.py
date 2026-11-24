"""Generate notebooks/kaggle_csb1_latency.ipynb: fair latency B1 vs CS-B1@0.7 on one Kaggle T4.

Environment cells follow the Colab notebook that ran A2r/the T4 dump (same pins, same install order), with the
Google-Drive parts replaced by a Kaggle Dataset (read-only, /kaggle/input) and /kaggle/working for outputs.
Re-run: python scripts/build_kaggle_notebook.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'notebooks/kaggle_csb1_latency.ipynb'
CELLS = []


def md(text):
    CELLS.append(dict(cell_type='markdown', metadata={}, source=text.strip('\n')))


def code(text):
    CELLS.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=text.strip('\n')))


md(r'''
# CS431 · Đo latency công bằng: B1 một mình vs CS-B1@0.7 (Kaggle, một T4)

**Notebook làm gì:** đo thời gian suy luận và bộ nhớ GPU của hai cách chạy trên **cùng máy, cùng phiên, cùng ảnh**:
- **B1:** giải mã ảnh → B1 → hậu xử lý DB (`epsilon_005`, score ≥ 0,5).
- **CS-B1@0.7:** giải mã ảnh **một lần** → B1 → B-Clean → hậu xử lý cả hai (B1 ≥ 0,5; B-Clean `mask_040` ≥ 0,7) → gộp box (IoU ≥ 0,5).

Không train, không dùng TEST, không chạy lại toàn bộ dự đoán: chỉ 220 ảnh LR2 cố định (20 khởi động + 200 đo). Dump T4 cũ chỉ dùng để **kiểm chéo** số box.

### Trước khi chạy (một lần)
1. Trên Kaggle: **Datasets → New Dataset**, tải lên file `cs431-csb1-latency.zip` (Claude đã chuẩn bị ở `runtime/kaggle_upload/`). Đặt tên tùy ý, để **Private**.
2. Tạo Notebook mới, **File → Import Notebook**, chọn file này.
3. Cột bên phải (**Settings**):
   - **Accelerator:** `GPU T4 x2` (hoặc `GPU T4` nếu có). Notebook chỉ dùng **GPU 0**.
   - **Internet:** **On** (cần tài khoản đã xác minh số điện thoại). Cần để cài torch, MMCV/MMOCR và clone repo của Tuấn.
   - **Add Input** → chọn dataset vừa tạo.
4. Chạy lần lượt **Ô 1 → Ô 6** (hoặc *Run All*). Ô 2 có thể mất 20–40 phút nếu phải build MMCV (chưa đo trên Kaggle).

### Kết quả
Tất cả nằm trong `/kaggle/working/cs431_out/`. Ô 6 tạo **một file zip** trong `/kaggle/working/`; tải về ở khung **Output** (bên phải) và gửi lại cho Claude.

### Không được làm
- Không đem số ở đây so trực tiếp với latency đo ở máy/GPU khác để kết luận "chậm hơn bao nhiêu".
- Không sửa ngưỡng hay luật gộp trong notebook.

**Trạng thái:** notebook đã được tạo và kiểm cú pháp ở máy local; **chưa chạy thử trên Kaggle**. Nếu một ô lỗi, đọc thông báo tiếng Việt và file log trong `/kaggle/working/cs431_out/logs/`.
''')

md(r'''
## Ô 1 · Cấu hình, tìm dataset, kiểm GPU

- Tìm dataset trong `/kaggle/input` (theo file `KAGGLE_MANIFEST.json`). Nếu Kaggle giữ nguyên file zip mà không giải nén, ô này tự giải nén vào thư mục tạm.
- Kiểm **SHA-256 của mọi file** trong dataset.
- Ghim **GPU 0** cho mọi tiến trình đo (`CUDA_VISIBLE_DEVICES=0`), vì code của Tuấn yêu cầu đúng một GPU.
''')

code(r'''
# ===================== Ô 1 · CẤU HÌNH =====================
import hashlib, json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path

PIN = dict(
    MONKEY_V2_URL='https://github.com/ncat218/MONKEY_V2.git',
    MONKEY_V2_COMMIT='dc19b85c8f73d56dd0510165069597363fe25380',     # commit của dump T4 và A2r
    MONKEYOCRV2_URL='https://github.com/Yuliang-Liu/MonkeyOCRv2.git',
    MONKEYOCRV2_COMMIT='da96afede92f198c19aca74e1e5f5aa7f4c26af2',
    TORCH='2.9.0', TORCHVISION='0.24.0', TORCHAUDIO='2.9.0', TORCH_INDEX='https://download.pytorch.org/whl/cu128',
    MMENGINE='0.10.7', MMCV='2.0.1', MMOCR='1.0.1', PILLOW='11.3.0', OPENCV='4.10.0.84',
    DB_PP_SHA256='9bfff8aaaca103dd08d7565f2b7a5410ab20a6427fe1f0baad5a5d04d207c8b8',
    BUNDLE_MANIFEST_SHA256='e06ef54e720b43bcb658fd734291fb1b6e4a08db13a8bd4048f16cfd3a2dc573',   # = dump_identity
)
N_TIMED, N_WARMUP, PASSES = 200, 20, 2      # KHÔNG đổi: tập ảnh cố định theo seed, đã khớp với dataset

W = Path('/kaggle/working'); OUT = W/'cs431_out'; LOGS = OUT/'logs'
T = Path('/tmp/cs431')                       # TẠM: mất khi phiên kết thúc
for p in (OUT, LOGS, T):
    p.mkdir(parents=True, exist_ok=True)

def fail(msg):
    raise SystemExit('LỖI: ' + msg)

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

# ---- tìm dataset ----
found = sorted(Path('/kaggle/input').glob('**/KAGGLE_MANIFEST.json'))
if not found:
    zips = [z for z in Path('/kaggle/input').glob('**/*.zip') if 'csb1' in z.name.lower()]
    if not zips:
        fail('Không thấy dataset. Bấm "Add Input" ở cột phải và chọn dataset chứa cs431-csb1-latency.zip.')
    with zipfile.ZipFile(zips[0]) as zf:
        zf.extractall(T/'dataset')
    found = sorted((T/'dataset').glob('**/KAGGLE_MANIFEST.json'))
if len(found) != 1:
    fail(f'Thấy {len(found)} KAGGLE_MANIFEST.json; chỉ gắn một dataset cs431-csb1-latency.')
DS = found[0].parent
manifest = json.loads(found[0].read_text())
bad = [rel for rel, info in manifest['files'].items()
       if rel != 'KAGGLE_MANIFEST.json' and (not (DS/rel).is_file() or sha256(DS/rel) != info['sha256'])]
if bad:
    fail(f'{len(bad)} file trong dataset thiếu hoặc sai SHA, ví dụ {bad[:3]}. Tải lại dataset.')
if (manifest['n'], manifest['warmup']) != (N_TIMED, N_WARMUP):
    fail('Dataset được tạo cho tập ảnh khác cấu hình notebook.')
print(f'Dataset: {DS}  ({len(manifest["files"])} file, SHA khớp hết)')

# ---- code của mình -> thư mục tạm (dataset chỉ đọc) ----
CODE = T/'code'
if CODE.exists():
    shutil.rmtree(CODE)
shutil.copytree(DS/'code', CODE)
code_sha = hashlib.sha256((DS/'code/CODE_MANIFEST.json').read_bytes()).hexdigest()
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
import importlib
import mixab.colab_helpers as ch
importlib.reload(ch)

# ---- GPU ----
gpus = (ch.output_of(['nvidia-smi', '--query-gpu=index,name,driver_version,memory.total', '--format=csv,noheader']) or '').splitlines()
if not gpus:
    fail('Không thấy GPU. Settings -> Accelerator -> GPU T4 x2 (hoặc GPU T4), rồi chạy lại Ô 1.')
print('GPU của máy:'); [print('  ', g) for g in gpus]
name0 = gpus[0].split(',')[1].strip()
GPU = ch.gpu_label(name0)
if GPU != 'T4':
    print(f'CẢNH BÁO: GPU 0 là {name0}, không phải T4. Kết quả vẫn đo được nhưng không phải cấu hình đã thống nhất.')
CONFIG_NOTE = f'{len(gpus)} GPU trên máy; chỉ dùng GPU 0 ({name0}) qua CUDA_VISIBLE_DEVICES=0'
print(CONFIG_NOTE)

PY = sys.executable
MV2, MOCR = T/'MONKEY_V2', T/'MonkeyOCRv2'
BUNDLE = MV2/'docs/handoff/artifacts/pr_quad_audit_v1'
WEIGHTS, CLEAN, DUMP = DS/'weights', DS/'clean', DS/'dump'
ENCODER = T/'encoder'                         # bản sao ghi được (dataset chỉ đọc)
if not (ENCODER/'model.safetensors').is_file():
    shutil.copytree(DS/'encoder', ENCODER, dirs_exist_ok=True)
PREPARED, CONFIG_PATH = T/'prepared', T/'inference_config_kaggle.py'
ENV = dict(MONKEY_V2_ROOT=str(MV2), PYTHONPATH=str(CODE), CUDA_VISIBLE_DEVICES='0')
print(f'Python {sys.version.split()[0]} · code manifest {code_sha[:12]}… · kết quả: {OUT}')
''')

md(r'''
## Ô 2 · Cài môi trường (cùng phiên bản với lần chạy dump T4 trên Colab)

Thứ tự: torch 2.9.0 cu128 → mmengine 0.10.7 → **MMCV 2.0.1** (dùng wheel trong dataset nếu cùng phiên bản Python, nếu không thì **build từ source**) → `install.sh` của MonkeyOCRv2 (MMOCR 1.0.1 + add-on) → ghim Pillow 11.3.0 và OpenCV 4.10.0.84.
- Build MMCV có thể mất **20–40 phút** (ước lượng, chưa đo trên Kaggle). Wheel build xong được chép vào `/kaggle/working/cs431_out/wheels/`: **tải về và thêm vào dataset** để lần sau khỏi build.
- Cần `nvcc` CUDA **12.x** để build. Ô này dùng toolkit có sẵn trên Kaggle (ưu tiên 12.8); nếu không có thì cài `cuda-toolkit-12-8` (không cài driver).
- Kiểm cuối chạy trong tiến trình mới: phiên bản, `MonkeyOCRv2ViTAEBackbone` đã đăng ký, SHA của DBPostprocessor phải đúng bản Tuấn ghim.
- Nếu Kaggle báo cần *Restart*, restart rồi chạy lại **Ô 1** và **Ô 2** (phần đã cài sẽ bỏ qua).
''')

code(r'''
# ===================== Ô 2 · CÀI MÔI TRƯỜNG =====================
import platform, re
LOG = LOGS/f'install_{GPU}.log'
pyver = sys.version_info
if not ((3, 10) <= pyver[:2] <= (3, 13)):
    print(f'CẢNH BÁO: Python {platform.python_version()} chưa được kiểm với MMCV 2.0.1.')

def ver(name):
    return ch.package_version(name)

def pip(*args):
    ch.run([PY, '-m', 'pip', 'install', '-q', *args], LOG, title='pip install ' + ' '.join(args))

if ver('torch') != f"{PIN['TORCH']}+cu128" or ver('torchvision') != f"{PIN['TORCHVISION']}+cu128":
    pip(f"torch=={PIN['TORCH']}", f"torchvision=={PIN['TORCHVISION']}", '--index-url', PIN['TORCH_INDEX'])
if ver('torchaudio') != f"{PIN['TORCHAUDIO']}+cu128":
    pip('--no-deps', f"torchaudio=={PIN['TORCHAUDIO']}+cu128", '--index-url', PIN['TORCH_INDEX'])
cuda_ok = ch.output_of([PY, '-c', 'import torch;print(torch.version.cuda, torch.cuda.is_available())'], env=ENV)
print('torch CUDA:', cuda_ok)
if not cuda_ok or 'True' not in cuda_ok:
    fail('torch không dùng được CUDA (driver Kaggle quá cũ cho cu128?). Gửi log: ' + str(LOG))
if ver('mmengine') != PIN['MMENGINE']:
    pip(f"mmengine=={PIN['MMENGINE']}")
pip('setuptools<81', 'wheel', 'ninja')

def mmcv_ok():
    if ver('mmcv') != PIN['MMCV']:
        return False
    probe = subprocess.run([PY, '-c', 'from mmcv.ops import nms; print("IMPORT_NMS_OK")'], capture_output=True, text=True)
    with LOG.open('a', encoding='utf-8') as log:
        log.write('\nMMCV import probe:\n' + probe.stdout + probe.stderr)
    return probe.returncode == 0 and 'IMPORT_NMS_OK' in probe.stdout.splitlines()

def find_toolkit():
    """CUDA 12.x toolkit with nvcc + cuda.h; prefer 12.8. torch only refuses a different MAJOR version."""
    import glob
    homes = ['/usr/local/cuda-12.8', os.environ.get('CUDA_HOME'), '/usr/local/cuda', *sorted(glob.glob('/usr/local/cuda-12*'))]
    nvcc = shutil.which('nvcc')
    if nvcc:
        homes.append(str(Path(nvcc).resolve().parents[1]))
    best = None
    for home in homes:
        if not home or not (Path(home)/'bin/nvcc').is_file() or not (Path(home)/'include/cuda.h').is_file():
            continue
        m = re.search(r'release (\d+)\.(\d+)', ch.output_of([str(Path(home)/'bin/nvcc'), '--version']) or '')
        if m and m.group(1) == '12':
            cand = (int(m.group(2)) == 8, int(m.group(2)), str(Path(home).resolve()))
            best = max(best, cand) if best else cand
    return best

tag = f'cp{pyver[0]}{pyver[1]}'
if not mmcv_ok():
    wheels = sorted((DS/'wheels').glob(f"mmcv-{PIN['MMCV']}-{tag}-*.whl")) + sorted(OUT.glob(f"wheels/mmcv-{PIN['MMCV']}-{tag}-*.whl"))
    if wheels:
        print('Cài MMCV từ wheel có sẵn:', wheels[0].name)
        pip('--no-deps', str(wheels[0]))
    else:
        tk = find_toolkit()
        if not tk:
            print('Không có CUDA toolkit 12.x. Cài cuda-toolkit-12-8 từ NVIDIA (không cài driver)...')
            distro = platform.freedesktop_os_release()
            repo = 'ubuntu' + distro.get('VERSION_ID', '22.04').replace('.', '')
            from urllib.request import urlretrieve
            keyring = T/'cuda-keyring_1.1-1_all.deb'
            urlretrieve(f'https://developer.download.nvidia.com/compute/cuda/repos/{repo}/x86_64/cuda-keyring_1.1-1_all.deb', keyring)
            ch.run(['dpkg', '-i', keyring], LOG)
            ch.run(['apt-get', 'update'], LOG)
            ch.run(['apt-get', 'install', '-y', '--no-install-recommends', 'cuda-toolkit-12-8'], LOG, env={'DEBIAN_FRONTEND': 'noninteractive'})
            tk = find_toolkit()
            if not tk:
                fail('Đã cài nhưng vẫn không thấy nvcc 12.x; xem ' + str(LOG))
        cuda_home = tk[2]
        print(f'Dùng CUDA toolkit {cuda_home} (12.{tk[1]})' + ('' if tk[0] else ' — khác 12.8, torch chỉ cảnh báo khác minor version'))
        src = T/'mmcv_src'
        if not (src/'setup.py').is_file():
            ch.run(['git', 'clone', '-b', f"v{PIN['MMCV']}", '--depth', '1', 'https://github.com/open-mmlab/mmcv.git', src], LOG)
        setup = (src/'setup.py').read_text()
        setup = setup.replace('-std=c++14', '-std=c++17')
        old_v = "        exec(compile(f.read(), version_file, 'exec'))\n    return locals()['__version__']"
        new_v = "        namespace = {}\n        exec(compile(f.read(), version_file, 'exec'), namespace)\n    return namespace['__version__']"
        if old_v in setup:
            setup = setup.replace(old_v, new_v)
        elif new_v not in setup:
            fail('MMCV setup.py khác cấu trúc đã kiểm.')
        (src/'setup.py').write_text(setup)
        print(f'Build MMCV ({os.cpu_count()} CPU). Có thể mất lâu...')
        build_env = dict(CUDA_HOME=cuda_home, MMCV_WITH_OPS='1', MAX_JOBS=str(os.cpu_count() or 2),
                         PATH=f"{cuda_home}/bin:{os.environ.get('PATH', '')}", TORCH_CUDA_ARCH_LIST='7.5;8.0')
        wheel_tmp = T/'wheelhouse'
        ch.run([PY, '-m', 'pip', 'wheel', '-v', '--no-build-isolation', '--no-deps', '-w', wheel_tmp, src], LOG,
               env=build_env, title='build MMCV wheel')
        built = sorted(wheel_tmp.glob('mmcv-*.whl'))
        if not built:
            fail('Build MMCV không tạo ra wheel; xem ' + str(LOG))
        (OUT/'wheels').mkdir(exist_ok=True)
        shutil.copy2(built[-1], OUT/'wheels'/built[-1].name)
        print('Đã lưu wheel để tải về:', OUT/'wheels'/built[-1].name)
        pip('--no-deps', str(built[-1]))
    if not mmcv_ok():
        fail('MMCV chưa import được mmcv.ops.nms; xem ' + str(LOG))

if not (MOCR/'.git').is_dir():
    ch.run(['git', 'clone', PIN['MONKEYOCRV2_URL'], MOCR], LOG)
ch.run(['git', '-C', MOCR, 'checkout', '--detach', PIN['MONKEYOCRV2_COMMIT']], LOG)
addon = MOCR/'detection/mmocr'
mmocr_root = addon/'mmocr'
backbone_file = mmocr_root/'mmocr/models/common/backbones/monkeyocr_v2_vitae.py'
mmocr_loc = ch.output_of([PY, '-c', 'import mmocr,os;print(os.path.dirname(mmocr.__file__))'])
if not (ver('mmocr') == PIN['MMOCR'] and backbone_file.is_file() and mmocr_loc and str(mmocr_root) in mmocr_loc):
    if not (mmocr_root/'.git').is_dir():
        ch.run(['git', 'clone', '--branch', 'v1.0.1', '--depth', '1', 'https://github.com/open-mmlab/mmocr.git', mmocr_root], LOG)
    setup_path = mmocr_root/'setup.py'
    setup = setup_path.read_text()
    old_exec = "        exec(compile(f.read(), version_file, 'exec'))"
    new_exec = "        namespace = {}\n        exec(compile(f.read(), version_file, 'exec'), namespace)"
    if old_exec in setup:
        setup = setup.replace(old_exec, new_exec)
        setup = setup.replace("locals()['short_version']", "namespace['short_version']").replace("locals()['__version__']", "namespace['__version__']")
        setup_path.write_text(setup)
    elif new_exec not in setup:
        fail('MMOCR setup.py khác cấu trúc đã kiểm.')
    ch.run(['bash', 'install.sh'], LOG, cwd=addon, title='MonkeyOCRv2 detection/mmocr/install.sh',
           env=dict(PATH=f"{Path(PY).parent}:{os.environ.get('PATH', '')}", PIP_VERBOSE='1'))
if ver('Pillow') != PIN['PILLOW']:
    pip(f"Pillow=={PIN['PILLOW']}")
if ver('opencv-python') != PIN['OPENCV'] or any(ver(n) for n in ('opencv-python-headless', 'opencv-contrib-python')):
    ch.run([PY, '-m', 'pip', 'uninstall', '-y', 'opencv-python-headless', 'opencv-contrib-python', 'opencv-contrib-python-headless'], LOG)
    pip('--force-reinstall', '--no-deps', f"opencv-python=={PIN['OPENCV']}")
if ver('shapely') is None:
    pip('shapely')

VERIFY = """
import hashlib, importlib.metadata as m, importlib.util, json, torch
from mmocr.utils import register_all_modules
from mmocr.registry import MODELS
from mmcv.ops import nms
register_all_modules(init_default_scope=True)
spec = importlib.util.find_spec('mmocr.models.textdet.postprocessors.db_postprocessor')
out = dict(versions={k: m.version(k) for k in ('torch','torchvision','mmengine','mmcv','mmdet','mmocr','Pillow','opencv-python','numpy','shapely')},
           cuda=torch.cuda.is_available(), device_count_visible=torch.cuda.device_count(), device=torch.cuda.get_device_name(0),
           backbone_registered=MODELS.get('MonkeyOCRv2ViTAEBackbone') is not None,
           db_postprocessor_sha256=hashlib.sha256(open(spec.origin,'rb').read()).hexdigest())
print(json.dumps(out))
"""
raw = ch.output_of([PY, '-c', VERIFY], env=ENV)
if not raw:
    fail('Kiểm cuối môi trường lỗi khi import; xem ' + str(LOG))
env_report = json.loads(raw.splitlines()[-1])
(OUT/f'env_{GPU}.json').write_text(json.dumps(dict(env_report, pins=PIN, gpus=gpus, config=CONFIG_NOTE), indent=1))
print(json.dumps(env_report, indent=1))
problems = []
if env_report['db_postprocessor_sha256'] != PIN['DB_PP_SHA256']:
    problems.append('DBPostprocessor khác bản Tuấn ghim')
if not env_report['backbone_registered']:
    problems.append('MonkeyOCRv2ViTAEBackbone chưa đăng ký')
if env_report['versions']['Pillow'] != PIN['PILLOW']:
    problems.append('Pillow không phải 11.3.0 (cần để tạo LR2 đúng byte)')
if env_report['device_count_visible'] != 1:
    problems.append('Tiến trình đo thấy nhiều hơn 1 GPU')
if problems:
    fail('Môi trường chưa đạt: ' + '; '.join(problems))
print('\nÔ 2 xong.')
''')

md(r'''
## Ô 3 · Dữ liệu: repo của Tuấn, config, checkpoint, ảnh LR2

1. Clone `MONKEY_V2` ở commit ghim `dc19b85` (commit của dump T4). Bundle dữ liệu nằm trong Git thường, **không cần Git LFS**. Kiểm SHA `bundle_manifest.json` trùng với dump.
2. Tạo config chỉ đổi **2 đường dẫn encoder** sang thư mục encoder trong dataset.
3. Kiểm SHA 2 checkpoint (B-Clean = `alpha_0p00.pth`, B1 = `alpha_1p00.pth`) với `merge_manifest.json`.
4. Tạo **220 ảnh LR2** từ ảnh Clean bằng đúng hàm `native_lr2` của Tuấn (Pillow 11.3.0) và kiểm SHA từng ảnh với bundle.
5. Chạy test CPU của `mixab`.
''')

code(r'''
# ===================== Ô 3 · DỮ LIỆU =====================
LOG = LOGS/f'data_{GPU}.log'
if not (MV2/'.git').is_dir():
    ch.run(['git', 'clone', PIN['MONKEY_V2_URL'], MV2], LOG, env=dict(GIT_LFS_SKIP_SMUDGE='1'))
ch.run(['git', '-C', MV2, 'checkout', '--detach', PIN['MONKEY_V2_COMMIT']], LOG, env=dict(GIT_LFS_SKIP_SMUDGE='1'))
dirty = ch.output_of(['git', '-C', MV2, 'status', '--porcelain', '--untracked-files=no'])
if dirty:
    fail(f'Checkout MONKEY_V2 bị sửa:\n{dirty}')
if sha256(BUNDLE/'bundle_manifest.json') != PIN['BUNDLE_MANIFEST_SHA256']:
    fail('bundle_manifest.json khác bản của dump T4.')
dump_id = json.loads((DUMP/'dump_identity.json').read_text())
if dump_id['bundle_manifest_sha256'] != PIN['BUNDLE_MANIFEST_SHA256'] or dump_id['monkey_v2']['commit'] != PIN['MONKEY_V2_COMMIT']:
    fail('Dump trong dataset không khớp commit/bundle đang dùng.')
patched = ch.patch_inference_config((MV2/'docs/handoff/inference_config.py').read_text(encoding='utf-8'), ENCODER)
CONFIG_PATH.write_text(patched, encoding='utf-8')
if hashlib.sha256((MV2/'docs/handoff/inference_config.py').read_bytes()).hexdigest() != dump_id['model_config_sha256'] and \
        sha256(CONFIG_PATH) != dump_id['model_config_sha256']:
    print('Ghi chú: SHA config đã vá khác dump (khác đường dẫn encoder là bình thường); nội dung khác giữ nguyên.')
m = json.loads((WEIGHTS/'merge_manifest.json').read_text())
for model in m['models']:
    if model['alpha'] in ('0', '1'):
        p = WEIGHTS/Path(model['checkpoint']).name
        if sha256(p) != model['checkpoint_sha256']:
            fail(f'{p.name} sai SHA so với merge_manifest.json')
print('Checkpoint B-Clean/B1: SHA khớp.')
ch.run([PY, '-m', 'mixab.csb1_latency', 'prepare', '--bundle', BUNDLE, '--clean-root', CLEAN, '--prepared-root', PREPARED,
        '--n', str(N_TIMED), '--warmup', str(N_WARMUP), '--subset-out', OUT/'subset.json'],
       LOG, cwd=CODE, env=ENV, title='prepare LR2 subset')
ch.run([PY, '-m', 'unittest', 'discover', '-s', 'tests', '-t', '.'], LOG, cwd=CODE, env=ENV, title='mixab unit tests')
print('\nÔ 3 xong: repo + config + checkpoint + 220 ảnh LR2 (SHA khớp) + test PASS.')
''')

md(r'''
## Ô 4 · Kiểm nhanh GPU trước khi đo

In trạng thái GPU 0 (nhiệt độ, xung nhịp, bộ nhớ đang dùng). Nếu GPU 0 đang bận bởi tiến trình khác (bộ nhớ dùng > 500 MiB) thì dừng: restart phiên rồi chạy lại.
''')

code(r'''
# ===================== Ô 4 · TRẠNG THÁI GPU =====================
state = ch.output_of(['nvidia-smi', '--id=0', '--query-gpu=name,temperature.gpu,clocks.sm,clocks.max.sm,memory.used,utilization.gpu',
                      '--format=csv,noheader']) or ''
print('GPU 0:', state)
used = re.search(r',\s*(\d+)\s*MiB', state)
if used and int(used.group(1)) > 500:
    fail('GPU 0 đang có tiến trình khác giữ bộ nhớ. Restart phiên (Run -> Restart) rồi chạy lại từ Ô 1.')
print('Sẵn sàng đo.')
''')

md(r'''
## Ô 5 · Đo latency B1 vs CS-B1@0.7

`mixab.csb1_latency run`:
- Nạp **B1 trước**, chạy 20 ảnh khởi động → **bộ nhớ đỉnh của B1 một mình**. Sau đó nạp B-Clean, chạy CS-B1 trên 20 ảnh khởi động → **bộ nhớ đỉnh của CS-B1** (giữ 2 model). Kiểm luôn: dùng chung ảnh đã giải mã cho B-Clean cho ra box **giống hệt** khi B-Clean tự giải mã.
- Đo **200 ảnh × 2 lượt**, mỗi ảnh chạy cả hai cách, **xen kẽ thứ tự** (lượt 2 đảo lại), `torch.cuda.synchronize()` trước và sau mỗi lần đo. Kiểm SHA ảnh và đọc file làm trước, không tính vào thời gian.
- So số box với dump T4 cũ trên cùng ảnh (chỉ báo cáo, không chặn).
- Ước lượng 10–15 phút (chưa đo). Kết quả: `cs431_out/csb1_latency_T4/latency.json` + `samples.csv`. Đã có `latency.json` thì chỉ đọc lại.
''')

code(r'''
# ===================== Ô 5 · ĐO LATENCY =====================
LAT = OUT/f'csb1_latency_{GPU}'
if not (LAT/'latency.json').is_file():
    log = LOGS/f'latency_{GPU}_{time.strftime("%Y%m%d-%H%M%S")}.log'
    ch.run([PY, '-m', 'mixab.csb1_latency', 'run', '--bundle', BUNDLE, '--clean-root', CLEAN, '--prepared-root', PREPARED,
            '--model-config', CONFIG_PATH, '--weights-dir', WEIGHTS, '--n', str(N_TIMED), '--warmup', str(N_WARMUP),
            '--passes', str(PASSES), '--dump', DUMP, '--output', LAT], log, cwd=CODE, env=ENV, title=f'csb1 latency on {GPU}')
r = json.loads((LAT/'latency.json').read_text())
print(f"GPU: {r['environment']['gpu']} · {CONFIG_NOTE}")
for arm in ('B1', 'CSB1'):
    t = r['ms'][arm]['total']
    print(f"  {arm:5} median {t['median']:.1f} ms · p10 {t['p10']:.1f} · p90 {t['p90']:.1f} · mean {t['mean']:.1f} (n={t['n']})")
    for st, q in r['ms'][arm]['stages'].items():
        print(f"         {st:7} median {q['median']:.1f} ms")
p = r['paired']
print(f"CS-B1 / B1 (tỉ số median): {p['ratio_of_medians']:.2f} [CI95 {p['ratio_of_medians_ci95'][0]:.2f}; {p['ratio_of_medians_ci95'][1]:.2f}]"
      f" · thêm trung bình {p['mean_extra_ms']:.1f} ms/ảnh")
mem = r['memory']
print(f"Bộ nhớ đỉnh (allocated): B1 {mem['B1_deployment_peak']['allocated_mib']:.0f} MiB · "
      f"CS-B1 {mem['CSB1_deployment_peak']['allocated_mib']:.0f} MiB")
print('Dùng chung ảnh đã giải mã:', r['shared_decode_check'])
print('So với dump T4 (Colab):', r.get('check_vs_t4_dump'))
''')

md(r'''
## Ô 6 · Đóng gói kết quả để gửi lại

Tạo `/kaggle/working/cs431_csb1_latency_<GPU>_<thời gian>.zip` gồm `latency.json`, `samples.csv`, `subset.json`, `env_*.json` và log. Tải file này ở khung **Output** bên phải (hoặc trong cây thư mục `/kaggle/working`). Nếu Ô 2 vừa build MMCV, tải thêm thư mục `cs431_out/wheels/`.
''')

code(r'''
# ===================== Ô 6 · ĐÓNG GÓI =====================
pack = W/f'cs431_csb1_latency_{GPU}_{time.strftime("%Y%m%d-%H%M%S")}.zip'
with zipfile.ZipFile(pack, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.suffix != '.whl':
            zf.write(p, p.relative_to(OUT).as_posix())
    zf.writestr('code_manifest_sha256.txt', code_sha + '\n')
print('Đã đóng gói:', pack, f'({pack.stat().st_size/1024:.0f} KiB)')
''')


def main():
    sys.stdout.reconfigure(errors='backslashreplace')
    notebook = dict(nbformat=4, nbformat_minor=5,
                    metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python'),
                                  language_info=dict(name='python'),
                                  kaggle=dict(accelerator='nvidiaTeslaT4', isInternetEnabled=True, isGpuEnabled=True)),
                    cells=[dict(c, id=f'cell-{i:02d}',
                                source=[line + '\n' for line in c['source'].split('\n')][:-1] +
                                [c['source'].split('\n')[-1]]) for i, c in enumerate(CELLS)])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(f'{OUT}: {len(CELLS)} cells')


if __name__ == '__main__':
    main()
