"""Notebook helpers: no-overwrite naming, config patch, image subset, probe summary."""
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import sys
import unittest
from unittest.mock import patch
import zipfile

from mixab import colab_helpers as ch
from mixab.monkey import monkey_root

ROOT = monkey_root(required=False)
BUNDLE = (ROOT or Path('.'))/'docs/handoff/artifacts/pr_quad_audit_v1'
CONFIG = (ROOT or Path('.'))/'docs/handoff/inference_config.py'


class HelpersTest(unittest.TestCase):
    def test_cuda_selection_ignores_path_13_and_checks_actual_compiler(self):
        with TemporaryDirectory() as tmp:
            roots = [Path(tmp)/name for name in ('default13', 'cuda-12.8', 'actual12')]
            for root in roots:
                (root/'bin').mkdir(parents=True)
                (root/'bin/nvcc').touch()
                (root/'include').mkdir()
                (root/'include/cuda.h').touch()
            versions = dict(zip([str(r/'bin/nvcc') for r in roots], ['13.0', '12.9', '12.8']))
            with patch.object(ch, 'output_of', side_effect=lambda cmd: 'release ' + versions[str(cmd[0])] + ', V'):
                self.assertEqual(ch.find_cuda_toolkit([None, *roots]), str(roots[2].resolve()))
                self.assertIsNone(ch.find_cuda_toolkit(roots[:2]))
                (roots[2]/'include/cuda.h').unlink()
                self.assertIsNone(ch.find_cuda_toolkit(roots))

    def test_gpu_labels(self):
        self.assertEqual(ch.gpu_label('Tesla T4'), 'T4')
        self.assertEqual(ch.gpu_label('NVIDIA A100-SXM4-40GB'), 'A100-SXM4-40GB')
        self.assertEqual(ch.gpu_label('NVIDIA L4'), 'NVIDIA_L4')
        self.assertTrue(ch.gpu_supported('Tesla T4') and ch.gpu_supported('NVIDIA A100-SXM4-80GB'))
        self.assertFalse(ch.gpu_supported('NVIDIA L4'))

    def test_new_path_never_overwrites_and_move_aside(self):
        with TemporaryDirectory() as tmp:
            a = ch.new_path(tmp, 'doctor', '.json')
            a.write_text('x')
            b = ch.new_path(tmp, 'doctor', '.json')
            self.assertNotEqual(a, b)
            self.assertFalse(b.exists())
            folder = Path(tmp)/'weights'
            folder.mkdir()
            moved = ch.move_aside(folder)
            self.assertTrue(moved.exists() and not folder.exists() and 'incomplete' in moved.name)
            self.assertIsNone(ch.move_aside(folder))

    def test_run_streams_logs_and_raises_friendly_error(self):
        with TemporaryDirectory() as tmp:
            log = Path(tmp)/'logs/x.log'
            ch.run([sys.executable, '-c', 'print("xin chao")'], log)
            self.assertIn('xin chao', log.read_text(encoding='utf-8'))
            with self.assertRaises(ch.StepError) as ctx:
                ch.run([sys.executable, '-c', 'import sys; print("boom"); sys.exit(3)'], log, title='demo')
            self.assertIn('exit 3', str(ctx.exception))
            self.assertIn('boom', str(ctx.exception))

    @unittest.skipUnless(CONFIG.is_file(), 'inference_config.py not present')
    def test_patch_tuans_config_only_changes_encoder_paths(self):
        original = CONFIG.read_text(encoding='utf-8')
        patched = ch.patch_inference_config(original, '/content/drive/MyDrive/CS431_MixA/cache/encoder')
        self.assertIn("'/content/drive/MyDrive/CS431_MixA/cache/encoder/model.safetensors'", patched)
        self.assertNotIn("'pretrained/monkeyocrv2_as", patched)
        changed = [(a, b) for a, b in zip(original.splitlines(), patched.splitlines()) if a != b]
        self.assertEqual(len(changed), 2)
        self.assertEqual(len(original.splitlines()), len(patched.splitlines()))
        from mixab import doctor
        self.assertEqual(doctor.encoder_paths(patched)['weights'],
                         '/content/drive/MyDrive/CS431_MixA/cache/encoder/model.safetensors')
        with self.assertRaises(ch.StepError):
            ch.patch_inference_config(patched, '/again')

    @unittest.skipUnless((BUNDLE/'data/source_cohort.csv').is_file(), 'bundle not present')
    def test_needed_images_from_real_cohort(self):
        items = ch.needed_clean_images(BUNDLE)
        self.assertEqual(len(items), 2262)
        self.assertTrue(all(r.startswith('train_images/') and r.endswith('.jpg') for r, _ in items))

    def test_subset_extract_verify_and_cache_zip(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            payload = {f'train_images/{i}.jpg': f'img{i}'.encode() for i in range(4)}
            with zipfile.ZipFile(tmp/'big.zip', 'w') as zf:
                zf.writestr('train_images/', '')
                for name, data in payload.items():
                    zf.writestr(name, data)
                zf.writestr('train_images/unused.jpg', b'zzz')
            items = [(n, hashlib.sha256(d).hexdigest()) for n, d in list(payload.items())[:3]]
            ch.extract_members(tmp/'big.zip', [n for n, _ in items], tmp/'out')
            self.assertFalse((tmp/'out/train_images/unused.jpg').exists())
            self.assertEqual(ch.verify_images(tmp/'out', items)['ok'], 3)
            (tmp/'out/train_images/0.jpg').write_bytes(b'corrupt')
            report = ch.verify_images(tmp/'out', items + [('train_images/9.jpg', 'x')])
            self.assertEqual((report['ok'], report['bad'], report['missing']), (2, 1, 1))
            with self.assertRaises(ch.StepError):
                ch.extract_members(tmp/'big.zip', ['train_images/404.jpg'], tmp/'out')
            cache = ch.build_cache_zip(tmp/'out', [n for n, _ in items], tmp/'cache/sub.zip')
            self.assertEqual(sorted(zipfile.ZipFile(cache).namelist()), sorted(n for n, _ in items))
            with self.assertRaises(ch.StepError):
                ch.build_cache_zip(tmp/'out', [n for n, _ in items], cache)

    def test_summarize_probe(self):
        gates = dict(gpu='Tesla T4', models={
            'B-Clean': dict(gate_3a=dict(status='PASS', mismatched=0, checked=80),
                            pmap_endpoint=dict(status='PASS', max_abs_diff=0.0)),
            'B1': dict(gate_3a=dict(status='FAIL', mismatched=2, checked=80),
                       pmap_endpoint=dict(status='PASS', max_abs_diff=0.0))})
        profile = {'B-Clean': dict(tasks=80, mean_seconds_per_task=2.0, peak_gpu_mib=3000),
                   'B1': dict(tasks=80, mean_seconds_per_task=4.0, peak_gpu_mib=3500)}
        s = ch.summarize_probe(gates, profile)
        self.assertFalse(s['ready_for_full_run'])
        self.assertTrue(s['ready_for_full_run_same_gpu'])        # 3a vs H200 does not gate option (a)
        gates['models']['B1']['pmap_endpoint']['status'] = 'FAIL'
        self.assertFalse(ch.summarize_probe(gates, profile)['ready_for_full_run_same_gpu'])
        self.assertAlmostEqual(s['full_run_estimate_hours'], 3.0*2*2262*5/3600)
        self.assertEqual(s['peak_gpu_mib'], 3500)
        self.assertEqual(s['endpoints'][1]['mismatched_3a'], 2)


if __name__ == '__main__':
    unittest.main()
