"""E-BN stage 0 (CPU parts): TRAIN input checks, fixed subset, BN statistics, hooks change nothing."""
import argparse
import gzip
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from mixab import bn_recal as br
from mixab.monkey import monkey_root

ROOT = monkey_root(required=False)
BUNDLE = (ROOT or Path('.'))/'docs/handoff/artifacts/pr_quad_audit_v1'


def fake_rows(n, start=0):
    return [dict(source_image_id=f'{i:016x}', clean_relpath=f'train_images/{i:016x}.jpg',
                 clean_sha256=hashlib.sha256(f'img{i}'.encode()).hexdigest(), clean_size=[64, 48])
            for i in range(start, start + n)]


def write_manifest(path, rows, gz=False):
    raw = ''.join(json.dumps(dict(r, schema_version='textdet-paired-v1')) + '\n' for r in rows).encode()
    Path(path).write_bytes(gzip.compress(raw) if gz else raw)
    return hashlib.sha256(raw).hexdigest()


def write_id_csv(path, rows, columns=br.ID_COLUMNS):
    lines = [','.join(columns)] + [','.join(r[c] for c in columns) for r in rows]
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


class TrainInputTest(unittest.TestCase):
    def test_full_manifest_requires_tuans_sha(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/br.TRAIN_MANIFEST_NAME
            digest = write_manifest(path, fake_rows(3))
            with self.assertRaisesRegex(ValueError, 'khác mã Tuấn'):
                br.read_train_input(path)
            with patch.object(br, 'TRAIN_MANIFEST_SHA256', digest):
                info = br.read_train_input(path)
                self.assertEqual((info['kind'], info['manifest_sha256_status']), ('full_manifest', 'VERIFIED_MATCH'))
                self.assertEqual(info['rows'][0]['clean_size'], [64, 48])
                gz = Path(tmp)/(br.TRAIN_MANIFEST_NAME + '.gz')
                write_manifest(gz, fake_rows(3), gz=True)
                self.assertEqual(br.read_train_input(gz)['manifest_sha256'], digest)   # hash of the plain bytes

    def test_id_list_is_accepted_but_marked_unverifiable(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'train_ids.csv'
            write_id_csv(path, fake_rows(4))
            info = br.read_train_input(path)
            self.assertEqual(info['manifest_sha256_status'], 'NOT_VERIFIABLE_ID_LIST')
            self.assertIsNone(info['manifest_sha256'])
            self.assertEqual(len(info['rows']), 4)
            write_id_csv(path, fake_rows(2), columns=('source_image_id', 'clean_relpath'))
            with self.assertRaisesRegex(ValueError, 'thiếu cột'):
                br.read_train_input(path)
            with self.assertRaises(ValueError):
                br.read_train_input(Path(tmp)/'missing.csv')
            (Path(tmp)/'x.txt').write_text('a')
            with self.assertRaises(ValueError):
                br.read_train_input(Path(tmp)/'x.txt')

    def test_validate_rows(self):
        rows = fake_rows(5)
        validation = {r['source_image_id']: r for r in fake_rows(3, start=100)}
        with patch.object(br, 'TRAIN_SOURCES', 5):
            self.assertEqual(br.validate_train_rows(rows, validation)['status'], 'PASS')
            self.assertEqual(br.validate_train_rows(rows[:4], validation)['status'], 'FAIL')        # count
            dup = rows[:4] + [dict(rows[0])]
            self.assertIn('trùng', ' '.join(br.validate_train_rows(dup, validation)['problems']))
            leak = rows[:4] + [dict(fake_rows(1, start=100)[0])]                                     # validation source
            report = br.validate_train_rows(leak, validation)
            self.assertEqual((report['validation_overlap']['id'], report['validation_overlap']['clean_sha']), (1, 1))
            bad = [dict(r) for r in rows]
            bad[1]['clean_relpath'] = '../test_images/x.jpg'
            bad[2]['clean_sha256'] = 'xyz'
            self.assertEqual(len(br.validate_train_rows(bad, validation)['problems']), 2)

    def test_subset_is_fixed_and_order_independent(self):
        rows = fake_rows(50)
        a = br.select_subset(rows, n=10)
        b = br.select_subset(list(reversed(rows)), n=10)
        self.assertEqual([r['source_image_id'] for r in a], [r['source_image_id'] for r in b])
        self.assertEqual(len({r['source_image_id'] for r in a}), 10)
        self.assertNotEqual([r['source_image_id'] for r in a], [r['source_image_id'] for r in br.select_subset(rows, 10, seed=1)])

    @unittest.skipUnless((BUNDLE/'data/validation_remaining.jsonl.gz').is_file(), 'bundle not present')
    def test_check_train_against_real_validation(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_id_csv(tmp/'train_ids.csv', fake_rows(1200))
            args = argparse.Namespace(train_input=tmp/'train_ids.csv', bundle=BUNDLE, output=tmp/'check')
            with patch.object(br, 'TRAIN_SOURCES', 1200):
                br.check_train(args)
                record = json.loads((tmp/'check/train_check.json').read_text(encoding='utf-8'))
                self.assertEqual(record['manifest_sha256_status'], 'NOT_VERIFIABLE_ID_LIST')
                self.assertEqual(record['check']['validation_overlap']['checked_against'], 2262)
                self.assertEqual(record['subset']['sources'], br.SUBSET_SOURCES)
                br.check_train(args)                                   # re-run: same subset accepted
                write_id_csv(tmp/'train_ids.csv', fake_rows(1200, start=5))
                with self.assertRaisesRegex(ValueError, 'Không ghi đè'):
                    br.check_train(args)
                # a real validation source inside "TRAIN" must stop before any image work
                val = json.loads(gzip.decompress((BUNDLE/'data/validation_remaining.jsonl.gz').read_bytes())
                                 .decode().splitlines()[0])
                leak = fake_rows(1199) + [dict(source_image_id=val['source_image_id'], clean_relpath=val['clean_relpath'],
                                               clean_sha256=val['clean_sha256'])]
                write_id_csv(tmp/'leak.csv', leak)
                with self.assertRaisesRegex(ValueError, 'không hợp lệ'):
                    br.check_train(argparse.Namespace(train_input=tmp/'leak.csv', bundle=BUNDLE, output=tmp/'c2'))


@unittest.skipUnless(ROOT is not None, 'MONKEY_V2 not present')
class PrepareImagesTest(unittest.TestCase):
    """Clean SHA check, LR2 written once, LR2 SHA recorded and re-checked on a new VM.
    Local Pillow is not 11.3.0, so Tuan's native_lr2 is replaced by a stand-in here; the
    real function (and its Pillow 11.3.0 requirement) runs only on Colab."""

    def test_prepare_images(self):
        from PIL import Image
        from mixab.monkey import use_monkey
        use_monkey()
        import rev4.training_data as td

        def stand_in(image):
            return image.convert('RGB').resize(td.lr2_dimensions(image.size), Image.Resampling.BICUBIC)

        with TemporaryDirectory() as tmp, patch.object(td, 'native_lr2', stand_in):
            tmp = Path(tmp)
            rows = []
            for i in range(3):
                rel = f'train_images/{i:016x}.jpg'
                (tmp/'clean/train_images').mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (20 + i, 10), (i*40, 0, 0)).save(tmp/'clean'/rel, format='JPEG')
                rows.append(dict(source_image_id=f'{i:016x}', clean_relpath=rel,
                                 clean_sha256=hashlib.sha256((tmp/'clean'/rel).read_bytes()).hexdigest(),
                                 clean_size=[20 + i, 10]))
            br.write_csv(tmp/'subset.csv', rows, ['source_image_id', 'clean_relpath', 'clean_sha256', 'clean_size'])
            args = argparse.Namespace(subset=tmp/'subset.csv', clean_root=tmp/'clean', lr2_root=tmp/'lr2',
                                      record=tmp/'rec/subset_images.csv')
            br.prepare_images(args)
            record = br.read_csv(tmp/'rec/subset_images.csv')
            self.assertEqual(len(record), 3)
            with Image.open(tmp/'lr2'/record[1]['lr2_name']) as im:
                self.assertEqual(im.size, (10, 5))
            for p in (tmp/'lr2').iterdir():                                  # new VM: LR2 regenerated
                p.unlink()
            br.prepare_images(args)                                          # same bytes -> accepted
            (tmp/'lr2'/record[0]['lr2_name']).write_bytes(b'not the same png')
            with self.assertRaisesRegex(ValueError, 'LR2 tạo lại khác'):
                br.prepare_images(args)
            (tmp/'lr2'/record[0]['lr2_name']).unlink()                       # regenerate the good one
            (tmp/'clean'/rows[2]['clean_relpath']).write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'sai SHA'):
                br.prepare_images(args)


class FakeDetector(nn.Module):
    """Same module names as MMOCR DBHead: det_head.binarize.{1,4} are BatchNorm2d; threshold unused."""

    def __init__(self):
        super().__init__()
        self.det_head = nn.Module()
        self.det_head.binarize = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(),
                                               nn.ConvTranspose2d(4, 4, 2, 2), nn.BatchNorm2d(4), nn.ReLU())
        self.det_head.threshold = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4))
        for bn in (self.det_head.binarize[1], self.det_head.binarize[4]):
            bn.running_mean.uniform_(-1, 1)
            bn.running_var.uniform_(.5, 2)

    def forward(self, x):
        return self.det_head.binarize(x)


class StatisticsTest(unittest.TestCase):
    def test_hooks_measure_bn_inputs_without_changing_the_model(self):
        torch.manual_seed(0)
        net = FakeDetector().eval()
        before = br.state_digest(net)
        acc = br.Accumulator({n: 4 for n in br.BN_LAYERS})
        cond = ['clean']
        handles = br.attach_hooks(net, acc, cond)
        images = {c: [torch.randn(1, 3, 6 + i, 5) for i in range(3)] for c in br.CONDITIONS}
        captured = {c: [] for c in br.CONDITIONS}
        with torch.inference_mode():
            for c in br.CONDITIONS:
                cond[0] = c
                for x in images[c]:
                    net(x)
                    captured[c].append(net.det_head.binarize[0](x).double())
        for h in handles:
            h.remove()
        self.assertEqual(br.state_digest(net), before)           # eval + pre-hooks: no buffer update
        flat = [x[0].reshape(4, -1) for c in br.CONDITIONS for x in captured[c]]
        mean, var, n = acc.stats('det_head.binarize.1')
        self.assertEqual(n, 6)
        np.testing.assert_allclose(mean, np.mean([f.mean(1).numpy() for f in flat], axis=0), rtol=1e-12)
        np.testing.assert_allclose(var, np.mean([f.var(1, unbiased=True).numpy() for f in flat], axis=0), rtol=1e-12)
        clean_mean, _, n_clean = acc.stats('det_head.binarize.1', ('clean',))
        self.assertEqual(n_clean, 3)
        self.assertGreater(float(acc.data['det_head.binarize.4|lr2|n'][0]), 0)
        with TemporaryDirectory() as tmp:                         # resume file round trip
            acc.save(Path(tmp)/'a.npz')
            other = br.Accumulator({n: 4 for n in br.BN_LAYERS})
            other.load(Path(tmp)/'a.npz')
            np.testing.assert_array_equal(other.stats('det_head.binarize.1')[0], mean)
        with self.assertRaises(ValueError):
            br.attach_hooks(nn.Sequential(nn.Conv2d(1, 1, 1)), acc, cond)

    def test_shift_metrics_and_nonlinearity(self):
        m = br.shift_metrics(np.array([1., 0.]), np.array([4., 1.]), np.array([0., 0.]), np.array([1., 1.]), eps=0)
        self.assertAlmostEqual(m['z_mean'], .5)
        self.assertAlmostEqual(m['z_max'], 1.)
        self.assertAlmostEqual(m['logvar_max'], np.log(4))
        measured = {Fraction(0): (np.zeros(2), np.ones(2)), Fraction(1): (np.full(2, 2.), np.full(2, 3.)),
                    Fraction(1, 2): (np.ones(2), np.full(2, 2.))}
        self.assertAlmostEqual(br.nonlinearity(measured, Fraction(1, 2))['z_mean'], 0., places=6)
        measured[Fraction(1, 2)] = (np.full(2, 1 + np.sqrt(2.)), np.full(2, 2.))
        self.assertAlmostEqual(br.nonlinearity(measured, Fraction(1, 2))['z_mean'], 1., places=5)

    def test_proposed_gate(self):
        def entry(z):
            return {'det_head.binarize.1': {'all': dict(z_mean=z)}}
        g = br.stage0_gate({'0': entry(.1), '1/2': entry(.25), '1': entry(.12)})
        self.assertEqual(g['status'], 'PROPOSED_NOT_APPROVED')
        self.assertTrue(g['signal'])
        self.assertEqual(g['alphas_with_signal'], ['1/2'])
        self.assertFalse(br.stage0_gate({'0': entry(.1), '1/2': entry(.2), '1': entry(.12)})['signal'])
        with self.assertRaises(ValueError):
            br.stage0_gate({'1/2': entry(.2), '1': entry(.1)})


class _FakeInferencer:
    """Stands in for MMOCR TextDetInferencer: preprocess -> data_preprocessor -> model(mode='tensor')."""
    calls = 0
    fail_after = None

    def __init__(self, checkpoint):
        seed = int(hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()[:8], 16)
        torch.manual_seed(seed)
        net = FakeDetector().eval()

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.det_head = net.det_head

            def data_preprocessor(self, inputs, training=False):
                return dict(inputs=inputs)

            def forward(self, inputs, mode='tensor'):
                return self.det_head.binarize(inputs)

        self.model = Model().eval()

    def preprocess(self, paths, batch_size=1):
        from PIL import Image
        _FakeInferencer.calls += 1
        if _FakeInferencer.fail_after is not None and _FakeInferencer.calls > _FakeInferencer.fail_after:
            raise RuntimeError('simulated Colab disconnect')
        with Image.open(paths[0]) as im:
            x = torch.from_numpy(np.asarray(im.convert('RGB'), dtype=np.float32)/255).permute(2, 0, 1)[None]
        yield None, x


class Stage0ResumeTest(unittest.TestCase):
    def setUp(self):
        import sys
        import types
        stage_a = types.ModuleType('rev4.stage_a')
        stage_a.detector_for = lambda a: (_FakeInferencer(a.checkpoint), None, None, None)
        self.patches = [patch.dict(sys.modules, {'rev4': types.ModuleType('rev4'), 'rev4.stage_a': stage_a}),
                        patch('mixab.monkey.use_monkey', lambda: dict(commit='test')),
                        patch.object(torch.cuda, 'get_device_name', lambda i=0: 'FakeGPU'),
                        patch.object(torch.cuda, 'empty_cache', lambda: None)]
        for p in self.patches:
            p.start()
        _FakeInferencer.calls, _FakeInferencer.fail_after = 0, None

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()

    def build(self, tmp):
        from PIL import Image
        from mixab import save_json
        tmp = Path(tmp)
        rows = []
        for i in range(3):
            (tmp/'clean/train_images').mkdir(parents=True, exist_ok=True)
            (tmp/'lr2').mkdir(exist_ok=True)
            clean, lr2 = tmp/f'clean/train_images/{i}.png', tmp/f'lr2/{i}.png'
            Image.new('RGB', (8 + i, 6), (30*i, 60, 90)).save(clean)
            Image.new('RGB', (4, 3), (90, 30*i, 10)).save(lr2)
            rows.append(dict(source_image_id=str(i), clean_relpath=f'train_images/{i}.png',
                             clean_sha256=br.sha256(clean), lr2_name=lr2.name, lr2_sha256=br.sha256(lr2)))
        br.write_csv(tmp/'subset_images.csv', rows, list(rows[0]))
        (tmp/'w').mkdir()
        models = []
        for alpha, name in (('0', 'a0.pth'), ('1/2', 'a12.pth'), ('1', 'a1.pth')):
            (tmp/'w'/name).write_bytes(name.encode())
            models.append(dict(alpha=alpha, checkpoint=f'/drive/{name}', checkpoint_sha256=br.sha256(tmp/'w'/name)))
        save_json(tmp/'w/merge_manifest.json', dict(models=models))
        (tmp/'cfg.py').write_text('x = 1\n')
        return tmp

    def args(self, tmp, out):
        return argparse.Namespace(subset_images=tmp/'subset_images.csv', train_check=None, clean_root=tmp/'clean',
                                  lr2_root=tmp/'lr2', model_config=tmp/'cfg.py', weights_dir=tmp/'w',
                                  output=tmp/out, chunk=2)

    def test_interrupted_run_resumes_to_the_same_result(self):
        with TemporaryDirectory() as tmp:
            tmp = self.build(tmp)
            br.stage0(self.args(tmp, 'straight'))
            straight = json.loads((tmp/'straight/stage0_summary.json').read_text(encoding='utf-8'))
            self.assertEqual(sorted(straight['stored_vs_measured']), ['0', '1', '1/2'])
            self.assertEqual(straight['gate']['status'], 'PROPOSED_NOT_APPROVED')
            self.assertIn('1/2', straight['nonlinearity_vs_endpoint_interp']['det_head.binarize.1'])
            # interrupt in the middle of the second model (6 images per model), then resume
            _FakeInferencer.calls, _FakeInferencer.fail_after = 0, 9
            with self.assertRaisesRegex(RuntimeError, 'disconnect'):
                br.stage0(self.args(tmp, 'resumed'))
            self.assertTrue((tmp/'resumed/stage0_alpha_0.json').is_file())
            self.assertTrue((tmp/'resumed/.alpha_1_2.done.json').is_file())
            _FakeInferencer.fail_after = None
            br.stage0(self.args(tmp, 'resumed'))
            resumed = json.loads((tmp/'resumed/stage0_summary.json').read_text(encoding='utf-8'))
            self.assertEqual(resumed['stored_vs_measured'], straight['stored_vs_measured'])
            self.assertFalse(any(p.name.startswith('.') for p in (tmp/'resumed').iterdir()))
            with self.assertRaisesRegex(ValueError, 'không ghi đè'):
                br.stage0(self.args(tmp, 'resumed'))
            # a corrupted image is refused
            (tmp/'lr2/1.png').write_bytes(b'x')
            with self.assertRaisesRegex(ValueError, 'sai SHA'):
                br.stage0(self.args(tmp, 'other'))


if __name__ == '__main__':
    unittest.main()
