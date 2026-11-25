"""Pre-flight doctor: config parsing, report status logic, behaviour on a CPU-only machine."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from mixab import doctor_janek as doctor 
from mixab.monkey_janek import monkey_root

ROOT = monkey_root(required=False)
CONFIG = (ROOT or Path('.'))/'docs/handoff/inference_config.py'


class DoctorTest(unittest.TestCase):
    def test_report_failed_only_on_fail(self):
        r = doctor.Report()
        r.add('a', 'PASS', 'x')
        r.add('b', 'WARN', 'y')
        self.assertFalse(r.failed)
        r.ok('c', False, 'z')
        self.assertTrue(r.failed)
        r2 = doctor.Report()
        r2.ok('d', False, 'soft', fail='WARN')
        self.assertFalse(r2.failed)

    @unittest.skipUnless(CONFIG.is_file(), 'inference_config.py not present')
    def test_encoder_paths_from_tuans_config(self):
        paths = doctor.encoder_paths(CONFIG.read_text(encoding='utf-8'))
        self.assertEqual(paths, dict(model_dir='pretrained/monkeyocrv2_as',
                                     weights='pretrained/monkeyocrv2_as/model.safetensors'))

    def test_config_check_flags_missing_encoder(self):
        with TemporaryDirectory() as tmp:
            cfg = Path(tmp)/'cfg.py'
            cfg.write_text("model=dict(backbone=dict(model_dir='nowhere/enc', use_bf16=False, "
                           "weights='nowhere/enc/model.safetensors'))\ntextdet_parameter_precision = 'fp32'\n",
                           encoding='utf-8')
            r = doctor.Report()
            doctor.check_config(r, cfg)
            status = {row['check']: row['status'] for row in r.rows}
            self.assertEqual(status['config:fp32'], 'PASS')
            self.assertEqual(status['config:model_dir'], 'FAIL')
            self.assertEqual(status['config:weights'], 'FAIL')

    def test_cuda_check_fails_cleanly_without_gpu(self):
        import torch
        if torch.cuda.is_available():
            self.skipTest('GPU present')
        r = doctor.Report()
        doctor.check_cuda(r)
        self.assertEqual(r.rows[0]['status'], 'FAIL')


if __name__ == '__main__':
    unittest.main()
