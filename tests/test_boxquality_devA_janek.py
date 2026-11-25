"""CPU tests for mixab.boxquality_devA_janek (frozen-scorer evaluation on development A)."""
import gzip
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mixab import boxquality_devA_janek as dv
from mixab import boxquality_janek as bq

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/'docs/handoff/artifacts/janek/boxquality_v1_T4'
MODEL, FIT = ART/'model_janek.npz', ART/'boxquality_fit_janek.json'
DUMP_LR2 = ROOT/'runtime/phase0_inputs/dualagree_dump_v1_T4'
DUMP_CLEAN = ROOT/'runtime/kaggle_downloads/csb1_clean_T4_20261123-110923/out/clean_dump'


def bundle():
    from mixab.monkey_janek import monkey_root
    return monkey_root()/'docs/handoff/artifacts/pr_quad_audit_v1'


class Selection(unittest.TestCase):
    def test_smoke_and_latency_subsets(self):
        src = [f'{i:016x}' for i in range(500)]
        sm = dv.smoke_sources(src)
        self.assertEqual(len(sm), dv.N_SMOKE)
        self.assertEqual(sm, dv.smoke_sources(list(reversed(src))))          # order independent
        warm, timed = dv.latency_sources(src)
        self.assertEqual((len(warm), len(timed)), (dv.N_WARM, dv.N_LAT))
        self.assertFalse(set(warm) & set(timed))


@unittest.skipUnless(MODEL.is_file() and FIT.is_file(), 'frozen scorer artifact not present')
class FrozenModel(unittest.TestCase):
    def test_loads_frozen_and_cutoffs(self):
        m = dv.load_model(MODEL, FIT)
        self.assertEqual(m['cut_full'], 0.3315908613227198)
        self.assertEqual(m['cut_raw'], 0.5976406401908236)

    def test_rejects_modified_model(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d)/'model.npz'
            shutil.copy(MODEL, bad)
            with bad.open('ab') as f:
                f.write(b'\0')
            with self.assertRaises(ValueError):
                dv.load_model(bad, FIT)

    def test_full_scores_match_manual_logistic(self):
        import numpy as np
        m = dv.load_model(MODEL, FIT)
        x = (np.asarray(m['mean']) + .5*np.asarray(m['std'])).tolist()
        z = (np.asarray(x) - m['mean'])/m['std']
        want = 1/(1 + np.exp(-(m['beta'][0] + z @ m['beta'][1:])))
        self.assertAlmostEqual(dv.full_scores(m, [x])[0], float(want), places=12)
        self.assertEqual(dv.full_scores(m, []), [])

    def test_kept_indices_inclusive(self):
        m = dict(cut_raw=.5976406401908236, cut_full=.3315908613227198)
        k = dv.kept_indices([.6, .5976406401908236, .59], [.3315908613227198, .2, .9], m)
        self.assertEqual(k, dict(F=[0], RAW=[0, 1], FULL=[0, 2]))


class Rows(unittest.TestCase):
    def test_bin_row(self):
        heights = {'a': 2., 'b': 5., 'c': 12., 'd': 40., 'e': 3.}
        self.assertEqual(dv.bin_row(3, 5, 7, ['a', 'c', 'd'], heights), [3, 5, 7, 2, 1, 1, 0, 1, 1, 1, 1])

    def test_totals(self):
        t = dv.totals([[3, 5, 7, 2, 1, 1, 0, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]])
        self.assertEqual((t['tp'], t['fp'], t['fn']), (4, 4, 2))
        self.assertAlmostEqual(t['hmean'], 8/14)
        self.assertEqual(t['tiny_recall'], .5)


class Outcome(unittest.TestCase):
    G_OK = dict(g1=True, g2=True, g3=True)
    G_NO = dict(g1=False, g2=True, g3=True)

    def lab(self, **kw):
        base = dict(valid=True, d=.3, ci=[.1, .5], ci_clean_hi=.4, ci_lr2_hi=.4, tiny_clean_pp=-1., tiny_lr2_pp=-1.,
                    latency_ratio=1.05, gates=self.G_NO)
        base.update(kw)
        return dv.outcome(base['valid'], base['d'], base['ci'], base['ci_clean_hi'], base['ci_lr2_hi'], base['tiny_clean_pp'],
                          base['tiny_lr2_pp'], base['latency_ratio'], base['gates'])

    def test_table(self):
        self.assertEqual(self.lab(valid=False)[0], 'INVALID_RUN')
        self.assertEqual(self.lab(gates=self.G_OK)[0], 'GROUP_GATE_PASS')
        self.assertEqual(self.lab(gates=self.G_OK, ci=[-.1, .5])[0], 'INCONCLUSIVE')     # gates alone are not enough
        self.assertEqual(self.lab(d=0., ci=[-.1, .1])[0], 'NO_BENEFIT')
        self.assertEqual(self.lab(ci=[0., .5])[0], 'INCONCLUSIVE')
        self.assertEqual(self.lab(), ('SMALL_USEFUL', []))
        self.assertEqual(self.lab(latency_ratio=1.11), ('POSITIVE_NOT_USEFUL', ['latency_median_ratio_above_1.10']))
        self.assertEqual(self.lab(latency_ratio=None)[1], ['latency_missing'])
        self.assertEqual(self.lab(tiny_clean_pp=-3.01, ci_lr2_hi=-.01)[1], ['lr2_significantly_worse', 'tiny_clean_below_F_minus_3pp'])

    def test_generalization(self):
        self.assertEqual(dv.generalization(.1, [.01, .2]), 'tong_quat_hoa')
        self.assertEqual(dv.generalization(.1, [-.01, .2]), 'mot_phan')
        self.assertEqual(dv.generalization(0., [-.1, .1]), 'khong')


def _have_inputs():
    try:
        return bundle().is_dir() and DUMP_LR2.is_dir() and DUMP_CLEAN.is_dir() and MODEL.is_file()
    except Exception:                                    # noqa: BLE001
        return False


@unittest.skipUnless(_have_inputs(), 'bundle / T4 dumps not present')
class EvaluateDryRun(unittest.TestCase):
    """`evaluate --subset smoke` on records rebuilt from the T4 dumps (features = raw, so FULL is meaningless): the
    official evaluator must agree with the fast path, and live F must equal the dump-based B1 .40/.60."""

    def test_smoke_dry_run(self):
        from mixab import dualagree_janek as da
        from mixab import read_json
        from mixab.monkey_janek import use_monkey
        use_monkey()
        from rev4.decision_plan import GEOMETRIES
        rows, A, _ = dv._context(bundle())
        gi = [g[0] for g in GEOMETRIES].index(bq.GEOM[0])
        with tempfile.TemporaryDirectory() as d:
            ex = Path(d)/'extract'
            ex.mkdir()
            with gzip.open(ex/'shard_0000.jsonl.gz', 'wt', encoding='utf-8') as f:
                for c, dd in (('lr2', DUMP_LR2), ('clean', DUMP_CLEAN)):
                    dump = da.load_model_dump(dd/'B1', len(GEOMETRIES), read_json(dd/'dump_identity.json')['identity_sha256'])
                    for s in dv.smoke_sources(A):
                        e = dump[f'{c}::{s}'][gi]
                        raw = [float(x) for x in e['raw']]
                        f.write(json.dumps(dict(source=s, condition=c, polygons=[[float(v) for v in p] for p in e['polygons']],
                                                raw=raw, packed=[float(x) for x in e['packed']], features=[[r]*13 for r in raw],
                                                consistency=True)) + '\n')
            lat = Path(d)/'latency.json'
            lat.write_text(json.dumps(dict(paired=dict(ratio_of_medians=1.0, ratio_of_p95=1.0), memory={})))
            out = Path(d)/'eval'
            subprocess.run([sys.executable, '-m', 'mixab.boxquality_devA_janek', 'evaluate', '--bundle', str(bundle()),
                            '--extract-dir', str(ex), '--model', str(MODEL), '--fit', str(FIT), '--dump-lr2', str(DUMP_LR2),
                            '--dump-clean', str(DUMP_CLEAN), '--latency', str(lat), '--subset', 'smoke', '--output', str(out)],
                           cwd=ROOT, check=True, capture_output=True)
            r = json.loads((out/'devA_result.json').read_text(encoding='utf-8'))
        self.assertTrue(r['checks']['fast_equals_official'])
        self.assertEqual(r['checks']['tasks'], 2*dv.N_SMOKE)
        for c in dv.CONDS:
            self.assertEqual(r['live_vs_t4_dump'][c]['identical_candidates'], dv.N_SMOKE)
            self.assertEqual(r['live_vs_t4_dump'][c]['F_hmean_live'], r['live_vs_t4_dump'][c]['F_hmean_dump'])


if __name__ == '__main__':
    unittest.main()
