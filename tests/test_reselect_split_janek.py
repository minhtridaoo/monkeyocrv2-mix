"""Mix A-core A3: R1 split/selection/lock rules, gate 3b join, verification status, bootstrap."""
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from mixab import read_json 
from mixab import reselect_split_janek as rs 


def setting(geometry, score, mask=.30, unclip=1.5, width=5, eps=.010):
    return dict(geometry=geometry, mask_thr=mask, unclip_ratio=unclip, min_text_width=width,
                epsilon_ratio=eps, min_text_score=score)


SETTINGS = [setting('base', .30), setting('base', .40), setting('unclip_1p2', .40, unclip=1.2),
            setting('width_3', .40, width=3)]
SOURCES = [f'src_{i:04d}' for i in range(120)]
SPLIT = [rs.split_of(s) for s in SOURCES]


def make_counts(tp_fn, fields=11, tiny_tp=1):
    """gt = pred = 10 everywhere, so micro-Hmean = tp/10; bins gt = 2 each."""
    counts = np.zeros((len(SOURCES), 2, len(SETTINGS), fields), dtype=np.int64)
    counts[..., 1] = 10
    counts[..., 2] = 10
    for i, split in enumerate(SPLIT):
        for c in range(2):
            for k in range(len(SETTINGS)):
                counts[i, c, k, 0] = tp_fn(split, c, k)
    for g, t in rs.bin_fields(fields):
        counts[..., g] = 2
        counts[..., t] = 1
    counts[..., 4] = tiny_tp
    return counts


def save_npz(path, counts, sources=SOURCES, settings=SETTINGS):
    np.savez_compressed(path, source_ids=np.asarray(sources),
                        settings=np.asarray([json.dumps(s, sort_keys=True) for s in settings]), counts=counts)
    return str(path)


class SplitAndSelectionTest(unittest.TestCase):
    def test_split_rule_is_sha256_parity_of_utf8_id(self):
        for source in ['src_0001', 'ảnh_Việt_42', '00000']:
            expected = 'A' if int(hashlib.sha256(source.encode('utf-8')).hexdigest(), 16) % 2 == 0 else 'B'
            self.assertEqual(rs.split_of(source), expected)
        self.assertTrue(30 < SPLIT.count('A') < 90)

    def test_native_tie_break_then_grid_order(self):
        # Put native base/.30 LAST, tied with a non-native setting placed first: native must still win,
        # so this distinguishes the native preference from plain grid order.
        reordered = [SETTINGS[2], SETTINGS[1], SETTINGS[3], SETTINGS[0]]
        counts = make_counts(lambda s, c, k: 7 if k in (0, 3) else 5)
        k, _ = rs.choose_setting(counts, reordered, np.arange(len(SOURCES)))
        self.assertEqual(reordered[k]['geometry'], 'base')
        self.assertEqual(reordered[k]['min_text_score'], .3)
        # non-native tie (2 vs 3): earlier grid entry wins
        counts = make_counts(lambda s, c, k: 7 if k in (2, 3) else 5)
        k, _ = rs.choose_setting(counts, SETTINGS, np.arange(len(SOURCES)))
        self.assertEqual(k, 2)

    def test_selection_uses_split_A_only(self):
        counts = make_counts(lambda s, c, k: (8 if k == 2 else 5) if s == 'A' else (9 if k == 3 else 5))
        a_index = np.array([i for i, s in enumerate(SPLIT) if s == 'A'])
        k, value = rs.choose_setting(counts, SETTINGS, a_index)
        self.assertEqual(k, 2)
        self.assertEqual(value, Fraction(16, 10))

    def test_alpha_tie_break_endpoint_then_larger(self):
        v = Fraction(3, 2)
        self.assertEqual(rs.choose_alpha([(Fraction(1, 2), v), (Fraction(0), v)]), 0)
        self.assertEqual(rs.choose_alpha([(Fraction(0), v), (Fraction(1), v), (Fraction(1, 2), v)]), 1)
        self.assertEqual(rs.choose_alpha([(Fraction(1, 4), v), (Fraction(3, 4), v)]), Fraction(3, 4))
        self.assertEqual(rs.choose_alpha([(Fraction(1), v), (Fraction(1, 2), v + Fraction(1, 100))]), Fraction(1, 2))

    def test_status_logic(self):
        self.assertEqual(rs.verification_status({}), 'PENDING_REFERENCE_CHECK')
        self.assertEqual(rs.verification_status(dict(gate_3b='PASS')), 'PENDING_REFERENCE_CHECK')
        self.assertEqual(rs.verification_status(dict(gate_3a='PASS', pmap_endpoint='PASS', gate_3b='PASS')), 'VERIFIED')
        self.assertEqual(rs.verification_status(dict(gate_3a='PASS', pmap_endpoint='PASS', gate_3b='FAIL')), 'INVALIDATED')


class EndToEndTest(unittest.TestCase):
    def build(self, tmp, tp_by_alpha, fields=11, tiny=None):
        models = []
        for alpha, fn in tp_by_alpha.items():
            path = save_npz(Path(tmp)/f'a{alpha.replace("/", "_")}.npz',
                            make_counts(fn, fields, tiny_tp=(tiny or {}).get(alpha, 1)))
            models.append(dict(name={'0': 'B-Clean', '1': 'B1'}.get(alpha, f'merge_{alpha}'), alpha=alpha, npz=path))
        return models

    def test_interior_alpha_wins_and_continues(self):
        tp = {'0': lambda s, c, k: 6 if k == 1 else 5, '1/4': lambda s, c, k: 5,
              '1/2': lambda s, c, k: 8 if k == 2 else 5, '3/4': lambda s, c, k: 5,
              '1': lambda s, c, k: 6 if k == 1 else 5}
        with TemporaryDirectory() as tmp:
            out = rs.run(self.build(tmp, tp), {}, {}, Path(tmp)/'out')
            d = out['decision']
            self.assertEqual(d['alpha_star'], '1/2')
            self.assertTrue(d['passes_margin'] and d['passes_tiny'] and d['continue_mix_a'])
            self.assertAlmostEqual(d['mean_hmean_B_alpha_star'], .8)
            self.assertAlmostEqual(d['best_endpoint_mean_hmean_B'], .6)
            self.assertEqual(d['verification'], 'PENDING_REFERENCE_CHECK')
            self.assertEqual(out['selection']['merge_1/2']['setting_key'], 'unclip_1p2/score_0.40')
            boot = out['bootstrap']['merge_1/2']['vs_B1']['mean']
            self.assertAlmostEqual(boot['delta_pp'], 20.0)
            self.assertAlmostEqual(boot['ci95_pp'][0], 20.0)   # constant per-source gain
            for name in ('split_manifest.csv', 'selection.json', 'evaluation_B.csv', 'bootstrap_B.json', 'decision.json'):
                self.assertTrue((Path(tmp)/'out'/name).is_file())
            with self.assertRaisesRegex(ValueError, 'Refusing to overwrite'):
                rs.run(self.build(tmp, tp), {}, {}, Path(tmp)/'out')

    def test_alpha_star_locked_on_A_even_if_other_alpha_is_better_on_B(self):
        tp = {'0': lambda s, c, k: 5, '1/2': lambda s, c, k: (6 if s == 'A' else 9) if k == 2 else 5,
              '1': lambda s, c, k: 7 if k == 1 else 5}
        with TemporaryDirectory() as tmp:
            d = rs.run(self.build(tmp, tp), {}, {}, Path(tmp)/'out')['decision']
            self.assertEqual(d['alpha_star'], '1')
            self.assertTrue(d['alpha_star_is_endpoint'])
            self.assertFalse(d['continue_mix_a'])
            self.assertEqual(d['note'], 'merge did not win on A')

    def test_tiny_recall_threshold_blocks_continuation(self):
        tp = {'0': lambda s, c, k: 5, '1/2': lambda s, c, k: 8, '1': lambda s, c, k: 5}
        with TemporaryDirectory() as tmp:
            d = rs.run(self.build(tmp, tp, tiny={'1/2': 0}), {}, {}, Path(tmp)/'out')['decision']
            self.assertTrue(d['passes_margin'])
            self.assertFalse(d['passes_tiny'])
            self.assertFalse(d['continue_mix_a'])

    def test_tiny_rule_is_relative_to_B1_on_split_B(self):
        """Tuan 20/11: alpha* tiny recall on B >= B1 tiny recall on B - 3 pp (exact Fractions)."""
        tp = {'0': lambda s, c, k: 5, '1/2': lambda s, c, k: 8, '1': lambda s, c, k: 5}
        b_rows = [i for i, s in enumerate(SPLIT) if s == 'B']
        gt_total = 2*len(b_rows)                      # every source has 2 tiny GT, B1 finds 1 -> 50 %
        for dropped, expected in ((0, True), (int(gt_total*3/100), True), (int(gt_total*3/100) + 1, False)):
            with TemporaryDirectory() as tmp:
                models = self.build(tmp, tp)
                star = next(m for m in models if m['alpha'] == '1/2')
                data = rs.load_counts(star['npz'])
                counts = data['counts'].copy()
                counts[b_rows[:dropped], 1, :, 4] = 0     # alpha* misses some tiny GT on B (LR2)
                star['npz'] = save_npz(Path(tmp)/'star.npz', counts)
                d = rs.run(models, {}, {}, Path(tmp)/'out')['decision']
                self.assertAlmostEqual(d['lr2_recall_lt4_B1'], .5)
                self.assertAlmostEqual(d['tiny_floor'], .47)
                self.assertEqual(d['passes_tiny'], expected, dropped)
                self.assertEqual(d['continue_mix_a'], expected)
                self.assertTrue(d['legacy_tiny_42']['passes'])     # reported only

    def test_gate_3b_joins_by_key_and_sets_status(self):
        tp = {'0': lambda s, c, k: 5 + k % 2, '1/2': lambda s, c, k: 5, '1': lambda s, c, k: 6 - k % 2}
        with TemporaryDirectory() as tmp:
            models = self.build(tmp, tp)
            refs = {}
            for m in models:
                if m['alpha'] in ('0', '1'):
                    data = rs.load_counts(m['npz'])
                    order, perm = list(range(len(SOURCES)))[::-1], [3, 1, 0, 2]
                    shuffled = data['counts'][order][:, :, perm, :5]
                    refs[m['name']] = save_npz(Path(tmp)/f'ref_{m["name"]}.npz', shuffled,
                                               [SOURCES[i] for i in order], [SETTINGS[k] for k in perm])
            d = rs.run(models, refs, {}, Path(tmp)/'o1')['decision']
            self.assertEqual(d['checks']['gate_3b'], 'PASS')
            self.assertEqual(d['verification'], 'PENDING_REFERENCE_CHECK')   # 3a / P-map not run
            d = rs.run(models, refs, dict(gate_3a='PASS', pmap_endpoint='PASS'), Path(tmp)/'o2')['decision']
            self.assertEqual(d['verification'], 'VERIFIED')
            bad = rs.load_counts(refs['B1'])['counts'].copy()
            bad[0, 1, 0, 0] -= 1
            refs['B1'] = save_npz(Path(tmp)/'ref_bad.npz', bad, rs.load_counts(refs['B1'])['source_ids'],
                                  rs.load_counts(refs['B1'])['settings'])
            d = rs.run(models, refs, dict(gate_3a='PASS', pmap_endpoint='PASS'), Path(tmp)/'o3')['decision']
            self.assertEqual(d['checks']['gate_3b'], 'FAIL')
            self.assertEqual(d['verification'], 'INVALIDATED')
            self.assertEqual(read_json(Path(tmp)/'o3/decision.json')['reference_checks']['B1']['mismatched'], 1)

    def test_same_gpu_policy_uses_exact_probe_match_and_reports_h200_as_drift(self):
        tp = {'0': lambda s, c, k: 5 + k % 2, '1/2': lambda s, c, k: 5, '1': lambda s, c, k: 6 - k % 2}
        with TemporaryDirectory() as tmp:
            models = self.build(tmp, tp)
            probe, h200 = {}, {}
            for m in models:
                if m['alpha'] in ('0', '1'):
                    data = rs.load_counts(m['npz'])
                    keep = [2, 0]                                  # probe = subset of sources, all 11 fields
                    probe[m['name']] = save_npz(Path(tmp)/f'probe_{m["name"]}.npz', data['counts'][keep],
                                                [SOURCES[i] for i in keep], SETTINGS)
                    drift = data['counts'][..., :5].copy()
                    drift[0, 0, 0, 2] += 1                          # H200 differs by one prediction
                    h200[m['name']] = save_npz(Path(tmp)/f'h200_{m["name"]}.npz', drift)
            same = dict(run_gpu='NVIDIA A100-SXM4-40GB', reference_gpu='NVIDIA A100-SXM4-40GB', npz=probe)
            checks = dict(gate_3a='FAIL', pmap_endpoint='PASS')
            d = rs.run(models, h200, checks, Path(tmp)/'s1', policy='same_gpu_endpoints', same_gpu=same)['decision']
            self.assertEqual(d['verification'], 'VERIFIED')
            self.assertEqual(d['checks'], dict(pmap_endpoint='PASS', gate_same_gpu='PASS'))
            self.assertEqual(d['cross_gpu_drift']['gate_3a'], 'FAIL')
            self.assertEqual(d['cross_gpu_drift']['gate_3b'], 'FAIL')
            # A difference in a bin column (index 5+) breaks the same-GPU gate: all 11 fields are compared.
            good = dict(probe)
            bad = rs.load_counts(probe['B1'])['counts'].copy()
            bad[1, 1, 3, 7] += 1
            probe['B1'] = save_npz(Path(tmp)/'probe_bad.npz', bad, [SOURCES[2], SOURCES[0]], SETTINGS)
            d = rs.run(models, h200, checks, Path(tmp)/'s2', policy='same_gpu_endpoints', same_gpu=same)['decision']
            self.assertEqual(d['verification'], 'INVALIDATED')
            self.assertEqual(read_json(Path(tmp)/'s2/decision.json')['same_gpu_checks']['B1']['mismatched'], 1)
            # P-map not run -> pending; another GPU model -> refused; H200 policy still invalidates on 3a FAIL.
            d = rs.run(models, h200, {}, Path(tmp)/'s3', policy='same_gpu_endpoints',
                       same_gpu=dict(same, npz=good))['decision']
            self.assertEqual(d['verification'], 'PENDING_REFERENCE_CHECK')
            with self.assertRaises(ValueError):
                rs.run(models, h200, checks, Path(tmp)/'s4', policy='same_gpu_endpoints',
                       same_gpu=dict(same, reference_gpu='NVIDIA A100-SXM4-80GB'))
            with self.assertRaises(ValueError):
                rs.run(models, h200, checks, Path(tmp)/'s5', policy='same_gpu_endpoints', same_gpu=None)
            self.assertEqual(rs.verification_status(dict(pmap_endpoint='PASS'), 'same_gpu_endpoints'),
                             'PENDING_REFERENCE_CHECK')
            self.assertEqual(rs.run(models, h200, checks, Path(tmp)/'s6')['decision']['verification'], 'INVALIDATED')

    def test_endpoints_only_is_baseline_mode_and_not_judged(self):
        tp = {'0': lambda s, c, k: 7, '1': lambda s, c, k: 5}
        with TemporaryDirectory() as tmp:
            d = rs.run(self.build(tmp, tp), {}, {}, Path(tmp)/'out')['decision']
            self.assertEqual(d['mode'], 'baseline_only')
            self.assertIsNone(d['continue_mix_a'])
            self.assertIsNone(d['passes_margin'])
            self.assertIn('baseline_only', d['note'])

    def test_tuan_five_column_format_is_accepted(self):
        tp = {'0': lambda s, c, k: 5, '1': lambda s, c, k: 6}
        with TemporaryDirectory() as tmp:
            out = rs.run(self.build(tmp, tp, fields=5), {}, {}, Path(tmp)/'out')
            row = out['results']['B1']['selected_on_A']['lr2']
            self.assertAlmostEqual(row['recall_lt4'], .5)
            self.assertNotIn('recall_4to8', row)


class LoaderAndBootstrapTest(unittest.TestCase):
    def test_loader_rejects_bad_inputs(self):
        with TemporaryDirectory() as tmp:
            bad = make_counts(lambda s, c, k: 5)[:, :, :, :7]
            with self.assertRaisesRegex(ValueError, '5 or 11'):
                rs.load_counts(save_npz(Path(tmp)/'a.npz', bad))
            bad = make_counts(lambda s, c, k: 11)
            with self.assertRaisesRegex(ValueError, 'tp exceeds'):
                rs.load_counts(save_npz(Path(tmp)/'b.npz', bad))

    def test_bootstrap_identical_models_and_point_estimate(self):
        counts = make_counts(lambda s, c, k: 5)
        b_index = np.array([i for i, s in enumerate(SPLIT) if s == 'B'])
        idx = np.random.default_rng(rs.BOOTSTRAP_SEED).integers(0, len(b_index), size=(200, len(b_index)))
        same = rs.bootstrap(counts[:, :, 1], counts[:, :, 1], b_index, idx)
        self.assertEqual(same['lr2']['delta_pp'], 0.)
        self.assertEqual(same['lr2']['ci95_pp'], [0., 0.])
        better = counts.copy()
        better[:, 1, :, 0] = 7
        diff = rs.bootstrap(better[:, :, 1], counts[:, :, 1], b_index, idx)
        self.assertAlmostEqual(diff['lr2']['delta_pp'], 20.0)
        self.assertAlmostEqual(diff['clean']['delta_pp'], 0.0)
        self.assertAlmostEqual(diff['mean']['delta_pp'], 10.0)


if __name__ == '__main__':
    unittest.main()
