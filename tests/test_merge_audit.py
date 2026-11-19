"""Mix A-core A2: CPU-testable parts of merge_audit, checked against Tuấn's real bundle when present."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from mixab import merge_audit as ma
from mixab import reselect_split as rs
from mixab.monkey import monkey_root

ROOT = monkey_root(required=False)
BUNDLE = (ROOT or Path('.'))/'docs/handoff/artifacts/pr_quad_audit_v1'
HAVE_BUNDLE = (BUNDLE/'counts/B1_source_counts.npz').is_file()


class PureHelpersTest(unittest.TestCase):
    def test_model_names(self):
        self.assertEqual([ma.model_name(a) for a in ('0', '1/4', '1/2', '3/4', '1')],
                         ['B-Clean', 'merge_a1_4', 'merge_a1_2', 'merge_a3_4', 'B1'])

    def test_probe_subset_is_fixed_and_order_independent(self):
        ids = [f's{i:04d}' for i in range(300)]
        first = ma.probe_sources(ids, 25)
        self.assertEqual(first, ma.probe_sources(list(reversed(ids)), 25))
        self.assertEqual(len(first), 25)
        self.assertEqual(first, sorted(first))
        self.assertNotEqual(first, ids[:25])

    def test_count_row_layout(self):
        scored = dict(counts=[7, 10, 9], bins=[[2, 1], [3, 2], [4, 3], [1, 1]])
        self.assertEqual(ma.count_row(scored), [7, 10, 9, 2, 1, 3, 2, 4, 3, 1, 1])
        with self.assertRaises(ValueError):
            ma.count_row(dict(counts=[1, 1, 1], bins=[[1, 1]]))

    def test_progress_resume_and_identity_guard(self):
        with TemporaryDirectory() as tmp:
            p = ma.Progress(tmp, 'B1', 'id-1', (3, 2, 4, ma.FIELDS))
            p.matrix[1, 0, 2] = np.arange(ma.FIELDS)
            p.done.update({'clean::a', 'lr2::a'})
            p.save()
            again = ma.Progress(tmp, 'B1', 'id-1', (3, 2, 4, ma.FIELDS))
            self.assertEqual(again.done, {'clean::a', 'lr2::a'})
            self.assertTrue(np.array_equal(again.matrix, p.matrix))
            with self.assertRaisesRegex(ValueError, 'another run identity'):
                ma.Progress(tmp, 'B1', 'id-2', (3, 2, 4, ma.FIELDS))
            again.clear()
            self.assertEqual(ma.Progress(tmp, 'B1', 'id-2', (3, 2, 4, ma.FIELDS)).done, set())

    def test_written_counts_are_readable_by_reselect(self):
        settings = ma.build_settings([('base', .3, 1.5, 5, .01)], (.3, .4))
        counts = np.zeros((2, 2, 2, ma.FIELDS), dtype=np.int64)
        counts[..., 1] = counts[..., 2] = 5
        counts[..., 0] = 3
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'x.npz'
            ma.write_counts(path, ['a', 'b'], settings, counts)
            loaded = rs.load_counts(path)
            self.assertEqual(loaded['settings'], settings)
            self.assertEqual(loaded['counts'].shape[-1], 11)

    def test_overall_status(self):
        self.assertEqual(ma.overall_status({}), 'NOT_RUN')
        self.assertEqual(ma.overall_status(dict(gate_3a=dict(status='PASS'), pmap_endpoint=dict(status='PASS'))), 'PASS')
        self.assertEqual(ma.overall_status(dict(gate_3a=dict(status='FAIL'), pmap_endpoint=dict(status='PASS'))), 'FAIL')


@unittest.skipUnless(HAVE_BUNDLE, "Tuấn's pr_quad_audit_v1 bundle not present")
class RealBundleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = ma.load_rows(BUNDLE)
        cls.meta = ma.load_task_metadata(BUNDLE)
        cls.eda = ma.load_eda_counts(BUNDLE)

    def test_manifest_and_tasks(self):
        self.assertEqual(len(self.rows), 2262)
        self.assertEqual(len(self.meta), 4524)
        roots = dict(TextVQA_clean_root='/data/textvqa', prepared_root='/data/prepared')
        tasks = ma.resolve_tasks(self.meta, roots, ma.probe_sources(sorted(self.rows), 5))
        self.assertEqual(len(tasks), 10)
        self.assertEqual(sorted({t['condition'] for t in tasks}), ['clean', 'lr2'])
        lr2 = next(t for t in tasks if t['condition'] == 'lr2')
        self.assertTrue(lr2['path'].replace('\\', '/').startswith('/data/prepared/validation_lr2/'))
        clean = next(t for t in tasks if t['source_image_id'] == lr2['source_image_id'] and t['condition'] == 'clean')
        self.assertEqual(lr2['size'], [max(1, clean['size'][0]//2), max(1, clean['size'][1]//2)])
        with self.assertRaisesRegex(ValueError, 'Need roots'):
            ma.resolve_tasks(self.meta, dict(prepared_root='/x'))
        self.assertEqual(len(ma.pmap_tasks(ma.resolve_tasks(self.meta, roots))), 20)

    def test_settings_rebuild_tuans_grid_exactly(self):
        from mixab.monkey import use_monkey
        use_monkey()
        from rev4.decision_plan import GEOMETRIES, SCORES   # constants only, no MMOCR import
        reference = rs.load_counts(BUNDLE/'counts/B1_source_counts.npz')['settings']
        self.assertEqual(ma.build_settings(GEOMETRIES, SCORES), reference)

    def test_gate_3a_passes_on_tuans_counts_and_detects_a_change(self):
        for name in ('B-Clean', 'B1'):
            data = rs.load_counts(BUNDLE/f'counts/{name}_source_counts.npz')
            common = rs.common_index(data['settings'])
            result = ma.gate_3a(name, data['counts'], data['source_ids'], common, self.eda)
            self.assertEqual(result['status'], 'PASS', result['examples'][:2])
            self.assertEqual(result['checked'], 2*2262)
        changed = data['counts'].copy()
        changed[5, 1, common, 0] += 1
        result = ma.gate_3a('B1', changed, data['source_ids'], common, self.eda)
        self.assertEqual((result['status'], result['mismatched']), ('FAIL', 1))


if __name__ == '__main__':
    unittest.main()
