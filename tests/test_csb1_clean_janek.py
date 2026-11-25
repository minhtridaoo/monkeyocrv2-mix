"""csb1_clean: per-condition rows from a tiny synthetic dump equal PoolScorer / CandidateMatcher directly."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from mixab import csb1_clean_janek as csb1_clean, dualagree_janek as da, phase0_janek as p0 
from mixab.monkey_janek import monkey_root

ROOT = monkey_root(required=False)
BUNDLE = (ROOT or Path('.'))/'docs/handoff/artifacts/pr_quad_audit_v1'
HAVE = (BUNDLE/'data/validation_remaining.jsonl.gz').is_file()


@unittest.skipUnless(HAVE, 'needs the MONKEY_V2 bundle')
class CleanRowsTest(unittest.TestCase):
    def test_rows_match_direct_scoring_on_one_clean_image(self):
        import numpy as np
        from shapely.geometry import shape
        rows, policy, lr2, geoms, scores = da._context(BUNDLE)
        src = sorted(rows)[5]
        row = rows[src]
        gts = [list(np.asarray(shape(g['geometry']).exterior.coords)[:-1].reshape(-1)) for g in row['gt_clean']['valid'][:3]]
        sid = f'clean::{src}'
        with TemporaryDirectory() as tmp:
            for name, raws in (('B1', [.9, .55, .45]), ('B-Clean', [.8, .75, .65])):
                (Path(tmp)/name).mkdir()
                recs = [(sid, gi, gts, raws, raws) for gi in range(len(geoms))]
                da.write_shard(Path(tmp)/name/'shard_0000.npz', 'idx', [sid], recs)
            (Path(tmp)/'dump_identity.json').write_text(json.dumps(dict(identity_sha256='idx', gpu='x', condition='clean')))
            out, _ = csb1_clean.condition_rows(tmp, 'clean', rows, policy, geoms)
        e = dict(polygons=gts, raw=[.9, .55, .45], packed=[.9, .55, .45])
        c = dict(polygons=gts, raw=[.8, .75, .65], packed=[.8, .75, .65])
        self.assertEqual(out[src]['CSB1'], da.PoolScorer(e, c, row, policy).fused(*p0.CSB1))
        self.assertEqual(out[src]['B1 eps005/.50'][0], 2)          # two B1 boxes >= .5 sit exactly on GT
        self.assertEqual(out[src]['B1 .40/.60'][0], 1)             # only the .9 box survives .60


if __name__ == '__main__':
    unittest.main()
