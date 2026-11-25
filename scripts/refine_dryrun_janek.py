"""CPU dry-run of the refinement pipeline WITHOUT a P-map (pipeline check only; numbers are meaningless and not read).
  * builds fake shard dirs from existing records: crops = 0, extras from the records' quads (approximate scale factor);
  * TRAIN: train (locked recipe) + select on TRAIN-calibration  -> checks loss finite, checkpoints, selection rule;
  * A: evaluate with an UNTRAINED (zero-init) checkpoint forced as 'chosen' -> v1R must equal v1 exactly and F / v1 must
    reproduce devA_result counts. No trained model is ever applied to A here.
Run from the workspace root:  python scripts/refine_dryrun_janek.py
"""
import gzip
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np  # noqa: E402

from mixab import boxquality_janek as bq  # noqa: E402
from mixab import refine_cmd_janek as rc  # noqa: E402
from mixab import refine_janek as rf  # noqa: E402
from mixab import save_json  # noqa: E402
from mixab.monkey_janek import monkey_root  # noqa: E402

TMP = ROOT/'runtime/scratch/refine_dryrun'
TRAIN_REC = ROOT/'runtime/kaggle_downloads/boxquality_T4_20261124-224054/out/extract'
A_REC = ROOT/'runtime/kaggle_downloads/boxquality_devA_T4_20261125-084951/out/extract'
ROWS = ROOT/'runtime/kaggle_upload/cs431-boxquality/inputs/train_rows.jsonl.gz'
ART = ROOT/'docs/handoff/artifacts/janek/boxquality_v1_T4'
BUNDLE = monkey_root()/'docs/handoff/artifacts/pr_quad_audit_v1'


def fake(src, dst):
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    recs = bq.load_records(src)
    for k in range(0, len(recs), 50):
        chunk, Cs, Es = recs[k:k + 50], [], []
        for r in chunk:
            W, H = r['size']
            clean = r['size'] if r['condition'] == 'clean' else [int(round(W*2)), int(round(H*2))]
            r['restore'] = [1., 1.] if r['condition'] == 'clean' else [2., 2.]
            h, w = r['img_shape']
            r['scale_factor'] = [w/W, h/H]
            _ = clean
            C = np.zeros((len(r['raw']), rf.GRID_T, rf.GRID_S), np.float16)
            E = np.stack([rf.extras(rc.net_quad(p, r['restore'], r['scale_factor']), v) for p, v in zip(r['polygons'], r['raw'])]) \
                if r['raw'] else np.zeros((0, rf.N_EXTRA), np.float32)
            Cs.append(C); Es.append(E)
        rc._save_shard(dst, k//50, chunk, Cs, Es)
    return len(recs)


def ns(**k):
    import argparse
    return argparse.Namespace(**k)


def main():
    common = dict(bundle=BUNDLE, model=ART/'model_janek.npz', fit=ART/'boxquality_fit_janek.json')
    if not (TMP/'selection.json').is_file() or '--fresh' in sys.argv:
        print('fake TRAIN shards', fake(TRAIN_REC, TMP/'train_extract'), flush=True)
        print('fake A shards', fake(A_REC, TMP/'a_extract'), flush=True)
        rc.train(ns(rows=ROWS, extract_dir=TMP/'train_extract', output=TMP/'train', **common))
        rc.select(ns(rows=ROWS, extract_dir=TMP/'train_extract', train_dir=TMP/'train', output=TMP/'selection.json', **common))
    # A: untrained checkpoint, forced
    z = TMP/'zero'
    z.mkdir(exist_ok=True)
    m = rf.make_model()
    rc._save_ckpt(z/'ckpt_e00.pt', m, dict(mean=np.zeros(rf.IN_DIM, np.float32), std=np.ones(rf.IN_DIM, np.float32)))
    save_json(z/'selection.json', dict(chosen='e00'))
    out = z/'eval'
    if out.exists():
        shutil.rmtree(out)
    rc.evaluate(ns(rows=None, extract_dir=TMP/'a_extract', train_dir=z, selection=z/'selection.json', output=out,
                   devA_ref=ROOT/'docs/handoff/artifacts/janek/boxquality_devA_20261125/devA_result_janek.json', **common))
    r = json.loads((out/'refine_devA.json').read_text(encoding='utf-8'))
    same = all(r['arms']['v1R'][c]['tp'] == r['arms']['v1'][c]['tp'] and r['arms']['v1R'][c]['pred'] == r['arms']['v1'][c]['pred']
               for c in ('clean', 'lr2'))
    print('DRYRUN offset0 == v1 on A:', same, '· reproduces devA_result:', r['reproduces_devA_result'], '· status', r['diag']['status'])


if __name__ == '__main__':
    main()
