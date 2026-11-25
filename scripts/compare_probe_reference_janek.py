"""Gate-3b-style comparison of probe endpoint counts against Tuan's H200 NPZ, restricted to the probe sources.

Usage (workspace root):  python scripts/compare_probe_reference_janek.py <probe_run_dir> [out.json]
Reports, per endpoint: mismatched cells over all 72 settings x 2 conditions, affected sources,
max |diff| per column, and the largest |delta Hmean| (pp) over the probe sources at any setting.
Descriptive only: it does not change any gate status.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mixab.monkey_janek import monkey_root  # noqa: E402
from mixab.reselect_split_janek import CONDITIONS, load_counts, setting_key  # noqa: E402

COLUMNS = ['tp', 'gt', 'pred', 'tiny_gt', 'tiny_tp']


def hmean(x):
    return 2*x[..., 0].sum()/max(x[..., 1].sum() + x[..., 2].sum(), 1)


def compare(probe_npz, ref_npz):
    m, r = load_counts(probe_npz), load_counts(ref_npz)
    ri = {s: i for i, s in enumerate(r['source_ids'])}
    rk = {setting_key(s): k for k, s in enumerate(r['settings'])}
    order = [rk[setting_key(s)] for s in m['settings']]
    ref = r['counts'][[ri[s] for s in m['source_ids']]][:, :, order, :5]
    got = m['counts'][..., :5]
    d = got - ref
    cell = (d != 0).any(-1)
    dh = [abs(hmean(got[:, c, k]) - hmean(ref[:, c, k]))*100 for k in range(got.shape[2]) for c in range(2)]
    return dict(sources=len(m['source_ids']), cells=int(cell.size), mismatched_cells=int(cell.sum()),
                mismatched_by_condition={c: int(cell[:, i].sum()) for i, c in enumerate(CONDITIONS)},
                sources_with_diff=int(cell.any(axis=(1, 2)).sum()),
                max_abs_diff=dict(zip(COLUMNS, np.abs(d).max(axis=(0, 1, 2)).tolist())),
                net_diff=dict(zip(COLUMNS, d.sum(axis=(0, 1, 2)).tolist())),
                max_abs_delta_hmean_pp=float(max(dh)))


def main():
    sys.stdout.reconfigure(errors='backslashreplace')
    run = Path(sys.argv[1])
    ref_dir = monkey_root()/'docs/handoff/artifacts/pr_quad_audit_v1/counts'
    out = {name: compare(run/f'{name}_source_counts.npz', ref_dir/f'{name}_source_counts.npz')
           for name in ('B-Clean', 'B1')}
    text = json.dumps(out, indent=1)
    print(text)
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
