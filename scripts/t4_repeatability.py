"""T4 run-to-run repeatability: the 40-source probe dump and the full dump were two separate T4 runs.
Compare their stored candidates (polygons + raw scores, all 9 geometries) on the shared sources."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mixab import dualagree as da  # noqa: E402

RUNS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('H:/My Drive/CS431_MixA/runs')
out = {}
for name in ('B-Clean', 'B1'):
    p = da.load_model_dump(RUNS/'dualagree_probe_v1_T4'/name, 9)
    f = da.load_model_dump(RUNS/'dualagree_dump_v1_T4'/name, 9)
    shared = sorted(set(p) & set(f))
    entries = same = 0
    diffs = []
    for s in shared:
        for g in range(9):
            entries += 1
            if p[s][g] == f[s][g]:
                same += 1
            elif len(diffs) < 5:
                diffs.append(dict(sample=s, geometry=g, n_probe=len(p[s][g]['raw']), n_full=len(f[s][g]['raw'])))
    out[name] = dict(shared_images=len(shared), entries=entries, identical_entries=same, examples=diffs)
print(json.dumps(out, indent=1))
(ROOT/'results/t4_repeatability.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
