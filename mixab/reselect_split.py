"""Mix A-core step A3: split A/B, select on A with rule R1, lock, evaluate on B.

Implements Ke_hoach_phoi_hop_MixA_MixB_20261119.md section 4.2 on per-source count matrices in the
MONKEY_V2 decision-audit layout (`rev4/decision_audit.py`, NPZ keys source_ids/settings/counts):

    counts[source, condition, setting, field], condition 0 = clean, 1 = lr2
    5 fields  (Tuan's audit):  tp, gt, pred, tiny_gt, tiny_tp            (tiny = GT height bin < 4 px)
    11 fields (merge audit):   tp, gt, pred, gt_b0, tp_b0, ..., gt_b3, tp_b3   bins <4, 4-8, 8-16, >=16 px

R1:
1. Split: SHA-256 of the UTF-8 source ID; even integer -> A, odd -> B. Clean/LR2 of a source share it.
2. Per model on A: maximise exact Fraction micro-Hmean(Clean) + micro-Hmean(LR2). Ties: setting
   `base` with score .30 (as rev4/decision_core.choose_setting), then first in grid order.
3. alpha* on A among all alphas at their own A-selected settings. Ties: endpoint before interior
   alpha, then larger alpha.
4. Lock, then report every model on B. Only the locked alpha* is judged by the continuation rule:
   interior alpha*, mean Hmean on B >= max(both endpoints) + 0.5 pp, LR2 recall <4 px >= 42 %.
   These two thresholds are pragmatic pilot thresholds, not proof of improvement.
5. Paired source bootstrap on B: 2,000 replicates, seed 20261107, micro metrics recomputed per
   replicate; indices drawn once as in rev4/decision_audit.py.

Verification status (section 4.3): VERIFIED only if gate 3a, the endpoint P-map check and gate 3b
all PASS; any FAIL -> INVALIDATED; otherwise PENDING_REFERENCE_CHECK. Missing checks are reported
as not run, never as PASS.
"""
import argparse
import csv
import hashlib
import json
from fractions import Fraction
from pathlib import Path

from mixab import read_json, require, save_json, sha256

STAGE = 'Mix A-core A->B reselection v1'
BOOTSTRAP_SEED = 20261107
REPETITIONS = 2000
MARGIN = Fraction(1, 200)          # 0.5 percentage point
TINY_MIN = Fraction(42, 100)       # LR2 recall of GT < 4 px
BINS = ('lt4', '4to8', '8to16', 'ge16')
CONDITIONS = ('clean', 'lr2')


def split_of(source_id):
    return 'A' if int(hashlib.sha256(str(source_id).encode('utf-8')).hexdigest(), 16) % 2 == 0 else 'B'


def setting_key(setting):
    return f'{setting["geometry"]}/score_{setting["min_text_score"]:.2f}'


def load_counts(path):
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        source_ids = [str(s) for s in data['source_ids'].tolist()]
        settings = [json.loads(str(s)) for s in data['settings'].tolist()]
        counts = np.asarray(data['counts'], dtype=np.int64)
    require(counts.ndim == 4 and counts.shape[1] == 2, f'{path}: counts must be [source, 2, setting, field]')
    require(counts.shape[0] == len(source_ids) == len(set(source_ids)), f'{path}: source_ids mismatch or duplicates')
    require(counts.shape[2] == len(settings), f'{path}: settings length differs from counts')
    require(counts.shape[3] in (5, 11), f'{path}: expected 5 or 11 count fields, got {counts.shape[3]}')
    require(bool((counts >= 0).all()), f'{path}: negative counts')
    require(bool((counts[..., 0] <= counts[..., 1]).all() and (counts[..., 0] <= counts[..., 2]).all()),
            f'{path}: tp exceeds gt or pred')
    return dict(path=str(path), sha256=sha256(path), source_ids=source_ids, settings=settings, counts=counts)


def bin_fields(fields):
    """Index pairs (gt, tp) of available height bins."""
    return [(3, 4)] if fields == 5 else [(3 + 2*i, 4 + 2*i) for i in range(4)]


def exact_hmean(tp, gt, pred):
    return Fraction(2*int(tp), int(gt) + int(pred)) if int(gt) + int(pred) else Fraction(0)


def metrics(total):
    """total: array [2, fields] summed over sources -> per-condition metrics."""
    fields = total.shape[-1]
    out = {}
    for c, condition in enumerate(CONDITIONS):
        tp, gt, pred = (int(x) for x in total[c, :3])
        row = dict(tp=tp, gt=gt, pred=pred, fp=pred-tp, fn=gt-tp,
                   precision=tp/pred if pred else 0., recall=tp/gt if gt else 0.,
                   hmean=float(exact_hmean(tp, gt, pred)))
        for name, (g, t) in zip(BINS, bin_fields(fields)):
            bgt, btp = int(total[c, g]), int(total[c, t])
            row[f'recall_{name}'] = btp/bgt if bgt else None
            row[f'gt_{name}'] = bgt
        out[condition] = row
    out['mean_hmean'] = (out['clean']['hmean'] + out['lr2']['hmean'])/2
    return out


def selection_value(total):
    return exact_hmean(*total[0, :3]) + exact_hmean(*total[1, :3])


def choose_setting(counts, settings, index):
    """R1 step 2 on the given source indices. Returns (setting index, exact value)."""
    totals = counts[index].sum(axis=0)          # [2, setting, field]
    best, best_rank = None, None
    for k, setting in enumerate(settings):
        native = setting['geometry'] == 'base' and setting['min_text_score'] == .3
        rank = (selection_value(totals[:, k, :]), native)
        if best_rank is None or rank > best_rank:   # strict: earlier grid order wins remaining ties
            best, best_rank = k, rank
    return best, best_rank[0]


def choose_alpha(candidates):
    """candidates: list of (alpha Fraction, value Fraction). R1 step 3."""
    def rank(item):
        alpha, value = item
        return value, alpha in (0, 1), alpha
    return max(candidates, key=rank)[0]


def common_index(settings):
    hits = [k for k, s in enumerate(settings)
            if s['geometry'] == 'base' and s['min_text_score'] == .4 and s.get('mask_thr', .3) == .3]
    require(len(hits) == 1, f'Expected one common base .30/.40 setting, found {len(hits)}')
    return hits[0]


def compare_reference(model, reference):
    """Gate 3b: join by (source_id, condition, setting_key); compare the 5 shared columns."""
    model_settings = {setting_key(s): k for k, s in enumerate(model['settings'])}
    model_sources = {s: i for i, s in enumerate(model['source_ids'])}
    shared = [0, 1, 2, 3, 4]        # tp, gt, pred, gt_b0/tiny_gt, tp_b0/tiny_tp
    checked = mismatched = missing = 0
    examples = []
    for ri, source in enumerate(reference['source_ids']):
        mi = model_sources.get(source)
        for rk, setting in enumerate(reference['settings']):
            mk = model_settings.get(setting_key(setting))
            for c, condition in enumerate(CONDITIONS):
                checked += 1
                if mi is None or mk is None:
                    missing += 1
                    continue
                left = model['counts'][mi, c, mk, shared].tolist()
                right = reference['counts'][ri, c, rk, :5].tolist()
                if left != right:
                    mismatched += 1
                    if len(examples) < 5:
                        examples.append(dict(source_id=source, condition=condition,
                                             setting=setting_key(setting), merge=left, reference=right))
    status = 'PASS' if checked and not mismatched and not missing else 'FAIL'
    return dict(status=status, checked=checked, mismatched=mismatched, missing=missing, examples=examples)


def verification_status(checks):
    values = [checks.get(k, 'NOT_RUN') for k in ('gate_3a', 'pmap_endpoint', 'gate_3b')]
    if 'FAIL' in values:
        return 'INVALIDATED'
    if all(v == 'PASS' for v in values):
        return 'VERIFIED'
    return 'PENDING_REFERENCE_CHECK'


def bootstrap(candidate, base, b_index, indices):
    """Paired source bootstrap of candidate - base on split B, at each model's locked setting.

    candidate/base: arrays [source, 2, fields] at the locked setting. indices: [reps, nB] into b_index.
    """
    import numpy as np

    cand = candidate[b_index][:, :, :3]
    ref = base[b_index][:, :, :3]

    def hm(x):
        return 2*x[..., 0]/np.maximum(x[..., 1] + x[..., 2], 1)
    sc, sr = cand[indices].sum(axis=1), ref[indices].sum(axis=1)       # [reps, 2, 3]
    oc, orr = cand.sum(axis=0), ref.sum(axis=0)
    out = {}
    for c, condition in enumerate(CONDITIONS + ('mean',)):
        if condition == 'mean':
            delta = 100*((hm(sc[:, 0]) + hm(sc[:, 1]))/2 - (hm(sr[:, 0]) + hm(sr[:, 1]))/2)
            point = 100*((hm(oc[0]) + hm(oc[1]))/2 - (hm(orr[0]) + hm(orr[1]))/2)
        else:
            delta = 100*(hm(sc[:, c]) - hm(sr[:, c]))
            point = 100*(hm(oc[c]) - hm(orr[c]))
        out[condition] = dict(delta_pp=float(point), ci95_pp=[float(q) for q in np.quantile(delta, [.025, .975])])
    return out


def run(models, references, checks, output):
    """models: list of dict(name, alpha (str|None), npz). Endpoints alpha 0 = B-Clean, 1 = B1 required."""
    import numpy as np

    output = Path(output)
    names = ['split_manifest.csv', 'selection.json', 'evaluation_B.csv', 'bootstrap_B.json', 'decision.json']
    clash = [n for n in names if (output/n).exists()]
    require(not clash, f'Refusing to overwrite existing outputs in {output}: {clash}')
    loaded = []
    for spec in models:
        data = load_counts(spec['npz'])
        alpha = None if spec.get('alpha') is None else Fraction(str(spec['alpha']))
        require(alpha is None or 0 <= alpha <= 1, f'{spec["name"]}: alpha outside [0, 1]')
        loaded.append(dict(name=spec['name'], alpha=alpha, **data))
    first = loaded[0]
    for item in loaded[1:]:
        require(item['source_ids'] == first['source_ids'], f'{item["name"]}: source order differs')
        require(item['settings'] == first['settings'], f'{item["name"]}: settings/grid order differs')
    alphas = [m['alpha'] for m in loaded if m['alpha'] is not None]
    require(len(alphas) == len(set(alphas)), 'Duplicate alpha values')
    require(0 in alphas and 1 in alphas, 'Both endpoints (alpha 0 = B-Clean, alpha 1 = B1) are required')
    sources, settings = first['source_ids'], first['settings']
    split = [split_of(s) for s in sources]
    a_index = np.array([i for i, s in enumerate(split) if s == 'A'], dtype=np.int64)
    b_index = np.array([i for i, s in enumerate(split) if s == 'B'], dtype=np.int64)
    require(len(a_index) and len(b_index), 'Split A or B is empty')
    common = common_index(settings)

    # R1 step 2: per-model setting on A only.
    selection = {}
    for m in loaded:
        k, value = choose_setting(m['counts'], settings, a_index)
        selection[m['name']] = dict(alpha=None if m['alpha'] is None else str(m['alpha']), setting_index=k,
                                    setting=settings[k], setting_key=setting_key(settings[k]),
                                    value_A=str(value), value_A_float=float(value))
    # R1 step 3: alpha* on A.
    by_alpha = {m['alpha']: m for m in loaded if m['alpha'] is not None}
    alpha_star = choose_alpha([(a, Fraction(selection[by_alpha[a]['name']]['value_A'])) for a in by_alpha])
    star = by_alpha[alpha_star]

    # Lock, then evaluate everything on B.
    rows, results = [], {}
    for m in loaded:
        results[m['name']] = {}
        for kind, k in (('common_030_040', common), ('selected_on_A', selection[m['name']]['setting_index'])):
            result = metrics(m['counts'][b_index][:, :, k, :].sum(axis=0))
            results[m['name']][kind] = result
            for condition in CONDITIONS:
                rows.append(dict(model=m['name'], alpha='' if m['alpha'] is None else str(m['alpha']),
                                 operating_point=kind, setting=setting_key(settings[k]), condition=condition,
                                 **result[condition], mean_hmean=result['mean_hmean']))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(b_index), size=(REPETITIONS, len(b_index)))
    locked = {m['name']: m['counts'][:, :, selection[m['name']]['setting_index'], :] for m in loaded}
    endpoints = {0: by_alpha[Fraction(0)]['name'], 1: by_alpha[Fraction(1)]['name']}
    boot = {}
    for a, m in sorted(by_alpha.items()):
        if a in (0, 1):
            continue
        boot[m['name']] = {f'vs_{endpoints[e]}': bootstrap(locked[m['name']], locked[endpoints[e]], b_index, indices)
                           for e in (0, 1)}

    star_b = results[star['name']]['selected_on_A']

    def exact_mean_b(name):   # exact Fraction mean Hmean on B at the A-locked setting
        return selection_value(locked[name][b_index].sum(axis=0))/2
    star_exact = exact_mean_b(star['name'])
    best_exact = max(exact_mean_b(endpoints[e]) for e in (0, 1))
    tiny_total = locked[star['name']][b_index].sum(axis=0)[1]          # LR2 row, bin 0 = <4 px
    tiny_exact = Fraction(int(tiny_total[4]), int(tiny_total[3])) if int(tiny_total[3]) else None
    decision = dict(
        alpha_star=str(alpha_star), alpha_star_model=star['name'], alpha_star_is_endpoint=alpha_star in (0, 1),
        mean_hmean_B_alpha_star=float(star_exact), best_endpoint_mean_hmean_B=float(best_exact),
        margin_pp=float(MARGIN*100), tiny_threshold=float(TINY_MIN),
        lr2_recall_lt4_alpha_star=None if tiny_exact is None else float(tiny_exact),
        passes_margin=bool(alpha_star not in (0, 1) and star_exact >= best_exact + MARGIN),
        passes_tiny=bool(tiny_exact is not None and tiny_exact >= TINY_MIN))
    interior = [a for a in by_alpha if a not in (0, 1)]
    decision['mode'] = 'mix_a' if interior else 'baseline_only'
    if interior:
        decision['continue_mix_a'] = decision['passes_margin'] and decision['passes_tiny']
        decision['note'] = ('merge did not win on A' if alpha_star in (0, 1) else
                            'only the A-locked alpha* is judged; other alphas on B are descriptive')
    else:
        # Endpoints only: a baseline reselection, not a Mix A experiment; nothing is judged.
        decision.update(continue_mix_a=None, passes_margin=None,
                        note='baseline_only: no interior alpha was evaluated; continuation rule not applied')
    ref_checks = {name: compare_reference(next(m for m in loaded if m['name'] == name), load_counts(path))
                  for name, path in references.items()}
    checks = dict(checks)
    if ref_checks:
        checks['gate_3b'] = 'PASS' if all(r['status'] == 'PASS' for r in ref_checks.values()) else 'FAIL'
    decision['verification'] = verification_status(checks)
    decision['checks'] = dict(gate_3a=checks.get('gate_3a', 'NOT_RUN'), pmap_endpoint=checks.get('pmap_endpoint', 'NOT_RUN'),
                              gate_3b=checks.get('gate_3b', 'NOT_RUN'))

    output.mkdir(parents=True, exist_ok=True)
    with (output/'split_manifest.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['source_id', 'split'])
        writer.writerows(zip(sources, split))
    save_json(output/'selection.json', dict(stage=STAGE, rule='R1', split_sizes=dict(A=len(a_index), B=len(b_index)),
                                            split_manifest_sha256=sha256(output/'split_manifest.csv'),
                                            inputs=[dict(name=m['name'], npz=m['path'], sha256=m['sha256']) for m in loaded],
                                            selection=selection, alpha_star=str(alpha_star)))
    fields = list(rows[0])
    with (output/'evaluation_B.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    save_json(output/'bootstrap_B.json', dict(seed=BOOTSTRAP_SEED, repetitions=REPETITIONS, unit='source on split B',
                                              comparisons=boot,
                                              caveat='descriptive post-selection CI; single training seed; '
                                                     'remaining-validation already influenced earlier choices'))
    save_json(output/'decision.json', dict(decision, reference_checks=ref_checks))
    return dict(selection=selection, results=results, bootstrap=boot, decision=decision)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--spec', required=True, type=Path,
                        help='JSON: {"models": [{"name", "alpha", "npz"}], "references": {name: npz}, '
                             '"checks": {"gate_3a": "PASS|FAIL", "pmap_endpoint": "PASS|FAIL"}}')
    parser.add_argument('--output', required=True, type=Path)
    a = parser.parse_args()
    spec = read_json(a.spec)
    out = run(spec['models'], spec.get('references', {}), spec.get('checks', {}), a.output)
    print(json.dumps(out['decision'], ensure_ascii=True, indent=1))


if __name__ == '__main__':
    main()
