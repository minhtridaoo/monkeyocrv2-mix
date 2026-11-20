"""Exploratory analysis after Mix A-core v1 (plan section 4.4, E-sel). Never replaces the v1 decision.

Rule fixed in the plan before running:
  t_ref  = LR2 recall <4 px on A of B1 at its R1-locked setting on A.
  feasible (model, setting): LR2 recall <4 px on A >= t_ref - 3 pp (exact Fractions).
  per model: best feasible setting by the R1 objective (sum of Clean + LR2 micro-Hmean on A), R1 tie-break.
  alpha*: R1 tie-break among models that have a feasible setting.
  Report on B (EXPLORATORY: B was already seen): mean Hmean vs both baselines (B-Clean at its R1
  setting, B1 at its constrained setting), tiny vs B1 - 3 pp, paired source bootstrap.
Also (descriptive): FP at matched recall. Per model, among settings whose LR2 recall on A >= B1's LR2
recall on A at its R1 setting, take the one with the fewest LR2 FP on A; report FP/recall on B.

Usage: python -m mixab.explore_selection --spec <reselect spec json> --output <new dir>
"""
import argparse
import json
from fractions import Fraction
from pathlib import Path

from mixab import read_json, require, save_json
from mixab import reselect_split as rs

TINY_MAX_DROP = rs.TINY_MAX_DROP          # 3 pp, the margin Tuan fixed on 20/11
LABEL = 'EXPLORATORY (plan 4.4 E-sel): B already seen; does not replace the Mix A-core v1 decision'


def tiny_lr2(total):
    """total [2, fields] -> exact LR2 recall of the <4 px bin (fields 3, 4), None if no GT."""
    gt, tp = int(total[1, 3]), int(total[1, 4])
    return Fraction(tp, gt) if gt else None


def lr2_recall(total):
    return Fraction(int(total[1, 0]), int(total[1, 1])) if int(total[1, 1]) else Fraction(0)


def lr2_fp(total):
    return int(total[1, 2]) - int(total[1, 0])


def constrained_setting(counts, settings, index, floor):
    """Best R1 objective among settings with tiny >= floor on `index`; None if none is feasible."""
    totals = counts[index].sum(axis=0)
    best, best_rank, feasible = None, None, 0
    for k, setting in enumerate(settings):
        tiny = tiny_lr2(totals[:, k, :])
        if tiny is None or tiny < floor:
            continue
        feasible += 1
        native = setting['geometry'] == 'base' and setting['min_text_score'] == .3
        rank = (rs.selection_value(totals[:, k, :]), native)
        if best_rank is None or rank > best_rank:
            best, best_rank = k, rank
    return (best, None if best is None else best_rank[0], feasible)


def matched_recall_setting(counts, index, recall_floor):
    """Fewest LR2 FP on `index` among settings with LR2 recall >= recall_floor (ties: grid order)."""
    totals = counts[index].sum(axis=0)
    best, best_fp = None, None
    for k in range(totals.shape[1]):
        if lr2_recall(totals[:, k, :]) >= recall_floor:
            fp = lr2_fp(totals[:, k, :])
            if best_fp is None or fp < best_fp:
                best, best_fp = k, fp
    return best


def analyse(models):
    import numpy as np

    loaded = []
    for spec in models:
        data = rs.load_counts(spec['npz'])
        loaded.append(dict(name=spec['name'], alpha=Fraction(str(spec['alpha'])), **data))
    first = loaded[0]
    for m in loaded[1:]:
        require(m['source_ids'] == first['source_ids'] and m['settings'] == first['settings'],
                f'{m["name"]}: sources/settings differ')
    sources, settings = first['source_ids'], first['settings']
    split = [rs.split_of(s) for s in sources]
    a_index = np.array([i for i, s in enumerate(split) if s == 'A'], dtype=np.int64)
    b_index = np.array([i for i, s in enumerate(split) if s == 'B'], dtype=np.int64)
    by_alpha = {m['alpha']: m for m in loaded}
    require(0 in by_alpha and 1 in by_alpha, 'Both endpoints are required')
    bclean, b1 = by_alpha[Fraction(0)], by_alpha[Fraction(1)]

    r1 = {m['name']: rs.choose_setting(m['counts'], settings, a_index)[0] for m in loaded}
    b1_a = b1['counts'][a_index].sum(axis=0)[:, r1[b1['name']], :]
    t_ref = tiny_lr2(b1_a)
    require(t_ref is not None, 'B1 has no tiny GT on A')
    floor = t_ref - TINY_MAX_DROP

    selection = {}
    for m in loaded:
        k, value, feasible = constrained_setting(m['counts'], settings, a_index, floor)
        a_tot = None if k is None else m['counts'][a_index].sum(axis=0)[:, k, :]
        selection[m['name']] = dict(alpha=str(m['alpha']), feasible_settings=feasible, setting_index=k,
                                    setting_key=None if k is None else rs.setting_key(settings[k]),
                                    value_A=None if k is None else str(value),
                                    mean_hmean_A=None if k is None else float(value/2),
                                    tiny_A=None if k is None else float(tiny_lr2(a_tot)),
                                    r1_setting_key=rs.setting_key(settings[r1[m['name']]]))
    candidates = [(m['alpha'], Fraction(selection[m['name']]['value_A'])) for m in loaded
                  if selection[m['name']]['setting_index'] is not None]
    alpha_star = rs.choose_alpha(candidates)
    star = by_alpha[alpha_star]

    def at(m, k):
        return m['counts'][:, :, k, :]

    def b_total(m, k):
        return at(m, k)[b_index].sum(axis=0)

    star_k = selection[star['name']]['setting_index']
    b1_k = selection[b1['name']]['setting_index']          # feasible by construction (its R1 setting is)
    bases = {bclean['name']: (bclean, r1[bclean['name']]), b1['name']: (b1, b1_k)}
    star_mean = rs.selection_value(b_total(star, star_k))/2
    exact_mean = {name: rs.selection_value(b_total(m, k))/2 for name, (m, k) in bases.items()}
    baselines = {name: rs.metrics(b_total(m, k)) for name, (m, k) in bases.items()}
    star_b = rs.metrics(b_total(star, star_k))
    tiny_star_b = tiny_lr2(b_total(star, star_k))
    tiny_b1_b = tiny_lr2(b_total(b1, b1_k))

    rng = np.random.default_rng(rs.BOOTSTRAP_SEED)
    indices = rng.integers(0, len(b_index), size=(rs.REPETITIONS, len(b_index)))
    boot = {}
    if star['alpha'] not in (0, 1):
        boot = {f'vs_{name}': rs.bootstrap(at(star, star_k), at(m, k), b_index, indices)
                for name, (m, k) in bases.items()}

    curve = {}
    for m in loaded:
        k = selection[m['name']]['setting_index']
        curve[m['name']] = None if k is None else dict(setting=rs.setting_key(settings[k]), **_flat(rs.metrics(b_total(m, k))))

    recall_floor = lr2_recall(b1_a)
    matched = {}
    for m in loaded:
        k = matched_recall_setting(m['counts'], a_index, recall_floor)
        if k is None:
            matched[m['name']] = None
            continue
        a_tot, b_tot = m['counts'][a_index].sum(axis=0)[:, k, :], b_total(m, k)
        matched[m['name']] = dict(setting=rs.setting_key(settings[k]), lr2_recall_A=float(lr2_recall(a_tot)),
                                  lr2_fp_A=lr2_fp(a_tot), lr2_recall_B=float(lr2_recall(b_tot)),
                                  lr2_fp_B=lr2_fp(b_tot), tiny_B=_f(tiny_lr2(b_tot)))

    decision = dict(
        label=LABEL, t_ref_tiny_A_B1=float(t_ref), tiny_floor_A=float(floor),
        alpha_star=str(alpha_star), alpha_star_model=star['name'], alpha_star_setting=selection[star['name']]['setting_key'],
        alpha_star_is_endpoint=alpha_star in (0, 1),
        mean_hmean_B_alpha_star=float(star_mean),
        baseline_mean_hmean_B={n: float(v) for n, v in exact_mean.items()},
        baseline_settings={n: rs.setting_key(settings[k]) for n, (_, k) in bases.items()},
        margin_pp=float(rs.MARGIN*100),
        passes_margin_exploratory=bool(alpha_star not in (0, 1) and all(star_mean >= v + rs.MARGIN for v in exact_mean.values())),
        tiny_B_alpha_star=_f(tiny_star_b), tiny_B_B1=_f(tiny_b1_b),
        passes_tiny_exploratory=bool(tiny_star_b is not None and tiny_b1_b is not None and
                                     tiny_star_b >= tiny_b1_b - TINY_MAX_DROP))
    return dict(decision=decision, selection=selection, alpha_star_B=_flat(star_b),
                baselines_B={n: _flat(v) for n, v in baselines.items()}, bootstrap_B=boot, curve_B=curve,
                matched_recall=dict(recall_floor_A=float(recall_floor),
                                    note='descriptive: fewest LR2 FP on A at LR2 recall >= B1 (R1 setting) on A',
                                    models=matched))


def _f(x):
    return None if x is None else float(x)


def _flat(metrics):
    out = {'mean_hmean': metrics['mean_hmean']}
    for c in rs.CONDITIONS:
        for key in ('hmean', 'precision', 'recall', 'fp', 'recall_lt4'):
            out[f'{c}_{key}'] = metrics[c][key]
    return out


def main():
    import sys
    sys.stdout.reconfigure(errors='backslashreplace')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--spec', required=True, type=Path, help='reselect spec (only "models" is used)')
    parser.add_argument('--output', required=True, type=Path)
    a = parser.parse_args()
    out_file = a.output/'explore_selection.json'
    require(not out_file.exists(), f'Refusing to overwrite {out_file}')
    result = analyse(read_json(a.spec)['models'])
    a.output.mkdir(parents=True, exist_ok=True)
    save_json(out_file, result)
    print(json.dumps(result['decision'], indent=1))


if __name__ == '__main__':
    main()
