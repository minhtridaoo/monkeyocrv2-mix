"""Alpha curve on split B for the Mix A-core v1 report (descriptive; alpha* was locked on A).

Reads results/mix_a_v1/evaluation_B.csv and decision.json, writes results/mix_a_v1/alpha_curve_B.png.
Run from the workspace root:  python scripts/plot_alpha_curve.py
"""
import csv
import json
from fractions import Fraction
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT/'results/mix_a_v1'
SURFACE, INK, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
SERIES = {'selected_on_A': ('#2a78d6', 'Setting chọn trên A (R1)'), 'common_030_040': ('#eb6834', 'Common .30/.40')}
PANELS = [('clean', 'hmean', 'Clean Hmean (%)', 100), ('lr2', 'hmean', 'LR2 Hmean (%)', 100),
          ('lr2', 'fp', 'LR2 FP (số box)', 1), ('lr2', 'recall_lt4', 'LR2 recall chữ <4 px (%)', 100)]


def main():
    rows = list(csv.DictReader((RES/'evaluation_B.csv').open(encoding='utf-8')))
    decision = json.loads((RES/'decision.json').read_text(encoding='utf-8'))
    star = float(Fraction(decision['alpha_star']))
    data = {}
    for r in rows:
        key = (r['operating_point'], r['condition'])
        data.setdefault(key, []).append((float(Fraction(r['alpha'])), r))
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                         'ytick.color': MUTED, 'text.color': INK})
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2), facecolor=SURFACE)
    for ax, (cond, field, title, scale) in zip(axes.flat, PANELS):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=.6)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for op, (color, label) in SERIES.items():
            pts = sorted(data[(op, cond)])
            xs = [a for a, _ in pts]
            ys = [float(r[field])*scale for _, r in pts]
            ax.plot(xs, ys, color=color, linewidth=2, marker='o', markersize=6, label=label,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
            if op == 'selected_on_A':
                y_star = ys[xs.index(star)]
                ax.plot([star], [y_star], marker='o', markersize=13, markerfacecolor='none', markeredgecolor=INK,
                        markeredgewidth=1.2)
        if field == 'recall_lt4':
            floor = decision['tiny_floor']*100
            ax.axhline(floor, color=MUTED, linewidth=1, linestyle='--')
            ax.text(0.0, floor + .6, f'mốc tiny: B1 − 3 pp = {floor:.1f}%', color=MUTED, ha='left', va='bottom', fontsize=8)
        ax.set_title(title, loc='left', fontsize=10, color=INK)
        ax.set_xticks([0, .25, .5, .75, 1])
        ax.set_xticklabels(['0\nB-Clean', '¼', '½', '¾', '1\nB1'])
    axes[0, 0].annotate('α* khóa trên A', xy=(star, float(next(r for a, r in sorted(data[('selected_on_A', 'clean')])
                                                                 if a == star)['hmean'])*100),
                        xytext=(4, -26), textcoords='offset points', fontsize=8, color=INK, va='top')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(.5, .985))
    fig.suptitle('Mix A-core v1 trên split B (A100) — chỉ mô tả; α* đã khóa trên A, không chọn lại trên B',
                 y=.935, fontsize=10, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, .9))
    out = RES/'alpha_curve_B.png'
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(str(out).encode('ascii', 'backslashreplace').decode())


if __name__ == '__main__':
    main()
