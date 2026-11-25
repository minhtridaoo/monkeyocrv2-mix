"""Box-quality scorer · P0 checks (CPU, no GPU, no model fitting). Output: docs/handoff/artifacts/janek/boxquality_p0_20261124/

1. TextOCR train JSON SHA = Tuan's TEXTOCR_TRAIN_SHA256 (rev4/train_recovery.py:20).
2. Vendored gt_for_source re-creates gt_clean (+ removed ids) of all 2,262 frozen development rows with the frozen
   roi_clean: functional equivalence of the label code. Reads dev GT only to check code; fits nothing.
3. Deviation of the TRAIN rule (ROI = full canvas; historical perspective metadata of TRAIN is HPC-only): on the same
   dev rows, how many GT change status when roi_clean is replaced by the full canvas.
4. TRAIN membership: unique, disjoint from dev by id AND by clean SHA-256, present in TextOCR, >= 1 valid GT.
5. Locked selection (mixab.boxquality_janek.SELECTION_RULE) -> train_rows.jsonl.gz + selection.csv + GT stats per fold.
"""
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding='utf-8')
from shapely.geometry import box, shape  # noqa: E402
from mixab import boxquality_janek as bq # noqa: E402
from mixab import merge_audit_janek as ma # noqa: E402
from mixab.monkey_janek import monkey_root  # noqa: E402
from mixab.vendor import shared_manifest_janek as sm # noqa: E402

MV2 = monkey_root()
TEXTOCR = ROOT/'runtime/external/textocr/TextOCR_0.1_train.json'
TRAIN_IDS = MV2/'docs/handoff/data/train_ids.csv'
OUT = ROOT/'docs/handoff/artifacts/janek/boxquality_p0_20261124'


def canon(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = dict(label='P0 checks for the box-quality scorer (CPU). No GPU, no fitting, no TEST.')
    res['textocr_sha256'] = bq.sha256_file(TEXTOCR)
    res['textocr_sha_matches_tuan'] = res['textocr_sha256'] == bq.TEXTOCR_TRAIN_SHA256
    res['vendored_shared_manifest_sha256'] = bq.sha256_file(ROOT/'mixab/vendor/shared_manifest_janek.py')
    textocr, report = sm.normalize_textocr(json.loads(TEXTOCR.read_text(encoding='utf-8')))
    res['textocr'] = {k: report[k] for k in ('canonical_images', 'canonical_annotations', 'mapping_source')}
    cfg = sm.DEFAULT_SHARED_CONFIG
    rows = ma.load_rows(MV2/'docs/handoff/artifacts/pr_quad_audit_v1')
    exact = removed_ok = 0
    examples = []
    changed_sources, changes, valid_lost = 0, Counter(), 0
    for s, r in rows.items():
        gt, removed = sm.gt_for_source(s, textocr, shape(r['roi_clean']), cfg, Counter())
        same = canon(gt) == canon(r['gt_clean'])
        exact += same
        removed_ok += sorted(removed) == sorted(r['removed_gt_ids'])
        if not same and len(examples) < 5:
            examples.append(s)
        W, H = r['clean_size']
        gf, _ = sm.gt_for_source(s, textocr, box(0, 0, W, H), cfg, Counter())
        a = {e['ann_id']: 'valid' for e in r['gt_clean']['valid']} | {e['ann_id']: 'ignored' for e in r['gt_clean']['ignored']}
        b = {e['ann_id']: 'valid' for e in gf['valid']} | {e['ann_id']: 'ignored' for e in gf['ignored']}
        ch = [k for k in set(a) | set(b) if a.get(k) != b.get(k)]
        if ch:
            changed_sources += 1
            for k in ch:
                changes[f'{a.get(k, "removed")}->{b.get(k, "removed")}'] += 1
        # geometry of valid GT also changes when clipped by a smaller ROI
        valid_lost += sum(1 for e in r['gt_clean']['valid'] if shape(e['geometry']).area < .999*shape(
            next(x for x in gf['valid'] + gf['ignored'] if x['ann_id'] == e['ann_id'])['geometry']).area)
    res['dev_label_code_equivalence'] = dict(rows=len(rows), gt_clean_exact=exact, removed_ids_exact=removed_ok,
                                             examples_differing=examples)
    res['dev_full_canvas_deviation'] = dict(
        sources_with_status_change=changed_sources, status_changes=dict(changes),
        valid_gt_with_clipped_geometry=valid_lost, dev_valid_gt=sum(len(r['gt_clean']['valid']) for r in rows.values()),
        dev_sources_roi_below_canvas=sum(shape(r['roi_clean']).area < .999*r['clean_size'][0]*r['clean_size'][1] for r in rows.values()),
        note='Sizes the label noise of the TRAIN rule only; development labels and the evaluator are unchanged.')
    train = list(csv.DictReader(open(TRAIN_IDS, encoding='utf-8')))
    tid = [t['source_image_id'] for t in train]
    dev_sha = {r['clean_sha256'] for r in rows.values()}
    zero, gts = [], {}
    for t in train:
        im = textocr['imgs'].get(t['source_image_id'])
        if im is None:
            continue
        row = bq.train_row(t['source_image_id'], textocr, im['width'], im['height'], sm)
        gts[t['source_image_id']] = row
        if not row['gt_clean']['valid']:
            zero.append(t['source_image_id'])
    res['train'] = dict(count=len(tid), unique=len(set(tid)), train_ids_sha256=bq.sha256_file(TRAIN_IDS),
                        train_ids_sha_ok=bq.sha256_file(TRAIN_IDS) == bq.TRAIN_IDS_SHA256,
                        id_overlap_dev=len(set(tid) & set(rows)), clean_sha_overlap_dev=len({t['clean_sha256'] for t in train} & dev_sha),
                        clean_sha_unique=len({t['clean_sha256'] for t in train}),
                        in_textocr=len(gts), zero_valid_gt_full_canvas=len(zero),
                        relpath_rule_ok=all(t['clean_relpath'] == f'train_images/{t["source_image_id"]}.jpg' for t in train))
    eligible = [t for t in train if t['source_image_id'] in gts and t['source_image_id'] not in set(zero)]
    sel = bq.select_train_sources(eligible)
    stats = {}
    for role in ('smoke', 'fit', 'calibration'):
        xs = [x for x in sel if x['role'] == role]
        v = [len(gts[x['source_image_id']]['gt_clean']['valid']) for x in xs]
        ig = [len(gts[x['source_image_id']]['gt_clean']['ignored']) for x in xs]
        stats[role] = dict(sources=len(xs), valid_gt=sum(v), ignored_gt=sum(ig), ignored_share=sum(ig)/max(sum(v) + sum(ig), 1),
                           max_valid_per_source=max(v))
    codex = {r['source_image_id'] for r in csv.DictReader(open(ROOT/'docs/handoff/artifacts/janek/boxquality_preparation_20261124/proposed_train_sources_janek.csv', encoding='utf-8'))}
    res['selection'] = dict(rule=bq.SELECTION_RULE, eligible=len(eligible), folds=stats,
                            selection_sha256=hashlib.sha256(canon(sel).encode()).hexdigest(),
                            overlap_with_codex_proposal=len({x['source_image_id'] for x in sel if x['role'] != 'smoke'} & codex),
                            codex_proposal_reproducible=False,
                            codex_note='No script or rule was recorded for proposed_train_sources.csv; common hash rules give only '
                                       '~250-280/2000 overlap (chance). Replaced by the documented rule above.')
    import gzip
    with gzip.open(OUT/'train_rows.jsonl_janek.gz', 'wt', encoding='utf-8') as f:
        for x in sel:
            f.write(json.dumps(dict(gts[x['source_image_id']], clean_relpath=x['clean_relpath'], clean_sha256=x['clean_sha256'],
                                    role=x['role'])) + '\n')
    with (OUT/'selection_janek.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, ['source_image_id', 'clean_relpath', 'clean_sha256', 'role', 'width', 'height'])
        w.writeheader()
        for x in sel:
            W, H = gts[x['source_image_id']]['clean_size']
            w.writerow(dict(x, width=W, height=H))
    res['train_rows_sha256'] = bq.sha256_file(OUT/'train_rows.jsonl_janek.gz')
    (OUT/'p0_janek.json').write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
