#!/usr/bin/env bash
# One-command Mix A probe on a T4/A100 Linux machine (runbook steps 0-3).
# Edit the five paths below (or export them before calling), then: bash scripts/run_probe.sh
# Stops at the first failure. Never submits jobs, never pushes, never touches Tuan's repo.
set -euo pipefail

M="${M:-/path/to/MONKEY_V2}"                      # Tuan's checkout (dc19b85 or newer)
CLEAN_ROOT="${CLEAN_ROOT:-/path/to/textvqa}"      # contains train_images/
PREPARED_ROOT="${PREPARED_ROOT:-/path/to/prepared}" # receives validation_lr2/
CONFIG="${CONFIG:-/path/to/my_inference_config.py}" # copy of docs/handoff/inference_config.py, encoder paths fixed
PROBE="${PROBE:-40}"

cd "$(dirname "$0")/.."
export MONKEY_V2_ROOT="$M"
B="$M/docs/handoff/artifacts/pr_quad_audit_v1"
WEIGHTS=runtime/merge_weights_v1
OUT="runtime/merge_audit_probe_v1_$(date +%Y%m%d_%H%M)"
mkdir -p runtime

echo "== 0. tests (CPU) =="
python -m unittest discover -s tests -t . >/dev/null

echo "== 1. merged weights =="
if [ ! -f "$WEIGHTS/merge_manifest.json" ]; then
  python -m mixab.weight_merge --bclean "$M/docs/handoff/weights/bclean-inference.pth" \
    --b1 "$M/docs/handoff/weights/b1-inference.pth" --output "$WEIGHTS"
fi

echo "== 2. doctor (must be 0 FAIL; LR2 may still be WARN) =="
python -m mixab.doctor --bundle "$B" --clean-root "$CLEAN_ROOT" --prepared-root "$PREPARED_ROOT" \
  --model-config "$CONFIG" --weights-dir "$WEIGHTS" --output "$OUT.doctor.json"

echo "== 3. native LR2 images (doctor passed) =="
python -m mixab.merge_audit prepare-lr2 --bundle "$B" --clean-root "$CLEAN_ROOT" --prepared-root "$PREPARED_ROOT"

echo "== 4. probe: $PROBE sources x 2 conditions x 5 models =="
python -m mixab.merge_audit run --bundle "$B" --clean-root "$CLEAN_ROOT" --prepared-root "$PREPARED_ROOT" \
  --model-config "$CONFIG" --weights-dir "$WEIGHTS" \
  --bclean-export "$M/docs/handoff/weights/bclean-inference.pth" --b1-export "$M/docs/handoff/weights/b1-inference.pth" \
  --probe "$PROBE" --output "$OUT"

echo "== done: send $OUT/gates.json, $OUT/profile.json and $OUT.doctor.json =="
