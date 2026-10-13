#!/usr/bin/env bash
# TrashCan cross-domain verification: retraining with the LIACI recipe (3 seeds) + zero-shot OOD test.
#   nohup bash scripts/run_trashcan.sh > runs/trashcan_nohup.log 2>&1 &
# data/trashcan_v2 must exist (revision.trashcan_prepare, run locally and shipped, or run here with the
# raw release: TRASHCAN_SOURCE=/path/to/material_version TRASHCAN_FORMAT=coco).
# Optional: OOD_MODELS=yolo26m,yolo26l bash scripts/run_trashcan.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
OOD_MODELS="${OOD_MODELS:-yolo26m}"

# TTA settings come from the LIACI validation choice (run scripts/run_inference.sh first)
[ -f results/inference_choice.json ] || { echo "results/inference_choice.json missing: run scripts/run_inference.sh first"; exit 1; }
if [ ! -f data/trashcan_manifest.json ]; then
  echo "=== $(date '+%F %T') preparing TrashCan ==="
  $PY -m revision.trashcan_prepare ${TRASHCAN_SOURCE:+--source "$TRASHCAN_SOURCE"} ${TRASHCAN_FORMAT:+--format "$TRASHCAN_FORMAT"}
fi
echo "=== $(date '+%F %T') TrashCan retraining ==="
$PY -m revision.trashcan_train --strict-env
echo "=== $(date '+%F %T') zero-shot OOD: LIACI models on TrashCan metal ==="
$PY -m revision.trashcan_ood --models "$OOD_MODELS"
echo "=== $(date '+%F %T') TrashCan finished ==="
