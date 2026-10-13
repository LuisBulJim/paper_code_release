#!/usr/bin/env bash
# SAM 3 box-prompted segmentation on the LIACI official test split with one final detector checkpoint.
#   SAM3_CKPT=runs/dtl/yolo26m_seed0/fold0/weights/best.pt bash scripts/run_sam3.sh
# Requires transformers with Sam3Model (pip install -U "transformers>=5.13" huggingface-hub) and access to
# facebook/sam3 on the Hugging Face Hub (HF_TOKEN if the checkpoint is gated). data/sam3_eval is shipped.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
: "${SAM3_CKPT:?set SAM3_CKPT to the detector checkpoint}"
OUT="runs/sam3/$(echo "$SAM3_CKPT" | cut -d/ -f2-4 | tr / _)"   # e.g. runs/sam3/dtl_yolo26m_seed0_fold0

$PY -c "from transformers import Sam3Model, Sam3Processor; print('SAM 3 classes available')"
echo "=== $(date '+%F %T') SAM 3 predict (standard detector) ==="
$PY -m revision.sam3_eval predict --checkpoint "$SAM3_CKPT" --out "${OUT}_standard"
$PY -m revision.sam3_eval metrics --preds "${OUT}_standard" --out results --figure figures/sam3_cases.png
if [ "${SAM3_TTA:-0}" = "1" ]; then
  echo "=== $(date '+%F %T') SAM 3 predict (TTA detector) ==="
  $PY -m revision.sam3_eval predict --checkpoint "$SAM3_CKPT" --tta --out "${OUT}_tta"
  $PY -m revision.sam3_eval metrics --preds "${OUT}_tta" --out results/sam3_tta --figure figures/sam3_cases_tta.png
fi
echo "=== $(date '+%F %T') SAM 3 finished ==="
