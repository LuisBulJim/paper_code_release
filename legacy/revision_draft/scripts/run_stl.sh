#!/usr/bin/env bash
# Sequential Transfer: aerial pre-training (30 epochs, same hparams) then the SAME K-fold fine-tuning as DTL.
#   AERIAL_DATA=/path/to/dataset_aereo/data.yaml nohup bash scripts/run_stl.sh > runs/stl_nohup.log 2>&1 &
# Optional: MODELS="yolo26m" SEEDS="0 1 2" bash scripts/run_stl.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs
: "${AERIAL_DATA:?set AERIAL_DATA to the aerial dataset yaml}"
MODELS="${MODELS:-yolo26m}"
SEEDS="${SEEDS:-0 1 2}"
PY=".venv/bin/python"

$PY -m revision.protocol --check
for model in $MODELS; do
  if [ ! -f "runs/aerial/$model/run_info.json" ]; then  # run_info.json is written only when pre-training finished
    echo "=== $(date '+%F %T') aerial pre-training $model ==="
    $PY -m revision.train_aerial --model "$model" --data "$AERIAL_DATA" --epochs 30
  fi
  for seed in $SEEDS; do
    echo "=== $(date '+%F %T') STL fine-tuning $model seed $seed ==="
    $PY -m revision.train_kfold --model "$model" --seed "$seed" --tag stl --init-weights "runs/aerial/$model/weights/best.pt" --strict-env
  done
done
$PY -m revision.collect_runs --tags dtl,stl
echo "=== $(date '+%F %T') STL finished ==="
