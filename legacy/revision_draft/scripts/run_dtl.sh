#!/usr/bin/env bash
# Direct Transfer: YOLO26 n/m/l x seeds x 5 folds on the RTX 4090. Resumable: finished folds are skipped.
#   nohup bash scripts/run_dtl.sh > runs/dtl_nohup.log 2>&1 &
#   tail -f runs/dtl_nohup.log
# Optional: MODELS="yolo26m" SEEDS="0" bash scripts/run_dtl.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs
MODELS="${MODELS:-yolo26n yolo26m yolo26l}"
SEEDS="${SEEDS:-0 1 2}"
PY=".venv/bin/python"

$PY -m revision.protocol --check
for model in $MODELS; do
  for seed in $SEEDS; do
    echo "=== $(date '+%F %T') DTL $model seed $seed ==="
    $PY -m revision.train_kfold --model "$model" --seed "$seed" --tag dtl --strict-env
  done
done
$PY -m revision.collect_runs --tags dtl
echo "=== $(date '+%F %T') DTL finished ==="
