#!/usr/bin/env bash
# R2.1(a) purge 5/15/30 (resumable) followed by the R2.5 SAM 3 re-evaluation. Detached so it survives the session.
cd "$(dirname "$0")/../.."
PY="${PYTHON:-python}"
L=results_rev2/r2_1/logs
for B in 5 15 30; do
  $PY results_rev2/r2_1ab_train.py --setting purge$B > $L/train_purge$B.log 2>&1 || echo "purge$B FAILED" >> $L/chain.log
  echo "purge$B done $(date -u)" >> $L/chain.log
done
bash results_rev2/r2_5_sam3_farthest.sh > results_rev2/r2_5/sam3_farthest.log 2>&1 && echo "sam3 done $(date -u)" >> $L/chain.log
