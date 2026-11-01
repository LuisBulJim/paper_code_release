#!/usr/bin/env bash
# R2.5 -- SAM 3 re-evaluated with the detector map of r2_5_neighbour_distance.py (fold model whose train+val frames are
# farthest in frame index). Same r1/sam3_eval.py, same thresholds as the paper (0.25 fixed, 0.1 chosen on validation).
# Environment: yolo13env (SAM 3 in transformers 5.8). Its ultralytics 8.4.14 gives detector boxes identical to the
# 8.4.5 used in the paper (25 images, conf 0.001: same count, max abs difference 0.0).
set -euo pipefail
cd "$(dirname "$0")/../revision_nested_cv"
PY="${PYTHON_SAM3:-python}"
export PYTHONPATH=$PWD NESTED_DATASET=$PWD/../dataset_yolo
MAP=../results_rev2/r2_5/sam3_map_farthest_r0_m.json
OUT=../results_rev2/r2_5
$PY -m r1.sam3_eval predict --checkpoint-map $MAP --out $OUT/sam3_farthest_std
$PY -m r1.sam3_eval metrics --preds $OUT/sam3_farthest_std --out $OUT/metrics_std --figure $OUT/sam3_cases_std.png
$PY -m r1.sam3_eval predict --checkpoint-map $MAP --conf 0.1 --out $OUT/sam3_farthest_valthr
$PY -m r1.sam3_eval metrics --preds $OUT/sam3_farthest_valthr --out $OUT/metrics_valthr --figure $OUT/sam3_cases_valthr.png
echo SAM3_DONE
