#!/usr/bin/env bash
# ext/: pre-registered variants of YOLO26-m on the grouped nested-CV splits, plus the SAM 3 bounds.
# Run from revision_nested_cv/. Training and evaluation use the yoloenv env (ultralytics 8.4.5);
# the sam3 step uses the env with transformers >= 5 (envs/sam2).
#   bash ext/run_ext.sh check      verify ext/data against its manifest and dry-run every variant (no GPU)
#   bash ext/run_ext.sh train      x1_recipe, x3_seg, x4_seg_cp, x2_multi (in that order; resumes if interrupted)
#   bash ext/run_ext.sh one x3_seg one variant only
#   bash ext/run_ext.sh evaluate   scores every finished variant against the baseline (CPU)
#   bash ext/run_ext.sh combine    trains x5_combined from decision.json if two or more variants beat the baseline
#   bash ext/run_ext.sh pixel      masks of x3/x4 (GPU, minutes) and pixel IoU of every model against the polygons
#   bash ext/run_ext.sh sam3       SAM 3 with ground-truth boxes and with text prompts (sam2 env)
#   bash ext/run_ext.sh ensemble   e1_ensemble (WBF of the four variants) and the c4_baselines control (CPU, minutes)
#   bash ext/run_ext.sh latency    GPU latency of one model, TTA, e1_ensemble and e1_ensemble + TTA (R2.6)
#   bash ext/run_ext.sh pack       ext_resultados.zip with results, logs, predictions, masks and training curves
#   bash ext/run_ext.sh closing    unattended: ensemble, evaluate, sam3 (if SAM_PY is set), x6_multi_seg, evaluate, pixel, pack
#   bash ext/run_ext.sh final      ensemble (with control), evaluate, latency, pixel, pack (~1 h, GPU only for latency)
set -euo pipefail
POOL=${POOL:-../dataset_yolo}
PY=${PY:-python}
LOGS=ext/out/logs
mkdir -p "$LOGS"
step=${1:-help}

case "$step" in
  check)
    $PY - <<'EOF'
import hashlib, json, pathlib
root = pathlib.Path("ext/data")
for name in ("manifest_ext.json", "manifest_seg_multi.json"):
    m = json.load(open(root / name))
    bad = [p for p, h in m["sha256"].items() if not (root / p).exists() or hashlib.sha256((root / p).read_bytes()).hexdigest() != h]
    print(f"ext/data ({name}): {len(m['sha256'])} files, {len(bad)} missing or changed")
    assert not bad, bad[:5]
EOF
    for v in x1_recipe x2_multi x3_seg x4_seg_cp x6_multi_seg; do
      $PY ext/train_ext.py --variant "$v" --pool "$POOL" --dry-run --only 0,0 2>&1 | tee -a "$LOGS/check.log"
    done
    $PY ext/train_ext.py --combine x1_recipe x2_multi --pool "$POOL" --dry-run --only 0,0 2>&1 | tee -a "$LOGS/check.log"
    $PY -c "import ultralytics; assert ultralytics.__version__ == '8.4.5', ultralytics.__version__; print('ultralytics', ultralytics.__version__)"
    # 60 best.pt of ~45-50 MB plus one temporary workspace (~0.2 GB for x2_multi)
    free_gb=$(df -Pk . | awk 'NR==2 {print int($4 / 1048576)}')
    echo "free disk: ${free_gb} GB (training needs ~3 GB)"
    [ "$free_gb" -ge 5 ] || { echo "not enough free disk for the 60 runs"; exit 1; }
    ;;
  train)
    # A failed variant does not stop the others; rerunning skips the folds already saved.
    failed=""
    for v in x1_recipe x3_seg x4_seg_cp x2_multi; do
      $PY ext/train_ext.py --variant "$v" --pool "$POOL" 2>&1 | tee -a "$LOGS/train_$v.log" || failed="$failed $v"
    done
    if [ -n "$failed" ]; then echo "FAILED:$failed (see $LOGS)"; exit 1; fi
    echo "all variants finished"
    ;;
  one)
    $PY ext/train_ext.py --variant "$2" --pool "$POOL" 2>&1 | tee -a "$LOGS/train_$2.log"
    ;;
  evaluate)
    NESTED_DATASET="$POOL" $PY ext/evaluate_ext.py 2>&1 | grep -v "Swap them" | tee "$LOGS/evaluate.log"
    ;;
  combine)
    # x5_combined = union of the variants chosen on validation by 'evaluate' (ext/results/decision.json)
    names=$($PY -c "import json; print(' '.join(json.load(open('ext/results/decision.json'))['combination']))")
    if [ "$(echo $names | wc -w)" -lt 2 ]; then
      echo "combination: '${names:-none}' (fewer than two variants beat the baseline on validation): nothing to train"
    else
      $PY ext/train_ext.py --combine $names --pool "$POOL" 2>&1 | tee -a "$LOGS/train_x5_combined.log"
      echo "now rerun: bash ext/run_ext.sh evaluate"
    fi
    ;;
  pixel)
    $PY ext/pixel_eval.py predict --pool "$POOL" 2>&1 | tee "$LOGS/pixel_predict.log"
    $PY ext/pixel_eval.py metrics --pool "$POOL" 2>&1 | grep -v "Swap them" | tee "$LOGS/pixel_metrics.log"
    ;;
  sam3)
    $PY ext/sam3_oracle.py --mode gt_boxes --out ext/out/sam3_gt_boxes 2>&1 | tee "$LOGS/sam3_gt_boxes.log"
    $PY -m r1.sam3_eval metrics --preds ext/out/sam3_gt_boxes --out ext/results/sam3_gt_boxes --figure ext/results/sam3_gt_boxes/cases.png
    $PY ext/sam3_oracle.py --mode text --out ext/out/sam3_text 2>&1 | tee "$LOGS/sam3_text.log"
    $PY -m r1.sam3_eval metrics --preds ext/out/sam3_text --out ext/results/sam3_text --figure ext/results/sam3_text/cases.png
    ;;
  ensemble)
    $PY ext/ensemble_eval.py 2>&1 | tee "$LOGS/ensemble.log"
    # control: the same fusion over four baseline models of the grouped run (direct and sequential, m and l)
    $PY ext/ensemble_eval.py --name c4_baselines --member "dm=out/preds/direct_r{r}_f{k}_m.pkl" "dl=out/preds/direct_r{r}_f{k}_l.pkl" "sm=out/preds/sequential_r{r}_f{k}_m.pkl" "sl=out/preds/sequential_r{r}_f{k}_l.pkl" 2>&1 | tee -a "$LOGS/ensemble.log"
    ;;
  latency)
    NESTED_DATASET="$POOL" $PY ext/latency_ext.py 2>&1 | tee "$LOGS/latency.log"
    ;;
  pack)
    $PY - <<'EOF'
import pathlib, zipfile
root = pathlib.Path("ext")
pats = ["results/**/*", "out/logs/*", "out/preds/*.pkl", "out/masks/*.pkl", "out/runs/*/results.csv", "out/runs/*/args.yaml"]
n = 0
with zipfile.ZipFile("ext_resultados.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for pat in pats:
        for p in sorted(root.glob(pat)):
            if p.is_file():
                z.write(p, p.as_posix())
                n += 1
print(f"ext_resultados.zip: {n} files")
EOF
    ;;
  closing)
    # quick results first (CPU, < 1 h), then SAM 3 and x6_multi_seg on the GPU (~5 h); a failed step does not stop the rest
    failed=""
    for s in ensemble evaluate pack; do bash "$0" "$s" || failed="$failed $s"; done
    if [ -n "${SAM_PY:-}" ]; then PY="$SAM_PY" bash "$0" sam3 || failed="$failed sam3"; else echo "SAM_PY not set: sam3 skipped"; fi
    bash "$0" one x6_multi_seg || failed="$failed x6_multi_seg"
    for s in evaluate pixel pack; do bash "$0" "$s" || failed="$failed $s"; done
    echo "CLOSING_DONE${failed:+ (failed:$failed)}"
    ;;
  final)
    failed=""
    for s in ensemble evaluate latency pixel pack; do bash "$0" "$s" || failed="$failed $s"; done
    echo "FINAL_DONE${failed:+ (failed:$failed)}"
    ;;
  *)
    sed -n 2,16p "$0"
    ;;
esac
