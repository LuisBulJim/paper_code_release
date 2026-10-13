#!/usr/bin/env bash
# Inference stage after DTL training, one resumable command:
#   validation sweep -> results/inference_choice.json (fixed rule, validation only) -> chosen config on test
#   -> latency of the chosen TTA/ensemble -> failure analysis -> checkpoint audit
#   nohup bash scripts/run_inference.sh > runs/inference_nohup.log 2>&1 &
# Optional: TAG=dtl MODELS=yolo26n,yolo26m,yolo26l LATENCY_MODEL=yolo26m bash scripts/run_inference.sh
# Finished dumps are re-used only when their settings match, so relaunching after a crash is safe.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
TAG="${TAG:-dtl}"
MODELS="${MODELS:-yolo26n,yolo26m,yolo26l}"
LATENCY_MODEL="${LATENCY_MODEL:-yolo26m}"
SUFFIX=""; [ "$TAG" != "dtl" ] && SUFFIX="_$TAG"
CHOICE="results/inference_choice$SUFFIX.json"

echo "=== $(date '+%F %T') validation sweep (TTA + WBF) ==="
$PY -m revision.sweep_inference --tag "$TAG" --models "$MODELS" --what both
cat "$CHOICE"

echo "=== $(date '+%F %T') test inference with the validation choice ==="
$PY -m revision.run_test_inference --tag "$TAG" --models "$MODELS" --choice "$CHOICE"

read -r TTA_DIR WBF_CKPTS < <($PY - "$CHOICE" "$TAG" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))
tta = f"preds_test_{c['tta']['config']}_iou{c['tta']['wbf_iou']}"
members = (c.get("wbf") or {}).get("models") or []
print(tta, " ".join(f"runs/{sys.argv[2]}/{m}_seed0/fold0/weights/best.pt" for m in members))
EOF
)

echo "=== $(date '+%F %T') latency benchmark ==="
$PY -m revision.benchmark_latency --standard "runs/$TAG/${LATENCY_MODEL}_seed0/fold0/weights/best.pt" --wbf $WBF_CKPTS --choice "$CHOICE" --iters 6 --out "results/latency$SUFFIX.csv"

echo "=== $(date '+%F %T') failure analysis ($LATENCY_MODEL seed0 fold0) ==="
$PY -m revision.failure_analysis --standard "runs/$TAG/${LATENCY_MODEL}_seed0/fold0/preds_test" --tta "runs/$TAG/${LATENCY_MODEL}_seed0/fold0/$TTA_DIR"

echo "=== $(date '+%F %T') checkpoint audit ==="
$PY -m revision.audit_checkpoints "runs/$TAG" --json "results/audit_checkpoints_$TAG.json"
echo "=== $(date '+%F %T') inference stage finished ==="
