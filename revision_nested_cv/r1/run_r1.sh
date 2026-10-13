#!/usr/bin/env bash
# Lanzador por fases de r1/: lo que le faltaba a la nested CV (evaluador único, TrashCan, SAM 3, latencia, tablas).
# Como run_revision.sh, todo es relativo a la carpeta PADRE de revision_nested_cv (Articulo-Corrosion) y solo
# escribe dentro de revision_nested_cv/r1/. No reentrena LIACI ni toca out/ (solo lo lee).
#
#   bash revision_nested_cv/r1/run_r1.sh check      # entorno (ultralytics 8.4.5), dataset_yolo, splits, .pkl y checkpoints
#   bash revision_nested_cv/r1/run_r1.sh evaluate   # CPU: re-evaluación de los .pkl (ya hecha en local, resultados en r1/results)
#   bash revision_nested_cv/r1/run_r1.sh link       # enlaza out/runs/*/best.pt en r1/runs y escribe el mapa de SAM 3
#   bash revision_nested_cv/r1/run_r1.sh trashcan   # GPU ~3.5 h: reentreno TrashCan (3 semillas) + OOD con los modelos LIACI
#   bash revision_nested_cv/r1/run_r1.sh sam3       # GPU ~1-2 h: SAM 3 con detector out-of-fold (necesita transformers con Sam3Model)
#   bash revision_nested_cv/r1/run_r1.sh latency    # GPU ~10 min: latencia de estándar / TTA elegida / WBF elegido
#   bash revision_nested_cv/r1/run_r1.sh report     # CPU: análisis de fallos, tablas LaTeX y figuras
#   bash revision_nested_cv/r1/run_r1.sh gpu        # link + trashcan + sam3 + latency + report
#
# Variables opcionales: PY=python DATA=dataset_yolo DEVICE=cuda:0 OOD_MODELS=yolo26m SAM3_TTA=0 (1 = también con TTA)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D="$(dirname "$HERE")"
ROOT="$(dirname "$D")"
cd "$ROOT"
PY="${PY:-python}"
DATA="${DATA:-dataset_yolo}"
OOD_MODELS="${OOD_MODELS:-yolo26m}"
export PYTHONPATH="$D${PYTHONPATH:+:$PYTHONPATH}"
export NESTED_DATASET="$ROOT/$DATA"
DEV=()
[ -n "${DEVICE:-}" ] && DEV=(--device "$DEVICE")
LOG="$HERE/results/logs"
mkdir -p "$LOG"

check() {
  echo "cwd: $(pwd) | dataset: $NESTED_DATASET"
  $PY - <<'PYEOF'
import sys
print("python", sys.version.split()[0])
missing = []
for m in ["ultralytics", "torch", "ensemble_boxes", "pycocotools", "cv2", "scipy", "matplotlib", "yaml", "PIL"]:
    try:
        mod = __import__(m)
        print(f"  {m:15s}", getattr(mod, "__version__", "ok"))
    except Exception as e:
        print(f"  {m:15s} FALTA ({e.__class__.__name__})")
        missing.append(m)
import torch, ultralytics
print("  CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
try:
    from transformers import Sam3Model  # noqa: F401
    print("  transformers con Sam3Model: OK")
except Exception as e:
    print(f"  transformers con Sam3Model: NO ({e.__class__.__name__}) -> solo hace falta para la fase sam3")
if ultralytics.__version__ != "8.4.5":
    sys.exit(f"ERROR: ultralytics {ultralytics.__version__}; los 90 modelos se entrenaron con 8.4.5 y la decodificación de YOLO26 cambia entre versiones")
if missing:
    sys.exit(f"ERROR: faltan {missing} (pip install pycocotools ensemble-boxes==1.0.9 ...)")
PYEOF
  $PY -m r1.protocol --check > "$LOG/protocol_check.json" && echo "  dataset_yolo: idéntico al manifest (sha256 de las 261 imágenes y etiquetas)"
  $PY -m r1.nested_eval check
}

evaluate() {
  $PY -m r1.nested_eval prepare
  $PY -m r1.nested_eval evaluate
}

link() {
  $PY -m r1.nested_eval link
  $PY -m r1.nested_eval sam3-map
  $PY -m r1.audit_checkpoints "$HERE/runs/direct" "$HERE/runs/sequential" > "$LOG/audit_checkpoints.txt" && echo "  checkpoints enlazados: todos YOLO26 ($LOG/audit_checkpoints.txt)"
}

need_choice() {
  [ -f "$HERE/results/inference_choice.json" ] || { echo "ERROR: falta r1/results/inference_choice.json (fase evaluate)"; exit 1; }
}

trashcan() {
  need_choice
  [ -d "$HERE/runs/direct" ] || link
  echo "=== $(date '+%F %T') TrashCan: reentreno con la receta LIACI ==="
  $PY -m r1.trashcan_train --strict-env "${DEV[@]}"
  echo "=== $(date '+%F %T') TrashCan: modelos LIACI sin reentrenar sobre metal (OOD) ==="
  $PY -m r1.trashcan_ood --tag direct --models "$OOD_MODELS" "${DEV[@]}"
  echo "=== $(date '+%F %T') TrashCan terminado ==="
}

sam3() {
  need_choice
  [ -f "$HERE/results/sam3_map_r0_m.json" ] || link
  $PY -c "from transformers import Sam3Model, Sam3Processor; print('SAM 3 disponible')"
  echo "=== $(date '+%F %T') SAM 3 (detector estándar out-of-fold, conf 0.25 fijada a priori) ==="
  $PY -m r1.sam3_eval predict --checkpoint-map r1/results/sam3_map_r0_m.json --out r1/runs/sam3/oof_standard "${DEV[@]}"
  $PY -m r1.sam3_eval metrics --preds r1/runs/sam3/oof_standard --out r1/results --figure r1/figures/sam3_cases.png
  VT=$($PY -c "import json; print(json.load(open('$HERE/results/inference_choice.json'))['operating_threshold_on_validation']['direct/yolo26m/standard'])")
  if [ "$VT" != "0.25" ]; then
    echo "=== $(date '+%F %T') SAM 3 (mismo detector, umbral elegido en validación: $VT) ==="
    $PY -m r1.sam3_eval predict --checkpoint-map r1/results/sam3_map_r0_m.json --conf "$VT" --out r1/runs/sam3/oof_standard_valthr "${DEV[@]}"
    $PY -m r1.sam3_eval metrics --preds r1/runs/sam3/oof_standard_valthr --out r1/results/sam3_valthr --figure r1/figures/sam3_cases_valthr.png
  fi
  if [ "${SAM3_TTA:-0}" = "1" ]; then
    echo "=== $(date '+%F %T') SAM 3 (detector con la TTA elegida) ==="
    $PY -m r1.sam3_eval predict --checkpoint-map r1/results/sam3_map_r0_m.json --tta --out r1/runs/sam3/oof_tta "${DEV[@]}"
    $PY -m r1.sam3_eval metrics --preds r1/runs/sam3/oof_tta --out r1/results/sam3_tta --figure r1/figures/sam3_cases_tta.png
  fi
  echo "=== $(date '+%F %T') SAM 3 terminado ==="
}

latency() {
  need_choice
  [ -d "$HERE/runs/direct" ] || link
  WBF=$($PY -c "import json; c=json.load(open('$HERE/results/inference_choice.json')); print(' '.join(f'r1/runs/direct/{m}_seed0/fold0/weights/best.pt' for m in (c.get('wbf') or {}).get('models', [])))")
  $PY -m r1.benchmark_latency --standard r1/runs/direct/yolo26m_seed0/fold0/weights/best.pt --wbf $WBF "${DEV[@]}"
}

report() {
  OOF="$HERE/runs/nested/grouped/oof"
  if [ -d "$OOF/direct_yolo26m_standard_r0" ]; then
    $PY -m r1.failure_analysis --standard "$OOF/direct_yolo26m_standard_r0" --tta "$OOF/direct_yolo26m_tta_r0" --out r1/results --figure r1/figures/failure_cases.png
  fi
  $PY -m r1.make_tables
  $PY -m r1.make_figures
}

case "${1:-}" in
  check) check ;;
  evaluate) evaluate ;;
  link) link ;;
  trashcan) trashcan ;;
  sam3) sam3 ;;
  latency) latency ;;
  report) report ;;
  gpu) link; trashcan; sam3; latency; report ;;
  *) sed -n '2,15p' "$0"; exit 1 ;;
esac
