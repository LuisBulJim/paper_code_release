#!/usr/bin/env bash
# Lanzador por fases del protocolo limpio (nested CV repetida).
# Se puede llamar desde cualquier sitio: todo es relativo a la carpeta PADRE de este directorio
# (Articulo-Corrosion). Solo escribe dentro de revision_nested_cv/ (salidas en revision_nested_cv/out y /smoke).
#
#   bash revision_nested_cv/run_revision.sh check        # entorno, datos, pesos
#   bash revision_nested_cv/run_revision.sh dryrun       # crea y verifica splits, no entrena
#   bash revision_nested_cv/run_revision.sh smoke        # 1 fold, Nano, 2 epocas (minutos)
#   bash revision_nested_cv/run_revision.sh direct       # 45 entrenamientos (transfer directo)
#   bash revision_nested_cv/run_revision.sh sequential   # 45 entrenamientos (transfer secuencial)
#   bash revision_nested_cv/run_revision.sh verify       # SIN entrenar: confirma que los .pkl corresponden
#                                                         # a los splits actuales (detecta checkpoints
#                                                         # desactualizados, p.ej. tras cambiar --groups-csv)
#   bash revision_nested_cv/run_revision.sh extra        # vistas TTA extra (reusa checkpoints, sin reentrenar)
#   bash revision_nested_cv/run_revision.sh eval         # metricas, IC bootstrap (CPU)
#   bash revision_nested_cv/run_revision.sh latency      # latencia y memoria GPU (aislado por proceso)
#   bash revision_nested_cv/run_revision.sh full         # direct + sequential + extra + eval + latency
#
# Variables opcionales: DATA=dataset_yolo REPEATS=3 PY=python GROUPS_CSV=ruta.csv DEVICE=0 JOBS=4 NBOOT=1000
#   FINE_SCALES="0.8 0.9 1.0 1.1 1.2"  ROT_ANGLES="5 8"  MAX_VIEWS=10  SAHI=1 (activa --sahi, experimental)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
cd "$ROOT"
D="revision_nested_cv"; OUT="$D/out"; SMOKE="$D/smoke"
DATA="${DATA:-dataset_yolo}"; REPEATS="${REPEATS:-3}"; PY="${PY:-python}"
JOBS="${JOBS:-4}"; NBOOT="${NBOOT:-1000}"
FINE_SCALES="${FINE_SCALES:-0.8 0.9 1.0 1.1 1.2}"; ROT_ANGLES="${ROT_ANGLES:-5 8}"; MAX_VIEWS="${MAX_VIEWS:-10}"
EXTRA=()
[ -n "${GROUPS_CSV:-}" ] && EXTRA+=(--groups-csv "$GROUPS_CSV")
[ -n "${DEVICE:-}" ] && EXTRA+=(--device "$DEVICE")
SEQ_W="Sequential_Transfer_Benchmark/phase1_aerial_yolo26%s/weights/best.pt"

need_data() { [ -d "$DATA/images" ] || { echo "ERROR: no existe $DATA/images (cd a Articulo-Corrosion o define DATA=)"; exit 1; }; }

check() {
  echo "cwd: $(pwd)"; need_data
  echo "imagenes: train=$(ls $DATA/images/train 2>/dev/null | wc -l) test=$(ls $DATA/images/test 2>/dev/null | wc -l) val=$(ls $DATA/images/val 2>/dev/null | wc -l)"
  $PY - <<'PYEOF'
import sys
print("python", sys.version.split()[0])
for m in ["ultralytics","torch","sklearn","ensemble_boxes","cv2","pandas","numpy"]:
    try:
        mod=__import__(m); print(f"  {m:15s}", getattr(mod,"__version__","ok"))
    except Exception as e:
        print(f"  {m:15s} FALTA ({e.__class__.__name__})")
try:
    import torch; print("  CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
except Exception:
    print("  CUDA: no se puede comprobar (falta torch)")
try:
    import ultralytics
    if ultralytics.__version__ != "8.4.5":
        print("  AVISO: el repo publicado fija ultralytics==8.4.5; con otra version YOLO26 puede dar otras detecciones")
except Exception: pass
try:
    import sahi; print("  sahi", getattr(sahi,"__version__","ok"), "(solo hace falta para --sahi en 'extra')")
except Exception:
    print("  sahi: no instalado (solo hace falta si vais a probar 'extra --sahi'; se puede omitir)")
PYEOF
  for s in n m l; do
    w=$(printf "$SEQ_W" $s); [ -f "$w" ] && echo "  secuencial phase1 $s: OK" || echo "  secuencial phase1 $s: NO ENCONTRADO ($w)"
    [ -f "yolo26$s.pt" ] && echo "  coco yolo26$s.pt: OK" || echo "  coco yolo26$s.pt: no esta en cwd (Ultralytics intentara descargarlo)"
  done
  echo "espacio libre:"; df -h . | tail -1
}
dryrun() { need_data; $PY $D/nested_cv.py --data-root "$DATA" --out "$OUT" --repeats "$REPEATS" --dry-run "${EXTRA[@]}"; }
smoke() {
  need_data
  $PY $D/nested_cv.py --data-root "$DATA" --out "$SMOKE" --repeats 1 --sizes n --epochs 2 --only-folds 0 "${EXTRA[@]}"
  $PY $D/evaluate_cv.py --out "$SMOKE" --n-boot 20 --jobs 2
  echo "SMOKE OK: si ves tablas arriba, el pipeline funciona de extremo a extremo."
}
direct()     { need_data; $PY $D/nested_cv.py --data-root "$DATA" --out "$OUT" --repeats "$REPEATS" --protocol direct "${EXTRA[@]}"; }
sequential() { need_data; $PY $D/nested_cv.py --data-root "$DATA" --out "$OUT" --repeats "$REPEATS" --protocol sequential "${EXTRA[@]}"; }
verify() {
  need_data
  echo "--- direct ---"
  $PY $D/nested_cv.py --data-root "$DATA" --out "$OUT" --repeats "$REPEATS" --protocol direct --verify-splits "${EXTRA[@]}"
  echo "--- sequential ---"
  $PY $D/nested_cv.py --data-root "$DATA" --out "$OUT" --repeats "$REPEATS" --protocol sequential --verify-splits "${EXTRA[@]}"
}
extra() {
  need_data
  SAHI_FLAG=(); [ -n "${SAHI:-}" ] && SAHI_FLAG=(--sahi)
  for tag in direct sequential; do
    ls "$OUT"/preds/${tag}_r* >/dev/null 2>&1 || { echo "skip $tag: no hay predicciones (lanza esa fase primero)"; continue; }
    $PY $D/predict_extra_views.py --out "$OUT" --tag "$tag" --data-root "$DATA" \
        --scales $FINE_SCALES --rot-angles $ROT_ANGLES "${SAHI_FLAG[@]}" "${EXTRA[@]}"
  done
}
evalr() {
  SEQ=(); ls "$OUT"/preds/sequential_r* >/dev/null 2>&1 && SEQ=(--seq-tag sequential)
  $PY $D/evaluate_cv.py --out "$OUT" "${SEQ[@]}" --n-boot "$NBOOT" --jobs "$JOBS" \
      --fine-scales $FINE_SCALES --rot-angles $ROT_ANGLES --max-views "$MAX_VIEWS"
}
latency() {
  R="$OUT/runs/direct_r0_f0"
  $PY $D/bench_latency.py --n $R"_n/weights/best.pt" --m $R"_m/weights/best.pt" --l $R"_l/weights/best.pt" \
      --images "$DATA/images/test" | tee "$OUT/latency.txt"
}
case "${1:-help}" in
  check) check;; dryrun) dryrun;; smoke) smoke;; direct) direct;; sequential) sequential;;
  verify) verify;;
  extra) extra;; eval) evalr;; latency) latency;;
  full) direct; sequential; verify; extra; evalr; latency;;
  *) sed -n '2,19p' "$0";;
esac
