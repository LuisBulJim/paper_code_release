# Paquete para el servidor (RTX 4090) — revisión R1

Guía para Luis. Todo está pensado para copiar el repositorio una vez, lanzar tres scripts y traer de vuelta
`results/`, `tables/` y `figures/`. Nada se elige mirando el test; nada requiere editar código en el servidor.

## 1. Qué copiar

Copiar el directorio `paper_code_release/` completo **excepto** `.venv/`, `runs/dryrun/`, `__pycache__/`.
Imprescindibles (ya generados en local, no volver a generarlos en el servidor):

| Ruta | Contenido | Tamaño aprox. |
|---|---|---|
| `data/dataset_yolo_v2/` | 261 imágenes 640×640 + etiquetas (pool 228 / test 33) | 60 MB |
| `data/manifest_v2.json`, `data/folds_v2.json`, `data/test_images.json` | manifiesto con sha256, folds 5×3, test congelado | < 1 MB |
| `data/sam3_eval/` + `data/sam3_manifest.json` | 191 imágenes + máscaras para SAM 3 | 40 MB |
| `data/trashcan_v2/` + `data/trashcan_manifest.json` | TrashCan Metal/Bio: 6008 train (5407 fit + 601 selección) / 1204 val | 190 MB |
| `revision/`, `scripts/`, `tests/`, `requirements-lock.txt` | código | — |

No copiar `data/trashcan_v2_dryrun/` ni `data/trashcan_manifest_dryrun.json` (subconjunto de pruebas locales).
Las listas y el `data.yaml` de TrashCan se reescriben solos con las rutas del servidor al lanzar el
entrenamiento. Si prefieres regenerar TrashCan allí, borra `data/trashcan_manifest.json` y pasa
`TRASHCAN_SOURCE=/ruta TRASHCAN_FORMAT=yolo|coco` a `scripts/run_trashcan.sh`.

Dataset aéreo (solo STL): apuntar `AERIAL_DATA` al `data.yaml` del dataset Roboflow ya descargado en julio.

## 2. Instalación (una vez, ~10 min)

```bash
cd paper_code_release
bash scripts/setup_server.sh            # venv, torch 2.8.0 cu128, ultralytics 8.4.164, pesos yolo26{n,m,l}.pt
.venv/bin/python -m pytest -q tests     # 33 tests, < 10 s
.venv/bin/python -m revision.protocol --check   # versiones + dataset completo (sha256 de las 261 imágenes)
```

Si el servidor ya tiene otra versión de torch/CUDA y `setup_server.sh` falla en el paso de torch, instalar la
que corresponda (`TORCH_INDEX=cu121 bash scripts/setup_server.sh`) y anotar la versión: `protocol.yaml`
fija `torch: "2.8.0"`; con `--strict-env` los scripts se niegan a entrenar si no coincide, así que o se
ajusta `environment.torch` en `protocol.yaml` a la versión instalada o se quita `--strict-env` de los
scripts. **Ultralytics debe ser 8.4.164** (es la versión con la que se validó todo).

Prueba de humo en GPU (2-3 min): 

```bash
.venv/bin/python -m revision.train_kfold --model yolo26n --seed 0 --dry-run --device 0
.venv/bin/python -m revision.collect_runs --include-dry-run
```

## 3. Ejecución (en este orden)

| Paso | Comando | Tiempo estimado | Produce |
|---|---|---|---|
| DTL | `nohup bash scripts/run_dtl.sh > runs/dtl_nohup.log 2>&1 &` | n ≈ 1.5 h, m ≈ 3 h, l ≈ 4 h (45 folds) | `runs/dtl/…`, `results/{cv,test}_metrics.csv`, `results/summary.csv` |
| STL (Medium) | `AERIAL_DATA=/ruta/data.yaml nohup bash scripts/run_stl.sh > runs/stl_nohup.log 2>&1 &` | pre-entreno ≈ 10-12 h + 3 h | `runs/aerial/`, `runs/stl/…` |
| Comparación | `.venv/bin/python -m revision.compare_protocols` | s | `results/protocol_comparison.csv` |
| Inferencia | `nohup bash scripts/run_inference.sh > runs/inference_nohup.log 2>&1 &` (ver 3.1) | 2-3 h | `results/sweep_*`, `results/inference_test*`, `results/latency.csv`, fallos |
| TrashCan (después de Inferencia) | `nohup bash scripts/run_trashcan.sh > runs/trashcan_nohup.log 2>&1 &` | 3 × 1 h + 30 min | `results/trashcan*.csv`, `results/trashcan_ood*.csv` |
| SAM 3 | `SAM3_CKPT=runs/dtl/yolo26m_seed0/fold0/weights/best.pt bash scripts/run_sam3.sh` | 20 min | `results/sam3_*.csv`, `figures/sam3_cases.png` |
| Tablas/figuras | `.venv/bin/python -m revision.make_tables && .venv/bin/python -m revision.make_figures` | s | `tables/`, `figures/` |

Todos los scripts son reanudables: si un job muere, se relanza el mismo comando y salta lo terminado
(`DONE` por fold/semilla, `_meta.json` por volcado).

### 3.1 Inferencia: sin decisiones manuales

`scripts/run_inference.sh` encadena todo con una regla fijada de antemano en el código:

1. Barrido en **validación** (7 configuraciones TTA × 3 IoU, y ensembles de 2-3 tamaños × pesos × IoU; ≈ 1.5-2 h).
2. `results/inference_choice.json`: TTA = la configuración con mayor mAP@50 de validación **promediado sobre los
   tres tamaños** (una sola configuración para todos); WBF = el ensemble con mayor mAP@50 medio de validación.
3. Esa elección se aplica al test con los 15 checkpoints por tamaño (`results/inference_test*.csv`,
   `results/inference_test_config.json`), a la latencia, al análisis de fallos, a TrashCan y a SAM 3 con TTA.

Los volcados llevan la configuración en el nombre y solo se reutilizan si su `_meta.json` coincide, así que
relanzar tras un corte es seguro. `run_trashcan.sh` se niega a arrancar si no existe `results/inference_choice.json`.
Para SAM 3 con el detector en modo TTA: `SAM3_TTA=1`.

### 3.2 SAM 3

`sam3_eval.py predict` necesita `transformers` con `Sam3Model` (`pip install -U "transformers>=5.13" huggingface-hub`,
no está en `requirements-lock.txt` a propósito) y acceso a `facebook/sam3` en el Hub (`export HF_TOKEN=…` si está
gated). Es la única parte que no se ha podido ejecutar en local: el resto del pipeline (lectura, métricas,
figura) está validado con `--mock-sam`. Si la API de `Sam3Processor` difiere de la del notebook 09
(`post_process_instance_segmentation` / `pred_masks`), el error saldrá en la primera imagen; enviarme el
traceback.

## 4. Qué traer de vuelta

* `results/` completo (CSV, JSON, `logs/`).
* `tables/`, `figures/`.
* `runs/**/run_info.json`, `runs/**/metrics_val.json`, `runs/**/preds_*/` (volcados, ~1 MB por fold), `runs/*_nohup.log`.
* `runs/**/weights/best.pt` de al menos `yolo26m` (15 × 40 MB) para el depósito en Zenodo; n y l si hay sitio.
* `results/audit_checkpoints_dtl.json` (lo genera `run_inference.sh`; debe decir `test-in-train=0` en todos).

## 5. Comprobaciones antes de dar por buenos los resultados

```bash
.venv/bin/python -m revision.audit_checkpoints runs/dtl runs/stl        # exit 0, todos YOLO26, solape 0
grep -c DONE -r runs/dtl | wc -l                                           # 45
.venv/bin/python -m revision.collect_runs --tags dtl,stl                  # rehace summary.csv
```

`results/summary.csv` debe tener, por (tag, modelo), `n_runs = 15` y `n_seeds = 3`. Rendimiento esperable en
validación (folds reales de julio, sin fuga): mAP@50 ≈ 0.28 (n), 0.37 (m), 0.39 (l). Si el test da valores
mucho más altos que la validación, algo está mal: parar y revisar.
