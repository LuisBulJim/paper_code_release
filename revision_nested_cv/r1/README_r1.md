# r1/ — lo que le faltaba a la nested CV

`r1/` se añade a `revision_nested_cv/` sin tocar ningún fichero existente: lee `out/` (splits, `.pkl`, checkpoints) y
`../dataset_yolo`, y escribe solo dentro de `r1/`. No reentrena LIACI.

## Qué añade

| Punto (revisores / auditoría) | Qué hace r1 | Dónde |
|---|---|---|
| Un solo evaluador, mAP@[.5:.95] | Reconstruye estándar / TTA / WBF a partir de las vistas guardadas en los `.pkl` y lo evalúa con pycocotools (mAP@50, mAP@[.5:.95]) más P/R/F1 a conf 0,25. No se usa el AP50 «pre» | `nested_eval.py evaluate` |
| Intervalos y significación | Bootstrap **por grupo de vídeo**: se remuestrean grupos enteros, con los mismos grupos para cada repetición y estrategia. IC 95 % de la media sobre repeticiones y deltas pareados | idem |
| TTA / WBF sin mirar test | Regla fija de `sweep_inference.py` aplicada a las **validaciones internas de los 15 folds** (media sobre folds y tamaños) → `r1/results/inference_choice.json` | idem |
| Cribado a nivel de imagen | AUROC, sensibilidad y especificidad (puntuación de la imagen = confianza máxima) | idem |
| Efecto del agrupado por vídeo | Mismo análisis sobre la ejecución sin agrupar (`.pkl.bak`, 28-sep) como ablación | `r1/results/nested_ungrouped/` |
| Integridad | Splits sin solape, cada imagen evaluada una vez por repetición, ningún grupo a ambos lados, `.pkl` = split (fingerprint), GT = etiquetas, 90 checkpoints YOLO26 entrenados en su propio workspace, misma configuración en los 90 | `nested_eval.py check` |
| TrashCan (R3) | Reentreno YOLO26-m con la receta LIACI exacta (`configs/hparams.yaml`, copiada de `out/runs/*/args.yaml`), 3 semillas, `best.pt` elegido en un 10 % del train oficial, evaluación en val oficial (Metal/Bio). OOD: los 15 modelos LIACI sin reentrenar sobre metal | `trashcan_train.py`, `trashcan_ood.py` |
| SAM 3 (R3) | Set de 191 imágenes del test oficial de LIACI; las 27 que están en la nested CV las detecta el modelo del fold que las tenía en test (nunca las vio) | `nested_eval.py sam3-map`, `sam3_eval.py` |
| Latencia | Estándar, la TTA elegida y el WBF elegido en la 4090 (fp32/fp16) | `benchmark_latency.py` |
| Tablas y figuras sin números a mano | Tablas LaTeX + `numbers.json`; `check_consistency.py --tex` señala cualquier número del paper que no salga de `r1/results` | `make_tables.py`, `make_figures.py`, `check_consistency.py` |

## Ejecución en el servidor

Entorno: `yoloenv` (Python 3.9, ultralytics 8.4.5, la misma versión con la que se entrenaron los 90 modelos; `check` aborta con otra). Solo le falta `pycocotools` (ensemble-boxes 1.0.9 ya está). El disco del servidor es compartido y va muy justo: no clonar entornos.

```bash
cd <project root>            # la carpeta que contiene revision_nested_cv/ y dataset_yolo/
conda activate yoloenv        # ultralytics 8.4.5, torch 2.8.0 (requirements-lock.txt)
pip install pycocotools

bash revision_nested_cv/r1/run_r1.sh check                 # crea results/logs; ningún ERROR
nohup bash -c "bash revision_nested_cv/r1/run_r1.sh link && bash revision_nested_cv/r1/run_r1.sh trashcan && bash revision_nested_cv/r1/run_r1.sh latency && bash revision_nested_cv/r1/run_r1.sh report" > revision_nested_cv/r1/results/logs/gpu_a.log 2>&1 &
tail -f revision_nested_cv/r1/results/logs/gpu_a.log
```

`link` → `trashcan` (unas 3,5 h) → `latency` (10 min) → `report`. Todo se puede reanudar: las semillas terminadas y los dumps con los mismos ajustes se reutilizan.

SAM 3 necesita `transformers>=5.13`, que exige Python ≥ 3.10, así que no cabe en `yoloenv`. Se ejecuta aparte (`run_r1.sh sam3`, 1-2 h) en un entorno con Python ≥ 3.10, ultralytics 8.4.5 y transformers, cuando haya unos 15 GB libres; después, `run_r1.sh report` otra vez en `yoloenv`.

Qué traer de vuelta: `revision_nested_cv/r1/results/`, `r1/tables/`, `r1/figures/`, `r1/runs/trashcan/*/run_info.json`, `r1/runs/sam3/*/_meta.json` y los logs.

## Decisiones fijadas antes de ver resultados

- Operating point: conf 0,25 e IoU 0,5. TTA y WBF elegidos solo en validación interna, con la regla de `sweep_inference.py`.
- Candidatos de TTA: las configuraciones de `TTA_GRID` cuyas vistas guardó `nested_cv.py` (escalas 0,9 / 1,0 / 1,1 y flip). Las de 0,8 y 1,2 no están en los `.pkl`.
- Los ensembles WBF se forman con los modelos de transfer directo del mismo repetición y fold.
- Grupos de vídeo: `nested.groups_csv` en `configs/protocol.yaml`. Se deja vacío: entonces se reconstruyen desde `splits.json` bloques de IDs consecutivos con los mismos folds de test en las 3 repeticiones. El `groups.csv` original de Juan (archivado en `results/groups/groups.csv` y `r1/configs/groups.csv`) da exactamente la misma partición; solo cambian las etiquetas, y los IC del paper se calcularon con las reconstruidas.

## Local (CPU)

```bash
python -m pytest -q revision_nested_cv/r1/tests
NESTED_DATASET=/ruta/dataset_yolo PYTHONPATH=revision_nested_cv python -m r1.nested_eval check
```
