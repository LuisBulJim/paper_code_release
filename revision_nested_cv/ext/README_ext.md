# ext/: improvements tested after the leakage correction

Self-contained package next to `r1/`. It never modifies files of `revision_nested_cv` or `r1/`.

## Why these variants

Each variant answers a failure mode measured on the existing grouped out-of-fold predictions (YOLO26-m, direct transfer). The measurements come from `ext/analysis/`.

| Failure mode (measured) | Variant |
|---|---|
| 66 % of background false positives fall on other classes LIACI annotates (paint peel 22 %, sea-chest grating 18 %, marine growth 15 %, others 11 %); 1,591 LIACI images are unused | `x2_multi`: 9 LIACI classes + the unused images, kept only if > 15 filename indices from every test and validation image |
| Corrosion is amorphous (polygon fills 57 % of its box); AP@0.3 0.234 vs AP@0.5 0.167 | `x3_seg`: YOLO26-m-seg on the corrosion polygons; `x4_seg_cp`: + copy-paste 0.5 (needs polygons) |
| Training recipe copied from a large-data default (SGD 0.01, mosaic 1.0) | `x1_recipe`: the official Ultralytics recipe for datasets under 1,000 images (AdamW, lr0 0.001, mosaic 0.5, patience 20) |

Not tested, with the reason:
- More epochs: validation loss is 12-14 % above its minimum at epoch 50.
- Higher resolution: recall is flat across box sizes (0.13 / 0.15 / 0.16).
- Underwater image enhancement: it lowers detection accuracy in large benchmarks (arXiv:2311.18814, arXiv:2411.14626).

## Rules fixed before training

- **Images and seeds:** the outer test images, inner validation images and seeds are those of `out/splits.json` (grouped run); only the named factor changes.
- **Inference choices:**
  - TTA configuration: the one r1 chose on validation for the baseline.
  - Operating threshold: chosen on each variant's own validation.
- **Primary metric:** out-of-fold mAP@50 of the standard pass. Intervals and paired differences come from a cluster bootstrap over the video groups, as in r1.
- **Combination:** `x5_combined` merges only the variants whose mean inner-validation mAP@50 beats the baseline (`ext/results/decision.json`).
  - 9-class boxes and corrosion polygons cannot be merged. If both types qualify, the type of the best validation variant is kept.
  - With fewer than two qualifying variants, nothing is trained.
  - Like r1's TTA choice, this decision pools inner validation over the 15 folds, so the combined model carries a small optimistic selection bias. It is always reported next to the individual variants.
- **Reporting:** every variant is reported, including the ones that do not help.

## Second round (added after the first results, reported as such)

The first round found two levers with gains whose paired intervals exclude zero: x2_multi and x3_seg. The second round adds two models.
- **`x6_multi_seg`:** both levers together, i.e. YOLO26-m-seg on the 9 classes as polygons plus the unused images.
  - Labels come from `ext/build_seg_multi.py`.
  - Each image keeps exactly the annotations of `labels_multi`, as polygons.
- **`e1_ensemble`:** WBF of the four pre-registered variants, all of them, so nothing is selected on test results.
  - It uses the model-ensemble fusion r1 chose on validation (equal weights, IoU 0.5).
  - `ext/ensemble_eval.py` reproduces r1's n+m+l ensemble exactly (out-of-fold mAP@50 0.1879, validation 0.2737).

Both are scored like the variants, with paired intervals against the baseline. They were decided after seeing the first round, so the paper must present them as confirmatory, not pre-registered.

**Control for the ensemble:** `c4_baselines` applies the same fusion to four baseline models of the grouped run (direct and sequential, m and l). It separates the gain of ensembling from the gain of the variants.

**Latency (R2.6):** `ext/latency_ext.py` times every recipe on fold r0/f0 like `r1/benchmark_latency.py`. The recipes are one model, TTA, e1, e1 + TTA, c4 and c4 + TTA.

```bash
bash ext/run_ext.sh final      # ensemble + control, evaluate, latency, pixel, pack
```

```bash
SAM_PY=<python of the env with transformers>=5.13> nohup bash ext/run_ext.sh closing > ext/out/logs/closing.log 2>&1 &
```

## Final analysis (CPU, after `final`)

Run on the downloaded server outputs (`ext/out/preds`, `ext/out/masks`, `ext/results`). Everything is written to `ext/results_final/`.

```bash
python ext/evaluate_ext.py --out ext/results_final       # all models, plus the paired e1 - c4 comparison (--pairs)
python ext/pixel_eval.py metrics --pool <dataset_yolo> --out ext/results_final
python ext/analysis/holm.py                               # Holm per family: planned variants, second round, e1 - c4
python ext/analysis/mechanism.py                          # recall and where background false positives land, per model
python ext/analysis/final_tables.py --server <server ext/results>   # table_latency, table_sam3, table_valbias
```

(In the public release the outputs of these commands are under `results/ext/results_final/`; the server-side
inputs `latency_ext.csv` and `sam3_*/` are under `results/ext/server_results_20261009/`.)

## Pixel-level evaluation (R2.9)

`ext/pixel_eval.py` scores corrosion as a region. The rules were fixed before any mask was scored:
- **Ground truth:** the corrosion polygons, rasterised at image size.
- **Output of a model:** the union of its masks (`x3_seg`, `x4_seg_cp`) or of its filled boxes (every model).
- **Threshold:** chosen on each model's inner validation.
- **Primary metric:** dataset-level pixel IoU over the out-of-fold test images, with cluster-bootstrap intervals and paired differences against the baseline's filled boxes.
- **Reference row:** filled ground-truth boxes, the best any box output can reach. Locally this measures pixel IoU 0.534; the baseline's filled boxes reach 0.158.

```bash
bash ext/run_ext.sh pixel      # predict masks with the x3/x4 checkpoints (GPU, minutes), then metrics (CPU)
```

## Server

```bash
cd <project root>/revision_nested_cv        # <project root> also holds dataset_yolo/ (see the top-level README)
conda activate yoloenv                       # ultralytics 8.4.5, torch 2.8.0 (requirements-lock.txt)
python -m zipfile -e ext_paquete.zip .   # creates ext/ (scripts + ext/data, 136 MB); or: unzip -q ext_paquete.zip
bash ext/run_ext.sh check                # data checksums + dry run + free disk (>= 5 GB), no GPU
mkdir -p ext/out/logs
nohup bash ext/run_ext.sh train > ext/out/logs/train_all.log 2>&1 &
tail -f ext/out/logs/train_all.log       # ends with "all variants finished" or "FAILED: ..."
bash ext/run_ext.sh evaluate             # when training has finished
bash ext/run_ext.sh combine              # trains x5_combined only if decision.json lists two or more variants
bash ext/run_ext.sh evaluate             # again, only if combine trained something
conda activate sam2                          # Python >= 3.10 with transformers >= 5.13 (Sam3Model)
bash ext/run_ext.sh sam3                 # SAM 3 with ground-truth boxes and with text prompts
```

Expected GPU time on the RTX 4090:
- x1, x3 and x4: about 30-45 min each (15 runs of 261 images).
- x2: about 3-4 h (about 1,300 training images per run).
- SAM 3: under 15 min.
- x5_combined, if trained: 30-45 min, or 3-4 h when it includes x2.

These estimates use the measured 2.0 min per YOLO26-m run of the grouped baseline.

Outputs land in `ext/out/` (predictions and runs) and `ext/results/`:
- `summary.csv`, `deltas.csv` and `validation.csv`.
- `decision.json`, the variants allowed into a combination.
- `table_variants.tex`, ready for `\input` in the paper.

## Local checks done before shipping

- **`build_data.py`:**
  - 569 polygons aligned one-to-one with the 569 evaluated boxes (3 rectangle fallbacks).
  - 1,591 extra images.
  - 35 LIACI originals unreadable in the local copy, skipped and listed in the manifest.
- **`train_ext.py`:**
  - `--dry-run` of every variant.
  - CPU smoke test with ultralytics 8.4.5 (1 epoch, fold r0/f0) for `x2_multi` and `x4_seg_cp`.
    - Each trains, predicts the 6 views and writes a pkl with the same images, ground truth and format as the baseline.
    - `x1_recipe` and `x3_seg` share those code paths (detection and segmentation).
  - `--combine` dry runs, including the incompatible and duplicate cases.
- **`evaluate_ext.py`:** reproduces r1 on the baseline. mAP@50 0.1665 standard / 0.1938 TTA, F1 at the validation threshold 0.273, sequential − direct +0.0013.
- **`sam3_oracle.py --mock-sam`:** box-fill bound IoU 0.587, Dice 0.727, 71.4 % with IoU > 0.5.
