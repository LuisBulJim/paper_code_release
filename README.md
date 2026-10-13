# Underwater Corrosion Detection with YOLO26 — code and results release

Code, splits, group assignment, per-fold results and table/figure scripts of the manuscript
**"Underwater Corrosion Detection with YOLO26: Comparing Model Scaling, Ensemble Fusion, and Test-Time
Augmentation"** (under review, *Ocean Engineering*, revised version).

Project page: <https://luisbuljim.github.io/paper_code_release/>

> **This release supersedes the original submission.** The notebooks of the first submission evaluated a
> single train/test split whose test images had also been used to train the cross-validation models, and
> the checkpoints behind its tables were not the networks the paper described. None of those numbers is
> used any more. The original material is kept unchanged under [`legacy/original_submission/`](legacy/original_submission/)
> for the record; see [`legacy/README.md`](legacy/README.md).

## 1. What the revised study does

* **Data.** The 261 LIACI images with corrosion annotations converted to boxes (`dataset_yolo/`, not
  redistributed; `revision_nested_cv/r1/data/nested_manifest.json` holds the sha256 of every image and label).
* **Protocol.** Repeated nested cross-validation, 3 repeats x 5 outer folds, with the outer folds **grouped by
  video**: LIACI does not state which video each frame comes from, so the groups are runs of at most 15 pool
  images in frame-index order, cut where the index gap exceeds 30 (**28 groups**,
  `results/groups/groups.csv`). Inner validation is the only data used for early stopping, checkpoint
  selection, the TTA / WBF configuration and the operating threshold. YOLO26 Nano / Medium / Large, direct and
  sequential transfer: **90 fold models**. An earlier ungrouped run of the same pipeline is kept as the ablation
  that measures the leakage.
* **Evaluation.** One evaluator (`pycocotools`) on out-of-fold predictions: mAP@50, mAP@[.5:.95], precision /
  recall / F1 at a fixed threshold and at the validation threshold, cluster bootstrap over the 28 video groups
  (1,000 replicates, paired differences, Holm correction within families).
* **Improvement experiments** on the same splits (`revision_nested_cv/ext/`): small-data training recipe,
  nine-class training with the 1,591 unused LIACI images, a segmentation head, copy-paste augmentation, their
  combination, a four-variant WBF ensemble and a control ensemble of four baseline models (75 variant models).
* **Region-level evaluation** (pixel IoU of masks and filled boxes against the corrosion polygons), **SAM 3**
  prompted with ground-truth boxes and with text, **latency** of every inference recipe, and a supplementary
  check on **TrashCan 1.0** (retraining with the LIACI recipe, 3 seeds, plus zero-shot transfer).

## 2. Headline numbers (from the manuscript; every value is in `results/`)

| Result | Value | Source file |
|---|---|---|
| YOLO26-Medium, direct transfer, single pass, out-of-fold mAP@50 | **0.299 ungrouped → 0.167 grouped** [0.119, 0.209] | `results/main_study/r1_results/nested_{ungrouped,grouped}/summary.csv` |
| Four-baseline WBF ensemble + TTA (recipe selected on validation) | **0.262** mAP@50 at **3.4 images/s** (RTX 4090, FP32, batch 1) | `results/ext/results_final/summary.csv`, `table_latency.tex` |
| Nine-class model (unused LIACI images), single pass | **0.204** mAP@50 at **125 images/s** | same |
| Pixel IoU of masks vs. filled boxes | masks **0.188–0.229** vs. boxes **0.158** | `results/ext/results_final/pixel_summary.csv`, `table_pixel.tex` |
| SAM 3 with ground-truth box prompts / with text prompts | IoU **0.650** [0.553, 0.734] / **0.065** [0.003, 0.138] | `results/ext/results_final/table_sam3.tex` (inputs in `results/ext/server_results_20261009/`) |

Model size, transfer protocol, TTA and WBF under the grouped protocol: `results/main_study/r1_tables/nested.tex`
and `nested_deltas.tex`; the full list of generated tables is in `results/README.md`.

## 3. Repository map

```
.
├── README.md, LICENSE, requirements-lock.txt, RELEASE_CHECKLIST.md
├── revision_nested_cv/            the code that produced every result (run layout, see section 4)
│   ├── nested_cv.py               training of the 90 fold models + caching of all TTA views (GPU)
│   ├── build_groups.py            video groups from frame indices (groups.csv for --groups-csv)
│   ├── evaluate_cv.py, predict_extra_views.py, bench_latency.py, metrics_coco.py, common.py
│   ├── run_revision.sh            phase launcher: check | dryrun | smoke | direct | sequential | verify | extra | eval | latency
│   ├── README_nested_cv.md        notes on the main pipeline (Spanish)
│   ├── r1/                        single evaluator, cluster bootstrap, TrashCan, SAM 3, latency, tables and figures
│   │   ├── nested_eval.py         check | prepare | evaluate | link | sam3-map  (CPU re-evaluation of the cached views)
│   │   ├── evaluate.py, boxes.py, predict_dump.py, sweep_inference.py, run_test_inference.py
│   │   ├── trashcan_prepare.py, trashcan_train.py, trashcan_ood.py, sam3_eval.py, benchmark_latency.py
│   │   ├── failure_analysis.py, make_tables.py, make_figures.py, check_consistency.py, audit_checkpoints.py
│   │   ├── configs/protocol.yaml  single source of truth (paths, thresholds, bootstrap, environment versions)
│   │   ├── configs/hparams.yaml   training recipe of the 90 fold models (copied from their args.yaml)
│   │   ├── data/*.json            manifests with sha256 (261-image pool, SAM 3 set, TrashCan)
│   │   ├── run_r1.sh, README_r1.md (Spanish), tests/
│   ├── ext/                       improvement experiments, pixel evaluation, SAM 3 bounds, ensemble latency
│   │   ├── build_data.py, build_seg_multi.py   derived labels and extra images from the LIACI release
│   │   ├── train_ext.py, evaluate_ext.py, ensemble_eval.py, pixel_eval.py, sam3_oracle.py, latency_ext.py
│   │   ├── analysis/              diagnostics, confusion, similarity, natural experiment, Holm, final tables
│   │   ├── data/                  folds.json + manifests (sha256) of the derived data
│   │   └── run_ext.sh, README_ext.md
├── results/                       outputs of the above (CSV / JSON / TeX / TXT / PNG; no weights, no pickles)
│   ├── splits.json                the 3 x 5 outer/inner splits (seeds, test, val, train image lists)
│   ├── groups/groups.csv          the 28 video groups given to the training run (image -> group)
│   ├── groups/groups_report.csv   the same, with frame index and index gap per image
│   ├── groups/nested_groups.csv   the same partition rebuilt from the splits by r1 (other labels)
│   ├── main_study/                per-fold and out-of-fold metrics, bootstrap, sweeps, tables, figures, run records
│   ├── ext/                       improvement experiments, pixel IoU, SAM 3, latency, diagnostics
│   ├── PREDICTIONS.md             where the per-fold predictions and checkpoints are (archived release)
│   └── README.md                  file-by-file description
├── docs/                          project page (GitHub Pages, served from /docs)
└── legacy/                        superseded material, kept for transparency (not used by the manuscript)
    ├── original_submission/       executed notebooks, src/, splits and figures of the first submission
    └── revision_draft/            an earlier, unfinished re-implementation of the revision (not what was run)
```

## 4. Reproducing

### Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-lock.txt          # ultralytics==8.4.5 (hard requirement), ensemble-boxes==1.0.9, pycocotools, ...
```

Every number was produced with Ultralytics 8.4.5, Python 3.9.13 and torch 2.8.0+cu128 on an RTX 4090.
YOLO26 decoding changes between Ultralytics releases, so inference with the released checkpoints under another
version gives different detections; `run_r1.sh check` and `run_ext.sh check` refuse to run with another version.
SAM 3 needs a second environment with Python >= 3.10 and `transformers>=5.13` (gated `facebook/sam3` checkpoint
on the Hugging Face Hub; log in with your own account).

### Layout expected by the scripts

The scripts are relative to a *project root* that contains both the dataset and this code:

```
<project root>/
├── dataset_yolo/                images/{train,test}, labels/{train,test}: the 261 LIACI corrosion images at 640x640
│                                (train + test are pooled; the nested CV re-splits all 261 images)
├── yolo26n.pt, yolo26m.pt, yolo26l.pt                       COCO weights (Ultralytics downloads them)
├── Sequential_Transfer_Benchmark/phase1_aerial_yolo26{n,m,l}/weights/best.pt   aerial pre-training (sequential transfer)
└── revision_nested_cv/          this folder, copied from the repository
```

`dataset_yolo/` is built from the LIACI release with the conversion of the original submission
(`legacy/original_submission/notebooks/00_dataset_underwater_prep.ipynb`: one box per connected corrosion
component at 640x640); `python -m r1.protocol --check` verifies a rebuilt copy against the sha256 manifest.
The aerial checkpoints come from the Roboflow pre-training of the original submission (notebook 02) and are
part of the archived release.

### Main study (GPU: 90 trainings, about 2 min each for Medium; then CPU)

```bash
cd <project root>
bash revision_nested_cv/run_revision.sh check
python revision_nested_cv/build_groups.py cluster --data-root dataset_yolo --out groups.csv --report groups_report.csv --method idgap --fixed-cut --min-id-gap 30 --max-group-size 15
GROUPS_CSV=groups.csv bash revision_nested_cv/run_revision.sh direct        # 45 models
GROUPS_CSV=groups.csv bash revision_nested_cv/run_revision.sh sequential    # 45 models
bash revision_nested_cv/run_revision.sh verify                              # every .pkl matches the current splits
```

The released `results/splits.json` is the exact split used in the paper (seeds 42 + repeat), and
`results/groups/groups.csv` is the group file given to that run: `run_revision.sh verify` with it reports every
model as trained on a matching split, and the command above regenerates it byte for byte. `r1/nested_eval.py`
rebuilds the groups from the splits (`results/groups/nested_groups.csv`): the partition is identical, only the
labels differ, and the paper's bootstrap intervals were computed with those rebuilt labels.

```bash
bash revision_nested_cv/r1/run_r1.sh check         # environment, dataset manifest, splits, pickles, checkpoints
bash revision_nested_cv/r1/run_r1.sh evaluate      # CPU: validation sweep -> inference_choice.json; OOF metrics, bootstrap
bash revision_nested_cv/r1/run_r1.sh link          # links checkpoints, writes the SAM 3 fold map
bash revision_nested_cv/r1/run_r1.sh trashcan      # GPU ~3.5 h: TrashCan retraining (3 seeds) + zero-shot OOD
bash revision_nested_cv/r1/run_r1.sh sam3          # GPU, SAM 3 environment: SAM 3 with the out-of-fold detector
bash revision_nested_cv/r1/run_r1.sh latency       # GPU ~10 min
bash revision_nested_cv/r1/run_r1.sh report        # failure analysis, LaTeX tables (r1/tables), figures (r1/figures)
python -m r1.check_consistency --tex <manuscript.tex>   # every number in the text must exist in r1/results
```

### Improvement experiments, pixel evaluation, SAM 3 bounds, latency (`ext/`)

```bash
cd <project root>/revision_nested_cv
python ext/build_data.py --liaci-images <LIACI images> --liaci-masks <LIACI masks> --coco <coco-labels.json>   # derived labels
python ext/build_seg_multi.py --coco <coco-labels.json>
bash ext/run_ext.sh check
bash ext/run_ext.sh train                          # x1_recipe, x3_seg, x4_seg_cp, x2_multi (15 folds each)
bash ext/run_ext.sh one x6_multi_seg
bash ext/run_ext.sh ensemble                       # e1_ensemble and the c4_baselines control (CPU)
bash ext/run_ext.sh evaluate                       # summary.csv, deltas.csv, validation.csv, decision.json, table_variants.tex
bash ext/run_ext.sh latency                        # GPU
bash ext/run_ext.sh pixel                          # masks of x3/x4 (GPU) + pixel IoU of every model (CPU)
bash ext/run_ext.sh sam3                           # SAM 3 environment: ground-truth boxes and text prompts
python ext/analysis/holm.py; python ext/analysis/mechanism.py; python ext/analysis/final_tables.py --server ../results/ext/server_results_20261009
```

`ext/analysis/diagnostics.py`, `confusion.py`, `similarity.py`, `official_split.py`, `natural_experiment.py`
and `extra_proximity.py` read the raw LIACI release (`coco-labels.json`, `train_test_split.csv`,
`embeddings_resnet101.json`); point `LIACI_ROOT` at that folder.

## 5. Datasets (third-party, not redistributed)

| Dataset | Role | Source |
|---|---|---|
| **SINTEF LIACI** | all headline results (261 corrosion images; nine classes and 1,591 extra images for `x2_multi` / `x6_multi_seg`; official test split for SAM 3) | <https://liaci.sintef.cloud/> |
| **TrashCan 1.0** (material version) | supplementary check | <https://conservancy.umn.edu/handle/11299/214865> (`r1/trashcan_prepare.py` converts it) |
| Aerial corrosion (Roboflow *Corrosion-hsmae*) | sequential-transfer pre-training | downloaded by `legacy/original_submission/notebooks/02_training_aerial_pretrain.ipynb` with your own `ROBOFLOW_API_KEY` |

The derived label files of the improvement experiments (corrosion polygons and nine-class labels, built from the
LIACI annotations by `ext/build_data.py`) are not in this repository; their sha256 manifests are, and the files are
included in the archived release.

## 6. Predictions and checkpoints

Per-fold predictions (all TTA views, 90 + 105 pickles), out-of-fold dumps, SAM 3 masks and the trained
checkpoints (90 fold models of the main study, 75 of the improvement variants) are too large for git and are
deposited in the archived release described in [`results/PREDICTIONS.md`](results/PREDICTIONS.md).

## 7. License and citation

Code and result files: MIT License (see `LICENSE`). The datasets keep the licenses of their original sources.

```bibtex
@article{underwater_corrosion_yolo26,
  title   = {Underwater Corrosion Detection with YOLO26: Comparing Model Scaling, Ensemble Fusion, and Test-Time Augmentation},
  journal = {Ocean Engineering (under review)},
  year    = {2026},
  note    = {Code and results: https://github.com/LuisBulJim/paper_code_release; archived release: DOI to be added}
}
```
