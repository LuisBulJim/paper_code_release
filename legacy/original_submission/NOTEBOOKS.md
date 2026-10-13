# What each notebook produced and why it was replaced

(Record written during the revision, September 2026, originally `legacy/README.md`; the "replaced by" column
refers to the scripts of the earlier revision draft, now under `legacy/revision_draft/revision/`. The final
revised study uses `revision_nested_cv/` instead; see the top-level `README.md`.)

The notebooks in `notebooks/` are the executed notebooks that accompanied the submitted manuscript
(OE-D-26-08406). They are kept unchanged for transparency. **None of their outputs is used in the revised
manuscript.**

| Notebook | What it produced in the submitted manuscript | Why it is replaced |
|---|---|---|
| `00_dataset_underwater_prep.ipynb` | `dataset_yolo/` (261 images: 209 corrosion + 52 background) from the raw LIACI masks | Background images were sampled from the whole release, including LIACI's official test split; `revision/build_dataset.py` rebuilds the set from the raw masks with the same conversion rule, keeps the same 33-image test set and samples background images with a fixed seed outside that split (`data/manifest_v2.json`). |
| `03_training_underwater_kfold.ipynb` | Direct-transfer 5-fold cross-validation and the "K-fold dispersion" (`data/kfold_sigma.csv`) | The folds were built over all 261 images, i.e. the 33 test images were inside the training folds (24-29 of them per fold). `revision/make_splits.py` folds only the 228-image training pool (`data/folds_v2.json`); `revision/train_kfold.py` retrains n/m/l x 3 seeds. |
| `02_training_aerial_pretrain.ipynb`, `04_training_mixed_sequential.ipynb` | Sequential-transfer baseline | Pre-training and fine-tuning used different hyper-parameters from direct transfer (lr0 0.005, freeze 10, 20 epochs) and validated on the training images. `revision/train_aerial.py` + `train_kfold.py --init-weights` use the single `hparams.yaml` and the same folds. |
| `05_model_comparison.ipynb` | 9-model comparison table (mAP@50 0.8196 for YOLO26-m etc.) | The evaluated checkpoints (`modelo-acuatico-*.pt`, Nov 2025) are YOLOv12 networks (A2C2f blocks), not YOLO26, and were trained with the leaking folds. Table replaced by the 15-checkpoint test evaluation of `revision/collect_runs.py`. |
| `06_wbf_ensemble_rebuild.ipynb` | WBF ensemble table (0.6290 / 0.7500 / 0.6842 / 0.7039) | Same checkpoints and evaluator issue; ensemble members and weights chosen on the test set. Replaced by `revision/sweep_inference.py` (validation) + `run_test_inference.py`. |
| `07_tta_stta_inference.ipynb` | Inference-strategy table and latencies (12 / 76 / 45 ms) | Custom evaluator with a 0.25 confidence floor (truncated PR curve); latencies were not measured with a common protocol. Replaced by `run_test_inference.py`, `benchmark_latency.py`, `failure_analysis.py`. |
| `08_trashcan_generalization.ipynb` | TrashCan cross-domain table (+6.28% recall) | Single train/val split, "standard" evaluated with Ultralytics `val()` and "TTA" with the custom evaluator. Replaced by `trashcan_prepare.py` / `trashcan_train.py` (3 seeds, single evaluator, selection split inside train) and `trashcan_ood.py` (zero-shot test). |
| `09_sam3_box_prompted_eval.ipynb` | SAM 3 validation (IoU 0.56, Dice 0.67, 71.4% on 21 images) | The 191-image set is LIACI's official test split, of which 4 images were in the detector's training set; images without corrosion and without mask were scored IoU = 1. Replaced by `sam3_prepare.py` (0 overlap with the v2 training pool) and `sam3_eval.py` (positives and negatives reported separately). |
| `01_dataset_visualization.ipynb` | Dataset figures | Replaced by `revision/dataset_stats.py` and `make_figures.py`. |

`src/utils_research.py` and `data/splits.json` are kept for the same reason; note that `splits.json` lists
the 33 test images also as the validation split (`val == test`), which is why the revision draft introduced
`data/folds_v2.json`.
