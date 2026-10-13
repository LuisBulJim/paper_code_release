# `revision/` — reproducible pipeline of the R1 revision

Everything reads `revision/configs/protocol.yaml` (paths, splits, thresholds, TTA/WBF defaults, pinned
versions) and `revision/configs/hparams.yaml` (training arguments shared by every run). Run modules from
the repository root with `python -m revision.<module>`; each one has `--help`.

## Stages and outputs

| Stage | Module | Reads | Writes |
|---|---|---|---|
| Dataset | `build_dataset` | raw LIACI (`images/`, `masks/corrosion/`, `train_test_split.csv`), `data/test_images.json` | `data/dataset_yolo_v2/`, `data/manifest_v2.json` |
| Splits | `make_splits` | manifest | `data/folds_v2.json` (stratified 5-fold x seeds 0,1,2 over the 228-image pool) |
| Stats | `dataset_stats` | dataset v2 | `results/dataset_stats.json`, `figures/dataset_boxes.png` |
| Training | `train_kfold --model yolo26{n,m,l} --seed S [--tag stl --init-weights ...]` | folds, hparams | `runs/<tag>/<model>_seed<S>/fold<k>/{weights/best.pt, preds_val, preds_test, metrics_val.json, run_info.json, DONE}` |
| Aerial pre-training | `train_aerial --data <aerial yaml>` | hparams | `runs/aerial/<model>/weights/best.pt` |
| Consolidation | `collect_runs` | run dirs | `results/cv_metrics.csv`, `results/test_metrics.csv`, `results/summary.csv` |
| Protocol comparison | `compare_protocols --a dtl --b stl` | the CSVs above | `results/protocol_comparison.csv` |
| Validation sweep + choice | `sweep_inference` | fold checkpoints | `results/sweep_tta_val*.csv`, `results/sweep_wbf_val*.csv`, `results/inference_choice.json` |
| Test inference | `run_test_inference` (reads the choice file) | fold checkpoints, choice | `results/inference_test*.csv`, `results/inference_test_config.json` |
| Latency | `benchmark_latency --standard ckpt --wbf ckpt ckpt` | checkpoints | `results/latency.csv` |
| Failures | `failure_analysis --standard dump --tta dump` | test dumps | `results/failure_by_size.csv`, `results/failure_cases.csv`, `figures/failure_cases.png` |
| TrashCan | `trashcan_prepare`, `trashcan_train`, `trashcan_ood` | TrashCan release, LIACI checkpoints | `data/trashcan_v2/`, `results/trashcan*.csv`, `results/trashcan_ood*.csv` |
| SAM 3 | `sam3_prepare`, `sam3_eval predict`, `sam3_eval metrics` | raw LIACI, detector checkpoint | `data/sam3_eval/`, `runs/sam3/*`, `results/sam3_*.csv`, `figures/sam3_cases.png` |
| Manuscript | `make_tables`, `make_figures`, `check_consistency --tex file.tex` | `results/`, `data/` | `tables/*.tex`, `tables/numbers.json`, `figures/*.png` |
| Audit | `audit_checkpoints runs/dtl`, `verify_submission_consistency` | checkpoints | report on architecture family and test-in-train overlap |

Shell wrappers for the GPU server: `scripts/setup_server.sh`, `run_dtl.sh`, `run_stl.sh`, `run_inference.sh`,
`run_trashcan.sh`, `run_sam3.sh` (all resumable: finished folds/seeds are skipped).

## Rules enforced by the code

* The 33 test images are frozen in `data/test_images.json`; `train_kfold` refuses fold files generated
  from another manifest, and `audit_checkpoints` reads the training list stored in every checkpoint and
  fails if a test image appears in it.
* Predictions are dumped once at confidence 0.001; every metric comes from `evaluate.py`
  (`docs/evaluation_protocol.md`).
* TTA/WBF settings are chosen on validation dumps by a fixed rule; every later step reads `results/inference_choice.json`.
* Prediction dumps encode their settings in the directory name and are re-used only when `_meta.json` matches.
* Every `run_info.json` records the hyper-parameters, the environment (ultralytics/torch/CUDA versions),
  the checksum of `best.pt`, the folds file and the manifest it was trained from.

## Local checks (CPU)

```bash
python -m pytest -q tests
python -m revision.protocol --check
python -m revision.train_kfold --model yolo26n --seed 0 --dry-run --device cpu   # ~2 min
python -m revision.collect_runs --include-dry-run
python -m revision.run_test_inference --include-dry-run --models yolo26n --tta-config tta_s3_flip --wbf-models none --device cpu --out runs/dryrun/results
```
