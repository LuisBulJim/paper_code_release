# Revision draft (September 2026) — not the code that was run

`revision/`, `scripts/`, `tests/`, `docs/`, `figures/`, `results/`, `data/` and `requirements-lock.txt` in this
folder are an earlier re-implementation of the revision, written before the final protocol was fixed. It keeps a
single frozen 33-image test set and folds a 228-image pool (`data/folds_v2.json`, `data/manifest_v2.json`), pins
`ultralytics==8.4.164`, and its `revision/` package diverges from `revision_nested_cv/r1/` in most files
(different `make_tables.py`, `make_figures.py`, `sweep_inference.py`, `run_test_inference.py`, `collect_runs.py`,
`protocol.py`, extra `build_dataset.py`, `make_splits.py`, `train_kfold.py`, `train_aerial.py`,
`sam3_prepare.py`, `compare_protocols.py`; no `nested_eval.py`).

It was replaced by the repeated nested cross-validation grouped by video of `revision_nested_cv/` (3 x 5 folds
over all 261 images, 28 video groups, `ultralytics==8.4.5`), which is what the revised manuscript reports.
The `results/` here are local CPU dry runs (`runs/dryrun`, `runs/localcheck`), not experiment results.

Local absolute paths in `revision/configs/protocol.yaml`, `revision/verify_submission_consistency.py`, the
`data/*.json` manifests and `results/logs/*.log` were replaced by placeholders (`<LIACI_ROOT>`,
`<TRASHCAN_ROOT>`, `<SUBMITTED_DATASET_ROOT>`, `<LOCAL_DATA_ROOT>`) before publication. The prepared datasets
(`data/dataset_yolo_v2/`, `data/sam3_eval/`, `data/trashcan_v2*/`) are not in git.
