# Per-fold predictions, masks and checkpoints

The per-fold outputs and the trained weights are too large for git. They are deposited, together with the
derived label files of the improvement experiments, in the **archived release** of this repository
(Zenodo, DOI: *to be added*; the same DOI is quoted in the manuscript's Data availability statement).

## `predictions_per_fold.zip` (236 MB zipped, 355 MB unpacked, 11,697 files)

| Folder in the archive | Content | Produced by |
|---|---|---|
| `main_study/preds_grouped/<tag>_r<r>_f<k>_<size>.pkl` | 90 pickles (direct / sequential x n / m / l x 3 repeats x 5 folds): raw predictions (conf >= 0.001, normalised xyxy) of every test-time view (`std`, `flip`, `imgsz\|<scale>\|<flip>`, fine scales, rotations) on the inner-validation and outer-test images of the fold, the ground truth, and a fingerprint of the split | `revision_nested_cv/nested_cv.py`, `predict_extra_views.py` (grouped run, final) |
| `main_study/preds_ungrouped_ablation/*.pkl.bak` | the same 90 files of the earlier **ungrouped** run (outer folds not grouped by video); the ablation of the manuscript | same scripts, earlier run |
| `main_study/oof_json/oof/<tag>_<model>_<strategy>_r<r>/image_XXXX.json` | out-of-fold detections per image (standard, TTA, WBF) rebuilt from the cached views, one folder per (protocol, model, strategy, repeat), with `_meta.json` recording the configuration chosen on validation | `r1/nested_eval.py evaluate` |
| `main_study/sam3_oof/{oof_standard,oof_standard_valthr}/` | SAM 3 masks (PNG) and prompting boxes (JSON) on the 191 images of the official LIACI test split, detector = the fold model that never saw the image | `r1/sam3_eval.py predict` |
| `ext/preds/<variant>_r<r>_f<k>_m.pkl` | 105 pickles of the improvement experiments: x1_recipe, x2_multi, x3_seg, x4_seg_cp, x6_multi_seg (15 each), e1_ensemble and c4_baselines (fused) | `ext/train_ext.py`, `ext/ensemble_eval.py` |
| `ext/masks/<variant>_r<r>_f<k>.pkl` | 45 pickles of segmentation masks of x3_seg, x4_seg_cp and x6_multi_seg for the pixel-level evaluation | `ext/pixel_eval.py predict` |
| `ext/sam3_boxfill/` | filled ground-truth boxes used as the box-fill bound of the SAM 3 table | `ext/sam3_oracle.py --mock-sam` |

Unpack it as `revision_nested_cv/out/preds/` (main study), `revision_nested_cv/r1/runs/nested/grouped/oof/`,
`revision_nested_cv/r1/runs/sam3/`, `revision_nested_cv/ext/out/preds/`, `revision_nested_cv/ext/out/masks/`
and `revision_nested_cv/ext/out/sam3_boxfill/` to re-run the CPU evaluations (`r1/run_r1.sh evaluate`,
`ext/run_ext.sh evaluate`, `ext/run_ext.sh pixel` metrics step, `ext/analysis/*.py`) without a GPU.

The SAM 3 masks of the ground-truth-box and text-prompt runs (`ext/out/sam3_gt_boxes/`, `ext/out/sam3_text/`)
stayed on the GPU server; their metrics are in `results/ext/server_results_20261009/`.

## Checkpoints

* 90 fold models of the main study: `out/runs/<tag>_r<r>_f<k>_<size>/weights/best.pt` (about 3.1 GB). Their
  `args.yaml` and training curves (`results.csv`) are in `results/main_study/runs/`.
* 75 models of the improvement variants: `ext/out/runs/<variant>_r<r>_f<k>_m/weights/best.pt`. Their `args.yaml`
  and `results.csv` are in `results/ext/runs/`.
* 3 TrashCan retrainings (`r1/runs/trashcan/yolo26m_seed{0,1,2}/weights/best.pt`) and the 3 aerial phase-1
  checkpoints of the sequential-transfer protocol.

Inference with these checkpoints must use `ultralytics==8.4.5` (see `requirements-lock.txt`).

## Derived labels of the improvement experiments (`ext_data_derived_labels.zip`)

`ext/data/labels_seg/` (corrosion polygons of the 261 images), `ext/data/labels_multi/` and
`ext/data/labels_seg_multi/` (nine-class boxes and polygons for the pool and the 1,591 extra images), built from
the LIACI annotations by `ext/build_data.py` and `ext/build_seg_multi.py`; their sha256 manifests are in
`revision_nested_cv/ext/data/`. The extra images themselves are LIACI images and must be taken from the LIACI
release.
