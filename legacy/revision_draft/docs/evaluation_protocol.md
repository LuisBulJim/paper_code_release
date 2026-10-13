# Evaluation protocol (revision R1)

Every number reported in the revised manuscript is produced by `revision/evaluate.py` from prediction
dumps written by `revision/predict_dump.py`. No table mixes evaluators, and no threshold is chosen by
looking at the held-out test set.

## Inputs

* **Predictions**: one JSON per image with pixel `xyxy` boxes, scores and class ids, dumped at
  confidence >= 0.001 (`evaluation.conf_dump`) so that the full precision-recall curve is available.
  The same dump serves mAP and every operating point; nothing is re-run at a different threshold.
* **Ground truth**: YOLO label files of `data/dataset_yolo_v2` (one box per connected corrosion region at
  640x640, see `data/manifest_v2.json` for the conversion rule and checksums).

## Metrics

| Metric | Definition | Implementation |
|---|---|---|
| mAP@50, mAP@[.5:.95] | COCO average precision, 101-point interpolation, `maxDets=100`, area = all | `pycocotools.COCOeval` on the dump |
| Precision, Recall, F1 | at the fixed operating point conf 0.25 (`evaluation.conf_fixed`), IoU 0.5 (`evaluation.iou_match`), greedy score-ordered one-to-one matching, class-aware | `evaluate.match_details` |
| 95% CI | stratified bootstrap over images (positives and background images resampled separately, 1000 replicates, seed 0) | `evaluate.bootstrap_ci` |
| Paired difference | bootstrap of (B - A) with the same resampled images for both runs; reports mean delta, 95% CI and `p_le0` = share of replicates with delta <= 0 | `evaluate.paired_bootstrap` |
| Across checkpoints | mean, SD and 95% t-interval over the 15 fold checkpoints (5 folds x 3 seeds) of each model | `collect_runs.summarize`, `run_test_inference.summarize` |

## Why one evaluator

* **Ultralytics `model.val()`** reports mAP with its own interpolation and, for YOLO26 (NMS-free head),
  with the end-to-end decoding of the installed version; its numbers change across releases and are not
  comparable with a custom evaluator. It was used for the model-comparison table of the submitted
  version while the inference tables used a custom evaluator (the source of the discrepancy the
  reviewers noticed).
* **The custom evaluator of the submitted version** matched at a confidence floor of 0.25, which
  truncates the precision-recall curve and under-estimates mAP; its F1 used the same threshold, so mAP and
  F1 were not computed on the same detections as `model.val()`.
* The revision dumps once at 0.001 and derives everything from that dump, so mAP, P/R/F1 at 0.25 and the
  bootstrap intervals describe the same predictions.

## Selection rules

* `best.pt` of each fold is chosen by Ultralytics fitness (0.1 mAP@50 + 0.9 mAP@[.5:.95]) on the fold's
  validation split; the test set is never seen during training.
* TTA and WBF settings (scales, flip, view weights, WBF IoU, ensemble members and weights) are compared on
  the validation dumps only (`sweep_inference.py`, `results/sweep_*_val_summary.csv`). The rule is fixed in
  code: TTA = highest validation mAP@50 averaged over the model sizes (one configuration for all sizes);
  WBF = highest mean validation mAP@50. The choice is written to `results/inference_choice.json` and applied
  unchanged to the test set, the latency benchmark, TrashCan and SAM 3.
* The operating point (0.25) is fixed a priori in `protocol.yaml`; the validation sweep also reports F1 at
  other thresholds as a sensitivity analysis, not as a selection.
* On the test set every one of the 15 checkpoints per model is evaluated; the manuscript reports the mean
  with its 95% interval instead of a single hand-picked checkpoint.

## Multi-class datasets

TrashCan (Metal, Bio) uses exactly the same code: class ids travel with the boxes, matching is class-aware,
COCO categories are the union of ground-truth and predicted classes, and `all_metrics(..., class_names)`
adds one block per class. LIACI dumps carry a single class.

## Segmentation (SAM 3)

`sam3_eval.py` reports positives and background images separately: per-image IoU and Dice with IoU = 0
when no mask is produced for a corroded image, the share of images with IoU > 0.5, the share detected at
all, and for background images the false-mask rate and false area. An empty prediction on an empty image is
a correct rejection and is not scored as IoU = 1.
