# Second revision (R2) analyses

Scripts and outputs added for the second round of review (OE-D-26-08406 R2). They reuse the main-study code in
`revision_nested_cv/` (`nested_cv.py`, `r1/`, `ext/pixel_eval.py`) unchanged and read its cached per-fold
predictions; every number quoted in the R2 manuscript, supplementary material and response letter comes from a
CSV / JSON in this folder. Run the scripts from the repository root, e.g. `python results_rev2/r2_2_nested_selection.py`.

| Reviewer point | Script | Outputs | Main numbers |
|---|---|---|---|
| R2.1(a) temporal buffer | `r2_1ab_train.py --setting purge{5,15,30}` (GPU, 15 Medium models each), `r2_1a_evaluate.py`, `r2_1/run_chain.sh` | `r2_1/purge*/splits.json`, `losses.csv`; `r2_1/purge_{summary,deltas,losses_summary}.csv`, `purge_numbers.json` | mAP@50 0.131 / 0.120 / 0.097 for B = 5 / 15 / 30 (paper split 0.167); up to 16 / 40 / 61 training + validation images lost per fold |
| R2.1(c) coarser bootstrap units | `r2_1c_supercluster_bootstrap.py` | `r2_1c/` | 28 → 17 / 10 / 6 groups; median interval width × 1.26 / 1.35 / 1.47; six comparisons no longer exclude zero with 17 groups |
| R2.2 selection within each outer fold | `r2_2_nested_selection.py` (CPU, about 10 min) | `r2_2/per_fold_choices.csv`, `validation_grid.csv`, `summary.csv`, `deltas.csv`, `numbers.json` | fixed recipe 0.262; same recipe with TTA / WBF / threshold per fold 0.245; full per-fold selection 0.150 |
| R2.3 chronology | `r2_3_chronology.py` | `r2_3/chronology.json` | archive and log dates of the improvement experiments (needs the `ext_*.zip` packages, not in git) |
| R2.4 box-shaped ground truth | `r2_4_tiled_gt.py` | `r2_4/` | one filled box per component 0.534; 32 px cells 0.578; 8 px cells 0.858 |
| R2.5 SAM 3 positives and neighbours | `r2_5_neighbour_distance.py`, `r2_5_sam3_farthest.sh`, `r2_5_exclude_empty_masks.py`, `r2_5_false_mask_by_distance.py`, `r2_5_audit_missing_positives.py` | `r2_5/` | 21 positives + 168 negatives after excluding the two images with an empty mask (`image_0105`, `image_0763`) |
| R2.7 data flow | `r2_7_data_flow.py` | `r2_7/` | 1,893 LIACI frames = 261 pool + 1,591 extra + 41 not used (35 unreadable, 6 without a mask) |
| Tables | `make_tex_tables.py` | `tex/*.tex` | the R2 tables `\input` by the supplementary material |

`r2_1ab_train.py --setting gap30` and `r2_1b_evaluate.py` (retraining with 17 groups) were prepared but not run;
the 17-group analysis reported in the manuscript is the bootstrap of `r2_1c`.

Interpreters: the shell scripts call `$PYTHON` (main environment, `ultralytics==8.4.5`) and `$PYTHON_SAM3`
(SAM 3 in `transformers>=5.8`), both defaulting to `python`. The `run.log` files are the console output of the
runs on the GPU server and keep its paths.

Not in git (archived with the predictions on Zenodo, see `results/PREDICTIONS.md`): the out-of-fold pickles of
the 45 buffer models (`r2_1/purge*/preds/`), their training curves and `args.yaml` (`r2_1/purge*/runs/`), the
training logs (`r2_1/logs/train_*.log`), the SAM 3 masks and boxes (`r2_5/sam3_farthest_{std,valthr}/`) and the
qualitative SAM 3 figures. The 45 buffer checkpoints are not archived; `splits.json` and the script regenerate them.
