# Original submission (July 2026) — superseded

This folder holds, unchanged, the material that accompanied the first submission of the manuscript: the ten
executed notebooks (`notebooks/`), the shared utilities (`src/utils_research.py`), the frozen single split
(`data/splits.json`, 33 test images, with `val == test`), the K-fold dispersion file (`data/kfold_sigma.csv`), the
environment file (`requirements.txt`) and the qualitative images of the former project page (`docs_assets/`).

**Why it is superseded.** The single train/test split of the submitted version was not clean: the 5-fold
cross-validation of notebook 03 was built over all 261 images, so the 33 held-out test images were inside the
training folds (24 to 29 of them per fold), the background images had been sampled from the whole LIACI release
including its official test split, and the checkpoints evaluated in notebooks 05 to 07 were not the YOLO26
networks the paper described. Every number of the submitted version (mAP@50 0.82 for the Medium model, the
WBF and TTA tables, the TrashCan +6.28 % recall, the SAM 3 IoU 0.56) is therefore invalid and is not used
anywhere in the revised manuscript. The revised study re-trains everything under a repeated nested
cross-validation grouped by video (see the top-level `README.md` and the manuscript); its code is in
`revision_nested_cv/` and its outputs in `results/`.

The notebooks are kept exactly as published (executed outputs included, with the server paths of the original
runs); only notebook 00 (dataset conversion from the LIACI masks) and notebook 02 (aerial pre-training used by the
sequential-transfer protocol) are still referred to by the revised pipeline, for the construction of
`dataset_yolo/` and of the aerial phase-1 checkpoints.
