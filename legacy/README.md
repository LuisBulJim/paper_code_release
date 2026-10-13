# Legacy material (superseded, kept for transparency)

Nothing in this folder is used by the revised manuscript. It is kept unchanged so that the record of the first
submission and of the revision process stays public.

| Folder | What it is | Status |
|---|---|---|
| `original_submission/` | The executed notebooks, `src/utils_research.py`, the frozen 33-image split (`data/splits.json`), `data/kfold_sigma.csv`, `requirements.txt` and the project-page images of the submitted version (July 2026). | **Superseded.** The test images of its single split had been used to train the cross-validation models, and the checkpoints behind its tables were not the networks described. See `original_submission/README.md`. |
| `revision_draft/` | An earlier re-implementation of the revision (`revision/` package, `scripts/`, `tests/`, draft protocol docs, dataset manifests and local dry-run results) written before the grouped nested cross-validation was adopted. | **Not what was run.** It diverges from `revision_nested_cv/` (different protocol: single frozen test set + 5-fold over a 228-image pool, different `ultralytics` pin) and none of its outputs appears in the manuscript. See `revision_draft/README.md`. |

The code and results that back the revised manuscript are in `revision_nested_cv/` and `results/` at the
repository root.
