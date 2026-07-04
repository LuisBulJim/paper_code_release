# Underwater Corrosion Detection via YOLO26 — Reproducibility Package

Code and executed notebooks backing the manuscript
**"Underwater Corrosion Detection via YOLO26: From Heterogeneous Ensembles to Robust Inference with Test-Time Augmentation."**

Every quantitative result in the paper is produced by a notebook in this repository, and the
notebooks are published **with their executed outputs** so each reported number can be checked
directly against the cell that computes it.

---

## 1. Repository structure

```
├── README.md
├── requirements.txt            # pinned environment (see the critical pin note inside)
├── notebooks/                  # one notebook per experiment, executed outputs included
│   ├── 00_dataset_underwater_prep.ipynb     # LIACI dataset preparation (N=261)
│   ├── 01_dataset_visualization.ipynb       # dataset inspection / sanity checks
│   ├── 02_training_aerial_pretrain.ipynb    # aerial pre-training (N=36,102)
│   ├── 03_training_underwater_kfold.ipynb   # Direct Transfer, Stratified 5-Fold CV
│   ├── 04_training_mixed_sequential.ipynb   # Sequential Transfer baseline
│   ├── 05_model_comparison.ipynb            # 9-model comparison
│   ├── 06_wbf_ensemble_rebuild.ipynb        # WBF ensemble (fixed thresholds)
│   ├── 07_tta_stta_inference.ipynb          # TTA / S-TTA inference ablation
│   ├── 08_trashcan_generalization.ipynb     # TrashCan cross-domain verification
│   └── 09_sam3_box_prompted_eval.ipynb      # SAM 3 zero-shot segmentation validation
├── src/
│   └── utils_research.py       # shared config, seeding, split loading, COCO-style metric
└── data/
    ├── splits.json             # frozen train/val/test partition (exact test set, 33 images)
    └── kfold_sigma.csv         # per-model mAP@50 dispersion across the 5 CV folds
```

## 2. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Two version notes (both enforced in `requirements.txt`):
- **`ultralytics==8.4.5` is a hard requirement.** YOLO26 is NMS-free and its detection
  decoding changes across Ultralytics releases; newer versions produce a different number of
  detections from the same weights and do not reproduce the reported metrics.
- **SAM 3 requires a `transformers` release providing `Sam3Model`/`Sam3Processor`**
  (verified with 5.13.0). Notebook 09 fails with an explicit error otherwise — SAM 2.x is not
  a valid substitute for the reported results.

GPU (CUDA) is required for training (02–04) and recommended for inference (05–09); the
reference machine was an NVIDIA RTX 4090.

## 3. Datasets

| Dataset | Role | Source |
|---|---|---|
| Aerial corrosion (36,102 imgs) | Sequential-transfer pre-training | Roboflow *Corrosion-hsmae* (notebook 02 downloads it; set `ROBOFLOW_API_KEY`) |
| **SINTEF LIACI** (261 imgs) | Target domain — all headline metrics | https://liaci.sintef.cloud/ |
| **TrashCan 1.0** (material_version) | Cross-domain verification | notebook 08 downloads it |

Datasets are not redistributed here (licensing). `data/splits.json` defines the exact
train/val/test partition used everywhere, so the held-out 33-image test set is reconstructible.
Trained checkpoints will be deposited in a DOI archive upon acceptance.

## 4. Which notebook produces which result

| Paper result | Notebook |
|---|---|
| 9-model comparison table | `05_model_comparison.ipynb` |
| WBF ensemble table + standalone references | `06_wbf_ensemble_rebuild.ipynb` |
| Inference-strategy table (Baseline / TTA / S-TTA) and latency | `07_tta_stta_inference.ipynb` |
| TrashCan cross-domain table (+6.28% Recall) | `08_trashcan_generalization.ipynb` |
| SAM 3 segmentation validation (IoU/Dice, N=21) | `09_sam3_box_prompted_eval.ipynb` |
| K-fold dispersion (σ of mAP@50) | `03_…kfold.ipynb` + `data/kfold_sigma.csv` |

## 5. Evaluation conventions

- The **model-comparison table** (notebook 05) uses the standard Ultralytics `val()` protocol.
- All **inference-strategy comparisons** (notebooks 06 and 07) use a single COCO-compatible
  implementation (`src/utils_research.py::compute_metrics_coco`) at **fixed** thresholds
  (confidence = 0.25, IoU = 0.5), with no parameter selected on the test partition.
  The two conventions are not directly comparable across tables (stated in the paper).
- Cross-check built in: the *Medium (standalone)* row of notebook 06 equals the *Baseline*
  row of notebook 07 exactly — both computed independently with the same metric.
- Reproducibility: seed 42, deterministic cuDNN, frozen `data/splits.json`.
