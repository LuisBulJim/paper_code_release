"""Mechanism check: where do the background false positives of each model land, and how much corrosion does it find?

Same definitions as confusion.py (background FP = prediction with IoU < 0.1 to every ground-truth box; it lands on the
LIACI class covering >= 20 % of its area), applied to every model at the same confidence and at its own validation
threshold. Counts are summed over the 3 repeats of out-of-fold test predictions (783 image evaluations).
"""
import csv
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from confusion import COVER, SIDE, class_masks  # noqa: E402
from diagnostics import COCO, FOLDS, REPEATS, iou_matrix  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODELS = {"x0_baseline": str(ROOT / "out" / "preds" / "direct_r{r}_f{k}_m.pkl")}
MODELS.update({m: str(ROOT / "ext" / "out" / "preds" / f"{m}_r{{r}}_f{{k}}_m.pkl") for m in ("x2_multi", "x3_seg", "x6_multi_seg", "e1_ensemble", "c4_baselines")})
GROUPS = {"paint_peel": "other LIACI class", "marine_growth": "other LIACI class", "sea_chest_grating": "other LIACI class", "anode": "other LIACI class",
          "propeller": "other LIACI class", "over_board_valve": "other LIACI class", "bilge_keel": "other LIACI class", "defect": "other LIACI class"}


def records(pattern: str, r: int) -> dict:
    out = {}
    for k in range(FOLDS):
        d = pickle.load(open(pattern.format(r=r, k=k), "rb"))
        for n, views in d["test"].items():
            b, s = views["std"]
            out[n] = (np.asarray(b, np.float32).reshape(-1, 4), np.asarray(s, np.float32), np.asarray(d["gt"][n], np.float32).reshape(-1, 4))
    return out


def main() -> None:
    conf_vt = {}
    val_csv = ROOT / "ext" / "results_final" / "validation.csv"
    if val_csv.exists():
        conf_vt = {r["model"]: float(r["conf_vt"]) for r in csv.DictReader(open(val_csv)) if r["strategy"] == "standard"}
    d = json.load(open(COCO, encoding="utf-8"))
    build, names, _ = class_masks(d)
    hull = next(k for k, v in names.items() if v == "ship_hull")
    corr = next(k for k, v in names.items() if v == "corrosion")
    cache: dict = {}
    rows = []
    for model, pattern in MODELS.items():
        for label, thr in (("0.25", 0.25), ("val", conf_vt.get(model))):
            if thr is None:
                continue
            land, n_pred, n_gt, found = Counter(), 0, 0, 0
            for r in range(REPEATS):
                for name, (boxes, sc, gt) in records(pattern, r).items():
                    b = boxes[sc >= thr]
                    n_pred += len(b)
                    n_gt += len(gt)
                    if len(gt) and len(b):
                        found += int((iou_matrix(gt, b).max(axis=1) >= 0.5).sum())
                    if not len(b):
                        continue
                    best = iou_matrix(b, gt).max(axis=1) if len(gt) else np.zeros(len(b))
                    bg = b[best < 0.1]
                    if not len(bg):
                        continue
                    if name not in cache:
                        cache[name] = build(name)
                    masks = cache[name]
                    for box in bg:
                        x1, y1, x2, y2 = (np.clip(box, 0, 1) * SIDE).round().astype(int)
                        if x2 <= x1 or y2 <= y1:
                            land["degenerate"] += 1
                            continue
                        cover = {names[c]: float(m[y1:y2, x1:x2].mean()) for c, m in masks.items() if c not in (hull, corr)}
                        top = max(cover.items(), key=lambda kv: kv[1]) if cover else ("none", 0.0)
                        if corr in masks and masks[corr][y1:y2, x1:x2].mean() >= COVER:
                            land["corrosion region (unmatched)"] += 1
                        elif top[1] >= COVER:
                            land[GROUPS.get(top[0], top[0])] += 1
                            land[f"  {top[0]}"] += 1
                        elif hull in masks and masks[hull][y1:y2, x1:x2].mean() >= 0.5:
                            land["plain hull"] += 1
                        else:
                            land["water / other"] += 1
            bg_total = sum(v for k, v in land.items() if not k.startswith("  "))
            row = {"model": model, "threshold": label, "conf": thr, "predictions": n_pred, "gt_boxes": n_gt,
                   "recall_iou50": round(found / max(n_gt, 1), 4), "background_fp": bg_total,
                   "fp_other_classes": land["other LIACI class"], "fp_corrosion_region": land["corrosion region (unmatched)"],
                   "fp_plain_hull": land["plain hull"], "fp_water_other": land["water / other"],
                   **{f"fp_{k.strip()}": v for k, v in land.items() if k.startswith("  ")}}
            rows.append(row)
            print(f"{model:13s} conf {thr:<5} | recall@0.5 {row['recall_iou50']:.3f} | predictions {n_pred:5d} | background FP {bg_total:4d} "
                  f"(other classes {row['fp_other_classes']:4d}, corrosion region {row['fp_corrosion_region']:3d}, hull {row['fp_plain_hull']:3d}, water {row['fp_water_other']:3d})")
    keys = sorted({k for r in rows for k in r}, key=lambda k: (not k in rows[0], k))
    out = ROOT / "ext" / "results_final" / "mechanism.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="\n", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
