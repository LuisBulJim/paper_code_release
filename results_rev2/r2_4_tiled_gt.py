"""R2.4 (optional) -- 0.534 is the IoU of one filled box per annotated connected component, not a ceiling of box output.

Ground truth and pooling exactly as ext/pixel_eval.py: union of the corrosion polygons of ext/data/labels_seg rasterised
at image size, pixels pooled over the 261 images, IoU = TP / (TP + FP + FN). First the paper's reference row is
reproduced (filled ground-truth boxes, 0.534); then the ground-truth mask is covered by axis-aligned square cells of
side s pixels (grid-aligned; a cell is kept if it contains any ground-truth pixel), i.e. a set of small filled boxes.
As s shrinks, the IoU of this box output tends to 1, so a box-based output is not bounded by 0.534; the number of
boxes it needs grows accordingly.

Usage (repository root): python results_rev2/r2_4_tiled_gt.py
Writes results_rev2/r2_4/tiled_gt.csv and tiled_gt.json.
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
NESTED = REPO / "revision_nested_cv"
sys.path.insert(0, str(NESTED))
sys.path.insert(0, str(NESTED / "ext"))
import os  # noqa: E402

os.chdir(NESTED)
from pixel_eval import box_fill, counts, gt_mask  # noqa: E402
from r1.nested_eval import image_sizes  # noqa: E402
from r1.protocol import load_protocol  # noqa: E402

OUT = REPO / "results_rev2" / "r2_4"
CELLS = [128, 64, 32, 16, 8]


def iou(tot: np.ndarray) -> float:
    """pixel_eval.counts gives (intersection, predicted pixels, ground-truth pixels)."""
    inter, pred, gt = tot
    return float(inter / (pred + gt - inter))


def main() -> None:
    proto = load_protocol()
    gt_boxes = {}
    for k in range(5):
        gt_boxes.update(pickle.load(open(f"out/preds/direct_r0_f{k}_m.pkl", "rb"))["gt"])
    names = sorted(gt_boxes)
    sizes = {n: (wh[1], wh[0]) for n, wh in image_sizes(proto, names).items()}  # (h, w) as pixel_eval
    data = NESTED / "ext" / "data"
    gts = {n: gt_mask(n, *sizes[n], data) for n in names}

    ref = np.sum([counts(box_fill(gt_boxes[n], *sizes[n]), gts[n]) for n in names], axis=0)
    rows = [{"output": "one filled box per annotated component (paper reference)", "cell_px": "", "iou": round(iou(ref), 4),
             "boxes_total": int(sum(len(np.asarray(gt_boxes[n]).reshape(-1, 4)) for n in names))}]
    for s in CELLS:
        tot, n_boxes = np.zeros(3), 0
        for n in names:
            g = gts[n]
            h, w = g.shape
            pred = np.zeros_like(g)
            for y in range(0, h, s):
                for x in range(0, w, s):
                    if g[y:y + s, x:x + s].any():
                        pred[y:y + s, x:x + s] = 1
                        n_boxes += 1
            tot += counts(pred, g)
        rows.append({"output": f"grid cells of {s} px covering the ground truth", "cell_px": s, "iou": round(iou(tot), 4), "boxes_total": n_boxes})
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "tiled_gt.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (OUT / "tiled_gt.json").write_text(json.dumps({"images": len(names), "image_px": "native image size (as pixel_eval)", "rows": rows}, indent=1), encoding="utf-8")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
