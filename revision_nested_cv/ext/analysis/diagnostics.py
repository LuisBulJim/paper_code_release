"""Zero-GPU diagnostics on the grouped out-of-fold predictions (standard view).

AP as a function of the IoU threshold, recall ceiling, error taxonomy at a fixed
operating point and region-level (instance-agnostic) agreement.
"""
import json
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

# revision_nested_cv/ (this file lives in ext/analysis/); LIACI_ROOT must point at the raw LIACI release
# (the folder holding coco-labels.json, train_test_split.csv and embeddings_resnet101.json)
ROOT = Path(os.environ.get("NESTED_ROOT", Path(__file__).resolve().parents[2]))
COCO = Path(os.environ.get("LIACI_ROOT", ROOT.parent / "LIACI")) / "coco-labels.json"
IOUS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75]
REPEATS, FOLDS = 3, 5


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-12)


def load_repeat(tag: str, size: str, r: int, view: str = "std") -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out = {}
    for k in range(FOLDS):
        d = pickle.load(open(ROOT / "out" / "preds" / f"{tag}_r{r}_f{k}_{size}.pkl", "rb"))
        for name, views in d["test"].items():
            boxes, scores = views[view]
            out[name] = (np.asarray(boxes, dtype=np.float32).reshape(-1, 4), np.asarray(scores, dtype=np.float32), np.asarray(d["gt"][name], dtype=np.float32).reshape(-1, 4))
    return out


def ap_at(records: dict, thr: float) -> tuple[float, float]:
    """COCO-style AP (101-point) and recall ceiling at one IoU threshold."""
    scores, tps, n_gt = [], [], 0
    for boxes, sc, gt in records.values():
        n_gt += len(gt)
        order = np.argsort(-sc)
        ious = iou_matrix(boxes[order], gt)
        taken = np.zeros(len(gt), dtype=bool)
        for i in range(len(order)):
            tp = False
            if len(gt):
                cand = np.where(~taken, ious[i], -1.0)
                j = int(cand.argmax())
                if cand[j] >= thr:
                    taken[j] = True
                    tp = True
            scores.append(sc[order][i])
            tps.append(tp)
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    tp = np.asarray(tps, dtype=bool)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(~tp)
    rec = ctp / max(n_gt, 1)
    prec = ctp / np.maximum(ctp + cfp, 1)
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    grid = np.linspace(0, 1, 101)
    idx = np.searchsorted(rec, grid, side="left")
    ap = float(np.mean([prec[i] if i < len(prec) else 0.0 for i in idx]))
    return ap, float(rec[-1]) if len(rec) else 0.0


def taxonomy(records: dict, conf: float) -> dict[str, float]:
    """Where the errors are at IoU 0.5 and a fixed confidence."""
    fp = {"localisation": 0, "duplicate": 0, "background": 0}
    fn = {"localisation": 0, "low_confidence": 0, "missed": 0}
    tp = 0
    for boxes, sc, gt in records.values():
        keep = sc >= conf
        b, s = boxes[keep], sc[keep]
        order = np.argsort(-s)
        b = b[order]
        ious = iou_matrix(b, gt)
        taken = np.zeros(len(gt), dtype=bool)
        for i in range(len(b)):
            best = float(ious[i].max()) if len(gt) else 0.0
            cand = np.where(~taken, ious[i], -1.0) if len(gt) else np.array([-1.0])
            j = int(cand.argmax())
            if len(gt) and cand[j] >= 0.5:
                taken[j] = True
                tp += 1
            elif best >= 0.5:
                fp["duplicate"] += 1
            elif best >= 0.1:
                fp["localisation"] += 1
            else:
                fp["background"] += 1
        all_ious = iou_matrix(boxes, gt)
        for j in np.where(~taken)[0]:
            if len(b) and ious[:, j].max() >= 0.1:
                fn["localisation"] += 1
            elif len(boxes) and all_ious[:, j].max() >= 0.5:
                fn["low_confidence"] += 1
            else:
                fn["missed"] += 1
    return {"tp": tp, **{f"fp_{k}": v for k, v in fp.items()}, **{f"fn_{k}": v for k, v in fn.items()}}


_COCO: dict[str, tuple[int, int, list]] = {}


def coco_mask(name: str, category: int = 5) -> np.ndarray:
    """Corrosion mask rebuilt from the dataset's own COCO polygons (12 of the local .bmp masks are empty files)."""
    if not _COCO:
        d = json.load(open(COCO, encoding="utf-8"))
        by_id = {im["id"]: im for im in d["images"]}
        for im in d["images"]:
            _COCO[im["file_name"]] = (im["height"], im["width"], [])
        for a in d["annotations"]:
            if a["category_id"] == category:
                _COCO[by_id[a["image_id"]]["file_name"]][2].append(a["segmentation"])
    h, w, segs = _COCO[name]
    out = np.zeros((h, w), dtype=np.uint8)
    for seg in segs:
        rles = mask_utils.frPyObjects(seg, h, w)
        out |= mask_utils.decode(mask_utils.merge(rles) if isinstance(rles, list) else rles)
    return out


def raster(boxes: np.ndarray, h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        m[max(int(round(y1 * h)), 0):min(int(round(y2 * h)), h), max(int(round(x1 * w)), 0):min(int(round(x2 * w)), w)] = True
    return m


def region(records: dict, conf: float, side: int = 320) -> dict[str, float]:
    """Instance-agnostic agreement: union of predicted boxes against the GT mask and the GT-box union."""
    inter_b = union_b = pred_px = gt_box_px = hit_mask = mask_px = 0
    for name, (boxes, sc, gt) in records.items():
        pm = raster(boxes[sc >= conf], side, side)
        gm = raster(gt, side, side)
        inter_b += int((pm & gm).sum())
        union_b += int((pm | gm).sum())
        pred_px += int(pm.sum())
        gt_box_px += int(gm.sum())
        if len(gt):
            mask = cv2.resize(coco_mask(name), (side, side), interpolation=cv2.INTER_NEAREST) > 0
            mask_px += int(mask.sum())
            hit_mask += int((pm & mask).sum())
    return {
        "region_iou": inter_b / max(union_b, 1),
        "region_precision": inter_b / max(pred_px, 1),
        "region_recall": inter_b / max(gt_box_px, 1),
        "mask_recall": hit_mask / max(mask_px, 1),
    }


def main() -> None:
    out = {}
    for tag, size in [("direct", "n"), ("direct", "m"), ("direct", "l"), ("sequential", "m")]:
        per_iou = {t: [] for t in IOUS}
        ceil = {t: [] for t in IOUS}
        tax = []
        reg = {0.25: [], 0.10: []}
        for r in range(REPEATS):
            rec = load_repeat(tag, size, r)
            for t in IOUS:
                ap, rc = ap_at(rec, t)
                per_iou[t].append(ap)
                ceil[t].append(rc)
            if size == "m" and tag == "direct":
                tax.append(taxonomy(rec, 0.25))
                for c in reg:
                    reg[c].append(region(rec, c))
        row = {f"ap@{t}": round(float(np.mean(v)), 4) for t, v in per_iou.items()}
        row.update({f"maxrec@{t}": round(float(np.mean(ceil[t])), 3) for t in (0.1, 0.3, 0.5)})
        out[f"{tag}-{size}"] = row
        print(f"{tag}-{size}", json.dumps(row))
        if tax:
            mean_tax = {k: round(float(np.mean([t[k] for t in tax])), 1) for k in tax[0]}
            print("  taxonomy@0.25", json.dumps(mean_tax))
            out["taxonomy_direct_m_conf025"] = mean_tax
            for c, vals in reg.items():
                mean_reg = {k: round(float(np.mean([v[k] for v in vals])), 3) for k in vals[0]}
                print(f"  region@{c}", json.dumps(mean_reg))
                out[f"region_direct_m_conf{c}"] = mean_reg
    Path(sys.argv[1]).write_text(json.dumps(out, indent=2), encoding="utf-8") if len(sys.argv) > 1 else None


if __name__ == "__main__":
    main()
