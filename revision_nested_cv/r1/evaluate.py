"""Single evaluator used for every table of the revision.

Inputs are prediction dumps (one JSON per image, pixel xyxy boxes + scores at conf >= 0.001) and YOLO
ground-truth labels. Metrics:
  * mAP@50 and mAP@[.5:.95] with pycocotools (COCO 101-point interpolation, maxDets=100)
  * Precision / Recall / F1 at the fixed operating point (conf_fixed, IoU 0.5, greedy matching)
  * 95% bootstrap confidence intervals over images, and paired bootstrap of the difference between two runs
Nothing here selects thresholds; conf_fixed comes from protocol.yaml.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
from typing import Callable

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from r1 import boxes as bx
from r1.protocol import get_logger, load_protocol, read_json, resolve, write_json

Preds = dict[str, dict]  # image name -> {"boxes": (n,4) xyxy px, "scores": (n,), "classes": (n,) int}
GT = dict[str, dict]     # image name -> {"boxes": (m,4) xyxy px, "classes": (m,) int, "width": int, "height": int}

EMPTY = {"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "classes": np.zeros(0, dtype=int)}


def _classes(d: dict, n: int) -> np.ndarray:
    """Class ids of a prediction/GT record; single-class dumps (LIACI) carry none and default to 0."""
    c = d.get("classes")
    return np.asarray(c, dtype=int) if c is not None and len(c) == n else np.zeros(n, dtype=int)


# ----------------------------------------------------------------------------- loading

def load_ground_truth(dataset_root: Path, split: str, image_names: list[str] | None = None, sizes: dict[str, tuple[int, int]] | None = None) -> GT:
    """GT for every image of <dataset_root>/images/<split> (or the given names). Sizes read from the image when not given."""
    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels" / split
    names = image_names or sorted(p.name for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
    gt: GT = {}
    for name in names:
        if sizes and name in sizes:
            w, h = sizes[name]
        else:
            from PIL import Image

            with Image.open(images_dir / name) as im:
                w, h = im.size
        boxes, classes = bx.read_yolo_labels(labels_dir / f"{Path(name).stem}.txt", w, h, with_classes=True)
        gt[name] = {"boxes": boxes, "classes": classes, "width": w, "height": h}
    return gt


def load_predictions(dump_dir: Path, image_names: list[str] | None = None) -> Preds:
    preds: Preds = {}
    files = sorted(p for p in dump_dir.glob("*.json") if not p.name.startswith("_"))
    for p in files:
        d = read_json(p)
        if image_names is not None and d["image"] not in image_names:
            continue
        boxes = np.asarray(d["boxes"], dtype=float).reshape(-1, 4)
        preds[d["image"]] = {"boxes": boxes, "scores": np.asarray(d["scores"], dtype=float), "classes": _classes(d, len(boxes))}
    if image_names is not None:
        missing = [n for n in image_names if n not in preds]
        if missing:
            raise FileNotFoundError(f"{len(missing)} images without prediction dump in {dump_dir}: {missing[:5]}")
    return preds


def filter_class(records: dict, class_id: int) -> dict:
    """Keep only the boxes of one class (used for per-class metrics and for single-class OOD tests)."""
    out = {}
    for name, r in records.items():
        keep = _classes(r, len(r["boxes"])) == class_id
        out[name] = {**r, "boxes": r["boxes"][keep], "classes": np.zeros(int(keep.sum()), dtype=int)}
        if "scores" in r:
            out[name]["scores"] = r["scores"][keep]
    return out


# ----------------------------------------------------------------------------- COCO mAP

def _coco_structs(preds: Preds, gt: GT, names: list[str]) -> tuple[dict, list[dict]]:
    """Build COCO gt dict and detection list; repeated names (bootstrap) get distinct image ids.
    COCO category id = class id + 1; the category set is the union of GT and predicted classes."""
    images, anns, dets = [], [], []
    ann_id = 1
    cats = set()
    for img_id, name in enumerate(names, start=1):
        g = gt[name]
        images.append({"id": img_id, "width": int(g["width"]), "height": int(g["height"]), "file_name": name})
        for (x1, y1, x2, y2), c in zip(g["boxes"], _classes(g, len(g["boxes"]))):
            w, h = float(x2 - x1), float(y2 - y1)
            anns.append({"id": ann_id, "image_id": img_id, "category_id": int(c) + 1, "bbox": [float(x1), float(y1), w, h], "area": w * h, "iscrowd": 0})
            ann_id += 1
            cats.add(int(c) + 1)
        p = preds.get(name, EMPTY)
        for (x1, y1, x2, y2), s, c in zip(p["boxes"], p["scores"], _classes(p, len(p["boxes"]))):
            dets.append({"image_id": img_id, "category_id": int(c) + 1, "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(s)})
            cats.add(int(c) + 1)
    coco_gt = {"images": images, "annotations": anns, "categories": [{"id": c, "name": str(c - 1)} for c in sorted(cats)]}
    return coco_gt, dets


def coco_map(preds: Preds, gt: GT, names: list[str]) -> dict[str, float]:
    coco_gt_dict, dets = _coco_structs(preds, gt, names)
    if not coco_gt_dict["annotations"]:
        raise ValueError("no ground-truth boxes in the evaluated images")
    if not dets:
        return {"map50": 0.0, "map5095": 0.0, "ar100": 0.0}
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = coco_gt_dict
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(dets)
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return {"map50": float(ev.stats[1]), "map5095": float(ev.stats[0]), "ar100": float(ev.stats[8])}


# ----------------------------------------------------------------------------- fixed operating point

def match_details(pred_boxes: np.ndarray, pred_scores: np.ndarray, gt_boxes: np.ndarray, conf: float, iou_thr: float, pred_classes: np.ndarray | None = None, gt_classes: np.ndarray | None = None) -> dict:
    """Greedy score-ordered matching at one threshold, class-aware when classes are given.
    Returns the kept prediction indices (score order), the GT index matched by each kept prediction
    (-1 = false positive) and, per GT box, whether it was found."""
    keep_idx = np.where(pred_scores >= conf)[0]
    keep_idx = keep_idx[np.argsort(-pred_scores[keep_idx], kind="stable")]
    boxes = pred_boxes[keep_idx]
    matched = np.zeros(len(gt_boxes), dtype=bool)
    assigned = np.full(len(keep_idx), -1, dtype=int)
    if len(boxes) and len(gt_boxes):
        ious = bx.iou_matrix(boxes, gt_boxes)
        if pred_classes is not None and gt_classes is not None:
            ious = np.where(pred_classes[keep_idx][:, None] == gt_classes[None, :], ious, 0.0)
        for i in range(len(boxes)):
            cand = np.where(~matched)[0]
            if len(cand) == 0:
                break
            j = cand[np.argmax(ious[i, cand])]
            if ious[i, j] >= iou_thr:
                matched[j] = True
                assigned[i] = j
    return {"pred_idx": keep_idx, "pred_match": assigned, "gt_found": matched}


def match_image(pred_boxes: np.ndarray, pred_scores: np.ndarray, gt_boxes: np.ndarray, conf: float, iou_thr: float, pred_classes: np.ndarray | None = None, gt_classes: np.ndarray | None = None) -> tuple[int, int, int]:
    """(tp, fp, fn) of one image at one threshold."""
    m = match_details(pred_boxes, pred_scores, gt_boxes, conf, iou_thr, pred_classes, gt_classes)
    tp = int((m["pred_match"] >= 0).sum())
    return tp, len(m["pred_idx"]) - tp, len(gt_boxes) - tp


def fixed_point_metrics(preds: Preds, gt: GT, names: list[str], conf: float, iou_thr: float) -> dict[str, float]:
    tp = fp = fn = 0
    for name in names:
        p = preds.get(name, EMPTY)
        g = gt[name]
        t, f, n = match_image(p["boxes"], p["scores"], g["boxes"], conf, iou_thr, _classes(p, len(p["boxes"])), _classes(g, len(g["boxes"])))
        tp, fp, fn = tp + t, fp + f, fn + n
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "conf": conf, "iou": iou_thr}


def all_metrics(preds: Preds, gt: GT, names: list[str], conf: float, iou_thr: float, class_names: dict[int, str] | None = None) -> dict:
    """Global metrics; with class_names (multi-class datasets) also one block per class."""
    out = coco_map(preds, gt, names)
    out.update(fixed_point_metrics(preds, gt, names, conf, iou_thr))
    out["n_images"] = len(names)
    out["n_gt_boxes"] = int(sum(len(gt[n]["boxes"]) for n in names))
    if class_names:
        out["classes"] = {}
        for cid, cname in class_names.items():
            g_c, p_c = filter_class(gt, cid), filter_class(preds, cid)
            n_gt = int(sum(len(g_c[n]["boxes"]) for n in names))
            m = coco_map(p_c, g_c, names) if n_gt else {"map50": float("nan"), "map5095": float("nan"), "ar100": float("nan")}
            m.update(fixed_point_metrics(p_c, g_c, names, conf, iou_thr))
            out["classes"][cname] = {**m, "n_gt_boxes": n_gt}
    return out


# ----------------------------------------------------------------------------- bootstrap

def bootstrap_samples(names: list[str], gt: GT, iters: int, seed: int) -> list[list[str]]:
    """Stratified resampling: positives and negatives are resampled separately so every replicate keeps
    the same number of images with and without ground-truth boxes (a plain resample of 33 images can
    otherwise produce replicates without any box, on which mAP is undefined)."""
    rng = np.random.default_rng(seed)
    pos = [n for n in names if len(gt[n]["boxes"]) > 0]
    neg = [n for n in names if len(gt[n]["boxes"]) == 0]
    samples = []
    for _ in range(iters):
        rep = [pos[i] for i in rng.integers(0, len(pos), size=len(pos))]
        if neg:
            rep += [neg[i] for i in rng.integers(0, len(neg), size=len(neg))]
        samples.append(rep)
    return samples


def bootstrap_ci(metric_fn: Callable[[list[str]], dict[str, float]], names: list[str], gt: GT, iters: int, seed: int, keys=("map50", "f1", "recall", "precision")) -> dict[str, dict]:
    samples = {k: [] for k in keys}
    for rep in bootstrap_samples(names, gt, iters, seed):
        m = metric_fn(rep)
        for k in keys:
            samples[k].append(m[k])
    return {k: {"mean": float(np.mean(v)), "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]} for k, v in samples.items()}


def paired_bootstrap(preds_a: Preds, preds_b: Preds, gt: GT, names: list[str], conf: float, iou_thr: float, iters: int, seed: int, keys=("map50", "f1", "recall", "precision")) -> dict[str, dict]:
    """Bootstrap of (b - a) with the same resampled images for both runs. p_le0 = fraction of resamples with delta <= 0."""
    deltas = {k: [] for k in keys}
    for sub in bootstrap_samples(names, gt, iters, seed):
        ma = all_metrics(preds_a, gt, sub, conf, iou_thr)
        mb = all_metrics(preds_b, gt, sub, conf, iou_thr)
        for k in keys:
            deltas[k].append(mb[k] - ma[k])
    return {
        k: {"delta_mean": float(np.mean(v)), "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], "p_le0": float(np.mean(np.asarray(v) <= 0))}
        for k, v in deltas.items()
    }


# ----------------------------------------------------------------------------- CLI

def evaluate_dump(dump_dir: Path, dataset_root: Path, split: str, proto: dict, image_names: list[str] | None = None, bootstrap: bool = False, compare: Path | None = None, class_names: dict[int, str] | None = None) -> dict:
    ev = proto["evaluation"]
    gt = load_ground_truth(dataset_root, split, image_names)
    names = sorted(gt)
    preds = load_predictions(dump_dir, names)
    result = {"dump": str(dump_dir), "split": split, "metrics": all_metrics(preds, gt, names, ev["conf_fixed"], ev["iou_match"], class_names)}
    if bootstrap:
        result["bootstrap"] = bootstrap_ci(lambda sub: all_metrics(preds, gt, sub, ev["conf_fixed"], ev["iou_match"]), names, gt, ev["bootstrap_iters"], ev["bootstrap_seed"])
    if compare is not None:
        other = load_predictions(compare, names)
        result["compare"] = {
            "reference": str(compare),
            "reference_metrics": all_metrics(other, gt, names, ev["conf_fixed"], ev["iou_match"]),
            "delta_this_minus_reference": paired_bootstrap(other, preds, gt, names, ev["conf_fixed"], ev["iou_match"], ev["bootstrap_iters"], ev["bootstrap_seed"]),
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", required=True, help="directory with one JSON per image")
    ap.add_argument("--dataset", required=True, help="dataset root with images/<split> and labels/<split>")
    ap.add_argument("--split", default="test")
    ap.add_argument("--images", default=None, help="JSON file with a list of image names to restrict the evaluation")
    ap.add_argument("--out", default=None, help="write the result JSON here")
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--compare", default=None, help="reference dump directory for a paired bootstrap")
    args = ap.parse_args()
    proto = load_protocol()
    log = get_logger("evaluate")
    dataset_root = resolve(args.dataset)
    names = read_json(Path(args.images)) if args.images else None
    if isinstance(names, dict):
        names = names.get("images")
    res = evaluate_dump(Path(args.preds), dataset_root, args.split, proto, names, args.bootstrap, Path(args.compare) if args.compare else None)
    log.info(json.dumps(res["metrics"]))
    if args.out:
        write_json(resolve(args.out), res)
        log.info(f"written {args.out}")


if __name__ == "__main__":
    main()
