"""Error analysis at the fixed operating point on out-of-fold predictions of the 261 nested-CV images.

Takes two dumps of the same models (standard and TTA, written by `nested_eval evaluate` under
r1/runs/nested/<variant>/oof/) and produces:
  results/failure_by_size.csv    TP / FN per ground-truth size bucket (COCO small/medium/large) and per
                                 aspect-ratio bucket, FP per predicted-box size bucket, for both strategies
  results/failure_cases.csv      one row per image: TP/FP/FN of each strategy, and whether TTA helped or hurt
  figures/failure_cases.png      up to 6 panels: false negatives, false positives and inflated boxes
                                 (matched predictions whose area exceeds the GT box by > 50%), GT in green,
                                 predictions in red (standard) / orange (TTA)
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from r1 import boxes as bx
from r1 import evaluate as ev
from r1.collect_runs import write_csv
from r1.protocol import dataset_images, get_logger, load_protocol, read_json, resolve

SIZE_ORDER = ("small", "medium", "large")


def aspect_bucket(box: np.ndarray) -> str:
    w, h = box[2] - box[0], box[3] - box[1]
    r = w / max(h, 1e-6)
    return "wide" if r > 1.5 else ("tall" if r < 1 / 1.5 else "square")


def analyse(preds: ev.Preds, gt: ev.GT, names: list[str], conf: float, iou: float) -> tuple[dict, list[dict]]:
    """Per-bucket counts and per-image records with the matched/unmatched boxes."""
    counts = {"gt_found": Counter(), "gt_missed": Counter(), "fp_size": Counter(), "asp_found": Counter(), "asp_missed": Counter(), "inflated": 0, "matched": 0}
    records = []
    for name in names:
        p = preds.get(name, ev.EMPTY)
        g = gt[name]
        m = ev.match_details(p["boxes"], p["scores"], g["boxes"], conf, iou)
        gt_areas = bx.box_area(g["boxes"]) if len(g["boxes"]) else np.zeros(0)
        for j, found in enumerate(m["gt_found"]):
            counts["gt_found" if found else "gt_missed"][bx.size_bucket(gt_areas[j])] += 1
            counts["asp_found" if found else "asp_missed"][aspect_bucket(g["boxes"][j])] += 1
        fp_boxes, inflated = [], []
        for i, j in zip(m["pred_idx"], m["pred_match"]):
            if j < 0:
                counts["fp_size"][bx.size_bucket(float(bx.box_area(p["boxes"][i : i + 1])[0]))] += 1
                fp_boxes.append(p["boxes"][i].tolist())
            else:
                counts["matched"] += 1
                if bx.box_area(p["boxes"][i : i + 1])[0] > 1.5 * gt_areas[j]:
                    counts["inflated"] += 1
                    inflated.append((p["boxes"][i].tolist(), g["boxes"][j].tolist()))
        tp = int((m["pred_match"] >= 0).sum())
        records.append({"image": name, "tp": tp, "fp": len(m["pred_idx"]) - tp, "fn": int(len(g["boxes"]) - tp), "fp_boxes": fp_boxes, "fn_boxes": [g["boxes"][j].tolist() for j in np.where(~m["gt_found"])[0]], "inflated": inflated})
    return counts, records


def size_rows(strategy: str, counts: dict) -> list[dict]:
    rows = []
    for b in SIZE_ORDER:
        found, missed = counts["gt_found"][b], counts["gt_missed"][b]
        rows.append({"strategy": strategy, "bucket_type": "gt_size", "bucket": b, "n_gt": found + missed, "tp": found, "fn": missed, "recall": round(found / (found + missed), 4) if found + missed else float("nan"), "fp": counts["fp_size"][b]})
    for b in ("wide", "square", "tall"):
        found, missed = counts["asp_found"][b], counts["asp_missed"][b]
        rows.append({"strategy": strategy, "bucket_type": "gt_aspect", "bucket": b, "n_gt": found + missed, "tp": found, "fn": missed, "recall": round(found / (found + missed), 4) if found + missed else float("nan"), "fp": ""})
    rows.append({"strategy": strategy, "bucket_type": "matched", "bucket": "inflated_area_gt_1.5x", "n_gt": counts["matched"], "tp": counts["inflated"], "fn": "", "recall": "", "fp": ""})
    return rows


def draw_panel(image: np.ndarray, gt_boxes: np.ndarray, std_boxes: list, tta_boxes: list, title: str) -> np.ndarray:
    im = image.copy()
    for b in gt_boxes:
        cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 200, 0), 2)
    for b in std_boxes:
        cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
    for b in tta_boxes:
        cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 140, 255), 2)
    cv2.rectangle(im, (0, 0), (im.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(im, title, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def make_figure(image_paths: dict[str, Path], gt: ev.GT, preds_std: ev.Preds, preds_tta: ev.Preds, rec_std: list[dict], rec_tta: list[dict], conf: float, out: Path) -> list[str]:
    """Six panels: 2 FN, 2 FP, 2 inflated (taken from the standard strategy; TTA boxes overlaid)."""
    by_name = {r["image"]: r for r in rec_std}
    picks = []
    for kind, key in (("FN", "fn_boxes"), ("FP", "fp_boxes"), ("inflated", "inflated")):
        cands = sorted((r for r in rec_std if r[key] and r["image"] not in {p[1] for p in picks}), key=lambda r: -len(r[key]))
        picks += [(kind, r["image"]) for r in cands[:2]]
    panels = []
    for kind, name in picks:
        im = cv2.imread(str(image_paths[name]))
        std_b = preds_std[name]["boxes"][preds_std[name]["scores"] >= conf]
        tta_b = preds_tta[name]["boxes"][preds_tta[name]["scores"] >= conf] if name in preds_tta else []
        r = by_name[name]
        panels.append(draw_panel(im, gt[name]["boxes"], std_b.tolist(), list(tta_b), f"{kind} | {name} | std TP{r['tp']} FP{r['fp']} FN{r['fn']}"))
    if not panels:
        return []
    h, w = panels[0].shape[:2]
    panels = [cv2.resize(p, (w, h)) for p in panels]
    while len(panels) % 3:
        panels.append(np.zeros_like(panels[0]))
    rows = [np.concatenate(panels[i : i + 3], axis=1) for i in range(0, len(panels), 3)]
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.concatenate(rows, axis=0))
    return [f"{k}:{n}" for k, n in picks]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--standard", required=True, help="standard out-of-fold dump directory")
    ap.add_argument("--tta", required=True, help="TTA out-of-fold dump directory of the same models")
    ap.add_argument("--images", default=None, help="JSON list restricting the images (default: the 261 images)")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--out", default="r1/results")
    ap.add_argument("--figure", default="r1/figures/failure_cases.png")
    args = ap.parse_args()
    proto = load_protocol()
    log = get_logger("failure_analysis")
    evc = proto["evaluation"]
    conf = args.conf if args.conf is not None else evc["conf_fixed"]
    from r1.nested_eval import load_gt

    paths = dataset_images(proto)
    names = sorted(read_json(Path(args.images))["images"] if args.images else paths)
    gt = load_gt(proto, names)
    preds_std = ev.load_predictions(resolve(args.standard), names)
    preds_tta = ev.load_predictions(resolve(args.tta), names)
    c_std, r_std = analyse(preds_std, gt, names, conf, evc["iou_match"])
    c_tta, r_tta = analyse(preds_tta, gt, names, conf, evc["iou_match"])

    out = resolve(args.out)
    write_csv(out / "failure_by_size.csv", size_rows("standard", c_std) + size_rows("tta", c_tta))
    cases = []
    for a, b in zip(r_std, r_tta):
        err_std, err_tta = a["fp"] + a["fn"], b["fp"] + b["fn"]
        cases.append({"image": a["image"], "n_gt": a["tp"] + a["fn"], "std_tp": a["tp"], "std_fp": a["fp"], "std_fn": a["fn"], "tta_tp": b["tp"], "tta_fp": b["fp"], "tta_fn": b["fn"], "tta_effect": "helps" if err_tta < err_std else ("hurts" if err_tta > err_std else "same")})
    write_csv(out / "failure_cases.csv", cases)
    picked = make_figure(paths, gt, preds_std, preds_tta, r_std, r_tta, conf, resolve(args.figure))
    effect = Counter(c["tta_effect"] for c in cases)
    log.info(f"standard: matched {c_std['matched']} inflated {c_std['inflated']} FN by size {dict(c_std['gt_missed'])} | tta: matched {c_tta['matched']} inflated {c_tta['inflated']} FN by size {dict(c_tta['gt_missed'])}")
    log.info(f"TTA effect per image: {dict(effect)} | figure panels: {picked}")


if __name__ == "__main__":
    main()
