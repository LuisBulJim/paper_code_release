"""SAM 3 box-prompted segmentation on the LIACI evaluation set, in two stages.

  predict   (GPU server) detector checkpoint -> boxes at conf_fixed -> SAM 3 (facebook/sam3, transformers)
            --checkpoint-map (written by `nested_eval sam3-map`) picks the detector per image, so an image that
            belongs to the nested-CV data is detected by the fold model that had it in its outer test fold
            prompted with those boxes -> one binary mask per image saved as PNG under <out>/masks, plus the
            boxes and timing in <out>/boxes/<image>.json and <out>/_meta.json. --mock-sam fills the boxes
            instead of calling SAM 3 so the whole pipeline can be checked on a CPU machine.
  metrics   (anywhere) predicted masks vs ground-truth masks:
            positives: IoU and Dice per image (0 when nothing is predicted), share with IoU > 0.5,
                       mean +- sd and 95% bootstrap CI, IoU against mask area
            negatives: false-mask rate (any predicted pixel) and mean false area fraction
            Outputs r1/results/sam3_positives.csv, sam3_negatives.csv, sam3_summary.csv and
            r1/figures/sam3_cases.png (2 best + 2 worst positives).
An image without corrosion and without predicted mask is a correct rejection, never an IoU of 1.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from r1 import predict_dump as pdump
from r1.collect_runs import write_csv
from r1.protocol import environment_report, get_logger, load_protocol, read_json, resolve, sha256_file, write_json


# ----------------------------------------------------------------------------- predict

class Sam3Segmenter:
    """Thin wrapper around transformers' Sam3Model so the rest of the script is testable without it."""

    def __init__(self, model_id: str, device: str, threshold: float):
        import torch
        from transformers import Sam3Model, Sam3Processor

        self.torch = torch
        self.device = device
        self.threshold = threshold
        self.model = Sam3Model.from_pretrained(model_id).to(device).eval()
        self.processor = Sam3Processor.from_pretrained(model_id)

    def __call__(self, image_rgb: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        h, w = image_rgb.shape[:2]
        if len(boxes) == 0:
            return np.zeros((h, w), dtype=bool)
        inputs = self.processor(images=Image.fromarray(image_rgb), input_boxes=[boxes.tolist()], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        if hasattr(self.processor, "post_process_instance_segmentation"):
            res = self.processor.post_process_instance_segmentation(outputs, threshold=self.threshold, target_sizes=inputs.get("original_sizes").tolist())[0]
            masks = res.get("masks", [])
            masks = masks.cpu().numpy() if self.torch.is_tensor(masks) else np.asarray(masks)
        else:
            pm = outputs.pred_masks
            if pm.dim() == 5:
                pm = pm.squeeze(0)
            masks = (self.torch.nn.functional.interpolate(pm.float(), (h, w), mode="bilinear") > 0.0).cpu().numpy()
            if masks.ndim == 4:
                scores = outputs.iou_scores.squeeze(0) if outputs.iou_scores.dim() == 3 else outputs.iou_scores
                masks = np.stack([masks[i, int(scores[i].argmax())] for i in range(len(masks))])
        if len(masks) == 0:
            return np.zeros((h, w), dtype=bool)
        return np.logical_or.reduce(np.asarray(masks).reshape(-1, h, w).astype(bool))


def mock_segmenter(image_rgb: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    mask = np.zeros(image_rgb.shape[:2], dtype=bool)
    for x1, y1, x2, y2 in boxes.astype(int):
        mask[max(y1, 0) : y2, max(x1, 0) : x2] = True
    return mask


def predict(args, proto: dict, log) -> None:
    from ultralytics import YOLO

    cfg = proto["sam3"]
    data_root = resolve(cfg["output_root"])
    manifest = read_json(resolve(cfg["manifest_file"]))
    names = [e["image"] for e in manifest["images"]]
    if args.limit:  # pipeline checks: a balanced subset, N positives + N negatives
        names = [e["image"] for e in manifest["images"] if e["positive"]][: args.limit] + [e["image"] for e in manifest["images"] if not e["positive"]][: args.limit]
    out = resolve(args.out)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    (out / "boxes").mkdir(parents=True, exist_ok=True)
    device = args.device
    if device is None:
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.checkpoint_map:
        cmap = read_json(resolve(args.checkpoint_map))
        missing = [n for n in names if n not in cmap["images"]]
        if missing:
            raise SystemExit(f"{len(missing)} images without checkpoint in {args.checkpoint_map}: {missing[:5]}")
        ckpt_of = {n: cmap["images"][n] for n in names}
    else:
        ckpt_of = {n: args.checkpoint for n in names}
    detectors = {c: YOLO(str(resolve(c))) for c in sorted(set(ckpt_of.values()))}
    log.info(f"{len(names)} images, {len(detectors)} detector checkpoint(s)")
    segment = mock_segmenter if args.mock_sam else Sam3Segmenter(cfg["model_id"], device, cfg["mask_threshold"])
    conf = args.conf if args.conf is not None else proto["evaluation"]["conf_fixed"]
    tta_config = None
    if args.tta:
        from r1.run_test_inference import chosen_tta

        tta_config, _, local = chosen_tta(proto, args.choice, args.tta_config, None)
        inf = local["inference"]
    else:
        inf = proto["inference"]
    conf_dump = proto["evaluation"]["conf_dump"]
    t_det, t_seg = [], []
    for i, name in enumerate(names, 1):
        bgr = cv2.imread(str(data_root / "images" / name))
        t0 = time.perf_counter()
        # same detections as the evaluated dumps: predict at conf_dump, fuse, then keep scores >= conf
        if args.tta:
            boxes, scores, _ = pdump.predict_tta(detectors[ckpt_of[name]], bgr, inf["imgsz"], conf_dump, device, inf["tta"])
        else:
            boxes, scores, _ = pdump.predict_standard(detectors[ckpt_of[name]], bgr, inf["imgsz"], conf_dump, device)
        keep = scores >= conf
        boxes, scores = boxes[keep], scores[keep]
        t1 = time.perf_counter()
        mask = segment(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), boxes)
        t_det.append(t1 - t0)
        t_seg.append(time.perf_counter() - t1)
        Image.fromarray((mask.astype(np.uint8) * 255)).save(out / "masks" / f"{Path(name).stem}.png")
        write_json(out / "boxes" / f"{Path(name).stem}.json", {"image": name, "checkpoint": ckpt_of[name], "boxes": np.round(boxes, 2).tolist(), "scores": np.round(scores, 5).tolist()})
        if i % 25 == 0 or i == len(names):
            log.info(f"  {i}/{len(names)} images")
    write_json(out / "_meta.json", {"checkpoint": args.checkpoint_map or args.checkpoint, "checkpoint_sha256": {c: sha256_file(resolve(c)) for c in detectors}, "checkpoint_map": args.checkpoint_map, "conf": conf, "tta": bool(args.tta), "tta_config": tta_config, "tta_params": inf["tta"] if args.tta else None, "sam_model": "mock" if args.mock_sam else cfg["model_id"], "mask_threshold": cfg["mask_threshold"], "n_images": len(names), "device": device, "mean_ms_detector": float(np.mean(t_det) * 1000), "mean_ms_sam": float(np.mean(t_seg) * 1000), "environment": environment_report()})
    log.info(f"predictions written to {out}")


# ----------------------------------------------------------------------------- metrics

def pixel_metrics(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = inter / union if union else 0.0
    dice = 2 * inter / (pred.sum() + gt.sum()) if pred.sum() + gt.sum() else 0.0
    return float(iou), float(dice)


def bootstrap_mean(values: np.ndarray, iters: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = [values[rng.integers(0, len(values), size=len(values))].mean() for _ in range(iters)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def metrics(args, proto: dict, log) -> None:
    cfg, evc = proto["sam3"], proto["evaluation"]
    data_root = resolve(cfg["output_root"])
    manifest = read_json(resolve(cfg["manifest_file"]))
    pred_root = resolve(args.preds)
    meta = read_json(pred_root / "_meta.json")
    positives, negatives = [], []
    for e in manifest["images"]:
        stem = Path(e["image"]).stem
        pred_path = pred_root / "masks" / f"{stem}.png"
        if not pred_path.exists():
            if args.allow_missing:
                continue
            raise FileNotFoundError(pred_path)
        pred = np.asarray(Image.open(pred_path).convert("L")) > 127
        gt = np.asarray(Image.open(data_root / "masks" / f"{stem}.png").convert("L")) > 127
        n_boxes = len(read_json(pred_root / "boxes" / f"{stem}.json")["boxes"])
        if e["positive"]:
            iou, dice = pixel_metrics(pred, gt)
            positives.append({"image": e["image"], "iou": round(iou, 4), "dice": round(dice, 4), "detected": n_boxes > 0, "n_boxes": n_boxes, "gt_area_fraction": round(float(gt.mean()), 5), "pred_area_fraction": round(float(pred.mean()), 5), "gt_components": e["n_components"], "in_frozen_detection_test": e["in_frozen_detection_test"]})
        else:
            negatives.append({"image": e["image"], "false_mask": bool(pred.any()), "n_boxes": n_boxes, "false_area_fraction": round(float(pred.mean()), 5)})
    if not positives:
        raise SystemExit("no positive images evaluated")
    iou = np.array([p["iou"] for p in positives])
    dice = np.array([p["dice"] for p in positives])
    detected = np.array([p["detected"] for p in positives])
    iou_lo, iou_hi = bootstrap_mean(iou, evc["bootstrap_iters"], evc["bootstrap_seed"])
    dice_lo, dice_hi = bootstrap_mean(dice, evc["bootstrap_iters"], evc["bootstrap_seed"])
    summary = [
        {"subset": "positives", "n": len(positives), "metric": "iou_mean", "value": round(float(iou.mean()), 4), "sd": round(float(iou.std(ddof=1)), 4) if len(iou) > 1 else float("nan"), "ci_lo": round(iou_lo, 4), "ci_hi": round(iou_hi, 4)},
        {"subset": "positives", "n": len(positives), "metric": "dice_mean", "value": round(float(dice.mean()), 4), "sd": round(float(dice.std(ddof=1)), 4) if len(dice) > 1 else float("nan"), "ci_lo": round(dice_lo, 4), "ci_hi": round(dice_hi, 4)},
        {"subset": "positives", "n": len(positives), "metric": "iou_median", "value": round(float(np.median(iou)), 4), "sd": "", "ci_lo": "", "ci_hi": ""},
        {"subset": "positives", "n": len(positives), "metric": "share_iou_gt_0.5", "value": round(float((iou > 0.5).mean()), 4), "sd": "", "ci_lo": "", "ci_hi": ""},
        {"subset": "positives", "n": len(positives), "metric": "share_detected", "value": round(float(detected.mean()), 4), "sd": "", "ci_lo": "", "ci_hi": ""},
        {"subset": "positives_detected", "n": int(detected.sum()), "metric": "iou_mean", "value": round(float(iou[detected].mean()), 4) if detected.any() else float("nan"), "sd": "", "ci_lo": "", "ci_hi": ""},
        {"subset": "negatives", "n": len(negatives), "metric": "false_mask_rate", "value": round(float(np.mean([n["false_mask"] for n in negatives])), 4) if negatives else float("nan"), "sd": "", "ci_lo": "", "ci_hi": ""},
        {"subset": "negatives", "n": len(negatives), "metric": "mean_false_area_fraction", "value": round(float(np.mean([n["false_area_fraction"] for n in negatives])), 5) if negatives else float("nan"), "sd": "", "ci_lo": "", "ci_hi": ""},
    ]
    if len(iou) > 2:
        from scipy import stats

        rho, pval = stats.spearmanr(iou, [p["gt_area_fraction"] for p in positives])
        summary.append({"subset": "positives", "n": len(positives), "metric": "spearman_iou_vs_gt_area", "value": round(float(rho), 4), "sd": "", "ci_lo": "", "ci_hi": ""})
        summary.append({"subset": "positives", "n": len(positives), "metric": "spearman_p_value", "value": round(float(pval), 4), "sd": "", "ci_lo": "", "ci_hi": ""})
    out = resolve(args.out)
    write_csv(out / "sam3_positives.csv", positives)
    write_csv(out / "sam3_negatives.csv", negatives)
    write_csv(out / "sam3_summary.csv", [{**s, "sam_model": meta["sam_model"], "detector": Path(meta["checkpoint"]).name, "tta": meta["tta"]} for s in summary])
    make_figure(data_root, pred_root, positives, resolve(args.figure))
    log.info(f"positives {len(positives)}: IoU {iou.mean():.3f} [{iou_lo:.3f}, {iou_hi:.3f}] Dice {dice.mean():.3f} | detected {detected.mean():.1%} | negatives {len(negatives)}: false-mask rate {summary[6]['value']}")


def make_figure(data_root: Path, pred_root: Path, positives: list[dict], out: Path) -> None:
    ordered = sorted(positives, key=lambda p: p["iou"])
    picks = ordered[-2:][::-1] + ordered[:2] if len(ordered) >= 4 else ordered
    panels = []
    for p in picks:
        stem = Path(p["image"]).stem
        img = cv2.imread(str(data_root / "images" / p["image"]))
        gt = np.asarray(Image.open(data_root / "masks" / f"{stem}.png").convert("L")) > 127
        pred = np.asarray(Image.open(pred_root / "masks" / f"{stem}.png").convert("L")) > 127
        over = img.copy()
        over[gt] = (0.5 * over[gt] + 0.5 * np.array([0, 200, 0])).astype(np.uint8)
        over[pred] = (0.5 * over[pred] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
        for x1, y1, x2, y2 in read_json(pred_root / "boxes" / f"{stem}.json")["boxes"]:
            cv2.rectangle(over, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
        cv2.rectangle(over, (0, 0), (over.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(over, f"{stem} IoU {p['iou']:.2f} Dice {p['dice']:.2f}", (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(over)
    if panels:
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), np.concatenate(panels, axis=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["predict", "metrics"])
    ap.add_argument("--checkpoint", default=None, help="predict: detector checkpoint")
    ap.add_argument("--checkpoint-map", default=None, help="predict: JSON {'images': {name: checkpoint}} written by nested_eval sam3-map")
    ap.add_argument("--tta", action="store_true", help="predict: use the TTA chosen on validation for the detector")
    ap.add_argument("--choice", default=None, help="predict: validation choice file (default results/inference_choice.json)")
    ap.add_argument("--tta-config", default=None, help="predict: override the chosen TTA configuration")
    ap.add_argument("--conf", type=float, default=None, help="predict: detector threshold (default conf_fixed)")
    ap.add_argument("--mock-sam", action="store_true", help="predict: fill the boxes instead of calling SAM 3 (pipeline check)")
    ap.add_argument("--limit", type=int, default=None, help="predict: first N positive + N negative images only (pipeline checks)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="predict: prediction dir (default runs/sam3/<checkpoint stem>); metrics: results dir")
    ap.add_argument("--preds", default=None, help="metrics: prediction dir written by predict")
    ap.add_argument("--figure", default="r1/figures/sam3_cases.png")
    ap.add_argument("--allow-missing", action="store_true", help="metrics: skip images without prediction (partial runs)")
    args = ap.parse_args()
    proto = load_protocol()
    log = get_logger(f"sam3_{args.stage}", resolve(f"r1/results/logs/sam3_{args.stage}.log"))
    if args.stage == "predict":
        if not (args.checkpoint or args.checkpoint_map):
            raise SystemExit("--checkpoint or --checkpoint-map is required for predict")
        args.out = args.out or (f"r1/runs/sam3/{Path(args.checkpoint_map).stem}" if args.checkpoint_map else f"r1/runs/sam3/{Path(args.checkpoint).parent.parent.name}_{Path(args.checkpoint).stem}")
        predict(args, proto, log)
    else:
        if not args.preds:
            raise SystemExit("--preds is required for metrics")
        args.out = args.out or "results"
        metrics(args, proto, log)


if __name__ == "__main__":
    main()
