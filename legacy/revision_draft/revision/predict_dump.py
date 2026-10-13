"""Dump detections for a list of images so that every table is computed by evaluate.py from the same files.

Strategies:
  standard  single pass, conf >= conf_dump
  tta       multi-scale (network input 640*s rounded to a multiple of 32) x horizontal flip, views fused with WBF
  wbf       several checkpoints, each run in standard mode, fused with WBF
Output: <out>/<image>.json with pixel xyxy boxes and scores in the original image frame, plus <out>/_meta.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ensemble_boxes import weighted_boxes_fusion

from revision.protocol import environment_report, get_logger, load_protocol, read_json, resolve, sha256_file, write_json


Detections = tuple[np.ndarray, np.ndarray, np.ndarray]  # boxes (n,4) xyxy px, scores (n,), classes (n,) int


def _to_arrays(result) -> Detections:
    boxes = result.boxes.xyxy.cpu().numpy().reshape(-1, 4).astype(float)
    scores = result.boxes.conf.cpu().numpy().astype(float)
    classes = result.boxes.cls.cpu().numpy().astype(int)
    return boxes, scores, classes


def predict_standard(model, image: np.ndarray, imgsz: int, conf: float, device, half: bool = False) -> Detections:
    precision = {"quantize": 16} if half else {}  # ultralytics >= 8.4 spells fp16 inference as quantize=16
    res = model.predict(image, imgsz=imgsz, conf=conf, max_det=300, verbose=False, device=device, **precision)[0]
    return _to_arrays(res)


def fuse(views: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]], width: int, height: int, iou_thr: float, skip_thr: float) -> Detections:
    """WBF over (boxes, scores, classes, weight) views; boxes are normalised to [0,1] for ensemble_boxes
    and mapped back. Boxes of different classes are never fused."""
    boxes_list, scores_list, labels_list, weights = [], [], [], []
    scale = np.array([width, height, width, height], dtype=float)
    for boxes, scores, classes, w in views:
        b = np.clip(boxes / scale, 0.0, 1.0) if len(boxes) else np.zeros((0, 4))
        boxes_list.append(b.tolist())
        scores_list.append(scores.tolist())
        labels_list.append([int(c) for c in classes])
        weights.append(w)
    if sum(len(s) for s in scores_list) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    fb, fs, fl = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_thr)
    return np.asarray(fb, dtype=float).reshape(-1, 4) * scale, np.asarray(fs, dtype=float), np.asarray(fl, dtype=int)


def predict_tta(model, image: np.ndarray, base_imgsz: int, conf: float, device, tta: dict, half: bool = False) -> Detections:
    h, w = image.shape[:2]
    views = []
    for s in tta["scales"]:
        imgsz = int(round(base_imgsz * s / 32) * 32)
        weight = tta["weight_native"] if float(s) == 1.0 else tta["weight_rescaled"]
        views.append((*predict_standard(model, image, imgsz, conf, device, half), weight))
        if tta["flip"]:
            fb, fs, fc = predict_standard(model, cv2.flip(image, 1), imgsz, conf, device, half)
            if len(fb):
                fb = fb.copy()
                fb[:, [0, 2]] = w - fb[:, [2, 0]]
            views.append((fb, fs, fc, weight))
    return fuse(views, w, h, tta["wbf_iou"], tta["wbf_skip"])


def run(strategy: str, checkpoints: list[Path], images: list[Path], out: Path, proto: dict, device, model_weights: list[float] | None, log) -> dict:
    from ultralytics import YOLO

    inf = proto["inference"]
    conf = proto["evaluation"]["conf_dump"]
    imgsz = inf["imgsz"]
    models = [YOLO(str(c)) for c in checkpoints]
    if strategy != "wbf" and len(models) != 1:
        raise SystemExit(f"strategy {strategy} needs exactly one checkpoint")
    if strategy == "wbf" and len(models) < 2:
        raise SystemExit("strategy wbf needs at least two checkpoints")
    weights = model_weights or [1.0] * len(models)
    out.mkdir(parents=True, exist_ok=True)
    times = []
    for i, path in enumerate(images, 1):
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        h, w = image.shape[:2]
        t0 = time.perf_counter()
        if strategy == "standard":
            boxes, scores, classes = predict_standard(models[0], image, imgsz, conf, device)
        elif strategy == "tta":
            boxes, scores, classes = predict_tta(models[0], image, imgsz, conf, device, inf["tta"])
        else:
            views = [(*predict_standard(m, image, imgsz, conf, device), wt) for m, wt in zip(models, weights)]
            boxes, scores, classes = fuse(views, w, h, inf["wbf_models"]["iou"], inf["wbf_models"]["skip"])
        times.append(time.perf_counter() - t0)
        write_json(out / f"{path.stem}.json", {"image": path.name, "width": w, "height": h, "boxes": np.round(boxes, 2).tolist(), "scores": np.round(scores, 5).tolist(), "classes": classes.tolist()})
        if i % 50 == 0 or i == len(images):
            log.info(f"  {i}/{len(images)} images")
    meta = {
        "strategy": strategy,
        "checkpoints": [{"path": str(c), "sha256": sha256_file(c)} for c in checkpoints],
        "model_weights": weights,
        "conf_dump": conf,
        "imgsz": imgsz,
        "tta": inf["tta"] if strategy == "tta" else None,
        "wbf_models": inf["wbf_models"] if strategy == "wbf" else None,
        "n_images": len(images),
        "mean_ms_per_image_informal": float(np.mean(times) * 1000),
        "device": str(device),
        "environment": environment_report(),
    }
    write_json(out / "_meta.json", meta)
    return meta


def tta_protocol(proto: dict, overrides: dict | None, wbf_iou: float | None = None) -> dict:
    """Copy of the protocol with the TTA block overridden (a TTA_GRID entry and/or the view-fusion IoU)."""
    tta = {**proto["inference"]["tta"], **(overrides or {})}
    if wbf_iou is not None:
        tta["wbf_iou"] = wbf_iou
    return {**proto, "inference": {**proto["inference"], "tta": tta}}


def wbf_protocol(proto: dict, iou: float | None) -> dict:
    """Copy of the protocol with the model-ensemble WBF IoU overridden."""
    if iou is None:
        return proto
    return {**proto, "inference": {**proto["inference"], "wbf_models": {**proto["inference"]["wbf_models"], "iou": iou}}}


def dump_matches(out: Path, strategy: str, checkpoints: list[Path], images: list[Path], proto: dict, model_weights: list[float] | None) -> bool:
    """True when <out> was produced with exactly this strategy, checkpoints, images and inference settings."""
    meta_path = out / "_meta.json"
    if not meta_path.exists():
        return False
    meta = read_json(meta_path)
    inf = proto["inference"]
    expected = {
        "strategy": strategy,
        "checkpoints": [str(c) for c in checkpoints],
        "model_weights": model_weights or [1.0] * len(checkpoints),
        "conf_dump": proto["evaluation"]["conf_dump"],
        "imgsz": inf["imgsz"],
        "tta": inf["tta"] if strategy == "tta" else None,
        "wbf_models": inf["wbf_models"] if strategy == "wbf" else None,
        "n_images": len(images),
    }
    found = {k: meta.get(k) for k in expected}
    found["checkpoints"] = [c["path"] for c in meta.get("checkpoints", [])]
    return found == expected and all((out / f"{p.stem}.json").exists() for p in images)


def ensure_dump(strategy: str, checkpoints: list[Path], images: list[Path], out: Path, proto: dict, device, model_weights: list[float] | None, log) -> Path:
    """Re-use <out> only if its metadata matches the requested run; otherwise (re)generate it."""
    if dump_matches(out, strategy, checkpoints, images, proto, model_weights):
        return out
    if (out / "_meta.json").exists():
        log.warning(f"{out} was produced with different settings: regenerating")
        (out / "_meta.json").unlink()
    run(strategy, checkpoints, images, out, proto, device, model_weights, log)
    return out


def resolve_images(dataset_root: Path, split: str, names_file: Path | None) -> list[Path]:
    images_dir = dataset_root / "images" / split
    if names_file is not None:
        names = read_json(names_file)
        names = names.get("images", names) if isinstance(names, dict) else names
        return [images_dir / n for n in names]
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", choices=["standard", "tta", "wbf"], required=True)
    ap.add_argument("--checkpoint", action="append", required=True, help="repeat for wbf")
    ap.add_argument("--model-weight", action="append", type=float, default=None, help="WBF weight per checkpoint (same order)")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--images", default=None, help="JSON list (or {'images': [...]}) restricting the images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None, help="cuda:0 | cpu (default: auto)")
    args = ap.parse_args()
    proto = load_protocol()
    log = get_logger("predict_dump")
    dataset_root = resolve(args.dataset or proto["dataset"]["output_root"])
    images = resolve_images(dataset_root, args.split, Path(args.images) if args.images else None)
    device = args.device
    if device is None:
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    meta = run(args.strategy, [Path(c) for c in args.checkpoint], images, resolve(args.out), proto, device, args.model_weight, log)
    log.info(json.dumps({k: v for k, v in meta.items() if k not in ("environment", "tta", "wbf_models")}))


if __name__ == "__main__":
    main()
