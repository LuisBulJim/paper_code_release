#!/usr/bin/env python3
"""
predict_extra_views.py -- inference-only pass that ADDS new TTA views to the .pkl files produced
by nested_cv.py, reusing the already-trained checkpoints (no retraining, no GPU time spent on
training). Safe to re-run: it loads the existing pickle, only adds keys that are missing, and
writes a timestamped .bak backup of the original before the first write.

New views added (see README section "Extra TTA views" for the reasoning behind each):
  finescale : 5 scales (0.8, 0.9, 1.0, 1.1, 1.2) x flip  -> keys paper|<s>|<f> / imgsz|<s>|<f>
  rot       : small-angle rotation (+-5, +-8 deg by default) x flip, imgsz mode
              CAVEAT: rotation is around the IMAGE center, so boxes far from the center are
              shifted more than boxes near it when the box is mapped back with the (approximate)
              bounding-rectangle-of-the-rotated-box method. A synthetic check found IoU vs. the
              true box drops to ~0.44 at 15 degrees and ~0.37 at 20 degrees for an off-center box,
              i.e. real detections could fail the IoU>=0.5 matching criterion at those angles.
              Keep angles small (<=8 deg) unless you have verified this is not a problem for your
              corrosion regions (which often sit near an edge of the frame).
  vflip     : vertical flip alone (imgsz mode, scale 1.0) -- tests an assumption the original
              paper stated without evidence ("vertical flip excluded to preserve scene orientation")

Usage:
  python predict_extra_views.py --out nested_out --tag direct   --scales 0.8 0.9 1.0 1.1 1.2 --rot-angles 5 8
  python predict_extra_views.py --out nested_out --tag sequential --scales 0.8 0.9 1.0 1.1 1.2 --rot-angles 5 8
"""
import argparse
import pickle
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import collect_pool, rotate_image, unrotate_boxes_norm, view_key  # noqa: E402


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="nested_out")
    ap.add_argument("--tag", default="direct")
    ap.add_argument("--data-root", default="dataset_yolo",
                    help="must be the SAME data root used for the nested_cv.py run being extended")
    ap.add_argument("--scales", nargs="+", type=float, default=[0.8, 0.9, 1.0, 1.1, 1.2])
    ap.add_argument("--rot-angles", nargs="+", type=float, default=[5.0, 8.0],
                    help="rotation TTA angles in degrees; 0 disables rotation views")
    ap.add_argument("--no-vflip", action="store_true")
    ap.add_argument("--pred-conf", type=float, default=0.001)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--sahi", action="store_true", help="also try SAHI-sliced inference (experimental, guarded)")
    ap.add_argument("--sahi-slice", type=int, default=320)
    ap.add_argument("--sahi-overlap", type=float, default=0.2)
    return ap.parse_args()


def _predict(model, img, imgsz, conf, device):
    r = model.predict(img, imgsz=imgsz, conf=conf, verbose=False, device=device)[0]
    if len(r.boxes) == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    return r.boxes.xyxyn.cpu().numpy().astype(np.float32), r.boxes.conf.cpu().numpy().astype(np.float32)


def extra_views_for_image(model, img_path, args, existing_keys):
    img = cv2.imread(img_path)
    assert img is not None, img_path
    h, w = img.shape[:2]
    new = {}

    # finer multi-scale, both 'paper' (resize + fixed imgsz) and 'imgsz' (true multi-scale) modes
    for s in args.scales:
        for flip in (False, True):
            for mode in ("paper", "imgsz"):
                key = view_key(mode, s, flip)
                if key in existing_keys or key in new:
                    continue
                if mode == "paper":
                    v = cv2.resize(img, (int(w * s), int(h * s))) if abs(s - 1) > 1e-9 else img
                    isz = args.imgsz
                else:
                    v = img
                    isz = int(round(args.imgsz * s / 32) * 32)
                if flip:
                    v = cv2.flip(v, 1)
                b, sc = _predict(model, v, isz, args.pred_conf, args.device)
                if flip and len(b):
                    b = b.copy()
                    b[:, [0, 2]] = 1.0 - b[:, [2, 0]]
                new[key] = (b, sc)

    # vertical flip (scale 1.0, imgsz mode)
    if not args.no_vflip:
        key = "vflip"
        if key not in existing_keys and key not in new:
            v = cv2.flip(img, 0)
            b, sc = _predict(model, v, args.imgsz, args.pred_conf, args.device)
            if len(b):
                b = b.copy()
                b[:, [1, 3]] = 1.0 - b[:, [3, 1]]
            new[key] = (b, sc)

    # rotation (small angles, imgsz mode); see the module docstring for the accuracy caveat
    for ang in args.rot_angles:
        for signed in (ang, -ang):
            for flip in (False, True):
                key = f"rot|{signed:g}|{int(flip)}"
                if key in existing_keys or key in new:
                    continue
                rimg, M = rotate_image(img, signed)
                v = cv2.flip(rimg, 1) if flip else rimg
                b, sc = _predict(model, v, args.imgsz, args.pred_conf, args.device)
                if flip and len(b):
                    b = b.copy()
                    b[:, [0, 2]] = 1.0 - b[:, [2, 0]]
                if len(b):
                    b = unrotate_boxes_norm(b, M, w, h, signed)
                new[key] = (b, sc)

    # SAHI (optional, guarded: never lets a failure here take down the rest of the run)
    if args.sahi:
        key = f"sahi|{args.sahi_slice}|{args.sahi_overlap:g}"
        if key not in existing_keys and key not in new:
            try:
                from sahi import AutoDetectionModel
                from sahi.predict import get_sliced_prediction
                if not hasattr(extra_views_for_image, "_sahi_cache"):
                    extra_views_for_image._sahi_cache = {}
                cache = extra_views_for_image._sahi_cache
                mid = id(model)
                if mid not in cache:
                    cache[mid] = AutoDetectionModel.from_pretrained(
                        model_type="ultralytics", model=model.model, confidence_threshold=args.pred_conf,
                        device=args.device or "cuda:0")
                sm = cache[mid]
                res = get_sliced_prediction(img, sm, slice_height=args.sahi_slice, slice_width=args.sahi_slice,
                                            overlap_height_ratio=args.sahi_overlap, overlap_width_ratio=args.sahi_overlap,
                                            verbose=0)
                b = np.array([[p.bbox.minx / w, p.bbox.miny / h, p.bbox.maxx / w, p.bbox.maxy / h]
                             for p in res.object_prediction_list], dtype=np.float32).reshape(-1, 4)
                sc = np.array([p.score.value for p in res.object_prediction_list], dtype=np.float32)
                new[key] = (b, sc)
            except Exception as e:
                if not hasattr(extra_views_for_image, "_sahi_warned"):
                    print(f"  [SAHI disabled: {type(e).__name__}: {e}]")
                    extra_views_for_image._sahi_warned = True
    return new


def main():
    args = parse()
    out = Path(args.out)
    pkls = sorted((out / "preds").glob(f"{args.tag}_r*_f*_*.pkl"))
    assert pkls, f"no predictions found for tag {args.tag} in {out/'preds'}"
    items = collect_pool(args.data_root)
    by_name = {i["name"]: i["img"] for i in items}
    from ultralytics import YOLO
    t0 = time.time()
    for i, pkl in enumerate(pkls):
        rec = pickle.load(open(pkl, "rb"))
        best = rec["meta"].get("best")
        if not best or not Path(best).exists():
            print(f"[{i+1}/{len(pkls)}] SKIP {pkl.name}: checkpoint not found ({best})")
            continue
        model = YOLO(best)
        changed = False
        for part in ("val", "test"):
            for name, views in rec[part].items():
                # `by_name` image path isn't stored in the pkl; reconstruct from meta if present,
                # else fall back to the original data-root layout recorded at training time.
                img_path = by_name.get(name)
                if img_path is None:
                    print(f"  WARNING: {name} not found under --data-root {args.data_root}; "
                          f"skipping (check you passed the same data root as nested_cv.py).")
                    continue
                new = extra_views_for_image(model, img_path, args, set(views.keys()))
                if new:
                    views.update(new)
                    changed = True
        if changed:
            bak = pkl.with_suffix(".pkl.bak")
            if not bak.exists():
                shutil.copy(pkl, bak)
            tmp = pkl.with_suffix(".pkl.tmp")
            pickle.dump(rec, open(tmp, "wb"))
            tmp.replace(pkl)
            print(f"[{i+1}/{len(pkls)}] updated {pkl.name}")
        else:
            print(f"[{i+1}/{len(pkls)}] {pkl.name}: nothing new (already has these views)")
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
