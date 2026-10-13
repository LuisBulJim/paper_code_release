"""Latency of the ext/ inference recipes on the GPU, measured exactly like r1/benchmark_latency.py (R2.6).

Strategies, with the checkpoints of fold r0/f0:
  single      one variant, one pass (x2_multi)
  single_tta  the same model with the TTA r1 chose on validation (6 passes fused by WBF)
  e1          e1_ensemble: the four variants, one pass each, fused by the model-ensemble WBF r1 chose on validation
  e1_tta      e1_ensemble with TTA: each view fused across the four variants, then the views fused (24 passes)
  c4, c4_tta  the control ensemble of four baseline models (direct and sequential, m and l), same fusion
Batch size 1, the 33 images of dataset_yolo/images/test, 20 warm-up images, CUDA synchronised, FP32 and FP16.

Usage (from revision_nested_cv/, GPU):  NESTED_DATASET=../dataset_yolo python ext/latency_ext.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from r1 import predict_dump as pdump  # noqa: E402
from r1.benchmark_latency import timed  # noqa: E402
from r1.protocol import load_protocol, resolve  # noqa: E402
from r1.sweep_inference import TTA_GRID  # noqa: E402

MEMBERS = ["x1_recipe", "x2_multi", "x3_seg", "x4_seg_cp"]
CONTROL = [("direct", "m"), ("direct", "l"), ("sequential", "m"), ("sequential", "l")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=HERE / "out" / "runs", type=Path)
    ap.add_argument("--baseline-runs", default=ROOT / "out" / "runs", type=Path)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=HERE / "results" / "latency_ext.csv", type=Path)
    args = ap.parse_args()
    import torch
    from ultralytics import YOLO

    proto = load_protocol()
    choice = json.load(open(resolve("r1/results/nested_grouped/inference_choice.json")))
    tta = TTA_GRID[choice["tta"]["config"]]
    tta_iou, ens_iou = choice["tta"]["wbf_iou"], choice["wbf"]["iou"]
    skip, conf = proto["inference"]["wbf_models"]["skip"], proto["evaluation"]["conf_dump"]
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    cuda = device.startswith("cuda")
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    images = [cv2.imread(str(p)) for p in sorted((resolve(proto["dataset"]["root"]) / "images" / "test").glob("*.jpg"))]
    assert images, "no test images found"

    def views(model, img, half):
        h, w = img.shape[:2]
        out = []
        for s in tta["scales"]:
            weight = tta["weight_native"] if float(s) == 1.0 else tta["weight_rescaled"]
            for flip in ((False, True) if tta["flip"] else (False,)):
                b, sc, c = pdump.predict_standard(model, cv2.flip(img, 1) if flip else img, int(round(640 * s / 32) * 32), conf, device, half)
                if flip and len(b):
                    b = b.copy()
                    b[:, [0, 2]] = w - b[:, [2, 0]]
                out.append((b, sc, c, weight))
        return out

    def ens_tta(models, img, half):
        h, w = img.shape[:2]
        per_model = [views(m, img, half) for m in models]
        fused = []
        for i in range(len(per_model[0])):
            b, s, c = pdump.fuse([(*mv[i][:3], 1.0) for mv in per_model], w, h, ens_iou, skip)
            fused.append((b, s, c, per_model[0][i][3]))
        return pdump.fuse(fused, w, h, tta_iou, skip)

    rows = []
    for half in ((False, True) if cuda else (False,)):
        models = [YOLO(str(args.runs / f"{m}_r0_f0_m" / "weights" / "best.pt")) for m in MEMBERS]
        control = [YOLO(str(args.baseline_runs / f"{t}_r0_f0_{sz}" / "weights" / "best.pt")) for t, sz in CONTROL]
        single = models[MEMBERS.index("x2_multi")]
        strategies = {
            "single": lambda img, h=half: pdump.predict_standard(single, img, 640, conf, device, h),
            "single_tta": lambda img, h=half: pdump.fuse(views(single, img, h), img.shape[1], img.shape[0], tta_iou, skip),
            "e1": lambda img, h=half: pdump.fuse([(*pdump.predict_standard(m, img, 640, conf, device, h), 1.0) for m in models], img.shape[1], img.shape[0], ens_iou, skip),
            "e1_tta": lambda img, h=half: ens_tta(models, img, h),
            "c4": lambda img, h=half: pdump.fuse([(*pdump.predict_standard(m, img, 640, conf, device, h), 1.0) for m in control], img.shape[1], img.shape[0], ens_iou, skip),
            "c4_tta": lambda img, h=half: ens_tta(control, img, h),
        }
        passes = {"single": 1, "single_tta": len(tta["scales"]) * (2 if tta["flip"] else 1)}
        passes.update({"e1": len(models), "e1_tta": len(models) * passes["single_tta"], "c4": len(control), "c4_tta": len(control) * passes["single_tta"]})
        for name, fn in strategies.items():
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            res = timed(fn, images, warmup=20, iters=args.iters, sync=sync)
            res.update({"strategy": name, "precision": "fp16" if half else "fp32", "forward_passes_per_image": passes[name],
                        "device": torch.cuda.get_device_name(0) if cuda else "cpu",
                        "peak_vram_mb": float(torch.cuda.max_memory_allocated() / 1e6) if cuda else float("nan")})
            rows.append(res)
            print(f"{name:10s} {res['precision']}: {res['ms_mean']:.1f} ms (p95 {res['ms_p95']:.1f}), {res['img_per_s']:.1f} img/s, "
                  f"{passes[name]} passes, VRAM {res['peak_vram_mb']:.0f} MB")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="\n", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
