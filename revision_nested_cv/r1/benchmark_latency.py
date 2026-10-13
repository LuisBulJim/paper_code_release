"""Latency, throughput and GPU memory of the inference strategies, measured with one script and one protocol.

Batch size 1, 640x640 inputs from the test set, 20 warm-up iterations, N timed iterations per image set,
CUDA synchronised, FP16 (half) and FP32. Reports mean and p95 latency in ms, images per second and peak
allocated VRAM. Output: results/latency.csv.
"""

from __future__ import annotations

import argparse
import csv
import time

import cv2
import numpy as np

from r1 import predict_dump as pdump
from r1.run_test_inference import chosen_tta, load_choice
from r1.protocol import environment_report, get_logger, load_protocol, resolve


def timed(fn, images: list[np.ndarray], warmup: int, iters: int, sync) -> dict:
    for img in images[:warmup]:
        fn(img)
    sync()
    times = []
    for _ in range(iters):
        for img in images:
            t0 = time.perf_counter()
            fn(img)
            sync()
            times.append(time.perf_counter() - t0)
    t = np.asarray(times) * 1000
    return {"ms_mean": float(t.mean()), "ms_p95": float(np.percentile(t, 95)), "ms_std": float(t.std()), "img_per_s": float(1000 / t.mean()), "n_timed": int(len(t))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--standard", required=True, help="checkpoint for standard and TTA (e.g. the Medium fold checkpoint)")
    ap.add_argument("--wbf", nargs="*", default=[], help="checkpoints fused by the WBF ensemble, in the order of the chosen members")
    ap.add_argument("--choice", default=None, help="validation choice file (default r1/results/inference_choice.json)")
    ap.add_argument("--tta-config", default=None, help="override the chosen TTA configuration")
    ap.add_argument("--iters", type=int, default=5, help="passes over the image set (33 images) per measurement")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="r1/results/latency.csv")
    args = ap.parse_args()
    import torch
    from ultralytics import YOLO

    proto = load_protocol()
    log = get_logger("benchmark_latency")
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    cuda = device.startswith("cuda")
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    test_dir = resolve(proto["dataset"]["root"]) / "images" / "test"  # the 33 images also timed by bench_latency.py
    images = [cv2.imread(str(p)) for p in sorted(test_dir.glob("*.jpg"))]
    if not images:
        raise SystemExit(f"no images in {test_dir}")
    tta_config, tta_iou, local = chosen_tta(proto, args.choice, args.tta_config, None)
    inf = local["inference"]
    wbf_choice = load_choice(args.choice, required=False).get("wbf") or {}
    wbf_weights = wbf_choice.get("weights") or [1.0] * len(args.wbf)
    wbf_iou = wbf_choice.get("iou", inf["wbf_models"]["iou"])
    if args.wbf and len(wbf_weights) != len(args.wbf):
        raise SystemExit(f"--wbf got {len(args.wbf)} checkpoints but the chosen ensemble has {len(wbf_weights)} members {wbf_choice.get('models')}")
    log.info(f"TTA {tta_config} iou {tta_iou} | WBF members {wbf_choice.get('models')} weights {wbf_weights} iou {wbf_iou}")
    conf = proto["evaluation"]["conf_dump"]
    rows = []
    for half in ((True, False) if cuda else (False,)):
        std_model = YOLO(str(resolve(args.standard)))
        wbf_models = [YOLO(str(resolve(c))) for c in args.wbf]
        strategies = {
            "standard": lambda img, h=half: pdump.predict_standard(std_model, img, inf["imgsz"], conf, device, h),
            "tta": lambda img, h=half: pdump.predict_tta(std_model, img, inf["imgsz"], conf, device, inf["tta"], h),
        }
        if wbf_models:
            strategies["wbf"] = lambda img, h=half: pdump.fuse([(*pdump.predict_standard(m, img, inf["imgsz"], conf, device, h), w) for m, w in zip(wbf_models, wbf_weights)], img.shape[1], img.shape[0], wbf_iou, inf["wbf_models"]["skip"])
        for name, fn in strategies.items():
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            res = timed(fn, images, warmup=20, iters=args.iters, sync=sync)
            res.update({"strategy": name, "precision": "fp16" if half else "fp32", "device": torch.cuda.get_device_name(0) if cuda else "cpu", "peak_vram_mb": float(torch.cuda.max_memory_allocated() / 1e6) if cuda else float("nan"), "n_models": 1 if name != "wbf" else len(wbf_models), "tta_config": tta_config, "forward_passes_per_image": 1 if name == "standard" else (len(inf["tta"]["scales"]) * (2 if inf["tta"]["flip"] else 1) if name == "tta" else len(wbf_models))})
            rows.append(res)
            log.info(f"{name} {res['precision']}: {res['ms_mean']:.1f} ms (p95 {res['ms_p95']:.1f}), {res['img_per_s']:.1f} img/s, VRAM {res['peak_vram_mb']:.0f} MB")
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log.info(f"written {out} | env {environment_report()}")


if __name__ == "__main__":
    main()
