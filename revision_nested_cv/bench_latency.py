#!/usr/bin/env python3
"""
bench_latency.py -- measured latency and GPU memory of the inference strategies.

Measures, with batch size 1, warm-up excluded, CUDA synchronization and images pre-loaded in RAM
(disk/decoding time is not counted; WBF fusion time IS counted):
  single_m  |  tta6_m (3 scales x flip, WBF)  |  ens_nm (Nano+Medium, WBF)  |  ens_ml (Medium+Large, WBF)
Reports mean, std, p95 (ms), FPS and peak GPU memory (MB). Run it on the machine whose numbers
you will quote and state that machine in the paper.

  python bench_latency.py --n path/n.pt --m path/m.pt --l path/l.pt --images dataset_yolo/images/test
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import IMG_EXT, tta_views, wbf  # noqa: E402


def load_imgs(d, n):
    ps = sorted(p for p in Path(d).iterdir() if p.suffix.lower() in IMG_EXT)[:n]
    return [cv2.imread(str(p)) for p in ps]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n"), ap.add_argument("--m"), ap.add_argument("--l")
    ap.add_argument("--images", required=True)
    ap.add_argument("--n-images", type=int, default=33)
    ap.add_argument("--passes", type=int, default=5, help="passes over the image list")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--scales", nargs="+", type=float, default=[0.9, 1.0, 1.1])
    ap.add_argument("--device", default=None)
    ap.add_argument("--only", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()

    import torch
    from ultralytics import YOLO
    cuda = torch.cuda.is_available() and (a.device is None or str(a.device) != "cpu")
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    imgs = load_imgs(a.images, a.n_images)
    print(f"{len(imgs)} images | device: {torch.cuda.get_device_name(0) if cuda else 'cpu'} | "
          f"torch {torch.__version__}")

    def pred(model, img, imgsz):
        r = model.predict(img, imgsz=imgsz, conf=a.conf, verbose=False, device=a.device)[0]
        if len(r.boxes) == 0:
            return np.zeros((0, 4)), np.zeros(0)
        return r.boxes.xyxyn.cpu().numpy(), r.boxes.conf.cpu().numpy()

    def run_single(models, img):
        return pred(models[0], img, a.imgsz)

    def run_tta(models, img):
        mem, w = [], []
        for s in a.scales:
            for flip in (False, True):
                v = cv2.flip(img, 1) if flip else img
                b, sc = pred(models[0], v, int(round(a.imgsz * s / 32) * 32))
                if flip and len(b):
                    b = b.copy(); b[:, [0, 2]] = 1 - b[:, [2, 0]]
                mem.append((b, sc)); w.append(1.0 if s == 1.0 else 0.8)
        return wbf(mem, w)

    def run_ens(models, img):
        return wbf([pred(m, img, a.imgsz) for m in models], [1] * len(models))

    plans = {"single_m": ([a.m], run_single), "tta6_m": ([a.m], run_tta),
             "ens_nm": ([a.n, a.m], run_ens), "ens_ml": ([a.m, a.l], run_ens)}
    if a.only:
        name, (paths, fn) = a.only, plans[a.only]
        if any(p is None for p in paths):
            print(f"SKIP {name}: missing checkpoint path"); return
        models = [YOLO(p) for p in paths]
        for m in models:
            m.predict(imgs[0], imgsz=a.imgsz, verbose=False, device=a.device)
        if cuda:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        w_mb = torch.cuda.memory_allocated() / 2**20 if cuda else float("nan")
        for i in range(a.warmup):
            fn(models, imgs[i % len(imgs)])
        sync()
        t = []
        for _ in range(a.passes):
            for img in imgs:
                sync(); t0 = time.perf_counter(); fn(models, img); sync()
                t.append((time.perf_counter() - t0) * 1000)
        t = np.array(t)
        pk = torch.cuda.max_memory_allocated() / 2**20 if cuda else float("nan")
        # single machine-readable line for the parent process to parse; human-readable on stderr
        print(f"RESULT {name} {t.mean():.3f} {t.std():.3f} {np.percentile(t,95):.3f} {1000/t.mean():.3f} {pk:.1f} {w_mb:.1f}")
        return
    # ---- parent mode: relaunch this same script once per strategy in a FRESH process, so CUDA
    # memory from one strategy can never leak into the next one's "weights_MB"/"peak_MB" reading.
    import subprocess
    print(f"\n{'strategy':10s} {'mean_ms':>9s} {'std_ms':>8s} {'p95_ms':>8s} {'FPS':>7s} {'peak_MB':>9s} {'weights_MB':>11s}")
    for name, (paths, _) in plans.items():
        if any(p is None for p in paths):
            continue
        cmd = [sys.executable, __file__, "--images", a.images, "--n-images", str(a.n_images),
               "--passes", str(a.passes), "--warmup", str(a.warmup), "--imgsz", str(a.imgsz),
               "--conf", str(a.conf), "--only", name]
        for k, v in [("--n", a.n), ("--m", a.m), ("--l", a.l), ("--device", a.device)]:
            if v is not None:
                cmd += [k, str(v)]
        out = subprocess.run(cmd, capture_output=True, text=True)
        line = next((ln for ln in out.stdout.splitlines() if ln.startswith("RESULT")), None)
        if line is None:
            print(f"{name:10s} FAILED -- stderr:\n{out.stderr[-800:]}")
            continue
        _, _, mean_ms, std_ms, p95, fps, pk, w_mb = line.split()
        print(f"{name:10s} {float(mean_ms):9.1f} {float(std_ms):8.1f} {float(p95):8.1f} {float(fps):7.1f} "
              f"{float(pk):9.0f} {float(w_mb):11.0f}")


if __name__ == "__main__":
    main()
