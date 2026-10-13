"""Aerial pre-training for the Sequential Transfer protocol (source domain: Roboflow Corrosion-hsmae, 36,102 images).

Uses exactly the hyperparameters of hparams.yaml (only the number of epochs differs and is recorded) so
that the fine-tuning stage run by train_kfold.py --init-weights is comparable to Direct Transfer.
The aerial dataset yaml must already exist on the machine (train/val splits as prepared for the submission).
Output: runs/aerial/<model>/weights/best.pt + run_info.json.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from revision.protocol import check_environment, get_logger, load_hparams, load_protocol, resolve, seed_everything, sha256_file, write_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolo26m")
    ap.add_argument("--data", required=True, help="aerial dataset yaml (Ultralytics format)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/aerial")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    from ultralytics import YOLO

    proto = load_protocol()
    hparams = load_hparams()
    run_dir = resolve(args.out) / args.model
    log = get_logger("train_aerial", run_dir / "train.log")
    device = args.device
    if device is None:
        import torch

        device = "0" if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed)
    kwargs = dict(hparams)
    kwargs.update({"data": args.data, "epochs": args.epochs, "project": str(run_dir.parent), "name": run_dir.name, "exist_ok": True, "seed": args.seed, "device": device, "plots": False, "verbose": False})
    if args.dry_run:
        kwargs.update({"epochs": 1, "imgsz": 320, "batch": 4, "workers": 0, "amp": False, "fraction": 0.01})
    t0 = time.perf_counter()
    YOLO(f"{args.model}.pt").train(**kwargs)
    best = run_dir / "weights" / "best.pt"
    info = {"model": args.model, "data": args.data, "hparams": {k: v for k, v in kwargs.items() if k not in ("project", "name", "exist_ok", "verbose", "plots")}, "train_seconds": time.perf_counter() - t0, "best_pt_sha256": sha256_file(best), "environment": check_environment(proto)}
    write_json(run_dir / "run_info.json", info)
    log.info(f"aerial pre-training finished in {info['train_seconds']/3600:.2f} h -> {best}")


if __name__ == "__main__":
    main()
