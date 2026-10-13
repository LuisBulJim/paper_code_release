"""TrashCan cross-domain verification: train YOLO26 with the LIACI hyperparameters, 3 seeds, report on the
official val split with the single evaluator (per class and overall), standard and TTA.

Per seed: r1/runs/trashcan/<model>_seed<seed>/{weights/best.pt, preds_val, preds_val_<tta>, run_info.json, DONE}
Initial weights: <model>.pt resolved from the working directory exactly like nested_cv.py --coco-weights
(run_r1.sh runs from the parent of revision_nested_cv, as run_revision.sh does).
best.pt is selected by Ultralytics fitness on train_sel (a 10% slice of the official train split); the
official val split is only used after training. Re-running skips finished seeds.

Outputs: r1/results/trashcan.csv (seed x strategy x class), r1/results/trashcan_summary.csv (mean, std, 95% t-CI
over seeds), r1/results/trashcan_deltas.csv (paired bootstrap of TTA - standard per seed and mean).
Dry run: --dry-run trains 1 epoch at imgsz 320 on a 2% fraction and evaluates on 30 val images.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from r1 import evaluate as ev
from r1 import predict_dump as pdump
from r1.collect_runs import t_interval, write_csv
from r1.protocol import check_environment, get_logger, load_hparams, load_protocol, read_json, resolve, seed_everything, sha256_file, write_json
from r1.run_test_inference import chosen_tta, tta_dump_name
from r1.trashcan_prepare import refresh_training_files

METRIC_KEYS = ("map50", "map5095", "precision", "recall", "f1")


def best_epoch_from_results(results_csv: Path) -> dict:
    """Ultralytics fitness = 0.1*mAP50 + 0.9*mAP50-95 on the val split; best.pt is the epoch with max fitness."""
    with open(results_csv, encoding="utf-8") as f:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
    if not rows:
        return {}
    fitness = [0.1 * float(r["metrics/mAP50(B)"]) + 0.9 * float(r["metrics/mAP50-95(B)"]) for r in rows]
    k = max(range(len(rows)), key=lambda i: fitness[i])
    return {
        "epochs_run": len(rows),
        "best_epoch": int(float(rows[k]["epoch"])),
        "best_fitness": fitness[k],
        "best_val_map50_ultralytics": float(rows[k]["metrics/mAP50(B)"]),
        "best_val_map5095_ultralytics": float(rows[k]["metrics/mAP50-95(B)"]),
    }


def train_seed(model_name: str, seed: int, run_dir: Path, data_yaml: Path, hparams: dict, proto: dict, device, dry_run: bool, log) -> None:
    from ultralytics import YOLO

    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    kwargs = dict(hparams)
    kwargs.update({"data": str(data_yaml), "project": str(run_dir.parent), "name": run_dir.name, "exist_ok": True, "seed": seed, "device": device, "plots": False, "verbose": False})
    if dry_run:
        kwargs.update({"epochs": 1, "imgsz": 320, "batch": 4, "workers": 0, "amp": False, "fraction": 0.02})
    init = Path(f"{model_name}.pt")
    t0 = time.perf_counter()
    YOLO(str(init)).train(**kwargs)
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"best.pt missing after training: {best}")
    info = {"model": model_name, "seed": seed, "dataset": "trashcan_v2", "dry_run": dry_run, "hparams": {k: v for k, v in kwargs.items() if k not in ("data", "project", "name", "exist_ok", "verbose", "plots")}, "init_weights": str(init.resolve()), "init_weights_sha256": sha256_file(init) if init.exists() else None, "train_seconds": time.perf_counter() - t0, "best_pt_sha256": sha256_file(best), "environment": check_environment(proto), **best_epoch_from_results(run_dir / "results.csv")}
    write_json(run_dir / "run_info.json", info)
    (run_dir / "weights" / "last.pt").unlink(missing_ok=True)
    (run_dir / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    log.info(f"seed {seed} trained in {info['train_seconds']/60:.1f} min | best epoch {info.get('best_epoch')}")


def metric_rows(seed: int, strategy: str, m: dict) -> list[dict]:
    rows = [{"seed": seed, "strategy": strategy, "class": "all", **{k: m[k] for k in METRIC_KEYS}, "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "n_gt_boxes": m["n_gt_boxes"]}]
    for cname, mc in m["classes"].items():
        rows.append({"seed": seed, "strategy": strategy, "class": cname, **{k: mc[k] for k in METRIC_KEYS}, "tp": mc["tp"], "fp": mc["fp"], "fn": mc["fn"], "n_gt_boxes": mc["n_gt_boxes"]})
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    for strategy, cname in sorted({(r["strategy"], r["class"]) for r in rows}):
        sub = [r for r in rows if (r["strategy"], r["class"]) == (strategy, cname)]
        row = {"strategy": strategy, "class": cname, "n_seeds": len(sub)}
        for k in METRIC_KEYS:
            mean, lo, hi = t_interval([r[k] for r in sub])
            row[f"{k}_mean"] = round(mean, 4)
            row[f"{k}_std"] = round(float(np.std([r[k] for r in sub], ddof=1)), 4) if len(sub) > 1 else float("nan")
            row[f"{k}_ci_lo"] = round(lo, 4)
            row[f"{k}_ci_hi"] = round(hi, 4)
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="default: protocol trashcan.model")
    ap.add_argument("--seeds", default=None, help="comma-separated (default: protocol trashcan.seeds)")
    ap.add_argument("--choice", default=None, help="validation choice file (default r1/results/inference_choice.json)")
    ap.add_argument("--tta-config", default=None, help="override the TTA configuration chosen on LIACI validation")
    ap.add_argument("--wbf-iou", type=float, default=None)
    ap.add_argument("--out", default="r1/runs/trashcan")
    ap.add_argument("--results", default="r1/results")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--strict-env", action="store_true")
    args = ap.parse_args()

    proto = load_protocol()
    hparams = load_hparams()
    check_environment(proto, strict=args.strict_env)
    cfg = proto["trashcan"]
    evc = proto["evaluation"]
    model_name = args.model or cfg["model"]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else cfg["seeds"]
    data_root = resolve(cfg["output_root"])
    manifest_path = resolve(cfg["manifest_file"])
    if not manifest_path.exists() or not (data_root / "images").exists():
        raise SystemExit(f"{data_root} or {manifest_path} missing: run python -m r1.trashcan_prepare first")
    manifest = read_json(manifest_path)
    data_yaml = refresh_training_files(data_root, manifest)
    class_names = {int(k): v for k, v in manifest["names"].items()}
    val_names = sorted(e["image"] for e in manifest["images"] if e["split"] == "val")
    device = args.device
    if device is None:
        import torch

        device = "0" if torch.cuda.is_available() else "cpu"
    out_root = resolve(args.out) if not args.dry_run else resolve(args.out) / "dryrun"
    if args.dry_run:
        seeds, val_names = seeds[:1], val_names[:30]
    log = get_logger(f"trashcan_{model_name}", out_root / "train.log")
    tta_config, tta_iou, local = chosen_tta(proto, args.choice, args.tta_config, args.wbf_iou)
    log.info(f"model={model_name} seeds={seeds} device={device} val images={len(val_names)} tta={tta_config} iou={tta_iou}")

    gt = ev.load_ground_truth(data_root, "val", val_names)
    names = sorted(val_names)
    images = [data_root / "images" / "val" / n for n in val_names]
    pred_device = "cuda:0" if device == "0" else device
    rows, deltas = [], []
    for seed in seeds:
        run_dir = out_root / f"{model_name}_seed{seed}"
        if not (run_dir / "DONE").exists():
            train_seed(model_name, seed, run_dir, data_yaml, hparams, proto, device, args.dry_run, log)
        best = run_dir / "weights" / "best.pt"
        dumps = {
            "standard": pdump.ensure_dump("standard", [best], images, run_dir / "preds_val", proto, pred_device, None, log),
            "tta": pdump.ensure_dump("tta", [best], images, run_dir / tta_dump_name("preds_val", tta_config, tta_iou), local, pred_device, None, log),
        }
        preds = {s: ev.load_predictions(d, names) for s, d in dumps.items()}
        for s in dumps:
            rows += metric_rows(seed, s, ev.all_metrics(preds[s], gt, names, evc["conf_fixed"], evc["iou_match"], class_names))
        d = ev.paired_bootstrap(preds["standard"], preds["tta"], gt, names, evc["conf_fixed"], evc["iou_match"], evc["bootstrap_iters"], evc["bootstrap_seed"])
        deltas.append({"seed": seed, "comparison": "tta-standard", **{f"{k}_{s}": v for k, dd in d.items() for s, v in (("delta", dd["delta_mean"]), ("ci_lo", dd["ci95"][0]), ("ci_hi", dd["ci95"][1]), ("p_le0", dd["p_le0"]))}})
        log.info(f"seed {seed}: standard mAP50 {rows[-2 * (1 + len(class_names))]['map50']:.4f} | tta {rows[-(1 + len(class_names))]['map50']:.4f} | delta F1 {d['f1']['delta_mean']:+.4f}")

    res = resolve(args.results) if not args.dry_run else out_root
    write_csv(res / "trashcan.csv", rows)
    write_csv(res / "trashcan_summary.csv", summarize(rows))
    mean = {"seed": "all", "comparison": "tta-standard"}
    for k in ("map50", "f1", "recall", "precision"):
        m, lo, hi = t_interval([r[f"{k}_delta"] for r in deltas])
        mean.update({f"{k}_delta": round(m, 4), f"{k}_ci_lo": round(lo, 4), f"{k}_ci_hi": round(hi, 4), f"{k}_p_le0": round(float(np.mean([r[f"{k}_p_le0"] for r in deltas])), 4)})
    write_csv(res / "trashcan_deltas.csv", deltas + [mean])
    log.info(f"written {res / 'trashcan.csv'}, summary and deltas")


if __name__ == "__main__":
    main()
