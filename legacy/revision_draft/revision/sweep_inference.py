"""Sensitivity analysis of the inference strategies on the VALIDATION folds (never on test), and the choice
of the configuration that run_test_inference.py applies to the test set.

For every finished fold checkpoint of a (tag, model) group and every candidate configuration the val
predictions are dumped once at conf 0.001 and evaluated at several confidence thresholds, so the threshold
sweep costs no extra inference. Outputs (suffixed with _<tag> for tags other than dtl):
  results/sweep_tta_val.csv, results/sweep_wbf_val.csv          one row per (config, threshold, seed, fold)
  results/sweep_tta_val_summary.csv, ..._wbf_val_summary.csv    mean and SD over folds x seeds
  results/inference_choice.json                                 the selected configurations

Selection rule, fixed before looking at any result:
  TTA  the (configuration, view-fusion IoU) with the highest validation mAP@50 averaged over all models of
       the group, so one configuration is applied to every model size
  WBF  the (members, weights, IoU) ensemble with the highest mean validation mAP@50
The operating point stays at conf_fixed (protocol.yaml); the other thresholds are reported as sensitivity only.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np

from revision import evaluate as ev
from revision import predict_dump as pdump
from revision.protocol import get_logger, load_protocol, read_json, resolve, write_json

CONF_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

TTA_GRID = {
    "standard": None,
    "tta_s3_flip": {"scales": [0.9, 1.0, 1.1], "flip": True, "weight_native": 1.0, "weight_rescaled": 0.8},
    "tta_s3_noflip": {"scales": [0.9, 1.0, 1.1], "flip": False, "weight_native": 1.0, "weight_rescaled": 0.8},
    "tta_flip_only": {"scales": [1.0], "flip": True, "weight_native": 1.0, "weight_rescaled": 1.0},
    "tta_s3_flip_equal": {"scales": [0.9, 1.0, 1.1], "flip": True, "weight_native": 1.0, "weight_rescaled": 1.0},
    "tta_s3wide_flip": {"scales": [0.8, 1.0, 1.2], "flip": True, "weight_native": 1.0, "weight_rescaled": 0.8},
    "tta_s5_flip": {"scales": [0.8, 0.9, 1.0, 1.1, 1.2], "flip": True, "weight_native": 1.0, "weight_rescaled": 0.8},
}
WBF_IOU_GRID = [0.4, 0.5, 0.6]
WBF_WEIGHT_MODES = {"equal": None, "favour_larger": "asc"}


def fold_runs(runs_root: Path, tag: str, model: str, include_dry_run: bool = False) -> list[dict]:
    """Finished fold checkpoints of one (tag, model) group; dry runs are excluded unless asked for."""
    out = []
    for info in sorted(runs_root.glob(f"{tag}/{model}_seed*/fold*/run_info.json")):
        d = read_json(info)
        if (info.parent / "DONE").exists() and (include_dry_run or not d.get("dry_run")):
            out.append({"seed": d["seed"], "fold": d["fold"], "dir": info.parent, "best": info.parent / "weights" / "best.pt"})
    return out


def evaluate_at_thresholds(dump_dir: Path, dataset_root: Path, val_names: list[str], proto: dict) -> list[dict]:
    gt = ev.load_ground_truth(dataset_root, "pool", val_names)
    preds = ev.load_predictions(dump_dir, val_names)
    names = sorted(val_names)
    coco = ev.coco_map(preds, gt, names)
    rows = []
    for conf in CONF_GRID:
        fp = ev.fixed_point_metrics(preds, gt, names, conf, proto["evaluation"]["iou_match"])
        rows.append({"conf": conf, **coco, "precision": fp["precision"], "recall": fp["recall"], "f1": fp["f1"]})
    return rows


def sweep_tta(runs_root: Path, tag: str, model: str, proto: dict, device, log, folds_doc: dict) -> list[dict]:
    dataset_root = resolve(proto["dataset"]["output_root"])
    rows = []
    for run in fold_runs(runs_root, tag, model):
        val_names = next(f for f in folds_doc["seeds"][str(run["seed"])] if f["fold"] == run["fold"])["val"]
        images = [dataset_root / "images" / "pool" / n for n in val_names]
        for name, cfg in TTA_GRID.items():
            for wbf_iou in (WBF_IOU_GRID if cfg else [None]):
                if cfg is None:
                    dump = run["dir"] / "preds_val"  # produced by train_kfold
                else:
                    dump = pdump.ensure_dump("tta", [run["best"]], images, run["dir"] / "sweep" / f"{name}_iou{wbf_iou}", pdump.tta_protocol(proto, cfg, wbf_iou), device, None, log)
                for r in evaluate_at_thresholds(dump, dataset_root, val_names, proto):
                    rows.append({"group": "tta", "config": name, "wbf_iou": wbf_iou, "weights": "", "model": model, "seed": run["seed"], "fold": run["fold"], **r})
        log.info(f"{model} seed{run['seed']} fold{run['fold']}: TTA sweep done")
    return rows


def sweep_wbf(runs_root: Path, tag: str, models: list[str], proto: dict, device, log, folds_doc: dict) -> list[dict]:
    """Fuse checkpoints of different sizes trained on the same seed and fold."""
    dataset_root = resolve(proto["dataset"]["output_root"])
    per_model = {m: {(r["seed"], r["fold"]): r for r in fold_runs(runs_root, tag, m)} for m in models}
    keys = set.intersection(*(set(v) for v in per_model.values()))
    rows = []
    combos = [c for k in (2, 3) for c in itertools.combinations(models, k) if k <= len(models)]
    for seed, fold in sorted(keys):
        val_names = next(f for f in folds_doc["seeds"][str(seed)] if f["fold"] == fold)["val"]
        images = [dataset_root / "images" / "pool" / n for n in val_names]
        for combo in combos:
            for wname, mode in WBF_WEIGHT_MODES.items():
                weights = None if mode is None else [float(i + 1) for i in range(len(combo))]
                for wbf_iou in WBF_IOU_GRID:
                    cname = "+".join(combo) + f"_{wname}_iou{wbf_iou}"
                    ckpts = [per_model[m][(seed, fold)]["best"] for m in combo]
                    dump = pdump.ensure_dump("wbf", ckpts, images, per_model[combo[0]][(seed, fold)]["dir"] / "sweep_wbf" / cname, pdump.wbf_protocol(proto, wbf_iou), device, weights, log)
                    for r in evaluate_at_thresholds(dump, dataset_root, val_names, proto):
                        rows.append({"group": "wbf", "config": "+".join(combo) + f"_{wname}", "wbf_iou": wbf_iou, "weights": "" if weights is None else ",".join(f"{w:g}" for w in weights), "model": "+".join(combo), "seed": seed, "fold": fold, **r})
        log.info(f"seed{seed} fold{fold}: WBF sweep done")
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    key_fields = ("group", "config", "wbf_iou", "weights", "model", "conf")
    out = []
    for key in sorted({tuple(r[k] for k in key_fields) for r in rows}, key=str):
        sub = [r for r in rows if tuple(r[k] for k in key_fields) == key]
        row = {**dict(zip(key_fields, key)), "n_runs": len(sub)}
        for k in ("map50", "map5095", "precision", "recall", "f1"):
            v = [r[k] for r in sub]
            row[f"{k}_mean"] = round(float(np.mean(v)), 4)
            row[f"{k}_std"] = round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else float("nan")
        out.append(row)
    return out


def choose_tta(summary: list[dict], conf: float) -> dict:
    """Configuration (excluding 'standard') with the highest validation mAP@50 averaged over models."""
    cand = [r for r in summary if r["group"] == "tta" and r["config"] != "standard" and float(r["conf"]) == conf]
    models = sorted({r["model"] for r in cand})
    scores = {}
    for cfg, iou in {(r["config"], r["wbf_iou"]) for r in cand}:
        per_model = [r["map50_mean"] for r in cand if (r["config"], r["wbf_iou"]) == (cfg, iou)]
        if len(per_model) == len(models):
            scores[(cfg, iou)] = float(np.mean(per_model))
    (cfg, iou), score = max(scores.items(), key=lambda kv: (kv[1], kv[0][0] == "tta_s3_flip"))
    std = [r["map50_mean"] for r in summary if r["group"] == "tta" and r["config"] == "standard" and float(r["conf"]) == conf]
    return {"config": cfg, "wbf_iou": float(iou), "val_map50_mean_over_models": round(score, 4), "standard_val_map50_mean_over_models": round(float(np.mean(std)), 4) if std else None, "models": models}


def choose_wbf(summary: list[dict], conf: float) -> dict | None:
    cand = [r for r in summary if r["group"] == "wbf" and float(r["conf"]) == conf]
    if not cand:
        return None
    best = max(cand, key=lambda r: r["map50_mean"])
    return {"models": best["model"].split("+"), "weights": [float(w) for w in best["weights"].split(",")] if best["weights"] else None, "iou": float(best["wbf_iou"]), "val_map50_mean": best["map50_mean"], "config": best["config"]}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise SystemExit(f"nothing to write to {path}: no finished checkpoints for this tag/models")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--tag", default="dtl")
    ap.add_argument("--models", default="yolo26n,yolo26m,yolo26l")
    ap.add_argument("--what", choices=["tta", "wbf", "both"], default="both")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    proto = load_protocol()
    log = get_logger("sweep_inference", resolve("results/logs/sweep_inference.log"))
    folds_doc = read_json(resolve(proto["splits"]["folds_file"]))
    device = args.device
    if device is None:
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    models = args.models.split(",")
    runs_root = resolve(args.runs)
    out = resolve(args.out)
    suffix = "" if args.tag == "dtl" else f"_{args.tag}"
    conf = proto["evaluation"]["conf_fixed"]
    choice_path = out / f"inference_choice{suffix}.json"
    choice = read_json(choice_path) if choice_path.exists() else {}
    choice.update({"tag": args.tag, "conf": conf, "rule": "TTA: highest validation mAP@50 averaged over models; WBF: highest mean validation mAP@50"})
    if args.what in ("tta", "both"):
        rows = [r for m in models for r in sweep_tta(runs_root, args.tag, m, proto, device, log, folds_doc)]
        write_csv(out / f"sweep_tta_val{suffix}.csv", rows)
        summary = summarize(rows)
        write_csv(out / f"sweep_tta_val{suffix}_summary.csv", summary)
        choice["tta"] = choose_tta(summary, conf)
    if args.what in ("wbf", "both") and len(models) > 1:
        rows = sweep_wbf(runs_root, args.tag, models, proto, device, log, folds_doc)
        write_csv(out / f"sweep_wbf_val{suffix}.csv", rows)
        summary = summarize(rows)
        write_csv(out / f"sweep_wbf_val{suffix}_summary.csv", summary)
        choice["wbf"] = choose_wbf(summary, conf)
    write_json(choice_path, choice)
    log.info(f"choice written to {choice_path}: tta={choice.get('tta')} wbf={choice.get('wbf')}")


if __name__ == "__main__":
    main()
