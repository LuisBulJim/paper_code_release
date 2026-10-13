"""Candidate TTA / WBF configurations and the selection rule applied to validation predictions.

Selection rule, fixed before looking at any result:
  TTA  the (configuration, view-fusion IoU) with the highest validation mAP@50 averaged over all models of
       the group, so one configuration is applied to every model size
  WBF  the (members, weights, IoU) ensemble with the highest mean validation mAP@50
The operating point stays at conf_fixed (protocol.yaml); the other thresholds are reported as sensitivity only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from r1.protocol import read_json

CONF_GRID = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

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
