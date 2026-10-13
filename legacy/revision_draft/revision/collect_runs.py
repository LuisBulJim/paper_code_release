"""Consolidate every finished fold into CSV tables and evaluate the test dumps.

Reads runs/<tag>/<model>_seed<seed>/fold<k>/{run_info.json, preds_test/} and writes:
  results/cv_metrics.csv      one row per (tag, model, seed, fold) with validation metrics and best epoch
  results/test_metrics.csv    the same rows evaluated on the frozen test set (single evaluator, conf_fixed)
  results/summary.csv         mean, std, n and 95% t-interval per (tag, model), validation and test
Test evaluation happens only here, after training, and never feeds back into any selection.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import stats

from revision import evaluate as ev
from revision.protocol import get_logger, load_protocol, read_json, resolve

VAL_KEYS = ("map50", "map5095", "precision", "recall", "f1")


def t_interval(values: list[float]) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    if len(x) < 2:
        return float(x.mean()) if len(x) else float("nan"), float("nan"), float("nan")
    half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(x.mean() - half), float(x.mean() + half)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def collect(runs_root: Path, proto: dict, log, tags: list[str] | None = None, include_dry_run: bool = False) -> tuple[list[dict], list[dict]]:
    dataset_root = resolve(proto["dataset"]["output_root"])
    evc = proto["evaluation"]
    test_names = read_json(resolve(proto["dataset"]["test_images_file"]))["images"]
    gt_test = ev.load_ground_truth(dataset_root, "test", test_names)
    cv_rows, test_rows = [], []
    for info_path in sorted(runs_root.glob("*/*/fold*/run_info.json")):
        info = read_json(info_path)
        run_dir = info_path.parent
        tag = run_dir.parents[1].name
        if tags and tag not in tags:
            continue
        if not (run_dir / "DONE").exists() or (info.get("dry_run") and not include_dry_run):
            continue
        if info.get("dry_run"):
            test_names = [n for n in test_names if (run_dir / "preds_test" / f"{Path(n).stem}.json").exists()]
            gt_test = {n: gt_test[n] for n in test_names}
        base = {"tag": tag, "model": info["model"], "seed": info["seed"], "fold": info["fold"], "best_epoch": info.get("best_epoch"), "epochs_run": info.get("epochs_run")}
        cv_rows.append({**base, **{k: info["val_metrics"][k] for k in VAL_KEYS}, "n_val": info["n_val"]})
        preds = ev.load_predictions(run_dir / "preds_test", test_names)
        m = ev.all_metrics(preds, gt_test, sorted(test_names), evc["conf_fixed"], evc["iou_match"])
        test_rows.append({**base, **{k: m[k] for k in VAL_KEYS}, "tp": m["tp"], "fp": m["fp"], "fn": m["fn"]})
        log.info(f"{tag} {info['model']} seed{info['seed']} fold{info['fold']}: val mAP50 {info['val_metrics']['map50']:.4f} | test mAP50 {m['map50']:.4f} F1 {m['f1']:.4f}")
    return cv_rows, test_rows


def summarize(cv_rows: list[dict], test_rows: list[dict]) -> list[dict]:
    out = []
    groups = sorted({(r["tag"], r["model"]) for r in cv_rows})
    for tag, model in groups:
        for stage, rows in (("val", cv_rows), ("test", test_rows)):
            sub = [r for r in rows if r["tag"] == tag and r["model"] == model]
            row = {"tag": tag, "model": model, "stage": stage, "n_runs": len(sub), "n_seeds": len({r["seed"] for r in sub})}
            for k in VAL_KEYS:
                mean, lo, hi = t_interval([r[k] for r in sub])
                row[f"{k}_mean"] = round(mean, 4)
                row[f"{k}_std"] = round(float(np.std([r[k] for r in sub], ddof=1)), 4) if len(sub) > 1 else float("nan")
                row[f"{k}_ci_lo"] = round(lo, 4)
                row[f"{k}_ci_hi"] = round(hi, 4)
            out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--tags", default=None, help="comma-separated experiment tags to include (default: all)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--include-dry-run", action="store_true", help="local pipeline check only")
    args = ap.parse_args()
    proto = load_protocol()
    log = get_logger("collect_runs")
    cv_rows, test_rows = collect(resolve(args.runs), proto, log, args.tags.split(",") if args.tags else None, args.include_dry_run)
    out = resolve(args.out)
    write_csv(out / "cv_metrics.csv", cv_rows)
    write_csv(out / "test_metrics.csv", test_rows)
    write_csv(out / "summary.csv", summarize(cv_rows, test_rows))
    log.info(f"{len(cv_rows)} runs consolidated into {out}")


if __name__ == "__main__":
    main()
