"""Generate the manuscript figures from data/ and results/ (no number typed by hand).

  figures/partitions.png       data flow: LIACI -> frozen test / training pool -> 5 folds x 3 seeds -> test once
  figures/cv_dispersion.png    per-fold validation and test mAP@50 of every checkpoint, by protocol and model
  figures/pr_test.png          precision-recall curves on the test set (IoU 0.5) per model and strategy,
                               detections pooled over the fold checkpoints
  figures/deltas_test.png      per-checkpoint paired-bootstrap delta of F1 (strategy - standard) with 95% CI
  figures/sweep_tta.png        validation sensitivity of TTA: mAP@50 per configuration and F1 vs threshold
  figures/latency.png          latency per strategy and precision
Figures whose inputs are missing are skipped with a log line.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from revision import evaluate as ev
from revision.make_tables import MODEL_LABEL, STRATEGY_LABEL, TAG_LABEL, read_csv
from revision.protocol import get_logger, load_protocol, read_json, resolve

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
CONF_GRID = np.round(np.arange(0.001, 1.0, 0.01), 3)


def fig_partitions(manifest: dict, folds: dict, out: Path) -> None:
    s = manifest["summary"]
    n_seeds, n_folds = len(folds["seeds"]), folds["n_folds"]
    f0 = folds["seeds"][next(iter(folds["seeds"]))][0]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.axis("off")

    def box(x, y, w, h, text, fc="#f2f2f2"):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec="#333333", lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#333333"})

    liaci = manifest["liaci_counts"]
    box(0.01, 0.35, 0.17, 0.3, f"SINTEF LIACI\n{liaci['n_images']} images\n{liaci['n_positive']} with corrosion")
    box(0.25, 0.62, 0.3, 0.32, f"Held-out test (frozen)\n{s['test'].get('positive', 0)} corrosion + {s['test'].get('negative', 0)} background, {s['n_boxes_test']} boxes\nused once, at the end", fc="#fde2c8")
    box(0.25, 0.06, 0.3, 0.32, f"Training pool\n{s['pool'].get('positive', 0)} corrosion + {s['pool'].get('negative', 0)} background\n{s['n_boxes_pool']} boxes")
    box(0.62, 0.06, 0.37, 0.32, f"Stratified {n_folds}-fold x {n_seeds} seeds\ntrain {len(f0['train'])} / val {len(f0['val'])} images per fold\n{n_folds * n_seeds} checkpoints per model (best.pt on val)")
    box(0.62, 0.62, 0.37, 0.32, f"Every checkpoint evaluated on the test set\n(single evaluator, fixed thresholds)\nmean and 95% CI over {n_folds * n_seeds} checkpoints", fc="#fde2c8")
    arrow(0.18, 0.5, 0.25, 0.78)
    arrow(0.18, 0.5, 0.25, 0.22)
    arrow(0.55, 0.22, 0.62, 0.22)
    arrow(0.805, 0.38, 0.805, 0.62)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_cv_dispersion(cv_rows: list[dict], test_rows: list[dict], out: Path) -> None:
    groups = sorted({(r["tag"], r["model"]) for r in cv_rows})
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for ax, rows, title in ((axes[0], cv_rows, "Validation fold (mAP@50)"), (axes[1], test_rows, "Held-out test (mAP@50)")):
        for i, (tag, model) in enumerate(groups):
            vals = [float(r["map50"]) for r in rows if (r["tag"], r["model"]) == (tag, model)]
            if not vals:
                continue
            x = np.full(len(vals), i) + np.random.default_rng(0).uniform(-0.12, 0.12, len(vals))
            ax.scatter(x, vals, s=14, alpha=0.7)
            m = np.mean(vals)
            half = 1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
            ax.errorbar(i, m, yerr=half, fmt="_", color="black", capsize=4, ms=14)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([f"{TAG_LABEL.get(t, t)}\n{MODEL_LABEL.get(m, m)}" for t, m in groups], fontsize=7.5)
        ax.set_title(title)
        ax.grid(axis="y", lw=0.3)
    axes[0].set_ylabel("mAP@50")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def pr_curve(dumps: list[Path], gt: ev.GT, names: list[str], iou: float) -> tuple[np.ndarray, np.ndarray]:
    preds = [ev.load_predictions(d, names) for d in dumps]
    P, R = [], []
    for conf in CONF_GRID:
        tp = fp = fn = 0
        for p in preds:
            m = ev.fixed_point_metrics(p, gt, names, float(conf), iou)
            tp, fp, fn = tp + m["tp"], fp + m["fp"], fn + m["fn"]
        P.append(tp / (tp + fp) if tp + fp else 1.0)
        R.append(tp / (tp + fn) if tp + fn else 0.0)
    return np.asarray(P), np.asarray(R)


def fig_pr_test(inf_rows: list[dict], proto: dict, out: Path, log) -> None:
    dataset_root = resolve(proto["dataset"]["output_root"])
    names = read_json(resolve(proto["dataset"]["test_images_file"]))["images"]
    by_group = defaultdict(list)
    for r in inf_rows:
        by_group[(r["model"], r["strategy"])].append(Path(r["dump"]))
    available = [n for n in names if all((d / f"{Path(n).stem}.json").exists() for ds in by_group.values() for d in ds)]
    if len(available) < len(names):
        log.warning(f"PR figure: only {len(available)}/{len(names)} test images have dumps in every group")
    gt = ev.load_ground_truth(dataset_root, "test", available)
    names = sorted(available)
    models = sorted({m for m, _ in by_group})
    fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models), 3.2), squeeze=False)
    for ax, model in zip(axes[0], models):
        for (m, strategy), dumps in sorted(by_group.items()):
            if m != model:
                continue
            P, R = pr_curve(dumps, gt, names, proto["evaluation"]["iou_match"])
            ax.plot(R, P, lw=1.2, label=f"{STRATEGY_LABEL.get(strategy, strategy)} ({len(dumps)} ckpt)")
        ax.set_title(MODEL_LABEL.get(model, model))
        ax.set_xlabel("Recall")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(lw=0.3)
        ax.legend(fontsize=7)
    axes[0][0].set_ylabel("Precision")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_deltas(delta_rows: list[dict], out: Path) -> None:
    rows = [r for r in delta_rows if r["seed"] != "all"]
    groups = sorted({(r["model"], r["comparison"]) for r in rows})
    fig, axes = plt.subplots(1, len(groups), figsize=(2.6 * len(groups) + 1, 3.4), squeeze=False, sharey=False)
    for ax, (model, comparison) in zip(axes[0], groups):
        sub = sorted((r for r in rows if (r["model"], r["comparison"]) == (model, comparison)), key=lambda r: (int(r["seed"]), int(r["fold"])))
        y = np.arange(len(sub))
        d = np.array([float(r["f1_delta"]) for r in sub])
        lo = d - np.array([float(r["f1_ci_lo"]) for r in sub])
        hi = np.array([float(r["f1_ci_hi"]) for r in sub]) - d
        ax.errorbar(d, y, xerr=[lo, hi], fmt="o", ms=3, capsize=2, lw=0.8)
        ax.axvline(0, color="black", lw=0.6)
        ax.axvline(d.mean(), color="#c0392b", lw=0.8, ls="--", label=f"mean {d.mean():+.3f}")
        ax.set_yticks(y)
        ax.set_yticklabels([f"s{r['seed']}f{r['fold']}" for r in sub], fontsize=6.5)
        ax.set_title(f"{MODEL_LABEL.get(model, model)}: {comparison}", fontsize=8.5)
        ax.set_xlabel("Delta F1 (95% bootstrap CI)")
        ax.legend(fontsize=7)
        ax.grid(axis="x", lw=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_sweep(summary: list[dict], conf_fixed: float, out: Path) -> None:
    tta = [r for r in summary if r["group"] == "tta"]
    models = sorted({r["model"] for r in tta})
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
    configs = sorted({(r["config"], r["wbf_iou"]) for r in tta}, key=str)
    for model in models:
        vals = []
        for c, iou in configs:
            m = [float(r["map50_mean"]) for r in tta if (r["model"], r["config"], r["wbf_iou"], float(r["conf"])) == (model, c, iou, conf_fixed)]
            vals.append(m[0] if m else np.nan)
        axes[0].plot(range(len(configs)), vals, marker="o", ms=3, lw=1, label=MODEL_LABEL.get(model, model))
        best = max((r for r in tta if r["model"] == model and float(r["conf"]) == conf_fixed), key=lambda r: float(r["map50_mean"]))
        curve = sorted((float(r["conf"]), float(r["f1_mean"])) for r in tta if (r["model"], r["config"], r["wbf_iou"]) == (model, best["config"], best["wbf_iou"]))
        axes[1].plot([c for c, _ in curve], [f for _, f in curve], marker="o", ms=3, lw=1, label=f"{MODEL_LABEL.get(model, model)} ({best['config']})")
    axes[0].set_xticks(range(len(configs)))
    axes[0].set_xticklabels([f"{c}\niou{iou}" if iou else c for c, iou in configs], rotation=60, fontsize=6.5, ha="right")
    axes[0].set_ylabel(f"mean val mAP@50 (conf {conf_fixed})")
    axes[0].grid(axis="y", lw=0.3)
    axes[0].legend(fontsize=7)
    axes[1].axvline(conf_fixed, color="black", lw=0.6, ls="--")
    axes[1].set_xlabel("confidence threshold")
    axes[1].set_ylabel("mean val F1")
    axes[1].grid(lw=0.3)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_latency(rows: list[dict], out: Path) -> None:
    strategies = sorted({r["strategy"] for r in rows}, key=lambda s: ["standard", "wbf", "tta"].index(s) if s in ("standard", "wbf", "tta") else 9)
    precisions = sorted({r["precision"] for r in rows})
    fig, ax = plt.subplots(figsize=(4.5, 3))
    w = 0.8 / len(precisions)
    for i, prec in enumerate(precisions):
        vals = [next((float(r["ms_mean"]) for r in rows if (r["strategy"], r["precision"]) == (s, prec)), np.nan) for s in strategies]
        p95 = [next((float(r["ms_p95"]) for r in rows if (r["strategy"], r["precision"]) == (s, prec)), np.nan) for s in strategies]
        ax.bar(np.arange(len(strategies)) + i * w, vals, w, yerr=[np.zeros(len(vals)), np.array(p95) - np.array(vals)], capsize=3, label=prec)
    ax.set_xticks(np.arange(len(strategies)) + w * (len(precisions) - 1) / 2)
    ax.set_xticklabels([STRATEGY_LABEL.get(s, s) for s in strategies])
    ax.set_ylabel("latency, ms (bar = mean, whisker to p95)")
    ax.legend(fontsize=7)
    ax.grid(axis="y", lw=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    proto = load_protocol()
    log = get_logger("make_figures")
    res, out = resolve(args.results), resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written, skipped = [], []

    manifest_path, folds_path = resolve(proto["dataset"]["manifest_file"]), resolve(proto["splits"]["folds_file"])
    if manifest_path.exists() and folds_path.exists():
        fig_partitions(read_json(manifest_path), read_json(folds_path), out / "partitions.png")
        written.append("partitions")
    else:
        skipped.append("partitions")
    cv_rows, test_rows = read_csv(res / "cv_metrics.csv"), read_csv(res / "test_metrics.csv")
    if cv_rows:
        fig_cv_dispersion(cv_rows, test_rows, out / "cv_dispersion.png")
        written.append("cv_dispersion")
    else:
        skipped.append("cv_dispersion")
    inf_rows = read_csv(res / "inference_test.csv")
    if inf_rows:
        fig_pr_test(inf_rows, proto, out / "pr_test.png", log)
        written.append("pr_test")
    else:
        skipped.append("pr_test")
    deltas = read_csv(res / "inference_test_deltas.csv")
    if deltas:
        fig_deltas(deltas, out / "deltas_test.png")
        written.append("deltas_test")
    else:
        skipped.append("deltas_test")
    sweep = read_csv(res / "sweep_tta_val_summary.csv")
    if sweep:
        fig_sweep(sweep, proto["evaluation"]["conf_fixed"], out / "sweep_tta.png")
        written.append("sweep_tta")
    else:
        skipped.append("sweep_tta")
    latency = read_csv(res / "latency.csv")
    if latency:
        fig_latency(latency, out / "latency.png")
        written.append("latency")
    else:
        skipped.append("latency")
    log.info(f"written: {written} | skipped (inputs missing): {skipped} -> {out}")


if __name__ == "__main__":
    main()
