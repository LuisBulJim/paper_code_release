"""Generate the manuscript figures from r1/data, r1/results and the out-of-fold dumps (no number typed by hand).

  r1/figures/nested_design.png    data flow of the repeated nested cross-validation grouped by video
  r1/figures/fold_dispersion.png  out-of-fold mAP@50 of every outer fold (standard inference), by protocol and
                                  model, folds grouped by video vs not grouped
  r1/figures/pr_oof.png           precision-recall curves (IoU 0.5) of the out-of-fold predictions on the 261
                                  images, detections of the 3 repeats pooled, per model and strategy
  r1/figures/deltas.png           paired deltas (mAP@50 and F1) with 95% cluster-bootstrap intervals
  r1/figures/ablation.png         mAP@50 with folds grouped by video vs not grouped
  r1/figures/sweep_tta.png        validation sensitivity of TTA: mAP@50 per configuration and F1 vs threshold
  r1/figures/latency.png          latency per strategy and precision
Figures whose inputs are missing are skipped with a log line.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from r1 import evaluate as ev
from r1.make_tables import STRATEGY_LABEL, TAG_LABEL, VARIANT_LABEL, comparison_label, model_label, read_csv
from r1.protocol import get_logger, load_protocol, read_json, resolve

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
CONF_GRID = np.round(np.arange(0.001, 1.0, 0.01), 3)
SIZE_ORDER = {"n": 0, "m": 1, "l": 2}


def model_key(model: str) -> tuple:
    return tuple(SIZE_ORDER.get(m[-1], 9) for m in model.split("+"))


def group_key(g: tuple[str, str]) -> tuple:
    return (g[0], model_key(g[1]))


def short_group(tag: str, model: str) -> str:
    return f"{TAG_LABEL.get(tag, tag).split()[0]}\n{model_label(model)}"


def fig_design(manifest: dict, groups: list[dict], proto: dict, out: Path) -> None:
    s, nc = manifest["summary"], proto["nested"]
    n_groups = len({g["group"] for g in groups}) if groups else None
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.axis("off")

    def box(x, y, w, h, text, fc="#f2f2f2"):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec="#333333", lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#333333"})

    box(0.01, 0.35, 0.18, 0.3, f"LIACI corrosion subset\n{s['images']} images ({s['positive']} corrosion,\n{s['negative']} background), {s['boxes']} boxes" + (f"\n{n_groups} video groups" if n_groups else ""))
    box(0.25, 0.35, 0.22, 0.3, f"{nc['repeats']} repeats x {nc['folds']} outer folds\n(StratifiedGroupKFold:\nno video on both sides)")
    box(0.53, 0.62, 0.22, 0.32, "Outer train portion\ninner split: train / 15% val\nbest.pt and TTA/WBF choice\non inner val only", fc="#e3eef3")
    box(0.53, 0.06, 0.22, 0.32, "Outer test fold\nnever seen by its model", fc="#fde2c8")
    box(0.79, 0.33, 0.21, 0.34, "Out-of-fold predictions:\nevery image once\nper repeat; metrics on\nall images, cluster\nbootstrap by video", fc="#fde2c8")
    arrow(0.19, 0.5, 0.25, 0.5)
    arrow(0.47, 0.5, 0.53, 0.78)
    arrow(0.47, 0.5, 0.53, 0.22)
    arrow(0.75, 0.78, 0.79, 0.57)
    arrow(0.75, 0.22, 0.79, 0.43)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_fold_dispersion(rows: list[dict], out: Path) -> None:
    rows = [r for r in rows if r["strategy"] == "standard"]
    variants = [v for v in ("grouped", "ungrouped") if any(r["variant"] == v for r in rows)]
    groups = sorted({(r["tag"], r["model"]) for r in rows}, key=group_key)
    fig, axes = plt.subplots(1, len(variants), figsize=(4.2 * len(variants), 3.3), sharey=True, squeeze=False)
    for ax, variant in zip(axes[0], variants):
        for i, (tag, model) in enumerate(groups):
            sub = [r for r in rows if (r["variant"], r["tag"], r["model"]) == (variant, tag, model)]
            vals = np.array([float(r["map50"]) for r in sub])
            sizes = np.array([float(r["n_images"]) for r in sub])
            x = np.full(len(vals), i) + np.random.default_rng(0).uniform(-0.15, 0.15, len(vals))
            ax.scatter(x, vals, s=6 + sizes / 3, alpha=0.6)
            ax.plot([i - 0.25, i + 0.25], [vals.mean()] * 2, color="black", lw=1)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([short_group(t, m) for t, m in groups], fontsize=7)
        ax.set_title(f"Outer folds: {VARIANT_LABEL[variant].lower()} (marker size = fold images)", fontsize=8.5)
        ax.grid(axis="y", lw=0.3)
    axes[0][0].set_ylabel("out-of-fold mAP@50 per outer fold")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def pr_points(dumps: list[Path], gt: ev.GT, names: list[str], iou: float) -> tuple[np.ndarray, np.ndarray]:
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


def fig_pr(oof_root: Path, proto: dict, out: Path, conf_fixed: float) -> None:
    from r1.nested_eval import load_gt

    by_group = defaultdict(list)
    for meta_path in sorted(oof_root.glob("*/_meta.json")):
        m = read_json(meta_path)
        if m["tag"] == proto["nested"]["wbf_tag"]:
            by_group[(m["model"], m["strategy"])].append(meta_path.parent)
    first = next(iter(by_group.values()))[0]
    names = sorted(read_json(p)["image"] for p in first.glob("*.json") if not p.name.startswith("_"))
    gt = load_gt(proto, names)
    singles = sorted({m for m, _ in by_group if "+" not in m}, key=model_key)
    fig, axes = plt.subplots(1, len(singles), figsize=(3.2 * len(singles), 3.6), squeeze=False)
    for ax, model in zip(axes[0], singles):
        for (m, strategy), dumps in sorted(by_group.items()):
            if m != model and not (strategy == "wbf" and model in m.split("+")):
                continue
            P, R = pr_points(dumps, gt, names, proto["evaluation"]["iou_match"])
            label = STRATEGY_LABEL.get(strategy, strategy) + (f" ({model_label(m)})" if strategy == "wbf" else "")
            line, = ax.plot(R, P, lw=1.2, label=label)
            k = int(np.argmin(np.abs(CONF_GRID - conf_fixed)))
            ax.plot(R[k], P[k], "o", ms=4, color=line.get_color())
        ax.set_title(f"{model_label(model)} ({TAG_LABEL.get(proto['nested']['wbf_tag'])})", fontsize=8.5)
        ax.set_xlabel("Recall")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(lw=0.3)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=7, frameon=False)
    axes[0][0].set_ylabel(f"Precision (dot: conf {conf_fixed})")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out)
    plt.close(fig)


def fig_deltas(rows: list[dict], out: Path) -> None:
    rows = [r for r in rows if r["variant"] == "grouped"]
    panels = [("map50", "Delta mAP@50"), ("f1", "Delta F1 (conf 0.25)")] + ([("f1_vt", "Delta F1 (threshold chosen on val)")] if "f1_vt_delta" in rows[0] else [])
    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 0.35 * len(rows) + 1.2), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, (key, title) in zip(axes, panels):
        d = np.array([float(r[f"{key}_delta"]) for r in rows])
        lo = d - np.array([float(r[f"{key}_ci_lo"]) for r in rows])
        hi = np.array([float(r[f"{key}_ci_hi"]) for r in rows]) - d
        ax.errorbar(d, y, xerr=[lo, hi], fmt="o", ms=3, capsize=2, lw=0.8)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_xlabel(f"{title}\n95% cluster-bootstrap CI")
        ax.grid(axis="x", lw=0.3)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([comparison_label(r["comparison"]) for r in rows], fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_ablation(summary: list[dict], out: Path) -> None:
    rows = [r for r in summary if r["strategy"] == "standard"]
    groups = sorted({(r["tag"], r["model"]) for r in rows}, key=group_key)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    w = 0.38
    for j, variant in enumerate(("ungrouped", "grouped")):
        vals, lo, hi = [], [], []
        for g in groups:
            r = next((r for r in rows if (r["variant"], r["tag"], r["model"]) == (variant, *g)), None)
            vals.append(float(r["map50_mean"]) if r else np.nan)
            lo.append(float(r["map50_mean"]) - float(r["map50_ci_lo"]) if r else 0)
            hi.append(float(r["map50_ci_hi"]) - float(r["map50_mean"]) if r else 0)
        ax.bar(np.arange(len(groups)) + (j - 0.5) * w, vals, w, yerr=[lo, hi], capsize=3, label=f"outer folds {VARIANT_LABEL[variant].lower()}")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([short_group(t, m) for t, m in groups], fontsize=7)
    ax.set_ylabel("out-of-fold mAP@50 (95% CI)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.legend(fontsize=7, ncol=2, loc="upper center")
    ax.grid(axis="y", lw=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_sweep(summary: list[dict], conf_fixed: float, out: Path) -> None:
    tta = [r for r in summary if r["group"] == "tta"]
    models = sorted({r["model"] for r in tta}, key=model_key)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
    configs = sorted({(r["config"], r["wbf_iou"]) for r in tta}, key=str)
    for model in models:
        vals = []
        for c, iou in configs:
            m = [float(r["map50_mean"]) for r in tta if (r["model"], r["config"], r["wbf_iou"], float(r["conf"])) == (model, c, iou, conf_fixed)]
            vals.append(m[0] if m else np.nan)
        axes[0].plot(range(len(configs)), vals, marker="o", ms=3, lw=1, label=model_label(model))
        best = max((r for r in tta if r["model"] == model and float(r["conf"]) == conf_fixed), key=lambda r: float(r["map50_mean"]))
        curve = sorted((float(r["conf"]), float(r["f1_mean"])) for r in tta if (r["model"], r["config"], r["wbf_iou"]) == (model, best["config"], best["wbf_iou"]))
        axes[1].plot([c for c, _ in curve], [f for _, f in curve], marker="o", ms=3, lw=1, label=f"{model_label(model)} ({best['config']})")
    axes[0].set_xticks(range(len(configs)))
    axes[0].set_xticklabels([f"{c}\niou{iou}" if iou else c for c, iou in configs], rotation=60, fontsize=6.5, ha="right")
    axes[0].set_ylabel(f"mean inner-val mAP@50 (conf {conf_fixed})")
    axes[0].grid(axis="y", lw=0.3)
    axes[0].legend(fontsize=7)
    axes[1].axvline(conf_fixed, color="black", lw=0.6, ls="--")
    axes[1].set_xlabel("confidence threshold")
    axes[1].set_ylabel("mean inner-val F1")
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
    ap.add_argument("--results", default="r1/results")
    ap.add_argument("--oof", default="r1/runs/nested/grouped/oof", help="out-of-fold dumps written by nested_eval evaluate")
    ap.add_argument("--out", default="r1/figures")
    args = ap.parse_args()
    proto = load_protocol()
    log = get_logger("make_figures")
    res, out = resolve(args.results), resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    conf = proto["evaluation"]["conf_fixed"]
    written, skipped = [], []

    def run(name: str, ok: bool, fn) -> None:
        if not ok:
            skipped.append(name)
            return
        fn(out / f"{name}.png")
        written.append(name)

    manifest_path = resolve(proto["dataset"]["manifest_file"])
    groups = read_csv(res / "nested_groups.csv")
    run("nested_design", manifest_path.exists(), lambda p: fig_design(read_json(manifest_path), groups, proto, p))
    folds = read_csv(res / "nested_grouped" / "per_fold.csv") + read_csv(res / "nested_ungrouped" / "per_fold.csv")
    run("fold_dispersion", bool(folds), lambda p: fig_fold_dispersion(folds, p))
    oof = resolve(args.oof)
    run("pr_oof", oof.exists() and any(oof.glob("*/_meta.json")), lambda p: fig_pr(oof, proto, p, conf))
    deltas = read_csv(res / "nested_deltas.csv")
    run("deltas", bool(deltas), lambda p: fig_deltas(deltas, p))
    summary = read_csv(res / "nested_summary.csv")
    run("ablation", {r["variant"] for r in summary} >= {"grouped", "ungrouped"}, lambda p: fig_ablation(summary, p))
    sweep = read_csv(res / "sweep_tta_val_summary.csv")
    run("sweep_tta", bool(sweep), lambda p: fig_sweep(sweep, conf, p))
    latency = read_csv(res / "latency.csv")
    run("latency", bool(latency), lambda p: fig_latency(latency, p))
    log.info(f"written: {written} | skipped (inputs missing): {skipped} -> {out}")


if __name__ == "__main__":
    main()
