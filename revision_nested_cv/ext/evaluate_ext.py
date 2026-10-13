"""Score the ext/ variants against the grouped baseline (YOLO26-m, direct transfer) on identical images.

Rules fixed before any variant was trained:
  * outer test images, inner validation images and seeds are those of out/splits.json (grouped run)
  * TTA configuration = the one r1 chose on validation for the baseline (r1/results/nested_grouped/inference_choice.json)
  * operating threshold of every variant and strategy chosen on its own inner validation (max mean F1 over CONF_GRID)
  * primary metric: out-of-fold mAP@50 of the standard pass; 95 % intervals and paired differences by cluster
    bootstrap over the video groups (1000 replicates, seed 0), exactly as r1
  * decision rule for a combined model: a variant enters only if its mean inner-validation mAP@50 beats the baseline

Writes validation.csv, summary.csv, deltas.csv, decision.json and table_variants.tex to --out.

Usage (from revision_nested_cv/, CPU):
  NESTED_DATASET=../dataset_yolo python ext/evaluate_ext.py
  python ext/evaluate_ext.py --variants x1_recipe x3_seg --preds ext/out/preds
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from r1 import evaluate as ev  # noqa: E402
from r1.nested_eval import Scored, cluster_samples, det_standard, det_tta, gt_from_records, image_sizes, reconstruct_groups  # noqa: E402
from r1.protocol import load_protocol, resolve  # noqa: E402
from r1.sweep_inference import CONF_GRID, TTA_GRID  # noqa: E402

VARIANTS = ["x1_recipe", "x2_multi", "x3_seg", "x4_seg_cp", "x5_combined", "x6_multi_seg", "e1_ensemble", "c4_baselines"]
LABELS = {"x0_baseline": "Baseline (direct transfer)", "x1_recipe": "Small-data recipe", "x2_multi": "9 classes + unused LIACI images",
          "x3_seg": "Segmentation head", "x4_seg_cp": "Segmentation + copy-paste", "x5_combined": "Combination (validation-selected)",
          "x6_multi_seg": "9 classes as polygons + unused images", "e1_ensemble": "Ensemble of the four variants (WBF)",
          "c4_baselines": "Control: ensemble of four baseline models (WBF)"}
# label type of each combinable variant; 9-class boxes and corrosion polygons cannot be merged
KIND = {"x1_recipe": "det", "x2_multi": "multi", "x3_seg": "seg", "x4_seg_cp": "seg"}


def combination(entering: list[str], val: dict[str, float]) -> list[str]:
    """Variants to merge; if 9-class boxes and polygons both entered, keep the type of the best validation variant."""
    cand = [m for m in entering if m in KIND]
    kinds = {KIND[m] for m in cand} - {"det"}
    if len(kinds) > 1:
        keep = KIND[max((m for m in cand if KIND[m] != "det"), key=lambda m: val[m])]
        cand = [m for m in cand if KIND[m] in ("det", keep)]
    return cand
METRICS = ("map50", "map5095", "precision", "recall", "f1", "precision_vt", "recall_vt", "f1_vt")
REPEATS, FOLDS = 3, 5


def load(pattern: str) -> dict[tuple[int, int], dict] | None:
    recs = {}
    for r in range(REPEATS):
        for k in range(FOLDS):
            p = Path(pattern.format(r=r, k=k))
            if not p.exists():
                return None
            with open(p, "rb") as f:
                recs[(r, k)] = pickle.load(f)
    return recs


def detections(views: dict, strategy: str, w: int, h: int, tta: dict) -> ev.Preds:
    if strategy == "standard":
        return det_standard(views, w, h)
    return det_tta(views, TTA_GRID[tta["config"]], tta["wbf_iou"], tta["skip"], w, h)


def validation(recs: dict, gt: ev.GT, strategy: str, tta: dict, iou: float) -> tuple[float, float, list[float]]:
    """(mean inner-validation mAP@50, threshold with the best mean validation F1, per-fold validation mAP@50)."""
    maps, f1 = [], defaultdict(list)
    for (r, k), rec in sorted(recs.items()):
        names = sorted(rec["val"])
        if not any(len(gt[n]["boxes"]) for n in names):
            continue
        preds = {n: detections(rec["val"][n], strategy, gt[n]["width"], gt[n]["height"], tta) for n in names}
        maps.append(ev.coco_map(preds, gt, names)["map50"])
        for c in CONF_GRID:
            f1[c].append(ev.fixed_point_metrics(preds, gt, names, c, iou)["f1"])
    means = {c: float(np.mean(v)) for c, v in f1.items()}
    conf = max(means, key=lambda c: (round(means[c], 12), -abs(c - 0.25)))
    return float(np.mean(maps)), conf, maps


def oof(recs: dict, gt: ev.GT, strategy: str, tta: dict) -> dict[int, ev.Preds]:
    out = defaultdict(dict)
    for (r, k), rec in recs.items():
        for n, views in rec["test"].items():
            out[r][n] = detections(views, strategy, gt[n]["width"], gt[n]["height"], tta)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_latex(path: Path, val_rows: list[dict], summary: list[dict], deltas: list[dict]) -> None:
    val = {(r["model"], r["strategy"]): r["val_map50_mean"] for r in val_rows}
    delta = {(d["comparison"].split(" - ")[0], d["strategy"]): d for d in deltas if d["comparison"].endswith(" - x0_baseline")}
    lines = ["% generated by ext/evaluate_ext.py; differences are paired against the baseline with the same pass",
             r"\begin{tabular}{llccccc}", r"\toprule",
             r"Variant & Pass & Val. mAP@50 & mAP@50 [95\% CI] & $\Delta$mAP@50 [95\% CI] & mAP@[.5:.95] & F1 \\", r"\midrule"]
    for row in summary:
        key = (row["model"], row["strategy"])
        d = delta.get(key)
        dtxt = "--" if d is None else f"{d['map50_delta']:+.3f} [{d['map50_ci_lo']:+.3f}, {d['map50_ci_hi']:+.3f}]"
        label = LABELS.get(row["model"], row["model"].replace("_", r"\_"))
        lines.append(f"{label} & {'TTA' if row['strategy'] == 'tta' else 'Standard'} & {val[key]:.3f} & "
                     f"{row['map50_mean']:.3f} [{row['map50_ci_lo']:.3f}, {row['map50_ci_hi']:.3f}] & {dtxt} & {row['map5095_mean']:.3f} & {row['f1_vt_mean']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="*", default=VARIANTS)
    ap.add_argument("--preds", default=str(HERE / "out" / "preds"))
    ap.add_argument("--baseline", default=str(ROOT / "out" / "preds" / "direct_r{r}_f{k}_m.pkl"))
    ap.add_argument("--extra-pattern", nargs="*", default=[], help="name=pattern pairs scored like a variant (e.g. sequential checks)")
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--pairs", nargs="*", default=["e1_ensemble:c4_baselines"], help="a:b pairs scored as paired differences a - b")
    args = ap.parse_args()

    proto = load_protocol()
    evc = proto["evaluation"]
    iters = args.iters or evc["bootstrap_iters"]
    choice = json.load(open(resolve("r1/results/nested_grouped/inference_choice.json")))
    tta = {"config": choice["tta"]["config"], "wbf_iou": choice["tta"]["wbf_iou"], "skip": proto["inference"]["tta"]["wbf_skip"]}

    base = load(args.baseline)
    assert base is not None, f"baseline predictions missing: {args.baseline}"
    patterns = {v: str(Path(args.preds) / f"{v}_r{{r}}_f{{k}}_m.pkl") for v in args.variants}
    patterns.update(dict(p.split("=", 1) for p in args.extra_pattern))
    models = {"x0_baseline": base}
    for name, pat in patterns.items():
        recs = load(pat)
        if recs is None:
            print(f"skip {name}: predictions incomplete ({pat})")
            continue
        for key, rec in recs.items():
            if sorted(rec["test"]) != sorted(base[key]["test"]) or sorted(rec["val"]) != sorted(base[key]["val"]):
                raise ValueError(f"{name} r{key[0]} f{key[1]}: test/val images differ from the baseline")
        models[name] = recs

    names = sorted({n for rec in base.values() for n in rec["test"]})
    gt = gt_from_records(base, image_sizes(proto, names))
    splits = json.load(open(resolve(proto["nested"]["out"]) / "splits.json"))
    groups = reconstruct_groups(splits)
    positive = {n: len(gt[n]["boxes"]) > 0 for n in names}
    samples, _ = cluster_samples(groups, names, positive, iters, evc["bootstrap_seed"])

    val_rows, summary, deltas, boot, full = [], [], [], {}, {}
    for name, recs in models.items():
        for strategy in ("standard", "tta"):
            vmap, conf_vt, per_fold = validation(recs, gt, strategy, tta, evc["iou_match"])
            val_rows.append({"model": name, "strategy": strategy, "val_map50_mean": round(vmap, 4), "val_folds": len(per_fold), "conf_vt": conf_vt})
            per_rep = {r: Scored(p, gt, names, evc["conf_fixed"], evc["iou_match"], conf_vt) for r, p in oof(recs, gt, strategy, tta).items()}
            full[(name, strategy)] = {r: s.metrics(names) for r, s in per_rep.items()}
            arr = np.array([[np.mean([per_rep[r].metrics(smp)[m] for r in sorted(per_rep)]) for m in METRICS] for smp in samples])
            boot[(name, strategy)] = arr
            row = {"model": name, "strategy": strategy, "conf_vt": conf_vt}
            for j, m in enumerate(METRICS):
                row[f"{m}_mean"] = round(float(np.mean([full[(name, strategy)][r][m] for r in sorted(per_rep)])), 4)
                row[f"{m}_ci_lo"] = round(float(np.nanpercentile(arr[:, j], 2.5)), 4)
                row[f"{m}_ci_hi"] = round(float(np.nanpercentile(arr[:, j], 97.5)), 4)
            summary.append(row)
            print(f"{name:12s} {strategy:8s} val mAP50 {vmap:.4f} | OOF mAP50 {row['map50_mean']:.4f} [{row['map50_ci_lo']:.4f}, {row['map50_ci_hi']:.4f}] | F1@vt {row['f1_vt_mean']:.4f} (conf {conf_vt})")
            if name != "x0_baseline":
                d = {"comparison": f"{name} - x0_baseline", "strategy": strategy}
                for j, m in enumerate(METRICS):
                    diff = arr[:, j] - boot[("x0_baseline", strategy)][:, j]
                    point = float(np.mean([full[(name, strategy)][r][m] - full[("x0_baseline", strategy)][r][m] for r in sorted(per_rep)]))
                    d.update({f"{m}_delta": round(point, 4), f"{m}_ci_lo": round(float(np.nanpercentile(diff, 2.5)), 4), f"{m}_ci_hi": round(float(np.nanpercentile(diff, 97.5)), 4), f"{m}_p_le0": round(float(np.mean(diff <= 0)), 4)})
                deltas.append(d)
    for pair in args.pairs:
        a, b = pair.split(":")
        if a not in models or b not in models:
            print(f"skip pair {pair}: model missing")
            continue
        for strategy in ("standard", "tta"):
            d = {"comparison": f"{a} - {b}", "strategy": strategy}
            for j, m in enumerate(METRICS):
                diff = boot[(a, strategy)][:, j] - boot[(b, strategy)][:, j]
                point = float(np.mean([full[(a, strategy)][r][m] - full[(b, strategy)][r][m] for r in full[(a, strategy)]]))
                d.update({f"{m}_delta": round(point, 4), f"{m}_ci_lo": round(float(np.nanpercentile(diff, 2.5)), 4), f"{m}_ci_hi": round(float(np.nanpercentile(diff, 97.5)), 4), f"{m}_p_le0": round(float(np.mean(diff <= 0)), 4)})
            deltas.append(d)

    out = Path(args.out)
    write_csv(out / "validation.csv", val_rows)
    write_csv(out / "summary.csv", summary)
    if deltas:
        write_csv(out / "deltas.csv", deltas)
    write_latex(out / "table_variants.tex", val_rows, summary, deltas)
    base_val = {r["strategy"]: r["val_map50_mean"] for r in val_rows if r["model"] == "x0_baseline"}
    val_std = {r["model"]: r["val_map50_mean"] for r in val_rows if r["strategy"] == "standard"}
    entering = [m for m, v in val_std.items() if m in KIND and v > base_val["standard"]]
    combo = combination(entering, val_std)
    decision = {"rule": "a variant enters the combined model only if its mean inner-validation mAP@50 (standard pass) beats the baseline; "
                        "if 9-class boxes and polygons both enter, the label type of the best validation variant is kept",
                "baseline_val_map50": base_val, "val_map50_standard": val_std, "entering": entering, "combination": combo, "tta": tta,
                "bootstrap": {"iters": iters, "unit": "video group", "n_groups": len(set(groups.values()))}}
    (out / "decision.json").write_text(json.dumps(decision, indent=1))
    print("variants entering the combination (validation only):", entering or "none", "| combination:", combo or "none")


if __name__ == "__main__":
    main()
