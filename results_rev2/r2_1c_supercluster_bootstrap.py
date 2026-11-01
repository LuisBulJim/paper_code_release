"""R2.1(c) -- cluster bootstrap over coarser index-run groups (superclusters), no retraining.

The paper resamples the 28 index-run groups of groups.csv (index gap > 30, then runs capped at 15 pool images).
Neighbouring groups may still be dependent, which would make those intervals optimistic. Here the same stored
out-of-fold predictions and the same point estimates are bootstrapped over coarser groups, cut only where the
gap between consecutive frame indices exceeds a threshold, with no size cap:
  g28    groups.csv, with r1's labels (the paper; control: reproduces the published intervals)
  gap30  index gap > 30, no cap   (17 groups)
  gap40  index gap > 40, no cap   (10 groups)
  gap45  index gap > 45, no cap   ( 6 groups; extreme case, a percentile interval from 6 clusters is crude)
Only the resampling unit changes: same estimator (metric of each repeat on the resampled images, mean over the
3 repeats), 1000 replicates, seed 0, replicates without positives or negatives redrawn (as r1).

Strategies: every r1 configuration (direct/sequential x n/m/l, single pass and TTA, 3-model WBF) with r1's
validation choices, and the improvement-round models (baseline, x2_multi, x3_seg, e1 and the c4 control, single
pass and TTA) with ext's choices. Metrics: mAP@50 and F1 at each strategy's validation threshold.
Main comparisons: the paired differences reported in the paper.

Usage (from the repository root, CPU):
  python results_rev2/r2_1c_supercluster_bootstrap.py
Writes results_rev2/r2_1c/: groupings.csv, intervals.csv, deltas.csv, width_ratio.csv, numbers.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r2_2_nested_selection as r22  # noqa: E402  (also puts revision_nested_cv on sys.path and chdirs there)
from r1.nested_eval import (MODEL, Scored, cluster_samples, coco_from_per_image, gt_from_records, image_sizes,  # noqa: E402
                            load_records, numeric_id, oof_predictions, prf, video_groups)
from r1.protocol import load_protocol, read_json, resolve  # noqa: E402

REPO = r22.REPO
OUT = REPO / "results_rev2" / "r2_1c"
THRESHOLDS = {"gap30": 30, "gap40": 40, "gap45": 45}
METRICS = ("map50", "f1_vt")
EXT_MODELS = ["x0_baseline", "x2_multi", "x3_seg", "e1_ensemble", "c4_baselines"]


def coarse_groups(names: list[str], threshold: int) -> dict[str, str]:
    order = sorted(names, key=numeric_id)
    out, g = {}, 0
    for i, n in enumerate(order):
        if i and numeric_id(n) - numeric_id(order[i - 1]) > threshold:
            g += 1
        out[n] = f"sc{g:02d}"
    return out


def light(s: Scored, names: list[str]) -> tuple[float, float]:
    return coco_from_per_image(s.coco, names)["map50"], prf(s.counts_vt, names)[2]


_SCORED: dict = {}


def _boot_one(args: tuple[str, list]) -> tuple[str, np.ndarray]:
    key, samples = args
    per = _SCORED[key]
    return key, np.array([[np.mean(v) for v in zip(*[light(per[r], smp) for r in sorted(per)])] for smp in samples])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    proto = load_protocol()
    evc = proto["evaluation"]
    iters = args.iters or evc["bootstrap_iters"]

    # ---------------- r1 strategies, exactly as nested_eval.evaluate_variant
    recs = load_records(proto, proto["nested"]["grouped_suffix"])
    names = sorted({n for rec in recs.values() for n in rec["test"]})
    gt = gt_from_records(recs, image_sizes(proto, names))
    choice = read_json(resolve("r1/results/nested_grouped/inference_choice.json"))
    thr = choice["operating_threshold_on_validation"]
    oof = oof_predictions(recs, gt, proto, choice, names)
    scored = {}
    for key, per_rep in oof.items():
        scored["/".join(key)] = {r: Scored(p, gt, names, evc["conf_fixed"], evc["iou_match"], thr["/".join(key)]) for r, p in per_rep.items()}

    # ---------------- improvement-round models, as ext/evaluate_ext.py (fixed TTA and WBF, own validation threshold)
    with open(resolve("ext/results/validation.csv"), encoding="utf-8") as fh:
        ext_thr = {(x["model"], x["strategy"]): float(x["conf_vt"]) for x in csv.DictReader(fh)}
    fixed_tta = {"config": choice["tta"]["config"], "iou": choice["tta"]["wbf_iou"]}
    fixed_wbf = {"members": "+".join(choice["wbf"]["models"]), "weight_mode": "equal", "iou": choice["wbf"]["iou"]}
    c4_conf = read_json(REPO / "results_rev2" / "r2_2" / "numbers.json")["fixed_recipe"]["conf_vt"]  # c4+TTA (R2.2)
    store = r22.Store()
    for model in EXT_MODELS:
        for strategy in ("standard", "tta"):
            per_rep = {}
            for r in range(r22.REPEATS):
                preds = {}
                for k in range(r22.FOLDS):
                    preds.update(r22.recipe_preds(store, (model, strategy), r, k, "test", fixed_tta, fixed_wbf, gt))
                assert sorted(preds) == names
                conf = ext_thr.get((model, strategy))
                if conf is None:  # c4 is missing from the local ext/results; derive its threshold the same way
                    conf = c4_conf if strategy == "tta" else None
                if conf is None:
                    conf = c4_threshold_standard(store, gt, fixed_tta, fixed_wbf, evc["iou_match"])
                per_rep[r] = Scored(preds, gt, names, evc["conf_fixed"], evc["iou_match"], conf)
            scored[f"ext/{model}/{strategy}"] = per_rep

    full = {k: [np.mean([light(per[r], names)[j] for r in sorted(per)]) for j in range(len(METRICS))] for k, per in scored.items()}

    # ---------------- groupings
    # the paper's 28 groups with r1's labels: same partition as groups.csv, and the labels fix the order in which
    # cluster_samples draws groups, so only these labels reproduce the published replicates exactly
    paper_groups, _ = video_groups(proto, read_json(resolve(proto["nested"]["out"]) / "splits.json"))
    ref = {row["name"]: row["group"] for row in csv.DictReader(open(REPO / "groups.csv"))}
    assert sorted(sorted(n for n in names if paper_groups[n] == g) for g in set(paper_groups.values())) == \
        sorted(sorted(n for n in names if ref[n] == g) for g in set(ref.values())), "r1 blocks differ from groups.csv"
    groupings = {"g28": paper_groups, **{lab: coarse_groups(names, t) for lab, t in THRESHOLDS.items()}}
    positive = {n: len(gt[n]["boxes"]) > 0 for n in names}
    group_rows = []
    for lab, g in groupings.items():
        sizes = Counter(g[n] for n in names)
        group_rows.append({"grouping": lab, "n_groups": len(sizes), "largest": max(sizes.values()),
                           "largest_share": round(max(sizes.values()) / len(names), 3),
                           "groups_with_positives": len({g[n] for n in names if positive[n]}),
                           "groups_with_negatives": len({g[n] for n in names if not positive[n]}),
                           "sizes": " ".join(str(v) for v in sorted(sizes.values(), reverse=True))})

    # ---------------- bootstrap
    global _SCORED
    _SCORED = scored
    boot, redraws = {}, {}
    for lab, g in groupings.items():
        samples, redraws[lab] = cluster_samples(g, names, positive, iters, evc["bootstrap_seed"])
        with Pool(args.jobs) as pool:
            boot[lab] = dict(pool.map(_boot_one, [(k, samples) for k in scored]))
        print(f"{lab}: bootstrap done ({redraws[lab]} redraws)")

    comps = []
    for tag in ("direct", "sequential"):
        for s in ("n", "m", "l"):
            comps.append((f"{tag}/{MODEL[s]}/standard", f"{tag}/{MODEL[s]}/tta", f"{tag} {MODEL[s]}: TTA - single pass"))
    wbf_key = next(k for k in scored if k.endswith("/wbf"))
    comps += [("direct/yolo26m/standard", wbf_key, "3-model WBF - Medium"),
              ("direct/yolo26n/standard", "direct/yolo26m/standard", "Medium - Nano (direct)"),
              ("direct/yolo26m/standard", "direct/yolo26l/standard", "Large - Medium (direct)")]
    for s in ("n", "m", "l"):
        comps.append((f"direct/{MODEL[s]}/standard", f"sequential/{MODEL[s]}/standard", f"{MODEL[s]}: sequential - direct"))
    for st in ("standard", "tta"):
        for m in ("x2_multi", "x3_seg", "e1_ensemble", "c4_baselines"):
            comps.append((f"ext/x0_baseline/{st}", f"ext/{m}/{st}", f"{m} - baseline ({st})"))
        comps.append((f"ext/c4_baselines/{st}", f"ext/e1_ensemble/{st}", f"e1 - c4 ({st})"))

    def ci(a: np.ndarray) -> tuple[float, float]:
        return float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))

    interval_rows, delta_rows = [], []
    for key in scored:
        row = {"strategy": key}
        for j, m in enumerate(METRICS):
            row[f"{m}"] = round(full[key][j], 4)
            for lab in groupings:
                lo, hi = ci(boot[lab][key][:, j])
                row[f"{m}_{lab}_lo"], row[f"{m}_{lab}_hi"], row[f"{m}_{lab}_width"] = round(lo, 4), round(hi, 4), round(hi - lo, 4)
        interval_rows.append(row)
    for a, b, label in comps:
        row = {"comparison": label}
        for j, m in enumerate(METRICS):
            row[f"{m}_delta"] = round(full[b][j] - full[a][j], 4)
            for lab in groupings:
                d = boot[lab][b][:, j] - boot[lab][a][:, j]
                lo, hi = ci(d)
                row.update({f"{m}_{lab}_lo": round(lo, 4), f"{m}_{lab}_hi": round(hi, 4), f"{m}_{lab}_width": round(hi - lo, 4),
                            f"{m}_{lab}_excludes0": bool(lo > 0 or hi < 0), f"{m}_{lab}_p_le0": round(float(np.mean(d <= 0)), 4)})
        delta_rows.append(row)
    ratio_rows = []
    for lab in groupings:
        if lab == "g28":
            continue
        for kind, rows, key in (("interval", interval_rows, "strategy"), ("delta", delta_rows, "comparison")):
            for m in METRICS:
                r_ = [row[f"{m}_{lab}_width"] / row[f"{m}_g28_width"] for row in rows if row[f"{m}_g28_width"] > 0]
                ratio_rows.append({"grouping": lab, "kind": kind, "metric": m, "n": len(r_), "median_width_ratio_vs_g28": round(float(np.median(r_)), 3),
                                   "min": round(float(np.min(r_)), 3), "max": round(float(np.max(r_)), 3)})

    OUT.mkdir(parents=True, exist_ok=True)
    for fname, rows in (("groupings.csv", group_rows), ("intervals.csv", interval_rows), ("deltas.csv", delta_rows), ("width_ratio.csv", ratio_rows)):
        with open(OUT / fname, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    numbers = {"iters": iters, "seed": evc["bootstrap_seed"], "redraws": redraws, "groupings": group_rows, "width_ratio": ratio_rows,
               "excludes_zero_map50": {row["comparison"]: {lab: row[f"map50_{lab}_excludes0"] for lab in groupings} for row in delta_rows}}
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=1), encoding="utf-8")
    print(json.dumps(numbers, indent=1))


def c4_threshold_standard(store: r22.Store, gt: dict, fixed_tta: dict, fixed_wbf: dict, iou_match: float) -> float:
    """Pooled-validation F1 threshold of c4 single pass (same rule as ext/evaluate_ext.validation)."""
    from collections import defaultdict
    from r1 import evaluate as ev
    f1 = defaultdict(list)
    for r in range(r22.REPEATS):
        for k in range(r22.FOLDS):
            vn = sorted(store.base("m", r, k)["val"])
            preds = r22.recipe_preds(store, ("c4_baselines", "standard"), r, k, "val", fixed_tta, fixed_wbf, gt)
            for c in r22.CONF_GRID:
                f1[c].append(ev.fixed_point_metrics(preds, gt, vn, c, iou_match)["f1"])
    means = {c: float(np.mean(v)) for c, v in f1.items()}
    return max(means, key=lambda c: (round(means[c], 12), -abs(c - 0.25)))


if __name__ == "__main__":
    main()
