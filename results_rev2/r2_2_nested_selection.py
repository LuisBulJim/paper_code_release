"""R2.2 -- every inference choice made separately in each outer fold, using only that fold's inner validation.

The paper chose the checkpoint per fold, but the TTA configuration, the WBF configuration, the operating threshold
and the final recipe on the mean inner validation over the 15 outer folds. Inner-validation images of one fold are
test images of other folds, so the fixed recipe (control ensemble of four baselines + TTA, OOF mAP@50 0.262) is a
fixed-recipe comparison, not an evaluation of the selection procedure. This script evaluates the procedure.

For each outer fold (repeat r, fold k), on that fold's inner validation only:
  1. TTA   the 12 configurations of r1 (4 view sets x WBF IoU 0.4/0.5/0.6) plus the single pass; same rule as r1
           (highest validation mAP@50 averaged over the direct-transfer n/m/l models of the fold)
  2. WBF   the 24 model-ensemble candidates of r1 (pairs and triple of direct n/m/l, equal or larger-favouring
           weights, IoU 0.4/0.5/0.6); same rule as r1 (highest validation mAP@50). The four-model ensembles take its
           IoU and weight mode (larger-favouring: weight 1 for m, 2 for l; e1's members are all Medium -> equal)
  3. recipe the 16 recipes of Table 4 / Table S4 (baseline, x1, x2, x3, x4, x6, e1, c4; single pass and TTA),
           built with the fold's TTA and WBF choices; highest validation mAP@50 of the fold
  4. threshold CONF_GRID value with the highest validation F1 of the chosen recipe (ties -> closest to 0.25)
Each choice is applied to the fold's outer test images; the out-of-fold predictions of a repeat pool the 5 folds.

Reported (same evaluator as the paper: pycocotools mAP@50, stored predictions down to conf 0.001, cluster bootstrap
over the 28 index-run groups, 1000 paired replicates, seed 0, metric averaged over the 3 repeats):
  fixed        the paper's recipe: c4 + TTA, fixed TTA/WBF from r1, pooled-validation threshold (reproduces 0.262)
  c4_nested    the same recipe with TTA, WBF and threshold chosen per fold (isolates steps 1, 2 and 4)
  nested       the full selection procedure (steps 1-4 per fold)
  baseline     YOLO26-m direct, single pass (reference)
P/R/F1 of the per-fold rows sum TP/FP/FN over images, each image scored at the threshold of its own fold.

Usage (from the repository root, CPU, about 10-20 min):
  python results_rev2/r2_2_nested_selection.py
Writes results_rev2/r2_2/: per_fold_choices.csv, validation_grid.csv, summary.csv, deltas.csv, numbers.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
NESTED = REPO / "revision_nested_cv"
os.chdir(NESTED)  # r1.protocol resolves its relative paths against revision_nested_cv/
sys.path.insert(0, str(NESTED))
from r1 import evaluate as ev  # noqa: E402
from r1.nested_eval import (cluster_samples, coco_from_per_image, coco_per_image, det_standard, det_tta,  # noqa: E402
                            gt_from_records, image_sizes, prf, video_groups)
from r1.predict_dump import fuse  # noqa: E402
from r1.protocol import load_protocol, read_json, resolve  # noqa: E402
from r1.sweep_inference import CONF_GRID, TTA_GRID, WBF_IOU_GRID, WBF_WEIGHT_MODES  # noqa: E402

REPEATS, FOLDS = 3, 5
SIZES = ["n", "m", "l"]
VIEW_KEYS = ["std", "flip", "imgsz|0.9|0", "imgsz|0.9|1", "imgsz|1.1|0", "imgsz|1.1|1"]
TTA_CONFIGS = ["tta_s3_flip", "tta_s3_noflip", "tta_flip_only", "tta_s3_flip_equal"]  # the 4 whose views are stored
SINGLE = ["x0_baseline", "x1_recipe", "x2_multi", "x3_seg", "x4_seg_cp", "x6_multi_seg"]
ENSEMBLES = {  # member name -> (pkl pattern, size used by the larger-favouring weight mode)
    "e1_ensemble": {m: (f"ext/out/preds/{m}_r{{r}}_f{{k}}_m.pkl", "m") for m in ["x1_recipe", "x2_multi", "x3_seg", "x4_seg_cp"]},
    "c4_baselines": {"dm": ("out/preds/direct_r{r}_f{k}_m.pkl", "m"), "dl": ("out/preds/direct_r{r}_f{k}_l.pkl", "l"),
                     "sm": ("out/preds/sequential_r{r}_f{k}_m.pkl", "m"), "sl": ("out/preds/sequential_r{r}_f{k}_l.pkl", "l")},
}
SINGLE_PATTERN = {"x0_baseline": "out/preds/direct_r{r}_f{k}_m.pkl",
                  **{m: f"ext/out/preds/{m}_r{{r}}_f{{k}}_m.pkl" for m in SINGLE[1:]}}
MODELS = SINGLE + list(ENSEMBLES)
RECIPES = [(m, s) for m in MODELS for s in ("standard", "tta")]  # order = tie-break (earlier wins)
FIXED_RECIPE = ("c4_baselines", "tta")
OUT = REPO / "results_rev2" / "r2_2"


# ----------------------------------------------------------------------------- predictions

class Store:
    """Lazy access to the stored views of every model, and the WBF of the ensembles under any fusion setting."""

    def __init__(self):
        self.pkl, self.fused = {}, {}

    def rec(self, pattern: str, r: int, k: int) -> dict:
        key = (pattern, r, k)
        if key not in self.pkl:
            with open(pattern.format(r=r, k=k), "rb") as f:
                self.pkl[key] = pickle.load(f)
        return self.pkl[key]

    def base(self, size: str, r: int, k: int) -> dict:
        return self.rec(f"out/preds/direct_r{{r}}_f{{k}}_{size}.pkl", r, k)

    def views(self, model: str, r: int, k: int, part: str, wbf: dict) -> dict[str, dict]:
        """name -> {view key -> (normalised xyxy boxes, scores)}."""
        if model in SINGLE_PATTERN:
            return self.rec(SINGLE_PATTERN[model], r, k)[part]
        weights_mode = wbf["weight_mode"] if model == "c4_baselines" else "equal"
        key = (model, r, k, part, wbf["iou"], weights_mode)
        if key not in self.fused:
            members = ENSEMBLES[model]
            recs = [self.rec(p, r, k) for p, _ in members.values()]
            weights = [1.0 if weights_mode == "equal" else {"m": 1.0, "l": 2.0}[sz] for _, sz in members.values()]
            out = {}
            for n in recs[0][part]:
                out[n] = {}
                for vk in VIEW_KEYS:
                    parts = []
                    for rec, w in zip(recs, weights):
                        b, s = rec[part][n][vk]
                        b, s = np.asarray(b, dtype=float).reshape(-1, 4), np.asarray(s, dtype=float).reshape(-1)
                        parts.append((b, s, np.zeros(len(s), dtype=int), w))
                    fb, fs, _ = fuse(parts, 1, 1, wbf["iou"], 0.001)
                    out[n][vk] = (fb.astype(np.float32), fs.astype(np.float32))  # same dtype as ensemble_eval.py
            self.fused[key] = out
        return self.fused[key]


def detections(views: dict, tta: dict | None, w: int, h: int) -> ev.Preds:
    if tta is None:
        return det_standard(views, w, h)
    return det_tta(views, TTA_GRID[tta["config"]], tta["iou"], 0.001, w, h)


def recipe_preds(store: Store, recipe: tuple, r: int, k: int, part: str, tta: dict | None, wbf: dict, gt: ev.GT) -> ev.Preds:
    model, strategy = recipe
    vs = store.views(model, r, k, part, wbf)
    use_tta = tta if strategy == "tta" else None
    return {n: detections(v, use_tta, gt[n]["width"], gt[n]["height"]) for n, v in vs.items()}


def best_threshold(preds: ev.Preds, gt: ev.GT, names: list[str], iou: float) -> tuple[float, dict]:
    f1 = {c: ev.fixed_point_metrics(preds, gt, names, c, iou)["f1"] for c in CONF_GRID}
    conf = max(f1, key=lambda c: (round(f1[c], 12), -abs(c - 0.25)))
    return conf, f1


# ----------------------------------------------------------------------------- per-fold selection

def select_fold(store: Store, r: int, k: int, gt: ev.GT, iou_match: float, grid_rows: list) -> dict:
    names = sorted(store.base("m", r, k)["val"])
    n_pos = sum(len(gt[n]["boxes"]) > 0 for n in names)
    assert n_pos, f"r{r} f{k}: validation without positives"

    def val_map(preds):
        return ev.coco_map(preds, gt, names)["map50"]

    # 1. TTA: 12 configurations + single pass, mean over direct n/m/l
    tta_scores = {}
    for opt in [None] + [{"config": c, "iou": i} for c in TTA_CONFIGS for i in WBF_IOU_GRID]:
        per = [val_map({n: detections(store.base(s, r, k)["val"][n], opt, gt[n]["width"], gt[n]["height"]) for n in names}) for s in SIZES]
        label = "standard" if opt is None else f"{opt['config']}@{opt['iou']}"
        tta_scores[label] = (float(np.mean(per)), opt)
        grid_rows.append({"repeat": r, "fold": k, "step": "tta", "candidate": label, "val_map50": round(float(np.mean(per)), 6),
                          **{f"val_map50_{s}": round(p, 6) for s, p in zip(SIZES, per)}})
    # same tie-break as r1 (tta_s3_flip preferred), then the single pass last
    tta_label = max(tta_scores, key=lambda c: (tta_scores[c][0], c.startswith("tta_s3_flip@"), c != "standard"))
    tta = tta_scores[tta_label][1]

    # 2. WBF: 24 candidates of r1 on direct n/m/l
    wbf_scores = {}
    for combo in [c for n_m in (2, 3) for c in itertools.combinations(SIZES, n_m)]:
        for wname, mode in WBF_WEIGHT_MODES.items():
            weights = [1.0] * len(combo) if mode is None else [float(i + 1) for i in range(len(combo))]
            for iou in WBF_IOU_GRID:
                preds = {}
                for n in names:
                    parts = []
                    for s, wt in zip(combo, weights):
                        d = det_standard(store.base(s, r, k)["val"][n], gt[n]["width"], gt[n]["height"])
                        parts.append((d["boxes"], d["scores"], d["classes"], wt))
                    b, sc, c = fuse(parts, gt[n]["width"], gt[n]["height"], iou, 0.001)
                    preds[n] = {"boxes": b, "scores": sc, "classes": c}
                label = f"{'+'.join(combo)}_{wname}@{iou}"
                wbf_scores[label] = (val_map(preds), {"members": "+".join(combo), "weight_mode": wname, "iou": iou})
                grid_rows.append({"repeat": r, "fold": k, "step": "wbf", "candidate": label, "val_map50": round(wbf_scores[label][0], 6)})
    wbf_label = max(wbf_scores, key=lambda c: wbf_scores[c][0])  # first max in candidate order, as r1's max()
    wbf = wbf_scores[wbf_label][1]

    # 3. recipe: 16 candidates with the fold's TTA and WBF
    rec_scores = {}
    for recipe in RECIPES:
        rec_scores[recipe] = val_map(recipe_preds(store, recipe, r, k, "val", tta, wbf, gt))
        grid_rows.append({"repeat": r, "fold": k, "step": "recipe", "candidate": f"{recipe[0]}/{recipe[1]}", "val_map50": round(rec_scores[recipe], 6)})
    recipe = max(RECIPES, key=lambda rc: rec_scores[rc])  # ties -> earlier in RECIPES

    # 4. threshold of the chosen recipe; and of the fixed recipe under the fold's own TTA/WBF (c4_nested)
    conf, _ = best_threshold(recipe_preds(store, recipe, r, k, "val", tta, wbf, gt), gt, names, iou_match)
    conf_c4, _ = best_threshold(recipe_preds(store, FIXED_RECIPE, r, k, "val", tta, wbf, gt), gt, names, iou_match)
    return {"repeat": r, "fold": k, "n_val": len(names), "n_val_pos": n_pos,
            "tta": tta, "tta_label": tta_label, "tta_val_map50": tta_scores[tta_label][0], "standard_val_map50": tta_scores["standard"][0],
            "wbf": wbf, "wbf_label": wbf_label, "wbf_val_map50": wbf_scores[wbf_label][0],
            "recipe": recipe, "recipe_val_map50": rec_scores[recipe], "fixed_recipe_val_map50": rec_scores[FIXED_RECIPE],
            "conf": conf, "conf_c4": conf_c4}


# ----------------------------------------------------------------------------- scoring

class PerFoldScored:
    """Out-of-fold detections of one repeat, each image matched at its own fold's threshold."""

    def __init__(self, preds: ev.Preds, gt: ev.GT, names: list[str], conf_of: dict[str, float], iou: float):
        self.coco = coco_per_image(preds, gt, names)
        self.counts = {n: ev.match_image(preds[n]["boxes"], preds[n]["scores"], gt[n]["boxes"], conf_of[n], iou) for n in names}

    def metrics(self, names: list[str]) -> dict[str, float]:
        m = coco_from_per_image(self.coco, names)
        p, rc, f1 = prf(self.counts, names)
        return {**m, "precision": p, "recall": rc, "f1": f1}


METRICS = ("map50", "map5095", "precision", "recall", "f1")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=None, help="bootstrap replicates (default: protocol, 1000)")
    args = ap.parse_args()
    proto = load_protocol()
    evc = proto["evaluation"]
    iou_match = evc["iou_match"]
    iters = args.iters or evc["bootstrap_iters"]
    fixed_choice = read_json(resolve("r1/results/nested_grouped/inference_choice.json"))
    fixed_tta = {"config": fixed_choice["tta"]["config"], "iou": fixed_choice["tta"]["wbf_iou"]}
    assert fixed_choice["wbf"]["weights"] is None
    fixed_wbf = {"members": "+".join(fixed_choice["wbf"]["models"]), "weight_mode": "equal", "iou": fixed_choice["wbf"]["iou"]}

    store = Store()
    base_recs = {(r, k): store.base("m", r, k) for r in range(REPEATS) for k in range(FOLDS)}
    names = sorted({n for rec in base_recs.values() for n in rec["test"]})
    gt = gt_from_records(base_recs, image_sizes(proto, names))
    splits = read_json(resolve(proto["nested"]["out"]) / "splits.json")
    groups, group_src = video_groups(proto, splits)
    print(f"{len(names)} images, {sum(len(g['boxes']) for g in gt.values())} boxes, {len(set(groups.values()))} groups ({group_src})")

    # integrity: every member covers the fold's val/test images exactly
    for (r, k), rec in base_recs.items():
        for pattern in list(SINGLE_PATTERN.values()) + [p for mem in ENSEMBLES.values() for p, _ in mem.values()]:
            other = store.rec(pattern, r, k)
            assert sorted(other["val"]) == sorted(rec["val"]) and sorted(other["test"]) == sorted(rec["test"]), (pattern, r, k)
    # the on-the-fly e1 fusion must equal the stored ext/out/preds/e1_ensemble pkls (fixed WBF)
    for r, k in [(0, 0), (1, 3), (2, 4)]:
        stored = store.rec("ext/out/preds/e1_ensemble_r{r}_f{k}_m.pkl", r, k)
        rebuilt = store.views("e1_ensemble", r, k, "test", fixed_wbf)
        for n in list(rebuilt)[:20]:
            for vk in VIEW_KEYS:
                a, b = np.asarray(stored["test"][n][vk][1]), rebuilt[n][vk][1]
                assert a.shape == b.shape and np.allclose(a, b, atol=1e-6), f"e1 rebuild differs at r{r} f{k} {n} {vk}"
    print("checks OK: member images match every fold; e1 rebuild equals the stored e1 predictions")

    # ---------------- per-fold selection
    grid_rows, folds = [], {}
    for r in range(REPEATS):
        for k in range(FOLDS):
            folds[(r, k)] = select_fold(store, r, k, gt, iou_match, grid_rows)
            f = folds[(r, k)]
            print(f"r{r} f{k} val {f['n_val']:2d} ({f['n_val_pos']} pos) | TTA {f['tta_label']:24s} | WBF {f['wbf_label']:22s} | "
                  f"recipe {f['recipe'][0]}/{f['recipe'][1]} val {f['recipe_val_map50']:.3f} | conf {f['conf']}")

    # ---------------- fixed recipe: pooled-validation threshold, as in the paper
    val_f1 = defaultdict(list)
    for (r, k), rec in base_recs.items():
        vn = sorted(rec["val"])
        preds = recipe_preds(store, FIXED_RECIPE, r, k, "val", fixed_tta, fixed_wbf, gt)
        for c in CONF_GRID:
            val_f1[c].append(ev.fixed_point_metrics(preds, gt, vn, c, iou_match)["f1"])
    means = {c: float(np.mean(v)) for c, v in val_f1.items()}
    conf_fixed_vt = max(means, key=lambda c: (round(means[c], 12), -abs(c - 0.25)))
    fixed_val = float(np.mean([ev.coco_map(recipe_preds(store, FIXED_RECIPE, r, k, "val", fixed_tta, fixed_wbf, gt), gt, sorted(rec["val"]))["map50"]
                               for (r, k), rec in base_recs.items()]))

    with open(resolve("ext/results/validation.csv"), encoding="utf-8") as fh:  # the baseline's pooled-validation threshold
        import csv
        conf_base_vt = next(float(x["conf_vt"]) for x in csv.DictReader(fh) if (x["model"], x["strategy"]) == ("x0_baseline", "standard"))

    # ---------------- out-of-fold predictions of each procedure
    procedures = {"baseline": {}, "fixed": {}, "c4_nested": {}, "nested": {}}
    confs = {p: defaultdict(dict) for p in procedures}
    for (r, k), rec in base_recs.items():
        f = folds[(r, k)]
        test = sorted(rec["test"])
        sets = {
            "baseline": (recipe_preds(store, ("x0_baseline", "standard"), r, k, "test", None, fixed_wbf, gt), conf_base_vt),
            "fixed": (recipe_preds(store, FIXED_RECIPE, r, k, "test", fixed_tta, fixed_wbf, gt), conf_fixed_vt),
            "c4_nested": (recipe_preds(store, FIXED_RECIPE, r, k, "test", f["tta"], f["wbf"], gt), f["conf_c4"]),
            "nested": (recipe_preds(store, f["recipe"], r, k, "test", f["tta"], f["wbf"], gt), f["conf"]),
        }
        for p, (preds, c) in sets.items():
            procedures[p].setdefault(r, {}).update(preds)
            for n in test:
                confs[p][r][n] = c
        f["n_test"] = len(test)
        f["test_map50"] = ev.coco_map(sets["nested"][0], gt, test)["map50"] if any(len(gt[n]["boxes"]) for n in test) else float("nan")
    for p, per_rep in procedures.items():
        for r, preds in per_rep.items():
            assert sorted(preds) == names, f"{p} r{r}: OOF covers {len(preds)} images"

    scored = {p: {r: PerFoldScored(procedures[p][r], gt, names, confs[p][r], iou_match) for r in range(REPEATS)} for p in procedures}
    full = {p: {r: s.metrics(names) for r, s in per.items()} for p, per in scored.items()}
    for p in procedures:  # fast accumulate == pycocotools on the full set
        ref = ev.coco_map(procedures[p][0], gt, names)["map50"]
        assert abs(ref - full[p][0]["map50"]) < 1e-9, (p, ref, full[p][0]["map50"])

    # diagnostic: test mAP@50 computed within each fold (no pooling of scores across folds), mean over the 15 folds.
    # Pooled OOF mAP ranks boxes of different recipes together, so a procedure that mixes recipes with different
    # confidence scales is penalised even where each fold's choice is good; the fold mean has no such penalty.
    fold_map = {p: {} for p in procedures}
    for (r, k), rec in base_recs.items():
        test = sorted(rec["test"])
        for p in procedures:
            fold_map[p][(r, k)] = scored[p][r].metrics(test)["map50"]
        folds[(r, k)]["test_map50_fixed"] = fold_map["fixed"][(r, k)]

    positive = {n: len(gt[n]["boxes"]) > 0 for n in names}
    samples, _ = cluster_samples(groups, names, positive, iters, evc["bootstrap_seed"])
    boot = {}
    for p, per in scored.items():
        boot[p] = np.array([[np.mean([per[r].metrics(s)[m] for r in range(REPEATS)]) for m in METRICS] for s in samples])
        print(f"bootstrap {p} done")

    # ---------------- outputs
    OUT.mkdir(parents=True, exist_ok=True)
    labels = {"baseline": "YOLO26-m direct, single pass (reference)",
              "fixed": "Fixed recipe: c4 + TTA, TTA/WBF/threshold pooled over the 15 validations (paper)",
              "c4_nested": "Fixed recipe c4 + TTA, TTA/WBF/threshold chosen per fold",
              "nested": "Nested selection procedure: TTA, WBF, recipe and threshold chosen per fold"}
    summary = []
    for p in procedures:
        row = {"procedure": p, "description": labels[p]}
        for j, m in enumerate(METRICS):
            v = [full[p][r][m] for r in range(REPEATS)]
            row[f"{m}_mean"] = round(float(np.mean(v)), 4)
            row[f"{m}_ci_lo"] = round(float(np.nanpercentile(boot[p][:, j], 2.5)), 4)
            row[f"{m}_ci_hi"] = round(float(np.nanpercentile(boot[p][:, j], 97.5)), 4)
            row[f"{m}_per_repeat"] = ";".join(f"{x:.4f}" for x in v)
        row["map50_fold_mean"] = round(float(np.mean(list(fold_map[p].values()))), 4)
        summary.append(row)
    deltas = []
    for a, b in [("fixed", "nested"), ("fixed", "c4_nested"), ("c4_nested", "nested"), ("baseline", "nested"), ("baseline", "fixed")]:
        row = {"comparison": f"{b} - {a}"}
        for j, m in enumerate(METRICS):
            d = boot[b][:, j] - boot[a][:, j]
            row[f"{m}_delta"] = round(float(np.mean([full[b][r][m] - full[a][r][m] for r in range(REPEATS)])), 4)
            row[f"{m}_ci_lo"] = round(float(np.nanpercentile(d, 2.5)), 4)
            row[f"{m}_ci_hi"] = round(float(np.nanpercentile(d, 97.5)), 4)
            row[f"{m}_p_le0"] = round(float(np.mean(d <= 0)), 4)
        deltas.append(row)
    fold_rows = [{"repeat": f["repeat"], "fold": f["fold"], "n_val": f["n_val"], "n_val_pos": f["n_val_pos"], "n_test": f["n_test"],
                  "tta": f["tta_label"], "tta_val_map50": round(f["tta_val_map50"], 4), "standard_val_map50": round(f["standard_val_map50"], 4),
                  "wbf": f["wbf_label"], "wbf_val_map50": round(f["wbf_val_map50"], 4),
                  "recipe": f"{f['recipe'][0]}/{f['recipe'][1]}", "recipe_val_map50": round(f["recipe_val_map50"], 4),
                  "fixed_recipe_val_map50": round(f["fixed_recipe_val_map50"], 4), "conf": f["conf"], "conf_c4": f["conf_c4"],
                  "test_map50_of_choice": round(f["test_map50"], 4), "test_map50_fixed_recipe": round(f["test_map50_fixed"], 4)} for f in folds.values()]

    def write_csv(path, rows):
        import csv
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write_csv(OUT / "per_fold_choices.csv", fold_rows)
    write_csv(OUT / "validation_grid.csv", [{**{"val_map50_n": "", "val_map50_m": "", "val_map50_l": ""}, **g} for g in grid_rows])
    write_csv(OUT / "summary.csv", summary)
    write_csv(OUT / "deltas.csv", deltas)
    def count(key):
        return dict(Counter(str(x[key]) for x in fold_rows).most_common())

    numbers = {"bootstrap": {"iters": len(samples), "seed": evc["bootstrap_seed"], "unit": "index-run group", "n_groups": len(set(groups.values()))},
               "fixed_recipe": {"recipe": "/".join(FIXED_RECIPE), "tta": fixed_tta, "wbf": fixed_wbf, "conf_vt": conf_fixed_vt, "val_map50_mean_15_folds": round(fixed_val, 4)},
               "summary": {row["procedure"]: {**{m: [row[f"{m}_mean"], row[f"{m}_ci_lo"], row[f"{m}_ci_hi"]] for m in METRICS}, "map50_fold_mean": row["map50_fold_mean"]} for row in summary},
               "deltas": {row["comparison"]: {m: [row[f"{m}_delta"], row[f"{m}_ci_lo"], row[f"{m}_ci_hi"], row[f"{m}_p_le0"]] for m in METRICS} for row in deltas},
               "choices_count": {"recipe": count("recipe"), "tta": count("tta"), "wbf": count("wbf"), "conf": count("conf")}}
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=1), encoding="utf-8")
    print(json.dumps(numbers, indent=1))
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
