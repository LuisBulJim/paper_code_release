"""R2.1(b) -- score the coarse-grouping retraining (r2_1ab_train.py --setting gap30) against the paper's baseline.

gap30: outer folds built with StratifiedGroupKFold over 17 groups cut only where the frame-index gap exceeds 30 (no
size cap), seeds 42-44, YOLO26-m direct transfer with the paper's recipe. Test folds differ from the paper's, but
every image has exactly one out-of-fold prediction per repeat in both runs, so differences are paired per image.

Same evaluator as the paper (pycocotools mAP@50, stored views, TTA tta_s3_flip_equal IoU 0.6 chosen by r1 on
validation, metric of each repeat averaged over the 3 repeats, 1000 paired replicates, seed 0). Two resampling units:
  g28   the paper's 28 index-run groups (r1 labels)
  g17   the 17 coarse groups used to build the gap30 folds
Also reports each fold's sizes and flags folds whose inner validation has no positive image (checkpoint selection
on such a split cannot improve after epoch 1).
Usage (repository root): python results_rev2/r2_1b_evaluate.py
Writes results_rev2/r2_1/gap30_summary.csv, gap30_deltas.csv, gap30_folds.csv, gap30_numbers.json.
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r2_2_nested_selection as r22  # noqa: E402
from r1 import evaluate as ev  # noqa: E402
from r1.nested_eval import cluster_samples, coco_from_per_image, coco_per_image, gt_from_records, image_sizes, video_groups  # noqa: E402
from r1.protocol import load_protocol, read_json, resolve  # noqa: E402

REPO = r22.REPO
OUT = REPO / "results_rev2" / "r2_1"
RUNS = {"paper": REPO / "revision_nested_cv/out/preds", "gap30": OUT / "gap30/preds"}


def main() -> None:
    proto = load_protocol()
    evc = proto["evaluation"]
    choice = read_json(resolve("r1/results/nested_grouped/inference_choice.json"))
    tta = {"config": choice["tta"]["config"], "iou": choice["tta"]["wbf_iou"]}
    recs = {lab: {(r, k): pickle.load(open(d / f"direct_r{r}_f{k}_m.pkl", "rb")) for r in range(3) for k in range(5)} for lab, d in RUNS.items()}
    names = sorted({n for rec in recs["paper"].values() for n in rec["test"]})
    gt = gt_from_records(recs["paper"], image_sizes(proto, names))
    for r in range(3):
        assert sorted(n for k in range(5) for n in recs["gap30"][(r, k)]["test"]) == names, f"gap30 r{r}: not every image tested once"
    g28, _ = video_groups(proto, read_json(resolve(proto["nested"]["out"]) / "splits.json"))
    g17 = json.load(open(OUT / "gap30" / "groups.json"))
    positive = {n: len(gt[n]["boxes"]) > 0 for n in names}
    samples = {lab: cluster_samples(g, names, positive, evc["bootstrap_iters"], evc["bootstrap_seed"])[0] for lab, g in (("g28", g28), ("g17", g17))}

    per, full, boot = {}, {}, {}
    for lab, rr in recs.items():
        for st in ("standard", "tta"):
            reps = {}
            for r in range(3):
                preds = {}
                for k in range(5):
                    for n, v in rr[(r, k)]["test"].items():
                        preds[n] = r22.detections(v, tta if st == "tta" else None, gt[n]["width"], gt[n]["height"])
                reps[r] = coco_per_image(preds, gt, names)
                if lab == "gap30" and st == "standard":
                    per[r] = preds
            full[(lab, st)] = float(np.mean([coco_from_per_image(reps[r], names)["map50"] for r in range(3)]))
            for u, smp in samples.items():
                boot[(lab, st, u)] = np.array([np.mean([coco_from_per_image(reps[r], s)["map50"] for r in range(3)]) for s in smp])
            print(f"{lab:6s} {st:8s} OOF mAP@50 {full[(lab, st)]:.4f}", flush=True)

    def ci(a):
        return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]

    summary, deltas = [], []
    for lab in recs:
        row = {"run": lab}
        for st in ("standard", "tta"):
            row[f"map50_{st}"] = round(full[(lab, st)], 4)
            for u in samples:
                row[f"ci_{st}_{u}"] = ci(boot[(lab, st, u)])
        row["delta_tta"] = round(full[(lab, "tta")] - full[(lab, "standard")], 4)
        for u in samples:
            d = boot[(lab, "tta", u)] - boot[(lab, "standard", u)]
            row[f"ci_delta_tta_{u}"], row[f"p_delta_tta_le0_{u}"] = ci(d), round(float(np.mean(d <= 0)), 4)
        summary.append(row)
    for st in ("standard", "tta"):
        row = {"comparison": f"gap30 - paper ({st})", "delta": round(full[("gap30", st)] - full[("paper", st)], 4)}
        for u in samples:
            d = boot[("gap30", st, u)] - boot[("paper", st, u)]
            row[f"ci_{u}"], row[f"p_le0_{u}"] = ci(d), round(float(np.mean(d <= 0)), 4)
        deltas.append(row)

    splits = json.load(open(OUT / "gap30" / "splits.json"))
    fold_rows = []
    for r in range(3):
        for k in range(5):
            f = splits[str(r)][k]
            test = sorted(f["test"])
            has_pos = any(positive[n] for n in test)
            fold_rows.append({"repeat": r, "fold": k, "n_train": len(f["train"]), "n_val": len(f["val"]), "n_val_pos": sum(positive[n] for n in f["val"]),
                              "n_test": len(test), "n_test_pos": sum(positive[n] for n in test),
                              "test_map50_standard": round(ev.coco_map({n: per[r][n] for n in test}, gt, test)["map50"], 4) if has_pos else "",
                              "val_without_positives": not any(positive[n] for n in f["val"])})
    for fname, rows in (("gap30_summary.csv", summary), ("gap30_deltas.csv", deltas), ("gap30_folds.csv", fold_rows)):
        with open(OUT / fname, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    numbers = {"tta": tta, "bootstrap": {"iters": evc["bootstrap_iters"], "seed": evc["bootstrap_seed"], "units": {"g28": 28, "g17": len(set(g17.values()))}},
               "summary": summary, "deltas": deltas, "folds_without_val_positives": [(x["repeat"], x["fold"]) for x in fold_rows if x["val_without_positives"]],
               "test_fold_sizes": [x["n_test"] for x in fold_rows]}
    (OUT / "gap30_numbers.json").write_text(json.dumps(numbers, indent=1), encoding="utf-8")
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
