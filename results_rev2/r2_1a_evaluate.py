"""R2.1(a) -- score the purged-CV retraining (r2_1ab_train.py, settings purge5/15/30) against the paper's baseline.

Same evaluator and bootstrap as the paper: pycocotools mAP@50 on the stored views (conf >= 0.001), TTA with the
configuration r1 chose on validation (tta_s3_flip_equal, IoU 0.6), cluster bootstrap over the 28 index-run groups
(r1's labels, 1000 paired replicates, seed 0), metric of each repeat averaged over the 3 repeats. The outer test
images are those of the paper, so every comparison with the paper's baseline is paired on identical images.

Reported per setting: OOF mAP@50 single pass and TTA with 95% CI; delta TTA - single pass with CI; paired difference
against the paper's baseline (B = 0) for each pass; training images lost per fold (from losses.csv).
Usage (repository root): python results_rev2/r2_1a_evaluate.py
Writes results_rev2/r2_1/purge_summary.csv, purge_deltas.csv, purge_losses_summary.csv, purge_numbers.json.
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r2_2_nested_selection as r22  # noqa: E402  (puts revision_nested_cv on sys.path and chdirs there)
from r1.nested_eval import cluster_samples, coco_from_per_image, coco_per_image, gt_from_records, image_sizes, video_groups  # noqa: E402
from r1.protocol import load_protocol, read_json, resolve  # noqa: E402

REPO = r22.REPO
OUT = REPO / "results_rev2" / "r2_1"
SETTINGS = {"B0 (paper)": REPO / "revision_nested_cv/out/preds", "B5": OUT / "purge5/preds", "B15": OUT / "purge15/preds", "B30": OUT / "purge30/preds"}


def main() -> None:
    proto = load_protocol()
    evc = proto["evaluation"]
    choice = read_json(resolve("r1/results/nested_grouped/inference_choice.json"))
    tta = {"config": choice["tta"]["config"], "iou": choice["tta"]["wbf_iou"]}
    recs = {}
    for lab, d in SETTINGS.items():
        paths = {(r, k): d / f"direct_r{r}_f{k}_m.pkl" for r in range(3) for k in range(5)}
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            print(f"skip {lab}: {len(missing)} fold models missing")
            continue
        recs[lab] = {key: pickle.load(open(p, "rb")) for key, p in paths.items()}
    base = recs["B0 (paper)"]
    names = sorted({n for rec in base.values() for n in rec["test"]})
    gt = gt_from_records(base, image_sizes(proto, names))
    for lab, rr in recs.items():
        for key, rec in rr.items():
            assert sorted(rec["test"]) == sorted(base[key]["test"]), f"{lab} {key}: test images differ from the paper"
    groups, _ = video_groups(proto, read_json(resolve(proto["nested"]["out"]) / "splits.json"))
    positive = {n: len(gt[n]["boxes"]) > 0 for n in names}
    samples, _ = cluster_samples(groups, names, positive, evc["bootstrap_iters"], evc["bootstrap_seed"])

    per_image, full, boot = {}, {}, {}
    for lab, rr in recs.items():
        for strategy in ("standard", "tta"):
            reps = {}
            for r in range(3):
                preds = {}
                for k in range(5):
                    for n, v in rr[(r, k)]["test"].items():
                        preds[n] = r22.detections(v, tta if strategy == "tta" else None, gt[n]["width"], gt[n]["height"])
                reps[r] = coco_per_image(preds, gt, names)
            key = (lab, strategy)
            full[key] = float(np.mean([coco_from_per_image(reps[r], names)["map50"] for r in range(3)]))
            boot[key] = np.array([np.mean([coco_from_per_image(reps[r], s)["map50"] for r in range(3)]) for s in samples])
            print(f"{lab:11s} {strategy:8s} OOF mAP@50 {full[key]:.4f}", flush=True)

    def ci(a: np.ndarray) -> list[float]:
        return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]

    summary, deltas = [], []
    for lab in recs:
        s, t = (lab, "standard"), (lab, "tta")
        summary.append({"setting": lab, "map50_standard": round(full[s], 4), "ci_standard": ci(boot[s]), "map50_tta": round(full[t], 4),
                        "ci_tta": ci(boot[t]), "delta_tta": round(full[t] - full[s], 4), "ci_delta_tta": ci(boot[t] - boot[s]),
                        "p_delta_tta_le0": round(float(np.mean(boot[t] - boot[s] <= 0)), 4)})
        if lab != "B0 (paper)":
            for st in ("standard", "tta"):
                d = boot[(lab, st)] - boot[("B0 (paper)", st)]
                deltas.append({"comparison": f"{lab} - B0 ({st})", "delta": round(full[(lab, st)] - full[("B0 (paper)", st)], 4), "ci": ci(d),
                               "p_le0": round(float(np.mean(d <= 0)), 4)})
    loss_rows = []
    for lab in recs:
        if lab.startswith("B0"):
            continue
        rows = list(csv.DictReader(open(OUT / f"purge{lab[1:]}" / "losses.csv")))
        lost = [int(x["removed_train"]) + int(x["removed_val"]) for x in rows]
        kept = [int(x["n_train"]) for x in rows]
        loss_rows.append({"setting": lab, "lost_train_val_min": min(lost), "lost_median": float(np.median(lost)), "lost_max": max(lost),
                          "lost_total_over_15": sum(lost), "train_left_min": min(kept), "train_left_max": max(kept),
                          "val_left_min": min(int(x["n_val"]) for x in rows), "val_pos_left_min": min(int(x["n_val_pos"]) for x in rows)})
    for fname, rows in (("purge_summary.csv", summary), ("purge_deltas.csv", deltas), ("purge_losses_summary.csv", loss_rows)):
        if rows:
            with open(OUT / fname, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    numbers = {"bootstrap": {"iters": len(samples), "unit": "index-run group (28)", "seed": evc["bootstrap_seed"]}, "tta": tta,
               "summary": summary, "deltas_vs_paper": deltas, "losses": loss_rows}
    (OUT / "purge_numbers.json").write_text(json.dumps(numbers, indent=1), encoding="utf-8")
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
