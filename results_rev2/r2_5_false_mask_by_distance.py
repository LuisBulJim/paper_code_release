"""R2.5 -- why the false-mask rate on negatives changes when detector boxes come from the farthest fold model.

Pairs, image by image, the paper map (r1: fold-0 model of repeat 0 for images outside the pool) with the farthest
map (r2_5_neighbour_distance.py), for both thresholds. Negatives only, excluding the two empty-mask images
(image_0105, image_0763). Breaks the false masks down by whether the fold model changed and by the index distance of
the paper-map model (<=5 vs >5). Images whose model did not change must give identical results (consistency check).

Usage (repository root): python results_rev2/r2_5_false_mask_by_distance.py
Writes results_rev2/r2_5/false_mask_by_distance.csv and false_mask_by_distance.json
"""
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXCLUDE = {"image_0105.jpg", "image_0763.jpg"}
RUNS = {
    "conf 0.25": ("revision_nested_cv/r1/results", "results_rev2/r2_5/metrics_std"),
    "val thr 0.1": ("revision_nested_cv/r1/results/sam3_valthr", "results_rev2/r2_5/metrics_valthr"),
}


def load_neg(d: str) -> dict[str, bool]:
    return {r["image"]: r["false_mask"] == "True" for r in csv.DictReader(open(REPO / d / "sam3_negatives.csv"))}


def main() -> None:
    dist = {r["image"]: r for r in csv.DictReader(open(REPO / "results_rev2/r2_5/neighbour_distance.csv"))}
    rows, summary = [], {}
    for run, (old_dir, new_dir) in RUNS.items():
        old, new = load_neg(old_dir), load_neg(new_dir)
        imgs = sorted(i for i in old if i not in EXCLUDE)
        assert set(imgs) == set(new) - EXCLUDE
        groups = {
            "all": imgs,
            "model unchanged": [i for i in imgs if dist[i]["r1_fold"] == dist[i]["farthest_fold"]],
            "model changed": [i for i in imgs if dist[i]["r1_fold"] != dist[i]["farthest_fold"]],
            "paper-map distance <=5": [i for i in imgs if int(dist[i]["r1_distance"]) <= 5],
            "paper-map distance >5": [i for i in imgs if int(dist[i]["r1_distance"]) > 5],
        }
        for g, ims in groups.items():
            n = len(ims)
            o, w = sum(old[i] for i in ims), sum(new[i] for i in ims)
            rows.append({"threshold": run, "group": g, "n_neg": n, "false_paper_map": o,
                         "rate_paper_map": round(o / n, 4) if n else None, "false_farthest_map": w,
                         "rate_farthest_map": round(w / n, 4) if n else None,
                         "only_paper_map": sum(old[i] and not new[i] for i in ims),
                         "only_farthest_map": sum(new[i] and not old[i] for i in ims)})
        unchanged = groups["model unchanged"]
        summary[run] = {"unchanged_identical": all(old[i] == new[i] for i in unchanged)}
    neg = [dist[i] for i in imgs]
    d_old, d_new = sorted(int(r["r1_distance"]) for r in neg), sorted(int(r["farthest_distance"]) for r in neg)
    summary["negatives_168"] = {"median_distance_paper_map": d_old[len(d_old) // 2], "le5_paper_map": sum(d <= 5 for d in d_old),
                                "median_distance_farthest": d_new[len(d_new) // 2], "le5_farthest": sum(d <= 5 for d in d_new),
                                "pool_images_model_changed": sum(r["in_pool"] == "True" and r["r1_fold"] != r["farthest_fold"]
                                                                 for r in dist.values())}
    out = REPO / "results_rev2/r2_5"
    with open(out / "false_mask_by_distance.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json.dump({"rows": rows, "checks": summary}, open(out / "false_mask_by_distance.json", "w"), indent=1)
    for r in rows:
        print(r)
    print(summary)


if __name__ == "__main__":
    main()
