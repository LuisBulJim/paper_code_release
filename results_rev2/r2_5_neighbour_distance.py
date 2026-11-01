"""R2.5 -- frame-index distance between each SAM 3 image and the training/validation frames of its detector.

The SAM 3 set is LIACI's official test split (191 images). r1 gave each image a YOLO26-m fold model of repeat 0: the
fold that had it in its outer test fold for the 27 images of the nested-CV pool, fold 0 for the rest. Fold models
were trained on pool frames, and official-test frames are often index neighbours of pool frames, so a detector box
may come from a model that saw an adjacent frame.

For every image and every fold model (repeat 0) this script computes the smallest |index difference| to the fold's
train + val images, and assigns the fold model with the LARGEST such distance (choice by index only, never by
performance; ties -> the r1 assignment if tied, else the lowest fold). Pool images keep the constraint that the
detector must not have trained on the image itself (distance 0 is impossible for their own test fold, and the
farthest rule never picks a fold that trained on them). Images still within 5 indices of the chosen model's
train/val frames are flagged for the fallback exclusion.

Writes results_rev2/r2_5/: neighbour_distance.csv (per image), sam3_map_farthest_r0_m.json (detector map in the
format of r1/results/sam3_map_r0_m.json, for the GPU re-evaluation), neighbour_summary.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
NESTED = REPO / "revision_nested_cv"
OUT = REPO / "results_rev2" / "r2_5"
REPEAT, SIZE, TAG, MODEL = 0, "m", "direct", "yolo26m"
BUFFER = 5


def num(n: str) -> int:
    return int("".join(c for c in Path(n).stem if c.isdigit()))


def main() -> None:
    splits = json.load(open(NESTED / "out" / "splits.json"))[str(REPEAT)]
    manifest = json.load(open(NESTED / "r1" / "data" / "sam3_manifest.json"))
    old = json.load(open(NESTED / "r1" / "results" / f"sam3_map_r{REPEAT}_{SIZE}.json"))
    seen = [np.array(sorted(num(n) for n in f["train"] + f["val"])) for f in splits]
    rows, images = [], {}
    for e in manifest["images"]:
        n, i = e["image"], num(e["image"])
        dist = [int(np.min(np.abs(s - i))) for s in seen]
        old_fold = int(old["images"][n].split("/fold")[1].split("/")[0])
        best = max(dist)
        cands = [k for k, d in enumerate(dist) if d == best]
        fold = old_fold if old_fold in cands else cands[0]
        images[n] = f"r1/runs/{TAG}/{MODEL}_seed{REPEAT}/fold{fold}/weights/best.pt"
        rows.append({"image": n, "index": i, "positive": e["positive"], "in_pool": n in old["in_nested_cv"],
                     "r1_fold": old_fold, "r1_distance": dist[old_fold], "farthest_fold": fold, "farthest_distance": best,
                     **{f"dist_fold{k}": d for k, d in enumerate(dist)},
                     f"within_{BUFFER}_after": best <= BUFFER})
    OUT.mkdir(parents=True, exist_ok=True)
    import csv
    with open(OUT / "neighbour_distance.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (OUT / f"sam3_map_farthest_r{REPEAT}_{SIZE}.json").write_text(json.dumps(
        {"tag": TAG, "model": MODEL, "repeat": REPEAT, "rule": "fold model of repeat 0 whose train+val frames are farthest in frame index (index only)",
         "n_images": len(images), "images": images}, indent=1), encoding="utf-8")

    def stats(sub, key):
        v = np.array([r[key] for r in sub])
        return {"n": len(v), "min": int(v.min()), "median": float(np.median(v)), "max": int(v.max()),
                "le1": int((v <= 1).sum()), f"le{BUFFER}": int((v <= BUFFER).sum())}
    pos = [r for r in rows if r["positive"]]
    summary = {"buffer": BUFFER,
               "positives_r1": stats(pos, "r1_distance"), "positives_farthest": stats(pos, "farthest_distance"),
               "all_r1": stats(rows, "r1_distance"), "all_farthest": stats(rows, "farthest_distance"),
               "negatives_farthest": stats([r for r in rows if not r["positive"]], "farthest_distance"),
               "fold_changed": sum(r["r1_fold"] != r["farthest_fold"] for r in rows),
               "positives_within_buffer_after": [r["image"] for r in pos if r[f"within_{BUFFER}_after"]]}
    (OUT / "neighbour_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    for r in pos:
        print(r["image"], "pool" if r["in_pool"] else "    ", "r1 fold", r["r1_fold"], "dist", r["r1_distance"], "-> fold", r["farthest_fold"], "dist", r["farthest_distance"])


if __name__ == "__main__":
    main()
