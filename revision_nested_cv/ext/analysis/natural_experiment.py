"""Same test images, with and without a neighbouring frame in training.

Pairs the grouped and the ungrouped run of YOLO26-m (same repeat, same image) and compares AP50 on
subsets defined by the filename-index distance to the nearest training image under each protocol.
Also breaks the pool down by resolution run (a lower bound on the number of source videos).
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from diagnostics import COCO, REPEATS, ROOT, ap_at  # noqa: E402
from similarity import folds, num  # noqa: E402

NEAR, FAR = 5, 15


def index_rows(suffix: str) -> dict[tuple[int, str], tuple[int, tuple]]:
    out = {}
    for f in folds(suffix):
        tr = np.array([num(n) for n in f["train"]])
        for n, rec in f["test"].items():
            out[(f["r"], n)] = (int(np.abs(tr - num(n)).min()), rec)
    return out


def ap_subset(rows: dict, keys: list[tuple[int, str]]) -> float:
    vals = []
    for r in range(REPEATS):
        sub = {n: rows[(rr, n)][1] for rr, n in keys if rr == r}
        if sub:
            vals.append(ap_at(sub, 0.5)[0])
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    g, u = index_rows(""), index_rows(".bak")
    keys = sorted(set(g) & set(u))
    print("paired evaluations:", len(keys))
    subsets = {
        "far in grouped, near in ungrouped": [k for k in keys if g[k][0] > FAR and u[k][0] <= NEAR],
        "near in both": [k for k in keys if g[k][0] <= NEAR and u[k][0] <= NEAR],
        "far in both": [k for k in keys if g[k][0] > FAR and u[k][0] > FAR],
        "all": keys,
    }
    for label, ks in subsets.items():
        n_gt = sum(len(g[k][1][2]) for k in ks)
        print(f"  {label:36s} images {len(ks):4d}  GT {n_gt:5d}  AP50 grouped {ap_subset(g, ks):.3f}  AP50 ungrouped {ap_subset(u, ks):.3f}")

    d = json.load(open(COCO, encoding="utf-8"))
    ims = sorted(d["images"], key=lambda im: num(im["file_name"]))
    run_of, start, n_runs = {}, 0, 0
    res = [(im["width"], im["height"]) for im in ims]
    for i in range(1, len(res) + 1):
        if i == len(res) or res[i] != res[start]:
            for j in range(start, i):
                run_of[ims[j]["file_name"]] = n_runs
            n_runs += 1
            start = i
    manifest = json.load(open(ROOT / "r1" / "data" / "nested_manifest.json"))["images"]
    pos = Counter(run_of[im["image"]] for im in manifest if im["positive"])
    boxes = Counter()
    for im in manifest:
        boxes[run_of[im["image"]]] += im["n_boxes"]
    total_pos, total_box = sum(pos.values()), sum(boxes.values())
    print("\nresolution runs with corrosion:", len(pos), "| positives", total_pos, "| boxes", total_box)
    cum = 0
    by_run = defaultdict(dict)
    for r in range(REPEATS):
        for (rr, n), (_, rec) in g.items():
            if rr == r:
                by_run[run_of[n]][f"{r}|{n}"] = rec
    for run, c in pos.most_common():
        cum += c
        recs = by_run[run]
        ap = float(np.mean([ap_at({k: v for k, v in recs.items() if k.startswith(f"{r}|")}, 0.5)[0] for r in range(REPEATS)]))
        print(f"  run {run:2d}: positives {c:3d} ({100 * c / total_pos:4.1f}%, cumulative {100 * cum / total_pos:5.1f}%)  boxes {boxes[run]:3d}  AP50 when tested {ap:.3f}")


if __name__ == "__main__":
    main()
