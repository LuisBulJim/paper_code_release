"""R2.1(a)(b) -- retrain YOLO26-Medium (direct transfer) under stricter separation between test and training frames.

Both settings reuse revision_nested_cv/nested_cv.py unchanged (make_splits, build_workspace, run_one): same recipe,
same COCO initialisation, same seeds (42 + repeat), same stored views (std, flip, 0.9/1.1 x flip), so the r1
evaluator scores them exactly like the paper's baseline.

  purge B  (a) the paper's outer folds and inner validation splits (out/splits.json); every frame within B frame
           indices of ANY outer-test image of the fold is removed from training and from inner validation,
           B in {5, 15, 30}. Test images are unchanged, so the paired comparison with the paper's baseline is on the
           same images.
  gap30    (b) new grouping: cut only where the gap between consecutive frame indices exceeds 30, no cap on group
           size (17 groups); StratifiedGroupKFold k=5 with seeds 42-44, inner validation as nested_cv.py.

Outputs go to results_rev2/r2_1/<setting>/ (splits.json, preds/direct_r{r}_f{k}_m.pkl, runs/, losses.csv).
Nothing under revision_nested_cv/ is written. Resumable: a finished (r, k) with the same split is skipped.

Usage (from the repository root):
  python results_rev2/r2_1ab_train.py --setting purge5 --dry-run
  python results_rev2/r2_1ab_train.py --setting purge5
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "revision_nested_cv"))
import nested_cv as nc  # noqa: E402
from common import collect_pool  # noqa: E402

SETTINGS = ["purge5", "purge15", "purge30", "gap30"]
REPEATS, FOLDS, SIZE, TAG = 3, 5, "m", "direct"


def num(n: str) -> int:
    return int("".join(c for c in Path(n).stem if c.isdigit()))


def recipe_args() -> argparse.Namespace:
    """nested_cv.py defaults, checked against the training arguments stored with the paper's baseline."""
    argv, sys.argv = sys.argv, [sys.argv[0], "--data-root", "dataset_yolo", "--protocol", "direct", "--sizes", "m"]
    try:
        a = nc.parse()
    finally:
        sys.argv = argv
    ref = pickle.load(open(REPO / "revision_nested_cv/out/preds/direct_r0_f0_m.pkl", "rb"))["meta"]["train_args"]
    for key in ("epochs", "imgsz", "batch", "optimizer", "lr0", "lrf", "patience", "workers", "device"):
        assert getattr(a, key) == ref[key], f"{key}: {getattr(a, key)} != paper {ref[key]}"
    return a


def purged_splits(B: int) -> dict:
    paper = json.load(open(REPO / "revision_nested_cv/out/splits.json"))
    out = {}
    for r, folds in paper.items():
        out[r] = []
        for f in folds:
            test_ids = np.array(sorted(num(n) for n in f["test"]))
            near = lambda n: int(np.min(np.abs(test_ids - num(n)))) <= B  # noqa: E731
            out[r].append({"train": [n for n in f["train"] if not near(n)], "val": [n for n in f["val"] if not near(n)],
                           "test": list(f["test"]),
                           "removed_train": [n for n in f["train"] if near(n)], "removed_val": [n for n in f["val"] if near(n)]})
    return out


def coarse_splits(items: list[dict], args: argparse.Namespace, threshold: int = 30) -> tuple[dict, dict]:
    order = sorted((i["name"] for i in items), key=num)
    groups, g = {}, 0
    for i, n in enumerate(order):
        if i and num(n) - num(order[i - 1]) > threshold:
            g += 1
        groups[n] = f"sc{g:02d}"
    return {str(r): nc.make_splits(items, r, args, groups) for r in range(REPEATS)}, groups


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setting", choices=SETTINGS, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="+", default=None, help="r,k pairs (smoke tests)")
    ap.add_argument("--epochs", type=int, default=None, help="smoke tests only; the real runs keep 50")
    cli = ap.parse_args()
    import os
    os.chdir(REPO)  # the paper's runs were launched from the repository root (data_root dataset_yolo, yolo26m.pt)
    args = recipe_args()
    if cli.epochs:
        args.epochs = cli.epochs
    items = collect_pool(args.data_root)
    by_name = {i["name"]: i for i in items}
    out = REPO / "results_rev2" / "r2_1" / cli.setting
    if cli.setting.startswith("purge"):
        splits = purged_splits(int(cli.setting[5:]))
        groups = None
    else:
        splits, groups = coarse_splits(items, args)

    rows = []
    for r, folds in splits.items():
        for k, f in enumerate(folds):
            pos = lambda ns: sum(by_name[n]["pos"] for n in ns)  # noqa: E731
            rows.append({"repeat": r, "fold": k, "n_test": len(f["test"]), "n_test_pos": pos(f["test"]),
                         "n_train": len(f["train"]), "n_train_pos": pos(f["train"]), "n_val": len(f["val"]), "n_val_pos": pos(f["val"]),
                         "removed_train": len(f.get("removed_train", [])), "removed_val": len(f.get("removed_val", []))})
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "losses.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    clean = {r: [{p: f[p] for p in ("train", "val", "test")} for f in folds] for r, folds in splits.items()}
    (out / "splits.json").write_text(json.dumps(clean), encoding="utf-8")
    if groups:
        (out / "groups.json").write_text(json.dumps(groups, indent=0), encoding="utf-8")
    for row in rows:
        print(row)
    bad = [row for row in rows if row["n_val_pos"] == 0 or row["n_train_pos"] == 0]
    if bad:
        print(f"WARNING: {len(bad)} folds without positives in train or validation: {[(b['repeat'], b['fold']) for b in bad]}")
    if cli.dry_run:
        print("dry run: nothing trained")
        return

    only = {tuple(p.split(",")) for p in cli.only} if cli.only else None
    for r, folds in clean.items():
        for k, split in enumerate(folds):
            if only and (r, str(k)) not in only:
                continue
            ws = out / "workspaces" / f"r{r}_f{k}"
            yaml_path = nc.build_workspace(ws, split, by_name)
            print(f"[{cli.setting}] repeat {r} fold {k}: train {len(split['train'])} val {len(split['val'])} test {len(split['test'])}", flush=True)
            nc.run_one(args, out, TAG, int(r), k, SIZE, split, by_name, yaml_path)
            shutil.rmtree(ws, ignore_errors=True)
    print(f"[{cli.setting}] finished")


if __name__ == "__main__":
    main()
