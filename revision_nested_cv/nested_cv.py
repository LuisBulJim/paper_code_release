#!/usr/bin/env python3
"""
nested_cv.py -- leakage-free repeated cross-validation for underwater corrosion detection.

Protocol (per repeat r, seed = base_seed + r):
  * OUTER: stratified 5-fold over ALL images (positive / negative). The outer fold is the TEST
    set of that run. It is never used for training, early stopping or checkpoint selection.
  * INNER: a stratified `val_frac` split of the outer-train images is the VALIDATION set. It is
    the only data used by Ultralytics for early stopping / best.pt selection.
  * Every image is a test image exactly once per repeat, so out-of-fold predictions cover the
    whole dataset (N=261) and confidence intervals can be computed by resampling images.
  * All model sizes and both transfer protocols share the same splits and the same
    hyper-parameters (only the initial weights differ).
For every trained model, raw low-confidence predictions are stored for all TTA views on the
validation and test images; fusion / thresholding / metrics are done later by evaluate_cv.py
(CPU only), so thresholds can be varied without retraining.

Usage:
  python nested_cv.py --data-root dataset_yolo --out nested_out --repeats 3 --dry-run
  python nested_cv.py --data-root dataset_yolo --out nested_out --repeats 3 --protocol direct
  python nested_cv.py --data-root dataset_yolo --out nested_out --repeats 3 --protocol sequential
"""
import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from common import SIZES, collect_pool, content_duplicates, read_gt, view_key  # noqa: E402


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="dataset_yolo")
    ap.add_argument("--out", default="nested_out")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--sizes", nargs="+", default=SIZES)
    ap.add_argument("--protocol", choices=["direct", "sequential"], default="direct")
    ap.add_argument("--tag", default="", help="suffix for outputs (e.g. a control experiment)")
    ap.add_argument("--groups-csv", default=None, help="optional CSV with columns name,group (video/sequence id)")
    # training hyper-parameters (identical for every model / protocol by default)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--optimizer", default="SGD")
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--lrf", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--freeze", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    # weights
    ap.add_argument("--coco-weights", default="yolo26{s}.pt", help="{s} -> n/m/l")
    ap.add_argument("--seq-init", default="Sequential_Transfer_Benchmark/phase1_aerial_yolo26{s}/weights/best.pt")
    # TTA views to cache
    ap.add_argument("--scales", nargs="+", type=float, default=[0.9, 1.0, 1.1])
    ap.add_argument("--pred-conf", type=float, default=0.001)
    ap.add_argument("--dry-run", action="store_true", help="only build and check splits, train nothing")
    ap.add_argument("--keep-workspaces", action="store_true")
    ap.add_argument("--only-folds", nargs="+", type=int, default=None,
                    help="train only these outer folds (smoke test, e.g. --only-folds 0 --epochs 2)")
    ap.add_argument("--verify-splits", action="store_true",
                    help="do not train anything: compute the current splits (honouring --groups-csv "
                         "etc.) and report, per (repeat,fold,size), whether the existing .pkl in --out "
                         "(if any) was trained on a MATCHING split, is STALE (trained on a different "
                         "split -- e.g. left over from before a --groups-csv change), or MISSING. Run "
                         "this right after a training pass to directly confirm retraining actually "
                         "happened on the splits you intended, instead of discovering identical "
                         "results several steps later in evaluate_cv.py.")
    return ap.parse_args()


# ----------------------------------------------------------------------------- splits
def make_splits(items, repeat, args, groups=None):
    seed = args.base_seed + repeat
    names = np.array([i["name"] for i in items])
    y = np.array([int(i["pos"]) for i in items])
    idx = np.arange(len(items))
    if groups is not None:
        g = np.array([groups[n] for n in names])
        outer = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=seed).split(idx, y, g)
    else:
        outer = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed).split(idx, y)
    folds = []
    for k, (tr, te) in enumerate(outer):
        if groups is not None:
            n_in = max(2, int(round(1 / args.val_frac)))
            inner = StratifiedGroupKFold(n_splits=n_in, shuffle=True, random_state=seed * 100 + k)
            tr2, va2 = next(inner.split(tr, y[tr], g[tr]))
        else:
            tr2, va2 = train_test_split(np.arange(len(tr)), test_size=args.val_frac,
                                        stratify=y[tr], random_state=seed * 100 + k)
        folds.append(dict(train=names[tr[tr2]].tolist(), val=names[tr[va2]].tolist(), test=names[te].tolist()))
    # ---- hard checks: no leakage, every image tested exactly once
    all_test = []
    for f in folds:
        tr, va, te = set(f["train"]), set(f["val"]), set(f["test"])
        assert not (tr & va) and not (tr & te) and not (va & te), "overlap between train/val/test"
        assert len(tr) + len(va) + len(te) == len(items)
        all_test += f["test"]
    assert sorted(all_test) == sorted(names.tolist()), "each image must be in exactly one test fold"
    if groups is not None:  # groups must not straddle test and train
        gmap = groups
        for f in folds:
            gt = {gmap[n] for n in f["test"]}
            assert not (gt & {gmap[n] for n in f["train"] + f["val"]}), "group leakage"
    return folds


# ----------------------------------------------------------------------------- workspace
def build_workspace(ws, split, by_name):
    if ws.exists():
        shutil.rmtree(ws)
    for sp in ["train", "val"]:
        (ws / "images" / sp).mkdir(parents=True)
        (ws / "labels" / sp).mkdir(parents=True)
        for n in split[sp]:
            it = by_name[n]
            shutil.copy(it["img"], ws / "images" / sp / n)
            if it["lab"]:
                shutil.copy(it["lab"], ws / "labels" / sp / (Path(n).stem + ".txt"))
    (ws / "data.yaml").write_text(
        f"path: {ws.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: corrosion\n")
    return ws / "data.yaml"


# ----------------------------------------------------------------------------- inference
def _predict(model, img, imgsz, conf, device):
    r = model.predict(img, imgsz=imgsz, conf=conf, verbose=False, device=device)[0]
    if len(r.boxes) == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    return r.boxes.xyxyn.cpu().numpy().astype(np.float32), r.boxes.conf.cpu().numpy().astype(np.float32)


def predict_views(model, img_path, args):
    """Raw predictions of every TTA view (unfused, low confidence)."""
    import cv2
    img = cv2.imread(img_path)
    assert img is not None, img_path
    h, w = img.shape[:2]
    views = {}
    for s in args.scales:
        for flip in (False, True):
            # (a) 'paper' mode: resize the image, keep default imgsz  (what the released notebooks do)
            # (b) 'imgsz' mode: keep the image, change the network input size (true multi-scale)
            for mode in ("paper", "imgsz"):
                key = view_key(mode, s, flip)
                if key in views:
                    continue
                if mode == "paper":
                    v = cv2.resize(img, (int(w * s), int(h * s))) if abs(s - 1) > 1e-9 else img
                    isz = args.imgsz
                else:
                    v = img
                    isz = int(round(args.imgsz * s / 32) * 32)
                if flip:
                    v = cv2.flip(v, 1)
                b, sc = _predict(model, v, isz, args.pred_conf, args.device)
                if flip and len(b):
                    b = b.copy()
                    b[:, [0, 2]] = 1.0 - b[:, [2, 0]]
                views[key] = (b, sc)
    return views


def _split_fingerprint(split):
    """Order-independent identity of a fold's train/val/test image assignment, used to tell a
    genuine resume-after-interruption (same split, safe to reuse) apart from a split that changed
    underneath an unchanged output directory (e.g. a different --groups-csv re-run into the same
    --out) -- which must NOT be silently skipped, even though the .pkl filename looks identical."""
    return {part: tuple(sorted(split[part])) for part in ("train", "val", "test")}


def run_one(args, out, tag, r, k, size, split, by_name, yaml_path):
    from ultralytics import YOLO
    pkl = out / "preds" / f"{tag}_r{r}_f{k}_{size}.pkl"
    fp = _split_fingerprint(split)
    if pkl.exists():
        # BUGFIX: an earlier version skipped retraining whenever a .pkl of this name already
        # existed, with NO check that the split it was trained on still matches the split being
        # requested now. Re-running with a different --groups-csv but the SAME --out directory
        # therefore silently reused the OLD (ungrouped) checkpoints and predictions -- the results
        # looked identical to the ungrouped run not because grouping had no effect, but because
        # grouping was never actually applied. Fixed by fingerprinting the train/val/test image
        # sets: only skip if they match exactly; otherwise retrain and overwrite, regardless of
        # the output directory or filename.
        try:
            old_fp = pickle.load(open(pkl, "rb"))["meta"].get("split_fingerprint")
        except Exception:
            old_fp = None
        if old_fp == fp:
            print("  skip (exists, same split):", pkl.name)
            return
        else:
            print("  RETRAINING (split changed since last run, old .pkl is stale):", pkl.name)
    init = (args.coco_weights if args.protocol == "direct" else args.seq_init).format(s=size)
    seed = args.base_seed + r
    name = f"{tag}_r{r}_f{k}_{size}"
    kw = dict(data=str(yaml_path), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
              optimizer=args.optimizer, lr0=args.lr0, lrf=args.lrf, momentum=0.937,
              weight_decay=0.0005, patience=args.patience, seed=seed, deterministic=True,
              project=str((out / "runs").resolve()), name=name, exist_ok=True, plots=False,
              verbose=False, workers=args.workers, device=args.device)
    if args.freeze is not None:
        kw["freeze"] = args.freeze
    model = YOLO(init)
    res = model.train(**kw)
    best = Path(res.save_dir) / "weights" / "best.pt"
    (Path(res.save_dir) / "weights" / "last.pt").unlink(missing_ok=True)  # halve disk usage
    model = YOLO(str(best))
    rec = dict(meta=dict(tag=tag, repeat=r, fold=k, size=size, protocol=args.protocol, seed=seed, init=init,
                         train_args=kw, best=str(best), n_train=len(split["train"]),
                         n_val=len(split["val"]), n_test=len(split["test"]),
                         split_fingerprint=fp),
               val={}, test={}, gt={})
    for part in ("val", "test"):
        for n in split[part]:
            rec[part][n] = predict_views(model, by_name[n]["img"], args)
            rec["gt"][n] = read_gt(by_name[n]["lab"])
    (out / "preds").mkdir(parents=True, exist_ok=True)
    with open(pkl, "wb") as f:
        pickle.dump(rec, f)
    print("  saved:", pkl.name)


def main():
    args = parse()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.protocol + (f"_{args.tag}" if args.tag else "")
    items = collect_pool(args.data_root)
    by_name = {i["name"]: i for i in items}
    print(f"pool: {len(items)} images, {sum(i['pos'] for i in items)} positive, "
          f"{sum(not i['pos'] for i in items)} negative")
    dups = content_duplicates(items)
    if dups:
        print(f"WARNING: {len(dups)} images with identical content (they could straddle folds): {dups[:5]}")
    groups = None
    if args.groups_csv:
        import csv
        groups = {row["name"]: row["group"] for row in csv.DictReader(open(args.groups_csv))}
        missing = [n for n in by_name if n not in groups]
        assert not missing, f"{len(missing)} images without group, e.g. {missing[:3]}"
        print(f"grouped folds: {len(set(groups.values()))} groups")
    else:
        print("WARNING: no --groups-csv, folds are NOT grouped by video/sequence "
              "(adjacent frames may straddle train and test).")

    all_splits = {}
    for r in range(args.repeats):
        all_splits[r] = make_splits(items, r, args, groups)
        n_pos = [sum(by_name[n]["pos"] for n in f["test"]) for f in all_splits[r]]
        print(f"repeat {r} (seed {args.base_seed + r}): test sizes "
              f"{[len(f['test']) for f in all_splits[r]]}, test positives {n_pos}")
    with open(out / "splits.json", "w") as f:
        json.dump({str(r): v for r, v in all_splits.items()}, f)
    print("splits OK (no overlap, every image tested once per repeat) ->", out / "splits.json")

    if args.dry_run:
        for s in args.sizes:
            w = (args.coco_weights if args.protocol == "direct" else args.seq_init).format(s=s)
            print(f"  init weights for size {s}: {w}  ->", "found" if Path(w).exists() else "not found locally")
        print("dry-run: nothing trained.")
        return

    if args.verify_splits:
        tag_v = args.protocol + (f"_{args.tag}" if args.tag else "")
        counts = {"match": 0, "stale": 0, "missing": 0}
        for r in range(args.repeats):
            for k, split in enumerate(all_splits[r]):
                fp = _split_fingerprint(split)
                for size in args.sizes:
                    pkl = out / "preds" / f"{tag_v}_r{r}_f{k}_{size}.pkl"
                    if not pkl.exists():
                        print(f"  MISSING  {pkl.name}"); counts["missing"] += 1; continue
                    try:
                        old_fp = pickle.load(open(pkl, "rb"))["meta"].get("split_fingerprint")
                    except Exception:
                        old_fp = "<sin fingerprint -- generado por una version anterior del script>"
                    if old_fp == fp:
                        counts["match"] += 1
                    else:
                        print(f"  STALE    {pkl.name}  (split guardado no coincide con el actual)")
                        counts["stale"] += 1
        print(f"\nresumen: {counts['match']} al dia, {counts['stale']} desactualizados (hace falta "
              f"reentrenar), {counts['missing']} sin entrenar todavia.")
        if counts["stale"] > 0:
            print("Relanzad 'direct'/'sequential' (con el mismo --groups-csv que querais) para "
                  "reentrenar SOLO los desactualizados -- el resto se salta automaticamente.")
        return

    for r in range(args.repeats):
        for k, split in enumerate(all_splits[r]):
            if args.only_folds is not None and k not in args.only_folds:
                continue
            ws = out / "workspaces" / f"r{r}_f{k}"
            yaml_path = build_workspace(ws, split, by_name)
            for size in args.sizes:
                print(f"[{tag}] repeat {r} fold {k} size {size}")
                run_one(args, out, tag, r, k, size, split, by_name, yaml_path)
            if not args.keep_workspaces:
                shutil.rmtree(ws, ignore_errors=True)


if __name__ == "__main__":
    main()
