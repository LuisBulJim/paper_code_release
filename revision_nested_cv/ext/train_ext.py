"""Train the pre-registered ext/ variants of YOLO26-m on the grouped nested-CV splits and store predictions.

Every variant reuses the outer test and inner validation images of out/splits.json (the grouped run), the
seeds of nested_cv.py (42 + repeat) and its record format, so ext/evaluate_ext.py scores all variants against
the baseline on identical images. Only the factor named by the variant changes.

  x1_recipe   official Ultralytics small-dataset recipe for YOLO26: AdamW, lr0 0.001, mosaic 0.5, patience 20
  x2_multi    9 LIACI classes (corrosion + 8 others, ship_hull dropped) and the LIACI images outside the pool,
              kept only if farther than 15 filename indices from every test and validation image of the fold;
              predictions keep class 0 (corrosion) only
  x3_seg      YOLO26-m-seg trained on the corrosion polygons (one per evaluated box); boxes come from the seg head
  x4_seg_cp   x3_seg + copy_paste 0.5 (Ultralytics 'flip' mode: objects pasted inside the same image)
  x5_combined union of the variants listed in ext/results/decision.json (those that beat the baseline on validation)
  x6_multi_seg x2_multi with polygons: YOLO26-m-seg on the 9 classes and the extra images (added after the first round)

Usage (server, from revision_nested_cv/, env with ultralytics 8.4.5):
  python ext/train_ext.py --variant x1_recipe --pool ../dataset_yolo
  python ext/train_ext.py --variant x2_multi --pool ../dataset_yolo --dry-run    # build and check workspaces only
  python ext/train_ext.py --combine x1_recipe x2_multi --pool ../dataset_yolo
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from common import read_gt, view_key  # noqa: E402

BASE = dict(epochs=50, imgsz=640, batch=16, optimizer="SGD", lr0=0.01, lrf=0.1, momentum=0.937, weight_decay=0.0005, patience=100)
VARIANTS = {
    "x1_recipe": dict(weights="yolo26m.pt", labels="det", extra=False, train=dict(optimizer="AdamW", lr0=0.001, mosaic=0.5, mixup=0.0, copy_paste=0.0, patience=20)),
    "x2_multi": dict(weights="yolo26m.pt", labels="multi", extra=True, train={}),
    "x3_seg": dict(weights="yolo26m-seg.pt", labels="seg", extra=False, train={}),
    "x4_seg_cp": dict(weights="yolo26m-seg.pt", labels="seg", extra=False, train=dict(copy_paste=0.5)),
    "x6_multi_seg": dict(weights="yolo26m-seg.pt", labels="seg_multi", extra=True, train={}),
}
LABEL_DIRS = {"multi": "labels_multi", "seg": "labels_seg", "seg_multi": "labels_seg_multi"}
MULTI_CLASS = {"multi", "seg_multi"}
POLYGONS = {"seg", "seg_multi"}
CLASSES = ["corrosion", "paint_peel", "marine_growth", "sea_chest_grating", "anode", "propeller", "over_board_valve", "bilge_keel", "defect"]
SCALES = (0.9, 1.0, 1.1)


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(VARIANTS))
    ap.add_argument("--combine", nargs="+", choices=sorted(VARIANTS), help="train the union of these variants as x5_combined")
    ap.add_argument("--pool", required=True, type=Path, help="dataset_yolo with the 261 images (images/{train,test})")
    ap.add_argument("--splits", default=ROOT / "out" / "splits.json", type=Path)
    ap.add_argument("--data", default=HERE / "data", type=Path, help="output of ext/build_data.py")
    ap.add_argument("--out", default=HERE / "out", type=Path)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only", nargs="+", default=None, help="subset of r,k pairs, e.g. 0,0 1,3 (smoke tests)")
    ap.add_argument("--epochs", type=int, default=None, help="override (smoke tests only)")
    ap.add_argument("--max-extra", type=int, default=None, help="cap on extra images per fold (smoke tests only)")
    ap.add_argument("--max-train", type=int, default=None, help="cap on pool training images per fold (smoke tests only)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pred-conf", type=float, default=0.001)
    ap.add_argument("--dry-run", action="store_true", help="build and check the workspaces, train nothing")
    ap.add_argument("--keep-workspaces", action="store_true")
    return ap.parse_args()


def combine(names: list[str]) -> dict:
    """Union of variants; the recipe is applied first so a later factor (copy_paste of x4) overrides it."""
    labels = {VARIANTS[n]["labels"] for n in names} - {"det"}
    if len(labels) > 1:
        raise SystemExit(f"cannot combine {sorted(names)}: there are no multi-class polygons for {sorted(labels)} together")
    label = labels.pop() if labels else "det"
    train = {}
    for n in sorted(names):
        train.update(VARIANTS[n]["train"])
    return dict(weights="yolo26m-seg.pt" if label in POLYGONS else "yolo26m.pt", labels=label, extra=any(VARIANTS[n]["extra"] for n in names),
                train=train, combine=sorted(names))


def pool_index(pool: Path) -> dict[str, dict]:
    out = {}
    for split in ("train", "val", "test"):
        for img in sorted((pool / "images" / split).glob("*.jpg")):
            out[img.name] = {"img": img, "lab": pool / "labels" / split / f"{img.stem}.txt"}
    return out


def label_source(variant: dict, data: Path, name: str, pool: dict[str, dict]) -> Path | None:
    stem = Path(name).stem
    if variant["labels"] == "det":
        return pool[name]["lab"]
    return data / LABEL_DIRS[variant["labels"]] / f"{stem}.txt"


def build_workspace(ws: Path, split: dict, extra: list[str], variant: dict, data: Path, pool: dict[str, dict]) -> tuple[Path, dict]:
    if ws.exists():
        shutil.rmtree(ws)
    counts = {"train": 0, "val": 0, "extra": 0, "boxes_train": 0}
    for part in ("train", "val"):
        (ws / "images" / part).mkdir(parents=True)
        (ws / "labels" / part).mkdir(parents=True)
        names = split[part] + (extra if part == "train" else [])
        for n in names:
            src = pool[n]["img"] if n in pool else data / "extra" / "images" / n
            if not src.exists():
                raise FileNotFoundError(src)
            shutil.copy(src, ws / "images" / part / n)
            lab = label_source(variant, data, n, pool)
            if lab is not None and lab.exists():
                text = lab.read_text()
                (ws / "labels" / part / f"{Path(n).stem}.txt").write_text(text)
                if part == "train":
                    counts["boxes_train"] += sum(1 for l in text.splitlines() if l.strip())
            counts[part] += 1
        counts["extra"] = len(extra)
    names = CLASSES if variant["labels"] in MULTI_CLASS else ["corrosion"]
    yaml = ws / "data.yaml"
    yaml.write_text(f"path: {ws.resolve()}\ntrain: images/train\nval: images/val\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
    return yaml, counts


def check_workspace(ws: Path, variant: dict) -> None:
    """Labels parse, classes are in range and polygons have at least three points."""
    n_cls = len(CLASSES) if variant["labels"] in MULTI_CLASS else 1
    for lab in (ws / "labels").rglob("*.txt"):
        for line in lab.read_text().splitlines():
            if not line.strip():
                continue
            vals = line.split()
            cls = int(vals[0])
            coords = np.array(vals[1:], dtype=float)
            assert 0 <= cls < n_cls, f"{lab}: class {cls}"
            assert np.all((coords >= 0) & (coords <= 1)), f"{lab}: coordinate outside [0, 1]"
            if variant["labels"] in POLYGONS:
                assert len(coords) >= 6 and len(coords) % 2 == 0, f"{lab}: bad polygon"
            else:
                assert len(coords) == 4, f"{lab}: bad box"


def predict(model, img_path: Path, imgsz: int, conf: float, device, classes: list[int] | None):
    import cv2

    img = cv2.imread(str(img_path))
    assert img is not None, img_path
    views = {}
    for s in SCALES:
        for flip in (False, True):
            key = view_key("imgsz", s, flip)
            v = cv2.flip(img, 1) if flip else img
            r = model.predict(v, imgsz=int(round(imgsz * s / 32) * 32), conf=conf, verbose=False, device=device, classes=classes)[0]
            if len(r.boxes) == 0:
                b, sc = np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
            else:
                b = r.boxes.xyxyn.cpu().numpy().astype(np.float32)
                sc = r.boxes.conf.cpu().numpy().astype(np.float32)
                if flip:
                    b = b.copy()
                    b[:, [0, 2]] = 1.0 - b[:, [2, 0]]
            views[key] = (b, sc)
    return views


def run_one(args, variant_name: str, variant: dict, r: int, k: int, split: dict, extra: list[str], pool: dict[str, dict]) -> None:
    from ultralytics import YOLO

    pkl = args.out / "preds" / f"{variant_name}_r{r}_f{k}_m.pkl"
    if pkl.exists():
        old = pickle.load(open(pkl, "rb"))["meta"]
        if (old.get("test") == sorted(split["test"]) and old.get("val") == sorted(split["val"]) and old.get("epochs") == (args.epochs or BASE["epochs"])
                and old.get("combine") == variant.get("combine")):
            print("  skip (exists, same split):", pkl.name)
            return
    ws = args.out / "workspaces" / f"{variant_name}_r{r}_f{k}"
    yaml, counts = build_workspace(ws, split, extra, variant, args.data, pool)
    check_workspace(ws, variant)
    kw = dict(BASE, **variant["train"])
    if args.epochs:
        kw["epochs"] = args.epochs
    kw.update(data=str(yaml), seed=42 + r, deterministic=True, project=str((args.out / "runs").resolve()), name=f"{variant_name}_r{r}_f{k}_m",
              exist_ok=True, plots=False, verbose=False, workers=args.workers, device=args.device)
    model = YOLO(variant["weights"])
    res = model.train(**kw)
    best = Path(res.save_dir) / "weights" / "best.pt"
    (Path(res.save_dir) / "weights" / "last.pt").unlink(missing_ok=True)
    model = YOLO(str(best))
    classes = [0] if variant["labels"] in MULTI_CLASS else None
    rec = {"meta": {"tag": variant_name, "repeat": r, "fold": k, "size": "m", "seed": 42 + r, "init": variant["weights"], "train_args": kw,
                    "best": str(best), "counts": counts, "test": sorted(split["test"]), "val": sorted(split["val"]), "epochs": kw["epochs"],
                    "combine": variant.get("combine")},
           "val": {}, "test": {}, "gt": {}}
    for part in ("val", "test"):
        for n in split[part]:
            rec[part][n] = predict(model, pool[n]["img"], kw["imgsz"], args.pred_conf, args.device, classes)
            rec["gt"][n] = read_gt(str(pool[n]["lab"]))
    pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(pkl, "wb") as f:
        pickle.dump(rec, f)
    print("  saved:", pkl.name, counts)
    if not args.keep_workspaces:
        shutil.rmtree(ws, ignore_errors=True)


def main() -> None:
    args = parse()
    if bool(args.variant) == bool(args.combine):
        raise SystemExit("give exactly one of --variant or --combine")
    name = args.variant or "x5_combined"
    variant = VARIANTS[args.variant] if args.variant else combine(args.combine)
    same = [n for n, v in VARIANTS.items() if args.combine and v == {k: variant[k] for k in v}]
    if same:
        print(f"the union of {args.combine} equals {same[0]}, already trained: nothing to do")
        return
    pool = pool_index(args.pool)
    assert len(pool) == 261, f"pool has {len(pool)} images, expected 261"
    splits = json.load(open(args.splits))
    folds = json.load(open(args.data / "folds.json")) if variant["extra"] else None
    only = {tuple(int(x) for x in p.split(",")) for p in args.only} if args.only else None
    for r in range(args.repeats):
        for k in range(args.folds):
            if only and (r, k) not in only:
                continue
            split = splits[str(r)][k]
            extra = folds[str(r)][k]["extra_train"] if folds else []
            if args.max_extra is not None:
                extra = extra[: args.max_extra]
            if args.max_train is not None:
                split = dict(split, train=split["train"][: args.max_train])
            print(f"[{name}] repeat {r} fold {k}: train {len(split['train'])} + extra {len(extra)}, val {len(split['val'])}, test {len(split['test'])}")
            if args.dry_run:
                ws = args.out / "workspaces" / f"dry_{name}_r{r}_f{k}"
                _, counts = build_workspace(ws, split, extra, variant, args.data, pool)
                check_workspace(ws, variant)
                print("  workspace OK", counts)
                shutil.rmtree(ws, ignore_errors=True)
                continue
            run_one(args, name, variant, r, k, split, extra, pool)


if __name__ == "__main__":
    main()
