"""Pixel-level evaluation of corrosion: masks of the segmentation variants against filled boxes of every detector.

Corrosion is an amorphous region (R2.9), so besides box mAP the outputs are scored the way the LIACI benchmark
scores segmentation: pixel IoU of the corrosion class, on the same out-of-fold test images for every model.

Rules fixed before any mask was scored:
  * ground truth: union of the corrosion polygons in ext/data/labels_seg (one per evaluated box), rasterised at image size
  * output of a model: union of its masks (x3_seg, x4_seg_cp) or of its filled boxes (every model), standard pass only
  * confidence threshold chosen per model and output on its inner validation (max mean pixel IoU over CONF_GRID)
  * primary metric: dataset-level pixel IoU over the out-of-fold test images (mean of the 3 repeats); 95 % intervals
    and paired differences against the baseline's filled boxes by cluster bootstrap over the video groups, as in r1
  * reference row: filled ground-truth boxes, the best any box output can reach

Usage (from revision_nested_cv/):
  python ext/pixel_eval.py predict --pool ../dataset_yolo    # GPU, minutes: masks of the x3/x4 checkpoints -> ext/out/masks
  python ext/pixel_eval.py metrics --pool ../dataset_yolo    # CPU: ext/results/pixel_*.csv and table_pixel.tex
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from r1.nested_eval import cluster_samples, reconstruct_groups  # noqa: E402
from r1.protocol import load_protocol  # noqa: E402
from r1.sweep_inference import CONF_GRID  # noqa: E402
from train_ext import MULTI_CLASS, VARIANTS, pool_index  # noqa: E402

REPEATS, FOLDS = 3, 5
SEG_VARIANTS = ["x3_seg", "x4_seg_cp", "x6_multi_seg"]
BOX_MODELS = {"x0_baseline": str(ROOT / "out" / "preds" / "direct_r{r}_f{k}_m.pkl")}
BOX_MODELS.update({v: str(HERE / "out" / "preds" / f"{v}_r{{r}}_f{{k}}_m.pkl") for v in ("x1_recipe", "x2_multi", "x3_seg", "x4_seg_cp", "x6_multi_seg", "e1_ensemble", "c4_baselines")})
REFERENCE = ("x0_baseline", "boxes")
METRICS = ("iou", "dice", "precision", "recall")
LABELS = {"x0_baseline": "Baseline (direct transfer)", "x1_recipe": "Small-data recipe", "x2_multi": "9 classes + unused LIACI images",
          "x3_seg": "Segmentation head", "x4_seg_cp": "Segmentation + copy-paste", "x6_multi_seg": "9 classes as polygons + unused images",
          "e1_ensemble": "Ensemble of the four variants (WBF)", "c4_baselines": "Control: ensemble of four baseline models (WBF)", "gt": "Ground-truth boxes (bound)"}


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["predict", "metrics"])
    ap.add_argument("--pool", required=True, type=Path, help="dataset_yolo with the 261 images")
    ap.add_argument("--splits", default=ROOT / "out" / "splits.json", type=Path)
    ap.add_argument("--data", default=HERE / "data", type=Path)
    ap.add_argument("--masks", default=HERE / "out" / "masks", type=Path)
    ap.add_argument("--out", default=HERE / "results", type=Path)
    ap.add_argument("--variants", nargs="+", default=SEG_VARIANTS, choices=SEG_VARIANTS)
    ap.add_argument("--weights", default=str(HERE / "out" / "runs" / "{v}_r{r}_f{k}_m" / "weights" / "best.pt"))
    ap.add_argument("--only", nargs="+", default=None, help="subset of r,k pairs (smoke tests)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--iters", type=int, default=None)
    return ap.parse_args()


def gt_mask(name: str, h: int, w: int, data: Path) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    lab = data / "labels_seg" / f"{Path(name).stem}.txt"
    if lab.exists():
        for line in lab.read_text().splitlines():
            v = line.split()
            if len(v) >= 7:
                pts = np.array(v[1:], dtype=float).reshape(-1, 2) * [w, h]
                cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    return m.astype(bool)


def box_fill(boxes: np.ndarray, h: int, w: int) -> np.ndarray:
    """Same rounding as ext/analysis/diagnostics.raster."""
    m = np.zeros((h, w), bool)
    for x1, y1, x2, y2 in np.asarray(boxes, dtype=float).reshape(-1, 4):
        m[max(int(round(y1 * h)), 0):min(int(round(y2 * h)), h), max(int(round(x1 * w)), 0):min(int(round(x2 * w)), w)] = True
    return m


def counts(pred: np.ndarray, gt: np.ndarray) -> tuple[int, int, int]:
    return int((pred & gt).sum()), int(pred.sum()), int(gt.sum())


def predict(args: argparse.Namespace) -> None:
    from pycocotools import mask as mask_utils
    from ultralytics import YOLO

    pool = pool_index(args.pool)
    splits = json.load(open(args.splits))
    only = {tuple(int(x) for x in p.split(",")) for p in args.only} if args.only else None
    args.masks.mkdir(parents=True, exist_ok=True)
    for v in args.variants:
        for r in range(REPEATS):
            for k in range(FOLDS):
                if only and (r, k) not in only:
                    continue
                dst = args.masks / f"{v}_r{r}_f{k}.pkl"
                split = splits[str(r)][k]
                if dst.exists() and pickle.load(open(dst, "rb"))["meta"]["test"] == sorted(split["test"]):
                    print("  skip (exists):", dst.name)
                    continue
                weights = Path(args.weights.format(v=v, r=r, k=k))
                model = YOLO(str(weights))
                classes = [0] if VARIANTS[v]["labels"] in MULTI_CLASS else None
                rec = {"meta": {"variant": v, "repeat": r, "fold": k, "weights": str(weights), "test": sorted(split["test"]), "val": sorted(split["val"])},
                       "conf_grid": list(CONF_GRID), "val": {}, "test": {}}
                for part in ("val", "test"):
                    for n in split[part]:
                        img = cv2.imread(str(pool[n]["img"]))
                        h, w = img.shape[:2]
                        res = model.predict(img, imgsz=640, conf=min(CONF_GRID), retina_masks=True, verbose=False, device=args.device, classes=classes)[0]
                        if res.masks is None or len(res.boxes) == 0:
                            masks, sc = np.zeros((0, h, w), bool), np.zeros(0)
                        else:
                            masks, sc = res.masks.data.cpu().numpy() > 0.5, res.boxes.conf.cpu().numpy()
                        assert masks.shape[1:] == (h, w), (n, masks.shape)
                        rec[part][n] = [mask_utils.encode(np.asfortranarray(masks[sc >= c].any(0).astype(np.uint8))) for c in CONF_GRID]
                with open(dst, "wb") as f:
                    pickle.dump(rec, f)
                print(f"  saved: {dst.name} ({len(rec['val'])} val, {len(rec['test'])} test)")


def fold_counts(splits: dict, load_fold) -> dict | None:
    """{(r, k): {"val"|"test": {name: array (len(CONF_GRID), 3) of intersection, predicted and GT pixels}}}."""
    out = {}
    for r in range(REPEATS):
        for k in range(FOLDS):
            rec = load_fold(r, k)
            if rec is None:
                return None
            split = splits[str(r)][k]
            if sorted(rec["test"]) != sorted(split["test"]):
                raise ValueError(f"r{r} f{k}: test images differ from splits.json")
            out[(r, k)] = {part: {n: rec["counts"](n, part) for n in split[part]} for part in ("val", "test")}
    return out


def box_loader(pattern: str, gts: dict, sizes: dict):
    def load(r: int, k: int):
        p = Path(pattern.format(r=r, k=k))
        if not p.exists():
            return None
        rec = pickle.load(open(p, "rb"))

        def cnt(n: str, part: str) -> np.ndarray:
            b, s = rec[part][n]["std"]
            b, s = np.asarray(b, dtype=float).reshape(-1, 4), np.asarray(s, dtype=float)
            h, w = sizes[n]
            return np.array([counts(box_fill(b[s >= c], h, w), gts[n]) for c in CONF_GRID])
        return {"test": rec["test"], "counts": cnt}
    return load


def mask_loader(masks_dir: Path, variant: str, gts: dict):
    from pycocotools import mask as mask_utils

    def load(r: int, k: int):
        p = masks_dir / f"{variant}_r{r}_f{k}.pkl"
        if not p.exists():
            return None
        rec = pickle.load(open(p, "rb"))
        assert rec["conf_grid"] == list(CONF_GRID), "masks stored with another confidence grid"

        def cnt(n: str, part: str) -> np.ndarray:
            return np.array([counts(mask_utils.decode(m).astype(bool), gts[n]) for m in rec[part][n]])
        return {"test": rec["test"], "counts": cnt}
    return load


def scores(tot: np.ndarray) -> dict[str, float]:
    i, p, g = (float(x) for x in tot)
    return {"iou": i / (p + g - i) if p + g - i else float("nan"), "dice": 2 * i / (p + g) if p + g else float("nan"),
            "precision": i / p if p else float("nan"), "recall": i / g if g else float("nan")}


def choose_conf(fc: dict) -> tuple[int, float]:
    """Index of the confidence with the best mean validation IoU (folds without corrosion pixels skipped)."""
    means = []
    for ci in range(len(CONF_GRID)):
        vals = []
        for parts in fc.values():
            tot = sum(parts["val"].values())
            if tot[ci][2] > 0:
                vals.append(scores(tot[ci])["iou"])
        means.append(float(np.mean(vals)))
    best = max(range(len(CONF_GRID)), key=lambda i: (round(means[i], 12), -abs(CONF_GRID[i] - 0.25)))
    return best, means[best]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def metrics(args: argparse.Namespace) -> None:
    proto = load_protocol()
    evc = proto["evaluation"]
    iters = args.iters or evc["bootstrap_iters"]
    pool = pool_index(args.pool)
    splits = json.load(open(args.splits))
    names = sorted({n for f in splits["0"] for n in f["test"]})
    sizes = {}
    for n in pool:
        img = cv2.imread(str(pool[n]["img"]))
        sizes[n] = img.shape[:2]
    gts = {n: gt_mask(n, *sizes[n], args.data) for n in pool}
    groups = reconstruct_groups(splits)
    positive = {n: bool(gts[n].any()) for n in names}
    samples, _ = cluster_samples(groups, names, positive, iters, evc["bootstrap_seed"])
    idx = {n: i for i, n in enumerate(names)}
    samples = [np.array([idx[n] for n in s]) for s in samples]

    models = {}
    for v, pat in BOX_MODELS.items():
        fc = fold_counts(splits, box_loader(pat, gts, sizes))
        if fc is None:
            print(f"skip {v} boxes: predictions incomplete ({pat})")
            continue
        models[(v, "boxes")] = fc
    for v in SEG_VARIANTS:
        fc = fold_counts(splits, mask_loader(args.masks, v, gts))
        if fc is None:
            print(f"skip {v} masks: masks incomplete ({args.masks})")
            continue
        models[(v, "masks")] = fc
    assert REFERENCE in models, "baseline predictions are needed as the reference"

    # per repeat: (n_images, 3) pixel counts of the out-of-fold test predictions at the validation threshold
    table, val_rows = {}, []
    gt_boxes = {}
    for k in range(FOLDS):
        gt_boxes.update(pickle.load(open(BOX_MODELS["x0_baseline"].format(r=0, k=k), "rb"))["gt"])
    gt_box = np.array([counts(box_fill(gt_boxes[n], *sizes[n]), gts[n]) for n in names])
    table[("gt", "boxes")] = (None, None, [gt_box] * REPEATS)
    for key, fc in models.items():
        ci, vmean = choose_conf(fc)
        per_rep = []
        for r in range(REPEATS):
            arr = np.zeros((len(names), 3))
            for k in range(FOLDS):
                for n, c in fc[(r, k)]["test"].items():
                    arr[idx[n]] = c[ci]
            per_rep.append(arr)
        table[key] = (CONF_GRID[ci], vmean, per_rep)
        val_rows.append({"model": key[0], "output": key[1], "conf_vt": CONF_GRID[ci], "val_iou_mean": round(vmean, 4)})

    def boot(per_rep: list[np.ndarray]) -> np.ndarray:
        return np.array([[np.mean([scores(a[s].sum(0))[m] for a in per_rep]) for m in METRICS] for s in samples])

    summary, deltas, boots = [], [], {}
    for key, (conf, vmean, per_rep) in table.items():
        full = {m: float(np.mean([scores(a.sum(0))[m] for a in per_rep])) for m in METRICS}
        boots[key] = boot(per_rep)
        row = {"model": key[0], "output": key[1], "conf_vt": conf, "val_iou_mean": None if vmean is None else round(vmean, 4)}
        for j, m in enumerate(METRICS):
            row.update({f"{m}_mean": round(full[m], 4), f"{m}_ci_lo": round(float(np.nanpercentile(boots[key][:, j], 2.5)), 4),
                        f"{m}_ci_hi": round(float(np.nanpercentile(boots[key][:, j], 97.5)), 4)})
        summary.append(row)
        print(f"{key[0]:12s} {key[1]:6s} conf {conf} | pixel IoU {row['iou_mean']:.4f} [{row['iou_ci_lo']:.4f}, {row['iou_ci_hi']:.4f}] | "
              f"Dice {row['dice_mean']:.4f} | P {row['precision_mean']:.4f} R {row['recall_mean']:.4f}")
    ref_full = {m: [scores(a.sum(0))[m] for a in table[REFERENCE][2]] for m in METRICS}
    for key, (conf, vmean, per_rep) in table.items():
        if key in (REFERENCE, ("gt", "boxes")):
            continue
        d = {"comparison": f"{key[0]} {key[1]} - {REFERENCE[0]} {REFERENCE[1]}"}
        for j, m in enumerate(METRICS):
            diff = boots[key][:, j] - boots[REFERENCE][:, j]
            point = float(np.mean([scores(a.sum(0))[m] for a in per_rep]) - np.mean(ref_full[m]))
            d.update({f"{m}_delta": round(point, 4), f"{m}_ci_lo": round(float(np.nanpercentile(diff, 2.5)), 4),
                      f"{m}_ci_hi": round(float(np.nanpercentile(diff, 97.5)), 4), f"{m}_p_le0": round(float(np.mean(diff <= 0)), 4)})
        deltas.append(d)

    write_csv(args.out / "pixel_validation.csv", val_rows)
    write_csv(args.out / "pixel_summary.csv", summary)
    if deltas:
        write_csv(args.out / "pixel_deltas.csv", deltas)
    dd = {d["comparison"].split(" - ")[0]: d for d in deltas}
    lines = ["% generated by ext/pixel_eval.py; differences are paired against the baseline's filled boxes",
             r"\begin{tabular}{llcccc}", r"\toprule",
             r"Model & Output & Pixel IoU [95\% CI] & $\Delta$IoU [95\% CI] & Precision & Recall \\", r"\midrule"]
    for row in summary:
        d = dd.get(f"{row['model']} {row['output']}")
        dtxt = "--" if d is None else f"{d['iou_delta']:+.3f} [{d['iou_ci_lo']:+.3f}, {d['iou_ci_hi']:+.3f}]"
        out = "filled boxes" if row["output"] == "boxes" else "masks"
        lines.append(f"{LABELS.get(row['model'], row['model'])} & {out} & {row['iou_mean']:.3f} [{row['iou_ci_lo']:.3f}, {row['iou_ci_hi']:.3f}] & {dtxt} & "
                     f"{row['precision_mean']:.3f} & {row['recall_mean']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (args.out / "table_pixel.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}/pixel_summary.csv, pixel_deltas.csv, pixel_validation.csv, table_pixel.tex ({iters} bootstrap replicates)")


def main() -> None:
    args = parse()
    predict(args) if args.step == "predict" else metrics(args)


if __name__ == "__main__":
    main()
