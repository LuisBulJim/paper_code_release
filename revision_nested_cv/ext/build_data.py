"""Build the data of the ext/ experiments from the full LIACI release (CPU only, run once locally).

Outputs, under ext/data/ (shipped to the server as one tarball):
  extra/images/      LIACI images outside the 261-image pool and without corrosion, resized to 640x640
                     exactly like dataset_yolo (cv2.resize, aspect ratio not kept)
  labels_multi/      YOLO detection labels for pool + extra: class 0 = corrosion (the dataset_yolo boxes,
                     unchanged), classes 1-8 = the other LIACI classes (annotator boxes); ship_hull dropped
  labels_seg/        YOLO polygon labels of corrosion for the pool (one polygon per connected component,
                     the same components that define the dataset_yolo boxes)
  folds.json         per (repeat, fold): extra images allowed for training (index buffer around test and val)
  manifest_ext.json  counts, consistency checks and sha256 of every generated file

Usage (from revision_nested_cv/):
  python ext/build_data.py --liaci-images <dir with image_XXXX.jpg> --liaci-masks <masks dir with corrosion/> \
      --coco <coco-labels.json> --pool <dataset_yolo> --splits out/splits.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
SIDE = 640
BUFFER = 15  # same threshold as the id-gap grouping of build_groups.py
CLASSES = ["corrosion", "paint_peel", "marine_growth", "sea_chest_grating", "anode", "propeller", "over_board_valve", "bilge_keel", "defect"]
DROPPED = {"ship_hull"}  # background surface under 97.5 % of the corrosion pixels, not an object


def num(name: str) -> int:
    return int(re.findall(r"\d+", name)[-1])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def imread(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED) if path.exists() else None
    if img is None:
        raise FileNotFoundError(path)
    return img


def pool_items(pool: Path) -> dict[str, dict]:
    items = {}
    for split in ("train", "val", "test"):
        for img in sorted((pool / "images" / split).glob("*.jpg")):
            lab = pool / "labels" / split / f"{img.stem}.txt"
            lines = [l for l in lab.read_text().splitlines() if l.strip()] if lab.exists() else []
            items[img.name] = {"img": img, "lines": lines}
    return items


def mask_components(mask_path: Path) -> list[np.ndarray]:
    """Polygons (pixel coords at 640x640) exactly as the submitted pipeline: nearest resize, >0, external contours."""
    mask = imread(mask_path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    small = cv2.resize(mask, (SIDE, SIDE), interpolation=cv2.INTER_NEAREST)
    _, binary = cv2.threshold(small.astype(np.uint8), 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.squeeze(axis=1).astype(float) for c in contours if c.shape[0] >= 3]


def box_lines(boxes: np.ndarray, cls: int) -> list[str]:
    out = []
    for x1, y1, x2, y2 in boxes:
        x1, x2 = np.clip([x1, x2], 0, SIDE)
        y1, y2 = np.clip([y1, y2], 0, SIDE)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        out.append(f"{cls} {(x1 + x2) / 2 / SIDE:.6f} {(y1 + y2) / 2 / SIDE:.6f} {(x2 - x1) / SIDE:.6f} {(y2 - y1) / SIDE:.6f}")
    return out


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def parse_box_line(line: str) -> np.ndarray:
    _, xc, yc, w, h = map(float, line.split()[:5])
    return np.array([(xc - w / 2) * SIDE, (yc - h / 2) * SIDE, (xc + w / 2) * SIDE, (yc + h / 2) * SIDE])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--liaci-images", required=True, type=Path)
    ap.add_argument("--liaci-masks", required=True, type=Path, help="folder with corrosion/image_XXXX.bmp")
    ap.add_argument("--coco", required=True, type=Path)
    ap.add_argument("--pool", required=True, type=Path, help="dataset_yolo of the nested CV (261 images)")
    ap.add_argument("--splits", required=True, type=Path, help="out/splits.json of the grouped nested CV")
    ap.add_argument("--out", default=HERE / "data", type=Path)
    args = ap.parse_args()

    out = args.out
    if out.exists():
        shutil.rmtree(out)
    for sub in ("extra/images", "labels_multi", "labels_seg"):
        (out / sub).mkdir(parents=True)

    coco = json.load(open(args.coco, encoding="utf-8"))
    names = {c["id"]: c["name"] for c in coco["categories"]}
    cls_of = {cid: CLASSES.index(n) for cid, n in names.items() if n in CLASSES}
    by_id = {im["id"]: im for im in coco["images"]}
    anns: dict[str, list[dict]] = {im["file_name"]: [] for im in coco["images"]}
    for a in coco["annotations"]:
        anns[by_id[a["image_id"]]["file_name"]].append(a)
    corrosion_id = next(cid for cid, n in names.items() if n == "corrosion")

    pool = pool_items(args.pool)
    has_corr = {n for n, al in anns.items() if any(a["category_id"] == corrosion_id for a in al)}
    extra, unreadable = [], []
    for n in sorted((n for n in anns if n not in pool and n not in has_corr), key=num):
        try:
            imread(args.liaci_images / n)
            extra.append(n)
        except FileNotFoundError:
            unreadable.append(n)
    report: dict = {"pool": len(pool), "extra": len(extra), "extra_unreadable_skipped": unreadable, "liaci_images": len(anns),
                    "excluded_corrosion_outside_pool": sorted(has_corr - set(pool), key=num),
                    "classes": CLASSES, "dropped": sorted(DROPPED), "buffer": BUFFER}

    # pool: corrosion boxes unchanged + polygons from the same masks; other classes from COCO
    count_mismatch, fallback, min_iou, n_poly = [], [], 1.0, 0
    for name, it in sorted(pool.items(), key=lambda kv: num(kv[0])):
        im = next(i for i in coco["images"] if i["file_name"] == name)
        sx, sy = SIDE / im["width"], SIDE / im["height"]
        lines = list(it["lines"])
        polys = mask_components(args.liaci_masks / "corrosion" / f"{Path(name).stem}.bmp")
        if len(polys) != len(lines):
            count_mismatch.append(name)
        seg = []
        for line in lines:  # one polygon per evaluated box, so seg and box ground truth stay aligned
            gt = parse_box_line(line)
            best, best_iou = None, 0.0
            for p in polys:
                pb = np.array([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()])
                iou = box_iou(gt, pb)
                if iou > best_iou:
                    best, best_iou = p, iou
            if best is None or best_iou < 0.5:
                x1, y1, x2, y2 = gt
                best = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                fallback.append(name)
            else:
                min_iou = min(min_iou, best_iou)
            seg.append("0 " + " ".join(f"{np.clip(x, 0, SIDE) / SIDE:.6f} {np.clip(y, 0, SIDE) / SIDE:.6f}" for x, y in best))
        n_poly += len(seg)
        (out / "labels_seg" / f"{Path(name).stem}.txt").write_text("\n".join(seg) + ("\n" if seg else ""))
        for a in anns[name]:
            n = names[a["category_id"]]
            if n in DROPPED or n == "corrosion":
                continue
            x, y, w, h = a["bbox"]
            lines += box_lines(np.array([[x * sx, y * sy, (x + w) * sx, (y + h) * sy]]), cls_of[a["category_id"]])
        (out / "labels_multi" / f"{Path(name).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    report["seg_polygons"] = n_poly
    report["pool_boxes"] = sum(len(it["lines"]) for it in pool.values())
    report["mask_components_vs_boxes_count_mismatch"] = count_mismatch
    report["seg_rectangle_fallbacks"] = {"boxes": len(fallback), "images": sorted(set(fallback), key=num)}
    report["seg_min_matched_box_iou"] = round(min_iou, 3)

    # extra images: resized exactly like dataset_yolo, labels of the other classes only
    per_class = Counter()
    for name in extra:
        im = next(i for i in coco["images"] if i["file_name"] == name)
        img = imread(args.liaci_images / name)
        cv2.imencode(".jpg", cv2.resize(img, (SIDE, SIDE)))[1].tofile(str(out / "extra" / "images" / name))
        sx, sy = SIDE / im["width"], SIDE / im["height"]
        lines = []
        for a in anns[name]:
            n = names[a["category_id"]]
            if n in DROPPED:
                continue
            x, y, w, h = a["bbox"]
            new = box_lines(np.array([[x * sx, y * sy, (x + w) * sx, (y + h) * sy]]), cls_of[a["category_id"]])
            per_class[n] += len(new)
            lines += new
        (out / "labels_multi" / f"{Path(name).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    report["extra_boxes_per_class"] = dict(per_class.most_common())
    report["extra_without_any_label"] = sum(1 for n in extra if not (out / "labels_multi" / f"{Path(n).stem}.txt").read_text().strip())

    # extra images allowed per fold: farther than BUFFER filename indices from every test and val image
    splits = json.load(open(args.splits))
    extra_ids = np.array([num(n) for n in extra])
    folds, sizes = {}, []
    for r, fs in splits.items():
        folds[r] = []
        for k, f in enumerate(fs):
            held = np.array(sorted(num(n) for n in f["test"] + f["val"]))
            dist = np.abs(extra_ids[:, None] - held[None, :]).min(axis=1)
            allowed = [n for n, d in zip(extra, dist) if d > BUFFER]
            folds[r].append({"extra_train": allowed})
            sizes.append(len(allowed))
    (out / "folds.json").write_text(json.dumps(folds))
    report["extra_train_per_fold"] = {"min": int(min(sizes)), "median": float(np.median(sizes)), "max": int(max(sizes))}

    files = sorted(p for p in out.rglob("*") if p.is_file())
    report["sha256"] = {str(p.relative_to(out)).replace("\\", "/"): sha256(p) for p in files}
    (out / "manifest_ext.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "sha256"}, indent=1))


if __name__ == "__main__":
    main()
