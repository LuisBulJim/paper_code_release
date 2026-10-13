"""What do the false positives land on? Uses LIACI's own masks for the other nine classes.

Also audits the 261-image pool against the dataset's COCO annotations and counts what the
rest of LIACI (1,632 images) could add.
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

sys.path.insert(0, str(Path(__file__).parent))
from diagnostics import COCO, REPEATS, ROOT, iou_matrix, load_repeat  # noqa: E402

SPLIT = COCO.parent / "train_test_split.csv"
SIDE = 320
COVER = 0.2


def class_masks(d: dict) -> tuple[dict[str, dict[int, np.ndarray]], dict[int, str]]:
    names = {c["id"]: c["name"] for c in d["categories"]}
    by_id = {im["id"]: im for im in d["images"]}
    segs: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for a in d["annotations"]:
        segs[by_id[a["image_id"]]["file_name"]][a["category_id"]].append(a["segmentation"])
    size = {im["file_name"]: (im["height"], im["width"]) for im in d["images"]}

    def build(name: str) -> dict[int, np.ndarray]:
        import cv2

        h, w = size[name]
        out = {}
        for cid, items in segs[name].items():
            m = np.zeros((h, w), dtype=np.uint8)
            for seg in items:
                m |= mask_utils.decode(mask_utils.merge(mask_utils.frPyObjects(seg, h, w)))
            out[cid] = cv2.resize(m, (SIDE, SIDE), interpolation=cv2.INTER_NEAREST) > 0
        return out

    return build, names, segs


def main() -> None:
    d = json.load(open(COCO, encoding="utf-8"))
    build, names, segs = class_masks(d)
    hull = next(k for k, v in names.items() if v == "ship_hull")
    corr = next(k for k, v in names.items() if v == "corrosion")

    manifest = json.load(open(ROOT / "r1" / "data" / "nested_manifest.json"))
    pool = {im["image"]: im for im in manifest["images"]}
    coco_pos = {n for n, by in segs.items() if corr in by}
    pool_pos = {n for n, im in pool.items() if im["positive"]}
    pool_neg = set(pool) - pool_pos
    print("pool", len(pool), "pos", len(pool_pos), "neg", len(pool_neg))
    print("COCO corrosion images", len(coco_pos), "| in pool as positive", len(coco_pos & pool_pos), "| in pool as background", len(coco_pos & pool_neg), "| not in pool", len(coco_pos - set(pool)))
    print("pool positives without COCO corrosion", len(pool_pos - coco_pos))
    n_inst = sum(len(segs[n][corr]) for n in pool_pos & coco_pos)
    print("annotator instances in pool positives", n_inst, "| connected-component boxes", sum(im["n_boxes"] for im in pool.values()))

    split = {r["file_name"]: r["split"] for r in csv.DictReader(open(SPLIT, encoding="utf-8"))}
    print("official split of pool images", Counter(split[n] for n in pool))
    rest = [im["file_name"] for im in d["images"] if im["file_name"] not in pool]
    print("images outside the pool", len(rest), "| official split", Counter(split[n] for n in rest))
    cooc = Counter()
    for n in pool_pos:
        for cid in segs[n]:
            cooc[names[cid]] += 1
    print("classes present in the pool positives", dict(cooc.most_common()))
    outside = Counter()
    for n in rest:
        for cid in segs[n]:
            outside[names[cid]] += 1
    print("classes present outside the pool", dict(outside.most_common()))

    land = Counter()
    total = 0
    area_share = Counter()
    n_img = 0
    for r in range(REPEATS):
        rec = load_repeat("direct", "m", r)
        for name, (boxes, sc, gt) in rec.items():
            masks = build(name)
            n_img += 1
            for cid, m in masks.items():
                if cid not in (hull, corr):
                    area_share[names[cid]] += float(m.mean())
            keep = sc >= 0.25
            b = boxes[keep]
            if not len(b):
                continue
            best = iou_matrix(b, gt).max(axis=1) if len(gt) else np.zeros(len(b))
            for box in b[best < 0.1]:
                total += 1
                x1, y1, x2, y2 = (np.clip(box, 0, 1) * SIDE).round().astype(int)
                if x2 <= x1 or y2 <= y1:
                    land["degenerate"] += 1
                    continue
                cover = {names[cid]: float(m[y1:y2, x1:x2].mean()) for cid, m in masks.items() if cid not in (hull, corr)}
                top = max(cover.items(), key=lambda kv: kv[1]) if cover else ("none", 0.0)
                on_corr = float(masks[corr][y1:y2, x1:x2].mean()) if corr in masks else 0.0
                if on_corr >= COVER:
                    land["corrosion_mask (unmatched region)"] += 1
                elif top[1] >= COVER:
                    land[top[0]] += 1
                elif hull in masks and masks[hull][y1:y2, x1:x2].mean() >= 0.5:
                    land["plain hull"] += 1
                else:
                    land["water / other"] += 1
    print("background false positives (conf>=0.25, IoU<0.1), 3 repeats:", total)
    for k, v in land.most_common():
        print(f"  {k:36s} {v:4d}  {100 * v / max(total, 1):5.1f}%")
    print("mean image-area share of each class in the evaluated images (reference):")
    for k, v in area_share.most_common():
        print(f"  {k:20s} {100 * v / n_img:5.2f}%")


if __name__ == "__main__":
    main()
