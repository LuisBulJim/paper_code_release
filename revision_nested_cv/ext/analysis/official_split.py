"""How close is LIACI's official test split to its official training split, frame-wise?
For every official test image: is the image with filename index +-1 (or within k indices) in the official training split?
Uses the LIACI COCO annotations (image file names carry the global frame index) and the official split lists.
"""
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from confusion import SPLIT  # noqa: E402
from diagnostics import COCO  # noqa: E402

import csv  # noqa: E402
import json  # noqa: E402


def main() -> None:
    d = json.load(open(COCO, encoding="utf-8"))
    names = [im["file_name"] for im in d["images"]]
    split = {r["file_name"]: r["split"] for r in csv.DictReader(open(SPLIT, encoding="utf-8"))}
    idx = {n: int(re.search(r"(\d+)", Path(n).stem).group(1)) for n in names}
    by_split = Counter(split.values())
    print("official split sizes", dict(by_split))
    train_idx = {idx[n] for n in names if split[n] == "train"}
    test = [n for n in names if split[n] == "test"]
    corr = next(c["id"] for c in d["categories"] if c["name"] == "corrosion")
    with_corr = set()
    img_by_id = {im["id"]: im["file_name"] for im in d["images"]}
    for a in d["annotations"]:
        if a["category_id"] == corr:
            with_corr.add(img_by_id[a["image_id"]])
    for k in (1, 2, 5, 15):
        near = [n for n in test if any((idx[n] + s) in train_idx for s in range(-k, k + 1) if s)]
        near_pos = [n for n in near if n in with_corr]
        print(f"test images with a training frame within +-{k:2d} indices: {len(near)}/{len(test)} = {100 * len(near) / len(test):.1f}% "
              f"| among test images with corrosion: {len(near_pos)}/{len([n for n in test if n in with_corr])}")


if __name__ == "__main__":
    main()
