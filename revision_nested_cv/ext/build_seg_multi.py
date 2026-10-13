"""Polygon labels of the 9 LIACI classes for x6_multi_seg (CPU, run once locally, next to the outputs of build_data.py).

  labels_seg_multi/  class 0 = the corrosion polygons of labels_seg (one per evaluated box, unchanged),
                     classes 1-8 = the COCO polygons of the other classes, scaled to 640x640 like the images;
                     the same annotations as labels_multi (ship_hull dropped, degenerate boxes skipped)
  manifest_seg_multi.json  counts, the per-image check against labels_multi and sha256 of every file

Usage (from revision_nested_cv/):
  python ext/build_seg_multi.py --coco <coco-labels.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from build_data import CLASSES, DROPPED, SIDE

HERE = Path(__file__).resolve().parent


def polygon_line(points: np.ndarray, cls: int) -> str | None:
    """YOLO polygon line, or None when the polygon is degenerate (same rule as build_data.box_lines)."""
    p = np.clip(points, 0, SIDE)
    if len(p) < 3 or np.ptp(p[:, 0]) < 1 or np.ptp(p[:, 1]) < 1:
        return None
    return f"{cls} " + " ".join(f"{x / SIDE:.6f} {y / SIDE:.6f}" for x, y in p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco", required=True, type=Path)
    ap.add_argument("--data", default=HERE / "data", type=Path)
    args = ap.parse_args()

    out = args.data / "labels_seg_multi"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    coco = json.load(open(args.coco, encoding="utf-8"))
    names = {c["id"]: c["name"] for c in coco["categories"]}
    images = {im["file_name"]: im for im in coco["images"]}
    anns: dict[str, list[dict]] = {n: [] for n in images}
    by_id = {im["id"]: im["file_name"] for im in coco["images"]}
    for a in coco["annotations"]:
        anns[by_id[a["image_id"]]].append(a)

    targets = sorted(p.stem + ".jpg" for p in (args.data / "labels_multi").glob("*.txt"))
    per_class, mismatch = Counter(), []
    for name in targets:
        im = images[name]
        sx, sy = SIDE / float(im["width"]), SIDE / float(im["height"])
        seg = args.data / "labels_seg" / f"{Path(name).stem}.txt"
        lines = [l for l in seg.read_text().splitlines() if l.strip()] if seg.exists() else []
        for a in anns[name]:
            n = names[a["category_id"]]
            if n in DROPPED or n == "corrosion":
                continue
            s = a["segmentation"]
            s = json.loads(s) if isinstance(s, str) else s
            pts = np.array(s[0], dtype=float).reshape(-1, 2) * [sx, sy]
            line = polygon_line(pts, CLASSES.index(n))
            if line:
                lines.append(line)
                per_class[n] += 1
        (out / f"{Path(name).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), newline="\n")
        multi = [l.split()[0] for l in (args.data / "labels_multi" / f"{Path(name).stem}.txt").read_text().splitlines() if l.strip()]
        if Counter(multi) != Counter(l.split()[0] for l in lines):
            mismatch.append(name)

    files = sorted(out.glob("*.txt"))
    report = {"images": len(files), "polygons_per_class": dict(per_class.most_common()),
              "images_whose_class_counts_differ_from_labels_multi": mismatch,
              "sha256": {f"labels_seg_multi/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (args.data / "manifest_seg_multi.json").write_text(json.dumps(report, indent=1), newline="\n")
    print(json.dumps({k: v for k, v in report.items() if k != "sha256"}, indent=1))


if __name__ == "__main__":
    main()
