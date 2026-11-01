"""R2.5 -- why the converted dataset has 21 corrosion images in LIACI's official test split while the COCO file has 23.

Finding on this server (no raw LIACI needed): the 21 SAM 3 positives are exactly the 21 pool positives of the official
test split; image_0105 and image_0763 carry corrosion annotations in the LIACI COCO file (ext/data/manifest_ext.json,
'excluded_corrosion_outside_pool'), are in the official test split (r1/data/sam3_manifest.json) and were scored as
negatives. Both the 261-image pool and sam3_eval_dataset were built from masks/corrosion/<stem>.bmp, not from COCO.

This script needs the raw LIACI release (images, masks/corrosion, coco-labels.json, train_test_split.csv) and tells,
for every official-test image with a COCO corrosion annotation, which step loses it:
  mask file missing | mask empty | mask non-empty but empty after the 640x640 nearest resize | empty after the
  > 128 binarisation of 12.Data_Prep_SAM3_Eval (e.g. a coloured or low-grey mask) | kept

Usage:
  python results_rev2/r2_5_audit_missing_positives.py --liaci <LIACI root> [--coco <coco json>] [--split <csv>]
Writes results_rev2/r2_5/missing_positives_audit.csv and prints a summary.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent / "r2_5"
SIDE = 640


def find(root: Path, names: list[str]) -> Path | None:
    for n in names:
        hits = sorted(root.rglob(n))
        if hits:
            return hits[0]
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--liaci", required=True, type=Path)
    ap.add_argument("--coco", type=Path, default=None)
    ap.add_argument("--split", type=Path, default=None)
    ap.add_argument("--masks", type=Path, default=None, help="default: <liaci>/**/masks/corrosion")
    args = ap.parse_args()
    coco_path = args.coco or find(args.liaci, ["coco-labels.json", "*coco*.json"])
    split_path = args.split or find(args.liaci, ["train_test_split.csv"])
    masks = args.masks or next(p for p in args.liaci.rglob("corrosion") if p.is_dir() and p.parent.name == "masks")
    print(f"coco {coco_path}\nsplit {split_path}\nmasks {masks}")

    coco = json.load(open(coco_path, encoding="utf-8"))
    cid = next(c["id"] for c in coco["categories"] if c["name"] == "corrosion")
    by_id = {im["id"]: im for im in coco["images"]}
    corr = {}
    for a in coco["annotations"]:
        if a["category_id"] == cid:
            corr.setdefault(by_id[a["image_id"]]["file_name"], []).append(a)
    with open(split_path, encoding="utf-8") as f:
        test = {r["file_name"] for r in csv.DictReader(f) if r["split"].strip().lower() == "test"}

    rows = []
    for n in sorted(test & set(corr)):
        im = next(i for i in coco["images"] if i["file_name"] == n)
        mp = next((masks / f"{Path(n).stem}{e}" for e in (".bmp", ".png", ".jpg") if (masks / f"{Path(n).stem}{e}").exists()), None)
        row = {"image": n, "coco_corrosion_anns": len(corr[n]), "coco_area_px": round(sum(a.get("area", 0) for a in corr[n]), 1),
               "coco_area_frac": round(sum(a.get("area", 0) for a in corr[n]) / (im["width"] * im["height"]), 5),
               "mask_file": mp.name if mp else "", "mask_mode": "", "mask_values": "", "nonzero_orig": 0,
               "nonzero_640_nearest": 0, "nonzero_640_gt128_L": 0}
        if mp:
            with Image.open(mp) as m:
                row["mask_mode"] = m.mode
                a = np.array(m)
                row["mask_values"] = " ".join(str(v) for v in np.unique(a.reshape(-1, a.shape[-1]) if a.ndim == 3 else a, axis=0).tolist()[:6])
                g = a if a.ndim == 2 else a[..., 0]
                row["nonzero_orig"] = int((g > 0).sum()) if a.ndim == 2 else int((a > 0).any(-1).sum())
                small = np.array(m.convert("L").resize((SIDE, SIDE), Image.Resampling.NEAREST))
                row["nonzero_640_nearest"] = int((small > 0).sum())
                row["nonzero_640_gt128_L"] = int((small > 128).sum())
        row["lost_at"] = ("mask file missing" if not mp else "mask empty" if row["nonzero_orig"] == 0 else
                          "640 nearest resize" if row["nonzero_640_nearest"] == 0 else
                          "> 128 binarisation" if row["nonzero_640_gt128_L"] == 0 else "kept")
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "missing_positives_audit.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"official test images with COCO corrosion: {len(rows)}; kept: {sum(r['lost_at'] == 'kept' for r in rows)}")
    for r in rows:
        if r["lost_at"] != "kept":
            print(r)


if __name__ == "__main__":
    main()
