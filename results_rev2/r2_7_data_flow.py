"""R2.7 -- data flow from the 1,893 LIACI frames to the 261-image pool and the 1,591 extra images (Table S3).

Sources on this server: ext/data/manifest_ext.json (built from the full LIACI release by ext/build_data.py),
r1/data/sam3_manifest.json (official test split, 191 images), dataset_yolo/ (the pool), and the pool-building
notebook 0.Tratamiento_dataset_acuatico.ipynb, whose rule is:
  positives  every LIACI image whose masks/corrosion/<stem> file has any non-zero pixel (all of them, no sampling)
  negatives  int(0.25 * n_positives) images drawn by an unseeded random.shuffle from the images without such a mask
With the official split CSV (--split), the pool is also broken down into official train / test / outside the split.

Usage: python results_rev2/r2_7_data_flow.py [--split <LIACI train_test_split.csv>]
Writes results_rev2/r2_7/data_flow.csv and data_flow.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results_rev2" / "r2_7"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", type=Path, default=None)
    args = ap.parse_args()
    ext = json.load(open(REPO / "revision_nested_cv/ext/data/manifest_ext.json"))
    sam3 = json.load(open(REPO / "revision_nested_cv/r1/data/sam3_manifest.json"))
    pool = {}
    for sp in ("train", "test"):
        for f in os.listdir(REPO / "dataset_yolo/images" / sp):
            lab = REPO / "dataset_yolo/labels" / sp / f"{Path(f).stem}.txt"
            pool[f] = lab.exists() and lab.stat().st_size > 0
    official_test = {e["image"] for e in sam3["images"]}
    n_pos = sum(pool.values())
    flow = {
        "liaci_frames": ext["liaci_images"],
        "pool": len(pool), "pool_positive": n_pos, "pool_negative": len(pool) - n_pos,
        "negatives_rule_int_0.25_x_positives": int(0.25 * n_pos),
        "pool_in_official_test": len(set(pool) & official_test),
        "official_test": len(official_test),
        "official_test_positive_in_converted": sum(e["positive"] for e in sam3["images"]),
        "extra": ext["extra"],
        "not_used_missing_in_local_copy": len(ext["extra_unreadable_skipped"]),
        "not_used_coco_corrosion_without_corrosion_mask": len(ext["excluded_corrosion_outside_pool"]),
        "coco_corrosion_without_mask_in_official_test": sorted(set(ext["excluded_corrosion_outside_pool"]) & official_test),
        "missing_in_local_copy": ext["extra_unreadable_skipped"],
        "coco_corrosion_without_mask": ext["excluded_corrosion_outside_pool"],
    }
    flow["check_sum"] = flow["pool"] + flow["extra"] + flow["not_used_missing_in_local_copy"] + flow["not_used_coco_corrosion_without_corrosion_mask"]
    assert flow["check_sum"] == flow["liaci_frames"], flow["check_sum"]
    assert flow["pool_negative"] == flow["negatives_rule_int_0.25_x_positives"]
    if args.split:
        with open(args.split, encoding="utf-8") as f:
            split = {r["file_name"]: r["split"].strip().lower() for r in csv.DictReader(f)}
        for s in ("train", "test"):
            flow[f"liaci_official_{s}"] = sum(v == s for v in split.values())
            flow[f"pool_in_official_{s}"] = sum(split.get(n) == s for n in pool)
        flow["liaci_outside_split"] = flow["liaci_frames"] - flow["liaci_official_train"] - flow["liaci_official_test"]
        flow["pool_outside_split"] = sum(split.get(n) not in ("train", "test") for n in pool)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data_flow.json").write_text(json.dumps(flow, indent=1), encoding="utf-8")
    with open(OUT / "data_flow.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "value"])
        for k, v in flow.items():
            if not isinstance(v, list):
                w.writerow([k, v])
    print(json.dumps({k: v for k, v in flow.items() if not isinstance(v, list) or len(v) < 8}, indent=1))


if __name__ == "__main__":
    main()
