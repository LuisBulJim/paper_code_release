"""Local check that dataset v2 reproduces the submitted dataset where it should, and differs only where intended.

Expected outcome:
  * the 33 test images and their 52 boxes are identical to the submitted dataset_yolo (same conversion)
  * the set of positive images (pool + test) equals the submitted train + test positives (209)
  * pool negatives are a fresh seeded sample that never comes from the official LIACI test split
Usage: python -m revision.verify_submission_consistency [--submitted <path to submitted dataset_yolo>]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from revision import boxes as bx
from revision.protocol import load_protocol, read_json, resolve

DEFAULT_SUBMITTED = Path("dataset_yolo")   # the submitted dataset_yolo; override with --submitted


def label_boxes(path: Path) -> np.ndarray:
    return bx.read_yolo_labels(path, 640, 640)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submitted", default=str(DEFAULT_SUBMITTED))
    args = ap.parse_args()
    proto = load_protocol()
    v2 = resolve(proto["dataset"]["output_root"])
    sub = Path(args.submitted)
    manifest = read_json(resolve(proto["dataset"]["manifest_file"]))
    problems = []

    # 1. test labels box-by-box
    test_entries = [e for e in manifest["images"] if e["split"] == "test"]
    n_boxes = 0
    max_dev = 0.0
    for e in test_entries:
        stem = Path(e["image"]).stem
        a = label_boxes(v2 / "labels" / "test" / f"{stem}.txt")
        b = label_boxes(sub / "labels" / "test" / f"{stem}.txt")
        if a.shape != b.shape:
            problems.append(f"test {stem}: {len(a)} boxes in v2 vs {len(b)} submitted")
            continue
        if len(a):
            a_sorted, b_sorted = a[np.lexsort(a.T[::-1])], b[np.lexsort(b.T[::-1])]
            dev = float(np.abs(a_sorted - b_sorted).max())
            max_dev = max(max_dev, dev)
            if dev > 0.05:
                problems.append(f"test {stem}: max box deviation {dev:.3f} px")
        n_boxes += len(a)
    print(f"test labels: {len(test_entries)} images, {n_boxes} boxes, max coordinate deviation {max_dev:.4f} px")

    # 2. test images pixel comparison
    worst = 0.0
    for e in test_entries:
        with Image.open(v2 / "images" / "test" / e["image"]) as im_a, Image.open(sub / "images" / "test" / e["image"]) as im_b:
            diff = np.abs(np.asarray(im_a.convert("RGB"), dtype=float) - np.asarray(im_b.convert("RGB"), dtype=float)).mean()
        worst = max(worst, diff)
    print(f"test images: mean absolute pixel difference, worst image = {worst:.3f} (0 = identical JPEG encoding)")
    if worst > 2.0:
        problems.append(f"test images differ from the submitted ones (worst mean abs diff {worst:.2f})")

    # 3. positives set
    v2_pos = {Path(e["image"]).stem for e in manifest["images"] if e["positive"]}
    sub_pos = {p.stem for s in ("train", "test") for p in (sub / "labels" / s).glob("*.txt") if p.read_text().strip()}
    print(f"positives: v2 {len(v2_pos)} vs submitted {len(sub_pos)} | only v2 {sorted(v2_pos - sub_pos)[:5]} | only submitted {sorted(sub_pos - v2_pos)[:5]}")
    if v2_pos != sub_pos:
        problems.append("positive image sets differ")

    # 4. negatives provenance
    pool_neg = [e for e in manifest["images"] if e["split"] == "pool" and not e["positive"]]
    from_liaci_test = [e["image"] for e in pool_neg if e["liaci_split"] == "test"]
    sub_train_neg = {p.stem for p in (sub / "images" / "train").glob("*") if not (sub / "labels" / "train" / f"{p.stem}.txt").exists() or not (sub / "labels" / "train" / f"{p.stem}.txt").read_text().strip()}
    kept = len({Path(e["image"]).stem for e in pool_neg} & sub_train_neg)
    print(f"pool negatives: {len(pool_neg)} sampled, {kept} coincide with the submitted train negatives, {len(from_liaci_test)} from LIACI test split (must be 0)")
    if from_liaci_test:
        problems.append(f"pool negatives from LIACI test split: {from_liaci_test}")

    print("fallback copies used:", len(manifest["summary"].get("fallback_copies_used", {})))
    if problems:
        print("PROBLEMS:\n  " + "\n  ".join(problems))
        raise SystemExit(1)
    print("OK: dataset v2 is consistent with the submission on the test set and differs only in the sampled negatives.")


if __name__ == "__main__":
    main()
