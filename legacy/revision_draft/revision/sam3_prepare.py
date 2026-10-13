"""Build the SAM 3 evaluation set from the raw LIACI release.

Set = official LIACI 'test' split (train_test_split.csv, 191 images) minus any image present in the
training pool of dataset v2 (0 with dataset v2; excluded names are listed in the manifest if any). Images
are resized to the detector input size (640x640, LANCZOS) and the corrosion masks with nearest neighbour,
binarised (>0 -> 255). Output: <output_root>/{images,masks} and data/sam3_manifest.json with, per image,
positive flag, mask area fraction, number of connected components and whether it also belongs to the frozen
33-image detection test set.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from revision import boxes as bx
from revision.build_dataset import IMAGE_EXTS, RawFiles, read_liaci_split
from revision.protocol import get_logger, load_protocol, read_json, resolve, sha256_file, write_json


def build(proto: dict, log) -> dict:
    ds, cfg = proto["dataset"], proto["sam3"]
    src = Path(ds["source_root"])
    out = resolve(cfg["output_root"])
    size = tuple(ds["target_size"])
    raw = RawFiles([src] + [Path(p) for p in ds.get("fallback_roots", [])], log)
    liaci_split = read_liaci_split(src / "train_test_split.csv")
    manifest_v2 = read_json(resolve(ds["manifest_file"]))
    pool = {e["image"] for e in manifest_v2["images"] if e["split"] == "pool"}
    frozen_test = set(read_json(resolve(ds["test_images_file"]))["images"])
    index = read_json(resolve("data/liaci_mask_index.json"))

    test_split = sorted(n for n, s in liaci_split.items() if s == "test")
    excluded = sorted(n for n in test_split if n in pool)
    names = [n for n in test_split if n not in pool]
    log.info(f"LIACI test split: {len(test_split)} images | in training pool (excluded): {len(excluded)} | evaluation set: {len(names)}")

    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    entries = []
    for name in names:
        stem = Path(name).stem
        img_src = raw.locate("image", stem, (Path(name).suffix,))
        if img_src is None:
            raise SystemExit(f"image {name} unreadable in every copy")
        with Image.open(img_src) as im:
            im.convert("RGB").resize(size, Image.Resampling.LANCZOS).save(out / "images" / f"{stem}.jpg", quality=95)
        mask_path = index[stem]["mask"] if stem in index else None
        if mask_path and index[stem]["positive"]:
            with Image.open(mask_path) as m:
                arr = (np.asarray(m.convert("L").resize(size, Image.Resampling.NEAREST)) > 0)
        else:
            arr = np.zeros(size[::-1], dtype=bool)
        Image.fromarray((arr * 255).astype(np.uint8)).save(out / "masks" / f"{stem}.png")
        boxes = bx.mask_to_boxes(arr.astype(np.uint8)) if arr.any() else np.zeros((0, 4))
        entries.append({"image": f"{stem}.jpg", "source_image": name, "positive": bool(arr.any()), "mask_area_fraction": float(arr.mean()), "n_components": int(len(boxes)), "gt_boxes_xyxy": np.round(boxes, 1).tolist(), "in_frozen_detection_test": name in frozen_test, "image_sha256": sha256_file(out / "images" / f"{stem}.jpg")})
    n_pos = sum(e["positive"] for e in entries)
    manifest = {
        "source_root": str(src),
        "definition": "official LIACI test split minus images of the dataset v2 training pool",
        "target_size": list(size),
        "n_images": len(entries),
        "n_positive": n_pos,
        "n_negative": len(entries) - n_pos,
        "n_in_frozen_detection_test": sum(e["in_frozen_detection_test"] for e in entries),
        "excluded_in_training_pool": excluded,
        "fallback_copies_used": raw.fallback_uses,
        "images": entries,
    }
    write_json(resolve(cfg["manifest_file"]), manifest)
    log.info(f"manifest written: {resolve(cfg['manifest_file'])} | {n_pos} positive, {len(entries) - n_pos} negative, {manifest['n_in_frozen_detection_test']} also in the frozen detection test")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    proto = load_protocol()
    log = get_logger("sam3_prepare", resolve("results/logs/sam3_prepare.log"))
    build(proto, log)


if __name__ == "__main__":
    main()
