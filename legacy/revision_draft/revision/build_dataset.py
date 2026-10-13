"""Build the LIACI corrosion detection dataset (v2) from the raw SINTEF LIACI release.

Rules (protocol.yaml, section dataset):
  * positive image  = corrosion mask with at least one non-zero pixel (checked at native resolution)
  * test set        = the frozen 33 images of data/test_images.json (identical to the submission)
  * pool positives  = every positive image not in the test set
  * pool negatives  = round(negatives_ratio * n_positives) - n_test_negatives images sampled with a fixed
                      seed from mask-free images that are neither in the test set nor in the official
                      LIACI 'test' split (so the SAM 3 evaluation set never overlaps the training pool)
  * images resized to 640x640 (LANCZOS); masks resized with nearest neighbour and converted to one box
    per external contour (see boxes.py). Negatives get an empty label file.

Outputs: <output_root>/{images,labels}/{pool,test}/ and data/manifest_v2.json.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from revision import boxes as bx
from revision.protocol import get_logger, load_protocol, read_json, resolve, sha256_file, sha256_json, write_json

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def read_liaci_split(csv_path: Path) -> dict[str, str]:
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["file_name"]: r["split"].strip().lower() for r in rows}


class RawFiles:
    """Locates raw images/masks across the source root and fallback copies, refusing unreadable files.

    A file is accepted only if PIL can fully decode it; otherwise the next copy is tried. Nothing is ever
    assumed (the original notebook treated an unreadable mask as a negative image).
    """

    def __init__(self, roots: list[Path], log):
        self.roots = roots
        self.log = log
        self.fallback_uses: dict[str, str] = {}

    @staticmethod
    def _decodable(path: Path) -> bool:
        try:
            with Image.open(path) as im:
                im.load()
            return True
        except Exception:
            return False

    def locate(self, kind: str, stem: str, exts: tuple[str, ...] = IMAGE_EXTS) -> Path | None:
        sub = ("images",) if kind == "image" else ("masks", "corrosion")
        for i, root in enumerate(self.roots):
            for ext in exts:
                cand = root.joinpath(*sub) / f"{stem}{ext}"
                if cand.exists():
                    if self._decodable(cand):
                        if i > 0:
                            self.fallback_uses[f"{kind}/{stem}"] = str(root)
                            self.log.warning(f"{kind} {stem}: unreadable in {self.roots[0]}, using copy from {root}")
                        return cand
                    break  # unreadable here, try the next root
        return None


def mask_is_positive(mask_path: Path) -> bool:
    with Image.open(mask_path) as m:
        return bool(np.asarray(m.convert("L")).sum() > 0)


def mask_boxes_resized(mask_path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(mask_path) as m:
        resized = m.convert("L").resize(size, Image.Resampling.NEAREST)
        return bx.mask_to_boxes(np.asarray(resized))


def resize_and_save(src: Path, dst: Path, size: tuple[int, int]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img.resize(size, Image.Resampling.LANCZOS).save(dst)


def index_masks(images: list[str], raw: RawFiles, cache: Path, log) -> dict[str, dict]:
    """stem -> {'image': name, 'mask': path|None, 'positive': bool}; cached because the raw masks are large."""
    index = read_json(cache) if cache.exists() else {}
    todo = [n for n in images if Path(n).stem not in index]
    log.info(f"mask index: {len(index)} cached, {len(todo)} to read")
    unreadable = []
    for i, name in enumerate(todo, 1):
        stem = Path(name).stem
        mask = raw.locate("mask", stem)
        if mask is None and any(root.joinpath("masks", "corrosion", f"{stem}{e}").exists() for root in raw.roots for e in IMAGE_EXTS):
            unreadable.append(stem)
            continue
        index[stem] = {"image": name, "mask": str(mask) if mask else None, "positive": bool(mask and mask_is_positive(mask))}
        if i % 100 == 0 or i == len(todo):
            log.info(f"  read {i}/{len(todo)} masks")
            write_json(cache, index)
    write_json(cache, index)
    if unreadable:
        raise SystemExit(f"{len(unreadable)} masks exist but are unreadable in every copy: {unreadable[:10]}")
    return index


def build(proto: dict, log) -> dict:
    ds = proto["dataset"]
    src = Path(ds["source_root"])
    out = resolve(ds["output_root"])
    size = tuple(ds["target_size"])
    raw = RawFiles([src] + [Path(p) for p in ds.get("fallback_roots", [])], log)
    liaci_split = read_liaci_split(src / "train_test_split.csv")
    test_names = read_json(resolve(ds["test_images_file"]))["images"]

    images = sorted(f.name for f in (src / "images").iterdir() if f.suffix.lower() in IMAGE_EXTS)
    index = index_masks(images, raw, resolve("data/liaci_mask_index.json"), log)
    positives = {v["image"] for v in index.values() if v["positive"]}
    negatives = {v["image"] for v in index.values() if not v["positive"]}
    log.info(f"LIACI: {len(images)} images, {len(positives)} positive, {len(negatives)} negative")

    missing = [n for n in test_names if Path(n).stem not in index or index[Path(n).stem]["image"] != n]
    if missing:
        raise SystemExit(f"test images not found in LIACI: {missing}")
    test_pos = [n for n in test_names if n in positives]
    test_neg = [n for n in test_names if n in negatives]
    log.info(f"test: {len(test_pos)} positive + {len(test_neg)} negative (frozen)")

    pool_pos = sorted(positives - set(test_names))
    n_total_neg = round(ds["negatives_ratio"] * len(positives))
    n_pool_neg = n_total_neg - len(test_neg)
    candidates = sorted(n for n in negatives if n not in test_names and liaci_split.get(n) in ("train", "not_used"))
    pool_neg = sorted(random.Random(ds["negatives_seed"]).sample(candidates, n_pool_neg))
    log.info(f"pool: {len(pool_pos)} positive + {len(pool_neg)} negative (sampled {n_pool_neg} of {len(candidates)} candidates, seed {ds['negatives_seed']})")

    entries = []
    skipped = []
    for split, names in (("pool", pool_pos + pool_neg), ("test", test_names)):
        for name in sorted(names):
            stem = Path(name).stem
            positive = name in positives
            boxes = mask_boxes_resized(Path(index[stem]["mask"]), size) if positive else np.zeros((0, 4))
            if positive and len(boxes) == 0:
                skipped.append(name)
                continue
            img_src = raw.locate("image", stem, (Path(name).suffix,))
            if img_src is None:
                raise SystemExit(f"image {name} is unreadable in every copy")
            img_out = out / "images" / split / name
            lbl_out = out / "labels" / split / f"{stem}.txt"
            resize_and_save(img_src, img_out, size)
            lbl_out.parent.mkdir(parents=True, exist_ok=True)
            lbl_out.write_text("\n".join(bx.boxes_to_yolo_lines(boxes, *size)), encoding="utf-8")
            areas = bx.box_area(boxes)
            entries.append(
                {
                    "image": name,
                    "split": split,
                    "liaci_split": liaci_split.get(name, "unknown"),
                    "positive": positive,
                    "n_boxes": int(len(boxes)),
                    "box_sizes": dict(Counter(bx.size_bucket(a) for a in areas)),
                    "image_sha256": sha256_file(img_out),
                }
            )
    if skipped:
        log.warning(f"{len(skipped)} positive images produced no box after resizing and were excluded: {skipped}")

    def counts(split: str) -> dict[str, int]:
        return dict(Counter("positive" if e["positive"] else "negative" for e in entries if e["split"] == split))

    summary = {
        "n_images": len(entries),
        "pool": counts("pool"),
        "test": counts("test"),
        "n_boxes_pool": sum(e["n_boxes"] for e in entries if e["split"] == "pool"),
        "n_boxes_test": sum(e["n_boxes"] for e in entries if e["split"] == "test"),
        "excluded_no_box": skipped,
        "fallback_copies_used": raw.fallback_uses,
    }
    manifest = {
        "dataset": ds["name"],
        "source_root": str(src),
        "liaci_counts": {"n_images": len(images), "n_positive": len(positives), "n_negative": len(negatives)},
        "target_size": list(size),
        "negatives_ratio": ds["negatives_ratio"],
        "negatives_seed": ds["negatives_seed"],
        "conversion": "mask resized 640x640 nearest, binarised >0, cv2.RETR_EXTERNAL contours (>=3 vertices), one minimal enclosing box per contour",
        "summary": summary,
        "images": entries,
    }
    manifest["sha256"] = sha256_json(manifest["images"])
    write_json(resolve(ds["manifest_file"]), manifest)
    log.info(f"manifest written: {resolve(ds['manifest_file'])} | summary: {summary}")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    proto = load_protocol()
    log = get_logger("build_dataset", resolve("results/logs/build_dataset.log"))
    build(proto, log)


if __name__ == "__main__":
    main()
