"""Prepare TrashCan 1.0 (material version) for the cross-domain experiment.

Source (protocol.yaml, section trashcan):
  coco  raw release: instances_train_trashcan.json / instances_val_trashcan.json + images
  yolo  an Ultralytics conversion with the 16 original classes (data.yaml names + labels/{train,val})
Both are mapped to two classes, Metal (trash_metal) and Bio (plant + animal_*); other boxes are dropped and
images that end up without boxes are kept as background images (they belong to the official splits).

Output: <output_root>/{images,labels}/{train,val}, list files train_fit.txt / train_sel.txt / val.txt,
data.yaml (train=train_fit, val=train_sel) and the manifest with counts and label checksums.
The official val split is only used for reporting; best.pt is selected on train_sel.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml

from r1.protocol import get_logger, load_protocol, resolve, sha256_json, write_json

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def class_lookup(class_map: dict[str, list[str]]) -> tuple[dict[str, int], dict[int, str]]:
    """original name -> new id, and new id -> new name (ids follow the order of class_map)."""
    to_id, names = {}, {}
    for new_id, (new_name, originals) in enumerate(class_map.items()):
        names[new_id] = new_name
        for o in originals:
            to_id[o] = new_id
    return to_id, names


def read_coco_split(json_path: Path, to_id: dict[str, int]) -> dict[str, list[tuple[int, float, float, float, float]]]:
    """file_name -> [(new_class, xc, yc, w, h) normalised]; images without kept boxes map to []."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    cat_to_new = {c["id"]: to_id[c["name"]] for c in data["categories"] if c["name"] in to_id}
    images = {im["id"]: im for im in data["images"]}
    labels: dict[str, list] = {im["file_name"]: [] for im in data["images"]}
    for ann in data["annotations"]:
        if ann["category_id"] not in cat_to_new:
            continue
        im = images[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            continue
        labels[im["file_name"]].append((cat_to_new[ann["category_id"]], (x + w / 2) / im["width"], (y + h / 2) / im["height"], w / im["width"], h / im["height"]))
    return labels


def read_yolo_split(root: Path, split: str, names: dict[int, str], to_id: dict[str, int], limit: int | None = None) -> dict[str, list[tuple[int, float, float, float, float]]]:
    labels = {}
    images = sorted(p for p in (root / "images" / split).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    for img in images[:limit] if limit else images:
        rows = []
        lbl = root / "labels" / split / f"{img.stem}.txt"
        if lbl.exists():
            for line in lbl.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                original = names[int(parts[0])]
                if original in to_id:
                    rows.append((to_id[original], *(float(v) for v in parts[1:5])))
        labels[img.name] = rows
    return labels


def find_image(name: str, roots: list[Path], index: dict[str, Path]) -> Path:
    if not index:
        for r in roots:
            for p in r.rglob("*"):
                if p.suffix.lower() in IMAGE_EXTS:
                    index.setdefault(p.name, p)
    if name not in index:
        raise FileNotFoundError(f"image {name} not found under {roots}")
    return index[name]


def split_lists(entries: list[dict], train_sel: list[str]) -> dict[str, list[str]]:
    sel_set = set(train_sel)
    return {
        "train_fit": sorted(e["image"] for e in entries if e["split"] == "train" and e["image"] not in sel_set),
        "train_sel": sorted(train_sel),
        "val": sorted(e["image"] for e in entries if e["split"] == "val"),
    }


def write_training_files(out: Path, lists: dict[str, list[str]], names: dict[int, str]) -> Path:
    """List files and data.yaml hold absolute paths, so they are rewritten on every machine before training."""
    for lname, names_ in lists.items():
        split_dir = "val" if lname == "val" else "train"
        (out / f"{lname}.txt").write_text("\n".join(str((out / "images" / split_dir / n).resolve()) for n in names_) + "\n", encoding="utf-8")
    data_yaml = out / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({"path": str(out.resolve()), "train": str((out / "train_fit.txt").resolve()), "val": str((out / "train_sel.txt").resolve()), "names": names}, sort_keys=False), encoding="utf-8")
    return data_yaml


def refresh_training_files(out: Path, manifest: dict) -> Path:
    names = {int(k): v for k, v in manifest["names"].items()}
    return write_training_files(out, split_lists(manifest["images"], manifest["train_sel_images"]), names)


def build(proto: dict, log, limit: int | None = None) -> dict:
    cfg = proto["trashcan"]
    src = Path(cfg["source_root"])
    out = resolve(cfg["output_root"])
    to_id, new_names = class_lookup(cfg["class_map"])
    splits: dict[str, dict[str, list]] = {}
    if cfg["source_format"] == "coco":
        jsons = {s: next(iter(src.rglob(f"instances_{s}_trashcan.json")), None) for s in ("train", "val")}
        if not all(jsons.values()):
            raise SystemExit(f"COCO annotation files not found under {src}: {jsons}")
        for s, j in jsons.items():
            splits[s] = read_coco_split(j, to_id)
        image_roots = [src]
    else:
        names_yaml = yaml.safe_load((src / "data.yaml").read_text(encoding="utf-8"))["names"]
        names = {int(k): v for k, v in (names_yaml.items() if isinstance(names_yaml, dict) else enumerate(names_yaml))}
        for s in ("train", "val"):
            splits[s] = read_yolo_split(src, s, names, to_id, limit)
        image_roots = [src / "images" / s for s in ("train", "val")]

    if limit:  # pipeline checks only: the first N images of each split, in name order
        splits = {s: dict(sorted(labels.items())[:limit]) for s, labels in splits.items()}
        log.warning(f"--limit {limit}: partial dataset for pipeline checks, not for reporting")
    index: dict[str, Path] = {}
    entries = []
    for split, labels in splits.items():
        img_dir, lbl_dir = out / "images" / split, out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for i, (name, rows) in enumerate(sorted(labels.items()), 1):
            dst = img_dir / name
            if not dst.exists():
                shutil.copy2(find_image(name, image_roots, index), dst)
            (lbl_dir / f"{Path(name).stem}.txt").write_text("\n".join(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for c, xc, yc, w, h in rows) + ("\n" if rows else ""), encoding="utf-8")
            per_class = Counter(new_names[c] for c, *_ in rows)
            entries.append({"image": name, "split": split, "n_boxes": len(rows), **{f"n_{n}": per_class.get(n, 0) for n in new_names.values()}})
            if i % 1000 == 0 or i == len(labels):
                log.info(f"  {split}: {i}/{len(labels)} images")

    # selection split inside the official train split, stratified by the presence of Metal boxes
    train_entries = [e for e in entries if e["split"] == "train"]
    rng = random.Random(cfg["selection_seed"])
    sel = []
    for has_metal in (True, False):
        group = [e["image"] for e in train_entries if (e["n_Metal"] > 0) == has_metal]
        rng.shuffle(group)
        sel += group[: round(cfg["selection_fraction"] * len(group))]
    lists = split_lists(entries, sorted(sel))
    write_training_files(out, lists, new_names)
    metal_val = [e["image"] for e in entries if e["split"] == "val" and e["n_Metal"] > 0]
    write_json(out / "val_metal.json", {"images": sorted(metal_val)})

    def summary(names_: list[str]) -> dict:
        sub = [e for e in entries if e["image"] in set(names_)]
        return {"n_images": len(sub), "n_boxes": sum(e["n_boxes"] for e in sub), **{f"images_with_{n}": sum(e[f"n_{n}"] > 0 for e in sub) for n in new_names.values()}, **{f"boxes_{n}": sum(e[f"n_{n}"] for e in sub) for n in new_names.values()}, "background_images": sum(e["n_boxes"] == 0 for e in sub)}

    manifest = {
        "source_root": str(src),
        "source_format": cfg["source_format"],
        "class_map": cfg["class_map"],
        "names": new_names,
        "selection_fraction": cfg["selection_fraction"],
        "selection_seed": cfg["selection_seed"],
        "limit": limit,
        "train_sel_images": lists["train_sel"],
        "lists": {k: summary(v) for k, v in lists.items()},
        "official_train": summary(lists["train_fit"] + lists["train_sel"]),
        "val_metal_images": len(metal_val),
        "labels_sha256": sha256_json(entries),
        "images": entries,
    }
    write_json(resolve(cfg["manifest_file"]), manifest)
    log.info(f"manifest written: {resolve(cfg['manifest_file'])} | {json.dumps({k: v for k, v in manifest.items() if k in ('lists', 'val_metal_images')})}")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=None, help="override trashcan.source_root")
    ap.add_argument("--format", choices=["coco", "yolo"], default=None, help="override trashcan.source_format")
    ap.add_argument("--output", default=None, help="override trashcan.output_root (with --manifest)")
    ap.add_argument("--manifest", default=None, help="override trashcan.manifest_file")
    ap.add_argument("--limit", type=int, default=None, help="first N images per split (pipeline checks only)")
    args = ap.parse_args()
    proto = load_protocol()
    for key, value in (("source_root", args.source), ("source_format", args.format), ("output_root", args.output), ("manifest_file", args.manifest)):
        if value:
            proto["trashcan"][key] = value
    log = get_logger("trashcan_prepare", resolve("r1/results/logs/trashcan_prepare.log"))
    build(proto, log, args.limit)


if __name__ == "__main__":
    main()
