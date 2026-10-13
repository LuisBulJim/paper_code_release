import json
import logging
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from r1 import trashcan_prepare as tp

CLASS_MAP = {"Metal": ["trash_metal"], "Bio": ["plant", "animal_fish"]}
ORIGINAL = ["rov", "plant", "animal_fish", "trash_metal", "trash_plastic"]


def _image(path: Path, size=(64, 48)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((size[1], size[0], 3), dtype=np.uint8)).save(path)


def _yolo_source(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(yaml.safe_dump({"names": {i: n for i, n in enumerate(ORIGINAL)}}))
    labels = {
        "train": {"t1.jpg": ["3 0.5 0.5 0.2 0.2", "1 0.3 0.3 0.1 0.1"], "t2.jpg": ["4 0.5 0.5 0.2 0.2"], "t3.jpg": ["0 0.5 0.5 0.9 0.9", "2 0.2 0.2 0.1 0.1"]},
        "val": {"v1.jpg": ["3 0.5 0.5 0.2 0.2"], "v2.jpg": ["1 0.5 0.5 0.2 0.2"]},
    }
    for split, files in labels.items():
        for name, lines in files.items():
            _image(root / "images" / split / name)
            (root / "labels" / split / f"{Path(name).stem}.txt").parent.mkdir(parents=True, exist_ok=True)
            (root / "labels" / split / f"{Path(name).stem}.txt").write_text("\n".join(lines) + "\n")
    return root


def _coco_source(root: Path) -> Path:
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(ORIGINAL)]
    for split, files in (("train", ["t1.jpg", "t2.jpg"]), ("val", ["v1.jpg"])):
        images = [{"id": i + 1, "file_name": f, "width": 64, "height": 48} for i, f in enumerate(files)]
        for f in files:
            _image(root / split / f)
        anns = [{"id": 1, "image_id": 1, "category_id": 4, "bbox": [16, 12, 32, 24]}, {"id": 2, "image_id": 1, "category_id": 5, "bbox": [0, 0, 10, 10]}]
        if split == "train":
            anns.append({"id": 3, "image_id": 2, "category_id": 2, "bbox": [0, 0, 16, 16]})
        (root / f"instances_{split}_trashcan.json").write_text(json.dumps({"images": images, "annotations": anns, "categories": cats}))
    return root


def _proto(tmp_path: Path, src: Path, fmt: str) -> dict:
    return {"trashcan": {"source_root": str(src), "source_format": fmt, "output_root": str(tmp_path / "out"), "manifest_file": str(tmp_path / "manifest.json"), "class_map": CLASS_MAP, "selection_fraction": 0.5, "selection_seed": 0}}


def test_yolo_source_is_remapped_and_split(tmp_path):
    src = _yolo_source(tmp_path / "src")
    man = tp.build(_proto(tmp_path, src, "yolo"), logging.getLogger("t"))
    out = tmp_path / "out"
    assert (out / "labels" / "train" / "t1.txt").read_text().splitlines() == ["0 0.500000 0.500000 0.200000 0.200000", "1 0.300000 0.300000 0.100000 0.100000"]
    assert (out / "labels" / "train" / "t2.txt").read_text() == ""            # plastic dropped -> background image
    assert (out / "labels" / "train" / "t3.txt").read_text().splitlines() == ["1 0.200000 0.200000 0.100000 0.100000"]  # rov dropped
    assert man["official_train"] == {"n_images": 3, "n_boxes": 3, "images_with_Metal": 1, "images_with_Bio": 2, "boxes_Metal": 1, "boxes_Bio": 2, "background_images": 1}
    fit = (out / "train_fit.txt").read_text().split()
    sel = (out / "train_sel.txt").read_text().split()
    assert len(fit) + len(sel) == 3 and not set(fit) & set(sel)
    assert json.loads((out / "val_metal.json").read_text())["images"] == ["v1.jpg"]
    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["names"] == {0: "Metal", 1: "Bio"} and data["val"].endswith("train_sel.txt")
    assert all((out / "images" / "val" / n).exists() for n in ("v1.jpg", "v2.jpg"))


def test_coco_source_gives_normalised_boxes(tmp_path):
    src = _coco_source(tmp_path / "src")
    man = tp.build(_proto(tmp_path, src, "coco"), logging.getLogger("t"))
    lines = (tmp_path / "out" / "labels" / "train" / "t1.txt").read_text().splitlines()
    assert lines == ["0 0.500000 0.500000 0.500000 0.500000"]           # plastic box dropped
    assert (tmp_path / "out" / "labels" / "train" / "t2.txt").read_text().splitlines() == ["1 0.125000 0.166667 0.250000 0.333333"]
    assert man["lists"]["val"]["boxes_Metal"] == 1 and man["val_metal_images"] == 1


def test_training_files_are_rebuilt_after_moving_the_dataset(tmp_path):
    import shutil

    src = _yolo_source(tmp_path / "src")
    man = tp.build(_proto(tmp_path, src, "yolo"), logging.getLogger("t"))
    moved = tmp_path / "server" / "trashcan_v2"
    shutil.copytree(tmp_path / "out", moved)
    data_yaml = tp.refresh_training_files(moved, man)
    data = yaml.safe_load(data_yaml.read_text())
    assert data["path"] == str(moved.resolve())
    listed = (moved / "train_fit.txt").read_text().split() + (moved / "train_sel.txt").read_text().split()
    assert listed and all(p.startswith(str(moved.resolve())) and Path(p).exists() for p in listed)
    assert sorted(Path(p).name for p in (moved / "train_sel.txt").read_text().split()) == man["train_sel_images"]
