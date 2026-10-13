"""Audit Ultralytics checkpoints without loading torch: architecture family, training date, data yaml,
and whether any frozen test image appears in the fold's training list.

Reads the pickled metadata strings inside the .pt zip, so it also works on machines without a GPU stack.
Exit code 1 when a checkpoint evaluated on the test set was trained with a test image or is not YOLO26.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

from revision.protocol import load_protocol, read_json, resolve

FAMILY_MARKERS = {"A2C2f": "YOLOv12", "C2PSA": "YOLO11/YOLO26", "C2f": "YOLOv8", "A2C2f\x00": "YOLOv12"}


def checkpoint_strings(pt: Path) -> set[str]:
    with zipfile.ZipFile(pt) as z:
        pkl = [n for n in z.namelist() if n.endswith("data.pkl")][0]
        raw = z.read(pkl)
    return {m.decode("utf-8", "ignore") for m in re.findall(rb"[\x20-\x7e]{4,200}", raw)}


def audit(pt: Path, test_names: set[str]) -> dict:
    strings = checkpoint_strings(pt)
    families = sorted({fam for marker, fam in FAMILY_MARKERS.items() if any(marker in s for s in strings)})
    model_names = sorted({m for s in strings for m in re.findall(r"yolo(?:v)?\d+[nsmlx]?\.(?:pt|yaml)", s)})  # .yaml: fine-tuned runs start from a path, not a hub name
    dates = sorted({m for s in strings for m in re.findall(r"20\d\d-\d\d-\d\dT\d\d:\d\d", s)})
    # pickled strings carry a length prefix, so the path is extracted from inside each printable run
    yaml_paths = sorted({m for s in strings for m in re.findall(r"(?:[A-Za-z]:)?[\\/][^\"'<>|\x00]*?\.ya?ml", s)})
    data_yaml = [p for p in yaml_paths if "data" in Path(p).name or "fold" in p]
    train_list_hits: list[str] = []
    train_source = None
    for s in yaml_paths:
        if Path(s).exists():
            import yaml

            cfg = yaml.safe_load(Path(s).read_text(encoding="utf-8"))
            train = cfg.get("train") if isinstance(cfg, dict) else None
            if isinstance(train, str) and train.endswith(".txt") and Path(train).exists():
                train_source = train
                listed = {Path(line.strip()).name for line in Path(train).read_text(encoding="utf-8").splitlines() if line.strip()}
                train_list_hits = sorted(listed & test_names)
            elif isinstance(train, str) and Path(train).is_dir():
                train_source = train
                train_list_hits = sorted({p.name for p in Path(train).iterdir()} & test_names)
    return {
        "checkpoint": str(pt),
        "size_mb": round(pt.stat().st_size / 1e6, 1),
        "families": families,
        "model_names": model_names,
        "dates": dates,
        "data_yaml": data_yaml,
        "train_source_checked": train_source,
        "test_images_in_train_list": train_list_hits,
        "is_yolo26": "YOLO26" in " ".join(families) and not any(f == "YOLOv12" for f in families) and any("yolo26" in m for m in model_names),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+", help=".pt files or directories (searched recursively for best.pt/*.pt)")
    ap.add_argument("--json", default=None, help="write the report here")
    args = ap.parse_args()
    proto = load_protocol()
    test_names = set(read_json(resolve(proto["dataset"]["test_images_file"]))["images"])
    pts: list[Path] = []
    for c in args.checkpoints:
        p = Path(c)
        pts += sorted(p.rglob("*.pt")) if p.is_dir() else [p]
    report = [audit(p, test_names) for p in pts]
    bad = [r for r in report if r["test_images_in_train_list"] or not r["is_yolo26"]]
    for r in report:
        flag = "OK " if r not in bad else "BAD"
        print(f"{flag} {r['checkpoint']} | {r['families']} {r['model_names']} | trained {r['dates'][:1]} | test-in-train={len(r['test_images_in_train_list'])}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
