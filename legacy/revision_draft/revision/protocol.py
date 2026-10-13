"""Protocol loading, paths, seeding and small shared helpers for the R1 revision scripts."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def load_protocol() -> dict[str, Any]:
    with open(CONFIG_DIR / "protocol.yaml", encoding="utf-8") as f:
        proto = yaml.safe_load(f)
    if os.environ.get("LIACI_ROOT"):
        proto["dataset"]["source_root"] = os.environ["LIACI_ROOT"]
    if os.environ.get("DATASET_ROOT"):
        proto["dataset"]["output_root"] = os.environ["DATASET_ROOT"]
    if os.environ.get("TRASHCAN_ROOT"):
        proto["trashcan"]["output_root"] = os.environ["TRASHCAN_ROOT"]
    if os.environ.get("TRASHCAN_MANIFEST"):
        proto["trashcan"]["manifest_file"] = os.environ["TRASHCAN_MANIFEST"]
    return proto


def load_hparams() -> dict[str, Any]:
    with open(CONFIG_DIR / "hparams.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
        f.write("\n")


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def environment_report() -> dict[str, Any]:
    report: dict[str, Any] = {"python": sys.version.split()[0], "platform": sys.platform}
    try:
        import torch

        report["torch"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        report["torch"] = None
    try:
        import ultralytics

        report["ultralytics"] = ultralytics.__version__
    except ImportError:
        report["ultralytics"] = None
    return report


def check_environment(proto: dict[str, Any], strict: bool = False) -> dict[str, Any]:
    """Compare installed versions with protocol.yaml. Mismatch is fatal only when strict."""
    report = environment_report()
    expected = proto["environment"]
    problems = []
    if report.get("ultralytics") != expected["ultralytics"]:
        problems.append(f"ultralytics {report.get('ultralytics')} != {expected['ultralytics']}")
    if report.get("torch") and not str(report["torch"]).startswith(expected["torch"]):
        problems.append(f"torch {report.get('torch')} != {expected['torch']}*")
    report["problems"] = problems
    if problems and strict:
        raise RuntimeError("Environment does not match protocol.yaml: " + "; ".join(problems))
    return report


def check_data(proto: dict[str, Any]) -> list[str]:
    """Problems with the shipped LIACI v2 dataset: missing or modified images, folds from another manifest."""
    ds = proto["dataset"]
    problems = []
    manifest_path, folds_path = resolve(ds["manifest_file"]), resolve(proto["splits"]["folds_file"])
    for p in (manifest_path, folds_path, resolve(ds["test_images_file"])):
        if not p.exists():
            problems.append(f"missing {p}")
    if problems:
        return problems
    manifest = read_json(manifest_path)
    root = resolve(ds["output_root"])
    for e in manifest["images"]:
        img = root / "images" / e["split"] / e["image"]
        if not img.exists():
            problems.append(f"missing image {img}")
        elif sha256_file(img) != e["image_sha256"]:
            problems.append(f"modified image {img}")
        if not (root / "labels" / e["split"] / f"{Path(e['image']).stem}.txt").exists():
            problems.append(f"missing label for {e['image']}")
    if read_json(folds_path)["manifest_sha256"] != manifest["sha256"]:
        problems.append("folds file was generated from another manifest")
    return problems


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Print the protocol and check the environment and the shipped dataset.")
    ap.add_argument("--check", action="store_true", help="fail if versions differ from protocol.yaml or the dataset is incomplete")
    args = ap.parse_args()
    proto = load_protocol()
    print(json.dumps({"protocol": proto, "hparams": load_hparams()}, indent=2))
    print(json.dumps(check_environment(proto, strict=args.check), indent=2))
    data_problems = check_data(proto)
    print(f"dataset check: {'OK' if not data_problems else data_problems[:10]}")
    if data_problems and args.check:
        raise SystemExit(f"{len(data_problems)} dataset problems")
