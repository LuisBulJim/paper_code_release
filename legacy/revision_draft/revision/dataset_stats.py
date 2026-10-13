"""Dataset statistics for the manuscript (section 4.2) and for the box-size analysis requested by the reviewers.

Writes results/dataset_stats.json (counts per partition, boxes per image, COCO size buckets, aspect ratios)
and figures/dataset_boxes.png (box area and aspect-ratio histograms, pool vs test).
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from revision import boxes as bx
from revision.protocol import get_logger, load_protocol, read_json, resolve, write_json


def collect_boxes(dataset_root: Path, manifest: dict) -> dict[str, np.ndarray]:
    out: dict[str, list] = {"pool": [], "test": []}
    for e in manifest["images"]:
        lbl = dataset_root / "labels" / e["split"] / f"{Path(e['image']).stem}.txt"
        w, h = manifest["target_size"]
        for box in bx.read_yolo_labels(lbl, w, h):
            out[e["split"]].append(box)
    return {k: np.asarray(v, dtype=float).reshape(-1, 4) for k, v in out.items()}


def describe(boxes: np.ndarray, entries: list[dict]) -> dict:
    areas = bx.box_area(boxes)
    widths, heights = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    per_image = [e["n_boxes"] for e in entries if e["positive"]]
    return {
        "n_images": len(entries),
        "n_positive": sum(e["positive"] for e in entries),
        "n_negative": sum(not e["positive"] for e in entries),
        "n_boxes": int(len(boxes)),
        "boxes_per_positive_image": {"mean": float(np.mean(per_image)), "median": float(np.median(per_image)), "max": int(max(per_image))} if per_image else {},
        "size_buckets": dict(Counter(bx.size_bucket(a) for a in areas)),
        "area_px": {"median": float(np.median(areas)), "p10": float(np.percentile(areas, 10)), "p90": float(np.percentile(areas, 90))} if len(areas) else {},
        "relative_area": {"median": float(np.median(areas / (640 * 640)))} if len(areas) else {},
        "aspect_ratio_w_over_h": {"median": float(np.median(widths / np.maximum(heights, 1))), "p10": float(np.percentile(widths / np.maximum(heights, 1), 10)), "p90": float(np.percentile(widths / np.maximum(heights, 1), 90))} if len(areas) else {},
        "liaci_split": dict(Counter(e["liaci_split"] for e in entries)),
    }


def plot(boxes: dict[str, np.ndarray], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for split, color in (("pool", "#0E6B70"), ("test", "#B4322A")):
        b = boxes[split]
        if not len(b):
            continue
        axes[0].hist(np.sqrt(bx.box_area(b)), bins=30, alpha=0.6, color=color, label=f"{split} (n={len(b)})")
        ar = (b[:, 2] - b[:, 0]) / np.maximum(b[:, 3] - b[:, 1], 1)
        axes[1].hist(np.log2(ar), bins=30, alpha=0.6, color=color, label=split)
    axes[0].axvline(32, ls="--", c="grey", lw=0.8)
    axes[0].axvline(96, ls="--", c="grey", lw=0.8)
    axes[0].set_xlabel("sqrt(box area) [px, 640x640 frame]")
    axes[0].set_ylabel("boxes")
    axes[1].set_xlabel("log2(width / height)")
    axes[0].legend(frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    proto = load_protocol()
    log = get_logger("dataset_stats")
    manifest = read_json(resolve(proto["dataset"]["manifest_file"]))
    dataset_root = resolve(proto["dataset"]["output_root"])
    boxes = collect_boxes(dataset_root, manifest)
    stats = {split: describe(boxes[split], [e for e in manifest["images"] if e["split"] == split]) for split in ("pool", "test")}
    stats["manifest_sha256"] = manifest["sha256"]
    write_json(resolve("results/dataset_stats.json"), stats)
    plot(boxes, resolve("figures/dataset_boxes.png"))
    log.info(f"pool: {stats['pool']['n_boxes']} boxes in {stats['pool']['n_positive']} positive images; test: {stats['test']['n_boxes']} boxes in {stats['test']['n_positive']}")


if __name__ == "__main__":
    main()
