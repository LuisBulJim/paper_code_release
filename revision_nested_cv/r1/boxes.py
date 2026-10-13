"""Mask-to-box conversion, YOLO label I/O and IoU helpers shared by every revision script.

The conversion replicates the submitted pipeline exactly (0.Tratamiento_dataset_acuatico.ipynb):
mask resized to 640x640 with nearest neighbour, binarised at >0, external contours
(cv2.RETR_EXTERNAL, CHAIN_APPROX_SIMPLE), contours with fewer than 3 vertices dropped, one box
(minimal enclosing rectangle) per contour. One connected component == one instance.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SMALL_AREA = 32**2
MEDIUM_AREA = 96**2


def mask_to_polygons(mask: np.ndarray) -> list[np.ndarray]:
    """Return external contours (N,2 float arrays, pixel coords) of a single-channel mask."""
    _, binary = cv2.threshold(mask.astype(np.uint8), 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.squeeze(axis=1).astype(float) for c in contours if c.shape[0] >= 3]


def polygon_to_box(poly: np.ndarray) -> tuple[float, float, float, float]:
    return float(poly[:, 0].min()), float(poly[:, 1].min()), float(poly[:, 0].max()), float(poly[:, 1].max())


def mask_to_boxes(mask: np.ndarray) -> np.ndarray:
    """(n,4) xyxy pixel boxes, one per external contour."""
    polys = mask_to_polygons(mask)
    if not polys:
        return np.zeros((0, 4), dtype=float)
    return np.array([polygon_to_box(p) for p in polys], dtype=float)


def boxes_to_yolo_lines(boxes: np.ndarray, width: int, height: int, class_id: int = 0) -> list[str]:
    """Normalised 'class xc yc w h' lines with 6 decimals, matching the original label format."""
    lines = []
    for x1, y1, x2, y2 in np.asarray(boxes, dtype=float).reshape(-1, 4):
        xc = (x1 + x2) / 2 / width
        yc = (y1 + y2) / 2 / height
        w = (x2 - x1) / width
        h = (y2 - y1) / height
        lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return lines


def yolo_to_xyxy(rows: np.ndarray, width: int, height: int) -> np.ndarray:
    rows = np.asarray(rows, dtype=float).reshape(-1, 5)
    xc, yc, w, h = rows[:, 1] * width, rows[:, 2] * height, rows[:, 3] * width, rows[:, 4] * height
    return np.stack([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], axis=1)


def read_yolo_labels(path: Path, width: int, height: int, with_classes: bool = False):
    """(n,4) xyxy pixel boxes from a YOLO label file (and the (n,) class ids when with_classes);
    empty arrays when the file is missing or empty."""
    path = Path(path)
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 5:
                rows.append([float(v) for v in parts[:5]])
    boxes = yolo_to_xyxy(np.array(rows), width, height) if rows else np.zeros((0, 4), dtype=float)
    if not with_classes:
        return boxes
    classes = np.array([int(r[0]) for r in rows], dtype=int) if rows else np.zeros(0, dtype=int)
    return boxes, classes


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (n,4) and (m,4) xyxy boxes."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=float)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def box_area(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def size_bucket(area: float) -> str:
    """COCO size buckets on pixel area."""
    if area < SMALL_AREA:
        return "small"
    if area < MEDIUM_AREA:
        return "medium"
    return "large"
