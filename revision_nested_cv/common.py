"""Shared helpers for the nested-CV protocol (no torch / ultralytics dependency)."""
import cv2
import hashlib
from pathlib import Path

import numpy as np
from ensemble_boxes import weighted_boxes_fusion

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
SIZES = ["n", "m", "l"]


def view_key(mode, scale, flip):
    """Views at scale 1.0 are identical in both scale modes, so they share a key."""
    if abs(scale - 1.0) < 1e-9:
        return "flip" if flip else "std"
    return f"{mode}|{scale:g}|{int(flip)}"


def view_weight(scale):
    """Same rule as the released notebooks: 1.0 for native scale, 0.8 otherwise."""
    return 1.0 if abs(scale - 1.0) < 1e-9 else 0.8


def tta_views(mode, scales, flip=True):
    """List of (key, weight) for a TTA configuration."""
    out = []
    for s in scales:
        for f in ([False, True] if flip else [False]):
            out.append((view_key(mode, s, f), view_weight(s)))
    return out


def collect_pool(root):
    """All images/labels under root/images/{train,val,test} -> list of dicts."""
    root = Path(root)
    items = []
    for split in ["train", "val", "test"]:
        d = root / "images" / split
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMG_EXT:
                lab = root / "labels" / split / (p.stem + ".txt")
                pos = lab.exists() and any(l.strip() for l in lab.read_text().splitlines())
                items.append(dict(name=p.name, img=str(p.resolve()),
                                  lab=str(lab.resolve()) if lab.exists() else None,
                                  pos=bool(pos), orig_split=split))
    names = [i["name"] for i in items]
    assert len(names) == len(set(names)), "duplicate file names across splits"
    return items


def content_duplicates(items):
    """Groups of images with identical bytes (would leak across folds)."""
    seen, dups = {}, []
    for it in items:
        h = hashlib.md5(Path(it["img"]).read_bytes()).hexdigest()
        if h in seen:
            dups.append((seen[h], it["name"]))
        else:
            seen[h] = it["name"]
    return dups


def read_gt(lab_path):
    """YOLO label file -> normalized xyxy array (n,4)."""
    if not lab_path or not Path(lab_path).exists():
        return np.zeros((0, 4), dtype=np.float32)
    rows = []
    for line in Path(lab_path).read_text().splitlines():
        p = line.split()
        if len(p) >= 5:
            cx, cy, w, h = map(float, p[1:5])
            rows.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return np.array(rows, dtype=np.float32).reshape(-1, 4)


def wbf(members, weights, iou_thr=0.5, skip_box_thr=0.0, conf_type='avg'):
    """members: list of (boxes (n,4) normalized xyxy, scores (n,)). Returns fused boxes, scores."""
    boxes = [np.clip(np.asarray(b, dtype=float).reshape(-1, 4), 0, 1) for b, _ in members]
    scores = [np.asarray(s, dtype=float).reshape(-1) for _, s in members]
    if all(len(s) == 0 for s in scores):
        return np.zeros((0, 4)), np.zeros(0)
    labels = [np.zeros(len(s)) for s in scores]
    b, s, _ = weighted_boxes_fusion([x.tolist() for x in boxes], [x.tolist() for x in scores],
                                    [x.tolist() for x in labels], weights=list(weights),
                                    iou_thr=iou_thr, skip_box_thr=skip_box_thr, conf_type=conf_type)
    return np.asarray(b).reshape(-1, 4), np.asarray(s).reshape(-1)


def fuse(members, weights, thr, mode, conf_type="avg"):
    """
    mode 'pre'  : filter every member at conf>=thr, then WBF, no re-filter
                  (this is what notebooks 06/07 of the released code do).
    mode 'post' : WBF on all members with score>=0.01, caller filters the output at thr.
    A single member is never fused (returned as is; filtered only in 'pre').
    """
    if len(members) == 1:
        b, s = members[0]
        b = np.asarray(b).reshape(-1, 4)
        s = np.asarray(s).reshape(-1)
        if mode == "pre":
            k = s >= thr
            return b[k], s[k]
        return b, s
    if mode == "pre":
        members = [(np.asarray(b).reshape(-1, 4)[np.asarray(s) >= thr], np.asarray(s)[np.asarray(s) >= thr])
                   for b, s in members]
        return wbf(members, weights, skip_box_thr=0.0, conf_type=conf_type)
    return wbf(members, weights, skip_box_thr=0.01, conf_type=conf_type)


# ----------------------------------------------------------------------------- rotation TTA
def rotate_image(img, angle_deg):
    """Rotate around the image center, same canvas size (small-angle crop at the borders)."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101), M


def unrotate_boxes_norm(boxes, M, w, h, angle_deg):
    """
    boxes: (n,4) normalized xyxy in the ROTATED image. Returns axis-aligned boxes, normalized,
    in the ORIGINAL image frame (bounding rectangle of the 4 inverse-rotated corners; an
    approximation, standard for rotation-TTA with axis-aligned detectors).
    """
    if len(boxes) == 0:
        return boxes
    Minv = cv2.invertAffineTransform(M)
    out = np.zeros_like(boxes)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        pts = np.array([[x1 * w, y1 * h], [x2 * w, y1 * h], [x2 * w, y2 * h], [x1 * w, y2 * h]], dtype=np.float32)
        pts = (Minv[:, :2] @ pts.T + Minv[:, 2:3]).T
        out[i] = [pts[:, 0].min() / w, pts[:, 1].min() / h, pts[:, 0].max() / w, pts[:, 1].max() / h]
    return np.clip(out, 0, 1)


try:
    import cv2  # noqa: F401 (re-import guard; already imported by callers)
except Exception:
    pass
