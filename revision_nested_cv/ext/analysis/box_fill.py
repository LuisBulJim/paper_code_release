"""How much of its bounding box does a corrosion polygon fill? Over the 569 polygons of ext/data/labels_seg
(normalised YOLO polygon format), area by the shoelace formula divided by the area of the enclosing box."""
from pathlib import Path

import numpy as np

LABELS = Path(__file__).resolve().parents[1] / "data" / "labels_seg"


def main() -> None:
    ratios = []
    for f in sorted(LABELS.glob("*.txt")):
        for line in f.read_text().splitlines():
            v = np.array(line.split()[1:], dtype=float).reshape(-1, 2)
            if len(v) < 3:
                continue
            x, y = v[:, 0], v[:, 1]
            area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            box = (x.max() - x.min()) * (y.max() - y.min())
            if box > 0:
                ratios.append(area / box)
    r = np.array(ratios)
    print(f"polygons {len(r)} | polygon area / box area: mean {r.mean():.3f} median {np.median(r):.3f} p10 {np.percentile(r, 10):.3f} p90 {np.percentile(r, 90):.3f}")
    print(f"share of polygons filling less than half of their box: {(r < 0.5).mean():.3f}")


if __name__ == "__main__":
    main()
