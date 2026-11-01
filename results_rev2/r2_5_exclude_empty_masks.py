"""R2.5 -- false-mask rate on negatives after excluding the two official-test images whose LIACI corrosion mask is empty.

image_0105 and image_0763 carry a corrosion annotation in LIACI's COCO file and show visible rust, but the
masks/corrosion/<stem>.bmp distributed with LIACI is entirely black (checked by the authors on the original release,
28-Oct-2026; the manifests record mask_area_fraction 0.0). They cannot be scored as positives (no ground-truth
mask) and must not be scored as negatives, so they are removed: 191 -> 189 images, 21 positives + 168 negatives.
Positives are unaffected. Reads the per-image sam3_negatives.csv of every SAM 3 run (paper r1/ext and, when present,
the R2.5 farthest-model runs) and recomputes the false-mask rate and mean false-area fraction.

Usage (repository root): python results_rev2/r2_5_exclude_empty_masks.py
Writes results_rev2/r2_5/negatives_excluding_empty_masks.csv
"""
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXCLUDE = {"image_0105.jpg", "image_0763.jpg"}
RUNS = {
    "text prompts (paper)": "revision_nested_cv/ext/results/sam3_text",
    "GT box prompts (paper)": "revision_nested_cv/ext/results/sam3_gt_boxes",
    "detector boxes conf 0.25 (paper map)": "revision_nested_cv/r1/results",
    "detector boxes val thr 0.1 (paper map)": "revision_nested_cv/r1/results/sam3_valthr",
    "detector boxes conf 0.25 (farthest map)": "results_rev2/r2_5/metrics_std",
    "detector boxes val thr 0.1 (farthest map)": "results_rev2/r2_5/metrics_valthr",
}


def main():
    rows = []
    for name, d in RUNS.items():
        p = REPO / d / "sam3_negatives.csv"
        if not p.exists():
            print(f"skip {name}: {p} not found")
            continue
        neg = list(csv.DictReader(open(p, encoding="utf-8")))
        kept = [n for n in neg if n["image"] not in EXCLUDE]
        fm = lambda ns: sum(n["false_mask"] == "True" for n in ns)  # noqa: E731
        rows.append({"run": name, "n_neg_before": len(neg), "false_masks_before": fm(neg),
                     "rate_before": round(fm(neg) / len(neg), 4), "excluded_present": len(neg) - len(kept),
                     "excluded_with_false_mask": fm([n for n in neg if n["image"] in EXCLUDE]),
                     "n_neg_after": len(kept), "false_masks_after": fm(kept), "rate_after": round(fm(kept) / len(kept), 4),
                     "mean_false_area_after": round(sum(float(n["false_area_fraction"]) for n in kept) / len(kept), 5)})
    out = REPO / "results_rev2/r2_5/negatives_excluding_empty_masks.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
