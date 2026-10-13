"""Paired comparison of two training protocols (e.g. Sequential vs Direct Transfer) fold by fold.

Both protocols use the same folds and seeds, so for every model the differences (B - A) are paired over the
15 (seed, fold) checkpoints: mean difference with 95% t-interval, Wilcoxon signed-rank p-value, on the
validation metrics (cv_metrics.csv) and on the test metrics (test_metrics.csv).
Output: results/protocol_comparison.csv
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy import stats

from revision.collect_runs import t_interval, write_csv
from revision.make_tables import read_csv
from revision.protocol import get_logger, resolve

KEYS = ("map50", "map5095", "precision", "recall", "f1")


def paired_rows(rows: list[dict], stage: str, tag_a: str, tag_b: str) -> list[dict]:
    out = []
    for model in sorted({r["model"] for r in rows}):
        a = {(r["seed"], r["fold"]): r for r in rows if r["tag"] == tag_a and r["model"] == model}
        b = {(r["seed"], r["fold"]): r for r in rows if r["tag"] == tag_b and r["model"] == model}
        keys = sorted(set(a) & set(b))
        if not keys:
            continue
        row = {"stage": stage, "model": model, "comparison": f"{tag_b}-{tag_a}", "n_pairs": len(keys)}
        for k in KEYS:
            d = np.array([float(b[key][k]) - float(a[key][k]) for key in keys])
            mean, lo, hi = t_interval(d.tolist())
            p = float(stats.wilcoxon(d).pvalue) if len(d) > 1 and np.any(d != 0) else float("nan")
            row.update({f"{k}_delta": round(mean, 4), f"{k}_ci_lo": round(lo, 4), f"{k}_ci_hi": round(hi, 4), f"{k}_wilcoxon_p": round(p, 4), f"{k}_wins": int((d > 0).sum())})
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default="dtl", help="reference protocol tag")
    ap.add_argument("--b", default="stl", help="compared protocol tag")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    log = get_logger("compare_protocols")
    res = resolve(args.results)
    rows = paired_rows(read_csv(res / "cv_metrics.csv"), "val", args.a, args.b) + paired_rows(read_csv(res / "test_metrics.csv"), "test", args.a, args.b)
    if not rows:
        raise SystemExit(f"no paired runs between tags {args.a} and {args.b}")
    write_csv(res / "protocol_comparison.csv", rows)
    for r in rows:
        log.info(f"{r['stage']} {r['model']} {r['comparison']} (n={r['n_pairs']}): delta mAP50 {r['map50_delta']:+.4f} [{r['map50_ci_lo']:+.4f}, {r['map50_ci_hi']:+.4f}] Wilcoxon p={r['map50_wilcoxon_p']}")


if __name__ == "__main__":
    main()
