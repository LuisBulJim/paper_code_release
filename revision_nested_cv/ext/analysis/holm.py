"""Holm correction of the paired differences in ext/results_final/deltas.csv, per family of comparisons.

  pre-registered  x1-x4 against the baseline (standard pass, the primary metric)
  second round    x6, e1 and c4 against the baseline (decided after the first round)
  ensembles       e1 against c4: do the variants add anything over an ensemble of baseline models?
p is the one-sided bootstrap share of replicates with a difference <= 0. With 1000 replicates a share of 0 only means
p < 0.001, so the adjustment uses that bound and the adjusted value is reported as an upper bound ("<").
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ITERS = 1000
FAMILIES = {
    "pre-registered": ["x1_recipe - x0_baseline", "x2_multi - x0_baseline", "x3_seg - x0_baseline", "x4_seg_cp - x0_baseline"],
    "second round": ["x6_multi_seg - x0_baseline", "e1_ensemble - x0_baseline", "c4_baselines - x0_baseline"],
    "ensembles": ["e1_ensemble - c4_baselines"],
}


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def main() -> None:
    rows = list(csv.DictReader(open(ROOT / "ext" / "results_final" / "deltas.csv")))
    out = []
    for strategy in ("standard", "tta"):
        for fam, comps in FAMILIES.items():
            sel = [next(r for r in rows if r["comparison"] == c and r["strategy"] == strategy) for c in comps]
            raw = [float(r["map50_p_le0"]) for r in sel]
            ps = [max(p, 1 / ITERS) for p in raw]
            for r, p, a in zip(sel, raw, holm(ps)):
                out.append({"family": fam, "strategy": strategy, "comparison": r["comparison"], "delta": r["map50_delta"],
                            "ci_lo": r["map50_ci_lo"], "ci_hi": r["map50_ci_hi"], "p_one_sided": f"<{1 / ITERS:g}" if p == 0 else p,
                            "p_holm": f"<{a:.3f}" if p == 0 else round(a, 3)})
                print(f"{strategy:8s} {fam:15s} {r['comparison']:30s} {float(r['map50_delta']):+.4f} [{float(r['map50_ci_lo']):+.4f}, {float(r['map50_ci_hi']):+.4f}] "
                      f"p {out[-1]['p_one_sided']} | Holm {out[-1]['p_holm']}")
    with open(ROOT / "ext" / "results_final" / "holm.csv", "w", newline="\n", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)


if __name__ == "__main__":
    main()
