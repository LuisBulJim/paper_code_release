"""Are the extra LIACI images of x2_multi closer to the test images than the pool training images already are?

For every outer fold and test image: highest ResNet-101 cosine similarity to the pool training images (train + val)
and to the extra images x2_multi added. If the extra images were near-duplicates of test frames, x2's gain would be
leakage rather than learning.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from similarity import load_sim, num  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    sim, idx = load_sim()
    splits = json.load(open(ROOT / "out" / "splits.json"))
    folds = json.load(open(ROOT / "ext" / "data" / "folds.json"))
    ids = np.array(sorted(num(n) for n in idx))
    consecutive = [sim[idx[a], idx[b]] for a, b in zip(sorted(idx, key=num), sorted(idx, key=num)[1:]) if num(b) - num(a) == 1]
    s_pool, s_extra, gap = [], [], []
    for r in map(str, range(3)):
        for k in range(5):
            sp = splits[r][k]
            pool_tr = [idx[n] for n in sp["train"] + sp["val"]]
            extra = [idx[n] for n in folds[r][k]["extra_train"]]
            extra_ids = np.array([num(n) for n in folds[r][k]["extra_train"]])
            for n in sp["test"]:
                s_pool.append(sim[idx[n], pool_tr].max())
                s_extra.append(sim[idx[n], extra].max())
                gap.append(np.abs(extra_ids - num(n)).min())
    s_pool, s_extra, gap = map(np.array, (s_pool, s_extra, gap))
    q = lambda a: " / ".join(f"{np.percentile(a, p):.3f}" for p in (10, 50, 90))
    print(f"test images x repeats: {len(s_pool)}; LIACI images with embeddings: {len(ids)}")
    print(f"reference, consecutive frames (index +1): median cosine {np.median(consecutive):.3f}")
    print(f"nearest pool training image  p10/p50/p90: {q(s_pool)}")
    print(f"nearest extra image          p10/p50/p90: {q(s_extra)}")
    print(f"extra closer than pool: {np.mean(s_extra > s_pool):.1%}; by more than 0.02: {np.mean(s_extra - s_pool > 0.02):.1%}")
    for t in (0.90, 0.95):
        print(f"nearest similarity > {t}: pool {np.mean(s_pool > t):.1%} | extra {np.mean(s_extra > t):.1%} | consecutive frames {np.mean(np.array(consecutive) > t):.1%}")
    print(f"index gap test -> nearest extra image: min {gap.min()}, p10 {np.percentile(gap, 10):.0f}, median {np.median(gap):.0f}")


if __name__ == "__main__":
    main()
