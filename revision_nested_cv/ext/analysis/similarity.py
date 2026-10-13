"""Residual train/test proximity in the grouped folds, using the ResNet-101 cosine similarities shipped with LIACI.

Reports how similar neighbouring filename indices are, how close each test image is to its nearest
training image under each protocol, and whether detection success depends on that proximity.
"""
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from diagnostics import COCO, FOLDS, REPEATS, ROOT, ap_at, iou_matrix  # noqa: E402

SIM = COCO.parent / "embeddings_resnet101.json"   # shipped with the LIACI release, next to coco-labels.json


def num(name: str) -> int:
    return int(re.findall(r"\d+", name)[-1])


def load_sim() -> tuple[np.ndarray, dict[str, int]]:
    emb = json.load(open(SIM, encoding="utf-8"))["embeddings"]
    header = sorted(emb)
    e = np.array([emb[n] for n in header], dtype=np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    return e @ e.T, {n: i for i, n in enumerate(header)}


def folds(suffix: str) -> list[dict]:
    """One dict per (repeat, fold): test records and the names used for training (train + val)."""
    pool = {im["image"] for im in json.load(open(ROOT / "r1" / "data" / "nested_manifest.json"))["images"]}
    out = []
    for r in range(REPEATS):
        for k in range(FOLDS):
            d = pickle.load(open(ROOT / "out" / "preds" / f"direct_r{r}_f{k}_m.pkl{suffix}", "rb"))
            test = {n: (np.asarray(v["std"][0], dtype=np.float32).reshape(-1, 4), np.asarray(v["std"][1], dtype=np.float32), np.asarray(d["gt"][n], dtype=np.float32).reshape(-1, 4)) for n, v in d["test"].items()}
            out.append({"r": r, "k": k, "test": test, "train": sorted(pool - set(test))})
    return out


def proximity(fs: list[dict], sim: np.ndarray, idx: dict[str, int]) -> list[tuple[str, int, float, int, tuple]]:
    rows = []
    for f in fs:
        tr = np.array([idx[n] for n in f["train"]])
        tr_ids = np.array([num(n) for n in f["train"]])
        for n, rec in f["test"].items():
            rows.append((n, f["r"], float(sim[idx[n], tr].max()), int(np.abs(tr_ids - num(n)).min()), rec))
    return rows


def recall_at(recs: list[tuple], conf: float = 0.25) -> tuple[float, int]:
    tp = n_gt = 0
    for boxes, sc, gt in recs:
        n_gt += len(gt)
        b = boxes[sc >= conf]
        if len(gt) and len(b):
            ious = iou_matrix(b[np.argsort(-sc[sc >= conf])], gt)
            taken = np.zeros(len(gt), dtype=bool)
            for i in range(len(b)):
                cand = np.where(~taken, ious[i], -1.0)
                j = int(cand.argmax())
                if cand[j] >= 0.5:
                    taken[j] = True
            tp += int(taken.sum())
    return tp / max(n_gt, 1), n_gt


def main() -> None:
    sim, idx = load_sim()
    names = sorted(idx, key=num)
    order = np.array([idx[n] for n in names])
    s = sim[np.ix_(order, order)]
    n = len(names)
    rng = np.random.default_rng(0)
    rand = s[rng.integers(0, n, 20000), rng.integers(0, n, 20000)]
    print("cosine similarity between images k filename-indices apart (median / p10):")
    for k in (1, 2, 5, 10, 15, 30, 50, 100):
        d = np.diagonal(s, offset=k)
        print(f"  k={k:3d}  median {np.median(d):.3f}  p10 {np.percentile(d, 10):.3f}  share>=0.90 {np.mean(d >= 0.90):.2f}")
    print(f"  random pairs  median {np.median(rand):.3f}  p90 {np.percentile(rand, 90):.3f}  share>=0.90 {np.mean(rand >= 0.90):.3f}")

    for label, suffix in (("grouped (current)", ""), ("ungrouped (28-Sep)", ".bak")):
        rows = proximity(folds(suffix), sim, idx)
        sims = np.array([r[2] for r in rows])
        dist = np.array([r[3] for r in rows])
        print(f"\n{label}: {len(rows)} test evaluations")
        print(f"  nearest-train similarity: median {np.median(sims):.3f}  share>=0.90 {np.mean(sims >= 0.90):.2f}  share>=0.95 {np.mean(sims >= 0.95):.2f}")
        print(f"  nearest-train index distance: median {np.median(dist):.0f}  share<=2 {np.mean(dist <= 2):.2f}  share<=5 {np.mean(dist <= 5):.2f}  share<=15 {np.mean(dist <= 15):.2f}")
        edges = np.quantile(sims, [0, 1 / 3, 2 / 3, 1])
        for i in range(3):
            sel = [r for r in rows if edges[i] <= r[2] <= edges[i + 1]] if i == 2 else [r for r in rows if edges[i] <= r[2] < edges[i + 1]]
            rec, n_gt = recall_at([r[4] for r in sel])
            ap = float(np.mean([ap_at({f"{q[0]}|{q[1]}": q[4] for q in sel if q[1] == rep}, 0.5)[0] for rep in range(REPEATS)]))
            print(f"  similarity tertile {i + 1} [{edges[i]:.2f}, {edges[i + 1]:.2f}]: images {len(sel)}, GT {n_gt}, recall@0.25 {rec:.3f}, AP50 {ap:.3f}")
        for lo, hi in ((0, 5), (6, 15), (16, 10**6)):
            sel = [r for r in rows if lo <= r[3] <= hi]
            rec, n_gt = recall_at([r[4] for r in sel])
            ap = float(np.mean([ap_at({f"{q[0]}|{q[1]}": q[4] for q in sel if q[1] == rep}, 0.5)[0] for rep in range(REPEATS)]))
            print(f"  index distance {lo}-{hi if hi < 10**6 else 'inf'}: images {len(sel)}, GT {n_gt}, recall@0.25 {rec:.3f}, AP50 {ap:.3f}")


if __name__ == "__main__":
    main()
