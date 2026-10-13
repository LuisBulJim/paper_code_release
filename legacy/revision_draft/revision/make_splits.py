"""Stratified K-Fold partitions of the training pool for every protocol seed.

Reads data/manifest_v2.json, stratifies on positive/negative, and writes data/folds_v2.json:
  {"test": [...], "pool_size": N, "manifest_sha256": ..., "n_folds": 5,
   "seeds": {"0": [{"fold": 0, "train": [...], "val": [...]}, ...], ...}}
The test set is copied verbatim from the manifest and never enters any fold.
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.model_selection import StratifiedKFold

from revision.protocol import get_logger, load_protocol, read_json, resolve, write_json


def make_folds(pool: list[str], strata: list[int], n_folds: int, seed: int) -> list[dict]:
    names = np.array(pool)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [
        {"fold": k, "train": sorted(names[tr].tolist()), "val": sorted(names[va].tolist())}
        for k, (tr, va) in enumerate(skf.split(names, np.array(strata)))
    ]


def verify_folds(folds: list[dict], pool: set[str], test: set[str]) -> None:
    seen_val: list[str] = []
    for f in folds:
        tr, va = set(f["train"]), set(f["val"])
        assert not tr & va, f"fold {f['fold']}: train/val overlap"
        assert tr | va == pool, f"fold {f['fold']}: train+val != pool"
        assert not (tr | va) & test, f"fold {f['fold']}: test leaked into fold"
        seen_val.extend(f["val"])
    assert sorted(seen_val) == sorted(pool), "each pool image must be in val exactly once"


def build_splits(proto: dict, log) -> dict:
    manifest = read_json(resolve(proto["dataset"]["manifest_file"]))
    pool_entries = sorted((e for e in manifest["images"] if e["split"] == "pool"), key=lambda e: e["image"])
    pool = [e["image"] for e in pool_entries]
    strata = [int(e["positive"]) for e in pool_entries]
    test = sorted(e["image"] for e in manifest["images"] if e["split"] == "test")
    cfg = proto["splits"]
    out = {"test": test, "pool_size": len(pool), "manifest_sha256": manifest["sha256"], "n_folds": cfg["n_folds"], "seeds": {}}
    for seed in cfg["seeds"]:
        folds = make_folds(pool, strata, cfg["n_folds"], seed)
        verify_folds(folds, set(pool), set(test))
        out["seeds"][str(seed)] = folds
        pos_per_val = [sum(strata[pool.index(n)] for n in f["val"]) for f in folds]
        log.info(f"seed {seed}: val sizes {[len(f['val']) for f in folds]}, positives per val {pos_per_val}")
    write_json(resolve(cfg["folds_file"]), out)
    log.info(f"written {resolve(cfg['folds_file'])}: pool {len(pool)}, test {len(test)}, seeds {cfg['seeds']}")
    return out


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    build_splits(load_protocol(), get_logger("make_splits", resolve("results/logs/make_splits.log")))


if __name__ == "__main__":
    main()
