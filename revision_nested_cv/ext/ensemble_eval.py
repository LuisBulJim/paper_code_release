"""e1_ensemble: weighted box fusion of the four pre-registered variants, stored as one more set of nested-CV records.

Rules fixed before the ensemble was scored:
  * members: x1_recipe, x2_multi, x3_seg and x4_seg_cp, all of them, so nothing is selected on test results
  * fusion: the model-ensemble WBF that r1 chose on inner validation for the baseline sizes
    (r1/results/nested_grouped/inference_choice.json: equal weights, IoU 0.5; skip threshold of the protocol)
  * every stored view is fused across members, so ext/evaluate_ext.py scores the ensemble with and without TTA
    exactly like a variant (threshold on its own validation, paired cluster bootstrap against the baseline)

Usage (from revision_nested_cv/, CPU):
  python ext/ensemble_eval.py                     # writes ext/out/preds/e1_ensemble_r{r}_f{k}_m.pkl
  python ext/evaluate_ext.py                      # then scores it with the variants
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from r1.predict_dump import fuse  # noqa: E402
from r1.protocol import load_protocol, resolve  # noqa: E402

MEMBERS = ["x1_recipe", "x2_multi", "x3_seg", "x4_seg_cp"]
REPEATS, FOLDS = 3, 5


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--member", nargs="+", default=[f"{m}={HERE / 'out' / 'preds' / m}_r{{r}}_f{{k}}_m.pkl" for m in MEMBERS],
                    help="name=pattern pairs (default: the four variants in ext/out/preds)")
    ap.add_argument("--name", default="e1_ensemble")
    ap.add_argument("--out", default=HERE / "out" / "preds", type=Path)
    ap.add_argument("--only", nargs="+", default=None, help="subset of r,k pairs (smoke tests)")
    return ap.parse_args()


def fuse_views(member_views: list[dict], iou: float, skip: float) -> dict:
    out = {}
    for key in member_views[0]:
        parts = []
        for views in member_views:
            b, s = views[key]
            b, s = np.asarray(b, dtype=float).reshape(-1, 4), np.asarray(s, dtype=float).reshape(-1)
            parts.append((b, s, np.zeros(len(s), dtype=int), 1.0))
        fb, fs, _ = fuse(parts, 1, 1, iou, skip)
        out[key] = (fb.astype(np.float32), fs.astype(np.float32))
    return out


def main() -> None:
    args = parse()
    members = dict(m.split("=", 1) for m in args.member)
    proto = load_protocol()
    choice = json.load(open(resolve("r1/results/nested_grouped/inference_choice.json")))
    iou, skip = choice["wbf"]["iou"], proto["inference"]["wbf_models"]["skip"]
    assert choice["wbf"]["weights"] is None, "r1 chose unequal weights; this script only implements equal weights"
    only = {tuple(int(x) for x in p.split(",")) for p in args.only} if args.only else None
    args.out.mkdir(parents=True, exist_ok=True)
    done = 0
    for r in range(REPEATS):
        for k in range(FOLDS):
            if only and (r, k) not in only:
                continue
            paths = {m: Path(p.format(r=r, k=k)) for m, p in members.items()}
            missing = [str(p) for p in paths.values() if not p.exists()]
            if missing:
                print(f"skip r{r} f{k}: missing {missing}")
                continue
            recs = {m: pickle.load(open(p, "rb")) for m, p in paths.items()}
            first = next(iter(recs.values()))
            for m, rec in recs.items():
                if sorted(rec["test"]) != sorted(first["test"]) or sorted(rec["val"]) != sorted(first["val"]):
                    raise ValueError(f"{m} r{r} f{k}: test/val images differ between members")
            out = {"meta": {"tag": args.name, "repeat": r, "fold": k, "members": members, "wbf_iou": iou, "wbf_skip": skip,
                            "test": sorted(first["test"]), "val": sorted(first["val"])},
                   "val": {}, "test": {}, "gt": first["gt"]}
            for part in ("val", "test"):
                for n in first[part]:
                    out[part][n] = fuse_views([rec[part][n] for rec in recs.values()], iou, skip)
            with open(args.out / f"{args.name}_r{r}_f{k}_m.pkl", "wb") as f:
                pickle.dump(out, f)
            done += 1
            print(f"  saved: {args.name}_r{r}_f{k}_m.pkl")
    print(f"{args.name}: {done} folds written (WBF IoU {iou}, skip {skip}, members {list(members)})")


if __name__ == "__main__":
    main()
