"""Re-evaluation of the repeated nested cross-validation of revision_nested_cv with the single evaluator.

nested_cv.py stored, for each of its 90 fold models (protocol direct|sequential x size n|m|l x 3 repeats x 5 outer
folds), the raw predictions of every test-time view (conf 0.001, normalised xyxy) on the inner-validation and
outer-test images of the fold. The views 'std', 'flip' and 'imgsz|<s>|<flip>' are exactly the passes of
predict_dump.predict_tta (network input 640*s, horizontal flip mapped back), so the standard / TTA / WBF detections
of the r1 protocol are rebuilt here without a GPU and scored with evaluate.py.

  prepare   r1/data/nested_manifest.json: sha256 of the 261 images and labels; fails unless the ground truth
            stored in every .pkl equals the labels
  check     splits.json (no overlap, every image tested once per repeat, no video group across train and test),
            every .pkl against its split, every checkpoint YOLO26 and trained on the workspace of its own fold
  evaluate  per variant (grouped = final run, ungrouped = earlier run kept as .pkl.bak):
              validation sweep of the TTA / WBF candidates -> inference_choice.json (rule of sweep_inference.py)
              out-of-fold test metrics per repeat: mAP@50, mAP@[.5:.95], P/R/F1 at conf_fixed, image-level AUROC,
              sensitivity and specificity; mean and SD over the repeats
              cluster bootstrap over video groups (the same resampled groups for every repeat and strategy):
              95% CI of each metric averaged over repeats, and paired deltas between strategies
              out-of-fold dumps (one JSON per image, evaluate.py format) for failure_analysis and the figures
  link      r1/runs/<tag>/yolo26<s>_seed<repeat>/fold<k>/{weights/best.pt, run_info.json, DONE} pointing at
            out/runs/<tag>_r<repeat>_f<k>_<s>/weights/best.pt (layout read by trashcan_ood and benchmark_latency)
  sam3-map  detector per SAM 3 image: images of the 261 go to the fold model that had them in its outer test
            fold, the rest to fold 0 (none of them was in any training set)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import os
import pickle
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from r1 import boxes as bx
from r1 import evaluate as ev
from r1 import predict_dump as pdump
from r1.collect_runs import write_csv
from r1.protocol import dataset_images, get_logger, label_path, load_protocol, read_json, resolve, sha256_file, write_json
from r1.sweep_inference import CONF_GRID, TTA_GRID, WBF_IOU_GRID, WBF_WEIGHT_MODES, choose_tta, choose_wbf, summarize

MODEL = {"n": "yolo26n", "m": "yolo26m", "l": "yolo26l"}
SIZE_OF = {v: k for k, v in MODEL.items()}
METRICS = ("map50", "map5095", "precision", "recall", "f1", "precision_vt", "recall_vt", "f1_vt", "auroc", "sensitivity", "specificity")
COCO_IOUS = np.linspace(0.5, 0.95, 10)
COCO_RECALLS = np.linspace(0.0, 1.0, 101)
RESULTS = "r1/results"


# ----------------------------------------------------------------------------- loading

def pkl_path(proto: dict, tag: str, size: str, r: int, k: int, suffix: str) -> Path:
    return resolve(proto["nested"]["out"]) / "preds" / f"{tag}_r{r}_f{k}_{size}{suffix}"


def load_records(proto: dict, suffix: str, tags: list[str] | None = None) -> dict[tuple, dict]:
    """(tag, size, repeat, fold) -> record of nested_cv.py: {'meta', 'val', 'test', 'gt'}."""
    nc = proto["nested"]
    recs = {}
    for tag in tags or nc["tags"]:
        for size in nc["sizes"]:
            for r in range(nc["repeats"]):
                for k in range(nc["folds"]):
                    p = pkl_path(proto, tag, size, r, k, suffix)
                    if not p.exists():
                        raise FileNotFoundError(p)
                    with open(p, "rb") as f:
                        recs[(tag, size, r, k)] = pickle.load(f)
    return recs


def image_sizes(proto: dict, names: list[str]) -> dict[str, tuple[int, int]]:
    paths = dataset_images(proto)
    out = {}
    for n in names:
        with Image.open(paths[n]) as im:
            out[n] = im.size
    return out


def read_labels(image: Path, w: int, h: int) -> np.ndarray:
    return bx.read_yolo_labels(label_path(image), w, h)


def load_gt(proto: dict, names: list[str]) -> ev.GT:
    """GT of dataset_yolo for the given names (images live in images/train and images/test)."""
    paths = dataset_images(proto)
    gt = {}
    for n in names:
        with Image.open(paths[n]) as im:
            w, h = im.size
        b = read_labels(paths[n], w, h)
        gt[n] = {"boxes": b, "classes": np.zeros(len(b), dtype=int), "width": w, "height": h}
    return gt


def gt_from_records(recs: dict[tuple, dict], sizes: dict[str, tuple[int, int]]) -> ev.GT:
    gt = {}
    for rec in recs.values():
        for n, g in rec["gt"].items():
            if n not in gt:
                w, h = sizes[n]
                b = np.asarray(g, dtype=float).reshape(-1, 4) * [w, h, w, h]
                gt[n] = {"boxes": b, "classes": np.zeros(len(b), dtype=int), "width": w, "height": h}
    return gt


# ----------------------------------------------------------------------------- detections from the stored views

def view_key(scale: float, flip: bool) -> str:
    """Key of nested_cv.predict_views for a pass at network input 640*scale ('imgsz' mode)."""
    if abs(scale - 1.0) < 1e-9:
        return "flip" if flip else "std"
    return f"imgsz|{scale:g}|{int(flip)}"


def view_px(views: dict, key: str, w: int, h: int) -> ev.Preds:
    b, s = views[key]
    b = np.asarray(b, dtype=float).reshape(-1, 4) * [w, h, w, h]
    s = np.asarray(s, dtype=float).reshape(-1)
    return {"boxes": b, "scores": s, "classes": np.zeros(len(s), dtype=int)}


def tta_keys(cfg: dict) -> list[tuple[str, float]]:
    out = []
    for s in cfg["scales"]:
        weight = cfg["weight_native"] if float(s) == 1.0 else cfg["weight_rescaled"]
        out.append((view_key(s, False), weight))
        if cfg["flip"]:
            out.append((view_key(s, True), weight))
    return out


def available_tta(views: dict) -> dict[str, dict]:
    """TTA_GRID configurations whose passes were all stored by nested_cv.py."""
    return {name: cfg for name, cfg in TTA_GRID.items() if cfg and all(k in views for k, _ in tta_keys(cfg))}


def det_standard(views: dict, w: int, h: int) -> dict:
    return view_px(views, "std", w, h)


def det_tta(views: dict, cfg: dict, iou: float, skip: float, w: int, h: int) -> dict:
    parts = []
    for key, weight in tta_keys(cfg):
        d = view_px(views, key, w, h)
        parts.append((d["boxes"], d["scores"], d["classes"], weight))
    b, s, c = pdump.fuse(parts, w, h, iou, skip)
    return {"boxes": b, "scores": s, "classes": c}


def det_wbf(member_views: list[dict], weights: list[float], iou: float, skip: float, w: int, h: int) -> dict:
    parts = []
    for views, weight in zip(member_views, weights):
        d = view_px(views, "std", w, h)
        parts.append((d["boxes"], d["scores"], d["classes"], weight))
    b, s, c = pdump.fuse(parts, w, h, iou, skip)
    return {"boxes": b, "scores": s, "classes": c}


# ----------------------------------------------------------------------------- fast COCO resampling

def coco_per_image(preds: ev.Preds, gt: ev.GT, names: list[str]) -> dict[str, dict]:
    """Per-image COCOeval.evaluateImg output (area 'all', maxDets 100) of each name, computed once with pycocotools.
    coco_from_per_image() then reproduces COCOeval.accumulate/summarize for any multiset of these images."""
    import contextlib
    import io

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt_dict, dets = ev._coco_structs(preds, gt, names)
    out = {}
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = coco_gt_dict
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(dets) if dets else COCO()
        if not dets:
            coco_dt.dataset = {"images": coco_gt_dict["images"], "annotations": [], "categories": coco_gt_dict["categories"]}
            coco_dt.createIndex()
        e = COCOeval(coco_gt, coco_dt, iouType="bbox")
        e.evaluate()
    n_t = len(e.params.iouThrs)
    for item in e.evalImgs:
        if item is None or item["aRng"] != e.params.areaRng[0] or item["maxDet"] != e.params.maxDets[-1]:
            continue
        name = names[item["image_id"] - 1]
        out[name] = {"scores": np.asarray(item["dtScores"], dtype=float), "matched": np.asarray(item["dtMatches"]).reshape(n_t, -1) > 0, "dt_ignore": np.asarray(item["dtIgnore"]).reshape(n_t, -1).astype(bool), "gt_ignore": np.asarray(item["gtIgnore"]).astype(bool)}
    empty = {"scores": np.zeros(0), "matched": np.zeros((n_t, 0), dtype=bool), "dt_ignore": np.zeros((n_t, 0), dtype=bool), "gt_ignore": np.zeros(0, dtype=bool)}
    return {n: out.get(n, empty) for n in names}


def coco_from_per_image(per_image: dict[str, dict], names: list[str]) -> dict[str, float]:
    """COCOeval.accumulate + summarize (single category, area 'all', maxDets 100) over the listed images,
    duplicates allowed. Same arithmetic as pycocotools, including the 101-point interpolation."""
    items = [per_image[n] for n in names]
    scores = np.concatenate([it["scores"] for it in items]) if items else np.zeros(0)
    n_gt = int(sum(int((~it["gt_ignore"]).sum()) for it in items))
    if n_gt == 0:
        return {"map50": float("nan"), "map5095": float("nan")}
    order = np.argsort(-scores, kind="mergesort")
    matched = np.concatenate([it["matched"] for it in items], axis=1)[:, order]
    dt_ignore = np.concatenate([it["dt_ignore"] for it in items], axis=1)[:, order]
    tps = np.logical_and(matched, ~dt_ignore)
    fps = np.logical_and(~matched, ~dt_ignore)
    tp_sum = np.cumsum(tps, axis=1).astype(float)
    fp_sum = np.cumsum(fps, axis=1).astype(float)
    precision = np.zeros((len(COCO_IOUS), len(COCO_RECALLS)))
    for t in range(len(COCO_IOUS)):
        tp, fp = tp_sum[t], fp_sum[t]
        if len(tp) == 0:
            continue
        rc = tp / n_gt
        pr = tp / (fp + tp + np.spacing(1))
        pr = np.maximum.accumulate(pr[::-1])[::-1]
        idx = np.searchsorted(rc, COCO_RECALLS, side="left")
        valid = idx < len(pr)
        precision[t, valid] = pr[idx[valid]]
    return {"map50": float(precision[0].mean()), "map5095": float(precision.mean())}


# ----------------------------------------------------------------------------- per-image summaries

def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney, ties counted 1/2)."""
    from scipy.stats import rankdata

    pos, neg = int(labels.sum()), int((~labels).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


def prf(counts: dict[str, tuple[int, int, int]], names: list[str]) -> tuple[float, float, float]:
    tp = sum(counts[n][0] for n in names)
    fp = sum(counts[n][1] for n in names)
    fn = sum(counts[n][2] for n in names)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


class Scored:
    """Detections of one strategy and repeat on the 261 images, with what the bootstrap needs per image.
    P/R/F1 at conf (fixed a priori) and at conf_vt (chosen on inner validation for this strategy)."""

    def __init__(self, preds: ev.Preds, gt: ev.GT, names: list[str], conf: float, iou: float, conf_vt: float | None = None):
        self.preds = preds
        self.coco = coco_per_image(preds, gt, names)
        self.conf_vt = conf if conf_vt is None else conf_vt
        self.counts, self.counts_vt, self.img_score = {}, {}, {}
        for n in names:
            p = preds.get(n, ev.EMPTY)
            self.counts[n] = ev.match_image(p["boxes"], p["scores"], gt[n]["boxes"], conf, iou)
            self.counts_vt[n] = ev.match_image(p["boxes"], p["scores"], gt[n]["boxes"], self.conf_vt, iou)
            self.img_score[n] = float(p["scores"].max()) if len(p["scores"]) else 0.0
        self.positive = {n: len(gt[n]["boxes"]) > 0 for n in names}
        self.conf = conf

    def metrics(self, names: list[str]) -> dict[str, float]:
        m = coco_from_per_image(self.coco, names)
        p, r, f1 = prf(self.counts, names)
        pv, rv, f1v = prf(self.counts_vt, names)
        s = np.array([self.img_score[n] for n in names])
        y = np.array([self.positive[n] for n in names])
        flagged = s >= self.conf
        m.update({
            "precision": p, "recall": r, "f1": f1,
            "precision_vt": pv, "recall_vt": rv, "f1_vt": f1v,
            "auroc": auroc(s, y),
            "sensitivity": float(flagged[y].mean()) if y.any() else float("nan"),
            "specificity": float((~flagged[~y]).mean()) if (~y).any() else float("nan"),
        })
        return m


# ----------------------------------------------------------------------------- video groups

def numeric_id(name: str) -> int:
    return int(re.findall(r"\d+", name)[-1])


def reconstruct_groups(splits: dict) -> dict[str, str]:
    """Blocks of consecutive image IDs that share the outer test fold in every repeat of a grouped split.
    Each real group (contiguous IDs, nested_cv --groups-csv) lies inside one block; a block may merge
    neighbouring groups, which only makes the cluster bootstrap more conservative."""
    names = sorted({n for f in splits["0"] for n in f["test"]}, key=numeric_id)
    sig = {n: tuple(next(k for k, f in enumerate(splits[r]) if n in f["test"]) for r in sorted(splits)) for n in names}
    groups, g = {}, 0
    for i, n in enumerate(names):
        if i and sig[n] != sig[names[i - 1]]:
            g += 1
        groups[n] = f"block{g:02d}"
    return groups


def video_groups(proto: dict, splits: dict) -> tuple[dict[str, str], str]:
    path = proto["nested"].get("groups_csv")
    if path:
        with open(resolve(path), encoding="utf-8") as f:
            return {row["name"]: str(row["group"]) for row in csv.DictReader(f)}, str(resolve(path))
    return reconstruct_groups(splits), "reconstructed from splits.json (contiguous blocks with identical test folds)"


def cluster_samples(groups: dict[str, str], names: list[str], positive: dict[str, bool], iters: int, seed: int) -> tuple[list[list[str]], int]:
    """Resample whole groups with replacement; replicates without positives or negatives are redrawn."""
    members = defaultdict(list)
    for n in names:
        members[groups[n]].append(n)
    keys = sorted(members)
    rng = np.random.default_rng(seed)
    out, redraws = [], 0
    while len(out) < iters:
        sample = [n for g in rng.integers(0, len(keys), size=len(keys)) for n in members[keys[g]]]
        flags = [positive[n] for n in sample]
        if all(flags) or not any(flags):
            redraws += 1
            continue
        out.append(sample)
    return out, redraws


# ----------------------------------------------------------------------------- validation sweep

def val_rows(preds: ev.Preds, gt: ev.GT, names: list[str], iou: float) -> list[dict]:
    coco = ev.coco_map(preds, gt, names)
    rows = []
    for conf in CONF_GRID:
        fp = ev.fixed_point_metrics(preds, gt, names, conf, iou)
        rows.append({"conf": conf, **coco, "precision": fp["precision"], "recall": fp["recall"], "f1": fp["f1"]})
    return rows


def sweep(recs: dict, gt: ev.GT, proto: dict, tag: str, log) -> tuple[list[dict], list[dict]]:
    nc, inf, evc = proto["nested"], proto["inference"], proto["evaluation"]
    tta_rows, wbf_rows = [], []
    sizes = nc["sizes"]
    for r in range(nc["repeats"]):
        for k in range(nc["folds"]):
            names = sorted(recs[(tag, sizes[0], r, k)]["val"])
            for s in sizes[1:]:
                if sorted(recs[(tag, s, r, k)]["val"]) != names:
                    raise ValueError(f"inner validation differs between sizes at r{r} f{k}")
            if not any(len(gt[n]["boxes"]) for n in names):
                log.warning(f"r{r} f{k}: validation without boxes, skipped")
                continue
            sub = {n: gt[n] for n in names}
            for s in sizes:
                views = recs[(tag, s, r, k)]["val"]
                base = {"model": MODEL[s], "seed": r, "fold": k}
                for row in val_rows({n: det_standard(views[n], sub[n]["width"], sub[n]["height"]) for n in names}, sub, names, evc["iou_match"]):
                    tta_rows.append({"group": "tta", "config": "standard", "wbf_iou": None, "weights": "", **base, **row})
                for name, cfg in available_tta(views[names[0]]).items():
                    for iou in WBF_IOU_GRID:
                        preds = {n: det_tta(views[n], cfg, iou, inf["tta"]["wbf_skip"], sub[n]["width"], sub[n]["height"]) for n in names}
                        for row in val_rows(preds, sub, names, evc["iou_match"]):
                            tta_rows.append({"group": "tta", "config": name, "wbf_iou": iou, "weights": "", **base, **row})
            for combo in [c for n_m in (2, 3) for c in itertools.combinations(sizes, n_m)]:
                for wname, mode in WBF_WEIGHT_MODES.items():
                    weights = [1.0] * len(combo) if mode is None else [float(i + 1) for i in range(len(combo))]
                    for iou in WBF_IOU_GRID:
                        preds = {n: det_wbf([recs[(tag, s, r, k)]["val"][n] for s in combo], weights, iou, inf["wbf_models"]["skip"], sub[n]["width"], sub[n]["height"]) for n in names}
                        label = "+".join(MODEL[s] for s in combo)
                        for row in val_rows(preds, sub, names, evc["iou_match"]):
                            wbf_rows.append({"group": "wbf", "config": f"{label}_{wname}", "wbf_iou": iou, "weights": "" if mode is None else ",".join(f"{w:g}" for w in weights), "model": label, "seed": r, "fold": k, **row})
            log.info(f"  validation sweep r{r} f{k}: {len(names)} images")
    return tta_rows, wbf_rows


def val_thresholds(recs: dict, gt: ev.GT, proto: dict, choice: dict) -> tuple[dict[tuple, float], list[dict]]:
    """Operating threshold per (tag, model, strategy): the CONF_GRID value with the highest mean inner-validation F1
    over the 15 folds (ties -> closest to conf_fixed). Reported next to the a-priori conf_fixed, never instead of it."""
    nc, inf, evc = proto["nested"], proto["inference"], proto["evaluation"]
    tta_cfg, tta_iou = TTA_GRID[choice["tta"]["config"]], choice["tta"]["wbf_iou"]
    wbf = choice.get("wbf")
    f1s = defaultdict(lambda: defaultdict(list))
    for r in range(nc["repeats"]):
        for k in range(nc["folds"]):
            names = sorted(recs[(nc["tags"][0], nc["sizes"][0], r, k)]["val"])
            if not any(len(gt[n]["boxes"]) for n in names):
                continue
            sub = {n: gt[n] for n in names}
            cands = {}
            for tag in nc["tags"]:
                for s in nc["sizes"]:
                    views = recs[(tag, s, r, k)]["val"]
                    cands[(tag, MODEL[s], "standard")] = {n: det_standard(views[n], sub[n]["width"], sub[n]["height"]) for n in names}
                    cands[(tag, MODEL[s], "tta")] = {n: det_tta(views[n], tta_cfg, tta_iou, inf["tta"]["wbf_skip"], sub[n]["width"], sub[n]["height"]) for n in names}
            if wbf:
                members = [SIZE_OF[m] for m in wbf["models"]]
                weights = wbf["weights"] or [1.0] * len(members)
                cands[(nc["wbf_tag"], "+".join(wbf["models"]), "wbf")] = {n: det_wbf([recs[(nc["wbf_tag"], s, r, k)]["val"][n] for s in members], weights, wbf["iou"], inf["wbf_models"]["skip"], sub[n]["width"], sub[n]["height"]) for n in names}
            for key, preds in cands.items():
                for conf in CONF_GRID:
                    f1s[key][conf].append(ev.fixed_point_metrics(preds, sub, names, conf, evc["iou_match"])["f1"])
    chosen, rows = {}, []
    for key, per_conf in f1s.items():
        means = {c: float(np.mean(v)) for c, v in per_conf.items()}
        chosen[key] = max(means, key=lambda c: (round(means[c], 12), -abs(c - evc["conf_fixed"])))
        rows += [{"tag": key[0], "model": key[1], "strategy": key[2], "conf": c, "val_f1_mean": round(means[c], 4), "n_folds": len(per_conf[c]), "chosen": c == chosen[key]} for c in sorted(means)]
    return chosen, rows


# ----------------------------------------------------------------------------- out-of-fold test

def oof_predictions(recs: dict, gt: ev.GT, proto: dict, choice: dict, names: list[str]) -> dict[tuple, dict[int, ev.Preds]]:
    """(tag, model, strategy) -> repeat -> detections on every image, each from the fold model that had it in test."""
    nc, inf = proto["nested"], proto["inference"]
    tta_cfg, tta_iou = TTA_GRID[choice["tta"]["config"]], choice["tta"]["wbf_iou"]
    wbf = choice.get("wbf")
    out = defaultdict(dict)
    for r in range(nc["repeats"]):
        for tag in nc["tags"]:
            for s in nc["sizes"]:
                std, tta = {}, {}
                for k in range(nc["folds"]):
                    for n, views in recs[(tag, s, r, k)]["test"].items():
                        w, h = gt[n]["width"], gt[n]["height"]
                        std[n] = det_standard(views, w, h)
                        tta[n] = det_tta(views, tta_cfg, tta_iou, inf["tta"]["wbf_skip"], w, h)
                out[(tag, MODEL[s], "standard")][r] = std
                out[(tag, MODEL[s], "tta")][r] = tta
        if wbf:
            tag = nc["wbf_tag"]
            members = [SIZE_OF[m] for m in wbf["models"]]
            weights = wbf["weights"] or [1.0] * len(members)
            preds = {}
            for k in range(nc["folds"]):
                for n in recs[(tag, members[0], r, k)]["test"]:
                    preds[n] = det_wbf([recs[(tag, s, r, k)]["test"][n] for s in members], weights, wbf["iou"], inf["wbf_models"]["skip"], gt[n]["width"], gt[n]["height"])
            out[(tag, "+".join(wbf["models"]), "wbf")][r] = preds
    for key, per_rep in out.items():
        for r, preds in per_rep.items():
            if sorted(preds) != names:
                raise ValueError(f"{key} repeat {r}: out-of-fold predictions cover {len(preds)} of {len(names)} images")
    return out


def comparisons(keys: list[tuple], proto: dict) -> list[tuple[tuple, tuple, str]]:
    """(a, b, label) pairs reported as b - a."""
    nc = proto["nested"]
    pairs = []
    for tag in nc["tags"]:
        for s in nc["sizes"]:
            pairs.append(((tag, MODEL[s], "standard"), (tag, MODEL[s], "tta"), f"{tag} {MODEL[s]}: tta - standard"))
    for key in keys:
        if key[2] == "wbf":
            for m in key[1].split("+"):
                pairs.append(((key[0], m, "standard"), key, f"{key[0]} {key[1]}: wbf - {m}"))
    tag0 = nc["tags"][0]
    for a, b in (("n", "m"), ("m", "l")):
        pairs.append(((tag0, MODEL[a], "standard"), (tag0, MODEL[b], "standard"), f"{tag0}: {MODEL[b]} - {MODEL[a]} (standard)"))
    if len(nc["tags"]) > 1:
        for s in nc["sizes"]:
            pairs.append(((nc["tags"][0], MODEL[s], "standard"), (nc["tags"][1], MODEL[s], "standard"), f"{MODEL[s]}: {nc['tags'][1]} - {nc['tags'][0]} (standard)"))
    return [p for p in pairs if p[0] in keys and p[1] in keys]


def write_oof_dumps(root: Path, oof: dict, gt: ev.GT, choice: dict, variant: str) -> None:
    for (tag, model, strategy), per_rep in oof.items():
        for r, preds in per_rep.items():
            d = root / f"{tag}_{model}_{strategy}_r{r}"
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
            for n, p in preds.items():
                write_json(d / f"{Path(n).stem}.json", {"image": n, "width": gt[n]["width"], "height": gt[n]["height"], "boxes": np.round(p["boxes"], 2).tolist(), "scores": np.round(p["scores"], 5).tolist(), "classes": p["classes"].tolist()})
            write_json(d / "_meta.json", {"variant": variant, "tag": tag, "model": model, "strategy": strategy, "repeat": r, "source": "out-of-fold predictions rebuilt from revision_nested_cv views", "choice": choice, "n_images": len(preds)})


def evaluate_variant(proto: dict, variant: str, groups: dict[str, str], log, dumps: bool) -> dict:
    nc, evc = proto["nested"], proto["evaluation"]
    suffix = nc["grouped_suffix"] if variant == "grouped" else nc["ungrouped_suffix"]
    out = resolve(RESULTS) / f"nested_{variant}"
    recs = load_records(proto, suffix)
    names = sorted({n for rec in recs.values() for n in rec["test"]})
    sizes = image_sizes(proto, names)
    gt = gt_from_records(recs, sizes)
    log.info(f"[{variant}] {len(recs)} fold models, {len(names)} images, {sum(len(g['boxes']) for g in gt.values())} boxes")

    tag = nc["wbf_tag"]
    tta_rows, wbf_rows = sweep(recs, gt, proto, tag, log)
    write_csv(out / "sweep_tta_val.csv", tta_rows)
    write_csv(out / "sweep_wbf_val.csv", wbf_rows)
    tta_summary, wbf_summary = summarize(tta_rows), summarize(wbf_rows)
    write_csv(out / "sweep_tta_val_summary.csv", tta_summary)
    write_csv(out / "sweep_wbf_val_summary.csv", wbf_summary)
    choice = {
        "tag": tag, "variant": variant, "conf": evc["conf_fixed"],
        "rule": "TTA: highest validation mAP@50 averaged over models; WBF: highest mean validation mAP@50",
        "validation": "inner validation split of every outer fold of nested_cv.py (mean over 3 repeats x 5 folds)",
        "tta": choose_tta(tta_summary, evc["conf_fixed"]), "wbf": choose_wbf(wbf_summary, evc["conf_fixed"]),
    }
    write_json(out / "inference_choice.json", choice)
    log.info(f"[{variant}] choice: tta={choice['tta']} wbf={choice['wbf']}")

    thresholds, thr_rows = val_thresholds(recs, gt, proto, choice)
    write_csv(out / "val_thresholds.csv", thr_rows)
    choice["operating_threshold_on_validation"] = {"/".join(k): v for k, v in sorted(thresholds.items())}
    write_json(out / "inference_choice.json", choice)

    oof = oof_predictions(recs, gt, proto, choice, names)
    scored = {key: {r: Scored(p, gt, names, evc["conf_fixed"], evc["iou_match"], thresholds[key]) for r, p in per_rep.items()} for key, per_rep in oof.items()}
    full = {key: {r: s.metrics(names) for r, s in per_rep.items()} for key, per_rep in scored.items()}
    for key, per_rep in full.items():  # the fast accumulate must agree with pycocotools on the full set
        ref = ev.coco_map(oof[key][0], gt, names)
        if abs(ref["map50"] - per_rep[0]["map50"]) > 1e-9 or abs(ref["map5095"] - per_rep[0]["map5095"]) > 1e-9:
            raise AssertionError(f"{key}: fast COCO {per_rep[0]['map50']:.6f}/{per_rep[0]['map5095']:.6f} != pycocotools {ref['map50']:.6f}/{ref['map5095']:.6f}")
    metric_rows = [{"variant": variant, "tag": t, "model": m, "strategy": s, "repeat": r, **{k: round(v, 6) for k, v in mm.items()}} for (t, m, s), per_rep in full.items() for r, mm in per_rep.items()]
    write_csv(out / "oof_metrics.csv", metric_rows)

    fold_rows = []
    for (t, m, s), per_rep in scored.items():
        if s == "wbf":
            continue
        for r, sc in per_rep.items():
            for k in range(nc["folds"]):
                fold_names = sorted(recs[(t, SIZE_OF[m], r, k)]["test"])
                fm = sc.metrics(fold_names)
                fold_rows.append({"variant": variant, "tag": t, "model": m, "strategy": s, "repeat": r, "fold": k, "n_images": len(fold_names), "n_boxes": int(sum(len(gt[n]["boxes"]) for n in fold_names)), **{kk: round(v, 6) for kk, v in fm.items()}})
    write_csv(out / "per_fold.csv", fold_rows)

    positive = {n: len(gt[n]["boxes"]) > 0 for n in names}
    samples, redraws = cluster_samples(groups, names, positive, evc["bootstrap_iters"], evc["bootstrap_seed"])
    boot = {}
    for key, per_rep in scored.items():
        reps = sorted(per_rep)
        arr = np.full((len(samples), len(METRICS)), np.nan)
        for i, smp in enumerate(samples):
            vals = [per_rep[r].metrics(smp) for r in reps]
            arr[i] = [np.mean([v[m] for v in vals]) for m in METRICS]
        boot[key] = arr
        log.info(f"[{variant}] bootstrap {key} done")
    summary_rows = []
    for key, per_rep in full.items():
        row = {"variant": variant, "tag": key[0], "model": key[1], "strategy": key[2], "n_repeats": len(per_rep), "conf_fixed": evc["conf_fixed"], "conf_vt": thresholds[key]}
        for j, m in enumerate(METRICS):
            v = [per_rep[r][m] for r in sorted(per_rep)]
            row[f"{m}_mean"] = round(float(np.mean(v)), 4)
            row[f"{m}_std"] = round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else float("nan")
            row[f"{m}_ci_lo"] = round(float(np.nanpercentile(boot[key][:, j], 2.5)), 4)
            row[f"{m}_ci_hi"] = round(float(np.nanpercentile(boot[key][:, j], 97.5)), 4)
        summary_rows.append(row)
    write_csv(out / "summary.csv", summary_rows)
    delta_rows = []
    for a, b, label in comparisons(list(full), proto):
        row = {"variant": variant, "comparison": label, "a": "/".join(a), "b": "/".join(b)}
        for j, m in enumerate(METRICS):
            point = float(np.mean([full[b][r][m] - full[a][r][m] for r in sorted(full[a])]))
            d = boot[b][:, j] - boot[a][:, j]
            row.update({f"{m}_delta": round(point, 4), f"{m}_ci_lo": round(float(np.nanpercentile(d, 2.5)), 4), f"{m}_ci_hi": round(float(np.nanpercentile(d, 97.5)), 4), f"{m}_p_le0": round(float(np.mean(d[~np.isnan(d)] <= 0)), 4)})
        delta_rows.append(row)
    write_csv(out / "deltas.csv", delta_rows)
    write_json(out / "bootstrap.json", {"iters": len(samples), "seed": evc["bootstrap_seed"], "unit": "video group", "n_groups": len(set(groups[n] for n in names)), "redraws_without_both_classes": redraws, "statistic": "metric of each repeat on the resampled images, averaged over repeats"})
    if dumps:
        write_oof_dumps(resolve("r1/runs/nested") / variant / "oof", oof, gt, choice, variant)
    return {"choice": choice, "summary": summary_rows}


# ----------------------------------------------------------------------------- commands

def cmd_prepare(proto: dict, log) -> None:
    paths = dataset_images(proto)
    if len(paths) != 261:
        raise SystemExit(f"expected 261 images under {resolve(proto['dataset']['root'])}/images/*, found {len(paths)}")
    entries, n_boxes, buckets = [], 0, defaultdict(int)
    for n, p in paths.items():
        with Image.open(p) as im:
            w, h = im.size
        b = read_labels(p, w, h)
        lab = label_path(p)
        entries.append({"image": n, "split_dir": p.parent.name, "positive": bool(len(b)), "n_boxes": int(len(b)), "width": w, "height": h, "image_sha256": sha256_file(p), "label_sha256": sha256_file(lab) if lab.exists() else None})
        n_boxes += len(b)
        for a in bx.box_area(b) if len(b) else []:
            buckets[bx.size_bucket(float(a))] += 1
    mismatches = []
    for suffix in (proto["nested"]["grouped_suffix"], proto["nested"]["ungrouped_suffix"]):
        for p in sorted((resolve(proto["nested"]["out"]) / "preds").glob(f"*{suffix}")):
            if suffix == ".pkl" and p.name.endswith(".bak"):
                continue
            with open(p, "rb") as f:
                rec = pickle.load(f)
            for n, g in rec["gt"].items():
                e = next(x for x in entries if x["image"] == n)
                ref = read_labels(paths[n], e["width"], e["height"]) / [e["width"], e["height"], e["width"], e["height"]]
                if ref.shape != np.asarray(g).reshape(-1, 4).shape or not np.allclose(ref, np.asarray(g).reshape(-1, 4), atol=1e-5):
                    mismatches.append(f"{p.name}:{n}")
    if mismatches:
        raise SystemExit(f"ground truth in the .pkl files differs from the labels: {mismatches[:10]}")
    manifest = {
        "dataset": proto["dataset"]["name"], "root": str(resolve(proto["dataset"]["root"])),
        "summary": {"images": len(entries), "positive": sum(e["positive"] for e in entries), "negative": sum(not e["positive"] for e in entries), "boxes": n_boxes, "size_buckets": dict(buckets)},
        "images": entries,
    }
    write_json(resolve(proto["dataset"]["manifest_file"]), manifest)
    log.info(f"manifest written: {manifest['summary']} | ground truth of every .pkl equals the labels")


def checkpoint_path(proto: dict, tag: str, size: str, r: int, k: int) -> Path:
    return resolve(proto["nested"]["out"]) / "runs" / f"{tag}_r{r}_f{k}_{size}" / "weights" / "best.pt"


def cmd_check(proto: dict, log, skip_checkpoints: bool) -> None:
    from r1.audit_checkpoints import audit

    nc = proto["nested"]
    problems = []
    splits = read_json(resolve(nc["out"]) / "splits.json")
    groups, source = video_groups(proto, splits)
    all_names = sorted(dataset_images(proto))
    for r, folds in splits.items():
        tested = []
        for k, f in enumerate(folds):
            tr, va, te = set(f["train"]), set(f["val"]), set(f["test"])
            if tr & va or tr & te or va & te:
                problems.append(f"r{r} f{k}: overlap between train/val/test")
            if sorted(tr | va | te) != all_names:
                problems.append(f"r{r} f{k}: train+val+test is not the 261 images")
            if {groups[n] for n in te} & {groups[n] for n in tr | va}:
                problems.append(f"r{r} f{k}: a video group is split between test and train/val")
            tested += f["test"]
        if sorted(tested) != all_names:
            problems.append(f"r{r}: not every image tested exactly once")
    recs = load_records(proto, nc["grouped_suffix"])
    args_hash = set()
    for (tag, size, r, k), rec in recs.items():
        f = splits[str(r)][k]
        fp = rec["meta"].get("split_fingerprint")
        expected = {part: tuple(sorted(f[part])) for part in ("train", "val", "test")}
        if fp != expected:
            problems.append(f"{tag} {size} r{r} f{k}: split fingerprint differs from splits.json")
        if set(rec["test"]) != set(f["test"]) or set(rec["val"]) != set(f["val"]):
            problems.append(f"{tag} {size} r{r} f{k}: stored predictions are not the fold's val/test images")
        ta = rec["meta"]["train_args"]
        if ta.get("seed") != 42 + r or f"workspaces/r{r}_f{k}/" not in str(ta.get("data")).replace("\\", "/"):
            problems.append(f"{tag} {size} r{r} f{k}: unexpected seed or data yaml {ta.get('seed')} {ta.get('data')}")
        args_hash.add(hashlib.sha256(repr(sorted((a, str(b)) for a, b in ta.items() if a not in ("data", "name", "seed"))).encode()).hexdigest())
        if not skip_checkpoints:
            pt = checkpoint_path(proto, tag, size, r, k)
            if not pt.exists():
                problems.append(f"missing checkpoint {pt}")
            else:
                a = audit(pt)
                if not a["is_yolo26"]:
                    problems.append(f"{pt}: not YOLO26 ({a['families']} {a['model_names']})")
                if not any(f"workspaces/r{r}_f{k}/" in d.replace("\\", "/") for d in a["data_yaml"]):
                    problems.append(f"{pt}: trained on {a['data_yaml']}, not on workspace r{r}_f{k}")
    if len(args_hash) != 1:
        problems.append(f"{len(args_hash)} different training configurations among the fold models")
    bak = load_records(proto, nc["ungrouped_suffix"])
    for (tag, size, r, k), rec in bak.items():
        if set(rec["test"]) & set(rec["val"]):
            problems.append(f"ungrouped {tag} {size} r{r} f{k}: val and test overlap")
    for tag in nc["tags"]:
        for size in nc["sizes"]:
            for r in range(nc["repeats"]):
                tested = sorted(n for k in range(nc["folds"]) for n in bak[(tag, size, r, k)]["test"])
                if tested != all_names:
                    problems.append(f"ungrouped {tag} {size} r{r}: not every image tested exactly once")
    n_groups = len(set(groups.values()))
    if problems:
        for p in problems[:30]:
            log.error(p)
        raise SystemExit(f"check failed: {len(problems)} problems")
    log.info(f"check OK: {len(splits)} repeats x {len(splits['0'])} folds, {len(recs)} grouped + {len(bak)} ungrouped fold records, {n_groups} video groups ({source}), checkpoints {'skipped' if skip_checkpoints else 'YOLO26, own workspace'}")


def cmd_evaluate(proto: dict, log, variants: list[str], dumps: bool) -> None:
    splits = read_json(resolve(proto["nested"]["out"]) / "splits.json")
    groups, source = video_groups(proto, splits)
    res = resolve(RESULTS)
    write_csv(res / "nested_groups.csv", [{"name": n, "group": g} for n, g in sorted(groups.items(), key=lambda x: numeric_id(x[0]))])
    results = {v: evaluate_variant(proto, v, groups, log, dumps and v == "grouped") for v in variants}
    if (res / "nested_grouped" / "inference_choice.json").exists():  # the primary variant drives every later step
        shutil.copy(res / "nested_grouped" / "inference_choice.json", res / "inference_choice.json")
        shutil.copy(res / "nested_grouped" / "sweep_tta_val_summary.csv", res / "sweep_tta_val_summary.csv")
    rows, deltas = [], []
    for v in ("grouped", "ungrouped"):  # every variant evaluated so far, also by earlier calls
        if (res / f"nested_{v}" / "summary.csv").exists():
            with open(res / f"nested_{v}" / "summary.csv", encoding="utf-8") as f:
                rows += list(csv.DictReader(f))
            with open(res / f"nested_{v}" / "deltas.csv", encoding="utf-8") as f:
                deltas += list(csv.DictReader(f))
    write_csv(res / "nested_summary.csv", rows)
    write_csv(res / "nested_deltas.csv", deltas)
    log.info(f"written {res / 'nested_summary.csv'}, nested_deltas.csv and inference_choice.json (groups: {source})")


def place_checkpoint(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src.resolve(), dst)
        return "symlink"
    except OSError:
        os.link(src.resolve(), dst)  # Windows without symlink privilege: hard link on the same volume
        return "hardlink"


def cmd_link(proto: dict, log, tags: list[str], sizes: list[str]) -> None:
    nc = proto["nested"]
    splits = read_json(resolve(nc["out"]) / "splits.json")
    root = resolve("r1/runs")
    kinds = set()
    for tag in tags:
        for size in sizes:
            for r in range(nc["repeats"]):
                for k in range(nc["folds"]):
                    src = checkpoint_path(proto, tag, size, r, k)
                    if not src.exists():
                        raise SystemExit(f"missing checkpoint {src}")
                    d = root / tag / f"{MODEL[size]}_seed{r}" / f"fold{k}"
                    kinds.add(place_checkpoint(src, d / "weights" / "best.pt"))
                    f = splits[str(r)][k]
                    write_json(d / "run_info.json", {"tag": tag, "model": MODEL[size], "seed": r, "repeat": r, "fold": k, "dry_run": False, "checkpoint": str(src), "source_pkl": str(pkl_path(proto, tag, size, r, k, nc["grouped_suffix"])), "train_images": f["train"], "val_images": f["val"], "test_images": f["test"]})
                    (d / "DONE").write_text("linked from revision_nested_cv\n", encoding="utf-8")
    log.info(f"linked {len(tags) * len(sizes) * nc['repeats'] * nc['folds']} checkpoints under {root} ({', '.join(sorted(kinds))})")


def cmd_sam3_map(proto: dict, log) -> None:
    nc = proto["nested"]
    r, size, tag = nc["sam3_repeat"], nc["sam3_size"], nc["wbf_tag"]
    splits = read_json(resolve(nc["out"]) / "splits.json")
    fold_of = {n: k for k, f in enumerate(splits[str(r)]) for n in f["test"]}
    manifest = read_json(resolve(proto["sam3"]["manifest_file"]))
    paths = dataset_images(proto)
    images, pixel_diff = {}, {}
    for e in manifest["images"]:
        n = e["image"]
        k = fold_of.get(n, 0)
        ckpt = resolve("r1/runs") / tag / f"{MODEL[size]}_seed{r}" / f"fold{k}" / "weights" / "best.pt"
        if not ckpt.exists():
            raise SystemExit(f"{ckpt} missing: run python -m r1.nested_eval link first")
        images[n] = str(ckpt.relative_to(resolve(".")).as_posix())
        if n in fold_of:  # same LIACI frame in both sets: mean absolute pixel difference of the two 640x640 copies
            a = np.asarray(Image.open(resolve(proto["sam3"]["output_root"]) / "images" / n).convert("RGB"), dtype=float)
            b = np.asarray(Image.open(paths[n]).convert("RGB"), dtype=float)
            pixel_diff[n] = round(float(np.abs(a - b).mean()), 2) if a.shape == b.shape else None
    in_nested = sorted(n for n in images if n in fold_of)
    out = resolve(RESULTS) / f"sam3_map_r{r}_{size}.json"
    write_json(out, {"tag": tag, "model": MODEL[size], "repeat": r, "rule": "image of the nested CV -> fold model with the image in its outer test fold; other images -> fold 0", "n_images": len(images), "n_in_nested_cv": len(in_nested), "in_nested_cv": {n: {"fold": fold_of[n], "mean_abs_pixel_diff_vs_dataset_yolo": pixel_diff[n]} for n in in_nested}, "images": images})
    diffs = [d for d in pixel_diff.values() if d is not None]
    log.info(f"written {out}: {len(images)} images, {len(in_nested)} of them in the nested CV (mean abs pixel difference vs dataset_yolo: max {max(diffs) if diffs else 'n/a'})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    c = sub.add_parser("check")
    c.add_argument("--skip-checkpoints", action="store_true", help="do not open the best.pt files")
    e = sub.add_parser("evaluate")
    e.add_argument("--variants", default="grouped,ungrouped")
    e.add_argument("--no-dumps", action="store_true", help="do not write the out-of-fold dumps")
    lk = sub.add_parser("link")
    lk.add_argument("--tags", default=None, help="default: protocol nested.tags")
    lk.add_argument("--sizes", default=None, help="default: protocol nested.sizes")
    sub.add_parser("sam3-map")
    args = ap.parse_args()
    proto = load_protocol()
    log = get_logger(f"nested_{args.cmd}", resolve(f"{RESULTS}/logs/nested_{args.cmd}.log"))
    if args.cmd == "prepare":
        cmd_prepare(proto, log)
    elif args.cmd == "check":
        cmd_check(proto, log, args.skip_checkpoints)
    elif args.cmd == "evaluate":
        cmd_evaluate(proto, log, args.variants.split(","), not args.no_dumps)
    elif args.cmd == "link":
        cmd_link(proto, log, args.tags.split(",") if args.tags else proto["nested"]["tags"], args.sizes.split(",") if args.sizes else proto["nested"]["sizes"])
    else:
        cmd_sam3_map(proto, log)


if __name__ == "__main__":
    main()
