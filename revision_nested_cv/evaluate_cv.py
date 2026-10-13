#!/usr/bin/env python3
"""
evaluate_cv.py -- CPU-only analysis of the predictions cached by nested_cv.py.

For every strategy (single pass, several TTA variants, Nano+Medium / Medium+Large WBF
ensembles) it reports, on out-of-fold predictions of all images:
  * per-fold and per-repeat Precision / Recall / F1 / AP50 at the fixed threshold 0.25,
  * mean +- std over repeats,
  * paired bootstrap 95% CIs (resampling images) for the key differences,
in three threshold modes:
  pre       filter each view/model at conf>=0.25, then fuse, no re-filter
            (the protocol of notebooks 06/07 of the released code; AP is truncated)
  post      fuse all low-confidence detections, then threshold the output at 0.25
            (same operating rule for single pass, TTA and ensembles; AP is the full curve)
  post_val  as 'post', but the threshold is chosen on the inner-validation images (max F1)

Usage:
  python evaluate_cv.py --out nested_out [--seq-tag sequential] [--n-boot 1000] [--jobs 4]
"""
import argparse
import itertools
import multiprocessing as mp
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import fuse, tta_views  # noqa: E402
from metrics_coco import compute_ap_coco, compute_iou_matrix, compute_metrics_coco  # noqa: E402

THR = 0.25
GRID = np.round(np.arange(0.05, 0.95, 0.05), 2)
G = {}  # globals shared with worker processes (fork)


def load(out, tag):
    P = {}
    for f in sorted((Path(out) / "preds").glob(f"{tag}_r*_f*_*.pkl")):
        rec = pickle.load(open(f, "rb"))
        m = rec["meta"]
        P[(m["repeat"], m["fold"], m["size"])] = rec
    return P


def strategies(sizes, scales, fine_scales=None, rot_angles=None):
    S = {}
    for s in sizes:
        S[f"single_{s}"] = ([(s, "std")], [1.0])
        v = tta_views("paper", scales, True)
        S[f"tta_paper_{s}"] = ([(s, k) for k, _ in v], [w for _, w in v])
        v = tta_views("imgsz", scales, True)
        S[f"tta_imgsz_{s}"] = ([(s, k) for k, _ in v], [w for _, w in v])
        S[f"tta_flip_{s}"] = ([(s, "std"), (s, "flip")], [1.0, 1.0])
        v = tta_views("imgsz", scales, False)
        S[f"tta_scale_{s}"] = ([(s, k) for k, _ in v], [w for _, w in v])
        if fine_scales:
            v = tta_views("imgsz", fine_scales, True)
            S[f"tta_finescale_{s}"] = ([(s, k) for k, _ in v], [w for _, w in v])
        S[f"tta_vflip_{s}"] = ([(s, "std"), (s, "vflip")], [1.0, 0.8])
        if rot_angles:
            keys = ["std", "flip"]
            for a in rot_angles:
                for signed in (a, -a):
                    for flip in (0, 1):
                        keys.append(f"rot|{signed:g}|{flip}")
            S[f"tta_rot_{s}"] = ([(s, k) for k in keys], [1.0 if k in ("std", "flip") else 0.6 for k in keys])
            if fine_scales:
                v = tta_views("imgsz", fine_scales, True)
                keys2 = [k for k, _ in v] + keys[2:]  # finescale x flip + the rotation views
                w2 = [w for _, w in v] + [0.6] * len(keys[2:])
                S[f"tta_kitchen_{s}"] = ([(s, k) for k in keys2], w2)
    if "n" in sizes and "m" in sizes:
        for tag, w in [("11", [1, 1]), ("21", [2, 1]), ("12", [1, 2])]:
            S[f"ens_nm_{tag}"] = ([("n", "std"), ("m", "std")], w)
    if "m" in sizes and "l" in sizes:
        S["ens_ml_11"] = ([("m", "std"), ("l", "std")], [1, 1])
    S = {k: (mem, w, "avg") for k, (mem, w) in S.items()}
    # AP-ranking-only companions using conf_type='max' (does not change P/R/F1, only the AP50 curve
    # -- see README: 'avg'/'box_and_model_avg'/'absent_model_aware_avg' are mathematically identical
    # here because each TTA view contributes at most one box per cluster; 'max' is the only variant
    # that differs).
    for base in [f"tta_imgsz_{s}" for s in sizes if f"tta_imgsz_{s}" in S] + \
                [f"tta_paper_{s}" for s in sizes if f"tta_paper_{s}" in S] + \
                (["ens_ml_11"] if "ens_ml_11" in S else []):
        mem, w, _ = S[base]
        S[f"{base}_maxconf"] = (mem, w, "max")
    return S


def oof(P, r, strat, part, mode, folds):
    """
    Fused predictions for every image of `part` in repeat r (dict name -> (boxes, scores)).
    Returns (None, None) if any required view is missing for this repeat (e.g. a strategy that
    needs views from predict_extra_views.py before that script has been run) -- the caller then
    silently drops this (strategy, repeat) combination instead of crashing the whole batch.
    """
    members, weights, conf_type = strat
    out, gt = {}, {}
    for k in folds:
        recs = [P.get((r, k, s)) for s, _ in members]
        if any(x is None for x in recs):
            return None, None
        probe = recs[0][part]
        if probe:
            probe_name = next(iter(probe))
            if any(v not in rec[part][probe_name] for rec, (_, v) in zip(recs, members)):
                return None, None  # a required view isn't cached yet for this fold
        for n in probe:
            mem = [(rec[part][n][v][0], rec[part][n][v][1]) for rec, (_, v) in zip(recs, members)]
            out[n] = fuse(mem, weights, THR, "pre" if mode == "pre" else "post", conf_type=conf_type)
            gt[n] = recs[0]["gt"][n]
    return out, gt


def _weight_for_key(key):
    if key in ("std", "flip"):
        return 1.0
    if key.startswith("rot") or key == "vflip":
        return 0.6
    parts = key.split("|")
    try:
        return 1.0 if abs(float(parts[1]) - 1.0) < 1e-9 else 0.8
    except Exception:
        return 0.8


def greedy_select_view(rec, gt_by_name, names, thr, max_views=10):
    """
    Forward selection on ONE fold's validation images: start from the empty set and repeatedly
    add whichever cached view most improves F1 (mode 'pre': filter each selected view at `thr`,
    WBF-fuse the union, score without a second threshold -- same semantics as the other TTA
    strategies). Stops when no candidate improves F1, or after `max_views` views. Selection uses
    validation data ONLY; the chosen (keys, weights) are then applied, unchanged, to that fold's
    test images by the caller.
    """
    candidates = sorted(set.intersection(*[set(rec[n].keys()) for n in names])) if names else []
    chosen, best_f1 = [], -1.0

    def score_with(keys):
        w = [_weight_for_key(k) for k in keys]
        preds = {n: fuse([rec[n][k] for k in keys], w, thr, "pre") for n in names}
        return score_lists([preds[n] for n in names], [gt_by_name[n] for n in names], thr, "pre")["F1"]

    remaining = list(candidates)
    for _ in range(min(max_views, len(candidates))):
        best_gain, best_k = None, None
        for k in remaining:
            f1 = score_with(chosen + [k])
            if best_gain is None or f1 > best_gain:
                best_gain, best_k = f1, k
        if best_gain is None or best_gain <= best_f1 + 1e-6:
            break
        chosen.append(best_k)
        remaining.remove(best_k)
        best_f1 = best_gain
    if not chosen:  # fall back to the plain single view if selection improves nothing
        chosen = ["std"] if "std" in candidates else candidates[:1]
    return chosen, [_weight_for_key(k) for k in chosen], best_f1


def oof_selected(P, r, size, folds, thr, max_views):
    out, gt, chosen_log = {}, {}, {}
    for k in folds:
        rec = P.get((r, k, size))
        if rec is None:
            return None, None, None
        val_names = list(rec["val"].keys())
        keys, weights, val_f1 = greedy_select_view(rec["val"], rec["gt"], val_names, thr, max_views)
        chosen_log[k] = dict(keys=keys, weights=weights, val_f1=val_f1)
        for n in rec["test"]:
            out[n] = fuse([rec["test"][n][kk] for kk in keys], weights, thr, "pre")
            gt[n] = rec["gt"][n]
    return out, gt, chosen_log


def score_lists(preds, gts, thr, mode):
    """preds/gts: lists aligned per image. mode 'pre' -> preds already thresholded."""
    pl = [{"boxes": b, "scores": s} for b, s in preds]
    gl = [{"boxes": g} for g in gts]
    ap, p, r, f1 = compute_metrics_coco(pl, gl)
    if mode == "pre":
        return dict(P=p, R=r, F1=f1, AP50=ap)
    sub = [{"boxes": b[s >= thr], "scores": s[s >= thr]} for b, s in preds]
    _, p, r, f1 = compute_metrics_coco(sub, gl)
    return dict(P=p, R=r, F1=f1, AP50=ap)


def per_image_match(boxes, scores, gt_boxes, iou_thr=0.5):
    """
    Per-image greedy match (same rule as metrics_coco.compute_metrics_coco's inner loop, copied
    verbatim so results are identical): returns (scores, is_tp) for this image's predictions,
    computed once regardless of which OTHER images later end up in a bootstrap resample.
    """
    boxes = np.asarray(boxes).reshape(-1, 4)
    scores = np.asarray(scores).reshape(-1)
    gt_boxes = np.asarray(gt_boxes).reshape(-1, 4)
    if len(boxes) == 0:
        return np.zeros(0), np.zeros(0, dtype=bool)
    if len(gt_boxes) == 0:
        return scores, np.zeros(len(scores), dtype=bool)
    iou = compute_iou_matrix(boxes, gt_boxes)
    order = np.argsort(-scores)
    matched, tp = set(), np.zeros(len(scores), dtype=bool)
    for idx in order:
        ious = iou[idx]
        j = int(np.argmax(ious))
        if ious[j] >= iou_thr and j not in matched:
            tp[idx] = True
            matched.add(j)
    return scores, tp


def precompute_images(names, preds, gts, thr, mode):
    """
    Per image: n_gt, plus (scores, is_tp) for the OPERATING-POINT set (thresholded at `thr`,
    used for P/R/F1) and for the AP-RANKING set ('pre': the same thresholded set, since in 'pre'
    mode the final predictions already ARE the thresholded/fused output; 'post': the full raw,
    un-thresholded set, matching what score_lists() did before this fix). Each image's matching
    is independent of every other image, so this only needs to run once per (strategy, mode, repeat)
    no matter how many bootstrap resamples or comparisons use it afterwards.
    """
    out = []
    for b, s in preds:
        b = np.asarray(b).reshape(-1, 4)
        s = np.asarray(s).reshape(-1)
        if mode == "pre":
            bm, sm = b, s
        else:
            keep = s >= thr
            bm, sm = b[keep], s[keep]
        out.append((bm, sm, b, s))
    results = []
    for (bm, sm, bap, sap), gt in zip(out, gts):
        sm_scores, sm_tp = per_image_match(bm, sm, gt)
        if mode == "pre":
            sap_scores, sap_tp = sm_scores, sm_tp
        else:
            sap_scores, sap_tp = per_image_match(bap, sap, gt)
        results.append(dict(n_gt=len(gt), m_scores=sm_scores, m_tp=sm_tp, ap_scores=sap_scores, ap_tp=sap_tp))
    return results


def agg_precomputed(precomp, sel):
    """sel: array of indices into `precomp`, with repetition (a bootstrap resample or the identity)."""
    total_gt = sum(precomp[i]["n_gt"] for i in sel)
    m_tp = np.concatenate([precomp[i]["m_tp"] for i in sel]) if len(sel) else np.zeros(0, dtype=bool)
    total_tp = int(m_tp.sum())
    total_fp = len(m_tp) - total_tp
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / total_gt if total_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    ap_scores = np.concatenate([precomp[i]["ap_scores"] for i in sel]) if len(sel) else np.zeros(0)
    ap_tp = np.concatenate([precomp[i]["ap_tp"] for i in sel]) if len(sel) else np.zeros(0, dtype=bool)
    if len(ap_scores) == 0 or total_gt == 0:
        ap = 0.0
    else:
        order = np.argsort(-ap_scores)
        tp_sorted = ap_tp[order]
        tp_cum = np.cumsum(tp_sorted)
        fp_cum = np.cumsum(~tp_sorted)
        precisions = tp_cum / (tp_cum + fp_cum)
        recalls = tp_cum / total_gt
        ap = compute_ap_coco(recalls, precisions)
    return dict(P=precision, R=recall, F1=f1, AP50=ap)


def pick_thr(val_preds, val_gts):
    best, bt = -1, THR
    for t in GRID:
        m = score_lists(val_preds, val_gts, t, "post")
        if m["F1"] > best:
            best, bt = m["F1"], float(t)
    return bt


def task(args):
    """One (strategy, repeat) job -> list of metric rows + OOF predictions for bootstrapping."""
    name, r = args
    P, S, folds = G["P"], G["S"], G["folds"]
    rows, keep = [], {}
    if name.startswith("tta_selected_"):
        size = name.split("_")[-1]
        pred, gt, chosen = oof_selected(P, r, size, folds, THR, G["max_views"])
        if pred is None:
            return name, r, rows, keep
        names = sorted(pred)
        m = score_lists([pred[n] for n in names], [gt[n] for n in names], THR, "pre")
        rows.append(dict(strategy=name, mode="pre", repeat=r, fold="pooled", thr=THR, n_img=len(names), **m))
        keep["pre"] = (names, [pred[n] for n in names], [gt[n] for n in names])
        for k in folds:
            fk = [n for n in names if n in P[(r, k, size)]["test"]]
            mk = score_lists([pred[n] for n in fk], [gt[n] for n in fk], THR, "pre")
            rows.append(dict(strategy=name, mode="pre", repeat=r, fold=k, thr=THR, n_img=len(fk),
                             chosen=str(chosen.get(k)), **mk))
        return name, r, rows, keep
    for mode in ("pre", "post", "post_val"):
        base = "pre" if mode == "pre" else "post"
        pred, gt = oof(P, r, S[name], "test", base, folds)
        if pred is None:
            continue
        thr = THR
        if mode == "post_val":
            vp, vg = oof(P, r, S[name], "val", "post", folds)
            thr = pick_thr([vp[n] for n in vp], [vg[n] for n in vp])
        names = sorted(pred)
        m = score_lists([pred[n] for n in names], [gt[n] for n in names], thr, base)
        rows.append(dict(strategy=name, mode=mode, repeat=r, fold="pooled", thr=thr, n_img=len(names), **m))
        if mode != "post_val":
            keep[mode] = (names, [pred[n] for n in names], [gt[n] for n in names])
            for k in folds:
                # per-fold metrics on the test images of that fold
                fk = [n for n in names if n in P[(r, k, S[name][0][0][0])]["test"]]
                idx = [names.index(n) for n in fk]
                mk = score_lists([pred[n] for n in fk], [gt[n] for n in fk], thr, base)
                rows.append(dict(strategy=name, mode=mode, repeat=r, fold=k, thr=thr, n_img=len(fk), **mk))
    return name, r, rows, keep


def boot_task(args):
    """
    Same statistic as before (mean over repeats of the metric on a bootstrap resample of images,
    A - B, image-level percentile CI), but built from precomputed per-image match results instead
    of re-running the full greedy box-matching on every one of the n_boot resamples: matching is
    the same for a given image no matter which OTHER images end up in a resample, so it only needs
    to be done once (see precompute_images / agg_precomputed above). Verified to reproduce the
    slower reference implementation exactly on synthetic data before this was shipped.
    """
    (a, b, mode, seed, n_boot) = args
    A, B = G["OOF"][(a, mode)], G["OOF"][(b, mode)]
    reps = sorted(set(A) & set(B))
    names = A[reps[0]][0]
    N = len(names)
    base = "pre" if mode == "pre" else "post"
    keys = ["P", "R", "F1", "AP50"]

    pcA = {r: precompute_images(names, A[r][1], A[r][2], THR, base) for r in reps}
    pcB = {r: precompute_images(names, B[r][1], B[r][2], THR, base) for r in reps}

    def mean_metric(pc, sel):
        acc = {k: [] for k in keys}
        for r in reps:
            m = agg_precomputed(pc[r], sel)
            for k in keys:
                acc[k].append(m[k])
        return {k: float(np.mean(v)) for k, v in acc.items()}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, N, size=(n_boot, N))
    full = np.arange(N)
    pa, pb = mean_metric(pcA, full), mean_metric(pcB, full)
    diffs = {k: np.empty(n_boot) for k in keys}
    for i in range(n_boot):
        ma, mb = mean_metric(pcA, idx[i]), mean_metric(pcB, idx[i])
        for k in keys:
            diffs[k][i] = ma[k] - mb[k]
    rows = []
    for k in keys:
        d = diffs[k]
        rows.append(dict(A=a, B=b, mode=mode, metric=k, A_val=pa[k], B_val=pb[k], diff=pa[k] - pb[k],
                         ci_lo=np.percentile(d, 2.5), ci_hi=np.percentile(d, 97.5),
                         boot_p_two_sided=float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))),
                         n_boot=n_boot, n_repeats=len(reps)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="nested_out")
    ap.add_argument("--tag", default="direct")
    ap.add_argument("--seq-tag", default=None, help="tag of the sequential-transfer run, e.g. sequential")
    ap.add_argument("--scales", nargs="+", type=float, default=[0.9, 1.0, 1.1])
    ap.add_argument("--fine-scales", nargs="+", type=float, default=None,
                    help="e.g. 0.8 0.9 1.0 1.1 1.2 -- only used for strategies whose views were "
                         "computed by predict_extra_views.py; ignored (strategy silently absent "
                         "from the output) if those views are not cached yet")
    ap.add_argument("--rot-angles", nargs="+", type=float, default=None,
                    help="e.g. 5 8 -- must match --rot-angles used in predict_extra_views.py")
    ap.add_argument("--max-views", type=int, default=10, help="cap for the greedy view-selection search")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    ev = out / "eval"
    ev.mkdir(exist_ok=True)

    P = load(out, a.tag)
    assert P, f"no predictions found for tag {a.tag}"
    repeats = sorted({k[0] for k in P})
    folds = sorted({k[1] for k in P})
    sizes = [s for s in ["n", "m", "l"] if any(k[2] == s for k in P)]
    S = strategies(sizes, a.scales, a.fine_scales, a.rot_angles)
    S_extra = {}
    if a.seq_tag:
        Pseq = load(out, a.seq_tag)
        for k, v in Pseq.items():
            P[(k[0], k[1], "seq_" + k[2])] = v
        for s in sizes:
            S_extra[f"seq_single_{s}"] = ([("seq_" + s, "std")], [1.0], "avg")
    S.update(S_extra)
    for s in sizes:  # greedy per-fold view selection; handled specially in task(), no entry needed in S
        pass
    print(f"loaded: repeats={repeats} folds={folds} sizes={sizes} strategies={len(S)} "
          f"(+ tta_selected_{{size}} via greedy per-fold selection, max {a.max_views} views)")
    G.update(P=P, S=S, folds=folds, max_views=a.max_views)

    selected_names = [f"tta_selected_{s}" for s in sizes]
    jobs = [(n, r) for n in list(S) + selected_names for r in repeats]
    with mp.get_context("fork").Pool(a.jobs) as pool:
        res = pool.map(task, jobs)
    rows, OOF = [], {}
    for name, r, rr, keep in res:
        rows += rr
        for mode, v in keep.items():
            OOF.setdefault((name, mode), {})[r] = v
    df = pd.DataFrame(rows)
    df.to_csv(ev / "metrics_long.csv", index=False)
    pooled = df[df.fold == "pooled"]
    summ = (pooled.groupby(["strategy", "mode"])[["P", "R", "F1", "AP50", "thr"]]
            .agg(["mean", "std"]).round(4))
    summ.columns = ["_".join(c) for c in summ.columns]
    summ.reset_index().to_csv(ev / "summary_over_repeats.csv", index=False)
    perfold = df[df.fold != "pooled"]
    perfold.to_csv(ev / "per_fold.csv", index=False)

    # ------------------------------------------------ paired bootstrap on key contrasts
    pairs = []
    have = lambda n: (n, "pre") in OOF
    if "m" in sizes:
        base_tta = ["tta_paper_m", "tta_imgsz_m", "tta_flip_m", "tta_scale_m",
                    "tta_finescale_m", "tta_vflip_m", "tta_rot_m", "tta_kitchen_m", "tta_selected_m"]
        for t in base_tta:
            pairs.append((t, "single_m"))
        # is each new variant actually better than the previous best (tta_imgsz), not just than single-pass?
        for t in ["tta_finescale_m", "tta_vflip_m", "tta_rot_m", "tta_kitchen_m", "tta_selected_m"]:
            pairs.append((t, "tta_imgsz_m"))
            pairs.append((t, "tta_paper_m"))
        if "n" in sizes:
            for t in base_tta:
                pairs.append((t, "ens_nm_11"))
            pairs += [("ens_nm_11", "single_m"), ("single_m", "single_n")]
        if "l" in sizes:
            for t in base_tta:
                pairs.append((t, "ens_ml_11"))
            pairs += [("ens_ml_11", "single_m"), ("single_m", "single_l")]
    if a.seq_tag and "m" in sizes:
        pairs.append(("single_m", "seq_single_m"))
    pairs = sorted(set((x, y) for x, y in pairs if have(x) and have(y) and x != y))
    G["OOF"] = OOF
    def modes_for(x, y):
        return ["pre"] if x.startswith("tta_selected_") or y.startswith("tta_selected_") else ["pre", "post"]
    bjobs = [(x, y, mode, a.seed, a.n_boot) for x, y in pairs for mode in modes_for(x, y)
             if (x, mode) in OOF and (y, mode) in OOF]
    print(f"bootstrap: {len(bjobs)} comparisons x {a.n_boot} resamples")
    with mp.get_context("fork").Pool(a.jobs) as pool:
        brows = sum(pool.map(boot_task, bjobs), [])
    bd = pd.DataFrame(brows, columns=["A", "B", "mode", "metric", "A_val", "B_val", "diff", "ci_lo", "ci_hi",
                                      "boot_p_two_sided", "n_boot", "n_repeats"])
    bd.to_csv(ev / "paired_bootstrap.csv", index=False)
    if bd.empty:
        print("(no paired comparisons available with the sizes/tags present)")

    # ------------------------------------------------ human-readable summary
    pd.set_option("display.width", 200)
    show = summ.reset_index()
    main_rows = ["single_n", "single_m", "single_l", "ens_nm_11", "ens_ml_11", "tta_paper_m", "tta_imgsz_m",
                 "tta_flip_m", "tta_scale_m", "tta_finescale_m", "tta_vflip_m", "tta_rot_m", "tta_kitchen_m",
                 "tta_selected_m", "tta_imgsz_m_maxconf", "seq_single_m"]
    print("\n=== mean +- std over repeats (pooled out-of-fold, all images) ===")
    for mode in ("pre", "post", "post_val"):
        t = show[(show["mode"] == mode) & (show.strategy.isin(main_rows))]
        print(f"\n[{mode}]")
        print(t.to_string(index=False))
    print("\n=== paired bootstrap (A - B, 95% CI over images) ===")
    print(bd[bd.metric.isin(["R", "F1", "AP50"])].round(4).to_string(index=False))
    tex = ["\\begin{tabular}{llcccc}", "\\toprule", "Strategy & Mode & Precision & Recall & F1 & AP50 \\\\", "\\midrule"]
    for _, r_ in show[show.strategy.isin(main_rows) & show["mode"].isin(["pre", "post"])].iterrows():
        f = lambda k: f"{r_[k + '_mean']:.3f} $\\pm$ {0 if np.isnan(r_[k + '_std']) else r_[k + '_std']:.3f}"
        tex.append(f"{r_.strategy.replace('_', ' ')} & {r_['mode']} & {f('P')} & {f('R')} & {f('F1')} & {f('AP50')} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (ev / "summary_table.tex").write_text("\n".join(tex))
    print("\nwritten to", ev)


if __name__ == "__main__":
    main()
