"""Zero-shot out-of-distribution test: LIACI corrosion detectors, without any retraining, on the TrashCan
val images that contain trash_metal objects. Every 'corrosion' prediction is scored against the Metal boxes
(other TrashCan classes are ignored), so the numbers answer "does a corrosion detector fire on metal debris
in another underwater domain?" and nothing more.

For every nested-CV fold checkpoint of a (tag, model) group, linked by `nested_eval link` under
r1/runs/<tag>/<model>_seed<repeat>/fold<k>: standard and TTA dumps on those images, metrics with the single
evaluator at conf_fixed, and a per-checkpoint paired bootstrap of TTA - standard.
Outputs: r1/results/trashcan_ood.csv, r1/results/trashcan_ood_summary.csv, r1/results/trashcan_ood_deltas.csv.
"""

from __future__ import annotations

import argparse

import numpy as np

from r1 import evaluate as ev
from r1 import predict_dump as pdump
from r1.collect_runs import t_interval, write_csv
from r1.protocol import get_logger, load_protocol, read_json, resolve
from r1.run_test_inference import METRIC_KEYS, chosen_tta, summarize, tta_dump_name
from r1.sweep_inference import fold_runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="r1/runs")
    ap.add_argument("--tag", default="direct", help="nested-CV protocol linked under r1/runs (direct | sequential)")
    ap.add_argument("--models", default="yolo26m")
    ap.add_argument("--choice", default=None, help="validation choice file (default r1/results/inference_choice.json)")
    ap.add_argument("--tta-config", default=None, help="override the TTA configuration chosen on LIACI validation")
    ap.add_argument("--wbf-iou", type=float, default=None)
    ap.add_argument("--subset", choices=["metal", "all"], default="metal", help="val images with Metal boxes (default) or the whole val split")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N images (local checks)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="r1/results")
    ap.add_argument("--include-dry-run", action="store_true")
    args = ap.parse_args()

    proto = load_protocol()
    log = get_logger("trashcan_ood", resolve("r1/results/logs/trashcan_ood.log"))
    evc = proto["evaluation"]
    cfg = proto["trashcan"]
    data_root = resolve(cfg["output_root"])
    manifest = read_json(resolve(cfg["manifest_file"]))
    metal_id = next(int(k) for k, v in manifest["names"].items() if v == "Metal")
    if args.subset == "metal":
        names = read_json(data_root / "val_metal.json")["images"]
    else:
        names = sorted(e["image"] for e in manifest["images"] if e["split"] == "val")
    if args.limit:
        names = names[: args.limit]
    gt = ev.filter_class(ev.load_ground_truth(data_root, "val", names), metal_id)
    names = sorted(names)
    images = [data_root / "images" / "val" / n for n in names]
    device = args.device
    if device is None:
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tta_config, tta_iou, local = chosen_tta(proto, args.choice, args.tta_config, args.wbf_iou)
    tag = "dryrun" if args.include_dry_run else args.tag
    log.info(f"{len(names)} TrashCan val images ({args.subset}), {sum(len(g['boxes']) for g in gt.values())} Metal boxes | tag={tag} models={args.models}")

    rows, deltas = [], []
    for model in args.models.split(","):
        for run in fold_runs(resolve(args.runs), tag, model, include_dry_run=args.include_dry_run):
            dumps = {
                "standard": pdump.ensure_dump("standard", [run["best"]], images, run["dir"] / f"preds_trashcan_{args.subset}", proto, device, None, log),
                "tta": pdump.ensure_dump("tta", [run["best"]], images, run["dir"] / tta_dump_name(f"preds_trashcan_{args.subset}", tta_config, tta_iou), local, device, None, log),
            }
            preds = {s: ev.load_predictions(d, names) for s, d in dumps.items()}
            for s in dumps:
                m = ev.all_metrics(preds[s], gt, names, evc["conf_fixed"], evc["iou_match"])
                rows.append({"model": model, "seed": run["seed"], "fold": run["fold"], "strategy": s, "dump": str(dumps[s]), **{k: m[k] for k in METRIC_KEYS}, "tp": m["tp"], "fp": m["fp"], "fn": m["fn"]})
            d = ev.paired_bootstrap(preds["standard"], preds["tta"], gt, names, evc["conf_fixed"], evc["iou_match"], evc["bootstrap_iters"], evc["bootstrap_seed"])
            deltas.append({"model": model, "seed": run["seed"], "fold": run["fold"], "comparison": "tta-standard", **{f"{k}_{s}": v for k, dd in d.items() for s, v in (("delta", dd["delta_mean"]), ("ci_lo", dd["ci95"][0]), ("ci_hi", dd["ci95"][1]), ("p_le0", dd["p_le0"]))}})
            log.info(f"{model} seed{run['seed']} fold{run['fold']}: standard recall {rows[-2]['recall']:.3f} P {rows[-2]['precision']:.3f} mAP50 {rows[-2]['map50']:.3f} | tta recall {rows[-1]['recall']:.3f}")
    if not rows:
        raise SystemExit("no finished checkpoints found")
    out = resolve(args.out) if not args.include_dry_run else resolve(args.runs) / "dryrun" / "results"
    write_csv(out / "trashcan_ood.csv", rows)
    write_csv(out / "trashcan_ood_summary.csv", summarize(rows))
    mean_rows = []
    for model in sorted({r["model"] for r in deltas}):
        sub = [r for r in deltas if r["model"] == model]
        row = {"model": model, "seed": "all", "fold": "all", "comparison": "tta-standard"}
        for k in ("map50", "f1", "recall", "precision"):
            m, lo, hi = t_interval([r[f"{k}_delta"] for r in sub])
            row.update({f"{k}_delta": round(m, 4), f"{k}_ci_lo": round(lo, 4), f"{k}_ci_hi": round(hi, 4), f"{k}_p_le0": round(float(np.mean([r[f"{k}_p_le0"] for r in sub])), 4)})
        mean_rows.append(row)
    write_csv(out / "trashcan_ood_deltas.csv", deltas + mean_rows)
    log.info(f"written {out / 'trashcan_ood.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
