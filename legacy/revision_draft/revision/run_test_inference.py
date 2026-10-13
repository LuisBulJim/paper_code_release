"""Apply the inference strategies chosen on validation to the frozen TEST set, for every finished checkpoint.

The choice comes from results/inference_choice.json, written by sweep_inference.py with a fixed rule
(validation only). --tta-config / --wbf-* override it only for explicit sensitivity runs.

For each (tag, model) group and each of its checkpoints (5 folds x 3 seeds = 15 per size):
  standard  the dump produced by train_kfold.py (preds_test) is re-used
  tta       one dump with the chosen TTA configuration
  wbf       (optional) one dump fusing the same seed/fold checkpoints of the chosen sizes
Dump directories encode every setting and are re-used only if their _meta.json matches.
All dumps are evaluated with evaluate.py at conf_fixed and the paired bootstrap of (strategy - reference)
is computed per checkpoint.

Outputs (results/, suffixed with _<tag> for tags other than dtl):
  inference_test.csv           one row per (model, seed, fold, strategy) with the metrics
  inference_test_summary.csv   mean, std, 95% t-interval over checkpoints per (model, strategy)
  inference_test_deltas.csv    per checkpoint paired-bootstrap delta with CI and p_le0, plus the mean over checkpoints
  inference_test_config.json   the settings actually applied
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from revision import evaluate as ev
from revision import predict_dump as pdump
from revision.collect_runs import t_interval, write_csv
from revision.protocol import get_logger, load_protocol, read_json, resolve, write_json
from revision.sweep_inference import TTA_GRID, fold_runs

METRIC_KEYS = ("map50", "map5095", "precision", "recall", "f1")
DEFAULT_CHOICE = "results/inference_choice.json"


def tta_protocol(proto: dict, config: str, wbf_iou: float | None) -> dict:
    if config not in TTA_GRID or TTA_GRID[config] is None:
        raise SystemExit(f"TTA configuration must be one of {[k for k, v in TTA_GRID.items() if v]}, got {config!r}")
    return pdump.tta_protocol(proto, TTA_GRID[config], wbf_iou)


def load_choice(path: str | None, required: bool) -> dict:
    p = resolve(path or DEFAULT_CHOICE)
    if not p.exists():
        if required:
            raise SystemExit(f"{p} missing: run revision.sweep_inference first, or pass --tta-config explicitly")
        return {}
    return read_json(p)


def chosen_tta(proto: dict, choice_path: str | None, tta_config: str | None, wbf_iou: float | None) -> tuple[str, float, dict]:
    """(config, view IoU, protocol copy) from the validation choice file unless explicitly overridden."""
    choice = load_choice(choice_path, required=tta_config is None)
    config = tta_config or choice["tta"]["config"]
    iou = wbf_iou if wbf_iou is not None else choice.get("tta", {}).get("wbf_iou", proto["inference"]["tta"]["wbf_iou"])
    return config, iou, tta_protocol(proto, config, iou)


def tta_dump_name(prefix: str, config: str, wbf_iou: float) -> str:
    return f"{prefix}_{config}_iou{wbf_iou}"


def wbf_dump_name(prefix: str, combo: list[str], weights: list[float] | None, iou: float) -> str:
    w = "equal" if not weights else "-".join(f"{x:g}" for x in weights)
    return f"{prefix}_wbf_{'+'.join(combo)}_w{w}_iou{iou}"


def delta_row(base: dict, comparison: str, d: dict) -> dict:
    return {**base, "comparison": comparison, **{f"{k}_{s}": v for k, dd in d.items() for s, v in (("delta", dd["delta_mean"]), ("ci_lo", dd["ci95"][0]), ("ci_hi", dd["ci95"][1]), ("p_le0", dd["p_le0"]))}}


def mean_delta_rows(deltas: list[dict], group_keys: tuple[str, ...]) -> list[dict]:
    out = []
    for key in sorted({tuple(r[k] for k in group_keys) for r in deltas}):
        sub = [r for r in deltas if tuple(r[k] for k in group_keys) == key]
        row = {**dict(zip(group_keys, key)), "seed": "all", "fold": "all"}
        for k in ("map50", "f1", "recall", "precision"):
            mean, lo, hi = t_interval([r[f"{k}_delta"] for r in sub])
            row.update({f"{k}_delta": round(mean, 4), f"{k}_ci_lo": round(lo, 4), f"{k}_ci_hi": round(hi, 4), f"{k}_p_le0": round(float(np.mean([r[f"{k}_p_le0"] for r in sub])), 4)})
        out.append(row)
    return out


def evaluate_checkpoint(run: dict, strategy: str, dump: Path, gt: ev.GT, names: list[str], conf: float, iou: float, model: str) -> dict:
    preds = ev.load_predictions(dump, names)
    m = ev.all_metrics(preds, gt, names, conf, iou)
    return {"model": model, "seed": run["seed"], "fold": run["fold"], "strategy": strategy, "dump": str(dump), **{k: m[k] for k in METRIC_KEYS}, "tp": m["tp"], "fp": m["fp"], "fn": m["fn"]}


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    for model, strategy in sorted({(r["model"], r["strategy"]) for r in rows}):
        sub = [r for r in rows if (r["model"], r["strategy"]) == (model, strategy)]
        row = {"model": model, "strategy": strategy, "n_checkpoints": len(sub)}
        for k in METRIC_KEYS:
            mean, lo, hi = t_interval([r[k] for r in sub])
            row[f"{k}_mean"] = round(mean, 4)
            row[f"{k}_std"] = round(float(np.std([r[k] for r in sub], ddof=1)), 4) if len(sub) > 1 else float("nan")
            row[f"{k}_ci_lo"] = round(lo, 4)
            row[f"{k}_ci_hi"] = round(hi, 4)
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--tag", default="dtl")
    ap.add_argument("--models", default="yolo26n,yolo26m,yolo26l")
    ap.add_argument("--choice", default=None, help=f"validation choice file (default {DEFAULT_CHOICE})")
    ap.add_argument("--tta-config", default=None, help="override the chosen TTA configuration (sensitivity runs only)")
    ap.add_argument("--wbf-iou", type=float, default=None, help="override the chosen WBF IoU of the TTA views")
    ap.add_argument("--wbf-models", default=None, help="override the chosen ensemble, e.g. yolo26n,yolo26m; 'none' disables it")
    ap.add_argument("--wbf-weights", default=None, help="override the ensemble weights, comma-separated")
    ap.add_argument("--wbf-models-iou", type=float, default=None, help="override the ensemble WBF IoU")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--include-dry-run", action="store_true", help="local pipeline check only")
    args = ap.parse_args()

    proto = load_protocol()
    log = get_logger("run_test_inference", resolve("results/logs/run_test_inference.log"))
    evc = proto["evaluation"]
    conf = evc["conf_fixed"]
    choice = load_choice(args.choice, required=args.tta_config is None)
    tta_config = args.tta_config or choice["tta"]["config"]
    tta_iou = args.wbf_iou if args.wbf_iou is not None else choice.get("tta", {}).get("wbf_iou", proto["inference"]["tta"]["wbf_iou"])
    wbf_choice = choice.get("wbf") or {}
    wbf_models = None if args.wbf_models == "none" else (args.wbf_models.split(",") if args.wbf_models else wbf_choice.get("models"))
    wbf_weights = [float(w) for w in args.wbf_weights.split(",")] if args.wbf_weights else wbf_choice.get("weights")
    wbf_iou = args.wbf_models_iou if args.wbf_models_iou is not None else wbf_choice.get("iou", proto["inference"]["wbf_models"]["iou"])

    dataset_root = resolve(proto["dataset"]["output_root"])
    test_names = read_json(resolve(proto["dataset"]["test_images_file"]))["images"]
    device = args.device
    if device is None:
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    runs_root = resolve(args.runs)
    tag = "dryrun" if args.include_dry_run else args.tag
    models = args.models.split(",")
    local_tta = tta_protocol(proto, tta_config, tta_iou)
    local_wbf = pdump.wbf_protocol(proto, wbf_iou)
    log.info(f"tag={tag} models={models} tta={tta_config} iou={tta_iou} | wbf={wbf_models} weights={wbf_weights} iou={wbf_iou} | conf={conf} device={device}")

    def test_images(run_dir: Path) -> list[str]:
        return read_json(run_dir.parent / "dryrun_test_images.json")["images"] if args.include_dry_run else test_names

    rows, deltas = [], []
    per_model = {m: {(r["seed"], r["fold"]): r for r in fold_runs(runs_root, tag, m, include_dry_run=args.include_dry_run)} for m in models}
    for model in models:
        for (seed, fold), run in sorted(per_model[model].items()):
            names = test_images(run["dir"])
            images = [dataset_root / "images" / "test" / n for n in names]
            gt = ev.load_ground_truth(dataset_root, "test", names)
            names = sorted(names)
            std_dump = run["dir"] / "preds_test"
            rows.append(evaluate_checkpoint(run, "standard", std_dump, gt, names, conf, evc["iou_match"], model))
            tta_dump = pdump.ensure_dump("tta", [run["best"]], images, run["dir"] / tta_dump_name("preds_test", tta_config, tta_iou), local_tta, device, None, log)
            rows.append(evaluate_checkpoint(run, "tta", tta_dump, gt, names, conf, evc["iou_match"], model))
            d = ev.paired_bootstrap(ev.load_predictions(std_dump, names), ev.load_predictions(tta_dump, names), gt, names, conf, evc["iou_match"], evc["bootstrap_iters"], evc["bootstrap_seed"])
            deltas.append(delta_row({"model": model, "seed": seed, "fold": fold}, "tta-standard", d))
            log.info(f"{model} seed{seed} fold{fold}: standard mAP50 {rows[-2]['map50']:.4f} | tta {rows[-1]['map50']:.4f} | delta F1 {d['f1']['delta_mean']:+.4f} [{d['f1']['ci95'][0]:+.3f}, {d['f1']['ci95'][1]:+.3f}]")

    if wbf_models:
        missing = [m for m in wbf_models if m not in per_model]
        if missing:
            raise SystemExit(f"ensemble members {missing} are not in --models {models}")
        label = "+".join(wbf_models)
        keys = set.intersection(*(set(per_model[m]) for m in wbf_models))
        for seed, fold in sorted(keys):
            run0 = per_model[wbf_models[0]][(seed, fold)]
            names = test_images(run0["dir"])
            images = [dataset_root / "images" / "test" / n for n in names]
            gt = ev.load_ground_truth(dataset_root, "test", names)
            names = sorted(names)
            ckpts = [per_model[m][(seed, fold)]["best"] for m in wbf_models]
            wbf_dump = pdump.ensure_dump("wbf", ckpts, images, run0["dir"] / wbf_dump_name("preds_test", wbf_models, wbf_weights, wbf_iou), local_wbf, device, wbf_weights, log)
            rows.append(evaluate_checkpoint(run0, "wbf", wbf_dump, gt, names, conf, evc["iou_match"], label))
            for ref_model in wbf_models:
                ref = per_model[ref_model][(seed, fold)]["dir"] / "preds_test"
                d = ev.paired_bootstrap(ev.load_predictions(ref, names), ev.load_predictions(wbf_dump, names), gt, names, conf, evc["iou_match"], evc["bootstrap_iters"], evc["bootstrap_seed"])
                deltas.append(delta_row({"model": label, "seed": seed, "fold": fold}, f"wbf-{ref_model}", d))
            log.info(f"wbf {label} seed{seed} fold{fold}: mAP50 {rows[-1]['map50']:.4f} F1 {rows[-1]['f1']:.4f}")

    if not rows:
        raise SystemExit("no finished checkpoints found")
    out = resolve(args.out)
    suffix = "" if tag in ("dtl", "dryrun") else f"_{tag}"
    write_csv(out / f"inference_test{suffix}.csv", rows)
    write_csv(out / f"inference_test_summary{suffix}.csv", summarize(rows))
    write_csv(out / f"inference_test_deltas{suffix}.csv", deltas + mean_delta_rows(deltas, ("model", "comparison")))
    write_json(out / f"inference_test_config{suffix}.json", {"tag": tag, "choice_file": str(resolve(args.choice or DEFAULT_CHOICE)) if choice else None, "overridden": bool(args.tta_config or args.wbf_models or args.wbf_weights or args.wbf_iou is not None or args.wbf_models_iou is not None), "tta_config": tta_config, "tta": local_tta["inference"]["tta"], "wbf_models": wbf_models, "wbf_weights": wbf_weights, "wbf_iou": wbf_iou, "conf": conf, "iou_match": evc["iou_match"], "n_rows": len(rows)})
    log.info(f"written {out / f'inference_test{suffix}.csv'} ({len(rows)} rows), summary and deltas")


if __name__ == "__main__":
    main()
