"""Generate every LaTeX table of the manuscript from results/*.csv, data/*.json and hparams.yaml.

No number is typed by hand: each table is written to tables/<name>.tex and every value that appears in a
table is also recorded in tables/numbers.json (used by check_consistency.py to verify the manuscript).
Tables whose inputs do not exist yet are skipped with a log line, so the script can run at any stage.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from revision.protocol import get_logger, load_hparams, load_protocol, read_json, resolve

MODEL_LABEL = {"yolo26n": "YOLO26-n", "yolo26m": "YOLO26-m", "yolo26l": "YOLO26-l"}
TAG_LABEL = {"dtl": "Direct transfer", "stl": "Sequential transfer"}
STRATEGY_LABEL = {"standard": "Standard", "tta": "TTA", "wbf": "WBF ensemble"}


class Numbers:
    def __init__(self):
        self.values: dict[str, list[float]] = {}

    def add(self, table: str, value, fmt: str | None = None) -> str:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return str(value)
        if v != v:  # nan
            return "--"
        self.values.setdefault(table, []).append(v)
        if fmt:
            return fmt.format(v)
        if float(v).is_integer() and abs(v) >= 2:  # counts
            return f"{int(v)}"
        if abs(v) >= 100:
            return f"{v:.1f}"
        return f"{v:.3f}"


def tex(s) -> str:
    return str(s).replace("_", "\\_")


def read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def latex_table(header: list[str], rows: list[list[str]], caption: str, label: str, align: str | None = None) -> str:
    align = align or "l" + "r" * (len(header) - 1)
    lines = ["\\begin{table}[htbp]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def ci(num: Numbers, table: str, row: dict, key: str) -> str:
    return f"{num.add(table, row[f'{key}_mean'])} [{num.add(table, row[f'{key}_ci_lo'])}, {num.add(table, row[f'{key}_ci_hi'])}]"


def table_hparams(hp: dict, num: Numbers) -> str:
    rows = [[tex(k), num.add("hparams", v) if not isinstance(v, (str, bool)) and v is not None else str(v)] for k, v in hp.items()]
    return latex_table(["Hyperparameter", "Value"], rows, "Training hyperparameters shared by every run (Direct Transfer, Sequential Transfer fine-tuning and TrashCan), as passed to Ultralytics.", "tab:hparams", "ll")


def table_dataset(manifest: dict, stats: dict | None, num: Numbers) -> str:
    s = manifest["summary"]
    rows = [
        ["Training pool (5-fold CV)", num.add("dataset", s["pool"].get("positive", 0)), num.add("dataset", s["pool"].get("negative", 0)), num.add("dataset", s["n_boxes_pool"])],
        ["Held-out test", num.add("dataset", s["test"].get("positive", 0)), num.add("dataset", s["test"].get("negative", 0)), num.add("dataset", s["n_boxes_test"])],
    ]
    out = latex_table(["Partition", "Images with corrosion", "Background images", "Boxes"], rows, "LIACI corrosion detection dataset (v2). Background images sampled with a fixed seed from LIACI images outside its official test split.", "tab:dataset")
    if stats and "pool" in stats and "test" in stats:
        rows = [[b, num.add("dataset", stats["pool"]["size_buckets"].get(b, 0)), num.add("dataset", stats["test"]["size_buckets"].get(b, 0))] for b in ("small", "medium", "large")]
        rows.append(["Boxes per corroded image (median / max)", f"{num.add('dataset', stats['pool']['boxes_per_positive_image']['median'], '{:.0f}')} / {num.add('dataset', stats['pool']['boxes_per_positive_image']['max'])}", f"{num.add('dataset', stats['test']['boxes_per_positive_image']['median'], '{:.0f}')} / {num.add('dataset', stats['test']['boxes_per_positive_image']['max'])}"])
        out += latex_table(["Ground-truth boxes", "Training pool", "Test"], rows, "Box sizes at 640$\\times$640 (COCO buckets: small $<32^2$, medium $<96^2$ px$^2$, large otherwise).", "tab:box_sizes", "lrr")
    return out


def table_cv(summary: list[dict], num: Numbers) -> str:
    rows = []
    for r in summary:
        if r["stage"] != "val":
            continue
        rows.append([TAG_LABEL.get(r["tag"], r["tag"]), MODEL_LABEL.get(r["model"], r["model"]), r["n_runs"], f"{num.add('cv', r['map50_mean'])} $\\pm$ {num.add('cv', r['map50_std'])}", f"{num.add('cv', r['map5095_mean'])} $\\pm$ {num.add('cv', r['map5095_std'])}", f"{num.add('cv', r['f1_mean'])} $\\pm$ {num.add('cv', r['f1_std'])}"])
    return latex_table(["Protocol", "Model", "Runs", "mAP@50 (val)", "mAP@[.5:.95] (val)", "F1 (val)"], rows, "Stratified 5-fold cross-validation over 3 seeds on the training pool: mean $\\pm$ SD over the 15 folds of the metrics on each fold's validation split.", "tab:cv")


def table_test(summary: list[dict], num: Numbers) -> str:
    rows = []
    for r in summary:
        if r["stage"] != "test":
            continue
        rows.append([TAG_LABEL.get(r["tag"], r["tag"]), MODEL_LABEL.get(r["model"], r["model"]), ci(num, "test", r, "map50"), ci(num, "test", r, "map5095"), ci(num, "test", r, "precision"), ci(num, "test", r, "recall"), ci(num, "test", r, "f1")])
    return latex_table(["Protocol", "Model", "mAP@50", "mAP@[.5:.95]", "Precision", "Recall", "F1"], rows, "Held-out test set (33 images): mean over the 15 fold checkpoints with 95\\% t-interval; P/R/F1 at the fixed operating point (conf 0.25, IoU 0.5). No checkpoint was selected using the test set.", "tab:test")


def table_inference(summary: list[dict], deltas: list[dict], num: Numbers) -> str:
    rows = []
    for r in summary:
        rows.append([MODEL_LABEL.get(r["model"], r["model"]), STRATEGY_LABEL.get(r["strategy"], r["strategy"]), r["n_checkpoints"], ci(num, "inference", r, "map50"), ci(num, "inference", r, "precision"), ci(num, "inference", r, "recall"), ci(num, "inference", r, "f1")])
    out = latex_table(["Model", "Strategy", "Ckpts", "mAP@50", "Precision", "Recall", "F1"], rows, "Inference strategies on the held-out test set, applied to every fold checkpoint (mean and 95\\% t-interval). TTA/WBF configurations were chosen on validation folds only.", "tab:inference")
    drows = [[MODEL_LABEL.get(r["model"], r["model"]), r["comparison"], f"{num.add('inference_deltas', r['map50_delta'])} [{num.add('inference_deltas', r['map50_ci_lo'])}, {num.add('inference_deltas', r['map50_ci_hi'])}]", f"{num.add('inference_deltas', r['f1_delta'])} [{num.add('inference_deltas', r['f1_ci_lo'])}, {num.add('inference_deltas', r['f1_ci_hi'])}]", f"{num.add('inference_deltas', r['recall_delta'])}", num.add("inference_deltas", r["f1_p_le0"])] for r in deltas if r["seed"] == "all"]
    out += latex_table(["Model", "Comparison", "$\\Delta$ mAP@50 [95\\% CI]", "$\\Delta$ F1 [95\\% CI]", "$\\Delta$ Recall", "$P(\\Delta F1 \\le 0)$"], drows, "Paired differences (strategy $-$ standard) on the test set: mean over checkpoints of the per-checkpoint paired-bootstrap delta, with t-interval over checkpoints and mean bootstrap probability that the difference is not positive.", "tab:inference_deltas")
    return out


def table_latency(rows_in: list[dict], num: Numbers) -> str:
    rows = [[STRATEGY_LABEL.get(r["strategy"], r["strategy"]), r["precision"], num.add("latency", r["ms_mean"]), num.add("latency", r["ms_p95"]), num.add("latency", r["img_per_s"]), num.add("latency", r["peak_vram_mb"]), r["forward_passes_per_image"]] for r in rows_in]
    return latex_table(["Strategy", "Precision", "Latency (ms)", "p95 (ms)", "img/s", "Peak VRAM (MB)", "Passes"], rows, f"Inference cost measured on {rows_in[0]['device']} (batch 1, 640 px, 20 warm-up iterations, {rows_in[0]['n_timed']} timed inferences).", "tab:latency")


def table_trashcan(summary: list[dict], deltas: list[dict], num: Numbers) -> str:
    rows = [[STRATEGY_LABEL.get(r["strategy"], r["strategy"]), r["class"], r["n_seeds"], ci(num, "trashcan", r, "map50"), ci(num, "trashcan", r, "precision"), ci(num, "trashcan", r, "recall"), ci(num, "trashcan", r, "f1")] for r in summary]
    out = latex_table(["Strategy", "Class", "Seeds", "mAP@50", "Precision", "Recall", "F1"], rows, "TrashCan 1.0 (material version, Metal and Bio classes), official validation split: mean over 3 training seeds with 95\\% t-interval, same hyperparameters and evaluator as LIACI.", "tab:trashcan")
    d = [r for r in deltas if r["seed"] == "all"]
    if d:
        r = d[0]
        out += latex_table(["Comparison", "$\\Delta$ mAP@50 [95\\% CI]", "$\\Delta$ F1 [95\\% CI]", "$\\Delta$ Recall", "$P(\\Delta F1 \\le 0)$"], [[r["comparison"], f"{num.add('trashcan_deltas', r['map50_delta'])} [{num.add('trashcan_deltas', r['map50_ci_lo'])}, {num.add('trashcan_deltas', r['map50_ci_hi'])}]", f"{num.add('trashcan_deltas', r['f1_delta'])} [{num.add('trashcan_deltas', r['f1_ci_lo'])}, {num.add('trashcan_deltas', r['f1_ci_hi'])}]", num.add("trashcan_deltas", r["recall_delta"]), num.add("trashcan_deltas", r["f1_p_le0"])]], "TrashCan: paired difference TTA $-$ standard (mean over seeds of the per-seed paired bootstrap).", "tab:trashcan_deltas")
    return out


def table_trashcan_ood(summary: list[dict], num: Numbers) -> str:
    rows = [[MODEL_LABEL.get(r["model"], r["model"]), STRATEGY_LABEL.get(r["strategy"], r["strategy"]), r["n_checkpoints"], ci(num, "trashcan_ood", r, "map50"), ci(num, "trashcan_ood", r, "precision"), ci(num, "trashcan_ood", r, "recall"), ci(num, "trashcan_ood", r, "f1")] for r in summary]
    return latex_table(["LIACI model", "Strategy", "Ckpts", "mAP@50", "Precision", "Recall", "F1"], rows, "Zero-shot out-of-distribution test: LIACI corrosion detectors without retraining on the TrashCan validation images containing metal debris (corrosion predictions scored against the Metal boxes).", "tab:trashcan_ood")


def table_sam3(summary: list[dict], num: Numbers) -> str:
    rows = [[tex(r["subset"]), r["n"], tex(r["metric"]), num.add("sam3", r["value"]), num.add("sam3", r["sd"]) if r["sd"] else "--", f"[{num.add('sam3', r['ci_lo'])}, {num.add('sam3', r['ci_hi'])}]" if r["ci_lo"] else "--"] for r in summary]
    return latex_table(["Subset", "N", "Metric", "Value", "SD", "95\\% CI"], rows, f"SAM 3 box-prompted segmentation on the LIACI official test split (detector: {tex(summary[0]['detector'])}, TTA={summary[0]['tta']}). Positives and background images are reported separately; an undetected positive counts as IoU 0.", "tab:sam3")


def table_sweep(summary: list[dict], conf: float, num: Numbers) -> str:
    rows = []
    for r in sorted((r for r in summary if float(r["conf"]) == conf and r["group"] == "tta"), key=lambda r: (r["model"], -float(r["map50_mean"]))):
        rows.append([MODEL_LABEL.get(r["model"], r["model"]), tex(r["config"]), r["wbf_iou"] or "--", f"{num.add('sweep', r['map50_mean'])} $\\pm$ {num.add('sweep', r['map50_std'])}", f"{num.add('sweep', r['f1_mean'])} $\\pm$ {num.add('sweep', r['f1_std'])}", f"{num.add('sweep', r['recall_mean'])}"])
    return latex_table(["Model", "TTA configuration", "WBF IoU", "mAP@50 (val)", "F1 (val)", "Recall (val)"], rows, f"Sensitivity of TTA to its settings on the validation folds (mean $\\pm$ SD over 15 folds, operating point {conf}). The configuration used on the test set is the best mean validation mAP@50.", "tab:sweep_tta")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="tables")
    args = ap.parse_args()
    proto = load_protocol()
    log = get_logger("make_tables")
    res, out = resolve(args.results), resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    num = Numbers()
    written, skipped = [], []

    def emit(name: str, text: str | None) -> None:
        if text is None:
            skipped.append(name)
            return
        (out / f"{name}.tex").write_text(text, encoding="utf-8")
        written.append(name)

    emit("hparams", table_hparams(load_hparams(), num))
    manifest_path = resolve(proto["dataset"]["manifest_file"])
    stats_path = res / "dataset_stats.json"
    emit("dataset", table_dataset(read_json(manifest_path), read_json(stats_path) if stats_path.exists() else None, num) if manifest_path.exists() else None)
    summary = read_csv(res / "summary.csv")
    emit("cv", table_cv(summary, num) if summary else None)
    emit("test", table_test(summary, num) if summary else None)
    inf_summary, inf_deltas = read_csv(res / "inference_test_summary.csv"), read_csv(res / "inference_test_deltas.csv")
    emit("inference", table_inference(inf_summary, inf_deltas, num) if inf_summary else None)
    latency = read_csv(res / "latency.csv")
    emit("latency", table_latency(latency, num) if latency else None)
    tc_summary, tc_deltas = read_csv(res / "trashcan_summary.csv"), read_csv(res / "trashcan_deltas.csv")
    emit("trashcan", table_trashcan(tc_summary, tc_deltas, num) if tc_summary else None)
    ood = read_csv(res / "trashcan_ood_summary.csv")
    emit("trashcan_ood", table_trashcan_ood(ood, num) if ood else None)
    sam = read_csv(res / "sam3_summary.csv")
    emit("sam3", table_sam3(sam, num) if sam else None)
    sweep = read_csv(res / "sweep_tta_val_summary.csv")
    emit("sweep_tta", table_sweep(sweep, proto["evaluation"]["conf_fixed"], num) if sweep else None)

    (out / "numbers.json").write_text(json.dumps(num.values, indent=1), encoding="utf-8")
    log.info(f"written: {written} | skipped (inputs missing): {skipped} | {sum(len(v) for v in num.values.values())} numbers recorded in {out / 'numbers.json'}")


if __name__ == "__main__":
    main()
