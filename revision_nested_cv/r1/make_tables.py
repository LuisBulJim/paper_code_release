"""Generate every LaTeX table of the manuscript from r1/results, r1/data and r1/configs/hparams.yaml.

No number is typed by hand: each table is written to r1/tables/<name>.tex and every value that appears in a
table is also recorded in r1/tables/numbers.json (used by check_consistency.py to verify the manuscript).
Tables whose inputs do not exist yet are skipped with a log line, so the script can run at any stage.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from r1.protocol import get_logger, load_hparams, load_protocol, read_json, resolve

MODEL_LABEL = {"yolo26n": "YOLO26-n", "yolo26m": "YOLO26-m", "yolo26l": "YOLO26-l"}
TAG_LABEL = {"direct": "Direct transfer", "sequential": "Sequential transfer"}
VARIANT_LABEL = {"grouped": "Grouped by video", "ungrouped": "Not grouped"}
NESTED = "repeated nested cross-validation (3 repeats $\\times$ 5 outer folds grouped by video; each image predicted by the fold model that had it in its outer test fold)"
STRATEGY_LABEL = {"standard": "Standard", "tta": "TTA", "wbf": "WBF ensemble"}


def model_label(m: str) -> str:
    return " + ".join(MODEL_LABEL.get(x, x) for x in m.split("+"))


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


def comparison_label(comp: str) -> str:
    out = comp
    for m in sorted(MODEL_LABEL, key=len, reverse=True):
        out = out.replace(m, MODEL_LABEL[m])
    return out.replace("+", " + ").replace("direct", "Direct").replace("sequential", "Sequential").replace("tta", "TTA").replace("wbf", "WBF").replace(" - ", " $-$ ")


def latex_table(header: list[str], rows: list[list[str]], caption: str, label: str, align: str | None = None, text_cols: int = 1) -> str:
    align = align or "l" * text_cols + "r" * (len(header) - text_cols)
    wide = len(header) >= 6  # interval columns overflow a text column; graphicx is loaded by the manuscript
    tab = [f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    tab += [" & ".join(r) + " \\\\" for r in rows]
    tab += ["\\bottomrule", "\\end{tabular}"]
    if wide:
        tab = ["\\resizebox{\\textwidth}{!}{%"] + tab + ["}"]
    lines = ["\\begin{table}[htbp]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}"] + tab + ["\\end{table}", ""]
    return "\n".join(lines)


def ci(num: Numbers, table: str, row: dict, key: str) -> str:
    # t-intervals over few seeds/checkpoints can leave [0, 1]; every metric shown with ci() is bounded there
    lo, hi = max(0.0, float(row[f"{key}_ci_lo"])), min(1.0, float(row[f"{key}_ci_hi"]))
    return f"{num.add(table, row[f'{key}_mean'])} [{num.add(table, lo)}, {num.add(table, hi)}]"


def hparam_value(num: Numbers, v) -> str:
    if isinstance(v, (str, bool)) or v is None:
        return str(v)
    s = num.add("hparams", v)
    return s if float(s) == float(v) else f"{float(v):g}"  # 3 decimals would print weight_decay 0.0005 as 0.001


def table_hparams(hp: dict, num: Numbers) -> str:
    rows = [[tex(k), hparam_value(num, v)] for k, v in hp.items()]
    return latex_table(["Hyperparameter", "Value"], rows, "Training hyperparameters shared by every run (Direct Transfer, Sequential Transfer fine-tuning and TrashCan), as passed to Ultralytics.", "tab:hparams", "ll")


def table_dataset(manifest: dict, groups: list[dict], num: Numbers) -> str:
    s = manifest["summary"]
    rows = [
        ["Images", num.add("dataset", s["images"])],
        ["Images with corrosion / background", f"{num.add('dataset', s['positive'])} / {num.add('dataset', s['negative'])}"],
        ["Ground-truth boxes", num.add("dataset", s["boxes"])],
        ["Boxes small / medium / large", " / ".join(num.add("dataset", s["size_buckets"].get(b, 0)) for b in ("small", "medium", "large"))],
    ]
    if groups:
        rows.append(["Video groups (cluster bootstrap units)", num.add("dataset", len({g["group"] for g in groups}))])
    return latex_table(["LIACI corrosion subset", "Value"], rows, "Dataset used by the nested cross-validation. Box sizes at 640$\\times$640 (COCO buckets: small $<32^2$, medium $<96^2$ px$^2$, large otherwise).", "tab:dataset", "lr")


def strategy_label(r: dict) -> str:
    return STRATEGY_LABEL.get(r["strategy"], r["strategy"])


def table_nested(summary: list[dict], num: Numbers) -> str:
    rows = [[TAG_LABEL.get(r["tag"], r["tag"]), model_label(r["model"]), strategy_label(r), ci(num, "nested", r, "map50"), ci(num, "nested", r, "map5095"), ci(num, "nested", r, "precision"), ci(num, "nested", r, "recall"), ci(num, "nested", r, "f1")] for r in summary if r["variant"] == "grouped"]
    return latex_table(["Protocol", "Model", "Strategy", "mAP@50", "mAP@[.5:.95]", "Precision", "Recall", "F1"], rows, f"Out-of-fold detection results of the {NESTED}: mean over the 3 repeats of the metric on all 261 images, with 95\\% cluster-bootstrap interval (video groups resampled). P/R/F1 at the fixed operating point (conf 0.25, IoU 0.5). TTA and WBF configurations were chosen on the inner validation splits only.", "tab:nested", text_cols=3)


def table_nested_image(summary: list[dict], num: Numbers) -> str:
    rows = [[TAG_LABEL.get(r["tag"], r["tag"]), model_label(r["model"]), strategy_label(r), ci(num, "nested_image", r, "auroc"), ci(num, "nested_image", r, "sensitivity"), ci(num, "nested_image", r, "specificity")] for r in summary if r["variant"] == "grouped"]
    return latex_table(["Protocol", "Model", "Strategy", "AUROC", "Sensitivity", "Specificity"], rows, "Image-level screening (does the image contain corrosion?) from the same out-of-fold predictions: image score = highest box confidence; sensitivity and specificity at conf 0.25. Mean over repeats with 95\\% cluster-bootstrap interval.", "tab:nested_image", text_cols=3)


def table_nested_threshold(summary: list[dict], num: Numbers) -> str:
    rows = [[TAG_LABEL.get(r["tag"], r["tag"]), model_label(r["model"]), strategy_label(r), num.add("nested_threshold", r["conf_vt"], "{:.2f}"), ci(num, "nested_threshold", r, "precision_vt"), ci(num, "nested_threshold", r, "recall_vt"), ci(num, "nested_threshold", r, "f1_vt"), ci(num, "nested_threshold", r, "f1")] for r in summary if r["variant"] == "grouped"]
    return latex_table(["Protocol", "Model", "Strategy", "Threshold", "Precision", "Recall", "F1", "F1 at 0.25"], rows, "Operating point chosen on the inner validation splits for each strategy (confidence with the highest mean validation F1), applied to the out-of-fold predictions; the last column repeats F1 at the a-priori threshold 0.25. WBF and TTA average the confidences of the fused boxes, which shifts their useful threshold downwards. Mean over repeats with 95\\% cluster-bootstrap interval.", "tab:nested_threshold", text_cols=3)


def table_nested_deltas(deltas: list[dict], num: Numbers) -> str:
    def d(r, k):
        return f"{num.add('nested_deltas', r[f'{k}_delta'])} [{num.add('nested_deltas', r[f'{k}_ci_lo'])}, {num.add('nested_deltas', r[f'{k}_ci_hi'])}]"

    rows = [[comparison_label(r["comparison"]), d(r, "map50"), d(r, "map5095"), d(r, "f1"), d(r, "f1_vt"), num.add("nested_deltas", r["map50_p_le0"])] for r in deltas if r["variant"] == "grouped"]
    return latex_table(["Comparison", "$\\Delta$ mAP@50", "$\\Delta$ mAP@[.5:.95]", "$\\Delta$ F1 (0.25)", "$\\Delta$ F1 (val. threshold)", "$P(\\Delta \\mathrm{mAP@50} \\le 0)$"], rows, "Paired differences on the out-of-fold predictions (mean over repeats, 95\\% cluster-bootstrap interval with the same resampled video groups for both strategies, and bootstrap probability that the mAP@50 difference is not positive). F1 at the a-priori threshold 0.25 and at the threshold chosen on inner validation for each strategy.", "tab:nested_deltas")


def table_ablation(summary: list[dict], num: Numbers) -> str:
    rows = []
    for r in summary:
        if r["strategy"] != "standard":
            continue
        rows.append([VARIANT_LABEL.get(r["variant"], r["variant"]), TAG_LABEL.get(r["tag"], r["tag"]), model_label(r["model"]), ci(num, "ablation", r, "map50"), ci(num, "ablation", r, "f1"), ci(num, "ablation", r, "auroc")])
    return latex_table(["Outer folds", "Protocol", "Model", "mAP@50", "F1", "AUROC"], rows, "Effect of grouping the outer folds by inspection video: the same nested protocol with outer folds that ignore the video (adjacent frames may fall on both sides) against folds grouped by video. Standard inference, mean over repeats with 95\\% cluster-bootstrap interval.", "tab:ablation", text_cols=3)


def table_latency(rows_in: list[dict], num: Numbers) -> str:
    rows = [[STRATEGY_LABEL.get(r["strategy"], r["strategy"]), r["precision"].upper(), num.add("latency", r["ms_mean"], "{:.1f}"), num.add("latency", r["ms_p95"], "{:.1f}"), num.add("latency", r["img_per_s"], "{:.1f}"), num.add("latency", r["peak_vram_mb"], "{:.0f}"), r["forward_passes_per_image"]] for r in rows_in]
    return latex_table(["Strategy", "Precision", "Latency (ms)", "p95 (ms)", "img/s", "Peak VRAM (MB)", "Passes"], rows, f"Inference cost measured on {rows_in[0]['device']}: batch 1, 640 px, 20 warm-up iterations, {rows_in[0]['n_timed']} timed inferences per row, CUDA synchronised; latency includes pre/post-processing and, for TTA and WBF, the box fusion. Peak VRAM is the memory allocated by PyTorch (CUDA context excluded). At batch 1 the runtime is dominated by pre/post-processing, so FP16 brings no speed-up for these model sizes.", "tab:latency", text_cols=2)


def table_trashcan(summary: list[dict], deltas: list[dict], num: Numbers) -> str:
    rows = [[STRATEGY_LABEL.get(r["strategy"], r["strategy"]), r["class"], r["n_seeds"], ci(num, "trashcan", r, "map50"), ci(num, "trashcan", r, "map5095"), ci(num, "trashcan", r, "precision"), ci(num, "trashcan", r, "recall"), ci(num, "trashcan", r, "f1")] for r in summary]
    out = latex_table(["Strategy", "Class", "Seeds", "mAP@50", "mAP@[.5:.95]", "Precision", "Recall", "F1"], rows, "TrashCan 1.0 (material version, Metal and Bio classes), official validation split, not used for training or checkpoint selection: mean over 3 training seeds with 95\\% t-interval, same hyperparameters and evaluator as LIACI. P/R/F1 at conf 0.25, IoU 0.5.", "tab:trashcan", text_cols=2)
    if not deltas:
        return out

    def d(r, k):
        return f"{num.add('trashcan_deltas', r[f'{k}_delta'])} [{num.add('trashcan_deltas', r[f'{k}_ci_lo'])}, {num.add('trashcan_deltas', r[f'{k}_ci_hi'])}]"

    rows = []
    for r in deltas:
        per_seed = r["seed"] != "all"
        rows.append([f"Seed {r['seed']}" if per_seed else "Mean over seeds", d(r, "map50"), d(r, "f1"), d(r, "precision"), d(r, "recall"), num.add("trashcan_deltas", r["map50_p_le0"]) if per_seed else "--"])
    out += latex_table(["", "$\\Delta$ mAP@50", "$\\Delta$ F1", "$\\Delta$ Precision", "$\\Delta$ Recall", "$P(\\Delta \\mathrm{mAP@50} \\le 0)$"], rows, "TrashCan: paired difference TTA $-$ standard. Per seed: paired image bootstrap with 95\\% interval and bootstrap probability that the mAP@50 difference is not positive. Last row: mean of the per-seed differences with 95\\% t-interval across the 3 seeds.", "tab:trashcan_deltas")
    return out


def table_trashcan_ood(summary: list[dict], num: Numbers) -> str:
    rows = [[MODEL_LABEL.get(r["model"], r["model"]), STRATEGY_LABEL.get(r["strategy"], r["strategy"]), r["n_checkpoints"], ci(num, "trashcan_ood", r, "map50"), ci(num, "trashcan_ood", r, "precision"), ci(num, "trashcan_ood", r, "recall"), ci(num, "trashcan_ood", r, "f1")] for r in summary]
    return latex_table(["LIACI model", "Strategy", "Ckpts", "mAP@50", "Precision", "Recall", "F1"], rows, "Zero-shot out-of-distribution test: the 15 nested-CV fold models of each LIACI corrosion detector (3 repeats $\\times$ 5 outer folds), without retraining, on the TrashCan validation images containing metal debris (corrosion predictions scored against the Metal boxes); mean with 95\\% t-interval over fold models.", "tab:trashcan_ood")


SAM3_ROWS = [
    ("positives", "share_detected", "Images with corrosion that get a detector box"),
    ("positives", "iou_mean", "Mean IoU, images with corrosion (undetected = 0)"),
    ("positives", "dice_mean", "Mean Dice, images with corrosion (undetected = 0)"),
    ("positives", "iou_median", "Median IoU, images with corrosion"),
    ("positives", "share_iou_gt_0.5", "Images with corrosion and IoU $>$ 0.5"),
    ("positives_detected", "iou_mean", "Mean IoU, detected images with corrosion only"),
    ("negatives", "false_mask_rate", "Background images with a false mask"),
    ("negatives", "mean_false_area_fraction", "Mean false-mask area (fraction of the image)"),
]


def table_sam3(conditions: list[tuple[str, list[dict]]], num: Numbers) -> str:
    def cell(summary: list[dict], subset: str, metric: str) -> tuple[str, str]:
        r = next((r for r in summary if (r["subset"], r["metric"]) == (subset, metric)), None)
        if r is None:
            return "--", "--"
        value = num.add("sam3", r["value"]) + (f" [{num.add('sam3', r['ci_lo'])}, {num.add('sam3', r['ci_hi'])}]" if r["ci_lo"] else "")
        return value, r["n"]

    rows = []
    for subset, metric, label in SAM3_ROWS:
        cells = [cell(s, subset, metric) for _, s in conditions]
        ns = sorted({n for _, n in cells if n != "--"}, key=int)
        rows.append([label, " / ".join(ns)] + [v for v, _ in cells])
    s0 = conditions[0][1]
    n_pos = next(r["n"] for r in s0 if r["subset"] == "positives")
    n_neg = next(r["n"] for r in s0 if r["subset"] == "negatives")
    return latex_table(
        ["Metric", "N"] + [f"Detector conf {c}" for c, _ in conditions],
        rows,
        f"SAM 3 ({tex(s0[0]['sam_model'])}) prompted with the boxes of the Direct-transfer YOLO26-m detector on the LIACI segmentation set "
        f"({n_pos} images with corrosion, {n_neg} background). Images that belong to the nested cross-validation are detected by the repeat-0 fold model that had them "
        "in its outer test fold, the rest by a fold model that never saw them. Columns: the a-priori threshold 0.25 and the threshold chosen on inner validation. "
        "An undetected image with corrosion counts as IoU 0; 95\\% bootstrap interval over images in brackets.",
        "tab:sam3",
    )


def table_sweep(summary: list[dict], conf: float, num: Numbers) -> str:
    rows = []
    for r in sorted((r for r in summary if float(r["conf"]) == conf and r["group"] == "tta"), key=lambda r: (r["model"], -float(r["map50_mean"]))):
        rows.append([MODEL_LABEL.get(r["model"], r["model"]), tex(r["config"]), r["wbf_iou"] or "--", f"{num.add('sweep', r['map50_mean'])} $\\pm$ {num.add('sweep', r['map50_std'])}", f"{num.add('sweep', r['f1_mean'])} $\\pm$ {num.add('sweep', r['f1_std'])}", f"{num.add('sweep', r['recall_mean'])}"])
    return latex_table(["Model", "TTA configuration", "WBF IoU", "mAP@50 (val)", "F1 (val)", "Recall (val)"], rows, f"Sensitivity of TTA to its settings on the inner validation splits of the nested cross-validation (mean $\\pm$ SD over 3 repeats $\\times$ 5 outer folds, operating point {conf}). The configuration applied to the outer test folds is the best mean validation mAP@50 averaged over model sizes.", "tab:sweep_tta")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="r1/results")
    ap.add_argument("--out", default="r1/tables")
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
        with open(out / f"{name}.tex", "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        written.append(name)

    emit("hparams", table_hparams(load_hparams(), num))
    manifest_path = resolve(proto["dataset"]["manifest_file"])
    emit("dataset", table_dataset(read_json(manifest_path), read_csv(res / "nested_groups.csv"), num) if manifest_path.exists() else None)
    summary, deltas = read_csv(res / "nested_summary.csv"), read_csv(res / "nested_deltas.csv")
    emit("nested", table_nested(summary, num) if summary else None)
    emit("nested_image", table_nested_image(summary, num) if summary else None)
    emit("nested_threshold", table_nested_threshold(summary, num) if summary and "conf_vt" in summary[0] else None)
    emit("nested_deltas", table_nested_deltas(deltas, num) if deltas else None)
    emit("ablation", table_ablation(summary, num) if {r["variant"] for r in summary} >= {"grouped", "ungrouped"} else None)
    latency = read_csv(res / "latency.csv")
    emit("latency", table_latency(latency, num) if latency else None)
    tc_summary, tc_deltas = read_csv(res / "trashcan_summary.csv"), read_csv(res / "trashcan_deltas.csv")
    emit("trashcan", table_trashcan(tc_summary, tc_deltas, num) if tc_summary else None)
    ood = read_csv(res / "trashcan_ood_summary.csv")
    emit("trashcan_ood", table_trashcan_ood(ood, num) if ood else None)
    sam = read_csv(res / "sam3_summary.csv")
    sam_vt = read_csv(res / "sam3_valthr" / "sam3_summary.csv")
    choice_path = res / "inference_choice.json"
    vt = read_json(choice_path).get("operating_threshold_on_validation", {}).get("direct/yolo26m/standard") if choice_path.exists() else None
    conditions = [("0.25 (a priori)", sam)] + ([(f"{vt} (validation)", sam_vt)] if sam_vt and vt is not None else [])
    emit("sam3", table_sam3(conditions, num) if sam else None)
    sweep = read_csv(res / "sweep_tta_val_summary.csv")
    emit("sweep_tta", table_sweep(sweep, proto["evaluation"]["conf_fixed"], num) if sweep else None)

    (out / "numbers.json").write_text(json.dumps(num.values, indent=1), encoding="utf-8")
    log.info(f"written: {written} | skipped (inputs missing): {skipped} | {sum(len(v) for v in num.values.values())} numbers recorded in {out / 'numbers.json'}")


if __name__ == "__main__":
    main()
