"""Writes the LaTeX tables of the second revision from the CSVs of the analysis scripts (no number is typed by hand).

  tex/r2_2_per_fold.tex       R2.2  choices of the nested selection procedure in each outer fold
  tex/r2_2_summary.tex        R2.2  fixed recipe vs. selection procedure
  tex/r2_1c_supercluster.tex  R2.1(c) paired differences under coarser bootstrap units
  tex/r2_4_tiled.tex          R2.4  filled-box reference vs. tilings
  tex/r2_1a_purge.tex         R2.1(a) purged CV (buffer B) for direct-transfer Medium

Copies are written to paper/tex/ for \\input in the supplementary material.

Usage (from the repository root): python results_rev2/make_tex_tables.py
"""
from __future__ import annotations

import csv
from pathlib import Path

R = Path(__file__).resolve().parent
OUT = R / "tex"


def rows(p: str) -> list[dict[str, str]]:
    return list(csv.DictReader(open(R / p, encoding="utf-8")))


def f3(x: str | float) -> str:
    return f"{float(x):.3f}"


def sg(x: str | float) -> str:
    return f"${float(x):+.3f}$"


def ci(lo: str | float, hi: str | float, signed: bool = False) -> str:
    g = sg if signed else f3
    return f"[{g(lo)}, {g(hi)}]"


RECIPE = {"x0_baseline": "Baseline", "x1_recipe": "Small-data recipe", "x2_multi": "Nine classes", "x3_seg": "Segmentation",
          "x4_seg_cp": "Seg.\\ + copy-paste", "x6_multi_seg": "Nine-class polygons", "e1_ensemble": "Ens.\\ of variants",
          "c4_baselines": "Ens.\\ of baselines"}


def tex_name(s: str) -> str:
    return s.replace("_", "\\_")


def per_fold() -> None:
    lines = [r"\begin{tabular}{@{} c c r r l l l c c c @{}}", r"\toprule",
             r"\textbf{Rep.} & \textbf{Fold} & \textbf{Val.} & \textbf{Val.\ pos.} & \textbf{TTA config.} & \textbf{WBF config.} & "
             r"\textbf{Recipe (pass)} & \textbf{Thr.} & \textbf{Test, choice} & \textbf{Test, fixed} \\", r"\midrule"]
    for r in rows("r2_2/per_fold_choices.csv"):
        rec, ps = r["recipe"].split("/")
        lines.append(f"{r['repeat']} & {r['fold']} & {r['n_val']} & {r['n_val_pos']} & \\texttt{{{tex_name(r['tta'])}}} & "
                     f"\\texttt{{{tex_name(r['wbf'])}}} & {RECIPE.get(rec, tex_name(rec))} ({'TTA' if ps == 'tta' else 'single'}) & "
                     f"{r['conf']} & {f3(r['test_map50_of_choice'])} & {f3(r['test_map50_fixed_recipe'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "r2_2_per_fold.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summary() -> None:
    lab = {"baseline": "Baseline Medium, single pass", "fixed": "Fixed recipe (control ensemble + TTA), settings pooled over 15 validations",
           "c4_nested": "Same recipe, TTA/WBF/threshold chosen in each outer fold",
           "nested": "Selection procedure: TTA, WBF, recipe and threshold chosen in each outer fold"}
    lines = [r"\begin{tabular}{@{} l c c c @{}}", r"\toprule",
             r"\textbf{Procedure} & \textbf{OOF mAP@50 [95\% CI]} & \textbf{Mean per-fold mAP@50} & \textbf{F1 (per-fold thr.)} \\", r"\midrule"]
    for r in rows("r2_2/summary.csv"):
        lines.append(f"{lab[r['procedure']]} & {f3(r['map50_mean'])} {ci(r['map50_ci_lo'], r['map50_ci_hi'])} & "
                     f"{f3(r['map50_fold_mean'])} & {f3(r['f1_mean'])} \\\\")
    lines += [r"\midrule", r"\multicolumn{4}{@{}l}{\textit{Paired differences in OOF mAP@50}} \\"]
    dl = {"nested - fixed": "Procedure $-$ fixed recipe", "c4_nested - fixed": "Per-fold settings $-$ fixed recipe",
          "nested - baseline": "Procedure $-$ baseline", "fixed - baseline": "Fixed recipe $-$ baseline"}
    for r in rows("r2_2/deltas.csv"):
        if r["comparison"] in dl:
            lines.append(f"{dl[r['comparison']]} & {sg(r['map50_delta'])} {ci(r['map50_ci_lo'], r['map50_ci_hi'], True)} & & \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "r2_2_summary.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def supercluster() -> None:
    g = {r["grouping"]: r for r in rows("r2_1c/groupings.csv")}
    keys = ["g28", "gap30", "gap40", "gap45"]
    head = " & ".join(f"\\textbf{{{g[k]['n_groups']} groups}}" for k in keys)
    lines = [r"\begin{tabular}{@{} l c " + "c " * len(keys) + "@{}}", r"\toprule",
             r"\textbf{Comparison (mAP@50)} & $\boldsymbol{\Delta}$ & " + head + r" \\", r"\midrule"]
    for r in rows("r2_1c/deltas.csv"):
        cells = [ci(r[f"map50_{k}_lo"], r[f"map50_{k}_hi"], True) for k in keys]
        lines.append(f"{tex_name(r['comparison']).replace(' - ', ' $-$ ')} & {sg(r['map50_delta'])} & " + " & ".join(cells) + r" \\")
    lines += [r"\midrule"]
    w = {(r["grouping"], r["kind"], r["metric"]): r for r in rows("r2_1c/width_ratio.csv")}
    for kind, lab in (("interval", "Median width ratio, intervals of each strategy"), ("delta", "Median width ratio, paired differences")):
        lines.append(f"{lab} & & 1 & " + " & ".join(f"$\\times${float(w[(k, kind, 'map50')]['median_width_ratio_vs_g28']):.2f}" for k in keys[1:]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "r2_1c_supercluster.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def tiled() -> None:
    lines = [r"\begin{tabular}{@{} l c r @{}}", r"\toprule", r"\textbf{Box output built from the ground truth} & \textbf{Pixel IoU} & \textbf{Boxes} \\", r"\midrule"]
    for r in rows("r2_4/tiled_gt.csv"):
        name = "One filled box per annotated connected component" if not r["cell_px"] else f"Grid cells of {r['cell_px']} px covering the ground truth"
        lines.append(f"{name} & {f3(r['iou'])} & {int(r['boxes_total']):,} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "r2_4_tiled.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def purge() -> None:
    import json
    import re
    summ = {r["setting"]: r for r in rows("r2_1/purge_summary.csv")}
    dl = {r["comparison"]: r for r in rows("r2_1/purge_deltas.csv")}
    loss = {x["setting"]: x for x in json.load(open(R / "r2_1/purge_numbers.json"))["losses"]}

    def c(s: str, signed: bool = False) -> str:
        lo, hi = (float(v) for v in re.findall(r"-?[0-9.]+", s))
        return ci(lo, hi, signed)
    lines = [r"\begin{tabular}{@{} l c c c c c c @{}}", r"\toprule",
             r"\textbf{Buffer} & \textbf{Lost train+val images per fold} & \textbf{Single pass} & $\Delta$\textbf{ vs.\ $B=0$} & "
             r"\textbf{TTA} & $\Delta$\textbf{ vs.\ $B=0$} & $\Delta$\textbf{TTA} \\", r"\midrule"]
    for key, lab in (("B0 (paper)", "$B=0$ (paper)"), ("B5", "$B=5$"), ("B15", "$B=15$"), ("B30", "$B=30$")):
        r = summ[key]
        if key in loss:
            l = loss[key]
            lost = f"{l['lost_train_val_min']}--{l['lost_max']} (median {l['lost_median']:g})"
            ds = dl[f"{key} - B0 (standard)"]
            dt = dl[f"{key} - B0 (tta)"]
            d1 = f"{sg(ds['delta'])} {c(ds['ci'], True)}"
            d2 = f"{sg(dt['delta'])} {c(dt['ci'], True)}"
        else:
            lost, d1, d2 = "0", "--", "--"
        lines.append(f"{lab} & {lost} & {f3(r['map50_standard'])} {c(r['ci_standard'])} & {d1} & {f3(r['map50_tta'])} {c(r['ci_tta'])} & {d2} & "
                     f"{sg(r['delta_tta'])} {c(r['ci_delta_tta'], True)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "r2_1a_purge.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    per_fold(); summary(); supercluster(); tiled(); purge()
    paper_tex = R.parent / "paper" / "tex"  # \input{tex/...} in supplementary_material.tex
    paper_tex.mkdir(exist_ok=True)
    for p in sorted(OUT.glob("*.tex")):
        (paper_tex / p.name).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"== {p.name}\n{p.read_text()}")
