"""Helpers that apply the inference choice made on validation (r1/results/inference_choice.json, written by
nested_eval.py with the fixed rule of sweep_inference.py) to new data: TrashCan, SAM 3 and the latency benchmark."""

from __future__ import annotations

import numpy as np

from r1 import predict_dump as pdump
from r1.collect_runs import t_interval
from r1.protocol import read_json, resolve
from r1.sweep_inference import TTA_GRID

METRIC_KEYS = ("map50", "map5095", "precision", "recall", "f1")
DEFAULT_CHOICE = "r1/results/inference_choice.json"


def tta_protocol(proto: dict, config: str, wbf_iou: float | None) -> dict:
    if config not in TTA_GRID or TTA_GRID[config] is None:
        raise SystemExit(f"TTA configuration must be one of {[k for k, v in TTA_GRID.items() if v]}, got {config!r}")
    return pdump.tta_protocol(proto, TTA_GRID[config], wbf_iou)


def load_choice(path: str | None, required: bool) -> dict:
    p = resolve(path or DEFAULT_CHOICE)
    if not p.exists():
        if required:
            raise SystemExit(f"{p} missing: run python -m r1.nested_eval evaluate first, or pass --tta-config explicitly")
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
