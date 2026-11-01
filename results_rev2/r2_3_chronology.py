"""R2.3 -- chronology of the improvement variants, from the archive timestamps of the packages that carried them.

git does not track revision_nested_cv/ (the repository has three commits from 2025), so the evidence is the member
dates inside the zip packages (local time of the machine that built them) and the mtimes of the training logs (UTC).

Usage (from the repository root): python results_rev2/r2_3_chronology.py
"""
import datetime as dt
import json
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NCV = REPO / "revision_nested_cv"
PACKAGES = {"ext_paquete.zip": "first round (spec + failure analysis)", "ext_cierre.zip": "second round",
            "ext_final.zip": "final README"}
LOGS = ["out/preds", "ext/out/logs/train_x1_recipe.log", "ext/out/logs/train_x2_multi.log", "ext/out/logs/closing.log"]


def main() -> None:
    out = {"zip_members_local_time": {}, "mtime_utc": {}}
    for z, what in PACKAGES.items():
        with zipfile.ZipFile(NCV / z) as zf:
            out["zip_members_local_time"][z] = {"role": what, "members": {
                i.filename: dt.datetime(*i.date_time).isoformat(" ") for i in zf.infolist()
                if any(k in i.filename for k in ("README_ext", "train_ext", "diagnostics", "build_seg_multi", "ensemble_eval"))}}
    for p in LOGS:
        out["mtime_utc"][p] = dt.datetime.utcfromtimestamp((NCV / p).stat().st_mtime).isoformat(" ", "seconds")
    print(json.dumps(out, indent=1))
    (REPO / "results_rev2/r2_3/chronology.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
