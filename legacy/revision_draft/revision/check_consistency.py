"""Check that every number written in the manuscript .tex exists in the generated results.

Sources of truth: tables/numbers.json (written by make_tables.py) plus every numeric cell of results/*.csv
and the dataset manifests. A number in the text matches when some source value rounds to it at the
precision written in the text (0.82 matches 0.8196, 0.820 does not). Numbers inside \\cite, \\ref, \\label,
years, equation counters, figure dimensions and \\input lines are ignored. Exit code 1 if orphans remain,
listing each with its line and context, and if a table referenced by \\input is missing.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from revision.protocol import get_logger, resolve

IGNORE_INLINE = re.compile(r"\\(cite[tp]?|ref|label|eqref|input|includegraphics|bibitem|url|href)\*?(\[[^\]]*\])?\{[^}]*\}")
NUMBER = re.compile(r"(?<![\w.\-])(\d+(?:\.\d+)?)(?:\s*\\%|%)?(?![\w.])")
YEAR = re.compile(r"^(19|20)\d\d$")


def source_values(results: Path, tables: Path, extra_json: list[Path]) -> set[float]:
    values: set[float] = set()
    numbers = tables / "numbers.json"
    if numbers.exists():
        for vs in json.loads(numbers.read_text(encoding="utf-8")).values():
            values.update(float(v) for v in vs)
    for csv_path in results.glob("*.csv"):
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.reader(f):
                for cell in row:
                    try:
                        values.add(float(cell))
                    except ValueError:
                        pass

    def walk(o):
        if isinstance(o, bool):
            return
        if isinstance(o, (int, float)):
            values.add(float(o))
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    for j in extra_json:
        if j.exists():
            walk(json.loads(j.read_text(encoding="utf-8")))
    return values


def matches(text_value: str, values: set[float]) -> bool:
    """True when some source value, rounded to the precision written in the text, equals the text value;
    a fraction stored as 0.8196 also matches the percentage 82 or 81.96."""
    decimals = len(text_value.split(".")[1]) if "." in text_value else 0
    target = round(float(text_value), decimals)
    for v in values:
        if round(v, decimals) == target or round(v * 100, decimals) == target:
            return True
    return False


def check(tex: Path, values: set[float], tables: Path, min_int: int) -> tuple[list[dict], list[str]]:
    orphans, missing_inputs = [], []
    in_comment_block = False
    for lineno, raw in enumerate(tex.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("%")[0] if not raw.lstrip().startswith("%") else ""
        if "\\begin{comment}" in raw:
            in_comment_block = True
        if "\\end{comment}" in raw:
            in_comment_block = False
            continue
        if in_comment_block or not line.strip():
            continue
        for m in re.finditer(r"\\input\{([^}]*)\}", line):
            target = Path(m.group(1))
            if "tables" in target.parts and not (tables / (target.name if target.suffix else target.name + ".tex")).exists():
                missing_inputs.append(m.group(1))
        cleaned = IGNORE_INLINE.sub(" ", line)
        cleaned = re.sub(r"\\(section|subsection|subsubsection|caption|label)\*?\{", " ", cleaned)
        for m in NUMBER.finditer(cleaned):
            tok = m.group(1)
            if YEAR.match(tok) or ("." not in tok and int(tok) < min_int):
                continue
            if "." not in tok and re.search(r"(cm|mm|pt|px|em|\\times)", cleaned[m.end() : m.end() + 8]):
                continue
            if not matches(tok, values):
                orphans.append({"line": lineno, "value": tok, "context": cleaned.strip()[:120]})
    return orphans, missing_inputs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tex", required=True, help="manuscript .tex")
    ap.add_argument("--results", default="results")
    ap.add_argument("--tables", default="tables")
    ap.add_argument("--min-int", type=int, default=10, help="integers below this are not checked (list markers, small counts)")
    ap.add_argument("--json", default=None, help="write the report here")
    args = ap.parse_args()
    log = get_logger("check_consistency")
    extra = [resolve("data/manifest_v2.json"), resolve("data/trashcan_manifest.json"), resolve("data/sam3_manifest.json"), resolve("results/dataset_stats.json")]
    values = source_values(resolve(args.results), resolve(args.tables), extra)
    orphans, missing = check(Path(args.tex), values, resolve(args.tables), args.min_int)
    for o in orphans:
        print(f"line {o['line']:5d}  {o['value']:>10}  | {o['context']}")
    for m in missing:
        print(f"missing table input: {m}")
    log.info(f"{len(values)} source values | {len(orphans)} orphan numbers | {len(missing)} missing table inputs")
    if args.json:
        Path(args.json).write_text(json.dumps({"orphans": orphans, "missing_inputs": missing, "n_source_values": len(values)}, indent=1), encoding="utf-8")
    sys.exit(1 if orphans or missing else 0)


if __name__ == "__main__":
    main()
