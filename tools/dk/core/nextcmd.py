"""dk next: a deterministic work queue built from the last objdiff report.

1. Unfinished functions in declared (source-backed) units, in link order: finish what's open.
2. Otherwise, undeclared auto units as candidates to split next: units containing names
   dtk already recognised (library code with known source) first, then smallest first.
"""

from __future__ import annotations

import argparse
import json

from .command import OK, USAGE, Command, hint
from .config import Config
from .progress import load_report


def _addr(entry: dict) -> int:
    return int(entry.get("metadata", {}).get("virtual_address", 0) or 0)


def queue(report: dict) -> tuple[list[dict], list[dict]]:
    open_functions = []
    candidates = []
    for unit in report.get("units", []):
        meta = unit.get("metadata", {})
        functions = unit.get("functions", [])
        if meta.get("auto_generated"):
            if functions:
                named = sorted(f["name"] for f in functions if not f["name"].startswith(("fn_", "lbl_")))
                candidates.append({
                    "unit": unit["name"],
                    "address": min(_addr(f) for f in functions),
                    "code": int(unit.get("measures", {}).get("total_code", 0) or 0),
                    "functions": len(functions),
                    "known_names": named[:6],
                })
            continue
        for f in functions:
            pct = float(f.get("fuzzy_match_percent", 0.0))
            if pct < 100.0:
                open_functions.append({
                    "unit": unit["name"],
                    "source": meta.get("source_path", ""),
                    "function": f["name"],
                    "address": _addr(f),
                    "size": int(f.get("size", 0) or 0),
                    "fuzzy": pct,
                })
    open_functions.sort(key=lambda f: (f["address"], f["function"]))
    candidates.sort(key=lambda c: (not c["known_names"], c["code"], c["address"], c["unit"]))
    return open_functions, candidates


def run(cfg: Config, args: argparse.Namespace) -> int:
    report = load_report(cfg)
    if report is None:
        print("error: no objdiff report yet")
        hint("build")
        return USAGE
    open_functions, candidates = queue(report)
    if args.json:
        print(json.dumps({"open": open_functions[:args.n], "candidates": candidates[:args.n]}, indent=1))
        return OK
    if open_functions:
        print(f"{len(open_functions)} unfinished function(s) in declared units:")
        for f in open_functions[:args.n]:
            print(f"  {f['fuzzy']:6.2f}%  0x{f['address']:08X}  {f['function']:<44} 0x{f['size']:X}  {f['source']}")
        hint(f"ctx {open_functions[0]['function']}")
        return OK
    print(f"No open functions. {len(candidates)} undeclared auto unit(s); known names first, then smallest:")
    for c in candidates[:args.n]:
        names = ", ".join(c["known_names"])
        print(f"  0x{c['address']:08X}  {c['code']:>7} bytes  {c['functions']:>4} fn  {c['unit']}  {names}")
    hint("unit add <src path> --lib <lib> .text:0xSTART-0xEND")
    return OK


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("-n", type=int, default=15, help="how many entries to show (default 15)")
    p.add_argument("--json", action="store_true", help="machine-readable output")


COMMAND = Command("next", "matching", "what to work on: open functions first, else units to split", run, _configure)
