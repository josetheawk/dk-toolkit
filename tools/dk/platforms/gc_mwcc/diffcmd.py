"""dk diff <unit> [--function F]: side-by-side target vs compiled instructions, mismatches only.

Uses objdiff's one-shot JSON with the same strict relocation scoring as `dk check`, so what
it shows is exactly what keeps a function below 100%. Data sections below 100% get a chunk list.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess

from ...core.command import BUILD_ERROR, MISMATCH, OK, USAGE, Command, hint
from ...core.config import Config
from . import freshness as fresh
from .matching import MatchError, resolve_unit

MARK = {"DIFF_OP_MISMATCH": "op", "DIFF_ARG_MISMATCH": "arg", "DIFF_REPLACE": "rep",
        "DIFF_DELETE": "-", "DIFF_INSERT": "+"}


def _text(entry: dict) -> str:
    return entry.get("instruction", {}).get("formatted", "")


def function_rows(left: dict, right: dict | None, context: int) -> list[str]:
    li = left.get("instructions", [])
    ri = right.get("instructions", []) if right else []
    n = max(len(li), len(ri))
    diff = [bool((li[i].get("diff_kind") if i < len(li) else None) or (ri[i].get("diff_kind") if i < len(ri) else None))
            for i in range(n)]
    show = set()
    for i, d in enumerate(diff):
        if d:
            show.update(range(max(0, i - context), min(n, i + context + 1)))
    rows, last = [], -2
    for i in sorted(show):
        if i != last + 1:
            rows.append("    ...")
        l = li[i] if i < len(li) else {}
        r = ri[i] if i < len(ri) else {}
        kind = l.get("diff_kind") or r.get("diff_kind")
        addr = l.get("instruction", {}).get("address") or r.get("instruction", {}).get("address") or "0"
        rows.append(f"  {int(addr):>5X}  {_text(l):<42} {MARK.get(kind, '  '):>3}  {_text(r)}")
        last = i
    return rows


def data_rows(left_sec: dict, right_sec: dict) -> list[str]:
    rows = []
    for side, sec in (("target", left_sec), ("ours", right_sec)):
        off = 0
        for d in sec.get("data_diff", []):
            size = int(d.get("size", 0) or 0)
            raw = base64.b64decode(d["data"]) if d.get("data") else b""
            if d.get("kind"):
                rows.append(f"  {side:<6} +0x{off:<5X} {MARK.get(d['kind'], d['kind']):>3} 0x{size:X}  {raw[:16].hex()}{'…' if len(raw) > 16 else ''}")
            if d.get("data"):
                off += size
    return rows


def run(cfg: Config, args: argparse.Namespace) -> int:
    try:
        unit = resolve_unit(cfg, args.target)
    except MatchError as e:
        print(f"error: {e}")
        return USAGE
    build = fresh.build(cfg, [unit["base_path"]])
    if build.returncode != 0:
        print("BUILD ERROR:")
        print("\n".join(line for line in build.stdout.splitlines() if "Usage Warning" not in line)[-3000:])
        return BUILD_ERROR
    objdiff = cfg.root / "build" / "tools" / "objdiff-cli"
    proc = subprocess.run([str(objdiff), "diff", "-1", unit["target_path"], "-2", unit["base_path"], "-o", "-",
                           "--format", "json", "-c", "functionRelocDiffs=data_value"],
                          cwd=cfg.root, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"error: objdiff failed: {proc.stderr.strip()}")
        return USAGE
    doc = json.loads(proc.stdout)
    left, right = doc["left"], doc.get("right", {})
    rsyms = right.get("symbols", [])
    shown = 0
    for sym in left.get("symbols", []):
        if "instructions" not in sym:
            continue
        pct = float(sym.get("match_percent", 0.0))
        if args.function and sym["name"] != args.function:
            continue
        if pct == 100.0 and not args.function:
            continue
        # Pair by name; objdiff's target_symbol is a 1-based symbol-table index (0 = null symbol).
        other = next((r for r in rsyms if r["name"] == sym["name"] and "instructions" in r), None)
        idx = sym.get("target_symbol")
        if other is None and isinstance(idx, int) and 0 < idx <= len(rsyms):
            other = rsyms[idx - 1]
        print(f"{sym['name']}  {pct:.2f}%" + ("" if other else "  (missing from compiled object)"))
        print(f"  {'addr':>5}  {'target':<42} {'':>3}  ours")
        for row in function_rows(sym, other, args.context):
            print(row)
        shown += 1
    rsecs = {s["name"]: s for s in right.get("sections", [])}
    for sec in left.get("sections", []):
        if sec.get("kind") in ("SECTION_DATA", "SECTION_BSS") and float(sec.get("match_percent", 0.0)) < 100.0 and not args.function:
            print(f"{sec['name']}  {float(sec.get('match_percent', 0.0)):.2f}% (raw objdiff, before dk's weak-copy allowances)")
            for row in data_rows(sec, rsecs.get(sec["name"], {})):
                print(row)
            shown += 1
    if not shown:
        print("no differences")
        hint(f"check {args.target}")
        return OK
    hint(f"check {args.target}")
    return MISMATCH


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("target", help="unit (source path or name) or a function inside one")
    p.add_argument("--function", help="show only this function (even if exact)")
    p.add_argument("--context", type=int, default=2, help="matching lines shown around each difference (default 2)")


COMMAND = Command("diff", "matching", "side-by-side target vs compiled instructions (mismatches + context)", run, _configure)
