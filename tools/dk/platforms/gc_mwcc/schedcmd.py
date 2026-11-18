"""`dk sched UNIT FN`: why an instruction-order difference happens, from the compiler's own scheduler records.

Compiles the unit's current source under the compiler-oracle runner with `--sched` (schema sched.v1: every
scheduled block's codegen order, dependence DAG and scheduled order; schedmodel.py), lines our rows up with the
records, and for every SCHEDULE region (residue.py) prints the first pair issued in the other order, the rule
that decided it (codegen order, deadline, uncovering, height, rank, not ready) and the codegen-order levers
that give the target's order on the simulator. `--check` re-simulates every block of the unit against the
compiler's order (the model's self-test).
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from ...core.command import BUILD_ERROR, MISMATCH, OK, USAGE, Command, hint
from ...core.config import Config
from . import objcache, residue, schedmodel
from .matching import MatchError, resolve_unit


def sched_wrap(command: list[str], out: Path, symbol: str | None):
    def go(argv: list[str]) -> list[str]:
        from .oracle import oracle_name
        funcs = ["--funcs", oracle_name(symbol)] if symbol else []
        return list(command) + [str(out)] + funcs + ["--sched", "--"] + list(argv)
    return go


def records_for(cfg: Config, unit: dict, text: str, symbol: str | None, work: Path):
    """(sched.v1 document, our object) for `text` compiled as the unit, or (None, error text)."""
    from . import oracle
    from .reorder import compile_variant
    cmd = oracle.runner(cfg)
    if cmd is None:
        return None, oracle.UNAVAILABLE
    out = work.resolve() / "sched.json"
    obj, err = compile_variant(cfg, unit, text, work, (), wrap=sched_wrap(cmd, out, symbol))
    if obj is None:
        return None, f"compile failed: {err[-300:]}"
    try:
        doc = json.loads(out.read_text())
    except (OSError, ValueError):
        return None, "the oracle runner wrote no scheduler records (does it support --sched?)"
    if doc.get("schema") != "sched.v1":
        return None, "the oracle runner's output is not sched.v1"
    return (doc, obj), ""


def render(rep: list[dict]) -> list[str]:
    out = []
    for r in rep:
        head = f"rows {r['rows'][0]}-{r['rows'][1]}: "
        st = r.get("status")
        if st == "mixed":
            out.append(head + "order and shape differ together (no window with the same instructions): a shape fix first")
            continue
        if st in ("unmapped", "same order"):
            out.append(head + ("no scheduler record lines up with these rows" if st == "unmapped" else "same order"))
            continue
        out.append(head + f"{r['pass']}-colouring pass, block {r['block']}")
        for t, o in zip(r["target"], r["ours"]):
            out.append(f"    {t:36} | {o}")
        d = r["decide"]
        out.append(f"    target first: `{r['x']['text']}` (line {r['x']['line']}); ours first: `{r['y']['text']}` "
                   f"(line {r['y']['line']}); decided at cycle {d.get('cycle', '?')} by: {d.get('rule')}")
        if r["levers"]:
            out += [f"    lever: {l}" for l in r["levers"]]
        if "pre" in r:
            p = r["pre"]
            out.append(f"    this pass kept its input order, which is the pre-colouring schedule (block {p['block']}): "
                       f"there it was decided by: {p['decide'].get('rule')}")
            out += [f"    pre-colouring lever: {l}" for l in p["levers"]]
        final = r.get("pre", r)
        if not final["levers"]:
            rule = final["decide"].get("rule")
            out.append("    no codegen-order lever: " + schedmodel.ADVICE.get(rule, "read the rows"))
    return out


def run(cfg: Config, args: argparse.Namespace) -> int:
    from .regs import unit_of_function
    unit_q, symbol = (args.unit, args.function) if args.function else (None, args.unit)
    if not symbol:
        print("usage: dk sched [UNIT] FN  |  dk sched UNIT --check")
        return USAGE
    if args.check and not args.function:
        unit_q, symbol = symbol, None
    unit_q = unit_q or unit_of_function(cfg, symbol)
    try:
        unit = resolve_unit(cfg, unit_q) if unit_q else None
    except MatchError as e:
        print(f"error: {e}")
        return USAGE
    if unit is None:
        print(f"error: {symbol} is not in a declared unit (give the unit)")
        return USAGE
    text = Path(args.source).read_text() if args.source else (cfg.root / unit["metadata"]["source_path"]).read_text()
    work = Path(tempfile.mkdtemp(prefix="dk-sched-"))
    try:
        got, err = records_for(cfg, unit, text, None if args.check else symbol, work)
        if got is None:
            print(f"error: {err}")
            return USAGE if "oracle" in err else BUILD_ERROR
        doc, obj = got
        if args.check:
            c = schedmodel.check(doc)
            for (pas, ok), n in sorted(c.items()):
                print(f"  {'pre' if pas == 'pre' else 'post'}-colouring blocks {'reproduced' if ok else 'DIFFERENT'}: {n}")
            bad = sum(n for (_, ok), n in c.items() if not ok)
            print("model: every scheduled block reproduced" if not bad else f"model: {bad} block(s) differ")
            hint(f"sched {unit['name']} FN")
            return OK if not bad else MISMATCH
        pair = objcache.function_pair(cfg, unit["target_path"], obj, symbol)
        if pair is None:
            print(f"error: {symbol} is not in both the target and our object")
            return BUILD_ERROR
        t, o = pair[0]["instructions"], pair[1]["instructions"]
        rep = schedmodel.explain(doc, symbol, t, o, residue.classify(t, o))
        print(f"{symbol}  {float(pair[0].get('match_percent', 0.0)):.2f}%  {len(rep)} scheduling region(s)")
        for line in render(rep):
            print(line)
        if args.json:
            Path(args.json).write_text(json.dumps(rep, indent=1) + "\n")
        hint(f"try {unit['name']} {symbol} --replace OLD NEW", f"regs --residue {unit['name']} {symbol}")
        return OK if not rep else MISMATCH
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("unit", nargs="?", help="unit (source path or name); optional when the function is unique")
    p.add_argument("function", nargs="?", help="function symbol, as dk check lists it")
    p.add_argument("--check", action="store_true", help="re-simulate every scheduled block of the unit against "
                   "the compiler's own order (the model's self-test)")
    p.add_argument("--source", help="explain this file instead of the unit's source")
    p.add_argument("--json", help="write the regions as JSON")


COMMAND = Command("sched", "matching", "explain an instruction-order difference from the compiler's scheduler "
                  "records: the deciding rule and the source-order levers (needs an oracle runner with --sched)",
                  run, _configure)
