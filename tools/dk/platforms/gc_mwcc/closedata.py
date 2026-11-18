"""dk close-data: close a unit's open data sections with verified typed definitions, and the honest data census.

  ./dk close-data <unit…> [--apply] [--records PATH] [--jobs N]
  ./dk close-data --census [--json PATH]

Per unit: the source is scored privately (dk's scoring: weak copies and stripped objects allowed for, the
same as `dk check`), its data objects are drafted and verified by `gen --data`'s engine (typed from the
unit's externs, dtk's types, the original values and relocations; compiler data left alone; no byte dumps),
the verified drafts are written into a private copy and that copy is scored again. The result is kept
(`--apply` writes the unit's source) only when no function or data section scores lower and an open data
section rose; a section at 100 % afterwards is "closed exactly". `--records PATH` appends one JSON line per
verified draft (the target's bytes and relocations, the symbol's attributes, the evidence and the exact
definition): tool output, verified by compile and comparison, for training data.

`--census`: every WIP unit built once (one ninja run) and scored with dk's allowances; per unit the data
sections open under dk's scoring beside the raw objdiff percentages of the report. Written to
build/dk/data-census.json, which `dk route --unlock` then prefers over the report's raw sections (the raw
report counts a discarded weak copy or a stripped object as a data miss, so it overstates data blockers).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ...core.command import MISMATCH, OK, USAGE, Command, hint
from ...core.config import Config
from .matching import MatchError, UnitScore, resolve_unit, score

CENSUS = Path("build") / "dk" / "data-census.json"
DATA_SECTIONS = (".data", ".rodata", ".sdata", ".sdata2", ".bss", ".sbss", ".sbss2", ".ctors")


def open_sections(s: UnitScore) -> dict[str, float]:
    """The data sections below 100 % under dk's scoring."""
    return {sec: round(p, 2) for sec, p in sorted(s.data.items()) if p < 100.0}


def outcome(before: UnitScore, after: UnitScore | None) -> dict:
    """What writing the drafts did: sections closed exactly, raised, lowered (with any function that got worse),
    and whether the result may be kept (nothing lower, something open rose)."""
    from .reorder import verdict
    if after is None:
        return {"compiles": False, "closed": [], "raised": [], "worse": ["does not compile"], "keep": False}
    was = open_sections(before)
    now = {sec: round(after.data.get(sec, 0.0), 2) for sec in was}
    closed = [sec for sec in was if now[sec] >= 100.0]
    raised = [sec for sec in was if was[sec] < now[sec] < 100.0]
    worse = verdict(before, after)
    return {"compiles": True, "closed": closed, "raised": raised, "worse": worse,
            "keep": not worse and bool(closed or raised), "exact": after.exact,
            "before": was, "after": now}


def record(ud, d) -> dict:
    """One training record for a verified draft: the target object as bytes and relocations plus its symbol
    attributes and evidence, and the exact definition the compiler reproduced it from."""
    orig = ud.original(d.name, d.size) if d.section not in (".bss", ".sbss") else None
    attrs = ud.attrs.get(d.name, {})
    return {"unit": ud.unit["name"], "source": ud.rel, "name": d.name, "section": d.section,
            "address": f"{d.address:#x}", "size": d.size,
            "symbol": {k: v for k, v in attrs.items() if k in ("type", "size", "scope", "data", "align")},
            "target": {"bytes": orig[0].hex() if orig else None,
                       "relocs": {f"{k:#x}": v for k, v in sorted(orig[1].items())} if orig else {}},
            "evidence": d.evidence, "needs": list(d.needs), "replaces_extern": d.replaces is not None,
            "definition": d.text, "verified": "size, bytes and relocations against the target object"}


def census_rows(units: list[dict], scores: dict) -> dict:
    """Per WIP unit: data sections open under dk's scoring and under the report's raw objdiff percentages."""
    out = {}
    for u in units:
        if u.get("metadata", {}).get("complete"):
            continue
        s = scores.get(u["name"])
        raw = {sec["name"]: round(float(sec.get("fuzzy_match_percent") or 0.0), 2) for sec in u.get("sections", [])
               if sec["name"] in DATA_SECTIONS and float(sec.get("fuzzy_match_percent") or 0.0) < 100.0}
        out[u["name"]] = {"dk": open_sections(s) if s is not None else None, "raw": raw,
                          "open_functions": sum(1 for f in (s.functions if s else []) if f.percent < 100.0)}
    return out


def census_summary(rows: dict) -> list[str]:
    scored = {n: r for n, r in rows.items() if r["dk"] is not None}
    dk_open = [n for n, r in scored.items() if r["dk"]]
    raw_open = [n for n, r in scored.items() if r["raw"]]
    over = [n for n in raw_open if not scored[n]["dk"]]
    zero = [n for n in dk_open if scored[n]["open_functions"] == 0]
    out = [f"data census: {len(scored)} WIP unit(s) scored ({len(rows) - len(scored)} did not build)",
           f"  data open under dk's scoring: {len(dk_open)}; under the raw report: {len(raw_open)}; "
           f"raw-only (weak copies or stripped objects, no work): {len(over)}",
           f"  data open with every function exact (data is the last blocker): {len(zero)}"]
    for n in sorted(dk_open, key=lambda n: (scored[n]["open_functions"], n)):
        r = scored[n]
        out.append(f"    {n}: {r['open_functions']} open fn, " + ", ".join(f"{s} {p:.2f}%" for s, p in r["dk"].items()))
    return out


def _score_text(cfg: Config, unit: dict, text: str, slot: str) -> UnitScore | None:
    from .reorder import compile_variant
    work = cfg.root / "build" / "dk" / "close-data" / slot
    obj, _ = compile_variant(cfg, unit, text, work)
    try:
        return score(cfg, unit["name"], cfg.root / unit["target_path"], obj) if obj is not None else None
    finally:
        shutil.rmtree(work, ignore_errors=True)


def close_unit(cfg: Config, ctx, unit: dict, jobs: int, apply_it: bool, records) -> dict:
    """Draft, verify, score before and after; write the source when the outcome may be kept."""
    from . import datagen
    ud = datagen.draft_unit(cfg, ctx, unit, jobs)
    if ud is None:
        return {"unit": unit["name"], "error": "not in splits.txt"}
    before = _score_text(cfg, unit, ud.src, "before")
    if before is None:
        return {"unit": unit["name"], "error": "the source does not compile"}
    verified = ud.verified
    todo = [d for d in ud.drafts if d.status in ("todo", "failed")]
    res = {"unit": unit["name"], "open_before": open_sections(before), "verified": len(verified),
           "todo": [(d.name, d.section, d.why[:80]) for d in todo], "written": False}
    if records is not None:
        for d in verified:
            records.write(json.dumps(record(ud, d), sort_keys=True) + "\n")
    if not verified:
        res["outcome"] = None
        return res
    new = datagen.apply(ud.src, verified)
    after = _score_text(cfg, unit, new, "after")
    res["outcome"] = outcome(before, after)
    if apply_it and res["outcome"]["keep"]:
        ud.source.write_text(new)
        res["written"] = True
    return res


def render(res: dict) -> list[str]:
    out = [f"{res['unit']}:"]
    if res.get("error"):
        return out + [f"  error: {res['error']}"]
    ob = res["open_before"]
    out.append("  data open (dk scoring): " + (", ".join(f"{s} {p:.2f}%" for s, p in ob.items()) or "none"))
    out.append(f"  verified drafts: {res['verified']}; no type evidence or failed: {len(res['todo'])}")
    for name, sec, why in res["todo"][:6]:
        out.append(f"    todo {sec:<8} {name:<22} {why}")
    o = res["outcome"]
    if o is None:
        out.append("  nothing to write")
    elif not o["compiles"]:
        out.append("  the verified drafts do not compile together")
    else:
        out.append(f"  closed exactly: {', '.join(o['closed']) or 'none'}; raised: {', '.join(o['raised']) or 'none'}"
                   + (f"; worse: {'; '.join(o['worse'])}" if o["worse"] else ""))
        out.append("  " + ("written" if res["written"] else "kept (pass --apply to write)" if o["keep"] else "not kept")
                   + (" — unit exact: ./dk accept" if o.get("exact") else ""))
    return out


def run_census(cfg: Config, args: argparse.Namespace) -> int:
    from .matchcmds import score_units
    from .matching import load_objdiff_units
    try:
        units = [u for u in load_objdiff_units(cfg) if u.get("base_path") and not u["metadata"].get("complete")]
    except MatchError as e:
        print(f"error: {e}")
        return USAGE
    from ...core.progress import load_report
    sections = {u["name"]: u.get("sections", []) for u in (load_report(cfg) or {}).get("units", [])}
    units = [{**u, "sections": sections.get(u["name"], [])} for u in units]     # the raw percentages, for the comparison
    rows = census_rows(units, score_units(cfg, units))
    out = cfg.root / (Path(args.json) if args.json else CENSUS)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1, sort_keys=True) + "\n")
    print("\n".join(census_summary(rows)))
    print(f"written: {cfg.rel(out)} (dk route --unlock reads it)")
    hint("route --unlock --data", "close-data <unit> --apply")
    return OK


def run(cfg: Config, args: argparse.Namespace) -> int:
    if args.census:
        return run_census(cfg, args)
    if not args.unit:
        print("error: give units or --census")
        return USAGE
    from .gencmd import Context
    from .jobs import resolve
    try:
        units = [resolve_unit(cfg, u) for u in args.unit]
    except MatchError as e:
        print(f"error: {e}")
        return USAGE
    ctx = Context(cfg)
    jobs = resolve(args.jobs, cfg)
    records = None
    if args.records:
        Path(args.records).parent.mkdir(parents=True, exist_ok=True)
        records = open(args.records, "a")
    closed = written = 0
    try:
        for unit in units:
            res = close_unit(cfg, ctx, unit, jobs, args.apply, records)
            print("\n".join(render(res)))
            o = res.get("outcome") or {}
            closed += len(o.get("closed", []))
            written += res.get("written", False)
    finally:
        if records is not None:
            records.close()
    print(f"{closed} data section(s) closed exactly in {len(units)} unit(s); {written} source(s) written")
    hint("check <unit> --private", "save <unit> --note \"data closed\"")
    return OK if closed else MISMATCH


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("unit", nargs="*", help="unit(s) (source path or name)")
    p.add_argument("--apply", action="store_true", help="write the verified definitions into the unit's source when "
                                                        "nothing scores lower and an open data section rises")
    p.add_argument("--records", metavar="PATH", help="append one JSON line per verified definition (target bytes, "
                                                     "relocations, symbol attributes, evidence, definition)")
    p.add_argument("--census", action="store_true", help="score every WIP unit with dk's allowances and write the data "
                                                         "sections open per unit to build/dk/data-census.json")
    p.add_argument("--json", metavar="PATH", help="with --census: write the census here instead")
    p.add_argument("--jobs", type=int, default=0, help="parallel compiles (default: dk.toml [platform] jobs, DK_JOBS, "
                                                       "else the idle cores)")


COMMAND = Command("close-data", "matching", "close a unit's open data sections with verified typed definitions "
                  "(gen --data's engine, dk's scoring); --census: the data sections open per WIP unit", run, _configure)
