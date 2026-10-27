"""dk check / dk compiler-search."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

from . import freshness as fresh
from ...core.build import failed_outputs
from ...core.command import BUILD_ERROR, MISMATCH, OK, USAGE, Command, hint
from ...core.config import Config
from ...core.util import run_ninja
from .linkproof import function_proof
from .matching import (MatchError, UnitScore, compile_command, load_objdiff_units, render, resolve_unit,
                       retarget, run_compile, score)


def run_check(cfg: Config, args: argparse.Namespace) -> int:
    if args.all:
        return run_check_all(cfg, args.near)
    if not args.target:
        print("error: give a unit or function, or --all")
        return USAGE
    try:
        unit = resolve_unit(cfg, args.target)
    except MatchError as e:
        print(f"error: {e}")
        return USAGE
    proof = None
    if args.private:
        s = _private_score(cfg, unit, args.function)
        if isinstance(s, int):
            return s
        s, proof = s
    else:
        proc = fresh.build(cfg, [unit["base_path"]])
        if proc.returncode != 0:
            print("BUILD ERROR:")
            print("\n".join(line for line in proc.stdout.splitlines() if "Usage Warning" not in line)[-3000:])
            return BUILD_ERROR
        try:
            s = score(cfg, unit["name"], cfg.root / unit["target_path"], cfg.root / unit["base_path"])
        except MatchError as e:
            print(f"error: {e}")
            return USAGE
        if args.function:
            proof = function_proof(cfg, unit, cfg.root / unit["base_path"], args.function)

    for line in render(s):
        print(line)
    if args.function:
        f = next((f for f in s.functions if f.name == args.function), None)
        if f is None:
            print(f"error: {args.function} not in {unit['name']}")
            return USAGE
        ok = f.percent == 100.0
        if proof is not None:
            # the hand link: our function relocated to its final addresses vs the original's bytes
            print(f"link proof: {proof.detail}")
            if proof.judged and not proof.ok and ok:
                print("note: objdiff calls it equal, but linked it would differ: fix that before accepting")
            ok = ok and (proof.ok or not proof.judged)
        print("MATCH" if ok else f"MISMATCH ({f.percent:.2f}%)")
        return OK if ok else MISMATCH
    print("MATCH" if s.exact else f"MISMATCH (code {s.code_percent:.2f}%)")
    if s.exact:
        hint(f"accept {args.target}")
    return OK if s.exact else MISMATCH


def _private_score(cfg: Config, unit: dict, function: str | None = None):
    """Compile the unit into a private directory and score it there: no project lock and no
    shared build outputs, so iterating on a unit never waits for other agents (accept and save
    still prove the real build). Returns (score, link proof of `function` or None)."""
    import os
    import shutil
    work = cfg.root / "build" / "dk" / "try" / str(os.getpid())
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        try:
            command = compile_command(cfg, unit["base_path"], locked=False)
        except MatchError as e:
            print(f"error: {e}")
            return USAGE
        ok, out = run_compile(cfg, retarget(command, work))
        objs = list(work.glob("*.o"))
        if not ok or not objs:
            print("BUILD ERROR:")
            print(out[-3000:])
            return BUILD_ERROR
        try:
            s = score(cfg, unit["name"], cfg.root / unit["target_path"], objs[0])
            return s, (function_proof(cfg, unit, objs[0], function) if function else None)
        except MatchError as e:
            print(f"error: {e}")
            return USAGE
    finally:
        shutil.rmtree(work, ignore_errors=True)


def check_all_rows(units: list[dict], scores: dict[str, UnitScore | None]) -> tuple[list[str], int]:
    """One line per unit (linked ones first, then WIP by code %), plus the exit code: a linked unit
    that no longer compiles or is no longer exact fails the check; WIP units are only reported."""
    linked = [u for u in units if u["metadata"].get("complete")]
    wip = [u for u in units if not u["metadata"].get("complete")]
    lines, failed, broken = [], 0, 0

    def line(u: dict, tag: str) -> str:
        s = scores.get(u["name"])
        if s is None:
            return f"  {tag:<7} {'BUILD ERROR':>21}  {u['name']}"
        n_exact = sum(f.percent == 100.0 for f in s.functions)
        data = "ok" if all(v == 100.0 for v in s.data.values()) and not s.data_problems else "DIFF"
        layout = "ok" if not s.layout_problems else "DIFF"
        return f"  {tag:<7} {n_exact:>4}/{len(s.functions):<4} {s.code_percent:7.2f}%  data {data:<4} layout {layout:<4} {u['name']}"

    for u in sorted(linked, key=lambda u: u["name"]):
        s = scores.get(u["name"])
        if s is None:
            broken += 1
        elif not s.exact:
            failed += 1
        if s is None or not s.exact:
            lines.append(line(u, "LINKED"))
    lines.insert(0, f"linked units: {len(linked) - failed - broken}/{len(linked)} exact"
                    + (f", {failed} NOT EXACT" if failed else "") + (f", {broken} BUILD ERROR" if broken else ""))
    lines.append(f"wip units: {len(wip)}")
    rank = lambda u: -(scores[u["name"]].code_percent if scores.get(u["name"]) else -1.0)
    lines += [line(u, "wip") for u in sorted(wip, key=lambda u: (rank(u), u["name"]))]
    return lines, BUILD_ERROR if broken else (MISMATCH if failed else OK)


def near_rows(units: list[dict], scores: dict, limit: int = 3) -> list[str]:
    """WIP units at most `limit` functions from exact, cheapest first, each blocking item with
    its reason: what a closer would fix to link the unit."""
    rows = []
    for u in units:
        s = scores.get(u["name"])
        if s is None or u["metadata"].get("complete") or s.exact:
            continue
        open_fns = [f for f in s.functions if f.percent < 100.0]
        if len(open_fns) > limit:
            continue
        items = [f"{f.name} {f.percent:.2f}%: {f.note or 'code differs (dk diff --function)'}" for f in open_fns]
        items += [f"vtable {p}" for p in s.data_problems[:3]]
        items += [f"{sec} {pct:.2f}% (dk diff --literals {sec})" for sec, pct in sorted(s.data.items()) if pct < 100.0]
        if s.layout_problems:
            items.append(f"layout: {s.layout_problems[0]}" + (f" (+{len(s.layout_problems) - 1} more)" if len(s.layout_problems) > 1 else ""))
        rows.append((len(open_fns), sum(f.size for f in open_fns), len(items), u["name"], items))
    out = [f"near link ({len(rows)} WIP unit(s) at most {limit} function(s) from exact):"]
    for n, size, _, name, items in sorted(rows):
        out.append(f"  {name}: {n} function(s) open, 0x{size:X} bytes")
        out += [f"      {item}" for item in items]
    return out


def run_check_all(cfg: Config, near: bool = False) -> int:
    try:
        units = [u for u in load_objdiff_units(cfg) if u.get("base_path")]
    except MatchError as e:
        print(f"error: {e}")
        return USAGE
    proc = fresh.build(cfg, ["-k", "0", *(u["base_path"] for u in units)])
    failed = failed_outputs(proc.stdout)
    scores: dict[str, UnitScore | None] = {}
    for u in units:
        base = cfg.root / u["base_path"]
        if u["base_path"] in failed or not base.is_file():
            scores[u["name"]] = None
            continue
        try:
            scores[u["name"]] = score(cfg, u["name"], cfg.root / u["target_path"], base)
        except MatchError:
            scores[u["name"]] = None
    lines, code = check_all_rows(units, scores)
    print("\n".join(lines))
    if near:
        print("\n".join(near_rows(units, scores)))
    print("MATCH (every linked unit exact)" if code == OK else "REGRESSION: a linked unit is not exact (fix before any commit)")
    return code


def _gc_versions(cfg: Config) -> list[str]:
    base = cfg.root / "build" / "compilers" / "GC"
    return sorted((p.name for p in base.iterdir() if (p / "mwcceppc.exe").is_file()), key=_version_key) if base.is_dir() else []


def _version_key(v: str) -> tuple:
    parts = []
    for token in v.replace("p", ".p").replace("n", ".n").replace("r", ".r").replace("a", ".a").split("."):
        parts.append((0, int(token)) if token.isdigit() else (1, token))
    return tuple(parts)


def run_search(cfg: Config, args: argparse.Namespace) -> int:
    try:
        unit = resolve_unit(cfg, args.target)
        command = compile_command(cfg, unit["base_path"])
    except MatchError as e:
        print(f"error: {e}")
        return USAGE
    versions = args.versions.split(",") if args.versions else _gc_versions(cfg)
    variants: list[tuple[str, ...]] = [()] + [tuple(v.split()) for v in args.flags]
    rows = []
    for version in versions:
        for flags in variants:
            label = " ".join(flags) or "(project flags)"
            out_dir = cfg.root / "build" / "dk" / "search" / version / (hashlib.sha1(label.encode()).hexdigest()[:8] if flags else "base")
            out_dir.mkdir(parents=True, exist_ok=True)
            ok, text = run_compile(cfg, retarget(command, out_dir, f"GC/{version}", flags))
            obj = out_dir / Path(unit["base_path"]).name
            if not ok or not obj.is_file():
                rows.append((-1.0, version, label, "compile error", text.splitlines()[-1] if text else ""))
                continue
            s = score(cfg, unit["name"], cfg.root / unit["target_path"], obj)
            verdict = "EXACT" if s.exact else ("layout" if s.code_percent == 100.0 else "")
            rows.append((s.code_percent, version, label, verdict, ""))

    order = {v: i for i, v in enumerate(versions)}
    rows.sort(key=lambda r: (-r[0], order[r[1]], r[2]))
    print(f"compiler search for {unit['name']} ({len(rows)} builds)")
    print(f"  {'code':>8}  {'compiler':<10} {'verdict':<8} flags")
    for pct, version, label, verdict, note in rows:
        pct_s = "   error" if pct < 0 else f"{pct:7.2f}%"
        print(f"  {pct_s}  GC/{version:<7} {verdict:<8} {label} {note}".rstrip())
    exact = [r for r in rows if r[3] == "EXACT"]
    return OK if exact else MISMATCH


def _configure_check(p: argparse.ArgumentParser) -> None:
    p.add_argument("target", nargs="?", help="unit (source path or name) or a function symbol inside a declared unit")
    p.add_argument("--function", help="judge only this function: exact, and hand-linked byte-identical to the original")
    p.add_argument("--private", action="store_true",
                   help="compile into a private directory, without the project lock (fast iteration; accept proves the real build)")
    p.add_argument("--all", action="store_true",
                   help="re-score every source unit; exit 1 if a linked unit is no longer exact, list WIP by code %%")
    p.add_argument("--near", action="store_true",
                   help="with --all: list WIP units at most 3 functions from exact, with what blocks each item")


def _configure_search(p: argparse.ArgumentParser) -> None:
    p.add_argument("target", help="unit (source path or name) or a function inside one")
    p.add_argument("--versions", help="comma list, e.g. 1.3,1.3.2,2.0 (default: every GC compiler)")
    p.add_argument("--flags", action="append", default=[], help="extra flag variant to try (repeatable), e.g. '-inline deferred'")


CHECK = Command("check", "matching", "compile a unit and score it vs the target (functions, data, layout)", run_check, _configure_check)
SEARCH = Command("compiler-search", "matching", "rank every compiler version x flag variant for a unit", run_search, _configure_search)
