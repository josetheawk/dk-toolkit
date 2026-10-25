"""dk accept: promote an exact unit to Matching (linked), prove the hash, commit."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from . import freshness as fresh
from ...core.command import MISMATCH, OK, USAGE, Command, hint
from ...core.config import Config
from ...core.util import run_ninja
from ...core.progress import load_report
from .dtk_symbols import build_errors
from .matching import MatchError, commit_gate, render, resolve_unit, score
from .unitcommit import commit_unit
from .units import read_units_file, render_units_file, set_status


def _matched_functions(cfg: Config) -> int:
    report = load_report(cfg) or {}
    return int(report.get("measures", {}).get("matched_functions", 0) or 0)


def _build(cfg: Config) -> tuple[bool, str]:
    proc = fresh.build(cfg)
    return proc.returncode == 0, proc.stdout + proc.stderr


def run(cfg: Config, args: argparse.Namespace) -> int:
    try:
        unit = resolve_unit(cfg, args.target)
    except MatchError as e:
        print(f"error: {e}")
        return USAGE
    source = unit["metadata"]["source_path"]
    rel_source = source.removeprefix(cfg.platform_options.get("src", "src") + "/")

    if fresh.build(cfg, [unit["base_path"]]).returncode != 0:
        print("error: unit does not compile")
        hint(f"check {args.target}")
        return MISMATCH
    s = score(cfg, unit["name"], cfg.root / unit["target_path"], cfg.root / unit["base_path"])
    if not s.exact:
        for line in render(s):
            print(line)
        print("REJECTED: unit is not an exact match")
        return MISMATCH

    forbidden = [p for p in commit_gate(cfg, unit) if p.startswith("forbidden")]
    if forbidden:
        for prob in forbidden:
            print(f"REJECTED: {prob}")
        return MISMATCH

    before = _matched_functions(cfg)
    units_path = cfg.root / cfg.platform_options["units"]
    original_units = units_path.read_text()
    units = read_units_file(units_path)
    set_status(units, rel_source, "Matching")
    units_path.write_text(render_units_file(units))

    ok, output = _build(cfg)
    after = _matched_functions(cfg)
    # The shared tree also holds other agents' half-edited units, so its numbers only warn;
    # the floor is enforced on the clean HEAD + this unit build that the commit is made from.
    if ok and after < before:
        print(f"note: matched functions in the shared tree fell {before} -> {after}; "
              f"the commit's clean build is checked against the floor")
    if not ok:
        print(build_errors(output, 30))
        if cfg.built.is_file() and "did NOT match" in output:
            from .firstdiff import first_diff             # where the linked binary differs
            print("first differences in the linked binary:")
            for line in first_diff(cfg, cfg.built, cfg.original, 6):
                print(f"  {line}")
        return _restore(cfg, units_path, original_units, "hash mismatch or build error after linking")
    print(f"ACCEPTED: {unit['name']} linked, hash OK, matched functions {before} -> {after}")

    if args.no_commit:
        return OK
    names = ", ".join(f.name for f in s.functions) or "(data only)"
    message = f"Match {rel_source}\n\nFunctions: {names}\n"
    if args.trailer:
        message += "\n" + "\n".join(args.trailer) + "\n"
    ok, detail = commit_unit(cfg, unit, message)
    if not ok and detail.startswith("below the progress floor"):
        return _restore(cfg, units_path, original_units, detail)
    print(f"committed: Match {rel_source} ({detail})" if ok else f"note: not committed ({detail})")
    hint("next", "status")
    return OK


def _restore(cfg: Config, units_path: Path, original: str, reason: str) -> int:
    units_path.write_text(original)
    _build(cfg)
    print(f"REJECTED: {reason}; units.toml restored")
    return MISMATCH


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("target", help="unit (source path or name) or a function inside one")
    p.add_argument("--no-commit", action="store_true", help="link and verify, but do not commit")
    p.add_argument("--trailer", action="append", default=[], help="commit trailer line, e.g. 'Co-Authored-By: ...'")


COMMAND = Command("accept", "matching", "link an exact unit, verify hash + no regression, commit it", run, _configure, locked=True)
