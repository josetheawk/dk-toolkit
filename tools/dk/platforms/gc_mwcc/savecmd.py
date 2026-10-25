"""dk save <unit>: commit work in progress on a unit (not linked yet), safely.

Runs under the project lock, proves the whole build still matches, and commits exactly
HEAD + this unit's own delta, verified in a clean worktree (see unitcommit).
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from . import freshness as fresh
from ...core.command import MISMATCH, OK, USAGE, Command, hint
from ...core.config import Config
from ...core.util import run_ninja
from ...core import floor
from .matching import MatchError, commit_gate, resolve_unit, score
from .unitcommit import commit_unit


def run(cfg: Config, args: argparse.Namespace) -> int:
    try:
        unit = resolve_unit(cfg, args.target)
    except MatchError as e:
        print(f"error: {e}")
        return USAGE
    if fresh.build(cfg).returncode != 0:
        print("REJECTED: the build does not match; fix it before saving")
        hint("build --verbose")
        return MISMATCH
    s = score(cfg, unit["name"], cfg.root / unit["target_path"], cfg.root / unit["base_path"])
    exact = sum(1 for f in s.functions if f.percent == 100.0)
    rel = unit["metadata"]["source_path"].removeprefix(cfg.platform_options.get("src", "src") + "/")
    problems = commit_gate(cfg, unit)
    if problems:
        for prob in problems:
            print(f"REJECTED: {prob}")
        return MISMATCH
    floor.raise_floor(cfg)
    message = f"WIP {rel}: {exact}/{len(s.functions)} functions exact\n"
    if args.note:
        message += "\n" + "\n".join(args.note) + "\n"
    if args.trailer:
        message += "\n" + "\n".join(args.trailer) + "\n"
    ok, detail = commit_unit(cfg, unit, message)
    if not ok:
        print(f"note: not committed ({detail})")
        return OK
    print(f"committed: WIP {rel} ({exact}/{len(s.functions)} exact, {detail})")
    hint(f"check {rel}", "next")
    return OK


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("target", help="unit (source path or name) or a function inside one")
    p.add_argument("--note", action="append", default=[], help="what is left / what was tried (commit body; repeatable)")
    p.add_argument("--trailer", action="append", default=[], help="commit trailer line")


COMMAND = Command("save", "matching", "commit work in progress on a unit (build must match)", run, _configure, locked=True)
