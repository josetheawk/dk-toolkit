"""dk save <unit>: commit work in progress on a unit (not linked yet), safely.

Compiles the unit in the shared tree to score it, then commits exactly HEAD + this unit's own
delta once that builds and matches in a clean worktree (see unitcommit).
"""

from __future__ import annotations

import argparse

from . import freshness as fresh
from ...core.command import MISMATCH, OK, USAGE, Command, hint
from ...core.config import Config
from .matching import MatchError, commit_gate, resolve_unit, score
from .unitcommit import commit_units


def run(cfg: Config, args: argparse.Namespace) -> int:
    try:
        units = [resolve_unit(cfg, t) for t in args.target]
    except MatchError as e:
        print(f"error: {e}")
        return USAGE
    # Only these units' objects, to score them: the clean-worktree verify in commit_units proves
    # the hash, so another agent's half-done edit in the shared tree cannot block this save.
    if fresh.build(cfg, [u["base_path"] for u in units], own=tuple(u["base_path"] for u in units)).returncode != 0:
        print("REJECTED: a unit does not compile")
        hint(f"check {args.target[0]} --private")
        return MISMATCH
    lines = []
    for unit in units:
        s = score(cfg, unit["name"], cfg.root / unit["target_path"], cfg.root / unit["base_path"])
        exact = sum(1 for f in s.functions if f.percent == 100.0)
        rel = unit["metadata"]["source_path"].removeprefix(cfg.platform_options.get("src", "src") + "/")
        lines.append(f"{rel}: {exact}/{len(s.functions)} functions exact")
        problems = commit_gate(cfg, unit)
        if problems:
            for prob in problems:
                print(f"REJECTED: {prob}")
            return MISMATCH
    message = (f"WIP {lines[0]}\n" if len(lines) == 1
               else f"WIP {len(lines)} units together\n\n" + "\n".join(lines) + "\n")
    if args.note:
        message += "\n" + "\n".join(args.note) + "\n"
    if args.trailer:
        message += "\n" + "\n".join(args.trailer) + "\n"
    ok, detail = commit_units(cfg, units, message)
    if not ok:
        print(f"{'REJECTED' if detail.startswith('below the progress floor') else 'note: not committed'}: {detail}")
        return MISMATCH if detail.startswith("below the progress floor") else OK
    print(f"committed: WIP {'; '.join(lines)} ({detail})")
    hint(f"check {args.target[0]}", "next")
    return OK


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("target", nargs="+", help="unit(s) (source path or name) or a function inside one; several "
                   "units commit together (a unit split in several, or units that need each other)")
    p.add_argument("--note", action="append", default=[], help="what is left / what was tried (commit body; repeatable)")
    p.add_argument("--trailer", action="append", default=[], help="commit trailer line")


COMMAND = Command("save", "matching", "commit work in progress on a unit (build must match)", run, _configure, locked=True)
