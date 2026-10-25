"""dk commit-tools PATH... -m MSG: commit toolkit or docs changes, proven first.

The unit commands (accept, save) commit source; this commits everything else a maintainer
changes (tools/dk, tests, playbook, docs, dk.toml). In order, refusing at the first failure:
  1. the toolkit tests pass
  2. doc-lint passes
  3. HEAD plus exactly these paths builds and matches in a clean worktree
Then the paths are committed like a unit's delta (private index, compare-and-swap on HEAD).
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from . import doclint, isolated
from .command import MISMATCH, OK, USAGE, Command
from .config import Config


def run(cfg: Config, args: argparse.Namespace) -> int:
    if not args.message:
        print("error: give -m MESSAGE")
        return USAGE
    # the shared config files hold every agent's in-progress entries: committing one whole
    # would land other agents' unfinished work (unit commits compose them entry by entry)
    shared = {str(v) for k, v in cfg.platform_options.items() if k in ("units", "splits", "symbols")}
    blocked = [p for p in args.paths if p in shared]
    if blocked:
        print(f"REFUSED: {', '.join(blocked)} must change through ./dk accept, save, unit or rename commands")
        return USAGE
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=cfg.root, capture_output=True, text=True)
    if tests.returncode != 0:
        print("REFUSED: the toolkit tests fail:")
        print("\n".join(tests.stdout.strip().splitlines()[-15:]))
        return MISMATCH
    problems = doclint.lint(cfg)
    if problems:
        print("REFUSED: doc-lint fails:")
        for p in problems[:15]:
            print(f"  {p}")
        return MISMATCH
    overlay = isolated.working_overlay(cfg, args.paths)
    ok, detail = isolated.verify_overlay(cfg, overlay)
    if not ok:
        print(f"REFUSED: HEAD plus these paths does not build and match on its own:\n{detail}")
        return MISMATCH
    message = args.message
    for trailer in args.trailer:
        message += ("\n\n" if trailer == args.trailer[0] else "\n") + trailer
    ok, detail = isolated.commit_overlay(cfg, overlay, message)
    print(f"committed: {detail}" if ok else f"REFUSED: {detail}")
    return OK if ok else MISMATCH


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("paths", nargs="+", help="files to commit (their working-tree content)")
    p.add_argument("-m", "--message", help="commit message")
    p.add_argument("--trailer", action="append", default=[], help="trailer line (repeatable), e.g. Co-Authored-By: ...")


COMMAND = Command("commit-tools", "docs", "commit toolkit/docs changes after tests, doc-lint and a clean-worktree build",
                  run, _configure, locked=True)
