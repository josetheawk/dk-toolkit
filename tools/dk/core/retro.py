"""dk retro: the end-of-phase routine that keeps the playbook improving and small."""

from __future__ import annotations

import argparse

from .command import MISMATCH, OK, Command
from .config import Config
from .doclint import budget_usage, lint

CHECKLIST = """\
Retro checklist (end of every phase, or after anything that cost an hour or more):
  1. What cost time? Write it down as rule -> why -> how to detect/apply.
  2. Generic (any project on this platform or any platform)? -> playbook/
     Only true for this project?                                -> project docs
  3. Could a script have caught it or done it? -> add or extend a dk command + its test.
  4. Delete or merge anything stale or duplicated. Smaller is better.
  5. Re-run until doc-lint passes, then commit the docs with the work they describe."""


def run(cfg: Config, args: argparse.Namespace) -> int:
    print(CHECKLIST)
    print("\nBudgets:")
    for label, used, limit in budget_usage(cfg):
        flag = "  OVER" if used > limit else ""
        print(f"  {label:<45} {used:>5} / {limit}{flag}")
    problems = lint(cfg)
    print(f"\ndoc-lint: {'OK' if not problems else f'{len(problems)} problem(s)'}")
    for p in problems:
        print(f"  {p}")
    return MISMATCH if problems else OK


COMMAND = Command("retro", "docs", "print the retro checklist, doc budget usage and doc-lint result", run)
