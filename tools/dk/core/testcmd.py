"""dk test: run the toolkit's own tests (pytest if installed, else unittest)."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys

from .command import MISMATCH, OK, USAGE, Command, hint
from .config import Config


def run(cfg: Config, args: argparse.Namespace) -> int:
    if importlib.util.find_spec("pytest"):
        cmd = [sys.executable, "-m", "pytest", "-q", "tests", *args.extra]
    else:
        cmd = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", ".", *args.extra]
    code = subprocess.run(cmd, cwd=cfg.root).returncode
    if code == 0:
        hint("status")
        return OK
    if code == 1:                       # tests failed (both runners)
        hint("test -x")
        return MISMATCH
    print("the test runner itself failed (collection error, usage or no tests)")
    hint("test")
    return USAGE                        # pytest 2-5: interrupted, internal, usage, no tests collected


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("extra", nargs=argparse.REMAINDER, help="extra arguments for the test runner")


COMMAND = Command("test", "docs", "run the dk toolkit tests", run, _configure)
