"""dk test: run the toolkit's own tests (pytest if installed, else unittest)."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys

from .command import MISMATCH, OK, Command
from .config import Config


def run(cfg: Config, args: argparse.Namespace) -> int:
    if importlib.util.find_spec("pytest"):
        cmd = [sys.executable, "-m", "pytest", "-q", "tests", *args.extra]
    else:
        cmd = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", ".", *args.extra]
    return OK if subprocess.run(cmd, cwd=cfg.root).returncode == 0 else MISMATCH


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("extra", nargs=argparse.REMAINDER, help="extra arguments for the test runner")


COMMAND = Command("test", "docs", "run the dk toolkit tests", run, _configure)
