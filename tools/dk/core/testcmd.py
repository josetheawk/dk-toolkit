"""dk test: run the toolkit's own tests (pytest if installed, else unittest), then the project's own.

The toolkit owns tests/, and a toolkit import mirrors it, so tests that only make sense in one project
(golden tests on its original binary) live outside it: dk.toml [test] paths, each a package folder
run after tests/.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import tomllib

from .command import MISMATCH, OK, USAGE, Command, hint
from .config import Config


def project_paths(root) -> list[str]:
    """dk.toml [test] paths that exist: the project's own test packages."""
    try:
        with (root / "dk.toml").open("rb") as f:
            paths = tomllib.load(f).get("test", {}).get("paths", [])
    except (OSError, tomllib.TOMLDecodeError):
        return []
    return [str(p) for p in paths if (root / str(p)).is_dir()]


def commands(root, extra: list[str], pytest: bool) -> list[list[str]]:
    """One runner call for tests/ and the project's paths (pytest), else one unittest discover per folder."""
    dirs = ["tests", *project_paths(root)]
    if pytest:
        return [[sys.executable, "-m", "pytest", "-q", *dirs, *extra]]
    return [[sys.executable, "-m", "unittest", "discover", "-s", d, "-t", ".", *extra] for d in dirs]


def run(cfg: Config, args: argparse.Namespace) -> int:
    code = 0
    for cmd in commands(cfg.root, args.extra, bool(importlib.util.find_spec("pytest"))):
        code = max(code, subprocess.run(cmd, cwd=cfg.root).returncode)
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
