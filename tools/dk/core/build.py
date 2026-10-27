"""dk build: run ninja and give a verdict the caller can branch on."""

from __future__ import annotations

import argparse
import re

from .. import platforms
from .command import BUILD_ERROR, MISMATCH, OK, USAGE, Command, hint
from .config import Config
from .progress import summary_lines
from .util import run_ninja, sha1_file, tail

_FAILED_EDGE = re.compile(r"^FAILED: (?:\[code=\d+\] )?(?P<outputs>.+)$", re.M)


def failed_outputs(output: str) -> set[str]:
    """Output paths of the edges ninja reported as FAILED."""
    return {path for m in _FAILED_EDGE.finditer(output) for path in m["outputs"].split()}


def _hash_check_failed(output: str) -> bool:
    """True when the only failing ninja edge is the SHA-1 check."""
    edges = [m["outputs"] for m in _FAILED_EDGE.finditer(output)]
    return bool(edges) and all(e.rstrip().endswith("/ok") for e in edges)


def run(cfg: Config, args: argparse.Namespace) -> int:
    if not (cfg.root / "build.ninja").is_file():
        print("error: build.ninja missing")
        hint("setup")
        return USAGE

    prebuild = getattr(platforms.load(cfg.platform), "prebuild", None)
    if prebuild:
        prebuild(cfg)       # e.g. force a re-split when its config changed (never prove against stale outputs)
    proc = run_ninja(cfg.root, [*args.ninja_args])
    output = proc.stdout + proc.stderr
    if args.verbose:
        print(output.rstrip())

    if proc.returncode == 0:
        print(f"OK: {cfg.rel(cfg.built)} matches {cfg.sha1[:12]}")
        for line in summary_lines(cfg):
            print(line)
        hint("status")
        return OK

    if _hash_check_failed(output) and cfg.built.is_file():
        print(f"MISMATCH: {cfg.rel(cfg.built)} is {sha1_file(cfg.built)[:12]}, expected {cfg.sha1[:12]}")
        for line in platforms.load(cfg.platform).first_diff(cfg, cfg.built, cfg.original, args.limit):
            print("  " + line)
        hint("build --verbose")
        return MISMATCH

    print("BUILD ERROR (ninja output tail):")
    print(tail(output, args.lines))
    hint("build --verbose")
    return BUILD_ERROR


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("-v", "--verbose", action="store_true", help="print full ninja output")
    p.add_argument("--limit", type=int, default=8, help="differences to list on mismatch (default 8)")
    p.add_argument("--lines", type=int, default=40, help="output lines to show on build error (default 40)")
    p.add_argument("ninja_args", nargs=argparse.REMAINDER, help="extra arguments passed to ninja")


COMMAND = Command("build", "build", "run ninja; exit 0 match, 1 hash mismatch (+ first diff), 3 build error", run, _configure)
