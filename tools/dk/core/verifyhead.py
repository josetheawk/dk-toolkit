"""dk verify-head: prove the committed state builds and matches, independent of the working tree.

Several agents share one working tree; this builds a commit in a throwaway git worktree
(core.isolated, which reuses the original binary and the downloaded tools without touching
them) and checks the hash.
"""

from __future__ import annotations

import argparse
import subprocess

from .command import BUILD_ERROR, OK, USAGE, Command, hint
from .config import Config
from .isolated import verify_overlay


def run(cfg: Config, args: argparse.Namespace) -> int:
    if args.rev != "HEAD":
        resolved = subprocess.run(["git", "rev-parse", "--verify", args.rev], cwd=cfg.root, capture_output=True, text=True)
        if resolved.returncode != 0:
            print(f"error: unknown revision {args.rev}")
            return USAGE
        print("note: verify-head checks HEAD; to check another revision, check it out in a separate clone")
        return USAGE
    ok, detail = verify_overlay(cfg, {}, timeout=args.timeout, keep=args.keep, name="verify-head")
    if not ok:
        print(f"FAILED: HEAD does not build and match on its own:\n{detail}")
        return BUILD_ERROR
    print("OK: HEAD builds and matches from a clean worktree")
    hint("status")
    return OK


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("rev", nargs="?", default="HEAD", help="only HEAD is supported")
    p.add_argument("--timeout", type=int, default=900, help="seconds per step (default 900)")
    p.add_argument("--keep", action="store_true", help="keep the worktree for inspection")


COMMAND = Command("verify-head", "build", "build HEAD in a clean worktree and prove it matches (catches incomplete commits)",
                  run, _configure)
