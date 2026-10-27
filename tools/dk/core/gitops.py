"""git calls that are safe while several agents share one working tree.

A held .git/index.lock (another agent's `git status`) is retried, not treated as failure.
Commits themselves go through isolated.py (private index, compare-and-swap on HEAD).
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

LOCK_RETRIES = 60
LOCK_WAIT = 0.25


def git(root: Path, *args: str) -> subprocess.CompletedProcess:
    for _ in range(LOCK_RETRIES):
        proc = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
        if proc.returncode == 0 or "index.lock" not in proc.stderr:
            return proc
        time.sleep(LOCK_WAIT)
    return proc


def tracked(root: Path, path: str) -> bool:
    return bool(git(root, "ls-files", "--", path).stdout.strip())
