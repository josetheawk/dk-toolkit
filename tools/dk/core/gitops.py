"""Commits that are safe when several agents share one working tree.

- Only the given paths are committed (`git commit -- <paths>`), whatever else is staged.
- A held .git/index.lock (another agent's `git status`) is retried, not treated as failure.
- If the commit still fails, the paths this call staged are unstaged again.
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


def changed(root: Path, paths: list[str]) -> list[str]:
    """Paths (files) under `paths` that are new or modified relative to HEAD."""
    if not paths:
        return []
    out = git(root, "status", "--porcelain", "--untracked-files=all", "--", *paths).stdout
    return sorted({line[3:].split(" -> ")[-1] for line in out.splitlines() if line.strip()})


def tracked(root: Path, path: str) -> bool:
    return bool(git(root, "ls-files", "--", path).stdout.strip())


def commit_paths(root: Path, paths: list[str], message: str) -> tuple[bool, str]:
    files = changed(root, [p for p in paths if (root / p).exists() or tracked(root, p)])
    if not files:
        return False, "nothing to commit"
    add = git(root, "add", "--", *files)
    if add.returncode != 0:
        git(root, "restore", "--staged", "--", *files)
        return False, add.stderr.strip()
    proc = git(root, "commit", "-q", "-m", message, "--", *files)
    if proc.returncode != 0:
        git(root, "restore", "--staged", "--", *files)
        return False, (proc.stderr or proc.stdout).strip()
    return True, f"{len(files)} file(s)"
