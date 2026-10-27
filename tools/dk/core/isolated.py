"""Commits that are exactly "HEAD + these files", proven to build before they land.

verify_overlay()  builds HEAD plus an overlay of file contents in a throwaway git worktree
                  (original binary and downloaded tools are symlinked in) and checks the hash.
commit_overlay()  writes the overlay with git plumbing through a private index, so the
                  shared index is never touched; HEAD advances with a compare-and-swap, so a
                  concurrent commit can never be overwritten. The shared index is then synced
                  for the committed paths only.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .config import Config
from .gitops import git
from .util import kill_session, released, tail

Overlay = dict[str, "bytes | None"]   # path -> new content (None = delete)

# The commit a verify ran against. Verifies build without the project lock, so a commit lands
# only on that same HEAD; when another commit got in first, the caller recomposes and retries.
_pinned: str | None = None
_unlocked = True
MOVED = "HEAD moved concurrently; retry"
UNLOCKED_TRIES = 2   # then verify holding the lock: commits take it too, so HEAD cannot move


def pin_head(cfg: Config, attempt: int = 0) -> str:
    """Pin HEAD for attempt `attempt` of a verify-and-commit. Early attempts build without the
    project lock (other agents keep committing); a later one keeps it, so it always lands."""
    global _pinned, _unlocked
    _pinned = git(cfg.root, "rev-parse", "HEAD").stdout.strip()
    _unlocked = attempt < UNLOCKED_TRIES
    return _pinned


def show(cfg: Config, rel: str) -> bytes | None:
    """`rel` at the pinned commit (HEAD if none is pinned), or None if it does not exist there."""
    proc = subprocess.run(["git", "show", f"{_pinned or 'HEAD'}:{rel}"], cwd=cfg.root, capture_output=True)
    return proc.stdout if proc.returncode == 0 else None


def _worktree(cfg: Config, name: str) -> Path:
    tree = cfg.root / "build" / "dk" / name
    subprocess.run(["git", "worktree", "remove", "--force", str(tree)], cwd=cfg.root, capture_output=True)
    shutil.rmtree(tree, ignore_errors=True)
    subprocess.run(["git", "worktree", "prune"], cwd=cfg.root, capture_output=True)
    proc = subprocess.run(["git", "worktree", "add", "--detach", str(tree), _pinned or "HEAD"], cwd=cfg.root,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip())
    return tree


def _drop(cfg: Config, tree: Path) -> None:
    subprocess.run(["git", "worktree", "remove", "--force", str(tree)], cwd=cfg.root, capture_output=True)
    shutil.rmtree(tree, ignore_errors=True)


def verify_overlay(cfg: Config, overlay: Overlay, timeout: int = 900, keep: bool = False,
                   name: str = "", measures: dict | None = None) -> tuple[bool, str]:
    """`measures`, when given, receives the clean tree's progress measures (what would be committed)."""
    tree = _worktree(cfg, name or f"verify-{os.getpid()}")
    try:
        for rel, data in overlay.items():
            dest = tree / rel
            if data is None:
                dest.unlink(missing_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
        original = tree / cfg.rel(cfg.original)
        original.parent.mkdir(parents=True, exist_ok=True)
        original.unlink(missing_ok=True)
        original.symlink_to(cfg.original)
        # Point configure at the already-downloaded tools instead of symlinking build/tools:
        # a fresh checkout makes download_tool.py look newer, and ninja would re-download
        # *through* a symlink into the shared tools dir while other builds are running them.
        tools = cfg.root / "build" / "tools"
        tool_args = []
        for flag, name in (("--dtk", "dtk"), ("--objdiff", "objdiff-cli"), ("--wrapper", "wibo"), ("--sjiswrap", "sjiswrap.exe")):
            if (tools / name).is_file():
                tool_args += [flag, str(tools / name)]
        if (cfg.root / "build" / "compilers").is_dir():
            tool_args += ["--compilers", str(cfg.root / "build" / "compilers")]
        for step in ([sys.executable, *cfg.configure, *tool_args], ["ninja"]):
            p = subprocess.Popen(step, cwd=tree, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 start_new_session=True)
            try:
                with released() if _unlocked else contextlib.nullcontext():   # minutes; nothing shared
                    out, _ = p.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                kill_session(p.pid)
                return False, f"timeout after {timeout}s in {' '.join(step)}"
            if p.returncode != 0:
                noise = ("FORCEACTIVE", "Usage Warning", "Specified directory", "doesn't exist")
                lines = [l for l in out.splitlines() if not any(n in l for n in noise)]
                # the failing edge and its errors first: warnings often fill the tail
                key = [i for i, l in enumerate(lines) if l.startswith("FAILED:") or "Error:" in l or "rror   :" in l]
                picked = sorted({j for i in key for j in range(max(0, i - 3), min(len(lines), i + 4))})
                head = "\n".join(lines[j] for j in picked[:60])
                return False, (head + "\n...\n" if head else "") + tail("\n".join(lines), 10)
        report = tree / cfg.rel(cfg.report)
        if measures is not None and report.is_file():
            measures.update(json.loads(report.read_text()).get("measures", {}))
        return True, "builds and matches"
    finally:
        if not keep:
            _drop(cfg, tree)


def commit_overlay(cfg: Config, overlay: Overlay, message: str) -> tuple[bool, str]:
    """Commit exactly HEAD + overlay. Never touches the shared index except to sync those paths."""
    head = git(cfg.root, "rev-parse", "HEAD").stdout.strip()
    if _pinned and head != _pinned:
        return False, MOVED
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(tmp) / "index")}

        def g(*args: str, data: bytes | None = None) -> subprocess.CompletedProcess:
            return subprocess.run(["git", *args], cwd=cfg.root, env=env, input=data, capture_output=True)

        if g("read-tree", head).returncode != 0:
            return False, "read-tree failed"
        for rel, data in sorted(overlay.items()):
            if data is None:
                g("update-index", "--force-remove", "--", rel)
                continue
            blob = g("hash-object", "-w", "--stdin", data=data).stdout.decode().strip()
            mode = "100755" if (cfg.root / rel).is_file() and os.access(cfg.root / rel, os.X_OK) else "100644"
            if g("update-index", "--add", "--cacheinfo", f"{mode},{blob},{rel}").returncode != 0:
                return False, f"update-index failed for {rel}"
        tree = g("write-tree").stdout.decode().strip()
        if tree == g("rev-parse", f"{head}^{{tree}}").stdout.decode().strip():
            return False, "nothing to commit"
        commit = g("commit-tree", tree, "-p", head, data=message.encode()).stdout.decode().strip()
        if not commit or g("update-ref", "HEAD", commit, head).returncode != 0:
            return False, MOVED
    git(cfg.root, "reset", "-q", "--", *sorted(overlay))      # shared index := new HEAD for these paths
    return True, f"{commit[:9]} ({len(overlay)} file(s))"


def working_overlay(cfg: Config, paths: list[str]) -> Overlay:
    """Overlay of the working-tree content of `paths` (missing files become deletions)."""
    out: Overlay = {}
    for rel in paths:
        f = cfg.root / rel
        out[rel] = f.read_bytes() if f.is_file() else None
    return out
