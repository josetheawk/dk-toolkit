"""Commits that are exactly "HEAD + these files", proven to build before they land.

verify_overlay()  builds HEAD plus an overlay of file contents in a throwaway git worktree
                  (original binary and downloaded tools are symlinked in) and checks the hash.
commit_overlay()  writes the overlay with git plumbing through a private index, so the
                  shared index is never touched; HEAD advances with a compare-and-swap, so a
                  concurrent commit can never be overwritten. The shared index is then synced
                  for the committed paths only.
Inside a dk worktree (landing.py) the same calls land on the shared branch instead: pin_head
syncs the worktree to the shared tip first, and commit_overlay swaps the shared ref, copies the
landed paths into the other checkouts of that branch, then moves the worktree's branch forward.
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
_holding = False     # this attempt holds the land lock through its verify
MOVED = "HEAD moved concurrently; retry"
UNLOCKED_TRIES = 2   # then verify holding the lock: commits take it too, so HEAD cannot move
TOOLKIT = Path(__file__).resolve().parents[1].relative_to(Path(__file__).resolve().parents[3]).as_posix()   # tools/dk


class LandError(Exception):
    """A dk worktree could not be brought up to the shared branch (its sync conflicts)."""


def pin_head(cfg: Config, attempt: int = 0) -> str:
    """Pin HEAD for attempt `attempt` of a verify-and-commit. Early attempts build without the
    project lock (other agents keep committing); a later one keeps it, and the land lock that
    every commit takes (dk worktrees share it), so it always lands. In a dk worktree the worktree
    is first synced to the shared branch's tip (LandError if that conflicts)."""
    global _pinned, _unlocked, _holding
    from . import landing
    if _holding:
        landing.release()
        _holding = False
    _unlocked = attempt < UNLOCKED_TRIES
    if not _unlocked:
        landing.hold(cfg.root)
        _holding = True
    if landing.meta(cfg.root):
        before = git(cfg.root, "rev-parse", "HEAD").stdout.strip()
        ok, detail = landing.sync(cfg.root)
        after = git(cfg.root, "rev-parse", "HEAD").stdout.strip()
        if ok and before != after and git(cfg.root, "diff", "--name-only", before, after, "--", TOOLKIT).stdout.strip():
            # this process runs the toolkit it loaded before the sync, which may not know the formats
            # the new commits wrote (a units.toml table): stop instead of composing with it
            ok, detail = False, (f"the sync brought in toolkit changes ({TOOLKIT}): this process still runs the old "
                                 "toolkit; run the same command again")
        if not ok:
            if _holding:
                landing.release()
                _holding = False
            raise LandError(detail)
    _pinned = git(cfg.root, "rev-parse", "HEAD").stdout.strip()
    return _pinned


def show(cfg: Config, rel: str) -> bytes | None:
    """`rel` at the pinned commit (HEAD if none is pinned), or None if it does not exist there."""
    proc = subprocess.run(["git", "show", f"{_pinned or 'HEAD'}:{rel}"], cwd=cfg.root, capture_output=True)
    return proc.stdout if proc.returncode == 0 else None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def sweep_dead(cfg: Config) -> list[str]:
    """Drop `verify-<pid>` worktrees whose process is gone (a killed verify leaves ~60 MB behind)."""
    gone = []
    for tree in sorted((cfg.root / "build" / "dk").glob("verify-*")):
        pid = tree.name.removeprefix("verify-")
        if pid.isdigit() and int(pid) != os.getpid() and not _alive(int(pid)):
            _drop(cfg, tree)
            gone.append(tree.name)
    return gone


def _worktree(cfg: Config, name: str) -> Path:
    tree = cfg.root / "build" / "dk" / name
    sweep_dead(cfg)
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


TOOLS = (("--dtk", "dtk"), ("--objdiff", "objdiff-cli"), ("--wrapper", "wibo"), ("--sjiswrap", "sjiswrap.exe"))


def main_worktree(root: Path) -> Path:
    """The repository's main checkout (`root` itself outside a linked worktree)."""
    proc = subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=root, capture_output=True, text=True)
    first = next((l for l in proc.stdout.splitlines() if l.startswith("worktree ")), "")
    return Path(first.removeprefix("worktree ")) if first else Path(root)


def tool_args(root: Path) -> list[str]:
    """configure arguments pointing at already-downloaded tools: this checkout's build/tools and
    build/compilers, else the main checkout's (a linked worktree shares them).

    Point configure at the tools instead of symlinking build/tools into a fresh tree: a fresh
    checkout makes download_tool.py look newer, and ninja would re-download *through* a symlink
    into the shared tools dir while other builds are running them."""
    homes = [Path(root) / "build"]
    main = main_worktree(root) / "build"
    if main != homes[0]:
        homes.append(main)
    out = []
    for flag, name in TOOLS:
        found = next((h / "tools" / name for h in homes if (h / "tools" / name).is_file()), None)
        if found is not None:
            out += [flag, str(found.resolve())]
    compilers = next((h / "compilers" for h in homes if (h / "compilers").is_dir()), None)
    if compilers is not None:
        out += ["--compilers", str(compilers.resolve())]
    return out


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
        if not original.is_relative_to(tree):    # never unlink a file outside the throwaway tree
            return False, f"the original binary {cfg.original} is not inside the project root"
        original.parent.mkdir(parents=True, exist_ok=True)
        original.unlink(missing_ok=True)
        original.symlink_to(cfg.original.resolve())
        for step in ([sys.executable, *cfg.configure, *tool_args(cfg.root)], ["ninja"]):
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
            data = json.loads(report.read_text())
            measures.update(data.get("measures", {}))
            # per unit, for the floor's table (an integration holds back a unit that fell below it)
            measures["units"] = {u["name"]: int(u.get("measures", {}).get("matched_functions", 0) or 0)
                                 for u in data.get("units", []) if u.get("name")}
        return True, "builds and matches"
    finally:
        if not keep:
            _drop(cfg, tree)


def commit_overlay(cfg: Config, overlay: Overlay, message: str) -> tuple[bool, str]:
    """Commit exactly HEAD + overlay. Never touches the shared index except to sync those paths.
    In a dk worktree the commit lands on the shared branch (see landing.py)."""
    global _holding
    from . import landing
    try:
        with landing.locked(cfg.root):
            return _commit_locked(cfg, overlay, message)
    finally:
        if _holding:                    # the final attempt's hold ends with its commit
            landing.release()
            _holding = False


def _commit_locked(cfg: Config, overlay: Overlay, message: str) -> tuple[bool, str]:
    from . import landing
    head = git(cfg.root, "rev-parse", "HEAD").stdout.strip()
    if _pinned and head != _pinned:
        return False, MOVED
    worktree = landing.meta(cfg.root)
    # the ref the commit lands on: the shared branch from a dk worktree, else this checkout's own
    target = worktree["shared"] if worktree else (git(cfg.root, "symbolic-ref", "-q", "HEAD").stdout.strip() or "HEAD")
    from . import frozen
    refusals = frozen.guard(cfg, overlay, lambda rel: _blob(cfg, head, rel))
    if refusals:
        return False, "frozen function edited:\n  " + "\n  ".join(refusals)
    if worktree and git(cfg.root, "rev-parse", target).stdout.strip() != head:
        # the shared branch moved past what this was composed on
        return False, MOVED if _pinned else f"{target} moved past this worktree: ./dk worktree sync, then retry"
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
        if not commit:
            return False, "commit-tree failed"
        # other checkouts of the target branch (the main checkout, from a dk worktree) would be
        # left with stale copies that a later commit there reverts: refuse what can't be carried in
        copies, refusals = landing.plan_copy(cfg.root, target, head, commit, sorted(overlay)) \
            if target != "HEAD" else ([], [])
        if refusals:
            return False, "not landed: " + "; ".join(refusals)
        if g("update-ref", target if worktree else "HEAD", commit, head).returncode != 0:
            return False, MOVED
        if worktree and g("update-ref", "HEAD", commit, head).returncode != 0:
            print(f"note: landed on {target}, but this worktree's branch moved meanwhile: ./dk worktree sync")
    for note in landing.apply_copy(copies):
        print(f"note: {note}")
    git(cfg.root, "reset", "-q", "--", *sorted(overlay))      # shared index := new HEAD for these paths
    where = f" on {target.removeprefix('refs/heads/')}" if worktree else ""
    return True, f"{commit[:9]}{where} ({len(overlay)} file(s))"


def frozen_refusal(cfg: Config, overlay: Overlay) -> str | None:
    """Checked before the verify build too, so a frozen edit is refused in seconds, not after a build."""
    from . import frozen
    refusals = frozen.guard(cfg, overlay, lambda rel: show(cfg, rel))
    return "frozen function edited:\n  " + "\n  ".join(refusals) if refusals else None


def _blob(cfg: Config, rev: str, rel: str) -> bytes | None:
    p = subprocess.run(["git", "show", f"{rev}:{rel}"], cwd=cfg.root, capture_output=True)
    return p.stdout if p.returncode == 0 else None


def working_overlay(cfg: Config, paths: list[str]) -> Overlay:
    """Overlay of the working-tree content of `paths` (missing files become deletions)."""
    out: Overlay = {}
    for rel in paths:
        f = cfg.root / rel
        out[rel] = f.read_bytes() if f.is_file() else None
    return out
