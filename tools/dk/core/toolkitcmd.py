"""dk toolkit diff|export|import: keep the shared toolkit repo and this project's copy in sync.

The toolkit (the dk launcher, tools/dk, its tests, the portable playbook) has its own repo,
so every new decomp starts from the latest one. Each project vendors a copy at the same
paths, which keeps a fresh clone self-contained. Only the owned paths are ever touched:

  diff     what differs between this project's copy and the toolkit repo (exit 1 if any)
  export   copy this project's copy into the toolkit repo (then commit there)
  import   copy the toolkit repo's version into this project (then commit here)

The toolkit repo is dk.toml [toolkit] repo (a path relative to this project's root).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from .command import MISMATCH, OK, USAGE, Command, hint
from .config import Config

OWNED = ("dk", "tools/dk", "tests", "playbook")
_SKIP_PARTS = {"__pycache__", ".pytest_cache"}


def owned_files(root: Path) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for rel in OWNED:
        path = root / rel
        if path.is_file():
            out[rel] = path.read_bytes()
        elif path.is_dir():
            for f in sorted(path.rglob("*")):
                if f.is_file() and not _SKIP_PARTS & set(f.relative_to(root).parts) and f.suffix != ".pyc":
                    out[str(f.relative_to(root))] = f.read_bytes()
    return out


def plan(src: Path, dst: Path) -> tuple[list[str], list[str], list[str]]:
    """(added, changed, removed) to make dst's owned files equal src's."""
    a, b = owned_files(src), owned_files(dst)
    added = sorted(set(a) - set(b))
    changed = sorted(k for k in set(a) & set(b) if a[k] != b[k])
    removed = sorted(set(b) - set(a))
    return added, changed, removed


def apply(src: Path, dst: Path) -> tuple[list[str], list[str], list[str]]:
    added, changed, removed = plan(src, dst)
    for rel in added + changed:
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / rel, target)
    for rel in removed:
        (dst / rel).unlink()
    return added, changed, removed


def _dirty(root: Path) -> list[str]:
    """Uncommitted changes under the owned paths (git), so a sync never clobbers work."""
    proc = subprocess.run(["git", "status", "--porcelain", "--", *OWNED], cwd=root, capture_output=True, text=True)
    return [line for line in proc.stdout.splitlines() if line.strip()] if proc.returncode == 0 else []


def _report(added, changed, removed) -> None:
    for tag, items in (("added", added), ("changed", changed), ("removed", removed)):
        for rel in items:
            print(f"  {tag:<8} {rel}")
    print(f"{len(added)} added, {len(changed)} changed, {len(removed)} removed")


def run(cfg: Config, args: argparse.Namespace) -> int:
    repo = cfg.toolkit_repo
    if repo is None:
        print("error: dk.toml has no [toolkit] repo")
        return USAGE
    if not (repo / "tools" / "dk").is_dir() and args.action != "export":
        print(f"error: {repo} is not a toolkit repo (no tools/dk)")
        return USAGE
    if args.action == "diff":
        added, changed, removed = plan(cfg.root, repo)
        _report(added, changed, removed)
        return MISMATCH if added or changed or removed else OK
    src, dst = (cfg.root, repo) if args.action == "export" else (repo, cfg.root)
    dirty = _dirty(dst)
    if dirty and not args.force:
        print(f"error: {dst} has uncommitted changes in toolkit paths (commit them or pass --force):")
        print("\n".join(f"  {line}" for line in dirty[:20]))
        return USAGE
    _report(*apply(src, dst))
    hint("test")
    return OK


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("action", choices=("diff", "export", "import"))
    p.add_argument("--force", action="store_true", help="overwrite uncommitted changes in the destination")


COMMAND = Command("toolkit", "docs", "sync the shared toolkit repo with this project's copy (diff, export, import)",
                  run, _configure)
