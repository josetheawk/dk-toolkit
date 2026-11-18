"""dk toolkit diff|export|import: keep the shared toolkit repo and this project's copy in sync.

The toolkit (the dk launcher, tools/dk, its tests, the portable playbook) has its own repo,
so every new decomp starts from the latest one. Each project vendors a copy at the same
paths, which keeps a fresh clone self-contained. Only the owned paths are ever touched:

  diff     what differs between this project's copy and the toolkit repo (exit 1 if any)
  export   copy this project's copy into the toolkit repo (then commit there)
  import   copy the toolkit repo's version into this project (then commit here)

The toolkit repo is dk.toml [toolkit] repo (a path relative to this project's root).

Several projects write the toolkit repo, and a mirror removes what the source side lacks. So each
sync records its base in dk.toml [toolkit]: `base`, the toolkit commit synced with, and
`base_digest`, the owned files' digest right after the sync. `export` refuses when the toolkit's
owned files are no longer that state (another project exported since), `import` refuses when this
project's are not (our changes would be dropped); both print the git recipe that merges instead.
`base --set` records the current toolkit HEAD when the two copies are equal; `--force` mirrors
regardless (and records the base).
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import tomllib
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


def digest(root: Path) -> str:
    """One hash for the owned files (names and contents)."""
    h = hashlib.sha1()
    for rel, data in sorted(owned_files(root).items()):
        h.update(rel.encode() + b"\0" + hashlib.sha1(data).digest())
    return h.hexdigest()


def head(repo: Path) -> str | None:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else None


def read_base(root: Path) -> tuple[str | None, str | None]:
    """(toolkit commit, owned-files digest) of the last sync, from dk.toml [toolkit]."""
    try:
        with (root / "dk.toml").open("rb") as f:
            table = tomllib.load(f).get("toolkit", {})
    except (OSError, tomllib.TOMLDecodeError):
        return None, None
    return table.get("base"), table.get("base_digest")


def write_base(root: Path, commit: str, files: str) -> None:
    """Record the sync base in dk.toml [toolkit] (other keys and comments kept)."""
    path = root / "dk.toml"
    text = path.read_text()
    lines = {"base": f'base = "{commit}"                # dk toolkit: the toolkit commit last synced',
             "base_digest": f'base_digest = "{files}"   # dk toolkit: the owned files right after it'}
    m = re.search(r"^\[toolkit\][^\n]*\n", text, re.M)
    if m is None:
        text = text.rstrip("\n") + "\n\n[toolkit]\n" + "\n".join(lines.values()) + "\n"
    else:
        end = re.search(r"^\[", text[m.end():], re.M)
        body = text[m.end():m.end() + end.start()] if end else text[m.end():]
        for key, line in lines.items():
            if re.search(rf"^{key}\s*=", body, re.M):
                body = re.sub(rf"^{key}\s*=.*$", line.replace("\\", "\\\\"), body, count=1, flags=re.M)
            else:
                body = line + "\n" + body
        text = text[:m.end()] + body + (text[m.end() + end.start():] if end else "")
    path.write_text(text)


def recipe(cfg: Config, repo: Path, base: str | None) -> list[str]:
    """The git steps that merge this project's copy into the toolkit instead of mirroring over it."""
    branch = f"sync-{cfg.name}"
    start = base[:12] if base else "<the toolkit commit this project last synced with>"
    return [f"cd {repo}",
            f"git switch -c {branch} {start}           # the toolkit as this project last saw it",
            f"(cd {cfg.root} && ./dk toolkit export --force)   # this project's copy onto that branch",
            f"git add -A && git commit -m \"{cfg.name}: toolkit changes\"",
            "git switch - && git merge " + branch + "          # resolve any conflicts here, in the toolkit repo",
            f"(cd {cfg.root} && ./dk toolkit import --force)   # the merged toolkit back; ./dk test, then commit here"]


def _guard(cfg: Config, repo: Path, action: str) -> list[str]:
    """Why a sync would lose someone's work (empty: safe)."""
    base, files = read_base(cfg.root)
    if base is None or files is None:
        return ["no sync base recorded in dk.toml [toolkit]: if `dk toolkit diff` shows the two copies equal, "
                "record it with `dk toolkit base --set`; otherwise merge with git"]
    if action == "export":
        now = digest(repo)
        if now != files:
            moved = head(repo)
            return [f"the toolkit's owned files changed since this project's last sync (base {base[:12]}"
                    + (f", toolkit HEAD now {moved[:12]}" if moved and moved != base else "")
                    + "): another project exported; a mirror would delete or revert its work"]
    else:
        if digest(cfg.root) != files:
            return [f"this project's owned files changed since its last sync (base {base[:12]}): an import "
                    "would drop them; export them with the merge recipe first"]
    return []


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
        base, files = read_base(cfg.root)
        if base:
            print(f"sync base {base[:12]}: toolkit {'unchanged' if digest(repo) == files else 'changed'} since, "
                  f"this project {'unchanged' if digest(cfg.root) == files else 'changed'} since")
        return MISMATCH if added or changed or removed else OK
    if args.action == "base":
        base, _ = read_base(cfg.root)
        if not args.set:
            print(f"sync base: {base or '(none recorded)'}")
            return OK
        if any(plan(cfg.root, repo)) and not args.force:
            print("error: the two copies differ (dk toolkit diff); merge them first, or pass --force")
            return USAGE
        commit = head(repo)
        if commit is None:
            print(f"error: {repo} is not a git repo")
            return USAGE
        write_base(cfg.root, commit, digest(repo))
        print(f"sync base recorded: {commit[:12]} (dk.toml [toolkit]; commit it)")
        return OK
    src, dst = (cfg.root, repo) if args.action == "export" else (repo, cfg.root)
    dirty = _dirty(dst)
    if dirty and not args.force:
        print(f"error: {dst} has uncommitted changes in toolkit paths (commit them or pass --force):")
        print("\n".join(f"  {line}" for line in dirty[:20]))
        return USAGE
    why = _guard(cfg, repo, args.action)
    if why and not args.force:
        print(f"error: refusing to {args.action}: " + "; ".join(why))
        print("merge with git instead:")
        print("\n".join(f"  {line}" for line in recipe(cfg, repo, read_base(cfg.root)[0])))
        print("(--force mirrors anyway: it removes what the source side lacks)")
        return USAGE
    _report(*apply(src, dst))
    commit = head(repo)
    if commit is not None:
        write_base(cfg.root, commit, digest(repo))
        print(f"sync base recorded: {commit[:12]}" + (" (commit the exported files in the toolkit repo; the base "
                                                     "follows by content)" if args.action == "export" else ""))
    hint("test")
    return OK


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("action", choices=("diff", "export", "import", "base"))
    p.add_argument("--force", action="store_true", help="mirror even over uncommitted changes, or when the other side "
                                                        "changed since the last sync (removes what the source lacks)")
    p.add_argument("--set", action="store_true", help="with base: record the toolkit's HEAD as the sync base "
                                                      "(the copies must be equal)")


COMMAND = Command("toolkit", "docs", "sync the shared toolkit repo with this project's copy (diff, export, import, "
                  "base); refuses a sync that would lose another side's changes",
                  run, _configure)
