"""dk worktree add|list|remove|sync: one git worktree per agent.

An agent working in its own worktree cannot break another agent's build with a half-done edit.
`add <name>` makes worktrees/<name> (git-ignored, under the main checkout) on branch dk/<name>,
started from the shared branch (the one the main checkout has checked out) or `--from REV`:
  - a copy of the original binary (a copy, never a link: a build tree must not be able to
    touch the main checkout's file); .venv, build/tools and build/compilers linked to the main
    checkout's
  - configured with those tools (so ninja never downloads into the shared tools dir), built and
    hash-checked, then the platform's setup steps run (its WORKTREE_SETUP, e.g. `analyze rtti`)
`list` shows each worktree's branch, how far the shared branch has moved past it, and its
uncommitted files. `remove <name>` refuses while the worktree has uncommitted changes
(`--force` drops them); it deletes dk/<name> when the shared branch contains it.
`sync` (inside a worktree) brings it up to the shared branch: a per-path 3-way merge of what
others landed into your files, refused as a whole on any conflict (landing.sync). save, accept
and commit-tools sync on their own before they land; run it to pick up others' work sooner.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .. import platforms
from . import landing
from .command import MISMATCH, OK, USAGE, Command, hint
from .config import Config
from .isolated import main_worktree, tool_args

NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
LINKED = (".venv", "build/tools", "build/compilers")   # shared with the main checkout


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)


def _path(cfg: Config, name: str) -> Path:
    return main_worktree(cfg.root) / landing.DIR / name


def _short(ref: str | None) -> str:
    return (ref or "").removeprefix("refs/heads/")


def changes(tree: Path) -> list[str]:
    """Uncommitted tracked changes and untracked, not-ignored files of a checkout."""
    return [l[3:] for l in _git(tree, "status", "--porcelain", "--untracked-files=all").stdout.splitlines() if l]


def provision(cfg: Config, tree: Path) -> list[str]:
    """Put the original binary and the shared tool links into a fresh worktree. Returns notes."""
    main = main_worktree(cfg.root)
    notes = []
    rel = cfg.rel(cfg.original)
    source = next((p for p in (main / rel, cfg.original) if p.is_file()), None)
    if source is None:
        notes.append(f"no original binary at {rel} in the main checkout: run ./dk setup there first")
    elif not Path(rel).is_absolute():
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, tree / rel)
    for link in LINKED:
        target = main / link
        if not target.exists():
            continue
        dest = tree / link
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() and not dest.is_symlink():
            dest.symlink_to(target.resolve())
    return notes


def setup_steps(cfg: Config, tree: Path) -> list[list[str]]:
    """configure (with the shared tools), `dk build`, then the platform's own steps."""
    dk = [sys.executable, "-m", "tools.dk"]
    extra = getattr(platforms.load(cfg.platform), "WORKTREE_SETUP", ())
    return [[sys.executable, *cfg.configure, *tool_args(tree)], [*dk, "build"], *([*dk, *step] for step in extra)]


def _add(cfg: Config, args: argparse.Namespace) -> int:
    if not NAME.match(args.name):
        print(f"error: worktree name {args.name!r}: use letters, digits, '.', '_' or '-'")
        return USAGE
    shared = landing.shared_ref(cfg.root)
    if shared is None:
        print("error: the main checkout has no branch checked out (detached HEAD): nothing to land on")
        return USAGE
    tree, branch = _path(cfg, args.name), landing.BRANCH + args.name
    if tree.exists():
        print(f"error: {tree} already exists")
        return USAGE
    exists = _git(cfg.root, "rev-parse", "--verify", "-q", f"refs/heads/{branch}").returncode == 0
    if exists and args.rev:
        print(f"error: branch {branch} already exists; pick another name (or drop --from to resume it)")
        return USAGE
    start = args.rev or shared
    cmd = ["worktree", "add", str(tree), branch] if exists else ["worktree", "add", "-b", branch, str(tree), start]
    proc = _git(cfg.root, *cmd)
    if proc.returncode != 0:
        print(f"error: git {' '.join(cmd)}: {proc.stderr.strip()}")
        return USAGE
    landing.write_meta(tree, {"name": args.name, "shared": shared})
    for note in provision(cfg, tree):
        print(f"note: {note}")
    head = _git(tree, "rev-parse", "--short", "HEAD").stdout.strip()
    print(f"worktree {cfg.rel(tree) if tree.is_relative_to(cfg.root) else tree}: branch {branch} at {head}"
          f"{' (resumed)' if exists else ''}, lands on {_short(shared)}")
    if not args.no_build:
        for step in setup_steps(cfg, tree):
            label = " ".join(step[1:] if step[0] == sys.executable else step).replace("-m tools.dk ", "dk ", 1)
            print(f"  {label.split(' --')[0]}")
            run = subprocess.run(step, cwd=tree, capture_output=True, text=True)
            if run.returncode != 0:
                print(f"error: `{label}` failed in {tree} (exit {run.returncode}):")
                print("\n".join((run.stdout + run.stderr).strip().splitlines()[-15:]))
                return run.returncode if run.returncode in (MISMATCH, 3) else USAGE
    print(f"next: cd {tree} && ./dk status")
    return OK


def _list(cfg: Config, args: argparse.Namespace) -> int:
    shared = landing.shared_ref(cfg.root)
    tip = _git(cfg.root, "rev-parse", "--short", shared).stdout.strip() if shared else ""
    print(f"shared branch: {_short(shared) or '(detached)'} at {tip}")
    rows = [w for w in landing.worktrees(cfg.root) if w["meta"]]
    if not rows:
        print("no dk worktrees")
        hint("worktree add <name>")
        return OK
    print(f"{'name':<20} {'branch':<24} {'head':<10} {'behind':>6} {'changed':>7}  path")
    for w in sorted(rows, key=lambda w: w["meta"].get("name", "")):
        head = w["head"][:9]
        behind = _git(cfg.root, "rev-list", "--count", f"{w['head']}..{w['meta'].get('shared', shared)}").stdout.strip() or "?"
        print(f"{w['meta'].get('name', '?'):<20} {_short(w['branch']):<24} {head:<10} {behind:>6} "
              f"{len(changes(w['path'])):>7}  {w['path']}")
    hint("worktree add <name>", "worktree remove <name>")
    return OK


def _remove(cfg: Config, args: argparse.Namespace) -> int:
    tree = _path(cfg, args.name)
    row = next((w for w in landing.worktrees(cfg.root) if w["path"] == tree), None)
    if row is None or not row["meta"]:
        print(f"error: no dk worktree named {args.name} ({tree})")
        return USAGE
    if Path(cfg.root).resolve() == tree.resolve():
        print("error: run remove from another checkout (the main one), not from inside the worktree")
        return USAGE
    dirty = changes(tree)
    if dirty and not args.force:
        print(f"REFUSED: {tree} has {len(dirty)} uncommitted change(s): {', '.join(dirty[:8])}"
              f"{' ...' if len(dirty) > 8 else ''}")
        print("land them (./dk save / commit-tools inside it) or pass --force to drop them")
        return MISMATCH
    proc = _git(cfg.root, "worktree", "remove", "--force", str(tree))
    if proc.returncode != 0:
        print(f"error: git worktree remove: {proc.stderr.strip()}")
        return USAGE
    _git(cfg.root, "worktree", "prune")          # its own verify trees, if any
    branch, shared = row["branch"], row["meta"].get("shared") or landing.shared_ref(cfg.root)
    note = ""
    if branch and branch.startswith("refs/heads/" + landing.BRANCH):
        ahead = _git(cfg.root, "rev-list", "--count", f"{shared}..{branch}").stdout.strip()
        if ahead == "0":
            _git(cfg.root, "branch", "-D", _short(branch))
            note = f"; deleted branch {_short(branch)}"
        else:
            note = f"; kept branch {_short(branch)} ({ahead} commit(s) not on {_short(shared)})"
    print(f"removed {tree}{note}")
    hint("worktree list")
    return OK


def _sync(cfg: Config, args: argparse.Namespace) -> int:
    if not landing.meta(cfg.root):
        print("error: not inside a dk worktree (make one with ./dk worktree add <name>, then run this there)")
        return USAGE
    keep = tuple(os.path.normpath(p) for p in args.keep_mine)     # paths from the worktree's root
    before = _git(cfg.root, "rev-parse", "HEAD").stdout.strip()
    ok, detail = landing.sync(cfg.root, keep)
    print(detail if ok else f"REFUSED: {detail}")
    if ok:
        after = _git(cfg.root, "rev-parse", "HEAD").stdout.strip()
        changed = _git(cfg.root, "diff", "--name-only", before, after, "--", "config").stdout.split() if before != after else []
        if changed:            # symbols or splits moved: build outputs made from them are stale
            refresh = getattr(platforms.load(cfg.platform), "after_config_sync", None)
            if refresh is not None and refresh(cfg):
                print(f"config changed ({', '.join(changed[:3])}{' …' if len(changed) > 3 else ''}): "
                      "the build's split outputs were regenerated")
            else:
                print(f"config changed ({', '.join(changed[:3])}): build outputs made from it are stale")
                hint("build")
                return OK
        hint("status")
        return OK
    hint("worktree sync --keep-mine PATH")
    return MISMATCH


def run(cfg: Config, args: argparse.Namespace) -> int:
    return {"add": _add, "list": _list, "remove": _remove, "sync": _sync}[args.action](cfg, args)


def _configure(p: argparse.ArgumentParser) -> None:
    sub = p.add_subparsers(dest="action", required=True)
    a = sub.add_parser("add", help="make worktrees/<name> on branch dk/<name>, set up and built")
    a.add_argument("name")
    a.add_argument("--from", dest="rev", metavar="REV", help="start from REV instead of the shared branch's tip")
    a.add_argument("--no-build", action="store_true", help="skip configure, build and the platform's setup steps")
    sub.add_parser("list", help="dk worktrees: branch, commits behind the shared branch, uncommitted files")
    r = sub.add_parser("remove", help="delete a worktree (refused while it has uncommitted changes)")
    r.add_argument("name")
    r.add_argument("--force", action="store_true", help="drop its uncommitted changes")
    s = sub.add_parser("sync", help="inside a worktree: 3-way merge the shared branch's new commits into it "
                                    "(refused as a whole on any conflict)")
    s.add_argument("--keep-mine", action="append", default=[], metavar="PATH",
                   help="keep your version of PATH where both sides changed it (repeatable)")


COMMAND = Command("worktree", "setup",
                  "per-agent worktrees: add <name> [--from REV] / list / remove <name> / sync [--keep-mine PATH]",
                  run, _configure)
