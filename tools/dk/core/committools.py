"""dk commit-tools PATH... -m MSG: commit toolkit or docs changes, proven first.

The unit commands (accept, save) commit source; this commits everything else a maintainer
changes (tools/dk, tests, playbook, docs, dk.toml). In order, refusing at the first failure:
  1. the toolkit tests pass
  2. doc-lint passes
  3. HEAD plus exactly these paths builds and matches in a clean worktree
  4. that build is not below the committed progress floor (which it raises)
Then the paths are committed like a unit's delta (private index, compare-and-swap on HEAD).

--integrate commits the decomp's whole working state instead (every changed file under src/,
include/ and config/, shared config files included; `--exclude` leaves a file mid-edit out),
through the same gates. Toolkit and doc changes stay out: their authors commit them when done,
and the toolkit tests and doc-lint run only when such files are part of the commit. It is for
the coordinator: interdependent work (a header rename with the units and symbol renames that
depend on it) lands together.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from . import doclint, floor, isolated
from .command import MISMATCH, OK, USAGE, Command
from .config import Config


INTEGRATED = ("src/", "include/", "config/")   # what --integrate carries: the decomp itself


def _working_changes(cfg: Config) -> list[str]:
    tracked = subprocess.run(["git", "diff", "--name-only", "HEAD"], cwd=cfg.root, capture_output=True, text=True).stdout.split()
    roots = [cfg.platform_options.get("src", "src"), "include"]
    new = subprocess.run(["git", "ls-files", "--others", "--exclude-standard", *roots], cwd=cfg.root,
                         capture_output=True, text=True).stdout.split()
    return sorted(set(tracked) | set(new))


def _failed_source(cfg: Config, detail: str) -> str | None:
    """The source of the first object whose compile failed in a clean build's output."""
    import re
    m = re.search(r"FAILED: (?:\[code=\d+\] )?build/[^/]+/(src/\S+)\.o", detail)
    if not m:
        return None
    stem = cfg.root / m[1]
    for ext in (".cpp", ".cp", ".c"):
        if stem.with_suffix(ext).is_file():
            return cfg.rel(stem.with_suffix(ext))
    return None


def run(cfg: Config, args: argparse.Namespace) -> int:
    if not args.message:
        print("error: give -m MESSAGE")
        return USAGE
    if args.integrate:
        # decomp work only: toolkit and doc changes are their authors' to commit when finished
        work = [p for p in _working_changes(cfg) if p.startswith(INTEGRATED) and p not in args.exclude]
        args.paths = sorted(set(args.paths) | set(work))
        print(f"integrating {len(args.paths)} changed file(s) of the working tree")
    if not args.paths:
        print("error: give PATH... or --integrate")
        return USAGE
    # the shared config files hold every agent's in-progress entries: committing one whole
    # would land other agents' unfinished work (unit commits compose them entry by entry)
    shared = {str(v) for k, v in cfg.platform_options.items() if k in ("units", "splits", "symbols")}
    blocked = [p for p in args.paths if p in shared and not args.integrate]
    if blocked:
        print(f"REFUSED: {', '.join(blocked)} must change through ./dk accept, save, unit or rename commands")
        return USAGE
    tooling = [p for p in args.paths if not p.startswith(INTEGRATED)]
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=cfg.root, capture_output=True, text=True) \
        if tooling else None
    if tests is not None and tests.returncode != 0:
        print("REFUSED: the toolkit tests fail:")
        print("\n".join(tests.stdout.strip().splitlines()[-15:]))
        return MISMATCH
    problems = doclint.lint(cfg) if tooling else []
    if problems:
        print("REFUSED: doc-lint fails:")
        for p in problems[:15]:
            print(f"  {p}")
        return MISMATCH
    message = args.message
    for trailer in args.trailer:
        message += ("\n\n" if trailer == args.trailer[0] else "\n") + trailer
    for attempt in range(isolated.UNLOCKED_TRIES + 1):   # redo on a HEAD that moved meanwhile
        isolated.pin_head(cfg, attempt)
        code, detail = _verify_and_commit(cfg, args, message)
        if detail != isolated.MOVED:
            break
    print(f"committed: {detail}" if code == OK else f"REFUSED: {detail}")
    return code


def _verify_and_commit(cfg: Config, args: argparse.Namespace, message: str) -> tuple[int, str]:
    overlay = isolated.working_overlay(cfg, args.paths)
    measures: dict = {}
    ok, detail = isolated.verify_overlay(cfg, overlay, measures=measures)
    held = []
    while not ok and args.integrate and len(held) < 8:
        # an agent mid-edit: keep HEAD's version of a source that does not compile (a new one
        # stays out), and integrate the rest; unlinked units' objects are not in the binary
        src = _failed_source(cfg, detail)
        if src is None or src not in overlay or src in held:
            break
        head = isolated.show(cfg, src)
        if head is not None:
            overlay[src] = head
        else:
            overlay[src] = f"// {src}: in progress (held back from an integration commit)\n".encode()
        held.append(src)
        print(f"holding back {src} (does not compile on its own yet)")
        measures = {}
        ok, detail = isolated.verify_overlay(cfg, overlay, measures=measures)
    if not ok:
        return MISMATCH, f"HEAD plus these paths does not build and match on its own:\n{detail}"
    below, floor_text = floor.gate(cfg, measures)
    if below:
        return MISMATCH, "below the progress floor: " + "; ".join(below)
    overlay[cfg.rel(floor.path(cfg))] = floor_text.encode()
    ok, detail = isolated.commit_overlay(cfg, overlay, message)
    if ok:
        floor.path(cfg).write_text(floor_text)
    return (OK if ok else MISMATCH), detail


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("paths", nargs="*", help="files to commit (their working-tree content)")
    p.add_argument("--exclude", action="append", default=[], metavar="PATH",
                   help="with --integrate: leave this path out (keep HEAD's version), e.g. an agent's file mid-edit")
    p.add_argument("--integrate", action="store_true",
                   help="commit the whole working state, shared config files included (maintainers, no agents running)")
    p.add_argument("-m", "--message", help="commit message")
    p.add_argument("--trailer", action="append", default=[], help="trailer line (repeatable), e.g. Co-Authored-By: ...")


COMMAND = Command("commit-tools", "docs", "commit toolkit/docs changes after tests, doc-lint and a clean-worktree build",
                  run, _configure, locked=True)
