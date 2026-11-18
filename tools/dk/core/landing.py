"""Per-agent worktrees and landing their commits on the shared branch.

A dk worktree (`dk worktree add`) is a linked git worktree on its own branch `dk/<name>`, with
`dk-worktree.json` in its git dir naming the shared branch (the one the main checkout has
checked out). Each agent edits only its own worktree, so a half-edit never breaks another's build.

Landing (isolated.pin_head / commit_overlay, used by save, accept, commit-tools, rm):
  1. sync(): bring the worktree up to the shared tip, a per-path 3-way merge of what others
     landed into the agent's files; any conflict refuses the whole sync and changes nothing
  2. compose tip + the agent's delta and verify it in a clean tree (isolated.verify_overlay)
  3. under the land lock (a file in the git common dir): compare-and-swap the shared ref from
     the tip to the new commit, copy the landed paths into every other checkout of the shared
     branch (plan_copy: refused if one has those paths staged differently or edits that do not
     merge), then move the worktree's own branch forward
The verified tree is the landed tree. A lander that loses the race re-syncs and retries; the
last attempt holds the land lock through its verify, so it always lands.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from .isolated import main_worktree

META = "dk-worktree.json"
DIR = "worktrees"          # under the main checkout, git-ignored
BRANCH = "dk/"             # branch prefix
LOCK = "dk-land.lock"      # in the git common dir: one lander moves the shared branch at a time
ZERO = "0" * 40


def _git(root: Path, *args: str, data: bytes | None = None, text: bool = True) -> subprocess.CompletedProcess:
    """git in `root`, with literal pathspecs; a held index.lock (another process's status) is retried."""
    for _ in range(60):
        proc = subprocess.run(["git", "--literal-pathspecs", *args], cwd=root, capture_output=True,
                              input=data if not text else (data.decode() if data is not None else None), text=text)
        err = proc.stderr if text else proc.stderr.decode(errors="replace")
        if proc.returncode == 0 or "index.lock" not in err:
            return proc
        time.sleep(0.25)
    return proc


def _rev(root: Path, rev: str) -> str:
    return _git(root, "rev-parse", "--verify", "-q", rev).stdout.strip()


def git_dir(root: Path) -> Path:
    out = _git(root, "rev-parse", "--absolute-git-dir").stdout.strip()
    return Path(out) if out else Path(root) / ".git"


def common_dir(root: Path) -> Path:
    out = _git(root, "rev-parse", "--path-format=absolute", "--git-common-dir").stdout.strip()
    return Path(out) if out else Path(root) / ".git"


def meta(root: Path) -> dict | None:
    """This checkout's worktree metadata, or None outside a dk worktree."""
    path = git_dir(root) / META
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def write_meta(root: Path, data: dict) -> None:
    (git_dir(root) / META).write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")


def shared_ref(root: Path) -> str | None:
    """The branch the main checkout has checked out (what dk worktrees land on)."""
    return _git(main_worktree(root), "symbolic-ref", "-q", "HEAD").stdout.strip() or None


def worktrees(root: Path, with_meta: bool = True) -> list[dict]:
    """Every linked worktree of the repository: {path, head, branch, meta}, sorted by path."""
    out, entry = [], {}
    for line in _git(root, "worktree", "list", "--porcelain").stdout.splitlines() + [""]:
        if not line:
            if entry:
                out.append(entry)
            entry = {}
            continue
        key, _, value = line.partition(" ")
        if key == "worktree":
            entry = {"path": Path(value), "head": "", "branch": None}
        elif key == "HEAD":
            entry["head"] = value
        elif key == "branch":
            entry["branch"] = value
    rows = sorted(out[1:], key=lambda e: str(e["path"]))
    for e in rows:
        e["meta"] = meta(e["path"]) if with_meta and e["path"].is_dir() else None
    return rows


def checkouts_of(root: Path, ref: str) -> list[Path]:
    """Checkouts that have `ref` checked out (the main checkout, for the shared branch)."""
    main = main_worktree(root)
    rows = [{"path": main, "branch": _git(main, "symbolic-ref", "-q", "HEAD").stdout.strip()}] + worktrees(root, False)
    return [r["path"] for r in rows if r["branch"] == ref and r["path"].is_dir()]


# --- the land lock ---------------------------------------------------------------------------

_lock: dict = {"file": None, "depth": 0}


def hold(root: Path) -> None:
    """Take the land lock (reentrant within this process)."""
    if _lock["depth"] == 0:
        f = open(common_dir(root) / LOCK, "w")
        fcntl.flock(f, fcntl.LOCK_EX)
        _lock["file"] = f
    _lock["depth"] += 1


def release() -> None:
    if _lock["depth"] == 0:
        return
    _lock["depth"] -= 1
    if _lock["depth"] == 0:
        fcntl.flock(_lock["file"], fcntl.LOCK_UN)
        _lock["file"].close()
        _lock["file"] = None


@contextlib.contextmanager
def locked(root: Path):
    hold(root)
    try:
        yield
    finally:
        release()


# --- blobs and files -------------------------------------------------------------------------

Entry = tuple[str, str]            # (mode, sha)


def _chunks(items: list[str], n: int = 400):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def entries(root: Path, rev: str, paths: list[str]) -> dict[str, Entry]:
    """{path: (mode, sha)} at `rev` for those of `paths` that exist there."""
    out = {}
    for part in _chunks(sorted(paths)):
        for rec in _git(root, "ls-tree", "-z", "--full-tree", rev, "--", *part).stdout.split("\0"):
            if rec:
                info, rel = rec.split("\t", 1)
                mode, kind, sha = info.split()
                if kind == "blob":
                    out[rel] = (mode, sha)
    return out


def staged(root: Path, paths: list[str]) -> dict[str, Entry]:
    """{path: (mode, sha)} of a checkout's index (stage 0) for those of `paths` it holds."""
    out = {}
    for part in _chunks(sorted(paths)):
        for rec in _git(root, "ls-files", "-s", "-z", "--", *part).stdout.split("\0"):
            if rec:
                info, rel = rec.split("\t", 1)
                mode, sha, stage = info.split()
                out[rel] = (mode, sha) if stage == "0" else ("conflict", sha)
    return out


def contents(root: Path, shas: set[str]) -> dict[str, bytes]:
    """{sha: bytes} for blobs, in one `cat-file --batch`."""
    shas = sorted(shas)
    if not shas:
        return {}
    raw = _git(root, "cat-file", "--batch", data=("\n".join(shas) + "\n").encode(), text=False).stdout
    out, pos = {}, 0
    for sha in shas:
        end = raw.index(b"\n", pos)
        size = int(raw[pos:end].split()[2])
        out[sha] = raw[end + 1:end + 1 + size]
        pos = end + 1 + size + 1
    return out


def read_work(path: Path) -> bytes | None:
    """A working file as git would store it (a symlink's target), or None if it is not there."""
    if path.is_symlink():
        return os.readlink(path).encode()
    return path.read_bytes() if path.is_file() else None


def write_work(path: Path, entry: Entry | None, data: bytes | None) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    if data is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = entry[0] if entry else "100644"
    if mode == "120000":
        path.symlink_to(data.decode())
        return
    path.write_bytes(data)
    os.chmod(path, 0o755 if mode == "100755" else 0o644)


def merge(mine: bytes, base: bytes, theirs: bytes) -> bytes | None:
    """git merge-file of three versions: the merged bytes, or None on a conflict."""
    with tempfile.TemporaryDirectory() as tmp:
        names = []
        for name, data in (("mine", mine), ("base", base), ("theirs", theirs)):
            (Path(tmp) / name).write_bytes(data)
            names.append(str(Path(tmp) / name))
        proc = subprocess.run(["git", "merge-file", "-p", *names], capture_output=True)
    return proc.stdout if proc.returncode == 0 else None


def _resolve(rel: str, mine: bytes | None, base: bytes | None, theirs: bytes | None) -> tuple[str, bytes | None]:
    """How a path both sides may have changed ends up: ("theirs" | "same" | "merged" | "conflict", data)."""
    if mine == base:
        return "theirs", theirs
    if mine == theirs:
        return "same", mine
    if mine is None or base is None or theirs is None:
        return "conflict", None             # added or deleted on one side, changed on the other
    merged = merge(mine, base, theirs)
    return ("merged", merged) if merged is not None else ("conflict", None)


# --- sync ------------------------------------------------------------------------------------

def sync(root: Path, keep_mine: tuple[str, ...] = ()) -> tuple[bool, str]:
    """Bring a dk worktree up to the shared branch's tip: for every path the shared branch changed
    since the worktree's HEAD, a 3-way merge of the working file (mine), HEAD's (base) and the
    tip's (theirs). Any conflict refuses the whole sync: nothing is written and HEAD stays.
    `keep_mine`: paths whose working version is kept as it is where both sides changed them."""
    m = meta(root)
    if not m:
        return False, "not a dk worktree (make one with ./dk worktree add <name>)"
    head, tip = _rev(root, "HEAD"), _rev(root, m["shared"])
    if not tip:
        return False, f"the shared branch {m['shared']} does not exist"
    if head == tip:
        return True, f"up to date with {_short(m['shared'])} at {tip[:9]}"
    if _rev(root, f"{head}^{{commit}}") and _git(root, "merge-base", "--is-ancestor", head, tip).returncode != 0:
        ahead = _git(root, "rev-list", "--count", f"{tip}..{head}").stdout.strip()
        return False, (f"HEAD has {ahead} commit(s) that are not on {_short(m['shared'])}: land work with "
                       "./dk save / accept / commit-tools, not raw commits")
    paths = [p for p in _git(root, "diff", "--name-only", "--no-renames", "-z", head, tip).stdout.split("\0") if p]
    old, new = entries(root, head, paths), entries(root, tip, paths)
    blobs = contents(root, {e[1] for e in (*old.values(), *new.values())})
    plan, conflicts, merged, kept = [], [], 0, 0
    for rel in paths:
        mine = read_work(root / rel)
        base = blobs[old[rel][1]] if rel in old else None
        theirs = blobs[new[rel][1]] if rel in new else None
        how, data = _resolve(rel, mine, base, theirs)
        if how in ("merged", "conflict") and rel in keep_mine:
            kept += 1
            continue
        if how == "conflict":
            conflicts.append(rel)
        elif how in ("theirs", "merged"):
            plan.append((rel, new.get(rel), data, mine))
            merged += how == "merged"
    if conflicts:
        return False, (f"sync refused, nothing changed: {len(conflicts)} path(s) changed both here and on "
                       f"{_short(m['shared'])} in ways that do not merge: {', '.join(conflicts)} "
                       "(fix them by hand, or --keep-mine PATH to keep yours)")
    if _git(root, "update-ref", "HEAD", tip, head).returncode != 0:
        return False, "the worktree's branch moved during the sync; retry"
    for rel, entry, data, _ in plan:
        write_work(root / rel, entry, data)
    if paths:       # (an empty pathspec would reset the whole index)
        _git(root, "reset", "-q", "--pathspec-from-file=-", "--pathspec-file-nul", data="\0".join(paths).encode())
    return True, (f"synced to {_short(m['shared'])} at {tip[:9]}: {len(paths)} path(s) changed there, "
                  f"{merged} merged with yours, {kept} kept as yours")


def _short(ref: str) -> str:
    return ref.removeprefix("refs/heads/")


# --- copying a landed commit into other checkouts of the branch ------------------------------

def plan_copy(root: Path, ref: str, old: str, new: str, paths: list[str]) -> tuple[list[tuple], list[str]]:
    """What landing `new` (on top of `old`) on `ref` does to every other checkout of `ref`: a
    checkout whose file still equals the old version gets the new one; one whose own edits merge
    cleanly gets the merge. Returns (actions, refusals): a refusal is a path that checkout has
    staged differently from `old`, or edits there that do not merge."""
    actions, refusals = [], []
    me = Path(root).resolve()
    trees = [t for t in checkouts_of(root, ref) if t.resolve() != me]
    if not trees:
        return [], []
    before, after = entries(root, old, paths), entries(root, new, paths)
    changed = [p for p in paths if before.get(p) != after.get(p)]
    if not changed:
        return [], []
    blobs = contents(root, {e[1] for e in (*before.values(), *after.values())})
    for tree in trees:
        index = staged(tree, changed)
        for rel in changed:
            if index.get(rel, (None, None))[1] != before.get(rel, (None, None))[1]:
                refusals.append(f"{tree}: {rel} is staged there differently from {old[:9]}")
                continue
            mine = read_work(tree / rel)
            base = blobs[before[rel][1]] if rel in before else None
            theirs = blobs[after[rel][1]] if rel in after else None
            how, data = _resolve(rel, mine, base, theirs)
            if how == "conflict":
                refusals.append(f"{tree}: {rel} has uncommitted edits there that do not merge with this change")
            else:
                actions.append((tree, rel, after.get(rel), data, mine, how))
    return actions, refusals


def apply_copy(actions: list[tuple]) -> list[str]:
    """Carry out plan_copy's actions (after the ref moved): write the files, then point each
    checkout's index at the new blobs. Returns notes (a file edited since the plan is left alone)."""
    notes, index_info = [], {}
    for tree, rel, entry, data, mine, how in actions:
        if how != "same":
            if read_work(tree / rel) != mine:
                notes.append(f"{tree}: {rel} changed meanwhile; left as it is")
            else:
                write_work(tree / rel, entry, data)
        line = f"{entry[0]} {entry[1]}\t{rel}" if entry else f"0 {ZERO}\t{rel}"
        index_info.setdefault(tree, []).append(line)
    for tree, lines in index_info.items():
        proc = _git(tree, "update-index", "--index-info", data=("\n".join(lines) + "\n").encode())
        if proc.returncode != 0:
            notes.append(f"{tree}: index not updated ({proc.stderr.strip()}): run `git status` there")
    return notes
