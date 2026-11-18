"""Small shared helpers."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import os
import signal
import subprocess
from pathlib import Path

DISC_SUFFIXES = (".iso", ".gcm", ".ciso", ".gcz", ".rvz", ".wia", ".wbfs", ".nfs", ".tgc")


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_disc_images(dirs: tuple[Path, ...]) -> list[Path]:
    found = []
    for d in dirs:
        if d.is_dir():
            found.extend(sorted(p for p in d.iterdir() if p.suffix.lower() in DISC_SUFFIXES))
    return found


def tail(text: str, lines: int) -> str:
    return "\n".join(text.rstrip().splitlines()[-lines:])


_lock_depth = 0
_lock_file = None


@contextlib.contextmanager
def project_lock(root: Path):
    """Serialize builds and shared-file edits across concurrent dk processes (parallel agents).

    Reentrant within one process; an flock on build/.dk.lock across processes.
    """
    global _lock_depth, _lock_file
    if _lock_depth == 0:
        (root / "build").mkdir(exist_ok=True)
        _lock_file = open(root / "build" / ".dk.lock", "w")
        fcntl.flock(_lock_file, fcntl.LOCK_EX)
    _lock_depth += 1
    try:
        yield
    finally:
        _lock_depth -= 1
        if _lock_depth == 0 and _lock_file is not None:
            fcntl.flock(_lock_file, fcntl.LOCK_UN)
            _lock_file.close()
            _lock_file = None


@contextlib.contextmanager
def released():
    """Give up this process's project lock for a while (a clean-worktree build touches nothing
    shared), then take it back. A no-op when the lock is not held."""
    held = _lock_depth > 0 and _lock_file is not None
    if held:
        fcntl.flock(_lock_file, fcntl.LOCK_UN)
    try:
        yield
    finally:
        if held:
            fcntl.flock(_lock_file, fcntl.LOCK_EX)


NINJA_TIMEOUT = 180  # seconds; a full re-split + link takes ~2 s, so this is a hang


def run_ninja(root: Path, targets: list[str] | None = None, timeout: int = NINJA_TIMEOUT) -> subprocess.CompletedProcess:
    """Run ninja with output captured. On timeout, kill the whole process group (compilers and
    linkers run under wibo and would otherwise keep spinning) and return code 124."""
    cmd = ["ninja", *(targets or [])]
    with project_lock(root):
        proc = subprocess.Popen(cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                start_new_session=True)
        try:
            out, err = proc.communicate(timeout=timeout)
            return subprocess.CompletedProcess(cmd, proc.returncode, out, err)
        except subprocess.TimeoutExpired:
            kill_session(proc.pid)
            out, err = proc.communicate()
            return subprocess.CompletedProcess(cmd, 124, out, (err or "") + f"\nTIMEOUT: ninja killed after {timeout}s\n")


def kill_session(sid: int) -> None:
    """SIGKILL every process in session `sid`. wibo moves itself into its own process group,
    so killing ninja's group alone leaves hung compilers/linkers spinning; the session holds all."""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
        except OSError:
            continue
        if int(fields[3]) == sid:       # stat field 6 (session) = index 3 after "pid (comm)"
            try:
                os.kill(int(entry.name), signal.SIGKILL)
            except ProcessLookupError:
                pass
