"""Synthetic fixtures. Nothing here comes from a game, so it is safe to commit."""

from __future__ import annotations

import struct
import textwrap
from pathlib import Path

from tools.dk.core.config import Config, load


def make_dol(sections: list[tuple[str, int, bytes]], bss: tuple[int, int] = (0, 0), entry: int = 0) -> bytes:
    """sections: (slot like "T0"/"D3", load address, contents)."""
    offsets = [0] * 18
    addresses = [0] * 18
    sizes = [0] * 18
    body = bytearray()
    cursor = 0x100
    for slot, address, contents in sections:
        i = int(slot[1:]) + (0 if slot[0] == "T" else 7)
        offsets[i], addresses[i], sizes[i] = cursor, address, len(contents)
        body += contents
        cursor += len(contents)
    header = struct.pack(">18I18I18I3I", *offsets, *addresses, *sizes, bss[0], bss[1], entry)
    return header.ljust(0x100, b"\0") + bytes(body)


DK_TOML = """\
[project]
name = "demo"
game_id = "DEMO01"
platform = "gc_mwcc"

[target]
original = "orig/main.dol"
built = "build/main.dol"
sha1 = "{sha1}"

[build]
report = "build/report.json"

[platform]
symbols = "config/symbols.txt"
splits = "config/splits.txt"

[docs]
playbook = "playbook"
project = "docs/demo"
generated = "docs/generated"
exclude = ["docs/upstream"]
banned_in_playbook = ["DemoGame", "DEMO01"]

[docs.budget]
playbook_file_lines = 10
playbook_total_lines = 15
agents_md_lines = 5
"""


def make_project(root: Path, sha1: str = "0" * 40, files: dict[str, str] | None = None) -> Config:
    (root / "dk.toml").write_text(DK_TOML.format(sha1=sha1))
    for rel, text in (files or {}).items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text))
    return load(root)


# A throwaway git project whose "build" is a script: it fails when a tracked .txt says BROKEN.
FAKE_CONFIGURE = """\
open("build.ninja", "w").write("rule check\\n  command = python3 check.py && mkdir -p build && touch $out\\n"
                               "build build/ok: check\\ndefault build/ok\\n")
"""
FAKE_CHECK = """\
import os, pathlib, sys, time
barrier = os.environ.get("DK_TEST_BARRIER")     # wait until two builds have started (forces a race)
if barrier:
    pathlib.Path(barrier, str(os.getpid())).touch()
    end = time.time() + 20
    while len(os.listdir(barrier)) < 2 and time.time() < end:
        time.sleep(0.05)
sys.exit(1 if any("BROKEN" in p.read_text() for p in pathlib.Path("src").rglob("*.txt")) else 0)
"""


def git(root: Path, *args: str) -> str:
    import subprocess
    proc = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {proc.stderr.strip()}")
    return proc.stdout.strip()


def make_git_project(root: Path, files: dict[str, str] | None = None, branch: str = "shared") -> Config:
    """A git repo (one commit on `branch`) holding a dk project that verify_overlay can build."""
    root.mkdir(parents=True, exist_ok=True)
    git(root, "init", "-q", "-b", branch)
    git(root, "config", "user.name", "dk test")
    git(root, "config", "user.email", "dk@test.invalid")
    base = {"configure.py": FAKE_CONFIGURE, "check.py": FAKE_CHECK,
            ".gitignore": "build/\nbuild.ninja\n.ninja_*\norig/\n/worktrees/\n", "src/a.txt": "a\n", **(files or {})}
    cfg = make_project(root, files=base)
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "start")
    return cfg
