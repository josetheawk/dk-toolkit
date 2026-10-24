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
