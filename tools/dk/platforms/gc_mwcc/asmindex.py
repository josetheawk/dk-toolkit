"""Index dtk's disassembly: functions, data objects and what each references.

dtk never deletes asm for splits that no longer exist, so only files belonging to units in
the current build/<id>/config.json are read.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ...core.config import Config

_START = re.compile(r'^\.(?P<kind>fn|obj) "?(?P<name>[^,\s"]+)"?')     # dtk quotes names with @ etc.
_END = re.compile(r'^\.end(?:fn|obj) "?(?P<name>[^\s"]+)"?')
_CODE_REF = re.compile(r"\b(?P<sym>[A-Za-z_.$@?][\w.$@?<>,:*&-]*)@(?:ha|h|l|sda21|sda2)\b")
_CALL = re.compile(r"\bbl?\s+(?P<sym>[A-Za-z_$@?][\w.$@?<>,:*&-]*)")   # bl calls and b tail calls (not .L labels)
_DATA_WORD = re.compile(r"^\s*\.4byte (?P<sym>[A-Za-z_.$@?][\w.$@?<>,:*&-]*)")


@dataclass
class AsmEntry:
    name: str
    kind: str                      # "fn" or "obj"
    file: Path
    text: str
    refs: list[str] = field(default_factory=list)    # data symbols referenced (code: @ha/@l/@sda21, data: .4byte)
    calls: list[str] = field(default_factory=list)


def current_asm_files(cfg: Config) -> list[Path]:
    build = cfg.root / "build" / cfg.game_id
    config_json = build / "config.json"
    asm_dir = cfg.root / cfg.platform_options["asm"]
    if not config_json.is_file():
        return sorted(asm_dir.rglob("*.s"))
    units = json.loads(config_json.read_text()).get("units", [])
    files = []
    for u in units:
        rel = (cfg.root / u["object"]).relative_to(build / "obj").with_suffix(".s")
        if (asm_dir / rel).is_file():
            files.append(asm_dir / rel)
    return sorted(files)


def _unique(items) -> list[str]:
    return list(dict.fromkeys(items))


def index(files: list[Path]) -> dict[str, AsmEntry]:
    entries: dict[str, AsmEntry] = {}
    for path in files:
        current: tuple[str, str] | None = None
        lines: list[str] = []
        for line in path.read_text().splitlines():
            m = _START.match(line)
            if m:
                current, lines = (m["kind"], m["name"]), [line]
                continue
            if current is None:
                continue
            lines.append(line)
            if _END.match(line):
                kind, name = current
                body = "\n".join(lines)
                if kind == "fn":
                    refs = [x["sym"] for x in _CODE_REF.finditer(body)]
                    calls = [x["sym"] for x in _CALL.finditer(body)]
                else:
                    refs = [x["sym"] for l in lines if (x := _DATA_WORD.match(l))]
                    calls = []
                entries[name] = AsmEntry(name, kind, path, body, _unique(refs), _unique(calls))
                current = None
    return entries


def referrers(entries: dict[str, AsmEntry]) -> dict[str, set[str]]:
    """symbol -> names of functions/objects that reference it (data refs and calls)."""
    out: dict[str, set[str]] = defaultdict(set)
    for e in entries.values():
        for r in e.refs + e.calls:
            out[r].add(e.name)
    return out
