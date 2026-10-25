"""Deterministic edits of units.toml (read by configure.py) and dtk's splits.txt."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .dtk_symbols import read_units

STATUSES = ("Matching", "NonMatching", "Equivalent")
LIB_KEYS = ("name", "mw_version", "cflags", "extra_cflags", "progress_category")


@dataclass
class Lib:
    name: str
    settings: dict[str, Any] = field(default_factory=dict)     # mw_version, cflags, ...
    objects: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class UnitsFile:
    header: list[str]
    libs: list[Lib]

    def find(self, path: str) -> tuple[Lib, dict[str, Any]] | None:
        for lib in self.libs:
            for obj in lib.objects:
                if obj["path"] == path:
                    return lib, obj
        return None

    def lib(self, name: str) -> Lib | None:
        return next((lib for lib in self.libs if lib.name == name), None)


def read_units_file(path: Path) -> UnitsFile:
    text = path.read_text() if path.is_file() else ""
    header = []
    for line in text.splitlines():
        if not line.startswith("#"):
            break
        header.append(line)
    data = tomllib.loads(text)
    libs = []
    for raw in data.get("lib", []):
        settings = {k: raw[k] for k in LIB_KEYS[1:] if k in raw}
        libs.append(Lib(raw["name"], settings, [dict(o) for o in raw.get("objects", [])]))
    return UnitsFile(header, libs)


def _value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, list):
        return "[" + ", ".join(_value(x) for x in v) + "]"
    raise TypeError(f"unsupported value {v!r}")


def _inline(obj: dict[str, Any]) -> str:
    keys = ["path", "status"] + sorted(k for k in obj if k not in ("path", "status"))
    return "{ " + ", ".join(f"{k} = {_value(obj[k])}" for k in keys if k in obj) + " }"


def render_units_file(units: UnitsFile) -> str:
    out = list(units.header)
    for lib in units.libs:
        out += ["", "[[lib]]", f"name = {_value(lib.name)}"]
        out += [f"{k} = {_value(lib.settings[k])}" for k in LIB_KEYS[1:] if k in lib.settings]
        out.append("objects = [")
        out += [f"    {_inline(o)}," for o in lib.objects]
        out.append("]")
    return "\n".join(out) + "\n"


def add_object(units: UnitsFile, lib_name: str, path: str, status: str, lib_settings: dict[str, Any],
               options: dict[str, Any] | None = None) -> bool:
    """Returns False if the object already exists (no change). options = per-object overrides."""
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    if units.find(path):
        return False
    lib = units.lib(lib_name)
    if lib is None:
        lib = Lib(lib_name, {k: v for k, v in lib_settings.items() if v is not None})
        units.libs.append(lib)
    lib.objects.append({"path": path, "status": status, **{k: v for k, v in (options or {}).items() if v}})
    return True


def set_status(units: UnitsFile, path: str, status: str) -> bool:
    found = units.find(path)
    if found is None:
        raise KeyError(path)
    _, obj = found
    changed = obj["status"] != status
    obj["status"] = status
    return changed


def set_flags(units: UnitsFile, path: str, mw_version: str | None, cflags: str | None,
              extra: list[str] | None) -> list[str]:
    """Set an object's compiler, flag set and extra flags (extra=[] clears them). Returns
    what changed, as 'key: old -> new' lines."""
    found = units.find(path)
    if found is None:
        raise KeyError(path)
    _, obj = found
    changes = []
    for key, value in (("mw_version", mw_version), ("cflags", cflags), ("extra_cflags", extra)):
        if value is None:
            continue
        old = obj.get(key)
        if key == "extra_cflags" and value == []:
            if "extra_cflags" in obj:
                del obj["extra_cflags"]
                changes.append(f"extra_cflags: {old} -> (none)")
            continue
        if old != value:
            obj[key] = value
            changes.append(f"{key}: {old} -> {value}")
    return changes


# --- splits.txt -----------------------------------------------------------------------------

def _split_line(section: str, start: int, end: int) -> str:
    return f"\t{section:<11} start:0x{start:08X} end:0x{end:08X}"


def add_split_unit(path: Path, name: str, splits: list[tuple[str, int, int]]) -> str:
    """Return splits.txt text with a new unit block inserted by address.

    Existing lines are kept verbatim. Raises on a duplicate name or an overlapping range.
    """
    existing = read_units(path)
    if any(u.name == name for u in existing):
        raise ValueError(f"unit {name} already in {path.name}")
    for u in existing:
        for sec, start, end in u.splits:
            for nsec, nstart, nend in splits:
                if sec == nsec and nstart < end and start < nend:
                    raise ValueError(f"{nsec} 0x{nstart:08X}-0x{nend:08X} overlaps {u.name}")

    lines = path.read_text().splitlines()
    # Indices of unit header lines (non-indented, not "Sections:").
    headers = [i for i, line in enumerate(lines)
               if line and not line[0].isspace() and not line.startswith(("Sections:", "//"))]
    first_addr = {u.name: min((s[1] for s in u.splits), default=0) for u in existing}
    new_addr = min(s[1] for s in splits)
    insert_at = len(lines)
    for i in headers:
        unit_name = lines[i].split(":", 1)[0]
        if first_addr.get(unit_name, 0) > new_addr:
            insert_at = i
            break
    block = [f"{name}:"] + [_split_line(*s) for s in splits] + [""]
    if insert_at == len(lines) and lines and lines[-1].strip():
        block = [""] + block
    return "\n".join(lines[:insert_at] + block + lines[insert_at:]).rstrip("\n") + "\n"


def _unit_block(lines: list[str], name: str) -> tuple[int, int] | None:
    """[start, end) line indices of a unit's block (header + its split lines)."""
    for i, line in enumerate(lines):
        if line and not line[0].isspace() and line.split(":", 1)[0] == name and not line.startswith("Sections:"):
            j = i + 1
            while j < len(lines) and lines[j].startswith(("\t", " ")) and lines[j].strip():
                j += 1
            return i, j
    return None


def replace_split_unit(path: Path, name: str, splits: list[tuple[str, int, int]], drop: tuple[str, ...] = ()) -> str:
    """New splits.txt text with `name`'s split lines replaced (header line kept verbatim).
    A section the unit had must be listed again or named in `drop`: leaving one out by
    accident would silently hand its data back to the auto split."""
    for u in read_units(path):
        if u.name == name:
            missing = sorted({sec for sec, _, _ in u.splits} - {sec for sec, _, _ in splits} - set(drop))
            if missing:
                raise ValueError(f"{name} has {', '.join(missing)}, missing from the new list "
                                 f"(repeat them, or --drop {' --drop '.join(missing)} to remove)")
            continue
        for sec, start, end in u.splits:
            for nsec, nstart, nend in splits:
                if sec == nsec and nstart < end and start < nend:
                    raise ValueError(f"{nsec} 0x{nstart:08X}-0x{nend:08X} overlaps {u.name}")
    lines = path.read_text().splitlines()
    block = _unit_block(lines, name)
    if block is None:
        raise ValueError(f"unit {name} not in {path.name}")
    i, j = block
    lines[i + 1:j] = [_split_line(*sp) for sp in splits]
    return "\n".join(lines).rstrip("\n") + "\n"


def remove_split_unit(path: Path, name: str) -> str:
    lines = path.read_text().splitlines()
    block = _unit_block(lines, name)
    if block is None:
        raise ValueError(f"unit {name} not in {path.name}")
    i, j = block
    if j < len(lines) and not lines[j].strip():
        j += 1
    del lines[i:j]
    return "\n".join(lines).rstrip("\n") + "\n"


def remove_object(units: UnitsFile, path: str) -> bool:
    for lib in units.libs:
        for obj in lib.objects:
            if obj["path"] == path:
                lib.objects.remove(obj)
                if not lib.objects:
                    units.libs.remove(lib)
                return True
    return False
