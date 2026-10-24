"""Read decomp-toolkit symbols.txt / splits.txt files."""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from pathlib import Path

# name = .section:0xADDRESS; // key:value key:value ...
_SYMBOL = re.compile(r"^(?P<name>\S+)\s*=\s*(?P<section>[^:\s]+):0x(?P<addr>[0-9A-Fa-f]+);\s*(?://\s*(?P<attrs>.*))?$")
# Unit header in splits.txt: a non-indented "path:" line, optionally followed by attributes.
_UNIT = re.compile(r"^(?P<name>\S.*?):(?:\s+(?P<attrs>\S.*?))?\s*$")
# Indented split line: ".section start:0x... end:0x... [more]"
_SPLIT = re.compile(r"^\s+(?P<section>\S+)\s+start:0x(?P<start>[0-9A-Fa-f]+)\s+end:0x(?P<end>[0-9A-Fa-f]+)")


@dataclass(frozen=True)
class Symbol:
    name: str
    section: str
    address: int
    attrs: dict[str, str]

    @property
    def size(self) -> int:
        return int(self.attrs.get("size", "0"), 16)

    @property
    def type(self) -> str:
        return self.attrs.get("type", "")


def _parse_attrs(text: str | None) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for token in (text or "").split():
        key, sep, value = token.partition(":")
        attrs[key] = value if sep else ""
    return attrs


def read_symbols(path: Path) -> list[Symbol]:
    symbols = []
    for line in path.read_text().splitlines():
        m = _SYMBOL.match(line.strip())
        if m:
            symbols.append(Symbol(m["name"], m["section"], int(m["addr"], 16), _parse_attrs(m["attrs"])))
    symbols.sort(key=lambda s: (s.address, s.name))
    return symbols


class SymbolIndex:
    """Nearest-symbol lookup by address."""

    def __init__(self, symbols: list[Symbol]):
        self._symbols = sorted(symbols, key=lambda s: s.address)
        self._addresses = [s.address for s in self._symbols]

    def at_or_before(self, address: int) -> Symbol | None:
        i = bisect.bisect_right(self._addresses, address) - 1
        return self._symbols[i] if i >= 0 else None

    def first_at_or_after(self, address: int, section: str) -> int | None:
        i = bisect.bisect_left(self._addresses, address)
        while i < len(self._symbols):
            if self._symbols[i].section == section:
                return self._symbols[i].address
            i += 1
        return None

    def describe(self, address: int) -> str:
        sym = self.at_or_before(address)
        if sym is None:
            return f"0x{address:08X}"
        return f"{sym.section} {sym.name}+0x{address - sym.address:X}"


AUTO_PREFIXES = ("fn_", "lbl_", "jumptable_", "gap_")


def is_auto_name(name: str) -> bool:
    return name.startswith(AUTO_PREFIXES)


def rename_symbols(path: Path, names: dict[int, str]) -> tuple[str, int]:
    """Rename auto-named symbols at the given addresses. Returns (new text, renames applied).

    Only names dtk invented (fn_/lbl_/...) are replaced, a new name already used elsewhere is
    skipped, and every other byte of the file is preserved.
    """
    lines = path.read_text().splitlines(keepends=True)
    taken = {m["name"] for line in lines if (m := _SYMBOL.match(line.strip()))}
    applied = 0
    for i, line in enumerate(lines):
        m = _SYMBOL.match(line.strip())
        if not m:
            continue
        new = names.get(int(m["addr"], 16))
        if new is None or new == m["name"] or not is_auto_name(m["name"]) or new in taken:
            continue
        lines[i] = line.replace(m["name"], new, 1)
        taken.add(new)
        applied += 1
    return "".join(lines), applied


@dataclass
class Unit:
    name: str
    splits: list[tuple[str, int, int]] = field(default_factory=list)  # (section, start, end)


def read_units(path: Path) -> list[Unit]:
    """Translation units declared in splits.txt (the Sections: block is skipped)."""
    units: list[Unit] = []
    current: Unit | None = None
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        m = _UNIT.match(line)
        if m:
            current = None if m["name"] == "Sections" else Unit(m["name"])
            if current is not None:
                units.append(current)
            continue
        s = _SPLIT.match(line)
        if s and current is not None:
            current.splits.append((s["section"], int(s["start"], 16), int(s["end"], 16)))
    return units


_OVERLAP = re.compile(r"Symbol (?P<a>\S+) \((?P<a_start>0x[0-9A-Fa-f]+)\.\.[^)]*\) overlaps with symbol "
                      r"(?P<b>\S+) \((?P<b_start>0x[0-9A-Fa-f]+)\.\.")


def overlap_victim(message: str) -> tuple[str, int] | None:
    """From a dtk 'Symbol A overlaps with symbol B' error, the auto-named one to drop."""
    m = _OVERLAP.search(message)
    if not m:
        return None
    for name, start in ((m["b"], m["b_start"]), (m["a"], m["a_start"])):
        if is_auto_name(name):
            return name, int(start, 16)
    return None


def remove_symbol(path: Path, name: str, address: int) -> bool:
    lines = path.read_text().splitlines(keepends=True)
    keep = [line for line in lines
            if not ((m := _SYMBOL.match(line.strip())) and m["name"] == name and int(m["addr"], 16) == address)]
    if len(keep) == len(lines):
        return False
    path.write_text("".join(keep))
    return True


_WITHIN = re.compile(r"Split \S+ \S+ \((?P<a>0x[0-9A-Fa-f]+)\.\.(?P<b>0x[0-9A-Fa-f]+)\) (?P<kind>starts|ends) within symbol "
                     r"'(?P<name>[^']+)' \((?P<s>0x[0-9A-Fa-f]+)\.\.(?P<e>0x[0-9A-Fa-f]+)\)")


def boundary_fix(message: str) -> tuple[str, int, int] | None:
    """From dtk 'Split ... starts/ends within symbol X', the (auto) symbol and the size that ends it
    at the split boundary. dtk's sizes for auto labels are guesses, so trimming them is safe."""
    m = _WITHIN.search(message)
    # __vt__/__RTTI__ names were assigned by `dk analyze rtti`, but their sizes are still dtk's guesses.
    if not m or not (is_auto_name(m["name"]) or m["name"].startswith(("__vt__", "__RTTI__"))):
        return None
    start = int(m["s"], 16)
    boundary = int(m["b"] if m["kind"] == "ends" else m["a"], 16)
    return (m["name"], start, boundary - start) if boundary > start else None


def set_symbol_size(text: str, name: str, address: int, size: int) -> str | None:
    out = []
    changed = False
    for line in text.splitlines(keepends=True):
        m = _SYMBOL.match(line.strip())
        if m and m["name"] == name and int(m["addr"], 16) == address:
            new = re.sub(r"size:0x[0-9A-Fa-f]+", f"size:0x{size:X}", line, count=1)
            if new == line:
                new = line.rstrip("\n") + f" size:0x{size:X}\n"
            changed = new != line
            line = new
        out.append(line)
    return "".join(out) if changed else None


def caused_by(message: str) -> str:
    """The informative line of a dtk failure, for one-line reports."""
    lines = [l.strip() for l in message.splitlines()]
    for i, line in enumerate(lines):
        if line == "Caused by:" and i + 1 < len(lines):
            return lines[i + 1]
    # objdiff and the compiler/linker say what went wrong; ninja's FAILED line only names the edge
    specific = [l for l in lines if l.startswith(("Failed:", "error", "Error", "ERROR")) or "undefined" in l]
    if specific:
        return specific[-1]
    failed = [l for l in lines if l.startswith(("FAILED", "TIMEOUT"))]
    return failed[-1] if failed else (lines[-1] if lines else "")


_NOISE = ("FORCEACTIVE", "Usage Warning", "Specified directory")


def build_errors(output: str, lines: int = 25) -> str:
    """The last lines of a failed build without linker/compiler noise, plus a hint for the
    known split mistakes."""
    kept = [l for l in output.rstrip().splitlines() if not any(n in l for n in _NOISE)]
    text = "\n".join(kept[-lines:])
    if "gap_" in output and "out of section bounds" in output:
        text += ("\nhint: a few bytes of padding were left between two units as a tiny gap split; "
                 "extend the previous unit's split over them (./dk unit set)")
    return text
