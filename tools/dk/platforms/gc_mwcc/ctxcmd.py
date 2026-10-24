"""dk ctx <function>: everything an agent (or person) needs to start matching one function."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ...core.command import OK, USAGE, Command, hint
from ...core.config import Config
from . import asmindex
from .dtk_symbols import read_symbols, read_units

_FN_START = re.compile(r"^\.fn (?P<name>[^,\s]+)")
_FN_END = re.compile(r"^\.endfn (?P<name>\S+)")
_CALL = re.compile(r"\bbl\s+(?P<target>[A-Za-z_.$@][\w.$@]*)")
_DATA_REF = re.compile(r"\b(?P<sym>[A-Za-z_.$@][\w.$@]*)@(?:ha|h|l|sda21|sda2)\b")


@dataclass
class AsmFunction:
    name: str
    file: Path
    text: str


def index_asm(asm_dir: Path) -> dict[str, AsmFunction]:
    functions: dict[str, AsmFunction] = {}
    for path in sorted(asm_dir.rglob("*.s")):
        current: str | None = None
        lines: list[str] = []
        for line in path.read_text().splitlines():
            m = _FN_START.match(line)
            if m:
                current, lines = m["name"], [line]
                continue
            if current is not None:
                lines.append(line)
                if _FN_END.match(line):
                    functions[current] = AsmFunction(current, path, "\n".join(lines))
                    current = None
    return functions


def callers_of(name: str, functions: dict[str, AsmFunction]) -> list[str]:
    pattern = re.compile(rf"\bbl\s+{re.escape(name)}\b")
    return sorted(f.name for f in functions.values() if f.name != name and pattern.search(f.text))


def _ordered_unique(items) -> list[str]:
    seen: dict[str, None] = {}
    for item in items:
        seen.setdefault(item, None)
    return list(seen)


def class_slots(cfg: Config, address: int) -> tuple[list[str], str | None]:
    """Vtable slots holding this function, plus the likely defining class.

    The owner is the class that every other listing class derives from (the most-base
    class whose vtable carries the function). None when that is ambiguous.
    """
    path = cfg.root / "build" / "dk" / "rtti.json"
    if not path.is_file():
        return [], None
    classes = json.loads(path.read_text())
    name_of = {c["rtti"]: c["name"] for c in classes}
    bases: dict[str, set[str]] = {}
    for c in classes:
        bases.setdefault(c["name"], set()).update(name_of[b] for b, _ in c["bases"])
    rows, listing = [], set()
    for cls in classes:
        for vt in cls["vtables"]:
            for i, slot in enumerate(vt["slots"]):
                if slot == address:
                    kind = "primary" if vt["offset"] == 0 else f"secondary {vt['offset']:#x}"
                    rows.append(f"{cls['name']} slot {i} ({kind} vtable 0x{vt['address']:08X})")
                    listing.add(cls["name"])
    owners = [c for c in sorted(listing) if all(c == o or c in bases.get(o, set()) for o in listing)]
    return sorted(set(rows)), owners[0] if len(owners) == 1 else None


def run(cfg: Config, args: argparse.Namespace) -> int:
    opts = cfg.platform_options
    symbols = {s.name: s for s in read_symbols(cfg.root / opts["symbols"])}
    sym = symbols.get(args.function)
    if sym is None:
        print(f"error: no symbol {args.function!r} in {opts['symbols']}")
        return USAGE
    asm_dir = cfg.root / opts["asm"]
    if not asm_dir.is_dir():
        print(f"error: {opts['asm']} missing")
        hint("build")
        return USAGE
    functions = {n: AsmFunction(e.name, e.file, e.text)
                 for n, e in asmindex.index(asmindex.current_asm_files(cfg)).items() if e.kind == "fn"}
    fn = functions.get(sym.name)
    if fn is None:
        print(f"error: {sym.name} not found in {opts['asm']} (is it a function?)")
        return USAGE

    unit = next((u.name for u in read_units(cfg.root / opts["splits"])
                 if any(sec == sym.section and a <= sym.address < b for sec, a, b in u.splits)), None)
    print(f"# {sym.name}  {sym.section}:0x{sym.address:08X}  size 0x{sym.size:X}")
    print(f"unit: {unit or 'not declared yet (auto unit ' + fn.file.stem + ')'}")
    if not (cfg.root / "build" / "dk" / "rtti.json").is_file():
        print("class: unknown (run ./dk analyze rtti for vtable/owner info)")
    rows, owner = class_slots(cfg, sym.address)
    if owner:
        print(f"likely owner: {owner} (most-base class whose vtable has it)")
    for row in rows:
        print(f"vtable: {row}")
    calls = _ordered_unique(m["target"] for m in _CALL.finditer(fn.text))
    refs = _ordered_unique(m["sym"] for m in _DATA_REF.finditer(fn.text))
    print(f"calls: {', '.join(calls) or '-'}")
    print(f"data: {', '.join(refs) or '-'}")
    callers = callers_of(sym.name, functions)
    print(f"callers ({len(callers)}): {', '.join(callers[:20])}{' …' if len(callers) > 20 else ''}")
    print("\n## asm\n" + fn.text)

    if not args.no_m2c:
        m2c = cfg.root / opts.get("m2c", ".venv/bin/m2c")
        if not m2c.is_file():
            print("\n## m2c\n(m2c not installed: run ./dk setup)")
        else:
            lang = "c" if (unit or "").endswith(".c") else "c++"
            cmd = [str(m2c), "--target", f"ppc-mwcc-{lang}", "-f", sym.name, str(fn.file)]
            proc = subprocess.run(cmd, cwd=cfg.root, capture_output=True, text=True, timeout=120)
            print("\n## m2c draft (machine output: rewrite, don't paste)\n" + (proc.stdout or proc.stderr).rstrip())
    if unit:
        hint(f"check {unit} --function {sym.name}")
    else:
        hint("unit add <src path> --lib <lib> SECTION:0xSTART-0xEND")
    return OK


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("function", help="function symbol name (from symbols.txt)")
    p.add_argument("--no-m2c", action="store_true", help="skip the m2c draft")


COMMAND = Command("ctx", "matching", "context pack for one function: unit, class slot, calls, callers, asm, m2c", run, _configure)
