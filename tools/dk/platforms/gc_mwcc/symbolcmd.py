"""dk symbol split|size|add: correct dtk's guessed data labels, then prove the build still matches.

dtk folds several literals that one function addresses as base+offset into a single auto
label; splitting it lets each source-level object pair with its own target symbol. The
opposite also happens (a string or u64 cut into several auto labels), and objects reached
only through a neighbour's base+offset get no label at all.
"""

from __future__ import annotations

import argparse
import re

from ...core.command import MISMATCH, OK, USAGE, Command, hint
from ...core.config import Config
from .dtk_symbols import is_auto_name
from .unitcmd import Transaction, _finish, rebuild

_LINE = re.compile(r"^(?P<name>\S+) = (?P<section>[^:\s]+):0x(?P<addr>[0-9A-Fa-f]+);(?P<rest>.*)$")


def split_text(text: str, address: int, new_name: str | None) -> tuple[str, str]:
    """Split the symbol containing `address` at it. Returns (new text, description)."""
    lines = text.splitlines(keepends=True)
    syms = read_symbols_from_text(lines)
    owner = next((s for s in syms if s["start"] < address < s["start"] + s["size"]), None)
    if owner is None:
        raise ValueError(f"no sized symbol contains 0x{address:08X} (strictly inside)")
    if not is_auto_name(owner["name"]) and not owner["name"].startswith("@"):
        raise ValueError(f"{owner['name']} is a named symbol; only dtk's auto labels are split")
    name = new_name or f"lbl_{address:08X}"
    if any(s["name"] == name for s in syms):
        raise ValueError(f"{name} already exists")
    head_size = address - owner["start"]
    tail_size = owner["start"] + owner["size"] - address
    line = lines[owner["index"]]
    lines[owner["index"]] = re.sub(r"size:0x[0-9A-Fa-f]+", f"size:0x{head_size:X}", line, count=1)
    new_line = f"{name} = {owner['section']}:0x{address:08X}; // type:object size:0x{tail_size:X}\n"
    lines.insert(owner["index"] + 1, new_line)
    return "".join(lines), f"{owner['name']} -> 0x{head_size:X} bytes, new {name} (0x{tail_size:X} bytes)"


def read_symbols_from_text(lines: list[str]) -> list[dict]:
    out = []
    for i, line in enumerate(lines):
        m = _LINE.match(line.strip())
        if m:
            size = re.search(r"size:0x([0-9A-Fa-f]+)", m["rest"])
            out.append({"index": i, "name": m["name"], "section": m["section"], "start": int(m["addr"], 16),
                        "size": int(size[1], 16) if size else 0})
    return out


def _absorb(syms: list[dict], section: str, start: int, size: int, keep: int | None) -> list[int]:
    """Line indexes of the auto labels a new object [start, start+size) swallows whole. Raises if
    it would overlap a named symbol or only part of a label."""
    end = start + size
    out = []
    for s in syms:
        if s["index"] == keep or s["section"] != section or not s["size"]:
            continue
        if s["start"] < end and start < s["start"] + s["size"]:
            if not (is_auto_name(s["name"]) or s["name"].startswith("@")):
                raise ValueError(f"would overlap named symbol {s['name']} at 0x{s['start']:08X}")
            if s["start"] < start or s["start"] + s["size"] > end:
                raise ValueError(f"would cut {s['name']} (0x{s['start']:08X}+0x{s['size']:X}); `symbol split` it first")
            out.append(s["index"])
    return out


def size_text(text: str, name: str, size: int) -> tuple[str, list[str]]:
    """Set a symbol's size; auto labels it now covers whole are removed. Returns (text, removed)."""
    lines = text.splitlines(keepends=True)
    syms = read_symbols_from_text(lines)
    hits = [s for s in syms if s["name"] == name]
    if len(hits) != 1:
        raise ValueError(f"{name} occurs {len(hits)} times")
    i = hits[0]["index"]
    gone = _absorb(syms, hits[0]["section"], hits[0]["start"], size, keep=i)
    line = lines[i]
    new = re.sub(r"size:0x[0-9A-Fa-f]+", f"size:0x{size:X}", line, count=1)
    lines[i] = new if new != line else line.rstrip("\n") + f" size:0x{size:X}\n"
    removed = [lines[j].split(" = ", 1)[0] for j in gone]
    return "".join(line for j, line in enumerate(lines) if j not in gone), removed


def add_text(text: str, name: str, section: str, address: int, size: int | None, scope: str | None) -> tuple[str, list[str]]:
    """Add an object (with size) or a label (size None) in address order. An object swallows
    the auto labels inside its range whole; a label may sit anywhere. Returns (text, removed)."""
    lines = text.splitlines(keepends=True)
    syms = read_symbols_from_text(lines)
    if any(s["name"] == name for s in syms):
        raise ValueError(f"{name} already exists")
    gone = _absorb(syms, section, address, size, keep=None) if size else []
    if size and any(s["section"] == section and s["start"] == address and s["size"] and s["index"] not in gone
                    for s in syms):
        raise ValueError(f"a symbol already starts at 0x{address:08X}; rename or size it instead")
    kind = f"type:object size:0x{size:X}" if size else "type:label"
    new_line = f"{name} = {section}:0x{address:08X}; // {kind}" + (f" scope:{scope}" if scope else "") + "\n"
    after = [s for s in syms if s["start"] < address or (s["start"] == address and not size)]
    pos = max((s["index"] for s in after), default=-1) + 1
    removed = [lines[j].split(" = ", 1)[0] for j in gone]
    lines.insert(pos, new_line)
    gone = {j + (j >= pos) for j in gone}
    return "".join(line for j, line in enumerate(lines) if j not in gone), removed


def run(cfg: Config, args: argparse.Namespace) -> int:
    path = cfg.root / cfg.platform_options["symbols"]
    try:
        if args.action == "split":
            text, desc = split_text(path.read_text(), int(args.address, 16), args.name)
        elif args.action == "size":
            text, removed = size_text(path.read_text(), args.name, int(args.size, 16))
            desc = f"{args.name} size 0x{int(args.size, 16):X}" + (f", absorbed {', '.join(removed)}" if removed else "")
        else:
            section, _, addr = args.where.partition(":")
            if not section.startswith(".") or not addr:
                raise ValueError(f"expected SECTION:ADDR (e.g. .bss:0x80382258), got {args.where!r}")
            if (args.size is None) == (not args.label):
                raise ValueError("give a SIZE for an object, or --label for a size-less label")
            size = int(args.size, 16) if args.size else None
            text, removed = add_text(path.read_text(), args.name, section, int(addr, 16), size, args.scope)
            desc = f"added {args.name} at {section}:0x{int(addr, 16):08X}" + (f", absorbed {', '.join(removed)}" if removed else "")
    except ValueError as e:
        print(f"error: {e}")
        return USAGE
    tx = Transaction()
    tx.write(path, text)
    code = _finish(cfg, tx, rebuild(cfg, tx), f"{desc}; build still matches")
    if code == OK:
        hint("check <unit>")
    return code


def _configure(p: argparse.ArgumentParser) -> None:
    sub = p.add_subparsers(dest="action", required=True)
    sp = sub.add_parser("split", help="split the auto label containing ADDR at ADDR")
    sp.add_argument("address", help="hex address inside the label, e.g. 0x80250F70")
    sp.add_argument("--name", help="name for the new symbol (default lbl_<ADDR>)")
    sz = sub.add_parser("size", help="set a symbol's size (auto labels it then covers whole are removed)")
    sz.add_argument("name")
    sz.add_argument("size", help="hex size, e.g. 0x30")
    ad = sub.add_parser("add", help="add an object dtk never labelled, or a label (e.g. __RAS_*_begin)")
    ad.add_argument("name")
    ad.add_argument("where", help="SECTION:ADDR, e.g. .bss:0x80382258")
    ad.add_argument("size", nargs="?", help="hex size of an object (auto labels inside it are removed)")
    ad.add_argument("--label", action="store_true", help="a size-less label (may sit inside a function)")
    ad.add_argument("--scope", choices=("global", "local", "weak"), help="symbol scope (default: dtk's)")


COMMAND = Command("symbol", "analysis", "correct dtk's guessed data labels (split, size, add); proves the hash", run, _configure,
                  locked=True)
