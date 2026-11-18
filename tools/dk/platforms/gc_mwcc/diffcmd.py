"""dk diff <unit> [--function F]: side-by-side target vs compiled instructions, mismatches only.

Uses objdiff's one-shot JSON with the same strict relocation scoring as `dk check`, so what
it shows is exactly what keeps a function below 100%. Data sections below 100% get a chunk list.
"""

from __future__ import annotations

import argparse
import re
import base64

from ...core.command import BUILD_ERROR, MISMATCH, OK, USAGE, Command, hint
from ...core.config import Config
from . import freshness as fresh
from .matching import MatchError, resolve_unit

MARK = {"DIFF_OP_MISMATCH": "op", "DIFF_ARG_MISMATCH": "arg", "DIFF_REPLACE": "rep",
        "DIFF_DELETE": "-", "DIFF_INSERT": "+"}


def _text(entry: dict) -> str:
    return entry.get("instruction", {}).get("formatted", "")


def function_rows(left: dict, right: dict | None, context: int) -> list[str]:
    li = left.get("instructions", [])
    ri = right.get("instructions", []) if right else []
    n = max(len(li), len(ri))
    diff = [bool((li[i].get("diff_kind") if i < len(li) else None) or (ri[i].get("diff_kind") if i < len(ri) else None))
            for i in range(n)]
    show = set()
    for i, d in enumerate(diff):
        if d:
            show.update(range(max(0, i - context), min(n, i + context + 1)))
    rows, last = [], -2
    for i in sorted(show):
        if i != last + 1:
            rows.append("    ...")
        l = li[i] if i < len(li) else {}
        r = ri[i] if i < len(ri) else {}
        kind = l.get("diff_kind") or r.get("diff_kind")
        addr = (l.get("instruction") or r.get("instruction") or {}).get("address", "0")   # objdiff leaves out a 0
        rows.append(f"  {int(addr):>5X}  {_text(l):<42} {MARK.get(kind, '  '):>3}  {_text(r)}")
        last = i
    return rows


_STACK_SLOT = re.compile(r"(-?0x[0-9a-fA-F]+|-?\d+)\(r1\)")
_STACK_ADDR = re.compile(r"^addi r\d+, r1, (-?0x[0-9a-fA-F]+|-?\d+)$")


def _r1_offset(text: str) -> int | None:
    m = _STACK_SLOT.search(text) or _STACK_ADDR.match(text)
    return int(m.group(1), 0) if m else None


def stack_rows(left: dict, right: dict | None) -> list[str]:
    """Where our stack slots sit in the target's frame, from the aligned instructions that use
    them (`lwz rX, N(r1)`, `addi rX, r1, N`): runs of slots that moved by the same amount, the
    frame sizes, and where locals are missing (or too small) in ours."""
    if right is None:
        return []
    li, ri = left.get("instructions", []), right.get("instructions", [])
    moves: dict[int, int] = {}
    frame = None
    for t, o in zip(li, ri):
        a, b = _r1_offset(_text(t)), _r1_offset(_text(o))
        if a is None or b is None:
            continue
        if _text(o).startswith("stwu r1"):
            frame = (-b, -a)
            continue
        moves.setdefault(b, a - b)
    if not moves or all(d == 0 for d in moves.values()) and (frame is None or frame[0] == frame[1]):
        return []
    runs: list[list] = []                          # [first ours, last ours, delta]
    for off in sorted(moves):
        if runs and runs[-1][2] == moves[off]:
            runs[-1][1] = off
        else:
            runs.append([off, off, moves[off]])
    out = []
    if frame is not None and frame[0] != frame[1]:
        out.append(f"  stack: frame 0x{frame[0]:X} in ours, 0x{frame[1]:X} in the target ({frame[1] - frame[0]:+#x})")
    for lo, hi, d in runs:
        where = f"0x{lo:X}" if lo == hi else f"0x{lo:X}-0x{hi:X}"
        out.append(f"    ours {where}: " + ("same" if d == 0 else f"target {d:+#x}"))
    for (lo1, hi1, d1), (lo2, hi2, d2) in zip(runs, runs[1:]):
        if d2 > d1:
            out.append(f"    -> the target has 0x{d2 - d1:X} more bytes of locals between ours 0x{hi1:X} and 0x{lo2:X}: "
                       "a local there is bigger, or one is missing (aggregates sit above scalars, earlier declarations higher)")
        elif d2 < d1:
            out.append(f"    -> ours has 0x{d1 - d2:X} more bytes of locals between 0x{hi1:X} and 0x{lo2:X}: "
                       "one is smaller, or not a local, in the target")
    return out


def data_rows(left_sec: dict, right_sec: dict) -> list[str]:
    rows = []
    for side, sec in (("target", left_sec), ("ours", right_sec)):
        off = 0
        for d in sec.get("data_diff", []):
            size = int(d.get("size", 0) or 0)
            raw = base64.b64decode(d["data"]) if d.get("data") else b""
            if d.get("kind"):
                rows.append(f"  {side:<6} +0x{off:<5X} {MARK.get(d['kind'], d['kind']):>3} 0x{size:X}  {raw[:16].hex()}{'…' if len(raw) > 16 else ''}")
            if d.get("data"):
                off += size
    return rows


def _objects(elf, section: str) -> list[tuple[int, int, str]]:
    """(offset, size, name) of the objects in one section, in address order."""
    sec = elf.section(section)
    if sec is None:
        return []
    found = {}
    for x in elf.symbols:
        if x.shndx == sec.index and x.type in (0, 1) and x.name and x.size:
            found.setdefault(x.value, (x.value, x.size, x.name))
    return sorted(found.values())


def _first_uses(elf, section: str) -> dict[int, tuple[int, int, str]]:
    """Section offset -> (function rank in .text order, offset of the use inside it, function name)
    of the first use that relocates against it."""
    out: dict[int, tuple[int, int, str]] = {}
    fns = sorted(elf.functions().values(), key=lambda f: next(
        (x.value for x in elf.symbols if x.name == f.name and x.type == 2), 0))
    for rank, f in enumerate(fns):
        for r in sorted(f.relocs, key=lambda r: r.offset):
            if r.target_section == section:
                out.setdefault(r.target_value + r.addend, (rank, r.offset, f.name))
    return out


def _first_users(elf, section: str) -> dict[int, str]:
    """Section offset -> the first function (in .text order) that relocates against it."""
    return {off: use[2] for off, use in _first_uses(elf, section).items()}


def stripped_hints(offsets: list[int], uses: dict[int, tuple[int, int, str]]) -> dict[int, str]:
    """Pool objects the kept code cannot have placed where they are. A compiler appends each
    literal to the pool at its first use, so first uses run in code order along the pool; an object
    whose first kept use comes after the first use of an object placed behind it was used earlier by
    code the linker dropped (a dead-stripped function) or code we write later. offsets are the
    target's objects in pool order; uses as _first_uses gives them. Returns offset -> hint."""
    hints: dict[int, str] = {}
    later = [(uses[o], o) for o in offsets if o in uses]
    for i, (use, off) in enumerate(later):
        # function rank only: inside one function the scheduler moves loads, so the use offsets
        # are no evidence
        behind = [(u, o) for u, o in later[i + 1:] if u[0] < use[0]]
        if behind:
            (_, _, who), where = min(behind)
            hints[off] = (f"placed before +0x{where:X} yet its first kept use ({use[2]}) comes after "
                          f"{who}'s: a function dead-stripped before {who} used it first")
    return hints


def _value(raw: bytes) -> str:
    import struct
    text = raw[:8].hex() + ("…" if len(raw) > 8 else "")
    if len(raw) == 4:
        text += f" ({struct.unpack('>f', raw)[0]:g}f)"
    elif len(raw) == 8:
        text += f" ({struct.unpack('>d', raw)[0]:g})"
    elif raw.rstrip(b"\0") and all(32 <= c < 127 for c in raw.rstrip(b"\0")):
        text = repr(raw.rstrip(b"\0").decode())
    return text


def literal_rows(target, base, section: str) -> list[str]:
    """The unit's objects in one section side by side, by offset: what each holds and the first
    function using it. Compilers place literals in order of first use, so this is the order to
    write them in when a function differs only by a literal's offset."""
    rows = []
    t, b = _objects(target, section), _objects(base, section)
    tsec, bsec = target.section(section), base.section(section)
    if not t or bsec is None:
        return rows
    tuses = _first_uses(target, section)
    tu, bu = {off: use[2] for off, use in tuses.items()}, _first_users(base, section)
    stripped = stripped_hints([off for off, _, _ in t], tuses)
    ours = {off: (size, name) for off, size, name in b}
    data = lambda sec, off, size: b"" if sec.type == 8 else sec.data[off:off + size]
    rows.append(f"{section}: offset, target (first user), ours (first user)")
    for off, size, name in t:
        theirs = data(tsec, off, size)
        mine = ours.get(off)
        # the same bytes over the target object's range (dtk's label may span several of ours)
        same = mine is not None and data(bsec, off, size) == theirs
        right = f"{mine[1]} {_value(data(bsec, off, mine[0]))} ({bu.get(off, '-')})" if mine else "-"
        rows.append(f"  {'  ' if same else '!='} +0x{off:<4X} {name} {_value(theirs)} ({tu.get(off, '-')})  |  {right}")
        if off in stripped and not same:
            rows.append(f"        stripped? {stripped[off]}")
    return rows


def run(cfg: Config, args: argparse.Namespace) -> int:
    try:
        unit = resolve_unit(cfg, args.target)
    except MatchError as e:
        print(f"error: {e}")
        return USAGE
    build = fresh.build(cfg, [unit["base_path"]])
    if build.returncode != 0:
        print("BUILD ERROR:")
        print("\n".join(line for line in build.stdout.splitlines() if "Usage Warning" not in line)[-3000:])
        return BUILD_ERROR
    from .objcache import objdiff_json
    doc, err = objdiff_json(cfg, unit["target_path"], unit["base_path"])
    if doc is None:
        print(f"error: objdiff failed: {err}")
        return USAGE
    if args.literals:
        from .elf import Elf
        target, base = Elf.parse((cfg.root / unit["target_path"]).read_bytes()), Elf.parse((cfg.root / unit["base_path"]).read_bytes())
        for section in args.literals.split(","):
            for row in literal_rows(target, base, section):
                print(row)
        hint(f"check {args.target}")
        return OK
    left, right = doc["left"], doc.get("right", {})
    rsyms = right.get("symbols", [])
    shown = 0
    for sym in left.get("symbols", []):
        if "instructions" not in sym:
            continue
        pct = float(sym.get("match_percent", 0.0))
        if args.function and sym["name"] != args.function:
            continue
        if pct == 100.0 and not args.function:
            continue
        # Pair by name; objdiff's target_symbol is a 1-based symbol-table index (0 = null symbol).
        other = next((r for r in rsyms if r["name"] == sym["name"] and "instructions" in r), None)
        idx = sym.get("target_symbol")
        if other is None and isinstance(idx, int) and 0 < idx <= len(rsyms):
            other = rsyms[idx - 1]
        print(f"{sym['name']}  {pct:.2f}%" + ("" if other else "  (missing from compiled object)"))
        print(f"  {'addr':>5}  {'target':<42} {'':>3}  ours")
        for row in function_rows(sym, other, args.context):
            print(row)
        for row in stack_rows(sym, other):
            print(row)
        shown += 1
    rsecs = {s["name"]: s for s in right.get("sections", [])}
    for sec in left.get("sections", []):
        if sec.get("kind") in ("SECTION_DATA", "SECTION_BSS") and float(sec.get("match_percent", 0.0)) < 100.0 and not args.function:
            print(f"{sec['name']}  {float(sec.get('match_percent', 0.0)):.2f}% (raw objdiff, before dk's weak-copy allowances)")
            for row in data_rows(sec, rsecs.get(sec["name"], {})):
                print(row)
            shown += 1
    if not shown:
        print("no differences")
        hint(f"check {args.target}")
        return OK
    hint(f"check {args.target}")
    return MISMATCH


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("target", help="unit (source path or name) or a function inside one")
    p.add_argument("--function", help="show only this function (even if exact)")
    p.add_argument("--context", type=int, default=2, help="matching lines shown around each difference (default 2)")
    p.add_argument("--literals", nargs="?", const=".sdata2,.rodata", metavar="SECTIONS",
                   help="list the unit's objects in these sections (default .sdata2,.rodata) side by side with "
                        "values and first users: the order to write literals in; a differing object whose first "
                        "kept user comes after a later object's is flagged 'stripped?' (a dead-stripped function "
                        "used it first: the case for a labelled stand-in)")


COMMAND = Command("diff", "matching", "side-by-side target vs compiled instructions (mismatches + context)", run, _configure)
