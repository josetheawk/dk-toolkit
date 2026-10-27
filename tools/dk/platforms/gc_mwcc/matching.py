"""Compile a unit, score it against its target object, and check in-unit layout.

objdiff scores functions by name. It does not notice when every function matches but
their order inside the unit differs, which still breaks the linked binary, so layout is
compared separately from the ELF symbol offsets.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ...core import floor, quality
from ...core.config import Config
from ...core.util import run_ninja
from .dtk_symbols import read_symbols, read_units
from .elf import Elf


class MatchError(Exception):
    """Environment or input problem (unknown unit, missing build files)."""


@dataclass
class FunctionScore:
    name: str
    size: int
    percent: float
    note: str = ""          # why a function that differs only in relocations is not equivalent


@dataclass
class UnitScore:
    unit: str
    functions: list[FunctionScore] = field(default_factory=list)
    data: dict[str, float] = field(default_factory=dict)        # section -> percent
    layout_problems: list[str] = field(default_factory=list)
    stripped: list[str] = field(default_factory=list)    # in source, absent from target (linker drops them)
    weak_copies: list[str] = field(default_factory=list)  # data the original link discarded (weak copies, stripped-only)
    data_problems: list[str] = field(default_factory=list)  # vtable slots pointing at other functions

    @property
    def code_percent(self) -> float:
        total = sum(f.size for f in self.functions)
        return 100.0 if not total else sum(f.size * f.percent for f in self.functions) / total

    @property
    def exact(self) -> bool:
        return (all(f.percent == 100.0 for f in self.functions)
                and all(p == 100.0 for p in self.data.values())
                and not self.data_problems and not self.layout_problems)


def load_objdiff_units(cfg: Config) -> list[dict]:
    path = cfg.root / "objdiff.json"
    if not path.is_file():
        raise MatchError("objdiff.json missing (run ./dk setup or ./dk build)")
    return json.loads(path.read_text())["units"]


def resolve_unit(cfg: Config, query: str) -> dict:
    """Find a source-backed objdiff unit by unit name, source path, or contained function."""
    units = [u for u in load_objdiff_units(cfg) if u.get("base_path")]
    q = query.removeprefix("src/")
    for u in units:
        src = u.get("metadata", {}).get("source_path", "").removeprefix("src/")
        if q in (u["name"], u["name"].split("/", 1)[-1], src) or (src and src.rsplit(".", 1)[0] == q):
            return u
    symbols = {s.name: s for s in read_symbols(cfg.root / cfg.platform_options["symbols"])}
    sym = symbols.get(query)
    if sym is None:
        raise MatchError(f"no unit or symbol named {query!r} (declare one with ./dk unit add)")
    for split_unit in read_units(cfg.root / cfg.platform_options["splits"]):
        if any(sec == sym.section and start <= sym.address < end for sec, start, end in split_unit.splits):
            stem = split_unit.name.rsplit(".", 1)[0]
            for u in units:
                if u["name"].split("/", 1)[-1] == stem:
                    return u
            raise MatchError(f"{query} is in {split_unit.name}, which has no source unit in units.toml")
    raise MatchError(f"{query} at 0x{sym.address:08X} is not inside any declared unit (./dk unit add)")


def compile_command(cfg: Config, base_path: str, locked: bool = True) -> str:
    """The project's own compile command for an object, without the dependency post-step.
    `ninja -t commands` only reads build.ninja; locked=False skips the project lock."""
    proc = run_ninja(cfg.root, ["-t", "commands", base_path]) if locked else \
        subprocess.run(["ninja", "-t", "commands", base_path], cwd=cfg.root, capture_output=True, text=True)
    if proc.returncode != 0 or not proc.stdout.strip():
        raise MatchError(f"no build command for {base_path}: {proc.stderr.strip()}")
    return proc.stdout.strip().splitlines()[-1].split(" && ", 1)[0]


def retarget(command: str, out_dir: Path, compiler: str | None = None, extra_flags: tuple[str, ...] = ()) -> list[str]:
    """Rewrite a compile command: output dir, compiler version, extra trailing flags."""
    args = shlex.split(command)
    out: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "-MMD":
            i += 1
            continue
        if a == "-o":
            out += ["-o", str(out_dir)]
            i += 2
            continue
        if compiler and re.search(r"compilers/[^/]+/[^/]+/mwcceppc\.exe$", a):
            a = re.sub(r"compilers/[^/]+/[^/]+/mwcceppc\.exe$", f"compilers/{compiler}/mwcceppc.exe", a)
        if a == "-c" and extra_flags:
            out += list(extra_flags)
        out.append(a)
        i += 1
    return out


def run_compile(cfg: Config, args: list[str]) -> tuple[bool, str]:
    proc = subprocess.run(args, cwd=cfg.root, capture_output=True, text=True)
    text = "\n".join(line for line in (proc.stdout + proc.stderr).splitlines()
                     if "Usage Warning" not in line and "Specified directory" not in line)
    return proc.returncode == 0, text.strip()


def _function_offsets(path: Path) -> dict[str, tuple[str, int, int]]:
    elf = Elf.parse(path.read_bytes())
    out = {}
    for s in elf.symbols:
        if s.type == 2 and 0 < s.shndx < len(elf.sections):
            out[s.name] = (elf.sections[s.shndx].name, s.value, s.size)
    return out


STB_WEAK = 2


def _weak_extras(target: Path, base: Path) -> tuple[dict[str, list[tuple[int, int, str]]], set[str]]:
    """Data the compiled object has that the original link discarded, per section, and which of
    those names the original linker may have zero-filled in place (weak copies and their closure).

    Inline virtuals make every TU emit a weak copy of the base class's vtable (and similar);
    the linker keeps only the first. Objects referenced *only* by such discarded copies (the
    local RTTI record a weak vtable points at) were dead-stripped with them, transitively.
    Functions the target lacks were dead-stripped too, and with them the literals and statics
    only they referenced, plus objects the target lacks that nothing references (those are
    removed, never zero-filled).
    """
    want = {s.name for s in Elf.parse(target.read_bytes()).symbols if s.shndx != 0}   # defined there
    elf = Elf.parse(base.read_bytes())
    sized = [s for s in elf.symbols if 0 < s.shndx < len(elf.sections) and s.size and s.name]
    extras = {s.name for s in sized if s.bind == STB_WEAK and s.type != 2 and s.name not in want}

    # Who references each symbol: the sized symbol (function or object) containing each relocation.
    referrers: dict[str, set[str]] = {}
    by_section: dict[str, list[tuple[int, int, str]]] = {}
    for s in sized:
        by_section.setdefault(elf.sections[s.shndx].name, []).append((s.value, s.value + s.size, s.name))

    def symbol_at(section: str, offset: int) -> str | None:
        return next((n for lo, hi, n in by_section.get(section, []) if lo <= offset < hi), None)

    for sec in elf.sections:
        relocs = elf.relocations(sec.index)
        if not relocs:
            continue
        for r in relocs:
            holder = symbol_at(sec.name, r.offset) or f"@{sec.name}"
            # Resolve by address: references often go through a section symbol plus an addend.
            target_name = symbol_at(r.target_section, r.target_value + r.addend) if r.target_section else None
            referrers.setdefault(target_name or r.symbol, set()).add(holder)
    objects = {s.name for s in sized if s.type != 2 and s.name not in want}
    section_of = {s.name: elf.sections[s.shndx].name for s in sized}
    # A C++ __sinit that reaches a section through its section symbol ("...bss.0"+N,
    # "...data.0"+N) walks its global-object/destructor-chain records, or the literals of its
    # initializers, by register arithmetic: those objects have no relocation of their own, yet
    # the link keeps them.
    sinit_secs = set()
    for sec in elf.sections:
        for r in elf.relocations(sec.index):
            holder = symbol_at(sec.name, r.offset) or ""
            if holder.startswith("__sinit") and r.symbol.startswith("...") and r.target_section:
                sinit_secs.add(r.target_section)
    # Objects flagged force-active (`#pragma force_active on`) are never dead-stripped, even when
    # nothing references them or only discarded weak copies do; weak copies still lose to the
    # first definition, so the flag only protects the unit's own (non-weak) objects.
    forced = {s.name for s in sized if s.bind != STB_WEAK and s.name in elf.force_active()}
    objects -= forced
    zero_fillable = _discarded(objects, referrers, extras)
    stripped = {s.name for s in sized if s.type == 2 and s.name not in want}
    # The target lacks these and nothing in the unit uses them: the link found no user at all.
    unused = {n for n in objects if not referrers.get(n) and section_of.get(n) not in sinit_secs}
    extras = _discarded(objects, referrers, zero_fillable | stripped | unused)

    out: dict[str, list[tuple[int, int, str]]] = {}
    for s in sized:
        if s.name in extras and s.type != 2:
            out.setdefault(elf.sections[s.shndx].name, []).append((s.value, s.value + s.size, s.name))
    return out, zero_fillable


def _kept_beside_zero_filled(base: Path, weak: dict[str, list[tuple[int, int, str]]],
                             zero_fillable: set[str]) -> set[str]:
    """Objects of `zero_fillable` that the link keeps after all: a local object in a section
    where weak copies are zero-filled in place keeps its slot, its bytes and its relocations,
    and so does everything it points at, transitively (a base-class list next to a zero-filled
    weak vtable keeps the base's RTTI record and class name in .sdata alive)."""
    elf = Elf.parse(base.read_bytes())
    sized = [s for s in elf.symbols if 0 < s.shndx < len(elf.sections) and s.size and s.name]
    by_section: dict[str, list[tuple[int, int, str]]] = {}
    for s in sized:
        by_section.setdefault(elf.sections[s.shndx].name, []).append((s.value, s.value + s.size, s.name))

    def symbol_at(section: str, offset: int) -> str | None:
        return next((n for lo, hi, n in by_section.get(section, []) if lo <= offset < hi), None)

    points_at: dict[str, set[str]] = {}
    for sec in elf.sections:
        for r in elf.relocations(sec.index) or []:
            holder = symbol_at(sec.name, r.offset)
            to = symbol_at(r.target_section, r.target_value + r.addend) if r.target_section else r.symbol
            if holder and to:
                points_at.setdefault(holder, set()).add(to)
    weak_names = {s.name for s in sized if s.bind == STB_WEAK}
    section_of = {s.name: elf.sections[s.shndx].name for s in sized}
    zero_filled_secs = {sec for sec, ranges in weak.items() if any(n in weak_names for _, _, n in ranges)}
    kept = {n for n in zero_fillable if n not in weak_names and section_of.get(n) in zero_filled_secs}
    todo = list(kept)
    while todo:
        for to in points_at.get(todo.pop(), set()):
            if to in zero_fillable and to not in weak_names and to not in kept:
                kept.add(to)
                todo.append(to)
    return kept


def _discarded(objects: set[str], referrers: dict[str, set[str]], seeds: set[str]) -> set[str]:
    """`seeds` plus every one of `objects` whose referrers are all discarded, transitively.
    An object with no known referrer is kept: it may be reached from outside the unit."""
    out = set(seeds)
    changed = True
    while changed:
        changed = False
        for name in sorted(objects - out):
            who = referrers.get(name, set())
            if who and who <= out:
                out.add(name)
                changed = True
    return out


def _alignment(value: int, size: int) -> int:
    """Natural alignment of an object: a power of two up to 8, no larger than its size and
    dividing its offset in the compiled object."""
    align = 8
    while align > 1 and (align > size or value % align):
        align //= 2
    return align


def _packed_offsets(base: Elf, section: str, dropped: set[str]) -> dict[str, int]:
    """Where the original link put each kept object of a section: in place when nothing was
    discarded, else packed in order at its alignment."""
    b = next((x for x in base.sections if x.name == section), None)
    if b is None:
        return {}
    syms = sorted((x for x in base.symbols if x.shndx == b.index and x.size and x.name), key=lambda x: x.value)
    if not any(x.name in dropped for x in syms):
        return {x.name: x.value for x in syms}
    out, cursor = {}, 0
    for x in syms:
        if x.name in dropped:
            continue
        align = _alignment(x.value, x.size)
        cursor = (cursor + align - 1) & ~(align - 1)
        out[x.name] = cursor
        cursor += x.size
    return out


def _nobits_layout_problems(target: Elf, base: Elf, dropped: dict[str, set[str]]) -> list[str]:
    """BSS has no bytes, so a byte diff passes any order: compare the offsets of objects both
    sides name (after the link's discards)."""
    problems = []
    for t in target.sections:
        if t.type != 8 or not t.name:
            continue
        want = {x.name: x.value for x in target.symbols if x.shndx == t.index and x.size and x.name}
        have = _packed_offsets(base, t.name, dropped.get(t.name, set()))
        for name in sorted(set(want) & set(have), key=lambda n: want[n]):
            if want[name] != have[name]:
                problems.append(f"{name}: {t.name}+0x{have[name]:X}, target {t.name}+0x{want[name]:X}")
    return problems


def _object_at(elf: Elf, section: str, offset: int):
    """(symbol, offset inside it) of the sized object holding section+offset."""
    sec = elf.section(section)
    if sec is None:
        return None
    for s in elf.symbols:
        if s.shndx == sec.index and s.size and s.value <= offset < s.value + s.size:
            return s, offset - s.value
    return None


def _relocated_section_matches(target: Elf, base: Elf, section: str, dropped: dict[str, set[str]]) -> bool:
    """A data section whose bytes match and whose pointers point at the same things.

    objdiff compares a data relocation by where it points; when the original link dropped an
    object earlier in the pointed-to section (a weak vtable), our pointers land at shifted
    offsets of identical objects. Equal here: same size and bytes (relocated words masked),
    relocations at the same offsets and of the same type, each pointing at an object with the
    same bytes and the same offset inside it, under the same name unless both are local
    compiler/auto names. Only one level deep: what the pointed-at objects point to is not
    followed."""
    t, b = target.section(section), base.section(section)
    if t is None or b is None or t.type == 8:
        return False
    tr = {r.offset: r for r in target.relocations(t.index)}
    br = {r.offset: r for r in base.relocations(b.index)}
    data = b.data
    gone = dropped.get(section, set())
    if gone:                                       # re-pack the kept objects, relocations included
        packed = _packed_offsets(base, section, gone)
        syms = {x.name: x for x in base.symbols if x.shndx == b.index and x.size and x.name}
        end = max((packed[n] + syms[n].size for n in packed), default=0)
        buf = bytearray(end)
        moved = {}
        for n, new_off in packed.items():
            x = syms[n]
            buf[new_off:new_off + x.size] = b.data[x.value:x.value + x.size]
            for o, r in br.items():
                if x.value <= o < x.value + x.size:
                    moved[new_off + o - x.value] = r
        data, br = bytes(buf), moved
    if len(data) > t.size or t.size - len(data) >= 8 or t.data[len(data):].strip(b"\0"):
        return False                               # only zero alignment padding may follow
    if set(tr) != set(br):
        return False
    masked = lambda d, rels: bytes(0 if any(o <= i < o + 4 for o in rels) else x for i, x in enumerate(d))
    if masked(t.data[:len(data)], tr) != masked(data, br):
        return False
    # a unit's private copy (an RTTI record, a literal) has a local binding; dtk names the
    # original's copies lbl_/gap_, the compiler names ours, so only bytes and size must agree
    local = lambda sym: sym.bind == 0 or not sym.name or sym.name.startswith(("@", "lbl_", "gap_")) or "$" in sym.name
    for off, r in tr.items():
        q = br[off]
        if r.type != q.type:
            return False
        if not r.target_section or not q.target_section:
            if r.symbol != q.symbol or r.addend != q.addend:
                return False
            continue
        a = _object_at(target, r.target_section, r.target_value + r.addend)
        c = _object_at(base, q.target_section, q.target_value + q.addend)
        if a is None or c is None or a[1] != c[1] or a[0].size != c[0].size:
            return False
        if a[0].name != c[0].name and not (local(a[0]) and local(c[0])):
            return False
        ta = target.section(r.target_section)
        cb = base.section(q.target_section)
        da = ta.data[a[0].value:a[0].value + a[0].size] if ta.type != 8 else b""
        dc = cb.data[c[0].value:c[0].value + c[0].size] if cb.type != 8 else b""
        if masked(da, {x.offset - a[0].value for x in target.relocations(ta.index)
                        if a[0].value <= x.offset < a[0].value + a[0].size}) != \
                masked(dc, {x.offset - c[0].value for x in base.relocations(cb.index)
                            if c[0].value <= x.offset < c[0].value + c[0].size}):
            return False
    return True


def _packed_section_matches(target: Elf, base: Elf, section: str, dropped: set[str]) -> bool:
    """Model the original link of one relocation-free section: remove the discarded objects,
    pack the kept ones in order at their alignment, and compare with the target. Alignment gaps
    and the end may hold zero padding. BSS has no bytes, so each kept object must instead start
    where the target has a label. Literal pools lose entries from the middle this way, which
    objdiff's byte diff cannot line up."""
    t = next((x for x in target.sections if x.name == section), None)
    b = next((x for x in base.sections if x.name == section), None)
    if t is None or b is None or target.relocations(t.index) or base.relocations(b.index):
        return False
    kept = sorted((x.value, x.size) for x in base.symbols
                  if x.shndx == b.index and x.size and x.name not in dropped)
    if not kept:
        return False
    nobits = b.type == 8
    labels = {x.value for x in target.symbols if x.shndx == t.index and x.size}
    cursor = 0
    for value, size in kept:
        align = _alignment(value, size)
        pos = (cursor + align - 1) & ~(align - 1)
        if nobits:
            if pos not in labels:
                return False
        elif t.data[cursor:pos].strip(b"\0") or t.data[pos:pos + size] != b.data[value:value + size]:
            return False
        cursor = pos + size
    rest = t.size - cursor
    return 0 <= rest < 8 and (nobits or not t.data[cursor:].strip(b"\0"))


def _only_trailing_padding(left: dict, right: dict) -> bool:
    """True if a data section differs only by zero bytes (< 32) at the end of the target's copy:
    alignment padding before the next object, which a split boundary may give to either side."""
    import base64

    chunks = left.get("data_diff", [])
    if [i for i, d in enumerate(chunks) if d.get("kind")] != [len(chunks) - 1]:
        return False
    raw = base64.b64decode(chunks[-1].get("data", ""))
    return (0 < len(raw) < 32 and not raw.strip(b"\0")
            and not any(d.get("kind") and d.get("data") for d in right.get("data_diff", [])))


def _only_weak_extras(left: dict, right: dict, ranges: list[tuple[int, int, str]],
                      zero_fillable: set[str] | None = None) -> bool:
    """True if a data section differs only where weak copies explain it.

    Two shapes, both from the original link discarding duplicate weak objects:
      - base-only bytes inside a weak symbol the target does not define (trailing copy dropped)
      - target bytes that are all zero, of the size of such a weak symbol (a copy in the middle
        of the section that the original linker zero-filled instead of removing)
    """
    import base64

    weak_sizes = sorted(hi - lo for lo, hi, name in ranges if zero_fillable is None or name in zero_fillable)
    for d in left.get("data_diff", []):
        if d.get("kind") and d.get("data"):
            raw = base64.b64decode(d["data"])
            if raw.strip(b"\0") or len(raw) not in weak_sizes:
                return False        # the target has real bytes the compiled object does not reproduce
            weak_sizes.remove(len(raw))
    # Base-only chunks must sit inside the extras, where adjacent extras (and the alignment
    # padding after each) form one span: objdiff reports a string, its padding and the next
    # weak object as a single chunk. All-zero chunks (objects whose pointers are relocations)
    # align ambiguously in objdiff's diff, so those may be explained by size.
    spans: list[list[int]] = []
    for lo, hi, _ in sorted(ranges):
        hi = (hi + 3) & ~3
        if spans and lo <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], hi)
        else:
            spans.append([lo, hi])
    ranges = [(lo, hi, "") for lo, hi in spans]
    budget = sum(hi - lo for lo, hi, _ in ranges)
    offset = 0
    for d in right.get("data_diff", []):
        size = int(d.get("size", 0) or 0)
        if d.get("kind") and d.get("data"):
            raw = base64.b64decode(d["data"])
            # a chunk may start in the zero alignment padding just before a span
            if any(lo - 7 <= offset and offset + size <= hi and not raw[:max(0, lo - offset)].strip(b"\0")
                   for lo, hi, _ in ranges):
                budget -= size
            elif not base64.b64decode(d["data"]).strip(b"\0") and size <= budget:
                budget -= size
            else:
                return False
        if d.get("data"):
            offset += size
    return True


_AUTO_NAME = re.compile(r"^(?:fn|lbl)_[0-9A-F]{8}$")


def _slots(elf: Elf, sym) -> dict[int, tuple[str, int]]:
    """Offset in the object -> (function or object it points at, addend), section-relative
    relocations resolved to the symbol defined there."""
    defined = {(elf.sections[x.shndx].name, x.value): x.name for x in elf.symbols
               if 0 < x.shndx < len(elf.sections) and x.name and x.type in (1, 2)}
    out = {}
    for r in elf.relocations(sym.shndx):
        if sym.value <= r.offset < sym.value + sym.size:
            name, addend = r.symbol, r.addend
            if r.target_section and name == r.target_section:
                name = defined.get((r.target_section, r.target_value + r.addend), name)
                addend = 0 if name != r.target_section else addend
            out[r.offset - sym.value] = (name, addend)
    return out


def vtable_problems(target: Elf, base: Elf) -> list[str]:
    """Slots of the vtables both objects define that point at different functions.

    objdiff compares a vtable's bytes, which are zero where relocations go, so an override
    declared in the wrong slot scores 100% and only the link catches it. Names are compared:
    the target's come from symbols.txt, ours from the compiler. A slot is not judged while the
    target still has a placeholder (fn_ADDR) there that ours cannot be checked against: one of
    the unit's own functions not renamed yet, or an unnamed function elsewhere."""
    def kinds(elf):
        return {x.name: (x.type, x.bind) for x in elf.symbols if 0 < x.shndx < len(elf.sections) and x.name}
    t_kind, b_kind = kinds(target), kinds(base)
    data = lambda n, k: k.get(n, (0, 0))[0] == 1 or n.startswith(("__RTTI__", "lbl_")) or re.fullmatch(r"@\d+", n)
    vtables = lambda elf: {x.name: x for x in elf.symbols if x.type == 1 and 0 < x.shndx < len(elf.sections)
                           and x.name.startswith("__vt__")}
    tv, bv = vtables(target), vtables(base)
    problems = []
    for name in sorted(set(tv) & set(bv)):
        ts, bs = _slots(target, tv[name]), _slots(base, bv[name])
        size = min(tv[name].size, bv[name].size)      # dtk's symbol may run on over a weak copy after it
        for off in sorted(o for o in set(ts) | set(bs) if o < size):
            t, b = ts.get(off), bs.get(off)
            if t == b:
                continue
            if t and data(t[0], t_kind) or b and data(b[0], b_kind):
                continue                       # RTTI and other data pointers: judged with the data
            if t and b and _AUTO_NAME.match(t[0]):
                ours_override = b_kind.get(b[0]) == (2, 1)          # a global function this unit defines
                if t[0] in t_kind or not ours_override:
                    continue                   # unrenamed own function, or an unnamed one elsewhere
            problems.append(f"{name}+0x{off:X}: target {t[0] if t else 'null'}, ours {b[0] if b else 'null'}")
    return problems


# bits each relocation type writes, from the relocation's offset
_RELOC_MASK = {1: b"\xff\xff\xff\xff", 4: b"\xff\xff", 5: b"\xff\xff", 6: b"\xff\xff",
               10: b"\x03\xff\xff\xfc", 109: b"\x00\x1f\xff\xff"}


def _masked(code: bytes, relocs) -> bytes | None:
    out = bytearray(code)
    for r in relocs:
        mask = _RELOC_MASK.get(r.type)
        if mask is None:
            return None
        for i, m in enumerate(mask):
            if r.offset + i < len(out):
                out[r.offset + i] &= ~m & 0xFF
    return bytes(out)


def relocation_difference(target: Elf, base: Elf, tf, bf, name_at=None) -> str | None:
    """None when two functions are the same once linked: equal code outside the relocated
    fields, and every relocation resolves to the same thing, either the same symbol or the same
    offset of the same section of this unit (our `.data.0+0xA0` or `@123` literal against dtk's
    `lbl_...`), with the same bytes there. Otherwise the first relocation that differs, or ""
    when the code itself differs. `name_at(address)`: symbols.txt's name there, to explain a
    call through a stale C name."""
    if len(tf.code) != len(bf.code):
        return ""
    tr, br = {r.offset: r for r in tf.relocs}, {r.offset: r for r in bf.relocs}
    mt, mb = _masked(tf.code, tf.relocs), _masked(bf.code, bf.relocs)
    if set(tr) != set(br) or mt is None or mb is None or mt != mb:
        return ""
    for off, r in sorted(tr.items()):
        q = br[off]
        if r.type != q.type:
            return ""
        if r.symbol == q.symbol and r.addend == q.addend:
            continue
        theirs, ours = r.target_value + r.addend, q.target_value + q.addend
        label = f"+0x{off:X}: target {r.symbol}{r.addend:+#x}, ours {q.symbol}{q.addend:+#x}"
        if r.target_section and q.target_section and r.target_section != q.target_section:
            return (f"{label}: the target has it in {r.target_section}, ours in {q.target_section} "
                    f"(a const literal where the original has a non-const static, or the reverse)")
        stale = _AUTO_NAME.match(q.symbol) and q.symbol.startswith("fn_") and (
            r.symbol.startswith(q.symbol + "__") or name_at is not None and name_at(int(q.symbol[3:], 16)) == r.symbol)
        if stale:
            return (f"{label}: ours still uses the C name {q.symbol}, which symbols.txt now names {r.symbol}: "
                    f"declare it as that member (its class's header) and call it as one")
        if not r.target_section and q.target_section and q.target_section != ".text":
            return (f"{label}: {r.symbol} lies outside this unit's split, so the unit's data range misses it "
                    f"(./dk unit propose shows the range; widen it with ./dk unit set)")
        if not r.target_section and q.target_section == ".text":
            return f"{label}: the original calls another unit's {r.symbol}, ours a function of this unit"
        if not r.target_section or r.target_section != q.target_section:
            return f"{label} (a different symbol)"
        if theirs != ours:
            return (f"{label}: the target has it at {r.target_section}+0x{theirs:X}, ours at +0x{ours:X} "
                    f"(define the unit's literals/strings in the target's order)")
        if r.target_section == ".text":
            continue                                   # the same code address
        ts, bs = target.section(r.target_section), base.section(q.target_section)
        if ts is None or bs is None:
            return f"{label} (section missing)"
        if ts.type == 8:
            continue                                   # .bss: the same offset is the same object
        owner = [x for x in target.symbols if 0 < x.shndx < len(target.sections) and x.shndx == ts.index
                 and x.size and x.value <= theirs < x.value + x.size and x.type in (1, 2)]
        stop = min((x.value + x.size for x in owner), default=theirs + 4)
        # dtk's label may run on over neighbouring data; the compiler's object has its true size
        mine = [x for x in base.symbols if 0 < x.shndx < len(base.sections) and x.shndx == bs.index
                and x.size and x.value <= ours < x.value + x.size and x.type in (1, 2) and x.name != q.target_section]
        if mine:
            stop = min(stop, theirs + min(x.value + x.size for x in mine) - ours)
        if ts.data[theirs:stop] != bs.data[ours:ours + stop - theirs]:
            return f"{label} (different bytes at {r.target_section}+0x{theirs:X})"
        # the object's own pointers (a jump table's case labels, a table of strings) must land
        # on the same offsets too: their bytes are zero in both objects
        inner = lambda elf, sec, lo, hi: {x.offset - lo: (x.target_section, x.target_value + x.addend)
                                          if x.target_section else (x.symbol, x.addend)
                                          for x in elf.relocations(sec.index) if lo <= x.offset < hi}
        if inner(target, ts, theirs, stop) != inner(base, bs, ours, ours + stop - theirs):
            return f"{label} (its pointers differ at {r.target_section}+0x{theirs:X})"
    return None


_NAMES: dict = {}


def _names_by_address(cfg: Config) -> dict[int, str]:
    """symbols.txt address -> name, cached per file version (read only to explain a near miss)."""
    path = cfg.root / cfg.platform_options["symbols"]
    key = (path, path.stat().st_mtime_ns)
    if key not in _NAMES:
        _NAMES.clear()
        _NAMES[key] = {s.address: s.name for s in read_symbols(path) if s.type == "function"}
    return _NAMES[key]


def score(cfg: Config, unit_name: str, target: Path, base: Path) -> UnitScore:
    objdiff = cfg.root / "build" / "tools" / "objdiff-cli"
    # data_value: a relocation only matches if the data it points at matches too (catches a wrong
    # float literal in otherwise identical code, which the lenient default scores as 100%).
    proc = subprocess.run([str(objdiff), "diff", "-1", str(target), "-2", str(base), "-o", "-", "--format", "json",
                           "-c", "functionRelocDiffs=data_value"],
                          cwd=cfg.root, capture_output=True, text=True)
    if proc.returncode != 0:
        raise MatchError(f"objdiff failed: {proc.stderr.strip()}")
    doc = json.loads(proc.stdout)
    left, right = doc["left"], doc.get("right", {})
    result = UnitScore(unit_name)
    for sym in left.get("symbols", []):
        if "instructions" in sym:
            result.functions.append(FunctionScore(sym["name"], int(sym.get("size", 0)), float(sym.get("match_percent", 0.0))))
    weak, zero_fillable = _weak_extras(target, base)
    target_elf, base_elf = Elf.parse(target.read_bytes()), Elf.parse(base.read_bytes())
    # dtk labels the zero padding between functions aligned by -func_align as gap_ functions;
    # a compiled object has no symbol there, so they are padding, not missing functions
    padding = {n for n, f in target_elf.functions().items() if n.startswith("gap_") and not f.code.strip(b"\0")}
    result.functions = [f for f in result.functions if f.name not in padding]
    target_fns, base_fns = target_elf.functions(), base_elf.functions()
    name_at = lambda address: _names_by_address(cfg).get(address)
    for f in result.functions:
        if f.percent < 100.0 and f.name in target_fns and f.name in base_fns:
            why = relocation_difference(target_elf, base_elf, target_fns[f.name], base_fns[f.name], name_at)
            if why is None:
                f.percent = 100.0          # the same addresses and bytes, under other names
            elif why:
                f.note = why
    right_sections = {sec["name"]: sec for sec in right.get("sections", [])}
    for sec in left.get("sections", []):
        if sec.get("kind") in ("SECTION_DATA", "SECTION_BSS"):
            pct = float(sec.get("match_percent", 0.0))
            ranges = weak.get(sec["name"], [])
            if pct < 100.0 and ranges and _only_weak_extras(sec, right_sections.get(sec["name"], {}), ranges, zero_fillable):
                pct = 100.0
                result.weak_copies += [name for _, _, name in ranges]
            if pct < 100.0 and _only_trailing_padding(sec, right_sections.get(sec["name"], {})):
                pct = 100.0
            dropped = {name for _, _, name in ranges}
            if pct < 100.0 and _packed_section_matches(target_elf, base_elf, sec["name"], dropped):
                pct = 100.0
                result.weak_copies += sorted(dropped)
            if pct < 100.0 and _relocated_section_matches(target_elf, base_elf, sec["name"],
                                                          {k: {n for _, _, n in v} for k, v in weak.items()}):
                pct = 100.0
            if pct < 100.0 and ranges:
                kept = _kept_beside_zero_filled(base, weak, zero_fillable)
                if kept and _relocated_section_matches(target_elf, base_elf, sec["name"],
                                                       {k: {n for _, _, n in v} - kept for k, v in weak.items()}):
                    pct = 100.0
                    result.weak_copies += [name for _, _, name in ranges if name not in kept]
            result.data[sec["name"]] = pct

    # Layout after dead-stripping: functions the target lacks are dropped by the linker when
    # unreferenced (the hash check at accept proves it), so compare packed offsets instead.
    want = {n: v for n, v in _function_offsets(target).items() if n not in padding}
    have = _function_offsets(base)
    stripped = sorted(set(have) - set(want))
    shift: dict[str, int] = {}
    for name, (sec, off, size) in sorted(have.items(), key=lambda kv: (kv[1][0], kv[1][1])):
        removed = sum(have[x][2] for x in stripped if have[x][0] == sec and have[x][1] < off)
        shift[name] = off - removed
    for name, (sec, off, size) in sorted(want.items(), key=lambda kv: (kv[1][0], kv[1][1])):
        got = have.get(name)
        if got is None:
            result.layout_problems.append(f"{name}: missing from compiled object")
        elif got[0] != sec or shift[name] != off:
            result.layout_problems.append(f"{name}: at {got[0]}+0x{shift[name]:X} (after stripping), target {sec}+0x{off:X}")
    result.stripped = stripped
    result.data_problems = vtable_problems(target_elf, base_elf)
    result.layout_problems += _nobits_layout_problems(
        target_elf, base_elf, {sec: {name for _, _, name in ranges} for sec, ranges in weak.items()})
    return result


def render(s: UnitScore) -> list[str]:
    lines = [f"unit {s.unit}"]
    for f in s.functions:
        lines.append(f"  {f.percent:7.2f}%  {f.name:<50} 0x{f.size:X}")
        if f.note:
            lines.append(f"            only a relocation differs: {f.note}")
    if s.data:
        lines.append("  data: " + "  ".join(f"{k} {v:.2f}%" for k, v in sorted(s.data.items())))
    if s.stripped:
        lines.append(f"  dead-stripped (in source, not in target): {', '.join(s.stripped)}")
    if s.weak_copies:
        lines.append(f"  data the original link discarded (weak copies, used only by stripped code; ignored): {', '.join(sorted(set(s.weak_copies)))}")
    if s.data_problems:
        lines.append("  vtables: MISMATCH (a slot points at another function: an override in the wrong slot?)")
        lines += [f"    {p}" for p in s.data_problems]
    lines.append("  layout: " + ("OK" if not s.layout_problems else "MISMATCH"))
    lines += [f"    {p}" for p in s.layout_problems]
    return lines


def unit_commit_paths(cfg: Config, unit: dict) -> list[str]:
    """What a unit's commit must contain: its source, the repo headers it includes (from
    ninja's recorded deps), and the shared config files (units, splits, symbols)."""
    paths = [unit["metadata"]["source_path"]]
    proc = subprocess.run(["ninja", "-t", "deps", unit["base_path"]], cwd=cfg.root, capture_output=True, text=True)
    for line in proc.stdout.splitlines()[1:]:
        dep = Path(line.strip())
        if not line.startswith(" ") or not dep.suffix:
            continue
        full = dep if dep.is_absolute() else cfg.root / dep
        try:
            rel = full.resolve().relative_to(cfg.root)
        except ValueError:
            continue
        if rel.parts and rel.parts[0] not in ("build", "orig"):
            paths.append(str(rel))
    paths += quoted_includes(cfg, cfg.root / unit["metadata"]["source_path"])
    opts = cfg.platform_options
    paths += [opts["units"], opts["splits"], opts["symbols"], cfg.rel(floor.path(cfg))]
    return sorted(set(paths))


_INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.M)


def quoted_includes(cfg: Config, source: Path) -> list[str]:
    """Repo headers reached through #include "..." from `source`, recursively. Ninja's recorded
    deps lag behind a header added since the unit's last full build; this does not."""
    seen: set[Path] = set()
    todo = [source]
    while todo:
        f = todo.pop()
        try:
            text = f.read_text(errors="replace")
        except OSError:
            continue
        for name in _INCLUDE.findall(text):
            for base in (f.parent, cfg.root / "include"):
                h = (base / name).resolve()
                if h.is_file() and h not in seen and h.is_relative_to(cfg.root):
                    seen.add(h)
                    todo.append(h)
                    break
    return [str(h.relative_to(cfg.root)) for h in seen]


def commit_gate(cfg: Config, unit: dict) -> list[str]:
    """Reasons a unit's commit must be refused: forbidden patterns in its own files. Empty list =
    allowed. The progress floor is judged on the clean tree in unitcommit, never here: the shared
    tree's report is stale (hash builds skip it) and holds other agents' work."""
    files = [cfg.root / p for p in unit_commit_paths(cfg, unit) if p.endswith((".c", ".cpp", ".cp", ".h", ".hpp", ".inc"))]
    problems = [f"forbidden: {h}" for h in quality.forbidden_hits(cfg, quality.judged_files(cfg, files))]
    return problems
