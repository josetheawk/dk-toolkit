"""Recover classes from CodeWarrior RTTI.

Layout (verified on a CW GC 1.x-era binary):
    RTTI record (usually .sdata):  { const char* name; BaseEntry* bases /* or 0 */ }
    base list:                      { RTTI* base; s32 offset; } ... terminated by one 0 word.
                                    Lists every base, direct and indirect.
    vtable (.data):                 { RTTI* rtti; s32 offset; fn* slots[] }
                                    offset 0 = primary, negative = secondary (subobject).
Records are accepted only if a vtable or an accepted record's base list
references them (fixpoint), which filters out look-alike {string, ptr} tables.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .dol import Dol

_NAME = re.compile(rb"^[A-Za-z_][A-Za-z0-9_:<>,*& ]{0,200}$")
MAX_BASES = 64
MAX_OFFSET = 0x100000


@dataclass
class Vtable:
    address: int
    offset: int                 # 0 primary, negative secondary
    slots: list[int]


@dataclass
class ClassInfo:
    name: str
    rtti: int
    bases: list[tuple[int, int]] = field(default_factory=list)   # (rtti address, byte offset)
    vtables: list[Vtable] = field(default_factory=list)

    @property
    def primary(self) -> Vtable | None:
        return next((v for v in self.vtables if v.offset == 0), None)


def _s32(x: int) -> int:
    return x - (1 << 32) if x & 0x80000000 else x


def _cstring(dol: Dol, address: int) -> str | None:
    s = dol.section_at(address)
    if s is None or s.kind != "data":
        return None
    start = s.offset + address - s.address
    raw = dol.data[start:min(start + 256, s.offset + s.size)].split(b"\0", 1)[0]
    return raw.decode("ascii") if _NAME.match(raw) else None


def _data_words(dol: Dol) -> dict[int, int]:
    words: dict[int, int] = {}
    for s in dol.sections:
        if s.kind == "data":
            for i, w in enumerate(dol.words(s)):
                words[s.address + 4 * i] = w
    return words


def _read_bases(words: dict[int, int], address: int, records: set[int]) -> list[tuple[int, int]] | None:
    bases = []
    a = address
    for _ in range(MAX_BASES):
        r = words.get(a)
        if r is None:
            return None
        if r == 0:
            return bases
        off = words.get(a + 4)
        if r not in records or off is None or not 0 <= off < MAX_OFFSET:
            return None
        bases.append((r, off))
        a += 8
    return None


def analyze(dol: Dol, functions: set[int]) -> dict[int, ClassInfo]:
    """Return accepted classes keyed by RTTI record address."""
    words = _data_words(dol)

    candidates: dict[int, str] = {}
    for a, w in words.items():
        nxt = words.get(a + 4)
        if nxt is None or (nxt != 0 and nxt not in words):
            continue
        name = _cstring(dol, w)
        if name:
            candidates[a] = name

    # Valid base lists, iterated so every listed base is itself a candidate with a valid list.
    bases: dict[int, list[tuple[int, int]]] = {}
    live = set(candidates)
    while True:
        bases = {}
        for r in live:
            ptr = words[r + 4]
            parsed = [] if ptr == 0 else _read_bases(words, ptr, live)
            if parsed is not None:
                bases[r] = parsed
        if set(bases) == live:
            break
        live = set(bases)

    base_list_words = set()
    for r, lst in bases.items():
        ptr = words[r + 4]
        for i in range(len(lst)):
            base_list_words.add(ptr + 8 * i)

    vtables: dict[int, list[Vtable]] = {}
    for a, w in words.items():
        if w not in live or a in base_list_words:
            continue
        off = words.get(a + 4)
        if off is None:
            continue
        off = _s32(off)
        if not -MAX_OFFSET < off <= 0:
            continue
        slots = []
        p = a + 8
        while words.get(p) in functions:
            slots.append(words[p])
            p += 4
        if slots:
            vtables.setdefault(w, []).append(Vtable(a, off, slots))

    # Accept: records with a vtable, plus everything their base lists reach.
    accepted = set(vtables)
    frontier = list(accepted)
    while frontier:
        r = frontier.pop()
        for b, _ in bases.get(r, []):
            if b not in accepted:
                accepted.add(b)
                frontier.append(b)

    result = {}
    for r in sorted(accepted):
        info = ClassInfo(candidates[r], r, bases.get(r, []), sorted(vtables.get(r, []), key=lambda v: v.address))
        result[r] = info
    return result


def mangle(name: str) -> str | None:
    """CodeWarrior class-name mangling for plain and namespaced names (templates not handled)."""
    if any(c in name for c in "<>,*& "):
        return None
    parts = name.split("::")
    if not all(parts):
        return None
    if len(parts) == 1:
        return f"{len(name)}{name}"
    return f"Q{len(parts)}" + "".join(f"{len(p)}{p}" for p in parts)
