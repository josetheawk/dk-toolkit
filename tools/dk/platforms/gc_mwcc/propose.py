"""Propose a translation unit's full split from its .text range.

  1. seeds   = data symbols referenced by the unit's functions
  2. local   = objects whose every referrer is a unit function or another local object
               (fixpoint; walks vtable -> RTTI record -> name string / base list)
  3. ranges  = per section, [first local, last local end], extended forward over the
               contiguous objects the unit references even if others do too (weak vtables)
  4. .ctors/.dtors entries pointing into the unit belong to it
Everything referenced but not local is reported as an extern (declare it, don't define it).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .asmindex import AsmEntry, referrers
from .dtk_symbols import Symbol

DATA_SECTIONS = (".ctors", ".dtors", ".rodata", ".data", ".bss", ".sdata", ".sbss", ".sdata2", ".sbss2")


@dataclass
class Proposal:
    text: tuple[int, int]
    functions: list[str]
    sections: dict[str, tuple[int, int]] = field(default_factory=dict)
    members: dict[str, list[str]] = field(default_factory=dict)
    extern_data: list[str] = field(default_factory=list)
    extern_calls: list[str] = field(default_factory=list)

    def splits(self) -> list[tuple[str, int, int]]:
        out = [(".text", *self.text)]
        out += [(sec, *rng) for sec, rng in sorted(self.sections.items(), key=lambda kv: DATA_SECTIONS.index(kv[0]))]
        return out


def ctor_slots(dol, symbols: list[Symbol]) -> dict[int, int]:
    """Static-initializer address -> its .ctors slot address (dtk names few of the slots)."""
    starts = [s.address for s in symbols if s.section == ".ctors"]
    section = dol.section_at(min(starts)) if starts else None
    if section is None:
        return {}
    out = {}
    for address in range(min(starts), section.end, 4):
        value = int.from_bytes(dol.read(address, 4), "big")
        if value:
            out.setdefault(value, address)
    return out


def propose(symbols: list[Symbol], entries: dict[str, AsmEntry], start: int, end: int,
            slots: dict[int, int] | None = None) -> Proposal:
    by_name = {s.name: s for s in symbols}
    functions = sorted((s for s in symbols if s.type == "function" and s.section == ".text" and start <= s.address < end),
                       key=lambda s: s.address)
    unit = {f.name for f in functions}
    refs = referrers(entries)
    data_syms = {s.name for s in symbols if s.section in DATA_SECTIONS}

    local: set[str] = set()
    # .ctors/.dtors entries that point at the unit's functions
    for name, e in entries.items():
        sym = by_name.get(name)
        if sym and sym.section in (".ctors", ".dtors") and e.refs and set(e.refs) <= unit:
            local.add(name)
    seeds = {r for f in unit if f in entries for r in entries[f].refs if r in data_syms}

    # Alternate locality and range extension until stable: objects pulled in by extension
    # (a weak base vtable) can make their own referents (its RTTI copy) local in turn.
    result = Proposal((start, end), [f.name for f in functions])
    included: set[str] = set()
    for _ in range(10):
        inside = unit | local | included
        candidates = set(seeds) | {r for n in inside if n in entries for r in entries[n].refs if r in data_syms}
        changed = True
        while changed:
            changed = False
            for obj in sorted(candidates - local):
                who = refs.get(obj, set())
                if who and who <= unit | local | included:
                    local.add(obj)
                    changed = True
                    if obj in entries:
                        candidates.update(r for r in entries[obj].refs if r in data_syms)
        referenced_by_unit = {r for n in unit | local | included if n in entries for r in entries[n].refs}
        result.sections, result.members = {}, {}
        for section in DATA_SECTIONS:
            members = sorted((by_name[n] for n in local if by_name[n].section == section), key=lambda s: s.address)
            if not members:
                continue
            lo = members[0].address
            hi = max(s.address + s.size for s in members)
            in_section = sorted((s for s in symbols if s.section == section and s.type != "label"), key=lambda s: s.address)
            for s in in_section:     # extend over contiguous objects the unit references (e.g. weak vtables)
                if s.address < hi:
                    continue
                if s.address == hi and s.name in referenced_by_unit and not s.name.startswith("gap_"):
                    hi = s.address + s.size
                else:
                    break
            names = {s.name for s in in_section if lo <= s.address < hi and not s.name.startswith("gap_")}
            result.sections[section] = (lo, (hi + 3) & ~3)
            result.members[section] = sorted(names, key=lambda n: by_name[n].address)
        new_included = {n for ms in result.members.values() for n in ms}
        if new_included == included:
            break
        included = new_included
    own = sorted(slots[f.address] for f in functions if slots and f.address in slots)
    if own:
        lo, hi = result.sections.get(".ctors", (own[0], own[-1] + 4))
        result.sections[".ctors"] = (min(lo, own[0]), max(hi, own[-1] + 4))
        result.members.setdefault(".ctors", [])
    included = {n for ms in result.members.values() for n in ms}
    all_refs = [r for f in result.functions if f in entries for r in entries[f].refs]
    result.extern_data = sorted({r for r in all_refs if r in data_syms and r not in included})
    calls = [c for f in result.functions if f in entries for c in entries[f].calls]
    result.extern_calls = sorted({c for c in calls if c not in unit})
    return result


def _affinity(name: str, group: set[str], entries: dict[str, AsmEntry], data_of: dict[str, set[str]]) -> int:
    """How strongly `name` belongs with `group` (a set of functions).

    +1 per call/data reference from `name` into the group or the group's data, and +2 when the
    group's data (e.g. its vtables) points at `name` (an inline virtual that references nothing).
    """
    e = entries.get(name)
    if e is None:
        return 0
    group_data = set().union(*(data_of.get(g, set()) for g in group)) if group else set()
    score = sum(1 for r in e.calls + e.refs if r in group or r in group_data)
    if any(name in entries[d].refs for d in group_data if d in entries and entries[d].kind == "obj"):
        score += 2
    return score


def self_contained(name: str, entries: dict[str, AsmEntry]) -> bool:
    """A function that calls nothing and references only data no other function touches."""
    e = entries.get(name)
    if e is None or e.calls:
        return False
    users = referrers(entries)
    return all(users.get(r, set()) <= {name} for r in e.refs)


MARGIN = 2   # evidence a function needs to change TU: a vtable back-reference, or 2 more references
_INSN = re.compile(r"^/\* [0-9A-F]{8} ", re.M)


def _thunk_into(name: str, group: set[str], entries: dict[str, AsmEntry]) -> bool:
    """A this-adjusting thunk (`addi r3, r3, X` + `b target`) whose target is in `group`."""
    e = entries.get(name)
    if e is None or name.startswith("@"):
        return bool(e)
    return len(_INSN.findall(e.text)) == 2 and "addi r3, r3," in e.text and bool(e.calls) and e.calls[0] in group


def adjust_tu_bounds(functions: list[Symbol], previous: list[Symbol], following: list[Symbol],
                     entries: dict[str, AsmEntry]) -> tuple[list[Symbol], list[str], list[str]]:
    """Fix dtk's static-initializer auto units, whose start holds the previous TU's tail.

    A TU's factory functions and virtual-base thunks come after its __sinit, so dtk (splitting at
    __sinit) puts them at the start of the *next* auto unit. Leading functions that are thunks
    ('@' names) or whose references point more into the previous unit than into this one move
    back; the next unit's leading functions that point more into this unit move in.
    Returns (functions, moved out, moved in).
    """
    data_of = {f: set(entries[f].refs) for f in entries if entries[f].kind == "fn"}

    def is_sinit(sym: Symbol) -> bool:
        e = entries.get(sym.name)
        return sym.name.startswith("__sinit") or bool(e and e.file.stem.startswith("auto_fn_"))

    current = list(functions)
    moved_out: list[str] = []
    prev_names = {s.name for s in previous}
    # A TU's tail exists only after its own __sinit: move functions out only when the unit starts
    # right after the previous TU's initializer, and in only after this unit's initializer.
    if not (previous and is_sinit(previous[-1])):
        previous = []
    if following and not is_sinit(following[0]) and not (current and is_sinit(current[-1])):
        following = []
    while len(current) > 1:
        head = current[0]
        rest = {s.name for s in current[1:]}
        to_prev = _affinity(head.name, prev_names, entries, data_of)
        to_here = _affinity(head.name, rest, entries, data_of)
        if previous and (_thunk_into(head.name, prev_names, entries) or to_prev - to_here >= MARGIN):
            moved_out.append(head.name)
            prev_names.add(head.name)
            current.pop(0)
        else:
            break
    moved_in: list[str] = []
    here = {s.name for s in current}
    # dtk splits each __sinit into its own object right after its TU. It belongs here when it
    # touches anything of this unit (one reference is enough: initializers are short), or when
    # it touches only data of its own: a header's static initialised in every TU including it.
    if following and is_sinit(following[0]) and not is_sinit(current[-1]):
        sinit = following[0]
        later = {s.name for s in following[1:]}
        if (_affinity(sinit.name, here, entries, data_of)
                - _affinity(sinit.name, later, entries, data_of)) >= 1 or self_contained(sinit.name, entries):
            following.pop(0)
            moved_in.append(sinit.name)
            here.add(sinit.name)
            current.append(sinit)
        else:
            following = []
    for nxt in following:
        later = {s.name for s in following if s.address > nxt.address}
        to_here = _affinity(nxt.name, here, entries, data_of)
        to_next = _affinity(nxt.name, later, entries, data_of)
        if _thunk_into(nxt.name, here, entries) or to_here - to_next >= MARGIN:
            moved_in.append(nxt.name)
            here.add(nxt.name)
            current.append(nxt)
        else:
            break
    return current, moved_out, moved_in
