"""Propose a translation unit's full split from its .text range.

  1. local    = data objects whose every referrer is a unit function or another local object
               (fixpoint; walks vtable -> RTTI record -> name string / base list)
  2. owned    = local objects, plus vtables with a slot pointing at a unit function (the TU that
               defines a class's virtuals emits its vtable, even when an inlined ctor elsewhere
               also references it), plus everything those reach through data references
  3. bounds   = per section, the declared units before and after this one in link order (.text
               address): the range never overlaps them nor crosses them (that is a link-order cycle)
  4. ranges   = per section, the contiguous stretch around the owned objects, extended over the
               objects the unit or its data reference (weak vtables, their RTTI copies) up to the
               first object only others reference; then over trailing zero padding up to the next
               symbol, and over unlabeled bytes left between the previous declared unit and us
  5. .ctors/.dtors entries pointing into the unit belong to it
Everything referenced but not in a range is reported as an extern (declare it, don't define it).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from .asmindex import AsmEntry, referrers
from .dtk_symbols import Symbol

DATA_SECTIONS = (".ctors", ".dtors", ".rodata", ".data", ".bss", ".sdata", ".sbss", ".sdata2", ".sbss2")
NOBITS = (".bss", ".sbss", ".sbss2")
_DEBUG: dict | None = None   # tests/diagnostics: the ownership sets of the last propose()
SPLIT_AT = 3       # this many objects in a row that only other units reach end a unit's stretch
MAX_PAD = 0x20      # alignment padding between two units is shorter than the largest alignment (32)


@dataclass
class Proposal:
    text: tuple[int, int]
    functions: list[str]
    sections: dict[str, tuple[int, int]] = field(default_factory=dict)
    members: dict[str, list[str]] = field(default_factory=dict)
    extern_data: list[str] = field(default_factory=list)
    extern_calls: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

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


@dataclass
class _Bounds:
    lo: int
    hi: int
    before: str | None = None     # the declared unit ending at lo
    after: str | None = None      # the declared unit starting at hi


def _bounds(section: str, start: int, units, core: tuple[int, int] | None, limit: tuple[int, int]) -> _Bounds:
    """[lo, hi) this unit's range in `section` may use: after every declared unit that precedes it
    in link order and before every one that follows it. A unit without .text is placed by
    address against `core` (this unit's own objects)."""
    b = _Bounds(*limit)
    for u in units:
        text = next(((a, z) for s, a, z in u.splits if s == ".text"), None)
        for sec, a, z in u.splits:
            if sec != section or z <= a:
                continue
            if text is not None:
                before = text[0] < start
            elif core is not None:
                before = z <= core[0] or (a < core[0] and core[0] - a > core[1] - z)
            else:
                before = False
            if before and z > b.lo:
                b.lo, b.before = z, u.name
            elif not before and a < b.hi:
                b.hi, b.after = a, u.name
    return b


def range_end(end: int, following: int, notes: list[str], section: str) -> int:
    """Where a data range ending with an object at `end` stops, given the next symbol (or
    unit) at `following`. dtk fills the bytes after a unit's last object with a gap_ symbol
    reaching the next symbol, so an unaligned end must reach it (objdiff refuses a gap_ that
    leaves its section); an aligned end stops there, leaving the padding as a hole."""
    if following <= ((end + 3) & ~3) or end % 4 == 0:
        return min(following, (end + 3) & ~3) if following > end else end
    if following - end < MAX_PAD:
        return following
    notes.append(f"{section}: 0x{following - end:X} unlabeled byte(s) after the last object (0x{end:08X}); "
                 "the range stops word-aligned")
    return (end + 3) & ~3


def propose(symbols: list[Symbol], entries: dict[str, AsmEntry], start: int, end: int,
            slots: dict[int, int] | None = None, units=(), read: Callable[[int, int], bytes] | None = None) -> Proposal:
    """`units`: the declared units (splits.Unit; one whose .text overlaps this range is ignored);
    `read(address, size)`: the original's bytes, to tell padding from unlabeled data."""
    by_name = {s.name: s for s in symbols}
    functions = sorted((s for s in symbols if s.type == "function" and s.section == ".text" and start <= s.address < end),
                       key=lambda s: s.address)
    unit = {f.name for f in functions}
    refs = referrers(entries)
    data_syms = {s.name for s in symbols if s.section in DATA_SECTIONS}
    units = [u for u in units if not any(s == ".text" and a < end and start < z for s, a, z in u.splits)]

    def data_refs(name: str) -> list[str]:
        e = entries.get(name)
        return [r for r in e.refs if r in data_syms] if e else []

    def closure(seed: set[str], blocked: set[str] = frozenset()) -> set[str]:
        out, todo = set(seed), [n for n in seed if n not in blocked]
        while todo:
            for r in data_refs(todo.pop()):
                if r not in out and by_name[r].section not in (".ctors", ".dtors"):
                    out.add(r)
                    if r not in blocked:
                        todo.append(r)
        return out

    def slots_into(fns: set[str]) -> set[str]:
        return {n for n, e in entries.items() if e.kind == "obj" and n in data_syms
                and by_name[n].section not in (".ctors", ".dtors") and fns & set(e.calls + e.refs)}

    def exclusive(inside: set[str]) -> set[str]:
        """Data whose every referrer is in `inside` or found so far (fixpoint)."""
        found: set[str] = set()
        candidates = {r for n in inside for r in data_refs(n)}
        changed = True
        while changed:
            changed = False
            for obj in sorted(candidates - found):
                who = refs.get(obj, set()) - {obj}          # a table pointing at itself
                if who and who <= inside | found:
                    found.add(obj)
                    changed = True
                    candidates.update(data_refs(obj))
        return found

    # .ctors/.dtors entries that point at the unit's functions
    ctor_entries = {n for n, e in entries.items() if (sym := by_name.get(n)) and sym.section in (".ctors", ".dtors")
                    and e.refs and set(e.refs) <= unit}
    fn_names = {s.name for s in symbols if s.type == "function"}
    slotted = slots_into(unit)                     # vtables (and tables) pointing at unit functions
    # .bss a static initializer of the unit constructs is the unit's (other units may use it)
    initializers = {f.name for f in functions if f.name.startswith("__sinit") or slots and f.address in slots}
    constructed = {r for f in initializers for r in data_refs(f) if by_name[r].section in NOBITS}
    earlier = {n for n in fn_names if n not in unit and by_name[n].address < start}
    others = fn_names - unit
    direct = {r for f in unit for r in data_refs(f)}
    result = Proposal((start, end), [f.name for f in functions])
    objects_of = {sec: sorted((s for s in symbols if s.section == sec and s.type != "label"
                               and not s.name.startswith("gap_")), key=lambda s: (s.address, -s.size))
                  for sec in DATA_SECTIONS[2:]}
    # Objects the chosen ranges hold make their own referents ours in turn (an unreferenced table
    # in .data pointing at .sdata strings, a weak vtable's RTTI copy): repeat until stable.
    # What other units reach through data: an object only they reach belongs to one of them; a
    # weak copy (RTTI, vtable) an earlier unit also reaches is that unit's (the link keeps the first).
    other_slots = slots_into(others)
    other_seed = {r for f in others for r in data_refs(f)} | other_slots
    earlier_seed = {r for f in earlier for r in data_refs(f)} | {n for n in other_slots - slotted
                                                                if unit.isdisjoint(entries[n].refs)
                                                                and set(entries[n].refs) & earlier}
    members: set[str] = set()
    local: set[str] = set(ctor_entries)
    for _ in range(6):
        local |= exclusive(unit | local | members)
        anchors = local | slotted | constructed
        owned = closure(anchors)
        used = owned | closure(direct) | closure(members, anchors)
        reach_other = closure(other_seed - anchors, anchors) - members
        reach_earlier = closure(earlier_seed - anchors, anchors) - members
        if _DEBUG is not None:
            _DEBUG.update(local=local, anchors=anchors, owned=owned, used=used, reach_other=reach_other,
                          reach_earlier=reach_earlier)
        sets = _Sets(local, anchors, owned, used, reach_other, reach_earlier)
        result.sections, result.members, result.notes = {}, {}, []
        for section, objects in objects_of.items():
            _section_range(section, objects, start, units, read, result, sets)
        grown = {n for ms in result.members.values() for n in ms}
        if grown == members:
            break
        members = grown
    for section in (".ctors", ".dtors"):
        ms = sorted((by_name[n] for n in ctor_entries if by_name[n].section == section), key=lambda s: s.address)
        if ms:
            result.sections[section] = (ms[0].address, (max(s.address + s.size for s in ms) + 3) & ~3)
            result.members[section] = [s.name for s in ms]
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


@dataclass
class _Sets:
    local: set[str]          # referenced only by the unit (and its local data)
    anchors: set[str]        # local, vtables pointing at unit functions, .bss its initializers construct
    owned: set[str]          # anchors and what they reach through data
    used: set[str]           # owned, and what the unit's code and chosen objects reach
    reach_other: set[str]    # what other units' code and vtables reach
    reach_earlier: set[str]  # the same, for units before this one


def _section_range(section: str, objects: list[Symbol], start: int, units, read, result: Proposal,
                   sets: _Sets) -> None:
    mine = [s for s in objects if s.name in sets.owned]
    if not objects or not mine:
        return
    limit = (objects[0].address, (max(s.address + s.size for s in objects) + 3) & ~3)   # sections end aligned
    b = _bounds(section, start, units, (mine[0].address, mine[-1].address + mine[-1].size), limit)
    inside = [s for s in objects if b.lo <= s.address and s.address + max(s.size, 1) <= b.hi]
    dropped = [s.name for s in mine if s not in inside]
    foreign = lambda s: s.name in sets.reach_other and s.name not in sets.used
    # Split at stretches of objects only other units reach (another unit's data between ours, or
    # a stray far away); keep the stretch holding the most own objects. One or two such objects
    # are globals this unit defines for others.
    runs: list[list[Symbol]] = [[]]
    pending: list[Symbol] = []
    for s in inside:
        if foreign(s):
            pending.append(s)
            continue
        if len(pending) >= SPLIT_AT:
            runs.append([])
        elif pending:
            runs[-1] += pending
        pending = []
        runs[-1].append(s)
    if pending and len(pending) < SPLIT_AT:
        runs[-1] += pending
    best = max(runs, key=lambda run: (sum(s.name in sets.anchors for s in run), sum(s.name in sets.owned for s in run)))
    dropped += [s.name for run in runs if run is not best for s in run if s.name in sets.owned]
    # A leading weak copy an earlier unit also reaches is that unit's (the link keeps the first).
    firm = [k for k, s in enumerate(best) if s.name in sets.anchors
            or s.name in sets.owned and s.name not in sets.reach_earlier]
    if not firm:
        if dropped:
            result.notes.append(f"{section}: own objects outside {b.lo:08X}-{b.hi:08X}: {', '.join(dropped[:6])}")
        return
    # First..last own object, widened backwards over what the unit uses and no earlier unit
    # reaches, forwards over what it uses (weak vtables, RTTI copies), with the unreferenced
    # objects among those; trailing unreferenced ones too (strings and .bss reached at an offset
    # from a referenced base: the next unit's first object is its base, so it is referenced).
    i = firm[0]
    j = max(k for k, s in enumerate(best) if s.name in sets.owned)
    orphan = lambda s: s.name not in sets.used and s.name not in sets.reach_other
    k = i
    while k > 0 and (best[k - 1].name in sets.used or orphan(best[k - 1])) and best[k - 1].name not in sets.reach_earlier:
        k -= 1
        if best[k].name in sets.used:
            i = k
    # Leading unreferenced objects are the previous unit's tail (its strings reached at an offset).
    tail = [s.name for s in best[k:i] if orphan(s)]
    if tail and b.before and best[k].address >= b.lo:
        result.notes.append(f"{section}: {len(tail)} unreferenced object(s) before this unit's first ({tail[0]}…) "
                            f"are probably the tail of {b.before}: widen it with ./dk unit set")
    while j + 1 < len(best) and (best[j + 1].name in sets.used or orphan(best[j + 1])):
        j += 1
    members = best[i:j + 1]
    # a unit's section starts word-aligned: the bytes (and objects) just before an odd start are ours
    if members[0].address % 4 and members[0].address & ~3 >= b.lo:
        members = [s for s in inside if members[0].address & ~3 <= s.address < members[0].address] + members
    lo = members[0].address & ~3 if members[0].address & ~3 >= b.lo else members[0].address
    hi = max(s.address + s.size for s in members)
    hi = range_end(hi, min([s.address for s in objects if s.address >= hi] + [b.hi]), result.notes, section)
    # unlabeled bytes between the previous declared unit and our first object are ours
    if b.before and b.lo < lo and not any(b.lo <= s.address < lo for s in objects) \
            and section not in NOBITS and read is not None and read(b.lo, lo - b.lo).strip(b"\0"):
        lo = b.lo
    result.sections[section] = (lo, hi)
    result.members[section] = [s.name for s in members]
    if dropped:
        result.notes.append(f"{section}: own objects left out (outside {b.lo:08X}-{b.hi:08X} or not contiguous): "
                            f"{', '.join(dropped[:6])}{' …' if len(dropped) > 6 else ''}")


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


def is_initializer(name: str, entries: dict[str, AsmEntry]) -> bool:
    e = entries.get(name)
    return name.startswith("__sinit") or bool(e and e.file.stem.startswith("auto_fn_"))


def lone_initializer(functions: list[Symbol], previous: list[Symbol], entries: dict[str, AsmEntry]) -> bool:
    """`functions` are only static initializers and the TU before them has none of its own.

    An object whose only code is a static initializer has nothing referenced from outside, so
    the linker drops it whole (.ctors entry included). The original kept this one: it is the
    tail of the TU before it."""
    return bool(functions and previous) and all(is_initializer(f.name, entries) for f in functions) \
        and not any(is_initializer(f.name, entries) for f in previous)


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
