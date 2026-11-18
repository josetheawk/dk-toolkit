"""dk unit boundaries: signs that one declared unit is several translation units.

Each object file's data sections start on 8 bytes, and CodeWarrior pools a TU's literals once
and ends its `.data` with its classes' RTTI names and vtables (playbook, matching.md). So, for a
unit's text range and the data its functions reach:

- literal: the same float, double or string twice in one pool (each TU pools its own);
- align:   a declared data range that starts off an 8-byte boundary (the TU starts elsewhere);
- gap:     zero pad bytes before an 8-aligned object that needs no 8 alignment (a TU's data
           section starting on 8, not a double);
- vtable:  data after a vtable that is not the next class's RTTI name (the next TU's data).

Every sign gives a data address where the next TU would start, and a cut: the first function
(by address) that uses data from there on, "clean" when every function using the data before it
comes earlier. Cuts that several signs agree on come first, each with the `unit propose` line
that shows the TU it starts.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from .dtk_symbols import Symbol, Unit

DATA = (".rodata", ".data", ".sdata", ".sdata2", ".sbss", ".bss")
POOLS = (".sdata2", ".rodata", ".data")
NO_BYTES = (".sbss", ".bss")
CLASS_NAME = re.compile(rb"^[A-Za-z_][\w:<>, *&]*\x00+$")
TYPE_SIZES = {"float": 4, "4byte": 4, "2byte": 2, "byte": 1}


@dataclass
class Evidence:
    section: str
    address: int                   # where the next TU's data would start
    signal: str                    # literal, align, gap, vtable
    detail: str
    cut: Symbol | None = None      # the first function using data from `address` on
    clean: bool = False            # every user of the data before `address` comes before `cut`
    earliest: Symbol | None = None # with clean: the function after the last user of the data before (the TU may start here)


@dataclass
class Cut:
    function: Symbol
    evidence: list[Evidence] = field(default_factory=list)

    @property
    def signals(self) -> list[str]:
        return sorted({e.signal for e in self.evidence})

    @property
    def clean(self) -> bool:
        return any(e.clean for e in self.evidence)

    start: Symbol | None = None    # the call graph's pick between `earliest` and `function`

    @property
    def earliest(self) -> Symbol:
        """Where the TU may start at the earliest: after the last function using the data before."""
        starts = [e.earliest for e in self.evidence if e.clean and e.earliest is not None]
        return max(starts, key=lambda f: f.address) if starts else self.function

    @property
    def first(self) -> Symbol:
        """The function the next TU most likely starts with."""
        return self.start or self.function


@dataclass
class Report:
    unit: Unit
    functions: list[Symbol]
    evidence: list[Evidence] = field(default_factory=list)
    cuts: list[Cut] = field(default_factory=list)
    vetoed: list[tuple[Cut, str]] = field(default_factory=list)   # a cut and the pooled literal both sides use

    @property
    def score(self) -> int:
        return sum(len(c.signals) * (2 if c.clean else 1) for c in self.cuts) + sum(1 for e in self.evidence if e.cut is None)


def _text(unit: Unit) -> list[tuple[int, int]]:
    return [(a, z) for sec, a, z in unit.splits if sec == ".text"]


def _bytes(read, s: Symbol) -> bytes | None:
    if s.section in NO_BYTES or s.size <= 0 or read is None:
        return None
    try:
        return read(s.address, s.size)
    except ValueError:
        return None


def _value(section: str, data: bytes) -> str:
    import struct
    if len(data) == 4 and section in (".sdata2", ".rodata"):
        return f"{struct.unpack('>f', data)[0]:g}f"
    if len(data) == 8 and section in (".sdata2", ".rodata"):
        return f"{struct.unpack('>d', data)[0]:g}"
    text = data.rstrip(b"\0")
    if text and all(32 <= c < 127 for c in text):
        return '"' + text.decode()[:24] + '"'
    return data.hex()[:16]


def _number(s: Symbol, data: bytes) -> bool:
    """A float or double literal the compiler pools once per TU: finite and normal (or zero).
    Bit patterns such as -1 masks are often `const` objects, kept once per definition."""
    import math
    import struct
    kind = s.attrs.get("data", "")
    if kind == "float" and len(data) == 8 and not any(data[4:]):
        data = data[:4]                                # a float with dtk's pad sized in
    if len(data) == 4 and kind == "float":
        v = struct.unpack(">f", data)[0]
    elif len(data) == 8 and kind == "double":
        v = struct.unpack(">d", data)[0]
    else:
        return False
    return math.isfinite(v) and (v == 0 or abs(v) >= (1.2e-38 if len(data) == 4 else 2.3e-308))


def _key(section: str, s: Symbol, data: bytes) -> bytes:
    """The literal's value: a string without its NUL padding, a float without the pad dtk sized in."""
    if section == ".sdata2" or s.attrs.get("data") in ("float", "double"):
        return data[:4] if s.attrs.get("data") == "float" and len(data) == 8 and not any(data[4:]) else data
    return data.rstrip(b"\0")


def _literal_like(section: str, s: Symbol, data: bytes) -> bool:
    if section == ".sdata2":
        return _number(s, data)
    text = data.rstrip(b"\0")
    is_string = bool(text) and len(text) < len(data) and all(32 <= c < 127 or c in (9, 10) for c in text)
    if section == ".rodata":
        return _number(s, data) or is_string
    return is_string and not s.name.startswith("__vt__")


def analyse(unit: Unit, symbols: list[Symbol], refs: dict[str, list[str]], read, owners: dict[str, list] | None = None,
            data_refs: dict[str, list[str]] | None = None, calls: dict[str, list[str]] | None = None,
            asm: dict[str, str] | None = None) -> Report:
    """The evidence for one unit. `refs`: function name -> data symbols it references; `read(address,
    size)` the original bytes; `owners`: section -> [(start, end, unit name)] of declared units, so a
    data stretch another unit declares is not counted as this one's; `data_refs`: data object ->
    the symbols its words point at (a vtable's RTTI record, a record's name and bases)."""
    text = _text(unit)
    fns = sorted((s for s in symbols if s.section == ".text" and s.type == "function"
                  and any(a <= s.address < z for a, z in text)), key=lambda s: s.address)
    rep = Report(unit, fns)
    if not fns:
        return rep
    by_name = {s.name: s for s in symbols}
    users: dict[str, list[Symbol]] = defaultdict(list)
    for f in fns:
        for r in refs.get(f.name, []):
            if r in by_name:
                users[r].append(f)
    declared = {sec: (a, z) for sec, a, z in unit.splits if sec in DATA}
    mine = {f.name for f in fns}
    elsewhere = {r for f, rs in refs.items() if f not in mine for r in rs}    # data other units' code reads
    first_sinit = next((k for k, f in enumerate(fns) if f.name.startswith("__sinit_")), None)
    tail = {f.name for f in fns[first_sinit:]} if first_sinit is not None else set()   # the __sinit, then factories
    pooled: dict[str, list[Symbol]] = {}
    for sec in DATA:
        objs = sorted((s for s in symbols if s.section == sec), key=lambda s: s.address)
        if sec in declared:
            lo, hi = declared[sec]
            if lo % 8:
                rep.evidence.append(Evidence(sec, lo, "align", f"declared {sec} starts at 0x{lo:08X}, off an 8-byte "
                                             "boundary: the TU's data starts earlier, or the unit before ends later"))
        else:
            used = [s for s in objs if users.get(s.name)]
            if not used:
                continue
            lo, hi = used[0].address, used[-1].address + max(used[-1].size, 1)
            for a, z, name in (owners or {}).get(sec, []):
                if name != unit.name and a < hi and lo < z:      # another unit declares part of it
                    if a <= lo:
                        lo = max(lo, z)
                    else:
                        hi = min(hi, a)
            if lo >= hi:
                continue
        inside = [s for s in objs if lo <= s.address < hi and s.size > 0]
        copies = {x.name for x in inside if _copied_whole(x, users.get(x.name, []), asm or {})}
        copies |= {x.name for x in inside if x.name in elsewhere}    # a named const other units use: not a pooled literal
        for x in inside:
            data = _bytes(read, x)
            if data is not None and x.name not in copies and (sec in POOLS + (".sdata",)) and _literal_like(sec, x, data):
                pooled[x.name] = users.get(x.name, [])
        if len(inside) < 2:
            continue
        virtuals = {r for s in inside if s.name.startswith("__vt__") for r in (data_refs or {}).get(s.name, [])}
        virtuals |= tail
        _signals(rep, sec, inside, users, read, fns, _rtti(inside, data_refs or {}), virtuals, copies, asm)
    _cuts(rep)
    for c in rep.cuts:
        c.start = _start(rep, c, refs, data_refs or {}, calls or {})
    _veto(rep, pooled)
    return rep


def _veto(rep: Report, pooled: dict[str, list[Symbol]]) -> None:
    """A pooled literal used on both sides of a cut puts both sides in one TU (literals are pooled
    only within a TU): such a cut is dropped, and kept in `rep.vetoed` with the literal."""
    keep = []
    for c in rep.cuts:
        at = c.first.address
        both = next((n for n, us in pooled.items() if any(f.address < at for f in us) and any(f.address >= at for f in us)),
                    None)
        if both is None:
            keep.append(c)
        else:
            rep.vetoed.append((c, both))
    rep.cuts = keep


def _copied_whole(obj: Symbol, users: list[Symbol], asm: dict[str, str]) -> bool:
    """Does every user copy the whole object (a local array's initializer: `char s[0x20] = "x";`
    loads every word of it, or passes it to a copy)? Initializers are not pooled: each function
    has its own, so a repeat is no TU sign."""
    if obj.size < 8 or not users or not asm:
        return False
    for f in users:
        text = asm.get(f.name)
        if text is None:
            return False
        base = re.search(rf"addi (r\d+), r\d+, {re.escape(obj.name)}@l", text)
        if base is None:
            return False
        rest = text[base.end():]
        reg = base.group(1)
        offsets = {int(m.group(1), 0) for m in re.finditer(rf"\b(?:lwz|lfd) [rf]\d+, (0x[0-9a-fA-F]+|\d+)\({reg}\)", rest[:4000])}
        words = set(range(0, obj.size - obj.size % 4, 4))
        covered = {w for o in offsets for w in (o, o + 4)}
        lmw = re.search(rf"\blmw r\d+, 0x0\({reg}\)|\blmw r\d+, 0\({reg}\)", rest[:400])
        if not words <= covered and not lmw and not re.search(r"\bbl (?:memcpy|__copy)", rest[:600]):
            return False
    return True


def _edges(rep: Report, refs, data_refs, calls) -> dict[str, set[str]]:
    """The unit's functions that belong together: a call either way, or a virtual and the
    functions that use its vtable (the class's constructors and destructors)."""
    names = {f.name for f in rep.functions}
    out: dict[str, set[str]] = {n: set() for n in names}
    for f in names:
        for c in calls.get(f, []):
            if c in names and c != f:
                out[f].add(c)
                out[c].add(f)
    vt_users: dict[str, set[str]] = {}
    for f in names:
        for r in refs.get(f, []):
            if r.startswith("__vt__"):
                vt_users.setdefault(r, set()).add(f)
    for vt, users in vt_users.items():
        for v in data_refs.get(vt, []):
            if v in names:
                for u in users - {v}:
                    out[v].add(u)
                    out[u].add(v)
    return out


def _start(rep: Report, cut: Cut, refs, data_refs, calls) -> Symbol | None:
    """Where between `earliest` and the cut the next TU starts, by the call graph: each function in
    between leans to the TU before (it calls or is called from functions before the range, or from
    ones already leaning there) or the TU after (the same with functions from the cut on). The start
    moves before the cut only on evidence: the first function leaning ahead after the last one
    leaning back. None: the cut stands."""
    fns = rep.functions
    lo = next(i for i, f in enumerate(fns) if f is cut.earliest)
    hi = next(i for i, f in enumerate(fns) if f is cut.function)
    if hi <= lo:
        return None
    edges = _edges(rep, refs, data_refs, calls)
    index = {f.name: i for i, f in enumerate(fns)}
    sides = {n: {(-1 if index[m] < lo else 1 if index[m] >= hi else 0) for m in ms} for n, ms in edges.items()}
    shared = {n for n, sd in sides.items() if {-1, 1} <= sd}     # used from both sides (a base destructor): no evidence
    lean = [0] * len(fns)
    for i in range(lo, hi):
        for n in edges[fns[i].name] - shared:
            j = index[n]
            lean[i] += 1 if j < lo else -1 if j >= hi else 0
    for i in range(lo, hi):                 # one round through the range: a helper of a leaning function
        for n in edges[fns[i].name]:
            j = index[n]
            if lo <= j < hi and j != i and lean[j]:
                lean[i] += 1 if lean[j] > 0 else -1
    # only on evidence: the first function leaning to the next TU after the last one leaning back
    last_back = max((i for i in range(lo, hi) if lean[i] > 0), default=lo - 1)
    ahead = next((i for i in range(last_back + 1, hi) if lean[i] < 0), None)
    return fns[ahead] if ahead is not None else None


def _cut(rep: Report, ev: Evidence, before: list[Symbol], after: list[Symbol], users) -> None:
    after_users = sorted({f.address: f for s in after for f in users.get(s.name, [])}.values(), key=lambda f: f.address)
    before_users = [f for s in before for f in users.get(s.name, [])]
    _place(rep, ev, before_users, after_users)


def _place(rep: Report, ev: Evidence, before_users: list[Symbol], after_users: list[Symbol]) -> None:
    after_users = sorted(after_users, key=lambda f: f.address)
    if not after_users or after_users[0] is rep.functions[0]:
        return
    ev.cut = after_users[0]
    last = max((f.address for f in before_users), default=None)
    ev.clean = last is None or last < ev.cut.address
    if ev.clean:
        k = next((i for i, f in enumerate(rep.functions) if last is not None and f.address > last), 0)
        ev.earliest = rep.functions[k] if last is not None else rep.functions[1]


def _rtti(objs: list[Symbol], data_refs: dict[str, list[str]]) -> set[str]:
    """Objects a vtable reaches through its words (its RTTI record, the record's name and base list):
    CodeWarrior puts them with the class's vtable, so they are no sign of a new TU."""
    out: set[str] = set()
    todo = [s.name for s in objs if s.name.startswith("__vt__")]
    while todo:
        for r in data_refs.get(todo.pop(), []):
            if r not in out and not r.startswith("__vt__"):
                out.add(r)
                todo.append(r)
    return out


def _is_string(data: bytes | None) -> bool:
    text = (data or b"").rstrip(b"\0")
    return bool(text) and len(text) < len(data) and all(32 <= c < 127 or c in (9, 10) for c in text)


def _addressed(obj: Symbol, users: list[Symbol], asm: dict[str, str]) -> bool:
    """Does code take the object's address into a register (`li rN, x@sda21`, `addi rN, rM, x@l`) or
    reach it with an addend (`x+0x4@sda21`)? Then any offset may be read through it, so the bytes
    after it may be its own (an array dtk sized as one element), not a pad."""
    name = re.escape(obj.name)
    pat = re.compile(rf"\b(?:li r\d+|addi r\d+, r\d+), {name}@(?:sda21|l)\b|\b{name}\+(?:0x[0-9a-fA-F]+|\d+)@")
    return any(pat.search(asm.get(f.name, "")) for f in users)


def _signals(rep: Report, sec: str, objs: list[Symbol], users, read, fns, rtti: set[str] = frozenset(),
             virtuals: set[str] = frozenset(), copies: set[str] = frozenset(), asm: dict | None = None) -> None:
    datas = {s.name: _bytes(read, s) for s in objs}
    asm = asm or {}
    # literal: the same value twice in one pool
    if sec in POOLS:
        first: dict[bytes, Symbol] = {}
        for k, s in enumerate(objs):
            data = datas[s.name]
            if data is None or s.name in copies or not _literal_like(sec, s, data):
                continue
            key = _key(sec, s, data)
            if key in first:                              # against the latest copy: the TU just before
                o1 = first[key]
                ev = Evidence(sec, s.address, "literal", f"{_value(sec, key)} at 0x{o1.address:08X} and 0x{s.address:08X}")
                k1 = objs.index(o1)
                second = sorted(users.get(s.name, []), key=lambda f: f.address)
                if second:                                # the next pool starts at its first object used from there on
                    start = next(i for i in range(k1 + 1, k + 1)
                                 if i == k or any(f.address >= second[0].address for f in users.get(objs[i].name, [])))
                    ev.address = objs[start].address
                    _cut(rep, ev, objs[:start], objs[start:], users)
                rep.evidence.append(ev)
            first[key] = s
    # pad: an object dtk sized past its type with zero bytes up to an 8-aligned next object
    for k in range(len(objs) - 1):
        a, b = objs[k], objs[k + 1]
        if sec in NO_BYTES:
            break                                      # .bss sizes are dtk's guesses: no pads to read
        unit_size = TYPE_SIZES.get(a.attrs.get("data", ""))
        data = datas.get(a.name)
        if unit_size is None or a.size <= unit_size or a.size % unit_size or b.address % 8 or b.address != a.address + a.size:
            continue
        if _addressed(a, users.get(a.name, []), asm):
            continue                                   # the "pad" may be the rest of an array
        tail = a.size - unit_size
        if tail >= 8 or (data is not None and any(data[unit_size:])) or (data is None and sec not in NO_BYTES):
            continue
        if sec in (".sdata2", ".sdata") and b.size % 8 == 0:
            continue
        ev = Evidence(sec, b.address, "gap", f"{a.name} is a {a.attrs['data']} with {tail} zero bytes after it, "
                      f"before 8-aligned {b.name}")
        _cut(rep, ev, objs[:k + 1], objs[k + 1:], users)
        rep.evidence.append(ev)
    # gap: zero bytes before an 8-aligned object that does not need 8 (the next TU's section start)
    for k in range(1, len(objs)):
        a, b = objs[k - 1], objs[k]
        if a.name.startswith("gap_") or (not _is_string(datas.get(a.name)) and _addressed(a, users.get(a.name, []), asm)):
            continue                                   # a string's address is always taken: its pad stays a pad
        pad = b.address - (a.address + a.size)
        if pad < 4 or pad >= 8 or b.address % 8 or b.name.startswith("gap_") or sec in NO_BYTES:
            continue
        if sec in (".sdata2", ".sdata") and b.size % 8 == 0:
            continue                                   # a double (8-aligned in any TU)
        if sec not in NO_BYTES and read is not None:
            try:
                if any(read(a.address + a.size, pad)):
                    continue
            except ValueError:
                continue
        ev = Evidence(sec, b.address, "gap", f"{pad} zero bytes after {a.name} before 8-aligned {b.name} ({b.size} bytes)")
        _cut(rep, ev, objs[:k], objs[k:], users)
        rep.evidence.append(ev)
    # vtable: after a vtable comes the next class's RTTI name, or the TU's end
    if sec == ".data":
        for k, s in enumerate(objs):
            if not s.name.startswith("__vt__") or k + 1 >= len(objs):
                continue
            j = k + 1
            while j < len(objs) and objs[j].name in rtti:
                j += 1                                 # RTTI records and names go with the vtables
            if j >= len(objs):
                continue
            nxt = objs[j]
            if nxt.name.startswith("__vt__"):
                continue
            data = datas.get(nxt.name)
            later_vtable = any(x.name.startswith("__vt__") for x in objs[k + 1:])
            if data is not None and CLASS_NAME.match(data) and later_vtable:
                continue                               # the next class's RTTI name
            after_users = {f.name for x in objs[j:] for f in users.get(x.name, [])}
            if not after_users or after_users <= virtuals:
                continue      # only compiler data, or used only by inline virtuals, the __sinit and the factories
                              # after it (all emitted at the TU's end)
            ev = Evidence(sec, nxt.address, "vtable", f"{nxt.name} ({nxt.size} bytes"
                          + (f", {_value(sec, data)}" if data is not None else "") + f") follows {s.name}"
                          + ("" if later_vtable else ", the unit's last vtable"))
            _cut(rep, ev, objs[:j], objs[j:], users)
            rep.evidence.append(ev)


def _explained(rep: Report) -> None:
    """A repeated literal whose two copies straddle a boundary another sign (or an earlier repeat)
    already places joins that boundary's cut: both say the same TU starts there."""
    for sec in {e.section for e in rep.evidence}:
        evs = [e for e in rep.evidence if e.section == sec and e.cut is not None]
        placed = sorted((e for e in evs if e.signal != "literal"), key=lambda e: e.address)
        for e in sorted((e for e in evs if e.signal == "literal"), key=lambda e: e.address):
            first = int(e.detail.split(" at 0x", 1)[1][:8], 16)
            hit = next((b for b in placed if first < b.address <= e.address and b.cut is not e.cut), None)
            if hit is not None and hit.cut.address < e.cut.address:
                e.cut, e.clean, e.earliest, e.address = hit.cut, hit.clean, hit.earliest, hit.address
            placed.append(e)
            placed.sort(key=lambda x: x.address)


def _cuts(rep: Report) -> None:
    _explained(rep)
    by: dict[int, Cut] = {}
    for e in rep.evidence:
        if e.cut is not None:
            by.setdefault(e.cut.address, Cut(e.cut)).evidence.append(e)
    rep.cuts = sorted(by.values(), key=lambda c: (-len(c.signals), not c.clean, c.function.address))


def propose_line(rep: Report, cut: Cut) -> str:
    """The `unit propose` range of the TU a cut starts (from the cut's first function): up to the
    next cut or the unit's end."""
    at = cut.first.address
    later = sorted(c.first.address for c in rep.cuts if c.first.address > at)
    end = next((z for a, z in _text(rep.unit) if a <= at < z), rep.functions[-1].address)
    stop = later[0] if later else end
    return f"./dk unit propose 0x{at:08X} 0x{stop:08X}"


def render(rep: Report, limit: int = 6) -> list[str]:
    out = []
    text = ", ".join(f"0x{a:08X}-0x{z:08X}" for a, z in _text(rep.unit))
    out.append(f"{rep.unit.name}  (.text {text}, {len(rep.functions)} functions)")
    for c in rep.cuts:
        span = "" if c.earliest is c.function else (f" (the data puts it between 0x{c.earliest.address:08X} and "
                                                     f"0x{c.function.address:08X}"
                                                     + ("; the call graph picks this one)" if c.start else ")"))
        out.append(f"  suspected TU boundary before {c.first.name} (0x{c.first.address:08X}){span}: "
                   f"{', '.join(c.signals)}{' (clean cut)' if c.clean else ' (users interleave)'}")
        for e in c.evidence[:limit]:
            out.append(f"    {e.signal:<7} {e.section:<8} 0x{e.address:08X}  {e.detail}")
        if len(c.evidence) > limit:
            out.append(f"    ... {len(c.evidence) - limit} more")
        out.append(f"    {propose_line(rep, c)}")
    for c, name in rep.vetoed:
        out.append(f"  (no boundary before {c.first.name}: {', '.join(c.signals)} sign, but the pooled literal {name} "
                   "is used on both sides)")
    loose = [e for e in rep.evidence if e.cut is None]
    for e in loose[:limit]:
        out.append(f"  {e.signal:<7} {e.section:<8} 0x{e.address:08X}  {e.detail}")
    return out
