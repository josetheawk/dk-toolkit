"""Prompt v8: the target asm annotated with what each load, store and call touches (the LLM plan, Round 2).

`lwz r3, 0x24(r31)  ; this->speed (float)`. Only from evidence, never a guess:

- registers are typed by asmflow's value numbering (copies and address arithmetic folded, a join of
  different values is unknown): `this` (when the unit's DWARF or the class declaration says the method
  is not static), the mangled name's pointer-to-class parameters (`arg0`... as m2c names them),
  globals, and what a typed load or call returns;
- members from the unit's `-sym on` DWARF layouts (bases flattened, structs held by value followed,
  unions and duplicate offsets left out); a load or store whose width does not fit the member's
  type gets nothing;
- globals by DWARF (else their declaration in the visible text), with their type;
- calls by their demangled C++ name, with the return type when the declaration index (prompt v7's)
  has one prototype for it; a virtual call through `__vptr$` by the static type's vtable slot;
- strings and float constants by their data (dtk's `.string`/`.float`/`.double` objects).

Anything else gets no comment. The output is deterministic and budgeted: first occurrences of
members, then calls, then globals and data, then repeats, each tier in address order, until
`budget` characters of comments.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass

from . import asmflow

BUDGET = 2400            # comment characters at most per function (~1.1k tokens at 2.2 chars/token)
MAX_EXPR = 48            # a longer access chain is shown as Class::member
MAX_TEXT = 40            # string literals cut beyond this

TAG_MEMBER, TAG_INHERITANCE, TAG_UNION, TAG_PARAM, TAG_GLOBAL = 0x0D, 0x1C, 0x17, 0x05, 0x07
TAG_SUBROUTINES = (0x06, 0x14)
AT_NAME, AT_FUND, AT_UDT, AT_MOD_UDT, AT_MOD_FUND, AT_BYTE_SIZE, AT_LOCATION, AT_MW_MANGLED = \
    0x03, 0x05, 0x07, 0x08, 0x06, 0x0B, 0x02, 0x200
SIZES = {"char": 1, "signed char": 1, "unsigned char": 1, "bool": 1, "short": 2, "unsigned short": 2, "int": 4,
         "unsigned int": 4, "long": 4, "unsigned long": 4, "float": 4, "double": 8, "long double": 8,
         "long long": 8, "unsigned long long": 8, "void*": 4}
WIDTH = {"lbz": 1, "lbzu": 1, "stb": 1, "stbu": 1, "lhz": 2, "lhzu": 2, "lha": 2, "lhau": 2, "sth": 2, "sthu": 2,
         "lwz": 4, "lwzu": 4, "stw": 4, "stwu": 4, "lfs": 4, "lfsu": 4, "stfs": 4, "stfsu": 4, "lfd": 8, "lfdu": 8,
         "stfd": 8, "stfdu": 8}
BIAS = {4503601774854144.0: "int→float bias", 4503599627370496.0: "unsigned→float bias"}
_LINE = re.compile(r"^(\s*)([0-9A-F]+)  (\S.*?)\s*$")


@dataclass
class Typed:
    ref: int                 # the struct (DWARF offset) a register points to, or holds by value
    expr: str                # how the source reaches it (`this`, `this->mNext`, `gCar`)
    value: bool = False      # the struct itself (a global held by value): members with `.`


class Evidence:
    """What annotate may say about one function: layouts (draftfields.Layouts or None), the declaration
    index (llmdecls.Index or None) and dtk's asm entries (name -> AsmEntry: data objects, vtables)."""

    def __init__(self, lay=None, index=None, entries: dict | None = None):
        self.lay = lay if lay is not None and getattr(lay, "all", None) else None
        self.index, self.entries = index, entries or {}
        self._globals: dict[str, object] | None = None
        self._vt: dict[str, str] | None = None

    # ---- DWARF
    def struct(self, name: str) -> int | None:
        return self.lay.by_name.get(name) if self.lay else None

    def struct_name(self, ref: int) -> str:
        return str(self.lay.all[ref].attrs.get(AT_NAME, "?")) if self.lay else "?"

    def type_of(self, die) -> str | None:
        return self.lay.namer.of(die) if self.lay else None

    def size_of(self, die, depth: int = 0) -> int | None:
        a = die.attrs
        if AT_FUND in a:
            from .dwarftypes import FUND
            return SIZES.get(FUND.get(int(a[AT_FUND]), ""))
        if AT_MOD_UDT in a or AT_MOD_FUND in a:
            block = bytes(a.get(AT_MOD_UDT, a.get(AT_MOD_FUND)))
            mods = block[:-4] if AT_MOD_UDT in a else block[:-2]
            if mods and mods[0] in (1, 2):              # a pointer or a reference
                return 4
            return None
        if AT_UDT in a and depth < 8:
            d = self.lay.all.get(a[AT_UDT])
            if d is None:
                return None
            if AT_BYTE_SIZE in d.attrs:
                return int(d.attrs[AT_BYTE_SIZE])
            return self.size_of(d, depth + 1) if d.tag == 0x16 else None   # a typedef
        return None

    def _die(self, ref: int, name: str, depth: int = 0):
        """The member DIE called `name` in a struct or its bases (None when absent or ambiguous)."""
        found = []
        for d in self.lay._children(ref):
            if d.tag == TAG_MEMBER and d.attrs.get(AT_NAME) == name:
                found.append(d)
            elif d.tag == TAG_INHERITANCE and depth < 8:
                base = self.lay.struct(d.attrs.get(AT_UDT))
                hit = self._die(base, name, depth + 1) if base is not None else None
                if hit is not None:
                    found.append(hit)
        return found[0] if len(found) == 1 else None

    def at(self, ref: int, off: int, depth: int = 0, whole: bool = False) -> list | None:
        """The member path (draftfields.Member list) at a byte offset; None when unknown or ambiguous.
        `whole`: a struct member held by value at the offset itself, not its first member (an address)."""
        if self.lay is None or depth > 4:
            return None
        ms = [m for m in self.lay.members(ref) if not m.bits]
        exact = [m for m in ms if m.offset == off]
        if self.lay.all[ref].tag == TAG_UNION or len(exact) > 1:
            return None
        if exact:
            m = exact[0]
            if m.embedded is not None and not whole:
                inner = self.at(m.embedded, 0, depth + 1)
                return [m] + inner if inner else None
            return [m]
        inside = [m for m in ms if m.embedded is not None and m.offset < off < m.offset + self.lay.size(m.embedded)]
        if len(inside) == 1:
            inner = self.at(inside[0].embedded, off - inside[0].offset, depth + 1, whole)
            return [inside[0]] + inner if inner else None
        return None

    def leaf(self, ref: int, path: list):
        """The DIE of a member path's last member."""
        cur, die = ref, None
        for m in path:
            die = self._die(cur, m.name) if cur is not None else None
            if die is None:
                return None
            cur = m.embedded
        return die

    def vptr_at(self, ref: int, off: int, depth: int = 0) -> bool:
        """Does the struct (or a base) keep its vtable pointer (`__vptr$`) at this offset?"""
        for d in self.lay._children(ref):
            loc = d.attrs.get(AT_LOCATION)
            o = struct.unpack(">I", bytes(loc)[1:5])[0] if isinstance(loc, bytes) and len(loc) >= 5 else None
            if o is None:
                continue
            if d.tag == TAG_MEMBER and str(d.attrs.get(AT_NAME, "")).startswith("__vptr") and o == off:
                return True
            if d.tag == TAG_INHERITANCE and depth < 8:
                base = self.lay.struct(d.attrs.get(AT_UDT))
                if base is not None and self.vptr_at(base, off - o, depth + 1):
                    return True
        return False

    # ---- globals
    def global_(self, name: str) -> tuple[str | None, int | None, bool] | None:
        """(type text, struct ref, held by value) of a global; None when nothing declares it."""
        if self._globals is None:
            self._globals = {}
            if self.lay:
                for d in self.lay.all.values():
                    if d.tag == TAG_GLOBAL and d.attrs.get(AT_NAME):
                        self._globals.setdefault(str(d.attrs[AT_NAME]), d)
        d = self._globals.get(name)
        if d is not None:
            t = self.type_of(d)
            if AT_UDT in d.attrs:
                ref = self.lay.struct(d.attrs[AT_UDT])
                return t, ref, ref is not None
            if AT_MOD_UDT in d.attrs:
                block = bytes(d.attrs[AT_MOD_UDT])
                if block[:-4] == bytes([1]):
                    at = d.block_at[AT_MOD_UDT] + len(block) - 4
                    ref = self.lay.struct(self.lay.namer.refs.get(at, struct.unpack(">I", block[-4:])[0]))
                    return t, ref, False
            return t, None, False
        if self.index is not None and name in self.index.globals:
            decl = self.index.globals[name][1]
            m = re.match(rf"^(?:(?:extern|static)\s+)*(.+?)\s*\b{re.escape(name)}\s*(\[[^\]]*\])?\s*;$", decl)
            if not m or "(" in m[1]:
                return None
            t = " ".join(m[1].replace("*", " * ").split()).replace(" *", "*")
            if m[2]:
                return t + "[]", None, False
            base = re.fullmatch(r"(?:const\s+)?(\w+)(\*?)", t)
            ref = self.struct(base[1]) if base else None
            return t, ref, ref is not None and not base[2]
        return None

    def data(self, name: str) -> str | None:
        """A string literal or float constant's value as the comment shows it."""
        e = self.entries.get(name)
        if e is None or getattr(e, "kind", "") != "obj":
            return None
        body = [l.strip() for l in e.text.splitlines()[1:-1] if l.strip() and not l.strip().startswith("#")]
        if len(body) > 1 and all(l.startswith(".float ") for l in body):
            body = body[:1]                             # a float table: the load at the label reads the first
        if len(body) != 1:
            return None
        m = re.match(r'^\.string\s+(".*")$', body[0])
        if m:
            s = m[1]
            return s if len(s) <= MAX_TEXT else s[:MAX_TEXT - 2] + '…"'
        m = re.match(r"^\.(float|double)\s+(\S+)$", body[0])
        if m:
            try:
                v = float(m[2])
            except ValueError:
                return None
            if m[1] == "double" and v in BIAS:
                return BIAS[v]
            import math
            if not math.isfinite(v):
                return None
            text = repr(v) if v != int(v) or abs(v) >= 1e16 else f"{int(v)}.0"
            return text + ("f" if m[1] == "float" else "")
        return None

    # ---- calls
    def returns(self, symbol: str) -> str | None:
        """A called function's return type from the declaration index (one consistent declaration)."""
        if self.index is None:
            return None
        from .demangle import demangle
        try:
            dm = demangle(symbol)
        except Exception:
            dm = None
        if dm is None:
            got = self.index.funcs.get(symbol)
            return _ret(got[1], symbol) if got else None
        if dm.name in ("ctor", "dtor") or dm.name.startswith("__"):
            return None
        if not dm.quals:
            got = self.index.funcs.get(dm.name)
            return _ret(got[1], dm.name) if got and _arity(got[1]) == len(dm.params) else None
        decl = self.index.types.get(dm.quals[-1])
        if decl is None or "{" not in decl.text:
            return None
        from .llmdecls import members
        _h, ms, _t = members(decl)
        rets = {_ret(m.text, m.name) for m in ms if m.name == dm.name and m.kind in ("method", "virtual", "inline")
                and _arity(m.text) == len(dm.params)}
        return rets.pop() if len(rets) == 1 else None

    def is_static(self, symbol: str, dm) -> bool | None:
        """Is a member function static? From the unit's DWARF (a `this` parameter), else its declaration."""
        if dm.name in ("ctor", "dtor"):
            return False
        if self.lay:
            order = self.lay.order
            for k, off in enumerate(order):
                d = self.lay.all[off]
                if d.tag in TAG_SUBROUTINES and d.attrs.get(AT_MW_MANGLED) == symbol:
                    nxt = self.lay.all[order[k + 1]] if k + 1 < len(order) else None
                    # a method's first child is its `this` parameter; none at all: static
                    return not (nxt is not None and nxt.tag == TAG_PARAM and nxt.attrs.get(AT_NAME) == "this")
        decl = self.index.types.get(dm.quals[-1]) if self.index is not None and dm.quals else None
        if decl is None or "{" not in decl.text:
            return None
        from .llmdecls import members
        _h, ms, _t = members(decl)
        same = [m for m in ms if m.name == dm.name and m.kind in ("method", "virtual", "inline")]
        if not same:
            return None
        flags = {bool(re.match(r"\s*static\b", m.text)) for m in same}
        return flags.pop() if len(flags) == 1 else None

    def vtable(self, ref: int) -> list[str] | None:
        """The function symbols of a class's vtable (slot 8 first), from its `__vt__` object."""
        if self._vt is None:
            from .demangle import demangle
            self._vt = {}
            for name, e in self.entries.items():
                if name.startswith("__vt__") and getattr(e, "kind", "") == "obj":
                    try:
                        dm = demangle("f__" + name[6:] + "Fv")
                    except Exception:
                        dm = None
                    if dm is not None and dm.quals:
                        last = dm.quals[-1]
                        self._vt[last] = name if last not in self._vt else ""      # ambiguous: none
        sym = self._vt.get(self.struct_name(ref)) if self.lay else None
        if not sym:
            return None
        words = re.findall(r'^\s*\.4byte\s+(?:"([^"]+)"|(\S+))', self.entries[sym].text, re.M)
        return [a or b for a, b in words][2:]


def _arity(proto: str) -> int:
    inner = proto[proto.find("(") + 1:proto.rfind(")")] if "(" in proto else ""
    inner = re.sub(r"\([^()]*\)", "", inner).strip()
    if inner in ("", "void"):
        return 0
    return inner.count(",") + 1


def _ret(proto: str, name: str) -> str | None:
    head = proto.split("(", 1)[0]
    m = re.match(rf"^(.*?)\b{re.escape(name)}\s*$", head.strip())
    if not m:
        return None
    t = re.sub(r"\b(?:virtual|static|inline|explicit|extern)\b|\"C\"", "", m[1])
    t = " ".join(t.replace("*", " * ").replace("&", " & ").split()).replace(" *", "*").replace(" &", "&")
    return t or None


def call_text(symbol: str) -> str | None:
    """`Class::name(int, Car*) const` for a mangled function, else None."""
    from .demangle import demangle
    from .llmtry import qualified
    q = qualified(symbol)
    if not q:
        return None
    try:
        dm = demangle(symbol)
    except Exception:
        return None
    if dm is None:
        return None
    return f"{q}({', '.join(p.text for p in dm.params)})" + (" const" if dm.const else "")


# ---- the annotation ---------------------------------------------------------------------------------

def _raw(asm: str) -> tuple[str, dict[int, int]]:
    """A compact listing (llmtry.compact_asm) back in dtk's form for asmflow, and offset -> line index."""
    out, where = [], {}
    for i, line in enumerate(asm.splitlines()):
        m = _LINE.match(line)
        if m:
            off = int(m[2], 16)
            mnem, _, ops = m[3].partition(" ")
            out.append(f"/* {off:08X} 00000000  00 00 00 00 */\t{mnem}" + (f" {ops.strip()}" if ops.strip() else ""))
            where[off] = i
        else:
            out.append(line)
    return "\n".join(out), where


def _short(expr: str, ev: Evidence, ref: int, path: str) -> str:
    return expr if len(expr) <= MAX_EXPR else f"{ev.struct_name(ref)}::{path}"


def notes(asm: str, symbol: str, ev: Evidence, budget: int = BUDGET) -> dict[int, str]:
    """offset -> comment for one function's compact asm."""
    from .demangle import demangle
    raw, where = _raw(asm)
    try:
        flow = asmflow.flow_of(symbol, raw)
    except Exception:
        return {}
    try:
        dm = demangle(symbol)
    except Exception:
        dm = None
    roots: dict[tuple, Typed] = {}
    if dm is not None and ev.lay is not None:
        gprs = [f"r{i}" for i in range(3, 11)]
        fprs = [f"f{i}" for i in range(1, 9)]
        if dm.quals:
            static = ev.is_static(symbol, dm)
            cls = ev.struct(dm.quals[-1])
            if static is False and cls is not None:
                roots[("arg", "r3")] = Typed(cls, "this")
            if static is not False:
                gprs = gprs if static is True else []     # unknown: no argument is typed
            else:
                gprs = gprs[1:]
        for i, p in enumerate(dm.params):
            if p.reg == "fpr" and fprs:
                fprs.pop(0)
                continue
            if p.reg != "gpr" or "long long" in p.text or not gprs:
                break
            reg = gprs.pop(0)
            m = re.fullmatch(r"(?:const\s+)?(?:\w+::)*(\w+)\s*[*&]", p.text.strip())
            ref = ev.struct(m[1]) if m else None
            if ref is not None:
                roots[("arg", reg)] = Typed(ref, f"arg{i}")
    results: dict[int, Typed] = {}

    def typed(v) -> Typed | None:
        root, off = v
        if off != 0:
            return None
        if root[0] == "arg":
            return roots.get(root)
        if root[0] == "n":
            return results.get(root[1]) if root[2] in ("", "r3") else None
        if root[0] == "sym":
            g = ev.global_(root[1])
            if g and g[1] is not None and g[2]:
                return Typed(g[1], root[1], True)
        return None

    def member(v, mnem: str) -> tuple[str, list, object, Typed] | None:
        root, off = v
        base = typed((root, 0))
        if base is None:
            return None
        path = ev.at(base.ref, off, whole=not mnem)
        if not path:
            return None
        text = ".".join(m.name + ("[0]" if m.array else "") for m in path)
        die = ev.leaf(base.ref, path)
        t = ev.type_of(die) if die is not None else None
        size = ev.size_of(die) if die is not None else None
        want = WIDTH.get(mnem)
        if want is not None and size is not None and not path[-1].array and (
                size != want or mnem.startswith(("lf", "stf")) and t is not None and "float" not in t and "double" not in t
                and size == 4 and not t.startswith(("f32", "f64", "Float"))):
            return None
        sep = "." if base.value else "->"
        return _short(f"{base.expr}{sep}{text}", ev, base.ref, text), path, t, base

    found: list[tuple[int, int, str]] = []           # (tier, offset, comment)
    for n in flow.nodes:
        off = n.insn.addr
        m = n.mnem
        if n.kind == "call":
            sym = n.key[3:] if n.key.startswith("bl ") else ""
            text = call_text(sym) if sym else None
            ret = ev.returns(sym) if sym else None
            if sym and text is None and ret is not None:
                text = f"{sym}()"
            if text:
                found.append((1, off, f"{ret + ' ' if ret else ''}{text}"))
            if ret:
                r = re.fullmatch(r"(?:const\s+)?(?:\w+::)*(\w+)\*", ret)
                ref = ev.struct(r[1]) if r else None
                if ref is not None:
                    name = (text or sym).split("(", 1)[0]
                    results[n.i] = Typed(ref, f"{name}()")
            if n.key == "bctrl" and n.ins:
                vt = _virtual(n.ins[0], flow, ev, typed)
                if vt:
                    found.append((1, off, f"virtual {vt}"))
            continue
        if n.kind == "pure" and m in ("li", "addi", "ori"):
            v = n.outs.get(n.insn.ops[0]) if n.insn.ops else None
            if v is not None and v[0][0] == "sym" and v[1] == 0 and any("@" in o for o in n.insn.ops[1:]):
                name = v[0][1]
                d = ev.data(name)
                if d:
                    found.append((2, off, d))
                else:
                    g = ev.global_(name)
                    if g and g[0]:
                        found.append((2, off, f"&{name} ({g[0]})"))
            elif v is not None and m == "addi" and v[1] != 0:
                hit = member(v, "")
                if hit and len(hit[1]) and not hit[1][-1].array:
                    found.append((0, off, f"&{hit[0]}"))
            continue
        if n.mem is None or n.kind != "op" or m.startswith("psq"):
            continue
        root, moff = n.mem
        if root[0] == "sym" and moff == 0:
            d = ev.data(root[1])
            if d:
                found.append((2, off, d))
                continue
            g = ev.global_(root[1])
            if g and g[0] and not g[2]:
                found.append((2, off, f"{root[1]} ({g[0]})"))
                if g[1] is not None and m in ("lwz", "lwzu"):
                    results[n.i] = Typed(g[1], root[1])
                continue
        hit = member(n.mem, m)
        if hit is None:
            continue
        expr, path, t, base = hit
        found.append((0, off, f"{expr} ({t})" if t else expr))
        if not n.store and m in ("lwz", "lwzu") and path[-1].points_to is not None:
            results[n.i] = Typed(path[-1].points_to, expr)
    return _budget(found, budget, where)


def _virtual(ctr, flow, ev: Evidence, typed) -> str | None:
    """A bctrl's callee: ctr = load(load(obj + vptr) + slot) with obj's class known and `__vptr$` there."""
    root, off = ctr
    if root[0] != "n" or off != 0:
        return None
    slot_load = flow.nodes[root[1]]
    if slot_load.mnem != "lwz" or slot_load.mem is None:
        return None
    vroot, slot = slot_load.mem
    if vroot[0] != "n" or slot < 8 or slot % 4:
        return None
    vptr_load = flow.nodes[vroot[1]]
    if vptr_load.mnem != "lwz" or vptr_load.mem is None:
        return None
    obj = typed((vptr_load.mem[0], 0))
    if obj is None or obj.value or not ev.vptr_at(obj.ref, vptr_load.mem[1]):
        return None
    table = ev.vtable(obj.ref)
    k = (slot - 8) // 4
    if not table or k >= len(table) or table[k] in ("0x00000000", "0"):
        return None
    return call_text(table[k]) or table[k]


def _budget(found: list[tuple[int, int, str]], budget: int, where: dict[int, int]) -> dict[int, str]:
    """First occurrences by tier (members, calls, globals and data), then repeats, in address order,
    while the comments fit the budget."""
    seen: set[str] = set()
    ranked = []
    for tier, off, text in sorted(found, key=lambda x: x[1]):
        ranked.append((tier + (3 if text in seen else 0), off, text))
        seen.add(text)
    out: dict[int, str] = {}
    used = 0
    for tier, off, text in sorted(ranked):
        if off in out or off not in where:
            continue
        cost = len(text) + 4
        if used + cost > budget:
            continue
        out[off] = text
        used += cost
    return out


def apply(asm: str, comments: dict[int, str]) -> str:
    """The compact asm with `  ; comment` after the annotated instructions."""
    lines = asm.splitlines()
    for i, line in enumerate(lines):
        m = _LINE.match(line)
        if m and int(m[2], 16) in comments:
            lines[i] = line.rstrip() + "  ; " + comments[int(m[2], 16)]
    return "\n".join(lines)


def annotate(asm: str, symbol: str, ev: Evidence, budget: int = BUDGET) -> str:
    return apply(asm, notes(asm, symbol, ev, budget))


def stored(asm: str, comments: dict[int, str]) -> list[list[str]]:
    """Comments as a record keeps them: [offset hex, instruction, comment], so they are applied only
    to the same instruction (llmdata's annotation file)."""
    lines = {int(m[2], 16): m[3] for m in map(_LINE.match, asm.splitlines()) if m}
    return [[f"{off:X}", lines[off], text] for off, text in sorted(comments.items()) if off in lines]


def apply_stored(asm: str, rows: list[list[str]]) -> str:
    lines = {int(m[2], 16): m[3] for m in map(_LINE.match, asm.splitlines()) if m}
    return apply(asm, {int(o, 16): c for o, insn, c in rows if lines.get(int(o, 16)) == insn})


# ---- coverage (dk dataset annotate --report) -------------------------------------------------------------

_ACCESS = re.compile(r"(?:->|\.)\s*([A-Za-z_]\w*)\b(?!\s*\()")
_NAMED = re.compile(r"(?:->|\.|::)([A-Za-z_]\w*)")


def named_members(comments) -> set[str]:
    """The member names the comments spell (`this->a.b` -> {a, b})."""
    out: set[str] = set()
    for c in comments:
        head = c.split(" (", 1)[0]
        if "(" in head:                                  # a call: its return type and name, not members
            continue
        out |= set(_NAMED.findall(head.replace("[0]", "")))
    return out


def answer_fields(code: str, own_members: set[str] = frozenset()) -> set[str]:
    """The data members an answer reaches: after `->` or `.`, plus the owner class's members used bare."""
    from .llmdecls import strip_comments
    text = strip_comments(code)
    out = {m for m in _ACCESS.findall(text) if not re.fullmatch(r"\d.*|f", m)}
    if own_members:
        out |= {w for w in re.findall(r"\b[A-Za-z_]\w*\b", text) if w in own_members}
    return out


# ---- dk dataset annotate: the comments of every record's function, and their coverage --------------------

ANNOT_FILE = "annot.jsonl"


def kind_of(comment: str) -> str:
    if comment.startswith("virtual "):
        return "virtual call"
    if comment.startswith('"'):
        return "string"
    if comment.startswith("&") and "->" not in comment and "." not in comment.split(" (")[0]:
        return "global address"
    head = comment.split(" (", 1)[0]
    if "(" in head or comment.endswith(")") and "(" in comment and "->" not in comment and " (" not in comment:
        return "call"
    if re.fullmatch(r"-?[\d.e+-]+f?|.*bias", comment):
        return "constant"
    if "->" in head or "." in head or head.startswith("&"):
        return "member"
    return "global"


def configure(sub) -> None:
    a = sub.add_parser("annotate", help="round 2 (prompt v8): llmannot's asm comments for every record's function "
                       "(the unit's -sym on layouts, its declarations, the data) -> records/annot.jsonl, and their "
                       "coverage: the base answers' members named by a comment, token growth; CPU, one compile per unit")
    a.add_argument("--only", default="", help="units whose name contains one of these (comma-separated)")
    a.add_argument("--limit", type=int, default=0, help="first N units only (a test)")
    a.add_argument("--budget", type=int, default=BUDGET, help=f"comment characters per function (default {BUDGET})")
    a.add_argument("--jobs", type=int, default=0, help="parallel units (default the idle cores, at most 6)")
    a.add_argument("--out", default="", help="dataset directory (default llm-data/dataset)")


def _sha(text: str) -> str:
    import hashlib
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def run_dataset(cfg, args) -> int:
    import json
    import time
    from concurrent.futures import ThreadPoolExecutor
    from .llmdata import SOURCES, out_dir, read_jsonl
    from .llmdecls import unit_index
    from .llmtry import CHARS_PER_TOKEN, asm_entries, unit_layouts
    from .matching import resolve_unit
    from .trycmd import locate_target
    t0 = time.time()
    root = out_dir(cfg, args)
    asms: dict[tuple[str, str], dict[str, str]] = {}
    bases: dict[tuple[str, str], dict] = {}
    for s in SOURCES:
        for r in read_jsonl(root / "records" / f"{s}.jsonl") if (root / "records" / f"{s}.jsonl").is_file() else []:
            k = (r["unit"], r["function"])
            asms.setdefault(k, {})[_sha(r["asm"])] = r["asm"]
            if s == "base":
                bases[k] = r
    only = [x for x in args.only.split(",") if x]
    units = sorted({u for u, _f in asms if not only or any(o in u for o in only)})
    if args.limit:
        units = units[:args.limit]
    entries = asm_entries(cfg)

    def one(unit: str) -> list[dict]:
        try:
            u = resolve_unit(cfg, unit)
            source = cfg.root / u["metadata"]["source_path"]
            src = source.read_text()
            lay = unit_layouts(cfg, u, src)
        except Exception:
            return []
        out = []
        for (un, fn), texts in sorted(asms.items()):
            if un != unit:
                continue
            try:
                span, _loc, _n = locate_target(cfg, src, source, fn)
                ev = Evidence(lay, unit_index(cfg, src, source, span), entries)
            except Exception:
                continue
            own = set()
            from .demangle import demangle
            try:
                dm = demangle(fn)
            except Exception:
                dm = None
            if dm is not None and dm.quals and ev.struct(dm.quals[-1]) is not None:
                own = {m.name for m in lay.members(ev.struct(dm.quals[-1]))}
            for sha, asm in sorted(texts.items()):
                try:
                    c = notes(asm, fn, ev, args.budget)
                except Exception as e:                   # one function's odd asm never stops the run
                    print(f"  {fn}: no annotation ({type(e).__name__}: {e})")
                    c = {}
                out.append({"unit": un, "function": fn, "asm_sha": sha, "notes": stored(asm, c), "own": sorted(own)})
        return out
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        rows = [r for rs in pool.map(one, units) for r in rs]
    dest = root / "records" / ANNOT_FILE
    dest.write_text("".join(json.dumps({k: v for k, v in r.items() if k != "own"}) + "\n" for r in rows))
    report = coverage_report(rows, bases, CHARS_PER_TOKEN)
    (root / "annot-report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"annotate: {len(rows)} function asms in {len(units)} units into {cfg.rel(dest)} ({time.time() - t0:.0f} s)")
    for k, v in report.items():
        print(f"  {k}: {v}")
    return 0


def coverage_report(rows: list[dict], bases: dict, cpt: float) -> dict:
    """Over the base records: the answers' members named by a comment, comments by kind, token growth."""
    from .llmdata import builder, render
    by = {(r["unit"], r["function"], r["asm_sha"]): r for r in rows}
    build7 = builder(7)
    fields = named = recs = with_any = 0
    per_rec: list[float] = []
    grow: list[float] = []
    rel: list[float] = []
    kinds: dict[str, int] = {}
    access_insns = annotated_insns = 0
    for k, b in sorted(bases.items()):
        r = by.get((k[0], k[1], _sha(b["asm"])))
        if r is None:
            continue
        recs += 1
        comments = [c for _o, _i, c in r["notes"]]
        with_any += bool(comments)
        for c in comments:
            kinds[kind_of(c)] = kinds.get(kind_of(c), 0) + 1
        for line in b["asm"].splitlines():
            m = _LINE.match(line)
            if m and re.match(r"(?:l[bhwf]|st[bhwf]|bl\b|bctrl)", m[3]) and "(r1)" not in m[3]:
                access_insns += 1
        annotated_insns += len(comments)
        want = answer_fields(b["exact"], set(r.get("own", ())))
        got = want & named_members(comments)
        fields += len(want)
        named += len(got)
        if want:
            per_rec.append(len(got) / len(want))
        plain = b["asm"]
        ann = apply_stored(plain, r["notes"])
        added = (len(ann) - len(plain)) / cpt
        grow.append(added)
        view = b.get("views", {}).get("7")
        total = sum(len(m["content"]) for m in render(b, build7, view)["messages"][:-1]) / cpt
        rel.append(added / max(1.0, total))

    def q(xs: list[float], at: float) -> float:
        xs = sorted(xs)
        return round(xs[min(len(xs) - 1, int(at * len(xs)))], 3) if xs else 0.0
    return {"base_records": recs, "with_a_comment": with_any,
            "answer_member_accesses": fields, "named_by_a_comment": named,
            "named_share": round(named / max(1, fields), 3),
            "per_record_named_share": {"mean": round(sum(per_rec) / max(1, len(per_rec)), 3), "median": q(per_rec, 0.5)},
            "comments_by_kind": dict(sorted(kinds.items(), key=lambda kv: -kv[1])),
            "loads_stores_calls_annotated": round(annotated_insns / max(1, access_insns), 3),
            "added_tokens": {"median": q(grow, 0.5), "p90": q(grow, 0.9), "max": q(grow, 1.0)},
            "growth_of_v7_prompt": {"median": q(rel, 0.5), "p90": q(rel, 0.9)}}
