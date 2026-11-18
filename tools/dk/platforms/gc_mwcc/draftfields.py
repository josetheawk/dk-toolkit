"""Clean m2c drafts for the prompt (version 6): m2c's guessed `p->unkNN` becomes the member the
declarations put at that offset, and a constructor's compiler-written parts go.

The layouts come from the unit's own `-sym on` compile (DWARF 1 member DIEs with their offsets and
types), so no C parser is needed. Only accesses whose base type is known are renamed: `this` (the
method's class), `argN` (the mangled name's parameter types), a global declared as a pointer to a
class, and chains through members of known type. Bitfields and offsets inside arrays are left as
m2c wrote them.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass

from .dwarftypes import (AT_MOD_UDT, AT_NAME, AT_SIBLING, AT_UDT, MOD_POINTER, TAG_CLASS, TAG_INHERITANCE,
                         TAG_STRUCT, TAG_TYPEDEF, TAG_UNION, TypeNamer)
from .regs import debug_refs

AT_LOCATION, AT_BYTE_SIZE, AT_BIT_SIZE = 0x02, 0x0B, 0x0D
TAG_MEMBER, TAG_ARRAY = 0x0D, 0x01
OP_CONST = 0x04


@dataclass
class Member:
    name: str
    offset: int
    embedded: int | None = None     # the struct it holds by value
    points_to: int | None = None    # the struct it points to
    array: bool = False
    bits: bool = False


class Layouts:
    """Struct layouts of one `-sym on` object."""

    def __init__(self, elf):
        sec = elf.section(".debug")
        self.namer = TypeNamer(sec.data, debug_refs(elf)) if sec is not None else None
        self.all = self.namer.all if self.namer else {}
        self.order = sorted(self.all)
        self.by_name: dict[str, int] = {}
        for off in self.order:
            d = self.all[off]
            if d.tag in (TAG_STRUCT, TAG_CLASS, TAG_UNION) and d.attrs.get(AT_NAME) and AT_BYTE_SIZE in d.attrs:
                self.by_name.setdefault(str(d.attrs[AT_NAME]), off)
        self._cache: dict[int, list[Member]] = {}

    def struct(self, ref) -> int | None:
        for _ in range(8):
            d = self.all.get(ref) if isinstance(ref, int) else None
            if d is None:
                return None
            if d.tag == TAG_TYPEDEF and isinstance(d.attrs.get(AT_UDT), int):
                ref = d.attrs[AT_UDT]
                continue
            if d.tag in (TAG_STRUCT, TAG_CLASS, TAG_UNION):
                name = d.attrs.get(AT_NAME)         # a declaration only: the definition by name
                return ref if AT_BYTE_SIZE in d.attrs or not name else self.by_name.get(str(name), ref)
            return None
        return None

    def size(self, ref: int) -> int:
        return int(self.all[ref].attrs.get(AT_BYTE_SIZE, 0))

    def _children(self, ref: int):
        end = self.all[ref].attrs.get(AT_SIBLING)
        k = self.order.index(ref) + 1
        child = self.order[k] if k < len(self.order) else None
        while child is not None and child in self.all and (not isinstance(end, int) or child < end):
            yield self.all[child]
            nxt = self.all[child].attrs.get(AT_SIBLING)
            child = nxt if isinstance(nxt, int) and nxt > child else None

    def members(self, ref: int, depth: int = 0) -> list[Member]:
        """Data members at their offsets, bases' included."""
        if ref in self._cache:
            return self._cache[ref]
        out: list[Member] = []
        if depth > 8:
            return out
        for d in self._children(ref):
            loc = d.attrs.get(AT_LOCATION)
            off = struct.unpack(">I", bytes(loc)[1:5])[0] if isinstance(loc, bytes) and len(loc) >= 5 \
                and loc[0] == OP_CONST else None
            if d.tag == TAG_INHERITANCE and off is not None:
                base = self.struct(d.attrs.get(AT_UDT))
                if base is not None:
                    out += [Member(m.name, m.offset + off, m.embedded, m.points_to, m.array, m.bits)
                            for m in self.members(base, depth + 1)]
            elif d.tag == TAG_MEMBER and off is not None and d.attrs.get(AT_NAME) and \
                    not str(d.attrs[AT_NAME]).startswith("__"):
                m = Member(str(d.attrs[AT_NAME]), off, bits=AT_BIT_SIZE in d.attrs)
                if isinstance(d.attrs.get(AT_UDT), int):
                    t = self.all.get(d.attrs[AT_UDT])
                    m.array = t is not None and t.tag == TAG_ARRAY
                    m.embedded = None if m.array else self.struct(d.attrs[AT_UDT])
                elif AT_MOD_UDT in d.attrs:
                    block = bytes(d.attrs[AT_MOD_UDT])
                    if block[:-4] == bytes([MOD_POINTER]):
                        at = d.block_at[AT_MOD_UDT] + len(block) - 4
                        m.points_to = self.struct(self.namer.refs.get(at, struct.unpack(">I", block[-4:])[0]))
                out.append(m)
        self._cache[ref] = out
        return out

    def at(self, ref: int, off: int, depth: int = 0) -> tuple[str, Member] | None:
        """The member access (`m10` or `m10.m4`) at a byte offset, and its member."""
        ms = self.members(ref)
        for m in ms:
            if m.offset == off and not m.bits:
                if m.embedded is not None and depth < 4:       # a struct held by value: its first member
                    inner = self.at(m.embedded, 0, depth + 1)
                    if inner is not None:
                        return f"{m.name}.{inner[0]}", inner[1]
                return (m.name + "[0]" if m.array else m.name), m
        if depth < 4:
            for m in ms:
                if m.embedded is not None and m.offset < off < m.offset + self.size(m.embedded):
                    inner = self.at(m.embedded, off - m.offset, depth + 1)
                    if inner is not None:
                        return f"{m.name}.{inner[0]}", inner[1]
        return None

    def named(self, ref: int, name: str) -> Member | None:
        return next((m for m in self.members(ref) if m.name == name), None)


_CHAIN = re.compile(r"\b(this|arg\d+|lbl_[0-9A-Fa-f]+)((?:\s*(?:->|\.)\s*\w+(?:\[0\])?)+)")
_STEP = re.compile(r"\s*(->|\.)\s*(\w+)(\[0\])?")


def name_fields(draft: str, lay: Layouts, this_class: str | None, params: list[str], visible: str = "") -> str:
    """The draft with `base->unkNN` chains renamed where the base's type is known."""
    if not lay.all:
        return draft
    roots: dict[str, int] = {}
    if this_class and this_class in lay.by_name:
        roots["this"] = lay.by_name[this_class]
    for i, t in enumerate(params):
        m = re.fullmatch(r"(?:const\s+)?(\w+)\s*\*", t.strip())
        if m and m[1] in lay.by_name:
            roots[f"arg{i}"] = lay.by_name[m[1]]

    def root(name: str) -> int | None:
        if name in roots:
            return roots[name]
        if name.startswith("lbl_"):
            m = re.search(rf"\b(\w+)\s*\*\s*{name}\s*;", visible)
            if m and m[1] in lay.by_name:
                return lay.by_name[m[1]]
        return None

    def chain(m: re.Match) -> str:
        cur = root(m[1])
        if cur is None:
            return m[0]
        out, pointer = m[1], True       # every root is a pointer to its struct
        rest = m[2]
        for s in _STEP.finditer(rest):
            arrow, name = s[1] == "->", s[2]
            if cur is None or arrow != pointer:
                out += rest[s.start():]
                break
            u = re.fullmatch(r"unk([0-9A-Fa-f]+)", name)
            hit = lay.at(cur, int(u[1], 16)) if u else None
            if u and hit is None:
                out += rest[s.start():]
                break
            if hit is not None:
                text, mem = hit
                out += s[1] + text + (s[3] or "" if not text.endswith("]") else "")
            else:
                mem = lay.named(cur, name)
                out += s[0]
                if mem is None:
                    cur = None
                    continue
            cur, pointer = (mem.points_to, True) if mem.points_to is not None else (mem.embedded, False)
            if mem.array:
                cur = None
        return out
    return _CHAIN.sub(chain, draft)


_VTABLE_STORE = re.compile(r"^[ \t]*[^;\n=]*=\s*(?:\([^;\n()]*\)\s*)?&?\s*__vt__\w+[^;\n]*;[ \t]*\n", re.M)
_BASE_CTOR = re.compile(r"^[ \t]*(?P<cls>[\w:]+)::(?P<name>\w+)\(\s*\((?P=cls)\s*\*\)\s*this\s*(?:,\s*(?P<args>[^;\n]*))?\);[ \t]*\n",
                        re.M)


_DELETE_THIS = re.compile(r"^[ \t]*if \(\w+ > 0\) \{[ \t]*\n[ \t]*(?:operator delete|__dl__FPv)\(this\);[ \t]*\n[ \t]*\}[ \t]*\n", re.M)
_THIS_GUARD = re.compile(r"^([ \t]*)if \(this != 0\) \{[ \t]*\n", re.M)


def clean_structors(draft: str, symbol: str) -> str:
    """A constructor's or destructor's compiler-written parts out of the draft: the vtable stores
    and `return this;`; a constructor's base constructor call becomes its initializer; a
    destructor loses its hidden flag parameter, its `this != 0` guard and its `operator delete`."""
    if not symbol.startswith(("__ct__", "__dt__")):
        return draft
    draft = _VTABLE_STORE.sub("", draft)
    draft = re.sub(r"^[ \t]*return this;[ \t]*\n", "", draft, flags=re.M)
    if symbol.startswith("__dt__"):
        from ...core import cbody
        draft = re.sub(r"(~\w+)\(short \w+\)", r"\1()", draft, count=1)
        draft = _DELETE_THIS.sub("", draft)
        m = _THIS_GUARD.search(draft)
        close = cbody.matching_brace(draft, m.end() - 2) if m else None
        if m and close is not None:
            inner = draft[m.end():close].rstrip(" \t")
            inner = re.sub(r"^    ", "", inner, flags=re.M)
            rest = draft[close + 1:]
            draft = draft[:m.start()] + inner + (rest[1:] if rest.startswith("\n") else rest)
    if symbol.startswith("__ct__"):
        inits = []

        def base(m: re.Match) -> str:
            if m["cls"].split("::")[-1] != m["name"]:
                return m[0]
            inits.append(f"{m['cls']}({(m['args'] or '').strip()})")
            return ""
        body = draft.find("{")
        if body > 0:
            rest = _BASE_CTOR.sub(base, draft[body:], count=1)
            if inits:
                draft = draft[:body].rstrip() + " : " + ", ".join(inits) + " " + rest
    return draft
