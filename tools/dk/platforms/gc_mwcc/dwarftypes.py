"""The C type of each local, from MWCC's DWARF 1 `.debug` (`-sym on`), written back as source.

`dk close` names sub-expressions as `__typeof__(expr) t0 = expr;` while it searches; before the
final check the real type is read here and written in its place, so no `__typeof__` is kept.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from .regs import TAG_SUBROUTINES, debug_refs

TAG_PARAM, TAG_LOCAL, TAG_BLOCK = 0x05, 0x0C, 0x0B
TAG_STRUCT, TAG_CLASS, TAG_UNION, TAG_ENUM, TAG_TYPEDEF = 0x13, 0x02, 0x17, 0x04, 0x16
TAG_POINTER, TAG_REFERENCE = 0x0F, 0x10
AT_SIBLING, AT_NAME, AT_FUND, AT_MOD_FUND, AT_UDT, AT_MOD_UDT, AT_MW_MANGLED = 0x01, 0x03, 0x05, 0x06, 0x07, 0x08, 0x200
FUND = {0x01: "char", 0x02: "signed char", 0x03: "unsigned char", 0x04: "short", 0x05: "short", 0x06: "unsigned short",
        0x07: "int", 0x08: "int", 0x09: "unsigned int", 0x0A: "long", 0x0B: "long", 0x0C: "unsigned long",
        0x0D: "void*", 0x0E: "float", 0x0F: "double", 0x10: "long double", 0x14: "void", 0x15: "bool",
        0x8008: "long long", 0x8108: "long long", 0x8208: "unsigned long long"}
MOD_POINTER, MOD_REFERENCE, MOD_CONST, MOD_VOLATILE = 1, 2, 3, 4


@dataclass
class Die:
    offset: int
    tag: int
    attrs: dict[int, object] = field(default_factory=dict)
    block_at: dict[int, int] = field(default_factory=dict)     # attribute -> offset of its block's data


def dies(data: bytes, refs: dict[int, int]) -> dict[int, Die]:
    out: dict[int, Die] = {}
    off = 0
    while off + 6 <= len(data):
        ln, = struct.unpack_from(">I", data, off)
        if ln < 8:
            off += max(ln, 4)
            continue
        tag, = struct.unpack_from(">H", data, off + 4)
        die = Die(off, tag)
        p, end = off + 6, off + ln
        while p + 2 <= end:
            an, = struct.unpack_from(">H", data, p)
            p += 2
            form = an & 0xF
            if form in (1, 2, 6):
                v, = struct.unpack_from(">I", data, p)
                v = refs.get(p, v)
                p += 4
            elif form == 5:
                v, = struct.unpack_from(">H", data, p)
                p += 2
            elif form == 7:
                v, p = data[p:p + 8], p + 8
            elif form == 3:
                n, = struct.unpack_from(">H", data, p)
                die.block_at[an >> 4] = p + 2
                v, p = data[p + 2:p + 2 + n], p + 2 + n
            elif form == 4:
                n, = struct.unpack_from(">I", data, p)
                die.block_at[an >> 4] = p + 4
                v, p = data[p + 4:p + 4 + n], p + 4 + n
            elif form == 8:
                z = data.index(b"\0", p)
                v, p = data[p:z].decode("latin-1"), z + 1
            else:
                break
            die.attrs[an >> 4] = v
        out[off] = die
        off = end
    return out


class TypeNamer:
    def __init__(self, data: bytes, refs: dict[int, int], c: bool = False):
        self.data, self.refs, self.c = data, refs, c
        self.all = dies(data, refs)
        self.typedef_of: dict[int, str] = {}
        for d in self.all.values():
            if d.tag == TAG_TYPEDEF and isinstance(d.attrs.get(AT_UDT), int) and d.attrs.get(AT_NAME):
                self.typedef_of.setdefault(d.attrs[AT_UDT], str(d.attrs[AT_NAME]))

    def udt(self, ref: int) -> str | None:
        d = self.all.get(ref)
        if d is None:
            return None
        name = d.attrs.get(AT_NAME)
        if d.tag == TAG_TYPEDEF:
            return str(name) if name else None
        if d.tag in (TAG_STRUCT, TAG_CLASS, TAG_UNION, TAG_ENUM):
            if self.c:
                if ref in self.typedef_of:
                    return self.typedef_of[ref]
                if not name:
                    return None
                return {TAG_STRUCT: "struct ", TAG_CLASS: "struct ", TAG_UNION: "union ", TAG_ENUM: "enum "}[d.tag] + str(name)
            return str(name) if name else self.typedef_of.get(ref)
        if d.tag in (TAG_POINTER, TAG_REFERENCE):
            inner = self.of(d)
            return None if inner is None else inner + ("*" if d.tag == TAG_POINTER else "&")
        return None

    def of(self, die: Die) -> str | None:
        """The type of a variable (or pointer type) DIE as source text, None when unknown."""
        a = die.attrs
        if AT_FUND in a:
            return FUND.get(int(a[AT_FUND]))
        if AT_UDT in a:
            return self.udt(int(a[AT_UDT]))
        if AT_MOD_FUND in a:
            block = bytes(a[AT_MOD_FUND])
            base = FUND.get(struct.unpack(">H", block[-2:])[0])
            return None if base is None else _modified(block[:-2], base)
        if AT_MOD_UDT in a:
            block = bytes(a[AT_MOD_UDT])
            at = die.block_at[AT_MOD_UDT] + len(block) - 4
            ref = self.refs.get(at, struct.unpack(">I", block[-4:])[0])
            base = self.udt(ref)
            return None if base is None else _modified(block[:-4], base)
        return None


def _modified(mods: bytes, base: str) -> str | None:
    """Apply DWARF 1 modifiers (the first is the outermost) to a base type name."""
    out = base
    for m in reversed(mods):
        if m == MOD_POINTER:
            out += "*"
        elif m == MOD_REFERENCE:
            out += "&"
        elif m in (MOD_CONST, MOD_VOLATILE):
            word = "const" if m == MOD_CONST else "volatile"
            out = f"{word} {out}" if not out.endswith(("*", "&")) else f"{out} {word}"
        else:
            return None
    return out


def local_types(elf, symbol: str, c: bool = False) -> dict[str, str]:
    """Local variable name -> its type as source text, for one function of a `-sym on` object.
    Names declared more than once (in different blocks) are left out."""
    sec = elf.section(".debug")
    if sec is None:
        return {}
    namer = TypeNamer(sec.data, debug_refs(elf), c)
    plain = symbol.split("__")[0] if not symbol.startswith("__") else symbol
    out: dict[str, str] = {}
    seen: dict[str, int] = {}
    inside = None          # end offset of the matching function's DIE tree
    for off in sorted(namer.all):
        d = namer.all[off]
        if d.tag in TAG_SUBROUTINES:
            if inside is not None and off >= inside:
                break
            hit = d.attrs.get(AT_MW_MANGLED) == symbol or (AT_MW_MANGLED not in d.attrs and d.attrs.get(AT_NAME) == plain)
            if hit and inside is None:
                sib = d.attrs.get(AT_SIBLING)
                inside = int(sib) if isinstance(sib, int) else len(sec.data)
            continue
        if inside is not None and off < inside and d.tag == TAG_LOCAL and d.attrs.get(AT_NAME):
            name = str(d.attrs[AT_NAME])
            seen[name] = seen.get(name, 0) + 1
            t = namer.of(d)
            if t is not None:
                out[name] = t
    return {k: v for k, v in out.items() if seen.get(k) == 1}


TAG_MEMBER, TAG_INHERITANCE = 0x0D, 0x1C


def class_members(elf, symbol: str) -> frozenset[str]:
    """The data members (bases' included) of the class a member function's `this` points to, from
    a `-sym on` object: what a helper written outside the class reaches as `self->name`."""
    sec = elf.section(".debug")
    if sec is None:
        return frozenset()
    refs = debug_refs(elf)
    all_ = dies(sec.data, refs)
    plain = symbol.split("__")[0] if not symbol.startswith("__") else symbol
    order = sorted(all_)
    this_type = None
    for k, off in enumerate(order):
        d = all_[off]
        if d.tag not in TAG_SUBROUTINES:
            continue
        if not (d.attrs.get(AT_MW_MANGLED) == symbol or (AT_MW_MANGLED not in d.attrs and d.attrs.get(AT_NAME) == plain)):
            continue
        for off2 in order[k + 1:]:
            x = all_[off2]
            if x.tag in TAG_SUBROUTINES:
                break
            if x.tag == TAG_PARAM and x.attrs.get(AT_NAME) == "this" and AT_MOD_UDT in x.attrs:
                block = bytes(x.attrs[AT_MOD_UDT])
                at = x.block_at[AT_MOD_UDT] + len(block) - 4
                this_type = refs.get(at, struct.unpack(">I", block[-4:])[0])
                break
        break
    out: set[str] = set()

    def walk(ref, depth=0):
        d = all_.get(ref)
        if d is None or depth > 8:
            return
        if d.tag == TAG_TYPEDEF and isinstance(d.attrs.get(AT_UDT), int):
            return walk(d.attrs[AT_UDT], depth + 1)
        end = d.attrs.get(AT_SIBLING)
        if not isinstance(end, int):
            return
        for off in order:
            if ref < off < end:
                x = all_[off]
                if x.tag == TAG_MEMBER and x.attrs.get(AT_NAME):
                    out.add(str(x.attrs[AT_NAME]))
                elif x.tag == TAG_INHERITANCE and isinstance(x.attrs.get(AT_UDT), int):
                    walk(x.attrs[AT_UDT], depth + 1)

    if this_type is not None:
        walk(this_type)
    return frozenset(n for n in out if not n.startswith("__"))
