"""Minimal ELF32 big-endian relocatable-object reader (what mwcceppc and dtk emit)."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

SHT_SYMTAB = 2
SHT_RELA = 4
STT_FUNC = 2


@dataclass(frozen=True)
class ElfSection:
    index: int
    name: str
    type: int
    offset: int
    size: int
    link: int
    info: int
    data: bytes


@dataclass(frozen=True)
class ElfSymbol:
    name: str
    value: int
    size: int
    type: int
    bind: int
    shndx: int


@dataclass(frozen=True)
class Reloc:
    offset: int
    type: int
    symbol: str
    addend: int
    target_section: str = ""     # section the target symbol is defined in ("" if undefined)
    target_value: int = 0        # target symbol's value within that section


@dataclass
class Function:
    name: str
    section: str
    code: bytes
    relocs: list[Reloc] = field(default_factory=list)   # offsets relative to the function


@dataclass(frozen=True)
class Elf:
    sections: tuple[ElfSection, ...]
    symbols: tuple[ElfSymbol, ...]

    @classmethod
    def parse(cls, data: bytes) -> "Elf":
        if data[:4] != b"\x7fELF" or data[4] != 1 or data[5] != 2:
            raise ValueError("not an ELF32 big-endian file")
        shoff, = struct.unpack_from(">I", data, 0x20)
        shentsize, shnum, shstrndx = struct.unpack_from(">HHH", data, 0x2E)
        raw = []
        for i in range(shnum):
            name, typ, _flags, _addr, off, size, link, info, _align, _ent = struct.unpack_from(">10I", data, shoff + i * shentsize)
            body = b"" if typ == 8 else data[off:off + size]   # SHT_NOBITS
            raw.append((name, typ, off, size, link, info, body))
        strtab = raw[shstrndx][6]

        def cstr(table: bytes, at: int) -> str:
            return table[at:table.index(b"\0", at)].decode("latin-1")

        sections = tuple(ElfSection(i, cstr(strtab, r[0]), r[1], r[2], r[3], r[4], r[5], r[6]) for i, r in enumerate(raw))
        symbols: tuple[ElfSymbol, ...] = ()
        for s in sections:
            if s.type == SHT_SYMTAB:
                names = sections[s.link].data
                symbols = tuple(
                    ElfSymbol(cstr(names, n), v, sz, info & 0xF, info >> 4, shndx)
                    for n, v, sz, info, _other, shndx in struct.iter_unpack(">IIIBBH", s.data)
                )
        return cls(sections, symbols)

    def relocations(self, section_index: int) -> list[Reloc]:
        out = []
        for s in self.sections:
            if s.type == SHT_RELA and s.info == section_index:
                for off, info, addend in struct.iter_unpack(">IIi", s.data):
                    sym = self.symbols[info >> 8]
                    defined = 0 < sym.shndx < len(self.sections)
                    target = self.sections[sym.shndx].name if defined else ""
                    out.append(Reloc(off, info & 0xFF, sym.name or target, addend, target, sym.value if defined else 0))
        return sorted(out, key=lambda r: r.offset)

    def section(self, name: str) -> ElfSection | None:
        return next((s for s in self.sections if s.name == name), None)

    def force_active(self) -> set[str]:
        """Names of the symbols mwcc flagged force-active (`#pragma force_active on`, or
        `__declspec(force_export)`): the linker never dead-strips them. mwcc keeps per-symbol
        flags in its `.comment` section: a 0x2C-byte "CodeWarrior" header, then 8 bytes per
        symbol-table entry (alignment, visibility flags, active flags); active bit 0x08."""
        com = self.section(".comment")
        if com is None or not com.data.startswith(b"CodeWarrior") or len(com.data) < 0x2C:
            return set()
        out = set()
        for i, sym in enumerate(self.symbols):
            at = 0x2C + 8 * i + 5
            if at < len(com.data) and com.data[at] & 0x08 and sym.name:
                out.add(sym.name)
        return out

    def functions(self) -> dict[str, Function]:
        result = {}
        for sym in self.symbols:
            if sym.type != STT_FUNC or not 0 < sym.shndx < len(self.sections):
                continue
            sec = self.sections[sym.shndx]
            fn = Function(sym.name, sec.name, sec.data[sym.value:sym.value + sym.size])
            fn.relocs = [
                Reloc(r.offset - sym.value, r.type, r.symbol, r.addend, r.target_section, r.target_value)
                for r in self.relocations(sec.index)
                if sym.value <= r.offset < sym.value + sym.size
            ]
            result[sym.name] = fn
        return result
