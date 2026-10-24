"""Derive a located reference object's full split (text + data) in the DOL.

For every relocation from a matched function into the reference object's own data, the
DOL instruction(s) at the same place are decoded to the address they really use:
lis/addi pairs (ADDR16_HA/HI + LO) or SDA21 (r13 = _SDA_BASE_, r2 = _SDA2_BASE_).
That gives one (reference offset -> DOL address) sample per use. When all samples of a
section agree on a base, the section's range is [base, base + size).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .dol import Dol
from .dtk_symbols import SymbolIndex
from .elf import Elf

R_ADDR16_LO, R_ADDR16_HI, R_ADDR16_HA, R_SDA21 = 4, 5, 6, 109
DATA_SECTIONS = (".rodata", ".data", ".bss", ".sdata", ".sbss", ".sdata2", ".sbss2")


def _s16(x: int) -> int:
    return x - 0x10000 if x & 0x8000 else x


def sda_bases(dol: Dol, init_registers: int) -> dict[int, int]:
    """r2 / r13 values set by __init_registers (lis rX,hi; ori rX,rX,lo)."""
    words = [int.from_bytes(dol.read(init_registers + 4 * i, 4), "big") for i in range(12)]
    hi: dict[int, int] = {}
    regs: dict[int, int] = {}
    for w in words:
        op, rd, ra, imm = w >> 26, (w >> 21) & 31, (w >> 16) & 31, w & 0xFFFF
        if op == 15 and ra == 0:          # lis rD, imm
            hi[rd] = imm << 16
        elif op == 24 and rd in hi:       # ori rA, rS, imm  (rS in rd field)
            regs[ra] = hi[rd] | imm
    return {r: v for r, v in regs.items() if r in (2, 13)}


@dataclass
class SectionPlacement:
    name: str
    size: int
    samples: list[tuple[int, int]] = field(default_factory=list)   # (ref offset, DOL address)

    @property
    def bases(self) -> set[int]:
        return {dol - ref for ref, dol in self.samples}


@dataclass
class Placement:
    text: tuple[int, int]
    sections: dict[str, SectionPlacement]
    problems: list[str] = field(default_factory=list)

    def splits(self, symbols: SymbolIndex) -> list[tuple[str, int, int]]:
        out = [(".text", *self.text)]
        for name, sp in sorted(self.sections.items(), key=lambda kv: DATA_SECTIONS.index(kv[0])):
            if len(sp.bases) == 1 and sp.size:
                base = next(iter(sp.bases))
                sym = symbols.at_or_before(base)
                section = sym.section if sym else name
                out.append((section, base, padded_end(symbols, section, base + sp.size)))
        return out


def padded_end(symbols: SymbolIndex, section: str, raw_end: int) -> int:
    """End a data split at the object's real end, word-aligned. Inter-TU alignment padding is
    left to the following unit: objdiff counts trailing padding inside a unit as a mismatch,
    and the linker re-creates it from the next object's alignment anyway."""
    return (raw_end + 3) & ~3


def place(dol: Dol, ref: Elf, matched: list[tuple[str, int, int]], sda: dict[int, int]) -> Placement:
    funcs = ref.functions()
    sections: dict[str, SectionPlacement] = {}
    for s in ref.sections:
        if s.name in DATA_SECTIONS and s.size:
            sections[s.name] = SectionPlacement(s.name, s.size)
    result = Placement((matched[0][1], matched[-1][1] + matched[-1][2]), sections)

    def word(address: int) -> int:
        return int.from_bytes(dol.read(address, 4), "big")

    for name, address, _ in matched:
        fn = funcs.get(name)
        if fn is None:
            continue
        pending_hi: dict[tuple[str, int], tuple[int, int]] = {}
        for r in fn.relocs:
            if r.target_section not in sections:
                continue
            ref_off = r.target_value + r.addend
            key = (r.symbol, r.addend)
            insn = word(address + (r.offset & ~3))
            if r.type in (R_ADDR16_HA, R_ADDR16_HI):
                pending_hi[key] = (r.type, insn & 0xFFFF)
            elif r.type == R_ADDR16_LO and key in pending_hi:
                kind, hi = pending_hi[key]
                lo = insn & 0xFFFF
                dol_addr = (hi << 16) + _s16(lo) if kind == R_ADDR16_HA else (hi << 16) | lo
                sections[r.target_section].samples.append((ref_off, dol_addr & 0xFFFFFFFF))
            elif r.type == R_SDA21:
                ra = (insn >> 16) & 31
                base = 0 if ra == 0 else sda.get(ra)
                if base is None:
                    result.problems.append(f"{name}+0x{r.offset:X}: SDA21 via r{ra} (unknown base)")
                    continue
                sections[r.target_section].samples.append((ref_off, (base + _s16(insn & 0xFFFF)) & 0xFFFFFFFF))
    for sp in sections.values():
        if not sp.samples:
            result.problems.append(f"{sp.name}: 0x{sp.size:X} bytes never referenced from matched code (range unknown)")
        elif len(sp.bases) > 1:
            result.problems.append(f"{sp.name}: inconsistent bases {sorted(hex(b) for b in sp.bases)} (data reordered or stripped)")
    return result


R_REL24 = 10


def relocation_names(dol: Dol, ref: Elf, matched: list[tuple[str, int, int]], sda: dict[int, int]) -> list[tuple[int, str]]:
    """Names of what matched functions call or touch, read off the DOL at the same instruction.

    A byte-identical function's `bl foo` lands on foo in the DOL too, so every named relocation
    target of a matched function names a DOL address (callees, globals, SDK tables).
    """
    funcs = ref.functions()
    out: list[tuple[int, str]] = []

    def word(address: int) -> int:
        return int.from_bytes(dol.read(address, 4), "big")

    for name, address, _ in matched:
        fn = funcs.get(name)
        if fn is None:
            continue
        pending_hi: dict[tuple[str, int], tuple[int, int]] = {}
        for r in fn.relocs:
            target = r.symbol
            usable = target and not target.startswith((".", "@")) and r.addend == 0
            at = address + (r.offset & ~3)
            insn = word(at)
            if r.type == R_REL24:
                disp = insn & 0x03FFFFFC
                if disp & 0x02000000:
                    disp -= 0x04000000
                dest = disp if insn & 2 else at + disp
                if usable:
                    out.append((dest & 0xFFFFFFFF, target))
            elif r.type in (R_ADDR16_HA, R_ADDR16_HI):
                pending_hi[(target, r.addend)] = (r.type, insn & 0xFFFF)
            elif r.type == R_ADDR16_LO and (target, r.addend) in pending_hi:
                kind, hi = pending_hi[(target, r.addend)]
                lo = insn & 0xFFFF
                dest = (hi << 16) + _s16(lo) if kind == R_ADDR16_HA else (hi << 16) | lo
                if usable:
                    out.append((dest & 0xFFFFFFFF, target))
            elif r.type == R_SDA21:
                ra = (insn >> 16) & 31
                base = 0 if ra == 0 else sda.get(ra)
                if base is not None and usable:
                    out.append(((base + _s16(insn & 0xFFFF)) & 0xFFFFFFFF, target))
    return out


def consistent_names(samples: list[tuple[int, str]]) -> dict[int, str]:
    """Keep only names that always point at one address, and addresses that always get one name."""
    by_addr: dict[int, set[str]] = defaultdict(set)
    by_name: dict[str, set[int]] = defaultdict(set)
    for addr, name in samples:
        by_addr[addr].add(name)
        by_name[name].add(addr)
    return {addr: next(iter(names)) for addr, names in sorted(by_addr.items())
            if len(names) == 1 and len(by_name[next(iter(names))]) == 1}
