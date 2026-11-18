"""ARM entry points in a flat image: classic exception vectors and Cortex-M vector tables.

Detectors are structural: a table counts only when its literals put every handler on a
real word inside the image, which also gives the link address the image was built for.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from .layout import Span

_LDR_PC = 0xE59FF000
_LDR_PC_NEEDLE = b"\xf0\x9f\xe5"   # upper three bytes of `ldr pc, [pc, #imm]`, little-endian


@dataclass(frozen=True)
class ArmVectors:
    offset: int
    link_base: int             # virtual address of `offset`
    reset_file: int | None     # file offset of a PC-relative reset branch, when it lands in the image
    handlers: tuple[int, ...]  # file offsets of the other vectors' literal targets


@dataclass(frozen=True)
class CortexMVectors:
    offset: int
    stack: int
    reset: int                 # file offset, thumb bit cleared


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def branch_target(word: int, address: int) -> int | None:
    """Target of an ARM B (not BL). `address` is the instruction's address in the same units."""
    if (word & 0x0F000000) != 0x0A000000 or (word >> 28) == 0xF:
        return None
    imm = word & 0xFFFFFF
    if imm & 0x800000:
        imm -= 0x1000000
    return address + 8 + (imm << 2)


def is_ldr_pc(word: int) -> bool:
    return (word & 0xFFFFF000) == _LDR_PC and (word & 3) == 0


def _infer_link(data: bytes, table: int, literals: list[int]) -> tuple[int, tuple[int, ...]] | None:
    """Link address of `table` such that every literal lands on a word inside the image."""
    if len(literals) < 4:
        return None
    for bits in (16, 20, 24):
        link = literals[0] & ~((1 << bits) - 1)
        offsets: list[int] = []
        for literal in literals:
            delta = literal - link
            offset = table + delta
            if delta < 0 or delta > 0x02000000 or offset < 0 or offset + 4 > len(data) or offset % 4:
                break
            offsets.append(offset)
        else:
            if any(_u32(data, offset) not in (0, 0xFFFFFFFF) for offset in offsets):
                return link, tuple(offsets)
    return None


def _parse_arm_table(data: bytes, table: int) -> ArmVectors | None:
    if table < 0 or table % 4 or table + 32 > len(data):
        return None
    words = [_u32(data, table + i * 4) for i in range(8)]
    if not all(is_ldr_pc(word) for word in words[1:]):
        return None
    if not (is_ldr_pc(words[0]) or branch_target(words[0], 0) is not None):
        return None
    literals: list[int] = []
    for index, word in enumerate(words):
        if not is_ldr_pc(word):
            continue
        lit = table + index * 4 + 8 + (word & 0xFFF)
        if lit + 4 > len(data) or lit % 4:
            return None
        literals.append(_u32(data, lit))
    inferred = _infer_link(data, table, literals)
    if inferred is None:
        return None
    link, handlers = inferred
    reset = branch_target(words[0], table)
    if reset is not None and not (0 <= reset < len(data)):
        reset = None
    return ArmVectors(table, link, reset, handlers)


def find_arm_vectors(data: bytes) -> tuple[ArmVectors, ...]:
    found: dict[int, ArmVectors] = {}
    start = 0
    while (hit := data.find(_LDR_PC_NEEDLE, start)) >= 0:
        start = hit + 1
        if hit % 4 != 1:
            continue
        for back in range(8):
            table = hit - 1 - back * 4
            if table < 0:
                break
            parsed = _parse_arm_table(data, table)
            if parsed is not None:
                found[parsed.offset] = parsed
                break
    return tuple(found[key] for key in sorted(found))


def _thumb_file(word: int, size: int) -> int | None:
    if (word & 1) == 0:
        return None
    offset = word & ~1
    if offset <= 0 or offset >= size or offset % 2:
        return None
    return offset


def find_cortex_m(data: bytes, islands: tuple[Span, ...]) -> tuple[CortexMVectors, ...]:
    """Vector tables at the start of a programmed island. Thumb reset, SRAM stack."""
    found: list[CortexMVectors] = []
    size = len(data)
    for island in islands:
        off = island.offset
        if off % 4 or off + 32 > size:
            continue
        stack = _u32(data, off)
        if stack % 8 or not 0x20000000 <= stack < 0x20100000:
            continue
        reset = _thumb_file(_u32(data, off + 4), size)
        if reset is None:
            continue
        extras = sum(1 for index in range(2, 8)
                     if (word := _u32(data, off + index * 4)) == 0 or _thumb_file(word, size) is not None)
        if extras >= 4:
            found.append(CortexMVectors(off, stack, reset))
    return tuple(found)


def disassemble_builtin(blob: bytes, va: int) -> list[str]:
    """ARM-state listing. B, BL and `ldr pc, [pc, #imm]` are decoded; other words are shown as data."""
    lines: list[str] = []
    for index in range(0, len(blob) - len(blob) % 4, 4):
        word = _u32(blob, index)
        address = va + index
        target = branch_target(word, address)
        if target is not None:
            text = f"b 0x{target:08x}"
        elif (word & 0xFF000000) == 0xEB000000:
            imm = word & 0xFFFFFF
            if imm & 0x800000:
                imm -= 0x1000000
            text = f"bl 0x{address + 8 + (imm << 2):08x}"
        elif is_ldr_pc(word):
            text = f"ldr pc, [pc, #0x{word & 0xFFF:x}]"
        else:
            text = f".word 0x{word:08x}"
        lines.append(f"{address:08x}  {blob[index:index + 4].hex()}  {text}")
    return lines
