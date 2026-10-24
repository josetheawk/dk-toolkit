"""GameCube/Wii DOL executable parsing."""

from __future__ import annotations

import struct
from dataclasses import dataclass

HEADER_SIZE = 0x100
TEXT_SLOTS = 7
DATA_SLOTS = 11


@dataclass(frozen=True)
class Section:
    slot: str        # "T0".."T6", "D0".."D10": the DOL header slot
    kind: str        # "text" or "data"
    offset: int      # file offset
    address: int     # load address
    size: int

    @property
    def end(self) -> int:
        return self.address + self.size

    def contains(self, address: int) -> bool:
        return self.address <= address < self.end


@dataclass(frozen=True)
class Dol:
    data: bytes
    sections: tuple[Section, ...]
    bss_address: int
    bss_size: int
    entry: int

    @classmethod
    def parse(cls, data: bytes) -> "Dol":
        if len(data) < HEADER_SIZE:
            raise ValueError("file too small for a DOL header")
        slots = TEXT_SLOTS + DATA_SLOTS
        offsets = struct.unpack_from(f">{slots}I", data, 0x00)
        addresses = struct.unpack_from(f">{slots}I", data, 0x48)
        sizes = struct.unpack_from(f">{slots}I", data, 0x90)
        bss_address, bss_size, entry = struct.unpack_from(">3I", data, 0xD8)
        sections = []
        for i in range(slots):
            if not sizes[i]:
                continue
            text = i < TEXT_SLOTS
            slot = f"T{i}" if text else f"D{i - TEXT_SLOTS}"
            if offsets[i] + sizes[i] > len(data):
                raise ValueError(f"section {slot} runs past end of file")
            sections.append(Section(slot, "text" if text else "data", offsets[i], addresses[i], sizes[i]))
        sections.sort(key=lambda s: s.address)
        return cls(data, tuple(sections), bss_address, bss_size, entry)

    def section_at(self, address: int) -> Section | None:
        for s in self.sections:
            if s.contains(address):
                return s
        return None

    def address_of_offset(self, offset: int) -> int | None:
        for s in self.sections:
            if s.offset <= offset < s.offset + s.size:
                return s.address + offset - s.offset
        return None

    def read(self, address: int, size: int) -> bytes:
        s = self.section_at(address)
        if s is None or address + size > s.end:
            raise ValueError(f"0x{address:08X}+0x{size:X} is not inside one section")
        start = s.offset + address - s.address
        return self.data[start:start + size]

    def words(self, section: Section) -> tuple[int, ...]:
        raw = self.data[section.offset:section.offset + section.size]
        return struct.unpack(f">{len(raw) // 4}I", raw[: len(raw) // 4 * 4])
