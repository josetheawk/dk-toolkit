"""Just enough PE/COFF to reach a .NET assembly: headers, sections, data directories, RVA → offset."""

from __future__ import annotations

import struct
from dataclasses import dataclass

CLR_DIRECTORY = 14          # IMAGE_DIRECTORY_ENTRY_COM_DESCRIPTOR

MACHINES = {0x14C: "i386", 0x8664: "amd64", 0x1C4: "armnt", 0xAA64: "arm64"}
SUBSYSTEMS = {1: "native", 2: "windows-gui", 3: "windows-console", 9: "windows-ce"}


class FormatError(Exception):
    pass


@dataclass(frozen=True)
class Section:
    name: str
    rva: int
    vsize: int
    offset: int
    rawsize: int

    def holds(self, rva: int) -> bool:
        return self.rva <= rva < self.rva + max(self.vsize, self.rawsize)


@dataclass(frozen=True)
class PE:
    data: bytes
    machine: int
    timestamp: int
    characteristics: int
    pe32plus: bool
    subsystem: int
    image_base: int
    entry_rva: int
    header_size: int
    sections: tuple[Section, ...]
    directories: tuple[tuple[int, int], ...]     # (rva, size) per data directory

    @classmethod
    def parse(cls, data: bytes) -> PE:
        if len(data) < 0x40 or data[:2] != b"MZ":
            raise FormatError("not a PE file (no MZ header)")
        (lfanew,) = struct.unpack_from("<I", data, 0x3C)
        if data[lfanew:lfanew + 4] != b"PE\0\0":
            raise FormatError("not a PE file (no PE signature)")
        machine, nsections, timestamp, _, _, optsize, chars = struct.unpack_from("<HHIIIHH", data, lfanew + 4)
        opt = lfanew + 24
        (magic,) = struct.unpack_from("<H", data, opt)
        if magic not in (0x10B, 0x20B):
            raise FormatError(f"unknown optional header magic {magic:#x}")
        plus = magic == 0x20B
        (entry,) = struct.unpack_from("<I", data, opt + 16)
        image_base = struct.unpack_from("<Q", data, opt + 24)[0] if plus else struct.unpack_from("<I", data, opt + 28)[0]
        (header_size,) = struct.unpack_from("<I", data, opt + 60)
        (subsystem,) = struct.unpack_from("<H", data, opt + 68)
        ndirs_at, dirs_at = (opt + 108, opt + 112) if plus else (opt + 92, opt + 96)
        (ndirs,) = struct.unpack_from("<I", data, ndirs_at)
        dirs = tuple(struct.unpack_from("<II", data, dirs_at + 8 * i) for i in range(min(ndirs, 16)))
        sections = []
        for i in range(nsections):
            at = opt + optsize + 40 * i
            name, vsize, rva, rawsize, rawptr = struct.unpack_from("<8sIIII", data, at)
            sections.append(Section(name.rstrip(b"\0").decode("latin-1"), rva, vsize, rawptr, rawsize))
        return cls(data, machine, timestamp, chars, plus, subsystem, image_base, entry, header_size,
                   tuple(sections), dirs)

    def directory(self, index: int) -> tuple[int, int]:
        return self.directories[index] if index < len(self.directories) else (0, 0)

    def section_of(self, rva: int) -> Section | None:
        return next((s for s in self.sections if s.holds(rva)), None)

    def offset(self, rva: int) -> int:
        if rva < self.header_size:
            return rva
        section = self.section_of(rva)
        if section is None:
            raise FormatError(f"RVA {rva:#x} is outside every section")
        return section.offset + rva - section.rva

    def read(self, rva: int, size: int) -> bytes:
        at = self.offset(rva)
        return self.data[at:at + size]

    def view(self, rva: int) -> memoryview:
        """Everything from rva to the end of the file (for parsers that find their own length)."""
        return memoryview(self.data)[self.offset(rva):]
