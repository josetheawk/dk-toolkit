"""Explain where a built DOL differs from the original (the first_diff.py idea)."""

from __future__ import annotations

from pathlib import Path

from ...core.config import Config
from .dol import Dol
from .dtk_symbols import SymbolIndex, read_symbols


def _index(cfg: Config) -> SymbolIndex | None:
    path = cfg.platform_options.get("symbols")
    if not path or not (cfg.root / path).is_file():
        return None
    return SymbolIndex(read_symbols(cfg.root / path))


def first_diff(cfg: Config, built: Path, original: Path, limit: int = 8) -> list[str]:
    a = Dol.parse(original.read_bytes())
    b = Dol.parse(built.read_bytes())
    lines: list[str] = []

    layout_a = [(s.address, s.size) for s in a.sections]
    layout_b = [(s.address, s.size) for s in b.sections]
    if layout_a != layout_b:
        lines.append("section layout differs (address, size):")
        for sa, sb in zip(a.sections, b.sections):
            if (sa.address, sa.size) != (sb.address, sb.size):
                lines.append(f"  {sa.slot}: original 0x{sa.address:08X}+0x{sa.size:X}  built 0x{sb.address:08X}+0x{sb.size:X}")
        if len(a.sections) != len(b.sections):
            lines.append(f"  section count: original {len(a.sections)}, built {len(b.sections)}")
    if (a.bss_address, a.bss_size, a.entry) != (b.bss_address, b.bss_size, b.entry):
        lines.append(f"bss/entry differ: original bss 0x{a.bss_address:08X}+0x{a.bss_size:X} entry 0x{a.entry:08X}; "
                     f"built bss 0x{b.bss_address:08X}+0x{b.bss_size:X} entry 0x{b.entry:08X}")

    symbols = _index(cfg)
    found = 0
    for sa in a.sections:
        sb = b.section_at(sa.address)
        if sb is None:
            continue
        size = min(sa.size, sb.end - sa.address)
        da = a.read(sa.address, size)
        db = b.read(sa.address, size)
        if da == db:
            continue
        for off in range(0, size, 4):
            if da[off:off + 4] != db[off:off + 4]:
                addr = sa.address + off
                where = symbols.describe(addr) if symbols else ""
                lines.append(f"{sa.slot} 0x{addr:08X} {where:<40} original {da[off:off+4].hex()}  built {db[off:off+4].hex()}")
                found += 1
                if found >= limit:
                    lines.append(f"(stopped after {limit} differences)")
                    return lines
    if not lines:
        lines.append("section contents match; difference is in padding or the header")
    return lines
