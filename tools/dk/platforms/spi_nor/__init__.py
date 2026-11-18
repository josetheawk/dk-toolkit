"""SPI NOR flash dumps: flat full-chip reads. Builds on `raw`.

What this adds to `raw`: erased cells read 0xFF (a gap is erased flash, not padding), a
gap is at least one erase sector, and the dump's length names the density and geometry.
The part name in dk.toml is a label: the JEDEC id is not stored in the array.

dk.toml [platform] keys (all optional): part, erased (default 255), erase_span (default
4096), out (default "build/spi"), arch (default "auto").
"""

from __future__ import annotations

from ...core.config import Config
from ..raw.analysis import Profile

BASE = "raw"
COMMANDS: list = []

_DENSITIES = {128 * 1024 << n: f"{1 << n} Mbit" for n in range(9)}   # 1 Mbit .. 256 Mbit


def geometry(size: int) -> str:
    density = _DENSITIES.get(size)
    if density is None:
        return "geometry: size is not a standard SPI NOR density"
    return f"geometry: {density} SPI NOR (page 256, sector 4096, block 65536)"


def profile(cfg: Config) -> Profile:
    opts = cfg.platform_options
    notes = [geometry(cfg.original.stat().st_size)] if cfg.original.is_file() else []
    if opts.get("part"):
        notes.append(f"part: {opts['part']} (label from dk.toml, not read from the image)")
    return Profile(
        fill=(int(opts.get("erased", 0xFF)),),
        min_gap=int(opts.get("erase_span", 4096)),
        fill_name="erased",
        out=str(opts.get("out", "build/spi")),
        arch=str(opts.get("arch", "auto")),
        notes=tuple(notes),
    )


__all__ = ["BASE", "COMMANDS", "profile"]
