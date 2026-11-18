"""Any binary: a firmware image, a ROM, a flash or disk dump, an unknown file.

The starting platform when nothing more specific fits, and the base of specialised
analysis platforms (`spi_nor`). It maps the image, finds entry points and containers,
carves them with a checked manifest, and disassembles for any Capstone arch. There is no
build, so the matching-flow core commands are skipped.

dk.toml [platform] keys (all optional): fill (list of byte values, default [0, 255]),
min_gap (default 4096), out (default "build/raw"), arch (default "auto").
"""

from __future__ import annotations

from ...core.config import Config
from ...core.util import sha1_file
from . import carvecmd, discmd, mapcmd
from .analysis import Profile
from .common import out_dir

COMMANDS = [mapcmd.COMMAND, carvecmd.COMMAND, discmd.COMMAND]

# Core commands that need a ninja build, an objdiff report or units.
SKIP_CORE = ("setup", "worktree", "build", "verify-head", "progress", "floor", "next", "rm", "commit-tools", "brief")


def profile(cfg: Config) -> Profile:
    opts = cfg.platform_options
    return Profile(
        fill=tuple(int(b) for b in opts.get("fill", (0x00, 0xFF))),
        min_gap=int(opts.get("min_gap", 4096)),
        out=str(opts.get("out", "build/raw")),
        arch=str(opts.get("arch", "auto")),
    )


def status_checks(cfg: Config) -> list[tuple[bool, str, str]]:
    image_ok = cfg.original.is_file() and sha1_file(cfg.original) == cfg.sha1
    manifest = out_dir(cfg) / "manifest.txt"
    return [
        (image_ok, f"image present and verified ({cfg.rel(cfg.original)})", "map"),
        (manifest.is_file(), f"carve manifest present ({cfg.rel(manifest)})", "carve"),
    ]


def status_ready_hint(cfg: Config) -> str:
    return "dis"


__all__ = ["COMMANDS", "SKIP_CORE", "profile", "status_checks", "status_ready_hint"]
