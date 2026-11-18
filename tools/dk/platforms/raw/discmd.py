"""dk dis: disassemble a range of the image.

The CPU comes from --arch, else from an entry point found in the same region (an ARM vector
table means arm, a Cortex-M table thumb), else from dk.toml [platform] arch. The virtual
address comes from --va, else from the region's ARM vector table, else it is the file offset.
"""

from __future__ import annotations

import argparse

from ...core.command import OK, USAGE, Command, hint
from ...core.config import Config
from .analysis import Analysis, analyze
from .common import profile_of, read_image
from .disasm import ARCHES, disassemble


def _same_island(report: Analysis, a: int, b: int) -> bool:
    return any(i.offset <= a < i.end and i.offset <= b < i.end for i in report.islands)


def _arch_and_va(report: Analysis, at: int, default_arch: str) -> tuple[str | None, int, str]:
    """(arch or None, va, how the va was chosen) for file offset `at`."""
    arm = [t for t in report.arm if t.offset <= at and _same_island(report, t.offset, at)]
    if arm:
        t = arm[-1]
        return "arm", t.link_base + (at - t.offset), f"link 0x{t.link_base:08x} of vector at 0x{t.offset:08x}"
    if any(_same_island(report, t.offset, at) for t in report.cortex_m):
        return "thumb", at, "file offset (Cortex-M region)"
    return (None if default_arch == "auto" else default_arch), at, "file offset"


def run(cfg: Config, args: argparse.Namespace) -> int:
    loaded = read_image(cfg)
    if loaded is None:
        hint("status")
        return USAGE
    data, _ = loaded
    if args.length < 1:
        print("error: --length must be >= 1")
        return USAGE
    profile = profile_of(cfg)
    report = analyze(data, profile, string_limit=0)
    at = args.at
    if at is None:
        starts = [t.offset for t in report.arm] + [t.reset for t in report.cortex_m] + [i.offset for i in report.islands]
        at = starts[0] if starts else 0
    if not 0 <= at < len(data):
        print(f"error: --at 0x{at:x} is outside the image")
        return USAGE
    arch, va, how = _arch_and_va(report, at, profile.arch)
    if args.arch != "auto":
        arch = args.arch
    if args.va is not None:
        va, how = args.va, "given"
    if arch is None:
        print(f"error: no entry point decides the CPU here; pass --arch ({', '.join(ARCHES)})")
        hint(f"dis --at 0x{at:x} --arch arm")
        return USAGE
    length = min(args.length, len(data) - at)
    try:
        lines, engine = disassemble(data[at:at + length], va, arch, args.engine)
    except ImportError as e:
        print(f"error: {e or 'capstone is not installed'}")
        hint("dis --engine builtin --arch arm" if arch == "arm" else "dis --help")
        return USAGE
    print(f"dis file 0x{at:08x} va 0x{va:08x} arch {arch} engine {engine} bytes {length} ({how})")
    for line in lines:
        print(line)
    if length != args.length:
        print(f"clipped to {length} bytes (end of image)")
    hint("map")
    return OK


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("--at", type=lambda text: int(text, 0), default=None,
                   help="file offset (default: the first entry point, else the first programmed region)")
    p.add_argument("--va", type=lambda text: int(text, 0), default=None, help="virtual address of --at (default: inferred)")
    p.add_argument("--length", type=lambda text: int(text, 0), default=0x80, help="bytes to decode (default 0x80)")
    p.add_argument("--arch", choices=("auto", *ARCHES), default="auto")
    p.add_argument("--engine", choices=("auto", "builtin", "capstone"), default="auto")


COMMAND = Command("dis", "analysis", "disassemble part of the image (arm, thumb, arm64, mips, ppc, x86, riscv, ...)", run, _configure)
