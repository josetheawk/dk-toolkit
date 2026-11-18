"""Disassembly of a byte range for any CPU Capstone knows, with a dependency-free ARM fallback."""

from __future__ import annotations

from .arm import disassemble_builtin

# name -> (capstone arch constant, mode constants OR-ed together)
ARCHES: dict[str, tuple[str, tuple[str, ...]]] = {
    "arm": ("CS_ARCH_ARM", ("CS_MODE_ARM",)),
    "armbe": ("CS_ARCH_ARM", ("CS_MODE_ARM", "CS_MODE_BIG_ENDIAN")),
    "thumb": ("CS_ARCH_ARM", ("CS_MODE_THUMB",)),
    "arm64": ("CS_ARCH_ARM64", ("CS_MODE_ARM",)),
    "mips": ("CS_ARCH_MIPS", ("CS_MODE_MIPS32", "CS_MODE_BIG_ENDIAN")),
    "mipsel": ("CS_ARCH_MIPS", ("CS_MODE_MIPS32", "CS_MODE_LITTLE_ENDIAN")),
    "mips64": ("CS_ARCH_MIPS", ("CS_MODE_MIPS64", "CS_MODE_BIG_ENDIAN")),
    "ppc": ("CS_ARCH_PPC", ("CS_MODE_32", "CS_MODE_BIG_ENDIAN")),
    "ppc64": ("CS_ARCH_PPC", ("CS_MODE_64", "CS_MODE_BIG_ENDIAN")),
    "x86": ("CS_ARCH_X86", ("CS_MODE_32",)),
    "x86_64": ("CS_ARCH_X86", ("CS_MODE_64",)),
    "riscv32": ("CS_ARCH_RISCV", ("CS_MODE_RISCV32", "CS_MODE_RISCVC")),
    "riscv64": ("CS_ARCH_RISCV", ("CS_MODE_RISCV64", "CS_MODE_RISCVC")),
    "sh": ("CS_ARCH_SH", ("CS_MODE_SH4",)),
    "m68k": ("CS_ARCH_M68K", ("CS_MODE_M68K_040",)),
}
_RENAMED = {"CS_ARCH_ARM64": "CS_ARCH_AARCH64"}   # Capstone 6 names


def capstone_lines(blob: bytes, va: int, arch: str) -> list[str]:
    """Linear sweep. Raises ImportError without Capstone, KeyError for an unknown arch."""
    import capstone

    arch_name, mode_names = ARCHES[arch]
    cs_arch = getattr(capstone, arch_name, None)
    if cs_arch is None:
        cs_arch = getattr(capstone, _RENAMED[arch_name])
    mode = 0
    for name in mode_names:
        mode |= getattr(capstone, name)
    md = capstone.Cs(cs_arch, mode)
    lines = []
    for insn in md.disasm(blob, va):
        op = f" {insn.op_str}" if insn.op_str else ""
        lines.append(f"{insn.address:08x}  {insn.bytes.hex()}  {insn.mnemonic}{op}")
    return lines


def disassemble(blob: bytes, va: int, arch: str, engine: str = "auto") -> tuple[list[str], str]:
    """(lines, engine used). `builtin` covers only `arm`; `auto` prefers Capstone."""
    if engine in ("auto", "capstone"):
        try:
            return capstone_lines(blob, va, arch), "capstone"
        except ImportError:
            if engine == "capstone":
                raise
    if arch != "arm":
        raise ImportError(f"the builtin engine only decodes arm; install capstone for {arch}")
    return disassemble_builtin(blob, va), "builtin"
