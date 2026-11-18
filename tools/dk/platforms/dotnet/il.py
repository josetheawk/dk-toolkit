"""CIL method bodies (II.25.4) and instruction decoding (Partition III)."""

from __future__ import annotations

import struct
from dataclasses import dataclass

from .pe import FormatError

# Operand kinds → byte width (switch is variable)
_WIDTH = {"": 0, "v8": 1, "v16": 2, "i8": 1, "i32": 4, "i64": 8, "r32": 4, "r64": 8, "br8": 1, "br32": 4,
          "tok": 4, "str": 4}

# "code name [operand kind]"; two-byte opcodes are FExx
_OPCODES = """\
00 nop|01 break|02 ldarg.0|03 ldarg.1|04 ldarg.2|05 ldarg.3|06 ldloc.0|07 ldloc.1|08 ldloc.2|09 ldloc.3
0A stloc.0|0B stloc.1|0C stloc.2|0D stloc.3|0E ldarg.s v8|0F ldarga.s v8|10 starg.s v8|11 ldloc.s v8
12 ldloca.s v8|13 stloc.s v8|14 ldnull|15 ldc.i4.m1|16 ldc.i4.0|17 ldc.i4.1|18 ldc.i4.2|19 ldc.i4.3
1A ldc.i4.4|1B ldc.i4.5|1C ldc.i4.6|1D ldc.i4.7|1E ldc.i4.8|1F ldc.i4.s i8|20 ldc.i4 i32|21 ldc.i8 i64
22 ldc.r4 r32|23 ldc.r8 r64|25 dup|26 pop|27 jmp tok|28 call tok|29 calli tok|2A ret|2B br.s br8
2C brfalse.s br8|2D brtrue.s br8|2E beq.s br8|2F bge.s br8|30 bgt.s br8|31 ble.s br8|32 blt.s br8
33 bne.un.s br8|34 bge.un.s br8|35 bgt.un.s br8|36 ble.un.s br8|37 blt.un.s br8|38 br br32|39 brfalse br32
3A brtrue br32|3B beq br32|3C bge br32|3D bgt br32|3E ble br32|3F blt br32|40 bne.un br32|41 bge.un br32
42 bgt.un br32|43 ble.un br32|44 blt.un br32|45 switch switch|46 ldind.i1|47 ldind.u1|48 ldind.i2
49 ldind.u2|4A ldind.i4|4B ldind.u4|4C ldind.i8|4D ldind.i|4E ldind.r4|4F ldind.r8|50 ldind.ref
51 stind.ref|52 stind.i1|53 stind.i2|54 stind.i4|55 stind.i8|56 stind.r4|57 stind.r8|58 add|59 sub|5A mul
5B div|5C div.un|5D rem|5E rem.un|5F and|60 or|61 xor|62 shl|63 shr|64 shr.un|65 neg|66 not|67 conv.i1
68 conv.i2|69 conv.i4|6A conv.i8|6B conv.r4|6C conv.r8|6D conv.u4|6E conv.u8|6F callvirt tok|70 cpobj tok
71 ldobj tok|72 ldstr str|73 newobj tok|74 castclass tok|75 isinst tok|76 conv.r.un|79 unbox tok|7A throw
7B ldfld tok|7C ldflda tok|7D stfld tok|7E ldsfld tok|7F ldsflda tok|80 stsfld tok|81 stobj tok
82 conv.ovf.i1.un|83 conv.ovf.i2.un|84 conv.ovf.i4.un|85 conv.ovf.i8.un|86 conv.ovf.u1.un|87 conv.ovf.u2.un
88 conv.ovf.u4.un|89 conv.ovf.u8.un|8A conv.ovf.i.un|8B conv.ovf.u.un|8C box tok|8D newarr tok|8E ldlen
8F ldelema tok|90 ldelem.i1|91 ldelem.u1|92 ldelem.i2|93 ldelem.u2|94 ldelem.i4|95 ldelem.u4|96 ldelem.i8
97 ldelem.i|98 ldelem.r4|99 ldelem.r8|9A ldelem.ref|9B stelem.i|9C stelem.i1|9D stelem.i2|9E stelem.i4
9F stelem.i8|A0 stelem.r4|A1 stelem.r8|A2 stelem.ref|A3 ldelem tok|A4 stelem tok|A5 unbox.any tok
B3 conv.ovf.i1|B4 conv.ovf.u1|B5 conv.ovf.i2|B6 conv.ovf.u2|B7 conv.ovf.i4|B8 conv.ovf.u4|B9 conv.ovf.i8
BA conv.ovf.u8|C2 refanyval tok|C3 ckfinite|C6 mkrefany tok|D0 ldtoken tok|D1 conv.u2|D2 conv.u1|D3 conv.i
D4 conv.ovf.i|D5 conv.ovf.u|D6 add.ovf|D7 add.ovf.un|D8 mul.ovf|D9 mul.ovf.un|DA sub.ovf|DB sub.ovf.un
DC endfinally|DD leave br32|DE leave.s br8|DF stind.i|E0 conv.u
FE00 arglist|FE01 ceq|FE02 cgt|FE03 cgt.un|FE04 clt|FE05 clt.un|FE06 ldftn tok|FE07 ldvirtftn tok
FE09 ldarg v16|FE0A ldarga v16|FE0B starg v16|FE0C ldloc v16|FE0D ldloca v16|FE0E stloc v16|FE0F localloc
FE11 endfilter|FE12 unaligned. i8|FE13 volatile.|FE14 tail.|FE15 initobj tok|FE16 constrained. tok
FE17 cpblk|FE18 initblk|FE19 no. i8|FE1A rethrow|FE1C sizeof tok|FE1D refanytype|FE1E readonly.
"""

OPCODES: dict[int, tuple[str, str]] = {}
for _entry in _OPCODES.replace("\n", "|").split("|"):
    if _entry.strip():
        _code, _name, *_kind = _entry.split()
        OPCODES[int(_code, 16)] = (_name, _kind[0] if _kind else "")

CLAUSE_KINDS = {0: "catch", 1: "filter", 2: "finally", 4: "fault"}


@dataclass(frozen=True)
class Clause:
    kind: str
    try_start: int
    try_end: int
    handler_start: int
    handler_end: int
    extra: int          # catch: type token; filter: filter start offset


@dataclass(frozen=True)
class Body:
    code: bytes
    max_stack: int
    local_sig: int      # StandAloneSig token, 0 = no locals
    init_locals: bool
    clauses: tuple[Clause, ...]


def parse_body(data: bytes | memoryview) -> Body:
    """A method body at the start of `data` (II.25.4.1-6)."""
    first = data[0]
    if first & 3 == 2:                          # tiny header: code size in the top 6 bits
        size = first >> 2
        return Body(bytes(data[1:1 + size]), 8, 0, False, ())
    if first & 3 != 3:
        raise FormatError(f"bad method header byte {first:#x}")
    flags_size, max_stack, code_size, local_sig = struct.unpack_from("<HHII", data, 0)
    header = (flags_size >> 12) * 4
    code = bytes(data[header:header + code_size])
    if len(code) != code_size:
        raise FormatError("method body truncated")
    clauses: list[Clause] = []
    more, at = bool(flags_size & 0x08), header + code_size
    while more:
        at = (at + 3) & ~3
        kind = data[at]
        if kind & 0x40:                         # fat section: 24-bit size, 24-byte clauses
            size = int.from_bytes(bytes(data[at + 1:at + 4]), "little")
            fmt, step = "<IIIIII", 24
        else:
            size = data[at + 1]
            fmt, step = "<HHBHBI", 12
        # Compilers disagree on whether DataSize counts the 4-byte section header; the runtime
        # takes DataSize // clause size, which is right either way.
        count = size // step
        if kind & 0x01:                         # exception handling table
            for i in range(count):
                flags, ts, tl, hs, hl, extra = struct.unpack_from(fmt, data, at + 4 + i * step)
                clauses.append(Clause(CLAUSE_KINDS.get(flags & 7, f"kind{flags}"), ts, ts + tl, hs, hs + hl, extra))
        more, at = bool(kind & 0x80), at + 4 + count * step
    return Body(code, max_stack, local_sig, bool(flags_size & 0x10), tuple(clauses))


@dataclass(frozen=True)
class Insn:
    offset: int
    size: int
    opcode: int                                  # FExx for two-byte opcodes
    name: str
    kind: str                                    # operand kind ("" = none)
    operand: int | float | tuple[int, ...] | None   # branch kinds: absolute target offsets


def decode(code: bytes) -> list[Insn]:
    """All instructions in a method's code; raises FormatError on an unknown opcode or truncation."""
    out, at = [], 0
    while at < len(code):
        start = at
        opcode = code[at]
        at += 1
        if opcode == 0xFE:
            if at >= len(code):
                raise FormatError(f"IL_{start:04x}: truncated two-byte opcode")
            opcode = 0xFE00 | code[at]
            at += 1
        if opcode not in OPCODES:
            raise FormatError(f"IL_{start:04x}: unknown opcode {opcode:#x}")
        name, kind = OPCODES[opcode]
        operand: int | float | tuple[int, ...] | None = None
        if kind == "switch":
            (n,) = struct.unpack_from("<I", code, at)
            rel = struct.unpack_from(f"<{n}i", code, at + 4)
            at += 4 + 4 * n
            operand = tuple(at + r for r in rel)
        elif kind:
            width = _WIDTH[kind]
            raw = code[at:at + width]
            if len(raw) != width:
                raise FormatError(f"IL_{start:04x}: {name} operand truncated")
            at += width
            if kind in ("r32", "r64"):
                (operand,) = struct.unpack("<f" if kind == "r32" else "<d", raw)
            else:
                signed = kind in ("i8", "i32", "i64", "br8", "br32")
                operand = int.from_bytes(raw, "little", signed=signed)
                if kind in ("br8", "br32"):
                    operand += at
        out.append(Insn(start, at - start, opcode, name, kind, operand))
    return out
