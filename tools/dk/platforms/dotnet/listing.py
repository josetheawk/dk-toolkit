"""IL listings (ildasm-like, tokens resolved to names) and token cross-references."""

from __future__ import annotations

from collections import defaultdict

from .assembly import Assembly, token
from .il import Insn, decode
from .pe import FormatError

_ACCESS = {1: "private", 2: "famandassem", 3: "assembly", 4: "family", 5: "famorassem", 6: "public"}
# "static" comes from the signature
_METHOD_FLAGS = ((0x20, "final"), (0x40, "virtual"), (0x400, "abstract"),
                 (0x800, "specialname"), (0x2000, "pinvokeimpl"))


def method_flags(flags: int) -> str:
    words = [_ACCESS.get(flags & 7, "compilercontrolled")]
    words += [name for bit, name in _METHOD_FLAGS if flags & bit]
    return " ".join(words)


def render(asm: Assembly, insn: Insn) -> str:
    text = f"IL_{insn.offset:04x}: {insn.name}"
    if insn.kind in ("br8", "br32"):
        return f"{text} IL_{insn.operand:04x}"
    if insn.kind == "switch":
        return f"{text} ({', '.join(f'IL_{t:04x}' for t in insn.operand)})"
    if insn.kind in ("tok", "str"):
        return f"{text} {asm.token_name(insn.operand)}"
    if insn.kind:
        return f"{text} {insn.operand!r}" if insn.kind in ("r32", "r64") else f"{text} {insn.operand}"
    return text


def method_listing(asm: Assembly, method: int) -> list[str]:
    row = asm.md.row("MethodDef", method)
    lines = [f".method {method_flags(row['Flags'])} {asm.method_signature(method)}"
             f"   // token {token('MethodDef', method):#010x}, RVA {row['RVA']:#x}"]
    try:
        body = asm.body(method)
    except FormatError as e:
        return lines + [f"  // body unreadable: {e}"]
    if body is None:
        return lines + ["  // no body"]
    lines.append(f"  .maxstack {body.max_stack}")
    local_types = asm.locals(body)
    if local_types:
        lines.append("  .locals " + ("init " if body.init_locals else "") +
                     "(" + ", ".join(f"[{i}] {t}" for i, t in enumerate(local_types)) + ")")
    for c in body.clauses:
        handler = {"catch": f"catch {asm.token_name(c.extra)}", "filter": f"filter IL_{c.extra:04x}"}.get(c.kind, c.kind)
        lines.append(f"  .try IL_{c.try_start:04x} to IL_{c.try_end:04x} {handler}"
                     f" handler IL_{c.handler_start:04x} to IL_{c.handler_end:04x}")
    try:
        lines += ["  " + render(asm, insn) for insn in decode(body.code)]
    except FormatError as e:
        lines.append(f"  // decode stopped: {e}")
    return lines


def type_listing(asm: Assembly, typedef: int) -> list[str]:
    row = asm.md.row("TypeDef", typedef)
    extends = asm.type_name(row["Extends"])
    lines = [f".class {asm.typedef_name(typedef)}" + (f" extends {extends}" if extends else "")]
    for field in asm.fields_of(typedef):
        lines.append(f"  .field {asm.token_name(token('Field', field))}")
    for method in asm.methods_of(typedef):
        lines.append("")
        lines += ["  " + line for line in method_listing(asm, method)]
    return lines


def xrefs(asm: Assembly) -> tuple[dict[int, set[int]], list[str]]:
    """({token: methods whose IL uses it}, problems). Tokens include user strings (0x70xxxxxx)."""
    used: dict[int, set[int]] = defaultdict(set)
    problems = []
    for method in range(1, len(asm.md.rows("MethodDef")) + 1):
        try:
            body = asm.body(method)
            insns = decode(body.code) if body else []
        except FormatError as e:
            problems.append(f"{asm.method_name(method)}: {e}")
            continue
        for insn in insns:
            if insn.kind in ("tok", "str"):
                used[insn.operand].add(method)
                if insn.operand >> 24 == 0x2B:          # MethodSpec: also credit the generic method
                    spec = asm.md.row("MethodSpec", insn.operand & 0xFFFFFF)
                    if spec:
                        used[token(*spec["Method"])].add(method)
    return used, problems
