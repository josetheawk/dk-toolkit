"""Hand-written assembly as CodeWarrior `asm` functions, from dtk's disassembly.

A function the original programmers wrote in assembly can only be reproduced as assembly.
This turns dtk's listing of it into the source form CodeWarrior compiles back to the same
bytes (the form the SDK's own sources use):

    asm void Class::name(float) {
        nofralloc
        lwz r3, var(r13)            // dtk: var@sda21(r0); r2 for .sdata2/.sbss2
        lis r4, table@ha
    L_80148718:                     // dtk: .L_80148718
        psq_st f14, 0x0(r4), 0, 0   // dtk: qr0
        crclr 6                     // dtk: cr1eq
        bdnz L_80148718
        blr
    }

Symbols it references are declared `extern "C"` under their linker names (mangled names are
valid identifiers; CodeWarrior's assembler takes no qualified names). The caller compiles the
result with the unit's flags and verifies it like any other draft.
"""

from __future__ import annotations

import re

from .demangle import demangle
from .gen import Unsupported

_LINE = re.compile(r"^/\* [0-9A-F]{8} [0-9A-F]{8}  (?:[0-9A-F]{2} ){4}\*/\t(?P<op>\S+)(?:\s+(?P<args>.*))?$")
_LABEL = re.compile(r"^\.L_(?P<addr>[0-9A-F]{8}):$")
_SDA21 = re.compile(r"(?P<sym>[A-Za-z_$][\w$]*)@sda21\(r0\)")
_SYM_RELOC = re.compile(r"(?P<sym>[A-Za-z_$@][\w$@.]*)@(?:ha|h|l|sda21)\b")
_CR_BIT = {"lt": 0, "gt": 1, "eq": 2, "so": 3, "un": 3}
_SPR = {"XER": 1, "LR": 8, "CTR": 9, "DSISR": 18, "DAR": 19, "DEC": 22, "SDR1": 25, "SRR0": 26, "SRR1": 27,
        "SPRG0": 272, "SPRG1": 273, "SPRG2": 274, "SPRG3": 275, "EAR": 282, "PVR": 287,
        **{f"GQR{n}": 912 + n for n in range(8)}, "HID2": 920, "WPAR": 921, "DMA_U": 922, "DMA_L": 923,
        "HID0": 1008, "HID1": 1009, "IABR": 1010, "DABR": 1013, "L2CR": 1017, "ICTC": 1019, "THRM1": 1020}
_SMALL = {".sdata": "r13", ".sbss": "r13", ".sdata2": "r2", ".sbss2": "r2"}


def _cr_bits(args: str) -> str:
    """cr1eq -> 6, and bare lt/gt/eq/so (cr0) -> 0..3, in condition-register logic operands."""
    def one(tok: str) -> str:
        t = tok.strip()
        m = re.fullmatch(r"cr([0-7])(lt|gt|eq|so|un)", t)
        if m:
            return str(4 * int(m[1]) + _CR_BIT[m[2]])
        return str(_CR_BIT[t]) if t in _CR_BIT else t
    return ", ".join(one(t) for t in args.split(","))


def convert_line(op: str, args: str, section_of) -> tuple[str, set[str], set[str]]:
    """One instruction in CodeWarrior asm syntax: (text, functions called, data referenced)."""
    calls: set[str] = set()
    data: set[str] = set()
    args = args or ""
    if op in ("bl", "b") and args and not args.startswith(".L_"):
        calls.add(args.strip())
    args = re.sub(r"\.L_([0-9A-F]{8})", r"L_\1", args)

    def small(sym: str) -> str:
        sec = section_of(sym)
        base = _SMALL.get(sec or "")
        if base is None:
            raise Unsupported(f"small-data reference to {sym} in {sec or 'an unknown section'}")
        data.add(sym)
        return f"{sym}({base})"
    args = _SDA21.sub(lambda m: small(m["sym"]), args)
    address = re.fullmatch(r"(?P<rd>r\d+), (?P<sym>[A-Za-z_]\w*)@sda21", args) if op == "li" else None
    if address:                          # the address of a small-data object: la rD, x(r13)
        op, args = "la", f"{address['rd']}, {small(address['sym'])}"
    if "@sda21" in args:
        raise Unsupported(f"small-data address form: {op} {args}")
    for m in _SYM_RELOC.finditer(args):
        data.add(m["sym"])
    if op.startswith("psq_"):
        args = re.sub(r"\bqr([0-7])\b", r"\1", args)
    if op.startswith("cr") and op not in ("cmpw", "cmpwi", "cmplw", "cmplwi"):
        args = _cr_bits(args)
    if op in ("mtspr", "mfspr"):
        parts = [p.strip() for p in args.split(",")]
        k = 0 if op == "mtspr" else 1
        if parts[k] in _SPR:
            parts[k] = str(_SPR[parts[k]])
        elif not parts[k].isdigit():
            raise Unsupported(f"special register {parts[k]}")
        args = ", ".join(parts)
    for sym in data | calls:
        if not re.fullmatch(r"[A-Za-z_]\w*", sym):
            raise Unsupported(f"symbol {sym} is not an identifier")
    return (f"{op} {args}".rstrip(), calls, data)


def _qualified(quals: tuple, name: str) -> str:
    return "::".join((*quals, name))


def declaration(symbol: str) -> tuple[list[str], str]:
    """(declarations needed before it, the asm function's head) for the function `symbol`."""
    fn = demangle(symbol) if "__" in symbol.lstrip("_") else None
    if fn is None:
        if not re.fullmatch(r"[A-Za-z_]\w*", symbol) or re.search(r"__(?:F|\d|Q\d|C)", symbol):
            raise Unsupported(f"cannot declare {symbol}")
        return [], f'extern "C" asm void {symbol}(void)'
    if fn.special:
        raise Unsupported(f"assembly {fn.special} (write the class instead)")
    if any(t.reg == "none" and t.text != "..." for t in fn.params):
        raise Unsupported("a parameter type it cannot spell")
    pre: list[str] = []
    classes = {c for t in fn.params for c in t.classes} - {fn.quals}
    for c in sorted(classes):
        pre.append(_scoped(c, f"struct {c[-1]};"))
    params = ", ".join(t.text for t in fn.params if t.text != "void") or "void"
    const = " const" if fn.const else ""
    if fn.quals:
        pre.append(_scoped(fn.quals, f"struct {fn.quals[-1]} {{ void {fn.name}({params}){const}; }};"))
        return pre, f"asm void {_qualified(fn.quals, fn.name)}({params}){const}"
    pre.append(f"void {fn.name}({params});")
    return pre, f"asm void {fn.name}({params})"


def _scoped(path: tuple, decl: str) -> str:
    """A declaration inside the namespaces of all but the last name of `path`."""
    for ns in reversed(path[:-1]):
        decl = f"namespace {ns} {{ {decl} }}"
    return decl


_LITERAL_TYPE = {"lfs": "float", "lfd": "double", "psq_l": "float"}


def literal_definition(sym: str, op: str, raw: bytes) -> str:
    """A compiler-pool literal the assembly reads, defined so the draft carries its value."""
    from .gen import float_literal
    ctype = _LITERAL_TYPE.get(op)
    if ctype == "float" and len(raw) == 4:
        return f'extern "C" const float {sym} = {float_literal(raw)};'
    if ctype == "double" and len(raw) == 8:
        import struct
        return f'extern "C" const double {sym} = {struct.unpack(">d", raw)[0]!r};'
    if len(raw) == 4:
        return f'extern "C" const unsigned int {sym} = 0x{int.from_bytes(raw, "big"):08X};'
    raise Unsupported(f"literal {sym} of {len(raw)} bytes")


def asm_function(symbol: str, text: str, section_of, literal=None) -> tuple[str, set[str]]:
    """(C++ source: declarations + the asm function, the literals it defines) for dtk's listing
    `text` of `symbol`. `literal(sym, size)` gives the bytes of a compiler literal (None for a
    variable). A literal the code loads is defined with its value; an object it only takes the
    address of (a table) is declared, since the unit's data defines it."""
    body: list[str] = []
    calls: set[str] = set()
    data: set[str] = set()
    first_op: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith((".fn", ".endfn", ".sym", "#", ".global", ".balign")):
            continue
        label = _LABEL.match(line)
        if label:
            body.append(f"L_{label['addr']}:")
            continue
        m = _LINE.match(raw.rstrip())
        if not m:
            raise Unsupported(f"cannot read {line[:60]}")
        insn, c, d = convert_line(m["op"], m["args"] or "", section_of)
        calls |= c
        data |= d
        for sym in d:
            first_op.setdefault(sym, m["op"])
        body.append(f"    {insn}")
    pre, head = declaration(symbol)
    if symbol in calls and not head.startswith('extern "C"'):
        raise Unsupported("calls itself by its C++ name, which CodeWarrior's assembler cannot reference")
    externs = [f'extern "C" void {c}(void);' for c in sorted(calls - {symbol})]
    defined: set[str] = set()
    for d in sorted(data - calls - {symbol}):
        size = 8 if first_op.get(d) == "lfd" else 4
        loaded = first_op.get(d, "").startswith(("lf", "lw", "psq_l"))
        raw = literal(d, size) if literal and loaded else None
        if raw is not None:
            externs.append(literal_definition(d, first_op[d], raw))
            defined.add(d)
        else:
            externs.append(f'extern "C" int {d};')
    return "\n".join(pre + externs + [f"{head} {{", "    nofralloc", *body, "}"]) + "\n", defined
