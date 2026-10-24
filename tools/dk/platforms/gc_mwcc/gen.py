"""Verified drafts for small functions.

Straight-line leaf functions (field getters/setters, constant returns, empty bodies...) are
lifted from dtk's disassembly to C++, compiled with the project's own compiler and flags, and
kept only when every instruction and relocation equals the original. Nothing is guessed:
whatever the lifter does not understand is reported with a reason.

  parse    dtk asm text -> instructions
  lift     symbolic execution over registers -> stores and a return value on typed fields
  render   C++ in the project's conventions (fields m<OFF>, Unk<ADDR> placeholder types), as
           several variants (statement order, return mode, bool returns)
  verify   compile all variants of one function in its own scratch TU and compare each with
           the original bytes: relocated fields masked, literals compared by value, globals
           by name
"""

from __future__ import annotations

import itertools
import re
import struct
from dataclasses import dataclass, field

from .demangle import demangle
from .elf import Elf

# --- parse -----------------------------------------------------------------------------------

_INSN = re.compile(r"^/\* (?P<addr>[0-9A-F]{8}) [0-9A-F]{8}  (?P<bytes>(?:[0-9A-F]{2} ){4})\*/\s+(?P<op>\S+)\s*(?P<args>.*)$")
_SYMREF = re.compile(r"^(?P<sym>[^@\s()+]+)(?:\+(?P<add>0x[0-9A-Fa-f]+|\d+))?@(?P<kind>ha|h|l|sda21)$")
_MEM = re.compile(r"^(?P<disp>[^()]+)\((?P<base>r\d+)\)$")


@dataclass(frozen=True)
class Insn:
    address: int
    word: int
    op: str
    args: tuple[str, ...]

    def reloc(self) -> tuple[str, str, int] | None:
        """(kind, symbol, addend) of the relocated operand, if any."""
        if self.op in ("bl", "b") and self.args and not self.args[0].startswith(".L"):
            return "rel24", self.args[0], 0
        for a in self.args:
            m = _MEM.match(a)
            ref = _SYMREF.match(m["disp"] if m else a)
            if ref:
                return ref["kind"], ref["sym"], int(ref["add"], 0) if ref["add"] else 0
        return None


def parse(text: str) -> list[Insn]:
    out = []
    for line in text.splitlines():
        m = _INSN.match(line.strip())
        if m:
            args = tuple(a.strip() for a in m["args"].split(",")) if m["args"].strip() else ()
            out.append(Insn(int(m["addr"], 16), int(m["bytes"].replace(" ", ""), 16), m["op"], args))
    return out


# --- expressions -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Param:          # an incoming argument register (r3-r10, f1-f8)
    reg: str


@dataclass(frozen=True)
class Const:
    value: int


@dataclass(frozen=True)
class HaPart:         # lis rD, sym@ha: half of an absolute address
    sym: str


@dataclass(frozen=True)
class GlobalObj:      # a global used as a struct (absolute addressing)
    sym: str


@dataclass(frozen=True)
class StackObj:       # a local variable: the stack region starting at this frame offset
    offset: int


@dataclass(frozen=True)
class GlobalVar:      # a small global (sda21 addressing): a location
    sym: str


@dataclass(frozen=True)
class Field:          # location: pointer base + offset
    base: object      # Param | Load | GlobalObj
    offset: int


@dataclass(frozen=True)
class Load:
    loc: object       # Field | GlobalVar
    ctype: str
    seq: int          # instruction index: identity, and when it was read


@dataclass(frozen=True)
class Literal:
    sym: str
    ctype: str
    raw: bytes


@dataclass(frozen=True)
class IndexPtr:       # base + disp + index*size: an element address, resolved by how it is used
    base: object
    index: object
    size: int
    disp: int
    raw: object = None  # the integer reading (base + offset) when nothing proves base a pointer


@dataclass(frozen=True)
class Elem:           # location: element `index` of an array of `size`-byte elements
    base: object      # the object holding the array, or (via_pointer) the pointer to the elements
    start: int        # where the array starts inside base's object (0 when via_pointer)
    index: object
    size: int
    sub: int          # offset of the accessed field inside the element
    via_pointer: bool


@dataclass(frozen=True)
class Offset:         # pointer + constant, resolved by how it is used
    base: object
    delta: int


@dataclass(frozen=True)
class AddrOf:
    loc: object       # Field | GlobalObj | GlobalVar


@dataclass(frozen=True)
class Op:
    op: str
    args: tuple
    ctype: str


@dataclass(frozen=True)
class Store:
    loc: object
    value: object
    ctype: str
    seq: int


@dataclass(frozen=True)
class Call:
    callee: str       # a symbol, or vf<slot> for a virtual call
    args: tuple       # GPR arguments (r3...) then FPR arguments (f1...); args[0] = object if virtual
    n_gpr: int
    seq: int
    virtual: bool = False
    cxx: object = None    # demangle.Function for a C++ callee (args follow its parameter order)


@dataclass(frozen=True)
class CallResult:     # what a call left in r3 (fp=False) or f1 (fp=True)
    call: Call
    fp: bool


class Unsupported(Exception):
    """The lifter does not handle this function (the message says why)."""


LOADS_X = {"lwzx": ("int", 4), "lhzx": ("unsigned short", 2), "lhax": ("short", 2), "lbzx": ("unsigned char", 1),
           "lfsx": ("float", 4), "lfdx": ("double", 8)}
STORES_X = {"stwx": ("int", 4), "sthx": ("short", 2), "stbx": ("unsigned char", 1), "stfsx": ("float", 4), "stfdx": ("double", 8)}
LOADS = {"lwz": ("int", 4), "lhz": ("unsigned short", 2), "lha": ("short", 2), "lbz": ("unsigned char", 1),
         "lfs": ("float", 4), "lfd": ("double", 8)}
STORES = {"stw": ("int", 4), "sth": ("short", 2), "stb": ("unsigned char", 1), "stfs": ("float", 4), "stfd": ("double", 8)}
SIZES = {"int": 4, "unsigned int": 4, "short": 2, "unsigned short": 2, "unsigned char": 1, "signed char": 1,
         "float": 4, "double": 8, "bool": 1}
_INT_OPS = {"add": "+", "subf": "-", "mullw": "*", "and": "&", "or": "|", "xor": "^"}
_FLOAT_OPS = {"fadds": ("+", "float"), "fsubs": ("-", "float"), "fmuls": ("*", "float"), "fdivs": ("/", "float"),
              "fadd": ("+", "double"), "fsub": ("-", "double"), "fmul": ("*", "double"), "fdiv": ("/", "double")}


@dataclass
class Lifted:
    stores: list[Store]
    regs: dict[str, object]                       # final register values
    written: set[str]
    nodes: dict[object, dict[int, str]] = field(default_factory=dict)   # struct base -> offset -> ctype
    embedded: dict[object, set[int]] = field(default_factory=dict)      # struct base -> &member offsets
    small_globals: dict[str, str] = field(default_factory=dict)         # sym -> ctype
    global_objs: set[str] = field(default_factory=set)
    int_params: dict[str, str] = field(default_factory=dict)            # reg -> ctype when used as a value
    calls: list[Call] = field(default_factory=list)
    tree: list = field(default_factory=list)                          # Store | Call | If | Return
    vptrs: dict = field(default_factory=dict)                         # object -> vtable pointer offset
    alias: dict = field(default_factory=dict)                         # load -> the load that names its type
    bound: dict = field(default_factory=dict)                         # object -> qualified class (tuple)
    expected: dict = field(default_factory=dict)                      # expression -> type a C++ callee wants
    notes: list = field(default_factory=list)                         # caveats printed with the draft
    base_call: object = None                                          # constructor mode: the base constructor call
    locals: dict = field(default_factory=dict)                        # frame offset -> size of each local used
    arrays: dict = field(default_factory=dict)                        # (base, start, via_pointer) -> {"size", "fields"}
    unsigned_results: set = field(default_factory=set)                # call results compared unsigned
    ret: object | None = None
    ret_fp: bool = False
    void_first: bool = False


def is_auto(name: str) -> bool:
    return bool(re.fullmatch(r"(?:fn|lbl)_[0-9A-F]{8}", name))


def _imm(text: str) -> int:
    v = int(text, 0)
    return v


def _s32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v & 0x80000000 else v


def thunk_kind(insns: list[Insn]) -> str | None:
    """Compiler-generated adjustor thunks: they come from the class layout, never from source."""
    ops = [i.op for i in insns]
    if len(insns) == 2 and ops[0] in ("subi", "addi") and insns[0].args[:2] == ("r3", "r3") and ops[1] == "b":
        return "this-adjusting thunk (multiple inheritance)"
    if ops[:3] == ["li", "lwzx", "add"] and ops[-1] == "b" and insns[0].args[0] == "r11":
        return "virtual-base thunk"
    return None


_REG = re.compile(r"\b([rf])(\d{1,2})\b")
_VOLATILE = [f"r{n}" for n in (0, *range(3, 13))] + [f"f{n}" for n in range(0, 14)]
_FRAME_OPS = ("stwu", "mflr", "mtlr")


def strip_frame(insns: list[Insn]) -> list[Insn]:
    """The body without the stack frame: stwu/addi r1, LR save/restore and the saves/restores
    of callee-saved registers. Any other use of the stack (locals, varargs) is refused."""
    if not insns or insns[0].op != "stwu":
        return insns
    if insns[0].args[0] != "r1":
        raise Unsupported("unusual frame")
    frame = _MEM.match(insns[0].args[1])
    frame_size = -int(frame["disp"], 0) if frame else 0
    body = []
    # register save/restore helpers: addi r11, r1, N; bl _savegpr_NN (or _restgpr/_savefpr/_restfpr)
    helper = re.compile(r"^_(?:save|rest)(?:gpr|fpr)_\d+$")
    skip = set()
    for i in range(1, len(insns) - 1):
        if insns[i].op == "addi" and insns[i].args[:2] == ("r11", "r1") and insns[i + 1].op == "bl" \
                and helper.match(insns[i + 1].args[0]):
            skip.update((i, i + 1))
    for i, insn in enumerate(insns[1:], 1):
        if i in skip:
            continue
        op, a = insn.op, insn.args
        mem = _MEM.match(a[1]) if len(a) > 1 else None
        on_stack = bool(mem and mem["base"] == "r1")
        reg = _REG.fullmatch(a[0]) if a else None
        lr_slot = a[0] == "r0" and mem is not None and int(mem["disp"], 0) == frame_size + 4 \
            if mem and re.fullmatch(r"-?(?:0x[0-9A-Fa-f]+|\d+)", mem["disp"]) else False
        saved = bool(reg and int(reg[2]) >= 14) or lr_slot
        if op in ("mflr", "mtlr") and a[0] == "r0":
            continue
        if op == "addi" and a[:2] == ("r1", "r1"):
            continue
        if on_stack and op in ("stw", "lwz", "stfd", "lfd", "stmw", "lmw") and saved:
            continue
        if op in ("psq_st", "psq_l") and on_stack and saved:
            continue
        if on_stack and (op in LOADS or op in STORES):
            body.append(insn)                   # a local or a conversion scratch slot (the executor decides)
            continue
        if op == "addi" and len(a) == 3 and a[1] == "r1" and a[0] != "r1":
            body.append(insn)                   # the address of a local
            continue
        if on_stack or any(x == "r1" or x.endswith("(r1)") for x in a):
            raise Unsupported("stack locals or arguments in memory")
        body.append(insn)
    return body


def frame_layout(insns: list[Insn]) -> tuple[int, int]:
    """(frame size, lowest offset of the register save area) of a function with a frame."""
    if not insns or insns[0].op != "stwu":
        return 0, 0
    m = _MEM.match(insns[0].args[1])
    size = -int(m["disp"], 0) if m else 0
    low = size
    for insn in insns:
        a = insn.args
        mem = _MEM.match(a[1]) if len(a) > 1 else None
        reg = _REG.fullmatch(a[0]) if a else None
        if mem and mem["base"] == "r1" and reg and int(reg[2]) >= 14 and insn.op in ("stw", "stfd", "stmw", "psq_st"):
            if re.fullmatch(r"-?(?:0x[0-9A-Fa-f]+|\d+)", mem["disp"]):
                low = min(low, int(mem["disp"], 0))
        h = re.fullmatch(r"_save(gpr|fpr)_(\d+)", a[0]) if insn.op == "bl" and a else None
        if h:
            low = min(low, size - (4 if h[1] == "gpr" else 8) * (32 - int(h[2])) * (2 if h[1] == "fpr" else 1))
    return size, low


def stack_plan(body: list[Insn], save_low: int) -> tuple[set, dict]:
    """(conversion scratch offsets, local regions {start: size}) of a frame. A region starts at
    each address-taken offset (addi rN, r1, off) and at each directly accessed offset outside
    one; it runs to the next start or to the register save area."""
    scratch: set = set()
    for i in body:
        mem = _MEM.match(i.args[1]) if len(i.args) > 1 else None
        if mem and mem["base"] == "r1" and i.op in ("lfd", "stfd"):
            off = int(mem["disp"], 0)
            scratch |= {off, off + 4}
    starts: set = set()
    for i in body:
        if i.op == "addi" and len(i.args) == 3 and i.args[1] == "r1":
            starts.add(_imm(i.args[2]))
    addressed = sorted(starts)
    for i in body:
        mem = _MEM.match(i.args[1]) if len(i.args) > 1 else None
        if mem and mem["base"] == "r1" and (i.op in LOADS or i.op in STORES):
            off = int(mem["disp"], 0)
            if off in scratch:
                continue
            inside = [a for a in addressed if a <= off]
            if not inside or off >= _region_end(inside[-1], addressed, save_low):
                starts.add(off)
    ordered = sorted(starts)
    regions = {s0: _region_end(s0, ordered, save_low) - s0 for s0 in ordered}
    if any(size <= 0 for size in regions.values()) or any(s0 in scratch for s0 in regions):
        raise Unsupported("stack layout the lifter cannot model")
    return scratch, regions


def _region_end(start: int, starts: list, save_low: int) -> int:
    later = [x for x in starts if x > start]
    return min(later) if later else save_low


def _reads_writes(insn: Insn, signature) -> tuple[list[str], list[str]]:
    """Registers an instruction reads and writes (enough to infer parameters)."""
    op, a = insn.op, insn.args
    regs = lambda xs: [f"{m[1]}{m[2]}" for x in xs for m in _REG.finditer(x)]
    if op in ("bl", "b"):
        sig = signature(a[0]) if signature and a else None
        n_gpr, n_fpr = sig or (0, 0)
        return [f"r{n}" for n in range(3, 3 + n_gpr)] + [f"f{n}" for n in range(1, 1 + n_fpr)], []
    if op.startswith("st") or op.startswith("cmp") or op.startswith("mt") or op.startswith("psq_st") or op.startswith("b"):
        return regs(a), []
    return regs(a[1:]), regs(a[:1])


def infer_signature(insns: list[Insn], signature=None) -> tuple[int, int]:
    """(GPR, FPR) parameter counts: the highest argument register read before it is written,
    scanning up to the first call (whose own parameters count as reads)."""
    written: set[str] = set()
    top_r = top_f = 0
    for insn in insns:
        reads, writes = _reads_writes(insn, signature)
        for r in reads:
            if r in written:
                continue
            n = int(r[1:])
            if r[0] == "r" and 3 <= n <= 10:
                top_r = max(top_r, n - 2)
            elif r[0] == "f" and 1 <= n <= 8:
                top_f = max(top_f, n)
        written.update(writes)
        if insn.op in ("bl", "b", "bctrl", "blr"):
            break
    return top_r, top_f


@dataclass(frozen=True)
class Phi:            # a value that depends on which arm of an if ran
    cond: object
    then: object
    other: object


@dataclass
class If:
    cond: object      # the source condition: `then` runs when it holds
    then: list
    other: list
    seq: int


@dataclass
class Return:
    value: object | None
    seq: int


@dataclass(frozen=True)
class VtStore:          # constructor mode: this->vptr = &__vt__X
    vtable: str
    offset: int
    seq: int


_BRANCH = re.compile(r"^b(eq|ne|lt|le|gt|ge)(-|\+)?$")
_CMP_OF = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}
_NEGATE = {"==": "!=", "!=": "==", "<": ">=", ">=": "<", ">": "<=", "<=": ">"}


@dataclass
class _State:
    regs: dict
    written: set
    clobbered: set
    stored: set
    cr: dict          # cr field -> (signed | unsigned | float, a, b)
    ctr: object = None
    fresh: set = field(default_factory=set)     # registers written since the last call
    stack: dict = field(default_factory=dict)   # frame offset -> value (conversion scratch slots)
    when: dict = field(default_factory=dict)    # register -> write counter (which was written last)

    def copy(self) -> "_State":
        return _State(dict(self.regs), set(self.written), set(self.clobbered), set(self.stored), dict(self.cr),
                      self.ctr, set(self.fresh), dict(self.stack), dict(self.when))


def _target(insn: Insn) -> int | None:
    """Address a local branch jumps to (.L_<addr> labels), or None."""
    arg = insn.args[-1] if insn.args else ""
    return int(arg[3:], 16) if arg.startswith(".L_") else None


class _Executor:
    def __init__(self, literal, this_is_object: bool, signature, vsignature=None, ctor_mode: bool = False):
        self.literal, self.this_is_object, self.signature = literal, this_is_object, signature
        self.ctor_mode = ctor_mode          # vtable stores and the base constructor call are markers
        self.vsignature = vsignature        # slot -> (GPR, FPR) for virtual calls on `this`
        self.out = Lifted([], {}, set())
        self.as_base: set[object] = set()
        self.data_loads: set[Load] = set()  # loads of data (not vtable pointers or slots)
        self.scratch: set = set()           # stack offsets used by conversion idioms
        self.raw_bases: set = set()         # bases whose element addresses were used as plain integers
        self.regions: dict = {}             # local variables: frame offset -> size

    # helpers -----------------------------------------------------------------------------

    def get(self, st: _State, reg: str):
        if reg not in st.regs:
            raise Unsupported(f"uses {reg} after a call clobbered it" if reg in st.clobbered else f"reads {reg} before writing it")
        value = st.regs[reg]
        if isinstance(value, Phi):
            raise Unsupported("value merged from two branches used later")
        if isinstance(value, tuple):
            raise Unsupported("a half-built int->float conversion used")
        return value

    def put(self, st: _State, reg: str, value) -> None:
        st.regs[reg] = value
        st.written.add(reg)
        st.fresh.add(reg)
        st.when[reg] = max(st.when.values(), default=0) + 1

    def node(self, base, offset: int, ctype: str) -> None:
        fields = self.out.nodes.setdefault(base, {})
        old = fields.get(offset)
        if old and old != ctype:
            pair = {old, ctype}
            # sth says "short", lhz says "unsigned short": the load knows the signedness;
            # an unsigned compare of a word makes it unsigned
            if pair == {"short", "unsigned short"}:
                fields[offset] = "unsigned short"
            elif pair == {"int", "unsigned int"}:
                fields[offset] = "unsigned int"
            else:
                raise Unsupported(f"field +0x{offset:X} accessed as {old} and {ctype}")
        else:
            fields[offset] = ctype

    def scaled(self, v, byte_ok: bool = False):
        """(index, element size, extra displacement) for i<<k, i*n, (i*n)+c (or a plain int
        index for byte arrays)."""
        if isinstance(v, Op) and v.op == "+" and isinstance(v.args[1], Const):
            inner = self.scaled(v.args[0])
            if inner:
                return inner[0], inner[1], inner[2] + v.args[1].value
        if isinstance(v, Op) and v.op == "<<" and isinstance(v.args[1], Const) and 0 < v.args[1].value < 8:
            return v.args[0], 1 << v.args[1].value, 0
        if isinstance(v, Op) and v.op == "*" and isinstance(v.args[1], Const) and v.args[1].value > 1:
            return v.args[0], v.args[1].value, 0
        if byte_ok and (isinstance(v, (Load, CallResult)) or isinstance(v, Param) and v not in self.as_base):
            return v, 1, 0
        return None

    def index_ptr(self, p, k, byte_ok: bool = False):
        """p + k as an element address, if p is a pointer and k a scaled index."""
        sc = self.scaled(k, byte_ok)
        if sc is None:
            return None
        index, size, disp = sc
        if isinstance(p, Offset):
            p, disp = p.base, disp + p.delta
        if isinstance(p, AddrOf) and isinstance(p.loc, Field):
            p, disp = p.loc.base, disp + p.loc.offset
            if isinstance(p, (Param, Load, CallResult)):
                return IndexPtr(("embedded-in", p), index, size, disp)
            return IndexPtr(("embedded-in", p), index, size, disp)
        if isinstance(p, AddrOf) and isinstance(p.loc, GlobalObj):
            self.out.global_objs.add(p.loc.sym)
            return IndexPtr(("embedded-in", p.loc), index, size, disp)
        if isinstance(p, Param) and p.reg.startswith("r") and p.reg not in self.out.int_params \
                or isinstance(p, Load) and p.ctype in ("int", "unsigned int") or isinstance(p, CallResult) and not p.fp:
            if isinstance(index, Param) and index.reg.startswith("r"):
                self.out.int_params.setdefault(index.reg, "int")
            return IndexPtr(p, index, size, disp, Op("+", (p, k), "int") if not disp else None)
        return None

    def element(self, ix: IndexPtr, disp: int, width: int) -> Elem:
        """The element location an access at disp(ix) reads or writes."""
        d = ix.disp + disp
        base = ix.base
        holder = isinstance(base, tuple)
        container = base[1] if holder else base
        if not holder and isinstance(container, Param) and container.reg == "r3" and self.this_is_object:
            holder = True                       # this->mS[i]: an array inside the object
        if not holder and 0 <= d < ix.size:
            start, sub_off, via = 0, d, True    # p[i] (.mSUB)
        else:
            sub_off = 0 if width == ix.size else d % ix.size
            start, via = d - sub_off, False
            if start < 0:
                raise Unsupported("element before its array")
        if isinstance(container, Param) and container.reg.startswith("r"):
            if container.reg in self.out.int_params:
                raise Unsupported(f"{container.reg} is used both as a number and as a pointer")
            self.as_base.add(container)
        elif isinstance(container, (Load, CallResult)):
            self.as_base.add(container)
        key = (container, start, via)
        arr = self.out.arrays.setdefault(key, {"size": ix.size, "fields": {}})
        if arr["size"] != ix.size:
            raise Unsupported("one array indexed with two element sizes")
        self.out.notes.append("array bounds and element layout are one possible reading of base + offset + i*size") \
            if "array bounds" not in " ".join(self.out.notes) else None
        if not via:
            self.out.nodes.setdefault(container, {})
        return Elem(container, start, ix.index, ix.size, sub_off, via)

    def pointer(self, value, disp: int) -> Field:
        """The location `disp(value)`, where value must be a pointer."""
        out = self.out
        if isinstance(value, IndexPtr):
            return value                        # resolved by the caller with the access width
        if isinstance(value, Param) and value.reg.startswith("r"):
            if value.reg in out.int_params:
                raise Unsupported(f"{value.reg} is used both as a number and as a pointer")
            self.as_base.add(value)
            return Field(value, disp)
        if isinstance(value, Load) and value.ctype in ("int", "unsigned int") or isinstance(value, CallResult) and not value.fp:
            self.as_base.add(value)
            return Field(value, disp)
        if isinstance(value, AddrOf) and isinstance(value.loc, GlobalObj):
            return Field(value.loc, disp)
        if isinstance(value, AddrOf) and isinstance(value.loc, Field) and isinstance(value.loc.base, StackObj):
            return Field(value.loc.base, value.loc.offset + disp)
        if isinstance(value, AddrOf) and isinstance(value.loc, Field):
            return Field(value.loc.base, value.loc.offset + disp)
        if isinstance(value, Offset):
            return self.pointer(value.base, value.delta + disp)
        raise Unsupported("memory access through a computed pointer")

    def location(self, st: _State, arg: str, size: int):
        m = _MEM.match(arg)
        if not m:
            raise Unsupported(f"memory operand {arg!r}")
        ref = _SYMREF.match(m["disp"])
        if ref and ref["kind"] == "sda21":
            raw = self.literal(ref["sym"], size)
            return Literal(ref["sym"], "", raw) if raw is not None else GlobalVar(ref["sym"])
        if ref and ref["kind"] == "l":
            base = self.get(st, m["base"])
            if not (isinstance(base, HaPart) and base.sym == ref["sym"]):
                raise Unsupported("@l access without its @ha half")
            raw = self.literal(ref["sym"], size)
            if raw is not None:
                return Literal(ref["sym"], "", raw)
            self.out.global_objs.add(ref["sym"])
            return Field(GlobalObj(ref["sym"]), 0)
        if ref:
            raise Unsupported(f"relocation {ref['kind']} in a memory operand")
        if m["base"] == "r1":
            return self.local_field(_imm(m["disp"]))
        base = self.get(st, m["base"])
        if isinstance(base, IndexPtr):
            return self.element(base, _imm(m["disp"]), size)
        return self.pointer(base, _imm(m["disp"]))

    def elem_address(self, ix: IndexPtr) -> AddrOf:
        """&element for an element address that is passed or returned (not dereferenced)."""
        loc = self.element(ix, 0, ix.size)
        if loc.sub:                                     # inside an element: a member of unknown type
            self.out.arrays[(loc.base, loc.start, loc.via_pointer)]["fields"].setdefault(loc.sub, "char")
        return AddrOf(loc)

    def elem_field(self, loc: Elem, ctype: str) -> None:
        fields = self.out.arrays[(loc.base, loc.start, loc.via_pointer)]["fields"]
        old = fields.get(loc.sub)
        if old and old != ctype and {old, ctype} not in ({"short", "unsigned short"}, {"int", "unsigned int"}):
            raise Unsupported(f"array element field accessed as {old} and {ctype}")
        fields[loc.sub] = ctype if not old or ctype.startswith("unsigned") else old

    def local_field(self, off: int) -> Field:
        start = max((r for r in self.regions if r <= off), default=None)
        if start is None or off >= start + self.regions[start]:
            raise Unsupported("a stack access outside the planned locals")
        self.out.locals[start] = self.regions[start]
        self.out.nodes.setdefault(StackObj(start), {})
        return Field(StackObj(start), off - start)

    def as_value(self, value):
        """A register value used as a number: params become int params."""
        if isinstance(value, IndexPtr) and value.raw is not None and not isinstance(value.base, tuple) \
                and value.base not in self.as_base and not (isinstance(value.base, Param) and value.base.reg == "r3"
                                                            and self.this_is_object):
            self.raw_bases.add(value.base)
            value = value.raw                   # nothing made it a pointer: plain integer arithmetic
            for x in _walk(value):
                if isinstance(x, Param) and x.reg.startswith("r"):
                    self.out.int_params.setdefault(x.reg, "int")
        if isinstance(value, Param) and value.reg.startswith("r"):
            if value in self.as_base:
                raise Unsupported(f"{value.reg} is used both as a number and as a pointer")
            self.out.int_params.setdefault(value.reg, "int")
        if isinstance(value, (Offset, AddrOf, HaPart, IndexPtr)):
            raise Unsupported("arithmetic on a pointer (casts or pointer math)")
        return value

    def mark_unsigned(self, value) -> None:
        """An unsigned compare: word-sized values of unknown type are unsigned (or pointers)."""
        if isinstance(value, Load) and isinstance(value.loc, Field) and value.ctype == "int" and value not in self.as_base:
            self.node(value.loc.base, value.loc.offset, "unsigned int")
        elif isinstance(value, Param) and value.reg.startswith("r") and value not in self.as_base:
            self.out.int_params[value.reg] = "unsigned int"
        elif isinstance(value, CallResult) and not value.fp:
            self.out.unsigned_results.add(value)

    # one instruction ---------------------------------------------------------------------

    def step(self, st: _State, seq: int, insn: Insn, stmts: list) -> None:
        out = self.out
        op, a = insn.op, insn.args
        get = lambda reg: self.get(st, reg)
        put = lambda reg, v: self.put(st, reg, v)
        if (r := insn.reloc()) and r[2]:
            raise Unsupported("symbol + offset operand")
        dotted = op.endswith(".") and op != "andi."
        if dotted:
            op = op[:-1]
        mem = _MEM.match(a[1]) if len(a) > 1 else None
        if mem and mem["base"] == "r1" and int(mem["disp"], 0) in self.scratch:
            return self.stack_slot(st, op, a, int(mem["disp"], 0))
        if op == "addi" and len(a) == 3 and a[1] == "r1":
            put(a[0], AddrOf(self.local_field(_imm(a[2]))))
            return
        if op in LOADS_X or op in STORES_X:
            ctype, size = (LOADS_X.get(op) or STORES_X[op])
            x, y = get(a[1]), get(a[2])
            ix = self.index_ptr(x, y, size == 1) or self.index_ptr(y, x, size == 1)
            if ix is None:
                raise Unsupported(f"{op} that is not array indexing")
            loc = self.element(ix, 0, size)
            if op in LOADS_X:
                self.elem_field(loc, ctype)
                load = Load(loc, ctype, seq)
                self.data_loads.add(load)
                put(a[0], load)
            else:
                value = get(a[0])
                if isinstance(value, (Offset, AddrOf, HaPart, IndexPtr)):
                    raise Unsupported("stores a pointer value into an array")
                self.elem_field(loc, ctype)
                st.stored.add(loc)
                store = Store(loc, value, ctype, seq)
                out.stores.append(store)
                stmts.append(store)
            return
        if op in LOADS:
            ctype, size = LOADS[op]
            loc = self.location(st, a[1], size)
            if isinstance(loc, Literal):
                put(a[0], Literal(loc.sym, ctype, loc.raw))
                return
            if loc in st.stored and not (isinstance(loc, Field) and isinstance(loc.base, StackObj)):
                raise Unsupported("reads a location it wrote")
            if isinstance(loc, Field):
                self.node(loc.base, loc.offset, ctype)
            elif isinstance(loc, Elem):
                self.elem_field(loc, ctype)
            else:
                out.small_globals[loc.sym] = ctype
            load = Load(loc, ctype, seq)
            self.data_loads.add(load)
            put(a[0], load)
        elif op in STORES:
            ctype, size = STORES[op]
            loc = self.location(st, a[1], size)
            if isinstance(loc, Literal):
                raise Unsupported("store into a literal")
            value = get(a[0])
            if (self.ctor_mode and isinstance(value, AddrOf) and isinstance(value.loc, GlobalObj)
                    and value.loc.sym.startswith("__vt__") and isinstance(loc, Field) and loc.base == Param("r3")):
                out.global_objs.discard(value.loc.sym)
                stmts.append(VtStore(value.loc.sym, loc.offset, seq))
                return
            if isinstance(value, (Offset, AddrOf, HaPart)):
                raise Unsupported("stores a pointer value (constructor or pointer field)")
            if isinstance(value, Param) and value.reg.startswith("r"):
                if value in self.as_base:
                    raise Unsupported(f"{value.reg} is used both as a value and as a pointer")
                out.int_params[value.reg] = ctype if ctype != "int" else out.int_params.get(value.reg, "int")
            if isinstance(value, Param) and value.reg.startswith("f") and ctype == "double":
                out.int_params[value.reg] = "double"
            if isinstance(value, IndexPtr):
                value = self.as_value(value)
                if isinstance(value, IndexPtr):
                    raise Unsupported("stores an element address")
            if isinstance(loc, Field):
                self.node(loc.base, loc.offset, ctype)
            elif isinstance(loc, Elem):
                self.elem_field(loc, ctype)
            else:
                out.small_globals[loc.sym] = ctype
            st.stored.add(loc)
            store = Store(loc, value, ctype, seq)
            out.stores.append(store)
            stmts.append(store)
        elif op == "li":
            ref = _SYMREF.match(a[1])
            if ref:
                if ref["kind"] != "sda21":
                    raise Unsupported(f"li with {ref['kind']} relocation")
                out.small_globals.setdefault(ref["sym"], "int")
                put(a[0], AddrOf(GlobalVar(ref["sym"])))
            else:
                put(a[0], Const(_s32(_imm(a[1]))))
        elif op == "lis":
            ref = _SYMREF.match(a[1])
            if ref:
                if ref["kind"] != "ha":
                    raise Unsupported(f"lis with {ref['kind']} relocation")
                put(a[0], HaPart(ref["sym"]))
            else:
                put(a[0], Const(_s32(_imm(a[1]) << 16)))
        elif op in ("addi", "subi"):
            src = get(a[1])
            ref = _SYMREF.match(a[2])
            if ref:
                if op != "addi" or ref["kind"] != "l" or not (isinstance(src, HaPart) and src.sym == ref["sym"]):
                    raise Unsupported("addi relocation without its @ha half")
                out.global_objs.add(ref["sym"])
                put(a[0], AddrOf(GlobalObj(ref["sym"])))
                return
            k = _imm(a[2]) * (-1 if op == "subi" else 1)
            if isinstance(src, IndexPtr):
                raw = Op("+", (src.raw, Const(k)), "int") if src.raw is not None else None
                put(a[0], IndexPtr(src.base, src.index, src.size, src.disp + k, raw))
                return
            if isinstance(src, Const):
                put(a[0], Const(_s32(src.value + k)))
            elif isinstance(src, (Param, Load, AddrOf, Offset)) and not (isinstance(src, Param) and src.reg in out.int_params):
                put(a[0], Offset(src.base, src.delta + k) if isinstance(src, Offset) else Offset(src, k))
            else:
                put(a[0], Op("+", (self.as_value(src), Const(k)), "int"))
        elif op in ("mr", "fmr"):
            put(a[0], get(a[1]))
        elif op == "add" and (ix := self.index_ptr(get(a[1]), get(a[2])) or self.index_ptr(get(a[2]), get(a[1]))):
            put(a[0], ix)
        elif op in _INT_OPS:
            x, y = self.as_value(get(a[1])), self.as_value(get(a[2]))
            put(a[0], Op(_INT_OPS[op], (y, x) if op == "subf" else (x, y), "int"))
        elif op in ("mulli", "ori", "xori", "andi."):
            sym = {"mulli": "*", "ori": "|", "xori": "^", "andi.": "&"}[op]
            put(a[0], Op(sym, (self.as_value(get(a[1])), Const(_imm(a[2]))), "int"))
            if op == "andi.":
                st.cr["cr0"] = ("signed", st.regs[a[0]], Const(0))
        elif op == "subfic":
            put(a[0], Op("-", (Const(_imm(a[2])), self.as_value(get(a[1]))), "int"))
        elif op in ("slwi", "srwi", "srawi"):
            sym = {"slwi": "<<", "srwi": ">>u", "srawi": ">>"}[op]
            put(a[0], Op(sym, (self.as_value(get(a[1])), Const(_imm(a[2]))), "int"))
        elif op == "clrlwi":
            n = _imm(a[2])
            put(a[0], Op("&", (self.as_value(get(a[1])), Const((1 << (32 - n)) - 1)), "int"))
        elif op in ("neg", "cntlzw", "extsb", "extsh"):
            put(a[0], Op(op, (self.as_value(get(a[1])),), "int"))
        elif op in _FLOAT_OPS:
            sym, ctype = _FLOAT_OPS[op]
            x, y = st.regs.get(a[1]), st.regs.get(a[2])
            if sym == "-" and self.finish_conversion(st, a[0], x, y, ctype):
                return
            if isinstance(x, tuple) or isinstance(y, tuple):
                raise Unsupported("a half-built conversion used")
            put(a[0], Op(sym, (get(a[1]), get(a[2])), ctype))
        elif op == "fctiwz":
            put(a[0], Op("fctiwz", (get(a[1]),), "double"))
        elif op == "xoris":
            put(a[0], Op("^", (self.as_value(get(a[1])), Const(_s32(_imm(a[2]) << 16))), "int"))
        elif op in ("fneg", "fabs"):
            put(a[0], Op(op, (get(a[1]),), "float"))
        elif op in ("cmpwi", "cmplwi", "cmpw", "cmplw"):
            field_, ops = ("cr0", a) if not a[0].startswith("cr") else (a[0], a[1:])
            x = get(ops[0])
            y = Const(_imm(ops[1])) if op.endswith("i") else get(ops[1])
            kind = "unsigned" if op.startswith("cmpl") else "signed"
            if kind == "unsigned":
                self.mark_unsigned(x)
                self.mark_unsigned(y)
            for v in (x, y):
                if not (isinstance(v, (Param, Load, CallResult)) and v in self.as_base):
                    self.as_value(v) if isinstance(v, Param) else None
            st.cr[field_] = (kind, x, y)
        elif op in ("fcmpu", "fcmpo"):
            st.cr[a[0]] = ("float", get(a[1]), get(a[2]))
        elif op == "bl":
            self.call(st, seq, a[0], stmts)
        elif op == "mtctr":
            st.ctr = get(a[0])
        elif op == "bctrl":
            self.virtual_call(st, seq, stmts)
        else:
            raise Unsupported(f"instruction {insn.op}")
        if dotted:
            st.cr["cr0"] = ("signed", st.regs[a[0]], Const(0))

    _I2F_SIGNED = bytes.fromhex("4330000080000000")
    _I2F_UNSIGNED = bytes.fromhex("4330000000000000")

    def stack_slot(self, st: _State, op: str, a: tuple, off: int) -> None:
        """CodeWarrior's int<->float conversions through a scratch double on the stack:
        int->float  stw 0x43300000, off; stw x^0x80000000 (signed) or x, off+4; lfd; fsub magic
        float->int  fctiwz; stfd off; lwz off+4
        Anything else on the stack is a local the lifter does not model."""
        if op == "stw":
            st.stack[off] = self.get(st, a[0])
        elif op == "stfd":
            value = self.get(st, a[0])
            if not (isinstance(value, Op) and value.op == "fctiwz"):
                raise Unsupported("stack locals (a double spilled)")
            st.stack[off] = ("fctiwz", value.args[0])
        elif op == "lwz":
            slot = st.stack.get(off - 4)
            if isinstance(slot, tuple) and slot[0] == "fctiwz":
                self.put(st, a[0], Op("cvt_i", (slot[1],), "int"))
            else:
                raise Unsupported("stack locals (a word read back)")
        elif op == "lfd":
            hi, lo = st.stack.get(off), st.stack.get(off + 4)
            if hi != Const(0x43300000) or lo is None:
                raise Unsupported("stack locals (a double read back)")
            self.put(st, a[0], ("i2f", lo))
        else:
            raise Unsupported("stack locals or arguments in memory")

    def finish_conversion(self, st: _State, dest: str, x, y, ctype: str) -> bool:
        """fsub(s) fD, <i2f pending>, magic: the int->float conversion completes."""
        if not (isinstance(x, tuple) and x[0] == "i2f" and isinstance(y, Literal)):
            return False
        lo = x[1]
        if y.raw == self._I2F_SIGNED and isinstance(lo, Op) and lo.op == "^" and lo.args[1] == Const(_s32(0x80000000)):
            source = lo.args[0]
        elif y.raw == self._I2F_UNSIGNED:
            source = lo
            self.mark_unsigned(source)
        else:
            raise Unsupported("unknown int->float magic constant")
        self.put(st, dest, Op("cvt_d" if ctype == "double" else "cvt_f", (self.as_value(source),), ctype))
        return True

    def call(self, st: _State, seq: int, callee: str, stmts: list) -> Call:
        out = self.out
        if not re.fullmatch(r"[A-Za-z_]\w*", callee):
            raise Unsupported(f"call to {callee}")
        cxx = demangle(callee) if re.search(r"__(?:F|\d|Q\d|C)", callee) else None
        if cxx is None and re.search(r"__(?:F|\d|Q\d|C)", callee):
            raise Unsupported("calls a C++ function whose name it cannot demangle")
        if self.ctor_mode and self.get(st, "r3") == Param("r3") and not any(isinstance(x, VtStore) for x in stmts) \
                and not out.calls and (cxx is None or cxx.special == "ctor"):
            return self.base_ctor_call(st, seq, callee, cxx, stmts)
        if cxx is not None:
            return self.cxx_call(st, seq, callee, cxx, stmts)
        sig = self.signature(callee) if self.signature else None
        if sig is None:
            raise Unsupported(f"call to {callee}: signature unknown")
        n_gpr, n_fpr = sig
        args = tuple(self.get(st, f"r{n}") for n in range(3, 3 + n_gpr)) + tuple(self.get(st, f"f{n}") for n in range(1, 1 + n_fpr))
        # &member passed to a call: an embedded object (same evidence rule as for returns)
        args = tuple(_resolve_offset(out, v, self.this_is_object) if isinstance(v, Offset) else self.elem_address(v) if isinstance(v, IndexPtr) else v for v in args)
        for v in args:
            if isinstance(v, HaPart):
                raise Unsupported("passes half an address")
        return self.finish_call(st, Call(callee, args, n_gpr, seq), stmts)

    def cxx_call(self, st: _State, seq: int, callee: str, cxx, stmts: list) -> Call:
        """A call to a demangled C++ function: arguments in parameter order, each from the
        register its type uses; the receiver of a member call is bound to the class."""
        out = self.out
        if cxx.special and cxx.special not in ("operator delete", "operator delete[]"):
            raise Unsupported(f"calls a {cxx.special} (needs the class model)")
        if any(t.reg == "none" for t in cxx.params):
            raise Unsupported("passes a class by value or varargs")
        gpr, fpr = (4, 1) if cxx.quals else (3, 1)
        args = [self.get(st, "r3")] if cxx.quals else []
        for t in cxx.params:
            if t.reg == "fpr":
                args.append(self.get(st, f"f{fpr}"))
                fpr += 1
            else:
                args.append(self.get(st, f"r{gpr}"))
                gpr += 1
            if gpr > 11 or fpr > 9:
                raise Unsupported("arguments on the stack")
        args = [_resolve_offset(out, v, self.this_is_object) if isinstance(v, Offset) else self.elem_address(v) if isinstance(v, IndexPtr) else v for v in args]
        if any(isinstance(v, HaPart) for v in args):
            raise Unsupported("passes half an address")
        if cxx.quals:
            self.bind(args[0], cxx.quals)
        for v, t in zip(args[1:] if cxx.quals else args, cxx.params):
            if isinstance(v, Const):
                if t.text.endswith(("*", "&")) and v.value != 0:
                    raise Unsupported("a constant passed as a pointer")
                continue
            if t.text.endswith(("*", "&")) and t.classes and t.text.rstrip("*& ").removeprefix("const ") == "::".join(t.classes[0]) \
                    and t.text.count("*") + t.text.count("&") == 1:
                self.bind(v, t.classes[0])                    # (a pointer to) an object of that class
            elif t.text.endswith(("*", "&")):
                if out.expected.setdefault(v, t.text) != t.text:
                    raise Unsupported("one value passed as two pointer types")
        return self.finish_call(st, Call(callee, tuple(args), gpr - 3, seq, cxx=cxx), stmts)

    def base_ctor_call(self, st: _State, seq: int, callee: str, cxx, stmts: list) -> Call:
        """The base class constructor, called on this first thing (constructor mode)."""
        if cxx is not None:
            if any(t.reg == "none" for t in cxx.params):
                raise Unsupported("base constructor takes a class by value")
            gpr, fpr, args = 4, 1, [Param("r3")]
            for t in cxx.params:
                if t.reg == "fpr":
                    args.append(self.get(st, f"f{fpr}"))
                    fpr += 1
                else:
                    args.append(self.get(st, f"r{gpr}"))
                    gpr += 1
            call = Call(callee, tuple(args), gpr - 3, seq, cxx=cxx)
        else:
            sig = self.signature(callee) if self.signature else None
            n_gpr, n_fpr = sig or (1, 0)
            args = tuple(self.get(st, f"r{n}") for n in range(3, 3 + max(n_gpr, 1))) + tuple(self.get(st, f"f{n}") for n in range(1, 1 + n_fpr))
            call = Call(callee, args, max(n_gpr, 1), seq)
        self.out.base_call = call
        return self.finish_call(st, call, stmts)

    def bind(self, obj, quals: tuple) -> None:
        """`obj` is (a pointer to) an object of class `quals`."""
        out = self.out
        if isinstance(obj, Param) and obj.reg == "r3" and self.this_is_object:
            key = ("this",)
        elif isinstance(obj, AddrOf) and isinstance(obj.loc, Field):
            key = ("embedded", obj.loc.base, obj.loc.offset)
        elif isinstance(obj, AddrOf) and isinstance(obj.loc, GlobalObj):
            key = obj.loc
        elif isinstance(obj, (Param, Load, CallResult)):
            if isinstance(obj, Param) and obj.reg in out.int_params:
                raise Unsupported(f"{obj.reg} is used both as a number and as an object")
            self.as_base.add(obj)
            out.nodes.setdefault(obj, {})
            key = obj
        else:
            raise Unsupported("member call on a computed object")
        if out.bound.setdefault(key, quals) != quals:
            raise Unsupported("one object used as two classes")
        embedded = isinstance(key, tuple) and key[:1] == ("embedded",)
        if embedded and out.bound.get(key[1]) == quals or \
                embedded and key[1] == Param("r3") and out.bound.get(("this",)) == quals:
            raise Unsupported("a member of its container's own class (base class or composition?)")
        if isinstance(key, (Param, Load, CallResult)) and any(k[:1] == ("embedded",) and k[1] == key and q == quals
                                                             for k, q in out.bound.items() if isinstance(k, tuple)):
            raise Unsupported("a member of its container's own class (base class or composition?)")

    def finish_call(self, st: _State, call: Call, stmts: list) -> Call:
        out = self.out
        out.calls.append(call)
        stmts.append(call)
        for reg in _VOLATILE:
            st.regs.pop(reg, None)
            st.clobbered.add(reg)
        if call.cxx is not None and call.cxx.special in ("operator delete", "operator delete[]"):
            st.cr.clear()
            st.stored.clear()
            st.fresh = set()
            return call                     # returns nothing
        self.put(st, "r3", CallResult(call, False))
        self.put(st, "f1", CallResult(call, True))
        st.cr.clear()
        st.stored.clear()          # memory may have changed: later loads read it afresh
        st.fresh = {"r3", "f1"}
        return call

    def virtual_call(self, st: _State, seq: int, stmts: list) -> Call:
        """bctrl through obj->vptr[slot]: lwz rV, voff(obj); lwz rF, 8+4*slot(rV); mtctr rF."""
        out, fn = self.out, st.ctr
        vptr = fn.loc.base if isinstance(fn, Load) and isinstance(fn.loc, Field) else None
        if not (isinstance(vptr, Load) and isinstance(vptr.loc, Field)) or (fn.loc.offset - 8) % 4 or fn.loc.offset < 8:
            raise Unsupported("indirect call that is not a virtual call")
        obj, voff, slot = vptr.loc.base, vptr.loc.offset, (fn.loc.offset - 8) // 4
        if self.get(st, "r3") != obj:
            raise Unsupported("virtual call with an adjusted this")
        # the two loads are the vtable pointer and the slot, not data
        self.data_loads.discard(vptr)
        self.data_loads.discard(fn)
        out.nodes.pop(vptr, None)
        self.as_base.discard(vptr)
        if out.vptrs.get(obj, voff) != voff:
            raise Unsupported("two vtable pointers")
        out.vptrs[obj] = voff
        sig = self.vsignature(slot) if self.vsignature and obj == Param("r3") and self.this_is_object else None
        if sig is None:
            # unknown slot: an argument passed through an untouched register is invisible
            out.notes.append(f"vf{slot}'s parameters are unknown: arguments passed through unchanged registers may be missing")
            fresh_r = [n for n in range(4, 11) if f"r{n}" in st.fresh]
            fresh_f = [n for n in range(1, 9) if f"f{n}" in st.fresh]
            sig = (max(fresh_r, default=3) - 2, max(fresh_f, default=0))
        n_gpr, n_fpr = sig
        n_gpr = max(n_gpr, 1)
        args = (obj,) + tuple(self.get(st, f"r{n}") for n in range(4, 3 + n_gpr)) + tuple(self.get(st, f"f{n}") for n in range(1, 1 + n_fpr))
        args = (obj,) + tuple(_resolve_offset(out, v, self.this_is_object) if isinstance(v, Offset) else self.elem_address(v) if isinstance(v, IndexPtr) else v for v in args[1:])
        call = Call(f"vf{slot}", args, n_gpr, seq, virtual=True)
        out.calls.append(call)
        stmts.append(call)
        for reg in _VOLATILE + ["r12"]:
            st.regs.pop(reg, None)
            st.clobbered.add(reg)
        self.put(st, "r3", CallResult(call, False))
        self.put(st, "f1", CallResult(call, True))
        st.cr.clear()
        st.stored.clear()
        st.fresh = {"r3", "f1"}
        st.ctr = None
        return call

    def finish_vptrs(self) -> None:
        """A vtable pointer slot must not also be read as data."""
        for obj, voff in self.out.vptrs.items():
            if any(isinstance(l.loc, Field) and l.loc.base == obj and l.loc.offset == voff for l in self.data_loads) \
                    or any(isinstance(s.loc, Field) and s.loc.base == obj and s.loc.offset == voff for s in self.out.stores):
                raise Unsupported("vtable pointer also accessed as data")
            self.out.nodes.get(obj, {}).pop(voff, None)

    # the return value ----------------------------------------------------------------------

    def returned(self, st: _State):
        """What a path returns at blr: (value, fp, void_first)."""
        if isinstance(st.regs.get("r3"), IndexPtr):
            ix = st.regs["r3"]
            # the integer reading if the body already used this base that way, else &element
            st.regs["r3"] = self.as_value(ix) if ix.base in self.raw_bases else self.elem_address(ix)
        r3, f1 = st.regs.get("r3"), st.regs.get("f1")
        if isinstance(r3, CallResult) and isinstance(f1, CallResult) and r3.call == f1.call:
            return r3, False, True          # ends with a call: returning its result compiles the same
        if "f1" in st.written and not isinstance(f1, CallResult) and \
                ("r3" not in st.written or r3 in self.as_base or isinstance(r3, HaPart)
                 or st.when.get("f1", 0) > st.when.get("r3", 0)):
            return f1, True, False          # r3 was only a temporary (a pointer, or written before f1)
        if "r3" in st.written and not isinstance(r3, HaPart):
            # a merged value at the only exit, or r3 reused as a pointer temporary: void is likelier
            return r3, False, isinstance(r3, Phi) or r3 in self.as_base or isinstance(r3, AddrOf) and bool(self.out.stores)
        if "f1" in st.written:
            return f1, True, False
        return None, False, False


def lift(insns: list[Insn], literal, this_is_object: bool = False, signature=None, vsignature=None,
         ctor_mode: bool = False) -> Lifted:
    """Symbolically execute a function ending in blr: straight-line code with calls, or small
    acyclic control flow (if / if-else / early returns) when the body branches.

    `literal(sym, size)` returns the bytes of a compiler literal (a constant in .sdata2/.rodata)
    or None when `sym` is a variable. `this_is_object`: r3 is `this` of a known class (the
    function sits in a vtable). `signature(sym)` gives a callee's (GPR, FPR) parameter counts.
    """
    frame_size, save_low = frame_layout(insns)
    insns = strip_frame(insns)
    if not insns or insns[-1].op not in ("blr", "b"):
        raise Unsupported("does not end in a return")
    ex = _Executor(literal, this_is_object, signature, vsignature, ctor_mode)
    out = ex.out
    ex.scratch, ex.regions = stack_plan(insns, save_low) if frame_size else (set(), {})
    branches = [i for i in insns[:-1] if i.op in ("blr", "b") or _BRANCH.match(i.op) or i.op.startswith("bdnz")]
    if not branches and insns[-1].op == "blr":
        st = _State({f"r{n}": Param(f"r{n}") for n in range(3, 11)} | {f"f{n}": Param(f"f{n}") for n in range(1, 9)},
                    set(), set(), set(), {})
        stmts: list = []
        for seq, insn in enumerate(insns[:-1]):
            ex.step(st, seq, insn, stmts)
        out.regs, out.written = st.regs, st.written
        out.ret, out.ret_fp, out.void_first = ex.returned(st)
        out.tree = stmts
        ex.finish_vptrs()
        return _finish_params(out, ex.as_base, this_is_object)
    _lift_cfg(ex, insns)
    ex.finish_vptrs()
    return _finish_params(out, ex.as_base, this_is_object)


def _lift_cfg(ex: _Executor, insns: list[Insn]) -> None:
    """Acyclic control flow: each conditional branch becomes an If whose arms run up to the
    branch's immediate post-dominator, where the two register states merge."""
    out = ex.out
    index_of = {}
    for i, insn in enumerate(insns):
        index_of.setdefault(insn.address, i)

    def resolve(addr: int) -> int:
        # a target may be an epilogue instruction strip_frame removed: take the next kept one
        j = next((i for i, insn in enumerate(insns) if insn.address >= addr), None)
        if j is None:
            raise Unsupported("branch outside the body")
        return j

    starts = {0}
    for i, insn in enumerate(insns):
        t = _target(insn)
        if t is not None and (insn.op == "b" or _BRANCH.match(insn.op)):
            j = resolve(t)
            if j <= i:
                raise Unsupported("loop (backward branch)")
            starts.add(j)
            starts.add(i + 1)
        elif insn.op in ("blr", "b") or _BRANCH.match(insn.op) or insn.op.startswith("bdnz"):
            if insn.op != "blr" and insn.op != "b":
                raise Unsupported(f"branch form {insn.op} {', '.join(insn.args)}")
            starts.add(i + 1)
    starts = sorted(s for s in starts if s < len(insns))
    block_of = {}
    blocks = []
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else len(insns)
        blocks.append((s, e))
        for i in range(s, e):
            block_of[i] = k
    EXIT = len(blocks)

    def succs(k: int) -> list[int]:
        s, e = blocks[k]
        last = insns[e - 1]
        t = _target(last)
        if last.op == "blr" or last.op == "b" and t is None:
            return [EXIT]
        if last.op == "b":
            return [block_of[resolve(t)]]
        if _BRANCH.match(last.op):
            return [k + 1, block_of[resolve(t)]] if k + 1 < len(blocks) else [block_of[resolve(t)]]
        return [k + 1] if k + 1 < len(blocks) else [EXIT]

    if len(blocks) > 12 or sum(_BRANCH.match(insns[e - 1].op) is not None for _, e in blocks) > 4:
        raise Unsupported("control flow too large")
    # immediate post-dominators (acyclic, so a reverse pass over blocks in address order)
    pdom: dict[int, set[int]] = {EXIT: {EXIT}}
    for k in range(len(blocks) - 1, -1, -1):
        ss = succs(k)
        common = set.intersection(*(pdom[x] for x in ss))
        pdom[k] = {k} | common

    def ipdom(k: int) -> int:
        cands = pdom[k] - {k}
        return min(cands, key=lambda x: len(pdom[x]) * -1 if x != EXIT else 0) if len(cands) > 1 else next(iter(cands))

    executed: set[int] = set()

    def run(k: int, stop: int, st: _State) -> tuple[list, _State, bool]:
        stmts: list = []
        while k != stop and k != EXIT:
            if k in executed:
                raise Unsupported("code shared by two branches (goto-like flow)")
            executed.add(k)
            s, e = blocks[k]
            body_end = e - 1 if (insns[e - 1].op in ("blr", "b") or _BRANCH.match(insns[e - 1].op)) else e
            for i in range(s, body_end):
                ex.step(st, i, insns[i], stmts)
            last = insns[e - 1]
            if last.op == "blr":
                value, fp, void_first = ex.returned(st)
                stmts.append(Return(value, e - 1))
                out.ret_fp = out.ret_fp or fp
                out.void_first = out.void_first or void_first
                return stmts, st, True
            if last.op == "b" and _target(last) is None:          # tail call
                call = ex.call(st, e - 1, last.args[0], stmts)
                stmts.append(Return(CallResult(call, False), e - 1))
                out.void_first = True
                return stmts, st, True
            if _BRANCH.match(last.op):
                field_ = last.args[0] if len(last.args) > 1 else "cr0"
                if field_ not in st.cr:
                    raise Unsupported("branch on a condition the lifter did not see")
                kind, x, y = st.cr[field_]
                taken = _CMP_OF[_BRANCH.match(last.op)[1]]
                cond = Op(_NEGATE[taken], (x, y), "float" if kind == "float" else "int")   # fallthrough runs when it holds
                j = ipdom(k)
                then, st_then, exit_then = run(k + 1, j, st.copy())
                other, st_other, exit_other = run(block_of[resolve(_target(last))], j, st.copy())
                stmts.append(If(cond, then, other, e - 1))
                if exit_then and exit_other:
                    return stmts, st_then, True
                st = st_other if exit_then else st_then if exit_other else _merge(cond, st_then, st_other)
                k = j
                continue
            k = succs(k)[0]
        return stmts, st, False

    st = _State({f"r{n}": Param(f"r{n}") for n in range(3, 11)} | {f"f{n}": Param(f"f{n}") for n in range(1, 9)},
                set(), set(), set(), {})
    tree, st, ended = run(0, EXIT, st)
    if not ended:
        raise Unsupported("falls off the end")
    out.tree = tree
    out.regs, out.written = st.regs, st.written


def _merge(cond, a: _State, b: _State) -> _State:
    regs = {}
    for reg in set(a.regs) & set(b.regs):
        va, vb = a.regs[reg], b.regs[reg]
        regs[reg] = va if va == vb else Phi(cond, va, vb)
    return _State(regs, a.written | b.written, a.clobbered | b.clobbered, a.stored | b.stored, {},
                  None, a.fresh & b.fresh)


def location_key(b):
    """Identity of what a load reads: loads of one location are one object."""
    if isinstance(b, Load) and isinstance(b.loc, Elem):
        e = b.loc
        return ("elem", location_key(e.base), e.start, e.index, e.size, e.sub, e.via_pointer)
    if isinstance(b, Load):
        return ("load", location_key(b.loc.base) if isinstance(b.loc, Field) else ("global", b.loc.sym),
                b.loc.offset if isinstance(b.loc, Field) else 0)
    return b


def _canonicalize(out: Lifted) -> None:
    """The same pointer read twice (two loads of one location) is one object: merge their
    layouts under the first load, so the object gets one placeholder type."""
    key = location_key
    rep: dict = {}
    for b in sorted(list(out.nodes), key=lambda b: b.seq if isinstance(b, Load) else -1):
        k = key(b)
        if k not in rep:
            rep[k] = b
            continue
        r = rep[k]
        for off, ctype in out.nodes.pop(b).items():
            old = out.nodes[r].get(off)
            if old and old != ctype and {old, ctype} not in ({"short", "unsigned short"}, {"int", "unsigned int"}):
                raise Unsupported(f"field +0x{off:X} accessed as {old} and {ctype}")
            out.nodes[r][off] = ctype if not old or ctype.startswith("unsigned") else old
        out.embedded.setdefault(r, set()).update(out.embedded.pop(b, set()))
        if b in out.vptrs:
            v = out.vptrs.pop(b)
            if out.vptrs.setdefault(r, v) != v:
                raise Unsupported("two vtable pointers")
        out.alias[b] = r
    out.alias.update({("key", k): r for k, r in rep.items() if isinstance(r, Load)})
    for b in list(out.embedded):
        if not out.embedded[b]:
            del out.embedded[b]


def _finish_params(out: Lifted, as_base: set, this_is_object: bool) -> Lifted:
    _canonicalize(out)
    if isinstance(out.ret, Offset):
        out.ret = _resolve_offset(out, out.ret, this_is_object)
    if isinstance(out.ret, Param) and out.ret.reg.startswith("r") and out.ret not in as_base:
        out.int_params.setdefault(out.ret.reg, "int")
    for st in _returns(out.tree):
        if isinstance(st.value, Offset):
            st.value = _resolve_offset(out, st.value, this_is_object)
    return out


def _returns(tree: list):
    for st in tree:
        if isinstance(st, Return):
            yield st
        elif isinstance(st, If):
            yield from _returns(st.then)
            yield from _returns(st.other)


def _resolve_offset(out: Lifted, value: Offset, this_is_object: bool = False) -> AddrOf:
    """A pointer + constant that is returned is the address of an embedded member, but only
    with evidence that the base is a pointer: it is also used for a field access, it is a
    loaded pointer or a global, or it is `this` of a known class. Otherwise the same
    instruction could be integer addition, and the lifter does not guess."""
    base = value.base
    if isinstance(base, AddrOf) and isinstance(base.loc, GlobalObj):
        base = base.loc
    if not isinstance(base, (Param, Load, GlobalObj)):
        raise Unsupported("returns a computed address")
    if isinstance(base, Param) and base not in out.nodes and not (this_is_object and base.reg == "r3"):
        raise Unsupported("pointer or integer? (register + constant with no other evidence)")
    out.embedded.setdefault(base, set()).add(value.delta)
    out.nodes.setdefault(base, {})
    return AddrOf(Field(base, value.delta))


# --- render ----------------------------------------------------------------------------------

def float_literal(raw: bytes) -> str:
    """Shortest decimal that rounds back to exactly these float/double bits."""
    if len(raw) == 4:
        (v,) = struct.unpack(">f", raw)
        if v != v or v in (float("inf"), float("-inf")) or (v == 0 and raw[0] & 0x80):
            raise Unsupported("special float literal")
        for digits in range(1, 18):
            text = f"{v:.{digits}g}"
            if struct.pack(">f", float(text)) == raw:
                break
        if "e" in text and 1e-5 <= abs(v) < 1e9:       # prefer plain decimals: 60.0f, not 6e+01f
            for places in range(0, 12):
                fixed = f"{v:.{places}f}"
                if struct.pack(">f", float(fixed)) == raw:
                    text = fixed
                    break
        text = text if any(c in text for c in ".e") else text + ".0"
        return text + "f"
    (v,) = struct.unpack(">d", raw)
    if v != v or v in (float("inf"), float("-inf")) or (v == 0 and raw[0] & 0x80):
        raise Unsupported("special double literal")
    text = repr(v)
    return text if any(c in text for c in ".e") else text + ".0"


def _int_text(v: int) -> str:
    return str(v) if -256 < v < 256 else (f"-0x{-v:X}" if v < 0 else f"0x{v:X}")


@dataclass
class Names:
    """How nodes print: the function's own class (member form), other struct types, names."""
    member: bool
    this_type: str
    function: str
    types: dict[object, str] = field(default_factory=dict)
    params: dict[str, str] = field(default_factory=dict)


def _simplify(e):
    """mwcc's branch-free comparison idioms back to comparisons."""
    if isinstance(e, Op) and e.op == ">>u" and e.args[1] == Const(5) and isinstance(e.args[0], Op) and e.args[0].op == "cntlzw":
        x = e.args[0].args[0]
        if isinstance(x, Op) and x.op == "-" and isinstance(x.args[0], Const):
            return Op("==", (x.args[1], x.args[0]), "int")
        if isinstance(x, Op) and x.op == "^":
            return Op("==", x.args, "int")
        return Op("==", (x, Const(0)), "int")
    if isinstance(e, Op) and e.op == ">>u" and e.args[1] == Const(31) and isinstance(e.args[0], Op) and e.args[0].op == "|":
        p, q = e.args[0].args
        if isinstance(p, Op) and p.op == "neg" and p.args[0] == q or isinstance(q, Op) and q.op == "neg" and q.args[0] == p:
            return Op("!=", (q if isinstance(p, Op) and p.op == "neg" else p, Const(0)), "int")
    if isinstance(e, Op):
        return Op(e.op, tuple(_simplify(x) for x in e.args), e.ctype)
    return e


class Renderer:
    def __init__(self, lifted: Lifted, names: Names, ret_used: bool = True):
        self.l, self.n = lifted, names
        self.ret_used = ret_used           # some rendered variant returns the value
        self.temps: dict[Load, str] = {}
        self.unify_pointer_types()
        self.check_store_types()

    def check_store_types(self) -> None:
        for st in self.l.stores:
            if not isinstance(st.loc, Field) or isinstance(st.value, (Const, Literal)):
                continue
            try:
                field_t, value_t = self.field_type(st.loc.base, st.loc.offset), self.ctype_of(st.value)
            except (KeyError, Unsupported):
                continue
            if field_t.endswith("*") and value_t.endswith("*") and field_t != value_t and "void*" not in (field_t, value_t):
                raise Unsupported("a field holding two pointer types")

    def unify_pointer_types(self) -> None:
        """A value compared with, or stored into, a pointer takes that pointer type."""
        def pointer_type(e):
            try:
                t = self.ctype_of(e)
            except (Unsupported, KeyError):
                return None
            return t if t.endswith("*") else None

        def untyped(e) -> bool:
            return (isinstance(e, (Param, CallResult)) or isinstance(e, Load) and self.rep(e) not in self.l.nodes) \
                and e not in self.l.expected and self.rep(e) not in self.l.nodes
        for root in self.roots_all():
            for e in _walk(root):
                if isinstance(e, Op) and e.op in ("==", "!=", "<", "<=", ">", ">=") and len(e.args) == 2:
                    a, b = e.args
                    for x, y in ((a, b), (b, a)):
                        t = pointer_type(x)
                        if t and untyped(y) and not isinstance(y, Const):
                            self.l.expected[y] = t
        for st in self.l.stores:
            if isinstance(st.loc, Field) and untyped(st.value):
                try:
                    t = self.field_type(st.loc.base, st.loc.offset)
                except (KeyError, Unsupported):
                    continue
                if t.endswith("*"):
                    self.l.expected[st.value] = t

    def roots_all(self) -> list:
        roots = [st.value for st in self.l.stores] + [a for c in self.l.calls for a in c.args]
        roots += [st.cond for st in _flatten(self.l.tree) if isinstance(st, If)]
        roots += [self.l.ret] + [st.value for st in _flatten(self.l.tree) if isinstance(st, Return)]
        return [r for r in roots if r is not None]

    def field_name(self, offset: int) -> str:
        return f"m{offset:X}"

    # arrays --------------------------------------------------------------------------------

    def array_key(self, loc: Elem) -> tuple:
        return (loc.base, loc.start, loc.via_pointer)

    def elem_type(self, key: tuple) -> str:
        arr = self.l.arrays[key]
        fields = arr["fields"]
        if list(fields) == [0] and SIZES.get(fields[0], 4) == arr["size"]:
            return fields[0]
        base, start, via = key
        return f"{self.object_name(base)}_{'p' if via else f'm{start:X}'}_elem"

    def object_name(self, base) -> str:
        """An identifier for an object, whether or not it got a placeholder struct."""
        if isinstance(base, Param) and base.reg == "r3" and self.n.member:
            return re.sub(r"\W", "_", self.n.this_type)
        try:
            return re.sub(r"\W", "_", self.type_name(base))
        except KeyError:
            pass
        prefix = re.sub(r"\W", "_", self.n.this_type)
        if isinstance(base, Load) and isinstance(base.loc, Field):
            return f"{self.object_name(base.loc.base)}_m{base.loc.offset:X}"
        if isinstance(base, Load) and isinstance(base.loc, GlobalVar):
            return f"Unk_{base.loc.sym}_target"
        if isinstance(base, Param):
            return f"{prefix}_arg{int(base.reg[1:]) - 3}"
        if isinstance(base, CallResult):
            return f"{prefix}_{base.call.callee}_ret"
        return f"{prefix}_obj"

    def elem_structs(self) -> list[str]:
        out = []
        for key, arr in sorted(self.l.arrays.items(), key=lambda kv: self.elem_type(kv[0])):
            t = self.elem_type(key)
            if t in SIZES:
                continue
            body, cur = [], 0
            for off in sorted(arr["fields"]):
                ctype = arr["fields"][off]
                width = SIZES.get(ctype, 4)
                if off < cur or off % width:
                    raise Unsupported("array element fields overlap")
                if off > cur:
                    body.append(f"    char _pad{cur:X}[0x{off - cur:X}];")
                body.append(f"    {ctype} m{off:X};")
                cur = off + width
            if arr["size"] > cur:
                body.append(f"    char _pad{cur:X}[0x{arr['size'] - cur:X}];")
            out += [f"struct {t} {{"] + body + ["};"]
        return out

    def via_pointer_type(self, base) -> str | None:
        for key in self.l.arrays:
            if key[2] and self.rep(key[0]) == self.rep(base):
                if self.l.nodes.get(self.rep(base)):
                    raise Unsupported("a pointer used both for fields and as an array")
                return f"{self.elem_type(key)}*"
        return None

    def local_scalar(self, b) -> str | None:
        """A local holding one scalar (int l8;), as opposed to a struct local."""
        if not isinstance(b, StackObj) or ("embedded", b, 0) in self.l.bound:
            return None
        fields = self.l.nodes.get(b, {})
        if list(fields) == [0] and not self.l.embedded.get(b) and SIZES.get(fields[0], 4) >= min(self.l.locals[b.offset], 4):
            return self.field_type(b, 0)
        return None

    def local_type(self, b) -> str:
        q = self.l.bound.get(("embedded", b, 0))
        if q:
            return "::".join(q)
        return self.local_scalar(b) or self.type_name(b)

    def local_decls(self, descending: bool) -> list[str]:
        order = sorted(self.l.locals, reverse=descending)
        return [f"    {self.local_type(StackObj(o))} l{o:X};" for o in order]

    def bound_class(self, b) -> str | None:
        """The class an object was bound to by a member call, as C++ text."""
        if isinstance(b, Param) and b.reg == "r3" and self.n.member:
            q = self.l.bound.get(("this",)) or self.l.bound.get(b)
            return "::".join(q) if q else None
        q = self.l.bound.get(self.rep(b))
        return "::".join(q) if q else None

    def rep(self, b):
        """The object a base stands for (loads of one location share a type)."""
        if b in self.l.alias:
            return self.l.alias[b]
        if isinstance(b, Load):
            return self.l.alias.get(("key", location_key(b)), b)
        return b

    def scalar_deref(self, base) -> bool:
        """A pointer param used for one scalar at offset 0: print it as *argN."""
        base = self.rep(base)
        fields = self.l.nodes.get(base, {})
        return (isinstance(base, Param) and not (self.n.member and base.reg == "r3")
                and list(fields) == [0] and not self.l.embedded.get(base) and base not in self.l.vptrs)

    def lvalue(self, loc) -> str:
        if isinstance(loc, GlobalVar):
            return loc.sym
        if isinstance(loc, Elem):
            index = self.expr(loc.index)
            if loc.via_pointer:
                head = f"{self.expr(loc.base, False)}[{index}]"
            else:
                head = f"{self.lvalue(Field(loc.base, loc.start))}[{index}]"
            return head if self.elem_type(self.array_key(loc)) in SIZES else f"{head}.{self.field_name(loc.sub)}"
        base = loc.base
        if isinstance(base, StackObj):
            if self.local_scalar(base) or (loc.offset == 0 and ("embedded", base, 0) in self.l.bound):
                return f"l{base.offset:X}"
            return f"l{base.offset:X}.{self.field_name(loc.offset)}"
        if isinstance(base, GlobalObj):
            return f"{base.sym}.{self.field_name(loc.offset)}"
        if isinstance(base, Param) and self.n.member and base.reg == "r3":
            return self.field_name(loc.offset)
        if self.scalar_deref(base):
            return f"*{self.n.params[base.reg]}"
        return f"{self.expr(base)}->{self.field_name(loc.offset)}"

    def expr(self, e, top: bool = True) -> str:
        if e in self.temps:
            return self.temps[e]
        if isinstance(e, CallResult):
            raise Unsupported("call result used without a temporary")
        if isinstance(e, Param):
            if self.n.member and e.reg == "r3":
                return "this"
            return self.n.params[e.reg]
        if isinstance(e, Const):
            return _int_text(e.value)
        if isinstance(e, Literal):
            return float_literal(e.raw)
        if isinstance(e, Load):
            return self.lvalue(e.loc)
        if isinstance(e, AddrOf) and isinstance(e.loc, Elem):
            loc = e.loc
            if loc.sub == 0:                               # the whole element
                index = self.expr(loc.index)
                head = f"{self.expr(loc.base, False)}[{index}]" if loc.via_pointer else \
                    f"{self.lvalue(Field(loc.base, loc.start))}[{index}]"
                return f"&{head}"
            return f"&{self.lvalue(loc)}"
        if isinstance(e, AddrOf):
            if isinstance(e.loc, Field) and isinstance(e.loc.base, StackObj) and e.loc.offset == 0:
                return f"&l{e.loc.base.offset:X}"          # the whole local
            if isinstance(e.loc, GlobalObj) and self.array_global(e.loc.sym):
                return e.loc.sym
            return f"&{e.loc.sym}" if isinstance(e.loc, (GlobalObj, GlobalVar)) else f"&{self.lvalue(e.loc)}"
        if isinstance(e, Op):
            text = self._op(e)
            return text if top else f"({text})"
        if isinstance(e, Phi):
            text = f"{self.expr(e.cond, False)} ? {self.expr(e.then, False)} : {self.expr(e.other, False)}"
            return text if top else f"({text})"
        raise Unsupported(f"cannot print {type(e).__name__}")

    def _op(self, e: Op) -> str:
        a = [self.expr(x, top=False) for x in e.args]
        if e.op == "neg" or e.op == "fneg":
            return f"-{a[0]}"
        if e.op == "fabs":
            raise Unsupported("fabs")
        if e.op == "cntlzw":
            raise Unsupported("count-leading-zeros outside a comparison idiom")
        if e.op in ("cvt_f", "cvt_d", "cvt_i"):
            return f"({ {'cvt_f': 'float', 'cvt_d': 'double', 'cvt_i': 'int'}[e.op] }){a[0]}"
        if e.op == "fctiwz":
            raise Unsupported("fctiwz outside a float->int conversion")
        if e.op == "extsb":
            return f"(signed char){a[0]}"
        if e.op == "extsh":
            return f"(short){a[0]}"
        if e.op == ">>u":
            return f"(unsigned int){a[0]} >> {a[1]}"
        return f"{a[0]} {e.op} {a[1]}"

    # types

    def ctype_of(self, e, bool_ret: bool = False) -> str:
        e = _simplify(e)
        if isinstance(e, (Param, Load, CallResult)) and not (isinstance(e, Param) and self.n.member and e.reg == "r3"):
            vp = self.via_pointer_type(e)
            if vp:
                return vp
        if isinstance(e, CallResult):
            if self.rep(e) in self.l.nodes:
                return f"{self.type_name(e)}*"
            if e in self.l.expected:
                return self.l.expected[e]
            return "float" if e.fp else "unsigned int" if e in self.l.unsigned_results else "int"
        if isinstance(e, Phi):
            return self.ctype_of(e.then, bool_ret)
        if isinstance(e, Load) and self.rep(e) in self.l.nodes:
            return f"{self.type_name(e)}*"
        if isinstance(e, Load) and e in self.l.expected:
            return self.l.expected[e]
        if isinstance(e, Load) and isinstance(e.loc, Field) and not getattr(self, "_typing", None):
            self._typing = True                         # field_type may ask about stored values
            try:
                return self.field_type(e.loc.base, e.loc.offset)
            except KeyError:
                pass
            finally:
                self._typing = None
        if isinstance(e, (Load, Literal)):
            return e.ctype
        if isinstance(e, Const):
            return "int"
        if isinstance(e, Op):
            if e.op in ("==", "!=") and bool_ret:
                return "bool"
            return e.ctype
        if isinstance(e, Param):
            if self.n.member and e.reg == "r3":
                return f"{self.n.this_type}*"
            return self.param_type(e.reg)
        if isinstance(e, AddrOf) and isinstance(e.loc, Elem):
            t = self.elem_type(self.array_key(e.loc))
            if t in SIZES or e.loc.sub == 0:
                return f"{t}*"
            fields = self.l.arrays[self.array_key(e.loc)]["fields"]
            return f"{fields.get(e.loc.sub, 'char')}*"
        if isinstance(e, Load) and isinstance(e.loc, Elem):
            return e.ctype
        if isinstance(e, AddrOf):
            loc = e.loc
            if isinstance(loc, Field) and isinstance(loc.base, StackObj) and loc.offset == 0:
                return f"{self.local_type(loc.base)}*"
            if isinstance(loc, GlobalObj):
                return f"{self.n.types[loc]}*"
            if isinstance(loc, GlobalVar):
                return f"{self.l.small_globals[loc.sym]}*"
            ctype = self.l.nodes.get(self.rep(loc.base), {}).get(loc.offset)
            if ctype is None:
                return f"{self.embedded_type(loc.base, loc.offset)}*"
            return f"{self.field_type(loc.base, loc.offset)}*"
        raise Unsupported(f"no type for {type(e).__name__}")

    def pointee(self, e) -> str | None:
        """What a scalar pointer parameter wants `e` to point at ("int*" -> "int")."""
        want = self.l.expected.get(e)
        if want and want.endswith("*") and want.count("*") == 1 and "(" not in want:
            return want[:-1].rstrip()
        return None

    def array_global(self, sym: str) -> str | None:
        """A global whose address is passed as a scalar pointer is an array of that scalar."""
        return self.pointee(AddrOf(GlobalObj(sym)))

    def embedded_type(self, base, offset: int) -> str:
        scalar = self.pointee(AddrOf(Field(base, offset))) or self.pointee(AddrOf(Field(self.rep(base), offset)))
        if scalar:
            return scalar
        q = self.l.bound.get(("embedded", self.rep(base), offset))
        if q is None and isinstance(base, Param) and base.reg == "r3":
            q = self.l.bound.get(("embedded", base, offset))
        return "::".join(q) if q else re.sub(r"\W", "_", self.type_name(base)) + f"_m{offset:X}"

    def type_name(self, base) -> str:
        if isinstance(base, Param) and self.n.member and base.reg == "r3":
            return self.n.this_type
        return self.bound_class(base) or self.n.types[self.rep(base)]

    def field_type(self, base, offset: int) -> str:
        """A field's type: a pointer to the object loaded from it, else a pointer stored into
        it, else what its accesses say."""
        base = self.rep(base)
        child = next((b for b in self.l.nodes if isinstance(b, Load) and isinstance(b.loc, Field)
                      and self.rep(b.loc.base) == base and b.loc.offset == offset), None)
        if child is not None:
            return f"{self.type_name(child)}*"
        for key in self.l.arrays:                          # a pointer to array elements
            k0 = key[0]
            if key[2] and isinstance(k0, Load) and isinstance(k0.loc, Field) and self.rep(k0.loc.base) == base \
                    and k0.loc.offset == offset:
                return self.via_pointer_type(k0)
        for e, want in self.l.expected.items():
            if isinstance(e, Load) and isinstance(e.loc, Field) and self.rep(e.loc.base) == base and e.loc.offset == offset:
                return want
        for st in self.l.stores:
            if isinstance(st.loc, Field) and self.rep(st.loc.base) == base and st.loc.offset == offset \
                    and not isinstance(st.value, (Const, Literal)) and self.ctype_of(st.value).endswith("*"):
                return self.ctype_of(st.value)
        return self.l.nodes[base][offset]

    def param_type(self, reg: str) -> str:
        p = Param(reg)
        if p in self.l.expected:
            return self.l.expected[p]
        vp = self.via_pointer_type(p)
        if vp:
            return vp
        if p in self.l.nodes:
            if self.scalar_deref(p):
                return f"{self.l.nodes[p][0]}*"
            return f"{self.type_name(p)}*"
        if reg.startswith("f"):
            return self.l.int_params.get(reg, "float")
        return self.l.int_params.get(reg, "int")

    # declarations

    def struct_decls(self) -> list[str]:
        """Forward declarations of every placeholder struct, and embedded unknown members."""
        lines = []
        bases = sorted(set(self.l.nodes) | {GlobalObj(g) for g in self.l.global_objs}, key=lambda b: self.type_name(b))
        for b in bases:
            if self.scalar_deref(b) or self.local_scalar(b) or isinstance(b, StackObj) and ("embedded", b, 0) in self.l.bound:
                continue
            lines.append(f"struct {self.type_name(b)};")
        for b in bases:
            for off in sorted(self.l.embedded.get(b, ())) if b in self.l.nodes else ():
                if off not in self.l.nodes[b] and self.embedded_type(b, off).endswith(f"_m{off:X}") \
                        and ("embedded", b, off) not in self.l.bound:
                    lines.append(f"struct {self.embedded_type(b, off)} {{ char _unknown; }};")
        return [l for l in dict.fromkeys(lines) if not self._is_class_decl(l)]

    def _is_class_decl(self, line: str) -> bool:
        """Forward declarations of bound classes are emitted by the class model instead."""
        name = line.removeprefix("struct ").rstrip(";")
        return any("::".join(q) == name for q in self.classes())

    # --- the C++ class model (member calls) ---------------------------------------------

    def classes(self) -> set:
        """Every qualified class the TU needs: bound objects, callee classes, parameter classes."""
        out = set(self.l.bound.values())
        for c in self.l.calls:
            if c.cxx:
                out.add(c.cxx.quals) if c.cxx.quals else None
                for t in c.cxx.params:
                    out.update(t.classes)
        return {q for q in out if q}

    def methods(self) -> dict:
        """class -> declarations of the methods called on it."""
        out: dict = {}
        for c in self.l.calls:
            if not (c.cxx and c.cxx.quals):
                continue
            ret = "void"
            for fp in (False, True):
                if CallResult(c, fp) in self._used_results():
                    ret = self.ctype_of(CallResult(c, fp))
            decl = f"{ret} {c.cxx.name}({', '.join(t.text for t in c.cxx.params)}){' const' if c.cxx.const else ''};"
            out.setdefault(c.cxx.quals, [])
            if decl not in out[c.cxx.quals]:
                out[c.cxx.quals].append(decl)
        return out

    def _used_results(self) -> set:
        return {x for r in self.roots() for x in _walk(r) if isinstance(x, CallResult)}

    def class_model(self, this_members: list[str] | None) -> tuple[list[str], list[str]]:
        """(skeletons, definitions): nested forward declarations for every class first, then
        each class with members defined out of line (fields, virtuals, called methods)."""
        classes = self.classes()
        this_q = tuple(self.n.this_type.split("::")) if self.n.member else None
        if this_q and this_members is not None:
            classes.add(this_q)
        members: dict = {q: list(m) for q, m in self.methods().items()}
        for obj, q in self.l.bound.items():
            if isinstance(obj, tuple):
                continue
            if isinstance(obj, Param) and obj.reg == "r3" and self.n.member:
                continue
            body = self.struct_body(obj, [], 0, own_type=False)
            members.setdefault(q, [])
            members[q] = [m.strip() for m in body] + members[q]
        if this_q and this_members is not None:
            members[this_q] = [m.strip() for m in this_members] + members.get(this_q, [])
        # all paths, including enclosing ones
        paths = set()
        for q in classes:
            for i in range(1, len(q) + 1):
                paths.add(q[:i])
        skeletons, definitions = [], []
        for root in sorted({q[:1] for q in paths}):
            kids = sorted(q for q in paths if len(q) == 2 and q[0] == root[0])
            if not kids:
                skeletons.append(f"struct {root[0]};")
            else:
                inner = " ".join(f"struct {k[1]};" for k in kids)
                skeletons.append(f"struct {root[0]} {{ {inner} }};")
                if members.get(root):
                    raise Unsupported(f"{root[0]} is both a class with members and an enclosing scope")
            if len({q for q in paths if q[0] == root[0] and len(q) > 2}):
                raise Unsupported("classes nested three deep")
        embedded = {q for o, q in self.l.bound.items() if isinstance(o, tuple) and o[:1] == ("embedded",)}
        order = sorted(paths, key=lambda q: (q == this_q, q not in embedded, len(q), q))
        for q in order:
            if q in embedded and q not in members:
                members[q] = []
            if members.get(q) is not None and (members[q] or q == this_q or q in embedded):
                definitions.append(f"struct {'::'.join(q)} {{")
                definitions += [f"    {m}" for m in members[q]]
                definitions.append("};")
        return skeletons, definitions

    def virtual_decls(self, base) -> list[str]:
        """vf0..vfN with the called slots' signatures: CodeWarrior places the vtable pointer
        where the first virtual function is declared."""
        calls = [c for c in self.l.calls if c.virtual and self.rep(c.args[0]) == base]
        sigs = self.call_signatures(virtual=True)
        top = max(int(c.callee[2:]) for c in calls)
        out = []
        for slot in range(top + 1):
            ret, params = sigs.get((base, slot), ("void", []))
            out.append(f"virtual {ret} vf{slot}({', '.join(params)});")
        return out

    def struct_body(self, base, extra_members: list[str], size: int = 0, own_type: bool = True) -> list[str]:
        members, cur = [], 0
        entries = {off: self.field_type(base, off) for off in self.l.nodes.get(base, {})}
        for off in self.l.embedded.get(base, ()):
            entries.setdefault(off, self.embedded_type(base, off))
        if base in self.l.vptrs:
            entries[self.l.vptrs[base]] = "@vptr"
        arrays = {k[1]: k for k in self.l.arrays if not k[2] and self.rep(k[0]) == self.rep(base)}
        for start in arrays:
            if start in entries:
                raise Unsupported("an array overlaps a field")
            entries[start] = "@array"
        for off in sorted(entries):
            ctype = entries[off]
            if ctype == "@array":
                key = arrays[off]
                size = self.l.arrays[key]["size"]
                later = [o for o in entries if o > off]
                count = max(1, (min(later) - off) // size) if later else 1
                if off > cur:
                    members.append(f"    char _pad{cur:X}[0x{off - cur:X}];")
                if off < cur:
                    raise Unsupported(f"overlapping fields at +0x{off:X}")
                members.append(f"    {self.elem_type(key)} {self.field_name(off)}[{count}];")
                cur = off + count * size
                continue
            if ctype == "@vptr":
                if off > cur:
                    members.append(f"    char _pad{cur:X}[0x{off - cur:X}];")
                members += [f"    {d}" for d in self.virtual_decls(base)]
                cur = off + 4
                continue
            width = 4 if ctype.endswith("*") else SIZES.get(ctype, 1)
            if off < cur:
                raise Unsupported(f"overlapping fields at +0x{off:X}")
            if off % width:
                raise Unsupported(f"misaligned {ctype} at +0x{off:X}")
            if off > cur:
                members.append(f"    char _pad{cur:X}[0x{off - cur:X}];")
            members.append(f"    {ctype} {self.field_name(off)};")
            cur = off + width
        if size > cur:
            members.append(f"    char _pad{cur:X}[0x{size - cur:X}];")
        return members + [f"    {m}" for m in extra_members]

    def signature(self) -> list[str]:
        """Parameters in register order: GPRs (after `this`), then FPRs."""
        used_r = [int(r[1:]) for r in self._used_regs("r")]
        used_f = [int(r[1:]) for r in self._used_regs("f")]
        first = 4 if self.n.member else 3
        params = []
        for n in range(first, max(used_r, default=first - 1) + 1):
            reg = f"r{n}"
            params.append(f"{self.param_type(reg)} {self.n.params[reg]}")
        for n in range(1, max(used_f, default=0) + 1):
            reg = f"f{n}"
            params.append(f"{self.param_type(reg)} {self.n.params[reg]}")
        return params

    def _used_regs(self, kind: str) -> list[str]:
        used = set(self.l.int_params) | {b.reg for b in self.l.nodes if isinstance(b, Param)}
        for st in self.l.stores:
            for e in _walk(st.value):
                if isinstance(e, Param):
                    used.add(e.reg)
        for e in _walk(self.l.ret):
            if isinstance(e, Param):
                used.add(e.reg)
        for root in self.roots():
            for e in _walk(root):
                if isinstance(e, Param):
                    used.add(e.reg)
        return sorted((r for r in used if r.startswith(kind)), key=lambda r: int(r[1:]))

    def body(self, order: list[Store], ret_mode: str, cache: bool = False) -> list[str]:
        self.cache = cache
        if has_flow(self.l):
            return self.body_tree(ret_mode)
        return self.body_flat(order, ret_mode)

    def shared_loads(self, statements_exprs: list[list]) -> set:
        """Loads that appear in two or more statements (the asm read them once)."""
        if not getattr(self, "cache", False):
            return set()
        count: dict = {}
        for roots in statements_exprs:
            for load in {x for r in roots for x in _walk(r) if isinstance(x, Load)}:
                count[load] = count.get(load, 0) + 1
        return {l for l, n in count.items() if n >= 2}

    def body_tree(self, ret_mode: str) -> list[str]:
        """Nested if/else and early returns, in address order. Temporaries as in body_flat:
        a load invalidated (by a store to it or any call) before its use is declared right
        after it was read; a call result used later gets one."""
        self.temps = {}
        flat = list(_flatten(self.l.tree))
        void = ret_mode == "void"

        def exprs(st) -> list:
            if isinstance(st, Store):
                return [st.value] + ([st.loc.base] if isinstance(st.loc, Field) else [])
            if isinstance(st, Call):
                return list(st.args)
            if isinstance(st, If):
                return [st.cond]
            return [] if void or st.value is None else [st.value]

        def seq_of(st) -> int:
            return st.seq

        hoist_before: dict[int, list[Load]] = {}
        seen: set[Load] = set()
        shared = self.shared_loads([exprs(st) for st in flat])
        for i, st in enumerate(flat):
            for e in (x for root in exprs(st) for x in _walk(root)):
                if not isinstance(e, Load) or e in seen:
                    continue
                if e in shared or any(e.seq < seq_of(ev) < seq_of(st) and (isinstance(ev, Call) or isinstance(ev, Store) and ev.loc == e.loc)
                                      for ev in flat[:i]):
                    seen.add(e)
                    first = min(i, next(j for j, ev in enumerate(flat) if seq_of(ev) > e.seq))
                    hoist_before.setdefault(id(flat[first]), []).append(e)
        n_hoist = sum(len(v) for v in hoist_before.values())

        def used_later(cr: CallResult, i: int) -> bool:
            return any(x == cr for st in flat[i + 1:] for root in exprs(st) for x in _walk(root))

        position = {id(st): i for i, st in enumerate(flat)}
        lines: list[str] = []
        scopes: list[set] = [set()]
        counter = [0]

        def checked(e) -> str:
            """An expression, refusing temporaries declared in a block that has closed."""
            for x in _walk(e):
                name = self.temps.get(x)
                if name and not any(name in sc for sc in scopes):
                    raise Unsupported("a value from inside a branch used after it")
            return self.expr(e)

        def declare(name: str) -> str:
            scopes[-1].add(name)
            return name

        def emit(block: list, depth: int, top: bool) -> None:
            pad = "    " * depth
            k = 0
            while k < len(block):
                st = block[k]
                for load in hoist_before.get(id(st), []):
                    counter[0] += 1
                    name = declare("old" if n_hoist == 1 else f"tmp{counter[0]}")
                    lines.append(f"{pad}{self.ctype_of(load)} {name} = {self.lvalue(load.loc)};")
                    self.temps[load] = name
                if isinstance(st, Store):
                    checked(st.value)
                    lines.append(f"{pad}{self.lvalue(st.loc)} = {self.expr(st.value)};")
                elif isinstance(st, Call):
                    for a in st.args:
                        checked(a)
                    text = self.call_text(st)
                    nxt = block[k + 1] if k + 1 < len(block) else None
                    if (isinstance(nxt, Return) and not void and isinstance(nxt.value, CallResult)
                            and nxt.value.call == st and not hoist_before.get(id(nxt))):
                        lines.append(f"{pad}return {text};")
                        k += 2
                        continue
                    i = position[id(st)]
                    for fp in (False, True):
                        cr = CallResult(st, fp)
                        if used_later(cr, i):
                            counter[0] += 1
                            name = declare(f"t{counter[0]}")
                            lines.append(f"{pad}{self.ctype_of(cr)} {name} = {text};")
                            self.temps[cr] = name
                            break
                    else:
                        lines.append(f"{pad}{text};")
                elif isinstance(st, If):
                    then, other, cond = st.then, st.other, _simplify(st.cond)
                    if not then and not other:
                        k += 1                  # the branch only selects a value (a ternary)
                        continue
                    if not then and other:
                        then, other, cond = other, [], _negate(cond)
                    checked(cond)
                    lines.append(f"{pad}if ({self.expr(cond)}) {{")
                    scopes.append(set())
                    emit(then, depth + 1, False)
                    scopes.pop()
                    if other:
                        lines.append(f"{pad}}} else {{")
                        scopes.append(set())
                        emit(other, depth + 1, False)
                        scopes.pop()
                    lines.append(f"{pad}}}")
                elif isinstance(st, Return):
                    last = top and k == len(block) - 1
                    if void:
                        if not last:
                            lines.append(f"{pad}return;")
                    else:
                        if st.value is None:
                            raise Unsupported("a path returns nothing")
                        checked(_simplify(st.value))
                        lines.append(f"{pad}return {self.expr(_simplify(st.value))};")
                k += 1

        emit(self.l.tree, 1, True)
        return lines

    def body_flat(self, order: list[Store], ret_mode: str) -> list[str]:
        """Statements in order (stores in `order`, calls where they happen), then the return.

        A load whose location is written, or that any call could change, between where it was
        read and where it is used goes into a temporary declared right after the read. A call
        result used later goes into a temporary; one returned at once becomes `return f(...)`.
        """
        self.temps = {}
        if self.l.calls:
            stmts = sorted([(st.seq, "store", st) for st in self.l.stores] + [(c.seq, "call", c) for c in self.l.calls],
                           key=lambda t: t[0])
        else:
            stmts = [(st.seq, "store", st) for st in order]
        ret = _simplify(self.l.ret) if ret_mode != "void" else None

        def exprs(kind, item):
            if kind == "store":
                return [item.value] + ([item.loc.base] if isinstance(item.loc, Field) else [])
            if kind == "call":
                return list(item.args)
            return [item]

        # temporaries for loads that a later write or call invalidates before their use
        hoist_at: dict[int, list[Load]] = {}
        seen: set[Load] = set()
        uses = stmts + ([(10**9, "return", ret)] if ret is not None else [])
        shared = self.shared_loads([exprs(k, it) for _, k, it in uses])
        for i, (seq, kind, item) in enumerate(uses):
            for e in (x for root in exprs(kind, item) for x in _walk(root)):
                if not isinstance(e, Load) or e in seen:
                    continue
                if e in shared or any(e.seq < s2 < seq and (k2 == "call" or k2 == "store" and it2.loc == e.loc)
                                      for s2, k2, it2 in stmts[:i]):
                    seen.add(e)
                    first = min(i, next(j for j, (s2, _, _) in enumerate(stmts) if s2 > e.seq))
                    hoist_at.setdefault(first, []).append(e)
        n_hoist = sum(len(v) for v in hoist_at.values())

        def used_after(call: Call, i: int) -> bool:
            later = [x for _, k, it in uses[i + 1:] for root in exprs(k, it) for x in _walk(root)]
            return any(isinstance(x, CallResult) and x.call == call for x in later)

        lines: list[str] = []
        merged = False
        for i, (seq, kind, item) in enumerate(stmts):
            for load in hoist_at.get(i, []):
                name = "old" if n_hoist == 1 else f"tmp{len(self.temps)}"
                lines.append(f"    {self.ctype_of(load)} {name} = {self.lvalue(load.loc)};")
                self.temps[load] = name
            if kind == "store":
                lines.append(f"    {self.lvalue(item.loc)} = {self.expr(item.value)};")
                continue
            text = self.call_text(item)
            last = i == len(stmts) - 1
            if last and isinstance(ret, CallResult) and ret.call == item:
                lines.append(f"    return {text};")
                merged = True
            elif used_after(item, i):
                for fp in (False, True):
                    cr = CallResult(item, fp)
                    if any(isinstance(x, CallResult) and x == cr for _, k, it in uses[i + 1:]
                           for root in exprs(k, it) for x in _walk(root)):
                        name = f"t{len([t for t in self.temps.values() if t.startswith('t') and t[1:].isdigit()])}"
                        lines.append(f"    {self.ctype_of(cr)} {name} = {text};")
                        self.temps[cr] = name
                        break
            else:
                lines.append(f"    {text};")
        if ret is not None and not merged:
            lines.append(f"    return {self.expr(ret)};")
        return lines

    def arg_type(self, e, gpr: bool) -> str:
        if e in self.l.expected:
            return self.l.expected[e]
        if gpr and isinstance(e, Load) and self.ctype_of(e).endswith("*"):
            return "void*"
        if not gpr:
            return "double" if getattr(e, "ctype", "") == "double" else "float"
        if isinstance(e, AddrOf) or self.rep(e) in self.l.nodes or (isinstance(e, Param) and self.n.member and e.reg == "r3"):
            return "void*"
        return "int"

    def roots(self) -> list:
        """Every expression the body evaluates."""
        roots = [st.value for st in self.l.stores] + [st.loc.base for st in self.l.stores if isinstance(st.loc, Field)]
        roots += [a for c in self.l.calls for a in c.args]
        roots += [st.cond for st in _flatten(self.l.tree) if isinstance(st, If)]
        if self.ret_used:
            roots += [self.l.ret] + [st.value for st in _flatten(self.l.tree) if isinstance(st, Return)]
        return roots

    def cxx_arg(self, e, t) -> str:
        if not t.text.endswith(("*", "&")) and t.reg == "gpr" and self.ctype_of(e).endswith("*"):
            raise Unsupported(f"passes a pointer where {t.text} is expected")
        if t.text.endswith("&"):                      # a reference: pass the object itself
            if isinstance(e, AddrOf) and isinstance(e.loc, Field) and isinstance(e.loc.base, StackObj) and e.loc.offset == 0:
                return f"l{e.loc.base.offset:X}"
            if isinstance(e, AddrOf):
                return e.loc.sym if isinstance(e.loc, (GlobalObj, GlobalVar)) else self.lvalue(e.loc)
            return f"*{self.expr(e, False)}"
        return self.expr(e)

    def call_text(self, call: Call) -> str:
        if call.cxx:
            fn = call.cxx
            args = call.args[1:] if fn.quals else call.args
            text = ", ".join(self.cxx_arg(a, t) for a, t in zip(args, fn.params))
            if not fn.quals:
                return f"{fn.name}({text})"
            obj = call.args[0]
            if isinstance(obj, Param) and obj.reg == "r3" and self.n.member:
                return f"{fn.name}({text})"
            if isinstance(obj, AddrOf):
                if isinstance(obj.loc, Field) and isinstance(obj.loc.base, StackObj) and obj.loc.offset == 0:
                    target = f"l{obj.loc.base.offset:X}"
                else:
                    target = obj.loc.sym if isinstance(obj.loc, (GlobalObj, GlobalVar)) else self.lvalue(obj.loc)
                return f"{target}.{fn.name}({text})"
            return f"{self.expr(obj, False)}->{fn.name}({text})"
        if not call.virtual:
            return f"{call.callee}({', '.join(self.expr(a) for a in call.args)})"
        obj, rest = call.args[0], call.args[1:]
        target = "" if isinstance(obj, Param) and self.n.member and obj.reg == "r3" else f"{self.expr(obj, False)}->"
        return f"{target}{call.callee}({', '.join(self.expr(a) for a in rest)})"

    def call_signatures(self, virtual: bool = False) -> dict:
        """Callee declarations: parameter kinds from the arguments, return type from how the
        result is used (void when unused). Keyed by callee name, or (object, slot) for
        virtual calls."""
        roots = self.roots()
        used = {x for r in roots for x in _walk(r) if isinstance(x, CallResult)}
        sigs: dict = {}
        for c in self.l.calls:
            if c.virtual != virtual or c.cxx:
                continue
            if CallResult(c, False) in used and CallResult(c, True) in used:
                raise Unsupported(f"{c.callee}: result used as both int and float")
            ret = "void"
            for fp in (False, True):
                if CallResult(c, fp) in used:
                    ret = self.ctype_of(CallResult(c, fp))
            params = [self.arg_type(a, i < c.n_gpr) for i, a in enumerate(c.args)]
            key = c.callee
            if c.virtual:
                params, key = params[1:], (self.rep(c.args[0]), int(c.callee[2:]))
            if key in sigs:
                old_ret, old_params = sigs[key]
                if old_params != params or (old_ret != ret and "void" not in (old_ret, ret)):
                    raise Unsupported(f"{c.callee} called with different argument kinds")
                ret = old_ret if ret == "void" else ret
            sigs[key] = (ret, params)
        return sigs


def _negate(cond):
    if isinstance(cond, Op) and cond.op in _NEGATE:
        return Op(_NEGATE[cond.op], cond.args, cond.ctype)
    return Op("!", (cond,), "int")


def _walk(e):
    if e is None:
        return
    yield e
    if isinstance(e, Op):
        for x in e.args:
            yield from _walk(x)
    elif isinstance(e, Load) and isinstance(e.loc, Field):
        yield from _walk(e.loc.base)
    elif isinstance(e, (Load, AddrOf)) and isinstance(e.loc, Elem):
        yield from _walk(e.loc.base)
        yield from _walk(e.loc.index)
    elif isinstance(e, AddrOf) and isinstance(e.loc, Field):
        yield from _walk(e.loc.base)
    elif isinstance(e, Phi):
        for x in (e.cond, e.then, e.other):
            yield from _walk(x)


def _flatten(tree: list):
    """Statements in execution (= address) order, nested ones included."""
    for st in tree:
        yield st
        if isinstance(st, If):
            yield from _flatten(st.then)
            yield from _flatten(st.other)


def has_flow(lifted) -> bool:
    return any(isinstance(st, (If, Return)) for st in lifted.tree)


def return_value(lifted):
    """The value a function returns (for branching bodies, the first returned value that is
    not a constant, since a constant fits any type)."""
    if not has_flow(lifted):
        return lifted.ret
    values = [st.value for st in _flatten(lifted.tree) if isinstance(st, Return) and st.value is not None]
    return next((v for v in values if not isinstance(v, Const)), values[0] if values else None)


@dataclass(frozen=True)
class Variant:
    order: tuple[int, ...]      # store indexes in emission order
    ret_mode: str               # "return" | "bool" | "void"
    cache: bool = False         # values read once but used by several statements go in locals
    locals_desc: bool = False   # declare stack locals from the highest offset down


def variants(lifted: Lifted, limit: int = 32) -> list[Variant]:
    out = _variants(lifted, limit)
    if len(lifted.locals) >= 2:                 # which declaration order reproduces the frame?
        out = [v for v in out] + [Variant(v.order, v.ret_mode, v.cache, True) for v in out]
    return out[:limit]


def _variants(lifted: Lifted, limit: int = 32) -> list[Variant]:
    if has_flow(lifted):
        values = [st.value for st in _flatten(lifted.tree) if isinstance(st, Return)]
        if all(v is None for v in values):
            return [Variant((), "void")]
        modes = ["void", "return"] if lifted.void_first else ["return", "void"]
        if all(isinstance(_simplify(v), Op) and _simplify(v).op in ("==", "!=") or v == Const(0) or v == Const(1)
               for v in values if v is not None):
            modes.insert(modes.index("return"), "bool")
        return [Variant((), m) for m in modes] + [Variant((), m, True) for m in modes]
    stores = lifted.stores
    modes = ["void"] if lifted.ret is None else ["return"]
    if lifted.ret is not None:
        ret = _simplify(lifted.ret)
        if isinstance(ret, Op) and ret.op in ("==", "!="):
            modes.insert(0, "bool")         # both may match; bool is the natural C++
        if not lifted.ret_fp:
            modes = ["void"] + modes if lifted.void_first else modes + ["void"]
    base = tuple(range(len(stores)))
    orders = [base]
    # Statement order may differ from instruction order (scheduling). Only reorder stores that
    # neither read what another writes nor write the same location twice.
    reads = [{e.loc for e in _walk(st.value) if isinstance(e, Load)} for st in stores]
    locs = [st.loc for st in stores]
    # provably distinct locations only: one base (different offsets) or distinct globals
    if any(isinstance(l, Elem) for l in locs):
        return [Variant(base, m) for m in modes] + [Variant(base, m, True) for m in modes]
    bases = {location_key(l.base) if isinstance(l, Field) else ("global", l.sym) for l in locs}
    distinct = len(bases) == 1 or all(isinstance(l, GlobalVar) or isinstance(l, Field) and isinstance(l.base, GlobalObj)
                                      for l in locs)
    independent = distinct and len(set(locs)) == len(locs) and not any(r & set(locs) for r in reads)
    if independent and 2 <= len(stores) <= 4 and not lifted.calls:
        orders += [p for p in itertools.permutations(base) if p != base]
    out = [Variant(o, m) for o in orders for m in modes]
    out[len(modes):len(modes)] = [Variant(base, m, True) for m in modes]       # right after the plain base order
    return out[:limit]


def name_nodes(lifted: Lifted, prefix: str) -> dict[object, str]:
    """Placeholder type names: Unk<ADDR> for globals named lbl_<ADDR>, <prefix>_argN for
    pointer params, <parent>_m<OFF> for structs reached through a pointer field."""
    types: dict[object, str] = {}

    def name(b) -> str:
        if b in types:
            return types[b]
        if isinstance(b, StackObj):
            t = f"{prefix}_l{b.offset:X}"
        elif isinstance(b, GlobalObj):
            m = re.fullmatch(r"lbl_([0-9A-F]{8})", b.sym)
            t = f"Unk{m[1]}" if m else f"Unk_{b.sym}"
        elif isinstance(b, Param):
            t = f"{prefix}_arg{int(b.reg[1:]) - 3}"
        elif isinstance(b, CallResult):
            t = f"{prefix}_{b.call.callee}_ret"
        elif isinstance(b.loc, Elem):                      # pointer held in an array element
            t = f"{prefix}_a{b.loc.start:X}_e{b.loc.sub:X}"
        elif isinstance(b.loc, GlobalVar):                 # pointer held in a small global
            m = re.fullmatch(r"lbl_([0-9A-F]{8})", b.loc.sym)
            t = f"Unk{m[1]}_target" if m else f"Unk_{b.loc.sym}_target"
        else:
            parent = b.loc.base
            t = f"{name(parent) if not (isinstance(parent, Param) and parent.reg == 'r3') else prefix}_m{b.loc.offset:X}"
        types[b] = t
        return t

    for b in lifted.nodes:
        name(b)
    for b in lifted.global_objs:
        name(GlobalObj(b))
    for b, r in lifted.alias.items():
        if not (isinstance(b, tuple) and b[:1] == ("key",)):
            types[b] = types[r]
    return types


def param_names(lifted: Lifted) -> dict[str, str]:
    names = {f"r{n}": f"arg{n - 3}" for n in range(3, 11)}
    names.update({f"f{n}": f"farg{n - 1}" for n in range(1, 9)})
    return names


def render_tu(lifted: Lifted, member: bool, this_type: str, fn_name: str, chosen: list[Variant],
              global_sizes: dict[str, int], display: bool = False) -> tuple[str, list[str]]:
    """One compilable TU holding every variant as v0..vN (or, for display, the first variant
    under the function's own name). Returns (source, struct layout notes)."""
    prefix = re.sub(r"\W", "_", this_type)
    names = Names(member, this_type, fn_name, name_nodes(lifted, prefix), param_names(lifted))
    r = Renderer(lifted, names, any(v.ret_mode != "void" for v in chosen))
    this = Param("r3")
    out: list[str] = []
    notes: list[str] = []
    decls = r.struct_decls()
    if display and member:
        decls = [d for d in decls if d != f"struct {this_type};"]
    out += decls + r.elem_structs()
    sig = ", ".join(r.signature())
    fn_names = [fn_name] if display else [f"v{i}" for i in range(len(chosen))]
    value = return_value(lifted)
    ret_types = [("void" if v.ret_mode == "void" else r.ctype_of(value, v.ret_mode == "bool")) for v in chosen]
    this_in_model = member and ("::" in this_type or tuple(this_type.split("::")) in r.classes())
    this_members = None
    if this_in_model and not display:
        this_members = r.struct_body(this, [f"{t} {n}({sig});" for t, n in zip(ret_types, fn_names)])
    bound_objs = [o for o in lifted.bound if not isinstance(o, tuple) and not (isinstance(o, Param) and o.reg == "r3" and member)]
    if len({lifted.bound[o] for o in bound_objs}) != len(bound_objs) and any(lifted.nodes.get(o) for o in bound_objs):
        raise Unsupported("two objects of one class with fields")
    skeletons, definitions = r.class_model(this_members)
    out += skeletons + definitions
    for b in sorted(lifted.nodes, key=lambda b: r.type_name(b)):
        if r.scalar_deref(b) or (member and b == this) or r.bound_class(b) or r.local_scalar(b):
            continue
        if isinstance(b, StackObj) and ("embedded", b, 0) in lifted.bound:
            continue
        size = global_sizes.get(b.sym, 0) if isinstance(b, GlobalObj) else lifted.locals.get(b.offset, 0) if isinstance(b, StackObj) else 0
        body = r.struct_body(b, [], size)
        if isinstance(b, GlobalObj):
            end = max([size] + [off + (4 if t.endswith("*") else SIZES.get(t, 1)) for off, t in lifted.nodes[b].items()])
            if end <= 8:
                raise Unsupported("small global addressed absolutely (its declaration is unknown)")
        out += [f"struct {r.type_name(b)} {{"] + (body or [f"    char _pad0[0x{max(size, 1):X}];"]) + ["};"]
        if not isinstance(b, StackObj):
            notes.append(_layout_note(r, b))
    for sym in sorted(lifted.global_objs):
        if r.array_global(sym):
            if lifted.nodes.get(GlobalObj(sym)):
                raise Unsupported(f"{sym} used both as a struct and as an array")
            out.append(f"extern {r.array_global(sym)} {sym}[];")
        else:
            out.append(f"extern {r.type_name(GlobalObj(sym))} {sym};")
    for sym, ctype in sorted(lifted.small_globals.items()):
        held = next((b for b in lifted.nodes if isinstance(b, Load) and b.loc == GlobalVar(sym)), None)
        out.append(f"extern {r.type_name(held) + '*' if held is not None else ctype} {sym};")
    for callee, (ret, params) in sorted(r.call_signatures().items()):
        out.append(f'extern "C" {ret} {callee}({", ".join(params)});')
    for c in lifted.calls:
        if c.cxx and not c.cxx.quals and c.cxx.special in ("operator delete", "operator delete[]") and len(c.cxx.params) > 1:
            decl = f"void {c.cxx.special}({', '.join(t.text for t in c.cxx.params)});"
            if decl not in out:
                out.append(decl)
        if c.cxx and not c.cxx.quals and not c.cxx.special:
            ret = "void"
            for fp in (False, True):
                if CallResult(c, fp) in r._used_results():
                    ret = r.ctype_of(CallResult(c, fp))
            decl = f"{ret} {c.cxx.name}({', '.join(t.text for t in c.cxx.params)});"
            if decl not in out:
                out.append(decl)
    if member:
        if display:
            notes.insert(0, _layout_note(r, this, this_type))
        elif not this_in_model:
            decls = [f"{t} {n}({sig});" for t, n in zip(ret_types, fn_names)]
            out += [f"struct {this_type} {{"] + r.struct_body(this, decls) + ["};"]
    for v, t, n in zip(chosen, ret_types, fn_names):
        stores = [lifted.stores[i] for i in v.order]
        head = f"{t} {this_type}::{n}({sig})" if member else f'extern "C" {t} {n}({sig})'
        lines = r.body(stores, v.ret_mode, v.cache)
        if lifted.locals:
            lines = r.local_decls(v.locals_desc) + lines
        out += [head + " {"] + lines + ["}"] if lines else [head + " {}"]
    return "\n".join(out) + "\n", [n for n in notes if n]


def _layout_note(r: Renderer, base, name: str | None = None) -> str:
    entries = {off: r.field_type(base, off) for off in r.l.nodes.get(base, {})}
    for off in r.l.embedded.get(base, ()):
        entries.setdefault(off, "embedded object, type unknown")
    if base in r.l.vptrs:
        slots = sorted({int(c.callee[2:]) for c in r.l.calls if c.virtual and r.rep(c.args[0]) == base})
        entries[r.l.vptrs[base]] = f"vtable pointer (calls {', '.join(f'vf{n}' for n in slots)})"
    if not entries:
        return ""
    fields = ", ".join(f"+0x{off:X} {t}" + ("" if t.startswith("vtable pointer") else f" {r.field_name(off)}")
                       for off, t in sorted(entries.items()))
    return f"{name or r.type_name(base)}: {fields}"


# --- verify ----------------------------------------------------------------------------------

R_PPC_ADDR16_LO, R_PPC_ADDR16_HI, R_PPC_ADDR16_HA, R_PPC_REL24, R_PPC_EMB_SDA21 = 4, 5, 6, 10, 109
_KIND = {R_PPC_ADDR16_LO: "l", R_PPC_ADDR16_HI: "h", R_PPC_ADDR16_HA: "ha", R_PPC_REL24: "rel24", R_PPC_EMB_SDA21: "sda21"}
_MASK = {"l": 0xFFFF, "h": 0xFFFF, "ha": 0xFFFF, "sda21": 0x1FFFFF, "rel24": 0x03FFFFFC}


def compare(insns: list[Insn], obj: Elf, fn_symbol: str, literal, renames: dict | None = None,
            literal_info=None) -> str | None:
    """None if the compiled function equals the original, else the first difference.

    Relocated fields are masked; the target must be the same symbol and addend, except a
    compiler literal (auto-named .sdata2/.rodata data), which must hold the same bytes in the
    same section: a function-level match, as objdiff scores it (the unit's literal pool
    layout is checked by dk check and the accept-time hash). `renames` maps our symbol to an
    auto-named target symbol the draft asks the agent to rename (fn_X -> __dt__1BFv).
    `literal_info(sym)` gives (section, size) of a target literal.
    """
    fn = obj.functions().get(fn_symbol)
    if fn is None:
        return f"{fn_symbol} missing from the compiled object"
    if len(fn.code) != 4 * len(insns):
        return f"size 0x{len(fn.code):X} vs 0x{4 * len(insns):X}"
    relocs = {}
    for r in fn.relocs:                                   # HA/LO sit on the immediate halfword
        if (r.offset & ~3) in relocs:
            return f"two relocations on instruction {r.offset // 4}"
        relocs[r.offset & ~3] = r
    for i, insn in enumerate(insns):
        ours = struct.unpack_from(">I", fn.code, 4 * i)[0]
        want = insn.reloc()
        got = relocs.get(4 * i)
        if want is None and got is None:
            if ours != insn.word:
                return f"instruction {i} ({insn.op}) differs"
            continue
        if want is None or got is None or _KIND.get(got.type) != want[0]:
            return f"instruction {i} ({insn.op}) relocation differs"
        mask = _MASK[want[0]]
        if ours & ~mask != insn.word & ~mask:
            return f"instruction {i} ({insn.op}) differs"
        kind, sym, addend = want
        access = LOADS.get(insn.op, STORES.get(insn.op, (None, None)))[1]
        info = literal_info(sym) if literal_info else None
        size = access or (info[1] - addend if info else 4)    # an address: the whole object
        raw = literal(sym, size, addend) if size and size > 0 else None
        if raw is not None:
            sec = obj.section(got.target_section) if got.target_section else None
            start = got.target_value + got.addend
            if sec is None or sec.data[start:start + len(raw)] != raw:
                return f"instruction {i}: literal value differs"
            if info and got.target_section != info[0]:
                return f"instruction {i}: literal in {got.target_section}, original in {info[0]}"
        elif renames and renames.get(got.symbol) == sym and not got.target_section and got.addend == addend:
            pass
        elif got.symbol != sym or got.addend != addend:
            # (a same-named symbol this unit defines, like a vtable, is the same object linked)
            return f"instruction {i}: refers to {got.symbol}+{got.addend:#x}, original {sym}+{addend:#x}"
    return None
