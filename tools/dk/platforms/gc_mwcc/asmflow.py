"""Value numbering over dtk's PowerPC disassembly: the data-flow view `dk inlines` matches on.

Each instruction becomes a Node. Copies (mr, fmr, mtctr) and address or constant arithmetic
(li, lis, addi, sym@ha/@l, @sda21) fold into the values they produce, so register renaming,
shared constants and moved stack slots do not matter. A value is (root, offset); the root is
the stack pointer, a register's entry value, a symbol, a constant (the offset is the number),
the result of a node, a join (phi) at a label, or a clobbered register.

Straight-line numbering with joins: at a label, a register keeps its value only when every
predecessor (fall-through, forward branches, the last jump-table dispatch) agrees and no
backward branch to it writes the register; otherwise it gets a fresh phi value.

The frame (stwu/addi r1, LR save and restore, callee-saved register spills and reloads,
_savegpr/_restgpr, blr) is marked kind "frame" so matching and coverage can leave it out.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

Value = tuple            # (root, offset)
SP = ("sp",)
K = ("k",)               # constant root: the offset is the value (32-bit unsigned)
LR = ("lr",)

MASK = 0xFFFFFFFF
ARG_GPRS = tuple(f"r{i}" for i in range(3, 11))
ARG_FPRS = tuple(f"f{i}" for i in range(1, 9))
ARG_REGS = ARG_GPRS + ARG_FPRS
VOLATILE = (("r0",) + tuple(f"r{i}" for i in range(3, 13)) + tuple(f"f{i}" for i in range(0, 14))
            + ("cr0", "cr1", "cr5", "cr6", "cr7", "ctr"))
CALLEE_SAVED = frozenset([f"r{i}" for i in range(14, 32)] + [f"f{i}" for i in range(14, 32)])
FRAME_HELPERS = ("_savegpr_", "_restgpr_", "_savefpr_", "_restfpr_")
COMMUTATIVE = frozenset({"add", "add.", "and", "and.", "or", "or.", "xor", "xor.", "mullw", "mullw.", "nand",
                         "nor", "eqv", "mulhw", "mulhwu", "fadds", "fadd", "fmuls", "fmul", "addc", "adde",
                         "lwzx", "lbzx", "lhzx", "lhax", "lfsx", "lfdx"})

_LINE = re.compile(r"^/\* (?P<addr>[0-9A-F]{8}) [0-9A-F]+\s+(?:[0-9A-F]{2} ){4}\*/\t(?P<mnem>\S+)(?:\s+(?P<ops>.*))?$")
_LABEL = re.compile(r'^(?P<name>[^\s/#"][^\s]*|"[^"]+"):$')
_REG = re.compile(r"^(?:r|f)(?:[12]?\d|3[01])$")
_CR = re.compile(r"^cr([0-7])$")
_SYMREL = re.compile(r'^(?:"(?P<q>[^"]+)"|(?P<s>[^\s"@]+))@(?P<rel>ha|h|l|sda21|sda2)$')
_MEM = re.compile(r"^(?P<disp>.*)\((?P<base>r\d+)\)$")
_INT = re.compile(r"^-?(?:0x[0-9A-Fa-f]+|\d+)$")
_CRBIT = re.compile(r"^(?:4\*)?(?:cr(?P<n>[0-7]))?\+?(?P<bit>lt|gt|eq|so|un)$")


@dataclass
class Insn:
    addr: int
    mnem: str
    ops: tuple[str, ...]
    labels: tuple[str, ...] = ()     # labels defined right before this instruction

    @property
    def text(self) -> str:
        return f"{self.mnem} {', '.join(self.ops)}".rstrip()


@dataclass
class Node:
    i: int
    insn: Insn
    kind: str                         # frame | copy | pure | op | call | branch | jump | skip
    key: str = ""                     # equal keys are a precondition for pairing two nodes
    ins: tuple = ()                   # input values (memory operands contribute their address)
    deps: tuple[int, ...] = ()        # nodes whose outputs this one reads
    outs: dict = field(default_factory=dict)   # register -> value after this instruction
    mem: Value | None = None          # address of a load/store
    store: bool = False
    target: str | None = None         # branch target label (conditional branches and jumps)
    args: tuple = ()                  # calls: (register, value) set up since the previous call
    regs: dict | None = None          # calls: every argument register's value at the call

    @property
    def addr(self) -> int:
        return self.insn.addr

    @property
    def mnem(self) -> str:
        return self.insn.mnem


@dataclass
class Flow:
    name: str
    nodes: list[Node]
    labels: dict[str, int]            # label -> index of the node it precedes

    def __len__(self) -> int:
        return len(self.nodes)


# ---------------------------------------------------------------- parsing

def split_operands(text: str) -> tuple[str, ...]:
    """Split on commas outside double quotes (dtk quotes names such as "f<1,2>")."""
    out, cur, quoted = [], [], False
    for ch in text:
        if ch == '"':
            quoted = not quoted
        if ch == "," and not quoted:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail or out:
        out.append(tail)
    return tuple(out)


def parse_insns(text: str) -> list[Insn]:
    """Instructions of one function's (or a file's) disassembly, with the labels before each."""
    insns: list[Insn] = []
    pending: list[str] = []
    for line in text.splitlines():
        m = _LINE.match(line)
        if m:
            ops = split_operands(m["ops"] or "")
            insns.append(Insn(int(m["addr"], 16), m["mnem"], ops, tuple(pending)))
            pending = []
            continue
        lm = _LABEL.match(line.strip())
        if lm and not line.startswith((".fn", ".endfn", ".obj", ".endobj")):
            pending.append(lm["name"].strip('"'))
    return insns


def symref(op: str) -> tuple[str, str] | None:
    """`name@rel` (name possibly quoted) -> (name, rel)."""
    m = _SYMREL.match(op)
    return (m["q"] or m["s"], m["rel"]) if m else None


def parse_int(op: str) -> int | None:
    return int(op, 0) if _INT.match(op) else None


def is_reg(op: str) -> bool:
    return bool(_REG.match(op))


def _crfield(op: str) -> str | None:
    m = _CR.match(op)
    return f"cr{m[1]}" if m else None


def _crbit_field(op: str) -> str:
    m = _CRBIT.match(op)
    return f"cr{m['n'] or 0}" if m else "cr0"


def _branch_parts(mnem: str) -> tuple[str, bool, bool]:
    """(mnemonic without the +/- hint, is it to lr, is it to ctr)."""
    base = mnem.rstrip("+-")
    return base, base.endswith("lr") or base.endswith("lrl"), base.endswith("ctr") or base.endswith("ctrl")


# ---------------------------------------------------------------- values

def const(v: int) -> Value:
    return (K, v & MASK)


def is_const(v: Value) -> bool:
    return v[0] == K


def add(v: Value, k: int) -> Value:
    return const(v[1] + k) if v[0] == K else (v[0], v[1] + k)


def describe(v: Value) -> str:
    """Short text for a value, as reports print bindings (this = sp+0x26c)."""
    root, off = v
    tail = f"+{off:#x}" if off > 0 else (f"-{-off:#x}" if off < 0 else "")
    kind = root[0]
    if kind == "k":
        s = off if off < 0x80000000 else off - (1 << 32)
        return f"{s:#x}" if abs(s) > 9 else str(s)
    if kind == "sp":
        return "sp" + tail
    if kind == "arg":
        return f"entry {root[1]}" + tail
    if kind in ("sym", "ha"):
        return ("&" if kind == "sym" else "ha ") + root[1] + tail
    if kind == "n":
        return f"result@{root[1]}" + tail
    if kind == "phi":
        return f"phi {root[2]}@{root[1]}" + tail
    if kind == "lit":
        return f"literal {root[1].hex()}" + tail
    if kind == "str":
        return f"string {root[1]!r}" + tail
    return "?" + str(root) + tail


# ---------------------------------------------------------------- numbering

class _State:
    def __init__(self):
        self.val: dict[str, Value] = {}
        self.prod: dict[str, int | None] = {}

    def copy(self) -> "_State":
        s = _State()
        s.val = dict(self.val)
        s.prod = dict(self.prod)
        return s


def _initial() -> _State:
    s = _State()
    for i in range(32):
        s.val[f"r{i}"] = (("arg", f"r{i}"), 0)
        s.val[f"f{i}"] = (("arg", f"f{i}"), 0)
    for i in range(8):
        s.val[f"cr{i}"] = (("arg", f"cr{i}"), 0)
    s.val["r1"] = (SP, 0)
    s.val["ctr"] = (("arg", "ctr"), 0)
    s.val["lr"] = (LR, 0)
    return s


def _unconditional(insn: Insn) -> bool:
    m = insn.mnem.rstrip("+-")
    return m in ("b", "blr", "bctr", "rfi", "ba")


def _branch_target(insn: Insn, local: set[str]) -> str | None:
    m = insn.mnem.rstrip("+-")
    if not m.startswith("b") or m in ("bl", "bla", "bctrl", "blrl") or m.endswith(("lr", "ctr", "lrl")):
        return None
    t = insn.ops[-1].strip('"') if insn.ops else None
    return t if t in local else None


def build(name: str, insns: list[Insn]) -> Flow:
    """Number the values of one function."""
    local = {lab for ins in insns for lab in ins.labels}
    labels = {lab: i for i, ins in enumerate(insns) for lab in ins.labels}
    # backward branches: registers written in [label, branch] are phi at the label
    written: list[set[str]] = [set() for _ in insns]
    sources: dict[str, list[int]] = {}
    for i, ins in enumerate(insns):
        t = _branch_target(ins, local)
        if t is not None:
            sources.setdefault(t, []).append(i)
    nodes: list[Node] = []
    st = _initial()
    fresh: set[str] = set()
    pending: dict[str, list[_State]] = {}
    save_slots: dict[int, str] = {}
    bctr_state: _State | None = None
    back_written: dict[str, set[str]] = {}

    # first pass only to learn which registers each instruction writes (for backward joins)
    for i, ins in enumerate(insns):
        written[i] = _writes(ins)
    for lab, srcs in sources.items():
        j = labels[lab]
        regs: set[str] = set()
        for b in srcs:
            if b >= j:
                for k in range(j, b + 1):
                    regs |= written[k]
        if regs:
            back_written[lab] = regs

    for i, ins in enumerate(insns):
        if ins.labels and i > 0:
            preds: list[_State] = []
            if not _unconditional(insns[i - 1]):
                preds.append(st)
            for lab in ins.labels:
                preds.extend(pending.pop(lab, []))
            if not preds:
                preds = [bctr_state] if bctr_state is not None else [st]
            phi_regs = set().union(*(back_written.get(lab, set()) for lab in ins.labels))
            st = _merge(preds, phi_regs, ins.labels[0])
        node = _number(i, ins, st, fresh, save_slots, local)
        nodes.append(node)
        t = _branch_target(ins, local)
        if t is not None and labels[t] > i:
            pending.setdefault(t, []).append(st.copy())
        if ins.mnem.rstrip("+-") == "bctr":
            bctr_state = st.copy()
    return Flow(name, nodes, labels)


def _merge(preds: list[_State], phi_regs: set[str], label: str) -> _State:
    out = _State()
    first = preds[0]
    for reg, v in first.val.items():
        if reg != "r1" and (reg in phi_regs or any(p.val.get(reg) != v for p in preds[1:])):
            out.val[reg] = (("phi", label, reg), 0)
            out.prod[reg] = None
        else:
            out.val[reg] = v
            prods = {p.prod.get(reg) for p in preds}
            out.prod[reg] = prods.pop() if len(prods) == 1 else None
    return out


def _writes(ins: Insn) -> set[str]:
    """Registers an instruction writes (approximate, for backward joins)."""
    m = ins.mnem.rstrip("+-")
    ops = ins.ops
    if m in ("bl", "bla", "bctrl", "blrl"):
        return set(VOLATILE) | {"r3", "r4", "f1"}
    if m.startswith("b"):
        return {"ctr"} if m.startswith("bd") else set()
    out: set[str] = set()
    if m.startswith(("cmp", "fcmp")):
        out.add(_crfield(ops[0]) or "cr0" if ops else "cr0")
        return out
    if m.startswith("cr"):
        out.add(_crbit_field(ops[0]) if ops else "cr0")
        return out
    if m == "mtctr":
        return {"ctr"}
    if m.startswith("st"):
        if "u" in m[2:] and ops:
            base = _MEM.match(ops[1]) if len(ops) > 1 else None
            out.add(base["base"] if base else ops[1] if len(ops) > 1 else "")
    elif ops and is_reg(ops[0]):
        out.add(ops[0])
        if m.startswith("l") and "u" in m[1:] and len(ops) > 1:
            base = _MEM.match(ops[1])
            out.add(base["base"] if base else ops[1])
    if m.endswith("."):
        out.add("cr0")
    return out


def _number(i: int, ins: Insn, st: _State, fresh: set[str], save_slots: dict[int, str], local: set[str]) -> Node:
    m = ins.mnem
    base_m, to_lr, to_ctr = _branch_parts(m)
    ops = ins.ops
    deps: list[int] = []
    outs: dict[str, Value] = {}

    def rd(reg: str) -> Value:
        p = st.prod.get(reg)
        if p is not None:
            deps.append(p)
        return st.val[reg]

    def wr(reg: str, v: Value) -> None:
        st.val[reg] = v
        st.prod[reg] = i
        outs[reg] = v
        if reg in ARG_REGS:
            fresh.add(reg)

    def node(kind: str, key: str = "", ins_: tuple = (), **kw) -> Node:
        return Node(i, ins, kind, key, ins_, tuple(dict.fromkeys(deps)), outs, **kw)

    def result(slot: str = "") -> Value:
        return (("n", i, slot), 0)

    def operand(op: str) -> Value:
        if is_reg(op):
            return rd(op)
        n = parse_int(op)
        if n is not None:
            return const(n)
        cr = _crfield(op)
        if cr:
            return rd(cr)
        sr = symref(op)
        if sr:
            return (("sym", sr[0]), 0)
        return (("tok", op), 0)

    def address(op: str) -> Value:
        mm = _MEM.match(op)
        if not mm:
            return operand(op)
        disp, base = mm["disp"], mm["base"]
        sr = symref(disp)
        if sr and sr[1] in ("sda21", "sda2"):
            return (("sym", sr[0]), 0)
        if sr:            # sym@l(rX): rX holds sym@ha
            rd(base)
            return (("sym", sr[0]), 0)
        d = parse_int(disp) or 0
        if base == "r0":
            return const(d)
        return add(rd(base), d)

    # ---- frame: anything that writes r1, the LR dance, frame helpers, blr
    if m in ("stwu", "stwux") and ops and ops[0] == "r1" or (ops and ops[0] == "r1" and not m.startswith("st")
                                                            and not m.startswith(("cmp", "b", "tw"))):
        return node("frame")
    if m == "mflr":
        wr(ops[0], (LR, 0))
        return node("frame")
    if m == "mtlr":
        st.val["lr"] = rd(ops[0])
        return node("frame")
    if base_m == "blr":
        return node("frame")
    if m in ("stmw", "lmw"):
        return node("frame")
    if m == "bl" and ops and ops[0].strip('"').startswith(FRAME_HELPERS):
        rd("r11")                     # addi r11, r1, N before it belongs to the frame too
        return node("frame")
    if m in ("nop", "sync", "isync"):
        return node("skip", m)

    # ---- copies
    if m in ("mr", "fmr", "ps_mr"):
        wr(ops[0], rd(ops[1]))
        return node("copy")
    if m == "mtctr":
        wr("ctr", rd(ops[0]))
        return node("copy")
    if m == "mfctr":
        wr(ops[0], rd("ctr"))
        return node("copy")

    # ---- pure address / constant arithmetic
    if m == "li":
        sr = symref(ops[1])
        wr(ops[0], (("sym", sr[0]), 0) if sr else const(parse_int(ops[1]) or 0))
        return node("pure")
    if m == "lis":
        sr = symref(ops[1])
        wr(ops[0], (("ha", sr[0]), 0) if sr else const((parse_int(ops[1]) or 0) << 16))
        return node("pure")
    if m in ("addi", "subi", "addis", "subis"):
        sr = symref(ops[2])
        if sr:
            if ops[1] != "r0":
                rd(ops[1])
            wr(ops[0], (("sym", sr[0]), 0))
            return node("pure")
        k = parse_int(ops[2]) or 0
        if m in ("subi", "subis"):
            k = -k
        if m in ("addis", "subis"):
            k <<= 16
        base = const(0) if ops[1] == "r0" else rd(ops[1])
        wr(ops[0], add(base, k))
        return node("pure")
    if m in ("ori", "oris") and len(ops) == 3:
        sr = symref(ops[2])
        src = rd(ops[1])
        if sr and src[0] == ("ha", sr[0]):
            wr(ops[0], (("sym", sr[0]), 0))
            return node("pure")
        k = parse_int(ops[2])
        if k is not None and is_const(src):
            wr(ops[0], const(src[1] | (k << (16 if m == "oris" else 0))))
            return node("pure")

    # ---- calls
    if m in ("bl", "bla", "bctrl", "blrl") or (base_m == "b" and ops and ops[0].strip('"') not in local):
        args = tuple((r, st.val[r]) for r in ARG_REGS if r in fresh)
        regs = {r: st.val[r] for r in ARG_REGS}
        for r, _ in args:
            rd(r)
        if m == "bctrl":
            key, ins_ = "bctrl", (rd("ctr"),)
        elif m == "blrl":
            key, ins_ = "blrl", (rd("lr"),)
        else:
            key, ins_ = f"bl {ops[0].strip(chr(34))}", ()
        n = node("call", key, ins_, args=args, regs=regs)
        for r in VOLATILE:
            st.val[r] = (("x", i, r), 0)
            st.prod[r] = None
        for r in ("r3", "r4", "f1"):
            st.val[r] = result(r)
            st.prod[r] = i
            outs[r] = result(r)
        fresh.clear()
        return n

    # ---- branches
    if base_m == "b":
        return node("jump", "b", target=ops[0].strip('"') if ops else None)
    if base_m in ("bctr",):
        return node("op", "bctr", (rd("ctr"),))
    if base_m.startswith("b"):
        target = None if (to_lr or to_ctr) else (ops[-1].strip('"') if ops else None)
        if base_m.startswith("bd"):
            v = rd("ctr")
            n = node("branch", base_m, (v,), target=target)
            wr("ctr", result("ctr"))
            return n
        field_ = next((f for f in (_crfield(o) for o in ops[:-1] if not to_lr) if f), None)
        if to_lr and ops:
            field_ = _crfield(ops[0])
        v = rd(field_ or "cr0")
        return node("branch", base_m, (v,), target=target)

    # ---- compares and condition-register logic
    if m.startswith(("cmp", "fcmp")):
        field_ = _crfield(ops[0]) if ops else None
        rest = ops[1:] if field_ else ops
        n = node("op", m, tuple(operand(o) for o in rest))
        wr(field_ or "cr0", result())
        return n
    if m.startswith("cr") and m not in ("cror", "crand", "crxor", "crnor", "creqv", "crandc", "crorc", "crnand"):
        # crclr/crset: a constant bit (cr1eq before varargs calls)
        n = node("op", f"{m} {','.join(ops)}")
        wr(_crbit_field(ops[0]) if ops else "cr0", result())
        return n
    if m.startswith("cr"):
        fields = list(dict.fromkeys(_crbit_field(o) for o in ops[1:]))
        n = node("op", f"{m} {','.join(ops)}", tuple(rd(f) for f in fields))
        wr(_crbit_field(ops[0]), result())
        return n

    # ---- record-form copy: mr. is a copy plus a compare with zero
    if m == "mr.":
        v = rd(ops[1])
        n = node("op", "mr.", (v,))
        wr(ops[0], v)
        wr("cr0", result("cr0"))
        return n

    # ---- loads and stores
    if m.startswith("psq_"):
        addr = address(ops[1])
        key = f"{m} {ops[2]},{ops[3]}" if len(ops) > 3 else m
        if m.startswith("psq_st"):
            v = rd(ops[0])
            if addr[0] == SP and v[0] == ("arg", ops[0]) and ops[0] in CALLEE_SAVED:
                save_slots[addr[1]] = ops[0]
                return node("frame")
            return node("op", key, (v, addr), mem=addr, store=True)
        if addr[0] == SP and save_slots.get(addr[1]) == ops[0]:
            return node("frame")
        n = node("op", key, (addr,), mem=addr)
        wr(ops[0], result())
        return n
    if _is_store(m):
        indexed = m.endswith("x")
        v = rd(ops[0])
        if indexed:
            a, b = rd(ops[1]), rd(ops[2])
            n = node("op", m, (v, a, b), store=True)
            if "u" in m[2:]:
                wr(ops[1], result("ea"))
            return n
        addr = address(ops[1])
        if addr[0] == SP and (v[0] == ("arg", ops[0]) and ops[0] in CALLEE_SAVED or v[0] == LR):
            save_slots[addr[1]] = ops[0] if v[0] != LR else "lr"
            return node("frame")
        n = node("op", m, (v, addr), mem=addr, store=True)
        if "u" in m[2:]:
            mm = _MEM.match(ops[1])
            if mm:
                wr(mm["base"], addr)
        return n
    if _is_load(m):
        if m.endswith("x"):
            a, b = rd(ops[1]), rd(ops[2])
            n = node("op", m, (a, b))
            wr(ops[0], result())
            if "u" in m[1:]:
                wr(ops[1], result("ea"))
            return n
        addr = address(ops[1])
        slot = save_slots.get(addr[1]) if addr[0] == SP else None
        if slot is not None and (slot == ops[0] or slot == "lr" and ops[0] == "r0"):
            st.val[ops[0]] = (("arg", ops[0]), 0) if slot != "lr" else (LR, 0)
            return node("frame")
        n = node("op", m, (addr,), mem=addr)
        wr(ops[0], result())
        if "u" in m[1:]:
            mm = _MEM.match(ops[1])
            if mm:
                wr(mm["base"], addr)
        return n

    # ---- special registers
    if m.startswith("mt") and m != "mtcrf":
        return node("op", m, tuple(rd(o) for o in ops if is_reg(o)))
    if m.startswith("mf"):
        n = node("op", m + (" " + ",".join(ops[1:]) if len(ops) > 1 else ""))
        if ops and is_reg(ops[0]):
            wr(ops[0], result())
        return n

    # ---- everything else: ALU / FPU, destination first (traps have none)
    if not ops or not is_reg(ops[0]) or m.startswith("tw"):
        return node("op", m, tuple(operand(o) for o in ops))
    srcs = list(ops[1:])
    ins_ = [operand(o) for o in srcs]
    if m.startswith("rlwimi"):
        ins_.insert(0, rd(ops[0]))
    n = node("op", m, tuple(ins_))
    wr(ops[0], result())
    if m.endswith("."):
        wr("cr0", result("cr0"))
    return n


def _is_store(m: str) -> bool:
    return m.startswith("st") and m not in ("stwcx.",)


def _is_load(m: str) -> bool:
    return m[:2] in ("lw", "lb", "lh", "lf") and m not in ("lmw",)


# ---------------------------------------------------------------- function-level helpers

def flow_of(name: str, text: str) -> Flow:
    return build(name, parse_insns(text))


def consumers(flow: Flow) -> list[list[int]]:
    """Reverse of Node.deps."""
    out: list[list[int]] = [[] for _ in flow.nodes]
    for n in flow.nodes:
        for d in n.deps:
            out[d].append(n.i)
    return out


def frame_closure(flow: Flow) -> set[int]:
    """Frame nodes plus pure/copy nodes that only feed the frame (addi r11, r1, N before _savegpr)."""
    cons = consumers(flow)
    frame = {n.i for n in flow.nodes if n.kind == "frame"}
    for n in reversed(flow.nodes):
        if n.kind in ("pure", "copy") and cons[n.i] and all(c in frame for c in cons[n.i]):
            frame.add(n.i)
    return frame
