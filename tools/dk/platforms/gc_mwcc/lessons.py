"""dk dataset lessons: compiler-verified minimal pairs from the portable playbook's rules.

  ./dk dataset lessons [--rules ID,..] [--compilers GC/1.3,GC/1.3.2] [--jobs 2]   build records/lessons.jsonl
  ./dk dataset lessons --list                                                    the rules and their sources
  ./dk dataset lessons --show ID                                                 one rule's first pair, both asms

Each lesson (LESSONS) is one "asm symptom -> source idiom" rule of playbook/matching.md, the
CodeWarrior platform notes or dk close's rewrite table (rewrites.py, the same knowledge as rows the
beam search applies), with small, natural, generic C++ in both shapes: the idiom (right) and the
natural alternative (wrong), each a template expanded over several parameter sets (other names,
types, sizes and constants, so a model learns the rule rather than the snippet). Both shapes are
compiled with each compiler the rule is stated for (wibo + mwcceppc with --flags, the game set by
default, and the -inline setting the rule depends on), privately: nothing of the project is
included, so the records are portable CodeWarrior GC/Wii knowledge. A pair is kept only when the
two shapes compile to different code AND the difference is the one the rule names (its `check`,
on the two instruction lists); the record is a repair in llm-try's format: the right shape's asm
as the target, the wrong shape as the current definition with its score and diff, the right shape
as the answer (and, for a symmetric rule, the reverse direction too). Records carry source=lessons,
rule=<id> and the rule's symptom and idiom (a one-line diagnosis for round 2). A rule with no kept
pair did not reproduce in toys: a finding to check, never a reason to edit the playbook unseen.

Records go to llm-data/dataset/records/lessons.jsonl; `dataset export` reads them only when asked
(`--sources ...,lessons`), so the default export is unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Callable

from ...core.command import OK, USAGE
from ...core.config import Config

# the game flag set (CodeWarrior GC defaults as games used them; -inline comes from the lesson)
FLAGS = ("-nodefaults -proc gekko -align powerpc -enum int -fp hardware -Cpp_exceptions off -O4,p "
         "-pragma \"cats off\" -pragma \"warn_notinlined off\" -maxerrors 1 -nosyspath -str reuse -RTTI on "
         "-fp_contract off -lang=c++")
COMPILERS = ("GC/1.3", "GC/1.3.2")
SOURCE = "lessons"

Insn = tuple[str, str]                     # (mnemonic, operands) with branch targets and labels abstracted


# ---- the asm checks ---------------------------------------------------------------------------------

_LINE = re.compile(r"^\s*[0-9A-F]+  (\S+)\s*(.*)$")
_REGS = re.compile(r"\b[rf]\d+\b")


def insns(asm: str) -> list[Insn]:
    """compact_asm text -> instructions; branch targets and local labels become `L`."""
    out = []
    for line in asm.splitlines():
        m = _LINE.match(line)
        if m:
            ops = re.sub(r"\.L_[0-9A-Fa-f]+|\blbl_[0-9A-Fa-f]+\b", "L", m[2].strip())
            out.append((m[1], ops))
    return out


def mn(xs: list[Insn]) -> list[str]:
    return [m for m, _ in xs]


def count(xs: list[Insn], *names: str) -> int:
    return sum(1 for m, _ in xs if m in names)


def text(xs: list[Insn]) -> str:
    return "\n".join(f"{m} {o}" for m, o in xs)


def only_registers(r: list[Insn], w: list[Insn]) -> bool:
    """The same instructions in the same order, only register numbers differ."""
    return r != w and len(r) == len(w) and all(a[0] == b[0] and _REGS.sub("R", a[1]) == _REGS.sub("R", b[1])
                                               for a, b in zip(r, w))


def reordered(r: list[Insn], w: list[Insn], names: tuple[str, ...] = ()) -> bool:
    """The same instructions (registers aside), in another order; `names`: among those mnemonics."""
    norm = lambda xs: [(m, _REGS.sub("R", o)) for m, o in xs]
    a, b = norm(r), norm(w)
    if sorted(a) != sorted(b) or a == b:
        return False
    return not names or [x for x in a if x[0] in names] != [x for x in b if x[0] in names]


def swapped(r: list[Insn], w: list[Insn], names: tuple[str, ...]) -> bool:
    """One of `names` takes its two source operands the other way round (the rest may move with it)."""
    for (m1, o1), (m2, o2) in zip(r, w):
        if m1 == m2 and m1 in names and o1 != o2:
            a, b = [x.strip() for x in o1.split(",")], [x.strip() for x in o2.split(",")]
            if len(a) == len(b) == 3 and a[1:] == b[:0:-1]:
                return True
    return False


_NO_DEST = ("st", "cmp", "fcmp", "b", "mt", "dcb", "icb", "sync", "isync")


def roles(xs: list[Insn], names: tuple[str, ...]) -> list[tuple[str, ...]]:
    """For each instruction in `names`, where its source operands come from: the defining instruction
    (registers abstracted: `lfs 0x0(R)`, `lfs L@sda21(R)`, `fmuls`) or `arg <reg>` for an incoming value.
    Two shapes whose loads land in other registers then compare by what each operand *is*."""
    defs: dict[str, str] = {}
    out = []
    for m, o in xs:
        ops = [x.strip() for x in o.split(",")] if o else []
        regs = [x for x in ops if re.fullmatch(r"[rf]\d+", x)]
        has_dest = bool(regs) and not m.startswith(_NO_DEST) and ops and ops[0] == regs[0]
        srcs = regs[1:] if has_dest else regs
        if m in names:
            out.append(tuple(defs.get(r, f"arg {r}") for r in srcs))
        if has_dest:
            defs[regs[0]] = (m + " " + _REGS.sub("R", ", ".join(ops[1:]))).strip() if m.startswith("l") else m
    return out


def roles_swapped(r: list[Insn], w: list[Insn], names: tuple[str, ...]) -> bool:
    """An instruction of `names` takes the same operands the other way round (by role, see roles)."""
    return any(a != b and sorted(a) == sorted(b) for a, b in zip(roles(r, names), roles(w, names)))


def before(xs: list[Insn], first: str, then: str) -> bool:
    """A call to `first` comes before a call to `then`."""
    pos = lambda c: next((i for i, (m, o) in enumerate(xs) if m == "bl" and o.startswith(c)), -1)
    return 0 <= pos(first) < pos(then)


def more(r: list[Insn], w: list[Insn], *names: str) -> bool:
    """The right shape has more of these mnemonics than the wrong one."""
    return count(r, *names) > count(w, *names)


def calls(xs: list[Insn], callee: str) -> bool:
    return any(m == "bl" and o.startswith(callee) for m, o in xs)


# ---- the lessons -------------------------------------------------------------------------------------

@dataclass
class Shape:
    decls: str                             # generic declarations (no includes), a Template
    right: str                             # the idiom's definition, a Template
    wrong: str                             # the natural alternative, a Template


@dataclass
class Lesson:
    id: str
    ref: str                               # where the rule is stated
    symptom: str
    idiom: str
    check: Callable[[list[Insn], list[Insn]], bool]     # (right asm, wrong asm) -> the rule's difference
    shapes: list[Shape]
    params: list[dict]                     # each a parameter set for every shape ($F is the function name)
    compilers: tuple[str, ...] = COMPILERS
    inline: str = "auto"                   # the -inline setting the rule holds under
    both: bool = True                      # the reverse direction is a valid repair too


FN = ("Update", "Apply", "Reset", "Scale", "Blend", "Count", "Pick", "Step", "Fill", "Setup", "Mix", "Advance",
      "Select", "Merge", "Limit", "Adjust", "Store", "Refresh", "Measure", "Combine")


def grid(n: int, salt: str, **pools) -> list[dict]:
    """n parameter sets: pool values cycled with co-prime strides (deterministic), $F from FN."""
    keys = sorted(pools)
    start = int(hashlib.sha1(salt.encode()).hexdigest()[:6], 16)
    out = []
    for i in range(n):
        p = {k: pools[k][(start + i * (j + 1)) % len(pools[k])] for j, k in enumerate(keys)}
        p.setdefault("F", FN[(start + 7 * i) % len(FN)])
        out.append({**{k: (v(p) if callable(v) else v) for k, v in p.items()}})
    return out


STRUCT = ("Item", "Node", "Entry", "Slot", "Track", "Light", "Sample", "Timer", "Cell", "Body", "Link", "Agent")
INT_T = ("int", "short", "long")
TRIPLE = (("x", "y", "z"), ("mA", "mB", "mC"), ("count", "flags", "state"), ("left", "top", "right"),
          ("min", "mid", "max"), ("r", "g", "b"))
KS = ("1", "2", "3", "5", "7", "8", "12", "16", "100", "255")
FKS = ("0.5f", "2.0f", "3.0f", "0.25f", "10.0f", "1.5f", "4.0f", "0.75f")


def _triples(**extra) -> dict:
    return {"T3": TRIPLE, **extra}


def expand(p: dict) -> dict:
    """$a $b $c from a triple, and $T3 dropped."""
    out = dict(p)
    if "T3" in out:
        out["a"], out["b"], out["c"] = out.pop("T3")
    return out


def _l(*a, **k) -> Lesson:
    les = Lesson(*a, **k)
    les.params = [expand(p) for p in les.params]
    return les


LESSONS: list[Lesson] = [
    _l("chained-assign", "matching.md: Stores in reverse order (z, y, x); rewrites.py chained-assign",
       "Stores in reverse order (z, y, x)", "chained assignment `a = b = c = K`, which stores the last one first",
       lambda r, w: reordered(r, w, ("stw", "sth", "stb", "stfs")),
       [Shape("struct $S { $T $a; $T $b; $T $c; };\n",
              "void $F($S* p) {\n    p->$a = p->$b = p->$c = $K;\n}",
              "void $F($S* p) {\n    p->$a = $K;\n    p->$b = $K;\n    p->$c = $K;\n}"),
        Shape("struct $S { $T $a; $T $b; $T $c; int other; };\n",
              "void $F($S* p, int v) {\n    p->other = v;\n    p->$c = p->$b = p->$a = $K;\n}",
              "void $F($S* p, int v) {\n    p->other = v;\n    p->$c = $K;\n    p->$b = $K;\n    p->$a = $K;\n}")],
       grid(8, "chained", S=STRUCT, T=("int", "short", "unsigned char", "float", "long"), K=("0", "1", "4", "16"),
            **_triples())),
    _l("operand-swap", "gamecube-mwcc.md: Between plain operands, source order sets a commutative instruction's "
       "operand order; rewrites.py operand-swap",
       "A commutative instruction's operands the other way round (`add`, `mullw`, `and`)",
       "swap the plain operands (`x + y` vs `y + x`): source order sets the operand order",
       lambda r, w: roles_swapped(r, w, ("add", "mullw", "and", "or", "xor", "fadds", "fmuls")),
       [Shape("", "$T $F($T $a, $T $b) {\n    return $a $op $b;\n}", "$T $F($T $a, $T $b) {\n    return $b $op $a;\n}"),
        Shape("struct $S { $T $a; $T $b; };\n", "$T $F(const $S* p, $T $c) {\n    return $c $op p->$a;\n}",
              "$T $F(const $S* p, $T $c) {\n    return p->$a $op $c;\n}")],
       grid(8, "opswap", S=STRUCT, T=("int", "unsigned int"), op=("+", "*", "&", "|", "^"), **_triples())
       + grid(3, "opswapf", S=STRUCT, T=("float",), op=("+", "*"), **_triples())),
    _l("op-assign", "matching.md: `fmuls` operand order: `q * 2.0f` puts the literal first; `x *= K` the variable",
       "`fmuls` operand order: constant first vs variable first",
       "`x *= K` puts the variable first; `x = K * x` the constant",
       lambda r, w: roles_swapped(r, w, ("fmuls", "fadds")),
       [Shape("struct $S { float $a; float $b; };\n", "void $F($S* p) {\n    p->$a *= $K;\n}",
              "void $F($S* p) {\n    p->$a = $K * p->$a;\n}"),
        Shape("", "float $F(float $a, float $b) {\n    $a *= $K;\n    return $a + $b;\n}",
              "float $F(float $a, float $b) {\n    $a = $K * $a;\n    return $a + $b;\n}")],
       grid(8, "opassign", S=STRUCT, K=("3.0f", "0.75f", "1.5f", "10.0f", "0.1f"), **_triples())),
    _l("compare-flip", "matching.md: `fcmpu f, 0` vs `fcmpu 0, f`: `if (f)` vs `f != 0.0f`; rewrites.py compare-flip",
       "A compare's operands the other way round (`cmpw r3,r4` vs `cmpw r4,r3`; `fcmpu f, 0` vs `fcmpu 0, f`)",
       "`a < b` vs `b > a` (the operand written first is compared first); `if (f)` vs `f != 0.0f`",
       lambda r, w: roles_swapped(r, w, ("cmpw", "cmplw", "fcmpu", "fcmpo")),
       [Shape("", "int $F($T $a, $T $b) {\n    if ($a < $b) {\n        return $K;\n    }\n    return 0;\n}",
              "int $F($T $a, $T $b) {\n    if ($b > $a) {\n        return $K;\n    }\n    return 0;\n}"),
        Shape("struct $S { float $a; int $b; };\n",
              "void $F($S* p) {\n    if (p->$a) {\n        p->$b = $K;\n    }\n}",
              "void $F($S* p) {\n    if (p->$a != 0.0f) {\n        p->$b = $K;\n    }\n}"),
        Shape("struct $S { int $a; int $b; int $c; };\n",
              "int $F(const $S* p, const $S* q) {\n    if (p->$a < q->$b) {\n        return $K;\n    }\n    return p->$c;\n}",
              "int $F(const $S* p, const $S* q) {\n    if (q->$b > p->$a) {\n        return $K;\n    }\n    return p->$c;\n}")],
       grid(8, "cmpflip", S=STRUCT, T=("int", "unsigned int", "float"), K=KS, **_triples())),
    _l("half", "matching.md: `x * 0.5f` with `x` loaded before the constant -> `x / 2.0f`; rewrites.py half",
       "`x * 0.5f` with `x` loaded before the constant", "`x / 2.0f` (a literal `x * 0.5f` loads the constant first)",
       lambda r, w: roles_swapped(r, w, ("fmuls",)),
       [Shape("struct $S { float $a; float $b; };\n", "float $F(const $S* p) {\n    return p->$a / $D;\n}",
              "float $F(const $S* p) {\n    return p->$a * $H;\n}"),
        Shape("struct $S { float $a; float $b; };\n", "void $F($S* p) {\n    p->$b = (p->$a + p->$b) / $D;\n}",
              "void $F($S* p) {\n    p->$b = (p->$a + p->$b) * $H;\n}")],
       [{**p, "H": h} for p, h in zip(grid(8, "half", S=STRUCT, D=("2.0f", "4.0f", "8.0f"), **_triples()),
                                       ("0.5f", "0.25f", "0.125f", "0.5f", "0.25f", "0.125f"))]),
    _l("reassociate", "matching.md: `addi; add` where ours reassociates to `add; addi` -> plain `a + b + K`",
       "`addi; add` where ours reassociates to `add; addi`", "plain `a + b + K`, not `a + (b + K)`",
       lambda r, w: reordered(r, w) or (r != w and count(r, "add") == count(w, "add")),
       [Shape("", "int $F(int $a, int $b) {\n    return $a + $b + $K;\n}",
              "int $F(int $a, int $b) {\n    return $a + ($b + $K);\n}"),
        Shape("struct $S { int $a; int $b; int $c; };\n", "void $F($S* p, int n) {\n    p->$c = p->$a + n + $K;\n}",
              "void $F($S* p, int n) {\n    p->$c = p->$a + (n + $K);\n}")],
       grid(8, "reassoc", S=STRUCT, K=("4", "8", "12", "0x10", "100", "3"), **_triples())),
    _l("ternary-min", "matching.md: `bge; b; fmr` vs `cror` for a min/max: a ternary gives the branches, a plain "
       "`if` gives `cror`; rewrites.py ternary-if",
       "`bge; b; fmr` vs `cror` for a min/max", "a ternary `t < 0 ? t : 0` gives the branches; a plain `if` gives `cror`",
       lambda r, w: count(r, "cror") != count(w, "cror") or count(r, "fmr") != count(w, "fmr"),
       [Shape("", "float $F(float $a, float $b) {\n    return $a < $b ? $a : $b;\n}",
              "float $F(float $a, float $b) {\n    if ($a < $b) {\n        return $a;\n    }\n    return $b;\n}"),
        Shape("struct $S { float $a; float $b; };\n",
              "void $F($S* p, float v) {\n    p->$a = v > $K ? v : $K;\n}",
              "void $F($S* p, float v) {\n    if (v <= $K) {\n        v = $K;\n    }\n    p->$a = v;\n}")],
       grid(8, "tmin", S=STRUCT, K=("0.0f", "1.0f", "0.5f", "-1.0f"), **_triples())),
    _l("ternary-register", "matching.md: A min whose result gets its own saved register -> a ternary; `r = b; "
       "if (a < b) r = a;` lets `r` share a register",
       "A min/max result in its own register (`mr r6, rB; bge; mr r6, rA`) where ours reuses an operand's",
       "a ternary `a < b ? a : b` (a new value); `r = b; if (a < b) r = a;` lets `r` share a register",
       lambda r, w: r != w and (count(r, "mr") != count(w, "mr") or only_registers(r, w)),
       [Shape("extern void Use(int);\n",
              "void $F(int $a, int $b) {\n    int m = $a < $b ? $a : $b;\n    Use(m);\n    Use($a + $b);\n}",
              "void $F(int $a, int $b) {\n    int m = $b;\n    if ($a < $b) {\n        m = $a;\n    }\n    Use(m);\n"
              "    Use($a + $b);\n}")],
       grid(8, "treg", **_triples())),
    _l("one-case-switch", "matching.md: `cmpwi; beq case; b default` where ours gives `bne` -> a one-case switch",
       "`cmpwi; beq case; b default` where ours gives `bne`", "a `switch` with one `case`, not an `if`",
       lambda r, w: r != w and (count(r, "b") > count(w, "b") or mn(r) != mn(w)),
       [Shape("struct $S { int $a; int $b; };\nextern void Notify(int);\n",
              "void $F($S* p) {\n    switch (p->$a) {\n    case $K:\n        p->$b = 0;\n        Notify(p->$a);\n"
              "        break;\n    }\n}",
              "void $F($S* p) {\n    if (p->$a == $K) {\n        p->$b = 0;\n        Notify(p->$a);\n    }\n}")],
       grid(8, "onecase", S=STRUCT, K=KS, **_triples())),
    _l("or-return", "matching.md: `if (!a || !b) return;` shape: `bne next; b exit` -> exactly that `||` early return",
       "`if (!a || !b) return;` shape: `bne next; b exit`", "exactly that `||` early return (not two separate `if`s)",
       lambda r, w: r != w and mn(r) != mn(w),
       [Shape("struct $S { int $a; int $b; int $c; };\nextern void Run($S*);\n",
              "void $F($S* p) {\n    if (!p->$a || !p->$b) {\n        return;\n    }\n    p->$c = $K;\n    Run(p);\n}",
              "void $F($S* p) {\n    if (!p->$a) {\n        return;\n    }\n    if (!p->$b) {\n        return;\n    }\n"
              "    p->$c = $K;\n    Run(p);\n}")],
       grid(8, "orret", S=STRUCT, K=KS, **_triples())),
    _l("if-invert", "rewrites.py if-invert: a conditional branch inverted with the same blocks",
       "A conditional branch inverted, the same two blocks the other way round",
       "`if (!c) B else A` for `if (c) A else B`: blocks are laid out in source order",
       lambda r, w: r != w and sorted(mn(r)) != sorted(mn(w)) or reordered(r, w),
       [Shape("struct $S { int $a; int $b; int $c; };\nextern void Open($S*);\nextern void Close($S*, int);\n",
              "void $F($S* p) {\n    if (p->$a != 0) {\n        Open(p);\n    } else {\n        Close(p, $K);\n    }\n}",
              "void $F($S* p) {\n    if (p->$a == 0) {\n        Close(p, $K);\n    } else {\n        Open(p);\n    }\n}")],
       grid(8, "ifinv", S=STRUCT, K=KS, **_triples())),
    _l("store-order", "matching.md: Stores to one object in another offset order -> write the assignments in the "
       "target's order (MWCC keeps stores through one base in source order); rewrites.py statement-order",
       "Stores to one object in another offset order than ours (`stw 0x8`, `stw 0x4`)",
       "write the assignments in the target's order: stores through one base stay in source order",
       lambda r, w: reordered(r, w, ("stw", "sth", "stb", "stfs")),
       [Shape("struct $S { $T $a; $T $b; $T $c; };\n",
              "void $F($S* p, $T u, $T v) {\n    p->$a = u;\n    p->$c = v;\n    p->$b = $K;\n}",
              "void $F($S* p, $T u, $T v) {\n    p->$a = u;\n    p->$b = $K;\n    p->$c = v;\n}"),
        Shape("struct $S { $T $a; $T $b; $T $c; };\n",
              "void $F($S* p) {\n    p->$b = 0;\n    p->$a = $K;\n    p->$c = $K + 1;\n}",
              "void $F($S* p) {\n    p->$a = $K;\n    p->$b = 0;\n    p->$c = $K + 1;\n}")],
       grid(8, "storeord", S=STRUCT, T=("int", "short", "unsigned char"), K=KS, **_triples())),
    _l("decl-order", "gamecube-mwcc.md: Register allocation (named locals rank by declaration order); "
       "rewrites.py decl-order",
       "Callee-saved register numbers swapped between two locals",
       "declare the locals in the rank order: the first declared gets the higher register (r31)",
       only_registers,
       [Shape("extern int First(int);\nextern int Second(int);\nextern void Flush();\n",
              "int $F(int n) {\n    int $a;\n    int $b;\n    $a = First(n);\n    $b = Second(n);\n    Flush();\n"
              "    return $a - $b;\n}",
              "int $F(int n) {\n    int $b;\n    int $a;\n    $a = First(n);\n    $b = Second(n);\n    Flush();\n"
              "    return $a - $b;\n}"),
        Shape("struct $S { int $a; int $b; };\nextern void Touch($S*);\n",
              "int $F($S* p, $S* q) {\n    int $a = p->$a;\n    int $b = q->$b;\n    Touch(p);\n    Touch(q);\n"
              "    return $a * $b;\n}",
              "int $F($S* p, $S* q) {\n    int $b = q->$b;\n    int $a = p->$a;\n    Touch(p);\n    Touch(q);\n"
              "    return $a * $b;\n}")],
       grid(8, "declord", S=STRUCT, **_triples())),
    _l("shared-loop-index", "matching.md: Saved registers differ across consecutive loops -> one shared `int i`",
       "Saved registers differ across consecutive loops", "declare one shared `int i` for all the loops",
       lambda r, w: r != w,
       [Shape("struct $S { int $a; int $b; };\nextern void Visit($S*, int);\nextern void Leave($S*, int);\n",
              "void $F($S* p, int n) {\n    int i;\n    for (i = 0; i < n; i++) {\n        Visit(p, i);\n    }\n"
              "    for (i = 0; i < $K; i++) {\n        Leave(p, i);\n    }\n}",
              "void $F($S* p, int n) {\n    for (int i = 0; i < n; i++) {\n        Visit(p, i);\n    }\n"
              "    for (int j = 0; j < $K; j++) {\n        Leave(p, j);\n    }\n}"),
        Shape("struct $S { int $a; int $b; };\nextern int Get($S*, int);\nextern void Put($S*, int, int);\n",
              "void $F($S* p, int n) {\n    int i;\n    int sum = 0;\n    for (i = 0; i < n; i++) {\n        sum += Get(p, i);\n"
              "    }\n    for (i = 0; i < n; i++) {\n        Put(p, i, sum);\n    }\n}",
              "void $F($S* p, int n) {\n    int sum = 0;\n    for (int i = 0; i < n; i++) {\n        sum += Get(p, i);\n"
              "    }\n    for (int i = 0; i < n; i++) {\n        Put(p, i, sum);\n    }\n}")],
       grid(8, "sharedi", S=STRUCT, K=("4", "8", "3", "16", "10"), **_triples())),
    _l("unsigned-index", "gamecube-mwcc.md: A loop index that is a signed counter converted to unsigned is "
       "strength-reduced to an offset (`addi r0, rOff, K; lwzx`), not a pointer; rewrites.py unsigned-index",
       "`addi r0, rOff, K; lwzx` vs a stepped pointer in a loop",
       "an accessor with an `unsigned int` index called with an `int` counter gives the offset form; an "
       "`unsigned int` counter gives a stepped pointer",
       lambda r, w: count(r, "lwzx", "lhzx", "lfsx", "lbzx") != count(w, "lwzx", "lhzx", "lfsx", "lbzx"),
       [Shape("struct $S {\n    $T m[$N];\n    $T Get(unsigned int i) const { return m[i]; }\n};\n",
              "$T $F(const $S* p, int n) {\n    $T s = 0;\n    for (int i = 0; i < n; i++) {\n        s += p->Get(i);\n"
              "    }\n    return s;\n}",
              "$T $F(const $S* p, int n) {\n    $T s = 0;\n    for (unsigned int i = 0; i < n; i++) {\n"
              "        s += p->Get(i);\n    }\n    return s;\n}")],
       grid(8, "uidx", S=STRUCT, T=("int", "float", "short"), N=("8", "16", "32", "4"))),
    _l("ref-element", "matching.md: `mulli; addi; lfsx` on an array element -> the element read through a "
       "reference `T& e = a[i];`; rewrites.py ref-element",
       "`mulli; addi; lfsx` on an array element", "the element read through a reference: `T& e = a[i];`",
       lambda r, w: r != w and (count(r, "lfsx", "lwzx", "mulli") != count(w, "lfsx", "lwzx", "mulli")
                                or mn(r) != mn(w)),
       [Shape("struct $S { float $a; float $b; float $c; };\nstruct Table { $S e[$N]; };\n",
              "float $F(Table* t, int i) {\n    $S& e = t->e[i];\n    return e.$a * e.$b + e.$c;\n}",
              "float $F(Table* t, int i) {\n    return t->e[i].$a * t->e[i].$b + t->e[i].$c;\n}")],
       grid(8, "refel", S=STRUCT, N=("4", "8", "16", "12"), **_triples())),
    _l("flag-form", "matching.md: The target's `bool` flag starts in the register of the block's other zero stores "
       "-> `bool b = a || c;`; rewrites.py flag-form",
       "A `bool` flag in its own zero register where the target shares the block's `li r30, 0`",
       "`bool b = a || c;`, not `bool b = false; if (a || c) b = true;`",
       lambda r, w: r != w,
       [Shape("struct $S { int $a; int $b; int $c; };\nextern void Report(bool);\n",
              "void $F($S* p) {\n    bool hit = p->$a > $K || p->$b < 0;\n    p->$c = 0;\n    Report(hit);\n}",
              "void $F($S* p) {\n    bool hit = false;\n    if (p->$a > $K || p->$b < 0) {\n        hit = true;\n    }\n"
              "    p->$c = 0;\n    Report(hit);\n}"),
        Shape("struct $S { int $a; int $b; int $c; int other; };\nextern void Touch($S*);\nextern void Report($S*);\n",
              "void $F($S* p) {\n    bool hit = p->$a > $K || p->$b < 0;\n    Touch(p);\n    p->$c = 0;\n    p->other = 0;\n"
              "    if (hit) {\n        Report(p);\n    }\n}",
              "void $F($S* p) {\n    bool hit = false;\n    if (p->$a > $K || p->$b < 0) {\n        hit = true;\n    }\n"
              "    Touch(p);\n    p->$c = 0;\n    p->other = 0;\n    if (hit) {\n        Report(p);\n    }\n}")],
       grid(8, "flagform", S=STRUCT, K=KS, **_triples())),
    _l("assign-in-condition", "matching.md: A global loaded into r3 and copied (`cmplwi r3, 0; mr r31, r3`) where "
       "the target loads it straight into r31 -> assign in the condition; rewrites.py assign-in-condition",
       "A value loaded into r3 and copied (`cmplwi r3, 0; mr r31, r3`) where the target loads it straight",
       "`if ((p = g) != 0)`; `T* p = g; if (p != 0)` makes the copy",
       lambda r, w: count(r, "mr") < count(w, "mr") or (r != w and len(r) < len(w)),
       [Shape("struct $S { int $a; int $b; };\nextern $S* g$S;\nextern void Use($S*);\n",
              "void $F() {\n    $S* p;\n    if ((p = g$S) != 0) {\n        Use(p);\n        p->$a = $K;\n    }\n}",
              "void $F() {\n    $S* p = g$S;\n    if (p != 0) {\n        Use(p);\n        p->$a = $K;\n    }\n}"),
        Shape("struct $S { int $a; int $b; };\nextern $S* Find(int);\nextern void Use($S*);\n",
              "void $F(int id) {\n    $S* p;\n    if ((p = Find(id)) != 0) {\n        Use(p);\n        p->$a = $K;\n    }\n}",
              "void $F(int id) {\n    $S* p = Find(id);\n    if (p != 0) {\n        Use(p);\n        p->$a = $K;\n    }\n}")],
       grid(8, "assigncond", S=STRUCT, K=KS, **_triples())),
    _l("unrolled-loop", "matching.md: Four copies of the same stores, or a run of constant stores to one array -> a "
       "short constant-count `for` loop that the compiler unrolled",
       "A run of constant stores to one array (`li` of 1, 2, 3 ... interleaved with the `stw`s) in another schedule",
       "a short constant-count `for` loop, unrolled by the compiler: write the loop, not the copies",
       lambda r, w: r != w,
       [Shape("struct $S { $T m[$N]; };\n",
              "void $F($S* p) {\n    for (int i = 0; i < $N; i++) {\n        p->m[i] = i + 1;\n    }\n}",
              "void $F($S* p) {\n$STORES}")],
       [{"S": s, "T": t, "N": n, "F": f, "STORES": "".join(f"    p->m[{i}] = {i + 1};\n" for i in range(int(n)))}
        for s, t, n, f in (("Item", "int", "8", "Fill"), ("Slot", "short", "9", "Setup"), ("Track", "int", "10", "Reset"),
                           ("Cell", "unsigned char", "12", "Prepare"), ("Light", "int", "16", "Init"),
                           ("Body", "int", "6", "Prime"))]),
    _l("int-comparison-value", "matching.md: `cntlzw; extrwi rD, r0, 8, 19` (a bool widened) where the target has "
       "`cntlzw; srwi rD, r0, 5` -> the comparison's value used as an int straight away",
       "`cntlzw; extrwi rD, r0, 8, 19` (a bool widened) where the target has `cntlzw; srwi rD, r0, 5`",
       "the comparison's value used as an int (`(c ? 1 : 0) + 1`), not `(a == b) + 1`; a returned test as `bool`",
       lambda r, w: count(r, "extrwi") != count(w, "extrwi") or count(r, "srwi") != count(w, "srwi"),
       [Shape("", "bool $F(int $a, int $b) {\n    return $a == $b;\n}", "int $F(int $a, int $b) {\n    return $a == $b;\n}"),
        Shape("struct $S { int $a; int $b; };\n", "int $F(const $S* p) {\n    return (p->$a == $K ? 1 : 0) + 1;\n}",
              "int $F(const $S* p) {\n    return (p->$a == $K) + 1;\n}")],
       grid(8, "cmpval", S=STRUCT, K=KS, **_triples())),
    _l("bit-test-compare", "matching.md: `extrwi` + `xori 1` passed on as a `bool` -> a comparison `(m & 2) == 0`",
       "`extrwi` + `xori 1` where the target normalises with `neg; or; srwi` (or the reverse)",
       "a comparison `(m & 2) == 0`, not `((m >> 1) & 1) ^ 1`",
       lambda r, w: r != w and mn(r) != mn(w),
       [Shape("extern void Show(bool);\n", "void $F(unsigned int m) {\n    Show((m & $B) == 0);\n}",
              "void $F(unsigned int m) {\n    Show(((m >> $S) & 1) ^ 1);\n}")],
       [{"F": f, "B": str(1 << s), "S": str(s)} for f, s in (("Apply", 1), ("Select", 2), ("Mix", 3), ("Pick", 4),
                                                             ("Refresh", 5))]),
    _l("guard-return-value", "matching.md: `cmpwi r3, 0; li r3, K; beq end; li r3, 0` where ours is branchless -> "
       "name the call's result and set the return value before the test",
       "`cmpwi r3, 0; li r3, K; beq end; li r3, 0` where ours is branchless (`cntlzw`/`neg`/`andc`)",
       "`int ok = f(); int r = K; if (ok) r = 0; return r;` (`return f() ? 0 : K` folds to branchless code)",
       lambda r, w: r != w and count(r, "beq", "bne") > count(w, "beq", "bne"),
       [Shape("extern int Ready(int);\n",
              "int $F(int n) {\n    int ok = Ready(n);\n    int r = $K;\n    if (ok) {\n        r = 0;\n    }\n"
              "    return r;\n}",
              "int $F(int n) {\n    return Ready(n) ? 0 : $K;\n}")],
       grid(8, "guardret", K=("1", "2", "4", "8", "16")), both=False),
    _l("floor-minus-one", "matching.md: `srwi; subfc; subfe; andc` computing `n - 1` floored at 0 -> "
       "`int k = n < 1 ? 0 : n - 1;`",
       "`srwi; subfc; subfe; andc` computing `n - 1` floored at 0", "`n < 1 ? 0 : n - 1` (`n > 0` gives another sequence)",
       lambda r, w: count(r, "subfe", "andc") != count(w, "subfe", "andc"),
       [Shape("struct $S { int $a; int $b; };\n", "void $F($S* p) {\n    int k = p->$a < 1 ? 0 : p->$a - 1;\n"
              "    p->$b = k;\n}", "void $F($S* p) {\n    int k = p->$a > 0 ? p->$a - 1 : 0;\n    p->$b = k;\n}")],
       grid(8, "floor1", S=STRUCT, **_triples())),
    _l("named-product", "matching.md: `fadds` operands swapped where `p` is a product in the same expression -> name "
       "the product in a local",
       "`fadds` operands swapped (ours `x + p`, target `p + x`) where `p` is a product",
       "name the product in a local (`float y = v * k; out = y + x;`); written inline, CW may put the plain operand first",
       lambda r, w: roles_swapped(r, w, ("fadds", "add")),
       [Shape("struct $S { float $a; float $b; float $c; };\n",
              "void $F($S* p, float k) {\n    float y = p->$a * k;\n    p->$c = y + p->$b;\n}",
              "void $F($S* p, float k) {\n    p->$c = p->$a * k + p->$b;\n}"),
        Shape("", "int $F(int $a, int $b, int $c) {\n    int t = $a * $b;\n    return t + $c;\n}",
              "int $F(int $a, int $b, int $c) {\n    return $a * $b + $c;\n}")],
       grid(8, "namedprod", S=STRUCT, **_triples())),
    _l("named-converted-operand", "matching.md: `mullw` operands swapped (`conv, n` vs `n, conv`) -> the converted "
       "operand in a named local; swapping the source operands does nothing",
       "`mullw` operands swapped (`conv, n` vs `n, conv`)",
       "put the converted operand in a named local (`int q = (int)(t / n); q * n`)",
       lambda r, w: roles_swapped(r, w, ("mullw",)),
       [Shape("", "int $F(float t, int n) {\n    int q = (int)(t / n);\n    return q * n;\n}",
              "int $F(float t, int n) {\n    return (int)(t / n) * n;\n}")],
       [{"F": f} for f in ("Snap", "Quantize", "Round", "Align")]),
    _l("pack-halves", "matching.md: Two halfwords packed the other way round -> swap the `|` operands (the right "
       "operand is the base the left one is inserted into)",
       "Two halfwords packed the other way round (`rlwimi` into the other register)",
       "swap the `|` operands: `(lo & 0xFFFF) | (hi << 16)` shifts `hi` and inserts `lo`",
       lambda r, w: r != w and (count(r, "rlwimi") or count(w, "rlwimi")),
       [Shape("", "unsigned int $F(unsigned int $a, unsigned int $b) {\n    return ($a & 0xFFFF) | ($b << 16);\n}",
              "unsigned int $F(unsigned int $a, unsigned int $b) {\n    return ($b << 16) | ($a & 0xFFFF);\n}")],
       grid(8, "pack", **_triples())),
    _l("signed-search-index", "matching.md: In a search loop the target increments the counter before the offset -> "
       "declare the index `int` (signed), still compared with the unsigned count",
       "In a search loop, the counter's and the offset's increments in the other order",
       "declare the index `int` (signed), compared with the unsigned count (the compare stays `cmplw`)",
       lambda r, w: reordered(r, w, ("addi",)) or r != w,
       [Shape("struct $S { int id; int $a; };\nstruct List { unsigned int count; $S* items[$N]; };\n",
              "$S* $F(List* l, int id) {\n    for (int i = 0; i < l->count; i++) {\n        $S* t = l->items[i];\n"
              "        if (t && t->id == id) {\n            return t;\n        }\n    }\n    return 0;\n}",
              "$S* $F(List* l, int id) {\n    for (unsigned int i = 0; i < l->count; i++) {\n        $S* t = l->items[i];\n"
              "        if (t && t->id == id) {\n            return t;\n        }\n    }\n    return 0;\n}"),
        Shape("struct $S { int id; int $a; };\nstruct List { $S** items; unsigned int count; };\n",
              "$S* $F(List* l, int id) {\n    for (int i = 0; i < l->count; i++) {\n        if (l->items[i]->id == id) {\n"
              "            return l->items[i];\n        }\n    }\n    return 0;\n}",
              "$S* $F(List* l, int id) {\n    for (unsigned int i = 0; i < l->count; i++) {\n        if (l->items[i]->id == id) {\n"
              "            return l->items[i];\n        }\n    }\n    return 0;\n}")],
       grid(8, "ssearch", S=STRUCT, N=("8", "16", "32", "4", "64"), F=("Find", "Lookup", "Search", "Locate", "Get"),
            **_triples())),
    _l("decl-load-order", "matching.md: Loads scheduled in another order than the target's -> declare the locals in "
       "the target's load order; their initializing loads follow declaration order",
       "Loads scheduled in another order than the target's (`lhz mC`, `lhz *data`, `lwz m8`)",
       "declare the locals in the target's load order: their initializing loads follow declaration order",
       lambda r, w: reordered(r, w, ("lwz", "lhz", "lha", "lbz", "lfs")),
       [Shape("struct $S { int $a; short $b; int $c; };\nextern void Send(int, int, int);\n",
              "void $F($S* p) {\n    int $c = p->$c;\n    int $b = p->$b;\n    int $a = p->$a;\n    Send($a, $b, $c);\n}",
              "void $F($S* p) {\n    int $a = p->$a;\n    int $b = p->$b;\n    int $c = p->$c;\n    Send($a, $b, $c);\n}"),
        Shape("struct $S { int $a; short $b; int $c; };\nextern void Touch($S*);\n",
              "int $F($S* p) {\n    int $c = p->$c;\n    int $a = p->$a;\n    Touch(p);\n    return $a - $c;\n}",
              "int $F($S* p) {\n    int $a = p->$a;\n    int $c = p->$c;\n    Touch(p);\n    return $a - $c;\n}")],
       grid(8, "loadord", S=STRUCT, **_triples())),
    _l("named-arg-local", "gamecube-mwcc.md: A field the target loads before a computed argument comes from a named "
       "local (`void* h = m4; f(a, Compute(), h);`)",
       "A field loaded before a later argument's call (into a saved register across it), where ours loads it after",
       "copy the field into a local on an earlier line: `int h = p->m4; f(a, Compute(), h);`",
       lambda r, w: r != w,
       [Shape("struct $S { int $a; int $b; };\nextern int Compute(int);\nextern void Post(int, int, int);\n",
              "void $F($S* p, int n) {\n    int h = p->$b;\n    Post(n, Compute(n), h);\n}",
              "void $F($S* p, int n) {\n    Post(n, Compute(n), p->$b);\n}"),
        Shape("extern int Compute(int);\nextern void Post(int, int, void*);\n"
              "struct $S {\n    int $a;\n    void* $b;\n    void $F(int n);\n};\n",
              "void $S::$F(int n) {\n    void* h = $b;\n    Post(n, Compute(n), h);\n}",
              "void $S::$F(int n) {\n    Post(n, Compute(n), $b);\n}")],
       grid(8, "namedarg", S=STRUCT, **_triples())),
    _l("inline-getter", "gamecube-mwcc.md: A field that an inline member function reads is not reused by the "
       "caller's later code",
       "A field tested and then loaded again in each branch, where ours reuses the first load",
       "the test reads it through an inline getter (`p->IsOn()`); a plain `p->flags & 1` is CSE'd into both arms",
       lambda r, w: more(r, w, "lwz", "lhz", "lbz") or (r != w and len(r) > len(w)),
       [Shape("struct $S {\n    unsigned int $a;\n    int $b;\n    bool IsOn() const { return ($a & $K) != 0; }\n};\n",
              "void $F($S* p) {\n    if (p->IsOn()) {\n        p->$a |= 0x10;\n    } else {\n        p->$a &= ~0x20;\n    }\n}",
              "void $F($S* p) {\n    if ((p->$a & $K) != 0) {\n        p->$a |= 0x10;\n    } else {\n        p->$a &= ~0x20;\n"
              "    }\n}")],
       grid(8, "getter", S=STRUCT, K=("1", "2", "4", "8", "0x40"), **_triples())),
    _l("default-first", "matching.md: A `switch` whose `default` body comes before the case bodies -> write "
       "`default:` first (bodies are laid out in source order)",
       "A `switch` whose `default` body comes before the case bodies", "write `default:` first in the source",
       lambda r, w: before(r, "Other", "On") != before(w, "Other", "On"),
       [Shape("extern void On(int);\nextern void Off(int);\nextern void Other(int);\n",
              "void $F(int m, int v) {\n    switch (m) {\n    default:\n        Other(v);\n        break;\n    case $K:\n"
              "        On(v);\n        break;\n    case $K2:\n        Off(v);\n        break;\n    }\n}",
              "void $F(int m, int v) {\n    switch (m) {\n    case $K:\n        On(v);\n        break;\n    case $K2:\n"
              "        Off(v);\n        break;\n    default:\n        Other(v);\n        break;\n    }\n}")],
       [{"F": f, "K": k, "K2": k2} for f, k, k2 in (("Apply", "1", "2"), ("Select", "3", "7"), ("Mix", "0", "4"),
                                                    ("Route", "5", "9"), ("Handle", "2", "6"))]),
    _l("for-header-step", "matching.md: A copy loop that stores through a stepped pointer where ours uses indexed "
       "stores -> advance the destination index in the `for` header",
       "A copy loop storing through a stepped pointer (`sth r0, 0(r6)` ... `addi r6, r6, 2`) where ours uses "
       "indexed stores (`sthx`)",
       "advance the destination index in the `for` header (`j++, i++`), not with `dst[i++] = src[j]`",
       lambda r, w: count(r, "sthx", "stwx", "stbx") != count(w, "sthx", "stwx", "stbx") or r != w,
       [Shape("",
              "void $F($T* dst, const $T* src, int i, int n) {\n    for (int j = 0; j < n; j++, i++) {\n"
              "        dst[i] = src[j];\n    }\n}",
              "void $F($T* dst, const $T* src, int i, int n) {\n    for (int j = 0; j < n; j++) {\n"
              "        dst[i++] = src[j];\n    }\n}")],
       grid(8, "forstep", T=("short", "int", "unsigned char", "unsigned short", "float"))),
    _l("index-from-zero", "matching.md: A short loop's element offset left in a pointer -> start the counter at 0 and "
       "index `a[c + 1]`",
       "A short loop's element offset left in a pointer (`addi rP, rBase, K` then `0x44(rP)`) where the target uses "
       "fixed displacements", "start the counter at 0 and index `a[c + 1]` instead of counting from 1",
       lambda r, w: r != w,
       [Shape("struct $S { int $a; int m[$N]; };\nextern void Put(int);\n",
              "void $F($S* p) {\n    for (int c = 0; c < $M; c++) {\n        Put(p->m[c + 1]);\n    }\n}",
              "void $F($S* p) {\n    for (int c = 1; c < $N; c++) {\n        Put(p->m[c]);\n    }\n}")],
       [{"S": s, "N": n, "M": str(int(n) - 1), "F": f, "a": a} for s, n, f, a in
        (("Item", "8", "Emit", "count"), ("Track", "5", "Flush", "flags"), ("Body", "12", "Dump", "x"),
         ("Cell", "4", "Print", "mA"))]),
    _l("call-inline-helper", "matching.md: An inlined copy of another function's body differing only by `li rX, 0; "
       "mr rY, rX` (ours: two `li`) -> call that function and let `-inline auto` inline it",
       "An inlined copy of a helper's body in another register or schedule than ours (e.g. `li rX, 0; mr rY, rX` vs "
       "two `li`)", "call the helper (defined earlier in the TU) and let `-inline auto` inline it; a hand-written copy "
       "of its body does not reproduce it",
       lambda r, w: r != w and not calls(r, "Clear") and not calls(r, "Sum"),
       [Shape("struct $S { $T $a; $T $b; $T $c; int used; };\nstatic void Clear($S* p) {\n    p->$a = 0;\n    p->$b = 0;\n"
              "    p->$c = 0;\n}\n",
              "void $F($S* p, $S* q, int n) {\n    Clear(p);\n    q->used = n;\n    Clear(q);\n}",
              "void $F($S* p, $S* q, int n) {\n    p->$a = 0;\n    p->$b = 0;\n    p->$c = 0;\n    q->used = n;\n"
              "    q->$a = 0;\n    q->$b = 0;\n    q->$c = 0;\n}"),
        Shape("struct $S { $T $a; $T $b; $T $c; };\nstatic $T Sum(const $S* p) {\n    return p->$a + p->$b + p->$c;\n}\n",
              "$T $F(const $S* p, const $S* q) {\n    return Sum(p) - Sum(q);\n}",
              "$T $F(const $S* p, const $S* q) {\n    return (p->$a + p->$b + p->$c) - (q->$a + q->$b + q->$c);\n}")],
       grid(8, "callinline", S=STRUCT, T=("int", "short", "float"), **_triples()), both=False),
]


BY_ID = {les.id: les for les in LESSONS}


# ---- compiling a snippet -------------------------------------------------------------------------------

@dataclass
class Built:
    obj: Path | None
    error: str = ""
    asm: dict = field(default_factory=dict)          # {symbol: compact asm}


def compile_snippet(cfg: Config, text: str, compiler: str, flags: list[str], work: Path) -> Built:
    """Compile one generic snippet (nothing included) with `compiler`; its functions' compact asm."""
    import shlex
    from .llmgen import object_asm
    from .llmtry import compact_asm, compiler_errors
    exe = cfg.root / "build" / "compilers" / compiler / "mwcceppc.exe"
    wibo = cfg.root / "build" / "tools" / "wibo"
    work.mkdir(parents=True, exist_ok=True)
    (work / "l.cpp").write_text(text)
    cmd = [str(wibo), str(exe), *flags, "-c", "l.cpp", "-o", "."]
    proc = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
    obj = work / "l.o"
    if proc.returncode != 0 or not obj.is_file():
        return Built(None, compiler_errors(proc.stdout + proc.stderr) or (proc.stdout + proc.stderr)[-300:] or
                     f"compile error ({shlex.join(cmd[:3])})")
    return Built(obj, "", {s: compact_asm(e.text) for s, e in object_asm(cfg, obj).items()})


def diff_rows(cfg: Config, target: Path, ours: Path, symbol: str) -> tuple[float, list[str], str]:
    """objdiff of our object against the target's for one function: (percent, llm-try's rows, class)."""
    from .llmtry import classify, relative_rows
    from .objcache import objdiff_json, symbol_pair
    doc, _ = objdiff_json(cfg, target, ours)
    pair = symbol_pair(doc, symbol) if doc else None
    if pair is None:
        return 0.0, [], ""
    left, right = pair
    return float(left.get("match_percent", 0.0)), relative_rows(left, right), classify(left, right)


def symbol_of(asm: dict, name: str) -> str | None:
    """The compiled symbol of the lesson's function `name` (mangled C++, or a C name)."""
    return next((s for s in sorted(asm) if s == name or s.startswith(name + "__")), None)


def definition_span(text: str, name: str) -> tuple[int, int] | None:
    from ...core import cbody
    for f in cbody.functions(text):
        if f.name.split("::")[-1] == name:
            return f.head_start, f.close + 1
    return None


# ---- building the records ------------------------------------------------------------------------------

@dataclass
class Pair:
    lesson: str
    shape: int
    variant: int
    compiler: str
    status: str                            # kept, same asm, rule not seen, compile error, judge
    records: list = field(default_factory=list)
    note: str = ""


def flag_list(text: str, inline: str) -> list[str]:
    import shlex
    return [*shlex.split(text), "-inline", inline]


def build_pair(cfg: Config, les: Lesson, si: int, vi: int, params: dict, compiler: str, flags: str,
               judge=None) -> Pair:
    from .llmdata import RECORD_VERSION
    from .llmtry import PROMPT_VERSION, asm_shape, declarations
    shape = les.shapes[si]
    pr = Pair(les.id, si, vi, compiler, "")
    try:
        decls = Template(shape.decls).substitute(params)
        right = Template(shape.right).substitute(params)
        wrong = Template(shape.wrong).substitute(params)
    except (KeyError, ValueError) as e:
        pr.status, pr.note = "template error", str(e)
        return pr
    name = params["F"]
    tmp = Path(tempfile.mkdtemp(prefix="dk-l-"))
    try:
        built = {}
        for side, code in (("right", right), ("wrong", wrong)):
            built[side] = compile_snippet(cfg, decls + "\n" + code + "\n", compiler, flag_list(flags, les.inline), tmp / side)
            if built[side].obj is None:
                pr.status, pr.note = "compile error", f"{side}: {built[side].error[:200]}"
                return pr
        sym = symbol_of(built["right"].asm, name)
        if sym is None or sym not in built["wrong"].asm:
            pr.status, pr.note = "compile error", f"no function {name} in both objects"
            return pr
        ra, wa = built["right"].asm[sym], built["wrong"].asm[sym]
        ri, wi = insns(ra), insns(wa)
        if ri == wi:
            pr.status = "same asm"
            return pr
        try:
            seen = les.check(ri, wi)
        except Exception as e:                       # a broken check is a bug in the catalog: reported
            seen, pr.note = False, f"check failed: {e}"
        if not seen:
            pr.status = "rule not seen"
            pr.note = pr.note or text(ri)[:160] + " | " + text(wi)[:160]
            return pr
        sides = [("right", "wrong", right, wrong, ra, wa)] + ([("wrong", "right", wrong, right, wa, ra)] if les.both else [])
        for tgt, cur, exact, wrong_code, tasm, casm in sides:
            pct, rows, dclass = diff_rows(cfg, built[tgt].obj, built[cur].obj, sym)
            if pct >= 100.0 or not rows:
                continue
            full = decls + "\n" + wrong_code + "\n"
            span = definition_span(full, name)
            d = declarations(full, span, set(), 8000, False) if span else decls
            n_insns = len(insns(tasm))
            rec = {"v": RECORD_VERSION, "source": SOURCE, "kind": "repair", "rule": les.id,
                   "unit": f"{SOURCE}/{les.id}", "function": sym, "size": 4 * n_insns, "compiler": compiler,
                   "flags": " ".join(flag_list(flags, les.inline)), "asm": tasm, "asm_shape": asm_shape(tasm),
                   "mutant_shape": asm_shape(casm), "decls": d, "wrong": wrong_code, "wrong_percent": pct,
                   "rows": rows, "rows_fixed": 1, "diff_class": dclass, "exact": exact, "first_round": True,
                   "lesson": {"symptom": les.symptom, "idiom": les.idiom, "ref": les.ref,
                              "direction": "idiom" if tgt == "right" else "reverse"},
                   "provenance": {"rule": les.id, "shape": si, "variant": vi, "compiler": compiler, "inline": les.inline,
                                  "direction": "idiom" if tgt == "right" else "reverse", "time": time.time()},
                   "prompt_version": PROMPT_VERSION}
            bad = judge(rec) if judge else []
            if bad:
                pr.note = "judge: " + "; ".join(bad[:2])
                continue
            pr.records.append(rec)
        pr.status = "kept" if pr.records else ("judge" if pr.note.startswith("judge") else "same asm")
        return pr
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def merge_compilers(recs: list[dict]) -> list[dict]:
    """One record per (rule, function, wrong, exact): a pair both compilers compile alike is one lesson
    (provenance.compilers lists them); a pair whose asm differs between compilers keeps the first
    compiler's record (the export's dedupe key has no compiler)."""
    from .llmdata import norm
    out: dict[tuple, dict] = {}
    for r in recs:
        k = (r["rule"], r["function"], norm(r["wrong"]), norm(r["exact"]))
        if k not in out:
            out[k] = {**r, "provenance": {**r["provenance"], "compilers": [r["compiler"]]}}
        elif r["asm"] == out[k]["asm"]:
            out[k]["provenance"]["compilers"].append(r["compiler"])
        else:
            out[k]["provenance"].setdefault("differs_on", []).append(r["compiler"])
    return list(out.values())


def jobs_for(les: Lesson, compilers: tuple[str, ...]) -> list[tuple[Lesson, int, int, dict, str]]:
    out = []
    for comp in les.compilers:
        if comp not in compilers:
            continue
        params = les.params
        for si in range(len(les.shapes)):
            for vi, p in enumerate(params):
                out.append((les, si, vi, p, comp))
    return out


def summarize(pairs: list[Pair]) -> dict:
    """{rule: {kept pairs, records, by status, compilers kept}} in LESSONS order."""
    out: dict[str, dict] = {les.id: {"pairs": 0, "kept": 0, "records": 0, "status": {}, "kept_on": []} for les in LESSONS}
    for p in pairs:
        s = out.setdefault(p.lesson, {"pairs": 0, "kept": 0, "records": 0, "status": {}, "kept_on": []})
        s["pairs"] += 1
        s["status"][p.status] = s["status"].get(p.status, 0) + 1
        if p.status == "kept":
            s["kept"] += 1
            s["records"] += len(p.records)
            if p.compiler not in s["kept_on"]:
                s["kept_on"].append(p.compiler)
    return {k: v for k, v in out.items() if v["pairs"]}


def run(cfg: Config, args: argparse.Namespace) -> int:
    from .llmdata import AnswerJudge, out_dir, write_records
    if args.list:
        for les in LESSONS:
            print(f"{les.id:24} {les.ref}")
        return OK
    want = [x for x in (args.rules.split(",") if args.rules else []) if x]
    unknown = [x for x in want + ([args.show] if args.show else []) if x not in BY_ID]
    if unknown:
        print(f"error: unknown rule(s) {', '.join(unknown)} (--list)")
        return USAGE
    compilers = tuple(c for c in args.compilers.split(",") if c)
    missing = [c for c in compilers if not (cfg.root / "build" / "compilers" / c / "mwcceppc.exe").is_file()]
    if missing or not (cfg.root / "build" / "tools" / "wibo").is_file():
        print(f"error: compiler(s) not installed: {', '.join(missing) or 'wibo'} (./dk setup)")
        return USAGE
    if args.show:
        return show(cfg, BY_ID[args.show], compilers, args.flags)
    lessons = [BY_ID[x] for x in want] if want else LESSONS
    work = [j for les in lessons for j in jobs_for(les, compilers)]
    judge = AnswerJudge(cfg) if not args.no_judge else None
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        pairs = list(pool.map(lambda j: build_pair(cfg, j[0], j[1], j[2], j[3], j[4], args.flags, judge), work))
    recs = merge_compilers([r for p in pairs for r in p.records])
    root = out_dir(cfg, args)
    dest = root / "records" / f"{SOURCE}.jsonl"
    if want and dest.is_file():                       # a partial run replaces only its rules' records
        from .llmdata import read_jsonl
        recs = [r for r in read_jsonl(dest) if r.get("rule") not in set(want)] + recs
    write_records(dest, recs)
    summary = summarize(pairs)
    for k, v in summary.items():
        v["records"] = sum(1 for r in recs if r.get("rule") == k)
    report = {"made": time.strftime("%Y-%m-%d %H:%M"), "compilers": list(compilers), "flags": args.flags,
              "records": len(recs), "rules": summary,
              "not_reproduced": [k for k, v in summary.items() if not v["kept"]],
              "notes": {f"{p.lesson}#{p.shape}.{p.variant}@{p.compiler}": f"{p.status}: {p.note}" for p in pairs
                        if p.status != "kept" and p.note}}
    (root / "records" / f"{SOURCE}.report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"{len(pairs)} pairs compiled in {time.time() - t0:.0f} s; {len(recs)} records into {cfg.rel(dest)}")
    for k, v in summary.items():
        st = ", ".join(f"{n} {s}" for s, n in sorted(v["status"].items()))
        print(f"  {k:24} {v['records']:4d} records from {v['kept']:3d}/{v['pairs']:3d} pairs  "
              f"({st}){'  kept on ' + '+'.join(v['kept_on']) if v['kept_on'] else ''}")
    if report["not_reproduced"]:
        print(f"  not reproduced: {', '.join(report['not_reproduced'])} (findings: see {cfg.rel(root)}/records/"
              f"{SOURCE}.report.json notes; the playbook stays as is without evidence)")
    return OK


def show(cfg: Config, les: Lesson, compilers: tuple[str, ...], flags: str) -> int:
    """The rule's first pair on each compiler: both shapes, both asms, whether the check holds."""
    for comp in compilers:
        params = les.params[:1]
        for si, shape in enumerate(les.shapes):
            for p in params:
                tmp = Path(tempfile.mkdtemp(prefix="dk-l-"))
                try:
                    decls = Template(shape.decls).substitute(p)
                    asms = {}
                    for side in ("right", "wrong"):
                        code = Template(getattr(shape, side)).substitute(p)
                        b = compile_snippet(cfg, decls + "\n" + code + "\n", comp, flag_list(flags, les.inline), tmp / side)
                        sym = symbol_of(b.asm, p["F"]) if b.obj else None
                        asms[side] = b.asm.get(sym, "") if sym else ""
                        print(f"== {les.id} shape {si} {comp} {side}: {code.splitlines()[0]} ...\n"
                              + (asms[side] or b.error))
                    ok = les.check(insns(asms["right"]), insns(asms["wrong"]))
                    print(f"-- check: {'holds' if ok else 'NOT seen'}"
                          + ("  (same asm)" if insns(asms["right"]) == insns(asms["wrong"]) else ""))
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)
    return OK


def configure(sub) -> None:
    s = sub.add_parser("lessons", help="compiler-verified minimal pairs from the playbook's asm-pattern -> idiom rules "
                       "(generic code, both shapes compiled; kept only when the asm differs as the rule says)")
    s.add_argument("--rules", default="", help="only these rule ids (comma-separated; --list)")
    s.add_argument("--compilers", default=",".join(COMPILERS), help=f"compilers (default {','.join(COMPILERS)})")
    s.add_argument("--flags", default=FLAGS, help="compiler flags without -inline (default: the game set)")
    s.add_argument("--jobs", type=int, default=2, help="parallel pairs (default 2)")
    s.add_argument("--list", action="store_true", help="list the rules and where the playbook states them")
    s.add_argument("--show", default="", help="print one rule's first pair on each compiler (both asms, the check)")
    s.add_argument("--no-judge", action="store_true", help="skip llm-review's rules on the answers (a test)")
    s.add_argument("--out", default="", help="dataset directory (default llm-data/dataset)")
