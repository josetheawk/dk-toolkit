"""Self-repeat finder: stretches of ONE function that occur again inside it (an inline called twice).

`inlines.repeated_blocks` also counts stretches shared with other functions, and a shape match
alone accepts copies that call other functions or read other literals. This finder works inside
one function and adds an identity check:

1. shape: each instruction is abstracted (registers, stack displacements, branch labels, the
   names of symbols); equal shapes are the same operation on other registers;
2. identity: the relocation targets (callee names, literal pool objects by their contents,
   globals by name) must be equal as well, copy against copy;
3. a repeat is a stretch of at least KGRAM instructions with equal shape and identity at two
   non-overlapping places; copies are found longest first and each instruction belongs to one;
4. value check: `inlines.same_block` of the first copy against every other (BLOCK_CHECK); a
   copy that fails is dropped, the block stays while two copies remain.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import asmflow, inlines
from .asmflow import is_reg, symref

KGRAM = inlines.KGRAM
BLOCK_CHECK = inlines.BLOCK_CHECK
MIN_COPIES = 2
MAX_BUCKET = 48          # positions kept per k-gram: very common grams carry no information
MAX_ROUNDS = 400


# ---------------------------------------------------------------- tokens

def _sym_of(op: str) -> tuple[str, str] | None:
    """(symbol, relocation kind) of an operand: `sym@ha`, `sym@l(r3)`, `sym@sda21(r13)`."""
    inner = op[:op.rindex("(")] if op.endswith(")") and "(" in op else op
    s = symref(inner)
    return (s[0], s[1]) if s else None


def shape_and_identity(node) -> tuple[str, tuple]:
    """(shape token, relocation targets) of one instruction. The shape keeps field offsets and
    immediates, drops register numbers, stack displacements and symbol names; the identity is the
    tuple of symbol names it refers to (a `bl` callee included)."""
    ops, ids = [], []
    mnem = node.mnem.rstrip("+-")
    for op in node.insn.ops:
        if op == "r1":
            ops.append("sp")
        elif is_reg(op):
            ops.append("F" if op.startswith("f") else "R")
        elif op.endswith("(r1)"):
            ops.append("sp")
        elif (s := _sym_of(op)) is not None:
            ids.append(s[0])
            tail = op[op.rindex("("):] if op.endswith(")") and "(" in op else ""
            ops.append(f"S@{s[1]}" + ("(R)" if tail and tail != "(r1)" else ""))
        elif op.endswith(")") and "(" in op:
            ops.append(op[:op.index("(")] + "(R)")
        elif op.startswith((".L_", '".L_')):
            ops.append("L")
        elif mnem in ("bl", "b") and not op.startswith("0x"):
            ids.append(op)
            ops.append("CALL" if mnem == "bl" else "TAIL")
        else:
            ops.append(op)
    if node.mnem in ("addi", "subi") and len(node.insn.ops) == 3 and node.insn.ops[1] == "r1":
        ops[2] = "*"                                   # addi rX, r1, K: a stack address
    return mnem + " " + ",".join(ops), tuple(ids)


def canonical_ids(ids: tuple, canon) -> tuple:
    """Relocation targets by value: auto-named literals compare by their contents."""
    if canon is None:
        return ids
    out = []
    for name in ids:
        base = name.split("+", 1)[0]
        v = canon((("sym", base), 0))
        out.append(name if v == (("sym", base), 0) else repr(v))
    return tuple(out)


def sequence(p: inlines.Prepared, canon=None) -> tuple[list[int], list[str], list[tuple]]:
    """(node indexes, shape tokens, canonical identities) of the nodes a repeat search looks at."""
    seq = inlines.block_sequence(p)
    shapes, ids = [], []
    for i in seq:
        s, d = shape_and_identity(p.flow.nodes[i])
        shapes.append(s)
        ids.append(canonical_ids(d, canon))
    return seq, shapes, ids


# ---------------------------------------------------------------- finding

@dataclass
class Repeat:
    length: int
    starts: list[int]                    # positions in the sequence, one per copy
    verified: float = 1.0
    dropped: int = 0                     # copies the value check removed

    def spans(self) -> list[tuple[int, int]]:
        return [(s, s + self.length) for s in self.starts]


def find_repeats(keys: list, k: int = KGRAM) -> list[Repeat]:
    """Repeats in a list of hashable per-instruction keys, longest first, no position twice."""
    n = len(keys)
    work = list(keys)
    out: list[Repeat] = []
    for _ in range(MAX_ROUNDS):
        index: dict[tuple, list[int]] = {}
        for p in range(n - k + 1):
            gram = tuple(work[p:p + k])
            if len(set(gram)) >= 3 and not any(isinstance(x, _Claimed) for x in gram):
                bucket = index.setdefault(gram, [])
                if len(bucket) < MAX_BUCKET:
                    bucket.append(p)
        best = None
        for bucket in index.values():
            for a, p in enumerate(bucket):
                for q in bucket[a + 1:]:
                    if q - p < k or (p > 0 and work[p - 1] == work[q - 1] and not isinstance(work[p - 1], _Claimed)):
                        continue                          # not a start of a maximal run
                    length = 0
                    d = q - p
                    while q + length < n and length < d and work[p + length] == work[q + length] \
                            and not isinstance(work[p + length], _Claimed):
                        length += 1
                    if length >= k and (best is None or length > best[0]):
                        best = (length, p, q)
        if best is None:
            break
        length, p, _q = best
        content = work[p:p + length]
        starts, nxt = [], 0
        gram = tuple(content[:k])
        for r in range(n - length + 1):
            if r >= nxt and work[r:r + length] == content:
                starts.append(r)
                nxt = r + length
        for r in starts:
            for x in range(r, r + length):
                work[x] = _Claimed(x)
        out.append(Repeat(length, starts))
    return out


class _Claimed:
    """A position already part of a reported block (never equal to anything else)."""
    __slots__ = ("i",)

    def __init__(self, i: int):
        self.i = i

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)


def check_values(flow: asmflow.Flow, seq: list[int], reps: list[Repeat], threshold: float = BLOCK_CHECK) -> list[Repeat]:
    """Drop copies whose values disagree with the first copy; blocks left with one copy go."""
    out = []
    for r in reps:
        first = seq[r.starts[0]:r.starts[0] + r.length]
        keep, worst = [r.starts[0]], 1.0
        for s in r.starts[1:]:
            share, _bind = inlines.same_block(flow, first, flow, seq[s:s + r.length])
            if share >= threshold:
                keep.append(s)
                worst = min(worst, share)
        if len(keep) >= MIN_COPIES:
            out.append(Repeat(r.length, keep, round(worst, 2), len(r.starts) - len(keep)))
    return out


@dataclass
class Report:
    function: str
    size: int                                # instructions of the sequence
    repeats: list[Repeat] = field(default_factory=list)
    seq: list[int] = field(default_factory=list)
    addrs: list[int] = field(default_factory=list)       # address of each sequence position

    def covered(self, redundant: bool = False) -> int:
        """Instructions inside copies of repeats (redundant: only the copies after the first)."""
        mark = set()
        for r in self.repeats:
            for s in (r.starts[1:] if redundant else r.starts):
                mark.update(range(s, s + r.length))
        return len(mark)


def analyze(name: str, p: inlines.Prepared, canon=None, identity: bool = True, values: bool = True,
            k: int = KGRAM) -> Report:
    seq, shapes, ids = sequence(p, canon)
    keys = [(s, i) for s, i in zip(shapes, ids)] if identity else list(shapes)
    reps = find_repeats(keys, k)
    if values:
        reps = check_values(p.flow, seq, reps)
    reps = [r for r in reps if len(r.starts) >= MIN_COPIES]
    return Report(name, len(seq), sorted(reps, key=lambda r: r.starts[0]), seq,
                  [p.flow.nodes[i].addr for i in seq])


def shuffled(keys: list, seed: int) -> list:
    out = list(keys)
    random.Random(seed).shuffle(out)
    return out
