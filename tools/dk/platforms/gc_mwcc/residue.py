"""What is left of a register near miss once the target's colours are forced into the compiler.

`regs --reach` says whether forcing the target colours gives the exact function. When it does not, the rows
that still differ are not a colouring problem, and each class has its own source lever:

  OPERAND-SWAP  same commutative instruction, same operands in another order: write the operands of that
                expression (or comparison) the other way round
  COALESCE      a register copy (`mr`/`fmr`) one side has and the other not, or made the other way: a copy stays
                separate only while both values are live at once, so use the old value later (or stop using it)
  STACK         a load or store one side has and the other not: the target re-reads memory (a stack local whose
                address is taken, a member or global) where ours keeps a register, or the reverse
  FRAME         prologue/epilogue or stack-slot differences (save area, frame size, slot order)
  SCHEDULE      the same instruction a few rows away on the other side: statement or expression order
  REGISTER      still another register: the force did not take for this value (an oracle limit, not a lever)
  RELOCATION    the same text both sides, or the same branch to another address: a relocation points elsewhere
                (a literal, data object or code placed otherwise, often allowed by `dk check --function`'s
                hand-link in an unfinished unit)
  OTHER         anything else

Rows come from objdiff's aligned instruction lists (target, ours), compared with registers named.
"""

from __future__ import annotations

import re
from collections import Counter

CLASSES = ("OPERAND-SWAP", "COALESCE", "STACK", "FRAME", "SCHEDULE", "REGISTER", "RELOCATION", "OTHER")
COMMUTATIVE = {"add", "add.", "addc", "adde", "mullw", "mulhw", "mulhwu", "and", "and.", "or", "or.", "xor", "nand",
               "nor", "eqv", "fadd", "fadds", "fmul", "fmuls", "fcmpu", "fcmpo", "cmpw", "cmplw", "cmp", "cmpl",
               "ps_add", "ps_mul"}
COPIES = {"mr", "fmr", "mr."}
FRAME_OPS = {"stwu", "mflr", "mtlr", "stmw", "lmw", "blr"}
MEMORY = re.compile(r"^(lbz|lhz|lha|lwz|lfs|lfd|stb|sth|stw|stfs|stfd|psq_l|psq_st)(u|x|ux)?$")
REG = re.compile(r"\b[rf]\d+\b")
BRANCH_TO = re.compile(r"0x[0-9a-fA-F]+$")
WINDOW = 4                     # rows either way a SCHEDULE match may sit
EDGE = 12                      # rows at either end counted as prologue/epilogue


def _text(entry: dict) -> str:
    return entry.get("instruction", {}).get("formatted", "") if entry else ""


def _split(text: str) -> tuple[str, list[str]]:
    op, _, rest = text.partition(" ")
    return op, [a.strip() for a in rest.split(",")] if rest else []


def _masked(text: str) -> str:
    return REG.sub("R", text)


def _is_memory(op: str) -> bool:
    return bool(MEMORY.match(op or ""))


def _base(args: list[str]) -> str | None:
    m = re.search(r"\((r\d+)\)", args[-1]) if args else None
    return m.group(1) if m else None


def line_of(entry: dict) -> int | None:
    return entry.get("instruction", {}).get("line_number") if entry else None


def classify_row(i: int, t: dict, o: dict, ts: list[str], os_: list[str], n: int) -> str:
    """One differing aligned row. ts/os_: each side's register-masked texts (for SCHEDULE)."""
    tt, ot = _text(t), _text(o)
    top, targs = _split(tt)
    oop, oargs = _split(ot)
    edge = i < EDGE or i >= n - EDGE
    if tt and tt == ot:                # the same text: objdiff differs on a relocation's target
        return "RELOCATION"
    if tt and ot and top == oop and top.startswith("b") and BRANCH_TO.sub("@", tt) == BRANCH_TO.sub("@", ot):
        return "RELOCATION"            # the same branch to another address: code placed elsewhere (the hand-link decides)
    if tt and ot and top == oop:
        if sorted(targs) == sorted(oargs) and targs != oargs:
            if top in COPIES:
                return "COALESCE"
            if top in COMMUTATIVE or (top.startswith("f") and top.rstrip("s").endswith(("add", "mul"))):
                return "OPERAND-SWAP"
        if _masked(tt) == _masked(ot):
            if edge and (top in FRAME_OPS or (_is_memory(top) and _base(targs) == "r1")):
                return "FRAME"
            return "REGISTER"
        if _is_memory(top) and _base(targs) == "r1" and _base(oargs) == "r1":
            return "FRAME"
    one = tt if not ot else ot if not tt else None
    if one is not None:
        op, args = _split(one)
        if op in COPIES:
            return "COALESCE"
        if edge and (op in FRAME_OPS or (_is_memory(op) and _base(args) == "r1")):
            return "FRAME"
    near = range(max(0, i - WINDOW), min(n, i + WINDOW + 1))
    if (tt and any(j != i and os_[j] == _masked(tt) for j in near)) or \
            (ot and any(j != i and ts[j] == _masked(ot) for j in near)):
        return "SCHEDULE"
    if one is not None:
        op, args = _split(one)
        if _is_memory(op):
            return "FRAME" if edge and _base(args) == "r1" else "STACK"
    if tt and ot and top in COPIES and oop in COPIES:
        return "COALESCE"
    if edge and (top in FRAME_OPS or oop in FRAME_OPS):
        return "FRAME"
    return "OTHER"


def classify(target: list[dict], ours: list[dict]) -> list[dict]:
    """Every differing aligned row: {row, class, target, ours, line}."""
    n = max(len(target), len(ours))
    pad = lambda xs: xs + [{}] * (n - len(xs))
    target, ours = pad(list(target)), pad(list(ours))
    ts, os_ = [_masked(_text(t)) for t in target], [_masked(_text(o)) for o in ours]
    rows_at = [{_address(x): i for i, x in enumerate(side) if x} for side in (target, ours)]
    out = []
    for i, (t, o) in enumerate(zip(target, ours)):
        if not (t.get("diff_kind") or o.get("diff_kind")):
            continue
        r = {"row": i, "class": classify_row(i, t, o, ts, os_, n), "target": _text(t), "ours": _text(o),
             "line": line_of(o) or line_of(t)}
        if r["class"] == "RELOCATION" and BRANCH_TO.search(r["target"]) and r["target"].startswith("b"):
            # where each side lands, as the source line of OUR code on that aligned row (ours has the line table)
            dest = [rows_at[k].get(int(BRANCH_TO.search(x)[0], 16)) for k, x in ((0, r["target"]), (1, r["ours"]))]
            r["target_dest_line"], r["ours_dest_line"] = (_line_near(ours, d) for d in dest)
        out.append(r)
    return out


def _address(entry: dict) -> int:
    return int(entry.get("instruction", {}).get("address", 0))


def _line_near(ours: list[dict], row: int | None) -> int | None:
    """The source line of our instruction at `row`, or of the next one that has a line."""
    if row is None:
        return None
    return next((line_of(x) for x in ours[row:row + 8] if line_of(x)), None)


LEVERS = {
    "OPERAND-SWAP": "write this expression's operands (or the comparison) the other way round at line {line}",
    "COALESCE": "a copy is kept or merged differently near line {line}: use the old value after the copy (keeps it "
                "separate) or stop using it (lets the compiler merge it)",
    "STACK": "the target re-reads memory where ours keeps a register (or the reverse) near line {line}: read the "
             "member/global again instead of a local, take the local's address, or the reverse",
    "FRAME": "frame layout: declaration order of stack locals, their sizes, or a local the target has and ours not",
    "SCHEDULE": "statement or expression order near line {line}",
    "REGISTER": "the forced colour did not take here (oracle limit); work the register order with close --moves oracle",
    "RELOCATION": "a relocation points elsewhere: literal or data order (dk diff --literals), or allowed by "
                  "dk check --function's hand-link while the unit is unfinished; a branch landing elsewhere: "
                  "make the code at the target's landing line follow this test (join tests with &&, break out of "
                  "the loop, move the statement out of the if)",
    "OTHER": "unclassified: read the rows",
}


def summary(rows: list[dict]) -> Counter:
    return Counter(r["class"] for r in rows)
