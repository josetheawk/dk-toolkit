"""A model of Metrowerks CodeWarrior's PowerPC instruction scheduler, for `dk sched`.

The compiler schedules each basic block with a list scheduler (the public GC 2.x decompilation's Scheduler.c and
MachineSimulation750.c; GC/1.3.2 is the same code): the dependence DAG is built from the block's instructions in
codegen order, each node gets its height (longest latency path to the block end) and deadline (critical path
minus height), then every cycle up to `issue_width` (2 on the 750) nodes are issued. Among the nodes that are
ready (no unissued predecessor, earliest cycle reached, the unit free), the FIRST in codegen order wins unless a
later one beats it on, in this order: deadline (its deadline has come and the current one's has not), uncovering
(more successors waiting on it alone), height, and before register colouring only, the lower opcode rank. So the
codegen order (statement order, operand order) only decides ties.

`-O4,p` runs the scheduler twice: before colouring on every body block, and after colouring on the prologue and
epilogue blocks and on every block colouring or the peephole changed (a coalesced copy deleted, two instructions
merged); that pass sees real registers, so register reuse adds ordering edges.

The records come from a compiler-oracle runner (contract `--sched`, schema sched.v1): per scheduled block, the
codegen-order instructions (opcode, flags, operands, source line), the dependence DAG as the compiler built it
(latency, height, deadline, predecessor count, successors with edge latencies), the scheduled order, and the
opcode table (name, rank, unit, latency, cycles). This module simulates a record, explains a pair of
instructions issued in the other order, and tests codegen-order levers on the simulator.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter

GPR, FPR, SPR = 4, 3, 0
fIsBranch, fIsRead, fIsWrite, fIsCall = 1, 2, 4, 8
fSideEffects, fOpFromMemory, fOpToMemory, fLink, fRecordBit = 0x100, 0x20000, 0x40000, 0x1000000, 0x20000000
BPU, IU1, IU2, LSU1, LSU2, FPU1, FPU2, FPU3, SRU = range(9)
REG, MEM = 0, 3


class Node:
    __slots__ = ("i", "lat", "height", "latest", "preds", "succ", "earliest", "op", "flags", "args")

    def __init__(s, i, ins, lat):
        s.i, s.lat, s.height, s.latest, s.preds, s.succ, s.earliest = i, lat, lat, 0, 0, [], 0
        s.op, s.flags, s.args = ins["op"], ins["flags"], ins["args"]


def nodes_of(rec: dict, order: list[int] | None = None) -> list[Node]:
    """The record's DAG as nodes, in codegen order (or in `order`, a permutation of the codegen indexes: the
    same DAG fed in another order, which is what a source-order edit does when the moved instructions do not
    depend on the ones they pass)."""
    nodes = [Node(i, x, 0) for i, x in enumerate(rec["insns"])]
    for d in rec["dag"]:
        n = nodes[d["i"]]
        n.lat, n.height, n.latest, n.preds = d["lat"], d["height"], d["latest"], d["preds"]
        n.succ = [(nodes[j], l) for j, l in d["succ"]]
    return [nodes[i] for i in order] if order is not None else nodes


class M750:
    """The 750's issue model (MachineSimulation750.c): units, a 6-entry completion buffer, IU2 pairing."""

    def __init__(s, ops):
        s.ops = ops
        s.pipe = [None] * 9
        s.rem = [0] * 9
        s.iu1c = s.iu2c = None
        s.buf = []                              # completion buffer, in issue order: [node, completed]

    def t(s, n):
        return s.ops[str(n.op)]

    @staticmethod
    def dependent(a, b):
        if b is None or not b.args:
            return False
        d = b.args[0]
        if not (d[0] == REG and d[1] == GPR and d[2] & 2):
            return False
        return any(x[0] == REG and x[1] == GPR and x[2] & 3 and x[3] == d[3] for x in a.args)

    def can_issue(s, n):
        if len(s.buf) >= 6:
            return False
        st = s.t(n)["stage"]
        if st == IU2:
            c1, c2 = s.pipe[IU1] is None, s.pipe[IU2] is None
            if not c1 and not c2:
                return False
            if c1 and c2:
                return True
            chk = s.pipe[IU1] if not c1 else s.pipe[IU2]
            if s.dependent(n, chk) or s.dependent(n, s.iu1c) or s.dependent(n, s.iu2c):
                return False
        elif s.pipe[st] is not None:
            return False
        if n.flags & fIsWrite and s.pipe[LSU2] is not None and s.pipe[LSU2].flags & fIsWrite:
            return False
        return True

    def issue(s, n):
        t = s.t(n)
        st = t["stage"]
        s.buf.append([n, False])
        if st == IU2 and s.pipe[IU1] is None:
            st = IU1
        s.pipe[st], s.rem[st] = n, t["cycles"][0]

    def complete(s, st):
        n = s.pipe[st]
        for e in s.buf:
            if e[0] is n:
                e[1] = True
                break
        s.pipe[st] = None
        if st == IU1:
            s.iu1c = n
        elif st == IU2:
            s.iu2c = n

    def adv(s, first, old, new):
        n = s.pipe[old]
        s.pipe[new], s.rem[new] = n, s.t(n)["cycles"][new - first]
        s.pipe[old] = None

    def clock(s):
        s.iu1c = s.iu2c = None
        for st in range(9):
            if s.pipe[st] is not None and s.rem[st]:
                s.rem[st] -= 1
        for _ in range(2):
            if s.buf and s.buf[0][1]:
                s.buf.pop(0)
            else:
                break
        for st in (IU1, LSU2, FPU3, SRU, BPU, IU2):
            if s.pipe[st] is not None and s.rem[st] == 0:
                s.complete(st)
        if s.pipe[FPU1] is not None and s.rem[FPU1] == 0 and s.t(s.pipe[FPU1])["name"] in ("FDIV", "FDIVS"):
            s.complete(FPU1)
        if s.pipe[FPU2] is not None and s.rem[FPU2] == 0 and s.pipe[FPU3] is None:
            s.adv(FPU1, FPU2, FPU3)
        if s.pipe[FPU1] is not None and s.rem[FPU1] == 0 and s.pipe[FPU2] is None:
            s.adv(FPU1, FPU1, FPU2)
        if s.pipe[LSU1] is not None and s.rem[LSU1] == 0 and s.pipe[LSU2] is None:
            s.adv(LSU1, LSU1, LSU2)


def uncovering(n):
    return sum(1 for m, _ in n.succ if m.preds == 1)


def select(head, counter, mach, ops, pre, why=None):
    """selectinstruction; `why` (a list) gets (winner, loser, rule) for every comparison."""
    ready = [n for n in head if n.preds == 0 and n.earliest <= counter and mach.can_issue(n)]
    if not ready:
        return None
    node = ready[0]
    for c in ready[1:]:
        if not (node.latest > counter or c.latest <= counter):
            if why is not None: why.append((node, c, "deadline"))
            continue
        if node.latest > counter and c.latest <= counter:
            if why is not None: why.append((c, node, "deadline"))
            node = c
            continue
        nu, cu = uncovering(node), uncovering(c)
        if nu != cu:
            if why is not None: why.append(((node, c) if nu > cu else (c, node)) + ("uncovering",))
            node = node if nu > cu else c
            continue
        if node.height != c.height:
            if why is not None: why.append(((node, c) if node.height > c.height else (c, node)) + ("height",))
            node = node if node.height > c.height else c
            continue
        if pre:
            rn, rc = ops[str(node.op)]["rank"], ops[str(c.op)]["rank"]
            if rn != rc:
                if why is not None: why.append(((node, c) if rn < rc else (c, node)) + ("rank",))
                node = node if rn < rc else c
                continue
        if why is not None: why.append((node, c, "codegen order"))
    return node


def schedule(nodes, ops, pre=True, trace=None):
    """scheduleblock: the issue order (codegen indexes) and each one's cycle; `trace` gets (cycle, node,
    comparisons) per pick."""
    for n in nodes:
        n.earliest = 0
    preds0 = {n.i: n.preds for n in nodes}
    head = list(nodes)
    mach = M750(ops)
    order, cyc = [], {}
    counter = 0
    guard = 0
    while head:
        for _ in range(2):
            if not head:
                break
            why = [] if trace is not None else None
            n = select(head, counter, mach, ops, pre, why)
            if n is None:
                break
            if trace is not None:
                trace.append((counter, n.i, why))
            for m, l in n.succ:
                m.preds -= 1
                m.earliest = max(m.earliest, counter + l)
            order.append(n.i)
            cyc[n.i] = counter
            mach.issue(n)
            head.remove(n)
        mach.clock()
        counter += 1
        guard += 1
        if guard > 100000:
            raise RuntimeError("scheduler did not finish")
    for n in nodes:
        n.preds = preds0[n.i]
    return order, cyc


def simulate(rec: dict, order: list[int] | None = None) -> tuple[list[int], list]:
    trace: list = []
    out, _ = schedule(nodes_of(rec, order), rec["_ops"], rec["pre"], trace)
    return out, trace


def check(doc: dict) -> Counter:
    """Model self-test: every record re-simulated from its dumped DAG against the compiler's own order."""
    c: Counter = Counter()
    for rec in records(doc):
        c[("pre" if rec["pre"] else "post", simulate(rec)[0] == rec["out"])] += 1
    return c


def records(doc: dict, fn: str | None = None) -> list[dict]:
    """The usable block records (with the opcode table attached), optionally of one function name."""
    out = []
    for r in doc.get("blocks", []):
        if "dag" in r and "out" in r and (fn is None or r.get("fn") == fn):
            r["_ops"] = doc["ops"]
            out.append(r)
    return out


# ---- matching the records to objdiff rows ---------------------------------------------------------------

EXT = {"clrlwi": "rlwinm", "slwi": "rlwinm", "srwi": "rlwinm", "clrrwi": "rlwinm", "rotlwi": "rlwinm",
       "extlwi": "rlwinm", "extrwi": "rlwinm", "rotlw": "rlwnm", "inslwi": "rlwimi", "insrwi": "rlwimi",
       "cmpwi": "cmpi", "cmplwi": "cmpli", "cmpw": "cmp", "cmplw": "cmpl", "addic.": "addicr", "nop": "ori",
       "subi": "addi", "subic": "addic", "subis": "addis", "not": "nor", "mtlr": "mtspr", "mflr": "mfspr",
       "mtctr": "mtspr", "mfctr": "mfspr", "bdnz": "bcond", "bdz": "bcond", "crclr": "crxor", "crset": "creqv",
       "sub": "subf", "subc": "subfc"}
_BR = re.compile(r"^b(eq|ne|lt|gt|le|ge|ns|so|un|nu)(lr|ctr)?[+-]?$")
_REG = re.compile(r"\b[rf]\d+\b")


def family_asm(text: str) -> str:
    """A disassembled mnemonic as the compiler's opcode family (extended mnemonics folded)."""
    m = text.split()[0] if text else ""
    if _BR.match(m) or m in ("bdnzlr", "bdzlr"):
        return "bcond"
    if m in EXT:
        return EXT[m]
    return m[:-1] + "r" if m.endswith(".") else m


def family_op(name: str) -> str:
    n = name.lower()
    if n in ("bf", "bt", "bc", "bclr", "bfl", "btl"):
        return "bcond"
    if n in ("mtlr", "mtctr"):
        return "mtspr"
    if n in ("mflr", "mfctr"):
        return "mfspr"
    return n


def _txt(e: dict) -> str:
    return e.get("instruction", {}).get("formatted", "") if e else ""


def _mask(t: str) -> str:
    return _REG.sub("R", t)


def short_name(symbol: str) -> str:
    """The compiler's name for a function before mangling (what the records carry)."""
    return symbol if symbol.startswith("__") else symbol.split("__")[0]


def row_map(recs: list[dict], ours: list[dict]) -> dict:
    """Our objdiff rows -> (record, node) per pass, by aligning our final order with each pass's scheduled
    output (a record's `out` is that block's final order unless a later pass or the peephole changed it)."""
    fam_rows = [family_asm(_txt(e)) for e in ours]
    maps = {}
    for pre in (False, True):
        seq = [(r, k) for r in recs if r["pre"] == pre for k in r["out"]]
        fam = [family_op(r["_ops"][str(r["insns"][k]["op"])]["name"]) for r, k in seq]
        sm = difflib.SequenceMatcher(None, fam_rows, fam, autojunk=False)
        m = {}
        for b in sm.get_matching_blocks():
            for d in range(b.size):
                m[b.a + d] = seq[b.b + d]
        maps[pre] = m
    return maps


def _record_of(maps: dict, rows: list[int]):
    for pre in (False, True):
        m = maps[pre]
        if rows and all(i in m for i in rows) and len({id(m[i][0]) for i in rows}) == 1:
            return m[rows[0]][0], [m[i][1] for i in rows]
    return None


def regions(classes: list[dict]) -> list[tuple[int, int]]:
    """Runs of rows that residue.classify calls SCHEDULE."""
    rows = sorted(r["row"] for r in classes if r["class"] == "SCHEDULE")
    out: list[list[int]] = []
    for i in rows:
        if out and i <= out[-1][-1] + 1:
            out[-1].append(i)
        else:
            out.append([i])
    return [(c[0], c[-1]) for c in out]


# ---- explaining a pair ----------------------------------------------------------------------------------

def decide(rec: dict, x: int, y: int) -> dict:
    """Why the scheduler issued y before x: the pick of y, whether x was ready, and every comparison."""
    _, trace = simulate(rec)
    for cyc, k, why in trace:
        if k != y:
            continue
        seen = {w.i for w, l, _ in why} | {l.i for w, l, _ in why} | {y}
        rules = [(w.i, l.i, rule) for w, l, rule in why]
        direct = next((rule for w, l, rule in rules if w == y and l == x), None)
        won = [rule for w, l, rule in rules if w == y]
        rule = direct or (won[-1] if won else "codegen order")
        return {"cycle": cyc, "ready": x in seen, "comparisons": rules,
                "rule": rule if x in seen else "not ready"}
    return {"rule": "unknown"}


def levers(rec: dict, x: int, y: int) -> list[str]:
    """Codegen-order edits the simulator says make x issue before y (same DAG, other input order): x moved
    before y, y moved after x, x's whole source line moved before y's."""
    base = list(range(len(rec["insns"])))
    line = lambda k: rec["insns"][k]["src"]
    cands = []
    if x > y:
        cands.append((f"line {line(x)}'s instruction earlier (before line {line(y)}'s)",
                      base[:y] + [x] + [j for j in base[y:] if j != x]))
    else:
        cands.append((f"line {line(y)}'s instruction later (after line {line(x)}'s)",
                      base[:x + 1] + [y] + [j for j in base[x + 1:] if j != y]))
    if line(x) != line(y):
        grp = [j for j in base if line(j) == line(x)]
        rest = [j for j in base if line(j) != line(x)]
        at = sum(1 for j in rest if j < min(x, y))
        cands.append((f"statement order: line {line(x)} before line {line(y)}", rest[:at] + grp + rest[at:]))
    good = []
    for label, order in cands:
        out, _ = simulate(rec, order)
        if out.index(x) < out.index(y):
            good.append(label)
    return good


def pre_of(recs: list[dict], rec: dict) -> tuple[dict | None, dict]:
    """The pre-colouring record whose output became this post-colouring record's input, and the node map."""
    ops = rec["_ops"]
    key = lambda r, k: family_op(ops[str(r["insns"][k]["op"])]["name"]) + "@%s" % r["insns"][k]["src"]
    post = [key(rec, k) for k in range(len(rec["insns"]))]
    best = None
    for r in recs:
        if not r["pre"]:
            continue
        sm = difflib.SequenceMatcher(None, [key(r, k) for k in r["out"]], post, autojunk=False)
        score = sum(b.size for b in sm.get_matching_blocks())
        if best is None or score > best[0]:
            best = (score, r, sm)
    if best is None or best[0] < max(2, len(post) // 2):
        return None, {}
    _, r, sm = best
    return r, {b.b + d: r["out"][b.a + d] for b in sm.get_matching_blocks() for d in range(b.size)}


def explain(doc: dict, symbol: str, target: list[dict], ours: list[dict], classes: list[dict]) -> list[dict]:
    """Every SCHEDULE region of one function: the window where both sides hold the same instructions, the
    block record holding ours, the first pair issued in the other order, the deciding rule, and the
    codegen-order levers that reproduce the target's order on the simulator (followed into the pre-colouring
    pass when the post-colouring pass only kept its input order)."""
    recs = records(doc, short_name(symbol))
    maps = row_map(recs, ours)
    n = max(len(target), len(ours))
    rep = []
    for a, b in regions(classes):
        r: dict = {"rows": [a, b]}
        win = None
        for k in range(5):
            lo, hi = max(0, a - k), min(n - 1, b + k)
            tr = [i for i in range(lo, hi + 1) if i < len(target) and _txt(target[i])]
            orr = [i for i in range(lo, hi + 1) if i < len(ours) and _txt(ours[i])]
            if Counter(_mask(_txt(target[i])) for i in tr) == Counter(_mask(_txt(ours[i])) for i in orr):
                win = (tr, orr)
                break
        if win is None:
            r.update(status="mixed", target=[_txt(target[i]) for i in range(a, min(b + 1, len(target)))],
                     ours=[_txt(ours[i]) for i in range(a, min(b + 1, len(ours)))])
            rep.append(r)
            continue
        tr, orr = win
        r["target"], r["ours"] = [_txt(target[i]) for i in tr], [_txt(ours[i]) for i in orr]
        hit = _record_of(maps, orr)
        if hit is None:
            r["status"] = "unmapped"
            rep.append(r)
            continue
        rec, nodes = hit
        r.update({"pass": "pre" if rec["pre"] else "post", "block": rec["block"]})
        om = [_mask(t) for t in r["ours"]]
        free, tseq = list(range(len(om))), []
        for t in r["target"]:
            j = next(j for j in free if om[j] == _mask(t))
            free.remove(j)
            tseq.append(nodes[j])
        i = next((i for i in range(len(nodes)) if nodes[i] != tseq[i]), None)
        if i is None:
            r["status"] = "same order"
            rep.append(r)
            continue
        x, y = tseq[i], nodes[i]
        line = lambda rr, k: rr["insns"][k]["src"]
        r.update(x={"node": x, "line": line(rec, x), "text": r["target"][i]},
                 y={"node": y, "line": line(rec, y), "text": r["ours"][i]},
                 decide=decide(rec, x, y), levers=levers(rec, x, y))
        if not rec["pre"] and r["decide"].get("rule") == "codegen order":
            pr, m = pre_of(recs, rec)
            if pr is not None and x in m and y in m:
                r["pre"] = {"block": pr["block"], "x": m[x], "y": m[y], "decide": decide(pr, m[x], m[y]),
                            "levers": levers(pr, m[x], m[y])}
        final = r.get("pre", r)
        r["status"] = "codegen order" if final["levers"] else "dag"
        rep.append(r)
    return rep


ADVICE = {
    "codegen order": "a tie: only the codegen order decides; the lever moves the target's instruction earlier",
    "deadline": "the other instruction is on the critical path (its deadline has come): change the dependence "
                "chains (which value feeds which, a temporary, the expression's shape), not the statement order",
    "uncovering": "the other instruction is the only predecessor of more instructions (often a copy into an "
                  "argument register or a temporary): remove or add that copy (inline argument, named local vs "
                  "temporary, use the value directly)",
    "height": "the other instruction heads a longer latency chain: change the expression's shape",
    "rank": "before colouring the lower opcode rank wins a tie (e.g. loads before li): only a DAG change helps",
    "not ready": "the target's instruction waits on a predecessor (or its unit): it depends on something "
                 "ours computes later; change which value it reads or where that is computed",
}
