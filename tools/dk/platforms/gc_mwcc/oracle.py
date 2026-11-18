"""Compiler oracle reader: which source value each register holds, read from the compiler itself.

The oracle is a local runner outside the repo (a debugger script around the project's compiler); the
toolkit only knows its command line and its output. It is found through the DK_ORACLE environment
variable or dk.toml `[platform] oracle`, both a command: `<command> OUT.json --funcs NAMES -- <compile
argv>`. It runs the compile unchanged and writes, per compiled function, every value the register
allocator coloured (schema `vregs.v1`):

  {"schema": "vregs.v1", "compiler": "...", "exit": 0, "functions": [
     {"name": "fn_80001234", "insns": 40, "lines": [source line per final instruction],
      "vregs": [{"index": 35, "cls": "gpr", "kind": "temp|local|param|fe-temp", "name": "",
                 "color": 31, "coalesced_into": null, "spilled": false,
                 "defs": [instruction indices that write it], "reads": [...], "line": 12}]}]}

`index` is the value's creation number; the allocator colours values newest first, so a value created
later is coloured earlier. This module lines each value up with the target's instruction rows (objdiff)
and reports, per value, our register and the one the target uses at the same operands, plus a cause for
each mismatch (two values taking each other's registers: their creation order is reversed in the
target). Without a configured runner every entry point says so and nothing else changes.

Contract v2 (optional; a runner that lacks it still serves `--explain --oracle`):

- `<command> OUT.json --funcs NAMES --extra -- <argv>`: each record also carries `graph.<cls>.order`, the
  values (indexes) in the order the allocator coloured them; every other `graph` key is the runner's own and
  dk only passes it back to `--solve`. Each value also lists `operands` ([instruction, operand] pairs).
- `<command> OUT.json --funcs NAMES --force SPEC.json -- <argv>`: the same compile with the colouring
  steered: SPEC = {"<compiler name>": {"gpr"|"fpr": {"order": [value...], "call": N} | {"colors": {value: reg}}}}.
  A diagnosis only: a forced object is never kept.
- `<command> --solve IN.json OUT.json` (schema `reach.v1`): IN = {"schema", "record": one `--extra`
  record, "targets": {cls: {value: target register number}}} (or {"schema", "probe": true}, answered with
  {"schema", "modes": [...]}). OUT = {"schema", "reachable": true|false|null, "classes": {cls: {
  "reachable", "mode" (already/strict/loose), "reason", "rules": [{"before": V, "after": V, "flip", "cell"}]
  (an irredundant set of "colour `before` before `after`"; flip = ours does the opposite; cell =
  decl-order/param-order/temp-temp/cross-tier), "wrong_values": [V + target], "blockers",
  "force": {order, call} (an order that satisfies the rules), "decl_order": null | {"changed", "declare":
  [[name, value, line]...] (first = declared first), "rules", "force"}}}}, V = {index, kind, name, line, color}.

Contract v3 (optional; the probe's `modes` lists "temps"): `<command> OUT.json --funcs NAMES --extra --temps -- <argv>`
also gives every `fe-temp` value an `origin`: {"phase": front|linear|prepass|prop0|unroll|loop|cse0|prop1|cse1|late|
backend, "rank": the phase's place in that fixed order, "line": where the phase made it (CSE: the expression's
first redundant, i.e. second, occurrence; loop: the loop test; others: the first reference), "first_line" (CSE:
the original occurrence), "role" (front: arg|local|result of an inline expansion), "inline" (the inlined function)}.
`regs --why-temp` reads it (whytemp.py).
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA = "vregs.v1"
REACH = "reach.v1"
UNAVAILABLE = ("no compiler oracle configured: set DK_ORACLE (or dk.toml [platform] oracle) to a runner command "
               "`CMD OUT.json --funcs NAMES -- <compile argv>` that writes a vregs.v1 table; without it use "
               "`regs --explain` (debug info) instead")


def runner(cfg) -> list[str] | None:
    """The oracle command (argv prefix), or None when none is configured or its program is missing."""
    raw = os.environ.get("DK_ORACLE") or (cfg.platform_options.get("oracle") if cfg is not None else None)
    if not raw:
        return None
    argv = [os.path.expanduser(a) for a in shlex.split(str(raw))]
    prog = argv[-1] if argv[0].startswith("python") and len(argv) > 1 else argv[0]
    if os.sep in prog and not Path(prog).exists():
        return None
    return argv


def oracle_name(symbol: str) -> str:
    """The name the compiler gives a function before mangling: `fn_80001234__5ClassFv` -> fn_80001234,
    `__ct__5ClassFv` -> __ct, an extern "C" name as is."""
    if symbol.startswith("__"):
        return "__" + symbol[2:].split("__", 1)[0]
    return symbol.split("__", 1)[0] if "__" in symbol else symbol


def wrap(command: list[str], out: Path, symbol: str | None, extra: bool = False, force: Path | None = None,
         temps: bool = False):
    """A compile-argv rewriter for reorder.compile_variant: runs the compile under the oracle (`extra`: with
    the colouring order; `force`: a write-mode spec file, contract v2; `temps`: temp origins, contract v3)."""
    def go(argv: list[str]) -> list[str]:
        funcs = ["--funcs", oracle_name(symbol)] if symbol else []
        more = (["--extra"] if extra else []) + (["--force", str(force)] if force is not None else []) + \
            (["--temps"] if temps else [])
        return list(command) + [str(out)] + funcs + more + ["--"] + list(argv)
    return go


# ---- contract v2: order solver and write mode ----------------------------------------------------

def _call_solve(command: list[str], request: dict, work: Path, timeout: int = 300) -> dict:
    import subprocess
    work.mkdir(parents=True, exist_ok=True)
    src, out = work / "reach-in.json", work / "reach-out.json"
    src.write_text(json.dumps(request))
    out.unlink(missing_ok=True)
    try:
        subprocess.run(list(command) + ["--solve", str(src), str(out)], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"error": f"the oracle's solver did not run: {e}"}
    try:
        doc = json.loads(out.read_text())
    except (OSError, ValueError):
        return {"error": "the oracle runner has no `--solve` mode (contract v2: reach.v1)"}
    if doc.get("schema") != REACH:
        return {"error": f"unknown solver schema {doc.get('schema')!r}"}
    return doc


def modes(command: list[str], work: Path) -> list[str]:
    """The runner's modes from the reach.v1 probe ([] when it has none)."""
    return list(_call_solve(command, {"schema": REACH, "probe": True}, work, timeout=30).get("modes") or [])


def supports_solve(command: list[str], work: Path) -> bool:
    """True when the runner answers the reach.v1 probe (contract v2)."""
    return "solve" in modes(command, work)


def target_colours(vals: list[Value]) -> dict[str, dict[str, int]]:
    """{cls: {value index: register number}} of every value with an aligned target register of its class."""
    out: dict[str, dict[str, int]] = {}
    for v in vals:
        if v.target is not None and v.target[:1] == ("r" if v.cls == "gpr" else "f") and v.target[1:].isdigit():
            out.setdefault(v.cls, {})[str(v.index)] = int(v.target[1:])
    return out


def solve(command: list[str], record: dict, vals: list[Value], work: Path) -> dict:
    """The runner's reach.v1 answer for one function record compiled with `--extra`."""
    if not record.get("graph"):
        return {"error": "the oracle record has no colouring order (the runner needs `--extra`, contract v2)"}
    return _call_solve(command, {"schema": REACH, "record": record, "targets": target_colours(vals)}, work)


def force_spec(symbol: str, analysis: dict, lever: str = "rules") -> dict | None:
    """A `--force` spec from a solve answer: `rules` = the order that satisfies the minimal rules, `decl` = the
    declaration-order lever's order. None when that lever has no order."""
    spec: dict = {}
    for cls, r in (analysis.get("classes") or {}).items():
        f = r.get("force") if lever == "rules" else ((r.get("decl_order") or {}).get("force")
                                                      if isinstance(r.get("decl_order"), dict) else None)
        if f and f.get("order"):
            spec.setdefault(oracle_name(symbol), {})[cls] = f
    return spec or None


def colour_spec(symbol: str, vals: list[Value]) -> dict | None:
    """A `--force` spec that gives every value with a known target register that register."""
    spec = {cls: {"colors": t} for cls, t in target_colours(vals).items() if t}
    return {oracle_name(symbol): spec} if spec else None


def flat_rules(analysis: dict) -> list[dict]:
    """Every class's rules as one list: {cls, a, b, flip, cell, before, after} (a coloured before b)."""
    out = []
    for cls, r in (analysis.get("classes") or {}).items():
        for x in r.get("rules") or []:
            out.append({"cls": cls, "a": x["before"]["index"], "b": x["after"]["index"], "flip": bool(x.get("flip")),
                        "cell": x.get("cell", ""), "before": x["before"], "after": x["after"]})
    return out


def wanted_orders(analysis: dict) -> list[list[tuple[str, int | None]]]:
    """The declaration-order lever per class: [(name, line)...] in the order to declare, when it changes."""
    out = []
    for r in (analysis.get("classes") or {}).values():
        d = r.get("decl_order")
        if isinstance(d, dict) and d.get("changed"):
            out.append([(x[0], x[2] if len(x) > 2 else None) for x in d.get("declare", []) if x and x[0]])
    return out


def _live(record: dict) -> dict:
    """(cls, operand positions) -> value index, for values the record uses (a value's identity across compiles)."""
    out: dict = {}
    dup: set = set()
    for v in record.get("vregs", []):
        if v.get("coalesced_into") is not None or not v.get("operands"):
            continue
        key = (v.get("cls"), frozenset(tuple(x) for x in v["operands"]))
        if key in out:
            dup.add(key)
        out[key] = v["index"]
    return {k: i for k, i in out.items() if k not in dup}


def map_values(base: dict, record: dict) -> dict:
    """{(cls, base value index): record value index} for values found at the same operand positions."""
    theirs = _live(record)
    return {(k[0], i): theirs[k] for k, i in _live(base).items() if k in theirs}


def rule_status(rules: list[dict], base: dict, record: dict) -> list[bool | None]:
    """Per rule (values of `base`): True when `record` colours them in the rule's order, False when the
    other way, None when a value cannot be found in `record` (by the same operand positions)."""
    ours = {i: k for k, i in _live(base).items()}
    theirs = _live(record)
    out: list[bool | None] = []
    for r in rules:
        order = (((record.get("graph") or {}).get(r["cls"]) or {}).get("order")) or []
        pos = {v: k for k, v in enumerate(order)}
        ka, kb = ours.get(r["a"]), ours.get(r["b"])
        a = theirs.get(ka) if ka and ka[0] == r["cls"] else None
        b = theirs.get(kb) if kb and kb[0] == r["cls"] else None
        out.append(None if a is None or b is None or a not in pos or b not in pos else pos[a] < pos[b])
    return out


def load(path: Path) -> dict:
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {"error": "the oracle wrote no table"}
    if doc.get("schema") != SCHEMA and "error" not in doc:
        return {"error": f"unknown oracle schema {doc.get('schema')!r}"}
    return doc


def find_record(doc: dict, symbol: str, size: int | None, ordinal: int = 0) -> dict | None:
    """The function's record: same compiler name and, when known, the same size; `ordinal` picks among
    same-name same-size functions (overloads), in object order."""
    name = oracle_name(symbol)
    hits = [f for f in doc.get("functions", []) if f.get("name") == name]
    if size is not None:
        hits = [f for f in hits if f.get("insns", 0) * 4 == size] or hits
    if not hits:
        return None
    return hits[min(ordinal, len(hits) - 1)]


def _mnemonic(entry: dict) -> str | None:
    for a in entry.get("instruction", {}).get("parts", []):
        if "opcode" in a:
            return a["opcode"].get("mnemonic")
    return None


def _regs(entry: dict) -> list[str]:
    import re
    out = []
    for a in entry.get("instruction", {}).get("parts", []):
        if "arg" in a and isinstance(a["arg"], dict):
            v = str(a["arg"].get("opaque", ""))
            if re.fullmatch(r"[rf]\d+", v):
                out.append(v)
    return out


@dataclass
class Value:
    index: int
    cls: str                 # "gpr" or "fpr"
    kind: str                # temp, local, param, fe-temp
    name: str
    line: int | None
    ours: str                # our register (r31, f1)
    target: str | None       # the target's register at the same operands (None: no aligned operand)
    insn: str = ""           # our first instruction writing it (or reading it, for a parameter)
    address: str = ""
    votes: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.target is None or self.target == self.ours

    @property
    def label(self) -> str:
        who = f"{self.kind} {self.name}" if self.name else self.kind
        return f"#{self.index} {who}"


def values(record: dict, target_rows: list[dict], our_rows: list[dict]) -> list[Value] | None:
    """Every coloured value of the record lined up with objdiff's rows (target, ours: one row each,
    padded to one length). None when the record's instructions cannot be lined up with ours (emitted_rows)."""
    ours_at = [i for i, o in enumerate(our_rows) if "instruction" in o]
    emitted = emitted_rows(record, [_mnemonic(our_rows[i]) for i in ours_at],
                           lines=[our_rows[i]["instruction"].get("line_number") for i in ours_at])
    if emitted is None:
        return None
    ours_at = [ours_at[e] if e is not None else None for e in emitted]   # record index -> row (None: not emitted)
    out: list[Value] = []
    for v in record.get("vregs", []):
        if v.get("color") is None or v.get("spilled") or v.get("coalesced_into") is not None:
            continue
        if not v.get("defs") and not v.get("reads"):
            continue
        reg = ("r" if v["cls"] == "gpr" else "f") + str(v["color"])
        votes: Counter = Counter()
        first = None
        for group in (v.get("defs", []), v.get("reads", [])):
            for k in group:
                if k >= len(ours_at) or ours_at[k] is None:
                    continue
                row = ours_at[k]
                o, t = our_rows[row], target_rows[row] if row < len(target_rows) else {}
                if first is None:
                    first = o
                if "instruction" not in t or _mnemonic(t) != _mnemonic(o):
                    continue
                a, b = _regs(o), _regs(t)
                if len(a) != len(b):
                    continue
                for x, y in zip(a, b):
                    if x == reg:
                        votes[y] += 1
            if votes:
                break
        target = votes.most_common(1)[0][0] if votes else None
        ins = (first or {}).get("instruction", {})
        out.append(Value(v["index"], v["cls"], v.get("kind", "temp"), v.get("name", ""), v.get("line"), reg, target,
                         ins.get("formatted", ""), str(ins.get("address", "")), dict(votes)))
    out.sort(key=lambda x: (x.cls, -x.index))
    res = Values(out)
    lines = record.get("lines") or []
    wrong_regs = {v.ours for v in out if not v.ok}
    touched = {k for v in record.get("vregs", []) if (("r" if v.get("cls") == "gpr" else "f") + str(v.get("color"))) in wrong_regs
               for k in v.get("defs", []) + v.get("reads", [])}
    for k, row in enumerate(ours_at):
        if row is None or row >= len(target_rows) or k in touched:     # a wrong value's own operands: not order
            continue
        o, t = our_rows[row], target_rows[row]
        if "instruction" not in t or _mnemonic(t) != _mnemonic(o):
            continue
        a, b = _regs(o), _regs(t)
        if a != b and sorted(a) == sorted(b):
            line = lines[k] if k < len(lines) and lines[k] > 0 else None
            res.orders.append((line, o["instruction"].get("formatted", ""), t["instruction"].get("formatted", "")))
    return res


class Values(list):
    """oracle.values' result: the Value rows, plus `orders`: instructions whose registers are right but in
    another operand order (line, ours, target), a commutative operand order in the source."""
    def __init__(self, items=()):
        super().__init__(items)
        self.orders: list[tuple[int | None, str, str]] = []


def emitted_rows(record: dict, mnemonics: list[str | None], most: int = 3,
                 lines: list[int | None] | None = None) -> list[int | None] | None:
    """Record instruction index -> our instruction index (None: listed but not emitted). The assembler drops
    listed instructions (a branch to the next one), and giants or unrolled code drop many; when the counts
    differ the two sequences are aligned by dynamic programming in a band around the diagonal: a pair costs
    nothing when the record's opcode has been seen as our mnemonic and the record's source line (when it
    has one) is ours, two for another mnemonic and one for another line; a dropped record instruction or an
    extra one of ours costs two (ties drop late).
    The opcode -> mnemonic table is learnt from the sequences themselves (per window, the most consistent
    shift first, then the alignment) until it stops changing. None when the counts differ by more than `most` and by more than an
    eighth of the function, or when under 80% of the aligned pairs agree on the opcode."""
    ops = list(record.get("ops") or [])
    n, m = record.get("insns", len(ops)), len(mnemonics)
    if n == m:
        return list(range(n))
    extra = n - m
    if len(ops) != n or (abs(extra) > most and abs(extra) * 8 > n):
        return None
    rec_lines = list(record.get("lines") or [])
    use_lines = lines is not None and len(lines) == m and len(rec_lines) == n
    out = _align(ops, mnemonics, rec_lines if use_lines else None, lines if use_lines else None)
    if out is None:
        return None
    seen = _opcode_table([(ops[k], mnemonics[j]) for k, j in enumerate(out) if j is not None])
    pairs = [(ops[k], mnemonics[j]) for k, j in enumerate(out) if j is not None]
    if not pairs or sum(1 for p in pairs if p in seen) * 5 < len(pairs) * 4:
        return None
    return out


def _opcode_table(pairs: list[tuple]) -> set[tuple]:
    """(opcode, mnemonic) pairs that make up at least a fifth of that opcode's pairings (one opcode may stand
    for several mnemonics: rlwinm's forms)."""
    by: dict = {}
    for o, mn in pairs:
        by.setdefault(o, Counter())[mn] += 1
    return {(o, mn) for o, c in by.items() for mn, k in c.items() if k * 5 >= sum(c.values())}


def _window_pairs(ops: list, mns: list, lo: int, hi: int, size: int = 32) -> list[tuple]:
    """A first guess at the pairing: in each window of the record, the shift (within the band) whose pairs
    are the fewest distinct (opcode, mnemonic) pairs, i.e. the most consistent one."""
    out = []
    for start in range(0, len(ops), size):
        best = None
        for d in range(lo, hi + 1):
            span = range(start, min(start + size, len(ops)))
            pairs = [(ops[k], mns[k + d]) for k in span if 0 <= k + d < len(mns)]
            key = (len(set(pairs)) + len(span) - len(pairs), abs(d))     # a record entry left unpaired counts as new
            if best is None or key < best[0]:
                best = (key, pairs)
        if best is not None:
            out += best[1]
    return out


def _align(ops: list, mns: list, rec_lines: list | None, our_lines: list | None, pad: int = 8,
           rounds: int = 4) -> list[int | None] | None:
    n, m = len(ops), len(mns)
    lo, hi = min(0, m - n) - pad, max(0, m - n) + pad          # band: j - k in [lo, hi]
    width = hi - lo + 1
    seen = _opcode_table(_window_pairs(ops, mns, lo, hi))
    big = 1 << 30

    def cost(k: int, j: int) -> int:
        c = 0 if (ops[k], mns[j]) in seen else 2
        if rec_lines is not None:
            a = rec_lines[k]
            if a is not None and a > 0 and a != our_lines[j]:
                c += 1
        return c

    out: list[int | None] | None = None
    for _ in range(rounds):
        dp = [[big] * width for _ in range(n + 1)]
        back = [bytearray(width) for _ in range(n + 1)]
        dp[0][-lo] = 0
        for k in range(n + 1):
            row, nxt = dp[k], (dp[k + 1] if k < n else None)
            for o in range(width):
                c = row[o]
                if c >= big:
                    continue
                j = k + lo + o
                if j < 0 or j > m:
                    continue
                if nxt is not None and j < m:                        # paired
                    cc = c + cost(k, j)
                    if cc < nxt[o]:
                        nxt[o], back[k + 1][o] = cc, 0
                if nxt is not None and o > 0 and nxt[o - 1] >= c + 2:  # record instruction not emitted (ties: late)
                    nxt[o - 1], back[k + 1][o - 1] = c + 2, 1
                if j < m and o + 1 < width and row[o + 1] > c + 2:   # an instruction of ours not in the record
                    row[o + 1], back[k][o + 1] = c + 2, 2
        end = m - n - lo
        if not (0 <= end < width) or dp[n][end] >= big:
            return None
        got: list[int | None] = [None] * n
        k, j = n, m
        while k > 0 or j > 0:
            step = back[k][j - k - lo]
            if step == 0:
                k, j = k - 1, j - 1
                got[k] = j
            elif step == 1:
                k -= 1
            else:
                j -= 1
        out = got
        again = _opcode_table([(ops[k], mns[j]) for k, j in enumerate(got) if j is not None])
        if again == seen:
            break
        seen = again
    return out


def registers_right(vals: list[Value]) -> tuple[int, int]:
    """(values coloured as the target, values with an aligned target operand)."""
    known = [v for v in vals if v.target is not None]
    return sum(1 for v in known if v.target == v.ours), len(known)


@dataclass
class Cause:
    kind: str                # swap, rotation, free
    values: list[Value]
    text: str


def causes(vals: list[Value], line_text=None) -> list[Cause]:
    """Why each wrong value got its register. Two values that take each other's registers: the one our
    compile creates later is coloured first and gets the register the target gives the other, so the
    target creates them in the other order (or one of them no longer overlaps the other). A longer
    cycle is a rotation: one value created at another point shifts the rest by one."""
    def where(v: Value) -> str:
        t = line_text(v.line) if line_text and v.line else None
        at = f"line {v.line}" if v.line else "no line"
        return f"{v.label} ({at}{': `' + t + '`' if t else ''}; `{v.insn}`)"

    wrong = [v for v in vals if not v.ok]

    def holder(v: Value, pool: list[Value]) -> Value | None:
        """the value of `pool` holding v's target register in ours, nearest in creation order"""
        hits = [w for w in pool if w is not v and w.cls == v.cls and w.ours == v.target]
        return min(hits, key=lambda w: abs(w.index - v.index)) if hits else None

    out: list[Cause] = []
    done: set[int] = set()
    for v in wrong:
        if id(v) in done:
            continue
        cycle, cur = [v], v
        while True:
            nxt = holder(cur, wrong)
            if nxt is None or nxt is v or nxt.ok or any(nxt is c for c in cycle) or len(cycle) > 8:
                closed = nxt is v
                break
            cycle.append(nxt)
            cur = nxt
        for c in cycle:
            done.add(id(c))
        if len(cycle) == 2 and closed:
            a, b = sorted(cycle, key=lambda x: -x.index)          # a: created later, coloured first
            out.append(Cause("swap", [a, b],
                             f"swap {a.ours}/{b.ours}: ours creates {where(a)} after {where(b)}, so it is coloured first; "
                             f"the target colours the other first: create {a.label}'s value before {b.label}'s "
                             f"(name/un-name or reorder the expression), or end their overlap"))
        elif len(cycle) > 2 and closed:
            names = ", ".join(f"{c.label} {c.ours}->{c.target}" for c in cycle)
            out.append(Cause("rotation", cycle,
                             f"rotation of {len(cycle)} values ({names}): one of them is created at another point in the "
                             f"target; first suspect {where(max(cycle, key=lambda c: abs(c.index - min(x.index for x in cycle))))}"))
        else:
            for c in cycle:
                h = holder(c, vals)
                extra = f" (ours gives {c.target} to {h.label})" if h else f" (ours gives {c.target} to no value)"
                out.append(Cause("free", [c], f"{where(c)}: ours {c.ours}, target {c.target}{extra}"))
    for line, ours, target in getattr(vals, "orders", []):
        t = line_text(line) if line_text and line else None
        out.append(Cause("operand-order", [], f"operand order: `{ours}` where the target has `{target}` "
                         f"({'line ' + str(line) if line else 'no line'}{': `' + t + '`' if t else ''}): "
                         f"write the commutative operands the other way round"))
    return out


def blamed_lines(vals: list[Value]) -> set[int]:
    """Source lines (1-based) of the wrong values: where a directed search edits."""
    return {v.line for v in vals if not v.ok and v.line and v.line > 0} | \
        {line for line, _o, _t in getattr(vals, "orders", []) if line}


def render(vals: list[Value], cs: list[Cause], compiler: str = "", line_text=None, top: int = 40) -> list[str]:
    right, known = registers_right(vals)
    out = [f"  oracle{(' ' + compiler) if compiler else ''}: {len(vals)} values, {right} of {known} coloured as the target"]
    out.append("  values, colour order (newest first); !! = another register in the target:")
    shown = [v for v in vals if not v.ok] + [v for v in vals if v.ok]
    for v in sorted(shown[:top], key=lambda x: (x.cls, -x.index)):
        t = line_text(v.line) if line_text and v.line else None
        mark = "!!" if not v.ok else "  "
        tgt = v.target or "?"
        src = f"line {v.line}" + (f" `{t}`" if t else "") if v.line else "(no line)"
        out.append(f"    {mark} {v.label:<22} ours {v.ours:<4} target {tgt:<4} {src}  [{v.insn}]")
    if cs:
        out.append("  causes:")
        cs = cs[:top]
        out += [f"    - {c.text}" for c in cs]
    elif known and right == known:
        out.append("  every value is coloured as the target")
    return out


# ---- directed moves (close --moves oracle) ------------------------------------------------------

LINE_RULES = ("operand-swap", "statement-order", "compare-flip", "reassociate", "op-assign", "chained-assign",
              "assign-in-condition", "cast-call", "split-init", "ternary-if", "inline-helper")


def moves(site, vals: list[Value], per_value: int = 12) -> list[tuple[str, str, str]]:
    """(kind, label, new source) edits at the statements that create the wrong values, most direct
    first: name an operand of the creating statement (or a call argument) as a local, fold a named local
    into its one use / join or split its initialisation, declaration order against the swap partner,
    then the rewrite table's shape rules (commutative operand order, statement order, ...) focused on
    those lines only. Lines outside the function body (an inlined header body) get no edit."""
    from .regexplain import (_fold_single_use, _join_init, _name_operand, _param_reuse, _rule_moves, _split_init,
                             reorder_edits, single_decls, _emit)
    from ...core import cbody
    lo, hi = cbody.line_of(site.src, site.body.start), cbody.line_of(site.src, site.body.end)
    seen: set[str] = {site.src}
    out: list[tuple[str, str, str]] = []
    decls = single_decls(site)
    wrong = [v for v in vals if not v.ok]
    def partner_of(v):
        hits = [w for w in wrong if w is not v and w.cls == v.cls and w.ours == v.target] or \
               [w for w in vals if w is not v and w.cls == v.cls and w.ours == v.target]
        return min(hits, key=lambda w: abs(w.index - v.index)) if hits else None
    order = []
    for v in wrong:                                     # a value, then its partner (who holds its target register)
        for x in (v, partner_of(v)):
            if x is not None and x not in order:
                order.append(x)
    for line in sorted({ln for ln, _o, _t in getattr(vals, "orders", []) if ln}):
        if lo <= line - 1 <= hi:
            for rule in ("operand-swap", "reassociate", "op-assign"):
                for label, new in _rule_moves(site, rule, {line - 1}):
                    if new not in seen:
                        seen.add(new)
                        out.append((rule, f"{label} [operand order]", new))
    for v in order:
        found = 0

        def add(kind, label, edits_or_src):
            nonlocal found
            if found >= per_value or edits_or_src is None:
                return
            if isinstance(edits_or_src, str):
                if edits_or_src in seen:
                    return
                seen.add(edits_or_src)
                new = edits_or_src
            else:
                new = _emit(site, edits_or_src, seen)
                if new is None:
                    return
            out.append((kind, f"{label} [{v.label}]", new))
            found += 1

        if v.name and v.name in decls and v.kind == "local":
            add("fold", f"{v.name} folded into its one use", _fold_single_use(site, v.name))
            add("join", f"{v.name} initialised where declared", _join_init(site, v.name))
            add("split", f"{v.name} declared, then assigned", _split_init(site, v.name))
            partner = partner_of(v) if v.target else None
            if partner is not None and partner.name in decls and partner.kind == "local":
                add("decl-order", f"{v.name} declared before {partner.name}", reorder_edits(site, [v.name, partner.name]))
                add("decl-order", f"{partner.name} declared before {v.name}", reorder_edits(site, [partner.name, v.name]))
            for label, edits in _param_reuse(site, v.name):
                add("param-reuse", label, edits)
        line = (v.line - 1) if v.line else None
        if line is None or not (lo <= line <= hi):
            continue
        for label, edits in _name_operand(site, line):
            add("name-temp", label, edits)
        for label, edits in name_in_condition(site, line):
            add("name-temp", label, edits)
        for rule in LINE_RULES:
            for label, new in _rule_moves(site, rule, {line}):
                add(rule, label, new)
    return out


def name_in_condition(site, line: int):
    """(label, edits) naming one operand of the condition of the `if`/`switch` whose head is on 0-based
    `line`, in a new local declared right before the statement (the condition is evaluated once there,
    so the value and the order of the calls stay the same)."""
    from ...core import cbody
    for st in cbody.statements(site.body):
        if st.kind not in ("if", "switch") or site.line(st.start) != line or not site.in_block(st):
            continue
        tops = list(cbody.statement_exprs(st))
        nodes = [x for t in tops for x in cbody.expressions(t)]
        calls = [x for x in nodes if x.kind == "call"]
        for e in nodes:
            if e.kind not in ("call", "member", "index", "binary", "unary") or any(e is t for t in tops):
                continue
            if e.kind == "binary" and e.op in ("&&", "||", ","):
                continue
            if any(c.start < e.start and not (c.start <= e.start and e.end <= c.end) for c in calls):
                continue
            if any(x.kind == "binary" and x.op in ("&&", "||") and x.start < e.start and e.start >= x.kids[0].end
                   for t in tops for x in cbody.expressions(t)):
                continue                                   # right of && / ||: evaluated only sometimes
            name = site.fresh("t")
            text = site.text(e)
            if site.cplusplus:
                decl = site.declare(st, f"__typeof__({text}) {name} = {text}")
            else:
                decl = site.declare(st, f"__typeof__({text}) {name}")
                if decl is not None:
                    decl += [(st.start, st.start, f"{name} = {text};\n{cbody.indent_at(site.src, st.start)}")]
            site.taken.discard(name)
            if decl is None:
                continue
            from .rewrites import _short
            yield f"named {_short(text)} as {name} before the {st.kind}", decl + [(e.start, e.end, name)]
        return
