"""The near-miss rewrite table: source changes that keep a function's meaning but change how
MWCC lays it out, each keyed to the asm symptom it answers.

`symptoms(target, ours)` reads objdiff's aligned instructions and counts symptom signs (operands
swapped, compare flipped, branch inverted, …) and the source lines they come from. `RULES` is the
table: every row names its symptom, the idiom (as in the playbook's idiom table) and the signs it
answers, and generates candidate sources from the parsed body (`tools/dk/core/cbody.py`), one
edit per candidate, at the lines the mismatches come from. `dk close` compiles them; nothing here
decides whether a candidate is kept.

New locals are typed `__typeof__(expr)` while searching; the closer swaps in the real type from
the debug info before its final check, and the judge rejects any `__typeof__` left.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterator

from ...core import cbody
from ...core.cbody import Expr, Stmt, apply_edits

# ---- symptoms ------------------------------------------------------------------------------------

SIGNS = {
    "swap": "a commutative instruction with its operands the other way round",
    "compare": "a compare with its operands the other way round",
    "branch": "a conditional branch inverted, or an unconditional branch only one side has",
    "landing": "the same branch with another destination: its test leads elsewhere (another block or exit)",
    "mask": "a comparison's bit shifted out plain on one side, masked to a byte (`extrwi`) on the other",
    "signed": "a signed instruction where the other side has the unsigned one (or another extension)",
    "order": "the same instructions in another order",
    "zero": "a register set by `li` where the other side copies one with `mr`",
    "regs": "only register numbers differ",
    "count": "one side has instructions the other lacks",
    "index": "indexed (`x`-form) or multiplied addressing on one side only",
    "cror": "a `cror` on one side only (min/max and float compares)",
    "reloc": "only a relocation's target differs (literal pool order, not a source shape)",
    "stack": "a stack slot at another offset (the order stack locals are declared in)",
}
COMMUTATIVE_INSNS = {"add", "add.", "addc", "adde", "mullw", "mullw.", "mulhw", "mulhwu", "and", "and.", "or", "or.",
                     "xor", "xor.", "nand", "nor", "eqv", "fadd", "fadds", "fmul", "fmuls"}
COMPARES = {"cmpw", "cmplw", "fcmpu", "fcmpo"}
SIGNED_PAIRS = [{"cmpw", "cmplw"}, {"cmpwi", "cmplwi"}, {"extsb", "clrlwi"}, {"extsh", "clrlwi"}, {"mulhw", "mulhwu"},
                {"srawi", "srwi"}, {"divw", "divwu"}, {"lha", "lhz"}, {"extsb.", "clrlwi."}, {"extsh.", "clrlwi."},
                {"cmplwi", "clrlwi."}, {"cmpwi", "extsb."}, {"cmpwi", "extsh."}]
REG = re.compile(r"[rf]\d+")


def _mnemonic(entry: dict) -> str | None:
    for a in entry.get("instruction", {}).get("parts", []):
        if "opcode" in a:
            return a["opcode"].get("mnemonic")
    return None


def _args(entry: dict) -> list[str]:
    out = []
    for a in entry.get("instruction", {}).get("parts", []):
        if "arg" in a:
            v = a["arg"]
            out.append(str(next(iter(v.values()))) if isinstance(v, dict) else str(v))
    return out


def _conditional(m: str | None) -> bool:
    return bool(m) and m.startswith("b") and m not in ("b", "bl", "blr", "bctr", "bctrl")


@dataclass
class Symptoms:
    signs: Counter = field(default_factory=Counter)
    lines: Counter = field(default_factory=Counter)     # our source lines (1-based) of differing rows

    def top(self) -> list[str]:
        return [s for s, _ in self.signs.most_common()]


def symptoms(target: list[dict], ours: list[dict]) -> Symptoms:
    """Signs of how the target differs from ours, from objdiff's aligned instruction lists (both
    sides padded to one length; our rows carry `line_number` when compiled with `-sym on`)."""
    out = Symptoms()
    last_line = None
    moved_t, moved_o = Counter(), Counter()
    rows = list(zip(target, ours))
    for k, (t, o) in enumerate(rows):
        line = o.get("instruction", {}).get("line_number")
        if line:
            last_line = line
        if not (t.get("diff_kind") or o.get("diff_kind")):
            continue
        if last_line:
            out.lines[last_line] += 1
        mt, mo = _mnemonic(t), _mnemonic(o)
        if mt is None or mo is None:
            out.signs["count"] += 1
            present = mt or mo
            near = [_mnemonic(x) for a, b in rows[max(0, k - 2):k + 3] for x in (a, b)]
            if present == "b" and any(_conditional(m) for m in near):
                out.signs["branch"] += 1
            if present == "cror":
                out.signs["cror"] += 1
            if present in ("mr", "fmr"):
                out.signs["regs"] += 1
            (moved_t if mt else moved_o)[present] += 1
            continue
        at, ao = _args(t), _args(o)
        if mt == mo:
            if at == ao:
                continue
            rt, ro = [a for a in at if not REG.fullmatch(a)], [a for a in ao if not REG.fullmatch(a)]
            if _conditional(mt) or mt == "b":
                out.signs["landing"] += 1                  # only the destination moved: lands elsewhere
                continue
            if rt != ro:
                reloc = any("@" in a or "lbl_" in a for a in rt + ro)
                out.signs["reloc" if reloc else "stack" if "r1" in at and "r1" in ao else "count"] += 1
                continue
            if mt in COMMUTATIVE_INSNS and len(at) == 3 and at[0] == ao[0] and at[1:] == ao[2:0:-1]:
                out.signs["swap"] += 1
            elif mt in COMPARES and at[-2:] == ao[-1:-3:-1]:
                out.signs["compare"] += 1
            else:
                out.signs["regs"] += 1
            continue
        pair = {mt, mo}
        if _conditional(mt) and _conditional(mo):
            out.signs["branch"] += 1
        elif pair in SIGNED_PAIRS:
            out.signs["signed"] += 1
        elif pair == {"li", "mr"} or pair == {"li", "addi"}:
            out.signs["zero"] += 1
        elif pair == {"srwi", "extrwi"}:
            out.signs["mask"] += 1
        elif any(m.endswith("x") and m[:-1] in (mt, mo) for m in (mt, mo)) or "mulli" in pair:
            out.signs["index"] += 1
        elif "cror" in pair:
            out.signs["cror"] += 1
        else:
            moved_t[mt] += 1
            moved_o[mo] += 1
    if moved_t or moved_o:
        common = sum((moved_t & moved_o).values())
        if common:
            out.signs["order"] += common
        rest = sum((moved_t - moved_o).values()) + sum((moved_o - moved_t).values())
        if rest:
            out.signs["count"] += rest
    return out


# ---- the rewrite table -----------------------------------------------------------------------------

@dataclass
class Site:
    """What a rule sees: the whole source, the parsed body, the lines to focus on (0-based; None
    for everywhere) and whether the file is C++ (C gets declarations at the top of a block)."""
    src: str
    body: Stmt
    focus: set[int] | None = None
    cplusplus: bool = True
    taken: set[str] = field(default_factory=set)         # identifiers already used in the body
    span: object = None                                   # the function's cbody.FunctionSpan (found when None)
    members: frozenset | None = None                      # data members of `this`'s class (debug info)
    prefer: frozenset = frozenset()                       # locals the rank solver flags (tried first)
    accessors: dict | None = None                         # {array member: [(accessor, "ptr"|"ref")]} (shaperules)

    def __post_init__(self):
        self.parents = cbody.parents(self.body)
        if not self.taken:
            self.taken = {t.text for t in cbody.tokens(self.src, self.body.start, self.body.end) if t.kind == "id"}
        self._starts = cbody.line_starts(self.src)

    def line(self, offset: int) -> int:
        import bisect
        return bisect.bisect_right(self._starts, offset) - 1

    def near(self, node) -> bool:
        if self.focus is None:
            return True
        a, b = self.line(node.start), self.line(max(node.start, node.end - 1))
        return any(a <= ln <= b for ln in self.focus)

    def text(self, node) -> str:
        return self.src[node.start:node.end]

    def fresh(self, stem: str) -> str:
        """A new local name; each call gives another one, so the edits of several sites combine."""
        k = 0
        while f"{stem}{k}" in self.taken:
            k += 1
        self.taken.add(f"{stem}{k}")
        return f"{stem}{k}"

    def in_block(self, s: Stmt) -> bool:
        p = self.parents.get(id(s))
        if p is None or p[0].kind != "block":
            return False
        prev = p[0].kids[p[1] - 1] if p[1] > 0 else None
        return prev is None or prev.kind not in ("case", "default", "label")

    def stmt_of(self, e: Expr) -> Stmt | None:
        best = None
        for s in cbody.statements(self.body):
            if s.kind != "block" and s.start <= e.start and e.end <= s.end and any(
                    x is e for y in cbody.statement_exprs(s) for x in cbody.expressions(y)):
                best = s
        return best

    def declare(self, before: Stmt, decl: str) -> list[tuple[int, int, str]] | None:
        """Edits that add a declaration `decl` (no `;`) so it is in scope at `before`: right before it
        in C++, at the top of its block in C (then `decl` must have no initializer)."""
        if not self.in_block(before):
            return None
        ind = cbody.indent_at(self.src, before.start)
        if self.cplusplus:
            return [(before.start, before.start, f"{decl};\n{ind}")]
        block = self.parents[id(before)][0]
        first = next((k for k in block.kids if k.kind != "decl"), None)
        at = first.start if first is not None else before.start
        return [(at, at, f"{decl};\n{cbody.indent_at(self.src, at)}")]


Gen = Callable[[Site], Iterator[tuple[str, list[tuple[int, int, str]]]]]


@dataclass(frozen=True)
class Rule:
    name: str
    symptom: str        # how the miss shows in the asm
    idiom: str          # the source change
    signs: tuple[str, ...]
    gen: Gen


SHAPE = re.compile(r"(?:0x[0-9A-Fa-f]+|\d+\.?\d*)[fFuUlL]*")


def shape(label: str) -> str:
    """A label with its numbers blanked: the same rewrite at sites that differ only in constants
    (an unrolled sequence, one statement per element)."""
    return SHAPE.sub("#", label)


NEW_NAME = re.compile(r"\b([te])(\d+)\b")


def _renumber(site: Site, taken: set[str], new: str, label: str) -> tuple[str, str]:
    """Number one candidate's new locals from 0 (`t0`, `t1`, `e0`) in order of appearance."""
    lo, hi = site.body.start, site.body.end + len(new) - len(site.src)
    fresh = []
    for m in NEW_NAME.finditer(new, lo, hi):
        if m.group(0) not in taken and m.group(0) not in fresh:
            fresh.append(m.group(0))
    if not fresh:
        return new, label
    mapping, used = {}, set(taken)
    for name in fresh:
        k = 0
        while f"{name[0]}{k}" in used:
            k += 1
        mapping[name] = f"{name[0]}{k}"
        used.add(mapping[name])
    if all(a == b for a, b in mapping.items()):
        return new, label
    swap = lambda m: mapping.get(m.group(0), m.group(0))
    return new[:lo] + NEW_NAME.sub(swap, new[lo:hi]) + new[hi:], NEW_NAME.sub(swap, label)


def candidates(site: Site, rules: list[Rule] | None = None, groups: bool = True) -> Iterator[tuple[str, str, str]]:
    """(rule name, label, new source) for every rule, de-duplicated, in table order. A rewrite that
    applies at several sites of the same shape comes first as one candidate for all of them."""
    seen = {site.src}
    for rule in (RULES if rules is None else rules):
        taken = set(site.taken)               # names only need to be unique within one candidate
        singles = list(rule.gen(site))
        focused = {shape(label) for label, _ in singles}
        focus, site.focus, site.taken = site.focus, None, set(taken)
        try:
            wide = list(rule.gen(site)) if focus is not None else singles
        finally:
            site.focus, site.taken = focus, taken
        shapes: dict[str, list[tuple[str, list]]] = {}
        for label, edits in wide:
            shapes.setdefault(shape(label), []).append((label, edits))
        combined = []
        for key, members in shapes.items():
            if not groups or len(members) < 2 or key not in focused:
                continue
            edits = [e for _, es in members for e in es]
            combined.append((f"{members[0][0]} (and {len(members) - 1} more like it)", edits))
        for label, edits in combined + singles:
            try:
                new = apply_edits(site.src, edits)
            except ValueError:
                continue
            new, label = _renumber(site, taken, new, label)
            if new not in seen:
                seen.add(new)
                yield rule.name, label, new


# ---- helpers ---------------------------------------------------------------------------------------

COMM_OPS = {"+", "*", "&", "|", "^", "==", "!="}
MIRROR = {"<": ">", ">": "<", "<=": ">=", ">=": "<=", "==": "==", "!=": "!="}
NEGATE = {"<": ">=", ">=": "<", ">": "<=", "<=": ">", "==": "!=", "!=": "=="}
JUMPS = ("return", "break", "continue", "goto")


def plain(e: Expr) -> bool:
    """A variable or a load (name, member, element, literal, call): source order sets where it goes."""
    return e.kind in ("name", "num", "member", "index", "call", "str", "chr")


def pure(e: Expr | None) -> bool:
    """No assignment, increment or call inside."""
    return e is not None and not any(x.kind in ("assign", "postfix", "call") or (x.kind == "unary" and x.op in ("++", "--", "new", "delete"))
                                     for x in cbody.expressions(e))


def needs_parens(e: Expr) -> bool:
    return e.kind in ("binary", "ternary", "assign", "comma")


def wrap(site: Site, e: Expr) -> str:
    t = site.text(e)
    return f"({t})" if needs_parens(e) else t


def exprs_of(site: Site, near: bool = True) -> Iterator[Expr]:
    for s in cbody.statements(site.body):
        for top in cbody.statement_exprs(s):
            for e in cbody.expressions(top):
                if not near or site.near(e):
                    yield e


def stmts_of(site: Site, *kinds: str, near: bool = True) -> Iterator[Stmt]:
    for s in cbody.statements(site.body):
        if (not kinds or s.kind in kinds) and (not near or site.near(s)):
            yield s


def unbraced(s: Stmt) -> Stmt | None:
    """The one statement a block holds, or the statement itself."""
    if s.kind == "block":
        return s.kids[0] if len(s.kids) == 1 else None
    return s


def single_assign(s: Stmt) -> Expr | None:
    one = unbraced(s)
    return one.expr if one is not None and one.kind == "expr" and one.expr.kind == "assign" and one.expr.op == "=" else None


def braced(site: Site, at: Stmt, lines: list[str]) -> str:
    """A `{ … }` block of statements for a statement written at `at`'s indentation."""
    ind = cbody.indent_at(site.src, at.start)
    inner = ind + "    "
    return "{\n" + "".join(f"{inner}{ln}\n" for ln in lines) + ind + "}"


def body_lines(site: Site, s: Stmt) -> list[str]:
    """A statement's text as lines for `braced` (a block contributes its statements)."""
    kids = s.kids if s.kind == "block" else [s]
    return [" ".join(site.text(k).split()) if "\n" not in site.text(k) else site.text(k).strip() for k in kids]


def binds_break(s: Stmt) -> bool:
    """Does a `break` inside s leave s's enclosing loop or switch (so wrapping s in a switch would
    change it)?"""
    def walk(x: Stmt) -> bool:
        if x.kind == "break":
            return True
        if x.kind in ("for", "while", "do", "switch"):
            return False
        return any(walk(k) for k in x.kids)
    return walk(s)


def negate(site: Site, e: Expr) -> str:
    if e.kind == "binary" and e.op in NEGATE:
        a, b = e.kids
        return f"{site.text(a)} {NEGATE[e.op]} {site.text(b)}"
    if e.kind == "unary" and e.op == "!":
        inner = e.kids[0]
        return site.text(inner.kids[0]) if inner.kind == "paren" else site.text(inner)
    if plain(e) or e.kind == "paren":
        return "!" + site.text(e)
    return f"!({site.text(e)})"


def swapped(site: Site, e: Expr, op: str | None = None) -> str:
    """`b op a` for `a op b`. The old left operand goes right, under the operator's left associativity:
    one of the same precedence (`x / n * 12` -> `12 * (x / n)`) needs parentheses there."""
    a, b = e.kids
    mid = site.src[a.end:b.start]
    if op is not None:
        mid = mid.replace(e.op, op, 1)
    right = site.text(a)
    if a.kind in ("binary", "ternary", "assign", "comma") and cbody.BINARY.get(a.op, 0) <= cbody.BINARY.get(op or e.op, 0):
        right = f"({right})"
    return site.text(b) + mid + right


def _no_swap(site: Site, e: Expr) -> bool:
    """Operand swaps that are not the same code: a string literal's `+` (a string class's concatenation:
    order is meaning), and integer rounding `x / N * N` (truncation)."""
    def leaves(x: Expr) -> Iterator[Expr]:
        if x.kind == "binary" and x.op == "+":
            for k in x.kids:
                yield from leaves(k)
        else:
            yield x
    if e.op == "+" and any(x.kind == "str" for x in leaves(e)):
        return True
    return e.op == "*" and any(k.kind == "binary" and k.op in ("/", "%") for k in e.kids)


# ---- the rules -------------------------------------------------------------------------------------

def r_operand_swap(site: Site):
    """Between plain operands source order sets the operand order; a compound operand named in a
    local is plain again."""
    for e in exprs_of(site):
        if e.kind == "unary" and e.op == "&" and e.kids[0].kind == "index":
            base, sub = e.kids[0].kids                  # `&a[i]` is `a + i`: its add takes the base first
            for new in (f"{wrap(site, sub)} + {wrap(site, base)}", f"{wrap(site, base)} + {wrap(site, sub)}"):
                yield f"{site.text(e)} -> {new}", [(e.start, e.end, f"({new})" if s_needs_parens(site, e) else new)]
            continue
        if e.kind != "binary" or e.op not in COMM_OPS or _no_swap(site, e):
            continue
        a, b = e.kids
        yield f"{site.text(e)} -> {swapped(site, e)}", [(e.start, e.end, swapped(site, e))]
        if plain(a) and plain(b):
            continue
        s = site.stmt_of(e)
        if s is None or s.kind in ("for", "while", "do") or _guarded(site, s, e):
            continue
        for part, other in ((a, b), (b, a)):
            if plain(part) or not pure(part) or not pure(other):
                continue
            name = site.fresh("t")
            part = part.kids[0] if part.kind == "paren" else part
            types = [f"__typeof__({site.text(part)})"]
            if part.kind == "cast" and SMALL_INT.match(site.src[part.type_span[0]:part.type_span[1]].strip()):
                types.append("int")                     # a converted small integer kept as an int
            for typ in types:
                decl = site.declare(s, f"{typ} {name}" + (f" = {site.text(part)}" if site.cplusplus else ""))
                if decl is None:
                    continue
                if not site.cplusplus:
                    decl += [(s.start, s.start, f"{name} = {site.text(part)};\n{cbody.indent_at(site.src, s.start)}")]
                for first, second in ((name, site.text(other)), (site.text(other), name)):
                    yield (f"named {site.text(part)} as {'int ' if typ == 'int' else ''}{name}: {first} {e.op} {second}",
                           decl + [(e.start, e.end, f"{first} {e.op} {second}")])


SMALL_INT = re.compile(r"^(?:(?:un)?signed\s+)?(?:char|short)(?:\s+int)?$|^(?:u8|s8|u16|s16|bool)$")


def s_needs_parens(site: Site, e: Expr) -> bool:
    """Does an expression replacing e need parentheses (it sits under a tighter operator)?"""
    for x in exprs_of(site, near=False):
        if any(k is e for k in x.kids) and x.kind in ("binary", "unary", "cast", "member", "index", "call", "postfix"):
            return not (x.kind == "call" and x.kids[0] is not e)
    return False


def _guarded(site: Site, s: Stmt, e: Expr) -> bool:
    """Is e evaluated only sometimes within its statement (right of && / ||, inside ?:)?"""
    for top in cbody.statement_exprs(s):
        for x in cbody.expressions(top):
            if x.kind == "binary" and x.op in ("&&", "||") and x.kids[1].start <= e.start and e.end <= x.kids[1].end:
                return True
            if x.kind == "ternary" and x.kids[1].start <= e.start and e.end <= x.end:
                return True
    return False


def r_compare_flip(site: Site):
    """A compare's operand order follows the source: `a < b` vs `b > a`; `if (x)` vs `x != 0`."""
    tested = set()
    for s in stmts_of(site, "if", "while", "do", "for"):
        cond = s.cond if s.kind == "for" else s.expr
        if cond is not None:
            tested.add(id(cond))
            for x in cbody.expressions(cond):
                if x.kind == "binary" and x.op in ("&&", "||"):
                    tested.update(id(k) for k in x.kids)
                if x.kind == "unary" and x.op == "!":
                    tested.add(id(x.kids[0]))
    for e in exprs_of(site):
        if e.kind == "binary" and e.op in MIRROR:
            new = swapped(site, e, MIRROR[e.op])
            yield f"{site.text(e)} -> {new}", [(e.start, e.end, new)]
            a, b = e.kids
            if e.op in ("!=", "==") and site.text(b) in ("0", "0.0f", "0.0", "NULL") and id(e) in tested:
                new = site.text(a) if e.op == "!=" else "!" + wrap(site, a)
                yield f"{site.text(e)} -> {new}", [(e.start, e.end, new)]
        elif id(e) in tested and (plain(e) or e.kind == "paren") and e.kind not in ("num", "str", "chr"):
            yield f"{site.text(e)} -> {site.text(e)} != 0", [(e.start, e.end, f"{site.text(e)} != 0")]
        elif e.kind == "unary" and e.op == "!" and id(e) in tested and plain(e.kids[0]):
            yield f"{site.text(e)} -> {site.text(e.kids[0])} == 0", [(e.start, e.end, f"{site.text(e.kids[0])} == 0")]


def r_reassociate(site: Site):
    """`a + b + K` vs `a + (b + K)`: association sets which add comes first."""
    for e in exprs_of(site):
        if e.kind != "binary" or e.op not in ("+", "*", "&", "|", "^"):
            continue
        a, b = e.kids
        if a.kind == "binary" and a.op == e.op:
            x, y = a.kids
            new = f"{site.text(x)} {e.op} ({site.text(y)} {e.op} {site.text(b)})"
            yield f"{site.text(e)} -> {new}", [(e.start, e.end, new)]
            new = f"{site.text(x)} {e.op} {site.text(b)} {e.op} {site.text(y)}"
            yield f"{site.text(e)} -> {new}", [(e.start, e.end, new)]
        if b.kind == "paren" and b.kids[0].kind == "binary" and b.kids[0].op == e.op:
            y, z = b.kids[0].kids
            new = f"{site.text(a)} {e.op} {site.text(y)} {e.op} {site.text(z)}"
            yield f"{site.text(e)} -> {new}", [(e.start, e.end, new)]


HALVES = {"0.5": "2.0", "0.25": "4.0", "0.125": "8.0", "2.0": "0.5", "4.0": "0.25", "8.0": "0.125", "2": "0.5"}


def _float_literal(t: str) -> tuple[str, str] | None:
    m = re.fullmatch(r"(\d+\.?\d*)([fF]?)", t)
    if not m or m.group(1) not in HALVES or (m.group(1) == "2" and not m.group(2)):
        return None
    return m.group(1), m.group(2)


def r_half(site: Site):
    """`x / 2.0f` loads x before the constant; a literal `x * 0.5f` loads the constant first."""
    for e in exprs_of(site):
        if e.kind == "binary" and e.op in ("*", "/"):
            a, b = e.kids
            for lit, other in ((b, a), (a, b)) if e.op == "*" else ((b, a),):
                f = _float_literal(site.text(lit)) if lit.kind == "num" else None
                if f is None or (e.op == "*" and float(f[0]) >= 1) or (e.op == "/" and float(f[0]) < 1):
                    continue
                inv = HALVES[f[0]] + (f[1] or "f")
                if e.op == "*":
                    new = f"{wrap(site, other)} / {inv}"
                    yield f"{site.text(e)} -> {new}", [(e.start, e.end, new)]
                else:
                    for new in (f"{wrap(site, other)} * {inv}", f"{inv} * {wrap(site, other)}"):
                        yield f"{site.text(e)} -> {new}", [(e.start, e.end, new)]
        elif e.kind == "assign" and e.op in ("*=", "/=") and e.kids[1].kind == "num":
            f = _float_literal(site.text(e.kids[1]))
            if f is None or (e.op == "*=") != (float(f[0]) < 1):
                continue
            new = f"{site.text(e.kids[0])} {'/=' if e.op == '*=' else '*='} {HALVES[f[0]]}{f[1] or 'f'}"
            yield f"{site.text(e)} -> {new}", [(e.start, e.end, new)]


BOOL_ARMS = {("true", "false"), ("false", "true"), ("1", "0"), ("0", "1")}


def r_ternary_if(site: Site):
    """`x = c ? a : b` makes branches; `if (c) x = a; else x = b;` may give `cror` (and the reverse).
    Never written: a branch that assigns x to itself, or `c ? true : false` (nobody writes either)."""
    for s in stmts_of(site, "expr", "return", "decl"):
        e = s.expr if s.kind != "decl" else (s.decl.declarators[0].init if len(s.decl.declarators) == 1 else None)
        if s.kind == "expr" and e.kind == "assign" and e.op == "=" and e.kids[1].kind == "ternary":
            c, a, b = e.kids[1].kids
            lhs = site.text(e.kids[0])
            if lhs in (site.text(a).strip(), site.text(b).strip()):
                continue
            new = (f"if ({site.text(c)}) " + braced(site, s, [f"{lhs} = {site.text(a)};"]) + " else "
                   + braced(site, s, [f"{lhs} = {site.text(b)};"]))
            yield f"{site.text(s)} -> if/else", [(s.start, s.end, new)]
        elif s.kind == "return" and e is not None and e.kind == "ternary" and site.in_block(s):
            c, a, b = e.kids
            ind = cbody.indent_at(site.src, s.start)
            new = f"if ({site.text(c)}) " + braced(site, s, [f"return {site.text(a)};"]) + f"\n{ind}return {site.text(b)};"
            yield f"{site.text(s)} -> if/return", [(s.start, s.end, new)]
        elif s.kind == "decl" and e is not None and e.kind == "ternary" and s.decl.declarators[0].init_kind == "=" \
                and "&" not in s.decl.declarators[0].ptr and site.in_block(s):
            d = s.decl.declarators[0]
            c, a, b = e.kids
            ind = cbody.indent_at(site.src, s.start)
            head = site.src[s.start:d.name_end] + ";"
            new = (f"{head}\n{ind}if ({site.text(c)}) " + braced(site, s, [f"{d.name} = {site.text(a)};"]) + " else "
                   + braced(site, s, [f"{d.name} = {site.text(b)};"]))
            yield f"{site.text(s)[:40]} -> declared, then if/else", [(s.start, s.end, new)]
    for s in stmts_of(site, "if"):
        if len(s.kids) == 2:
            x, y = single_assign(s.kids[0]), single_assign(s.kids[1])
            if x is not None and y is not None and site.text(x.kids[0]) == site.text(y.kids[0]) \
                    and (site.text(x.kids[1]), site.text(y.kids[1])) not in BOOL_ARMS:
                new = f"{site.text(x.kids[0])} = {site.text(s.expr)} ? {site.text(x.kids[1])} : {site.text(y.kids[1])};"
                yield f"if/else -> {new}", [(s.start, s.end, new)]
            ra, rb = unbraced(s.kids[0]), unbraced(s.kids[1])
            if ra is not None and rb is not None and ra.kind == rb.kind == "return" and ra.expr and rb.expr \
                    and (site.text(ra.expr), site.text(rb.expr)) not in BOOL_ARMS:
                new = f"return {site.text(s.expr)} ? {site.text(ra.expr)} : {site.text(rb.expr)};"
                yield f"if/else return -> {new}", [(s.start, s.end, new)]
        elif len(s.kids) == 1 and site.in_block(s):
            block, i = site.parents[id(s)]
            ra = unbraced(s.kids[0])
            nxt = block.kids[i + 1] if i + 1 < len(block.kids) else None
            if ra is not None and ra.kind == "return" and ra.expr and nxt is not None and nxt.kind == "return" and nxt.expr \
                    and (site.text(ra.expr), site.text(nxt.expr)) not in BOOL_ARMS:
                new = f"return {site.text(s.expr)} ? {site.text(ra.expr)} : {site.text(nxt.expr)};"
                yield f"if return; return -> {new}", [(s.start, nxt.end, new)]


def r_one_case_switch(site: Site):
    """`cmpwi; beq case; b default` where ours gives `bne`: a `switch` with one `case`, not an `if`."""
    for s in stmts_of(site, "if"):
        c = s.expr
        if c is None or c.kind != "binary" or c.op != "==" or binds_break(s.kids[0]):
            continue
        a, b = c.kids
        if b.kind not in ("num", "name", "chr") and a.kind in ("num", "chr"):
            a, b = b, a
        if b.kind not in ("num", "name", "chr", "member", "unary"):
            continue
        ind = cbody.indent_at(site.src, s.start)
        then = body_lines(site, s.kids[0]) + ([] if _ends_in_jump(s.kids[0]) else ["break;"])
        cases = [f"case {site.text(b)}:"] + ["    " + ln for ln in then]
        if len(s.kids) == 2:
            if binds_break(s.kids[1]):
                continue
            cases += ["default:"] + ["    " + ln for ln in body_lines(site, s.kids[1])] + \
                     ([] if _ends_in_jump(s.kids[1]) else ["    break;"])
        new = f"switch ({site.text(a)}) " + "{\n" + "".join(f"{ind}{ln}\n" for ln in cases) + ind + "}"
        yield f"if ({site.text(c)}) -> switch", [(s.start, s.end, new)]
    for s in stmts_of(site, "switch"):
        body = s.kids[0]
        if body.kind != "block":
            continue
        labels = [k for k in body.kids if k.kind in ("case", "default")]
        if len(labels) != 1 or labels[0].kind != "case" or body.kids[0] is not labels[0]:
            continue
        rest = body.kids[1:]
        if rest and rest[-1].kind == "break":
            rest = rest[:-1]
        if any(binds_break(k) for k in rest):
            continue
        lines = [site.text(k).strip() for k in rest]
        new = f"if ({site.text(s.expr)} == {site.text(labels[0].expr)}) " + braced(site, s, lines)
        yield "one-case switch -> if", [(s.start, s.end, new)]


def _ends_in_jump(s: Stmt) -> bool:
    last = s.kids[-1] if s.kind == "block" and s.kids else s
    return last.kind in JUMPS


def r_or_return(site: Site):
    """`if (!a || !b) return;` gives `bne next; b exit`; two separate `if`s do not (and the reverse)."""
    for block in stmts_of(site, "block", near=False):
        for i in range(len(block.kids) - 1):
            x, y = block.kids[i], block.kids[i + 1]
            if x.kind == y.kind == "if" and len(x.kids) == len(y.kids) == 1 and (site.near(x) or site.near(y)):
                jx, jy = unbraced(x.kids[0]), unbraced(y.kids[0])
                if jx is not None and jy is not None and jx.kind in JUMPS and \
                        " ".join(site.text(jx).split()) == " ".join(site.text(jy).split()):
                    cond = f"{_or_operand(site, x.expr)} || {_or_operand(site, y.expr)}"
                    new = f"if ({cond}) " + site.text(x.kids[0])
                    yield f"two ifs -> if ({cond})", [(x.start, y.end, new)]
    for s in stmts_of(site, "if"):
        j = unbraced(s.kids[0])
        if len(s.kids) == 1 and s.expr.kind == "binary" and s.expr.op == "||" and j is not None and j.kind in JUMPS \
                and site.in_block(s):
            a, b = s.expr.kids
            ind = cbody.indent_at(site.src, s.start)
            body = site.text(s.kids[0])
            new = f"if ({site.text(a)}) {body}\n{ind}if ({site.text(b)}) {body}"
            yield f"if (a || b) -> two ifs", [(s.start, s.end, new)]


def _or_operand(site: Site, e: Expr) -> str:
    return f"({site.text(e)})" if e.kind in ("ternary", "assign", "comma") or (e.kind == "binary" and e.op == "&&") \
        else site.text(e)


def r_chained_assign(site: Site):
    """Stores in reverse order: `a = b = c = K` stores the last one first (and the reverse)."""
    for block in stmts_of(site, "block", near=False):
        kids = block.kids
        i = 0
        while i < len(kids):
            run = [kids[i]]
            e0 = kids[i].expr if kids[i].kind == "expr" else None
            if e0 is not None and e0.kind == "assign" and e0.op == "=" and pure(e0.kids[1]) and e0.kids[1].kind != "assign":
                for k in kids[i + 1:]:
                    e = k.expr if k.kind == "expr" else None
                    if e is None or e.kind != "assign" or e.op != "=" or site.text(e.kids[1]) != site.text(e0.kids[1]):
                        break
                    run.append(k)
            if len(run) > 1 and any(site.near(k) for k in run):
                lhs = [site.text(k.expr.kids[0]) for k in run]
                rhs = site.text(e0.kids[1])
                for order in (lhs, lhs[::-1]):
                    new = " = ".join(order) + f" = {rhs};"
                    yield f"{len(run)} assignments -> {new}", [(run[0].start, run[-1].end, new)]
            i += len(run)
    for s in stmts_of(site, "expr"):
        e = s.expr
        if e.kind == "assign" and e.op == "=" and e.kids[1].kind == "assign" and e.kids[1].op == "=" and site.in_block(s):
            chain, x = [e.kids[0]], e.kids[1]
            while x.kind == "assign" and x.op == "=":
                chain.append(x.kids[0])
                x = x.kids[1]
            if not pure(x):
                continue
            ind = cbody.indent_at(site.src, s.start)
            for order in (chain, chain[::-1]):
                new = f"\n{ind}".join(f"{site.text(t)} = {site.text(x)};" for t in order)
                yield f"chained -> {len(chain)} assignments", [(s.start, s.end, new)]


def r_op_assign(site: Site):
    """`x *= K` puts the variable first; `x = x * K` / `x = K * x` follow the written order."""
    for e in exprs_of(site):
        if e.kind == "assign" and e.op == "=" and e.kids[1].kind == "binary" and e.kids[1].op in ("+", "-", "*", "/", "|", "&", "^", "<<", ">>"):
            lhs, (a, b) = e.kids[0], e.kids[1].kids
            op = e.kids[1].op
            if site.text(a) == site.text(lhs):
                new = f"{site.text(lhs)} {op}= {site.text(b)}"
                yield f"{site.text(e)} -> {new}", [(e.start, e.end, new)]
            elif op in ("+", "*", "|", "&", "^") and site.text(b) == site.text(lhs):
                new = f"{site.text(lhs)} {op}= {site.text(a)}"
                yield f"{site.text(e)} -> {new}", [(e.start, e.end, new)]
        elif e.kind == "assign" and e.op not in ("=", "<<=", ">>=") and pure(e.kids[0]):
            lhs, rhs, op = site.text(e.kids[0]), wrap(site, e.kids[1]), e.op[:-1]
            outs = [f"{lhs} = {lhs} {op} {rhs}"] + ([f"{lhs} = {rhs} {op} {lhs}"] if op in ("+", "*", "|", "&", "^") else [])
            for new in outs:
                yield f"{site.text(e)} -> {new}", [(e.start, e.end, new)]


INT_TYPES = re.compile(r"^(?:signed\s+)?(?:int|long|short|s32|s16)$|^(?:unsigned(?:\s+(?:int|long|short))?|u32|u16|uint)$")


def r_unsigned_index(site: Site):
    """`addi r0, rOff, K; lwzx` vs a stepped pointer: a signed counter converted to unsigned indexes
    through an offset (`a[(unsigned int)i]`); a plain `a[i]` steps a pointer."""
    for s in stmts_of(site, "for", near=False):
        if s.init is None or s.init.kind != "decl" or len(s.init.decl.declarators) != 1:
            continue
        typ = s.init.decl.base_type(site.src)
        if not INT_TYPES.match(typ) or not site.near(s):
            continue
        i = s.init.decl.declarators[0].name
        body = s.kids[0]
        subs = [x for top in _all_exprs(body) for x in cbody.expressions(top)
                if x.kind == "index" and x.kids[1].kind == "name" and x.kids[1].name == i]
        if subs and not typ.startswith(("unsigned", "u")):
            yield f"[{i}] -> [(unsigned int){i}]", [(x.kids[1].start, x.kids[1].end, f"(unsigned int){i}") for x in subs]
        casts = [x for top in _all_exprs(body) for x in cbody.expressions(top) if x.kind == "cast"
                 and site.src[x.type_span[0]:x.type_span[1]].split()[0] in ("unsigned", "u32") and x.kids[0].kind == "name"
                 and x.kids[0].name == i]
        if casts:
            yield f"casts of {i} dropped", [(x.start, x.end, i) for x in casts]
        ts, te = s.init.decl.type_start, s.init.decl.type_end
        other = "int" if typ.startswith(("unsigned", "u")) else "unsigned int"
        yield f"counter {other} {i}", [(ts, te, other)]


def _all_exprs(s: Stmt) -> Iterator[Expr]:
    for x in cbody.statements(s):
        yield from cbody.statement_exprs(x)


def r_ref_element(site: Site):
    """`mulli; addi; lfsx` on an array element: the element was read through `T& e = a[i];` (and the
    reverse: a reference local written out)."""
    for block in stmts_of(site, "block", near=False):
        groups: dict[str, list[int]] = {}
        for k, s in enumerate(block.kids):
            for top in _all_exprs(s):
                for x in cbody.expressions(top):
                    if x.kind == "member" and x.op == "." and x.kids[0].kind == "index" and pure(x.kids[0]):
                        groups.setdefault(site.text(x.kids[0]), []).append(k)
        for elem, ks in groups.items():
            k0 = ks[0]
            first = block.kids[k0]
            if not site.near(first) or not site.in_block(first):
                continue
            written = set(cbody.names(cbody.parse_expression(elem)))
            end = k0
            while end + 1 < len(block.kids) and not (_writes(site, block.kids[end + 1]) & written):
                end += 1
            if _writes(site, first) & written:
                continue
            name = site.fresh("e")
            edits = site.declare(first, f"__typeof__({elem})& {name} = {elem}")
            if edits is None or not site.cplusplus:
                continue
            for s in block.kids[k0:end + 1]:
                for top in _all_exprs(s):
                    for x in cbody.expressions(top):
                        if x.kind == "index" and site.text(x) == elem:
                            edits.append((x.start, x.end, name))
            edits = _outermost(edits)
            yield f"{elem} read through {name}", edits
    for s in stmts_of(site, "decl"):
        if len(s.decl.declarators) != 1 or not site.in_block(s):
            continue
        d = s.decl.declarators[0]
        if d.ptr != "&" or d.init is None or not pure(d.init):
            continue
        block, k = site.parents[id(s)]
        written = set(cbody.names(d.init))
        edits = [(s.start, block.kids[k + 1].start if k + 1 < len(block.kids) else s.end, "")]
        ok = True
        for t in block.kids[k + 1:]:
            if _writes(site, t) & written:
                ok = False
            for top in _all_exprs(t):
                for x in cbody.expressions(top):
                    if x.kind == "name" and x.name == d.name:
                        edits.append((x.start, x.end, wrap(site, d.init)))
        if ok:
            yield f"reference {d.name} written out", edits


def _writes(site: Site, s: Stmt) -> set[str]:
    out = set()
    for top in _all_exprs(s):
        for x in cbody.expressions(top):
            if x.kind == "assign" or x.kind == "postfix" or (x.kind == "unary" and x.op in ("++", "--")):
                out.update(cbody.names(x.kids[0]) if x.kids[0].kind == "name" else [])
    return out


def _outermost(edits: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    for e in sorted(edits, key=lambda x: (x[0], -x[1])):
        if out and e[0] < out[-1][1] and e[0] != e[1]:
            continue
        out.append(e)
    return out


TRUE, FALSE = ("true", "1"), ("false", "0")


def r_flag_form(site: Site):
    """A `bool` flag that starts in the block's shared zero register: `bool b = a || c;`, not
    `bool b = false; if (a || c) b = true;` (and the reverse)."""
    for block in stmts_of(site, "block", near=False):
        for k in range(len(block.kids) - 1):
            d, s = block.kids[k], block.kids[k + 1]
            if d.kind != "decl" or len(d.decl.declarators) != 1 or s.kind != "if" or len(s.kids) != 1:
                continue
            x = d.decl.declarators[0]
            a = single_assign(s.kids[0])
            if x.init is None or site.text(x.init) not in FALSE or a is None or site.text(a.kids[0]) != x.name \
                    or site.text(a.kids[1]) not in TRUE or not (site.near(d) or site.near(s)):
                continue
            c = s.expr
            boolean = c.kind == "binary" and c.op in ("||", "&&", "<", ">", "<=", ">=", "==", "!=") or \
                (c.kind == "unary" and c.op == "!") or d.decl.base_type(site.src) == "bool"
            val = site.text(c) if boolean else f"{wrap(site, c)} != 0"
            yield f"flag {x.name} = {val}", [(x.init.start, x.init.end, val), (d.end, s.end, "")]
    for s in stmts_of(site, "decl"):
        if len(s.decl.declarators) != 1 or not site.in_block(s):
            continue
        x = s.decl.declarators[0]
        c = x.init
        if c is None or x.init_kind != "=" or not (c.kind == "binary" and c.op in ("||", "&&", "<", ">", "<=", ">=", "==", "!=")):
            continue
        t, f = ("true", "false") if s.decl.base_type(site.src) == "bool" else ("1", "0")
        ind = cbody.indent_at(site.src, s.start)
        new_if = f"\n{ind}if ({site.text(c)}) " + braced(site, s, [f"{x.name} = {t};"])
        yield f"flag {x.name} set by if", [(c.start, c.end, f), (s.end, s.end, new_if)]


def r_assign_in_condition(site: Site):
    """A value loaded and copied (`lwz r3, g; cmplwi r3, 0; mr r31, r3`) where the target loads it
    straight into its register: `if ((p = g) != 0)` (and the reverse)."""
    for block in stmts_of(site, "block", near=False):
        for k in range(len(block.kids) - 1):
            d, s = block.kids[k], block.kids[k + 1]
            if s.kind != "if" or not (site.near(d) or site.near(s)):
                continue
            if d.kind == "decl" and len(d.decl.declarators) == 1 and d.decl.declarators[0].init_kind == "=" \
                    and "&" not in d.decl.declarators[0].ptr and not d.decl.declarators[0].array:
                x = d.decl.declarators[0]
                name, value, keep = x.name, x.init, [(x.name_end, x.end, "")]
            elif d.kind == "expr" and d.expr.kind == "assign" and d.expr.op == "=" and d.expr.kids[0].kind == "name":
                name, value, keep = d.expr.kids[0].name, d.expr.kids[1], [(d.start, s.start, "")]
            else:
                continue
            uses = [x for x in cbody.expressions(s.expr) if x.kind == "name" and x.name == name]
            if len(uses) != 1:
                continue
            u = uses[0]
            if any(x.kind == "call" and x.kids and x.kids[0].end <= u.start and u.end <= x.end
                   for x in cbody.expressions(s.expr)):
                continue                                # `f((p = g), c)`: nobody writes an assignment in an argument
            yield (f"({name} = {site.text(value)}) in the condition",
                   keep + [(u.start, u.end, f"({name} = {site.text(value)})")])
    for s in stmts_of(site, "if", "while"):
        for x in cbody.expressions(s.expr):
            if x.kind == "paren" and x.kids[0].kind == "assign" and x.kids[0].op == "=" and x.kids[0].kids[0].kind == "name" \
                    and site.in_block(s) and s.kind == "if":
                a = x.kids[0]
                ind = cbody.indent_at(site.src, s.start)
                yield (f"{site.text(a)} before the condition",
                       [(s.start, s.start, f"{site.text(a)};\n{ind}"), (x.start, x.end, site.text(a.kids[0]))])


def r_declaration_order(site: Site):
    """Callee-saved numbers swapped between locals: named locals rank by declaration order (the rank
    rule). Moves one declaration before an earlier one of an enclosing block; an initializer stays
    behind as an assignment."""
    decls = [s for s in cbody.statements(site.body) if s.kind == "decl" and len(s.decl.declarators) == 1
             and site.in_block(s) and "static" not in site.src[s.decl.type_start:s.decl.type_end]]
    counts = Counter(s.decl.declarators[0].name for s in decls)
    for x in decls:
        dx = x.decl.declarators[0]
        if counts[dx.name] > 1:
            continue
        for y in decls:
            if y.start >= x.start or not _encloses(site, y, x):
                continue
            edits = _move_decl(site, x, y)
            if edits:
                yield f"{dx.name} declared before {y.decl.declarators[0].name}", edits


def _encloses(site: Site, outer: Stmt, inner: Stmt) -> bool:
    block = site.parents[id(outer)][0]
    return block.start <= inner.start and inner.end <= block.end


def decl_parts(site: Site, x: Stmt) -> tuple[str, tuple[int, int, str]] | None:
    """What moving the declaration statement x takes: (the bare declaration `T x;`, the edit that
    leaves its initializer behind as an assignment or removes the statement). None when it cannot
    move (a const, a reference, an array or constructor initializer)."""
    d = x.decl.declarators[0]
    typ = site.src[x.decl.type_start:x.decl.type_end]
    if d.init is not None and ("const" in typ.split() or "&" in d.ptr or d.init_kind != "=" or d.array):
        return None
    head = site.src[x.decl.type_start:d.name_end] + d.array + ";"
    if d.init is not None:
        return head, (x.start, x.end, f"{d.name} = {site.text(d.init)};")
    block, k = site.parents[id(x)]
    nxt = block.kids[k + 1].start if k + 1 < len(block.kids) else x.end
    return head, (x.start, max(nxt, x.end), "")


def _move_decl(site: Site, x: Stmt, before: Stmt) -> list[tuple[int, int, str]] | None:
    parts = decl_parts(site, x)
    if parts is None:
        return None
    head, stay = parts
    return [(before.start, before.start, f"{head}\n{cbody.indent_at(site.src, before.start)}"), stay]


def r_split_init(site: Site):
    """`T x = e;` vs `T x; x = e;`: where the target's rank needs it, the declaration and the
    initialization part."""
    for s in stmts_of(site, "decl"):
        if len(s.decl.declarators) != 1 or not site.in_block(s):
            continue
        d = s.decl.declarators[0]
        if d.init_kind != "=" or "&" in d.ptr or "const" in site.src[s.decl.type_start:s.decl.type_end].split() or d.array:
            continue
        ind = cbody.indent_at(site.src, s.start)
        new = site.src[s.decl.type_start:d.name_end] + f";\n{ind}{d.name} = {site.text(d.init)};"
        yield f"{d.name} declared, then assigned", [(s.start, s.end, new)]


def r_cast_call(site: Site):
    """An inline's result tested in r3 and only then copied to a saved register: a cast on the call,
    even a redundant one (`T* p = (T*)f(w);`)."""
    for s in stmts_of(site, "decl"):
        if len(s.decl.declarators) != 1:
            continue
        d = s.decl.declarators[0]
        if d.init is None or d.init_kind != "=" or not d.ptr.startswith("*"):
            continue
        typ = site.src[s.decl.type_start:s.decl.type_end].replace("static", "").strip() + d.ptr
        if d.init.kind == "call":
            yield f"({typ}) on {d.name}'s call", [(d.init.start, d.init.start, f"({typ})")]
        elif d.init.kind == "cast" and d.init.kids[0].kind == "call":
            yield f"cast on {d.name}'s call dropped", [(d.init.start, d.init.end, site.text(d.init.kids[0]))]


def r_compare_cast(site: Site):
    """`lbz; cmpwi` vs `lbz; cmplwi` (and `cmpw`/`cmplw`): `(int)c == 3` vs `c == 3`."""
    for e in exprs_of(site):
        if e.kind != "binary" or e.op not in MIRROR:
            continue
        a, b = e.kids
        for side, lit in ((a, b), (b, a)):
            if lit.kind not in ("num", "chr") or side.kind in ("num", "chr"):
                continue
            if side.kind == "cast":
                yield f"cast dropped in {site.text(e)}", [(side.start, side.kids[0].start, "")]
                continue
            for t in ("int", "unsigned int"):
                yield f"({t}) in {site.text(e)}", [(side.start, side.end, f"({t}){wrap(site, side)}")]


def r_if_invert(site: Site):
    """A branch the other way round with the same blocks: `if (c) A else B` vs `if (!c) B else A`."""
    for s in stmts_of(site, "if"):
        if len(s.kids) != 2 or s.kids[1].kind == "if":
            continue
        a, b = s.kids
        new = f"if ({negate(site, s.expr)}) {_as_block(site, s, b)} else {_as_block(site, s, a)}"
        yield f"if ({site.text(s.expr)}) branches swapped", [(s.start, s.end, new)]


def r_nest_join(site: Site):
    """A branch that lands on the next test instead of leaving: `if (a) { if (b) X } [else Y]` written as
    `if (a && b) X [else Y]`. Not equivalent when there is an else (a true, b false now runs Y): only the
    compiler's exact match keeps it (tool-gaps #6, a tree class vf39)."""
    for s in stmts_of(site, "if"):
        then = unbraced(s.kids[0])
        if then is None or then.kind != "if" or len(then.kids) != 1:
            continue
        new = f"if ({_and_operand(site, s.expr)} && {_and_operand(site, then.expr)}) {_as_block(site, then, then.kids[0])}"
        if len(s.kids) == 2:
            new += f" else {site.text(s.kids[1])}"
        yield f"if ({site.text(s.expr)}) {{ if ({site.text(then.expr)}) }} joined with &&", [(s.start, s.end, new)]
    for s in stmts_of(site, "if"):                  # `if (a) { if (b) X  rest }` -> `if (a && b) { X  rest }`
        block = s.kids[0]
        if len(s.kids) != 1 or block.kind != "block" or len(block.kids) < 2:
            continue
        first = block.kids[0]
        if first.kind != "if" or len(first.kids) != 1:
            continue
        inner = first.kids[0].kids if first.kids[0].kind == "block" else [first.kids[0]]
        ind = cbody.indent_at(site.src, block.kids[1].start)
        body = "".join(f"\n{ind}{site.text(x)}" for x in inner + block.kids[1:])
        new = f"if ({_and_operand(site, s.expr)} && {_and_operand(site, first.expr)}) {{{body}\n{cbody.indent_at(site.src, s.start)}}}"
        yield f"if ({site.text(s.expr)}) {{ if ({site.text(first.expr)}) … }} joined with && over the block", \
            [(s.start, s.end, new)]


def _and_operand(site: Site, e: Expr) -> str:
    """An operand of `&&`: parenthesized only when it binds looser (`||`, `?:`, an assignment, a comma)."""
    loose = e.kind in ("ternary", "assign", "comma") or (e.kind == "binary" and e.op == "||")
    return f"({site.text(e)})" if loose else site.text(e)


COMPARE_OPS = ("==", "!=", "<", ">", "<=", ">=")


def r_int_flag(site: Site):
    """A comparison returned or stored as an int: `c ? 1 : 0` keeps the compiler's plain shift where a bare `c`
    goes through bool (a byte mask), and the reverse (a window class vf11)."""
    for s in stmts_of(site, "return", "expr"):
        e = s.expr
        if e is None:
            continue
        if s.kind == "expr" and e.kind == "assign" and e.op == "=":
            e = e.kids[1]
        if e.kind == "binary" and e.op in COMPARE_OPS or e.kind == "unary" and e.op == "!":
            yield f"{site.text(e)} -> ({site.text(e)}) ? 1 : 0", [(e.start, e.end, f"({site.text(e)}) ? 1 : 0")]
        elif e.kind == "ternary" and [site.text(k) for k in e.kids[1:]] == ["1", "0"]:
            c = e.kids[0]
            inner = c.kids[0] if c.kind == "paren" else c
            yield f"{site.text(e)} -> {site.text(inner)}", [(e.start, e.end, site.text(inner))]


def r_fold_single(site: Site):
    """A local used once, in the next statement: `T x = E; f(x);` written `f(E);`. The value becomes a temporary,
    whose stack slot comes after the named locals' (a by-reference argument)."""
    for s in stmts_of(site, "decl"):
        if not site.in_block(s) or len(s.decl.declarators) != 1:
            continue
        d = s.decl.declarators[0]
        if d.init is None or d.init_kind != "=" or d.ptr or d.array or "static" in site.src[s.decl.type_start:s.decl.type_end]:
            continue
        block, i = site.parents[id(s)]
        if i + 1 >= len(block.kids):
            continue
        nxt = block.kids[i + 1]
        uses = [t for t in cbody.tokens(site.src, s.end, block.end) if t.kind == "id" and t.text == d.name]
        if len(uses) != 1 or not nxt.start <= uses[0].start < nxt.end:
            continue
        yield f"{d.name} folded into its one use", [(s.start, nxt.start, ""), (uses[0].start, uses[0].end, wrap(site, d.init))]


def r_hoist_tail(site: Site):
    """A call the target makes on both paths: the last statement of an if's only branch moved after the if
    (tool-gaps #6, a text-edit class vf31)."""
    for s in stmts_of(site, "if"):
        if len(s.kids) != 1 or s.kids[0].kind != "block" or len(s.kids[0].kids) < 2:
            continue
        last = s.kids[0].kids[-1]
        if last.kind != "expr":
            continue
        cut = site.src.rfind("\n", 0, last.start)
        ind = cbody.indent_at(site.src, s.start)
        yield f"{site.text(last)} moved after if ({site.text(s.expr)})", \
            [(cut, last.end, ""), (s.end, s.end, f"\n{ind}{site.text(last)}")]


def r_break_guard(site: Site):
    """A loop that leaves as soon as a value fails its test: `if (p) { S }` closing a loop body written as
    `if (!p) break; S` (tool-gaps #6)."""
    for loop in stmts_of(site, "while", "for", "do", near=False):
        body = loop.kids[0] if loop.kids else None
        if body is None or body.kind != "block" or not body.kids:
            continue
        last = body.kids[-1]
        if last.kind != "if" or len(last.kids) != 1 or not site.near(last):
            continue
        inner = last.kids[0].kids if last.kids[0].kind == "block" else [last.kids[0]]
        if not inner:
            continue
        ind = cbody.indent_at(site.src, last.start)
        rest = "".join(f"\n{ind}{site.text(x)}" for x in inner)
        yield f"if ({site.text(last.expr)}) {{…}} ending the loop -> break", \
            [(last.start, last.end, f"if ({negate(site, last.expr)}) {{\n{ind}    break;\n{ind}}}{rest}")]


def _as_block(site: Site, at: Stmt, s: Stmt) -> str:
    return site.text(s) if s.kind == "block" else braced(site, at, [site.text(s)])


def r_statement_order(site: Site):
    """Loads or stores scheduled in another order: statements' order sets it (a store is never moved
    past a load that may alias it; locals' initializing loads follow declaration order). Swaps two
    adjacent statements that do not depend on each other through a named variable."""
    for block in stmts_of(site, "block", near=False):
        kids = block.kids
        for i in range(len(kids) - 1):
            a, b = kids[i], kids[i + 1]
            if a.kind not in ("expr", "decl") or b.kind not in ("expr", "decl") or not (site.near(a) or site.near(b)):
                continue
            if i > 0 and kids[i - 1].kind in ("case", "default", "label"):
                continue
            wa, ra = _effects(site, a)
            wb, rb = _effects(site, b)
            if wa & (rb | wb) or wb & ra:
                continue
            if _same_call_run(site, kids, i) >= 3:
                continue                                # two of a run of identical calls: the asm gives it away
            text_a, text_b = site.text(a), site.text(b)
            yield f"swapped {_short(text_a)} / {_short(text_b)}", [(a.start, a.end, text_b), (b.start, b.end, text_a)]


def _callee(site: Site, s: Stmt) -> str | None:
    if s.kind == "expr" and s.expr is not None and s.expr.kind == "call" and s.expr.kids:
        return site.text(s.expr.kids[0])
    return None


def _same_call_run(site: Site, kids: list[Stmt], i: int) -> int:
    """How many adjacent statements around kids[i], kids[i+1] call the same function (0: they don't)."""
    name = _callee(site, kids[i])
    if name is None or _callee(site, kids[i + 1]) != name:
        return 0
    lo, hi = i, i + 1
    while lo > 0 and _callee(site, kids[lo - 1]) == name:
        lo -= 1
    while hi + 1 < len(kids) and _callee(site, kids[hi + 1]) == name:
        hi += 1
    return hi - lo + 1


def _effects(site: Site, s: Stmt) -> tuple[set[str], set[str]]:
    """(names written or declared, names read) by a statement."""
    writes = _writes(site, s)
    if s.decl is not None:
        writes |= {d.name for d in s.decl.declarators}
    reads = {n for top in _all_exprs(s) for n in cbody.names(top)}
    return writes, reads - writes if s.kind == "decl" else reads


def _short(t: str) -> str:
    t = " ".join(t.split())
    return t if len(t) <= 40 else t[:37] + "..."


RULES: list[Rule] = [
    Rule("decl-order", "Callee-saved register numbers swapped between locals (or with `this`)",
         "declare the locals in the rank order (the rank rule; `dk close` solves it from the registers)",
         ("regs", "stack"), r_declaration_order),
    Rule("operand-swap", "A commutative instruction's operands the other way round (`add`, `mullw`, `fmuls`)",
         "swap plain operands (`x + y` vs `y + x`); name a compound operand in a local, then order it",
         ("swap", "regs"), r_operand_swap),
    Rule("op-assign", "`fmuls` operand order: constant first vs variable first",
         "`x *= K` puts the variable first; `x = K * x` the constant", ("swap", "regs"), r_op_assign),
    Rule("compare-flip", "A compare's operands the other way round (`cmplw r0,r3` vs `cmplw r3,r0`; `fcmpu f, 0`)",
         "`a < b` vs `b > a`; `if (f)` vs `f != 0.0f`", ("compare", "branch", "regs"), r_compare_flip),
    Rule("half", "`x * 0.5f` with `x` loaded before the constant", "`x / 2.0f` (a literal `x * 0.5f` loads the constant first)",
         ("swap", "order", "regs"), r_half),
    Rule("reassociate", "`addi; add` where ours reassociates to `add; addi`", "plain `a + b + K`, not `a + (b + K)`",
         ("order", "regs", "count", "swap"), r_reassociate),
    Rule("ternary-if", "`bge; b; fmr` vs `cror` for a min/max; a value in another register after a branch",
         "a ternary gives the branches; `if`/`else` assignments may give `cror` or another rank", ("cror", "branch", "regs", "count"),
         r_ternary_if),
    Rule("one-case-switch", "`cmpwi; beq case; b default` where ours gives `bne`", "a `switch` with one `case`, not an `if`",
         ("branch", "count"), r_one_case_switch),
    Rule("or-return", "`if (!a || !b) return;` shape: `bne next; b exit`", "exactly that `||` early return (not two `if`s)",
         ("branch", "count"), r_or_return),
    Rule("if-invert", "A conditional branch inverted with the same blocks", "`if (!c) B else A` for `if (c) A else B`",
         ("branch",), r_if_invert),
    Rule("nest-join", "A test's false path lands on the next test, not past the block",
         "`if (a && b) X else Y` for `if (a) { if (b) X } else Y`", ("branch", "landing"), r_nest_join),
    Rule("int-flag", "A comparison's result masked to a byte on one side, plain on the other",
         "`(c) ? 1 : 0` for a bare `c` returned or stored as an int (and the reverse)", ("mask",), r_int_flag),
    Rule("fold-single", "Stack slots in another order: a value ours names (slot with the locals) the target keeps as a temporary",
         "`f(E);` for `T x = E; f(x);` when x has no other use", ("stack", "count"), r_fold_single),
    Rule("hoist-tail", "A branch lands on a call that ours makes inside the if",
         "the if branch's last statement after the if", ("branch", "landing"), r_hoist_tail),
    Rule("break-guard", "A loop's null test jumps past the loop, not to its own test",
         "`if (!p) break;` then the rest, for `if (p) { … }` closing the loop body", ("branch", "landing"),
         r_break_guard),
    Rule("chained-assign", "Stores in reverse order (z, y, x); `li` where the target copies a zero with `mr`",
         "chained assignment `a = b = c = K` stores the last one first", ("order", "zero"), r_chained_assign),
    Rule("unsigned-index", "`addi r0, rOff, K; lwzx` vs a stepped pointer in a loop",
         "a signed counter converted to unsigned (`a[(unsigned int)i]`), or the counter's type", ("index", "regs", "count"),
         r_unsigned_index),
    Rule("ref-element", "`mulli; addi; lfsx` on an array element", "the element read through a reference: `T& e = a[i];`",
         ("index", "count", "regs"), r_ref_element),
    Rule("flag-form", "A `bool` flag in its own zero register where the target shares the block's `li r30, 0`",
         "`bool b = a || c;`, not `bool b = false; if (a || c) b = true;`", ("zero", "regs", "branch", "count"), r_flag_form),
    Rule("assign-in-condition", "A value loaded into r3 and copied (`cmplwi r3, 0; mr r31, r3`) where the target loads it straight",
         "`if ((p = g) != 0)`; `T* p = g; if (p != 0)` makes the copy", ("regs", "count"), r_assign_in_condition),
    Rule("cast-call", "An inline's result tested in r3 and then copied to a saved register",
         "a cast on the inline call, even a redundant one: `T* p = (T*)f(w);`", ("regs", "count"), r_cast_call),
    Rule("compare-cast", "`lbz; cmpwi` vs `lbz; cmplwi`; `cmpw` vs `cmplw`", "`(int)c == 3` vs `c == 3`", ("signed",), r_compare_cast),
    Rule("statement-order", "Loads or stores scheduled in another order; a store the target keeps ahead of a load",
         "the statements in the target's order (independent neighbours swapped)", ("order", "stack", "regs"),
         r_statement_order),
    Rule("split-init", "A local ranked as a temporary (it merged with the value that made it)",
         "`T x; x = e;` instead of `T x = e;`", ("regs",), r_split_init),
]

from .inlinehelper import PARAMS_RULE as _HELPER_PARAMS, RULE as _INLINE_HELPER   # noqa: E402 (builds on the table)

RULES += [_INLINE_HELPER, _HELPER_PARAMS]

from .shaperules import RULES as _SHAPE_RULES   # noqa: E402 (shape rows measured on landed fixes)

RULES += _SHAPE_RULES

from .loopmoves import RULES as _LOOP_RULES   # noqa: E402 (loop rows: T47 card 3)

RULES += _LOOP_RULES

from .giantmoves import RULES as _SCOPE_RULES   # noqa: E402 (scope rows: T49 card 1)

RULES += _SCOPE_RULES
BY_NAME = {r.name: r for r in RULES}


SPECIFIC = ("landing", "mask", "stack")       # symptoms few rows show and few rules answer


def ranked(sym: Symptoms) -> list[Rule]:
    """The rules ordered by how much of the symptom they answer (table order breaks ties)."""
    weight = lambda r: sum(sym.signs.get(s, 0) for s in r.signs)
    first = lambda r: any(sym.signs.get(x) and x in r.signs for x in SPECIFIC)  # rare signs: never crowded out by regs
    return sorted(RULES, key=lambda r: (not first(r), -weight(r), RULES.index(r)))
