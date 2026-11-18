"""Loop rows of the rewrite table (T47 card 3, a save-file loop). Each is a forward move and its
inverse, so the same generator also makes synthetic before-versions.

- for-init-list: locals declared with a literal initializer before a `for`, and not used until it, set
  in the loop's init list instead (`int a = 0; for (i = 0; ...)` -> `int a; for (i = 0, a = 0; ...)`),
  in any order after the loop's own; and the reverse (an init-list assignment back onto the
  declaration). Scheduling only: the pre-header `li` rows come in init-list order.
- iv-alias: two counters provably equal inside a loop body (the same literal at loop entry, one the
  `for` step `i++`, the other `c++` as the body's last statement, neither written anywhere else, no
  `continue`) read through the other one in a `c * K`, `c << K` or `a[c]` (both ways). The compiler
  derives a strength-reduced IV from the counter the source names, so its latch `addi` moves.

Legality is checked from the parse alone (cbody): only scalars (no class types, references, arrays),
each local declared once in the function, no label or case crossed, values that are literals. Nothing
here compiles anything: the closer does, and its judge keeps only an exact, faithful result.
"""

from __future__ import annotations

import itertools
import re

from ...core import cbody
from ...core.cbody import Expr, Stmt
from .rewrites import Rule, Site, _short, stmts_of

SCALAR_WORDS = cbody.TYPE_WORDS | cbody.COMMON_TYPES | {"register"}
MAX_MOVED = 3


# ---- helpers ---------------------------------------------------------------------------------------

def _decl_counts(site: Site) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in cbody.statements(site.body):
        if s.kind == "decl" and s.decl is not None:
            for d in s.decl.declarators:
                out[d.name] = out.get(d.name, 0) + 1
    return out


def _scalar_decl(site: Site, k: Stmt) -> bool:
    """One declarator of a plain arithmetic type (or a pointer to anything), not const/static."""
    if k.kind != "decl" or k.decl is None or len(k.decl.declarators) != 1:
        return False
    d = k.decl.declarators[0]
    if d.array or "&" in d.ptr:
        return False
    words = site.src[k.decl.type_start:k.decl.type_end].split()
    if any(w in ("const", "static", "volatile", "extern") for w in words):
        return False
    return d.ptr.startswith("*") or all(w in SCALAR_WORDS for w in words)


def _lit(site: Site, e: Expr | None) -> str | None:
    """The literal's text (`0`, `-1`, `'a'`), or None."""
    if e is None:
        return None
    while e.kind == "paren":
        e = e.kids[0]
    if e.kind in ("num", "chr") or (e.kind == "name" and e.name in ("NULL", "true", "false")):
        return " ".join(site.text(e).split())
    if e.kind == "unary" and e.op == "-" and e.kids[0].kind == "num":
        return "-" + site.text(e.kids[0]).strip()
    return None


def _lit_value(t: str) -> object:
    try:
        return int(re.sub(r"[uUlL]+$", "", t), 0)
    except ValueError:
        return t


def _comma_parts(e: Expr | None) -> list[Expr]:
    if e is None:
        return []
    if e.kind == "comma":
        return [p for k in e.kids for p in _comma_parts(k)]
    return [e]


def _simple_assign(e: Expr) -> str | None:
    """`v = X` with a plain name on the left: v, else None."""
    if e.kind == "assign" and e.op == "=" and e.kids[0].kind == "name":
        return e.kids[0].name
    return None


def _mentions(site: Site, lo: int, hi: int, name: str) -> bool:
    toks = cbody.tokens(site.src, lo, hi)
    return any(t.kind == "id" and t.text == name and not (k and toks[k - 1].text in (".", "->", "::"))
               for k, t in enumerate(toks))


def _crosses_label(block: Stmt, a: int, b: int) -> bool:
    return any(block.kids[j].kind in ("case", "default", "label", "pp") for j in range(a, b))


def _decl_of(block: Stmt, k: Stmt) -> int | None:
    return next((j for j, x in enumerate(block.kids) if x is k), None)


# ---- for-init-list ---------------------------------------------------------------------------------

def r_for_init_list(site: Site):
    """Literal initializers moved between their declarations and a `for`'s init list, both ways."""
    counts = _decl_counts(site)
    for s in stmts_of(site, "for"):
        if not site.in_block(s) or s.init is None or s.init.kind not in ("expr", "empty"):
            continue
        block, i = site.parents[id(s)]
        parts = _comma_parts(s.init.expr) if s.init.kind == "expr" else []
        part_names = {n for p in parts for n in cbody.names(p)}
        lo_init = s.init.start

        # forward: declarations with a literal initializer, nearest first
        movable: list[tuple[Stmt, str, str]] = []
        for j in range(i - 1, -1, -1):
            k = block.kids[j]
            if k.kind in ("case", "default", "label", "pp"):
                break
            if not _scalar_decl(site, k):
                continue
            d = k.decl.declarators[0]
            lit = _lit(site, d.init) if d.init_kind == "=" else None
            if lit is None or counts.get(d.name) != 1 or d.name in part_names:
                continue
            if _mentions(site, k.end, s.start, d.name) or _mentions(site, s.start, lo_init, d.name):
                continue
            movable.append((k, d.name, lit))
            if len(movable) >= MAX_MOVED:
                break
        movable.reverse()
        groups = [[m] for m in movable]
        if len(movable) >= 2:
            groups += [list(p) for p in itertools.permutations(movable)]
        for g in groups:
            edits = [(k.decl.declarators[0].name_end, k.decl.declarators[0].end, "") for k, _, _ in g]
            add = ", ".join(f"{n} = {lit}" for _, n, lit in g)
            if s.init.kind == "expr":
                edits.append((s.init.expr.end, s.init.expr.end, ", " + add))
            else:
                edits.append((s.init.start, s.init.start, add))
            yield (f"{', '.join(n for _, n, _ in g)} initialised in the for init list", edits)

        # inverse: init-list assignments of a literal back onto the local's declaration
        back: list[tuple[int, Stmt, str]] = []
        for t, p in enumerate(parts):
            v = _simple_assign(p)
            lit = _lit(site, p.kids[1]) if v else None
            if lit is None or counts.get(v) != 1:
                continue
            k = next((x for x in block.kids[:i] if x.kind == "decl" and x.decl is not None
                      and any(d.name == v for d in x.decl.declarators)), None)
            if k is None or not _scalar_decl(site, k) or k.decl.declarators[0].init_kind:
                continue
            j = _decl_of(block, k)
            if _crosses_label(block, j + 1, i) or _mentions(site, k.end, s.start, v) \
                    or any(v in cbody.names(q) for q in parts[:t]):
                continue
            back.append((t, k, lit))
        groups2 = [[b] for b in back] + ([back] if len(back) >= 2 else [])
        for g in groups2:
            gone = {t for t, _, _ in g}
            rest = ", ".join(site.text(p) for t, p in enumerate(parts) if t not in gone)
            edits = [(k.decl.declarators[0].name_end, k.decl.declarators[0].name_end, f" = {lit}") for _, k, lit in g]
            edits.append((s.init.expr.start, s.init.expr.end, rest))
            names = ", ".join(_simple_assign(parts[t]) for t, _, _ in g)
            yield (f"{names} initialised at the declaration, not in the for init list", edits)


# ---- iv-alias --------------------------------------------------------------------------------------

def _increment(site: Site, e: Expr | None) -> str | None:
    """`i++`, `++i`, `i += 1`: i."""
    if e is not None and e.kind == "assign" and e.op == "+=" and e.kids[0].kind == "name":
        lit = _lit(site, e.kids[1])
        return e.kids[0].name if lit is not None and _lit_value(lit) == 1 else None
    if e is not None and e.kind in ("postfix", "unary") and e.op == "++" and e.kids[0].kind == "name":
        return e.kids[0].name
    return None


def _continues(s: Stmt) -> bool:
    """A `continue` that binds to the loop whose body s is (nested loops' own are fine)."""
    if s.kind == "continue":
        return True
    if s.kind in ("for", "while", "do"):
        return False
    return any(_continues(k) for k in s.kids)


def _written(e: Expr) -> set[str]:
    out = set()
    for x in cbody.expressions(e):
        if x.kind == "assign" or x.kind == "postfix" or (x.kind == "unary" and x.op in ("++", "--")):
            if x.kids[0].kind == "name":
                out.add(x.kids[0].name)
    return out


def _decl_info(site: Site, name: str, s: Stmt) -> tuple[Stmt, str] | None:
    """The one declaration of `name` (a scalar) and its type key."""
    for x in cbody.statements(site.body):
        if x.kind == "decl" and x.decl is not None and any(d.name == name for d in x.decl.declarators):
            if not _scalar_decl(site, x) or x.decl.declarators[0].ptr:
                return None
            return x, re.sub(r"\s+", " ", site.src[x.decl.type_start:x.decl.type_end].replace("register", "")).strip()
    return None


def _entry_value(site: Site, s: Stmt, name: str, parts: list[Expr]) -> str | None:
    """The literal `name` holds when the loop starts: set in the init list (a declaration there or an
    assignment), or by its declaration in the block holding the loop with nothing in between using it."""
    if s.init is not None and s.init.kind == "decl":
        for d in s.init.decl.declarators:
            if d.name == name:
                return _lit(site, d.init) if d.init_kind == "=" else None
    hits = [p for p in parts if name in _written(p)]
    if hits:
        p = hits[-1]
        return _lit(site, p.kids[1]) if _simple_assign(p) == name and len(hits) == 1 else None
    if not site.in_block(s):
        return None
    block, i = site.parents[id(s)]
    for j in range(i - 1, -1, -1):
        k = block.kids[j]
        if k.kind in ("case", "default", "label", "pp"):
            return None
        if k.kind == "decl" and k.decl is not None and any(d.name == name for d in k.decl.declarators):
            d = k.decl.declarators[0]
            if d.init_kind != "=" or _mentions(site, k.end, s.start, name):
                return None
            return _lit(site, d.init)
    return None


def r_iv_alias(site: Site):
    """In a loop whose two counters are equal throughout its body, one read through the other in a
    `c * K`, `c << K` or `a[c]`."""
    counts = _decl_counts(site)
    for s in stmts_of(site, "for"):
        i = _increment(site, s.step)
        body = s.kids[0] if s.kids else None
        if i is None or body is None or body.kind != "block" or len(body.kids) < 2:
            continue
        last = body.kids[-1]
        c = _increment(site, last.expr) if last.kind == "expr" else None
        if c is None or c == i or counts.get(i) != 1 or counts.get(c) != 1:
            continue
        di, dc = _decl_info(site, i, s), _decl_info(site, c, s)
        if di is None or dc is None or di[1] != dc[1]:
            continue
        parts = _comma_parts(s.init.expr) if s.init is not None and s.init.kind == "expr" else []
        vi, vc = _entry_value(site, s, i, parts), _entry_value(site, s, c, parts)
        if vi is None or vc is None or _lit_value(vi) != _lit_value(vc):
            continue
        if s.cond is not None and _written(s.cond) & {i, c}:
            continue
        inner = [x for x in body.kids[:-1]]
        if any(_continues(x) for x in inner) or any(x.kind in ("label", "goto") for k in inner
                                                     for x in cbody.statements(k)):
            continue
        tops = [e for k in inner for x in cbody.statements(k) for e in cbody.statement_exprs(x)]
        if any(_written(e) & {i, c} for e in tops):
            continue
        if re.search(rf"&\s*(?:{re.escape(i)}|{re.escape(c)})\b", site.src[body.start:body.end]):
            continue
        nodes = [x for e in tops for x in cbody.expressions(e)]
        par = {id(k): x for x in nodes for k in x.kids}
        for x in nodes:
            if x.kind != "name" or x.name not in (i, c) or not site.near(x):
                continue
            p = par.get(id(x))
            if p is None:
                continue
            ok = (p.kind == "index" and p.kids[1] is x) or (
                p.kind == "binary" and p.op in ("*", "<<") and any(k is not x and k.kind == "num" for k in p.kids))
            if not ok:
                continue
            other = c if x.name == i else i
            yield (f"{x.name} -> {other} in {_short(site.text(p))} (equal counters)", [(x.start, x.end, other)])


RULES: list[Rule] = [
    Rule("for-init-list", "Pre-header `li`/`mr` rows in another order (same registers, same mnemonics)",
         "the loop's other locals initialised in the `for` init list (`for (i = 0, a = 0, b = 0; ...)`), "
         "in the target's order; or on their declarations", ("order",), r_for_init_list),
    Rule("iv-alias", "The latch's `addi` rows in another order: a `*K` IV (`addi rX, rX, K`) next to a `+1` counter",
         "read the strength-reduced expression (`c * 2`, `a[c]`) through the other of two equal counters",
         ("order", "regs"), r_iv_alias),
]
