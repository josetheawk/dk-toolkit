"""Shape rows of the rewrite table: idioms measured on landed shape fixes (`dk dataset fixes`). Every row
is a forward move and its inverse, so the same generator also makes synthetic before-versions.

- else-return: `if (c) return; S` / `if (!c) { S } else { return; }` / `if (c) { A; return; } else { B }`
  with the dead `else` dropped, and an empty then-block (`if (c) {} else { B }`);
- name-value: a call or load used as an operand or argument named in a local just before its
  statement (or before any of the declarations ahead of it), and a single-use local folded back in;
- inline-local: a local that only holds a pure value (or a call on pure arguments) written out at
  every use, and a repeated value named once, at its first use or any earlier statement it stays valid
  at (the fold's inverse restores where the declaration was);
- local-reuse: a local written as a dead parameter or earlier local of the same type, and the reverse;
- default-else: `T x = K; ... if (c) x = a;` with the default moved into an `else` (or a ternary);
- unsigned-compare: a literal compared as unsigned (`c == 10u` for `c == '\\n'`), a loop counter
  declared outside its `for` typed unsigned (and the reverses);
- accessor: `o->m[i].f` through the class's own indexing accessor (`o->Get(i)->f`, by pointer,
  reference or value), and back; only where the object's declared class is the accessor's;
- shared-labels: two cases with the same body sharing it, and the reverse. Only labels the switch
  already has: an invented case value is a device, never a rewrite;
- cached-chain: a pointer chain (`g->a->b`) read through a cached local in the statement right after
  the local's declaration, re-read through the chain with the declaration moved below it (the compiler
  merges the pointer, not the field: the target reloads the field), and the reverse;
- bool-index: a comparison used as an integer, `(int)(a >= b)` (a bool, zero-extended: `clrlslwi`)
  vs `a >= b ? 1 : 0` (an int select: plain `slwi`), both ways.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from ...core import cbody
from ...core.cbody import Expr, Stmt
from .rewrites import (INT_TYPES, JUMPS, MIRROR, TRUE, FALSE, Rule, Site, _guarded, _short, _writes, braced, negate,
                       single_assign, stmts_of, exprs_of, pure, wrap)

LOOPS = ("for", "while", "do")


# ---- helpers ---------------------------------------------------------------------------------------

def _ids(site: Site, lo: int, hi: int, name: str) -> list[int]:
    """Offsets of `name` used as a variable (not as a member after `.`/`->`) in src[lo:hi]."""
    toks = cbody.tokens(site.src, lo, hi)
    return [t.start for k, t in enumerate(toks)
            if t.kind == "id" and t.text == name and not (k and toks[k - 1].text in (".", "->", "::"))]


def _next_start(block: Stmt, k: int, s: Stmt) -> int:
    return block.kids[k + 1].start if k + 1 < len(block.kids) else s.end


def _in_loop(site: Site, s: Stmt) -> Stmt | None:
    x = s
    while True:
        p = site.parents.get(id(x))
        if p is None:
            return None
        if p[0].kind in LOOPS:
            return p[0]
        x = p[0]


def _reindent(site: Site, s: Stmt, new: str) -> str:
    old = cbody.indent_at(site.src, s.start)
    return site.text(s).replace("\n" + old, "\n" + new) if old != new else site.text(s)


def _block_text(site: Site, at: Stmt, stmts: list[Stmt]) -> str:
    """`{ stmts }` written at `at`'s indentation."""
    ind = cbody.indent_at(site.src, at.start)
    inner = ind + "    "
    return "{\n" + "".join(f"{inner}{_reindent(site, k, inner)}\n" for k in stmts) + ind + "}"


def _as_block(site: Site, at: Stmt, s: Stmt) -> str:
    return site.text(s) if s.kind == "block" else braced(site, at, [site.text(s)])


def _flat(site: Site, at: Stmt, s: Stmt) -> str:
    """A block's statements written out at `at`'s level (no braces)."""
    ind = cbody.indent_at(site.src, at.start)
    kids = s.kids if s.kind == "block" else [s]
    return f"\n{ind}".join(_reindent(site, k, ind) for k in kids)


def _empty(s: Stmt) -> bool:
    return (s.kind == "block" and not s.kids) or s.kind == "empty"


def _ends_jump(s: Stmt) -> bool:
    if s.kind == "block":
        return bool(s.kids) and s.kids[-1].kind in JUMPS
    return s.kind in JUMPS


def _literal(site: Site, e: Expr | None) -> bool:
    return e is not None and (e.kind in ("num", "chr") or (e.kind == "name" and e.name in TRUE + FALSE + ("NULL",))
                              or (e.kind == "unary" and e.op == "-" and e.kids[0].kind == "num"))


def _type_key(text: str) -> str:
    return re.sub(r"\s+", "", re.sub(r"\b(?:register|static)\b", "", text))


def _declare(site: Site, before: Stmt, name: str, text: str) -> list[tuple[int, int, str]] | None:
    if site.cplusplus:
        return site.declare(before, f"__typeof__({text}) {name} = {text}")
    decl = site.declare(before, f"__typeof__({text}) {name}")
    if decl is not None:
        decl.append((before.start, before.start, f"{name} = {text};\n{cbody.indent_at(site.src, before.start)}"))
    return decl


def _points(site: Site, block: Stmt, i: int, e: Expr, hi: int, decls_only: bool = False,
            skip: tuple = ()) -> list[int]:
    """Indices j <= i (nearest first) of `block`'s statements before which a local holding e's value can be
    declared and still hold it at kids[i], its first use: from kids[j] on nothing e reads is written or
    declared and no label is crossed. decls_only (name-value's fold): only declarations are crossed, pure ones
    when e calls. In C only within the declarations at the top of the block. A fold removes a declaration
    from any of these places, so its inverse offers each of them (the original place, hence the statement
    order, comes back)."""
    inames = set(cbody.names(e))
    has_call = any(x.kind == "call" for x in cbody.expressions(e))
    ndecl = next((k for k, x in enumerate(block.kids) if x.kind != "decl"), len(block.kids))
    out = [i] if site.cplusplus or i <= ndecl else []
    for j in range(i, 0, -1):
        k = block.kids[j - 1]
        if k.kind in ("case", "default", "label", "pp") or not site.in_block(k) \
                or (k.kind == "decl" and {d.name for d in k.decl.declarators} & inames):
            break
        if decls_only:
            if k.kind != "decl" or _writes(site, k) & inames or (has_call and not all(pure(t) for t in _value_nodes(site, k))):
                break
        elif not _stable(site, e, k.start, hi, skip):
            break
        if site.cplusplus or j - 1 <= ndecl:
            out.append(j - 1)
    return out


def _declare_at(site: Site, block: Stmt, j: int, name: str, text: str) -> list[tuple[int, int, str]]:
    """`__typeof__(v) name = v;` right before block.kids[j] (a place _points gave)."""
    at = block.kids[j]
    return [(at.start, at.start, f"__typeof__({text}) {name} = {text};\n{cbody.indent_at(site.src, at.start)}")]


def _parent_map(nodes: list[Expr]) -> dict[int, Expr]:
    return {id(k): x for x in nodes for k in x.kids}


def _single_decls(site: Site) -> dict[str, Stmt]:
    from .ranksolve import single_decls
    return single_decls(site)


# ---- else-return -----------------------------------------------------------------------------------

def r_else_return(site: Site):
    """A dead `else` after a then-block that jumps, the jump as the else-branch, an empty then-block."""
    for s in stmts_of(site, "if"):
        if not site.in_block(s):
            continue
        block, i = site.parents[id(s)]
        ind = cbody.indent_at(site.src, s.start)
        then = s.kids[0]
        if len(s.kids) == 1:
            rest = block.kids[i + 1:]
            if _ends_jump(then) and rest and not any(k.kind in ("case", "default", "label", "pp") for k in rest):
                end = rest[-1].end
                yield (f"if ({_short(site.text(s.expr))}) jump; rest -> rest under if (!c), jump in the else",
                       [(s.start, end, f"if ({negate(site, s.expr)}) {_block_text(site, s, rest)} else {_as_block(site, s, then)}")])
                yield (f"if ({_short(site.text(s.expr))}) jump; rest -> rest in an else",
                       [(s.start, end, f"if ({site.text(s.expr)}) {_as_block(site, s, then)} else {_block_text(site, s, rest)}")])
            elif not _empty(then) and not _ends_jump(then):           # inverse of the empty then-block
                yield (f"if ({_short(site.text(s.expr))}) B -> if (!c) {{}} else B",
                       [(s.start, s.end, f"if ({negate(site, s.expr)}) {{\n{ind}}} else {_as_block(site, s, then)}")])
            continue
        a, b = s.kids
        if b.kind == "if":
            continue
        if _empty(a):
            yield (f"if ({_short(site.text(s.expr))}) {{}} else B -> if (!c) B",
                   [(s.start, s.end, f"if ({negate(site, s.expr)}) {_as_block(site, s, b)}")])
            continue
        if _ends_jump(a) and not _empty(b):
            yield (f"if ({_short(site.text(s.expr))}) {{...jump}} else B -> else dropped",
                   [(s.start, s.end, f"if ({site.text(s.expr)}) {_as_block(site, s, a)}\n{ind}{_flat(site, s, b)}")])
        if _ends_jump(b) and not _empty(b):
            yield (f"if ({_short(site.text(s.expr))}) A else {{...jump}} -> if (!c) jump; A",
                   [(s.start, s.end, f"if ({negate(site, s.expr)}) {_as_block(site, s, b)}\n{ind}{_flat(site, s, a)}")])


# ---- name-value / fold -----------------------------------------------------------------------------

def _value_nodes(site: Site, s: Stmt) -> list[Expr]:
    if s.kind == "decl":
        return [d.init for d in s.decl.declarators if d.init is not None and d.init_kind == "="]
    return [s.expr] if s.expr is not None else []


def _earlier_call(calls: list[Expr], e: Expr) -> bool:
    """A call evaluated before e that does not hold e (naming e would move e ahead of it)."""
    return any(c.start < e.start and not (c.start <= e.start and e.end <= c.end) for c in calls)


def r_name_value(site: Site):
    """`f(g(x))` vs `T t = g(x); f(t);`: a call or load named in a local right before its statement, or
    before the declarations that precede it; and a local used once folded back into its use."""
    for s in stmts_of(site, "expr", "decl", "return", "if", "switch"):
        if not site.in_block(s):
            continue
        tops = _value_nodes(site, s)
        nodes = [x for t in tops for x in cbody.expressions(t)]
        par = _parent_map(nodes)
        calls = [x for x in nodes if x.kind == "call"]
        block, i = site.parents[id(s)]
        for e in nodes:
            if not site.near(e):
                continue
            p = par.get(id(e))
            vf = None
            if e.kind not in ("call", "member", "index"):
                # T37: the value forms the fold writes (in parentheses: `(a * b)`, `((T*)p)`, `(&x)`), named as
                # the inner value; also a whole return value or initializer, and the base of `->`/`[]` (a pointer)
                vf = _value_form(site, e, par) if e.kind == "paren" else None
                if vf is None or s.kind == "expr" and any(e is t for t in tops) or not _pure_or_call(vf[1]):
                    continue
                if p is not None and p.kind == "member" and p.op == "." and p.kids[0] is e:
                    continue
            else:
                top = any(e is t for t in tops)
                if top and s.kind not in ("if", "switch"):
                    continue                               # a whole initializer or statement: split-init's job
                if p is not None and (p.kind in ("member", "index") and p.kids[0] is e):
                    continue
            if p is not None and (p.kind == "call" and p.kids[0] is e or p.kind == "assign" and p.kids[0] is e
                                  or p.kind in ("unary", "postfix") and p.op in ("&", "++", "--")):
                continue
            if _guarded(site, s, e) or _earlier_call(calls, e):
                continue
            text = site.text(e) if vf is None else site.text(vf[1])
            name = site.fresh("t")
            decl = _declare(site, s, name, text)
            if decl is not None:
                yield f"named {_short(text)} as {name}", decl + [(e.start, e.end, name)]
            pts = [j for j in _points(site, block, i, e if vf is None else vf[1], s.start, decls_only=True)
                   if j < i or not site.cplusplus]
            for j in pts:
                where = "ahead of the declarations" if j == pts[-1] else f"{i - j} declarations earlier"
                if j == i:
                    where = "initialized at the top"
                yield f"named {_short(text)} as {name} {where}", _declare_at(site, block, j, name, text) + [(e.start, e.end, name)]
            site.taken.discard(name)
    for s in stmts_of(site, "decl"):
        if not site.in_block(s) or len(s.decl.declarators) != 1:
            continue
        d = s.decl.declarators[0]
        if d.init is None or d.init_kind != "=" or "&" in d.ptr or d.array or "static" in site.src[s.decl.type_start:s.decl.type_end]:
            continue
        block, i = site.parents[id(s)]
        uses = _ids(site, s.end, block.end, d.name)
        if len(uses) != 1:
            continue
        at = uses[0]
        j = next((k for k in range(i + 1, len(block.kids)) if block.kids[k].start <= at < block.kids[k].end), None)
        if j is None:
            continue
        use = block.kids[j]
        if use.kind not in ("expr", "decl", "return", "if", "switch"):
            continue
        tops = _value_nodes(site, use)
        node = next((x for t in tops for x in cbody.expressions(t) if x.kind == "name" and x.start == at), None)
        if node is None or _guarded(site, use, node):
            continue
        inames = set(cbody.names(d.init))
        has_call = any(x.kind == "call" for x in cbody.expressions(d.init))
        if has_call and _earlier_call([x for t in tops for x in cbody.expressions(t) if x.kind == "call"], node):
            continue
        between = block.kids[i + 1:j]
        if any(k.kind != "decl" or _writes(site, k) & inames or {x.name for x in k.decl.declarators} & inames
               or (has_call and not all(pure(t) for t in _value_nodes(site, k))) for k in between):
            continue
        init = site.text(d.init) if d.init.kind in ("name", "num", "call", "member", "index", "paren") else f"({site.text(d.init)})"
        yield f"{d.name} folded into its use", [(s.start, _next_start(block, i, s), ""), (at, at + len(d.name), init)]


# ---- inline-local / repeated value named -----------------------------------------------------------

def _pure_or_call(e: Expr) -> bool:
    """No assignment or increment; calls allowed only with pure arguments."""
    for x in cbody.expressions(e):
        if x.kind in ("assign", "postfix") or (x.kind == "unary" and x.op in ("++", "--", "new", "delete")):
            return False
        if x.kind == "call" and not all(pure(a) for a in x.kids[1:]):
            return False
    return True


def _lhs_texts(site: Site, lo: int, hi: int) -> list[str]:
    out = []
    for s in cbody.statements(site.body):
        if s.start < lo or s.start >= hi:
            continue
        for top in cbody.statement_exprs(s):
            for x in cbody.expressions(top):
                if x.kind == "assign" or x.kind == "postfix" or (x.kind == "unary" and x.op in ("++", "--")):
                    out.append(" ".join(site.text(x.kids[0]).split()))
    return out


def _stable(site: Site, e: Expr, lo: int, hi: int, skip: tuple = ()) -> bool:
    """Nothing e reads (a name, or the value e itself) is written between lo and hi. skip: the spans of the
    uses being named (their own `&x` is the value, not an alias)."""
    inames = set(cbody.names(e))
    text = " ".join(site.text(e).split())
    for lhs in _lhs_texts(site, lo, hi):
        if lhs == text or (re.fullmatch(r"[A-Za-z_]\w*", lhs) and lhs in inames):
            return False
    seg = site.src[lo:hi]
    for a, b in skip:
        if lo <= a and b <= hi:
            seg = seg[:a - lo] + " " * (b - a) + seg[b - lo:]
    return not any(re.search(rf"&\s*{re.escape(n)}\b", seg) for n in inames)


# the value forms a fold writes out besides calls and loads (T37): named back by the naming moves
VALUE_KINDS = ("binary", "cast", "ternary", "unary")
UNARY_VALUES = ("-", "!", "~", "*", "&")


def _value_form(site: Site, e: Expr, par: dict) -> tuple[str, Expr] | None:
    """(key, value) when e is a value form a fold writes out: a binary expression, cast, ternary or unary
    (-, !, ~, *, &) that reads a name, alone or in the parentheses the fold put around it. The use is e (the
    parenthesized span when there are parentheses), the value the inner expression (declared as written)."""
    v = e.kids[0] if e.kind == "paren" else e
    if v.kind not in VALUE_KINDS or (v.kind == "unary" and v.op not in UNARY_VALUES) or not cbody.names(v):
        return None
    p = par.get(id(e))
    if e.kind != "paren" and p is not None and p.kind == "paren":
        return None                                     # the parentheses are the use
    return _key(site, v), v


def _key(site: Site, v: Expr) -> str:
    if v.kind == "unary" and v.op == "&":
        return "&" + " ".join(site.text(v.kids[0]).split())
    return " ".join(site.text(v).split())


def _postfix_wrap(site: Site, e: Expr, after: str) -> str:
    """The fold's text for a use followed by `after`: a cast or unary value before `.`, `->`, `[` or `(`
    needs parentheses too (`(T*)p` at `q->f` is `((T*)p)->f`, not `(T*)p->f`)."""
    if e.kind in ("cast", "unary") and re.match(r"(?:\.|->|\[|\()", after):
        return f"({site.text(e)})"
    return wrap(site, e)


def r_inline_local(site: Site):
    """`T* a = f(x); a->g(); a->h();` vs `f(x)->g(); f(x)->h();`; `T* e = p->m; e[i].a = 1;` vs
    `p->m[i].a = 1;`: a local holding a value nothing changes, written out at every use; and the reverse."""
    for s in stmts_of(site, "decl"):
        if not site.in_block(s) or len(s.decl.declarators) != 1:
            continue
        d = s.decl.declarators[0]
        typ = site.src[s.decl.type_start:s.decl.type_end]
        if d.init is None or d.init_kind != "=" or "&" in d.ptr or d.array or "static" in typ or not _pure_or_call(d.init):
            continue
        if d.init.kind in ("num", "chr", "str", "name"):
            continue
        block, i = site.parents[id(s)]
        uses = _ids(site, s.end, block.end, d.name)
        if len(uses) < 2 or not _stable(site, d.init, s.end, block.end):
            continue
        if d.name in _lhs_texts(site, s.end, block.end) or re.search(rf"&\s*{re.escape(d.name)}\b", site.src[s.end:block.end]):
            continue
        edits = [(s.start, _next_start(block, i, s), "")]
        addr = d.init.kind == "unary" and d.init.op == "&"
        for at in uses:
            after = site.src[at + len(d.name):at + len(d.name) + 4].lstrip()
            if addr and after.startswith("->"):
                arrow = site.src.index("->", at + len(d.name))
                edits.append((at, arrow + 2, f"{wrap(site, d.init.kids[0])}."))
            else:
                edits.append((at, at + len(d.name), _postfix_wrap(site, d.init, after)))
        yield f"{d.name} written out at its {len(uses)} uses", edits
    for block in stmts_of(site, "block", near=False):
        # key -> [(start, end, replacement after the name, value node)]: the same value at each use; `&E` also
        # at `E.f` (what the fold of `T* p = &E;` writes for `p->f`), named back as `t->f`
        found: dict[str, list[tuple[int, int, str, Expr]]] = {}
        for k in block.kids:
            for top in (x for st in cbody.statements(k) for x in cbody.statement_exprs(st)):
                nodes = list(cbody.expressions(top))
                par = _parent_map(nodes)
                for e in nodes:
                    if not site.near(e) or not _pure_or_call(e):
                        continue
                    p = par.get(id(e))
                    lvalue = p is not None and (p.kind == "call" and p.kids[0] is e or p.kind == "assign" and p.kids[0] is e
                                                or p.kind in ("unary", "postfix") and p.op in ("&", "++", "--"))
                    if e.kind == "member" and e.op == "." and e.kids[0].kind in ("name", "member", "index") \
                            and not (p is not None and p.kind == "unary" and p.op == "&"):
                        b = e.kids[0]
                        dot = site.src.index(".", b.end)
                        found.setdefault("&" + " ".join(site.text(b).split()), []).append((b.start, dot + 1, "->", b))
                    if lvalue:
                        continue
                    if e.kind in ("call", "member", "index"):
                        if p is not None and p.kind == "member" and p.op == "." and p.kids[0] is e:
                            continue
                        if e.kind == "member" and not any(x.kind in ("call", "index") or x.op == "->" for x in cbody.expressions(e)):
                            continue
                        found.setdefault(" ".join(site.text(e).split()), []).append((e.start, e.end, "", e))
                        continue
                    vf = _value_form(site, e, par)
                    if vf is not None and not (p is not None and p.kind == "member" and p.op == "." and p.kids[0] is e):
                        found.setdefault(vf[0], []).append((e.start, e.end, "", vf[1]))
        for text, occ in found.items():
            if len(occ) < 2:
                continue
            occ.sort(key=lambda o: o[0])
            outer = [o for o in occ if not any(q is not o and q[0] <= o[0] and o[1] <= q[1] for q in occ)]
            if len(outer) < 2:
                continue
            addr = text.startswith("&")
            v0 = next((o[3] for o in occ if not o[2]), None)
            if addr:                                       # the value is &E; E is what the uses read
                base = next(o[3] for o in occ if o[2]) if v0 is None else v0.kids[0]
                value = "&" + site.text(base)
            else:
                base, value = v0, site.text(v0)
            new_form = addr or base.kind not in ("call", "member", "index")
            calls = any(x.kind == "call" for x in cbody.expressions(base))
            spans = tuple((o[0], o[1]) for o in occ) if new_form and not calls else ()
            i, first = next((k, x) for k, x in enumerate(block.kids) if x.start <= occ[0][0] < x.end)
            if not site.in_block(first) or not _stable(site, base, first.start, occ[-1][1], spans):
                continue
            name = site.fresh("t")
            uses = [(a, b, name + rest) for a, b, rest, _v in outer]
            decl = _declare(site, first, name, value)
            if decl is not None:
                yield f"{_short(text)} named once as {name} ({len(outer)} uses)", decl + uses
            for j in _points(site, block, i, base, occ[-1][1], skip=spans):
                if j < i or not site.cplusplus:
                    where = f"{i - j} statements earlier" if j < i else "initialized at the top"
                    yield (f"{_short(text)} named once as {name} ({len(outer)} uses), {where}",
                           _declare_at(site, block, j, name, value) + uses)
            site.taken.discard(name)


# ---- local-reuse -------------------------------------------------------------------------------------

def _first_def(site: Site, s: Stmt, name: str) -> tuple[Stmt, Expr] | None:
    d = s.decl.declarators[0]
    if d.init is not None:
        return (s, d.init) if d.init_kind == "=" else None
    for x in cbody.statements(site.body):
        if x.start < s.end or x.kind != "expr":
            continue
        e = x.expr
        if e is not None and e.kind == "assign" and e.op == "=" and e.kids[0].kind == "name" and e.kids[0].name == name:
            if _ids(site, s.end, x.start, name):
                return None
            return x, e.kids[1]
    return None


def r_local_reuse(site: Site):
    """`int sh = v << 16;` vs `v = v << 16;`: a local written as a parameter (or an earlier local) of the
    same type whose old value is dead from there on; and a reassigned variable split into a new local."""
    from .inlinehelper import parameters, span_of
    span = span_of(site)
    params = (parameters(site.src, span) or []) if span is not None else []
    decls = _single_decls(site)
    body = (site.body.start, site.body.end)
    pool = [(p.name, _type_key(p.decl[:p.name_at]), None) for p in params]
    for n, s in decls.items():
        d = s.decl.declarators[0]
        pool.append((n, _type_key(site.src[s.decl.type_start:s.decl.type_end] + d.ptr), s))
    for name, s in decls.items():
        d = s.decl.declarators[0]
        if d.array or "&" in d.ptr:
            continue
        got = _first_def(site, s, name)
        if got is None:
            continue
        def_stmt, rhs = got
        if not (site.near(s) or site.near(def_stmt)) or (def_stmt is s and not site.cplusplus):
            continue
        ltype = _type_key(site.src[s.decl.type_start:s.decl.type_end] + d.ptr)
        loop = _in_loop(site, def_stmt)
        for pname, ptype, pdecl in pool:
            if pname == name or ptype != ltype or site.text(rhs).strip() == pname:
                continue
            if pdecl is not None and (pdecl.start >= s.start or not _encloses(site, pdecl, def_stmt)):
                continue
            pm = _ids(site, *body, pname)
            if any(m >= rhs.end for m in pm) or re.search(rf"&\s*{re.escape(pname)}\b", site.src[body[0]:body[1]]):
                continue
            if loop is not None and any(loop.start <= m < loop.end for m in pm):
                continue
            if def_stmt is s:
                block, i = site.parents[id(s)]
                edits = [(s.start, s.end, f"{pname} = {site.text(rhs)};")]
            else:
                block, i = site.parents[id(s)]
                lhs = def_stmt.expr.kids[0]
                edits = [(s.start, _next_start(block, i, s), ""), (lhs.start, lhs.end, pname)]
            edits += [(m, m + len(name), pname) for m in _ids(site, rhs.end, body[1], name)]
            yield f"{name} written as {pname} reused", edits
    for s in stmts_of(site, "expr"):
        e = s.expr
        if not site.cplusplus or not site.in_block(s) or e is None or e.kind != "assign" or e.op != "=" \
                or e.kids[0].kind != "name" or _in_loop(site, s) is not None:
            continue
        p = e.kids[0].name
        if p not in {x[0] for x in pool} or re.search(rf"&\s*{re.escape(p)}\b", site.src[body[0]:body[1]]):
            continue
        block, _i = site.parents[id(s)]
        later = _ids(site, s.end, body[1], p)
        if not later or any(m >= block.end for m in later):
            continue
        name = site.fresh("t")
        yield (f"{p} reassigned -> new local {name}",
               [(s.start, s.end, f"__typeof__({p}) {name} = {site.text(e.kids[1])};")] + [(m, m + len(p), name) for m in later])
        site.taken.discard(name)


def _encloses(site: Site, outer: Stmt, inner: Stmt) -> bool:
    p = site.parents.get(id(outer))
    return p is not None and p[0].start <= inner.start and inner.end <= p[0].end


# ---- default-else ------------------------------------------------------------------------------------

def r_default_else(site: Site):
    """`float s = 1.0f; ... if (t < 1.0f) s = t;` vs `float s; if (t < 1.0f) s = t; else s = 1.0f;`
    (the declaration where it was, or just before the `if`), vs `float s = t < 1.0f ? t : 1.0f;`."""
    for s in stmts_of(site, "decl", "expr"):
        if not site.in_block(s):
            continue
        if s.kind == "decl":
            if len(s.decl.declarators) != 1:
                continue
            d = s.decl.declarators[0]
            if d.init_kind != "=" or "&" in d.ptr or d.array or not _literal(site, d.init):
                continue
            x, k_text = d.name, site.text(d.init)
        else:
            e = s.expr
            if e is None or e.kind != "assign" or e.op != "=" or e.kids[0].kind != "name" or not _literal(site, e.kids[1]):
                continue
            x, k_text = e.kids[0].name, site.text(e.kids[1])
        block, i = site.parents[id(s)]
        mentions = _ids(site, s.end, block.end, x)
        if not mentions:
            continue
        t = next((y for y in cbody.statements(block) if y.kind == "if" and len(y.kids) == 1 and y.start > s.end
                  and single_assign(y.kids[0]) is not None and single_assign(y.kids[0]).kids[0].kind == "name"
                  and single_assign(y.kids[0]).kids[0].start == mentions[0]), None)
        if t is None or not site.in_block(t):
            continue
        tb, _ti = site.parents[id(t)]
        if any(not (t.end <= m < tb.end) for m in mentions[1:]):
            continue
        a = single_assign(t.kids[0])
        tind = cbody.indent_at(site.src, t.start)
        new_if = (f"if ({site.text(t.expr)}) {_as_block(site, t, t.kids[0])} else "
                  + braced(site, t, [f"{x} = {k_text};"]))
        if s.kind == "decl":
            d = s.decl.declarators[0]
            yield f"{x}'s default {k_text} moved into an else", [(d.name_end, d.end, ""), (t.start, t.end, new_if)]
            if site.cplusplus:
                head = site.src[s.decl.type_start:d.name_end] + ";"
                yield (f"{x} declared at the if, default {k_text} in its else",
                       [(s.start, _next_start(block, i, s), ""), (t.start, t.end, f"{head}\n{tind}{new_if}")])
        else:
            yield f"{x}'s default {k_text} moved into an else", [(s.start, _next_start(block, i, s), ""), (t.start, t.end, new_if)]
        if i + 1 < len(block.kids) and block.kids[i + 1] is t and not _ids(site, a.kids[1].start, a.kids[1].end, x):
            val = f"{site.text(t.expr)} ? {site.text(a.kids[1])} : {k_text}"
            if s.kind == "decl":
                d = s.decl.declarators[0]
                yield f"{x} = c ? a : {k_text}", [(d.init.start, d.init.end, val), (s.end, t.end, "")]
            else:
                yield f"{x} = c ? a : {k_text}", [(s.start, t.end, f"{x} = {val};")]
    for s in stmts_of(site, "if"):                      # the reverse: default first, overridden by the if
        if len(s.kids) != 2 or not site.in_block(s):
            continue
        a, b = single_assign(s.kids[0]), single_assign(s.kids[1])
        if a is None or b is None or a.kids[0].kind != "name" or site.text(a.kids[0]) != site.text(b.kids[0]) \
                or not _literal(site, b.kids[1]):
            continue
        x, k_text = site.text(a.kids[0]), site.text(b.kids[1])
        block, i = site.parents[id(s)]
        ind = cbody.indent_at(site.src, s.start)
        new_if = f"if ({site.text(s.expr)}) {_as_block(site, s, s.kids[0])}"
        prev = block.kids[i - 1] if i > 0 else None
        if prev is not None and prev.kind == "decl" and len(prev.decl.declarators) == 1 \
                and prev.decl.declarators[0].name == x and prev.decl.declarators[0].init is None:
            pd = prev.decl.declarators[0]
            yield f"{x} = {k_text} by default, then the if", [(pd.name_end, pd.end, f" = {k_text}"), (s.start, s.end, new_if)]
        yield f"{x} = {k_text}; before the if", [(s.start, s.end, f"{x} = {k_text};\n{ind}{new_if}")]
    for s in stmts_of(site, "decl"):
        if len(s.decl.declarators) != 1 or not site.in_block(s):
            continue
        d = s.decl.declarators[0]
        e = d.init
        if e is None or e.kind != "ternary" or d.init_kind != "=" or "&" in d.ptr:
            continue
        c, a, b = e.kids
        if _ids(site, e.start, e.end, d.name):
            continue
        ind = cbody.indent_at(site.src, s.start)
        for lit, other, cond in ((b, a, site.text(c)), (a, b, negate(site, c))):
            if _literal(site, lit):
                yield (f"{d.name} = {site.text(lit)} by default, then if",
                       [(e.start, e.end, site.text(lit)),
                        (s.end, s.end, f"\n{ind}if ({cond}) " + braced(site, s, [f"{d.name} = {site.text(other)};"]))])


# ---- unsigned-compare --------------------------------------------------------------------------------

ESC = {"n": 10, "t": 9, "r": 13, "0": 0, "\\": 92, "'": 39, '"': 34, "a": 7, "b": 8, "f": 12, "v": 11}
COMPARE_OPS = set(MIRROR)


def _char_value(t: str) -> int | None:
    body = t[1:-1]
    if len(body) == 1:
        return ord(body)
    if body.startswith("\\x"):
        return int(body[2:], 16)
    if body.startswith("\\") and len(body) == 2:
        return ESC.get(body[1])
    return None


def r_unsigned_compare(site: Site):
    """`lbz; cmplwi` vs `cmpwi`, `bge` vs `blt` on a counter: an unsigned literal (`c == 10u`), or the loop
    counter declared unsigned; and back."""
    for e in exprs_of(site):
        if e.kind != "binary" or e.op not in COMPARE_OPS:
            continue
        for lit in e.kids:
            t = site.text(lit)
            if lit.kind == "chr":
                v = _char_value(t)
                if v is not None:
                    yield f"{t} -> {v}u in {_short(site.text(e))}", [(lit.start, lit.end, f"{v}u")]
            elif lit.kind == "num" and re.fullmatch(r"(?:0x[0-9A-Fa-f]+|\d+)[uUlL]*", t):
                new = t.rstrip("uUlL") if "u" in t.lower() else t + "u"
                yield f"{t} -> {new} in {_short(site.text(e))}", [(lit.start, lit.end, new)]
    decls = _single_decls(site)
    for s in stmts_of(site, "for", "while"):
        names = []
        if s.kind == "for" and s.init is not None and s.init.kind == "expr" and s.init.expr.kind == "assign" \
                and s.init.expr.kids[0].kind == "name":
            names.append(s.init.expr.kids[0].name)
        cond = s.cond if s.kind == "for" else s.expr
        if cond is not None and cond.kind == "binary" and cond.op in COMPARE_OPS:
            names += [k.name for k in cond.kids if k.kind == "name"]
        for n in dict.fromkeys(names):
            x = decls.get(n)
            if x is None:
                continue
            typ = x.decl.base_type(site.src)
            if not INT_TYPES.match(typ) or x.decl.declarators[0].ptr:
                continue
            other = "int" if typ.startswith(("unsigned", "u")) else "unsigned int"
            yield f"{n} declared {other}", [(x.decl.type_start, x.decl.type_end, other)]


# ---- accessor ----------------------------------------------------------------------------------------

ACCESSOR = re.compile(r"(?:(?P<ret>[*&])|\b(?!return\b|else\b)[A-Za-z_]\w*\s)\s*(?:\w+::)*(?P<name>\w+)\s*\(\s*"
                      r"(?:(?:unsigned|signed|const)\s+)*(?:int|long|short|char|s32|u32|s16|u16|s8|u8|size_t|unsigned)\s+"
                      r"(?P<i>\w+)\s*\)\s*(?:const\s*)?\{\s*return\s+(?P<cast>\(\s*[\w:]+\s*[*&]?\s*\)\s*)?(?P<amp>&\s*)?"
                      r"(?:this\s*->\s*)?(?P<member>m\w+|_\w+)\s*\[\s*(?P=i)\s*\]\s*;\s*\}")
_CLASS_OPEN = re.compile(r"\b(?:struct|class)\s+(\w+)[^;{}()]*\{")
_FIELD = re.compile(r"^\s*(?:(?:const|struct|class|unsigned|signed|volatile)\s+)*([A-Za-z_][\w:]*)\s*([*&]*)\s*(\w+)\s*"
                    r"(?:\[[^\]]*\]\s*)*;", re.M)
_GLOBAL = re.compile(r"^(?:extern\s+(?:\"C\"\s+)?)?(?:(?:const|struct|class|volatile)\s+)*([A-Za-z_][\w:]*)\s*\*+\s*"
                     r"(\w+)\s*(?:\[[^\]]*\]\s*)*;", re.M)
_ACC_CACHE: dict[str, "Accessors"] = {}


class Accessors(dict):
    """{array member: [(accessor method, "ptr" | "ref" | "value", class)]}, plus what binds a site to the
    class: `fields` {(class, member): {type}} and `globals` {name: {type}} (bare class names)."""

    def __init__(self, *a):
        super().__init__(*a)
        self.fields: dict[tuple[str, str], set[str]] = {}
        self.globals: dict[str, set[str]] = {}


def bare(type_text: str) -> str:
    """`const Ns::Foo*` -> Foo."""
    t = re.sub(r"\b(?:const|volatile|struct|class)\b|[*&]", " ", type_text).split()
    return t[-1].split("::")[-1] if t else ""


def _class_spans(text: str) -> list[tuple[int, int, str]]:
    out = []
    for m in _CLASS_OPEN.finditer(text):
        close = cbody.matching_brace(text, m.end() - 1)
        if close is not None:
            out.append((m.end(), close, m.group(1)))
    return out


def accessors(texts) -> Accessors:
    """The sibling indexing accessors of inline definitions, each with its class: `T* Get(int i) { return
    &mArr[i]; }` (ptr), `T& At(int i) { return mArr[i]; }` (ref), `T* Get(int i) { return mPtrs[i]; }` and
    `float Len(int i) { return mLen[i]; }` (value). One whose result is cast is a ptr accessor only for member
    access through it (`o->Get(i)->f` vs `o->m[i].f`); outside a class body none (its class is unknown)."""
    out = Accessors()
    for text in texts:
        spans = _class_spans(text)
        for st, en, cls in spans:
            for m in _FIELD.finditer(text, st, en):
                if m.group(1) not in ("return", "typedef", "using", "friend", "virtual", "static"):
                    out.fields.setdefault((cls, m.group(3)), set()).add(bare(m.group(1)))
        for m in _GLOBAL.finditer(text):
            if not any(st <= m.start() < en for st, en, _ in spans):
                out.globals.setdefault(m.group(2), set()).add(bare(m.group(1)))
        for m in ACCESSOR.finditer(text):
            inner = [c for st, en, c in spans if st <= m.start() < en]
            if not inner:
                continue
            ret, amp = m.group("ret"), bool(m.group("amp"))
            kind = "ptr" if ret == "*" and amp else "ref" if ret == "&" and not amp else "value" if not amp else None
            if kind is None or (m.group("cast") and kind != "ptr"):
                continue
            kind += "-cast" if m.group("cast") else ""
            x = (m.group("name"), kind, inner[-1])
            if x not in out.get(m.group("member"), []):
                out.setdefault(m.group("member"), []).append(x)
    return out


def project_accessors(root: Path, unit_text: str = "") -> Accessors:
    """The accessors defined in the project's headers (cached per include/ tree) plus the unit's own."""
    key = str(Path(root).resolve() / "include")
    if key not in _ACC_CACHE:
        texts = []
        for p in sorted(Path(key).rglob("*")):
            if p.suffix in (".h", ".hpp", ".inc") and p.is_file():
                try:
                    texts.append(p.read_text(errors="replace"))
                except OSError:
                    pass
        _ACC_CACHE[key] = accessors(texts)
    base = _ACC_CACHE[key]
    own = accessors([unit_text]) if unit_text else Accessors()
    merged = Accessors({k: list(v) for k, v in base.items()})
    for k, v in own.items():
        merged.setdefault(k, []).extend(x for x in v if x not in merged.get(k, []))
    for src_, dst in ((base.fields, merged.fields), (own.fields, merged.fields),
                      (base.globals, merged.globals), (own.globals, merged.globals)):
        for k, v in src_.items():
            dst.setdefault(k, set()).update(v)
    return merged


def _classes_of(site: Site, e: Expr, acc: Accessors, depth: int = 0) -> set[str] | None:
    """The classes the object expression e may have (its declarations in the function, the file and the
    headers), None when unknown."""
    if depth > 6:
        return None
    if e.kind == "paren":
        return _classes_of(site, e.kids[0], acc, depth + 1)
    if e.kind == "cast" and e.type_span:
        t = site.src[e.type_span[0]:e.type_span[1]]
        return {bare(t)} if "*" in t else None
    if e.kind == "name":
        lo = max(0, site.src.rfind("\n}", 0, site.body.start))             # the definition's head and body
        head = site.src[lo:site.body.start]
        if e.name == "this":
            m = re.search(r"(\w+)\s*::\s*~?\w+\s*\(", head)
            return {m.group(1)} if m else None
        decl = re.compile(rf"(?:^|[(,;{{}}]\s*|\n\s*)(?:(?:const|struct|class|volatile)\s+)*([A-Za-z_][\w:]*)\s*"
                          rf"(?:const\s*)?\*+\s*(?:const\s+)?{re.escape(e.name)}\s*[=;,)\[]")
        got = {bare(m.group(1)) for m in decl.finditer(site.src, lo, site.body.end)
               if m.group(1) not in ("return", "else", "case", "sizeof")}
        return got or acc.globals.get(e.name) or None
    if e.kind == "member":
        base = _classes_of(site, e.kids[0], acc, depth + 1)
        if base is None:
            return None
        got = set().union(*(acc.fields.get((c, e.name), set()) for c in base))
        return got or None
    return None


def _lvalue_use(par: dict, e: Expr) -> bool:
    p = par.get(id(e))
    return p is not None and (p.kind == "assign" and p.kids[0] is e or p.kind == "postfix"
                              or p.kind == "unary" and p.op in ("&", "++", "--"))


def r_accessor(site: Site):
    """`o->m4[i].f` vs `o->Get(i)->f` when o's class has `T* Get(int i) { return &m4[i]; }` (and the ref and
    value forms): the sibling accessor instead of the indexed member, and back. Only where the object's
    class is the accessor's (its declared type; a call of an accessor whose name no other class uses
    otherwise): a member of the same name in another class is not that array."""
    acc = getattr(site, "accessors", None)
    if not acc:
        return
    fields = getattr(acc, "fields", None)
    by_method: dict[str, list[tuple[str, str, str]]] = {}
    for member, ms in acc.items():
        for x in ms:
            by_method.setdefault(x[0], []).append((member,) + tuple(x[1:]))
    nodes = list(exprs_of(site, near=True))
    par = _parent_map(list(exprs_of(site, near=False)))

    def classes(obj):
        return _classes_of(site, obj, acc) if fields is not None else None
    for e in nodes:
        if e.kind == "index" and e.kids[0].kind == "member" and e.kids[0].name in acc and pure(e.kids[1]):
            base = e.kids[0]
            obj, arrow = site.text(base.kids[0]), base.op
            cls = classes(base.kids[0])
            p = par.get(id(e))
            for x in acc[base.name]:
                method, kind = x[0], x[1]
                if len(x) > 2 and (cls is None or x[2] not in cls):
                    continue
                call = f"{obj}{arrow}{method}({site.text(e.kids[1])})"
                if p is not None and p.kind == "member" and p.op == "." and p.kids[0] is e and kind != "value":
                    yield f"{_short(site.text(e))} -> {method}()", [(e.start, p.end, f"{call}{'.' if kind == 'ref' else '->'}{p.name}")]
                elif kind == "ptr" and p is not None and p.kind == "unary" and p.op == "&":
                    yield f"&{_short(site.text(e))} -> {method}()", [(p.start, p.end, call)]
                elif kind == "ref" or (kind == "value" and not _lvalue_use(par, e)):
                    yield f"{_short(site.text(e))} -> {method}()", [(e.start, e.end, call)]
        elif e.kind == "call" and len(e.kids) == 2 and e.kids[0].kind == "member" and e.kids[0].name in by_method:
            callee = e.kids[0]
            cls = classes(callee.kids[0])
            opts = by_method[callee.name]
            if cls is not None:
                opts = [x for x in opts if len(x) < 3 or x[2] in cls]
            elif len({x[2:] for x in opts}) > 1:
                continue
            p = par.get(id(e))
            for x in opts[:1]:
                member, kind = x[0], x[1]
                elem = f"{site.text(callee.kids[0])}{callee.op}{member}[{site.text(e.kids[1])}]"
                if kind.startswith("ptr") and p is not None and p.kind == "member" and p.op == "->" and p.kids[0] is e:
                    yield f"{callee.name}() -> {member}[]", [(e.start, p.end, f"{elem}.{p.name}")]
                elif kind == "ptr":
                    yield f"{callee.name}() -> &{member}[]", [(e.start, e.end, f"&{elem}")]
                elif kind in ("ref", "value"):
                    yield f"{callee.name}() -> {member}[]", [(e.start, e.end, elem)]


# ---- shared-labels -----------------------------------------------------------------------------------

def _segments(body: Stmt) -> list[tuple[list[Stmt], list[Stmt]]] | None:
    segs: list[tuple[list[Stmt], list[Stmt]]] = []
    for k in body.kids:
        if k.kind in ("case", "default"):
            if segs and not segs[-1][1]:
                segs[-1][0].append(k)
            else:
                segs.append(([k], []))
        elif not segs:
            return None
        else:
            segs[-1][1].append(k)
    return segs


def r_shared_labels(site: Site):
    """Two cases with the same body vs one body under both labels (`case 1: case 2: f(); break;`). Only the
    labels the switch has: never a new case value."""
    for s in stmts_of(site, "switch"):
        body = s.kids[0] if s.kids else None
        segs = _segments(body) if body is not None and body.kind == "block" else None
        if not segs:
            continue
        norm = lambda ks: " ".join(" ".join(site.text(k).split()) for k in ks)
        for i in range(len(segs) - 1):
            (la, sa), (lb, sb) = segs[i], segs[i + 1]
            if sa and sb and norm(sa) == norm(sb) and sa[-1].kind in ("break", "return") \
                    and not any(k.kind == "decl" for k in sa):
                lind = cbody.indent_at(site.src, la[0].start)
                yield (f"{' '.join(site.text(x) for x in la)} shares the next case's body",
                       [(la[-1].end, lb[0].start, f"\n{lind}")])
        for labels, stmts in segs:
            if len(labels) < 2 or not stmts or stmts[-1].kind not in ("break", "return") \
                    or any(x.kind == "default" for x in labels) or any(k.kind == "decl" for k in stmts):
                continue
            sind = cbody.indent_at(site.src, stmts[0].start)
            copy = f"\n{sind}".join(_reindent(site, k, sind) for k in stmts)
            yield (f"{site.text(labels[0])} given its own copy of the body",
                   [(labels[0].end, labels[0].end, f"\n{sind}{copy}")])


# ---- cached-chain ------------------------------------------------------------------------------------

CHAIN_KINDS = ("member", "name", "index", "paren")


def _chain(e: Expr | None) -> bool:
    """A pure load path: names, members and constant-free subscripts, no calls."""
    if e is None or not pure(e):
        return False
    return all(x.kind in CHAIN_KINDS or (x.kind == "num") for x in cbody.expressions(e)) and e.kind in ("member", "index")


def _same_exprs(site: Site, s: Stmt, text: str) -> list[Expr]:
    """The outermost expressions of statement s whose text is `text` (whitespace folded)."""
    want = " ".join(text.split())
    out: list[Expr] = []
    for top in cbody.statement_exprs(s):
        stack = [top]
        while stack:
            x = stack.pop()
            if " ".join(site.text(x).split()) == want:
                out.append(x)
            else:
                stack.extend(x.kids)
    return out


def r_cached_chain(site: Site):
    """`T* s = g->a; if (s->f == 0) ..` vs `if (g->a->f == 0) ..; T* s = g->a;` (the pointer is CSE'd,
    the field is read again: the target's `lwz` reload), and the reverse."""
    for s in stmts_of(site, "decl"):
        if not site.in_block(s) or len(s.decl.declarators) != 1:
            continue
        d = s.decl.declarators[0]
        if d.init_kind != "=" or "*" not in d.ptr or d.array or not _chain(d.init):
            continue
        block, i = site.parents[id(s)]
        e_text = site.text(d.init)
        decl_text = site.src[s.start:s.end]
        # forward: the first later statement that reads s (declarations and other statements that do not
        # mention it may sit between); write it through the chain, declare s right below it
        u = next((y for y in block.kids[i + 1:] if _ids(site, y.start, y.end, d.name)), None)
        if u is not None and u.kind in ("if", "expr") and site.in_block(u) \
                and all(y.kind in ("decl", "expr") for y in block.kids[i + 1:block.kids.index(u)]):
            uses = _ids(site, u.start, u.end, d.name)
            later = _ids(site, u.end, block.end, d.name)
            if later and d.name not in _writes(site, u) and _stable(site, d.init, s.end, u.end) \
                    and not re.search(rf"&\s*{re.escape(d.name)}\b", site.src[u.start:u.end]):
                ind = cbody.indent_at(site.src, u.start)
                edits = [(s.start, _next_start(block, i, s), "")]
                edits += [(m, m + len(d.name), e_text) for m in uses]
                edits.append((u.end, u.end, f"\n{ind}{decl_text}"))
                yield f"{d.name} taken after its first use: {_short(e_text)} read through the chain", edits
        # reverse: the nearest earlier statement that reads the chain (only declarations and expression
        # statements that do not read it between); cache it in s declared above that statement
        pv = next((y for y in reversed(block.kids[:i]) if _same_exprs(site, y, e_text)), None)
        if pv is not None and pv.kind in ("if", "expr") and site.in_block(pv) and not _ids(site, pv.start, s.start, d.name) \
                and all(y.kind in ("decl", "expr") for y in block.kids[block.kids.index(pv) + 1:i]):
            hits = _same_exprs(site, pv, e_text)
            if True:
                if hits and _stable(site, d.init, pv.start, s.start):
                    ind = cbody.indent_at(site.src, pv.start)
                    edits = [(pv.start, pv.start, f"{decl_text}\n{ind}")]
                    edits += [(h.start, h.end, d.name) for h in hits]
                    edits.append((s.start, _next_start(block, i, s), ""))
                    yield f"{d.name} declared before the statement that reads {_short(e_text)}", edits


# ---- bool-index --------------------------------------------------------------------------------------

def _compare(e: Expr | None) -> Expr | None:
    """The comparison inside parens, or None."""
    while e is not None and e.kind == "paren":
        e = e.kids[0]
    return e if e is not None and e.kind == "binary" and e.op in MIRROR else None


def r_bool_index(site: Site):
    """`p[(int)(a >= 2)]` (a bool zero-extended: `clrlslwi`) vs `p[a >= 2 ? 1 : 0]` (`slwi`), both ways."""
    nodes = list(exprs_of(site))
    par = _parent_map(nodes)
    for x in nodes:
        parent = par.get(id(x))
        bare_ok = parent is not None and parent.kind == "index" and parent.kids[1] is x
        if x.kind == "cast" and x.type_span and INT_TYPES.match(site.src[x.type_span[0]:x.type_span[1]].strip()):
            c = _compare(x.kids[0])
            if c is None:
                continue
            new = f"{site.text(c)} ? 1 : 0"
            yield f"{_short(site.text(x))} as a select", [(x.start, x.end, new if bare_ok else f"({new})")]
        elif x.kind == "ternary" and site.text(x.kids[1]).strip() == "1" and site.text(x.kids[2]).strip() == "0":
            c = _compare(x.kids[0])
            if c is None:
                continue
            yield f"{_short(site.text(x))} as a bool cast", [(x.start, x.end, f"(int)({site.text(c)})")]


RULES: list[Rule] = [
    Rule("else-return", "A branch around a dead `else` (`b` after a jump) or a jump placed as the other branch",
         "drop the `else` after a then-block that returns, or put the jump in the `else`; no empty then-blocks",
         ("branch", "count", "order"), r_else_return),
    Rule("name-value", "A value computed into another register or at another point than the target (call result, load)",
         "name the call or load in a local just before its statement (or before the declarations ahead of it); "
         "or fold a single-use local into its use", ("regs", "order", "count"), r_name_value),
    Rule("inline-local", "A pointer kept in a saved register where the target recomputes it (or the reverse)",
         "write a local that only holds a value out at every use (`f(x)->g()` twice); or name a repeated value once",
         ("regs", "count", "order"), r_inline_local),
    Rule("local-reuse", "One local more (or less) than the target: a dead parameter's register reused",
         "write the value into the dead parameter (`v = v << 16;`) instead of a new local, or the reverse",
         ("regs", "stack"), r_local_reuse),
    Rule("default-else", "`li` before a branch vs a value set on both arms (`b` over an `else`)",
         "the default value in an `else` (declared at the `if`), or `x = c ? a : K`, or the default first",
         ("branch", "count", "zero", "regs"), r_default_else),
    Rule("unsigned-compare", "`cmplwi` vs `cmpwi` against a literal; `blt` vs `bge`/`ble` on a loop counter",
         "an unsigned literal (`c == 10u`), or the counter declared `unsigned int`", ("signed", "branch"),
         r_unsigned_compare),
    Rule("accessor", "`mulli`/`add` for an element where the target calls (or inlines) the class's accessor",
         "the sibling accessor `o->Get(i)->f` instead of `o->m[i].f`, or back", ("index", "count", "regs"), r_accessor),
    Rule("shared-labels", "A jump table or compare chain with two targets where the target has one (or the reverse)",
         "two cases with the same body share it (`case 1: case 2:`); never a new case value",
         ("branch", "count"), r_shared_labels),
    Rule("cached-chain", "The target reloads a field (`lwz rX, off(rP)` again) that ours reads once from a cached pointer",
         "read the first use through the pointer chain (`g->a->f`) and take the local `s = g->a` after it; "
         "the pointer is merged, the field is not; or cache the chain above the statement", ("count", "regs"),
         r_cached_chain),
    Rule("bool-index", "`clrlslwi rX, rX, 24, K` (a bool zero-extended before a shift) where the target has `slwi`",
         "`a >= b ? 1 : 0` instead of `(int)(a >= b)` as an index or integer, or the reverse", ("count",),
         r_bool_index),
]
