"""Scope rows of the rewrite table (T49 card 1, a gauge vf23; T52). Each is a forward move and its inverse, so
`dk dataset stacked` makes synthetic before-versions with the same code.

- scope-local: a function-level local that only the two branches of one if/else use, each branch starting its
  use with a plain assignment `x = E;`, declared instead in each branch (`T x = E;`): two locals, coloured in the
  local tier; and the reverse (two same-named, same-typed block locals of sibling branches hoisted to one
  declaration before the if, or at the top of the block's declarations).
- shared-tail: the same statement at the end of both branches of an if/else, written once after it (the
  compiler never tail-merges, so the `b` to the join moves by the statement); and the reverse (a statement
  after an if/else pushed to the end of both branches).

Legality is checked from the parse alone (cbody): scalars and pointers only (no const, static, reference,
array or class type), each local declared once in the function, the name not a parameter, no mention of the
local outside the if/else, no label or case where a statement moves, a pushed statement names nothing a branch
declares, and no branch ending in a jump when a statement is pushed into it. Nothing here compiles anything.
"""

from __future__ import annotations

from ...core import cbody
from ...core.cbody import Stmt
from .loopmoves import _decl_counts, _mentions, _scalar_decl
from .rewrites import Rule, Site, _ends_in_jump, _short, stmts_of


def _norm(t: str) -> str:
    return " ".join(t.split())


def _in_head(site: Site, name: str) -> bool:
    """The name is written in the function head (a parameter)."""
    b = site.body.start
    lo = max(site.src.rfind("}", 0, b), site.src.rfind(";", 0, b)) + 1
    return _mentions(site, lo, b, name)


def _two_blocks(s: Stmt) -> bool:
    return s.kind == "if" and len(s.kids) == 2 and all(k.kind == "block" for k in s.kids)


def _branch_decl_names(branches) -> set[str]:
    return {d.name for br in branches for s in br.kids if s.kind == "decl" and s.decl is not None for d in s.decl.declarators}


# ---- scope-local -----------------------------------------------------------------------------------

def _first_assign(site: Site, br: Stmt, x: str):
    """The branch's first statement mentioning x, when it is a plain `x = E;` kid of the branch (E without x)."""
    for s in br.kids:
        if not _mentions(site, s.start, s.end, x):
            continue
        if s.kind != "expr" or s.expr is None or s.expr.kind != "assign" or s.expr.op != "=" or not site.in_block(s):
            return None
        lhs = s.expr.kids[0]
        if lhs.kind != "name" or lhs.name != x or x in cbody.names(s.expr.kids[1]):
            return None
        return s
    return None


def r_scope_local(site: Site):
    counts = _decl_counts(site)
    # forward: narrow a function-level local to the two branches that set it
    for d in stmts_of(site, "decl"):
        if not _scalar_decl(site, d) or d.decl.declarators[0].init is not None or not site.in_block(d):
            continue
        x = d.decl.declarators[0].name
        if counts[x] != 1 or _in_head(site, x):
            continue
        block, k = site.parents[id(d)]
        holders = [s for s in block.kids[k + 1:] if _mentions(site, s.start, s.end, x)]
        if len(holders) != 1 or not _two_blocks(holders[0]):
            continue
        s_if = holders[0]
        if _mentions(site, s_if.start, s_if.kids[0].start, x):
            continue
        firsts = [_first_assign(site, br, x) for br in s_if.kids]
        if any(f is None for f in firsts):
            continue
        head = site.src[d.decl.type_start:d.decl.declarators[0].name_end]
        nxt = block.kids[k + 1].start
        edits = [(d.start, nxt, "")] + [(f.start, f.expr.kids[0].end, head) for f in firsts]
        yield f"{x} declared in each branch that sets it", edits
    # inverse: hoist two same-named block locals of sibling branches to one declaration before the if
    for s_if in stmts_of(site, "if"):
        if not _two_blocks(s_if) or not site.in_block(s_if):
            continue
        a, b = s_if.kids
        block, k = site.parents[id(s_if)]
        for da in a.kids:
            if da.kind != "decl" or not _scalar_decl(site, da) or da.decl.declarators[0].init_kind != "=":
                continue
            x = da.decl.declarators[0].name
            db = next((s for s in b.kids if s.kind == "decl" and _scalar_decl(site, s) and s.decl.declarators[0].name == x
                       and s.decl.declarators[0].init_kind == "="), None)
            if db is None or counts[x] != 2 or _in_head(site, x):
                continue
            head = site.src[da.decl.type_start:da.decl.declarators[0].name_end]
            if _norm(head) != _norm(site.src[db.decl.type_start:db.decl.declarators[0].name_end]):
                continue
            if _mentions(site, site.body.start, s_if.start, x) or _mentions(site, s_if.end, site.body.end, x):
                continue
            if any(_mentions(site, br.start, d.start, x) or x in cbody.names(d.decl.declarators[0].init)
                   for br, d in ((a, da), (b, db))):
                continue
            strip = [(d.decl.type_start, d.decl.declarators[0].name_start, "") for d in (da, db)]
            places = [s_if.start]
            if site.cplusplus:
                top = next((s.start for s in block.kids if s.kind != "decl"), s_if.start)
                if top != s_if.start:
                    places.append(top)
            else:
                places = [next((s.start for s in block.kids if s.kind != "decl"), s_if.start)]
            for at in places:
                where = "before the if" if at == s_if.start else "with the block's declarations"
                yield f"{x} declared once {where}", strip + [(at, at, f"{head};\n{cbody.indent_at(site.src, at)}")]


# ---- shared-tail -----------------------------------------------------------------------------------

def r_shared_tail(site: Site):
    for s_if in stmts_of(site, "if"):
        if not _two_blocks(s_if) or not site.in_block(s_if):
            continue
        a, b = s_if.kids
        block, k = site.parents[id(s_if)]
        decl_names = _branch_decl_names((a, b))
        # forward: the same last statement of both branches, once after the if/else
        if len(a.kids) >= 2 and len(b.kids) >= 2:
            ta, tb = a.kids[-1], b.kids[-1]
            if (ta.kind == "expr" and tb.kind == "expr" and _norm(site.text(ta)) == _norm(site.text(tb))
                    and site.in_block(ta) and site.in_block(tb) and not (set(cbody.names(ta.expr)) & decl_names)):
                ind = cbody.indent_at(site.src, s_if.start)
                yield (f"{_short(site.text(ta))} once after the if/else",
                       [(a.kids[-2].end, ta.end, ""), (b.kids[-2].end, tb.end, ""), (s_if.end, s_if.end, f"\n{ind}{site.text(ta)}")])
        # inverse: the statement after the if/else at the end of both branches
        if k + 1 < len(block.kids):
            t = block.kids[k + 1]
            if (t.kind == "expr" and t.expr is not None and a.kids and b.kids and site.in_block(t)
                    and not _ends_in_jump(a) and not _ends_in_jump(b) and not (set(cbody.names(t.expr)) & decl_names)):
                edits = [(s_if.end, t.end, "")]
                for br in (a, b):
                    last = br.kids[-1]
                    edits.append((last.end, last.end, f"\n{cbody.indent_at(site.src, last.start)}{site.text(t)}"))
                yield f"{_short(site.text(t))} at the end of both branches", edits


RULES: list[Rule] = [
    Rule("scope-local", "Register-only rows: a local's value in another register in one branch (its live range split)",
         "declare the local in each branch that sets it (`T x = e;` twice), not once at function level; or the reverse",
         ("regs",), r_scope_local),
    Rule("shared-tail", "One unconditional `b` whose target is inside the other branch, off by that branch's last statement",
         "the statement both branches end with written once after the if/else; or pushed into both",
         ("branch",), r_shared_tail),
]
