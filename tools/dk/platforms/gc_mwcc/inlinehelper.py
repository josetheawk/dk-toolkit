"""The inline-helper rewrite: a loop of the function (with the declarations only it uses, or the
`if` that guards it) moved into a `static inline` helper defined just before the function, and
called where it was.

Why: inside an inlined helper the compiler ranks the helper's locals and temporaries below the
caller's parameters, where the same loop written in the caller ranks its counter as a named
local, above them (playbook, gamecube-mwcc.md "Register allocation"). The rank solver flags the
values the target ranks below every parameter (`ranksolve.below_parameters`); `dk close` passes
their names here as `site.prefer`, and the loops that hold them are tried first.

The helper takes every variable of the function the block reads (by value), those it writes (by
reference in C++, through a pointer in C), and `this` as `self` when the block uses members (the
debug info's member list, `site.members`). Two parameter orders are tried: first use in the block
and declaration order. A search loop that ends in `x = v; break;` with `x` set just before it
becomes a helper that returns: `x = Helper(...)`. Nothing here compiles anything: the closer does,
and its judge keeps only an exact, faithful result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ...core import cbody
from ...core.cbody import Expr, Stmt
from .rewrites import Rule, Site

LOOPS = ("for", "while", "do")
SCALAR = re.compile(r"^(?:(?:const|volatile|signed|unsigned|short|long|int|char|float|double|bool|wchar_t|"
                    r"u8|u16|u32|u64|s8|s16|s32|s64|f32|f64|BOOL|size_t)\s*)+$")


@dataclass
class Var:
    name: str
    decl: str                  # its declaration as written (`const char* name`, `int buf[4]`)
    name_at: int               # offset of the name inside `decl`
    order: int                 # declaration order in the function (parameters first)
    param: bool = False

    @property
    def pointerish(self) -> bool:
        head = self.decl[:self.name_at]
        return "*" in head or "&" in head or "[" in self.decl[self.name_at:]

    def as_param(self, mode: str) -> str:
        """`decl` as a helper parameter: by value, by reference (`&`) or through a pointer (`*`)."""
        decl = re.sub(r"\b(?:static|register|extern|auto|mutable)\b\s*", "", self.decl)
        at = self.name_at - (len(self.decl) - len(decl))
        if mode == "value":
            return decl
        return decl[:at].rstrip() + ("& " if mode == "ref" else "* ") + decl[at:]


def span_of(site: Site) -> cbody.FunctionSpan | None:
    span = getattr(site, "span", None)
    if span is not None:
        return span
    hits = [f for f in cbody.functions(site.src) if f.open == site.body.start]
    return hits[0] if hits else None


def parameters(src: str, span: cbody.FunctionSpan) -> list[Var] | None:
    """The function's parameters from its head, each with its declaration as written."""
    toks = cbody.tokens(src, span.head_start, span.open)
    texts = [t.text for t in toks]
    lp = next((j for j, t in enumerate(texts) if t == "(" and "".join(texts[max(0, j - 8):j]).endswith(span.name)), None)
    if lp is None:
        return None
    depth, rp = 0, None
    for j in range(lp, len(toks)):
        depth += texts[j] == "("
        depth -= texts[j] == ")"
        if depth == 0:
            rp = j
            break
    if rp is None:
        return None
    args, cur, depth = [], [], 0
    for t in toks[lp + 1:rp]:
        if t.text in ("(", "[", "<"):
            depth += 1
        elif t.text in (")", "]", ">"):
            depth -= 1
        if t.text == "," and depth == 0:
            args.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        args.append(cur)
    out = []
    for k, a in enumerate(args):
        words = [t.text for t in a]
        if not a or words == ["void"] or "..." in words or "=" in words:
            continue
        if "(" in words:                                   # a function pointer `void (*cb)(int)`
            i = words.index("(")
            name = next((t for t in a[i + 1:] if t.kind == "id"), None)
        elif "[" in words:
            name = a[words.index("[") - 1]
        else:
            name = a[-1]
        if name is None or name.kind != "id" or name.text in cbody.TYPE_WORDS | cbody.QUALIFIERS or len(a) < 2:
            continue
        decl = src[a[0].start:a[-1].end]
        out.append(Var(name.text, decl, name.start - a[0].start, k, True))
    return out


def const_member(src: str, span: cbody.FunctionSpan) -> bool:
    head = src[span.head_start:span.open]
    close = head.rfind(")")
    return close >= 0 and re.search(r"\bconst\b", head[close:]) is not None


def owner_class(span: cbody.FunctionSpan) -> str | None:
    parts = span.name.split("::")
    return "::".join(parts[:-1]) if len(parts) > 1 else None


# ---- the block's variables -------------------------------------------------------------------------

def _decl_var(src: str, s: Stmt, d: cbody.Declarator, order: int) -> Var:
    if d is s.decl.declarators[0]:
        return Var(d.name, src[s.decl.type_start:d.name_end] + d.array, d.name_start - s.decl.type_start, order)
    typ = src[s.decl.type_start:s.decl.type_end].rstrip()
    declarator = src[d.start:d.name_end] + d.array
    return Var(d.name, f"{typ} {declarator}", len(typ) + 1 + (d.name_start - d.start), order)


def visible(site: Site, start: int, end: int, params: list[Var]) -> dict[str, Var]:
    """The function's variables in scope at [start, end): its parameters, then the locals declared
    before it in an enclosing block (or a `for` that encloses it), later ones shadowing earlier."""
    out = {v.name: v for v in params}
    order = len(params)
    for s in cbody.statements(site.body):
        if s.start >= start:
            break
        decls: list[Stmt] = []
        if s.kind == "decl" and s.decl is not None:
            p = site.parents.get(id(s))
            if p is not None and p[0].start <= start and end <= p[0].end:
                decls.append(s)
        if s.kind == "for" and s.init is not None and s.init.kind == "decl" and s.start <= start and end <= s.end:
            decls.append(s.init)
        for x in decls:
            for d in x.decl.declarators:
                out[d.name] = _decl_var(site.src, x, d, order)
                order += 1
    return out


def _root(e: Expr) -> tuple[Expr | None, bool]:
    """The variable an lvalue writes, and whether it writes it in place (the variable itself or a
    member of it through `.`), not through a pointer (`->`, an element)."""
    in_place = True
    while e.kind in ("member", "index", "paren"):
        if e.kind == "index" or (e.kind == "member" and e.op == "->"):
            in_place = False
        e = e.kids[0]
    return (e if e.kind == "name" else None), in_place


@dataclass
class Block:
    stmts: list[Stmt]              # the statements moved (contiguous in one block, or one statement)
    start: int
    end: int
    loop: Stmt                     # the loop it holds
    tail: bool = False             # the function's last statements: returns inside move along


def _exprs(stmts: list[Stmt]):
    for s in stmts:
        for x in cbody.statements(s):
            for top in cbody.statement_exprs(x):
                yield from cbody.expressions(top)


def blocks(site: Site) -> list[Block]:
    """Loops to move, each alone, with the declarations right before it that only it uses, and the
    `if` that holds it as its only statement."""
    out: list[Block] = []
    for loop in cbody.statements(site.body):
        if loop.kind not in LOOPS or not site.near(loop):
            continue
        bare = Block([loop], loop.start, loop.end, loop)
        p = site.parents.get(id(loop))
        if p is None:
            out.append(bare)
            continue
        holder, k = p
        if holder.kind == "block":
            j = k
            after = holder.kids[k + 1:]
            used_after = {n for x in after for top in _all(x) for n in cbody.names(top)}
            while j > 0 and holder.kids[j - 1].kind == "decl" and k - j < 3:
                prev = holder.kids[j - 1]
                names = {d.name for d in prev.decl.declarators}
                if names & used_after or "static" in site.src[prev.decl.type_start:prev.decl.type_end].split():
                    break
                j -= 1
            if j < k:                                  # the more natural helper first: it declares its own locals
                out.append(Block(holder.kids[j:k + 1], holder.kids[j].start, loop.end, loop))
            out.append(bare)
        else:
            out.append(bare)
        if holder.kind == "block":
            guard = site.parents.get(id(holder))
            if guard is not None and guard[0].kind == "if" and len(holder.kids) == 1:
                holder = guard[0]
            elif guard is not None:
                continue
        if holder.kind == "if" and holder.kids and (holder.kids[0] is loop or
                                                   (holder.kids[0].kind == "block" and len(holder.kids[0].kids) == 1)):
            if holder.end > holder.start:
                out.append(Block([holder], holder.start, holder.end, loop))
    for name in sorted(getattr(site, "prefer", ()) or ()):   # a flagged local's own statements (no loop needed)
        b = _local_run(site, name)
        if b is not None:
            out.append(b)
    kids = site.body.kids
    if kids:                                          # the function's tail: a loop (then its final return)
        last = kids[-1]
        if last.kind == "return" and len(kids) >= 2 and kids[-2].kind in LOOPS and site.near(kids[-2]):
            out.append(Block([kids[-2], last], kids[-2].start, last.end, kids[-2], True))
        elif last.kind in LOOPS and site.near(last) and return_type(site) == "void":
            out.append(Block([last], last.start, last.end, last, True))
    seen, uniq = set(), []
    for b in out:
        if (b.start, b.end, b.tail) not in seen and _movable_src(site, b.stmts, b.tail):
            seen.add((b.start, b.end, b.tail))
            uniq.append(b)
    return uniq


def _local_run(site: Site, name: str) -> Block | None:
    """The statements of the local's block from its declaration to its last use, when nothing they
    declare is used after them: a block the target compiled as an inline helper."""
    decl = next((s for s in cbody.statements(site.body) if s.kind == "decl" and s.decl is not None
                 and any(d.name == name for d in s.decl.declarators)), None)
    p = site.parents.get(id(decl)) if decl is not None else None
    if p is None or p[0].kind != "block":
        return None
    block, k = p
    uses = [i for i in range(k, len(block.kids)) if name in {n for top in _all(block.kids[i]) for n in cbody.names(top)}]
    m = max(uses, default=k)
    run = block.kids[k:m + 1]
    declared = {d.name for x in run for s in cbody.statements(x) if s.decl is not None for d in s.decl.declarators}
    after = {n for x in block.kids[m + 1:] for top in _all(x) for n in cbody.names(top)}
    if declared & after or (len(run) == 1 and run[0] is decl):
        return None
    return Block(run, run[0].start, run[-1].end, run[0])


def return_type(site: Site) -> str | None:
    """The function's return type as written (`void` for constructors and destructors)."""
    span = span_of(site)
    if span is None:
        return None
    head = site.src[span.head_start:span.open]
    m = None
    for m in re.finditer(re.escape(span.name) + r"\s*\(", head):
        break
    if m is None:
        return None
    words = re.sub(r'\bextern\s+"C"|\b(?:static|inline|virtual|explicit)\b', " ", head[:m.start()])
    return " ".join(words.split()) or "void"


def _all(s: Stmt):
    for x in cbody.statements(s):
        yield from cbody.statement_exprs(x)


def _movable_src(site: Site, stmts: list[Stmt], tail: bool = False) -> bool:
    """No statement that leaves the block (return, goto, a break or continue of an enclosing loop),
    is a target from outside it (labels, cases of an enclosing switch), or is a static local. At
    the function's tail a return may stay: it returns from the helper, whose result is returned."""
    for s in stmts:
        for x in cbody.statements(s):
            if x.kind in ("goto", "label", "pp") or (x.kind == "return" and not tail):
                return False
            inside = [y.kind for y in cbody.statements(s) if y.start < x.start < y.end]
            if x.kind in ("case", "default") and "switch" not in inside:
                return False
            if x.kind == "break" and not any(k in LOOPS + ("switch",) for k in inside):
                return False
            if x.kind == "continue" and not any(k in LOOPS for k in inside):
                return False
            if x.kind == "decl" and "static" in site.src[x.decl.type_start:x.decl.type_end].split():
                return False
    return True


# ---- the rewrite -------------------------------------------------------------------------------------

def _reindent(text: str, base: str) -> str:
    lines = text.split("\n")
    out = ["    " + lines[0]]
    for ln in lines[1:]:
        if not ln.strip():
            out.append("")
        else:
            out.append("    " + (ln[len(base):] if ln.startswith(base) else ln.lstrip()))
    return "\n".join(out)


def _insert_at(src: str, span: cbody.FunctionSpan) -> int:
    """Where a helper goes: before the function and the comment lines right above it."""
    at = src.rfind("\n", 0, span.head_start) + 1
    while at > 0:
        prev = src.rfind("\n", 0, at - 1) + 1
        if src[prev:at].strip().startswith("//"):
            at = prev
        else:
            break
    return at


def _returning(site: Site, b: Block, written: set[str]) -> tuple[str, Expr, list[tuple[Stmt, Stmt]]] | None:
    """A search loop: (x, its value before the loop, [(`x = v;`, its `break;`)]) when the block is
    `x = K;` (or `T x = K;`) then a loop whose only write is `x = v; break;`."""
    if len(b.stmts) != 1 or b.stmts[0] is not b.loop:
        return None
    p = site.parents.get(id(b.loop))
    if p is None or p[0].kind != "block" or p[1] == 0:
        return None
    init = p[0].kids[p[1] - 1]
    if init.kind == "decl" and len(init.decl.declarators) == 1 and init.decl.declarators[0].init_kind == "=":
        x, k = init.decl.declarators[0].name, init.decl.declarators[0].init
    elif init.kind == "expr" and init.expr.kind == "assign" and init.expr.op == "=" and init.expr.kids[0].kind == "name":
        x, k = init.expr.kids[0].name, init.expr.kids[1]
    else:
        return None
    if k.kind not in ("num", "name") or written != {x}:
        return None
    stores = []
    for s in cbody.statements(b.loop):
        if s.kind == "expr" and s.expr.kind == "assign" and s.expr.kids[0].kind == "name" and s.expr.kids[0].name == x:
            if s.expr.op != "=" or x in cbody.names(s.expr.kids[1]):
                return None
            q = site.parents.get(id(s))
            if q is None or q[0].kind != "block" or q[1] + 1 >= len(q[0].kids) or q[0].kids[q[1] + 1].kind != "break":
                return None
            brk = q[0].kids[q[1] + 1]
            if any(y.kind in LOOPS + ("switch",) and y is not b.loop and y.start < brk.start < y.end
                   for y in cbody.statements(b.loop)):
                return None
            stores.append((s, brk))
    reads = [e for e in _exprs([b.loop]) if e.kind == "name" and e.name == x]
    if not stores or len(reads) != len(stores):
        return None
    return x, k, stores


def extract(site: Site, b: Block, params: list[Var], members: set[str], cls: str | None, const: bool,
            name: str) -> list[tuple[str, list[tuple[int, int, str]]]]:
    """The candidates that move block b into a helper called `name`: (label, edits) per form and
    parameter order."""
    span = span_of(site)
    scope = visible(site, b.start, b.end, params)
    inner = {d.name for x in b.stmts for s in cbody.statements(x) if s.decl is not None for d in s.decl.declarators}
    edits: list[tuple[int, int, str]] = []
    used: list[str] = []
    written: set[str] = set()
    self_used = False
    for e in _exprs(b.stmts):
        if e.kind == "name":
            if e.name == "this":
                self_used = True
                edits.append((e.start, e.end, "self"))
            elif e.name in inner:
                if e.name in scope and e.name not in used:
                    return []                    # a name both declared inside and outside: shadowing
            elif e.name in scope:
                if e.name not in used:
                    used.append(e.name)
            elif e.name in members and cls is not None:
                self_used = True
                edits.append((e.start, e.start, "self->"))
        if e.kind in ("assign", "postfix") or (e.kind == "unary" and e.op in ("++", "--", "&")):
            r, in_place = _root(e.kids[0])
            if r is not None and r.name in scope and r.name not in inner and in_place:
                if "[" in scope[r.name].decl[scope[r.name].name_at:]:
                    return []                    # an array itself assigned or its address taken
                written.add(r.name)
        if e.kind == "member" and e.op == "." and site.cplusplus:
            r, in_place = _root(e.kids[0])
            if r is not None and r.name in scope and in_place and not scope[r.name].pointerish:
                written.add(r.name)              # an object used in place: by reference
    if self_used and (cls is None or not site.cplusplus):
        return []
    if not used and not self_used:
        return []
    for n in used:
        v = scope[n]
        if site.cplusplus and not v.pointerish and not SCALAR.match(v.decl[:v.name_at].strip()):
            written.add(n)                       # an object (not a pointer or number): by reference
    ret = None if b.tail else _returning(site, b, {w for w in written if w in used})
    out = []
    base = cbody.indent_at(site.src, b.start)
    insert = _insert_at(site.src, span)
    self_decl = f"{'const ' if const else ''}{cls}* self" if self_used else None
    orders = [used, sorted(used, key=lambda n: scope[n].order)]
    for form in (["return"] if ret else []) + ["plain"]:
        for k, order in enumerate(orders):
            if k and order == orders[0]:
                continue
            body_edits = list(edits)
            names = [n for n in order]
            modes = {}
            for n in names:
                if form == "return" and n == ret[0]:
                    continue
                modes[n] = "value" if n not in written else ("ref" if site.cplusplus else "ptr")
            if form == "return":
                names = [n for n in names if n != ret[0]]
                for st, brk in ret[2]:
                    body_edits.append((st.start, st.end, f"return {site.text(st.expr.kids[1])};"))
                    gap_end = brk.end
                    body_edits.append((st.end, gap_end, ""))
            for e in _exprs(b.stmts):
                if e.kind == "name" and modes.get(e.name) == "ptr":
                    body_edits.append((e.start, e.end, f"(*{e.name})"))
            try:
                text = cbody.apply_edits(site.src[b.start:b.end], [(x - b.start, y - b.start, t) for x, y, t in
                                                                    _outer(body_edits)])
            except ValueError:
                continue
            plist = ([self_decl] if self_decl else []) + [scope[n].as_param(modes[n]) for n in names]
            args = (["this"] if self_decl else []) + [("&" + n) if modes[n] == "ptr" else n for n in names]
            call = f"{name}({', '.join(args)})"
            if form == "return":
                x, kval, _ = ret
                v = scope[x]
                rtype = v.as_param("value")
                rtype = rtype[:v.name_at - (len(v.decl) - len(rtype))].strip() if rtype.endswith(v.decl[v.name_at:]) else None
                if not rtype or "[" in v.decl[v.name_at:]:
                    continue
                helper = (f"static inline {rtype} {name}({', '.join(plist)}) {{\n{_reindent(text, base)}\n"
                          f"    return {site.text(kval)};\n}}\n\n")
                p = site.parents[id(b.loop)]
                init = p[0].kids[p[1] - 1]
                if init.kind == "decl":
                    d = init.decl.declarators[0]
                    repl = [(d.init.start, b.end, f"{call};")]
                    lo = init.start
                else:
                    repl = [(init.expr.kids[1].start, b.end, f"{call};")]
                    lo = init.start
                if lo < insert:
                    continue
                label = f"search loop at line {cbody.line_of(site.src, b.start) + 1} as inline helper {x} = {call}"
            elif b.tail:
                rtype = return_type(site)
                if rtype is None:
                    continue
                helper = f"static inline {rtype} {name}({', '.join(plist)}) {{\n{_reindent(text, base)}\n}}\n\n"
                repl = [(b.start, b.end, f"{call};" if rtype == "void" else f"return {call};")]
                label = f"loop at line {cbody.line_of(site.src, b.start) + 1} and the rest as inline helper {call}"
            else:
                helper = f"static inline void {name}({', '.join(plist)}) {{\n{_reindent(text, base)}\n}}\n\n"
                repl = [(b.start, b.end, f"{call};")]
                what = ("guarded loop" if b.stmts[0].kind == "if" else "loop" if len(b.stmts) == 1 else
                        "loop and its locals" if b.loop.kind in LOOPS else "block")
                label = f"{what} at line {cbody.line_of(site.src, b.start) + 1} as inline helper {call}"
            out.append((label, [(insert, insert, helper)] + repl))
    return out


def _outer(edits: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """Drop edits inside a span another edit replaces (a whole statement turned into a return)."""
    out = []
    for e in sorted(edits, key=lambda x: (x[0], -x[1])):
        if out and e[0] < out[-1][1] and out[-1][0] != out[-1][1]:
            continue
        out.append(e)
    return out


def r_inline_helper(site: Site):
    """A loop (with the locals only it uses, or its guard) moved into a `static inline` helper:
    inside an inlined helper its locals rank below the caller's parameters."""
    span = span_of(site)
    if span is None or "template" in site.src[span.head_start:span.open]:
        return
    params = parameters(site.src, span)
    if params is None:
        return
    cls = owner_class(span)
    members = set(getattr(site, "members", None) or ())
    if not members and cls is not None:
        members = {t.text for t in cbody.tokens(site.src, site.body.start, site.body.end)
                   if t.kind == "id" and re.fullmatch(r"m[0-9A-F]+|m[A-Z_]\w*", t.text)}
    prefer = set(getattr(site, "prefer", ()) or ())
    found = blocks(site)

    def key(b: Block):
        names = {e.name for e in _exprs(b.stmts) if e.kind == "name"} | \
                {d.name for x in b.stmts for s in cbody.statements(x) if s.decl is not None for d in s.decl.declarators}
        depth = sum(1 for s in cbody.statements(site.body) if s.kind in LOOPS and s.start < b.loop.start < s.end)
        return (not (names & prefer), -depth, b.start, b.end)

    for b in sorted(found, key=key):
        name = f"Unk{'Loop' if b.loop.kind in LOOPS else 'Block'}{cbody.line_of(site.src, b.loop.start) + 1}"
        if re.search(rf"\b{name}\b", site.src):
            continue
        yield from extract(site, b, params, members, cls, const_member(site.src, span), name)


def _soft(gen):
    """A rule that never stops the search: a source shape it does not expect gives no candidates."""
    def run(site: Site):
        try:
            yield from list(gen(site))
        except (ValueError, KeyError, IndexError, AttributeError, StopIteration, TypeError):
            return
    run.__doc__ = gen.__doc__
    return run


RULE = Rule("inline-helper", "A loop counter (or a loop's temporaries) ranked below the parameters in the target, "
            "above them in ours", "the loop was a `static inline` helper: move it there (its locals rank below the "
            "caller's parameters)", ("regs",), _soft(r_inline_helper))


# ---- an existing helper's parameter order --------------------------------------------------------

def _args_at(toks: list, k: int) -> tuple[int, list[tuple[int, int]]] | None:
    """For tokens[k] == '(' : the index of its ')' and each top-level argument's (start, end) offsets."""
    depth, spans, start = 0, [], None
    for j in range(k, len(toks)):
        t = toks[j]
        if t.text in ("(", "[", "{"):
            depth += 1
            if depth == 1:
                start = None
                continue
        elif t.text in (")", "]", "}"):
            depth -= 1
            if depth == 0:
                if start is not None:
                    spans.append((start, toks[j - 1].end))
                return j, spans
        if depth == 1 and t.text == "," and t.kind == "op":
            spans.append((start, toks[j - 1].end))
            start = None
            continue
        if start is None:
            start = t.start
    return None


def helper_calls(src: str, name: str) -> list[list[tuple[int, int]]] | None:
    """Every call of `name` in the file as its arguments' spans (None if one cannot be read)."""
    toks = cbody.tokens(src)
    out = []
    for k, t in enumerate(toks[:-1]):
        if t.kind == "id" and t.text == name and toks[k + 1].text == "(" and (k == 0 or toks[k - 1].text not in (".", "->", "::")):
            got = _args_at(toks, k + 1)
            if got is None:
                return None
            out.append(got[1])
    return out


def r_helper_params(site: Site):
    """The parameter order of a `static inline` helper the function calls: the inline's parameters
    are temporaries, ranked by their order (adjacent pairs swapped, at the definition and every call)."""
    called = {e.kids[0].name for e in (x for s in cbody.statements(site.body) for top in cbody.statement_exprs(s)
                                       for x in cbody.expressions(top)) if e.kind == "call" and e.kids[0].kind == "name"}
    spans = {f.name: f for f in cbody.functions(site.src)}
    todo = list(called)
    while todo:                                       # and the helpers those helpers call
        f = spans.get(todo.pop())
        if f is None:
            continue
        for t in cbody.tokens(site.src, f.open, f.close):
            if t.kind == "id" and t.text in spans and t.text not in called:
                called.add(t.text)
                todo.append(t.text)
    for span in cbody.functions(site.src):
        if span.name not in called or "::" in span.name:
            continue
        head = site.src[span.head_start:span.open]
        if not re.search(r"\bstatic\b", head) or not re.search(r"\binline\b", head):
            continue
        params = parameters(site.src, span)
        if not params or len(params) < 2:
            continue
        calls = helper_calls(site.src, span.name)
        if not calls:
            continue
        defs = [c for c in calls if span.head_start <= c[0][0] < span.open] if calls and calls[0] else []
        uses = [c for c in calls if not (span.head_start <= (c[0][0] if c else -1) < span.open)]
        if any(len(c) != len(params) for c in uses):
            continue                                  # default arguments or a mismatch: leave it
        for i in range(len(params) - 1):
            a, b = params[i], params[i + 1]
            a_at = site.src.index(a.decl, span.head_start)
            b_at = site.src.index(b.decl, a_at + len(a.decl))
            edits = [(a_at, a_at + len(a.decl), b.decl), (b_at, b_at + len(b.decl), a.decl)]
            for c in uses:
                (x0, x1), (y0, y1) = c[i], c[i + 1]
                edits += [(x0, x1, site.src[y0:y1]), (y0, y1, site.src[x0:x1])]
            yield f"{span.name}'s parameters {a.name}, {b.name} swapped", edits


PARAMS_RULE = Rule("helper-params", "Registers of an inlined helper's values in another order",
                   "the helper's parameters in another order (they are temporaries, ranked by it)", ("regs",),
                   _soft(r_helper_params))
