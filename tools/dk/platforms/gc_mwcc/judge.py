"""The closer's judge: a candidate source is kept only if it is byte-exact AND faithful.

Exact: the function scores 100% (objdiff, data relocations included) and hand-links
byte-identical to the original; nothing else in the unit scores lower than before (functions,
data sections, layout, vtables).

Faithful (the project's "strict" decision): compared with the source it started from, the
candidate adds no `volatile`, no pragma, no `asm`, no `__typeof__`, no padding member, no hit of a
forbidden quality rule, no inline-forcing attribute, no hard-coded byte-offset cast and no copy-only local (a new
local whose every value is a plain copy of another variable, or one that is never read). Those are devices that steer the compiler, not the
code the original programmers wrote. (`review` adds the ones that need a look: invented empty cases,
a hand-copied inline body, a cast to the operand's own type; and it rejects a placeholder-named new local
(`t0`, `tmp1`), which this gate accepts because `dk close`'s own rewrite names its temporaries `t0`.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ...core import cbody
from .matching import UnitScore

GATES = [
    ("volatile", re.compile(r"\bvolatile\b")),
    ("pragma", re.compile(r"^\s*#\s*pragma\b|\b_Pragma\s*\(", re.M)),
    ("asm", re.compile(r"^\s*(?:static\s+|extern\s+)*asm\b|\b(?:__)?asm\s*[({]|\basm\s+volatile\b", re.M)),
    ("__typeof__", re.compile(r"\b__typeof__\b|\btypeof\s*\(")),
    ("padding", None),                    # filled in below: `padding()` counts it
    ("inline attribute", re.compile(r"__attribute__\s*\(\(\s*(?:__)?(?:always_inline|noinline|flatten)|"
                                    r"__declspec\s*\(\s*noinline|\b__forceinline\b")),
    ("register keyword", re.compile(r"\bregister\b")),
]


@dataclass
class Verdict:
    reasons: list[str] = field(default_factory=list)
    review: tuple[str, list[str]] | None = None     # (category, flags) of `exact_review`, when it ran

    @property
    def ok(self) -> bool:
        return not self.reasons


def source_reasons(before: str, after: str, forbidden: list[tuple[str, re.Pattern]] = ()) -> list[str]:
    """Devices the candidate adds to the file: any gate or forbidden quality rule that matches more
    often after than before, and alignment attributes on statics and file-level objects."""
    out = []
    for name, pat in list(GATES) + list(forbidden):
        b, a = len(pat.findall(before)), len(pat.findall(after))
        if a > b:
            out.append(f"adds {a - b} {name}")
    if (n := aligned_decls(after)[0] - aligned_decls(before)[0]) > 0:
        out.append(f"adds {n} aligned static (an alignment attribute on a static or file-level object)")
    return out


_PAD = re.compile(r"\b(?:pad|dummy)\w*\s*\[", re.I)
_PAD_LOCAL = re.compile(r"\b\w*(?:pad|dummy|filler|unused|spacer)\w*\s*\[", re.I)


class _Padding:
    """Looks like a compiled pattern to `source_reasons`: findall returns one hit per filler array."""
    @staticmethod
    def findall(src: str) -> list[int]:
        return [0] * padding(src)


GATES[[n for n, _ in GATES].index("padding")] = ("padding", _Padding)


def padding(src: str) -> int:
    """Filler arrays: `pad[..]`/`dummy[..]` anywhere, and inside function bodies any array named like filler
    (`stackpad[4]`, `unused[16]`, `_pad0[8]`): stack space the original had no use for. Struct members are layout
    (`char _pad0[0x58]`) and not counted."""
    try:
        spans = [(f.open, f.close) for f in cbody.functions(src)]
    except cbody.ParseError:
        spans = []
    return len(_PAD.findall(src)) + sum(1 for m in _PAD_LOCAL.finditer(src) if not _PAD.match(src, m.start())
                                       and any(a < m.start() < b for a, b in spans))


_ALIGN = re.compile(r"__attribute__\s*\(\(\s*(?:__)?aligned(?:__)?\b|\b__declspec\s*\(\s*align\b|\bATTRIBUTE_ALIGN\s*\(")


def aligned_decls(src: str) -> tuple[int, int]:
    """(statics and file-level objects, stack locals) that carry an alignment attribute
    (`__attribute__((aligned(N)))`, `__declspec(align)`, `ATTRIBUTE_ALIGN`). A static that is only
    aligned to fill bytes is a device; a stack matrix the target realigns is an open question
    (faithfulness.md), so the two are counted apart."""
    hits = [m.start() for m in _ALIGN.finditer(src)]
    if not hits:
        return 0, 0
    spans = [(f.open, f.close) for f in cbody.functions(src)]
    data = local = 0
    for at in hits:
        lo = max(src.rfind(c, 0, at) for c in ";{}") + 1
        hi = min((i for i in (src.find(c, at) for c in ";{=") if i >= 0), default=len(src))
        static = re.search(r"\bstatic\b", src[lo:hi]) is not None
        if static or not any(a < at < b for a, b in spans):
            data += 1
        else:
            local += 1
    return data, local


def declared(body: cbody.Stmt) -> list[tuple[str, cbody.Declarator, cbody.Stmt]]:
    return [(d.name, d, s) for s in cbody.statements(body) if s.decl is not None for d in s.decl.declarators]


def copy_only_locals(before_src: str, before_body: cbody.Stmt, after_src: str, after_body: cbody.Stmt,
                     params: dict[str, str] | None = None) -> list[str]:
    """New locals that are never read, or that are a pure alias of one variable: every value a plain
    copy of the same variable, which is not changed after the copy (nor anywhere in a loop around
    it), so the local stands in for it. Not aliases: constructor arguments (`TxtStr s(lbl_x)`), a
    merge of several values (`r = a` or `r = b`), a copy kept while the source changes (`old = m;
    m = x; return old;`, a best-so-far in a loop), a conversion to another type, and a member or
    global read kept across a call or a store (a snapshot: the compiler must reload the source).
    `params` (name -> type) tells parameters from members and globals; without it every source
    counts as a variable."""
    old = {n for n, _, _ in declared(before_body)}
    decls = declared(after_body)
    kinds = {n: _bare(_norm(after_src[s.decl.type_start:s.decl.type_end]) + d.ptr) for n, d, s in decls}
    kinds.update({n: _bare(t) for n, t in (params or {}).items()})
    loops = [s for s in cbody.statements(after_body) if s.kind in ("for", "while", "do")]
    out = []
    for name, d, st in decls:
        if name in old or re.search(r"\bstatic\b", after_src[st.decl.type_start:st.decl.type_end]):
            continue                                # a static local is data: the original's (its .bss slot)
        values = [d.init] if d.init is not None else []
        reads: list[cbody.Expr] = []
        for s in cbody.statements(after_body):
            for top in cbody.statement_exprs(s):
                for e in cbody.expressions(top):
                    if e.kind == "assign" and e.op == "=" and e.kids[0].kind == "name" and e.kids[0].name == name:
                        values.append(e.kids[1])
                    elif e.kind == "name" and e.name == name and not _assigned(top, e):
                        reads.append(e)
        if d.init is not None and any(x is d.init for x in values):
            inside = {id(e) for e in cbody.expressions(d.init)}
            reads = [e for e in reads if id(e) not in inside]
        if d.init_kind in ("()", "{}"):            # `XFile file(name, 0, mode);`: a constructor call is code
            continue
        if not reads:
            if d.init is None or not (any(e.kind == "call" for e in cbody.expressions(d.init))
                                      or re.search(r"\w\s*<[^;]*>\s*\(", after_src[d.init.start:d.init.end])):
                out.append(f"new local {name} is never read")   # `int n = f();` unread: the call is the code
            continue
        if not values or not all(v.kind == "name" and v.name not in ("true", "false", "NULL") for v in values) \
                or "&" in d.ptr or _worked_on(after_body, name):
            continue
        srcs = {v.name for v in values}
        if len(srcs) != 1:
            continue                                # a merge of several values, not an alias of one
        (src,) = srcs
        if src in kinds and kinds[src] != kinds[name] and not d.array:
            continue                                # a conversion (`int x = c;` for a char c) is code
        at = min(v.start for v in values)
        if _changed_after(after_body, src, at) or _redeclared(after_body, src, at) \
                or any(lp.start <= at < lp.end and _changed_in(after_body, src, lp) for lp in loops):
            continue                                # keeps a value the source no longer has
        memory = params is not None and src not in kinds
        if memory or src in _addressed(after_body):  # a member, a global or an object whose address is
            last = max(e.start for e in reads)         # out: kept across a call or a store?
            for lp in loops:                        # a use in a loop the copy is outside of: the whole loop
                if not lp.start <= at < lp.end and any(lp.start <= e.start < lp.end for e in reads):
                    last = max(last, lp.end)
            if _call_or_store(after_body, at, last, set(kinds)):
                continue
        if memory:
            out.append(f"new local {name} only caches {src} (a member or global nothing changes in between)")
            continue
        out.append(f"new local {name} only copies {', '.join(after_src[v.start:v.end] for v in values)}")
    return out


def _bare(t: str) -> str:
    """A type with spaces removed, without its qualifiers and storage class."""
    return re.sub(r"register|auto|const|volatile|static", "", t)


def _targets(body: cbody.Stmt):
    """(name, offset) of every variable an expression changes or works on in place: compound
    assignments, ++/--, member access, address-of."""
    for s in cbody.statements(body):
        for top in cbody.statement_exprs(s):
            for e in cbody.expressions(top):
                k = e.kids[0] if e.kids else None
                if k is None or k.kind != "name":
                    continue
                if (e.kind == "assign") or (e.kind in ("unary", "postfix") and e.op in ("++", "--", "&")) \
                        or e.kind == "member":
                    yield k.name, e.start, e.kind == "assign" and e.op == "="


def _worked_on(body: cbody.Stmt, name: str) -> bool:
    """Is the local changed in place or used as an object (`r += b`, `r++`, `r.f()`, `&r`)? Then
    it is not a plain copy."""
    return any(n == name and not plain for n, _, plain in _targets(body))


def _changed_after(body: cbody.Stmt, name: str, at: int) -> bool:
    """Is `name` changed after `at`? A copy taken before then keeps the old value (`old = m; m = x;
    return old;`), which is code, not a device."""
    return any(n == name and off > at for n, off, _ in _targets(body))


def _addressed(body: cbody.Stmt) -> set[str]:
    """Variables whose address (or a member's or element's) is formed: `&v`, `&v.x`, `&v[i]`."""
    out = set()
    for s in cbody.statements(body):
        for top in cbody.statement_exprs(s):
            for e in cbody.expressions(top):
                if e.kind == "unary" and e.op == "&" and e.kids:
                    k = e.kids[0]
                    while k.kind in ("member", "index", "paren") and k.kids and (k.kind != "member" or k.op == "."):
                        k = k.kids[0]
                    if k.kind == "name":
                        out.add(k.name)
    return out


def _redeclared(body: cbody.Stmt, name: str, at: int) -> bool:
    """Is a variable called `name` declared after `at` (a new value under the same name)?"""
    return any(n == name and d.start > at for n, d, _ in declared(body))


def _changed_in(body: cbody.Stmt, name: str, loop: cbody.Stmt) -> bool:
    """Is `name` changed or declared anywhere in the loop (a new value on the next iteration)?"""
    return any(n == name and loop.start <= off < loop.end for n, off, _ in _targets(body)) \
        or any(n == name and loop.start <= d.start < loop.end for n, d, _ in declared(body))


def _call_or_store(body: cbody.Stmt, lo: int, hi: int, local: set[str]) -> bool:
    """A call, `new`/`delete`, or a store to anything but a local, done after lo and before hi
    (`g(x)` reading x at hi is not one): either may change a member or a global."""
    for s in cbody.statements(body):
        for top in cbody.statement_exprs(s):
            for e in cbody.expressions(top):
                if not (lo < e.start and e.end <= hi):
                    continue
                if e.kind == "call" or (e.kind == "unary" and e.op in ("new", "delete")):
                    return True
                if e.kind == "assign" or (e.kind in ("unary", "postfix") and e.op in ("++", "--")):
                    k = e.kids[0]
                    if not (k.kind == "name" and k.name in local):
                        return True
    return False


def _assigned(top: cbody.Expr, e: cbody.Expr) -> bool:
    """Is e the left side of a plain `=` inside top?"""
    return any(x.kind == "assign" and x.op == "=" and x.kids[0] is e for x in cbody.expressions(top))


def unit_reasons(before: UnitScore, after: UnitScore, function: str) -> list[str]:
    """Why the candidate's unit is not acceptable: the function not exact, or anything else scoring
    lower than before."""
    out = []
    got = {f.name: f.percent for f in after.functions}
    if got.get(function) != 100.0:
        out.append(f"{function} is at {got.get(function, 0.0):.2f}%, not exact")
    for f in before.functions:
        if f.name != function and got.get(f.name, -1.0) < f.percent:
            out.append(f"{f.name} {'was exact' if f.percent == 100.0 else f'was {f.percent:.2f}%'}, now "
                       + (f"{got[f.name]:.2f}%" if f.name in got else "missing"))
    out += [f"{sec} {p:.2f}% before, now {after.data.get(sec, 0.0):.2f}%" for sec, p in sorted(before.data.items())
            if after.data.get(sec, 0.0) < p]
    known = {p.split(":", 1)[0] for p in before.layout_problems}   # a function already off: its offsets move with
    out += [f"new layout problem: {p}" for p in after.layout_problems       # every size change above it, not a new problem
            if p not in before.layout_problems and p.split(":", 1)[0] not in known]
    out += [f"new vtable problem: {p}" for p in after.data_problems if p not in before.data_problems]
    return out


# ---- devices found by reading the body: placeholder temps, own-type casts, copied inline bodies -----

PLACEHOLDER_LOCAL = re.compile(r"(?:t|tmp|temp|var|v|val|unk)\d+|tmp|temp", re.I)


def placeholder_locals(before_body: cbody.Stmt, after_body: cbody.Stmt) -> list[str]:
    """New locals named like a placeholder (`t0`, `tmp1`, `var3`): a name the original never had, left by a
    search that only wanted a temporary there (`dk try` refuses them too)."""
    old = {n for n, _, _ in declared(before_body)}
    return [f"new local {n} has a placeholder name" for n, _, _ in declared(after_body)
            if n not in old and PLACEHOLDER_LOCAL.fullmatch(n)]


def self_casts(before_src: str, before_body: cbody.Stmt, after_src: str, after_body: cbody.Stmt,
               params: dict[str, str] | None) -> list[str]:
    """Added casts of a plain local or parameter to the type it was declared with, `(int)i` for an
    `int i`: no compiler needed (`same_type_casts` measures the effect of the ones it can probe)."""
    def kinds(src: str, body: cbody.Stmt) -> dict[str, str]:
        out = {n: _bare(_norm(src[s.decl.type_start:s.decl.type_end]) + d.ptr) for n, d, s in declared(body)}
        out.update({n: _bare(t) for n, t in (params or {}).items()})
        return out
    have = kinds(after_src, after_body)
    out = []
    for e in added_casts(before_src, before_body, after_src, after_body):
        x = e.kids[0]
        while x.kind == "paren" and x.kids:
            x = x.kids[0]
        if x.kind == "name" and have.get(x.name) == _bare(_norm(after_src[e.type_span[0]:e.type_span[1]])):
            out.append(f"redundant cast to its own type ({after_src[e.type_span[0]:e.type_span[1]].strip()}){x.name}")
    return out


_INTEGER_WIDTH = {**dict.fromkeys(("char", "signedchar", "unsignedchar", "s8", "u8"), 8),
                  **dict.fromkeys(("short", "shortint", "signedshort", "unsignedshort", "unsignedshortint", "s16", "u16"), 16),
                  **dict.fromkeys(("int", "signed", "signedint", "unsigned", "unsignedint", "long", "longint", "signedlong",
                                   "unsignedlong", "unsignedlongint", "s32", "u32"), 32)}


def integer_casts(before_src: str, before_body: cbody.Stmt, after_src: str, after_body: cbody.Stmt,
                  params: dict[str, str] | None) -> list[str]:
    """Added casts of a plain local or parameter to the other signedness of its own width, `(int)n` for an
    `unsigned n`: the value is unchanged, only the code steered; the declaration may be what differs (RL1:
    the human rejection of an `(int)n` index cast; rlgate's declarative twin measures it with the compiler).
    A width change (`(unsigned char)i`) is a real conversion, not flagged."""
    have = {n: _bare(_norm(after_src[s.decl.type_start:s.decl.type_end]) + d.ptr) for n, d, s in declared(after_body)}
    have.update({n: _bare(t) for n, t in (params or {}).items()})
    out = []
    for e in added_casts(before_src, before_body, after_src, after_body):
        x = e.kids[0]
        while x.kind == "paren" and x.kids:
            x = x.kids[0]
        t = _bare(_norm(after_src[e.type_span[0]:e.type_span[1]]))
        d = have.get(x.name) if x.kind == "name" else None
        if d and d != t and _INTEGER_WIDTH.get(d) is not None and _INTEGER_WIDTH.get(d) == _INTEGER_WIDTH.get(t):
            out.append(f"adds an integer cast ({after_src[e.type_span[0]:e.type_span[1]].strip()}){x.name} of a {d} "
                       f"(signedness only: the declaration may be what differs)")
    return out


_WORD = re.compile(r"[A-Za-z_]\w*")


def _skeleton(src: str, helper: cbody.FunctionSpan, params: dict[str, str]) -> re.Pattern | None:
    """A regular expression over tokens for the statements of an inline helper (its final `return` left
    off): the helper's parameters match any one expression (the same text wherever repeated), its own
    locals any name. None for a helper too small to copy (under two statements, or under 8 tokens)."""
    body = cbody.parse_body(src, helper.open)
    kids = body.kids[:-1] if body.kids and body.kids[-1].kind == "return" else body.kids
    if len(body.kids) < 2 or not kids:
        return None
    toks = cbody.tokens(src, kids[0].start, kids[-1].end)
    if len(toks) < 8:
        return None
    locals_ = {n for n, _, _ in declared(body)}
    groups: dict[str, int] = {}
    parts = []
    for t in toks:
        if t.kind == "id" and (t.text in params or t.text in locals_):
            if t.text in groups:
                parts.append(f"(?P=g{groups[t.text]})")
            else:
                groups[t.text] = len(groups)
                parts.append(f"(?P<g{groups[t.text]}>{'[A-Za-z_]\\w*' if t.text in locals_ else '[^;{}]+?'})")
        else:
            parts.append(re.escape(t.text))
    return re.compile(" ".join(parts))


def copied_inlines(before_src: str, after_src: str, symbol: str) -> list[str]:
    """Explicitly `inline` helpers of the file whose statements the candidate's definition now writes out by hand,
    with the helper's parameters replaced by arguments: `if (v < 0) v = 0; if (v > 100) v = 100;` where the
    helper `ClampPct(v)` exists. The compiler inlines the call anyway; a hand copy is a way to steer the
    allocator, not what the original wrote (unless the original wrote it out: needs a look)."""
    from .trycmd import definition
    a_span, b_span = definition(after_src, symbol), definition(before_src, symbol)
    if a_span is None:
        return []
    out = []
    for f in cbody.functions(after_src):
        if f.name == symbol or (f.head_start, f.close + 1) == a_span \
                or not re.search(r"\binline\b", after_src[f.head_start:f.open]):
            continue                                # only helpers written `inline`: a twin function is its own code
        pat = _skeleton(after_src, f, parameters(after_src, (f.head_start, f.close + 1)))
        if pat is None:
            continue
        count = lambda src, span: len(pat.findall(" ".join(t.text for t in cbody.tokens(src, span[0], span[1])))) if span else 0
        if count(after_src, a_span) > count(before_src, b_span):
            out.append(f"hand-copied inline body of {f.name}")
    return out



def judge(before_src: str, after_src: str, before_body: cbody.Stmt | None, after_body: cbody.Stmt | None,
          before: UnitScore | None, after: UnitScore | None, function: str,
          forbidden: list[tuple[str, re.Pattern]] = (), proof=None, probe=None) -> Verdict:
    """`probe(text) -> percent | None` (the unit's compiler, unit_probe): also reject casts to the
    operand's own type that change the code."""
    v = Verdict()
    if before is not None and after is not None:
        v.reasons += unit_reasons(before, after, function)
    if proof is not None and getattr(proof, "judged", True) and not proof.ok:
        v.reasons.append(f"hand-linked, it differs from the original: {proof.detail}")
    v.reasons += source_reasons(before_src, after_src, forbidden)
    if before_body is None or after_body is None:       # the callers that pass no bodies get the same checks
        before_body, after_body, _ = bodies(before_src, after_src, function)
    if before_body is not None and after_body is not None:
        from .trycmd import definition
        span = definition(after_src, function)
        v.reasons += copy_only_locals(before_src, before_body, after_src, after_body,
                                      parameters(after_src, span) if span else None)
        if offset_casts(after_src, after_body) > offset_casts(before_src, before_body):
            v.reasons.append("hard-coded offset cast")
    if probe is not None:
        b_body, a_body, _ = bodies(before_src, after_src, function)
        if a_body is not None:
            v.reasons += same_type_casts(before_src, b_body, after_src, a_body, probe)[0]
    return v


# ---- casts that only steer the compiler, and aliasing locals --------------------------------------

def bodies(before_src: str, after_src: str, symbol: str):
    """(before body or an empty one, after body or None, after span) of the symbol's definition."""
    from .trycmd import definition
    b_span, a_span = definition(before_src, symbol), definition(after_src, symbol)
    b_body = _parse(before_src, b_span) if b_span else None
    a_body = _parse(after_src, a_span) if a_span else None
    return (b_body if b_body is not None else cbody.parse_body("{}", 0)), a_body, a_span


def _norm(text: str) -> str:
    return "".join(text.split())


def _casts(src: str, body: cbody.Stmt) -> list[cbody.Expr]:
    return [e for s in cbody.statements(body) for top in cbody.statement_exprs(s) for e in cbody.expressions(top)
            if e.kind == "cast" and e.type_span is not None and e.kids]


def added_casts(before_src: str, before_body: cbody.Stmt, after_src: str, after_body: cbody.Stmt) -> list[cbody.Expr]:
    """The candidate's casts that the source's definition does not have (same type and operand)."""
    have: dict[tuple[str, str], int] = {}
    for e in _casts(before_src, before_body):
        k = (_norm(before_src[e.type_span[0]:e.type_span[1]]), _norm(before_src[e.kids[0].start:e.kids[0].end]))
        have[k] = have.get(k, 0) + 1
    out = []
    for e in _casts(after_src, after_body):
        k = (_norm(after_src[e.type_span[0]:e.type_span[1]]), _norm(after_src[e.kids[0].start:e.kids[0].end]))
        if have.get(k, 0) > 0:
            have[k] -= 1
        else:
            out.append(e)
    return out


def same_type_probe(src: str, e: cbody.Expr) -> str | None:
    """src with the cast `(T)x` wrapped in an unevaluated check that compiles only when x already
    has type T: `(sizeof(*(T**)0 = (__typeof__(x)*)0), (T)x)` (a T** takes no other pointer type
    implicitly, in C++ or in mwcc's C). None for a type it cannot write that way."""
    t = src[e.type_span[0]:e.type_span[1]].strip()
    if not t or re.search(r"[()\[\]]", t) or _norm(t) in ("void", "constvoid"):
        return None
    x = src[e.kids[0].start:e.kids[0].end]
    probe = f"(sizeof(*({t}**)0 = (__typeof__({x})*)0), {src[e.start:e.end]})"
    return cbody.apply_edits(src, [(e.start, e.end, probe)])


def same_type_casts(before_src: str, before_body: cbody.Stmt, after_src: str, after_body: cbody.Stmt,
                    probe) -> tuple[list[str], list[str]]:
    """Added casts to the type their operand already has, `(NpcBase*)chase->NthFollower(0)` where
    NthFollower returns NpcBase*: (rejects, needs judgment). `probe(text)` compiles with the unit's
    compiler and returns the function's match percent (None: no compile). A cast whose removal
    changes the code only steers the compiler (or the declared type is wrong): reject. One without
    effect is noise to drop."""
    casts = [(e, p) for e in added_casts(before_src, before_body, after_src, after_body)
             if (p := same_type_probe(after_src, e)) is not None]
    if not casts:
        return [], []
    base = probe(after_src)
    if base is None:
        return [], [f"cast check could not run: the candidate does not compile now ({len(casts)} added casts)"]
    reject, judgment = [], []
    for e, text in casts:
        if probe(text) is None:
            continue
        what = (f"({' '.join(after_src[e.type_span[0]:e.type_span[1]].split())})"
                f"{' '.join(after_src[e.kids[0].start:e.kids[0].end].split())[:60]}")
        bare = probe(cbody.apply_edits(after_src, [(e.start, e.end, f"({after_src[e.kids[0].start:e.kids[0].end]})")]))
        if bare != base:
            reject.append(f"cast to its own type changes the code {what}")
        else:
            judgment.append(f"redundant cast to its own type {what}")
    return reject, judgment


def parameters(src: str, span: tuple[int, int]) -> dict[str, str]:
    """A definition's parameters: name -> type as written (spaces removed)."""
    try:
        brace = src.index("{", span[0], span[1])
    except ValueError:
        return {}
    h = cbody.tokens(src, span[0], brace)
    k = len(h) - 1
    depth = 0
    for j, x in enumerate(h):                       # a constructor's `) : a(x)`: the `)` before the `:`
        depth += x.text in ("(", "[") and x.kind == "op"
        depth -= x.text in (")", "]") and x.kind == "op"
        if depth == 0 and x.text == ":" and x.kind == "op" and j > 0 and h[j - 1].text == ")":
            k = j - 1
            break
    while k >= 0 and h[k].text != ")":
        k -= 1
    depth, j = 0, k
    while j >= 0:
        depth += h[j].text == ")"
        depth -= h[j].text == "("
        if depth == 0:
            break
        j -= 1
    if j < 0:
        return {}
    out, cur, depth = {}, [], 0
    for x in h[j + 1:k] + [cbody.Tok("op", ",", 0, 0)]:
        if x.kind == "op" and x.text == "," and depth == 0:
            ids = [t for t in cur if t.kind == "id"]
            fp = [cur[i + 2] for i in range(len(cur) - 2) if cur[i].text == "(" and cur[i + 1].text == "*"
                  and cur[i + 2].kind == "id"]
            if "=" in [t.text for t in cur]:
                cur = cur[:[t.text for t in cur].index("=")]
                ids = [t for t in cur if t.kind == "id"]
            name = fp[0] if fp else (ids[-1] if len(cur) > 1 and ids and ids[-1] is cur[-1]
                                     and ids[-1].text not in cbody.KEYWORDS else None)
            if name is not None:
                out[name.text] = _norm("".join(t.text + " " for t in cur if t is not name))
            cur = []
            continue
        depth += x.text in ("(", "[") and x.kind == "op"
        depth -= x.text in (")", "]") and x.kind == "op"
        cur.append(x)
    return out


def aliasing_locals(before_body: cbody.Stmt, after_src: str, after_body: cbody.Stmt,
                    params: dict[str, str]) -> list[str]:
    """New locals of a parameter's or local's own type, initialised from it, that take its place:
    the original is never used after the copy (`int temp_base = base;` then only temp_base). Some
    originals do copy a parameter, so this needs judgment rather than a reject."""
    old = {n for n, _, _ in declared(before_body)}
    types = dict(params)
    uses: dict[str, list[int]] = {}
    for s in cbody.statements(after_body):
        for top in cbody.statement_exprs(s):
            for e in cbody.expressions(top):
                if e.kind == "name":
                    uses.setdefault(e.name, []).append(e.start)
    out = []
    for name, d, s in declared(after_body):
        spec = after_src[s.decl.type_start:s.decl.type_end]
        mine = _norm(re.sub(r"\b(?:register|auto)\b", "", spec)) + d.ptr
        if name not in old and d.init is not None and d.init.kind == "name" and d.init_kind == "=" \
                and not d.array and "&" not in d.ptr and not re.search(r"\bstatic\b", spec):
            was = d.init.name
            if was != name and types.get(was) == mine and not any(at > d.end for at in uses.get(was, [])) \
                    and any(at > d.end for at in uses.get(name, [])):
                out.append(f"new local {name} aliases {was} ({was} is not used after it)")
        types.setdefault(name, mine)
    return out


def forbidden_rules(cfg) -> list[tuple[str, re.Pattern]]:
    """The project's forbidden quality rules (dk.toml [quality])."""
    return [(f"forbidden {r['name']}", re.compile(r["pattern"], re.M)) for r in cfg.quality.get("forbidden", [])]


def replace_typeof(src: str, body: cbody.Stmt, types: dict[str, str]) -> tuple[str, list[str]]:
    """Write each `__typeof__(…)` declaration's real type (local name -> type, from the debug
    info). Returns the new source and the names it could not type."""
    edits, missing = [], []
    for s in cbody.statements(body):
        if s.decl is None or not src[s.decl.type_start:s.decl.type_end].lstrip().startswith("__typeof__"):
            continue
        d = s.decl.declarators[0]
        t = types.get(d.name)
        if t is None or len(s.decl.declarators) != 1:
            missing.append(d.name)
            continue
        if d.ptr == "&" and t.endswith("&"):
            t = t[:-1]
        edits.append((s.decl.type_start, s.decl.type_end, t))
    return cbody.apply_edits(src, edits), missing


# ---- pre-review (dk llm-review): the same gates, sorted for a human ---------------------------------

REJECT, JUDGMENT, CLEAN = "reject", "needs-judgment", "clean"
_OFFSET_CAST = re.compile(r"\(\s*(?:unsigned\s+)?char\s*\*\s*\)[^;]*?\+\s*(?:0x[0-9A-Fa-f]+|\d+)\s*\)")
_RAW_CAST = re.compile(r"\(\s*(?:const\s+)?(?:void|(?:unsigned\s+)?(?:char|int|short|long)|u8|u16|u32|s8|s16|s32)\s*\*+\s*\)")
_DECL_LINE = re.compile(r"^\s*(?:(?:extern|static|typedef|struct|class|enum|union|const|unsigned|signed|virtual|inline)\s+)*"
                        r"[\w:<>,\s\*&\[\]()]*;\s*(?://.*)?$|^\s*(?:public|private|protected)\s*:\s*$|^\s*[{}];?\s*$")


_BYTE_PTR = re.compile(r"(?:const)?(?:(?:unsigned|signed)?char|u8|s8)\*")


def offset_casts(src: str, body: cbody.Stmt, loose: bool = False) -> int:
    """Fields reached by their byte offset instead of by name: a sum that starts from a cast to a
    byte pointer, adds a number and is then read or written as another type, `*(int*)((char*)p +
    0x248)`, `*((char*)p + 0x15E30) = 1`. Not byte arithmetic that stays bytes (`memcpy((char*)a +
    n + 4, …)`, `end = (char*)s + size + 0x10`) or offsets from data (`(char*)self + e->m4`).
    `loose`: every byte-pointer sum with a number, kept as bytes or not."""
    exprs = [e for s in cbody.statements(body) for top in cbody.statement_exprs(s) for e in cbody.expressions(top)]
    up = {id(k): e for e in exprs for k in e.kids}
    n = 0
    for e in exprs:
        if e.kind != "binary" or e.op not in ("+", "-") or (id(e) in up and up[id(e)].kind == "binary"
                                                             and up[id(e)].op in ("+", "-")):
            continue
        terms, todo = [], [e]
        while todo:
            x = todo.pop(0)
            if x.kind == "binary" and x.op in ("+", "-"):
                todo[:0] = x.kids
            else:
                terms.append(x)
        first = terms[0]
        while first.kind == "paren" and first.kids:
            first = first.kids[0]
        if not (first.kind == "cast" and first.type_span and _BYTE_PTR.fullmatch(_norm(src[first.type_span[0]:first.type_span[1]]))
                and any(t.kind == "num" for t in terms[1:])):
            continue
        if loose:
            n += 1
            continue
        at = up.get(id(e))
        while at is not None and at.kind == "paren":
            at = up.get(id(at))
        if at is None:
            continue
        typed = at.kind == "cast" and at.type_span and src[at.type_span[0]:at.type_span[1]].rstrip().endswith("*") \
            and not _BYTE_PTR.fullmatch(_norm(src[at.type_span[0]:at.type_span[1]]))
        if typed or (at.kind == "unary" and at.op == "*") or (at.kind in ("index", "member") and at.kids[0] is not None
                                                             and id(at.kids[0]) in {id(e)} | {id(p) for p in exprs if p.kind == "paren" and p.kids and p.kids[0] is e}):
            n += 1
    return n


def _idle(s: cbody.Stmt) -> bool:
    return s.kind in ("break", "empty") or (s.kind == "block" and all(_idle(k) for k in s.kids))


def empty_cases(body: cbody.Stmt | None) -> int:
    """`case` labels with no body of their own, in switches where they outnumber the labels that have
    one: the group of labels (`case 1: case 2:`) up to the next statement does nothing but `break`.
    Such labels only shape the compare tree: without evidence from the target, labels the original
    never wrote. (A lone `case 9: break;` closing a table of real cases is natural: 0 hits among the
    5,200 accepted game functions that have one.)"""
    n = 0
    for sw in [s for s in cbody.statements(body) if s.kind == "switch"] if body is not None else []:
        stmts = sw.kids[0].kids if sw.kids and sw.kids[0].kind == "block" else []
        empty = full = i = 0
        while i < len(stmts):
            j = i
            while j < len(stmts) and stmts[j].kind in ("case", "default"):
                j += 1
            k = max(j, i + 1)
            while k < len(stmts) and stmts[k].kind not in ("case", "default"):
                k += 1
            if j > i:
                if all(_idle(x) for x in stmts[j:k]):
                    empty += sum(1 for x in stmts[i:j] if x.kind == "case")
                else:
                    full += 1
            i = k
        n += empty if empty > full else 0
    return n


def _count(pat: re.Pattern, text: str) -> int:
    return len(pat.findall(text))


# ---- in-span devices (council RL1): constructs a reward-maximising search adds inside the definition --------

def code_only(text: str) -> str:
    """text with comments, string and character literals blanked to spaces (same length: offsets stay valid)."""
    return re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'',
                  lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.S)


_LVALUE = r"[A-Za-z_]\w*(?:\s*(?:\.|->)\s*[A-Za-z_]\w*|\s*\[[^\[\];]*\])*"
_IDENTITY = re.compile(                   # x + 0, x - 0, x | 0, x ^ 0, x << 0, x * 1, x / 1, x & ~0, 0 + x, 1 * x
    r"(?<=[\w)\]])\s*(?:(?<!\+)\+(?![+=])|(?<!-)-(?![-=>])|(?<!\|)\|(?![|=])|\^(?!=)|<<(?!=)|>>(?!=))\s*0(?![\w.])"
    r"|(?<=[\w)\]])\s*[*/](?![=*/])\s*1(?![\w.])"
    r"|(?<=[\w)\]])\s*(?<!&)&(?![&=])\s*(?:~\s*0|-\s*1)(?![\w.])"
    r"|(?<=[=(,\[?:{};])\s*0\s*\+(?![+=])\s*(?=[\w(])|(?<=[=(,\[?:{};])\s*1\s*\*(?![=*/])\s*(?=[\w(])")
_SELF_ASSIGN = re.compile(rf"(?P<l>{_LVALUE})\s*=(?!=)\s*(?P<r>{_LVALUE})\s*;")
_COMPOUND_IDENTITY = re.compile(rf"{_LVALUE}\s*(?:(?:\+|-|\||\^|<<|>>)=\s*0|(?:\*|/)=\s*1)(?![\w.])\s*;")
_FALSE_COND = re.compile(r"\b(?:if|while)\s*\(\s*(?:0+|0[xX]0+|false|NULL)\s*(?:\)|&&)")
_VOID0 = re.compile(r"\(\s*void\s*\)\s*0(?![\w.])")
_NOTNOT = re.compile(r"!\s*!(?!=)")
_PREPROCESSOR = re.compile(r"^[ \t]*#", re.M)
_ADDRESS_PUN = re.compile(r"\(\s*(?:const\s+)?[A-Za-z_][\w:]*(?:<[^<>()]*>)?\s*\*\s*\)\s*\(?\s*&(?!&)")
_LOCAL_TYPE = re.compile(r"\b(?:struct|class|union)\b[^;{}()]*\{")
SPAN_DEVICES = {                          # kind -> (flag, rejects?)
    "preprocessor": ("adds a preprocessor line inside the definition", True),
    "false-condition": ("adds a constant-false condition (if (0))", True),
    "self-assign": ("adds a self-assignment (x = x)", True),
    "identity": ("adds identity arithmetic (x + 0, x * 1)", True),
    "void0": ("adds a (void)0", True),
    "not-not": ("adds a double negation (!!)", False),
    "local-type-methods": ("adds a local struct or class with methods", False),
    "address-pun": ("adds a pointer cast of an address ((T*)&x: a type pun)", False),
}


def _close_brace(code: str, i: int) -> int:
    """The index of the `}` that closes the `{` at code[i] (len(code) when unbalanced)."""
    depth = 0
    for k in range(i, len(code)):
        depth += {"{": 1, "}": -1}.get(code[k], 0)
        if depth == 0:
            return k
    return len(code)


def span_devices(src: str, span: tuple[int, int] | None) -> list[tuple[str, int, int]]:
    """(kind, start, end) of every in-span device construct in src[span] (absolute offsets; comments and
    literals ignored): a preprocessor line, a constant-false `if`/`while` (not a do-while's), `x = x;`,
    identity arithmetic, `(void)0`, `!!`, a local struct/class with a method body, `(T*)&x`."""
    if not span:
        return []
    lo, hi = span
    code = code_only(src[:hi])
    out = [("preprocessor", m.start(), m.end()) for m in _PREPROCESSOR.finditer(code, lo, hi)]
    for m in _FALSE_COND.finditer(code, lo, hi):
        if m.group(0).startswith("while") and code[lo:m.start()].rstrip().endswith("}"):
            continue                                    # do { } while (0): an odd shape, flagged as one
        out.append(("false-condition", m.start(), m.end()))
    for m in _SELF_ASSIGN.finditer(code, lo, hi):
        before = code[lo:m.start()].rstrip()
        if "".join(m["l"].split()) == "".join(m["r"].split()) and (not before or before[-1] in ";{}"):
            out.append(("self-assign", m.start(), m.end()))
    out += [("identity", m.start(), m.end()) for m in _IDENTITY.finditer(code, lo, hi)]
    for m in _COMPOUND_IDENTITY.finditer(code, lo, hi):
        before = code[lo:m.start()].rstrip()
        if not before or before[-1] in ";{}":
            out.append(("identity", m.start(), m.end()))
    out += [("void0", m.start(), m.end()) for m in _VOID0.finditer(code, lo, hi)]
    out += [("not-not", m.start(), m.end()) for m in _NOTNOT.finditer(code, lo, hi)]
    out += [("address-pun", m.start(), m.end()) for m in _ADDRESS_PUN.finditer(code, lo, hi)]
    body = code.find("{", lo, hi)
    for m in _LOCAL_TYPE.finditer(code, body + 1 if body >= 0 else hi, hi):
        end = _close_brace(code, m.end() - 1)
        if re.search(r"\)\s*(?:const\s*)?\{", code[m.end():end]):
            out.append(("local-type-methods", m.start(), end + 1))
    return out


def span_device_flags(before_src: str, b_span, after_src: str, a_span) -> tuple[list[str], list[str]]:
    """(rejects, needs judgment): each in-span device kind the candidate's definition has more of than the
    source's (a new definition: any)."""
    have: dict[str, int] = {}
    for k, _a, _b in span_devices(before_src, b_span):
        have[k] = have.get(k, 0) + 1
    got: dict[str, int] = {}
    for k, _a, _b in span_devices(after_src, a_span):
        got[k] = got.get(k, 0) + 1
    reject, judgment = [], []
    for k, n in got.items():
        if n > have.get(k, 0):
            flag, rej = SPAN_DEVICES[k]
            (reject if rej else judgment).append(flag)
    return reject, judgment


def _inside(body: cbody.Stmt | None) -> dict[str, int]:
    """Empty statements (a for header's empty clauses aside) and comma operators (a for header's
    aside) in a body."""
    out = {"empty": 0, "comma": 0}
    if body is None:
        return out
    heads = {id(s.init) for s in cbody.statements(body) if s.kind == "for" and s.init is not None}
    for s in cbody.statements(body):
        if id(s) in heads:
            continue
        out["empty"] += s.kind == "empty"
        exprs = [s.expr] if s.kind == "for" else list(cbody.statement_exprs(s))
        args = {id(e) for d in (s.decl.declarators if s.decl is not None else []) if d.init_kind == "()"
                for e in _separators(d.init)}
        args |= {id(x) for top in exprs for e in cbody.expressions(top) if e.kind == "unary" and e.op == "new"
                 for k in e.kids for x in _separators(k)}
        out["comma"] += sum(1 for top in exprs for e in cbody.expressions(top) if e.kind == "comma" and id(e) not in args)
    return out


def _separators(e: cbody.Expr | None):
    """The commas between constructor or placement-new arguments (`XFile file(name, 0, mode)`,
    `new (0, "XContainer") XContainer`: the parser reads each list as one comma expression); not
    comma operators."""
    while e is not None and e.kind == "comma":
        yield e
        e = e.kids[0] if e.kids else None


def _parse(src: str, span) -> cbody.Stmt | None:
    try:
        brace = src.index("{", span[0], span[1])
        return cbody.parse_body(src, brace, cbody.type_names(src))
    except (ValueError, cbody.ParseError):
        return None


def review(before_src: str, after_src: str, symbol: str, forbidden: list[tuple[str, re.Pattern]] = (),
           probe=None) -> tuple[str, list[str]]:
    """(category, flags): `review_split`'s rejects and judgment flags as one verdict."""
    reject, judgment = review_split(before_src, after_src, symbol, forbidden, probe)
    if reject:
        return REJECT, reject + judgment
    return (JUDGMENT, judgment) if judgment else (CLEAN, [])


def exact_review(before_src: str, after_src: str, symbol: str, forbidden: list[tuple[str, re.Pattern]] = (),
                 probe=None, placeholders_ok: bool = False) -> tuple[list[str], str, list[str]]:
    """review() as the automated exact flag's second check (the judge alone let an invented inline helper
    through as a clean exact): (rejects that block the flag, category, flags). `placeholders_ok`: a
    placeholder-named new local does not block (`dk close` names its own temporaries `t0`), it stays in
    the flags and the category."""
    reject, judgment = review_split(before_src, after_src, symbol, forbidden, probe)
    blocking = [r for r in reject if not (placeholders_ok and "has a placeholder name" in r)]
    category = REJECT if reject else JUDGMENT if judgment else CLEAN
    return blocking, category, reject + judgment


def review_split(before_src: str, after_src: str, symbol: str, forbidden: list[tuple[str, re.Pattern]] = (),
                 probe=None) -> tuple[list[str], list[str]]:
    """Pre-review of a candidate that is already exact and judged: (category, flags). The judge's
    gates plus what a reviewer must see: edits outside the target's definition, new functions,
    macros, empty statements, comma operators, raw and offset casts, casts to the operand's own type
    (with `probe`, the unit's compiler: reject when removing one changes the code), aliasing locals,
    odd shapes. `volatile`, aliasing locals, cached member reads, casts without effect and
    declaration-only edits need judgment; any other flag rejects."""
    from .trycmd import _param_count, candidates, definition
    reject, judgment = [], []
    for r in source_reasons(before_src, after_src, forbidden):
        (judgment if "volatile" in r else reject).append(r)
    b_span, a_span = definition(before_src, symbol), definition(after_src, symbol)
    if b_span and a_span:                           # with the function unwritten, the one hit can be another
        (fb,), (fa,) = candidates(before_src, symbol), candidates(after_src, symbol)    # overload of it
        if _param_count(before_src, fb) != _param_count(after_src, fa):
            b_span = None
    if a_span is None:
        return reject + [f"{symbol}: no single definition in the candidate"], []
    b_def = before_src[b_span[0]:b_span[1]] if b_span else ""
    a_def = after_src[a_span[0]:a_span[1]]
    b_out = before_src[:b_span[0]] + before_src[b_span[1]:] if b_span else before_src
    # outside the definition: new functions, then every other changed line
    have = {f.name for f in cbody.functions(before_src)}
    cuts = [a_span]
    for f in cbody.functions(after_src):
        if f.name in have or (f.head_start, f.close + 1) == a_span:
            continue
        head = after_src[f.head_start:f.open]
        inline = re.search(r"\binline\b", head) is not None
        (judgment if inline else reject).append(f"{'new inline helper' if inline else 'adds function'} {f.name}")
        cuts.append((f.head_start, f.close + 1))
    a_out, last = "", 0
    for lo, hi in sorted(cuts):
        a_out, last = a_out + after_src[last:lo], max(last, hi)
    a_out += after_src[last:]
    if b_out.split() != a_out.split():
        import difflib
        changed = [x[1:] for x in difflib.unified_diff(b_out.splitlines(), a_out.splitlines(), lineterm="", n=0)
                   if x[:1] in "+-" and not x.startswith(("+++", "---")) and x[1:].strip()]
        macros = [x for x in changed if re.match(r"\s*#\s*define\b", x)]
        if macros:
            judgment.append(f"macro edits ({len(macros)} lines)")
        rest = [x for x in changed if x not in macros]
        # a unit-local `static inline` helper rewritten as a macro of the same name (T47 card 1): the removed
        # definition is the macro's own; every other function of the unit is still held to its score
        macro_names = {m.group(1) for x in macros for m in [re.match(r"\s*#\s*define\s+([A-Za-z_]\w*)\s*\(", x)] if m}
        gone = [x for x in rest if re.match(r"\s*static\s+inline\b.*\b(%s)\s*\(" % "|".join(map(re.escape, macro_names)), x)
                and x not in after_src.splitlines()] if macro_names else []
        if gone and len(gone) == len(rest):
            judgment.append(f"inline helper rewritten as a macro ({', '.join(sorted(macro_names))})")
            rest = []
        if rest and all(_DECL_LINE.match(x) for x in rest):
            judgment.append(f"declaration edits outside the function ({len(rest)} lines)")
        elif rest:
            reject.append(f"edits outside the target's definition ({len(rest)} lines)")
    # inside it
    a_body, b_body = _parse(after_src, a_span), _parse(before_src, b_span) if b_span else None
    a_n, b_n = _inside(a_body), _inside(b_body)
    if a_n["empty"] > b_n["empty"]:
        reject.append("adds an empty statement")
    if a_n["comma"] > b_n["comma"]:
        reject.append("adds a comma operator")
    if a_body is not None:
        empty = cbody.parse_body("{}", 0)
        for r in copy_only_locals(before_src if b_body is not None else "{}",
                                  b_body if b_body is not None else empty, after_src, a_body,
                                  parameters(after_src, a_span)):
            (judgment if " only caches " in r else reject).append(r)      # a named member read: a look
        judgment += aliasing_locals(b_body if b_body is not None else empty, after_src, a_body,
                                    parameters(after_src, a_span))
        reject += placeholder_locals(b_body if b_body is not None else empty, a_body)
        static = self_casts(before_src if b_body is not None else "{}", b_body if b_body is not None else empty,
                            after_src, a_body, parameters(after_src, a_span))
        if probe is None:                           # with the compiler, same_type_casts measures them
            judgment += static
        judgment += integer_casts(before_src if b_body is not None else "{}", b_body if b_body is not None else empty,
                                  after_src, a_body, parameters(after_src, a_span))
        judgment += copied_inlines(before_src, after_src, symbol)
        if probe is not None:
            bad, odd = same_type_casts(before_src if b_body is not None else "{}", b_body if b_body is not None else empty,
                                       after_src, a_body, probe)
            reject += bad
            judgment += odd
            measured = " ".join(bad + odd).replace(" ", "")     # a cast the probe could not check keeps its static flag
            judgment += [f for f in static if f.split("own type ", 1)[1].replace(" ", "") not in measured]
    rj, jd = span_device_flags(before_src, b_span, after_src, a_span)
    reject += rj
    judgment += jd
    if (offset_casts(after_src, a_body) if a_body is not None else _count(_OFFSET_CAST, a_def)) > \
            (offset_casts(before_src, b_body) if b_body is not None else _count(_OFFSET_CAST, b_def) if b_def else 0):
        reject.append("hard-coded offset cast")
    elif a_body is not None and offset_casts(after_src, a_body, True) > (offset_casts(before_src, b_body, True) if b_body else 0):
        judgment.append("byte-pointer arithmetic with a constant")
    if (n := aligned_decls(after_src)[1] - aligned_decls(before_src)[1]) > 0:
        judgment.append(f"adds {n} aligned stack local (the target may realign the frame: faithfulness.md, open)")
    if empty_cases(a_body) > empty_cases(b_body):
        judgment.append(f"empty case labels: {empty_cases(a_body) - empty_cases(b_body)} with no body of their own "
                        f"(only shape the compare tree)")
    if _count(_RAW_CAST, a_def) > _count(_RAW_CAST, b_def):
        judgment.append("adds a cast to a raw pointer type")
    for what, pat in (("goto", r"\bgoto\b"), ("do-while(0)", r"\bwhile\s*\(\s*0\s*\)"), ("switch case", r"\bcase\b")):
        added = len(re.findall(pat, a_def)) - len(re.findall(pat, b_def))
        if added > 0 and (what != "switch case" or b_def):
            judgment.append(f"odd shape: adds {added} {what}")
    return reject, judgment


def unit_probe(cfg, unit: dict, symbol: str, work):
    """`probe(text) -> the symbol's match percent, or None when it does not compile`, with the unit's
    compiler and flags in a private work directory."""
    import itertools
    import shutil
    from pathlib import Path
    from .matching import MatchError, score
    from .reorder import compile_variant
    count = itertools.count()

    def probe(text: str) -> float | None:
        d = Path(work) / f"probe{next(count)}"
        try:
            obj, _ = compile_variant(cfg, unit, text, d)
            if obj is None:
                return None
            try:
                s = score(cfg, unit["name"], cfg.root / unit["target_path"], obj)
            except MatchError:
                return None
            return {f.name: f.percent for f in s.functions}.get(symbol, 0.0)
        finally:
            shutil.rmtree(d, ignore_errors=True)
    return probe
