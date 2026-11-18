"""Register predictors for `dk regs --truth`: what a function's source says about the registers the
compiler gives its parameters and locals, before any compile.

Both predictors read one function's source (cbody's parse) and the names of its variables (debug
info order = declaration order) and return, per variable, `saved:rN`, `temp` (the named variable is
replaced by a compiler temporary), `volatile` (it does not live across a call, so no callee-saved
register) or None (not predicted: the rule needs information the source does not give cheaply).

(a) `rank_rule`: the existing rank rule (regs.py docstring): named locals in declaration order, then
    parameters last first; each takes the lowest callee-saved register already handed out that no
    interfering value holds, else the next one down from r31.
(b) `behaviour_rule`: the observable behaviour of the compiler's allocator (colour order: compiler
    temporaries newest first, then named locals in declaration order, then parameters last first;
    a call-defined local that feeds arithmetic and is passed bare only as the first argument of the
    next statement's call becomes a temporary; a call result's copy is placed after the next
    call's first argument move, so the value does not interfere with one that dies there; stack
    slots ascend by power-of-two size class, reverse declaration order inside a class).

Liveness is approximated from source positions: a value is live from where it is defined to its last
use (a use inside a call's arguments happens at the call), widened to the end of a loop that
contains both a use and an earlier definition; it crosses a call when a call ends strictly inside
that range.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ...core import cbody

SAVED_FIRST = 31
SIZES = {"char": 1, "u8": 1, "s8": 1, "bool": 1, "_Bool": 1, "short": 2, "u16": 2, "s16": 2, "int": 4, "u32": 4,
         "s32": 4, "float": 4, "f32": 4, "long": 4, "BOOL": 4, "size_t": 4, "double": 8, "f64": 8, "u64": 8,
         "s64": 8, "__int64": 8}
QUAL = {"const", "volatile", "unsigned", "signed", "static", "register", "struct", "class", "union", "enum"}


@dataclass
class Mention:
    name: str
    pos: int                 # source offset of the name
    use_at: int              # when the value is needed: the enclosing call's end for call arguments
    stmt: cbody.Stmt
    role: str                # "def" | "use"
    kind: str = "other"      # use: "arg" (bare call argument), "ret", "arith", "other"
    call: cbody.Expr | None = None
    arg: int = -1
    def_end: int = 0         # def: where the new value exists (end of the defining statement)
    def_call: bool = False   # def: the value comes straight from a call


@dataclass
class Value:
    name: str
    kind: str                # "param" | "local"
    order: int               # declaration index
    start: int = 0
    end: int = 0
    cross: bool = False
    temp: bool = False       # becomes a compiler temporary (behaviour rule)
    defined_by_call: bool = False
    free_of: set = field(default_factory=set)      # names this value does not interfere with (delayed copy)
    def_stmt: cbody.Stmt | None = None
    ambiguous: bool = False


def _strip(e: cbody.Expr | None) -> cbody.Expr | None:
    while e is not None and e.kind in ("paren", "cast") and e.kids:
        e = e.kids[0]
    return e


def _is_call(e: cbody.Expr | None) -> bool:
    e = _strip(e)
    return e is not None and e.kind == "call"


class Body:
    """One parsed function body: mentions per name, calls, loops, statements."""

    def __init__(self, src: str, open_brace: int, types):
        self.src = src
        self.root = cbody.parse_body(src, open_brace, types)
        self.parent = cbody.parents(self.root)
        self.calls: list[cbody.Expr] = []
        self.loops: list[cbody.Stmt] = []
        self.mentions: dict[str, list[Mention]] = {}
        self.decls: dict[str, tuple[cbody.Decl, cbody.Declarator]] = {}
        self.exprs: list[cbody.Expr] = []
        self.call_stmt: dict[int, cbody.Stmt] = {}      # id(call) -> the statement holding it
        for s in cbody.statements(self.root):
            if s.kind in ("for", "while", "do"):
                self.loops.append(s)
            if s.decl is not None:
                for d in s.decl.declarators:
                    if d.name in self.decls:
                        self.decls[d.name] = None          # redeclared in another block: ambiguous
                    else:
                        self.decls[d.name] = (s.decl, d)
                    if d.init is not None:
                        m = Mention(d.name, d.name_start, d.end, s, "def", def_end=d.end, def_call=_is_call(d.init))
                        self.mentions.setdefault(d.name, []).append(m)
                    elif d.init_kind == "()":
                        pass
            for e in cbody.statement_exprs(s):
                self._visit(e, None, None, -1, s, None)
            if s.kind == "return" and s.expr is not None:
                inner = _strip(s.expr)
                if inner is not None and inner.kind == "name":
                    for m in self.mentions.get(inner.name, []):
                        if m.pos == inner.start:
                            m.kind = "ret"
        for ms in self.mentions.values():
            ms.sort(key=lambda m: m.pos)

    def _visit(self, e, parent, call, arg, stmt, top):
        self.exprs.append(e)
        if e.kind == "call":
            self.calls.append(e)
            self.call_stmt[id(e)] = stmt
            self._visit(e.kids[0], e, e, -1, stmt, top)
            for i, a in enumerate(e.kids[1:]):
                self._visit(a, e, e, i, stmt, top)
            return
        if e.kind == "name":
            kind = "other"
            use_at = call.end if call is not None else e.start
            if parent is not None and parent.kind == "call" and arg >= 0:
                kind = "arg"
            elif parent is not None and (parent.kind in ("binary", "unary") and parent.op not in ("*", "&", "new", "delete")
                                         or (parent.kind == "assign" and parent.kids[1] is e)):
                kind = "arith"
            self.mentions.setdefault(e.name, []).append(
                Mention(e.name, e.start, use_at, stmt, "use", kind, call if kind == "arg" else None, arg))
            return
        if e.kind == "assign" and e.kids[0].kind == "name":
            left, right = e.kids
            if e.op == "=":
                self._visit(right, e, call, arg, stmt, top)
                m = Mention(left.name, left.start, left.start, stmt, "def", def_end=stmt.end, def_call=_is_call(right))
                self.mentions.setdefault(left.name, []).append(m)
                self.exprs += [e, left]               # an unqualified member written is an implicit `this` use
                return
            self.mentions.setdefault(left.name, []).append(
                Mention(left.name, left.start, call.end if call else left.start, stmt, "use", "arith"))
            self._visit(right, e, call, arg, stmt, top)
            return
        for k in e.kids:
            self._visit(k, e, call, arg if e.kind == "paren" else -1, stmt, top)


def callee(c: cbody.Expr) -> str:
    f = c.kids[0] if c.kids else None
    while f is not None and f.kind == "paren" and f.kids:
        f = f.kids[0]
    return f.name if f is not None and f.kind in ("name", "member") else ""


def real_calls(body: Body, inline: frozenset | set = frozenset()) -> list[cbody.Expr]:
    """The calls that stay calls: not those to functions declared `inline` (their bodies are copied in)."""
    return [c for c in body.calls if callee(c) not in inline] if inline else body.calls


def values(body: Body, variables: list[tuple[str, str]], inline: frozenset | set = frozenset()) -> list[Value]:
    """Live range and call crossing per variable [(name, kind)] in declaration order (calls to `inline`
    callees do not count)."""
    calls = real_calls(body, inline)
    names = [n for n, _ in variables]
    out: list[Value] = []
    for i, (name, kind) in enumerate(variables):
        v = Value(name, kind, i, ambiguous=names.count(name) != 1 or (kind == "local" and body.decls.get(name, 0) is None)
                  or (kind == "local" and name not in body.decls))
        ms = body.mentions.get(name, [])
        if kind == "param":
            v.start = -1
        elif ms:
            first = ms[0]
            v.start = first.def_end if first.role == "def" else first.pos
            if first.role == "def":
                v.def_stmt = first.stmt
                v.defined_by_call = first.def_call
        uses = [m for m in ms if m.role == "use"]
        last = max((m.use_at for m in uses), default=v.start)
        loops = []
        for loop in body.loops:
            inside = [m for m in ms if loop.start <= m.pos < loop.end]
            if inside and (v.start < loop.start or (loop.init is not None and ms[0].stmt is loop.init)):
                last = max(last, loop.end)
                loops.append(loop)
        v.end = max(last, v.start)
        v.cross = any(v.start < c.end < v.end and (any(lp.start <= c.start < lp.end for lp in loops)
                                                    or any(m.use_at > c.end and reaches(body, c, m.stmt) for m in uses))
                      for c in calls)
        out.append(v)
    return out


JUMPS = ("return", "break", "continue", "goto")


def reaches(body: Body, call: cbody.Expr, target: cbody.Stmt) -> bool:
    """Whether control can go from `call` on to the statement `target` (later in the source) without a
    loop's back edge: not when `call` sits in the other branch of an `if` that holds `target`, nor
    when a jump statement follows `call` in a block that `target` lies after."""
    a = body.call_stmt.get(id(call))
    if a is None or a is target:
        return True
    inside = lambda s: s.start <= target.start and target.end <= s.end
    if a.kind in JUMPS and not inside(a):
        return False                                   # `return f(x);`: nothing after it runs
    while id(a) in body.parent and not inside(a):
        holder, idx = body.parent[id(a)]
        if holder.kind == "if" and idx == 0 and len(holder.kids) == 2 and inside(holder.kids[1]):
            return False
        if holder.kind == "block":
            for sib in holder.kids[idx + 1:]:
                if inside(sib):
                    break
                if sib.kind in ("return", "goto"):
                    return False
                if sib.kind in ("break", "continue"):          # leaves the loop (or switch): stops only inside it
                    out = _escape(body, sib, ("for", "while", "do") + (("switch",) if sib.kind == "break" else ()))
                    if out is None or inside(out):
                        return False
                    break
        a = holder
    return True


def _escape(body: Body, s: cbody.Stmt, kinds: tuple[str, ...]) -> cbody.Stmt | None:
    """The innermost statement of one of `kinds` holding s."""
    while id(s) in body.parent:
        s = body.parent[id(s)][0]
        if s.kind in kinds:
            return s
    return None


IMPLICIT_NOT = re.compile(r"^(?:[A-Z][A-Z_0-9]*|k[A-Z]\w*|g[A-Z]\w*|fn_\w+|lbl_\w+|NULL|true|false)$")


def implicit_this_end(body: Body, declared: set[str]) -> int | None:
    """Where `this` is last needed: its explicit uses, an unqualified call (a member's implicit this) at its
    end, or an unqualified name that is no variable (a member read). None: never."""
    end = None
    callee_ids = {id(c.kids[0]): c for c in body.calls}
    for e in body.exprs:
        if e.kind != "name" or e.name in declared or IMPLICIT_NOT.match(e.name):
            continue
        at = callee_ids[id(e)].end if id(e) in callee_ids else e.end
        end = at if end is None else max(end, at)
    for m in body.mentions.get("this", []):
        end = m.use_at if end is None else max(end, m.use_at)
    return end


def special_members(symbol: str, vals: list[Value], body_end: int) -> None:
    """Constructors and destructors: `this` is returned, so it lives to the end, and the calls the
    source does not show (base and member constructors first, member and base destructors and
    `operator delete` last) make it cross one. A destructor's hidden last parameter (the delete
    flag) lives as long and is coloured first (parameters go last first), so `this` takes the
    register after it. The flag has no debug entry: it is appended (callers zip by position)."""
    role = "ct" if symbol.startswith("__ct__") else "dt" if symbol.startswith("__dt__") else None
    this = next((v for v in vals if v.name == "this" and v.kind == "param"), None)
    if role is None or this is None or this.ambiguous:
        return
    this.end, this.cross = max(this.end, body_end), True
    if role == "dt":
        vals.append(Value("<delete flag>", "param", len(vals), start=-1, end=body_end, cross=True))


def next_statement(body: Body, s: cbody.Stmt | None) -> cbody.Stmt | None:
    if s is None or id(s) not in body.parent:
        return None
    holder, idx = body.parent[id(s)]
    if holder.kind == "block" and idx + 1 < len(holder.kids):
        return holder.kids[idx + 1]
    return None


def becomes_temp(body: Body, v: Value) -> bool:
    """The behaviour rule's local-becomes-temporary condition."""
    if v.kind != "local" or v.ambiguous or not v.defined_by_call or v.def_stmt is None:
        return False
    ms = body.mentions.get(v.name, [])
    if sum(1 for m in ms if m.role == "def") != 1:
        return False
    nxt = next_statement(body, v.def_stmt)
    arith = 0
    for m in ms:
        if m.role != "use":
            continue
        if m.kind == "arith":
            arith += 1
        elif m.kind == "ret":
            return False
        elif m.kind == "arg":
            if m.arg != 0 or nxt is None or not (nxt.start <= m.pos < nxt.end):
                return False
    return arith >= 1


def delayed_copy(body: Body, vals: list[Value]) -> None:
    """A call result's copy comes after the next call's first argument move (after its second when the
    value itself is that first argument): values dying in that move do not interfere with it."""
    by = {v.name: v for v in vals}
    for v in vals:
        if v.kind != "local" or not v.defined_by_call or v.temp:
            continue
        after = [c for c in body.calls if c.start >= v.start]
        if not after:
            continue
        nxt = min(after, key=lambda c: c.start)
        v_arg = next((m.arg for m in body.mentions.get(v.name, []) if m.call is nxt and m.role == "use"), None)
        slot = 1 if v_arg == 0 else 0
        for w in vals:
            if w is v:
                continue
            uses = [m for m in body.mentions.get(w.name, []) if m.role == "use"]
            last = max(uses, key=lambda m: m.use_at, default=None)
            if last is not None and last.call is nxt and last.arg == slot and last.use_at == w.end:
                v.free_of.add(w.name)


def interferes(a: Value, b: Value) -> bool:
    if b.name in a.free_of or a.name in b.free_of:
        return False
    return a.start < b.end and b.start < a.end


def colour(order: list[Value], bank: dict[str, str]) -> dict[str, str]:
    """Walk `order`: each call-crossing value takes the lowest register already handed out (per
    register class) that no interfering coloured value holds, else the next one down from 31."""
    got: dict[str, str] = {}
    obtained: dict[str, list[int]] = {"r": [], "f": []}
    coloured: list[Value] = []
    for v in order:
        cls = bank.get(v.name, "r")
        busy = {got[w.name] for w in coloured if w.name in got and got[w.name][0] == cls and interferes(v, w)}
        free = [n for n in sorted(obtained[cls]) if f"{cls}{n}" not in busy]
        if free:
            n = free[0]
        else:
            n = SAVED_FIRST - len(obtained[cls])
            if n < 14:
                coloured.append(v)
                continue
            obtained[cls].append(n)
        got[v.name] = f"{cls}{n}"
        coloured.append(v)
    return got


def rank_rule(vals: list[Value], bank: dict[str, str]) -> dict[str, str | None]:
    """Predictor (a)."""
    cand = [v for v in vals if v.cross and not v.ambiguous]
    order = [v for v in cand if v.kind == "local"] + [v for v in reversed(cand) if v.kind == "param"]
    got = colour(order, bank)
    out: dict[str, str | None] = {}
    for v in vals:
        out[v.name] = None if v.ambiguous else (f"saved:{got[v.name]}" if v.name in got else "volatile")
    return out


def behaviour_rule(body: Body, vals: list[Value], bank: dict[str, str], unnamed_temps: int | None) -> tuple[dict, str]:
    """Predictor (b): (variable -> prediction, note). With more unexplained temporaries in the
    compiled function than the rule creates (`unnamed_temps` counts callee-saved registers no named
    variable holds), the variable predictions are left None."""
    for v in vals:
        v.temp = becomes_temp(body, v)
    delayed_copy(body, vals)
    cand = [v for v in vals if v.cross and not v.ambiguous]
    temps = sorted((v for v in cand if v.temp), key=lambda v: -v.start)
    locals_ = [v for v in cand if v.kind == "local" and not v.temp]
    params = [v for v in reversed(cand) if v.kind == "param"]
    got = colour(temps + locals_ + params, bank)
    out: dict[str, str | None] = {}
    for v in vals:
        if v.ambiguous:
            out[v.name] = None
        elif v.temp and v.cross:
            out[v.name] = "temp"
        elif v.name in got:
            out[v.name] = f"saved:{got[v.name]}"
        else:
            out[v.name] = "volatile"
    note = ""
    if unnamed_temps is not None and unnamed_temps > len(temps):
        note = f"{unnamed_temps - len(temps)} callee-saved registers belong to temporaries the source does not show"
        out = {k: (None if p != "volatile" else p) for k, p in out.items()}
    return out, note


# ---- stack slots (behaviour rule) --------------------------------------------------------------------

def type_text(body: Body, name: str) -> str:
    entry = body.decls.get(name)
    if not entry:
        return ""
    decl, d = entry
    return f"{decl.base_type(body.src)}{d.ptr}{d.array}"


def type_size(body: Body, name: str) -> int | None:
    entry = body.decls.get(name)
    if not entry:
        return None
    decl, d = entry
    words = [w for w in re.findall(r"[A-Za-z_]\w*", decl.base_type(body.src)) if w not in QUAL]
    ptr = d.ptr.replace("const", "")
    if ptr and set(ptr) != {"*"}:
        return None                       # a reference, or something odd
    if ptr:
        base = 4
    elif len(words) == 1 and words[0] in SIZES:
        base = SIZES[words[0]]
    elif words and all(w in SIZES for w in words):
        base = max(SIZES[w] for w in words)
    else:
        return None
    for n in re.findall(r"\[\s*(0[xX][0-9a-fA-F]+|\d+)\s*\]", d.array):
        base *= int(n, 0)
    if d.array and not re.findall(r"\[\s*(?:0[xX][0-9a-fA-F]+|\d+)\s*\]", d.array):
        return None
    return base


def size_class(n: int) -> int:
    c = 1
    while c < n and c < 512:
        c *= 2
    return c if n <= 512 else 1 << 20


def stack_order(body: Body, stack_names: list[str], decl_index: dict[str, int],
                known: dict[str, int | None] | None = None) -> list[str] | None:
    """Predicted ascending-offset order of the named stack variables, or None if a size is unknown
    (`known`: sizes from the debug info, tried before the declaration's type text)."""
    sized = []
    for n in stack_names:
        sz = (known or {}).get(n) or type_size(body, n)
        if sz is None:
            return None
        sized.append((size_class(sz), -decl_index[n], n))
    return [n for _, _, n in sorted(sized)]
