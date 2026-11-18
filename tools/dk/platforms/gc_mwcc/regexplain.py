"""The register explainer: why each variable of a near miss has its register, and which faithful
source edits are expected to move it, read from a compile (never predicted from the source).

The closer compiles a candidate with `-sym on` (code unchanged), so the debug info names every
variable's register exactly; objdiff's alignment gives each of our registers its target register
(`regs.register_map`). The explainer lays the callee-saved values out in the allocator's colour
order (the playbook's rule: compiler temporaries newest first, then named locals in declaration
order, then parameters last first; each takes the lowest register already handed out that no
interfering value holds, else the next one down from r31): our position, the target's, and
whether the variable has to be coloured earlier or later.

The edits (signatures are fixed, so parameters never move):
- `decl-order`: a named local declared before (or after) the local that holds its target
  register; hoisted to the top when it must outrank every local;
- `temp-regime`: a local the target ranks above every named local is a compiler temporary there:
  `T x; x = e;` joined into `T x = e;`, or a single-use local folded into its one use; a local
  ours ranks as a temporary that the target ranks among the named locals is split (`T x; x = e;`);
- `name-temp`: an unnamed temporary the target colours after a named local was a named local there:
  each operand of the line that first writes it named in a local (`__typeof__`, typed from the debug
  info before the judge), when that keeps the evaluation order;
- `statement-order`: independent neighbouring statements swapped where the variable (or an
  unnamed temporary) is used: that moves where a temporary is created and which values overlap;
- `param-reuse`: a local the target colours after every parameter (a register below theirs) may be a
  parameter the original reused once its first value was dead (`p = f(p);` for `T x = f(p);`), when
  the types match and the parameter is not read after the local's definition;
- `inline-helper`: a value the target ranks below every parameter was an inlined helper's local.

Each move carries the variable it is for and a score (how many of our register operands the
variable accounts for, the kind's prior): `dk regs --explain` prints them, `dk close --moves truth`
compiles them first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ...core import cbody
from . import rewrites
from .ranksolve import _target, reorder_edits, saved, single_decls

PRIOR = {"decl-order": 3.0, "param-reuse": 2.5, "temp-regime": 2.0, "name-temp": 1.8, "inline-helper": 1.5,
         "statement-order": 1.0}
NO_DEF = ("st", "cmp", "b", "mt", "tw", "dcb", "icb", "sync", "isync")


@dataclass
class VarRow:
    name: str
    kind: str                     # "param", "local" or "temp" (a register no variable holds)
    ours: str
    target: str | None            # None: ours maps to several target registers
    position: int | None          # our colour position in the callee-saved bank (0 = r31), None: volatile
    target_position: int | None
    tier: str                     # "temporary", "local #k" (declaration order), "param #k" (last first)
    uses: int = 0                 # register operands of ours this value accounts for

    @property
    def status(self) -> str:
        if self.target is None:
            return "unmapped"
        if self.target == self.ours:
            return "ok"
        if saved(self.ours) != saved(self.target):
            return "bank"
        return "number"

    @property
    def want(self) -> str:
        if self.position is None or self.target_position is None or self.position == self.target_position:
            return ""
        return "earlier" if self.target_position < self.position else "later"


@dataclass
class Move:
    kind: str
    var: str
    label: str
    source: str
    why: str
    score: float = 0.0


@dataclass
class Explanation:
    rows: list[VarRow] = field(default_factory=list)
    moves: list[Move] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _bank(reg: str | None, cls: str) -> bool:
    return bool(reg) and reg[0] == cls and reg[1:].isdigit() and reg not in ("r1", "r2", "r13")


def _pos(reg: str | None) -> int | None:
    return 31 - int(reg[1:]) if reg and reg[1:].isdigit() and saved(reg) else None


def rows(dwarf, pairs: dict) -> list[VarRow]:
    """Every variable (and unnamed temporary) in a register, ours vs target, in our colour order."""
    if dwarf is None:
        return []
    out: list[VarRow] = []
    entry = getattr(pairs, "entry", {})
    regd = [v for v in dwarf.vars if v.reg and v.reg != "stack" and (v.reg[0] in "rf") and v.reg[1:].isdigit()]
    locals_ = [v for v in regd if v.kind == "local"]
    params = [v for v in regd if v.kind == "param"]
    held = {v.reg for v in regd}
    for v in regd:
        t = entry.get(v.reg) if v.kind == "param" and v.reg in entry else _target(pairs, v.reg)
        if v.kind == "local":
            tier = f"local #{locals_.index(v)}"
        else:
            tier = f"param #{len(params) - 1 - params.index(v)} (last first)"
        out.append(VarRow(v.name, v.kind, v.reg, t, _pos(v.reg), _pos(t), tier, sum(pairs.get(v.reg, {}).values())))
    for reg in sorted(pairs, key=lambda r: (r[0], -int(r[1:]) if r[1:].isdigit() else 0)):
        if reg in held or not reg[1:].isdigit() or not saved(reg) or reg in ("r1", "r2", "r13"):
            continue
        t = _target(pairs, reg)
        out.append(VarRow(f"<temp {reg}>", "temp", reg, t, _pos(reg), _pos(t), "temporary", sum(pairs[reg].values())))
    out.sort(key=lambda r: (r.ours[0], r.position if r.position is not None else 99, r.ours))
    return out


def _lines_of(site: rewrites.Site, name: str) -> set[int]:
    """0-based source lines inside the body that mention `name`."""
    out = set()
    for m in re.finditer(rf"\b{re.escape(name)}\b", site.src[site.body.start:site.body.end]):
        out.add(site.line(site.body.start + m.start()))
    return out


def _emit(site: rewrites.Site, edits, seen: set[str]) -> str | None:
    if not edits:
        return None
    try:
        new = cbody.apply_edits(site.src, edits)
    except ValueError:
        return None
    if new in seen:
        return None
    seen.add(new)
    return new


def _join_init(site: rewrites.Site, name: str):
    """`T x; x = e;` (adjacent) joined into `T x = e;`."""
    decls = single_decls(site)
    s = decls.get(name)
    if s is None or s.decl.declarators[0].init is not None:
        return None
    block, k = site.parents[id(s)]
    if k + 1 >= len(block.kids):
        return None
    nxt = block.kids[k + 1]
    e = nxt.expr if nxt.kind == "expr" else None
    if e is None or e.kind != "assign" or e.op != "=" or e.kids[0].kind != "name" or e.kids[0].name != name:
        return None
    d = s.decl.declarators[0]
    head = site.src[s.decl.type_start:d.name_end] + d.array
    return [(s.start, nxt.end, f"{head} = {site.text(e.kids[1])};")]


def _fold_single_use(site: rewrites.Site, name: str):
    """`T x = e; S(x);` with x used once, in the next statement, and no other call there: `S(e);`."""
    s = single_decls(site).get(name)
    if s is None:
        return None
    d = s.decl.declarators[0]
    if d.init is None or d.init_kind != "=" or d.ptr.count("&") or d.array:
        return None
    block, k = site.parents[id(s)]
    if k + 1 >= len(block.kids):
        return None
    nxt = block.kids[k + 1]
    if nxt.kind not in ("expr", "return", "decl"):
        return None
    uses = [m for m in re.finditer(rf"\b{re.escape(name)}\b", site.src[site.body.start:site.body.end])]
    if len(uses) != 2:                               # its declaration and the one use
        return None
    at = site.body.start + uses[1].start()
    if not (nxt.start <= at < nxt.end):
        return None
    calls = [x for top in cbody.statement_exprs(nxt) for x in cbody.expressions(top) if x.kind == "call"]
    others = [c for c in calls if not (c.start <= at < c.end)]        # calls that do not take x (evaluation order)
    if others and any(x.kind == "call" for x in cbody.expressions(d.init)):
        return None
    init = site.text(d.init)
    if d.init.kind not in ("name", "num", "call", "member", "index", "paren"):
        init = f"({init})"
    block_end = block.kids[k + 1].start
    return [(s.start, block_end, ""), (at, at + len(name), init)]


def _split_init(site: rewrites.Site, name: str):
    s = single_decls(site).get(name)
    if s is None:
        return None
    d = s.decl.declarators[0]
    if d.init_kind != "=" or "&" in d.ptr or "const" in site.src[s.decl.type_start:s.decl.type_end].split() or d.array:
        return None
    ind = cbody.indent_at(site.src, s.start)
    return [(s.start, s.end, site.src[s.decl.type_start:d.name_end] + f";\n{ind}{d.name} = {site.text(d.init)};")]


def _type_key(text: str) -> str:
    return re.sub(r"\s+", "", re.sub(r"\b(?:register|static)\b", "", text))


def _param_reuse(site: rewrites.Site, name: str):
    """(label, edits) that write the local `name` as a parameter of the same type reused: the declaration
    becomes an assignment to the parameter and every later mention is renamed. Only when the parameter
    is not mentioned after the declaration (its old value is dead) and is never address-taken."""
    from .inlinehelper import parameters, span_of
    s = single_decls(site).get(name)
    span = span_of(site)
    if s is None or span is None:
        return
    d = s.decl.declarators[0]
    if d.init is None or d.init_kind != "=" or d.array or "&" in d.ptr:
        return
    local_type = _type_key(site.src[s.decl.type_start:s.decl.type_end] + d.ptr)
    body = site.src[site.body.start:site.body.end]
    for p in parameters(site.src, span) or []:
        if _type_key(p.decl[:p.name_at]) != local_type or site.text(d.init).strip() == p.name:
            continue
        later = [m.start() + site.body.start for m in re.finditer(rf"\b{re.escape(p.name)}\b", body)
                 if m.start() + site.body.start >= d.init.end]
        if later or re.search(rf"&\s*{re.escape(p.name)}\b", body):
            continue
        edits = [(s.start, s.end, f"{p.name} = {site.text(d.init)};")]
        for m in re.finditer(rf"\b{re.escape(name)}\b", body):
            at = m.start() + site.body.start
            if at >= s.end:
                edits.append((at, at + len(name), p.name))
        yield f"{name} written as the parameter {p.name} reused", edits


def def_lines(ours: list[dict]) -> dict[str, int]:
    """Our register -> the source line (1-based) of the first instruction that writes it."""
    from .regs import _mnemonic, _regs
    out: dict[str, int] = {}
    for o in ours:
        m = _mnemonic(o)
        line = o.get("instruction", {}).get("line_number")
        if not m or not line or m.startswith(NO_DEF):
            continue
        regs = _regs(o)
        if regs and regs[0] not in out:
            out[regs[0]] = line
    return out


def _name_operand(site: rewrites.Site, line: int):
    """(label, edits) naming one operand (a call, load or arithmetic) of the statement at 0-based `line`
    in a new local, declared right before it."""
    stmts = [x for x in cbody.statements(site.body) if x.kind in ("expr", "decl", "return")
             and site.line(x.start) <= line <= site.line(max(x.start, x.end - 1)) and site.in_block(x)]
    for st in stmts[:1]:
        tops = list(cbody.statement_exprs(st))
        if st.decl is not None:
            tops = [d.init for d in st.decl.declarators if d.init is not None]
        nodes = [x for t in tops for x in cbody.expressions(t)]
        calls = [x for x in nodes if x.kind == "call"]
        for e in nodes:
            if e.kind not in ("call", "member", "index", "binary") or any(e is t for t in tops):
                continue
            if e.kind == "binary" and e.op in ("&&", "||", ","):
                continue
            if any(c.start < e.start and not (c.start <= e.start and e.end <= c.end) for c in calls):
                continue                                 # a call evaluated before it would move after it
            name = site.fresh("t")
            text = site.text(e)
            if site.cplusplus:
                decl = site.declare(st, f"__typeof__({text}) {name} = {text}")
            else:
                decl = site.declare(st, f"__typeof__({text}) {name}")
                if decl is not None:
                    decl += [(st.start, st.start, f"{name} = {text};\n{cbody.indent_at(site.src, st.start)}")]
            site.taken.discard(name)                     # each candidate names one operand: t0 again
            if decl is None:
                continue
            yield f"named {rewrites._short(text)} as {name}", decl + [(e.start, e.end, name)]


def _rule_moves(site: rewrites.Site, rule: str, focus: set[int], prefer: frozenset = frozenset()):
    """(label, source) of a table rule, focused on the given lines."""
    old = (site.focus, site.prefer)
    site.focus, site.prefer = (focus or None), (prefer or site.prefer)
    try:
        for _name, label, new in rewrites.candidates(site, [rewrites.BY_NAME[rule]], groups=False):
            yield label, new
    finally:
        site.focus, site.prefer = old


def explain(site: rewrites.Site | None, dwarf, pairs: dict, limit: int = 12, defs: dict[str, int] | None = None) -> Explanation:
    """Rows ours vs target and the ranked moves for every mismatched variable (at most `limit` per
    variable)."""
    ex = Explanation(rows(dwarf, pairs))
    if site is None:
        ex.notes.append("the function's body could not be read: no edits")
        return ex
    seen: set[str] = {site.src}
    decls = single_decls(site)
    by_target = {r.target: r for r in ex.rows if r.target and r.status != "unmapped"}
    by_ours = {r.ours: r for r in ex.rows}
    named_target = [r.target_position for r in ex.rows if r.kind == "local" and r.target_position is not None]
    params_target = [r.target_position for r in ex.rows if r.kind == "param" and r.target_position is not None]
    lowest_param = max(params_target) if params_target else None
    for r in ex.rows:
        if r.status in ("ok", "unmapped"):
            continue
        weight = max(1, r.uses)
        found: list[Move] = []

        def add(kind, label, edits_or_src, why):
            if isinstance(edits_or_src, str):
                if edits_or_src in seen:
                    return
                new = edits_or_src
                seen.add(new)
            else:
                new = _emit(site, edits_or_src, seen)
                if new is None:
                    return
            found.append(Move(kind, r.name, label, new, why, PRIOR[kind] * weight))

        if r.kind == "local" and r.name in decls:
            holder = by_ours.get(r.target) if r.target else None
            other = holder.name if holder is not None and holder.kind == "local" and holder.name in decls else None
            if r.want == "earlier":
                if other:
                    add("decl-order", f"{r.name} declared before {other}", reorder_edits(site, [r.name, other]),
                        f"{other} holds {r.target} in ours; locals rank in declaration order")
                rest = [x for x in decls if x != r.name]
                if rest:
                    add("decl-order", f"{r.name} declared first", reorder_edits(site, [r.name] + rest[:1], hoist=True),
                        "the target colours it before the other locals")
                if r.target_position is not None and named_target and r.target_position <= min(named_target):
                    add("temp-regime", f"{r.name} initialized where declared", _join_init(site, r.name),
                        "the target ranks it above every named local: there it is a compiler temporary")
                    add("temp-regime", f"{r.name} folded into its one use", _fold_single_use(site, r.name),
                        "the target ranks it above every named local: there it is a compiler temporary")
            elif r.want == "later":
                if other:
                    add("decl-order", f"{other} declared before {r.name}", reorder_edits(site, [other, r.name]),
                        f"{other} holds {r.target} in ours; locals rank in declaration order")
                add("temp-regime", f"{r.name} declared, then assigned", _split_init(site, r.name),
                    "ours may rank it as a temporary (merged with the value that made it)")
        if (r.kind == "local" and r.target_position is not None and lowest_param is not None
                and r.target_position > lowest_param):
            for label, edits in _param_reuse(site, r.name):
                add("param-reuse", label, edits, "the target colours it after every parameter: a parameter reused")
        if (r.target_position is not None and lowest_param is not None and r.kind != "param"
                and r.target_position > lowest_param):
            for label, new in _rule_moves(site, "inline-helper", _lines_of(site, r.name) if r.kind == "local" else set(),
                                          frozenset([r.name]) if r.kind == "local" else frozenset()):
                add("inline-helper", label, new, "the target ranks it below every parameter: an inlined helper's local")
                if sum(1 for m in found if m.kind == "inline-helper") >= 3:
                    break
        if (r.kind == "temp" and defs and r.ours in defs and r.target_position is not None and named_target
                and r.target_position > min(named_target)):
            n = 0
            for label, edits in _name_operand(site, defs[r.ours] - 1):
                add("name-temp", label, edits, "the target colours it after a named local: a named local there")
                n += 1
                if n >= 4:
                    break
        focus = _lines_of(site, r.name) if r.kind != "temp" else set()
        if r.kind == "temp" or focus:
            n = 0
            for label, new in _rule_moves(site, "statement-order", focus):
                add("statement-order", label, new, "moves where a temporary is created and which values overlap")
                n += 1
                if n >= 4:
                    break
        found.sort(key=lambda m: -PRIOR[m.kind])
        ex.moves += found[:limit]
        if not found:
            ex.notes.append(f"{r.name}: no faithful edit known for a {r.status} miss ({r.ours} -> {r.target})")
    ex.moves.sort(key=lambda m: -m.score)
    return ex


def render(ex: Explanation, top: int = 12) -> list[str]:
    out = ["  value         tier                     ours  target  colour position ours -> target"]
    for r in ex.rows:
        pos = f"{r.position} -> {r.target_position}" if r.position is not None or r.target_position is not None else "volatile"
        mark = "  " if r.status == "ok" else "!="
        out.append(f"{mark}{r.name:<13} {r.tier:<24} {r.ours:<5} {str(r.target or '?'):<7} {pos}"
                   + (f"  ({r.status}, {r.want})" if r.status not in ("ok",) and r.want else
                      (f"  ({r.status})" if r.status != "ok" else "")))
    if ex.moves:
        out.append(f"  faithful edits, best first ({len(ex.moves)}):")
        for k, m in enumerate(ex.moves[:top], 1):
            out.append(f"   {k:>2}. [{m.kind}] {m.label}  (for {m.var}: {m.why})")
    out += [f"  note: {n}" for n in ex.notes]
    return out
