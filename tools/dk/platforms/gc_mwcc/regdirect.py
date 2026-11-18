"""Register-diff-directed moves for `dk close --moves directed`.

A near miss whose diff rows are register operands only gets one of three diagnoses from the compiler
oracle (oracle.py, contract v2/v3), and each diagnosis names the source lines and values involved:

- `order`: the target's colouring is reachable by creation order; the solver's flipped rules say which
  pair must flip. Each rule's cell (whytemp.py) picks the moves that can flip exactly that pair:
  local/local → declaration order; temp vs local/param → fold, join, split or name the local, name the
  temp; temp/temp in one phase → operand and statement order at the two lines; temp/temp across phases
  → no move exists (STOP), so no compile is spent.
- `stuck`: not reachable by any order, so the interference itself must change: moves that remove or
  add a value at the wrong values' lines (fold a single-use local, inline a local, turn a unit-local
  `static inline` helper into a macro so its argument is evaluated twice and CSE'd, join `v = A; v op= B;`).
- `sched`: every value already has the target's register, so the rows are scheduling: the loop rows
  (`for-init-list`, `iv-alias`), statement order, chained stores and the join at the differing lines.

The table rows are reused through regexplain's focused generators; the two rows that exist only here
(`helper-to-macro`, `join-op-assign`) are the moves the two landed register-only fixes needed (T47 cards
1 and 3) and stay out of `rewrites.RULES` (the LoRA prompt lists that table). Everything is pure source
→ source; close compiles and scores each candidate as in `--moves oracle`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ...core import cbody
from . import rewrites
from .rewrites import Site

ORDER_RULES_SAME_PHASE = ("operand-swap", "reassociate", "statement-order", "op-assign", "half")
ORDER_RULES_CODE = ("operand-swap", "reassociate", "statement-order", "chained-assign", "ref-element")
STUCK_RULES = ("name-value", "inline-local", "statement-order", "split-init", "local-reuse")
SCHED_RULES = ("for-init-list", "iv-alias", "statement-order", "chained-assign", "ref-element", "unsigned-index",
               "name-value")
CROSS_TIER_TEMP_RULES = ("name-value", "inline-local", "operand-swap")

_PHASES = ("front", "linear", "prepass", "prop0", "unroll", "loop", "cse0", "prop1", "cse1", "late", "backend")


@dataclass
class RuleView:
    cls: str
    a: dict
    b: dict
    cell: str            # decl-order | cross-tier | same-phase | cross-phase | code-eval | param | front-inline
    lines: set[int] = field(default_factory=set)      # 0-based source lines of the two values


@dataclass
class Diagnosis:
    kind: str                                   # order | stuck | sched | none
    rules: list[RuleView] = field(default_factory=list)
    wrong_lines: set[int] = field(default_factory=set)     # 0-based lines of the values in another register
    stop: list[str] = field(default_factory=list)          # rules with no faithful lever (reported, not compiled)

    @property
    def open_lines(self) -> set[int]:
        return {ln for r in self.rules if r.cell != "cross-phase" for ln in r.lines} | self.wrong_lines


def _tier(v: dict) -> str:
    k = v.get("kind")
    return {"local": "local", "param": "param", "fe-temp": "temp", "temp": "code"}.get(k, "temp")


def _phase(v: dict) -> str | None:
    return (v.get("origin") or {}).get("phase")


def _cell(a: dict, b: dict) -> str:
    ta, tb = _tier(a), _tier(b)
    if ta == tb == "local":
        return "decl-order"
    if "param" in (ta, tb) and {ta, tb} <= {"param", "local"}:
        return "param"
    if {ta, tb} & {"temp", "code"} and {ta, tb} & {"local", "param"}:
        return "cross-tier"
    if ta == tb == "temp":
        pa, pb = _phase(a), _phase(b)
        if pa == "front" or pb == "front":
            return "front-inline"
        if pa and pb and pa in _PHASES and pb in _PHASES and _PHASES.index(pa) > _PHASES.index(pb):
            return "cross-phase"
        return "same-phase"
    return "code-eval"


def _line0(v: dict) -> int | None:
    ln = v.get("line") or (v.get("origin") or {}).get("line")
    return ln - 1 if ln else None


def diagnose(analysis: dict | None, record: dict | None, values: list | None) -> Diagnosis:
    """The diagnosis of a start state from the solver's analysis (oracle.solve), its record and the
    values lined up with the target."""
    if analysis is None or "error" in analysis or record is None:
        return Diagnosis("none")
    by_index = {(v.get("cls"), v["index"]): v for v in record.get("vregs", [])}
    wrong = {ln for v in (values or []) if not v.ok and v.line for ln in (v.line - 1,)}
    rules: list[RuleView] = []
    stop: list[str] = []
    for cls, c in (analysis.get("classes") or {}).items():
        for r in c.get("rules") or []:
            if not r.get("flip"):
                continue
            a = dict(r["before"]) if isinstance(r.get("before"), dict) else {"index": r.get("before")}
            b = dict(r["after"]) if isinstance(r.get("after"), dict) else {"index": r.get("after")}
            a = {**by_index.get((cls, a.get("index")), {}), **a}
            b = {**by_index.get((cls, b.get("index")), {}), **b}
            cell = _cell(a, b)
            lines = {ln for ln in (_line0(a), _line0(b)) if ln is not None}
            view = RuleView(cls, a, b, cell, lines)
            rules.append(view)
            if cell in ("cross-phase", "param"):
                na, nb = a.get("name") or f"#{a.get('index')}", b.get("name") or f"#{b.get('index')}"
                stop.append(f"{cls} {cell}: colour {na} before {nb} ({_phase(a) or _tier(a)} vs {_phase(b) or _tier(b)})")
    if analysis.get("reachable") is False:
        return Diagnosis("stuck", rules, wrong, stop)
    if not rules and not wrong:
        return Diagnosis("sched", [], set(), [])
    return Diagnosis("order", rules, wrong, stop)


# ---- the two rows of this module ------------------------------------------------------------------

_HELPER = re.compile(r"^[ \t]*static[ \t]+inline[ \t]+([A-Za-z_][\w:<> *&]*?)[ \t]+([A-Za-z_]\w*)\s*\(([^)]*)\)\s*\{"
                     r"\s*return\s+(.*?);\s*\}[ \t]*$", re.M | re.S)
_SIDE = re.compile(r"\+\+|--|[^=!<>]=[^=]|\w\s*\(")


def _params(text: str) -> list[str] | None:
    out = []
    for p in text.split(","):
        p = p.strip()
        if not p:
            return None
        m = re.search(r"([A-Za-z_]\w*)\s*$", p)
        if m is None or "&" in p or "[" in p:
            return None
        out.append(m.group(1))
    return out or None


def _call_args(src: str, name: str) -> list[list[str]] | None:
    """The argument texts of every call of `name` in `src` (None when a call cannot be read)."""
    calls = []
    for m in re.finditer(rf"\b{re.escape(name)}\s*\(", src):
        depth, i, start, args = 1, m.end(), m.end(), []
        while i < len(src) and depth:
            ch = src[i]
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
                if depth == 0:
                    args.append(src[start:i].strip())
                    break
            elif ch == "," and depth == 1:
                args.append(src[start:i].strip())
                start = i + 1
            i += 1
        else:
            return None
        calls.append(args)
    return calls


def helper_to_macro(site: Site, lines: set[int] | None = None):
    """(label, new source) for each unit-local `static inline T f(args) { return EXPR; }` called inside
    the body (at `lines` when given) whose body reads some parameter twice: the helper becomes
    `#define f(args) ((T)(EXPR))` with every parameter parenthesised, so the argument is evaluated once
    per occurrence (the compiler CSEs the copies; T47 card 1). Legal only when every call in the unit
    passes side-effect-free arguments (no call, assignment or `++`/`--`) and no other definition of the
    name exists."""
    body = site.src[site.body.start:site.body.end]
    for m in _HELPER.finditer(site.src):
        rtype, name, params_text, expr = m.group(1).strip(), m.group(2), m.group(3), m.group(4).strip()
        if m.start() >= site.body.start or len(re.findall(rf"\b{re.escape(name)}\s*\(", site.src)) < 2:
            continue
        params = _params(params_text)
        if params is None or not any(len(re.findall(rf"\b{re.escape(p)}\b", expr)) >= 2 for p in params):
            continue
        uses = [mm for mm in re.finditer(rf"\b{re.escape(name)}\s*\(", body)]
        if not uses or (lines is not None and not any(site.line(site.body.start + u.start()) in lines for u in uses)):
            continue
        calls = _call_args(site.src, name)
        if calls is None or any(len(a) != len(params) for a in calls[1:]) or \
                any(_SIDE.search(arg) for a in calls[1:] for arg in a):
            continue
        macro = expr
        for p in params:
            macro = re.sub(rf"\b{re.escape(p)}\b", f"({p})", macro)
        macro = " ".join(macro.split())
        cast = "" if re.match(rf"^\(\s*{re.escape(rtype)}\s*\)", macro) or rtype == "void" else f"({rtype})"
        line = f"#define {name}({', '.join(params)}) " + (f"({cast}({macro}))" if cast else f"({macro})")
        yield f"{name} as a macro (its argument evaluated per occurrence)", [(m.start(), m.end(), line)]


_OPS = ("+", "-", "*", "/", "&", "|", "^")
_PREC = {"*": 5, "/": 5, "%": 5, "+": 4, "-": 4, "<<": 3, ">>": 3, "&": 2, "^": 1, "|": 0}


def join_op_assign(site: Site, lines: set[int] | None = None):
    """(label, edits) joining adjacent `v = A; v op= B;` into `v = A op B;` when B has no side effect and
    does not read v (the two statements compute the same value in the same order)."""
    for s in cbody.statements(site.body):
        if s.kind != "expr" or s.expr is None or s.expr.kind != "assign" or s.expr.op != "=" or not site.in_block(s):
            continue
        block, k = site.parents[id(s)]
        if k + 1 >= len(block.kids):
            continue
        nxt = block.kids[k + 1]
        if nxt.kind != "expr" or nxt.expr is None or nxt.expr.kind != "assign" or nxt.expr.op == "=":
            continue
        op = nxt.expr.op[:-1]
        if op not in _OPS:
            continue
        lhs, rhs = s.expr.kids[0], nxt.expr.kids[0]
        if lhs.kind != "name" or rhs.kind != "name" or lhs.name != rhs.name:
            continue
        if lines is not None and site.line(s.start) not in lines and site.line(nxt.start) not in lines:
            continue
        b = site.text(nxt.expr.kids[1])
        if _SIDE.search(b) or lhs.name in cbody.names(nxt.expr.kids[1]):
            continue
        a = site.text(s.expr.kids[1])
        ka, kb = s.expr.kids[1], nxt.expr.kids[1]
        if ka.kind in ("ternary", "assign", "comma") or (ka.kind == "binary" and _PREC.get(ka.op, 0) < _PREC[op]):
            a = f"({a})"
        bb = f"({b})" if kb.kind in ("ternary", "assign", "comma") or \
            (kb.kind == "binary" and _PREC.get(kb.op, 0) <= _PREC[op]) else b
        yield f"{lhs.name} = {rewrites._short(a)} {op} {rewrites._short(bb)} in one statement", \
            [(s.start, nxt.end, f"{lhs.name} = {a} {op} {bb};")]


LOCAL_ROWS = {"helper-to-macro": helper_to_macro, "join-op-assign": join_op_assign}


# ---- the directed candidate list -------------------------------------------------------------------

def _emit(site: Site, edits, seen: set[str]) -> str | None:
    try:
        new = cbody.apply_edits(site.src, edits)
    except ValueError:
        return None
    if new in seen:
        return None
    seen.add(new)
    return new


def _rule_at(site: Site, rule: str, lines: set[int]):
    from .regexplain import _rule_moves
    if rule not in rewrites.BY_NAME:
        return
    yield from _rule_moves(site, rule, lines)


def candidates(site: Site, diag: Diagnosis, values: list | None, wanted: list, rule_names: set[str] | frozenset,
               diff_lines: set[int], per_rule: int = 6) -> list[tuple[str, str, str]]:
    """(kind, label, new source), the moves the diagnosis points at, most direct first. `wanted` are the
    solver's declaration orders (oracle.wanted_orders), `diff_lines` the 0-based lines of the differing
    rows (used when the oracle names no line)."""
    from .regexplain import _fold_single_use, _join_init, _name_operand, _param_reuse, _split_init, single_decls
    seen: set[str] = {site.src}
    out: list[tuple[str, str, str]] = []
    lo, hi = cbody.line_of(site.src, site.body.start), cbody.line_of(site.src, site.body.end)
    inside = lambda lines: {ln for ln in lines if lo <= ln <= hi}

    def add(kind, label, edits_or_src):
        if edits_or_src is None:
            return
        if isinstance(edits_or_src, str):
            new = edits_or_src
            if new in seen:
                return
            seen.add(new)
        else:
            new = _emit(site, edits_or_src, seen)
            if new is None:
                return
        out.append((kind, label, new))

    def rules_at(names, lines, tag):
        lines = inside(lines)
        if not lines:
            return
        for rule in names:
            n = 0
            for label, new in _rule_at(site, rule, lines):
                if new in seen:
                    continue
                add(f"directed:{rule}", f"{label} [{tag}]", new)
                n += 1
                if n >= per_rule:
                    break

    def local_rows(lines, tag):
        lines = inside(lines) or None
        for rule, gen in LOCAL_ROWS.items():
            for label, edits in gen(site, lines):
                add(f"directed:{rule}", f"{label} [{tag}]", edits)

    decls = single_decls(site)
    if diag.kind == "order":
        if wanted:
            from .blockorders import block_orders
            for w in wanted:
                try:
                    found = block_orders(site, w, rule_names)
                except (ValueError, KeyError, IndexError, AttributeError):
                    found = []
                for kind, label, new in found:
                    add(f"directed:{kind}", label, new)
        for r in diag.rules:
            na, nb = r.a.get("name") or f"#{r.a.get('index')}", r.b.get("name") or f"#{r.b.get('index')}"
            tag = f"{na} before {nb}"
            if r.cell in ("cross-phase", "param"):
                continue
            if r.cell == "decl-order":
                from .regexplain import reorder_edits
                if r.a.get("name") in decls and r.b.get("name") in decls:
                    add("directed:decl-order", f"{na} declared before {nb}", reorder_edits(site, [r.a["name"], r.b["name"]]))
                for v in (r.a, r.b):
                    if v.get("name") in decls:
                        add("directed:split", f"{v['name']} declared, then assigned [{tag}]", _split_init(site, v["name"]))
                        add("directed:join", f"{v['name']} initialised where declared [{tag}]", _join_init(site, v["name"]))
            elif r.cell == "cross-tier":
                named = [v for v in (r.a, r.b) if _tier(v) in ("local", "param")]
                temps = [v for v in (r.a, r.b) if _tier(v) in ("temp", "code")]
                for v in named:
                    if v.get("name") in decls and _tier(v) == "local":
                        add("directed:fold", f"{v['name']} folded into its one use [{tag}]", _fold_single_use(site, v["name"]))
                        add("directed:join", f"{v['name']} initialised where declared [{tag}]", _join_init(site, v["name"]))
                        add("directed:split", f"{v['name']} declared, then assigned [{tag}]", _split_init(site, v["name"]))
                        for label, edits in _param_reuse(site, v["name"]):
                            add("directed:param-reuse", f"{label} [{tag}]", edits)
                for v in temps:
                    ln = _line0(v)
                    if ln is not None and lo <= ln <= hi:
                        for label, edits in _name_operand(site, ln):
                            add("directed:name-temp", f"{label} [{tag}]", edits)
                rules_at(CROSS_TIER_TEMP_RULES, {ln for v in temps for ln in (_line0(v),) if ln is not None}, tag)
            elif r.cell == "same-phase":
                rules_at(ORDER_RULES_SAME_PHASE, r.lines, tag)
                if (_phase(r.a) or "").startswith("cse") or (_phase(r.b) or "").startswith("cse"):
                    local_rows(r.lines, tag)
            elif r.cell == "front-inline":
                local_rows(r.lines, tag)
                rules_at(("inline-helper", "helper-params", "name-value", "inline-local"), r.lines, tag)
            else:                                   # code-eval
                rules_at(ORDER_RULES_CODE, r.lines, tag)
                local_rows(r.lines, tag)
    elif diag.kind == "stuck":
        lines = diag.wrong_lines | set(diff_lines)
        local_rows({ln + d for ln in lines for d in (-1, 0, 1)}, "stuck")
        for v in (values or []):
            if not v.ok and v.name and v.name in decls and v.kind == "local":
                add("directed:fold", f"{v.name} folded into its one use [stuck]", _fold_single_use(site, v.name))
                add("directed:join", f"{v.name} initialised where declared [stuck]", _join_init(site, v.name))
        rules_at(STUCK_RULES, lines, "stuck")
        for v in (values or []):
            if not v.ok and v.line and lo <= v.line - 1 <= hi:
                for label, edits in _name_operand(site, v.line - 1):
                    add("directed:name-temp", f"{label} [stuck]", edits)
    elif diag.kind == "sched":
        lines = set(diff_lines)
        wide = {ln + d for ln in lines for d in (-2, -1, 0, 1, 2)}
        rules_at(SCHED_RULES[:2], wide, "sched")
        local_rows(wide, "sched")
        rules_at(SCHED_RULES[2:], lines, "sched")
    return out


def render(diag: Diagnosis) -> list[str]:
    cells = {}
    for r in diag.rules:
        cells[r.cell] = cells.get(r.cell, 0) + 1
    line = f"  directed: {diag.kind}" + (f", rules by cell: " + ", ".join(f"{k} {n}" for k, n in sorted(cells.items()))
                                        if cells else "")
    out = [line]
    for s in diag.stop:
        out.append(f"    STOP (no faithful lever): {s}")
    return out
