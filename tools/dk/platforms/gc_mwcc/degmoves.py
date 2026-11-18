"""Degree-lever moves for `dk close --moves degree` (H10/H10b: the colouring order follows simplify's raw
degrees, and a value that coalesces away is the lasting lever).

The order solver (oracle contract v2, `solve`) adds a `degree` block to a class whose rules flip:
- `levers`: one value (or two) whose raw degree, moved by +-d, makes simplify colour every known value as the
  target. +d: a new value that coalesces away must be live while that value is; -d: one such value must go.
- `spans` (when the runner gives them): source line ranges where one new coalesced-away value does it.
- `drops`: existing temps/locals whose removal does it.

This module turns those into source edits at the named lines, the sign-matched family first:
ADD (one more value): name an operand in a local (a call result, load or arithmetic: R6's named local),
split `v = A op B;` into `v = A; v op= B;`, split a local's initialiser, the table's name-value / split-init /
accessor / cached-chain rows. DROP (one fewer): fold a single-use local into its use, join `T x; x = e;`,
join `v = A; v op= B;`, a unit-local helper as a macro, the table's inline-local / fold-single / local-reuse /
chained-assign rows. Every edit is a faithful rewrite (same operations, same order); close compiles, scores
and judges each one as in `--moves directed`, so a dummy never lands.

`ctor_hints` names C-style allocate-then-initialise code (`p = alloc(...); if (p) p = init(p, ...);` where
`init` returns its first parameter): MWCC's `new T(...)` keeps the result in the call's register (T67's
VideoDriver case); writing it needs a class with that constructor, so it is reported, never compiled.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ...core import cbody
from . import rewrites
from .rewrites import Site

ADD_RULES = ("name-value", "split-init", "accessor", "cached-chain", "helper-params")
DROP_RULES = ("inline-local", "fold-single", "local-reuse", "chained-assign")


@dataclass
class Target:
    sign: int                  # +1: add a coalesced-away value; -1: remove one
    lines: list[int]           # 0-based source lines, most direct first
    names: list[str] = field(default_factory=list)    # named values (locals/params) involved
    why: str = ""


def _line0(v: dict) -> int | None:
    ln = v.get("line")
    return ln - 1 if isinstance(ln, int) and ln > 0 else None


def _live_lines(site: Site | None, v: dict, cap: int = 12) -> list[int]:
    """0-based lines where a named value is live (between its first and last mention), nearest the mentions
    first; a temp's own line otherwise."""
    ln = _line0(v)
    name = v.get("name") or ""
    if site is not None and v.get("kind") in ("local", "param") and re.fullmatch(r"[A-Za-z_]\w*", name):
        body = site.src[site.body.start:site.body.end]
        at = sorted({site.line(site.body.start + m.start()) for m in re.finditer(rf"\b{re.escape(name)}\b", body)})
        if at:
            span = range(at[0], at[-1] + 1)
            return sorted(span, key=lambda x: (min(abs(x - a) for a in at), x))[:cap]
    return [ln] if ln is not None else []


def targets(analysis: dict | None, site: Site | None = None) -> list[Target]:
    """The degree block of every class, as edit targets in priority order: single-value levers, spans
    (narrowest first), drops, then two-value levers."""
    out: list[Target] = []
    if not analysis:
        return out
    singles, pairs, spans, drops = [], [], [], []
    for cls, c in (analysis.get("classes") or {}).items():
        dg = c.get("degree") or {}
        for lv in dg.get("levers") or []:
            vals, d = lv.get("values") or [], int(lv.get("delta") or 0)
            if not vals or not d:
                continue
            lines: list[int] = []
            for v in vals:
                lines += [x for x in _live_lines(site, v) if x not in lines]
            names = [v["name"] for v in vals if v.get("kind") in ("local", "param") and v.get("name")]
            what = " and ".join(v.get("name") or f"#{v.get('index')}" for v in vals)
            t = Target(1 if d > 0 else -1, lines, names, f"{cls} {d:+d} across {what}")
            (singles if len(vals) == 1 else pairs).append(t)
        for sp in dg.get("spans") or []:
            a, b = sp.get("lines") or (None, None)
            if isinstance(a, int) and isinstance(b, int) and a > 0:
                spans.append(Target(1, list(range(a - 1, b)), [], f"{cls} new value over lines {a}-{b}"))
        for v in dg.get("drops") or []:
            ln = _line0(v)
            name = v.get("name") if v.get("kind") == "local" else None
            if ln is not None or name:
                drops.append(Target(-1, [ln] if ln is not None else [], [name] if name else [],
                                    f"{cls} drop {v.get('name') or '#' + str(v.get('index'))}"))
    return singles + spans + drops + pairs


# ---- the one row of this module -------------------------------------------------------------------

_OPS = ("+", "-", "*", "/", "&", "|", "^", "<<", ">>")
_SIDE = re.compile(r"\+\+|--|[^=!<>]=[^=]|\w\s*\(")


def split_op_assign(site: Site, lines: set[int] | None = None):
    """(label, edits) splitting `v = A op B;` into `v = A; v op= B;` (the inverse of join-op-assign) when v
    is a plain name, A is not v itself and B has no side effect and does not read v."""
    for s in cbody.statements(site.body):
        if s.kind != "expr" or s.expr is None or s.expr.kind != "assign" or s.expr.op != "=" or not site.in_block(s):
            continue
        if lines is not None and site.line(s.start) not in lines:
            continue
        lhs, rhs = s.expr.kids[0], s.expr.kids[1]
        if lhs.kind != "name" or rhs.kind != "binary" or rhs.op not in _OPS:
            continue
        a, b = rhs.kids[0], rhs.kids[1]
        if (a.kind == "name" and a.name == lhs.name) or lhs.name in cbody.names(b) or _SIDE.search(site.text(b)):
            continue
        ind = cbody.indent_at(site.src, s.start)
        yield (f"{lhs.name} = {rewrites._short(site.text(a))}; {lhs.name} {rhs.op}= {rewrites._short(site.text(b))}",
               [(s.start, s.end, f"{lhs.name} = {site.text(a)};\n{ind}{lhs.name} {rhs.op}= {site.text(b)};")])


# ---- candidates -----------------------------------------------------------------------------------

def _emit(site: Site, edits, seen: set[str]) -> str | None:
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


def _family(site: Site, sign: int, lines: set[int], names: list[str]):
    """(kind, label, edits or source) of one sign at the lines (and the named locals)."""
    from .regdirect import helper_to_macro, join_op_assign
    from .regexplain import _fold_single_use, _join_init, _name_operand, _param_reuse, _rule_moves, _split_init, \
        single_decls
    decls = single_decls(site)
    local_names = [n for n in names if n in decls]
    local_names += [n for n, s in decls.items() if n not in local_names and site.line(s.start) in lines]
    if sign > 0:
        for ln in sorted(lines):
            for label, edits in _name_operand(site, ln):
                yield "degree:name-temp", label, edits
        for label, edits in split_op_assign(site, lines):
            yield "degree:split-op-assign", label, edits
        for n in local_names:
            yield "degree:split", f"{n} declared, then assigned", _split_init(site, n)
        rules = ADD_RULES
    else:
        for n in local_names:
            yield "degree:fold", f"{n} folded into its one use", _fold_single_use(site, n)
            yield "degree:join", f"{n} initialised where declared", _join_init(site, n)
            for label, edits in _param_reuse(site, n):
                yield "degree:param-reuse", label, edits
        for label, edits in join_op_assign(site, lines):
            yield "degree:join-op-assign", label, edits
        for label, new in helper_to_macro(site, lines):
            yield "degree:helper-to-macro", label, new
        rules = DROP_RULES
    for rule in rules:
        if rule in rewrites.BY_NAME:
            for label, new in _rule_moves(site, rule, lines):
                yield f"degree:{rule}", label, new


def candidates(site: Site, analysis: dict | None, per_target: int = 8) -> list[tuple[str, str, str]]:
    """(kind, label, new source) for every degree target, sign-matched families first (per target at most
    `per_target`), then the opposite sign at the same lines (the lever's sign is the model's, not a proof)."""
    lo, hi = cbody.line_of(site.src, site.body.start), cbody.line_of(site.src, site.body.end)
    seen: set[str] = {site.src}
    out: list[tuple[str, str, str]] = []
    tg = targets(analysis, site)
    for flip in (1, -1):
        for t in tg:
            lines = {ln + d for ln in t.lines for d in (0,) if lo <= ln <= hi}
            if not lines and not t.names:
                continue
            n = 0
            for kind, label, item in _family(site, t.sign * flip, lines, t.names):
                if item is None:
                    continue
                new = item if isinstance(item, str) else _emit(site, item, seen)
                if new is None or (isinstance(item, str) and new in seen):
                    continue
                seen.add(new)
                out.append((kind, f"{label} [{t.why}{'' if flip == 1 else ', opposite sign'}]", new))
                n += 1
                if n >= per_target:
                    break
    return out


# ---- C-style constructors (reported) --------------------------------------------------------------

_ALLOC_INIT = re.compile(
    r"\b(?P<p>[A-Za-z_]\w*)\s*=\s*(?:\([^;()]*\)\s*)?[^;]*?\([^;]*\)\s*;\s*"
    r"if\s*\(\s*(?P=p)\s*(?:!=\s*0\s*)?\)\s*\{?\s*(?P=p)\s*=\s*(?:\([^;()]*\)\s*)?(?P<init>[A-Za-z_]\w*)\s*\(\s*(?P=p)\s*[,)]",
    re.S)


def _returns_first_param(src: str, name: str) -> bool:
    m = re.search(rf"\b{re.escape(name)}\s*\(\s*[^,)]*?\b([A-Za-z_]\w*)\s*[,)][^;{{]*\{{", src)
    if not m:
        return False
    depth, i = 1, m.end()
    while i < len(src) and depth:
        depth += {"{": 1, "}": -1}.get(src[i], 0)
        i += 1
    return re.search(rf"\breturn\s+{re.escape(m.group(1))}\s*;", src[m.end():i]) is not None


def ctor_hints(site: Site, texts: list[str] | None = None) -> list[str]:
    """Lines (1-based) of C-style allocate-then-initialise code whose initialiser returns its first parameter
    (defined in the unit or in `texts`): a `new (...) T(...)` with a real constructor is the faithful form."""
    body = site.src[site.body.start:site.body.end]
    out = []
    for m in _ALLOC_INIT.finditer(body):
        init = m.group("init")
        if any(_returns_first_param(t, init) for t in [site.src] + list(texts or [])):
            out.append(f"line {site.line(site.body.start + m.start()) + 1}: {m.group('p')} = alloc; "
                       f"if ({m.group('p')}) {m.group('p')} = {init}({m.group('p')}, ...) -- {init} returns its first "
                       f"argument: write it as `new (...) T(...)` with {init} as T's constructor")
    return out
