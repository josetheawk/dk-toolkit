"""`regs --why-temp`: for each "colour A before B" rule of the order solver, which kind of value each side is,
which optimizer phase created each temporary (oracle contract v3, `origin`), and the source-shape levers that
the creation-order rules imply. Behaviour only (playbook/matching.md, "temporary creation order"):

- Values are coloured by tier: code-generation values (no source object) newest first, then the optimizer's
  temporaries oldest first, then locals in declaration order, then parameters. So "colour A before B" between
  two temporaries means "create A first".
- The optimizer's phases run in a fixed order: front end (inline expansion objects and other front-end temps)
  < linearize (the value of a `?:`, `&&`/`||` or comma expression, an inline call used as a value, a saved call
  argument) < pass-0 propagation < unroller < loop pass (invariant motion, strength reduction) < CSE pass 0 <
  pass-1 propagation < CSE pass 1 < late (lifetime splitting). A temp of an earlier phase is always created first.
- Inside a phase the walk is the flowgraph node list: statement order, except that a loop's test sits after its
  body. CSE makes its temp at the expression's SECOND occurrence. Linearize goes statement by statement and, in
  one statement, the costlier operand first (the right one when equal). The loop pass takes loops in the order
  of their tests (an inner loop before its outer one). The inliner expands the calls written in the function
  first, in statement order, then the calls those bodies brought, and so on; one expansion makes its result,
  then its locals, then its arguments.
"""

from __future__ import annotations

PHASES = ["front", "linear", "prepass", "prop0", "unroll", "loop", "cse0", "prop1", "cse1", "late", "backend"]

WHAT = {
    "front": "a front-end temp",
    "linear": "a linearize temp",
    "prepass": "a pre-pass temp",
    "prop0": "a pass-0 propagation temp",
    "unroll": "a loop-unroller temp",
    "loop": "a loop-pass temp (invariant motion or strength reduction)",
    "cse0": "a common-subexpression temp (CSE pass 0)",
    "prop1": "a pass-1 propagation temp",
    "cse1": "a common-subexpression temp (CSE pass 1)",
    "late": "a lifetime-splitting temp (late)",
    "backend": "a temp made after the optimizer",
}

# how to move a temp to an EARLIER phase (when it must be created before a temp of an earlier phase)
EARLIER = {
    "front": "already the earliest phase: create it earlier inside the front end (call its inline sooner, or make "
             "the helper's value an argument of a call written directly in the function)",
    "linear": "make the value an inline helper's argument or local (front end), e.g. pass it to a static inline "
              "instead of writing the ?:/comma expression in place",
    "loop": "make it a linearize temp (a ?: or && value) or an inline helper's object; or move it out of the loop",
    "cse0": "make it a linearize temp (the value of a ?: / &&-|| / inline call on a statement before the other "
            "temp) or an inline helper's argument/local (front end)",
    "cse1": "make it a CSE pass-0 temp (repeat the expression plainly), a linearize temp or an inline object",
    "late": "make it a CSE temp or a named value used plainly",
}
# how to move a temp to a LATER phase (when it must be created after a temp of a later phase)
LATER = {
    "front": "write the helper so the inliner needs no object for it (an argument used once and never modified, "
             "an expression-level body), or write the call's work out in place",
    "linear": "drop the ?: / &&-|| / comma VALUE: write it as an if/else statement or as plain arithmetic, so "
              "CSE makes the temp (at its 2nd occurrence) or no temp is made",
    "loop": "make the value an expression repeated inside the loop (a CSE temp) rather than an invariant, or "
            "index the loop naturally so no strength-reduced pointer is needed",
    "cse0": "make it a pass-1 temp (an expression that only becomes common after propagation) or no temp",
}


def tier(v: dict) -> str:
    k = v.get("kind") or "?"
    return {"fe-temp": "temp", "temp": "code"}.get(k, k)


def phase(v: dict) -> str | None:
    o = v.get("origin")
    return o.get("phase") if isinstance(o, dict) else None


def line_text(lines: list[str], body: tuple[int, int] | None, n: int | None) -> str:
    """` (line N: text)` when line N is inside the function's body, else ` (line N)`."""
    if not n:
        return ""
    if body and body[0] <= n - 1 <= body[1] and n - 1 < len(lines):
        return f" (line {n}: {lines[n - 1].strip()[:70]})"
    return f" (line {n}, outside the body: an inline helper's line?)"


def construct(text: str) -> str:
    """The linearize construct a source line most likely holds."""
    t = text.split("//")[0]
    if "?" in t:
        return "the value of a ?: expression"
    if "&&" in t or "||" in t:
        return "the value of an && / || expression"
    return "the value of an inline call used in an expression (or a comma value / saved call argument)"


def describe(v: dict, lines: list[str], body) -> str:
    """One value in words: tier, phase and where that phase met it."""
    nm = v.get("name") or ""
    head = f"#{v.get('index')}" + (f" {nm}" if nm else "")
    t = tier(v)
    if t == "local":
        return f"{head}: local, ordered by declaration{line_text(lines, body, v.get('line'))}"
    if t == "param":
        return f"{head}: parameter{line_text(lines, body, v.get('line'))}"
    if t == "code":
        return f"{head}: code-generation value (no source object; newest coloured first){line_text(lines, body, v.get('line'))}"
    o = v.get("origin") or {}
    p = o.get("phase")
    if not p:
        return f"{head}: optimizer temp, phase unknown (the runner has no temp origins){line_text(lines, body, v.get('line'))}"
    s = f"{head}: {WHAT.get(p, p)}"
    if p.startswith("cse"):
        s += f", made at its 2nd occurrence{line_text(lines, body, o.get('line'))}"
        if o.get("first_line"):
            s += f"; 1st occurrence line {o['first_line']}"
    elif p == "loop":
        s += f", loop test{line_text(lines, body, o.get('line'))}"
    elif p == "linear":
        n = o.get("line") or v.get("line")
        txt = lines[n - 1] if n and body and body[0] <= n - 1 <= body[1] and n - 1 < len(lines) else ""
        s += f": {construct(txt)}{line_text(lines, body, n)}"
    elif p == "front" and o.get("role"):
        s += f": inline expansion {o['role']}" + (f" of {o['inline']}" if o.get("inline") else "") + \
            line_text(lines, body, o.get("line"))
    else:
        s += line_text(lines, body, o.get("line") or v.get("line"))
    return s


def _ln(v: dict) -> int | None:
    o = v.get("origin") or {}
    return o.get("line") or v.get("line")


def levers(a: dict, b: dict) -> tuple[str, list[str]]:
    """(verdict, levers) for the rule "colour a before b" that ours breaks (b is coloured first now)."""
    ta, tb = tier(a), tier(b)
    na, nb = a.get("name") or f"#{a.get('index')}", b.get("name") or f"#{b.get('index')}"
    if ta == "temp" and tb == "temp":
        pa, pb = phase(a), phase(b)
        if not pa or not pb:
            return "phase unknown", [f"run with a runner that gives temp origins (contract v3) to see why {na} comes after {nb}"]
        ra, rb = PHASES.index(pa) if pa in PHASES else 99, PHASES.index(pb) if pb in PHASES else 99
        if ra > rb:
            return (f"different phases: {pa} after {pb}, so no reordering inside a phase can flip this pair "
                    f"(phase: {WHAT.get(pb, pb)} always precedes {WHAT.get(pa, pa)})"), [
                f"{na} earlier: {EARLIER.get(pa, 'move it to an earlier phase')}",
                f"{nb} later: {LATER.get(pb, 'move it to a later phase')}"]
        if ra < rb:
            return (f"phase order already creates {na} first ({pa} before {pb}); the allocator's simplify order "
                    f"(interference, not creation) puts {nb} first"), [
                "change liveness instead: shorten or split one value's lifetime (use it closer to its definition, "
                "or recompute it) so its interference count changes"]
        la, lb = _ln(a), _ln(b)
        same = la is not None and la == lb
        out: list[str]
        if pa.startswith("cse"):
            out = [f"{na} is made at its 2nd occurrence (line {la}), {nb} at its 2nd occurrence (line {lb}): "
                   f"{na}'s 2nd occurrence must come first in statement order"]
            if same:
                out.append("same statement: swap the operands so that " + na + "'s repeated expression is met first")
            else:
                out.append(f"move the statement with {nb}'s 2nd occurrence after line {la}, or the one with {na}'s "
                           f"2nd occurrence before line {lb}")
                out.append(f"make {nb}'s first use come second: if {nb}'s expression is first written at line "
                           f"{(b.get('origin') or {}).get('first_line')}, delay that use so its repeat comes later")
            out.append("a loop's test is visited after its body: an occurrence in a for/while condition counts late")
        elif pa == "linear":
            out = [f"linearize walks statements in order: {na}'s construct (line {la}) must be in a statement "
                   f"before {nb}'s (line {lb})"]
            if same:
                out.append("same statement: the costlier operand is linearized first (the right one when equal; "
                           "a ?: before a call): swap the operands or make " + na + "'s side the costlier")
            else:
                out.append(f"move {na}'s ?:/&&/inline-call value to a statement before line {lb}, or {nb}'s after")
        elif pa == "loop":
            out = [f"loops are visited by the position of their test (an inner loop before its outer loop): "
                   f"{na}'s loop (test line {la}) must have its test before {nb}'s (line {lb})",
                   "put the loops in the other order, or nest one inside the other"]
        elif pa == "front":
            ra_, rb_ = (a.get("origin") or {}).get("role"), (b.get("origin") or {}).get("role")
            out = ["the inliner expands the calls written in the function first, in statement order, then the calls "
                   "inside those bodies (later rounds): call " + na + "'s helper directly or earlier"]
            if ra_ and rb_ and (a.get("origin") or {}).get("inline") == (b.get("origin") or {}).get("inline"):
                out.append(f"same expansion: the inliner makes the result, then the locals, then the arguments "
                           f"({na} is a {ra_}, {nb} a {rb_}): turn one into the other role (name an argument as a "
                           f"local first in natural order, or pass a local as an argument)")
        else:
            out = [f"same phase ({pa}): {na}'s trigger must come before {nb}'s in statement order"]
        return f"same phase ({pa}): flip by visit order", out
    if "temp" in (ta, tb) and ({ta, tb} & {"local", "param"}):
        t, o = (a, b) if ta == "temp" else (b, a)
        nt, no = t.get("name") or f"#{t.get('index')}", o.get("name") or f"#{o.get('index')}"
        pt = phase(t) or "unknown phase"
        return (f"cross-tier: {nt} is {WHAT.get(pt, pt)}, {no} a {tier(o)}; temps and declared values are coloured "
                f"in separate tiers, so declaration order alone does not flip this; a degree change can (`regs --reach` "
                f"degree levers: one value more or fewer live across {no}'s range)"), [
            f"degree: add or drop one value live across {no}'s range that coalesces away (an inline's argument or "
            f"result, a `?:` result copied into an argument register; merged values keep counting, values with "
            f"their own register stop counting after the first scan), so simplify defers {no} past {nt} or not",
            f"remove the temp {nt}: compute its value straight into an existing local, or use the value once",
            f"make {no} an expression instead of a {tier(o)} (re-read the member or repeat the expression; CSE then "
            f"makes the temp)" if tier(o) == "local" else f"reuse the parameter {no} where the original did",
            f"make {nt} a named local (declare it where the declaration order puts it in the right place)",
            f"{nt}'s own phase: " + (LATER.get(pt) or EARLIER.get(pt) or "see the phase order")]
    if "code" in (ta, tb):
        return ("code-generation value: made by instruction selection in evaluation order, newest coloured first"), [
            "change the evaluation order of the expressions involved (operand order, statement order)",
            "or give the value a source object (a named local) so it moves to another tier"]
    if ta == tb == "local":
        return "declaration order", ["declare " + na + " before " + nb + " (the solver's declaration-order lever)"]
    return f"{ta} vs {tb}", ["parameter order is the signature's: change the other value",
                             f"or the degree: one value more or fewer live across {na}'s range makes simplify defer "
                             f"it (coloured first) or not (`regs --reach` lists the degree levers)"]


def explain(analysis: dict, record: dict, text: str, body: tuple[int, int] | None, flipped_only: bool = True) -> list[dict]:
    """One entry per solver rule (flipped ones only by default): {cls, cell, before, after, verdict, levers}."""
    byidx = {(v.get("cls"), v.get("index")): v for v in record.get("vregs", [])}
    lines = text.splitlines()
    out = []
    for cls, r in sorted((analysis.get("classes") or {}).items()):
        for x in r.get("rules") or []:
            if flipped_only and not x.get("flip"):
                continue
            a = dict(byidx.get((cls, x["before"]["index"])) or x["before"])
            b = dict(byidx.get((cls, x["after"]["index"])) or x["after"])
            verdict, lv = levers(a, b)
            out.append({"cls": cls, "cell": x.get("cell", ""), "flip": bool(x.get("flip")),
                        "before": {"index": a.get("index"), "name": a.get("name"), "kind": a.get("kind"),
                                   "origin": a.get("origin"), "text": describe(a, lines, body)},
                        "after": {"index": b.get("index"), "name": b.get("name"), "kind": b.get("kind"),
                                  "origin": b.get("origin"), "text": describe(b, lines, body)},
                        "verdict": verdict, "levers": lv})
    return out


def render(entries: list[dict]) -> list[str]:
    out = []
    for k, e in enumerate(entries, 1):
        out.append(f"  rule {k} [{e['cls']} {e['cell']}]{' (flipped)' if e['flip'] else ''}: colour "
                   f"{e['before']['name'] or '#' + str(e['before']['index'])} before "
                   f"{e['after']['name'] or '#' + str(e['after']['index'])}")
        out.append(f"    A {e['before']['text']}")
        out.append(f"    B {e['after']['text']}")
        out.append(f"    why: {e['verdict']}")
        for lv in e["levers"]:
            out.append(f"    - {lv}")
    return out
