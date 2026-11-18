"""dk inline-cost: why the compiler inlines a call or keeps it, and how far it is from flipping.

  ./dk inline-cost <callee> [--caller F] [--unit U] [--probes]

The rule (read from the compiler's inliner, CInline, and confirmed on toys and on every call of the
project's units; GC 1.3 and 1.3.2 run the "bottom-up" inliner by default):

1. Candidacy. A function defined with `inline` (or inside its class) is a candidate once its body has been
   seen. Any other function becomes an auto-inline candidate (`-inline auto`, `#pragma auto_inline` on)
   when it is defined *before* the caller is compiled, has at most S statements as lowered (S = 15 on
   GC 1.3, 30 on GC 1.3.2, counting the implicit final return; an expression or a declaration with an
   initializer 1, `if` 2, `if` with `&&` 3, `if/else` 4, an early `return` inside an `if` 3, `while` 3,
   `do/while` 2, a small `for` 5, a 3-case `switch` with breaks 7, `?:` 1; `;`, `{}` and a declaration
   without an initializer 0; a call to an inline helper 1 whatever its size), at most 1024 bytes of
   locals, no `...`, and no class with a destructor taken or returned by value.
2. Cost. A candidate call is inlined when the callee's cost is at most the caller's limit. Cost = the
   callee's statements after the compiler's first clean-up (labels, gotos and the return count) plus its
   expression nodes (each operand, operator, cast, call and argument 1), with each inlinable callee of
   its own counted at its own cost, one nesting level less each time (7 levels at the first round).
3. Limit. 512 per caller, halved while the caller's own size estimate (its statements and nodes, with
   every inlinable call counted at its cost) exceeds 7000. A caller whose own body is over 7000 ends at 1:
   nothing is inlined into it, `inline` functions included ("the budget runs out" in a giant).
4. Rounds. Inlining runs 8 rounds; calls that appear inside an inlined body are judged in the next round
   with one level less, so recursion unrolls until the cost at the remaining depth fits, and the calls in
   bodies inlined in the last round stay calls. GC 1.3 also inlines an `inline` function whatever its cost
   in the last round (so a too-big explicit inline is still inlined, one level deep).

With the compiler oracle (DK_ORACLE runner with the `inline` mode, contract inline.v1 below) every number is
read from the compiler: the callee's cost and statements, the caller's limit, the round. Without it (or with
--probes) the slack is measured with probes on a private copy, without touching the source:
  - inlined now: how many no-code statements (`((void)0);`) still fit before it stops being inlined;
  - called now: how many of its expression statements, blanked from the end, make it inlined.
Either way, the method for using the number (natural shapes first) is in the platform playbook.

Contract inline.v1 (oracle runner, optional): `<command> OUT.json --inline -- <compile argv>` runs the compile
unchanged and writes {"schema": "inline.v1", "compiler", "limits": {"stmts", "locals", "max_size", "total"},
"functions": [{"name" (unmangled), "link" (mangled, or ""), "obj" (an id), "max" (the caller's limit after
halving), "totals" (its size estimates), "checks": [{"callee", "link", "obj", "round", "explicit", "auto",
"stmts", "can" (>= 6 expression, 3 statement level, else not inlinable), "est" (cost), "max"}]}],
"small": [{"name", "link", "obj", "stmts", "locals"}] (each auto-inline candidacy test)}. A check without
"est" is a callee that is no candidate there.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from ...core import cbody
from ...core.command import MISMATCH, OK, USAGE, Command, hint
from ...core.config import Config
from .closecmd import plain_name
from .elf import Elf
from .matching import MatchError, compile_command, resolve_unit, retarget
from .reorder import mirror
from .trycmd import candidates, definition, in_header

STATEMENT = "((void)0);"                  # a statement that makes no code
STORE = "*(volatile char*)0 = 0;"         # a statement with a size (explicit inline on size-measuring versions)
MAX_PROBES = 64


def named(symbol: str, query: str) -> bool:
    """`symbol` (a name in an object) is the function `query` names: the same symbol, or the same
    written name (`fn_80001234` for `fn_80001234__5ClassFv`)."""
    if symbol == query:
        return True
    a, b = plain_name(symbol), plain_name(query)
    return a is not None and a == b and (query == b or symbol == b)


def calls(elf: Elf, caller: str, callee: str) -> bool | None:
    """Whether `caller` (in `elf`) has a call or reference to `callee`; None if it is not there."""
    fns = elf.functions()
    fn = fns.get(caller) or next((f for n, f in fns.items() if named(n, caller)), None)
    if fn is None:
        return None
    return any(named(r.symbol, callee) for r in fn.relocs)


def callers_of(elf: Elf, callee: str) -> list[str]:
    return sorted(n for n, f in elf.functions().items() if not named(n, callee)
                  and any(named(r.symbol, callee) for r in f.relocs))


def with_probes(text: str, open_brace: int, count: int, probe: str) -> str:
    return text[:open_brace + 1] + "".join(f" {probe}" for _ in range(count)) + text[open_brace + 1:]


def _open_brace(text: str, span: tuple[int, int]) -> int:
    return text.index("{", span[0])


def expression_statements(text: str, span: tuple[int, int]) -> list[tuple[int, int]]:
    """Spans of the expression statements of the function at `span`, in source order (nested ones too)."""
    try:
        body = cbody.parse_body(text, _open_brace(text, span), cbody.type_names(text))
    except cbody.ParseError:
        return []
    return sorted((s.start, s.end) for s in cbody.statements(body) if s.kind == "expr")


def edited(text: str, span: tuple[int, int], count: int = 0, probe: str = STATEMENT,
           blank: list[tuple[int, int]] = ()) -> str:
    """`text` with the spans in `blank` (expression statements) emptied to `;` and `count` probes
    at the start of the function's body."""
    for lo, hi in sorted(blank, reverse=True):          # later spans first: offsets stay valid
        text = text[:lo] + ";" + text[hi:]
    return with_probes(text, _open_brace(text, span), count, probe)


class Probe:
    """Compiles the unit privately with the callee's definition (in the source or a header) changed."""

    def __init__(self, cfg: Config, unit: dict, callee: str):
        self.cfg, self.unit, self.callee = cfg, unit, callee
        self.source_rel = unit["metadata"]["source_path"]
        self.source = (cfg.root / self.source_rel).read_text()
        self.header_rel = None
        span = definition(self.source, callee)
        if span is None and not candidates(self.source, callee):
            self.header_rel = in_header(cfg, self.source, callee)
            if self.header_rel is not None:
                self.header = (cfg.root / self.header_rel).read_text()
                span = definition(self.header, callee)
        self.span = span

    @property
    def where(self) -> str:
        text = self.header if self.header_rel else self.source
        return f"{self.header_rel or self.source_rel}:{text.count(chr(10), 0, self.span[0]) + 1}"

    def expression_statements(self) -> list[tuple[int, int]]:
        return expression_statements(self.header if self.header_rel else self.source, self.span)

    def compile(self, count: int = 0, probe: str = STATEMENT, blank: list[tuple[int, int]] = (),
                wrap=None) -> Elf | None:
        cfg = self.cfg
        edit = lambda text: edited(text, self.span, count, probe, blank)
        source = self.source
        header = self.header if self.header_rel else None
        if self.header_rel:
            header = edit(header)
        else:
            source = edit(source)
        work = Path(tempfile.mkdtemp(prefix="dk-inline-cost-"))
        try:
            out = work / "out"
            out.mkdir()
            argv = retarget(compile_command(cfg, self.unit["base_path"], locked=False), out)
            if header is not None:
                shadow = work / "shadow"
                dest = shadow / Path(self.header_rel).relative_to("include")
                dest.parent.mkdir(parents=True)
                dest.write_text(header)
                first = next((k for k, a in enumerate(argv) if a == "-i"), len(argv))
                argv[first:first] = ["-i", str(shadow)]
            tree = mirror(cfg, self.source_rel, source, work / "tree")
            if wrap is not None:
                argv = wrap([a for a in argv if not a.lower().endswith("sjiswrap.exe")])
            proc = subprocess.run(argv, cwd=tree, capture_output=True, text=True)
            objs = list(out.glob("*.o"))
            return Elf.parse(objs[0].read_bytes()) if proc.returncode == 0 and objs else None
        finally:
            shutil.rmtree(work, ignore_errors=True)


INLINE_SCHEMA = "inline.v1"


def _named_record(records: list[dict], symbol: str, key: str = "name") -> list[dict]:
    from .oracle import oracle_name
    exact = [r for r in records if r.get("link") == symbol]
    if exact:
        return exact
    base = oracle_name(symbol)
    return [r for r in records if r.get(key) == base and (not r.get("link") or r.get("link") == symbol
                                                          or symbol == base)]


def caller_record(doc: dict, caller: str, ordinal: int = 0) -> dict | None:
    """The oracle's record of `caller` (its `ordinal`-th function of that unmangled name)."""
    found = _named_record(doc.get("functions") or [], caller)
    return found[ordinal] if ordinal < len(found) else (found[0] if found else None)


def callee_checks(record: dict, callee: str) -> list[dict]:
    return _named_record(record.get("checks") or [], callee, key="callee")


def small_record(doc: dict, callee: str) -> dict | None:
    found = _named_record(doc.get("small") or [], callee)
    return found[0] if found else None


def explain(doc: dict, caller: str, callee: str, ordinal: int = 0) -> list[str]:
    """Lines saying why the compiler inlines `callee` into `caller` or keeps the call, with the numbers
    (an inline.v1 document), and what moves the decision."""
    lim = doc.get("limits") or {}
    limit, local_limit = lim.get("stmts"), lim.get("locals", 1024)
    full, total_cap = lim.get("max_size", 512), lim.get("total", 7000)
    rec = caller_record(doc, caller, ordinal)
    if rec is None:
        return [f"the oracle has no record of {caller} (no inlining ran in it: `#pragma dont_inline`, or not compiled)"]
    checks = callee_checks(rec, callee)
    small = small_record(doc, callee)
    stmts = f"{small['stmts']} statement(s)" if small else None
    if not checks:
        return ["the compiler never judged a call to it here (no call in the source, or it was folded away)"]
    judged = [c for c in checks if "est" in c]
    c0 = judged[0] if judged else checks[0]
    cap = rec.get("max", full)
    budget = (f"; the caller's limit is {cap}, not {full}: its own size estimate is over {total_cap}"
              f" ({(rec.get('totals') or ['?'])[0]})") if isinstance(cap, int) and cap < full else ""
    kind = "inline" if c0.get("explicit") else "auto-inline" if c0.get("auto") else "not inline"
    out = []
    if not judged:
        if c0.get("explicit"):
            out.append("not inlined: declared inline but its body comes after the caller (no body yet when the "
                       "caller was compiled); define it before the caller to inline it")
        elif small is None:
            out.append("not inlined: no candidate (defined after the caller, in another unit, or virtual); "
                       "only a body defined before the caller can be auto-inlined")
        elif limit is not None and small["stmts"] > limit:
            out.append(f"not inlined: {stmts} as lowered, over the auto-inline limit of {limit} by "
                       f"{small['stmts'] - limit}; {small['stmts'] - limit} fewer make it a candidate")
        elif small["locals"] > local_limit:
            out.append(f"not inlined: {small['locals']} bytes of locals, over {local_limit}")
        else:
            out.append(f"not inlined: {stmts} fits the limit ({limit}), so something else refused it: "
                       "`#pragma auto_inline off`, `-inline noauto`, `...`, or a class with a destructor "
                       "passed or returned by value")
        return out
    if any(c.get("can", 6) < 3 for c in judged):
        return [f"not inlined: its body cannot be inlined (a class with a destructor by value, `...`), cost {c0['est']}"]
    over = [c for c in judged if c["est"] > c["max"]]
    rounds = sorted({c["round"] for c in checks})
    many = f"; judged in rounds {rounds} (nested or recursive inlining)" if len(rounds) > 1 else ""
    head = f"{kind}, cost {c0['est']} against the caller's limit {c0['max']}"
    late = [c for c in checks if "est" not in c and c.get("explicit") and "stmts" in c and c["round"] > c0["round"]]
    if over and len(over) == len(judged) and late:
        out.append(f"inlined in round {late[-1]['round']} only: {head}, {c0['est'] - c0['max']} over, but this "
                   f"compiler inlines an `inline` function whatever its cost in the last round (its own calls stay "
                   f"calls){budget}")
        out.append(f"with {c0['est'] - c0['max']} fewer cost units it is inlined in round {c0['round']} instead")
        return out
    if over and len(over) == len(judged):
        out.append(f"not inlined: {head}, {c0['est'] - c0['max']} over{budget}{many}")
        out.append(f"to inline it: {c0['est'] - c0['max']} fewer cost units (an operand, operator or statement each)"
                   + (" or a smaller caller" if budget else ""))
    else:
        if over:
            out.append(f"inlined at some sites only: {head}{budget}{many}")
        else:
            out.append(f"inlined: {head}, {c0['max'] - c0['est']} unit(s) of slack{budget}{many}")
        flips = [f"{c0['max'] - c0['est'] + 1} more cost unit(s)"]
        if c0.get("auto") and not c0.get("explicit") and small and limit is not None:
            flips.insert(0, f"{limit - small['stmts'] + 1} more statement(s) ({stmts} of {limit})")
            flips.append("defining it after the caller")
        out.append("to keep the call: " + ", or ".join(flips))
    return out


def oracle_doc(probe: "Probe", command: list[str]) -> dict | None:
    """The unit's compile under the oracle's inline mode (inline.v1), or None."""
    work = Path(tempfile.mkdtemp(prefix="dk-inline-oracle-"))
    try:
        out = work / "inline.json"
        probe.compile(wrap=lambda argv: list(command) + [str(out), "--inline", "--"] + list(argv))
        try:
            doc = json.loads(out.read_text())
        except (OSError, ValueError):
            return None
        return doc if doc.get("schema") == INLINE_SCHEMA else None
    finally:
        shutil.rmtree(work, ignore_errors=True)


def largest(fits, limit: int) -> int | None:
    """The largest k in [0, limit] with fits(k), assuming fits is monotone (None: not even 0)."""
    if not fits(0):
        return None
    lo, hi = 0, limit
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def smallest(fits, lo: int, hi: int) -> int | None:
    """The smallest k in [lo, hi] with fits(k) (None: not even hi)."""
    if not fits(hi):
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if fits(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def explicitly_inline(text: str, span: tuple[int, int]) -> bool:
    """Declared `inline`, or defined inside its class body."""
    from .trycmd import scopes
    head = text[span[0]:_open_brace(text, span)]
    return "inline" in head.split() or any(lo < span[0] < hi for lo, hi, _ in scopes(text))


def measure(probe, caller: str, explicit: bool = False) -> list[str]:
    """Lines describing the callee's distance from the inline limit in `caller`. `probe` compiles
    the unit with the callee edited (Probe); `explicit`: the callee is explicitly inline, which
    some versions judge by size (no-code statements nearly free) instead of by statements."""
    inlined = lambda elf: elf is not None and calls(elf, caller, probe.callee) is False
    base = probe.compile()
    if base is None:
        return ["the unit does not compile"]
    if calls(base, caller, probe.callee) is None:
        return [f"{caller} is not in our object"]
    if inlined(base):
        if explicit:
            size = largest(lambda k: inlined(probe.compile(k, STORE)), MAX_PROBES)
            if size is not None and size < MAX_PROBES:
                return [f"ours inlines it (explicitly inline: judged by size): {size} more probe store(s) fit, "
                        f"with {size + 1} it stays out of line (a store is about one simple assignment)"]
            return [f"ours inlines it (explicitly inline): {MAX_PROBES} probe stores do not change that "
                    "(this compiler inlines it at any size)"]
        slack = largest(lambda k: inlined(probe.compile(k)), MAX_PROBES)
        if slack is not None and slack < MAX_PROBES:
            return [f"ours inlines it: {slack} more statement(s) fit; with {slack + 1} more it stays out of line"]
        return [f"ours inlines it, and {MAX_PROBES} more statements do not change that"]
    spans = probe.expression_statements()
    over = smallest(lambda n: inlined(probe.compile(blank=spans[len(spans) - n:])), 1, len(spans)) if spans else None
    if over is None:
        return [f"ours calls it, and blanking its {len(spans)} expression statement(s) does not inline it (control "
                "flow or declarations carry the weight, or it is not a size decision: recursion, address taken)"]
    return [f"ours calls it: {over} statement(s) over the limit (with its last {over} expression statement(s) "
            "blanked it is inlined)"]


def run(cfg: Config, args: argparse.Namespace) -> int:
    try:
        unit = resolve_unit(cfg, args.unit or args.callee)
    except MatchError as e:
        print(f"error: {e}")
        return USAGE
    probe = Probe(cfg, unit, args.callee)
    if probe.span is None:
        print(f"error: {args.callee} is not one definition in {unit['metadata']['source_path']} or the headers it "
              "includes (pass --unit for a callee defined in another unit's header)")
        return USAGE
    if probe.header_rel and not probe.header_rel.startswith("include/"):
        print(f"error: {args.callee} is defined in {probe.header_rel}, outside include/: dk inline-cost changes "
              "only the unit's source or a header under include/")
        return USAGE
    base = probe.compile()
    if base is None:
        print(f"error: {unit['metadata']['source_path']} does not compile")
        return USAGE
    target = Elf.parse((cfg.root / unit["target_path"]).read_bytes())
    callers = [args.caller] if args.caller else sorted(set(callers_of(base, args.callee)) | set(callers_of(target, args.callee)))
    print(f"{args.callee}: defined at {probe.where}")
    if not callers:
        print("  nothing in our object or the target calls it: name the function to inline it into with --caller")
        return USAGE
    from . import oracle as _oracle
    command = None if args.probes else _oracle.runner(cfg)
    doc = None
    if command is not None and "inline" in _oracle.modes(command, Path(tempfile.mkdtemp(prefix="dk-inline-probe-"))):
        doc = oracle_doc(probe, command)
    if doc is not None:
        print(f"  (numbers read from the compiler: {doc.get('compiler', '?')}, limits {doc.get('limits')})")
    verdict = OK
    names = list(base.functions())
    for caller in callers:
        t = calls(target, caller, args.callee)
        theirs = "the target calls it" if t else "the target does not call it (inlined, or not used)" if t is False \
            else "not in the target"
        print(f"  {caller}: {theirs}")
        if doc is not None:
            sym = next((n for n in names if named(n, caller)), caller)
            same = [n for n in names if _oracle.oracle_name(n) == _oracle.oracle_name(sym)]
            lines = explain(doc, sym, args.callee, same.index(sym) if sym in same else 0)
        else:
            lines = measure(probe, caller, explicitly_inline(probe.header if probe.header_rel else probe.source, probe.span))
        for line in lines:
            print(f"    {line}")
        ours_calls = calls(base, caller, args.callee)
        if t is not None and ours_calls is not None and bool(t) != bool(ours_calls):
            verdict = MISMATCH
    hint("try <unit> <callee> VARIANTS.cpp   (natural shapes first: the playbook's \"make a callee heavier\")")
    return verdict


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("callee", help="the function whose inlining is in question (symbol or the name it is written with)")
    p.add_argument("--caller", help="the function to judge it in (default: every function either object calls it from)")
    p.add_argument("--unit", help="the unit to compile (default: the one that defines the callee)")
    p.add_argument("--probes", action="store_true", help="measure with probe compiles even when the oracle can read the numbers")


COMMAND = Command("inline-cost", "matching",
                  "why a call is inlined or kept, and how far it is from flipping (the compiler's numbers via the oracle, else probes)",
                  run, _configure)
