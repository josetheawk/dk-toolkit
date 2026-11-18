"""dk inline-cost: how far a callee is from the compiler's inline limit, in its unit's own build.

  ./dk inline-cost <callee> [--caller F] [--unit U]

MWCC decides auto-inlining by the callee's own source: it counts statements as lowered (an
expression statement 1, `if` 2, `if/else` 4, a call to an inline helper 1 whatever the helper's
size; `;`, `{}`, `do {} while (0)` and declarations without an initializer 0) against a limit
that depends on the compiler version, and caps the size of the expressions. A callee the target
calls but ours inlines is "N statements too light"; one the target inlines but ours calls is
"N too heavy". This measures N on a private copy, without touching the source:
  - inlined now: how many no-code statements (`((void)0);`) still fit before it stops being
    inlined (the slack; one more and it stays out of line);
  - called now: how many of its expression statements, blanked from the end, make it inlined
    (the statements it is over the limit).
Explicitly inline functions are judged by size on some versions, where no-code statements are
free: then the slack is also given in probe stores (`*(volatile char*)0 = 0;`).
The probes exist only in the private compile; the method for using the number
(natural shapes first) is in the platform playbook.
"""

from __future__ import annotations

import argparse
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

    def compile(self, count: int = 0, probe: str = STATEMENT, blank: list[tuple[int, int]] = ()) -> Elf | None:
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
            proc = subprocess.run(argv, cwd=tree, capture_output=True, text=True)
            objs = list(out.glob("*.o"))
            return Elf.parse(objs[0].read_bytes()) if proc.returncode == 0 and objs else None
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
    verdict = OK
    for caller in callers:
        t = calls(target, caller, args.callee)
        theirs = "the target calls it" if t else "the target does not call it (inlined, or not used)" if t is False \
            else "not in the target"
        print(f"  {caller}: {theirs}")
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


COMMAND = Command("inline-cost", "matching",
                  "how many statements a callee is from the auto-inline limit in its unit's build (probes in a private compile)",
                  run, _configure)
