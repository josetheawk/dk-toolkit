"""dk try: compile alternative versions of one function and score each, without touching the source.

  ./dk try <unit> <fn> VARIANTS.cpp [--apply]
  ./dk try <unit> <fn> [VARIANTS.cpp] [--replace OLD NEW]… [--atoms FILE] [--grid] [--oracle]   (tryatoms.py)

VARIANTS.cpp holds the alternatives, separated by lines `// ---`. Each one replaces the function's
definition in the unit's source (it may bring its own helpers, types or statics with it), is
compiled privately in parallel (a function it defines again, such as an inline helper it rewrites,
replaces the source's copy), and is reported with the function's score and any function of the
unit that it makes worse. `--apply` writes the best one into the source when it is exact and
nothing else got worse.

Why: the inner loop of matching by hand is "write three versions of this body, see which one
scores best"; without this every agent scripts it again (copy the file, splice, compile, score).
"""

from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ...core import cbody
from ...core.command import BUILD_ERROR, MISMATCH, OK, USAGE, Command, hint
from ...core.config import Config
from .closecmd import plain_name
from .jobs import resolve
from .matching import MatchError, resolve_unit, score
from .reorder import compile_variant

SEPARATOR = "// ---"


def split_variants(text: str) -> list[str]:
    chunks, cur = [], []
    for line in text.splitlines(keepends=True):
        if line.strip() == SEPARATOR:
            chunks.append("".join(cur))
            cur = []
        else:
            cur.append(line)
    chunks.append("".join(cur))
    return [c.strip("\n") + "\n" for c in chunks if c.strip()]


def scopes(src: str) -> list[tuple[int, int, str]]:
    """(open, close, name) of each namespace, class, struct or union body in the file."""
    out, stack, head = [], [], 0
    toks = cbody.tokens(src)
    for i, t in enumerate(toks):
        if t.kind == "pp" or (t.kind == "op" and t.text in (";", "}")):
            if t.kind == "op" and t.text == "}" and stack:
                opened, name = stack.pop()
                if name:
                    out.append((opened, t.start, name))
            head = i + 1
        elif t.kind == "op" and t.text == "{":
            words = [x.text for x in toks[head:i]]
            name = ""
            for k, w in enumerate(words):
                if w in ("namespace", "class", "struct", "union") and k + 1 < len(words) and \
                        cbody.IDENT.fullmatch(words[k + 1]) and "=" not in words:
                    name = words[k + 1]
                    break
            stack.append((t.start, name))
            head = i + 1
    return out


def qualified(f, spans: list[tuple[int, int, str]]) -> list[str]:
    """The function's full name as parts: the scopes it is defined in, then its name as written."""
    outer = [name for lo, hi, name in sorted(spans) if lo < f.head_start < hi]
    return outer + f.name.split("::")


def _param_count(src: str, f) -> int | None:
    head = src[f.head_start:f.open]
    last = f.name.split("::")[-1]
    at = head.rfind(last)
    lo = head.find("(", at if at >= 0 else 0)
    if lo < 0:
        return None
    depth, parts, cur = 0, [], ""
    for c in head[lo:]:
        if c in "(<[":
            depth += 1
            if depth == 1:
                continue
        elif c in ")>]":
            depth -= 1
            if depth == 0:
                break
        if c == "," and depth == 1:
            parts.append(cur)
            cur = ""
        else:
            cur += c
    parts.append(cur)
    parts = [p.strip() for p in parts]
    return 0 if parts in ([""], ["void"]) else len(parts)


def _suffix(a: list[str], b: list[str]) -> bool:
    """One qualifier ends the other (a namespace may be written around the definition or not)."""
    n = min(len(a), len(b))
    return n > 0 and a[-n:] == b[-n:]


def candidates(src: str, symbol: str) -> list:
    """The definitions in `src` that the symbol may name: by name, then (when that leaves several or
    a member of another class) by the mangled class qualifiers and the parameter count."""
    from .demangle import demangle
    plain = plain_name(symbol)
    fns = cbody.functions(src)
    hits = [f for f in fns if f.name.split("::")[-1] == plain or f.name == plain]
    try:
        dm = demangle(symbol)
    except Exception:                                # not a mangled name
        dm = None
    want = list(dm.quals) if dm is not None and dm.quals else []
    if not hits or not want:
        return hits
    spans = scopes(src)
    def qual(f):
        return qualified(f, spans)[:-1]
    if len(hits) == 1:
        q = qual(hits[0])
        return hits if not q or _suffix(q, want) else []
    hits = [f for f in hits if _suffix(qual(f), want)]
    if len(hits) > 1 and dm is not None:
        same = [f for f in hits if _param_count(src, f) == len(dm.params)]
        hits = same or hits
    return hits


def in_header(cfg: Config, src: str, symbol: str, seen: set | None = None, depth: int = 0) -> str | None:
    """The header (followed through the source's quoted includes) that defines the function."""
    import re
    seen = set() if seen is None else seen
    for inc in re.findall(r'^\s*#\s*include\s+"([^"]+)"', src, re.M):
        for base in ("include", "src"):
            p = cfg.root / base / inc
            if p.is_file() and p not in seen:
                seen.add(p)
                text = p.read_text(errors="replace")
                if candidates(text, symbol):
                    return cfg.rel(p)
                if depth < 3 and (found := in_header(cfg, text, symbol, seen, depth + 1)):
                    return found
                break
    return None


def definition(src: str, symbol: str) -> tuple[int, int] | None:
    """The span (from its head to past its closing brace) of the function's definition."""
    hits = candidates(src, symbol)
    if len(hits) != 1:
        return None
    f = hits[0]
    return f.head_start, f.close + 1


def spliced(src: str, span: tuple[int, int], chunk: str) -> str:
    if span[0] == span[1]:                           # a new function: its own paragraph
        return src[:span[0]] + chunk.rstrip("\n") + "\n\n" + src[span[1]:]
    return src[:span[0]] + chunk.rstrip("\n") + src[span[1]:]


def without_redefined(src: str, symbol: str, chunk: str) -> tuple[str, list[str]]:
    """`src` without the other functions the variant defines again (a helper it rewrites), so
    the variant's version replaces them instead of clashing: (source, their names)."""
    plain = plain_name(symbol)
    mine = {f.name for f in cbody.functions(chunk)} - {plain}
    cuts = []
    for f in cbody.functions(src):
        if f.name in mine and f.name.split("::")[-1] != plain:
            cuts.append(f)
    by_name: dict[str, list] = {}
    for f in cuts:
        by_name.setdefault(f.name, []).append(f)
    cuts = [fs[0] for fs in by_name.values() if len(fs) == 1]   # an overload set stays as it is
    for f in sorted(cuts, key=lambda f: f.head_start, reverse=True):
        end = f.close + 1
        while end < len(src) and src[end] == "\n" and end - f.close <= 2:
            end += 1                                 # and the blank line after it
        src = src[:f.head_start] + src[end:]
    return src, sorted(f.name for f in cuts)


def variant_text(src: str, symbol: str, chunk: str, span: tuple[int, int],
                 locate=None) -> str:
    """The unit's source with the variant in place of the function (and of any helper it
    redefines). `locate(text)` finds the function's span again after a helper is cut."""
    base, cut = without_redefined(src, symbol, chunk)
    if not cut:
        return spliced(src, span, chunk)
    again = (locate or (lambda t: definition(t, symbol)))(base)
    return spliced(base, again if again is not None else span, chunk)


def insertion_point(src: str, address: int, addresses: dict[str, int]) -> int:
    """Where a function not yet written goes: before the first defined function at a higher
    address (with the comment lines above it), else at the end of the file."""
    later = []
    for f in cbody.functions(src):
        plain = f.name.split("::")[-1]
        a = addresses.get(plain)
        if a is not None and a > address:
            later.append((a, f.head_start))
    if not later:
        return len(src.rstrip("\n")) + 1 if src.endswith("\n") else len(src)
    at = src.rfind("\n", 0, min(later)[1]) + 1
    while at > 0:
        prev = src.rfind("\n", 0, at - 1) + 1
        if src[prev:at].strip().startswith("//"):
            at = prev
        else:
            break
    return at


def locate_target(cfg: Config, src: str, source: Path, function: str):
    """Where the function's definition is in the unit's source: (span, locate, note). An unwritten
    function gets an empty span at its address and a `locate(text)` that finds that place again in
    a changed source; `note` says so. Raises MatchError when the function is not one definition."""
    span = definition(src, function)
    if span is not None:
        return span, None, ""
    from .dtk_symbols import read_symbols
    symbols = read_symbols(cfg.root / cfg.platform_options["symbols"])
    me = next((x for x in symbols if x.name == function), None)
    several = candidates(src, function)
    header = None if several else in_header(cfg, src, function)
    if header is not None:
        raise MatchError(f"{function} is defined in {header}, not in {cfg.rel(source)}: dk try changes "
                         f"only the unit's source (edit the header, or define the function out of line in the unit)")
    if me is None or several:
        spans = scopes(src)
        named = ", ".join("::".join(qualified(f, spans)) for f in several[:4])
        raise MatchError(f"{function} is not one definition in {cfg.rel(source)}"
                         + (f" (it could be any of {named})" if several else " nor a known symbol"))
    addresses = {}
    for x in symbols:
        if x.type == "function" and x.section == ".text":
            addresses.setdefault(plain_name(x.name) or x.name, x.address)
    at = insertion_point(src, me.address, addresses)

    def locate(text: str) -> tuple[int, int]:
        here = insertion_point(text, me.address, addresses)
        return here, here
    return (at, at), locate, (f"{function} is not written yet: each variant goes in by address "
                              f"(line {src.count(chr(10), 0, at) + 1})")


def label(current: str, variant: str) -> str:
    """The variant's first line that differs from the current definition (whitespace folded)."""
    import difflib
    a = [" ".join(x.split()) for x in current.splitlines()]
    b = [" ".join(x.split()) for x in variant.splitlines()]
    for tag, _i1, _i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag in ("replace", "insert") and j1 < len(b):
            return f"line {j1 + 1}: {b[j1][:70]}"
        if tag == "delete":
            return f"line {j1 + 1}: (lines removed)"
    return "(same as the current definition)"


def diff_rows(cfg: Config, unit: dict, obj: Path, function: str) -> list[str]:
    """The function's target-vs-ours instruction rows (context 2) and stack map for one compiled object."""
    from .diffcmd import function_rows, stack_rows
    from .regs import _diff
    pair = _diff(cfg, unit, obj, function)
    if pair is None:
        return []
    return function_rows(pair[0], pair[1], 2) + stack_rows(pair[0], pair[1])


def clip_rows(rows: list[str], cap: int) -> str:
    """The rows joined, at most `cap` of them (0 = all), with a note when some are left out."""
    if cap <= 0 or len(rows) <= cap:
        return "\n".join(rows)
    return "\n".join(rows[:cap]) + f"\n    … {len(rows) - cap} more rows (--show-rows 0 prints them all)"


def show_variant(cfg: Config, unit: dict, text: str, function: str, work: Path, cap: int) -> None:
    """Compile one variant's unit text privately and print its diff (`--show` after an atoms run)."""
    obj, _err = compile_variant(cfg, unit, text, work)
    try:
        rows = diff_rows(cfg, unit, obj, function) if obj is not None else []
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(clip_rows(rows, cap) if rows else "    (no diff rows: the variant does not compile or the function is missing)")


def run(cfg: Config, args: argparse.Namespace) -> int:
    try:
        unit = resolve_unit(cfg, args.unit)
    except MatchError as e:
        print(f"error: {e}")
        return USAGE
    source = cfg.root / unit["metadata"]["source_path"]
    src = source.read_text()
    try:
        span, locate, note = locate_target(cfg, src, source, args.function)
    except MatchError as e:
        print(f"error: {e}")
        return USAGE
    if note:
        print(note)
    variants = split_variants(Path(args.variants).read_text()) if args.variants else []
    if args.replace or args.atoms or args.grid or args.oracle:
        from .tryatoms import run_atoms
        return run_atoms(cfg, args, unit, src, span, [variant_text(src, args.function, v, span, locate) for v in variants])
    if not variants:
        print(f"error: no variants in {args.variants} (separate them with lines `{SEPARATOR}`)")
        return USAGE
    work = cfg.root / "build" / "dk" / "try-variants"
    texts = [src] + [variant_text(src, args.function, v, span, locate) for v in variants]

    shown: list[str] = []

    def one(k: int):
        obj, err = compile_variant(cfg, unit, texts[k], work / f"v{k}")
        try:
            if obj is None:
                return None, (err.strip().splitlines() or ["compile error"])[-3:]
            if k == args.show:
                shown.extend(diff_rows(cfg, unit, obj, args.function))
            return score(cfg, unit["name"], cfg.root / unit["target_path"], obj), []
        except MatchError as e:
            return None, [str(e)]
        finally:
            shutil.rmtree(work / f"v{k}", ignore_errors=True)

    with ThreadPoolExecutor(max_workers=resolve(args.jobs, cfg)) as pool:
        results = list(pool.map(one, range(len(texts))))
    base, _ = results[0]
    if base is None:
        print("error: the unit's current source does not compile")
        return BUILD_ERROR
    before = {f.name: f.percent for f in base.functions}
    print(f"{args.function}: now {before.get(args.function, 0.0):.2f}%" + ("" if span[0] != span[1] else " (not written)"))
    best = None
    for k, (s, err) in enumerate(results[1:], 1):
        first = label(src[span[0]:span[1]], variants[k - 1]) if span[0] != span[1] else f"variant {k}"
        if s is None:
            print(f"  variant {k}: does not compile: {' / '.join(x.strip() for x in err)[:160]}")
            continue
        got = {f.name: f.percent for f in s.functions}
        pct = got.get(args.function, 0.0)
        worse = [n for n, p in before.items() if n != args.function and got.get(n, -1.0) < p]
        print(f"  variant {k}: {pct:6.2f}%" + (f"  (worse: {', '.join(worse[:4])})" if worse else "") + f"   {first}")
        if not worse and (best is None or pct > best[1]):
            best = (k, pct)
    if shown:
        print(f"variant {args.show}, target vs ours:")
        print(clip_rows(shown, args.show_rows))
    if best is None:
        return MISMATCH
    k, pct = best
    if args.apply:
        if pct == 100.0:
            source.write_text(texts[k])
            print(f"  applied variant {k} to {cfg.rel(source)}")
            hint(f"check {args.unit} --function {args.function} --private")
            return OK
        print(f"  not applied: the best variant ({k}) is {pct:.2f}%")
    return OK if pct == 100.0 else MISMATCH


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("unit", help="unit (source path or name)")
    p.add_argument("function", help="function symbol, as dk check lists it")
    p.add_argument("variants", nargs="?", help=f"a file of alternative definitions, separated by lines `{SEPARATOR}` "
                                               "(optional with --replace / --atoms)")
    p.add_argument("--apply", action="store_true", help="write the best variant when it is exact and nothing got worse")
    p.add_argument("--show", type=int, default=-1, metavar="K", help="also print variant K's instruction diff (1 = the "
                   "first; with atoms: the K-th in the ranking, one more compile)")
    p.add_argument("--show-rows", type=int, default=80, metavar="N", help="rows --show prints at most (default 80; 0 = all)")
    p.add_argument("--jobs", type=int, default=0, help="parallel compiles (default: dk.toml [platform] jobs, DK_JOBS, "
                                                       "else the idle cores)")
    p.add_argument("--replace", nargs=2, action="append", metavar=("OLD", "NEW"),
                   help="a line-level edit (atom): OLD, once in the function (else once in the file; whitespace may "
                        "differ), becomes NEW; repeatable; each atom is one variant, or see --grid")
    p.add_argument("--atoms", metavar="FILE", help="atoms from a file: blocks separated by `// ---`, each OLD lines, "
                                                   "a line `// =>`, NEW lines (first line `// label: …` optional)")
    p.add_argument("--grid", action="store_true", help="every combination of 1 to --grid-max atoms (best singles' "
                   "combinations first, repeated code skipped by a batched compile, at most --max-compiles)")
    p.add_argument("--grid-max", type=int, default=0, metavar="K", help="largest combination (default 3; 2 for a "
                                                                        "function over 2 KB)")
    p.add_argument("--max-compiles", type=int, default=150, metavar="N", help="compile cap with atoms (default 150)")
    p.add_argument("--oracle", action="store_true", help="score each variant under the compiler oracle (DK_ORACLE, "
                   "contract v2): values coloured as the target N/M, the solver's rules left K with their values, "
                   "and close v2's ranking key next to the fuzzy %%")
    p.add_argument("--top", type=int, default=20, help="variants listed with atoms (default 20)")
    p.add_argument("--json", metavar="PATH", help="with atoms or --oracle: every variant's verdict as JSON")


COMMAND = Command("try", "matching", "compile alternative versions of one function (a file, `// ---` between them) "
                  "and score each, without touching the source", run, _configure)
