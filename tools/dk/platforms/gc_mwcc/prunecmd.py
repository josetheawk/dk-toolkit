"""dk prune: drop the top-level declarations of a unit's own source that its object does not need.

  ./dk prune <unit> [--apply] [--jobs N]

Each top-level item of the unit's source file (a struct, class, union or enum definition, a
typedef, an extern or prototype, a global, an inline function; inside `extern "C" { }` and
namespace blocks too) is tried without it, with the comment lines right above it: kept out only
when the object compiles byte-identical (the same code, data, symbols and relocations). Only the
unit's own file changes, never a header. Greedy: each round compiles every remaining item's
removal privately in parallel, applies all that pass together, proves the result identical (or
applies them one by one when together they are not), and repeats until a round removes nothing.
Without --apply the pruned source is written to build/dk/prune/.

Why: split units inherit every declaration their parent needed (one TU's 839 dead lines after
a split), and dead declarations hide which types a TU really uses.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from ...core import cbody
from ...core.command import BUILD_ERROR, MISMATCH, OK, USAGE, Command, hint
from ...core.config import Config
from .jobs import resolve
from .matching import MatchError, resolve_unit
from .reorder import compile_variant


@dataclass
class Item:
    start: int                 # the removal span: the item and the comment lines right above it
    end: int
    label: str                 # its first line, for the report
    kind: str                  # "decl", "inline" (an inline function), "function", "pp"
    key: str = ""              # its text, whitespace folded (finds it again after other removals)


def _line_start(src: str, at: int) -> int:
    return src.rfind("\n", 0, at) + 1


def _span(src: str, start: int, end: int) -> tuple[int, int]:
    """The item with its own line(s) when it has them to itself, and the comment lines right above."""
    lo, hi = start, end
    ls = _line_start(src, start)
    nl = src.find("\n", end)
    nl = len(src) if nl < 0 else nl
    if not src[ls:start].strip() and not src[end:nl].strip():
        lo, hi = ls, min(len(src), nl + 1)
        while lo > 0:
            prev = _line_start(src, lo - 1)
            if src[prev:lo].strip().startswith("//"):
                lo = prev
            else:
                break
    return lo, hi


def items(src: str) -> list[Item]:
    """The file's top-level items (and those inside extern "C" and namespace blocks), in order."""
    toks = cbody.tokens(src)
    funcs = {f.open: f for f in cbody.functions(src)}
    out: list[Item] = []
    i, head = 0, 0

    def match(k: int) -> int:
        depth = 0
        for j in range(k, len(toks)):
            if toks[j].kind == "op" and toks[j].text in "{}":
                depth += 1 if toks[j].text == "{" else -1
                if depth == 0:
                    return j
        return len(toks) - 1

    def add(a: int, b: int, kind: str) -> None:
        start, end = toks[a].start, toks[b].end
        first = src[start:end].split("\n", 1)[0].strip()
        lo, hi = _span(src, start, end)
        out.append(Item(lo, hi, first[:80], kind, " ".join(src[start:end].split())))

    while i < len(toks):
        t = toks[i]
        if t.kind == "pp":
            add(i, i, "pp")
            i = head = i + 1
            continue
        if t.kind == "op" and t.text == "}":               # the end of a container block
            i = head = i + 1
            continue
        if t.kind == "op" and t.text == "{":
            words = [x.text for x in toks[head:i]]
            if words in (["extern", '"C"'],) or (words[:1] == ["namespace"] and len(words) <= 2):
                i = head = i + 1                             # into the container
                continue
            close = match(i)
            if t.start in funcs:                             # a function definition ends at its body
                text = src[toks[head].start:t.start]
                add(head, close, "inline" if "inline" in text.split() else "function")
                i = head = close + 1
                continue
            i = close + 1                                    # a struct body or an initializer: on to the `;`
            continue
        if t.kind == "op" and t.text == ";":
            if i > head:
                add(head, i, "decl")
            i = head = i + 1
            continue
        i += 1
    return out


def removable(item: Item) -> bool:
    """What is worth a compile: not preprocessor lines or out-of-line function definitions (they
    define symbols, so their absence always shows)."""
    return item.kind in ("decl", "inline")


def without(src: str, item: Item) -> str:
    return src[:item.start] + src[item.end:]


def without_all(src: str, chosen: list[Item]) -> str:
    out, last = [], 0
    for it in sorted(chosen, key=lambda x: x.start):
        if it.start < last:
            continue
        out.append(src[last:it.start])
        last = it.end
    out.append(src[last:])
    return "".join(out)


class Pruner:
    def __init__(self, cfg: Config, unit: dict, jobs: int, log=print):
        self.cfg, self.unit, self.jobs, self.log = cfg, unit, jobs, log
        self.work = cfg.root / "build" / "dk" / "prune" / f"{os.getpid()}"
        self.counter = 0

    def digest(self, text: str, slot: str) -> str | None:
        work = self.work / slot
        try:
            obj, _ = compile_variant(self.cfg, self.unit, text, work)
            return hashlib.sha1(obj.read_bytes()).hexdigest() if obj is not None else None
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def run(self, text: str, rounds: int = 4) -> tuple[str, list[Item], int]:
        """(pruned source, removed items, compiles)."""
        want = self.digest(text, "base")
        if want is None:
            raise MatchError("the unit's source does not compile")
        compiles, removed = 1, []
        for _ in range(rounds):
            cands = [it for it in reversed(items(text)) if removable(it)]
            if not cands:
                break
            with ThreadPoolExecutor(max_workers=max(1, self.jobs)) as pool:
                got = list(pool.map(lambda kv: self.digest(without(text, kv[1]), f"c{kv[0]}"), enumerate(cands)))
            compiles += len(cands)
            passing = [it for it, d in zip(cands, got) if d == want]
            if not passing:
                break
            together = without_all(text, passing)
            compiles += 1
            if self.digest(together, "all") == want:
                text, removed = together, removed + passing
            else:                                        # one by one, each proved on the result so far
                for it in passing:
                    fresh = next((x for x in items(text) if x.key == it.key), None)
                    if fresh is None:
                        continue
                    trial = without(text, fresh)
                    compiles += 1
                    if self.digest(trial, "one") == want:
                        text, removed = trial, removed + [fresh]
            self.log(f"  round: {len(passing)} removable, {len(removed)} removed so far ({compiles} compiles)")
        shutil.rmtree(self.work, ignore_errors=True)
        return text, removed, compiles


def run(cfg: Config, args: argparse.Namespace) -> int:
    try:
        unit = resolve_unit(cfg, args.unit)
    except MatchError as e:
        print(f"error: {e}")
        return USAGE
    source = cfg.root / unit["metadata"]["source_path"]
    text = source.read_text()
    pruner = Pruner(cfg, unit, resolve(args.jobs, cfg))
    try:
        new, removed, compiles = pruner.run(text)
    except MatchError as e:
        print(f"error: {e}")
        return BUILD_ERROR
    lines = text.count("\n") - new.count("\n")
    print(f"{cfg.rel(source)}: {len(removed)} declaration(s) not needed ({lines} lines), object byte-identical "
          f"without them ({compiles} compiles)")
    for it in sorted(removed, key=lambda x: x.start):
        print(f"  - {it.label}")
    if not removed:
        return OK
    if args.apply:
        source.write_text(new)
        print(f"  applied to {cfg.rel(source)}")
        hint(f"check {args.unit} --private", f"save {args.unit} --note \"pruned dead declarations\"")
    else:
        out = cfg.root / "build" / "dk" / "prune" / source.name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(new)
        print(f"  pruned source written to {cfg.rel(out)} (--apply writes the unit's file)")
    return OK


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("unit", help="unit (source path or name)")
    p.add_argument("--apply", action="store_true", help="write the pruned source into the unit's file")
    p.add_argument("--jobs", type=int, default=0, help="parallel compiles (default: dk.toml [platform] jobs, DK_JOBS, "
                                                       "else the idle cores)")


COMMAND = Command("prune", "matching", "drop a unit's own top-level declarations its object does not need "
                  "(byte-identical without them)", run, _configure)
