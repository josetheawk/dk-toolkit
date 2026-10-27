"""dk permute: search small source rewrites of one near-miss function until it matches.

Register colouring and scheduling follow the source in ways no rule fully predicts: the
order of declarations and statements, operand order, compound assignments. This tries such
rewrites at random on a private copy of the unit, compiles each with the unit's own command
(no project lock, ~0.1 s), keeps any that raise the function's score, and stops at 100%.
An exact result is correct by construction: identical machine code. The real source is only
replaced with --apply; otherwise the best version is written next to the scratch files.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import random
import re
import shutil
import time
from pathlib import Path

from ...core.command import MISMATCH, OK, USAGE, Command, hint
from ...core.config import Config
from .matching import MatchError, compile_command, resolve_unit, retarget, run_compile, score

_DECL = re.compile(r"^\s*(?:const\s+|unsigned\s+|signed\s+|register\s+)*[A-Za-z_][\w:<>]*[\s*&]+\w+(?:\s*\[[^\]]*\])?\s*(?:=[^;]*)?;\s*$")
_COMMUTE = re.compile(r"(?P<a>[\w.>\-\[\]]+) (?P<op>[+*&|^]) (?P<b>[\w.>\-\[\]]+)")
_COMPOUND = re.compile(r"^(?P<ind>\s*)(?P<lhs>[\w.>\-\[\]]+) (?P<op>[+\-*|&^])= (?P<rhs>[^;]+);\s*$")
_EXPANDED = re.compile(r"^(?P<ind>\s*)(?P<lhs>[\w.>\-\[\]]+) = (?P=lhs) (?P<op>[+\-*|&^]) (?P<rhs>[^;]+);\s*$")


def body_span(text: str, name: str) -> tuple[int, int] | None:
    """(first, last) line index of the body of the function `name` (its unqualified name)."""
    lines = text.splitlines()
    head = re.compile(rf"(^|[\s:*&]){re.escape(name)}\s*\(")
    for i, line in enumerate(lines):
        if head.search(line) and not line.rstrip().endswith(";") and not line.lstrip().startswith(("//", "#")):
            depth, start = 0, None
            for j in range(i, len(lines)):
                for ch in lines[j]:
                    if ch == "{":
                        depth += 1
                        start = j if start is None else start
                    elif ch == "}":
                        depth -= 1
                        if depth == 0 and start is not None:
                            return start + 1, j - 1
                if j > i + 3 and start is None:
                    break
    return None


def _simple(line: str) -> bool:
    s = line.strip()
    return s.endswith(";") and not s.startswith(("return", "break", "continue", "goto", "case", "default", "//", "#")) \
        and "{" not in s and "}" not in s


def mutate(lines: list[str], lo: int, hi: int, rng: random.Random) -> list[str] | None:
    """One random rewrite inside lines[lo..hi], or None if the chosen one does not apply."""
    out = list(lines)
    kind = rng.choice(("swap", "swap", "move", "move", "decl", "commute", "compound", "compare"))
    i = rng.randint(lo, hi)
    indent = lambda s: len(s) - len(s.lstrip())
    if kind in ("move", "decl"):
        # move a statement (or, for decl, a declaration) up to 6 lines, staying at its depth
        if not (_simple(out[i]) and (kind == "move" or _DECL.match(out[i]))):
            return None
        j = max(lo, min(hi, i + rng.choice((-6, -5, -4, -3, -2, 2, 3, 4, 5, 6))))
        step = 1 if j > i else -1
        if any(not _simple(out[k]) or indent(out[k]) != indent(out[i]) for k in range(i + step, j + step, step)):
            return None
        line = out.pop(i)
        out.insert(j, line)
        return out
    if kind == "compare":
        m = re.search(r"\((?P<a>[^()=!<>]+?) (?P<op>==|!=) (?P<b>[^()=!<>]+?)\)", out[i])
        if not m:
            return None
        out[i] = out[i][:m.start()] + f"({m['b']} {m['op']} {m['a']})" + out[i][m.end():]
        return out
    if kind == "swap" and i < hi:
        a, b = out[i], out[i + 1]
        if _simple(a) and _simple(b) and indent(a) == indent(b):
            out[i], out[i + 1] = b, a
            return out
        return None
    line = out[i]
    if kind == "commute":
        ms = list(_COMMUTE.finditer(line))
        if not ms or line.strip().startswith(("#", "//")):
            return None
        m = rng.choice(ms)
        out[i] = line[:m.start()] + f"{m['b']} {m['op']} {m['a']}" + line[m.end():]
        return out
    m = _COMPOUND.match(line)
    if m:
        out[i] = f"{m['ind']}{m['lhs']} = {m['lhs']} {m['op']} {m['rhs']};"
        return out
    m = _EXPANDED.match(line)
    if m:
        out[i] = f"{m['ind']}{m['lhs']} {m['op']}= {m['rhs']};"
        return out
    return None


SLOTS = 4                                   # concurrent searches per machine; each keeps a core busy


def take_slot(root: Path, slots: int = SLOTS):
    """An open, locked slot file (held until the process exits), or None if all are taken."""
    d = root / "build" / "dk" / "permute-slots"
    d.mkdir(parents=True, exist_ok=True)
    for n in range(slots):
        f = open(d / f"slot{n}", "w")
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            f.close()
            continue
        f.write(str(os.getpid()))
        f.flush()
        return f
    return None


def run(cfg: Config, args: argparse.Namespace) -> int:
    slot = take_slot(cfg.root)
    if slot is None:
        print(f"busy: {SLOTS} permute searches are already running on this machine; work on another "
              "function and try later (searches compete for the same cores)")
        return USAGE
    try:
        unit = resolve_unit(cfg, args.unit)
        command = compile_command(cfg, unit["base_path"], locked=False)
    except MatchError as e:
        print(f"error: {e}")
        return USAGE
    source = cfg.root / unit["metadata"]["source_path"]
    text = source.read_text()
    plain = re.sub(r"__.*$", "", args.function) if "__" in args.function.lstrip("_") else args.function
    span = body_span(text, plain)
    if span is None:
        print(f"error: no definition of {plain} found in {cfg.rel(source)}")
        return USAGE
    work = cfg.root / "build" / "dk" / "permute" / re.sub(r"\W", "_", args.function)[:60]
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    candidate = work / source.name

    def measure(content: str) -> float | None:
        candidate.write_text(content)
        argv = retarget(command, work)
        argv[argv.index("-c") + 1] = str(candidate)
        for o in work.glob("*.o"):
            o.unlink()
        ok, _ = run_compile(cfg, argv)
        objs = list(work.glob("*.o"))
        if not ok or not objs:
            return None
        try:
            s = score(cfg, unit["name"], cfg.root / unit["target_path"], objs[0])
        except MatchError:
            return None
        return next((f.percent for f in s.functions
                     if f.name == args.function or f.name.startswith(args.function + "__")), None)

    best_text, best = text, measure(text)
    if best is None:
        print(f"error: {args.function} does not compile from a scratch copy or is not in {unit['name']}")
        return USAGE
    print(f"start: {best:.2f}%  ({args.function}, lines {span[0] + 1}-{span[1] + 1})")
    rng = random.Random(args.seed)
    deadline = time.time() + min(args.minutes, 10.0) * 60
    tried = 0
    lines = best_text.splitlines()
    while time.time() < deadline and best < 100.0:
        cand = lines
        for _ in range(rng.randint(1, 3)):            # stack a few rewrites to leave local optima
            nxt = mutate(cand, span[0], span[1], rng)
            cand = nxt or cand
        if cand is lines:
            continue
        tried += 1
        pct = measure("\n".join(cand) + "\n")
        if pct is not None and pct > best:
            best, lines = pct, cand
            best_text = "\n".join(cand) + "\n"
            print(f"  {best:.2f}% after {tried} tries")
    (work / f"best_{source.name}").write_text(best_text)
    print(f"best: {best:.2f}% after {tried} tries; written to {cfg.rel(work / ('best_' + source.name))}")
    if best == 100.0 and args.apply:
        source.write_text(best_text)
        print(f"applied to {cfg.rel(source)}")
        hint(f"check {args.unit} --private")
    return OK if best == 100.0 else MISMATCH


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("unit", help="unit (source path or name)")
    p.add_argument("function", help="function symbol to raise (as dk check lists it)")
    p.add_argument("--minutes", type=float, default=5.0, help="time budget (default 5, at most 10)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--apply", action="store_true", help="write an exact result into the unit's source")


COMMAND = Command("permute", "matching", "search source rewrites of one near-miss function (private, no lock)", run, _configure)
