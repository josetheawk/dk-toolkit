"""dk rename OLD NEW [OLD NEW ...]: rename symbols in symbols.txt and prove the build still matches."""

from __future__ import annotations

import argparse
import re

from . import freshness as fresh
from ...core.command import MISMATCH, OK, USAGE, Command, hint
from ...core.config import Config
from ...core.util import run_ninja, tail
from .unitcmd import Transaction

_SYMBOL_NAME = re.compile(r"^[A-Za-z_.$@?][\w.$@?<>,:*&-]*$")


def rename_text(text: str, pairs: list[tuple[str, str]], local: bool = False) -> tuple[str, list[str]]:
    """Apply renames to symbols.txt text. Returns (new text, errors).

    With `local`, the new names may repeat existing ones: static functions/objects of the same
    name in different TUs. Every symbol carrying such a name becomes scope:local (the build
    check then proves nothing outside its own TU referenced it).
    """
    lines = text.splitlines(keepends=True)
    names = [line.split(" = ", 1)[0] for line in lines if " = " in line]
    # OLD may be an address (0x80391540): the one symbol there, even if its name repeats
    by_address: dict[int, int] = {}
    for i, line in enumerate(lines):
        m = re.search(r" = [^:]+:0x([0-9A-Fa-f]+);", line)
        if m:
            by_address.setdefault(int(m[1], 16), i)
    at_line: dict[int, str] = {}
    resolved = []
    for old, new in pairs:
        if re.fullmatch(r"0x[0-9A-Fa-f]+", old):
            i = by_address.get(int(old, 16))
            if i is None:
                return text, [f"no symbol at {old}"]
            at_line[i] = new
            resolved.append((lines[i].split(" = ", 1)[0], new, i))
        else:
            resolved.append((old, new, None))
    errors = []
    taken = set(names)
    for old, new, line_no in resolved:
        if line_no is not None:
            if new in taken and new != old and not local:
                errors.append(f"{new!r} already exists (a static of that name in another TU: use --local)")
            taken.add(new)
            continue
        if not _SYMBOL_NAME.match(new):
            errors.append(f"{new!r} is not a valid symbol name")
        if names.count(old) != 1:
            errors.append(f"{old!r} occurs {names.count(old)} times (need exactly 1)")
        if new in taken and new != old and not local:
            errors.append(f"{new!r} already exists (a static of that name in another TU: use --local)")
        taken.add(new)
    if errors:
        return text, errors
    mapping = {old: new for old, new, line_no in resolved if line_no is None}
    shared = {new for _, new, _ in resolved} & set(names) if local else set()
    out = []
    for i, line in enumerate(lines):
        name, sep, rest = line.partition(" = ")
        target = at_line.get(i, mapping.get(name)) if sep else None
        if sep and (target is not None and local or name in shared):
            rest = re.sub(r"scope:\w+", "scope:local", rest) if "scope:" in rest else rest.rstrip("\n") + " scope:local\n"
        if sep and target is not None:
            line = f"{target}{sep}{rest}"
        elif sep and name in shared:
            line = f"{name}{sep}{rest}"
        out.append(line)
    return "".join(out), []


def stage_renames(cfg: Config, tx: Transaction, pairs: list[tuple[str, str]], local: bool = False,
                  sources: bool = True) -> tuple[list[str], list[str], list[str]]:
    """Write renames into a transaction (symbols.txt + placeholder references in sources).
    Returns (errors, source files changed, references to update by hand)."""
    path = cfg.root / cfg.platform_options["symbols"]
    new_text, errors = rename_text(path.read_text(), pairs, local)
    if errors:
        return errors, [], []
    tx.write(path, new_text)
    changed, manual = rewrite_sources(cfg, pairs, tx) if sources else ([], [])
    return [], changed, manual


_MANGLED = re.compile(r"__(?:\d+[A-Za-z_]|F|Q\d)")
_PLACEHOLDER = re.compile(r"^(?:fn|lbl)_[0-9A-F]{8}$")
SOURCE_SUFFIXES = (".c", ".cpp", ".cp", ".h", ".hpp", ".inc")


def rewrite_sources(cfg: Config, pairs: list[tuple[str, str]], tx: Transaction | None = None) -> tuple[list[str], list[str]]:
    """Rewrite whole-word references to renamed placeholders in src/ and include/.

    Only placeholder -> plain C identifier renames are rewritten (SDK/library names that an
    `extern "C"` declaration can carry). A C++-mangled new name needs a real declaration, so
    such references are returned for hand updating instead. Returns (files changed, manual refs).
    """
    rewrite = {old: new for old, new in pairs if _PLACEHOLDER.match(old) and not _MANGLED.search(new)}
    manual_names = {old for old, new in pairs if _PLACEHOLDER.match(old) and old not in rewrite}
    if not rewrite and not manual_names:
        return [], []
    word = re.compile(r"\b(" + "|".join(map(re.escape, sorted(set(rewrite) | manual_names))) + r")\b")
    changed, manual = [], []
    for root in (cfg.root / "src", cfg.root / "include"):
        for f in sorted(root.rglob("*")) if root.is_dir() else []:
            if f.suffix not in SOURCE_SUFFIXES or not f.is_file():
                continue
            text = f.read_text(errors="replace")
            if not word.search(text):
                continue
            for m in word.finditer(text):
                if m[1] in manual_names:
                    manual.append(f"{cfg.rel(f)}: {m[1]}")
            new_text = word.sub(lambda m: rewrite.get(m[1], m[1]), text)
            if new_text != text:
                (tx.write(f, new_text) if tx else f.write_text(new_text))
                changed.append(cfg.rel(f))
    return sorted(changed), sorted(set(manual))


def run(cfg: Config, args: argparse.Namespace) -> int:
    if len(args.names) % 2:
        print("error: give OLD NEW pairs")
        return USAGE
    pairs = list(zip(args.names[::2], args.names[1::2]))
    text = (cfg.root / cfg.platform_options["symbols"]).read_text()
    for i, (old, new) in enumerate(pairs):                 # addresses name one symbol each
        if re.fullmatch(r"0x[0-9A-Fa-f]+", old):
            m = re.search(rf"^(\S+) = [^:]+:0x0*{int(old, 16):X};", text, re.M | re.I)
            if m is None:
                print(f"error: no symbol at {old}")
                return USAGE
    tx = Transaction()
    errors, changed, manual = stage_renames(cfg, tx, pairs, args.local, not args.no_sources)
    if errors:
        for e in errors:
            print(f"error: {e}")
        return USAGE
    proc = fresh.build(cfg, timeout=120)
    if proc.returncode != 0:
        tx.rollback()
        fresh.build(cfg)
        print("REJECTED: build no longer matches; symbols.txt restored")
        print(tail(proc.stdout + proc.stderr, 15))
        return MISMATCH
    print(f"renamed {len(pairs)} symbol(s); build still matches")
    if changed:
        print(f"updated references in {len(changed)} source file(s): {', '.join(changed)}")
    for ref in manual:
        print(f"note: update by hand (C++ name needs a real declaration): {ref}")
    hint("status")
    return OK


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("names", nargs="+", help="OLD NEW pairs; OLD may be an address (0x80391540) for a repeated name")
    p.add_argument("--no-sources", action="store_true", help="do not rewrite placeholder references in src/ include/")
    p.add_argument("--local", action="store_true",
                   help="new names are scope:local and may repeat other TUs' local names (same-named statics)")


COMMAND = Command("rename", "analysis", "rename symbols (OLD NEW pairs), re-split, prove the hash", run, _configure, locked=True)
