"""dk rename OLD NEW [OLD NEW ...]: rename symbols in symbols.txt and prove the build still matches."""

from __future__ import annotations

import argparse
import re

from . import dtkconfig
from . import freshness as fresh
from ...core.command import MISMATCH, OK, USAGE, Command, hint
from ...core.config import Config
from .dtk_symbols import build_errors
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
                  sources: bool = True, check_c: bool = True) -> tuple[list[str], list[str], list[str]]:
    """Write renames into a transaction (symbols.txt + placeholder references in sources).
    Returns (errors, source files changed, references to update by hand)."""
    if check_c:
        callers = c_name_callers(cfg, pairs)
        if callers:
            return [refusal(callers)], [], []
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
            new_text = _drop_new_duplicates(text, word.sub(lambda m: rewrite.get(m[1], m[1]), text))
            if new_text != text:
                (tx.write(f, new_text) if tx else f.write_text(new_text))
                changed.append(cfg.rel(f))
    return sorted(changed), sorted(set(manual))


_DECL = re.compile(r"^extern [^{\n]*;\s*$")


def _drop_new_duplicates(old: str, new: str) -> str:
    """`new` without the one-line extern declarations the rewrite turned into a repeat of an
    earlier one (a placeholder declared beside the real name it was renamed to)."""
    before, after = old.split("\n"), new.split("\n")
    if len(before) != len(after):
        return new
    seen_old, seen_new, out = set(), set(), []
    for o, n in zip(before, after):
        if _DECL.match(n) and n in seen_new and o not in seen_old:
            continue
        seen_old.add(o)
        seen_new.add(n)
        out.append(n)
    return "\n".join(out)


_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_EXTERN_C = re.compile(r'\bextern\s+"C"\s*')
# a branch mnemonic (`b`, `bl`, `beq+`, `bdnz`…) before its target; not a C type such as `bool`
_BRANCH = r"\bb(?:eq|ne|lt|le|gt|ge|so|ns|dnz|dz|c)?(?:l|a|la)?[+-]?\s+"


def _extern_c_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of `extern "C"` blocks and declarations (up to `;` or a body's `{`)."""
    spans = []
    for m in _EXTERN_C.finditer(text):
        i = m.end()
        if text.startswith("{", i):
            depth, j = 0, i
            while j < len(text):
                depth += {"{": 1, "}": -1}.get(text[j], 0)
                if depth == 0:
                    break
                j += 1
            spans.append((i, j))
        else:
            ends = [k for k in (text.find(";", i), text.find("{", i)) if k >= 0]
            spans.append((i, min(ends) if ends else len(text)))
    return spans


def c_uses(text: str, names: set[str], c_file: bool = False) -> list[tuple[str, int]]:
    """(name, line) for each use of a name as a C link name: in an `extern "C"` declaration or
    block, in assembly (`bl name`, `name@ha`), or anywhere in a C file. A C++ use (member call,
    qualified name, mangled declaration) links to the mangled name and is not listed."""
    if not names:
        return []
    text = _COMMENT.sub(lambda m: re.sub(r"[^\n]", " ", m[0]), text)
    word = re.compile(r"(?<![\w.:>])(" + "|".join(map(re.escape, sorted(names))) + r")\b(?!__)")
    spans = _extern_c_spans(text)
    out = []
    for m in word.finditer(text):
        line_start = text.rfind("\n", 0, m.start()) + 1
        line = text[line_start:text.find("\n", m.end()) if "\n" in text[m.end():] else len(text)]
        asm = re.search(_BRANCH + re.escape(m[1]) + r"\b", line) or re.search(re.escape(m[1]) + r"@(?:ha|h|l|sda21)\b", line)
        if c_file or asm or any(a <= m.start() < b for a, b in spans):
            out.append((m[1], text.count("\n", 0, m.start()) + 1))
    return out


def c_callers(cfg: Config, names: set[str]) -> dict[str, list[str]]:
    """name -> 'path:line' of each place in src/ and include/ that uses it as a C link name."""
    out: dict[str, list[str]] = {}
    if not names:
        return out
    quick = re.compile("|".join(map(re.escape, sorted(names))))
    for root in (cfg.root / "src", cfg.root / "include"):
        for f in sorted(root.rglob("*")) if root.is_dir() else []:
            if f.suffix not in SOURCE_SUFFIXES or not f.is_file():
                continue
            text = f.read_text(errors="replace")
            if not quick.search(text):
                continue
            for name, line in c_uses(text, names, f.suffix == ".c"):
                out.setdefault(name, []).append(f"{cfg.rel(f)}:{line}")
    return out


def c_name_callers(cfg: Config, pairs: list[tuple[str, str]]) -> dict[str, list[str]]:
    """Placeholders this rename gives a C++ (mangled) name that a compiled source still uses as
    a C link name: after the rename, those units fail to link (undefined: fn_ADDR)."""
    risky = {old for old, new in pairs if _PLACEHOLDER.match(old) and _MANGLED.search(new)}
    if not risky:
        return {}
    from .units import read_units_file
    src = cfg.root / cfg.platform_options.get("src", "src")
    units = read_units_file(cfg.root / cfg.platform_options["units"])
    compiled = {f"{cfg.rel(src)}/{o['path']}" for lib in units.libs for o in lib.objects}
    callers = c_callers(cfg, risky)
    # a source no unit compiles (a stale file) cannot break the link; headers may be included anywhere
    keep = lambda ref: not ref.startswith(f"{cfg.rel(src)}/") or ref.rsplit(":", 1)[0] in compiled \
        or ref.rsplit(":", 1)[0].endswith((".h", ".hpp", ".inc"))
    out = {name: [r for r in refs if keep(r)] for name, refs in callers.items()}
    return {name: refs for name, refs in out.items() if refs}


def refusal(callers: dict[str, list[str]]) -> str:
    lines = [f"{len(callers)} name(s) are still used as C link names (extern \"C\", asm or C code); renaming "
             "them to C++ names breaks those units' link (undefined: fn_ADDR):"]
    lines += [f"  {name}: {', '.join(refs[:6])}{' …' if len(refs) > 6 else ''}" for name, refs in sorted(callers.items())]
    lines.append("fix those (declare the member and call it as one), or pass --keep-c to rename the rest "
                 "and keep these names")
    return "\n".join(lines)


def placeholder_pairs(compiled: list[str], known: set[str]) -> list[tuple[str, str]]:
    """(fn_ADDR, its mangled name) for each compiled function named after a placeholder that
    symbols.txt still holds under the bare placeholder: what a draft's member needs."""
    out = []
    for name in sorted(set(compiled)):
        m = re.match(r"(fn_[0-9A-F]{8})__\w", name)
        if m and m[1] in known and name not in known:
            out.append((m[1], name))
    return out


def address_pairs(base, target, known: set[str], taken: set[str] = frozenset()) -> list[tuple[str, str]]:
    """(fn_ADDR, compiled name) for functions the unit defines under a name symbols.txt lacks
    (virtuals, destructors, thunks) where the original object (dtk's split) has a bare fn_ADDR
    at the same place: the same slot of the same-named vtable, or the same .text offset with
    the same size. Weak copies (inline functions other units define) are never renamed; this-adjusting
    thunks (weak `@N@...`) are, by their vtable slot."""
    from collections import defaultdict
    from .elf import STT_FUNC
    defined = lambda e, x: 0 < x.shndx < len(e.sections)
    ours = {x.name: x for x in base.symbols if x.type == STT_FUNC and defined(base, x) and x.name
            and (x.bind != 2 or x.name.startswith("@")) and x.name not in known}   # thunks are weak
    theirs = {x.name: x for x in target.symbols if x.type == STT_FUNC and defined(target, x)}
    at = {(target.sections[x.shndx].name, x.value): x for x in theirs.values()}
    found: dict[str, set[str]] = defaultdict(set)
    for name, x in ours.items():
        t = at.get((base.sections[x.shndx].name, x.value))
        if t and t.size == x.size and _PLACEHOLDER.match(t.name):
            found[name].add(t.name)
    objects = lambda e: {x.name: x for x in e.symbols if x.type == 1 and defined(e, x) and x.name}
    target_objects = objects(target)
    for vt, x in objects(base).items():
        t = target_objects.get(vt)
        if t is None or not vt.startswith("__vt__"):
            continue
        cls = vt[len("__vt__"):]
        slots = lambda e, sym: {r.offset - sym.value: r.symbol for r in e.relocations(sym.shndx)
                                if sym.value <= r.offset < sym.value + sym.size}
        theirs_at = slots(target, t)
        for off, name in slots(base, x).items():
            fn = theirs_at.get(off)
            if name not in ours or not fn or not _PLACEHOLDER.match(fn) or fn not in theirs:
                continue
            # the slot's override is this class's (or the same code): not an inherited function
            if f"__{cls}F" in name or f"__{cls}CF" in name or theirs[fn].size == ours[name].size:
                found[name].add(fn)
    olds = defaultdict(set)
    for name, fns in found.items():
        for fn in fns:
            olds[fn].add(name)
    return sorted((next(iter(fns)), name) for name, fns in found.items()
                  if len(fns) == 1 and len(olds[next(iter(fns))]) == 1 and next(iter(fns)) in known
                  and next(iter(fns)) not in taken)


def _unit_pairs(cfg: Config, unit_arg: str) -> list[tuple[str, str]] | None:
    import os
    import shutil
    from .elf import Elf
    from .matching import MatchError, compile_command, resolve_unit, retarget, run_compile
    try:
        unit = resolve_unit(cfg, unit_arg)
        command = compile_command(cfg, unit["base_path"], locked=False)
    except MatchError as e:
        print(f"error: {e}")
        return None
    work = cfg.root / "build" / "dk" / "rename" / str(os.getpid())
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        ok, out = run_compile(cfg, retarget(command, work))
        objs = list(work.glob("*.o"))
        if not ok or not objs:
            print("error: the unit does not compile:\n" + out[-2000:])
            return None
        base = Elf.parse(objs[0].read_bytes())
    finally:
        shutil.rmtree(work, ignore_errors=True)
    text = (cfg.root / cfg.platform_options["symbols"]).read_text()
    known = set(re.findall(r"^(\S+) = ", text, re.M))
    pairs = placeholder_pairs([sym.name for sym in base.symbols if sym.type == 2], known)
    target = cfg.root / unit["target_path"]
    if target.is_file():      # virtuals, destructors, thunks: matched by vtable slot or offset
        pairs += address_pairs(base, Elf.parse(target.read_bytes()), known, {old for old, _ in pairs})
    return sorted(pairs)


def run(cfg: Config, args: argparse.Namespace) -> int:
    if args.unit:
        pairs = _unit_pairs(cfg, args.unit)
        if pairs is None:
            return USAGE
        if not pairs:
            print(f"nothing to rename: no function of {args.unit} is still under its fn_ placeholder")
            return OK
        args.names = [n for pair in pairs for n in pair] + list(args.names)
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
    _, invalid = rename_text((cfg.root / cfg.platform_options["symbols"]).read_text(), pairs, args.local)
    if invalid:
        for e in invalid:
            print(f"error: {e}")
        return USAGE
    callers = c_name_callers(cfg, pairs)
    if callers and not args.keep_c:
        if args.dry_run:
            for old, new in pairs:
                print(f"{old} -> {new}")
        print(f"REJECTED: {refusal(callers)}")
        return MISMATCH
    for name, refs in sorted(callers.items()):
        print(f"kept {name} (a C link name in {', '.join(refs[:3])}{' …' if len(refs) > 3 else ''})")
    pairs = [(old, new) for old, new in pairs if old not in callers]
    if args.dry_run or not pairs:
        for old, new in pairs:
            print(f"{old} -> {new}")
        print(f"{len(pairs)} rename(s){' (dry run: nothing changed)' if args.dry_run else ''}")
        return OK
    tx = Transaction()
    errors, changed, manual = stage_renames(cfg, tx, pairs, args.local, not args.no_sources, check_c=False)
    if errors:
        for e in errors:
            print(f"error: {e}")
        return USAGE
    cfg_path = cfg.root / dtkconfig.path(cfg)
    kept = dtkconfig.force_active_names(cfg_path.read_text()) if cfg_path.is_file() else []
    mapping = {old: new for old, new in pairs}
    if any(n in mapping for n in kept):
        tx.write(cfg_path, dtkconfig.set_force_active(cfg_path.read_text(), [mapping.get(n, n) for n in kept]))
    # sources this rename rewrote are its own: a compile error there is not someone else's
    own = tuple(f"build/{cfg.game_id}/{c.rsplit('.', 1)[0]}.o" for c in changed)
    proc = fresh.build(cfg, fresh.hash_target(cfg), timeout=120, own=own)
    if proc.returncode != 0:
        # the reason first; the restore relinks only the binary, so another agent's unfinished
        # (unlinked) source can neither block it nor appear as the cause
        print(f"REJECTED: the linked build no longer matches with this rename:\n{build_errors(proc.stdout + proc.stderr)}")
        tx.rollback()
        fresh.build(cfg, fresh.hash_target(cfg))
        print("(symbols.txt and sources restored)")
        return MISMATCH
    print(f"renamed {len(pairs)} symbol(s); build still matches")
    if changed:
        print(f"updated references in {len(changed)} source file(s): {', '.join(changed)}")
    # (C link-name uses were refused above; the other references are C++ uses of the new name)
    hint("status")
    return OK


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("--unit", help="rename every fn_ADDR the unit's source now defines as a member "
                   "(fn_ADDR__Class...), and virtuals/destructors/thunks found by vtable slot or offset, "
                   "to the compiled mangled name")
    p.add_argument("names", nargs="*", help="OLD NEW pairs; OLD may be an address (0x80391540) for a repeated name")
    p.add_argument("--no-sources", action="store_true", help="do not rewrite placeholder references in src/ include/")
    p.add_argument("--local", action="store_true",
                   help="new names are scope:local and may repeat other TUs' local names (same-named statics)")
    p.add_argument("--keep-c", action="store_true",
                   help="skip (keep) placeholders other sources still use as C link names instead of refusing")
    p.add_argument("--dry-run", action="store_true", help="print the renames (and C callers); change nothing")


COMMAND = Command("rename", "analysis", "rename symbols (OLD NEW pairs), re-split, prove the hash", run, _configure, locked=True)
