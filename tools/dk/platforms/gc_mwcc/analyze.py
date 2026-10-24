"""dk analyze <kind>: derive names and structure from the original binary.

Every analysis is deterministic: same binary + same symbols.txt -> byte-identical output.
  --check  exit 1 if the generated doc would change (CI)
  --apply  write safe renames into symbols.txt (auto-named symbols only)
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from . import freshness as fresh
from ...core.command import MISMATCH, OK, USAGE, Command, hint
from ...core.config import Config
from ...core.util import run_ninja
from ...core.doclint import GENERATED_MARK
from .dol import Dol
from .dtk_symbols import overlap_victim, read_symbols, remove_symbol, rename_symbols
from .elf import Elf
from .libmatch import DolIndex, ObjectMatch, Variant, compile_all, locate, resolve_conflicts
from .port import consistent_names, relocation_names, sda_bases
from .references import ensure, load_references
from .renamecmd import rewrite_sources
from .rtti import ClassInfo, analyze as analyze_rtti, mangle


def _inputs(cfg: Config) -> tuple[Dol, Path, set[int]] | None:
    symbols = cfg.root / cfg.platform_options.get("symbols", "")
    if not cfg.original.is_file() or not symbols.is_file():
        return None
    syms = read_symbols(symbols)
    return Dol.parse(cfg.original.read_bytes()), symbols, {s.address for s in syms if s.type == "function"}


def _group(classes: dict[int, ClassInfo]) -> dict[str, list[ClassInfo]]:
    by_name: dict[str, list[ClassInfo]] = defaultdict(list)
    for info in classes.values():
        by_name[info.name].append(info)
    return dict(sorted(by_name.items()))


def rtti_renames(classes: dict[int, ClassInfo]) -> dict[int, str]:
    """Names that are unambiguous before splitting: primary vtables, single-copy RTTI records."""
    names: dict[int, str] = {}
    for name, copies in _group(classes).items():
        mangled = mangle(name)
        if mangled is None:
            continue
        primaries = [v for c in copies for v in c.vtables if v.offset == 0]
        if len(primaries) == 1:
            names[primaries[0].address] = f"__vt__{mangled}"
        if len(copies) == 1:
            names[copies[0].rtti] = f"__RTTI__{mangled}"
    return names


def render_classes(classes: dict[int, ClassInfo]) -> str:
    groups = _group(classes)
    name_of = {c.rtti: c.name for c in classes.values()}
    derived: dict[str, set[str]] = defaultdict(set)
    for info in classes.values():
        for base, _ in info.bases:
            derived[name_of[base]].add(info.name)

    def bases_of(copies: list[ClassInfo]) -> str:
        best = max(copies, key=lambda c: len(c.bases))
        return ", ".join(f"{name_of[b]}@0x{o:X}" for b, o in best.bases) or "-"

    out = [
        f"{GENERATED_MARK} analyze rtti; do not edit -->",
        "# Classes (from RTTI)",
        "",
        f"{len(groups)} classes, {len(classes)} RTTI records. `copies` = RTTI records with this name;",
        "CodeWarrior emits one local copy per translation unit that needs it, so copies > 1 marks",
        "classes used across several TUs. Bases list every base (direct and indirect) with its offset.",
        "",
        "| Class | vtable | slots | secondary vtables | copies | bases | derived |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, copies in groups.items():
        vts = [v for c in copies for v in c.vtables]
        primary = next((v for v in vts if v.offset == 0), None)
        secondary = [v for v in vts if v.offset != 0]
        out.append(
            f"| `{name}` | {f'0x{primary.address:08X}' if primary else '-'} | {len(primary.slots) if primary else '-'}"
            f" | {len(secondary) or '-'} | {len(copies)} | {bases_of(copies)} | {', '.join(sorted(derived.get(name, ()))) or '-'} |"
        )
    return "\n".join(out) + "\n"


def rtti_json(classes: dict[int, ClassInfo]) -> str:
    rows = [
        {
            "name": c.name,
            "rtti": c.rtti,
            "bases": [[b, o] for b, o in c.bases],
            "vtables": [{"address": v.address, "offset": v.offset, "slots": v.slots} for v in c.vtables],
        }
        for c in classes.values()
    ]
    return json.dumps(rows, indent=1, sort_keys=True) + "\n"


def resplit_after_rename(cfg: Config, symbols_path: Path, max_rounds: int = 50) -> bool:
    """Re-run dtk's split. Real names let dtk's signatures add known data symbols, which can
    overlap labels dtk invented earlier; drop those stale auto labels until the split succeeds."""
    config_json = cfg.root / "build" / cfg.game_id / "config.json"
    for _ in range(max_rounds):
        proc = fresh.build(cfg, [cfg.rel(config_json)])
        if proc.returncode == 0:
            return True
        victim = overlap_victim(proc.stdout + proc.stderr)
        if victim is None or not remove_symbol(symbols_path, *victim):
            print((proc.stdout + proc.stderr)[-2000:])
            return False
        print(f"dropped stale auto label {victim[0]} (overlapped a signature symbol)")
    return False


def run_rtti(cfg: Config, args: argparse.Namespace) -> int:
    inputs = _inputs(cfg)
    if inputs is None:
        print("error: needs the original binary and symbols.txt")
        hint("setup", "build")
        return USAGE
    dol, symbols_path, functions = inputs
    classes = analyze_rtti(dol, functions)

    doc = cfg.docs.generated / "classes.md"
    text = render_classes(classes)
    if args.check:
        if not doc.is_file() or doc.read_text() != text:
            print(f"{cfg.rel(doc)} is out of date")
            hint("analyze rtti")
            return MISMATCH
        print(f"{cfg.rel(doc)} up to date")
        return OK

    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(text)
    machine = cfg.root / "build" / "dk" / "rtti.json"
    machine.parent.mkdir(parents=True, exist_ok=True)
    machine.write_text(rtti_json(classes))
    groups = _group(classes)
    print(f"{len(groups)} classes, {len(classes)} RTTI records -> {cfg.rel(doc)}, {cfg.rel(machine)}")

    renames = rtti_renames(classes)
    if args.apply:
        new_text, applied = rename_symbols(symbols_path, renames)
        symbols_path.write_text(new_text)
        print(f"renamed {applied} symbol(s) in {cfg.rel(symbols_path)} ({len(renames)} candidates)")
        hint("build")
    else:
        print(f"{len(renames)} rename candidate(s); re-run with --apply to write them")
        hint("analyze rtti --apply")
    return OK


def find_library_objects(cfg: Config, dol: Dol, symbols_path: Path) -> list[ObjectMatch]:
    syms = read_symbols(symbols_path)
    index = DolIndex(dol, [(s.address, s.size) for s in syms if s.type == "function"])
    wibo = cfg.root / "build" / "tools" / "wibo"
    compilers = cfg.root / "build" / "compilers"
    best: dict[str, ObjectMatch] = {}
    for ref in load_references(cfg):
        root = ensure(cfg, ref)
        sources = sorted(f for d in ref.sources for pattern in ("*.c", "*.cpp", "*.cp")
                         for f in (root / d).rglob(pattern))
        for compiler in ref.compilers:
            for extra in ref.variants:
                label = "_".join([compiler.replace("/", "-"), *[x.lstrip("-") for x in extra]]) or compiler
                variant = Variant(label, compiler, ref.flags + extra)
                objects = compile_all(cfg.root, wibo, compilers, root, sources, variant, list(ref.includes),
                                      cfg.root / "build" / "dk" / "lib" / ref.name)
                for src, obj in sorted(objects.items()):
                    m = locate(index, f"{ref.name}:{src.relative_to(root / ref.src_root)}", label, obj)
                    m.object, m.compiler, m.flags = cfg.rel(obj), compiler, variant.flags
                    key = m.source
                    if key not in best or (len(m.matched), m.clean) > (len(best[key].matched), best[key].clean):
                        best[key] = m
    return resolve_conflicts(list(best.values()))


def render_libraries(matches: list[ObjectMatch], total_functions: int) -> str:
    found = sum(len(m.matched) for m in matches)
    clean = [m for m in matches if m.clean]
    out = [
        f"{GENERATED_MARK} analyze lib; do not edit -->",
        "# Library objects found by reference matching",
        "",
        f"{len(matches)} objects located, {len(clean)} clean; {found} of {total_functions} DOL functions identified.",
        "`clean` = every DOL function inside the span is explained by the reference (missing reference",
        "functions were dead-stripped). `gaps` = DOL functions in the span that matched nothing",
        "(revision differences to port by hand).",
        "",
        "| Span | Object | Variant | Matched / in source | Gaps |",
        "|---|---|---|---|---|",
    ]
    for m in matches:
        out.append(f"| 0x{m.start:08X}-0x{m.end:08X} | `{m.source}` | {m.variant} | {len(m.matched)} / {m.ref_functions}"
                   f" | {len(m.gap_functions) or 'clean'} |")
    return "\n".join(out) + "\n"


def run_lib(cfg: Config, args: argparse.Namespace) -> int:
    inputs = _inputs(cfg)
    if inputs is None:
        print("error: needs the original binary and symbols.txt")
        hint("setup", "build")
        return USAGE
    dol, symbols_path, functions = inputs
    if not load_references(cfg):
        print("error: no [[platform.references]] in dk.toml")
        return USAGE
    matches = find_library_objects(cfg, dol, symbols_path)
    doc = cfg.docs.generated / "libraries.md"
    text = render_libraries(matches, len(functions))
    if args.check:
        stale = not doc.is_file() or doc.read_text() != text
        print(f"{cfg.rel(doc)} {'is out of date' if stale else 'up to date'}")
        return MISMATCH if stale else OK
    doc.write_text(text)
    machine = cfg.root / "build" / "dk" / "libmatch.json"
    machine.write_text(json.dumps([
        {"source": m.source, "variant": m.variant, "compiler": m.compiler, "flags": list(m.flags),
         "object": m.object, "clean": m.clean, "matched": m.matched,
         "unmatched": m.unmatched, "gaps": m.gap_functions} for m in matches], indent=1) + "\n")
    found = sum(len(m.matched) for m in matches)
    print(f"{len(matches)} objects, {sum(m.clean for m in matches)} clean, {found} functions -> {cfg.rel(doc)}")
    names = {addr: name for m in matches for name, addr, _ in m.matched}
    # Callees and globals of matched functions, read off the DOL (see port.relocation_names).
    syms = read_symbols(symbols_path)
    init_registers = next((s.address for s in syms if s.name == "__init_registers"), None)
    sda = sda_bases(dol, init_registers) if init_registers else {}
    samples = []
    for m in matches:
        samples += relocation_names(dol, Elf.parse((cfg.root / m.object).read_bytes()), m.matched, sda)
    samples += [(a, n) for a, n in names.items()]
    names = consistent_names(samples)
    if args.apply:
        before = {s.address: s.name for s in read_symbols(symbols_path)}
        # A name our compiled units already define elsewhere means that unit's split is too small
        # (its copy was silently dead-stripped). Renaming would make the link multiply-defined.
        defined = set()
        for obj in sorted((cfg.root / "build" / cfg.game_id / "src").rglob("*.o")):
            defined |= {x.name for x in Elf.parse(obj.read_bytes()).symbols if x.shndx != 0 and x.bind in (1, 2) and x.name}
        clashes = sorted((n, a) for a, n in names.items() if n in defined and before.get(a) != n)
        for n, a in clashes:
            print(f"skip {n} at 0x{a:08X}: already defined by a compiled unit (that unit's split is probably too small)")
        names = {a: n for a, n in names.items() if (n, a) not in set(clashes)}
        new_text, applied = rename_symbols(symbols_path, names)
        symbols_path.write_text(new_text)
        after = {s.address: s.name for s in read_symbols(symbols_path)}
        pairs = [(before[a], after[a]) for a in sorted(after) if a in before and before[a] != after[a]]
        changed, manual = rewrite_sources(cfg, pairs)
        print(f"renamed {applied} symbol(s) in {cfg.rel(symbols_path)} ({len(names)} candidates); "
              f"updated {len(changed)} source file(s)")
        for ref in manual:
            print(f"note: update by hand: {ref}")
        if not resplit_after_rename(cfg, symbols_path):
            print("error: dtk split still fails after renames")
            return MISMATCH
        hint("build")
    else:
        print(f"{len(names)} rename candidate(s); re-run with --apply to write them")
    return OK


def _configure(p: argparse.ArgumentParser) -> None:
    sub = p.add_subparsers(dest="kind", required=True)
    r = sub.add_parser("rtti", help="classes, bases and vtables from CodeWarrior RTTI")
    r.add_argument("--apply", action="store_true", help="write safe renames into symbols.txt")
    r.add_argument("--check", action="store_true", help="exit 1 if the generated doc is stale")
    lib = sub.add_parser("lib", help="find reference-source objects (SDK, runtime, ...) in the binary")
    lib.add_argument("--apply", action="store_true", help="rename matched functions in symbols.txt")
    lib.add_argument("--check", action="store_true", help="exit 1 if the generated doc is stale")


def run(cfg: Config, args: argparse.Namespace) -> int:
    return {"rtti": run_rtti, "lib": run_lib}[args.kind](cfg, args)


COMMAND = Command("analyze", "analysis", "derive names/structure from the binary: rtti, lib", run, _configure, locked=True)
