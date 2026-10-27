"""dk unit add|list: declare translation units (splits.txt + units.toml + source stub)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from . import freshness as fresh
from ...core.command import MISMATCH, OK, USAGE, Command, hint
from ...core.config import Config
from ...core.progress import load_report
from . import asmindex
from .dol import Dol
from .dtk_symbols import boundary_fix, build_errors, caused_by, read_symbols, read_units, set_symbol_size
from .elf import Elf
from .matching import MatchError, resolve_unit
from .propose import DATA_SECTIONS, adjust_tu_bounds, ctor_slots, lone_initializer, propose, range_end, self_contained
from .units import (add_object, add_split_unit, read_units_file, remove_object, remove_split_unit, set_flags,
                    render_units_file, replace_split_unit)

_SPLIT_ARG = re.compile(r"^(?P<section>\.?[\w$.]+):0x(?P<start>[0-9A-Fa-f]+)-0x(?P<end>[0-9A-Fa-f]+)$")


def paths(cfg: Config):
    opts = cfg.platform_options
    return cfg.root / opts["splits"], cfg.root / opts["units"], cfg.root / opts.get("src", "src")


def _section_order(splits_text: str) -> list[str]:
    order = []
    in_sections = False
    for line in splits_text.splitlines():
        if line.startswith("Sections:"):
            in_sections = True
            continue
        if in_sections:
            if not line.strip():
                break
            order.append(line.split()[0])
    return order


class Transaction:
    """Snapshot files, write new content, and undo everything on rollback()."""

    def __init__(self) -> None:
        self._saved: dict[Path, str | None] = {}

    def write(self, path: Path, text: str) -> None:
        if path not in self._saved:
            self._saved[path] = path.read_text() if path.exists() else None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def rollback(self) -> None:
        for path, text in self._saved.items():
            if text is None:
                path.unlink(missing_ok=True)
            else:
                path.write_text(text)


def parse_split(raw: str) -> tuple[str, int, int]:
    m = _SPLIT_ARG.match(raw)
    if not m:
        raise ValueError(f"split {raw!r} is not SECTION:0xSTART-0xEND")
    start, end = int(m["start"], 16), int(m["end"], 16)
    if end <= start:
        raise ValueError(f"split {raw!r} ends before it starts")
    return m["section"], start, end


def declare(cfg: Config, tx: Transaction, path: str, splits: list[tuple[str, int, int]], lib: str, status: str,
            lib_settings: dict, options: dict | None = None, sources: dict[Path, str] | None = None) -> str | None:
    """Write the unit (splits, units.toml, sources) into tx and prove the re-split build matches.

    Returns None on success, or an error message (the caller decides whether to roll back).
    """
    splits_path, units_path, src_dir = paths(cfg)
    order = _section_order(splits_path.read_text())
    splits = sorted(splits, key=lambda s: (order.index(s[0]) if s[0] in order else len(order), s[1]))
    try:
        new_splits = add_split_unit(splits_path, path, splits)
    except ValueError as e:
        return str(e)
    units = read_units_file(units_path)
    if not add_object(units, lib, path, status, lib_settings, options):
        return f"{path} already in {cfg.rel(units_path)}"
    tx.write(splits_path, new_splits)
    tx.write(units_path, render_units_file(units))
    for dest, text in (sources or {}).items():
        if not dest.exists():
            tx.write(dest, text)
    src = src_dir / path
    if not src.exists():
        tx.write(src, "")
    return rebuild(cfg, tx)


def rebuild(cfg: Config, tx: Transaction) -> str | None:
    """Re-split and relink, trimming dtk's guessed label sizes at split boundaries, until the
    hash matches. Returns None on success or an error message (caller decides on rollback)."""
    # A normal re-split + link takes ~2 s; 60 s means the linker hung on this split.
    symbols_path = cfg.root / cfg.platform_options["symbols"]
    for _ in range(20):
        proc = fresh.build(cfg, fresh.hash_target(cfg), timeout=60)
        if proc.returncode == 124:
            return "linker hung with this split (killed after 60 s)"
        if proc.returncode == 0:
            leaks = section_overruns(cfg)
            if leaks:
                # the hash matches, yet objdiff (report, check, save) rejects these target objects
                return ("the build matches, but objdiff would reject a target object whose symbol runs past its "
                        "section (fix that unit's range too: ./dk unit propose / unit set):\n  " + "\n  ".join(leaks[:8]))
            return None
        output = proc.stdout + proc.stderr
        fix = boundary_fix(output)
        resized = set_symbol_size(symbols_path.read_text(), *fix) if fix else None
        if resized is None:
            return f"{caused_by(output)}\n" + build_errors(output)
        tx.write(symbols_path, resized)
    return "gave up after 20 symbol boundary fixes"


def section_overruns(cfg: Config) -> list[str]:
    """Symbols of the split target objects that run past the end of their section, which objdiff
    refuses (`size out of section bounds`): typically a neighbour's gap_ padding label left
    reaching over a boundary that moved."""
    import json
    path = cfg.root / "objdiff.json"
    if not path.is_file():
        return []
    out = []
    for u in json.loads(path.read_text()).get("units", []):
        target = cfg.root / u.get("target_path", "")
        if not u.get("target_path") or not target.is_file():
            continue
        try:
            elf = Elf.parse(target.read_bytes())
        except ValueError:
            continue
        for x in elf.symbols:
            if x.name and x.size and 0 < x.shndx < len(elf.sections):
                sec = elf.sections[x.shndx]
                if sec.type != 8 and x.value + x.size > sec.size:      # objdiff tolerates it in .bss
                    out.append(f"{u['name']}: {x.name} ends at {sec.name}+0x{x.value + x.size:X}, past its "
                               f"0x{sec.size:X} bytes")
    return out


def run_add(cfg: Config, args: argparse.Namespace) -> int:
    try:
        splits = [parse_split(raw) for raw in args.split]
    except ValueError as e:
        print(f"error: {e}")
        return USAGE
    tx = Transaction()
    settings = {"mw_version": args.mw_version, "cflags": args.cflags, "progress_category": args.category}
    # Explicit flags always apply to this object, even when the lib already exists with others.
    options = {"mw_version": args.mw_version, "cflags": args.cflags}
    error = declare(cfg, tx, args.path, splits, args.lib, args.status, settings, options)
    if error:
        print(f"REJECTED: {error}")
        tx.rollback()
        fresh.build(cfg, fresh.hash_target(cfg))     # relink only: unlinked sources stay out of it
        print("(files restored)")
        return MISMATCH
    print(f"added {args.path} ({len(splits)} split(s)); re-split build still matches")
    hint("ctx <function>", f"check {args.path}")
    return OK


def _finish(cfg: Config, tx: Transaction, error: str | None, done: str) -> int:
    if error:
        print(f"REJECTED: {error}")
        tx.rollback()
        fresh.build(cfg, fresh.hash_target(cfg))     # relink only: unlinked sources stay out of it
        print("(files restored)")
        return MISMATCH
    print(done)
    return OK


def run_set(cfg: Config, args: argparse.Namespace) -> int:
    try:
        splits = [parse_split(raw) for raw in args.split]
        splits_path, _, _ = paths(cfg)
        text = replace_split_unit(splits_path, args.path, splits, tuple(args.drop))
    except ValueError as e:
        print(f"error: {e}")
        return USAGE
    tx = Transaction()
    tx.write(splits_path, text)
    done = f"{args.path}: splits replaced ({len(splits)})"
    if args.rename:
        # One transaction: widening a linked unit over functions it defines under their real
        # names needs the renames and the new ranges together (either alone breaks the link).
        from .renamecmd import stage_renames
        errors, changed, manual = stage_renames(cfg, tx, [tuple(p) for p in args.rename], args.local)
        if errors:
            tx.rollback()
            for e in errors:
                print(f"error: {e}")
            return USAGE
        done += f", {len(args.rename)} rename(s)" + (f", references updated in {', '.join(changed)}" if changed else "")
        for ref in manual:
            print(f"note: update by hand (C++ name needs a real declaration): {ref}")
    code = _finish(cfg, tx, rebuild(cfg, tx), f"{done}; build still matches")
    if code == OK:
        hint(f"check {args.path}")
    return code


def run_remove(cfg: Config, args: argparse.Namespace) -> int:
    splits_path, units_path, _ = paths(cfg)
    units = read_units_file(units_path)
    try:
        text = remove_split_unit(splits_path, args.path)
    except ValueError as e:
        print(f"error: {e}")
        return USAGE
    if not remove_object(units, args.path):
        print(f"error: {args.path} not in {cfg.rel(units_path)}")
        return USAGE
    tx = Transaction()
    tx.write(splits_path, text)
    tx.write(units_path, render_units_file(units))
    return _finish(cfg, tx, rebuild(cfg, tx), f"{args.path}: undeclared (source kept); build still matches")


def run_fit(cfg: Config, args: argparse.Namespace) -> int:
    """Trim data ranges to the compiled object's real section sizes when the target bytes
    beyond them are zero padding (dtk's label sizes include inter-unit padding)."""
    try:
        unit = resolve_unit(cfg, args.path)
    except MatchError as e:
        print(f"error: {e}")
        return USAGE
    if fresh.build(cfg, [unit["base_path"]]).returncode != 0:
        print("error: unit does not compile")
        return MISMATCH
    base = Elf.parse((cfg.root / unit["base_path"]).read_bytes())
    dol = Dol.parse(cfg.original.read_bytes())
    splits_path, _, _ = paths(cfg)
    split_unit = next((u for u in read_units(splits_path) if u.name == unit["metadata"]["source_path"].removeprefix("src/")), None)
    if split_unit is None:
        print("error: unit not found in splits.txt")
        return USAGE
    symbols = read_symbols(cfg.root / cfg.platform_options["symbols"])
    new, changes = [], []
    for sec, start, end in split_unit.splits:
        section = base.section(sec)
        size = (section.size + 3) & ~3 if section else 0
        if sec != ".text" and section and 0 < size < end - start:
            tail = b"" if sec.startswith((".bss", ".sbss")) else dol.read(start + size, end - start - size)
            # dtk fills from the last object's end to the next symbol with a gap_ symbol: never
            # end a range between an unaligned object end and the next symbol
            cut = start + size
            last = max((min(x.address + x.size, cut) for x in symbols
                        if x.section == sec and start <= x.address < cut and x.type != "label"), default=cut)
            following = min((x.address for x in symbols if x.section == sec and x.address >= cut), default=end)
            cut = max(cut, range_end(last, min(following, end), [], sec))
            if not tail.strip(b"\0") and cut < end:
                changes.append(f"{sec} 0x{end:08X} -> 0x{cut:08X}")
                end = cut
        new.append((sec, start, end))
    if not changes:
        print("nothing to trim (data ranges already match the compiled sizes, or the tail is not padding)")
        return OK
    tx = Transaction()
    tx.write(splits_path, replace_split_unit(splits_path, split_unit.name, new))
    code = _finish(cfg, tx, rebuild(cfg, tx), "trimmed: " + ", ".join(changes) + "; build still matches")
    if code == OK:
        hint(f"check {args.path}")
    return code


def _neighbour_tu(ordered: list, entries: dict) -> list:
    """Functions of the TU-sized neighbour, walking away from a unit: dtk gives each __sinit its
    own object (auto_fn_*), so a neighbouring TU is up to one regular object plus sinit objects."""
    out, regular = [], None
    for s in ordered:
        f = entries[s.name].file
        is_sinit = f.stem.startswith("auto_fn_")
        if not is_sinit:
            if regular is None:
                regular = f
            elif f != regular:
                break
        out.append(s)
    return out


def run_propose(cfg: Config, args: argparse.Namespace) -> int:
    symbols = read_symbols(cfg.root / cfg.platform_options["symbols"])
    by_name = {s.name: s for s in symbols}
    if args.end is None and args.start not in by_name:   # one address: the function holding it
        try:
            address = int(args.start, 16)
        except ValueError:
            print(f"error: {args.start} is not a symbol or a hex address")
            return USAGE
        holder = next((s for s in symbols if s.type == "function" and s.section == ".text"
                       and s.address <= address < s.address + max(s.size, 1)), None)
        if holder is None:
            print(f"error: no function at 0x{address:08X}")
            return USAGE
        args.start = holder.name
    if args.start in by_name:   # a function name: use the dtk unit (or declared unit) holding it
        sym = by_name[args.start]
        entries = asmindex.index(asmindex.current_asm_files(cfg))
        holder = entries.get(sym.name)
        if holder is None:
            print(f"error: {args.start} not found in current asm")
            return USAGE
        text_fns = sorted((s for s in symbols if s.type == "function" and s.section == ".text"
                           and entries.get(s.name)), key=lambda s: s.address)
        fns = [s for s in text_fns if entries[s.name].file == holder.file]
        before = [s for s in text_fns if s.address < fns[0].address]
        after = [s for s in text_fns if s.address > fns[-1].address]
        previous = list(reversed(_neighbour_tu(list(reversed(before)), entries)))
        following = _neighbour_tu(after, entries)
        declared = not holder.file.stem.startswith("auto_")
        if declared and following and self_contained(following[0].name, entries) \
                and entries[following[0].name].file.stem.startswith("auto_fn_"):
            print(f"note: {following[0].name} right after this declared unit initialises only its own data "
                  f"(a header static?): it likely belongs here; widen with ./dk unit set")
        if not args.raw and not declared:   # a declared unit's range was chosen and linked: keep it
            fns, moved_out, moved_in = adjust_tu_bounds(fns, previous, following, entries)
            if moved_out:
                print(f"note: {len(moved_out)} leading function(s) belong to the previous TU "
                      f"(thunks/factories after its __sinit): {', '.join(moved_out)}")
            if moved_in:
                print(f"note: {len(moved_in)} function(s) after this unit belong to it: {', '.join(moved_in)}")
        if not args.raw and lone_initializer(fns, previous, entries):
            print(f"note: {', '.join(s.name for s in fns)} is only a static initializer: a unit of just that is "
                  f"dropped by the linker (accept fails, .ctors shifts). It is the tail of the TU before it "
                  f"(0x{previous[0].address:08X}-), which has no initializer: propose the two as one unit")
            merged = previous + fns
            owners = [u for u in read_units(paths(cfg)[0])
                      if any(sec == ".text" and a <= s.address < z for sec, a, z in u.splits for s in (merged[0], fns[0]))]
            args.merge = owners
            fns = merged
        start, end = fns[0].address, fns[-1].address + fns[-1].size
    else:
        try:
            start, end = int(args.start, 16), int(args.end, 16)
        except (TypeError, ValueError):
            print("error: give START END as hex addresses, or a function name")
            return USAGE
        entries = asmindex.index(asmindex.current_asm_files(cfg))
    dol = Dol.parse(cfg.original.read_bytes())
    splits_path, _, _ = paths(cfg)
    # declared neighbours bound every data range (no overlap, same link order); a unit whose
    # .text is this range is ignored, so re-proposing a declared unit shows what would change
    p = propose(symbols, entries, start, end, ctor_slots(dol, symbols), units=read_units(splits_path), read=dol.read)
    print(f"text 0x{start:08X}-0x{end:08X}: {len(p.functions)} function(s)")
    for sec, (lo, hi) in sorted(p.sections.items(), key=lambda kv: kv[1]):
        names = ", ".join(p.members[sec][:8]) or "(static-initializer slot)"
        print(f"  {sec:<8} 0x{lo:08X}-0x{hi:08X}  {names}{' …' if len(p.members[sec]) > 8 else ''}")
    if p.extern_data:
        print(f"extern data ({len(p.extern_data)}): {', '.join(p.extern_data[:12])}")
    if p.extern_calls:
        print(f"extern calls ({len(p.extern_calls)}): {', '.join(p.extern_calls[:12])}")
    for note in p.notes:
        print(f"note: {note}")
    splits = " ".join(f"{sec}:0x{lo:08X}-0x{hi:08X}" for sec, lo, hi in p.splits())
    merge = getattr(args, "merge", [])
    first = next((u for u in merge if any(sec == ".text" and a == start for sec, a, _ in u.splits)), None)
    if first:     # the initializer joins a declared unit: widen it, retire the initializer's own unit
        others = [u for u in merge if u is not first]
        print(f"\n./dk unit set {first.name} {splits}")
        for u in others:
            print(f"./dk unit remove {u.name}")
        if others:
            print(f"# move the header static from src/{others[0].name} into src/{first.name}, then:")
        print(f"./dk save {first.name} --note \"absorbed its static initializer\"")
        for u in others:     # only once HEAD no longer declares it (rm refuses before)
            print(f"./dk rm src/{u.name} --reason \"merged into {first.name}\"")
        return OK
    current = next((u for u in read_units(splits_path) if (".text", start, end) in u.splits), None)
    if current:   # already declared: show what would change, as one unit set
        have = {sec: (a, z) for sec, a, z in current.splits}
        want = dict(p.sections) | {".text": (start, end)}
        changes = [f"  {sec:<8} {'-' if sec not in have else f'0x{have[sec][0]:08X}-0x{have[sec][1]:08X}'} -> "
                   f"0x{want[sec][0]:08X}-0x{want[sec][1]:08X}" for sec in want if have.get(sec) != want[sec]]
        kept = [sec for sec in have if sec not in want]
        if not changes:
            print(f"\n{current.name} already has these ranges")
            return OK
        print(f"\n{current.name} is declared; proposed changes:")
        print("\n".join(changes))
        for sec in kept:
            print(f"  {sec:<8} kept as declared (the proposal has no range there)")
            want[sec] = have[sec]
        order = lambda sec: (sec != ".text", DATA_SECTIONS.index(sec) if sec in DATA_SECTIONS else 99)
        print(f"\n./dk unit set {current.name} " + " ".join(f"{sec}:0x{a:08X}-0x{z:08X}" for sec, (a, z) in
                                                         sorted(want.items(), key=lambda kv: order(kv[0]))))
        return OK
    print(f"\n./dk unit add {args.path or '<src path>'} --lib {args.lib} --cflags {args.cflags} "
          f"--mw-version {args.mw_version} --category {args.category} {splits}")
    return OK


def run_list(cfg: Config, args: argparse.Namespace) -> int:
    _, units_path, _ = paths(cfg)
    units = read_units_file(units_path)
    report = load_report(cfg) or {}
    measures = {u["name"].split("/", 1)[1] if "/" in u["name"] else u["name"]: u.get("measures", {})
                for u in report.get("units", [])}
    if not units.libs:
        print("no units declared yet")
        hint("unit add <path> --lib <lib> .text:0xSTART-0xEND")
        return OK
    for lib in units.libs:
        print(f"{lib.name}  ({', '.join(f'{k}={v}' for k, v in lib.settings.items())})")
        for obj in lib.objects:
            stem = obj["path"].rsplit(".", 1)[0]
            m = measures.get(stem, {})
            pct = m.get("fuzzy_match_percent")
            print(f"  {obj['status']:<12} {obj['path']:<60} {'' if pct is None else f'{pct:6.2f}% fuzzy'}")
    return OK


def _configure(p: argparse.ArgumentParser) -> None:
    sub = p.add_subparsers(dest="action", required=True)
    a = sub.add_parser("add", help="declare a unit: splits.txt block + units.toml entry + empty source")
    a.add_argument("path", help="source path relative to src/, e.g. Runtime.PPCEABI.H/global_destructor_chain.c")
    a.add_argument("split", nargs="+", help="SECTION:0xSTART-0xEND, e.g. .text:0x801A7344-0x801A73A4")
    a.add_argument("--lib", required=True, help="library name (grouping + shared flags)")
    a.add_argument("--status", default="NonMatching", help="initial status (default NonMatching)")
    a.add_argument("--cflags", help="CFLAGS key in configure.py for this object (and the lib, if new)")
    a.add_argument("--mw-version", help="compiler, e.g. GC/1.3.2, for this object (and the lib, if new)")
    a.add_argument("--category", help="progress category (game, sdk), for a new lib")
    sub.add_parser("list", help="units with status and fuzzy match from the last report")
    st = sub.add_parser("set", help="replace a declared unit's splits (all of them) and prove the hash")
    st.add_argument("path", help="unit source path relative to src/")
    st.add_argument("split", nargs="+", help="SECTION:0xSTART-0xEND (the complete new list)")
    st.add_argument("--rename", nargs=2, action="append", metavar=("OLD", "NEW"), default=[],
                    help="rename a symbol in the same transaction (repeatable)")
    st.add_argument("--local", action="store_true", help="--rename names are scope:local (see dk rename --local)")
    st.add_argument("--drop", action="append", default=[], metavar="SECTION",
                    help="remove this section from the unit (otherwise every current section must be listed)")
    fl = sub.add_parser("flags", help="set a unit's compiler version / flag set / extra flags and prove the build")
    fl.add_argument("path", help="unit source path relative to src/")
    fl.add_argument("--mw-version", help="e.g. GC/1.3 (the compiler is per TU)")
    fl.add_argument("--cflags", help="flag set name (a key of CFLAGS), e.g. game")
    fl.add_argument("--extra-cflags", help='extra flags, e.g. "-char signed" or "-opt nostrength"')
    fl.add_argument("--clear-extra", action="store_true", help="remove the unit's extra flags")
    rm = sub.add_parser("remove", help="undeclare a unit (splits + units.toml); the source file is kept")
    rm.add_argument("path", help="unit source path relative to src/")
    ft = sub.add_parser("fit", help="trim data ranges to the compiled sizes where the target tail is padding")
    ft.add_argument("path", help="unit source path relative to src/")
    pr = sub.add_parser("propose", help="propose a unit's full split (text + data) from a text range or function")
    pr.add_argument("start", help="a function name or address (uses the unit holding it), or START with END")
    pr.add_argument("end", nargs="?", help="text end (hex), when start is an address")
    pr.add_argument("--path", help="source path to print in the suggested command")
    pr.add_argument("--raw", action="store_true", help="use dtk's auto unit as is (no TU-boundary adjustment)")
    pr.add_argument("--lib", default="game")
    pr.add_argument("--cflags", default="game")
    pr.add_argument("--mw-version", default="GC/1.3.2")
    pr.add_argument("--category", default="game")


def run_flags(cfg: Config, args: argparse.Namespace) -> int:
    """Per-unit compiler and flags (the compiler is per TU: some need GC/1.3 with -char signed)."""
    _, units_path, _ = paths(cfg)
    units = read_units_file(units_path)
    extra = None
    if args.clear_extra:
        extra = []
    elif args.extra_cflags is not None:
        extra = args.extra_cflags.split()
    try:
        changes = set_flags(units, args.path, args.mw_version, args.cflags, extra)
    except KeyError:
        print(f"error: {args.path} not in {cfg.rel(units_path)}")
        return USAGE
    if not changes:
        print("nothing to change")
        return OK
    tx = Transaction()
    tx.write(units_path, render_units_file(units))         # a reconfigure dependency: ninja regenerates
    code = _finish(cfg, tx, rebuild(cfg, tx), f"{args.path}: " + "; ".join(changes) + "; build still matches")
    if code == OK:
        hint(f"check {args.path}")
    return code


def run(cfg: Config, args: argparse.Namespace) -> int:
    if getattr(args, "path", None):
        args.path = args.path.removeprefix("src/")     # paths are relative to src/
    return {"add": run_add, "list": run_list, "propose": run_propose, "set": run_set,
            "remove": run_remove, "fit": run_fit, "flags": run_flags}[args.action](cfg, args)


COMMAND = Command("unit", "matching", "declare and edit translation units (propose, add, set, flags, fit, remove, list)", run, _configure, locked=True)
