"""dk accept: promote an exact unit to Matching (linked), prove the hash, commit."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import freshness as fresh
from ...core.command import MISMATCH, OK, USAGE, Command, hint
from ...core.config import Config
from ...core.progress import load_report
from .dtk_symbols import build_errors
from .matching import MatchError, commit_gate, render, resolve_unit, score
from .unitcommit import commit_unit
from .units import read_units_file, render_units_file, set_status


def _matched_functions(cfg: Config) -> int:
    report = load_report(cfg) or {}
    return int(report.get("measures", {}).get("matched_functions", 0) or 0)


def _build(cfg: Config, own: tuple[str, ...] = ()) -> tuple[bool, str]:
    proc = fresh.build(cfg, fresh.hash_target(cfg), own=own)
    return proc.returncode == 0, proc.stdout + proc.stderr


def dropped_by_link(cfg: Config, unit: dict) -> list[str]:
    """The unit's functions (as the original has them) that the linked ELF lacks."""
    from .elf import Elf
    from .matching import _function_offsets
    elf_path = cfg.built.with_suffix(".elf")
    if not elf_path.is_file():
        return []
    linked_elf = Elf.parse(elf_path.read_bytes())
    linked = {x.name for x in linked_elf.symbols if x.name and x.shndx}
    ours = {x.name: x.bind for x in Elf.parse((cfg.root / unit["base_path"]).read_bytes()).symbols if x.type == 2}
    original = {name: size for name, (_sec, _off, size) in _function_offsets(cfg.root / unit["target_path"]).items()}
    return dropped_lines(original, linked, ours) + data_shortfall(cfg, unit, linked_elf)


def data_shortfall(cfg: Config, unit: dict, linked_elf) -> list[str]:
    """Data sections of the unit that link shorter than the original's (the shift of the symbols
    after its range against those before it), with the objects of ours that only weak copies the
    link discards reference: the original kept those bytes (in place) where ours lost them."""
    from collections import Counter
    from .dtk_symbols import read_symbols, read_units
    from .matching import _weak_extras
    symbols = read_symbols(cfg.root / cfg.platform_options["symbols"])
    count = Counter(x.name for x in symbols)
    lcount = Counter(x.name for x in linked_elf.symbols if x.name)
    at = {x.name: x.value for x in linked_elf.symbols if x.name and x.shndx and lcount[x.name] == 1}
    source = unit["metadata"]["source_path"].removeprefix(cfg.platform_options.get("src", "src") + "/")
    split = next((u for u in read_units(cfg.root / cfg.platform_options["splits"]) if u.name == source), None)
    if split is None:
        return []
    extras, _ = _weak_extras(cfg.root / unit["target_path"], cfg.root / unit["base_path"])
    out = []
    for sec, start, end in split.splits:
        if sec == ".text":
            continue
        known = [x for x in symbols if x.section == sec and count[x.name] == 1 and x.name in at]
        before = max((x for x in known if x.address < start), key=lambda x: x.address, default=None)
        after = min((x for x in known if x.address >= end), key=lambda x: x.address, default=None)
        if before is None or after is None:
            continue
        delta = (at[after.name] - after.address) - (at[before.name] - before.address)
        if delta >= 0:
            continue
        lost = extras.get(sec, [])
        out.append(f"this unit's {sec} links 0x{-delta:X} bytes short of the original's 0x{end - start:X}"
                   + (": the original kept what only discarded weak copies use: "
                      + ", ".join(f"{name} (0x{hi - lo:X})" for lo, hi, name in lost[:6]) if lost else ""))
    return out


def dropped_lines(original: dict[str, int], linked: set[str], ours: dict[str, int]) -> list[str]:
    """`original`: the unit's functions and sizes; `linked`: names the linked ELF defines; `ours`:
    our object's functions and their binding. Dead-stripped ones nothing references were kept by
    the original (force-active); weak ones the link took from an earlier object."""
    missing = sorted((name, size) for name, size in original.items() if name not in linked and not name.startswith("gap_"))
    if not missing:
        return []
    strong = [name for name, _ in missing if ours.get(name) != 2]
    weak = [name for name, _ in missing if ours.get(name) == 2]
    out = [f"the link lacks 0x{sum(size for _, size in missing):X} bytes of this unit's functions: "
           + ", ".join(f"{name} (0x{size:X})" for name, size in missing[:8]) + (" …" if len(missing) > 8 else "")]
    if strong:
        out.append("nothing references them, but the original kept them: ./dk symbol force-active " + " ".join(strong))
    if weak:
        out.append("weak copies the link took from another object (an earlier copy of the same name wins): " + ", ".join(weak))
    return out


def run(cfg: Config, args: argparse.Namespace) -> int:
    try:
        unit = resolve_unit(cfg, args.target)
    except MatchError as e:
        print(f"error: {e}")
        return USAGE
    source = unit["metadata"]["source_path"]
    rel_source = source.removeprefix(cfg.platform_options.get("src", "src") + "/")

    if fresh.build(cfg, [unit["base_path"]]).returncode != 0:
        print("error: unit does not compile")
        hint(f"check {args.target}")
        return MISMATCH
    s = score(cfg, unit["name"], cfg.root / unit["target_path"], cfg.root / unit["base_path"])
    if not s.exact:
        for line in render(s):
            print(line)
        print("REJECTED: unit is not an exact match")
        return MISMATCH

    forbidden = [p for p in commit_gate(cfg, unit) if p.startswith("forbidden")]
    if forbidden:
        for prob in forbidden:
            print(f"REJECTED: {prob}")
        return MISMATCH

    before = _matched_functions(cfg)
    units_path = cfg.root / cfg.platform_options["units"]
    original_units = units_path.read_text()
    units = read_units_file(units_path)
    set_status(units, rel_source, "Matching")
    units_path.write_text(render_units_file(units))

    ok, output = _build(cfg, (unit["base_path"],))
    after = _matched_functions(cfg)
    # The shared tree also holds other agents' half-edited units, so its numbers only warn;
    # the floor is enforced on the clean HEAD + this unit build that the commit is made from.
    if ok and after < before:
        print(f"note: matched functions in the shared tree fell {before} -> {after}; "
              f"the commit's clean build is checked against the floor")
    if not ok:
        print(build_errors(output, 30))
        for note in s.link_notes:                      # a weak copy the link keeps because the original's is unnamed
            print(f"hint: {note}")
        for line in dropped_by_link(cfg, unit):
            print(f"hint: {line}")
        if cfg.built.is_file() and "did NOT match" in output:
            from .firstdiff import first_diff             # where the linked binary differs
            print("first differences in the linked binary:")
            for line in first_diff(cfg, cfg.built, cfg.original, 6):
                print(f"  {line}")
        if s.functions and all(f.name.startswith("__sinit") for f in s.functions):
            print("hint: this unit's only code is a static initializer; nothing references it, so the linker "
                  "drops the whole object (.ctors entry included). If the TU before it has no initializer, it "
                  f"is that TU's tail (./dk unit propose {s.functions[0].name} prints the merged unit); "
                  "otherwise keep it with `#pragma force_active on` after the includes")
        return _restore(cfg, units_path, original_units, "hash mismatch or build error after linking")
    print(f"ACCEPTED: {unit['name']} linked, hash OK, matched functions {before} -> {after}")

    if args.no_commit:
        return OK
    names = ", ".join(f.name for f in s.functions) or "(data only)"
    message = f"Match {rel_source}\n\nFunctions: {names}\n"
    if args.trailer:
        message += "\n" + "\n".join(args.trailer) + "\n"
    ok, detail = commit_unit(cfg, unit, message)
    if not ok and detail.startswith("below the progress floor"):
        return _restore(cfg, units_path, original_units, detail)
    print(f"committed: Match {rel_source} ({detail})" if ok else f"note: not committed ({detail})")
    hint("next", "status")
    return OK


def _restore(cfg: Config, units_path: Path, original: str, reason: str) -> int:
    units_path.write_text(original)
    _build(cfg)
    print(f"REJECTED: {reason}; units.toml restored")
    return MISMATCH


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("target", help="unit (source path or name) or a function inside one")
    p.add_argument("--no-commit", action="store_true", help="link and verify, but do not commit")
    p.add_argument("--trailer", action="append", default=[], help="commit trailer line, e.g. 'Co-Authored-By: ...'")


COMMAND = Command("accept", "matching", "link an exact unit, verify hash + no regression, commit it", run, _configure, locked=True)
