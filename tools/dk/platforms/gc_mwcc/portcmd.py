"""dk port: turn reference-matched library objects into linked units.

For each object from `dk analyze lib` (build/dk/libmatch.json):
  1. derive its full split (text + data) from the DOL      (port.place)
  2. vendor the reference's public headers once            include/sdk/<ref>/
  3. copy the source and its sibling private headers       src/dolphin/<path>
  4. declare the unit (re-split build must still match)    unitcmd.declare
  5. check it; exact -> accept (link + hash + commit), else roll everything back
Results go to docs/generated/ports.md so failures become a hand-work list.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import subprocess
from pathlib import Path

from . import freshness as fresh
from ...core.command import MISMATCH, OK, USAGE, Command, hint
from ...core.config import Config
from ...core.util import run_ninja
from ...core.doclint import GENERATED_MARK
from . import acceptcmd
from .dol import Dol
from .dtk_symbols import SymbolIndex, read_symbols
from .elf import Elf
from .matching import MatchError, resolve_unit, score
from .port import place, sda_bases
from .references import ensure, load_references
from .unitcmd import Transaction, declare, paths

def _vendor_headers(cfg: Config, ref, ref_root: Path) -> None:
    """Copy the reference's header trees (dk.toml `vendor`) into the repo, once.

    An entry is (reference path, our path) or (reference path, our path, [subpaths]); the
    subpath list limits copying to what this project needs (not a reference game's own headers).
    """
    skip = shutil.ignore_patterns("*.c", "*.cpp", "*.cp", "*.s")
    for entry in ref.vendor:
        src, dest = ref_root / entry[0], cfg.root / entry[1]
        parts = entry[2] if len(entry) > 2 else [""]
        for part in parts:
            source, target = (src / part, dest / part) if part else (src, dest)
            if target.exists() or not source.exists():
                continue
            if source.is_dir():
                shutil.copytree(source, target, ignore=skip)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        marker = dest / "SOURCE.md"
        if dest.is_dir() and not marker.exists():
            marker.write_text(f"Vendored from the `{ref.checkout or ref.name}` reference decomp "
                              f"({entry[0]}, see dk.toml). Do not edit by hand.\n")


def port_one(cfg: Config, entry: dict, dol: Dol, symbols: SymbolIndex, sda: dict[int, int],
             commit: bool, trailers: list[str]) -> tuple[str, str]:
    """Returns (status, detail): status is ported | skipped | failed."""
    ref_name, rel = entry["source"].split(":", 1)
    ref = next((r for r in load_references(cfg) if r.name == ref_name), None)
    if ref is None:
        return "skipped", f"reference {ref_name} not in dk.toml"
    if not entry["clean"]:
        return "skipped", f"{len(entry['gaps'])} unexplained function(s) in span"
    ref_root = ensure(cfg, ref)
    placement = place(dol, Elf.parse((cfg.root / entry["object"]).read_bytes()), [tuple(x) for x in entry["matched"]], sda)
    blocking = [p for p in placement.problems if "never referenced" not in p]
    if blocking:
        return "skipped", "; ".join(blocking)

    unit_path = ref.unit_path(rel)
    _, _, src_dir = paths(cfg)
    source_file = ref_root / ref.src_root / rel
    sources = {src_dir / unit_path: source_file.read_text(errors="replace")}
    for header in sorted(source_file.parent.glob("*.h")):
        sources[src_dir / Path(unit_path).parent / header.name] = header.read_text(errors="replace")
    char_flags = [f for f in entry["flags"] if f in ("signed", "unsigned")]
    options = {"cflags": ref_name, "mw_version": entry["compiler"],
               "extra_cflags": ["-char", char_flags[-1]] if char_flags else []}
    lib = str(Path(unit_path).parent)

    _vendor_headers(cfg, ref, ref_root)
    tx = Transaction()
    error = declare(cfg, tx, unit_path, placement.splits(symbols), lib, "NonMatching",
                    {"progress_category": "sdk"}, options, sources)

    def undo(reason: str) -> tuple[str, str]:
        tx.rollback()
        fresh.build(cfg)
        return "failed", reason

    if error:
        return undo(error.split("\n", 1)[0])
    try:
        unit = resolve_unit(cfg, unit_path)
        s = score(cfg, unit["name"], cfg.root / unit["target_path"], cfg.root / unit["base_path"])
    except (MatchError, OSError) as e:
        return undo(f"check failed: {e}")
    if not s.exact:
        bad = [f"{f.name} {f.percent:.1f}%" for f in s.functions if f.percent < 100.0][:4]
        bad += [f"{k} {v:.1f}%" for k, v in s.data.items() if v < 100.0]
        bad += s.layout_problems[:2]
        return undo("not exact: " + ", ".join(bad))

    # In-process: a child `./dk accept` would wait forever on the project lock this command holds.
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = acceptcmd.run(cfg, argparse.Namespace(target=unit_path, no_commit=not commit, trailer=trailers))
    if code != 0:
        lines = [l for l in out.getvalue().splitlines() if l.startswith(("REJECTED", "error"))]
        return undo("accept rejected: " + (lines[-1] if lines else "?"))
    return "ported", f"{len(entry['matched'])} functions, splits {len(placement.splits(symbols))}"


def render(results: list[tuple[str, str, str]]) -> str:
    out = [f"{GENERATED_MARK} port; do not edit -->", "# Library port results", "",
           "Objects from `docs/generated/libraries.md` that `./dk port` tried. `failed`/`skipped` entries",
           "need hand work (revision differences, data layout).", "",
           "| Object | Result | Detail |", "|---|---|---|"]
    for source, status, detail in sorted(results):
        out.append(f"| `{source}` | {status} | {detail.replace('|', '/')} |")
    return "\n".join(out) + "\n"


def run(cfg: Config, args: argparse.Namespace) -> int:
    machine = cfg.root / "build" / "dk" / "libmatch.json"
    if not machine.is_file():
        print("error: no library matches yet")
        hint("analyze lib --apply")
        return USAGE
    entries = json.loads(machine.read_text())
    if args.source:
        entries = [e for e in entries if e["source"] in args.source]
        if not entries:
            print("error: none of those sources are in libmatch.json")
            return USAGE
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=cfg.root).returncode != 0 and not args.no_commit:
        print("error: git index has staged changes; commit or unstage them first")
        return USAGE

    syms = read_symbols(cfg.root / cfg.platform_options["symbols"])
    dol = Dol.parse(cfg.original.read_bytes())
    init_registers = next((s.address for s in syms if s.name == "__init_registers"), None)
    sda = sda_bases(dol, init_registers) if init_registers else {}
    results = []
    _, units_path, _ = paths(cfg)
    refs = {r.name: r for r in load_references(cfg)}
    for entry in sorted(entries, key=lambda e: e["matched"][0][1]):
        ref_name, rel = entry["source"].split(":", 1)
        if ref_name in refs and f'"{refs[ref_name].unit_path(rel)}"' in units_path.read_text():
            continue
        status, detail = port_one(cfg, entry, dol, SymbolIndex(read_symbols(cfg.root / cfg.platform_options["symbols"])),
                                  sda, not args.no_commit, args.trailer)
        results.append((entry["source"], status, detail))
        print(f"{status:<8} {entry['source']:<40} {detail[:110]}")
    doc = cfg.docs.generated / "ports.md"
    previous = {}
    if doc.is_file():
        for line in doc.read_text().splitlines():
            if line.startswith("| `"):
                cells = [c.strip() for c in line.strip("|").split("|")]
                previous[cells[0].strip("`")] = (cells[1], cells[2])
    merged = {**previous, **{s: (st, d) for s, st, d in results}}
    doc.write_text(render([(s, st, d) for s, (st, d) in merged.items()]))
    ported = sum(1 for _, st, _ in results if st == "ported")
    print(f"ported {ported} of {len(results)} attempted -> {cfg.rel(doc)}")
    hint("progress")
    return OK if ported or not results else MISMATCH


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("source", nargs="*", help="libmatch sources to port (default: all), e.g. dolsdk2001:os/OSAlloc.c")
    p.add_argument("--no-commit", action="store_true", help="link and verify, but leave changes uncommitted")
    p.add_argument("--trailer", action="append", default=[], help="commit trailer for each accepted unit")


COMMAND = Command("port", "matching", "turn reference-matched library objects into linked, committed units", run, _configure, locked=True)
