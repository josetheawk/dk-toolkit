"""Commit one unit as exactly "HEAD + this unit's own delta", verified in a clean worktree.

The shared files (units.toml, splits.txt, symbols.txt) hold every agent's in-progress work.
Committing them whole once let another agent's uncommitted rename land without the header it
depended on, so HEAD did not build. Instead each shared file is composed from HEAD plus only:
  units.toml   this unit's object entry (and its lib, if new)
  splits.txt   this unit's block
  symbols.txt  symbols inside or straddling this unit's ranges, and symbols whose (old or new)
               name appears in the files committed with the unit (callees it renamed)
The overlay (composed shared files + the unit's source + the repo headers it includes + the
progress floor) must build and match from a clean checkout before it is committed.
"""

from __future__ import annotations

import re
import subprocess
import tomllib
from collections import defaultdict

from ...core import floor
from ...core.config import Config
from ...core.isolated import commit_overlay, verify_overlay
from .matching import unit_commit_paths
from .elf import Elf
from .units import Lib, UnitsFile, render_units_file

_SYMBOL = re.compile(r"^(?P<name>\S+) = (?P<section>[^:\s]+):0x(?P<addr>[0-9A-Fa-f]+);")
_SIZE = re.compile(r"size:0x([0-9A-Fa-f]+)")


def _head(cfg: Config, rel: str) -> str:
    proc = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=cfg.root, capture_output=True, text=True)
    return proc.stdout if proc.returncode == 0 else ""


# --- units.toml ------------------------------------------------------------------------------

def compose_units(head_text: str, work_text: str, path: str, absorbed: tuple = ()) -> str:
    def parse(text: str) -> UnitsFile:
        header = []
        for line in text.splitlines():
            if not line.startswith("#"):
                break
            header.append(line)
        data = tomllib.loads(text)
        libs = []
        for raw in data.get("lib", []):
            settings = {k: raw[k] for k in ("mw_version", "cflags", "extra_cflags", "progress_category") if k in raw}
            libs.append(Lib(raw["name"], settings, [dict(o) for o in raw.get("objects", [])]))
        return UnitsFile(header, libs)

    head, work = parse(head_text) if head_text else parse(work_text), parse(work_text)
    if not head_text:
        head.libs = []
    for gone in absorbed:                         # merged into this unit: their entries go too
        if work.find(gone) is None:
            for lib in head.libs:
                lib.objects = [o for o in lib.objects if o["path"] != gone]
    found = work.find(path)
    existing = head.find(path)
    if found is not None and existing is not None and existing[0].name == found[0].name:
        lib, obj = existing                       # same lib: replace in place (keeps order)
        lib.objects[lib.objects.index(obj)] = dict(found[1])
        return render_units_file(head)
    for lib in head.libs:                         # moved or removed: drop the HEAD entry
        lib.objects = [o for o in lib.objects if o["path"] != path]
    if found is not None:
        wlib, wobj = found
        hlib = head.lib(wlib.name)
        if hlib is None:
            hlib = Lib(wlib.name, dict(wlib.settings), [])
            head.libs.append(hlib)
        hlib.objects.append(dict(wobj))
    head.libs = [lib for lib in head.libs if lib.objects]
    return render_units_file(head)


# --- splits.txt ------------------------------------------------------------------------------

def _blocks(text: str) -> tuple[list[str], dict[str, list[str]], list[str]]:
    """(lines before the first unit, {unit: block lines}, unit order)."""
    lines = text.splitlines()
    pre: list[str] = []
    blocks: dict[str, list[str]] = {}
    order: list[str] = []
    current = None
    for line in lines:
        is_header = line and not line[0].isspace() and not line.startswith(("Sections:", "//"))
        if is_header:
            current = line.split(":", 1)[0]
            blocks[current] = [line]
            order.append(current)
        elif current is None:
            pre.append(line)
        elif line.strip():
            blocks[current].append(line)
    return pre, blocks, order


def _first_address(block: list[str]) -> int:
    addrs = [int(m[1], 16) for line in block[1:] if (m := re.search(r"start:0x([0-9A-Fa-f]+)", line))]
    return min(addrs, default=0)


def absorbed_units(head_text: str, work_text: str, unit: str) -> list[str]:
    """HEAD units that this unit's new ranges overlap and that the working tree no longer
    declares: merged into it (their blocks must leave the same commit, or splits overlap)."""
    _, hblocks, _ = _blocks(head_text)
    _, wblocks, _ = _blocks(work_text)
    ranges = unit_ranges(work_text, unit)
    out = []
    for name, block in hblocks.items():
        if name == unit or name in wblocks:
            continue
        if any(sec == rsec and a < rend and rstart < b
               for sec, a, b in unit_ranges(head_text, name) for rsec, rstart, rend in ranges):
            out.append(name)
    return sorted(out)


def compose_splits(head_text: str, work_text: str, unit: str, absorbed: tuple = ()) -> str:
    pre, blocks, order = _blocks(head_text or work_text)
    if not head_text:
        blocks, order = {}, []
    _, wblocks, _ = _blocks(work_text)
    for gone in absorbed:
        blocks.pop(gone, None)
    order = [u for u in order if u not in absorbed]
    blocks.pop(unit, None)
    order = [u for u in order if u != unit]
    if unit in wblocks:
        blocks[unit] = wblocks[unit]
        order.append(unit)
    order.sort(key=lambda u: _first_address(blocks[u]))
    out = "\n".join(pre).rstrip("\n") + "\n\n"
    out += "\n\n".join("\n".join(blocks[u]) for u in order)
    return out.rstrip("\n") + "\n"


def unit_ranges(work_splits: str, unit: str) -> list[tuple[str, int, int]]:
    _, blocks, _ = _blocks(work_splits)
    out = []
    for line in blocks.get(unit, [])[1:]:
        m = re.match(r"^\s+(\S+)\s+start:0x([0-9A-Fa-f]+)\s+end:0x([0-9A-Fa-f]+)", line)
        if m:
            out.append((m[1], int(m[2], 16), int(m[3], 16)))
    return out


# --- symbols.txt -----------------------------------------------------------------------------

def _by_address(text: str) -> dict[tuple[str, int], list[str]]:
    out: dict[tuple[str, int], list[str]] = defaultdict(list)
    for line in text.splitlines():
        m = _SYMBOL.match(line.strip())
        if m:
            out[(m["section"], int(m["addr"], 16))].append(line)
    return out


def compose_symbols(head_text: str, work_text: str, ranges: list[tuple[str, int, int]], words: set[str]) -> str:
    head, work = _by_address(head_text), _by_address(work_text)

    def relevant(key: tuple[str, int]) -> bool:
        section, addr = key
        for line in head.get(key, []) + work.get(key, []):
            m = _SYMBOL.match(line.strip())
            size = int(s[1], 16) if (s := _SIZE.search(line)) else 0
            if m["name"] in words:
                return True
            for sec, lo, hi in ranges:
                if sec == section and addr < hi and addr + max(size, 1) > lo:
                    return True
        return False

    lines = []
    for key in sorted(set(head) | set(work), key=lambda k: (k[1], k[0])):
        lines += work.get(key, []) if relevant(key) else head.get(key, [])
    return "\n".join(lines) + "\n"


# --- the commit ------------------------------------------------------------------------------

def commit_unit(cfg: Config, unit: dict, message: str) -> tuple[bool, str]:
    opts = cfg.platform_options
    source = unit["metadata"]["source_path"]
    rel_unit = source.removeprefix(opts.get("src", "src") + "/")
    files = [p for p in unit_commit_paths(cfg, unit) if p not in (opts["units"], opts["splits"], opts["symbols"])]
    overlay = {}
    for rel in files:
        f = cfg.root / rel
        overlay[rel] = f.read_bytes() if f.is_file() else None
    words = set()
    for rel in files:
        if rel.endswith((".c", ".cpp", ".cp", ".h", ".hpp", ".inc")) and (cfg.root / rel).is_file():
            words |= set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", (cfg.root / rel).read_text(errors="replace")))
    work_splits = (cfg.root / opts["splits"]).read_text()
    ranges = unit_ranges(work_splits, rel_unit)
    # mangled names of functions defined in the unit's source also count (rename targets)
    words |= _mangled_names(cfg, unit)
    absorbed = tuple(absorbed_units(_head(cfg, opts["splits"]), work_splits, rel_unit))
    for gone in absorbed:                          # a merged unit's source leaves with it, if removed
        src = f"{opts.get('src', 'src')}/{gone}"
        if not (cfg.root / src).exists():
            overlay[src] = None
    overlay[opts["units"]] = compose_units(_head(cfg, opts["units"]), (cfg.root / opts["units"]).read_text(), rel_unit,
                                           absorbed).encode()
    overlay[opts["splits"]] = compose_splits(_head(cfg, opts["splits"]), work_splits, rel_unit, absorbed).encode()
    overlay[opts["symbols"]] = compose_symbols(_head(cfg, opts["symbols"]), (cfg.root / opts["symbols"]).read_text(),
                                               ranges, words).encode()
    measures: dict = {}
    ok, detail = verify_overlay(cfg, overlay, measures=measures)
    if not ok:
        return False, ("this unit's delta does not build on top of HEAD by itself (it probably depends on "
                       "another uncommitted change; retry after that lands):\n" + detail)
    return commit_with_floor(cfg, overlay, measures, message)


def commit_with_floor(cfg: Config, overlay: dict, measures: dict, message: str) -> tuple[bool, str]:
    """Commit a verified overlay if its clean-tree progress is not below the committed floor,
    raising the floor to it. Judged on the clean tree, never on the shared one (where other
    agents' half-edited units come and go)."""
    below, floor_text = floor.gate(cfg, measures)
    if below:
        return False, "below the progress floor: " + "; ".join(below)
    floor_rel = cfg.rel(floor.path(cfg))
    overlay[floor_rel] = floor_text.encode()
    ok, detail = commit_overlay(cfg, overlay, message)
    if ok:
        floor.path(cfg).write_text(floor_text)
    return ok, detail


def _mangled_names(cfg: Config, unit: dict) -> set[str]:
    base = cfg.root / unit["base_path"]
    if not base.is_file():
        return set()
    return {s.name for s in Elf.parse(base.read_bytes()).symbols if s.name}
