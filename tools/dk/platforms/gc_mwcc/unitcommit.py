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
import tomllib
from collections import defaultdict

from ...core import floor, isolated
from ...core.config import Config
from ...core.isolated import commit_overlay, verify_overlay
from . import dtkconfig
from .matching import unit_commit_paths
from .elf import Elf
from .units import Lib, UnitsFile, render_units_file

_SYMBOL = re.compile(r"^(?P<name>\S+) = (?P<section>[^:\s]+):0x(?P<addr>[0-9A-Fa-f]+);")
_SIZE = re.compile(r"size:0x([0-9A-Fa-f]+)")


_base: dict[str, str] = {}      # shared files already composed by an earlier unit of the same commit


def _head(cfg: Config, rel: str) -> str:
    if rel in _base:
        return _base[rel]
    data = isolated.show(cfg, rel)
    return data.decode() if data is not None else ""


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
            if m["name"] in words or any(part in words for part in m["name"].split("@") if part):
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
    return commit_units(cfg, [unit], message)


def commit_units(cfg: Config, units: list[dict], message: str) -> tuple[bool, str]:
    """Commit HEAD plus the deltas of `units` as one verified commit (a unit split in several,
    or units that need each other)."""
    for attempt in range(isolated.UNLOCKED_TRIES + 1):   # recompose on a HEAD that moved meanwhile
        isolated.pin_head(cfg, attempt)
        ok, detail = _commit_units_once(cfg, units, message)
        if detail != isolated.MOVED:
            return ok, detail
    return ok, detail


def _commit_units_once(cfg: Config, units: list[dict], message: str) -> tuple[bool, str]:
    overlay: dict = {}
    files: list[str] = []
    _base.clear()
    try:
        for unit in units:
            part, own = unit_overlay(cfg, unit)
            for rel, data in part.items():
                if rel not in own and data is not None:      # composed shared file: the next unit builds on it
                    _base[rel] = data.decode()
            overlay.update(part)
            files += own
    finally:
        _base.clear()
    sources = {u["metadata"]["source_path"] for u in units}
    measures: dict = {}
    ok, detail = verify_overlay(cfg, overlay, measures=measures)
    if ok:
        done, result = commit_with_floor(cfg, dict(overlay), measures, message)
        if done or not result.startswith("below the progress floor"):
            return done, result
        failure = result
    dirty = dirty_headers(cfg, overlay, files, sources)
    if not ok:
        why = (f"it includes header(s) with uncommitted edits: {', '.join(dirty)}; if they are another agent's, "
               "retry after that agent commits them" if dirty else "it probably depends on another uncommitted change")
        failure = f"this unit's delta does not build on top of HEAD by itself ({why}):\n" + detail
    # Another agent's half-done edit to a shared header often rides along. Leave out the edits the
    # units do not need (their sources still compile and score as well with HEAD's copy).
    keep = choose_headers(dirty, lambda kept: private_scores(cfg, units, [h for h in dirty if h not in kept]))
    left = [h for h in dirty if h not in keep]
    if not left:
        return False, failure
    retry = dict(overlay)
    for rel in left:
        retry[rel] = isolated.show(cfg, rel)
    note = f"left {', '.join(left)} at HEAD: another agent's in-progress edit"
    measures = {}
    ok, detail = verify_overlay(cfg, retry, measures=measures)
    if not ok:
        return False, f"{failure}\n(also tried with {', '.join(left)} at HEAD; that does not build either)"
    done, result = commit_with_floor(cfg, retry, measures, message.rstrip("\n") + f"\n\n{note[0].upper()}{note[1:]}.\n")
    return done, f"{result}; {note}" if done else result


def dirty_headers(cfg: Config, overlay: dict, files: list[str], sources: set[str]) -> list[str]:
    """Headers in the overlay whose working copy differs from HEAD's (a header new in this
    commit is the unit's own and is never left out)."""
    out = []
    for rel in files:
        if rel in sources or not rel.endswith((".h", ".hpp", ".inc")) or overlay.get(rel) is None:
            continue
        head = isolated.show(cfg, rel)
        if head is not None and head != overlay[rel]:
            out.append(rel)
    return out


def choose_headers(dirty: list[str], evaluate) -> set[str]:
    """The dirty headers whose working edits the units need. `evaluate(kept)` scores the units
    with only `kept` at their working version (a comparable tuple, or None if they do not
    compile). An edit is left out when the units score at least as well without it: all at once
    first, then one at a time."""
    if not dirty:
        return set()
    full = evaluate(set(dirty))
    bare = evaluate(set())
    if bare is not None and (full is None or bare >= full):
        return set()
    keep = set(dirty)
    for rel in dirty:
        trial = evaluate(keep - {rel})
        if trial is not None and (full is None or trial >= full):
            keep.discard(rel)
    return keep


def private_scores(cfg: Config, units: list[dict], at_head: list[str]):
    """(exact functions, exact units) of the units compiled privately with the headers in
    `at_head` replaced by HEAD's copies, or None if one does not compile."""
    import os
    import shutil
    from .matching import MatchError, compile_command, retarget, run_compile, score
    work = cfg.root / "build" / "dk" / "headers" / str(os.getpid())
    shutil.rmtree(work, ignore_errors=True)
    shadow = work / "include"
    try:
        for rel in at_head:
            if not rel.startswith("include/"):
                return None                       # only include/ headers can be shadowed
            dest = shadow / rel.removeprefix("include/")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(isolated.show(cfg, rel) or b"")
        exact_functions = exact_units = 0
        for i, unit in enumerate(units):
            out = work / f"out{i}"
            out.mkdir(parents=True, exist_ok=True)
            try:
                args = retarget(compile_command(cfg, unit["base_path"], locked=False), out)
            except MatchError:
                return None
            first = next((k for k, a in enumerate(args) if a == "-i"), len(args))
            args[first:first] = ["-i", str(shadow)]        # searched before the working include/
            ok, _ = run_compile(cfg, args)
            objs = list(out.glob("*.o"))
            if not ok or not objs:
                return None
            s = score(cfg, unit["name"], cfg.root / unit["target_path"], objs[0])
            exact_functions += sum(f.percent == 100.0 for f in s.functions)
            exact_units += s.exact
        return exact_functions, exact_units
    finally:
        shutil.rmtree(work, ignore_errors=True)


def unit_overlay(cfg: Config, unit: dict) -> tuple[dict, list[str]]:
    """(HEAD + this unit's delta as an overlay, the unit's own files in it)."""
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
    work_symbols = (cfg.root / opts["symbols"]).read_text()
    addr_of = {m["name"]: (m["section"], int(m["addr"], 16)) for line in work_symbols.splitlines()
               if (m := _SYMBOL.match(line.strip()))}
    in_unit = lambda n: n in addr_of and any(sec == addr_of[n][0] and lo <= addr_of[n][1] < hi for sec, lo, hi in ranges)
    dtk_rel = dtkconfig.path(cfg)
    head_cfg = _head(cfg, dtk_rel)
    composed = dtkconfig.compose(head_cfg, (cfg.root / dtk_rel).read_text(), in_unit)
    if composed != head_cfg:
        overlay[dtk_rel] = composed.encode()
    overlay[opts["symbols"]] = compose_symbols(_head(cfg, opts["symbols"]), (cfg.root / opts["symbols"]).read_text(),
                                               ranges, words).encode()
    return overlay, files


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
