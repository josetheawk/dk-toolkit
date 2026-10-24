"""Compile a unit, score it against its target object, and check in-unit layout.

objdiff scores functions by name. It does not notice when every function matches but
their order inside the unit differs, which still breaks the linked binary, so layout is
compared separately from the ELF symbol offsets.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ...core import floor, quality
from ...core.config import Config
from ...core.util import run_ninja
from .dtk_symbols import read_symbols, read_units
from .elf import Elf


class MatchError(Exception):
    """Environment or input problem (unknown unit, missing build files)."""


@dataclass
class FunctionScore:
    name: str
    size: int
    percent: float


@dataclass
class UnitScore:
    unit: str
    functions: list[FunctionScore] = field(default_factory=list)
    data: dict[str, float] = field(default_factory=dict)        # section -> percent
    layout_problems: list[str] = field(default_factory=list)
    stripped: list[str] = field(default_factory=list)    # in source, absent from target (linker drops them)
    weak_copies: list[str] = field(default_factory=list)  # data the original link discarded (weak copies, stripped-only)

    @property
    def code_percent(self) -> float:
        total = sum(f.size for f in self.functions)
        return 100.0 if not total else sum(f.size * f.percent for f in self.functions) / total

    @property
    def exact(self) -> bool:
        return (all(f.percent == 100.0 for f in self.functions)
                and all(p == 100.0 for p in self.data.values())
                and not self.layout_problems)


def load_objdiff_units(cfg: Config) -> list[dict]:
    path = cfg.root / "objdiff.json"
    if not path.is_file():
        raise MatchError("objdiff.json missing (run ./dk setup or ./dk build)")
    return json.loads(path.read_text())["units"]


def resolve_unit(cfg: Config, query: str) -> dict:
    """Find a source-backed objdiff unit by unit name, source path, or contained function."""
    units = [u for u in load_objdiff_units(cfg) if u.get("base_path")]
    q = query.removeprefix("src/")
    for u in units:
        src = u.get("metadata", {}).get("source_path", "").removeprefix("src/")
        if q in (u["name"], u["name"].split("/", 1)[-1], src) or (src and src.rsplit(".", 1)[0] == q):
            return u
    symbols = {s.name: s for s in read_symbols(cfg.root / cfg.platform_options["symbols"])}
    sym = symbols.get(query)
    if sym is None:
        raise MatchError(f"no unit or symbol named {query!r} (declare one with ./dk unit add)")
    for split_unit in read_units(cfg.root / cfg.platform_options["splits"]):
        if any(sec == sym.section and start <= sym.address < end for sec, start, end in split_unit.splits):
            stem = split_unit.name.rsplit(".", 1)[0]
            for u in units:
                if u["name"].split("/", 1)[-1] == stem:
                    return u
            raise MatchError(f"{query} is in {split_unit.name}, which has no source unit in units.toml")
    raise MatchError(f"{query} at 0x{sym.address:08X} is not inside any declared unit (./dk unit add)")


def compile_command(cfg: Config, base_path: str) -> str:
    """The project's own compile command for an object, without the dependency post-step."""
    proc = run_ninja(cfg.root, ["-t", "commands", base_path])
    if proc.returncode != 0 or not proc.stdout.strip():
        raise MatchError(f"no build command for {base_path}: {proc.stderr.strip()}")
    return proc.stdout.strip().splitlines()[-1].split(" && ", 1)[0]


def retarget(command: str, out_dir: Path, compiler: str | None = None, extra_flags: tuple[str, ...] = ()) -> list[str]:
    """Rewrite a compile command: output dir, compiler version, extra trailing flags."""
    args = shlex.split(command)
    out: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "-MMD":
            i += 1
            continue
        if a == "-o":
            out += ["-o", str(out_dir)]
            i += 2
            continue
        if compiler and re.search(r"compilers/[^/]+/[^/]+/mwcceppc\.exe$", a):
            a = re.sub(r"compilers/[^/]+/[^/]+/mwcceppc\.exe$", f"compilers/{compiler}/mwcceppc.exe", a)
        if a == "-c" and extra_flags:
            out += list(extra_flags)
        out.append(a)
        i += 1
    return out


def run_compile(cfg: Config, args: list[str]) -> tuple[bool, str]:
    proc = subprocess.run(args, cwd=cfg.root, capture_output=True, text=True)
    text = "\n".join(line for line in (proc.stdout + proc.stderr).splitlines()
                     if "Usage Warning" not in line and "Specified directory" not in line)
    return proc.returncode == 0, text.strip()


def _function_offsets(path: Path) -> dict[str, tuple[str, int, int]]:
    elf = Elf.parse(path.read_bytes())
    out = {}
    for s in elf.symbols:
        if s.type == 2 and 0 < s.shndx < len(elf.sections):
            out[s.name] = (elf.sections[s.shndx].name, s.value, s.size)
    return out


STB_WEAK = 2


def _weak_extras(target: Path, base: Path) -> tuple[dict[str, list[tuple[int, int, str]]], set[str]]:
    """Data the compiled object has that the original link discarded, per section, and which of
    those names the original linker may have zero-filled in place (weak copies and their closure).

    Inline virtuals make every TU emit a weak copy of the base class's vtable (and similar);
    the linker keeps only the first. Objects referenced *only* by such discarded copies (the
    local RTTI record a weak vtable points at) were dead-stripped with them, transitively.
    Functions the target lacks were dead-stripped too, and with them the literals and statics
    only they referenced, plus objects the target lacks that nothing references (those are
    removed, never zero-filled).
    """
    want = {s.name for s in Elf.parse(target.read_bytes()).symbols if s.shndx != 0}   # defined there
    elf = Elf.parse(base.read_bytes())
    sized = [s for s in elf.symbols if 0 < s.shndx < len(elf.sections) and s.size and s.name]
    extras = {s.name for s in sized if s.bind == STB_WEAK and s.type != 2 and s.name not in want}

    # Who references each symbol: the sized symbol (function or object) containing each relocation.
    referrers: dict[str, set[str]] = {}
    by_section: dict[str, list[tuple[int, int, str]]] = {}
    for s in sized:
        by_section.setdefault(elf.sections[s.shndx].name, []).append((s.value, s.value + s.size, s.name))

    def symbol_at(section: str, offset: int) -> str | None:
        return next((n for lo, hi, n in by_section.get(section, []) if lo <= offset < hi), None)

    for sec in elf.sections:
        relocs = elf.relocations(sec.index)
        if not relocs:
            continue
        for r in relocs:
            holder = symbol_at(sec.name, r.offset) or f"@{sec.name}"
            # Resolve by address: references often go through a section symbol plus an addend.
            target_name = symbol_at(r.target_section, r.target_value + r.addend) if r.target_section else None
            referrers.setdefault(target_name or r.symbol, set()).add(holder)
    objects = {s.name for s in sized if s.type != 2 and s.name not in want}
    zero_fillable = _discarded(objects, referrers, extras)
    stripped = {s.name for s in sized if s.type == 2 and s.name not in want}
    # The target lacks these and nothing in the unit uses them: the link found no user at all.
    unused = {n for n in objects if not referrers.get(n)}
    extras = _discarded(objects, referrers, zero_fillable | stripped | unused)

    out: dict[str, list[tuple[int, int, str]]] = {}
    for s in sized:
        if s.name in extras and s.type != 2:
            out.setdefault(elf.sections[s.shndx].name, []).append((s.value, s.value + s.size, s.name))
    return out, zero_fillable


def _discarded(objects: set[str], referrers: dict[str, set[str]], seeds: set[str]) -> set[str]:
    """`seeds` plus every one of `objects` whose referrers are all discarded, transitively.
    An object with no known referrer is kept: it may be reached from outside the unit."""
    out = set(seeds)
    changed = True
    while changed:
        changed = False
        for name in sorted(objects - out):
            who = referrers.get(name, set())
            if who and who <= out:
                out.add(name)
                changed = True
    return out


def _alignment(value: int, size: int) -> int:
    """Natural alignment of an object: a power of two up to 8, no larger than its size and
    dividing its offset in the compiled object."""
    align = 8
    while align > 1 and (align > size or value % align):
        align //= 2
    return align


def _packed_offsets(base: Elf, section: str, dropped: set[str]) -> dict[str, int]:
    """Where the original link put each kept object of a section: in place when nothing was
    discarded, else packed in order at its alignment."""
    b = next((x for x in base.sections if x.name == section), None)
    if b is None:
        return {}
    syms = sorted((x for x in base.symbols if x.shndx == b.index and x.size and x.name), key=lambda x: x.value)
    if not any(x.name in dropped for x in syms):
        return {x.name: x.value for x in syms}
    out, cursor = {}, 0
    for x in syms:
        if x.name in dropped:
            continue
        align = _alignment(x.value, x.size)
        cursor = (cursor + align - 1) & ~(align - 1)
        out[x.name] = cursor
        cursor += x.size
    return out


def _nobits_layout_problems(target: Elf, base: Elf, dropped: dict[str, set[str]]) -> list[str]:
    """BSS has no bytes, so a byte diff passes any order: compare the offsets of objects both
    sides name (after the link's discards)."""
    problems = []
    for t in target.sections:
        if t.type != 8 or not t.name:
            continue
        want = {x.name: x.value for x in target.symbols if x.shndx == t.index and x.size and x.name}
        have = _packed_offsets(base, t.name, dropped.get(t.name, set()))
        for name in sorted(set(want) & set(have), key=lambda n: want[n]):
            if want[name] != have[name]:
                problems.append(f"{name}: {t.name}+0x{have[name]:X}, target {t.name}+0x{want[name]:X}")
    return problems


def _packed_section_matches(target: Elf, base: Elf, section: str, dropped: set[str]) -> bool:
    """Model the original link of one relocation-free section: remove the discarded objects,
    pack the kept ones in order at their alignment, and compare with the target. Alignment gaps
    and the end may hold zero padding. BSS has no bytes, so each kept object must instead start
    where the target has a label. Literal pools lose entries from the middle this way, which
    objdiff's byte diff cannot line up."""
    t = next((x for x in target.sections if x.name == section), None)
    b = next((x for x in base.sections if x.name == section), None)
    if t is None or b is None or target.relocations(t.index) or base.relocations(b.index):
        return False
    kept = sorted((x.value, x.size) for x in base.symbols
                  if x.shndx == b.index and x.size and x.name not in dropped)
    if not kept:
        return False
    nobits = b.type == 8
    labels = {x.value for x in target.symbols if x.shndx == t.index and x.size}
    cursor = 0
    for value, size in kept:
        align = _alignment(value, size)
        pos = (cursor + align - 1) & ~(align - 1)
        if nobits:
            if pos not in labels:
                return False
        elif t.data[cursor:pos].strip(b"\0") or t.data[pos:pos + size] != b.data[value:value + size]:
            return False
        cursor = pos + size
    rest = t.size - cursor
    return 0 <= rest < 8 and (nobits or not t.data[cursor:].strip(b"\0"))


def _only_trailing_padding(left: dict, right: dict) -> bool:
    """True if a data section differs only by zero bytes (< 32) at the end of the target's copy:
    alignment padding before the next object, which a split boundary may give to either side."""
    import base64

    chunks = left.get("data_diff", [])
    if [i for i, d in enumerate(chunks) if d.get("kind")] != [len(chunks) - 1]:
        return False
    raw = base64.b64decode(chunks[-1].get("data", ""))
    return (0 < len(raw) < 32 and not raw.strip(b"\0")
            and not any(d.get("kind") and d.get("data") for d in right.get("data_diff", [])))


def _only_weak_extras(left: dict, right: dict, ranges: list[tuple[int, int, str]],
                      zero_fillable: set[str] | None = None) -> bool:
    """True if a data section differs only where weak copies explain it.

    Two shapes, both from the original link discarding duplicate weak objects:
      - base-only bytes inside a weak symbol the target does not define (trailing copy dropped)
      - target bytes that are all zero, of the size of such a weak symbol (a copy in the middle
        of the section that the original linker zero-filled instead of removing)
    """
    import base64

    weak_sizes = sorted(hi - lo for lo, hi, name in ranges if zero_fillable is None or name in zero_fillable)
    for d in left.get("data_diff", []):
        if d.get("kind") and d.get("data"):
            raw = base64.b64decode(d["data"])
            if raw.strip(b"\0") or len(raw) not in weak_sizes:
                return False        # the target has real bytes the compiled object does not reproduce
            weak_sizes.remove(len(raw))
    # Base-only chunks must sit inside the extras, where adjacent extras (and the alignment
    # padding after each) form one span: objdiff reports a string, its padding and the next
    # weak object as a single chunk. All-zero chunks (objects whose pointers are relocations)
    # align ambiguously in objdiff's diff, so those may be explained by size.
    spans: list[list[int]] = []
    for lo, hi, _ in sorted(ranges):
        hi = (hi + 3) & ~3
        if spans and lo <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], hi)
        else:
            spans.append([lo, hi])
    ranges = [(lo, hi, "") for lo, hi in spans]
    budget = sum(hi - lo for lo, hi, _ in ranges)
    offset = 0
    for d in right.get("data_diff", []):
        size = int(d.get("size", 0) or 0)
        if d.get("kind") and d.get("data"):
            raw = base64.b64decode(d["data"])
            # a chunk may start in the zero alignment padding just before a span
            if any(lo - 7 <= offset and offset + size <= hi and not raw[:max(0, lo - offset)].strip(b"\0")
                   for lo, hi, _ in ranges):
                budget -= size
            elif not base64.b64decode(d["data"]).strip(b"\0") and size <= budget:
                budget -= size
            else:
                return False
        if d.get("data"):
            offset += size
    return True


def score(cfg: Config, unit_name: str, target: Path, base: Path) -> UnitScore:
    objdiff = cfg.root / "build" / "tools" / "objdiff-cli"
    # data_value: a relocation only matches if the data it points at matches too (catches a wrong
    # float literal in otherwise identical code, which the lenient default scores as 100%).
    proc = subprocess.run([str(objdiff), "diff", "-1", str(target), "-2", str(base), "-o", "-", "--format", "json",
                           "-c", "functionRelocDiffs=data_value"],
                          cwd=cfg.root, capture_output=True, text=True)
    if proc.returncode != 0:
        raise MatchError(f"objdiff failed: {proc.stderr.strip()}")
    doc = json.loads(proc.stdout)
    left, right = doc["left"], doc.get("right", {})
    result = UnitScore(unit_name)
    for sym in left.get("symbols", []):
        if "instructions" in sym:
            result.functions.append(FunctionScore(sym["name"], int(sym.get("size", 0)), float(sym.get("match_percent", 0.0))))
    weak, zero_fillable = _weak_extras(target, base)
    target_elf, base_elf = Elf.parse(target.read_bytes()), Elf.parse(base.read_bytes())
    right_sections = {sec["name"]: sec for sec in right.get("sections", [])}
    for sec in left.get("sections", []):
        if sec.get("kind") in ("SECTION_DATA", "SECTION_BSS"):
            pct = float(sec.get("match_percent", 0.0))
            ranges = weak.get(sec["name"], [])
            if pct < 100.0 and ranges and _only_weak_extras(sec, right_sections.get(sec["name"], {}), ranges, zero_fillable):
                pct = 100.0
                result.weak_copies += [name for _, _, name in ranges]
            if pct < 100.0 and _only_trailing_padding(sec, right_sections.get(sec["name"], {})):
                pct = 100.0
            dropped = {name for _, _, name in ranges}
            if pct < 100.0 and _packed_section_matches(target_elf, base_elf, sec["name"], dropped):
                pct = 100.0
                result.weak_copies += sorted(dropped)
            result.data[sec["name"]] = pct

    # Layout after dead-stripping: functions the target lacks are dropped by the linker when
    # unreferenced (the hash check at accept proves it), so compare packed offsets instead.
    want = _function_offsets(target)
    have = _function_offsets(base)
    stripped = sorted(set(have) - set(want))
    shift: dict[str, int] = {}
    for name, (sec, off, size) in sorted(have.items(), key=lambda kv: (kv[1][0], kv[1][1])):
        removed = sum(have[x][2] for x in stripped if have[x][0] == sec and have[x][1] < off)
        shift[name] = off - removed
    for name, (sec, off, size) in sorted(want.items(), key=lambda kv: (kv[1][0], kv[1][1])):
        got = have.get(name)
        if got is None:
            result.layout_problems.append(f"{name}: missing from compiled object")
        elif got[0] != sec or shift[name] != off:
            result.layout_problems.append(f"{name}: at {got[0]}+0x{shift[name]:X} (after stripping), target {sec}+0x{off:X}")
    result.stripped = stripped
    result.layout_problems += _nobits_layout_problems(
        target_elf, base_elf, {sec: {name for _, _, name in ranges} for sec, ranges in weak.items()})
    return result


def render(s: UnitScore) -> list[str]:
    lines = [f"unit {s.unit}"]
    for f in s.functions:
        lines.append(f"  {f.percent:7.2f}%  {f.name:<50} 0x{f.size:X}")
    if s.data:
        lines.append("  data: " + "  ".join(f"{k} {v:.2f}%" for k, v in sorted(s.data.items())))
    if s.stripped:
        lines.append(f"  dead-stripped (in source, not in target): {', '.join(s.stripped)}")
    if s.weak_copies:
        lines.append(f"  data the original link discarded (weak copies, used only by stripped code; ignored): {', '.join(sorted(set(s.weak_copies)))}")
    lines.append("  layout: " + ("OK" if not s.layout_problems else "MISMATCH"))
    lines += [f"    {p}" for p in s.layout_problems]
    return lines


def unit_commit_paths(cfg: Config, unit: dict) -> list[str]:
    """What a unit's commit must contain: its source, the repo headers it includes (from
    ninja's recorded deps), and the shared config files (units, splits, symbols)."""
    paths = [unit["metadata"]["source_path"]]
    proc = subprocess.run(["ninja", "-t", "deps", unit["base_path"]], cwd=cfg.root, capture_output=True, text=True)
    for line in proc.stdout.splitlines()[1:]:
        dep = Path(line.strip())
        if not line.startswith(" ") or not dep.suffix:
            continue
        full = dep if dep.is_absolute() else cfg.root / dep
        try:
            rel = full.resolve().relative_to(cfg.root)
        except ValueError:
            continue
        if rel.parts and rel.parts[0] not in ("build", "orig"):
            paths.append(str(rel))
    opts = cfg.platform_options
    paths += [opts["units"], opts["splits"], opts["symbols"], cfg.rel(floor.path(cfg))]
    return sorted(set(paths))


def commit_gate(cfg: Config, unit: dict) -> list[str]:
    """Reasons a unit's commit must be refused: forbidden patterns in its own files, or progress
    below the floor. Empty list = allowed."""
    files = [cfg.root / p for p in unit_commit_paths(cfg, unit) if p.endswith((".c", ".cpp", ".cp", ".h", ".hpp", ".inc"))]
    problems = [f"forbidden: {h}" for h in quality.forbidden_hits(cfg, quality.judged_files(cfg, files))]
    problems += [f"below progress floor: {m}" for m in floor.check(cfg)]
    return problems
