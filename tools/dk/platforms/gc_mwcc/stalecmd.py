"""dk stale [--fix]: names the sources use that symbols.txt has since changed.

Both kinds only show up as near-miss notes otherwise, one unit at a time:

1. C link names. A source that declares `fn_ADDR` as a C name (`extern "C"`, asm, a C file) and
   calls it, while symbols.txt now names that address otherwise, links to a symbol the original
   does not have. `--fix` (sources only; headers are listed for a hand fix) drops such a
   declaration when nothing in the file uses it, and otherwise renames the C name to the
   symbols.txt name throughout the file (a C declaration may carry a mangled name; the call
   links the same). It leaves a file alone that also declares or calls a member of that name.
   Each changed unit is compiled privately before and after, and the change is kept only if
   nothing scores lower.
2. Virtual names. Vtable slots where the target (symbols.txt) and our object name different
   functions, from the last build's objects (`./dk check --all` refreshes them), with where each
   name is declared and defined. When both are members of one class, one of the two names is
   stale: usually symbols.txt's, named after a placeholder member some unit defined, so rename
   it to the header's slot name and rename that definition with it.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ...core.command import MISMATCH, OK, USAGE, Command, hint
from ...core.config import Config
from ...core.gitops import git
from .dtk_symbols import read_symbols
from .elf import Elf
from .matching import MatchError, load_objdiff_units, vtable_problems
from .renamecmd import SOURCE_SUFFIXES, _COMMENT, c_uses

FN = re.compile(r"\bfn_([0-9A-F]{8})\b")


@dataclass
class StaleUse:
    path: str
    name: str                   # the C name the source uses
    new: str                    # symbols.txt's name for that address
    decl_lines: list[int] = field(default_factory=list)    # C-name declarations / asm uses
    use_lines: list[int] = field(default_factory=list)     # other plain uses (calls, addresses)
    member: bool = False        # the file also has a member of that name (`.x`, `->x`, `::x`, a class body)
    defines: bool = False       # the file defines the function under the C name

    @property
    def action(self) -> str:
        if self.path.endswith((".h", ".hpp", ".inc")):
            return "a header: fix by hand (every includer links through it)"
        if self.defines:
            return "the file defines it under the C name: rename the definition by hand"
        if self.member:
            return "the file also uses a member of that name: fix by hand"
        if not self.use_lines:
            return "unused declaration: --fix drops it"
        return f"{len(self.use_lines)} use(s): --fix renames it in this file"


def _strip_comments(text: str) -> str:
    return _COMMENT.sub(lambda m: re.sub(r"[^\n]", " ", m[0]), text)


def _class_body_lines(text: str) -> set[int]:
    """0-based lines inside struct/class/union bodies (where a name is a member)."""
    from .reorder import SplitError, split_items
    try:
        items = split_items(text)
    except SplitError:
        return set()
    return {n for it in items if re.match(r"\s*(template\s*<[^>]*>\s*)?(typedef\s+)?(struct|class|union)\b", it.head)
            for n in range(it.body, it.end + 1)}


def stale_uses(path: str, text: str, names_at: dict[int, str]) -> list[StaleUse]:
    """The C names `text` uses whose address symbols.txt names otherwise."""
    wanted = {f"fn_{a}" for a in FN.findall(text) if names_at.get(int(a, 16), f"fn_{a}") != f"fn_{a}"}
    if not wanted:
        return []
    is_c = path.endswith(".c")
    decls: dict[str, set[int]] = defaultdict(set)
    for name, line in c_uses(text, wanted, is_c):
        decls[name].add(line)
    if not decls:
        return []
    code = _strip_comments(text)
    body = _class_body_lines(text)
    out = []
    for name in sorted(decls):
        plain, member = set(), False
        for m in re.finditer(rf"(?<![\w])({re.escape(name)})\b(?!__)", code):
            line = code.count("\n", 0, m.start())
            before = code[max(0, m.start() - 2):m.start()]
            if before.endswith((".", "::", "->")) or line in body:
                member = True
            elif line + 1 not in decls[name]:
                plain.add(line + 1)
        new = names_at[int(name[3:], 16)]
        defines = bool(re.search(rf"\b{re.escape(name)}\s*\([^;{{}}]*\)\s*(const\s*)?\{{", code))
        out.append(StaleUse(path, name, new, sorted(decls[name]), sorted(plain), member, defines))
    return out


_DECL_LINE = r'^\s*(?:extern\s+"C"\s+)?[^;{{}}()]*\b{name}\s*\([^;{{}}]*\)\s*;\s*(?://.*)?$'


def fix_text(text: str, use: StaleUse) -> str | None:
    """The file with the stale C name fixed, or None when that is not mechanical."""
    if use.member or use.defines or use.path.endswith((".h", ".hpp", ".inc")):
        return None
    lines = text.split("\n")
    if not use.use_lines:
        pattern = re.compile(_DECL_LINE.format(name=re.escape(use.name)))
        if not all(pattern.match(lines[n - 1]) for n in use.decl_lines):
            return None
        return "\n".join(line for n, line in enumerate(lines, 1) if n not in use.decl_lines)
    return re.sub(rf"(?<![\w.:>]){re.escape(use.name)}\b(?!__)", use.new, text)


@dataclass
class SlotName:
    vtable: str
    offset: int
    target: str
    ours: str
    units: list[str] = field(default_factory=list)


def _class_of(mangled: str) -> str | None:
    m = re.match(r"(?:@\d+@)*(?:@\d+@)?[\w]+?__(\d+)([A-Za-z_]\w*)", mangled)
    if m:
        return m.group(2)[:int(m.group(1))]
    return None


def slot_names(cfg: Config) -> list[SlotName]:
    """Vtable slots named differently in the target and in our last build, merged across units."""
    found: dict[tuple[str, int, str, str], SlotName] = {}
    for u in load_objdiff_units(cfg):
        base, target = u.get("base_path"), u.get("target_path")
        if not base or not (cfg.root / base).is_file() or not target or not (cfg.root / target).is_file():
            continue
        try:
            problems = vtable_problems(Elf.parse((cfg.root / target).read_bytes()), Elf.parse((cfg.root / base).read_bytes()))
        except ValueError:
            continue
        for p in problems:
            m = re.match(r"(\S+)\+0x([0-9A-F]+): target (\S+), ours (\S+)", p)
            if not m:
                continue
            key = (m[1], int(m[2], 16), m[3], m[4])
            found.setdefault(key, SlotName(*key)).units.append(u["name"].split("/", 1)[-1])
    return sorted(found.values(), key=lambda s: (s.vtable, s.offset))


def _where(cfg: Config, cls: str | None, ident: str) -> list[str]:
    """'path:line' of declarations in a class body named `cls` and definitions `cls::ident(`."""
    if not cls:
        return []
    out = []
    decl = re.compile(rf"\b{re.escape(ident)}\s*\(")
    define = re.compile(rf"\b{re.escape(cls)}::{re.escape(ident)}\s*\(")
    opening = re.compile(rf"^\s*(?:class|struct)\s+{re.escape(cls)}\b[^;]*$")
    for root in (cfg.root / "include", cfg.root / "src"):
        for f in sorted(root.rglob("*")) if root.is_dir() else []:
            if f.suffix not in SOURCE_SUFFIXES or not f.is_file():
                continue
            text = f.read_text(errors="replace")
            if ident not in text:
                continue
            inside = 0
            for n, line in enumerate(text.split("\n"), 1):
                if opening.match(line):
                    inside = 1
                elif inside:
                    if decl.search(line):
                        out.append(f"{cfg.rel(f)}:{n}")
                    if line.startswith("};"):
                        inside = 0
                if define.search(line):
                    out.append(f"{cfg.rel(f)}:{n} (definition)")
    return out


def _ident(mangled: str) -> str:
    return re.sub(r"^(?:@\d+@)+", "", mangled).split("__", 1)[0] or mangled


def run(cfg: Config, args: argparse.Namespace) -> int:
    from .matching import score
    from .reorder import compile_variant, verdict
    names_at = {s.address: s.name for s in read_symbols(cfg.root / cfg.platform_options["symbols"]) if s.type == "function"}
    try:
        units = [u for u in load_objdiff_units(cfg) if u.get("base_path")]
    except MatchError as e:
        print(f"error: {e}")
        return USAGE
    unit_of = {u.get("metadata", {}).get("source_path", ""): u for u in units}
    uses: list[StaleUse] = []
    for root in (cfg.root / "src", cfg.root / "include"):
        for f in sorted(root.rglob("*")) if root.is_dir() else []:
            if f.suffix in SOURCE_SUFFIXES and f.is_file():
                uses += stale_uses(cfg.rel(f), f.read_text(errors="replace"), names_at)
    print(f"C link names symbols.txt has renamed: {len(uses)}")
    for u in uses:
        where = ",".join(str(n) for n in u.decl_lines[:3])
        print(f"  {u.path}:{where} {u.name} -> {u.new}: {u.action}")
    fixed, refused = [], []
    if args.fix:
        work = cfg.root / "build" / "dk" / "stale" / str(os.getpid())
        try:
            by_file: dict[str, list[StaleUse]] = defaultdict(list)
            for u in uses:
                by_file[u.path].append(u)
            for path, items in sorted(by_file.items()):
                unit = unit_of.get(path)
                if git(cfg.root, "status", "--porcelain", "--", path).stdout.strip():
                    refused.append(f"{path}: has uncommitted edits (left for whoever is editing it)")
                    continue
                text = (cfg.root / path).read_text()
                new, applied = text, []
                for u in items:
                    step = fix_text(new, u)
                    if step is not None and step != new:
                        new, applied = step, applied + [u.name]
                if new == text:
                    continue
                if unit is None:
                    refused.append(f"{path}: no unit compiles it")
                    continue
                a, out_a = compile_variant(cfg, unit, text, work / "a")
                b, out_b = compile_variant(cfg, unit, new, work / "b")
                if a is None or b is None:
                    refused.append(f"{path}: " + ("does not compile as it is" if a is None else
                                                  "the fix does not compile: " + (out_b.splitlines() or [""])[-1][:120]))
                    continue
                before, after = (score(cfg, unit["name"], cfg.root / unit["target_path"], o) for o in (a, b))
                why = verdict(before, after)
                if why:
                    refused.append(f"{path}: {why[0]}")
                    continue
                if (cfg.root / path).read_text() != text:
                    refused.append(f"{path}: changed meanwhile")
                    continue
                (cfg.root / path).write_text(new)
                gained = sum(f.percent == 100.0 for f in after.functions) - sum(f.percent == 100.0 for f in before.functions)
                fixed.append(f"{path}: {', '.join(applied)}" + (f" (+{gained} exact)" if gained else ""))
        finally:
            shutil.rmtree(work, ignore_errors=True)
        for line in fixed:
            print(f"  fixed {line}")
        for line in refused:
            print(f"  left {line}")
    slots = slot_names(cfg)
    same = [x for x in slots if _class_of(x.target) and _class_of(x.target) == _class_of(x.ours)]
    print(f"vtable slots where one class has two names (last build's objects; ./dk check --all refreshes): {len(same)}")
    for x in same:
        cls, ours, theirs = _class_of(x.ours), _ident(x.ours), _ident(x.target)
        mine, defs = _where(cfg, cls, ours), _where(cfg, cls, theirs)
        print(f"  {x.vtable}+0x{x.offset:X}: symbols.txt {x.target}, ours {x.ours}  [{', '.join(x.units[:4])}]")
        print(f"      ours declared at {', '.join(mine[:3]) or '?'}; symbols.txt's name in {', '.join(defs[:3]) or 'no source'}")
        if x.target.split("__", 1)[1] == x.ours.split("__", 1)[1]:
            print(f"      if the header's slot name is right: ./dk rename {x.target} {x.ours}"
                  + (" and rename those uses with it" if defs else ""))
        else:
            print(f"      the parameters differ too ({x.target.split('__', 1)[1]} vs {x.ours.split('__', 1)[1]}): settle them first")
    other = len(slots) - len(same)
    if other:
        where = sorted({u for x in slots if x not in same for u in x.units})
        print(f"  and {other} slot(s) pointing at another class's function or shifted (missing or misplaced "
              f"overrides; ./dk check shows them): {', '.join(where[:8])}{' …' if len(where) > 8 else ''}")
    from .matching import undefined_at_link
    pairs: dict[tuple[str, str], list[str]] = defaultdict(list)       # (placeholder, name) -> units
    for u in units:
        base, target = cfg.root / u["base_path"], cfg.root / u.get("target_path", "")
        if base.is_file() and target.is_file():
            t, b = Elf.parse(target.read_bytes()), Elf.parse(base.read_bytes())
            shifted = {p.split("+", 1)[0] for p in vtable_problems(t, b)}
            # exact=set(): only vtables and other data count as evidence (call sites drift in non-exact code)
            for note in undefined_at_link(cfg, t, b, set(), shifted):
                if ": rename " in note:
                    old, new = note.split(": rename ", 1)[1].split(" to ", 1)
                    pairs[(old, new)].append(u["name"].split("/", 1)[-1])
    olds, news = defaultdict(set), defaultdict(set)
    for old, new in pairs:
        olds[old].add(new)
        news[new].add(old)
    good = sorted(p for p in pairs if len(olds[p[0]]) == 1 and len(news[p[1]]) == 1)
    total = len(good)
    print(f"names our build needs that symbols.txt lacks, with the placeholder the original has in that vtable slot: "
          f"{total}" + (f" ({len(pairs) - total} more conflict between units: a misplaced override somewhere)" if len(pairs) > total else ""))
    for old, new in good:
        print(f"  ./dk rename {old} {new}  [{', '.join(sorted(set(pairs[(old, new)]))[:3])}]")
    hint("check --all --near", "rename OLD NEW")
    return OK if not uses and not same and not total else MISMATCH


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("--fix", action="store_true",
                   help="fix stale C names in sources where it is mechanical, each unit verified privately")


COMMAND = Command("stale", "analysis", "list (and --fix) source names that symbols.txt has since changed", run, _configure)
