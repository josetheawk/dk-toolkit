"""dk regs <unit> <function>: which variable sits in which register, ours vs the target, for a
near miss that differs only in register numbers, and the declaration order that fixes it.

How it works: the unit is compiled privately with `-sym on`, which adds DWARF 1 debug info and
leaves the code byte-identical, so the debug info names the register of every parameter and
local. The target register of each of our registers comes from objdiff's instruction alignment.

The allocation rule it applies (callee-saved r14-r31 and f14-f31; evidence in the playbook):
values get registers one at a time, in rank order, and each takes the lowest-numbered register
already handed out that no interfering value holds, else a new one below (r31 first). Rank:
compiler temporaries (newest first; a local or parameter merged with one takes its rank), then
named locals in declaration order, then parameters, last one first.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ...core.command import BUILD_ERROR, MISMATCH, OK, USAGE, Command, hint
from ...core.config import Config
from .elf import Elf
from .matching import MatchError, compile_command, resolve_unit, retarget, run_compile

TAG_PARAM, TAG_LOCAL, TAG_BLOCK = 0x05, 0x0C, 0x0B
TAG_SUBROUTINES = (0x06, 0x14)          # global_subroutine, subroutine
AT_SIBLING, AT_LOCATION, AT_NAME, AT_MW_MANGLED = 0x001, 0x002, 0x003, 0x200
OP_REG = 0x01
REG = re.compile(r"[rf]\d+")


@dataclass
class Var:
    kind: str           # "param" or "local"
    name: str
    reg: str | None     # "r30", "f31", "stack" or None (optimized away)
    depth: int = 0      # lexical block nesting (0 = function body)


@dataclass
class DwarfFunction:
    name: str
    mangled: str
    vars: list[Var] = field(default_factory=list)


def _attrs(d: bytes, p: int, end: int, refs: dict[int, int]) -> dict[int, object]:
    at: dict[int, object] = {}
    while p + 2 <= end:
        an, = struct.unpack_from(">H", d, p)
        p += 2
        form = an & 0xF
        if form in (1, 2, 6):
            v, = struct.unpack_from(">I", d, p)
            v = refs.get(p, v)
            p += 4
        elif form == 5:
            v, = struct.unpack_from(">H", d, p)
            p += 2
        elif form == 7:
            v, p = d[p:p + 8], p + 8
        elif form == 3:
            n, = struct.unpack_from(">H", d, p)
            v, p = d[p + 2:p + 2 + n], p + 2 + n
        elif form == 4:
            n, = struct.unpack_from(">I", d, p)
            v, p = d[p + 4:p + 4 + n], p + 4 + n
        elif form == 8:
            z = d.index(b"\0", p)
            v, p = d[p:z].decode("latin-1"), z + 1
        else:
            break
        at[an >> 4] = v
    return at


def _location(loc: object) -> str | None:
    if not isinstance(loc, bytes) or not loc:
        return None
    if loc[0] == OP_REG and len(loc) == 5:
        n, = struct.unpack(">I", loc[1:])
        return f"r{n}" if n < 32 else f"f{n - 32}"
    return "stack"


def parse_dwarf1(data: bytes, refs: dict[int, int] | None = None) -> list[DwarfFunction]:
    """Functions with their parameters and locals (declaration order) from a `.debug` section.
    `refs`: section offset -> value of relocated words (sibling references point into .debug
    through relocations in an object file)."""
    refs = refs or {}
    out: list[DwarfFunction] = []
    stack: list[tuple[int, str]] = []        # (sibling offset, kind) of open DIEs with children
    off = 0
    while off + 4 <= len(data):
        while stack and off >= stack[-1][0]:
            stack.pop()
        ln, = struct.unpack_from(">I", data, off)
        if ln < 8:
            off += max(ln, 4)
            continue
        tag, = struct.unpack_from(">H", data, off + 4)
        end = off + ln
        at = _attrs(data, off + 6, end, refs)
        if tag in TAG_SUBROUTINES:
            name = str(at.get(AT_NAME, ""))
            out.append(DwarfFunction(name, str(at.get(AT_MW_MANGLED, name))))
        elif tag in (TAG_PARAM, TAG_LOCAL) and out and any(k == "fn" for _, k in stack):
            depth = sum(1 for _, k in stack if k == "block")
            out[-1].vars.append(Var("param" if tag == TAG_PARAM else "local", str(at.get(AT_NAME, "?")),
                                    _location(at.get(AT_LOCATION)), depth))
        sib = at.get(AT_SIBLING)
        if isinstance(sib, int) and sib > end:
            stack.append((sib, "fn" if tag in TAG_SUBROUTINES else "block" if tag == TAG_BLOCK else "other"))
        off = end
    return out


def debug_refs(elf: Elf) -> dict[int, int]:
    """Relocated words of `.debug` that point back into it (DIE references)."""
    sec = elf.section(".debug")
    if sec is None:
        return {}
    return {r.offset: r.target_value + r.addend for r in elf.relocations(sec.index) if r.target_section == ".debug"}


def find_function(funcs: list[DwarfFunction], symbol: str) -> DwarfFunction | None:
    for f in funcs:
        if f.mangled == symbol:
            return f
    plain = symbol.split("__")[0] if not symbol.startswith("__") else symbol
    hits = [f for f in funcs if f.name == plain]
    return hits[0] if len(hits) == 1 else None


def _regs(entry: dict) -> list[str]:
    parts = entry.get("instruction", {}).get("parts", [])
    return [a["arg"]["opaque"] for a in parts
            if "arg" in a and isinstance(a["arg"], dict) and REG.fullmatch(str(a["arg"].get("opaque", "")))]


def _mnemonic(entry: dict) -> str | None:
    for a in entry.get("instruction", {}).get("parts", []):
        if "opcode" in a:
            return a["opcode"].get("mnemonic")
    return None


def register_map(target: list[dict], ours: list[dict]) -> dict[str, Counter]:
    """For each of our registers, how often each target register sits at the same operand of
    the same aligned instruction (objdiff pads both sides to one length)."""
    pairs: dict[str, Counter] = defaultdict(Counter)
    for t, o in zip(target, ours):
        if "instruction" not in t or "instruction" not in o or _mnemonic(t) != _mnemonic(o):
            continue
        a, b = _regs(t), _regs(o)
        if len(a) != len(b):
            continue
        for x, y in zip(b, a):
            pairs[x][y] += 1
    return pairs


def callee_saved(reg: str) -> bool:
    return bool(REG.fullmatch(reg)) and int(reg[1:]) >= 14


def _num(reg: str) -> int:
    return int(reg[1:])


@dataclass
class Row:
    ours: str
    target: str                  # dominant target register
    names: list[str]             # variables our debug info puts in this register ("" = temporary)
    other: list[tuple[str, int]]  # other target registers seen for it (several values share ours)


def rows(fn: DwarfFunction | None, pairs: dict[str, Counter], cls: str, saved: bool = True) -> list[Row]:
    """Callee-saved (or, saved=False, volatile) registers of one class ("r" or "f"), ours,
    highest first. r1, r2 and r13 (stack and small-data bases) are left out."""
    named: dict[str, list[str]] = defaultdict(list)
    for v in (fn.vars if fn else []):
        if v.reg and v.reg[0] == cls:
            named[v.reg].append(f"{v.kind} {v.name}")
    out = []
    keep = lambda r: r[0] == cls and callee_saved(r) == saved and r not in ("r1", "r2", "r13")
    for reg in sorted((r for r in pairs if keep(r)), key=_num, reverse=True):
        (top, _), *rest = pairs[reg].most_common()
        out.append(Row(reg, top, named.get(reg, []), rest))
    return out


def wanted_order(fn: DwarfFunction | None, table: list[Row]) -> tuple[list[str], list[str], dict[str, int]]:
    """(declaration order, target order, target register) of the named locals whose relative
    order differs between ours and the target (same register class, declared once)."""
    decl = [v.name for v in (fn.vars if fn else []) if v.kind == "local"]
    target_of: dict[str, int] = {}
    ours_of: dict[str, int] = {}
    for r in table:
        for n in r.names:
            kind, name = n.split(" ", 1)
            if kind == "local" and decl.count(name) == 1 and not r.other:
                bias = 100 if r.target[0] == "f" else 0
                target_of[name] = _num(r.target) + bias
                ours_of[name] = _num(r.ours) + bias
    names = [n for n in decl if n in target_of]
    same_class = lambda a, b: (target_of[a] >= 100) == (target_of[b] >= 100)
    involved = {z for x in names for y in names if x != y and same_class(x, y)
                and ours_of[x] > ours_of[y] and target_of[x] < target_of[y] for z in (x, y)}
    current = [n for n in names if n in involved]
    want = sorted(current, key=lambda n: (-target_of[n], current.index(n)))
    return current, want, target_of


def advice(fn: DwarfFunction | None, table: list[Row]) -> list[str]:
    """The source changes the rank rule implies, most direct first."""
    if not table or all(r.ours == r.target and not r.other for r in table):
        return []
    lines = []
    decl = [v.name for v in (fn.vars if fn else []) if v.kind == "local"]
    current, want, target_of = wanted_order(fn, table)
    if want != current:
        lines.append("declare these locals in this order (first gets the highest register): " + ", ".join(want)
                     + "  (`--try` moves the declarations in a private copy and compiles each variant)")
    gprs = [r for r in table if r.ours[0] == "r"]
    params = [r for r in gprs if any(n.startswith("param ") for n in r.names)]
    locs = [r for r in gprs if any(n.startswith("local ") for n in r.names)]
    if params and locs:
        if max(_num(r.target) for r in params) > min(_num(r.target) for r in locs) and \
                max(_num(r.ours) for r in params) < min(_num(r.ours) for r in locs):
            lines.append("the target allocates a parameter before some locals: parameters normally come last, so in "
                         "the target that parameter merged with a temporary (a strength-reduced loop pointer, a copy "
                         "an inline function made), or those locals are not plain locals there")
    moved = [r for r in table if not r.names and r.ours != r.target]
    if moved:
        lines.append("unnamed registers are compiler temporaries (an expression value kept across a call, a CSE or "
                     "loop-invariant address, an inline function's value, or a local merged with one); they rank "
                     "before named locals, newest first: " + ", ".join(f"ours {r.ours} -> target {r.target}" for r in moved))
    ours_of = {}
    for r in table:
        if r.ours == r.target and not r.other:
            continue
        for n in r.names:
            kind, name = n.split(" ", 1)
            if kind == "local" and decl.count(name) == 1:
                ours_of[name] = (r.ours[0], _num(r.ours))
    order = [n for n in decl if n in ours_of]
    for i, n in enumerate(order):
        cls, reg = ours_of[n]
        earlier = [m for m in order[:i] if ours_of[m][0] == cls and ours_of[m][1] < reg]
        if earlier and n not in current:
            lines.append(f"ours ranks `{n}` above locals declared before it ({', '.join(earlier)}): it merged with a "
                         "temporary (its value feeds an operation, a compare or a store straight from the call or "
                         "expression that made it; a local only passed on as an argument or returned keeps its "
                         "declaration rank)")
    return lines


# ---- source rewrite for --try ------------------------------------------------------------------

DECL = re.compile(r"^(?P<indent>\s*)(?P<type>(?:const\s+|unsigned\s+|signed\s+|struct\s+|class\s+)*[A-Za-z_][\w:<>]*"
                  r"(?:\s*\*+\s*|\s+))(?P<name>[A-Za-z_]\w*)\s*(?:=\s*(?P<init>[^;]*))?;\s*(?P<tail>//.*)?$")
FOR_DECL = re.compile(r"for\s*\(\s*(?P<type>(?:unsigned\s+|signed\s+)*[A-Za-z_][\w:<>]*(?:\s*\*+\s*|\s+))(?P<name>[A-Za-z_]\w*)\s*=")


def body_lines(text: str, symbol: str) -> tuple[int, int] | None:
    """(line of the opening brace, line of the closing brace) of the function's definition."""
    plain = symbol.split("__")[0] if "__" in symbol.lstrip("_") else symbol
    lines = text.splitlines()
    pat = re.compile(r"\b" + re.escape(plain) + r"\s*\(")
    for i, line in enumerate(lines):
        if not pat.search(line) or line.rstrip().endswith(";"):
            continue
        depth, start = 0, None
        for j in range(i, len(lines)):
            if start is None and ";" in lines[j] and "{" not in lines[j]:
                break
            for ch in lines[j]:
                if ch == "{":
                    depth += 1
                    if start is None:
                        start = j
                elif ch == "}":
                    depth -= 1
                    if start is not None and depth == 0:
                        return start, j
    return None


def _declarations(lines: list[str], lo: int, hi: int, names: list[str]) -> dict[str, tuple[int, str, str | None]] | None:
    """name -> (line, type, initializer or None or "for") for simple single-name declarations."""
    found: dict[str, tuple[int, str, str | None]] = {}
    for i in range(lo + 1, hi):
        m = DECL.match(lines[i])
        if m and m.group("name") in names and m.group("type").strip() not in ("return", "delete", "goto", "else"):
            if m.group("name") in found:
                return None
            found[m.group("name")] = (i, m.group("type").strip(), m.group("init"))
            continue
        m = FOR_DECL.search(lines[i])
        if m and m.group("name") in names:
            if m.group("name") in found:
                return None
            found[m.group("name")] = (i, m.group("type").strip(), "for")
    return found if set(found) == set(names) else None


def _strip_declaration(line: str, name: str, init: str | None) -> str | None:
    """The statement that stays where a moved declaration was (None: nothing stays)."""
    if init == "for":
        return FOR_DECL.sub(lambda m: m.group(0).replace(m.group("type"), "", 1), line, count=1)
    m = DECL.match(line)
    tail = f" {m.group('tail')}" if m.group("tail") else ""
    return f"{m.group('indent')}{name} = {init.strip()};{tail}" if init is not None else None


def reorder(text: str, symbol: str, order: list[str], minimal: bool = True) -> str | None:
    """Make `order` the declaration order of those locals. minimal: move only the declarations
    that must come earlier, each to just before the one it has to precede; otherwise declare all of
    them at the top of the function. A moved declaration with an initializer leaves an assignment
    behind. None when a declaration is not a simple single-name statement."""
    span = body_lines(text, symbol)
    if span is None:
        return None
    lo, hi = span
    lines = text.splitlines()
    found = _declarations(lines, lo, hi, order)
    if found is None:
        return None
    indent_of = lambda i: re.match(r"\s*", lines[i]).group(0)
    inserts: dict[int, list[str]] = defaultdict(list)     # line -> declarations to put before it
    moved: set[str] = set()
    if minimal:
        pos = {n: float(found[n][0]) for n in order}
        anchor = {n: found[n][0] for n in order}
        for k in range(len(order) - 2, -1, -1):
            x, nxt = order[k], order[k + 1]
            if pos[x] < pos[nxt]:
                continue
            moved.add(x)
            pos[x] = pos[nxt] - 0.5 ** (len(order) - k)
            anchor[x] = anchor[nxt]
        for n in order:                    # keep the wanted order among moves onto one anchor
            if n in moved:
                inserts[anchor[n]].append(n)
    else:
        moved = set(order)
        inserts[lo + 1] = list(order)
    if not moved:
        return None
    out: list[str | None] = list(lines)
    for n in moved:
        i, _typ, init = found[n]
        out[i] = _strip_declaration(lines[i], n, init)
    result: list[str] = []
    for i, line in enumerate(out):
        if i in inserts:
            ind = indent_of(i) if i < hi else "    "
            result += [f"{ind}{found[n][1]} {n};" for n in inserts[i]]
        if line is not None:
            result.append(line)
    return "\n".join(result) + "\n"


def tie_orders(order: list[str], target: dict[str, int], limit: int = 6) -> list[list[str]]:
    """`order` plus variants that permute locals sharing a target register (the target does not fix
    their relative order)."""
    import itertools
    groups: list[list[str]] = []
    for n in order:
        if groups and target[groups[-1][0]] == target[n]:
            groups[-1].append(n)
        else:
            groups.append([n])
    out = [order]
    for gi, g in enumerate(groups):
        if len(g) < 2:
            continue
        for perm in itertools.permutations(g):
            cand = [x for h in groups[:gi] for x in h] + list(perm) + [x for h in groups[gi + 1:] for x in h]
            if cand not in out:
                out.append(cand)
            if len(out) >= limit:
                return out
    return out


# ---- command -----------------------------------------------------------------------------------

def _compile(cfg: Config, unit: dict, source: Path | None, work: Path) -> tuple[Path | None, str]:
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    argv = retarget(compile_command(cfg, unit["base_path"], locked=False), work, extra_flags=("-sym", "on"))
    if source is not None:
        argv[argv.index("-c") + 1] = str(source)
    ok, out = run_compile(cfg, argv)
    objs = list(work.glob("*.o"))
    return (objs[0] if ok and objs else None), out


def _diff(cfg: Config, unit: dict, obj: Path, symbol: str) -> tuple[dict, dict] | None:
    proc = subprocess.run([str(cfg.root / "build" / "tools" / "objdiff-cli"), "diff", "-1", unit["target_path"],
                           "-2", str(obj), "-o", "-", "--format", "json", "-c", "functionRelocDiffs=data_value"],
                          cwd=cfg.root, capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    doc = json.loads(proc.stdout)
    left = next((s for s in doc["left"].get("symbols", []) if s["name"] == symbol and "instructions" in s), None)
    right = next((s for s in doc.get("right", {}).get("symbols", []) if s["name"] == symbol and "instructions" in s), None)
    return (left, right) if left and right else None


def analyse(cfg: Config, unit: dict, symbol: str, source: Path | None, work: Path):
    obj, out = _compile(cfg, unit, source, work)
    if obj is None:
        return None, out
    pair = _diff(cfg, unit, obj, symbol)
    if pair is None:
        return None, f"{symbol} is not in both the target and our object"
    elf = Elf.parse(obj.read_bytes())
    dbg = elf.section(".debug")
    fn = find_function(parse_dwarf1(dbg.data, debug_refs(elf)), symbol) if dbg else None
    return (float(pair[0].get("match_percent", 0.0)), fn, register_map(pair[0]["instructions"], pair[1]["instructions"])), ""


def _print(percent: float, fn: DwarfFunction | None, pairs: dict[str, Counter], symbol: str) -> list[Row]:
    print(f"{symbol}  {percent:.2f}%")
    every = []
    for cls, label in (("r", "GPR"), ("f", "FPR")):
        table = rows(fn, pairs, cls)
        if not table:
            continue
        every += table
        print(f"  callee-saved {label}s (ours, highest first):")
        for r in table:
            mark = "  " if r.ours == r.target and not r.other else "!="
            extra = ("  also " + ", ".join(f"{t}x{n}" for t, n in r.other)) if r.other else ""
            print(f"    {mark} ours {r.ours:<4} target {r.target:<4} {', '.join(r.names) or '(temporary)'}{extra}")
    vol = sorted((o, t) for o, c in pairs.items() for t in c if not callee_saved(o) and o != t and o not in ("r1", "r2", "r13"))
    if vol:
        print("  volatile registers differ too (not covered by the rank rule): "
              + ", ".join(f"{o}->{t}" for o, t in vol[:12]) + (" …" if len(vol) > 12 else ""))
    if fn is None:
        print("  (no debug info for this function: names unknown)")
    return every


def _candidates(text: str, symbol: str, fn: DwarfFunction, table: list[Row], volatile: list[Row] = ()):
    """(label, source) variants to compile: the target's order for inverted locals first (ties
    permuted, minimal moves then hoisting), then every single move of one register local's
    declaration to just before another declared earlier."""
    current, want, target_of = wanted_order(fn, table)
    seen = {text}
    if want != current:
        for order in tie_orders(want, target_of):
            for minimal in (True, False):
                new = reorder(text, symbol, order, minimal)
                if new is not None and new not in seen:
                    seen.add(new)
                    yield f"{'moved' if minimal else 'hoisted'} {', '.join(order)}", new
    decl = [v.name for v in fn.vars if v.kind == "local"]
    every = list(table) + list(volatile)       # declaration order also moves volatile registers
    in_regs = {n.split(" ", 1)[1] for r in every for n in r.names if n.startswith("local ")}
    differing = {n.split(" ", 1)[1] for r in every if r.ours != r.target or r.other for n in r.names}
    names = [n for n in decl if n in in_regs and decl.count(n) == 1]
    pairs = [(x, y) for i, x in enumerate(names) for y in names[:i]]
    pairs.sort(key=lambda p: (p[0] not in differing and p[1] not in differing, p[0] not in differing))
    for x, y in pairs:
        order = [x, y]
        new = reorder(text, symbol, order, True)
        if new is not None and new not in seen:
            seen.add(new)
            yield f"moved {x} before {y}", new


def _try(cfg: Config, unit: dict, args: argparse.Namespace, source: Path | None, work: Path,
         fn: DwarfFunction, table: list[Row], pairs: dict[str, Counter], percent: float) -> None:
    src_path = source or cfg.root / unit["metadata"]["source_path"]
    text = src_path.read_text()
    out_dir = cfg.root / "build" / "dk" / "regs"
    out_dir.mkdir(parents=True, exist_ok=True)
    cand = out_dir / f"{re.sub(r'[^A-Za-z0-9_]', '_', args.function)[:60]}_{src_path.name}"
    best: tuple[float, str, str] | None = None
    tried = 0
    volatile = [r for cls in ("r", "f") for r in rows(fn, pairs, cls, saved=False)]
    for label, new in _candidates(text, args.function, fn, table, volatile):
        if tried >= args.limit:
            print(f"  --try: stopped after {tried} variants (--limit)")
            break
        tried += 1
        cand.write_text(new)
        res, _err = analyse(cfg, unit, args.function, cand, work)
        if res is None:
            continue
        if res[0] > percent:
            print(f"  --try: {label}: {res[0]:.2f}%")
        if best is None or res[0] > best[0]:
            best = (res[0], new, label)
        if res[0] == 100.0:
            break
    if best is None or best[0] <= percent:
        cand.unlink(missing_ok=True)
        print(f"  --try: none of {tried} declaration-order variants beat {percent:.2f}%")
        return
    cand.write_text(best[1])
    print(f"  --try: best {best[0]:.2f}% (was {percent:.2f}%) with {best[2]}; written to {cfg.rel(cand)}")
    if best[0] == 100.0 and args.apply and source is None:
        src_path.write_text(best[1])
        print(f"  applied to {cfg.rel(src_path)}")


def run(cfg: Config, args: argparse.Namespace) -> int:
    try:
        unit = resolve_unit(cfg, args.unit)
    except MatchError as e:
        print(f"error: {e}")
        return USAGE
    source = Path(args.source).resolve() if args.source else None
    work = cfg.root / "build" / "dk" / "regs" / str(os.getpid())
    try:
        res, err = analyse(cfg, unit, args.function, source, work)
        if res is None:
            print(f"BUILD ERROR or missing function:\n{err[-2000:]}")
            return BUILD_ERROR
        percent, fn, pairs = res
        table = _print(percent, fn, pairs, args.function)
        if fn is not None:
            hidden = [f"{v.name} ({v.reg})" for v in fn.vars if v.reg and v.reg != "stack" and not callee_saved(v.reg)]
            if hidden and any(not r.names for r in table):
                print("  debug info puts these in volatile registers; an unnamed row may be one of them: " + ", ".join(hidden))
        tips = advice(fn, table)
        for t in tips:
            print("  - " + t)
        if percent == 100.0:
            print("MATCH")
            return OK
        if args.try_ and fn is not None:
            _try(cfg, unit, args, source, work, fn, table, pairs, percent)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    hint(f"diff {args.unit} --function {args.function}", f"check {args.unit} --private")
    return MISMATCH


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("unit", help="unit (source path or name)")
    p.add_argument("function", help="function symbol, as dk check lists it")
    p.add_argument("--source", help="compile this file instead of the unit's source (a private copy)")
    p.add_argument("--try", dest="try_", action="store_true",
                   help="compile declaration-order variants of the locals in a private copy (target order first, then single moves)")
    p.add_argument("--apply", action="store_true", help="with --try: write the copy into the unit's source if exact")
    p.add_argument("--limit", type=int, default=40, help="with --try: most variants to compile (default 40)")


COMMAND = Command("regs", "matching", "per-variable registers ours vs target (register-numbering near misses)", run, _configure)
