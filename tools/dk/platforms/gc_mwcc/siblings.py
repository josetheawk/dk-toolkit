"""Siblings: exact functions whose target code shares an instruction pattern with an open function.

An open function often repeats a stretch the compiler already produced for another, exact function
(the same file walk, the same container idiom). The exact one's source shows the shape to copy.

Normalisation (per instruction): the mnemonic; registers abstracted (r1, r2 and r13 kept: stack and
small-data bases); immediates and memory offsets kept (they carry the structure: a 0x7c stride, a
+0xc header) except stack offsets (frame layout); symbols and labels masked. A window of N instructions renames its registers by first
use, so `add rA, r0, rB; addi rA, rA, 0xc` keeps its data flow and nothing else.

Scoring: 2- and 3-instruction windows weighted by inverse document frequency over every function of
the game (common prologue and epilogue shapes weigh ~0). Hits are chained into spans along the
diagonal (a few instructions of scheduling slack); a sibling's score is its best span, plus a small
share of the rest. With our diff, a span covering an offset where ours differs counts double.

The constructor detector: `p = alloc(...); if (p) p = fn(p, ...)` in the target (a call, `mr. rP, r3`
or a compare, a branch over one more call whose result goes back to rP) where fn returns its first
argument (or is named as a constructor): the source is a new-expression `new (...) T(...)`, and fn a
constructor; the C form compiles to the same instructions with one value fewer (other registers).
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field

from . import asmflow

KEEP_REGS = {"r1", "r2", "r13"}
_GPR = re.compile(r"^r([0-9]|[12][0-9]|3[01])$")
_FPR = re.compile(r"^f([0-9]|[12][0-9]|3[01])$")
_CR = re.compile(r"^cr[0-7]$")
_MEM = re.compile(r"^(?P<off>.+)\((?P<base>r\d+)\)$")
_SYM = re.compile(r"@(ha|h|l|sda21|sda2)\b")
_NUM = re.compile(r"^-?(0x[0-9a-fA-F]+|\d+)$")
GAP = 3                            # pairs: instructions apart at most
SLACK = 3                          # instructions of scheduling slack inside one span
MAX_SHOWN = 24                     # instruction rows shown per span
MAX_DF_SHARE = 0.05                # windows found in more functions than this share are skipped


# ---------------------------------------------------------------- normalisation

@dataclass
class Tok:
    shape: str                     # mnemonic and operands, registers as {}
    regs: tuple[str, ...]          # the abstracted registers, in operand order
    loose: str                     # the shape with numbers masked too


def _operand(op: str, regs: list[str]) -> tuple[str, str]:
    m = _MEM.match(op)
    if m:
        off, _ = _operand(m["off"], regs)
        base = m["base"]
        if _SYM.search(m["off"]):
            return "S", "S"
        if base == "r1":                         # stack slots: frame layout, not structure
            return "N(r1)", "N(r1)"
        if base in KEEP_REGS:
            return f"{off}({base})", f"N({base})"
        regs.append(base)
        return f"{off}({{}})", "N({})"
    if _SYM.search(op):
        return "S", "S"
    if op.startswith(".L") or op.startswith('".L'):
        return "L", "L"
    if op in KEEP_REGS:
        return op, op
    if _GPR.match(op) or _FPR.match(op) or _CR.match(op):
        regs.append(op)
        return "{}", "{}"
    if _NUM.match(op):
        return op.lower(), "N"
    return "S", "S"                # call targets and other symbols


def _is_jump(mnem: str) -> bool:
    return mnem.startswith("b") and mnem not in ("bl", "blr", "bctr", "bctrl", "blrl") \
        and not mnem.endswith(("lr", "ctr", "lrl", "ctrl"))


def tokenize(insns: list[asmflow.Insn]) -> list[Tok]:
    out = []
    for ins in insns:
        regs: list[str] = []
        parts = [_operand(o, regs) for o in ins.ops]
        if ins.mnem in ("addi", "subi") and "r1" in ins.ops[1:2]:
            parts[-1] = ("N", "N")             # an address in the frame
        if _is_jump(ins.mnem) and parts and parts[-1][0] not in ("L", "{}"):
            parts[-1] = ("L", "L")             # a jump written as an address (objdiff's form)
        out.append(Tok(f"{ins.mnem} " + ",".join(p[0] for p in parts),
                       tuple(regs), f"{ins.mnem} " + ",".join(p[1] for p in parts)))
    return out


def window(toks: list[Tok], idx: tuple[int, ...]) -> str:
    """The key of the instructions at idx: shapes with registers renamed by first use inside them."""
    names: dict[str, str] = {}
    parts = []
    for i in idx:
        t = toks[i]
        args = []
        for r in t.regs:
            if r not in names:
                names[r] = ("F" if r[0] == "f" else "C" if r[0] == "c" else "R") + chr(97 + len(names))
            args.append(names[r])
        parts.append(t.shape.format(*args))
    return "|".join(parts)


def grams(toks: list[Tok]) -> list[tuple[int, tuple[int, ...], str]]:
    """(start, instruction indexes, key): contiguous triples, and pairs up to GAP apart (the
    compiler interleaves independent instructions, so a pair keeps a data flow across them)."""
    out = []
    n = len(toks)
    for i in range(n):
        if i + 2 < n:
            idx = (i, i + 1, i + 2)
            out.append((i, idx, "3:" + window(toks, idx)))
        for k in range(1, GAP + 1):
            if i + k < n:
                idx = (i, i + k)
                out.append((i, idx, "2:" + window(toks, idx)))
    return out


# ---------------------------------------------------------------- index and spans

@dataclass
class Span:
    q_lo: int
    q_hi: int                      # instruction indexes, inclusive
    c_lo: int
    c_hi: int
    weight: float
    diverges: bool = False
    contrast: float = 0.0          # the part of the weight our code lacks (with our asm)


@dataclass
class Hit:
    function: str
    score: float
    best: Span
    spans: list[Span] = field(default_factory=list)


class Index:
    """Every function's windows; postings only for the candidate pool (exact functions)."""

    def __init__(self, texts: dict[str, str], pool: set[str]):
        self.insns: dict[str, list[asmflow.Insn]] = {}
        self.toks: dict[str, list[Tok]] = {}
        df: dict[str, int] = defaultdict(int)
        self.postings: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
        for name, text in texts.items():
            insns = asmflow.parse_insns(text)
            toks = tokenize(insns)
            self.insns[name], self.toks[name] = insns, toks
            seen = set()
            for i, idx, key in grams(toks):
                if key not in seen:
                    seen.add(key)
                    df[key] += 1
                if name in pool:
                    self.postings[key].append((name, i, idx))
        self.n = max(1, len(texts))
        cap = max(2, int(self.n * MAX_DF_SHARE))
        self.idf = {k: math.log(self.n / d) for k, d in df.items() if d <= cap}

    def query(self, name: str, exclude: set[str] = frozenset(), diverging: set[int] | None = None,
              top: int = 5, ours: set[str] | None = None) -> list[Hit]:
        """The best exact siblings of `name` (instruction indexes of ours-differs in `diverging`;
        `ours`: the windows of our compiled code, to weigh what the sibling shares with the target only)."""
        toks = self.toks[name]
        per: dict[str, list[tuple[int, int, tuple, float]]] = defaultdict(list)
        for i, idx, key in grams(toks):
            w = self.idf.get(key)
            if not w:
                continue
            for other, j, _ in self.postings.get(key, ()):
                if other != name and other not in exclude:
                    per[other].append((i, j, idx, w, None if ours is None else key not in ours))
        hits = [self._spans(other, h, diverging) for other, h in per.items()]
        hits.sort(key=lambda h: (-h.score, h.function))
        return hits[:top]

    @staticmethod
    def _spans(other: str, hits: list[tuple], diverging: set[int] | None) -> Hit:
        hits.sort()
        spans: list[Span] = []
        covered: list[set[int]] = []
        active: list[int] = []            # spans that can still grow (hits come in query order)
        for i, j, idx, w, lacking in hits:
            n = len(idx)
            last = idx[-1] - i
            active = [k for k in active if spans[k].q_hi + SLACK >= i]
            for s, cov in ((spans[k], covered[k]) for k in active):
                if s.q_lo - SLACK <= i <= s.q_hi + SLACK and abs((i - j) - (s.q_lo - s.c_lo)) <= SLACK:
                    new = set(idx) - cov
                    s.weight += w * len(new) / n
                    if lacking:
                        s.contrast += w * len(new) / n
                    cov |= new
                    s.q_hi, s.c_hi = max(s.q_hi, i + last), max(s.c_hi, j + last)
                    s.c_lo = min(s.c_lo, j)
                    break
            else:
                active.append(len(spans))
                spans.append(Span(i, i + last, j, j + last, w, contrast=w if lacking else 0.0))
                covered.append(set(idx))
        if diverging:
            for s in spans:
                if any(s.q_lo - 2 <= d <= s.q_hi + 1 for d in diverging):
                    s.diverges = True
                    s.weight *= 2
                    s.contrast *= 2
        contrast = hits[0][4] is not None
        spans.sort(key=lambda s: (-s.contrast, -s.weight))
        chosen: list[Span] = []           # distinct stretches of the query (a repeated idiom counts once)
        for s in spans:
            if all(s.q_hi < c.q_lo or s.q_lo > c.q_hi for c in chosen):
                chosen.append(s)
            if len(chosen) == 3:
                break
        score = sum((s.contrast + 0.1 * s.weight if contrast else s.weight) * f
                    for s, f in zip(chosen, (1.0, 0.5, 0.25)))
        return Hit(other, round(score, 2), chosen[0], chosen)


# ---------------------------------------------------------------- constructor detector

_CALL = {"bl", "bctrl", "blrl"}
_NO_DEST = {"cmpwi", "cmplwi", "cmpw", "cmplw", "fcmpu", "fcmpo", "mtctr", "mtlr", "mtcrf", "b", "bl", "bctr",
            "bctrl", "blr", "sync", "isync", "nop", "dcbf", "dcbst", "icbi"}
_VOLATILE = {"r0"} | {f"r{i}" for i in range(3, 13)}


def returns_first_arg(text: str) -> bool:
    """Does every return hand back the r3 the function was called with (a constructor's shape)?
    A straight-line approximation: copies of the incoming r3 are followed through `mr`; any other
    write, and volatile registers across a call, end a copy."""
    insns = asmflow.parse_insns(text)
    holders = {"r3"}
    returns = 0
    for ins in insns:
        m, ops = ins.mnem, ins.ops
        if m == "blr" or (m.endswith("lr") and m.startswith("b") and m != "bctrl"):
            returns += 1
            if "r3" not in holders:
                return False
            continue
        if m in ("bl", "bctrl", "blrl"):
            target = ops[0] if ops else ""
            if not target.startswith(("_savegpr", "_restgpr")):
                holders -= _VOLATILE
            continue
        if not ops or m in _NO_DEST or m.startswith(("st", "b")) or m.startswith(("cmp", "fcmp")):
            continue
        dest = ops[0]
        if m in ("mr", "mr.") and len(ops) == 2 and ops[1] in holders:
            holders.add(dest)
        elif m == "lmw" and _GPR.match(dest):
            holders -= {f"r{i}" for i in range(int(dest[1:]), 32)}
        else:
            holders.discard(dest)
    return returns > 0


@dataclass
class CtorSite:
    offset: int                    # of the constructor call, from the function start
    alloc: str                     # the allocating call's target ("(virtual)" for bctrl)
    ctor: str
    lines: list[str]


def _target(ins: asmflow.Insn) -> str:
    if ins.mnem == "bl":
        ref = asmflow.symref(ins.ops[0]) if ins.ops else None
        return ref[0] if ref else ins.ops[0].strip('"')
    return "(virtual)"


def ctor_sites(text: str, returns_self) -> list[CtorSite]:
    """`p = alloc(...); if (p) p = fn(p, ...)` where returns_self(fn) is true."""
    insns = asmflow.parse_insns(text)
    if not insns:
        return []
    base = insns[0].addr
    out = []
    for k, ins in enumerate(insns[:-3]):
        if ins.mnem not in _CALL:
            continue
        nxt = insns[k + 1]
        p = None
        j = k + 2
        if nxt.mnem == "mr." and nxt.ops[1:] == ("r3",):
            p = nxt.ops[0]
        elif nxt.mnem in ("cmplwi", "cmpwi") and nxt.ops[-2:] == ("r3", "0x0") and insns[j].mnem == "mr" \
                and insns[j].ops[1:] == ("r3",):
            p, j = insns[j].ops[0], j + 1
        elif nxt.mnem == "mr" and nxt.ops[1:] == ("r3",) and insns[j].mnem in ("cmplwi", "cmpwi") \
                and insns[j].ops[-2:] in ((nxt.ops[0], "0x0"), ("r3", "0x0")):
            p, j = nxt.ops[0], j + 1
        if p is None or j >= len(insns) or insns[j].mnem != "beq":
            continue
        label = insns[j].ops[-1]
        end = next((m for m in range(j + 1, len(insns)) if label in insns[m].labels), None)
        if end is None or end - j > 40:
            continue
        calls = [m for m in range(j + 1, end) if insns[m].mnem in _CALL]
        if len(calls) != 1 or insns[calls[0]].mnem != "bl":
            continue
        c = calls[0]
        if not (insns[end - 1].mnem == "mr" and insns[end - 1].ops == (p, "r3") and end - 1 == c + 1):
            continue
        r3_ok = all(not (x.ops and x.ops[0] == "r3") or (x.mnem == "mr" and x.ops[1] == p) for x in insns[j + 1:c])
        fn = _target(insns[c])
        if not r3_ok or not returns_self(fn):
            continue
        out.append(CtorSite(insns[c].addr - base, _target(ins), fn,
                            [f"{x.addr - base:#06x}  {x.text}" for x in insns[k:end]]))
    return out


def is_ctor_name(name: str) -> bool:
    return name.startswith("__ct__")


# ---------------------------------------------------------------- the command

def resolve(names, query: str) -> str | None:
    """A function by its name, or by an auto-name prefix (`fn_ADDR` of a since-renamed member)."""
    if query in names:
        return query
    hits = sorted(n for n in names if n.startswith(query + "__") or n.startswith(query))
    return hits[0] if len(hits) == 1 else None


def source_lines(cfg, unit_name: str | None, function: str, offsets: list[int]) -> list[tuple[int, str]]:
    """The source lines (number, text) of an exact function's instructions at `offsets`, from a
    private `-sym on` compile of its unit (the line table)."""
    import tempfile
    from pathlib import Path
    from .elf import Elf
    from .matching import MatchError, compile_command, load_objdiff_units, retarget, run_compile
    from .regs import line_table
    if not unit_name:
        return []
    unit = next((u for u in load_objdiff_units(cfg)
                 if u.get("metadata", {}).get("source_path", "").removeprefix("src/") == unit_name), None)
    if unit is None or not unit.get("base_path"):
        return []
    src = cfg.root / unit["metadata"]["source_path"]
    with tempfile.TemporaryDirectory(prefix="dk-siblings-") as tmp:
        try:
            argv = retarget(compile_command(cfg, unit["base_path"], locked=False), Path(tmp), extra_flags=("-sym", "on"))
        except MatchError:
            return []
        ok, _ = run_compile(cfg, argv)
        objs = list(Path(tmp).glob("*.o"))
        if not ok or not objs:
            return []
        table = line_table(Elf.parse(objs[0].read_bytes()), function)
    if not table:
        return []
    text = src.read_text(errors="replace").splitlines()
    want = []
    for off in offsets:
        line = None
        for o, ln in table:
            if o > off:
                break
            if ln:                                   # 0: an end-of-sequence entry
                line = ln
        if line and line not in want:
            want.append(line)
    if not want:
        return []
    lo = min(want)
    hi = min(max(max(want), lo + 4), lo + 14)               # the stretch: 5 to 15 lines
    return [(ln, text[ln - 1].rstrip()) for ln in range(max(1, lo - 1), hi + 1) if ln <= len(text)]


def our_insns(rows: list[dict]) -> list[asmflow.Insn]:
    """Our instructions from objdiff's rows (its formatted text)."""
    out = []
    for r in rows:
        ins = r.get("instruction")
        if ins and ins.get("formatted"):
            mnem, _, ops = ins["formatted"].partition(" ")
            out.append(asmflow.Insn(int(ins.get("address", 0)), mnem, asmflow.split_operands(ops.strip())))
    return out


def unit_pairs(cfg, ctx, names: set[str]) -> dict[str, tuple[list, list]]:
    """function -> (target rows, our rows), one objdiff run per unit with a source."""
    from .matching import load_objdiff_units
    from .route import unit_diff
    wanted = {(ctx.unit(n) or "").rsplit(".", 1)[0] for n in names if ctx.unit(n)}
    out = {}
    for u in load_objdiff_units(cfg):
        src = u.get("metadata", {}).get("source_path", "").removeprefix("src/").rsplit(".", 1)[0]
        if not u.get("base_path") or src not in wanted:
            continue
        pairs, _ = unit_diff(cfg, u)
        out.update({n: p for n, p in pairs.items() if n in names})
    return out


def source_ctor_hints(cfg, unit_name: str | None, function: str, callee_units: list[str]) -> list[str]:
    """The source side, from `close --moves degree`'s detector (degmoves.ctor_hints): lines of our
    definition written `p = alloc; if (p) p = init(p, ...)` where init returns its first parameter."""
    from ...core import cbody
    from . import degmoves, rewrites
    from .trycmd import candidates
    if not unit_name:
        return []
    path = cfg.root / "src" / unit_name
    if not path.is_file():
        return []
    src = path.read_text(errors="replace")
    hits = candidates(src, function)
    if len(hits) != 1:
        return []
    try:
        body = cbody.parse_body(src, hits[0].open, cbody.type_names(src))
    except cbody.ParseError:
        return []
    others = [(cfg.root / "src" / u).read_text(errors="replace") for u in dict.fromkeys(callee_units)
              if u and u != unit_name and (cfg.root / "src" / u).is_file()]
    site = rewrites.Site(src, body, None, not unit_name.endswith(".c"))
    return degmoves.ctor_hints(site, others)


def _span_lines(ix: Index, name: str, lo: int, hi: int) -> list[str]:
    insns = ix.insns[name]
    base = insns[0].addr if insns else 0
    return [f"{x.addr - base:#06x}  {x.text}" for x in insns[lo:hi + 1]]


def _returns_self(ix: Index, texts: dict[str, str]):
    memo: dict[str, bool] = {}

    def check(fn: str) -> bool:
        if fn not in memo:
            memo[fn] = is_ctor_name(fn) or (fn in texts and returns_first_arg(texts[fn]))
        return memo[fn]
    return check


def run(cfg, args) -> int:
    import json
    from pathlib import Path
    from ...core.command import OK, USAGE, hint
    from . import inlinescan
    from .inlinescmd import load_context
    ctx = load_context(cfg)
    if ctx is None:
        print("error: needs the disassembly, the objdiff report and the original binary")
        hint("build")
        return USAGE
    texts = {n: e.text for n, e in ctx.entries.items()}
    pct: dict[str, float] = {}
    for u in json.loads(cfg.report.read_text()).get("units", []):
        for f in u.get("functions", []):
            pct[f["name"]] = float(f.get("fuzzy_match_percent") or 0.0)
    frozen = inlinescan.frozen_names(cfg)
    words = [w for w in args.target if w]
    if args.scan:
        if args.only:
            targets = [n for n in (resolve(texts, q) for q in args.only.split(",")) if n]
        else:
            targets = sorted(n for n in texts if n not in ctx.exact and n in pct and n not in frozen)
            if args.unit:
                u = args.unit.removeprefix("src/")
                targets = [n for n in targets if (ctx.unit(n) or "") == u or (ctx.unit(n) or "").rsplit(".", 1)[0] == u]
    else:
        if not words:
            print("error: give a function (optionally after its unit), or --scan")
            return USAGE
        name = resolve(texts, words[-1])
        if name is None:
            print(f"error: no single function named {words[-1]!r} in the disassembly")
            return USAGE
        targets = [name]
    if not targets:
        print("error: no open, unfrozen functions to scan")
        return USAGE
    pool = set(ctx.exact) - set(targets) - frozen
    ix = Index(texts, pool)
    returns_self = _returns_self(ix, texts)
    diverging: dict[str, set[int]] = {}
    ours: dict[str, set[str]] = {}
    if args.diff:
        open_ = {n for n in targets if pct.get(n, 0.0) < 100.0}
        for fn, (t, o) in unit_pairs(cfg, ctx, open_).items():
            diverging[fn] = {x // 4 for x in inlinescan.diverging_offsets(t, o)}
            ours[fn] = {key for _, _, key in grams(tokenize(our_insns(o)))}
    results = []
    for n in targets:
        hits = ix.query(n, diverging=diverging.get(n), top=args.top, ours=ours.get(n))
        size = ctx.sizes.get(n, len(ix.insns[n]) * 4)
        p = pct.get(n, 0.0)
        sites = ctor_sites(texts[n], returns_self)
        results.append({
            "function": n, "unit": ctx.unit(n), "size": size, "match": p,
            "at_stake": round(size * (100.0 - p) / 100.0),
            "siblings": [{"function": h.function, "unit": ctx.unit(h.function), "score": h.score,
                          "spans": [{"target": [s.q_lo * 4, s.q_hi * 4 + 4], "sibling": [s.c_lo * 4, s.c_hi * 4 + 4],
                                     "weight": round(s.weight, 2), "ours_differs": s.diverges,
                                     "contrast": round(s.contrast, 2) if n in ours else None} for s in h.spans]}
                         for h in hits],
            "constructors": [{"offset": c.offset, "alloc": c.alloc, "callee": c.ctor,
                              "named_ctor": is_ctor_name(c.ctor), "lines": c.lines} for c in sites]})
    if args.scan:
        print("\n".join(render_scan(results, args.limit, args.min_weight)))
    else:
        r = results[0]
        r["source_constructors"] = source_ctor_hints(
            cfg, r["unit"], r["function"], [ctx.unit(c["callee"]) for c in r["constructors"]])
        if args.source:
            for s in r["siblings"][:args.source]:
                sp = s["spans"][0]
                s["source"] = source_lines(cfg, s["unit"], s["function"], list(range(sp["sibling"][0], sp["sibling"][1], 4)))
        print("\n".join(render_one(r, ix)))
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
    from ...core.usage import note
    note(scanned=len(results), functions=[r["function"] for r in results[:30]])
    if results and results[0]["siblings"]:
        hint(f"ctx {results[0]['siblings'][0]['function']}")
    return OK


def render_one(r: dict, ix: Index) -> list[str]:
    out = [f"# {r['function']}  {r['unit'] or 'unit not declared'}  {r['size']:#x} bytes, {r['match']:.2f}%"]
    for c in r["constructors"]:
        out.append(f"\nconstructor call at {c['offset']:#x}: `p = alloc(...); if (p) p = {c['callee']}(p, ...)`"
                   f" ({c['callee']} returns its first argument): write `new (...) T(...)`"
                   + (" (already a constructor: check that the source says new)" if c["named_ctor"]
                      else f" and rename {c['callee']} to the class's __ct__"))
        out += [f"    {x}" for x in c["lines"]]
    for h in r.get("source_constructors", []):
        out.append(f"our source, {h}")
    if not r["siblings"]:
        out.append("\nno exact sibling shares a rare instruction pattern")
    for s in r["siblings"]:
        out.append(f"\n## {s['function']}  {s['unit'] or 'unit not declared'}  score {s['score']}")
        for k, sp in enumerate(s["spans"]):
            here = "  [ours differs here]" if sp["ours_differs"] else ""
            lack = f", {sp['contrast']} of it not in ours" if sp.get("contrast") is not None else ""
            out.append(f"- span weight {sp['weight']}{lack}: target {sp['target'][0]:#x}-{sp['target'][1]:#x}, "
                       f"sibling {sp['sibling'][0]:#x}-{sp['sibling'][1]:#x}{here}")
            if k == 0:
                q = _span_lines(ix, r["function"], sp["target"][0] // 4, sp["target"][1] // 4 - 1)
                c = _span_lines(ix, s["function"], sp["sibling"][0] // 4, sp["sibling"][1] // 4 - 1)
                width = max((len(x) for x in q), default=0)
                for i in range(min(max(len(q), len(c)), MAX_SHOWN)):
                    out.append(f"    {(q[i] if i < len(q) else ''):{width}}   |  {c[i] if i < len(c) else ''}")
        if s.get("source"):
            out.append("  its exact source there:")
            out += [f"    {ln:5}: {t}" for ln, t in s["source"]]
    return out


def hints(results: list[dict], min_weight: float) -> list[dict]:
    """One hint per open function and kind: its constructor calls to a function not yet named as a
    constructor; its best sibling span: with our asm (--diff), the span with the most weight our
    code lacks, else the heaviest. Constructor hints first, then bytes at stake."""
    out = []
    for r in results:
        new = [c for c in r["constructors"] if not c["named_ctor"]]
        if new:
            out.append({"function": r["function"], "match": r["match"], "at_stake": r["at_stake"], "kind": "constructor",
                        "text": f"constructor: {new[0]['callee']} at " + ", ".join(f"{c['offset']:#x}" for c in new)
                                + " returns its first argument -> `new (...) T(...)`, rename it __ct__"})
        spans = [(s, sp) for s in r["siblings"] for sp in s["spans"]]
        key = (lambda x: x[1]["contrast"]) if spans and spans[0][1].get("contrast") is not None else (lambda x: x[1]["weight"])
        best = max(spans, key=key, default=None)
        if best and key(best) >= min_weight:
            s, sp = best
            lack = f", {sp['contrast']} not in ours" if sp.get("contrast") is not None else ""
            out.append({"function": r["function"], "match": r["match"], "at_stake": r["at_stake"], "kind": "sibling",
                        "sibling": s["function"], "score": key(best),
                        "text": f"sibling {s['function']} ({s['unit'] or '?'}): target {sp['target'][0]:#x}-"
                                f"{sp['target'][1]:#x} ~ its {sp['sibling'][0]:#x}-{sp['sibling'][1]:#x}, "
                                f"weight {sp['weight']}{lack}" + ("  [ours differs here]" if sp["ours_differs"] else "")})
    out.sort(key=lambda h: (h["kind"] != "constructor", -h["at_stake"], h["function"]))
    return out


def render_scan(results: list[dict], limit: int, min_weight: float) -> list[str]:
    hs = hints(results, min_weight)
    out = [f"{len(hs)} hints over {len(results)} open functions (constructor calls first, then bytes at stake)"]
    for h in hs[:limit]:
        out.append(f"- {h['function']}  {h['match']:.2f}%  {h['at_stake']} at stake: {h['text']}")
    if len(hs) > limit:
        out.append(f"... {len(hs) - limit} more (--limit, --json)")
    return out


def _configure(p) -> None:
    p.add_argument("target", nargs="*", help="[unit] function (a since-renamed fn_ADDR prefix works)")
    p.add_argument("--top", type=int, default=5, help="siblings per function (default 5)")
    p.add_argument("--source", type=int, default=3, metavar="K",
                   help="one function: the source lines of the top K siblings' spans (a private -sym on compile each; 0 = off)")
    p.add_argument("--diff", action="store_true", help="one objdiff run per unit: spans where ours differs count double")
    p.add_argument("--scan", action="store_true", help="every open, unfrozen function: the best hints by bytes at stake")
    p.add_argument("--unit", help="with --scan: one unit's open functions")
    p.add_argument("--only", metavar="FN[,FN]", help="with --scan: these functions, exact or not (replay: each left out of the pool)")
    p.add_argument("--min-weight", type=float, default=20.0, help="with --scan: smallest span weight listed; with --diff, of the weight our code lacks (default 20)")
    p.add_argument("--limit", type=int, default=20, help="with --scan: hints listed (default 20)")
    p.add_argument("--json", metavar="PATH", help="write every sibling, span and constructor call as JSON")


def _command():
    from ...core.command import Command
    return Command("siblings", "matching",
                   "exact functions sharing an instruction pattern with an open one, with their source lines; "
                   "C-style constructor calls (`alloc; if (p) p = fn(p, ...)`)", run, _configure)


COMMAND = _command()
