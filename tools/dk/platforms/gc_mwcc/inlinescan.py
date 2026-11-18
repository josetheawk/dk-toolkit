"""dk inlines --scan: inline-boundary fingerprints in the TARGET asm of every open function.

CodeWarrior leaves marks where an inlined callee begins or ends:
  zero-copy   `li rX,0` then `mr rY,rX` (GC/1.3.x) or `addi rY,rX,0` (GC/1.2.5n): the callee's local
              (often its loop counter) starts as a copy of a zero the caller already holds; plain code
              loads the zero twice. Strong.
  reread      a field loaded twice in one block with no store or call between: plain code reuses the
              first load, an inline boundary does not. Weak (aliasing through stores is excluded).
  body        a run of instructions that is an exact function's body (value-numbered match, inlines.py).
              Strong; names the callee and the call to write.
A function is flagged on one strong mark or two weak ones. Rows rank by bytes at stake (the
function's unmatched bytes). Frozen functions are never listed (frozen_names, the one call site).
"""

from __future__ import annotations

import json
import multiprocessing
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import asmflow, inlines

ZERO_WINDOW = 8           # instructions between the zero and its copy
LOOP_WINDOW = 4           # a label this close after the copy = the copy starts a loop
_MEMOP = re.compile(r"^(?P<off>-?(?:0x[0-9a-fA-F]+|\d+)|[^()]+@\w+)\((?P<base>r\d+)\)$")
_SKIP_BASE = {"r1", "r2", "r13"}      # stack and small-data bases: reloads there are spills or globals
_NOT_PARAM = _SKIP_BASE | {"r0", "ctr", "lr"} | {f"cr{i}" for i in range(8)}


@dataclass
class Mark:
    kind: str                 # zero-copy | reread | body
    start: int                # offset in the function
    end: int                  # offset after the region the mark suggests
    lines: list[str]          # the fingerprint instructions, `+off: mnem ops`
    callee: str | None = None
    call: str | None = None
    params: list[str] = field(default_factory=list)
    loop: bool = False
    differs: bool | None = None   # ours differs from the target inside the region (None: not diffed)

    @property
    def strong(self) -> bool:
        return self.kind in ("zero-copy", "body")

    def shape(self) -> str:
        span = f"+{self.start:#x}..+{self.end:#x}"
        args = f"({', '.join(self.params)})" if self.params else "()"
        if self.kind == "body":
            return (f"call exact {self.callee} inline here as `{self.call}`: an inline helper with its body "
                    f"(keep {self.callee} itself plain)")
        if self.kind == "zero-copy":
            what = "the loop" if self.loop else "the code"
            return f"write {what} at {span} as an inline helper taking {args}; its local starts at the caller's zero"
        return (f"read the field again inside an inline helper at {span} taking {args} "
                f"(pass the object, do not reuse the caller's load)")


def _line(ins: asmflow.Insn, base: int) -> str:
    return f"+{ins.addr - base:#x}: {ins.mnem} {', '.join(ins.ops)}".rstrip()


def _reads(ins: asmflow.Insn) -> set[str]:
    """Registers an instruction reads (approximate)."""
    m = ins.mnem.rstrip("+-.")
    ops = list(ins.ops)
    out: set[str] = set()
    if m in ("li", "lis") or m.startswith("b") or not ops:
        return out
    first_is_dest = not (m.startswith("st") or m.startswith(("cmp", "fcmp")) or m.startswith("mt"))
    for k, op in enumerate(ops):
        mm = _MEMOP.match(op)
        if mm:
            out.add(mm["base"])
        elif asmflow.is_reg(op) and not (k == 0 and first_is_dest):
            out.add(op)
    return out


def _region_params(insns: list[asmflow.Insn], lo: int, hi: int, skip: set[str]) -> list[str]:
    """Registers read in insns[lo:hi] before being written there: the helper's inputs."""
    written = set(skip)
    out: list[str] = []
    for ins in insns[lo:hi]:
        for r in sorted(_reads(ins)):
            if r not in written and r not in _NOT_PARAM and r not in out:
                out.append(r)
        written |= asmflow._writes(ins)
    return out


def _is_zero(ins: asmflow.Insn) -> str | None:
    if ins.mnem == "li" and len(ins.ops) == 2 and asmflow.parse_int(ins.ops[1]) == 0:
        return ins.ops[0]
    return None


def _copies(ins: asmflow.Insn, src: str) -> str | None:
    if ins.mnem == "mr" and len(ins.ops) == 2 and ins.ops[1] == src:
        return ins.ops[0]
    if ins.mnem in ("addi", "ori") and len(ins.ops) == 3 and ins.ops[1] == src and asmflow.parse_int(ins.ops[2]) == 0:
        return ins.ops[0]
    return None


def zero_copies(insns: list[asmflow.Insn]) -> list[Mark]:
    base = insns[0].addr if insns else 0
    out: list[Mark] = []
    for i, ins in enumerate(insns):
        z = _is_zero(ins)
        if not z or z == "r1":
            continue
        for j in range(i + 1, min(len(insns), i + 1 + ZERO_WINDOW)):
            nxt = insns[j]
            if nxt.mnem.startswith("b") or (j > i + 1 and nxt.labels):
                break
            dst = _copies(nxt, z)
            if dst and dst != "r1":
                loop_at = next((k for k in range(j + 1, min(len(insns), j + 1 + LOOP_WINDOW)) if insns[k].labels), None)
                end = j + 1
                if loop_at is not None:
                    names = set(insns[loop_at].labels)
                    backs = [k for k in range(loop_at, len(insns))
                             if insns[k].mnem.startswith("b") and insns[k].ops and insns[k].ops[-1] in names]
                    end = (backs[-1] + 1) if backs else loop_at + 1
                params = _region_params(insns, j + 1, end, {z, dst})
                out.append(Mark("zero-copy", ins.addr - base, insns[end - 1].addr + 4 - base,
                                [_line(ins, base), _line(nxt, base)], params=params, loop=loop_at is not None))
                break
            if z in asmflow._writes(nxt):
                break
    return out


def rereads(insns: list[asmflow.Insn]) -> list[Mark]:
    """A load repeated in one block (same address operand, base unchanged, no store/call between)."""
    base = insns[0].addr if insns else 0
    out: list[Mark] = []
    seen: dict[tuple[str, str], int] = {}
    for i, ins in enumerate(insns):
        m = ins.mnem
        if ins.labels:
            seen.clear()                              # a new block
        if m.startswith("b") or asmflow._is_store(m):
            seen.clear()
            continue
        if asmflow._is_load(m) and len(ins.ops) == 2 and "u" not in m[2:]:
            mm = _MEMOP.match(ins.ops[1])
            if mm and mm["base"] not in _SKIP_BASE and ins.ops[0] != mm["base"]:
                key = (m, ins.ops[1])
                if key in seen:
                    k = seen[key]
                    first = insns[k]
                    out.append(Mark("reread", ins.addr - base, ins.addr + 4 - base,
                                    [_line(first, base), _line(ins, base)], params=[mm["base"]]))
                    del seen[key]
                else:
                    seen[key] = i
        w = asmflow._writes(ins)
        for key in [k for k in seen if _MEMOP.match(k[1])["base"] in w]:
            del seen[key]
    return out


def fingerprints(text: str) -> list[Mark]:
    insns = asmflow.parse_insns(text)
    return zero_copies(insns) + rereads(insns)


def flagged(marks: list[Mark]) -> bool:
    return any(m.strong for m in marks) or sum(1 for m in marks if not m.strong) >= 2


# ---------------------------------------------------------------- the frozen list (one call site)

def frozen_names(cfg) -> set[str]:
    """Functions no tool may list: the one frozen union (gate.frozen_keys, every eval set minus the released)."""
    try:
        from .gate import frozen_keys
        return set(frozen_keys(cfg))
    except Exception:                                         # no sets in this checkout
        return set()


# ---------------------------------------------------------------- scanning

_WORK: dict = {}


def _scan_one(name: str) -> dict:
    from .inlinescmd import call_text, prepared
    w = _WORK
    ctx = w["ctx"]
    text = ctx.entries[name].text
    insns = asmflow.parse_insns(text)
    marks = zero_copies(insns) + rereads(insns)
    if w["templates"]:
        g = prepared(ctx, name)
        nodes = g.flow.nodes
        base = insns[0].addr if insns else 0
        tmpl = {k: v for k, v in w["templates"].items() if k != name and k not in w["same_as"].get(name, ())}
        for m in inlines.find_copies(tmpl, g, min_weight=w["min_weight"], max_weight=inlines.MAX_WEIGHT):
            idx = sorted(m.claims)
            if not idx:
                continue
            lo, hi = nodes[idx[0]].insn, nodes[idx[-1]].insn
            marks.append(Mark("body", lo.addr - base, hi.addr + 4 - base,
                              [_line(lo, base), _line(hi, base)], callee=m.template,
                              call=call_text(m.template, m.bind, g.flow)))
    marks.sort(key=lambda m: (m.start, m.kind))
    size = ctx.sizes.get(name, len(insns) * 4)
    pct = w["pct"].get(name, 0.0)
    return {"function": name, "unit": ctx.unit(name), "size": size, "match": pct,
            "at_stake": round(size * (100.0 - pct) / 100.0), "flagged": flagged(marks),
            "marks": [{"kind": m.kind, "start": m.start, "end": m.end, "lines": m.lines, "callee": m.callee,
                       "call": m.call, "params": m.params, "loop": m.loop, "shape": m.shape()} for m in marks]}


def scan(ctx, targets: list[str], pct: dict[str, float], jobs: int = 1, bodies: bool = True,
         min_weight: int = 10) -> list[dict]:
    """Fingerprints of each target; with bodies, copies of exact functions (identical code once)."""
    templates: dict = {}
    same_as: dict[str, set] = {}
    if bodies:
        from .inlinescmd import build_pool
        others = {n for n in ctx.entries if n not in ctx.exact} | set(targets)
        templates, helpers = build_pool(ctx, others, min_weight, inlines.MAX_WEIGHT)
        for t, h in helpers.items():
            for s in h.same:
                same_as.setdefault(s, set()).add(t)
    _WORK.update(ctx=ctx, templates=templates, same_as=same_as, pct=pct, min_weight=min_weight)
    order = sorted(targets, key=lambda n: -len(ctx.entries[n].text))
    if jobs > 1 and len(targets) > 1:
        with multiprocessing.get_context("fork").Pool(min(jobs, len(targets))) as pool:
            done = dict(zip(order, pool.map(_scan_one, order, chunksize=4)))
    else:
        done = {n: _scan_one(n) for n in order}
    return [done[n] for n in targets]


def render(rows: list[dict], limit: int) -> list[str]:
    hits = sorted((r for r in rows if r["flagged"]),
                  key=lambda r: (not r.get("differs_at_mark", True), -r["at_stake"], r["function"]))
    out = [f"{len(hits)} of {len(rows)} open functions show inline-boundary fingerprints "
           f"({sum(r['at_stake'] for r in hits)} bytes at stake); biggest first"]
    for r in hits[:limit]:
        unit = f"src/{r['unit']}" if r["unit"] else "unit not declared"
        out.append(f"\n# {r['function']}  {unit}  {r['size']:#x} bytes, {r['match']:.2f}%, {r['at_stake']} at stake")
        for m in r["marks"][:limit]:
            if m["kind"] == "reread" and sum(1 for x in r["marks"] if x["kind"] != "reread") and len(r["marks"]) > 6:
                continue
            here = {True: "  [ours differs here]", False: "", None: ""}[m.get("differs")]
            out.append(f"- {m['kind']:9} {' ; '.join(m['lines'])}{here}")
            out.append(f"    -> {m['shape']}")
    if len(hits) > limit:
        out.append(f"\n... {len(hits) - limit} more (--limit, --json)")
    return out


def diverging_offsets(target_rows: list[dict], our_rows: list[dict]) -> list[int]:
    """Target offsets of the objdiff rows where ours differs (an instruction missing on one side counts)."""
    addrs = [int(r["instruction"].get("address", 0)) for r in target_rows if "instruction" in r]
    if not addrs:
        return []
    base, out, last = addrs[0], [], addrs[0]
    for t, o in zip(target_rows, our_rows):
        ti, oi = t.get("instruction"), o.get("instruction")
        if ti:
            last = int(ti.get("address", 0))
        if t.get("diff_kind") or o.get("diff_kind") or not ti or not oi or ti.get("formatted") != oi.get("formatted"):
            out.append(last - base)
    return out


def mark_divergence(rows: list[dict], offsets: dict[str, list[int]]) -> None:
    """Per mark: does ours differ inside its region (a few instructions of slack before it)?"""
    for r in rows:
        offs = offsets.get(r["function"])
        if offs is None:
            continue
        for m in r["marks"]:
            m["differs"] = any(m["start"] - 8 <= x < m["end"] + 4 for x in offs)
        r["differs_at_mark"] = any(m["differs"] for m in r["marks"])


def write_json(path: str, rows: list[dict]) -> None:
    Path(path).write_text(json.dumps(rows, indent=1, sort_keys=True) + "\n")
