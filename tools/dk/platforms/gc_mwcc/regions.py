"""dk regions: cut a giant function into small regions tied to its source statements, each scored alone.

A lens only: nothing here compiles a region by itself or changes a source. One private `-sym on` compile of the
unit gives objdiff's aligned rows with our line numbers; each row goes to the innermost statement holding its line
(carried forward like rewrites.symptoms; a line outside the body, such as an inlined helper's header line, stays
with the previous in-body row: the call site). Rows on the body's `{`/`}` lines are the `frame` region. The statement
tree is cut at R_MAX rows, adjacent pieces with one parent are packed up to R_MAX, an oversize leaf is `atomic`, and
regions whose row spans overlap by more than 25 % merge (scheduling interleaves them). Each region is classed by a
local difflib alignment of its own rows, registers normalised: exact, reg (register numbers only), sched (the same
instructions in another order) or shape (`shape_ops` rows differ).
"""

from __future__ import annotations

import argparse
import bisect
import difflib
import json
import re
import statistics
import tempfile
from pathlib import Path

from ...core import cbody
from ...core.command import OK, USAGE, Command
from ...core.config import Config

R_MAX = 48
MERGE = 0.25
_REG = re.compile(r"\b[rf]\d+\b|\bcr\d\b")
_BRANCH = re.compile(r"^(b\w*)\s+(?:cr\d,\s*)?-?0x[0-9a-fA-F]+$")


def _text(row: dict) -> str | None:
    ins = row.get("instruction")
    return ins.get("formatted") if ins else None


def row_lines(rt: list[dict]) -> list[int | None]:
    """Our line per row, carried forward from the last row that has one (None before the first)."""
    out, last = [], None
    for o in rt:
        line = (o.get("instruction") or {}).get("line_number")
        last = line or last
        out.append(last)
    return out


def classify(lt: list[dict], rt: list[dict], idx: list[int]) -> tuple[str, int]:
    """(class, shape_ops) of these rows, re-aligned locally. Rows objdiff does not flag count as equal (their
    branch offsets may differ only because code moved elsewhere); flagged rows compare with branch targets masked."""
    t, o = [], []
    for k in idx:
        a, b = _text(lt[k]), _text(rt[k])
        if not (lt[k].get("diff_kind") or rt[k].get("diff_kind")):
            a = b = a if a is not None else b
        if a is not None:
            t.append(_BRANCH.sub(r"\1 T", a))
        if b is not None:
            o.append(_BRANCH.sub(r"\1 T", b))
    if t == o:
        return "exact", 0
    nt, no = [_REG.sub("R", x) for x in t], [_REG.sub("R", x) for x in o]
    if nt == no:
        return "reg", 0
    if sorted(nt) == sorted(no):
        return "sched", 0
    ops = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in
              difflib.SequenceMatcher(None, nt, no, autojunk=False).get_opcodes() if tag != "equal")
    return "shape", ops


def segment(lt: list[dict], rt: list[dict], src: str, body: cbody.Stmt, r_max: int = R_MAX) -> list[dict]:
    """The regions of one function: rid, kind (frame/stmt/atomic), first and last source line (1-based), row
    indices, class and shape_ops. Every row is in exactly one region."""
    starts = cbody.line_starts(src)
    line_of = lambda off: bisect.bisect_right(starts, off)          # 1-based line of an offset
    open_line, close_line = line_of(body.start), line_of(body.end - 1)
    stmts = [s for s in cbody.statements(body) if s is not body]
    spans = [(line_of(s.start), line_of(s.end - 1), s) for s in stmts]
    ids = {id(s): k for k, s in enumerate(stmts)}
    parent = {}
    for p in cbody.statements(body):
        for k in ([p.init] if p.init is not None else []) + p.kids:
            parent[id(k)] = p

    def innermost(line: int):
        best = None
        for a, b, s in spans:
            if a <= line <= b and (best is None or (b - a, -a) < (best[1] - best[0], -best[0])):
                best = (a, b, s)
        return best[2] if best else None

    owner: list[object] = []                                        # per row: "frame" or a statement
    prev = "frame"
    for line in row_lines(rt):
        if line is None or line in (open_line, close_line):
            o = "frame"
        elif open_line < line < close_line:
            o = innermost(line) or prev
        else:
            o = prev                                                # an inlined helper's header line: the call site
        owner.append(o)
        if o != "frame":
            prev = o
    rows_of: dict[int, list[int]] = {}
    for k, o in enumerate(owner):
        if o != "frame":
            rows_of.setdefault(id(o), []).append(k)

    def subtree_rows(s) -> list[int]:
        return sorted(k for x in cbody.statements(s) for k in rows_of.get(id(x), []))

    pieces: list[tuple[object, object, list[int], bool]] = []      # (statement, parent, rows, atomic)

    def cut(s) -> None:
        rows = subtree_rows(s)
        if not rows:
            return
        kids = ([s.init] if s.init is not None else []) + s.kids
        if len(rows) <= r_max or not kids:
            pieces.append((s, parent.get(id(s)), rows, len(rows) > r_max))
            return
        own = sorted(rows_of.get(id(s), []))
        if own:
            pieces.append((s, parent.get(id(s)), own, len(own) > r_max))
        for k in kids:
            cut(k)

    for k in body.kids:
        cut(k)
    packed: list[dict] = []
    for s, par, rows, atomic in pieces:
        last = packed[-1] if packed else None
        if last and not atomic and not last["atomic"] and last["parent"] is par and par is not None \
                and len(last["rows"]) + len(rows) <= r_max:
            last["rows"] += rows
            last["last"] = max(last["last"], line_of(s.end - 1))
            continue
        packed.append({"parent": par, "rows": list(rows), "atomic": atomic, "first": line_of(s.start),
                       "last": line_of(s.end - 1), "kind": "atomic" if atomic else "stmt"})
    frame = [k for k, o in enumerate(owner) if o == "frame"]
    if frame:
        packed.append({"parent": None, "rows": frame, "atomic": False, "first": open_line, "last": close_line,
                       "kind": "frame"})
    merged = True
    while merged:                                                   # interleaved row spans: one region
        merged = False
        spans_ = [(min(r["rows"]), max(r["rows"])) for r in packed]
        for i in range(len(packed)):
            for j in range(i + 1, len(packed)):
                if packed[i]["kind"] == "frame" or packed[j]["kind"] == "frame":
                    continue
                (a0, a1), (b0, b1) = spans_[i], spans_[j]
                over = min(a1, b1) - max(a0, b0) + 1
                if over > 0 and over > MERGE * min(a1 - a0 + 1, b1 - b0 + 1):
                    a, b = packed[i], packed.pop(j)
                    a["rows"] = sorted(a["rows"] + b["rows"])
                    a["first"], a["last"] = min(a["first"], b["first"]), max(a["last"], b["last"])
                    a["atomic"] = a["atomic"] or b["atomic"] or len(a["rows"]) > r_max
                    a["kind"] = "atomic" if a["atomic"] else "stmt"
                    merged = True
                    break
            if merged:
                break
    out = []
    for r in sorted(packed, key=lambda r: (r["kind"] == "frame", min(r["rows"]), r["first"])):
        cls, ops = classify(lt, rt, r["rows"])
        out.append({"rid": f"{r['kind']}:L{r['first'] - open_line}-{r['last'] - open_line}", "kind": r["kind"],
                    "first": r["first"], "last": r["last"], "rows": len(r["rows"]), "row_span": [min(r["rows"]), max(r["rows"])],
                    "differing": sum(1 for k in r["rows"] if lt[k].get("diff_kind") or rt[k].get("diff_kind")),
                    "class": cls, "shape_ops": ops, "indices": sorted(r["rows"])})
    seen: dict[str, int] = {}
    for r in out:                                                   # unique ids, stable in row order
        n = seen.get(r["rid"], 0)
        seen[r["rid"]] = n + 1
        if n:
            r["rid"] += f"#{n}"
    return out


def stats(regions: list[dict], rows: int, unmapped: int) -> dict:
    shape = [r for r in regions if r["class"] == "shape"]
    diff = sorted((r["differing"] for r in regions), reverse=True)
    total_diff = sum(diff) or 1
    return {"regions": len(regions), "rows": rows, "shape_regions": len(shape),
            "shape_rows_share": round(sum(r["rows"] for r in shape) / rows, 4) if rows else 0.0,
            "atomic_shape": sum(1 for r in shape if r["kind"] == "atomic"),
            "top3_share": round(sum(diff[:3]) / total_diff, 4) if sum(diff) else 1.0,
            "unmapped_share": round(unmapped / rows, 4) if rows else 0.0,
            "classes": {c: sum(1 for r in regions if r["class"] == c) for c in ("exact", "reg", "sched", "shape")}}


# ---- compile and census -------------------------------------------------------------------------------

def body_of(text: str, symbol: str, rt: list[dict]):
    from .closecmd import plain_name
    first = next((o["instruction"]["line_number"] for o in rt if (o.get("instruction") or {}).get("line_number")), None)
    plain = plain_name(symbol)
    span = cbody.function_at_line(text, first - 1) if first else None
    if span is not None and plain is not None and span.name.split("::")[-1] != plain:
        span = None
    if span is None and plain is not None:
        from .trycmd import candidates
        hits = candidates(text, symbol)
        span = hits[0] if len(hits) == 1 else None
    if span is None:
        return None
    try:
        return cbody.parse_body(text, span.open, cbody.type_names(text))
    except cbody.ParseError:
        return None


def unit_regions(cfg: Config, unit: dict, symbols: list[str], keep_indices: bool = False) -> dict[str, dict]:
    """symbol -> {regions, stats} or {error}, from one private -sym on compile of the unit."""
    from .regs import _diff
    from .reorder import compile_variant
    src = cfg.root / unit["metadata"]["source_path"]
    text = src.read_text()
    out: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="dk-regions-") as work:
        obj, _err = compile_variant(cfg, unit, text, Path(work), ("-sym", "on"))
        if obj is None:
            return {s: {"error": "compile failed"} for s in symbols}
        for s in symbols:
            pair = _diff(cfg, unit, obj, s)
            if pair is None:
                out[s] = {"error": "no objdiff pair"}
                continue
            lt, rt = pair[0]["instructions"], pair[1]["instructions"]
            body = body_of(text, s, rt)
            if body is None:
                out[s] = {"error": "body not found or not parsed"}
                continue
            regs = segment(lt, rt, text, body)
            unmapped = sum(1 for x in row_lines(rt) if x is None)
            if not keep_indices:
                for r in regs:
                    r.pop("indices")
            out[s] = {"percent": round(float(pair[0].get("match_percent", 0.0)), 4), "regions": regs,
                      "stats": stats(regs, len(rt), unmapped)}
    return out


def giants(cfg: Config) -> list[dict]:
    from . import route
    rows = route.table(cfg) or []
    return sorted((r for r in rows if r["class"] == "giant" and (r["fuzzy"] or 0.0) > 0.0),
                  key=lambda r: (r["unit"], r["address"], r["function"]))


def census(cfg: Config, rows: list[dict], jobs: int = 4) -> dict:
    from concurrent.futures import ThreadPoolExecutor
    from .matching import MatchError, resolve_unit
    by_unit: dict[str, list[str]] = {}
    for r in rows:
        by_unit.setdefault(r["unit"], []).append(r["function"])

    def one(name: str) -> tuple[str, dict]:
        try:
            unit = resolve_unit(cfg, name)
        except MatchError as e:
            return name, {f: {"error": str(e)} for f in by_unit[name]}
        return name, unit_regions(cfg, unit, by_unit[name])

    with ThreadPoolExecutor(max(1, jobs)) as ex:
        done = dict(ex.map(one, sorted(by_unit)))
    return {"functions": {f"{u}::{f}": done[u][f] for u in sorted(done) for f in sorted(done[u])}}


def summarize(doc: dict, keys: list[str] | None = None) -> dict:
    fs = [v for k, v in doc["functions"].items() if (keys is None or k in keys) and "stats" in v]
    if not fs:
        return {"functions": 0}
    st = [f["stats"] for f in fs]
    med = lambda xs: statistics.median(xs)
    return {"functions": len(fs), "errors": sum(1 for k, v in doc["functions"].items()
                                                  if (keys is None or k in keys) and "error" in v),
            "median_shape_regions": med([s["shape_regions"] for s in st]),
            "median_shape_rows_share": round(med([s["shape_rows_share"] for s in st]), 4),
            "no_atomic_shape": sum(1 for s in st if s["atomic_shape"] == 0),
            "median_regions": med([s["regions"] for s in st]), "median_top3_share": round(med([s["top3_share"] for s in st]), 4),
            "median_unmapped_share": round(med([s["unmapped_share"] for s in st]), 4)}


def run(cfg: Config, args: argparse.Namespace) -> int:
    from .jobs import resolve
    from .matching import MatchError, resolve_unit
    if args.giants:
        rows = giants(cfg)
        if args.from_file:
            want = {tuple(x) for x in json.loads(Path(args.from_file).read_text())}
            rows = [r for r in rows if (r["unit"], r["function"]) in want]
        doc = census(cfg, rows, min(4, resolve(args.jobs, cfg)))
        out = Path(args.out) if args.out else cfg.root / "build" / "dk" / "regions" / "census.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        print(json.dumps(summarize(doc), indent=1))
        print(f"wrote {out}")
        return OK
    if not args.function:
        print("error: name a function, or --giants")
        return USAGE
    from .regs import unit_of_function
    try:
        unit = resolve_unit(cfg, args.unit or unit_of_function(cfg, args.function) or "")
    except MatchError as e:
        print(f"error: {e}")
        return USAGE
    res = unit_regions(cfg, unit, [args.function])[args.function]
    if args.json:
        print(json.dumps(res, indent=1))
        return OK
    if "error" in res:
        print(f"error: {res['error']}")
        return USAGE
    print(f"{args.function}: {res['percent']:.2f}%, {res['stats']['regions']} region(s), "
          f"{res['stats']['shape_regions']} shape, {res['stats']['rows']} rows")
    for r in res["regions"]:
        print(f"  {r['rid']:<22} lines {r['first']}-{r['last']}  rows {r['rows']:>3} ({r['differing']} differ)  "
              f"{r['class']}" + (f" ({r['shape_ops']} ops)" if r["shape_ops"] else ""))
    return OK


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("function", nargs="?", help="the function to cut into regions")
    p.add_argument("--unit", default="", help="its unit (default: found from the symbol)")
    p.add_argument("--json", action="store_true", help="the regions as JSON")
    p.add_argument("--giants", action="store_true", help="census of every written giant (route class giant, > 0 %%) "
                   "into build/dk/regions/census.json, one private compile per unit")
    p.add_argument("--from", dest="from_file", default="", metavar="JSON", help="with --giants: only these [unit, fn] pairs")
    p.add_argument("--out", default="", help="with --giants: write the census here instead")
    p.add_argument("--jobs", type=int, default=0, help="units compiled at once (default: dk.toml [platform] jobs)")


COMMAND = Command("regions", "matching", "cut a giant function into statement regions, each scored alone "
                  "(exact, register, scheduling, shape); --giants: the census", run, _configure)
