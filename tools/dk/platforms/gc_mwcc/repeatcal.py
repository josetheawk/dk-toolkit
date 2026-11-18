"""Calibration and census of the self-repeat finder (`dk inlines --repeats --calibrate | --census`).

Ground truth comes from source: matched functions are compiled privately with `-sym on`; the
`.line` table gives each instruction's source line (an inlined body carries the caller's
call-statement line), the source text gives the names called on each line, and the object gives
the `bl`s that stayed calls. A name declared `inline` that is called on two or more lines with no
`bl` to it is an inline called twice (positive); a function that calls nothing twice as an inline
is a negative. See the pre-registration (PREREG.md) for the frozen definitions and pass marks.
"""

from __future__ import annotations

import json
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ...core import cbody
from ...core.config import Config
from . import asmflow, inlines, repeats
from .elf import Elf
from .matching import MatchError, compile_command, load_objdiff_units, retarget, run_compile
from .regs import line_table
from .regtruth import base_name, virtual_names

OUT = "build/dk/repeats"
CALL = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
KEYWORDS = {"if", "while", "for", "switch", "return", "sizeof", "do", "else", "case", "catch", "operator",
            "inline", "static", "const", "new", "delete", "typedef", "struct", "class", "union", "enum"}
MIN_INLINE_INSNS = 10        # instructions a call line must carry for its inline to be in scope
MIN_NEGATIVE_INSNS = 60
SEED = 20261106
INLINE_DECL = re.compile(r"\binline\b[^;{}()]*?\b([A-Za-z_]\w*)\s*\(")


def inline_names(cfg: Config) -> set[str]:
    names: set[str] = set()
    for top, pats in (("include", ("*.h", "*.hpp")), ("src", ("*.h", "*.hpp", "*.c", "*.cpp", "*.cp"))):
        for pat in pats:
            for path in (cfg.root / top).rglob(pat):
                try:
                    names.update(INLINE_DECL.findall(path.read_text(errors="replace")))
                except OSError:
                    pass
    return names - KEYWORDS


def strip_line(text: str) -> str:
    text = re.sub(r'"(?:\\.|[^"\\])*"', '""', text)
    return re.sub(r"//.*", "", text)


# ---------------------------------------------------------------- ground truth

def unit_truth(cfg: Config, unit: dict, workdir: Path, exact: set[str], inl: set[str], virt: set[str]) -> dict:
    out = {"unit": unit["name"], "status": "ok", "functions": []}
    src_path = cfg.root / unit.get("metadata", {}).get("source_path", "")
    base_path = cfg.root / unit["base_path"]
    if not src_path.is_file() or not base_path.is_file():
        out["status"] = "no-source"
        return out
    workdir.mkdir(parents=True, exist_ok=True)
    for old in workdir.glob("*.o"):
        old.unlink()
    try:
        argv = retarget(compile_command(cfg, unit["base_path"], locked=False), workdir, extra_flags=("-sym", "on"))
    except MatchError as e:
        out.update(status="no-command", error=str(e))
        return out
    ok, log = run_compile(cfg, argv)
    objs = list(workdir.glob("*.o"))
    if not ok or not objs:
        out.update(status="compile-error", error=log[-200:])
        return out
    obj = objs[0]
    ours, base = Elf.parse(obj.read_bytes()), Elf.parse(base_path.read_bytes())
    of, bf = ours.functions(), base.functions()
    if set(of) != set(bf) or any(of[n].code != bf[n].code for n in bf):
        out["status"] = "code-differs"
        obj.unlink(missing_ok=True)
        return out
    src = src_path.read_text(errors="replace")
    spans = cbody.functions(src)
    starts = cbody.line_starts(src)
    lines_text = src.split("\n")
    from .elf import STT_FUNC
    names_at = {(ours.sections[y.shndx].name, y.value): y.name for y in ours.symbols
                if y.type == STT_FUNC and 0 < y.shndx < len(ours.sections)}
    for name in sorted(n for n in of if n in exact):
        fn = of[name]
        table = line_table(ours, name)
        if not table:
            continue
        ninsn = len(fn.code) // 4
        per_insn, k, cur = [], 0, 0
        for i in range(ninsn):
            while k < len(table) and table[k][0] <= 4 * i:
                cur = table[k][1]
                k += 1
            per_insn.append(cur)
        votes = Counter()
        for ln in set(per_insn):
            if 1 <= ln <= len(starts):
                best = None
                for s in spans:
                    if s.head_start <= starts[ln - 1] <= s.close and (best is None or s.open > best.open):
                        best = s
                if best is not None:
                    votes[best.open] += 1
        if not votes:
            continue
        span = next(s for s in spans if s.open == votes.most_common(1)[0][0])
        lo, hi = cbody.line_of(src, span.open) + 1, cbody.line_of(src, span.close) + 1
        calls: dict[str, list[int]] = {}
        for ln in range(lo, hi + 1):
            for n in set(CALL.findall(strip_line(lines_text[ln - 1]))):
                if n not in KEYWORDS:
                    calls.setdefault(n, []).append(ln)
        bl = Counter()
        for r in fn.relocs:
            if r.type == 10:
                sym = r.symbol
                if (not sym or sym.startswith(".")) and r.target_section:
                    sym = names_at.get((r.target_section, r.target_value + r.addend), sym)
                bl[base_name(sym)] += 1
        positives, doubtful = {}, False
        for n, lns in calls.items():
            if len(lns) < 2:
                continue
            if n in inl and bl[n] == 0 and n not in virt:
                groups = {ln: [i for i, l2 in enumerate(per_insn) if l2 == ln] for ln in lns}
                good = {ln: g for ln, g in groups.items() if len(g) >= MIN_INLINE_INSNS}
                if len(good) >= 2:
                    positives[n] = good
                else:
                    doubtful = True              # called twice as an inline but too small for the finder
            elif not (bl[n] >= len(lns) or n in virt):
                doubtful = True                  # repeated name whose calls we cannot account for
        out["functions"].append({"name": name, "insns": ninsn, "positives": {n: {str(l): g for l, g in d.items()}
                                                                          for n, d in positives.items()},
                                 "negative_ok": not doubtful and not positives, "small_inline": doubtful and not positives,
                                 "lines": per_insn})
    obj.unlink(missing_ok=True)
    return out


def gather_truth(cfg: Config, exact: set[str], jobs: int, force: bool = False) -> list[dict]:
    path = cfg.root / OUT / "truth.json"
    if path.is_file() and not force:
        return json.loads(path.read_text())
    from .assets import ensure
    ensure(cfg)
    units = [u for u in load_objdiff_units(cfg) if u.get("base_path") and u.get("metadata", {}).get("source_path")]
    inl, virt = inline_names(cfg), virtual_names(cfg)
    base = cfg.root / OUT / "t"

    def one(args):
        i, u = args
        try:
            return unit_truth(cfg, u, base / str(i), exact, inl, virt)
        except Exception as e:
            return {"unit": u["name"], "status": "error", "error": f"{type(e).__name__}: {e}", "functions": []}

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(one, enumerate(units)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results))
    return results


# ---------------------------------------------------------------- scoring

def true_block(report: repeats.Report, fn_start: int, rec: dict, rep: repeats.Repeat) -> str | None:
    """The inline name a reported block belongs to, or None (false block)."""
    lines = rec["lines"]
    by_group: dict[tuple, set[int]] = {}
    for n, d in rec["positives"].items():
        for ln, insns in d.items():
            by_group[(n, ln)] = set(insns)
    hits: dict[str, set[str]] = {}
    for s in rep.starts:
        addrs = report.addrs[s:s + rep.length]
        idx = [(a - fn_start) // 4 for a in addrs]
        for (n, ln), insns in by_group.items():
            if sum(1 for i in idx if i in insns) >= 0.7 * len(idx):
                hits.setdefault(n, set()).add(ln)
    for n, lns in hits.items():
        if len(lns) >= 2:
            return n
    return None


def evaluate(ctx, truth: list[dict], jobs: int) -> dict:
    funcs = [dict(f, unit=u["unit"]) for u in truth for f in u["functions"] if f["name"] in ctx.entries]
    positives = sorted((f for f in funcs if f["positives"]), key=lambda f: f["name"])
    negatives = sorted((f for f in funcs if f["negative_ok"] and f["insns"] >= MIN_NEGATIVE_INSNS), key=lambda f: f["name"])
    small = sum(1 for f in funcs if f["small_inline"])
    rng = random.Random(SEED)
    pos = rng.sample(positives, min(60, len(positives)))
    neg = rng.sample(negatives, min(30, len(negatives)))
    res = {"population": {"matched_with_truth": len(funcs), "positives": len(positives), "negatives": len(negatives),
                          "small_inline_only": small, "units": len(truth),
                          "unit_status": dict(Counter(u["status"] for u in truth))},
           "sample": {"positives": len(pos), "negatives": len(neg)}}

    def run(rec, **kw):
        name = rec["name"]
        p = inlinescmd_prepared(ctx, name)
        return p, repeats.analyze(name, p, ctx.canon, **kw)

    rows = []
    for rec in pos:
        p, rep = run(rec)
        start = p.flow.nodes[0].addr
        tb = [(r, true_block(rep, start, rec, r)) for r in rep.repeats]
        _p2, rep_nv = run(rec, values=False)
        tb_nv = [true_block(rep_nv, start, rec, r) for r in rep_nv.repeats]
        found = {n for _r, n in tb if n}
        rows.append({"function": rec["name"], "unit": rec["unit"], "names": sorted(rec["positives"]),
                     "blocks": len(tb), "true_blocks": sum(1 for _r, n in tb if n), "found": sorted(found),
                     "blocks_novalue": len(tb_nv), "true_novalue": sum(1 for n in tb_nv if n)})
    res["positives"] = rows
    tp, allb = sum(r["true_blocks"] for r in rows), sum(r["blocks"] for r in rows)
    res["precision"] = tp / allb if allb else None
    res["recall"] = sum(1 for r in rows if r["true_blocks"]) / len(rows) if rows else None
    res["recall_novalue"] = sum(1 for r in rows if r["true_novalue"]) / len(rows) if rows else None
    names_all = sum(len(r["names"]) for r in rows)
    res["name_recall"] = sum(len(set(r["found"]) & set(r["names"])) for r in rows) / names_all if names_all else None
    nrows = []
    for rec in neg:
        _p, rep = run(rec)
        nrows.append({"function": rec["name"], "unit": rec["unit"], "insns": rec["insns"], "false_blocks": len(rep.repeats),
                      "covered": rep.covered()})
    res["negatives"] = nrows
    res["false_in_negatives"] = sum(r["false_blocks"] for r in nrows)
    # shuffled controls (shape + identity only)
    shuf = []
    for rec in pos + neg:
        p = inlinescmd_prepared(ctx, rec["name"])
        _seq, shapes, ids = repeats.sequence(p, ctx.canon)
        keys = repeats.shuffled(list(zip(shapes, ids)), SEED)
        shuf.append(len(repeats.find_repeats(keys)))
    res["shuffled"] = {"functions": len(shuf), "blocks": sum(shuf)}
    # literal-swap control
    swaps = []
    for rec, row in zip(pos, rows):
        if not row["true_blocks"]:
            continue
        sw = literal_swap(ctx, rec)
        if sw:
            swaps.append(sw)
    res["swaps"] = swaps
    res["swap_rejected"] = sum(1 for s in swaps if not s["in_pair_identity"]) / len(swaps) if swaps else None
    res["swap_shape_still_reports"] = sum(1 for s in swaps if s["in_pair_shape"]) / len(swaps) if swaps else None
    return res


def inlinescmd_prepared(ctx, name):
    from . import inlinescmd
    return inlinescmd.prepared(ctx, name)


def literal_swap(ctx, rec: dict) -> dict | None:
    """Change one relocation target in copy 2 of the function's first true block and rerun."""
    name = rec["name"]
    p = inlinescmd_prepared(ctx, name)
    rep = repeats.analyze(name, p, ctx.canon)
    start = p.flow.nodes[0].addr
    seq, shapes, ids = repeats.sequence(p, ctx.canon)
    for r in rep.repeats:
        if true_block(rep, start, rec, r) is None:
            continue
        for pos in range(r.starts[1], r.starts[1] + r.length):
            raw = repeats.shape_and_identity(p.flow.nodes[seq[pos]])[1]
            if raw:
                addr = p.flow.nodes[seq[pos]].addr
                text = ctx.entries[name].text
                new, hit = [], False
                for line in text.split("\n"):
                    if not hit and line.startswith(f"/* {addr:08X} "):
                        line, hit = line.replace(raw[0], "swapped_sym_x", 1), True
                    new.append(line)
                if not hit:
                    return None
                flow = asmflow.flow_of(name, "\n".join(new))
                p2 = inlines.prepare(inlines.canonicalize(flow, ctx.canon))
                out = {}
                for key, ident in (("in_pair_identity", True), ("in_pair_shape", False)):
                    rep2 = repeats.analyze(name, p2, ctx.canon, identity=ident)
                    idx = rep2.addrs.index(addr) if addr in rep2.addrs else -1
                    out[key] = any(s2 <= idx < s2 + rr.length for rr in rep2.repeats for s2 in rr.starts)
                return {"function": name, "addr": f"{addr:08X}", "symbol": raw[0], **out}
        return None
    return None


# ---------------------------------------------------------------- census

_CENSUS: dict = {}


def _census_one(name: str) -> dict:
    ctx = _CENSUS["ctx"]
    p = inlinescmd_prepared(ctx, name)
    rep = repeats.analyze(name, p, ctx.canon)
    body = rep.size
    return {"function": name, "unit": ctx.unit(name), "size": ctx.sizes.get(name, 4 * len(p.flow.nodes)), "body": body,
            "blocks": len(rep.repeats), "max_copies": max((len(r.starts) for r in rep.repeats), default=0),
            "copies": sum(len(r.starts) for r in rep.repeats), "covered": rep.covered(),
            "redundant": rep.covered(redundant=True), "longest": max((r.length for r in rep.repeats), default=0),
            "blocks_detail": [{"insns": r.length, "copies": [f"{rep.addrs[s]:08X}" for s in r.starts], "verified": r.verified}
                              for r in rep.repeats]}


def census(ctx, jobs: int, skip: tuple = ()) -> dict:
    import multiprocessing
    names = sorted(n for n in ctx.entries if n not in ctx.exact)
    _CENSUS["ctx"] = ctx
    if jobs > 1:
        with multiprocessing.get_context("fork").Pool(jobs) as pool:
            rows = pool.map(_census_one, names, chunksize=8)
    else:
        rows = [_census_one(n) for n in names]
    keep = [r for r in rows if r["function"] not in skip]
    total_bytes = sum(r["size"] for r in keep)
    cov = sum(4 * r["covered"] for r in keep)
    red = sum(4 * r["redundant"] for r in keep)
    giants = [r for r in keep if r["size"] >= 0x800]
    return {"open_functions": len(rows), "skipped": list(skip), "open_bytes": total_bytes, "covered_bytes": cov,
            "coverage": cov / total_bytes if total_bytes else 0.0, "redundant_bytes": red,
            "redundant_coverage": red / total_bytes if total_bytes else 0.0,
            "giants": len(giants), "giant_bytes": sum(r["size"] for r in giants),
            "giant_coverage": (sum(4 * r["covered"] for r in giants) / sum(r["size"] for r in giants)) if giants else 0.0,
            "functions_with_repeats": sum(1 for r in keep if r["blocks"]), "rows": rows}


def census_table(doc: dict, top: int = 20) -> list[str]:
    rows = [r for r in doc["rows"] if r["function"] not in doc["skipped"]]
    out = ["| function | unit | bytes | blocks | max copies | covered bytes | covered % | redundant bytes |",
           "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (-r["covered"], r["function"]))[:top]:
        out.append(f"| {r['function']} | {r['unit'] or '-'} | {r['size']} | {r['blocks']} | {r['max_copies']} | "
                   f"{4 * r['covered']} | {100.0 * 4 * r['covered'] / r['size']:.0f}% | {4 * r['redundant']} |")
    return out


def run_all(cfg: Config, ctx, jobs: int, force: bool = False) -> dict:
    t0 = time.time()
    truth = gather_truth(cfg, ctx.exact, jobs, force)
    cal = evaluate(ctx, truth, jobs)
    cen = census(ctx, jobs, tuple((cfg.platform_options.get("repeatcal") or {}).get("census_skip") or ()))
    out = cfg.root / OUT
    out.mkdir(parents=True, exist_ok=True)
    doc = {"calibration": cal, "census": cen, "seconds": time.time() - t0}
    (out / "result.json").write_text(json.dumps(doc, indent=1))
    return doc
