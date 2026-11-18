"""dk regs --truth: ground truth for register allocation over every exact function.

Every unit with a source is compiled privately with `-sym on` (its code must be byte-identical to the
normal build, else the unit is skipped and reported); each exact function's parameters and locals are
read with their register or stack slot from the debug info, and two predictors (regpredict.py) are run
from the source alone: (a) the rank rule, (b) the compiler's observable behaviour. Per variable and
per function the report gives predicted vs actual for both and a failure class.

Output: build/dk/regs/truth.json and truth.md.
"""

from __future__ import annotations

import json
import re
import struct
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from ...core import cbody
from ...core.config import Config
from . import regpredict as rp
from .dtk_symbols import read_symbols
from .elf import STT_FUNC, Elf
from .matching import MatchError, compile_command, load_objdiff_units, retarget, run_compile
from .regs import callee_saved, debug_refs, find_function, line_table, parse_dwarf1

CLASSES = ("param-merged-temp", "this-high", "inline-param-temp", "cse-load-temp", "strength-reduced-ptr",
           "struct-copy-regs", "volatile-loop-perm", "stack-slot-order", "local-temp-regime", "crossing-misjudged",
           "unexplained")
OUT = "build/dk/regs"


# ---- facts read from the compiled function ---------------------------------------------------------------

def saved_registers(code: bytes) -> dict[str, set[int]]:
    """Callee-saved registers the prologue stores (stmw, stw, stfd, psq_st off r1), up to the first branch."""
    out: dict[str, set[int]] = {"r": set(), "f": set()}
    for i in range(0, min(len(code), 200), 4):
        w, = struct.unpack_from(">I", code, i)
        op, rs, ra = w >> 26, (w >> 21) & 31, (w >> 16) & 31
        if op == 18:
            break
        if ra != 1 or rs < 14:
            continue
        if op == 47:
            out["r"].update(range(rs, 32))
        elif op == 36:
            out["r"].add(rs)
        elif op in (54, 60):
            out["f"].add(rs)
    return out


def exact_functions(cfg: Config, unit: dict, obj: Path) -> set[str]:
    from .objcache import objdiff_json
    doc, _ = objdiff_json(cfg, unit["target_path"], obj)
    if doc is None:
        return set()
    return {s["name"] for s in doc["left"].get("symbols", [])
            if "instructions" in s and float(s.get("match_percent", 0.0)) == 100.0}


# ---- source features for the failure classes ---------------------------------------------------------------

def callee_names(body: rp.Body) -> set[str]:
    out = set()
    for c in body.calls:
        f = c.kids[0]
        while f.kind in ("member", "paren") and f.kids and f.kind == "paren":
            f = f.kids[0]
        out.add(f.name if f.kind in ("name", "member") else "")
    out.discard("")
    return out


def source_features(body: rp.Body, src: str, local_functions: set[str], inline_names: set[str]) -> dict:
    loops = [(l.start, l.end) for l in body.loops]
    indexed_in_loop = any(e.kind == "index" and any(a <= e.start < b for a, b in loops) for e in body.exprs)
    texts = Counter(" ".join(src[e.start:e.end].split()) for e in body.exprs if e.kind in ("index", "member") and e.op != ".")
    names = callee_names(body)
    struct_local = any(d is not None and d[1].init is not None and not d[1].ptr and not d[1].array and
                       not any(w in rp.SIZES or w in rp.QUAL for w in re.findall(r"[A-Za-z_]\w*", d[0].base_type(src)))
                       for d in body.decls.values())
    return {"loop": bool(loops), "loop_index": indexed_in_loop, "repeated_load": any(n >= 2 for n in texts.values()),
            "inline_call": bool(names & (local_functions | inline_names)), "struct_local": struct_local}


def classify(v: dict, fn: dict) -> str:
    """The failure class of one variable the predictors miss (heuristic, from evidence recorded for the function)."""
    f = fn["features"]
    if v["cat"] == "stack":
        return "stack-slot-order"
    if v["cat"] == "vol":
        return "volatile-loop-perm"
    bank_miss = (v["actual"] == "volatile" and str(v["b"] or v["a"]).startswith("saved")) or \
        (str(v["actual"]).startswith("saved") and "volatile" in (v["a"], v["b"]))
    if bank_miss:                  # a call the source shows is not one (inlined), or the reverse
        return "inline-param-temp" if f["inline_call"] else "crossing-misjudged"
    if v["name"] == "this" and v["kind"] == "param":
        return "this-high"
    if v["b"] is not None and v["actual"] != v["b"] and "temp" in (v["actual"], v["b"]):
        return "local-temp-regime"
    if fn["unexplained_temps"] > 0:
        if f["inline_call"]:
            return "inline-param-temp"
        if f["loop_index"]:
            return "strength-reduced-ptr"
        if f["repeated_load"]:
            return "cse-load-temp"
    if f["struct_local"]:
        return "struct-copy-regs"
    if v["kind"] == "param":
        return "param-merged-temp"
    return "unexplained"


# ---- one function / one unit ---------------------------------------------------------------------------------

def category(reg: str | None, cross: bool, local: bool = True) -> str | None:
    if reg is None:
        return None
    if reg == "stack":
        return "stack"
    if callee_saved(reg):
        return f"saved:{reg}"
    if reg == "r0" and cross and local:
        return "temp"
    return "volatile"


def find_span(src_spans: list, line_numbers: list[int], names: list[str], src: str, starts: list[int]):
    """The definition whose body mentions every variable name (the line table also holds inlined lines)."""
    seen = set()
    for ln in sorted(set(line_numbers))[:6]:
        if ln < 1 or ln > len(starts):
            continue
        off = starts[ln - 1]
        best = None
        for f in src_spans:
            if f.head_start <= off <= f.close and (best is None or f.open > best.open):
                best = f
        if best is None or best.open in seen:
            continue
        seen.add(best.open)
        words = set(re.findall(r"[A-Za-z_]\w*", src[best.head_start:best.close]))
        hit = [n for n in names if n in words or n == "this"]
        if len(hit) * 2 >= len(names):         # macros bring in locals the source never names
            return best
    return None


def base_name(symbol: str) -> str:
    """`fn_IsEmpty__6TxtStrCFv` -> `fn_IsEmpty` (the name a source call writes)."""
    i = symbol.find("__", 2 if symbol.startswith("__") else 1)
    return symbol[:i] if i > 0 else symbol


def branch_targets(fn, names_at: dict) -> set[str]:
    """Base names of the functions the compiled function branches to (`bl`, REL24 relocations)."""
    out = set()
    for r in getattr(fn, "relocs", ()):
        if r.type != 10:
            continue
        sym = r.symbol
        if (not sym or sym.startswith(".")) and r.target_section:
            sym = names_at.get((r.target_section, r.target_value + r.addend), sym)
        out.add(base_name(sym))
    return out


def inlined_leaf_calls(body: rp.Body, targets: set[str], bodies: dict[str, str], known: set[str],
                       virtual: set[str] = frozenset()) -> set[str]:
    """Callees of `body` the compiled code never branches to (copied in) whose own calls were all copied in
    too, recursively: such a call is no call. Only callees with a known definition (unit or header `inline`)
    and not declared `virtual` qualify, since a virtual or pointer call has no `bl` either."""
    memo: dict[str, bool] = {}

    def leaf(c: str, depth: int = 0) -> bool:
        if c in memo:
            return memo[c]
        memo[c] = False                                     # recursion guard
        ok = c not in targets and c in known and c not in virtual and depth < 8
        if ok and c in bodies:
            for n in CALL_IN.findall(bodies[c]):
                if n in NOT_CALLS or n == c:
                    continue
                if n in targets or n in virtual or (n in known and not leaf(n, depth + 1)):
                    ok = False
                    break
        memo[c] = ok
        return ok

    return {c for c in callee_names(body) if leaf(c)}


def function_truth(name: str, dfn, span, src: str, types, code: bytes, local_functions: set[str],
                   inline_names: set[str], fn_obj=None, names_at: dict | None = None,
                   bodies: dict | None = None) -> dict | None:
    try:
        body = rp.Body(src, span.open, types)
    except (cbody.ParseError, RecursionError, IndexError, ValueError):
        return None
    variables = [v for v in dfn.vars if v.reg is not None]
    if not variables:
        return {"function": name, "vars": [], "features": {}, "unexplained_temps": 0, "note": "no variables"}
    leaf = getattr(inline_names, "leaf", frozenset())
    if fn_obj is not None:
        every = {**getattr(inline_names, "bodies", {}), **(bodies or {})}
        leaf = frozenset(leaf) | inlined_leaf_calls(body, branch_targets(fn_obj, names_at or {}), every,
                                                    local_functions | set(inline_names),
                                                    getattr(inline_names, "virtual", frozenset()))
    vals = rp.values(body, [(v.name, v.kind) for v in variables], leaf)
    end_of_body = span.close
    for v, val in zip(variables, vals):
        if v.name == "this" and v.kind == "param":
            use = rp.implicit_this_end(body, {x.name for x in variables})
            val.end = max(val.end, use if use is not None else val.start)
            val.cross = any(val.start < c.end < val.end for c in rp.real_calls(body, leaf))
        if v.reg == "stack":
            val.cross = False
    rp.special_members(name, vals, span.close)
    bank = {v.name: ("f" if v.reg and v.reg[0] == "f" else "r") for v in variables}
    used = saved_registers(code)
    named = {"r": set(), "f": set()}
    for v in variables:
        if v.reg and v.reg != "stack" and callee_saved(v.reg):
            named[v.reg[0]].add(int(v.reg[1:]))
    unnamed = sum(max(0, len(used[c]) - len(named[c])) for c in ("r", "f"))
    unnamed = max(0, unnamed - sum(1 for v in vals[len(variables):]))      # the hidden parameters added above
    a = rp.rank_rule(vals, bank)
    b, note = rp.behaviour_rule(body, vals, bank, unnamed)
    temps_predicted = sum(1 for v in vals if v.temp and v.cross)
    stack = [v for v in variables if v.reg == "stack" and v.slot is not None]
    decl_index = {v.name: i for i, v in enumerate(variables)}
    actual_order = [v.name for v in sorted(stack, key=lambda v: (v.slot, decl_index[v.name]))]
    pred_order = rp.stack_order(body, [v.name for v in stack], decl_index,
                                                    {v.name: v.size for v in stack}) if stack else None
    feats = source_features(body, src, local_functions, inline_names)
    fn = {"function": name, "vars": [], "features": feats, "unexplained_temps": max(0, unnamed - temps_predicted),
          "unnamed_temps": unnamed, "note": note}
    rows = fn["vars"]
    for v, val in zip(variables, vals):
        cat = category(v.reg, val.cross, v.kind == "local")
        row = {"name": v.name, "kind": v.kind, "actual": cat, "a": a[v.name], "b": b[v.name], "cross": val.cross,
               "temp_pred": val.temp}
        if cat == "stack":
            row.update(cat="stack", a=None, b=None, type=rp.type_text(body, v.name))
            if pred_order is not None and len(set(actual_order)) == len(actual_order) and v.name in pred_order:
                row.update(b=f"slot#{pred_order.index(v.name)}", actual=f"slot#{actual_order.index(v.name)}")
        elif cat == "volatile" and not any(p not in (None, "volatile") for p in (row["a"], row["b"])):
            row["cat"] = "vol"
        else:
            row["cat"] = "reg"
        rows.append(row)
    vol = [r for r in rows if r["cat"] == "vol"]
    for r in rows:
        r["a_ok"] = None if r["a"] is None else r["a"] == r["actual"]
        r["b_ok"] = None if r["b"] is None else r["b"] == r["actual"]
        if r["cat"] == "vol":
            r["a_ok"] = r["b_ok"] = None
            r["scored"] = feats["loop"] and len(vol) >= 2
            r["cls"] = "volatile-loop-perm" if r["scored"] else ""
        else:
            r["scored"] = True
            r["cls"] = classify(r, fn)          # the class this variable belongs to, right or wrong
    return fn


@dataclass
class Job:
    unit: dict
    index: int


def unit_truth(cfg: Config, unit: dict, workdir: Path, addresses: dict[str, int], inline_names: set[str]) -> dict:
    out = {"unit": unit["name"], "status": "ok", "functions": []}
    src_rel = unit.get("metadata", {}).get("source_path", "")
    src_path = cfg.root / src_rel
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
        out.update(status="compile-error", error=log[-300:])
        return out
    obj = objs[0]
    ours, base = Elf.parse(obj.read_bytes()), Elf.parse(base_path.read_bytes())
    of, bf = ours.functions(), base.functions()
    if set(of) != set(bf) or any(of[n].code != bf[n].code for n in bf):
        out["status"] = "code-differs"
        obj.unlink(missing_ok=True)
        return out
    exact = exact_functions(cfg, unit, obj)
    dbg = ours.section(".debug")
    if dbg is None or not exact:
        out["status"] = "no-debug" if dbg is None else "no-exact"
        obj.unlink(missing_ok=True)
        return out
    dwarf = parse_dwarf1(dbg.data, debug_refs(ours))
    src = src_path.read_text(errors="replace")
    types = cbody.type_names(src)
    spans = cbody.functions(src)
    starts = cbody.line_starts(src)
    local_functions = {s.name.split("::")[-1] for s in spans}
    bodies = {s.name.split("::")[-1]: src[s.open + 1:s.close] for s in spans}
    names_at = {(ours.sections[y.shndx].name, y.value): y.name for y in ours.symbols
                if y.type == STT_FUNC and 0 < y.shndx < len(ours.sections)}
    skipped = Counter()
    for name in sorted(exact):
        dfn = find_function(dwarf, name)
        if dfn is None:
            skipped["no-debug-entry"] += 1
            continue
        lines = [ln for _, ln in line_table(ours, name)]
        span = find_span(spans, lines, [v.name for v in dfn.vars], src, starts)
        if span is None:
            skipped["no-source-match"] += 1
            continue
        fn = function_truth(name, dfn, span, src, types, of[name].code, local_functions, inline_names,
                            of[name], names_at, bodies)
        if fn is None:
            skipped["parse-error"] += 1
            continue
        fn["address"] = addresses.get(name)
        out["functions"].append(fn)
    out["exact"] = len(exact)
    out["skipped"] = dict(skipped)
    obj.unlink(missing_ok=True)
    return out


class InlineNames(set):
    """Names declared `inline`, with `leaf`: those whose bodies call nothing."""
    leaf: frozenset = frozenset()
    bodies: dict = {}
    virtual: frozenset = frozenset()


CALL_IN = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
NOT_CALLS = {"if", "while", "for", "switch", "return", "sizeof", "do"}


def header_inline_bodies(cfg: Config) -> dict[str, str]:
    """Bodies of the `inline` functions defined in the project's headers, by name (concatenated per name)."""
    out: dict[str, str] = {}
    pat = re.compile(r"\binline\b[^;{}()]*?\b([A-Za-z_]\w*)\s*\([^;{}]*\)[^;{}]*\{")
    for path in (cfg.root / "include").rglob("*.h"):
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for m in pat.finditer(text):
            depth, k = 1, m.end()
            while k < len(text) and depth:
                depth += {"{": 1, "}": -1}.get(text[k], 0)
                k += 1
            out[m.group(1)] = out.get(m.group(1), "") + text[m.end():k - 1]
    return out


def virtual_names(cfg: Config) -> set[str]:
    """Names declared `virtual` anywhere in the project's headers or sources (called through the vtable)."""
    names: set[str] = set()
    pat = re.compile(r"\bvirtual\b[^;{}()]*?(?:\b([A-Za-z_]\w*)|operator\s*\S+)\s*\(")
    for top in ("include", "src"):
        for path in (cfg.root / top).rglob("*"):
            if path.suffix in (".h", ".hpp", ".c", ".cpp", ".cp"):
                try:
                    names.update(n for n in pat.findall(path.read_text(errors="replace")) if n)
                except OSError:
                    pass
    return names


def leaf_inline_names(cfg: Config) -> set[str]:
    """Names of `inline` functions in the project's headers whose bodies call nothing: copied into
    the caller, they are no call (an inline that calls still crosses)."""
    names: set[str] = set()
    pat = re.compile(r"\binline\b[^;{}()]*?\b([A-Za-z_]\w*)\s*\([^;{}]*\)[^;{}]*\{")
    calls: dict[str, bool] = {}
    for path in (cfg.root / "include").rglob("*.h"):
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for m in pat.finditer(text):
            depth, k = 1, m.end()
            while k < len(text) and depth:
                depth += {"{": 1, "}": -1}.get(text[k], 0)
                k += 1
            body = text[m.end():k - 1]
            has = any(c not in NOT_CALLS for c in CALL_IN.findall(body))
            calls[m.group(1)] = calls.get(m.group(1), False) or has
    return {n for n, has in calls.items() if not has}


def inline_function_names(cfg: Config) -> set[str]:
    """Names declared `inline` in the project's headers (cheap evidence that a call is inlined)."""
    names: set[str] = set()
    pat = re.compile(r"\binline\b[^;{}()]*?\b([A-Za-z_]\w*)\s*\(")
    for path in (cfg.root / "include").rglob("*.h"):
        try:
            names.update(pat.findall(path.read_text(errors="replace")))
        except OSError:
            pass
    return names


# ---- aggregation and report ---------------------------------------------------------------------------------

def tally(units: list[dict]) -> dict:
    pred = {p: {"n": 0, "ok": 0, "unpredicted": 0} for p in ("a", "b", "a_on_b")}
    stack = {"n": 0, "ok": 0, "unpredicted": 0}
    by_class: dict[str, dict] = defaultdict(lambda: {"vars": 0, "a_ok": 0, "b_ok": 0, "b_unpredicted": 0,
                                                       "wrong_b": 0, "examples": []})
    fns = {"functions": 0, "scored": 0, "a_ok": 0, "b_ok": 0, "b_unpredicted": 0}
    for u in units:
        for fn in u.get("functions", []):
            fns["functions"] += 1
            rows = [r for r in fn["vars"] if r["cat"] == "reg"]
            if rows:
                fns["scored"] += 1
                fns["a_ok"] += all(r["a_ok"] is True for r in rows)
                fns["b_ok"] += all(r["b_ok"] is True for r in rows)
                fns["b_unpredicted"] += any(r["b"] is None for r in rows) and not any(r["b_ok"] is False for r in rows)
            for r in fn["vars"]:
                if r["cat"] == "stack":
                    stack["n"] += 1
                    stack["ok"] += r["b_ok"] is True
                    stack["unpredicted"] += r["b"] is None
                if r["cat"] == "reg":
                    pred["a"]["n"] += 1
                    pred["a"]["ok"] += r["a_ok"] is True
                    pred["a"]["unpredicted"] += r["a"] is None
                    pred["b"]["n"] += 1
                    pred["b"]["ok"] += r["b_ok"] is True
                    pred["b"]["unpredicted"] += r["b"] is None
                    if r["b"] is not None:
                        pred["a_on_b"]["n"] += 1
                        pred["a_on_b"]["ok"] += r["a_ok"] is True
                cls = r.get("cls", "")
                if cls:
                    c = by_class[cls]
                    c["vars"] += 1
                    c["a_ok"] += r["a_ok"] is True
                    c["b_ok"] += r["b_ok"] is True
                    c["b_unpredicted"] += r["b"] is None
                    c["wrong_b"] += r["b_ok"] is False
                    miss = r["b_ok"] is not True if r["cat"] == "stack" else (r["a_ok"] is not True or r["b_ok"] is not True)
                    if miss and len(c["examples"]) < 3:
                        c["examples"].append({"address": fn.get("address"), "function": fn["function"], "variable": r["name"],
                                              "actual": r["actual"], "a": r["a"], "b": r["b"]})
    return {"predictors": pred, "stack": stack, "classes": dict(by_class), "functions": fns}


def pct(a: int, n: int) -> str:
    return f"{100.0 * a / n:.1f}%" if n else "-"


def render(doc: dict) -> str:
    t, meta = doc["totals"], doc["meta"]
    p = t["predictors"]
    lines = [f"# Register truth ({meta['date']})", "",
             f"{meta['units_ok']} of {meta['units_run']} units read ({meta['mode']}); {t['functions']['functions']} exact functions with "
             f"source and debug info; {p['a']['n']} variables scored (callee-saved or temporary, either side). "
             f"Skipped units: {meta['skipped_units']}. Skipped functions: {meta['skipped_functions']}. "
             f"Runtime {meta['seconds']:.0f} s, {meta['jobs']} jobs.", "",
             "## Accuracy per predictor", "",
             "| predictor | variables | correct | accuracy | not predicted |", "|---|---|---|---|---|"]
    names = {"a": "(a) rank rule", "b": "(b) behaviour rules", "a_on_b": "(a) on (b)'s predicted subset"}
    for k in ("a", "b", "a_on_b"):
        x = p[k]
        lines.append(f"| {names[k]} | {x['n']} | {x['ok']} | {pct(x['ok'], x['n'])} | {x['unpredicted']} |")
    st = t["stack"]
    lines += ["", f"Stack slots (b only, named stack variables whose sizes are known): {st['ok']}/{st['n']} right "
              f"({pct(st['ok'], st['n'])}), {st['unpredicted']} not predicted (unknown type size).", ""]
    f = t["functions"]
    lines += [f"Functions (all scored variables right): (a) {f['a_ok']}/{f['scored']} = {pct(f['a_ok'], f['scored'])}, "
              f"(b) {f['b_ok']}/{f['scored']} = {pct(f['b_ok'], f['scored'])}, (b) has an unpredicted variable and no wrong one in "
              f"{f['b_unpredicted']}.", "",
              "## Classes (each variable has the class its failure would get; examples are misses)", "",
              "| class | variables | (a) right | (b) right | (b) wrong | (b) not predicted | examples (address function variable: actual / a / b) |",
              "|---|---|---|---|---|---|---|"]
    for cls in sorted(t["classes"], key=lambda c: -t["classes"][c]["vars"]):
        c = t["classes"][cls]
        ex = "<br>".join(f"{e['address'] and hex(e['address'])} {e['function']} {e['variable']}: {e['actual']} / {e['a']} / {e['b']}"
                         for e in c["examples"])
        lines.append(f"| {cls} | {c['vars']} | {pct(c['a_ok'], c['vars'])} | {pct(c['b_ok'], c['vars'])} | {c['wrong_b']} | "
                     f"{c['b_unpredicted']} | {ex} |")
    lines += ["", "Classes are heuristic labels from evidence in the source and the compiled function (the unexplained "
              "temporaries, inline calls, loop indexing, repeated loads, struct locals); `volatile-loop-perm` marks "
              "volatile-bank variables in loop functions, which neither predictor covers (never counted right); `local-temp-regime` is a "
              "named local whose temporary status the behaviour rule got wrong; `crossing-misjudged` is a variable the source's "
              "call positions put in the wrong bank (callee-saved vs volatile) with no inline call to blame."]
    return "\n".join(lines) + "\n"


def stratified(units: list[dict], n: int, seed: int) -> list[dict]:
    """About n units, spread round-robin over the first two path components, seeded."""
    import random
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for u in units:
        groups["/".join(u["name"].split("/")[:3])].append(u)
    for g in groups.values():
        rng.shuffle(g)
    out: list[dict] = []
    keys = sorted(groups)
    while len(out) < n and any(groups.values()):
        for k in keys:
            if groups[k] and len(out) < n:
                out.append(groups[k].pop())
    return out


def run_truth(cfg: Config, unit: str | None, all_units: bool, jobs: int, sample: int, seed: int) -> int:
    from .assets import ensure
    from .jobs import resolve
    from .matching import resolve_unit
    ensure(cfg)
    try:
        units = [resolve_unit(cfg, unit)] if unit else [u for u in load_objdiff_units(cfg) if u.get("base_path")]
    except MatchError as e:
        print(f"error: {e}")
        return 2
    units = [u for u in units if u.get("metadata", {}).get("source_path")]
    mode = f"unit {unit}" if unit else "all units"
    if not unit and sample and sample < len(units):
        units = stratified(units, sample, seed)
        mode = f"stratified sample of {len(units)} units, seed {seed}"
    jobs = resolve(jobs, cfg, cap=4)
    addresses = {s.name: s.address for s in read_symbols(cfg.root / cfg.platform_options["symbols"]) if s.type == "function"}
    inline_names = InlineNames(inline_function_names(cfg))
    inline_names.leaf = frozenset(leaf_inline_names(cfg))
    inline_names.bodies = header_inline_bodies(cfg)
    inline_names.virtual = frozenset(virtual_names(cfg))
    base = cfg.root / OUT / "t"
    t0 = time.time()

    def one(args):
        i, u = args
        try:
            return unit_truth(cfg, u, base / str(i), addresses, inline_names)
        except Exception as e:             # one unit must not stop the run
            return {"unit": u["name"], "status": "error", "error": f"{type(e).__name__}: {e}", "functions": []}

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(one, enumerate(units)))
    doc = {"meta": {"date": time.strftime("%Y-%m-%d"), "mode": mode, "units_run": len(units),
                    "units_ok": sum(r["status"] == "ok" for r in results), "jobs": jobs, "seconds": time.time() - t0,
                    "skipped_units": dict(Counter(r["status"] for r in results if r["status"] != "ok")),
                    "skipped_functions": dict(sum((Counter(r.get("skipped", {})) for r in results), Counter()))},
           "totals": tally(results), "units": results}
    out = cfg.root / OUT
    out.mkdir(parents=True, exist_ok=True)
    (out / "truth.json").write_text(json.dumps(doc, indent=1))
    (out / "truth.md").write_text(render(doc))
    print(render(doc).split("\n## Failure")[0])
    print(f"wrote {OUT}/truth.json and {OUT}/truth.md")
    return 0
