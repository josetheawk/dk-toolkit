"""dk dataset real-nbhd: the enumerator census over the real near-miss queue and its compiler-labelled pairs
(council T43 ideas 2 and 3, T44 B).

  ./dk dataset real-nbhd [--out DIR] [--budget 100] [--jobs 4] [--until HH:MM] [--limit N] [--function F]

For every near-* function of `route` (near-register, near-shape, near-scheduling), from its source now: compile
the start privately (`-sym on`, objdiff), then every candidate of every rewrite-table row (rewrites.RULES, the
whole definition, one site per move; shapesynth.Fn.moves), rules in the symptom table's order and taken
round-robin so each row gets a share; depth 2 from the best <= 3 strictly improved children; at most --budget
compiles per function (depth 1 first, up to --depth1 of them). States are compared in the pure objdiff tiers
(RL3): exact > fewer shape rows > the same shape rows and fewer rows. No oracle, no rank solver.

A move that writes `__typeof__` locals or placeholder names is made natural first (debug-info types, names from
the value: shapesynth's rename) and compiled again; it is kept only when the code is the same.

Outputs (DIR, default llm-data/datasets/real-nbhd/v0):
- census.jsonl: one row per function: start/best percent and rows, outcome (exact / improves / no-gain / skip),
  compiles, frozen tag;
- pairs.jsonl: shape-fix-schema records (before, after definitions; before's rows and percent, diff_class, the
  move) for every compiled (parent, child) whose tiers differ, in the improving direction ("forward" parent ->
  better child, "reverse" worse child -> parent); review() of the step must not reject; frozen eval functions
  (llmfixes.frozen, route's `frozen`) never get a record. No model text: project source and compiler output only;
- closes.jsonl: judged exact results (close's final check: real types, hand-linked, nothing else worse, no
  device) with their review() category: candidates only, never applied here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from ...core.command import OK, USAGE
from ...core.config import Config

NEAR = ("near-register", "near-shape", "near-scheduling")
PER_RULE = 40
NEW_FUNCTIONS = ("inline-helper", "helper-params")     # moves that change code outside the definition: no pair
WIDTH2 = 3


# ---- pure helpers (tested) ---------------------------------------------------------------------------------

def tier_key(percent: float, shape_rows: int, rows: int) -> tuple:
    """Smaller is better: exact first, then fewer shape rows, then fewer differing rows."""
    return (percent != 100.0, shape_rows, rows)


def better(a: tuple, b: tuple) -> bool:
    return a < b


def round_robin(moves: dict[str, list], order: list[str]) -> list[tuple[str, object]]:
    """[(rule, move)] taking one move of each rule in `order` per pass (rules missing from order come last)."""
    rules = [r for r in order if r in moves] + [r for r in moves if r not in order]
    out, k = [], 0
    while True:
        got = False
        for r in rules:
            if k < len(moves[r]):
                out.append((r, moves[r][k]))
                got = True
        if not got:
            return out
        k += 1


def direction(parent: tuple, child: tuple) -> str | None:
    """"forward" (the child is better), "reverse" (the parent is better), None (the same tier)."""
    if better(child, parent):
        return "forward"
    if better(parent, child):
        return "reverse"
    return None


def diff_class(shape_rows: int, rows: int) -> str:
    return "register" if rows and not shape_rows else "shape"


# ---- one function ------------------------------------------------------------------------------------------

class State:
    def __init__(self, text: str, e, depth: int, rule: str = "", label: str = "", parent=None):
        self.text, self.e, self.depth, self.rule, self.label, self.parent = text, e, depth, rule, label, parent

    @property
    def key(self) -> tuple:
        return tier_key(self.e.percent, self.e.shape_rows, self.e.rows)


def census_one(root: str, row: dict, budget: int, depth1: int, work: str, frozen_why: str | None) -> dict:
    """The census of one function (runs in a worker process): its row, pairs and closes."""
    from ...core.config import load
    from . import rewrites
    from .closecmd import Closer, Options
    from .judge import replace_typeof, review
    from .matching import resolve_unit
    from .shapesynth import Fn
    os.chdir(root)
    cfg = load()
    fn_name = row["function"]
    out = {"unit": row["unit"], "function": fn_name, "class": row["class"], "frozen": frozen_why, "compiles": 0,
           "verify_compiles": 0}
    t0 = time.time()
    try:
        unit = resolve_unit(cfg, row["unit"])
    except Exception as exc:                             # noqa: BLE001
        return {"row": {**out, "outcome": "skip", "why": f"unit: {exc}"[:200]}, "pairs": [], "closes": []}
    tag = hashlib.sha1(f"{unit['name']}:{fn_name}".encode()).hexdigest()[:8]
    wdir = Path(work) / f"{os.getpid()}_{tag}"
    closer = Closer(cfg, unit, fn_name, Options(moves="table", jobs=1), log=lambda s: None)
    closer.work = wdir
    closer.keep_rows = True
    src = closer.source.read_text()
    fn = Fn(cfg, unit, fn_name, src, wdir / "fn")
    slot = [0]

    def evaluate(text: str):
        slot[0] += 1
        return closer.evaluate(text, slot[0] if slot[0] > 1 else 0)

    pairs, closes, seen = [], [], {src}
    try:
        e0 = evaluate(src)
        out["compiles"] += 1
        if e0 is None:
            return {"row": {**out, "outcome": "skip", "why": "the start does not compile"}, "pairs": [], "closes": []}
        if e0.percent == 100.0:
            return {"row": {**out, "outcome": "skip", "why": "exact already", "start": 100.0}, "pairs": [], "closes": []}
        fn.members = closer.members
        start = State(src, e0, 0)
        out.update(start=e0.percent, start_shape_rows=e0.shape_rows, start_rows=e0.rows)
        order = [r.name for r in rewrites.ranked(e0.sym)]
        names = [r.name for r in rewrites.RULES]
        before_unit = None
        best = start
        improved: list[State] = []

        def natural(parent: State, text: str, e):
            """(text, eval) with real types and natural names when the move wrote placeholders; None: drop."""
            new = text
            if "__typeof__" in new:
                body = closer.body(new, e.first_line)
                if body is None:
                    return None
                new, missing = replace_typeof(new, body, dict(e.types))
                if missing:
                    return None
            new = fn.rename_new(parent.text, new)
            if new == text:
                return text, e
            e2 = evaluate(new)
            out["verify_compiles"] += 1
            if e2 is None or e2.code != e.code:
                return None
            return new, e2

        def expand(parent: State, cap: int) -> list[State]:
            moves = fn.moves(parent.text, rules=names, per_rule=PER_RULE)
            kids = []
            for rule, (label, new) in round_robin(moves, order):
                if len(kids) >= cap or out["compiles"] >= budget:
                    break
                if new in seen:
                    continue
                seen.add(new)
                e = evaluate(new)
                out["compiles"] += 1
                if e is None:
                    continue
                kid = State(new, e, parent.depth + 1, rule, label, parent)
                kids.append(kid)
                nat = natural(parent, new, e)
                if nat is not None:
                    kid.text, kid.e = nat
                    record(parent, kid)
            return kids

        def record(parent: State, kid: State) -> None:
            d = direction(parent.key, kid.key)
            if d is None or frozen_why or kid.rule in NEW_FUNCTIONS:
                return
            before, after = (parent, kid) if d == "forward" else (kid, parent)
            b_def, a_def = fn.defn(before.text), fn.defn(after.text)
            if not b_def or not a_def:
                return
            cat, flags = review(before.text, after.text, fn_name)
            if cat == "reject":
                return
            pairs.append({"schema": 1, "source": "real-nbhd", "unit": unit["name"], "function": fn_name,
                          "path": unit["metadata"]["source_path"], "asm_shape": row.get("asm_shape"),
                          "before": b_def, "after": a_def, "before_percent": before.e.percent,
                          "after_percent": after.e.percent, "before_shape_rows": before.e.shape_rows,
                          "before_rows": before.e.rows, "after_shape_rows": after.e.shape_rows, "after_rows": after.e.rows,
                          "diff_class": diff_class(before.e.shape_rows, before.e.rows), "rows": before.e.view,
                          "move": {"rule": kid.rule, "label": kid.label, "depth": kid.depth, "direction": d},
                          "review": [cat, flags], "oracle": None})

        kids = expand(start, depth1)
        improved = sorted([k for k in kids if better(k.key, start.key)], key=lambda s: s.key)
        out["depth1"] = len(kids)
        for k in kids:
            best = k if better(k.key, best.key) else best
        firsts = improved[:WIDTH2]
        for i, p in enumerate(firsts):
            if out["compiles"] >= budget or p.e.percent == 100.0:
                continue
            cap = (budget - out["compiles"]) // (len(firsts) - i)
            for k in expand(p, cap):
                best = k if better(k.key, best.key) else best
                if better(k.key, start.key):
                    improved.append(k)
        out.update(best=best.e.percent, best_shape_rows=best.e.shape_rows, best_rows=best.e.rows,
                   improved_states=len(improved), best_depth=best.depth,
                   best_moves=[s.rule for s in _chain(best)])
        if best.e.percent == 100.0:
            if before_unit is None:
                before_unit, _p, _err = closer.unit_score(src, "before")
            verdict, final = (closer.final_check(best.e, src, before_unit) if before_unit is not None
                              else (None, best.text))
            ok = bool(verdict and verdict.ok)
            rv = list(verdict.review) if verdict is not None and verdict.review else None
            out.update(outcome="exact" if ok else "improves", judged=ok,
                       judge=(verdict.reasons[:4] if verdict is not None else ["the start does not score"]), review=rv)
            if ok:
                closes.append({"unit": unit["name"], "function": fn_name, "path": unit["metadata"]["source_path"],
                               "start": e0.percent, "moves": [(s.rule, s.label) for s in _chain(best)],
                               "review": rv, "frozen": frozen_why, "before": fn.defn(src), "after": fn.defn(final)})
        else:
            out["outcome"] = "improves" if better(best.key, start.key) else "no-gain"
    except Exception as exc:                             # noqa: BLE001 (one function never stops the census)
        out.update(outcome="skip", why=f"{type(exc).__name__}: {exc}"[:200])
    finally:
        shutil.rmtree(wdir, ignore_errors=True)
    out["secs"] = round(time.time() - t0, 1)
    return {"row": out, "pairs": pairs, "closes": closes}


def _chain(s: State) -> list[State]:
    out = []
    while s is not None and s.parent is not None:
        out.append(s)
        s = s.parent
    return out[::-1]


# ---- the run -----------------------------------------------------------------------------------------------

def queue(cfg: Config) -> list[dict]:
    from .route import table
    return [r for r in table(cfg) or [] if r.get("class") in NEAR]


def run(cfg: Config, args: argparse.Namespace) -> int:
    from . import llmfixes
    from .llmtry import _until, asm_entries, asm_shape, data_dir
    out = Path(args.out).expanduser() if args.out else data_dir(cfg) / "datasets" / "real-nbhd" / "v0"
    out.mkdir(parents=True, exist_ok=True)
    work = Path(args.work).expanduser() if args.work else out / "work"
    work.mkdir(parents=True, exist_ok=True)
    rows = queue(cfg)
    if args.function:
        rows = [r for r in rows if r["function"] in args.function]
    if args.limit:
        rows = rows[:args.limit]
    entries = asm_entries(cfg)
    fz = llmfixes.frozen(cfg)
    for r in rows:
        e = entries.get(r["function"])
        r["asm_shape"] = asm_shape(e.text) if e is not None else None
    done = set()
    census = out / "census.jsonl"
    if args.resume and census.is_file():
        done = {json.loads(x)["function"] for x in census.read_text().splitlines() if x.strip()}
    todo = [r for r in rows if r["function"] not in done]
    until = _until(args.until)
    print(f"{len(todo)} of {len(rows)} near-* function(s) to do; budget {args.budget} compiles each, "
          f"{args.jobs} at once; out {out}", flush=True)

    def why_frozen(r: dict) -> str | None:
        if r.get("frozen"):
            return f"route ({r['frozen']})"
        if r["unit"] in fz["units"]:
            return f"unit ({fz['units'][r['unit']]})"
        if r["function"] in fz["functions"]:
            return f"function ({fz['functions'][r['function']]})"
        if r.get("asm_shape") and r["asm_shape"] in fz["shapes"]:
            return f"asm shape ({fz['shapes'][r['asm_shape']]})"
        return None

    files = {k: (out / f"{k}.jsonl").open("a") for k in ("census", "pairs", "closes")}
    n = 0
    try:
        with ProcessPoolExecutor(max_workers=max(1, min(args.jobs, 4))) as pool:
            futs = {}
            for r in todo:
                futs[pool.submit(census_one, str(cfg.root), r, args.budget, args.depth1, str(work), why_frozen(r))] = r
            for f in as_completed(futs):
                res = f.result()
                for p in res["pairs"]:
                    if llmfixes.leak(p, fz):                 # the hash filter on top of the function filter
                        continue
                    files["pairs"].write(json.dumps(p) + "\n")
                for c in res["closes"]:
                    files["closes"].write(json.dumps(c) + "\n")
                files["census"].write(json.dumps(res["row"]) + "\n")
                for fh in files.values():
                    fh.flush()
                n += 1
                row = res["row"]
                print(f"[{n}/{len(todo)}] {row['function']}: {row.get('outcome')} {row.get('start')} -> {row.get('best')} "
                      f"({row.get('compiles')} compiles, {len(res['pairs'])} pairs, {row.get('secs')} s)", flush=True)
                if until and time.time() >= until:
                    print("--until reached: stopping (resume with --resume)", flush=True)
                    for g in futs:
                        g.cancel()
                    break
    finally:
        for fh in files.values():
            fh.close()
    return OK


def configure(sub) -> None:
    s = sub.add_parser("real-nbhd", help="enumerator census over route's near-* queue: every table move (depth 1-2, "
                       "a compile budget per function) compiled and scored; compiler-labelled real-neighbourhood "
                       "pairs, the tools-alone ceiling and judged exact candidates -> llm-data/datasets/real-nbhd/v0")
    s.add_argument("--out", default="", help="output directory (default llm-data/datasets/real-nbhd/v0)")
    s.add_argument("--work", default="", help="private compile directory (default OUT/work; one per process)")
    s.add_argument("--budget", type=int, default=100, help="compiles per function (default 100)")
    s.add_argument("--depth1", type=int, default=70, help="depth-1 compiles at most (default 70)")
    s.add_argument("--jobs", type=int, default=4, help="functions at once, one compile each (default and at most 4)")
    s.add_argument("--until", default="", help="stop at HH:MM (--resume goes on)")
    s.add_argument("--resume", action="store_true", help="skip functions already in census.jsonl")
    s.add_argument("--limit", type=int, default=0, help="first N functions only (a test)")
    s.add_argument("--function", action="append", default=[], help="only this function (repeatable)")
