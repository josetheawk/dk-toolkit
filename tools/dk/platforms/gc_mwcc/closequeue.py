"""dk close --queue: the CPU enumerator over `dk route`'s open queue, before any model or agent grind.

Every open function that has a near miss (>= --min %), is not frozen (the LoRA-2 gate and near-samples:
gate.py, `route`'s `frozen` worker), not blocked, not in a benchmark or held-out unit and not
being edited in another worktree, gets `close` with the rewrite table and a compile budget. Functions of
<= --small bytes go first (closest first), then the rest up to --max-size. A judged-exact result becomes a
candidate in llm-data/candidates/<date>/ (`llm-grind`'s format, route `close-table`), never applied:
`llm-review <date>` and `llm-land ID` take it from there. The state file makes a night resumable; the
frontier memo (build/dk/close/<fn>.json) makes the next night skip what this one compiled.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from ...core.command import OK, USAGE
from ...core.config import Config

STATE = "close-queue-state.json"
SKIP_WORKERS = ("frozen", "blocked")
SKIP_CLASSES = ("sinit", "unit-data", "fresh")


def queue_rows(rows: list[dict], frozen, exclude_units: set[str], extra_skip: set[tuple[str, str]],
               minimum: float, small: int, max_size: int) -> list[dict]:
    """The work list, in order: <= small bytes first (closest first, then smaller), then the rest."""
    out = []
    for r in rows:
        key = (r["unit"], r["function"])
        if r.get("worker") in SKIP_WORKERS or r.get("frozen") or key in frozen or key in extra_skip \
                or r.get("class") in SKIP_CLASSES or r["unit"] in exclude_units:
            continue
        if r["function"].startswith(("@", "__sinit")) or not minimum <= (r.get("fuzzy") or 0.0) < 100.0:
            continue
        if not 8 <= r["size"] <= max_size:
            continue
        out.append(r)
    out.sort(key=lambda r: (r["size"] > small, -(r.get("fuzzy") or 0.0), r["size"], r["address"]))
    return out


def read_skip(paths: list[str]) -> set[tuple[str, str]]:
    """[unit, fn] pairs from skip files: a list of pairs, {"functions": [...]} or records with unit/function."""
    from .closecmd import read_pairs
    out: set[tuple[str, str]] = set()
    for p in paths:
        path = Path(p).expanduser()
        if path.is_file():
            out |= set(read_pairs(json.loads(path.read_text())))
    return out


def run_queue(cfg: Config, args: argparse.Namespace, opts) -> int:
    from . import route
    from .closecmd import close_one
    from .gate import frozen_keys
    from .llmgrind import busy_sources, write_candidate
    from .llmtry import data_dir, never_train
    from .matching import MatchError, resolve_unit
    rows = route.table(cfg, 80.0, 2, None, "")
    if rows is None:
        print("error: no objdiff report yet (./dk build)")
        return USAGE
    ddir = data_dir(cfg)
    busy = busy_sources(cfg)
    frozen = frozen_keys(cfg)
    thaw = tuple(t for t in (getattr(args, "thaw", "") or "").split(",") if t)
    if thaw:                                              # CPU only; datasets keep filtering them (gate.never)
        from . import unlock
        rows = unlock.annotate(rows, route.unlock_ranking(cfg, rows, thaw))
        from ...core.frozen import FrozenMap
        frozen = FrozenMap({k: v for k, v in frozen.items() if v not in thaw})
    items = queue_rows(rows, frozen, never_train(cfg)[0], read_skip(args.skip_list or []),
                       args.min, args.small, args.max_size)
    first = read_skip(getattr(args, "first", None) or [])
    if first:                                             # e.g. `regs --residue`'s register-order-only functions
        items = [r for r in items if (r["unit"], r["function"]) in first] + \
                [r for r in items if (r["unit"], r["function"]) not in first]
    if args.sample:
        items = items[:args.sample]
    state_path = ddir / STATE
    state = json.loads(state_path.read_text()) if args.resume and state_path.is_file() else {}
    if not state:
        state = {"run": f"close-queue-{time.strftime('%Y%m%d-%H%M')}", "date": time.strftime("%Y-%m-%d"), "done": {}}
    cdir = ddir / "candidates" / state["date"]
    log_path = ddir / f"{state['run']}.jsonl"
    until = 0.0
    if args.until:
        hh, mm = (int(x) for x in args.until.split(":"))
        t = time.localtime()
        until = time.mktime((t.tm_year, t.tm_mon, t.tm_mday, hh, mm, 0, 0, 0, -1))
        if until <= time.time():
            until += 86400
    print(f"{len(items)} function(s) in the queue ({len(busy)} source(s) busy elsewhere; frozen, blocked, "
          f"benchmark and held-out units left out); run {state['run']}, candidates to {cdir}", flush=True)
    found = 0
    with log_path.open("a") as fh:
        for k, r in enumerate(items, 1):
            key = f"{r['unit']}:{r['function']}"
            if key in state["done"]:
                continue
            if until and time.time() >= until:
                print("stop time reached", flush=True)
                break
            if (ddir / "close-queue.stop").exists():
                print("stop file found (llm-data/close-queue.stop)", flush=True)
                break
            try:
                unit = resolve_unit(cfg, r["unit"])
            except MatchError as e:
                state["done"][key] = {"result": "error", "why": str(e)}
                continue
            if k % 10 == 0:
                busy = busy_sources(cfg)                         # another agent may start on a source tonight
            if unit["metadata"]["source_path"] in busy:
                print(f"[{k}/{len(items)}] {r['function']}: source busy in another worktree, skipped", flush=True)
                continue
            src = cfg.root / unit["metadata"]["source_path"]
            before_text = src.read_text()
            print(f"[{k}/{len(items)}] {r['unit']} {r['function']} {r['size']} B {r.get('fuzzy', 0):.2f}% [{r['class']}]",
                  flush=True)
            res = close_one(cfg, unit, r["function"], opts, False, log=lambda *_: None)
            out = {"kind": "close-queue", "unit": r["unit"], "function": r["function"], "size": r["size"],
                   "class": r["class"], "start": res.start, "best": res.best, "exact": res.exact, "compiles": res.compiles,
                   "secs": round(res.secs, 1), "labels": res.labels, "rules": res.rules, "rejected": res.reasons[:3],
                   "error": res.error, "time": time.strftime("%Y-%m-%d %H:%M")}
            if res.exact and res.text:
                if src.read_text() != before_text:                # the source moved under us: not a clean candidate
                    out["error"] = "source changed during the search"
                else:
                    out["id"] = write_candidate(cdir, {
                        "unit": r["unit"], "source": cfg.rel(src), "function": r["function"], "size": r["size"],
                        "kind": "register" if r["class"] == "near-register" else "shape", "route": "close-table",
                        "before": res.start, "after": 100.0, "labels": res.labels, "rules": res.rules,
                        "before_text": before_text, "text": res.text, "judge": [], "time": out["time"], "run": state["run"],
                        **({"review": list(res.review)} if res.review else {})})
                    out["review"] = res.review[0] if res.review else None
                    found += 1
            print(f"    {'EXACT (' + str(out.get('review')) + ') -> candidate ' + out['id'] if out.get('id') else out['error'] or f'best {res.best:.2f}%'}"
                  f" ({res.compiles} compiles, {res.secs:.0f} s)", flush=True)
            fh.write(json.dumps(out) + "\n")
            fh.flush()
            state["done"][key] = {x: out.get(x) for x in ("exact", "best", "id", "compiles", "error")}
            tmp = state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, indent=1))
            os.replace(tmp, state_path)
    done = [v for v in state["done"].values() if v.get("exact") is not None]
    print(f"queue: {sum(1 for v in done if v.get('exact'))} exact of {len(done)} worked this run ({found} new now); "
          f"log {log_path}; review with ./dk llm-review {state['date']}", flush=True)
    return OK
