"""The frozen eval sets H-claude-20, H-giant-40, repair-dev, repair-final (llm-data/eval/*.json).

Function-level sets in the gate's format ({"functions": [{unit, function, ...}]}); `gate.set_files` picks up
everything in llm-data/eval/, so llm-grind, route, close --queue and every dataset/leak filter bar them.
Structured fields only (no prose)."""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from ...core.config import Config


def claude_blocked(cfg: Config) -> tuple:
    """Open functions Claude helpers attempted and logged a blocker for, by address (hex, no 0x):
    [platform.eval] claude_blocked in dk.toml."""
    return tuple(str(a) for a in ((cfg.platform_options.get("eval") or {}).get("claude_blocked") or ()))


def top3_share(v: dict) -> float:
    d = sorted((r.get("differing", 0) for r in v.get("regions", [])), reverse=True)
    return sum(d[:3]) / sum(d) if sum(d) else 0.0


def stratified(items: list, key, n: int, rng: random.Random) -> list:
    items = sorted(items, key=key)
    if n >= len(items):
        return items
    step = len(items) / n                       # evenly spaced over the sorted key, jittered inside each bin
    return [items[min(len(items) - 1, int(i * step + rng.random() * step))] for i in range(n)]


def freeze(cfg: Config, args: argparse.Namespace) -> int:
    from .llmtry import data_dir, load_split
    d = data_dir(cfg)
    out = d / "eval"
    names = ("h-claude-20", "h-giant-40", "repair-dev", "repair-final")
    if any((out / f"{n}.json").exists() for n in names) and not args.force:
        print(f"error: {out} already holds frozen sets (--force voids results measured on them)")
        return 2
    rng = random.Random(args.seed)
    from .gate import frozen_keys
    held = frozen_keys(cfg)
    route = json.loads(__import__("subprocess").run(["./dk", "route", "--json"], cwd=cfg.root, capture_output=True,
                                                    text=True).stdout)["rows"]
    open_by = {r["function"]: r for r in route}
    # H-claude-20
    claude = []
    blocked = claude_blocked(cfg)
    for a in blocked:
        r = open_by.get("fn_" + a)
        if r:
            claude.append({"unit": r["unit"], "function": r["function"], "size": r["size"], "class": r["class"],
                           "also_held_by": held.get((r["unit"], r["function"]), "")})
    # H-giant-40
    cen = Path(args.census) if args.census else cfg.root / "build/dk/regions/census.json"
    if not cen.is_file():
        cen = d.parent / cfg.name / "build/dk/regions/census.json"     # the main checkout
    funcs = json.loads(cen.read_text())["functions"]
    giants = []
    for k, v in funcs.items():
        unit, _, fn = k.partition("::")
        r = open_by.get(fn)
        if top3_share(v) >= 0.9 and r and (unit, fn) not in held and fn not in {c["function"] for c in claude}:
            giants.append({"unit": unit, "function": fn, "size": r["size"], "percent": v.get("percent"),
                           "regions": len(v["regions"]), "top3_share": round(top3_share(v), 3)})
    n_giants_all = len(giants)
    giants = stratified(giants, lambda g: g["size"], 40, rng)
    # repair-27 -> 13 dev / 14 final, stratified by the changes each fix needs (diff rows of the wrong start)
    rep = json.loads((d / "repair-eval.json").read_text())["functions"]
    fin = set(load_split(cfg)["final"])
    rows = [{**r, "changes": r.get("start_rows") if isinstance(r.get("start_rows"), int) else len(r.get("start_rows") or []),
             "before": r["start_code"], "after": r["exact"]}      # the full repair-eval record, so llm-bench run --bench works
            for r in rep if r["function"] in fin]
    rows.sort(key=lambda r: (r["changes"], r["function"]))
    dev, final = [], []
    for i, r in enumerate(rows):                # alternate down the sorted list: both halves span the change counts
        (dev if i % 2 == (1 if len(rows) % 2 == 0 else 1) and len(dev) < 13 else final).append(r)
    if len(dev) != 13:
        dev, final = rows[1::2][:13], [r for r in rows if r not in rows[1::2][:13]]
    sets = {"h-claude-20": claude, "h-giant-40": giants, "repair-dev": dev, "repair-final": final}
    out.mkdir(exist_ok=True)
    for n, fs in sets.items():
        (out / f"{n}.json").write_text(json.dumps({"set": n, "seed": args.seed, "functions": fs}, indent=1) + "\n")
        print(f"  {n}: {len(fs)} function(s)")
    print(f"giants eligible {n_giants_all}; repair rows {len(rows)}; claude resolved {len(claude)}/{len(blocked)} open")
    return 0
