"""dk card: one compact block per hard function, everything a closer needs to read and nothing else.

For each function: its unit, size and score; the score with the target's colours forced (oracle) and every row
still different after that, classed, with where each side's branch lands (residue.py); with `--rules`, the
order solver's broken rules and what made each temporary (`regs --why-temp`); what `close` has already tried
(its frontier memo); and only the source lines the rows point at, numbered, with a little context.

`--batch N` picks N functions from the residue census files (`dk night`'s and `regs --residue --class`'s):
those that are not register order only and have the fewest rows left, the cases a reader fixes fastest.
Functions in `--skip` files and ones with a card from the last 20 hours are left out, so batches move on.
Cards go to stdout and to build/dk/cards/<date>-<n>.md.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ...core.command import OK, USAGE, Command, hint
from ...core.config import Config

CONTEXT = 2                  # source lines around each pointed-at line
MAX_ROWS = 10
RECENT_HOURS = 20


def census_files(cfg: Config) -> list[Path]:
    from .objcache import main_root
    roots = {Path(cfg.root), main_root(cfg)}
    out = []
    for r in roots:
        out += sorted((r / "build" / "dk" / "night").glob("residue-*.json"))
        out += [p for p in (r / "build" / "dk" / "regs" / "residue.json",) if p.is_file()]
    for wt in (main_root(cfg) / "worktrees").glob("*/build/dk/night/residue-*.json"):
        out.append(wt)
    return out


def pick(cfg: Config, n: int, skip: set[tuple[str, str]]) -> list[dict]:
    """The n functions with the fewest residue rows that are not register order only (newest census wins)."""
    seen: dict[tuple[str, str], dict] = {}
    for p in sorted(census_files(cfg), key=lambda p: p.stat().st_mtime):
        try:
            for r in json.loads(p.read_text()):
                if "forced" in r and not r.get("error"):
                    seen[(r["unit"], r["function"])] = r
        except (OSError, ValueError):
            continue
    rows = [r for k, r in seen.items() if k not in skip and r.get("rows")
            and r["forced"] < 100.0 and not r.get("forced_linked")]
    swap_only = lambda r: all(x["class"] == "OPERAND-SWAP" for x in r["rows"])   # value order, not text (tool-gaps #14)
    rows.sort(key=lambda r: (swap_only(r), len(r["rows"]), -r["forced"], r["function"]))
    return rows[:n]


def recent_cards(cfg: Config) -> set[tuple[str, str]]:
    out = set()
    d = Path(cfg.root) / "build" / "dk" / "cards"
    for p in d.glob("*.json") if d.is_dir() else []:
        if time.time() - p.stat().st_mtime < RECENT_HOURS * 3600:
            try:
                out |= {tuple(x) for x in json.loads(p.read_text())}
            except (OSError, ValueError):
                continue
    return out


def source_window(text: str, lines: list[int]) -> list[str]:
    src = text.splitlines()
    want = sorted({k for ln in lines for k in range(ln - CONTEXT, ln + CONTEXT + 1) if 0 < k <= len(src)})
    out, last = [], None
    for k in want:
        if last is not None and k != last + 1:
            out.append("      ...")
        out.append(f"{k:6}| {src[k - 1]}")
        last = k
    return out


def memo_line(cfg: Config, symbol: str, text: str) -> str:
    from .closecmd import _sha, memo_name
    from .objcache import main_root
    p = main_root(cfg) / "build" / "dk" / "close" / f"{memo_name(symbol)}.json"
    try:
        m = json.loads(p.read_text())
    except (OSError, ValueError):
        return "close: never run"
    same = " (on this source)" if m.get("source") == _sha(text) else " (on an older source)"
    return (f"close: {len(m.get('explored', []))} candidates tried{same}, {m.get('runs', 0)} run(s), best "
            f"{m.get('best', 0.0):.2f}%" + (f" with {' + '.join(m['labels'])}" if m.get("labels") else ""))


def card(cfg: Config, unit_name: str, symbol: str, rules: bool) -> tuple[list[str], dict]:
    from .matching import MatchError, resolve_unit
    from .regs import residue_report, why_temp_report
    try:
        unit = resolve_unit(cfg, unit_name)
        text = (cfg.root / unit["metadata"]["source_path"]).read_text()
    except (MatchError, OSError, KeyError) as e:
        return [f"## {symbol}: {e}"], {}
    quiet = lambda s: None
    _code, res = residue_report(cfg, unit, symbol, text, say=quiet)
    out = [f"## {symbol}  ({unit_name}, {unit['metadata']['source_path']})",
           f"now {res.get('percent', 0):.2f}%, target colours forced {res.get('forced', 0):.2f}%"
           + (" -> register order only" if res.get("forced") == 100.0 or res.get("forced_linked") else "")
           + (f"; {res['error']}" if res.get("error") else "")]
    lines = []
    for r in (res.get("rows") or [])[:MAX_ROWS]:
        land = (f"  [lands: target l{r['target_dest_line'] or '?'}, ours l{r['ours_dest_line'] or '?'}]"
                if "target_dest_line" in r else "")
        out.append(f"  {r['class']:<12} l{r['line'] or '?':<5} {r['target'] or '-':<30} | {r['ours'] or '-'}{land}")
        lines += [x for x in (r["line"], r.get("target_dest_line"), r.get("ours_dest_line")) if x]
    if len(res.get("rows") or []) > MAX_ROWS:
        out.append(f"  ... {len(res['rows']) - MAX_ROWS} more rows")
    if rules:
        _c, why = why_temp_report(cfg, unit, symbol, text, say=quiet)
        for r in [x for x in (why.get("rules") or []) if x.get("flip")][:4]:
            a, b = r.get("before", {}), r.get("after", {})
            out.append(f"  rule [{r.get('cls')} {r.get('cell')}] {a.get('text', '?')}  BEFORE  {b.get('text', '?')}")
            out.append(f"       why: {r.get('verdict', '')}" + (f"; lever: {r['levers'][0]}" if r.get("levers") else ""))
            lines += [x for x in (a.get("line"), b.get("line")) if x]
    out.append("  " + memo_line(cfg, symbol, text))
    out += source_window(text, lines)
    return out, res


def run(cfg: Config, args: argparse.Namespace) -> int:
    from .oracle import UNAVAILABLE, runner
    from .closecmd import read_pairs
    if runner(cfg) is None:
        print(f"error: {UNAVAILABLE}")
        return USAGE
    if args.functions:
        from .regs import unit_of_function
        todo = [(unit_of_function(cfg, f) or "", f) for f in args.functions]
    else:
        skip = recent_cards(cfg)
        for p in args.skip or []:
            try:
                skip |= set(read_pairs(json.loads(Path(p).read_text())))
            except (OSError, ValueError):
                pass
        todo = [(r["unit"], r["function"]) for r in pick(cfg, args.batch, skip)]
        if not todo:
            print("no census rows to pick from (run ./dk night or ./dk regs --residue --class C first)")
            return USAGE
    text = []
    for u, f in todo:
        block, _ = card(cfg, u, f, args.rules)
        text += block + [""]
        print("\n".join(block) + "\n", flush=True)
    d = Path(cfg.root) / "build" / "dk" / "cards"
    d.mkdir(parents=True, exist_ok=True)
    stem = time.strftime("%Y%m%d-%H%M%S")
    (d / f"{stem}.md").write_text("\n".join(text))
    (d / f"{stem}.json").write_text(json.dumps([list(t) for t in todo]))
    print(f"{len(todo)} card(s) written to {cfg.rel(d / (stem + '.md'))}")
    hint("try UNIT FN --replace OLD NEW  (then turn a fix that repeats into a rewrite rule)")
    return OK


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("functions", nargs="*", help="functions to card (default: --batch from the census)")
    p.add_argument("--batch", type=int, default=8, help="how many functions to pick from the census (default 8)")
    p.add_argument("--rules", action="store_true", help="also the order solver's broken rules (one oracle run more each)")
    p.add_argument("--skip", action="append", metavar="JSON", help="[unit, fn] pairs to leave out")


COMMAND = Command("card", "matching", "compact cards for hard functions (residue, landings, rules, memo, source lines)",
                  run, _configure)
