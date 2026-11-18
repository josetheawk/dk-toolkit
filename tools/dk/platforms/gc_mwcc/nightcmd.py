"""dk night: the unattended night, deterministic tools only (no agent, no Claude).

Steps, each an ordinary `./dk` command with its own log under build/dk/night/<date>/:
  0. worktree sync            in a dk worktree, merge the shared branch's landings first (stale sources waste the
                               night: the queue would re-find what was landed today); stop if the merge is refused
  1. doctor                    stop on any FAIL (a missing tool or model would waste the night)
  2. regs --residue --class C  per class (near-register, giant): forced-colour census, reused for 20 h. Functions
                               exact once the target colours are forced are register order only: they go first
  3. close --queue             the CPU enumerator, with the oracle's moves when a runner is set, register-order-only
                               functions first, up to --until; judged exacts become candidates, never applied
  4. llm-grind (opt-in)        only with --lora MODEL (the GPU runs only when the owner asks for it)
  5. llm-review DATE           pre-review of the night's candidates into digest.md
Then build/dk/night/<date>.md: what ran, candidates found, the commands to land them. Nothing is committed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from ...core.command import OK, USAGE, Command, hint
from ...core.config import Config

FRESH_HOURS = 20
CLASSES = ("near-register", "giant")


def _dk(cfg: Config, argv: list[str], log: Path) -> int:
    with log.open("a") as fh:
        fh.write(f"\n$ ./dk {' '.join(argv)}  ({time.strftime('%H:%M')})\n")
        fh.flush()
        return subprocess.run([sys.executable, "-m", "tools.dk", *argv], cwd=cfg.root, stdout=fh,
                              stderr=subprocess.STDOUT).returncode


def first_list(paths: list[Path], out: Path) -> int:
    """Functions exact once the target colours are forced, from residue census files, as [unit, fn] pairs."""
    pairs = []
    for p in paths:
        try:
            pairs += [[r["unit"], r["function"]] for r in json.loads(p.read_text())
                      if r.get("forced") == 100.0 or r.get("forced_linked")]
        except (OSError, ValueError, KeyError):
            continue
    out.write_text(json.dumps(pairs) + "\n")
    return len(pairs)


def candidates_of(ddir: Path, date: str) -> list[dict]:
    cdir = ddir / "candidates" / date
    out = []
    for p in sorted(cdir.glob("*.json")) if cdir.is_dir() else []:
        try:
            rec = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(rec, dict) and rec.get("function"):
            out.append(rec)
    return out


def run(cfg: Config, args: argparse.Namespace) -> int:
    from .llmtry import data_dir
    from .oracle import runner
    date = time.strftime("%Y-%m-%d")
    night = cfg.root / "build" / "dk" / "night"
    work = night / date
    work.mkdir(parents=True, exist_ok=True)
    log = work / "night.log"
    steps: list[tuple[str, str]] = []

    def step(name: str, argv: list[str]) -> int:
        code = _dk(cfg, argv, log)
        steps.append((name, f"exit {code}"))
        print(f"  {name}: exit {code}", flush=True)
        return code

    print(f"night {date}: logs in {cfg.rel(work)}", flush=True)
    from ...core.landing import meta, sync
    if meta(cfg.root) is not None:
        ok, detail = sync(cfg.root)
        steps.append(("worktree sync", "ok" if ok else "refused"))
        print(f"  worktree sync: {detail.splitlines()[0] if detail else ('ok' if ok else 'refused')}", flush=True)
        if not ok:
            print("the worktree could not take the shared branch's landings; resolve that first")
            return USAGE
    from .objcache import prune
    gone = prune(cfg)                                       # the compile cache stays under its size cap (DK_OBJCACHE_MB)
    steps.append(("compile cache prune", f"{gone} entries removed"))
    print(f"  compile cache: {gone} old entries removed", flush=True)
    if step("doctor", ["doctor"] + (["--model", args.lora] if args.lora else [])) != OK:
        print(f"doctor failed: see {cfg.rel(log)}; nothing else ran")
        return USAGE
    oracle = runner(cfg) is not None
    census = []
    for klass in (c for c in args.classes.split(",") if c):
        path = work.parent / f"residue-{klass}.json"
        if oracle and not args.no_census and not (path.is_file() and time.time() - path.stat().st_mtime < FRESH_HOURS * 3600):
            step(f"residue census {klass}", ["regs", "--residue", "--class", klass, "--json", str(path)])
        if path.is_file():
            census.append(path)
    first = work / "register-order-only.json"
    n_first = first_list(census, first)
    queue = ["close", "--queue", "--until", args.until, "--max-size", str(args.max_size), "--first", str(first)]
    if oracle:
        queue += ["--moves", "oracle"]
    step(f"close --queue ({n_first} register-order-only first)", queue)
    if args.lora:
        step("llm-grind (GPU)", ["llm-grind", "--until", args.until, "--model", args.lora])
    found = candidates_of(data_dir(cfg), date)
    if found:
        step("llm-review", ["llm-review", date])
    lines = [f"# Night {date}", "", f"oracle: {'yes' if oracle else 'no (close without oracle moves)'}; "
             f"register-order-only functions first: {n_first}", "", "| step | result |", "|---|---|"]
    lines += [f"| {name} | {res} |" for name, res in steps]
    lines += ["", f"candidates: {len(found)} (llm-data/candidates/{date}/)"]
    lines += [f"- {c.get('id', '?')}: {c.get('unit')} {c.get('function')} {c.get('before', 0):.2f}% -> "
              f"{c.get('after', 0):.0f}% ({c.get('route', '')}, review {', '.join(map(str, c.get('review') or [])) or '-'})"
              for c in found[:40]]
    if found:
        lines += ["", f"Next: read llm-data/candidates/{date}/digest.md, then `./dk llm-review {date} --approve ID…` "
                      "and `./dk llm-land ID…`."]
    (night / f"{date}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    hint(f"llm-review {date}" if found else "status")
    return OK


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("--until", required=True, metavar="HH:MM", help="stop time for the queue (and the grind)")
    p.add_argument("--classes", default=",".join(CLASSES),
                   help=f"route classes for the residue census (default {','.join(CLASSES)})")
    p.add_argument("--max-size", type=int, default=32768, help="largest function the queue tries (default 32768: giants too)")
    p.add_argument("--no-census", action="store_true", help="reuse the last census files whatever their age")
    p.add_argument("--lora", metavar="MODEL", help="also run llm-grind with this model (GPU; only when you ask for it)")


COMMAND = Command("night", "matching", "the unattended night: doctor, residue census, CPU close queue, review digest",
                  run, _configure)
