"""`./dk lora fix|stuck`: the structured wrapper around llm-try for the LoRA harness (D2/D4).

Hints are structured tokens only (rule:<id>, xform:<name>@<line>, region:<id>); anything else is rejected.
The wrapper never lands code: candidates are reported, nothing is applied."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from ...core.command import Command
from ...core.config import Config
from ...core.stuck import replay, stuck_state

HINT_RE = re.compile(r"^(?:rule:[a-z0-9][a-z0-9_-]{0,40}|xform:[a-z0-9][a-z0-9_-]{0,40}@[0-9]{1,5}|region:[A-Za-z0-9:_-]{1,24})$")
REGION_RE = re.compile(r"^(?:auto|[A-Za-z0-9:_-]{1,24})$")


def valid_hint(h: str) -> bool:
    return bool(HINT_RE.fullmatch(h))


def tool_block(pct: float, rows: list, hints: tuple[str, ...] = ()) -> str:
    """The tool-output block of a prompt: tokens and numbers only, no prose."""
    return "TOOLS: " + " ".join([f"pct={pct:.2f}", f"diff_rows={len(rows)}", *(f"hint={h}" for h in hints)])


def summarize(log: Path) -> dict:
    best, exact, cand, dev, rounds, rows = 0.0, False, None, False, {}, None
    for line in log.read_text(errors="replace").splitlines():
        try:
            x = json.loads(line)
        except ValueError:
            continue
        if x.get("kind") != "attempt":
            continue
        pct = x.get("percent") if x.get("compiled") else None
        rounds.setdefault(int(x.get("round", 0)), []).append(-1.0 if pct is None else float(pct))
        if pct is not None and pct > best:
            best, cand, rows = pct, x.get("variant"), len(x.get("rows") or [])
        if x.get("exact") or (pct is not None and pct >= 100.0):
            exact = True
        dev = dev or bool(x.get("devices"))
    seq = [(r, max(v), len(v)) for r, v in sorted(rounds.items())]
    return {"best_pct": round(best, 3), "exact": exact, "candidate": cand, "devices": dev, "stuck": stuck_state(seq),
            "diff_rows": rows}


def fix(cfg: Config, args: argparse.Namespace) -> int:
    bad = [h for h in args.hint if not valid_hint(h)]
    if bad or not REGION_RE.fullmatch(args.region):
        print(json.dumps({"error": "hint or region rejected (structured tokens only)", "bad": bad}))
        return 2
    if not args.model:
        args.model = str((cfg.platform_options.get("llm") or {}).get("lora_model", ""))
        if not args.model:
            print(json.dumps({"error": "no model: pass --model or set [platform.llm] lora_model"}))
            return 2
    if args.region != "auto" or any(h.startswith("region:") for h in args.hint):
        from .region import region_fix
        return region_fix(cfg, args)
    route = json.loads(subprocess.run(["./dk", "route", "--json"], cwd=cfg.root, capture_output=True, text=True).stdout)["rows"]
    row = next((r for r in route if r["function"] == args.fn), None)
    if not row:
        print(json.dumps({"error": "function not in the open queue"}))
        return 2
    run = f"lorafix-{args.fn}"
    cmd = ["./dk", "llm-try", row["unit"], args.fn, "--tool-block", "--model", args.model, "--url", args.url,
           "--n", "8", "--rounds", str(max(1, args.budget // 8)), "--run", run]
    for h in args.hint:
        cmd += ["--tool-hint", h]
    p = subprocess.run(cmd, cwd=cfg.root, capture_output=True, text=True)          # no --apply: candidates only
    from .llmtry import data_dir
    log = data_dir(cfg) / f"{run}.jsonl"
    if not log.is_file():
        print(json.dumps({"fn": args.fn, "error": f"llm-try wrote no log (exit {p.returncode})",
                          "detail": (p.stdout + p.stderr).strip()[-400:]}))
        return 2
    res = summarize(log)
    res.update(fn=args.fn)
    print(json.dumps(res) if args.json else f"{args.fn}: best {res['best_pct']}% exact {res['exact']}")
    return 0 if res["exact"] else 1


def stuck_cmd(cfg: Config, args: argparse.Namespace) -> int:
    from .llmtry import data_dir
    paths = [p for g in args.replay for p in sorted(data_dir(cfg).glob(g))]
    r = replay(paths)
    print(json.dumps(r, indent=1) if args.json else
          f"{r['functions']} fn, {r['attempts']} attempts: flagged {r['flagged']}, precision {r['precision']}, "
          f"saved {r['saved_attempts']} attempts ({r['saved_share']})")
    return 0


def census_cmd(cfg: Config, args: argparse.Namespace) -> int:
    from .census import census
    from .llmtry import data_dir
    d = data_dir(cfg)
    h = [p for g in args.history for p in sorted(d.glob(g))]
    s = [p for g in args.stacked for p in sorted(d.glob(g))]
    print(json.dumps(census(h, s), indent=1))
    return 0


def run(cfg: Config, args: argparse.Namespace) -> int:
    return {"fix": fix, "stuck": stuck_cmd, "census": census_cmd}[args.action](cfg, args)


def configure(p: argparse.ArgumentParser) -> None:
    sub = p.add_subparsers(dest="action", required=True)
    f = sub.add_parser("fix", help="llm-try with the tool block on one open function; candidates only, never landed")
    f.add_argument("fn")
    f.add_argument("--region", default="auto")
    f.add_argument("--hint", action="append", default=[], help="rule:<id> | xform:<name>@<line> | region:<id> (nothing else)")
    f.add_argument("--budget", type=int, default=32, help="attempts (8 per round)")
    f.add_argument("--model", default=None, help="the Ollama model (default: [platform.llm] lora_model)")
    f.add_argument("--url", default="http://127.0.0.1:11435")
    f.add_argument("--json", action="store_true")
    c = sub.add_parser("census", help="D5: multi-change real pairs by fix kind and size band, stacked coverage and monotone paths")
    c.add_argument("--history", nargs="+", default=["dataset-history-v2/records/history.jsonl", "datasets/shape-fixes/v1/fixes.jsonl"])
    c.add_argument("--stacked", nargs="+", default=["datasets/stacked/v0/stacked.jsonl"])
    s = sub.add_parser("stuck", help="the stuck rule replayed on finished attempt logs (globs under llm-data/)")
    s.add_argument("--replay", nargs="+", required=True)
    s.add_argument("--json", action="store_true")


COMMAND = Command("lora", "matching", "LoRA harness wrapper: fix (llm-try + tool block, structured hints) / stuck --replay",
                  run, configure)
