"""dk usage report: what the usage log (core/usage.py) says about how the toolkit is used.

Per command: runs, failures and their most common messages, median and 95th-percentile duration.
Per search tool (the platform's USAGE_SEARCH): how often a use on a function was followed by a landing
(a USAGE_LANDING command that committed that function exact) in the same worktree session: a run of
one worktree's lines with no gap over SESSION_GAP_H hours. Functions with the most runs and no
landing; guards that refused commits (`refused_by`). A plain summary first, then the tables.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from . import usage
from .command import OK, USAGE, Command, hint
from .config import Config

SESSION_GAP_H = 3.0
TOP = 10

# Used when the platform names none: `tool label -> (command, any of these flags or none needed)`.
DEFAULT_SEARCH: dict[str, tuple[str, tuple[str, ...]]] = {}
DEFAULT_LANDING: tuple[str, ...] = ()


def read_log(path: Path, since: str = "") -> list[dict]:
    rows = []
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue                                   # a torn or foreign line
        if isinstance(row, dict) and row.get("time") and (not since or row["time"][:10] >= since):
            rows.append(row)
    return sorted(rows, key=lambda r: _when(r))


def _when(row: dict) -> float:
    try:
        return datetime.fromisoformat(row["time"]).timestamp()
    except (ValueError, KeyError, TypeError):
        return 0.0


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def _normalize(msg: str) -> str:
    """Group messages that differ only in numbers, addresses or quoted names."""
    msg = re.sub(r"0x[0-9A-Fa-f]+|\b[0-9A-F]{8}\b", "ADDR", msg)
    msg = re.sub(r"\d+(\.\d+)?", "N", msg)
    msg = re.sub(r"'[^']*'|\"[^\"]*\"|`[^`]*`", "'…'", msg)
    return msg[:120]


def _sessions(rows: list[dict]) -> dict[int, tuple[str, int]]:
    """Row index -> (worktree, session number in it)."""
    last: dict[str, float] = {}
    number: dict[str, int] = defaultdict(int)
    out = {}
    for i, row in enumerate(rows):
        wt, t = row.get("worktree", "?"), _when(row)
        if wt in last and t - last[wt] > SESSION_GAP_H * 3600:
            number[wt] += 1
        last[wt] = t
        out[i] = (wt, number[wt])
    return out


def _is_search(row: dict, spec: tuple[str, tuple[str, ...]]) -> bool:
    cmd, flags = spec
    return row.get("cmd") == cmd and (not flags or any(f in row.get("flags", ()) for f in flags))


def _landed(row: dict, landing: tuple[str, ...]) -> set[str]:
    if row.get("cmd") not in landing or row.get("exit") != 0 or row.get("committed") is False:
        return set()
    return set(row.get("landed") or ())


def analyze(rows: list[dict], search: dict, landing: tuple[str, ...]) -> dict:
    session = _sessions(rows)
    # per command
    by_cmd: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_cmd[(row.get("cmd") or "(none)") + (f" {row['sub']}" if row.get("sub") else "")].append(row)
    commands = []
    for name, group in by_cmd.items():
        fails = [r for r in group if r.get("exit") != 0]
        msgs = Counter(_normalize(r["msg"]) for r in fails if r.get("msg"))
        crashes = Counter(r["error"] for r in group if r.get("error"))
        secs = [float(r.get("secs") or 0) for r in group]
        commands.append({
            "command": name, "runs": len(group), "failures": len(fails),
            "mismatch": sum(r.get("exit") == 1 for r in fails), "usage": sum(r.get("exit") == 2 for r in fails),
            "build_error": sum(r.get("exit") == 3 for r in fails), "crash": sum(r.get("exit") is None for r in fails),
            "crash_types": dict(crashes.most_common(3)),
            "top_messages": msgs.most_common(3),
            "median_s": round(_quantile(secs, 0.5), 2), "p95_s": round(_quantile(secs, 0.95), 2),
        })
    commands.sort(key=lambda c: (-c["runs"], c["command"]))

    # landings per session, with their times
    landings: dict[tuple[str, int], list[tuple[float, set[str]]]] = defaultdict(list)
    for i, row in enumerate(rows):
        got = _landed(row, landing)
        if got:
            landings[session[i]].append((_when(row), got))

    def landed_after(key, t: float, fn: str) -> bool:
        return any(lt >= t and fn in fns for lt, fns in landings.get(key, ()))

    # search tools: one use = a (session, function) pair, from its first run
    tools = []
    for label, spec in search.items():
        first: dict[tuple, float] = {}
        runs = 0
        for i, row in enumerate(rows):
            if not _is_search(row, spec):
                continue
            runs += 1
            for fn in [row["function"]] if row.get("function") else row.get("functions") or ():
                first.setdefault((session[i], fn), _when(row))   # a scan names several
        hits = sum(landed_after(k[0], t, k[1]) for k, t in first.items())
        tools.append({"tool": label, "runs": runs, "functions": len(first), "landed": hits,
                      "hit_rate": round(hits / len(first), 3) if first else None})

    # functions worked without a landing
    work: dict[tuple, dict] = {}
    for i, row in enumerate(rows):
        fn = row.get("function")
        if not fn or row.get("cmd") in landing:
            continue
        w = work.setdefault((session[i], fn), {"first": _when(row), "runs": 0, "cmds": Counter(), "last": row["time"]})
        w["runs"] += 1
        w["cmds"][row.get("cmd")] += 1
        w["last"] = row["time"]
    stuck: dict[str, dict] = {}
    for (key, fn), w in work.items():
        if landed_after(key, w["first"], fn):
            continue
        s = stuck.setdefault(fn, {"function": fn, "runs": 0, "sessions": 0, "commands": Counter(), "last": ""})
        s["runs"] += w["runs"]
        s["sessions"] += 1
        s["commands"].update(w["cmds"])
        s["last"] = max(s["last"], w["last"])
    unlanded = sorted(stuck.values(), key=lambda s: (-s["runs"], s["function"]))[:TOP]
    for s in unlanded:
        s["commands"] = dict(s["commands"].most_common())

    # guards
    guards = Counter((r.get("refused_by"), r.get("cmd")) for r in rows if r.get("refused_by"))
    guard_rows = [{"guard": g, "command": c, "refusals": n} for (g, c), n in guards.most_common()]

    days = sorted({r["time"][:10] for r in rows})
    return {"runs": len(rows), "first": days[0] if days else "", "last": days[-1] if days else "",
            "worktrees": len({r.get("worktree") for r in rows}), "sessions": len(set(session.values())),
            "landings": sum(len(v) for v in landings.values()),
            "commands": commands, "search_tools": tools, "unlanded": unlanded, "guards": guard_rows}


def summary(a: dict) -> list[str]:
    if not a["runs"]:
        return ["The usage log has no runs in this period."]
    out = [f"{a['runs']} dk runs from {a['first']} to {a['last']} in {a['worktrees']} worktree(s), "
           f"{a['sessions']} session(s), {a['landings']} landing(s)."]
    top = a["commands"][:3]
    out.append("Most used: " + ", ".join(f"{c['command']} ({c['runs']})" for c in top) + ".")
    usage_err = sorted((c for c in a["commands"] if c["usage"] + c["crash"]), key=lambda c: -(c["usage"] + c["crash"]))
    if usage_err:
        c = usage_err[0]
        msg = f": most often \"{c['top_messages'][0][0]}\"" if c["top_messages"] else ""
        out.append(f"Most usage errors or crashes: {c['command']}, {c['usage'] + c['crash']} of {c['runs']} runs{msg}.")
    slow = max(a["commands"], key=lambda c: c["median_s"] * c["runs"])
    out.append(f"Most time spent: {slow['command']} ({slow['runs']} runs, median {slow['median_s']} s, "
               f"95th {slow['p95_s']} s).")
    rated = [t for t in a["search_tools"] if t["hit_rate"] is not None]
    if rated:
        best = max(rated, key=lambda t: (t["hit_rate"], t["functions"]))
        worst = min(rated, key=lambda t: (t["hit_rate"], -t["functions"]))
        out.append(f"Search tools: {best['tool']} led to a landing for {best['landed']} of {best['functions']} "
                   f"function(s) ({best['hit_rate']:.0%})"
                   + (f"; {worst['tool']} for {worst['landed']} of {worst['functions']} ({worst['hit_rate']:.0%})."
                      if worst is not best else "."))
    if a["unlanded"]:
        u = a["unlanded"][0]
        out.append(f"Most work without a landing: {u['function']}, {u['runs']} runs in {u['sessions']} session(s).")
    if a["guards"]:
        g = a["guards"][0]
        out.append(f"Guards refused {sum(x['refusals'] for x in a['guards'])} commit(s); most often {g['guard']} "
                   f"({g['refusals']}, on {g['command']}).")
    return out


def _table(head: list[str], rows: list[list]) -> list[str]:
    cells = [head] + [[str(x) for x in r] for r in rows]
    width = [max(len(r[i]) for r in cells) for i in range(len(head))]
    return ["  " + "  ".join(c.ljust(w) for c, w in zip(r, width)).rstrip() for r in cells]


def render(a: dict) -> list[str]:
    out = summary(a)
    if not a["runs"]:
        return out
    out += ["", "Commands:"]
    out += _table(["command", "runs", "fail", "mism", "usage", "build", "crash", "median s", "p95 s"],
                  [[c["command"], c["runs"], c["failures"], c["mismatch"], c["usage"], c["build_error"], c["crash"],
                    c["median_s"], c["p95_s"]] for c in a["commands"]])
    msgs = [(c["command"], n, m) for c in a["commands"] for m, n in c["top_messages"]]
    if msgs:
        out += ["", "Top failure messages:"]
        out += _table(["command", "count", "message"], [list(m) for m in msgs])
    crashes = [(c["command"], t, n) for c in a["commands"] for t, n in c["crash_types"].items()]
    if crashes:
        out += ["", "Crashes:"]
        out += _table(["command", "exception", "count"], [list(c) for c in crashes])
    if a["search_tools"]:
        out += ["", f"Search tools (landing of the same function later in the same worktree session, gap ≤ {SESSION_GAP_H:g} h):"]
        out += _table(["tool", "runs", "functions", "landed", "hit rate"],
                      [[t["tool"], t["runs"], t["functions"], t["landed"],
                        "-" if t["hit_rate"] is None else f"{t['hit_rate']:.0%}"] for t in a["search_tools"]])
    if a["unlanded"]:
        out += ["", "Most runs without a landing:"]
        out += _table(["function", "runs", "sessions", "last", "commands"],
                      [[u["function"], u["runs"], u["sessions"], u["last"][:16],
                        ", ".join(f"{k} {v}" for k, v in u["commands"].items())] for u in a["unlanded"]])
    if a["guards"]:
        out += ["", "Guard refusals:"]
        out += _table(["guard", "command", "refusals"], [[g["guard"], g["command"], g["refusals"]] for g in a["guards"]])
    return out


def run(cfg: Config, args: argparse.Namespace) -> int:
    if args.since:
        try:
            datetime.strptime(args.since, "%Y-%m-%d")
        except ValueError:
            print(f"error: --since wants YYYY-MM-DD, not {args.since}")
            return USAGE
    path = Path(args.log).expanduser() if args.log else usage.log_path(cfg)
    if path is None:
        print("error: the usage log is off (DK_USAGE_LOG or dk.toml [usage] log); give --log PATH")
        return USAGE
    from .. import platforms
    search = (platforms.hook(cfg.platform, "USAGE_SEARCH") if cfg.platform else None) or DEFAULT_SEARCH
    landing = tuple((platforms.hook(cfg.platform, "USAGE_LANDING") if cfg.platform else None) or DEFAULT_LANDING)
    a = analyze(read_log(path, args.since), search, landing)
    if args.json:
        print(json.dumps(a, indent=1, default=str))
        return OK
    print(f"usage log: {cfg.rel(path)}")
    print("\n".join(render(a)))
    hint("usage report --since YYYY-MM-DD", "usage report --json")
    return OK


def _configure(p: argparse.ArgumentParser) -> None:
    sub = p.add_subparsers(dest="action", required=True)
    r = sub.add_parser("report", help="per command, per search tool, unlanded functions and guards")
    r.add_argument("--since", default="", metavar="DATE", help="only runs on or after YYYY-MM-DD")
    r.add_argument("--json", action="store_true", help="the analysis as JSON")
    r.add_argument("--log", default="", help="read this log instead of the configured one")


COMMAND = Command("usage", "docs", "what the dk usage log says: runs, failures, durations, search-tool hit rates, guards",
                  run, _configure)
