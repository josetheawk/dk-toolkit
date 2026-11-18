"""dk llm-bench report --showdown DIR: the scoreboard of a contestant race.

DIR holds `set.json` (tiers easy / medium / hard, each a list of open functions), one
`<contestant>.log` per contestant (a line per finished attempt, `HH:MM tier function start% -> end%
exact|miss (reason)`, whatever else the contestant wrote is skipped) and, for each function a
contestant finished exact, `<contestant>/<function>.cpp`: its final definition (and helpers).

Every `.cpp` goes through the toolkit's judge: spliced into the unit's source at HEAD, compiled and
scored privately (`llm-try`'s scorer), then pre-reviewed. Per contestant and tier:
- certain exact: 100%, judge clean, pre-review clean;
- needs judgment: 100% but a flag a person must read (aliasing local, empty case labels, ...);
- device: 100% only through a device (volatile, padding, aligned static, ...): counts as a miss for landing,
  kept as a negative;
- unjudged: the log says exact but there is no `.cpp` to judge (a contestant that only logs);
- miss: everything else, with a blocker class (`route`'s names) taken from the log's reason.
Also attempts (finished-attempt lines) and minutes from the tier's start (an explicit start line, else its
first logged line) to the first exact line. Writes DIR/scoreboard.md and scoreboard.json.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ...core.command import OK, USAGE
from ...core.config import Config
from .route import GIANT

TIERS = ("easy", "medium", "hard")
KINDS = ("certain", "judgment", "device", "unjudged", "miss")
_LINE = re.compile(r"^(?P<t>\d\d:\d\d)\s+(?P<tier>easy|medium|hard)\s+(?P<fn>[A-Za-z_@]\w*)(?:\([^)]*\))?\s+"
                   r"(?P<a>[\d.]+)\s*(?:->|→)\s*(?P<b>[\d.]+)\s*(?P<rest>.*)$", re.I)
_START = (re.compile(r"^(?P<t>\d\d:\d\d)\s+(?P<tier>easy|medium|hard)\s+tier\s+start\b", re.I),
          re.compile(r"^tier\s+(?P<tier>easy|medium|hard)\s+start\s+(?P<t>\d\d:\d\d)", re.I))
# log reason -> blocker class (route's names), first match wins
REASONS = (("sinit", r"__sinit|static initiali[sz]er"),
           ("unit-data", r"data layout|\.data\b|\.bss|\.sdata|string[- ]pool|pool offset|vtables?\b|base symbol|symbol/data|"
                         r"unwritten functions of the unit|literal"),
           ("near-scheduling", r"schedul|prologue|interleav"),
           ("near-register", r"regist|\brank|callee-saved|\br\d{1,2}\b|numbering|permut|rotate|decl\w* order|swap"),
           ("compiler-version", r"compiler-search|GC/1\.\d|unit flags|compiler version|1\.3\.2"))


def minutes(clock: str) -> int:
    h, m = clock.split(":")
    return int(h) * 60 + int(m)


def parse_log(text: str) -> tuple[list[dict], dict[str, int]]:
    """(attempt lines, tier starts in minutes). An attempt is a line with `start% -> end%`; its verdict is
    exact (`exact*`: the contestant flagged a caveat), miss, or None when the line says neither."""
    rows, starts = [], {}
    for line in text.splitlines():
        line = line.strip()
        s = next((m for m in (p.match(line) for p in _START) if m), None)
        if s:
            starts.setdefault(s["tier"].lower(), minutes(s["t"]))
            continue
        m = _LINE.match(line)
        if not m:
            continue
        rest = m["rest"].strip()
        v = re.match(r"(exact\*?|miss)(?!\w)", rest, re.I)
        verdict = v[1].lower().rstrip("*") if v else ("miss" if re.search(r"counts MISS", rest, re.I) else None)
        if verdict is None:
            continue
        reason = rest[v.end():] if v else rest
        rows.append({"time": minutes(m["t"]), "tier": m["tier"].lower(), "function": m["fn"], "start": float(m["a"]),
                     "end": float(m["b"]), "verdict": verdict, "caveat": bool(v and v[1].endswith("*")),
                     "reason": reason.strip("() ")[:200]})
    return rows, starts


def blocker(fn: dict, reason: str) -> str:
    """A miss's blocker class: `route`'s names, from the contestant's reason, else from the function's facts."""
    from .route import classify
    if fn["function"].startswith(("__sinit", "@")):
        return "sinit"
    for name, pat in REASONS:
        if re.search(pat, reason, re.I):
            return name
    if fn["size"] > GIANT:
        return "giant"
    cls = classify(fn["function"], fn["size"], fn["fuzzy"], None, False)
    return cls if cls in ("fresh", "far") else "unclassified"


# ---- judging ------------------------------------------------------------------------------------------

def notes_cut(chunk: str) -> str:
    """The file without its leading `//` notes (the brief asks for helper and declaration changes as a comment block)."""
    lines = chunk.splitlines()
    while lines and (lines[0].lstrip().startswith("//") or not lines[0].strip()):
        lines.pop(0)
    return "\n".join(lines) + "\n"


def judge_chunk(cfg: Config, fn: dict, chunk: str, scorers: dict) -> dict:
    """One contestant's definition for one function: {verdict, percent, flags}. The chunk is spliced into the
    unit's source at HEAD and scored privately; only a 100% result is pre-reviewed."""
    from .judge import JUDGMENT, REJECT, review, unit_probe
    from .llmtry import Scorer
    from .matching import resolve_unit
    from .trycmd import locate_target
    key = fn["function"]
    if key not in scorers:
        unit = resolve_unit(cfg, fn["unit"])
        source = cfg.root / unit["metadata"]["source_path"]
        src = source.read_text()
        span, locate, _ = locate_target(cfg, src, source, key)
        sc = Scorer(cfg, unit, key, src, span, locate)
        sc.baseline()
        scorers[key] = sc
    sc = scorers[key]
    ev = sc.evaluate(notes_cut(chunk), "showdown")
    if not ev.get("compiled"):
        return {"verdict": "miss", "percent": None, "flags": [ev.get("error") or "does not compile in the unit"]}
    pct = ev["percent"]
    if pct < 100.0:
        return {"verdict": "miss", "percent": pct, "flags": []}
    probe = unit_probe(cfg, sc.unit, key, sc.work / "showdown_casts")
    category, flags = review(sc.src, ev["text"], key, sc.forbidden, probe)
    flags = flags + [f"judge: {r}" for r in ev.get("judge", []) if r not in flags]
    if category == REJECT:
        return {"verdict": "device", "percent": pct, "flags": flags}
    if category == JUDGMENT or ev.get("judge") or ev.get("judgment"):
        return {"verdict": "judgment", "percent": pct, "flags": flags or ["judgment"]}
    return {"verdict": "certain", "percent": pct, "flags": []}


def contestants(ddir: Path) -> list[str]:
    names = {p.stem for p in ddir.glob("*.log")} | {p.name for p in ddir.iterdir() if p.is_dir()}
    return sorted(names)


def score_race(ddir: Path, judge) -> dict:
    """The scoreboard as data. `judge(fn_row, chunk) -> {verdict, percent, flags}` judges one `.cpp`."""
    tiers = json.loads((ddir / "set.json").read_text())
    functions = [{**f, "tier": t} for t in TIERS for f in tiers.get(t, [])]
    out = {"set": {t: len(tiers.get(t, [])) for t in TIERS}, "contestants": {}, "functions": functions}
    for name in contestants(ddir):
        log = ddir / f"{name}.log"
        attempts, starts = parse_log(log.read_text(errors="replace")) if log.is_file() else ([], {})
        cells = {t: {"functions": 0, "attempts": 0, "first_exact_min": None, **{k: 0 for k in KINDS}} for t in TIERS}
        rows = []
        for fn in functions:
            mine = [a for a in attempts if a["function"] == fn["function"]]
            cpp = ddir / name / f"{fn['function']}.cpp"
            logged = next((a for a in mine if a["verdict"] == "exact"), None)
            if cpp.is_file():
                res = judge(fn, cpp.read_text())
            elif logged:
                res = {"verdict": "unjudged", "percent": None, "flags": ["no .cpp to judge"]}
            else:
                res = {"verdict": "miss", "percent": None, "flags": []}
            last = mine[-1] if mine else None
            best = max((a["end"] for a in mine), default=None)
            row = {"function": fn["function"], "tier": fn["tier"], "contestant": name, "verdict": res["verdict"],
                   "percent": res["percent"], "flags": res["flags"], "log_exact": logged is not None,
                   "log_best": best, "attempts": len(mine)}
            if res["verdict"] == "miss":
                row["blocker"] = blocker(fn, last["reason"] if last else "")
                if logged:
                    row["note"] = "the log says exact; the judge could not confirm it"
            rows.append(row)
            cell = cells[fn["tier"]]
            cell["functions"] += 1
            cell["attempts"] += len(mine)
            cell[res["verdict"]] += 1
        for t in TIERS:                                   # minutes from the tier's start to its first exact line
            ex = [a["time"] for a in attempts if a["tier"] == t and a["verdict"] == "exact"]
            first = [a["time"] for a in attempts if a["tier"] == t]
            begin = starts.get(t, min(first) if first else None)
            if ex and begin is not None:
                cells[t]["first_exact_min"] = (min(ex) - begin) % 1440
        total = {k: sum(cells[t][k] for t in TIERS) for k in ("functions", "attempts", *KINDS)}
        out["contestants"][name] = {"cells": cells, "total": total, "rows": rows}
    return out


# ---- output -------------------------------------------------------------------------------------------

def render(board: dict) -> str:
    order = sorted(board["contestants"], key=lambda n: (-board["contestants"][n]["total"]["certain"], n))
    heads = "| contestant | tier | functions | certain exact | needs judgment | device | unjudged (log only) | miss | attempts | first exact (min) |"
    out = ["# Showdown scoreboard", "",
           "Certain exact = 100%, judge clean and pre-review clean. Needs judgment = 100% with a flag a person reads. "
           "Device = 100% only through a device (a miss for landing, kept as a negative). Unjudged = the log says exact, "
           "no `.cpp`. First exact = minutes from the tier's start (its explicit start line, else its first logged line).", "",
           heads, "|" + "---|" * 10]
    for n in order:
        c = board["contestants"][n]
        for t in TIERS:
            x = c["cells"][t]
            first = "" if x["first_exact_min"] is None else x["first_exact_min"]
            out.append(f"| {n} | {t} | {x['functions']} | {x['certain']} | {x['judgment']} | {x['device']} | {x['unjudged']} "
                       f"| {x['miss']} | {x['attempts']} | {first} |")
        x = c["total"]
        out.append(f"| **{n}** | **all** | {x['functions']} | **{x['certain']}** | {x['judgment']} | {x['device']} | "
                   f"{x['unjudged']} | {x['miss']} | {x['attempts']} | |")
    out += ["", "## Per function", "", "| tier | function | size | start % | " + " | ".join(order) + " |",
            "|" + "---|" * (4 + len(order))]
    by = {n: {r["function"]: r for r in board["contestants"][n]["rows"]} for n in order}
    short = {"certain": "EXACT", "judgment": "exact (judgment)", "device": "device", "unjudged": "exact (unjudged)", "miss": "miss"}
    for f in board["functions"]:
        cells = []
        for n in order:
            r = by[n][f["function"]]
            best = f" {r['log_best']:.1f}%" if r["verdict"] == "miss" and r["log_best"] is not None else ""
            cells.append(short[r["verdict"]] + best + (f" [{r['blocker']}]" if r.get("blocker") else ""))
        out.append(f"| {f['tier']} | {f['function']} | 0x{f['size']:X} | {f['fuzzy']:.1f} | " + " | ".join(cells) + " |")
    misses = {}
    for n in order:
        for r in board["contestants"][n]["rows"]:
            if r["verdict"] == "miss":
                misses.setdefault(r["blocker"], dict.fromkeys(order, 0))[n] += 1
    if misses:
        out += ["", "## Misses by blocker class", "", "| class | " + " | ".join(order) + " |", "|" + "---|" * (1 + len(order))]
        out += [f"| {c} | " + " | ".join(str(v[n]) for n in order) + " |" for c, v in sorted(misses.items())]
    flagged = [r for n in order for r in board["contestants"][n]["rows"] if r["verdict"] in ("device", "judgment") or r.get("note")]
    if flagged:
        out += ["", "## Devices, judgment calls and unconfirmed claims", ""]
        for r in flagged:
            what = r.get("note") or "; ".join(r["flags"])
            out.append(f"- {r['contestant']} {r['function']} ({r['tier']}): {r['verdict']}: {what}")
    return "\n".join(out) + "\n"


def run(cfg: Config, args: argparse.Namespace) -> int:
    ddir = Path(args.showdown).expanduser()
    if not (ddir / "set.json").is_file():
        print(f"error: {ddir}/set.json missing (the race's set: tiers easy, medium, hard)")
        return USAGE
    scorers: dict = {}
    board = score_race(ddir, lambda fn, chunk: judge_chunk(cfg, fn, chunk, scorers))
    (ddir / "scoreboard.json").write_text(json.dumps(board, indent=1) + "\n")
    (ddir / "scoreboard.md").write_text(render(board))
    for n, c in sorted(board["contestants"].items(), key=lambda kv: -kv[1]["total"]["certain"]):
        t = c["total"]
        print(f"{n}: {t['certain']} certain exact, {t['judgment']} needs judgment, {t['device']} device, "
              f"{t['unjudged']} unjudged, {t['miss']} miss of {t['functions']}")
    print(f"wrote {ddir}/scoreboard.md and scoreboard.json")
    return OK
