"""dk route: every open function classified by what blocks it, and queued for a worker.

Free data only (no compiles): the objdiff report for the open functions, one objdiff run per unit
for the instruction pairs and data sections (what `dk diff` prints), the last full
`compiler-search --sweep` (build/dk/sweep.json) and the grind's state files (past attempts).

Classes, first match wins: `sinit` (compiler-made `__sinit`/`@` names: the unit's naming and static
data decide them), `giant` (over 1 KB), `fresh` (not written), `compiler-version` (the unit gains on
another compiler or as C, per the sweep), `unit-data` (only relocations differ: pool offsets or data
layout), `far` (below --min-near), then by the diff: `near-register` (register numbers only),
`near-scheduling` (the same instructions in another order), `near-shape` (anything else).
Workers: cpu (regs/close), lora (the model), claude (giants, and anything idle for --promote
attempts), blocked (fix the unit first: data, naming, flags). `llm-grind --class C` reads this.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ...core.command import OK, USAGE, Command, hint
from ...core.config import Config
from .diffcmd import _text

GIANT = 0x400
CLASSES = ("near-register", "near-scheduling", "near-shape", "unit-data", "sinit", "compiler-version", "far", "fresh",
           "giant")
WORKERS = ("cpu", "lora", "claude", "blocked", "frozen")      # frozen: the LoRA-2 gate and near-samples (gate.py)
WORKER = {"near-register": "cpu", "near-scheduling": "lora", "near-shape": "lora", "far": "lora", "fresh": "lora",
          "giant": "claude", "unit-data": "blocked", "sinit": "blocked", "compiler-version": "blocked"}
WHY = {"near-register": "register numbers only (dk regs, dk close)", "near-scheduling": "same instructions, other order",
       "near-shape": "different instructions", "unit-data": "only relocations differ: pool offsets or data layout "
       "(dk diff --literals, dk gen --data)", "sinit": "compiler-made static-init function (unit naming and data)",
       "compiler-version": "the unit gains on another compiler or as C (dk unit flags)", "far": "written, far from exact",
       "fresh": "not written yet", "giant": "over 1 KB: inline helpers (dk inlines)"}
_REG = re.compile(r"\b[rf]\d+\b|\bcr\d\b")


_SYM = re.compile(r"[\w.$]+@(?:ha|l|h|sda21|sda2)\b|\blbl_[0-9A-Fa-f]+\b|\.\.\.\w+(?:\.\d+)?")
_IMM = re.compile(r"-?0x[0-9a-fA-F]+|-?\b\d+\b")


def _reloc(l: dict, r: dict) -> bool:
    return bool(l.get("instruction", {}).get("relocation") or r.get("instruction", {}).get("relocation"))


def diff_kind(left: list[dict], right: list[dict]) -> str:
    """How a written function differs from the target, from objdiff's instruction lists: `data` (a relocated
    operand differs: another symbol, pool offset or data layout; registers may differ too), `register`
    (register numbers only), `scheduling` (the same instructions in another order) or `shape`."""
    if len(left) != len(right):
        return "shape"
    data = reg = other = 0
    for l, r in zip(left, right):
        a, b = _text(l), _text(r)
        if not (l.get("diff_kind") or r.get("diff_kind")):
            continue
        if a == b:
            data += 1
        elif _REG.sub("R", a) == _REG.sub("R", b):
            reg += 1
        elif _reloc(l, r) and _IMM.sub("N", _REG.sub("R", _SYM.sub("S", a))) == _IMM.sub("N", _REG.sub("R", _SYM.sub("S", b))):
            data += 1
        else:
            other += 1
    if other:
        key = lambda ins: sorted(_REG.sub("R", _text(i)) for i in ins)
        return "scheduling" if key(left) == key(right) else "shape"
    return "data" if data else "register" if reg else "shape"


def classify(name: str, size: int, pct: float, kind: str | None, swept: bool, min_near: float = 80.0) -> str:
    if name.startswith(("__sinit", "@")):
        return "sinit"
    if size > GIANT:
        return "giant"
    if pct <= 0.0:
        return "fresh"
    if swept:
        return "compiler-version"
    if kind == "data":
        return "unit-data"
    if pct < min_near:
        return "far"
    return {"register": "near-register", "scheduling": "near-scheduling"}.get(kind or "", "near-shape")


def route_rows(open_functions: list[dict], diffs: dict, swept: dict, attempts: dict, min_near: float = 80.0,
               promote: int = 2, frozen: dict | None = None, blockers: dict | None = None, version: str | None = None,
               frozen_fns: set | None = None, strict: bool = False) -> list[dict]:
    """One row per open function: class, worker, why, data sections still open in its unit, past attempts,
    the diff `kind`, the latest `blocker` tag (blockers.jsonl) with the `playbook_version` it was recorded under,
    and `fresh` (see blockers.is_fresh). diffs: unit -> (function -> instruction-list pair, open data sections);
    swept: source path -> variant; frozen: (unit, function) -> gate set; blockers: function -> latest row;
    version: the current playbook version; frozen_fns: benchmark and held-out function names."""
    from . import blockers as bl
    rows = []
    for f in open_functions:
        pairs, data_open = diffs.get(f["unit"], ({}, []))
        pair = pairs.get(f["function"])
        kind = diff_kind(*pair) if pair else None
        variant = swept.get(f.get("source", "").removeprefix("src/"))
        cls = classify(f["function"], f["size"], f["fuzzy"], kind, variant is not None, min_near)
        a = attempts.get(f"{f['unit']}::{f['function']}", {})
        worker = WORKER[cls]
        if worker not in ("blocked", "claude") and a.get("idle", 0) >= promote:
            worker = "claude"                                   # idle nights promote to a closer
        held = (frozen or {}).get((f["unit"], f["function"]), "")
        if held:
            worker = "frozen"                                   # a gate or near-sample function: no worker may touch it
        why = WHY[cls] + (f": {variant}" if cls == "compiler-version" else "") \
            + (f": {', '.join(data_open)} open" if cls == "unit-data" and data_open else "") \
            + (" (no object to diff)" if cls.startswith("near") and pair is None else "") \
            + (f" [FROZEN: {held}, barred from grinding and training]" if held else "")
        b = (blockers or {}).get(f["function"]) or {}
        row = {"unit": f["unit"], "function": f["function"], "address": f["address"], "size": f["size"],
               "fuzzy": round(f["fuzzy"], 2), "class": cls, "worker": worker, "why": why, "data_open": data_open,
               "attempts": a.get("attempts", 0), "idle": a.get("idle", 0), "best": a.get("best"),
               "candidate": a.get("candidate"), "frozen": held, "kind": kind, "blocker": b.get("blocker"),
               "playbook_version": b.get("playbook_version")}
        row["fresh"] = not held and bl.is_fresh(row, version, frozen_fns or set(), min_near, strict)
        rows.append(row)
    return rows


def select(rows: list[dict], spec: str) -> list[dict]:
    """The rows in these classes or workers (comma-separated); ValueError for a name that is neither."""
    want = {c for c in spec.split(",") if c}
    bad = sorted(want - set(CLASSES) - set(WORKERS))
    if bad:
        raise ValueError(f"unknown class {', '.join(bad)} (one of {', '.join(CLASSES + WORKERS)})")
    return [r for r in rows if r["class"] in want or r["worker"] in want] if want else rows


def summary(rows: list[dict]) -> dict:
    def tally(field: str, names: tuple) -> dict:
        return {n: {"functions": sum(r[field] == n for r in rows), "bytes": sum(r["size"] for r in rows if r[field] == n)}
                for n in names}
    return {"functions": len(rows), "bytes": sum(r["size"] for r in rows), "classes": tally("class", CLASSES),
            "workers": tally("worker", WORKERS)}


# ---- inputs -----------------------------------------------------------------------------------------

def load_attempts(ddir: Path) -> dict[str, dict]:
    """`unit::function` -> attempts over the grind's state files (one per night): how many, how many made no
    progress (idle), the best score, and a candidate id when one was kept."""
    out: dict[str, dict] = {}
    for path in sorted(ddir.glob("grind-state*.json")):
        try:
            done = json.loads(path.read_text()).get("done", {})
        except (OSError, ValueError, AttributeError):
            continue
        for key, r in done.items():
            if r.get("result") in ("skip", "stopped", "error"):
                continue
            a = out.setdefault(key, {"attempts": 0, "idle": 0, "best": 0.0, "candidate": None})
            before = r.get("before") or 0.0
            best = r.get("best") if r.get("best") is not None else before
            a["attempts"] += 1
            a["best"] = max(a["best"], best, before)
            if r.get("id"):
                a["candidate"] = r["id"]
            elif best <= before + 0.01:
                a["idle"] += 1
    return out


def load_sweep(path: Path) -> dict[str, str]:
    """source path (without `src/`) -> the variant that gains with no exact function lost, from `compiler-search --sweep`."""
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    out = {}
    for u in doc if isinstance(doc, list) else []:
        won = next((v["try"] for v in u.get("variants", []) if v.get("gains") and not v.get("lost")), None)
        if won:
            out[u["unit"]] = won
    return out


def sweep_path(cfg: Config) -> Path:
    from ...core.landing import common_dir
    from .compilersweep import CACHE
    here = cfg.root / CACHE
    if here.is_file():
        return here
    common = common_dir(cfg.root)
    return (common.parent if common.name == ".git" else cfg.root) / CACHE


def unit_diff(cfg: Config, unit: dict) -> tuple[dict, list[str]]:
    """(function -> (target instructions, our instructions), data sections below 100%) from one objdiff run."""
    from .objcache import objdiff_json
    objdiff = cfg.root / "build" / "tools" / "objdiff-cli"
    base = cfg.root / unit.get("base_path", "")
    if not objdiff.is_file() or not base.is_file():
        return {}, []
    doc, _ = objdiff_json(cfg, unit["target_path"], unit["base_path"])
    if doc is None:
        return {}, []
    ours = {s["name"]: s for s in doc.get("right", {}).get("symbols", []) if "instructions" in s}
    pairs = {s["name"]: (s["instructions"], ours[s["name"]]["instructions"])
             for s in doc["left"].get("symbols", []) if "instructions" in s and s["name"] in ours}
    data = [s["name"] for s in doc["left"].get("sections", [])
            if s.get("kind") in ("SECTION_DATA", "SECTION_BSS") and float(s.get("match_percent", 0.0)) < 100.0]
    return pairs, data


def table(cfg: Config, min_near: float = 80.0, promote: int = 2, sweep: Path | None = None,
          only: str = "", strict: bool = False) -> list[dict] | None:
    """The rows for every open function (None without a report); `only` keeps units or functions with that text."""
    from ...core.nextcmd import queue
    from ...core.progress import load_report
    from .llmtry import data_dir
    from .matching import load_objdiff_units
    report = load_report(cfg)
    if report is None:
        return None
    open_functions, _ = queue(report)
    if only:
        open_functions = [f for f in open_functions if only in f["unit"] or only in f["function"]]
    wanted = {f["unit"] for f in open_functions}
    diffs = {u["name"]: unit_diff(cfg, u) for u in load_objdiff_units(cfg) if u["name"] in wanted and u.get("base_path")}
    from . import blockers as bl
    from .gate import frozen_keys
    ddir = data_dir(cfg)
    frozen = frozen_keys(cfg)                                # the one union (benchmark and held-out included)
    return route_rows(open_functions, diffs, load_sweep(sweep or sweep_path(cfg)), load_attempts(ddir),
                      min_near, promote, frozen, bl.load(ddir), bl.playbook_version(cfg),
                      set(frozen), strict)


# ---- the command ------------------------------------------------------------------------------------

def render(rows: list[dict], top: int) -> list[str]:
    s = summary(rows)
    out = [f"{s['functions']} open function(s), 0x{s['bytes']:X} bytes", "", f"{'class':<17}{'functions':>10}{'bytes':>9}  worker"]
    for c in CLASSES:
        n = s["classes"][c]
        if n["functions"]:
            out.append(f"{c:<17}{n['functions']:>10}{n['bytes']:>9X}  {WORKER[c]}")
    out += ["", "queue per worker (idle nights promote a function to claude):"]
    for w, n in s["workers"].items():
        out.append(f"  {w:<8}{n['functions']:>5} function(s)  0x{n['bytes']:X} bytes")
    for c in CLASSES:
        part = sorted((r for r in rows if r["class"] == c), key=lambda r: (-r["fuzzy"], r["size"], r["address"]))
        if part and top:
            out += ["", f"{c}: {WHY[c]}"]
            out += [f"  {r['function']} 0x{r['size']:X} {r['fuzzy']:.2f}% -> {r['worker']}"
                    + (f" ({r['attempts']} attempt(s), {r['idle']} idle)" if r["attempts"] else "")
                    + (f" [candidate {r['candidate']}]" if r["candidate"] else "") + f"  {r['unit']}" for r in part[:top]]
            out += [f"  ... {len(part) - top} more (--class {c} --top 0)"] if len(part) > top else []
    return out


def run_blockers(cfg: Config, args: argparse.Namespace) -> int:
    """--record FUNCTION TAG and --backfill LOG…: write blockers.jsonl."""
    from . import blockers as bl
    from .llmtry import data_dir
    ddir, version, day = data_dir(cfg), bl.playbook_version(cfg), args.date or bl.today()
    try:
        if args.record:
            fn, tag = args.record
            n = bl.append(ddir, [bl.make_row(fn, tag, version, day, args.outcome, args.best, args.note)])
            print(f"recorded {fn}: {tag} ({args.outcome}) under playbook {version} in {bl.path_of(ddir)}")
            return OK
        written, skipped = bl.backfill(ddir, [Path(p) for p in args.backfill], version, day)
    except (ValueError, OSError) as e:
        print(f"error: {e}")
        return USAGE
    print(f"backfilled {written} row(s) ({skipped} already there) under playbook {version} into {bl.path_of(ddir)}")
    return OK


def run_fresh(rows: list[dict], version: str, args: argparse.Namespace) -> int:
    from . import blockers as bl
    fresh = [r for r in rows if r["fresh"]]
    days = bl.depletion(len(fresh), args.per_day)
    if args.json:
        print(json.dumps({"playbook_version": version, "fresh": len(fresh), "bytes": sum(r["size"] for r in fresh),
                          "attempts_per_day": args.per_day, "days_left": None if days == float("inf") else round(days, 2),
                          "rows": fresh if args.fresh else []}, indent=1))
        return OK
    if args.fresh:
        for r in sorted(fresh, key=lambda r: (-r["fuzzy"], r["size"], r["address"])):
            print(f"{r['function']} 0x{r['size']:X} {r['fuzzy']:.2f}% {r['kind']}"
                  + (f" (was {r['blocker']} under {r['playbook_version']})" if r["blocker"] else "") + f"  {r['unit']}")
    print(f"fresh: {len(fresh)} function(s), 0x{sum(r['size'] for r in fresh):X} bytes (playbook {version})")
    if args.depletion:
        print(f"at {args.per_day:g} attempt(s)/day: {days:.1f} day(s) left"
              + ("; under one day: move the grind budget to clusters" if days < 1 else ""))
    return OK


def unlock_ranking(cfg: Config, rows: list[dict], thaw: tuple) -> dict:
    """unlock.rank over the report and these rows, with the CPU attempts from the close-queue logs and memos."""
    from ...core.landing import common_dir
    from ...core.progress import load_report
    from . import unlock
    from .llmtry import data_dir
    common = common_dir(cfg.root)
    main = common.parent if common.name == ".git" else cfg.root
    memos = sorted({cfg.root / "build" / "dk" / "close", main / "build" / "dk" / "close"})
    tried = unlock.tried_counts(sorted(data_dir(cfg).glob("close-queue-*.jsonl")), memos)
    from .closedata import CENSUS
    census_path = cfg.root / CENSUS
    census = json.loads(census_path.read_text()) if census_path.is_file() else None
    return unlock.rank(load_report(cfg) or {}, rows, cfg.platform_options.get("unlock", {}), thaw, tried, census)


def run_unlock(cfg: Config, rows: list[dict], args: argparse.Namespace) -> int:
    from . import unlock
    thaw = tuple(t for t in args.thaw.split(",") if t) if args.thaw is not None else unlock.THAW
    ranking = unlock_ranking(cfg, rows, thaw)
    if args.json:
        print(json.dumps(ranking, indent=1))
        return OK
    print("\n".join(unlock.render(ranking, args.top, args.data)))
    hint("close --queue --thaw gate-lora2,near-sample-60,near-sample-fresh-30", "gen --data <unit>", "diff <unit> --literals")
    return OK


def run(cfg: Config, args: argparse.Namespace) -> int:
    if args.record or args.backfill:
        return run_blockers(cfg, args)
    rows = table(cfg, args.min_near, args.promote, Path(args.sweep) if args.sweep else None, args.unit, args.strict)
    if rows is None:
        print("error: no objdiff report yet (./dk build)")
        return USAGE
    if args.unlock:
        return run_unlock(cfg, rows, args)
    try:
        rows = select(rows, args.klass)
    except ValueError as e:
        print(f"error: {e}")
        return USAGE
    if args.fresh or args.depletion:
        from . import blockers as bl
        return run_fresh(rows, bl.playbook_version(cfg), args)
    if args.json:
        from . import blockers as bl
        print(json.dumps({**summary(rows), "playbook_version": bl.playbook_version(cfg), "rows": rows}, indent=1))
        return OK
    print("\n".join(render(rows, len(rows) if args.klass or not args.top else args.top)))
    hint("llm-grind --class near-register", "close --near")
    return OK


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("--json", action="store_true", help="the summary and every row as JSON")
    p.add_argument("--class", dest="klass", default="", metavar="C[,C]", help="only these classes (or workers)")
    p.add_argument("--unit", default="", help="only units or functions whose name contains this")
    p.add_argument("--min-near", type=float, default=80.0, help="lowest %% for a near miss (default 80)")
    p.add_argument("--promote", type=int, default=2, help="idle grind attempts that promote a function to claude (default 2)")
    p.add_argument("--sweep", default="", help="the compiler-search --sweep JSON (default build/dk/sweep.json)")
    p.add_argument("--fresh", action="store_true", help="list the fresh pool: open, <=2 KB, near register/shape, not "
                   "frozen, never attempted or its blocker was recorded under another playbook version")
    p.add_argument("--strict", action="store_true", help="with --fresh: also count the local-model grind's state files "
                   "as attempts (default: only blockers.jsonl rows do)")
    p.add_argument("--depletion", action="store_true", help="print the fresh count and the days left at --per-day")
    p.add_argument("--per-day", type=float, default=300.0, metavar="N", help="attempts per day for --depletion (default 300)")
    p.add_argument("--record", nargs=2, metavar=("FUNCTION", "BLOCKER"), help="append a blocker row to blockers.jsonl "
                   "(tags: temp-temp cross-tier not-reachable device-only inline-policy float-reg frame scheduling "
                   "register-only branch-shape compare-form shape)")
    p.add_argument("--outcome", default="gave-up", help="with --record: gave-up (default), skipped, partial, landed")
    p.add_argument("--best", type=float, default=None, help="with --record: best %% reached")
    p.add_argument("--note", default="", help="with --record: a short note")
    p.add_argument("--date", default="", help="with --record/--backfill: the attempt date (default today)")
    p.add_argument("--backfill", nargs="+", metavar="LOG", help="append the rows of these grind logs (bullet or pipe form) "
                   "to blockers.jsonl, once")
    p.add_argument("--top", type=int, default=5, help="rows listed per class (default 5; 0: all)")
    p.add_argument("--unlock", action="store_true", help="unlinked units ranked by code linked per unit of blocker "
                   "cost (unlock.py: open functions, open data sections, layout; cost table overridable in dk.toml "
                   "[platform.unlock]), held units apart, and the linked %% per worker if its blockers were done")
    p.add_argument("--data", action="store_true", help="with --unlock: only the open data sections, per unit")
    p.add_argument("--thaw", default=None, metavar="SET[,SET]", help="with --unlock: frozen sets CPU tools may work "
                   "(default none since 2026-11-18: every eval set is frozen; a thawed function is still refused at commit "
                   "until it is in llm-data/eval/released.json; results never enter training data)")


COMMAND = Command("route", "matching", "classify every open function by its blocker (register, scheduling, shape, "
                  "unit data, __sinit, compiler, far, fresh, giant) and queue it for a worker", run, _configure)
