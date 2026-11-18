"""dk llm-grind / llm-review / llm-land: the overnight queue, its pre-review and landing (P2 of
the LLM plan).

The grinder never edits the repo. It takes `dk next`'s open functions minus the benchmark and
held-out units and the sources other worktrees are editing, routes register-only near misses to
the CPU first (`close`'s regs moves, then its beam; judged, never applied), the rest and any CPU
miss to `llm-try`'s search, and writes each exact, judged result as a candidate under
`llm-data/candidates/<date>/`. With `--keep-near P` (default 90) a miss whose best variant compiles,
has no device, passes the judge's review clean and scores at least P% (and, for a written function,
`--near-gain` points more than now) is kept too, as a `near` candidate: WIP progress.
`llm-review` sorts them with the judge's `review` into a digest (near ones in their own section);
`llm-land` applies approved ones and commits through `save`/`accept`; a near one only ever
through `save`, its before -> after score in the note.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from ...core.command import MISMATCH, OK, USAGE, Command, hint
from ...core.config import Config
from . import llmtry
from .llmtry import data_dir

DEFAULT_MODEL = "qwen3-coder:30b"
BANDS = (0x40, 0x80, 0x100, 0x200, 0x400, 0x800)
STATE = "grind-state.json"
LABELS = "labels.jsonl"


# ---- the queue --------------------------------------------------------------------------------------

def band(size: int) -> int:
    return next((b for b in BANDS if size <= b), 1 << 20)


def success_rates(logs: list[Path], sizes: dict[str, int]) -> dict[tuple[str, int], float]:
    """Exact rate per (kind, size band) from llm-bench logs: `unwritten` from held-out and unwritten
    benchmark rows, `near` from register/shape ones; smoothed toward 10%."""
    seen: dict[tuple[str, int], list[int]] = {}
    for path in logs:
        for line in path.read_text(errors="replace").splitlines():
            if '"kind": "function"' not in line:
                continue
            r = json.loads(line)
            size = sizes.get(r.get("function", ""))
            if size is None:
                continue
            kind = "unwritten" if r.get("bench_kind") in ("heldout", "unwritten") else "near"
            n = seen.setdefault((kind, band(size)), [0, 0])
            n[0] += bool(r.get("exact"))
            n[1] += 1
    return {k: (e + 0.2) / (n + 2) for k, (e, n) in seen.items()}


def bench_sizes(ddir: Path) -> dict[str, int]:
    out = {}
    for name in ("bench.json", "heldout.json"):
        p = ddir / name
        if p.is_file():
            out.update({r["function"]: r["size"] for r in json.loads(p.read_text())["functions"] if "size" in r})
    return out


def queue(open_functions: list[dict], exclude_units: set[str], busy_sources: set[str], rates: dict,
          min_near: float = 80.0, max_size: int = 0x400, frozen=()) -> list[dict]:
    """Work items from `dk next`'s open functions, best first: bytes gained x the predicted exact rate
    of the kind and size band (size order when there are no rates). A near miss gains its whole size
    (its last bytes decide whether the function, and so its unit, can link)."""
    items = []
    for f in open_functions:
        name, size, pct = f["function"], f["size"], f["fuzzy"]
        if name.startswith(("@", "__sinit")) or f["unit"] in exclude_units or f.get("source") in busy_sources \
                or (f["unit"], name) in frozen:                    # frozen: the LoRA-2 gate and near-samples (gate.py)
            continue
        if not 8 <= size <= max_size or not f.get("source", "").endswith((".c", ".cpp", ".cp")):
            continue
        kind = "unwritten" if pct == 0.0 else "near"
        if kind == "near" and pct < min_near:
            continue
        left = size if kind == "near" else size * (100.0 - pct) / 100.0     # a near miss gains its whole size
        rate = rates.get((kind, band(size)), min(rates.values()) if rates else 0.0)
        items.append({**f, "kind": kind, "priority": round(left * rate, 2)})
    if rates:
        items.sort(key=lambda r: (-r["priority"], r["size"], r["address"]))
    else:
        items.sort(key=lambda r: (r["size"], r["address"]))
    return items


def busy_sources(cfg: Config) -> set[str]:
    """Paths with uncommitted edits in the other worktrees (another agent is on them)."""
    from ...core import landing
    out = set()
    for w in landing.worktrees(cfg.root, with_meta=False):
        path = Path(w["path"])
        if path.resolve() == cfg.root.resolve() or path.name.startswith("verify-") or not path.is_dir():
            continue
        proc = subprocess.run(["git", "status", "--porcelain", "--", "src"], cwd=path, capture_output=True, text=True)
        out |= {line[3:].strip() for line in proc.stdout.splitlines() if line.strip()}
    return out


# ---- control ----------------------------------------------------------------------------------------

def other_models(client: llmtry.Ollama, model: str, also=()) -> list[str]:
    """Loaded models other than `model` (and `also`: the grind's own overseer)."""
    mine = {model, *also}
    return [m.get("name", "?") for m in client.loaded() if not mine & {m.get("name"), m.get("model")}]


def wait_gpu(client: llmtry.Ollama, model: str, until: float, say=print, nap: float = 60.0, also=()) -> bool:
    """Sleep while Ollama has another model loaded (someone else is using the GPU). False when the
    stop time comes first."""
    told = False
    while True:
        others = other_models(client, model, also)
        if not others:
            return True
        if until and time.time() + nap >= until:
            return False
        if not told:
            say(f"  waiting: Ollama has {', '.join(others)} loaded")
            told = True
        time.sleep(nap)


def load_state(path: Path, resume: bool, run: str, date: str) -> dict:
    if resume and path.is_file():
        return json.loads(path.read_text())
    return {"run": run, "date": date, "done": {}}


def save_state(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    os.replace(tmp, path)


# ---- candidates -------------------------------------------------------------------------------------

def unified(before: str, after: str, path: str, n: int = 2) -> str:
    return "\n".join(difflib.unified_diff(before.splitlines(), after.splitlines(), "a/" + path, "b/" + path,
                                          n=n, lineterm=""))


def write_candidate(cdir: Path, rec: dict) -> str:
    """The candidate as <id>.json (before and after source, so review and land need nothing else)."""
    cid = hashlib.sha1(f"{rec['unit']}\0{rec['function']}\0{rec['text']}".encode()).hexdigest()[:8]
    cdir.mkdir(parents=True, exist_ok=True)
    rec = {"id": cid, **rec, "base_sha1": hashlib.sha1(rec["before_text"].encode()).hexdigest(),
           "diff": unified(rec["before_text"], rec["text"], rec["source"])}
    (cdir / f"{cid}.json").write_text(json.dumps(rec, indent=1) + "\n")
    return cid


def find_candidate(ddir: Path, cid: str) -> Path | None:
    hits = sorted((ddir / "candidates").glob(f"*/{cid}*.json"))
    return hits[-1] if len(hits) == 1 or (hits and all(h.name == hits[0].name for h in hits)) else None


def add_label(ddir: Path, cid: str, label: str, **extra) -> None:
    p = ddir / "candidates" / LABELS
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as fh:
        fh.write(json.dumps({"id": cid, "label": label, "time": round(time.time()), **extra}) + "\n")


# ---- one item ---------------------------------------------------------------------------------------

def cpu_route(cfg: Config, unit: dict, symbol: str, budget: int, jobs: int, log) -> tuple[str | None, list[dict]]:
    """`regs --try`'s moves, then `close`'s beam, both judged and never applied: (exact text, steps)."""
    from .closecmd import Options, baseline_options, close_one
    steps = []
    for name, opts in (("regs", baseline_options(budget, jobs)), ("close", Options(budget=budget, jobs=jobs))):
        res = close_one(cfg, unit, symbol, opts, False, log=lambda *_: None)
        step = {"step": name, "best": res.best, "exact": res.exact, "compiles": res.compiles,
                "secs": round(res.secs, 1), "error": res.error, "reasons": res.reasons[:4]}
        steps.append(step)
        log({"kind": "cpu", "unit": unit["name"], "function": symbol, **step})
        if res.exact and res.text:
            return res.text, steps
        if res.error:
            break
    return None, steps


def work(cfg: Config, item: dict, args, st: llmtry.Settings, client, log, until: float, say=print) -> dict:
    """One function: (re)score it privately, route it, and return its outcome (with `candidate`
    holding the record for an exact one)."""
    from .regs import _diff
    unit, source, src, span, locate, ctx = llmtry.prepare(cfg, item["unit"], item["function"], version=st.prompt,
                                                          retrieval=getattr(args, "index", None), examples=st.examples)
    out = {"unit": unit["name"], "function": item["function"], "size": item["size"], "source": cfg.rel(source)}
    if ctx.current and re.match(r"\s*(?:static\s+|extern\s+)*asm\b", ctx.current):
        return {**out, "result": "skip", "why": "asm function"}
    scorer = llmtry.Scorer(cfg, unit, item["function"], src, span, locate)
    scorer.repair = getattr(st, "repair", False)
    start, rows = scorer.baseline()
    out["before"] = start
    if start == 100.0:
        return {**out, "result": "skip", "why": "already exact"}
    kind = "unwritten" if not ctx.current else "shape"
    base = cfg.root / unit.get("base_path", "")
    if ctx.current and unit.get("base_path") and base.is_file():
        pair = _diff(cfg, unit, base, item["function"])
        kind = llmtry.classify(*pair) if pair else "shape"
    out["class"] = kind
    text, route = None, None
    if kind == "register" and not args.no_cpu:
        text, out["cpu"] = cpu_route(cfg, unit, item["function"], args.cpu_budget, args.jobs, log)
        route = "cpu" if text else None
        say(f"    cpu: {'EXACT' if text else 'best %.2f%%' % max(s['best'] for s in out['cpu'])}")
    if text is None and not args.cpu_only:
        if not wait_gpu(client, st.model, until, say, also=(args.overseer,) if getattr(args, "overseer", "") else ()):
            return {**out, "result": "stopped", "why": "stop time while waiting for the GPU"}
        res = llmtry.search(ctx, ctx.current, start, rows, scorer.evaluate, client, st, log,
                            {"unit": unit["name"], "grind": True, "grind_kind": kind},
                            jobs=min(st.n, args.jobs), say=lambda *_: None)
        out.update(best=res.best, attempts=res.attempts)
        text, route = (res.exact_text, "llm") if res.exact_code else (None, None)
    if text is None:
        near = None
        if route is None and not args.cpu_only and getattr(args, "keep_near", 0) and "best" in out \
                and wants_near(kind, start, out["best"], args.keep_near, args.near_gain) and res.best_code:
            ev = scorer.evaluate(res.best_code, "near")
            from .judge import unit_probe
            probe = unit_probe(cfg, unit, item["function"], scorer.work / "near_casts")
            near = near_ok(ev, src, item["function"], start, args.keep_near, scorer.forbidden, probe)
        if near is None:
            return {**out, "result": "miss"}
        out.update(result="near", route="llm")
        out["candidate"] = {"unit": unit["name"], "source": cfg.rel(source), "function": item["function"],
                            "size": item["size"], "kind": "near", "class": kind, "route": "llm", "before": start,
                            "after": ev["percent"], "model": st.model, "before_text": src, "text": ev["text"],
                            "judge": [], "flags": near, "time": time.strftime("%Y-%m-%d %H:%M")}
        return out
    from .judge import review
    out.update(result="exact", route=route)
    out["candidate"] = {"unit": unit["name"], "source": cfg.rel(source), "function": item["function"],
                        "size": item["size"], "kind": kind, "route": route, "before": start, "after": 100.0,
                        "model": st.model if route == "llm" else None, "before_text": src, "text": text,
                        "judge": [], "time": time.strftime("%Y-%m-%d %H:%M"),
                        "review": list(review(src, text, item["function"], scorer.forbidden))}
    return out


def wants_near(kind: str, start: float, best: float, keep: float, gain: float) -> bool:
    """A miss worth keeping as WIP progress: at least `keep`% and better than now; for a written
    function, also at least `gain` points better."""
    return best >= keep and best > start and (kind == "unwritten" or best - start >= gain)


def near_ok(ev: dict, src: str, symbol: str, start: float, keep: float, forbidden=(), probe=None) -> list[str] | None:
    """[] for a near candidate the judge passes clean (compiles, no device, nothing else in the
    unit scores lower, at least `keep`% and better than now, and `review` finds nothing: not even
    a needs-judgment flag), else None."""
    if not ev.get("compiled") or ev.get("devices") or ev.get("worse") or ev.get("percent", 0.0) < keep \
            or ev.get("percent", 0.0) <= start:
        return None
    from .judge import CLEAN, review
    category, flags = review(src, ev["text"], symbol, forbidden, probe)
    return [] if category == CLEAN else None


# ---- dk llm-grind -----------------------------------------------------------------------------------

def run_grind(cfg: Config, args: argparse.Namespace) -> int:
    from ...core.nextcmd import queue as next_queue
    from ...core.progress import load_report
    report = load_report(cfg)
    if report is None:
        print("error: no objdiff report yet (./dk build)")
        return USAGE
    st = llmtry.settings_from(args)
    args.index = llmtry.retrieval_for(cfg, st)
    ddir = data_dir(cfg)
    ddir.mkdir(parents=True, exist_ok=True)
    state_path = ddir / STATE
    state = load_state(state_path, args.resume, args.run or f"grind-{time.strftime('%Y%m%d-%H%M')}",
                       time.strftime("%Y-%m-%d"))
    path, fh, log = llmtry.open_log(cfg, state["run"])
    exclude = llmtry.never_train(cfg)[0]
    busy = busy_sources(cfg)
    logs = [p for p in ddir.glob("*.jsonl") if p.name.startswith(("night", "bake", "bench"))]
    rates = success_rates(logs, bench_sizes(ddir)) if not args.size_order else {}
    open_functions, _ = next_queue(report)
    from .gate import frozen_keys
    frozen = frozen_keys(cfg)
    items = queue(open_functions, exclude, busy, rates, args.min_near, args.max_size, frozen)
    if args.only:
        items = [r for r in items if r["kind"] == args.only]
    if getattr(args, "klass", ""):                    # `dk route`'s classes (or workers), from the same free data
        from . import route
        try:
            keep = {(r["unit"], r["function"]) for r in route.select(route.table(cfg, args.min_near) or [], args.klass)}
        except ValueError as e:
            print(f"error: {e}")
            return USAGE
        items = [r for r in items if (r["unit"], r["function"]) in keep]
    until = llmtry._until(args.until)
    client = llmtry.client_for(args)
    if not llmtry.model_ready(client, st.model):
        return USAGE
    cdir = ddir / "candidates" / state["date"]
    log({"kind": "run", "grind": True, "model": st.model, "options": st.options(st.seed), "n": st.n,
         "rounds": st.rounds, "prompt_version": st.prompt, "examples": st.examples, "queue": len(items), "busy": sorted(busy),
         **llmtry.search_tag(st),
         "rates": {f"{k}/{b:#x}": round(v, 3) for (k, b), v in sorted(rates.items())}})
    print(f"{len(items)} item(s) in the queue ({len(busy)} source(s) busy in other worktrees, "
          f"{len(state['done'])} done before); {st.model} {st.n} x {st.rounds}; log {cfg.rel(path)}")
    worked = found = 0
    try:
        for item in items:
            key = f"{item['unit']}::{item['function']}"
            if key in state["done"]:
                continue
            if until and time.time() >= until:
                print(f"stopping at {args.until} (--until); the same command with --resume goes on")
                break
            if args.max and worked >= args.max:
                break
            worked += 1
            print(f"[{worked}] {item['kind']} {item['function']} 0x{item['size']:X} {item['fuzzy']:.2f}% ({item['unit']})")
            t0 = time.time()
            try:
                out = work(cfg, item, args, st, client, log, until)
            except Exception as e:                    # one bad item must not end the night
                out = {"unit": item["unit"], "function": item["function"], "result": "error", "why": str(e)[:300]}
            out["secs"] = round(time.time() - t0, 1)
            cand = out.pop("candidate", None)
            if cand is not None and args.overseer:   # its verdict is logged and shown; the judge still decides
                cand["overseer"] = ask_overseer(cfg, client, args.overseer, cand, args.num_ctx)
            if cand is not None:
                out["id"] = write_candidate(cdir, {**cand, "run": state["run"]})
                if cand.get("overseer"):
                    log({"kind": "overseer", "id": out["id"], "function": cand["function"], **cand["overseer"]})
                found += 1
            if out["result"] == "stopped":
                print("    stop time while waiting for the GPU")
                break
            print(f"    {out['result']}" + (f" [{out['class']}]" if out.get("class") else "")
                  + (f" ({out.get('why')})" if out.get("why") else "")
                  + (f" -> candidate {out['id']} via {out['route']}" if "id" in out else "")
                  + (f", best {out['best']:.2f}%" if "best" in out and "id" not in out else ""))
            log({"kind": "grind", **out})
            state["done"][key] = {k: out.get(k) for k in ("result", "route", "id", "before", "best", "secs")}
            save_state(state_path, state)
    finally:
        fh.close()
    print(f"{worked} worked, {found} candidate(s) in {cdir}")
    hint(f"llm-review {state['date']}")
    return OK


def _configure_grind(p: argparse.ArgumentParser) -> None:
    llmtry._model_args(p, model=DEFAULT_MODEL)
    p.set_defaults(repair=True, rank="shape")
    p.add_argument("--until", default="", help="stop starting new items at HH:MM")
    p.add_argument("--resume", action="store_true", help=f"go on from llm-data/{STATE} (same run, log and date)")
    p.add_argument("--max", type=int, default=0, help="most items to work (0: no limit)")
    p.add_argument("--only", choices=("unwritten", "near"), help="only this kind of item")
    p.add_argument("--class", dest="klass", default="", metavar="C[,C]", help="only functions `dk route` puts in these "
                   "classes (near-register, near-shape, fresh, ...) or workers (cpu, lora)")
    p.add_argument("--min-near", type=float, default=80.0, help="lowest %% for a near miss (default 80)")
    p.add_argument("--max-size", type=lambda x: int(x, 0), default=0x400, help="largest function (default 0x400)")
    p.add_argument("--size-order", action="store_true", help="smallest first, not by predicted bytes gained")
    p.add_argument("--jobs", type=int, default=6, help="parallel compiles (default 6)")
    p.add_argument("--cpu-budget", type=int, default=300, help="most compiles per CPU step (default 300)")
    p.add_argument("--no-cpu", action="store_true", help="every item to the model (A/B the routing)")
    p.add_argument("--cpu-only", action="store_true", help="no model: register misses on the CPU only")
    p.add_argument("--keep-near", type=float, default=90.0, help="also keep a non-exact result the judge passes clean "
                   "at this %% or more as a `near` candidate (WIP progress, landed with save; 0: off; default 90)")
    p.add_argument("--near-gain", type=float, default=5.0, help="for a written function, a near candidate must also "
                   "gain this many points (default 5)")
    p.add_argument("--overseer", default="", help="also ask this model (review prompt r1, same server) for a verdict, "
                   "rank and reason on each candidate, kept with it and logged; it never overrides the judge (default off)")


GRIND = Command("llm-grind", "matching", "overnight queue: open functions (not benchmark, held-out or busy), "
                "register misses to regs/close on the CPU first, the rest to llm-try's search; exact judged "
                "results become candidates in llm-data/ (never the repo)", run_grind, _configure_grind)


# ---- dk llm-review ----------------------------------------------------------------------------------

ORDER = {"clean": 0, "needs-judgment": 1, "reject": 2}


def ask_overseer(cfg: Config, client, model: str, rec: dict, num_ctx: int = 16384) -> dict:
    """The overseer's review of one candidate (llmreview's r1 prompt, with the judge's flags), plus the
    judge's own category so agreement can be measured."""
    from . import llmreview
    category, flags = review_candidate(rec)
    diff = rec.get("diff") or unified(rec["before_text"], rec["text"], rec["source"])
    got = llmreview.ask(client, model, llmreview.candidate_item(cfg, {**rec, "diff": diff}, flags), options={"num_ctx": num_ctx})
    return {"model": model, "review_prompt": llmreview.REVIEW_PROMPT, "judge": category,
            **{k: got.get(k) for k in ("verdict", "rank", "why", "parsed", "secs", "error") if got.get(k) is not None}}


def review_candidate(rec: dict, forbidden=(), probe=None) -> tuple[str, list[str]]:
    from .judge import review
    category, flags = review(rec["before_text"], rec["text"], rec["function"], forbidden, probe)
    if rec.get("judge"):
        return "reject", list(rec["judge"]) + flags
    return category, flags


def digest(recs: list[dict], max_diff: int = 40) -> str:
    """One short block per candidate, clean ones first; near (not exact) ones in their own section
    after the exact ones."""
    exact = [r for r in recs if r.get("kind") != "near"]
    near = [r for r in recs if r.get("kind") == "near"]
    counts = {c: sum(1 for r in exact if r["category"] == c) for c in ORDER}
    lines = [f"# Grind candidates ({len(exact)} exact: " + ", ".join(f"{n} {c}" for c, n in counts.items())
             + (f"; {len(near)} near" if near else "") + ")", "",
             "Approve: `./dk llm-land ID…`. Reject: `./dk llm-review --reject ID… --why TEXT`.", ""]
    lines += _blocks(exact, max_diff)
    if near:
        lines += ["# Near candidates (not exact: llm-land saves them as WIP progress, never accept)", ""]
        lines += _blocks(near, max_diff)
    return "\n".join(lines) + "\n"


def _blocks(recs: list[dict], max_diff: int) -> list[str]:
    lines = []
    for r in sorted(recs, key=lambda r: (ORDER[r["category"]], r["unit"], r["function"])):
        lines.append(f"## {r['id']} {r['category']}: {r['function']}")
        after = f"{r['after']:.0f}%" if r["after"] == 100.0 else f"{r['after']:.2f}%"
        lines.append(f"{r['unit']}, 0x{r['size']:X} bytes, {r['before']:.2f}% -> {after}, "
                     f"{r['kind']}, via {r['route']}" + (f" ({r['model']})" if r.get("model") else ""))
        if r["flags"]:
            lines.append("Flags: " + "; ".join(r["flags"]))
        o = r.get("overseer")
        if o:
            lines.append(f"Overseer ({o.get('model')}, never the gate): {o.get('verdict') or 'unparsed'}"
                         + (f", rank {o['rank']}" if o.get("rank") is not None else "") + (f": {o['why']}" if o.get("why") else ""))
        diff = r["diff"].splitlines()[2:]
        if len(diff) > max_diff:
            diff = diff[:max_diff] + [f"... {len(diff) - max_diff} more lines"]
        lines += ["```diff", *diff, "```", ""]
    return lines


def run_review(cfg: Config, args: argparse.Namespace) -> int:
    from .judge import forbidden_rules
    ddir = data_dir(cfg)
    if getattr(args, "audit", False):
        from .judgeaudit import run_audit
        run_audit(cfg, getattr(args, "jobs", 4), getattr(args, "only", ""))
        return OK
    if getattr(args, "agreement", False):
        from .llmreview import agreement
        agreement(ddir)
        return OK
    for ids, label in ((args.reject, "rejected"), (getattr(args, "approve", None), "approved")):
        if not ids:
            continue
        for cid in ids:
            p = find_candidate(ddir, cid)
            if p is None:
                print(f"error: no single candidate {cid}")
                return USAGE
            add_label(ddir, json.loads(p.read_text())["id"], label, why=args.why, **({"how": "review"} if label == "approved" else {}))
        print(f"labelled {len(ids)} {label}")
        return OK
    root = ddir / "candidates"
    dates = sorted(p.name for p in root.glob("????-??-??") if p.is_dir()) if root.is_dir() else []
    date = args.date or (dates[-1] if dates else "")
    cdir = root / date
    files = sorted(cdir.glob("*.json")) if date else []
    if not files:
        print(f"no candidates in {cdir if date else root}")
        return OK
    from .judge import unit_probe
    from .matching import resolve_unit
    forbidden = forbidden_rules(cfg)
    recs = []
    for p in files:
        rec = json.loads(p.read_text())
        try:
            probe = unit_probe(cfg, resolve_unit(cfg, rec["unit"]), rec["function"],
                               cfg.root / llmtry.LLM_DIR / "work" / f"review_{rec['id']}")
        except Exception:                           # the unit is gone: no cast check, the rest still runs
            probe = None
        rec["category"], rec["flags"] = review_candidate(rec, forbidden, probe)
        if getattr(args, "overseer", "") and not rec.get("overseer"):   # asked once, kept with the candidate
            rec["overseer"] = ask_overseer(cfg, llmtry.client_for(args), args.overseer, rec)
            p.write_text(json.dumps({k: v for k, v in rec.items() if k not in ("category", "flags")}, indent=1) + "\n")
        recs.append(rec)
    out = cdir / "digest.md"
    out.write_text(digest(recs, args.max_diff))
    for r in sorted(recs, key=lambda r: (ORDER[r["category"]], r["unit"])):
        print(f"  {r['id']}  {r['category']:<14} {r['route']:<3} {r['before']:6.2f}%"
              + (f" -> {r['after']:6.2f}% near" if r.get("kind") == "near" else "") + f"  {r['function']}"
              + (f"  [{'; '.join(r['flags'])[:80]}]" if r["flags"] else ""))
    print(f"digest: {out}")
    return OK


def _configure_review(p: argparse.ArgumentParser) -> None:
    p.add_argument("date", nargs="?", default="", help="candidates/<YYYY-MM-DD> (default: the latest)")
    p.add_argument("--max-diff", type=int, default=40, help="diff lines per block (default 40)")
    p.add_argument("--reject", nargs="+", metavar="ID", help="label these candidates rejected (no digest)")
    p.add_argument("--approve", nargs="+", metavar="ID", help="label these candidates approved without landing them "
                   "(a training label for the overseer; llm-land labels what it lands)")
    p.add_argument("--why", default="", help="with --reject/--approve: the reason, kept with the label")
    p.add_argument("--overseer", default="", help="ask this model for its verdict on candidates that have none (kept "
                   "with them, shown in the digest; never the gate)")
    p.add_argument("--url", default=os.environ.get("DK_OLLAMA", llmtry.DEFAULT_URL), help="with --overseer: the server")
    p.add_argument("--backend", choices=("ollama", "openai"), default="ollama", help="with --overseer: the server's API")
    p.add_argument("--agreement", action="store_true", help="instead: the overseer's verdicts against the judge's "
                   "category and the coordinator's labels, over every candidate")
    p.add_argument("--audit", action="store_true", help="instead: the same rules over every exact function's current "
                   "source (as if written into an empty definition) into llm-data/judge-audit.json: the judge's false "
                   "positives on accepted code, and the answers a training export quarantines")
    p.add_argument("--jobs", type=int, default=4, help="with --audit: units at once (default 4)")
    p.add_argument("--only", default="", help="with --audit: units whose name contains this")


REVIEW = Command("llm-review", "matching", "pre-review grind candidates with the judge's review (reject / needs "
                 "judgment / clean) into llm-data/candidates/<date>/digest.md", run_review, _configure_review)


# ---- dk llm-land ------------------------------------------------------------------------------------

def dk(cfg: Config, *argv: str) -> int:
    print("  $ ./dk " + " ".join(argv))
    return subprocess.run([sys.executable, "-m", "tools.dk", *argv], cwd=cfg.root).returncode


def landed_text(cfg: Config, rec: dict, current: str) -> tuple[str | None, str]:
    """The source to write: the candidate as it is when the file has not moved, else only its
    definition spliced into today's file (a clean candidate only)."""
    from .trycmd import definition, locate_target, variant_text
    if hashlib.sha1(current.encode()).hexdigest() == rec["base_sha1"]:
        return rec["text"], "as reviewed"
    category, _ = review_candidate(rec)
    if category != "clean":
        return None, f"the source changed since the grind and the candidate is {category}: re-grind it"
    span = definition(rec["text"], rec["function"])
    chunk = rec["text"][span[0]:span[1]]
    cur_span, locate, _ = locate_target(cfg, current, cfg.root / rec["source"], rec["function"])
    return variant_text(current, rec["function"], chunk, cur_span, locate), "definition spliced into today's file"


def function_scores(cfg: Config, rec: dict, text: str) -> dict[str, float] | None:
    """Every function's score in the unit with `text` as its source (a private compile), or None
    when it does not compile."""
    from .matching import resolve_unit
    unit = resolve_unit(cfg, rec["unit"])
    s, _, _, _ = llmtry.Scorer(cfg, unit, rec["function"], text, None, None).compile_score(text, "land")
    return None if s is None else {f.name: f.percent for f in s.functions}


def near_check(rec: dict, before: dict[str, float] | None, after: dict[str, float] | None) -> str:
    """Why a near candidate cannot land here ("" when it can): it must compile, score higher than
    the function does today, and leave every other function where it was."""
    if before is None or after is None:
        return "does not compile here"
    fn = rec["function"]
    if after.get(fn, 0.0) <= before.get(fn, 0.0):
        return f"no gain here ({before.get(fn, 0.0):.2f}% -> {after.get(fn, 0.0):.2f}%)"
    worse = [n for n, p in before.items() if n != fn and after.get(n, -1.0) < p]
    return f"scores lower: {', '.join(worse[:4])}" if worse else ""


def run_land(cfg: Config, args: argparse.Namespace, run_dk=dk, score=function_scores) -> int:
    """Exact candidates: check the function exact, then accept (unit exact) or save the unit. Near
    candidates: score before and after privately, then only ever save, with before -> after in the note."""
    ddir = data_dir(cfg)
    touched: dict[str, list[str]] = {}
    notes: dict[str, list[str]] = {}
    near_units: set[str] = set()
    failed = 0
    for cid in args.ids:
        p = find_candidate(ddir, cid)
        if p is None:
            print(f"{cid}: no single candidate")
            failed += 1
            continue
        rec = json.loads(p.read_text())
        source = cfg.root / rec["source"]
        current = source.read_text()
        try:
            new, how = landed_text(cfg, rec, current)
        except Exception as e:                        # the function moved or vanished
            new, how = None, f"cannot place it: {e}"
        if new is None:
            print(f"{rec['id']}: {how}")
            failed += 1
            continue
        if rec.get("kind") == "near":
            fn = rec["function"]
            before = score(cfg, rec, current)
            after = score(cfg, rec, new)
            why = near_check(rec, before, after)
            if why:
                print(f"{rec['id']}: {fn} near candidate not landed: {why}")
                add_label(ddir, rec["id"], "land-failed", why=why)
                failed += 1
                continue
            source.write_text(new)
            note = f"{fn} {before.get(fn, 0.0):.2f}% -> {after[fn]:.2f}%"
            print(f"{rec['id']}: {note} (near, {how})")
            add_label(ddir, rec["id"], "approved")
            touched.setdefault(rec["unit"], []).append(fn)
            notes.setdefault(rec["unit"], []).append(note + f" (near: llm-grind candidate {rec['id']})")
            near_units.add(rec["unit"])
            continue
        source.write_text(new)
        if run_dk(cfg, "check", rec["unit"], "--function", rec["function"], "--private") != OK:
            source.write_text(current)
            print(f"{rec['id']}: {rec['function']} is not exact here; restored {rec['source']}")
            add_label(ddir, rec["id"], "land-failed")
            failed += 1
            continue
        print(f"{rec['id']}: {rec['function']} exact ({how})")
        add_label(ddir, rec["id"], "approved")
        touched.setdefault(rec["unit"], []).append(rec["function"])
        notes.setdefault(rec["unit"], []).append(f"{rec['function']} exact (llm-grind candidate {rec['id']})")
    for unit in touched:
        if args.no_commit:
            continue
        if unit not in near_units and run_dk(cfg, "check", unit, "--private") == OK:
            rc = run_dk(cfg, "accept", unit)
        else:                                       # a near candidate never goes through accept
            rc = run_dk(cfg, "save", unit, "--note", "llm-land: " + "; ".join(notes[unit]))
        failed += rc != OK
    if not touched:
        print("nothing landed")
    return OK if not failed else MISMATCH


def _configure_land(p: argparse.ArgumentParser) -> None:
    p.add_argument("ids", nargs="+", metavar="ID", help="approved candidate ids (from the digest)")
    p.add_argument("--no-commit", action="store_true", help="apply and check only; no save/accept")


LAND = Command("llm-land", "matching", "apply approved grind candidates in this checkout, check each, then "
               "accept (unit exact) or save the unit; a near candidate (not exact) only with save, its "
               "before -> after score in the note", run_land, _configure_land)
