"""T63 D3: `lora fix --region`: rewrite one differing statement region of a giant, splice it, accept by score.

Outcomes logged per try (structured only): routed-close (register-only region), malformed (no code block),
unappliable (spliced function does not compile), applied-worse, applied-better, region-exact (function exact)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ...core.config import Config

CONTEXT = 3


def pick(regions: list[dict], want: str) -> dict | None:
    cand = [r for r in regions if r.get("class") != "exact"]
    if want != "auto":
        return next((r for r in regions if r.get("rid") == want), None)
    model = [r for r in cand if r.get("class") != "reg"] or cand        # register-only regions go to close last
    return max(model, key=lambda r: r.get("differing", 0), default=None)


def render_span(lines: list[str], r: dict, context: int = CONTEXT) -> tuple[str, int, int]:
    a, b = max(1, r["first"]), min(len(lines), r["last"])
    return "\n".join(lines[a - 1:b]), a, b


def splice(lines: list[str], a: int, b: int, new: str) -> str:
    return "\n".join(lines[:a - 1] + new.rstrip("\n").split("\n") + lines[b:]) + "\n"


def outcome(start: float, new: float | None, exact: bool, fence: bool) -> str:
    if not fence:
        return "malformed"
    if new is None:
        return "unappliable"
    if exact:
        return "region-exact"
    return "applied-better" if new > start else "applied-worse"


def region_fix(cfg: Config, args) -> int:
    from . import regions as rg
    from .llmtry import Scorer, client_for, extract, prepare, settings_from
    route = json.loads(subprocess.run(["./dk", "route", "--json"], cwd=cfg.root, capture_output=True, text=True).stdout)["rows"]
    row = next((r for r in route if r["function"] == args.fn), None)
    if not row:
        print(json.dumps({"error": "function not in the open queue"}))
        return 2
    want = args.region
    for h in args.hint:
        if h.startswith("region:"):
            want = h.split(":", 1)[1]
    from .matching import load_objdiff_units
    unit = next(u for u in load_objdiff_units(cfg) if u["name"] == row["unit"])
    info = rg.unit_regions(cfg, unit, [args.fn]).get(args.fn, {})
    res = {"fn": args.fn, "best_pct": info.get("percent", 0.0), "exact": False, "diff_rows": None, "candidate": None,
           "devices": False, "stuck": None, "outcomes": []}
    reg = pick(info.get("regions", []), want) if info.get("regions") else None
    if reg is None:
        res["error"] = info.get("error", "no region")
        print(json.dumps(res))
        return 2
    res["region"] = reg["rid"]
    if reg.get("class") == "reg":
        res["outcomes"].append("routed-close")              # register-only: close / the oracle, not the model
        print(json.dumps(res))
        return 1
    src = cfg.root / unit["metadata"]["source_path"]
    lines = src.read_text().split("\n")
    span, a, b = render_span(lines, reg)
    from .llmtry import prepare as _p
    st = settings_from(args_ns(args))
    _u, source, srcf, spn, locate, ctx = _p(cfg, row["unit"], args.fn, trim=True, clean=True, version=st.prompt)
    scorer = Scorer(cfg, _u, args.fn, srcf, spn, locate)
    start, _rows = scorer.baseline()
    start_off = srcf.find(ctx.current)
    client = client_for(args_ns(args))
    from .llmtry import model_ready
    if not model_ready(client, args.model):
        return 2
    prompt = (f"Rewrite lines {a}-{b} of {args.fn} (region {reg['rid']}, class {reg['class']}, "
              f"{reg.get('differing', 0)} differing rows) so the compiled code matches the target. "
              f"Reply with only the rewritten lines in a ```cpp block.\n```cpp\n{span}\n```\n"
              f"TOOLS: pct={start:.2f} " + " ".join(f"hint={h}" for h in args.hint))
    best = start
    n = max(1, args.budget)
    for k in range(n):
        msg = client.chat(args.model, [{"role": "user", "content": prompt}], {"temperature": 1.0, "num_predict": 2048, "seed": k}, False)
        code = extract(msg.get("message", {}).get("content", ""))
        code = code[0] if isinstance(code, tuple) else code
        if not code:
            res["outcomes"].append(outcome(start, None, False, False))
            continue
        new_def = splice(lines, a, b, code)
        text_def = new_def[start_off:start_off + len(ctx.current) + len(new_def) - len("\n".join(lines)) - 1] if start_off >= 0 else None
        ev = scorer.evaluate(text_def, f"region{k}") if text_def else {}
        pct = ev.get("percent") if ev.get("compiled") else None
        o = outcome(start, pct, bool(ev.get("exact") or (pct or 0) >= 100.0), True)
        res["outcomes"].append(o)
        res["devices"] = res["devices"] or bool(ev.get("devices"))
        if pct is not None and pct > best:
            best, res["candidate"], res["diff_rows"] = pct, text_def, len(ev.get("rows") or [])
    res["best_pct"], res["exact"] = round(best, 3), best >= 100.0
    print(json.dumps(res) if getattr(args, "json", False) else f"{args.fn} {reg['rid']}: {res['outcomes']}")
    return 0 if res["exact"] else 1


def args_ns(args):
    import argparse
    ns = argparse.Namespace(**vars(args))
    defaults = dict(n=8, rounds=1, think="off", num_ctx=16384, num_predict=0, temperature=1.0, top_p=1.0, top_k=20, seed=1,
                    no_asm=False, no_decls=False, no_diff=False, m2c="auto", no_hints=True, no_prefill=False, min_p=0.05,
                    prompt=7, examples=0, backend="ollama")
    for k, v in defaults.items():
        if not hasattr(ns, k):
            setattr(ns, k, v)
    return ns
