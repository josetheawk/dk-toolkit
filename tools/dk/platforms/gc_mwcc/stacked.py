"""dk dataset stacked: multi-change training puzzles (T48). An exact, non-frozen function of HEAD gets 2-3 stacked
inverse moves of the rewrite table (shape-synth's SHAPE_TABLE plus the loop rows `for-init-list` and `iv-alias`),
each reversible by a table move (shapesynth.build_path), and EVERY state is compiled privately (-sym on), so the
record carries the score path of the fix: S_d (the puzzle) -> ... -> S_0 (the exact source, the answer).

  ./dk dataset stacked [--n 2000] [--per-fn 2] [--depth 2-3] [--jobs 4] [--seed S] [--out DIR] [--until HH:MM]
  ./dk dataset stacked --min-size 0x400 --max-size 0x4000 --depth 3-5 --per-fn 10 --window [--tokenizer tokenizer.json]

Giants (T52): `--depth A-B` draws the stack depth in A..B (weights falling by depth), `--window` adds to each record a
region patch (one SEARCH/REPLACE hunk per changed region, each SEARCH unique in the whole before definition, the
hunks applied in order verified to give the exact definition: `patch_verified`) and the window render (the
signature and declaration head, then the changed regions with context lines, `// ... lines a-b elided ...` for
the rest; the diff rows capped) that `rlgate.render(..., window=)` shows instead of the whole definition. With
`--tokenizer tokenizer.json` (Qwen's; run with `--tokenizer-python`, a venv with `tokenizers`) each record gets
`tokens`: the real token count of system + user + assister messages under Qwen's chat template. Records also carry
the real fix-kind shares next to the puzzles' (`kind_table` in stats.json, from shape-fixes v0).

Kept when: every state compiles, the puzzle scores below 100 %, the answer compiles to 100 % (measured in the same
run), the judge's source checks pass (shapesynth.clean) and the record passes the leakage filter (shape-synth's,
plus by function name: the real shape-fix records, llm-data/heldout-split.json dev and final, gate-lora2). A
"valley" is a forward step that gains nothing (its score <= the one before it): the reason single-change data
does not teach these fixes (RL3-gpu STUDY-transfer: 16/27 real fixes need 2+ changes). Tool output only: no
model text. Output: stacked.jsonl, stats.json, CARD.md, SHA256SUMS in --out (default llm-data/datasets/stacked/v0).
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import multiprocessing
import random
import re
import shutil
import subprocess
import time
from pathlib import Path

from ...core.command import OK, USAGE
from ...core.config import Config
from . import shapesynth as ss

SCHEMA = 1
LOOP_ROWS = ("for-init-list", "iv-alias")
SCOPE_ROWS = ("scope-local", "shared-tail")              # giantmoves (T49 card 1)
TABLE = ss.SHAPE_TABLE + LOOP_ROWS + SCOPE_ROWS
DEPTHS = ((2, 60), (3, 40))
CONTEXT = 3                                              # window: context lines around each hunk
HEAD_LINES = 12                                          # window: at most this many leading declaration lines
ROW_CAP = 40                                             # window: diff rows shown at most
REAL_KINDS = ("control-flow", "operator", "local-added", "local-removed", "literal", "identifier", "call-or-accessor")
_POOL: dict = {}
_DECL_LINE = re.compile(r"^\s*(?:const\s+|unsigned\s+|signed\s+|struct\s+|register\s+|static\s+)*[A-Za-z_][\w:<>]*"
                        r"\s*[*&]*\s+\**[A-Za-z_]\w*(?:\s*\[[^\]]*\])?(?:\s*=[^;]*)?;\s*$")


def depth_weights(spec: str) -> tuple[tuple[int, int], ...]:
    """`A-B` (or `A`) -> ((A, wA), ..., (B, wB)): weights fall by one share per step (3-5: 50/33/17)."""
    a, _, b = spec.partition("-")
    lo, hi = int(a), int(b or a)
    if lo < 1 or hi < lo:
        raise ValueError(f"--depth {spec}: want A-B with 1 <= A <= B")
    n = hi - lo + 1
    shares = list(range(n, 0, -1))
    total = sum(shares)
    return tuple((lo + i, round(100 * s / total)) for i, s in enumerate(shares))


def pick_depth(rng: random.Random, depths=DEPTHS) -> int:
    x = rng.randrange(100)
    for d, w in depths:
        if x < w:
            return d
        x -= w
    return depths[-1][0]


def valleys(path: list[float]) -> int:
    """Forward steps (path[j] -> path[j+1], puzzle first) that gain nothing."""
    return sum(1 for a, b in zip(path, path[1:]) if b <= a)


# ---- window render and region patch (giants) --------------------------------------------------------------

def _widen(b: list[str], a: list[str], i1: int, i2: int, j1: int, j2: int, cur: str):
    while True:
        old = "\n".join(b[i1:i2])
        if old and cur.count(old) == 1:
            return old, "\n".join(a[j1:j2]), i1, i2
        if i1 == 0 and i2 >= len(b):
            return None
        i1, j1 = max(0, i1 - 1), max(0, j1 - 1)
        if i2 < len(b):
            i2, j2 = i2 + 1, j2 + 1


def hunks(before: str, after: str) -> list[dict] | None:
    """The region patch: whole-line SEARCH/REPLACE hunks (nearby changes merged), each SEARCH unique in the text
    it is applied to, applied in order; None unless the result is `after` byte for byte. Falls back to one hunk
    (llmagent.fix_hunk's) when the regions cannot be told apart."""
    b, a = before.split("\n"), after.split("\n")
    sm = difflib.SequenceMatcher(None, b, a, autojunk=False)
    ops = [list(op) for op in sm.get_opcodes() if op[0] != "equal"]
    if not ops:
        return None
    groups: list[list] = []
    for op in ops:
        if groups and op[1] - groups[-1][2] <= 2 * CONTEXT:
            groups[-1][2], groups[-1][4] = op[2], op[4]
        else:
            groups.append(op)
    for attempt in (groups, [["replace", ops[0][1], ops[-1][2], ops[0][3], ops[-1][4]]]):
        cur, out, ok = before, [], True
        for _k, i1, i2, j1, j2 in attempt:
            got = _widen(b, a, i1, i2, j1, j2, cur)
            if got is None:
                ok = False
                break
            old, new, l1, l2 = got
            out.append({"search": old, "replace": new, "lines": [l1 + 1, l2]})
            cur = cur.replace(old, new, 1)
        if ok and cur == after:
            return out
    return None


def apply_hunks(before: str, patch: list[dict]) -> str | None:
    cur = before
    for h in patch:
        if cur.count(h["search"]) != 1:
            return None
        cur = cur.replace(h["search"], h["replace"], 1)
    return cur


def window(before: str, patch: list[dict], context: int = CONTEXT, head_lines: int = HEAD_LINES) -> tuple[str, list]:
    """The definition's signature and leading declarations, then each hunk's lines with `context` lines around it;
    elided stretches as `// ... lines a-b elided ...`. (text, the 1-based line ranges shown)."""
    lines = before.split("\n")
    n = len(lines)
    head = 1
    while head < min(n, head_lines + 1) and (_DECL_LINE.match(lines[head]) or not lines[head].strip()):
        head += 1
    ranges = [[1, head]]
    for h in patch:
        l1, l2 = h["lines"]
        ranges.append([max(1, l1 - context), min(n, l2 + context)])
    ranges.sort()
    merged = [ranges[0]]
    for lo, hi in ranges[1:]:
        if lo <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    out, last = [], 0
    for lo, hi in merged:
        if lo > last + 1:
            out.append(f"// ... lines {last + 1}-{lo - 1} elided ...")
        out.extend(lines[lo - 1:hi])
        last = hi
    if last < n:
        out.append(f"// ... lines {last + 1}-{n} elided ...")
    return "\n".join(out), merged


def render_window(p: dict) -> list[dict]:
    """The SFT prompt (rlgate.render, 0 examples) with the window, and the region patch as the reply."""
    from .llmagent import EDIT_FORMAT, Score  # noqa: F401 (the format the reply follows)
    from .rlgate import render
    rows = p["rows"][:ROW_CAP] + (["  … (more differences)"] if len(p["rows"]) > ROW_CAP else [])
    msgs = render(p["function"], p["before"], Score(True, p["before_percent"], (), rows), [], [], 0, window=p["window"])
    reply = "\n".join(f"<<<<<<< SEARCH\n{h['search']}\n=======\n{h['replace']}\n>>>>>>> REPLACE" for h in p["patch"])
    return msgs + [{"role": "assistant", "content": reply}]


_COUNTER = r"""
import json, sys
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[1])
for line in sys.stdin:
    msgs = json.loads(line)
    text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
    print(len(tok.encode(text).ids))
"""


def count_tokens(renders: list[list[dict]], tokenizer: str, python: str) -> list[int]:
    """Real token counts of chat renders under Qwen's template, in a Python that has `tokenizers`."""
    inp = "".join(json.dumps(m) + "\n" for m in renders)
    r = subprocess.run([python, "-I", "-c", _COUNTER, tokenizer], input=inp, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("token count failed: " + r.stderr.strip()[-300:])
    return [int(x) for x in r.stdout.split()]


def kind_table(puzzles: list[dict], real: list[dict]) -> dict:
    """Each fix kind's share of the puzzles next to its share of the real fixes (a record counts once per kind)."""
    from collections import Counter
    cp, cr = Counter(k for p in puzzles for k in p["fix_kind"]), Counter(k for r in real for k in r["fix_kind"])
    kinds = sorted(set(cp) | set(cr) | set(REAL_KINDS), key=lambda k: -cr.get(k, 0))
    return {k: {"puzzles": cp.get(k, 0), "puzzle_share": round(cp.get(k, 0) / max(1, len(puzzles)), 3),
                "real": cr.get(k, 0), "real_share": round(cr.get(k, 0) / max(1, len(real)), 3)} for k in kinds}


def real_fixes(cfg: Config) -> list[dict]:
    from .llmtry import data_dir
    fx = data_dir(cfg) / "datasets" / "shape-fixes" / "v0" / "fixes.jsonl"
    out = []
    if fx.is_file():
        for line in fx.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def one_function(cfg: Config, f: dict, seed: int, per_fn: int, work: Path, depths=DEPTHS, use_window: bool = False,
                 fold_draw: float = 0.0) -> dict:
    """Stacked puzzles of one exact function (deterministic for a seed and source)."""
    from .llmfixes import fix_kinds
    u = f["u"]
    src = (cfg.root / u["metadata"]["source_path"]).read_text(errors="replace")
    stats: dict = {}
    fn = ss.Fn(cfg, u, f["function"], src, ss.work_dir(work, f["function"], "stacked"))
    fn.table = TABLE
    res = {"function": f["function"], "unit": f["unit"], "puzzles": [], "stats": stats}
    try:
        d0 = fn.defn(src)
        m0 = fn.measure(src) if d0 is not None else {}
        if not m0.get("compiled") or m0.get("percent", 0) < 100.0:
            stats["not exact privately"] = 1
            return res
        rng = random.Random(ss.hkey(seed, "stacked", f["unit"], f["function"]))
        seen: set[str] = set()
        for _attempt in range(per_fn * 3):
            if len(res["puzzles"]) >= per_fn:
                break
            fold = rng.random() < fold_draw                               # T38's fold draw: the first step folds a local
            steps = ss.build_path(fn, src, pick_depth(rng, depths), rng, stats, fold_first=fold)
            if len(steps) < max(2, depths[0][0]):
                stats["fewer moves than --depth"] = stats.get("fewer moves than --depth", 0) + 1
                if stats["fewer moves than --depth"] >= 3:
                    break
                continue
            before = steps[-1]["text"]
            key = ss.norm(fn.defn(before) or "")
            if key in seen:
                stats["repeat"] = stats.get("repeat", 0) + 1
                continue
            seen.add(key)
            ms = [fn.measure(s["text"]) for s in steps]                 # S_1 .. S_d
            if not all(m.get("compiled") for m in ms):
                stats["a state does not compile"] = stats.get("a state does not compile", 0) + 1
                continue
            mb = ms[-1]
            if mb["percent"] >= 100.0:
                stats["same code"] = stats.get("same code", 0) + 1
                continue
            ok, flags, why = ss.clean(fn, before, src)
            if not ok:
                k = "not clean: " + why.split(":")[0]
                stats[k] = stats.get(k, 0) + 1
                continue
            b_def = fn.defn(before)
            path = [m["percent"] for m in reversed(ms)] + [m0["percent"]]   # puzzle first, answer last
            extra: dict = {}
            if use_window:
                patch = hunks(b_def, d0)
                if patch is None or apply_hunks(b_def, patch) != d0:
                    stats["no region patch"] = stats.get("no region patch", 0) + 1
                    continue
                win, shown = window(b_def, patch)
                extra = {"patch": patch, "patch_verified": True, "window": win, "window_lines": shown,
                         "regions": len(patch), "def_lines": b_def.count("\n") + 1,
                         "window_share": round(sum(b - a + 1 for a, b in shown) / (b_def.count("\n") + 1), 3)}
            res["puzzles"].append({
                "v": SCHEMA, "source": "stacked", "id": ss.sha("stacked", f["unit"], f["function"], b_def)[:16],
                "unit": f["unit"], "function": f["function"], "size": f["size"], "commit": ss.HEAD.get("rev", ""),
                "path": u["metadata"]["source_path"], "asm_shape": f["asm_shape"], "asm": f.get("asm", ""),
                "before": b_def, "after": d0, "before_percent": mb["percent"], "after_percent": m0["percent"],
                "answer_verified": m0["percent"] >= 100.0, "diff_class": mb.get("diff_class", ""),
                "rows": mb.get("rows") or [], "fix_kind": fix_kinds(b_def, d0), "mutant_shape": mb.get("shape", ""),
                "depth": len(steps), "transforms": [s["rule"] for s in steps],
                "forward": [s["forward"] for s in reversed(steps)],
                "steps": [{"rule": s["rule"], "label": s["label"]} for s in steps],
                "score_path": path, "valleys": valleys(path), "review_flags": flags[:6], **extra})
        stats["puzzles"] = len(res["puzzles"])
    except Exception as exc:                             # noqa: BLE001 (one function never stops the run)
        stats[f"error {type(exc).__name__}"] = 1
    finally:
        stats["compiles"] = fn.compiles
        shutil.rmtree(fn.work, ignore_errors=True)
    return res


def _work(i: int) -> dict:
    c = _POOL
    return one_function(c["cfg"], c["fns"][i], c["seed"], c["per_fn"], c["work"], c["depths"], c["window"], c["fold_draw"])


def never_functions(cfg: Config) -> dict[str, str]:
    """Function names of the sets the prereg names beyond llmfixes.frozen: the real shape-fix records and both
    halves of the held-out split (its units are already left out by unit)."""
    from .llmtry import data_dir
    d = data_dir(cfg)
    out: dict[str, str] = {}
    fx = d / "datasets" / "shape-fixes" / "v0" / "fixes.jsonl"
    if fx.is_file():
        for line in fx.read_text().splitlines():
            try:
                out.setdefault(json.loads(line)["function"], "shape-fixes")
            except (ValueError, KeyError):
                pass
    hs = d / "heldout-split.json"
    if hs.is_file():
        doc = json.loads(hs.read_text())
        for half in ("dev", "final", "excluded"):
            for r in doc.get(half, []):
                fn = r.get("function") if isinstance(r, dict) else r if isinstance(r, str) else None
                if fn:
                    out.setdefault(fn, f"heldout-{half}")
    return out


def leak(p: dict, fz: dict, extra: dict[str, str]) -> str | None:
    if p["function"] in extra:
        return extra[p["function"]]
    if p["function"] in fz["functions"] or p["unit"] in fz["units"]:
        return "frozen"
    return ss.leak_free(p, fz)


def size_band(size: int) -> str:
    return "<0x400" if size < 0x400 else "0x400-0x800" if size < 0x800 else "0x800-0x1000" if size < 0x1000 else ">=0x1000"


def summarize(puzzles: list[dict], stats: dict, real: list[dict], args) -> None:
    """Into stats: fix-kind table vs the real fixes, size bands, depth, token fit, window shares."""
    from collections import Counter
    n = max(1, len(puzzles))
    stats["kind_table"] = kind_table(puzzles, real)
    stats["size_bands"] = dict(Counter(size_band(p["size"]) for p in puzzles))
    stats["depth"] = dict(Counter(p["depth"] for p in puzzles))
    stats["moves"] = dict(Counter(r for p in puzzles for r in p["transforms"]).most_common())
    stats["valley_share"] = round(sum(1 for p in puzzles if p["valleys"]) / n, 3)
    toks = [p["tokens"] for p in puzzles if p.get("tokens")]
    if toks:
        toks.sort()
        stats["tokens"] = {"counted": len(toks), "le_4096": sum(1 for x in toks if x <= 4096),
                           "fit_share": round(sum(1 for x in toks if x <= 4096) / len(toks), 3),
                           "p50": toks[len(toks) // 2], "p90": toks[int(len(toks) * 0.9)], "max": toks[-1],
                           "tokenizer": args.tokenizer}
    if any("window" in p for p in puzzles):
        ws = sorted(p["window_share"] for p in puzzles if "window" in p)
        stats["window"] = {"records": len(ws), "patch_verified": sum(1 for p in puzzles if p.get("patch_verified")),
                           "regions": dict(Counter(p.get("regions", 0) for p in puzzles)),
                           "window_share_p50": ws[len(ws) // 2] if ws else None}


def write_card(out: Path, puzzles: list[dict], stats: dict, args) -> None:
    n = len(puzzles)
    val = sum(1 for p in puzzles if p["valleys"])
    lines = [f"# stacked (dk dataset stacked, seed {args.seed}, depth {args.depth}, sizes {args.min_size:#x}-{args.max_size:#x}"
             f"{', window render + region patches' if args.window else ''})", "",
             f"{n} puzzles from {len({p['function'] for p in puzzles})} functions; Game-derived: private.",
             f"answers verified exact: {sum(p['answer_verified'] for p in puzzles)}/{n}",
             f"puzzles with a valley (a forward step with no score gain): {val} ({100 * val / max(n, 1):.1f} %)",
             f"depth: " + ", ".join(f"{k}: {v}" for k, v in sorted(stats.get("depth", {}).items())),
             f"size bands: " + ", ".join(f"{k} {v}" for k, v in sorted(stats.get("size_bands", {}).items())),
             "moves: " + ", ".join(f"{k} {v}" for k, v in stats.get("moves", {}).items()),
             f"leaks dropped: {stats.get('leaks', {})}",
             f"compiles {stats.get('compiles', 0)}, wall {stats.get('wall', 0)} s", ""]
    if stats.get("window"):
        w = stats["window"]
        lines += [f"region patches verified: {w['patch_verified']}/{w['records']}; regions per puzzle: {w['regions']}; "
                  f"window shows a median {w['window_share_p50']:.0%} of the definition's lines", ""]
    if stats.get("tokens"):
        tk = stats["tokens"]
        lines += [f"tokens (real tokenizer, SFT window render, 0 examples): <= 4096 for {tk['le_4096']}/{tk['counted']} "
                  f"({tk['fit_share']:.1%}); p50 {tk['p50']}, p90 {tk['p90']}, max {tk['max']}", ""]
    if stats.get("kind_table"):
        lines += ["| fix kind | puzzles | share | real fixes | real share |", "|---|---|---|---|---|"]
        lines += [f"| {k} | {v['puzzles']} | {v['puzzle_share']:.1%} | {v['real']} | {v['real_share']:.1%} |"
                  for k, v in stats["kind_table"].items()]
        lines.append("")
    lines.append("Tool output only (rewrite-table moves, compiler scores, difflib hunks); no model-written text.")
    (out / "CARD.md").write_text("\n".join(lines) + "\n")
    sums = [f"{hashlib.sha256((out / n).read_bytes()).hexdigest()}  {n}" for n in ("stacked.jsonl", "stats.json", "CARD.md")]
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")


def run(cfg: Config, args: argparse.Namespace) -> int:
    from .llmtry import LLM_DIR, data_dir
    t0 = time.time()
    until = 0.0
    if args.until:
        hh, mm = map(int, args.until.split(":"))
        lt = time.localtime()
        until = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, hh, mm, 0, 0, 0, -1))
        until += 86400 if until <= t0 else 0
    out = Path(args.out).expanduser() if args.out else data_dir(cfg) / "datasets" / "stacked" / "v0"
    out.mkdir(parents=True, exist_ok=True)
    ss.HEAD["rev"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cfg.root, capture_output=True, text=True).stdout.strip()
    try:
        depths = depth_weights(args.depth)
    except ValueError as exc:
        print(f"error: {exc}")
        return USAGE
    if args.tokenizer and not Path(args.tokenizer).expanduser().is_file():
        print(f"error: --tokenizer {args.tokenizer}: no such file")
        return USAGE
    got = ss.eligible(cfg, args)
    if got is None:
        return USAGE
    fns, fz, _entries = got
    extra = never_functions(cfg)
    fns = [f for f in fns if f["function"] not in extra]
    print(f"{len(fns)} exact functions after the leakage filter; target {args.n} puzzles", flush=True)
    cache_p = out / "cache.jsonl"
    cache: dict[str, dict] = {}
    if cache_p.is_file():
        for line in cache_p.read_text().splitlines():
            try:
                x = json.loads(line)
                cache[x["key"]] = x["res"]
            except (ValueError, KeyError):
                pass
    table = ss.table_version()
    srcs: dict[str, str] = {}

    def fkey(f):
        p = f["u"]["metadata"]["source_path"]
        if p not in srcs:
            srcs[p] = ss.sha((cfg.root / p).read_text(errors="replace"))
        return ss.sha("stacked", str(SCHEMA), table, str(args.seed), str(args.per_fn), args.depth, str(bool(args.window)),
                      str(args.fold_draw), f["unit"], f["function"], srcs[p])
    work = cfg.root / LLM_DIR / "work" / f"stacked{int(t0)}"
    _POOL.update(cfg=cfg, fns=fns, seed=args.seed, per_fn=args.per_fn, work=work, depths=depths, window=bool(args.window),
                 fold_draw=args.fold_draw)
    puzzles: list[dict] = []
    have: set[str] = set()
    stats: dict = {"functions": 0, "cached": 0, "leaks": {}}

    def take(res: dict) -> None:
        stats["functions"] += 1
        for k, v in res["stats"].items():
            stats[k] = stats.get(k, 0) + v
        for p in res["puzzles"]:
            why = leak(p, fz, extra)
            if why:
                stats["leaks"][why] = stats["leaks"].get(why, 0) + 1
            elif p["id"] not in have and p["answer_verified"]:
                have.add(p["id"])
                puzzles.append(p)
    todo = []
    for i, f in enumerate(fns):
        if len(puzzles) >= args.n:
            break
        k = fkey(f)
        if k in cache:
            stats["cached"] += 1
            take(cache[k])
        else:
            todo.append(i)
    with cache_p.open("a") as fh, multiprocessing.get_context("fork").Pool(args.jobs) as pool:
        for i, res in zip(todo, pool.imap(_work, todo, chunksize=1)):
            fh.write(json.dumps({"key": fkey(fns[i]), "res": res}) + "\n")
            fh.flush()
            take(res)
            if stats["functions"] % 50 == 0:
                print(f"  {stats['functions']} functions, {len(puzzles)} puzzles, {stats.get('compiles', 0)} compiles, "
                      f"{time.time() - t0:.0f} s", flush=True)
            if len(puzzles) >= args.n or (until and time.time() >= until):
                pool.terminate()
                break
    shutil.rmtree(work, ignore_errors=True)
    puzzles = puzzles[:args.n]
    if args.window and args.tokenizer and puzzles:
        renders = [render_window(p) for p in puzzles]
        for p, r, k in zip(puzzles, renders, count_tokens(renders, str(Path(args.tokenizer).expanduser()), args.tokenizer_python)):
            p["tokens"], p["render_chars"] = k, sum(len(m["content"]) for m in r)
    summarize(puzzles, stats, real_fixes(cfg), args)
    stats["wall"] = round(time.time() - t0)
    (out / "stacked.jsonl").write_text("".join(json.dumps(p) + "\n" for p in puzzles))
    (out / "stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    write_card(out, puzzles, stats, args)
    val = sum(1 for p in puzzles if p["valleys"])
    print(f"stacked: {len(puzzles)} puzzles from {stats['functions']} functions ({stats['cached']} cached), "
          f"{val} with a valley; {stats.get('compiles', 0)} compiles; wall {stats['wall']} s -> {out}")
    return OK


def configure(sub) -> None:
    s = sub.add_parser("stacked", help="multi-change puzzles: 2-3 stacked inverse table moves (with the loop rows) on "
                       "exact functions, every state compiled, the score path recorded -> llm-data/datasets/stacked/v0")
    s.add_argument("--n", type=int, default=2000, help="puzzles to make (default 2000)")
    s.add_argument("--per-fn", type=int, default=2, help="puzzles per function at most (default 2)")
    s.add_argument("--depth", default="2-3", help="stack depth range A-B, drawn with weights falling by depth (default 2-3)")
    s.add_argument("--fold-draw", type=float, default=0.0, help="share of the draws whose first step folds a local away, so "
                   "the fix adds one (shape-synth's T38 draw; real fixes add a local in 8/27; default 0)")
    s.add_argument("--window", action="store_true", help="giants: add the region patch (SEARCH/REPLACE hunks, verified) "
                   "and the window render of each puzzle; puzzles with no verified patch are dropped")
    s.add_argument("--tokenizer", default="", help="with --window: a tokenizer.json (Qwen) to count each render's tokens")
    s.add_argument("--tokenizer-python", default="python3", help="a Python with the `tokenizers` package (default python3)")
    s.add_argument("--seed", type=int, default=20261108)
    s.add_argument("--only", default="", help="units whose name contains one of these (comma-separated)")
    s.add_argument("--min-size", type=lambda x: int(x, 0), default=0x20, help="smallest function (default 0x20)")
    s.add_argument("--max-size", type=lambda x: int(x, 0), default=0x800, help="largest function (default 0x800)")
    s.add_argument("--jobs", type=int, default=4, help="parallel functions (default 4)")
    s.add_argument("--until", default="", help="stop starting new functions at HH:MM")
    s.add_argument("--out", default="", help="output directory (default llm-data/datasets/stacked/v0)")
