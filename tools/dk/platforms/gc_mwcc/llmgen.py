"""dk dataset base|attempts|mine|synth|rename|backtrans: more training records, stored structured (llmdata).
  base       every exact function as llm-try's prompt for an unwritten one (asm, declarations, m2c draft) -> source
  attempts   GPU: llm-bench's held-out loop over solved functions (never benchmark or held-out ones)
  mine LOG…  compiled attempts (attempts runs, grind logs) on functions exact today -> repair records
  synth      rule-based natural mutants from close's rewrite table that compile and change the asm -> repairs
  rename     a capped share of the exact functions with locals renamed (asm verified identical) -> direct pairs
  backtrans  GPU: the model writes new functions over a unit's real declarations; the compiler labels them
Why: ~20k records, mostly repair work in inference format, with a known exact answer for every one.
Every record leaves out benchmark and held-out units, functions and asm shapes; mutants and new
functions are checked by their own asm shape too.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ...core import cbody
from ...core.command import OK, USAGE
from ...core.config import Config

GENERATED = {"regs", "synth", "rename", "attempts", "backtrans", "lessons"}   # wrong side or answer made by a tool or model
REAL = {"base", "history", "grind"}
# close's rewrite rows that make natural code (the cast rows add devices; decl-order is `dataset regs`)
SYNTH_RULES = ("statement-order", "operand-swap", "compare-flip", "ternary-if", "if-invert", "or-return",
               "one-case-switch", "chained-assign", "op-assign", "reassociate", "half", "flag-form",
               "assign-in-condition", "split-init", "ref-element", "loop-form")


def hkey(*parts: str) -> int:
    return int(hashlib.sha1("\0".join(parts).encode()).hexdigest()[:8], 16)


# ---- asm of a compiled object (dtk elf disasm: the target listing's format) -------------------------

def object_asm(cfg: Config, obj: Path, keep: Path | None = None) -> dict:
    """{symbol: AsmEntry} of an object, as dtk lists the target (so asm_shape compares with it); `keep`:
    where to copy dtk's listing (m2c reads it)."""
    from . import asmindex
    tmp = Path(tempfile.mkdtemp(prefix="dk-dis-"))
    try:
        out = tmp / "o.s"
        subprocess.run([str(cfg.root / "build" / "tools" / "dtk"), "elf", "disasm", str(obj), str(out)],
                       capture_output=True, text=True)
        if not out.is_file():
            return {}
        if keep is not None:
            shutil.copy(out, keep)
        entries = asmindex.index([out])
        names: dict[str, str] = {}                # quoted local symbols ("@123") named like the target's lbl_
        for e in entries.values():
            e.text = re.sub(r'"[^"\s]+"', lambda m: names.setdefault(m[0], f"lbl_{0x80400000 + 8 * len(names):08X}"), e.text)
        return entries
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---- history recovery: an old version's own top-level declarations ---------------------------------

UNDEF = re.compile(r"undefined identifier '([A-Za-z_]\w*)'")


def defines(chunk: str, name: str) -> bool:
    """Does this top-level chunk declare `name` (a struct/class/enum/typedef, an extern, a prototype)?"""
    n = re.escape(name)
    if re.search(rf"\b(?:struct|class|union|enum)\s+{n}\b", chunk):
        return True
    if re.search(rf"\btypedef\b[^;]*\b{n}\s*(?:\[[^\]]*\]\s*)?;", chunk):
        return True
    s = chunk.strip()
    return s.endswith(";") and "{" not in s and re.search(rf"(?<![\w.>:]){n}\s*(?:\(|\[|;|=)", s) is not None


def old_declarations(old_src: str, names: set[str], have: str) -> list[str]:
    """The old source's top-level chunks that declare any of `names` and are not in `have` already."""
    from .llmtry import _chunks, declarations
    text = declarations(old_src, (0, 0), set(), budget=10 ** 9)
    out = []
    for c in _chunks(text):
        s = c.strip()
        if not s or s.startswith("#") or s in have:
            continue
        if any(defines(c, n) for n in names):
            out.append(s)
    return out


_HEADERS: dict[tuple[str, str], str] = {}
_HLOCK = threading.Lock()


def old_headers(cfg: Config, commit: str, src: str, depth: int = 2) -> str:
    """The quoted includes of an old source as they were at its commit (2 deep), concatenated."""
    out, todo, seen = [], [(src, 0)], set()
    while todo:
        text, d = todo.pop()
        for inc in re.findall(r'^\s*#\s*include\s+"([^"]+)"', text, re.M):
            if inc in seen:
                continue
            seen.add(inc)
            with _HLOCK:
                hit = _HEADERS.get((commit, inc))
            if hit is None:
                hit = ""
                for base in ("include", "src"):
                    proc = subprocess.run(["git", "show", f"{commit}:{base}/{inc}"], cwd=cfg.root, capture_output=True,
                                          text=True, errors="replace")
                    if proc.returncode == 0:
                        hit = proc.stdout
                        break
                with _HLOCK:
                    _HEADERS[(commit, inc)] = hit
            if hit:
                out.append(hit)
                if d + 1 < depth:
                    todo.append((hit, d + 1))
    return "\n".join(out)


def with_declarations(text: str, span_start: int, chunks: list[str]) -> str:
    return text[:span_start] + "".join(c + "\n\n" for c in chunks) + text[span_start:]


# ---- history recovery: old names mapped forward (the symbol at the same address today) ---------------

_SYMS: dict[str, dict[str, int]] = {}
_SLOCK = threading.Lock()
_SYMLINE = re.compile(r"^(\S+)\s*=\s*[^:\s]+:0x([0-9A-Fa-f]+);")


def symbol_names(text: str) -> dict[str, int]:
    """symbols.txt text -> {name: address}."""
    out = {}
    for line in text.splitlines():
        m = _SYMLINE.match(line.strip())
        if m:
            out[m[1]] = int(m[2], 16)
    return out


def symbols_at(cfg: Config, commit: str) -> dict[str, int]:
    """{name: address} of symbols.txt at a commit ("" = today's file), cached."""
    with _SLOCK:
        hit = _SYMS.get(commit)
    if hit is None:
        rel = cfg.platform_options["symbols"]
        text = (cfg.root / rel).read_text() if not commit else subprocess.run(
            ["git", "show", f"{commit}:{rel}"], cwd=cfg.root, capture_output=True, text=True, errors="replace").stdout
        hit = symbol_names(text)
        with _SLOCK:
            _SYMS[commit] = hit
    return hit


def split_mangled(name: str) -> tuple[str, str] | None:
    """(source-level base name, the mangled rest) of a symbol; None for operators, ctors and dtors."""
    if name.startswith("__"):
        return None
    i = name.find("__", 1)
    return (name, "") if i < 0 else (name[:i], name[i:])


def signature_part(rest: str) -> str:
    """A mangled rest without its class qualifier (`__Q22Ns9NsVehicleFi` -> `Fi`, `__3FooCFv` -> `CFv`)."""
    m = re.match(r"__(Q(\d)|(\d+))", rest)
    if not m:
        return rest[2:] if rest.startswith("__") else rest
    i, parts = m.end(1), int(m[2]) if m[2] else 1
    if m[2] is None:
        i = m.start(3)
    for _ in range(parts):
        n = re.match(r"\d+", rest[i:])
        if not n:
            return rest
        i += n.end() + int(n[0])
    return rest[i:]


def forward_names(old: dict[str, int], today: dict[str, int], missing: set[str]) -> dict[str, str]:
    """Old identifiers -> today's: each missing name's old symbols (same base name), followed by address to
    today's symbol there; mapped only when every one agrees and keeps its parameters (a member may have
    moved to a base or renamed class: `obj->name(…)` is still spelled the same way with the new name;
    a free function stays free)."""
    by_addr = {a: n for n, a in today.items()}
    bases: dict[str, list[tuple[str, int]]] = {}
    for n, a in old.items():
        sp = split_mangled(n)
        if sp and sp[0] in missing:
            bases.setdefault(sp[0], []).append((sp[1], a))
    out = {}
    for x, syms in bases.items():
        got = set()
        for rest, a in syms:
            now = by_addr.get(a)
            sp = split_mangled(now) if now else None
            same = sp is not None and (sp[1] == rest or bool(rest) and bool(sp[1]) and signature_part(sp[1]) == signature_part(rest)
                                           and sp[1].startswith("__F") == rest.startswith("__F"))
            got.add(sp[0] if same and sp[0] != x else None)
        if len(got) == 1 and None not in got:
            out[x] = got.pop()
    return out


def rename_all(chunk: str, mapping: dict[str, str]) -> str:
    """Every identifier token in `mapping` replaced (members and qualified names too)."""
    edits = [(t.start, t.end, mapping[t.text]) for t in cbody.tokens(chunk) if t.kind == "id" and t.text in mapping]
    return cbody.apply_edits(chunk, edits)


# ---- mining attempt logs --------------------------------------------------------------------------

def class_from_rows(rows: list[str]) -> str:
    """"register" when every diff row differs only in register numbers (llm-try's classify, from rows)."""
    reg = re.compile(r"\b[rf]\d+\b|\bcr\d\b")
    marked = 0
    for r in rows:
        m = re.match(r"^\s*[0-9A-F]*\s+(.*?)\s+(op|arg|rep|-|\+)\s+(.*)$", r)
        if not m:
            continue
        marked += 1
        if m[2] != "arg" or reg.sub("R", m[1].strip()) != reg.sub("R", m[3].strip()):
            return "shape"
    return "register" if marked else "shape"


def pick_spread(items: list[dict], cap: int) -> list[dict]:
    """At most `cap` attempts of one function, distinct code, spread over the score range (best kept)."""
    from .llmdata import norm
    seen, uniq = set(), []
    for a in sorted(items, key=lambda a: (-a["percent"], a.get("round", 0), a.get("k", 0))):
        n = norm(a["variant"])
        if n not in seen:
            seen.add(n)
            uniq.append(a)
    if len(uniq) <= cap:
        return uniq
    step = (len(uniq) - 1) / (cap - 1) if cap > 1 else 0
    return [uniq[round(i * step)] for i in range(cap)]


def mine(cfg: Config, args) -> int:
    """Attempt logs -> repair records for functions exact today (their source is the answer)."""
    from .llmdata import candidates, compiler_of, out_dir, record, write_records
    from .llmtry import prepare
    from .trycmd import variant_text
    root = out_dir(cfg, args)
    fns = {f["function"]: f for f in candidates(cfg, args, False)}
    per: dict[tuple, list[dict]] = {}
    seen_logs = 0
    for path in args.logs:
        p = Path(path).expanduser()
        if not p.is_file():
            print(f"error: no log {path}")
            return USAGE
        seen_logs += 1
        for line in p.read_text(errors="replace").splitlines():
            if '"kind": "attempt"' not in line:
                continue
            a = json.loads(line)
            if not a.get("compiled") or not a.get("variant") or a.get("percent", 0.0) >= 100.0:
                continue
            f = fns.get(a.get("function"))
            if f is None or f["unit"] != a.get("unit"):
                continue                               # not exact today, or never-train (candidates leaves those out)
            src_kind = "grind" if "grind" in a else "attempts"
            per.setdefault((src_kind, a["function"]), []).append({**a, "log": p.name})
    comp_memo: dict = {}
    out: dict[str, list[dict]] = {"attempts": [], "grind": []}
    neg = 0
    lock = threading.Lock()

    def one(item):
        (src_kind, fn), items = item
        f = fns[fn]
        try:
            u, _p, src, span, locate, ctx = prepare(cfg, f["unit"], fn, m2c=False)
        except Exception:
            return []
        if not ctx.current:
            return []
        clean = [a for a in items if not a.get("devices")]
        devs = [a for a in items if a.get("devices")]
        got = []
        for a in pick_spread(clean, args.per_fn) + pick_spread(devs, args.negatives):
            text = variant_text(src, fn, a["variant"], span, locate)
            res = {"percent": a["percent"], "rows": a.get("rows", []), "rows_fixed": a.get("rows_fixed", 0),
                   "diff_class": class_from_rows(a.get("rows", []))}
            prov = {"log": a["log"], "model": a.get("model", ""), "round": a.get("round", 0), "k": a.get("k", 0),
                    "seed": a.get("seed"), "time": a.get("time", 0), "hint": a.get("hint")}
            try:
                r = record(cfg, f, src_kind, a["variant"].rstrip("\n"), text, res, ctx.current, prov,
                           compiler_of(cfg, f["u"], comp_memo))
            except Exception:
                continue
            r["first_round"] = False                  # the model's own best attempt so far, as round two shows it
            if a.get("devices"):
                r["devices"] = a["devices"]           # DPO negative only, never an SFT input
            got.append(r)
        return got
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for recs in pool.map(one, sorted(per.items())):
            for r in recs:
                with lock:
                    out[r["source"]].append(r)
                    neg += bool(r.get("devices"))
    for name, recs in out.items():
        if recs or args.replace:
            path = root / "records" / f"{name}.jsonl"
            if not args.replace and path.is_file():          # merge with earlier nights (same attempt: replaced)
                from .llmdata import read_jsonl
                old = {(r["function"], r["wrong"]): r for r in read_jsonl(path)}
                old.update({(r["function"], r["wrong"]): r for r in recs})
                recs = list(old.values())
            write_records(path, recs)
            print(f"{name}: {len(recs)} records in {cfg.rel(path)}")
    print(f"mined {seen_logs} logs: {len(per)} (source, function) groups exact today; "
          f"{sum(len(v) for v in out.values())} new records ({neg} device attempts kept as DPO negatives)")
    return OK


# ---- the GPU generator: llm-bench's held-out loop over solved functions -------------------------------

def flagged(cfg: Config) -> set[str]:
    """The judge audit's flagged functions (llm-data/judge-audit.json; empty without it): never a GPU job's
    answer or style example, since the export quarantines them anyway."""
    from .llmdata import quarantined
    from .llmtry import data_dir
    return quarantined(data_dir(cfg) / "judge-audit.json") or set()


def solved_pick(cfg: Config, args) -> list[dict]:
    """Exact functions (never benchmark, held-out or vendored; leakage-filtered by unit and asm shape), the
    same number from each size band (the open work is mostly big functions), each band spread over areas,
    bands interleaved so a night stopped early still has all sizes: the held-out set's file format."""
    from .llmdata import BANDS, band, candidates
    from .llmtry import _pick, area
    by: dict[str, list[dict]] = {}
    q = flagged(cfg)
    for f in candidates(cfg, args, False):
        if args.min_size <= f["size"] <= args.max_size and f["function"] not in q:
            u = f["u"]
            by.setdefault(band(f["size"]), []).append(
                {"unit": f["unit"], "function": f["function"], "size": f["size"], "kind": "solved", "asm_shape": f["asm_shape"],
                 "source": u["metadata"]["source_path"], "area": area(f["unit"])})
    want = args.pick or sum(len(v) for v in by.values())
    lanes = [_pick(by[b], len(by[b])) for _t, b in BANDS if by.get(b)]
    out = []
    while len(out) < want and any(lanes):
        for lane in lanes:
            if lane and len(out) < want:
                out.append(lane.pop(0))
    return out


def attempts(cfg: Config, args) -> int:
    from .llmdata import out_dir
    from .llmtry import _configure_bench, data_dir, run_bench
    root = out_dir(cfg, args)
    pick = root / "solved.json"
    if not pick.is_file() or args.repick:
        items = solved_pick(cfg, args)
        doc = {"never_train": False, "note": "Solved functions for dk dataset attempts (exact, not benchmark or "
               "held-out): the model writes each from its asm; dk dataset mine turns the attempts into repair records.",
               "picked": time.strftime("%Y-%m-%d %H:%M"), "functions": items}
        pick.parent.mkdir(parents=True, exist_ok=True)
        pick.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"picked {len(items)} solved functions into {cfg.rel(pick)}")
    if args.pick_only:
        return OK
    p = argparse.ArgumentParser(prog="dk llm-bench")
    _configure_bench(p)
    bargs = p.parse_args(["run", "--set", "heldout", "--bench", str(pick)])
    for k, v in vars(args).items():                 # the model options, --until, --resume, --run, --limit
        if hasattr(bargs, k) and k not in ("action", "bench", "set", "only", "max_size"):
            setattr(bargs, k, v)
    bargs.run = bargs.run or f"attempts-{time.strftime('%Y%m%d')}"
    rc = run_bench(cfg, bargs)
    print(f"then: ./dk dataset mine {data_dir(cfg) / (bargs.run + '.jsonl')}")
    return rc


# ---- the repair eval set (held-out functions: never trained on) -------------------------------------

def repair_eval(cfg: Config, args) -> int:
    """Held-out logs -> llm-data/repair-eval.json: the model's compiling, judge-clean, non-exact attempts on
    held-out functions (args.min <= % < args.max, at most per_fn each, spread over the score range), each
    with its prompt context and the known exact answer. An eval of the grind's real job (repair)."""
    from .judge import REJECT, forbidden_rules, review
    from .llmdata import DEVICE_FLAGS
    from .llmtry import data_dir, prepare
    from .trycmd import definition, variant_text
    held = {r["function"]: r for r in json.loads((data_dir(cfg) / "heldout.json").read_text())["functions"]}
    forbidden = forbidden_rules(cfg)
    per: dict[str, list[dict]] = {}
    for path in args.logs:
        p = Path(path).expanduser()
        for line in p.read_text(errors="replace").splitlines():
            if '"kind": "attempt"' not in line:
                continue
            a = json.loads(line)
            if a.get("function") in held and a.get("compiled") and a.get("variant") and not a.get("devices") \
                    and args.min <= a.get("percent", 0.0) < args.max:
                per.setdefault(a["function"], []).append({**a, "log": p.name})
    frozen: dict[tuple, str] = {}
    out, judged_out = [], 0
    for fn, items in sorted(per.items()):
        h = held[fn]
        k = (h["source_commit"], h["source"])
        if k not in frozen:
            frozen[k] = subprocess.run(["git", "show", f"{k[0]}:{k[1]}"], cwd=cfg.root, capture_output=True, text=True,
                                       errors="replace").stdout
        src = frozen[k]
        span = definition(src, fn)
        if span is None:
            continue
        exact = src[span[0]:span[1]]
        stub = type_stub(src)
        clean = []
        for a in items:
            cat, flags = review(stub + exact + "\n", stub + a["variant"].strip() + "\n", fn, forbidden)   # vs the answer
            if cat == REJECT or any(w in f for f in flags for w in DEVICE_FLAGS):
                judged_out += 1
                continue
            clean.append(a)
        for a in pick_spread(clean, args.per_fn):
            text = variant_text(src, fn, a["variant"].rstrip("\n"), span, None)
            try:
                _u, _p, _s, _sp, _loc, ctx = prepare(cfg, h["unit"], fn, src_text=text, m2c=False)
            except Exception:
                continue
            out.append({"unit": h["unit"], "function": fn, "size": h["size"], "asm_shape": h.get("asm_shape"), "kind": "repair",
                        "area": h.get("area"), "source": h["source"], "source_commit": h["source_commit"],
                        "start_code": a["variant"].rstrip("\n"), "start_percent": a["percent"], "start_rows": a.get("rows", []),
                        "diff_class": class_from_rows(a.get("rows", [])), "exact": exact, "asm": ctx.asm, "decls": ctx.decls,
                        "provenance": {"log": a["log"], "model": a.get("model"), "round": a.get("round"), "k": a.get("k"),
                                       "seed": a.get("seed")}})
    dest = Path(args.dest) if args.dest else data_dir(cfg) / "repair-eval.json"
    doc = {"never_train": True, "note": "Repair eval: the model's compiling, judge-clean, non-exact attempts on held-out "
           "functions with the exact answer known. Held-out rules apply: never put any of it in a training set.",
           "made": time.strftime("%Y-%m-%d %H:%M"), "logs": [Path(x).name for x in args.logs],
           "range": [args.min, args.max], "per_function": args.per_fn, "functions": out}
    dest.write_text(json.dumps(doc, indent=1) + "\n")
    bands: dict[str, int] = {}
    from .llmdata import band
    for r in out:
        bands[band(r["size"])] = bands.get(band(r["size"]), 0) + 1
    print(f"repair eval: {len(out)} attempts on {len({r['function'] for r in out})} held-out functions into {dest} "
          f"({judged_out} judged out; register-only {sum(1 for r in out if r['diff_class'] == 'register')}; sizes {bands})")
    return OK


# ---- synthetic mutants -----------------------------------------------------------------------------

def loop_form(site):
    """`for (init; cond; step) body` -> `init; while (cond) { body step; }` (no `continue` inside)."""
    for s in cbody.statements(site.body):
        if s.kind != "for" or not s.kids or s.cond is None or not site.in_block(s):
            continue
        body = s.kids[0]
        if any(x.kind == "continue" for x in cbody.statements(body)):
            continue
        init = site.src[s.init.start:s.init.end].strip() if s.init is not None and s.init.kind != "empty" else ""
        step = site.src[s.step.start:s.step.end] if s.step is not None else ""
        ind = cbody.indent_at(site.src, s.start)
        inner = site.src[body.start + 1:body.end - 1].rstrip() if body.kind == "block" else \
            "\n" + ind + "    " + site.src[body.start:body.end]
        stepl = f"\n{ind}    {step};" if step else ""
        new = (f"{init}\n{ind}" if init else "") + f"while ({site.src[s.cond.start:s.cond.end]}) {{{inner}{stepl}\n{ind}}}"
        yield f"for -> while at {site.line(s.start) + 1}", [(s.start, s.end, new)]


def mutants(src: str, symbol: str, span: tuple[int, int], cplusplus: bool, per_fn: int, seed: str) -> list[tuple[str, str, str]]:
    """(rule, label, mutated definition) for one exact definition: one per rule (round robin), at most
    per_fn, the order of rules shuffled per function. No `__typeof__` locals, no new functions."""
    from . import rewrites
    try:
        brace = src.index("{", span[0], span[1])
        body = cbody.parse_body(src, brace, cbody.type_names(src))
    except (ValueError, cbody.ParseError):
        return []
    site = rewrites.Site(src, body, None, cplusplus)
    table = [r for r in rewrites.RULES if r.name in SYNTH_RULES]
    table.append(rewrites.Rule("loop-form", "", "", (), loop_form))
    table.sort(key=lambda r: hkey(seed, r.name))
    by_rule: dict[str, list[tuple[str, str]]] = {}
    for rule in table:
        try:
            got = list(rewrites.candidates(site, [rule], groups=False))
        except Exception:
            continue
        for name, label, new in got[:6]:
            d = len(new) - len(src)
            chunk = new[span[0]:span[1] + d]
            if "__typeof__" in chunk or not chunk.rstrip().endswith("}"):
                continue
            by_rule.setdefault(name, []).append((label, chunk))
    out, i = [], 0
    while len(out) < per_fn and any(by_rule.values()):
        for name in [r.name for r in table]:
            xs = by_rule.get(name)
            if xs and len(out) < per_fn:
                label, chunk = xs.pop(hkey(seed, name, str(i)) % len(xs))
                out.append((name, label, chunk))
        i += 1
    return out


_FORK: dict = {}


def type_stub(src: str) -> str:
    """Declarations of a source's type names: enough for the judge to parse one definition alone."""
    return "".join(f"struct {n};\n" for n in sorted(cbody.type_names(src)) if re.fullmatch(r"[A-Za-z_]\w*", n))


def _synth_unit(group: list[int]) -> list[tuple]:
    """Fork worker: one unit's functions -> (index, span, asm, decls, [(rule, label, chunk)]) of clean mutants."""
    from .judge import CLEAN, review
    from .llmtry import prepare
    cfg, fns, per_fn, forbidden = _FORK["cfg"], _FORK["fns"], _FORK["per_fn"], _FORK["forbidden"]
    out, stub = [], None
    for i in group:
        f = fns[i]
        try:
            u, _p, src, span, _locate, ctx = prepare(cfg, f["unit"], f["function"], m2c=False)
        except Exception:
            continue
        if not ctx.current:
            continue
        stub = stub if stub is not None else type_stub(src)
        cpp = not u["metadata"]["source_path"].endswith(".c")
        keep = []
        for rule, label, chunk in mutants(src, f["function"], span, cpp, per_fn, f["function"]):
            cat, _flags = review(stub + ctx.current + "\n", stub + chunk + "\n", f["function"], forbidden)
            if cat == CLEAN:
                keep.append((rule, label, chunk))
        if keep:
            out.append((i, span, ctx.asm, ctx.decls, ctx.current, keep))
    return out


RULE_CAPS = {"statement-order": 0.20}    # 2026-11-02 spot check: 36% of synth, often two of a run of calls


def cap_rule_share(recs: list[dict], share: float, caps: dict[str, float] | None = None) -> tuple[list[dict], int]:
    """No rule above `share` of the mutants (statement order applies almost everywhere), and a rule in
    `caps` (default RULE_CAPS) at most its own share: a rule's excess dropped by a hash, its records in
    big functions kept first (the open work is big)."""
    caps = RULE_CAPS if caps is None else caps
    if not recs or (share >= 1 and not caps):
        return recs, 0
    by: dict[str, list[dict]] = {}
    for r in recs:
        by.setdefault((r.get("provenance") or {}).get("rule", ""), []).append(r)
    out = []
    for rule, items in by.items():
        cap = min(share, caps.get(rule, 1.0))
        others = len(recs) - len(items)
        limit = len(items) if cap >= 1 else max(1, int(cap * others / (1 - cap))) if others else len(items)
        items.sort(key=lambda r: (-min(r.get("size", 0), 0x800), hkey(rule, r["function"], r["wrong"])))
        out += items[:limit]
    return out, len(recs) - len(out)


_STR = re.compile(r'"(?:[^"\\\n]|\\.)*"')
_SELF = re.compile(r"(?<![\w.>])([A-Za-z_][\w.\[\]>-]*)\s*=\s*\1\s*;")
_BOOL_ARMS = re.compile(r"\?\s*(?:true|false|1|0)\s*:\s*(?:true|false|1|0)\s*[;)]")


def _flat(t: str) -> str:
    """Parenthesized groups emptied (what is left is the top level)."""
    prev = None
    while prev != t:
        prev, t = t, re.sub(r"\([^()]*\)", "()", t)
    return t


def _in_call_args(text: str, at: int) -> bool:
    """Is text[at] inside a call's argument list (not an if/while/for/switch/return parenthesis)?"""
    depth = 0
    for i in range(at - 1, -1, -1):
        c = text[i]
        if c == ")":
            depth += 1
        elif c == "(":
            if depth:
                depth -= 1
                continue
            m = re.search(r"([\w\])>]+)\s*$", text[:i])
            return bool(m) and m[1] not in ("if", "while", "for", "switch", "return", "sizeof")
        elif c in ";{}":
            return False
    return False


def _call_run(exact: str, callee: str) -> int:
    best = run = 0
    for line in exact.splitlines():
        run = run + 1 if line.strip().startswith(callee + "(") else 0
        best = max(best, run)
    return best


def synth_artifact(rec: dict) -> str:
    """Why an existing synth record's wrong source is not one a person writes (the rule fixes of
    2026-11-02, rewrites.py, checked on the text): "" when it is fine."""
    prov = rec.get("provenance") or {}
    rule, label, wrong, exact = prov.get("rule", ""), prov.get("label", ""), rec.get("wrong", ""), rec.get("exact", "")
    if rule == "operand-swap" and " -> " in label:
        before = label.split(" -> ", 1)[0]
        top = _flat(_STR.sub('"s"', before))
        if '"s"' in top and "+" in top:
            return "operand swap of a string concatenation"
        if re.search(r"[/%][^*]*\*", _flat(before)):
            return "operand swap of integer rounding (x / N * N)"
    if rule == "ternary-if":
        if any(m[0] not in exact for m in _SELF.finditer(wrong)):
            return "ternary-if self-assignment"
        if any(m[0] not in exact for m in _BOOL_ARMS.finditer(wrong)):
            return "ternary-if ? true : false"
    if rule == "assign-in-condition":
        m = re.match(r"\((\w+) = ", label)
        if m:
            at = wrong.find(f"({m[1]} = ")
            if at >= 0 and _in_call_args(wrong, at):
                return "assignment inside a call argument"
    if rule == "statement-order":
        m = re.match(r"swapped ([\w:.>-]+)\(.* / ([\w:.>-]+)\(", label)
        if m and m[1] == m[2] and _call_run(exact, m[1]) >= 3:
            return "swap inside a run of identical calls"
    return ""


def literal_labels(draft: str, code: str) -> str:
    """A back-translated function's m2c draft names its string literals by compiler label (`&@412`);
    drafts of the game's own functions never do. The labels follow the literals' first appearance in the
    code (identical strings pooled), so each becomes its literal; left alone when the counts differ."""
    labels = sorted({int(x) for x in re.findall(r"&@(\d+)\b", draft)})
    if not labels:
        return draft
    lits = list(dict.fromkeys(m[0] for m in _STR.finditer(code)))
    if len(lits) != len(labels):
        return draft
    by = {n: lit for n, lit in zip(labels, lits)}
    return re.sub(r"&@(\d+)\b", lambda m: by[int(m[1])], draft)


def clean_answer(code: str) -> str:
    """Trailing whitespace off every line (back-translated answers had whitespace-only lines)."""
    return "\n".join(line.rstrip() for line in code.strip("\n").splitlines())


def synth(cfg: Config, args) -> int:
    import multiprocessing
    from .judge import forbidden_rules
    from .llmdata import (RECORD_VERSION, Cache, candidates, compiler_of, key, out_dir, scored, write_records)
    from .llmtry import PROMPT_VERSION, Scorer, never_train
    from .matching import resolve_unit
    from .trycmd import variant_text
    t0 = time.time()
    root = out_dir(cfg, args)
    cache = Cache(root / "cache" / "scores.jsonl")
    shapes = never_train(cfg)[2]
    fns = candidates(cfg, args, False)
    fns.sort(key=lambda f: hkey("synth", f["function"]))
    if args.limit:
        fns = fns[:args.limit]
    groups: dict[str, list[int]] = {}
    for i, f in enumerate(fns):
        groups.setdefault(f["unit"], []).append(i)
    _FORK.update(cfg=cfg, fns=fns, per_fn=args.per_fn, forbidden=forbidden_rules(cfg))
    with multiprocessing.get_context("fork").Pool(args.jobs) as pool:
        prepared = [x for xs in pool.imap_unordered(_synth_unit, list(groups.values())) for x in xs]
    srcs: dict[str, str] = {}
    work = []
    for i, span, asm, decls, exact, keep in prepared:
        f = fns[i]
        if f["unit"] not in srcs:
            srcs[f["unit"]] = (cfg.root / f["u"]["metadata"]["source_path"]).read_text()
        for rule, label, chunk in keep:
            work.append((f, span, asm, decls, exact, rule, label, chunk))
    print(f"{len(fns)} exact functions; {len(work)} clean mutants to compile ({time.time() - t0:.0f} s)")
    stats = {"compiled": 0, "same": 0, "failed": 0, "shape": 0, "secs": 0.0}
    lock = threading.Lock()
    units: dict[str, dict] = {}

    def one(i_w):
        i, (f, span, asm, decls, exact, rule, label, chunk) = i_w
        src = srcs[f["unit"]]
        u = units.setdefault(f["unit"], resolve_unit(cfg, f["unit"]))
        scorer = Scorer(cfg, u, f["function"], src, span, None)
        text = variant_text(src, f["function"], chunk, span, None)
        res = scored(cfg, cache, scorer, f["function"], key(src), text, chunk, f"s{i}", shape=True)
        with lock:
            stats["secs"] += 0 if res.get("cached") else res.get("secs", 0)
            if not res["compiled"]:
                stats["failed"] += 1
                return None
            stats["compiled"] += 1
            if res["percent"] >= 100.0:
                stats["same"] += 1
                return None
            if res.get("shape") in shapes:
                stats["shape"] += 1                    # the mutant compiles like a benchmark/held-out function
                return None
        return f, asm, decls, exact, rule, label, chunk, res
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        got = [r for r in pool.map(one, enumerate(work)) if r]
    comp_memo: dict = {}
    recs = []
    for f, asm, decls, exact, rule, label, chunk, res in got:      # the declarations of the exact version: same names
        recs.append({"v": RECORD_VERSION, "source": "synth", "kind": "repair", "unit": f["unit"], "function": f["function"],
                     "size": f["size"], **compiler_of(cfg, f["u"], comp_memo), "asm": asm, "asm_shape": f["asm_shape"],
                     "decls": decls, "wrong": chunk, "wrong_percent": res["percent"], "rows": res["rows"],
                     "rows_fixed": res.get("rows_fixed", 0), "diff_class": res.get("diff_class", ""), "exact": exact,
                     "mutant_shape": res.get("shape", ""),
                     "provenance": {"rule": rule, "label": label[:120]}, "prompt_version": PROMPT_VERSION})
    recs, capped = cap_rule_share(recs, args.max_rule_share)
    write_records(root / "records" / "synth.jsonl", recs)
    by: dict[str, int] = {}
    for r in recs:
        by[r["provenance"]["rule"]] = by.get(r["provenance"]["rule"], 0) + 1
    print(f"synth: {len(recs)} repair records into {cfg.rel(root)}/records/synth.jsonl ({capped} over a rule's "
          f"{args.max_rule_share:.0%} share dropped)")
    print("  by rule: " + ", ".join(f"{k} {v}" for k, v in sorted(by.items(), key=lambda kv: -kv[1])))
    print(f"  compiled {stats['compiled']} of {len(work)} ({stats['same']} same asm, {stats['failed']} failed, "
          f"{stats['shape']} with a never-train asm shape); compile CPU {stats['secs']:.0f} s; wall {time.time() - t0:.0f} s")
    return OK


# ---- rename-only copies ----------------------------------------------------------------------------

ABBREV = {"index": "idx", "count": "num", "result": "ret", "value": "val", "temp": "tmp", "pos": "position",
          "len": "length", "ptr": "p", "obj": "object", "cur": "current", "prev": "previous", "dist": "distance",
          "dir": "direction", "vel": "velocity", "num": "count", "idx": "index", "ret": "result", "val": "value",
          "tmp": "temp", "src": "source", "dst": "dest", "cnt": "n", "n": "cnt", "car": "vehicle",
          "vehicle": "car", "time": "t", "t": "time", "speed": "spd", "angle": "ang", "player": "plr"}
LETTERS = ("ijkmn", "pqrs", "xyzw", "abcdef", "uv", "fgh")


def new_name(name: str, taken: set[str], salt: str) -> str | None:
    """Another natural name for a local: a sibling letter, an abbreviation (both ways), camelCase <->
    snake_case; None when none is free."""
    cands = []
    if len(name) == 1:
        for group in LETTERS:
            if name in group:
                cands += [c for c in group if c != name]
    low = name[:1].lower() + name[1:]
    if low in ABBREV:
        cands.append(ABBREV[low])
    m = re.match(r"^([a-z]+)([A-Z]\w*)$", name)
    if m:
        cands.append(re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower())
        if m[1] in ABBREV:
            cands.append(ABBREV[m[1]] + m[2])
    if "_" in name.strip("_"):
        parts = name.strip("_").split("_")
        cands.append(parts[0] + "".join(p.title() for p in parts[1:]))
    cands = [c for c in cands if c and c not in taken and c != name and re.fullmatch(r"[A-Za-z_]\w*", c)]
    if not cands:
        return None
    return cands[hkey(salt, name) % len(cands)]


def renamed(chunk: str, mapping: dict[str, str]) -> str:
    """The definition with its locals renamed: identifier tokens only, never after `.`, `->` or `::`."""
    toks = cbody.tokens(chunk)
    edits = []
    for i, t in enumerate(toks):
        if t.kind == "id" and t.text in mapping and not (i and toks[i - 1].text in (".", "->", "::")) \
                and not (i + 1 < len(toks) and toks[i + 1].text == "::"):
            edits.append((t.start, t.end, mapping[t.text]))
    return cbody.apply_edits(chunk, edits)


def rename_cmd(cfg: Config, args) -> int:
    from .llmdata import Cache, candidates, compiler_of, key, local_names, out_dir, scored, write_records
    from .llmtry import Scorer, identifiers, prepare, visible_text
    from .trycmd import variant_text
    t0 = time.time()
    root = out_dir(cfg, args)
    cache = Cache(root / "cache" / "scores.jsonl")
    fns = [f for f in candidates(cfg, args, False) if hkey("rename", f["function"]) % 1000 < args.share * 1000]
    work = []
    for f in fns:
        try:
            u, source, src, span, locate, ctx = prepare(cfg, f["unit"], f["function"], m2c=False)
        except Exception:
            continue
        names = local_names(ctx.current) if ctx.current else []
        if not names:
            continue
        taken = identifiers(src) | identifiers(visible_text(cfg, src, source))
        mapping = {}
        for n in names:
            nn = new_name(n, taken | set(mapping.values()), f["function"])
            if nn:
                mapping[n] = nn
        if not mapping:
            continue
        chunk = renamed(ctx.current, mapping)
        if chunk != ctx.current:
            work.append((f, u, src, span, locate, chunk, mapping))
    print(f"{len(fns)} functions in the {args.share:.0%} share; {len(work)} renamed copies to verify ({time.time() - t0:.0f} s)")
    lock = threading.Lock()
    stats = {"same": 0, "differs": 0, "failed": 0}

    def one(i_w):
        i, (f, u, src, span, locate, chunk, mapping) = i_w
        scorer = Scorer(cfg, u, f["function"], src, span, locate)
        text = variant_text(src, f["function"], chunk, span, locate)
        res = scored(cfg, cache, scorer, f["function"], key(src), text, chunk, f"n{i}")
        with lock:
            if not res["compiled"]:
                stats["failed"] += 1
                return None
            if res["percent"] < 100.0:
                stats["differs"] += 1               # a rename that changes the asm is not rename-only
                return None
            stats["same"] += 1
        return f, chunk, text, mapping
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        got = [r for r in pool.map(one, enumerate(work)) if r]
    comp_memo: dict = {}
    recs = []
    for f, chunk, text, mapping in got:
        try:
            r = direct_record(cfg, f, "rename", chunk, text, {"renamed": mapping}, compiler_of(cfg, f["u"], comp_memo), args.m2c)
        except Exception:
            continue
        recs.append(r)
    write_records(root / "records" / "rename.jsonl", recs)
    print(f"rename: {len(recs)} direct pairs into {cfg.rel(root)}/records/rename.jsonl "
          f"({stats['same']} identical asm, {stats['differs']} changed it, {stats['failed']} failed)")
    return OK


# ---- base pairs, structured (prompt = llm-try's for an unwritten function) ---------------------------

def direct_record(cfg: Config, f: dict, source: str, answer: str, text: str, prov: dict, comp: dict, m2c: bool,
                  views: bool = False) -> dict:
    """asm (+ declarations and m2c draft as the held-out run builds them: the definition blanked) -> answer.
    `views`: the prompt of every llm-try prompt version ({version: m2c draft, declarations, notes}), so
    an export re-renders with any of them (repairs of the function take its notes from here)."""
    from .llmdata import RECORD_VERSION
    from .llmtry import PROMPT_VERSION, prepare
    from . import llmtry
    _u, _p, _s, _span, _loc, ctx = prepare(cfg, f["unit"], f["function"], src_text=text, m2c=m2c, blank=m2c)
    rec = {"v": RECORD_VERSION, "source": source, "kind": "direct", "unit": f["unit"], "function": f["function"],
           "size": f["size"], **comp, "asm": ctx.asm, "asm_shape": f["asm_shape"], "decls": ctx.decls,
           "m2c": ctx.m2c, "notes": ctx.notes, "wrong": "", "wrong_percent": 0.0, "rows": [], "diff_class": "",
           "exact": answer, "provenance": prov, "prompt_version": PROMPT_VERSION}
    if views:
        rec["views"] = {str(PROMPT_VERSION): view_of(ctx)}
        for v in getattr(llmtry, "PROMPT_VERSIONS", ()):
            if v != PROMPT_VERSION and v not in getattr(llmtry, "VIEW_OF", {}):   # v8: v7's view, asm annotated at export
                *_x, c = prepare(cfg, f["unit"], f["function"], src_text=text, m2c=True, blank=True, version=v)
                rec["views"][str(v)] = view_of(c)
    return rec


def view_of(ctx) -> dict:
    """A prompt version's stored view of a function: its m2c draft, declarations and notes (v7: context too)."""
    v = {"m2c": ctx.m2c, "decls": ctx.decls, "notes": ctx.notes}
    if getattr(ctx, "context", ""):
        v["context"] = ctx.context
    return v


class memo_views:
    """While building views: m2c runs once per function (not once per prompt version), and v6's layout
    compile (`-sym on` of the unit's source) once per unit, not once per function."""

    def __init__(self, cfg: Config):
        self.cfg, self.lock, self.m2c, self.objs = cfg, threading.Lock(), {}, {}
        self.dir = Path(tempfile.mkdtemp(prefix="dk-views-"))

    def __enter__(self):
        from . import llmtry, reorder
        self.saved = (llmtry.m2c_draft, reorder.compile_variant)
        m2c, comp = self.saved

        def m2c_memo(cfg, symbol, asm_file, c):
            k = (symbol, str(asm_file), c)
            with self.lock:
                hit = self.m2c.get(k)
            if hit is None:
                hit = m2c(cfg, symbol, asm_file, c)
                with self.lock:
                    self.m2c[k] = hit
            return hit

        def compile_memo(cfg, unit, text, work, extra=()):
            if tuple(extra) != ("-sym", "on"):
                return comp(cfg, unit, text, work, extra)
            k = hashlib.sha1(f"{unit['name']}\0{text}".encode()).hexdigest()
            with self.lock:
                hit = self.objs.get(k)
            if hit is None:
                obj, out = comp(cfg, unit, text, work, extra)
                if obj is not None:
                    keep = self.dir / f"{k}.o"
                    shutil.copy(obj, keep)
                    obj = keep
                hit = (obj, out)
                with self.lock:
                    self.objs[k] = hit
            return hit
        llmtry.m2c_draft, reorder.compile_variant = m2c_memo, compile_memo
        return self

    def __exit__(self, *exc):
        from . import llmtry, reorder
        llmtry.m2c_draft, reorder.compile_variant = self.saved
        shutil.rmtree(self.dir, ignore_errors=True)
        return False


def base(cfg: Config, args) -> int:
    from .llmdata import candidates, compiler_of, out_dir, write_records
    from .llmtry import prepare
    t0 = time.time()
    root = out_dir(cfg, args)
    fns = candidates(cfg, args, False)
    if args.limit:
        fns = fns[:args.limit]
    comp_memo: dict = {}
    lock = threading.Lock()

    def one(f):
        try:
            _u, _p, src, _span, _loc, ctx = prepare(cfg, f["unit"], f["function"], m2c=False)
            if not ctx.current:
                return None
            with lock:
                comp = compiler_of(cfg, f["u"], comp_memo)
            return direct_record(cfg, f, "base", ctx.current, src, {}, comp, args.m2c, views=True)
        except Exception:
            return None
    with memo_views(cfg), ThreadPoolExecutor(max_workers=args.jobs) as pool:
        recs = [r for r in pool.map(one, sorted(fns, key=lambda f: f["unit"])) if r]
    write_records(root / "records" / "base.jsonl", recs)
    print(f"base: {len(recs)} direct pairs into {cfg.rel(root)}/records/base.jsonl "
          f"({sum(1 for r in recs if r['m2c'])} with an m2c draft; {time.time() - t0:.0f} s)")
    return OK


# ---- back-translation --------------------------------------------------------------------------------

BT_SYSTEM = """You write C++98 game code for a GameCube racing game built with Metrowerks CodeWarrior.
Given a source file's declarations and one of its real functions, write ONE new function that the
original programmers could plausibly have written in this file: natural, plain code (5 to 40 lines)
that uses the declared classes, fields, functions and globals correctly. Rules:
- A FREE function, never a member (the class declarations cannot change): take the object as a
  pointer parameter (`void ResetTarget(CameraTarget* cam, float t)`) and use its fields through it.
- Give it a new descriptive CamelCase name that appears nowhere in the declarations. Never reuse or
  imitate the example's name, never an fn_/lbl_/vf name.
- Use only members, functions and globals that are declared; do not invent any. There is no `this`:
  reach every field through a pointer parameter, and skip macros that name fields directly.
- C++98 only (no nullptr/auto/lambdas). Never use volatile, #pragma, asm, casts of raw pointers
  plus offsets, or dummy code.
Reply with exactly one fenced ```cpp block holding only that function."""


def bt_prompt(decls: str, example: str, focus: str) -> list[dict]:
    user = (f"## Declarations in scope\n```cpp\n{decls}```\n\n## A real function of this file (style only: its name "
            f"is taken)\n```cpp\n{example.strip()}\n```\n\nWrite a new free function that works with {focus}.")
    return [{"role": "system", "content": BT_SYSTEM}, {"role": "user", "content": user}]


def new_symbols(before: dict, after: dict) -> list[str]:
    return [n for n, e in after.items() if e.kind == "fn" and n not in before and not n.startswith(("@", "__sinit"))]


def bt_fix(errors: str, first_line: int) -> str:
    """The feedback for a function that does not compile (its lines start at `first_line` in the file)."""
    return (f"It does not compile (your function starts at line {first_line} of the file):\n```\n{errors}\n```\n"
            "Reply with the whole corrected function, still a new free function using only declared names.")


def bt_one(cfg: Config, f: dict, client, st, k: int, shapes: set[str], forbidden, log, rounds: int = 2) -> dict | None:
    """One back-translation: prompt, reply, compile at the end of the unit (a compile error goes back to the
    model once), label with its own asm."""
    from .llmtry import PREFILL, compiler_errors, declarations, extract_prefilled, header_classes, prepare
    from .reorder import compile_variant
    u, source, src, span, _loc, ctx = prepare(cfg, f["unit"], f["function"], m2c=False)
    m = re.match(r"\s*(?:[\w:<>*&\s]+?)\b(\w+)::", ctx.current or "")
    focus = f"the class `{m[1]}`" if m else "the declared types"
    decls = declarations(src, (0, 0), set(), 9000, False)
    if m and not re.search(rf"(?:class|struct)\s+{re.escape(m[1])}\b[^;{{]*\{{", src):
        decls = header_classes(cfg, src, {m[1]}) + "\n\n" + decls
    msgs = bt_prompt(decls, ctx.current, focus)
    seed = st.seed + k
    row = {"kind": "backtrans", "unit": f["unit"], "example": f["function"], "k": k, "seed": seed, "model": st.model}
    work = cfg.root / "build" / "dk" / "llm" / "work" / f"bt_{hkey(f['unit'], str(k)):08x}"
    obj = None
    for rnd in range(max(1, rounds)):                  # a compile error goes back once, as llm-try's rounds do
        t0 = time.time()
        row["round"] = rnd
        try:
            reply = client.chat(st.model, msgs + [{"role": "assistant", "content": PREFILL}],
                                {**st.options(seed + 1000 * rnd), "num_predict": 1024, "stop": ["```"]}, False)
        except Exception as e:
            log({**row, "error": f"ollama: {e}"[:200]})
            return None
        code, fmt = extract_prefilled(reply.get("message", {}).get("content", ""), reply.get("done_reason"))
        row.update(secs=round(time.time() - t0, 1), gen_tokens=reply.get("eval_count", 0), format=fmt)
        if not code:
            log({**row, "result": "no code"})
            return None
        fns = cbody.functions(code)
        if len(fns) != 1:
            log({**row, "result": f"{len(fns)} functions", "code": code})
            return None
        text = src.rstrip("\n") + "\n\n" + code.strip("\n") + "\n"
        try:
            obj, err = compile_variant(cfg, u, text, work)
            if obj is None:
                errs = compiler_errors(err or "")
                log({**row, "result": "no compile", "code": code, "error": errs[:300]})
                msgs = msgs + [{"role": "assistant", "content": f"```cpp\n{code.strip()}\n```"},
                               {"role": "user", "content": bt_fix(errs, len(src.rstrip(chr(10)).splitlines()) + 2)}]
                continue
            base_obj = cfg.root / u["base_path"]
            before = object_asm(cfg, base_obj) if base_obj.is_file() else {}
            listing = Path(tempfile.mkdtemp(prefix="dk-bt-")) / "o.s"
            after = object_asm(cfg, obj, listing)
            break
        finally:
            shutil.rmtree(work, ignore_errors=True)
    if obj is None:
        return None
    try:
        return bt_label(cfg, f, u, source, src, code, text, before, after, listing, st, seed, shapes, forbidden, log, row)
    finally:
        shutil.rmtree(listing.parent, ignore_errors=True)


def bt_draft(cfg: Config, sym: str, listing: Path, source: Path, src: str) -> str:
    """m2c's draft of a new function from its object's listing, cleaned as llm-try's prompt cleans one (an
    unwritten function's prompt always has a draft)."""
    from .llmtry import clean_draft, m2c_draft, visible_text
    draft = m2c_draft(cfg, sym, listing, str(source).endswith(".c")) if listing.is_file() else ""
    return clean_draft(draft, visible_text(cfg, src, source), sym) if draft else ""


def bt_label(cfg: Config, f: dict, u: dict, source, src: str, code: str, text: str, before: dict, after: dict,
             listing: Path, st, seed: int, shapes: set[str], forbidden, log, row: dict) -> dict | None:
    """The compiled new function -> its record (one new symbol, judge-clean, not a never-train shape)."""
    from .judge import CLEAN, review
    from .llmtry import asm_shape, compact_asm, declarations, header_classes, identifiers
    new = new_symbols(before, after)
    if len(new) != 1:
        log({**row, "result": f"{len(new)} new symbols", "code": code})
        return None
    sym = new[0]
    entry = after[sym]
    cat, flags = review(src, text, sym, forbidden)
    shape = asm_shape(entry.text)
    n_insns = sum(1 for x in compact_asm(entry.text).splitlines() if re.match(r"^\s*[0-9A-F]+  \S", x))
    why = ("judge: " + "; ".join(flags[:3]) if cat != CLEAN else "never-train asm shape" if shape in shapes
           else "too small" if n_insns < 6 else "")
    log({**row, "result": why or "kept", "symbol": sym, "code": code, "insns": n_insns})
    if why:
        return None
    span2 = (len(src.rstrip("\n")) + 2, len(text))
    wanted = identifiers(code) | set(entry.calls) | set(entry.refs)
    d2 = declarations(text, span2, wanted, 8000, True)
    cls = {w for w in identifiers(code) if w[:1].isupper() and not re.search(rf"(?:class|struct)\s+{re.escape(w)}\b", src)}
    extra = header_classes(cfg, src, cls)
    from .llmdata import RECORD_VERSION
    from .llmtry import PROMPT_VERSION
    return {"v": RECORD_VERSION, "source": "backtrans", "kind": "direct", "unit": f["unit"], "function": sym,
            "size": 4 * n_insns, "asm": compact_asm(entry.text), "asm_shape": shape,
            "decls": (extra + "\n\n" + d2) if extra else d2, "m2c": literal_labels(bt_draft(cfg, sym, listing, source, src), code),
            "wrong": "", "wrong_percent": 0.0, "rows": [],
            "diff_class": "", "exact": clean_answer(code), "provenance": {"model": st.model, "seed": seed, "example": f["function"],
                                                                          "time": time.time()},
            "prompt_version": PROMPT_VERSION}


def backtrans(cfg: Config, args) -> int:
    from .judge import forbidden_rules
    from .llmdata import candidates, compiler_of, out_dir, read_jsonl, write_records
    from .llmtry import Ollama, Settings, _until, never_train, open_log
    root = out_dir(cfg, args)
    shapes = never_train(cfg)[2]
    forbidden = forbidden_rules(cfg)
    q = flagged(cfg)
    fns = [f for f in candidates(cfg, args, False) if 0x40 <= f["size"] <= 0x400 and f["function"] not in q]
    by_unit: dict[str, list[dict]] = {}
    for f in sorted(fns, key=lambda f: hkey("bt", f["function"])):
        by_unit.setdefault(f["unit"], []).append(f)
    order = sorted(by_unit, key=lambda n: hkey("bt-unit", n))
    picks = [by_unit[n][j % len(by_unit[n])] for j in range(args.per_unit) for n in order]
    if args.limit:
        picks = picks[:args.limit]
    run = args.run or f"backtrans-{time.strftime('%Y%m%d')}"
    path, fh, log = open_log(cfg, run)
    done = set()
    if args.resume:
        done = {(r.get("unit"), r.get("example"), r.get("k")) for r in read_jsonl(path) if r.get("kind") == "backtrans"}
    st = Settings(args.model, seed=args.seed, temperature=args.temperature)
    client = Ollama(args.url)
    until = _until(args.until)
    dest = root / "records" / "backtrans.jsonl"
    recs = read_jsonl(dest)                          # nights add up (one record per asm shape)
    have = {r["asm_shape"] for r in recs}
    comp_memo: dict = {}
    t0, tried = time.time(), 0
    try:
        for i, f in enumerate(picks):
            for k in range(args.n):
                if (f["unit"], f["function"], k) in done:
                    continue
                if until and time.time() >= until:
                    print(f"stopping at {args.until}; rerun with --resume --run {run}")
                    raise StopIteration
                tried += 1
                try:
                    r = bt_one(cfg, f, client, st, k, shapes, forbidden, log, args.rounds)
                except StopIteration:
                    raise
                except Exception as e:
                    log({"kind": "backtrans", "unit": f["unit"], "example": f["function"], "k": k, "error": str(e)[:200]})
                    r = None
                if r and r["asm_shape"] not in have:
                    r.update(compiler_of(cfg, f["u"], comp_memo))
                    have.add(r["asm_shape"])
                    recs.append(r)
                    write_records(dest, list(recs))
                print(f"[{i + 1}/{len(picks)}] {f['unit']} v{k}: " + (f"kept {r['function']}" if r else "dropped"))
    except StopIteration:
        pass
    finally:
        fh.close()
    print(f"backtrans: {len(recs)} pairs in {cfg.rel(dest)} ({tried} tried this run, {time.time() - t0:.0f} s); log {cfg.rel(path)}")
    return OK


# ---- argument wiring (called from llmdata's parser) ---------------------------------------------------

def configure(sub) -> None:
    def common(s, per_fn: int | None = None):
        s.add_argument("--only", default="", help="units whose name contains one of these (comma-separated)")
        s.add_argument("--min-size", type=lambda x: int(x, 0), default=0x10, help="skip smaller functions (default 0x10)")
        s.add_argument("--jobs", type=int, default=0, help="parallel compiles (default the idle cores, at most 6)")
        s.add_argument("--out", default="", help="dataset directory (default llm-data/dataset)")
        s.add_argument("--limit", type=int, default=0, help="first N functions only (a test)")
        if per_fn is not None:
            s.add_argument("--per-fn", type=int, default=per_fn, help=f"records per function (default {per_fn})")
    c = sub.add_parser("coverage", help="base records: answers using a member or call the prompt never shows, v6 "
                       "(stored) vs v7 (llmdecls, from HEAD's source and the same draft); CPU, one compile per unit")
    c.add_argument("--every", type=int, default=1, help="every Nth base record (default all)")
    c.add_argument("--out", default="", help="dataset directory (default llm-data/dataset)")
    b = sub.add_parser("base", help="every exact function as llm-try's unwritten-function prompt -> its source (structured)")
    common(b)
    b.add_argument("--no-m2c", dest="m2c", action="store_false", help="no m2c draft (declarations from the source instead)")
    a = sub.add_parser("attempts", help="GPU: the model writes solved functions from their asm (llm-bench run "
                       "--set heldout over llm-data/dataset/solved.json, the same loop and log)")
    common(a)
    a.add_argument("--pick", type=int, default=0, help="functions to pick (default all; spread over areas)")
    a.add_argument("--max-size", type=lambda x: int(x, 0), default=0x1000, help="largest function (default 0x1000)")
    a.add_argument("--repick", action="store_true", help="pick again (default: reuse solved.json)")
    a.add_argument("--pick-only", action="store_true", help="write solved.json and stop")
    a.add_argument("--until", default="", help="HH:MM: start no new function after this time")
    a.add_argument("--resume", action="store_true", help="append to --run's log, skipping functions it finished")
    from .llmtry import _model_args
    _model_args(a, model="qwen3-coder:30b")
    m = sub.add_parser("mine", help="attempt logs (dataset attempts, llm-grind) -> repair records for functions exact today")
    common(m, 4)
    m.add_argument("logs", nargs="+", help="attempt logs (never held-out or benchmark runs: their functions are refused)")
    m.add_argument("--negatives", type=int, default=2, help="device attempts per function kept as DPO negatives (default 2)")
    m.add_argument("--replace", action="store_true", help="replace the records (default: merge with earlier logs)")
    e = sub.add_parser("repair-eval", help="held-out logs -> llm-data/repair-eval.json: compiling, judge-clean, non-exact "
                       "attempts with the exact answer known (never trained on)")
    e.add_argument("logs", nargs="+", help="held-out run logs (llm-bench run --set heldout)")
    e.add_argument("--min", type=float, default=50.0, help="lowest score (default 50)")
    e.add_argument("--max", type=float, default=99.9, help="below this score (default 99.9)")
    e.add_argument("--per-fn", type=int, default=2, help="attempts per function (default 2)")
    e.add_argument("--dest", default="", help="where (default llm-data/repair-eval.json)")
    s = sub.add_parser("synth", help="natural mutants (close's rewrite table + loop form) that compile and change the asm -> repairs")
    common(s, 3)
    s.add_argument("--max-rule-share", type=float, default=0.35, help="no rule above this share of the mutants (default 0.35)")
    r = sub.add_parser("rename", help="a share of the exact functions with locals renamed; kept when the asm is identical")
    common(r)
    r.add_argument("--share", type=float, default=0.12, help="share of the functions (default 0.12)")
    r.add_argument("--no-m2c", dest="m2c", action="store_false", help="no m2c draft in the prompt")
    t = sub.add_parser("backtrans", help="GPU: new functions over a unit's real declarations, labelled by the compiler")
    common(t)
    t.add_argument("--model", default="qwen3-coder:30b", help="Ollama model (default qwen3-coder:30b)")
    t.add_argument("--url", default="http://127.0.0.1:11434")
    t.add_argument("--n", type=int, default=2, help="functions per example (default 2)")
    t.add_argument("--per-unit", type=int, default=3, help="examples per unit (default 3)")
    t.add_argument("--rounds", type=int, default=2, help="rounds per function: a compile error goes back (default 2)")
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--temperature", type=float, default=0.8)
    t.add_argument("--until", default="", help="HH:MM: stop then")
    t.add_argument("--resume", action="store_true", help="go on with --run's log (skip the examples it did)")
    t.add_argument("--run", default="", help="log name (default backtrans-<date>)")


def run(cfg: Config, args) -> int:
    if args.action == "coverage":
        from .llmdecls import coverage
        return coverage(cfg, args)
    return {"attempts": attempts, "repair-eval": repair_eval, "base": base, "mine": mine, "synth": synth, "rename": rename_cmd, "backtrans": backtrans}[args.action](cfg, args)
