"""dk dataset: training records for the local model, stored structured, rendered only at export.
  ./dk dataset history [--commits 40] [--per-fn 4] [--limit N]   committed WIP versions of exact functions
  ./dk dataset regs [--per-fn 3] [--limit N]                      declaration-order variants of exact functions
  ./dk dataset base|attempts|mine|synth|rename|backtrans          more sources (llmgen.py)
  ./dk dataset export [--weights SRC=W,..] [--prompt-version V]  graded, curriculum-ordered chat JSONL + DPO + card
A record holds the unit, function, compiler and flags, target asm, declarations, the wrong
definition with its score and diff rows (none for a direct pair), the exact definition and where
the wrong one came from. `history`: every earlier committed version of a now-exact function's
definition, spliced into today's source and re-scored privately (a version missing identifiers gets
its own old declarations of them added; scores cached by source hash). `regs`: dk regs' declaration
reorder of the exact definition's simple locals, kept when the asm differs. `export` renders every
record with llm-try's prompt builder for the chosen prompt version, runs llm-review's rules on every
answer, checks leakage (unit, asm shape, each mutant's own shape, never-train definitions in the
text), caps variants per function and the generated share, grades easy -> extreme and writes the
rows in curriculum order, DPO pairs, and a card. Data stays in llm-data/dataset/ (outside the repo).
Why: the grind is repair work, so the training data must be repair work in inference format.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from ...core.command import OK, USAGE, Command
from ...core.config import Config

WEIGHTS = {"base": 1, "synth": 1, "regs": 2, "history": 3, "grind": 3, "attempts": 3, "rename": 1, "backtrans": 1}
# real repair work and the model's own realistic attempts > register orders > cold-start pairs and rule
# mutants (2026-11-02: synth at 2 was 57% of the trained tokens). `export --weights src=w,...` overrides.
OPT_IN_SOURCES = ("lessons",)   # read only when --sources names them (the default export is frozen)
MAX_GENERATED_SHARE = 0.55   # generated sources' share of the trained (weighted) tokens at most
RECORD_VERSION = 2                                   # 2: kind direct|repair, first_round, m2c, devices


def norm(code: str) -> str:
    """Code with comments and whitespace runs removed (what counts as the same version)."""
    code = re.sub(r"//[^\n]*|/\*.*?\*/", " ", code, flags=re.S)
    return re.sub(r"\s+", " ", code).strip()


def key(*parts: str) -> str:
    return hashlib.sha1("\0".join(parts).encode()).hexdigest()


def distinct_versions(versions: list[tuple[dict, str]], exact: str) -> list[tuple[dict, str]]:
    """(commit meta, definition) newest first -> one per distinct version other than the exact one,
    each with the OLDEST commit that had it (when it was first written), returned oldest first."""
    seen: dict[str, tuple[dict, str]] = {}
    target = norm(exact)
    for meta, code in versions:
        n = norm(code)
        if n and n != target:
            seen[n] = (meta, code)                 # later in the list = older commit: overwrite
    return sorted(seen.values(), key=lambda v: v[0].get("time", 0))


def steps(scored: list[tuple[dict, str, float]]) -> list[dict]:
    """Oldest-first (meta, code, percent) versions of one function -> step facts: each non-exact
    version with the next version's score (100 after the last) and whether that step improved."""
    out = []
    for i, (meta, code, pct) in enumerate(scored):
        if pct >= 100.0:
            continue
        nxt = scored[i + 1][2] if i + 1 < len(scored) else 100.0
        out.append({"meta": meta, "code": code, "percent": pct, "next_percent": nxt, "improving": nxt > pct})
    return out


def local_names(chunk: str) -> list[str]:
    """Simple single-name local declarations of a definition, in source order (dk regs' pattern)."""
    from .regs import DECL, FOR_DECL
    lines = chunk.splitlines()
    names, depth = [], 0
    for line in lines:
        if depth > 0:
            m = DECL.match(line)
            if m and m.group("type").strip() not in ("return", "delete", "goto", "else", "case") \
                    and m.group("name") not in names:
                names.append(m.group("name"))
            else:
                m = FOR_DECL.search(line)
                if m and m.group("name") not in names:
                    names.append(m.group("name"))
        depth += line.count("{") - line.count("}")
    return names


def orders(names: list[str], limit: int) -> list[list[str]]:
    """Other declaration orders, nearest first: each adjacent swap, then one local moved first, reversed."""
    out: list[list[str]] = []
    cands = [names[:i] + [names[i + 1], names[i]] + names[i + 2:] for i in range(len(names) - 1)]
    cands += [[names[i]] + names[:i] + names[i + 1:] for i in range(2, len(names))]
    cands.append(list(reversed(names)))
    for c in cands:
        if c != names and c not in out:
            out.append(c)
    return out[:limit]


def reordered(chunk: str, symbol: str, order: list[str]) -> str | None:
    """The definition with `order` as its locals' declaration order (dk regs' minimal moves)."""
    from .regs import reorder
    got = reorder(chunk, symbol, order, minimal=True)
    if got is None:
        return None
    return got if chunk.endswith("\n") else got.rstrip("\n")


# ---- rendering (llm-try's own prompt builder) ---------------------------------------------------------

def builder(version: int = 0):
    """llm-try's prompt builder for a prompt version (0: the current default). Versions differ in the
    draft clean-up and notes (stored per function in the base records' views) and may differ in the
    builder: llmtry.PROMPT_BUILDERS {version: build_messages} when it keeps several."""
    from . import llmtry
    versions = set(getattr(llmtry, "PROMPT_VERSIONS", (llmtry.PROMPT_VERSION,)))
    version = version or llmtry.PROMPT_VERSION
    if version not in versions:
        raise ValueError(f"prompt version {version}: llm-try has {sorted(versions)}")
    return getattr(llmtry, "PROMPT_BUILDERS", {}).get(version, llmtry.build_messages)


def zero_starts(cfg: Config) -> set[tuple[str, str]]:
    """(unit stem, function) of the functions at offset 0 of a section of their unit's object: symbols.txt
    puts them at the start of the unit's split. objdiff leaves such an instruction's address out (0), which
    the rows of llm-try before the fix took the second instruction's address (4) for."""
    from .dtk_symbols import read_symbols, read_units
    starts = {(sec, a): u.name.rsplit(".", 1)[0] for u in read_units(cfg.root / cfg.platform_options["splits"])
              for sec, a, _end in u.splits}
    return {(starts[(s.section, s.address)], s.name) for s in read_symbols(cfg.root / cfg.platform_options["symbols"])
            if s.type == "function" and (s.section, s.address) in starts}


def correct_rows(recs: list[dict], zero: set[tuple[str, str]]) -> dict[str, int]:
    """Put right the diff rows of the records made before the relative_rows fix: a function at offset 0 of its
    unit (`zero`, from zero_starts) had every row offset and branch target 4 too small and its first offset
    blank; both sides are moved by 4 (ours is first in its object too, nearly always). `rows_fixed` marks rows
    that are right: new records carry it from birth, so this is idempotent. {source: records changed}."""
    from .llmtry import shift_rows
    changed: dict[str, int] = {}
    for r in recs:
        if r.get("rows_fixed") or not r.get("rows"):
            continue
        if (r["unit"].split("/", 1)[-1], r["function"]) in zero:
            rows = shift_rows(r["rows"])
            if rows is None:                              # does not parse back to itself: left alone, unmarked
                changed["unparsed"] = changed.get("unparsed", 0) + 1
                continue
            r["rows"] = rows
            changed[r["source"]] = changed.get(r["source"], 0) + 1
        r["rows_fixed"] = 1
    return changed


def view_version(version: int) -> int:
    """The prompt version whose stored view a version renders with (v8: v7's, its asm annotated on top)."""
    from . import llmtry
    return getattr(llmtry, "VIEW_OF", {}).get(version, version)


def render(rec: dict, build=None, view: dict | None = None, examples: str = "", diag: str = "") -> dict:
    """A structured record -> {"messages"}: llm-try's prompt + the exact definition as the reply.
    A repair (a wrong definition): round one shows it as the current definition, a model's own attempt
    (first_round false) as the best attempt so far, with its score and diff. A direct pair (no wrong
    definition): the prompt for an unwritten function (its m2c draft when stored, a member's signature).
    `view`: the function's prompt-version view (m2c draft, blank declarations, notes) from its base record;
    `examples`: llm-try's retrieval block (similar exact functions), as its prompt shows them. `diag` (round 2,
    opt-in): llmdiag's line, first in the answer before the fence."""
    from .llmtry import Context, Ingredients, signature
    build = build or builder()
    view = view or {}
    notes = view.get("notes", rec.get("notes", ""))
    if rec.get("wrong"):
        ctx = Context(rec["function"], rec["asm"], rec["decls"], "", rec["wrong"])
        ctx.notes, ctx.examples = notes, examples
        ctx.context = view.get("context", "")        # v7: merged before the record's own declarations
        msgs = build(ctx, rec["wrong"], rec["wrong_percent"], rec["rows"], Ingredients(), rec.get("first_round", True))
    else:
        try:
            sig = signature(rec["function"])
        except Exception:
            sig = ""
        m2c = view.get("m2c", rec.get("m2c", ""))
        ctx = Context(rec["function"], rec["asm"], view.get("decls", rec["decls"]), m2c, "", sig)
        ctx.notes, ctx.examples = notes, examples
        ctx.context = view.get("context", "")
        msgs = build(ctx, "", 0.0, [], Ingredients(m2c="auto" if m2c else "never"), True)
    msgs.append({"role": "assistant", "content": (diag + "\n" if diag else "") + f"```cpp\n{rec['exact'].strip()}\n```"})
    return {"messages": msgs}


class Round2:
    """Opt-in export extras (the LLM plan, Round 2): prompt v8's annotated asm on a hashed `annotate_share`
    of the records (the rest unannotated, so sparse layouts still work), from `dataset annotate`'s
    records/annot.jsonl; and llmdiag's DIAG line before the code on a hashed `diag_share` of the repairs."""

    def __init__(self, annot: dict | None, annotate_share: float = 0.0, diag_share: float = 0.0):
        self.annot, self.share, self.diag_share = annot or {}, annotate_share, diag_share
        self.card = {"annotate_share": annotate_share if annot is not None else 0.0, "annotated": 0,
                     "no_annotation": 0, "diag_share": diag_share, "diag": 0, "diag_classes": {}}

    def asm(self, rec: dict, salt: str = "") -> str | None:
        if self.share <= 0 or int(key("annot", rec["unit"], rec["function"], rec.get("wrong", ""), salt)[:8], 16) \
                % 1000 >= self.share * 1000:
            return None
        from .llmannot import _sha, apply_stored
        rows = self.annot.get((rec["unit"], rec["function"]))
        if not rows:
            self.card["no_annotation"] += 1
            return None
        row = rows.get(_sha(rec["asm"])) or next(iter(rows.values()))
        self.card["annotated"] += 1
        return apply_stored(rec["asm"], row["notes"])

    def diag(self, rec: dict) -> str:
        if self.diag_share <= 0 or not rec.get("wrong") or \
                int(key("diag", rec["unit"], rec["function"], rec["wrong"])[:8], 16) % 1000 >= self.diag_share * 1000:
            return ""
        from .llmdiag import diag_class, diagnose
        line = diagnose(rec)
        if line:
            self.card["diag"] += 1
            c = diag_class(rec)
            self.card["diag_classes"][c] = self.card["diag_classes"].get(c, 0) + 1
        return line

    def render(self, rec: dict, build, view, examples: str = "", salt: str = "") -> dict:
        a = self.asm(rec, salt)
        r = render({**rec, "asm": a} if a else rec, build, view, examples, self.diag(rec) if not salt else "")
        r["annotated"] = a is not None
        return r


def load_annotations(path: Path) -> dict:
    """records/annot.jsonl -> {(unit, function): {asm_sha: row}}."""
    out: dict = {}
    for r in read_jsonl(path) if path.is_file() else []:
        out.setdefault((r["unit"], r["function"]), {})[r["asm_sha"]] = r
    return out


class Examples:
    """llm-try's retrieval examples for a record, as its prompt adds them (the same index: HEAD's exact
    functions minus benchmark and held-out units and shapes, prompt.k most similar, own compiler first),
    leave-one-out (never the record's own function). A hashed `zero_share` of the records gets none, so
    the model also sees prompts where retrieval found nothing. Memoized per function (its target asm)."""

    def __init__(self, cfg: Config, k: int, zero_share: float):
        from .retrieval import Index, compiler_of
        from .llmtry import _git
        from .matching import resolve_unit
        self.k, self.zero, self.memo, self.comp = k, zero_share, {}, {}
        self.index = Index.load(cfg, _git(cfg, "rev-parse", "HEAD").strip(), current=True) if k > 0 else None
        self.cfg, self.resolve, self.compiler_of = cfg, resolve_unit, compiler_of

    def compiler(self, unit: str) -> str:
        if unit not in self.comp:
            try:
                self.comp[unit] = self.compiler_of(self.cfg, self.resolve(self.cfg, unit))
            except Exception:
                self.comp[unit] = ""
        return self.comp[unit]

    def __call__(self, rec: dict, salt: str = "") -> tuple[str, int]:
        """(the examples block, how many) for one record."""
        if self.index is None or int(key("ex", rec["unit"], rec["function"], rec.get("wrong", ""), salt)[:8], 16) \
                % 1000 < self.zero * 1000:
            return "", 0
        k = (rec["unit"], rec["function"])
        if k not in self.memo:
            from .retrieval import render as render_examples
            size = sum(1 for line in rec["asm"].splitlines() if re.match(r"^\s*[0-9A-F]+  \S", line)) * 4
            got = self.index.top(rec["asm"], size, self.compiler(rec["unit"]), {rec["function"]}, self.k)
            self.memo[k] = (render_examples(got), len(got))
        return self.memo[k]


def chars(messages: list[dict]) -> int:
    return sum(len(m["content"]) for m in messages)


GATE_HASHES: dict[str, str] = {}      # content hash -> frozen set (gate.hashes), filled by never_definitions


def never_definitions(cfg: Config, fns: set[str]) -> list[re.Pattern]:
    """Patterns that find a definition of a never-train function in a prompt or an answer. Also loads the frozen
    sets' definition and variant hashes that `leaks` checks against every fenced code block."""
    from . import gate
    GATE_HASHES.clear()
    GATE_HASHES.update(gate.hashes(cfg) if cfg is not None else {})
    from .closecmd import plain_name
    from .llmtry import qualified
    pats = []
    for f in sorted(fns):
        q = qualified(f)
        if (q or "").startswith("operator") or (plain_name(f) or "").startswith("operator"):
            continue                       # `operator=` alone names every class's: such functions are caught by name, shape and hash
        if q and "::" in q:
            pats.append(re.compile(re.escape(q) + r"\s*\([^;{}]*\)\s*(?:const\s*)?\{"))   # a body, not a call
            LITERAL[pats[-1].pattern] = q
        else:
            p = plain_name(f) or f
            pats.append(re.compile(r"(?<![\w:.>])" + re.escape(p) + r"\s*\([^;{)]*\)\s*(?:const\s*)?\{"))
            LITERAL[pats[-1].pattern] = p
    return pats


LITERAL: dict[str, str] = {}      # pattern -> the name it needs (a fast substring test before the regex)


def leaks(rec: dict, units: set[str], fns: set[str], shapes: set[str], pats: list[re.Pattern]) -> str | None:
    """Why a rendered record may not be trained on (None: clean)."""
    t = rec["tags"]
    if t["unit"] in units:
        return "unit"
    if t["function"] in fns:
        return "function"
    if t.get("asm_shape") in shapes:
        return "asm shape"
    if t.get("mutant_shape") and t["mutant_shape"] in shapes:
        return "mutant asm shape"
    text = "\n".join(m["content"] for m in rec["messages"][1:])
    if GATE_HASHES and "```" in text:
        from . import gate
        for block in gate.fenced(text):
            if gate.code_hashes(block) & GATE_HASHES.keys():
                return "frozen definition hash"
    for p in pats:
        lit = LITERAL.get(p.pattern)
        if lit is not None and lit not in text:
            continue
        if p.search(text):
            return "never-train definition in text"
    return None


# ---- shared plumbing ----------------------------------------------------------------------------------

def out_dir(cfg: Config, args) -> Path:
    from .llmtry import data_dir
    return Path(args.out) if getattr(args, "out", "") else data_dir(cfg) / "dataset"


class Cache:
    """Scores of spliced sources by hash (llm-data/dataset/cache/scores.jsonl), shared by threads."""

    def __init__(self, path: Path):
        self.path, self.lock, self.rows = path, threading.Lock(), {}
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            for line in path.read_text().splitlines():
                try:
                    r = json.loads(line)
                    self.rows[r["key"]] = r
                except (ValueError, KeyError):
                    pass

    def get(self, k: str):
        return self.rows.get(k)

    def put(self, row: dict) -> None:
        with self.lock:
            self.rows[row["key"]] = row
            with self.path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")


def candidates(cfg: Config, args, need_history: bool) -> list[dict]:
    """Exact functions of linked or WIP game units (benchmark, held-out and vendored ports left out by
    unit, function and asm shape), each with its unit, source and commits."""
    from .llmtry import _vendored_paths, asm_entries, exact_functions, never_train
    from .matching import load_objdiff_units
    never_units, never_fns, shapes = never_train(cfg)
    vend = _vendored_paths(cfg)
    units = [u for u in load_objdiff_units(cfg) if u.get("base_path") and u["name"] not in never_units
             and not any(u["metadata"]["source_path"].startswith(v) for v in vend)]
    if args.only:
        units = [u for u in units if any(x in u["name"] for x in args.only.split(","))]
    entries = asm_entries(cfg)

    def one(u):
        src = (cfg.root / u["metadata"]["source_path"]).read_text(errors="replace")
        got = [f for f in exact_functions(cfg, u, src, args.min_size)
               if f["function"] not in never_fns and f["asm_shape"] not in shapes and f["function"] in entries]
        return [{**f, "u": u} for f in got]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        return [f for fs in pool.map(one, units) for f in fs]


def compiler_of(cfg: Config, u: dict, memo: dict) -> dict:
    """The unit's compiler version and flags (from its build command)."""
    if u["name"] not in memo:
        from .matching import compile_command
        try:
            cmd = compile_command(cfg, u["base_path"], locked=False).split()
        except Exception:
            cmd = []
        ver = next((m.group(0) for a in cmd for m in [re.search(r"GC/[\d.]+\w*", a)] if m), "")
        flags, skip = [], False
        for a in cmd[1:]:                            # options with their values; no paths, outputs or deps
            if skip or a in ("-o", "-MMD", "-c", "-i", "-I"):
                skip = a in ("-o", "-i", "-I") and not skip
                continue
            if "/" not in a or a.startswith("-") and "=" in a:
                flags.append(a)
        memo[u["name"]] = {"compiler": ver, "flags": " ".join(flags)}
    return memo[u["name"]]


def scored(cfg: Config, cache: Cache, scorer, fn: str, src_key: str, text: str, chunk: str, slot: str,
           shape: bool = False, fresh_failures: bool = False) -> dict:
    """Compile and score a spliced source (cached): percent, rows, diff class, error, the undefined
    identifiers of a failed compile; `shape`: also the asm shape of our non-exact function (dtk's
    disassembly, comparable with the never-train shapes)."""
    from .llmtry import classify
    from .regs import _diff
    k = key(scorer.unit["name"], fn, src_key, chunk, *(["shape"] if shape else []))
    hit = cache.get(k)
    if hit and not (fresh_failures and not hit.get("compiled") and "undefined" not in hit):
        return {**hit, "cached": True}
    import shutil
    from .llmtry import compiler_errors, relative_rows
    from .matching import MatchError, score
    from .reorder import compile_variant
    t0 = time.time()
    work = scorer.work / slot
    row = {"key": k}
    try:                                             # Scorer.compile_score's steps, keeping the diff pair
        obj, err = compile_variant(cfg, scorer.unit, text, work)
        s = None
        if obj is not None:
            try:
                s = score(cfg, scorer.unit["name"], cfg.root / scorer.unit["target_path"], obj)
            except MatchError as e:
                err = str(e)
        if s is None:
            from .llmgen import UNDEF
            row.update(compiled=False, error=(compiler_errors(err) or "compile error")[:300],
                       undefined=sorted(set(UNDEF.findall(err or "")))[:20])
        else:
            pct = {f.name: f.percent for f in s.functions}.get(fn, 0.0)
            pair = _diff(cfg, scorer.unit, obj, fn) if pct < 100.0 else None
            row.update(compiled=True, percent=pct, rows=relative_rows(pair[0], pair[1]) if pair else [],
                       rows_fixed=1, diff_class=classify(pair[0], pair[1]) if pair else "")
            if shape and pct < 100.0:
                from .llmgen import object_asm
                from .llmtry import asm_shape
                e = object_asm(cfg, obj).get(fn)
                row["shape"] = asm_shape(e.text) if e is not None else ""
    finally:
        shutil.rmtree(work, ignore_errors=True)
    row["secs"] = round(time.time() - t0, 2)
    cache.put(row)
    return row


def record(cfg: Config, f: dict, source: str, wrong: str, text: str, res: dict, exact: str, prov: dict, comp: dict) -> dict:
    """The structured record (declarations from the source holding the wrong definition, as llm-try sees it)."""
    from .llmtry import PROMPT_VERSION, prepare
    _u, _p, _s, _span, _loc, ctx = prepare(cfg, f["unit"], f["function"], src_text=text, m2c=False)
    return {"v": RECORD_VERSION, "source": source, "unit": f["unit"], "function": f["function"], "size": f["size"],
            **comp, "asm": ctx.asm, "asm_shape": f["asm_shape"], "decls": ctx.decls, "wrong": wrong,
            "wrong_percent": res["percent"], "rows": res["rows"], "rows_fixed": res.get("rows_fixed", 0),
            "diff_class": res.get("diff_class", ""), "exact": exact, "provenance": prov, "prompt_version": PROMPT_VERSION}


def write_records(path: Path, recs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    recs.sort(key=lambda r: (r["unit"], r["function"], r["provenance"].get("time", 0), r["wrong"]))
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))


# ---- history ----------------------------------------------------------------------------------------

def _git(cfg: Config, *a: str) -> str:
    return subprocess.run(["git", *a], cwd=cfg.root, capture_output=True, text=True, errors="replace").stdout


class memo_parses:
    """While gathering history: each old source is parsed once, not once per function looked up in it
    (trycmd.candidates parses the whole file per call). Copies are returned, so callers may change them."""

    def __enter__(self):
        from ...core import cbody
        from . import trycmd
        self.saved = (cbody.functions, trycmd.scopes)
        memo, lock = {}, threading.Lock()

        def wrap(fn, tag):
            def inner(src):
                k = (tag, hash(src), len(src))
                with lock:
                    hit = memo.get(k)
                if hit is None:
                    hit = fn(src)
                    with lock:
                        memo[k] = hit
                return list(hit) if isinstance(hit, list) else hit
            return inner
        cbody.functions, trycmd.scopes = wrap(self.saved[0], "f"), wrap(self.saved[1], "s")
        return self

    def __exit__(self, *exc):
        from ...core import cbody
        from . import trycmd
        cbody.functions, trycmd.scopes = self.saved
        return False


def history(cfg: Config, args) -> int:
    from .llmtry import Scorer, prepare
    from .trycmd import definition, variant_text
    t_start = time.time()
    root = out_dir(cfg, args)
    cache = Cache(root / "cache" / "scores.jsonl")
    fns = candidates(cfg, args, True)
    by_unit: dict[str, list[dict]] = {}
    for f in fns:
        by_unit.setdefault(f["unit"], []).append(f)
    print(f"{len(fns)} exact functions in {len(by_unit)} units ({time.time() - t_start:.0f} s)")
    def gather(group: list[dict]) -> list[tuple]:
        path = group[0]["u"]["metadata"]["source_path"]
        log = _git(cfg, "log", f"-n{args.commits}", "--format=%H\t%ct\t%s", "--", path).splitlines()
        olds = []
        for line in log[1:] if log else []:           # the newest commit is (usually) today's text
            h, ct, subj = (line.split("\t", 2) + ["", ""])[:3]
            old = _git(cfg, "show", f"{h}:{path}")
            if old:
                olds.append(({"commit": h[:10], "time": int(ct), "subject": subj[:120]}, old))
        src = (cfg.root / path).read_text(errors="replace")
        got = []
        for f in group:
            span = definition(src, f["function"])
            if span is None:
                continue
            exact = src[span[0]:span[1]]
            vers, whole = [], {}
            for meta, old in olds:
                sp = definition(old, f["function"])
                if sp is not None:
                    vers.append((meta, old[sp[0]:sp[1]]))
                    whole[meta["commit"]] = old
            cap = args.per_fn_big if f["size"] >= 0x300 else args.per_fn       # the open work is big: keep more
            got += [(f, meta, chunk, whole[meta["commit"]]) for meta, chunk in distinct_versions(vers, exact)[-cap:]]
        return got
    with memo_parses(), ThreadPoolExecutor(max_workers=args.jobs) as pool:
        jobs_list = [j for js in pool.map(gather, [g for _n, g in sorted(by_unit.items())]) for j in js]
    print(f"{len(jobs_list)} earlier versions found")
    if args.limit:
        jobs_list.sort(key=lambda j: -j[1]["time"])   # recent history first
        jobs_list = jobs_list[:args.limit]
    print(f"{len(jobs_list)} earlier versions to score ({time.time() - t_start:.0f} s)")
    comp_memo: dict = {}
    stats = {"compiled": 0, "failed": 0, "exact": 0, "cached": 0, "secs": 0.0, "recovered": 0, "renamed": 0}
    lock = threading.Lock()

    known: dict[str, set[str]] = {}

    def score_one(i_job):
        i, (f, meta, chunk, old) = i_job
        try:
            u, src_path, src, span, locate, ctx = prepare(cfg, f["unit"], f["function"], m2c=False)
        except Exception:
            return None
        if args.recover:                                 # old names -> today's (the symbol at the same address)
            from .llmgen import forward_names, rename_all, symbols_at
            from .llmtry import identifiers, visible_text
            with lock:
                have = known.get(f["unit"])
            if have is None:
                have = identifiers(src) | identifiers(visible_text(cfg, src, src_path))
                with lock:
                    known[f["unit"]] = have
            missing = identifiers(chunk) - have
            mapping = forward_names(symbols_at(cfg, meta["commit"]), symbols_at(cfg, ""), missing) if missing else {}
            if mapping:
                chunk = rename_all(chunk, mapping)
                meta = {**meta, "renamed_forward": len(mapping)}
        scorer = Scorer(cfg, u, f["function"], src, span, locate)
        text = variant_text(src, f["function"], chunk, span, locate)
        res = scored(cfg, cache, scorer, f["function"], key(src), text, chunk, f"h{i}", fresh_failures=args.recover)
        added: list[str] = []
        for _round in range(6 if args.recover else 0):     # the old version's own declarations it still needs
            if res["compiled"] or not res.get("undefined"):
                break
            from .llmgen import old_declarations, old_headers, with_declarations
            new = old_declarations(old, set(res["undefined"]), src + "\n".join(added))
            if not new:                                  # declared in a header at that commit
                new = old_declarations(old_headers(cfg, meta["commit"], old), set(res["undefined"]), src + "\n".join(added))
            if not new:
                break
            added += new
            sp = definition(text, f["function"])
            text2 = with_declarations(text, sp[0] if sp else span[0], added)
            res = scored(cfg, cache, scorer, f["function"], key(src), text2, chunk + "\0" + "\n".join(added), f"h{i}")
            if res["compiled"]:
                text = text2
                meta = {**meta, "recovered_declarations": len(added)}
        with lock:
            stats["recovered"] += bool(added) and bool(res["compiled"])
            stats["renamed"] += bool(meta.get("renamed_forward")) and bool(res["compiled"])
            stats["cached" if res.get("cached") else "secs"] += 1 if res.get("cached") else res.get("secs", 0)
            if not res["compiled"]:
                stats["failed"] += 1
                return None
            stats["compiled"] += 1
            if res["percent"] >= 100.0:
                stats["exact"] += 1
        return f, meta, chunk, text, res, ctx.current
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = [r for r in pool.map(score_one, enumerate(jobs_list)) if r]
    per_fn: dict = {}
    for f, meta, chunk, text, res, exact in results:
        per_fn.setdefault((f["unit"], f["function"]), []).append((f, meta, chunk, text, res, exact))
    recs = []
    for items in per_fn.values():
        items.sort(key=lambda x: x[1]["time"])
        st = steps([(x[1], x[2], x[4]["percent"]) for x in items])
        by_code = {x[2]: x for x in items}
        for s in st:
            f, meta, chunk, text, res, exact = by_code[s["code"]]
            prov = {**meta, "next_percent": s["next_percent"], "improving": s["improving"]}
            try:
                recs.append(record(cfg, f, "history", chunk, text, res, exact, prov, compiler_of(cfg, f["u"], comp_memo)))
            except Exception:
                continue
    write_records(root / "records" / "history.jsonl", recs)
    print(f"history: {len(recs)} repair records from {len(per_fn)} functions into {cfg.rel(root)}/records/history.jsonl")
    print(f"  scored {len(jobs_list)}: {stats['compiled']} compiled ({stats['exact']} already exact), "
          f"{stats['failed']} did not compile (old names), {stats['renamed']} compiled with old names mapped forward, "
          f"{stats['recovered']} with their old declarations, "
          f"{stats['cached']} cached; "
          f"compile+score CPU {stats['secs']:.0f} s; wall {time.time() - t_start:.0f} s")
    return OK


# ---- register / declaration order -------------------------------------------------------------------

def regs_cmd(cfg: Config, args) -> int:
    from .llmtry import Scorer, devices, prepare
    from .trycmd import variant_text
    t_start = time.time()
    root = out_dir(cfg, args)
    cache = Cache(root / "cache" / "scores.jsonl")
    fns = candidates(cfg, args, False)
    work = []
    for f in fns:
        try:
            u, _p, src, span, locate, ctx = prepare(cfg, f["unit"], f["function"], m2c=False)
        except Exception:
            continue
        if not ctx.current:
            continue
        names = local_names(ctx.current)
        if len(names) < 2:
            continue
        for order in orders(names, args.per_fn):
            chunk = reordered(ctx.current, f["function"], order)
            if chunk and norm(chunk) != norm(ctx.current):
                work.append((f, u, src, span, locate, ctx.current, order, chunk))
    if args.limit:
        work = work[:args.limit]
    print(f"{len(fns)} exact functions; {len(work)} declaration-order variants to score ({time.time() - t_start:.0f} s)")
    comp_memo: dict = {}
    stats = {"compiled": 0, "same": 0, "failed": 0, "device": 0, "secs": 0.0}
    lock = threading.Lock()
    from .judge import forbidden_rules
    forbidden = forbidden_rules(cfg)

    def one(i_w):
        i, (f, u, src, span, locate, exact, order, chunk) = i_w
        scorer = Scorer(cfg, u, f["function"], src, span, locate)
        text = variant_text(src, f["function"], chunk, span, locate)
        res = scored(cfg, cache, scorer, f["function"], key(src), text, chunk, f"r{i}")
        dev = devices(src, text, chunk, f["function"], forbidden) if res.get("compiled") else []
        with lock:
            stats["secs"] += 0 if res.get("cached") else res.get("secs", 0)
            if not res["compiled"]:
                stats["failed"] += 1
                return None
            stats["compiled"] += 1
            if res["percent"] >= 100.0:
                stats["same"] += 1
                return None
            if dev:
                stats["device"] += 1
                return None
        return f, chunk, text, res, exact, order
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        got = [r for r in pool.map(one, enumerate(work)) if r]
    recs = []
    for f, chunk, text, res, exact, order in got:
        try:
            recs.append(record(cfg, f, "regs", chunk, text, res, exact, {"order": order},
                               compiler_of(cfg, f["u"], comp_memo)))
        except Exception:
            continue
    write_records(root / "records" / "regs.jsonl", recs)
    reg = sum(1 for r in recs if r["diff_class"] == "register")
    print(f"regs: {len(recs)} records ({reg} register-only, {len(recs) - reg} shape) into {cfg.rel(root)}/records/regs.jsonl")
    print(f"  scored {len(work)}: {stats['compiled']} compiled, {stats['same']} same asm, {stats['failed']} did not "
          f"compile, {stats['device']} with a device; compile+score CPU {stats['secs']:.0f} s; wall {time.time() - t_start:.0f} s")
    return OK


# ---- export -----------------------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.is_file() else []


SOURCES = ("base", "history", "grind", "attempts", "regs", "synth", "rename", "backtrans")
KEEP_ORDER = {s: i for i, s in enumerate(SOURCES)}   # the per-function cap drops from the end of this list
LEVELS = ("easy", "medium", "hard", "very hard", "extreme")
LEVEL_CUTS = (0.30, 0.55, 0.75, 0.90)                # quantiles of the difficulty score
LEVEL_WEIGHT = (1.0, 1.0, 1.25, 1.5, 2.0)            # hard ones weighted up
_INSN = re.compile(r"^\s*[0-9A-F]+  (\S+)\s*(.*)$")
_CALL = re.compile(r"(?<![\w.])([A-Za-z_]\w*)\s*\(")
_NOT_CALLS = frozenset("if for while switch return sizeof catch".split())


def features(rec: dict) -> dict:
    """Size and shape facts of a record's target asm and answer, for grading."""
    f = {"insns": 0, "branches": 0, "loops": 0, "calls": 0, "vcalls": 0, "floats": 0}
    seen = set()
    for line in rec.get("asm", "").splitlines():
        st = line.strip()
        m = _INSN.match(line)
        if not m:
            if st.endswith(":"):
                seen.add(st[:-1])
            continue
        op, arg = m[1].rstrip(".+-"), m[2]
        f["insns"] += 1
        if op == "bl":
            f["calls"] += 1
        elif op == "bctrl":
            f["vcalls"] += 1
        elif op.startswith("b") and op not in ("b", "blr", "bctr") and not op.endswith("lr"):
            f["branches"] += 1
        if op.startswith("b") and arg.split(",")[-1].strip() in seen:
            f["loops"] += 1                         # a branch back to a label above it
        if op.startswith(("f", "lf", "stf", "psq")):
            f["floats"] += 1
    called = [c for c in _CALL.findall(rec.get("exact", "")) if c not in _NOT_CALLS]
    f["inlined"] = max(0, len(called) - 1 - f["calls"] - f["vcalls"])     # calls written but not in the asm
    return f


def difficulty(rec: dict) -> float:
    """Grade: size, control flow (branches, loops), C++ features (virtual calls, inlined calls), and for a
    repair how far its start is from exact (a register-only miss counts as subtle)."""
    import math
    f = features(rec)
    d = (1.6 * math.log2(1 + f["insns"] / 8) + 0.5 * min(f["branches"], 30) ** 0.6 + 0.7 * min(f["loops"], 6)
         + 0.6 * min(f["vcalls"], 10) ** 0.7 + 0.5 * min(f["inlined"], 10) ** 0.7 + 0.2 * min(f["calls"], 30) ** 0.6
         + 0.15 * min(f["floats"], 40) ** 0.5)
    if rec.get("wrong"):
        d += 0.5 * math.log2(1 + max(0.0, 100.0 - rec.get("wrong_percent", 0.0)))
        d += 0.8 if rec.get("diff_class") == "register" else 0.0
    return round(d, 3)


def levels(scores: list[float]) -> list[int]:
    """Level 0..4 of each score by the quantiles in LEVEL_CUTS."""
    xs = sorted(scores)
    cuts = [xs[min(len(xs) - 1, int(q * len(xs)))] for q in LEVEL_CUTS] if xs else []
    return [sum(1 for c in cuts if sc > c) for sc in scores]


def stage_of(level: int, h: int, mix_share: float = 0.25) -> int:
    """Curriculum stage (0..4): the record's level, but a hashed share of the hard ones (level 2+) go to
    an earlier stage, so every stage has some hard records (~10% of the first)."""
    if level >= 2 and h % 1000 < mix_share * 1000:
        return (h // 1000) % level
    return level


BANDS = ((0x100, "<0x100"), (0x300, "0x100-0x300"), (0x800, "0x300-0x800"), (1 << 30, ">0x800"))
BAND_KEEP = {"<0x100": 1.0, "0x100-0x300": 0.75, "0x300-0x800": 0.5, ">0x800": 0.35}   # cap order: big ones kept first


def band(size: int) -> str:
    return next(name for top, name in BANDS if size < top)


def quarantined(path: Path) -> set[str] | None:
    """Functions the judge audit flags (llm-data/judge-audit.json): its top-level "quarantine" list when it
    has one (the triaged audit), else every {"function": …} object in it and the strings of its
    "functions"/"flagged"/"quarantine" lists. None when there is no audit file."""
    if not path.is_file():
        return None
    out: set[str] = set()
    top = json.loads(path.read_text())
    if isinstance(top, dict) and isinstance(top.get("quarantine"), list):
        return {x for x in top["quarantine"] if isinstance(x, str)}

    def walk(x, k=""):
        if isinstance(x, dict):
            if isinstance(x.get("function"), str):
                out.add(x["function"])
            for kk, v in x.items():
                walk(v, kk)
        elif isinstance(x, list):
            for v in x:
                if isinstance(v, str) and k in ("functions", "flagged", "quarantine"):
                    out.add(v)
                else:
                    walk(v, k)
    walk(top)
    return out


def cap_per_function(recs: list[dict], cap: int, big_cap: int = 0) -> tuple[list[dict], int]:
    """At most `cap` records per (unit, function), `big_cap` for one of 0x300 bytes or more (the open work is
    big and big functions are few): real sources first (KEEP_ORDER), then by a hash."""
    by: dict[tuple, list[dict]] = {}
    for r in recs:
        by.setdefault((r["unit"], r["function"]), []).append(r)
    out, dropped = [], 0
    for items in by.values():
        items.sort(key=lambda r: (KEEP_ORDER.get(r["source"], 99), key(r["function"], r.get("wrong", ""), r["exact"])))
        c = max(cap, big_cap) if items[0].get("size", 0) >= 0x300 else cap
        out += items[:c]
        dropped += max(0, len(items) - c)
    return out, dropped


GEN_TIERS = (("attempts",), ("regs", "backtrans"), ("synth",), ("rename",))   # kept first under the generated cap


def cap_generated(recs: list[dict], share: float) -> tuple[list[dict], int]:
    """Generated records (GENERATED sources) cut to `share` of the total, tier by tier (GEN_TIERS: the
    model's own realistic attempts first, renamed copies last); in the tier that overflows the same
    fraction from each source, small functions dropped first (the real work is big)."""
    from .llmgen import GENERATED
    gen = [r for r in recs if r["source"] in GENERATED]
    real = len(recs) - len(gen)
    allowed = int(real * share / (1 - share)) if share < 1 else len(gen)
    if len(gen) <= allowed:
        return recs, 0
    by: dict[str, list[dict]] = {}
    for r in gen:
        by.setdefault(r["source"], []).append(r)
    tiers = [t for t in GEN_TIERS] + [tuple(sorted(set(by) - {s for t in GEN_TIERS for s in t}))]
    drop, left = set(), allowed
    for tier in tiers:
        items_t = [r for s in tier for r in by.get(s, [])]
        if not items_t:
            continue
        frac = min(1.0, left / len(items_t))
        for src in tier:
            items = by.get(src, [])
            items.sort(key=lambda r: int(key("gen", r["function"], r.get("wrong", ""), r["exact"])[:8], 16)
                       * BAND_KEEP[band(r.get("size", 0))])
            cut = int(len(items) * frac)
            drop |= {id(r) for r in items[cut:]}
            left -= cut
        left = max(0, left)
    return [r for r in recs if id(r) not in drop], len(drop)


DEVICE_FLAGS = ("volatile", "own type", "aliases")     # judgment flags that are devices in a training answer
REPO_ANSWERS = {"base", "history", "grind", "attempts", "regs", "synth"}   # the answer is the repo's exact source


JUDGMENT = ("odd shape", "adds a cast to a raw", "redundant cast to its own type", "macro edits", "declaration edits",
            "new inline helper")


def answer_flags(flags: list[str], repo_answer: bool) -> list[str]:
    """review()'s flags that bar an answer from training: every reject, and the judgment flags that are
    devices. A repo answer has no earlier version to compare with, so a local that copies is not new."""
    if repo_answer:
        flags = [f for f in flags if not re.match(r"new local \w+ (?:only copies|is never read)", f)]
    judgment = [f for f in flags if "volatile" in f or "aliases" in f or f.startswith(JUDGMENT)]
    return [f for f in flags if f not in judgment] + [f for f in judgment if any(w in f for w in DEVICE_FLAGS)]


_INT_TEST = re.compile(r"\(\s*(?:int|long|u32|s32|unsigned(?: int| long)?)\s*\)\s*"
                       r"((?:this->)?[A-Za-z_]\w*(?:(?:->|\.)[A-Za-z_]\w*)*)\s*[!=]=\s*0\b")
_M2C_NAME = re.compile(r"\b(?:var|temp|saved|arg)_[rf]\d{1,2}\w*\b|\b(?:var|temp)_(?:sp|r)[0-9A-Fa-f]+\b")
_REG_LOCAL = re.compile(r"(?:[\w>*&]\s+\**|\*)\s*([rf](?:[12]?\d|3[01]))\s*(?:[;=,)\[])")
_BYTE_PTR = re.compile(r"\(\s*(?:const\s+)?(?:char|u8|s8|unsigned char|signed char)\s*\*\s*\)\s*\(?\s*&?\s*"
                       r"[\w.>\[\]-]+\s*\)?\s*\+\s*([^;]*?)\)")
_K = r"(?:0x[0-9A-Fa-f]+|\d+)"
_INDEX_TIMES_K = re.compile(rf"\(?\s*(?:[A-Za-z_][\w.>\[\]-]*\s*\*\s*{_K}|{_K}\s*\*\s*[A-Za-z_][\w.>\[\]-]*)\s*\)?")
_ASM_BLOCK = re.compile(r"\basm\s*\{[^}]*\}")


def answer_debts(code: str, decls: str = "") -> list[str]:
    """Typing debt the judge doesn't see, in a training answer (2026-11-02 spot check): a pointer tested
    through an int cast (`(int)p == 0`), m2c's or a register's names for locals (`var_r31`, `f31`), and an
    indexed array written as a byte offset (`(char*)&a + i * 0x5C`)."""
    out = []
    for m in _INT_TEST.finditer(code):
        name = re.split(r"->|\.", m[1])[-1]
        if re.search(rf"\*\s*(?:const\s+)?{re.escape(name)}\s*(?:[;,)=\[]|$)", code + "\n" + decls, re.M):
            out.append(f"pointer tested as an int: {m[0].strip()}")
            break
    m = _M2C_NAME.search(code)
    if m:
        out.append(f"m2c-style name {m[0]}")
    m = _REG_LOCAL.search(_ASM_BLOCK.sub("", code))
    if m:
        out.append(f"local named after a register: {m[1]}")
    for m in _BYTE_PTR.finditer(code):
        added = m[1].strip()                          # i * 0x5C, or a local set to one (not byte arithmetic)
        if _INDEX_TIMES_K.fullmatch(added) or (re.fullmatch(r"\w+", added) and re.search(
                rf"\b{re.escape(added)}\s*=\s*{_INDEX_TIMES_K.pattern}\s*;", code)):
            out.append("index x stride as a byte offset")
            break
    return out


class AnswerJudge:
    """llm-review's rules on every answer (the definition alone, beside stub declarations of the unit's
    types so it parses as in its file); memoized per (unit, answer). Drops rejects, volatile, redundant
    casts to the operand's own type, aliasing locals, and answer_debts' typing debt."""

    def __init__(self, cfg: Config):
        from .judge import forbidden_rules
        self.cfg, self.forbidden, self.types, self.memo = cfg, forbidden_rules(cfg), {}, {}

    def stub(self, unit: str) -> str:
        if unit not in self.types:
            from .matching import resolve_unit
            try:
                u = resolve_unit(self.cfg, unit)
                src = (self.cfg.root / u["metadata"]["source_path"]).read_text(errors="replace")
                from ...core import cbody
                names = sorted(n for n in cbody.type_names(src) if re.fullmatch(r"[A-Za-z_]\w*", n))
            except Exception:
                names = []
            self.types[unit] = "".join(f"struct {n};\n" for n in names)
        return self.types[unit]

    def __call__(self, rec: dict) -> list[str]:
        """Why the answer may not be trained on ([] = clean)."""
        k = (rec["unit"], rec["function"], norm(rec["exact"]))
        if k not in self.memo:
            from .judge import REJECT, review
            stub = self.stub(rec["unit"])
            cat, flags = review(stub, stub + rec["exact"].strip() + "\n", rec["function"], self.forbidden)
            if cat == REJECT and any("no single definition" in f for f in flags):
                from .judge import _OFFSET_CAST, source_reasons
                flags = source_reasons(stub, stub + rec["exact"], self.forbidden)    # name not found: gates only
                flags += ["hard-coded offset cast"] if _OFFSET_CAST.search(rec["exact"]) else []
                cat = REJECT if flags else "clean"
            bad = answer_flags(flags, rec["source"] in REPO_ANSWERS) + answer_debts(rec["exact"], rec.get("decls", ""))
            self.memo[k] = bad
        return self.memo[k]


def tags_of(rec: dict) -> dict:
    return {"unit": rec["unit"], "function": rec["function"], "size": rec.get("size", 0), "asm_shape": rec.get("asm_shape"),
            "mutant_shape": rec.get("mutant_shape", ""), "compiler": rec.get("compiler", ""), "flags": rec.get("flags", ""),
            "kind": "repair" if rec.get("wrong") else "direct", "wrong_percent": rec.get("wrong_percent", 0.0),
            "diff_class": rec.get("diff_class", ""), "provenance": rec.get("provenance", {}),
            "record_prompt_version": rec.get("prompt_version")}


def mix(structured: list[dict], legacy_base: list[dict], never: tuple[set, set, set], pats: list[re.Pattern],
        max_tokens: int, cpt: float, upsample: bool, build=None, judge=None, per_fn: int = 0,
        max_generated: float = 1.0, version: int = 0, quarantine: set[str] | None = None,
        examples=None, weights: dict | None = None, max_generated_share: float = 1.0,
        extras: Round2 | None = None) -> tuple[list[dict], dict]:
    """Structured records (+ pre-rendered legacy base pairs) -> rendered rows in curriculum order and the
    card counts. Order of filters: devices in the input (SFT never), judge on the answer, leakage,
    duplicates, window, per-function cap, generated share; then grading, stages and weights (source
    weight x difficulty), with the generated sources' weights scaled down when their share of the trained
    tokens is over `max_generated_share`."""
    weights = weights or WEIGHTS
    from .llmgen import GENERATED, RULE_CAPS, cap_rule_share, clean_answer, literal_labels, synth_artifact
    units, fns, shapes = never
    card = {"sources": {}, "dropped": {"duplicate": 0, "too long": 0, "judge": 0, "device attempt": 0,
                                       "per-function cap": 0, "generated cap": 0, "synth artifact": 0,
                                       "synth rule cap": 0}, "leakage": {}, "judge_reasons": {}, "synth_artifacts": {}}
    keep, seen = [], set()
    card["dropped"]["quarantined"] = 0
    for rec in structured:
        if quarantine and rec["function"] in quarantine:
            card["dropped"]["quarantined"] += 1        # the judge audit flags its exact source: until reviewed
            continue
        if rec.get("devices"):
            card["dropped"]["device attempt"] += 1   # a DPO negative, never an SFT input
            continue
        if rec["source"] == "synth" and (art := synth_artifact(rec)):
            card["dropped"]["synth artifact"] += 1   # a wrong source nobody writes (fixed in rewrites.py since)
            card["synth_artifacts"][art] = card["synth_artifacts"].get(art, 0) + 1
            continue
        if rec["source"] == "backtrans":
            rec = {**rec, "exact": clean_answer(rec["exact"]), "m2c": literal_labels(rec.get("m2c", ""), rec["exact"])}
        bad = judge(rec) if judge else []
        if bad:
            card["dropped"]["judge"] += 1
            for b in bad[:2]:
                b = re.sub(r"\d+", "N", b)
                card["judge_reasons"][b] = card["judge_reasons"].get(b, 0) + 1
            continue
        t = tags_of(rec)
        why = leaks({"messages": [], "tags": t}, units, fns, shapes, [])
        if why:
            card["leakage"][why] = card["leakage"].get(why, 0) + 1
            continue
        dk_ = key(rec["unit"], rec["function"], norm(rec.get("wrong", "")), norm(rec["exact"]))
        if dk_ in seen:
            card["dropped"]["duplicate"] += 1
            continue
        seen.add(dk_)
        keep.append(rec)
    syn = [r for r in keep if r["source"] == "synth"]
    if syn:                                           # statement order at most RULE_CAPS' share of synth
        kept, card["dropped"]["synth rule cap"] = cap_rule_share(syn, 1.0, RULE_CAPS)
        ids = {id(r) for r in kept}
        keep = [r for r in keep if r["source"] != "synth" or id(r) in ids]
    if per_fn:
        keep, n = cap_per_function(keep, per_fn, per_fn * 3 // 2)
        card["dropped"]["per-function cap"] = n
    keep, n = cap_generated(keep, max_generated)
    card["dropped"]["generated cap"] = n
    views = {(r["unit"], r["function"]): r["views"] for r in structured if r.get("views")}
    card["missing_views"] = 0
    rows = []
    for rec in keep:
        v = views.get((rec["unit"], rec["function"]), {}).get(str(view_version(version))) if version else None
        card["missing_views"] += bool(version) and v is None and rec["source"] != "backtrans"
        ex, n_ex = examples(rec) if examples else ("", 0)
        r = extras.render(rec, build, v, ex) if extras else render(rec, build, v, ex)
        r["tags"] = {**tags_of(rec), "source": rec["source"], "difficulty_score": difficulty(rec), "examples": n_ex,
                     "example_tokens": round(len(ex) / cpt)}
        if extras:
            r["tags"]["annotated"] = r.pop("annotated")
            r["tags"]["diag"] = r["messages"][-1]["content"].startswith("DIAG:")
        rows.append(r)
    for b in legacy_base:                             # pre-rendered (llm-bench dataset): the prompt it was made with
        if quarantine and b.get("tags", {}).get("function") in quarantine:
            card["dropped"]["quarantined"] += 1
            continue
        r = {"messages": b["messages"], "tags": {**b.get("tags", {}), "source": "base", "kind": "direct", "legacy": True}}
        r["tags"]["difficulty_score"] = difficulty({"asm": b["messages"][1]["content"], "exact": b["messages"][-1]["content"]})
        dk_ = key(r["tags"].get("unit", ""), r["tags"].get("function", ""), "", norm(b["messages"][-1]["content"]))
        if dk_ in seen:
            card["dropped"]["duplicate"] += 1
            continue
        seen.add(dk_)
        rows.append(r)
    final = []
    for r in rows:                                     # the rendered text: never-train definitions, the window
        why = leaks(r, units, fns, shapes, pats)
        if why:
            card["leakage"][why] = card["leakage"].get(why, 0) + 1
            continue
        toks = round(chars(r["messages"]) / cpt)
        if toks > max_tokens:
            card["dropped"]["too long"] += 1
            continue
        r["tags"]["approx_tokens"] = toks
        final.append(r)
    lv = levels([r["tags"]["difficulty_score"] for r in final])
    for r, level in zip(final, lv):
        t = r["tags"]
        h = key(t["source"], t.get("unit", ""), t.get("function", ""), r["messages"][-1]["content"][:400])
        hv = int(h[:8], 16)
        t["difficulty"] = LEVELS[level]
        t["stage"] = stage_of(level, hv) + 1
        t["sample_weight"] = round(weights.get(t["source"], 1) * LEVEL_WEIGHT[level], 2)
        t["order"] = h
    scale = cap_generated_tokens([r["tags"] for r in final], max_generated_share)
    out = []
    for r, level in zip(final, lv):
        t = r["tags"]
        src = t["source"]
        s = card["sources"].setdefault(src, {"records": 0, "tokens": 0, "answer_tokens": 0, "weight": weights.get(src, 1),
                                             "generated": src in GENERATED, "effective_tokens": 0,
                                             "by_difficulty": {x: 0 for x in LEVELS}, "by_size": {b: 0 for _t, b in BANDS}})
        s["by_size"][band(t.get("size") or 0)] += 1
        s["records"] += 1
        s["tokens"] += t["approx_tokens"]
        s["effective_tokens"] += t["approx_tokens"] * t["sample_weight"]
        s["answer_tokens"] += round(len(r["messages"][-1]["content"]) / cpt)
        s["by_difficulty"][LEVELS[level]] += 1
        for rep in range(max(1, round(t["sample_weight"])) if upsample else 1):
            out.append({**r, "tags": {**t, "repeat": rep}} if rep else r)
    eff = sum(s["effective_tokens"] for s in card["sources"].values())
    for s in card["sources"].values():
        s["effective_tokens"] = round(s["effective_tokens"])
        s["effective_share"] = round(s["effective_tokens"] / max(1, eff), 3)
    card["weights"] = dict(weights)
    card["level_weights"] = dict(zip(LEVELS, LEVEL_WEIGHT))
    card["effective_tokens_per_epoch"] = round(eff)
    card["effective_tokens_by_cap"] = {str(c): round(sum(r["tags"]["approx_tokens"] * r["tags"]["sample_weight"] for r in final
                                                         if r["tags"]["approx_tokens"] <= c)) for c in (8192, 16384)}
    card["generated_effective_share"] = round(sum(s["effective_tokens"] for s in card["sources"].values()
                                                  if s["generated"]) / max(1, eff), 3)
    card["max_generated_share"] = max_generated_share
    card["generated_weight_scale"] = scale
    out.sort(key=lambda r: (r["tags"]["stage"], key(r["tags"]["order"], str(r["tags"].get("repeat", 0)))))
    card["records"] = sum(s["records"] for s in card["sources"].values())
    card["generated_share"] = round(sum(s["records"] for s in card["sources"].values() if s["generated"])
                                    / max(1, card["records"]), 3)
    card["rows_written"] = len(out)
    card["with_examples"] = sum(1 for r in final if r["tags"].get("examples"))
    card["example_tokens"] = sum(r["tags"].get("example_tokens", 0) for r in final)
    card["tokens_per_epoch"] = sum(r["tags"]["approx_tokens"] for r in out)
    card["by_difficulty"] = {x: sum(s["by_difficulty"][x] for s in card["sources"].values()) for x in LEVELS}
    card["by_size"] = {b: sum(s["by_size"][b] for s in card["sources"].values()) for _t, b in BANDS}
    card["by_size_and_difficulty"] = {b: {x: 0 for x in LEVELS} for _t, b in BANDS}
    for r in final:
        card["by_size_and_difficulty"][band(r["tags"].get("size") or 0)][r["tags"]["difficulty"]] += 1
    card["stages"] = {}
    for r in out:
        st = card["stages"].setdefault(str(r["tags"]["stage"]), {"rows": 0, "hard_or_above": 0})
        st["rows"] += 1
        st["hard_or_above"] += r["tags"]["difficulty"] in LEVELS[2:]
    return out, card


def parse_weights(text: str) -> dict[str, float]:
    """`--weights src=w,...` over the defaults (WEIGHTS); an unknown source or a negative weight is an error."""
    out = dict(WEIGHTS)
    for part in (x.strip() for x in (text or "").split(",")):
        if not part:
            continue
        src, sep, w = part.partition("=")
        try:
            val = float(w)
        except ValueError:
            val = -1.0
        if not sep or src.strip() not in (*WEIGHTS, *OPT_IN_SOURCES) or val < 0:
            raise ValueError(f"--weights: bad entry {part!r} (SRC=W with W >= 0; sources: "
                             f"{', '.join((*WEIGHTS, *OPT_IN_SOURCES))})")
        out[src.strip()] = val
    return out


def cap_generated_tokens(tags: list[dict], share: float) -> float:
    """Generated sources (llmgen.GENERATED) scaled down, all by the same factor (their relative weights
    stay), when their share of the trained tokens (approx_tokens x sample_weight: what train.py repeats)
    is over `share`. Changes the tags' sample_weight in place; returns the factor (1.0: under the cap)."""
    from .llmgen import GENERATED
    gen = sum(t["approx_tokens"] * t["sample_weight"] for t in tags if t["source"] in GENERATED)
    real = sum(t["approx_tokens"] * t["sample_weight"] for t in tags if t["source"] not in GENERATED)
    if share >= 1 or gen <= 0 or gen <= share * (gen + real):
        return 1.0
    f = share * real / ((1 - share) * gen)
    for t in tags:
        if t["source"] in GENERATED:
            t["sample_weight"] = round(t["sample_weight"] * f, 3)
    return round(f, 4)


def dpo_pairs(structured: list[dict], build, per_fn: int, judge=None, version: int = 0,
              quarantine: set[str] | None = None, examples=None, extras: Round2 | None = None) -> list[dict]:
    """Better-versus-worse pairs, ranked (exact, device-free, percent): the exact answer over the function's
    wrong definitions (device attempts first: a device at 100% must never win), each under the prompt for
    the unwritten function (its base record's prompt when there is one)."""
    by: dict[tuple, list[dict]] = {}
    base_of: dict[tuple, dict] = {}
    for r in structured:
        k = (r["unit"], r["function"])
        if quarantine and r["function"] in quarantine:
            continue
        if r["source"] == "base":
            base_of[k] = r
        elif r.get("wrong"):
            by.setdefault(k, []).append(r)
    out = []
    for k, wrongs in sorted(by.items()):
        exact = wrongs[0]["exact"]
        if judge and judge(wrongs[0]):
            continue
        wrongs.sort(key=lambda r: (not r.get("devices"), r["source"] not in ("attempts", "grind"), -r.get("wrong_percent", 0)))
        seen = set()
        b = base_of.get(k)
        view = b.get("views", {}).get(str(view_version(version))) if b and version else None
        direct = {**(b or {**wrongs[0], "wrong": "", "m2c": ""}), "exact": exact}
        ex = examples(direct, "dpo")[0] if examples else ""
        prompt = (extras.render(direct, build, view, ex, "dpo") if extras else render(direct, build, view, ex))["messages"][:-1]
        for w in wrongs:
            n = norm(w["wrong"])
            if n in seen or len(seen) >= per_fn:
                continue
            seen.add(n)
            out.append({"prompt": prompt, "chosen": [{"role": "assistant", "content": f"```cpp\n{exact.strip()}\n```"}],
                        "rejected": [{"role": "assistant", "content": f"```cpp\n{w['wrong'].strip()}\n```"}],
                        "tags": {"unit": k[0], "function": k[1], "asm_shape": w.get("asm_shape"), "rejected_source": w["source"],
                                 "rank_chosen": [1, 1, 100.0], "rank_rejected": [0, int(not w.get("devices")), w.get("wrong_percent", 0.0)],
                                 "rejected_devices": w.get("devices", [])}})
    return out


def export(cfg: Config, args) -> int:
    from .llmtry import CHARS_PER_TOKEN, asm_entries, asm_shape, never_train
    from .matching import load_objdiff_units
    root = out_dir(cfg, args)
    never = never_train(cfg)
    if getattr(args, "never_review_eval", False):     # round 2, opt-in: the overseer eval's units stay out too
        from .llmreview import eval_never
        never = tuple(a | b for a, b in zip(never, eval_never(cfg)))
    if not never[0]:
        print("error: no benchmark/held-out set found (llm-data/bench.json, heldout.json): refusing to export")
        return USAGE
    try:
        build = builder(args.prompt_version)
    except ValueError as e:
        print(f"error: {e}")
        return USAGE
    from .llmtry import PROMPT_VERSION
    version = args.prompt_version or PROMPT_VERSION
    try:
        weights = parse_weights(args.weights)
    except ValueError as e:
        print(f"error: {e}")
        return USAGE
    pats = never_definitions(cfg, never[1])
    sources = [x for x in (args.sources.split(",") if args.sources else SOURCES)]
    structured = [r for s in sources for r in read_jsonl(root / "records" / f"{s}.jsonl")]
    rows_changed = correct_rows(structured, zero_starts(cfg))     # rows made before relative_rows was fixed
    legacy = []
    if "base" in sources and not (root / "records" / "base.jsonl").is_file():
        base_path = Path(args.base) if args.base else root / "base" / "data.jsonl"
        legacy = read_jsonl(base_path)
        entries = asm_entries(cfg)
        for b in legacy:                              # their asm shape for the leakage check
            t = b.setdefault("tags", {})
            if t.get("function") in entries:
                t["asm_shape"] = asm_shape(entries[t["function"]].text)
        if legacy and version != 5:                   # noqa: the legacy pairs were made with v5
            print(f"error: the base pairs are pre-rendered with prompt v5; run ./dk dataset base for v{version}")
            return USAGE
    judge = AnswerJudge(cfg) if not args.no_judge else None
    from .llmtry import data_dir
    audit = Path(args.quarantine) if args.quarantine else data_dir(cfg) / "judge-audit.json"
    quarantine = quarantined(audit)
    from .llmtry import DEFAULT_EXAMPLES
    k = DEFAULT_EXAMPLES if args.examples is None else args.examples
    examples = Examples(cfg, k, args.no_examples_share) if k > 0 else None
    extras = None
    if version >= 8 or args.diag_share > 0:          # round 2, opt-in: annotated asm (v8), DIAG lines
        annot_path = Path(args.annotations) if args.annotations else root / "records" / "annot.jsonl"
        annot = load_annotations(annot_path) if version >= 8 else None
        if version >= 8 and not annot:
            print(f"error: prompt v8 needs {cfg.rel(annot_path)}: run ./dk dataset annotate first")
            return USAGE
        extras = Round2(annot, args.annotate_share if version >= 8 else 0.0, args.diag_share)
    recs, card = mix(structured, legacy, never, pats, args.max_tokens, CHARS_PER_TOKEN, args.upsample, build, judge,
                     args.per_fn_cap, args.max_generated, version, quarantine, examples, weights,
                     args.max_generated_share, extras)
    card["retrieval"] = {"examples": k, "no_examples_share": args.no_examples_share if k > 0 else 1.0,
                         "leave_one_out": True, "index": "llm-try's (HEAD, benchmark and held-out units and shapes left out)"}
    card["rows_fixed"] = rows_changed
    card["quarantine"] = {"file": str(audit), "present": quarantine is not None, "functions": len(quarantine or ())}
    if extras:
        card["round2"] = dict(extras.card)
    if getattr(args, "with_review", False):          # LoRA 1 is joint: + the overseer's rows (llmreview.py)
        from .llmreview import join_export
        try:
            recs, card = join_export(cfg, root, never, recs, card, args.review_share, args.max_tokens)
        except ValueError as e:
            print(f"error: --with-review: {e}")
            return USAGE
    dpo = dpo_pairs(structured, build, args.dpo_per_fn, judge, version, quarantine, examples, extras)
    dpo = [d for d in dpo if not leaks({"messages": d["prompt"] + d["chosen"] + d["rejected"], "tags": d["tags"]}, *never, pats)]
    dest = root / (args.name or "mixed")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    (dest / "dpo.jsonl").write_text("".join(json.dumps(d) + "\n" for d in dpo))
    # independent re-check of what was written (the gate): every pattern on its own, every tag
    written = read_jsonl(dest / "train.jsonl")
    single = never_definitions(cfg, never[1])
    bad = sum(1 for r in written if leaks(r, *never, single))
    if getattr(args, "with_review", False):          # review rows never meet the overseer eval either
        from .llmreview import EVAL_FILE, joint_recheck
        bad += joint_recheck(written, data_dir(cfg) / EVAL_FILE)
    bad_dpo = sum(1 for d in read_jsonl(dest / "dpo.jsonl")
                  if leaks({"messages": d["prompt"] + d["chosen"] + d["rejected"], "tags": d["tags"]}, *never, single))
    card.update({"name": dest.name, "about": "dk dataset export: structured records rendered with llm-try's prompt builder "
                 "(repairs: current/best definition + diff -> exact; direct: the unwritten-function prompt -> source), "
                 "judged, leakage-checked, graded easy -> extreme, in curriculum order (tags.stage)",
                 "never_commit": True, "prompt_version": version, "dpo_pairs": len(dpo),
                 "head": _git(cfg, "rev-parse", "HEAD").strip(), "made": time.strftime("%Y-%m-%d %H:%M"),
                 "chars_per_token": CHARS_PER_TOKEN, "max_tokens": args.max_tokens, "upsampled": args.upsample,
                 "per_function_cap": args.per_fn_cap, "max_generated": args.max_generated,
                 "never_train": {"units": len(never[0]), "functions": len(never[1]), "asm_shapes": len(never[2])},
                 "leakage_check": {"passed": bad == 0 and bad_dpo == 0, "leaking_rows_written": bad, "leaking_dpo_pairs": bad_dpo,
                                   "checks": "unit, function, asm shape (also each mutant's own), never-train definitions in the text"},
                 "base": "structured (records/base.jsonl)" if not legacy else f"legacy pre-rendered v5 ({len(legacy)})",
                 "units_known": len(load_objdiff_units(cfg))})
    (dest / "dataset.json").write_text(json.dumps(card, indent=1) + "\n")
    print(f"{card['records']} records ({card['rows_written']} rows) into {cfg.rel(dest)}/train.jsonl, prompt v{version}, "
          f"~{card['tokens_per_epoch']} tokens per epoch; generated share {card['generated_share']:.0%}")
    for s, v in sorted(card["sources"].items(), key=lambda kv: KEEP_ORDER.get(kv[0], 99)):
        print(f"  {s:9} {v['records']:6d} records  ~{v['tokens']:>10d} tokens  weight {v['weight']}  "
              f"trained ~{v['effective_tokens']:>10d} ({v['effective_share']:.1%})  "
              + " ".join(f"{n[:4]} {c}" for n, c in v["by_difficulty"].items()))
    print(f"  trained tokens per epoch (x source weight x difficulty weight): "
          f"~{card.get('worker_effective_tokens_per_epoch', card['effective_tokens_per_epoch']):,}; "
          f"generated {card['generated_effective_share']:.1%} of them (at most {args.max_generated_share:.0%})")
    if card["generated_weight_scale"] < 1:
        print(f"  warning: generated sources were {card['generated_weight_scale']:.3f}x'd to stay at "
              f"{args.max_generated_share:.0%} of the trained tokens (--max-generated-share; add real records or lower their weights)")
    print(f"  retrieval: {k} examples on {card['with_examples']} of {card['records']} records "
          f"(~{card['example_tokens']} tokens of examples)")
    print(f"  difficulty: {card['by_difficulty']}")
    print(f"  size bands: {card['by_size']}")
    print(f"  quarantine ({cfg.rel(audit)}): " + (f"{len(quarantine)} functions, {card['dropped']['quarantined']} records "
          "dropped" if quarantine is not None else "no audit file: nothing quarantined"))
    print(f"  dropped: {card['dropped']}; judge: {dict(sorted(card['judge_reasons'].items(), key=lambda kv: -kv[1])[:5])}")
    if extras:
        r2 = card["round2"]
        print(f"  round 2: {r2['annotated']} records with annotated asm (share {r2['annotate_share']}, "
              f"{r2['no_annotation']} picked had none); {r2['diag']} repairs with a DIAG line (share {r2['diag_share']})")
    if card.get("tasks"):
        from .llmreview import print_joint
        print_joint(card)
    print(f"  {len(dpo)} DPO pairs into {cfg.rel(dest)}/dpo.jsonl")
    print(f"  leakage removed: {card['leakage'] or 'none'}; leakage check "
          f"{'PASSED' if card['leakage_check']['passed'] else f'FAILED ({bad} rows, {bad_dpo} pairs)'}")
    return OK if card["leakage_check"]["passed"] else USAGE


def run(cfg: Config, args: argparse.Namespace) -> int:
    from .jobs import resolve
    args.jobs = min(resolve(getattr(args, "jobs", 0), cfg), 6) if args.action != "export" else 1
    if args.action in ("history", "regs", "export"):
        return {"history": history, "regs": regs_cmd, "export": export}[args.action](cfg, args)
    if args.action == "annotate":                     # prompt v8's asm comments (round 2): llmannot.py
        from . import llmannot
        return llmannot.run_dataset(cfg, args)
    if args.action == "review":                       # the overseer task (round 2): llmreview.py
        from . import llmreview
        return llmreview.run_dataset(cfg, args)
    if args.action == "leakcheck":                   # the leakage filter over pairs of the existing datasets (gate.py)
        from . import gate
        return gate.leakcheck(cfg, args)
    if args.action == "fixes":                       # shape-fix records from landing commits (llmfixes.py)
        from . import llmfixes
        return llmfixes.run(cfg, args)
    if args.action == "shape-synth":                 # synthetic shape-fix pairs + policy examples (shapesynth.py)
        from . import shapesynth
        return shapesynth.run(cfg, args)
    if args.action == "stacked":                     # multi-change puzzles with score paths (stacked.py)
        from . import stacked
        return stacked.run(cfg, args)
    if args.action == "real-nbhd":                   # enumerator census + real-neighbourhood pairs (realnbhd.py)
        from . import realnbhd
        return realnbhd.run(cfg, args)
    if args.action == "lessons":                     # generic compiler-verified pairs (lessons.py)
        from . import lessons
        return lessons.run(cfg, args)
    from . import llmgen
    return llmgen.run(cfg, args)


def _configure(p: argparse.ArgumentParser) -> None:
    sub = p.add_subparsers(dest="action", required=True)
    for name, helptext in (("history", "earlier committed versions of each exact function, re-scored -> repair records"),
                           ("regs", "declaration-order variants of each exact function that change its asm -> repair records")):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--per-fn", type=int, default=4 if name == "history" else 3,
                       help="versions per function (history: the newest distinct ones; default 4/3)")
        s.add_argument("--limit", type=int, default=0, help="score at most N (history: the most recent first)")
        s.add_argument("--only", default="", help="units whose name contains one of these (comma-separated)")
        s.add_argument("--min-size", type=lambda x: int(x, 0), default=0x10, help="skip smaller functions (default 0x10)")
        s.add_argument("--jobs", type=int, default=0, help="parallel compiles (default the idle cores, at most 6)")
        s.add_argument("--out", default="", help="dataset directory (default llm-data/dataset)")
        if name == "history":
            s.add_argument("--commits", type=int, default=40, help="newest commits per unit source to read (default 40)")
            s.add_argument("--per-fn-big", type=int, default=8, help="versions per function of 0x300 bytes or more (default 8)")
            s.add_argument("--no-recover", dest="recover", action="store_false", help="no second try for a version that "
                           "misses identifiers (default: its own old declarations of them are added before it)")
    from . import lessons, llmannot, llmfixes, llmgen, llmreview, realnbhd, shapesynth, stacked
    llmfixes.configure(sub)
    shapesynth.configure(sub)
    stacked.configure(sub)
    realnbhd.configure(sub)
    llmgen.configure(sub)
    llmreview.configure(sub)
    lessons.configure(sub)
    llmannot.configure(sub)
    e = sub.add_parser("export", help="render every record with llm-try's prompt builder, judge the answers, check "
                       "leakage, grade and order easy -> extreme -> llm-data/dataset/mixed/{train,dpo}.jsonl + dataset.json")
    e.add_argument("--prompt-version", type=int, default=0, help="llm-try prompt version to render with (default the current)")
    e.add_argument("--examples", type=int, default=None, help="retrieval examples per prompt, as llm-try adds them "
                   "(default llm-try's default; 0: none), never the record's own function")
    e.add_argument("--no-examples-share", type=float, default=0.18, help="share of the records rendered with no "
                   "examples (default 0.18: robust when retrieval finds nothing good)")
    e.add_argument("--sources", default="", help=f"record sources to use (default all: {','.join(SOURCES)})")
    e.add_argument("--base", default="", help="legacy pre-rendered base pairs when records/base.jsonl is missing")
    e.add_argument("--max-tokens", type=int, default=16384, help="drop records longer than this (default 16384)")
    e.add_argument("--per-fn-cap", type=int, default=8, help="records per function at most (default 8, half as many again from 0x300 bytes; 0 = no cap)")
    e.add_argument("--max-generated", type=float, default=0.65, help="generated records' share at most (default 0.65)")
    e.add_argument("--dpo-per-fn", type=int, default=3, help="DPO pairs per function (default 3)")
    e.add_argument("--no-judge", action="store_true", help="skip llm-review's rules on the answers (a test)")
    e.add_argument("--quarantine", default="", help="judge audit of flagged exact functions, left out of every source "
                   "(default llm-data/judge-audit.json when it exists)")
    e.add_argument("--weights", default="", help="source weights SRC=W,... over the defaults "
                   f"({','.join(f'{k}={v}' for k, v in WEIGHTS.items())}); x difficulty (hard ones up) = each record's "
                   "sample_weight, which train.py applies as repeats")
    e.add_argument("--max-generated-share", type=float, default=MAX_GENERATED_SHARE, help="generated sources' share "
                   f"of the trained tokens at most (default {MAX_GENERATED_SHARE}): over it, their weights are scaled "
                   "down together; 1 = off")
    e.add_argument("--name", default="", help="export into llm-data/dataset/NAME (default mixed)")
    e.add_argument("--upsample", action="store_true", help="repeat each record by its sample weight (source x difficulty)")
    e.add_argument("--never-review-eval", action="store_true", help="round 2: also leave out llm-data/review-eval.json's "
                   "units, functions and asm shapes (the overseer eval), so the joint LoRA never sees them (default off)")
    e.add_argument("--annotate-share", type=float, default=0.75, help="prompt v8 (round 2): share of the records "
                   "rendered with the annotated asm, the rest without (default 0.75)")
    e.add_argument("--annotations", default="", help="prompt v8: dataset annotate's file (default records/annot.jsonl)")
    e.add_argument("--diag-share", type=float, default=0.0, help="round 2: share of the repair answers that open with "
                   "llmdiag's one-line DIAG (default 0: none; llm-try --diag prefills it)")
    e.add_argument("--with-review", action="store_true", help="LoRA 1 is joint: append the overseer's rendered "
                   "review rows (records/review.jsonl's train split, review prompt r1, tags.task=review, weight 1; "
                   "review-dev/review-final never) to train.jsonl; every worker row stays (default off)")
    e.add_argument("--review-share", type=float, default=0.05, help="with --with-review: review rows' share of the "
                   "trained tokens at most (default 0.05; people's labels and real attempts kept first)")
    e.add_argument("--out", default="", help="dataset directory (default llm-data/dataset)")


COMMAND = Command("dataset", "matching", "training records (git history, declaration orders, model attempts, mutants, "
                  "renames, back-translation) and a graded, leakage-checked chat JSONL export", run, _configure)
