"""`dk llm-review --audit`: the pre-review rules over the current source of every exact function.

Two uses: the judge's false-positive rate on accepted code (each function is reviewed as if it
were a candidate written into an empty definition), and the list of answers a training export
must quarantine so a model never learns a device from our own older code. Writes
llm-data/judge-audit.json (function names and flags only, no code).

Known exceptions are kept apart, not hidden: `volatile` rows <docs project>/faithfulness.md calls
(likely) original, a file-level pragma block before the unit's first function, and
`force_active` (data only). The triage table in <docs project>/faithfulness-audit.md gives a verdict per
(unit, function, rule); `quarantine` is every function with a flag it does not clear. Same-type casts are found per unit in one compile: every cast gets a
probe that fails only when its operand already has the cast's type, and failing groups are
halved until each hit is confirmed with judge.same_type_probe and scored without the cast.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ...core import cbody
from . import judge, llmtry

NE = ("template <class DkA, class DkB> struct DkNotSame { typedef int t; };\n"
      "template <class DkA> struct DkNotSame<DkA, DkA> {};\n")

RULES = [                       # (rule, flag pattern, reject?)
    ("volatile", r"^adds \d+ volatile", False),
    ("pragma in the definition", r"^adds \d+ pragma", True),
    ("asm", r"^adds \d+ asm", True),
    ("__typeof__", r"^adds \d+ __typeof__", True),
    ("padding", r"^adds \d+ padding", True),
    ("register keyword", r"^adds \d+ register", True),
    ("forbidden quality rule", r"^adds \d+ forbidden", True),
    ("unused local", r"^new local \w+ is never read", True),
    ("copy-only local", r"^new local \w+ only copies", True),
    ("placeholder-named local", r"^new local \w+ has a placeholder name", True),
    ("hand-copied inline body", r"^hand-copied inline body", False),
    ("inline attribute", r"^adds \d+ inline attribute", True),
    ("aliasing local", r"^new local \w+ aliases", False),
    ("cached member or global read", r"^new local \w+ only caches", False),
    ("same-type cast that changes the code", r"^cast to its own type changes the code", True),
    ("same-type cast without effect", r"^redundant cast to its own type", False),
    ("cast check could not run", r"^cast check could not run", False),
    ("hard-coded offset cast", r"^hard-coded offset cast", True),
    ("byte-pointer arithmetic with a constant", r"^byte-pointer arithmetic", False),
    ("raw pointer cast", r"^adds a cast to a raw pointer type", False),
    ("empty statement", r"^adds an empty statement", True),
    ("comma operator", r"^adds a comma operator", True),
    ("aligned static", r"^adds \d+ aligned static", True),
    ("aligned stack local, open (faithfulness.md)", r"^adds \d+ aligned stack local", False),
    ("empty case labels", r"^empty case labels", False),
    ("goto", r"^odd shape: adds \d+ goto", False),
    ("do-while(0)", r"^odd shape: adds \d+ do-while", False),
    ("function-level pragma", r"^compiled under #pragma", True),
    ("struct packing pragma", r"^struct packing #pragma", False),
    ("file-level opt pragma, likely original (faithfulness.md)", r"^file-level #pragma", False),
    ("volatile, likely original (faithfulness.md)", r"^volatile, likely original", False),
    ("preprocessor line in the definition", r"^adds a preprocessor line inside", True),     # RL1 in-span detectors
    ("constant-false condition", r"^adds a constant-false condition", True),
    ("self-assignment", r"^adds a self-assignment", True),
    ("identity arithmetic", r"^adds identity arithmetic", True),
    ("(void)0", r"^adds a \(void\)0", True),
    ("double negation", r"^adds a double negation", False),
    ("local type with methods", r"^adds a local struct or class with methods", False),
    ("address pun", r"^adds a pointer cast of an address", False),
    ("integer cast of a variable", r"^adds an integer cast", False),
]
TRIAGED = "triaged allowed or false positive (faithfulness-audit.md)"
KNOWN = {"file-level opt pragma, likely original (faithfulness.md)", "volatile, likely original (faithfulness.md)", TRIAGED,
         "aligned stack local, open (faithfulness.md)"}
RELEASED = {"raw pointer cast",     # emits no code; a 20-function sample held typing debt, not devices
            "address pun",          # RL1: 144 exact functions have one (typing debt like the raw casts)
            "preprocessor line in the definition"}   # RL1: 72 exact ones, conditional compilation in ported SDK code
TRIAGE_DOC = "faithfulness-audit.md"       # in [docs] project
CLEARED = ("allowed", "false positive")         # triage verdicts that keep a function in training
LIKELY_ORIGINAL = "likely-original"             # only known exceptions: fine as a training answer
OPT_PRAGMAS = ("optimization_level", "optimize_for_size", "scheduling", "peephole")
DATA_PRAGMAS = ("force_active", "explicit_zero_data", "once", "section")   # data or include guards: not code


def triage_rows(doc: str) -> dict[tuple[str, str, str], str]:
    """(unit, function, rule) -> verdict from the triage table (<docs project>/faithfulness-audit.md):
    `device`, `allowed`, `false positive` or `open`."""
    out = {}
    for row in doc.splitlines():
        cells = [c.strip().strip("`") for c in row.strip().strip("|").split("|")]
        if len(cells) < 4 or not row.lstrip().startswith("|") or set(cells[0]) <= set("-: ") or cells[0] == "Unit":
            continue
        verdict = next((v for v in ("device", "allowed", "false positive", "open")
                        if cells[3].lower().strip("* ").startswith(v)), None)
        if verdict is not None:
            for rule in cells[2].split(","):
                out[(cells[0], cells[1], rule.strip())] = verdict
    return out


def apply_triage(r: dict, triage: dict) -> None:
    """Fill r's `triage` (rule -> verdict), `category` and `quarantine`: a function stays in training
    when every flag is a known exception, a released rule, or triaged allowed / false positive."""
    rules = [rule_of(fl) for fl in r["flags"]]
    verdicts = {n: triage[(r["unit"], r["function"], n)] for n, _ in rules if (r["unit"], r["function"], n) in triage}
    if verdicts:
        r["triage"] = verdicts
    cleared = [(TRIAGED, False) if verdicts.get(n) in CLEARED else (n, rej) for n, rej in rules]
    r["category"] = category_of(cleared)
    r["quarantine"] = any(n not in KNOWN | RELEASED for n, _ in cleared)


def category_of(rules: list[tuple[str, bool]]) -> str:
    """reject > needs judgment > likely original (only known exceptions) > clean."""
    if any(rej for _, rej in rules):
        return judge.REJECT
    if any(n not in KNOWN for n, _ in rules):
        return judge.JUDGMENT
    return LIKELY_ORIGINAL if rules else judge.CLEAN


def rule_of(flag: str) -> tuple[str, bool]:
    for name, pat, rej in RULES:
        if re.match(pat, flag):
            return name, rej
    return "other", True


# ---- pragmas ---------------------------------------------------------------------------------------

_PRAGMA = re.compile(r"^[ \t]*#[ \t]*pragma[ \t]+(\w+)[ \t]*(.*?)[ \t]*$", re.M)


def pragmas_at(src: str, offset: int) -> list[tuple[str, str, int]]:
    """The pragmas in force at `offset`: (name, value, offset of the line), `reset`, `pop` and the
    opposite switch (`dont_inline on` ... `off`) undoing them, data-only ones (`force_active`,
    `explicit_zero_data`) left out."""
    state: dict[str, tuple[str, int]] = {}
    stack: list[dict] = []
    for m in _PRAGMA.finditer(src, 0, offset):
        name, value = m[1], m[2]
        if name == "push":
            stack.append(dict(state))
        elif name == "pop":
            state = stack.pop() if stack else {}
        elif value.split()[:1] == ["reset"] or (name in state and {state[name][0], value} == {"on", "off"}):
            state.pop(name, None)                   # `on` ... `off` closes the region like `reset`
        elif name not in DATA_PRAGMAS:
            state[name] = (value, m.start())
    return [(n, v, at) for n, (v, at) in state.items()]


# ---- faithfulness.md's volatile rows ------------------------------------------------------------------

def volatile_rows(doc: str) -> list[tuple[str, set[int] | None, bool]]:
    """(path, lines or None, likely original?) for each file the volatile table names."""
    out = []
    part = doc.split("## `volatile`", 1)[-1].split("\n## ", 1)[0]
    for row in part.splitlines():
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        if len(cells) < 3 or set(cells[0]) <= set("-: "):
            continue
        original = re.search(r"\boriginal\b", cells[-1]) is not None and "device" not in cells[-1].split(";")[0]
        for m in re.finditer(r"([\w/]+\.(?:cpp|c|h))((?::\d+(?:[-,]\d+)*)?)", cells[0]):
            lines = None
            if m[2]:
                lines = set()
                for part2 in m[2][1:].split(","):
                    a, _, b = part2.partition("-")
                    lines |= set(range(int(a), int(b or a) + 1))
            out.append((m[1], lines, original))
    return out


def volatile_original(rows, path: str, lines: list[int]) -> bool:
    hits = [(ls, orig) for p, ls, orig in rows if path.endswith(p)]
    exact = [orig for ls, orig in hits if ls and any(x in ls for x in lines)]
    if exact:
        return all(exact)
    loose = [orig for ls, orig in hits if ls is None]
    return bool(loose) and all(loose)


# ---- same-type casts, a unit at a time -------------------------------------------------------------

def _insert(src: str, edits: list[tuple[int, str]]) -> str:
    for at, text in sorted(edits, key=lambda x: x[0], reverse=True):
        src = src[:at] + text + src[at:]
    return src


def _casts(src: str, body: cbody.Stmt) -> list[tuple[cbody.Expr, str, str]]:
    out = []
    for e in judge.added_casts("{}", cbody.parse_body("{}", 0), src, body):
        t = src[e.type_span[0]:e.type_span[1]].strip()
        if t and not re.search(r"[()\[\]<>]", t) and judge._norm(t) not in ("void", "constvoid"):
            out.append((e, t, src[e.kids[0].start:e.kids[0].end]))
    return out


def same_type_casts(src: str, casts: list, compiles, cplusplus: bool) -> list:
    """The casts whose operand already has the cast's type (one compile when there is none)."""
    def ne(group):
        return NE + _insert(src, [x for e, t, op in group for x in
                                  ((e.start, f"(sizeof(DkNotSame<{t}, __typeof__({op})>::t), "), (e.end, ")"))])

    def eq(item):
        e, t, op = item
        return _insert(src, [(e.start, f"(sizeof(*({t}**)0 = (__typeof__({op})*)0), "), (e.end, ")")])
    if not casts:
        return []
    if not cplusplus or not compiles(NE + src):     # C, or a compiler without partial specialization
        return [c for c in casts if compiles(eq(c))]
    out = []

    def split(group):
        if compiles(ne(group)):
            return
        if len(group) == 1:
            if compiles(eq(group[0])):
                out.append(group[0])
            return
        mid = len(group) // 2
        split(group[:mid])
        split(group[mid:])
    split(casts)
    return out


# ---- one unit ----------------------------------------------------------------------------------------

def audit_unit(cfg, u: dict, forbidden, vrows, vendored: tuple[str, ...]) -> list[dict]:
    from .judge import review, unit_probe
    from .reorder import compile_variant
    from .trycmd import definition
    path = u["metadata"].get("source_path", "")
    source = cfg.root / path
    if not path or not source.is_file():
        return []
    src = source.read_text()
    fns = llmtry.exact_functions(cfg, u, src, min_size=0)
    if not fns:
        return []
    work = cfg.root / llmtry.LLM_DIR / "work" / f"audit_{abs(hash(u['name'])) % 10**8}"

    def compiles(text: str) -> bool:
        import shutil
        d = work / "c"
        try:
            return compile_variant(cfg, u, text, d)[0] is not None
        finally:
            shutil.rmtree(d, ignore_errors=True)
    first = min((f.head_start for f in cbody.functions(src)), default=len(src))
    recs, all_casts, owner = [], [], {}
    for f in fns:
        name = f["function"]
        span = definition(src, name)
        if span is None:
            continue
        before = src[:span[0]] + src[span[1]:]
        try:
            category, flags = review(before, src, name, forbidden)
        except Exception as e:                      # an unparsable body: say so
            category, flags = judge.JUDGMENT, [f"could not review: {e}"[:120]]
        body = judge._parse(src, span)
        if body is not None:
            for c in _casts(src, body):
                all_casts.append(c)
                owner[id(c[0])] = name
        lines = [cbody.line_of(src, span[0] + m.start()) + 1 for m in re.finditer(r"\bvolatile\b", src[span[0]:span[1]])]
        out = []
        for fl in flags:
            if re.match(r"^adds \d+ volatile", fl) and volatile_original(vrows, path, lines):
                out.append("volatile, likely original (faithfulness.md)")
            else:
                out.append(fl)
        for pname, value, at in pragmas_at(src, span[0]):
            line = cbody.line_of(src, at) + 1
            if pname == "pack":
                out.append(f"struct packing #pragma pack {value} (line {line})")
            elif at < first and pname in OPT_PRAGMAS:       # a load screen's block (faithfulness.md)
                out.append(f"file-level #pragma {pname} {value} (line {line})")
            else:                                           # dont_inline, auto_inline regions: devices
                out.append(f"compiled under #pragma {pname} {value} (line {line})")
        recs.append({"unit": u["name"], "source": path, "function": name, "size": f["size"],
                     "vendored": path.startswith(vendored) or path in vendored, "flags": out})
    hits = same_type_casts(src, all_casts, compiles, source.suffix != ".c")
    for e, t, op in hits:
        name = owner[id(e)]
        probe = unit_probe(cfg, u, name, work / "p")
        what = f"({' '.join(t.split())}){' '.join(op.split())[:60]}"
        bare = probe(cbody.apply_edits(src, [(e.start, e.end, f"({op})")]))
        flag = (f"cast to its own type changes the code {what}" if bare != 100.0
                else f"redundant cast to its own type {what}")
        next(r for r in recs if r["function"] == name)["flags"].append(flag)
    for r in recs:
        r["rules"] = sorted({rule_of(fl)[0] for fl in r["flags"]})
        r["category"] = category_of([rule_of(fl) for fl in r["flags"]])
    return recs


def _audit_one(job) -> list[dict]:
    try:
        return audit_unit(*job)
    except Exception as e:                          # one bad unit must not stop the audit
        return [{"unit": job[1]["name"], "source": job[1]["metadata"].get("source_path", ""), "function": "",
                 "size": 0, "vendored": False, "flags": [f"could not audit the unit: {e}"[:200]],
                 "rules": ["other"], "category": judge.JUDGMENT}]


def run_audit(cfg, jobs: int = 4, only: str = "", say=print) -> dict:
    from .judge import forbidden_rules
    from .matching import load_objdiff_units
    forbidden = forbidden_rules(cfg)
    doc = cfg.docs.project / "faithfulness.md"
    vrows = volatile_rows(doc.read_text()) if doc.is_file() else []
    vendored = tuple(llmtry._vendored_paths(cfg))
    units = [u for u in load_objdiff_units(cfg) if u.get("base_path") and (not only or only in u["name"])]
    units.sort(key=lambda u: -(cfg.root / u["metadata"].get("source_path", "")).stat().st_size
               if (cfg.root / u["metadata"].get("source_path", "")).is_file() else 0)   # the big ones first
    with ProcessPoolExecutor(max_workers=max(1, jobs)) as pool:     # review() is Python-bound: processes
        recs = [r for rs in pool.map(_audit_one, [(cfg, u, forbidden, vrows, vendored) for u in units]) for r in rs]
    recs.sort(key=lambda r: (r["unit"], r["function"]))
    tdoc = cfg.docs.project / TRIAGE_DOC
    triage = triage_rows(tdoc.read_text()) if tdoc.is_file() else {}
    for r in recs:
        apply_triage(r, triage)
    game = [r for r in recs if not r["vendored"]]
    counts = {}
    for r in recs:
        for name in r["rules"]:
            c = counts.setdefault(name, {"game": 0, "vendored": 0})
            c["vendored" if r["vendored"] else "game"] += 1
    doc_out = {"about": "dk llm-review --audit: the pre-review rules over every exact function's current source "
                        "(reviewed as if written into an empty definition), with the verdicts of "
                        "<docs project>/faithfulness-audit.md applied. 'quarantine' lists the functions a training export "
                        "must leave out: every flag not a known exception (faithfulness.md), a released rule (raw "
                        "pointer casts) or triaged allowed / false positive. No code in this file.",
               "head": llmtry._git(cfg, "rev-parse", "HEAD").strip(), "never_commit": True,
               "functions": len(recs), "game_functions": len(game),
               "categories": {c: {"game": sum(1 for r in game if r["category"] == c),
                                  "vendored": sum(1 for r in recs if r["vendored"] and r["category"] == c)}
                              for c in (judge.REJECT, judge.JUDGMENT, LIKELY_ORIGINAL, judge.CLEAN)},
               "known_exceptions": sorted(KNOWN), "released_rules": sorted(RELEASED),
               "triaged": {v: sum(1 for r in recs for x in r.get("triage", {}).values() if x == v)
                           for v in ("device", "open", "allowed", "false positive")},
               "quarantine": sorted({r["function"] for r in recs if r.get("quarantine") and r["function"]}),
               "rules": dict(sorted(counts.items(), key=lambda kv: -kv[1]["game"])),
               "flagged": [r for r in recs if r["category"] != judge.CLEAN]}
    out = llmtry.data_dir(cfg) / "judge-audit.json"
    out.write_text(json.dumps(doc_out, indent=1) + "\n")
    say(f"{len(recs)} exact functions ({len(game)} game-side) in {len(units)} units: "
        + ", ".join(f"{c} {v['game']}+{v['vendored']}" for c, v in doc_out["categories"].items()) + " (game+vendored)")
    for name, c in doc_out["rules"].items():
        say(f"  {c['game']:5d} {c['vendored']:5d}  {name}")
    say(f"quarantine: {len(doc_out['quarantine'])} functions (devices and anything not cleared); triage rows used: "
        + ", ".join(f"{k} {v}" for k, v in doc_out["triaged"].items()))
    say(f"list: {out}")
    return doc_out
