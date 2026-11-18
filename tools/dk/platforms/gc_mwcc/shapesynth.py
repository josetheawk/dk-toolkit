"""dk dataset shape-synth: synthetic shape-fix pairs and single-step agent-policy examples (LoRA-2 data, plan B.2
and council item 5). Compiler-verified, deterministic; project code and tool output only, never model text.

  ./dk dataset shape-synth [--n 1000] [--per-fn 2] [--jobs 4] [--seed S] [--out DIR] [--until HH:MM] [--fold-draw P]
  ./dk dataset shape-synth --check [--real FIXES.jsonl] [--out DIR]

Pairs: an exact function of HEAD (the leakage filter's sets left out: llmfixes.frozen + the judge-audit quarantine)
gets 1-3 stacked moves of the shape rows of the rewrite table (SHAPE_TABLE; every row is a move and its inverse),
each step checked to be reversible by a table move (the state before it is among the table's candidates of the state
after it), so the forward fix restores the exact source. A pair is kept when the compiler scores its BEFORE below
100 % (a private `-sym on` compile of the unit), the judge's source checks pass for the fix, judge.review rejects
neither direction, no empty case label is added and no case value is invented (a case of the after whose value the
before does not hold). New locals of a move get their real type from the debug info and a name from the value they
hold (no t0/e0 placeholders). Records use `dataset fixes`' schema (before, after, rows, before_percent, diff_class,
fix_kind from llmfixes.fix_kinds, oracle null) with source "synthetic" and the transform ids.

Policy examples (llm-agent's single-step state): the definition, its rows and score line, the last <= 4 (action,
result) pairs; the target is ONE action: the edit (SEARCH/REPLACE, llmagent.fix_hunk) and, when the harness's
structured form reproduces the step, the transform and anchor. Steps are emitted while the forward path keeps the
harness's greedy rule (each step no worse). Wrong-edit turns: other table moves at a state that compile and score
worse (grid losers) go into the history, reverted, before the right edit. STOP examples (--stop-share of the policy
examples): real dead ends, open functions where `close --queue` found nothing better with the table, measured at
HEAD, their losers as history, label register-only (register diff) or no-idea.

Output: llm-data/datasets/shape-synth/v0/{pairs.jsonl, policy.jsonl, CARD.md, SHA256SUMS}; cache/ holds each
function's result (re-runs and larger --n reuse it). `--check`: the pre-registered distribution check (T35) of the
pairs against the real shape-fix records: total-variation distance of the fix-kind and diff-class histograms and the
share of real records whose every fix kind has >= 200 synthetic examples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import time
from pathlib import Path

from ...core import cbody
from ...core.command import OK, USAGE
from ...core.config import Config

SCHEMA = 1
SHAPE_TABLE = ("else-return", "name-value", "inline-local", "local-reuse", "default-else", "unsigned-compare",
               "accessor", "shared-labels", "one-case-switch", "or-return", "if-invert", "ternary-if", "flag-form",
               "assign-in-condition", "unsigned-index", "ref-element", "op-assign", "half", "chained-assign",
               "split-init")
# left out: decl-order, operand-swap, compare-flip, reassociate, statement-order (order/register mutants, §14b),
# cast-call and compare-cast (they add devices: llmgen.SYNTH_RULES), the inline-helper rows (new functions)
DEPTHS = ((1, 40), (2, 35), (3, 25))
PER_RULE = 8                       # candidates looked at per rule and state
REVERSE_PER_RULE = 400             # the reversibility check looks at every place a naming move offers
# the moves that fold a local away (their inverse names a value: the fix adds a local)
FOLDS = (("inline-local", "written out at its"), ("name-value", "folded into its use"))
FOLD_MARK = dict(FOLDS)
FOLD_DRAW = 0.30                   # share of fold draws (T38): the real share of local-added records, 8/27
FOLD_TRIES = 4                     # fold-side moves tried for a fold draw's first step
HISTORY = 4
LOSER_SHARE = 0.4                  # states that get 1-2 grid losers in their history
STOP_LABELS = ("register-only", "no-idea")
MIN_COVER = 200
PASS_TV, PASS_COVER = 0.3, 0.70
_CASE = re.compile(r"\bcase\s+([^:]+?)\s*:")
_NEW = re.compile(r"\b[te]\d+\b")
_ID = re.compile(r"[A-Za-z_]\w*")
_PLACEHOLDER = re.compile(r"(?:t|tmp|temp|var|v|val|unk)\d+|tmp|temp", re.I)
_KEYWORDS = {"if", "else", "for", "while", "do", "switch", "case", "return", "break", "continue", "sizeof", "this",
             "int", "float", "char", "short", "long", "unsigned", "signed", "bool", "void", "const", "struct", "true",
             "false", "new", "delete", "static", "double", "goto", "default"}


def hkey(*parts) -> int:
    return int(hashlib.sha1("\0".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def sha(*parts: str) -> str:
    return hashlib.sha1("\0".join(parts).encode()).hexdigest()


def norm(code: str) -> str:
    from .llmdata import norm as n
    return n(code)


# ---- pure helpers (tested) ---------------------------------------------------------------------------------

def case_values(code: str) -> set[str]:
    return {" ".join(v.split()) for v in _CASE.findall(re.sub(r"//[^\n]*|/\*.*?\*/", " ", code, flags=re.S))}


def invented_cases(before: str, after: str) -> list[str]:
    """Case values of the fix's result that the before holds neither as a case nor as a token: a device."""
    have = case_values(before)
    toks = set(re.findall(r"\b\w+\b|'(?:[^'\\]|\\.)'", before))
    return sorted(v for v in case_values(after) - have if v not in toks and not set(re.findall(r"\b\w+\b", v)) <= toks)


def natural_name(expr: str, type_text: str, taken: set[str]) -> str:
    """A name for a new local from the value it holds (`car->mTires[i]` -> tires, `GetSpeed()` -> speed),
    else from its type; never a placeholder (t0, tmp) and never one the file or class already uses."""
    cands = []
    for ident in reversed(_ID.findall(expr)):
        if ident in _KEYWORDS or re.fullmatch(r"(?:fn|lbl|vf)_\w+|m[0-9A-F]+|[A-Z]\w*_t|[A-Z_0-9]+", ident):
            continue
        m = re.fullmatch(r"m_?([A-Z]\w*)", ident) or re.fullmatch(r"(?:Get|get|Is|is)([A-Z]\w*)", ident)
        base = m[1] if m else ident
        if len(base) >= 3:
            cands.append(base[0].lower() + base[1:])
            break
    t = type_text.replace("const", "").strip()
    if t.endswith("&"):
        cands += ["elem", "entry", "ref"]
    elif t.endswith("*"):
        cands += ["ptr", "obj", "node"]
    elif re.search(r"\b(?:float|double|f32|f64)\b", t):
        cands += ["value", "amount", "scale"]
    elif re.search(r"\bbool\b", t):
        cands += ["flag", "ok", "done"]
    else:
        cands += ["value", "count", "idx"]
    cands += ["item", "cur", "res", "tmpval"]
    for c in cands:
        if c not in taken and c not in _KEYWORDS and not _PLACEHOLDER.fullmatch(c):
            return c
    for c in cands:
        for k in ("A", "B", "C", "Next", "Cur"):
            if c + k not in taken:
                return c + k
    return cands[0] + "Local"


def tv(p: dict, q: dict) -> float:
    keys = set(p) | set(q)
    sp, sq = sum(p.values()) or 1, sum(q.values()) or 1
    return 0.5 * sum(abs(p.get(k, 0) / sp - q.get(k, 0) / sq) for k in keys)


def histograms(recs: list[dict]) -> tuple[dict, dict]:
    kinds: dict[str, int] = {}
    classes: dict[str, int] = {}
    for r in recs:
        for k in r.get("fix_kind") or []:
            kinds[k] = kinds.get(k, 0) + 1
        c = r.get("diff_class") or ""
        classes[c] = classes.get(c, 0) + 1
    return kinds, classes


def check(real: list[dict], synth: list[dict], min_cover: int = MIN_COVER) -> dict:
    """The pre-registered T35 check: TV of the fix-kind and diff-class histograms, and the share of real records
    whose every fix kind is carried by >= min_cover synthetic records."""
    rk, rc = histograms(real)
    sk, sc = histograms(synth)
    covered = [r for r in real if r.get("fix_kind") and all(sk.get(k, 0) >= min_cover for k in r["fix_kind"])]
    sets: dict[tuple, int] = {}
    for r in synth:
        t = tuple(sorted(r.get("fix_kind") or []))
        sets[t] = sets.get(t, 0) + 1
    exact_set = [r for r in real if sets.get(tuple(sorted(r.get("fix_kind") or [])), 0) >= min_cover]
    out = {"real": len(real), "synthetic": len(synth), "tv_kind": round(tv(rk, sk), 4), "tv_class": round(tv(rc, sc), 4),
           "covered": len(covered), "cover_share": round(len(covered) / max(1, len(real)), 4),
           "exact_set_share": round(len(exact_set) / max(1, len(real)), 4),
           "kinds_real": rk, "kinds_synth": sk, "classes_real": rc, "classes_synth": sc}
    out["pass"] = out["tv_kind"] <= PASS_TV and out["tv_class"] <= PASS_TV and out["cover_share"] >= PASS_COVER
    return out


_TYPEOF = r"__typeof__\s*\((?:[^()]|\((?:[^()]|\([^()]*\))*\))*\)"


def canon(defn: str, base: str) -> str:
    """`defn` with the locals `base` does not have renamed L0, L1... (first use) and their declared types blanked,
    pointer stars included (`T* q` vs `__typeof__(f(x)) t0`: before T37 a pointer local never compared equal, so
    no fold of a pointer local was reversible); a reference `&` is kept (a naming writes it explicitly)."""
    have = set(_ID.findall(base))
    text = re.sub(_TYPEOF, "@#", defn)
    for n in [n for n in dict.fromkeys(_ID.findall(text)) if n not in have and n not in _KEYWORDS]:
        text = re.sub(r"(?:@#|(?:(?:const|unsigned|signed|struct)\s+)*[A-Za-z_][\w:<>]*)(\s*[*&]*\s*)\b%s\b(?=\s*[=;\[])" % re.escape(n),
                      lambda m: "@#" + m[1].strip().replace("*", "") + " " + n, text, count=1)
    fresh = [n for n in dict.fromkeys(_ID.findall(text)) if n not in have and n not in _KEYWORDS]
    for k, n in enumerate(fresh):
        text = re.sub(r"\b%s\b" % re.escape(n), f"L{k}", text)
    return norm(text)


def within(text: str, new: str, span: tuple[int, int]) -> bool:
    """Does the edit stay inside the definition text[span]?"""
    tail = len(text) - span[1]
    return new[:span[0]] == text[:span[0]] and (tail == 0 or new[-tail:] == text[-tail:])


# ---- one function -------------------------------------------------------------------------------------------

class Fn:
    """One exact function: private -sym on compiles, the table's moves, the reversibility check."""

    def __init__(self, cfg: Config, unit: dict, symbol: str, src: str, work: Path):
        from .shaperules import project_accessors
        self.cfg, self.unit, self.symbol, self.src, self.work = cfg, unit, symbol, src, work
        self.c = unit["metadata"]["source_path"].endswith(".c")
        self.types = cbody.type_names(src)
        self.accessors = project_accessors(cfg.root, src)
        self.members: frozenset | None = None
        self.slot = 0
        self.compiles = 0
        self.file_ids = {t for t in _ID.findall(src)}
        self.table = SHAPE_TABLE

    def span(self, text: str):
        from .trycmd import definition
        return definition(text, self.symbol)

    def defn(self, text: str) -> str | None:
        sp = self.span(text)
        return text[sp[0]:sp[1]] if sp else None

    def site(self, text: str, focus=None, accessors=True):
        from . import rewrites
        sp = self.span(text)
        if sp is None:
            return None, None
        try:
            body = cbody.parse_body(text, text.index("{", sp[0], sp[1]), self.types)
        except (ValueError, cbody.ParseError):
            return None, None
        return rewrites.Site(text, body, focus, not self.c, members=self.members,
                             accessors=self.accessors if accessors else None), sp

    def moves(self, text: str, rules=None, per_rule: int = PER_RULE) -> dict[str, list[tuple[str, str]]]:
        """{rule: [(label, new source)]} of the table's one-step moves inside the definition (self.table when
        rules is None: SHAPE_TABLE, or stacked.py's table with the loop rows)."""
        from . import rewrites
        rules = self.table if rules is None else rules
        site, sp = self.site(text)
        if site is None:
            return {}
        out: dict[str, list[tuple[str, str]]] = {}
        for name in rules:
            rule = rewrites.BY_NAME.get(name)
            if rule is None:
                continue
            try:
                got = list(rewrites.candidates(site, [rule], groups=False))
            except Exception:                            # noqa: BLE001 (a rule that fails on a body applies nowhere)
                continue
            mark = FOLD_MARK.get(name)
            if mark is not None:                         # the cap applies to the fold and the name side each (T37 b)
                got = ([g for g in got if mark in g[1]][:per_rule] + [g for g in got if mark not in g[1]][:per_rule])
            else:
                got = got[:per_rule]
            for _r, label, new in got:
                if within(text, new, sp):
                    out.setdefault(name, []).append((label, new))
        return out

    def fold_moves(self, text: str) -> list[tuple[str, str, str]]:
        """[(rule, label, new source)] of the fold side of the fold rows (a local written out at its uses)."""
        mv = self.moves(text, rules=tuple(r for r, _ in FOLDS))
        return [(r, label, new) for r, mark in FOLDS for label, new in mv.get(r, []) if mark in label]

    def reverse(self, cur: str, prev: str) -> str | None:
        """The rule whose move turns `cur` back into `prev` (the forward fix), or None. A move that names a
        value writes `__typeof__` and a placeholder (t0): it counts when it equals `prev` up to the new locals'
        names and declared types (what close types from the debug info and names at landing)."""
        d_cur = self.defn(cur) or ""
        want = canon(self.defn(prev) or "", d_cur)
        for rule, xs in self.moves(cur, per_rule=REVERSE_PER_RULE).items():
            for _label, new in xs:
                if canon(self.defn(new) or "", d_cur) == want:
                    return rule
        return None

    def measure(self, text: str, types: bool = False) -> dict:
        """Compile privately with -sym on: compiled, percent, rows (relative), nrows, diff_class, shape, types."""
        from .dwarftypes import class_members, local_types
        from .elf import Elf
        from .llmtry import asm_shape, classify, compiler_errors, relative_rows
        from .regs import _diff
        from .reorder import compile_variant
        self.slot += 1
        self.compiles += 1
        work = self.work / f"m{self.slot}"
        try:
            obj, err = compile_variant(self.cfg, self.unit, text, work, ("-sym", "on"))
            if obj is None:
                return {"compiled": False, "error": (compiler_errors(err) or "compile error")[:300]}
            pair = _diff(self.cfg, self.unit, obj, self.symbol)
            if pair is None:
                return {"compiled": False, "error": "the function is missing from the object"}
            left, right = pair
            pct = float(left.get("match_percent", 0.0))
            lt, rt = left["instructions"], right["instructions"]
            out = {"compiled": True, "percent": pct,
                   "nrows": sum(1 for t, o in zip(lt, rt) if t.get("diff_kind") or o.get("diff_kind")) + abs(len(lt) - len(rt))}
            elf = None
            if pct < 100.0:
                out["rows"] = relative_rows(left, right)
                out["diff_class"] = classify(left, right)
                from .llmgen import object_asm
                e = object_asm(self.cfg, obj).get(self.symbol)
                out["shape"] = asm_shape(e.text) if e is not None else ""
            if types or self.members is None:
                elf = Elf.parse(obj.read_bytes())
                if self.members is None:
                    self.members = class_members(elf, self.symbol)
                if types:
                    out["types"] = local_types(elf, self.symbol, self.c)
            return out
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def typed(self, text: str) -> str | None:
        """`__typeof__` locals written with their debug-info types (a probe compile when optimized out)."""
        from .judge import replace_typeof
        if "__typeof__" not in text:
            return text
        site, _sp = self.site(text)
        if site is None:
            return None
        m = self.measure(text, types=True)
        if not m.get("compiled"):
            return None
        types = dict(m.get("types") or {})
        out, missing = replace_typeof(text, site.body, types)
        if missing:
            types.update(self.probe(text, site.body, missing))
            out, missing = replace_typeof(text, site.body, types)
        return None if missing or "__typeof__" in (self.defn(out) or "__typeof__") else out

    def probe(self, text: str, body, names: list[str]) -> dict[str, str]:
        from .dwarftypes import local_types
        from .elf import Elf
        from .reorder import compile_variant
        edits = [(0, 0, ('extern "C" ' if not self.c else "") + "void dk_type_probe(const void*);\n")]
        for s in cbody.statements(body):
            if s.kind == "decl" and s.decl.declarators[0].name in names and "&" not in s.decl.declarators[0].ptr:
                edits.append((s.end, s.end, f" dk_type_probe(&{s.decl.declarators[0].name});"))
        if len(edits) == 1:
            return {}
        self.slot += 1
        self.compiles += 1
        work = self.work / f"p{self.slot}"
        try:
            obj, _err = compile_variant(self.cfg, self.unit, cbody.apply_edits(text, edits), work, ("-sym", "on"))
            if obj is None:
                return {}
            found = local_types(Elf.parse(obj.read_bytes()), self.symbol, self.c)
            return {k: v for k, v in found.items() if k in names}
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def rename_new(self, prev: str, text: str) -> str:
        """Placeholder locals a move introduced (t0, e1) named from the value they hold."""
        d_prev, d_new = self.defn(prev) or "", self.defn(text) or ""
        fresh = [n for n in dict.fromkeys(_NEW.findall(d_new)) if not re.search(r"\b%s\b" % n, d_prev)]
        if not fresh:
            return text
        sp = self.span(text)
        taken = set(self.file_ids) | set(self.members or ()) | set(_ID.findall(d_new))
        body = d_new
        for n in fresh:
            m = re.search(r"([\w:<>]+(?:\s*[*&]+)?)\s*([*&]*)\s*\b%s\b\s*=\s*([^;]+);" % n, body)
            name = natural_name(m[3] if m else "", ((m[1] + m[2]) if m else ""), taken)
            taken.add(name)
            body = re.sub(r"\b%s\b" % n, name, body)
        return text[:sp[0]] + body + text[sp[1]:]


def table_version() -> str:
    """A hash of the rewrite table's code: a cached function result is reused only with the same table."""
    here = Path(__file__).parent
    return sha(*((here / n).read_text(errors="replace") for n in ("rewrites.py", "shaperules.py", "shapesynth.py",
                                                                   "loopmoves.py", "giantmoves.py")))[:12]


def pick_depth(rng: random.Random) -> int:
    x = rng.randrange(100)
    for d, w in DEPTHS:
        if x < w:
            return d
        x -= w
    return DEPTHS[-1][0]


def sides(rule: str, moves: list[tuple[str, str]], rng: random.Random) -> list[tuple[str, str]]:
    """The moves of one row in the order they are tried. In the rows that fold a local or name a value
    (FOLDS), the fold side and the name side are drawn with probability 1/2 each when both have moves (T37 b),
    not in proportion to their counts (any call, load or value can be named; only existing locals fold);
    the drawn side's moves come first, shuffled. Other rows: all moves shuffled."""
    xs = list(moves)
    mark = FOLD_MARK.get(rule)
    folds = [m for m in xs if mark is not None and mark in m[0]]
    if not folds or len(folds) == len(xs):
        rng.shuffle(xs)
        return xs
    names = [m for m in xs if mark not in m[0]]
    rng.shuffle(folds)
    rng.shuffle(names)
    return folds + names if rng.random() < 0.5 else names + folds


def schedule(n_fns: int, pool: list[int], p: float, seed: int) -> list[tuple[int, str]]:
    """The run's draws (T38 targeted draw): each draw is, with probability p, a fold draw ("fold": the next
    function of the fold pool, in the input order) and otherwise an ordinary draw ("any": the next input
    function). Once the pool is used up, only ordinary draws remain. Deterministic for a seed."""
    rng = random.Random(hkey(seed, "schedule"))
    out: list[tuple[int, str]] = []
    i = k = 0
    while i < n_fns:
        if k < len(pool) and rng.random() < p:
            out.append((pool[k], "fold"))
            k += 1
        else:
            out.append((i, "any"))
            i += 1
    return out


def build_path(fn: Fn, s0: str, depth: int, rng: random.Random, stats: dict, fold_first: bool = False) -> list[dict]:
    """States S1..Sd from S0, each one table move, typed and named, reversible by a table move. With
    fold_first (a fold draw, T38) the first step is a fold-side move of the fold rows or the path is empty."""
    states = [s0]
    seen = {norm(fn.defn(s0) or "")}
    steps: list[dict] = []
    for k in range(depth):
        cur = states[-1]
        if fold_first and k == 0:
            folds = fn.fold_moves(cur)
            rng.shuffle(folds)
            tries = [(r, [(label, new) for r2, label, new in folds[:FOLD_TRIES] if r2 == r]) for r in
                     dict.fromkeys(r for r, _l, _n in folds[:FOLD_TRIES])]
        else:
            mv = fn.moves(cur)
            rules = sorted(mv)
            rng.shuffle(rules)
            tries = ((rule, sides(rule, mv[rule], rng)[:2]) for rule in rules[:4])     # lazy: T37's draw order
        done = False
        for rule, xs in tries:
            for label, new in xs:
                new = fn.typed(new)
                if new is None:
                    stats["untyped"] = stats.get("untyped", 0) + 1
                    continue
                new = fn.rename_new(cur, new)
                d = norm(fn.defn(new) or "")
                if not d or d in seen:
                    continue
                back = fn.reverse(new, cur)
                if back is None:
                    stats["not reversible"] = stats.get("not reversible", 0) + 1
                    continue
                seen.add(d)
                states.append(new)
                steps.append({"rule": rule, "label": label[:160], "forward": back, "text": new})
                done = True
                break
            if done:
                break
        if not done:
            if fold_first and k == 0:
                stats["fold draw: no fold made"] = stats.get("fold draw: no fold made", 0) + 1
            break
    return steps


def clean(fn: Fn, before_src: str, after_src: str) -> tuple[bool, list[str], str]:
    """Judge-clean (pre-registered): (ok, the review's judgment flags, why not)."""
    from .judge import REJECT, forbidden_rules, judge, review
    forbidden = _FORBIDDEN.setdefault("x", forbidden_rules(fn.cfg))
    v = judge(before_src, after_src, None, None, None, None, fn.symbol, forbidden)
    if v.reasons:
        return False, [], "judge: " + "; ".join(v.reasons)[:200]
    cat1, flags1 = review(before_src, after_src, fn.symbol, forbidden)
    cat2, flags2 = review(after_src, before_src, fn.symbol, forbidden)
    if cat1 == REJECT or cat2 == REJECT:
        return False, [], "review: " + "; ".join(flags1 if cat1 == REJECT else flags2)[:200]
    flags = flags1 + [f"(before) {x}" for x in flags2]
    if any("empty case labels" in x for x in flags):
        return False, [], "empty case labels"
    bad = invented_cases(fn.defn(before_src) or "", fn.defn(after_src) or "")
    if bad:
        return False, [], "invented case " + ", ".join(bad)
    return True, flags, ""


_FORBIDDEN: dict = {}


def score_line(m: dict) -> str:
    from .llmagent import Score
    if not m.get("compiled"):
        return Score(False, error=m.get("error", "")).line()
    return Score(True, m["percent"], (), m.get("rows") or []).line()


def key_of(m: dict) -> tuple:
    return (-m["percent"], m.get("nrows", 0)) if m.get("compiled") else (1, 1 << 30)


def edit_action(fn: Fn, cur: str, nxt: str, rule: str) -> dict:
    """The target action from cur to nxt: the edit (fix_hunk) and the structured form when the harness's
    own lookup (the rule's first candidates at the first changed line, no accessors) reproduces the step."""
    from .llmagent import Action, fix_hunk
    from . import rewrites
    a, b = fn.defn(cur) or "", fn.defn(nxt) or ""
    old, new = fix_hunk(a, b)
    out = {"kind": "edit", "old": old, "new": new, "summary": Action("edit", old=old, new=new).summary()}
    sp = fn.span(cur)
    al, bl = a.split("\n"), b.split("\n")
    first = next((i for i, (x, y) in enumerate(zip(al, bl)) if x != y), min(len(al), len(bl)) - 1)
    site, _ = fn.site(cur, focus={cur.count("\n", 0, sp[0]) + first}, accessors=False)
    if site is not None and rule in rewrites.BY_NAME:
        try:
            cands = [x for _r, _l, x in rewrites.candidates(site, [rewrites.BY_NAME[rule]], groups=False)]
        except Exception:                                # noqa: BLE001
            cands = []
        want = norm(b)
        for i, c in enumerate(cands[:8]):
            if norm(fn.defn(c) or "") == want:
                out["structured"] = {"transform": rule, "anchor": first + 1, "pick": i}
                break
    return out


def losers(fn: Fn, cur: str, m_cur: dict, avoid: set[str], rng: random.Random, want: int, rules) -> list[dict]:
    """Up to `want` grid losers at `cur`: table moves that compile and score worse (the harness reverts them)."""
    from .llmagent import Action, fix_hunk
    mv = fn.moves(cur, rules=rules, per_rule=4)
    pool = [(r, l, n) for r in sorted(mv) for l, n in mv[r] if "__typeof__" not in (fn.defn(n) or "")
            and norm(fn.defn(n) or "") not in avoid]
    rng.shuffle(pool)
    out = []
    for rule, _label, new in pool[:want + 2]:
        m = fn.measure(new)
        if not m.get("compiled") or key_of(m) <= key_of(m_cur):
            continue
        old, rep = fix_hunk(fn.defn(cur) or "", fn.defn(new) or "")
        out.append({"rule": rule, "summary": Action("edit", old=old, new=rep).summary(),
                    "result": score_line(m) + f" (worse than the best {m_cur['percent']:.2f}%: reverted)"})
        if len(out) >= want:
            break
    return out


def work_dir(work: Path, function: str, mode: str = "any") -> Path:
    """A draw's private compile directory: an ordinary and a fold draw of the same function may run at once
    (T38: a shared directory let one draw remove or read the other's objects)."""
    return work / hashlib.sha1((function + ("" if mode == "any" else "\0" + mode)).encode()).hexdigest()[:10]


def one_function(cfg: Config, f: dict, seed: int, per_fn: int, work: Path, mode: str = "any") -> dict:
    """Pairs and policy examples of one exact function (deterministic for a seed, source and draw kind;
    mode "fold": a fold draw, every path starts with a fold)."""
    from .llmfixes import fix_kinds
    u = f["u"]
    src = (cfg.root / u["metadata"]["source_path"]).read_text(errors="replace")
    stats: dict = {}
    fn = Fn(cfg, u, f["function"], src, work_dir(work, f["function"], mode))
    res = {"function": f["function"], "unit": f["unit"], "mode": mode, "pairs": [], "policy": [], "stats": stats}
    try:
        d0 = fn.defn(src)
        if d0 is None:
            stats["no definition"] = 1
            return res
        m0 = fn.measure(src)
        if not m0.get("compiled") or m0["percent"] < 100.0:
            stats["not exact privately"] = 1
            return res
        rng = random.Random(hkey(seed, f["unit"], f["function"]) if mode == "any" else
                            hkey(seed, "fold", f["unit"], f["function"]))
        seen: set[str] = set()
        for attempt in range(per_fn * 3):
            if len(res["pairs"]) >= per_fn:
                break
            steps = build_path(fn, src, pick_depth(rng), rng, stats, fold_first=mode == "fold")
            if not steps:
                stats["no move"] = stats.get("no move", 0) + 1
                if stats["no move"] >= 2:
                    break
                continue
            before = steps[-1]["text"]
            key = norm(fn.defn(before) or "")
            if key in seen:
                stats["repeat"] = stats.get("repeat", 0) + 1
                continue
            seen.add(key)
            mb = fn.measure(before)
            if not mb.get("compiled"):
                stats["before does not compile"] = stats.get("before does not compile", 0) + 1
                continue
            if mb["percent"] >= 100.0:
                stats["same code"] = stats.get("same code", 0) + 1
                continue
            ok, flags, why = clean(fn, before, src)
            if not ok:
                stats["not clean: " + why.split(":")[0]] = stats.get("not clean: " + why.split(":")[0], 0) + 1
                continue
            b_def = fn.defn(before)
            pid = sha(f["unit"], f["function"], b_def)[:16]
            pair = {"v": SCHEMA, "source": "synthetic", "id": pid, "unit": f["unit"], "function": f["function"],
                    "size": f["size"], "commit": HEAD.get("rev", ""), "parent": None, "commit_time": None,
                    "path": u["metadata"]["source_path"], "asm_shape": f["asm_shape"], "before": b_def, "after": d0,
                    "before_percent": mb["percent"], "after_percent": 100.0, "diff_class": mb.get("diff_class", ""),
                    "rows": mb.get("rows") or [], "oracle": None, "fix_kind": fix_kinds(b_def, d0),
                    "headers_changed": [], "grind_log": None, "asm": f.get("asm", ""),
                    "transforms": [s["rule"] for s in steps], "forward": [s["forward"] for s in reversed(steps)],
                    "steps": [{"rule": s["rule"], "label": s["label"]} for s in steps],
                    "mutant_shape": mb.get("shape", ""), "review_flags": flags[:6]}
            res["pairs"].append(pair)
            res["policy"] += policy_trace(fn, src, steps, mb, pair, rng)
        stats["pairs"] = len(res["pairs"])
    except Exception as exc:                             # noqa: BLE001 (one function never stops the run)
        stats[f"error {type(exc).__name__}"] = 1
    finally:
        stats["compiles"] = fn.compiles
        shutil.rmtree(fn.work, ignore_errors=True)
    return res


def policy_trace(fn: Fn, s0: str, steps: list[dict], mb: dict, pair: dict, rng: random.Random) -> list[dict]:
    """Single-step examples along the forward path S_d -> S_0 while each step is no worse (the harness keeps it)."""
    states = [s0] + [s["text"] for s in steps]                 # states[j] = S_j
    history: list[list[str]] = []
    out = []
    m_cur = mb
    for j in range(len(steps), 0, -1):
        cur, nxt = states[j], states[j - 1]
        m_next = fn.measure(nxt) if j > 1 else {"compiled": True, "percent": 100.0, "nrows": 0}
        if j > 1 and (not m_next.get("compiled") or key_of(m_next) > key_of(m_cur)):
            break                                              # the harness would revert this step: the trace ends
        hist = list(history)
        if rng.random() < LOSER_SHARE:
            avoid = {norm(fn.defn(x) or "") for x in states}
            for lo in losers(fn, cur, m_cur, avoid, rng, 1 + (rng.random() < 0.4), SHAPE_TABLE):
                hist.append([lo["summary"], lo["result"]])
        act = edit_action(fn, cur, nxt, steps[j - 1]["forward"])
        out.append({"v": SCHEMA, "source": "synthetic", "kind": "policy", "pair": pair["id"], "unit": pair["unit"],
                    "function": pair["function"], "asm_shape": pair["asm_shape"], "step": len(steps) - j + 1,
                    "steps": len(steps),
                    "state": {"definition": fn.defn(cur), "score": score_line(m_cur), "percent": m_cur["percent"],
                              "diff_class": m_cur.get("diff_class", ""), "rows": m_cur.get("rows") or [],
                              "history": hist[-HISTORY:]},
                    "target": {k: act[k] for k in ("kind", "old", "new")},
                    "structured": act.get("structured"), "rule": steps[j - 1]["forward"],
                    "losers_in_history": len(hist) - len(history),
                    "outcome": "exact" if j == 1 else "kept"})
        res_line = "exact (judged)" if j == 1 else score_line(m_next) + (
            " (kept: the new best)" if key_of(m_next) < key_of(m_cur) else " (kept: as good)")
        history = hist + [[act["summary"], res_line]]
        m_cur = m_next
    return out


# ---- dead ends (STOP examples) --------------------------------------------------------------------------------

def dead_ends(cfg: Config, fz: dict) -> list[dict]:
    """Open functions `close --queue` searched (>= 100 compiles) without any gain, newest row per function."""
    from .llmtry import data_dir
    rows: dict[str, dict] = {}
    for p in sorted(data_dir(cfg).glob("close-queue-*.jsonl")):
        for line in p.read_text(errors="replace").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("kind") != "close-queue" or r.get("error") or r.get("exact"):
                continue
            if r.get("compiles", 0) >= 100 and r.get("best", 0) <= r.get("start", 0) and \
                    r.get("class") in ("near-register", "near-shape", "near-scheduling"):
                rows[r["function"]] = r
    return [r for f, r in sorted(rows.items()) if f not in fz["functions"] and r["unit"] not in fz["units"]]


def one_dead_end(cfg: Config, row: dict, seed: int, variants: int, work: Path, units: dict, entries: dict) -> list[dict]:
    from .llmtry import asm_shape
    u = units.get(row["unit"])
    if u is None or row["function"] not in entries:
        return []
    src = (cfg.root / u["metadata"]["source_path"]).read_text(errors="replace")
    fn = Fn(cfg, u, row["function"], src, work / ("d" + hashlib.sha1(row["function"].encode()).hexdigest()[:10]))
    out = []
    try:
        d = fn.defn(src)
        if d is None:
            return []
        m = fn.measure(src)
        if not m.get("compiled") or m["percent"] >= 100.0:
            return []
        rng = random.Random(hkey(seed, "stop", row["function"]))
        from . import rewrites
        table = [r.name for r in rewrites.RULES if r.name not in ("inline-helper", "helper-params")]
        los = losers(fn, src, m, {norm(d)}, rng, 6, table)
        if not los:
            return []
        label = "register-only" if m.get("diff_class") == "register" else "no-idea"
        shape = asm_shape(entries[row["function"]].text)
        hists, seen = [], set()
        for k in range(1, min(HISTORY, len(los)) + 1):          # every history length, then other loser subsets
            hists.append(los[:k])
            seen.add(tuple(x["summary"] for x in los[:k]))
        for _ in range(4 * variants):
            if len(hists) >= variants:
                break
            sub = rng.sample(los, rng.randint(1, min(HISTORY, len(los))))
            if tuple(x["summary"] for x in sub) not in seen:
                seen.add(tuple(x["summary"] for x in sub))
                hists.append(sub)
        for h in hists[:variants]:
            k = len(h)
            out.append({"v": SCHEMA, "source": "dead-end", "kind": "policy", "pair": None, "unit": row["unit"],
                        "function": row["function"], "asm_shape": shape, "step": k + 1, "steps": None,
                        "state": {"definition": d, "score": score_line(m), "percent": m["percent"],
                                  "diff_class": m.get("diff_class", ""), "rows": m.get("rows") or [],
                                  "history": [[x["summary"], x["result"]] for x in h][-HISTORY:]},
                        "target": {"kind": "stop", "label": label}, "structured": None, "rule": None,
                        "losers_in_history": k, "outcome": "stop",
                        "evidence": {"close_compiles": row.get("compiles"), "close_start": row.get("start"),
                                     "close_time": row.get("time")}})
    except Exception:                                    # noqa: BLE001
        return out
    finally:
        shutil.rmtree(fn.work, ignore_errors=True)
    return out


# ---- the run ----------------------------------------------------------------------------------------------------

HEAD: dict = {}
_POOL: dict = {}


def _work(job: tuple[int, str]) -> dict:
    c = _POOL
    return one_function(c["cfg"], c["fns"][job[0]], c["seed"], c["per_fn"], c["work"], job[1])


def foldable(cfg: Config, f: dict) -> bool:
    """Does the exact source of f have a fold-side move of the fold rows (a local that can be written out)?
    Pure source work, no compile: the fold pool of the targeted draw (T38)."""
    try:
        src = (cfg.root / f["u"]["metadata"]["source_path"]).read_text(errors="replace")
        return bool(Fn(cfg, f["u"], f["function"], src, Path("/nonexistent")).fold_moves(src))
    except Exception:                                    # noqa: BLE001 (a body the parser refuses folds nowhere)
        return False


def _foldable(i: int) -> bool:
    c = _POOL
    return foldable(c["cfg"], c["fns"][i])


def _dead(i: int) -> list[dict]:
    c = _POOL
    return one_dead_end(c["cfg"], c["dead"][i], c["seed"], c["variants"], c["work"], c["units"], c["entries"])


def leak_free(r: dict, fz: dict) -> str | None:
    from .gate import agnostic_hash
    from .llmfixes import leak, norm_hash
    why = leak(r, fz) or text_leak([r["before"], r["after"]])
    if why:
        return why
    for k in ("mutant_shape",):
        if r.get(k) and r[k] in fz["shapes"]:
            return f"mutant asm shape ({fz['shapes'][r[k]]})"
    return None


PATS: list = []                    # llmdata.never_definitions: a frozen function's definition head in a text


def text_leak(texts) -> str | None:
    from .llmdata import LITERAL
    for t in texts:
        for pat in PATS:
            lit = LITERAL.get(pat.pattern)
            if (lit is None or lit in t) and pat.search(t):
                return f"never-train definition in text [{lit}]"
    return None


def policy_leak(p: dict, fz: dict) -> str | None:
    from .gate import code_hashes
    if text_leak([p["state"]["definition"]]):
        return "never-train definition in text"
    if p["unit"] in fz["units"]:
        return "unit"
    if p["function"] in fz["functions"]:
        return "function"
    if p.get("asm_shape") in fz["shapes"]:
        return "asm shape"
    if code_hashes(p["state"]["definition"]) & fz["hashes"].keys():
        return "content hash"
    return None


def out_dir(cfg: Config, args) -> Path:
    from .llmtry import data_dir
    return Path(args.out).expanduser() if args.out else data_dir(cfg) / "datasets" / "shape-synth" / "v0"


def eligible(cfg: Config, args) -> tuple[list[dict], dict, dict] | None:
    """The input functions (exact, sizes, leakage filter, quarantine) in the seed's order, the frozen sets, the asm."""
    from .llmdata import candidates, never_definitions, quarantined
    from .llmfixes import frozen
    from .llmtry import asm_entries, compact_asm, data_dir, never_train
    fz = frozen(cfg)
    if not fz["units"]:
        print("error: no frozen eval set found in llm-data/ (bench.json, heldout.json): refusing to build")
        return None
    quar = quarantined(data_dir(cfg) / "judge-audit.json") or set()
    PATS[:] = never_definitions(cfg, set(never_train(cfg)[1]) | set(fz["functions"]))
    ns = argparse.Namespace(only=args.only, min_size=args.min_size, jobs=args.jobs)
    fns = [f for f in candidates(cfg, ns, False) if f["size"] <= args.max_size and f["function"] not in quar
           and f["function"] not in fz["functions"] and f["unit"] not in fz["units"] and f["asm_shape"] not in fz["shapes"]]
    entries = asm_entries(cfg)
    for f in fns:
        f["asm"] = compact_asm(entries[f["function"]].text)
    fns.sort(key=lambda f: hkey(args.seed, f["unit"], f["function"]))
    return fns, fz, entries


def probe_function(cfg: Config, f: dict, work: Path) -> dict:
    """Every fold of a local the table makes on an exact function, and whether a table move restores the
    exact source (pure source work, no compile): {rule: [reversible, total]}."""
    src = (cfg.root / f["u"]["metadata"]["source_path"]).read_text(errors="replace")
    fn = Fn(cfg, f["u"], f["function"], src, work)
    out: dict[str, list] = {}
    for rule, mark in FOLDS:
        for label, new in fn.moves(src, rules=(rule,), per_rule=1000).get(rule, []):
            if mark not in label:
                continue
            c = out.setdefault(rule, [0, 0, []])
            c[1] += 1
            if fn.reverse(new, src) is not None:
                c[0] += 1
            elif len(c[2]) < 3:
                c[2].append(label[:80])
    return out


def _probe(i: int) -> dict:
    c = _POOL
    return probe_function(c["cfg"], c["fns"][i], c["work"])


def run_probe(cfg: Config, args) -> int:
    """Reversibility of the table's local folds on the first --probe N input functions (the T36 probe)."""
    import multiprocessing
    got = eligible(cfg, args)
    if got is None:
        return USAGE
    fns = got[0][:args.probe]
    _POOL.update(cfg=cfg, fns=fns, work=Path("/nonexistent"))
    tot: dict[str, list] = {}
    fails: list[str] = []
    with multiprocessing.get_context("fork").Pool(args.jobs) as pool:
        for f, res in zip(fns, pool.imap(_probe, range(len(fns)), chunksize=4)):
            for rule, (ok, n, why) in res.items():
                t = tot.setdefault(rule, [0, 0])
                t[0] += ok
                t[1] += n
                fails += [f"{f['function']}: {w}" for w in why]
    print(f"fold reversibility on {len(fns)} functions (seed {args.seed}):")
    for rule, (ok, n) in sorted(tot.items()):
        print(f"  {rule:14} {ok}/{n} reversible")
    for x in fails[:args.probe_show]:
        print(f"  not reversible: {x}")
    return OK


def run(cfg: Config, args: argparse.Namespace) -> int:
    if args.check:
        return run_check(cfg, args)
    if args.probe:
        return run_probe(cfg, args)
    import multiprocessing
    import subprocess
    from .llmtry import LLM_DIR
    from .matching import load_objdiff_units
    t0 = time.time()
    until = 0.0
    if args.until:
        hh, mm = map(int, args.until.split(":"))
        lt = time.localtime()
        until = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, hh, mm, 0, 0, 0, -1))
        until += 86400 if until <= t0 else 0
    out = out_dir(cfg, args)
    out.mkdir(parents=True, exist_ok=True)
    HEAD["rev"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cfg.root, capture_output=True, text=True).stdout.strip()
    got = eligible(cfg, args)
    if got is None:
        return USAGE
    fns, fz, entries = got
    print(f"{len(fns)} exact functions after the leakage filter ({time.time() - t0:.0f} s); target {args.n} pairs")
    cache_p = out / "cache" / "functions.jsonl"
    cache_p.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict] = {}
    if cache_p.is_file():
        for line in cache_p.read_text().splitlines():
            try:
                x = json.loads(line)
                cache[x["key"]] = x
            except (ValueError, KeyError):
                pass
    srcs: dict[str, str] = {}
    table = table_version()

    def fkey(f, mode="any"):
        p = f["u"]["metadata"]["source_path"]
        if p not in srcs:
            srcs[p] = sha((cfg.root / p).read_text(errors="replace"))
        return sha(str(SCHEMA), table, str(args.seed), str(args.per_fn), f["unit"], f["function"], srcs[p],
                   *(() if mode == "any" else (mode,)))
    work = cfg.root / LLM_DIR / "work" / f"shapesynth{int(t0)}"
    _POOL.update(cfg=cfg, fns=fns, seed=args.seed, per_fn=args.per_fn, work=work)
    # the fold pool of the targeted draw (T38): input functions with a local that can be folded, cached
    fold_p = out / "cache" / "foldable.json"
    fcache = json.loads(fold_p.read_text()) if fold_p.is_file() else {}
    need = [i for i, f in enumerate(fns) if fkey(f, "pool") not in fcache]
    if need and args.fold_draw > 0:
        with multiprocessing.get_context("fork").Pool(args.jobs) as pool:
            for i, ok in zip(need, pool.imap(_foldable, need, chunksize=8)):
                fcache[fkey(fns[i], "pool")] = ok
        fold_p.write_text(json.dumps(fcache))
    fold_pool = [i for i, f in enumerate(fns) if fcache.get(fkey(f, "pool"))] if args.fold_draw > 0 else []
    jobs = schedule(len(fns), fold_pool, args.fold_draw, args.seed)
    print(f"fold pool {len(fold_pool)}/{len(fns)} functions; fold draw share {args.fold_draw}")
    pairs: list[dict] = []
    policy: list[dict] = []
    stats: dict = {"functions": 0, "cached": 0, "leaks": {}, "draws any": 0, "draws fold": 0,
                   "fold pool": len(fold_pool), "pairs from fold draws": 0, "duplicate pair": 0}
    have: set[str] = set()

    def take(res: dict) -> None:
        stats["functions"] += 1
        mode = res.get("mode", "any")
        stats["draws " + mode] += 1
        for k, v in res["stats"].items():
            stats[k] = stats.get(k, 0) + v
        new: set[str] = set()
        for p in res["pairs"]:
            why = leak_free(p, fz)
            if why:
                stats["leaks"][why] = stats["leaks"].get(why, 0) + 1
                continue
            if p["id"] in have:                          # the same before from the other draw kind: kept once
                stats["duplicate pair"] += 1
                continue
            have.add(p["id"])
            new.add(p["id"])
            p["draw"] = mode
            stats["pairs from fold draws"] += mode == "fold"
            pairs.append(p)
        policy.extend(x for x in res["policy"] if x["pair"] in new and not policy_leak(x, fz))
    todo = []
    for job in jobs:
        if len(pairs) >= args.n:
            break
        k = fkey(fns[job[0]], job[1])
        if k in cache:
            stats["cached"] += 1
            take(cache[k]["res"])
        else:
            todo.append(job)
    with cache_p.open("a") as fh, multiprocessing.get_context("fork").Pool(args.jobs) as pool:
        for job, res in zip(todo, pool.imap(_work, todo, chunksize=1)):
            fh.write(json.dumps({"key": fkey(fns[job[0]], job[1]), "res": res}) + "\n")
            fh.flush()
            take(res)
            if stats["functions"] % 50 == 0:
                print(f"  {stats['functions']} functions, {len(pairs)} pairs, {len(policy)} policy examples, "
                      f"{stats.get('compiles', 0)} compiles, {time.time() - t0:.0f} s", flush=True)
            if len(pairs) >= args.n or (until and time.time() >= until):
                pool.terminate()
                break
    pairs = pairs[:args.n]
    ids = {p["id"] for p in pairs}
    policy = [x for x in policy if x["pair"] in ids]
    # STOP examples from the table's real dead ends
    want_stop = int(round(len(policy) * args.stop_share / (1 - args.stop_share)))
    dead = dead_ends(cfg, fz)
    units = {u["name"]: u for u in load_objdiff_units(cfg) if u.get("base_path")}
    _POOL.update(dead=dead, variants=args.stop_variants, units=units, entries=entries)
    stops: list[dict] = []
    if want_stop and dead and not (until and time.time() >= until):
        with multiprocessing.get_context("fork").Pool(args.jobs) as pool:
            for got in pool.imap(_dead, range(len(dead)), chunksize=1):
                stops += [x for x in got if not policy_leak(x, fz)]
                if len(stops) >= want_stop:
                    pool.terminate()
                    break
    stops = stops[:want_stop]
    policy += stops
    shutil.rmtree(work, ignore_errors=True)
    (out / "pairs.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pairs))
    (out / "policy.jsonl").write_text("".join(json.dumps(p) + "\n" for p in policy))
    stats.update(dead_ends=len(dead), stops=len(stops), want_stop=want_stop, wall=round(time.time() - t0),
                 cpu_compiles=stats.get("compiles", 0))
    (out / "stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    write_card(cfg, out, pairs, policy, stats, fz, args)
    print(f"shape-synth: {len(pairs)} pairs from {stats['functions']} functions ({stats['cached']} cached), "
          f"{len(policy)} policy examples ({len(stops)} STOP of {want_stop} wanted, {len(dead)} dead ends); "
          f"{stats.get('compiles', 0)} compiles; wall {time.time() - t0:.0f} s -> {out}")
    print("  " + ", ".join(f"{k} {v}" for k, v in sorted(stats.items()) if isinstance(v, int) and k not in ("functions",)))
    return OK


def write_card(cfg: Config, out: Path, pairs: list[dict], policy: list[dict], stats: dict, fz: dict, args) -> None:
    by_t: dict[str, int] = {}
    for p in pairs:
        for t in p["transforms"]:
            by_t[t] = by_t.get(t, 0) + 1
    depth: dict[int, int] = {}
    for p in pairs:
        depth[len(p["transforms"])] = depth.get(len(p["transforms"]), 0) + 1
    kinds, classes = histograms(pairs)
    targets: dict[str, int] = {}
    for x in policy:
        k = x["target"]["kind"] if x["target"]["kind"] != "stop" else "stop " + x["target"]["label"]
        targets[k] = targets.get(k, 0) + 1
    chk = (out / "check.json")
    lines = [f"# shape-synth {out.name}", "",
             "Synthetic shape-fix pairs and single-step agent-policy examples for LoRA 2 (plan B.2, council item 5).",
             "Generated by `./dk dataset shape-synth` (tools/dk/platforms/gc_mwcc/shapesynth.py). Game-derived code: private,",
             "never commit or publish. No model or Claude text: project code, compiler/objdiff output, rule ids, the",
             "llm-agent harness's fixed result strings.", "",
             f"- commit: `{HEAD.get('rev', '')}`  table {table_version()}  seed {args.seed}  per function <= {args.per_fn}  sizes {args.min_size:#x}-{args.max_size:#x}",
             f"- pairs: **{len(pairs)}** from {len({(p['unit'], p['function']) for p in pairs})} functions, "
             f"{len({p['unit'] for p in pairs})} units ({stats['functions']} draws read)",
             f"- stack depth: " + ", ".join(f"{k}: {v}" for k, v in sorted(depth.items())),
             f"- transforms (inverse moves applied): " + ", ".join(f"{k} {v}" for k, v in sorted(by_t.items(), key=lambda x: -x[1])),
             f"- diff class: " + ", ".join(f"{k} {v}" for k, v in sorted(classes.items())),
             f"- fix kinds: " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items(), key=lambda x: -x[1])),
             f"- policy examples: **{len(policy)}**: " + ", ".join(f"{k} {v}" for k, v in sorted(targets.items())) +
             f"; with grid losers in the history {sum(1 for x in policy if x['losers_in_history'])}; "
             f"with a structured form {sum(1 for x in policy if x.get('structured'))}",
             f"- left out: " + ", ".join(f"{k} {v}" for k, v in sorted(stats.items()) if isinstance(v, int)
                                         and k not in ("functions", "cached", "pairs", "compiles", "dead_ends", "stops",
                                                       "want_stop", "wall", "cpu_compiles", "draws any", "draws fold",
                                                       "fold pool", "pairs from fold draws")),
             f"- compiles {stats.get('compiles', 0)}, wall {stats.get('wall', 0)} s", "",
             "## Sampling (rebalancing stated)", "",
             "Each step draws up to 4 rows in random order, then tries up to 2 of the row's moves. In the rows that fold",
             "a local or name a value (" + ", ".join(r for r, _ in FOLDS) + ") the fold side and the name side are drawn",
             "1/2 each when both have moves, and the per-row cap (" + str(PER_RULE) + " moves) applies to each side (T37 b);",
             "before T37 they were drawn per move, so the namings (any call, load or value) outnumbered the folds (only",
             "existing locals). This raises the fixes that add a local (local-added) and lowers local-removed, on purpose.",
             f"Targeted draw (T38): the work list is a sequence of draws. Each is, with probability {args.fold_draw}, a",
             "fold draw: the next function (seed order) of the fold pool, the inputs whose exact source has a fold-side",
             "move (a local that can be written out at its uses); its paths' first step is such a fold (up to "
             f"{FOLD_TRIES} tried), the",
             "later steps ordinary. Otherwise an ordinary draw (the next input function, as in T37). Every pair passes the",
             "same checks (typed, reversible by a table move, before compiles non-exact, judge-clean, leakage); a pair whose",
             "before an earlier draw already made is kept once. This raises local-added toward its real share, on purpose.",
             f"Draws: ordinary {stats.get('draws any', 0)}, fold {stats.get('draws fold', 0)} (pool {stats.get('fold pool', 0)} "
             f"functions); pairs from fold draws {sum(1 for p in pairs if p.get('draw') == 'fold')}.",
             "No other row is reweighted.", "",
             "## Files", "",
             "- pairs.jsonl: `dataset fixes`' schema (v, source=synthetic, unit, function, size, commit (HEAD), parent/commit_time null,",
             "  path, asm_shape, before, after, before_percent, after_percent, diff_class, rows, oracle null, fix_kind,",
             "  headers_changed, grind_log, asm) + id, transforms (inverse rule ids, applied in order), forward (the fix's",
             "  rule ids, in order), steps (rule + the table's label), mutant_shape, review_flags (judge.review needs-judgment).",
             "- policy.jsonl: kind=policy; state {definition, score (llm-agent's Score.line), percent, diff_class, rows,",
             "  history [[action, result]] (<= 4)}; target {kind edit: old/new SEARCH/REPLACE | kind stop: label};",
             "  structured {transform, anchor, pick} or null; source synthetic (pair id) or dead-end (close-queue evidence).",
             "  Render with llm-agent's build_prompt at export (not stored: the prompt text is the harness's).",
             "- cache/functions.jsonl: each function's result by source hash (a larger --n reuses it).", "",
             "## Leakage", "",
             f"Inputs and records filtered by llmfixes.frozen (bench/heldout/repair-eval/review-eval units; gate-lora2 and",
             f"near-sample functions, asm shapes, definition and variant hashes incl. gate-freeze --rehash) and the",
             f"judge-audit quarantine; also the mutant's asm shape. Union: {len(fz['units'])} units, {len(fz['functions'])} functions,",
             f"{len(fz['shapes'])} shapes, {len(fz['hashes'])} hashes. Dropped records: {stats['leaks'] or 'none'}.", ""]
    if chk.is_file():
        c = json.loads(chk.read_text())
        lines += ["## Distribution check (pre-registered, T35 marks)", "",
                  f"vs {c.get('real_path', '')}: real {c['real']}, synthetic {c['synthetic']}; TV fix kind {c['tv_kind']}, "
                  f"TV diff class {c['tv_class']}, real records covered {c['covered']}/{c['real']} ({c['cover_share']:.0%}); "
                  f"**{'PASS' if c['pass'] else 'FAIL'}**", ""]
    lines += ["## Re-run", "", "`./dk dataset shape-synth --n N --jobs 4` in a worktree (taskset -c 4-15), then",
              "`./dk dataset shape-synth --check`. The cache makes a larger N continue where the last run stopped.", ""]
    (out / "CARD.md").write_text("\n".join(lines))
    names = sorted(p.name for p in out.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text("".join(f"{hashlib.sha256((out / n).read_bytes()).hexdigest()}  {n}\n" for n in names))


def run_check(cfg: Config, args) -> int:
    from .llmtry import data_dir
    out = out_dir(cfg, args)
    real_p = Path(args.real).expanduser() if args.real else data_dir(cfg) / "datasets" / "shape-fixes" / "v0" / "fixes.jsonl"
    if not (out / "pairs.jsonl").is_file() or not real_p.is_file():
        print(f"error: need {out / 'pairs.jsonl'} and {real_p}")
        return USAGE
    real = [json.loads(x) for x in real_p.read_text().splitlines() if x.strip()]
    synth = [json.loads(x) for x in (out / "pairs.jsonl").read_text().splitlines() if x.strip()]
    c = check(real, synth)
    c["real_path"] = str(real_p)
    print(f"real {c['real']} records, synthetic {c['synthetic']} pairs")
    print(f"  TV fix kind {c['tv_kind']:.3f} (<= {PASS_TV}), TV diff class {c['tv_class']:.3f} (<= {PASS_TV}), "
          f"covered {c['covered']}/{c['real']} = {c['cover_share']:.0%} (>= {PASS_COVER:.0%}); exact kind-set {c['exact_set_share']:.0%}")
    for k in sorted(set(c["kinds_real"]) | set(c["kinds_synth"])):
        r, s = c["kinds_real"].get(k, 0), c["kinds_synth"].get(k, 0)
        print(f"  {k:18} real {r:3} ({r / max(1, sum(c['kinds_real'].values())):5.1%})  synth {s:5} "
              f"({s / max(1, sum(c['kinds_synth'].values())):5.1%})")
    print(f"  diff class real {c['classes_real']}  synth {c['classes_synth']}")
    print(f"verdict: {'PASS' if c['pass'] else 'FAIL'}")
    (out / "check.json").write_text(json.dumps(c, indent=1) + "\n")
    return OK if c["pass"] else 1


def configure(sub) -> None:
    s = sub.add_parser("shape-synth", help="synthetic shape-fix pairs (inverse table moves on exact functions, 1-3 "
                       "stacked, compiler- and judge-checked) + single-step agent-policy examples (grid losers, STOP from "
                       "dead ends) -> llm-data/datasets/shape-synth/v0; --check: the T35 distribution check")
    s.add_argument("--n", type=int, default=1000, help="pairs to make (default 1000)")
    s.add_argument("--per-fn", type=int, default=2, help="pairs per function at most (default 2)")
    s.add_argument("--seed", type=int, default=20261106)
    s.add_argument("--only", default="", help="units whose name contains one of these (comma-separated)")
    s.add_argument("--min-size", type=lambda x: int(x, 0), default=0x20, help="smallest function (default 0x20)")
    s.add_argument("--max-size", type=lambda x: int(x, 0), default=0x800, help="largest function (default 0x800)")
    s.add_argument("--stop-share", type=float, default=0.25, help="STOP share of the policy examples (default 0.25)")
    s.add_argument("--stop-variants", type=int, default=8, help="STOP examples per dead end (loser histories; default 8)")
    s.add_argument("--fold-draw", type=float, default=FOLD_DRAW,
                   help=f"share of fold draws: the next function with a foldable local, its paths start with a fold "
                        f"(T38 targeted draw; default {FOLD_DRAW}, 0 = T37's draw)")
    s.add_argument("--jobs", type=int, default=4, help="parallel functions (one compile each at a time; default 4)")
    s.add_argument("--until", default="", help="stop starting new functions at HH:MM")
    s.add_argument("--out", default="", help="output directory (default llm-data/datasets/shape-synth/v0)")
    s.add_argument("--check", action="store_true", help="only the distribution check of pairs.jsonl vs the real records")
    s.add_argument("--probe", type=int, default=0, metavar="N",
                   help="only measure how often the table's local folds reverse, on the first N input functions")
    s.add_argument("--probe-show", type=int, default=10, help="with --probe: failures to list")
    s.add_argument("--real", default="", help="with --check: the real records (default shape-fixes/v0/fixes.jsonl)")
