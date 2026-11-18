"""The RL stage's reward gate and state renderer (council RL1 rows 1 and 3): the one place an RL run, and every
baseline compared with it (enumerator, expert iteration, closure), turns a turn into a prompt and a reward.

  render(fn, definition, score, history, examples, k)   the state exactly as the SFT data renders it (T40
      data/train.jsonl, llm-agent's build_prompt): no oracle section, the score line without oracle numbers
      (percent and rows only), at most k retrieved examples with their oracle lines dropped
  check_action(act, new_text, seen)    the allow-list: a structured action only from shapesynth's table; one
      that is unparsed, outside the table, does not apply or repeats a state already seen is malformed
  tier(start, new)    pure objdiff: exact > fewer shape rows > fewer rows; never the oracle's sat/wrong
  gate(before, after, symbol, tier, ...)    reward > 0 only if the judge passes, review() is CLEAN after the
      deletion twin, and the declarative twin does not reproduce the code:
      - deletion twin: a flagged construct whose removal leaves the code byte-identical is stripped and the
        stripped text is paid (no penalty);
      - declarative twin (opt-in, `declarative=True`): an added cast `(T)x` of a local or parameter, x retyped T
        and the casts dropped: the same code means the cast stands in for a declaration: -1, to the audit queue;
      - JUDGMENT pays 0 and goes to the audit queue; REJECT pays 0.
`measure(text) -> Eval | None` (`unit_measure`: the closer's compile and score, no oracle) runs the twins and
review()'s same-type cast probe; without it (synthetic fixtures) the gate is the static checks only.
"""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field

from ...core import cbody
from . import judge
from .judge import CLEAN, JUDGMENT, REJECT

_MEASURES = __import__("itertools").count(1)
REWARD = {"exact": 1.0, "shape_rows": 0.3, "rows": 0.1, "none": 0.0, "stop": 0.0, "malformed": -0.2,
          "declarative-twin": -1.0}
TIERS = ("exact", "shape_rows", "rows")
DELETABLE = ("preprocessor", "false-condition", "self-assign", "identity", "void0")


def allowed_transforms() -> frozenset[str]:
    from .shapesynth import SHAPE_TABLE
    return frozenset(SHAPE_TABLE)


# ---- the state prompt (SFT parity) -------------------------------------------------------------------------

def plain(score):
    """The score as the SFT states show it: percent and rows, no oracle counts."""
    from .llmagent import Score
    return Score(score.compiled, score.percent, (), list(score.rows or []), error=score.error)


def render(fn: str, definition: str, score, history: list[tuple[str, str]], examples: list[dict] = (),
           k: int = 1, window: str | None = None) -> list[dict]:
    """The chat messages for one state, as T40's SFT data renders them (build_data.py: build_prompt with no
    oracle, action "both", 0 or 1 example). The example pool is the SFT adapter's (shape-synth v4 pairs minus
    the leakage drop), never shape-fixes v0 (the eval fixes; their records carry oracle summaries)."""
    from .llmagent import build_prompt
    picked = [{**r, "oracle": None} for r in list(examples)[:max(0, k)]]
    return build_prompt(fn, definition, plain(score), "", picked, [tuple(h) for h in history], "both", window=window)


def outcome_line(new, best, better: bool, strictly: bool) -> str:
    """A history entry's result, in the SFT states' words."""
    line = plain(new).line()
    if not new.compiled:
        return line
    if better:
        return line + (" (kept: the new best)" if strictly else " (kept: as good)")
    return line + f" (worse than the best {best.percent:.2f}%: reverted)"


# ---- actions -------------------------------------------------------------------------------------------------

def check_action(act, new_text: str | None, seen: set[str]) -> str | None:
    """None, or why the action counts as malformed."""
    if act.error:
        return act.error
    if act.kind == "stop":
        return None
    if act.kind == "structured" and act.transform not in allowed_transforms():
        return f"transform {act.transform} is not in the shape table"
    if new_text is None:
        return "the action does not apply"
    if new_text in seen:
        return "repeats a state already seen"
    return None


# ---- tiers (pure objdiff) --------------------------------------------------------------------------------------

def tier(start, new) -> str:
    """exact | shape_rows | rows | none, from two closer Evals (percent, shape_rows, rows)."""
    if new is None:
        return "none"
    if new.percent == 100.0:
        return "exact"
    if new.shape_rows < start.shape_rows:
        return "shape_rows"
    if new.shape_rows == start.shape_rows and new.rows < start.rows:
        return "rows"
    return "none"


# ---- twins ----------------------------------------------------------------------------------------------------

def _statement_end(code: str, i: int) -> int | None:
    """The end (exclusive) of the statement starting at code[i]: a braced block or up to its `;`."""
    while i < len(code) and code[i].isspace():
        i += 1
    if i >= len(code):
        return None
    if code[i] == "{":
        k = judge._close_brace(code, i)
        return k + 1 if k < len(code) else None
    k = code.find(";", i)
    return k + 1 if k >= 0 else None


def _whole_line(src: str, a: int, b: int) -> tuple[int, int]:
    """(a, b) widened to the whole line(s) when nothing else is on them."""
    lo, hi = src.rfind("\n", 0, a) + 1, src.find("\n", b)
    hi = len(src) if hi < 0 else hi
    if not src[lo:a].strip() and not src[b:hi].strip():
        return lo, min(len(src), hi + 1)
    return a, b


def strip_devices(src: str, span: tuple[int, int], kinds: set[str]) -> str:
    """src with every deletable in-span device of `kinds` removed (the deletion twin's text)."""
    code = judge.code_only(src[:span[1]]) + src[span[1]:]
    edits = []
    for kind, a, b in judge.span_devices(src, span):
        if kind not in kinds or kind not in DELETABLE:
            continue
        if kind == "preprocessor":
            b = src.find("\n", a)
            b = len(src) if b < 0 else b
        elif kind == "void0":
            m = re.match(r"\s*;", code[b:])
            if not m or code[:a].rstrip()[-1:] not in (";", "{", "}"):
                continue
            b += m.end()
        elif kind == "false-condition":
            if not code[a:b].endswith(")"):
                continue
            end = _statement_end(code, b)
            if end is None or re.match(r"\s*else\b", code[end:]):
                continue
            b = end
        edits.append(_whole_line(src, a, b) if kind != "identity" else (a, b))
    edits.sort()
    keep, last = [], -1
    for a, b in edits:
        if a >= last:
            keep.append((a, b, ""))
            last = b
    return cbody.apply_edits(src, keep) if keep else src


def _no_effect_casts(before_src: str, after_src: str, symbol: str, measure, base_code: str) -> str:
    """after_src with each added same-type cast whose removal leaves the code identical dropped."""
    b_body, a_body, _ = judge.bodies(before_src, after_src, symbol)
    if a_body is None:
        return after_src
    edits = []
    for e in judge.added_casts(before_src, b_body, after_src, a_body):
        probe = judge.same_type_probe(after_src, e)
        if probe is None or measure(probe) is None:
            continue                                    # not its own type (or not checkable)
        inner = after_src[e.kids[0].start:e.kids[0].end]
        m = measure(cbody.apply_edits(after_src, [(e.start, e.end, f"({inner})")]))
        if m is not None and m.code == base_code:
            edits.append((e.start, e.end, f"({inner})"))
    try:
        return cbody.apply_edits(after_src, edits) if edits else after_src
    except ValueError:                                  # nested casts: leave them
        return after_src


def declarative_twins(before_src: str, after_src: str, symbol: str) -> list[tuple[str, str, str]]:
    """(variable, type, twin text) per added scalar cast `(T)x` of a plain local or parameter declared with
    another type: x declared T, and every `(T)x` in the definition dropped."""
    b_body, a_body, a_span = judge.bodies(before_src, after_src, symbol)
    if a_body is None or a_span is None:
        return []
    decls = {n: (d, s) for n, d, s in judge.declared(a_body)}
    brace = after_src.find("{", a_span[0], a_span[1])
    head_lo = after_src.find("(", a_span[0], brace)
    out, seen = [], set()
    for e in judge.added_casts(before_src, b_body, after_src, a_body):
        x = e.kids[0]
        while x.kind == "paren" and x.kids:
            x = x.kids[0]
        t = " ".join(after_src[e.type_span[0]:e.type_span[1]].split())
        if x.kind != "name" or re.search(r"[*&()\[\]]", t) or (x.name, t) in seen:
            continue
        seen.add((x.name, t))
        if x.name in decls:
            d, s = decls[x.name]
            if len(s.decl.declarators) != 1 or d.ptr or d.array:
                continue
            old = after_src[s.decl.type_start:s.decl.type_end]
            if judge._norm(old) == judge._norm(t) or "static" in old:
                continue
            edits = [(s.decl.type_start, s.decl.type_end, t)]
        elif head_lo >= 0:
            m = None
            for m in re.finditer(rf"(?<=[(,])([^(),]*?)\b{re.escape(x.name)}\s*(?=[,)])", after_src[head_lo:brace]):
                pass
            if m is None or "*" in m.group(1) or "&" in m.group(1) or not m.group(1).strip():
                continue
            if judge._norm(m.group(1)) == judge._norm(t):
                continue
            a = head_lo + m.start(1)
            edits = [(a, a + len(m.group(1)), (" " if m.group(1)[:1].isspace() else "") + t + " ")]
        else:
            continue
        for c in judge._casts(after_src, a_body):
            inner = c.kids[0]
            while inner.kind == "paren" and inner.kids:
                inner = inner.kids[0]
            if inner.kind == "name" and inner.name == x.name and \
                    judge._norm(after_src[c.type_span[0]:c.type_span[1]]) == judge._norm(t):
                edits.append((c.start, c.end, x.name))
        try:
            out.append((x.name, t, cbody.apply_edits(after_src, edits)))
        except ValueError:
            continue
    return out


# ---- the gate -------------------------------------------------------------------------------------------------

@dataclass
class Gate:
    reward: float
    tier: str
    category: str                       # reject | needs-judgment | clean (after the deletion twin)
    flags: list[str] = field(default_factory=list)
    text: str = ""                      # the text paid (the deletion twin's when it applied)
    audit: bool = False                 # to the audit queue (needs judgment, or a declarative twin)
    twins: dict = field(default_factory=dict)


def gate(before_src: str, after_src: str, symbol: str, tier_name: str, *, judge_reasons: list[str] | None = None,
         measure=None, forbidden=(), declarative: bool = False) -> Gate:
    """The reward of one outcome (see the module doc). `tier_name` from `tier()`, or "malformed"/"stop".
    `judge_reasons`: the closer's final check (unit scores, hand-link) when the caller has it; else the
    judge's source checks run here. `declarative`: run the declarative twin. Off by default: RL2-cpu hit its
    kill mark (40 of 49 twin compiles failed, a retyped parameter renames a C++ symbol); review()'s
    signedness-cast flag keeps those casts at 0 without it."""
    if tier_name not in TIERS:
        return Gate(REWARD.get(tier_name, 0.0), tier_name, CLEAN, text=after_src)
    probe = None
    if measure is not None:
        def probe(text: str) -> float | None:
            m = measure(text)
            return None if m is None else m.percent
    reasons = judge_reasons if judge_reasons is not None else \
        judge.judge(before_src, after_src, None, None, None, None, symbol, forbidden, probe=probe).reasons
    if reasons:
        return Gate(0.0, tier_name, REJECT, list(reasons), after_src)
    text, twins = after_src, {}
    reject, judgment = judge.review_split(before_src, text, symbol, forbidden, probe)
    base = measure(after_src) if measure is not None else None
    if (reject or judgment) and base is not None:
        kinds = {k for k, (flag, _rej) in judge.SPAN_DEVICES.items() if flag in reject + judgment}
        twin = strip_devices(text, judge.bodies(before_src, text, symbol)[2] or (0, 0), kinds) if kinds else text
        if any(f.startswith("redundant cast to its own type") for f in judgment):
            twin = _no_effect_casts(before_src, twin, symbol, measure, base.code)
        if twin != text:
            m = measure(twin)
            same = m is not None and m.code == base.code
            twins["deletion"] = "stripped" if same else ("does not compile" if m is None else "changes the code")
            if same:
                text = twin
                reject, judgment = judge.review_split(before_src, text, symbol, forbidden, probe)
    category = REJECT if reject else JUDGMENT if judgment else CLEAN
    flags = reject + judgment
    if base is not None and declarative:
        results = []
        for name, t, twin in declarative_twins(before_src, text, symbol):
            m = measure(twin)
            r = "does not compile" if m is None else "same code" if m.code == base.code else "other code"
            results.append({"variable": name, "type": t, "result": r})
            if r == "same code":
                flags = flags + [f"declarative twin: {name} declared {t} gives the same code without the cast"]
        if results:
            twins["declarative"] = results
        if any(r["result"] == "same code" for r in results):
            return Gate(REWARD["declarative-twin"], tier_name, REJECT, flags, text, True, twins)
    if category != CLEAN:
        return Gate(0.0, tier_name, category, flags, text, category == JUDGMENT, twins)
    return Gate(REWARD[tier_name], tier_name, CLEAN, [], text, False, twins)


def unit_measure(cfg, unit: dict, symbol: str):
    """(measure, close): `measure(text) -> closecmd.Eval | None` with the unit's compiler (table moves, no
    oracle), memoized by text; `close()` removes its work directory."""
    import itertools
    import threading
    from .closecmd import Closer, Options
    closer = Closer(cfg, unit, symbol, Options(moves="table", jobs=1), log=lambda s: None)
    closer.work = closer.work.with_name(f"{closer.work.name}_m{next(_MEASURES)}")   # several per function: own dirs
    count, memo, lock = itertools.count(1), {}, threading.Lock()

    def measure(text: str):
        if text in memo:
            return memo[text]
        with lock:
            slot = next(count)
        e = closer.evaluate(text, slot)
        memo[text] = e
        return e

    return measure, lambda: shutil.rmtree(closer.work, ignore_errors=True)
