"""dk llm-agent: a local model (Ollama) edits one near-miss function over several turns, one small action
per turn, and dk applies, compiles and scores each one (LoRA 2's harness, the plan's B and council item 6).

  ./dk llm-agent FN [--unit U] [--model M] [--turns 8] [--examples 3] [--action edit|structured|both]
  ./dk llm-agent --set FILE[#KEY=VALUE] … [--run NAME] [--resume] [--until HH:MM] [--jobs N]
  ./dk llm-agent … --render sft [--examples 0]   the state as the LoRA-2 SFT data renders it (rlgate.render)
  ./dk llm-agent --report RUN

The state each turn (a single-step prompt, no chat history): the function's current definition, its objdiff
rows (target | ours), the oracle summary (values coloured as the target, the solver's rules left, `regs
--reach` and `regs --explain --oracle` short forms when DK_ORACLE is configured), up to `--examples` real
fixes retrieved from llm-data/datasets/shape-fixes/v0 (leave-one-out: never the same function or asm shape;
comments stripped, so only project code, diff rows and tool output), and the last edits with their scores.
The model answers with ONE action:

  edit        <<<<<<< SEARCH / ======= / >>>>>>> REPLACE: SEARCH must occur exactly once in the current
              definition, else the reply is malformed (nothing is guessed or repaired)
  structured  {"transform": "<rewrites.RULES name>", "anchor": <line number in the function | line text>}:
              the rewrite table's first candidate at that line
  stop        STOP: <blocker label>  (BLOCKERS)

Each applied action is compiled privately and scored like `try --oracle` (closecmd.Closer's key: exact, rows
that differ in more than registers, the solver's rules satisfied, wrong values, fuzzy %); the best state is
kept and the next turn starts from it (ties take the newer). A 100% result goes through the judge (the
closer's final check: real types written, hand-linked identical, nothing else worse, no device); a judged
exact one is written as a candidate (llm-data/candidates/<date>/ for the repo's own function, the run's
`exact/` for a snapshot) and never landed here.

`--set` runs a list of functions: a JSON list or {"functions": [...]} (llm-data/gate-lora2/gate.json,
near-sample lists) or a jsonl of shape-fix records. A record with `parent`+`before` (shape fix) or
`commit`+`before` (a gate snapshot) is worked in a snapshot tree (that commit's include/, llmfixes.Snapshots)
from its BEFORE version, so the score is "did the agent reach exact"; any other row is the repo's function
now. `#KEY=VALUE` keeps the rows whose KEY is VALUE (`gate.json#why=gave-up`).

Logs (outside the repo): llm-data/agent/<run>/turns.jsonl (one row per turn: the reply, action, malformed
and why, applied, compiled, score, tokens), state.json (--resume), summary.json. Rates: malformed = replies
with no single valid action of an allowed kind (an edit whose SEARCH is not exactly once in the function
counts); apply rate = edit/structured actions that produced a new source, over all edit/structured actions
(malformed ones included).
"""

from __future__ import annotations

import argparse
import dataclasses
import difflib
import hashlib
import json
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from ...core.command import MISMATCH, OK, USAGE, Command, hint
from ...core.config import Config

DEFAULT_MODEL = "qwen3-coder:30b"
DEFAULT_TURNS = 8
DEFAULT_EXAMPLES = 3
HISTORY = 4                 # the last k (edit, score) pairs in the state
ACTION_MODES = ("edit", "structured", "both")
BLOCKERS = {
    "register-only": "only register numbers differ; the CPU closer (`close`, `regs --reach`) should take it",
    "header-type": "needs a declaration or type changed in a header, outside this function",
    "data-or-literal": "a literal pool, data or relocation order difference, not this function's code",
    "inline-helper": "the target calls (or inlines) a helper this source does not have",
    "other-function": "needs another function changed (emission order, a callee's signature)",
    "no-idea": "no faithful edit left to try",
}

RULES = """You are matching a decompilation: C/C++ that Metrowerks CodeWarrior for GameCube (mwcc, optimized)
compiles to exactly the target's PowerPC instructions. You fix ONE function that is almost right.
- Keep the code faithful: the code the original programmers wrote, with the declarations it already uses.
- Never use: volatile, #pragma, asm, the register keyword, pad/dummy variables or members, invented
  switch cases, casts to a type's own type, or field access as pointer plus offset.
- Do not add functions or globals. C++98 (C for a .c unit): no nullptr, auto, constexpr, lambdas.
- Each turn you make one small change; it is compiled and scored, and you see the result next turn."""

EDIT_FORMAT = """An edit (SEARCH must be copied exactly, character for character, from the current definition and
occur in it once; REPLACE is what it becomes):
<<<<<<< SEARCH
old lines
=======
new lines
>>>>>>> REPLACE"""

STRUCTURED_FORMAT = """A transform from the rewrite table, applied at one line of the current definition (anchor: its line
number, 1 = the definition's first line, or the line's exact text):
{"transform": "<id>", "anchor": 7}"""

STOP_FORMAT = "STOP: <label>   when no faithful change is left; label one of: " + ", ".join(BLOCKERS)

_THINK = re.compile(r"<think>.*?</think>", re.S)
_EDIT = re.compile(r"<{5,9} ?SEARCH[^\n]*\n(.*?)\n?={5,9}[^\n=]*\n(.*?)\n?>{5,9} ?REPLACE", re.S)
_STOP = re.compile(r"^\s*(?:\*\*)?STOP(?:\*\*)?\s*[:\-]\s*`?([A-Za-z][\w-]*)`?", re.M)


# ---- actions ------------------------------------------------------------------------------------------

@dataclass
class Action:
    kind: str | None                    # edit | structured | stop | None (malformed)
    old: str = ""
    new: str = ""
    transform: str = ""
    anchor: object = None
    pick: int = 0
    label: str = ""
    error: str = ""                     # why the reply is malformed ("" = well formed)

    def summary(self, width: int = 240) -> str:
        if self.kind == "edit":
            short = lambda t: " ".join(t.split())[:width // 2]
            return f"edit `{short(self.old)}` -> `{short(self.new)}`"
        if self.kind == "structured":
            return f"transform {self.transform} at {self.anchor!r}"
        if self.kind == "stop":
            return f"STOP {self.label}"
        return "malformed reply"


def _json_actions(text: str) -> list[dict]:
    out, dec, i = [], json.JSONDecoder(), 0
    while True:
        i = text.find("{", i)
        if i < 0:
            return out
        try:
            obj, end = dec.raw_decode(text, i)
        except ValueError:
            i += 1
            continue
        if isinstance(obj, dict) and "transform" in obj:
            out.append(obj)
        i = end


def parse_action(reply: str, allowed: str = "both") -> Action:
    """The one action in a reply, or an Action with `error` set (malformed). Nothing is repaired."""
    from .rewrites import BY_NAME
    text = _THINK.sub("", reply or "")
    edits = _EDIT.findall(text)
    structs = _json_actions(text)
    stops = _STOP.findall(text)
    found = (["edit"] * len(edits)) + (["structured"] * len(structs)) + (["stop"] * len(stops))
    if not found:
        return Action(None, error="no action in the reply")
    if len(found) > 1:
        return Action(None, error=f"{len(found)} actions in one reply (one per turn)")
    kind = found[0]
    if kind == "stop":
        label = stops[0].lower()
        if label not in BLOCKERS:
            return Action(None, error=f"unknown STOP label {label!r}")
        return Action("stop", label=label)
    if allowed != "both" and kind != allowed:
        return Action(None, error=f"a {kind} action, but only {allowed} actions are allowed")
    if kind == "edit":
        old, new = edits[0]
        if not old.strip():
            return Action(None, error="empty SEARCH")
        if old == new:
            return Action(None, error="SEARCH and REPLACE are the same")
        return Action("edit", old=old, new=new)
    obj = structs[0]
    name = str(obj.get("transform", ""))
    if name not in BY_NAME:
        return Action(None, error=f"unknown transform {name!r}")
    anchor = obj.get("anchor", obj.get("line"))
    if not isinstance(anchor, (int, str)) or isinstance(anchor, bool) or (isinstance(anchor, str) and not anchor.strip()):
        return Action(None, error="no anchor line")
    pick = obj.get("pick", 0)
    return Action("structured", transform=name, anchor=anchor, pick=pick if isinstance(pick, int) and pick >= 0 else 0)


def apply_edit(src: str, span: tuple[int, int], old: str, new: str) -> tuple[str | None, str]:
    """(new source, "") or (None, why): OLD must occur exactly once in the definition src[span]."""
    a, b = span
    body = src[a:b]
    n = body.count(old)
    if n != 1:
        return None, ("SEARCH is not an exact substring of the current definition" if n == 0
                      else f"SEARCH occurs {n} times in the definition")
    return src[:a] + body.replace(old, new, 1) + src[b:], ""


def anchor_line(src: str, span: tuple[int, int], anchor) -> tuple[int | None, str]:
    """The anchor as a 0-based line of the file, or (None, why)."""
    a, b = span
    first = src.count("\n", 0, a)
    lines = src[a:b].split("\n")
    if isinstance(anchor, int) or (isinstance(anchor, str) and anchor.strip().isdigit()):
        k = int(anchor)
        if not 1 <= k <= len(lines):
            return None, f"anchor line {k} is outside the definition (1-{len(lines)})"
        return first + k - 1, ""
    want = " ".join(str(anchor).split())
    hits = [i for i, ln in enumerate(lines) if " ".join(ln.split()) == want]
    if not hits:
        hits = [i for i, ln in enumerate(lines) if want and want in " ".join(ln.split())]
    if len(hits) != 1:
        return None, ("anchor text matches no line of the definition" if not hits
                      else f"anchor text matches {len(hits)} lines")
    return first + hits[0], ""


# ---- retrieval: real fixes from shape-fixes v0 ------------------------------------------------------------

def no_comments(code: str) -> str:
    """The definition without comments (they may hold notes); lines that held only a comment dropped."""
    out = []
    for line in re.sub(r"/\*.*?\*/", "", code, flags=re.S).split("\n"):
        kept = re.sub(r"//.*", "", line).rstrip()
        if kept.strip() or not line.strip():
            out.append(kept)
    return "\n".join(out)


def _row_tokens(rows: list[str]) -> set[str]:
    toks = set()
    for r in rows:
        for side, word in enumerate(re.findall(r"\b([a-z][a-z.]{1,8})\b", r)[:2]):
            toks.add(word)
        if " op " in r or "rep" in r:
            toks.add("~shape")
        if "arg" in r:
            toks.add("~arg")
    return toks


def fix_hunk(before: str, after: str) -> tuple[str, str]:
    """The smallest whole-line SEARCH/REPLACE that turns `before` into `after`, SEARCH unique in before."""
    b, a = before.split("\n"), after.split("\n")
    sm = difflib.SequenceMatcher(None, b, a, autojunk=False)
    ops = [op for op in sm.get_opcodes() if op[0] != "equal"]
    if not ops:
        return "", ""
    i1, i2, j1, j2 = ops[0][1], ops[-1][2], ops[0][3], ops[-1][4]
    while True:
        old = "\n".join(b[i1:i2])
        if (old and before.count(old) == 1) or (i1 == 0 and i2 == len(b)):
            break
        i1, j1 = max(0, i1 - 1), max(0, j1 - 1)
        if i2 < len(b):
            i2, j2 = i2 + 1, j2 + 1
    return "\n".join(b[i1:i2]), "\n".join(a[j1:j2])


class Examples:
    """The shape-fix records as retrieval examples (leave-one-out by function and asm shape)."""

    def __init__(self, records: list[dict]):
        self.records = [r for r in records if r.get("before") and r.get("after")]
        self.tokens = [_row_tokens(r.get("rows") or []) for r in self.records]

    @classmethod
    def load(cls, path: Path) -> "Examples":
        if not path.is_file():
            return cls([])
        return cls([json.loads(x) for x in path.read_text().splitlines() if x.strip()])

    def top(self, function: str, asm_shape: str | None, rows: list[str], register_only: bool, k: int) -> list[dict]:
        if k <= 0:
            return []
        q = _row_tokens(rows)
        scored = []
        for rec, toks in zip(self.records, self.tokens):
            if rec["function"] == function or (asm_shape and rec.get("asm_shape") == asm_shape):
                continue
            sim = len(q & toks) / max(1, len(q | toks))
            sim += 0.25 if (rec.get("diff_class") == "register") == register_only else 0.0
            scored.append((-sim, rec["function"], rec))
        return [r for _s, _f, r in sorted(scored, key=lambda x: (x[0], x[1]))[:k]]


def render_example(k: int, rec: dict, max_chars: int = 2400) -> str:
    before, after = no_comments(rec["before"]), no_comments(rec["after"])
    old, new = fix_hunk(before, after)
    shown = before
    if len(shown) > max_chars and old:                  # a long function: the lines around the fix only
        lines, at = before.split("\n"), before[:before.find(old)].count("\n")
        lo, hi = max(0, at - 6), min(len(lines), at + old.count("\n") + 7)
        shown = "\n".join((["..."] if lo else []) + lines[lo:hi] + (["..."] if hi < len(lines) else []))
    ora = rec.get("oracle") or {}
    ora_text = "\n".join(x for x in (ora.get("explain", "").split("\n")[:10] + ora.get("reach", "").split("\n")[:5]) if x)
    rows = (rec.get("rows") or [])[:20]
    return "\n".join([f"### Example {k}: {rec.get('diff_class', '?')} miss at {rec.get('before_percent', 0):.2f}%, "
                      f"fixed by: {', '.join(rec.get('fix_kind') or [])}",
                      "Before:", "```cpp", shown, "```", "Diff (target | ours):", *rows,
                      *(["Oracle:", ora_text[:900]] if ora_text else []),
                      "The fix that made it exact:", "<<<<<<< SEARCH", old, "=======", new, ">>>>>>> REPLACE", ""])


# ---- the state and the prompt ------------------------------------------------------------------------------

@dataclass
class Score:
    compiled: bool
    percent: float = 0.0
    key: tuple = ()
    rows: list[str] = field(default_factory=list)
    shape_rows: int | None = None       # rows that differ in more than registers (with the oracle)
    right: int | None = None            # values coloured as the target (oracle)
    known: int | None = None
    error: str = ""
    handle: object = None               # the env's own result (closecmd.Eval)

    def line(self) -> str:
        if not self.compiled:
            return "does not compile: " + self.error[:200]
        bits = [f"{self.percent:.2f}%", f"{sum(1 for r in self.rows if r.strip() != '...' and not r.startswith('  …'))} rows shown differ"]
        if self.shape_rows is not None:
            bits.append(f"{self.shape_rows} rows differ in more than registers")
        if self.right is not None and self.known:
            bits.append(f"values coloured as the target {self.right}/{self.known}")
        return ", ".join(bits)


def build_prompt(fn: str, definition: str, score: Score, oracle_text: str, examples: list[dict],
                 history: list[tuple[str, str]], allowed: str, window: str | None = None) -> list[dict]:
    """`window` (stacked --window, giants): the definition's changed regions with context instead of the whole
    definition and its numbered copy; `// ... lines a-b elided ...` marks the rest."""
    from .rewrites import RULES as TABLE
    forms = []
    if allowed in ("edit", "both"):
        forms.append(EDIT_FORMAT)
    if allowed in ("structured", "both"):
        table = "\n".join(f"  {r.name}: {r.idiom}" for r in TABLE)
        forms.append(STRUCTURED_FORMAT + "\nTransforms:\n" + table)
    forms.append(STOP_FORMAT)
    system = RULES + "\n\nReply with exactly ONE action and nothing else, in one of these forms:\n\n" + "\n\n".join(forms)
    parts = []
    if examples:
        parts += ["## Real fixes of other functions in this project", ""] + [render_example(i + 1, r) for i, r in enumerate(examples)]
    parts += [f"## {fn} now: {score.line()}", ""]
    if window is None:
        numbered = "\n".join(f"{i + 1:3} | {ln}" for i, ln in enumerate(definition.split("\n")))
        parts += ["```cpp", definition, "```", "",
                  "The same definition with line numbers (for an anchor; never copy the numbers into SEARCH):", numbered, ""]
    else:
        parts += ["A window of the definition: the regions to change, with context. `// ... lines a-b elided ...` "
                  "stands for unchanged lines; SEARCH text must be copied from the shown lines.", "```cpp", window, "```", ""]
    parts += ["Diff (target | ours; `arg` = operands differ, `op`/`rep` = another instruction, `+`/`-` = missing):",
              *(score.rows or ["  (no rows)"]), ""]
    if oracle_text:
        parts += ["Oracle:", oracle_text, ""]
    if history:
        parts += ["Your last actions and their results (each was tried on the best definition so far):"]
        parts += [f"- {a} -> {r}" for a, r in history[-HISTORY:]] + [""]
    parts.append("Your one action:")
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(parts)}]


# ---- the environment: compile, score, judge (real) ---------------------------------------------------------

ORACLE_LOCK = threading.Lock()


class CompilerEnv:
    """One function in one tree (the repo, or a snapshot of a past commit's headers): closecmd.Closer
    compiles and scores (`try --oracle`'s key), its final check judges a 100% result."""

    def __init__(self, cfg: Config, unit: dict, symbol: str, src: str, jobs: int = 1):
        from . import oracle
        from .closecmd import Closer, Options
        self.cfg, self.unit, self.symbol, self.src = cfg, unit, symbol, src
        self.oracle = oracle.runner(cfg) is not None
        self.closer = Closer(cfg, unit, symbol, Options(moves="oracle" if self.oracle else "table", jobs=jobs),
                             log=lambda s: None)
        self.closer.keep_rows = True
        self.slot = 0
        self.lock = threading.Lock()
        self.before = None

    def definition(self, text: str) -> tuple[int, int] | None:
        from .trycmd import definition
        return definition(text, self.symbol)

    def score(self, text: str) -> Score:
        with self.lock:
            self.slot += 1
            slot = self.slot
        e = self.closer.evaluate(text, slot)
        if e is None:
            from .reorder import compile_variant
            work = self.closer.work / f"err{slot}"
            try:
                obj, err = compile_variant(self.cfg, self.unit, text, work)
            finally:
                shutil.rmtree(work, ignore_errors=True)
            from .llmtry import compiler_errors
            return Score(False, error=(compiler_errors(err) or err[-300:] or "the function is missing") if obj is None
                         else "the function is missing from the object")
        if slot == 1:
            self.closer.diagnose(e)                      # the solver's rules on the start: `sat` in the key
        known = None if e.right is None or e.values is None else len([v for v in e.values if v is not None])
        return Score(True, e.percent, tuple(e.key[:-1]), e.view, e.shape_rows if e.right is not None else None,
                     e.right if e.values is not None else None, (e.right + e.wrong) if e.values is not None else known,
                     handle=e)

    def oracle_text(self, text: str, score: Score) -> str:
        """regs --explain --oracle and regs --reach, short (llmfixes' form), plus the solver's rules left."""
        if not self.oracle:
            return ""
        from .llmfixes import oracle_summary
        try:
            with ORACLE_LOCK:                            # it redirects sys.stdout (process-wide): one at a time
                ora = oracle_summary(self.cfg, self.unit, self.symbol, text) or {}
        except Exception as exc:                         # noqa: BLE001 (a summary never stops the agent)
            return f"(oracle failed: {type(exc).__name__})"
        return "\n".join(x for x in (ora.get("explain", "")[:1500], ora.get("reach", "")[:800]) if x)

    def structured(self, text: str, score: Score, rule: str, line0: int) -> list[str]:
        from . import rewrites
        e = score.handle
        body = self.closer.body(text, getattr(e, "first_line", None))
        if body is None:
            return []
        site = rewrites.Site(text, body, {line0}, not self.closer.c, members=self.closer.members)
        try:
            return [new for _r, _label, new in rewrites.candidates(site, [rewrites.BY_NAME[rule]], groups=False)]
        except Exception:                                # noqa: BLE001 (a rule that fails on a body applies nowhere)
            return []

    def judge(self, text: str, score: Score) -> tuple[bool, list[str], str]:
        if self.before is None:
            self.before, _proof, err = self.closer.unit_score(self.src, "before")
            if self.before is None:
                return False, ["the start does not compile: " + err[-200:]], text
        verdict, final = self.closer.final_check(score.handle, self.src, self.before)
        self.last_review = verdict.review               # (category, flags): only a clean one is a clean exact
        return verdict.ok, verdict.reasons[:6], final

    def close(self) -> None:
        shutil.rmtree(self.closer.work, ignore_errors=True)


# ---- tasks ------------------------------------------------------------------------------------------------

@dataclass
class Task:
    set: str
    unit: str
    function: str
    rev: str | None = None              # snapshot tree at this commit (None: the repo now)
    path: str = ""
    before: str | None = None           # the BEFORE definition (a shape fix's, a gate snapshot's)
    blob: str = ""
    after: str | None = None            # a shape fix's real fix (logged next to the result, never shown)
    asm_shape: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.set}::{self.unit}::{self.function}"


def load_set(spec: str) -> list[Task]:
    path, _, filt = spec.partition("#")
    p = Path(path).expanduser()
    text = p.read_text()
    if p.suffix == ".jsonl":
        rows = [json.loads(x) for x in text.splitlines() if x.strip()]
    else:
        doc = json.loads(text)
        rows = doc.get("functions", []) if isinstance(doc, dict) else doc
    if filt:
        k, _, v = filt.partition("=")
        rows = [r for r in rows if str(r.get(k)) == v]
    name = p.stem + (f"#{filt}" if filt else "")
    out = []
    for r in rows:
        if not isinstance(r, dict) or not r.get("function"):
            continue
        meta = {k: r[k] for k in ("why", "class", "fix_kind", "diff_class", "before_percent", "fuzzy", "commit")
                if k in r}
        if r.get("parent") and r.get("before"):
            out.append(Task(name, r["unit"], r["function"], r["parent"], r.get("path", ""), r["before"],
                            after=r.get("after"), asm_shape=r.get("asm_shape"), meta=meta))
        elif r.get("commit") and r.get("before"):
            out.append(Task(name, r["unit"], r["function"], r["commit"], r.get("source", ""), r["before"],
                            r.get("source_blob", ""), asm_shape=r.get("asm_shape"), meta=meta))
        else:
            out.append(Task(name, r.get("unit", ""), r["function"], asm_shape=r.get("asm_shape"), meta=meta))
    return out


class Trees:
    """Snapshot trees (llmfixes.Snapshots) shared by the tasks of a run."""

    def __init__(self, cfg: Config):
        from .llmfixes import Snapshots
        self.cfg = cfg
        self.snaps = Snapshots(cfg, cfg.root / "build" / "dk" / "llm" / "snap")

    def git(self, *a: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *a], cwd=self.cfg.root, capture_output=True, text=True)

    def open(self, task: Task) -> tuple[Config, dict, str, str]:
        """(cfg of the tree, unit, start source, note)."""
        from .matching import resolve_unit
        from .regs import unit_of_function
        from .trycmd import definition
        unit_name = task.unit or unit_of_function(self.cfg, task.function)
        if not unit_name:
            raise RuntimeError(f"{task.function}: no unit holds it")
        unit = resolve_unit(self.cfg, unit_name)
        if task.rev is None:
            return self.cfg, unit, (self.cfg.root / unit["metadata"]["source_path"]).read_text(), ""
        snap, _tid = self.snaps.at(task.rev)
        path = task.path or unit["metadata"]["source_path"]
        src, note = None, ""
        if task.blob and self.git("cat-file", "-e", task.blob).returncode == 0:
            src = self.git("cat-file", "-p", task.blob).stdout
        if src is None:
            got = self.git("show", f"{task.rev}:{path}")
            if got.returncode:
                raise RuntimeError(f"{path} at {task.rev[:10]}: {got.stderr.strip()[:200]}")
            src = got.stdout
        if task.before is not None:
            span = definition(src, task.function)
            if span is None:
                raise RuntimeError(f"{task.function}: no definition in {path} at {task.rev[:10]}")
            if src[span[0]:span[1]] != task.before:
                src, note = src[:span[0]] + task.before + src[span[1]:], "before spliced in"
        if unit["metadata"]["source_path"] != path:
            unit = {**unit, "metadata": {**unit["metadata"], "source_path": path}}
        return snap, unit, src, note


# ---- the agent loop ----------------------------------------------------------------------------------------

@dataclass
class Settings:
    model: str = DEFAULT_MODEL
    turns: int = DEFAULT_TURNS
    examples: int = DEFAULT_EXAMPLES
    action: str = "both"
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    num_ctx: int = 16384
    num_predict: int = 1024
    seed: int = 1
    render: str = "harness"     # harness: build_prompt with the oracle; sft: rlgate.render (T40 SFT data's form)
    menu: int = 0               # > 0: the legal-move menu in the state, at most this many moves (llmmenu.py)
    grammar: bool = True        # with the menu: the reply constrained to it (Ollama format / llama-server GBNF)

    def options(self, seed: int) -> dict:
        return {"temperature": self.temperature, "top_p": self.top_p, "top_k": self.top_k, "num_ctx": self.num_ctx,
                "num_predict": self.num_predict, "seed": seed, "repeat_penalty": 1.05}


def plain_line(sc: Score) -> str:
    from .rlgate import plain
    return plain(sc).line()


def run_task(task: Task, env, start_src: str, client, st: Settings, examples: Examples, log, until: float = 0.0) -> dict:
    """Up to st.turns actions on one function; the outcome (exact text and verdict when judged exact)."""
    out: dict = {"set": task.set, "unit": task.unit, "function": task.function, "turns": 0, "exact": False}
    start = env.score(start_src)
    if not start.compiled:
        return {**out, "result": "error", "why": "the start does not compile: " + start.error[:200]}
    out["start"] = out["best"] = start.percent
    if start.percent == 100.0:
        ok, reasons, _t = env.judge(start_src, start)
        if ok:
            return {**out, "result": "skip", "why": "already exact"}
    best_text, best = start_src, start
    sft = st.render == "sft"                            # the SFT data's prompt: no oracle section, plain score lines
    oracle_text = "" if sft else env.oracle_text(start_src, start)
    seen = {start_src}
    history: list[tuple[str, str]] = []
    span0 = env.definition(start_src)
    picked = examples.top(task.function, task.asm_shape, start.rows, start.shape_rows == 0, st.examples)
    out["examples"] = [r["function"] for r in picked]
    for turn in range(1, st.turns + 1):
        if until and time.time() >= until:
            return {**out, "result": "stopped", "why": "--until"}
        span = env.definition(best_text) or span0
        definition = best_text[span[0]:span[1]]
        if sft:
            from .rlgate import render
            messages = render(task.function, definition, best, history, picked, len(picked))
        else:
            messages = build_prompt(task.function, definition, best, oracle_text, picked, history, st.action)
        moves, n_all = [], 0
        if st.menu:
            from . import llmmenu
            moves, n_all = llmmenu.turn_menu(env, best_text, best, st.menu)
            messages = llmmenu.add_menu(messages, llmmenu.menu_block(moves, edits=st.action != "structured"))
        t0 = time.time()
        row = {"kind": "turn", "set": task.set, "unit": task.unit, "function": task.function, "turn": turn,
               "model": st.model, "prompt_chars": sum(len(m["content"]) for m in messages), "state_percent": best.percent}
        if st.menu:
            row.update(menu=len(moves), menu_all=n_all)
        try:
            if st.menu and st.grammar:
                r = llmmenu.constrained_chat(client, st.model, messages, st.options(st.seed + 1000 * turn), moves,
                                             BLOCKERS, edits=st.action != "structured")
            else:
                r = client.chat(st.model, messages, st.options(st.seed + 1000 * turn), think=None)
            reply = (r.get("message") or {}).get("content", "")
            row.update(tokens=r.get("eval_count"), prompt_tokens=r.get("prompt_eval_count"))
        except Exception as exc:                         # noqa: BLE001 (a dead server ends this function, not the run)
            log({**row, "error": f"{type(exc).__name__}: {exc}"[:300]})
            return {**out, "result": "error", "why": f"model: {type(exc).__name__}"}
        row["secs_model"] = round(time.time() - t0, 2)
        row["reply"] = reply[:4000]
        act = llmmenu.parse_reply(reply, st.action) if st.menu else parse_action(reply, st.action)
        if st.menu:
            row["on_menu"] = llmmenu.on_menu(act, moves)
        out["turns"] = turn
        row.update(action=act.kind, malformed=bool(act.error), why=act.error or None, applied=False, compiled=None)
        if act.kind == "stop":
            log({**row, "label": act.label})
            return {**out, "result": "stop", "label": act.label}
        if act.error:
            history.append(("(malformed reply)", "rejected: " + act.error))
            log(row)
            continue
        new, why = None, ""
        if act.kind == "edit":
            new, why = apply_edit(best_text, span, act.old, act.new)
            if new is None:                              # the pre-registered rule: an inexact SEARCH is malformed
                row.update(malformed=True, why=why)
                history.append((act.summary(), "rejected: " + why))
                log(row)
                continue
        else:
            line0, why = anchor_line(best_text, span, act.anchor)
            row["transform"] = act.transform
            if line0 is None:
                row.update(malformed=True, why=why)
                history.append((act.summary(), "rejected: " + why))
                log(row)
                continue
            cands = env.structured(best_text, best, act.transform, line0)
            if len(cands) > act.pick:
                new = cands[act.pick]
            else:
                why = f"{act.transform} has no rewrite at that line"
        if new is None:
            row["why"] = why
            history.append((act.summary(), "not applied: " + why))
            log(row)
            continue
        row["applied"] = True
        if new in seen:
            row["repeat"] = True
            history.append((act.summary(), "the same source as an earlier state (not compiled again)"))
            log(row)
            continue
        seen.add(new)
        t1 = time.time()
        sc = env.score(new)
        row.update(compiled=sc.compiled, percent=sc.percent if sc.compiled else None, key=list(sc.key) or None,
                   secs_score=round(time.time() - t1, 2))
        if not sc.compiled:
            history.append((act.summary(), plain_line(sc) if sft else sc.line()))
            log({**row, "error": sc.error[:300]})
            continue
        if sc.percent == 100.0:
            ok, reasons, final = env.judge(new, sc)
            review = getattr(env, "last_review", None) if ok else None
            row.update(judge=reasons, exact=ok, review=list(review) if review else None)
            log(row)
            if ok:
                return {**out, "result": "exact", "exact": True, "best": 100.0, "text": final,
                        "review": list(review) if review else None}
            history.append((act.summary(), "100% but the judge rejects it: " + "; ".join(reasons)[:300]))
            continue
        better = sc.key <= best.key
        history.append((act.summary(), (plain_line(sc) if sft else sc.line()) + (" (kept: the new best)" if better and sc.key < best.key
                                                    else " (kept: as good)" if better else f" (worse than the best {best.percent:.2f}%: reverted)")))
        log({**row, "kept": better})
        if better:
            if not sft and (sc.key < best.key or sc.percent != best.percent):
                oracle_text = env.oracle_text(new, sc)
            best_text, best = new, sc
            out["best"] = max(out["best"], sc.percent)
    return {**out, "result": "miss", "best_text": best_text if best_text != start_src else None}


# ---- runs: logs, state, summary -------------------------------------------------------------------------

def agent_dir(cfg: Config) -> Path:
    from .llmtry import data_dir
    return data_dir(cfg) / "agent"


def summarize(rows: list[dict]) -> dict:
    """Per set: functions, judged exact, stops by label, replies, malformed and apply rates."""
    sets: dict[str, dict] = {}
    for r in rows:
        s = sets.setdefault(r.get("set", "?"), {"functions": 0, "exact": 0, "stops": {}, "replies": 0, "malformed": 0,
                                                "actions": 0, "applied": 0, "compiled": 0, "scored": 0, "errors": 0})
        if r.get("kind") == "turn":
            if r.get("reply") is None:                  # the model call failed: no reply to rate
                continue
            s["replies"] += 1
            s["malformed"] += bool(r.get("malformed"))
            if r.get("action") != "stop":
                s["actions"] += 1
                s["applied"] += bool(r.get("applied"))
                if r.get("compiled") is not None:
                    s["scored"] += 1
                    s["compiled"] += bool(r.get("compiled"))
        elif r.get("kind") == "task":
            s["functions"] += 1
            s["exact"] += bool(r.get("exact"))
            s["errors"] += r.get("result") == "error"
            if r.get("result") == "stop":
                s["stops"][r.get("label")] = s["stops"].get(r.get("label"), 0) + 1
    for s in sets.values():
        s["malformed_rate"] = round(s["malformed"] / s["replies"], 4) if s["replies"] else None
        s["apply_rate"] = round(s["applied"] / s["actions"], 4) if s["actions"] else None
        s["compile_rate"] = round(s["compiled"] / s["scored"], 4) if s["scored"] else None
    return sets


def print_summary(sets: dict, say=print) -> None:
    pct = lambda x: "-" if x is None else f"{100 * x:.1f}%"
    for name, s in sets.items():
        say(f"{name}: exact {s['exact']}/{s['functions']}, malformed {pct(s['malformed_rate'])} ({s['malformed']}/"
            f"{s['replies']} replies), apply {pct(s['apply_rate'])} ({s['applied']}/{s['actions']}), compile "
            f"{pct(s['compile_rate'])}" + (f", stops {s['stops']}" if s["stops"] else "")
            + (f", errors {s['errors']}" if s["errors"] else ""))


def read_rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.is_file() else []


def write_exact(cfg: Config, rundir: Path, task: Task, unit: dict, src: str, res: dict, st: Settings, tree: str) -> str:
    """A judged exact result: the repo's own function -> llm-data/candidates/<date>/ (llm-review, llm-land);
    a snapshot (a past or frozen version) -> <run>/exact/, beside the real fix. Never applied here."""
    from .llmgrind import write_candidate
    from .llmtry import data_dir
    from .tryatoms import size_of
    rec = {"unit": unit["name"], "source": unit["metadata"]["source_path"], "function": task.function,
           "size": size_of(cfg, unit, task.function) or 0, "kind": "agent", "class": task.meta.get("class"),
           "route": "llm-agent", "before": res.get("start"), "after": 100.0, "model": st.model, "before_text": src,
           "text": res["text"], "judge": [], "time": time.strftime("%Y-%m-%d %H:%M"), "set": task.set,
           "turns": res.get("turns"), **({"review": res["review"]} if res.get("review") else {})}
    if task.rev is None:
        return write_candidate(data_dir(cfg) / "candidates" / time.strftime("%Y-%m-%d"), rec)
    rec.update(rev=task.rev, tree=tree, real_fix=task.after)
    return write_candidate(rundir / "exact", rec)


def run(cfg: Config, args: argparse.Namespace) -> int:
    from . import llmgrind, llmtry, oracle
    base = agent_dir(cfg)
    if args.report:
        p = Path(args.report).expanduser()
        p = p if p.is_file() else base / args.report / "turns.jsonl"
        rows = read_rows(p)
        if not rows:
            print(f"error: no log at {p}")
            return USAGE
        print_summary(summarize(rows))
        return OK
    if not args.set and not args.function:
        print("error: name a function or --set FILE")
        return USAGE
    if args.resume and not args.run:
        runs = sorted((d for d in base.glob("*") if (d / "state.json").is_file()), key=lambda d: d.stat().st_mtime)
        if not runs:
            print("error: no run to resume")
            return USAGE
        args.run = runs[-1].name
    try:
        tasks = [t for spec in args.set for t in load_set(spec)] if args.set else \
            [Task("one", args.unit or "", args.function)]
    except (OSError, ValueError) as e:
        print(f"error: {e}")
        return USAGE
    st = Settings(args.model, args.turns, args.examples, args.action, args.temperature, args.top_p, args.top_k,
                  args.num_ctx, args.num_predict, args.seed, getattr(args, "render", "harness"),
                  getattr(args, "menu", 0), getattr(args, "grammar", True))
    name = args.run or f"agent-{time.strftime('%Y%m%d-%H%M%S')}-{re.sub(r'[^A-Za-z0-9.]+', '-', st.model)}"
    rundir = base / name
    rundir.mkdir(parents=True, exist_ok=True)
    state_path = rundir / "state.json"
    state = json.loads(state_path.read_text()) if args.resume and state_path.is_file() else {"run": name, "done": {}}
    fixes = Path(args.fixes).expanduser() if args.fixes else llmtry.data_dir(cfg) / "datasets" / "shape-fixes" / "v0" / "fixes.jsonl"
    examples = Examples.load(fixes)
    client = llmtry.client_for(args)
    if not llmtry.model_ready(client, st.model):
        return USAGE
    until = llmtry._until(args.until)
    lock = threading.Lock()
    fh = (rundir / "turns.jsonl").open("a")

    def log(row: dict) -> None:
        row.setdefault("time", round(time.time(), 2))
        with lock:
            fh.write(json.dumps(row) + "\n")
            fh.flush()

    log({"kind": "run", "run": name, "model": st.model, "settings": dataclasses.asdict(st), "sets": args.set or ["one"],
         "tasks": len(tasks), "examples_from": str(fixes), "examples_available": len(examples.records),
         "oracle": oracle.runner(cfg) is not None})
    todo = [t for t in tasks if t.key not in state["done"]]
    print(f"{len(todo)} of {len(tasks)} function(s) to do; {st.model}, {st.turns} turns, {st.examples} examples, "
          f"actions {st.action}; log {rundir / 'turns.jsonl'}")
    trees = Trees(cfg)
    stop = threading.Event()

    def one(task: Task) -> None:
        if stop.is_set() or (until and time.time() >= until):
            stop.set()
            return
        if not llmgrind.wait_gpu(client, st.model, until, say=print):
            stop.set()
            return
        t0 = time.time()
        env = None
        try:
            tcfg, unit, src, note = trees.open(task)
            env = CompilerEnv(tcfg, unit, task.function, src)
            res = run_task(task, env, src, client, st, examples, log, until)
            if note:
                res["note"] = note
            if res.get("exact"):
                res["id"] = write_exact(cfg, rundir, task, unit, src, res, st, cfg.rel(tcfg.root) if tcfg is not cfg else "")
        except Exception as e:                           # noqa: BLE001 (one bad function must not end the night)
            res = {"set": task.set, "unit": task.unit, "function": task.function, "result": "error",
                   "why": f"{type(e).__name__}: {e}"[:300]}
        finally:
            if env is not None:
                env.close()
        res.pop("text", None)
        res.pop("best_text", None)
        res["secs"] = round(time.time() - t0, 1)
        res["meta"] = task.meta
        log({"kind": "task", **res})
        if res.get("result") == "stopped":
            stop.set()
            return
        with lock:
            state["done"][task.key] = {k: res.get(k) for k in ("result", "start", "best", "turns", "label", "id")}
            tmp = state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, indent=1))
            tmp.replace(state_path)
        print(f"  {task.function}: {res['result']}" + (f" ({res['why']})" if res.get("why") else "")
              + (f" {res.get('start', 0):.2f}% -> {res.get('best', 0):.2f}% in {res.get('turns')} turn(s)" if "start" in res else "")
              + (f", STOP {res['label']}" if res.get("label") else "") + (f", candidate {res['id']}" if res.get("id") else ""))

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            list(pool.map(one, todo))
    finally:
        fh.close()
    sets = summarize(read_rows(rundir / "turns.jsonl"))
    (rundir / "summary.json").write_text(json.dumps(sets, indent=1) + "\n")
    print_summary(sets)
    if stop.is_set():
        print("stopped (--until or GPU wait); the same command with --resume --run " + name + " goes on")
    hint(f"llm-agent --report {name}")
    return OK if any(s["exact"] for s in sets.values()) else MISMATCH


def _configure(p: argparse.ArgumentParser) -> None:
    import os
    from .llmtry import DEFAULT_URL
    p.add_argument("function", nargs="?", default="", help="one function (the repo's current version)")
    p.add_argument("--unit", default="", help="its unit (default: the unit whose splits hold it)")
    p.add_argument("--set", action="append", default=[], metavar="FILE[#KEY=VALUE]",
                   help="a list of functions (JSON list / {functions: [...]}, or shape-fix jsonl; repeatable); "
                        "#KEY=VALUE keeps matching rows (gate.json#why=gave-up)")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default {DEFAULT_MODEL})")
    p.add_argument("--url", default=os.environ.get("DK_OLLAMA", DEFAULT_URL), help="Ollama server (DK_OLLAMA)")
    p.add_argument("--backend", choices=("ollama", "openai"), default="ollama", help="openai: llama.cpp's llama-server")
    p.add_argument("--turns", type=int, default=DEFAULT_TURNS, help=f"actions per function (default {DEFAULT_TURNS})")
    p.add_argument("--examples", type=int, default=DEFAULT_EXAMPLES,
                   help=f"retrieved real fixes in the state (default {DEFAULT_EXAMPLES}; 0: none)")
    p.add_argument("--action", choices=ACTION_MODES, default="both", help="which actions the model may use (default both)")
    p.add_argument("--fixes", default="", help="retrieval records (default llm-data/datasets/shape-fixes/v0/fixes.jsonl)")
    p.add_argument("--run", default="", help="run name: llm-data/agent/<run>/ (default: time and model)")
    p.add_argument("--resume", action="store_true", help="skip the functions the run's state.json has done")
    p.add_argument("--until", default="", help="stop at HH:MM (between turns; --resume goes on)")
    p.add_argument("--jobs", type=int, default=2, help="functions worked at once (default 2)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--num-ctx", type=int, default=16384)
    p.add_argument("--num-predict", type=int, default=1024)
    p.add_argument("--seed", type=int, default=1, help="first seed (turn t: seed + 1000 t)")
    p.add_argument("--render", choices=("harness", "sft"), default="harness",
                   help="sft: the state as the LoRA-2 SFT data renders it (rlgate.render: no Oracle section, plain "
                        "score lines in the state and history; scoring still uses DK_ORACLE); with --examples 0 "
                        "it is the ex0 form")
    p.add_argument("--menu", type=int, nargs="?", const=40, default=0, metavar="N",
                   help="list the rewrite-table moves that apply to the current definition in the state (at most N, "
                        "default 40) and constrain the reply to them, a STOP label or an edit (llmmenu.py)")
    p.add_argument("--no-grammar", dest="grammar", action="store_false",
                   help="with --menu: the menu in the state only, the reply unconstrained")
    p.add_argument("--report", default="", metavar="RUN", help="print a run's summary (a run name or turns.jsonl) and stop")


COMMAND = Command("llm-agent", "matching", "a local model edits a near-miss function over up to 8 turns (one SEARCH/REPLACE "
                  "edit, rewrite-table transform or STOP per turn); each compiled, scored like try --oracle, best kept, "
                  "judged exact ones written as candidates to llm-data/", run, _configure)
