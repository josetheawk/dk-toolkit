"""Retrieval examples for llm-try's prompt (plan P1.3): the exact functions whose asm looks most
like the target's, as (asm, exact source) pairs within a fixed character budget.

Similarity: the asm normalized like `asm_shape` (mnemonic and operand kinds; registers, immediates
and symbols abstracted), cut into instruction trigrams, compared by Jaccard, with a small bonus
for a similar size. Examples from the target's own compiler version rank first.

Never an evaluation answer: the benchmark's and the held-out set's units and every function with
their asm shape are left out of the index, and the query function itself out of its results. The
index is built at one commit (`rev`): a function is kept only when it is exact now and its
definition there is the one here, so a benchmark run sees the freeze snapshot's sources. It holds
Game-derived code: it lives in llm-data/retrieval/, never in the repo.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from ...core.config import Config
from . import llmtry

BUDGET_CHARS = 6000                 # ~2.7k tokens at llm-try's 2.2 chars per token
MAX_EXAMPLES = 4
_INSN = re.compile(r"^\s*[0-9A-F]+  (\S+)\s*(.*)$")


def normalized(asm_text: str) -> list[str]:
    """One token per instruction: its mnemonic and operand kinds (asm_shape's normalization)."""
    out = []
    for line in llmtry.compact_asm(asm_text).splitlines():
        m = _INSN.match(line)
        if m:
            ops = llmtry._OPERAND.sub(lambda x: "r" if re.fullmatch(r"[rf]\d+|cr\d", x[0])
                                      else ("n" if re.match(r"-?\d|-?0x", x[0]) else "s"), m[2])
            out.append(f"{m[1]} {ops}".strip())
    return out


def grams(tokens: list[str], n: int = 3) -> set[str]:
    if len(tokens) < n:
        return {"|".join(tokens)} if tokens else set()
    return {"|".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def similarity(a: set[str], b: set[str], size_a: int, size_b: int) -> float:
    if not a or not b:
        return 0.0
    jac = len(a & b) / len(a | b)
    return jac + 0.1 * min(size_a, size_b) / max(size_a, size_b, 1)


def compiler_of(cfg: Config, unit: dict) -> str:
    from .reorder import compile_command
    m = re.search(r"compilers/([^/\s]+/[^/\s]+)/mwcc", compile_command(cfg, unit["base_path"], locked=False))
    return m[1] if m else ""


@dataclass
class Example:
    unit: str
    function: str
    size: int
    compiler: str
    asm: str                        # compact asm (llm-try's form)
    source: str                     # the exact definition
    grams: set


def index_path(cfg: Config, rev: str) -> Path:
    return llmtry.data_dir(cfg) / "retrieval" / f"index-{rev[:12]}.json"


def build(cfg: Config, rev: str, current: bool = False, jobs: int = 6, say=print) -> list[dict]:
    """Exact game-side functions at `rev` (exact now, the same definition there; `current`: the
    working tree's sources), minus the benchmark's and held-out units and asm shapes, and vendored
    ports."""
    from .matching import load_objdiff_units
    from .trycmd import definition
    never_units, never_fns, never_shapes = llmtry.never_train(cfg)
    vendored = tuple(llmtry._vendored_paths(cfg))
    entries = llmtry.asm_entries(cfg)
    units = [u for u in load_objdiff_units(cfg) if u.get("base_path") and u["name"] not in never_units
             and not u["metadata"].get("source_path", "").startswith(vendored)
             and u["metadata"].get("source_path", "") not in vendored]

    def one(u: dict) -> list[dict]:
        src_path = u["metadata"].get("source_path", "")
        path = cfg.root / src_path
        if not path.is_file():
            return []
        src = path.read_text()
        then = src if current else llmtry._git(cfg, "show", f"{rev}:{src_path}")
        out = []
        comp = compiler_of(cfg, u)
        for f in llmtry.exact_functions(cfg, u, src):
            if f["function"] in never_fns or f["asm_shape"] in never_shapes:
                continue
            span = definition(src, f["function"])
            if span is None:
                continue
            text = src[span[0]:span[1]]
            if then is not src:
                old = definition(then, f["function"])
                if old is None or then[old[0]:old[1]] != text:
                    continue
            asm = llmtry.compact_asm(entries[f["function"]].text)
            out.append({"unit": u["name"], "function": f["function"], "size": f["size"], "compiler": comp,
                        "asm": asm, "source": text, "tokens": normalized(entries[f["function"]].text)})
        return out

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        recs = [r for rs in pool.map(one, units) for r in rs]
    say(f"retrieval index: {len(recs)} exact functions from {len(units)} units at {rev[:12] or 'HEAD'}")
    return recs


class Index:
    def __init__(self, recs: list[dict]):
        self.items = [Example(r["unit"], r["function"], r["size"], r["compiler"], r["asm"], r["source"],
                              grams(r["tokens"])) for r in recs]

    @classmethod
    def load(cls, cfg: Config, rev: str, current: bool = False, jobs: int = 6, say=print) -> "Index":
        path = index_path(cfg, rev)
        if path.is_file():
            return cls(json.loads(path.read_text())["functions"])
        recs = build(cfg, rev, current, jobs, say)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"rev": rev, "never_commit": True, "functions": recs}))
        return cls(recs)

    def top(self, asm_text: str, size: int, compiler: str, exclude: set[str], k: int = MAX_EXAMPLES,
            budget: int = BUDGET_CHARS) -> list[Example]:
        """The most similar examples (own compiler version first) that fit the budget together."""
        q = grams(normalized(asm_text))
        ranked = sorted((x for x in self.items if x.function not in exclude),
                        key=lambda x: (x.compiler == compiler, similarity(q, x.grams, size, x.size)), reverse=True)
        out, used = [], 0
        for x in ranked[:50]:
            cost = len(x.asm) + len(x.source) + 80
            if cost > budget // 2 or used + cost > budget:
                continue
            out.append(x)
            used += cost
            if len(out) == k:
                break
        return out


def render(examples: list[Example]) -> str:
    if not examples:
        return ""
    parts = ["## Similar functions of this game already matched (their target assembly, then the exact source)"]
    for x in examples:
        parts.append(f"### {x.function}\n```\n{x.asm}\n```\n```cpp\n{x.source.strip()}\n```")
    return "\n\n".join(parts)
