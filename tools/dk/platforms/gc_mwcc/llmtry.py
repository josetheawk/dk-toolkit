"""dk llm-try / llm-bench: a local model (Ollama) proposes definitions of one function, and dk try's
machinery scores them.

  ./dk llm-try <unit> <fn> [--model M] [--n N] [--rounds R] [--think on|off] [--apply]
  ./dk llm-bench freeze [--near 20] [--zero 8]     llm-data/bench.json (never train on these)
  ./dk llm-bench run --model M [--n 8] [--rounds 2] every benchmark function, one log
  ./dk llm-bench report [LOG…]                     one row per run: compile, device, exact rates, speed

The prompt holds the faithfulness rules, the target asm (function-relative offsets), the unit's
declarations (other bodies cut to prototypes, the owner class from the headers), the current
definition or, when the function is unwritten, the m2c draft, and the current diff. Each round asks
for N definitions (one request each, its own seed; the reply is prefilled with the opening fence
and stops at the closing one unless thinking), compiles and scores them in parallel while the
model writes the next one, and feeds the best one and its diff into the next round.

Every attempt is logged to llm-data/<run>.jsonl beside the main checkout (DK_LLM_DATA; outside the
repo, never in a worktree: it holds asm and prompts): model,
options, prompt size, the variant, format, compile result, score, devices, time and tokens/s.
`--apply` writes a variant only when it is exact, hand-linked identical, nothing else in the unit
scores lower, and it adds no device: no volatile, pragma, asm, padding, forbidden quality pattern
or new function (judge.py).

Why: matching near misses is the token-heavy part of the work; this measures whether a local model
can do it under the same gate the closer uses.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from ...core import cbody
from ...core.command import MISMATCH, OK, USAGE, Command, hint
from ...core.config import Config

LLM_DIR = Path("build") / "dk" / "llm"     # scratch compiles (work/) only


def data_dir(cfg: Config) -> Path:
    """Where logs, the benchmark, best variants and datasets live: DK_LLM_DATA, else `llm-data/` next
    to the MAIN checkout (outside the repo: private, game-derived, survives worktree removal and
    cleans; never inside a worktree)."""
    env = os.environ.get("DK_LLM_DATA")
    if env:
        return Path(env).expanduser()
    from ...core.landing import common_dir
    common = common_dir(cfg.root)                    # a worktree's common dir is the main checkout's .git
    main = common.parent if common.name == ".git" else cfg.root
    return main.parent / "llm-data"


DEFAULT_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3:14b"

RULES = """You are matching a decompilation: C/C++ that Metrowerks CodeWarrior for GameCube (mwcc,
optimized) compiles to exactly the target's PowerPC instructions. Rules (a variant that breaks one
is rejected):
- Write the code the original programmers wrote, with the declarations given. Keep the function's
  name, class, parameters and return type.
- Never use: volatile, #pragma, asm, __typeof__, the register keyword, pad/dummy members, dummy
  variables or functions, or field access as pointer plus offset (`*(int*)((char*)p + 0x10)`):
  use the named fields.
- Do not add new functions or globals: rewrite only this one definition.
- The language is C++98 (C for a .c unit) as CodeWarrior accepts it: no nullptr, auto, constexpr,
  range-for, lambdas or other C++11. A null pointer is 0, or NULL only where the declarations use it.
- Use only the types, fields, functions, globals and macros in the declarations, the asm and the
  current definition. Never invent a member (unk4, m24, field_10): if one is not declared, keep the
  expression the current definition or draft uses.
- Match the target instruction for instruction: same control flow, the same order of loads, stores
  and calls; register numbers follow from the order variables are declared and first used."""
PROMPT_VERSION = 6                  # the default; bump when RULES, the message layout or the draft clean-up change (logged)
PROMPT_VERSIONS = (5, 6, 7, 8)      # 6: drafts with declared field names and without a constructor's vtable stores,
                                    # unit notes for recurring compile errors; selectable so baselines stay reproducible
                                    # 7: v6 plus the declarations from other headers it may need (llmdecls)
                                    # 8: v7 with the target asm annotated from evidence (llmannot; round 2, opt-in)
VIEW_OF = {8: 7}                    # a version whose stored view (m2c, decls, notes, context) is another's
DEFAULT_EXAMPLES = 4                # retrieval examples in the prompt. v6 with 4, paired on 47 held-out functions
                                    # (2026-11-02, the LLM plan P1): 11 exact vs v5's 6, best +18.6 points

HINTS = (
    "",
    "Try a different control-flow shape (if/else chain vs switch, early return vs nesting, loop form).",
    "Try declaring and first using the local variables in another order, or adding/removing a local temporary that holds a value the target keeps in a register.",
    "Try different types (signed/unsigned, int/short/char, pointer vs index) where the asm shows the width or the comparison kind.",
    "Try reading the fields through another expression (a member access chain, an index, a pointer kept in a local) or changing the order of the statements where the asm shows a different order.",
    "Try writing the arithmetic and comparisons in a different but equivalent form (operand order, a != 0 vs !a, combined conditions).",
)

FORMAT = ("Reply with exactly one fenced ```cpp code block that holds the complete definition of `{name}`, "
          "and nothing else: only code, no comments, no reasoning.")
PREFILL = "```cpp\n"
# A chat template that adds the empty thinking block only to a new assistant turn (Qwen3's) leaves
# it out of a prefilled one, and the model answers "</think>" first: then the prefill carries it.
NO_THINK = "<think>\n\n</think>\n\n"


# ---- the model ------------------------------------------------------------------------------------

class Ollama:
    def __init__(self, url: str = DEFAULT_URL, timeout: float = 900.0):
        self.url, self.timeout = url.rstrip("/"), timeout

    def _call(self, path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def chat(self, model: str, messages: list[dict], options: dict, think: bool | None = None,
             keep_alive: str = "15m") -> dict:
        body = {"model": model, "messages": messages, "stream": False, "options": options, "keep_alive": keep_alive}
        if think is not None:
            body["think"] = think
        return self._call("/api/chat", body)

    def loaded(self) -> list[dict]:
        try:
            return self._call("/api/ps").get("models", [])
        except (OSError, ValueError):
            return []

    def missing(self, model: str) -> str | None:
        """Why `model` can't answer (server down, not installed), or None. A run against a missing
        model would score every function 0 without saying why."""
        try:
            names = {m.get("name", "") for m in self._call("/api/tags").get("models", [])}
        except (OSError, ValueError) as e:
            return f"no Ollama server at {self.url} ({e})"
        want = model if ":" in model else f"{model}:latest"
        if want not in names and model not in names:
            return f"model {model} is not installed on {self.url} (./np pull it, or pick one of: " \
                   f"{', '.join(sorted(names)[:8]) or 'none'})"
        return None


class OpenAICompat(Ollama):
    """llama.cpp's llama-server (OpenAI-compatible /v1/chat/completions; it continues a trailing
    assistant message), answering in Ollama's reply shape. Model "lora=N" applies only the server's
    LoRA adapter N (`llama-server -m BASE.gguf --lora a.gguf,b.gguf`: one base, several training
    checkpoints), "base" none; any other name leaves the server's own adapter scales."""
    REPEAT_PENALTY = 1.05           # qwen3-coder's Ollama parameter, which llm-try doesn't send

    def chat(self, model: str, messages: list[dict], options: dict, think: bool | None = None,
             keep_alive: str = "15m") -> dict:
        body = {"model": model, "messages": messages, "stream": False, "max_tokens": options.get("num_predict"),
                "repeat_penalty": self.REPEAT_PENALTY, **{k: options[k] for k in
                ("temperature", "top_p", "top_k", "min_p", "seed", "stop") if k in options}}
        m = re.fullmatch(r"lora=(\d+)", model)
        if m or model == "base":
            body["lora"] = [{"id": int(m[1]), "scale": 1.0}] if m else []
        r = self._call("/v1/chat/completions", body)
        choice, t = r["choices"][0], r.get("timings") or {}
        return {"message": {"role": "assistant", "content": choice["message"].get("content") or ""},
                "done_reason": choice.get("finish_reason"), "load_duration": 0,
                "eval_count": r.get("usage", {}).get("completion_tokens", 0),
                "prompt_eval_count": r.get("usage", {}).get("prompt_tokens", 0),
                "eval_duration": int(t.get("predicted_ms", 0) * 1e6),
                "prompt_eval_duration": int(t.get("prompt_ms", 0) * 1e6)}

    def loaded(self) -> list[dict]:
        try:
            return [{"name": m.get("id"), "backend": "openai"} for m in self._call("/v1/models").get("data", [])]
        except (OSError, ValueError, KeyError):
            return []

    def missing(self, model: str) -> str | None:
        """Server down, or None (llama-server serves one base; lora=N/base choose its adapters)."""
        try:
            self._call("/v1/models")
        except (OSError, ValueError) as e:
            return f"no llama-server at {self.url} ({e})"
        return None


def client_for(args: argparse.Namespace) -> Ollama:
    return OpenAICompat(args.url) if getattr(args, "backend", "ollama") == "openai" else Ollama(args.url)


def model_ready(client: Ollama, model: str) -> bool:
    """Print why the model can't run and the fix; False then (callers return USAGE)."""
    check = getattr(client, "missing", None)       # test doubles have none
    why = check(model) if check else None
    if why:
        print(f"error: {why}")
        hint("status")
    return why is None


_THINK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"```[ \t]*([A-Za-z+]*)[ \t]*\n(.*?)```", re.S)


def extract_prefilled(content: str, done_reason: str | None) -> tuple[str | None, str]:
    """A reply that continued PREFILL and stopped at the closing fence: the code itself (Ollama
    may echo the prefill). "truncated" when it hit the token cap instead."""
    text = content[len(NO_THINK):] if content.startswith(NO_THINK) else content
    text = text[len(PREFILL):] if text.startswith(PREFILL) else text
    text = re.sub(r"^```[A-Za-z+]*\n", "", text)
    if done_reason == "length":
        return None, "truncated"
    return (text.strip("\n") + "\n", "prefill") if "{" in text else (None, "none")


DIAG_PREFILL = "DIAG:"               # round 2 (--diag): a repair answer's first line (llmdiag), then the fenced code


def diag_prefill(prefill: str) -> str:
    """The prefill of a DIAG answer: the same thinking block, then "DIAG:" in place of the fence."""
    return prefill[:-len(PREFILL)] + DIAG_PREFILL if prefill.endswith(PREFILL) else DIAG_PREFILL


def extract_diag(content: str, done_reason: str | None) -> tuple[str | None, str, str]:
    """(code, format, the DIAG line) of a reply that continued DIAG_PREFILL: the rest of the line, then a
    fenced block (closed, or cut by the stop right after it); "truncated" at the token cap."""
    text = content[len(NO_THINK):] if content.startswith(NO_THINK) else content
    text = text[len(DIAG_PREFILL):] if text.startswith(DIAG_PREFILL) else text
    line, _, rest = text.partition("\n")
    diag = (DIAG_PREFILL + " " + line.strip()).strip()
    m = re.match(r"\s*```[ \t]*[A-Za-z+]*[ \t]*\n", rest)
    if not m:
        return None, "none", diag
    body = rest[m.end():]
    if "```" in body:
        body = body.split("```", 1)[0]
    elif done_reason == "length":
        return None, "truncated", diag
    if "{" not in body:
        return None, "none", diag
    return body.strip("\n") + "\n", "diag", diag


def extract(reply: str) -> tuple[str | None, str]:
    """The definition in a reply: (code or None, format). Format is "fenced" (a ``` block),
    "several" (more than one block: the last one with a body is taken), "bare" (no fence, but the
    reply is code), "truncated" (a fence never closed) or "none"."""
    text = _THINK.sub("", reply)
    if "<think>" in text:                            # thinking never closed: no answer
        return None, "truncated"
    if text.lstrip().startswith(DIAG_PREFILL):       # round 2: a DIAG line first is not code
        text = text.lstrip().partition("\n")[2]
    blocks = [b for lang, b in _FENCE.findall(text) if "{" in b]
    if blocks:
        return blocks[-1].strip("\n") + "\n", "fenced" if len(blocks) == 1 else "several"
    if "```" in text:
        rest = text.split("```", 1)[1]
        return None, "truncated" if "{" in rest else "none"
    stripped = text.strip()
    if stripped and re.search(r"\)\s*(?:const\s*)?\{", stripped) and stripped.endswith("}"):
        return stripped + "\n", "bare"
    return None, "none"


# ---- what the model is shown ------------------------------------------------------------------------

_ASM_LINE = re.compile(r"^/\*\s*([0-9A-Fa-f]{8})\s+[0-9A-Fa-f]{8}\s+(?:[0-9A-Fa-f]{2}\s){4}\s*\*/\s*(.*)$")


def compact_asm(text: str) -> str:
    """dtk's listing with function-relative offsets in place of the address/offset/bytes comment
    (the offsets the diff rows use); labels and directives kept."""
    out, start = [], None
    for line in text.splitlines():
        m = _ASM_LINE.match(line.strip())
        if m:
            addr = int(m[1], 16)
            start = addr if start is None else start
            out.append(f"{addr - start:5X}  {m[2].strip()}")
        elif line.strip() and not line.startswith("# "):
            out.append(line.rstrip())
    return "\n".join(out)


_OPERAND = re.compile(r"\b(?:[rf]\d+|cr\d)\b|-?0x[0-9a-fA-F]+|-?\b\d+\b|\b[A-Za-z_.$@][\w.$@<>:]*(?:@\w+)?")


def asm_shape(text: str) -> str:
    """A function's instructions with register names, immediates and symbols abstracted (mnemonic and
    operand kinds only): two copies of a function under other names or registers hash the same.
    The leakage key for datasets (never the symbol name: `dk rename` changes names)."""
    out = []
    for line in compact_asm(text).splitlines():
        m = re.match(r"^\s*[0-9A-F]+  (\S+)\s*(.*)$", line)
        if m:
            ops = _OPERAND.sub(lambda x: "r" if x[0][0] in "rfc" and re.fullmatch(r"[rf]\d+|cr\d", x[0])
                               else ("n" if re.match(r"-?\d|-?0x", x[0]) else "s"), m[2])
            out.append(f"{m[1]} {ops}".strip())
    return hashlib.sha1("\n".join(out).encode()).hexdigest()[:16]


_BRANCH = re.compile(r"^(b\w*)\s+(?:(cr\d),\s*)?0x([0-9a-fA-F]+)$")
MARKS = {"DIFF_OP_MISMATCH": "op", "DIFF_ARG_MISMATCH": "arg", "DIFF_REPLACE": "rep", "DIFF_DELETE": "-",
         "DIFF_INSERT": "+"}


def relative_rows(left: dict, right: dict, context: int = 1, limit: int = 60) -> list[str]:
    """Target | ours rows around each difference, offsets (and branch targets) relative to each
    function's start, as the asm listing shows them."""
    li, ri = left.get("instructions", []), right.get("instructions", [])

    def base(xs):                                  # objdiff's JSON leaves out an `address` of 0
        return next((int(x["instruction"].get("address", 0)) for x in xs if "instruction" in x), 0)
    lb, rb = base(li), base(ri)

    def text(x, b):
        t = x.get("instruction", {}).get("formatted", "")
        m = _BRANCH.match(t)
        if m:
            t = f"{m[1]} " + (f"{m[2]}, " if m[2] else "") + f"{int(m[3], 16) - b:X}"
        return t
    n = max(len(li), len(ri))
    diff = [bool((li[i].get("diff_kind") if i < len(li) else None) or (ri[i].get("diff_kind") if i < len(ri) else None))
            for i in range(n)]
    show = set()
    for i, d in enumerate(diff):
        if d:
            show.update(range(max(0, i - context), min(n, i + context + 1)))
    rows, last = [], -2
    for i in sorted(show):
        if i != last + 1:
            rows.append("  ...")
        l = li[i] if i < len(li) else {}
        r = ri[i] if i < len(ri) else {}
        kind = l.get("diff_kind") or r.get("diff_kind")
        off = f"{int(l['instruction'].get('address', 0)) - lb:5X}" if "instruction" in l else "     "
        rows.append(f"{off}  {text(l, lb):<36} {MARKS.get(kind, ''):>3}  {text(r, rb)}")
        last = i
    return rows[:limit] + (["  … (more differences)"] if len(rows) > limit else [])


_ROW = re.compile(r"^(?P<off>.{5})  (?P<l>.*?) *(?: (?P<mark>  -|  \+| op|arg|rep|   )  )(?P<r>.*)$")
_REL_BRANCH = re.compile(r"^(b\w*)\s+(?:(cr\d),\s*)?([0-9A-F]+)$")


def shift_rows(rows: list[str], left: bool = True, right: bool = True, by: int = 4) -> list[str] | None:
    """relative_rows' rows with the offsets (and branch targets) of the target side (`left`) and/or our side
    (`right`) moved `by` forward: how a function at offset 0 of its section is put right when its rows
    were made with the old relative_rows (objdiff leaves out the first instruction's address 0, so the
    second one's address was taken as the base: every offset and target was 4 too small and the first
    offset blank). None when a row does not parse back to itself."""
    def target(t: str) -> str:
        m = _REL_BRANCH.match(t)
        return f"{m[1]} " + (f"{m[2]}, " if m[2] else "") + f"{int(m[3], 16) + by:X}" if m else t
    out = []
    for row in rows:
        if row.startswith(("  ...", "  …")):                      # the gap marker and the "more differences" line
            out.append(row)
            continue
        m = _ROW.match(row)
        if m is None or f"{m['off']}  {m['l']:<36} {m['mark']}  {m['r']}" != row:
            return None
        off, l, mark, r = m["off"], m["l"], m["mark"], m["r"]
        if left:
            off = f"{int(off, 16) + by:5X}" if off.strip() else (f"{0:5X}" if l else off)
            l = target(l)
        if right:
            r = target(r)
        out.append(f"{off}  {l:<36} {mark}  {r}")
    return out


def _top_level(fns: list) -> list:
    out, end = [], -1
    for f in sorted(fns, key=lambda f: f.head_start):
        if f.head_start > end:
            out.append(f)
            end = f.close
    return out


def declarations(src: str, skip: tuple[int, int], wanted: set[str], budget: int = 8000, trim: bool = False) -> str:
    """The unit's source as declarations: the function being matched left out, out-of-class member
    definitions dropped, other free functions cut to prototypes, short inline bodies (in a class)
    kept. With `trim` (or over budget), only the includes and the paragraphs naming a wanted
    identifier (one the function, its draft or its asm uses) stay."""
    fns = cbody.functions(src)
    parts, pos = [], 0
    for f in sorted(fns, key=lambda f: f.head_start):
        if f.head_start < pos:
            continue
        parts.append(src[pos:f.head_start])
        body = src[f.head_start:f.close + 1]
        if skip[0] <= f.head_start < max(skip[1], skip[0] + 1):
            parts.append("/* (the function being matched) */")
        elif len(body) <= 240 and "::" not in f.name:
            parts.append(body)
        elif "::" not in f.name:
            parts.append(src[f.head_start:f.open].rstrip() + ";")
        pos = f.close + 1
    parts.append(src[pos:])
    text = re.sub(r"\n\s*\n(\s*\n)+", "\n\n", "".join(parts)).strip() + "\n"
    if len(text) <= budget and not trim:
        return text
    keep = [c for c in _chunks(text) if c.lstrip().startswith(("#include", "#if", "#else", "#endif")) or
            identifiers(c) & wanted]
    trimmed = re.sub(r"\n\s*\n(\s*\n)+", "\n\n", "\n".join(keep)).strip() + "\n"
    if trimmed.strip() and len(trimmed) <= len(text):
        text = trimmed
    return text if len(text) <= budget else text[:budget] + "\n/* … (declarations cut at the budget) */\n"


def _chunks(text: str) -> list[str]:
    """Top-level pieces of a declarations text: a line, or from a line that opens a brace to the
    line that closes it (a whole class, enum or inline body), comments attached to what follows."""
    out, cur, depth = [], [], 0
    for line in text.splitlines():
        cur.append(line)
        depth += line.count("{") - line.count("}")
        stripped = line.strip()
        if depth <= 0 and not (stripped.startswith(("//", "/*")) and not stripped.endswith(";")) or not stripped and depth <= 0:
            depth = 0
            out.append("\n".join(cur))
            cur = []
    if cur:
        out.append("\n".join(cur))
    return out


def header_classes(cfg: Config, src: str, names: set[str], budget: int = 6000) -> str:
    """Definitions of the named classes/structs found in the source's quoted includes (3 deep)."""
    found: dict[str, str] = {}
    seen: set[Path] = set()

    def walk(text: str, depth: int) -> None:
        for inc in re.findall(r'^\s*#\s*include\s+"([^"]+)"', text, re.M):
            for base in ("include", "src"):
                p = cfg.root / base / inc
                if p.is_file():
                    if p not in seen:
                        seen.add(p)
                        body = p.read_text(errors="replace")
                        for n in names - set(found):
                            m = re.search(rf"^\s*(?:class|struct|union)\s+{re.escape(n)}\b[^;{{]*\{{", body, re.M)
                            if m:
                                close = cbody.matching_brace(body, m.end() - 1)
                                if close is not None:
                                    found[n] = f"// {cfg.rel(p)}\n" + body[m.start():close + 1].strip() + ";"
                        if depth < 3:
                            walk(body, depth + 1)
                    break
    walk(src, 0)
    out, size = [], 0
    for n in sorted(found, key=lambda n: len(found[n])):
        if size + len(found[n]) > budget:
            continue
        out.append(found[n])
        size += len(found[n])
    return "\n\n".join(out)


KEYWORDS = frozenset("""auto break case char const continue default double else enum extern float for goto
if inline int long register return short signed sizeof static struct switch typedef union unsigned void
volatile while bool true false class public private protected virtual this new delete operator template
typename namespace using friend static_cast reinterpret_cast const_cast dynamic_cast explicit mutable
NULL""".split())


def identifiers(text: str) -> set[str]:
    """The names a text uses (3+ characters, C/C++ keywords left out)."""
    return set(re.findall(r"\b[A-Za-z_]\w{2,}\b", text)) - KEYWORDS


# m2c's fixed-width types -> plain C types, for a unit that cannot see dolphin/types.h
M2C_TYPES = {"s8": "signed char", "u8": "unsigned char", "s16": "short", "u16": "unsigned short",
             "s32": "int", "u32": "unsigned int", "s64": "long long", "u64": "unsigned long long",
             "f32": "float", "f64": "double"}


def visible_files(cfg: Config, src: str, source: Path, depth: int = 4) -> list[tuple[Path, str]]:
    """Every header the unit's source includes (quoted or angled, a few levels deep, looked up beside
    it, in include/, src/ and the vendored SDK roots), in include order, as (path, text)."""
    roots = [source.parent, cfg.root / "include", cfg.root / "src", *sorted((cfg.root / "include" / "sdk").glob("*"))]
    seen: set[Path] = set()
    out: list[tuple[Path, str]] = []

    def walk(text: str, here: Path, d: int) -> None:
        for inc in re.findall(r'^\s*#\s*include\s+["<]([^">]+)[">]', text, re.M):
            for base in (here, *roots):
                p = base / inc
                if p.is_file():
                    if p not in seen:
                        seen.add(p)
                        body = p.read_text(errors="replace")
                        out.append((p, body))
                        if d < depth:
                            walk(body, p.parent, d + 1)
                    break
    walk(src, source.parent, 0)
    return out


def visible_text(cfg: Config, src: str, source: Path, depth: int = 4) -> str:
    """The unit's source plus every header it includes (visible_files): to tell which names it can see."""
    return "\n".join([src] + [t for _p, t in visible_files(cfg, src, source, depth)])


def qualified(symbol: str) -> str | None:
    """A mangled member's C++ spelling (`Class::name`, `Class::Class`, `Class::~Class`), else None."""
    from .demangle import demangle
    try:
        d = demangle(symbol)
    except Exception:
        return None
    if d is None:
        return None
    last = d.quals[-1] if d.quals else ""
    name = {"ctor": last, "dtor": "~" + last, "__nw": "operator new", "__dl": "operator delete",
            "__nwa": "operator new[]", "__dla": "operator delete[]"}.get(d.name, d.name)
    if name.startswith("operator "):
        return name
    if not isinstance(name, str) or not re.fullmatch(r"~?[A-Za-z_]\w*", name):
        return None
    return "::".join([*d.quals, name])


def draft_function(draft: str, symbol: str) -> str:
    """Only the target's definition from an m2c draft (its guessed structs and extern prototypes
    left out: they invent members the declarations already name), with mangled names spelled
    as C++ (`Class::name`) and m2c's explicit `this` parameter dropped."""
    lines = draft.splitlines()
    at = next((i for i, l in enumerate(lines) if re.search(rf"\b{re.escape(symbol)}\s*\(", l)
               and re.sub(r"/\*.*?\*/|//.*", "", l).rstrip().endswith("{")), None)
    if at is not None:
        draft = "\n".join(lines[at:])
    else:
        return draft
    own = qualified(symbol)
    if own:
        cls = re.escape(own.rsplit("::", 1)[0].split("::")[-1])
        draft = re.sub(rf"\b{re.escape(symbol)}\s*\(\s*(?:struct\s+)?{cls}\s*\*\s*this\s*(?:,\s*|(?=\)))",
                       own + "(", draft, count=1)
        last = own.rsplit("::", 1)[-1]
        if last.lstrip("~") == own.rsplit("::", 1)[0].split("::")[-1]:   # a constructor or destructor: no return type
            head, _, rest = draft.partition("\n")
            head = re.sub(rf"^.*?(?={re.escape(own)}\()", "", head)
            if last.startswith("~"):
                head = re.sub(r"\(\s*short\s+\w+\s*\)", "()", head)
            draft = head + "\n" + rest

    def name(m):
        q = qualified(m[0])
        return q if q else m[0]
    draft = re.sub(r"\b[A-Za-z_]\w*__\w+\b", name, draft)
    # m2c calls a member as Class::method(obj, …): C++ writes obj->method(…)
    return re.sub(r"\b(?:[A-Za-z_]\w*::)+([A-Za-z_~]\w*)\((this|[A-Za-z_]\w*)(?:,\s*|(?=\)))",
                  lambda m: f"{m[2]}->{m[1]}(", draft.replace(own + "(", "\0OWN\0") if own else draft).replace("\0OWN\0", (own or "") + "(")


def signature(symbol: str) -> str:
    """`Class::name(int, Wnd*)` for a mangled member (return type unknown), else ""."""
    from .demangle import demangle
    q = qualified(symbol)
    if not q or "::" not in q:
        return ""
    try:
        d = demangle(symbol)
    except Exception:
        return ""
    return f"{q}({', '.join(p.text for p in d.params)})" + (" const" if d.const else "")


def clean_draft(draft: str, visible: str, symbol: str = "") -> str:
    """Only the function (draft_function), m2c's types the unit cannot see as plain C types, NULL as
    0 where it is not defined, and m2c's unknown type `?` as int."""
    if symbol:
        draft = draft_function(draft, symbol)
    for t, plain in M2C_TYPES.items():
        if not re.search(rf"\btypedef\b[^;]*\b{t}\s*;", visible):
            draft = re.sub(rf"\b{t}\b", plain, draft)
    if not re.search(r"^\s*#\s*define\s+NULL\b", visible, re.M):
        draft = re.sub(r"\bNULL\b", "0", draft)
    return re.sub(r"(?<![\w?])\?(?=\s*[*\w(])", "int", draft)


@dataclass
class Ingredients:
    asm: bool = True
    decls: bool = True
    diff: bool = True
    m2c: str = "auto"               # auto: only when the function is unwritten; always; never

    def tag(self) -> str:
        off = [n for n in ("asm", "decls", "diff") if not getattr(self, n)]
        return "+".join([f"m2c={self.m2c}"] + [f"no-{n}" for n in off])


@dataclass
class Context:
    name: str                       # the symbol
    asm: str = ""
    decls: str = ""
    m2c: str = ""
    current: str = ""               # the current definition ("" when unwritten)
    signature: str = ""             # a member's C++ name and parameters (from the symbol)
    notes: str = ""                 # v6: what this function's asm and draft need (operator new, vtables, arrays)
    examples: str = ""              # retrieval: similar exact functions (asm -> source)
    context: str = ""               # v7: declarations from other headers (llmdecls.context), merged before decls


def score_text(pct: float) -> str:
    """A score as the prompt shows it: rounded down, so a source that still differs never reads "100.00%"."""
    import math
    return f"{math.floor(pct * 100 + 1e-6) / 100:.2f}%" if pct < 100 else "100.00%"


def build_messages(ctx: Context, best_code: str, best_pct: float, best_rows: list[str],
                   ing: Ingredients, first_round: bool, hint_text: str = "",
                   failed: tuple[str, str] | None = None) -> list[dict]:
    """System: the rules. User: asm, declarations, draft, current/best definition and its diff."""
    parts = [ctx.examples] if ctx.examples else []
    if ing.asm and ctx.asm:
        parts.append(f"## Target assembly of {ctx.name}\n```\n{ctx.asm}\n```")
    decls = ctx.decls
    if ctx.context:
        from .llmdecls import merge
        decls = merge(ctx.context, decls)
    if ing.decls and decls:
        parts.append(f"## Declarations in scope (the unit's source, bodies cut)\n```cpp\n{decls}```")
    use_m2c = ctx.m2c and (ing.m2c == "always" or (ing.m2c == "auto" and not ctx.current))
    if use_m2c:
        parts.append(f"## Machine draft (m2c: types and names are guesses)\n```c\n{ctx.m2c.strip()}\n```")
    if best_code:
        what = "Current definition" if first_round else "Best attempt so far"
        parts.append(f"## {what} (scores {score_text(best_pct)})\n```cpp\n{best_code.strip()}\n```")
        if ing.diff and best_rows:
            parts.append("## Where it differs (offset, target, mark, ours; op = other instruction, arg = other "
                         "operands, - = only the target has it, + = only ours)\n```\n"
                         + "\n".join(best_rows) + "\n```")
    elif not use_m2c:
        parts.append(f"## {ctx.name} is not written yet.")
    if not best_code and ctx.signature:
        parts.append(f"## Write the member function `{ctx.signature}` declared in its class, in C++98: never a "
                     "mangled name, never an explicit `this` parameter; call members as `obj->method(...)`.")
    if failed and not best_code:
        parts.append(f"## A previous attempt did not compile\n```cpp\n{failed[0].strip()}\n```\nCompiler: {failed[1]}")
    if ctx.notes:
        parts.append(ctx.notes)
    if hint_text:
        parts.append(hint_text)
    parts.append(("Rewrite the definition so it compiles to the target exactly. " if best_code else
                  "Write the definition so it compiles to the target exactly. ") + FORMAT.format(name=ctx.name))
    return [{"role": "system", "content": RULES}, {"role": "user", "content": "\n\n".join(parts)}]


def compiler_errors(out: str, limit: int = 3) -> str:
    """mwcc's first errors as `line N: <source> | <message>` (its listing is several lines each)."""
    lines = out.splitlines()
    found = []
    for i, line in enumerate(lines):
        if line.lstrip("#").strip().startswith("Error:") and i + 1 < len(lines):
            where = lines[i - 1].lstrip("#").strip() if i > 0 else ""
            found.append(f"line {where} | " + lines[i + 1].lstrip("#").strip())
            if len(found) == limit:
                break
    return "; ".join(found) or " / ".join(x.strip() for x in lines[-3:])


# ---- the gate ---------------------------------------------------------------------------------------

def new_functions(src: str, code: str, symbol: str) -> list[str]:
    """Functions the variant defines that the source does not (dummy or invented helpers)."""
    from .closecmd import plain_name
    plain = plain_name(symbol)
    have = {f.name for f in cbody.functions(src)}
    try:
        mine = cbody.functions(code)
    except Exception:                               # unparsable: the compiler will say so
        return []
    out = []
    for f in mine:
        last = f.name.split("::")[-1]
        if last == plain or f.name == plain or f.name in have:
            continue
        # the symbol's mangled name written as the C++ name: a wrong name, not an extra function
        out.append(f"writes the mangled name {last}" if last == symbol else f"adds function {f.name}")
    return out


def devices(src: str, text: str, code: str, symbol: str, forbidden) -> list[str]:
    from .judge import source_reasons
    return source_reasons(src, text, forbidden) + new_functions(src, code, symbol)


# ---- one search -------------------------------------------------------------------------------------

@dataclass
class Settings:
    model: str = DEFAULT_MODEL
    n: int = 8
    rounds: int = 2
    think: bool | None = False
    num_ctx: int = 16384
    num_predict: int = 0            # 0: 2048 without thinking, 8192 with
    temperature: float = 1.0
    top_p: float = 1.0              # 0.95 gave byte-identical copies on a rewrite; min_p cuts the tail instead
    top_k: int = 20
    seed: int = 1
    ing: Ingredients = field(default_factory=Ingredients)
    hints: bool = True              # variant k > 0 of a rewrite gets one of HINTS (spread)
    prefill: bool = True            # open the reply with the fence and stop at the closing one (not when thinking)
    min_p: float = 0.05
    prompt: int = PROMPT_VERSION    # the prompt template version (PROMPT_VERSIONS)
    examples: int = DEFAULT_EXAMPLES    # retrieval: similar exact functions in the prompt (0: off)
    search: str = "greedy"          # llmsearch.py (opt-in): "tree" branches across the top `beam` variants
    beam: int = 3
    adaptive: bool = False          # stop on no progress, extra rounds for improving functions (per-run bank)
    budget: object = None
    diag: bool = False              # round 2: repair rounds prefill "DIAG:" (a one-line diagnosis, then the fence)
    repair: bool = False            # llmrepair.py: declaration-visibility repair, invented-name and prototype feedback
    tool_block: bool = False        # lorafix.tool_block: a structured TOOLS line (score, diff rows, hints) in every prompt
    tool_hints: tuple = ()          # validated rule:/xform:/region: tokens (lorafix.HINT_RE)
    rank: str = "fuzzy"             # which attempt feeds the next round: "shape" (shape-equal first) or "fuzzy" (score)

    def prefilled(self) -> bool:
        return self.prefill and self.think is False

    def options(self, seed: int, expect_chars: int = 0, diag: bool = False) -> dict:
        """`expect_chars`: the size of the definition being rewritten (or the draft); without an
        explicit --num-predict the reply is capped at twice its tokens (a looping reply stops early)."""
        predict = self.num_predict or (8192 if self.think else 2048)
        if not self.num_predict and expect_chars and not self.think:
            predict = min(predict, max(512, int(2 * expect_chars / 2.2)))
        opts = {"num_ctx": self.num_ctx, "num_predict": predict, "temperature": self.temperature,
                "top_p": self.top_p, "top_k": self.top_k, "min_p": self.min_p, "seed": seed}
        if diag:                                   # DIAG line first: the opening fence must not stop it
            return {**opts, "num_predict": predict + 96, "stop": ["\n```\n"]}
        return {**opts, "stop": ["```"]} if self.prefilled() else opts


@dataclass
class Outcome:
    start: float
    best: float
    best_code: str = ""
    exact_code: str = ""            # judged exact and faithful
    exact_text: str = ""            # the whole source with it
    attempts: int = 0


def search(ctx: Context, start_code: str, start_pct: float, start_rows: list[str], evaluate, client: Ollama,
           st: Settings, log, record: dict | None = None, jobs: int = 4, say=print, start_shape: bool = False) -> Outcome:
    """R rounds of N requests. `evaluate(code, slot)` -> dict with compiled, percent, rows, devices,
    exact (judged), error, text. `log(dict)` gets every attempt. The default (greedy) builds every
    request of a round from the best variant so far; `st.search == "tree"` and `st.adaptive` are the
    opt-in policies of llmsearch.py (parents per request, early stop and a per-run bank)."""
    from . import llmsearch
    tree, adaptive = getattr(st, "search", "greedy") == "tree", bool(getattr(st, "adaptive", False))
    extra = tree or adaptive                         # the non-default modes log parents (the default's rows stay as they were)
    budget = getattr(st, "budget", None) if adaptive else None
    out = Outcome(start_pct, start_pct, start_code)
    from .llmrepair import feedback, rank_key
    best_code, best_pct, best_rows = start_code, start_pct, start_rows
    best_id = "start"
    rank = getattr(st, "rank", "fuzzy")
    best_key = rank_key({"compiled": True, "percent": start_pct, "shape_equal": start_shape}, rank)
    beam = [llmsearch.Node("start", start_code, start_pct, list(start_rows or []), llmsearch.node_key(
        {"percent": start_pct, "rows": start_rows or []}) if start_code else "start", -1)]
    failed = None
    prefill = PREFILL
    planned = st.rounds
    limit = 2 * st.rounds if adaptive else st.rounds
    r = 0
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        while r < planned:
            if tree:
                parents = llmsearch.allocate(beam, st.n)
            else:
                parents = [llmsearch.Node(best_id, best_code, best_pct, best_rows)] * st.n
            base: dict[str, tuple[list[dict], int]] = {}
            for node in parents:                     # one prompt per distinct parent (greedy: one per round)
                if node.id in base:
                    continue
                msgs = build_messages(ctx, node.code, node.pct, node.rows, st.ing, r == 0, "", failed)
                base[node.id] = (msgs, sum(len(m["content"]) for m in msgs))
                log({"kind": "prompt", **(record or {}), "function": ctx.name, "round": r, "messages": msgs,
                     **({"parent": node.id, "search": st.search} if extra else {})})
            pending, seen = [], {}
            for k in range(st.n):
                node = parents[k]
                base_msgs, prompt_chars = base[node.id]
                hint_text = HINTS[k % len(HINTS)] if st.hints and node.code else ""
                if st.tool_block and node.code:
                    from .lorafix import tool_block
                    hint_text = (hint_text + "\n" if hint_text else "") + tool_block(node.pct, node.rows, st.tool_hints)
                messages = build_messages(ctx, node.code, node.pct, node.rows, st.ing, r == 0, hint_text) \
                    if hint_text else base_msgs
                seed = st.seed + 1000 * r + k
                t0 = time.time()
                row = {"kind": "attempt", **(record or {}), "function": ctx.name, "round": r, "k": k, "seed": seed,
                       "model": st.model, "think": st.think, "ingredients": st.ing.tag(), "prompt_chars": prompt_chars,
                       "hint": k % len(HINTS) if hint_text else None,
                       **({"parent": node.id, "search": st.search} if extra else {})}
                diag = st.diag and bool(node.code) and st.prefilled()     # round 2: a repair answer opens with DIAG
                try:
                    pre = diag_prefill(prefill) if diag else prefill
                    sent = messages + [{"role": "assistant", "content": pre}] if st.prefilled() else messages
                    reply = client.chat(st.model, sent, st.options(seed, len(node.code or ctx.m2c), diag), st.think)
                    if st.prefilled() and prefill == PREFILL and \
                            reply.get("message", {}).get("content", "").lstrip().startswith("</think>"):
                        prefill = NO_THINK + PREFILL         # this template: the empty think block goes in the prefill
                        row["prefill_think_block"] = True
                        pre = diag_prefill(prefill) if diag else prefill
                        sent = messages + [{"role": "assistant", "content": pre}]
                        reply = client.chat(st.model, sent, st.options(seed, len(node.code or ctx.m2c), diag), st.think)
                except (OSError, ValueError, urllib.error.URLError) as e:
                    row.update(error=f"ollama: {e}", secs=round(time.time() - t0, 2), format="error")
                    log(row)
                    say(f"  r{r} v{k}: ollama error: {e}")
                    continue
                msg = reply.get("message", {})
                if diag:
                    code, fmt, row["diag"] = extract_diag(msg.get("content", ""), reply.get("done_reason"))
                else:
                    code, fmt = extract_prefilled(msg.get("content", ""), reply.get("done_reason")) if st.prefilled() \
                        else extract(msg.get("content", ""))
                ev, gen_ns = reply.get("eval_count", 0), reply.get("eval_duration", 0) or 1
                pv, prompt_ns = reply.get("prompt_eval_count", 0), reply.get("prompt_eval_duration", 0) or 1
                row.update(secs=round(time.time() - t0, 2), prompt_tokens=pv, gen_tokens=ev,
                           gen_tps=round(ev / gen_ns * 1e9, 1), prompt_tps=round(pv / prompt_ns * 1e9, 1),
                           load_s=round(reply.get("load_duration", 0) / 1e9, 2), done_reason=reply.get("done_reason"),
                           thinking_chars=len(msg.get("thinking") or ""), format=fmt, variant=code,
                           reply=None if code else msg.get("content", "")[:2000])
                key = " ".join((code or "").split())
                if code and key in seen:                 # the same definition again: its score, no compile
                    row["duplicate_of"] = seen[key]
                    pending.append((row, None))
                    continue
                if code:
                    seen[key] = k
                pending.append((row, pool.submit(evaluate, code, f"r{r}v{k}") if code else None))
            done: dict[int, dict] = {}
            round_best = best_pct
            for row, fut in pending:
                if "duplicate_of" in row:
                    res = dict(done.get(row["duplicate_of"], {"compiled": False, "error": "duplicate"}))
                else:
                    res = fut.result() if fut is not None else {"compiled": False, "error": "no code in the reply"}
                    done[row["k"]] = res
                row.update({k: v for k, v in res.items() if k != "text"})
                row["delta"] = round(res.get("percent", 0.0) - out.start, 3) if res.get("compiled") else None
                log(row)
                out.attempts += 1
                pct = res.get("percent", 0.0)
                mark = "EXACT" if res.get("exact") else (f"{pct:6.2f}%" if res.get("compiled") else "no compile")
                say(f"  r{row['round']} v{row['k']}: {mark:>10}  {row['format']:<9} {row.get('secs', 0):6.1f} s "
                    f"{row.get('gen_tps', 0):5.1f} tok/s" + (f"  devices: {', '.join(res['devices'])}" if res.get("devices") else "")
                    + (f"  ({res['error'][:90]})" if not res.get("compiled") and res.get("error") else "")
                    + ("  shape-equal" if res.get("shape_equal") and not res.get("exact") else "")
                    + (f"  reach: {'yes' if res['reach'].get('reachable') else 'no'}, forced "
                       f"{'exact' if res['reach'].get('force_exact') else 'not exact'}" if isinstance(res.get("reach"), dict)
                       and "error" not in res["reach"] else "")
                    + (f"  repaired: {'; '.join(res['repairs'])}" if res.get("repairs") else ""))
                if res.get("exact") and not out.exact_code:
                    out.exact_code, out.exact_text = row["variant"], res.get("text", "")
                if not res.get("compiled") and row.get("variant") and failed is None:
                    note = feedback(res.get("error", ""), "\n".join((ctx.decls, ctx.context, ctx.current, ctx.m2c, ctx.asm))) \
                        if getattr(st, "repair", False) else ""
                    failed = (row["variant"], res.get("error", "") + (f"\nNote: {note}" if note else ""))
                if res.get("compiled") and not res.get("devices"):
                    if tree and "duplicate_of" not in row:
                        beam.append(llmsearch.Node(f"r{r}v{row['k']}", row["variant"], pct, res.get("rows", []),
                                                   llmsearch.node_key(res), 1000 * r + row["k"]))
                    if rank_key(res, rank) > best_key:
                        best_code, best_pct, best_rows, best_id = row["variant"], pct, res.get("rows", []), f"r{r}v{row['k']}"
                        best_key = rank_key(res, rank)
            out.best, out.best_code = best_pct, best_code
            if tree:
                beam = llmsearch.select(beam, st.beam)
            if out.exact_code:
                break
            r += 1
            if adaptive:
                if best_pct <= round_best:               # no progress this round: stop and bank the rest
                    if budget is not None:
                        budget.deposit((planned - r) * st.n)
                    break
                if r == planned and planned < limit and budget is not None and budget.draw(st.n):
                    planned += 1                         # still improving: one more round from the bank
    if extra:
        log({"kind": "search", **(record or {}), "function": ctx.name, "search": st.search,
             "beam": st.beam if tree else 1, "adaptive": adaptive, "rounds_run": min(r + 1, planned) if out.exact_code
             else r, "planned": planned, "attempts": out.attempts, "best": out.best, "exact": bool(out.exact_code),
             "beam_ids": [n.id for n in beam] if tree else [best_id],
             **({"bank": budget.bank, "saved": budget.saved, "spent": budget.spent} if budget is not None else {})})
    return out

# ---- the real scorer ---------------------------------------------------------------------------------

class Scorer:
    """Compiles a variant privately in the unit (dk try's splice), scores it, and judges an exact one."""

    def __init__(self, cfg: Config, unit: dict, symbol: str, src: str, span, locate):
        from .judge import forbidden_rules
        self.cfg, self.unit, self.symbol, self.src, self.span, self.locate = cfg, unit, symbol, src, span, locate
        self.forbidden = forbidden_rules(cfg)
        tag = hashlib.sha1(f"{unit['name']}:{symbol}".encode()).hexdigest()[:8]
        self.work = cfg.root / LLM_DIR / "work" / f"{os.getpid()}_{tag}"
        self.base = None
        self.base_shape = False
        self.repair = False         # llmrepair: make declarations the unit makes later visible, then compile again
        self.reach = None           # text -> regs.reach_report result, for shape-equal attempts (oracle contract v2)
        self.reached = 0

    def compile_score(self, text: str, slot: str, info: dict | None = None):
        from .linkproof import function_proof
        from .matching import MatchError, score
        from .reorder import compile_variant
        from .regs import _diff
        work = self.work / slot
        try:
            obj, err = compile_variant(self.cfg, self.unit, text, work)
            if obj is None:
                return None, [compiler_errors(err) or "compile error"], [], None
            try:
                s = score(self.cfg, self.unit["name"], self.cfg.root / self.unit["target_path"], obj)
            except MatchError as e:
                return None, [str(e)], [], None
            pair = _diff(self.cfg, self.unit, obj, self.symbol)
            rows = relative_rows(pair[0], pair[1]) if pair else []
            if pair and info is not None:
                from .llmrepair import shape_rows
                info["shape_rows"] = shape_rows(pair[0]["instructions"], pair[1]["instructions"])
            got = {f.name: f.percent for f in s.functions}
            proof = function_proof(self.cfg, self.unit, obj, self.symbol) if got.get(self.symbol) == 100.0 else None
            return s, [], rows, proof
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def baseline(self) -> tuple[float, list[str]]:
        info: dict = {}
        s, err, rows, _ = self.compile_score(self.src, "base", info)
        if s is None:
            raise RuntimeError("the unit's current source does not compile: " + " / ".join(err))
        self.base = s
        self.base_shape = info.get("shape_rows") == 0
        return {f.name: f.percent for f in s.functions}.get(self.symbol, 0.0), rows

    def evaluate(self, code: str, slot: str) -> dict:
        from .judge import judge
        from .trycmd import variant_text
        text = variant_text(self.src, self.symbol, code, self.span, self.locate)
        info: dict = {}
        s, err, rows, proof = self.compile_score(text, slot, info)
        repairs: list[str] = []
        while s is None and self.repair and len(repairs) < 4:
            fixed = self.visibility(text, " ".join(err))
            if fixed is None:
                break
            text, label = fixed
            repairs.append(label)
            s, err, rows, proof = self.compile_score(text, slot, info)
        found = devices(self.src, text, code, self.symbol, self.forbidden)
        if s is None:
            return {"compiled": False, "error": " / ".join(x.strip() for x in err)[:400], "devices": found,
                    **({"repairs": repairs} if repairs else {})}
        got = {f.name: f.percent for f in s.functions}
        pct = got.get(self.symbol, 0.0)
        before = {f.name: f.percent for f in self.base.functions} if self.base else {}
        worse = [n for n, p in before.items() if n != self.symbol and got.get(n, -1.0) < p]
        res = {"compiled": True, "percent": pct, "worse": worse[:8], "devices": found, "rows": rows, "rows_fixed": 1,
               "exact": False, "text": text, "shape_equal": info.get("shape_rows") == 0, "shape_rows": info.get("shape_rows")}
        if repairs:
            res["repairs"] = repairs
        if pct == 100.0 and self.base is not None:
            from .judge import aliasing_locals, bodies, parameters, unit_probe
            probe = unit_probe(self.cfg, self.unit, self.symbol, self.work / f"{slot}_casts")
            v = judge(self.src, text, None, None, self.base, s, self.symbol, self.forbidden, proof, probe)
            res["judge"] = v.reasons[:6] + [d for d in found if d not in v.reasons]
            if not res["judge"]:                    # the exact flag consults review() too (RL1: an UnkLoop helper)
                from .judge import exact_review
                blocking, category, flags = exact_review(self.src, text, self.symbol, self.forbidden, probe)
                res["judge"] = blocking[:6]
                res["review"] = [category, flags]
            res["exact"] = not res["judge"]
            b_body, a_body, a_span = bodies(self.src, text, self.symbol)
            if a_body is not None:                  # needs judgment: logged, still exact
                res["judgment"] = aliasing_locals(b_body, text, a_body, parameters(text, a_span))
        if res["shape_equal"] and not res["exact"] and pct < 100.0 and self.reach is not None and self.reached < 8:
            self.reached += 1                       # a diagnosis: reachable by order? exact when forced?
            try:
                code_, out = self.reach(text)
                res["reach"] = {"reachable": (out.get("analysis") or {}).get("reachable"),
                                "force_exact": out.get("force_exact"),
                                "decl_order": [[x[0] for x in (r.get("decl_order") or {}).get("declare", [])]
                                               for r in (out.get("analysis") or {}).get("classes", {}).values()
                                               if isinstance(r.get("decl_order"), dict) and r["decl_order"].get("changed")]}
            except Exception as e:                  # noqa: BLE001 (a diagnosis never stops the search)
                res["reach"] = {"error": f"{type(e).__name__}: {e}"[:200]}
        return res

    def visibility(self, text: str, error: str):
        """(text, label) with the first undefined identifier the unit declares later made visible, or None."""
        from .llmrepair import undefined_names, visibility_fix
        from .trycmd import definition
        span = definition(text, self.symbol)
        if span is None:
            return None
        for name in undefined_names(error):
            fixed = visibility_fix(text, span[0], name)
            if fixed is not None:
                return fixed
        return None


def m2c_draft(cfg: Config, symbol: str, asm_file: Path, c: bool) -> str:
    m2c = cfg.root / cfg.platform_options.get("m2c", ".venv/bin/m2c")
    if not m2c.is_file():
        return ""
    cmd = [str(m2c), "--target", f"ppc-mwcc-{'c' if c else 'c++'}", "-f", symbol, str(asm_file)]
    try:
        proc = subprocess.run(cmd, cwd=cfg.root, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return ""
    return proc.stdout.strip()


_ASM_CACHE: dict = {}


def asm_entries(cfg: Config) -> dict:
    from . import asmindex
    if "x" not in _ASM_CACHE:
        _ASM_CACHE["x"] = asmindex.index(asmindex.current_asm_files(cfg))
    return _ASM_CACHE["x"]


def prepare(cfg: Config, unit_query: str, symbol: str, src_text: str | None = None, decl_budget: int = 8000,
            m2c: bool = True, trim: bool = True, clean: bool = True, blank: bool = False, version: int = PROMPT_VERSION,
            retrieval=None, examples: int = 0):
    """(unit, source path, src, span, locate, Context) for one function. `src_text` replaces the
    unit's source (a benchmark's frozen copy). `trim`: declarations cut to what the function uses;
    `clean`: the m2c draft's types mapped to ones the unit can see. `blank`: the definition is the
    answer (a held-out exact function): the prompt shows it as unwritten and nothing is taken from it.
    `version` 6: field names in the draft and notes (v6_draft, unit_notes); 7: also llmdecls' context. `retrieval` (a
    retrieval.Index) with `examples` > 0: that many similar exact functions first."""
    from .demangle import demangle
    from .matching import resolve_unit
    from .trycmd import locate_target
    unit = resolve_unit(cfg, unit_query)
    source = cfg.root / unit["metadata"]["source_path"]
    src = src_text if src_text is not None else source.read_text()
    span, locate, _note = locate_target(cfg, src, source, symbol)
    entry = asm_entries(cfg).get(symbol)
    if entry is None:
        raise RuntimeError(f"{symbol}: no asm (run ./dk build)")
    current = src[span[0]:span[1]] if span[0] != span[1] and not blank else ""
    draft = m2c_draft(cfg, symbol, entry.file, source.suffix == ".c") if m2c else ""
    visible = visible_text(cfg, src, source) if (draft and clean) or version >= 6 else ""
    if draft and clean:
        draft = clean_draft(draft, visible, symbol)
    layouts: list = []
    if draft and clean and version >= 6:
        draft = v6_draft(cfg, unit, src, symbol, draft, visible, layouts)
    wanted = identifiers(current or draft) | set(entry.calls) | set(entry.refs)
    try:
        dm = demangle(symbol)
    except Exception:
        dm = None
    classes = set(dm.quals) if dm is not None and dm.quals else set()
    classes |= {w for w in identifiers(current or draft) if w[:1].isupper()}
    decls = declarations(src, span, wanted, decl_budget, trim)
    context = ""
    if version >= 7:                                 # v7: llmdecls' blocks instead of header_classes' whole classes,
        from . import llmdecls                       # from the draft when there is one (as the base records' views)
        code = draft or current
        index = llmdecls.unit_index(cfg, src, source, span)
        context = llmdecls.context(index, symbol, code, list(entry.calls), list(entry.refs),
                                   identifiers(code) | set(entry.calls) | set(entry.refs),
                                   extra_types=llmdecls.draft_types(code, layouts[0] if layouts else None))
        extra = ""
    else:
        extra = header_classes(cfg, src, {c for c in classes if not re.search(rf"(?:class|struct)\s+{re.escape(c)}\b", src)})
    if extra:
        decls = extra + "\n\n" + decls
    ctx = Context(symbol, compact_asm(entry.text), decls, draft, current, signature(symbol))
    ctx.context = context
    if version >= 8:                                 # v8: loads, stores and calls commented from evidence
        ctx.asm = annotated_asm(cfg, unit, src, symbol, ctx.asm, index, layouts[0] if layouts else None)
    if version >= 6:
        ctx.notes = unit_notes(entry, draft, visible)
    if retrieval is not None and examples > 0:
        from .retrieval import compiler_of, render
        size = sum(1 for line in ctx.asm.splitlines() if re.match(r"^\s*[0-9A-F]+  \S", line)) * 4
        ctx.examples = render(retrieval.top(entry.text, size, compiler_of(cfg, unit), {symbol}, examples))
    return unit, source, src, span, locate, ctx


def unit_layouts(cfg: Config, unit: dict, src: str):
    """The struct layouts of the unit's `-sym on` compile (as v6_draft makes them), or None."""
    from .draftfields import Layouts
    from .elf import Elf
    from .reorder import compile_variant
    work = cfg.root / LLM_DIR / "work" / f"{os.getpid()}_{threading.get_ident()}_annot"
    try:
        obj, _ = compile_variant(cfg, unit, src, work, ("-sym", "on"))
        return Layouts(Elf.parse(obj.read_bytes())) if obj is not None else None
    finally:
        shutil.rmtree(work, ignore_errors=True)


def annotated_asm(cfg: Config, unit: dict, src: str, symbol: str, asm: str, index=None, lay=None) -> str:
    """v8: the compact asm with llmannot's comments (layouts compiled when v6_draft did not)."""
    from . import llmannot
    return llmannot.annotate(asm, symbol, llmannot.Evidence(lay or unit_layouts(cfg, unit, src), index, asm_entries(cfg)))


def v6_draft(cfg: Config, unit: dict, src: str, symbol: str, draft: str, visible: str, layouts: list | None = None) -> str:
    """m2c's `p->unkNN` as the declared member at that offset (layouts from the unit's own `-sym on`
    compile), and a constructor's or destructor's vtable stores and `return this;` dropped.
    `layouts`: a list that gets the Layouts (or None), for v7's draft_types."""
    from .demangle import demangle
    from .draftfields import Layouts, clean_structors, name_fields
    from .elf import Elf
    from .reorder import compile_variant
    work = cfg.root / LLM_DIR / "work" / f"{os.getpid()}_layout"
    try:
        obj, _ = compile_variant(cfg, unit, src, work, ("-sym", "on"))
        lay = Layouts(Elf.parse(obj.read_bytes())) if obj is not None else None
    finally:
        shutil.rmtree(work, ignore_errors=True)
    try:
        dm = demangle(symbol)
    except Exception:
        dm = None
    if layouts is not None:
        layouts.append(lay)
    if lay is not None:
        draft = name_fields(draft, lay, dm.quals[-1] if dm is not None and dm.quals else None,
                            [p.text for p in dm.params] if dm is not None else [], visible)
    return clean_structors(draft, symbol)


def unit_notes(entry, draft: str, visible: str) -> str:
    """v6: notes for the compile errors the logs show most, only where this function needs them."""
    out = []
    calls = set(entry.calls) | set(entry.refs)
    if calls & {"__nw__FUlPCci", "__nwa__FUlPCci"}:
        out.append("Allocate with the game's `void* operator new(unsigned long size, const char* name, int heap)` "
                   "(`__nw__FUlPCci`; `operator new[]` is `__nwa__FUlPCci`): write `new (\"name\", heap) T(args)` or "
                   "`new (\"name\", heap) T[n]`, never a call by the mangled name.")
    if any(c.startswith("__vt__") for c in calls) or "__vt__" in draft:
        out.append("Constructors and destructors write the vtable pointer themselves: never assign `__vt__...` or a "
                   "vtable member. Where the draft stores a vtable into a stack slot (`sp8 = &__vt__X;`), the code "
                   "constructs a local `X` object there.")
    for name in sorted(set(re.findall(r"\blbl_[0-9A-Fa-f]{8}\b", draft)) | {c for c in calls if c.startswith("lbl_")}):
        m = re.search(rf"^\s*(?:extern\s+|static\s+)*((?:(?:const|volatile|unsigned|signed|struct)\s+)*[\w:]+(?:\s*\*)*)"
                      rf"\s+{name}\s*(\[[^\]]*\](?:\s*\[[^\]]*\])*)\s*[;=]", visible, re.M)
        if m:
            out.append(f"`{name}` is declared `{' '.join(m[1].split())} {name}{m[2].replace(' ', '')}`: an array. "
                       f"Index it (`{name}[i]`) or pass `{name}` itself; never `&{name}` or `{name}->field`.")
    if not out:
        return ""
    return "## Notes for this function\n" + "\n".join(f"- {x}" for x in out[:6])


def open_log(cfg: Config, run_name: str):
    path = data_dir(cfg) / f"{run_name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("a")

    def log(row: dict) -> None:
        row.setdefault("time", round(time.time(), 2))
        fh.write(json.dumps(row) + "\n")
        fh.flush()
    return path, fh, log


def settings_from(args: argparse.Namespace) -> Settings:
    from .llmsearch import configure_settings
    st = _settings_from(args)
    configure_settings(st, args)
    return st


def search_tag(st: Settings) -> dict:
    """A run row's search fields: {} for the default greedy search (llmsearch.tag)."""
    from .llmsearch import tag
    return tag(st)


def _settings_from(args: argparse.Namespace) -> Settings:
    think = {"on": True, "off": False, "default": None}[args.think]
    return Settings(args.model, args.n, args.rounds, think, args.num_ctx, args.num_predict, args.temperature,
                    args.top_p, args.top_k, args.seed,
                    Ingredients(not args.no_asm, not args.no_decls, not args.no_diff, args.m2c), not args.no_hints,
                    not args.no_prefill, args.min_p, args.prompt, args.examples, diag=getattr(args, "diag", False),
                    repair=bool(getattr(args, "repair", False)), rank=getattr(args, "rank", None) or "fuzzy",
                    tool_block=bool(getattr(args, "tool_block", False)), tool_hints=tuple(getattr(args, "tool_hint", None) or ()))


def retrieval_for(cfg: Config, st: Settings, rev: str = "HEAD"):
    """The retrieval index when the settings ask for examples, built at `rev` (cached per commit)."""
    if st.examples <= 0:
        return None
    from .retrieval import Index
    return Index.load(cfg, _git(cfg, "rev-parse", rev).strip(), current=rev == "HEAD")


def run_name(args, prefix: str) -> str:
    if args.run:
        return args.run
    model = re.sub(r"[^A-Za-z0-9.]+", "-", args.model)
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{model}-think{args.think}"


def run_one(cfg: Config, args: argparse.Namespace) -> int:
    st = settings_from(args)
    try:
        unit, source, src, span, locate, ctx = prepare(cfg, args.unit, args.function, trim=not args.full_decls,
                                                       clean=not args.raw_draft, version=st.prompt,
                                                       retrieval=retrieval_for(cfg, st), examples=st.examples)
    except Exception as e:                          # MatchError, missing asm
        print(f"error: {e}")
        return USAGE
    scorer = Scorer(cfg, unit, args.function, src, span, locate)
    scorer.repair = st.repair
    scorer.reach = reach_for(cfg, unit, args.function)
    try:
        start, rows = scorer.baseline()
    except RuntimeError as e:
        print(f"error: {e}")
        return MISMATCH
    client = client_for(args)
    if not model_ready(client, st.model):
        return USAGE
    path, fh, log = open_log(cfg, run_name(args, "try"))
    log({"kind": "run", "model": st.model, "options": st.options(st.seed), "think": st.think, "n": st.n,
         "prompt_version": st.prompt, "examples": st.examples, "decls": "full" if args.full_decls else "trimmed",
         "draft": "raw" if args.raw_draft else "clean",
         "rounds": st.rounds, "ingredients": st.ing.tag(), "unit": unit["name"], "function": args.function,
         "repair": st.repair, "rank": st.rank, "reach": scorer.reach is not None, **search_tag(st)})
    print(f"{args.function}: now {start:.2f}%" + ("" if ctx.current else " (not written)")
          + f"; {st.model}, {st.n} x {st.rounds}, think {args.think}")
    try:
        out = search(ctx, ctx.current, start, rows, scorer.evaluate, client, st, log,
                     {"unit": unit["name"]}, jobs=min(st.n, os.cpu_count() or 4), start_shape=scorer.base_shape)
    finally:
        log({"kind": "loaded", "models": client.loaded()})
        fh.close()
    print(f"best {out.best:.2f}% (was {out.start:.2f}%), {out.attempts} attempts; log {cfg.rel(path)}")
    if out.exact_code:
        print("EXACT and judged faithful:\n" + out.exact_code)
        if args.apply:
            source.write_text(out.exact_text)
            print(f"  applied to {cfg.rel(source)}")
            hint(f"check {unit['name']} --function {args.function} --private")
        return OK
    if out.best_code and out.best > out.start:
        best = data_dir(cfg) / "best" / f"{hashlib.sha1(args.function.encode()).hexdigest()[:8]}.cpp"
        best.parent.mkdir(parents=True, exist_ok=True)
        best.write_text(out.best_code)
        print(f"  best variant written to {cfg.rel(best)}")
        hint(f"try {unit['name']} {args.function} {cfg.rel(best)} --show 1")
    return MISMATCH


def reach_for(cfg: Config, unit: dict, symbol: str):
    """text -> regs.reach_report(...) when the oracle runner has contract v2 (DK_ORACLE), else None."""
    from . import oracle
    cmd = oracle.runner(cfg)
    if cmd is None:
        return None
    probe = cfg.root / LLM_DIR / "work" / f"{os.getpid()}_reachprobe"
    try:
        if not oracle.supports_solve(cmd, probe):
            return None
    finally:
        shutil.rmtree(probe, ignore_errors=True)
    from .regs import reach_report
    return lambda text: reach_report(cfg, unit, symbol, text, say=lambda _s: None)


def _model_args(p: argparse.ArgumentParser, n: int = 8, rounds: int = 2, model: str = DEFAULT_MODEL) -> None:
    p.add_argument("--model", default=model, help=f"Ollama model (default {model})")
    p.add_argument("--url", default=os.environ.get("DK_OLLAMA", DEFAULT_URL), help="Ollama server (DK_OLLAMA)")
    p.add_argument("--backend", choices=("ollama", "openai"), default="ollama", help="openai: llama.cpp's llama-server at --url (base GGUF + --lora checkpoints; --model lora=N or base)")
    p.add_argument("--n", type=int, default=n, help=f"variants per round (default {n})")
    p.add_argument("--rounds", type=int, default=rounds, help=f"rounds, each fed the best variant and its diff (default {rounds})")
    p.add_argument("--think", choices=("on", "off", "default"), default="off", help="Qwen3 thinking (default off)")
    p.add_argument("--num-ctx", type=int, default=16384, help="context window (default 16384)")
    p.add_argument("--num-predict", type=int, default=0, help="max reply tokens (default 2048, 8192 thinking)")
    p.add_argument("--temperature", type=float, default=1.0, help="sampling temperature (default 1.0: N variants need spread)")
    p.add_argument("--top-p", type=float, default=1.0, help="nucleus cut (default 1.0: off; 0.95 collapsed rewrites to copies)")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--min-p", type=float, default=0.05, help="drop tokens below this share of the top one (default 0.05)")
    p.add_argument("--full-decls", action="store_true", help="the unit's whole declarations, not only what the function uses (A/B)")
    p.add_argument("--raw-draft", action="store_true", help="the m2c draft as m2c wrote it (types not mapped; A/B)")
    p.add_argument("--seed", type=int, default=1, help="first seed (variant k of round r: seed + 1000 r + k)")
    p.add_argument("--m2c", choices=("auto", "always", "never"), default="auto",
                   help="show the m2c draft: auto = only for an unwritten function")
    p.add_argument("--no-asm", action="store_true", help="leave the target asm out of the prompt (ablation)")
    p.add_argument("--no-decls", action="store_true", help="leave the declarations out (ablation)")
    p.add_argument("--no-diff", action="store_true", help="leave the diff out (ablation)")
    p.add_argument("--no-prefill", action="store_true", help="let the model open its own reply (default: the reply "
                   "starts with ```cpp and stops at the closing fence; never when thinking)")
    p.add_argument("--no-hints", action="store_true", help="the same prompt for every variant (no rewrite hints)")
    p.add_argument("--tool-block", action="store_true", help="T63: a structured TOOLS line (score, diff rows, hints) in every repair prompt")
    p.add_argument("--tool-hint", action="append", default=[], type=lambda h: h if __import__("re").fullmatch(
        r"rule:[a-z0-9][a-z0-9_-]{0,40}|xform:[a-z0-9][a-z0-9_-]{0,40}@[0-9]{1,5}|region:[A-Za-z0-9:_-]{1,24}", h) else
        (_ for _ in ()).throw(argparse.ArgumentTypeError("hint must be rule:<id>, xform:<name>@<line> or region:<id>")),
        help="with --tool-block: a structured hint token (free text is rejected)")
    p.add_argument("--diag", action="store_true", help="round 2: repair rounds prefill `DIAG:` (a model trained with "
                   "dataset export --diag-share writes a one-line diagnosis, then the code; logged as `diag`)")
    p.add_argument("--repair", action=argparse.BooleanOptionalAction, default=False, help="declaration-visibility repair (a type, global or "
                   "function the unit declares below the function made visible above it, then compiled again) and an "
                   "invented-name / real-prototype note after a compile error (llmrepair.py); on by default in llm-try and llm-grind")
    p.add_argument("--rank", choices=("shape", "fuzzy"), default="fuzzy", help="which attempt feeds the next round: "
                   "shape = an attempt equal to the target with registers masked first, then the score (the default "
                   "in llm-try and llm-grind); fuzzy = the score only (the benchmarks' default). With an oracle runner (DK_ORACLE, contract v2) each shape-equal attempt "
                   "also gets `regs --reach` (logged as `reach`)")
    p.add_argument("--run", default="", help="log name (llm-data/<run>.jsonl; default: time and model)")
    p.add_argument("--prompt", type=int, choices=PROMPT_VERSIONS, default=PROMPT_VERSION,
                   help=f"prompt template version (default {PROMPT_VERSION}; 6: field names in the draft, notes; "
                        "7: v6 + the declarations of base classes, touched types and callees from other headers; "
                        "8: v7 + the target asm annotated with members, globals, calls and literals)")
    p.add_argument("--examples", type=int, default=DEFAULT_EXAMPLES, help="retrieval: K similar exact functions (asm "
                   f"and source) first in the prompt, about 2.7k tokens at most (default {DEFAULT_EXAMPLES}; 0: off)")
    from .llmsearch import add_args
    add_args(p)


def _configure_try(p: argparse.ArgumentParser) -> None:
    p.add_argument("unit", help="unit (source path or name)")
    p.add_argument("function", help="function symbol, as dk check lists it")
    _model_args(p)
    p.set_defaults(repair=True, rank="shape")
    p.add_argument("--apply", action="store_true", help="write an exact variant the judge accepts (no device)")


TRY = Command("llm-try", "matching", "ask a local model (Ollama) for N definitions of a function over R rounds, "
              "score each like dk try, log every attempt to llm-data/ beside the main checkout", run_one, _configure_try)


# ---- the benchmark -----------------------------------------------------------------------------------

def area(unit_name: str) -> str:
    parts = unit_name.split("/")
    parts = parts[1:] if parts[0] in ("main",) else parts
    return "/".join(parts[:2]) if parts[0] in ("game", "lib") and len(parts) > 2 else parts[0]


_REG = re.compile(r"\b[rf]\d+\b|\bcr\d\b")


def classify(left: dict, right: dict) -> str:
    """"register" when ours has the target's instructions and only register numbers differ, else "shape"."""
    li, ri = left.get("instructions", []), right.get("instructions", [])
    if len(li) != len(ri):
        return "shape"
    from .diffcmd import _text
    for t, o in zip(li, ri):
        if t.get("diff_kind") or o.get("diff_kind"):
            if _REG.sub("R", _text(t)) != _REG.sub("R", _text(o)):
                return "shape"
    return "register"


def _pick(rows: list[dict], n: int) -> list[dict]:
    """n rows spread over areas: round-robin over areas, each area's rows in a fixed hashed order."""
    by: dict[str, list[dict]] = {}
    for r in sorted(rows, key=lambda r: hashlib.sha1(r["function"].encode()).hexdigest()):
        by.setdefault(r["area"], []).append(r)
    out = []
    while len(out) < n and any(by.values()):
        for a in sorted(by):
            if by[a] and len(out) < n:
                out.append(by[a].pop(0))
    return out


def _git(cfg: Config, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cfg.root, capture_output=True, text=True).stdout.strip()


def freeze(cfg: Config, args: argparse.Namespace) -> int:
    from .matching import load_objdiff_units, score, MatchError
    from .regs import _diff
    from .trycmd import definition
    from . import freshness as fresh
    from .jobs import resolve
    out = Path(args.out) if args.out else data_dir(cfg) / "bench.json"
    if out.exists() and not args.force:
        print(f"error: {cfg.rel(out)} exists (a benchmark stays frozen; --force replaces it)")
        return USAGE
    units = [u for u in load_objdiff_units(cfg) if u.get("base_path") and not u["metadata"].get("complete")]
    fresh.build(cfg, ["-k", "0", "-j", str(resolve(0, cfg)), *(u["base_path"] for u in units)])
    sizes = {e.name: e.text.count("\n") for e in asm_entries(cfg).values() if e.kind == "fn"}

    def one(u: dict) -> list[dict]:
        base = cfg.root / u["base_path"]
        if not base.is_file():
            return []
        try:
            s = score(cfg, u["name"], cfg.root / u["target_path"], base)
        except MatchError:
            return []
        src_path = cfg.root / u["metadata"]["source_path"]
        src = src_path.read_text(errors="replace")
        rows = []
        for f in s.functions:
            if f.name.startswith("@") or f.name.startswith("__sinit"):
                continue
            row = {"unit": u["name"], "function": f.name, "percent": f.percent, "size": f.size,
                   "area": area(u["name"]), "source": u["metadata"]["source_path"]}
            if args.min <= f.percent < args.max and f.size <= args.max_size:
                pair = _diff(cfg, u, base, f.name)
                if pair is not None:
                    rows.append({**row, "kind": classify(*pair)})
            elif f.percent == 0.0 and 0x20 <= f.size < args.zero_size and definition(src, f.name) is None \
                    and f.name in sizes:
                rows.append({**row, "kind": "unwritten"})
        return rows

    with ThreadPoolExecutor(max_workers=resolve(0, cfg)) as pool:
        rows = [r for rs in pool.map(one, units) for r in rs]
    reg = _pick([r for r in rows if r["kind"] == "register"], args.near // 2)
    shape = _pick([r for r in rows if r["kind"] == "shape"], args.near - len(reg))
    zero = _pick([r for r in rows if r["kind"] == "unwritten"], args.zero)
    head = _git(cfg, "rev-parse", "HEAD")
    chosen = reg + shape + zero
    for r in chosen:
        r["source_commit"] = _git(cfg, "log", "-1", "--format=%H", "--", r["source"]) or head
        r["source_sha1"] = hashlib.sha1((cfg.root / r["source"]).read_bytes()).hexdigest()
    doc = {"never_train": True, "note": "Benchmark functions: never put them (or their units' later exact source) "
           "in a training set.", "frozen": time.strftime("%Y-%m-%d %H:%M"), "head": head,
           "selection": {"near": [args.min, args.max], "max_size": args.max_size, "zero_size": args.zero_size,
                         "candidates": {k: sum(1 for r in rows if r["kind"] == k) for k in ("register", "shape", "unwritten")}},
           "functions": chosen}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"froze {len(chosen)} functions ({len(reg)} register, {len(shape)} shape, {len(zero)} unwritten) "
          f"from {doc['selection']['candidates']} candidates into {cfg.rel(out)}")
    for r in chosen:
        print(f"  {r['kind']:<9} {r['percent']:6.2f}%  0x{r['size']:<4X} {r['area']:<12} {r['unit']} {r['function']}")
    hint("llm-bench run --model qwen3:14b")
    return OK


HELDOUT_SALT = "heldout-v1"         # default; a project sets its own once ([platform.llm] heldout_salt) and never
                                    # changes it: another salt picks another (possibly trained-on) set


@functools.lru_cache(maxsize=None)
def llm_option(key: str, default: str) -> str:
    """[platform.llm] <key> from the project's dk.toml, else `default` (no project: the toolkit's tests)."""
    from ...core.config import ConfigError, load
    try:
        return str((load().platform_options.get("llm") or {}).get(key, default))
    except (ConfigError, OSError, ValueError):
        return default


def _vendored_paths(cfg: Config) -> list[str]:
    import tomllib
    doc = tomllib.loads((cfg.root / "dk.toml").read_text())
    return [x.rstrip("/") + "/" if not x.endswith((".c", ".cpp")) else x for x in doc.get("quality", {}).get("exclude", [])]


def heldout_pick(name: str, share: int = 20) -> bool:
    """A unit is held out when its salted hash falls in 1/share of the space (stable, ~5%)."""
    return int(hashlib.sha1(f"{llm_option('heldout_salt', HELDOUT_SALT)}:{name}".encode()).hexdigest()[:8], 16) % share == 0


def exact_functions(cfg: Config, u: dict, src: str, min_size: int = 0x10) -> list[dict]:
    """The unit's exact functions that have their own (not asm) definition in its source."""
    from .matching import score, MatchError
    from .trycmd import definition
    base = cfg.root / u["base_path"]
    if not base.is_file():
        return []
    try:
        s = score(cfg, u["name"], cfg.root / u["target_path"], base)
    except MatchError:
        return []
    entries = asm_entries(cfg)
    out = []
    for f in s.functions:
        if f.percent != 100.0 or f.name.startswith(("@", "__sinit")) or f.size < min_size or f.name not in entries:
            continue
        span = definition(src, f.name)
        if span is None or re.match(r"\s*(?:extern\s+\"C\"\s+)?asm\b", src[span[0]:span[1]]):
            continue
        out.append({"unit": u["name"], "function": f.name, "size": f.size, "asm_shape": asm_shape(entries[f.name].text)})
    return out


def holdout(cfg: Config, args: argparse.Namespace) -> int:
    """Pick the held-out units once: ~5% of the linked game-side units by a salted hash (benchmark
    units, vendored SDK/runtime ports and units with no own exact definition left out)."""
    from .matching import load_objdiff_units
    from . import freshness as fresh
    from .jobs import resolve
    out = Path(args.out) if args.out else data_dir(cfg) / "heldout.json"
    if out.exists() and not args.force:
        print(f"error: {cfg.rel(out)} exists (the held-out set stays fixed; --force replaces it)")
        return USAGE
    bench_units = {r["unit"] for r in load_bench(cfg, None)["functions"]} if (data_dir(cfg) / "bench.json").is_file() else set()
    vend = _vendored_paths(cfg)
    pool = [u for u in load_objdiff_units(cfg) if u.get("base_path") and u["metadata"].get("complete")
            and "game" in u["metadata"].get("progress_categories", []) and u["name"] not in bench_units
            and not any(u["metadata"]["source_path"].startswith(v) for v in vend)]
    picked = [u for u in pool if heldout_pick(u["name"])]
    fresh.build(cfg, ["-k", "0", "-j", str(resolve(0, cfg)), *(u["base_path"] for u in picked)])
    head = _git(cfg, "rev-parse", "HEAD")
    units, fns = [], []
    for u in sorted(picked, key=lambda u: u["name"]):
        src_rel = u["metadata"]["source_path"]
        got = exact_functions(cfg, u, (cfg.root / src_rel).read_text(errors="replace"))
        if not got:
            continue
        commit = _git(cfg, "log", "-1", "--format=%H", "--", src_rel) or head
        units.append({"unit": u["name"], "source": src_rel, "source_commit": commit, "functions": len(got)})
        fns += [{**f, "source": src_rel, "source_commit": commit, "kind": "heldout", "area": area(u["name"])} for f in got]
    doc = {"never_train": True, "note": "Held-out exact units: never put them, or any function with one of these asm "
           "shapes, in a training set. Run only to judge an adapter up for promotion.", "salt": llm_option("heldout_salt", HELDOUT_SALT),
           "share": "1/20 by sha1", "picked": time.strftime("%Y-%m-%d %H:%M"), "head": head,
           "pool_units": len(pool), "units": units, "functions": fns}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"held out {len(units)} of {len(pool)} linked game-side units ({len(fns)} exact functions) into {cfg.rel(out)}")
    for u in units:
        print(f"  {u['functions']:4d}  {u['unit']}")
    return OK


def never_train(cfg: Config, bench_path: str | None = None) -> tuple[set[str], set[str], set[str]]:
    """(units, functions, asm shapes) no training record may hold: the benchmark's and the held-out set's."""
    units, fns, shapes = set(), set(), set()
    bp = Path(bench_path) if bench_path else data_dir(cfg) / "bench.json"
    entries = asm_entries(cfg)
    for path in (bp, data_dir(cfg) / "heldout.json"):
        if not path.is_file():
            continue
        doc = json.loads(path.read_text())
        for r in doc["functions"]:
            units.add(r["unit"])
            fns.add(r["function"])
            shape = r.get("asm_shape") or (asm_shape(entries[r["function"]].text) if r["function"] in entries else None)
            if shape:
                shapes.add(shape)
    from . import gate                                 # the function-level frozen sets: gate-lora2, the near-samples
    g_fns, g_shapes = gate.never(cfg)
    return units, fns | set(g_fns), shapes | set(g_shapes)


def load_bench(cfg: Config, path: str | None) -> dict:
    return json.loads((Path(path) if path else data_dir(cfg) / "bench.json").read_text())


def bench_run(cfg: Config, args: argparse.Namespace) -> int:
    if args.set.startswith("review"):                 # the overseer eval (round 2): llmreview.py
        from .llmreview import bench_review
        return bench_review(cfg, args)
    st = settings_from(args)
    kind, _, part = args.set.partition("-")           # heldout-dev, heldout-final, repair-dev, repair-final
    heldout, repair = kind == "heldout", kind == "repair"
    bench = json.loads((Path(args.bench) if args.bench else data_dir(cfg) / f"{'repair-eval' if repair else 'heldout'}.json")
                       .read_text()) if heldout or repair else load_bench(cfg, args.bench)
    items = bench["functions"]
    if repair:                                        # a held-out function's wrong start (the model's own attempt)
        items = [{**r, "kind": "repair", "rid": f"{r['function']}#{i}"} for i, r in enumerate(items)]
    split = None
    if part:
        try:
            split = load_split(cfg)
        except (OSError, ValueError) as e:
            print(f"error: {e}")
            return USAGE
        names = split[part]
        if heldout:                                   # the split's own order (dev: the 47 of P1, as run then)
            pos = {f: i for i, f in enumerate(names)}
            items = sorted((r for r in items if r["function"] in pos), key=lambda r: pos[r["function"]])
        else:
            items = [r for r in items if r["function"] in set(names)]
    if heldout and args.max_size and not part:
        items = [r for r in items if r["size"] <= args.max_size]
    if heldout and not part:                          # a spread: every unit before a second from any
        items = _pick(items, len(items))
    if args.only:
        items = [r for r in items if r["kind"] in args.only.split(",")]
    if args.every > 1:
        items = items[::args.every]
    if args.limit:
        items = items[:args.limit]
    client = client_for(args)
    if not model_ready(client, st.model):
        return USAGE
    index = retrieval_for(cfg, st, "HEAD" if heldout else bench.get("head") or "HEAD")   # the freeze's sources
    name = run_name(args, "bench")
    path, fh, log = open_log(cfg, name)
    log({"kind": "run", "model": st.model, "options": st.options(st.seed), "think": st.think, "n": st.n,
         "prompt_version": st.prompt, "examples": st.examples, "decls": "full" if args.full_decls else "trimmed",
         "draft": "raw" if args.raw_draft else "clean",
         "rounds": st.rounds, "ingredients": st.ing.tag(), "bench_head": bench.get("head"), "functions": len(items),
         "prefill": st.prefilled(), "set": args.set, **({"split_made": split.get("made")} if split else {}),
         **search_tag(st)})
    done_before = set()
    if args.resume and path.stat().st_size:
        done_before = {json.loads(x).get("rid") or json.loads(x).get("function") for x in path.read_text().splitlines()
                       if '"kind": "function"' in x}
    until = _until(args.until)
    try:
        for i, r in enumerate(items):
            if r.get("rid", r["function"]) in done_before:
                continue
            if until and time.time() >= until:
                print(f"stopping at {args.until} (--until); rerun with --resume --run {name} to go on")
                break
            frozen = _git(cfg, "show", f"{r['source_commit']}:{r['source']}") if r.get("source_commit") else None
            try:
                if repair:                            # the exact definition replaced by the wrong start
                    from .trycmd import definition, variant_text
                    frozen = variant_text(frozen, r["function"], r["start_code"], definition(frozen, r["function"]), None)
                unit, source, src, span, locate, ctx = prepare(cfg, r["unit"], r["function"], frozen or None,
                                                               trim=not args.full_decls, clean=not args.raw_draft,
                                                               blank=heldout, version=st.prompt, retrieval=index,
                                                               examples=st.examples)
                scorer = Scorer(cfg, unit, r["function"], src, span, locate)
                start, rows = scorer.baseline()
                if heldout:                           # the answer is in the source: the model starts from nothing
                    if start != 100.0:
                        raise RuntimeError(f"not exact any more ({start:.2f}%)")
                    start, rows = 0.0, []
            except Exception as e:
                print(f"[{i + 1}/{len(items)}] {r['function']}: skipped: {e}")
                log({"kind": "skip", "function": r["function"], "error": str(e)[:300]})
                continue
            print(f"[{i + 1}/{len(items)}] {r['kind']} {r['function']} ({r['unit']}): {start:.2f}%"
                  + (f", 0x{r['size']:X} bytes, blanked" if heldout else ""))
            t0 = time.time()
            out = search(ctx, ctx.current, start, rows, scorer.evaluate, client, st, log,
                         {"unit": r["unit"], "bench_kind": r["kind"]}, jobs=min(st.n, os.cpu_count() or 4))
            log({"kind": "function", "function": r["function"], "unit": r["unit"], "bench_kind": r["kind"],
                 **({"rid": r["rid"]} if repair else {}), "start": start, "best": out.best, "exact": bool(out.exact_code),
                 "attempts": out.attempts,
                 "secs": round(time.time() - t0, 1), "exact_code": out.exact_code or None})
            print(f"    best {out.best:.2f}%" + ("  EXACT (judged)" if out.exact_code else ""))
            if i == 0:
                log({"kind": "loaded", "models": client.loaded()})
    finally:
        fh.close()
    print(f"log {cfg.rel(path)}")
    hint(f"llm-bench report {cfg.rel(path)}")
    return OK


def _until(hhmm: str) -> float:
    """--until HH:MM as a time stamp: the next such time (tomorrow when it has passed today)."""
    if not hhmm:
        return 0.0
    h, m = (int(x) for x in hhmm.split(":"))
    now = time.localtime()
    t = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, h, m, 0, 0, 0, -1))
    return t + 86400 if t <= time.time() else t


def device_rate(rows: list[dict]) -> float:
    """% of the generated variants (attempts with code) that carry a device."""
    v = [r for r in rows if r["kind"] == "attempt" and r.get("format") != "error" and r.get("variant")]
    return round(100.0 * sum(1 for r in v if r.get("devices")) / len(v), 1) if v else 0.0


def summarize(rows: list[dict]) -> dict:
    """One run's numbers from its log rows."""
    run = next((r for r in rows if r["kind"] == "run"), {})
    att = [r for r in rows if r["kind"] == "attempt"]
    fns = [r for r in rows if r["kind"] == "function"]
    loaded = next((r for r in rows if r["kind"] == "loaded"), {}).get("models", [])
    gen = [r for r in att if r.get("format") != "error"]

    def rate(xs, pred):
        return round(100.0 * sum(1 for x in xs if pred(x)) / len(xs), 1) if xs else 0.0

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 2) if xs else 0.0
    near = [f for f in fns if f.get("bench_kind") not in ("unwritten", "heldout", "repair")]
    zero = [f for f in fns if f.get("bench_kind") == "unwritten"]
    held = [f for f in fns if f.get("bench_kind") == "heldout"]
    rep = [f for f in fns if f.get("bench_kind") == "repair"]
    return {
        "model": run.get("model"), "think": run.get("think"), "ingredients": run.get("ingredients"),
        "n x rounds": f"{run.get('n')}x{run.get('rounds')}", "functions": len(fns), "attempts": len(att),
        "prefill": run.get("prefill", False),
        "truncated": rate(gen, lambda r: r.get("done_reason") == "length" or r.get("format") == "truncated"),
        "format_ok": rate(gen, lambda r: r.get("format") in ("fenced", "several", "bare", "prefill", "diag")),
        "compile": rate(gen, lambda r: r.get("compiled")),
        "compile_of_complete": rate([r for r in gen if r.get("done_reason") != "length"], lambda r: r.get("compiled")),
        "devices": device_rate(rows),
        "improved": f"{sum(1 for f in near if f['best'] > f['start'])}/{len(near)}",
        "mean_gain_near": mean([f["best"] - f["start"] for f in near]),
        "exact_near": f"{sum(1 for f in near if f['exact'])}/{len(near)}",
        "unwritten_best": mean([f["best"] for f in zero]),
        "exact_unwritten": f"{sum(1 for f in zero if f['exact'])}/{len(zero)}",
        "heldout_best": mean([f["best"] for f in held]),
        "exact_heldout": f"{sum(1 for f in held if f['exact'])}/{len(held)}",
        "exact_repair": f"{sum(1 for f in rep if f['exact'])}/{len(rep)}",
        "repair_gain": mean([f["best"] - f["start"] for f in rep]),
        "s_per_variant": mean([r.get("secs") for r in gen]),
        "gen_tps": mean([r.get("gen_tps") for r in gen]),
        "prompt_tokens_max": max([r.get("prompt_tokens") or 0 for r in gen] or [0]),
        "gen_tokens_mean": mean([r.get("gen_tokens") for r in gen]),
        "vram_mb": (loaded[0].get("size_vram", 0) // 2**20) if loaded else None,
        "size_mb": (loaded[0].get("size", 0) // 2**20) if loaded else None,
    }


def _compiled_per_function(rows: list[dict]) -> dict[str, int]:
    """Compiled attempts per finished function; a resumed function's interrupted attempts (before
    its last round-0 prompt) are left out."""
    cur: dict[str, int] = {}
    out: dict[str, int] = {}
    for r in rows:
        if r["kind"] == "prompt" and r.get("round") == 0:
            cur[r["function"]] = 0
        elif r["kind"] == "attempt" and r.get("compiled"):
            cur[r["function"]] = cur.get(r["function"], 0) + 1
        elif r["kind"] == "function":
            out[r["function"]] = cur.get(r["function"], 0)
    return out


def _sign_p(b: int, c: int) -> float:
    import math
    return min(1.0, 2 * sum(math.comb(b + c, i) for i in range(min(b, c) + 1)) / 2 ** (b + c)) if b + c else 1.0


def paired(runs: list[list[dict]]) -> list[dict]:
    """Each run against the first, on the functions every run finished (same functions, same
    seeds): exact count, the exact matches only one side has with a two-sided sign-test p, the best
    score's wins and losses, and the mean difference with its 95% interval; the functions where more
    (fewer) attempts compiled, with their sign-test p."""
    fns = [{r.get("rid") or r["function"]: r for r in rows if r["kind"] == "function"} for rows in runs]
    comp = [_compiled_per_function(rows) for rows in runs]
    common = sorted(set.intersection(*(set(f) for f in fns))) if fns else []
    out = []
    for f, cf in zip(fns, comp):
        b = sum(1 for k in common if f[k]["exact"] and not fns[0][k]["exact"])
        c = sum(1 for k in common if fns[0][k]["exact"] and not f[k]["exact"])
        p = _sign_p(b, c)
        cu = sum(1 for k in common if cf.get(k, 0) > comp[0].get(k, 0))
        cd = sum(1 for k in common if cf.get(k, 0) < comp[0].get(k, 0))
        d = [float(f[k]["best"]) - float(fns[0][k]["best"]) for k in common]
        m = sum(d) / len(d) if d else 0.0
        sd = (sum((x - m) ** 2 for x in d) / (len(d) - 1)) ** 0.5 if len(d) > 1 else 0.0
        out.append({"paired": len(common), "exact": sum(1 for k in common if f[k]["exact"]),
                    "exact_only_here": b, "exact_only_first": c, "sign_p": round(p, 3),
                    "best_up": sum(1 for x in d if x > 0), "best_down": sum(1 for x in d if x < 0),
                    "best_diff": round(m, 2), "ci95": round(1.96 * sd / len(d) ** 0.5, 2) if d else 0.0,
                    "best_mean": round(sum(float(f[k]["best"]) for k in common) / len(common), 2) if common else 0.0,
                    "compile_up": cu, "compile_down": cd, "compile_p": round(_sign_p(cu, cd), 3)})
    return out


def bench_report(cfg: Config, args: argparse.Namespace) -> int:
    if getattr(args, "showdown", ""):
        from . import showdown
        return showdown.run(cfg, args)
    logs = [Path(p) for p in args.logs] or sorted(data_dir(cfg).glob("bench-*.jsonl"))
    if not logs:
        print(f"no benchmark logs in {data_dir(cfg)}")
        return USAGE
    table, runs = [], []
    for p in logs:
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        if rows and str(rows[0].get("set", "")).startswith("review"):    # an overseer eval log (llmreview.py)
            from .llmreview import print_scores, report_rows
            print(f"{p.stem} ({rows[0].get('model')}, review prompt {rows[0].get('review_prompt')}):")
            print_scores(report_rows(rows))
            continue
        runs.append(rows)
        table.append({"log": p.stem, **summarize(rows)})
    if not table:
        return OK
    if args.paired:
        table = [{"log": t["log"], **x} for t, x in zip(table, paired(runs))]
    keys = list(table[0])
    print("| " + " | ".join(keys) + " |")
    print("|" + "---|" * len(keys))
    for t in table:
        print("| " + " | ".join(str(t[k]) for k in keys) + " |")
    if args.json:
        Path(args.json).write_text(json.dumps(table, indent=1))
    return OK


SPLIT_FILE = "heldout-split.json"
DEV_MAX, DEV_EVERY, FINAL_MAX = 0x200, 3, 0x300
CHOICE_LOGS = ("bake-*held*.jsonl", "p1-*.jsonl")      # held-out runs that chose something (model, prompt)


def heldout_split(functions: list[dict], used: set[str]) -> dict:
    """dev: the held-out functions up to 0x200 bytes, spread over areas (_pick), every 3rd: the 47 that chose
    prompt v6; they choose checkpoints and quants too. final: the rest up to 0x300, minus every function a
    choice already saw (`used`: the bake-off's, P1's); only the go/no-go gate reads it."""
    small = [r for r in functions if r["size"] <= DEV_MAX]
    dev = [r["function"] for r in _pick(small, len(small))[::DEV_EVERY]]
    rest = [r for r in functions if r["size"] <= FINAL_MAX and r["function"] not in set(dev)]
    final = [r for r in rest if r["function"] not in used]
    return {"dev": dev, "final": [r["function"] for r in _pick(final, len(final))],
            "excluded": sorted(r["function"] for r in rest if r["function"] in used)}


def load_split(cfg: Config) -> dict:
    path = data_dir(cfg) / SPLIT_FILE
    if not path.is_file():
        raise ValueError(f"no {path}: run `./dk llm-bench split` once (it freezes dev and final)")
    return json.loads(path.read_text())


def split_cmd(cfg: Config, args: argparse.Namespace) -> int:
    d = data_dir(cfg)
    path = d / SPLIT_FILE
    if path.exists() and not args.force:
        print(f"{path} exists: the split is frozen (--force makes it again, which would undo its point)")
        return USAGE
    functions = json.loads((d / "heldout.json").read_text())["functions"]
    held = {r["function"] for r in functions}
    logs = sorted({p for g in (args.choice_logs or CHOICE_LOGS) for p in d.glob(g)})
    used: dict[str, list[str]] = {}
    for p in logs:
        for line in p.read_text().splitlines():
            if '"kind": "function"' in line:
                f = json.loads(line).get("function")
                if f in held:
                    used.setdefault(f, []).append(p.name)
    sp = heldout_split(functions, set(used))
    sp.update({"made": time.strftime("%Y-%m-%d %H:%M"), "rule": f"dev: <= 0x{DEV_MAX:X} bytes, spread, every "
               f"{DEV_EVERY}th; final: the rest <= 0x{FINAL_MAX:X} minus functions the choice logs ran",
               "choice_logs": [p.name for p in logs], "excluded_by": {f: sorted(set(used[f])) for f in sp["excluded"]}})
    path.write_text(json.dumps(sp, indent=1) + "\n")
    rep = json.loads((d / "repair-eval.json").read_text())["functions"] if (d / "repair-eval.json").is_file() else []
    dev, fin = set(sp["dev"]), set(sp["final"])
    print(f"{cfg.rel(path)}: heldout-dev {len(sp['dev'])}, heldout-final {len(sp['final'])} "
          f"({len(sp['excluded'])} left out: a choice saw them); repair-dev "
          f"{sum(1 for r in rep if r['function'] in dev)}, repair-final {sum(1 for r in rep if r['function'] in fin)}")
    return OK


def _rows(path: str) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def gate_verdict(base: list[list[dict]], cand: list[list[dict]], final: set[str], rep_base: list[list[dict]],
                 rep_cand: list[list[dict]], rep_final: set[str], ratio: float = 1.5, alpha: float = 1.0) -> dict:
    """The go/no-go rule (the LLM plan, decided 2026-11-02) on heldout-final, the logs paired by
    position (the same seed in each arm) and cut to the final set: PASS if the candidate's exact matches
    are at least `ratio` x the baseline's, the sign test favors it (more functions only it matched than
    only the baseline matched, with p <= alpha), its device rate is not higher, and on the repair eval
    (repair-final) its exact count is not lower. Both arms must be complete (a function both skipped,
    e.g. no longer exact in the repo, is left out) and comparable (n, rounds, prompt, seeds).
    `final`: the heldout-final function names; `rep_final`: the repair-final rids (function#index)."""
    checks: list[tuple[str, bool, str]] = []

    def run_of(rows):
        return next((r for r in rows if r["kind"] == "run"), {})

    def cut(rows, keep):
        return [r for r in rows if r["kind"] not in ("function", "attempt", "prompt") or r.get("function") in keep]

    same = ("n", "rounds", "prompt_version", "examples", "decls", "draft")
    for name, b_runs, c_runs, keep in (("held-out", base, cand, final), ("repair", rep_base, rep_cand, rep_final)):
        if not b_runs or not c_runs or len(b_runs) != len(c_runs):
            checks.append((f"{name} logs paired", False, f"{len(b_runs)} baseline vs {len(c_runs)} candidate logs"))
            continue
        diffs = [k for b, c in zip(b_runs, c_runs) for k in same if run_of(b).get(k) != run_of(c).get(k)]
        seeds = [((run_of(b).get("options") or {}).get("seed"), (run_of(c).get("options") or {}).get("seed"))
                 for b, c in zip(b_runs, c_runs)]
        diffs += [f"seed {x} vs {y}" for x, y in seeds if x != y]
        checks.append((f"{name} comparable", not diffs, ", ".join(sorted(set(diffs))) or "same n, rounds, prompt, seeds"))
        fnames = {k.rsplit("#", 1)[0] for k in keep}       # repair keys are rids (function#index)
        bx = cx = wins = losses = 0
        short = []
        for b, c in zip(b_runs, c_runs):
            bf = {r.get("rid") or r["function"]: r for r in b if r["kind"] == "function"}
            cf = {r.get("rid") or r["function"]: r for r in c if r["kind"] == "function"}
            both_skipped = {r["function"] for r in b if r["kind"] == "skip"} & {r["function"] for r in c if r["kind"] == "skip"}
            want = {k for k in keep if k.rsplit("#", 1)[0] not in both_skipped}
            common = want & bf.keys() & cf.keys()
            short.append((len(common), len(want)))
            bx += sum(1 for k in common if bf[k]["exact"])
            cx += sum(1 for k in common if cf[k]["exact"])
            wins += sum(1 for k in common if cf[k]["exact"] and not bf[k]["exact"])
            losses += sum(1 for k in common if bf[k]["exact"] and not cf[k]["exact"])
        checks.append((f"{name} complete", all(n == w for n, w in short),
                       "paired " + ", ".join(f"{n}/{w}" for n, w in short)))
        if name == "held-out":
            p = _sign_p(wins, losses)
            checks.append((f"exact >= {ratio}x baseline", cx >= ratio * bx and cx > bx, f"{cx} vs {bx}"))
            checks.append(("sign test favors the candidate", wins > losses and p <= alpha,
                           f"only candidate {wins}, only baseline {losses}, p {p:.3f}" + (f" (needs <= {alpha})" if alpha < 1 else "")))
            bd = [device_rate(cut(b, fnames)) for b in b_runs]
            cd = [device_rate(cut(c, fnames)) for c in c_runs]
            mb, mc = sum(bd) / len(bd), sum(cd) / len(cd)
            checks.append(("device rate not higher", mc <= mb, f"{mc:.1f}% vs {mb:.1f}% of variants"))
        else:
            checks.append(("repair exact not lower", cx >= bx, f"{cx} vs {bx}"))
    return {"pass": all(ok for _n, ok, _w in checks), "checks": checks}


def gate_cmd(cfg: Config, args: argparse.Namespace) -> int:
    try:
        split = load_split(cfg)
    except (OSError, ValueError) as e:
        print(f"error: {e}")
        return USAGE
    final = set(split["final"])
    rep = json.loads((data_dir(cfg) / "repair-eval.json").read_text())["functions"]
    rep_final = {f"{r['function']}#{i}" for i, r in enumerate(rep) if r["function"] in final}
    v = gate_verdict([_rows(p) for p in args.baseline], [_rows(p) for p in args.candidate], final,
                     [_rows(p) for p in args.repair_baseline], [_rows(p) for p in args.repair_candidate], rep_final,
                     args.ratio, args.sign_p)
    print(f"go/no-go on heldout-final ({len(final)} functions; split made {split.get('made')}) and repair-final "
          f"({sum(1 for r in rep if r['function'] in final)} starts)")
    for name, ok, why in v["checks"]:
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}: {why}")
    print("PASS: adopt the LoRA" if v["pass"] else "FAIL: keep the baseline (the LLM plan risk 9: retrain on "
          "cleaner/teacher data or switch base before a second paid run)")
    return OK if v["pass"] else MISMATCH


def run_bench(cfg: Config, args: argparse.Namespace) -> int:
    return {"freeze": freeze, "holdout": holdout, "run": bench_run, "report": bench_report,
            "dataset": export, "split": split_cmd, "gate": gate_cmd,
            "t63-freeze": lambda c, a: __import__(__package__ + ".t63", fromlist=["freeze"]).freeze(c, a),
            "gate-freeze": lambda c, a: __import__(__package__ + ".gate", fromlist=["freeze"]).freeze(c, a)}[args.action](cfg, args)


def _configure_bench(p: argparse.ArgumentParser) -> None:
    sub = p.add_subparsers(dest="action", required=True)
    f = sub.add_parser("freeze", help="pick near misses and unwritten functions into llm-data/bench.json")
    f.add_argument("--near", type=int, default=20, help="near misses (half register-only, half shape; default 20)")
    f.add_argument("--zero", type=int, default=8, help="unwritten functions (default 8)")
    f.add_argument("--min", type=float, default=95.0, help="near-miss floor %% (default 95)")
    f.add_argument("--max", type=float, default=99.95, help="near-miss ceiling %% (default 99.95)")
    f.add_argument("--max-size", type=lambda x: int(x, 0), default=0x600, help="largest near miss in bytes (default 0x600)")
    f.add_argument("--zero-size", type=lambda x: int(x, 0), default=0x200, help="unwritten ones below this size (default 0x200)")
    f.add_argument("--out", default="", help="where (default llm-data/bench.json)")
    f.add_argument("--force", action="store_true", help="replace an existing benchmark")
    h = sub.add_parser("holdout", help="pick ~5%% of the linked game-side units once by hash into llm-data/heldout.json "
                       "(never trained on; dataset leaves them out by unit and asm shape)")
    h.add_argument("--out", default="", help="where (default llm-data/heldout.json)")
    h.add_argument("--force", action="store_true", help="replace an existing held-out set")
    r = sub.add_parser("run", help="run a model over the benchmark (each function from its frozen source), or "
                       "--set heldout: each held-out exact function blanked and written from its asm")
    _model_args(r)
    r.add_argument("--set", choices=("bench", "heldout", "heldout-dev", "heldout-final", "repair", "repair-dev",
                                      "repair-final", "review", "review-dev", "review-final"), default="bench",
                   help="which set (default bench; repair: llm-data/repair-eval.json, wrong held-out starts to fix; "
                   "-dev: the functions choices are made on (P1's 47), -final: only the go/no-go gate's; `split` freezes them; "
                   "review: the overseer eval of llm-data/review-eval.json, precision and recall per verdict; review-dev leaves out "
                   "heldout-final's functions, so choices made on it never see the gate's)")
    r.add_argument("--no-flags", action="store_true", help="with --set review: withhold the judge's flags from the prompt")
    r.add_argument("--bench", default="", help="set file (default llm-data/bench.json or heldout.json)")
    r.add_argument("--max-size", type=lambda x: int(x, 0), default=0, help="held-out functions up to this size only")
    r.add_argument("--until", default="", help="HH:MM: start no new function after this time")
    r.add_argument("--resume", action="store_true", help="append to --run's log, skipping functions it finished")
    r.add_argument("--only", default="", help="kinds to run: register,shape,unwritten")
    r.add_argument("--every", type=int, default=1, help="every K-th function only (a spread subset)")
    r.add_argument("--limit", type=int, default=0, help="first N functions only")
    from . import gate
    gate.configure_freeze(sub)
    t = sub.add_parser("t63-freeze", help="T63 D1: freeze llm-data/eval/{h-claude-20,h-giant-40,repair-dev,repair-final}.json "
                       "(barred from llm-grind, queues and datasets via gate.set_files)")
    t.add_argument("--council", default="", help="grind-log dir (default: [platform.gate] logs beside the data dir)")
    t.add_argument("--census", default="", help="regions census.json (default <repo>/build/dk/regions/census.json, else main checkout's)")
    t.add_argument("--seed", type=int, default=63)
    t.add_argument("--force", action="store_true")
    sp = sub.add_parser("split", help="freeze llm-data/heldout-split.json once: heldout-dev (<= 0x200, every 3rd: the "
                        "47 that chose the prompt) and heldout-final (the rest <= 0x300 no choice has seen)")
    sp.add_argument("--choice-logs", nargs="*", default=[], help=f"globs in llm-data of runs that made a choice "
                    f"(default {' '.join(CHOICE_LOGS)})")
    sp.add_argument("--force", action="store_true", help="make it again (don't, once a LoRA is being judged)")
    g = sub.add_parser("gate", help="the LoRA go/no-go: candidate vs baseline logs on heldout-final and repair-final, PASS/FAIL")
    g.add_argument("--baseline", nargs="+", required=True, help="the untrained model's heldout-final (or full held-out) logs")
    g.add_argument("--candidate", nargs="+", required=True, help="the LoRA's logs, same seeds in the same order")
    g.add_argument("--repair-baseline", nargs="+", default=[], help="the untrained model's repair-final (or repair) logs")
    g.add_argument("--repair-candidate", nargs="+", default=[], help="the LoRA's repair logs, same seeds")
    g.add_argument("--ratio", type=float, default=1.5, help="exact matches needed vs the baseline (default 1.5x)")
    g.add_argument("--sign-p", type=float, default=1.0, help="the sign test's p must be at most this too "
                   "(default 1: favoring is enough)")
    rep = sub.add_parser("report", help="one row per benchmark log")
    rep.add_argument("logs", nargs="*", help="logs (default every llm-data/bench-*.jsonl)")
    rep.add_argument("--paired", action="store_true", help="each log against the first on the functions all of "
                     "them finished: exact matches only one side has (sign test), best-score differences")
    rep.add_argument("--json", default="", help="also write the table as JSON")
    rep.add_argument("--showdown", default="", metavar="DIR", help="score a contestant race instead: every "
                     "DIR/<contestant>/<function>.cpp through the judge, per contestant and tier (certain exact, needs "
                     "judgment, device, miss with its blocker class from the log), into DIR/scoreboard.md and .json")
    d = sub.add_parser("dataset", help="training pairs (asm + declarations -> exact source) of every exact "
                       "function, benchmark held out, into llm-data/dataset/ (outside the repo)")
    d.add_argument("--bench", default="", help="benchmark to hold out (default llm-data/bench.json)")
    d.add_argument("--keep-bench-units", action="store_true", help="hold out only the benchmark functions, not their units")
    d.add_argument("--min-size", type=lambda x: int(x, 0), default=0x10, help="skip functions smaller than this (default 0x10)")
    d.add_argument("--only", default="", help="units whose name contains one of these (comma-separated)")
    d.add_argument("--out", default="", help="directory (default llm-data/dataset)")


BENCH = Command("llm-bench", "matching", "freeze a benchmark of near misses and unwritten functions, run a local "
                "model over it (llm-try's loop), report compile/device/exact rates and speed", run_bench, _configure_bench)


# ---- training pairs ----------------------------------------------------------------------------------

CHARS_PER_TOKEN = 2.2               # measured on llm-bench prompts (asm-heavy text, Qwen tokenizer)


def pair(ctx: Context, answer: str) -> dict:
    """One chat-style training record: the llm-try prompt for an unwritten function (no draft)
    -> the exact definition. NeuronPoke's dataset shape: {"messages": [...], "tags": {...}}."""
    msgs = build_messages(Context(ctx.name, ctx.asm, ctx.decls, "", ""), "", 0.0, [], Ingredients(m2c="never"), True)
    return {"messages": msgs + [{"role": "assistant", "content": f"```cpp\n{answer.strip()}\n```"}]}


def export(cfg: Config, args: argparse.Namespace) -> int:
    from .matching import load_objdiff_units, score, MatchError
    from .jobs import resolve
    bench = load_bench(cfg, args.bench) if (Path(args.bench) if args.bench else data_dir(cfg) / "bench.json").is_file() \
        else {"functions": []}
    held_fns = {r["function"] for r in bench["functions"]}
    held_units = {r["unit"] for r in bench["functions"]} if not args.keep_bench_units else set()
    never_units, never_fns, shapes = never_train(cfg, args.bench or None)
    heldout_units = never_units - {r["unit"] for r in bench["functions"]}
    held_units |= heldout_units                       # the held-out set always stays out, by unit and asm shape
    held_fns |= never_fns
    entries = asm_entries(cfg)
    dropped_shape = []
    units = [u for u in load_objdiff_units(cfg) if u.get("base_path") and u["name"] not in held_units]
    if args.only:
        units = [u for u in units if any(x in u["name"] for x in args.only.split(","))]
    out_dir = Path(args.out) if args.out else data_dir(cfg) / "dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    def one(u: dict) -> list[dict]:
        base = cfg.root / u["base_path"]
        if not base.is_file():
            return []
        try:
            s = score(cfg, u["name"], cfg.root / u["target_path"], base)
        except MatchError:
            return []
        recs = []
        for f in s.functions:
            if f.percent != 100.0 or f.name in held_fns or f.name.startswith(("@", "__sinit")) or f.size < args.min_size:
                continue
            try:
                _u, _src_path, _src, span, _loc, ctx = prepare(cfg, u["name"], f.name, m2c=False)
            except Exception:
                continue
            if not ctx.current:                      # inline from a header, a weak copy: no own definition
                continue
            if f.name in entries and asm_shape(entries[f.name].text) in shapes:
                dropped_shape.append(f.name)          # the same code as a benchmark or held-out function
                continue
            rec = pair(ctx, ctx.current)
            chars = sum(len(m["content"]) for m in rec["messages"])
            rec["tags"] = {"unit": u["name"], "function": f.name, "size": f.size, "linked": bool(u["metadata"].get("complete")),
                           "approx_tokens": round(chars / CHARS_PER_TOKEN), "answer_tokens": round(len(ctx.current) / CHARS_PER_TOKEN)}
            recs.append(rec)
        return recs

    with ThreadPoolExecutor(max_workers=resolve(0, cfg)) as pool:
        recs = [r for rs in pool.map(one, units) for r in rs]
    recs.sort(key=lambda r: (r["tags"]["unit"], r["tags"]["function"]))
    (out_dir / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    toks = sorted(r["tags"]["approx_tokens"] for r in recs)
    ans = sorted(r["tags"]["answer_tokens"] for r in recs)

    def q(xs, f):
        return xs[min(len(xs) - 1, int(f * len(xs)))] if xs else 0
    card = {"name": out_dir.name, "kind": "extra", "about": "dk llm-bench dataset: target asm + declarations -> exact source",
            "never_commit": True, "held_out": sorted(held_fns), "held_out_units": sorted(held_units),
            "heldout_units": len(heldout_units), "dropped_same_asm_shape": sorted(dropped_shape),
            "pairs": len(recs), "tokens": {"p50": q(toks, .5), "p90": q(toks, .9), "max": q(toks, 1)},
            "answer_tokens": {"p50": q(ans, .5), "p90": q(ans, .9), "max": q(ans, 1)},
            "fit": {str(w): sum(1 for t in toks if t <= w) for w in (512, 1024, 2048, 4096, 8192, 16384)}}
    (out_dir / "dataset.json").write_text(json.dumps(card, indent=1) + "\n")
    print(f"{len(recs)} pairs from {len(units)} units into {cfg.rel(out_dir)}/data.jsonl (benchmark held out: "
          f"{len(held_fns)} functions, {len(held_units)} units; {len(dropped_shape)} more with a held-out asm shape)")
    print(f"  prompt+answer tokens (~{CHARS_PER_TOKEN} chars each): p50 {card['tokens']['p50']}, p90 {card['tokens']['p90']}, "
          f"max {card['tokens']['max']}; answers p50 {card['answer_tokens']['p50']}, p90 {card['answer_tokens']['p90']}")
    print("  pairs that fit a window of N tokens: " + ", ".join(f"{k}: {v}" for k, v in card["fit"].items()))
    return OK
