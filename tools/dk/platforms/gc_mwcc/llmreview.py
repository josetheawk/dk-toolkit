"""The overseer task (round 2 of the project's LLM plan): one LoRA that writes definitions (the worker, prompt
v6/v7) and reviews candidates (this module, review prompt r1), told apart by the system prompt.
  ./dk dataset review [--export]               structured review records -> records/review.jsonl and
                                               llm-data/review-eval.json (held out by unit, never trained)
  ./dk llm-bench run --set review --model M    precision and recall per verdict against the labels
  ./dk llm-grind --overseer M                  ask the overseer about each candidate (logged; the judge decides)
A review prompt holds the target's asm as a short summary, the declarations the diff uses, the
candidate's unified diff against the current source, its score (exact? before -> after) and the
judge's mechanical flags. The answer is three lines: `VERDICT: clean|needs-judgment|reject`,
`RANK: 0-100`, `WHY: one line`. Labels come only from trustworthy sources: the faithfulness-audit
triage (people), the coordinator's approve/reject labels, the judge's rules (the hard gate), exactness,
and devices injected by construction. Nothing here overrides the rule-based judge.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
import sys
import time
from pathlib import Path

from ...core.command import OK, USAGE, hint
from ...core.config import Config

REVIEW_PROMPT = "r1"                 # bump when SYSTEM, the user message layout or the answer format change
VERDICTS = ("clean", "needs-judgment", "reject")
REVIEW_SALT = "review-v1"            # default; [platform.llm] review_salt: set once, never change (another eval set)
REVIEW_SHARE = 10                    # 1 in 10 train-pool units held out for the overseer eval
EVAL_FILE = "review-eval.json"       # in llm-data/
RECORDS = "review.jsonl"             # in llm-data/dataset/records/
TRIAGE_LABELS = "review-labels-triage.jsonl"   # in llm-data/: compiler-tested labels on audit-flagged exact
                                     # functions (labels.jsonl style + unit, function, rule, verdict, why, evidence)
NO_FLAGS_SHARE = 0.25                # rendered with the judge's flags withheld: the overseer must see devices itself
MAX_CLEAN_SHARE = 0.5                # clean labels' share of the rendered training rows at most
DECL_BUDGET = 1500                   # characters of declarations in a review prompt
DIFF_LINES = 80                      # diff lines in a review prompt

SYSTEM = """You review candidate definitions for a matching decompilation: C/C++ that Metrowerks
CodeWarrior for GameCube compiles to exactly the target's PowerPC instructions. A candidate must be
code the original programmers wrote. Devices that only steer the compiler never land: #pragma outside
the allowed file-level cases, asm, volatile without evidence, the register keyword, dummy or unused
locals, padding or dummy members and functions, field access as pointer plus offset, casts to the
operand's own type that change the code, locals that only copy or alias another, comma operators,
invented switch cases, and edits outside the target's definition other than declarations.
Answer with exactly three lines and nothing else:
VERDICT: clean|needs-judgment|reject
RANK: 0-100
WHY: one short line
clean: faithful, land it. needs-judgment: may be faithful, a person must check (volatile with
evidence, declaration or header edits, a known exception). reject: a device or another unfaithful
edit, never land. RANK is how much the candidate is worth landing: exact and faithful 100, a faithful
miss by its score and gain, needs-judgment at most 60, a reject at most 10."""

_INSN = re.compile(r"^\s*[0-9A-F]+  (\S+)\s*(.*)$")
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@")
_VERDICT = re.compile(r"^\W*VERDICT\W*:?\s*\W*(clean|needs[\s_-]*judg(?:e)?ment|reject)\b", re.I | re.M)
_RANK = re.compile(r"^\W*RANK\W*:?\s*(\d{1,3})", re.I | re.M)
_WHY = re.compile(r"^\W*WHY\W*:?\s*(.+)$", re.I | re.M)


# ---- the format ---------------------------------------------------------------------------------------

def asm_summary(asm: str) -> str:
    """The target's asm in one line: instructions, bytes, branches, loads, stores, float ops, calls."""
    ops, calls = [], []
    for line in asm.splitlines():
        m = _INSN.match(line)
        if not m:
            continue
        op = m[1]
        ops.append(op)
        if op == "bl":
            name = m[2].strip()
            if name not in calls:
                calls.append(name)
    n = len(ops)
    branch = sum(1 for o in ops if o.startswith("b") and o not in ("bl", "blr", "blrl", "bctrl"))
    loads = sum(1 for o in ops if o.startswith(("l", "psq_l")) and o not in ("li", "lis"))
    stores = sum(1 for o in ops if o.startswith(("st", "psq_st")))
    floats = sum(1 for o in ops if o.startswith(("f", "ps_")))
    virt = sum(1 for o in ops if o == "bctrl")
    shown = ", ".join(calls[:6]) + (f" (+{len(calls) - 6})" if len(calls) > 6 else "")
    return (f"{n} instructions (0x{4 * n:X} bytes): {branch} branches, {loads} loads, {stores} stores, "
            f"{floats} float ops" + (f", {virt} indirect calls" if virt else "") + f"; calls: {shown or 'none'}")


def diff_text(before_def: str, after_def: str) -> str:
    """The unified diff of a definition (before: "" for an unwritten function), as review prompts show it."""
    return "\n".join(difflib.unified_diff(before_def.splitlines(), after_def.splitlines(), "a", "b", n=2, lineterm=""))


def shown_diff(diff: str, limit: int = DIFF_LINES) -> str:
    """A unified diff without its file headers and hunk line numbers (a file's and a definition's look
    the same), at most `limit` lines."""
    lines = [("@@" if _HUNK.match(x) else x) for x in diff.splitlines() if not x.startswith(("--- ", "+++ "))]
    if len(lines) > limit:
        lines = lines[:limit] + [f"... {len(lines) - limit} more lines"]
    return "\n".join(lines)


def changed_code(diff: str) -> str:
    return "\n".join(x[1:] for x in diff.splitlines() if x[:1] in "+-" and not x.startswith(("+++", "---")))


def cut_decls(decls: str, code: str, budget: int = DECL_BUDGET) -> str:
    """The declarations the changed lines use: each top-level piece naming one of their identifiers, a big
    class cut to its head and those lines, all within `budget` characters."""
    from .llmtry import _chunks, identifiers
    wanted = identifiers(code)
    out, size = [], 0
    for chunk in _chunks(decls):
        s = chunk.strip()
        if not s or s.startswith(("#include", "#if", "#else", "#endif", "/* (the function")):
            continue
        if not identifiers(chunk) & wanted:
            continue
        code_part = "\n".join(x for x in chunk.splitlines() if not x.lstrip().startswith("//"))
        head = code_part.split("{", 1)[0]
        m = re.search(r"([A-Za-z_]\w*)\s*\([^;]*$", head)
        if "{" in code_part and m and not re.match(r"\s*(?:typedef\s+)?(?:class|struct|union|enum|namespace)\b", head) \
                and m[1] not in wanted:
            continue                                     # another function's body: only when the diff names it
        if len(chunk) > 600:
            lines = chunk.splitlines()
            opener = next((i for i, x in enumerate(lines) if "{" in x), 0)
            keep = lines[:opener + 1] + [x for x in lines[opener + 1:-1] if identifiers(x) & wanted] + [lines[-1]]
            chunk = "\n".join(keep if len(keep) < len(lines) else lines)
        if size + len(chunk) > budget:
            out.append("/* ... (cut) */")
            break
        out.append(chunk.rstrip())
        size += len(chunk) + 1
    return "\n".join(out)


def score_line(item: dict) -> str:
    after, before = item.get("after", 0.0), item.get("before", 0.0)
    if not item.get("compiled", True):
        return "does not compile"
    a = "exact (100%)" if item.get("exact") else f"not exact ({min(after, 99.99):.2f}%)"
    if item.get("before_def") or item.get("diff") and before:
        return f"{a}; the current source scores {min(before, 100.0):.2f}%"
    return f"{a}; the function was unwritten"


def build_review_messages(item: dict, show_flags: bool = True) -> list[dict]:
    """The review prompt for one candidate: {function, unit, path, asm, decls, before_def/after_def or diff,
    exact, before, after, flags}."""
    diff = item.get("diff") or diff_text(item.get("before_def", ""), item.get("after_def", ""))
    flags = item.get("flags") or []
    lang = "C" if str(item.get("path", "")).endswith(".c") else "C++"
    user = [f"Review this candidate for `{item['function']}` ({item.get('unit', '')}, {lang}).",
            f"Target: {asm_summary(item.get('asm', ''))}",
            f"Score: {score_line(item)}",
            "Judge flags: " + (("; ".join(flags) if flags else "none") if show_flags else "(not shown)")]
    decls = cut_decls(item.get("decls", ""), changed_code(diff))
    if decls:
        user += ["Declarations the diff uses:", "```cpp", decls, "```"]
    user += [f"Diff of {item.get('path') or 'the definition'}:", "```diff", shown_diff(diff), "```"]
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(user)}]


def format_verdict(verdict: str, rank: int, why: str) -> str:
    why = re.sub(r"\s+", " ", why).strip()
    return f"VERDICT: {verdict}\nRANK: {int(rank)}\nWHY: {why[:160]}"


def parse_verdict(text: str) -> dict | None:
    """{verdict, rank, why} from an overseer's reply (a thinking block or fence around it is fine), else None."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = _VERDICT.search(text)
    if not m:
        return None
    v = m[1].lower()
    verdict = "needs-judgment" if v.startswith("needs") else v
    r = _RANK.search(text)
    w = _WHY.search(text)
    return {"verdict": verdict, "rank": max(0, min(100, int(r[1]))) if r else None,
            "why": w[1].strip().strip("`") if w else ""}


def label_rank(verdict: str, after: float, exact: bool, before: float = 0.0) -> int:
    """The rank a label gets: clean exact 100; a clean miss 0.9 x its score (at most 89, at most 20 when it
    scores no higher than the current source); needs-judgment 0.6 x; a reject at most 10."""
    pct = 100.0 if exact else max(0.0, min(after, 99.99))
    if verdict == "reject":
        return min(10, round(pct / 10))
    if verdict == "needs-judgment":
        return min(60, round(0.6 * pct))
    if exact:
        return 100
    rank = min(89, round(0.9 * pct))
    return min(rank, 20) if before and pct <= before else rank


def answer(rec: dict) -> str:
    return format_verdict(rec["verdict"], rec["rank"], rec["why"])


def flags_shown(rec: dict, share: float = NO_FLAGS_SHARE) -> bool:
    """A hashed `share` of the records is rendered with the judge's flags withheld."""
    h = int(hashlib.sha1(f"flags\0{rec['unit']}\0{rec['function']}\0{rec.get('rid', '')}".encode()).hexdigest()[:8], 16)
    return h % 1000 >= share * 1000


def render(rec: dict, show_flags: bool | None = None) -> dict:
    show = flags_shown(rec) if show_flags is None else show_flags
    msgs = build_review_messages(rec, show)
    msgs.append({"role": "assistant", "content": answer(rec)})
    return {"messages": msgs}


# ---- labels -------------------------------------------------------------------------------------------

def _salt() -> str:
    from .llmtry import llm_option
    return llm_option("review_salt", REVIEW_SALT)


def review_split(unit: str) -> bool:
    """True for a train-pool unit held out for the overseer eval (1 in REVIEW_SHARE by salted hash)."""
    return int(hashlib.sha1((_salt() + unit).encode()).hexdigest()[:8], 16) % REVIEW_SHARE == 0


def triage_reasons(doc: str) -> dict[tuple[str, str], tuple[str, str]]:
    """(unit, function) -> (verdict, reason) from faithfulness-audit.md's triage tables."""
    out = {}
    for row in doc.splitlines():
        if not row.lstrip().startswith("|"):
            continue
        cells = [c.strip().strip("`") for c in row.strip().strip("|").split("|")]
        if len(cells) < 5 or cells[0] == "Unit" or set(cells[0]) <= set("-: "):
            continue
        verdict = next((v for v in ("device", "allowed", "false positive", "open")
                        if cells[3].lower().strip("* ").startswith(v)), None)
        if verdict:
            out[(cells[0], cells[1])] = (verdict, re.sub(r"`", "", cells[4]))
    return out


def audit_label(row: dict, reasons: dict) -> tuple[str, str] | None:
    """(verdict, why) for one judge-audit row, or None when no person decided it: a device -> reject;
    every rule allowed or a false positive (or the released raw-cast rule) -> clean; an open row or a
    known exception (faithfulness.md) -> needs-judgment; any undecided rule -> no label."""
    from .judgeaudit import KNOWN, RELEASED, rule_of
    triage = row.get("triage") or {}
    rules = [rule_of(f)[0] for f in row["flags"]]
    verdict, reason = reasons.get((row["unit"], row["function"]), ("", ""))
    if "device" in triage.values():
        return "reject", "device: " + (reason or ", ".join(r for r, v in triage.items() if v == "device"))
    undecided = [r for r in rules if r not in triage and r not in KNOWN and r not in RELEASED]
    if undecided:
        return None
    if "open" in triage.values():
        return "needs-judgment", f"{', '.join(r for r, v in triage.items() if v == 'open')}: open in the faithfulness audit, a person must decide"
    known = [r for r in rules if r in KNOWN]
    if known:
        return "needs-judgment", f"{known[0].split(' (')[0]} (faithfulness.md): keep only with its evidence"
    if triage:
        allowed = ", ".join(triage)
        return "clean", f"exact; {allowed} is allowed here" + (f": {reason}" if reason else "")
    return None                                          # only the released rule: nobody looked


STRICTNESS = {"clean": 0, "needs-judgment": 1, "reject": 2}


def triage_file_labels(path: Path) -> dict[tuple[str, str], dict]:
    """(unit, function) -> the newest row of a triage label file (TRIAGE_LABELS) with a known verdict."""
    out: dict[tuple[str, str], dict] = {}
    for r in sorted(_jsonl(path) if path.is_file() else [], key=lambda r: r.get("time") or 0):
        if r.get("verdict") in VERDICTS and r.get("unit") and r.get("function"):
            out[(r["unit"], r["function"])] = r
    return out


def merge_label(audit: tuple[str, str] | None, triage: dict | None,
                coordinator: tuple[str, str] | None) -> tuple[str, str, str, str] | None:
    """(verdict, why, source, label_source) for one accepted function, keyed (unit, function): a coordinator
    label overrides everything (also a triage-file row whose label_source is "coordinator"); otherwise the
    strictest of the faithfulness-audit triage and the triage file wins (the audit's on a tie)."""
    if triage and triage.get("label_source") == "coordinator":
        coordinator = coordinator or (triage["verdict"], triage.get("why") or "")
    if coordinator:
        return coordinator[0], coordinator[1], "triage" if triage and not audit else "audit", "coordinator"
    got = [(audit[0], audit[1], "audit", "triage")] if audit else []
    if triage:
        got.append((triage["verdict"], triage.get("why") or f"{triage.get('rule', '')}: {triage['verdict']}", "triage", "triage"))
    if not got:
        return None
    return max(got, key=lambda g: STRICTNESS[g[0]])        # max keeps the first (the audit's) on a tie


def coordinator_by_function(ddir: Path, labels: dict[str, dict]) -> dict[tuple[str, str], tuple[str, str]]:
    """(unit, function) -> the coordinator's (verdict, why) from labelled grind candidates (newest label)."""
    out, when = {}, {}
    for p in sorted((ddir / "candidates").glob("*/*.json")):
        try:
            c = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        lab = labels.get(c.get("id"))
        if lab is None or "unit" not in c or "function" not in c:
            continue
        k = (c["unit"], c["function"])
        if (lab.get("time") or 0) >= when.get(k, -1):
            when[k] = lab.get("time") or 0
            out[k] = ("clean", "approved by the coordinator") if lab["label"] == "approved" else \
                ("reject", "rejected by the coordinator: " + (lab.get("why") or "a device"))
    return out


def judged_label(category: str, flags: list[str], devices: list[str], exact: bool, after: float) -> tuple[str, str]:
    """(verdict, why) from the judge (the hard gate): its devices and rejects, its needs-judgment flags."""
    if devices:
        return "reject", f"device: {devices[0]}"
    if category == "reject":
        return "reject", f"device: {flags[0] if flags else 'the judge rejects it'}"
    if category == "needs-judgment":
        return "needs-judgment", f"check: {flags[0] if flags else 'an odd shape'}"
    if exact:
        return "clean", "exact and faithful: only the definition changed, no flags"
    return "clean", f"faithful but not exact ({min(after, 99.99):.2f}%)"


def record(src: str, label_source: str, f: dict, before_def: str, after_def: str, verdict: str, why: str,
           before: float, after: float, exact: bool, flags: list[str], prov: dict, diff: str = "",
           compiled: bool = True) -> dict:
    rec = {"v": 1, "task": "review", "review_prompt": REVIEW_PROMPT, "source": src, "label_source": label_source,
           "unit": f["unit"], "function": f["function"], "size": f.get("size", 0), "asm_shape": f.get("asm_shape"),
           "path": f.get("path", ""), "asm": f.get("asm", ""), "decls": f.get("decls", ""),
           "before_def": before_def, "after_def": after_def, "before": round(before, 3), "after": round(after, 3),
           "exact": exact, "compiled": compiled, "flags": flags, "verdict": verdict,
           "rank": label_rank(verdict, after, exact, before), "why": why, "provenance": prov}
    if diff:
        rec["diff"] = diff
    rec["rid"] = hashlib.sha1("\0".join([rec["unit"], rec["function"], src, before_def, after_def, diff]).encode()).hexdigest()[:12]
    return rec


# ---- synthetic devices (labels by construction) --------------------------------------------------------

_LOCAL = re.compile(r"^(?P<ind>[ \t]+)(?P<type>(?:unsigned |signed |const )?(?:int|short|char|long|float|double|"
                    r"u8|u16|u32|s8|s16|s32|f32|f64|bool)\s*\**)\s*(?P<name>[A-Za-z_]\w*)\s*(?P<rest>[=;])", re.M)


_STMT = re.compile(r"^(?P<ind>[ \t]+)(?P<expr>[A-Za-z_*(][^;{}\n]*?[^;{}\s])\s*;[ \t]*$", re.M)
_NOT_EXPR = re.compile(r"^(?:(?:return|if|for|while|do|switch|case|default|break|continue|goto|else|delete|new)\b|"
                       r"(?:const\s+|unsigned\s+|signed\s+|struct\s+|class\s+)*[A-Za-z_][\w:<>]*[\s*&]+[A-Za-z_]\w*"
                       r"\s*(?:=|;|\[|\(|$))")
PRAGMAS = (("dont_inline on", "dont_inline reset"), ("optimization_level 0", "optimization_level reset"),
           ("peephole off", "peephole reset"), ("scheduling off", "scheduling reset"))
KINDS = ("volatile", "dummy", "pragma", "register", "comma", "alias")
SYNTH_WHY = {"volatile": "device: volatile added to a local without evidence",
             "dummy": "device: an unused dummy local",
             "pragma": "device: a function-level pragma around the definition",
             "register": "device: the register keyword on a local",
             "comma": "device: a comma operator joining two statements",
             "alias": "device: a local that only aliases a parameter the original uses directly"}


def _params(code: str, body: int) -> list[tuple[str, str]]:
    """(type, name) of a definition's simple parameters."""
    head = code[:body]
    lo, hi = head.find("("), head.rfind(")")
    if lo < 0 or hi < lo:
        return []
    out = []
    for part in head[lo + 1:hi].split(","):
        m = re.fullmatch(r"\s*((?:const\s+|unsigned\s+|signed\s+|struct\s+)*[A-Za-z_][\w:]*\s*\**)\s*([A-Za-z_]\w*)\s*", part)
        if m and m[2] not in ("void",):
            out.append((m[1].strip(), m[2]))
    return out


def inject(code: str, kind: str, h: int = 0) -> str | None:
    """One device written into an exact definition (a label by construction): `volatile` or `register` on its
    first simple local, an unused dummy local, a function-level pragma around it, two statements joined by a
    comma, or a parameter copied into a local used in its place. None when it doesn't apply."""
    body = code.find("{")
    if body < 0:
        return None
    if kind in ("volatile", "register"):
        m = _LOCAL.search(code, body)
        if not m or kind in code:
            return None
        return code[:m.start("type")] + kind + " " + code[m.start("type"):]
    if kind == "dummy":
        nl = code.find("\n", body)
        if nl < 0 or re.search(r"\bdummy\b", code):
            return None
        return code[:nl + 1] + "    int dummy;\n" + code[nl + 1:]
    if kind == "pragma":
        if "#pragma" in code:
            return None
        on, off = PRAGMAS[h % len(PRAGMAS)]
        return f"#pragma {on}\n" + code.rstrip() + f"\n#pragma {off}\n"
    if kind == "comma":
        stmts = [m for m in _STMT.finditer(code, body) if not _NOT_EXPR.match(m["expr"])]
        for a, b in zip(stmts, stmts[1:]):
            if a["ind"] == b["ind"] and code[a.end():b.start()].strip() == "" and "," not in a["expr"] + b["expr"]:
                return code[:a.start()] + f"{a['ind']}{a['expr']}, {b['expr']};" + code[b.end():]
        return None
    if kind == "alias":
        inside = code[body + 1:]
        for t, name in _params(code, body):
            if re.search(rf"\b{name}\b", inside) and not re.search(rf"\b{name}2\b", code) \
                    and not re.search(rf"&\s*{name}\b|\b{name}\s*(?:\+\+|--|[-+*/|&^]?=(?!=))", inside):
                renamed = re.sub(rf"(?<![\w.>]){name}\b", name + "2", inside)
                return code[:body + 1] + f"\n    {t} {name}2 = {name};" + renamed
        return None
    return None


def synthetic_scores(cfg: Config, jobs: list[tuple[str, str, str, str]], workers: int, say=print) -> dict:
    """{(unit, function, code sha1): percent or None} for injected-device variants, each compiled privately in
    its unit (cached in llm-data/dataset/cache/review-synth.jsonl: the true score goes in the prompt)."""
    from concurrent.futures import ThreadPoolExecutor
    from . import llmtry
    from .matching import resolve_unit
    from .trycmd import locate_target, variant_text
    cache_path = llmtry.data_dir(cfg) / "dataset" / "cache" / "review-synth.jsonl"
    cache = {}
    for r in _jsonl(cache_path):
        cache[(r["unit"], r["function"], r["sha1"])] = r["percent"]
    todo = [j for j in jobs if (j[0], j[1], hashlib.sha1(j[3].encode()).hexdigest()) not in cache]
    say(f"  synthetic devices: {len(jobs)} variants, {len(todo)} to compile ({workers} at once)")
    units: dict[str, dict] = {}

    def one(job):
        unit, fn, path, code = job
        try:
            u = units.setdefault(unit, resolve_unit(cfg, unit))
            source = cfg.root / path
            src = source.read_text()
            span, locate, _ = locate_target(cfg, src, source, fn)
            sc = llmtry.Scorer(cfg, u, fn, src, span, locate)
            try:
                s, _, _, _ = sc.compile_score(variant_text(src, fn, code, span, locate), "synth")
            finally:
                import shutil
                shutil.rmtree(sc.work, ignore_errors=True)
            pct = None if s is None else {f.name: f.percent for f in s.functions}.get(fn)
        except Exception:
            pct = None
        return unit, fn, hashlib.sha1(code.encode()).hexdigest(), pct
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool, cache_path.open("a") as fh:
        for unit, fn, sha, pct in pool.map(one, todo):
            cache[(unit, fn, sha)] = pct
            fh.write(json.dumps({"unit": unit, "function": fn, "sha1": sha, "percent": pct}) + "\n")
    return cache


# ---- dk dataset review --------------------------------------------------------------------------------

def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.is_file() else []


def code_of(row: dict) -> str | None:
    """The definition an attempt row holds (llm-try logs it as `variant`)."""
    return row.get("variant") or None


class Facts:
    """A function's asm, declarations, source path, size and shape: from its structured worker record, else
    from llm-try's prepare (no m2c, the definition blanked so the declarations never show it)."""

    def __init__(self, cfg: Config, by_fn: dict):
        self.cfg, self.by_fn, self.memo, self.paths = cfg, by_fn, {}, {}

    def path(self, unit: str) -> str:
        if unit not in self.paths:
            from .matching import resolve_unit
            try:
                self.paths[unit] = resolve_unit(self.cfg, unit)["metadata"]["source_path"]
            except Exception:
                self.paths[unit] = ""
        return self.paths[unit]

    def __call__(self, unit: str, fn: str) -> dict | None:
        k = (unit, fn)
        if k not in self.memo:
            r = self.by_fn.get(k)
            if r is not None:
                self.memo[k] = {"unit": unit, "function": fn, "asm": r["asm"], "decls": r["decls"], "size": r.get("size", 0),
                                "asm_shape": r.get("asm_shape"), "path": self.path(unit)}
            else:
                self.memo[k] = self._prepare(unit, fn)
        return self.memo[k]

    def _prepare(self, unit: str, fn: str) -> dict | None:
        from . import llmtry
        try:
            u, source, src, span, locate, ctx = llmtry.prepare(self.cfg, unit, fn, m2c=False, blank=True, version=5)
            entry = llmtry.asm_entries(self.cfg)[fn]
        except Exception:
            return None
        n = sum(1 for x in ctx.asm.splitlines() if _INSN.match(x))
        return {"unit": u["name"], "function": fn, "asm": ctx.asm, "decls": ctx.decls, "size": 4 * n,
                "asm_shape": llmtry.asm_shape(entry.text), "path": self.cfg.rel(source)}


def current_definition(cfg: Config, path: str, fn: str) -> str | None:
    from .trycmd import definition
    try:
        src = (cfg.root / path).read_text(errors="replace")
    except OSError:
        return None
    span = definition(src, fn)
    return src[span[0]:span[1]] if span else None


def latest_labels(ddir: Path) -> dict[str, dict]:
    """Candidate id -> its last coordinator label (approved / rejected); a failed landing is no verdict."""
    out = {}
    for r in _jsonl(ddir / "candidates" / "labels.jsonl"):
        if r.get("label") in ("approved", "rejected"):
            out[r["id"]] = r
    return out


def collect(cfg: Config, args, say=print) -> dict:
    """Every review record, each with its split (train or eval); written by `dataset review`."""
    from . import llmtry
    from .llmdata import SOURCES, AnswerJudge, norm, out_dir
    from .judge import review
    ddir = llmtry.data_dir(cfg)
    never_units, never_fns, never_shapes = llmtry.never_train(cfg)
    held = json.loads((ddir / "heldout.json").read_text()) if (ddir / "heldout.json").is_file() else {"functions": []}
    held_units = {r["unit"] for r in held["functions"]}
    bench_units = never_units - held_units
    try:
        split_doc = llmtry.load_split(cfg)
    except (OSError, ValueError):
        split_doc = {}
    final = set(split_doc.get("final", []))
    rdir = out_dir(cfg, args) / "records"
    structured = {s: _jsonl(rdir / f"{s}.jsonl") for s in SOURCES if s not in ("rename", "backtrans")}
    by_fn = {}
    for s in ("base", "history", "regs", "synth", "attempts", "grind"):
        for r in structured.get(s, []):
            by_fn.setdefault((r["unit"], r["function"]), {**r, "path": ""})
    facts = Facts(cfg, by_fn)
    aj = AnswerJudge(cfg)
    audit_path = ddir / "judge-audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.is_file() else {"flagged": [], "quarantine": []}
    flagged = {(r["unit"], r["function"]) for r in audit["flagged"]}
    doc = cfg.docs.project / "faithfulness-audit.md"
    reasons = triage_reasons(doc.read_text()) if doc.is_file() else {}
    recs: list[dict] = []
    stats: dict[str, int] = {}
    synth: list[tuple[dict, str, str]] = []

    def judge(unit, fn, code):
        stub = aj.stub(unit)
        return review(stub, stub + code.strip() + "\n", fn, aj.forbidden)

    def add(rec):
        if rec["unit"] in bench_units or (rec["unit"] in never_units and rec["source"] != "heldout-attempt"
                                          and rec["source"] != "heldout-exact"):
            stats["dropped: benchmark or held-out unit"] = stats.get("dropped: benchmark or held-out unit", 0) + 1
            return
        recs.append(rec)

    # 1. the faithfulness-audit triage of accepted code (people) and the triage label file (compiler-tested),
    #    as if written into an empty definition; keyed (unit, function): the strictest wins, the coordinator's
    #    own label on the function overrides both
    tri = triage_file_labels(ddir / TRIAGE_LABELS)
    coord = coordinator_by_function(ddir, latest_labels(ddir))
    for row in audit["flagged"]:
        if row.get("vendored") or args.only and args.only not in row["unit"]:
            continue
        k = (row["unit"], row["function"])
        lab = merge_label(audit_label(row, reasons), tri.get(k), coord.get(k))
        if lab is None:
            continue
        f = facts(row["unit"], row["function"])
        code = current_definition(cfg, row["source"], row["function"])
        if f is None or code is None:
            stats["audit: no facts"] = stats.get("audit: no facts", 0) + 1
            continue
        prov = {"audit": audit.get("head", "")[:12]}
        if k in tri:
            prov.update(triage={x: tri[k].get(x) for x in ("rule", "evidence", "label_source", "judge_category")})
            stats[f"triage file: {lab[0]} ({lab[2]})"] = stats.get(f"triage file: {lab[0]} ({lab[2]})", 0) + 1
        add(record(lab[2], lab[3], {**f, "path": row["source"]}, "", code, lab[0], lab[1], 0.0, 100.0, True,
                   list(row["flags"]), prov))
    unseen = set(tri) - {(r["unit"], r["function"]) for r in audit["flagged"]}
    if unseen:
        stats["triage file: not in the judge audit (skipped)"] = len(unseen)

    # 2. structured worker records: exact answers (clean) and their wrong versions (judged)
    for (unit, fn), base in sorted(by_fn.items()):
        if args.only and args.only not in unit:
            continue
        if fn in set(audit["quarantine"]) or (unit, fn) in flagged:
            continue                                        # not judge-clean: its label comes from the audit or nobody
        wrongs, seen = [], set()
        for s in ("attempts", "grind", "history", "synth", "regs"):
            for r in structured.get(s, []):
                if (r["unit"], r["function"]) == (unit, fn) and r.get("wrong") and norm(r["wrong"]) not in seen:
                    seen.add(norm(r["wrong"]))
                    wrongs.append(r)
        f = facts(unit, fn)
        exact = base["exact"]
        h = int(hashlib.sha1(f"exact\0{unit}\0{fn}".encode()).hexdigest()[:8], 16)
        if wrongs and h % 2:                                # a repair: the wrong version is the current source
            w = wrongs[h % len(wrongs)]
            add(record("exact", "exact", f, w["wrong"], exact, "clean", "exact and faithful: only the definition "
                       "changed, no flags", w.get("wrong_percent", 0.0), 100.0, True, [], {"from": w["source"]}))
        else:
            add(record("exact", "exact", f, "", exact, "clean", "exact and faithful: only the definition changed, "
                       "no flags", 0.0, 100.0, True, [], {"from": base["source"]}))
        for w in wrongs[:args.per_fn]:
            cat, flags = judge(unit, fn, w["wrong"])
            verdict, why = judged_label(cat, flags, w.get("devices", []), False, w.get("wrong_percent", 0.0))
            add(record("worse", "judge", f, "", w["wrong"], verdict, why, 0.0, w.get("wrong_percent", 0.0), False,
                       flags, {"from": w["source"]}))
        if args.synthetic and h % 100 < args.synthetic and f.get("path"):
            for i in range(len(KINDS)):                     # the first kind that applies, from a hashed start
                kind = KINDS[(h // 100 + i) % len(KINDS)]
                bad = inject(exact, kind, h)
                if bad is not None:
                    synth.append((f, kind, bad))
                    break

    if synth:                                               # compiled: the prompt shows the true score
        got = synthetic_scores(cfg, [(f["unit"], f["function"], f["path"], bad) for f, _, bad in synth],
                               args.jobs or 2, say)
        for f, kind, bad in synth:
            pct = got.get((f["unit"], f["function"], hashlib.sha1(bad.encode()).hexdigest()))
            if pct is None:
                stats["synthetic: does not compile"] = stats.get("synthetic: does not compile", 0) + 1
                continue
            cat, flags = judge(f["unit"], f["function"], bad)
            add(record("synthetic", "construction", f, "", bad, "reject", SYNTH_WHY[kind], 0.0, pct, pct == 100.0,
                       flags, {"device": kind}))

    # 3. grind candidates with the coordinator's labels (llm-review --approve/--reject, llm-land)
    labels = latest_labels(ddir)
    for p in sorted((ddir / "candidates").glob("*/*.json")):
        c = json.loads(p.read_text())
        lab = labels.get(c.get("id"))
        if lab is None:
            continue
        f = facts(c["unit"], c["function"])
        if f is None:
            continue
        from .llmgrind import review_candidate
        cat, flags = review_candidate(c)
        exact = c.get("kind") != "near"
        if lab["label"] == "approved":
            verdict = "clean"
            why = "approved: exact and faithful" if exact else f"approved as WIP progress ({c['before']:.2f}% -> {c['after']:.2f}%)"
        else:
            verdict, why = "reject", "rejected in review: " + (lab.get("why") or (flags[0] if flags else "a device"))
        add(record("candidate", "coordinator", {**f, "path": c["source"]}, "", "", verdict, why, c.get("before", 0.0),
                   c.get("after", 100.0), exact, flags, {"id": c["id"], "run": c.get("run")}, diff=c["diff"]))

    # 4. the model's own attempts: grind logs (train pool) and held-out runs (eval only), judged
    logs = sorted(ddir.glob("grind-*.jsonl"))
    held_logs = [p for p in sorted(ddir.glob("*.jsonl")) if re.search(r"held|heldout", p.name)]
    for paths, eval_only in ((logs, False), (held_logs, True)):
        per: dict[tuple, list] = {}
        for path in paths:
            for line in path.read_text().splitlines():
                if '"attempt"' not in line:
                    continue
                r = json.loads(line)
                if r.get("kind") != "attempt" or not r.get("compiled") or not code_of(r):
                    continue
                k = (r.get("unit", ""), r["function"])
                if args.only and args.only not in k[0]:
                    continue
                if eval_only != (k[0] in held_units):
                    continue
                per.setdefault(k, []).append({**r, "log": path.name})
        for (unit, fn), rows in sorted(per.items()):
            if not eval_only and (unit, fn) in by_fn:
                continue                                    # exact since the grind: today's source isn't its start
            f = facts(unit, fn)
            if f is None:
                continue
            cur = "" if eval_only else (current_definition(cfg, f["path"], fn) or "") if f.get("path") else ""
            rows.sort(key=lambda r: (not r.get("devices"), not r.get("exact"), -r.get("percent", 0.0)))
            seen, kept = set(), 0
            for r in rows:
                code = code_of(r)
                n = norm(code)
                if n in seen or kept >= args.per_fn + 1:
                    continue
                seen.add(n)
                kept += 1
                exact = bool(r.get("exact")) and not r.get("judge")
                before = 0.0 if eval_only else r.get("percent", 0.0) - (r.get("delta") or 0.0)
                cat, flags = judge(unit, fn, code)
                verdict, why = judged_label(cat, flags, list(r.get("devices") or []) + list(r.get("judge") or []),
                                            exact, r.get("percent", 0.0))
                add(record("heldout-attempt" if eval_only else "attempt", "judge", f, "" if eval_only else cur, code,
                           verdict, why, before, r.get("percent", 0.0), exact, flags,
                           {"log": r["log"], "model": r.get("model"), "round": r.get("round"), "k": r.get("k")}))

    # 5. the held-out originals (eval only): accepted, judge-clean exact code
    for h in held["functions"]:
        if args.only and args.only not in h["unit"] or (h["unit"], h["function"]) in flagged:
            continue
        f = facts(h["unit"], h["function"])
        code = current_definition(cfg, h["source"], h["function"])
        if f is None or code is None:
            continue
        add(record("heldout-exact", "exact", {**f, "path": h["source"]}, "", code, "clean",
                   "exact and faithful: only the definition changed, no flags", 0.0, 100.0, True, [], {}))

    # the split: held-out units and 1 in REVIEW_SHARE train-pool units are eval; train leaves out their shapes too
    for r in recs:
        r["split"] = "eval" if r["unit"] in held_units or review_split(r["unit"]) else "train"
        if r["split"] == "eval":
            r["set"] = ("heldout-final" if r["function"] in final else "heldout") if r["unit"] in held_units else "review-units"
    eval_shapes = {r["asm_shape"] for r in recs if r["split"] == "eval" and r.get("asm_shape")}
    train = []
    for r in recs:
        if r["split"] != "train":
            continue
        if r["function"] in never_fns or r.get("asm_shape") in never_shapes | eval_shapes:
            stats["dropped: never-train or eval asm shape"] = stats.get("dropped: never-train or eval asm shape", 0) + 1
            continue
        train.append(r)
    evals = [r for r in recs if r["split"] == "eval"]
    return {"train": train, "eval": evals, "stats": stats}


def eval_never(cfg: Config) -> tuple[set[str], set[str], set[str]]:
    """(units, functions, asm shapes) of the overseer eval, for `dataset export --never-review-eval` (round 2)."""
    from . import llmtry
    path = llmtry.data_dir(cfg) / EVAL_FILE
    if not path.is_file():
        print(f"error: no {path} (./dk dataset review first)", file=sys.stderr)
        raise SystemExit(2)            # USAGE: a missing input, not a mismatch
    doc = json.loads(path.read_text())
    return (set(doc["units"]), {r["function"] for r in doc["functions"]},
            {r["asm_shape"] for r in doc["functions"] if r.get("asm_shape")})


def counts(recs: list[dict]) -> dict:
    out: dict[str, dict[str, int]] = {}
    for r in recs:
        row = out.setdefault(f"{r['source']} ({r['label_source']})", {v: 0 for v in VERDICTS})
        row[r["verdict"]] += 1
    out["all"] = {v: sum(1 for r in recs if r["verdict"] == v) for v in VERDICTS}
    return out


def table(title: str, c: dict) -> list[str]:
    lines = [title, f"  {'source (labels)':<34} {'clean':>7} {'needs':>7} {'reject':>7}"]
    for k, row in c.items():
        lines.append(f"  {k:<34} {row['clean']:>7} {row['needs-judgment']:>7} {row['reject']:>7}")
    return lines


def run_dataset(cfg: Config, args) -> int:
    from . import llmtry
    from .llmdata import out_dir
    ddir = llmtry.data_dir(cfg)
    rdir = out_dir(cfg, args) / "records"
    if args.export:
        return export(cfg, args)
    got = collect(cfg, args)
    rdir.mkdir(parents=True, exist_ok=True)
    got["train"].sort(key=lambda r: (r["unit"], r["function"], r["source"], r["rid"]))
    tmp = rdir / (RECORDS + ".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in got["train"]))
    tmp.replace(rdir / RECORDS)
    head = llmtry._git(cfg, "rev-parse", "HEAD").strip()
    units = sorted({r["unit"] for r in got["eval"]})      # before the cut: every eval unit stays out of training
    got["eval"], cut = balance(got["eval"], args.eval_clean_share, EVAL_FIRST)
    doc = {"about": "Overseer eval (LLM plan, Round 2): review records of the held-out units and of 1 in "
                    f"{REVIEW_SHARE} train-pool units by salted hash ({_salt()}); never trained. Game-derived code: "
                    "never commit.", "never_commit": True, "review_prompt": REVIEW_PROMPT, "head": head,
           "made": time.strftime("%Y-%m-%d %H:%M"), "units": units,
           "counts": counts(got["eval"]), "clean_cut": cut, "functions": got["eval"]}
    (ddir / EVAL_FILE).write_text(json.dumps(doc, indent=1) + "\n")
    for line in table(f"train: {len(got['train'])} records -> {cfg.rel(rdir / RECORDS)}", counts(got["train"])) + \
            table(f"eval: {len(got['eval'])} records in {len(doc['units'])} units -> {ddir / EVAL_FILE}", counts(got["eval"])):
        print(line)
    for k, v in got["stats"].items():
        print(f"  {k}: {v}")
    hint("dataset review --export")
    return OK


TRAIN_FIRST = ("audit", "triage", "candidate", "attempt")    # clean records kept first when the clean share is cut
EVAL_FIRST = ("audit", "candidate", "heldout-attempt", "attempt", "heldout-exact")   # never "triage" (or another
# new source) ahead of these: the eval's clean cut must not push heldout-final items out of review-final


def balance(recs: list[dict], share: float, first=TRAIN_FIRST) -> tuple[list[dict], int]:
    """Clean records cut to at most `share` of the rows: people's labels and real model attempts kept first
    (in `first`'s order), the rest by hash, one per function before a second."""
    clean = [r for r in recs if r["verdict"] == "clean"]
    other = len(recs) - len(clean)
    room = int(share * other / (1 - share)) if share < 1 else len(clean)
    if len(clean) <= room:
        return recs, 0
    seen: dict[tuple, int] = {}

    def order(r):
        k = (r["unit"], r["function"])
        seen[k] = seen.get(k, 0) + 1
        return (first.index(r["source"]) if r["source"] in first else len(first), seen[k],
                hashlib.sha1(("bal" + r["rid"]).encode()).hexdigest())
    ranked = sorted(sorted(clean, key=lambda r: hashlib.sha1(("bal" + r["rid"]).encode()).hexdigest()), key=order)
    kept = set(id(r) for r in ranked[:room])
    out = [r for r in recs if r["verdict"] != "clean" or id(r) in kept]
    return out, len(recs) - len(out)


def train_row(r: dict, no_flags_share: float | None, cpt: float) -> dict:
    """One review record as a training row: the prompt `ask` sends (build_review_messages) plus the
    three-line answer, tagged task=review with sample weight 1."""
    row = render(r, None if no_flags_share is None else flags_shown(r, no_flags_share))
    n = sum(len(m["content"]) for m in row["messages"]) / cpt
    row["tags"] = {"task": "review", "review_prompt": REVIEW_PROMPT, "unit": r["unit"], "function": r["function"],
                   "asm_shape": r.get("asm_shape"), "size": r.get("size", 0), "source": r["source"],
                   "label_source": r["label_source"], "verdict": r["verdict"], "rank": r["rank"],
                   "approx_tokens": round(n), "answer_tokens": round(len(row["messages"][-1]["content"]) / cpt),
                   "sample_weight": 1.0, "flags_shown": "Judge flags: (not shown)" not in row["messages"][1]["content"],
                   "order": hashlib.sha1(("ord" + r["rid"]).encode()).hexdigest()}
    return row


# ---- the joint worker + overseer export (LoRA 1, 2026-11-02) -------------------------------------------
REVIEW_TOKEN_SHARE = 0.05            # review rows' share of the joint export's trained tokens at most
JOINT_FIRST = TRAIN_FIRST            # kept first when the share cap binds: people's labels, real attempts


def review_cap(rows: list[dict], worker_tokens: float, share: float, first=JOINT_FIRST) -> tuple[list[dict], int]:
    """Review rows (weight 1) cut so their tokens stay at or under `share` of the joint trained tokens
    (worker_tokens + theirs): R <= share x W / (1 - share). People's labels and real attempts are kept
    first, then a fixed hash order (a uniform sample, so the verdict mix stays)."""
    if share >= 1:
        return rows, 0
    budget = share * worker_tokens / (1 - share) if share > 0 else 0
    ranked = sorted(rows, key=lambda x: (first.index(x["tags"]["source"]) if x["tags"]["source"] in first else len(first),
                                         hashlib.sha1(("cap" + x["tags"]["order"]).encode()).hexdigest()))
    kept, used = set(), 0
    for x in ranked:
        n = x["tags"]["approx_tokens"]
        if used + n > budget:
            continue
        used += n
        kept.add(id(x))
    out = [x for x in rows if id(x) in kept]
    return out, len(rows) - len(out)


def joint_review_rows(recs: list[dict], ev: dict, never: tuple[set, set, set], worker_tokens: float, share: float,
                      cpt: float, max_tokens: int, max_clean_share: float = MAX_CLEAN_SHARE,
                      no_flags_share: float | None = None) -> tuple[list[dict], dict]:
    """records/review.jsonl's training records -> rendered review rows for the joint export, and their card.
    Only split=train records (review-dev and review-final are the eval: never); a record of a benchmark,
    held-out or review-eval unit, function or asm shape is an error (stale records: re-run `dataset review`)."""
    never_units, never_fns, never_shapes = never
    eval_units = set(ev.get("units", ()))
    eval_fns = {r["function"] for r in ev.get("functions", ())}
    eval_shapes = {r.get("asm_shape") for r in ev.get("functions", ()) if r.get("asm_shape")}
    train = [r for r in recs if r.get("split", "train") == "train"]
    bad = [r for r in train if r["unit"] in never_units | eval_units or r["function"] in never_fns | eval_fns
           or r.get("asm_shape") in never_shapes | eval_shapes]
    if bad:
        raise ValueError(f"{len(bad)} review record(s) leak (a benchmark, held-out or review-eval unit, function or "
                         "asm shape): re-run ./dk dataset review")
    train, clean_cut = balance(train, max_clean_share)
    rows = [train_row(r, no_flags_share, cpt) for r in train]
    long = [x for x in rows if x["tags"]["approx_tokens"] > max_tokens]
    rows = [x for x in rows if x["tags"]["approx_tokens"] <= max_tokens]
    offered = sum(x["tags"]["approx_tokens"] for x in rows)
    rows, capped = review_cap(rows, worker_tokens, share)
    rows.sort(key=lambda x: x["tags"]["order"])
    tokens = sum(x["tags"]["approx_tokens"] for x in rows)
    card = {"review_prompt": REVIEW_PROMPT, "rows": len(rows), "tokens": tokens,
            "answer_tokens": sum(x["tags"]["answer_tokens"] for x in rows), "sample_weight": 1.0,
            "share_cap": share, "share": round(tokens / max(1, worker_tokens + tokens), 4),
            "dropped": {"eval split": sum(1 for r in recs if r.get("split", "train") != "train"),
                        "clean share cut": clean_cut, "too long": len(long), "share cap": capped},
            "tokens_offered": offered, "max_clean_share": max_clean_share,
            "flags_hidden": sum(1 for x in rows if not x["tags"]["flags_shown"]),
            "verdicts": {v: sum(1 for x in rows if x["tags"]["verdict"] == v) for v in VERDICTS},
            "by_source": {}, "eval_left_out": {"units": len(eval_units), "functions": len(eval_fns),
                                               "asm_shapes": len(eval_shapes)}}
    for x in rows:
        k = f"{x['tags']['source']} ({x['tags']['label_source']})"
        card["by_source"].setdefault(k, {v: 0 for v in VERDICTS})[x["tags"]["verdict"]] += 1
    return rows, card


def worker_overlap(worker_rows: list[dict], ev: dict) -> dict:
    """What the worker rows share with the overseer eval. review-final (heldout-final's functions) must share
    nothing; review-dev's 1-in-10 train-pool units ('review-units') are in the worker's training on purpose
    (the joint LoRA keeps every worker row): a known, accepted optimism in review-dev."""
    w_units = {r["tags"].get("unit") for r in worker_rows}
    w_fns = {r["tags"].get("function") for r in worker_rows}
    w_shapes = {r["tags"].get("asm_shape") for r in worker_rows if r["tags"].get("asm_shape")}
    out = {}
    for name, keep in (("review-final", lambda s: s == "heldout-final"), ("review-dev", lambda s: s != "heldout-final")):
        items = [r for r in ev.get("functions", ()) if keep(r.get("set"))]
        units = {r["unit"] for r in items}
        shared = units & w_units
        out[name] = {"items": len(items), "units": len(units), "units_in_worker_training": len(shared),
                     "items_in_those_units": sum(1 for r in items if r["unit"] in shared),
                     "functions_in_worker_training": sum(1 for r in items if r["function"] in w_fns),
                     "asm_shapes_in_worker_training": sum(1 for r in items if r.get("asm_shape") in w_shapes),
                     "worker_rows_in_those_units": sum(1 for r in worker_rows if r["tags"].get("unit") in shared)}
    f = out["review-final"]
    out["review_final_clean"] = not (f["units_in_worker_training"] or f["functions_in_worker_training"]
                                     or f["asm_shapes_in_worker_training"])
    return out


def join_export(cfg: Config, root: Path, never: tuple[set, set, set], worker_rows: list[dict], card: dict,
                share: float, max_tokens: int) -> tuple[list[dict], dict]:
    """`dataset export --with-review`: the worker rows (all of them, unchanged) plus the review rows, and
    the card's totals over both tasks (tasks.worker / tasks.review; the generated share stays the worker's).
    Raises ValueError when the records are missing or leak, or when review-final meets the worker rows."""
    from . import llmtry
    from .llmdata import read_jsonl
    recs = read_jsonl(root / "records" / RECORDS)
    path = llmtry.data_dir(cfg) / EVAL_FILE
    if not recs or not path.is_file():
        raise ValueError(f"no {root / 'records' / RECORDS} or {path} (./dk dataset review first)")
    ev = json.loads(path.read_text())
    w = card["effective_tokens_per_epoch"]
    rows, rcard = joint_review_rows(recs, ev, never, w, share, llmtry.CHARS_PER_TOKEN, max_tokens)
    rcard["overlap"] = worker_overlap(worker_rows, ev)
    if not rcard["overlap"]["review_final_clean"]:
        raise ValueError(f"review-final shares units, functions or asm shapes with the worker rows: "
                         f"{rcard['overlap']['review-final']}")
    worker = {"records": card["records"], "rows": card["rows_written"], "tokens": card["tokens_per_epoch"],
              "effective_tokens": w, "answer_tokens": sum(s["answer_tokens"] for s in card["sources"].values())}
    review = {"records": len(rows), "rows": len(rows), "tokens": rcard["tokens"], "effective_tokens": rcard["tokens"],
              "answer_tokens": rcard["answer_tokens"]}
    total = w + rcard["tokens"]
    for t in (worker, review):
        t["effective_share"] = round(t["effective_tokens"] / max(1, total), 4)
    by_cap = card.get("effective_tokens_by_cap", {})
    card.update({"tasks": {"worker": worker, "review": review}, "review": rcard, "review_prompt": REVIEW_PROMPT,
                 "worker_effective_tokens_per_epoch": w, "effective_tokens_per_epoch": total,
                 "effective_tokens_by_cap": {c: v + sum(x["tags"]["approx_tokens"] for x in rows
                                                        if x["tags"]["approx_tokens"] <= int(c)) for c, v in by_cap.items()},
                 "records": card["records"] + len(rows), "rows_written": card["rows_written"] + len(rows),
                 "tokens_per_epoch": card["tokens_per_epoch"] + rcard["tokens"],
                 "generated_effective_share_of": "the worker's trained tokens"})
    return worker_rows + rows, card


def print_joint(card: dict, say=print) -> None:
    w, r, rc = card["tasks"]["worker"], card["tasks"]["review"], card["review"]
    say(f"  joint (worker + overseer, review prompt {rc['review_prompt']}): trained tokens ~{card['effective_tokens_per_epoch']:,}"
        f" = worker ~{w['effective_tokens']:,} ({w['effective_share']:.1%}) + review ~{r['effective_tokens']:,} "
        f"({r['effective_share']:.1%}, at most {rc['share_cap']:.0%}); answer tokens worker ~{w['answer_tokens']:,}, "
        f"review ~{r['answer_tokens']:,}")
    say(f"  review rows {rc['rows']} (verdicts {rc['verdicts']}; {rc['flags_hidden']} hide the judge's flags); "
        f"dropped {rc['dropped']}")
    o = rc["overlap"]
    say(f"  review-final vs worker rows: {'no shared unit, function or asm shape' if o['review_final_clean'] else 'SHARED'}; "
        f"review-dev: {o['review-dev']['units_in_worker_training']} of {o['review-dev']['units']} units "
        f"({o['review-dev']['items_in_those_units']} of {o['review-dev']['items']} items, "
        f"{o['review-dev']['functions_in_worker_training']} functions) are in the worker's training (known optimism)")


def joint_recheck(rows: list[dict], ev_path: Path) -> int:
    """Written review rows that meet the overseer eval (unit, function or asm shape): must be 0."""
    if not ev_path.is_file():
        return 0
    ev = json.loads(ev_path.read_text())
    units, fns = set(ev.get("units", ())), {r["function"] for r in ev.get("functions", ())}
    shapes = {r.get("asm_shape") for r in ev.get("functions", ()) if r.get("asm_shape")}
    return sum(1 for r in rows if (r.get("tags") or {}).get("task") == "review"
               and (r["tags"]["unit"] in units or r["tags"]["function"] in fns or r["tags"].get("asm_shape") in shapes))


def export(cfg: Config, args) -> int:
    """records/review.jsonl -> dataset/<name>/train.jsonl (chat rows with tags like the worker's, sample_weight 1)
    and eval.jsonl (rendered for reference), plus a card. Separate from the worker export, which is frozen."""
    from . import llmtry
    from .llmdata import out_dir, read_jsonl
    base = out_dir(cfg, args)
    recs = read_jsonl(base / "records" / RECORDS)
    if not recs:
        print(f"error: no {base / 'records' / RECORDS} (./dk dataset review first)")
        return USAGE
    never_units, never_fns, never_shapes = llmtry.never_train(cfg)
    ev = json.loads((llmtry.data_dir(cfg) / EVAL_FILE).read_text())
    eval_units = set(ev["units"])
    eval_shapes = {r.get("asm_shape") for r in ev["functions"]}
    leaks = [r for r in recs if r["unit"] in never_units | eval_units or r["function"] in never_fns
             or r.get("asm_shape") in never_shapes | eval_shapes]
    if leaks:
        print(f"error: {len(leaks)} record(s) leak (benchmark, held-out or review-eval unit, function or shape)")
        return USAGE
    recs, cut = balance(recs, args.max_clean_share)
    out = base / (args.name or "review")
    out.mkdir(parents=True, exist_ok=True)
    rows = [train_row(r, args.no_flags_share, llmtry.CHARS_PER_TOKEN) for r in recs]
    tokens = sum(x["tags"]["approx_tokens"] for x in rows)
    rows.sort(key=lambda x: x["tags"]["order"])
    (out / "train.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    (out / "eval.jsonl").write_text("".join(json.dumps({**render(r, True), "tags": {"rid": r["rid"], "set": r.get("set")}}) + "\n"
                                            for r in ev["functions"]))
    card = {"about": "Overseer (review) rows for round 2, prompt " + REVIEW_PROMPT + "; train together with a worker "
                     "export. Game-derived: never commit.", "never_commit": True, "made": time.strftime("%Y-%m-%d %H:%M"),
            "rows": len(rows), "clean_cut": cut, "max_clean_share": args.max_clean_share,
            "tokens": round(tokens), "counts": counts(recs), "eval_rows": len(ev["functions"]),
            "leakage_check": "passed: no benchmark, held-out or review-eval unit, function or asm shape"}
    (out / "dataset.json").write_text(json.dumps(card, indent=1) + "\n")
    for line in table(f"{len(rows)} rows (~{tokens / 1e6:.2f}M tokens; {cut} clean ones cut to "
                      f"{args.max_clean_share:.0%}) -> {out}", counts(recs)):
        print(line)
    return OK


def configure(sub) -> None:
    s = sub.add_parser("review", help="overseer records (round 2): the audit triage, exact answers and their judged "
                       "wrong versions, labelled candidates, the model's judged attempts -> records/review.jsonl and "
                       "llm-data/review-eval.json (held-out units + 1 in 10 units, never trained); --export renders them")
    s.add_argument("--export", action="store_true", help="render records/review.jsonl into dataset/<name>/ (train.jsonl, "
                   "eval.jsonl, dataset.json)")
    s.add_argument("--name", default="", help="with --export: dataset/NAME (default review)")
    s.add_argument("--per-fn", type=int, default=2, help="wrong versions or attempts per function (default 2)")
    s.add_argument("--synthetic", type=int, default=25, help="percent of exact functions that also get one injected "
                   "device (volatile, register, dummy local, pragma, comma, parameter alias), compiled for its true "
                   "score, as a reject (default 25; 0: none)")
    s.add_argument("--max-clean-share", type=float, default=MAX_CLEAN_SHARE, help="with --export: clean rows' share at "
                   f"most (default {MAX_CLEAN_SHARE}; people's labels are never cut)")
    s.add_argument("--no-flags-share", type=float, default=None, help="with --export: share rendered with the judge's "
                   f"flags withheld (default {NO_FLAGS_SHARE})")
    s.add_argument("--eval-clean-share", type=float, default=MAX_CLEAN_SHARE, help="clean records' share of the eval "
                   f"set at most (default {MAX_CLEAN_SHARE}: real attempts and people's labels kept first)")
    s.add_argument("--only", default="", help="units whose name contains this")
    s.add_argument("--jobs", type=int, default=2, help="parallel compiles of the synthetic devices (default 2)")
    s.add_argument("--out", default="", help="dataset directory (default llm-data/dataset)")


# ---- asking an overseer -------------------------------------------------------------------------------

def ask(client, model: str, item: dict, show_flags: bool = True, options: dict | None = None) -> dict:
    """One review from a model: {verdict, rank, why, parsed, secs, reply} (verdict None when unparseable)."""
    t0 = time.time()
    opts = {"temperature": 0.0, "num_predict": 96, "num_ctx": 16384, "seed": 1, **(options or {})}   # the worker's
    # window: another num_ctx would make Ollama reload the model between the two tasks
    try:
        reply = client.chat(model, build_review_messages(item, show_flags), opts, think=False)
        text = reply.get("message", {}).get("content", "")
    except Exception as e:                                # a dead server must not end a grind
        return {"verdict": None, "rank": None, "why": "", "parsed": False, "secs": round(time.time() - t0, 1),
                "error": str(e)[:200]}
    got = parse_verdict(text) or {"verdict": None, "rank": None, "why": ""}
    return {**got, "parsed": got["verdict"] is not None, "secs": round(time.time() - t0, 1), "reply": text[:400]}


def candidate_item(cfg: Config, cand: dict, flags: list[str]) -> dict:
    """A grind candidate as a review item: its diff and score, the function's asm and declarations."""
    f = Facts(cfg, {})(cand["unit"], cand["function"]) or {}
    return {"unit": cand["unit"], "function": cand["function"], "path": cand.get("source", ""), "asm": f.get("asm", ""),
            "decls": f.get("decls", ""), "diff": cand.get("diff") or diff_text(cand.get("before_text", ""), cand.get("text", "")),
            "before": cand.get("before", 0.0), "after": cand.get("after", 100.0), "exact": cand.get("kind") != "near",
            "before_def": "x" if cand.get("before") else "", "flags": flags}


# ---- llm-bench run --set review ----------------------------------------------------------------------

def scores(rows: list[dict]) -> dict:
    """Precision and recall per verdict, accuracy, parse rate, reject recall (the costly miss), rank error."""
    out = {"n": len(rows), "parsed": sum(1 for r in rows if r.get("got")), "per_verdict": {}}
    for v in VERDICTS:
        tp = sum(1 for r in rows if r.get("got") == v and r["label"] == v)
        pred = sum(1 for r in rows if r.get("got") == v)
        gold = sum(1 for r in rows if r["label"] == v)
        out["per_verdict"][v] = {"labels": gold, "said": pred, "precision": round(tp / pred, 3) if pred else None,
                                 "recall": round(tp / gold, 3) if gold else None}
    out["accuracy"] = round(sum(1 for r in rows if r.get("got") == r["label"]) / len(rows), 3) if rows else None
    rej = [r for r in rows if r["label"] == "reject"]
    out["reject_recall"] = round(sum(1 for r in rej if r.get("got") == "reject") / len(rej), 3) if rej else None
    out["reject_missed_as_clean"] = sum(1 for r in rej if r.get("got") == "clean")
    ranked = [r for r in rows if r.get("rank") is not None]
    out["rank_mae"] = round(sum(abs(r["rank"] - r["rank_label"]) for r in ranked) / len(ranked), 1) if ranked else None
    out["by_source"] = {}
    for src in sorted({r.get("source", "") for r in rows}):
        sub = [r for r in rows if r.get("source", "") == src]
        rj = [r for r in sub if r["label"] == "reject"]
        out["by_source"][src] = {"n": len(sub), "accuracy": round(sum(1 for r in sub if r.get("got") == r["label"]) / len(sub), 3),
                                 "rejects": len(rj), "reject_recall": round(sum(1 for r in rj if r.get("got") == "reject")
                                                                            / len(rj), 3) if rj else None}
    return out


def print_scores(s: dict, say=None) -> None:
    say = say or print
    say(f"  {s['n']} reviews, {s['parsed']} parsed; accuracy {s['accuracy']}; reject recall {s['reject_recall']} "
        f"({s['reject_missed_as_clean']} rejects called clean); rank error {s['rank_mae']}")
    for v, p in s["per_verdict"].items():
        say(f"  {v:<15} labels {p['labels']:>4}  said {p['said']:>4}  precision {p['precision']}  recall {p['recall']}")
    for src, b in s.get("by_source", {}).items():
        say(f"  by source {src:<16} {b['n']:>4} reviews, accuracy {b['accuracy']}"
            + (f", reject recall {b['reject_recall']} of {b['rejects']}" if b["rejects"] else ""))


def bench_review(cfg: Config, args, client=None) -> int:
    """`llm-bench run --set review`: every eval record (or --every/--limit of them) to the model, the reply
    parsed and scored against the label; logged to llm-data/<run>.jsonl."""
    from . import llmtry
    path = Path(args.bench) if args.bench else llmtry.data_dir(cfg) / EVAL_FILE
    if not path.is_file():
        print(f"error: no {path} (./dk dataset review first)")
        return USAGE
    doc = json.loads(path.read_text())
    items = doc["functions"]
    part = args.set.partition("-")[2]
    if part:
        final = part == "final"                           # -final: heldout-final's functions only (the gate's);
        items = [r for r in items if (r.get("set") == "heldout-final") == final]   # -dev: the rest, for choices
    if args.only:
        items = [r for r in items if r["source"] in args.only.split(",")]
    items = sorted(items, key=lambda r: hashlib.sha1(("ev" + r["rid"]).encode()).hexdigest())   # a spread, fixed
    if args.every > 1:
        items = items[::args.every]
    if args.limit:
        items = items[:args.limit]
    client = client or llmtry.client_for(args)
    show = not getattr(args, "no_flags", False)
    name = llmtry.run_name(args, "review")
    lpath, fh, log = llmtry.open_log(cfg, name)
    log({"kind": "run", "set": args.set, "model": args.model, "review_prompt": REVIEW_PROMPT, "flags_shown": show,
         "items": len(items), "eval_made": doc.get("made")})
    done = set()
    if args.resume and lpath.stat().st_size:
        done = {json.loads(x).get("rid") for x in lpath.read_text().splitlines() if '"kind": "review"' in x}
    until = llmtry._until(args.until)
    rows = []
    try:
        for i, r in enumerate(items):
            if r["rid"] in done:
                continue
            if until and time.time() >= until:
                print(f"stopping at {args.until} (--until); rerun with --resume --run {name} to go on")
                break
            got = ask(client, args.model, r, show, {"num_ctx": args.num_ctx})
            row = {"kind": "review", "rid": r["rid"], "unit": r["unit"], "function": r["function"], "source": r["source"],
                   "set": r.get("set"), "label": r["verdict"], "rank_label": r["rank"], "got": got["verdict"],
                   "rank": got["rank"], "why": got.get("why", ""), "secs": got["secs"], "error": got.get("error")}
            log(row)
            rows.append(row)
            print(f"[{i + 1}/{len(items)}] {r['function'][:40]:<40} {r['verdict']:<15} -> {got['verdict']} {got['rank']}")
    finally:
        fh.close()
    print_scores(scores(rows))
    print(f"log {cfg.rel(lpath)}")
    return OK


def report_rows(rows: list[dict]) -> dict:
    return scores([r for r in rows if r.get("kind") == "review"])


# ---- agreement on real candidates (llm-review --agreement) ---------------------------------------------

def agreement(ddir: Path, say=print) -> dict:
    """The overseer's verdicts on grind candidates against the judge's category and the coordinator's label."""
    labels = latest_labels(ddir)
    rows = []
    for p in sorted((ddir / "candidates").glob("*/*.json")):
        c = json.loads(p.read_text())
        o = c.get("overseer")
        if not o:
            continue
        lab = labels.get(c["id"], {}).get("label")
        rows.append({"id": c["id"], "said": o.get("verdict"), "judge": o.get("judge"),
                     "label": {"approved": "clean", "rejected": "reject"}.get(lab)})
    out = {"candidates": len(rows),
           "agrees_with_judge": sum(1 for r in rows if r["said"] == r["judge"]),
           "labelled": sum(1 for r in rows if r["label"]),
           "agrees_with_label": sum(1 for r in rows if r["label"] and r["said"] == r["label"]),
           "rejects_called_clean": sum(1 for r in rows if r["label"] == "reject" and r["said"] == "clean")}
    say(f"{out['candidates']} candidate(s) with an overseer verdict: {out['agrees_with_judge']} agree with the judge; "
        f"{out['labelled']} labelled, {out['agrees_with_label']} agree with the label, "
        f"{out['rejects_called_clean']} rejected one(s) called clean")
    for r in rows:
        say(f"  {r['id']}  overseer {r['said']}  judge {r['judge']}  label {r['label'] or '-'}")
    return out
