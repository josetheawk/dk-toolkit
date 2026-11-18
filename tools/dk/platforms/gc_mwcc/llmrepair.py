"""Repairs and ranking for local-model attempts (`llm-try`, `llm-grind`; `--no-repair`, `--rank fuzzy` turn them off).

- Declaration visibility: a function written in its place in the unit (by address) cannot see what the unit
  declares further down. When the compiler names an undefined identifier that the unit's own source declares
  after the function, the repair makes it visible above the function and compiles again: a type (struct,
  class, union, enum, typedef) moves up whole; a later function gets a prototype; a later global gets an
  `extern` declaration. Definitions never move (their order is the object's order).
- Feedback: an undefined identifier that nothing in the prompt declares is an invented name; a call that
  matches no prototype gets the real prototype lines. Both go into the next round's "did not compile" note.
- Shape: a compiled attempt is shape-equal when its instructions equal the target's with register numbers
  masked (objdiff rows that differ only in registers). Shape-equal attempts rank above the rest, since
  register numbering is what the order tools fix (`regs --reach`, `close --moves oracle`).
"""

from __future__ import annotations

import re

from ...core import cbody

_UNDEFINED = re.compile(r"undefined identifier '([A-Za-z_]\w*)'")
_CALL = re.compile(r"function call '([A-Za-z_][\w:]*)\s*\(")
_REG = re.compile(r"\b(?:r|f|cr)\d+\b")


def undefined_names(error: str) -> list[str]:
    return list(dict.fromkeys(_UNDEFINED.findall(error or "")))


def top_items(src: str) -> list[tuple[int, int, str]]:
    """Top-level items of a source: (start, end, kind) with kind `type`, `function`, `variable`, `prototype`
    or `other`; preprocessor lines and comments are skipped. Items inside namespaces or extern "C" blocks are
    not listed (the repair leaves those alone)."""
    out = []
    toks = [t for t in cbody.tokens(src) if t.kind != "pp"]
    i, n = 0, len(toks)
    while i < n:
        start, depth, j, end, paren = i, 0, i, None, False
        while j < n:
            t = toks[j].text
            if t == "(" and depth == 0:
                paren = True
            if t == "{":
                depth += 1
            elif t == "}":
                depth -= 1
                if depth == 0:
                    if paren and j + 1 < n and toks[j + 1].text != ";" and toks[start].text not in ("typedef",):
                        end = j                                 # a function definition ends at its brace
                        break
                    while j < n and toks[j].text != ";":       # a type: up to its `;`
                        j += 1
                    end = min(j, n - 1)
                    break
            elif t == ";" and depth == 0:
                end = j
                break
            j += 1
        if end is None:
            break
        words = [t.text for t in toks[start:end + 1]]
        if words[0] in ("namespace", "extern") and "{" in words and (words[0] == "namespace" or '"C"' in words[1:2]):
            kind = "other"
        elif words[0] in ("typedef", "struct", "class", "union", "enum"):
            kind = "type"
        elif "{" in words and paren and words[-1] == "}":
            kind = "function"
        elif paren and "=" not in words[:words.index("(")]:
            kind = "prototype"
        else:
            kind = "variable"
        out.append((toks[start].start, toks[end].start + len(toks[end].text), kind))
        i = end + 1
    return out


def _declares(src: str, item: tuple[int, int, str], name: str) -> bool:
    text, kind, w = src[item[0]:item[1]], item[2], re.escape(name)
    if kind == "type":
        if re.search(rf"\b(?:struct|class|union|enum)\s+{w}\b", text):
            return True
        if text.startswith("typedef") and re.search(rf"\b{w}\s*(?:\[[^\]]*\]\s*)*;$", text):
            return True
        body = re.search(r"\benum\b[^{]*\{(.*)\}", text, re.S)
        return bool(body and re.search(rf"(?:^|[,{{\s]){w}\s*(?:=|,|$)", body.group(1).strip(), re.M))
    if kind in ("function", "prototype"):
        return re.search(rf"\b{w}\s*\(", text.split("{", 1)[0]) is not None
    if kind == "variable":
        return re.search(rf"\b{w}\s*(?:\[[^\]]*\]\s*)*$", text.split("=", 1)[0].rstrip(" ;")) is not None
    return False


def visibility_fix(text: str, fn_start: int, name: str) -> tuple[str, str] | None:
    """(new text, label) that makes `name`, declared in `text` after `fn_start`, visible before it; None when
    the unit does not declare it later at top level."""
    items = top_items(text)
    later = [it for it in items if it[0] >= fn_start and not (it[0] <= fn_start < it[1])]
    hit = next((it for it in later if _declares(text, it, name)), None)
    if hit is None:
        return None
    start, end, kind = hit
    chunk = text[start:end]
    at = text.rfind("\n", 0, fn_start) + 1
    if kind == "type":                          # above its first user (an earlier repair may have declared one)
        users = [it[0] for it in items if it[0] < start and it[0] <= fn_start
                 and re.search(rf"\b{re.escape(name)}\b", text[it[0]:it[1]])]
        if users:
            at = text.rfind("\n", 0, min(users)) + 1
        cut_end = end + 1 if text[end:end + 1] == "\n" else end
        new = text[:at] + chunk + "\n" + text[at:start] + text[cut_end:]
        return new, f"moved the declaration of {name} above the function"
    if kind == "prototype":
        return text[:at] + chunk + "\n" + text[at:], f"declared {name} above the function"
    if kind == "function":
        proto = text[start:text.index("{", start)].rstrip() + ";"
        return text[:at] + proto + "\n" + text[at:], f"declared {name}'s prototype above the function"
    head = chunk.split("=", 1)[0].rstrip(" ;")
    if head.startswith("static"):
        return text[:at] + head + ";\n" + text[at:], f"declared {name} above the function"
    decl = head if head.startswith("extern") else "extern " + head
    return text[:at] + decl + ";\n" + text[at:], f"declared {name} (extern) above the function"


def feedback(error: str, known: str) -> str:
    """Notes for the next round about a compile error: invented names and the real prototypes of mismatched calls."""
    notes = []
    words = set(re.findall(r"\b[A-Za-z_]\w*\b", known))
    invented = [n for n in undefined_names(error) if n not in words]
    if invented:
        notes.append(f"{', '.join('`' + n + '`' for n in invented)} is declared nowhere in the declarations: it does not "
                     "exist. Never invent a member, field or function name; use a declared one, or keep the expression "
                     "the current definition or draft uses.")
    for name in dict.fromkeys(_CALL.findall(error or "")):
        plain = name.split("::")[-1]
        protos = [ln.strip() for ln in known.splitlines() if re.search(rf"\b{re.escape(plain)}\s*\(", ln) and ln.rstrip().endswith(";")]
        if protos:
            notes.append(f"`{plain}` is declared as: " + " | ".join(dict.fromkeys(protos[:3])) + "; pass arguments of those types.")
    return " ".join(notes)


def masked(formatted: str) -> str:
    return _REG.sub("R", " ".join((formatted or "").split()))


def shape_rows(left: list[dict], right: list[dict]) -> int:
    """Rows where target and ours differ in more than register numbers (one row each, padded alike)."""
    out = 0
    for t, o in zip(left, right):
        kind = t.get("diff_kind") or o.get("diff_kind")
        if not kind:
            continue
        if kind == "DIFF_ARG_MISMATCH":
            a = t.get("instruction", {}).get("formatted", "")
            b = o.get("instruction", {}).get("formatted", "")
            if masked(a) == masked(b):
                continue
        out += 1
    return out + abs(len(left) - len(right))


def rank_key(res: dict, rank: str = "shape") -> tuple:
    """Higher is better: exact, then (with `shape`) shape-equal, then the score."""
    if not res.get("compiled"):
        return (0, 0, -1.0)
    return (1 if res.get("exact") else 0, 1 if rank == "shape" and res.get("shape_equal") else 0, res.get("percent", 0.0))
