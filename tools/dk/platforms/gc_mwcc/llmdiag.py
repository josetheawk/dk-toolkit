"""One-line diagnosis before a repair answer (the LLM plan, Round 2): `DIAG: <asm symptom> → <source fix>`.

Deterministic, from the repair record only: the symptom from its diff rows (and diff class), the fix
from the actual change wrong → exact (declaration or statement order, a declaration joined with or
split from its first assignment, a type, ?: vs if/else, the loop or switch form, early returns,
operand order, a flipped comparison, a compound assignment, a local added or dropped; else the first
changed line). Never text from a model. A record's own `diag` field wins (opt-in).

Rendered as the first line of the assistant answer (`dk dataset export --diag-share F`); llm-try's
`--diag` prefills `DIAG:` on repair rounds and takes the code from the fence after the line.
"""

from __future__ import annotations

import difflib
import re

PREFIX = "DIAG:"
MAX = 200                    # characters of a diagnosis line at most
SNIP = 48                    # a code snippet in it at most
_REG = re.compile(r"\b(?:r|f)(?:[12]?\d|3[01])\b")
_TOKEN = re.compile(r"[A-Za-z_]\w*|0x[0-9A-Fa-f]+|\d+\.?\d*f?|->|<<=|>>=|[-+*/%&|^!=<>]=|&&|\|\||<<|>>|\S")
_FLIP = {"<": ">", ">": "<", "<=": ">=", ">=": "<="}


def _snip(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= SNIP else text[:SNIP - 1] + "…"


def _body(code: str) -> list[str]:
    """The definition's lines inside its outer braces, comments and blank lines out, whitespace collapsed."""
    code = re.sub(r"//[^\n]*|/\*.*?\*/", "", code, flags=re.S)
    start, end = code.find("{"), code.rfind("}")
    inner = code[start + 1:end] if 0 <= start < end else code
    return [" ".join(l.split()) for l in inner.splitlines() if l.strip()]


def _rows(rows: list[str]) -> list[tuple[str, str, str, str]]:
    """(offset, target, mark, ours) of llmtry.relative_rows' lines."""
    out = []
    for row in rows:
        if row.strip() in ("...", "") or row.lstrip().startswith("…"):
            continue
        m = re.match(r"^([ 0-9A-F]{5})  (.{36}) (.{3})  (.*)$", row)
        if m:
            out.append((m[1].strip(), m[2].strip(), m[3].strip(), m[4].strip()))
    return out


def symptom(rows: list[str], diff_class: str = "") -> str:
    """What the asm shows: swapped registers, or the first differing instruction."""
    parsed = _rows(rows)
    marked = [r for r in parsed if r[2]]
    if not marked:
        return "the instructions differ"
    pairs: dict[tuple[str, str], int] = {}
    reg_only = True
    for off, t, mark, o in marked:
        tr, orr = _REG.findall(t), _REG.findall(o)
        if mark != "arg" or len(tr) != len(orr) or _REG.sub("R", t) != _REG.sub("R", o):
            reg_only = False
            continue
        for a, b in zip(tr, orr):
            if a != b:
                pairs[(a, b)] = pairs.get((a, b), 0) + 1
    if (reg_only or diff_class == "register") and pairs:
        swaps = sorted({tuple(sorted(p, key=lambda r: (r[0], int(r[1:])))) for p in pairs if (p[1], p[0]) in pairs})
        if swaps:
            return "registers " + " and ".join(f"{a}/{b}" for a, b in swaps[:2]) + " swapped"
        (a, b), _n = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))
        return f"ours has {b} where the target has {a}"
    first = next((r for r in marked if r[2] != "arg" or _REG.sub("R", r[1]) != _REG.sub("R", r[3])), marked[0])
    off, t, mark, o = first
    where = f"at {off}" if off else "first"
    if mark == "-":
        return f"{where} the target has `{_snip(t)}`, ours lacks it"
    if mark == "+":
        return f"{where} ours has an extra `{_snip(o)}`"
    return f"{where} the target has `{_snip(t)}`, ours `{_snip(o)}`"


def _middle(a: str, b: str) -> tuple[str, str]:
    """The parts of two lines that differ (common prefix and suffix cut at token edges)."""
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    while i > 0 and (a[i - 1:i].isalnum() or a[i - 1:i] == "_"):
        i -= 1
    j = 0
    while j < min(len(a), len(b)) - i and a[len(a) - 1 - j] == b[len(b) - 1 - j]:
        j += 1
    while j > 0 and (a[len(a) - j:len(a) - j + 1].isalnum() or a[len(a) - j:len(a) - j + 1] == "_"):
        j -= 1
    ma, mb = a[i:len(a) - j].strip(), b[i:len(b) - j].strip()
    return (ma or a, mb or b) if len(ma) >= 3 and len(mb) >= 3 else (a, b)


_CONTROL = re.compile(r"^(?:\}\s*)?(?:else\s*)?(?:if\s*\(.*\)\s*)?\{?$|^\}$")


def _moved(a: list[str], b: list[str]) -> tuple[int, int] | None:
    """The first element of b out of a's order (index in b) and its neighbour to name (index in b)."""
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    keep = set()
    for blk in sm.get_matching_blocks():
        keep.update(range(blk.b, blk.b + blk.size))
    for j in range(len(b)):
        if j not in keep:
            return j, (j + 1 if j + 1 < len(b) else j - 1)
    return None


def _decls(lines: list[str]) -> dict[str, tuple[str, str | None]]:
    """name -> (type, initializer) for simple local declarations (dk regs' pattern)."""
    from .regs import DECL
    out: dict[str, tuple[str, str | None]] = {}
    for line in lines:
        m = DECL.match(line)
        if m and m["type"].strip() not in ("return", "delete", "goto", "else", "case", "throw"):
            out.setdefault(m["name"], (" ".join(m["type"].split()), m["init"]))
    return out


def _decl_name(line: str) -> str | None:
    from .regs import DECL
    m = DECL.match(line)
    return m["name"] if m and m["type"].strip() not in ("return", "delete", "goto", "else", "case", "throw") else None


def _count(lines: list[str], word: str) -> int:
    return sum(len(re.findall(rf"\b{word}\b", l)) for l in lines)


def fix(wrong: str, exact: str) -> tuple[str, str]:
    """(class, the change from wrong to exact as a short instruction)."""
    w, e = _body(wrong), _body(exact)
    if w == e:
        return "none", "the code is the same apart from layout"
    dw, de = _decls(w), _decls(e)
    if sorted(w) == sorted(e):                           # the same lines in another order
        mv = _moved(w, e)
        if mv is not None:
            j, k = mv
            a, b = _decl_name(e[j]), _decl_name(e[k])
            if a and b:
                rel = "before" if k > j else "after"
                return "decl order", f"declare `{a}` {rel} `{b}`"
            rel = "before" if k > j else "after"
            return "statement order", f"move `{_snip(e[j])}` {rel} `{_snip(e[k])}`"
    for n, (t, init) in de.items():                      # a declaration joined with its first assignment, or split
        if n in dw and dw[n][1] is None and init is not None and any(re.match(rf"^{re.escape(n)}\s*=[^=]", l) for l in w):
            return "join init", f"declare `{n}` where it is first assigned"
        if n in dw and dw[n][1] is not None and init is None and any(re.match(rf"^{re.escape(n)}\s*=[^=]", l) for l in e):
            return "split init", f"declare `{n}` first, assign it later"
    retyped = [(n, de[n][0], dw[n][0]) for n in de if n in dw and de[n][0].replace(" ", "") != dw[n][0].replace(" ", "")]
    if retyped:
        n, te, tw = retyped[0]
        return "type", f"`{n}` is `{te}`, not `{tw}`"
    if set(dw) != set(de) and len(w) != len(e):
        added, dropped = sorted(set(de) - set(dw)), sorted(set(dw) - set(de))
        if added and not dropped:
            return "add local", f"add a local `{added[0]}`" + (f" ({de[added[0]][0]})" if de[added[0]][0] else "")
        if dropped and not added:
            return "drop local", f"drop the local `{dropped[0]}`"
    body_w, body_e = [l for l in w if not _CONTROL.match(l)], [l for l in e if not _CONTROL.match(l)]
    if sorted(body_w) == sorted(body_e) and body_w != body_e and _count(w, "if") == _count(e, "if"):
        cond = next((l for l in e if _CONTROL.match(l) and "if" in l and l not in w), "")
        return "if-invert", "invert the `if` and swap its branches" + (f": `{_snip(cond)}`" if cond else "")
    chained = [l for l in e if re.match(r"^[^=;]+?\s=\s[^=;]+?\s=\s[^=]", l) and l not in w]
    if chained and len(w) > len(e):
        return "chained", f"chain the assignments: `{_snip(chained[0])}`"
    chained = [l for l in w if re.match(r"^[^=;]+?\s=\s[^=;]+?\s=\s[^=]", l) and l not in e]
    if chained and len(e) > len(w):
        return "chained", "split the chained assignment into one per line"
    qw, qe = sum(l.count("?") for l in w), sum(l.count("?") for l in e)
    iw, ie = _count(w, "if"), _count(e, "if")
    if qe > qw and ie < iw:
        return "ternary", "use `?:` instead of if/else"
    if qe < qw and ie > iw:
        return "ternary", "use if/else instead of `?:`"
    loops = {k: (_count(w, k), _count(e, k)) for k in ("for", "while", "do")}
    more = [k for k, (a, b) in loops.items() if b > a]
    less = [k for k, (a, b) in loops.items() if b < a]
    if more and less:
        name = {"do": "do/while", "while": "while", "for": "for"}
        return "loop form", f"use a `{name[more[0]]}` loop, not `{name[less[0]]}`"
    sw, se = _count(w, "switch"), _count(e, "switch")
    if se > sw:
        return "switch", "use a switch, not an if chain"
    if se < sw:
        return "switch", "use if/else, not a switch"
    rw, re_ = _count(w, "return"), _count(e, "return")
    if re_ > rw and ie >= iw:
        return "early return", "return early"
    if re_ < rw:
        return "early return", "one return, no early exit"
    sm = difflib.SequenceMatcher(None, w, e, autojunk=False)
    changes = [op for op in sm.get_opcodes() if op[0] != "equal"]
    if len(changes) == 1 and changes[0][0] == "replace" and changes[0][2] - changes[0][1] == 1 \
            and changes[0][4] - changes[0][3] == 1:
        lw, le = w[changes[0][1]], e[changes[0][3]]
        tw, te = _TOKEN.findall(lw), _TOKEN.findall(le)
        mw, me = _middle(lw, le)
        if sorted(tw) == sorted(te):
            return "operands", f"operand order: `{_snip(me)}`"
        if sorted(_FLIP.get(t, t) for t in tw) == sorted(te):
            return "comparison", f"flip the comparison: `{_snip(me)}`"
        cw = [t for t in tw if re.fullmatch(r"[-+*/%&|^]=|<<=|>>=", t)]
        ce = [t for t in te if re.fullmatch(r"[-+*/%&|^]=|<<=|>>=", t)]
        if ce and not cw:
            return "compound", f"use `{ce[0]}`: `{_snip(le)}`"
        if cw and not ce:
            return "compound", f"write `{cw[0]}` out: `{_snip(le)}`"
        if ("!" in tw) != ("!" in te):
            return "test form", f"test as `{_snip(me)}`"
        return "line", f"change `{_snip(mw)}` to `{_snip(me)}`"
    if len(changes) == 1 and changes[0][0] == "delete":
        return "drop line", f"drop `{_snip(w[changes[0][1]])}`"
    if len(changes) == 1 and changes[0][0] == "insert":
        return "add line", f"add `{_snip(e[changes[0][3]])}`"
    first = changes[0] if changes else None
    n = sum(max(c[2] - c[1], c[4] - c[3]) for c in changes)
    at = e[first[3]] if first and first[3] < len(e) else (e[-1] if e else "")
    return "rewrite", f"rewrite {n} line{'s' if n != 1 else ''} from `{_snip(at)}`"


def diagnose(rec: dict) -> str:
    """The DIAG line of a repair record ("" for a direct pair)."""
    if rec.get("diag"):
        return rec["diag"]
    if not rec.get("wrong"):
        return ""
    _cls, text = fix(rec["wrong"], rec["exact"])
    line = f"{PREFIX} {symptom(rec.get('rows') or [], rec.get('diff_class', ''))} → {text}"
    return line if len(line) <= MAX else line[:MAX - 1] + "…"


def diag_class(rec: dict) -> str:
    return fix(rec["wrong"], rec["exact"])[0] if rec.get("wrong") else ""


def split(content: str) -> tuple[str, str]:
    """(the DIAG line, the rest) of a reply that may open with one."""
    text = content.lstrip()
    if not text.startswith(PREFIX):
        return "", content
    line, _, rest = text.partition("\n")
    return line.strip(), rest
