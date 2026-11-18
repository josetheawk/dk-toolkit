"""Types and names for the locals a search writes (`__typeof__(call(…)) t0 = call(…);`).

When the debug info and the probe compile give no type for such a local, a local that holds a call's result
takes the callee's declared return type (from the unit's source or a header). A new local with a placeholder
name (`t0`) that holds a call's result is named after the callee (`CARDUnmount` -> `unmount`, an address-named
function -> `value`). Both only rewrite text; the judge still decides (tool-gaps #11).
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from ...core import cbody

KEYWORDS = {"return", "else", "new", "delete", "case", "goto", "sizeof", "if", "while", "for", "switch", "do",
            "typedef", "operator", "class", "struct", "union", "enum", "default", "break", "continue"}
PROTO = re.compile(r"(?m)^[ \t]*(?:(?:extern|static|inline|virtual|const|friend)\s+|\"C\"\s+)*"
                   r"((?:unsigned\s+|signed\s+)?[A-Za-z_][\w:]*(?:\s*\*+|\s*&)?)\s+\**\s*(?:[\w:]+::)?(\w+)\s*"
                   r"\([^;{)]*\)\s*(?:const\s*)?[;{]")


def _prototypes(text: str) -> dict[str, str | None]:
    """name -> return type (None when declarations disagree)."""
    out: dict[str, str | None] = {}
    for m in PROTO.finditer(text):
        t, name = " ".join(m.group(1).split()), m.group(2)
        if t.split()[0] in KEYWORDS or name in KEYWORDS:
            continue
        out[name] = t if out.get(name, t) == t else None
    return out


@lru_cache(maxsize=4)
def _header_prototypes(root: str) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for p in sorted(Path(root, "include").rglob("*.h")):
        try:
            for k, v in _prototypes(p.read_text(errors="replace")).items():
                out[k] = v if out.get(k, v) == v else None
        except OSError:
            continue
    return out


def callee_of(expr: str) -> str | None:
    """The function an initializer calls at its top level (`atoi(s)`, `p->fn(x)`), or None (a cast, no call)."""
    expr = expr.strip()
    if not expr.endswith(")"):
        return None
    depth, last = 0, None
    for t in cbody.tokens(expr):
        if t.text == "(":
            if depth == 0:
                return last if last and last not in KEYWORDS else None
            depth += 1
        elif t.text == ")":
            depth -= 1
        elif t.kind == "id" and depth == 0:
            last = t.text
    return None


def call_return_types(root: Path, text: str, body: cbody.Stmt, names: list[str]) -> dict[str, str]:
    """For `__typeof__(call(…)) name` declarations among `names`: the callee's declared return type."""
    local, headers = _prototypes(text), _header_prototypes(str(root))
    out = {}
    for s in cbody.statements(body):
        if s.decl is None or len(s.decl.declarators) != 1:
            continue
        d = s.decl.declarators[0]
        head = text[s.decl.type_start:s.decl.type_end].strip()
        if d.name not in names or not head.startswith("__typeof__("):
            continue
        callee = callee_of(head[len("__typeof__("):-1])
        t = local.get(callee) if callee in local else headers.get(callee) if callee else None
        if t and t != "void":
            out[d.name] = t
    return out


def natural_name(callee: str, taken: set[str]) -> str:
    if re.fullmatch(r"fn_[0-9A-Fa-f]{8}|lbl_[0-9A-Fa-f]{8}|vf\d+.*", callee):
        base = "value"
    else:
        words = re.findall(r"[A-Z]+(?=[A-Z][a-z]|\b|\d)|[A-Z]?[a-z]+|\d+", callee.split("::")[-1]) or [callee]
        base = words[-1].lower() if words[-1].lower() not in KEYWORDS else "value"
        if base.isdigit():
            base = "value"
        elif base == callee.split("::")[-1].lower():   # a one-word callee: its own name would shadow it
            base = "result"
    name, k = base, 2
    while name in taken or name in KEYWORDS:
        name, k = f"{base}{k}", k + 1
    return name


def name_placeholders(text: str, body: cbody.Stmt, old_names: set[str], placeholder) -> tuple[str, dict[str, str]]:
    """Rename new placeholder locals that hold a call's result after the callee, inside the body only."""
    taken = {t.text for t in cbody.tokens(text) if t.kind == "id"}
    renames: dict[str, str] = {}
    for s in cbody.statements(body):
        if s.decl is None or len(s.decl.declarators) != 1:
            continue
        d = s.decl.declarators[0]
        if d.name in old_names or d.name in renames or not placeholder.fullmatch(d.name):
            continue                                    # one name per placeholder (several scopes may declare it)
        rest = text[s.decl.type_end:s.end]
        init = rest.split("=", 1)[1].rsplit(";", 1)[0] if "=" in rest else ""
        callee = callee_of(init) if init else None
        if callee:
            renames[d.name] = natural_name(callee, taken)
            taken.add(renames[d.name])
    if not renames:
        return text, {}
    edits = [(t.start, t.end, renames[t.text]) for t in cbody.tokens(text, body.start, body.end)
             if t.kind == "id" and t.text in renames]
    return cbody.apply_edits(text, edits), renames
