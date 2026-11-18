"""Prompt v7's declarations: what the function could legitimately need from outside its unit's text.

v6 shows the unit's own source (trimmed to what the function, its draft and its asm use) plus the
full definitions of the classes the draft names. Answers then use members of base classes, of the
types that locals and globals hold, and helpers declared in shared headers, none of which the prompt
shows (the LLM plan, P1 v7). v7 adds, compactly and within a budget:

- the owner class and its base classes (transitively), the parameters' classes, the classes the
  draft names, the classes of the members it touches (two levels), of the globals the asm refers
  to and of what the called functions return; each with its data members, its virtual and called
  methods and its inline methods (bodies only when short), comments and other bodies cut;
- the prototypes of the called free functions and the declarations of the referenced globals;
- the classes whose methods are called, slim (methods only) when they are not otherwise needed.

Everything comes from the text the unit already includes (llmtry.visible_files) and its own source,
so the model sees nothing a compile of the unit could not. Blocks are separate ("// path" first, no
blank line inside), so `merge` can drop the ones another declarations text (a record's stored one)
already defines. The output is deterministic: the same inputs give byte-identical text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

CONTEXT_BUDGET = 6600       # chars (~3k tokens at 2.2 chars/token); 8800 closed 2 points more for +450 tokens at p90
CLASS_CAP = 3600            # one class's block at most (slimmed, then cut, beyond it)
INLINE_BODY = 240           # an inline method keeps its body up to this many chars (as declarations() does)
FULL, TOUCHED, SLIM = 0, 1, 2   # a type's data members: all, only those the code names, none (a lower mode wins)

_TYPE_HEAD = re.compile(r"^(?:typedef\s+)?(class|struct|union|enum)\s+([A-Za-z_]\w*)?\s*([^{;]*)\{", re.S)
_IDENT = re.compile(r"\b[A-Za-z_]\w*\b")
_NOT_TYPES = frozenset("""const volatile unsigned signed int char short long float double void bool static
extern inline virtual struct class union enum typedef public private protected friend operator return
mutable explicit register""".split())


def strip_comments(text: str, strings: bool = True) -> str:
    """Comments (and with `strings`, string/char literal contents) as spaces: the same length, newlines
    kept, so offsets into the fully blanked text (for parsing) also cut the other (for showing)."""
    def blank(m):
        s = m[0]
        if s.startswith(("/", )):
            return re.sub(r"[^\n]", " ", s)
        if not strings:
            return s
        return s[0] + re.sub(r"[^\n]", " ", s[1:-1]) + s[-1]
    return re.sub(r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'", blank, text, flags=re.S)


def _one_line(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def statements(text: str) -> list[tuple[int, int]]:
    """Top-level statements of comment-stripped text as (start, end): a preprocessor line, a
    declaration up to its `;`, a function definition up to its closing brace. `extern "C" {` and
    `namespace X {` blocks are transparent."""
    out, i, n, depth, start, opened = [], 0, len(text), 0, 0, []
    line_start = True
    while i < n:
        c = text[i]
        if line_start and depth == 0 and c == "#":
            j = i
            while True:
                k = text.find("\n", j)
                k = n if k < 0 else k
                if text[j:k].rstrip().endswith("\\") and k < n:
                    j = k + 1
                    continue
                break
            out.append((i, k))
            i, start, line_start = k, k, True
            continue
        if c == "\n":
            line_start = True
        elif not c.isspace():
            line_start = False
        if c == "{":
            head = text[start:i]
            if depth == 0 and re.search(r"(?:\bextern\s*\"[^\"]*\"|\bnamespace(?:\s+\w+)?)\s*$", head):
                opened.append(i)
                start = i + 1
            else:
                depth += 1
        elif c == "}":
            if depth == 0 and opened:
                opened.pop()
                start = i + 1
            elif depth > 0:
                depth -= 1
                if depth == 0:
                    head = text[start:i]
                    first = head.find("{")
                    pre = head[:first] if first >= 0 else head
                    if "(" in pre and not re.match(r"\s*(?:typedef\b|class\b|struct\b|union\b|enum\b)", pre) \
                            and "=" not in pre.split("(")[0]:
                        out.append((start, i + 1))     # a function definition: no `;` follows
                        start = i + 1
        elif c == ";" and depth == 0:
            out.append((start, i + 1))
            start = i + 1
        i += 1
    return [(a + len(text[a:b]) - len(text[a:b].lstrip()), b) for a, b in out if text[a:b].strip()]


@dataclass
class TypeDecl:
    name: str
    path: str
    kind: str                   # class, struct, union, enum, typedef
    text: str                   # comment-stripped definition (the whole statement)
    bases: list[str] = field(default_factory=list)
    refs: list[str] = field(default_factory=list)       # a one-line typedef's identifiers
    parse: str = ""             # the same with string literals blanked (offsets match text)


def _bases(head: str) -> list[str]:
    if ":" not in head:
        return []
    out = []
    for part in head.split(":", 1)[1].split(","):
        words = [w for w in _IDENT.findall(re.sub(r"<.*?>", "", part)) if w not in ("public", "private", "protected", "virtual")]
        if words:
            out.append(words[-1])
    return out


def _decl_name(stmt: str) -> str | None:
    """The declared name of a variable or function-pointer declaration."""
    s = stmt.split("=", 1)[0]
    m = re.search(r"\(\s*\*\s*([A-Za-z_]\w*)\s*\)", s)
    if m:
        return m[1]
    s = re.sub(r"\[[^\]]*\]", "", s)
    s = re.sub(r":\s*\d+\s*;?\s*$", "", s)        # a bitfield's width
    names = _IDENT.findall(s)
    return names[-1] if names else None


class Index:
    """Types, free-function prototypes and globals declared in a set of texts, first declaration wins."""

    def __init__(self):
        self.types: dict[str, TypeDecl] = {}
        self.funcs: dict[str, tuple[str, str]] = {}      # name -> (path, one-line prototype)
        self.globals: dict[str, tuple[str, str]] = {}    # name -> (path, one-line declaration)
        self.inlines: dict[str, tuple[str, str, bool]] = {}   # an inline free function (whole when short), from the unit?
        self.unit_calls: set[str] = set()                # what the unit's other code calls (its helpers)
        self.unit = False

    def add(self, path: str, text: str, skip: tuple[int, int] = (0, 0), unit: bool = False) -> None:
        s = strip_comments(text)
        self.unit = unit
        if unit:
            body = s[:skip[0]] + s[skip[1]:] if skip[1] > skip[0] else s
            self.unit_calls |= set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", body))
        raw = strip_comments(text, strings=False)
        for a, b in statements(s):
            if skip[0] <= a < max(skip[1], skip[0] + 1):
                continue
            self._statement(path, s[a:b], raw[a:b])

    def _type(self, d: TypeDecl) -> None:
        """First definition wins, but a body beats a one-line typedef (`typedef struct X X;` before `struct X {`)."""
        have = self.types.get(d.name)
        if have is None or "{" in d.text and "{" not in have.text:
            self.types[d.name] = d

    def _statement(self, path: str, st: str, raw: str = "") -> None:
        """One top-level statement: `st` parsed (string literals blanked), `raw` shown (the same offsets)."""
        raw = raw or st
        if st.startswith("#"):
            return
        prefix = ""
        if st.startswith("template"):
            m = re.match(r"template\s*<[^{};]*?>\s*", st)
            if not m or "(" not in st.split("{", 1)[0]:
                return
            prefix, st, raw = _one_line(m[0]) + " ", st[m.end():], raw[m.end():]   # a function template
        m = _TYPE_HEAD.match(st)
        if m:
            kind, name, head = m[1], m[2], m[3]
            if st.startswith("typedef"):
                tail = st[st.rfind("}") + 1:].rstrip(";")
                names = [n for n in _IDENT.findall(tail) if n not in _NOT_TYPES]
                for n in ([name] if name else []) + names:
                    self._type(TypeDecl(n, path, kind, raw, _bases(head), parse=st))
            elif name:
                end = len(st.rstrip(";").rstrip())
                self._type(TypeDecl(name, path, kind, raw[:end] + ";", _bases(head), parse=st[:end] + ";"))
            return
        if st.startswith("typedef"):
            name = _decl_name(st.rstrip(";"))
            if name:
                refs = [w for w in _IDENT.findall(st) if w not in _NOT_TYPES and w != name and w != "typedef"]
                if name not in self.types:
                    self.types[name] = TypeDecl(name, path, "typedef", _one_line(raw), refs=refs)
            return
        if re.match(r"(?:class|struct|union|enum)\s+\w+\s*;", st):
            return                                       # a forward declaration
        head = st.split("{", 1)[0]
        paren = head.find("(")
        if paren >= 0 and not re.search(r"\(\s*\*", head[:paren + 3]):
            name_m = re.search(r"([A-Za-z_~]\w*(?:::[A-Za-z_~]\w*)*)\s*$", head[:paren])
            if not name_m or "::" in name_m[1]:
                return                                   # an out-of-class member definition
            proto = prefix + _one_line(raw[:len(head)]).rstrip(";").rstrip() + ";"
            self.funcs.setdefault(name_m[1], (path, proto))
            if "{" in st and (not self.unit or re.match(r"(?:static\s+)?inline\b", head.strip())):
                whole = prefix + _one_line(raw) if len(st) <= INLINE_BODY else proto
                self.inlines.setdefault(name_m[1], (path, whole, self.unit))
            return
        if st.endswith(";") and "{" not in head:
            name = _decl_name(st.rstrip(";"))
            if name and name not in _NOT_TYPES:
                decl = raw[:len(st.split("=", 1)[0])]
                self.globals.setdefault(name, (path, _one_line(decl).rstrip(";").rstrip() + ";"))


@dataclass
class Member:
    kind: str                   # data, method, virtual, inline, label, other
    name: str
    text: str


def members(decl: TypeDecl) -> tuple[str, list[Member], str]:
    """(head up to `{`, the members, the tail after `}`) of a class/struct/union definition."""
    t, raw = decl.parse or decl.text, decl.text      # parsed with literals blanked, shown as written
    open_ = t.find("{")
    close = t.rfind("}")
    head, body, tail = raw[:open_ + 1], t[open_ + 1:close], raw[close:]
    shown = raw[open_ + 1:close]
    out: list[Member] = []
    i, depth, start, n = 0, 0, 0, len(body)
    while i < n:
        c = body[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                piece = body[start:i + 1]
                pre = piece.split("{", 1)[0]
                if "(" in pre and not re.match(r"\s*(?:struct|union|enum|class)\b", pre):
                    out.append(_member(piece, shown[start:i + 1]))
                    start = i + 1
        elif c == ";" and depth == 0:
            out.append(_member(body[start:i + 1], shown[start:i + 1]))
            start = i + 1
        elif c == ":" and depth == 0 and re.fullmatch(r"\s*(?:public|private|protected)\s*", body[start:i]) \
                and body[i + 1:i + 2] != ":":
            out.append(Member("label", "", body[start:i + 1].strip()))
            start = i + 1
        i += 1
    return head, [m for m in out if m.text], tail


def _member(piece: str, shown: str = "") -> Member:
    """One member statement: classified from `piece` (literals blanked), shown as `shown`."""
    lead = len(piece) - len(piece.lstrip())
    s, r = piece.strip(), (shown or piece)[lead:lead + len(piece.strip())]
    pre = s.split("{", 1)[0]
    if s.startswith(("friend", "using")):
        return Member("other", "", "")
    if re.match(r"(?:enum|struct|union|class|typedef)\b", s):
        return Member("data", _decl_name(s.rstrip(";")) or "", _one_line(r))
    paren = pre.find("(")
    if paren >= 0 and not re.search(r"\(\s*\*", pre[:paren + 3]):
        m = re.search(r"(~?[A-Za-z_]\w*|operator\s*\S+?)\s*$", pre[:paren])
        name = m[1].replace(" ", "") if m else ""
        if "{" in s:
            return Member("inline", name, r)
        return Member("virtual" if re.match(r"virtual\b", s) else "method", name, _one_line(r))
    return Member("data", _decl_name(s.rstrip(";")) or "", _one_line(r))


def _inline(m: Member, keep_body: bool) -> str:
    if keep_body and len(m.text) <= INLINE_BODY:
        return _one_line(m.text)
    return _one_line(m.text.split("{", 1)[0]) + ";"


def render_type(decl: TypeDecl, wanted: set[str], mode: str = FULL, cap: int = CLASS_CAP) -> str:
    """A type's definition, compact: data members (FULL: all; TOUCHED: those `wanted` names; SLIM:
    none), virtual and wanted methods, inline methods (short bodies), one per line; other methods,
    comments and long bodies cut."""
    slim = mode == SLIM
    if decl.kind == "typedef" and "{" not in decl.text:
        return decl.text
    if decl.kind == "enum" or (decl.kind == "typedef" and re.match(r"typedef\s+enum\b", decl.text)):
        return _one_line(decl.text)
    head, ms, tail = members(decl)
    head = _one_line(head)
    tail = _one_line(tail)

    def build(level: int) -> str:
        lines, pending = [], ""
        for m in ms:
            if m.kind == "label":
                pending = m.text
                continue
            if m.kind == "data":
                if slim or mode == TOUCHED and m.name not in wanted:
                    continue
                line = m.text
            elif m.kind == "inline":
                if level >= 3 and m.name not in wanted:
                    continue
                line = _inline(m, level < 1 or m.name in wanted and level < 2)
            elif m.kind == "virtual":
                if level >= 2 and m.name not in wanted:
                    continue
                line = m.text
            elif m.kind == "method":
                if m.name not in wanted and m.name.lstrip("~") not in wanted:
                    continue
                line = m.text
            else:
                continue
            if pending:
                lines.append(pending)
                pending = ""
            lines.append("    " + line)
        if slim or mode == TOUCHED and any(m.kind == "data" and m.name not in wanted for m in ms):
            lines.append("    /* … other data members not shown */" if not slim else "    /* … data members not shown */")
        return head + "\n" + "\n".join(lines) + "\n" + tail
    text = build(0)
    level = 0
    while len(text) > cap and level < 3:
        level += 1
        text = build(level)
    if len(text) > cap:                                  # still too big: cut at a member line
        cut = text.rfind("\n", 0, cap)
        text = text[:cut] + "\n    /* … (cut) */\n" + tail
    return text


def _types_in(text: str, index: Index, own: str = "") -> list[str]:
    return [w for w in dict.fromkeys(_IDENT.findall(text)) if w in index.types and w != own]


def _called(calls: list[str]) -> tuple[dict[str, set[str]], list[str]]:
    """Mangled call targets as {class: method names} and the free functions' names."""
    from .demangle import demangle
    methods: dict[str, set[str]] = {}
    free: list[str] = []
    for c in calls:
        try:
            d = demangle(c)
        except Exception:
            d = None
        if d is None:
            free.append(c)
        elif d.quals:
            cls = d.quals[-1]
            name = {"ctor": cls, "dtor": "~" + cls}.get(d.name, d.name)
            methods.setdefault(cls, set()).add(name)
        elif isinstance(d.name, str):
            free.append(d.name)
    return methods, free


def build_index(files: list[tuple[str, str]], unit_path: str, src: str, skip: tuple[int, int]) -> Index:
    """The unit's own source first (its target function skipped), then the headers it includes."""
    ix = Index()
    ix.add(unit_path, src, skip, unit=True)
    for path, text in files:
        ix.add(path, text)
    return ix


def context(index: Index, symbol: str, code: str, calls: list[str], refs: list[str], wanted: set[str],
            budget: int = CONTEXT_BUDGET, extra_types: list[str] = ()) -> str:
    """v7's extra declarations for one function: blocks ("// path" + one definition, or a file's
    prototype lines), in priority order, within `budget` chars. `code`: the current definition or
    the m2c draft; `wanted`: the identifiers it and the asm use (llmtry.prepare's); `extra_types`:
    more types the code's pointers may hold (draft_types), shown in full.

    Types show all their data members; over the budget a type shows only the members the code
    names (TOUCHED), then none (SLIM); the called methods' classes start SLIM."""
    from .demangle import demangle
    try:
        dm = demangle(symbol)
    except Exception:
        dm = None
    methods, free = _called(calls)
    names = set(wanted) | {m for ms in methods.values() for m in ms} | set(free)
    names |= {f"m{int(h, 16):X}" for h in re.findall(r"\bunk([0-9A-Fa-f]+)\b", code)}   # the project's mOFFSET names
    order: list[list] = []                               # [type name, mode]
    seen: dict[str, int] = {}

    def add(name: str, mode: int = FULL, depth: int = 0) -> None:
        d = index.types.get(name)
        if d is None or depth > 8:
            return
        if name in seen:
            at = order[seen[name]]
            if mode < at[1]:
                at[1] = mode                             # needed with more members after all
                for b in d.bases + d.refs:
                    add(b, mode, depth + 1)
            return
        seen[name] = len(order)
        order.append([name, mode])
        for b in d.bases + d.refs:                       # base classes; a typedef's target
            add(b, mode, depth + 1)

    known = FULL            # measured: showing only the members the draft names loses more than it saves
    if dm is not None and dm.quals:
        add(dm.quals[-1], known)
    if dm is not None:
        for p_ in dm.params:
            for t in _types_in(p_.text, index):
                add(t, known)
    protos: list[tuple[str, str]] = []
    for n in sorted(set(free)):
        if n in index.funcs and n != symbol:
            protos.append(index.funcs[n])
    for n in sorted(set(refs)):
        if n in index.globals:
            protos.append(index.globals[n])
            for t in _types_in(index.globals[n][1], index):
                add(t, known)
    for cls in sorted(methods):                          # the called methods' classes, slim unless needed
        d = index.types.get(cls)
        if d is None:
            continue
        add(cls, SLIM)
        for m in members(d)[1] if "{" in d.text else []:
            if m.name in methods[cls] and m.kind in ("method", "virtual", "inline"):
                for t in _types_in(m.text.split("(", 1)[0], index, cls):
                    add(t)                               # what it returns: a local holds it
    for n in sorted(set(free)):                          # what called functions return
        if n in index.funcs:
            pr = index.funcs[n][1]
            for t in _types_in(pr[:pr.find("(")] if "(" in pr else pr, index):
                add(t)
    for t in _types_in(code, index):
        add(t)
    for t in list(extra_types) + convention_types(code, index):
        add(t)
    for n in sorted(set(free)):                          # what called functions take (a local passed by address)
        if n in index.funcs:
            pr = index.funcs[n][1]
            for t in _types_in(pr[pr.find("("):] if "(" in pr else "", index):
                add(t)
    for cls in sorted(methods):
        d = index.types.get(cls)
        for m in members(d)[1] if d is not None and "{" in d.text else []:
            if m.name in methods[cls] and m.kind in ("method", "virtual", "inline") and "(" in m.text:
                for t in _types_in(m.text[m.text.find("("):].split("{", 1)[0], index, cls):
                    add(t)
    for _round in range(2):                              # the types of the members it names, two levels
        for name, mode in [tuple(x) for x in order]:
            d = index.types[name]
            if mode == SLIM or "{" not in d.text or d.kind == "enum":
                continue
            for m in members(d)[1]:
                if m.kind == "data" and m.name in names:
                    for t in _types_in(m.text, index, name):
                        add(t)

    blocks, size = [], 0
    by_path: dict[str, list[str]] = {}
    for path, line in protos:
        if line not in by_path.setdefault(path, []):
            by_path[path].append(line)
    for path, lines in by_path.items():                 # prototypes first: one line each, the cheapest
        text = f"// {path}\n" + "\n".join(lines)
        if size + len(text) <= budget:
            blocks.append(text)
            size += len(text) + 2
    def helpers(own: bool) -> None:
        """Inline free functions (whole when short): the unit's own, or the headers' its other code calls."""
        nonlocal size
        by: dict[str, list[str]] = {}
        for n in sorted(index.inlines):
            path, body, mine = index.inlines[n]
            if n != symbol and mine == own and (own or n in index.unit_calls):
                by.setdefault(path, []).append(body)
        for path, bodies in by.items():
            keep = []
            for b in bodies:
                if size + len(b) + len(path) + 4 <= budget:
                    keep.append(b)
                    size += len(b) + 1
            if keep:
                blocks.append(f"// {path}\n" + "\n".join(keep))
                size += len(path) + 4
    helpers(True)                                        # the unit's own helpers: few, and v6's trim dropped them
    for name, mode in order:
        d = index.types[name]
        text = f"// {d.path}\n" + render_type(d, names, mode)
        if size + len(text) > budget and mode != SLIM and "{" in d.text and d.kind != "enum":
            text = f"// {d.path}\n" + render_type(d, names, TOUCHED if mode == FULL else SLIM)
        if size + len(text) > budget and mode == FULL and "{" in d.text and d.kind != "enum":
            text = f"// {d.path}\n" + render_type(d, names, SLIM)
        if size + len(text) <= budget:
            blocks.append(text)
            size += len(text) + 2
    helpers(False)
    return "\n\n".join(blocks)


_UNK = re.compile(r"\b([A-Za-z_]\w*)\s*->\s*unk([0-9A-Fa-f]+)\b")


def convention_types(code: str, index: Index, most: int = 3) -> list[str]:
    """draft_types by the project's naming (a field at offset 0x2A4 is often `m2A4`): for each variable
    the draft reads `v->unkNN` through, the indexed types with a data member `mNN` for every offset
    (at most `most`), else the ones an offset has alone."""
    offsets: dict[str, set[int]] = {}
    for m in _UNK.finditer(code):
        if m[1] != "this":
            offsets.setdefault(m[1], set()).add(int(m[2], 16))
    if not offsets:
        return []
    holders: dict[str, set[str]] = {}
    for name in sorted(index.types):
        d = index.types[name]
        if "{" in d.text and d.kind not in ("enum", "typedef"):
            for mem in members(d)[1]:
                if mem.kind == "data" and re.fullmatch(r"m[0-9A-F]+", mem.name):
                    holders.setdefault(mem.name, set()).add(name)
    out: list[str] = []
    for var in sorted(offsets):
        sets = [holders.get(f"m{o:X}", set()) for o in sorted(offsets[var]) if o]
        if not sets:
            continue
        both = set.intersection(*sets)
        pick = both if 1 <= len(both) <= most else {n for h in sets if len(h) == 1 for n in h}
        out += [n for n in sorted(pick) if n not in out]
    return out


def draft_types(code: str, lay, most: int = 3) -> list[str]:
    """The struct types a draft's still-unnamed pointers may hold: for each variable the draft
    reads `v->unkNN` through, the types of the unit's `-sym on` layouts (draftfields.Layouts) with
    a member at every one of its offsets, when at most `most` fit (fewer, smaller ones first)."""
    if lay is None or not getattr(lay, "by_name", None):
        return []
    offsets: dict[str, set[int]] = {}
    for m in _UNK.finditer(code):
        if m[1] != "this":
            offsets.setdefault(m[1], set()).add(int(m[2], 16))
    out: list[str] = []
    for var in sorted(offsets):
        offs = offsets[var]
        if offs == {0}:
            continue                                     # every struct has something at 0
        fits = []
        for name, ref in sorted(lay.by_name.items()):
            if lay.size(ref) <= max(offs) or name.startswith(("@", "__")):
                continue
            if all(lay.at(ref, o) is not None for o in offs):
                fits.append((lay.size(ref), name))
        if 1 <= len(fits) <= most:
            out += [n for _s, n in sorted(fits) if n not in out]
    return out


_DEFINES = r"(?:(?:class|struct|union|enum)\s+{n}\b[^;{{]*\{{|typedef\b[^;]*\b{n}\s*;|\}}\s*{n}\s*;)"


def defined(name: str, text: str) -> bool:
    return re.search(_DEFINES.format(n=re.escape(name)), text) is not None


def block_name(block: str) -> str | None:
    body = block.split("\n", 1)[1] if "\n" in block else ""
    m = re.match(r"(?:typedef\s+)?(?:class|struct|union|enum)\s+([A-Za-z_]\w*)", body)
    if m and not body.startswith("typedef"):
        return m[1]
    if body.startswith("typedef"):
        return _decl_name(body.rsplit("}", 1)[-1].rstrip(";")) if "{" in body else _decl_name(body.rstrip(";"))
    return None


def merge(context_text: str, decls: str) -> str:
    """The context blocks another declarations text does not already define (a record's stored
    declarations), before it: a type block whose type `decls` defines goes, and so does a prototype or
    global line whose name `decls` already declares."""
    if not context_text.strip():
        return decls
    keep = []
    for block in context_text.split("\n\n"):
        name = block_name(block)
        if name is not None:
            if not defined(name, decls):
                keep.append(block)
            continue
        head, *lines = block.split("\n")
        lines = [l for l in lines if not _declared(l, decls)]
        if lines:
            keep.append("\n".join([head, *lines]))
    if not keep:
        return decls
    return "\n\n".join(keep) + "\n\n" + decls


def _declared(line: str, decls: str) -> bool:
    paren = line.find("(")
    if paren >= 0 and not re.search(r"\(\s*\*", line[:paren + 3]):
        m = re.search(r"([A-Za-z_]\w*)\s*$", line[:paren])
        return bool(m) and re.search(rf"\b{m[1]}\s*\([^;{{]*\)[^;{{]*[;{{]", decls) is not None
    name = _decl_name(line.rstrip(";"))
    return bool(name) and re.search(rf"^[^\n(]*\b{re.escape(name)}\b\s*(?:\[[^\]\n]*\]\s*)*[;=]", decls, re.M) is not None


def unit_index(cfg, src: str, source: Path, span: tuple[int, int]) -> Index:
    """The Index of a unit: its own source (the target skipped) and every header it can see."""
    from .llmtry import visible_files
    return build_index([(cfg.rel(p), t) for p, t in visible_files(cfg, src, source)], cfg.rel(source), src, span)


# ---- coverage: does the prompt show what the exact answer uses? (dk dataset coverage) -------------------

_CPP_WORDS = frozenset("if for while switch return sizeof static_cast reinterpret_cast const_cast new delete case "
                       "default do else aligned __attribute__".split())


def answer_uses(code: str) -> tuple[set[str], set[str]]:
    """(the members it reads through `->`/`.`, the functions and macros it calls) of a definition."""
    from .llmtry import KEYWORDS
    code = re.sub(r"//[^\n]*|/\*.*?\*/", "", re.sub(r'"(?:\\.|[^"\\])*"', '""', code), flags=re.S)
    mem = set(re.findall(r"(?:->|\.)\s*([A-Za-z_]\w*)", code))
    calls = set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", code)) - _CPP_WORDS - KEYWORDS - mem
    return mem, calls


def gaps(answer: str, prompt: str, draft: str, own: set[str]) -> dict:
    """What an exact answer uses that a prompt never shows: any member or call (`members`, `calls`),
    and the spot check's two measures: an `mXX` field the draft's `unkXX` does not predict
    (`field`), a named helper call (not fn_/lbl_/vf_) (`helper`)."""
    words = set(_IDENT.findall(prompt))
    mem, calls = answer_uses(answer)
    calls -= own
    unk = {int(h, 16) for h in re.findall(r"\bunk([0-9A-Fa-f]+)\b", draft)}
    miss_m, miss_c = mem - words, calls - words
    return {"members": sorted(miss_m), "calls": sorted(miss_c),
            "field": sorted(m for m in miss_m if re.fullmatch(r"m[0-9A-F]+", m) and int(m[1:], 16) not in unk),
            "helper": sorted(c for c in miss_c if not re.match(r"(?:fn|lbl|vf\d*)_", c))}


def coverage(cfg, args) -> int:
    """Over the base records (exact answers known): how often the answer uses a member or call that
    the stored v6 prompt never shows, against prompt v7 built from the same draft and HEAD's source."""
    import json
    import shutil
    import statistics
    from .draftfields import Layouts
    from .elf import Elf
    from .llmdata import out_dir
    from .llmtry import CHARS_PER_TOKEN, LLM_DIR, asm_entries, declarations, identifiers, signature
    from .matching import resolve_unit
    from .reorder import compile_variant
    from .trycmd import locate_target
    path = out_dir(cfg, args) / "records" / "base.jsonl"
    recs = [json.loads(x) for x in path.read_text().splitlines() if x.strip()][::max(1, args.every)]
    entries, units, n = asm_entries(cfg), {}, 0
    count = {f"{v} {k}": 0 for v in ("v6", "v7") for k in ("any", "members", "calls", "field", "helper")}
    growth, sizes = [], {"v6": [], "v7": []}
    for rec in recs:
        view = rec.get("views", {}).get("6")
        entry = entries.get(rec["function"])
        if view is None or entry is None:
            continue
        if rec["unit"] not in units:
            u = resolve_unit(cfg, rec["unit"])
            source = cfg.root / u["metadata"]["source_path"]
            src = source.read_text()
            work = cfg.root / LLM_DIR / "work" / "coverage_layout"
            try:
                obj, _ = compile_variant(cfg, u, src, work, ("-sym", "on"))
                lay = Layouts(Elf.parse(obj.read_bytes())) if obj is not None else None
            finally:
                shutil.rmtree(work, ignore_errors=True)
            units[rec["unit"]] = (source, src, lay)
        source, src, lay = units[rec["unit"]]
        try:
            span = locate_target(cfg, src, source, rec["function"])[0]
        except Exception:
            continue                                     # no longer in the source
        draft = view["m2c"]
        wanted = identifiers(draft) | set(entry.calls) | set(entry.refs)
        ctx = context(unit_index(cfg, src, source, span), rec["function"], draft, list(entry.calls), list(entry.refs),
                      wanted, extra_types=draft_types(draft, lay))
        d7 = merge(ctx, declarations(src, span, wanted, 8000, True))
        sig = signature(rec["function"])
        own = set(_IDENT.findall(sig)) | {rec["function"]}
        for v, decls in (("v6", view["decls"]), ("v7", d7)):
            g = gaps(rec["exact"], "\n".join([rec["asm"], decls, draft, view.get("notes", ""), sig]), draft, own)
            count[f"{v} any"] += bool(g["members"] or g["calls"])
            for k in ("members", "calls", "field", "helper"):
                count[f"{v} {k}"] += bool(g[k])
            sizes[v].append(len(decls) / CHARS_PER_TOKEN)
        growth.append((len(d7) - len(view["decls"])) / CHARS_PER_TOKEN)
        n += 1
    if not n:
        print(f"no base records with a v6 view in {cfg.rel(path)}")
        return 2

    def pct(q: list[float], at: float) -> int:
        q = sorted(q)
        return round(q[min(len(q) - 1, int(at * len(q)))])
    print(f"{n} base records ({cfg.rel(path)}, every {args.every}): answers using something the prompt never shows")
    print(f"  {'':8} {'v6':>14} {'v7':>14}")
    for k, what in (("any", "any member or call"), ("members", "a member"), ("calls", "a call"),
                    ("field", "mXX not from unkXX"), ("helper", "a named helper")):
        print(f"  {what:19} " + " ".join(f"{count[f'{v} {k}']:6d} {100 * count[f'{v} {k}'] / n:5.1f}%" for v in ("v6", "v7")))
    print(f"  declarations tokens: v6 median {pct(sizes['v6'], .5)}, p90 {pct(sizes['v6'], .9)}; v7 median "
          f"{pct(sizes['v7'], .5)}, p90 {pct(sizes['v7'], .9)}; growth median {round(statistics.median(growth))}, "
          f"p90 {pct(growth, .9)}, max {pct(growth, 1)}")
    return 0
