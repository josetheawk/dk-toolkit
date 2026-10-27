"""dk reorder <unit>: put a unit's function definitions into the target's address order.

The compiler emits a unit's functions, and the strings and literals they introduce, in source
order, so a source written in another order fails layout and literal offsets even when every
function matches. This sorts it:

- the source is split into top-level items (comments, strings and directives understood); a
  function's leading comments move with it;
- a private `-sym on` compile maps every function in the main `.text` to the item that defines
  it (DWARF 1 line table); the target object gives each function's address;
- items that define functions are sorted by the lowest target address they define (an item whose
  functions the target lacks stays behind the item it followed); a data definition travels with
  the function item that follows it, so its strings keep their place among that function's;
  declarations, types and helpers that emit no function of their own move above the functions,
  in their original order;
- matched `#pragma X on|off|…` / `#pragma X reset`, `push`/`pop` and `#if`…`#endif` groups move
  as one item; any other directive between functions is a barrier no item crosses;
- everything before the first function item and after the last one stays where it is.

The new order is compiled privately (a mirror of the tree, so the real source is not touched
until it is proven) and kept only when it compiles and every function that was exact stays exact
and no data section that matched stops matching. The report lists the literal sections that
still differ, with the first object out of place and the function that first uses it.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import struct
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ...core.command import BUILD_ERROR, MISMATCH, OK, USAGE, Command, hint
from ...core.config import Config
from .elf import SHT_RELA, STT_FUNC, Elf
from .matching import MatchError, UnitScore, compile_command, resolve_unit, retarget, score

ADDR = re.compile(r"fn_([0-9A-Fa-f]{8})")
STB_WEAK = 2


# ---- splitting the source into top-level items -----------------------------------------------

@dataclass
class Item:
    start: int                  # first line (0-based), including the comments and blank lines that lead it
    body: int                   # first line of the item itself
    end: int                    # last line, inclusive
    kind: str                   # "code", "pp" (one directive), "region" (matched directives), "trivia"
    head: str = ""              # code: its significant text at brace depth 0 (strings blanked)
    block: bool = False         # code: ends in a `}` block (function, namespace, extern "C" {…})
    directive: str = ""         # pp: the directive, continuation lines joined
    children: list["Item"] = field(default_factory=list)   # region: the items inside
    text: list[str] | None = None   # lines written in place of the source's (a region split per item)

    def lines(self, src: list[str]) -> list[str]:
        return self.text if self.text is not None else src[self.start:self.end + 1]


class SplitError(Exception):
    """The source could not be split (unbalanced braces): nothing is reordered."""


def _needs_semicolon(head: str) -> bool:
    """Does a `{` after this depth-0 text open an initializer or a type body (closed by `;`)
    rather than a function body, namespace or linkage block (closed by `}`)?"""
    h = re.sub(r"\boperator\s*(\(\)|[^\s(]+)", "operator", head)
    flat = h
    while True:
        nxt = re.sub(r"\([^()]*\)", "\x00", flat)
        if nxt == flat:
            break
        flat = nxt
    if re.search(r"(?<![=!<>])=(?!=)", flat):
        return True
    flat = re.sub(r"^\s*template\s*<[^>]*>\s*", "", flat)
    return bool(re.match(r"\s*(typedef\s+)?(struct|class|union|enum)\b", flat)) and "\x00" not in flat


def split_items(text: str) -> list[Item]:
    """Top-level items, covering every line exactly once, in order."""
    n = len(text)
    items: list[Item] = []
    depth = 0
    line = 0
    i = 0
    start, body = 0, None           # current item: first line, first significant line
    head: list[str] = []
    needs_semi = False
    block = False
    ended = False                   # a terminator was seen; the item ends with this line
    line_clean = True               # only whitespace so far on this line

    def close(last: int, kind: str = "code", directive: str = "") -> None:
        nonlocal start, body, head, needs_semi, block, ended
        items.append(Item(start, start if body is None else body, last, kind, "".join(head), block, directive))
        start, body, head, needs_semi, block, ended = last + 1, None, [], False, False, False

    while i < n:
        c = text[i]
        if c == "\n":
            if ended and depth == 0:
                close(line)
            line += 1
            line_clean = True
            i += 1
            continue
        if c in " \t\r\f\v":
            i += 1
            continue
        if text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            line += text.count("\n", i, j)
            i = j
            continue
        if c == "#" and line_clean and depth == 0 and body is None:
            j = i
            while True:
                k = text.find("\n", j)
                if k < 0:
                    k = n
                    break
                if text[k - 1:k] == "\\" or text[k - 2:k] == "\\\r":
                    j = k + 1
                    continue
                break
            directive = re.sub(r"\\\r?\n", " ", text[i:k]).strip()
            body = line
            line += text.count("\n", i, k)
            close(line, "pp", directive)
            i = k
            continue
        line_clean = False
        if body is None:
            body = line
        ended = False
        if c in "\"'":
            j = i + 1
            while j < n and text[j] != c and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            if depth == 0:
                head.append('""')
            i = j + 1
            continue
        if c == "{":
            if depth == 0:
                needs_semi = _needs_semicolon("".join(head))
            depth += 1
        elif c == "}":
            depth -= 1
            if depth < 0:
                raise SplitError(f"unbalanced '}}' on line {line + 1}")
            if depth == 0:
                head.append("{}")                   # the body, so a declarator after it shows
                if not needs_semi:
                    ended = block = True
        elif c == ";" and depth == 0:
            ended = True
        elif depth == 0:
            head.append(c)
        i += 1
        # keep identifiers apart in the head (`struct S` must not read as `structS`)
        if depth == 0 and i < n and text[i] in " \t\n" and head and head[-1] != " ":
            head.append(" ")
    if depth != 0:
        raise SplitError("unbalanced braces at end of file")
    last = text.count("\n")
    if body is not None:
        close(last)
    elif start <= last:
        items.append(Item(start, start, last, "trivia"))
    return items


_REGION_IF = ("if", "ifdef", "ifndef")


def _pp_word(directive: str) -> tuple[str, str]:
    m = re.match(r"#\s*(\w+)\s*(.*)", directive)
    return (m.group(1), m.group(2).strip()) if m else ("", "")


def group_regions(items: list[Item]) -> list[Item]:
    """Fold `#if`…`#endif` groups and matched pragma regions (`#pragma X on|off|…` … `#pragma X
    reset`, `push`/`pop`, `pack(push…)`/`pack(pop)`) into single region items. Overlapping
    regions merge. Unmatched directives stay single `pp` items (barriers)."""
    spans: list[tuple[int, int]] = []
    ifs: list[int] = []
    opened: dict[str, list[tuple[int, str]]] = {}
    for idx, it in enumerate(items):
        if it.kind != "pp":
            continue
        word, rest = _pp_word(it.directive)
        if word in _REGION_IF:
            ifs.append(idx)
        elif word == "endif" and ifs:
            spans.append((ifs.pop(), idx))
        elif word == "pragma":
            m = re.match(r"(\w+)\s*(.*)", rest)
            if not m:
                continue
            name, arg = m.group(1), m.group(2).strip()
            if name == "pack":
                name, arg = "pack", "pop" if "pop" in arg else ("push" if "push" in arg else "")
                if arg == "push":
                    opened.setdefault(name, []).append((idx, arg))
                elif arg == "pop" and opened.get(name):
                    spans.append((opened[name].pop()[0], idx))
            elif name == "push":
                opened.setdefault("push", []).append((idx, ""))
            elif name == "pop":
                if opened.get("push"):
                    spans.append((opened["push"].pop()[0], idx))
            elif arg == "reset":
                if opened.get(name):
                    spans.append((opened[name].pop()[0], idx))
            elif arg:
                if opened.get(name) and opened[name][-1][1] != arg:      # `on` … `off` toggles back
                    spans.append((opened[name].pop()[0], idx))
                else:
                    opened.setdefault(name, []).append((idx, arg))
    spans.sort()
    merged: list[list[int]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    out: list[Item] = []
    idx = 0
    for a, b in merged:
        out += items[idx:a]
        kids = items[a:b + 1]
        out.append(Item(kids[0].start, kids[0].body, kids[-1].end, "region", children=kids))
        idx = b + 1
    out += items[idx:]
    return out


# ---- classifying items -----------------------------------------------------------------------

def _is_function_shaped(it: Item) -> bool:
    return it.kind == "code" and it.block and "(" in it.head


_ASSIGN = re.compile(r"(?<![=!<>])=(?!=)")


def is_data(it: Item) -> bool:
    """A definition that may emit data or static-init code (whose strings, literals and `__sinit`
    order follow source order): anything that is not clearly a function, a type, a
    typedef/using/template/friend, an `extern` declaration or a prototype. A macro invocation
    (`REGISTER(x, "y");`) counts as data."""
    if it.kind == "region":
        return any(is_data(k) for k in it.children)
    if it.kind != "code" or it.block:
        return False
    h = re.sub(r"\s+", " ", it.head).strip()
    h = re.sub(r'^extern ""\s*', "extern ", h)
    if not h or re.match(r"(typedef|using|template|friend|namespace)\b", h):
        return False
    if _ASSIGN.search(h) or '""' in h:
        return True
    if re.match(r"extern\b", h):
        return False
    if re.match(r"(struct|class|union|enum)(\s+[\w:]+)?\s*(:[^{]*)?(\{\})?$", h):    # forward declaration or type body
        return False
    # a prototype: a return type, a name, then a parameter list without literals
    m = re.match(r"(?:[\w:*&<>,~]+\s*[\s*&]\s*)+(?:operator\s*\S+?|[\w:~]+)\s*\((?P<params>[^()]*(\([^()]*\)[^()]*)*)\)\s*(const\s*)?$", h)
    if m and not re.search(r"(^|[\s,(])-?\d", m.group("params")) and not re.search(r"\(\s*\*", h):
        return False
    return True


# ---- what the compiled object says -----------------------------------------------------------

def line_table(elf: Elf) -> dict[int, list[tuple[int, int]]]:
    """DWARF 1 `.line`: section index -> [(offset, line)] (lines 1-based, 0 ends a sequence)."""
    sec = elf.section(".line")
    if sec is None:
        return {}
    base_of: dict[int, tuple[int, int]] = {}
    for s in elf.sections:
        if s.type == SHT_RELA and s.info == sec.index:
            for off, info, addend in struct.iter_unpack(">IIi", s.data):
                sym = elf.symbols[info >> 8]
                base_of[off] = (sym.shndx, sym.value + addend)
    out: dict[int, list[tuple[int, int]]] = {}
    off, d = 0, sec.data
    while off + 8 <= len(d):
        length, = struct.unpack_from(">I", d, off)
        if length < 8:
            break
        shndx, base = base_of.get(off + 4, (0, 0))
        rows = out.setdefault(shndx, [])
        for k in range((length - 8) // 10):
            ln, _col, delta = struct.unpack_from(">IHI", d, off + 8 + 10 * k)
            if ln:
                rows.append((base + delta, ln))
        off += length
    return out


def main_text_index(elf: Elf) -> int | None:
    return next((s.index for s in elf.sections if s.name == ".text"), None)


def function_lines(elf: Elf) -> dict[str, int]:
    """Function in the main `.text` -> its first source line (0-based). Left out: compiler-made
    functions (`__sinit_…`, `@…` thunks) and weak ones (inline functions, even those defined in
    this file: the compiler places their copy by first use or at the end, not by definition)."""
    main = main_text_index(elf)
    rows = sorted(line_table(elf).get(main, []))
    out = {}
    for s in elf.symbols:
        if s.type != STT_FUNC or s.shndx != main or not s.name or s.name.startswith(("__sinit_", "@")) or s.bind == STB_WEAK:
            continue
        inside = [ln for off, ln in rows if s.value <= off < s.value + max(s.size, 1)]
        first = [ln for off, ln in rows if off == s.value]
        if first or inside:
            out[s.name] = (first[0] if first else min(inside)) - 1
    return out


def deferred_functions(elf: Elf, fn_lines: dict[str, int]) -> set[str]:
    """Functions the compiler emitted after all the others, in reverse definition order (MWCC
    1.3-2.7 does this to an `extern "C"` function that assigns a struct): the tail of the main
    `.text` whose source lines run backwards, each defined before some function emitted ahead of
    it. Their place in the source does not move them in `.text`; only their order among
    themselves counts (reversed)."""
    main = main_text_index(elf)
    seq = [s.name for s in sorted((s for s in elf.symbols if s.type == STT_FUNC and s.shndx == main and s.name in fn_lines),
                                  key=lambda s: s.value)]
    if not seq:
        return set()
    k = len(seq) - 1                       # where the tail that runs backwards starts
    while k > 0 and fn_lines[seq[k - 1]] > fn_lines[seq[k]]:
        k -= 1
    tail, head = seq[k:], max((fn_lines[n] for n in seq[:k]), default=-1)
    # the tail's first function may just be the last ordinary one: only what runs backwards from
    # it, or sits above an ordinary function emitted before it, is known to come last
    return {n for i, n in enumerate(tail) if i > 0 or fn_lines[n] < head}


def text_offsets(elf: Elf) -> dict[str, int]:
    """Function name -> offset in the target's `.text` (dtk's padding labels left out)."""
    sec = elf.section(".text")
    if sec is None:
        return {}
    out = {}
    for s in elf.symbols:
        if s.type == STT_FUNC and s.shndx == sec.index and s.name and not (s.name.startswith("gap_")):
            out[s.name] = s.value
    return out


def _target_key(name: str, offsets: dict[str, int], by_addr: dict[str, int]) -> int | None:
    if name in offsets:
        return offsets[name]
    m = ADDR.search(name)
    return by_addr.get(m.group(1).upper()) if m else None


# ---- planning --------------------------------------------------------------------------------

@dataclass
class Plan:
    text: str
    functions: int = 0              # functions placed by the target's order
    unknown: list[str] = field(default_factory=list)       # ours, not in the target (stripped or misnamed)
    moved: list[str] = field(default_factory=list)          # first function of each item that moved
    notes: list[str] = field(default_factory=list)
    deferred: set[str] = field(default_factory=set)         # functions the compiler emits last (see deferred_functions)


def _shift(it: Item, by: int) -> Item:
    it.start, it.body, it.end = it.start + by, it.body + by, it.end + by
    for k in it.children:
        _shift(k, by)
    return it


_CODE = re.compile(r'"(?:\\.|[^"\\])*"|//.*|/\*.*?\*/')


def container(it: Item, src: list[str]) -> tuple[int, int] | None:
    """(opening line, closing line) of a `namespace … {` or `extern "C" {` block written with its
    braces at line ends, whose inside is arranged on its own; None for anything else."""
    if it.kind != "code" or not it.block or not re.match(r'\s*(namespace\b|extern\s*""\s*\{)', it.head):
        return None
    code = lambda n: _CODE.sub(lambda m: '""' if m.group(0).startswith('"') else "", src[n]).strip()
    opening = next((n for n in range(it.body, it.end) if "{" in code(n)), None)
    if opening is None or not code(opening).endswith("{") or code(opening).count("{") != 1:
        return None
    if not re.fullmatch(r"\};?", code(it.end)):
        return None
    return opening, it.end


def _strip_attributes(head: str) -> str:
    return re.sub(r"__(declspec|attribute__)\s*\((?:[^()]|\([^()]*\))*\)", " ", head)


def data_name(it: Item) -> str | None:
    """The name a data definition declares (`static T lbl_x[3] = {…}` -> lbl_x); None for a macro
    invocation or anything without a type before the name."""
    h = re.sub(r"\s+", " ", _strip_attributes(it.head)).strip()
    h = re.split(r"(?<![=!<>])=(?!=)", h, maxsplit=1)[0]
    m = re.search(r"\(\s*\*\s*([A-Za-z_]\w*)\s*\)", h)
    if m:
        return m.group(1)
    h = re.sub(r"\[[^\]]*\]", "", h)
    h = re.sub(r"\{\}", "", h)
    h = re.sub(r"\((?:[^()]|\([^()]*\))*\)\s*$", "", h).strip()
    words = re.findall(r"[A-Za-z_][\w:]*", h)
    return words[-1] if len(words) >= 2 and "::" not in words[-1] else None


def free_function(it: Item) -> str | None:
    """The name of a function item that is not a class member (`int f(…) {…}`, `operator+`)."""
    if not _is_function_shaped(it):
        return None
    h = _strip_attributes(it.head)
    if re.match(r"\s*template\b", h):
        return None
    m = re.search(r"(operator\s*[^\s(]+|[A-Za-z_~][\w:~]*)\s*\(", h)
    if not m or "::" in m.group(1) or m.group(1) in ("if", "while", "for", "switch", "sizeof"):
        return None
    return re.sub(r"\s+", "", m.group(1))


def prototype(text: str) -> str | None:
    """A declaration for the function defined in `text`: its declarator up to the body, one line."""
    depth, i, n = 0, 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            text = text[:i] + (text[j:] if j >= 0 else "")
            n = len(text)
            continue
        if text.startswith("/*", i):
            j = text.find("*/", i)
            text = text[:i] + " " + (text[j + 2:] if j >= 0 else "")
            n = len(text)
            continue
        if c in "\"'":
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == "\\" else 1
            i = j + 1
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "{" and depth == 0:
            decl = re.sub(r"\s+", " ", text[:i]).strip()
            decl = re.sub(r"\basm\s+", "", decl)
            return decl + ";" if decl else None
        i += 1
    return None


def _adjacent(hits: list[int], span: list[int]) -> bool:
    return span.index(hits[-1]) - span.index(hits[0]) + 1 == len(hits)


def split_regions(items: list[Item], src: list[str], fn_lines: dict[str, int], offsets: dict[str, int],
                  by_addr: dict[str, int], result: Plan) -> list[Item]:
    """A pragma region (`#pragma X on` … `#pragma X reset`) around several functions the target
    does not keep together becomes one region per function or data item, each wrapped in the same
    directives, so each can move on its own. Declarations inside leave the region."""
    span = sorted(offsets.values())
    out: list[Item] = []
    for it in items:
        kids = it.children
        opens = [k for k in kids[:next((n for n, k in enumerate(kids) if k.kind != "pp"), len(kids))]]
        closes = kids[len(kids) - next((n for n, k in enumerate(reversed(kids)) if k.kind != "pp"), len(kids)):]
        middle = kids[len(opens):len(kids) - len(closes)]
        simple = (it.kind == "region" and opens and closes and middle
                  and all(_pp_word(k.directive)[0] == "pragma" for k in opens + closes)
                  and all(k.kind in ("code", "trivia") for k in middle))
        if simple:
            hits = sorted(h for k in middle if _is_function_shaped(k)
                          for f, ln in fn_lines.items() if k.body <= ln <= k.end
                          if (h := _target_key(f, offsets, by_addr)) is not None)
            simple = len(hits) > 1 and not _adjacent(hits, span)
        if not simple:
            out.append(it)
            continue
        head = [ln for k in opens for ln in src[k.body:k.end + 1]]
        tail = [ln for k in closes for ln in src[k.body:k.end + 1]]
        lead = src[it.start:opens[0].body]                 # comments above the region stay above
        for n, k in enumerate(middle):
            if k.kind == "trivia" or not (_is_function_shaped(k) or is_data(k)):
                out.append(k)
                continue
            body = src[k.start:k.end + 1]
            wrapped = Item(k.start, k.body, k.end, "region", children=[k],
                           text=(lead if n == 0 else []) + head + body + tail)
            out.append(wrapped)
        result.notes.append(f"line {it.body + 1}: the region `{opens[0].directive[:40]}` wraps functions the target "
                            f"keeps apart: split into one region per item")
    return out


def plan(text: str, fn_lines: dict[str, int], offsets: dict[str, int], deferred: set[str] = frozenset()) -> Plan:
    src = text.split("\n")
    result = Plan(text, deferred=set(deferred))
    by_addr = {m.group(1).upper(): off for name, off in offsets.items() if (m := ADDR.search(name))}
    lines = _arrange(src, 0, len(src) - 1, fn_lines, offsets, by_addr, result)
    if result.functions == 0 and not result.unknown:
        result.notes.append("no function of the unit's main .text is defined in its source")
    result.text = "\n".join(lines)
    return result


def _arrange(src: list[str], lo: int, hi: int, fn_lines: dict[str, int], offsets: dict[str, int],
             by_addr: dict[str, int], result: Plan) -> list[str]:
    """The lines lo..hi rearranged (namespace and linkage blocks arranged inside, recursively)."""
    items = [_shift(it, lo) for it in group_regions(split_items("\n".join(src[lo:hi + 1])))]
    items = split_regions(items, src, fn_lines, offsets, by_addr, result)
    inner = {i: c for i, it in enumerate(items) if (c := container(it, src))}

    def fns_of(it: Item, i: int | None = None) -> list[str]:
        if it.kind == "region":
            return [f for k in it.children for f in fns_of(k)]
        if i in inner:
            a, b = inner[i]
            return sorted((f for f, ln in fn_lines.items() if a < ln < b), key=lambda f: fn_lines[f])
        if not _is_function_shaped(it):
            return []
        return sorted((f for f, ln in fn_lines.items() if it.body <= ln <= it.end), key=lambda f: fn_lines[f])

    def emit(i: int) -> list[str]:
        it = items[i]
        if i not in inner:
            return it.lines(src)
        a, b = inner[i]
        return src[it.start:a + 1] + _arrange(src, a + 1, b - 1, fn_lines, offsets, by_addr, result) + src[b:it.end + 1]

    fns = [fns_of(it, i) for i, it in enumerate(items)]
    anchored = [i for i, f in enumerate(fns) if f]
    if not anchored:
        return [line for i in range(len(items)) for line in emit(i)]
    first, last = anchored[0], anchored[-1]
    keys: dict[int, tuple] = {}
    prev: tuple = (-1,)
    span = sorted(offsets.values())
    for i in anchored:
        known = [(_target_key(f, offsets, by_addr), f) for f in fns[i]]
        hits = sorted(k for k, _ in known if k is not None)
        if i not in inner:
            result.unknown += [f for k, f in known if k is None]
            result.functions += len(hits)
        if hits and set(fns[i]) <= result.deferred:
            keys[i] = prev + (i,)                 # emitted last whatever its place: it stays behind its predecessor
            continue
        keys[i] = (hits[0], i) if hits else prev + (i,)
        prev = keys[i]
        if len(hits) > 1 and i not in inner:
            if not _adjacent(hits, span):
                result.notes.append(f"{fns[i][0]} and {len(fns[i]) - 1} more are defined in one item "
                                    f"(lines {items[i].body + 1}-{items[i].end + 1}) but are not adjacent in the target")

    out: list[int | str] = list(range(first))          # item indices, or a new line (a prototype)
    segment: list[int] = []
    text_of = lambda j: "\n".join(src[items[j].body:items[j].end + 1])

    def flush() -> None:
        groups: list[tuple[tuple, list[int]]] = []
        pending: list[int] = []
        for j in segment:
            if j in keys:
                groups.append((keys[j], pending + [j]))
                pending = []
            elif is_data(items[j]):
                pending.append(j)
        if pending:
            if groups:
                groups[-1][1].extend(pending)
            else:
                groups.append(((-1,), pending))
        ordered = sorted(groups, key=lambda g: g[0])
        # functions emitted last come out in reverse definition order: define them in reverse target order
        first_hit = lambda j: min((k for f in fns[j] if (k := _target_key(f, offsets, by_addr)) is not None), default=-1)
        is_late = lambda g: g[1][-1] in keys and fns[g[1][-1]] and set(fns[g[1][-1]]) <= result.deferred
        late = [n for n, g in enumerate(ordered) if is_late(g)]
        for n, g in zip(late, sorted((ordered[n] for n in late), key=lambda g: first_hit(g[1][-1]), reverse=True)):
            ordered[n] = g
        differ = [n for n, (a, b) in enumerate(zip(groups, ordered)) if a is not b]
        if not differ:
            out.extend(segment)                            # already in order: leave it verbatim
            segment.clear()
            return
        # only the stretch from the first group that moves to the last one is rearranged
        k, m = differ[0], differ[-1] + 1
        pos = {j: n for n, j in enumerate(segment)}
        start = min(pos[j] for j in groups[k][1])
        end = max(pos[j] for j in groups[m - 1][1])
        window = segment[start:end + 1]
        result.moved += [fns[b[1][-1]][0] for a, b in zip(groups[k:m], ordered[k:m]) if a is not b and b[1][-1] in keys]
        seq = [j for _, members in ordered[k:m] for j in members]
        earlier = "\n".join(src[lo:items[window[0]].start])
        # a data definition must still precede the first function that uses it, unless declared above
        for d in [j for j in seq if j not in keys]:
            name = data_name(items[d])
            if not name or re.search(rf"\b{re.escape(name)}\b", earlier):
                continue
            use = re.compile(rf"\b{re.escape(name)}\b")
            first_user = next((x for x in seq if x != d and use.search(text_of(x))), None)
            if first_user is not None and seq.index(first_user) < seq.index(d):
                seq.remove(d)
                seq.insert(seq.index(first_user), d)
        # a free function now defined after a use needs a prototype above the functions
        protos = []
        for n, j in enumerate(seq):
            for fn_item in (items[j].children or [items[j]]) if j in keys else []:
                name = free_function(fn_item)
                if not name:
                    continue
                use = re.compile(rf"\b{re.escape(name)}\b" if name[0].isalnum() or name[0] == "_" else re.escape(name))
                used_before = name.startswith("operator") or any(use.search(text_of(x)) for x in seq[:n])
                body = "\n".join(src[fn_item.body:fn_item.end + 1])
                if used_before and not use.search(earlier) and (proto := prototype(body)):
                    protos.append(proto)
        out.extend(segment[:start])
        out.extend(j for j in window if j not in keys and not is_data(items[j]))
        out.extend(protos)
        out.extend(seq)
        out.extend(segment[end + 1:])
        segment.clear()

    for i in range(first, last + 1):
        if items[i].kind == "pp" and _pp_word(items[i].directive)[0] != "define":   # a #define moves up with declarations
            flush()
            out.append(i)
            known = lambda r: [keys[k][0] for k in r if k in keys and len(keys[k]) == 2 and keys[k][0] >= 0]
            left, right = known(range(first, i)), known(range(i + 1, last + 1))
            if left and right and max(left) > min(right):
                result.notes.append(f"line {items[i].body + 1}: `{items[i].directive[:60]}` sits between functions the target "
                                    "orders across it: items are sorted on each side of it, never across (move it by hand)")
        else:
            segment.append(i)
    flush()
    out.extend(range(last + 1, len(items)))
    return [line for i in out for line in ([i] if isinstance(i, str) else emit(i))]


def out_of_order(elf: Elf, offsets: dict[str, int]) -> list[str]:
    """Functions of the compiled main `.text` not in the target's relative order: those outside
    the longest run (subsequence) that is."""
    main = main_text_index(elf)
    ours = [s.name for s in sorted((s for s in elf.symbols if s.type == STT_FUNC and s.shndx == main and s.name in offsets),
                                   key=lambda s: s.value)]
    seq = [offsets[n] for n in ours]
    # longest increasing subsequence, O(n log n), with back-pointers
    import bisect
    tails, tail_idx, prev = [], [], [-1] * len(seq)
    for i, v in enumerate(seq):
        k = bisect.bisect_left(tails, v)
        if k == len(tails):
            tails.append(v)
            tail_idx.append(i)
        else:
            tails[k] = v
            tail_idx[k] = i
        prev[i] = tail_idx[k - 1] if k else -1
    keep, i = set(), tail_idx[-1] if tail_idx else -1
    while i >= 0:
        keep.add(i)
        i = prev[i]
    return [n for i, n in enumerate(ours) if i not in keep]


# ---- compiling privately ---------------------------------------------------------------------

def mirror(cfg: Config, source_rel: str, text: str, work: Path) -> Path:
    """A tree under `work` where `source_rel` holds `text` and every other path links to the real
    one, so a variant compiles exactly as the real file would (same relative path for `__FILE__`
    and quoted includes) without touching the real source."""
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    parts = Path(source_rel).parts
    real, fake = Path(cfg.root).resolve(), work
    for depth, part in enumerate(parts):
        for entry in real.iterdir():
            if entry.name != part:
                (fake / entry.name).symlink_to(entry)
        real, fake = real / part, fake / part
        if depth < len(parts) - 1:
            fake.mkdir()
    fake.write_text(text)
    return work


def compile_variant(cfg: Config, unit: dict, text: str, work: Path, extra: tuple[str, ...] = ()) -> tuple[Path | None, str]:
    """Compile `text` as the unit's source in a private mirror; (object or None, compiler output).
    The mirror lives outside the repo (its links point back into it) and is removed at once."""
    import tempfile
    source_rel = unit["metadata"]["source_path"]
    out_dir = Path(work).resolve() / "out"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    argv = retarget(compile_command(cfg, unit["base_path"], locked=False), out_dir, extra_flags=extra)
    tree = Path(tempfile.mkdtemp(prefix="dk-mirror-"))
    try:
        mirror(cfg, source_rel, text, tree)
        proc = subprocess.run(argv, cwd=tree, capture_output=True, text=True)
    finally:
        shutil.rmtree(tree, ignore_errors=True)
    out = "\n".join(line for line in (proc.stdout + proc.stderr).splitlines()
                    if "Usage Warning" not in line and "Specified directory" not in line).strip()
    objs = list(out_dir.glob("*.o"))
    return (objs[0] if proc.returncode == 0 and objs else None), out


def _score(cfg: Config, unit: dict, obj: Path) -> UnitScore:
    return score(cfg, unit["name"], cfg.root / unit["target_path"], obj)


def verdict(before: UnitScore, after: UnitScore) -> list[str]:
    """Why the new order may not be kept (empty: keep it): a function or data section that scores
    lower than before (an exact one above all), or one that disappeared."""
    now = {f.name: f.percent for f in after.functions}
    why = [f"{f.name} {'was exact' if f.percent == 100.0 else f'was {f.percent:.2f}%'}, now "
           + (f"{now[f.name]:.2f}%" if f.name in now else "missing")
           for f in before.functions if now.get(f.name, -1.0) < f.percent]
    why += [f"{sec} {p:.2f}% before, now {after.data.get(sec, 0.0):.2f}%" for sec, p in sorted(before.data.items())
            if after.data.get(sec, 0.0) < p]
    return why


R_PPC_REL24 = 10


def inlining_changes(before: Elf, after: Elf, name: str) -> list[str]:
    """Calls to the unit's own functions that a regressed function lost or gained: a lost call was
    inlined (its callee is now defined above the caller), a gained one no longer is."""
    bf, af = before.functions().get(name), after.functions().get(name)
    if bf is None or af is None:
        return []
    own = set(before.functions())
    calls = lambda f: {r.symbol for r in f.relocs if r.type == R_PPC_REL24 and r.symbol in own}
    out = [f"now inlines {c}, which the original calls out of line although it defines it above"
           for c in sorted(calls(bf) - calls(af))]
    out += [f"now calls {c} out of line (it was inlined while defined above)" for c in sorted(calls(af) - calls(bf))]
    return out


def newly_inlined(before: Elf, after: Elf, name: str) -> set[str]:
    """The unit's own functions `name` called before the reorder and inlines after it."""
    bf, af = before.functions().get(name), after.functions().get(name)
    if bf is None or af is None:
        return set()
    own = set(before.functions())
    calls = lambda f: {r.symbol for r in f.relocs if r.type == R_PPC_REL24 and r.symbol in own}
    return calls(bf) - calls(af)


def wrap_dont_inline(text: str, fn_lines: dict[str, int], names: list[str]) -> str:
    """`text` with each named function's definition wrapped in `#pragma dont_inline on` …
    `#pragma dont_inline reset`, so callers defined below it keep calling it."""
    src = text.split("\n")
    spans = []

    def find(lo: int, hi: int, line: int) -> tuple[int, int] | None:
        for it in group_regions(split_items("\n".join(src[lo:hi + 1]))):
            it = _shift(it, lo)
            if it.body <= line <= it.end:
                c = container(it, src)
                if c and c[0] < line < c[1]:
                    return find(c[0] + 1, c[1] - 1, line)
                return (it.body, it.end) if _is_function_shaped(it) else None
        return None

    for name in names:
        if name in fn_lines and (span := find(0, len(src) - 1, fn_lines[name])) and span not in spans:
            spans.append(span)
    for body, end in sorted(spans, reverse=True):
        src[end + 1:end + 1] = ["#pragma dont_inline reset"]
        src[body:body] = ["#pragma dont_inline on"]
    return "\n".join(src)


def misplaced_copies(target: Elf, ours: Elf) -> list[str]:
    """Weak copies (inline functions) that follow another function in ours than in the target,
    with what moves each under the emission rule (playbook: an inline function's copy comes out
    at the end of the next C++ declaration after the function that first needs it). `ours` must
    be a normal compile: `-sym on` moves copies defined in headers into sections of their own."""
    tmain, omain = target.section(".text"), main_text_index(ours)
    if tmain is None or omain is None:
        return []
    seq = lambda elf, shndx: [x for x in sorted(elf.symbols, key=lambda x: x.value)
                              if x.type == STT_FUNC and x.shndx == shndx and x.name and not x.name.startswith("gap_")]
    weak = {x.name for x in ours.symbols if x.type == STT_FUNC and x.bind == STB_WEAK and x.shndx == omain}

    def after(order: list, name: str) -> str | None:
        names = [x.name for x in order]
        if name not in names:
            return None
        i = names.index(name) - 1
        while i >= 0 and names[i] in weak:
            i -= 1
        return names[i] if i >= 0 else None

    tseq, oseq, fns = seq(target, tmain.index), seq(ours, omain), ours.functions()
    groups: dict[tuple[str, str, str], list[str]] = {}
    for w in sorted(weak, key=lambda n: next((x.value for x in tseq if x.name == n), 0)):
        want, have = after(tseq, w), after(oseq, w)
        if want is None or want == have or w not in [x.name for x in tseq]:
            continue
        uses = want in fns and any(r.symbol == w for r in fns[want].relocs)
        names = [x.name for x in oseq]
        at_end = all(n in weak or n.startswith("__sinit") for n in names[names.index(w):])
        if at_end and not uses:
            fix = (f"ours emits it at the end of the unit (only a vtable needs it); the target has it after {want}, "
                   f"so the original needed it out of line there: reference it from {want}")
        elif "__" not in want.lstrip("_"):
            fix = f"{want} is extern \"C\": put a C++ declaration (any struct, typedef or static) right after it"
            if not uses:
                fix = f"reference {w} from {want}, and since {want} is extern \"C\", put a C++ declaration right after it"
        elif not uses:
            fix = f"reference {w} from {want} (its first user in the target's order)"
        else:
            fix = f"{want} already needs it: an earlier function must not (it would queue the copy first)"
        groups.setdefault((have or "the start", want, fix), []).append(w)
    out = []
    for (have, want, fix), copies in groups.items():
        shown = ", ".join(copies[:4]) + (f" (+{len(copies) - 4} more)" if len(copies) > 4 else "")
        out.append(f"{shown}: after {have} in ours, after {want} in the target: {fix.replace(copies[0], 'it')}")
    return out


def literal_report(target: Elf, ours: Elf, s: UnitScore) -> list[str]:
    """For each data section still below 100%: its first object out of place, with the function
    that first uses it in the target and in ours (`dk diff --literals` lists them all)."""
    from .diffcmd import literal_rows
    out = []
    for sec, pct in sorted(s.data.items()):
        if pct == 100.0:
            continue
        rows = literal_rows(target, ours, sec)
        bad = [r for r in rows[1:] if r.lstrip().startswith("!=") and "gap_" not in r.split("|")[0]]
        first = bad[0].strip() if bad else "(only padding or relocations differ)"
        out.append(f"  {sec} {pct:.2f}%: {len(bad)} object(s) differ; first: {first[2:].strip() if bad else first}")
    return out


def _summary(s: UnitScore) -> str:
    exact = sum(f.percent == 100.0 for f in s.functions)
    data = " ".join(f"{k} {v:.2f}%" for k, v in sorted(s.data.items()))
    return f"{exact}/{len(s.functions)} functions exact, code {s.code_percent:.2f}%, {len(s.layout_problems)} layout problem(s); data {data}"


def run(cfg: Config, args: argparse.Namespace) -> int:
    try:
        unit = resolve_unit(cfg, args.unit)
    except MatchError as e:
        print(f"error: {e}")
        return USAGE
    source = cfg.root / unit["metadata"]["source_path"]
    text = source.read_text()
    work = cfg.root / "build" / "dk" / "reorder" / str(os.getpid())
    try:
        obj, out = compile_variant(cfg, unit, text, work / "before", ("-sym", "on"))
        if obj is None:
            print("BUILD ERROR (the current source):")
            print(out[-3000:])
            return BUILD_ERROR
        before = _score(cfg, unit, obj)
        target = Elf.parse((cfg.root / unit["target_path"]).read_bytes())
        offsets = text_offsets(target)
        ours = Elf.parse(obj.read_bytes())
        try:
            lines = function_lines(ours)
            p = plan(text, lines, offsets, deferred_functions(ours, lines))
        except SplitError as e:
            print(f"error: cannot split {cfg.rel(source)}: {e}")
            return USAGE
        print(f"reorder {unit['name']}: {p.functions} function(s) placed by target address")
        late = out_of_order(ours, offsets)
        print(f"  before: {_summary(before)}")
        print(f"          {len(late)} function(s) out of target order"
              + (f": {', '.join(late[:6])}{' …' if len(late) > 6 else ''}" if late else ""))
        for n in p.notes:
            print(f"  note: {n}")
        if p.deferred:
            print(f"  emitted after all other functions, in reverse definition order: {', '.join(sorted(p.deferred))}")
            last_plain = max((offsets[n] for n in lines if n in offsets and n not in p.deferred), default=-1)
            early = sorted(n for n in p.deferred if offsets.get(n, -1) >= 0 and offsets[n] < last_plain)
            if early:
                print(f"  note: the target has {', '.join(early)} among the other functions, so the original's was not "
                      "emitted last (MWCC 1.3-2.7 defers an extern \"C\" function that assigns a struct): "
                      "the original did not assign a struct there, or was not extern \"C\"")
        if p.unknown:
            print(f"  not in the target (dead-stripped, or not renamed yet: ./dk rename --unit): "
                  f"{', '.join(p.unknown[:8])}{' …' if len(p.unknown) > 8 else ''}")
        if p.text == text:
            print("  already in target order")
            for row in literal_report(target, ours, before):
                print(row)
            plain, _ = compile_variant(cfg, unit, text, work / "plain")
            for row in misplaced_copies(target, Elf.parse(plain.read_bytes())) if plain else []:
                print(f"  inline copy {row}")
            hint(f"check {args.unit} --private", f"diff {args.unit} --literals")
            return OK
        print(f"  moves {len(p.moved)} item(s): {', '.join(p.moved[:10])}{' …' if len(p.moved) > 10 else ''}")
        keep = cfg.root / "build" / "dk" / "reorder" / Path(source.name)
        candidate, wrapped = p.text, []
        for attempt in range(4):
            new_obj, out = compile_variant(cfg, unit, candidate, work / "after", ("-sym", "on"))
            if new_obj is None:
                keep.write_text(candidate)
                print(f"BUILD ERROR with the new order (written to {cfg.rel(keep)}; source unchanged):")
                print(out[-3000:])
                return BUILD_ERROR
            after = _score(cfg, unit, new_obj)
            new_elf = Elf.parse(new_obj.read_bytes())
            why = verdict(before, after)
            # a function that now inlines a callee the original calls: keep that callee out of line
            inlined = sorted({c for w in why for c in newly_inlined(ours, new_elf, w.split(" ", 1)[0])} - set(wrapped))
            if not inlined or attempt == 3:
                break
            fixed = wrap_dont_inline(candidate, function_lines(new_elf), inlined)
            if fixed == candidate:
                break
            candidate, wrapped = fixed, wrapped + inlined
        p.text = candidate
        if wrapped:
            print(f"  kept out of line with #pragma dont_inline (the original calls them): {', '.join(wrapped)}")
        print(f"  after:  {_summary(after)}")
        late = out_of_order(new_elf, offsets)
        print(f"          {len(late)} function(s) out of target order"
              + (f": {', '.join(late[:6])}{' …' if len(late) > 6 else ''}" if late else ""))
        gained = [f.name for f in after.functions if f.percent == 100.0
                  and next((b.percent for b in before.functions if b.name == f.name), 0.0) != 100.0]
        if gained:
            print(f"  newly exact: {', '.join(gained)}")
        for row in literal_report(target, new_elf, after):
            print(row)
        plain, _ = compile_variant(cfg, unit, candidate, work / "plain")
        for row in misplaced_copies(target, Elf.parse(plain.read_bytes())) if plain else []:
            print(f"  inline copy {row}")
        was_late = len(out_of_order(ours, offsets))
        if len(late) > was_late or (len(late) == was_late and not gained):
            why.append(f"{len(late)} function(s) out of target order, {was_late} before: nothing gained")
        if why:
            keep.write_text(p.text)
            print(f"REFUSED (source unchanged; the new order is in {cfg.rel(keep)}):")
            for w in why:
                print(f"  {w}")
                for line in inlining_changes(ours, new_elf, w.split(" ", 1)[0]):
                    print(f"    {line}")
            return MISMATCH
        if args.dry_run:
            keep.write_text(p.text)
            print(f"dry run: the new order is in {cfg.rel(keep)}")
            return OK
        if source.read_text() != text:
            keep.write_text(p.text)
            print(f"error: {cfg.rel(source)} changed while reordering; the new order is in {cfg.rel(keep)}")
            return USAGE
        source.write_text(p.text)
        print(f"applied to {cfg.rel(source)}")
        hint(f"check {args.unit} --private", f"diff {args.unit} --literals")
        return OK
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("unit", help="unit (source path or name)")
    p.add_argument("--dry-run", action="store_true", help="verify the new order but leave the source unchanged")


COMMAND = Command("reorder", "matching",
                  "sort a unit's function definitions into target address order (verified: exact stays exact)", run, _configure)
