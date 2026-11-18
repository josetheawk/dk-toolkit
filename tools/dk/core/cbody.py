"""Span-preserving parser for C/C++ function bodies: statements, declarations and expressions.

Every node records where it sits in the source (character offsets into the whole file), so a
rewrite replaces the text of one node and leaves the rest (comments, spacing, macros) untouched.
It reads the code decompilers write, not all of C++: it needs no headers, tells types from
names by keywords, by the names the caller passes and by how a statement is shaped, and keeps a
statement it cannot read as an opaque `other` node instead of failing the whole body.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field

# ---- tokens ------------------------------------------------------------------------------------

PUNCT = sorted(("->*", "<<=", ">>=", "...", "::", "->", "++", "--", "<<", ">>", "<=", ">=", "==", "!=", "&&", "||",
                "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", ".*", "+", "-", "*", "/", "%", "<", ">", "=", "!",
                "~", "&", "|", "^", "?", ":", ";", ",", ".", "(", ")", "[", "]", "{", "}"), key=len, reverse=True)
NUMBER = re.compile(r"(?:0[xX][0-9A-Fa-f]+|(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)[fFuUlL]*")
IDENT = re.compile(r"[A-Za-z_$][\w$]*")
PUNCT_RE = re.compile("|".join(re.escape(p) for p in PUNCT))     # longest first, as PUNCT is sorted


@dataclass(frozen=True)
class Tok:
    kind: str       # "id", "num", "str", "chr", "op", "pp" (a whole preprocessor line)
    text: str
    start: int
    end: int


class ParseError(Exception):
    pass


def tokens(text: str, lo: int = 0, hi: int | None = None) -> list[Tok]:
    """Tokens of text[lo:hi]; comments and whitespace are skipped, a directive is one token."""
    return list(_tokens(text, lo, len(text) if hi is None else hi))


@functools.lru_cache(maxsize=256)       # a search re-reads the same file for every candidate it makes
def _tokens(text: str, lo: int, hi: int) -> tuple[Tok, ...]:
    out: list[Tok] = []
    i = lo
    line_start = True
    while i < hi:
        c = text[i]
        if c == "\n":
            line_start = True
            i += 1
            continue
        if c in " \t\r\f\v":
            i += 1
            continue
        if text.startswith("//", i):
            j = text.find("\n", i)
            i = hi if j < 0 or j > hi else j
            continue
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = hi if j < 0 else j + 2
            continue
        if c == "#" and line_start:
            j = i
            while True:
                k = text.find("\n", j)
                k = hi if k < 0 or k > hi else k
                if k > i and text[k - 1] == "\\" and k < hi:
                    j = k + 1
                    continue
                break
            out.append(Tok("pp", text[i:k], i, k))
            i = k
            continue
        line_start = False
        if c in "\"'" or (c == "L" and i + 1 < hi and text[i + 1] in "\"'"):
            q = text[i + 1] if c == "L" else c
            j = i + (2 if c == "L" else 1)
            while j < hi and text[j] != q:
                j += 2 if text[j] == "\\" else 1
            out.append(Tok("str" if q == '"' else "chr", text[i:j + 1], i, j + 1))
            i = j + 1
            continue
        m = NUMBER.match(text, i)
        if m and (c.isdigit() or (c == "." and i + 1 < hi and text[i + 1].isdigit())):
            out.append(Tok("num", m.group(0), i, m.end()))
            i = m.end()
            continue
        m = IDENT.match(text, i)
        if m:
            out.append(Tok("id", m.group(0), i, m.end()))
            i = m.end()
            continue
        m = PUNCT_RE.match(text, i)
        if m:
            out.append(Tok("op", m.group(0), i, m.end()))
            i = m.end()
        else:
            out.append(Tok("op", c, i, i + 1))       # `@`, a backslash: only inside asm and macros
            i += 1
    return tuple(out)


# ---- nodes -------------------------------------------------------------------------------------

@dataclass
class Expr:
    """kind: name, num, str, chr, paren, call, index, member, postfix, unary, cast, sizeof, binary,
    assign, ternary, comma, init (a braced initializer, opaque). kids: binary/assign [left, right];
    ternary [cond, then, else]; paren/unary/postfix/cast/sizeof [operand]; call [callee, args…];
    index [base, subscript]; member [base] (op "." or "->", `name` the member)."""
    kind: str
    start: int
    end: int
    op: str = ""
    kids: list["Expr"] = field(default_factory=list)
    name: str = ""
    type_span: tuple[int, int] | None = None     # cast and sizeof(type): where the type is written

    def text(self, src: str) -> str:
        return src[self.start:self.end]


@dataclass
class Declarator:
    name: str
    name_start: int
    name_end: int
    start: int                  # first `*`/`&` or the name
    end: int                    # end of the initializer (or of the name/array suffix)
    ptr: str = ""               # the pointer/reference part, spaces removed ("*", "&", "**", "*const")
    array: str = ""             # `[N]` suffixes as written
    init: Expr | None = None
    init_kind: str = ""         # "", "=", "()" (constructor arguments: init is a comma or one expr), "{}"


@dataclass
class Decl:
    type_start: int             # specifiers and base type (storage class included)
    type_end: int
    declarators: list[Declarator] = field(default_factory=list)

    def base_type(self, src: str) -> str:
        return " ".join(src[self.type_start:self.type_end].split())


@dataclass
class Stmt:
    """kind: block, decl, expr, if, for, while, do, switch, case, default, label, return, break,
    continue, goto, empty, pp, other. `head` is the inside of the parentheses of if/for/while/do/
    switch. kids: block its statements; if [then] or [then, else]; for/while/do/switch [body]."""
    kind: str
    start: int
    end: int                    # just past the statement (its `;` or `}`)
    expr: Expr | None = None    # expr statement, return value, if/while/do/switch condition, case value
    head: tuple[int, int] | None = None
    kids: list["Stmt"] = field(default_factory=list)
    decl: Decl | None = None
    init: "Stmt | None" = None  # for: the init statement (decl, expr or empty)
    cond: Expr | None = None    # for
    step: Expr | None = None    # for
    else_at: int | None = None  # if: offset of `else`
    label: str = ""             # label and goto

    def text(self, src: str) -> str:
        return src[self.start:self.end]


# ---- types -------------------------------------------------------------------------------------

TYPE_WORDS = {"void", "char", "short", "int", "long", "float", "double", "signed", "unsigned", "bool", "wchar_t",
              "__int64", "_Bool"}
QUALIFIERS = {"const", "volatile"}
STORAGE = {"static", "register", "extern", "auto", "inline", "mutable", "typedef"}
TAGS = {"struct", "class", "union", "enum", "typename"}
COMMON_TYPES = {"u8", "u16", "u32", "u64", "s8", "s16", "s32", "s64", "f32", "f64", "BOOL", "size_t", "uint",
                "uchar", "ushort", "ulong", "int8_t", "int16_t", "int32_t", "uint8_t", "uint16_t", "uint32_t",
                "va_list", "ptrdiff_t", "wint_t"}
KEYWORDS = {"if", "else", "for", "while", "do", "switch", "case", "default", "return", "break", "continue", "goto",
            "sizeof", "new", "delete", "this", "operator", "true", "false", "asm", "__asm"}
CASTS = {"static_cast", "reinterpret_cast", "const_cast", "dynamic_cast"}
ASSIGN = {"=", "+=", "-=", "*=", "/=", "%=", "<<=", ">>=", "&=", "^=", "|="}
BINARY = {"||": 4, "&&": 5, "|": 6, "^": 7, "&": 8, "==": 9, "!=": 9, "<": 10, ">": 10, "<=": 10, ">=": 10,
          "<<": 11, ">>": 11, "+": 12, "-": 12, "*": 13, "/": 13, "%": 13}
TYPE_NAME_HINT = re.compile(r"\b(?:struct|class|union|enum)\s+([A-Za-z_]\w*)|\btypedef\b[^;{}]*?\b([A-Za-z_]\w*)\s*;"
                            r"|\btypedef\s+(?:struct|union|enum)\b[^;]*?\}\s*([A-Za-z_]\w*)\s*;")


def type_names(text: str) -> set[str]:
    """Names the text declares as types (struct/class/union/enum tags and typedefs)."""
    out: set[str] = set()
    for m in TYPE_NAME_HINT.finditer(text):
        out.update(g for g in m.groups() if g)
    return out


# ---- the parser ----------------------------------------------------------------------------------

class Parser:
    def __init__(self, src: str, toks: list[Tok], types: set[str] | frozenset = frozenset()):
        self.src, self.toks, self.pos = src, toks, 0
        self.types = set(types) | COMMON_TYPES

    # -- helpers
    def peek(self, k: int = 0) -> Tok | None:
        i = self.pos + k
        return self.toks[i] if i < len(self.toks) else None

    def at(self, *texts: str, k: int = 0) -> bool:
        t = self.peek(k)
        return t is not None and t.text in texts and t.kind in ("op", "id")

    def take(self, text: str | None = None) -> Tok:
        t = self.peek()
        if t is None or (text is not None and t.text != text):
            raise ParseError(f"expected {text!r} at {t.start if t else 'end'}")
        self.pos += 1
        return t

    def end_of_prev(self) -> int:
        return self.toks[self.pos - 1].end

    def is_type_name(self, t: Tok | None) -> bool:
        return t is not None and t.kind == "id" and (t.text in TYPE_WORDS or t.text in self.types)

    # -- types (for declarations and casts)
    def scan_type(self, k: int) -> int | None:
        """Tokens of a type starting at offset k from pos: specifiers, an optional (qualified,
        templated) name, then `*`/`&`/`const`. Returns how many, or None when it is not a type."""
        n, named, words = k, False, False
        while True:
            t = self.peek(n)
            if t is None or t.kind != "id":
                break
            if t.text in QUALIFIERS or t.text in STORAGE:
                n += 1
            elif t.text in TYPE_WORDS:
                words, n = True, n + 1
            elif t.text in ("__typeof__", "typeof") and not named and not words and self.at("(", k=n + 1):
                depth, j = 0, n + 1
                while self.peek(j) is not None:
                    depth += (self.peek(j).text == "(") - (self.peek(j).text == ")")
                    j += 1
                    if depth == 0:
                        break
                named, n = True, j
            elif t.text in TAGS:
                nxt = self.peek(n + 1)
                if nxt is None or nxt.kind != "id":
                    return None
                named, n = True, n + 2
                n = self._qualified(n)
            elif not named and not words and t.text not in KEYWORDS:
                named, n = True, n + 1
                n = self._qualified(n)
                if n is None:
                    return None
            else:
                break
        if not (named or words):
            return None
        while self.peek(n) is not None and (self.peek(n).text in ("*", "&") or self.peek(n).text in QUALIFIERS):
            n += 1
        return n - k

    def _qualified(self, n: int) -> int | None:
        """After a name at n-1: `::name` parts and one `<…>` argument list."""
        while True:
            t = self.peek(n)
            if t is not None and t.text == "<":
                depth, j = 0, n
                while self.peek(j) is not None:
                    x = self.peek(j).text
                    if x in (";", "{", "}", "&&", "||"):
                        return None
                    depth += x == "<"
                    depth -= x == ">"
                    depth -= 2 * (x == ">>")
                    j += 1
                    if depth <= 0:
                        break
                n = j
                continue
            if t is not None and t.text == "::" and self.peek(n + 1) is not None and self.peek(n + 1).kind == "id":
                n += 2
                continue
            return n

    # -- expressions
    def expression(self, stop_colon: bool = False) -> Expr:
        e = self.assignment(stop_colon)
        while self.at(","):
            self.take()
            r = self.assignment(stop_colon)
            e = Expr("comma", e.start, r.end, ",", [e, r])
        return e

    def assignment(self, stop_colon: bool = False) -> Expr:
        left = self.ternary(stop_colon)
        t = self.peek()
        if t is not None and t.kind == "op" and t.text in ASSIGN:
            self.take()
            right = self.assignment(stop_colon)
            return Expr("assign", left.start, right.end, t.text, [left, right])
        return left

    def ternary(self, stop_colon: bool = False) -> Expr:
        cond = self.binary(4)
        if self.at("?") and not stop_colon:
            self.take()
            a = self.expression()
            self.take(":")
            b = self.assignment()
            return Expr("ternary", cond.start, b.end, "?", [cond, a, b])
        return cond

    def binary(self, level: int) -> Expr:
        left = self.unary()
        while True:
            t = self.peek()
            if t is None or t.kind != "op" or t.text not in BINARY or BINARY[t.text] < level:
                return left
            self.take()
            right = self.binary(BINARY[t.text] + 1)
            left = Expr("binary", left.start, right.end, t.text, [left, right])

    def cast_ahead(self) -> int | None:
        """At `(`: the number of type tokens inside when `( type ) operand` is a cast."""
        n = self.scan_type(1)
        if n is None or not self.at(")", k=1 + n):
            return None
        inner = self.toks[self.pos + 1:self.pos + 1 + n]
        after = self.peek(n + 2)
        if after is None or after.text in (")", "]", ";", ",", "?", ":", "=") or \
                (after.kind == "op" and after.text in BINARY and after.text not in ("-", "*", "&", "+")) or \
                after.text in ASSIGN:
            return None
        if any(x.text in TYPE_WORDS or x.text in TAGS or x.text in QUALIFIERS for x in inner) or \
                inner[-1].text in ("*", "&") or any(x.text == "<" for x in inner):
            return n
        names = [x for x in inner if x.kind == "id"]
        if len(names) == 1 and names[0].text in self.types:
            return n
        if all(x.kind == "id" or x.text == "::" for x in inner) and (after.kind in ("id", "num", "str", "chr")
                                                                       or after.text in ("(", "!", "~")):
            return n                     # `(Name) x`, `(Name)(…)`: nothing else reads that way
        return None

    def unary(self) -> Expr:
        t = self.peek()
        if t is None:
            raise ParseError("expression expected at end")
        if t.kind == "op" and t.text in ("-", "+", "!", "~", "*", "&", "++", "--", "::"):
            if t.text == "::" and self.peek(1) is not None and self.peek(1).text in ("new", "delete"):
                self.take()
                return self.unary()
            if t.text != "::":
                self.take()
                inner = self.unary()
                return Expr("unary", t.start, inner.end, t.text, [inner])
        if t.kind == "op" and t.text == "(":
            n = self.cast_ahead()
            if n is not None:
                self.take("(")
                ts = self.peek().start
                self.pos += n
                te = self.end_of_prev()
                self.take(")")
                inner = self.unary()
                return Expr("cast", t.start, inner.end, "()", [inner], type_span=(ts, te))
        if t.kind == "id" and t.text == "sizeof":
            self.take()
            if self.at("("):
                n = self.scan_type(1)
                if n is not None and self.at(")", k=1 + n):
                    self.take("(")
                    ts = self.peek().start
                    self.pos += n
                    te = self.end_of_prev()
                    close = self.take(")")
                    return Expr("sizeof", t.start, close.end, "sizeof", type_span=(ts, te))
            inner = self.unary()
            return Expr("sizeof", t.start, inner.end, "sizeof", [inner])
        if t.kind == "id" and t.text in ("new", "delete"):
            return self.new_delete()
        return self.postfix(self.primary())

    def new_delete(self) -> Expr:
        t = self.take()
        if t.text == "delete":
            if self.at("["):
                self.take("[")
                self.take("]")
            inner = self.unary()
            return Expr("unary", t.start, inner.end, "delete", [inner])
        kids = []
        if self.at("("):                              # placement arguments
            self.take("(")
            if not self.at(")"):
                kids.append(self.expression())
            self.take(")")
        n = self.scan_type(0)
        if n is None:
            raise ParseError(f"type expected after new at {t.start}")
        self.pos += n
        end = self.end_of_prev()
        if self.at("["):
            self.take("[")
            kids.append(self.expression())
            end = self.take("]").end
        if self.at("("):
            self.take("(")
            if not self.at(")"):
                kids.append(self.expression())
            end = self.take(")").end
        return Expr("unary", t.start, end, "new", kids)

    def primary(self) -> Expr:
        t = self.peek()
        if t is None:
            raise ParseError("expression expected at end")
        if t.kind == "num":
            self.take()
            return Expr("num", t.start, t.end)
        if t.kind in ("str", "chr"):
            self.take()
            end = t.end
            while t.kind == "str" and self.peek() is not None and self.peek().kind == "str":
                end = self.take().end                   # adjacent literals concatenate
            return Expr(t.kind, t.start, end)
        if t.kind == "op" and t.text == "(":
            self.take()
            inner = self.expression()
            close = self.take(")")
            return Expr("paren", t.start, close.end, "()", [inner])
        if t.kind == "id" and t.text in CASTS and self.at("<", k=1):
            self.take()
            self.take("<")
            n = self.scan_type(0)
            if n is None:
                raise ParseError(f"type expected in {t.text} at {t.start}")
            ts = self.peek().start
            self.pos += n
            te = self.end_of_prev()
            self.take(">")
            self.take("(")
            inner = self.expression()
            close = self.take(")")
            return Expr("cast", t.start, close.end, t.text, [inner], type_span=(ts, te))
        if (t.kind == "id" and t.text not in KEYWORDS - {"this", "true", "false", "operator"}) or \
                (t.kind == "op" and t.text == "::"):
            start = t.start
            if t.text == "::":
                self.take()
            self.take()
            if self.peek(-1).text == "operator":
                self.pos += 1                         # `operator=` and the like, as a name
            while self.at("::") and self.peek(1) is not None and self.peek(1).kind == "id":
                self.pos += 2
                if self.peek(-1).text == "operator":
                    self.pos += 1
            if self.at("<") and self.peek(-1).text in self.types:
                n = self._qualified(self.pos)
                if n is not None:
                    self.pos = n
            return Expr("name", start, self.end_of_prev(), name=self.src[start:self.end_of_prev()])
        raise ParseError(f"unexpected {t.text!r} at {t.start}")

    def postfix(self, e: Expr) -> Expr:
        while True:
            t = self.peek()
            if t is None or t.kind != "op":
                return e
            if t.text == "(":
                self.take()
                args = []
                if not self.at(")"):
                    args.append(self.argument())
                    while self.at(","):
                        self.take()
                        args.append(self.argument())
                close = self.take(")")
                e = Expr("call", e.start, close.end, "()", [e] + args)
            elif t.text == "[":
                self.take()
                sub = self.expression()
                close = self.take("]")
                e = Expr("index", e.start, close.end, "[]", [e, sub])
            elif t.text in (".", "->"):
                self.take()
                if self.at("~"):
                    self.take()
                m = self.take()
                if m.kind != "id":
                    raise ParseError(f"member name expected at {m.start}")
                while self.at("::") and self.peek(1) is not None and self.peek(1).kind == "id":
                    self.pos += 2
                e = Expr("member", e.start, self.end_of_prev(), t.text, [e], name=self.src[m.start:self.end_of_prev()])
            elif t.text in ("++", "--"):
                self.take()
                e = Expr("postfix", e.start, t.end, t.text, [e])
            else:
                return e

    # -- declarations
    def decl_ahead(self) -> int | None:
        """Type tokens when the statement at pos is a declaration."""
        t = self.peek()
        if t is None or t.kind != "id" or t.text in KEYWORDS:
            return None
        n = self.scan_type(0)
        if n is None:
            return None
        nxt = self.peek(n)
        if nxt is None:
            return None
        words = [x for x in self.toks[self.pos:self.pos + n] if x.kind == "id"]
        sure = any(w.text in TYPE_WORDS or w.text in QUALIFIERS or w.text in STORAGE or w.text in TAGS
                   or w.text in self.types for w in words)
        if nxt.kind == "id" and nxt.text not in KEYWORDS:
            after = self.peek(n + 1)
            if after is not None and after.text in ("=", ";", ",", "[", "(", ":", "__attribute__"):
                return n if (sure or after.text != ":") else None
            return None
        if sure and nxt.text in ("*", "&"):
            return n
        return None

    def declaration(self, n: int, terminators: tuple[str, ...] = (";",)) -> Decl:
        ts = self.peek().start
        # the base type ends before the first `*`/`&` (they belong to the first declarator)
        k, depth = 0, 0
        while k < n and (depth or self.peek(k).text not in ("*", "&")):
            depth += (self.peek(k).text in ("(", "<")) - (self.peek(k).text in (")", ">"))
            k += 1
        self.pos += k
        d = Decl(ts, self.end_of_prev())
        while True:
            d.declarators.append(self.declarator())
            if self.at(","):
                self.take()
                continue
            break
        t = self.peek()
        if t is None or t.text not in terminators:
            raise ParseError(f"declaration not terminated at {t.start if t else 'end'}")
        return d

    def declarator(self) -> Declarator:
        start = self.peek().start
        ptr = []
        while self.at("*", "&", "const", "volatile"):
            ptr.append(self.take().text)
        name = self.take()
        if name.kind != "id":
            raise ParseError(f"declarator name expected at {name.start}")
        dcl = Declarator(name.text, name.start, name.end, start, name.end, "".join(ptr))
        while self.at("["):
            a = self.take("[").start
            if not self.at("]"):
                self.expression()
            dcl.array += self.src[a:self.take("]").end]
            dcl.end = self.end_of_prev()
        while self.at("__attribute__", "__declspec"):
            self.take()
            self.skip_parens()
            dcl.end = self.end_of_prev()
        if self.at("="):
            self.take()
            dcl.init_kind = "="
            if self.at("{"):
                dcl.init = self.braced()
                dcl.init_kind = "{}"
            else:
                dcl.init = self.assignment()
            dcl.end = dcl.init.end
        elif self.at("("):
            self.take()
            dcl.init_kind = "()"
            dcl.init = self.expression() if not self.at(")") else None
            dcl.end = self.take(")").end
        return dcl

    def skip_parens(self) -> None:
        self.take("(")
        depth = 1
        while depth:
            t = self.take()
            depth += (t.text == "(") - (t.text == ")")

    def argument(self) -> Expr:
        """A call argument; a type stands for itself (`va_arg(args, char*)`)."""
        n = self.scan_type(0)
        if n is not None and self.at(",", ")", k=n) and (n > 1 or self.is_type_name(self.peek())):
            start = self.peek().start
            self.pos += n
            return Expr("type", start, self.end_of_prev(), type_span=(start, self.end_of_prev()))
        return self.assignment()

    def braced(self) -> Expr:
        open_ = self.take("{")
        depth = 1
        while depth:
            t = self.take()
            depth += (t.text == "{") - (t.text == "}")
        return Expr("init", open_.start, self.end_of_prev())

    # -- statements
    def block(self) -> Stmt:
        open_ = self.take("{")
        kids = []
        while not self.at("}"):
            if self.peek() is None:
                raise ParseError(f"unclosed block at {open_.start}")
            kids.append(self.statement())
        close = self.take("}")
        return Stmt("block", open_.start, close.end, kids=kids)

    def statement(self) -> Stmt:
        t = self.peek()
        if t is None:
            raise ParseError("statement expected at end")
        mark = self.pos
        try:
            return self._statement(t)
        except ParseError:
            self.pos = mark
            return self.skip_other(t)

    def skip_other(self, t: Tok) -> Stmt:
        """An unreadable statement: everything up to the `;` at depth 0 (or a balanced block)."""
        depth = 0
        while self.peek() is not None:
            x = self.take()
            if x.text in ("(", "[", "{"):
                depth += 1
            elif x.text in (")", "]", "}"):
                depth -= 1
                if depth < 0:
                    self.pos -= 1
                    break
                if depth == 0 and x.text == "}" and not self.at(";", ","):
                    break
            elif x.text == ";" and depth == 0:
                break
        if self.pos == 0 or self.toks[self.pos - 1].start < t.start:
            raise ParseError(f"cannot read the statement at {t.start}")
        return Stmt("other", t.start, self.end_of_prev())

    def paren_head(self) -> tuple[Expr | None, tuple[int, int]]:
        open_ = self.take("(")
        e = self.expression() if not self.at(")") else None
        close = self.take(")")
        return e, (open_.end, close.start)

    def _statement(self, t: Tok) -> Stmt:
        if t.kind == "pp":
            self.take()
            return Stmt("pp", t.start, t.end)
        if t.text == "{" and t.kind == "op":
            return self.block()
        if t.text == ";" and t.kind == "op":
            self.take()
            return Stmt("empty", t.start, t.end)
        if t.kind == "id":
            kw = t.text
            if kw == "if":
                self.take()
                e, head = self.paren_head()
                then = self.statement()
                s = Stmt("if", t.start, then.end, e, head, [then])
                if self.at("else"):
                    s.else_at = self.take().start
                    other = self.statement()
                    s.kids.append(other)
                    s.end = other.end
                return s
            if kw in ("while", "switch"):
                self.take()
                e, head = self.paren_head()
                body = self.statement()
                return Stmt(kw, t.start, body.end, e, head, [body])
            if kw == "do":
                self.take()
                body = self.statement()
                self.take("while")
                e, head = self.paren_head()
                return Stmt("do", t.start, self.take(";").end, e, head, [body])
            if kw == "for":
                return self.for_statement(t)
            if kw == "return":
                self.take()
                e = None if self.at(";") else self.expression()
                return Stmt("return", t.start, self.take(";").end, e)
            if kw in ("break", "continue"):
                self.take()
                return Stmt(kw, t.start, self.take(";").end)
            if kw == "goto":
                self.take()
                lab = self.take().text
                return Stmt("goto", t.start, self.take(";").end, label=lab)
            if kw == "case":
                self.take()
                e = self.expression(stop_colon=True)
                if self.at("..."):
                    self.take()
                    self.expression(stop_colon=True)
                return Stmt("case", t.start, self.take(":").end, e)
            if kw == "default" and self.at(":", k=1):
                self.take()
                return Stmt("default", t.start, self.take(":").end)
            if kw in ("asm", "__asm"):
                return self.skip_other(t)
            if kw not in KEYWORDS and self.at(":", k=1):
                self.take()
                return Stmt("label", t.start, self.take(":").end, label=kw)
            n = self.decl_ahead()
            if n is not None:
                d = self.declaration(n)
                return Stmt("decl", t.start, self.take(";").end, decl=d)
        e = self.expression()
        return Stmt("expr", t.start, self.take(";").end, e)

    def for_statement(self, t: Tok) -> Stmt:
        self.take("for")
        open_ = self.take("(")
        first = self.peek()
        if self.at(";"):
            init = Stmt("empty", first.start, self.take(";").end)
        else:
            n = self.decl_ahead()
            if n is not None:
                d = self.declaration(n)
                init = Stmt("decl", first.start, self.take(";").end, decl=d)
            else:
                e = self.expression()
                init = Stmt("expr", first.start, self.take(";").end, e)
        cond = None if self.at(";") else self.expression()
        self.take(";")
        step = None if self.at(")") else self.expression()
        close = self.take(")")
        body = self.statement()
        return Stmt("for", t.start, body.end, None, (open_.end, close.start), [body], init=init, cond=cond, step=step)


def parse_body(src: str, open_brace: int, types: set[str] | frozenset = frozenset()) -> Stmt:
    """The block that starts with the `{` at src[open_brace]."""
    close = matching_brace(src, open_brace)
    if close is None:
        raise ParseError(f"unbalanced braces from {open_brace}")
    p = Parser(src, tokens(src, open_brace, close + 1), types)
    return p.block()


def parse_expression(src: str, types: set[str] | frozenset = frozenset(), base: int = 0) -> Expr:
    """One expression (offsets relative to src, plus base)."""
    p = Parser(" " * base + src, tokens(" " * base + src, base), types)
    e = p.expression()
    if p.peek() is not None:
        raise ParseError(f"trailing {p.peek().text!r}")
    return e


def matching_brace(src: str, open_brace: int) -> int | None:
    for hi in (_block_end(src, open_brace), len(src)):
        depth = 0
        for t in tokens(src, open_brace, hi):
            if t.text == "{" and t.kind == "op":
                depth += 1
            elif t.text == "}" and t.kind == "op":
                depth -= 1
                if depth == 0:
                    return t.start
    return None


def _block_end(src: str, open_brace: int) -> int:
    """A cheap upper bound for the end of a block: the next `}` in column 0 (end of file if none)."""
    k = src.find("\n}", open_brace)
    return len(src) if k < 0 else k + 2


# ---- locating functions --------------------------------------------------------------------------

@dataclass
class FunctionSpan:
    head_start: int             # first token of the declaration (after the previous item)
    open: int                   # its `{`
    close: int                  # its `}`
    name: str                   # the declared name as written (`Class::fn`, `operator|`)


def functions(src: str) -> list[FunctionSpan]:
    """Function definitions in the file: any `{` block whose head ends with `)` (optionally
    `const` or a constructor's initializer list), at top level or inside a namespace, extern "C",
    class or struct body."""
    toks = tokens(src)
    out: list[FunctionSpan] = []
    stack: list[bool] = []          # open braces: may a function definition sit inside it?
    head = 0                        # token index where the current depth's item started
    depth_head: list[int] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.kind == "pp":
            head = i + 1
            i += 1
            continue
        if t.text == "{" and t.kind == "op":
            h = toks[head:i]
            name = _function_head(h)
            if name is not None and all(stack):
                depth, j = 0, i
                while j < len(toks):
                    if toks[j].kind == "op" and toks[j].text in "{}":
                        depth += 1 if toks[j].text == "{" else -1
                        if depth == 0:
                            break
                    j += 1
                if j >= len(toks):
                    break
                out.append(FunctionSpan(toks[head].start, t.start, toks[j].start, name))
                i = head = j + 1
                continue
            container = bool(h) and (h[0].text in ("namespace", "extern", "struct", "class", "union") or
                                     any(x.text in ("struct", "class", "union") for x in h[:3])) and \
                not any(x.text == "=" for x in h)
            stack.append(container)
            depth_head.append(head)
            head = i + 1
        elif t.text == "}" and t.kind == "op":
            if stack:
                stack.pop()
                depth_head.pop()
            head = i + 1
        elif t.text == ";" and t.kind == "op":
            head = i + 1
        elif t.text == ":" and t.kind == "op" and i > 0 and toks[i - 1].text in ("public", "private", "protected"):
            head = i + 1
        i += 1
    return out


def _function_head(h: list[Tok]) -> str | None:
    if not h or any(x.text == "=" and x.kind == "op" for x in h[:_paren_index(h)]):
        return None
    k = len(h) - 1
    while k >= 0 and h[k].text in ("const", "volatile", "throw"):
        k -= 1
    # a constructor's initializer list: `) : a(x), b(y)` — find the `)` before the `:`
    depth = 0
    colon = None
    for j, x in enumerate(h):
        depth += x.text in ("(", "[") and x.kind == "op"
        depth -= x.text in (")", "]") and x.kind == "op"
        if depth == 0 and x.text == ":" and x.kind == "op" and j > 0 and h[j - 1].text == ")":
            colon = j
            break
    if colon is not None:
        k = colon - 1
    if k < 0 or h[k].text != ")":
        return None
    depth, j = 0, k
    while j >= 0:
        depth += h[j].text == ")"
        depth -= h[j].text == "("
        if depth == 0:
            break
        j -= 1
    if j <= 0:
        return None
    name_end = j
    n = j - 1
    if h[n].kind != "id" and not (h[n].kind == "op" and n > 0):
        return None
    while n > 0 and (h[n - 1].text in ("::", "~", "operator") or (h[n].kind == "op" and h[n - 1].text == "operator")
                     or (h[n - 1].kind == "id" and h[n].text == "::")):
        n -= 1
    if h[n].text in KEYWORDS - {"operator"}:
        return None
    return "".join(x.text for x in h[n:name_end])


def _paren_index(h: list[Tok]) -> int:
    for j, x in enumerate(h):
        if x.text == "(":
            return j
    return len(h)


def function_at_line(src: str, line: int) -> FunctionSpan | None:
    """The definition containing the 0-based line, or else the first one that starts on it."""
    starts = line_starts(src)
    off = starts[line] if line < len(starts) else len(src)
    end = starts[line + 1] if line + 1 < len(starts) else len(src)
    best = None
    for f in functions(src):
        if f.head_start <= off <= f.close or off <= f.head_start < end or off <= f.open < end:
            if best is None or f.open > best.open:      # the innermost (a member defined in a class body)
                best = f
    return best


def line_starts(src: str) -> list[int]:
    out = [0]
    for i, c in enumerate(src):
        if c == "\n":
            out.append(i + 1)
    return out


def line_of(src: str, offset: int) -> int:
    """0-based line of an offset."""
    return src.count("\n", 0, offset)


# ---- walking and editing -------------------------------------------------------------------------

def statements(s: Stmt):
    """Every statement inside s (s included), in source order."""
    yield s
    if s.init is not None:
        yield from statements(s.init)
    for k in s.kids:
        yield from statements(k)


def expressions(e: Expr | None):
    """Every expression node inside e (e included), outermost first."""
    if e is None:
        return
    yield e
    for k in e.kids:
        yield from expressions(k)


def statement_exprs(s: Stmt):
    """The expressions a statement itself holds (not those of the statements inside it)."""
    for e in (s.expr, s.cond, s.step):
        if e is not None:
            yield e
    if s.decl is not None:
        for d in s.decl.declarators:
            if d.init is not None:
                yield d.init


def parents(root: Stmt) -> dict[int, tuple[Stmt, int]]:
    """id(statement) -> (the block or statement holding it, its index there)."""
    out = {}
    for s in statements(root):
        for i, k in enumerate(s.kids):
            out[id(k)] = (s, i)
    return out


def names(e: Expr | None) -> list[str]:
    """Plain names an expression reads or writes (not member names)."""
    return [x.name for x in expressions(e) if x.kind == "name"]


def apply_edits(src: str, edits: list[tuple[int, int, str]]) -> str:
    """Replace each (start, end) span with its text; spans must not overlap."""
    out, last = [], 0
    for a, b, new in sorted(edits, key=lambda x: (x[0], x[1])):
        if a < last:
            raise ValueError(f"overlapping edits at {a}")
        out.append(src[last:a])
        out.append(new)
        last = b
    out.append(src[last:])
    return "".join(out)


def indent_at(src: str, offset: int) -> str:
    start = src.rfind("\n", 0, offset) + 1
    m = re.match(r"[ \t]*", src[start:])
    return m.group(0) if m else ""
