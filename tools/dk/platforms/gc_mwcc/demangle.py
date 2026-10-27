"""CodeWarrior (Metrowerks) C++ symbol demangling, for the forms function calls use.

  name__F<params>                 free function          fn_X__FPCcPv
  name__<qual>[C]F<params>        member (C = const)     fn_X__Q24EAGL10SymbolPoolFPCc
  __ct / __dt / __nw / __dl ...   special members and operators
<qual> is <len><id> or Q<n><len><id>...; parameters use v i Ui l Ul s Us c Sc Uc b f d w,
P (pointer), R (reference), C (const), V (volatile), class names, F<params>_<ret>
(function), A<n>_<type> (array). Templates and anything else return None: callers refuse
what they cannot reproduce rather than guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_BUILTIN = {"v": "void", "i": "int", "l": "long", "s": "short", "c": "char", "b": "bool", "f": "float",
            "d": "double", "w": "wchar_t", "e": "..."}
_UNSIGNED = {"i": "unsigned int", "l": "unsigned long", "s": "unsigned short", "c": "unsigned char"}
SPECIAL = {"__ct": "ctor", "__dt": "dtor", "__nw": "operator new", "__dl": "operator delete",
           "__nwa": "operator new[]", "__dla": "operator delete[]", "__as": "operator="}
_OPERATORS = {"pl": "+", "mi": "-", "ml": "*", "dv": "/", "md": "%", "er": "^", "ad": "&", "or": "|", "co": "~",
              "nt": "!", "lt": "<", "gt": ">", "le": "<=", "ge": ">=", "eq": "==", "ne": "!=", "aa": "&&", "oo": "||",
              "ls": "<<", "rs": ">>", "apl": "+=", "ami": "-=", "amu": "*=", "adv": "/=", "amd": "%=", "aer": "^=",
              "aad": "&=", "aor": "|=", "als": "<<=", "ars": ">>=", "pp": "++", "mm": "--", "cm": ",", "rf": "->",
              "cl": "()", "vc": "[]"}
SPECIAL.update({f"__{code}": f"operator{op}" for code, op in _OPERATORS.items()})
OPERATORS = frozenset(f"operator{op}" for op in list(_OPERATORS.values()) + ["="])     # (plain member functions)


@dataclass(frozen=True)
class Type:
    text: str                 # C++ spelling, e.g. "const char*"
    reg: str                  # "gpr" | "fpr" | "void" | "none" (passed in memory / unsupported)
    classes: tuple            # qualified class names it mentions, e.g. (("EAGL", "SymbolPool"),)


@dataclass(frozen=True)
class Function:
    quals: tuple              # enclosing class path, () for a free function
    name: str                 # method name, or a SPECIAL value
    params: tuple             # Type...
    const: bool

    @property
    def special(self) -> str | None:
        return self.name if self.name in SPECIAL.values() else None


class _Parser:
    def __init__(self, text: str):
        self.s, self.i = text, 0

    def peek(self) -> str:
        return self.s[self.i] if self.i < len(self.s) else ""

    def take(self) -> str:
        ch = self.peek()
        self.i += 1
        return ch

    def number(self) -> int:
        m = re.match(r"\d+", self.s[self.i:])
        if not m:
            raise ValueError("expected a number")
        self.i += len(m[0])
        return int(m[0])

    def ident(self) -> str:
        n = self.number()
        name = self.s[self.i:self.i + n]
        if len(name) != n or not re.fullmatch(r"[A-Za-z_]\w*", name):
            raise ValueError("bad identifier")      # templates (Foo<...>) end up here
        self.i += n
        return name

    def qual(self) -> tuple:
        if self.peek() == "Q":
            self.take()
            count = int(self.take())
            return tuple(self.ident() for _ in range(count))
        return (self.ident(),)

    def type(self) -> Type:
        ch = self.peek()
        const = False
        if ch in "CV":
            self.take()
            const = ch == "C"
            inner = self.type()
            if inner.text.endswith("*") or inner.text.endswith("&"):
                return Type(f"{inner.text} const" if const else inner.text, inner.reg, inner.classes)
            return Type(f"const {inner.text}" if const else inner.text, inner.reg, inner.classes)
        if ch == "P" or ch == "R":
            self.take()
            if self.peek() == "F":
                fn = self.function_type()
                return Type(fn.replace("(*)", "(*)" if ch == "P" else "(&)"), "gpr", ())
            inner = self.type()
            return Type(inner.text + ("*" if ch == "P" else "&"), "gpr", inner.classes)
        if ch == "U" or ch == "S":
            self.take()
            base = self.take()
            if ch == "U" and base in _UNSIGNED:
                return Type(_UNSIGNED[base], "gpr", ())
            if ch == "S" and base == "c":
                return Type("signed char", "gpr", ())
            raise ValueError("unsupported sign prefix")
        if ch == "A":
            raise ValueError("array parameter")
        if ch == "F":
            raise ValueError("function parameter")
        if ch.isdigit() or ch == "Q":
            q = self.qual()
            return Type("::".join(q), "none", (q,))      # a class by value: passed in memory
        if ch in _BUILTIN:
            self.take()
            reg = {"v": "void", "f": "fpr", "d": "fpr", "e": "none"}.get(ch, "gpr")
            return Type(_BUILTIN[ch], reg, ())
        raise ValueError(f"unsupported type code {ch!r}")

    def params(self, stop: str = "") -> tuple:
        out = []
        while self.peek() and self.peek() != stop:
            out.append(self.type())
        if out == [Type("void", "void", ())]:
            return ()
        return tuple(out)

    def function_type(self) -> str:
        self.take()                                   # F
        params = self.params("_")
        if self.take() != "_":
            raise ValueError("function type without return")
        ret = self.type()
        return f"{ret.text} (*)({', '.join(p.text for p in params)})"


def demangle(symbol: str) -> Function | None:
    """The parts of a mangled function name, or None if this is not one we can reproduce."""
    m = re.match(r"^(?P<name>__[a-z]+|[A-Za-z_]\w*?)__(?P<rest>(?:\d|Q\d|F|C).*)$", symbol)
    if not m:
        return None
    name, rest = m["name"], m["rest"]
    if name.startswith("__") and name not in SPECIAL:
        return None
    p = _Parser(rest)
    try:
        quals = () if p.peek() in "FC" else p.qual()
        const = False
        if p.peek() == "C":
            p.take()
            const = True
        if p.take() != "F":
            return None
        params = p.params()
        if p.i != len(p.s):
            return None
    except (ValueError, IndexError):
        return None
    return Function(quals, SPECIAL.get(name, name), params, const)
