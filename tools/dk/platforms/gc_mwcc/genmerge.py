"""dk gen --write: merge a unit's exact drafts into ONE compilable source.

Each exact draft comes as a self-contained TU (its placeholder structs, externs and the
function under its own name). Merging them:

  * a struct several drafts define is merged field by field: every field sits at the offset
    its name gives (m<OFF>), gaps become `char _pad<OFF>[N]`, virtual slots and member
    declarations are united. Two drafts that disagree (one offset, two types; overlapping
    fields; two declarations of one symbol) cannot both go in: the later one is left out.
  * a struct with a base (`struct C : B {`, a class whose inherited methods a draft calls on
    this) keeps only its methods: every base sits at offset 0, so its fields, vtable pointer and
    virtual slots belong to the root of its chain, where drafts that saw C alone are merged too.
    Except a derived struct whose members start after its base (`char _pad<sizeof B>`, a
    destructor's class model: its member objects must stay its own): it keeps its fields, its
    `virtual ~C();` is an override (not a new slot), and its base must end before them.
  * the merged TU is compiled with the unit's command and every function is compared with
    the original again. A draft that stops being exact after merging is left out (greedy, in
    address order, so earlier drafts win), and the report says which and why.

Only the generator's own output format is parsed (top-level items start in column 0).

dk gen --merge (Paste) puts drafts into a source that already has code: a struct the source
itself defines in the generator's format is merged in place; a struct, forward declaration
or extern the source or its headers (the preprocessed text) already define is dropped, so the
draft uses theirs; everything else goes in one new block before the first function, and each
function goes before the source's first function at a higher address.
"""

from __future__ import annotations

import copy
import re
from collections import defaultdict
from dataclasses import dataclass, field

from . import gen

_FWD = re.compile(r"^struct ([\w:]+);$")
_OPEN = re.compile(r"^struct ([\w:]+)(?: : ([\w:]+))? \{$")
_ONE_LINE = re.compile(r"^struct ([\w:]+)( : [^{]+)? \{ ?(.*?) ?\};$")
_PAD = re.compile(r"^char _pad([0-9A-F]+)\[0x([0-9A-F]+)\];$")
_FIELD = re.compile(r"^(?P<type>[^()]+?) (?P<name>m(?P<off>[0-9A-F]+))(?:\[(?P<n>\d+)\])?;$")
_EXTERN_FN = re.compile(r'^extern "C" .*?(\w+)\(')
_EXTERN_VAR = re.compile(r"^extern .*?(\w+)(?:\[\])?;$")


class Conflict(Exception):
    pass


class Missing(Conflict):
    """A class the source or a header defines lacks members a draft needs."""
    def __init__(self, cls: str, fields: list, methods: list, text: str):
        super().__init__(text)
        self.cls = cls                  # the class (through which the fields are reached)
        self.fields = fields            # (offset, size, member text) the draft reads or writes
        self.methods = methods          # member declarations it calls or defines


@dataclass
class Layout:
    """A placeholder struct as offsets: what the generator writes, merged across drafts."""
    fields: dict = field(default_factory=dict)      # offset -> (member text, size)
    vptr: int | None = None
    virtuals: list = field(default_factory=list)    # in slot order
    methods: list = field(default_factory=list)     # member declarations (no offset)
    size: int = 0                                   # bytes up to the trailing padding's end
    placeholder: bool = False                       # `{ char _unknown; }`: layout unknown
    base: str | None = None                         # `struct C : base` (at offset 0)
    start: int = 0                                  # > 0: C's own members start here (after its base)

    def end(self) -> int:
        ends = [off + size for off, (_, size) in self.fields.items()] + ([self.vptr + 4] if self.vptr is not None else [])
        return max([self.size] + ends)

    def sizeof(self) -> int:
        """Its size as a by-value member (an empty struct still takes a byte)."""
        return max(self.end(), 1)


def _width(ctype: str, sizes: dict) -> int | None:
    if ctype.endswith("*"):
        return 4
    return gen.SIZES.get(ctype, sizes.get(ctype))


def parse_layout(members: list[str], sizes: dict, start: int | None = None) -> Layout | None:
    """The generator's struct body as a Layout, or None for anything else (bases, inline
    bodies, members it does not write). `start` (a derived struct): the size of its base,
    where its own members begin when it lists them from there; its virtual destructor is an
    override, not a slot of its own."""
    out, cur = Layout(), 0
    if start and members and (pad := _PAD.match(members[0])) and int(pad[1], 16) == start or \
            start and members and (f := _FIELD.match(members[0])) and int(f["off"], 16) == start:
        out.start = cur = start
    if members[:1] == ["char _unknown;"] and all("(" in m and "{" not in m and not m.startswith("virtual ")
                                                  for m in members[1:]):
        return Layout(size=1, placeholder=True, methods=members[1:])
    for m in members:
        if pad := _PAD.match(m):
            if int(pad[1], 16) != cur:
                return None
            cur = int(pad[1], 16) + int(pad[2], 16)
            out.size = cur
        elif m.startswith("virtual ") and start is not None and re.fullmatch(r"virtual ~\w+\(\)(?:;| \{\})", m):
            out.methods.append(m)                   # overrides the base's
        elif m.startswith("virtual "):
            if out.vptr is None:
                out.vptr, cur = cur, cur + 4
            out.virtuals.append(m)
        elif "(" in m:
            if "{" in m and not re.fullmatch(r"(?:virtual )?~\w+\(\) \{\}", m):     # (an empty inline destructor)
                return None
            out.methods.append(m)
        elif f := _FIELD.match(m):
            off = int(f["off"], 16)
            width = _width(f["type"].strip(), sizes)
            if width is None or off != cur:
                return None
            size = width * int(f["n"] or 1)
            out.fields[off] = (m, size)
            cur = off + size
        else:
            return None
    out.size = max(out.size, cur)
    return out


def merge_layouts(a: Layout, b: Layout, name: str) -> Layout:
    if a.placeholder or b.placeholder:
        out = copy.deepcopy(b if a.placeholder else a)
        out.methods += [m for m in (a if a.placeholder else b).methods if m not in out.methods]
        return out
    out = copy.deepcopy(a)
    if a.base and b.base and a.base != b.base:
        raise Conflict(f"{name}: derived from {a.base} and from {b.base}")
    out.base = a.base or b.base
    for off, (text, size) in b.fields.items():
        if off in out.fields and out.fields[off][0] != text:
            raise Conflict(f"{name} +0x{off:X}: {out.fields[off][0]} vs {text}")
        out.fields[off] = (text, size)
    if b.vptr is not None:
        if out.vptr is not None and out.vptr != b.vptr:
            raise Conflict(f"{name}: vtable pointer at +0x{out.vptr:X} and +0x{b.vptr:X}")
        out.vptr = b.vptr
    for i, v in enumerate(b.virtuals):
        if i >= len(out.virtuals):
            out.virtuals.append(v)
        elif out.virtuals[i] != v:
            default = f"virtual void vf{i}();"
            if out.virtuals[i] == default:
                out.virtuals[i] = v
            elif v != default:
                raise Conflict(f"{name} slot {i}: {out.virtuals[i]} vs {v}")
    dtor = lambda m: re.match(r"(?:virtual )?(~\w+)\(", m)
    for m in b.methods:
        if m not in out.methods and dtor(m) and any(dtor(o) and dtor(o)[1] == dtor(m)[1] for o in out.methods):
            raise Conflict(f"{name}: {next(o for o in out.methods if dtor(o) and dtor(o)[1] == dtor(m)[1])} vs {m}")
    out.methods += [m for m in b.methods if m not in out.methods]
    out.size = max(out.size, b.size)
    out.start = max(a.start, b.start)
    if out.start and (any(off < out.start for off in out.fields) or out.vptr is not None):
        raise Conflict(f"{name}: members below its own start +0x{out.start:X} (they belong to a base)")
    spans = sorted([(off, size) for off, (_, size) in out.fields.items()] +
                   ([(out.vptr, 4)] if out.vptr is not None else []))
    for (o1, s1), (o2, _) in zip(spans, spans[1:]):
        if o1 + s1 > o2:
            raise Conflict(f"{name}: fields at +0x{o1:X} and +0x{o2:X} overlap")
    return out


def render_layout(l: Layout) -> list[str]:
    if l.placeholder:
        return ["char _unknown;"] + l.methods
    entries = {off: [text] for off, (text, _) in l.fields.items()}
    sizes = {off: size for off, (_, size) in l.fields.items()}
    if l.vptr is not None:
        entries[l.vptr], sizes[l.vptr] = list(l.virtuals), 4
    out, cur = [], l.start
    for off in sorted(entries):
        if off > cur:
            out.append(f"char _pad{cur:X}[0x{off - cur:X}];")
        out += entries[off]
        cur = off + sizes[off]
    if l.size > cur:
        out.append(f"char _pad{cur:X}[0x{l.size - cur:X}];")
    return out + l.methods


@dataclass
class Unit:
    """The merged TU so far."""
    forward: dict = field(default_factory=dict)     # struct name -> None (ordered set)
    skeletons: dict = field(default_factory=dict)   # outer -> [inner names]
    structs: dict = field(default_factory=dict)     # name -> Layout | str (opaque, verbatim)
    decls: dict = field(default_factory=dict)       # declared symbol (or the text) -> text
    functions: list = field(default_factory=list)   # (address, text)

    def add(self, source: str, address: int) -> "Unit":
        """A copy with one draft's TU merged in (Conflict if they disagree)."""
        out = copy.deepcopy(self)
        sizes = {n: s.sizeof() for n, s in out.structs.items() if isinstance(s, Layout)}
        lines = source.splitlines()
        i, fns = 0, []
        while i < len(lines):
            line = lines[i]
            if not line.strip() or line.startswith("//"):
                i += 1
                continue
            if m := _FWD.match(line):
                out.forward.setdefault(m[1])
            elif m := _ONE_LINE.match(line):
                body = [x.strip() + ";" for x in m[3].split(";") if x.strip()] if "{" not in m[3] else None
                if body and not m[2] and all(_FWD.match(x) for x in body):
                    kids = out.skeletons.setdefault(m[1], [])
                    kids += [k for k in (_FWD.match(x)[1] for x in body) if k not in kids]
                else:
                    out._struct(m[1], line if m[2] or body is None else body, sizes)
            elif m := _OPEN.match(line):
                j = lines.index("};", i)
                out._struct(m[1], [x.strip() for x in lines[i + 1:j]], sizes, m[2])
                i = j
            elif line.endswith(";"):
                key = (_EXTERN_FN.match(line) or _EXTERN_VAR.match(line) or [None, line])[1]
                if out.decls.setdefault(key, line) != line:
                    raise Conflict(f"{key} declared twice: {out.decls[key]} / {line}")
            elif line.endswith("{"):
                j = lines.index("}", i)
                fns.append("\n".join(lines[i:j + 1]))
                i = j
            elif line.endswith("}"):
                fns.append(line)
            else:
                raise Conflict(f"cannot merge the line {line!r}")
            i += 1
        out.functions += [(address, f) for f in fns]
        out._sink_to_bases()
        return out

    def _struct(self, name: str, body, sizes: dict, base: str | None = None) -> None:
        if isinstance(body, str):                   # opaque: must be identical
            if self.structs.setdefault(name, body) != body:
                raise Conflict(f"struct {name} defined two ways")
            return
        layout = parse_layout(body, sizes, sizes.get(base, 0) if base else None)
        old = self.structs.get(name)
        if layout is None or isinstance(old, str):
            head = f"struct {name}" + (f" : {base}" if base else "")
            text = f"{head} {{\n" + "".join(f"    {m}\n" for m in body) + "};"
            if self.structs.setdefault(name, text) != text:
                raise Conflict(f"struct {name} defined two ways")
            return
        layout.base = base
        self.structs[name] = merge_layouts(old, layout, name) if old is not None else layout
        sizes[name] = self.structs[name].sizeof()

    def _sink_to_bases(self) -> None:
        """The fields, vtable pointer and virtual slots of a derived struct move to the root of
        its chain (all bases at offset 0, so every offset stays)."""
        for name, s in self.structs.items():
            if isinstance(s, Layout) and s.start:     # its own members: its base must end before them
                b = self.structs.get(s.base)
                end = b.sizeof() if isinstance(b, Layout) else s.start
                if end > s.start:
                    if any(off < end for off in s.fields):
                        raise Conflict(f"struct {name}: its base {s.base} grew over its members")
                    s.start = end
                    s.size = max(s.size, end)
                continue
            if not isinstance(s, Layout) or not s.base or not (s.fields or s.vptr is not None or s.virtuals or s.size):
                continue
            root, seen = name, {name}
            while isinstance(self.structs.get(root), Layout) and self.structs[root].base:
                root = self.structs[root].base
                if root in seen:
                    raise Conflict(f"struct {name}: its bases form a cycle")
                seen.add(root)
            if not isinstance(self.structs.get(root), Layout):
                raise Conflict(f"struct {name}: its base {root} is not a generated struct")
            moved = Layout(fields=s.fields, vptr=s.vptr, virtuals=s.virtuals, size=s.size)
            self.structs[root] = merge_layouts(self.structs[root], moved, root)
            self.structs[name] = Layout(methods=s.methods, base=s.base)

    def render(self, header: list[str]) -> str:
        out = list(header) + self.type_lines() + list(self.decls.values())
        out += [""] + [f + "\n" for _, f in sorted(self.functions, key=lambda af: af[0])]
        return "\n".join(out).rstrip() + "\n"

    def type_lines(self, skip: frozenset = frozenset(), definitions: bool = True, declarations: bool = True) -> list[str]:
        """Skeletons and forward declarations, then struct definitions (`skip`: defined elsewhere)."""
        roots = set(self.skeletons)
        out = []
        if declarations:
            out += [f"struct {o} {{ {' '.join(f'struct {k};' for k in kids)} }};" for o, kids in self.skeletons.items()]
            names = list(dict.fromkeys(list(self.forward) + [n for n in self.structs if n not in skip]))
            out += [f"struct {n};" for n in names if "::" not in n and n not in roots]
        for name in self._struct_order() if definitions else ():
            if name not in skip:
                out += struct_lines(name, self.structs[name])
        return out

    def merge_struct(self, name: str, s) -> None:
        """Another draft's parsed struct (a Layout, or opaque text) into this unit."""
        old = self.structs.get(name)
        if isinstance(s, str) or isinstance(old, str):
            if self.structs.setdefault(name, s) != s:
                raise Conflict(f"struct {name} defined two ways")
            return
        self.structs[name] = merge_layouts(old, s, name) if old is not None else copy.deepcopy(s)

    def _struct_order(self) -> list[str]:
        """Definitions with every struct held by value (a member, an array, a base) first."""
        def deps(name: str) -> set:
            s = self.structs[name]
            text = s if isinstance(s, str) else "\n".join(render_layout(s))
            words = set(re.findall(r"(?<![\w:])([A-Za-z_][\w:]*)(?! *\*)(?=[ \[{,]|$)", text))
            if isinstance(s, Layout) and s.base:
                words.add(s.base)
            return {w for w in words if w in self.structs and w != name}
        done: list[str] = []
        visiting: set = set()

        def visit(n: str) -> None:
            if n in done or n in visiting:
                return
            visiting.add(n)
            for d in sorted(deps(n)):
                visit(d)
            visiting.discard(n)
            done.append(n)
        for n in self.structs:
            visit(n)
        return done


def struct_lines(name: str, s) -> list[str]:
    if isinstance(s, str):
        return [s]
    head = f"struct {name}" + (f" : {s.base}" if s.base else "")
    return [f"{head} {{"] + [f"    {m}" for m in render_layout(s)] + ["};"]


# --- dk gen --merge: drafts into a source that already has code ---------------------------

_DEFINES = re.compile(r"\b(?:struct|class|union)\s+([A-Za-z_]\w*)\s*(?::[^;{}()]*)?\{")
_IDENT = re.compile(r"[A-Za-z_]\w*")


def definition_key(symbol: str) -> str | None:
    """A regex finding the start of `symbol`'s definition in a source (Class::name or name)."""
    from .demangle import demangle
    fn = demangle(symbol) if "__" in symbol.lstrip("_") else None
    if fn is None:
        return rf"(?<![\w:~]){re.escape(symbol)}" if re.fullmatch(r"[A-Za-z_]\w*", symbol) else None
    if not fn.quals:
        return None if fn.special else rf"(?<![\w:~]){re.escape(fn.name)}"
    cls = re.escape(fn.quals[-1])
    if fn.special == "ctor":
        return rf"\b{cls}::{cls}"
    if fn.special == "dtor":
        return rf"\b{cls}::~{cls}"
    if fn.special and fn.name.startswith("operator"):
        return rf"\b{cls}::operator\s*{re.escape(fn.name[8:].strip())}"
    return None if fn.special else rf"\b{cls}::{re.escape(fn.name)}"


def function_starts(lines: list[str], keys: dict[str, int]) -> dict[int, int]:
    """Line index -> address of each top-level definition (regex key -> address) found."""
    out = {}
    for key, address in keys.items():
        pattern = re.compile(rf"^[^\s#/][^;]*{key}\s*\(")
        for i, line in enumerate(lines):
            if pattern.match(line) and not line.rstrip().endswith(";"):
                out[i] = address
                break
    return out


_COMMENTS = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_DEPTH = re.compile(r"^/\* #line \d+\s+\"[^\"]*\"\s+/\* stack depth (\d+) \*/", re.M)


def header_text(preprocessed: str) -> str:
    """The parts of mwcc's preprocessed output that come from included files (depth > 0)."""
    parts = _DEPTH.split(preprocessed)
    return "\n".join(parts[i + 1] for i in range(1, len(parts) - 1, 2) if parts[i] != "0")


@dataclass
class Paste:
    """Exact drafts inserted into a source that already has code (see the module docstring).
    A name counts as known at a line when a header declares it or the source does above."""
    lines: list                                     # the source
    header_defined: set                             # class names the headers define
    header_idents: set                              # every identifier the headers mention
    editable: dict                                  # struct name -> (first line, last line): generator-format structs
    original: dict                                  # struct name -> its Layout in the source
    starts: dict                                    # line -> address of the source's functions
    unit: Unit = field(default_factory=Unit)        # editable and new structs, forward declarations, skeletons
    placed: list = field(default_factory=list)      # (line, address, [declarations], function text)
    moved: set = field(default_factory=set)         # editable structs a draft above them needs: moved up
    probe: object = None                            # (class, [member]) -> {member: offset or None}
    preprocessed: str = ""                          # the source with its headers (member declarations)

    @classmethod
    def of(cls, source: str, preprocessed: str, starts: dict, probe=None) -> "Paste":
        lines = source.splitlines()
        editable, original, sizes = {}, {}, {}
        i = 0
        while i < len(lines):
            m = _OPEN.match(lines[i])
            end = lines.index("};", i) if m and "};" in lines[i:] else None
            if m and end is not None:
                layout = parse_layout([x.strip() for x in lines[i + 1:end]], sizes, sizes.get(m[2], 0) if m[2] else None)
                if layout is not None and m[1] not in editable:
                    layout.base = m[2]
                    editable[m[1]], original[m[1]] = (i, end), layout
                    sizes[m[1]] = layout.sizeof()
                i = end
            i += 1
        headers = header_text(preprocessed)
        out = cls(lines, set(_DEFINES.findall(headers)), set(_IDENT.findall(headers)), editable, original, starts)
        out.unit.structs = copy.deepcopy(original)
        out.probe, out.preprocessed = probe, preprocessed
        return out

    def point(self, address: int) -> int:
        """The line a function at `address` goes before: the source's first function at a higher
        address (with the comment right above it), else the end."""
        later = [i for i, a in self.starts.items() if a > address]
        at = min(later) if later else len(self.lines)
        while later and at > 0 and self.lines[at - 1].startswith("//"):
            at -= 1
        return at

    def defines(self, name: str) -> bool:
        """A header or the source (outside its generator-format structs) defines this class."""
        return name not in self.editable and name.split("::")[-1] in self._above(len(self.lines))[0]

    def _above(self, line: int) -> tuple[set, set]:
        text = _COMMENTS.sub(" ", "\n".join(self.lines[:line]))
        return self.header_defined | set(_DEFINES.findall(text)), self.header_idents | set(_IDENT.findall(text))

    def add(self, draft: str, address: int) -> "Paste":
        """A copy with one draft merged in (Conflict if it cannot go in)."""
        d = Unit().add(draft, address)
        out = copy.copy(self)                       # (the probe and the source are shared)
        out.unit, out.placed, out.moved = copy.deepcopy(self.unit), list(self.placed), set(self.moved)
        u, at = out.unit, self.point(address)
        defined, idents = self._above(at)
        everywhere, mentioned = self._above(len(self.lines))
        for name in d.forward:
            if name.split("::")[-1] not in idents:
                u.forward.setdefault(name)
        for outer, kids in d.skeletons.items():
            if outer not in idents:
                have = u.skeletons.setdefault(outer, [])
                have += [k for k in kids if k not in have]
        for name, s in d.structs.items():
            if name in self.editable or name.split("::")[-1] not in everywhere:
                u.merge_struct(name, s)
                if name in self.editable and self.editable[name][0] >= at:
                    out.moved.add(name)
                continue
            # the source's or a header's definition is used: it must have the draft's members
            members = [] if isinstance(s, str) else ([t for t, _ in s.fields.values()] if self.probe is None else []) \
                + s.methods + s.virtuals
            names = {m[1]: t for t in members if (m := re.search(r"(~?\w+)\s*(?:\(|\[\d*\];$|;$)", t))}
            missing = sorted(n for n in names if n.lstrip("~") not in mentioned)
            if missing:
                plain = [names[n] for n in missing if "(" in names[n] and not names[n].startswith("virtual ")]
                text = f"{name} is defined by the source or a header without {', '.join(missing)}"
                if len(plain) == len(missing):
                    raise Missing(name, [], plain, text)
                raise Conflict(text)
            if self.probe is not None and isinstance(s, Layout) and s.fields:
                self._check_fields(name, s, d)
            if name.split("::")[-1] not in defined:
                raise Conflict(f"{name} is defined only below line {at + 1}, where the function goes")
        decls = []
        own = {k: line for line in self.lines if line.startswith("extern ") and line.endswith(";")
               and (k := (_EXTERN_FN.match(line) or _EXTERN_VAR.match(line) or [None, None])[1])}
        for key, text in d.decls.items():
            if key in idents:
                continue
            text = own.get(key, text)               # declared lower down in the source: that declaration
            other = next((l for l in self.lines if l.startswith('extern "C"') and re.search(rf"\b{re.escape(key)}\(", l)
                          and l.split("{")[0].strip().rstrip(";") != text.rstrip(";")), None)
            if other is not None and text.startswith('extern "C"'):
                raise Conflict(f"the source has {other.split('{')[0].strip()}, the draft needs {text}")
            other = next((t for _, _, ds, _ in out.placed for t in ds if (_EXTERN_FN.match(t) or _EXTERN_VAR.match(t)
                                                                           or [None, t])[1] == key), text)
            if other != text:
                raise Conflict(f"{key} declared twice: {other} / {text}")
            decls.append(text)
        out.placed += [(at, address, decls, out._as_declared(f)) for _, f in d.functions]
        u._sink_to_bases()
        return out

    def _as_declared(self, function: str) -> str:
        """A member function of a class the source or a header defines, with the head that class
        declares for it (return and parameter types; parameters keep the draft's register names)."""
        m = _HEAD.match(function)
        if m is None or not self.defines(m["cls"]) or m["name"].lstrip("~") == m["cls"].split("::")[-1]:
            return function
        decls = member_declarations(self.preprocessed, m["cls"], m["name"])
        if any(inline for _, inline in decls):
            raise Conflict(f"{m['cls']}::{m['name']} is defined inline in a header: the unit emits it where it is "
                           f"used (the draft is redundant)")
        if len(decls) != 1:
            return function
        head = declared_head(decls[0][0], m["cls"], m["name"])
        if head is None:
            return function
        body = function[m.end("const"):]
        if head.startswith("void ") and m["ret"].strip() != "void":      # declared void: no value back
            body = re.sub(r"^(\s*)return (.+);$", r"\1\2;", body, flags=re.M)
        return head + body

    def _check_fields(self, name: str, s: Layout, d: Unit) -> None:
        """The draft's fields of a struct it does not define must be members at those offsets
        (asked through the draft's most derived class: a header may hold them lower down)."""
        via = name
        while (child := next((n for n, x in d.structs.items() if isinstance(x, Layout) and x.base == via), None)):
            via = child
        want = {_FIELD.match(text)["name"]: off for off, (text, _) in s.fields.items()}
        got = self.probe(via, list(want))
        lost = sorted((off, s.fields[off]) for n, off in want.items() if got.get(n) is None)
        missing = [f"+0x{off:X} {text.rstrip(';')}" for off, (text, _) in lost]
        if missing:
            raise Missing(via, [(off, size, text) for off, (text, size) in lost], [],
                          f"{via} (the source's or a header's) has no member "
                          + ", ".join(missing[:6]) + (f" (+{len(missing) - 6} more)" if len(missing) > 6 else ""))
        moved = [f"{n} at +0x{got[n]:X}, not +0x{off:X}" for n, off in want.items() if got[n] != off]
        if moved:
            raise Conflict(f"{via} has " + ", ".join(moved))

    def render(self) -> str:
        lines = self.lines
        changed = {n: span for n, span in self.editable.items()
                   if self.unit.structs.get(n) != self.original[n] or n in self.moved}
        before: dict = defaultdict(list)
        skip = frozenset(self.editable) - self.moved
        first = min((at for at, _, _, _ in self.placed), default=len(lines))
        declarations = self.unit.type_lines(skip, definitions=False)
        if self.placed and declarations:            # above every struct that may now point at them
            before[min([first] + [a for n, (a, _) in changed.items()])] += declarations
        definitions = self.unit.type_lines(skip, declarations=False)
        if self.placed and definitions:
            before[first] += definitions + [""]
        placed_at: dict = {}                            # declaration -> the first place it goes
        for at, address, decls, text in sorted(self.placed, key=lambda x: (x[0], x[1])):
            fresh = [d for d in decls if d not in placed_at]    # one copy, at the highest place
            placed_at.update((d, at) for d in fresh)
            if at == len(lines) and lines and lines[-1].strip() and not before[at]:
                before[at].append("")
            before[at] += fresh + [text, ""]
        out, i = [], 0
        spans = {span[0]: (name, span[1]) for name, span in changed.items()}
        while i < len(lines):
            out += before.get(i, [])
            if i in spans:
                name, end = spans[i]
                out += [] if name in self.moved else struct_lines(name, self.unit.structs[name])
                i = end + 1
                continue
            if lines[i].startswith("extern ") and placed_at.get(lines[i], i + 1) <= i:
                i += 1                                  # the source's own copy, now declared higher up
                continue
            out.append(lines[i])
            i += 1
        out += before.get(len(lines), [])
        return "\n".join(out).rstrip() + "\n"


def grow_class(text: str, cls: str, fields: list, methods: list, pads: dict) -> str | None:
    """`text` (a header) with `cls`'s missing members added without changing its layout:
    each field (offset, size, member text) split out of the padding array that covers it
    (`pads`: array name -> (offset, size) in cls, from a probe), each method declared before
    the class body closes. None when a field falls outside the class's own padding."""
    m = re.search(rf"\b(?:class|struct)\s+{re.escape(cls.split('::')[-1])}\b[^;{{()]*\{{", text)
    if m is None:
        return None
    depth, i = 1, m.end()
    while depth and i < len(text):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    body_start, body_end = m.end(), i - 1                   # body_end: the closing brace
    body = text[body_start:body_end]
    by_pad: dict = defaultdict(list)
    for off, size, member in sorted(fields):
        pad = next((n for n, (p, n_bytes) in pads.items() if p <= off and off + size <= p + n_bytes), None)
        if pad is None:
            return None
        by_pad[pad].append((off, size, member))
    for pad, inside in by_pad.items():
        line = re.search(rf"^([ \t]*)(?:unsigned )?char\s+{re.escape(pad)}\s*\[[^\]]+\];[^\n]*$", body, re.M)
        if line is None:
            return None
        start, total = pads[pad]
        indent, cur, out = line[1], start, []
        for off, size, member in inside:
            if off < cur:
                return None                                  # two fields overlap
            if off > cur:
                out.append(f"{indent}char pad{cur:X}[0x{off - cur:X}];")
            out.append(f"{indent}{member}")
            cur = off + size
        if cur < start + total:
            out.append(f"{indent}char pad{cur:X}[0x{start + total - cur:X}];")
        body = body[:line.start()] + "\n".join(out) + body[line.end():]
    if methods:
        indent = re.search(r"^([ \t]+)\S", body, re.M)
        indent = indent[1] if indent else "    "
        tail = body.rstrip()
        access = re.findall(r"\b(public|private|protected)\s*:", body)
        lines = ([f"{indent[:-4] if len(indent) >= 4 else ''}public:"] if access and access[-1] != "public" else []) \
            + [f"{indent}{decl}" for decl in methods]
        body = tail + "\n" + "\n".join(lines) + "\n"
    return text[:body_start] + body + text[body_end:]


_HEAD = re.compile(r"^(?P<ret>[^(\n]*?)\b(?P<cls>[A-Za-z_][\w:]*)::(?P<name>~?\w+)\((?P<params>[^)]*)\)(?P<const>(?: const)?)")
_TYPE_WORDS = {"int", "char", "short", "long", "float", "double", "bool", "unsigned", "signed", "void", "const"}


def member_declarations(text: str, cls: str, name: str) -> list[tuple[str, bool]]:
    """(declaration, defined inline) of each member `name` in class `cls`'s body in `text`."""
    m = re.search(rf"\b(?:class|struct)\s+{re.escape(cls.split('::')[-1])}\b[^;{{()]*\{{", text)
    if m is None:
        return []
    out, depth, start, i = [], 1, m.end(), m.end()
    while i < len(text) and depth:
        c = text[i]
        if c == ";" and depth == 1:
            out.append((text[start:i], False))
            start = i + 1
        elif c == "{":
            if depth == 1:
                out.append((text[start:i], True))
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 1:
                start = i + 1
        i += 1
    found = []
    for decl, inline in out:
        decl = re.sub(r"/\*.*?\*/|//[^\n]*", " ", decl)
        decl = re.sub(r"^(?:\s*(?:public|private|protected)\s*:)+", "", decl.strip()).strip()
        if re.search(rf"(?<![\w~]){re.escape(name)}\s*\(", decl):
            found.append((" ".join(decl.split()), inline))
    return found


def declared_head(decl: str, cls: str, name: str) -> str | None:
    """`R C::name(T arg1, ...)` for a member declaration: its types, the draft's register names
    (arg<n> for r<n+3>, farg<n> for f<n+1>; this is r3). None for what it cannot parse."""
    m = re.fullmatch(rf"(?P<ret>.*?)\b{re.escape(name)}\s*\((?P<params>[^()]*)\)(?P<const>\s*const)?(?:\s*=\s*0)?", decl)
    if m is None:
        return None
    ret = " ".join(w for w in m["ret"].split() if w not in ("virtual", "static", "inline", "explicit"))
    if not ret or "static" in m["ret"].split():
        return None
    params, gpr, fpr = [], 1, 0
    for p in [x.strip() for x in m["params"].split(",")] if m["params"].strip() not in ("", "void") else []:
        p = p.split("=")[0].strip()
        words = re.match(r"^(.*?)([A-Za-z_]\w*)$", p)
        t = words[1].strip() if words and words[1].strip() and words[2] not in _TYPE_WORDS else p
        if not t or "..." in t:
            return None
        fp = t in ("float", "double")
        params.append(f"{t} {'farg' if fp else 'arg'}{fpr if fp else gpr}")
        fpr, gpr = (fpr + 1, gpr) if fp else (fpr, gpr + 1)
    return f"{ret} {cls}::{name}({', '.join(params)}){' const' if m['const'] else ''}"
