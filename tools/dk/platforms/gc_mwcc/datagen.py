"""dk gen --data <unit> [--write]: typed definitions for a unit's own data objects, each verified.

For every object symbols.txt places in the unit's data sections:
- what the compiler emits itself is left alone: vtables, RTTI, switch jump tables, `@` literals,
  .ctors, and pooled literals (strings and .sdata2/.rodata numbers that only this unit's code
  reads: they come from the functions that use them);
- the rest is typed from evidence: the unit's own `extern` declaration (its type, written as a
  definition), dtk's element type (float, double, 4byte, 2byte, byte) with the values from the
  original bytes, a table of relocations as a table of pointers to those C-named symbols, a
  NUL-terminated printable run as a string, and a zero-filled (.bss) object as a sized
  declaration of its element type;
- what has no evidence (an initialized object with mixed words, a pointer to a mangled name) is a
  TODO with its size and users: never a dump of the original bytes.
Each draft is compiled with the unit and kept only when its symbol comes out with the target's
size, bytes and relocations (verified, like gen's code drafts). --write applies the verified ones
(the externs become definitions in place; objects the source never names go in one block before
its first function, in address order).
"""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass, field

from ...core import cbody

DATA = (".data", ".rodata", ".sdata", ".sdata2", ".bss", ".sbss")
ZERO = (".bss", ".sbss")
ELEM = {"4byte": ("int", 4), "float": ("float", 4), "double": ("double", 8), "2byte": ("short", 2),
        "byte": ("unsigned char", 1)}
_ITEM = re.compile(r'^\s*\.(4byte|2byte|byte|float|double|string|skip|rel|asciz|ascii)\s+(.*?)\s*$')


@dataclass
class Item:
    kind: str                      # "word", "reloc", "half", "byte", "string", "skip", "rel"
    value: object
    size: int


@dataclass
class Draft:
    name: str
    section: str
    address: int
    size: int
    status: str                    # "draft", "verified", "skip", "todo", "failed", "defined"
    why: str = ""
    text: str = ""                 # the definition (without the final newline)
    needs: list[str] = field(default_factory=list)   # extern declarations it relies on
    replaces: tuple[int, int] | None = None          # the extern declaration it turns into a definition
    evidence: str = ""                               # what the type and values rest on (for review)


def items(text: str) -> list[Item]:
    out = []
    for line in text.splitlines():
        m = _ITEM.match(line)
        if not m:
            continue
        kind, arg = m.groups()
        if kind == "4byte":
            if re.fullmatch(r"0x[0-9A-Fa-f]+|-?\d+", arg):
                out.append(Item("word", int(arg, 0) & 0xFFFFFFFF, 4))
            else:
                out.append(Item("reloc", arg.strip('"'), 4))
        elif kind == "2byte":
            out.append(Item("half", int(arg, 0), 2))
        elif kind == "byte":
            out.extend(Item("byte", int(v.strip(), 0), 1) for v in arg.split(","))
        elif kind in ("string", "asciz"):
            s = bytes(arg[1:-1], "latin-1").decode("unicode_escape").encode("latin-1")
            out.append(Item("string", s, len(s) + 1))
        elif kind == "skip":
            out.append(Item("skip", None, int(arg, 0)))
        elif kind == "rel":
            out.append(Item("rel", arg, 4))
    return out


def c_name(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_]\w*", name)) and "__" not in name.lstrip("_")


def _float(v: int) -> str | None:
    f = struct.unpack(">f", struct.pack(">I", v))[0]
    if not math.isfinite(f):
        return None
    for digits in range(6, 10):
        text = f"{f:.{digits}g}"
        if struct.pack(">f", float(text)) == struct.pack(">I", v):
            break
    if "e" not in text and "." not in text:
        text += ".0"
    return text + "f"


LONG_TABLE = 32          # words: a longer 4byte table with no extern is not typed as int[] (a struct array, most likely)


def float_like(v: int) -> bool:
    """A word that reads as a plain float (|value| between about 1e-3 and 1e7 with a short decimal form) rather
    than as an int: 0x3F0CCCCD is 0.55f, and an int of 1,057,803,469 is the less likely reading."""
    if not (0x3A800000 <= (v & 0x7FFFFFFF) <= 0x4B800000):
        return False
    text = _float(v)
    return text is not None and len(text.rstrip("f").lstrip("-").replace(".", "")) <= 7


def _int(v: int) -> str:
    v = v - (1 << 32) if v & 0x80000000 else v
    return str(v)


def extern_of(src: str, name: str) -> tuple[str, tuple[int, int]] | None:
    """The unit's own `extern T name…;` declaration of the object: (definition text, its span)."""
    m = re.search(rf'^[ \t]*extern[ \t]+(?!"C")([^;\n]*?\b{re.escape(name)}\b[^;\n]*);[^\n]*\n?', src, re.M)
    if m is None:
        return None
    return m.group(1).strip() + ";", (m.start(), m.end())


_TYPE_HEAD = re.compile(r"^(?:(?:static|const|volatile|unsigned|signed|struct|class|union)\s+)*[A-Za-z_][\w:<>]*"
                        r"(?:\s+[A-Za-z_][\w:<>]*)*\s*[*&\s]*$")


def defined_in(src: str, name: str) -> bool:
    """Does the source already define the object: a top-level declaration of it that is not extern?"""
    for m in re.finditer(rf"^([^\s\n;(][^\n;(=]*?)\b{re.escape(name)}\b\s*(\[[^\]]*\])*\s*(=|;)", src, re.M):
        head = m.group(1)
        if "extern" not in head.split() and _TYPE_HEAD.match(head.strip()) and head.split()[0] not in ("return",):
            return True
    return False


_R5 = re.compile(r"^(?:addi r5, r5, |li r5, |addi r5, r\d+, )\"?([^\"@\s]+)\"?@(?:l|sda21)")


def destructor_records(asm_texts) -> set[str]:
    """Objects passed as the record (third argument) to __register_global_object: the compiler's
    destructor-chain entries for static objects with destructors."""
    out = set()
    for text in asm_texts:
        if "__register_global_object" not in text:
            continue
        r5 = None
        for line in text.splitlines():
            ins = re.sub(r"/\*.*?\*/", "", line).strip()
            m = _R5.match(ins)
            if m:
                r5 = m.group(1)
            elif re.match(r"^(?:mr|li|lwz|addi|lis) r5,", ins):
                r5 = None
            if ins.startswith("bl") and "__register_global_object" in ins:
                if r5:
                    out.add(r5)
                r5 = None
    return out


def draft(sym, kind_items: list[Item], section: str, src: str, users_here: list[str], users_elsewhere: list[str],
          data_users: list[str], scope: str, target: tuple[bytes, dict] | None = None,
          class_names: frozenset = frozenset(), copied: bool = False, strings: dict | None = None,
          dtor_record: bool = False, indexed: bool = False) -> Draft:
    if dtor_record:
        return Draft(sym.name, section, sym.address, sym.size, "skip",
                     "a destructor-chain record (__register_global_object): emitted with its static object")
    d = _draft(sym, kind_items, section, src, users_here, users_elsewhere, data_users, scope, target, class_names,
               copied, strings or {}, indexed)
    return d


def _string_blob(data: bytes) -> bool:
    """Bytes that are NUL-terminated printable strings (runs of 3 or more), with only NUL padding."""
    runs = [r for r in data.split(b"\0") if r]
    return bool(runs) and all(len(r) >= 3 and all(32 <= c < 127 or c in (9, 10) for c in r) for r in runs)


def local_statics(objects: list, users: dict) -> set[str]:
    """Function-local statics and their guards: an object followed by a 1-byte object that every one
    of its users also reads (`static float x = f();` keeps `x` and an init flag side by side)."""
    out = set()
    for a, g in zip(objects, objects[1:]):
        if g.size == 1 and a.size > 1 and g.address - (a.address + a.size) < 4:
            ua, ug = set(users.get(a.name, [])), set(users.get(g.name, []))
            if ua and ua <= ug:
                out.update((a.name, g.name))
    return out


def copied_to_stack(obj: str, asm_texts) -> bool:
    """Does some user copy the object onto the stack word by word (loads through its address
    stored to r1 slots or through a pointer into the frame): a local array's initializer."""
    for text in asm_texts:
        held, frame, loaded = set(), set(), set()
        for line in text.splitlines():
            ins = re.sub(r"/\*.*?\*/", "", line).strip()
            m = re.match(rf"^(?:addi (r\d+), r\d+, |li (r\d+), )\"?{re.escape(obj)}\"?@(?:l|sda21)$", ins)
            if m:
                held.add(m.group(1) or m.group(2))
                continue
            f = re.match(r"^addi (r\d+), r1, ", ins)
            if f:
                frame.add(f.group(1))
                continue
            d = re.match(r"^(?:addi|subi) (r\d+), (r\d+), -?(?:0x[0-9a-fA-F]+|\d+)$", ins) or \
                re.match(r"^mr (r\d+), (r\d+)$", ins)
            if d and d.group(2) in held | frame:          # a pointer derived from either (a copy loop's)
                (held if d.group(2) in held else frame).add(d.group(1))
                continue
            ld = re.match(r"^(?:lwz|lwzu|lfd|lfdu|lfs) ([rf]\d+), -?(?:0x[0-9a-fA-F]+|\d+)\((r\d+)\)$", ins)
            if ld and ld.group(2) in held:
                loaded.add(ld.group(1))
                continue
            st = re.match(r"^(?:stw|stwu|stfd|stfdu|stfs) ([rf]\d+), -?(?:0x[0-9a-fA-F]+|\d+)\((r\d+)\)$", ins)
            if st and st.group(1) in loaded and (st.group(2) == "r1" or st.group(2) in frame):
                return True
    return False


def indexed_read(obj: str, asm_texts) -> bool:
    """Does code index into the object (its address in a register, then lwzx/lhzx/lbzx/lfsx off it)?
    Then it is a table, not literals."""
    for text in asm_texts:
        held = set()
        for line in text.splitlines():
            ins = re.sub(r"/\*.*?\*/", "", line).strip()
            m = re.match(rf"^addi (r\d+), r\d+, \"?{re.escape(obj)}\"?@l$", ins) or \
                re.match(rf"^li (r\d+), \"?{re.escape(obj)}\"?@sda21$", ins)
            if m:
                held.add(m.group(1))
                continue
            x = re.match(r"^(?:lwzx|lhzx|lhax|lbzx|lfsx|lfdx) [rf]\d+, (r\d+), (r\d+)$", ins)
            if x and (x.group(1) in held or x.group(2) in held):
                return True
    return False


def _draft(sym, kind_items, section, src, users_here, users_elsewhere, data_users, scope, target, class_names,
           copied, string_objs, indexed=False) -> Draft:
    d = Draft(sym.name, section, sym.address, sym.size, "draft")
    name = sym.name
    code_users = users_here + users_elsewhere
    if data_users and not code_users:
        d.status, d.why = "skip", (f"referenced only from data ({', '.join(data_users[:2])}): emitted with it "
                                   "(RTTI, a vtable, or a table's literals)")
        return d
    if copied:
        d.status, d.why = "skip", "the template of a local array's initializer (its users copy it whole)"
        return d
    text_bytes = b"".join(x.value for x in kind_items if x.kind == "string") or \
        b"".join(struct.pack(">I", x.value) for x in kind_items if x.kind == "word")
    if text_bytes.rstrip(b"\0").decode("latin-1") in class_names:
        d.status, d.why = "skip", "a class name (RTTI): emitted by the compiler"
        return d
    if name.startswith(("__vt__", "__RTTI__", "jumptable_")) or "@" in name or "..." in name or section == ".ctors":
        d.status, d.why = "skip", "emitted by the compiler (vtable, RTTI, jump table or literal)"
        return d
    if defined_in(src, name):
        d.status, d.why = "defined", "already defined in the source"
        return d
    strings = [x for x in kind_items if x.kind == "string"]
    words = [x for x in kind_items if x.kind in ("word", "half", "byte")]
    relocs = [x for x in kind_items if x.kind == "reloc"]
    if any(x.kind == "rel" for x in kind_items):
        d.status, d.why = "skip", "a jump table (emitted by its switch)"
        return d
    pooled = not users_elsewhere and not data_users
    if strings and not relocs and pooled and not indexed:
        d.status, d.why = "skip", "a string literal of the unit's code (pooled with its function)"
        return d
    if section in (".sdata2", ".rodata") and pooled and not relocs and sym.attrs.get("data") in ("float", "double"):
        d.status, d.why = "skip", "a number literal of the unit's code (pooled with its function)"
        return d
    blob = b"".join(struct.pack(">I", x.value) for x in words if x.kind == "word")
    if words and not relocs and pooled and not indexed and _string_blob(blob) and \
            sym.attrs.get("data") not in ("float", "4byte"):
        d.status, d.why = "skip", "string literals of the unit's code (pooled with their functions)"
        return d
    storage = "static " if scope == "local" else ""
    ext = extern_of(src, name)
    if section in ZERO:
        if ext is not None:
            d.text, d.replaces = storage + ext[0], ext[1]
            d.evidence = "zero-filled; the unit's extern gives the type"
            return d
        elem, size = ELEM.get(sym.attrs.get("data", ""), ("int", 4) if sym.size % 4 == 0 else ("unsigned char", 1))
        n = sym.size // size
        d.text = f"{storage}{elem} {name}" + (f"[{n}]" if n != 1 or sym.size % size else "") + ";"
        d.evidence = f"zero-filled {sym.size:#x} bytes; dtk element type {sym.attrs.get('data') or 'unknown'}"
        d.why = f"zero-filled {sym.size:#x} bytes" + ("" if sym.attrs.get("data") else " (element type unknown)")
        return d
    if ext is not None and (relocs or words) and re.search(r"\[|\*", ext[0]) is None and len(words) == 1 and not relocs:
        v = words[0].value
        typ = ext[0].split(name)[0].strip()
        value = _float(v) if typ.endswith("float") else _int(v)
        if value is not None:
            d.text, d.replaces = f"{storage}{ext[0][:-1]} = {value};", ext[1]
            d.evidence = "the unit's extern gives the type; value from the original"
            return d
    if relocs and not words and not strings:
        targets = [x.value for x in relocs]
        if not all(c_name(t) for t in targets):
            d.status, d.why = "todo", f"a table of {len(targets)} pointers, some to C++ names: write it with their types"
            return d
        fns = [t for t in targets if t.startswith("fn_")]
        if fns and len(fns) == len(targets):
            d.needs = [f'extern "C" void {t}();' for t in sorted(set(fns)) if not re.search(rf"\b{t}\b", src)]
            body = ", ".join(targets)
            d.text = f"{storage}void (*{name}[{len(targets)}])() = {{{body}}};" if len(targets) > 1 or sym.size > 4 \
                else f"{storage}void (*{name})() = {targets[0]};"
            d.evidence = f"{len(targets)} relocations to functions"
            return d
        if all(t in string_objs for t in targets):     # a table of strings: their literals, written in it
            lits = ", ".join('"' + string_objs[t].decode("latin-1").encode("unicode_escape").decode().replace('"', '\\"')
                             + '"' for t in targets)
            d.text = f"{storage}const char* {name}[{len(targets)}] = {{{lits}}};"
            d.evidence = f"{len(targets)} relocations to strings, written as literals"
            return d
        d.needs = [f"extern char {t}[];" for t in sorted(set(targets)) if not re.search(rf"\b{t}\b", src)]
        d.text = f"{storage}void* {name}[{len(targets)}] = {{{', '.join(targets)}}};"
        d.evidence = f"{len(targets)} relocations to named objects"
        return d
    if strings and len(strings) == 1 and not relocs and not words:
        s = strings[0].value
        text = s.decode("latin-1").encode("unicode_escape").decode().replace('"', '\\"')
        size = f"[{sym.size}]" if sym.size != len(s) + 1 else "[]"
        d.text = f'{storage}char {name}{size} = "{text}";'
        d.evidence = f"a string other code uses ({', '.join((users_elsewhere or data_users)[:2])})"
        return d
    kind = sym.attrs.get("data", "")
    if words and not relocs and not strings and kind in ("float", "4byte", "double") and all(x.kind == "word" for x in words):
        values = [x.value for x in words]
        if kind == "float":
            vals = [_float(v) for v in values]
            if None in vals:
                d.status, d.why = "todo", "floats that are not finite numbers"
                return d
            elem = "float"
        elif kind == "double":
            if len(values) % 2:
                d.status, d.why = "todo", "an odd number of words for doubles"
                return d
            vals = [repr(struct.unpack(">d", struct.pack(">II", a, b))[0]) for a, b in zip(values[::2], values[1::2])]
            elem = "double"
        else:
            if ext is None:
                floaty = [_float(v) for v in values if float_like(v)]
                if floaty:
                    d.status, d.why = "todo", (f"4byte word(s) that read as floats ({', '.join(floaty[:3])}): dtk's "
                                               "element type is unconfirmed, type it by hand (an extern)")
                    return d
                if len(values) >= LONG_TABLE:
                    d.status, d.why = "todo", (f"a table of {len(values)} words typed only by dtk's 4byte: probably "
                                               "an array of structs, give it an extern type")
                    return d
            vals, elem = [_int(v) for v in values], "int"
        const = "const " if section == ".rodata" else ""
        if ext is not None:
            d.replaces = ext[1]
        d.evidence = f"dtk element type {kind}; values from the original" + ("; the unit's extern" if ext else "")
        d.text = (f"{storage}{const}{elem} {name}" + (f"[{len(vals)}]" if len(vals) > 1 else "")
                  + " = " + (f"{{{', '.join(vals)}}}" if len(vals) > 1 else vals[0]) + ";")
        return d
    if ext is not None and target is not None:
        if ext[1] is not None:
            d.replaces = ext[1]
        aggregate(src, d, ext[0], target[0], target[1])
        if d.text:
            d.text = storage + d.text
            d.evidence = "the unit's struct type; members from the original bytes"
        return d
    users = (users_here + users_elsewhere)[:4]
    d.status, d.why = "todo", (f"{sym.size:#x} bytes with no type evidence (a struct?); "
                               f"used by {', '.join(users) or 'nothing named'}: declare its type, then define it")
    return d


def apply(src: str, drafts: list[Draft]) -> str:
    """The source with the drafts written: externs turned into definitions in place, the rest in
    one block (address order) before the first function, with the declarations they need."""
    edits = []
    block, needs = [], []
    for d in sorted(drafts, key=lambda x: x.address):
        if d.replaces is not None:
            edits.append((d.replaces[0], d.replaces[1], d.text + "\n"))
        else:
            block.append(d.text)
        needs += [n for n in d.needs if n not in needs]
    if block or needs:
        # at the end of the file: after the unit's own statics (which may be addressed off the
        # section's start), and the source names none of these objects before (they had no extern)
        at = len(src)
        edits.append((at, at, ("" if src.endswith("\n\n") else "\n" if src.endswith("\n") else "\n\n")
                      + "\n".join(needs + block) + "\n"))
    return cbody.apply_edits(src, edits)


def resolved(elf, sec_index: int, reloc) -> str:
    if reloc.symbol and not reloc.symbol.startswith("."):
        return reloc.symbol
    target = reloc.target_value + reloc.addend
    for s in elf.symbols:
        if s.name and elf.sections[s.shndx].name == reloc.target_section if 0 < s.shndx < len(elf.sections) else False:
            if s.value == target:
                return s.name
    return f"{reloc.target_section}+{target:#x}"


def verify(ours, target_bytes: bytes | None, target_relocs: dict[int, str], d: Draft) -> str | None:
    """Why our compiled object `d` differs from the target's (None: the same)."""
    sym = next((s for s in ours.symbols if s.name == d.name and 0 < s.shndx < len(ours.sections)), None)
    if sym is None:
        return "not in the compiled object (unreferenced and dropped?)"
    sec = ours.sections[sym.shndx]
    if sec.name != d.section:
        return f"compiled into {sec.name}, not {d.section}"
    if sym.size > d.size or d.size - sym.size >= 8:
        return f"size {sym.size:#x}, target {d.size:#x}"
    if d.section not in ZERO and target_bytes is not None:
        mine = sec.data[sym.value:sym.value + sym.size]
        if mine != target_bytes[:sym.size] or any(target_bytes[sym.size:]):
            return "bytes differ"
    rel = next((s for s in ours.sections if s.type == 4 and s.info == sym.shndx), None)
    mine_relocs = {}
    if rel is not None:
        for r in ours.relocations(sym.shndx):
            if sym.value <= r.offset < sym.value + sym.size:
                mine_relocs[r.offset - sym.value] = resolved(ours, sym.shndx, r)
    if mine_relocs != target_relocs:
        return "relocations differ"
    return None


@dataclass
class UnitData:
    """A unit's data objects drafted and verified against the target (gen --data's working state)."""
    unit: dict
    rel: str
    source: object                 # Path of the unit's source
    src: str
    drafts: list
    target: object                 # the target Elf
    tsyms: dict                    # target symbol name -> symbol
    attrs: dict = field(default_factory=dict)   # object name -> its symbols.txt attributes (type, size, scope, data)

    def original(self, name: str, size: int):
        """The target's bytes and relocations (offset -> symbol) for `name`, or None."""
        t = self.tsyms.get(name)
        if t is None:
            return None
        sec_ = self.target.sections[t.shndx]
        rel_ = {r.offset - t.value: resolved(self.target, t.shndx, r) for r in self.target.relocations(t.shndx)
                if t.value <= r.offset < t.value + size}
        return sec_.data[t.value:t.value + size], rel_

    @property
    def verified(self) -> list:
        return [d for d in self.drafts if d.status == "verified"]

    @property
    def counted(self) -> list:
        return [d for d in self.drafts if d.status != "skip"]


def draft_unit(cfg, ctx, unit: dict, jobs: int, audit: bool = False, hits: list | None = None):
    """Draft every data object of `unit` from its evidence and verify each against the target
    (size, bytes, relocations); None when the unit is not in splits.txt. With audit, only collect
    into `hits` the definitions the source has of what the compiler emits itself."""
    hits = hits if hits is not None else []
    import shutil
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor
    from .dtk_symbols import read_symbols, read_units
    from .elf import Elf
    from .reorder import compile_variant
    rel = unit["metadata"]["source_path"].removeprefix("src/")
    split = next((u for u in read_units(cfg.root / cfg.platform_options["splits"]) if u.name == rel), None)
    if split is None:
        return None
    source = cfg.root / unit["metadata"]["source_path"]
    src = source.read_text()
    symbols = read_symbols(cfg.root / cfg.platform_options["symbols"])
    text_ranges = [(a, z) for sec, a, z in split.splits if sec == ".text"]
    by_name = {s.name: s for s in symbols}
    mine = {s.name for s in symbols if s.type == "function" and any(a <= s.address < z for a, z in text_ranges)}
    from . import asmindex
    entries = asmindex.index(asmindex.current_asm_files(cfg))       # objects too (gen's context holds functions)
    users = defaultdict(list)
    data_users = defaultdict(list)
    for name, e in entries.items():
        for r in e.refs:
            (users if e.kind == "fn" else data_users)[r].append(name)
    target = Elf.parse((cfg.root / unit["target_path"]).read_bytes())
    tsyms = {s.name: s for s in target.symbols if s.name and 0 < s.shndx < len(target.sections)}
    ud = UnitData(unit, rel, source, src, [], target, tsyms)
    original = ud.original

    import json
    from .boundaries import _copied_whole
    rtti = cfg.root / "build" / "dk" / "rtti.json"
    class_names = frozenset(c["name"] for c in json.loads(rtti.read_text())) if rtti.is_file() else frozenset()
    if not class_names:
        print("note: no build/dk/rtti.json: class-name strings are not recognised (run ./dk analyze rtti)")
    asm_text = {n: e.text for n, e in entries.items() if e.kind == "fn"}
    records = destructor_records(t for n, t in asm_text.items() if n in mine)
    string_objs = {}
    for n, e in entries.items():
        if e.kind == "obj":
            its = items(e.text)
            if len(its) == 1 and its[0].kind == "string":
                string_objs[n] = its[0].value
    covered = {} if audit else placed(cfg, unit, src, target)   # target objects the source defines (any name)
    drafts: list[Draft] = ud.drafts
    unit_objects = sorted((x for x in symbols if x.section in DATA and any(
        sec_ == x.section and a <= x.address < z for sec_, a, z in split.splits)), key=lambda x: x.address)
    statics = local_statics(unit_objects, users)
    ud.attrs = {x.name: dict(x.attrs) for x in unit_objects}
    for sec, lo, hi in split.splits:
        if sec not in DATA:
            continue
        for s in sorted((x for x in symbols if x.section == sec and lo <= x.address < hi), key=lambda x: x.address):
            e = entries.get(s.name)
            its = items(e.text) if e is not None else [Item("skip", None, s.size)] if sec in ZERO else []
            here = [u for u in users.get(s.name, []) if u in mine]
            elsewhere = [u for u in users.get(s.name, []) if u not in mine]
            user_syms = [by_name[u] for u in users.get(s.name, []) if u in by_name]
            copied = sec not in ZERO and (_copied_whole(s, user_syms, asm_text) or copied_to_stack(
                s.name, (asm_text[u] for u in users.get(s.name, []) if u in asm_text)))
            if s.name in statics:
                d = Draft(s.name, sec, s.address, s.size, "skip",
                          "a function-local static or its guard (its users test the guard first)")
                if not audit:
                    drafts.append(d)
                continue
            d = draft(s, its, sec, "" if audit else src, here, elsewhere, data_users.get(s.name, []),
                      s.attrs.get("scope", "global"), original(s.name, s.size) if sec not in ZERO else None,
                      class_names, copied, string_objs, s.name in records,
                      indexed_read(s.name, (asm_text[u] for u in users.get(s.name, []) if u in asm_text)))
            if audit:
                if d.status == "skip" and defined_in(src, s.name):
                    hits.append((rel, s.name, sec, d.why))
                continue
            if d.status != "skip" and s.name in covered:
                d = Draft(s.name, sec, s.address, s.size, "defined", f"already defined in the source as {covered[s.name]}")
            drafts.append(d)
    address_of = {s.name: s.address for s in symbols if s.section in DATA}
    found: dict[str, dict[int, tuple[str, int]]] = defaultdict(dict)
    objects = sorted((s for s in symbols if s.section in DATA and any(
        sec == s.section and a <= s.address < z for sec, a, z in split.splits)), key=lambda x: x.address)
    starts = [o.address for o in objects]
    import bisect
    for fname in sorted(mine):
        if fname not in asm_text:
            continue
        for addr, typ, width in accesses(asm_text[fname], address_of):
            k = bisect.bisect_right(starts, addr) - 1
            if k >= 0 and addr < objects[k].address + objects[k].size:
                found[objects[k].name].setdefault(addr - objects[k].address, (typ, width))
    if audit:
        return ud
    for d in drafts:
        if d.status == "todo" and d.section not in ZERO and "no type evidence" in d.why and \
                extern_of(src, d.name) is None and d.name in tsyms:
            o = original(d.name, d.size)
            if o is not None:
                infer_struct(d, o[0], o[1], found.get(d.name, {}))
    todo = [d for d in drafts if d.status == "draft"]
    work = cfg.root / "build" / "dk" / "gen-data"

    def compiled(chosen: list[Draft], slot: str):
        obj, err = compile_variant(cfg, unit, apply(src, chosen), work / slot)
        try:
            return (Elf.parse(obj.read_bytes()), obj.read_bytes()) if obj is not None else (None, err)
        finally:
            shutil.rmtree(work / slot, ignore_errors=True)

    ours, _ = compiled(todo, "all") if todo else (None, "")
    if ours is None and todo:                         # one bad draft spoils the compile: find the good ones
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            alone = list(pool.map(lambda kd: compiled([kd[1]], f"one{kd[0]}")[0], enumerate(todo)))
        for d, o in zip(todo, alone):
            if o is None:
                d.status, d.why = "failed", "does not compile"
        todo = [d for d in todo if d.status == "draft"]
        ours, _ = compiled(todo, "all") if todo else (None, "")
    for d in todo:
        t = tsyms.get(d.name)
        if t is None or ours is None:
            d.status, d.why = "failed", "no target symbol to compare with" if t is None else "does not compile"
            continue
        tsec = target.sections[t.shndx]
        tbytes = tsec.data[t.value:t.value + d.size] if d.section not in ZERO else None
        trel = {r.offset - t.value: resolved(target, t.shndx, r) for r in target.relocations(t.shndx)
                if t.value <= r.offset < t.value + d.size}
        why = verify(ours, tbytes, trel, d)
        d.status, d.why = ("verified", d.why) if why is None else ("failed", why)
    return ud


def report_lines(ud: UnitData, list_verified: bool = False) -> list[str]:
    """gen --data's report: counts per status, the listed drafts and the coverage line."""
    drafts, total = ud.drafts, ud.counted
    out = [f"dk gen --data {ud.rel}: {len(drafts)} data objects, {len(drafts) - len(total)} emitted by the compiler"]
    for status in ("verified", "defined", "failed", "todo"):
        group = [d for d in drafts if d.status == status]
        if group:
            out.append(f"  {status:<9} {len(group):3} objects, {sum(d.size for d in group):#7x} bytes")
    for d in drafts:
        if d.status in ("verified", "failed", "todo") and (list_verified or d.status != "verified"):
            line = f"{d.text[:90]}   [{d.evidence}]" if d.status == "verified" else d.why
            out.append(f"    {d.status:<8} {d.section:<8} {d.name:<22} {line[:170]}")
    done = sum(d.size for d in total if d.status in ("verified", "defined"))
    out.append(f"coverage: {len([d for d in total if d.status in ('verified', 'defined')])}/{len(total)} objects, "
               f"{done:#x}/{sum(d.size for d in total):#x} bytes (verified drafts plus those already defined)")
    return out


def run_data(cfg, ctx, args, audit: bool = False, hits: list | None = None) -> int:
    """dk gen --data <unit>: draft, verify, report coverage, and with --write apply. With audit,
    only collect into `hits` the definitions the source has of what the compiler emits itself."""
    import shutil
    from ...core.command import OK, USAGE, hint
    from .matching import MatchError, resolve_unit, score
    from .reorder import compile_variant
    try:
        unit = resolve_unit(cfg, args.data)
    except MatchError as e:
        print(f"error: {e}")
        return USAGE
    ud = draft_unit(cfg, ctx, unit, args.jobs, audit, hits)
    if ud is None:
        print(f"error: {unit['metadata']['source_path'].removeprefix('src/')} is not in splits.txt")
        return USAGE
    if audit:
        return 0
    print("\n".join(report_lines(ud, getattr(args, 'list', False))))
    verified = ud.verified
    if args.write and verified:
        work = cfg.root / "build" / "dk" / "gen-data"
        new = apply(ud.src, verified)
        obj, err = compile_variant(cfg, unit, new, work / "final")
        if obj is None:
            print("not written: the verified drafts do not compile together")
            return OK
        s_new = score(cfg, unit["name"], cfg.root / unit["target_path"], obj)
        shutil.rmtree(work / "final", ignore_errors=True)
        ud.source.write_text(new)
        print(f"written: {len(verified)} definitions into {cfg.rel(ud.source)}; data now "
              + ", ".join(f"{k} {v:.1f}%" for k, v in sorted(s_new.data.items())))
        hint(f"check {ud.rel} --private", f"save {ud.rel} --note \"data drafted\"")
    return OK


# ---- aggregates: an initialized struct (or array of structs) the source defines ------------------

SCALAR = {"char": 1, "signed char": 1, "unsigned char": 1, "bool": 1, "short": 2, "unsigned short": 2,
          "signed short": 2, "int": 4, "unsigned int": 4, "signed int": 4, "unsigned": 4, "long": 4,
          "unsigned long": 4, "signed long": 4, "float": 4, "double": 8, "long long": 8, "unsigned long long": 8,
          "u8": 1, "s8": 1, "u16": 2, "s16": 2, "u32": 4, "s32": 4, "f32": 4, "f64": 8}
_MEMBER = re.compile(r"^\s*((?:const\s+|volatile\s+|unsigned\s+|signed\s+)*[A-Za-z_][\w:]*(?:\s+(?:int|long|char|short))?"
                     r"(?:\s*\*+)?)\s*(\**)\s*([A-Za-z_]\w*)\s*((?:\[[^\]]+\])*)\s*;")


@dataclass
class Member:
    name: str
    type: str          # base type ("int", "Foo", "char")
    pointer: bool
    dims: list[int]
    offset: int
    size: int          # of one element


def struct_body(src: str, name: str) -> str | None:
    m = re.search(rf"\b(?:struct|class)\s+{re.escape(name)}\s*\{{", src)
    if m is None or re.search(rf"\b(?:struct|class)\s+{re.escape(name)}\s*:", src):
        return None
    close = cbody.matching_brace(src, m.end() - 1)
    return src[m.end():close] if close is not None else None


def layout(src: str, name: str, depth: int = 0) -> tuple[list[Member], int, int] | None:
    """(members, size, alignment) of a plain struct the source defines (no bases, no virtuals)."""
    body = struct_body(src, name)
    if body is None or depth > 6 or "virtual" in body:
        return None
    members, off, align = [], 0, 1
    for line in body.split(";"):
        line = line.strip()
        if not line or line.startswith(("public", "private", "protected", "//", "static", "typedef", "enum")) \
                or "(" in line:
            if "(" in line and "{" not in line and line.split("(")[0].split()[-1:] and not line.startswith("//"):
                continue                               # a method declaration: no storage
            continue
        line = re.sub(r"//[^\n]*", "", line).strip()
        m = _MEMBER.match(line + ";")
        if m is None:
            return None
        typ, stars, mname, dims = m.groups()
        pointer = "*" in typ or bool(stars)
        base = typ.replace("*", "").strip()
        base = re.sub(r"^(?:const|volatile)\s+", "", base)
        if pointer:
            size, al = 4, 4
        elif base in SCALAR:
            size = al = SCALAR[base]
        else:
            inner = layout(src, base, depth + 1)
            if inner is None:
                return None
            size, al = inner[1], inner[2]
        n = [_dim(d) for d in re.findall(r"\[([^\]]+)\]", dims)] if dims else []
        if any(x is None for x in n):
            return None
        off = (off + al - 1) // al * al
        members.append(Member(mname, base, pointer, n, off, size))
        count = 1
        for d in n:
            count *= d
        off += size * count
        align = max(align, al)
    return members, (off + align - 1) // align * align, align


def _dim(text: str) -> int | None:
    """An array dimension: a number or simple integer arithmetic on numbers (`0x2EC - 0x8C`)."""
    text = text.strip()
    if not re.fullmatch(r"[0-9A-Fa-fx\s+\-*/()]+", text):
        return None
    try:
        value = eval(re.sub(r"\b0x[0-9A-Fa-f]+\b", lambda m: str(int(m.group(0), 16)), text).replace("/", "//"),
                     {"__builtins__": {}})
    except Exception:
        return None
    return value if isinstance(value, int) and value >= 0 else None


def _scalar(typ: str, data: bytes, at: int, size: int) -> str | None:
    raw = int.from_bytes(data[at:at + size], "big")
    if typ in ("float", "f32"):
        return _float(raw)
    if typ in ("double", "f64"):
        v = struct.unpack(">d", data[at:at + 8])[0]
        return repr(v) if math.isfinite(v) else None
    signed = not typ.startswith(("unsigned", "u")) and typ != "bool"
    if signed and raw >> (size * 8 - 1):
        raw -= 1 << (size * 8)
    return str(raw)


class Blocked(Exception):
    """Why an aggregate cannot be written from its type (a pad that holds data, a mangled pointer)."""


def initializer(src: str, m_type: str, pointer: bool, size: int, data: bytes, relocs: dict[int, str], at: int,
                needs: list[str], where: str) -> str:
    if pointer:
        target = relocs.get(at)
        if target is None:
            if any(data[at:at + 4]):
                raise Blocked(f"{where}: a pointer that holds a number")
            return "0"
        if not c_name(target):
            raise Blocked(f"{where}: points at the C++ name {target}")
        needs.append(target)
        return target if target.startswith("fn_") else f"&{target}"
    if any(at <= r < at + size for r in relocs):
        raise Blocked(f"{where}: an address inside a {m_type}")
    if m_type in SCALAR:
        v = _scalar(m_type, data, at, size)
        if v is None:
            raise Blocked(f"{where}: not a finite {m_type}")
        return v
    lay = layout(src, m_type)
    if lay is None:
        raise Blocked(f"{where}: {m_type} is not a plain struct of the unit")
    return struct_init(src, lay[0], data, relocs, at, needs, where)


def struct_init(src: str, members: list[Member], data: bytes, relocs: dict[int, str], at: int, needs: list[str],
                where: str) -> str:
    parts = []
    for mem in members:
        pos = at + mem.offset
        count = 1
        for d in mem.dims:
            count *= d
        span = data[pos:pos + mem.size * count]
        is_pad = re.match(r"_?pad", mem.name, re.I) is not None
        if is_pad:
            if any(span) or any(pos <= r < pos + len(span) for r in relocs):
                nz = [i for i, b in enumerate(span) if b]
                raise Blocked(f"{where}: {mem.name} (+{mem.offset:#x}, {len(span):#x} bytes) holds data at "
                              f"+{mem.offset + nz[0]:#x}..+{mem.offset + nz[-1] + 1:#x}: give it fields")
            parts.append("{0}" if mem.dims else "0")
            continue
        if mem.dims and mem.type in ("char", "signed char", "unsigned char") and not mem.pointer:
            text = span.rstrip(b"\0")
            if text and all(32 <= c < 127 for c in text) and len(text) < len(span):
                parts.append('"' + text.decode().replace("\\", "\\\\").replace('"', '\\"') + '"')
                continue
        if mem.dims:
            vals = [initializer(src, mem.type, mem.pointer, mem.size, data, relocs, pos + k * mem.size, needs,
                                f"{where}.{mem.name}[{k}]") for k in range(count)]
            parts.append("{" + ", ".join(vals) + "}")
        else:
            parts.append(initializer(src, mem.type, mem.pointer, mem.size, data, relocs, pos, needs,
                                     f"{where}.{mem.name}"))
    while len(parts) > 1 and parts[-1] in ("0", "{0}"):       # trailing zero members may be left out
        parts.pop()
    return "{" + ", ".join(parts) + "}"


def aggregate(src: str, d: Draft, decl: str, data: bytes, relocs: dict[int, str]) -> None:
    """Fill d (an initialized object the source declares as a struct, or an array of them) with its
    definition, or make it a TODO saying which pad holds data."""
    m = re.match(r"^(?:const\s+)?([A-Za-z_][\w:]*)\s+" + re.escape(d.name) + r"\s*((?:\[[^\]]*\])*)\s*;$", decl.strip())
    if m is None or m.group(1) in SCALAR:
        d.status, d.why = "todo", f"declared as `{decl.strip()}`: not a struct the drafter can fill"
        return
    lay = layout(src, m.group(1))
    if lay is None:
        d.status, d.why = "todo", f"{m.group(1)} is not a plain struct defined in the unit (bases, virtuals, unknown types)"
        return
    members, size, _ = lay
    dims = [_dim(x) for x in re.findall(r"\[([^\]]+)\]", m.group(2) or "")]
    if any(x is None for x in dims):
        d.status, d.why = "todo", f"`{decl.strip()}`: an array size the drafter cannot read"
        return
    count = dims[0] if dims else 1
    if len(dims) > 1 or size * count > d.size or d.size - size * count >= 8 + size:
        d.status, d.why = "todo", f"{m.group(1)} is {size:#x} bytes; {count} of them do not fill {d.size:#x}"
        return
    needs: list[str] = []
    try:
        elems = [struct_init(src, members, data, relocs, k * size, needs, f"{d.name}[{k}]" if dims else d.name)
                 for k in range(count)]
    except Blocked as e:
        d.status, d.why = "todo", str(e)
        return
    d.needs = [f'extern "C" void {t}();' if t.startswith("fn_") else f"extern char {t}[];"
               for t in sorted(set(needs)) if not re.search(rf"\b{re.escape(t)}\b", src)]
    body = ("{\n    " + ",\n    ".join(elems) + ",\n}") if dims else elems[0]
    d.text = decl.strip()[:-1] + " = " + body + ";"


# ---- fields from the code's accesses (a struct the unit does not define) --------------------------

ACCESS = {"lwz": ("int", 4), "stw": ("int", 4), "lwzu": ("int", 4), "stwu": ("int", 4), "lfs": ("float", 4),
          "stfs": ("float", 4), "lfd": ("double", 8), "stfd": ("double", 8), "lhz": ("unsigned short", 2),
          "lha": ("short", 2), "sth": ("short", 2), "lbz": ("unsigned char", 1), "stb": ("unsigned char", 1)}
_BASE = re.compile(r"^(?:addi (r\d+), r\d+, |li (r\d+), )\"?([^\"@\s+]+)\"?(?:\+(0x[0-9A-Fa-f]+|\d+))?@(?:l|sda21)$")
_MEM = re.compile(r"^(\w+) [rf]\d+, (?:(-?0x[0-9A-Fa-f]+|-?\d+)\((r\d+)\)|\"?([^\"@\s+]+)\"?(?:\+(0x[0-9A-Fa-f]+|\d+))?"
                  r"@(?:l|sda21)\((r\d+)\))$")


def accesses(asm: str, address_of: dict[str, int]) -> list[tuple[int, str, int]]:
    """(address, type, width) of each load or store a function makes through a symbol's address:
    the symbol materialized in a register (lis/addi, li @sda21) plus a displacement, or used
    directly (sym@l(rX), sym@sda21(r13)). A register holds its address until it is written again."""
    out, held = [], {}
    for line in asm.splitlines():
        ins = re.sub(r"/\*.*?\*/", "", line).strip()
        if not ins or ins.startswith((".", "#")) or ins.endswith(":"):
            continue
        b = _BASE.match(ins)
        if b and (b.group(3) in address_of):
            reg = b.group(1) or b.group(2)
            held[reg] = address_of[b.group(3)] + (int(b.group(4), 0) if b.group(4) else 0)
            continue
        m = _MEM.match(ins)
        if m and m.group(1) in ACCESS:
            typ, width = ACCESS[m.group(1)]
            if m.group(3) is not None and m.group(3) in held:
                out.append((held[m.group(3)] + int(m.group(2), 0), typ, width))
            elif m.group(4) in address_of:
                out.append((address_of[m.group(4)] + (int(m.group(5), 0) if m.group(5) else 0), typ, width))
            if m.group(1).endswith("u") and m.group(3) in held:
                held.pop(m.group(3), None)
        parts = ins.split(None, 1)
        if len(parts) == 2:
            dest = parts[1].split(",")[0].strip()
            if dest in held and not ins.startswith(("st", "cmp")):
                held.pop(dest, None)            # the register is written again
        if ins.startswith(("bl ", "bctrl")):
            for r in ("r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11", "r12", "r0"):
                held.pop(r, None)
    return out


def infer_struct(d: Draft, data: bytes, relocs: dict[int, str], fields: dict[int, tuple[str, int]]) -> None:
    """Define the object as a struct of the fields its users access (named by offset), with zero
    gaps as pad members; any data no access explains (or a relocation) leaves it a TODO."""
    if relocs:
        d.status, d.why = "todo", "holds addresses at " + ", ".join(f"+{o:#x}" for o in sorted(relocs)[:4])
        return
    members, pos, inits = [], 0, []
    for off in sorted(fields):
        typ, width = fields[off]
        if off < pos or off % width or off + width > d.size:
            d.status, d.why = "todo", f"overlapping or misaligned accesses at +{off:#x}"
            return
        if off > pos:
            if any(data[pos:off]):
                d.status, d.why = "todo", f"+{pos:#x}..+{off:#x} holds data no access explains"
                return
            members.append(f"char _pad{pos:X}[{off - pos:#x}];")
            inits.append("{0}")
        value = _scalar(typ, data, off, width)
        if value is None:
            d.status, d.why = "todo", f"+{off:#x} is not a finite {typ}"
            return
        members.append(f"{typ} m{off:X};")
        inits.append(value)
        pos = off + width
    if pos < d.size and any(data[pos:d.size]):
        d.status, d.why = "todo", f"+{pos:#x}..+{d.size:#x} holds data no access explains"
        return
    if not fields:
        d.status, d.why = "todo", "no access through it in the unit's code"
        return
    while inits and inits[-1] in ("0", "{0}"):
        inits.pop()
    tname = "Unk" + d.name.removeprefix("lbl_")
    d.needs = [f"struct {tname} {{ " + " ".join(members) + " };"]
    d.text = f"{tname} {d.name} = {{{', '.join(inits) or '0'}}};"
    d.evidence = f"a struct of the {len(fields)} field(s) its users access; values from the original"
    d.status = "draft"


def placed(cfg, unit: dict, src: str, target) -> dict[str, str]:
    """Target objects the unit's source already defines, whatever it calls them (a function-local
    static, an @LOCAL@ object, another name): each of our initialized symbols' bytes found at one
    place in the target's section covers the target objects there. target name -> our name."""
    import shutil
    from .elf import Elf
    from .reorder import compile_variant
    work = cfg.root / "build" / "dk" / "gen-data" / "placed"
    obj, _ = compile_variant(cfg, unit, src, work)
    try:
        if obj is None:
            return {}
        ours = Elf.parse(obj.read_bytes())
    finally:
        shutil.rmtree(work, ignore_errors=True)
    out: dict[str, str] = {}
    tsecs = {sec.name: sec for sec in target.sections}
    for s in ours.symbols:
        if not s.name or not 0 < s.shndx < len(ours.sections) or s.size < 4:
            continue
        sec = ours.sections[s.shndx]
        if sec.name not in DATA or sec.name in ZERO or sec.name not in tsecs:
            continue
        mine = sec.data[s.value:s.value + s.size]
        if not any(mine):
            continue
        tdata = tsecs[sec.name].data
        at = tdata.find(mine)
        if at < 0 or tdata.find(mine, at + 1) >= 0:
            continue                                     # not there, or not one place
        tindex = tsecs[sec.name].index
        for t in target.symbols:
            if t.name and t.shndx == tindex and at <= t.value < at + s.size and t.name != s.name:
                out.setdefault(t.name, s.name)
    # zero-filled objects have no bytes to find: map names through the code that addresses them
    # (the same instruction of the same function relocates against target A and our B: A is B)
    defined = {x.name for x in ours.symbols if x.name and 0 < x.shndx < len(ours.sections)
               and ours.sections[x.shndx].name in DATA}
    for tname, oname in same_sites(_sites(ours), _sites(target), defined).items():
        out.setdefault(tname, oname)
    return out


def same_sites(ours_sites: dict, target_sites: dict, defined: set[str]) -> dict[str, str]:
    """target name -> our name, where most instructions that relocate against the target's object
    relocate against one object of ours that the source defines under another name."""
    from collections import Counter
    votes: dict[str, Counter] = {}
    for site, tname in target_sites.items():
        oname = ours_sites.get(site)
        if oname and oname != tname:
            votes.setdefault(tname, Counter())[oname] += 1
    out = {}
    for tname, c in votes.items():
        oname, n = c.most_common(1)[0]
        if oname in defined and n > sum(c.values()) / 2:
            out[tname] = oname
    return out


def _sites(elf) -> dict[tuple[str, int], str]:
    """(function, offset in it) -> the data symbol its relocation reaches (section-relative ones resolved)."""
    funcs = {}
    for x in elf.symbols:
        if x.name and x.type == 2 and 0 < x.shndx < len(elf.sections):
            funcs.setdefault(x.shndx, []).append((x.value, x.value + max(x.size, 1), x.name))
    out = {}
    for sec in elf.sections:
        if sec.type != 4 or not 0 < sec.info < len(elf.sections) or elf.sections[sec.info].name != ".text":
            continue
        for r in elf.relocations(sec.info):
            hit = next(((n, r.offset - lo) for lo, hi, n in funcs.get(sec.info, []) if lo <= r.offset < hi), None)
            if hit is None:
                continue
            name = resolved(elf, sec.info, r)
            if r.target_section in DATA or not r.target_section:
                out[hit] = name
    return out


def run_audit(cfg, ctx, args) -> int:
    """dk gen --data-audit [--since REV]: definitions in the sources of data the compiler emits
    itself (RTTI names and base lists, initializer templates, a table's strings): devices to remove."""
    import subprocess
    from types import SimpleNamespace
    from ...core.command import MISMATCH, OK
    from .matching import load_objdiff_units
    units = [u for u in load_objdiff_units(cfg) if u.get("base_path")]
    if args.since:
        changed = set(subprocess.run(["git", "diff", "--name-only", args.since, "HEAD", "--", "src"], cwd=cfg.root,
                                     capture_output=True, text=True).stdout.split())
        units = [u for u in units if u["metadata"].get("source_path") in changed]
    hits: list = []
    for u in units:
        run_data(cfg, ctx, SimpleNamespace(data=u["metadata"]["source_path"], jobs=args.jobs, write=False,
                                           list=False), audit=True, hits=hits)
    for rel, name, sec, why in hits:
        print(f"  {rel:<40} {name:<24} {sec:<8} {why}")
    print(f"dk gen --data-audit: {len(hits)} definition(s) of compiler-emitted data in {len(units)} unit(s)"
          + (f" changed since {args.since}" if args.since else ""))
    return MISMATCH if hits else OK
