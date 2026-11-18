"""mwcc's per-symbol force-active flag (.comment) and what dk check keeps because of it."""

import struct
import tempfile
import unittest
from pathlib import Path

from tools.dk.platforms.gc_mwcc.elf import Elf
from tools.dk.platforms.gc_mwcc.matching import _packed_offsets, _weak_extras

LOCAL, GLOBAL, WEAK = 0, 1, 2
OBJECT = 1


def make_object(data: bytes, symbols: list[tuple[str, int, int, int]], forced: set[str] = frozenset(),
                comment: bool = True, aligns: dict[str, int] | None = None) -> bytes:
    """A minimal mwcc-style relocatable: .data, .symtab, .strtab, optional .comment, .shstrtab.
    symbols: (name, value, size, bind) of objects in .data; aligns: recorded alignment (default 4)."""
    strtab = b"\0"
    entries = [struct.pack(">IIIBBH", 0, 0, 0, 0, 0, 0)]
    names = [""]
    for name, value, size, bind in symbols:
        entries.append(struct.pack(">IIIBBH", len(strtab), value, size, (bind << 4) | OBJECT, 0, 1))
        strtab += name.encode() + b"\0"
        names.append(name)
    symtab = b"".join(entries)
    com = b"CodeWarrior\n" + b"\0" * (0x2C - 12)
    for name in names:
        com += struct.pack(">IBBH", (aligns or {}).get(name, 4), 0, 0x08 if name in forced else 0, 0)
    bodies = [(".data", 1, data, 0, 0), (".symtab", 2, symtab, 3, 1), (".strtab", 3, strtab, 0, 0)]
    if comment:
        bodies.append((".comment", 1, com, 0, 0))
    shstr = b"\0"
    offsets = []
    blob = b"\0" * 0x34
    for name, _typ, body, _link, _info in bodies:
        offsets.append((len(shstr), len(blob)))
        shstr += name.encode() + b"\0"
        blob += body
    shstr_name = len(shstr)
    shstr += b".shstrtab\0"
    shstr_off = len(blob)
    blob += shstr
    shoff = len(blob)
    headers = struct.pack(">10I", *([0] * 10))
    for (nm, off), (_name, typ, body, link, info) in zip(offsets, bodies):
        headers += struct.pack(">10I", nm, typ, 0, 0, off, len(body), link, info, 4, 16 if typ == 2 else 0)
    headers += struct.pack(">10I", shstr_name, 3, 0, 0, shstr_off, len(shstr), 0, 0, 1, 0)
    head = b"\x7fELF\x01\x02\x01" + b"\0" * 9 + struct.pack(">HHIIIIIHHHHHH", 1, 20, 1, 0, 0, shoff, 0,
                                                              0x34, 0, 0, 40, len(bodies) + 2, len(bodies) + 1)
    return head + blob[0x34:] + headers


class ForceActiveTest(unittest.TestCase):
    def test_reads_flag(self):
        obj = make_object(b"\0" * 8, [("@1", 0, 4, LOCAL), ("@2", 4, 4, LOCAL)], forced={"@2"})
        self.assertEqual(Elf.parse(obj).force_active(), {"@2"})

    def test_no_comment_means_none(self):
        obj = make_object(b"\0" * 8, [("@1", 0, 4, LOCAL)], forced={"@1"}, comment=False)
        self.assertEqual(Elf.parse(obj).force_active(), set())

    def test_check_keeps_forced_unreferenced_locals(self):
        # Target has no names for them (dtk labels), nothing references them in the unit:
        # unflagged they count as dropped by the link, flagged they stay.
        target = make_object(b"\0" * 8, [("lbl_80001000", 0, 8, GLOBAL)])
        with tempfile.TemporaryDirectory() as d:
            t = Path(d) / "t.o"
            t.write_bytes(target)
            plain, forced = Path(d) / "a.o", Path(d) / "b.o"
            syms = [("@1", 0, 4, LOCAL), ("@2", 4, 4, LOCAL)]
            plain.write_bytes(make_object(b"\0" * 8, syms))
            forced.write_bytes(make_object(b"\0" * 8, syms, forced={"@2"}))
            dropped_plain = {n for _, _, n in _weak_extras(t, plain)[0].get(".data", [])}
            dropped_forced = {n for _, _, n in _weak_extras(t, forced)[0].get(".data", [])}
        self.assertEqual(dropped_plain, {"@1", "@2"})
        self.assertEqual(dropped_forced, {"@1"})


class RecordedAlignmentTest(unittest.TestCase):
    """The link packs kept objects at the alignment mwcc recorded in .comment, not at whatever
    power of two happens to divide their offset in the compiled object."""

    SYMS = [("@1", 0, 12, LOCAL), ("__vt__1A", 12, 12, WEAK), ("@2", 24, 12, LOCAL)]

    def test_reads_alignment(self):
        obj = Elf.parse(make_object(b"\0" * 36, self.SYMS, aligns={"@2": 8}))
        by_name = {obj.symbols[i].name: a for i, a in obj.alignments().items()}
        self.assertEqual((by_name["@1"], by_name["@2"]), (4, 8))
        self.assertEqual(Elf.parse(make_object(b"\0" * 36, self.SYMS, comment=False)).alignments(), {})

    def test_packs_at_recorded_alignment(self):
        # @2 sits at 0x18 (8-aligned by chance); recorded 4, it packs right after @1
        recorded = Elf.parse(make_object(b"\0" * 36, self.SYMS))
        self.assertEqual(_packed_offsets(recorded, ".data", {"__vt__1A"}), {"@1": 0, "@2": 12})
        guessed = Elf.parse(make_object(b"\0" * 36, self.SYMS, comment=False))
        self.assertEqual(_packed_offsets(guessed, ".data", {"__vt__1A"}), {"@1": 0, "@2": 16})


def make_relocatable(sections: list[tuple[str, int, bytes]], symbols: list[tuple[str, int, int, int, int, str]],
                     relocs: dict[str, list[tuple[int, int, str]]]) -> bytes:
    """A relocatable with several sections. sections: (name, type, body); symbols: (name, value,
    size, bind, type, section name); relocs: section name -> (offset, type, symbol name)."""
    names = [n for n, _, _ in sections]
    strtab, entries, index = b"\0", [struct.pack(">IIIBBH", 0, 0, 0, 0, 0, 0)], {}
    for name, value, size, bind, typ, sec in symbols:
        index[name] = len(entries)
        entries.append(struct.pack(">IIIBBH", len(strtab), value, size, (bind << 4) | typ, 0, names.index(sec) + 1))
        strtab += name.encode() + b"\0"
    bodies = [(n, t, b, 0, 0) for n, t, b in sections]
    symtab_at = len(bodies) + 1
    bodies += [(".symtab", 2, b"".join(entries), symtab_at + 1, 1), (".strtab", 3, strtab, 0, 0)]
    for sec, rows in relocs.items():
        body = b"".join(struct.pack(">IIi", off, (index[sym] << 8) | typ, 0) for off, typ, sym in rows)
        bodies.append((".rela" + sec, 4, body, symtab_at, names.index(sec) + 1))
    shstr, offsets, blob = b"\0", [], b"\0" * 0x34
    for name, _typ, body, _link, _info in bodies:
        offsets.append((len(shstr), len(blob)))
        shstr += name.encode() + b"\0"
        blob += body
    shstr_name, shstr_off = len(shstr), len(blob)
    shstr += b".shstrtab\0"
    blob += shstr
    shoff = len(blob)
    headers = struct.pack(">10I", *([0] * 10))
    for (nm, off), (_name, typ, body, link, info) in zip(offsets, bodies):
        headers += struct.pack(">10I", nm, typ, 0, 0, off, len(body), link, info, 4, {2: 16, 4: 12}.get(typ, 0))
    headers += struct.pack(">10I", shstr_name, 3, 0, 0, shstr_off, len(shstr), 0, 0, 1, 0)
    head = b"\x7fELF\x01\x02\x01" + b"\0" * 9 + struct.pack(">HHIIIIIHHHHHH", 1, 20, 1, 0, 0, shoff, 0,
                                                              0x34, 0, 0, 40, len(bodies) + 2, len(bodies) + 1)
    return head + blob[0x34:] + headers


class SectionBaseTest(unittest.TestCase):
    """A kept function that copies a local array's initializer addresses the pool from the
    section start ("...rodata.0"+0, then register arithmetic): the table has no relocation of
    its own, yet the link keeps it, and with it the strings it points at."""

    def test_pool_reached_from_section_start_is_kept(self):
        FUNC, SECTION = 2, 3
        text, rodata, data = (".text", 1, bytes(8)), (".rodata", 1, bytes(16)), (".data", 1, b"abc\0")
        ours = make_relocatable(
            [text, rodata, data],
            [("f", 0, 8, GLOBAL, FUNC, ".text"), ("...rodata.0", 0, 0, LOCAL, SECTION, ".rodata"),
             ("@1", 0, 8, LOCAL, OBJECT, ".rodata"), ("@2", 8, 8, LOCAL, OBJECT, ".rodata"),
             ("@3", 0, 4, LOCAL, OBJECT, ".data")],
            {".text": [(2, 6, "...rodata.0")], ".rodata": [(8, 1, "@3")]})
        target = make_relocatable(
            [text, rodata, data],
            [("f", 0, 8, GLOBAL, FUNC, ".text"), ("lbl_80001000", 0, 16, GLOBAL, OBJECT, ".rodata"),
             ("lbl_80002000", 0, 4, GLOBAL, OBJECT, ".data")], {})
        with tempfile.TemporaryDirectory() as d:
            t, b = Path(d) / "t.o", Path(d) / "b.o"
            t.write_bytes(target)
            b.write_bytes(ours)
            dropped = {n for ranges in _weak_extras(t, b)[0].values() for _, _, n in ranges}
        self.assertFalse(dropped & {"@1", "@2", "@3"})


if __name__ == "__main__":
    unittest.main()
