"""mwcc's per-symbol force-active flag (.comment) and what dk check keeps because of it."""

import struct
import tempfile
import unittest
from pathlib import Path

from tools.dk.platforms.gc_mwcc.elf import Elf
from tools.dk.platforms.gc_mwcc.matching import _weak_extras

LOCAL, GLOBAL, WEAK = 0, 1, 2
OBJECT = 1


def make_object(data: bytes, symbols: list[tuple[str, int, int, int]], forced: set[str] = frozenset(),
                comment: bool = True) -> bytes:
    """A minimal mwcc-style relocatable: .data, .symtab, .strtab, optional .comment, .shstrtab.
    symbols: (name, value, size, bind) of objects in .data."""
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
        com += struct.pack(">IBBH", 4, 0, 0x08 if name in forced else 0, 0)
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


if __name__ == "__main__":
    unittest.main()
