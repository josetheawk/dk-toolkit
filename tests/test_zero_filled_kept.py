"""What dk check keeps next to weak copies the original linker zero-filled in place."""

import struct
import tempfile
import unittest
from pathlib import Path

from tools.dk.platforms.gc_mwcc.matching import _kept_beside_zero_filled, _weak_extras

LOCAL, GLOBAL, WEAK = 0, 1, 2
OBJECT = 1
R_PPC_ADDR32 = 1


def make_object(sections: dict[str, bytes], symbols: list[tuple[str, str, int, int, int]],
                relocs: list[tuple[str, int, str]] = ()) -> bytes:
    """A minimal relocatable: progbits `sections` (in order), their .rela sections, .symtab,
    .strtab, .shstrtab. symbols: (name, section, value, size, bind); relocs: (section, offset,
    symbol name), all R_PPC_ADDR32 with addend 0."""
    names = list(sections)
    index = {name: i + 1 for i, name in enumerate(names)}
    strtab = b"\0"
    entries = [struct.pack(">IIIBBH", 0, 0, 0, 0, 0, 0)]
    symbol_index = {}
    # locals first, as ELF requires
    for name, sec, value, size, bind in sorted(symbols, key=lambda s: s[4] != LOCAL):
        symbol_index[name] = len(entries)
        entries.append(struct.pack(">IIIBBH", len(strtab), value, size, (bind << 4) | OBJECT, 0, index[sec]))
        strtab += name.encode() + b"\0"
    n_local = 1 + sum(1 for s in symbols if s[4] == LOCAL)
    symtab_index = len(names) * 2 + 1
    bodies = [(name, 1, sections[name], 0, 0, 4, 0) for name in names]
    for name in names:
        rela = b"".join(struct.pack(">IIi", off, (symbol_index[sym] << 8) | R_PPC_ADDR32, 0)
                        for sec, off, sym in relocs if sec == name)
        bodies.append((".rela" + name, 4, rela, symtab_index, index[name], 4, 12))
    bodies.append((".symtab", 2, b"".join(entries), symtab_index + 1, n_local, 4, 16))
    bodies.append((".strtab", 3, strtab, 0, 0, 1, 0))
    shstr = b"\0"
    blob = b""
    placed = []
    for name, typ, body, link, info, align, entsize in bodies:
        placed.append((len(shstr), 0x34 + len(blob), name, typ, body, link, info, align, entsize))
        shstr += name.encode() + b"\0"
        blob += body
    shstr_name = len(shstr)
    shstr += b".shstrtab\0"
    shstr_off = 0x34 + len(blob)
    blob += shstr
    shoff = 0x34 + len(blob)
    headers = struct.pack(">10I", *([0] * 10))
    for nm, off, _name, typ, body, link, info, align, entsize in placed:
        headers += struct.pack(">10I", nm, typ, 0, 0, off, len(body), link, info, align, entsize)
    headers += struct.pack(">10I", shstr_name, 3, 0, 0, shstr_off, len(shstr), 0, 0, 1, 0)
    count = len(bodies) + 2
    head = b"\x7fELF\x01\x02\x01" + b"\0" * 9 + struct.pack(">HHIIIIIHHHHHH", 1, 20, 1, 0, 0, shoff, 0,
                                                              0x34, 0, 0, 40, count, count - 1)
    return head + blob + headers


class KeptBesideZeroFilledTest(unittest.TestCase):
    def test_base_list_keeps_base_rtti(self):
        # .data: a local base-class list, then a weak vtable the link zero-fills.
        # .sdata: the derived and base class names, the base RTTI (kept through the list) and
        # the derived RTTI (reachable only from the zero-filled vtable: dropped).
        base = make_object(
            {".data": b"\0" * 24, ".sdata": b"\0" * 32},
            [("@list", ".data", 0, 12, LOCAL), ("__vt__1A", ".data", 12, 12, WEAK),
             ("@nameA", ".sdata", 0, 8, LOCAL), ("@nameObj", ".sdata", 8, 8, LOCAL),
             ("__RTTI__3Obj", ".sdata", 16, 8, LOCAL), ("__RTTI__1A", ".sdata", 24, 8, LOCAL)],
            [(".data", 0, "__RTTI__3Obj"), (".data", 12, "__RTTI__1A"),
             (".sdata", 16, "@nameObj"), (".sdata", 24, "@nameA"), (".sdata", 28, "@list")])
        target = make_object({".data": b"\0" * 24, ".sdata": b"\0" * 16},
                             [("lbl_80001000", ".data", 0, 24, GLOBAL), ("lbl_80002000", ".sdata", 0, 16, GLOBAL)])
        with tempfile.TemporaryDirectory() as d:
            t, b = Path(d) / "t.o", Path(d) / "b.o"
            t.write_bytes(target)
            b.write_bytes(base)
            weak, zero_fillable = _weak_extras(t, b)
            kept = _kept_beside_zero_filled(b, weak, zero_fillable)
        self.assertIn("__RTTI__1A", zero_fillable)
        self.assertEqual(kept, {"@list", "__RTTI__3Obj", "@nameObj"})


if __name__ == "__main__":
    unittest.main()
