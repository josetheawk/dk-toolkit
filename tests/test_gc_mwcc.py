import struct
import tempfile
import unittest
from pathlib import Path

from tests.helpers import make_dol, make_project
from tools.dk.platforms.gc_mwcc.dol import Dol
from tools.dk.platforms.gc_mwcc.dtk_symbols import SymbolIndex, read_symbols, read_units
from tools.dk.platforms.gc_mwcc.firstdiff import first_diff

TEXT = struct.pack(">4I", 0x38600000, 0x4E800020, 0x38600001, 0x4E800020)
DATA = b"\x3f\x80\x00\x00" * 2

SYMBOLS = """\
fn_80003100 = .text:0x80003100; // type:function size:0x8 scope:global align:4
fn_80003108 = .text:0x80003108; // type:function size:0x8
lbl_80004000 = .rodata:0x80004000; // type:object size:0x8 scope:local data:float
"""

SPLITS = """\
Sections:
\t.text       type:code align:4

game/a.cpp:
\t.text       start:0x80003100 end:0x80003108
\t.rodata     start:0x80004000 end:0x80004008

game/b.cpp:
\t.text       start:0x80003108 end:0x80003110
"""


class DolTest(unittest.TestCase):
    def test_parse_and_read(self):
        dol = Dol.parse(make_dol([("T1", 0x80003100, TEXT), ("D4", 0x80004000, DATA)], bss=(0x80005000, 0x40), entry=0x80003100))
        self.assertEqual([s.slot for s in dol.sections], ["T1", "D4"])
        self.assertEqual(dol.entry, 0x80003100)
        self.assertEqual(dol.read(0x80003104, 4), struct.pack(">I", 0x4E800020))
        self.assertEqual(dol.address_of_offset(0x100 + 8), 0x80003108)
        self.assertIsNone(dol.section_at(0x80009000))
        self.assertEqual(dol.words(dol.sections[0])[1], 0x4E800020)

    def test_rejects_truncated(self):
        with self.assertRaises(ValueError):
            Dol.parse(b"\0" * 16)
        bad = bytearray(make_dol([("T0", 0x80003100, TEXT)]))
        with self.assertRaises(ValueError):
            Dol.parse(bytes(bad[:-4]))


class SymbolsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "symbols.txt").write_text(SYMBOLS)
        (self.root / "splits.txt").write_text(SPLITS)

    def tearDown(self):
        self.tmp.cleanup()

    def test_symbols(self):
        syms = read_symbols(self.root / "symbols.txt")
        self.assertEqual([s.name for s in syms], ["fn_80003100", "fn_80003108", "lbl_80004000"])
        self.assertEqual(syms[0].size, 8)
        self.assertEqual(syms[0].type, "function")
        index = SymbolIndex(syms)
        self.assertEqual(index.describe(0x8000310C), ".text fn_80003108+0x4")
        self.assertEqual(index.describe(0x80000000), "0x80000000")

    def test_units(self):
        units = read_units(self.root / "splits.txt")
        self.assertEqual([u.name for u in units], ["game/a.cpp", "game/b.cpp"])
        self.assertEqual(units[0].splits[1], (".rodata", 0x80004000, 0x80004008))


class FirstDiffTest(unittest.TestCase):
    def test_reports_symbol_and_words(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = make_project(root, files={"config/symbols.txt": SYMBOLS})
            orig = root / "orig.dol"
            built = root / "built.dol"
            orig.write_bytes(make_dol([("T1", 0x80003100, TEXT), ("D4", 0x80004000, DATA)]))
            changed = bytearray(TEXT)
            changed[12:16] = struct.pack(">I", 0x60000000)
            built.write_bytes(make_dol([("T1", 0x80003100, bytes(changed)), ("D4", 0x80004000, DATA)]))
            lines = first_diff(cfg, built, orig)
            self.assertEqual(len(lines), 1)
            self.assertIn("0x8000310C", lines[0])
            self.assertIn("fn_80003108+0x4", lines[0])
            self.assertIn("original 4e800020", lines[0])
            self.assertIn("built 60000000", lines[0])

    def test_layout_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = make_project(root)
            orig = root / "orig.dol"
            built = root / "built.dol"
            orig.write_bytes(make_dol([("T1", 0x80003100, TEXT)]))
            built.write_bytes(make_dol([("T1", 0x80003100, TEXT + TEXT)]))
            lines = first_diff(cfg, built, orig)
            self.assertTrue(lines[0].startswith("section layout differs"))


class GoldenTest(unittest.TestCase):
    """Checks against the real original binary; skipped when it is not present."""

    def test_original_dol(self):
        from tools.dk.core.config import load
        from tools.dk.core.util import sha1_file

        from tools.dk.core.config import ConfigError
        try:
            cfg = load()
        except ConfigError:
            self.skipTest("no project here (the toolkit repo has no dk.toml)")
        if not cfg.original.is_file():
            self.skipTest("original binary not present")
        self.assertEqual(sha1_file(cfg.original), cfg.sha1)
        dol = Dol.parse(cfg.original.read_bytes())
        self.assertTrue(dol.sections)
        self.assertIsNotNone(dol.section_at(dol.entry))


if __name__ == "__main__":
    unittest.main()
