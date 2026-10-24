import unittest

from tools.dk.platforms.gc_mwcc.unitcommit import compose_splits, compose_symbols, compose_units, unit_ranges

UNITS_HEAD = """# h

[[lib]]
name = "game"
objects = [
    { path = "a.cpp", status = "Matching" },
    { path = "b.cpp", status = "NonMatching" },
]
"""
UNITS_WORK = """# h

[[lib]]
name = "game"
objects = [
    { path = "a.cpp", status = "Matching" },
    { path = "b.cpp", status = "Matching" },
    { path = "c.cpp", status = "NonMatching" },
]
"""

SPLITS_HEAD = "Sections:\n\t.text       type:code align:32\n\na.cpp:\n\t.text       start:0x80003000 end:0x80003100\n"
SPLITS_WORK = (SPLITS_HEAD + "\nb.cpp:\n\t.text       start:0x80003100 end:0x80003200\n"
               "\nc.cpp:\n\t.text       start:0x80003200 end:0x80003300\n")

SYMS_HEAD = ("fn_80003000 = .text:0x80003000; // type:function size:0x100\n"
             "fn_80003100 = .text:0x80003100; // type:function size:0x100\n"
             "fn_80003200 = .text:0x80003200; // type:function size:0x100\n"
             "fn_80009000 = .text:0x80009000; // type:function size:0x10\n")
SYMS_WORK = ("fn_80003000 = .text:0x80003000; // type:function size:0x100\n"
             "Bfunc__1BFv = .text:0x80003100; // type:function size:0x100\n"     # this unit (b) renamed its own
             "Cfunc__1CFv = .text:0x80003200; // type:function size:0x100\n"     # another agent's unit (c)
             "Callee = .text:0x80009000; // type:function size:0x10\n")          # callee b renamed


class ComposeTest(unittest.TestCase):
    def test_units_takes_only_this_unit(self):
        out = compose_units(UNITS_HEAD, UNITS_WORK, "b.cpp")
        self.assertIn('{ path = "b.cpp", status = "Matching" }', out)
        self.assertNotIn("c.cpp", out)                                   # other agent's new unit stays out
        self.assertLess(out.index("a.cpp"), out.index("b.cpp"))          # order kept

    def test_splits_takes_only_this_block(self):
        out = compose_splits(SPLITS_HEAD, SPLITS_WORK, "b.cpp")
        self.assertIn("b.cpp:\n\t.text       start:0x80003100 end:0x80003200", out)
        self.assertNotIn("c.cpp", out)
        self.assertEqual(unit_ranges(SPLITS_WORK, "b.cpp"), [(".text", 0x80003100, 0x80003200)])

    def test_symbols_in_range_or_named(self):
        out = compose_symbols(SYMS_HEAD, SYMS_WORK, [(".text", 0x80003100, 0x80003200)], {"Callee"})
        self.assertIn("Bfunc__1BFv = .text:0x80003100", out)            # inside this unit's range
        self.assertIn("Callee = .text:0x80009000", out)                 # named in this unit's files
        self.assertIn("fn_80003200 = .text:0x80003200", out)            # other unit's rename stays out
        self.assertNotIn("Cfunc__1CFv", out)


if __name__ == "__main__":
    unittest.main()
