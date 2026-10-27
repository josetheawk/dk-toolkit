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


class ThunkRenameTest(unittest.TestCase):
    def test_a_renamed_thunk_travels_with_its_function_name(self):
        head = "@164@8@vf14__Q22AI12CopBaseProtoFP11Unk800F11CCPCc = .text:0x800C0000; // type:function size:0x8\n"
        work = "@164@8@vf14__Q22AI12CopBaseProtoFP8PProfilePCc = .text:0x800C0000; // type:function size:0x8\n"
        out = compose_symbols(head, work, [(".text", 0x800F0000, 0x800F1000)], {"vf14__Q22AI12CopBaseProtoFP8PProfilePCc"})
        self.assertIn("FP8PProfilePCc", out)


if __name__ == "__main__":
    unittest.main()


class MergeTest(unittest.TestCase):
    def test_a_merged_unit_leaves_with_the_unit_that_absorbed_it(self):
        from tools.dk.platforms.gc_mwcc.unitcommit import absorbed_units, compose_splits, compose_units
        head = ("Sections:\n\t.text type:code\n\n"
                "game/a.cpp:\n\t.text       start:0x80001000 end:0x80001100\n\n"
                "game/b.cpp:\n\t.text       start:0x80001100 end:0x80001200\n\n"
                "game/c.cpp:\n\t.text       start:0x80002000 end:0x80002100\n")
        work = ("Sections:\n\t.text type:code\n\n"
                "game/a.cpp:\n\t.text       start:0x80001000 end:0x80001200\n\n"
                "game/c.cpp:\n\t.text       start:0x80002000 end:0x80002100\n")
        self.assertEqual(absorbed_units(head, work, "game/a.cpp"), ["game/b.cpp"])
        merged = compose_splits(head, work, "game/a.cpp", ("game/b.cpp",))
        self.assertNotIn("game/b.cpp", merged)
        self.assertIn("start:0x80001000 end:0x80001200", merged)
        units_head = ('[[lib]]\nname = "game"\nobjects = [\n    { path = "game/a.cpp", status = "Matching" },\n'
                      '    { path = "game/b.cpp", status = "NonMatching" },\n]\n')
        units_work = '[[lib]]\nname = "game"\nobjects = [\n    { path = "game/a.cpp", status = "Matching" },\n]\n'
        self.assertNotIn("game/b.cpp", compose_units(units_head, units_work, "game/a.cpp", ("game/b.cpp",)))


class HeaderChurnTest(unittest.TestCase):
    """Another agent's half-done edit to a shared header rides along in a unit's commit; leave
    out the edits the unit does not need, keep the ones its source needs."""

    def test_leaves_out_unneeded_edits_and_keeps_needed_ones(self):
        from tools.dk.platforms.gc_mwcc.unitcommit import choose_headers
        a, b = "include/a.h", "include/b.h"

        def unit_needs_a(kept):            # a.h adds a member the unit uses; b.h is another agent's edit
            if a not in kept:
                return None                # does not compile without a.h's edit
            return (5, 0) if b in kept else (8, 1)
        self.assertEqual(choose_headers([a, b], unit_needs_a), {a})

        def needs_none(kept):
            return (8, 1) if b not in kept else (3, 0)
        self.assertEqual(choose_headers([a, b], needs_none), set())
        self.assertEqual(choose_headers([], needs_none), set())

        def needs_both(kept):
            return (9, 1) if kept == {a, b} else None
        self.assertEqual(choose_headers([a, b], needs_both), {a, b})
