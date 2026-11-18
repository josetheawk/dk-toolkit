"""dk inlines --repeats: the self-repeat finder (shape, identity, value check, controls) on synthetic code."""

from __future__ import annotations

import unittest

from tests.test_asmflow import asm, flow
from tests.test_inlines import PART_A, PART_B, renamed
from tools.dk.platforms.gc_mwcc import asmflow, inlines, repeats

PROLOGUE = "stwu r1, -0x80(r1)\nmflr r0\nstw r0, 0x84(r1)\nstw r31, 0x7c(r1)\nmr r31, r3\n"
EPILOGUE = "\nlwz r0, 0x84(r1)\nlwz r31, 0x7c(r1)\nmtlr r0\naddi r1, r1, 0x80\nblr\n"
MID = "lwz r9, 0x100(r31)\nmullw r9, r9, r9\nstw r9, 0x104(r31)\nli r0, 0x5\nstw r0, 0x108(r31)\n"


def prep(text: str) -> inlines.Prepared:
    return inlines.prepare(asmflow.flow_of("f", asm(text)))


def two_copies(second: str) -> str:
    return PROLOGUE + PART_A + PART_B + MID + second + EPILOGUE


class FinderTest(unittest.TestCase):
    def test_copy_inside_one_function_is_found(self):
        rep = repeats.analyze("f", prep(two_copies(renamed(PART_A + PART_B))))
        self.assertEqual(len(rep.repeats), 1)
        r = rep.repeats[0]
        self.assertEqual(len(r.starts), 2)
        self.assertGreaterEqual(r.length, 18)
        self.assertGreater(rep.covered(), 2 * 18 - 1)
        self.assertEqual(rep.covered(redundant=True), r.length)

    def test_other_callee_breaks_the_identity_but_not_the_shape(self):
        other = (PART_A + PART_B).replace("helper_b", "helper_z")
        rep = repeats.analyze("f", prep(two_copies(renamed(other))))
        self.assertTrue(all(r.length < 18 for r in rep.repeats))          # the callee cuts the stretch
        shape = repeats.analyze("f", prep(two_copies(renamed(other))), identity=False)
        self.assertGreaterEqual(shape.repeats[0].length, 18)               # shape alone accepts it

    def test_no_repeat_in_distinct_code(self):
        rep = repeats.analyze("f", prep(PROLOGUE + PART_A + MID + PART_B + EPILOGUE))
        self.assertEqual(rep.repeats, [])

    def test_shuffled_control_finds_nothing(self):
        p = prep(two_copies(renamed(PART_A + PART_B)))
        _seq, shapes, ids = repeats.sequence(p)
        self.assertEqual(repeats.find_repeats(repeats.shuffled(list(zip(shapes, ids)), 7)), [])

    def test_find_repeats_longest_first_and_claims_positions_once(self):
        keys = list("abcdefghijk") + ["x1", "x2", "x3"] + list("abcdefghijk") + ["y1", "y2"] + list("abcdefghijk")
        reps = repeats.find_repeats(keys)
        self.assertEqual((reps[0].length, reps[0].starts), (11, [0, 14, 27]))

    def test_value_check_drops_a_copy_that_disagrees(self):
        # the third copy reads another field: same shape except the field offset, so it is not a copy at all
        third = PART_A.replace("0x14(r30)", "0x15(r30)")
        rep = repeats.analyze("f", prep(PROLOGUE + PART_A + MID + renamed(PART_A) + MID + third + EPILOGUE))
        self.assertEqual([len(r.starts) for r in rep.repeats], [2])


class TokenTest(unittest.TestCase):
    def test_shape_drops_symbol_names_and_identity_keeps_them(self):
        f = flow("bl helper_a\nlis r3, lbl_80001234@ha\naddi r3, r3, lbl_80001234@l\nlwz r4, lbl_5@sda21(r13)\nstw r4, 0x20(r1)")
        got = [repeats.shape_and_identity(n) for n in f.nodes]
        self.assertEqual(got[0], ("bl CALL", ("helper_a",)))
        self.assertEqual(got[1], ("lis R,S@ha", ("lbl_80001234",)))
        self.assertEqual(got[2], ("addi R,R,S@l", ("lbl_80001234",)))
        self.assertEqual(got[3][1], ("lbl_5",))
        self.assertEqual(got[4], ("stw R,sp", ()))

    def test_canonical_ids_compare_literals_by_contents(self):
        canon = lambda v: (("lit", b"\x3f\x80\x00\x00"), 0) if v[0][1] in ("@1", "@2") else v
        self.assertEqual(repeats.canonical_ids(("@1",), canon), repeats.canonical_ids(("@2",), canon))
        self.assertNotEqual(repeats.canonical_ids(("@1",), canon), repeats.canonical_ids(("named",), canon))


class CommandTest(unittest.TestCase):
    def test_repeats_mode_prints_the_blocks(self):
        import argparse
        import contextlib
        import io
        from tests.test_inlinescmd import context
        args = argparse.Namespace(shape_only=False, no_values=False, limit=5, json=None)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            from tools.dk.platforms.gc_mwcc import inlinescmd
            inlinescmd.run_repeats(context(), ["fn_giant"], args)
        self.assertIn("# fn_giant", out.getvalue())


if __name__ == "__main__":
    unittest.main()
