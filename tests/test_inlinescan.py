"""dk inlines --scan: inline-boundary fingerprints on synthetic target asm (no game data)."""

from __future__ import annotations

import unittest

from tests.test_asmflow import DTOR, asm
from tests.test_inlinescmd import entry
from tools.dk.platforms.gc_mwcc import asmflow, inlinescan, inlinescmd

ZERO_LOOP = """
li r30, 0x0
lwz r29, 0x8(r3)
mr r31, r30
.L_loop:
lwzx r0, r29, r31
addi r31, r31, 0x4
addi r30, r30, 0x1
cmpw r30, r4
blt .L_loop
blr
"""

ZERO_LOOP_125 = ZERO_LOOP.replace("mr r31, r30", "addi r31, r30, 0x0")

TWO_ZEROS = """
li r30, 0x0
li r31, 0x0
.L_loop:
addi r31, r31, 0x4
addi r30, r30, 0x1
cmpw r30, r4
blt .L_loop
blr
"""

REREAD = """
lwz r4, 0x10(r31)
addi r5, r4, 0x1
lwz r6, 0x10(r31)
lwz r7, 0x14(r31)
lwz r8, 0x14(r31)
blr
"""

REREAD_AFTER_STORE = """
lwz r4, 0x10(r31)
stw r5, 0x0(r30)
lwz r6, 0x10(r31)
lwz r7, 0x8(r1)
lwz r8, 0x8(r1)
blr
"""


def marks(lines: str) -> list[inlinescan.Mark]:
    return inlinescan.fingerprints(asm(lines))


class FingerprintTest(unittest.TestCase):
    def test_zero_copy_starts_a_loop_on_both_compilers(self):
        for text in (ZERO_LOOP, ZERO_LOOP_125):
            m, = marks(text)
            self.assertEqual((m.kind, m.start, m.loop), ("zero-copy", 0x0, True))
            self.assertEqual(m.end, 0x20)                    # through the backward branch
            self.assertEqual(m.params, ["r29", "r4"])        # the helper's inputs
            self.assertIn("inline helper taking (r29, r4)", m.shape())
            self.assertTrue(inlinescan.flagged([m]))

    def test_plain_code_loads_the_zero_twice(self):
        self.assertEqual(marks(TWO_ZEROS), [])

    def test_zero_redefined_before_the_copy_is_not_a_mark(self):
        self.assertEqual(marks("li r30, 0x0\naddi r30, r3, 0x4\nmr r31, r30\nblr"), [])

    def test_reread_in_one_block_is_weak(self):
        ms = marks(REREAD)
        self.assertEqual([(m.kind, m.start) for m in ms], [("reread", 0x8), ("reread", 0x10)])
        self.assertFalse(inlinescan.flagged(ms[:1]))
        self.assertTrue(inlinescan.flagged(ms))             # two weak marks flag

    def test_store_and_stack_reloads_are_not_rereads(self):
        self.assertEqual(marks(REREAD_AFTER_STORE), [])


class ScanTest(unittest.TestCase):
    def test_scan_names_the_exact_callee_and_skips_its_own_template(self):
        body = DTOR.replace("blr", "nop") if "blr" in DTOR else DTOR
        entries = {"__dt__3FooFv": entry("__dt__3FooFv", DTOR, "a.s"),
                   "fn_open": entry("fn_open", "stwu r1, -0x30(r1)\n" + body, "b.s")}
        ctx = inlinescmd.Context(cfg=None, entries=entries, unit_of_file={}, exact={"__dt__3FooFv"},
                                 sizes={"fn_open": 0x80, "__dt__3FooFv": 0x60}, canon=lambda v: v)
        row, = inlinescan.scan(ctx, ["fn_open"], {"fn_open": 50.0}, jobs=1)
        self.assertEqual(row["at_stake"], 0x40)
        bodies = [m for m in row["marks"] if m["kind"] == "body"]
        self.assertTrue(bodies and bodies[0]["callee"] == "__dt__3FooFv", row["marks"])
        self.assertTrue(row["flagged"])
        self.assertIn("call exact __dt__3FooFv inline", bodies[0]["shape"])
        text = "\n".join(inlinescan.render([row], 5))
        self.assertIn("1 of 1 open functions", text)

    def test_reads_ignore_destinations(self):
        ins = asmflow.parse_insns(asm("stw r5, 0x4(r6)\nmr r7, r8"))
        self.assertEqual(inlinescan._reads(ins[0]), {"r5", "r6"})
        self.assertEqual(inlinescan._reads(ins[1]), {"r8"})


class DivergenceTest(unittest.TestCase):
    def test_rows_that_differ_mark_the_region(self):
        def row(addr, text, kind=None):
            r = {"instruction": {"address": str(addr), "formatted": text}}
            if kind:
                r["diff_kind"] = kind
            return r
        target = [row(100, "li r30, 0x0"), row(104, "mr r31, r30"), row(108, "blr")]
        ours = [row(200, "li r30, 0x0"), row(204, "li r31, 0x0", "DIFF_OP"), row(208, "blr")]
        self.assertEqual(inlinescan.diverging_offsets(target, ours), [4])
        rows = [{"function": "f", "marks": [{"start": 0, "end": 8}, {"start": 0x40, "end": 0x48}]}]
        inlinescan.mark_divergence(rows, {"f": [4]})
        self.assertEqual([m["differs"] for m in rows[0]["marks"]], [True, False])
        self.assertTrue(rows[0]["differs_at_mark"])


if __name__ == "__main__":
    unittest.main()
