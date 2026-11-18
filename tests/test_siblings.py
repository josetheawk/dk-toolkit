"""dk siblings: shared instruction patterns and the constructor detector on synthetic asm (no game data)."""

from __future__ import annotations

import unittest

from tests.test_asmflow import asm
from tools.dk.platforms.gc_mwcc import asmflow, siblings

FRAME = """
stwu r1, -0x20(r1)
mflr r0
stw r0, 0x24(r1)
"""
TAIL = """
lwz r0, 0x24(r1)
mtlr r0
addi r1, r1, 0x20
blr
"""
# A record walk: index * 0x54 + header 0x18 (the distinctive stretch), with an unrelated li between.
WALK = """
lwz r0, 0x4(r3)
mulli r0, r0, 0x54
li r9, 0x0
add r8, r0, r3
addi r8, r8, 0x18
lwz r4, 0x0(r8)
"""
WALK_OTHER_REGS = WALK.replace("r8", "r27").replace("r3", "r30").replace("li r9, 0x0", "li r9, 0x0\nnop")   # one more instruction scheduled in
WALK_OTHER_STRIDE = WALK.replace("0x54", "0x60").replace("0x18", "0x20")
FILLER = "\n".join(f"lwz r{5 + i % 3}, {0x10 + 4 * i:#x}(r31)\nstw r{5 + i % 3}, {0x40 + 4 * i:#x}(r1)"
                   for i in range(6))


def fn(*parts: str) -> str:
    return asm("\n".join(p.strip() for p in parts))


class Normalise(unittest.TestCase):
    def test_registers_renamed_immediates_kept_symbols_masked(self):
        toks = siblings.tokenize(asmflow.parse_insns(asm("add r8, r0, r3\naddi r8, r8, 0x18\nlwz r4, lbl_1@sda21(r0)\nbl foo")))
        self.assertEqual(siblings.window(toks, (0, 1)), "add Ra,Rb,Rc|addi Ra,Ra,0x18")
        self.assertEqual(toks[2].shape, "lwz {},S")
        self.assertEqual(toks[3].shape, "bl S")

    def test_stack_offsets_masked(self):
        toks = siblings.tokenize(asmflow.parse_insns(asm("stw r0, 0x24(r1)")))
        self.assertEqual(siblings.window(toks, (0,)), "stw Ra,N(r1)")


class Query(unittest.TestCase):
    def setUp(self):
        noise = {f"n{i}": fn(FRAME, FILLER.replace("0x10", hex(0x100 + i * 8)), TAIL) for i in range(40)}
        self.texts = {"open": fn(FRAME, FILLER, WALK, TAIL),
                      "same_walk": fn(FRAME, WALK_OTHER_REGS, "lwz r5, 0x8(r4)", TAIL),
                      "other_stride": fn(FRAME, WALK_OTHER_STRIDE, "lwz r5, 0x8(r4)", TAIL), **noise}
        self.pool = set(self.texts) - {"open"}

    def test_sibling_with_same_walk_ranks_first(self):
        ix = siblings.Index(self.texts, self.pool)
        hits = ix.query("open", top=3)
        self.assertEqual(hits[0].function, "same_walk")
        self.assertNotIn("open", [h.function for h in hits])
        span = hits[0].best
        self.assertLessEqual(span.q_lo, 15)               # the walk starts after frame + filler (3 + 12)
        self.assertGreaterEqual(span.q_hi, 18)

    def test_stride_change_scores_lower(self):
        ix = siblings.Index(self.texts, self.pool)
        scores = {h.function: h.score for h in ix.query("open", top=50)}
        self.assertGreater(scores["same_walk"], scores.get("other_stride", 0.0))

    def test_diverging_span_counts_double(self):
        ix = siblings.Index(self.texts, self.pool)
        plain = ix.query("open", top=1)[0]
        boosted = ix.query("open", diverging={17}, top=1)[0]
        self.assertTrue(boosted.best.diverges)
        self.assertAlmostEqual(boosted.best.weight, 2 * plain.best.weight, places=3)

    def test_resolve_by_prefix(self):
        names = {"fn_80001000__3FooFv", "fn_80002000"}
        self.assertEqual(siblings.resolve(names, "fn_80001000"), "fn_80001000__3FooFv")
        self.assertEqual(siblings.resolve(names, "fn_80002000"), "fn_80002000")
        self.assertIsNone(siblings.resolve(names, "fn_8000"))


CTOR = """
stwu r1, -0x10(r1)
mflr r0
stw r0, 0x14(r1)
stw r31, 0xc(r1)
mr. r31, r3
beq .L_end
bl init_parts
stw r0, 0x4(r31)
.L_end:
mr r3, r31
lwz r31, 0xc(r1)
lwz r0, 0x14(r1)
mtlr r0
addi r1, r1, 0x10
blr
"""
NOT_SELF = CTOR.replace("mr r3, r31", "li r3, 0x1")
LEAF_SELF = """
stw r4, 0x0(r3)
blr
"""
CALLER = """
li r3, 0x40
bl alloc
mr. r31, r3
beq .L_skip
mr r4, r30
li r5, 0x1
bl {callee}
mr r31, r3
.L_skip:
mr r3, r31
blr
"""
CALLER_VIRTUAL = CALLER.replace("bl alloc", "mtctr r12\nbctrl")
CALLER_VOID = CALLER.replace("mr r31, r3\n", "nop\n")


class Constructors(unittest.TestCase):
    def test_returns_first_arg(self):
        self.assertTrue(siblings.returns_first_arg(asm(CTOR)))
        self.assertTrue(siblings.returns_first_arg(asm(LEAF_SELF)))
        self.assertFalse(siblings.returns_first_arg(asm(NOT_SELF)))
        self.assertFalse(siblings.returns_first_arg(asm("bl other\nblr")))     # r3 is the callee's result

    def check(self, caller: str, callee_text: str) -> list:
        texts = {"maker": callee_text}
        return siblings.ctor_sites(asm(caller.format(callee="maker")), lambda f: siblings.returns_first_arg(texts[f]))

    def test_flags_new_expression(self):
        sites = self.check(CALLER, asm(CTOR))
        self.assertEqual([(s.alloc, s.ctor) for s in sites], [("alloc", "maker")])
        sites = self.check(CALLER_VIRTUAL, asm(CTOR))
        self.assertEqual([s.alloc for s in sites], ["(virtual)"])

    def test_callee_not_returning_self(self):
        self.assertEqual(self.check(CALLER, asm(NOT_SELF)), [])

    def test_result_not_assigned(self):
        self.assertEqual(self.check(CALLER_VOID, asm(CTOR)), [])

    def test_ctor_name(self):
        self.assertTrue(siblings.is_ctor_name("__ct__3FooFv"))
        self.assertFalse(siblings.is_ctor_name("fn_80001000"))


if __name__ == "__main__":
    unittest.main()
