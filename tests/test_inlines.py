"""inlines: matching inlined copies on value-numbered code (synthetic functions)."""

from __future__ import annotations

import unittest

from tests.test_asmflow import DTOR, asm
from tools.dk.platforms.gc_mwcc import asmflow, inlines
from tools.dk.platforms.gc_mwcc.asmflow import SP


def prep(name: str, lines: str) -> inlines.Prepared:
    return inlines.prepare(asmflow.flow_of(name, asm(lines)))


# Two locals destroyed one after the other: the copies are rescheduled (the vtable store moves
# after the load), share the hoisted vtable address, use other registers, sit at stack addresses
# (so every displacement shifts), and have lost the null checks and the delete branch.
GIANT = """
stwu r1, -0x60(r1)
mflr r0
stw r0, 0x64(r1)
stw r31, 0x5c(r1)
lis r3, __vt__3Foo@ha
addi r31, r3, __vt__3Foo@l
li r0, 0x0
stw r31, 0x20(r1)
stw r0, 0x24(r1)
stw r31, 0x34(r1)
stw r0, 0x38(r1)
addi r3, r1, 0x20
addi r4, r1, 0x34
bl use_both
lwz r0, 0x24(r1)
stw r31, 0x20(r1)
cmplwi r0, 0x0
beq .L_a
mr r3, r0
bl free_buffer
li r0, 0x0
stw r0, 0x24(r1)
.L_a:
lwz r3, 0x38(r1)
stw r31, 0x34(r1)
cmplwi r3, 0x0
beq .L_b
bl free_buffer
li r4, 0x0
stw r4, 0x38(r1)
.L_b:
lwz r31, 0x5c(r1)
lwz r0, 0x64(r1)
mtlr r0
addi r1, r1, 0x60
blr
"""

# The same locals constructed but never destroyed: no copy of the destructor here.
CTOR_ONLY = """
lis r3, __vt__3Foo@ha
addi r31, r3, __vt__3Foo@l
li r0, 0x0
stw r31, 0x20(r1)
stw r0, 0x24(r1)
addi r3, r1, 0x20
bl use_both
blr
"""


class MatchTest(unittest.TestCase):
    def setUp(self):
        self.t = prep("dtor", DTOR)
        self.g = prep("giant", GIANT)

    def test_finds_both_copies_with_their_bindings(self):
        found = inlines.find_copies({"dtor": self.t}, self.g)
        self.assertEqual(len(found), 2)
        binds = [m.bind[("arg", "r3")] for m in found]
        self.assertEqual(binds, [(SP, 0x20), (SP, 0x34)])
        for m in found:
            # lis, addi, vtable store, load, compare, branch, call, li, store: 9 of 9
            self.assertEqual((m.matched, m.total), (9, 9))
            self.assertEqual(m.calls, (1, 1))
            self.assertEqual(sorted(set(m.folded.values())), ["null", "param"])

    def test_claims_include_the_shared_constants(self):
        found = inlines.find_copies({"dtor": self.t}, self.g)
        hoisted = next(n.i for n in self.g.flow.nodes if n.insn.text == "addi r31, r3, __vt__3Foo@l")
        self.assertTrue(all(hoisted in m.claims for m in found))
        self.assertFalse(any(i in m.claims for m in found for i in self.g.frame))

    def test_constructor_sites_are_not_copies(self):
        g = prep("ctors", CTOR_ONLY)
        self.assertEqual(inlines.find_copies({"dtor": self.t}, g), [])

    def test_deterministic(self):
        a = inlines.find_copies({"dtor": self.t}, self.g)
        b = inlines.find_copies({"dtor": prep("dtor", DTOR)}, prep("giant", GIANT))
        self.assertEqual([(m.span, m.pairs, m.matched) for m in a], [(m.span, m.pairs, m.matched) for m in b])

    def test_overlaps_keep_the_heavier_then_better_ranked(self):
        a = inlines.Match("b_helper", (0, 0), {1: 10, 2: 11}, {}, 5, 5)
        b = inlines.Match("a_helper", (0, 0), {1: 11, 2: 12}, {}, 5, 5)
        c = inlines.Match("big", (0, 0), {1: 12, 2: 13, 3: 14}, {}, 7, 8)
        kept = inlines.resolve([a, b, c], rank=lambda name: 0 if name == "b_helper" else 1)
        self.assertEqual([m.template for m in kept], ["b_helper", "big"])


class LiteralTest(unittest.TestCase):
    def test_pool_symbols_compare_by_contents(self):
        data = {0x80700000: b"\x3f\x80\x00\x00", 0x80600010: b"hello\0" + b"\0" * 58}
        symbols = {"lbl_80700000": (".sdata2", 0x80700000, 4), "lbl_80600000": (".data", 0x80600000, 0x40),
                   "gCount": (".sbss", 0x80710000, 4)}

        def read(addr, n):
            for base, raw in data.items():
                if base <= addr < base + len(raw):
                    return raw[addr - base:addr - base + n]
            raise ValueError(addr)

        canon = inlines.literal_canon(symbols, read)
        self.assertEqual(canon(((("sym", "lbl_80700000")), 0)), (("lit", b"\x3f\x80\x00\x00"), 0))
        self.assertEqual(canon(((("sym", "lbl_80600000")), 0x10)), (("str", "hello"), 0))
        self.assertEqual(canon(((("sym", "gCount")), 0)), (("sym", "gCount"), 0))


PART_A = """
lwz r3, 0x10(r30)
addi r4, r1, 0x20
bl helper_a
stw r3, 0x8(r30)
lwz r0, 0x14(r30)
cmpwi r0, 0x0
beq .L_x
li r5, 0x3
stw r5, 0x18(r30)
.L_x:
lwz r6, 0x1c(r30)
"""
PART_B = """
lwz r3, 0x40(r30)
lfs f1, 0x44(r30)
bl helper_b
stfs f1, 0x48(r30)
addi r3, r1, 0x30
li r4, 0x7
bl helper_c
stw r3, 0x4c(r30)
lwz r7, 0x50(r30)
stw r7, 0x54(r30)
"""


def renamed(text: str) -> str:
    """The same code with another register for the object and other stack slots."""
    return text.replace("r30", "r29").replace("0x20(r1)", "0x60(r1)").replace("r1, 0x20", "r1, 0x60") \
        .replace("r1, 0x30", "r1, 0x70")


def corpus(funcs: dict[str, str]):
    preps = {n: prep(n, text) for n, text in funcs.items()}
    seqs = {n: inlines.block_sequence(p) for n, p in preps.items()}
    toks = {n: [inlines.token(preps[n].flow.nodes[i]) for i in s] for n, s in seqs.items()}
    return preps, seqs, toks, inlines.kgram_index(toks)


class RepeatedBlockTest(unittest.TestCase):
    def test_tokens_abstract_registers_and_stack_slots(self):
        f = prep("f", "lwz r3, 0x10(r30)\naddi r4, r1, 0x20\nstw r5, 0x8(r1)\nbl g")
        self.assertEqual([inlines.token(n) for n in f.flow.nodes],
                         ["lwz R,0x10(R)", "addi R,sp,*", "stw R,sp", "bl g"])

    def test_shared_stretch_is_one_verified_block(self):
        preps, seqs, toks, index = corpus({"f1": PART_A + PART_B, "f2": renamed(PART_A + PART_B)})
        blocks = inlines.check_blocks(inlines.repeated_blocks("f1", toks, index), preps, seqs)
        self.assertEqual([(b.start, b.end) for b in blocks], [(0, 20)])
        seg, = blocks[0].segments
        self.assertEqual(seg.partners, [("f2", 0)])
        self.assertEqual(seg.verified, 1.0)
        self.assertEqual(seg.bind[("arg", "r30")], (("arg", "r29"), 0))
        self.assertEqual(seg.bind[SP], (SP, 0x40))

    def test_cut_where_the_partner_copies_change(self):
        funcs = {"f1": PART_A + PART_B, "f2": renamed(PART_A + PART_B), "f3": PART_B, "f4": renamed(PART_B)}
        preps, seqs, toks, index = corpus(funcs)
        blk, = inlines.check_blocks(inlines.repeated_blocks("f1", toks, index), preps, seqs)
        self.assertEqual([(s.start, s.end) for s in blk.segments], [(0, 10), (10, 20)])
        self.assertEqual(blk.segments[0].partners, [("f2", 0)])
        self.assertEqual(blk.segments[1].partners, [("f2", 0), ("f3", -10), ("f4", -10)])

    def test_runs_of_one_store_are_not_blocks(self):
        zeros = "\n".join(f"stw r0, {0x10 + 4 * i:#x}(r1)" for i in range(12))
        preps, seqs, toks, index = corpus({"f1": zeros, "f2": zeros})
        self.assertEqual(inlines.repeated_blocks("f1", toks, index), [])

    def test_values_must_agree_with_one_offset_per_base(self):
        a = prep("a", "addi r3, r1, 0x20\nbl g\naddi r3, r1, 0x30\nbl g")
        b = prep("b", "addi r3, r1, 0x40\nbl g\naddi r3, r1, 0x90\nbl g")
        seq_a, seq_b = inlines.block_sequence(a), inlines.block_sequence(b)
        share, _ = inlines.same_block(a.flow, seq_a, b.flow, seq_b)
        self.assertEqual(share, 0.5)


if __name__ == "__main__":
    unittest.main()
