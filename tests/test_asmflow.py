"""asmflow: dtk asm parsing and value numbering, on synthetic functions (no game code)."""

from __future__ import annotations

import unittest

from tools.dk.platforms.gc_mwcc import asmflow
from tools.dk.platforms.gc_mwcc.asmflow import SP, const, split_operands


def asm(lines: str, start: int = 0x80401000) -> str:
    """dtk-style listing from 'mnem ops' lines; 'NAME:' lines are labels."""
    out, addr = [".fn f, global"], start
    for line in lines.strip().splitlines():
        line = line.strip()
        if line.endswith(":"):
            out.append(line)
            continue
        out.append(f"/* {addr:08X} {addr - 0x80400000:08X}  00 00 00 00 */\t{line}")
        addr += 4
    out.append(".endfn f")
    return "\n".join(out)


def flow(lines: str) -> asmflow.Flow:
    return asmflow.flow_of("f", asm(lines))


# A destructor shaped like the ones the compiler inlines (null check, vtable, owned buffer).
DTOR = """
stwu r1, -0x10(r1)
mflr r0
stw r0, 0x14(r1)
stw r31, 0xc(r1)
stw r30, 0x8(r1)
mr r30, r3
mr r31, r4
cmplwi r30, 0x0
beq .L_end
lis r3, __vt__3Foo@ha
addi r0, r3, __vt__3Foo@l
stw r0, 0x0(r30)
lwz r3, 0x4(r30)
cmplwi r3, 0x0
beq .L_skip
bl free_buffer
li r0, 0x0
stw r0, 0x4(r30)
.L_skip:
extsh r0, r31
cmpwi r0, 0x0
ble .L_end
mr r3, r30
bl __dl__FPv
.L_end:
mr r3, r30
lwz r0, 0x14(r1)
lwz r31, 0xc(r1)
lwz r30, 0x8(r1)
mtlr r0
addi r1, r1, 0x10
blr
"""


class ParseTest(unittest.TestCase):
    def test_operands_respect_quotes(self):
        self.assertEqual(split_operands('"f<1,2>"'), ('"f<1,2>"',))
        self.assertEqual(split_operands('r3, "@73"@ha'), ("r3", '"@73"@ha'))
        self.assertEqual(split_operands("eq, gt, eq"), ("eq", "gt", "eq"))
        self.assertEqual(split_operands(""), ())

    def test_labels_attach_to_the_next_instruction(self):
        insns = asmflow.parse_insns(asm("li r3, 0x1\n.L_a:\nblr"))
        self.assertEqual([(i.mnem, i.labels) for i in insns], [("li", ()), ("blr", (".L_a",))])
        self.assertEqual(insns[0].ops, ("r3", "0x1"))
        self.assertEqual(insns[1].addr, 0x80401004)

    def test_symbol_references(self):
        self.assertEqual(asmflow.symref('"@73"@ha'), ("@73", "ha"))
        self.assertEqual(asmflow.symref("lbl_80401234@sda21"), ("lbl_80401234", "sda21"))
        self.assertIsNone(asmflow.symref("0x10"))


class NumberingTest(unittest.TestCase):
    def setUp(self):
        self.f = flow(DTOR)
        self.by_text = {n.insn.text: n for n in self.f.nodes}

    def test_frame_is_marked(self):
        kinds = [n.kind for n in self.f.nodes]
        self.assertEqual(kinds[:5], ["frame"] * 5)
        self.assertEqual(kinds[-6:], ["frame"] * 6)
        self.assertEqual(self.by_text["mr r30, r3"].kind, "copy")

    def test_values_follow_copies_and_displacements(self):
        store = self.by_text["stw r0, 0x0(r30)"]
        self.assertEqual(store.ins, ((("sym", "__vt__3Foo"), 0), (("arg", "r3"), 0)))
        self.assertEqual(store.mem, (("arg", "r3"), 0))
        load = self.by_text["lwz r3, 0x4(r30)"]
        self.assertEqual(load.ins, ((("arg", "r3"), 4),))
        self.assertEqual(self.by_text["lis r3, __vt__3Foo@ha"].kind, "pure")

    def test_calls_take_fresh_arguments_and_clobber(self):
        call = self.by_text["bl free_buffer"]
        load = self.by_text["lwz r3, 0x4(r30)"]
        self.assertEqual(call.key, "bl free_buffer")
        self.assertEqual(call.args, (("r3", (("n", load.i, ""), 0)),))
        store0 = self.by_text["stw r0, 0x4(r30)"]
        self.assertEqual(store0.ins[0], const(0))
        dl = self.by_text["bl __dl__FPv"]
        self.assertEqual(dl.args, (("r3", (("arg", "r3"), 0)),))

    def test_join_keeps_untouched_registers_and_merges_written_ones(self):
        ext = self.by_text["extsh r0, r31"]           # after .L_skip: r31 untouched on both paths
        self.assertEqual(ext.ins, ((("arg", "r4"), 0),))
        f = flow("""
            cmpwi r3, 0x0
            beq .L_j
            li r4, 0x5
            .L_j:
            stw r4, 0x0(r5)
            stw r6, 0x4(r5)
        """)
        first, second = f.nodes[-2], f.nodes[-1]
        self.assertEqual(first.ins[0], (("phi", ".L_j", "r4"), 0))
        self.assertEqual(second.ins[0], (("arg", "r6"), 0))

    def test_branch_reads_the_compare_it_depends_on(self):
        beq = [n for n in self.f.nodes if n.kind == "branch"][1]
        self.assertEqual(self.f.nodes[beq.ins[0][0][1]].insn.text, "cmplwi r3, 0x0")
        self.assertEqual(beq.target, ".L_skip")


class InstructionFormsTest(unittest.TestCase):
    def test_small_data_and_constants(self):
        f = flow("""
            lwz r3, lbl_80790000@sda21(r0)
            li r4, lbl_80790010@sda21
            lis r5, 0x1
            subi r5, r5, 0x5464
            stw r4, 0x10(r1)
            stw r5, 0x14(r1)
        """)
        self.assertEqual(f.nodes[0].ins, ((("sym", "lbl_80790000"), 0),))
        self.assertEqual(f.nodes[4].ins, ((("sym", "lbl_80790010"), 0), (SP, 0x10)))
        self.assertEqual(f.nodes[5].ins[0], const(0xAB9C))

    def test_condition_register_fields(self):
        f = flow("""
            cmpwi cr1, r3, 0x0
            fcmpo cr0, f1, f2
            cror eq, gt, eq
            beq cr1, .L_x
            bne .L_x
            .L_x:
            crclr cr1eq
        """)
        cmp1, fcmp, cror, beq, bne, crclr = f.nodes
        self.assertEqual(beq.ins, ((("n", cmp1.i, ""), 0),))
        self.assertEqual(cror.ins, ((("n", fcmp.i, ""), 0),))
        self.assertEqual(bne.ins, ((("n", cror.i, ""), 0),))
        self.assertEqual(crclr.key, "crclr cr1eq")

    def test_backward_branch_makes_loop_registers_phi(self):
        f = flow("""
            li r4, 0x0
            .L_top:
            stw r4, 0x0(r3)
            addi r4, r4, 0x1
            cmpwi r4, 0xa
            blt .L_top
        """)
        self.assertEqual(f.nodes[1].ins[0], (("phi", ".L_top", "r4"), 0))
        self.assertEqual(f.nodes[1].ins[1], (("arg", "r3"), 0))

    def test_save_helpers_and_virtual_calls(self):
        f = flow("""
            stwu r1, -0x40(r1)
            mflr r0
            stw r0, 0x44(r1)
            addi r11, r1, 0x40
            bl _savegpr_25
            lwz r12, 0x0(r3)
            lwz r12, 0x10(r12)
            mtctr r12
            bctrl
            mr. r31, r3
        """)
        self.assertEqual(asmflow.frame_closure(f), {0, 1, 2, 3, 4})
        call = f.nodes[8]
        self.assertEqual((call.kind, call.key), ("call", "bctrl"))
        self.assertEqual(call.ins, ((("n", 6, ""), 0),))
        rec = f.nodes[9]
        self.assertEqual((rec.key, rec.ins), ("mr.", ((("n", 8, "r3"), 0),)))

    def test_quoted_names(self):
        f = flow("""
            lis r3, "@73"@ha
            addi r31, r3, "@73"@l
            bl "Init__5ArrayFP3Foo<1>"
            stw r31, 0x8(r30)
        """)
        self.assertEqual(f.nodes[2].key, "bl Init__5ArrayFP3Foo<1>")
        self.assertEqual(f.nodes[3].ins[0], (("sym", "@73"), 0))


if __name__ == "__main__":
    unittest.main()
