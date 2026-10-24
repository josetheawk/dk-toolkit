import unittest

from tools.dk.platforms.gc_mwcc import gen

ASM = """
/* 80023970 00020890  38 00 00 00 */\tli r0, 0x0
/* 80023974 00020894  90 03 00 08 */\tstw r0, 0x8(r3)
/* 80023978 00020898  4E 80 00 20 */\tblr
"""
SWAP = """
/* 80077AE8 00000000  80 03 05 AC */\tlwz r0, 0x5ac(r3)
/* 80077AEC 00000000  90 83 05 AC */\tstw r4, 0x5ac(r3)
/* 80077AF0 00000000  7C 03 03 78 */\tmr r3, r0
/* 80077AF4 00000000  4E 80 00 20 */\tblr
"""
no_literals = lambda sym, size: None


class GenTest(unittest.TestCase):
    def test_store_constant_member(self):
        lifted = gen.lift(gen.parse(ASM), no_literals)
        src, notes = gen.render_tu(lifted, True, "Foo", "fn_80023970", gen.variants(lifted), {}, display=True)
        self.assertIn("void Foo::fn_80023970() {\n    m8 = 0;\n}", src)
        self.assertEqual(notes, ["Foo: +0x8 int m8"])

    def test_old_value_is_hoisted_before_the_store(self):
        lifted = gen.lift(gen.parse(SWAP), no_literals)
        src, _ = gen.render_tu(lifted, True, "Foo", "f", gen.variants(lifted)[:1], {}, display=True)
        self.assertIn("int old = m5AC;\n    m5AC = arg1;\n    return old;", src)

    def test_thunks_and_unknown_instructions_are_refused(self):
        thunk = gen.parse("/* 8000EEE4 00000000  38 63 F9 20 */\tsubi r3, r3, 0x6e0\n/* 8000EEE8 00000000  48 00 00 00 */\tb fn_8000ED5C\n")
        self.assertTrue(gen.thunk_kind(thunk))
        with self.assertRaises(gen.Unsupported):
            gen.lift(gen.parse("/* 80000000 00000000  7C 63 20 2E */\tlwzx r3, r3, r4\n/* 80000004 00000000  4E 80 00 20 */\tblr\n"), no_literals)
        with self.assertRaises(gen.Unsupported):      # r3 + constant with no evidence it is a pointer
            gen.lift(gen.parse("/* 80000000 00000000  38 63 00 10 */\taddi r3, r3, 0x10\n/* 80000004 00000000  4E 80 00 20 */\tblr\n"), no_literals)

    def test_float_literal_round_trips(self):
        self.assertEqual(gen.float_literal(bytes.fromhex("3eaaaa9f")), "0.333333f")
        self.assertEqual(gen.float_literal(bytes.fromhex("3f800000")), "1.0f")
        self.assertEqual(gen.float_literal(bytes.fromhex("42700000")), "60.0f")


class CompareTest(unittest.TestCase):
    """The verifier must reject a relocation to the right symbol with the wrong addend."""

    def test_addend_mismatch_is_rejected(self):
        from tools.dk.platforms.gc_mwcc.elf import Elf, ElfSection, ElfSymbol, Function, Reloc
        insns = gen.parse("/* 80000000 00000000  80 6D 00 00 */\tlwz r3, lbl_80390000@sda21(r0)\n"
                          "/* 80000004 00000000  4E 80 00 20 */\tblr\n")

        class FakeElf:
            def __init__(self, addend):
                self.fn = Function("v0", ".text", bytes.fromhex("806000004e800020"),
                                   [Reloc(0, gen.R_PPC_EMB_SDA21, "lbl_80390000", addend)])

            def functions(self):
                return {"v0": self.fn}

        no_lit = lambda sym, size, offset=0: None
        self.assertIsNone(gen.compare(insns, FakeElf(0), "v0", no_lit))
        self.assertIsNotNone(gen.compare(insns, FakeElf(4), "v0", no_lit))

    def test_two_relocations_on_one_word_are_rejected(self):
        from tools.dk.platforms.gc_mwcc.elf import Function, Reloc
        insns = gen.parse("/* 80000000 00000000  80 6D 00 00 */\tlwz r3, lbl_80390000@sda21(r0)\n"
                          "/* 80000004 00000000  4E 80 00 20 */\tblr\n")

        class FakeElf:
            def functions(self):
                return {"v0": Function("v0", ".text", bytes.fromhex("806000004e800020"),
                                       [Reloc(0, gen.R_PPC_EMB_SDA21, "lbl_80390000", 0),
                                        Reloc(2, gen.R_PPC_ADDR16_LO, "lbl_80390000", 0)])}
        self.assertIsNotNone(gen.compare(insns, FakeElf(), "v0", lambda s, n, o=0: None))

    def test_literal_must_be_in_the_same_section(self):
        from tools.dk.platforms.gc_mwcc.elf import ElfSection, Function, Reloc
        insns = gen.parse("/* 80000000 00000000  C0 22 00 00 */\tlfs f1, lbl_80394000@sda21(r0)\n"
                          "/* 80000004 00000000  4E 80 00 20 */\tblr\n")
        one = bytes.fromhex("3f800000")

        class FakeElf:
            def __init__(self, section):
                self.sec = ElfSection(2, section, 1, 0, 4, 0, 0, one)
                self.fn = Function("v0", ".text", bytes.fromhex("c02000004e800020"),
                                   [Reloc(0, gen.R_PPC_EMB_SDA21, "@1", 0, section, 0)])

            def functions(self):
                return {"v0": self.fn}

            def section(self, name):
                return self.sec if name == self.sec.name else None
        lit = lambda sym, size, offset=0: one
        info = lambda sym: (".sdata2", 4)
        self.assertIsNone(gen.compare(insns, FakeElf(".sdata2"), "v0", lit, literal_info=info))
        self.assertIsNotNone(gen.compare(insns, FakeElf(".sdata"), "v0", lit, literal_info=info))


CALLS = """
/* 8000F094 00000000  94 21 FF F0 */\tstwu r1, -0x10(r1)
/* 8000F098 00000000  7C 08 02 A6 */\tmflr r0
/* 8000F09C 00000000  90 01 00 14 */\tstw r0, 0x14(r1)
/* 8000F0A0 00000000  93 E1 00 0C */\tstw r31, 0xc(r1)
/* 8000F0A4 00000000  7C 7F 1B 78 */\tmr r31, r3
/* 8000F0A8 00000000  48 00 00 01 */\tbl fn_80010660
/* 8000F0AC 00000000  7F E3 FB 78 */\tmr r3, r31
/* 8000F0B0 00000000  48 00 00 01 */\tbl fn_80010434
/* 8000F0B4 00000000  80 01 00 14 */\tlwz r0, 0x14(r1)
/* 8000F0B8 00000000  83 E1 00 0C */\tlwz r31, 0xc(r1)
/* 8000F0BC 00000000  7C 08 03 A6 */\tmtlr r0
/* 8000F0C0 00000000  38 21 00 10 */\taddi r1, r1, 0x10
/* 8000F0C4 00000000  4E 80 00 20 */\tblr
"""


class CallTest(unittest.TestCase):
    def test_frame_and_calls(self):
        sigs = {"fn_80010660": (0, 0), "fn_80010434": (1, 0)}
        lifted = gen.lift(gen.parse(CALLS), no_literals, True, sigs.get)
        self.assertEqual([c.callee for c in lifted.calls], ["fn_80010660", "fn_80010434"])
        src, _ = gen.render_tu(lifted, True, "CarManager", "fn_8000F094", gen.variants(lifted)[:1], {}, display=True)
        self.assertIn("void CarManager::fn_8000F094() {\n    fn_80010660();\n    fn_80010434(this);\n}", src)
        self.assertIn('extern "C" void fn_80010660();', src)

    def test_signature_inference(self):
        callee = gen.parse("/* 80000000 00000000  90 A3 00 08 */\tstw r5, 0x8(r3)\n"
                           "/* 80000004 00000000  4E 80 00 20 */\tblr\n")
        self.assertEqual(gen.infer_signature(callee), (3, 0))      # reads r3 and r5: three int params

    def test_value_loaded_before_a_call_is_kept_in_a_temporary(self):
        asm = CALLS.replace("\tmr r31, r3", "\tlwz r31, 0x8(r3)")
        lifted = gen.lift(gen.parse(asm), no_literals, True, {"fn_80010660": (0, 0), "fn_80010434": (1, 0)}.get)
        src, _ = gen.render_tu(lifted, True, "C", "f", gen.variants(lifted)[:1], {}, display=True)
        self.assertIn("int old = m8;\n    fn_80010660();\n    fn_80010434(old);", src)


GUARDED = """
/* 8001FF74 00000000  94 21 FF F0 */\tstwu r1, -0x10(r1)
/* 8001FF78 00000000  7C 08 02 A6 */\tmflr r0
/* 8001FF7C 00000000  90 01 00 14 */\tstw r0, 0x14(r1)
/* 8001FF80 00000000  80 63 10 AC */\tlwz r3, 0x10ac(r3)
/* 8001FF84 00000000  28 03 00 00 */\tcmplwi r3, 0x0
/* 8001FF88 00000000  41 82 00 08 */\tbeq .L_8001FF90
/* 8001FF8C 00000000  48 00 00 01 */\tbl fn_8000E9B8
.L_8001FF90:
/* 8001FF90 00000000  80 01 00 14 */\tlwz r0, 0x14(r1)
/* 8001FF94 00000000  7C 08 03 A6 */\tmtlr r0
/* 8001FF98 00000000  38 21 00 10 */\taddi r1, r1, 0x10
/* 8001FF9C 00000000  4E 80 00 20 */\tblr
"""
EARLY = """
/* 80038940 00000000  80 63 01 78 */\tlwz r3, 0x178(r3)
/* 80038944 00000000  28 03 00 00 */\tcmplwi r3, 0x0
/* 80038948 00000000  41 82 00 0C */\tbeq .L_80038954
/* 8003894C 00000000  80 63 06 E8 */\tlwz r3, 0x6e8(r3)
/* 80038950 00000000  4E 80 00 20 */\tblr
.L_80038954:
/* 80038954 00000000  38 60 00 00 */\tli r3, 0x0
/* 80038958 00000000  4E 80 00 20 */\tblr
"""


class FlowTest(unittest.TestCase):
    def test_guarded_call(self):
        lifted = gen.lift(gen.parse(GUARDED), no_literals, True, {"fn_8000E9B8": (1, 0)}.get)
        self.assertEqual(gen.variants(lifted)[0].ret_mode, "void")
        src, notes = gen.render_tu(lifted, True, "C", "f", gen.variants(lifted)[:1], {}, display=True)
        self.assertIn("    if (m10AC != 0) {\n        fn_8000E9B8(m10AC);\n    }\n}", src)
        self.assertEqual(notes, ["C: +0x10AC unsigned int m10AC"])     # cmplwi: unsigned

    def test_early_return(self):
        lifted = gen.lift(gen.parse(EARLY), no_literals, True)
        src, _ = gen.render_tu(lifted, True, "C", "f", [gen.Variant((), "return")], {}, display=True)
        self.assertIn("if (m178 != 0) {\n        return m178->m6E8;\n    } else {\n        return 0;\n    }", src)

    def test_loops_are_refused(self):
        loop = EARLY.replace("beq .L_80038954", "beq .L_80038940").replace("80038940 00000000  80 63 01 78 */\tlwz", ".L_80038940:\n/* 80038940 00000000  80 63 01 78 */\tlwz")
        with self.assertRaises(gen.Unsupported):
            gen.lift(gen.parse(loop), no_literals, True)


VCALL = """
/* 8002BE74 00000000  94 21 FF F0 */\tstwu r1, -0x10(r1)
/* 8002BE78 00000000  7C 08 02 A6 */\tmflr r0
/* 8002BE7C 00000000  90 01 00 14 */\tstw r0, 0x14(r1)
/* 8002BE80 00000000  80 63 00 04 */\tlwz r3, 0x4(r3)
/* 8002BE84 00000000  28 03 00 00 */\tcmplwi r3, 0x0
/* 8002BE88 00000000  41 82 00 14 */\tbeq .L_8002BE9C
/* 8002BE8C 00000000  81 83 00 00 */\tlwz r12, 0x0(r3)
/* 8002BE90 00000000  81 8C 00 08 */\tlwz r12, 0x8(r12)
/* 8002BE94 00000000  7D 89 03 A6 */\tmtctr r12
/* 8002BE98 00000000  4E 80 04 21 */\tbctrl
.L_8002BE9C:
/* 8002BE9C 00000000  80 01 00 14 */\tlwz r0, 0x14(r1)
/* 8002BEA0 00000000  7C 08 03 A6 */\tmtlr r0
/* 8002BEA4 00000000  38 21 00 10 */\taddi r1, r1, 0x10
/* 8002BEA8 00000000  4E 80 00 20 */\tblr
"""


class VirtualCallTest(unittest.TestCase):
    def test_slot_and_vptr_position(self):
        lifted = gen.lift(gen.parse(VCALL), no_literals, True)
        (call,) = lifted.calls
        self.assertTrue(call.virtual)
        self.assertEqual(call.callee, "vf0")                            # vtable + 8 = slot 0
        src, notes = gen.render_tu(lifted, True, "H", "f", gen.variants(lifted)[:1], {}, display=True)
        self.assertIn("virtual void vf0();", src)
        self.assertIn("if (m4 != 0) {\n        m4->vf0();\n    }", src)
        self.assertIn("H_m4: +0x0 vtable pointer (calls vf0)", notes)


MEMBER_CALL = """
/* 80134A98 00000000  94 21 FF F0 */\tstwu r1, -0x10(r1)
/* 80134A9C 00000000  7C 08 02 A6 */\tmflr r0
/* 80134AA0 00000000  90 01 00 14 */\tstw r0, 0x14(r1)
/* 80134AA4 00000000  80 63 00 14 */\tlwz r3, 0x14(r3)
/* 80134AA8 00000000  48 00 00 01 */\tbl fn_80151EEC__Q24EAGL10SymbolPoolFPCcPv
/* 80134AAC 00000000  80 01 00 14 */\tlwz r0, 0x14(r1)
/* 80134AB0 00000000  7C 08 03 A6 */\tmtlr r0
/* 80134AB4 00000000  38 21 00 10 */\taddi r1, r1, 0x10
/* 80134AB8 00000000  4E 80 00 20 */\tblr
"""


class CxxCallTest(unittest.TestCase):
    def test_member_call_binds_the_receiver_and_types_the_arguments(self):
        lifted = gen.lift(gen.parse(MEMBER_CALL), no_literals, True)
        (call,) = lifted.calls
        self.assertEqual(call.cxx.quals, ("EAGL", "SymbolPool"))
        src, _ = gen.render_tu(lifted, True, "Owner", "f", gen.variants(lifted)[:1], {}, display=True)
        self.assertIn("struct EAGL { struct SymbolPool; };", src)
        self.assertIn("    void fn_80151EEC(const char*, void*);", src)
        self.assertIn("void Owner::f(const char* arg1, void* arg2) {\n    m14->fn_80151EEC(arg1, arg2);\n}", src)


I2F = """
/* 80000000 00000000  94 21 FF F0 */\tstwu r1, -0x10(r1)
/* 80000004 00000000  3C 00 43 30 */\tlis r0, 0x4330
/* 80000008 00000000  C8 22 00 00 */\tlfd f1, lbl_80394000@sda21(r0)
/* 8000000C 00000000  80 63 00 00 */\tlwz r3, 0x0(r3)
/* 80000010 00000000  90 01 00 08 */\tstw r0, 0x8(r1)
/* 80000014 00000000  6C 60 80 00 */\txoris r0, r3, 0x8000
/* 80000018 00000000  90 01 00 0C */\tstw r0, 0xc(r1)
/* 8000001C 00000000  C8 01 00 08 */\tlfd f0, 0x8(r1)
/* 80000020 00000000  EC 20 08 28 */\tfsubs f1, f0, f1
/* 80000024 00000000  38 21 00 10 */\taddi r1, r1, 0x10
/* 80000028 00000000  4E 80 00 20 */\tblr
"""


class ConversionTest(unittest.TestCase):
    def test_int_to_float_idiom(self):
        magic = lambda sym, size, offset=0: bytes.fromhex("4330000080000000") if sym == "lbl_80394000" else None
        lifted = gen.lift(gen.parse(I2F), magic, True)
        src, _ = gen.render_tu(lifted, True, "C", "f", gen.variants(lifted)[:1], {}, display=True)
        self.assertIn("float C::f() {\n    return (float)m0;\n}", src)

    def test_other_stack_use_is_refused(self):
        with self.assertRaises(gen.Unsupported):
            gen.lift(gen.parse(I2F.replace("lfd f0, 0x8(r1)", "lfd f0, 0x10(r1)")), lambda s, n, o=0: None, True)


class ArrayTest(unittest.TestCase):
    def lift(self, body):
        return gen.lift(gen.parse(body), no_literals, True)

    def test_embedded_int_array(self):
        lifted = self.lift("/* 800C4664 00000000  54 80 10 3A */\tslwi r0, r4, 2\n"
                           "/* 800C4668 00000000  7C 63 02 14 */\tadd r3, r3, r0\n"
                           "/* 800C466C 00000000  80 63 00 10 */\tlwz r3, 0x10(r3)\n"
                           "/* 800C4670 00000000  4E 80 00 20 */\tblr\n")
        src, _ = gen.render_tu(lifted, True, "C", "f", gen.variants(lifted)[:1], {}, display=True)
        self.assertIn("return m10[arg1];", src)

    def test_pointer_to_struct_elements(self):
        lifted = self.lift("/* 800D1818 00000000  80 63 00 28 */\tlwz r3, 0x28(r3)\n"
                           "/* 800D181C 00000000  54 80 28 34 */\tslwi r0, r4, 5\n"
                           "/* 800D1820 00000000  7C 63 02 14 */\tadd r3, r3, r0\n"
                           "/* 800D1824 00000000  4E 80 00 20 */\tblr\n")
        src, _ = gen.render_tu(lifted, True, "C", "f", gen.variants(lifted)[:1], {}, display=True)
        self.assertIn("return &m28[arg1];", src)
        self.assertIn("char _pad0[0x20];", src)                    # the element is 32 bytes
