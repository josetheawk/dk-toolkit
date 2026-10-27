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
        self.assertTrue(gen.float_literal(bytes.fromhex("7f7fffff")).endswith("f"))      # FLT_MAX


class CompareTest(unittest.TestCase):
    """The verifier must reject a relocation to the right symbol with the wrong addend."""

    def test_addend_mismatch_is_rejected(self):
        from tools.dk.platforms.gc_mwcc.elf import Function, Reloc
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

    def test_do_while_loop_with_a_loop_variable(self):
        asm = """
/* 80009484 00000000  94 21 FF F0 */\tstwu r1, -0x10(r1)
/* 80009488 00000000  7C 08 02 A6 */\tmflr r0
/* 8000948C 00000000  90 01 00 14 */\tstw r0, 0x14(r1)
/* 80009490 00000000  93 E1 00 0C */\tstw r31, 0xc(r1)
/* 80009494 00000000  93 C1 00 08 */\tstw r30, 0x8(r1)
/* 80009498 00000000  7C 7E 1B 78 */\tmr r30, r3
/* 8000949C 00000000  3B E0 00 00 */\tli r31, 0x0
.L_800094A0:
/* 800094A0 00000000  7F C3 F3 78 */\tmr r3, r30
/* 800094A4 00000000  7F E4 FB 78 */\tmr r4, r31
/* 800094A8 00000000  48 00 00 01 */\tbl fn_8000C270
/* 800094AC 00000000  3B FF 00 01 */\taddi r31, r31, 0x1
/* 800094B0 00000000  2C 1F 00 04 */\tcmpwi r31, 0x4
/* 800094B4 00000000  41 80 FF EC */\tblt .L_800094A0
/* 800094B8 00000000  80 01 00 14 */\tlwz r0, 0x14(r1)
/* 800094BC 00000000  83 E1 00 0C */\tlwz r31, 0xc(r1)
/* 800094C0 00000000  83 C1 00 08 */\tlwz r30, 0x8(r1)
/* 800094C4 00000000  7C 08 03 A6 */\tmtlr r0
/* 800094C8 00000000  38 21 00 10 */\taddi r1, r1, 0x10
/* 800094CC 00000000  4E 80 00 20 */\tblr
"""
        lifted = gen.lift(gen.parse(asm), no_literals, True, {"fn_8000C270": (2, 0)}.get)
        src, _ = gen.render_tu(lifted, True, "C", "f", [gen.Variant((), "void")], {}, display=True)
        self.assertIn("int v31 = 0;\n    do {\n        fn_8000C270(this, v31);\n        v31 = v31 + 1;\n    } while (v31 < 4);", src)

    def test_loop_closed_by_an_unconditional_branch_is_refused(self):
        loop = EARLY.replace("beq .L_80038954", "beq .L_80038954").replace(
            "/* 80038950 00000000  4E 80 00 20 */\tblr", "/* 80038950 00000000  4B FF FF F0 */\tb .L_80038940").replace(
            "80038940 00000000  80 63 01 78 */\tlwz", ".L_80038940:\n/* 80038940 00000000  80 63 01 78 */\tlwz")
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


class ConditionalReturnTest(unittest.TestCase):
    def test_a_conditional_return_is_an_early_return(self):
        asm = """
/* 8001C974 00000000  88 03 00 11 */\tlbz r0, 0x11(r3)
/* 8001C978 00000000  28 00 00 00 */\tcmplwi r0, 0x0
/* 8001C97C 00000000  4C 82 00 20 */\tbnelr
/* 8001C980 00000000  38 00 00 FF */\tli r0, 0xff
/* 8001C984 00000000  98 03 06 28 */\tstb r0, 0x628(r3)
/* 8001C988 00000000  4E 80 00 20 */\tblr
"""
        lifted = gen.lift(gen.parse(asm), no_literals, True)
        src, _ = gen.render_tu(lifted, True, "C", "f", [gen.Variant((), "void")], {}, display=True)
        self.assertIn("void C::f() {\n    if (m11 == 0) {\n        m628 = 255;\n    }\n}", src)

    def test_a_constructor_is_written_as_one(self):
        asm = """
/* 80000000 00000000  38 00 00 00 */\tli r0, 0x0
/* 80000004 00000000  90 03 00 04 */\tstw r0, 0x4(r3)
/* 80000008 00000000  4E 80 00 20 */\tblr
"""
        lifted = gen.lift(gen.parse(asm), no_literals, True)
        void = [v for v in gen.variants(lifted) if v.ret_mode == "void"][:1]
        src, _ = gen.render_tu(lifted, True, "Pos", "__ct", void, {}, standalone=True)
        self.assertIn("    Pos();\n", src)
        self.assertIn("Pos::Pos() {\n    m4 = 0;\n}", src)


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

    def test_inherited_method_called_on_this_is_a_member_of_the_derived_class(self):
        asm = MEMBER_CALL.replace("\tlwz r3, 0x14(r3)", "\tstw r4, 0x8(r3)").replace("Q24EAGL10SymbolPoolFPCcPv", "4BaseFv")
        lifted = gen.lift(gen.parse(asm), no_literals, True)
        src, notes = gen.render_tu(lifted, True, "Base", "f", gen.variants(lifted)[:1], {}, standalone=True,
                                   chain=["Base", "Mid", "Leaf"])
        self.assertIn("struct Base {\n    char _pad0[0x8];\n    int m8;\n    void fn_80151EEC();\n};", src)
        self.assertIn("struct Mid : Base {\n};\nstruct Leaf : Mid {\n    void f(int arg1);\n};", src)
        self.assertIn("void Leaf::f(int arg1) {\n    m8 = arg1;\n    Base::fn_80151EEC();\n}", src)
        self.assertTrue(any(n.startswith("class chain (RTTI, root last): Leaf : Mid : Base") for n in notes))

    def test_a_slot_called_by_name_is_one_virtual_declaration(self):
        members = ["virtual void vf0();", "virtual void vf1();", "float vf1(int);", "void fn_1();"]
        self.assertEqual(gen._fold_called_slots(members), ["virtual void vf0();", "virtual float vf1(int);", "void fn_1();"])


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


class BitfieldTest(unittest.TestCase):
    def test_rotate_forms_become_shifts_and_masks(self):
        x = gen.Param("r3")
        self.assertEqual(gen._rotated(x, 20, gen._mask(31, 31)),                  # extrwi r, x, 1, 12
                         gen.Op("&", (gen.Op(">>u", (x, gen.Const(12)), "int"), gen.Const(1)), "int"))
        self.assertEqual(gen._rotated(x, 4, gen._mask(0, 27)), gen.Op("<<", (x, gen.Const(4)), "int"))   # slwi
        with self.assertRaises(gen.Unsupported):
            gen._rotated(x, 8, gen._mask(0, 31))                                  # a real rotate

    def test_insert_renders_as_mask_and_or(self):
        asm = ("/* 800F55C0 00000000  88 03 00 01 */\tlbz r0, 0x1(r3)\n"
               "/* 800F55C4 00000000  50 80 26 36 */\trlwimi r0, r4, 4, 25, 27\n"
               "/* 800F55C8 00000000  98 03 00 01 */\tstb r0, 0x1(r3)\n"
               "/* 800F55CC 00000000  4E 80 00 20 */\tblr\n")
        lifted = gen.lift(gen.parse(asm), no_literals, True)
        src, _ = gen.render_tu(lifted, True, "C", "f", gen.variants(lifted)[:1], {}, display=True)
        self.assertIn("m1 = (m1 & ~0x70) | ((arg1 << 4) & 0x70);", src)


class MergedValueTest(unittest.TestCase):
    def test_a_value_merged_from_two_arms_becomes_a_variable(self):
        asm = """
/* 80000000 00000000  2C 04 00 00 */\tcmpwi r4, 0x0
/* 80000004 00000000  41 82 00 0C */\tbeq .L_80000010
/* 80000008 00000000  38 00 00 01 */\tli r0, 0x1
/* 8000000C 00000000  48 00 00 08 */\tb .L_80000014
.L_80000010:
/* 80000010 00000000  38 00 00 02 */\tli r0, 0x2
.L_80000014:
/* 80000014 00000000  90 03 00 08 */\tstw r0, 0x8(r3)
/* 80000018 00000000  4E 80 00 20 */\tblr
"""
        lifted = gen.lift(gen.parse(asm), no_literals, True)
        src, _ = gen.render_tu(lifted, True, "C", "f", [gen.Variant((), "void")], {}, display=True)
        self.assertIn("int m1;\n    if (arg1 != 0) {\n        m1 = 1;\n    } else {\n        m1 = 2;\n    }\n    m8 = m1;", src)


    def test_a_merged_return_value_is_returned_in_each_arm(self):
        asm = """
/* 80000000 00000000  80 63 00 6C */\tlwz r3, 0x6c(r3)
/* 80000004 00000000  28 03 00 00 */\tcmplwi r3, 0x0
/* 80000008 00000000  41 82 00 0C */\tbeq .L_80000014
/* 8000000C 00000000  48 00 00 01 */\tbl fn_80001000
/* 80000010 00000000  48 00 00 08 */\tb .L_80000018
.L_80000014:
/* 80000014 00000000  38 60 00 00 */\tli r3, 0x0
.L_80000018:
/* 80000018 00000000  4E 80 00 20 */\tblr
"""
        lifted = gen.lift(gen.parse(asm), no_literals, True, lambda name: (1, 0))
        sink = [v for v in gen.variants(lifted) if v.sink]
        self.assertTrue(sink)
        src, _ = gen.render_tu(lifted, True, "C", "f", sink[:1], {}, display=True)
        self.assertIn("    if (m6C != 0) {\n        return fn_80001000(m6C);\n    }\n    return 0;\n}", src)


class CallArgumentTest(unittest.TestCase):
    def test_varargs_call_takes_the_registers_written_before_it(self):
        asm = """
/* 80000000 00000000  38 80 00 05 */\tli r4, 0x5
/* 80000004 00000000  4C C6 31 82 */\tcrclr cr1eq
/* 80000008 00000000  48 00 00 01 */\tbl printf
/* 8000000C 00000000  4E 80 00 20 */\tblr
"""
        lifted = gen.lift(gen.parse(asm), no_literals, False, lambda name: (8, 8))
        src, _ = gen.render_tu(lifted, False, "G", "f", [gen.Variant((), "void")], {}, display=True)
        self.assertIn('extern "C" int printf(const char*, ...);', src)     # the C library's prototype
        self.assertIn("    printf(arg0, 5);", src)

    def test_a_locals_address_in_a_later_register_is_an_argument(self):
        asm = """
/* 80000000 00000000  94 21 FF E0 */\tstwu r1, -0x20(r1)
/* 80000004 00000000  7C 08 02 A6 */\tmflr r0
/* 80000008 00000000  90 01 00 24 */\tstw r0, 0x24(r1)
/* 8000000C 00000000  38 81 00 08 */\taddi r4, r1, 0x8
/* 80000010 00000000  48 00 00 01 */\tbl fn_80001000
/* 80000014 00000000  80 01 00 24 */\tlwz r0, 0x24(r1)
/* 80000018 00000000  7C 08 03 A6 */\tmtlr r0
/* 8000001C 00000000  38 21 00 20 */\taddi r1, r1, 0x20
/* 80000020 00000000  4E 80 00 20 */\tblr
"""
        lifted = gen.lift(gen.parse(asm), no_literals, False, lambda name: (1, 0))
        self.assertEqual(len(lifted.calls[0].args), 2)

    def test_registers_set_for_a_call_and_read_by_nothing_else_are_arguments(self):
        asm = """
/* 80000000 00000000  38 A0 00 05 */\tli r5, 0x5
/* 80000004 00000000  90 A3 00 08 */\tstw r5, 0x8(r3)
/* 80000008 00000000  38 80 00 07 */\tli r4, 0x7
/* 8000000C 00000000  48 00 00 01 */\tbl fn_80001000
/* 80000010 00000000  4E 80 00 20 */\tblr
"""
        lifted = gen.lift(gen.parse(asm), no_literals, False, lambda name: (0, 0))
        self.assertEqual(len(lifted.calls[0].args), 2)     # r3, r4 (r5 was read by the store)

    def test_other_callers_show_a_passed_on_parameter(self):
        asm = """
/* 80000000 00000000  38 60 00 05 */\tli r3, 0x5
/* 80000004 00000000  48 00 00 01 */\tbl fn_80001000
/* 80000008 00000000  4E 80 00 20 */\tblr
"""
        sig, callers = lambda name: (1, 0), lambda name: (2, 0)
        known = gen.lift(gen.parse(asm), no_literals, False, sig, callers=callers, params=(2, 0))
        self.assertEqual(len(known.calls[0].args), 2)                # r4: this function's own parameter
        self.assertTrue(known.caller_evidence and known.review)
        src, _ = gen.render_tu(known, False, "G", "f", [gen.Variant((), "void")], {}, display=True)
        self.assertIn("    // REVIEW: fn_80001000 takes 2 int", src)
        unknown = gen.lift(gen.parse(asm), no_literals, False, sig, callers=callers, params=(0, 0))
        self.assertEqual(len(unknown.calls[0].args), 1)              # r4 holds nothing of this function's

    def test_a_result_used_once_by_the_next_call_is_nested(self):
        asm = """
/* 80000000 00000000  48 00 00 01 */\tbl fn_80001000
/* 80000004 00000000  48 00 00 01 */\tbl fn_80002000
/* 80000008 00000000  4E 80 00 20 */\tblr
"""
        lifted = gen.lift(gen.parse(asm), no_literals, False, lambda name: (0, 0) if name == "fn_80001000" else (1, 0))
        nest = [v for v in gen.variants(lifted) if v.nest and v.ret_mode == "void"]
        src, _ = gen.render_tu(lifted, False, "G", "f", nest[:1], {}, display=True)
        self.assertIn("    fn_80002000(fn_80001000());\n}", src)

class FloatCompareTest(unittest.TestCase):
    def _body(self, asm):
        lifted = gen.lift(gen.parse(asm), no_literals)
        src, _ = gen.render_tu(lifted, False, "", "f", gen.variants(lifted)[:1], {}, display=True)
        return src

    def test_cror_is_greater_or_equal_and_its_negation_stays_negated(self):
        asm = ("/* 80000000 00000000  FC 01 10 40 */\tfcmpo cr0, f1, f2\n"
               "/* 80000004 00000000  4C 41 13 82 */\tcror eq, gt, eq\n"
               "/* 80000008 00000000  40 82 00 0C */\tbne .L_80000014\n"
               "/* 8000000C 00000000  38 00 00 01 */\tli r0, 0x1\n"
               "/* 80000010 00000000  90 03 00 00 */\tstw r0, 0x0(r3)\n"
               "/* 80000014 00000000  4E 80 00 20 */\tblr\n")
        self.assertIn("if (farg0 >= farg1)", self._body(asm))
        self.assertIn("if (!(farg0 >= farg1))", self._body(asm.replace("bne .L", "beq .L")))

    def test_plain_float_branch_tests_one_bit(self):
        asm = ("/* 80000000 00000000  FC 01 10 40 */\tfcmpo cr0, f1, f2\n"
               "/* 80000004 00000000  40 80 00 0C */\tbge .L_80000010\n"
               "/* 80000008 00000000  38 00 00 01 */\tli r0, 0x1\n"
               "/* 8000000C 00000000  90 03 00 00 */\tstw r0, 0x0(r3)\n"
               "/* 80000010 00000000  4E 80 00 20 */\tblr\n")
        self.assertIn("if (farg0 < farg1)", self._body(asm))
        self.assertIn("if (!(farg0 < farg1))", self._body(asm.replace("bge .L", "blt .L")))


class FrameTest(unittest.TestCase):
    def test_save_helper_with_a_scheduled_instruction_in_between(self):
        asm = ("/* 80000000 00000000  94 21 FF E0 */\tstwu r1, -0x20(r1)\n"
               "/* 80000004 00000000  7C 08 02 A6 */\tmflr r0\n"
               "/* 80000008 00000000  90 01 00 24 */\tstw r0, 0x24(r1)\n"
               "/* 8000000C 00000000  39 61 00 20 */\taddi r11, r1, 0x20\n"
               "/* 80000010 00000000  48 00 00 01 */\tbl _savegpr_29\n"
               "/* 80000014 00000000  7C 7D 1B 78 */\tmr r29, r3\n"
               "/* 80000018 00000000  48 00 00 01 */\tbl fn_80001000\n"
               "/* 8000001C 00000000  39 61 00 20 */\taddi r11, r1, 0x20\n"
               "/* 80000020 00000000  7F A3 EB 78 */\tmr r3, r29\n"
               "/* 80000024 00000000  48 00 00 01 */\tbl _restgpr_29\n"
               "/* 80000028 00000000  80 01 00 24 */\tlwz r0, 0x24(r1)\n"
               "/* 8000002C 00000000  7C 08 03 A6 */\tmtlr r0\n"
               "/* 80000030 00000000  38 21 00 20 */\taddi r1, r1, 0x20\n"
               "/* 80000034 00000000  4E 80 00 20 */\tblr\n")
        body = gen.strip_frame(gen.parse(asm))
        self.assertEqual([i.op for i in body], ["mr", "bl", "mr", "blr"])

    def test_a_struct_local_can_be_sized_by_its_fields(self):
        asm = ("/* 80000000 00000000  94 21 FF E0 */\tstwu r1, -0x20(r1)\n"
               "/* 80000004 00000000  7C 08 02 A6 */\tmflr r0\n"
               "/* 80000008 00000000  90 01 00 24 */\tstw r0, 0x24(r1)\n"
               "/* 8000000C 00000000  C0 03 00 00 */\tlfs f0, 0x0(r3)\n"
               "/* 80000010 00000000  D0 01 00 08 */\tstfs f0, 0x8(r1)\n"
               "/* 80000014 00000000  D0 01 00 10 */\tstfs f0, 0x10(r1)\n"
               "/* 80000018 00000000  38 61 00 08 */\taddi r3, r1, 0x8\n"
               "/* 8000001C 00000000  48 00 00 01 */\tbl fn_80001000\n"
               "/* 80000020 00000000  80 01 00 24 */\tlwz r0, 0x24(r1)\n"
               "/* 80000024 00000000  7C 08 03 A6 */\tmtlr r0\n"
               "/* 80000028 00000000  38 21 00 20 */\taddi r1, r1, 0x20\n"
               "/* 8000002C 00000000  4E 80 00 20 */\tblr\n")
        lifted = gen.lift(gen.parse(asm), no_literals, False, lambda name: (1, 0))
        self.assertEqual(lifted.locals, {8: 0x18})                    # up to the frame's save area
        self.assertEqual(gen.tight_locals(lifted), {8: 0xC})          # m0 .. m8 + 4


class CompoundConditionTest(unittest.TestCase):
    OR = ("/* 80000000 00000000  80 03 00 00 */\tlwz r0, 0x0(r3)\n"
          "/* 80000004 00000000  2C 00 00 00 */\tcmpwi r0, 0x0\n"
          "/* 80000008 00000000  40 82 00 10 */\tbne .L_80000018\n"
          "/* 8000000C 00000000  80 03 00 04 */\tlwz r0, 0x4(r3)\n"
          "/* 80000010 00000000  2C 00 00 00 */\tcmpwi r0, 0x0\n"
          "/* 80000014 00000000  41 82 00 0C */\tbeq .L_80000020\n"
          ".L_80000018:\n"
          "/* 80000018 00000000  38 00 00 01 */\tli r0, 0x1\n"
          "/* 8000001C 00000000  90 03 00 08 */\tstw r0, 0x8(r3)\n"
          ".L_80000020:\n"
          "/* 80000020 00000000  4E 80 00 20 */\tblr\n")

    def test_or_reaching_one_block_from_two_tests(self):
        lifted = gen.lift(gen.parse(self.OR), no_literals, True)
        src, _ = gen.render_tu(lifted, True, "C", "f", gen.variants(lifted)[:1], {}, display=True)
        self.assertIn("if ((m0 != 0) || (m4 != 0)) {\n        m8 = 1;\n    }", src)

    def test_and_with_an_else_arm(self):
        asm = ("/* 80000000 00000000  80 03 00 00 */\tlwz r0, 0x0(r3)\n"
               "/* 80000004 00000000  2C 00 00 00 */\tcmpwi r0, 0x0\n"
               "/* 80000008 00000000  41 82 00 1C */\tbeq .L_80000024\n"
               "/* 8000000C 00000000  80 03 00 04 */\tlwz r0, 0x4(r3)\n"
               "/* 80000010 00000000  2C 00 00 00 */\tcmpwi r0, 0x0\n"
               "/* 80000014 00000000  41 82 00 10 */\tbeq .L_80000024\n"
               "/* 80000018 00000000  38 00 00 01 */\tli r0, 0x1\n"
               "/* 8000001C 00000000  90 03 00 08 */\tstw r0, 0x8(r3)\n"
               "/* 80000020 00000000  48 00 00 0C */\tb .L_8000002C\n"
               ".L_80000024:\n"
               "/* 80000024 00000000  38 00 00 02 */\tli r0, 0x2\n"
               "/* 80000028 00000000  90 03 00 08 */\tstw r0, 0x8(r3)\n"
               ".L_8000002C:\n"
               "/* 8000002C 00000000  4E 80 00 20 */\tblr\n")
        lifted = gen.lift(gen.parse(asm), no_literals, True)
        src, _ = gen.render_tu(lifted, True, "C", "f", gen.variants(lifted)[:1], {}, display=True)
        self.assertIn("if ((m0 != 0) && (m4 != 0)) {\n        m8 = 1;\n    } else {\n        m8 = 2;\n    }", src)


class CountedLoopTest(unittest.TestCase):
    def test_bdnz_is_a_for_loop(self):
        asm = ("/* 80000000 00000000  38 00 00 04 */\tli r0, 0x4\n"
               "/* 80000004 00000000  7C 09 03 A6 */\tmtctr r0\n"
               ".L_80000008:\n"
               "/* 80000008 00000000  90 83 00 00 */\tstw r4, 0x0(r3)\n"
               "/* 8000000C 00000000  38 63 00 04 */\taddi r3, r3, 0x4\n"
               "/* 80000010 00000000  42 00 FF F8 */\tbdnz .L_80000008\n"
               "/* 80000014 00000000  4E 80 00 20 */\tblr\n")
        lifted = gen.lift(gen.parse(asm), no_literals)
        loop = next(st for st in lifted.tree if isinstance(st, gen.Loop))
        self.assertEqual((loop.kind, loop.cond), ("for", gen.Const(4)))
        src, _ = gen.render_tu(lifted, False, "", "f", gen.variants(lifted)[:1], {}, display=True)
        self.assertIn("for (i = 0; i < 4; i++) {", src)


FACTORY = """
/* 801F92E8 001F6208  94 21 FF F0 */\tstwu r1, -0x10(r1)
/* 801F92EC 001F620C  7C 08 02 A6 */\tmflr r0
/* 801F92F0 001F6210  38 80 00 00 */\tli r4, 0x0
/* 801F92F4 001F6214  38 AD C3 A4 */\tli r5, lbl_80390DE4@sda21
/* 801F92F8 001F6218  90 01 00 14 */\tstw r0, 0x14(r1)
/* 801F92FC 001F621C  93 E1 00 0C */\tstw r31, 0xc(r1)
/* 801F9300 001F6220  7C 7F 1B 78 */\tmr r31, r3
/* 801F9304 001F6224  38 60 01 14 */\tli r3, 0x114
/* 801F9308 001F6228  48 00 88 09 */\tbl __nw__7GObjectFUliPCc
/* 801F930C 001F622C  7C 60 1B 79 */\tmr. r0, r3
/* 801F9310 001F6230  41 82 00 10 */\tbeq .L_801F9320
/* 801F9314 001F6234  7F E4 FB 78 */\tmr r4, r31
/* 801F9318 001F6238  48 03 41 89 */\tbl __ct__7GWidgetFP3GUI
/* 801F931C 001F623C  7C 60 1B 78 */\tmr r0, r3
.L_801F9320:
/* 801F9320 001F6240  7C 03 03 78 */\tmr r3, r0
/* 801F9324 001F6244  80 01 00 14 */\tlwz r0, 0x14(r1)
/* 801F9328 001F6248  83 E1 00 0C */\tlwz r31, 0xc(r1)
/* 801F932C 001F624C  7C 08 03 A6 */\tmtlr r0
/* 801F9330 001F6250  38 21 00 10 */\taddi r1, r1, 0x10
/* 801F9334 001F6254  4E 80 00 20 */\tblr
"""


class NewExpressionTest(unittest.TestCase):
    def test_new_with_placement_and_constructor(self):
        lifted = gen.lift(gen.parse(FACTORY), lambda s, n, o=0: None)
        (call,) = lifted.calls
        self.assertEqual((call.callee, call.new.callee, call.new.size, call.new.cls),
                         ("__ct__7GWidgetFP3GUI", "__nw__7GObjectFUliPCc", 0x114, ("GWidget",)))
        v = next(v for v in gen.variants(lifted) if v.ret_mode == "return")
        src, _ = gen.render_tu(lifted, False, "G", "f", [v], {"lbl_80390DE4": 8}, standalone=True)
        self.assertIn("struct GObject {\n    static void* operator new(unsigned long, int, const char*);\n};", src)
        self.assertIn("struct GWidget : GObject {", src)
        self.assertIn("char _pad0[0x114];", src)
        self.assertIn("extern const char lbl_80390DE4[8];", src)
        self.assertIn('extern "C" GWidget* f(GUI* arg0) {\n    return new (0, lbl_80390DE4) GWidget(arg0);\n}', src)

    def test_the_null_check_is_only_dropped_around_a_constructor(self):
        insns, news = gen._new_expressions(gen.strip_frame(gen.parse(FACTORY)))
        self.assertEqual([i.op for i in insns][:8], ["li", "li", "mr", "li", "bl", "mr", "bl", "mr"])
        self.assertEqual(news, {6: 4})
        plain = FACTORY.replace("bl __nw__7GObjectFUliPCc", "bl fn_80001000")
        self.assertEqual(gen._new_expressions(gen.strip_frame(gen.parse(plain)))[1], {})


class OwnParamsTest(unittest.TestCase):
    """Parameter types from the function's own mangled name."""

    def test_reference_parameter(self):
        from tools.dk.platforms.gc_mwcc.demangle import demangle
        insns = gen.parse("/* 80001000 00000000  80 64 00 04 */\tlwz r3, 0x4(r4)\n/* 80001004 00000000  4E 80 00 20 */\tblr\n")
        lifted = gen.lift(insns, lambda s, n, o=0: None, True, own=demangle("get__1CFRC1G"))
        v = next(v for v in gen.variants(lifted) if v.ret_mode == "return")
        src, _ = gen.render_tu(lifted, True, "C", "get", [v], {}, standalone=True)
        self.assertIn("int C::get(const G& arg1) {\n    return arg1.m4;\n}", src)

    def test_unused_parameters_stay_in_the_signature(self):
        from tools.dk.platforms.gc_mwcc.demangle import demangle
        insns = gen.parse("/* 80001000 00000000  38 60 00 01 */\tli r3, 0x1\n/* 80001004 00000000  4E 80 00 20 */\tblr\n")
        lifted = gen.lift(insns, lambda s, n, o=0: None, False, own=demangle("f__FiPCcf"))
        v = next(v for v in gen.variants(lifted) if v.ret_mode == "return")
        src, _ = gen.render_tu(lifted, False, "G", "f", [v], {}, standalone=True)
        self.assertIn("int f(int arg0, const char* arg1, float farg0)", src)

    def test_operator_calls_are_member_calls(self):
        insns = gen.parse("/* 80001000 00000000  48 00 00 01 */\tbl __as__6GUIStrFRC6GUIStr\n"
                          "/* 80001004 00000000  4E 80 00 20 */\tblr\n")
        lifted = gen.lift(insns, lambda s, n, o=0: None)
        self.assertEqual(lifted.calls[0].cxx.name, "operator=")


class FunctionPointerTest(unittest.TestCase):
    def test_call_through_a_global_function_pointer(self):
        insns = gen.parse("/* 80001000 00000000  94 21 FF F0 */\tstwu r1, -0x10(r1)\n"
                          "/* 80001004 00000000  7C 08 02 A6 */\tmflr r0\n"
                          "/* 80001008 00000000  90 01 00 14 */\tstw r0, 0x14(r1)\n"
                          "/* 8000100C 00000000  38 80 00 3C */\tli r4, 0x3c\n"
                          "/* 80001010 00000000  81 8D B7 D4 */\tlwz r12, lbl_80390214@sda21(r0)\n"
                          "/* 80001014 00000000  7D 89 03 A6 */\tmtctr r12\n"
                          "/* 80001018 00000000  4E 80 04 21 */\tbctrl\n"
                          "/* 8000101C 00000000  80 01 00 14 */\tlwz r0, 0x14(r1)\n"
                          "/* 80001020 00000000  7C 08 03 A6 */\tmtlr r0\n"
                          "/* 80001024 00000000  38 21 00 10 */\taddi r1, r1, 0x10\n"
                          "/* 80001028 00000000  4E 80 00 20 */\tblr\n")
        lifted = gen.lift(insns, lambda s, n, o=0: None, False, params=(2, 0))
        (call,) = lifted.calls
        self.assertIsNotNone(call.fnptr)
        src, _ = gen.render_tu(lifted, False, "G", "f", [gen.Variant((), "void")], {}, standalone=True)
        self.assertIn("extern void (*lbl_80390214)(int, int);", src)
        self.assertIn("    lbl_80390214(arg0, 60);", src)


class CodeAddressTest(unittest.TestCase):
    def test_a_parameter_compared_with_a_static_function(self):
        from tools.dk.platforms.gc_mwcc.demangle import demangle
        insns = gen.parse("/* 8004C90C 00000000  3C 60 80 05 */\tlis r3, ClassName__3GUIFv@ha\n"
                          "/* 8004C910 00000000  38 03 C9 28 */\taddi r0, r3, ClassName__3GUIFv@l\n"
                          "/* 8004C914 00000000  7C 04 00 50 */\tsubf r0, r4, r0\n"
                          "/* 8004C918 00000000  7C 00 00 34 */\tcntlzw r0, r0\n"
                          "/* 8004C91C 00000000  54 00 D9 7E */\tsrwi r0, r0, 5\n"
                          "/* 8004C920 00000000  54 03 06 3E */\tclrlwi r3, r0, 24\n"
                          "/* 8004C924 00000000  4E 80 00 20 */\tblr\n")
        lifted = gen.lift(insns, lambda s, n, o=0: None, True, own=demangle("vf3__3GUIFPFv_PCc"))
        v = next(v for v in gen.variants(lifted) if v.ret_mode == "return")
        src, _ = gen.render_tu(lifted, True, "GUI", "vf3", [v], {}, standalone=True)
        self.assertIn("    static const char* ClassName();", src)
        self.assertIn("int GUI::vf3(const char* (*arg1)()) {\n    return (arg1 == &GUI::ClassName) & 0xFF;", src)
        self.assertNotIn("extern", src)


class MangledNameTest(unittest.TestCase):
    """A mangled name is never drafted as an extern "C" function (that duplicates the member)."""

    def test_nested_class_member_and_free_function(self):
        from tools.dk.platforms.gc_mwcc import gencmd
        self.assertEqual(gencmd._mangled_prefix("fn_800CD9B0__Q22AI9AINetworkFv"), "fn_800CD9B0")
        self.assertEqual(gencmd._mangled_prefix("__ct__Q22AI7CopBaseFPv"), "__ct")
        self.assertEqual(gencmd._cxx_free("fn_80084438__Fi"), "fn_80084438")
        self.assertIsNone(gencmd._cxx_free("fn_800CD9B0__Q22AI9AINetworkFv"))
        self.assertIsNone(gencmd._cxx_free("fn_80084438"))
        draft = 'extern "C" void fn_800CD9B0__Q22AI9AINetworkFv() {}\n'
        self.assertEqual([m[1] for m in gencmd._C_DEFINITION.finditer(draft)], ["fn_800CD9B0__Q22AI9AINetworkFv"])

    def test_free_cxx_function_head(self):
        from tools.dk.platforms.gc_mwcc.demangle import demangle
        insns = gen.parse("/* 80001000 00000000  1C 63 00 03 */\tmulli r3, r3, 0x3\n/* 80001004 00000000  4E 80 00 20 */\tblr\n")
        lifted = gen.lift(insns, lambda s, n, o=0: None, False, own=demangle("fn_80084438__Fi"))
        v = next(v for v in gen.variants(lifted) if v.ret_mode == "return")
        src, _ = gen.render_tu(lifted, False, "G", "fn_80084438", [v], {}, standalone=True, cxx_free=True)
        self.assertIn("int fn_80084438(int arg0) {", src)
        self.assertNotIn('extern "C"', src)
