import unittest

from tools.dk.platforms.gc_mwcc import gen

# Hand-written PowerPC listings in dtk's form (synthetic addresses and fields; words encoded from the text).
ASM = """
/* 80001000 00000000  38 00 00 07 */\tli r0, 0x7
/* 80001004 00000004  90 03 00 24 */\tstw r0, 0x24(r3)
/* 80001008 00000008  4E 80 00 20 */\tblr
"""
SWAP = """
/* 80001010 00000000  80 A3 01 D8 */\tlwz r5, 0x1d8(r3)
/* 80001014 00000000  90 83 01 D8 */\tstw r4, 0x1d8(r3)
/* 80001018 00000000  7C A3 2B 78 */\tmr r3, r5
/* 8000101C 00000000  4E 80 00 20 */\tblr
"""
no_literals = lambda sym, size: None


class GenTest(unittest.TestCase):
    def test_store_constant_member(self):
        lifted = gen.lift(gen.parse(ASM), no_literals)
        src, notes = gen.render_tu(lifted, True, "Foo", "fn_80001000", gen.variants(lifted), {}, display=True)
        self.assertIn("void Foo::fn_80001000() {\n    m24 = 7;\n}", src)
        self.assertEqual(notes, ["Foo: +0x24 int m24"])

    def test_old_value_is_hoisted_before_the_store(self):
        lifted = gen.lift(gen.parse(SWAP), no_literals)
        src, _ = gen.render_tu(lifted, True, "Foo", "f", gen.variants(lifted)[:1], {}, display=True)
        self.assertIn("int old = m1D8;\n    m1D8 = arg1;\n    return old;", src)

    def test_thunks_and_unknown_instructions_are_refused(self):
        thunk = gen.parse("/* 80001020 00000000  38 63 FF E8 */\tsubi r3, r3, 0x18\n/* 80001024 00000000  48 00 00 00 */\tb fn_80001100\n")
        self.assertTrue(gen.thunk_kind(thunk))
        with self.assertRaises(gen.Unsupported):
            gen.lift(gen.parse("/* 80001030 00000000  7C 63 20 2E */\tlwzx r3, r3, r4\n/* 80001034 00000000  4E 80 00 20 */\tblr\n"), no_literals)
        with self.assertRaises(gen.Unsupported):      # r3 + constant with no evidence it is a pointer
            gen.lift(gen.parse("/* 80001030 00000000  38 63 00 2C */\taddi r3, r3, 0x2c\n/* 80001034 00000000  4E 80 00 20 */\tblr\n"), no_literals)

    def test_float_literal_round_trips(self):
        self.assertEqual(gen.float_literal(bytes.fromhex("3eaaaa9f")), "0.333333f")
        self.assertEqual(gen.float_literal(bytes.fromhex("3f800000")), "1.0f")
        self.assertEqual(gen.float_literal(bytes.fromhex("42700000")), "60.0f")
        self.assertTrue(gen.float_literal(bytes.fromhex("7f7fffff")).endswith("f"))      # FLT_MAX


class CompareTest(unittest.TestCase):
    """The verifier must reject a relocation to the right symbol with the wrong addend."""

    def test_addend_mismatch_is_rejected(self):
        from tools.dk.platforms.gc_mwcc.elf import Function, Reloc
        insns = gen.parse("/* 80001040 00000000  80 6D 00 00 */\tlwz r3, lbl_80002E00@sda21(r0)\n"
                          "/* 80001044 00000000  4E 80 00 20 */\tblr\n")

        class FakeElf:
            def __init__(self, addend):
                self.fn = Function("v0", ".text", bytes.fromhex("806000004e800020"),
                                   [Reloc(0, gen.R_PPC_EMB_SDA21, "lbl_80002E00", addend)])

            def functions(self):
                return {"v0": self.fn}

        no_lit = lambda sym, size, offset=0: None
        self.assertIsNone(gen.compare(insns, FakeElf(0), "v0", no_lit))
        self.assertIsNotNone(gen.compare(insns, FakeElf(4), "v0", no_lit))

    def test_two_relocations_on_one_word_are_rejected(self):
        from tools.dk.platforms.gc_mwcc.elf import Function, Reloc
        insns = gen.parse("/* 80001040 00000000  80 6D 00 00 */\tlwz r3, lbl_80002E00@sda21(r0)\n"
                          "/* 80001044 00000000  4E 80 00 20 */\tblr\n")

        class FakeElf:
            def functions(self):
                return {"v0": Function("v0", ".text", bytes.fromhex("806000004e800020"),
                                       [Reloc(0, gen.R_PPC_EMB_SDA21, "lbl_80002E00", 0),
                                        Reloc(2, gen.R_PPC_ADDR16_LO, "lbl_80002E00", 0)])}
        self.assertIsNotNone(gen.compare(insns, FakeElf(), "v0", lambda s, n, o=0: None))

    def test_literal_must_be_in_the_same_section(self):
        from tools.dk.platforms.gc_mwcc.elf import ElfSection, Function, Reloc
        insns = gen.parse("/* 80001040 00000000  C0 22 00 00 */\tlfs f1, lbl_80002E08@sda21(r0)\n"
                          "/* 80001044 00000000  4E 80 00 20 */\tblr\n")
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
/* 80001100 00000000  94 21 FF 68 */\tstwu r1, -0x98(r1)
/* 80001104 00000000  7C 08 02 A6 */\tmflr r0
/* 80001108 00000000  90 01 00 9C */\tstw r0, 0x9c(r1)
/* 8000110C 00000000  93 E1 00 94 */\tstw r31, 0x94(r1)
/* 80001110 00000000  7C 7F 1B 78 */\tmr r31, r3
/* 80001114 00000000  48 00 00 01 */\tbl fn_80002000
/* 80001118 00000000  7F E3 FB 78 */\tmr r3, r31
/* 8000111C 00000000  48 00 00 01 */\tbl fn_80002080
/* 80001120 00000000  80 01 00 9C */\tlwz r0, 0x9c(r1)
/* 80001124 00000000  83 E1 00 94 */\tlwz r31, 0x94(r1)
/* 80001128 00000000  7C 08 03 A6 */\tmtlr r0
/* 8000112C 00000000  38 21 00 98 */\taddi r1, r1, 0x98
/* 80001130 00000000  4E 80 00 20 */\tblr
"""


class CallTest(unittest.TestCase):
    def test_frame_and_calls(self):
        sigs = {"fn_80002000": (0, 0), "fn_80002080": (1, 0)}
        lifted = gen.lift(gen.parse(CALLS), no_literals, True, sigs.get)
        self.assertEqual([c.callee for c in lifted.calls], ["fn_80002000", "fn_80002080"])
        src, _ = gen.render_tu(lifted, True, "Registry", "fn_80001100", gen.variants(lifted)[:1], {}, display=True)
        self.assertIn("void Registry::fn_80001100() {\n    fn_80002000();\n    fn_80002080(this);\n}", src)
        self.assertIn('extern "C" void fn_80002000();', src)

    def test_signature_inference(self):
        callee = gen.parse("/* 80001170 00000000  90 A3 00 30 */\tstw r5, 0x30(r3)\n"
                           "/* 80001174 00000000  4E 80 00 20 */\tblr\n")
        self.assertEqual(gen.infer_signature(callee), (3, 0))      # reads r3 and r5: three int params

    def test_value_loaded_before_a_call_is_kept_in_a_temporary(self):
        asm = CALLS.replace("\tmr r31, r3", "\tlwz r31, 0x30(r3)")
        lifted = gen.lift(gen.parse(asm), no_literals, True, {"fn_80002000": (0, 0), "fn_80002080": (1, 0)}.get)
        src, _ = gen.render_tu(lifted, True, "C", "f", gen.variants(lifted)[:1], {}, display=True)
        self.assertIn("int old = m30;\n    fn_80002000();\n    fn_80002080(old);", src)


GUARDED = """
/* 80001140 00000000  94 21 FF 68 */\tstwu r1, -0x98(r1)
/* 80001144 00000000  7C 08 02 A6 */\tmflr r0
/* 80001148 00000000  90 01 00 9C */\tstw r0, 0x9c(r1)
/* 8000114C 00000000  80 63 02 E4 */\tlwz r3, 0x2e4(r3)
/* 80001150 00000000  28 03 00 00 */\tcmplwi r3, 0x0
/* 80001154 00000000  41 82 00 08 */\tbeq .L_8000115C
/* 80001158 00000000  48 00 00 01 */\tbl fn_80002100
.L_8000115C:
/* 8000115C 00000000  80 01 00 9C */\tlwz r0, 0x9c(r1)
/* 80001160 00000000  7C 08 03 A6 */\tmtlr r0
/* 80001164 00000000  38 21 00 98 */\taddi r1, r1, 0x98
/* 80001168 00000000  4E 80 00 20 */\tblr
"""
EARLY = """
/* 80001180 00000000  80 63 00 9C */\tlwz r3, 0x9c(r3)
/* 80001184 00000000  28 03 00 00 */\tcmplwi r3, 0x0
/* 80001188 00000000  41 82 00 0C */\tbeq .L_80001194
/* 8000118C 00000000  80 63 02 14 */\tlwz r3, 0x214(r3)
/* 80001190 00000000  4E 80 00 20 */\tblr
.L_80001194:
/* 80001194 00000000  38 60 03 E9 */\tli r3, 0x3e9
/* 80001198 00000000  4E 80 00 20 */\tblr
"""


class FlowTest(unittest.TestCase):
    def test_guarded_call(self):
        lifted = gen.lift(gen.parse(GUARDED), no_literals, True, {"fn_80002100": (1, 0)}.get)
        self.assertEqual(gen.variants(lifted)[0].ret_mode, "void")
        src, notes = gen.render_tu(lifted, True, "C", "f", gen.variants(lifted)[:1], {}, display=True)
        self.assertIn("    if (m2E4 != 0) {\n        fn_80002100(m2E4);\n    }\n}", src)
        self.assertEqual(notes, ["C: +0x2E4 unsigned int m2E4"])     # cmplwi: unsigned

    def test_early_return(self):
        lifted = gen.lift(gen.parse(EARLY), no_literals, True)
        src, _ = gen.render_tu(lifted, True, "C", "f", [gen.Variant((), "return")], {}, display=True)
        self.assertIn("if (m9C != 0) {\n        return m9C->m214;\n    } else {\n        return 0x3E9;\n    }", src)

    def test_do_while_loop_with_a_loop_variable(self):
        asm = """
/* 80001400 00000000  94 21 FF 48 */\tstwu r1, -0xb8(r1)
/* 80001404 00000000  7C 08 02 A6 */\tmflr r0
/* 80001408 00000000  90 01 00 BC */\tstw r0, 0xbc(r1)
/* 8000140C 00000000  93 E1 00 B4 */\tstw r31, 0xb4(r1)
/* 80001410 00000000  93 C1 00 B0 */\tstw r30, 0xb0(r1)
/* 80001414 00000000  7C 7E 1B 78 */\tmr r30, r3
/* 80001418 00000000  3B E0 00 02 */\tli r31, 0x2
.L_8000141C:
/* 8000141C 00000000  7F C3 F3 78 */\tmr r3, r30
/* 80001420 00000000  7F E4 FB 78 */\tmr r4, r31
/* 80001424 00000000  48 00 00 01 */\tbl fn_80002200
/* 80001428 00000000  3B FF 00 01 */\taddi r31, r31, 0x1
/* 8000142C 00000000  2C 1F 00 07 */\tcmpwi r31, 0x7
/* 80001430 00000000  41 80 FF EC */\tblt .L_8000141C
/* 80001434 00000000  80 01 00 BC */\tlwz r0, 0xbc(r1)
/* 80001438 00000000  83 E1 00 B4 */\tlwz r31, 0xb4(r1)
/* 8000143C 00000000  83 C1 00 B0 */\tlwz r30, 0xb0(r1)
/* 80001440 00000000  7C 08 03 A6 */\tmtlr r0
/* 80001444 00000000  38 21 00 B8 */\taddi r1, r1, 0xb8
/* 80001448 00000000  4E 80 00 20 */\tblr
"""
        lifted = gen.lift(gen.parse(asm), no_literals, True, {"fn_80002200": (2, 0)}.get)
        src, _ = gen.render_tu(lifted, True, "C", "f", [gen.Variant((), "void")], {}, display=True)
        self.assertIn("int v31 = 2;\n    do {\n        fn_80002200(this, v31);\n        v31 = v31 + 1;\n    } while (v31 < 7);", src)

    def test_a_return_inside_a_loop(self):
        # for (N* x = head; x != 0; x = x->next) { if (x->ref->key == k) return x; } return x;
        asm = """
/* 80001600 00000000  80 63 00 10 */\tlwz r3, 0x10(r3)
/* 80001604 00000000  48 00 00 18 */\tb .L_8000161C
.L_80001608:
/* 80001608 00000000  80 A3 00 04 */\tlwz r5, 0x4(r3)
/* 8000160C 00000000  80 05 00 0C */\tlwz r0, 0xc(r5)
/* 80001610 00000000  7C 00 20 00 */\tcmpw r0, r4
/* 80001614 00000000  4D 82 00 20 */\tbeqlr
/* 80001618 00000000  80 63 00 D4 */\tlwz r3, 0xd4(r3)
.L_8000161C:
/* 8000161C 00000000  28 03 00 00 */\tcmplwi r3, 0x0
/* 80001620 00000000  40 82 FF E8 */\tbne .L_80001608
/* 80001624 00000000  4E 80 00 20 */\tblr
"""
        lifted = gen.lift(gen.parse(asm), no_literals, True)
        src, _ = gen.render_tu(lifted, True, "L", "find", [gen.Variant((), "return")], {}, display=True)
        self.assertIn("    while (v3 != 0) {\n        if (v3->m4->mC == arg1) {\n            return v3;\n        }\n"
                      "        v3 = v3->mD4;\n    }\n    return v3;\n}", src)      # (v3 is 0 there)

    def test_a_jump_out_of_a_loop_that_does_not_return_is_refused(self):
        asm = """
/* 80001680 00000000  80 63 00 10 */\tlwz r3, 0x10(r3)
/* 80001684 00000000  48 00 00 18 */\tb .L_8000169C
.L_80001688:
/* 80001688 00000000  80 A3 00 04 */\tlwz r5, 0x4(r3)
/* 8000168C 00000000  80 05 00 0C */\tlwz r0, 0xc(r5)
/* 80001690 00000000  7C 00 20 00 */\tcmpw r0, r4
/* 80001694 00000000  41 82 00 18 */\tbeq .L_800016AC
/* 80001698 00000000  80 63 00 D4 */\tlwz r3, 0xd4(r3)
.L_8000169C:
/* 8000169C 00000000  28 03 00 00 */\tcmplwi r3, 0x0
/* 800016A0 00000000  40 82 FF E8 */\tbne .L_80001688
/* 800016A4 00000000  38 60 00 02 */\tli r3, 0x2
/* 800016A8 00000000  90 64 00 10 */\tstw r3, 0x10(r4)
.L_800016AC:
/* 800016AC 00000000  90 64 00 14 */\tstw r3, 0x14(r4)
/* 800016B0 00000000  4E 80 00 20 */\tblr
"""
        with self.assertRaises(gen.Unsupported):
            gen.lift(gen.parse(asm), no_literals, True)

    def test_loop_closed_by_an_unconditional_branch_is_refused(self):
        loop = EARLY.replace(
            "/* 80001190 00000000  4E 80 00 20 */\tblr", "/* 80001190 00000000  4B FF FF F0 */\tb .L_80001180").replace(
            "80001180 00000000  80 63 00 9C */\tlwz", ".L_80001180:\n/* 80001180 00000000  80 63 00 9C */\tlwz")
        self.assertIn(".L_80001180:\n", loop)
        with self.assertRaises(gen.Unsupported):
            gen.lift(gen.parse(loop), no_literals, True)


VCALL = """
/* 80001700 00000000  94 21 FF 68 */\tstwu r1, -0x98(r1)
/* 80001704 00000000  7C 08 02 A6 */\tmflr r0
/* 80001708 00000000  90 01 00 9C */\tstw r0, 0x9c(r1)
/* 8000170C 00000000  80 63 00 1C */\tlwz r3, 0x1c(r3)
/* 80001710 00000000  28 03 00 00 */\tcmplwi r3, 0x0
/* 80001714 00000000  41 82 00 14 */\tbeq .L_80001728
/* 80001718 00000000  81 83 00 04 */\tlwz r12, 0x4(r3)
/* 8000171C 00000000  81 8C 01 48 */\tlwz r12, 0x148(r12)
/* 80001720 00000000  7D 89 03 A6 */\tmtctr r12
/* 80001724 00000000  4E 80 04 21 */\tbctrl
.L_80001728:
/* 80001728 00000000  80 01 00 9C */\tlwz r0, 0x9c(r1)
/* 8000172C 00000000  7C 08 03 A6 */\tmtlr r0
/* 80001730 00000000  38 21 00 98 */\taddi r1, r1, 0x98
/* 80001734 00000000  4E 80 00 20 */\tblr
"""


class ConditionalReturnTest(unittest.TestCase):
    def test_a_conditional_return_is_an_early_return(self):
        asm = """
/* 80001780 00000000  88 03 00 2A */\tlbz r0, 0x2a(r3)
/* 80001784 00000000  28 00 00 00 */\tcmplwi r0, 0x0
/* 80001788 00000000  4C 82 00 20 */\tbnelr
/* 8000178C 00000000  38 00 00 03 */\tli r0, 0x3
/* 80001790 00000000  98 03 00 41 */\tstb r0, 0x41(r3)
/* 80001794 00000000  4E 80 00 20 */\tblr
"""
        lifted = gen.lift(gen.parse(asm), no_literals, True)
        src, _ = gen.render_tu(lifted, True, "C", "f", [gen.Variant((), "void")], {}, display=True)
        self.assertIn("void C::f() {\n    if (m2A == 0) {\n        m41 = 3;\n    }\n}", src)

    def test_a_constructor_is_written_as_one(self):
        asm = """
/* 800017A0 00000000  38 00 00 05 */\tli r0, 0x5
/* 800017A4 00000000  90 03 00 0C */\tstw r0, 0xc(r3)
/* 800017A8 00000000  4E 80 00 20 */\tblr
"""
        lifted = gen.lift(gen.parse(asm), no_literals, True)
        void = [v for v in gen.variants(lifted) if v.ret_mode == "void"][:1]
        src, _ = gen.render_tu(lifted, True, "Pos", "__ct", void, {}, standalone=True)
        self.assertIn("    Pos();\n", src)
        self.assertIn("Pos::Pos() {\n    mC = 5;\n}", src)


class VirtualCallTest(unittest.TestCase):
    def test_slot_and_vptr_position(self):
        lifted = gen.lift(gen.parse(VCALL), no_literals, True)
        (call,) = lifted.calls
        self.assertTrue(call.virtual)
        self.assertEqual(call.callee, "vf80")                           # vtable + 0x148 = slot 80
        src, notes = gen.render_tu(lifted, True, "H", "f", gen.variants(lifted)[:1], {}, display=True)
        self.assertIn("virtual void vf80();", src)
        self.assertIn("if (m1C != 0) {\n        m1C->vf80();\n    }", src)
        self.assertIn("H_m1C: +0x4 vtable pointer (calls vf80)", notes)


MEMBER_CALL = """
/* 80001800 00000000  94 21 FF 68 */\tstwu r1, -0x98(r1)
/* 80001804 00000000  7C 08 02 A6 */\tmflr r0
/* 80001808 00000000  90 01 00 9C */\tstw r0, 0x9c(r1)
/* 8000180C 00000000  80 63 00 28 */\tlwz r3, 0x28(r3)
/* 80001810 00000000  48 00 00 01 */\tbl fn_80002300__Q22Ns4PoolFPCcPv
/* 80001814 00000000  80 01 00 9C */\tlwz r0, 0x9c(r1)
/* 80001818 00000000  7C 08 03 A6 */\tmtlr r0
/* 8000181C 00000000  38 21 00 98 */\taddi r1, r1, 0x98
/* 80001820 00000000  4E 80 00 20 */\tblr
"""


class CxxCallTest(unittest.TestCase):
    def test_member_call_binds_the_receiver_and_types_the_arguments(self):
        lifted = gen.lift(gen.parse(MEMBER_CALL), no_literals, True)
        (call,) = lifted.calls
        self.assertEqual(call.cxx.quals, ("Ns", "Pool"))
        src, _ = gen.render_tu(lifted, True, "Owner", "f", gen.variants(lifted)[:1], {}, display=True)
        self.assertIn("struct Ns { struct Pool; };", src)
        self.assertIn("    void fn_80002300(const char*, void*);", src)
        self.assertIn("void Owner::f(const char* arg1, void* arg2) {\n    m28->fn_80002300(arg1, arg2);\n}", src)

    def test_inherited_method_called_on_this_is_a_member_of_the_derived_class(self):
        asm = MEMBER_CALL.replace("\tlwz r3, 0x28(r3)", "\tstw r4, 0x2c(r3)").replace("Q22Ns4PoolFPCcPv", "4BaseFv")
        lifted = gen.lift(gen.parse(asm), no_literals, True)
        src, notes = gen.render_tu(lifted, True, "Base", "f", gen.variants(lifted)[:1], {}, standalone=True,
                                   chain=["Base", "Mid", "Leaf"])
        self.assertIn("struct Base {\n    char _pad0[0x2C];\n    int m2C;\n    void fn_80002300();\n};", src)
        self.assertIn("struct Mid : Base {\n};\nstruct Leaf : Mid {\n    void f(int arg1);\n};", src)
        self.assertIn("void Leaf::f(int arg1) {\n    m2C = arg1;\n    Base::fn_80002300();\n}", src)
        self.assertTrue(any(n.startswith("class chain (RTTI, root last): Leaf : Mid : Base") for n in notes))

    def test_a_slot_called_by_name_is_one_virtual_declaration(self):
        members = ["virtual void vf0();", "virtual void vf1();", "float vf1(int);", "void fn_1();"]
        self.assertEqual(gen._fold_called_slots(members), ["virtual void vf0();", "virtual float vf1(int);", "void fn_1();"])


I2F = """
/* 80001880 00000000  94 21 FF 18 */\tstwu r1, -0xe8(r1)
/* 80001884 00000000  80 63 00 48 */\tlwz r3, 0x48(r3)
/* 80001888 00000000  3C 00 43 30 */\tlis r0, 0x4330
/* 8000188C 00000000  C8 22 00 00 */\tlfd f1, lbl_80002E10@sda21(r0)
/* 80001890 00000000  6C 63 80 00 */\txoris r3, r3, 0x8000
/* 80001894 00000000  90 01 00 18 */\tstw r0, 0x18(r1)
/* 80001898 00000000  90 61 00 1C */\tstw r3, 0x1c(r1)
/* 8000189C 00000000  C8 01 00 18 */\tlfd f0, 0x18(r1)
/* 800018A0 00000000  EC 20 08 28 */\tfsubs f1, f0, f1
/* 800018A4 00000000  38 21 00 E8 */\taddi r1, r1, 0xe8
/* 800018A8 00000000  4E 80 00 20 */\tblr
"""


class ConversionTest(unittest.TestCase):
    def test_int_to_float_idiom(self):
        magic = lambda sym, size, offset=0: bytes.fromhex("4330000080000000") if sym == "lbl_80002E10" else None
        lifted = gen.lift(gen.parse(I2F), magic, True)
        src, _ = gen.render_tu(lifted, True, "C", "f", gen.variants(lifted)[:1], {}, display=True)
        self.assertIn("float C::f() {\n    return (float)m48;\n}", src)

    def test_other_stack_use_is_refused(self):
        with self.assertRaises(gen.Unsupported):
            gen.lift(gen.parse(I2F.replace("lfd f0, 0x18(r1)", "lfd f0, 0x20(r1)")), lambda s, n, o=0: None, True)


class ArrayTest(unittest.TestCase):
    def lift(self, body):
        return gen.lift(gen.parse(body), no_literals, True)

    def test_embedded_int_array(self):
        lifted = self.lift("/* 80001900 00000000  54 85 10 3A */\tslwi r5, r4, 2\n"
                           "/* 80001904 00000000  7C 63 2A 14 */\tadd r3, r3, r5\n"
                           "/* 80001908 00000000  80 63 00 34 */\tlwz r3, 0x34(r3)\n"
                           "/* 8000190C 00000000  4E 80 00 20 */\tblr\n")
        src, _ = gen.render_tu(lifted, True, "C", "f", gen.variants(lifted)[:1], {}, display=True)
        self.assertIn("return m34[arg1];", src)

    def test_pointer_to_struct_elements(self):
        lifted = self.lift("/* 80001920 00000000  80 63 00 50 */\tlwz r3, 0x50(r3)\n"
                           "/* 80001924 00000000  54 85 20 36 */\tslwi r5, r4, 4\n"
                           "/* 80001928 00000000  7C 63 2A 14 */\tadd r3, r3, r5\n"
                           "/* 8000192C 00000000  4E 80 00 20 */\tblr\n")
        src, _ = gen.render_tu(lifted, True, "C", "f", gen.variants(lifted)[:1], {}, display=True)
        self.assertIn("return &m50[arg1];", src)
        self.assertIn("char _pad0[0x10];", src)                    # the element is 16 bytes


class BitfieldTest(unittest.TestCase):
    def test_rotate_forms_become_shifts_and_masks(self):
        x = gen.Param("r3")
        self.assertEqual(gen._rotated(x, 20, gen._mask(31, 31)),                  # extrwi r, x, 1, 12
                         gen.Op("&", (gen.Op(">>u", (x, gen.Const(12)), "int"), gen.Const(1)), "int"))
        self.assertEqual(gen._rotated(x, 4, gen._mask(0, 27)), gen.Op("<<", (x, gen.Const(4)), "int"))   # slwi
        with self.assertRaises(gen.Unsupported):
            gen._rotated(x, 8, gen._mask(0, 31))                                  # a real rotate

    def test_insert_renders_as_mask_and_or(self):
        asm = ("/* 80001940 00000000  88 A3 00 06 */\tlbz r5, 0x6(r3)\n"
               "/* 80001944 00000000  50 85 16 FA */\trlwimi r5, r4, 2, 27, 29\n"
               "/* 80001948 00000000  98 A3 00 06 */\tstb r5, 0x6(r3)\n"
               "/* 8000194C 00000000  4E 80 00 20 */\tblr\n")
        lifted = gen.lift(gen.parse(asm), no_literals, True)
        src, _ = gen.render_tu(lifted, True, "C", "f", gen.variants(lifted)[:1], {}, display=True)
        self.assertIn("m6 = (m6 & ~0x1C) | ((arg1 << 2) & 0x1C);", src)


class MergedValueTest(unittest.TestCase):
    def test_a_value_merged_from_two_arms_becomes_a_variable(self):
        asm = """
/* 80001980 00000000  2C 04 00 00 */\tcmpwi r4, 0x0
/* 80001984 00000000  41 82 00 0C */\tbeq .L_80001990
/* 80001988 00000000  38 00 00 13 */\tli r0, 0x13
/* 8000198C 00000000  48 00 00 08 */\tb .L_80001994
.L_80001990:
/* 80001990 00000000  38 00 00 09 */\tli r0, 0x9
.L_80001994:
/* 80001994 00000000  90 03 00 14 */\tstw r0, 0x14(r3)
/* 80001998 00000000  4E 80 00 20 */\tblr
"""
        lifted = gen.lift(gen.parse(asm), no_literals, True)
        src, _ = gen.render_tu(lifted, True, "C", "f", [gen.Variant((), "void")], {}, display=True)
        self.assertIn("int m1;\n    if (arg1 != 0) {\n        m1 = 19;\n    } else {\n        m1 = 9;\n    }\n    m14 = m1;", src)


    def test_a_merged_return_value_is_returned_in_each_arm(self):
        asm = """
/* 800019C0 00000000  80 63 00 D8 */\tlwz r3, 0xd8(r3)
/* 800019C4 00000000  28 03 00 00 */\tcmplwi r3, 0x0
/* 800019C8 00000000  41 82 00 0C */\tbeq .L_800019D4
/* 800019CC 00000000  48 00 00 01 */\tbl fn_80002400
/* 800019D0 00000000  48 00 00 08 */\tb .L_800019D8
.L_800019D4:
/* 800019D4 00000000  38 60 00 00 */\tli r3, 0x0
.L_800019D8:
/* 800019D8 00000000  4E 80 00 20 */\tblr
"""
        lifted = gen.lift(gen.parse(asm), no_literals, True, lambda name: (1, 0))
        sink = [v for v in gen.variants(lifted) if v.sink]
        self.assertTrue(sink)
        src, _ = gen.render_tu(lifted, True, "C", "f", sink[:1], {}, display=True)
        self.assertIn("    if (mD8 != 0) {\n        return fn_80002400(mD8);\n    }\n    return 0;\n}", src)


class CallArgumentTest(unittest.TestCase):
    def test_varargs_call_takes_the_registers_written_before_it(self):
        asm = """
/* 80001A00 00000000  38 80 00 09 */\tli r4, 0x9
/* 80001A04 00000000  4C C6 31 82 */\tcrclr cr1eq
/* 80001A08 00000000  48 00 00 01 */\tbl printf
/* 80001A0C 00000000  4E 80 00 20 */\tblr
"""
        lifted = gen.lift(gen.parse(asm), no_literals, False, lambda name: (8, 8))
        src, _ = gen.render_tu(lifted, False, "G", "f", [gen.Variant((), "void")], {}, display=True)
        self.assertIn('extern "C" int printf(const char*, ...);', src)     # the C library's prototype
        self.assertIn("    printf(arg0, 9);", src)

    def test_a_locals_address_in_a_later_register_is_an_argument(self):
        asm = """
/* 80001A40 00000000  94 21 FF 28 */\tstwu r1, -0xd8(r1)
/* 80001A44 00000000  7C 08 02 A6 */\tmflr r0
/* 80001A48 00000000  90 01 00 DC */\tstw r0, 0xdc(r1)
/* 80001A4C 00000000  38 81 00 0C */\taddi r4, r1, 0xc
/* 80001A50 00000000  48 00 00 01 */\tbl fn_80002400
/* 80001A54 00000000  80 01 00 DC */\tlwz r0, 0xdc(r1)
/* 80001A58 00000000  7C 08 03 A6 */\tmtlr r0
/* 80001A5C 00000000  38 21 00 D8 */\taddi r1, r1, 0xd8
/* 80001A60 00000000  4E 80 00 20 */\tblr
"""
        lifted = gen.lift(gen.parse(asm), no_literals, False, lambda name: (1, 0))
        self.assertEqual(len(lifted.calls[0].args), 2)

    def test_registers_set_for_a_call_and_read_by_nothing_else_are_arguments(self):
        asm = """
/* 80001A80 00000000  38 A0 00 06 */\tli r5, 0x6
/* 80001A84 00000000  90 A3 00 10 */\tstw r5, 0x10(r3)
/* 80001A88 00000000  38 80 00 0B */\tli r4, 0xb
/* 80001A8C 00000000  48 00 00 01 */\tbl fn_80002400
/* 80001A90 00000000  4E 80 00 20 */\tblr
"""
        lifted = gen.lift(gen.parse(asm), no_literals, False, lambda name: (0, 0))
        self.assertEqual(len(lifted.calls[0].args), 2)     # r3, r4 (r5 was read by the store)

    def test_other_callers_show_a_passed_on_parameter(self):
        asm = """
/* 80001AA0 00000000  38 60 00 06 */\tli r3, 0x6
/* 80001AA4 00000000  48 00 00 01 */\tbl fn_80002400
/* 80001AA8 00000000  4E 80 00 20 */\tblr
"""
        sig, callers = lambda name: (1, 0), lambda name: (2, 0)
        known = gen.lift(gen.parse(asm), no_literals, False, sig, callers=callers, params=(2, 0))
        self.assertEqual(len(known.calls[0].args), 2)                # r4: this function's own parameter
        self.assertTrue(known.caller_evidence and known.review)
        src, _ = gen.render_tu(known, False, "G", "f", [gen.Variant((), "void")], {}, display=True)
        self.assertIn("    // REVIEW: fn_80002400 takes 2 int", src)
        unknown = gen.lift(gen.parse(asm), no_literals, False, sig, callers=callers, params=(0, 0))
        self.assertEqual(len(unknown.calls[0].args), 1)              # r4 holds nothing of this function's

    def test_a_result_used_once_by_the_next_call_is_nested(self):
        asm = """
/* 80001AC0 00000000  48 00 00 01 */\tbl fn_80002400
/* 80001AC4 00000000  48 00 00 01 */\tbl fn_80002480
/* 80001AC8 00000000  4E 80 00 20 */\tblr
"""
        lifted = gen.lift(gen.parse(asm), no_literals, False, lambda name: (0, 0) if name == "fn_80002400" else (1, 0))
        nest = [v for v in gen.variants(lifted) if v.nest and v.ret_mode == "void"]
        src, _ = gen.render_tu(lifted, False, "G", "f", nest[:1], {}, display=True)
        self.assertIn("    fn_80002480(fn_80002400());\n}", src)

class FloatCompareTest(unittest.TestCase):
    def _body(self, asm):
        lifted = gen.lift(gen.parse(asm), no_literals)
        src, _ = gen.render_tu(lifted, False, "", "f", gen.variants(lifted)[:1], {}, display=True)
        return src

    def test_cror_is_greater_or_equal_and_its_negation_stays_negated(self):
        asm = ("/* 80001B00 00000000  FC 03 08 40 */\tfcmpo cr0, f3, f1\n"
               "/* 80001B04 00000000  4C 41 13 82 */\tcror eq, gt, eq\n"
               "/* 80001B08 00000000  40 82 00 0C */\tbne .L_80001B14\n"
               "/* 80001B0C 00000000  38 00 00 02 */\tli r0, 0x2\n"
               "/* 80001B10 00000000  90 03 00 18 */\tstw r0, 0x18(r3)\n"
               "/* 80001B14 00000000  4E 80 00 20 */\tblr\n")
        self.assertIn("if (farg2 >= farg0)", self._body(asm))
        self.assertIn("if (!(farg2 >= farg0))", self._body(asm.replace("bne .L", "beq .L")))

    def test_plain_float_branch_tests_one_bit(self):
        asm = ("/* 80001B40 00000000  FC 03 08 40 */\tfcmpo cr0, f3, f1\n"
               "/* 80001B44 00000000  40 80 00 0C */\tbge .L_80001B50\n"
               "/* 80001B48 00000000  38 00 00 02 */\tli r0, 0x2\n"
               "/* 80001B4C 00000000  90 03 00 18 */\tstw r0, 0x18(r3)\n"
               "/* 80001B50 00000000  4E 80 00 20 */\tblr\n")
        self.assertIn("if (farg2 < farg0)", self._body(asm))
        self.assertIn("if (!(farg2 < farg0))", self._body(asm.replace("bge .L", "blt .L")))


class FrameTest(unittest.TestCase):
    def test_save_helper_with_a_scheduled_instruction_in_between(self):
        asm = ("/* 80001C00 00000000  94 21 FF 38 */\tstwu r1, -0xc8(r1)\n"
               "/* 80001C04 00000000  7C 08 02 A6 */\tmflr r0\n"
               "/* 80001C08 00000000  90 01 00 CC */\tstw r0, 0xcc(r1)\n"
               "/* 80001C0C 00000000  39 61 00 C8 */\taddi r11, r1, 0xc8\n"
               "/* 80001C10 00000000  48 00 00 01 */\tbl _savegpr_29\n"
               "/* 80001C14 00000000  7C 7D 1B 78 */\tmr r29, r3\n"
               "/* 80001C18 00000000  48 00 00 01 */\tbl fn_80002400\n"
               "/* 80001C1C 00000000  39 61 00 C8 */\taddi r11, r1, 0xc8\n"
               "/* 80001C20 00000000  7F A3 EB 78 */\tmr r3, r29\n"
               "/* 80001C24 00000000  48 00 00 01 */\tbl _restgpr_29\n"
               "/* 80001C28 00000000  80 01 00 CC */\tlwz r0, 0xcc(r1)\n"
               "/* 80001C2C 00000000  7C 08 03 A6 */\tmtlr r0\n"
               "/* 80001C30 00000000  38 21 00 C8 */\taddi r1, r1, 0xc8\n"
               "/* 80001C34 00000000  4E 80 00 20 */\tblr\n")
        body = gen.strip_frame(gen.parse(asm))
        self.assertEqual([i.op for i in body], ["mr", "bl", "mr", "blr"])

    def test_a_struct_local_can_be_sized_by_its_fields(self):
        asm = ("/* 80001C40 00000000  94 21 FF 28 */\tstwu r1, -0xd8(r1)\n"
               "/* 80001C44 00000000  7C 08 02 A6 */\tmflr r0\n"
               "/* 80001C48 00000000  90 01 00 DC */\tstw r0, 0xdc(r1)\n"
               "/* 80001C4C 00000000  C0 03 00 04 */\tlfs f0, 0x4(r3)\n"
               "/* 80001C50 00000000  D0 01 00 08 */\tstfs f0, 0x8(r1)\n"
               "/* 80001C54 00000000  D0 01 00 14 */\tstfs f0, 0x14(r1)\n"
               "/* 80001C58 00000000  38 61 00 08 */\taddi r3, r1, 0x8\n"
               "/* 80001C5C 00000000  48 00 00 01 */\tbl fn_80002400\n"
               "/* 80001C60 00000000  80 01 00 DC */\tlwz r0, 0xdc(r1)\n"
               "/* 80001C64 00000000  7C 08 03 A6 */\tmtlr r0\n"
               "/* 80001C68 00000000  38 21 00 D8 */\taddi r1, r1, 0xd8\n"
               "/* 80001C6C 00000000  4E 80 00 20 */\tblr\n")
        lifted = gen.lift(gen.parse(asm), no_literals, False, lambda name: (1, 0))
        self.assertEqual(lifted.locals, {8: 0xD0})                    # up to the frame's save area
        self.assertEqual(gen.tight_locals(lifted), {8: 0x10})         # m0 .. mC + 4


class CompoundConditionTest(unittest.TestCase):
    OR = ("/* 80001D00 00000000  80 03 00 10 */\tlwz r0, 0x10(r3)\n"
          "/* 80001D04 00000000  2C 00 00 03 */\tcmpwi r0, 0x3\n"
          "/* 80001D08 00000000  40 82 00 10 */\tbne .L_80001D18\n"
          "/* 80001D0C 00000000  80 03 00 14 */\tlwz r0, 0x14(r3)\n"
          "/* 80001D10 00000000  2C 00 00 03 */\tcmpwi r0, 0x3\n"
          "/* 80001D14 00000000  41 82 00 0C */\tbeq .L_80001D20\n"
          ".L_80001D18:\n"
          "/* 80001D18 00000000  38 00 00 05 */\tli r0, 0x5\n"
          "/* 80001D1C 00000000  90 03 00 20 */\tstw r0, 0x20(r3)\n"
          ".L_80001D20:\n"
          "/* 80001D20 00000000  4E 80 00 20 */\tblr\n")

    def test_or_reaching_one_block_from_two_tests(self):
        lifted = gen.lift(gen.parse(self.OR), no_literals, True)
        src, _ = gen.render_tu(lifted, True, "C", "f", gen.variants(lifted)[:1], {}, display=True)
        self.assertIn("if ((m10 != 3) || (m14 != 3)) {\n        m20 = 5;\n    }", src)

    def test_and_with_an_else_arm(self):
        asm = ("/* 80001D40 00000000  80 03 00 10 */\tlwz r0, 0x10(r3)\n"
               "/* 80001D44 00000000  2C 00 00 03 */\tcmpwi r0, 0x3\n"
               "/* 80001D48 00000000  41 82 00 1C */\tbeq .L_80001D64\n"
               "/* 80001D4C 00000000  80 03 00 14 */\tlwz r0, 0x14(r3)\n"
               "/* 80001D50 00000000  2C 00 00 03 */\tcmpwi r0, 0x3\n"
               "/* 80001D54 00000000  41 82 00 10 */\tbeq .L_80001D64\n"
               "/* 80001D58 00000000  38 00 00 05 */\tli r0, 0x5\n"
               "/* 80001D5C 00000000  90 03 00 20 */\tstw r0, 0x20(r3)\n"
               "/* 80001D60 00000000  48 00 00 0C */\tb .L_80001D6C\n"
               ".L_80001D64:\n"
               "/* 80001D64 00000000  38 00 00 06 */\tli r0, 0x6\n"
               "/* 80001D68 00000000  90 03 00 20 */\tstw r0, 0x20(r3)\n"
               ".L_80001D6C:\n"
               "/* 80001D6C 00000000  4E 80 00 20 */\tblr\n")
        lifted = gen.lift(gen.parse(asm), no_literals, True)
        src, _ = gen.render_tu(lifted, True, "C", "f", gen.variants(lifted)[:1], {}, display=True)
        self.assertIn("if ((m10 != 3) && (m14 != 3)) {\n        m20 = 5;\n    } else {\n        m20 = 6;\n    }", src)


class CountedLoopTest(unittest.TestCase):
    def test_bdnz_is_a_for_loop(self):
        asm = ("/* 80001D80 00000000  38 00 00 06 */\tli r0, 0x6\n"
               "/* 80001D84 00000000  7C 09 03 A6 */\tmtctr r0\n"
               ".L_80001D88:\n"
               "/* 80001D88 00000000  90 83 00 08 */\tstw r4, 0x8(r3)\n"
               "/* 80001D8C 00000000  38 63 00 0C */\taddi r3, r3, 0xc\n"
               "/* 80001D90 00000000  42 00 FF F8 */\tbdnz .L_80001D88\n"
               "/* 80001D94 00000000  4E 80 00 20 */\tblr\n")
        lifted = gen.lift(gen.parse(asm), no_literals)
        loop = next(st for st in lifted.tree if isinstance(st, gen.Loop))
        self.assertEqual((loop.kind, loop.cond), ("for", gen.Const(6)))
        src, _ = gen.render_tu(lifted, False, "", "f", gen.variants(lifted)[:1], {}, display=True)
        self.assertIn("for (i = 0; i < 6; i++) {", src)


# `return new (0, name) Panel(host);` with a placement operator new on the base class
FACTORY = """
/* 80001E00 00000000  94 21 FF 68 */\tstwu r1, -0x98(r1)
/* 80001E04 00000000  7C 08 02 A6 */\tmflr r0
/* 80001E08 00000000  38 AD 00 00 */\tli r5, lbl_80002E20@sda21
/* 80001E0C 00000000  38 80 00 00 */\tli r4, 0x0
/* 80001E10 00000000  90 01 00 9C */\tstw r0, 0x9c(r1)
/* 80001E14 00000000  93 E1 00 94 */\tstw r31, 0x94(r1)
/* 80001E18 00000000  7C 7F 1B 78 */\tmr r31, r3
/* 80001E1C 00000000  38 60 00 9C */\tli r3, 0x9c
/* 80001E20 00000000  48 00 00 01 */\tbl __nw__4NodeFUliPCc
/* 80001E24 00000000  7C 67 1B 79 */\tmr. r7, r3
/* 80001E28 00000000  41 82 00 10 */\tbeq .L_80001E38
/* 80001E2C 00000000  7F E4 FB 78 */\tmr r4, r31
/* 80001E30 00000000  48 00 00 01 */\tbl __ct__5PanelFP4Host
/* 80001E34 00000000  7C 67 1B 78 */\tmr r7, r3
.L_80001E38:
/* 80001E38 00000000  7C E3 3B 78 */\tmr r3, r7
/* 80001E3C 00000000  80 01 00 9C */\tlwz r0, 0x9c(r1)
/* 80001E40 00000000  83 E1 00 94 */\tlwz r31, 0x94(r1)
/* 80001E44 00000000  7C 08 03 A6 */\tmtlr r0
/* 80001E48 00000000  38 21 00 98 */\taddi r1, r1, 0x98
/* 80001E4C 00000000  4E 80 00 20 */\tblr
"""


class NewExpressionTest(unittest.TestCase):
    def test_new_with_placement_and_constructor(self):
        lifted = gen.lift(gen.parse(FACTORY), lambda s, n, o=0: None)
        (call,) = lifted.calls
        self.assertEqual((call.callee, call.new.callee, call.new.size, call.new.cls),
                         ("__ct__5PanelFP4Host", "__nw__4NodeFUliPCc", 0x9c, ("Panel",)))
        v = next(v for v in gen.variants(lifted) if v.ret_mode == "return")
        src, _ = gen.render_tu(lifted, False, "G", "f", [v], {"lbl_80002E20": 8}, standalone=True)
        self.assertIn("struct Node {\n    static void* operator new(unsigned long, int, const char*);\n};", src)
        self.assertIn("struct Panel : Node {", src)
        self.assertIn("char _pad0[0x9C];", src)
        self.assertIn("extern const char lbl_80002E20[8];", src)
        self.assertIn('extern "C" Panel* f(Host* arg0) {\n    return new (0, lbl_80002E20) Panel(arg0);\n}', src)

    def test_the_null_check_is_only_dropped_around_a_constructor(self):
        insns, news = gen._new_expressions(gen.strip_frame(gen.parse(FACTORY)))
        self.assertEqual([i.op for i in insns][:8], ["li", "li", "mr", "li", "bl", "mr", "bl", "mr"])
        self.assertEqual(news, {6: 4})
        plain = FACTORY.replace("bl __nw__4NodeFUliPCc", "bl fn_80002400")
        self.assertEqual(gen._new_expressions(gen.strip_frame(gen.parse(plain)))[1], {})


class OwnParamsTest(unittest.TestCase):
    """Parameter types from the function's own mangled name."""

    def test_reference_parameter(self):
        from tools.dk.platforms.gc_mwcc.demangle import demangle
        insns = gen.parse("/* 80001F00 00000000  80 64 00 18 */\tlwz r3, 0x18(r4)\n/* 80001F04 00000000  4E 80 00 20 */\tblr\n")
        lifted = gen.lift(insns, lambda s, n, o=0: None, True, own=demangle("get__1CFRC1G"))
        v = next(v for v in gen.variants(lifted) if v.ret_mode == "return")
        src, _ = gen.render_tu(lifted, True, "C", "get", [v], {}, standalone=True)
        self.assertIn("int C::get(const G& arg1) {\n    return arg1.m18;\n}", src)

    def test_unused_parameters_stay_in_the_signature(self):
        from tools.dk.platforms.gc_mwcc.demangle import demangle
        insns = gen.parse("/* 80001F10 00000000  38 60 00 04 */\tli r3, 0x4\n/* 80001F14 00000000  4E 80 00 20 */\tblr\n")
        lifted = gen.lift(insns, lambda s, n, o=0: None, False, own=demangle("f__FiPCcf"))
        v = next(v for v in gen.variants(lifted) if v.ret_mode == "return")
        src, _ = gen.render_tu(lifted, False, "G", "f", [v], {}, standalone=True)
        self.assertIn("int f(int arg0, const char* arg1, float farg0)", src)

    def test_operator_calls_are_member_calls(self):
        insns = gen.parse("/* 80001F20 00000000  48 00 00 01 */\tbl __as__4TextFRC4Text\n"
                          "/* 80001F24 00000000  4E 80 00 20 */\tblr\n")
        lifted = gen.lift(insns, lambda s, n, o=0: None)
        self.assertEqual(lifted.calls[0].cxx.name, "operator=")


class FunctionPointerTest(unittest.TestCase):
    def test_call_through_a_global_function_pointer(self):
        insns = gen.parse("/* 80001F40 00000000  94 21 FF 68 */\tstwu r1, -0x98(r1)\n"
                          "/* 80001F44 00000000  7C 08 02 A6 */\tmflr r0\n"
                          "/* 80001F48 00000000  90 01 00 9C */\tstw r0, 0x9c(r1)\n"
                          "/* 80001F4C 00000000  81 8D 00 00 */\tlwz r12, lbl_80002E30@sda21(r0)\n"
                          "/* 80001F50 00000000  38 80 00 19 */\tli r4, 0x19\n"
                          "/* 80001F54 00000000  7D 89 03 A6 */\tmtctr r12\n"
                          "/* 80001F58 00000000  4E 80 04 21 */\tbctrl\n"
                          "/* 80001F5C 00000000  80 01 00 9C */\tlwz r0, 0x9c(r1)\n"
                          "/* 80001F60 00000000  7C 08 03 A6 */\tmtlr r0\n"
                          "/* 80001F64 00000000  38 21 00 98 */\taddi r1, r1, 0x98\n"
                          "/* 80001F68 00000000  4E 80 00 20 */\tblr\n")
        lifted = gen.lift(insns, lambda s, n, o=0: None, False, params=(2, 0))
        (call,) = lifted.calls
        self.assertIsNotNone(call.fnptr)
        src, _ = gen.render_tu(lifted, False, "G", "f", [gen.Variant((), "void")], {}, standalone=True)
        self.assertIn("extern void (*lbl_80002E30)(int, int);", src)
        self.assertIn("    lbl_80002E30(arg0, 25);", src)


class CodeAddressTest(unittest.TestCase):
    def test_a_parameter_compared_with_a_static_function(self):
        from tools.dk.platforms.gc_mwcc.demangle import demangle
        insns = gen.parse("/* 80001F80 00000000  3C A0 00 00 */\tlis r5, TypeName__4HostFv@ha\n"
                          "/* 80001F84 00000000  38 A5 00 00 */\taddi r5, r5, TypeName__4HostFv@l\n"
                          "/* 80001F88 00000000  7C A4 28 50 */\tsubf r5, r4, r5\n"
                          "/* 80001F8C 00000000  7C A5 00 34 */\tcntlzw r5, r5\n"
                          "/* 80001F90 00000000  54 A5 D9 7E */\tsrwi r5, r5, 5\n"
                          "/* 80001F94 00000000  54 A3 06 3E */\tclrlwi r3, r5, 24\n"
                          "/* 80001F98 00000000  4E 80 00 20 */\tblr\n")
        lifted = gen.lift(insns, lambda s, n, o=0: None, True, own=demangle("vf5__4HostFPFv_PCc"))
        v = next(v for v in gen.variants(lifted) if v.ret_mode == "return")
        src, _ = gen.render_tu(lifted, True, "Host", "vf5", [v], {}, standalone=True)
        self.assertIn("    static const char* TypeName();", src)
        self.assertIn("int Host::vf5(const char* (*arg1)()) {\n    return (arg1 == &Host::TypeName) & 0xFF;", src)
        self.assertNotIn("extern", src)


class MangledNameTest(unittest.TestCase):
    """A mangled name is never drafted as an extern "C" function (that duplicates the member)."""

    def test_nested_class_member_and_free_function(self):
        from tools.dk.platforms.gc_mwcc import gencmd
        self.assertEqual(gencmd._mangled_prefix("fn_80002500__Q22Ns6NsLinkFv"), "fn_80002500")
        self.assertEqual(gencmd._mangled_prefix("__ct__Q22Ns8UnitBaseFPv"), "__ct")
        self.assertEqual(gencmd._cxx_free("fn_80002580__Fi"), "fn_80002580")
        self.assertIsNone(gencmd._cxx_free("fn_80002500__Q22Ns6NsLinkFv"))
        self.assertIsNone(gencmd._cxx_free("fn_80002580"))
        draft = 'extern "C" void fn_80002500__Q22Ns6NsLinkFv() {}\n'
        self.assertEqual([m[1] for m in gencmd._C_DEFINITION.finditer(draft)], ["fn_80002500__Q22Ns6NsLinkFv"])

    def test_free_cxx_function_head(self):
        from tools.dk.platforms.gc_mwcc.demangle import demangle
        insns = gen.parse("/* 80001FC0 00000000  1C 63 00 05 */\tmulli r3, r3, 0x5\n/* 80001FC4 00000000  4E 80 00 20 */\tblr\n")
        lifted = gen.lift(insns, lambda s, n, o=0: None, False, own=demangle("fn_80002580__Fi"))
        v = next(v for v in gen.variants(lifted) if v.ret_mode == "return")
        src, _ = gen.render_tu(lifted, False, "G", "fn_80002580", [v], {}, standalone=True, cxx_free=True)
        self.assertIn("int fn_80002580(int arg0) {", src)
        self.assertNotIn('extern "C"', src)


def _asm(*lines: str) -> str:
    """Instruction lines (op and operands) at consecutive addresses; the encoding is a dummy."""
    return "".join(f"/* {0x80001000 + 4 * i:08X} 00000000  60 00 00 00 */\t{x}\n" for i, x in enumerate(lines))


FRAME_IN = ("stwu r1, -0xd8(r1)", "mflr r0", "stw r0, 0xdc(r1)", "stw r31, 0xd4(r1)", "mr r31, r3")
FRAME_OUT = ("lwz r0, 0xdc(r1)", "lwz r31, 0xd4(r1)", "mtlr r0", "addi r1, r1, 0xd8", "blr")


class LocalObjectTest(unittest.TestCase):
    """A constructor run on a local's address declares the local (RAII); its destructor calls
    end the scope and leave the body."""

    def test_a_lock_held_for_the_whole_body(self):
        asm = _asm(*FRAME_IN, "addi r3, r1, 0x8", "lwz r4, 0x14(r31)", "bl __ct__10ScopedLockFPv",
                   "lwz r31, 0x18(r31)", "addi r3, r1, 0x8", "li r4, -0x1", "bl __dt__10ScopedLockFv",
                   "lwz r0, 0xdc(r1)", "mr r3, r31", "lwz r31, 0xd4(r1)", "mtlr r0", "addi r1, r1, 0xd8", "blr")
        lifted = gen.lift(gen.parse(asm), no_literals, True)
        self.assertEqual(list(lifted.scoped), [8])
        self.assertEqual([c.callee for c in lifted.calls], ["__ct__10ScopedLockFPv"])     # the destructor left
        gen.size_objects(lifted, lambda callee: 4)
        v = next(v for v in gen.variants(lifted) if v.ret_mode == "return")
        src, _ = gen.render_tu(lifted, True, "Queue", "f", [v], {}, standalone=True)
        self.assertIn("struct ScopedLock {\n    char _pad0[0x4];\n    ScopedLock(void*);\n    ~ScopedLock();\n};", src)
        self.assertIn("int Queue::f() {\n    ScopedLock l8(m14);\n    return m18;\n}", src)

    def test_a_field_read_through_the_constructors_result_is_a_temporary(self):
        asm = _asm(*FRAME_IN, "addi r3, r1, 0x8", "bl __ct__10NameHasherFPCc", "lwz r0, 0x0(r3)",
                   "stw r0, 0x1c(r31)", *FRAME_OUT)
        lifted = gen.lift(gen.parse(asm), no_literals, True, params=(2, 0))
        self.assertEqual(lifted.temp_reads, {8})
        self.assertEqual(gen.single_use_objects(lifted), {8})
        v = gen.variants(lifted)[0]
        self.assertTrue(v.temps)                        # read through r3: the temporary first
        src, _ = gen.render_tu(lifted, True, "C", "f", [v], {}, standalone=True)
        self.assertIn("    m1C = NameHasher(arg1).m0;\n", src)
        self.assertIn("struct NameHasher {\n    int m0;", src)

    def test_a_destructor_without_its_constructor_is_refused(self):
        asm = _asm(*FRAME_IN, "addi r3, r1, 0x8", "li r4, -0x1", "bl __dt__10ScopedLockFv", *FRAME_OUT)
        with self.assertRaises(gen.Unsupported):
            gen.lift(gen.parse(asm), no_literals, True)

    def test_constructor_extent(self):
        insns = gen.parse(_asm("mr r31, r3", "stw r4, 0x4(r31)", "bl fn_80002400", "stw r0, 0x10(r3)",
                               "stb r0, 0xc(r31)", "blr"))
        self.assertEqual(gen.object_extent(insns), 0xC + 1)      # (r3 is the call's result after the bl)


class FunctionPointerParamTest(unittest.TestCase):
    def test_classes_inside_a_function_pointer_parameter(self):
        from tools.dk.platforms.gc_mwcc.demangle import demangle
        fn = demangle("__ct__9TypeEntryFPCcPFP4Host_P4HostPCc")
        self.assertEqual(fn.params[1].text, "Host* (*)(Host*)")
        self.assertEqual(fn.params[1].classes, (("Host",),))


class StaticMemberTest(unittest.TestCase):
    def test_a_static_callee_takes_its_parameters_from_r3(self):
        asm = _asm("stwu r1, -0x98(r1)", "mflr r0", "stw r0, 0x9c(r1)", "lwz r3, 0x58(r3)",
                   "bl fn_80002600__4HostFPv", "lwz r0, 0x9c(r1)", "mtlr r0", "addi r1, r1, 0x98", "blr")
        lifted = gen.lift(gen.parse(asm), no_literals, True, statics=lambda callee: True)
        (call,) = lifted.calls
        self.assertTrue(call.static)
        v = next(v for v in gen.variants(lifted) if v.ret_mode == "return")
        src, _ = gen.render_tu(lifted, True, "C", "f", [v], {}, standalone=True)
        self.assertIn("    static int fn_80002600(void*);", src)
        self.assertIn("    return Host::fn_80002600(m58);", src)

    def test_argument_reads_count_a_call_as_reading_unwritten_registers(self):
        reads, bases = gen.argument_reads(gen.parse(_asm("lwz r0, 0x8(r3)", "li r4, 0x3", "bl fn_80002400", "blr")))
        self.assertEqual(reads, {"r3"} | {f"r{n}" for n in range(5, 11)})     # r4 was written first
        self.assertEqual(bases, {"r3"})

    def test_a_static_member_function_is_written_as_one(self):
        from tools.dk.platforms.gc_mwcc.demangle import demangle
        insns = gen.parse(_asm("lwz r3, lbl_80002E40@sda21(r0)", "blr"))
        lifted = gen.lift(insns, no_literals, False, own=demangle("fn_80002680__3GfxFv"))
        v = next(v for v in gen.variants(lifted) if v.ret_mode == "return")
        src, _ = gen.render_tu(lifted, False, "Gfx", "fn_80002680", [v], {}, standalone=True, static_in="Gfx")
        self.assertIn("struct Gfx {\n    static int fn_80002680();\n};", src)
        self.assertIn("int Gfx::fn_80002680() {\n    return lbl_80002E40;\n}", src)


class StaticInitializerTest(unittest.TestCase):
    def test_a_static_initializer_without_a_draft_is_skipped(self):
        from unittest import mock
        from tools.dk.platforms.gc_mwcc import gencmd
        refused = lambda ctx, name, work: gencmd.Result(name, 0x80001000, "unsupported", "calls a ctor (needs the class model)")
        with mock.patch.object(gencmd, "_solve", refused):
            r = gencmd.solve(None, "__sinit_Foo_cpp", None)
            self.assertEqual((r.status, r.detail),
                             ("skipped", "a static initializer: define the unit's globals (calls a ctor (needs the class model))"))
            self.assertEqual(gencmd.solve(None, "fn_80001000", None).status, "unsupported")
