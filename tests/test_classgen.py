import unittest

from tools.dk.platforms.gc_mwcc import classgen, gen

DTOR = """
/* 8005164C 00000000  94 21 FF F0 */\tstwu r1, -0x10(r1)
/* 80051650 00000000  7C 08 02 A6 */\tmflr r0
/* 80051654 00000000  90 01 00 14 */\tstw r0, 0x14(r1)
/* 80051658 00000000  93 E1 00 0C */\tstw r31, 0xc(r1)
/* 8005165C 00000000  93 C1 00 08 */\tstw r30, 0x8(r1)
/* 80051660 00000000  7C 9F 23 78 */\tmr r31, r4
/* 80051664 00000000  7C 7E 1B 79 */\tmr. r30, r3
/* 80051668 00000000  41 82 00 28 */\tbeq .L_80051690
/* 8005166C 00000000  3C A0 80 24 */\tlis r5, __vt__1C@ha
/* 80051670 00000000  38 80 00 00 */\tli r4, 0x0
/* 80051674 00000000  38 05 00 00 */\taddi r0, r5, __vt__1C@l
/* 80051678 00000000  90 1E 00 00 */\tstw r0, 0x0(r30)
/* 8005167C 00000000  48 00 00 01 */\tbl fn_801EC05C
/* 80051680 00000000  7F E0 07 35 */\textsh. r0, r31
/* 80051684 00000000  40 81 00 0C */\tble .L_80051690
/* 80051688 00000000  7F C3 F3 78 */\tmr r3, r30
/* 8005168C 00000000  48 00 00 01 */\tbl __dl__FPv
.L_80051690:
/* 80051690 00000000  7F C3 F3 78 */\tmr r3, r30
/* 80051694 00000000  4E 80 00 20 */\tblr
"""
BASE_DTOR = """
/* 801EC05C 00000000  3C A0 80 24 */\tlis r5, __vt__1B@ha
/* 801EC060 00000000  38 05 00 00 */\taddi r0, r5, __vt__1B@l
/* 801EC064 00000000  90 1E 00 00 */\tstw r0, 0x0(r30)
"""
RECORDS = [{"name": "C", "rtti": 2, "bases": [[1, 0]]}, {"name": "B", "rtti": 1, "bases": []}]
CLASSES = {r["name"]: r for r in RECORDS}


class DtorTest(unittest.TestCase):
    def test_vtable_stores_and_chain(self):
        self.assertEqual(classgen.vtable_stores(gen.parse(DTOR)), [(("C",), 0)])
        self.assertEqual(classgen.chain_of(CLASSES, "C", RECORDS), ["C", "B"])

    def test_base_destructor_named_by_its_own_vtable_store(self):
        draft = classgen.solve_dtor(gen.parse(DTOR), CLASSES, {"fn_801EC05C": BASE_DTOR}.get, RECORDS)
        self.assertIsInstance(draft, classgen.ClassDraft)
        self.assertEqual(draft.renames, {"__dt__1BFv": "fn_801EC05C"})
        self.assertIn("struct C : B { virtual ~C(); };\nC::~C() {}", draft.source)

    def test_an_unrelated_callee_is_not_taken_for_the_base_destructor(self):
        unrelated = BASE_DTOR.replace("__vt__1B", "__vt__1X")
        result = classgen.solve_dtor(gen.parse(DTOR), CLASSES, {"fn_801EC05C": unrelated}.get, RECORDS)
        self.assertIsInstance(result, gen.Unsupported)

    def test_not_a_destructor(self):
        self.assertIsNone(classgen.solve_dtor(gen.parse(BASE_DTOR + "/* 801EC068 00000000  4E 80 00 20 */\tblr\n"),
                                              CLASSES, lambda n: None, RECORDS))


CTOR = """
/* 8004E8B8 00000000  94 21 FF F0 */\tstwu r1, -0x10(r1)
/* 8004E8BC 00000000  7C 08 02 A6 */\tmflr r0
/* 8004E8C0 00000000  90 01 00 14 */\tstw r0, 0x14(r1)
/* 8004E8C4 00000000  93 E1 00 0C */\tstw r31, 0xc(r1)
/* 8004E8C8 00000000  7C 7F 1B 78 */\tmr r31, r3
/* 8004E8CC 00000000  48 00 00 01 */\tbl __ct__1BFi
/* 8004E8D0 00000000  3C 60 80 24 */\tlis r3, __vt__1C@ha
/* 8004E8D4 00000000  38 00 00 00 */\tli r0, 0x0
/* 8004E8D8 00000000  38 63 00 00 */\taddi r3, r3, __vt__1C@l
/* 8004E8DC 00000000  90 7F 00 00 */\tstw r3, 0x0(r31)
/* 8004E8E0 00000000  90 1F 00 14 */\tstw r0, 0x14(r31)
/* 8004E8E4 00000000  7F E3 FB 78 */\tmr r3, r31
/* 8004E8E8 00000000  80 01 00 14 */\tlwz r0, 0x14(r1)
/* 8004E8EC 00000000  83 E1 00 0C */\tlwz r31, 0xc(r1)
/* 8004E8F0 00000000  7C 08 03 A6 */\tmtlr r0
/* 8004E8F4 00000000  38 21 00 10 */\taddi r1, r1, 0x10
/* 8004E8F8 00000000  4E 80 00 20 */\tblr
"""


class CtorTest(unittest.TestCase):
    def test_base_constructor_with_argument_and_field_init(self):
        drafts = classgen.solve_ctor(gen.parse(CTOR), CLASSES, lambda n: None, RECORDS, lambda s, n, o=0: None, None)
        self.assertIsInstance(drafts, list)
        src = drafts[0].source
        self.assertIn("struct B : ", src.replace("struct B {", "struct B : "))      # B is the root here
        self.assertIn("B(int);", src)
        self.assertIn("C::C(int arg1) : B(arg1) { m14 = 0; }", src)

    def test_destructors_are_not_constructors(self):
        self.assertIsNone(classgen.solve_ctor(gen.parse(DTOR), CLASSES, lambda n: None, RECORDS, lambda s, n, o=0: None, None))


MEMBER_DTOR = """
/* 80001000 00000000  94 21 FF F0 */\tstwu r1, -0x10(r1)
/* 80001004 00000000  7C 08 02 A6 */\tmflr r0
/* 80001008 00000000  90 01 00 14 */\tstw r0, 0x14(r1)
/* 8000100C 00000000  93 E1 00 0C */\tstw r31, 0xc(r1)
/* 80001010 00000000  7C 9F 23 78 */\tmr r31, r4
/* 80001014 00000000  93 C1 00 08 */\tstw r30, 0x8(r1)
/* 80001018 00000000  7C 7E 1B 79 */\tmr. r30, r3
/* 8000101C 00000000  41 82 00 3C */\tbeq .L_80001058
/* 80001020 00000000  80 7E 00 04 */\tlwz r3, 0x4(r30)
/* 80001024 00000000  48 00 00 01 */\tbl fn_80002000
/* 80001028 00000000  38 7E 00 08 */\taddi r3, r30, 0x8
/* 8000102C 00000000  38 80 FF FF */\tli r4, -0x1
/* 80001030 00000000  48 00 00 01 */\tbl fn_80003000
/* 80001034 00000000  7F C3 F3 78 */\tmr r3, r30
/* 80001038 00000000  38 80 00 00 */\tli r4, 0x0
/* 8000103C 00000000  48 00 00 01 */\tbl __dt__1BFv
/* 80001040 00000000  7F E0 07 35 */\textsh. r0, r31
/* 80001044 00000000  40 81 00 14 */\tble .L_80001058
/* 80001048 00000000  7F C3 F3 78 */\tmr r3, r30
/* 8000104C 00000000  48 00 00 01 */\tbl __dl__FPv
.L_80001058:
/* 80001058 00000000  80 01 00 14 */\tlwz r0, 0x14(r1)
/* 8000105C 00000000  7F C3 F3 78 */\tmr r3, r30
/* 80001060 00000000  83 E1 00 0C */\tlwz r31, 0xc(r1)
/* 80001064 00000000  83 C1 00 08 */\tlwz r30, 0x8(r1)
/* 80001068 00000000  7C 08 03 A6 */\tmtlr r0
/* 8000106C 00000000  38 21 00 10 */\taddi r1, r1, 0x10
/* 80001070 00000000  4E 80 00 20 */\tblr
"""
LEAF_DTOR = """
/* 80003000 00000000  7C 7F 1B 79 */\tmr. r31, r3
/* 80003004 00000000  41 82 00 14 */\tbeq .L_80003018
/* 80003008 00000000  7C 80 07 35 */\textsh. r0, r4
/* 8000300C 00000000  40 81 00 0C */\tble .L_80003018
/* 80003010 00000000  48 00 00 01 */\tbl __dl__FPv
.L_80003018:
/* 80003018 00000000  7F E3 FB 78 */\tmr r3, r31
/* 8000301C 00000000  4E 80 00 20 */\tblr
"""


class DtorModelTest(unittest.TestCase):
    def plan(self):
        asm = {"fn_80003000": LEAF_DTOR, "fn_80002000": "/* 80002000 00000000  4E 80 00 20 */\tblr\n"}
        return classgen.plan_dtor("__dt__1CFv", 0x80001000, gen.parse(MEMBER_DTOR), {}, asm.get, [],
                                  lambda s, n, o=0: None, lambda n: (1, 0))

    def test_frame(self):
        delete, synth = classgen.dtor_frame(gen.strip_frame(gen.parse(MEMBER_DTOR)))
        self.assertEqual(delete, "__dl__FPv")
        self.assertEqual((synth[0].op, synth[0].args), ("mr", ("r30", "r3")))
        self.assertEqual(synth[-1].op, "blr")
        self.assertEqual(len(synth), 1 + 8 + 1)          # this, the body and the destructor calls, blr
        self.assertTrue(classgen.looks_like_dtor(LEAF_DTOR))
        self.assertFalse(classgen.looks_like_dtor(CTOR))

    def test_members_base_and_body(self):
        plan = self.plan()
        self.assertIsInstance(plan, classgen.DtorPlan)
        self.assertEqual(plan.function, "__dt__1CFv")
        self.assertEqual(plan.model.objects, {8: ("Unk80003000", 0, 1)})
        self.assertEqual(plan.renames, {"__dt__11Unk80003000Fv": "fn_80003000"})
        self.assertEqual([type(x).__name__ for x in plan.lifted.tree], ["Call"])

    def test_rendered_destructor(self):
        plan = self.plan()
        v = next(v for v in gen.variants(plan.lifted) if v.ret_mode == "void")
        src, _ = gen.render_tu(plan.lifted, True, "C", "__dt", [v], {}, standalone=True, dtor=plan.model)
        self.assertIn("struct B {\n    char _pad0[0x1];\n    ~B();\n};", src)
        self.assertIn("struct C : B {", src)
        self.assertIn("    Unk80003000 m8;\n    ~C();\n};", src)
        self.assertIn("C::~C() {\n    fn_80002000(m4);\n}", src)

    def test_member_destructors_out_of_order(self):
        swapped = MEMBER_DTOR.replace("addi r3, r30, 0x8", "addi r3, r30, 0x0").replace("mr r3, r30\n/* 80001038", "addi r3, r30, 0x8\n/* 80001038")
        swapped = swapped.replace("li r4, 0x0\n/* 8000103C 00000000  48 00 00 01 */\tbl __dt__1BFv", "li r4, -0x1\n/* 8000103C 00000000  48 00 00 01 */\tbl __dt__1BFv")
        plan = classgen.plan_dtor("__dt__1CFv", 0x80001000, gen.parse(swapped), {}, {"fn_80003000": LEAF_DTOR}.get, [],
                                  lambda s, n, o=0: None, lambda n: (1, 0))
        self.assertIsInstance(plan, gen.Unsupported)


CTOR_MODEL = """
/* 80004000 00000000  94 21 FF F0 */\tstwu r1, -0x10(r1)
/* 80004004 00000000  7C 08 02 A6 */\tmflr r0
/* 80004008 00000000  90 01 00 14 */\tstw r0, 0x14(r1)
/* 8000400C 00000000  93 E1 00 0C */\tstw r31, 0xc(r1)
/* 80004010 00000000  7C 7F 1B 78 */\tmr r31, r3
/* 80004014 00000000  48 00 00 01 */\tbl __ct__1BFP1G
/* 80004018 00000000  3C 60 80 24 */\tlis r3, __vt__1C@ha
/* 8000401C 00000000  38 03 00 00 */\taddi r0, r3, __vt__1C@l
/* 80004020 00000000  38 80 00 00 */\tli r4, 0x0
/* 80004024 00000000  90 1F 00 00 */\tstw r0, 0x0(r31)
/* 80004028 00000000  90 9F 00 04 */\tstw r4, 0x4(r31)
/* 8000402C 00000000  38 7F 00 08 */\taddi r3, r31, 0x8
/* 80004030 00000000  48 00 00 01 */\tbl __ct__1MFv
/* 80004034 00000000  38 00 00 01 */\tli r0, 0x1
/* 80004038 00000000  90 1F 00 10 */\tstw r0, 0x10(r31)
/* 8000403C 00000000  7F E3 FB 78 */\tmr r3, r31
/* 80004040 00000000  80 01 00 14 */\tlwz r0, 0x14(r1)
/* 80004044 00000000  83 E1 00 0C */\tlwz r31, 0xc(r1)
/* 80004048 00000000  7C 08 03 A6 */\tmtlr r0
/* 8000404C 00000000  38 21 00 10 */\taddi r1, r1, 0x10
/* 80004050 00000000  4E 80 00 20 */\tblr
"""


class CtorModelTest(unittest.TestCase):
    def test_base_member_initializers_and_body(self):
        plan = classgen.plan_ctor("__ct__1CFP1G", 0x80004000, gen.parse(CTOR_MODEL), CLASSES, lambda n: None, RECORDS,
                                  lambda s, n, o=0: None, None)
        self.assertIsInstance(plan, classgen.CtorPlan)
        self.assertEqual((plan.model.base, plan.model.objects), ("B", {8: ("M", 0, 1)}))
        self.assertEqual([(o, type(x).__name__) for o, x in plan.model.inits], [(4, "Store"), (8, "MemberCtor")])
        v = next(v for v in gen.variants(plan.lifted) if v.ret_mode == "void")
        src, _ = gen.render_tu(plan.lifted, True, "C", "__ct", [v], {}, standalone=True, ctor_model=plan.model)
        self.assertIn("struct B {\n    virtual ~B();\n    B(G*);\n};", src)
        self.assertIn("struct M {\n    char _pad0[0x1];\n    M();\n};", src)
        self.assertIn("C::C(G* arg1) : B(arg1), m4(0) {\n    m10 = 1;\n}", src)

    def test_not_a_constructor_name(self):
        self.assertIsNone(classgen.plan_ctor("fn_80004000", 0x80004000, gen.parse(CTOR_MODEL), CLASSES, lambda n: None,
                                             RECORDS, lambda s, n, o=0: None, None))
