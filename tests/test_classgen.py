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
