import unittest

from tools.dk.platforms.gc_mwcc import classgen, gen

# Hand-written PowerPC listings in dtk's form (synthetic addresses and fields; words encoded from the text).
DTOR = """
/* 80001000 00000000  94 21 FF 68 */\tstwu r1, -0x98(r1)
/* 80001004 00000000  7C 08 02 A6 */\tmflr r0
/* 80001008 00000000  90 01 00 9C */\tstw r0, 0x9c(r1)
/* 8000100C 00000000  93 81 00 90 */\tstw r28, 0x90(r1)
/* 80001010 00000000  93 A1 00 94 */\tstw r29, 0x94(r1)
/* 80001014 00000000  7C 9C 23 78 */\tmr r28, r4
/* 80001018 00000000  7C 7D 1B 79 */\tmr. r29, r3
/* 8000101C 00000000  41 82 00 28 */\tbeq .L_80001044
/* 80001020 00000000  3C C0 00 00 */\tlis r6, __vt__1C@ha
/* 80001024 00000000  38 80 00 00 */\tli r4, 0x0
/* 80001028 00000000  38 06 00 00 */\taddi r0, r6, __vt__1C@l
/* 8000102C 00000000  90 1D 00 00 */\tstw r0, 0x0(r29)
/* 80001030 00000000  48 00 00 01 */\tbl fn_80001A00
/* 80001034 00000000  7F 80 07 35 */\textsh. r0, r28
/* 80001038 00000000  40 81 00 0C */\tble .L_80001044
/* 8000103C 00000000  7F A3 EB 78 */\tmr r3, r29
/* 80001040 00000000  48 00 00 01 */\tbl __dl__FPv
.L_80001044:
/* 80001044 00000000  7F A3 EB 78 */\tmr r3, r29
/* 80001048 00000000  4E 80 00 20 */\tblr
"""
BASE_DTOR = """
/* 80001A00 00000000  3C 80 00 00 */\tlis r4, __vt__1B@ha
/* 80001A04 00000000  38 04 00 00 */\taddi r0, r4, __vt__1B@l
/* 80001A08 00000000  90 1E 00 00 */\tstw r0, 0x0(r30)
"""
RECORDS = [{"name": "C", "rtti": 2, "bases": [[1, 0]]}, {"name": "B", "rtti": 1, "bases": []}]
CLASSES = {r["name"]: r for r in RECORDS}


class DtorTest(unittest.TestCase):
    def test_vtable_stores_and_chain(self):
        self.assertEqual(classgen.vtable_stores(gen.parse(DTOR)), [(("C",), 0)])
        self.assertEqual(classgen.chain_of(CLASSES, "C", RECORDS), ["C", "B"])

    def test_base_destructor_named_by_its_own_vtable_store(self):
        draft = classgen.solve_dtor(gen.parse(DTOR), CLASSES, {"fn_80001A00": BASE_DTOR}.get, RECORDS)
        self.assertIsInstance(draft, classgen.ClassDraft)
        self.assertEqual(draft.renames, {"__dt__1BFv": "fn_80001A00"})
        self.assertIn("struct C : B { virtual ~C(); };\nC::~C() {}", draft.source)

    def test_an_unrelated_callee_is_not_taken_for_the_base_destructor(self):
        unrelated = BASE_DTOR.replace("__vt__1B", "__vt__1X")
        result = classgen.solve_dtor(gen.parse(DTOR), CLASSES, {"fn_80001A00": unrelated}.get, RECORDS)
        self.assertIsInstance(result, gen.Unsupported)

    def test_not_a_destructor(self):
        self.assertIsNone(classgen.solve_dtor(gen.parse(BASE_DTOR + "/* 80001A0C 00000000  4E 80 00 20 */\tblr\n"),
                                              CLASSES, lambda n: None, RECORDS))


CTOR = """
/* 80001100 00000000  94 21 FF 68 */\tstwu r1, -0x98(r1)
/* 80001104 00000000  7C 08 02 A6 */\tmflr r0
/* 80001108 00000000  90 01 00 9C */\tstw r0, 0x9c(r1)
/* 8000110C 00000000  93 E1 00 94 */\tstw r31, 0x94(r1)
/* 80001110 00000000  7C 7F 1B 78 */\tmr r31, r3
/* 80001114 00000000  48 00 00 01 */\tbl __ct__1BFi
/* 80001118 00000000  3C 60 00 00 */\tlis r3, __vt__1C@ha
/* 8000111C 00000000  38 00 00 04 */\tli r0, 0x4
/* 80001120 00000000  38 63 00 00 */\taddi r3, r3, __vt__1C@l
/* 80001124 00000000  90 7F 00 00 */\tstw r3, 0x0(r31)
/* 80001128 00000000  90 1F 00 24 */\tstw r0, 0x24(r31)
/* 8000112C 00000000  7F E3 FB 78 */\tmr r3, r31
/* 80001130 00000000  80 01 00 9C */\tlwz r0, 0x9c(r1)
/* 80001134 00000000  83 E1 00 94 */\tlwz r31, 0x94(r1)
/* 80001138 00000000  7C 08 03 A6 */\tmtlr r0
/* 8000113C 00000000  38 21 00 98 */\taddi r1, r1, 0x98
/* 80001140 00000000  4E 80 00 20 */\tblr
"""


class CtorTest(unittest.TestCase):
    def test_base_constructor_with_argument_and_field_init(self):
        drafts = classgen.solve_ctor(gen.parse(CTOR), CLASSES, lambda n: None, RECORDS, lambda s, n, o=0: None, None)
        self.assertIsInstance(drafts, list)
        src = drafts[0].source
        self.assertIn("struct B : ", src.replace("struct B {", "struct B : "))      # B is the root here
        self.assertIn("B(int);", src)
        self.assertIn("C::C(int arg1) : B(arg1) { m24 = 4; }", src)

    def test_destructors_are_not_constructors(self):
        self.assertIsNone(classgen.solve_ctor(gen.parse(DTOR), CLASSES, lambda n: None, RECORDS, lambda s, n, o=0: None, None))


MEMBER_DTOR = """
/* 80001200 00000000  94 21 FF 68 */\tstwu r1, -0x98(r1)
/* 80001204 00000000  7C 08 02 A6 */\tmflr r0
/* 80001208 00000000  90 01 00 9C */\tstw r0, 0x9c(r1)
/* 8000120C 00000000  93 81 00 90 */\tstw r28, 0x90(r1)
/* 80001210 00000000  7C 9C 23 78 */\tmr r28, r4
/* 80001214 00000000  93 A1 00 94 */\tstw r29, 0x94(r1)
/* 80001218 00000000  7C 7D 1B 79 */\tmr. r29, r3
/* 8000121C 00000000  41 82 00 34 */\tbeq .L_80001250
/* 80001220 00000000  80 7D 00 04 */\tlwz r3, 0x4(r29)
/* 80001224 00000000  48 00 00 01 */\tbl fn_80001B00
/* 80001228 00000000  38 7D 00 10 */\taddi r3, r29, 0x10
/* 8000122C 00000000  38 80 FF FF */\tli r4, -0x1
/* 80001230 00000000  48 00 00 01 */\tbl fn_80001C00
/* 80001234 00000000  7F A3 EB 78 */\tmr r3, r29
/* 80001238 00000000  38 80 00 00 */\tli r4, 0x0
/* 8000123C 00000000  48 00 00 01 */\tbl __dt__1BFv
/* 80001240 00000000  7F 80 07 35 */\textsh. r0, r28
/* 80001244 00000000  40 81 00 0C */\tble .L_80001250
/* 80001248 00000000  7F A3 EB 78 */\tmr r3, r29
/* 8000124C 00000000  48 00 00 01 */\tbl __dl__FPv
.L_80001250:
/* 80001250 00000000  80 01 00 9C */\tlwz r0, 0x9c(r1)
/* 80001254 00000000  7F A3 EB 78 */\tmr r3, r29
/* 80001258 00000000  83 81 00 90 */\tlwz r28, 0x90(r1)
/* 8000125C 00000000  83 A1 00 94 */\tlwz r29, 0x94(r1)
/* 80001260 00000000  7C 08 03 A6 */\tmtlr r0
/* 80001264 00000000  38 21 00 98 */\taddi r1, r1, 0x98
/* 80001268 00000000  4E 80 00 20 */\tblr
"""
LEAF_DTOR = """
/* 80001C00 00000000  7C 7E 1B 79 */\tmr. r30, r3
/* 80001C04 00000000  41 82 00 10 */\tbeq .L_80001C14
/* 80001C08 00000000  7C 85 07 35 */\textsh. r5, r4
/* 80001C0C 00000000  40 81 00 08 */\tble .L_80001C14
/* 80001C10 00000000  48 00 00 01 */\tbl __dl__FPv
.L_80001C14:
/* 80001C14 00000000  7F C3 F3 78 */\tmr r3, r30
/* 80001C18 00000000  4E 80 00 20 */\tblr
"""


class DtorModelTest(unittest.TestCase):
    def plan(self):
        asm = {"fn_80001C00": LEAF_DTOR, "fn_80001B00": "/* 80001B00 00000000  4E 80 00 20 */\tblr\n"}
        return classgen.plan_dtor("__dt__1CFv", 0x80001200, gen.parse(MEMBER_DTOR), {}, asm.get, [],
                                  lambda s, n, o=0: None, lambda n: (1, 0))

    def test_frame(self):
        delete, synth = classgen.dtor_frame(gen.strip_frame(gen.parse(MEMBER_DTOR)))
        self.assertEqual(delete, "__dl__FPv")
        self.assertEqual((synth[0].op, synth[0].args), ("mr", ("r29", "r3")))
        self.assertEqual(synth[-1].op, "blr")
        self.assertEqual(len(synth), 1 + 8 + 1)          # this, the body and the destructor calls, blr
        self.assertTrue(classgen.looks_like_dtor(LEAF_DTOR))
        self.assertFalse(classgen.looks_like_dtor(CTOR))

    def test_members_base_and_body(self):
        plan = self.plan()
        self.assertIsInstance(plan, classgen.DtorPlan)
        self.assertEqual(plan.function, "__dt__1CFv")
        self.assertEqual(plan.model.objects, {0x10: ("Unk80001C00", 0, 1)})
        self.assertEqual(plan.renames, {"__dt__11Unk80001C00Fv": "fn_80001C00"})
        self.assertEqual([type(x).__name__ for x in plan.lifted.tree], ["Call"])

    def test_rendered_destructor(self):
        plan = self.plan()
        v = next(v for v in gen.variants(plan.lifted) if v.ret_mode == "void")
        src, _ = gen.render_tu(plan.lifted, True, "C", "__dt", [v], {}, standalone=True, dtor=plan.model)
        self.assertIn("struct B {\n    char _pad0[0x1];\n    ~B();\n};", src)
        self.assertIn("struct C : B {", src)
        self.assertIn("    Unk80001C00 m10;\n    ~C();\n};", src)
        self.assertIn("C::~C() {\n    fn_80001B00(m4);\n}", src)

    def test_member_destructors_out_of_order(self):
        swapped = MEMBER_DTOR.replace("addi r3, r29, 0x10", "addi r3, r29, 0x0").replace("mr r3, r29\n/* 80001238", "addi r3, r29, 0x10\n/* 80001238")
        swapped = swapped.replace("li r4, 0x0\n/* 8000123C 00000000  48 00 00 01 */\tbl __dt__1BFv", "li r4, -0x1\n/* 8000123C 00000000  48 00 00 01 */\tbl __dt__1BFv")
        self.assertEqual(swapped.count("li r4, -0x1"), 2)
        plan = classgen.plan_dtor("__dt__1CFv", 0x80001200, gen.parse(swapped), {}, {"fn_80001C00": LEAF_DTOR}.get, [],
                                  lambda s, n, o=0: None, lambda n: (1, 0))
        self.assertIsInstance(plan, gen.Unsupported)


CTOR_MODEL = """
/* 80001400 00000000  94 21 FF 68 */\tstwu r1, -0x98(r1)
/* 80001404 00000000  7C 08 02 A6 */\tmflr r0
/* 80001408 00000000  90 01 00 9C */\tstw r0, 0x9c(r1)
/* 8000140C 00000000  93 E1 00 94 */\tstw r31, 0x94(r1)
/* 80001410 00000000  7C 7F 1B 78 */\tmr r31, r3
/* 80001414 00000000  48 00 00 01 */\tbl __ct__1BFP1G
/* 80001418 00000000  3C A0 00 00 */\tlis r5, __vt__1C@ha
/* 8000141C 00000000  38 80 00 00 */\tli r4, 0x0
/* 80001420 00000000  38 05 00 00 */\taddi r0, r5, __vt__1C@l
/* 80001424 00000000  90 1F 00 00 */\tstw r0, 0x0(r31)
/* 80001428 00000000  90 9F 00 08 */\tstw r4, 0x8(r31)
/* 8000142C 00000000  38 7F 00 0C */\taddi r3, r31, 0xc
/* 80001430 00000000  48 00 00 01 */\tbl __ct__1MFv
/* 80001434 00000000  38 00 00 02 */\tli r0, 0x2
/* 80001438 00000000  90 1F 00 18 */\tstw r0, 0x18(r31)
/* 8000143C 00000000  7F E3 FB 78 */\tmr r3, r31
/* 80001440 00000000  80 01 00 9C */\tlwz r0, 0x9c(r1)
/* 80001444 00000000  83 E1 00 94 */\tlwz r31, 0x94(r1)
/* 80001448 00000000  7C 08 03 A6 */\tmtlr r0
/* 8000144C 00000000  38 21 00 98 */\taddi r1, r1, 0x98
/* 80001450 00000000  4E 80 00 20 */\tblr
"""


class CtorModelTest(unittest.TestCase):
    def test_base_member_initializers_and_body(self):
        plan = classgen.plan_ctor("__ct__1CFP1G", 0x80001400, gen.parse(CTOR_MODEL), CLASSES, lambda n: None, RECORDS,
                                  lambda s, n, o=0: None, None)
        self.assertIsInstance(plan, classgen.CtorPlan)
        self.assertEqual((plan.model.base, plan.model.objects), ("B", {0xC: ("M", 0, 1)}))
        self.assertEqual([(o, type(x).__name__) for o, x in plan.model.inits], [(8, "Store"), (0xC, "MemberCtor")])
        v = next(v for v in gen.variants(plan.lifted) if v.ret_mode == "void")
        src, _ = gen.render_tu(plan.lifted, True, "C", "__ct", [v], {}, standalone=True, ctor_model=plan.model)
        self.assertIn("struct B {\n    virtual ~B();\n    B(G*);\n};", src)
        self.assertIn("struct M {\n    char _pad0[0x1];\n    M();\n};", src)
        self.assertIn("C::C(G* arg1) : B(arg1), m8(0) {\n    m18 = 2;\n}", src)

    def test_not_a_constructor_name(self):
        self.assertIsNone(classgen.plan_ctor("fn_80001400", 0x80001400, gen.parse(CTOR_MODEL), CLASSES, lambda n: None,
                                             RECORDS, lambda s, n, o=0: None, None))


def _asm(*lines: str) -> str:
    """Instruction lines at consecutive addresses from 0x80001000; the encoding is a dummy."""
    return "".join(f"/* {0x80001000 + 4 * i:08X} 00000000  60 00 00 00 */\t{x}\n" for i, x in enumerate(lines))


# `return new (0, "Box") Box(host);` with Box(Host* p) : Panel(p) {} inlined (the join is at 0x48)
INLINED_FACTORY = _asm(
    "stwu r1, -0x98(r1)", "mflr r0", "lis r4, lbl_80002E50@ha", "stw r0, 0x9c(r1)", "addi r5, r4, lbl_80002E50@l",
    "li r4, 0x0", "stw r29, 0x94(r1)", "stw r28, 0x90(r1)", "mr r28, r3", "li r3, 0x9c", "bl __nw__4NodeFUliPCc",
    "mr. r29, r3", "beq .L_80001048", "mr r4, r28", "bl __ct__5PanelFP4Host", "lis r3, __vt__3Box@ha",
    "addi r0, r3, __vt__3Box@l", "stw r0, 0x0(r29)", "lwz r0, 0x9c(r1)", "mr r3, r29", "lwz r29, 0x94(r1)",
    "lwz r28, 0x90(r1)", "mtlr r0", "addi r1, r1, 0x98", "blr")
HOST_RECORDS = [{"name": "Node", "rtti": 10, "bases": []}, {"name": "Host", "rtti": 11, "bases": [[10, 0]]},
                {"name": "Panel", "rtti": 12, "bases": [[10, 0], [11, 0]]},
                {"name": "Box", "rtti": 13, "bases": [[10, 0], [11, 0], [12, 0]]}]
HOST_CLASSES = {c["name"]: c for c in HOST_RECORDS}


class FactoryTest(unittest.TestCase):
    def test_a_new_expression_with_the_constructor_inlined(self):
        from tools.dk.platforms.gc_mwcc import gencmd
        plan = classgen.plan_factory("fn_80001000__3BoxFP4Host", 0x80001000, gen.parse(INLINED_FACTORY), HOST_CLASSES,
                                     lambda n: None, HOST_RECORDS, lambda s, n, o=0: None, None)
        self.assertIsInstance(plan, classgen.FactoryPlan)
        self.assertEqual((plan.size, plan.placement, plan.params, plan.ret, plan.owner, plan.method),
                         (0x9c, [("const", 0), ("addr", "lbl_80002E50")], ["Host*"], "Host*", "Box", "fn_80001000"))
        sources, error = gencmd.factory_drafts(plan, {}, {})
        self.assertIsNone(error)
        src = sources[0]
        self.assertIn("struct Node {\n    virtual ~Node();\n    static void* operator new(unsigned long, int, const char*);", src)
        self.assertIn("struct Box : Panel {\n    char _pad4[0x98];\n    Box(Host* arg1);\n    static Host* fn_80001000(Host*);", src)
        self.assertIn("inline Box::Box(Host* arg1) : Panel(arg1) {}", src)
        self.assertIn("extern const char lbl_80002E50[];\nHost* Box::fn_80001000(Host* arg0) {\n"
                      "    return new (0, lbl_80002E50) Box(arg0);\n}", src)

    def test_a_called_constructor_is_not_an_inlined_factory(self):
        called = INLINED_FACTORY.replace("lis r3, __vt__3Box@ha", "mr r29, r3").replace("addi r0, r3, __vt__3Box@l", "nop")
        self.assertIsNone(classgen.plan_factory("fn_80001000__3BoxFP4Host", 0x80001000, gen.parse(called), HOST_CLASSES,
                                                lambda n: None, HOST_RECORDS, lambda s, n, o=0: None, None))


# Key::~Key() with its member `Str m14` destroyed inline (Str : Node, `~Str() { m4 = 0; }`)
INLINE_MEMBER_DTOR = _asm(
    "stwu r1, -0x98(r1)", "mflr r0", "stw r0, 0x9c(r1)", "stw r28, 0x94(r1)", "mr r28, r4", "stw r29, 0x90(r1)",
    "mr. r29, r3", "beq .L_80001080", "lis r3, __vt__3Key@ha", "addi r0, r3, __vt__3Key@l", "stw r0, 0x0(r29)",
    "addic. r0, r29, 0x14", "beq .L_8000105C", "lis r3, __vt__3Str@ha", "addi r0, r3, __vt__3Str@l",
    "stw r0, 0x14(r29)", "li r0, 0x0", "stw r0, 0x18(r29)", "addic. r0, r29, 0x14", "beq .L_8000105C",
    "lis r3, __vt__4Node@ha", "addi r0, r3, __vt__4Node@l", "stw r0, 0x14(r29)",
    "cmplwi r29, 0x0", "beq .L_80001070", "lis r3, __vt__4Node@ha", "addi r0, r3, __vt__4Node@l",
    "stw r0, 0x0(r29)", "extsh. r0, r28", "ble .L_80001080", "mr r3, r29", "bl __dl__FPv",
    "lwz r0, 0x9c(r1)", "mr r3, r29", "lwz r28, 0x94(r1)", "lwz r29, 0x90(r1)", "mtlr r0", "addi r1, r1, 0x98", "blr")
KEY_RECORDS = [{"name": "Node", "rtti": 20, "bases": []}, {"name": "Key", "rtti": 21, "bases": [[20, 0]]},
               {"name": "Str", "rtti": 22, "bases": [[20, 0]]}]


class InlineMemberDtorTest(unittest.TestCase):
    def test_an_inlined_member_destructor_becomes_the_members_inline_destructor(self):
        from tools.dk.platforms.gc_mwcc import gencmd
        plan = classgen.plan_dtor("__dt__3KeyFv", 0x80001000, gen.parse(INLINE_MEMBER_DTOR),
                                  {c["name"]: c for c in KEY_RECORDS}, lambda n: None, KEY_RECORDS,
                                  lambda s, n, o=0: None, None)
        self.assertIsInstance(plan, classgen.DtorPlan)
        self.assertEqual([(o, p.model.cls) for o, p in plan.inline], [(0x14, "Str")])
        sources, error = gencmd.plan_drafts(plan, {})
        self.assertIsNone(error)
        src = sources[0]
        self.assertIn("struct Str : Node {\n    int m4;\n    virtual ~Str();\n};", src)
        self.assertIn("struct Key : Node {\n    char _pad4[0x10];\n    Str m14;\n    virtual ~Key();\n};", src)
        self.assertIn("inline Str::~Str() {\n    m4 = 0;\n}\nKey::~Key() {\n}", src)
        self.assertLess(src.index("struct Node {"), src.index("struct Str : Node {"))

    def test_merging_keeps_one_class_and_the_inline_definition(self):
        main = "struct B;\nstruct M {\n    char _pad0[0x4];\n    ~M();\n};\nstruct B {\n    virtual ~B();\n};\nvoid f() {}\n"
        member = "struct B;\nstruct B {\n    virtual ~B() {}\n};\nstruct M : B {\n    int m4;\n};\nextern int g;\nM::~M() {\n    m4 = g;\n}\n"
        merged = gen.merge_inline(main, member, "M")
        self.assertEqual(merged, "struct B;\nstruct B {\n    virtual ~B() {}\n};\nstruct M : B {\n    int m4;\n};\n"
                                 "extern int g;\ninline M::~M() {\n    m4 = g;\n}\nvoid f() {\n}\n")


class VtableStoreTest(unittest.TestCase):
    def test_a_local_objects_vtable_is_not_the_functions_own(self):
        insns = gen.parse(_asm("lis r4, __vt__3Vec@ha", "addi r0, r4, __vt__3Vec@l", "stw r0, 0xc(r1)",
                               "lis r4, __vt__3Box@ha", "addi r0, r4, __vt__3Box@l", "stw r0, 0x0(r3)", "blr"))
        self.assertEqual(classgen.vtable_stores(insns), [(("Box",), 0)])
