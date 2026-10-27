import unittest

from tools.dk.platforms.gc_mwcc import genmerge

A = """struct Foo;
struct Foo_m8;
struct Foo_m8 {
    char _pad0[0x4];
    int m4;
};
struct Foo {
    char _pad0[0x8];
    Foo_m8* m8;
    void fn_1();
};
extern int lbl_1;
void Foo::fn_1() {
    m8->m4 = lbl_1;
}
"""
B = """struct Foo;
struct Foo {
    char _pad0[0x4];
    float m4;
    char _pad8[0x4];
    unsigned char mC;
    char _padD[0x3];
    virtual void vf0();
    void fn_2();
};
void Foo::fn_2() {
    mC = 0;
}
"""
C = """struct Foo {
    char _pad0[0x4];
    int m4;
    void fn_3();
};
void Foo::fn_3() {}
"""


class MergeTest(unittest.TestCase):
    def test_fields_of_two_drafts_merge_by_offset(self):
        unit = genmerge.Unit().add(B, 2).add(A, 1)
        src = unit.render([])
        self.assertIn("struct Foo {\n    char _pad0[0x4];\n    float m4;\n    Foo_m8* m8;\n    unsigned char mC;\n"
                      "    char _padD[0x3];\n    virtual void vf0();\n    void fn_2();\n    void fn_1();\n};", src)
        self.assertLess(src.index("void Foo::fn_1()"), src.index("void Foo::fn_2()"))     # address order
        self.assertEqual(src.count("extern int lbl_1;"), 1)

    def test_disagreeing_drafts_conflict(self):
        unit = genmerge.Unit().add(B, 2)
        with self.assertRaises(genmerge.Conflict):
            unit.add(C, 3)                          # +0x4 float vs int
        with self.assertRaises(genmerge.Conflict):
            genmerge.Unit().add("extern int lbl_1;\n", 0).add("extern float lbl_1;\n", 1)

    def test_by_value_members_are_defined_first(self):
        src = genmerge.Unit().add("struct A {\n    B_elem m0[2];\n};\nstruct B_elem {\n    int m0;\n};\n", 0).render([])
        self.assertLess(src.index("struct B_elem {"), src.index("struct A {"))

    def test_a_derived_structs_fields_move_to_its_base(self):
        derived = "struct Foo {\n    void fn_1();\n};\nstruct Bar : Foo {\n    void fn_4();\n};\nvoid Bar::fn_4() {}\n"
        plain = "struct Bar {\n    char _pad0[0xC];\n    unsigned char mC;\n    void fn_5();\n};\nvoid Bar::fn_5() {}\n"
        for unit in (genmerge.Unit().add(B, 2).add(derived, 4).add(plain, 5),
                     genmerge.Unit().add(plain, 5).add(B, 2).add(derived, 4)):
            src = unit.render([])
            self.assertIn("struct Bar : Foo {\n    void fn_5();\n    void fn_4();\n};", src.replace(
                "    void fn_4();\n    void fn_5();\n", "    void fn_5();\n    void fn_4();\n"))
            self.assertIn("    unsigned char mC;\n", src[src.index("struct Foo {"):src.index("struct Bar :")])
            self.assertLess(src.index("struct Foo {"), src.index("struct Bar : Foo {"))
        with self.assertRaises(genmerge.Conflict):
            genmerge.Unit().add(derived, 4).add(derived.replace(": Foo", ": Baz").replace("fn_4", "fn_6"), 6)



SOURCE = """#include "h.h"

struct Foo {
    char _pad0[0x4];
    int m4;
    void fn_100();
};

void Foo::fn_100() {
    m4 = 1;
}

// a later function
void Foo::fn_300() {}

struct Late {
    int m0;
};
"""
PRE = ('/* #line 1\t"x.cpp"\t/* stack depth 0 */\n/* #line 1\t"h.h"\t/* stack depth 1 */\n'
       "class Hdr { public: int mine; };\nextern \"C\" void known(int);\n/* #line 2\t\"x.cpp\"\t/* stack depth 0 */\n")
DRAFT = """struct Foo;
struct Hdr;
struct New {
    char _pad0[0x8];
    int m8;
};
struct Foo {
    char _pad0[0x8];
    New* m8;
    void fn_200();
};
extern "C" void known(int);
extern "C" void fresh(void*);
void Foo::fn_200() {
    fresh(m8);
}
"""


class PasteTest(unittest.TestCase):
    def paste(self, probe=None):
        lines = SOURCE.splitlines()
        starts = {lines.index("void Foo::fn_100() {"): 0x100, lines.index("void Foo::fn_300() {}"): 0x300}
        return genmerge.Paste.of(SOURCE, PRE, starts, probe)

    def test_a_draft_goes_in_by_address_and_extends_the_sources_struct(self):
        src = self.paste().add(DRAFT, 0x200).render()
        self.assertIn("struct Foo {\n    char _pad0[0x4];\n    int m4;\n    New* m8;\n    void fn_100();\n    void fn_200();\n};", src)
        self.assertNotIn("void known(int);", src)                  # the header declares it
        self.assertNotIn("struct Hdr;", src)
        new, fn, later = src.index("struct New {"), src.index("void Foo::fn_200()"), src.index("// a later function")
        self.assertTrue(src.index("void Foo::fn_100()") < new < src.index('extern "C" void fresh(void*);') < fn < later)

    def test_a_struct_defined_below_the_function_moves_up(self):
        draft = "struct Late {\n    int m0;\n    int m4;\n};\nextern Late lbl_1;\nvoid fn_150() {\n    lbl_1.m4 = 0;\n}\n"
        src = self.paste().add(draft, 0x150).render()
        self.assertEqual(src.count("struct Late {"), 1)
        self.assertLess(src.index("struct Late {\n    int m0;\n    int m4;\n};"), src.index("void fn_150()"))

    def test_a_header_struct_must_have_the_drafts_members(self):
        draft = "struct Hdr {\n    int m0;\n    int m4;\n};\nextern Hdr lbl_2;\nvoid fn_250() {\n    lbl_2.m4 = 0;\n}\n"
        probe = lambda cls, names: {"m0": 0, "m4": None}
        with self.assertRaises(genmerge.Conflict) as e:
            self.paste(probe).add(draft, 0x250)
        self.assertIn("Hdr (the source's or a header's) has no member +0x4 int m4", str(e.exception))
        ok = self.paste(lambda cls, names: {"m0": 0, "m4": 4}).add(draft, 0x250).render()
        self.assertNotIn("struct Hdr {", ok)



HEADER = """class Box : public Base {
public:
    virtual void vf26(Box* child);
    int fn_10(void* handle); // a comment
    static const char* ClassName() { return "Box"; }
    char pad4[0x20 - 0x4];
};
"""


class HeaderTest(unittest.TestCase):
    def test_a_member_the_header_declares_takes_its_head(self):
        self.assertEqual(genmerge.member_declarations(HEADER, "Box", "vf26"), [("virtual void vf26(Box* child)", False)])
        self.assertEqual(genmerge.member_declarations(HEADER, "Box", "ClassName")[0][1], True)     # inline
        self.assertEqual(genmerge.declared_head("virtual int vf31(int key, float t, int repeat)", "Box", "vf31"),
                         "int Box::vf31(int arg1, float farg0, int arg2)")

    def test_merge_adopts_the_declared_head_and_skips_inline_members(self):
        pre = '/* #line 1\t"box.h"\t/* stack depth 1 */\n' + HEADER
        paste = genmerge.Paste.of('#include "box.h"\n', pre, {})
        draft = "struct Box {\n    int vf26();\n};\nint Box::vf26() {\n    return fn_10(0);\n}\n"
        src = paste.add(draft.replace("fn_10(0)", "1"), 0x100).render()
        self.assertIn("void Box::vf26(Box* arg1) {\n    1;\n}", src)          # declared void: no value back
        with self.assertRaises(genmerge.Conflict):
            paste.add("struct Box {\n    const char* ClassName();\n};\nconst char* Box::ClassName() {\n    return 0;\n}\n", 0x200)

    def test_missing_fields_are_split_out_of_the_padding(self):
        grown = genmerge.grow_class(HEADER, "Box", [(0x8, 4, "float m8;")], ["int fn_20(int arg1);"], {"pad4": (4, 0x1C)})
        self.assertIn("    char pad4[0x4];\n    float m8;\n    char padC[0x14];\n    int fn_20(int arg1);\n};", grown)
        self.assertIsNone(genmerge.grow_class(HEADER, "Box", [(0x40, 4, "int m40;")], [], {"pad4": (4, 0x1C)}))


if __name__ == "__main__":
    unittest.main()
