import unittest

from tools.dk.platforms.gc_mwcc.renamecmd import c_uses
from tools.dk.platforms.gc_mwcc.stalecmd import _class_of, fix_text, stale_uses

NAMES = {0x80001000: "fn_80001000__3FooFv", 0x80002000: "fn_80002000"}


class CUsesTest(unittest.TestCase):
    def test_a_c_type_before_a_member_name_is_not_a_branch(self):
        text = "struct A { bool fn_80001000(int); };\nasm void f() {\n bl fn_80001000\n beq+ fn_80001000\n}\n"
        self.assertEqual(c_uses(text, {"fn_80001000"}), [("fn_80001000", 3), ("fn_80001000", 4)])


class StaleUseTest(unittest.TestCase):
    def test_a_called_c_name_is_renamed_in_its_file(self):
        text = 'extern "C" void fn_80001000(int* out);\nvoid g() { int x; fn_80001000(&x); }\n'
        [use] = stale_uses("src/a.cpp", text, NAMES)
        self.assertEqual((use.decl_lines, use.use_lines, use.member, use.defines), ([1], [2], False, False))
        self.assertEqual(fix_text(text, use),
                         'extern "C" void fn_80001000__3FooFv(int* out);\nvoid g() { int x; fn_80001000__3FooFv(&x); }\n')

    def test_an_unused_declaration_is_dropped(self):
        text = '#include "a.h"\nextern "C" void fn_80001000(int* out);\nvoid g() {}\n'
        [use] = stale_uses("src/a.cpp", text, NAMES)
        self.assertEqual(fix_text(text, use), '#include "a.h"\nvoid g() {}\n')

    def test_members_definitions_headers_and_current_names_are_left_alone(self):
        member = 'struct Foo { void fn_80001000(); };\nextern "C" void fn_80001000(void*);\nvoid g(Foo* f) { f->fn_80001000(); }\n'
        [use] = stale_uses("src/a.cpp", member, NAMES)
        self.assertTrue(use.member)
        self.assertIsNone(fix_text(member, use))
        definition = 'extern "C" void fn_80001000(int a) {\n}\n'
        [use] = stale_uses("src/a.cpp", definition, NAMES)
        self.assertTrue(use.defines)
        self.assertIsNone(fix_text(definition, use))
        header = 'extern "C" void fn_80001000(int* out);\ninline void g() { fn_80001000(0); }\n'
        [use] = stale_uses("include/a.h", header, NAMES)
        self.assertIsNone(fix_text(header, use))
        self.assertEqual(stale_uses("src/a.cpp", 'extern "C" void fn_80002000();\n', NAMES), [])

    def test_class_of_a_mangled_member(self):
        self.assertEqual(_class_of("fn_80214830__9GEditTextFv"), "GEditText")
        self.assertEqual(_class_of("vf49__9GEditTextFv"), "GEditText")
        self.assertIsNone(_class_of("fn_80002000"))


if __name__ == "__main__":
    unittest.main()
