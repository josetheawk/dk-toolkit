import unittest

from tools.dk.platforms.gc_mwcc.trycmd import definition, label, split_variants, spliced

SRC = 'int g;\nextern "C" int f(int a) {\n    return a + 1;\n}\nint h() { return 2; }\n'


class TryTest(unittest.TestCase):
    def test_variants_replace_the_definition(self):
        vs = split_variants("int f2;\nextern \"C\" int f(int a) {\n    return 1 + a;\n}\n// ---\n\n"
                            "extern \"C\" int f(int a) {\n    return a + 1;\n}\n")
        self.assertEqual(len(vs), 2)
        span = definition(SRC, "f")
        new = spliced(SRC, span, vs[0])
        self.assertEqual(new, 'int g;\nint f2;\nextern "C" int f(int a) {\n    return 1 + a;\n}\nint h() { return 2; }\n')
        self.assertEqual(label(SRC[span[0]:span[1]], vs[0]), "line 1: int f2;")
        self.assertEqual(label(SRC[span[0]:span[1]], vs[1]), "(same as the current definition)")

    def test_a_new_function_goes_in_by_address(self):
        from tools.dk.platforms.gc_mwcc.trycmd import insertion_point
        src = "int a() {\n    return 1;\n}\n\n// c's comment\nint c() {\n    return 3;\n}\n"
        at = insertion_point(src, 0x200, {"a": 0x100, "c": 0x300})
        self.assertEqual(src[at:].splitlines()[0], "// c's comment")
        new = spliced(src, (at, at), "int b() {\n    return 2;\n}\n")
        self.assertIn("}\n\nint b() {\n    return 2;\n}\n\n// c's comment\nint c()", new)
        self.assertEqual(insertion_point(src, 0x400, {"a": 0x100, "c": 0x300}), len(src))

    def test_a_mangled_symbol_finds_its_definition(self):
        src = "struct A { void m(int); };\nvoid A::m(int x) {\n}\n"
        self.assertEqual(definition(src, "m__1AFi"), (src.index("void A::m"), len(src) - 1))
        self.assertIsNone(definition(src, "missing__Fv"))

    def test_a_variant_rewrites_a_helper_too(self):
        from tools.dk.platforms.gc_mwcc.trycmd import variant_text
        src = ("static inline int h(int x) {\n    return x;\n}\n\n"
               "int f(int a) {\n    return h(a);\n}\n\nint g() { return h(1); }\n")
        chunk = "static inline int h(int x) {\n    return x + 0;\n}\n\nint f(int a) {\n    return h(a);\n}\n"
        new = variant_text(src, "f", chunk, definition(src, "f"))
        self.assertEqual(new.count("static inline int h("), 1)
        self.assertIn("return x + 0;", new)
        self.assertLess(new.index("int h("), new.index("int f("))
        self.assertIn("int g() { return h(1); }", new)
        plain = "int f(int a) {\n    return 2;\n}\n"
        self.assertEqual(variant_text(src, "f", plain, definition(src, "f")),
                         spliced(src, definition(src, "f"), plain))

    def test_a_member_is_found_by_its_class_and_parameters(self):
        src = ("class A { public: virtual int vf1(int); };\nclass B { public: virtual int vf1(int); };\n"
               "int A::vf1(int x) {\n    return x;\n}\n\nint B::vf1(int x) {\n    return x + 1;\n}\n"
               "namespace N {\nstruct C {\n    void f(int a) {\n    }\n    void f(int a, int b) {\n    }\n};\n}\n")
        a = definition(src, "vf1__1AFi")
        b = definition(src, "vf1__1BFi")
        self.assertEqual(src[a[0]:a[1]].splitlines()[0], "int A::vf1(int x) {")
        self.assertEqual(src[b[0]:b[1]].splitlines()[0], "int B::vf1(int x) {")
        two = definition(src, "f__Q21N1CFii")
        self.assertEqual(src[two[0]:two[1]].split("{")[0].strip(), "void f(int a, int b)")
        self.assertIsNone(definition("int B::vf1(int x) {\n    return x;\n}\n", "vf1__1AFi"))   # another class's
        self.assertIsNotNone(definition('extern "C" int g() {\n    return 1;\n}\n', "g"))


if __name__ == "__main__":
    unittest.main()


class ShowRowsTest(unittest.TestCase):
    """`--show` output: the row cap and its note (trycmd.clip_rows), and the option's default."""

    def test_clip_rows_keeps_the_cap_and_says_how_many_are_left(self):
        from tools.dk.platforms.gc_mwcc.trycmd import clip_rows
        rows = [f"r{i}" for i in range(5)]
        self.assertEqual(clip_rows(rows, 0), "r0\nr1\nr2\nr3\nr4")
        self.assertEqual(clip_rows(rows, 5), "r0\nr1\nr2\nr3\nr4")
        out = clip_rows(rows, 2)
        self.assertTrue(out.startswith("r0\nr1\n"))
        self.assertIn("3 more rows", out)
        self.assertIn("--show-rows 0", out)

    def test_show_rows_defaults_to_80_and_show_mentions_atoms(self):
        import argparse
        from tools.dk.platforms.gc_mwcc.trycmd import _configure
        p = argparse.ArgumentParser()
        _configure(p)
        a = p.parse_args(["u", "f"])
        self.assertEqual(a.show_rows, 80)
        self.assertEqual(a.show, -1)
        self.assertEqual(p.parse_args(["u", "f", "--show", "2", "--show-rows", "0"]).show_rows, 0)
        self.assertIn("atoms", next(x for x in p._actions if "--show" in x.option_strings).help)
