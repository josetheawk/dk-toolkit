import unittest

from tests.test_shaperules import has, norm, outs

LOOP = """    int palette = 0;
    int i;
    int frames = 0;
    for (i = 0; i < 8; i++) {
        int speed = (a >> (i * 2)) & 3;
        if (speed == 0) {
            break;
        }
        p->m[i] = speed;
        frames++;
    }
    b = frames;
"""

FIXED = LOOP.replace("int palette = 0;", "int palette;").replace("int frames = 0;", "int frames;") \
    .replace("for (i = 0;", "for (i = 0, frames = 0, palette = 0;")


class ForInitListTest(unittest.TestCase):
    def test_moves_initialisers_into_the_init_list_in_every_order(self):
        got = outs("for-init-list", LOOP)
        want = norm(FIXED)
        self.assertIn(want, [norm(g) for g in got])
        self.assertTrue(any("for (i = 0, palette = 0, frames = 0;" in g for g in got), got)
        self.assertTrue(any("int palette; int i; int frames = 0; for (i = 0, palette = 0;" in norm(g) for g in got))

    def test_inverse_puts_them_back_on_the_declarations(self):
        got = outs("for-init-list", FIXED)
        self.assertTrue(has(got, LOOP.replace("int palette = 0;", "int palette;")
                            .replace("for (i = 0;", "for (i = 0, palette = 0;")), got)
        self.assertTrue(any("int palette = 0; int i = 0; int frames = 0; for (; i < 8;" in norm(g) for g in got), got)

    def test_empty_init(self):
        got = outs("for-init-list", "    int k = 0;\n    for (; a < 3; a++) {\n        k += a;\n    }\n")
        self.assertTrue(has(got, "int k; for (k = 0; a < 3; a++) { k += a; }"), got)

    def test_legality(self):
        # used between the declaration and the loop
        self.assertEqual(outs("for-init-list", "    int k = 0;\n    b = k;\n    for (a = 0; a < 3; a++) {\n        k++;\n    }\n"), [])
        # not a literal, a class type, a const, a label crossed
        self.assertEqual(outs("for-init-list", "    int k = b;\n    for (a = 0; a < 3; a++) {\n        k++;\n    }\n"), [])
        self.assertEqual(outs("for-init-list", "    Vec k = 0;\n    for (a = 0; a < 3; a++) {\n        k++;\n    }\n"), [])
        self.assertEqual(outs("for-init-list", "    const int k = 0;\n    for (a = 0; a < 3; a++) {\n        b += k;\n    }\n"), [])
        self.assertEqual(outs("for-init-list", "    int k = 0;\nL:\n    for (a = 0; a < 3; a++) {\n        k++;\n    }\n"), [])
        # read by the loop's own init: never moved after it
        self.assertEqual(outs("for-init-list", "    int k = 0;\n    for (a = k; a < 3; a++) {\n        b++;\n    }\n"), [])


class IvAliasTest(unittest.TestCase):
    def test_swaps_equal_counters_in_iv_expressions(self):
        got = outs("iv-alias", FIXED)
        self.assertTrue(any("(a >> (frames * 2))" in g for g in got), got)
        self.assertTrue(any("p->m[frames] = speed" in g for g in got), got)
        back = outs("iv-alias", FIXED.replace("(i * 2)", "(frames * 2)"))
        self.assertTrue(any("(a >> (i * 2))" in g for g in back), back)

    def test_entry_value_from_the_declaration(self):
        self.assertEqual(len(outs("iv-alias", LOOP)), 2)

    def test_legality(self):
        # a continue skips frames++ but not i++
        self.assertEqual(outs("iv-alias", FIXED.replace("break;", "continue;")), [])
        # different entry values, the counter written elsewhere, another type
        self.assertEqual(outs("iv-alias", FIXED.replace("frames = 0,", "frames = 1,")), [])
        self.assertEqual(outs("iv-alias", FIXED.replace("p->m[i] = speed;", "p->m[i] = speed;\n        frames += 2;")), [])
        self.assertEqual(outs("iv-alias", FIXED.replace("int frames;", "unsigned int frames;")), [])
        # the increment is not the body's last statement
        self.assertEqual(outs("iv-alias", FIXED.replace("        frames++;\n    }", "        frames++;\n        b = 1;\n    }")), [])
        # a use outside iv shapes is left alone
        self.assertFalse(any("b = frames;\n        int" in g for g in outs("iv-alias", FIXED.replace("int speed", "b = i;\n        int speed"))))


if __name__ == "__main__":
    unittest.main()
