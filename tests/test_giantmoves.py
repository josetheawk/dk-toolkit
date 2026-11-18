import unittest

from tests.test_shaperules import has, norm, outs

WIDE = """    float amount;
    if (a) {
        amount = x - 1.0f;
        p->m = amount;
    } else {
        amount = x;
        p->n = amount;
    }
"""

NARROW = WIDE.replace("    float amount;\n", "").replace("amount = x - 1.0f;", "float amount = x - 1.0f;") \
    .replace("amount = x;", "float amount = x;")

TAIL_ONCE = """    if (a) {
        p->m = 1;
    } else {
        p->m = 2;
        p->n = 3;
    }
    p->k = x;
"""

TAIL_BOTH = """    if (a) {
        p->m = 1;
        p->k = x;
    } else {
        p->m = 2;
        p->n = 3;
        p->k = x;
    }
"""


class ScopeLocalTest(unittest.TestCase):
    def test_narrows_a_function_level_local_to_its_branches(self):
        got = outs("scope-local", WIDE)
        self.assertTrue(has(got, NARROW), got)

    def test_hoists_two_block_locals_before_the_if_and_to_the_declarations(self):
        got = outs("scope-local", NARROW)
        self.assertTrue(has(got, WIDE), got)
        two = "    int k = b;\n    p->q = k;\n" + NARROW
        got = outs("scope-local", two)
        self.assertTrue(has(got, "int k = b; p->q = k; float amount; if (a) {" + WIDE.split("if (a) {", 1)[1]), got)
        self.assertTrue(has(got, "int k = b; float amount; p->q = k; " + WIDE.replace("    float amount;\n", "")), got)

    def test_c_puts_the_hoisted_declaration_with_the_block_declarations(self):
        got = outs("scope-local", "    int k = b;\n    p->q = k;\n" + NARROW, cplusplus=False)
        self.assertEqual(len(got), 1, got)
        self.assertTrue(norm(got[0]).startswith("int k = b; float amount; p->q = k;"), got)

    def test_round_trip_is_the_same_text(self):
        back = outs("scope-local", outs("scope-local", WIDE)[0] + "\n")
        self.assertIn(WIDE[:-1], back)

    def test_legality(self):
        # read after the join: stays function level
        self.assertEqual(outs("scope-local", WIDE + "    p->k = amount;\n"), [])
        # read in the condition
        self.assertEqual(outs("scope-local", WIDE.replace("if (a)", "if (amount)")), [])
        # the branch's first use is not a plain assignment
        self.assertEqual(outs("scope-local", WIDE.replace("amount = x;", "p->n = amount;", 1)), [])
        # the assignment reads the local itself
        self.assertEqual(outs("scope-local", WIDE.replace("amount = x;", "amount = amount + x;")), [])
        # no else, or a non-block else
        self.assertEqual(outs("scope-local", "    float amount;\n    if (a) {\n        amount = x;\n        p->m = amount;\n    }\n"), [])
        # a parameter's name, an initialised declaration, a const
        self.assertEqual(outs("scope-local", WIDE.replace("amount", "x")), [])
        self.assertEqual(outs("scope-local", WIDE.replace("float amount;", "float amount = 0.0f;")), [])
        self.assertEqual(outs("scope-local", WIDE.replace("float amount;", "const float amount;")), [])
        # hoist: different types, a use outside the if, the name declared a third time
        self.assertEqual(outs("scope-local", NARROW.replace("float amount = x;", "int amount = x;")), [])
        self.assertEqual(outs("scope-local", NARROW + "    p->k = amount;\n"), [])
        self.assertEqual(outs("scope-local", NARROW + "    {\n        float amount = 2.0f;\n        p->k = amount;\n    }\n"), [])
        # hoist: the local is read in the branch before its declaration (another amount)
        self.assertEqual(outs("scope-local", NARROW.replace("        float amount = x;", "        p->n = amount;\n        float amount = x;")), [])


class SharedTailTest(unittest.TestCase):
    def test_both_ways(self):
        self.assertTrue(has(outs("shared-tail", TAIL_BOTH), TAIL_ONCE), outs("shared-tail", TAIL_BOTH))
        self.assertTrue(has(outs("shared-tail", TAIL_ONCE), TAIL_BOTH), outs("shared-tail", TAIL_ONCE))

    def test_round_trip_is_the_same_text(self):
        once = [g for g in outs("shared-tail", TAIL_BOTH) if norm(g) == norm(TAIL_ONCE)]
        self.assertEqual(once, [TAIL_ONCE[:-1]])
        both = [g for g in outs("shared-tail", TAIL_ONCE) if norm(g) == norm(TAIL_BOTH)]
        self.assertEqual(both, [TAIL_BOTH[:-1]])

    def test_legality(self):
        # a branch that ends in a jump: the statement after the if runs on one path only
        self.assertEqual(outs("shared-tail", TAIL_ONCE.replace("p->m = 1;", "p->m = 1;\n        return;")), [])
        # the pushed statement names a local a branch declares
        self.assertEqual(outs("shared-tail", TAIL_ONCE.replace("p->m = 2;", "int x = 2;")), [])
        # the tails differ, or a branch holds only the tail
        self.assertEqual(outs("shared-tail", TAIL_BOTH.replace("p->k = x;\n    } else", "p->k = b;\n    } else")), [])
        self.assertEqual(outs("shared-tail", "    if (a) {\n        p->k = x;\n    } else {\n        p->m = 2;\n        p->k = x;\n    }\n"), [])
        # a declaration or a return is not moved
        self.assertEqual(outs("shared-tail", TAIL_ONCE.replace("p->k = x;", "int k = x;")), [])
        self.assertEqual(outs("shared-tail", TAIL_ONCE.replace("p->k = x;", "return;")), [])
        # no else: nothing to share
        self.assertEqual(outs("shared-tail", "    if (a) {\n        p->m = 1;\n    }\n    p->k = x;\n"), [])


if __name__ == "__main__":
    unittest.main()
