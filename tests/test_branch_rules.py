"""The rewrite table's branch-landing rules: nest-join, hoist-tail, break-guard (tool-gaps #6)."""

import unittest

from tools.dk.core import cbody
from tools.dk.platforms.gc_mwcc import rewrites


def run(rule, src):
    span = next(f for f in cbody.functions(src) if f.name == "f")
    site = rewrites.Site(src, cbody.parse_body(src, span.open, cbody.type_names(src)))
    return [new for _, _, new in rewrites.candidates(site, [r for r in rewrites.RULES if r.name == rule], groups=False)]


class BranchRulesTest(unittest.TestCase):
    def test_nest_join_keeps_the_else(self):
        src = "void f(int a, int b) {\n    if (a > 1) {\n        if (b < 2) {\n            g(1);\n        }\n    } else if (a < 0) {\n        g(2);\n    }\n}\n"
        out = run("nest-join", src)
        self.assertTrue(any("if (a > 1 && b < 2) {" in o and "else if (a < 0)" in o for o in out), out)

    def test_nest_join_over_a_block(self):
        src = "int f(int a, int b) {\n    if (a) {\n        if (b) {\n            g(1);\n        }\n        return 1;\n    }\n    return 0;\n}\n"
        out = run("nest-join", src)
        self.assertTrue(any("if (a && b) {\n        g(1);\n        return 1;\n    }\n    return 0;" in o for o in out), out)

    def test_int_flag_both_ways(self):
        src = "int f(int a) {\n    int r = a == 1;\n    return g(a) == 0;\n}\n"
        out = run("int-flag", src)
        self.assertTrue(any("return (g(a) == 0) ? 1 : 0;" in o for o in out), out)
        self.assertTrue(any("int r = a == 1;" in o for o in out))
        back = run("int-flag", "int f(int a) {\n    return (a == 0) ? 1 : 0;\n}\n")
        self.assertIn("    return a == 0;", back[0])

    def test_fold_single(self):
        src = "void f(float a) {\n    float c = clamp(a, 1.0f);\n    g(&m, c);\n    h(c2);\n}\n"
        out = run("fold-single", src)
        self.assertTrue(any("    g(&m, clamp(a, 1.0f));\n    h(c2);" in o for o in out), out)
        self.assertEqual(run("fold-single", "void f(float a) {\n    float c = a;\n    g(c);\n    g(c);\n}\n"), [])

    def test_hoist_tail(self):
        src = "void f(int a) {\n    if (a) {\n        g(1);\n        h(2);\n    }\n    return;\n}\n"
        out = run("hoist-tail", src)
        self.assertIn("    if (a) {\n        g(1);\n    }\n    h(2);\n    return;", out[0])

    def test_break_guard(self):
        src = "void f(P* p) {\n    while (p != 0) {\n        p = p->n;\n        if (p != 0) {\n            g(p);\n        }\n    }\n}\n"
        out = run("break-guard", src)
        self.assertTrue(any("if (p == 0) {\n            break;\n        }\n        g(p);" in o for o in out), out)


if __name__ == "__main__":
    unittest.main()
