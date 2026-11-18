"""dk dataset fixes (llmfixes.py): fix kinds, the leakage filter, the grind-log pointer. Synthetic fixtures."""
import unittest

from tools.dk.platforms.gc_mwcc import llmfixes as F

BEFORE = "void f(Car* c) {\n    int a = c->m4;\n    float b = c->m8;\n    c->Use(a, b);\n}"


class FixKindTest(unittest.TestCase):
    def test_declaration_order(self):
        after = BEFORE.replace("    int a = c->m4;\n    float b = c->m8;", "    float b = c->m8;\n    int a = c->m4;")
        self.assertEqual(F.fix_kinds(BEFORE, after), ["declaration-order"])

    def test_statement_order(self):
        before = "void f() {\n    g();\n    h();\n}"
        self.assertEqual(F.fix_kinds(before, "void f() {\n    h();\n    g();\n}"), ["statement-order"])

    def test_local_added_and_literal(self):
        after = BEFORE.replace("c->Use(a, b);", "int k = 3;\n    c->Use(a, b);")
        self.assertEqual(F.fix_kinds(BEFORE, after), ["local-added", "literal"])

    def test_control_flow_and_comments_ignored(self):
        after = BEFORE.replace("c->Use(a, b);", "if (a) c->Use(a, b); // x")
        self.assertIn("control-flow", F.fix_kinds(BEFORE, after))


class LeakTest(unittest.TestCase):
    FZ = {"units": {"u/held": "heldout"}, "functions": {"fn_1": "bench"}, "shapes": {"S9": "repair-eval"},
          "hashes": {F.norm_hash(BEFORE): "review-eval"}}

    def rec(self, **kw):
        r = {"unit": "u/ok", "function": "fn_2", "asm_shape": "S1", "before": "int a;", "after": "int b;"}
        r.update(kw)
        return r

    def test_clean(self):
        self.assertIsNone(F.leak(self.rec(), self.FZ))

    def test_each_key(self):
        self.assertEqual(F.leak(self.rec(unit="u/held"), self.FZ), "unit (heldout)")
        self.assertEqual(F.leak(self.rec(function="fn_1"), self.FZ), "function (bench)")
        self.assertEqual(F.leak(self.rec(asm_shape="S9"), self.FZ), "asm shape (repair-eval)")
        self.assertEqual(F.leak(self.rec(before=BEFORE + " // c"), self.FZ), "content hash (review-eval)")


class LogTest(unittest.TestCase):
    def test_landing_named_by_hash_prefix(self):
        logs = [("a.md", "fn_804D2788__Q22Ns4UnitFv | 10% | LANDED 7a69c250e | 3 | note\nfn_2 | 1% | gave up | 2 | n")]
        self.assertEqual(F.logged_landing(logs, "fn_804D2788__Q22Ns4UnitFv", "7a69c250e0000"), "a.md")
        self.assertIsNone(F.logged_landing(logs, "fn_2", "7a69c250e0000"))


if __name__ == "__main__":
    unittest.main()
