import json
import unittest
from collections import Counter

from tools.dk.core import cbody
from tools.dk.platforms.gc_mwcc import regexplain as rx
from tools.dk.platforms.gc_mwcc.regs import DwarfFunction, Var
from tools.dk.platforms.gc_mwcc.rewrites import Site

from tests.test_close import FakeCloser


def dwarf(*vars_):
    return DwarfFunction("f", "f__Fv", [Var(k, n, r) for k, n, r in vars_])


def pairs(**m):
    return {ours: Counter({t: 4}) for ours, t in m.items()}


def site(src: str) -> Site:
    span = cbody.functions(src)[-1]
    return Site(src, cbody.parse_body(src, span.open), span=span)


SRC = ("int D(int); void U(int);\n"
       "int f(int p, int q) {\n"
       "    int a = D(p);\n"
       "    int b;\n"
       "    b = D(q);\n"
       "    U(0);\n"
       "    U(a);\n"
       "    U(b);\n"
       "    return q;\n"
       "}\n")


class RowsTest(unittest.TestCase):
    def test_colour_positions_and_what_each_variable_wants(self):
        d = dwarf(("param", "p", "r28"), ("param", "q", "r29"), ("local", "a", "r31"), ("local", "b", "r30"))
        rows = {r.name: r for r in rx.rows(d, pairs(r31="r30", r30="r31", r29="r29", r28="r28", r27="r26"))}
        self.assertEqual((rows["a"].position, rows["a"].target_position, rows["a"].want), (0, 1, "later"))
        self.assertEqual((rows["b"].want, rows["b"].status), ("earlier", "number"))
        self.assertEqual(rows["q"].tier, "param #0 (last first)")
        self.assertEqual(rows["p"].status, "ok")
        self.assertEqual(rows["<temp r27>"].kind, "temp")


class MovesTest(unittest.TestCase):
    def test_declaration_order_first_for_swapped_locals(self):
        d = dwarf(("param", "p", "r28"), ("param", "q", "r29"), ("local", "a", "r31"), ("local", "b", "r30"))
        ex = rx.explain(site(SRC), d, pairs(r31="r30", r30="r31", r29="r29", r28="r28"))
        self.assertEqual(ex.moves[0].kind, "decl-order")
        best = ex.moves[0].source
        self.assertLess(best.index("int b;"), best.index("int a"))
        self.assertTrue(all(m.source != SRC for m in ex.moves))

    def test_a_local_the_target_ranks_above_every_local_becomes_a_temporary(self):
        d = dwarf(("param", "p", "r28"), ("param", "q", "r29"), ("local", "a", "r31"), ("local", "b", "r30"))
        # b above a, and above every named local in the target: joined initializer is proposed
        ex = rx.explain(site(SRC), d, pairs(r31="r30", r30="r31", r29="r29", r28="r28"))
        joined = [m for m in ex.moves if m.kind == "temp-regime" and m.var == "b"]
        self.assertTrue(joined)
        self.assertIn("int b = D(q);", joined[0].source)

    def test_a_local_coloured_after_the_parameters_may_be_a_reused_parameter(self):
        src = ("void* G(void*); void H(void*);\n"
               "void* f(void* blk, int n) {\n"
               "    void* old = G(blk);\n"
               "    H(old);\n"
               "    H(old);\n"
               "    return old;\n"
               "}\n")
        d = dwarf(("param", "blk", "r30"), ("param", "n", "r31"), ("local", "old", "r29"))
        # target: n r31, old r29 below blk r30 is impossible for a local; written as blk, it takes blk's place
        ex = rx.explain(site(src), d, pairs(r31="r31", r30="r29", r29="r28"))
        reuse = [m for m in ex.moves if m.kind == "param-reuse"]
        self.assertEqual(len(reuse), 1)
        self.assertIn("blk = G(blk);", reuse[0].source)
        self.assertIn("return blk;", reuse[0].source)
        self.assertNotIn("old", reuse[0].source.split("{", 1)[1])

    def test_no_reuse_when_the_parameter_is_read_later_or_types_differ(self):
        src = ("void* G(void*); void H(void*);\n"
               "void* f(void* blk, int n) {\n"
               "    void* old = G(blk);\n"
               "    H(blk);\n"
               "    int k = n;\n"
               "    return old;\n"
               "}\n")
        s = site(src)
        self.assertEqual(list(rx._param_reuse(s, "old")), [])
        self.assertEqual(list(rx._param_reuse(s, "k")), [])

    def test_fold_a_single_use_local_keeps_call_order(self):
        src = "int D(); int E(); void U(int);\nvoid f() {\n    int x = D();\n    U(x + 1);\n}\n"
        self.assertIn("U((D()) + 1)".replace("(D())", "D()"),
                      cbody.apply_edits(src, rx._fold_single_use(site(src), "x")))
        src2 = "int D(); int E(); void U(int, int);\nvoid f() {\n    int x = D();\n    U(E(), x);\n}\n"
        self.assertIsNone(rx._fold_single_use(site(src2), "x"))

    def test_a_temporary_ranked_after_a_named_local_gets_its_operands_named(self):
        src = "int D(int); void U(int);\nint g;\nvoid f(int p) {\n    int a = D(p);\n    U(D(p) + g);\n    U(a);\n}\n"
        d = dwarf(("param", "p", "r29"), ("local", "a", "r30"))
        ex = rx.explain(site(src), d, pairs(r31="r30", r30="r31", r29="r29"), defs={"r31": 5})
        named = [m for m in ex.moves if m.kind == "name-temp"]
        self.assertTrue(named)
        self.assertIn("U(t0);", named[0].source)                       # outermost operand first
        self.assertTrue(any("__typeof__(D(p)) t0 = D(p);" in m.source and "U(t0 + g);" in m.source for m in named))

    def test_def_lines_take_the_first_write(self):
        ins = lambda m, regs, line: {"instruction": {"line_number": line, "parts": [{"opcode": {"mnemonic": m}}] +
                                                     [{"arg": {"opaque": r}} for r in regs]}}
        got = rx.def_lines([ins("stw", ["r31", "r1"], 3), ins("mr", ["r31", "r3"], 4), ins("addi", ["r31", "r31"], 9)])
        self.assertEqual(got, {"r31": 4})

    def test_render_lists_rows_and_moves(self):
        d = dwarf(("param", "p", "r28"), ("param", "q", "r29"), ("local", "a", "r31"), ("local", "b", "r30"))
        lines = rx.render(rx.explain(site(SRC), d, pairs(r31="r30", r30="r31", r29="r29", r28="r28")))
        self.assertTrue(any(l.startswith("!=a") for l in lines))
        self.assertTrue(any("[decl-order]" in l for l in lines))


class TruthMovesInCloseTest(unittest.TestCase):
    def test_truth_moves_are_compiled_first_and_the_memo_resumes(self):
        src = "int f(int a, int b) {\n    return a + b;\n}\n"
        c = FakeCloser(src, lambda t: 100.0 if "b + a" in t else 90.0, moves="truth")
        r = c.run()
        self.assertTrue(r.exact)
        self.assertTrue(c.memo_path.is_file())
        memo = json.loads(c.memo_path.read_text())
        self.assertEqual(memo["best"], 100.0)
        self.assertGreaterEqual(len(memo["explored"]), 1)

    def test_memo_skips_explored_candidates(self):
        src = "int f(int a, int b) {\n    return a + b;\n}\n"
        c = FakeCloser(src, lambda t: 90.0, budget=50)
        c.run()
        first = len(c.seen_texts)
        c2 = FakeCloser.__new__(FakeCloser)
        FakeCloser.__init__(c2, src, lambda t: 90.0, budget=50)
        c2.memo_path = c.memo_path
        c2.run()
        self.assertLess(len(c2.seen_texts), first)
        fresh = FakeCloser(src, lambda t: 90.0, budget=50, memo=False)
        fresh.memo_path = c.memo_path
        fresh.run()
        self.assertEqual(len(fresh.seen_texts), first)


class DiffsTest(unittest.TestCase):
    def test_exact_results_are_written_as_diffs(self):
        import argparse
        import tempfile
        from pathlib import Path
        from unittest import mock
        from tools.dk.platforms.gc_mwcc import closecmd
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "u.cpp").write_text("int f(int a, int b) {\n    return a + b;\n}\n")
            cfg = type("Cfg", (), {"root": root, "rel": lambda self, p: str(Path(p).relative_to(root))})()
            unit = {"name": "u", "metadata": {"source_path": "src/u.cpp"}}
            res = closecmd.Result("u", "f__Fii", 90.0, 100.0, exact=True,
                                  text="int f(int a, int b) {\n    return b + a;\n}\n", labels=["swap"])
            args = argparse.Namespace(from_=None, min=90.0, sample=0, jobs=1, apply=False, json=None, baseline=False,
                                      diffs=str(root / "d"))
            with mock.patch.object(closecmd, "near_misses", return_value=[("u", "f__Fii", 90.0)]), \
                    mock.patch.object(closecmd, "resolve_unit", return_value=unit), \
                    mock.patch.object(closecmd, "close_one", return_value=res):
                closecmd.run_near(cfg, args, closecmd.Options())
            diff = (root / "d" / "f__Fii.diff").read_text()
            self.assertIn("+    return b + a;", diff)
            self.assertIn("return a + b;", (root / "src" / "u.cpp").read_text())


if __name__ == "__main__":
    unittest.main()
