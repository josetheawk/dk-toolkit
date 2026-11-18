import unittest

from tools.dk.core import cbody
from tools.dk.platforms.gc_mwcc import degmoves
from tools.dk.platforms.gc_mwcc.rewrites import Site

HEAD = "void f(int a, int b, S* p) "


def unit(body: str, before: str = "") -> tuple[str, Site]:
    src = before + HEAD + "{\n" + body + "}\n"
    at = src.index(HEAD) + len(HEAD)
    return src, Site(src, cbody.parse_body(src, at), None, True)


def analysis(levers=(), spans=(), drops=()):
    return {"classes": {"gpr": {"degree": {"levers": list(levers), "spans": list(spans), "drops": list(drops)}}}}


class SplitOpAssignTest(unittest.TestCase):
    def test_binary_assignment_splits_into_assign_and_op_assign(self):
        src, s = unit("    int v;\n    v = g(a) + p->m;\n    h(v);\n")
        got = list(degmoves.split_op_assign(s, None))
        self.assertEqual(len(got), 1)
        new = cbody.apply_edits(src, got[0][1])
        self.assertIn("v = g(a);\n    v += p->m;", new)

    def test_right_operand_with_a_call_or_reading_v_is_left_alone(self):
        _, s = unit("    int v;\n    v = a + g(b);\n    v = a + v;\n    v = v + a;\n")
        self.assertEqual(list(degmoves.split_op_assign(s, None)), [])


class TargetsTest(unittest.TestCase):
    def test_single_levers_then_spans_then_drops_then_pairs(self):
        an = analysis(levers=[{"values": [{"index": 1, "kind": "local", "name": "x", "line": 3},
                                          {"index": 2, "kind": "local", "name": "y", "line": 4}], "delta": 1},
                              {"values": [{"index": 3, "kind": "temp", "line": 5}], "delta": -1}],
                      spans=[{"lines": [6, 7]}], drops=[{"index": 4, "kind": "local", "name": "z", "line": 8}])
        tg = degmoves.targets(an)
        self.assertEqual([t.sign for t in tg], [-1, 1, -1, 1])
        self.assertEqual(tg[0].lines, [4])
        self.assertEqual(tg[1].lines, [5, 6])
        self.assertEqual(tg[2].names, ["z"])
        self.assertEqual(tg[3].names, ["x", "y"])

    def test_a_named_value_targets_the_lines_where_it_is_live(self):
        src, s = unit("    int x = a;\n    b = 1;\n    b = x;\n    b = 2;\n")
        line = cbody.line_of(src, src.index("int x"))
        tg = degmoves.targets(analysis(levers=[{"values": [{"kind": "local", "name": "x", "line": line + 1}],
                                                "delta": 1}]), s)
        self.assertEqual(sorted(tg[0].lines), [line, line + 1, line + 2])


class CandidatesTest(unittest.TestCase):
    def test_plus_lever_names_a_call_result_first(self):
        src, s = unit("    b = g(a) + p->m;\n")
        line = cbody.line_of(src, src.index("b = g"))
        got = degmoves.candidates(s, analysis(levers=[{"values": [{"kind": "temp", "line": line + 1}], "delta": 1}]))
        self.assertTrue(got)
        self.assertEqual(got[0][0], "degree:name-temp")
        self.assertIn("t0", got[0][2])

    def test_minus_lever_folds_a_single_use_local(self):
        src, s = unit("    int x = g(a);\n    h(x);\n")
        line = cbody.line_of(src, src.index("int x"))
        got = degmoves.candidates(s, analysis(levers=[{"values": [{"kind": "local", "name": "x", "line": line + 1}],
                                                       "delta": -1}]))
        self.assertEqual(got[0][0], "degree:fold")
        self.assertIn("h(g(a));", got[0][2])
        self.assertTrue(all("[gpr" in label for _, label, _ in got))

    def test_no_degree_block_gives_nothing(self):
        _, s = unit("    b = g(a);\n")
        self.assertEqual(degmoves.candidates(s, {"classes": {"gpr": {}}}), [])
        self.assertEqual(degmoves.candidates(s, None), [])


class CtorHintTest(unittest.TestCase):
    INIT = "T* Init(T* self, int n) { self->n = n; return self; }\n"

    def test_alloc_then_init_returning_its_first_argument_is_reported(self):
        _, s = unit("    T* q = (T*)Alloc(8);\n    if (q) {\n        q = Init(q, a);\n    }\n", before=self.INIT)
        hints = degmoves.ctor_hints(s)
        self.assertEqual(len(hints), 1)
        self.assertIn("Init", hints[0])

    def test_init_that_returns_something_else_is_not(self):
        init = "T* Init(T* self, int n) { self->n = n; return 0; }\n"
        _, s = unit("    T* q = (T*)Alloc(8);\n    if (q) q = Init(q, a);\n", before=init)
        self.assertEqual(degmoves.ctor_hints(s), [])


if __name__ == "__main__":
    unittest.main()
