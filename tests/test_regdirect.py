import unittest

from tools.dk.core import cbody
from tools.dk.platforms.gc_mwcc import regdirect
from tools.dk.platforms.gc_mwcc.rewrites import Site

HELPER = "static inline int Round(float x) { return (int)(x + (x >= 0.0f ? 0.5f : -0.5f)); }\n"
ONCE = "static inline float Half(float x) { return x * 0.5f; }\n"


def unit(helper: str, body: str, other: str = "") -> tuple[str, Site]:
    src = helper + other + "void f(int a, int b, float x, S* p) {\n" + body + "}\n"
    at = src.index("void f(") + len("void f(int a, int b, float x, S* p) ")
    return src, Site(src, cbody.parse_body(src, at), None, True)


def norm(t: str) -> str:
    return " ".join(t.split())


class HelperToMacroTest(unittest.TestCase):
    def test_parameter_read_twice_becomes_a_macro_with_parenthesised_parameters(self):
        src, s = unit(HELPER, "    a = Round(x / (2.0f * b));\n    a -= p->m / 2;\n")
        got = list(regdirect.helper_to_macro(s, None))
        self.assertEqual(len(got), 1)
        new = cbody.apply_edits(src, got[0][1])
        self.assertIn("#define Round(x) ((int)((x) + ((x) >= 0.0f ? 0.5f : -0.5f)))", new)
        self.assertNotIn("static inline", new)
        self.assertIn("a = Round(x / (2.0f * b));", new)          # call sites untouched

    def test_cast_added_when_the_body_has_none(self):
        src, s = unit("static inline int Twice(float x) { return x + x; }\n", "    a = Twice(x);\n")
        new = cbody.apply_edits(src, list(regdirect.helper_to_macro(s, None))[0][1])
        self.assertIn("#define Twice(x) ((int)((x) + (x)))", new)

    def test_parameter_read_once_is_not_a_macro(self):
        _, s = unit(ONCE, "    x = Half(x);\n")
        self.assertEqual(list(regdirect.helper_to_macro(s, None)), [])

    def test_a_call_with_a_side_effect_argument_anywhere_in_the_unit_forbids_it(self):
        _, s = unit(HELPER, "    a = Round(x);\n", other="void g(float* q) { int k = Round(*q++); }\n")
        self.assertEqual(list(regdirect.helper_to_macro(s, None)), [])
        _, s = unit(HELPER, "    a = Round(h(x));\n")
        self.assertEqual(list(regdirect.helper_to_macro(s, None)), [])

    def test_only_at_the_given_lines(self):
        src, s = unit(HELPER, "    a = Round(x);\n    b = 1;\n")
        call_line = cbody.line_of(src, src.index("a = Round"))
        self.assertEqual(len(list(regdirect.helper_to_macro(s, {call_line}))), 1)
        self.assertEqual(list(regdirect.helper_to_macro(s, {call_line + 1})), [])

    def test_a_helper_not_called_in_the_body_is_left_alone(self):
        _, s = unit(HELPER, "    a = b;\n", other="void g() { int k = Round(1.0f); }\n")
        self.assertEqual(list(regdirect.helper_to_macro(s, None)), [])


class JoinOpAssignTest(unittest.TestCase):
    def test_adjacent_assign_and_op_assign_join_without_needless_parentheses(self):
        src, s = unit("", "    a = Round(x);\n    a -= p->m / 2;\n")
        got = list(regdirect.join_op_assign(s, None))
        self.assertEqual(len(got), 1)
        self.assertIn("a = Round(x) - p->m / 2;", cbody.apply_edits(src, got[0][1]))

    def test_parentheses_keep_precedence(self):
        src, s = unit("", "    a = b + 1;\n    a *= b - 2;\n")
        new = cbody.apply_edits(src, list(regdirect.join_op_assign(s, None))[0][1])
        self.assertIn("a = (b + 1) * (b - 2);", new)
        src, s = unit("", "    a = b;\n    a -= b + 2;\n")
        new = cbody.apply_edits(src, list(regdirect.join_op_assign(s, None))[0][1])
        self.assertIn("a = b - (b + 2);", new)

    def test_refused_when_the_second_reads_the_variable_or_has_a_side_effect(self):
        _, s = unit("", "    a = b;\n    a += a;\n")
        self.assertEqual(list(regdirect.join_op_assign(s, None)), [])
        _, s = unit("", "    a = b;\n    a += g(b);\n")
        self.assertEqual(list(regdirect.join_op_assign(s, None)), [])
        _, s = unit("", "    a = b;\n    b += 1;\n")
        self.assertEqual(list(regdirect.join_op_assign(s, None)), [])


def value(index, kind, line, name="", phase=None):
    v = {"index": index, "kind": kind, "line": line, "name": name, "cls": "gpr"}
    if phase:
        v["origin"] = {"phase": phase, "line": line}
    return v


def analysis(rules, reachable=True):
    return {"schema": "reach.v1", "reachable": reachable,
            "classes": {"gpr": {"reachable": reachable, "rules": [
                {"before": {"index": a}, "after": {"index": b}, "flip": True, "cell": ""} for a, b in rules]}}}


class DiagnoseTest(unittest.TestCase):
    def setUp(self):
        self.record = {"vregs": [value(1, "local", 3, "x"), value(2, "local", 4, "y"), value(3, "param", 1, "p"),
                                 value(10, "fe-temp", 7, phase="cse0"), value(11, "fe-temp", 9, phase="linear"),
                                 value(12, "fe-temp", 9, phase="linear"), value(20, "temp", 12)]}

    def cells(self, rules, reachable=True):
        d = regdirect.diagnose(analysis(rules, reachable), self.record, [])
        return d.kind, [r.cell for r in d.rules], d.stop

    def test_cells(self):
        self.assertEqual(self.cells([(1, 2)]), ("order", ["decl-order"], []))
        self.assertEqual(self.cells([(1, 3)])[1], ["param"])
        self.assertEqual(self.cells([(1, 10)])[1], ["cross-tier"])
        self.assertEqual(self.cells([(11, 12)])[1], ["same-phase"])
        self.assertEqual(self.cells([(20, 1)])[1], ["cross-tier"])
        self.assertEqual(self.cells([(20, 12)])[1], ["code-eval"])

    def test_cross_phase_is_a_stop_not_a_move(self):
        kind, cells, stop = self.cells([(10, 11)])          # cse0 after linear: no order inside a phase flips it
        self.assertEqual((kind, cells), ("order", ["cross-phase"]))
        self.assertEqual(len(stop), 1)
        self.assertIn("cross-phase", stop[0])
        d = regdirect.diagnose(analysis([(10, 11)]), self.record, [])
        self.assertEqual(d.open_lines, set())

    def test_stuck_and_sched(self):
        self.assertEqual(self.cells([], reachable=False)[0], "stuck")
        self.assertEqual(self.cells([])[0], "sched")
        self.assertEqual(regdirect.diagnose(None, None, None).kind, "none")
        self.assertEqual(regdirect.diagnose({"error": "x"}, self.record, None).kind, "none")


class CandidatesTest(unittest.TestCase):
    def test_sched_uses_the_differing_lines_only(self):
        src, s = unit(HELPER, "    a = Round(x);\n    a -= p->m / 2;\n    b = 0;\n    for (a = 0; a < 8; a++) {\n"
                              "        p->m = a;\n        b++;\n    }\n")
        diag = regdirect.Diagnosis("sched")
        line = cbody.line_of(src, src.index("a -= p->m"))
        got = regdirect.candidates(s, diag, None, [], frozenset(), {line})
        kinds = {k for k, _, _ in got}
        self.assertIn("directed:helper-to-macro", kinds)
        self.assertIn("directed:join-op-assign", kinds)
        far = regdirect.candidates(s, diag, None, [], frozenset(), {line + 4})
        self.assertNotIn("directed:join-op-assign", {k for k, _, _ in far})
        self.assertTrue(all(len({n for _, _, n in got}) == len(got) for _ in [0]))     # no duplicates

    def test_cross_phase_only_gives_nothing(self):
        src, s = unit("", "    a = b + 1;\n    a *= b - 2;\n")
        rec = {"vregs": [value(10, "fe-temp", 2, phase="cse0"), value(11, "fe-temp", 3, phase="linear")]}
        diag = regdirect.diagnose(analysis([(10, 11)]), rec, [])
        self.assertEqual(regdirect.candidates(s, diag, None, [], frozenset(), {1}), [])
        self.assertEqual(len(regdirect.render(diag)), 2)


if __name__ == "__main__":
    unittest.main()


class TraceCardTest(unittest.TestCase):
    def test_card_and_checksums_from_trace_records(self):
        import json, tempfile, hashlib
        from pathlib import Path
        from tools.dk.platforms.gc_mwcc.closecmd import trace_card
        with tempfile.TemporaryDirectory() as d:
            rec = {"schema": "close-trace.v1", "unit": "u", "function": "f", "moves": "directed", "compiles": 19,
                   "start": {"percent": 99.9}, "diagnosis": {"kind": "sched", "cells": [], "stop": []},
                   "steps": [{"rule": "directed:iv-alias", "percent": 99.97}, {"rule": "directed:for-init-list", "percent": 100.0}],
                   "final_sha": "x", "final_definition": "void f() {}"}
            (Path(d) / "traces.jsonl").write_text(json.dumps(rec) + "\n")
            self.assertEqual(trace_card(Path(d)), 0)
            card = (Path(d) / "CARD.md").read_text()
            self.assertIn("| u | f | 19 | 2 |", card)
            self.assertIn("directed:iv-alias 1", card)
            sums = (Path(d) / "SHA256SUMS").read_text()
            self.assertIn(hashlib.sha256((Path(d) / "traces.jsonl").read_bytes()).hexdigest() + "  traces.jsonl", sums)
            self.assertEqual(trace_card(Path(d) / "nope"), 2)
