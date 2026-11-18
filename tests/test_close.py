import tempfile
import unittest
from collections import Counter
from pathlib import Path

from tools.dk.platforms.gc_mwcc import closecmd, rewrites
from tools.dk.platforms.gc_mwcc.closecmd import Closer, Eval, Options, sample, source_diff
from tools.dk.platforms.gc_mwcc.linkproof import Proof
from tools.dk.platforms.gc_mwcc.matching import FunctionScore, UnitScore


class FakeCfg:
    def __init__(self, root: Path):
        self.root = root
        self.quality = {"forbidden": []}


class FakeCloser(Closer):
    """The search with a scoring function in place of the compiler: `want(text)` -> percent."""

    def __init__(self, src: str, want, types=None, suffix=".cpp", **opts):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "src").mkdir()
        (root / "src" / f"u{suffix}").write_text(src)
        unit = {"name": "u", "metadata": {"source_path": f"src/u{suffix}"}, "target_path": "t.o"}
        super().__init__(FakeCfg(root), unit, "f__Fii", Options(jobs=2, **opts), log=lambda s: None)
        self.want, self.types, self.seen_texts, self.probe = want, types or {}, [], None

    def evaluate(self, text, slot):
        self.seen_texts.append(text)
        p = self.want(text)
        if p is None:
            return None
        sym = rewrites.Symptoms(Counter({"swap": 1}), Counter({2: 1}))
        return Eval(text, p, 0 if p == 100.0 else 3, sym, 1, dict(self.types) if "__typeof__" in text else {})

    def probe_types(self, text, body, names, first_line):
        return {}

    def cast_probe(self):
        return self.probe

    def unit_score(self, text, slot):
        p = self.want(text) if "__typeof__" not in text else 0.0
        return UnitScore("u", [FunctionScore("f__Fii", 8, p)]), Proof(p == 100.0, "fake"), ""


SRC = "int f(int a, int b) {\n    return a * 2 + b;\n}\n"


class SearchTest(unittest.TestCase):
    def test_a_swap_closes_the_function(self):
        c = FakeCloser("int f(int a, int b) {\n    return a + b;\n}\n", lambda t: 100.0 if "b + a" in t else 90.0)
        r = c.run()
        self.assertTrue(r.exact)
        self.assertEqual(r.labels, ["a + b -> b + a"])
        self.assertIn("return b + a;", r.text)

    def test_named_operand_gets_its_real_type_before_the_final_check(self):
        want = lambda t: 100.0 if ("t0 + b" in t) else 90.0
        r = FakeCloser(SRC, want, types={"t0": "int"}).run()
        self.assertTrue(r.exact)
        self.assertIn("    int t0 = a * 2;\n    return t0 + b;", r.text)
        self.assertNotIn("__typeof__", r.text)

    def test_no_type_in_the_debug_info_means_rejected(self):
        r = FakeCloser(SRC, lambda t: 100.0 if "t0 + b" in t else 90.0).run()
        self.assertFalse(r.exact)
        self.assertEqual(r.reasons, ["no type in the debug info for t0"])

    def test_stacked_rewrites_need_more_than_one_level(self):
        src = "int f(int a, int b) {\n    a = a + b;\n    return a * 2;\n}\n"
        want = lambda t: 100.0 if ("b + a" in t and "2 * a" in t) else 95.0 if ("b + a" in t or "2 * a" in t) else 90.0
        r = FakeCloser(src, want).run()
        self.assertTrue(r.exact)
        self.assertEqual(len(r.labels), 2)

    def test_the_search_is_deterministic_and_respects_the_budget(self):
        src = "int f(int a, int b) {\n    a = a + b;\n    b = a * b;\n    return a - b;\n}\n"
        runs = []
        for _ in range(2):
            c = FakeCloser(src, lambda t: 90.0 + (len(t) % 7) / 10, budget=12)
            r = c.run()
            runs.append((r.best, r.labels, r.compiles))
        self.assertEqual(runs[0], runs[1])
        self.assertLessEqual(runs[0][2], 12)

    def test_an_exact_start_is_reported(self):
        self.assertEqual(FakeCloser(SRC, lambda t: 100.0).run().error, "already exact")


class SameTypeCastGateTest(unittest.TestCase):
    """The close path's final judge applies the same-type-cast rule llm-review does (it once passed no
    compiler probe, so a cast to the expression's own type that steers the code was accepted)."""
    HEAD = ("namespace Ns { struct NpcBase { int x; }; }\n"
            "struct Chase { Ns::NpcBase* NthFollower(unsigned int i); };\nextern Chase* g;\n")
    BEFORE = HEAD + "int f(int a, int b) {\n    Ns::NpcBase* first = g->NthFollower(a);\n    return first->x + b;\n}\n"
    AFTER = BEFORE.replace("= g->NthFollower(a)", "= (Ns::NpcBase*)g->NthFollower(a)")

    def reasons(self, steers: bool):
        def probe(text):                          # a stand-in compiler: the wrapped cast compiles (same type)
            if "__typeof__" in text:
                return 100.0
            return 100.0 if not steers or "(Ns::NpcBase*)g" in text else 98.5
        c = FakeCloser(self.BEFORE, lambda t: 100.0)
        c.probe = probe
        e = Eval(self.AFTER, 100.0, 0, rewrites.Symptoms(Counter(), Counter()), 1, {})
        before = UnitScore("u", [FunctionScore("f__Fii", 8, 90.0)])
        return c.final_check(e, self.BEFORE, before)[0].reasons

    def test_a_cast_to_its_own_type_that_steers_the_code_is_rejected(self):
        self.assertEqual(self.reasons(True), ["cast to its own type changes the code (Ns::NpcBase*)g->NthFollower(a)"])

    def test_one_without_effect_is_left_to_the_review(self):
        self.assertEqual(self.reasons(False), [])


class HelperTest(unittest.TestCase):
    def test_sample_is_a_stable_spread(self):
        items = [(f"u{i}", f"f{i}", 95.0) for i in range(50)]
        self.assertEqual(sample(items, 5), sample(list(reversed(items)), 5))
        self.assertEqual(len(sample(items, 5)), 5)

    def test_source_diff(self):
        got = source_diff("a\nb\nc\n", "a\nB\nc\n", "src/u.cpp")
        self.assertEqual(got[:2], ["--- src/u.cpp", "+++ src/u.cpp (closed)"])
        self.assertIn("-b", got)
        self.assertIn("+B", got)

    def test_baseline_uses_only_the_old_moves(self):
        self.assertEqual(set(closecmd.BASELINE_RULES) - set(rewrites.BY_NAME), set())


class HeldSetTests(unittest.TestCase):
    def test_released_sets_do_not_hold_but_others_do(self):
        frozen = {("u", "f"): "gate-lora2", ("u", "g"): "heldout", ("u", "h"): "near-sample-60"}
        thaw = ("gate-lora2", "near-sample-60")                             # an explicit --thaw
        self.assertIsNone(closecmd.held_set(frozen, "u", "f", thaw=thaw))
        self.assertIsNone(closecmd.held_set(frozen, "u", "h", thaw=thaw))
        self.assertEqual(closecmd.held_set(frozen, "u", "f"), "gate-lora2")  # default 2026-11-18: nothing thawed
        self.assertEqual(closecmd.held_set(frozen, "u", "g"), "heldout")    # never landed
        self.assertIsNone(closecmd.held_set(frozen, "u", "x"))
        self.assertEqual(closecmd.held_set(frozen, "u", "f", thaw=()), "gate-lora2")


class ReadPairsTests(unittest.TestCase):
    def test_from_accepts_pairs_results_and_frozen_sets(self):
        self.assertEqual(closecmd.read_pairs([["u", "f"]]), [("u", "f")])
        self.assertEqual(closecmd.read_pairs([{"unit": "u", "fn": "f", "exact": False}]), [("u", "f")])
        self.assertEqual(closecmd.read_pairs({"frozen": 1, "functions": [{"unit": "u", "function": "f"}]}),
                         [("u", "f")])


if __name__ == "__main__":
    unittest.main()
